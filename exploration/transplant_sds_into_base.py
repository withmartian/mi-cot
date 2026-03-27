"""
CEBRA-EM transfer evaluation on sentence-boundary hidden states.

----------------
1. **Source (e.g. reasoning model)** and **target (e.g. base model)** each provide one row per
   sentence: a hidden vector plus metadata (problem_id, sentence_idx, reasoning stage, ...).

2. **Fit representation + dynamics on source only**: contrastive CEBRA maps sentences to z,
   then a switching linear dynamical system (SLDS) learns K latent "regimes" and transitions
   between them along each problem’s sentence sequence.

3. **Do not retrain** CEBRA or SLDS on target. We apply the **same** scaler, PCA, and CEBRA
   encoder to target rows, then ask: if we run **target** sequences through the **source** SLDS,
   do we get useful structure? Target regimes come from the SLDS posterior (argmax per step),
   evaluated in **PCA space** (same space used elsewhere in the CEBRA-EM pipeline).

4. **Metrics**: Does a regime-conditioned linear map predict the next PCA state better than a
   single linear AR model (delta R²)? How well does the SLDS explain target trajectories (NLL)?
   Optionally: can discrete SLDS state predict the **next sentence’s stage label** better than
   guessing the global majority ("timing" lift), and can an LLM adjudicate disagreements.

No activation steering or intervention is used here.

Reported numbers
----------------
  - linear_ar_r2: pooled linear autoregression on PCA trajectories (baseline).
  - regime_r2_on_pca: predictions using SLDS-inferred regimes on target (from cebra_EM).
  - delta_r2 = regime_r2 - linear_ar_r2; bootstrap resamples trajectories for a CI.
  - slds_target_nll_per_transition: sequence log-likelihood from the forward-backward model.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
import urllib.error
import urllib.request
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import cebra_EM as cebra_mod
from sds_train_gsm8k_hf import SDS_TRAIN_GSM8K_REPO_ID, default_hub_features_relpath


# =============================================================================
# Reproducibility
# =============================================================================
# Seeds Python, NumPy, and PyTorch so triplet sampling, CEBRA init, and EM behave consistently.


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Feature I/O (pickle lists, Hugging Face download)
# =============================================================================
# Rows are dicts with at least hidden_state_last, problem_id, sentence_idx; optional stage.


def _load_pickle_list(path: str) -> List[dict]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"Expected list pickle at {path!r}, got {type(obj)}")
    return obj


def _save_pickle_list(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(list(rows), f)


def _download_features(repo_id: str, relpath: str, token: str | None) -> str:
    """Return a local path after caching the dataset file from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=relpath,
        token=token,
        repo_type="dataset",
    )


# =============================================================================
# Filtering and trajectory grouping
# =============================================================================
# We keep only non-NEUTRAL sentences (supervised stage signal) and optionally cap by problem_id.
# Trajectories are sentence chains per problem_id, ordered by sentence_idx, for sequence models.


def _subset_non_neutral_by_problem(features: List[dict], limit_problems: Optional[int]) -> List[dict]:
    out = []
    for row in features:
        if row.get("stage", "NEUTRAL") == "NEUTRAL":
            continue
        if limit_problems is not None and int(row["problem_id"]) >= int(limit_problems):
            continue
        out.append(row)
    return out


@dataclass
class PreparedSet:
    """One entry per qualifying problem: parallel lists of row indices into `features`."""

    features: List[dict]
    idx_seqs: List[List[int]]
    pids_sorted: List[int]


def _group_sequences(features: List[dict], min_len: int = 3) -> PreparedSet:
    p_map: Dict[int, List[Tuple[int, int]]] = {}
    for i, row in enumerate(features):
        pid = int(row["problem_id"])
        sent_idx = int(row["sentence_idx"])
        p_map.setdefault(pid, []).append((sent_idx, i))

    pids_sorted = []
    idx_seqs: List[List[int]] = []
    for pid in sorted(p_map):
        ordered = sorted(p_map[pid], key=lambda x: x[0])
        idxs = [i for _, i in ordered]
        if len(idxs) >= min_len:
            pids_sorted.append(pid)
            idx_seqs.append(idxs)
    return PreparedSet(features=features, idx_seqs=idx_seqs, pids_sorted=pids_sorted)


def _flatten_by_sequences(values_by_idx: np.ndarray, idx_seqs: List[List[int]]) -> np.ndarray:
    """Concatenate per-problem slices of a row-major array, in trajectory order (utility)."""
    return np.concatenate([values_by_idx[idxs] for idxs in idx_seqs], axis=0)


def _build_sequences_from_rows(rows_by_idx: np.ndarray, idx_seqs: List[List[int]]) -> List[np.ndarray]:
    """Turn row-major embeddings into a list of (T, dim) arrays, one trajectory per problem."""
    return [rows_by_idx[idxs] for idxs in idx_seqs]


# =============================================================================
# CEBRA source training (delegates to cebra_EM)
# =============================================================================
# Triplet loss trains the encoder on source only; returned scaler/PCA/encoder freeze semantics
# for the target pass (true "transplant": same readout, different domain).


def _train_cebra_source(
    source_features_path: str,
    *,
    limit_problems: int,
    max_triplets_per_pid: int,
    cebra_dim: int,
    cebra_epochs: int,
) -> Tuple[np.ndarray, np.ndarray, object, object, torch.nn.Module]:
    """Load triplets from disk, train CEBRA + PCA on source rows; returns z and fitted transforms."""
    all_features, triplets = cebra_mod.load_and_prepare_cebra(
        source_features_path,
        mode="temporal",
        limit_problems=int(limit_problems),
        max_triplets=int(max_triplets_per_pid),
    )
    if not all_features:
        raise RuntimeError("No source features available after filtering.")
    if not triplets:
        raise RuntimeError("No source triplets generated; increase limit_problems.")
    return cebra_mod.train_cebra_row_embeddings(
        all_features,
        triplets,
        d_out=int(cebra_dim),
        n_epochs=int(cebra_epochs),
        batch_size=int(cebra_mod.BATCH_SIZE),
        pca_n_components=int(min(cebra_mod.PCA_DIM, cebra_dim)),
    )


# =============================================================================
# Transfer metrics: R², per-trajectory predictions, bootstrap ΔR²
# =============================================================================
# Aligns with cebra_EM’s linear_ar_r2 / regime_r2_on_pca but stores per-trajectory predictions
# so we can bootstrap ΔR² by resampling whole problems (not i.i.d. transitions).


def _r2_from_pred_true(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0 or true.size == 0:
        return float("nan")
    sse = float(np.sum((true - pred) ** 2))
    sst = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    if sst <= 0:
        return float("nan")
    return float(1.0 - (sse / sst))


def _collect_pred_true_by_traj(
    pca_target_seqs: List[np.ndarray],
    state_seqs_target: List[np.ndarray],
    pca_dim: int,
) -> List[dict]:
    # One global linear AR vs. per-regime linear maps; only transitions where the regime model fits.
    # Fit AR baseline on all transitions.
    x_in = np.vstack([s[:-1] for s in pca_target_seqs])
    x_out = np.vstack([s[1:] for s in pca_target_seqs])
    x_aug = np.hstack([x_in, np.ones((len(x_in), 1))])
    ar_coef, *_ = np.linalg.lstsq(x_aug, x_out, rcond=None)

    # Fit per-regime transition maps on all transitions.
    k_regimes = max(int(np.max(s)) for s in state_seqs_target) + 1
    x_in_k: List[List[np.ndarray]] = [[] for _ in range(k_regimes)]
    x_out_k: List[List[np.ndarray]] = [[] for _ in range(k_regimes)]
    for s_seq, p_seq in zip(state_seqs_target, pca_target_seqs):
        for t in range(len(s_seq) - 1):
            k = int(s_seq[t])
            x_in_k[k].append(p_seq[t])
            x_out_k[k].append(p_seq[t + 1])

    reg_coefs: List[np.ndarray | None] = []
    for k in range(k_regimes):
        if len(x_in_k[k]) < int(pca_dim) + 2:
            reg_coefs.append(None)
            continue
        xi = np.array(x_in_k[k])
        xo = np.array(x_out_k[k])
        coef, *_ = np.linalg.lstsq(np.hstack([xi, np.ones((len(xi), 1))]), xo, rcond=None)
        reg_coefs.append(coef)

    by_traj: List[dict] = []
    for s_seq, p_seq in zip(state_seqs_target, pca_target_seqs):
        ar_preds = []
        reg_preds = []
        truths = []
        for t in range(len(s_seq) - 1):
            x_t = p_seq[t]
            true_next = p_seq[t + 1]
            ar_next = np.append(x_t, 1.0) @ ar_coef
            k = int(s_seq[t])
            if reg_coefs[k] is None:
                continue
            reg_next = np.append(x_t, 1.0) @ reg_coefs[k]
            truths.append(true_next)
            ar_preds.append(ar_next)
            reg_preds.append(reg_next)
        if truths:
            by_traj.append(
                {
                    "true": np.array(truths),
                    "ar": np.array(ar_preds),
                    "reg": np.array(reg_preds),
                }
            )
    return by_traj


def _bootstrap_delta_r2(
    by_traj: List[dict],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    # Resample trajectories with replacement; each replicate pools transitions then compares R²s.
    if not by_traj:
        return {
            "n_boot": int(n_boot),
            "n_trajectories_used": 0,
            "delta_r2_mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_delta_r2_le_zero": float("nan"),
        }

    rng = np.random.default_rng(seed)
    deltas = []
    n = len(by_traj)
    for _ in range(int(n_boot)):
        picks = rng.integers(0, n, size=n)
        true = np.concatenate([by_traj[i]["true"] for i in picks], axis=0)
        ar = np.concatenate([by_traj[i]["ar"] for i in picks], axis=0)
        reg = np.concatenate([by_traj[i]["reg"] for i in picks], axis=0)
        r2_ar = _r2_from_pred_true(ar, true)
        r2_reg = _r2_from_pred_true(reg, true)
        deltas.append(float(r2_reg - r2_ar))
    arr = np.array(deltas, dtype=np.float64)
    return {
        "n_boot": int(n_boot),
        "n_trajectories_used": int(n),
        "delta_r2_mean": float(np.mean(arr)),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "p_delta_r2_le_zero": float(np.mean(arr <= 0.0)),
    }


# =============================================================================
# Timing metrics (next-stage label lift vs global majority)
# =============================================================================
# At each step we observe SLDS state s_t and the *next* row’s human stage label. We train
# majority-vote predictors on 80% of trajectories: (a) conditioned on s_t, (b) global. Lift is
# (a) minus (b) accuracy on held-out trajectories, with bootstrap CIs over trajectories.


def _bootstrap_timing_lift(
    target_features: List[dict],
    idx_seqs: List[List[int]],
    state_seqs_target: List[np.ndarray],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    # Build per-trajectory transitions of (state_t, stage_{t+1}).
    traj_pairs: List[List[Tuple[int, str]]] = []
    for idxs, s_seq in zip(idx_seqs, state_seqs_target):
        pairs: List[Tuple[int, str]] = []
        for t in range(min(len(idxs), len(s_seq)) - 1):
            st = int(s_seq[t])
            nxt_stage = str(target_features[idxs[t + 1]].get("stage", "UNKNOWN"))
            pairs.append((st, nxt_stage))
        if pairs:
            traj_pairs.append(pairs)

    if len(traj_pairs) < 2:
        return {
            "n_boot": int(n_boot),
            "n_train_trajectories": 0,
            "n_test_trajectories": 0,
            "n_test_transitions": 0,
            "state_to_next_stage_acc": float("nan"),
            "global_next_stage_acc": float("nan"),
            "timing_lift": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_timing_lift_le_zero": float("nan"),
        }

    rng = np.random.default_rng(seed)
    n = len(traj_pairs)
    perm = rng.permutation(n)
    n_train = max(1, int(0.8 * n))
    train_ids = perm[:n_train]
    test_ids = perm[n_train:]
    if len(test_ids) == 0:
        test_ids = perm[-1:]
        train_ids = perm[:-1]

    # Train mappings.
    state_stage_counts: Dict[int, Dict[str, int]] = {}
    global_stage_counts: Dict[str, int] = {}
    for i in train_ids:
        for st, nxt_stage in traj_pairs[i]:
            state_stage_counts.setdefault(st, {})
            state_stage_counts[st][nxt_stage] = state_stage_counts[st].get(nxt_stage, 0) + 1
            global_stage_counts[nxt_stage] = global_stage_counts.get(nxt_stage, 0) + 1
    if not global_stage_counts:
        return {
            "n_boot": int(n_boot),
            "n_train_trajectories": int(len(train_ids)),
            "n_test_trajectories": int(len(test_ids)),
            "n_test_transitions": 0,
            "state_to_next_stage_acc": float("nan"),
            "global_next_stage_acc": float("nan"),
            "timing_lift": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_timing_lift_le_zero": float("nan"),
        }

    state_majority = {s: max(cnts, key=cnts.get) for s, cnts in state_stage_counts.items()}
    global_majority = max(global_stage_counts, key=global_stage_counts.get)

    # Flatten held-out transitions.
    test_pairs: List[Tuple[int, str]] = []
    for i in test_ids:
        test_pairs.extend(traj_pairs[i])
    if not test_pairs:
        return {
            "n_boot": int(n_boot),
            "n_train_trajectories": int(len(train_ids)),
            "n_test_trajectories": int(len(test_ids)),
            "n_test_transitions": 0,
            "state_to_next_stage_acc": float("nan"),
            "global_next_stage_acc": float("nan"),
            "timing_lift": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "p_timing_lift_le_zero": float("nan"),
        }

    def _acc(pairs: List[Tuple[int, str]]) -> Tuple[float, float]:
        state_ok = 0
        base_ok = 0
        for st, nxt_stage in pairs:
            pred_state = state_majority.get(st, global_majority)
            if pred_state == nxt_stage:
                state_ok += 1
            if global_majority == nxt_stage:
                base_ok += 1
        n_local = len(pairs)
        return state_ok / n_local, base_ok / n_local

    state_acc, base_acc = _acc(test_pairs)
    lift = state_acc - base_acc

    # Bootstrap over held-out trajectories.
    test_trajs = [traj_pairs[i] for i in test_ids]
    bt = []
    m = len(test_trajs)
    for _ in range(int(n_boot)):
        picks = rng.integers(0, m, size=m)
        sample_pairs: List[Tuple[int, str]] = []
        for j in picks:
            sample_pairs.extend(test_trajs[j])
        if not sample_pairs:
            continue
        s_acc, b_acc = _acc(sample_pairs)
        bt.append(s_acc - b_acc)
    arr = np.array(bt, dtype=np.float64) if bt else np.array([float("nan")], dtype=np.float64)

    return {
        "n_boot": int(n_boot),
        "n_train_trajectories": int(len(train_ids)),
        "n_test_trajectories": int(len(test_ids)),
        "n_test_transitions": int(len(test_pairs)),
        "state_to_next_stage_acc": float(state_acc),
        "global_next_stage_acc": float(base_acc),
        "timing_lift": float(lift),
        "ci95_low": float(np.quantile(arr, 0.025)) if bt else float("nan"),
        "ci95_high": float(np.quantile(arr, 0.975)) if bt else float("nan"),
        "p_timing_lift_le_zero": float(np.mean(arr <= 0.0)) if bt else float("nan"),
    }


# =============================================================================
# LLM judging (OpenAI: timing disagreement cases)
# =============================================================================
# Where state-conditional and global majority predict *different* next-stage labels, we can ask
# a model which label better matches the actual next sentence text (optional; needs OPENAI_API_KEY).


def _openai_chat_json_object(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: int,
    max_retries: int,
) -> Dict[str, object]:
    """POST to Chat Completions with response_format json_object; returns parsed JSON or {"error": ...}."""
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = str(e)
            last_err = f"HTTP {e.code}: {err_body[:500]}"
        except Exception as e:  # pragma: no cover
            last_err = str(e)
        time.sleep(1.0 + 0.5 * attempt)
    return {"error": last_err or "openai_request_failed"}


def _judge_timing_predictions(
    target_features: List[dict],
    idx_seqs: List[List[int]],
    state_seqs_target: List[np.ndarray],
    *,
    judge_samples: int,
    judge_model: str,
    seed: int,
) -> dict:
    """Sample up to judge_samples disagreement cases and score transplant vs baseline by LLM vote."""
    if judge_samples <= 0:
        return {"enabled": False, "reason": "judge_samples_zero"}

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"enabled": False, "reason": "missing_OPENAI_API_KEY"}

    # Same train/test split idea as timing-lift metric.
    traj_pairs: List[List[Tuple[int, int, str]]] = []
    for idxs, s_seq in zip(idx_seqs, state_seqs_target):
        pairs: List[Tuple[int, int, str]] = []
        for t in range(min(len(idxs), len(s_seq)) - 1):
            st = int(s_seq[t])
            nxt_idx = idxs[t + 1]
            nxt_stage = str(target_features[nxt_idx].get("stage", "UNKNOWN"))
            pairs.append((st, nxt_idx, nxt_stage))
        if pairs:
            traj_pairs.append(pairs)
    if len(traj_pairs) < 2:
        return {"enabled": False, "reason": "insufficient_trajectories_for_judge"}

    rng = np.random.default_rng(seed)
    n = len(traj_pairs)
    perm = rng.permutation(n)
    n_train = max(1, int(0.8 * n))
    train_ids = perm[:n_train]
    test_ids = perm[n_train:]
    if len(test_ids) == 0:
        test_ids = perm[-1:]
        train_ids = perm[:-1]

    state_stage_counts: Dict[int, Dict[str, int]] = {}
    global_stage_counts: Dict[str, int] = {}
    for i in train_ids:
        for st, _nxt_idx, nxt_stage in traj_pairs[i]:
            state_stage_counts.setdefault(st, {})
            state_stage_counts[st][nxt_stage] = state_stage_counts[st].get(nxt_stage, 0) + 1
            global_stage_counts[nxt_stage] = global_stage_counts.get(nxt_stage, 0) + 1
    if not global_stage_counts:
        return {"enabled": False, "reason": "no_stage_counts_for_judge"}

    state_majority = {s: max(cnts, key=cnts.get) for s, cnts in state_stage_counts.items()}
    global_majority = max(global_stage_counts, key=global_stage_counts.get)

    candidates: List[dict] = []
    for i in test_ids:
        for st, nxt_idx, nxt_stage in traj_pairs[i]:
            pred_trans = state_majority.get(st, global_majority)
            pred_base = global_majority
            if pred_trans == pred_base:
                continue
            curr_idx = nxt_idx - 1
            curr_sent = str(target_features[curr_idx].get("sentence", ""))
            next_sent = str(target_features[nxt_idx].get("sentence", ""))
            candidates.append(
                {
                    "state_t": int(st),
                    "pred_transplant": str(pred_trans),
                    "pred_baseline": str(pred_base),
                    "gold_next_stage": str(nxt_stage),
                    "current_sentence": curr_sent,
                    "next_sentence": next_sent,
                }
            )
    if not candidates:
        return {"enabled": False, "reason": "no_disagreement_cases_for_judge"}

    rng.shuffle(candidates)
    use_rows = candidates[: int(min(judge_samples, len(candidates)))]

    system_prompt = (
        "You evaluate which predicted reasoning-stage label better matches an observed next sentence. "
        "Return strict JSON only."
    )
    rows = []
    for row in use_rows:
        user_prompt = f"""Task: compare two candidate labels for the NEXT sentence in a reasoning trace.

Current sentence:
{row['current_sentence']!r}

Observed next sentence:
{row['next_sentence']!r}

Candidate A (transplant prediction): {row['pred_transplant']}
Candidate B (baseline prediction): {row['pred_baseline']}

Return JSON:
{{
  "winner": "A" | "B" | "tie",
  "confidence": 0.0-1.0,
  "reason": "one short sentence"
}}
"""
        judged = _openai_chat_json_object(
            api_key=api_key,
            model=judge_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_s=60,
            max_retries=3,
        )
        rows.append({"case": row, "judge": judged})

    valid = [r for r in rows if isinstance(r.get("judge"), dict) and "error" not in r["judge"]]
    if not valid:
        return {"enabled": True, "model": judge_model, "rows": rows, "summary": {"valid_count": 0}}

    a = sum(1 for r in valid if str(r["judge"].get("winner", "")).strip().lower() == "a")
    b = sum(1 for r in valid if str(r["judge"].get("winner", "")).strip().lower() == "b")
    tie = sum(1 for r in valid if str(r["judge"].get("winner", "")).strip().lower() == "tie")
    n = len(valid)
    return {
        "enabled": True,
        "model": judge_model,
        "rows": rows,
        "summary": {
            "valid_count": int(n),
            "transplant_win_rate": float(a / n),
            "baseline_win_rate": float(b / n),
            "tie_rate": float(tie / n),
            "non_tie_transplant_win_rate": float(a / max(1, (a + b))),
        },
    }


# =============================================================================
# Dataset defaults (Hugging Face repo / relpaths)
# =============================================================================
# Maps --dataset / --model-size to the Hub repo and default pickle paths (overridable via CLI).


def _dataset_defaults(dataset: str, model_size: str) -> Tuple[str, str, str]:
    ds = dataset.strip().lower()
    ms = model_size.strip().lower()
    if ds == "gsm8k":
        family = "qwen_1.5b" if ms == "1.5b" else "qwen_14b"
        source_rel = default_hub_features_relpath(family=family, role="reasoning")
        target_rel = default_hub_features_relpath(family=family, role="base")
        return SDS_TRAIN_GSM8K_REPO_ID, source_rel, target_rel
    if ds == "math500":
        repo = "withmartian/SDS_math500_test"
        if ms == "1.5b":
            return (
                repo,
                "Qwen_1_5B_reasoning/layer_27/all_sentences_features.pkl",
                "Qwen_1_5B_base/layer_27/all_sentences_features.pkl",
            )
        return (
            repo,
            "Qwen_14B_reasoning/layer_47/all_sentences_features.pkl",
            "Qwen_14B_base/layer_47/all_sentences_features.pkl",
        )
    raise ValueError(f"Unsupported --dataset={dataset!r}")


# =============================================================================
# CLI entrypoint
# =============================================================================


def main() -> int:
    # --- CLI: load data (Hub or local pickles), filter rows, snapshot subsets for reproducibility ---
    ap = argparse.ArgumentParser(
        description="Train CEBRA-EM on reasoning features and evaluate transfer on base features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500"])
    ap.add_argument("--dataset-repo", type=str, default=None)
    ap.add_argument("--source-relpath", type=str, default=None, help="HF dataset relpath for source features pickle.")
    ap.add_argument("--target-relpath", type=str, default=None, help="HF dataset relpath for target features pickle.")
    ap.add_argument("--source-features-pkl", type=str, default=None)
    ap.add_argument("--target-features-pkl", type=str, default=None)
    ap.add_argument("--model-size", type=str, default="1.5b", choices=["1.5b", "14b"])
    ap.add_argument("--limit-problems", type=int, default=8)
    ap.add_argument("--all-problems", action="store_true")
    ap.add_argument("--max-triplets-per-pid", type=int, default=10)
    ap.add_argument("--cebra-dim", type=int, default=40)
    ap.add_argument("--cebra-epochs", type=int, default=3)
    ap.add_argument("--k-regimes", type=int, default=4)
    ap.add_argument("--em-iters", type=int, default=3)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--judge-samples", type=int, default=0, help="Judge sample count for timing disagreements.")
    ap.add_argument("--judge-model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="transplant_sds_artifacts")
    ap.add_argument("--out-name", type=str, default="transplant_eval_summary.json")
    args = ap.parse_args()

    _seed_everything(int(args.seed))

    family = "qwen_1.5b" if args.model_size == "1.5b" else "qwen_14b"
    default_repo, source_rel, target_rel = _dataset_defaults(args.dataset, args.model_size)
    dataset_repo = args.dataset_repo or default_repo
    if args.source_relpath:
        source_rel = args.source_relpath
    if args.target_relpath:
        target_rel = args.target_relpath

    token = os.environ.get("HF_TOKEN", "").strip() or None
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    if args.source_features_pkl:
        source_path = os.path.abspath(args.source_features_pkl)
    else:
        source_path = _download_features(dataset_repo, source_rel, token)
    if args.target_features_pkl:
        target_path = os.path.abspath(args.target_features_pkl)
    else:
        target_path = _download_features(dataset_repo, target_rel, token)

    source_features_all = _load_pickle_list(source_path)
    target_features_all = _load_pickle_list(target_path)
    limit_problems = None if args.all_problems else int(args.limit_problems)
    source_features = _subset_non_neutral_by_problem(source_features_all, limit_problems)
    target_features = _subset_non_neutral_by_problem(target_features_all, limit_problems)

    if not source_features or not target_features:
        raise RuntimeError("No source/target features after non-neutral and problem filters.")

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    model_tag = "q15b" if args.model_size == "1.5b" else "q14b"
    ns_tag = f"s{len(source_features)}t{len(target_features)}"
    run_dir_name = f"xfer_{args.dataset}_{model_tag}_{ns_tag}_{stamp}"
    out_dir = os.path.join(out_root, run_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    prob_tag = "all" if args.all_problems else f"n{int(args.limit_problems)}"
    source_subset_path = os.path.join(out_dir, f"source_subset_{family}_{prob_tag}.pkl")
    target_subset_path = os.path.join(out_dir, f"target_subset_{family}_{prob_tag}.pkl")
    _save_pickle_list(source_subset_path, source_features)
    _save_pickle_list(target_subset_path, target_features)

    # --- Representation: CEBRA on source; same transforms applied to target (no target CEBRA training) ---
    z_source, pca_source, scaler, pca, encoder = _train_cebra_source(
        source_subset_path,
        limit_problems=10**9 if args.all_problems else int(args.limit_problems),
        max_triplets_per_pid=int(args.max_triplets_per_pid),
        cebra_dim=int(args.cebra_dim),
        cebra_epochs=int(args.cebra_epochs),
    )

    source_grouped = _group_sequences(source_features, min_len=3)
    if not source_grouped.idx_seqs:
        raise RuntimeError("Source set has no trajectories of length >= 3.")
    z_source_seqs = _build_sequences_from_rows(z_source, source_grouped.idx_seqs)

    # --- Sequences in z (for SLDS) and PCA (for R²); target z uses source-fitted encoder only ---
    z_target, pca_target = cebra_mod.embed_cebra_with_fitted_transforms(
        target_features, scaler, pca, encoder
    )
    target_grouped = _group_sequences(target_features, min_len=3)
    if not target_grouped.idx_seqs:
        raise RuntimeError("Target set has no trajectories of length >= 3.")
    z_target_seqs = _build_sequences_from_rows(z_target, target_grouped.idx_seqs)
    pca_target_seqs = _build_sequences_from_rows(pca_target, target_grouped.idx_seqs)

    # --- Dynamics: SLDS fit on source z-trajectories only; then decode target with same parameters ---
    pi, a, d_m, d_b, d_cov = cebra_mod.fit_slds_em_iters(
        z_source_seqs,
        int(args.k_regimes),
        int(args.cebra_dim),
        int(args.em_iters),
    )

    # Posterior over SLDS states on target z; hard states used for regime R² and timing.
    state_seqs_target, lls_target = cebra_mod.infer_gamma_argmax_states(
        z_target_seqs,
        pi,
        a,
        d_m,
        d_b,
        d_cov,
        int(args.k_regimes),
    )

    # --- Metrics: PCA prediction quality, SLDS likelihood, timing, optional LLM judge ---
    ar_r2 = float(cebra_mod.linear_ar_r2(pca_target_seqs))
    regime_r2 = float(cebra_mod.regime_r2_on_pca(state_seqs_target, pca_target_seqs))
    delta_r2 = float(regime_r2 - ar_r2)
    nll = cebra_mod.slds_nll_per_transition(lls_target, z_target_seqs)
    by_traj = _collect_pred_true_by_traj(
        pca_target_seqs=pca_target_seqs,
        state_seqs_target=state_seqs_target,
        pca_dim=int(args.cebra_dim),
    )
    delta_r2_boot = _bootstrap_delta_r2(
        by_traj,
        n_boot=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    timing = _bootstrap_timing_lift(
        target_features=target_features,
        idx_seqs=target_grouped.idx_seqs,
        state_seqs_target=state_seqs_target,
        n_boot=int(args.n_bootstrap),
        seed=int(args.seed) + 17,
    )
    timing_judge = _judge_timing_predictions(
        target_features=target_features,
        idx_seqs=target_grouped.idx_seqs,
        state_seqs_target=state_seqs_target,
        judge_samples=int(args.judge_samples),
        judge_model=str(args.judge_model),
        seed=int(args.seed) + 31,
    )

    # --- Write JSON summary and print the same headline numbers to stdout ---
    summary = {
        "mode": "cebra_em_transfer_no_steering",
        "dataset": args.dataset,
        "dataset_repo": dataset_repo,
        "run_dir": out_dir,
        "source_features_original": source_path,
        "target_features_original": target_path,
        "source_features_subset": source_subset_path,
        "target_features_subset": target_subset_path,
        "source_hub_relpath": source_rel,
        "target_hub_relpath": target_rel,
        "config": {
            "model_size": args.model_size,
            "all_problems": bool(args.all_problems),
            "limit_problems": None if args.all_problems else int(args.limit_problems),
            "max_triplets_per_pid": int(args.max_triplets_per_pid),
            "cebra_dim": int(args.cebra_dim),
            "cebra_epochs": int(args.cebra_epochs),
            "k_regimes": int(args.k_regimes),
            "em_iters": int(args.em_iters),
            "n_bootstrap": int(args.n_bootstrap),
            "judge_samples": int(args.judge_samples),
            "judge_model": str(args.judge_model),
            "seed": int(args.seed),
        },
        "counts": {
            "source_rows": int(len(source_features)),
            "target_rows": int(len(target_features)),
            "source_trajectories": int(len(z_source_seqs)),
            "target_trajectories": int(len(z_target_seqs)),
            "target_transitions": int(sum(max(0, len(s) - 1) for s in z_target_seqs)),
        },
        "metrics": {
            "linear_ar_r2": ar_r2,
            "regime_r2": regime_r2,
            "delta_r2": delta_r2,
            "slds_target_nll_per_transition": nll,
            "delta_r2_bootstrap": delta_r2_boot,
            "timing_metrics": timing,
            "timing_judge": timing_judge,
        },
    }

    out_path = os.path.join(out_dir, args.out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CEBRA-EM Transfer Evaluation ===")
    print(f"Dataset: {args.dataset} | Repo: {dataset_repo}")
    print(f"Source: {source_rel}")
    print(f"Target: {target_rel}")
    print(f"Trajectories (source/target): {len(z_source_seqs)}/{len(z_target_seqs)}")
    print(f"linear_ar_r2: {ar_r2:.6f}")
    print(f"regime_r2:    {regime_r2:.6f}")
    print(f"delta_r2:     {delta_r2:.6f}")
    print(
        "delta_r2 95% CI: "
        f"[{delta_r2_boot['ci95_low']:.6f}, {delta_r2_boot['ci95_high']:.6f}] "
        f"(p<=0: {delta_r2_boot['p_delta_r2_le_zero']:.4f})"
    )
    print(f"target_nll:   {nll:.6f}")
    print(
        "timing lift (held-out next-stage acc): "
        f"{timing['timing_lift']:.6f} "
        f"[{timing['ci95_low']:.6f}, {timing['ci95_high']:.6f}] "
        f"(p<=0: {timing['p_timing_lift_le_zero']:.4f})"
    )
    if timing_judge.get("enabled"):
        js = timing_judge.get("summary", {})
        print(
            "judge timing preference: "
            f"transplant={js.get('transplant_win_rate', float('nan')):.4f} "
            f"baseline={js.get('baseline_win_rate', float('nan')):.4f} "
            f"tie={js.get('tie_rate', float('nan')):.4f}"
        )
    else:
        print(f"judge timing preference: skipped ({timing_judge.get('reason', 'unknown')})")
    print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
