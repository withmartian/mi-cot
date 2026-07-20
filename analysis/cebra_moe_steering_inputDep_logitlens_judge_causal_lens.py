"""
CEBRA-MoE steering with judge-defined state annotations — **causal-style logit lens**.

Same steering + OpenAI judge protocol as ``cebra_moe_steering_inputDep_logitlens_judge_only.py``,
but logit-lens geometry matches ``cebra_causal.py``:

- **Unembedding:** ``lm_head.weight`` (not ``get_output_embeddings()``; usually tied, explicit here).
- **Activations:** **StandardScaler**-normalized hidden states for centroids and for pre/post lens
  (``cebra_causal`` uses scaled ``X_torch`` for ``centroid @ lm_head.weight.t()``).

Steering edits remain in **raw** activation space; only **readouts** use scaled space for lens.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

_here = os.path.abspath(os.path.dirname(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from sds_train_gsm8k_hf import SDS_TRAIN_GSM8K_REPO_ID, default_hub_features_relpath
from hf_steering_data_io import (
    add_hf_dataset_cli_args,
    apply_hf_dataset_cli_config,
    resolve_hf_subset_data_path,
)

try:
    import cebra_MoE as cebra_mod
    from cebra_EM import load_and_prepare_cebra
    from cebra_MoE import fit_regime_linear_dynamics, hard_transition_matrix, train_moe_projection
except Exception as e:  # pragma: no cover
    raise ImportError("Failed to import `cebra_MoE.py` / `cebra_EM.load_and_prepare_cebra`.") from e


@dataclass
class SteerConfig:
    data_path: str = "rpc_dataset_layer28_200/all_sentences_features.pkl"
    use_last_token: bool = True
    limit_problems: int = 500
    max_triplets_per_pid: int = 25
    cebra_dim: int = 40
    moe_epochs: int = 75
    k_regimes: int = 4
    transition_kappa: float = 1.0
    beta: float = 8.0
    steer_alpha: float = 8.0

    enable_openai_judge: bool = True
    judge_model: str = "gpt-4.1-mini"
    judge_max_samples: int = 100
    judge_timeout_s: int = 60
    judge_max_retries: int = 3
    logit_lens_model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    logit_lens_top_k: int = 10
    # None / "auto" -> cuda if torch.cuda.is_available() else cpu
    logit_lens_device: Optional[str] = None

    save_dir: str = "cebra_moe_steering_logitlens_judge_causal_lens"
    hf_auto_download_if_missing: bool = True
    hf_dataset_repo: str = SDS_TRAIN_GSM8K_REPO_ID
    hf_dataset_filename: str = default_hub_features_relpath("qwen_14b", "reasoning")
    hf_fallback_num_samples: int = 500
    hf_fallback_cache_name: str = "sds_hf500_default_qwen14b_reasoning_l47.pkl"


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "CEBRA-MoE steering with logit-lens-first; OpenAI annotates regimes as reasoning stages "
            "(cebra_MoE.STAGES), then judges post-steering readouts."
        )
    )
    add_hf_dataset_cli_args(p)
    p.add_argument(
        "--moe-epochs",
        type=int,
        default=None,
        metavar="N",
        help="MoE training epochs (default: SteerConfig.moe_epochs).",
    )
    p.add_argument("--no-openai-judge", action="store_true")
    p.add_argument(
        "--logit-lens-model-id",
        type=str,
        default=None,
        help="HF model id for unembedding in logit lens (must match hidden size of steering activations).",
    )
    p.add_argument(
        "--logit-lens-device",
        type=str,
        default=None,
        metavar="auto|cuda|cpu",
        help="Where to run logit-lens matmuls (default: auto = cuda if available).",
    )
    p.add_argument(
        "--judge-max-samples",
        type=int,
        default=None,
        metavar="N",
        help="OpenAI judge calls: max steered rows (sorted by index). 0 or negative = judge all rows in cache.",
    )
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--limit-problems", type=int, default=None)
    return p


def config_from_cli_args(cfg: SteerConfig, args: argparse.Namespace) -> SteerConfig:
    c = cfg
    if args.no_hf_fallback:
        c = replace(c, hf_auto_download_if_missing=False)
    if getattr(args, "no_openai_judge", False):
        c = replace(c, enable_openai_judge=False)
    if getattr(args, "logit_lens_model_id", None):
        c = replace(c, logit_lens_model_id=str(args.logit_lens_model_id).strip())
    if getattr(args, "logit_lens_device", None):
        c = replace(c, logit_lens_device=str(args.logit_lens_device).strip())
    if getattr(args, "moe_epochs", None) is not None:
        c = replace(c, moe_epochs=int(args.moe_epochs))
    if args.save_dir is not None:
        c = replace(c, save_dir=args.save_dir)
    if getattr(args, "judge_max_samples", None) is not None:
        c = replace(c, judge_max_samples=int(args.judge_max_samples))
    c = apply_hf_dataset_cli_config(c, args)
    if args.limit_problems is not None:
        c = replace(c, limit_problems=int(args.limit_problems))
    return c


def _openai_chat_json_object(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: int,
    max_retries: int,
) -> Dict[str, Any]:
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


def fit_latent_to_activation_decoder(z_flat: np.ndarray, x_scaled_flat: np.ndarray) -> np.ndarray:
    z_aug = np.hstack([z_flat, np.ones((len(z_flat), 1), dtype=z_flat.dtype)])
    coef, *_ = np.linalg.lstsq(z_aug, x_scaled_flat, rcond=None)
    return coef[:-1]


def kl_regularized_policy(p: np.ndarray, target_k: int, beta: float) -> np.ndarray:
    p = p / (np.sum(p) + 1e-12)
    g = np.zeros_like(p)
    g[target_k] = 1.0
    q_unnorm = p * np.exp(beta * g)
    return q_unnorm / np.sum(q_unnorm)


def compute_statewise_next_means(z_t: np.ndarray, d_m: np.ndarray, d_b: np.ndarray) -> np.ndarray:
    return np.stack([d_m[k] @ z_t + d_b[k] for k in range(d_m.shape[0])], axis=0)


def compute_steering_delta(
    z_t: np.ndarray,
    s_t: int,
    target_k: int,
    a: np.ndarray,
    d_m: np.ndarray,
    d_b: np.ndarray,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = a[s_t].copy()
    p = p / (np.sum(p) + 1e-12)
    q = kl_regularized_policy(p, target_k, beta)
    f_k = compute_statewise_next_means(z_t, d_m, d_b)
    mu_orig = np.sum(p[:, None] * f_k, axis=0)
    mu_steered = np.sum(q[:, None] * f_k, axis=0)
    return mu_steered - mu_orig, p, q


def resolve_logit_lens_device(explicit: Optional[str]) -> str:
    """Return ``cuda`` or ``cpu`` for logit-lens tensors."""
    import torch

    if explicit is None or str(explicit).strip().lower() in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    v = str(explicit).strip().lower()
    if v == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "logit_lens_device=cuda but torch.cuda.is_available() is False. "
                "Install a PyTorch build matching your NVIDIA driver (e.g. cu124 wheels for CUDA 12.x drivers)."
            )
        return "cuda"
    if v == "cpu":
        return "cpu"
    raise ValueError(f"Unknown logit_lens_device {explicit!r} (use auto, cuda, or cpu).")


def _load_logit_lens(cfg: SteerConfig):
    """Load ``lm_head.weight`` like ``cebra_causal.apply_logit_lens`` (logits = h @ W_U.t())."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lens_device = resolve_logit_lens_device(getattr(cfg, "logit_lens_device", None))
    dev = torch.device(lens_device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.logit_lens_model_id, trust_remote_code=True)
    load_kw: Dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if lens_device == "cuda":
        load_kw["dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(cfg.logit_lens_model_id, **load_kw)
    model.eval()
    model.to(dev)
    if not hasattr(model, "lm_head") or model.lm_head is None:
        raise RuntimeError("Model has no lm_head for causal-style logit lens.")
    with torch.no_grad():
        # HF: lm_head.weight is [vocab, d_model]; cebra_causal uses logits = centroid @ W_U.t()
        w_u = model.lm_head.weight.detach().to(device=dev, dtype=torch.float32).T.contiguous()
    del model
    gc.collect()
    if lens_device == "cuda":
        torch.cuda.empty_cache()
    print(
        f"[logit_lens/causal] device={lens_device}  lm_head^T shape={tuple(w_u.shape)}  dtype=float32",
        flush=True,
    )
    return tokenizer, w_u, lens_device


def build_state_signatures(
    cfg: SteerConfig,
    x_lens: np.ndarray,
    per_sample_state: np.ndarray,
    included_indices: List[int],
    tokenizer,
    w_u,
    lens_device: str,
) -> Dict[int, dict]:
    """
    Regime centroids in **lens space** (same as ``cebra_causal``: mean of **StandardScaler** rows).
    ``x_lens`` must align row-wise with ``x_raw`` (typically ``scaler.fit_transform(x_raw)``).
    """
    import torch
    import torch.nn.functional as F

    dev = torch.device(lens_device)
    state_to_idxs: Dict[int, List[int]] = {}
    for idx in included_indices:
        s = int(per_sample_state[idx])
        state_to_idxs.setdefault(s, []).append(int(idx))
    k = max(1, int(cfg.logit_lens_top_k))
    out: Dict[int, dict] = {}
    for s, idxs in state_to_idxs.items():
        centroid = np.mean(x_lens[idxs], axis=0).astype(np.float32)
        c = torch.from_numpy(centroid).to(device=dev, dtype=torch.float32)
        logits = c @ w_u
        top_vals, top_ids = torch.topk(logits, k=min(k, int(logits.shape[0])))
        probs = F.softmax(logits, dim=-1)
        top_probs = probs[top_ids].tolist()
        out[int(s)] = {
            "token_ids": [int(i) for i in top_ids.tolist()],
            "tokens": [tokenizer.decode([int(i)]).strip() for i in top_ids.tolist()],
            "logits": [float(v) for v in top_vals.tolist()],
            "probs": [float(p) for p in top_probs],
        }
    return out


def _reasoning_stage_annotation_system_prompt() -> str:
    """Paper-aligned guidance (cf. ``4_expms`` functional specialization + ``cebra_MoE.STAGES``)."""
    stages = ", ".join(cebra_mod.STAGES)
    return (
        "You name discrete latent **reasoning-policy regimes** for chain-of-thought models. "
        "Evidence is only the logit-lens top tokens at one layer (vocabulary projections of a regime centroid).\n\n"
        "Goals (match SDS / latent-policy framing in the paper draft):\n"
        "- Labels must describe **what kind of reasoning step** the model tends to emit next "
        "(planning, deriving, checking, consolidating, retrieving facts, emitting the answer, etc.).\n"
        "- Do **not** give a label whose *substance* is only punctuation, whitespace, or formatting "
        "(e.g. “periods and commas”). If tokens are mostly punctuation or layout, infer the **reasoning phase** "
        "those positions usually accompany (e.g. after-step verification, transition between steps, "
        "problem framing, final wrap-up).\n"
        "- Prefer language consistent with human reasoning-stage schemas used alongside SDS: "
        "computation-heavy vs verification vs planning/setup vs transitional scaffold vs consolidation.\n"
        "- Stay faithful: every claim in state_description must be plausible from the listed tokens "
        "(numbers, math words, “therefore/check/if”, domain terms, etc. count as evidence).\n\n"
        f"You must set aligned_stage to exactly one of these enum strings: {stages}.\n"
        "Pick the single best-matching stage; use the closest semantic match even when tokens are sparse.\n"
        "Return strict JSON only."
    )


def judge_state_annotations(cfg: SteerConfig, state_signatures: Dict[int, dict]) -> Dict[int, dict]:
    if not cfg.enable_openai_judge:
        return {s: {"state_annotation": f"State {s}", "confidence": 0.0} for s in state_signatures}
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {s: {"state_annotation": f"State {s}", "confidence": 0.0} for s in state_signatures}

    system_prompt = _reasoning_stage_annotation_system_prompt()
    valid_stages = set(cebra_mod.STAGES)
    annotations: Dict[int, dict] = {}
    for s, sig in sorted(state_signatures.items()):
        user_prompt = f"""Latent regime id: {s}
Logit-lens top tokens (centroid): {sig.get('tokens', [])}
Top token logits: {sig.get('logits', [])}

Return JSON with exactly these keys:
{{
  "state_annotation": "<3-8 words, human-readable reasoning regime name; NOT punctuation-focused>",
  "state_description": "<one sentence: how this token pattern supports that reasoning role>",
  "aligned_stage": "<exactly one of the allowed enum strings>",
  "confidence": <0 to 1>
}}"""
        judged = _openai_chat_json_object(
            api_key=api_key,
            model=cfg.judge_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_s=cfg.judge_timeout_s,
            max_retries=cfg.judge_max_retries,
        )
        if "error" in judged:
            annotations[s] = {
                "state_annotation": f"State {s}",
                "state_description": "Judge unavailable.",
                "confidence": 0.0,
            }
        else:
            raw_stage = judged.get("aligned_stage")
            if isinstance(raw_stage, str) and raw_stage.strip() in valid_stages:
                judged["aligned_stage"] = raw_stage.strip()
            elif isinstance(raw_stage, str):
                upper = raw_stage.strip().upper().replace(" ", "_")
                if upper in valid_stages:
                    judged["aligned_stage"] = upper
                else:
                    judged["aligned_stage"] = None
            else:
                judged["aligned_stage"] = None
            annotations[s] = judged
    return annotations


def activation_topk_logit_lens(
    x_vec_raw: np.ndarray,
    tokenizer,
    w_u,
    top_k: int,
    lens_device: str,
    scaler: StandardScaler,
) -> Dict[str, Any]:
    """Top-k logits/probs for one activation; **scaled** like ``cebra_causal`` (transform then lens)."""
    import torch
    import torch.nn.functional as F

    dev = torch.device(lens_device)
    k = max(1, int(top_k))
    x_flat = np.asarray(x_vec_raw, dtype=np.float64).reshape(1, -1)
    x_sc = scaler.transform(x_flat).astype(np.float32).flatten()
    c = torch.from_numpy(x_sc).to(device=dev, dtype=torch.float32)
    logits = c @ w_u
    top_vals, top_ids = torch.topk(logits, k=min(k, int(logits.shape[0])))
    probs = F.softmax(logits, dim=-1)
    top_probs = probs[top_ids].tolist()
    return {
        "token_ids": [int(i) for i in top_ids.tolist()],
        "tokens": [tokenizer.decode([int(i)]).strip() for i in top_ids.tolist()],
        "logits": [float(v) for v in top_vals.tolist()],
        "probs": [float(p) for p in top_probs],
    }


def summarize_logit_lens_rows(lens_rows: List[dict]) -> Dict[str, float]:
    if not lens_rows:
        return {
            "n": 0.0,
            "logit_lens_success_pct": 0.0,
            "mean_margin_lift": 0.0,
            "mean_target_logit_lift": 0.0,
        }
    n = len(lens_rows)
    return {
        "n": float(n),
        "logit_lens_success_pct": 100.0
        * float(np.mean([float(r["logit_lens_success"]) for r in lens_rows])),
        "mean_margin_lift": float(np.mean([float(r["margin_lift"]) for r in lens_rows])),
        "mean_target_logit_lift": float(np.mean([float(r["target_logit_lift"]) for r in lens_rows])),
    }


def run_logit_lens_rows(
    steering_cache: Dict[int, dict],
    x_scaled: np.ndarray,
    scaler: StandardScaler,
    state_signatures: Dict[int, dict],
    w_u,
    lens_device: str,
) -> List[dict]:
    import torch

    dev = torch.device(lens_device)
    rows: List[dict] = []
    for idx, entry in steering_cache.items():
        s = int(entry["source_state"])
        t = int(entry["target_state"])
        sig_t = state_signatures.get(t, {}).get("token_ids", [])
        sig_s = state_signatures.get(s, {}).get("token_ids", [])
        if not sig_t or not sig_s:
            continue
        x0 = torch.from_numpy(np.asarray(x_scaled[idx], dtype=np.float32)).to(device=dev, dtype=torch.float32)
        x1_raw = np.asarray(entry["x_steered"], dtype=np.float64).reshape(1, -1)
        x1_sc = scaler.transform(x1_raw).astype(np.float32).flatten()
        x1 = torch.from_numpy(x1_sc).to(device=dev, dtype=torch.float32)
        t0 = float((x0 @ w_u[:, sig_t]).mean().item())
        t1 = float((x1 @ w_u[:, sig_t]).mean().item())
        s0 = float((x0 @ w_u[:, sig_s]).mean().item())
        s1 = float((x1 @ w_u[:, sig_s]).mean().item())
        margin0 = t0 - s0
        margin1 = t1 - s1
        margin_lift = margin1 - margin0
        rows.append(
            {
                "idx": int(idx),
                "source_state": s,
                "target_state": t,
                "target_logit_lift": float(t1 - t0),
                "margin_lift": margin_lift,
                "logit_lens_success": int(margin_lift > 0.0),
            }
        )
    return rows


def _judge_steering_success_bool(judge_obj: Dict[str, Any]) -> bool:
    if not isinstance(judge_obj, dict):
        return False
    if "steering_toward_target_success" in judge_obj:
        return bool(judge_obj.get("steering_toward_target_success"))
    return bool(judge_obj.get("logitlens_annotated_steerability_success"))


def compute_transition_breakdown(
    valid_rows: List[dict],
    k_regimes: Optional[int] = None,
    edge_counts: Optional[Dict[Tuple[int, int], int]] = None,
) -> List[Dict[str, Any]]:
    """
    Aggregate judge outcomes per (source, target). If k_regimes is set, always emit one row per
    cyclic edge s -> (s+1) mod K.

    ``n_steered`` counts rows in the steering cache for that edge (full data).
    ``n_judged`` counts successful OpenAI rows used for percentages (subset if capped).
    """

    buckets: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for r in valid_rows:
        s = int(r["source_state"])
        t = int(r["target_state"])
        buckets[(s, t)].append(r)

    def row_for_bucket(s: int, t: int, lst: List[dict]) -> Dict[str, Any]:
        n_j = len(lst)
        n_st = int(edge_counts.get((s, t), 0)) if edge_counts is not None else n_j
        if n_j == 0:
            return {
                "source_state": s,
                "target_state": t,
                "n_steered": n_st,
                "n_judged": 0,
                "n": n_st,
                "steer_pct": 0.0,
                "coherence_pct": 0.0,
                "mean_confidence": 0.0,
            }
        steer = float(
            np.mean([1.0 if _judge_steering_success_bool(v["judge"]) else 0.0 for v in lst])
        )
        coh = float(
            np.mean(
                [1.0 if bool(v["judge"].get("coherence_success", False)) else 0.0 for v in lst]
            )
        )
        conf = float(np.mean([float(v["judge"].get("confidence", 0.0)) for v in lst]))
        return {
            "source_state": s,
            "target_state": t,
            "n_steered": n_st,
            "n_judged": n_j,
            "n": n_st,
            "steer_pct": 100.0 * steer,
            "coherence_pct": 100.0 * coh,
            "mean_confidence": conf,
        }

    if k_regimes is not None and int(k_regimes) > 0:
        k = int(k_regimes)
        out: List[Dict[str, Any]] = []
        for s in range(k):
            t = (s + 1) % k
            lst = buckets.get((s, t), [])
            out.append(row_for_bucket(s, t, lst))
        return out

    out_obs: List[Dict[str, Any]] = []
    for (s, t) in sorted(buckets.keys()):
        out_obs.append(row_for_bucket(s, t, buckets[(s, t)]))
    return out_obs


def judge_steering_post_activation_lens(
    cfg: SteerConfig,
    steering_cache: Dict[int, dict],
    state_annotations: Dict[int, dict],
    state_signatures: Dict[int, dict],
    x_raw: np.ndarray,
    scaler: StandardScaler,
    tokenizer,
    w_u,
    edge_counts: Dict[Tuple[int, int], int],
    lens_device: str,
) -> dict:
    """
    Judge uses centroid signatures + annotations, then pre/post top-k readouts on each steered activation.
    """
    k_reg = int(cfg.k_regimes)
    empty_br = compute_transition_breakdown([], k_regimes=k_reg, edge_counts=edge_counts)

    if not cfg.enable_openai_judge:
        return {
            "enabled": False,
            "reason": "judge_disabled",
            "rows": [],
            "summary": {},
            "transition_breakdown": empty_br,
        }
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "enabled": False,
            "reason": "missing_OPENAI_API_KEY",
            "rows": [],
            "summary": {},
            "transition_breakdown": empty_br,
        }

    system_prompt = (
        "You evaluate hidden-state steering for a single reasoning model. "
        "Regime labels were inferred from logit-lens centroid evidence; you now see centroid references "
        "and the logit-lens readout on one token's activation BEFORE and AFTER a steering vector was added. "
        "Judge only from the provided tokens and logits. Return strict JSON."
    )

    rows: List[dict] = []
    keys_sorted = sorted(steering_cache.keys())
    jm = int(cfg.judge_max_samples)
    ordered_ids = keys_sorted if jm <= 0 else keys_sorted[:jm]
    eps = 1e-12
    k = int(cfg.logit_lens_top_k)

    for idx in ordered_ids:
        row = steering_cache[int(idx)]
        s = int(row["source_state"])
        t = int(row["target_state"])
        p = np.asarray(row["p_next"], dtype=np.float64)
        q = np.asarray(row["q_next"], dtype=np.float64)
        p = p / (np.sum(p) + eps)
        q = q / (np.sum(q) + eps)
        src_ann = state_annotations.get(s, {}).get("state_annotation", f"State {s}")
        tgt_ann = state_annotations.get(t, {}).get("state_annotation", f"State {t}")
        sig_s = state_signatures.get(s, {})
        sig_t = state_signatures.get(t, {})

        x_pre = x_raw[int(idx)]
        x_post = np.asarray(row["x_steered"], dtype=np.float32)
        lens_pre = activation_topk_logit_lens(x_pre, tokenizer, w_u, k, lens_device, scaler)
        lens_post = activation_topk_logit_lens(x_post, tokenizer, w_u, k, lens_device, scaler)

        user_prompt = f"""Steering evaluation (one timestep).

SOURCE regime id: {s}
SOURCE annotation (from centroid logit lens): {src_ann!r}
SOURCE centroid top tokens: {sig_s.get("tokens", [])}
SOURCE centroid top logits: {[round(x, 4) for x in sig_s.get("logits", [])]}

TARGET regime id: {t}
TARGET annotation (from centroid logit lens): {tgt_ann!r}
TARGET centroid top tokens: {sig_t.get("tokens", [])}
TARGET centroid top logits: {[round(x, 4) for x in sig_t.get("logits", [])]}

This position — BEFORE steering (actual activation):
  top tokens: {lens_pre.get("tokens", [])}
  top logits: {[round(x, 4) for x in lens_pre.get("logits", [])]}

This position — AFTER steering:
  top tokens: {lens_post.get("tokens", [])}
  top logits: {[round(x, 4) for x in lens_post.get("logits", [])]}

Latent dynamics context (next-regime probabilities; p = natural, q = after steering intervention):
  p_next: {p.round(6).tolist()}
  q_next: {q.round(6).tolist()}

Return JSON with exactly these keys:
{{
  "steering_toward_target_success": <true or false>,
  "coherence_success": <true or false>,
  "confidence": <0 to 1>,
  "reason": "<one short sentence>"
}}

Definitions:
- steering_toward_target_success: true iff the change from BEFORE to AFTER moves the readout toward patterns associated with the TARGET regime/annotation (and not merely random drift), compared to the SOURCE.
- coherence_success: true iff BEFORE is plausibly consistent with the SOURCE annotation and the AFTER readout does not contradict a shift toward the TARGET (set false if evidence is too ambiguous)."""
        judged = _openai_chat_json_object(
            api_key=api_key,
            model=cfg.judge_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_s=cfg.judge_timeout_s,
            max_retries=cfg.judge_max_retries,
        )
        rows.append(
            {
                "idx": int(idx),
                "source_state": s,
                "target_state": t,
                "source_annotation": src_ann,
                "target_annotation": tgt_ann,
                "centroid_lens_source": sig_s,
                "centroid_lens_target": sig_t,
                "activation_lens_pre": lens_pre,
                "activation_lens_post": lens_post,
                "numeric_policy": {
                    "argmax_p": int(np.argmax(p)),
                    "argmax_q": int(np.argmax(q)),
                    "target_prob_lift": float(q[t] - p[t]),
                    "argmax_q_eq_target": int(np.argmax(q) == t),
                },
                "judge": judged,
            }
        )

    valid = [r for r in rows if isinstance(r.get("judge"), dict) and "error" not in r["judge"]]
    transition_breakdown = compute_transition_breakdown(
        valid, k_regimes=k_reg, edge_counts=edge_counts
    )
    if not valid:
        summary = {
            "count": len(rows),
            "valid_count": 0,
            "judge_steering_toward_target_pct": 0.0,
            "judge_coherence_success_pct": 0.0,
            "mean_confidence": 0.0,
        }
    else:
        steer = 100.0 * float(
            np.mean([1.0 if _judge_steering_success_bool(v["judge"]) else 0.0 for v in valid])
        )
        coh = 100.0 * float(
            np.mean([1.0 if bool(v["judge"].get("coherence_success", False)) else 0.0 for v in valid])
        )
        conf = float(np.mean([float(v["judge"].get("confidence", 0.0)) for v in valid]))
        summary = {
            "count": len(rows),
            "valid_count": len(valid),
            "judge_steering_toward_target_pct": steer,
            "judge_coherence_success_pct": coh,
            "mean_confidence": conf,
        }
    return {
        "enabled": True,
        "model": cfg.judge_model,
        "rows": rows,
        "summary": summary,
        "transition_breakdown": transition_breakdown,
    }


def _fmt_lens_token_for_appendix(tok: str) -> str:
    """Format a single decoded tokenizer string like ``appendix-expms`` centroid-lens tables."""
    raw = str(tok).replace("\r\n", "\n").replace("\r", "\n")
    if raw == "\n" or raw.strip() == "" and "\n" in raw:
        return r"\texttt{\textbackslash n}"
    # Single-line cell: collapse internal newlines (HF often uses ``" \\n"`` as a token).
    t = " ".join(raw.split())
    if not t:
        return r"\texttt{\textbackslash n}"
    return (
        t.replace("\\", "\\textbackslash{}")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("$", "\\$")
    )


def write_regime_lens_raw_centroids_appendix_tex(
    save_dir: str,
    state_signatures: Dict[int, dict],
    k_regimes: int,
    *,
    dataset_name: str = "GSM8K",
    model_col: str = "Reasoning",
    n_rows: int = 0,
    label: str = "tab:regime-lens-moe-qwen14b-gsm8k",
) -> str:
    """
    Appendix Sec.~7.1 style table: Dataset / Model / $k$ / comma-separated top centroid tokens.
    """
    os.makedirs(save_dir, exist_ok=True)
    tex_path = os.path.join(save_dir, "regime_lens_raw_centroids_table.tex")
    keys_sorted = sorted(int(k) if isinstance(k, str) and str(k).isdigit() else int(k) for k in state_signatures.keys())

    n_note = f"{n_rows} training rows" if n_rows else "HF subset"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\scriptsize\n")
        f.write(
            "\\caption{Raw centroid logit-lens top tokens for Qwen-14B (R1-distill) with "
            "\\textbf{CEBRA-MoE} latent regimes on "
            f"\\textbf{{{dataset_name}}} ({n_note}).}}\n"
        )
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{llcp{8.8cm}}\n")
        f.write("\\hline\n")
        f.write("Dataset & Model & $k$ & Top tokens \\\\\n")
        f.write("\\hline\n")
        f.write(f"\\multicolumn{{4}}{{l}}{{\\textbf{{{dataset_name}}}}} \\\\\n")
        f.write("\\hline\n")
        for sk in keys_sorted:
            sig = state_signatures.get(sk) or {}
            toks = sig.get("tokens") or []
            tok_cells = ", ".join(_fmt_lens_token_for_appendix(str(t)) for t in toks)
            f.write(f"{dataset_name} & {model_col} & {int(sk)} & {tok_cells} \\\\\n")
        if not keys_sorted:
            f.write(f"{dataset_name} & {model_col} & — & \\emph{{(no regimes)}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    return tex_path


def write_table_only(save_dir: str, judge_report: dict) -> str:
    """Global judge metrics: post-activation logit-lens evaluation."""
    os.makedirs(save_dir, exist_ok=True)
    tex_path = os.path.join(save_dir, "judge_logitlens_table.tex")
    summ = judge_report.get("summary", {})
    n = int(summ.get("valid_count", summ.get("count", 0)))
    steer = float(
        summ.get(
            "judge_steering_toward_target_pct",
            summ.get("judge_logitlens_annotated_steerability_success_pct", 0.0),
        )
    )
    coh = float(summ.get("judge_coherence_success_pct", 0.0))
    conf = float(summ.get("mean_confidence", 0.0))
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Judge evaluation using centroid annotations and pre/post activation logit-lens readouts.}\n"
        )
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("Metric & $N$ & Value \\\\\n")
        f.write("\\hline\n")
        f.write(f"Steering toward target regime (\\%) & {n} & {steer:.1f} \\\\\n")
        f.write(f"Coherence (\\%) & {n} & {coh:.1f} \\\\\n")
        f.write(f"Mean judge confidence & {n} & {conf:.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    return tex_path


def _tex_escape_cell(s: str, max_len: int = 200) -> str:
    t = (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )
    t = " ".join(t.split())
    if len(t) > max_len:
        t = t[: max_len - 3] + "..."
    return t


def write_state_annotations_tex(save_dir: str, state_annotations: Dict[Any, dict]) -> str:
    """One table: regime id, short label, optional ``aligned_stage``, confidence, one-line description."""
    os.makedirs(save_dir, exist_ok=True)
    tex_path = os.path.join(save_dir, "state_annotations_table.tex")
    keys_sorted = sorted(
        state_annotations.keys(),
        key=lambda k: int(k) if isinstance(k, int) or (isinstance(k, str) and str(k).isdigit()) else str(k),
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(
            "\\caption{Latent regime annotations (centroid logit lens + judge; "
            "\\texttt{aligned\\_stage} matches \\texttt{cebra\\_MoE.STAGES} when set).}\n"
        )
        f.write("\\begin{tabular}{cp{3.6cm}lp{1.1cm}p{4.2cm}}\n")
        f.write("\\hline\n")
        f.write("$k$ & Annotation & \\texttt{aligned\\_stage} & conf. & Description \\\\\n")
        f.write("\\hline\n")
        for k in keys_sorted:
            row = state_annotations[k] or {}
            ann = _tex_escape_cell(str(row.get("state_annotation", "")), 80)
            stg = row.get("aligned_stage")
            stg_tex = _tex_escape_cell(str(stg), 36) if stg else "---"
            conf = float(row.get("confidence", 0.0))
            desc = _tex_escape_cell(str(row.get("state_description", "")), 140)
            kid = int(k) if isinstance(k, int) or (isinstance(k, str) and str(k).isdigit()) else k
            f.write(f"{kid} & {ann} & {stg_tex} & {conf:.2f} & {desc} \\\\\n")
        if not keys_sorted:
            f.write("\\multicolumn{5}{c}{\\emph{(no regimes)}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    return tex_path


def build_regime_annotation_audit(
    state_signatures: Dict[int, dict],
    state_annotations: Dict[Any, dict],
) -> Dict[str, Any]:
    """
    Pair each regime's centroid logit-lens readout (tokens + logits) with the judge's annotation
    for human/JSON verification that labeling matches the evidence shown to the model.
    """
    sig_keys = set()
    for k in state_signatures.keys():
        sig_keys.add(int(k) if isinstance(k, str) and str(k).isdigit() else int(k))
    ann_keys = set()
    for k in state_annotations.keys():
        ann_keys.add(int(k) if isinstance(k, str) and str(k).isdigit() else int(k))
    all_k = sorted(sig_keys | ann_keys)
    out: Dict[str, Any] = {}
    for sk in all_k:
        sig = state_signatures.get(sk) or state_signatures.get(str(sk), {})
        ann = state_annotations.get(sk) or state_annotations.get(str(sk), {})
        out[str(sk)] = {
            "centroid_logit_lens": {
                "token_ids": list(sig.get("token_ids", [])),
                "tokens": list(sig.get("tokens", [])),
                "logits": list(sig.get("logits", [])),
            },
            "judge_annotation": dict(ann) if ann else {},
        }
    return out


def _format_logits_short(logits: List[float], max_k: int = 8) -> str:
    if not logits:
        return ""
    vals = [float(x) for x in logits[:max_k]]
    parts = [f"{v:.2f}" for v in vals]
    if len(logits) > max_k:
        parts.append("…")
    return ", ".join(parts)


def write_state_annotations_with_lens_tex(
    save_dir: str,
    state_signatures: Dict[int, dict],
    state_annotations: Dict[Any, dict],
) -> str:
    r"""Audit table: centroid logit-lens top-$K$ tokens/logits shown to the judge vs. GPT labels."""
    os.makedirs(save_dir, exist_ok=True)
    tex_path = os.path.join(save_dir, "state_annotations_with_lens_table.tex")
    keys_sorted = sorted(
        set(
            int(k) if isinstance(k, str) and str(k).isdigit() else int(k)
            for k in (list(state_signatures.keys()) + list(state_annotations.keys()))
        )
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\footnotesize\n")
        f.write(
            "\\caption{\\textbf{Audit:} centroid logit-lens readout (what the regime-label judge saw) "
            "paired with its annotation. Compare tokens to \\texttt{state\\_annotation} / "
            "\\texttt{aligned\\_stage} for sanity checks.}\n"
        )
        f.write("\\begin{tabular}{cp{4.8cm}p{2.4cm}p{3.0cm}lp{1.0cm}}\n")
        f.write("\\hline\n")
        f.write(
            "$k$ & Centroid top tokens & Top logits & Annotation & \\texttt{stage} & conf. \\\\\n"
        )
        f.write("\\hline\n")
        for sk in keys_sorted:
            sig = state_signatures.get(sk) or state_signatures.get(str(sk), {})
            ann = state_annotations.get(sk) or state_annotations.get(str(sk), {})
            toks = sig.get("tokens") or []
            tok_str = ", ".join(str(t) for t in toks) if toks else "(none)"
            tok_tex = _tex_escape_cell(tok_str, 220)
            log_tex = _tex_escape_cell(_format_logits_short(list(sig.get("logits") or [])), 120)
            ann_tex = _tex_escape_cell(str(ann.get("state_annotation", "")), 64)
            stg = ann.get("aligned_stage")
            stg_tex = _tex_escape_cell(str(stg), 28) if stg else "---"
            conf = float(ann.get("confidence", 0.0))
            f.write(f"{sk} & {tok_tex} & {log_tex} & {ann_tex} & {stg_tex} & {conf:.2f} \\\\\n")
        if not keys_sorted:
            f.write("\\multicolumn{6}{c}{\\emph{(no regimes)}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    return tex_path


def write_transition_breakdown_tex(save_dir: str, judge_report: dict) -> str:
    os.makedirs(save_dir, exist_ok=True)
    tex_path = os.path.join(save_dir, "judge_transition_breakdown_table.tex")
    br = judge_report.get("transition_breakdown") or []
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Per cyclic transition: $N_{\\mathrm{steer}}$ = rows in steering cache; $N_{\\mathrm{judge}}$ = OpenAI judgments (see cap).}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\hline\n")
        f.write(
            "Transition & $N_{\\mathrm{steer}}$ & $N_{\\mathrm{judge}}$ & Steer (\\%) & Coh. (\\%) & Mean conf. \\\\\n"
        )
        f.write("\\hline\n")
        for row in br:
            s = int(row["source_state"])
            t = int(row["target_state"])
            n_st = int(row.get("n_steered", row.get("n", 0)))
            n_j = int(row.get("n_judged", 0))
            if n_j == 0:
                f.write(
                    f"{s}$\\to${t} & {n_st} & 0 & {{\\footnotesize ---}} & {{\\footnotesize ---}} & {{\\footnotesize ---}} \\\\\n"
                )
                continue
            sp = float(row["steer_pct"])
            cp = float(row["coherence_pct"])
            mf = float(row["mean_confidence"])
            f.write(f"{s}$\\to${t} & {n_st} & {n_j} & {sp:.1f} & {cp:.1f} & {mf:.2f} \\\\\n")
        if not br:
            f.write("\\multicolumn{6}{c}{\\emph{(no transition breakdown)}} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    return tex_path


def run_pipeline(cfg: SteerConfig) -> Tuple[str, dict]:
    if not cfg.use_last_token:
        raise ValueError("This version assumes `hidden_state_last`.")
    os.makedirs(cfg.save_dir, exist_ok=True)

    data_path, limit_problems = resolve_hf_subset_data_path(cfg)
    all_features, triplets = load_and_prepare_cebra(
        data_path, mode="temporal", limit_problems=limit_problems, max_triplets=cfg.max_triplets_per_pid
    )
    cebra_seqs, _pca_seqs, _labels, _Z_full, gate_argmax = train_moe_projection(
        all_features,
        triplets,
        K=cfg.k_regimes,
        d_out=cfg.cebra_dim,
        epochs=cfg.moe_epochs,
        seed=42,
    )

    p_map: Dict[int, List[int]] = {}
    for i, f in enumerate(all_features):
        pid = int(f["problem_id"])
        p_map.setdefault(pid, []).append(i)
    pids_sorted = sorted([pid for pid, idxs in p_map.items() if len(idxs) >= 3])
    idx_seqs = [p_map[pid] for pid in pids_sorted]

    latent_dim = cfg.cebra_dim
    state_seqs = [gate_argmax[np.array(idxs, dtype=np.int64)] for idxs in idx_seqs]
    a = hard_transition_matrix(state_seqs, cfg.k_regimes).astype(np.float32)
    d_m, d_b, d_cov = fit_regime_linear_dynamics(cebra_seqs, state_seqs, cfg.k_regimes, latent_dim)
    pi = (np.ones(cfg.k_regimes, dtype=np.float32) / float(cfg.k_regimes)).astype(np.float32)

    n_feat = len(all_features)
    per_sample_state = np.zeros(n_feat, dtype=np.int64)
    z_by_index = np.zeros((n_feat, latent_dim), dtype=np.float32)
    included_mask = np.zeros(n_feat, dtype=bool)
    for idxs, z_seq, s_seq in zip(idx_seqs, cebra_seqs, state_seqs):
        for i_local, i_global in enumerate(idxs):
            included_mask[i_global] = True
            per_sample_state[i_global] = int(s_seq[i_local])
            z_by_index[i_global] = z_seq[i_local]

    x_raw = np.array([f["hidden_state_last"] for f in all_features], dtype=np.float32)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw).astype(np.float32)
    z_flat = np.concatenate([seq for seq in cebra_seqs], axis=0)
    x_flat = np.concatenate([x_scaled[idxs] for idxs in idx_seqs], axis=0)
    w_dec = fit_latent_to_activation_decoder(z_flat, x_flat)

    included_indices = np.nonzero(included_mask)[0].tolist()
    steering_cache: Dict[int, dict] = {}
    for i in included_indices:
        z_t = z_by_index[i]
        s_t = int(per_sample_state[i])
        target_k = (s_t + 1) % cfg.k_regimes
        delta_z, p, q = compute_steering_delta(z_t, s_t, target_k, a, d_m, d_b, cfg.beta)
        delta_x_scaled = delta_z @ w_dec
        delta_x_raw = delta_x_scaled * scaler.scale_
        x_edit = x_raw[i] + cfg.steer_alpha * delta_x_raw
        steering_cache[i] = {
            "source_state": s_t,
            "target_state": target_k,
            "p_next": p,
            "q_next": q,
            "delta_z": delta_z,
            "delta_x_raw": delta_x_raw,
            "x_steered": x_edit,
        }

    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for ent in steering_cache.values():
        edge_counts[(int(ent["source_state"]), int(ent["target_state"]))] += 1

    tokenizer, w_u, lens_device = _load_logit_lens(cfg)
    state_signatures = build_state_signatures(
        cfg, x_scaled, per_sample_state, included_indices, tokenizer, w_u, lens_device
    )
    state_annotations = judge_state_annotations(cfg, state_signatures)
    lens_rows = run_logit_lens_rows(
        steering_cache, x_scaled, scaler, state_signatures, w_u, lens_device
    )
    lens_numeric_summary = summarize_logit_lens_rows(lens_rows)

    judge_report = judge_steering_post_activation_lens(
        cfg=cfg,
        steering_cache=steering_cache,
        state_annotations=state_annotations,
        state_signatures=state_signatures,
        x_raw=x_raw,
        scaler=scaler,
        tokenizer=tokenizer,
        w_u=w_u,
        edge_counts=dict(edge_counts),
        lens_device=lens_device,
    )
    global_tex = write_table_only(cfg.save_dir, judge_report)
    transition_tex = write_transition_breakdown_tex(cfg.save_dir, judge_report)
    annotations_tex = write_state_annotations_tex(cfg.save_dir, state_annotations)
    annotations_lens_tex = write_state_annotations_with_lens_tex(
        cfg.save_dir, state_signatures, state_annotations
    )
    appendix_raw_lens_tex = write_regime_lens_raw_centroids_appendix_tex(
        cfg.save_dir,
        state_signatures,
        cfg.k_regimes,
        dataset_name="GSM8K",
        model_col="Reasoning (MoE)",
        n_rows=len(all_features),
        label="tab:regime-lens-moe-qwen14b-gsm8k-n500",
    )
    regime_annotation_audit = build_regime_annotation_audit(state_signatures, state_annotations)
    with open(os.path.join(cfg.save_dir, "judge_logitlens_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "state_annotations": state_annotations,
                "state_signatures": state_signatures,
                "regime_annotation_audit": regime_annotation_audit,
                "judge_report": judge_report,
                "diagnostic_centroid_margin_lens_rows_count": len(lens_rows),
                "diagnostic_lens_numeric_summary": lens_numeric_summary,
                "lens_geometry": "causal_aligned_lm_head_plus_scaled_activations",
                "note_steering_rows_include_lens": (
                    "Each judge_report.rows[] entry includes activation_lens_pre, activation_lens_post, "
                    "centroid_lens_source, centroid_lens_target (full logit-lens top-k for that evaluation). "
                    "Lens uses lm_head and StandardScaler like cebra_causal.py."
                ),
            },
            f,
            indent=2,
        )
    tex_path = global_tex
    payload = {
        "config": cfg.__dict__,
        "pi": pi,
        "A": a,
        "dM": d_m,
        "db": d_b,
        "dCov": d_cov,
        "per_sample_state": per_sample_state,
        "included_indices": included_indices,
        "decoder_W": w_dec,
        "state_signatures": state_signatures,
        "state_annotations": state_annotations,
        "judge_report": judge_report,
        "table_path": tex_path,
        "transition_breakdown_table_path": transition_tex,
        "state_annotations_table_path": annotations_tex,
        "state_annotations_with_lens_table_path": annotations_lens_tex,
        "regime_lens_raw_centroids_appendix_table_path": appendix_raw_lens_tex,
        "regime_annotation_audit": regime_annotation_audit,
    }
    out_path = os.path.join(cfg.save_dir, "steer_cebra_moe_logitlens_judge_causal_lens_artifacts.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    return out_path, payload


if __name__ == "__main__":  # pragma: no cover
    np.random.seed(42)
    parser = build_cli_parser()
    args = parser.parse_args()
    cfg = config_from_cli_args(SteerConfig(), args)
    out_file, payload = run_pipeline(cfg)
    print(f"Saved artifacts: {out_file}")
    print(f"LaTeX table: {payload.get('table_path')}")
    print(f"LaTeX transition breakdown: {payload.get('transition_breakdown_table_path')}")
    print(f"LaTeX state annotations: {payload.get('state_annotations_table_path')}")
    print(f"LaTeX state annotations + lens audit: {payload.get('state_annotations_with_lens_table_path')}")
    print(f"LaTeX appendix-style raw centroid lens table: {payload.get('regime_lens_raw_centroids_appendix_table_path')}")
    print(f"Logit-lens device: {resolve_logit_lens_device(cfg.logit_lens_device)}")
