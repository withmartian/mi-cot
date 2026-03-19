"""
CEBRA-EM steering that imports CEBRA+EM utilities from `cebra_EM.py`.
CEBRA embedding training and SLDS EM: init / E-step / M-step

This file steers the CEBRA-EM model by:
  - KL-regularized target-state policy q*(k) ∝ p(k) exp(β r(k))
  - steering delta Δz_t = μ_steered(z_t) - μ_orig(z_t)

If `SteerConfig.data_path` does not exist locally and `hf_auto_download_if_missing` is True,
`run_pipeline` downloads `hf_fallback_num_samples` (default 100) non-NEUTRAL rows from
https://huggingface.co/datasets/withmartian/SDS_train_gsm8k

Hub layout, HF-model↔dataset-path presets, and ``--model`` aliases are defined in
``sds_train_gsm8k_hf.py`` (same source as ``batch_cebra_steering_hf_dataset.py``).
"""

from __future__ import annotations

import argparse
import os
import pickle
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


# Make sure local imports work when running from repo root.
_here = os.path.abspath(os.path.dirname(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from sds_train_gsm8k_hf import (
    DEFAULT_STEERING_HF_FEATURES_RELPATH,
    SDS_TRAIN_GSM8K_REPO_ID,
    normalize_hub_features_filename,
)

try:
    import cebra_EM as cebra_mod
    from cebra_EM import load_and_prepare_cebra, train_cebra_projection
    from cebra_EM import init_params as em_init_params
    from cebra_EM import forward_backward as em_forward_backward
    from cebra_EM import m_step as em_m_step
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import `cebra_EM.py`. If this is due to PyTorch load issues, "
        "use `steer_rpc_cebra_em.py` which contains a PCA fallback."
    ) from e


# ---------------------------------------------------------------------------
# Judge taxonomies (SDS_train_gsm8k GSM8K stage labels + SDS early-results PDF)
#
# Reasoning *behaviors* — the judge must choose `behavior_stage_judged` from this list,
# passed as `stage_list` (= ``cebra_mod.STAGES``). These are the same string labels stored
# in dataset field ``stage`` (human-annotated reasoning stages). They align with the
# contingency discussion in "Switching Dynamical System Framework – early results"
# (e.g. Plan Generation ↔ PLAN_GENERATION, Problem Setup ↔ PROBLEM_SETUP,
# Self-Checking ↔ SELF_CHECKING, Fact Retrieval ↔ FACT_RETRIEVAL,
# Active Computation ↔ ACTIVE_COMPUTATION, Consolidation ↔ RESULT_CONSOLIDATION;
# plus UNCERTAINTY_MANAGEMENT, FINAL_ANSWER_EMISSION in the GSM8K annotation schema).
#
# Latent *states* — integers 0..K-1 from SLDS-EM (default K=4, as in the PDF's K=4 sweep).
# The PDF names four functional regimes; **EM indices are not guaranteed to match**
# the paper’s regime numbering without alignment — use dominant human stage per regime
# from the run (`state_behavior_profiles`) when interpreting s_t and k*.
# ---------------------------------------------------------------------------
_SDS_EARLY_RESULTS_REGIME_STAGE_SPECIALIZATION = """
Human stage specialization by latent regime (from SDS early-results PDF, "Distribution of Reasoning Policies"; regime IDs are the paper’s, not necessarily equal to EM indices in this code):
- Regime **core computation**: dominated by ACTIVE_COMPUTATION and RESULT_CONSOLIDATION ("Consolidation" in the PDF).
- Regime **transitional manifold**: mixes many stages; hand-off between setup, compute, and check.
- Regime **structural verification**: SELF_CHECKING (high), FACT_RETRIEVAL.
- Regime **contextual framing & strategy**: PLAN_GENERATION, PROBLEM_SETUP.

EM assigns arbitrary labels 0..K-1 to regimes; map semantics using empirical dominant_stage per regime in the payload, not by equating EM id to a PDF regime number.
""".strip()


@dataclass
class SteerConfig:
    data_path: str = "rpc_dataset_layer28_200/all_sentences_features.pkl"
    use_last_token: bool = True  # for compatibility with the other script

    limit_problems: int = 500
    max_triplets_per_pid: int = 25

    # latent dimension used for CEBRA embedding + EM dynamics
    cebra_dim: int = 40
    cebra_epochs: int = 100  # forwarded only if you modify cebra_EM; kept for interface parity

    # SLDS EM iterations
    em_iters: int = 50
    k_regimes: int = 4
    transition_kappa: float = 1.0  # maps to cebra_EM.KAPPA

    # steering
    beta: float = 8.0  # strength in q*(k) ∝ p(k) exp(β r(k))
    steer_alpha: float = 8.0  # scaling on Δx in activation space (decoder delta)

    # LLM-as-judge (behavior + discrete latent state steering)
    # Set OPENAI_API_KEY in the environment
    enable_openai_judge: bool = True
    judge_model: str = "gpt-4.1-mini"
    judge_max_samples: int = 100
    judge_timeout_s: int = 60
    judge_max_retries: int = 3

    save_dir: str = "cebra_em_steering"

    # If `data_path` is not a file, optionally download a small subset from Hugging Face
    # (see ``sds_train_gsm8k_hf.SDS_TRAIN_GSM8K_REPO_ID``). HF_TOKEN optional for public files.
    hf_auto_download_if_missing: bool = True
    hf_dataset_repo: str = SDS_TRAIN_GSM8K_REPO_ID
    hf_dataset_filename: str = DEFAULT_STEERING_HF_FEATURES_RELPATH
    hf_fallback_num_samples: int = 100
    hf_fallback_cache_name: str = "sds_hf100_default_qwen15b_reasoning_l27.pkl"


def parse_hf_dataset_repo(value: str) -> str:
    """
    Accepts:
      - org/name
      - https://huggingface.co/datasets/org/name
      - https://huggingface.co/datasets/org/name/tree/main
      - hf://org/name
    """
    v = value.strip().rstrip("/")
    if not v:
        raise ValueError("Empty --dataset value.")
    if v.lower().startswith("hf://"):
        return v[5:].strip().strip("/")
    if "huggingface.co/datasets/" in v:
        after = v.split("huggingface.co/datasets/", 1)[1]
        after = after.split("/tree/")[0].split("/blob/")[0].split("?")[0]
        return after.strip("/")
    return v


def _is_probably_hf_dataset_ref(s: str) -> bool:
    """True for Hub repo id, hf:// URL, or huggingface.co/datasets/... URL (not a local path)."""
    t = s.strip()
    if not t:
        return False
    if t.lower().startswith("hf://"):
        return True
    if "huggingface.co/datasets/" in t:
        return True
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", t):
        return True
    return False


def resolve_cli_data_path(dataset_arg: str) -> str:
    """
    Resolve --dataset when treated as local: file, or directory containing
    all_sentences_features.pkl; else abspath of the argument (may not exist yet).
    """
    raw = dataset_arg.strip()
    p = os.path.abspath(os.path.expanduser(raw))
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        cand = os.path.join(p, "all_sentences_features.pkl")
        if os.path.isfile(cand):
            return cand
    return p


def _safe_cache_slug(model_path: str, max_len: int = 72) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_path.strip().strip("/"))
    return s[:max_len] if s else "default"


def build_cli_parser() -> argparse.ArgumentParser:
    _epilog = (
        "Examples:\n"
        "  HF, 500 sentences, default dataset (see sds_train_gsm8k_hf.py), no API judge:\n"
        "  python cebra_em_steering_inputDep.py --num-samples 500 --no-openai-judge --save-dir out_run\n"
        "  By Hub subpath:\n"
        "  python cebra_em_steering_inputDep.py --model qwen1.5b_reasoning/layer_27 --num-samples 500\n"
        "  By Hugging Face model id preset (same mapping as batch_cebra_steering_hf_dataset.py):\n"
        "  python cebra_em_steering_inputDep.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\n"
        "  python cebra_em_steering_inputDep.py --model Qwen/Qwen2.5-1.5B\n"
        "  Gated / private Hub files need: export HF_TOKEN=\"...\"\n"
        "  Local pickle:\n"
        "  python cebra_em_steering_inputDep.py --local-data path/to/all_sentences_features.pkl\n"
    )
    p = argparse.ArgumentParser(
        description=(
            "CEBRA-EM steering (SLDS + KL-regularized policy). "
            "Uses local data_path when the file exists; otherwise downloads from Hugging Face "
            "if hf_auto_download_if_missing (see --no-hf-fallback). "
            "HF token: environment variable HF_TOKEN only."
        ),
        epilog=_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--local-data",
        type=str,
        default=None,
        metavar="PATH",
        help="Local features .pkl (e.g. all_sentences_features.pkl). If it exists, used directly.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="REPO_OR_URL_OR_PATH",
        help=(
            "Hugging Face dataset repo id (org/name), full datasets URL, hf://org/name, "
            "or a local .pkl path / directory containing all_sentences_features.pkl."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="SUBPATH",
        help=(
            "Path inside the Hub dataset, e.g. qwen1.5b_reasoning/layer_27, or a preset from "
            "sds_train_gsm8k_hf (HF model id like deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, "
            "Qwen/Qwen2.5-1.5B, shorthands qwen-1.5b-base, qwen2.5-1.5b-instruct, …)."
        ),
    )
    p.add_argument(
        "--hf-samples",
        "--num-samples",
        type=int,
        default=None,
        dest="hf_samples",
        metavar="N",
        help="When downloading from Hub, keep this many non-NEUTRAL rows (alias: --num-samples).",
    )
    p.add_argument(
        "--no-openai-judge",
        action="store_true",
        help="Skip OpenAI behavior/state judge (faster; no OPENAI_API_KEY needed).",
    )
    p.add_argument(
        "--no-hf-fallback",
        action="store_true",
        help="If local data_path is missing, fail instead of downloading from Hugging Face.",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output directory for artifacts and HF cache file.",
    )
    p.add_argument(
        "--limit-problems",
        type=int,
        default=None,
        help="Passed to load_and_prepare_cebra when using an existing local data file (not auto-set for HF subset).",
    )
    return p


def config_from_cli_args(cfg: SteerConfig, args: argparse.Namespace) -> SteerConfig:
    """Apply CLI flags onto a SteerConfig (returns a new instance)."""
    c = cfg

    if args.no_hf_fallback:
        c = replace(c, hf_auto_download_if_missing=False)

    if getattr(args, "no_openai_judge", False):
        c = replace(c, enable_openai_judge=False)

    if args.save_dir is not None:
        c = replace(c, save_dir=args.save_dir)

    if args.hf_samples is not None:
        c = replace(c, hf_fallback_num_samples=int(args.hf_samples))

    # Data source: --local-data wins; else --dataset is Hub id/URL or local path
    if args.local_data:
        c = replace(c, data_path=os.path.abspath(os.path.expanduser(args.local_data.strip())))
    elif args.dataset:
        if _is_probably_hf_dataset_ref(args.dataset):
            c = replace(c, hf_dataset_repo=parse_hf_dataset_repo(args.dataset))
        else:
            c = replace(c, data_path=resolve_cli_data_path(args.dataset))

    if args.model:
        rel = normalize_hub_features_filename(args.model)
        c = replace(c, hf_dataset_filename=rel)
        slug = _safe_cache_slug(rel.replace(".pkl", ""))
        c = replace(
            c,
            hf_fallback_cache_name=f"sds_hf{c.hf_fallback_num_samples}_{slug}.pkl",
        )

    if args.limit_problems is not None:
        c = replace(c, limit_problems=int(args.limit_problems))

    return c


def resolve_steering_data_path(cfg: SteerConfig) -> Tuple[str, int]:
    """
    Return (absolute path to features pickle, limit_problems for load_and_prepare_cebra).

    If cfg.data_path exists, use it with cfg.limit_problems.
    Otherwise, when hf_auto_download_if_missing, download from HF (HF_TOKEN optional for public datasets),
    take the first hf_fallback_num_samples non-NEUTRAL rows, cache under save_dir, and
    set limit_problems so load_and_prepare_cebra does not drop the subset.
    """
    path = os.path.abspath(os.path.expanduser(cfg.data_path))
    if os.path.isfile(path):
        return path, cfg.limit_problems

    if not cfg.hf_auto_download_if_missing:
        raise FileNotFoundError(
            f"Steering data not found: {path!r}. "
            "Point data_path to an existing pickle, or set SteerConfig(hf_auto_download_if_missing=True)."
        )

    token = os.environ.get("HF_TOKEN", "").strip() or None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for HF fallback. Install with: pip install huggingface_hub"
        ) from e

    os.makedirs(cfg.save_dir, exist_ok=True)
    print(
        f"[data] Local file not found ({path!r}). Downloading from Hugging Face "
        f"{cfg.hf_dataset_repo} / {cfg.hf_dataset_filename!r} …",
        flush=True,
    )
    try:
        remote = hf_hub_download(
            repo_id=cfg.hf_dataset_repo,
            filename=cfg.hf_dataset_filename,
            token=token,
            repo_type="dataset",
        )
    except Exception as e:
        hint = (
            " If the repo is gated, set HF_TOKEN (env var only)."
            if token is None
            else ""
        )
        raise FileNotFoundError(
            f"Could not download {cfg.hf_dataset_repo!r} / {cfg.hf_dataset_filename!r}:{hint}"
        ) from e
    with open(remote, "rb") as f:
        features: List[dict] = pickle.load(f)

    features = [feat for feat in features if feat.get("stage", "NEUTRAL") != "NEUTRAL"]
    n = cfg.hf_fallback_num_samples
    if len(features) < n:
        raise ValueError(
            f"Only {len(features)} non-NEUTRAL samples after filter; need >= {n}. "
            "Try another hf_dataset_filename or smaller hf_fallback_num_samples."
        )
    subset = features[:n]
    limit_problems = max(int(f["problem_id"]) for f in subset) + 1

    cache_path = os.path.join(cfg.save_dir, cfg.hf_fallback_cache_name)
    with open(cache_path, "wb") as f:
        pickle.dump(subset, f)

    print(
        f"[data] Cached {len(subset)} samples to {cache_path} (limit_problems={limit_problems}).",
        flush=True,
    )
    return os.path.abspath(cache_path), limit_problems


def fit_latent_to_activation_decoder(z_flat: np.ndarray, x_scaled_flat: np.ndarray) -> np.ndarray:
    """
    Linear map from z to x_scaled: x_scaled ≈ [z, 1] @ coef
    We return only the weights w such that Δx_scaled ≈ Δz @ w.
    """
    z_aug = np.hstack([z_flat, np.ones((len(z_flat), 1), dtype=z_flat.dtype)])
    coef, *_ = np.linalg.lstsq(z_aug, x_scaled_flat, rcond=None)
    w = coef[:-1]  # bias cancels when we compute deltas
    return w


def kl_regularized_policy(p: np.ndarray, target_k: int, beta: float) -> np.ndarray:
    """
    q*(k) ∝ p(k) exp(β r(k)), where r(k)=1 iff k=target_k else 0.
    """
    p = p / (np.sum(p) + 1e-12)
    g = np.zeros_like(p)
    g[target_k] = 1.0
    q_unnorm = p * np.exp(beta * g)
    return q_unnorm / np.sum(q_unnorm)


def compute_statewise_next_means(z_t: np.ndarray, d_m: np.ndarray, d_b: np.ndarray) -> np.ndarray:
    """
    f_k(z_t) = A_k z_t + b_k
    In code: d_m[k] is A_k, d_b[k] is b_k.
    Returns array with shape (K, latent_dim).
    """
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
    """
      p(k)  = P(s_{t+1}=k | s_t, z_t)  (we approximate with row a[s_t])
      q*(k) ∝ p(k) exp(β r(k))
      μ_orig(z_t)    = Σ_k p(k) f_k(z_t)
      μ_steered(z_t) = Σ_k q*(k) f_k(z_t)
      Δz_t = μ_steered - μ_orig
    """
    p = a[s_t].copy()
    p = p / (np.sum(p) + 1e-12)
    q = kl_regularized_policy(p, target_k, beta)

    f_k = compute_statewise_next_means(z_t, d_m, d_b)  # (K, latent_dim)
    mu_orig = np.sum(p[:, None] * f_k, axis=0)
    mu_steered = np.sum(q[:, None] * f_k, axis=0)
    delta_z = mu_steered - mu_orig
    return delta_z, p, q


def diffusion_postprocess_placeholder(acts_edit: np.ndarray) -> np.ndarray:
    """
    Paper placeholder (GLP / diffusion Fig 4 style):
    Replace with trained denoiser post-processing to project back onto the manifold.
    """
    return acts_edit


def compute_steering_reports(
    steering_cache: Dict[int, dict],
    x_raw: np.ndarray,
) -> dict:
    """
    Compute steering summaries analogous to the steering-result tables:
      - by target regime (steerability, coherence penalty, etc.)
      - by source->target transfer (success rates)

    Metric definitions used here:
      steerability(%): 100 * mean[argmax(q*) == target_k]
      coherence_penalty: mean(||delta_x_raw|| / ||x_raw||)
      target_prob_lift: mean(q*[target] - p[target])
      kl_q_p: mean(KL(q* || p))
    """
    if not steering_cache:
        return {
            "global": {},
            "by_target": {},
            "by_transfer": {},
        }

    eps = 1e-12
    rows = []
    for idx, entry in steering_cache.items():
        s = int(entry["source_state"])
        t = int(entry["target_state"])
        p = np.asarray(entry["p_next"], dtype=np.float64)
        q = np.asarray(entry["q_next"], dtype=np.float64)
        dx = np.asarray(entry["delta_x_raw"], dtype=np.float64)
        x = np.asarray(x_raw[idx], dtype=np.float64)

        lift = float(q[t] - p[t])
        kl = float(np.sum(q * (np.log(q + eps) - np.log(p + eps))))
        coh_pen = float(np.linalg.norm(dx) / (np.linalg.norm(x) + eps))
        success = int(np.argmax(q) == t)
        l1_shift = float(np.sum(np.abs(q - p)))

        rows.append(
            {
                "idx": int(idx),
                "source_state": s,
                "target_state": t,
                "lift": lift,
                "kl_q_p": kl,
                "coherence_penalty": coh_pen,
                "success": success,
                "l1_policy_shift": l1_shift,
            }
        )

    def _agg(items: List[dict]) -> dict:
        arr_lift = np.array([r["lift"] for r in items], dtype=np.float64)
        arr_kl = np.array([r["kl_q_p"] for r in items], dtype=np.float64)
        arr_cp = np.array([r["coherence_penalty"] for r in items], dtype=np.float64)
        arr_success = np.array([r["success"] for r in items], dtype=np.float64)
        arr_shift = np.array([r["l1_policy_shift"] for r in items], dtype=np.float64)
        return {
            "count": int(len(items)),
            "steerability_pct": float(100.0 * arr_success.mean()),
            "coherence_penalty": float(arr_cp.mean()),
            "target_prob_lift": float(arr_lift.mean()),
            "kl_q_p": float(arr_kl.mean()),
            "l1_policy_shift": float(arr_shift.mean()),
        }

    global_stats = _agg(rows)

    by_target: Dict[int, dict] = {}
    for t in sorted({r["target_state"] for r in rows}):
        group = [r for r in rows if r["target_state"] == t]
        by_target[int(t)] = _agg(group)

    by_transfer: Dict[str, dict] = {}
    pairs = sorted({(r["source_state"], r["target_state"]) for r in rows})
    for s, t in pairs:
        group = [r for r in rows if r["source_state"] == s and r["target_state"] == t]
        by_transfer[f"{s}->{t}"] = _agg(group)

    return {
        "global": global_stats,
        "by_target": by_target,
        "by_transfer": by_transfer,
        "rows": rows,
    }


def write_steering_report(
    save_dir: str,
    report: dict,
    cfg: SteerConfig,
) -> None:
    """
    Write results in clearly separated sections:
      1) Inputs
      2) Outputs
      3) What they mean
      4) How to analyze
    """
    os.makedirs(save_dir, exist_ok=True)

    # JSON (machine-readable)
    with open(os.path.join(save_dir, "steering_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Text report (human-readable)
    txt_path = os.path.join(save_dir, "steering_eval_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Steering Evaluation Report\n")
        f.write("=" * 80 + "\n\n")

        f.write("SECTION 1: INPUTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"- data_path: {cfg.data_path}\n")
        f.write(f"- k_regimes: {cfg.k_regimes}\n")
        f.write(f"- beta: {cfg.beta}\n")
        f.write(f"- steer_alpha: {cfg.steer_alpha}\n")
        f.write("- steering source: per-sample entries in steering_cache\n\n")

        f.write("SECTION 2: OUTPUTS\n")
        f.write("-" * 80 + "\n")
        g = report["global"]
        f.write(f"- global.count: {g.get('count', 0)}\n")
        f.write(f"- global.steerability_pct: {g.get('steerability_pct', 0.0):.4f}\n")
        f.write(f"- global.coherence_penalty: {g.get('coherence_penalty', 0.0):.6f}\n")
        f.write(f"- global.target_prob_lift: {g.get('target_prob_lift', 0.0):.6f}\n")
        f.write(f"- global.kl_q_p: {g.get('kl_q_p', 0.0):.6f}\n")
        f.write(f"- global.l1_policy_shift: {g.get('l1_policy_shift', 0.0):.6f}\n\n")

        f.write("By target regime:\n")
        for t, stats in report["by_target"].items():
            f.write(
                f"  - target={t} | count={stats['count']} | steerability_pct={stats['steerability_pct']:.4f} | "
                f"coherence_penalty={stats['coherence_penalty']:.6f} | target_prob_lift={stats['target_prob_lift']:.6f} | "
                f"kl_q_p={stats['kl_q_p']:.6f}\n"
            )
        f.write("\n")

        f.write("By source->target transfer:\n")
        for key, stats in report["by_transfer"].items():
            f.write(
                f"  - {key} | count={stats['count']} | steerability_pct={stats['steerability_pct']:.4f} | "
                f"coherence_penalty={stats['coherence_penalty']:.6f} | target_prob_lift={stats['target_prob_lift']:.6f}\n"
            )
        f.write("\n")

        f.write("SECTION 3: WHAT THEY MEAN\n")
        f.write("-" * 80 + "\n")
        f.write("- steerability_pct: how often the steered policy q* puts highest mass on the target regime.\n")
        f.write("- coherence_penalty: average relative activation perturbation size ||delta_x||/||x||.\n")
        f.write("- target_prob_lift: average increase in target regime probability from p to q*.\n")
        f.write("- kl_q_p: control cost; larger means stronger deviation from natural dynamics.\n")
        f.write("- l1_policy_shift: total distribution shift between p and q*.\n\n")

        f.write("SECTION 4: HOW TO ANALYZE\n")
        f.write("-" * 80 + "\n")
        f.write("- Good steering usually means higher steerability_pct and target_prob_lift.\n")
        f.write("- Keep coherence_penalty bounded to avoid overly large activation edits.\n")
        f.write("- Compare by_target and by_transfer to identify easy vs hard steering directions.\n")
        f.write("- If steerability is low with tiny kl_q_p and tiny shift, increase beta or improve target selection.\n")
        f.write("- If coherence_penalty is high, reduce steer_alpha or add denoising postprocess.\n")


def build_state_behavior_profiles(
    all_features: List[dict],
    per_sample_state: np.ndarray,
    included_indices: List[int],
    k_regimes: int,
    stage_list: List[str],
) -> Dict[int, dict]:
    """
    Per latent regime k: empirical distribution of human `stage` labels (reasoning behavior types).
    Used as priors so the judge can say whether behavior aligns with the *target* regime.
    """
    profiles: Dict[int, dict] = {}
    for s in range(k_regimes):
        profiles[s] = {
            "counts": {k: 0 for k in stage_list},
            "distribution": {k: 0.0 for k in stage_list},
            "dominant_stage": "UNKNOWN",
            "n": 0,
        }

    for idx in included_indices:
        st = int(per_sample_state[idx])
        stage = all_features[idx].get("stage", "UNKNOWN")
        if stage not in profiles[st]["counts"]:
            continue
        profiles[st]["counts"][stage] += 1
        profiles[st]["n"] += 1

    for s in range(k_regimes):
        n = max(1, profiles[s]["n"])
        for stg in stage_list:
            profiles[s]["distribution"][stg] = profiles[s]["counts"][stg] / n
        if profiles[s]["n"] > 0:
            profiles[s]["dominant_stage"] = max(stage_list, key=lambda stg: profiles[s]["counts"][stg])
        else:
            profiles[s]["dominant_stage"] = "UNKNOWN"
    return profiles


def _openai_chat_json_object(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Call Chat Completions with response_format=json_object; return parsed JSON or {error: ...}."""
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


def run_behavior_state_judge(
    cfg: SteerConfig,
    all_features: List[dict],
    steering_cache: Dict[int, dict],
    state_profiles: Dict[int, dict],
    stage_list: List[str],
) -> dict:
    """
    Behavior-level steerability: does the *reasoning behavior* (stage type) in the text match
    what we expect when steering toward the target latent regime?

    State-level steerability (judge): given p_next -> q_next from the KL-regularized policy,
    is the discrete next-state distribution successfully shifted toward the target regime id?

    Also records numeric policy metrics per row for analysis vs. the judge.
    """
    if not cfg.enable_openai_judge:
        return {"enabled": False, "reason": "judge_disabled", "rows": [], "summary": {}}

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "enabled": False,
            "reason": "missing_OPENAI_API_KEY (set env var; do not hardcode)",
            "rows": [],
            "summary": {},
        }

    system_prompt = (
        "You evaluate steering experiments on chain-of-thought sentences.\n"
        "Reasoning behavior must be one of the stage labels listed in the user message "
        "(human-annotated reasoning stages; same vocabulary as the SDS GSM8K dataset and "
        "the SDS early-results paper’s regime–stage analysis).\n"
        "Latent regimes s_t and k* are integer ids 0..K-1 from a learned switching dynamic "
        "model (default K=4). Use the paper-style regime–stage specialization text only as "
        "qualitative context; EM indices may be permuted relative to the PDF’s regime numbering.\n"
        "Return strict JSON only."
    )

    judge_rows: List[dict] = []
    ordered_ids = sorted(steering_cache.keys())[: cfg.judge_max_samples]
    eps = 1e-12

    for idx in ordered_ids:
        feat = all_features[int(idx)]
        row = steering_cache[int(idx)]
        s = int(row["source_state"])
        t = int(row["target_state"])
        p = np.asarray(row["p_next"], dtype=np.float64)
        q = np.asarray(row["q_next"], dtype=np.float64)
        p = p / (np.sum(p) + eps)
        q = q / (np.sum(q) + eps)

        source_stage_prior = state_profiles.get(s, {}).get("dominant_stage", "UNKNOWN")
        target_stage_prior = state_profiles.get(t, {}).get("dominant_stage", "UNKNOWN")
        text = str(feat.get("text", feat.get("sentence", ""))).strip()
        if not text:
            text = f"[no text field; sample index {idx}]"
        text_for_judge = text if len(text) <= 2500 else text[:2500] + "…"
        label = feat.get("stage", "UNKNOWN")

        num_argmax_p = int(np.argmax(p))
        num_argmax_q = int(np.argmax(q))
        num_lift = float(q[t] - p[t])
        num_top1_success = int(num_argmax_q == t)

        user_prompt = f"""Evaluate steering success for BOTH reasoning behavior and latent state.

Allowed reasoning behavior (stage) labels — use EXACTLY these strings for behavior_stage_judged (SDS_train_gsm8k / GSM8K human annotations; PDF: human-annotated reasoning stages):
{stage_list}

Qualitative regime–stage specialization (SDS early-results PDF; for intuition only, indices s_t/k* are EM ids):
{_SDS_EARLY_RESULTS_REGIME_STAGE_SPECIALIZATION}

Input:
- Sentence / CoT fragment: {text_for_judge!r}
- Dataset stage label for this token/sentence: {label!r}
- Source latent regime id s_t: {s}
- Target latent regime id k* (steering goal): {t}
- Empirical dominant stage among samples assigned to source regime {s}: {source_stage_prior!r}
- Empirical dominant stage among samples assigned to target regime {t}: {target_stage_prior!r}
- p_next (natural next-regime distribution P(s_{{t+1}}|s_t) as row of A): {p.round(6).tolist()}
- q_next (KL-regularized steered distribution q*): {q.round(6).tolist()}

Return a JSON object with exactly these keys:
{{
  "behavior_stage_judged": "<one string from the allowed list, or UNKNOWN if unclear>",
  "behavior_target_success": <true or false>,
  "state_target_success": <true or false>,
  "overall_success": <true or false>,
  "confidence": <number from 0 to 1>,
  "reason": "<one short sentence>"
}}

Definitions:
- behavior_target_success: true iff the sentence's *reasoning behavior* is substantially aligned with the kind of reasoning associated with target regime {t} (use target_stage_prior as a strong hint of what that regime "means" behaviorally).
- state_target_success: true iff q_next represents a successful shift *toward* regime {t} compared to p_next — e.g. q_next[{t}] > p_next[{t}], or argmax q_next == {t}, or clearly increased mass on {t} without contradiction.
- overall_success: true iff both behavior_target_success and state_target_success are true.
"""
        judged = _openai_chat_json_object(
            api_key=api_key,
            model=cfg.judge_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_s=cfg.judge_timeout_s,
            max_retries=cfg.judge_max_retries,
        )

        judge_rows.append(
            {
                "idx": int(idx),
                "source_state": s,
                "target_state": t,
                "annotated_stage": label,
                "source_stage_prior": source_stage_prior,
                "target_stage_prior": target_stage_prior,
                "numeric_policy": {
                    "argmax_p": num_argmax_p,
                    "argmax_q": num_argmax_q,
                    "target_prob_lift": num_lift,
                    "argmax_q_eq_target": num_top1_success,
                },
                "judge": judged,
            }
        )

    valid = [r for r in judge_rows if isinstance(r.get("judge"), dict) and "error" not in r["judge"]]
    if not valid:
        return {
            "enabled": True,
            "model": cfg.judge_model,
            "rows": judge_rows,
            "summary": {
                "count": len(judge_rows),
                "valid_count": 0,
                "behavior_steerability_success_pct": 0.0,
                "state_steerability_success_pct": 0.0,
                "overall_success_pct": 0.0,
                "mean_confidence": 0.0,
            },
        }

    def _rate(key: str) -> float:
        return float(
            np.mean([1.0 if bool(v["judge"].get(key, False)) else 0.0 for v in valid])
        )

    b = _rate("behavior_target_success")
    st = _rate("state_target_success")
    ov = _rate("overall_success")
    conf = float(
        np.mean([float(v["judge"].get("confidence", 0.0)) for v in valid])
    )

    return {
        "enabled": True,
        "model": cfg.judge_model,
        "rows": judge_rows,
        "summary": {
            "count": len(judge_rows),
            "valid_count": len(valid),
            "behavior_steerability_success_pct": 100.0 * b,
            "state_steerability_success_pct": 100.0 * st,
            "overall_success_pct": 100.0 * ov,
            "mean_confidence": conf,
        },
    }


def write_judge_report(save_dir: str, judge_report: dict, cfg: SteerConfig) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "steering_judge_summary.json"), "w", encoding="utf-8") as f:
        json.dump(judge_report, f, indent=2)

    txt_path = os.path.join(save_dir, "steering_judge_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Steering Judge Report (behavior + latent state)\n")
        f.write("=" * 80 + "\n\n")

        f.write("SECTION 1: INPUTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"- enable_openai_judge: {cfg.enable_openai_judge}\n")
        f.write(f"- judge_model: {cfg.judge_model}\n")
        f.write(f"- judge_max_samples: {cfg.judge_max_samples}\n")
        f.write(f"- reasoning stage types (from cebra_EM.STAGES): {cebra_mod.STAGES}\n")
        f.write("- API key: from environment variable OPENAI_API_KEY only\n\n")

        f.write("SECTION 2: OUTPUTS\n")
        f.write("-" * 80 + "\n")
        summ = judge_report.get("summary") or {}
        for k, v in summ.items():
            f.write(f"- {k}: {v}\n")
        if not judge_report.get("enabled"):
            f.write(f"- note: {judge_report.get('reason', '')}\n")
        f.write("\n")

        f.write("SECTION 3: WHAT THEY MEAN\n")
        f.write("-" * 80 + "\n")
        f.write(
            "- behavior_steerability_success_pct: fraction of judged samples where the model says "
            "the text's reasoning behavior matches the target regime's typical behavior (see target_stage_prior).\n"
        )
        f.write(
            "- state_steerability_success_pct: fraction where the judge says p->q reflects successful "
            "steering toward target regime id k*.\n"
        )
        f.write(
            "- overall_success_pct: both behavior and state judged true (strict bundle).\n"
        )
        f.write(
            "- Per-row `numeric_policy` is the objective policy view; compare to judge `state_target_success`.\n\n"
        )

        f.write("SECTION 4: HOW TO ANALYZE\n")
        f.write("-" * 80 + "\n")
        f.write("- If policy steerability (from steering_eval) is high but judge state success is low, "
                "the discrete policy may not match human-intuitive 'regime' semantics.\n")
        f.write("- If judge state success is high but behavior success is low, dynamics shifted on paper "
                "but the sentence still reads like the wrong reasoning stage.\n")
        f.write("- Use by-transfer slices in steering_eval + judge rows to find reliable source->target routes.\n")


def run_pipeline(cfg: SteerConfig) -> Tuple[str, dict]:
    # cfg.beta: β, how aggressively q*(k) is tilted toward target regime.
    # cfg.steer_alpha: α, how strongly latent steering is applied in activation space.
    if not cfg.use_last_token:
        raise ValueError("This import-based version assumes `hidden_state_last`, matching cebra_EM.py.")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # ---------------------------
    # 1) Load features + build triplets using cebra_EM
    # ---------------------------
    data_path, limit_problems = resolve_steering_data_path(cfg)
    # all_features: observed trajectories in activation space (contains z_t proxies in raw form)
    # triplets: temporal contrastive tuples (anchor, positive, negative) for latent encoder training
    all_features, triplets = load_and_prepare_cebra(
        data_path,
        mode="temporal",
        limit_problems=limit_problems,
        max_triplets=cfg.max_triplets_per_pid,
    )

    # ---------------------------
    # 2) Train CEBRA + get latent trajectories (grouped by problem)
    # ---------------------------
    # cebra_EM.train_cebra_projection returns:
    #   cebra_seqs: List[np.ndarray] where each is z_t for one problem trajectory (shape (T, latent_dim))
    #   pca_seqs, labels (we don't need them for steering)
    # cebra_seqs: sequence list; each element is z_{1:T} for one problem in latent space
    cebra_seqs, _pca_seqs, _labels = train_cebra_projection(all_features, triplets, d_out=cfg.cebra_dim)

    # Rebuild grouping indices so we can align:
    #   idx_seqs[pids_sorted[i]] matches cebra_seqs[i] row ordering.
    # p_map: problem_id -> list of row indices; lets us reconstruct each trajectory.
    p_map: Dict[int, List[int]] = {}
    for i, f in enumerate(all_features):
        pid = int(f["problem_id"])
        p_map.setdefault(pid, []).append(i)
    # pids_sorted: trajectory ids we keep (length >=3 for stable dynamics fitting).
    # idx_seqs: per-trajectory index lists aligned with cebra_seqs order.
    pids_sorted = sorted([pid for pid, idxs in p_map.items() if len(idxs) >= 3])
    idx_seqs = [p_map[pid] for pid in pids_sorted]

    # ---------------------------
    # 3) Fit SLDS EM using cebra_EM utilities
    # ---------------------------
    # Eq (1): z_{t+1} | (z_t, s_t=k) ~ N(A_k z_t + b_k, Σ_k)
    # Learned by EM loop via imported: em_init_params / em_forward_backward / em_m_step.
    cebra_mod.KAPPA = cfg.transition_kappa
    latent_dim = cfg.cebra_dim
    # pi   ↔ π(k) = P(s_1 = k)
    # a    ↔ transition matrix, a[i,j] = P(s_{t+1}=j | s_t=i)
    # d_m  ↔ {A_k} regime linear maps in f_k(z_t)=A_k z_t + b_k
    # d_b  ↔ {b_k} regime offsets in f_k(z_t)=A_k z_t + b_k
    # d_cov↔ {Σ_k} regime noise covariances
    pi, a, d_m, d_b, d_cov = em_init_params(cebra_seqs, cfg.k_regimes, latent_dim)

    # gammas/xis: per-sequence posterior stats used by EM updates.
    # final_gammas: cached last-iteration γ_t(k), used to derive hard states.
    final_gammas: List[np.ndarray] = []
    for _ in range(cfg.em_iters):
        gammas, xis = [], []  # batch-level γ_t(k), ξ_t(i,j) for all trajectories at this EM iteration.
        for seq in cebra_seqs:
            # seq: one trajectory z_{1:T} for a single problem.
            # g ↔ γ_t(k) = P(s_t=k | z_{1:T})
            # x ↔ ξ_t(i,j) = P(s_t=i, s_{t+1}=j | z_{1:T})
            g, x, _ll = em_forward_backward(seq, pi, a, d_m, d_b, d_cov, cfg.k_regimes)  # _ll is sequence log-likelihood
            gammas.append(g)
            xis.append(x)
        pi, a, d_m, d_b, d_cov = em_m_step(cebra_seqs, gammas, xis, cfg.k_regimes, latent_dim)
        final_gammas = gammas

    # state_seqs: hard regime path ŝ_t = argmax_k γ_t(k)
    state_seqs = [np.argmax(g, axis=1) for g in final_gammas]

    # ---------------------------
    # 4) Prepare latent z per global feature index
    # ---------------------------
    n_feat = len(all_features)  # total number of sentence-level samples
    # per_sample_state[i] ↔ inferred s_t for sample i
    # z_by_index[i]       ↔ latent z_t for sample i
    per_sample_state = np.zeros(n_feat, dtype=np.int64)
    z_by_index = np.zeros((n_feat, latent_dim), dtype=np.float32)
    # included_mask marks samples that belong to trajectories used by EM.
    included_mask = np.zeros(n_feat, dtype=bool)

    for idxs, z_seq, s_seq in zip(idx_seqs, cebra_seqs, state_seqs):
        for i_local, i_global in enumerate(idxs):
            included_mask[i_global] = True
            per_sample_state[i_global] = int(s_seq[i_local])
            z_by_index[i_global] = z_seq[i_local]

    # ---------------------------
    # 5) Fit decoder on the included (z, x_scaled) pairs
    # ---------------------------
    # X_raw[i] ↔ observed activation vector (original feature space) for sample i
    X_raw = np.array([f["hidden_state_last"] for f in all_features], dtype=np.float32)
    # scaler stores feature mean/std so deltas can be mapped back to raw activation scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    # flatten in the exact same order as z sequences:
    # z_flat/x_flat: aligned training pairs for decoder fit (latent -> standardized activation).
    z_flat = np.concatenate([seq for seq in cebra_seqs], axis=0)
    x_flat = np.concatenate([X_scaled[idxs] for idxs in idx_seqs], axis=0)

    # weights mapping: Δx_scaled ≈ Δz @ w_dec
    # w_dec ↔ decoder weights W_dec in Δx_scaled ≈ Δz W_dec
    w_dec = fit_latent_to_activation_decoder(z_flat, x_flat)

    # ---------------------------
    # 6) Compute steering delta per included sample
    # ---------------------------
    # steering_cache: per-sample steering outputs for downstream analysis/intervention.
    steering_cache: Dict[int, dict] = {}
    # included_indices: actual sample ids for which we have valid z_t and s_t.
    included_indices = np.nonzero(included_mask)[0].tolist()
    for i in included_indices:
        z_t = z_by_index[i]  # z_t
        s_t = int(per_sample_state[i])  # s_t
        target_k = (s_t + 1) % cfg.k_regimes  # k* (target regime to favor)
        # Reward r(k): implicit in compute_steering_delta -> kl_regularized_policy as 1 at k*, 0 otherwise.

        # Eq (2): p(k) = P(s_{t+1}=k | s_t, z_t)  (approximated with row a[s_t])
        # Eq (3): q*(k) = argmax_q [ Σ_k q(k) r(k) - (1/β) KL(q||p) ]
        #         closed-form q*(k) ∝ p(k) exp(β r(k))
        # Eq (4): f_k(z_t) = A_k z_t + b_k
        # Eq (5): μ_orig(z_t) = Σ_k p(k) f_k(z_t)
        # Eq (6): μ_steered(z_t) = Σ_k q*(k) f_k(z_t)
        # Eq (7): Δz_t = μ_steered(z_t) - μ_orig(z_t)
        delta_z, p, q = compute_steering_delta(
            z_t=z_t,
            s_t=s_t,
            target_k=target_k,
            a=a,             # transition prior p(k)=P(s_{t+1}=k|s_t,z_t) proxy
            d_m=d_m,         # {A_k}
            d_b=d_b,         # {b_k}
            beta=cfg.beta,   # β in KL-regularized policy
        )

        # Eq (8): Δx_scaled ≈ Δz_t W_dec ; Δx_raw = Δx_scaled ⊙ scaler.scale_
        # Δx_scaled ≈ Δz @ W_dec; unstandardize delta: Δx_raw = Δx_scaled * scaler.scale_
        delta_x_scaled = delta_z @ w_dec      # Δx_scaled
        delta_x_raw = delta_x_scaled * scaler.scale_  # Δx_raw
        # Eq (9): x_edit = x + α Δx_raw
        x_edit = X_raw[i] + cfg.steer_alpha * delta_x_raw  # x + αΔx
        # Eq (10): x_final = DiffusionPostprocess(x_edit)  (placeholder hook)
        x_edit = diffusion_postprocess_placeholder(x_edit)

        steering_cache[i] = {
            "source_state": s_t,      # s_t
            "target_state": target_k, # k*
            "p_next": p,              # p(k)
            "q_next": q,              # q*(k)
            "delta_z": delta_z,       # Δz_t
            "delta_x_raw": delta_x_raw,  # Δx_raw
            "x_steered": x_edit,      # x_edit
        }

    # ---------------------------
    # 7) Steering evaluation outputs (PDF-style + cross-state summary style)
    # ---------------------------
    steering_report = compute_steering_reports(steering_cache, X_raw)
    write_steering_report(cfg.save_dir, steering_report, cfg)

    # ---------------------------
    # 8) OpenAI judge: behavior-level + state-level steerability (optional)
    # ---------------------------
    state_profiles = build_state_behavior_profiles(
        all_features=all_features,
        per_sample_state=per_sample_state,
        included_indices=included_indices,
        k_regimes=cfg.k_regimes,
        stage_list=list(cebra_mod.STAGES),
    )
    judge_report = run_behavior_state_judge(
        cfg=cfg,
        all_features=all_features,
        steering_cache=steering_cache,
        state_profiles=state_profiles,
        stage_list=list(cebra_mod.STAGES),
    )
    write_judge_report(cfg.save_dir, judge_report, cfg)

    # payload bundles learned dynamics, state assignments, scaler/decoder, steering outputs, and eval summaries.
    payload = {
        "config": cfg.__dict__,
        "pi": pi,  # π
        "A": a,  # transition matrix for p(k)
        "dM": d_m,  # {A_k}
        "db": d_b,  # {b_k}
        "dCov": d_cov,  # {Σ_k}
        "per_sample_state": per_sample_state,  # inferred s_t per sample
        "included_indices": included_indices,  # indices with valid latent trajectory membership
        "scaler_mean": scaler.mean_,  # μ for activation standardization
        "scaler_scale": scaler.scale_,  # σ for activation standardization
        "decoder_W": w_dec,  # W_dec for Δx_scaled ≈ Δz W_dec
        "steering_cache": steering_cache,  # per-sample p(k), q*(k), Δz_t, Δx_raw, x_edit
        "steering_eval": steering_report,  # aggregated steering experiment outputs
        "state_behavior_profiles": state_profiles,  # dominant reasoning stage per latent regime
        "steering_judge_eval": judge_report,  # LLM behavior + state steerability
    }

    out_path = os.path.join(cfg.save_dir, "steer_rpc_cebra_em_imported_artifacts.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    return out_path, payload


if __name__ == "__main__":  # pragma: no cover
    np.random.seed(42)
    _parser = build_cli_parser()
    _args = _parser.parse_args()
    cfg = config_from_cli_args(SteerConfig(), _args)
    out_file, _payload = run_pipeline(cfg)
    print(f"Saved imported CEBRA-EM steering artifacts to: {out_file}")

