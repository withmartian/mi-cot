"""
CEBRA-EM steering on top of ``cebra_EM.py``: train CEBRA latents, fit an SLDS with EM, then steer via a
KL-tilted next-regime law (q* vs p) and map Δz into activation space (Δx) for edits.

**Docs:** CLI flags, data sources, example commands, **outputs**, and **how to analyze** them are in
``README_cebra_em_steering_inputDep.md`` (same directory as this file). From that folder, run
``python cebra_em_steering_inputDep.py --help`` for full argparse text.

Notation (glossary → symbol):
  x_t, X_raw   Hidden state in the *original* activation space (one vector per sentence).
  z_t         Low-dim CEBRA embedding of that sentence (“where we are” in latent trajectory space).
  s_t or k    Discrete *regime* index 0..K-1: which linear dynamics rule the SLDS thinks applies now.
  K / k_regimes   Number of regimes (MDP states for the switching model).
  π (pi)      Initial distribution over regimes at t=1.
  A matrix `a`  Transition matrix: ``a[i,j] = P(next regime j | current i)`` (Markov on regimes).
  A_k, b_k    Per-regime linear map for *one-step* latent prediction: f_k(z)=A_k z + b_k (SLDS).
  p(k)        “Natural” distribution over *next* regime given current regime row (here ≈ row ``a[s_t]``).
  q*(k)       *Steered* next-regime distribution: tilt p toward a target with reward r and KL cost 1/β.
  β (beta)    How hard to favor the target regime in q* vs stay close to p (higher = more aggressive).
  μ_orig, μ_steered  Expected next latent *under* p vs under q* (mixtures of the f_k(z_t)).
  Δz_t        Change in latent we want: steered mixture minus natural mixture (direction in z-space).
  W_dec / w_dec  Linear map latent→(scaled) activation so Δx ≈ Δz W (cheap decoder for steering).
  α (steer_alpha)  How much of Δx we actually add to activations before optional denoising.
  Σ_k (d_cov) Regime noise covariances from EM (used inside dynamics fit; not re-derived here).
"""

from __future__ import annotations

import argparse
import os
import pickle
import json
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
)
from hf_steering_data_io import (
    add_hf_dataset_cli_args,
    apply_hf_dataset_cli_config,
    resolve_hf_subset_data_path,
)

try:
    import cebra_EM as cebra_mod
    from cebra_EM import load_and_prepare_cebra, train_cebra_projection
    from cebra_EM import init_params as em_init_params
    from cebra_EM import forward_backward as em_forward_backward
    from cebra_EM import m_step as em_m_step
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import `cebra_EM.py`."
    ) from e


# ---------------------------------------------------------------------------
# Judge taxonomies
#
# Reasoning *behaviors* — the judge must choose `behavior_stage_judged` from this list,
# passed as `stage_list` (= ``cebra_mod.STAGES``). These are the same string labels stored
# in dataset field ``stage`` (human-annotated reasoning stages). 
# (e.g. Plan Generation ↔ PLAN_GENERATION, Problem Setup ↔ PROBLEM_SETUP,
# Self-Checking ↔ SELF_CHECKING, Fact Retrieval ↔ FACT_RETRIEVAL,
# Active Computation ↔ ACTIVE_COMPUTATION, Consolidation ↔ RESULT_CONSOLIDATION;
# plus UNCERTAINTY_MANAGEMENT, FINAL_ANSWER_EMISSION in the GSM8K annotation schema).
#
# Latent *states* — integers 0..K-1
# ---------------------------------------------------------------------------
REGIME_STAGE_SPECIALIZATION = """
Human stage specialization by latent regime :
- Regime **core computation**: dominated by ACTIVE_COMPUTATION and RESULT_CONSOLIDATION
- Regime **transitional manifold**: mixes many stages; hand-off between setup, compute, and check.
- Regime **structural verification**: SELF_CHECKING (high), FACT_RETRIEVAL.
- Regime **contextual framing & strategy**: PLAN_GENERATION, PROBLEM_SETUP.

""".strip()


@dataclass
class SteerConfig:
    # --- Data / preprocessing (not steering math; feeds CEBRA) ---
    data_path: str = "rpc_dataset_layer28_200/all_sentences_features.pkl"
    use_last_token: bool = True  # Must match feature field ``hidden_state_last`` in pickles.

    limit_problems: int = 500
    max_triplets_per_pid: int = 25
    # limit_problems: only keep rows with problem_id < this (defines which trajectories exist).
    # max_triplets_per_pid: cap contrastive triples per problem (CEBRA training cost vs diversity).

    # --- Geometry of z_t (CEBRA embedding) and how long we train dynamics ---
    cebra_dim: int = 40
    # cebra_dim: dimension of z_t; the SLDS (A_k, b_k, Σ_k) lives in this space.
    cebra_epochs: int = 100  # Used inside cebra_EM training loop only.

    # --- Switching linear dynamical system (SLDS) via EM ---
    em_iters: int = 50
    # em_iters: how many EM rounds to refine π, transitions a, and per-regime (A_k,b_k,Σ_k).
    k_regimes: int = 4
    # k_regimes: K discrete regimes s_t ∈ {0..K-1}; interpret as “reasoning modes” in latent space.
    transition_kappa: float = 1.0
    # transition_kappa → cebra_EM.KAPPA: prior smoothing on regime transitions (avoids trivial switches).

    # --- Steering objective (latent policy tilt + activation injection) ---
    beta: float = 8.0
    # β: KL regularization tradeoff — larger q* favors target regime k* more vs staying near natural p.
    steer_alpha: float = 8.0
    # α: gain from latent delta to activation delta (x_edit = x + α Δx); scales “how hard we nudge” the residual.

    # LLM-as-judge (separate from math: scores whether *text* matches target regime behavior)
    # Set OPENAI_API_KEY in the environment
    enable_openai_judge: bool = True
    judge_model: str = "gpt-4.1-mini"
    judge_max_samples: int = 100
    judge_timeout_s: int = 60
    judge_max_retries: int = 3
    logit_lens_model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    logit_lens_top_k: int = 10

    save_dir: str = "cebra_em_steering"

    # If `data_path` is not a file, optionally download a small subset from Hugging Face
    # (see ``sds_train_gsm8k_hf.SDS_TRAIN_GSM8K_REPO_ID``). HF_TOKEN optional for public files.
    hf_auto_download_if_missing: bool = True
    hf_dataset_repo: str = SDS_TRAIN_GSM8K_REPO_ID
    hf_dataset_filename: str = DEFAULT_STEERING_HF_FEATURES_RELPATH
    hf_fallback_num_samples: int = 100
    hf_fallback_cache_name: str = "sds_hf100_default_qwen15b_reasoning_l27.pkl"


# ---------------------------------------------------------------------------
# File Parsing / CLI Configuration
# ---------------------------------------------------------------------------
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
    add_hf_dataset_cli_args(p)
    p.add_argument(
        "--no-openai-judge",
        action="store_true",
        help="Skip OpenAI behavior/state judge (faster; no OPENAI_API_KEY needed).",
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

    c = apply_hf_dataset_cli_config(c, args)

    if args.limit_problems is not None:
        c = replace(c, limit_problems=int(args.limit_problems))

    return c


# ---------------------------------------------------------------------------
# Steering Core + Evaluation
# ---------------------------------------------------------------------------
def fit_latent_to_activation_decoder(z_flat: np.ndarray, x_scaled_flat: np.ndarray) -> np.ndarray:
    """
    Learn a *linear bridge* from CEBRA latents z to standardized activations x_scaled.

    Why: SLDS lives in z-space (μ_orig, μ_steered, Δz), but interventions are applied in the
    model’s native residual space x. A first-order map Δx_scaled ≈ Δz @ w lets us translate
    a small latent move into an activation-direction without re-running the full encoder.

    Math: fit x_scaled ≈ [z, 1] @ coef; return w = coef[:-1] so the intercept cancels in deltas:
          (z+Δz)@w + b0 − (z@w + b0) = Δz @ w.

    Args:
        z_flat: stacked z_t rows (N, dim_z) from trajectories used in EM.
        x_scaled_flat: same rows in standardized activation space (N, dim_x).
    """
    z_aug = np.hstack([z_flat, np.ones((len(z_flat), 1), dtype=z_flat.dtype)])
    coef, *_ = np.linalg.lstsq(z_aug, x_scaled_flat, rcond=None)
    w = coef[:-1]  # bias cancels when we compute deltas
    return w


def kl_regularized_policy(p: np.ndarray, target_k: int, beta: float) -> np.ndarray:
    """
    Closed-form *steered* next-regime distribution q* given natural p and a 0-1 reward on k*.

    Interpretation: “If we could pick tomorrow’s regime distribution q, we’d maximize expected
    reward Σ q(k) r(k) but pay KL(q‖p) so we don’t stray from the learned dynamics prior.”
    With r = one-hot(target_k), the optimizer is Gibbs tilt: q*(k) ∝ p(k) exp(β r(k)).

    Args:
        p:   p(k) = skeptical prior over next regime (here one row of the SLDS transition matrix).
        target_k: k* — which regime index we want to encourage (this script uses a cyclic demo rule).
        beta: β — inverse temperature; larger ⇒ q* puts more mass on k* (and higher KL to p).
    """
    p = p / (np.sum(p) + 1e-12)
    g = np.zeros_like(p)
    g[target_k] = 1.0
    q_unnorm = p * np.exp(beta * g)
    return q_unnorm / np.sum(q_unnorm)


def compute_statewise_next_means(z_t: np.ndarray, d_m: np.ndarray, d_b: np.ndarray) -> np.ndarray:
    """
    One-step *predicted latent mean* under each possible discrete regime k.

    Under SLDS, if we *knew* the next regime were k, the Gaussian emission model implies
    E[z_{t+1} | z_t, regime k] ≈ A_k z_t + b_k. Stacking all k gives every “next-landmark” in
    latent space we might blend.

    Args:
        z_t: current latent position (dim_z,).
        d_m: stack of A_k matrices, shape (K, dim_z, dim_z).
        d_b: stack of b_k vectors, shape (K, dim_z).

    Returns:
        f_k rows, shape (K, dim_z): f_k[k] = A_k z_t + b_k.
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
    Turn a *preference over next regimes* into a *concrete latent nudge* Δz_t.

    Straightforward equation references:
      p(k)  = P(s_{t+1}=k | s_t, z_t)  (we approximate with row a[s_t])
      q*(k) ∝ p(k) exp(β r(k))
      μ_orig(z_t)    = Σ_k p(k) f_k(z_t)
      μ_steered(z_t) = Σ_k q*(k) f_k(z_t)
      Δz_t = μ_steered - μ_orig

    Steps (why each):
      1) Build p(k): “Where would dynamics go next *without* steering?” → use Markov row on regimes.
      2) Build q*(k): tilt p toward target_k with strength β (KL-regularized planning in discrete space).
      3) μ_orig = Σ_k p(k) f_k(z_t): expected next latent if we followed natural regime uncertainty.
      4) μ_steered = Σ_k q*(k) f_k(z_t): expected next latent if we followed steered regime uncertainty.
      5) Δz_t = μ_steered − μ_orig: smallest latent move (in this linear mixture sense) that realizes
         the policy change from p to q*.

    Args:
        z_t:      current CEBRA embedding z_t.
        s_t:      current hard-assigned regime index (from EM posteriors).
        target_k: k* — reward vertex for q* (here: demo rule cycles (s_t+1)%K).
        a:        transition matrix; a[i,j] = P(s_{t+1}=j | s_t=i). Row a[s_t] is our p(k) proxy.
        d_m, d_b: per-regime (A_k, b_k) defining f_k.
        beta:     β — KL vs reward tradeoff for q*.

    Returns:
        delta_z: Δz_t (dim_z,)
        p:       normalized p(k)
        q:       normalized q*(k)
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
    Hook for *on-manifold* projection after a raw activation edit.

    x + αΔx may leave the “realizable” activation manifold; a learned denoiser / GLP
    can snap the vector back to plausible states before use.
    Here: identity so the pipeline runs without an extra model.
    """
    return acts_edit


def run_logit_lens_evaluation(
    cfg: SteerConfig,
    steering_cache: Dict[int, dict],
    x_raw: np.ndarray,
    per_sample_state: np.ndarray,
    included_indices: List[int],
) -> dict:
    """
    Logit-lens evaluation in the style of ``cebra_steering.py``:
    project activation vectors through unembedding W_U and score target-state token signatures.
    """
    if not steering_cache:
        return {"enabled": False, "reason": "empty_steering_cache", "global": {}, "rows": []}

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        return {
            "enabled": False,
            "reason": f"logit_lens_dependencies_missing: {e}",
            "global": {},
            "rows": [],
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.logit_lens_model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.logit_lens_model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        return {
            "enabled": False,
            "reason": f"logit_lens_model_load_failed: {e}",
            "global": {},
            "rows": [],
        }

    emb = model.get_output_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        return {
            "enabled": False,
            "reason": "logit_lens_no_output_embedding",
            "global": {},
            "rows": [],
        }

    w_u = emb.weight.detach().to(torch.float32).cpu().T  # (d_model, vocab)
    del model

    d_model = int(w_u.shape[0])
    if int(x_raw.shape[1]) != d_model:
        return {
            "enabled": False,
            "reason": f"logit_lens_dim_mismatch x_dim={x_raw.shape[1]} vs d_model={d_model}",
            "global": {},
            "rows": [],
        }

    # Build per-state centroid signatures (top-k tokens from centroid @ W_U).
    state_to_idxs: Dict[int, List[int]] = {}
    for idx in included_indices:
        s = int(per_sample_state[idx])
        state_to_idxs.setdefault(s, []).append(int(idx))

    k = max(1, int(cfg.logit_lens_top_k))
    state_signatures: Dict[int, dict] = {}
    for s, idxs in state_to_idxs.items():
        centroid = np.mean(x_raw[idxs], axis=0).astype(np.float32)
        c = torch.from_numpy(centroid).cpu()
        logits = c @ w_u
        top_vals, top_ids = torch.topk(logits, k=min(k, int(logits.shape[0])))
        tokens = [tokenizer.decode([int(i)]) for i in top_ids.tolist()]
        state_signatures[int(s)] = {
            "token_ids": [int(i) for i in top_ids.tolist()],
            "tokens": tokens,
            "logits": [float(v) for v in top_vals.tolist()],
        }

    rows: List[dict] = []
    for idx, entry in steering_cache.items():
        s = int(entry["source_state"])
        t = int(entry["target_state"])
        sig_t = state_signatures.get(t, {}).get("token_ids", [])
        sig_s = state_signatures.get(s, {}).get("token_ids", [])
        if not sig_t or not sig_s:
            continue

        x0 = torch.from_numpy(np.asarray(x_raw[idx], dtype=np.float32)).cpu()
        x1 = torch.from_numpy(np.asarray(entry["x_steered"], dtype=np.float32)).cpu()

        # Token-signature logit shifts: target-vs-source margin should increase when steering works.
        t0 = float((x0 @ w_u[:, sig_t]).mean().item())
        t1 = float((x1 @ w_u[:, sig_t]).mean().item())
        s0 = float((x0 @ w_u[:, sig_s]).mean().item())
        s1 = float((x1 @ w_u[:, sig_s]).mean().item())
        margin0 = t0 - s0
        margin1 = t1 - s1
        margin_lift = margin1 - margin0
        target_logit_lift = t1 - t0
        success = int(margin_lift > 0.0)

        rows.append(
            {
                "idx": int(idx),
                "source_state": s,
                "target_state": t,
                "target_signature_tokens": state_signatures[t]["tokens"],
                "source_signature_tokens": state_signatures[s]["tokens"],
                "target_logit_orig": t0,
                "target_logit_steered": t1,
                "target_logit_lift": target_logit_lift,
                "margin_orig": margin0,
                "margin_steered": margin1,
                "margin_lift": margin_lift,
                "logit_lens_success": success,
            }
        )

    if not rows:
        return {
            "enabled": False,
            "reason": "logit_lens_no_rows",
            "global": {},
            "rows": [],
            "state_signatures": state_signatures,
        }

    def _agg(items: List[dict]) -> dict:
        arr_t = np.array([r["target_logit_lift"] for r in items], dtype=np.float64)
        arr_m = np.array([r["margin_lift"] for r in items], dtype=np.float64)
        arr_s = np.array([r["logit_lens_success"] for r in items], dtype=np.float64)
        return {
            "count": int(len(items)),
            "target_logit_lift": float(arr_t.mean()),
            "margin_lift": float(arr_m.mean()),
            "logit_lens_success_pct": float(100.0 * arr_s.mean()),
        }

    global_stats = _agg(rows)
    by_target: Dict[int, dict] = {}
    for t in sorted({r["target_state"] for r in rows}):
        by_target[int(t)] = _agg([r for r in rows if r["target_state"] == t])

    by_transfer: Dict[str, dict] = {}
    for s, t in sorted({(r["source_state"], r["target_state"]) for r in rows}):
        by_transfer[f"{s}->{t}"] = _agg(
            [r for r in rows if r["source_state"] == s and r["target_state"] == t]
        )

    return {
        "enabled": True,
        "model_id": cfg.logit_lens_model_id,
        "top_k": k,
        "global": global_stats,
        "by_target": by_target,
        "by_transfer": by_transfer,
        "rows": rows,
        "state_signatures": state_signatures,
    }


# ---------------------------------------------------------------------------
# File Output / Reporting
# ---------------------------------------------------------------------------
def compute_steering_reports(
    steering_cache: Dict[int, dict],
    x_raw: np.ndarray,
) -> dict:
    """
    Aggregate *policy-level* and *magnitude* metrics per sample, then bucket by target / transfer.

    Metric definitions used here:
      steerability(%): 100 * mean[argmax(q*) == target_k]
      coherence_penalty: mean(||delta_x_raw|| / ||x_raw||)
      target_prob_lift: mean(q*[target] - p[target])
      kl_q_p: mean(KL(q* || p))

    Per-sample fields (from steering_cache + x_raw):
      p, q     Same p(k), q*(k) as in compute_steering_delta — “natural vs steered” next regime law.
      dx, x    Δx_raw and original x — how big the nudge is relative to the vector norm.

    Report scalars (meaning):
      steerability_pct — Did q* actually *privilege* the chosen target (argmax hits target_k)?
      target_prob_lift — How much probability mass moved onto target under q* vs p?
      kl_q_p            — Control cost: how far we moved the categorical law (bits-scale divergence).
      l1_policy_shift   — Total probability moved between bins (L1 distance p vs q*).
      coherence_penalty — ‖Δx‖/‖x‖: relative size of activation edit (physical “how loud” the nudge).
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
        s = int(entry["source_state"])  # s_t for this row
        t = int(entry["target_state"])  # k* (reward vertex) for this row
        p = np.asarray(entry["p_next"], dtype=np.float64)  # natural next-regime law
        q = np.asarray(entry["q_next"], dtype=np.float64)  # steered q* (Gibbs tilt of p)
        dx = np.asarray(entry["delta_x_raw"], dtype=np.float64)  # activation-space push Δx
        x = np.asarray(x_raw[idx], dtype=np.float64)  # original x_t (denominator for relative norm)

        # lift: extra mass on the chosen target bin; kl: information cost of replacing p by q*.
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


def write_logit_lens_report(save_dir: str, report: dict, cfg: SteerConfig) -> None:
    """Persist logit-lens steering report as JSON + text."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "steering_logit_lens_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    txt_path = os.path.join(save_dir, "steering_logit_lens_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Steering Logit-Lens Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"- enabled: {report.get('enabled', False)}\n")
        if not report.get("enabled", False):
            f.write(f"- reason: {report.get('reason', '')}\n")
            return
        f.write(f"- model_id: {report.get('model_id', '')}\n")
        f.write(f"- top_k: {report.get('top_k', 0)}\n\n")
        g = report.get("global", {})
        f.write("Global metrics:\n")
        f.write(f"- count: {g.get('count', 0)}\n")
        f.write(f"- target_logit_lift: {g.get('target_logit_lift', 0.0):.6f}\n")
        f.write(f"- margin_lift: {g.get('margin_lift', 0.0):.6f}\n")
        f.write(f"- logit_lens_success_pct: {g.get('logit_lens_success_pct', 0.0):.4f}\n")


def write_steering_report(
    save_dir: str,
    report: dict,
    cfg: SteerConfig,
) -> None:
    """
    Serialize ``compute_steering_reports`` to JSON + a prose guide.

    Why separate text report: downstream readers may not remember symbol meanings (β, α, p, q*, Δx);
    SECTION 3 ties metric names to steering objectives (policy mass vs activation norm).
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


def write_judge_report(save_dir: str, judge_report: dict, cfg: SteerConfig) -> None:
    """Persist LLM judge JSON + human-readable summary (behavior/state/coherence rates)."""
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
            "- coherence_success_pct: fraction where behavior, policy shift, and logit-lens evidence are "
            "internally coherent.\n"
        )
        f.write(
            "- overall_success_pct: behavior, state, and coherence all judged true (strict bundle).\n"
        )
        f.write(
            "- Per-row `numeric_policy` + `numeric_logit_lens` are objective views; compare to judge fields.\n\n"
        )

        f.write("SECTION 4: HOW TO ANALYZE\n")
        f.write("-" * 80 + "\n")
        f.write("- If policy steerability (from steering_eval) is high but judge state success is low, "
                "the discrete policy may not match human-intuitive 'regime' semantics.\n")
        f.write("- If judge state success is high but behavior success is low, dynamics shifted "
                "but the sentence still reads like the wrong reasoning stage.\n")
        f.write("- If coherence_success is low, steering cues disagree across text/policy/logit-lens signals.\n")
        f.write("- Use by-transfer slices in steering_eval + judge rows to find reliable source->target routes.\n")


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------
def build_state_behavior_profiles(
    all_features: List[dict],
    per_sample_state: np.ndarray,
    included_indices: List[int],
    k_regimes: int,
    stage_list: List[str],
) -> Dict[int, dict]:
    """
    Map each discrete regime index k to “what humans usually label when s_t=k”.

    Variables:
      per_sample_state[i] — SLDS hard state ŝ_t for row i (from EM).
      all_features[i]["stage"] — human reasoning-behavior tag (PROBLEM_SETUP, …).

    Why: EM’s regime IDs are unlabeled; *dominant_stage* per k tells the judge what behavioral
    persona k* “should” look like when we claim we steered toward target regime t.
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
    logit_lens_rows_by_idx: Optional[Dict[int, dict]] = None,
) -> dict:
    """
    LLM-as-judge on two *different* notions of “did steering work?”:

    1) Behavior: Does the *surface text* look like the human stage associated with target regime
       (via target_stage_prior from build_state_behavior_profiles)? This checks semantics, not p,q.

    2) State (judge view): Given p→q* and logit-lens margin shifts, does the judge agree steering
       moved toward k*?

    3) Coherence (judge view): Is the steering signal internally coherent (text behavior, policy
       shift, and logit-lens evidence point in a compatible direction) rather than contradictory?

    Loop variables:
      s, t — source regime s_t and target k* (same indices as steering_cache).
      p, q — normalized p_next, q_next (policy before/after tilt).
      source_stage_prior / target_stage_prior — empirical “persona” of regimes s and t.
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
        "Latent regimes s_t and k* are integer ids 0..K-1 from a learned switching dynamic "
        "model (default K=4).
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
        # Renormalize for the prompt (numerical safety; should already be probs summing to 1).
        p = p / (np.sum(p) + eps)
        q = q / (np.sum(q) + eps)

        source_stage_prior = state_profiles.get(s, {}).get("dominant_stage", "UNKNOWN")
        target_stage_prior = state_profiles.get(t, {}).get("dominant_stage", "UNKNOWN")
        text = str(feat.get("text", feat.get("sentence", ""))).strip()
        if not text:
            text = f"[no text field; sample index {idx}]"
        text_for_judge = text if len(text) <= 2500 else text[:2500] + "…"
        label = feat.get("stage", "UNKNOWN")

        lens = (logit_lens_rows_by_idx or {}).get(int(idx), {})
        num_argmax_p = int(np.argmax(p))
        num_argmax_q = int(np.argmax(q))
        num_lift = float(q[t] - p[t])
        num_top1_success = int(num_argmax_q == t)
        lens_margin_lift = float(lens.get("margin_lift", 0.0))
        lens_target_logit_lift = float(lens.get("target_logit_lift", 0.0))
        lens_success = int(bool(lens.get("logit_lens_success", 0)))

        user_prompt = f"""Evaluate steering success for BOTH reasoning behavior and latent state.

Allowed reasoning behavior (stage) labels — use EXACTLY these strings for behavior_stage_judged :
{stage_list}

Qualitative regime–stage specialization:
{REGIME_STAGE_SPECIALIZATION}

Input:
- Sentence / CoT fragment: {text_for_judge!r}
- Dataset stage label for this token/sentence: {label!r}
- Source latent regime id s_t: {s}
- Target latent regime id k* (steering goal): {t}
- Empirical dominant stage among samples assigned to source regime {s}: {source_stage_prior!r}
- Empirical dominant stage among samples assigned to target regime {t}: {target_stage_prior!r}
- p_next (natural next-regime distribution P(s_{{t+1}}|s_t) as row of A): {p.round(6).tolist()}
- q_next (KL-regularized steered distribution q*): {q.round(6).tolist()}
- logit_lens_margin_lift (target-signature minus source-signature margin shift): {lens_margin_lift:.6f}
- logit_lens_target_logit_lift (target-signature mean logit shift): {lens_target_logit_lift:.6f}
- logit_lens_success (margin_lift > 0): {lens_success}

Return a JSON object with exactly these keys:
{{
  "behavior_stage_judged": "<one string from the allowed list, or UNKNOWN if unclear>",
  "behavior_target_success": <true or false>,
  "state_target_success": <true or false>,
  "coherence_success": <true or false>,
  "overall_success": <true or false>,
  "confidence": <number from 0 to 1>,
  "reason": "<one short sentence>"
}}

Definitions:
- behavior_target_success: true iff the sentence's *reasoning behavior* is substantially aligned with the kind of reasoning associated with target regime {t} (use target_stage_prior as a strong hint of what that regime "means" behaviorally).
- state_target_success: true iff policy and logit-lens evidence both indicate movement toward target regime {t} (e.g., q_next[{t}] > p_next[{t}] and/or positive logit_lens_margin_lift).
- coherence_success: true iff the overall steering evidence is internally coherent (the judged behavior, policy shift, and logit-lens shift are not in strong contradiction).
- overall_success: true iff behavior_target_success, state_target_success, and coherence_success are all true.
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
                "numeric_logit_lens": {
                    "margin_lift": lens_margin_lift,
                    "target_logit_lift": lens_target_logit_lift,
                    "logit_lens_success": lens_success,
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
                "coherence_success_pct": 0.0,
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
    coh = _rate("coherence_success")
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
            "coherence_success_pct": 100.0 * coh,
            "overall_success_pct": 100.0 * ov,
            "mean_confidence": conf,
        },
    }


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(cfg: SteerConfig) -> Tuple[str, dict]:
    """
    End-to-end: raw activations → z_t (CEBRA) → SLDS EM → (p,q*,Δz,Δx) → reports + saved payload.

    High-level *why* each block exists:
      (1–2) Need smooth latent trajectories z_{1:T} per problem so a switching linear model is fit-table.
      (3)   SLDS gives discrete s_t and dynamics f_k — without this, we have no p(k) or blend targets μ.
      (4)   Flatten sequences back to per-row z_t, s_t for the same index order as ``all_features``.
      (5)   Decoder maps latent steering Δz into activation space so we can *intervene* on x.
      (6)   Per-sample steering_cache materializes p, q*, Δz, scaled/unscaled Δx, edited x.
      (7–8) Aggregate metrics + optional LLM judge (orthogonal checks on math vs behavior).
    """
    # cfg.beta: β, how aggressively q*(k) is tilted toward target regime.
    # cfg.steer_alpha: α, how strongly latent steering is applied in activation space.
    if not cfg.use_last_token:
        raise ValueError("This import-based version assumes `hidden_state_last`, matching cebra_EM.py.")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # ---------------------------
    # 1) Load features + build triplets using cebra_EM
    # ---------------------------
    # Brings activations onto disk path (local or HF cache); limit_problems may shrink after download.
    data_path, limit_problems = resolve_hf_subset_data_path(cfg)
    # load_and_prepare_cebra: each row is one timestep in a problem’s CoT; fields hold x_t in ``hidden_state_last``.
    # Why triplets: CEBRA is contrastive — anchors/positives along time pull z_t smooth, negatives separate windows.
    # Output symbols: all_features[i] ↔ one sentence; triplets drive encoder, not SLDS directly.
    all_features, triplets = load_and_prepare_cebra(
        data_path,
        mode="temporal",
        limit_problems=limit_problems,
        max_triplets=cfg.max_triplets_per_pid,
    )

    # ---------------------------
    # 2) Train CEBRA + get latent trajectories (grouped by problem)
    # ---------------------------
    # train_cebra_projection: learns encoder such that nearby *time* steps are nearby in z.
    # cebra_seqs[p] is z_{1:T_p} — latent trajectory for problem p; T_p = number of sentences in that problem.
    # dim of each row = cfg.cebra_dim (= dim of every z_t fed into EM and steering).
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
    # We posit: continuous z_t evolves under one of K linear-Gaussian regimes s_t.
    # Eq (1): z_{t+1} | (z_t, s_t=k) ~ N(A_k z_t + b_k, Σ_k); markov transitions on s_t with matrix ``a``.
    # EM alternates: infer γ_t(k)=P(s_t=k|obs), ξ_t(i,j)=P(s_t=i,s_{t+1}=j|obs) (E-step),
    # then update π, a, {(A_k,b_k,Σ_k)} (M-step). Without this, ``p = row a[s_t]`` is undefined.
    cebra_mod.KAPPA = cfg.transition_kappa
    latent_dim = cfg.cebra_dim
    # pi   ↔ π(k) = P(s_1 = k)
    # a    ↔ transition matrix, a[i,j] = P(s_{t+1}=j | s_t=i)
    # d_m  ↔ {A_k} regime linear maps in f_k(z_t)=A_k z_t + b_k
    # d_b  ↔ {b_k} regime offsets in f_k(z_t)=A_k z_t + b_k
    # d_cov↔ {Σ_k} regime noise covariances
    # Random / moment init for π, a, {A_k,b_k,Σ_k}; EM will move these to a local likelihood optimum.
    pi, a, d_m, d_b, d_cov = em_init_params(cebra_seqs, cfg.k_regimes, latent_dim)

    # gammas/xis: per-sequence posterior stats used by M-step.
    # final_gammas: last E-step γ_t(k) — we collapse to ŝ_t = argmax_k γ_t(k) for steering’s observed s_t.
    final_gammas: List[np.ndarray] = []
    for _ in range(cfg.em_iters):
        gammas, xis = [], []  # parallel list to cebra_seqs: one (γ,ξ) pair per trajectory.
        for seq in cebra_seqs:
            # seq: one z_{1:T}; forward_backward = HMM with emissions N(A_s z + b_s, Σ_s).
            # g ↔ γ_t(k) = P(s_t=k | z_{1:T})  (soft regime belief at each time)
            # x ↔ ξ_t(i,j) = P(s_t=i, s_{t+1}=j | z_{1:T})  (used to refine transition counts toward ``a``)
            # _ll: marginal log p(z_{1:T} | θ) — diagnostic only here.
            g, x, _ll = em_forward_backward(seq, pi, a, d_m, d_b, d_cov, cfg.k_regimes)
            gammas.append(g)
            xis.append(x)
        # M-step: treats responsibilities γ,ξ as complete data → closed-form-ish updates for θ={π,a,A_k,b_k,Σ_k}.
        pi, a, d_m, d_b, d_cov = em_m_step(cebra_seqs, gammas, xis, cfg.k_regimes, latent_dim)
        final_gammas = gammas

    # state_seqs: hard regime path ŝ_t = argmax_k γ_t(k) — plug-in s_t for ``a[s_t]`` and labeling.
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
    # X_raw[i] ↔ x_t in native model residual space (what we ultimately add Δx to).
    X_raw = np.array([f["hidden_state_last"] for f in all_features], dtype=np.float32)
    # Standardize per feature dim so least-squares isn’t dominated by high-variance coordinates.
    # scaler.mean_, scaler.scale_: used later to map Δx_scaled → Δx_raw via elementwise * scale_ (no shift on deltas).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    # Only trajectory rows participate: z_flat[i] must match x_flat[i] for the same global index order
    # as flattened idx_seqs (same ordering as EM’s cebra_seqs).
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
        z_t = z_by_index[i]  # current CEBRA latent for this sentence
        s_t = int(per_sample_state[i])  # hard EM regime at this timestep
        # Demo choice of target: cyclically favor “next” regime index (not from data — swap for real k*).
        target_k = (s_t + 1) % cfg.k_regimes  # k* for reward r(k)=1[k=k*]
        # reward r(k): implicit 1-hot inside kl_regularized_policy via target_k.

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
            a=a,             # Markov transitions → prior p over *next* regime from current s_t
            d_m=d_m,         # {A_k} in f_k(z)=A_k z+b_k
            d_b=d_b,         # {b_k}
            beta=cfg.beta,   # β — policy aggression vs staying near p
        )

        # Eq (8): Δx_scaled ≈ Δz_t W_dec ; Δx_raw = Δx_scaled ⊙ scaler.scale_
        # Δx_scaled ≈ Δz @ W_dec; unstandardize delta: Δx_raw = Δx_scaled * scaler.scale_
        delta_x_scaled = delta_z @ w_dec      # (dim_x,) in standardized space
        delta_x_raw = delta_x_scaled * scaler.scale_  # map STD-normalized delta back to raw feature scale
        # Eq (9): x_edit = x + α Δx_raw
        x_edit = X_raw[i] + cfg.steer_alpha * delta_x_raw
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
    # 7) Numeric steering report
    # ---------------------------
    steering_report = compute_steering_reports(steering_cache, X_raw)
    write_steering_report(cfg.save_dir, steering_report, cfg)

    # ---------------------------
    # 8) Logit-lens report (centroid signatures + target-margin lift)
    # ---------------------------
    logit_lens_report = run_logit_lens_evaluation(
        cfg=cfg,
        steering_cache=steering_cache,
        x_raw=X_raw,
        per_sample_state=per_sample_state,
        included_indices=included_indices,
    )
    write_logit_lens_report(cfg.save_dir, logit_lens_report, cfg)
    lens_rows_by_idx = {
        int(r["idx"]): r for r in logit_lens_report.get("rows", []) if "idx" in r
    }

    # ---------------------------
    # 9) OpenAI judge: behavior-level + state-level + coherence (optional)
    # ---------------------------
    # build_state_behavior_profiles: attach human ``stage`` histograms to each EM regime k (label alignment).
    state_profiles = build_state_behavior_profiles(
        all_features=all_features,
        per_sample_state=per_sample_state,
        included_indices=included_indices,
        k_regimes=cfg.k_regimes,
        stage_list=list(cebra_mod.STAGES),
    )
    # run_behavior_state_judge: LLM reads text + (p,q*) + regime priors; not used to *compute* Δz.
    judge_report = run_behavior_state_judge(
        cfg=cfg,
        all_features=all_features,
        steering_cache=steering_cache,
        state_profiles=state_profiles,
        stage_list=list(cebra_mod.STAGES),
        logit_lens_rows_by_idx=lens_rows_by_idx,
    )
    write_judge_report(cfg.save_dir, judge_report, cfg)

    # payload: reproducible bundle — θ (SLDS), per-row latent stats, steering vectors, plus eval/judge mirrors.
    payload = {
        "config": cfg.__dict__,
        "pi": pi,  # π(k)=P(s_1=k)
        "A": a,  # a[i,j]=P(s_{t+1}=j|s_t=i); rows feed p in steering
        "dM": d_m,  # A_k stacks
        "db": d_b,  # b_k stacks
        "dCov": d_cov,  # Σ_k — dynamics uncertainty (from EM)
        "per_sample_state": per_sample_state,  # ŝ_t per global row
        "included_indices": included_indices,  # rows with EM+decoder coverage
        "scaler_mean": scaler.mean_,  # feature means (for standardization context)
        "scaler_scale": scaler.scale_,  # feature stds — also δ un-normalization factor
        "decoder_W": w_dec,  # linear z→x_scaled steering directions
        "steering_cache": steering_cache,  # per-row p, q*, Δz, Δx_raw, x_steered
        "steering_eval": steering_report,  # scalar summaries / slices
        "steering_logit_lens_eval": logit_lens_report,  # token-signature projection metrics
        "state_behavior_profiles": state_profiles,  # regime → human stage distribution
        "steering_judge_eval": judge_report,  # qualitative agreement with math-only metrics
    }

    out_path = os.path.join(cfg.save_dir, "steer_cebra_em_imported_artifacts.pkl")
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

