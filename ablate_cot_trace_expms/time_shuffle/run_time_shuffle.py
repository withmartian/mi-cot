#!/usr/bin/env python3
"""
Ablation (2): time-shuffle sentence-level trajectories before SDS/EM fitting.

What gets shuffled
------------------
After CEBRA, each problem has a trajectory of embedding rows in real CoT order::

  cebra_seqs[p] = [z_0, z_1, z_2, ...]   # z_t = embedding for sentence t

shuffle_sequence_lists applies the same perm to cebra_seqs, pca_seqs, and labels.
Example::

  Real:      [z_0, z_1, z_2]   # sentences 0 -> 1 -> 2
  Shuffled:  [z_2, z_0, z_1]   # same sentence-linked vectors, wrong timeline

Activations stay tied to the sentence they were extracted from; we do not shuffle
text in the LM or decouple h_t from sentence t.

What is held fixed vs changed
-----------------------------
- Fixed: CEBRA training (temporal positive pairs on real order), number of sentences T.
- Changed: only the order of rows fed to SDS/EM (forward-backward + m_step).

What this tests
---------------
Whether SDS metrics (persistence, self-transition, regime R^2, BIC) require real
temporal order. Expected if the readout is valid: real_order >> shuffled_order.
Does not test base-vs-RFT (run twice on different feature pickles).
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exploration.cebra_EM import (  # noqa: E402
    K_SWEEP,
    fit_and_evaluate,
    linear_ar_r2,
    load_and_prepare_cebra,
    train_cebra_projection,
)

from shuffle import shuffle_sequence_lists  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Time-shuffle ablation: permute sentence-aligned embedding rows before SDS. "
            "Example: real [z0,z1,z2] vs shuffled [z2,z0,z1]; CEBRA unchanged. "
            "Tests whether SDS metrics need temporal order (sanity check, not length control)."
        ),
    )
    p.add_argument(
        "--features-path",
        type=Path,
        required=True,
        help="Pickle from create_dataset.py (all_sentences_features.pkl)",
    )
    p.add_argument("--limit-problems", type=int, default=500)
    p.add_argument("--cebra-mode", default="temporal", choices=["temporal"])
    p.add_argument(
        "--shuffle-mode",
        default="full",
        choices=["full", "block"],
        help="full: permute all sentences; block: permute blocks of sentences",
    )
    p.add_argument("--block-size", type=int, default=3)
    p.add_argument(
        "--shuffle-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Independent shuffle seeds; metrics averaged over shuffled runs",
    )
    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=None,
        help="Regime counts to evaluate (default: exploration.cebra_EM.K_SWEEP)",
    )
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--cebra-cache",
        type=Path,
        default=None,
        help="Save/load (cebra_seqs, pca_seqs, labels, ar_r2) to skip CEBRA retraining",
    )
    return p.parse_args()


def _metrics_row(order: str, k: int, seed: int | None, fit_tuple) -> dict:
    persist, mean_st, spec, _, _, score, k_eff, r2, bic = fit_tuple
    return {
        "order": order,
        "K": k,
        "shuffle_seed": seed,
        "persistence": float(persist),
        "mean_self_transition": float(mean_st),
        "specialization": float(spec),
        "sss": float(score),
        "K_eff": int(k_eff),
        "regime_r2": float(r2),
        "bic": float(bic),
    }


def _aggregate_shuffled(rows: list[dict], k: int) -> dict:
    sub = [r for r in rows if r["order"] == "shuffled" and r["K"] == k]
    if not sub:
        return {}
    keys = ["persistence", "mean_self_transition", "specialization", "sss", "K_eff", "regime_r2", "bic"]
    out = {"K": k, "order": "shuffled_mean", "n_seeds": len(sub)}
    for key in keys:
        vals = [r[key] for r in sub]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def execute_ablation(
    features_path: Path,
    *,
    limit_problems: int = 500,
    cebra_mode: str = "temporal",
    shuffle_mode: str = "full",
    block_size: int = 3,
    shuffle_seeds: list[int] | None = None,
    k_values: list[int] | None = None,
    cebra_cache: Path | None = None,
    out_dir: Path | None = None,
    model_tag: str = "model",
    verbose: bool = True,
) -> dict:
    """Train CEBRA, fit SDS on real and shuffled trajectories; return summary dict."""
    if shuffle_seeds is None:
        shuffle_seeds = [0, 1, 2, 3, 4]
    k_values = list(k_values) if k_values is not None else list(K_SWEEP)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if cebra_cache and cebra_cache.exists():
        log(f"[{model_tag}] Loading CEBRA cache from {cebra_cache}")
        with open(cebra_cache, "rb") as f:
            cache = pickle.load(f)
        cebra_seqs = cache["cebra_seqs"]
        pca_seqs = cache["pca_seqs"]
        labels = cache["labels"]
        ar_r2 = cache["ar_r2"]
    else:
        log(f"[{model_tag}] Loading features: {features_path}")
        all_f, triplets = load_and_prepare_cebra(
            str(features_path),
            mode=cebra_mode,
            limit_problems=limit_problems,
        )
        log(f"[{model_tag}] Training CEBRA (real temporal order)...")
        cebra_seqs, pca_seqs, labels = train_cebra_projection(all_f, triplets)
        ar_r2 = linear_ar_r2(pca_seqs)
        log(f"  Linear AR baseline R² (PCA): {ar_r2:.4f}")
        log(f"  Trajectories: {len(cebra_seqs)}")
        lengths = [len(s) for s in cebra_seqs]
        log(f"  Seq lengths: mean={np.mean(lengths):.1f} min={min(lengths)} max={max(lengths)}")

        if cebra_cache:
            cebra_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cebra_cache, "wb") as f:
                pickle.dump(
                    {
                        "cebra_seqs": cebra_seqs,
                        "pca_seqs": pca_seqs,
                        "labels": labels,
                        "ar_r2": ar_r2,
                        "features_path": str(features_path),
                        "limit_problems": limit_problems,
                    },
                    f,
                )
            log(f"  Saved CEBRA cache to {cebra_cache}")

    detail_rows: list[dict] = []

    log(f"\n[{model_tag}] === Real temporal order (SDS) ===")
    for k in k_values:
        fit = fit_and_evaluate(cebra_seqs, pca_seqs, labels, k)
        row = _metrics_row("real", k, None, fit)
        row["delta_r2"] = row["regime_r2"] - ar_r2
        detail_rows.append(row)
        log(
            f"  K={k}: persist={row['persistence']:.2f} self_trans={row['mean_self_transition']:.3f} "
            f"K_eff={row['K_eff']} R²={row['regime_r2']:.4f} ΔR²={row['delta_r2']:+.4f} BIC={row['bic']:.1f}"
        )

    log(f"\n[{model_tag}] === Shuffled order ({shuffle_mode}) ===")
    for shuffle_seed in shuffle_seeds:
        z_shuf, p_shuf, lab_shuf = shuffle_sequence_lists(
            cebra_seqs,
            pca_seqs,
            labels,
            seed=shuffle_seed,
            mode=shuffle_mode,
            block_size=block_size,
        )
        for k in k_values:
            fit = fit_and_evaluate(z_shuf, p_shuf, lab_shuf, k)
            row = _metrics_row("shuffled", k, shuffle_seed, fit)
            row["delta_r2"] = row["regime_r2"] - ar_r2
            row["shuffle_mode"] = shuffle_mode
            detail_rows.append(row)
        log(f"  seed={shuffle_seed}: done K={k_values[0]}..{k_values[-1]}")

    summary = {
        "model_tag": model_tag,
        "features_path": str(features_path),
        "limit_problems": limit_problems,
        "shuffle_mode": shuffle_mode,
        "block_size": block_size,
        "shuffle_seeds": shuffle_seeds,
        "k_values": k_values,
        "ar_r2": float(ar_r2),
        "n_trajectories": len(cebra_seqs),
        "per_run": detail_rows,
        "comparison": [],
    }

    for k in k_values:
        real = next(r for r in detail_rows if r["order"] == "real" and r["K"] == k)
        agg = _aggregate_shuffled(detail_rows, k)
        if not agg:
            continue
        comp = {
            "K": k,
            "real": {key: real[key] for key in real if key not in ("order", "shuffle_seed", "shuffle_mode")},
            "shuffled": agg,
            "delta_real_minus_shuffled_mean": {
                "persistence": real["persistence"] - agg["persistence_mean"],
                "mean_self_transition": real["mean_self_transition"] - agg["mean_self_transition_mean"],
                "regime_r2": real["regime_r2"] - agg["regime_r2_mean"],
                "delta_r2": real["delta_r2"] - (agg["regime_r2_mean"] - ar_r2),
                "bic": real["bic"] - agg["bic_mean"],
            },
        }
        summary["comparison"].append(comp)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        csv_path = out_dir / "summary.csv"
        if detail_rows:
            fieldnames = list(detail_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(detail_rows)
        log(f"\n[{model_tag}] Wrote {summary_path}")
        log(f"[{model_tag}] Wrote {csv_path}")
        log(f"\n[{model_tag}] Real vs shuffled (mean over seeds):")
        for comp in summary["comparison"]:
            d = comp["delta_real_minus_shuffled_mean"]
            log(
                f"  K={comp['K']}: Δpersist={d['persistence']:+.3f} "
                f"Δself_trans={d['mean_self_transition']:+.3f} ΔR²={d['regime_r2']:+.4f}"
            )

    return summary


def main():
    args = parse_args()
    k_values = list(args.k_values) if args.k_values is not None else list(K_SWEEP)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / run_name
    execute_ablation(
        args.features_path,
        limit_problems=args.limit_problems,
        cebra_mode=args.cebra_mode,
        shuffle_mode=args.shuffle_mode,
        block_size=args.block_size,
        shuffle_seeds=args.shuffle_seeds,
        k_values=k_values,
        cebra_cache=args.cebra_cache,
        out_dir=out_dir,
        model_tag="single",
        verbose=True,
    )


if __name__ == "__main__":
    main()
