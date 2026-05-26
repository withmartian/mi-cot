"""CEBRA + SDS on filtered problem subsets."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from exploration.cebra_EM import (  # noqa: E402
    K_SWEEP,
    fit_and_evaluate,
    linear_ar_r2,
    load_and_prepare_cebra,
    train_cebra_projection,
)


def _metrics_row(subset: str, k: int, fit_tuple) -> dict:
    persist, mean_st, spec, _, _, score, k_eff, r2, bic = fit_tuple
    return {
        "subset": subset,
        "K": k,
        "persistence": float(persist),
        "mean_self_transition": float(mean_st),
        "specialization": float(spec),
        "sss": float(score),
        "K_eff": int(k_eff),
        "regime_r2": float(r2),
        "bic": float(bic),
    }


def execute_correct_subset(
    features_path: Path,
    allowed_problem_ids: set[int],
    *,
    subset_name: str = "custom",
    limit_problems: int = 500,
    k_values: list[int] | None = None,
    cebra_cache: Path | None = None,
    out_dir: Path | None = None,
    model_tag: str = "model",
    verbose: bool = True,
) -> dict:
    """Train CEBRA on allowed problems only; fit SDS; return summary dict."""
    if k_values is None:
        k_values = list(K_SWEEP)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    allowed_list = sorted(allowed_problem_ids)
    cache_key = (subset_name, tuple(allowed_list), limit_problems)

    cache = None
    if cebra_cache and cebra_cache.exists():
        log(f"[{model_tag}] Loading CEBRA cache from {cebra_cache}")
        with open(cebra_cache, "rb") as f:
            loaded = pickle.load(f)
        if loaded.get("cache_key") != cache_key:
            log(f"[{model_tag}] Cache key mismatch; retraining CEBRA.")
        else:
            cache = loaded
            cebra_seqs = cache["cebra_seqs"]
            pca_seqs = cache["pca_seqs"]
            labels = cache["labels"]
            ar_r2 = cache["ar_r2"]

    if cache is None:
        log(
            f"[{model_tag}] Loading features (subset={subset_name}, "
            f"n_pids={len(allowed_list)})..."
        )
        all_f, triplets = load_and_prepare_cebra(
            str(features_path),
            limit_problems=limit_problems,
            allowed_problem_ids=allowed_list,
        )
        if len(all_f) < 10:
            raise ValueError(
                f"Too few features ({len(all_f)}) for subset {subset_name}; "
                f"check allowed_problem_ids and limit_problems."
            )
        log(f"[{model_tag}] Training CEBRA ({len(all_f)} steps, {len(triplets)} triplets)...")
        cebra_seqs, pca_seqs, labels = train_cebra_projection(all_f, triplets)
        ar_r2 = linear_ar_r2(pca_seqs)
        log(f"  Trajectories: {len(cebra_seqs)}  AR R²: {ar_r2:.4f}")
        if cebra_cache:
            cebra_cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cebra_cache, "wb") as f:
                pickle.dump(
                    {
                        "cache_key": cache_key,
                        "cebra_seqs": cebra_seqs,
                        "pca_seqs": pca_seqs,
                        "labels": labels,
                        "ar_r2": ar_r2,
                        "features_path": str(features_path),
                        "subset_name": subset_name,
                        "n_allowed": len(allowed_list),
                    },
                    f,
                )

    detail_rows: list[dict] = []
    log(f"\n[{model_tag}] === SDS on subset '{subset_name}' ===")
    for k in k_values:
        fit = fit_and_evaluate(cebra_seqs, pca_seqs, labels, k)
        row = _metrics_row(subset_name, k, fit)
        row["delta_r2"] = row["regime_r2"] - ar_r2
        detail_rows.append(row)
        log(
            f"  K={k}: persist={row['persistence']:.2f} self_trans={row['mean_self_transition']:.3f} "
            f"K_eff={row['K_eff']} ΔR²={row['delta_r2']:+.4f}"
        )

    summary = {
        "model_tag": model_tag,
        "subset_name": subset_name,
        "features_path": str(features_path),
        "limit_problems": limit_problems,
        "n_allowed_pids": len(allowed_list),
        "n_trajectories": len(cebra_seqs),
        "k_values": k_values,
        "ar_r2": float(ar_r2),
        "per_run": detail_rows,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if detail_rows:
            with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
                w.writeheader()
                w.writerows(detail_rows)

    return summary


def metric_at_k(summary: dict, k: int, field: str) -> float | None:
    for row in summary["per_run"]:
        if row["K"] == k:
            return float(row[field])
    return None
