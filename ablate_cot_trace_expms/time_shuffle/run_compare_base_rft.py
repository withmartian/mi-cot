#!/usr/bin/env python3
"""
Run time-shuffle ablation on base and RFT feature pickles, then check desired outcomes.

Each pickle run produces real-order and shuffled-order SDS metrics (two conditions per
model). This script runs both models and evaluates:

  1. Analyzer: real >> shuffled for each model (persistence, delta_r2, self-transition).
  2. Paper: RFT_real > Base_real on key metrics.
  3. Temporal gap: (RFT - Base)_real > (RFT - Base)_shuffled.

Exit code 0 if all checks pass at --k-focus (default 5); 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_time_shuffle import execute_ablation  # noqa: E402

METRICS_HIGHER_IS_RFT_BETTER = ("persistence", "mean_self_transition", "delta_r2", "K_eff")
METRICS_COLLAPSE_UNDER_SHUFFLE = ("persistence", "mean_self_transition", "delta_r2")


def _metric_at_k(summary: dict, k: int, order: str, field: str) -> float | None:
    """order: 'real' or shuffled mean field suffix _mean."""
    if order == "real":
        for row in summary["per_run"]:
            if row["order"] == "real" and row["K"] == k:
                return float(row[field])
        return None
    for comp in summary["comparison"]:
        if comp["K"] != k:
            continue
        shuf = comp["shuffled"]
        if field == "delta_r2":
            if "regime_r2_mean" in shuf:
                return float(shuf["regime_r2_mean"]) - float(summary["ar_r2"])
            key = f"{field}_mean"
            return float(shuf[key]) if key in shuf else None
        key = f"{field}_mean"
        return float(shuf[key]) if key in shuf else None
    return None


def evaluate_pair(
    base_summary: dict,
    rft_summary: dict,
    k: int,
    *,
    min_real_vs_shuf_ratio: float = 1.05,
    min_gap_shrink_ratio: float = 1.05,
    epsilon: float = 1e-9,
) -> dict:
    """
    Check desired outcomes at regime count K.

    min_real_vs_shuf_ratio: real metric must be >= this * shuffled (per model).
    min_gap_shrink_ratio: gap_real must be >= this * gap_shuffled (strict > if ratio 1.0).
    """
    checks: list[dict] = []

    def add_check(name: str, passed: bool, detail: str, values: dict | None = None) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail, "values": values or {}})

    for model_tag, summary in (("base", base_summary), ("rft", rft_summary)):
        for metric in METRICS_COLLAPSE_UNDER_SHUFFLE:
            real_v = _metric_at_k(summary, k, "real", metric)
            shuf_v = _metric_at_k(summary, k, "shuffled", metric)
            if real_v is None or shuf_v is None:
                add_check(
                    f"{model_tag}_{metric}_collapse",
                    False,
                    f"missing data for K={k}",
                    {"real": real_v, "shuffled": shuf_v},
                )
                continue
            # delta_r2 can be negative; use absolute drop or ratio only when shuf near zero
            if metric == "delta_r2":
                passed = real_v > shuf_v + 0.01
            else:
                passed = real_v >= min_real_vs_shuf_ratio * (shuf_v + epsilon)
            add_check(
                f"{model_tag}_{metric}_real_gt_shuffled",
                passed,
                f"{model_tag} real={real_v:.4f} vs shuffled={shuf_v:.4f}",
                {"real": real_v, "shuffled": shuf_v},
            )

    for metric in METRICS_HIGHER_IS_RFT_BETTER:
        b_real = _metric_at_k(base_summary, k, "real", metric)
        r_real = _metric_at_k(rft_summary, k, "real", metric)
        if b_real is None or r_real is None:
            add_check(f"rft_gt_base_real_{metric}", False, f"missing real-order data K={k}")
            continue
        add_check(
            f"rft_gt_base_real_{metric}",
            r_real > b_real,
            f"RFT_real={r_real:.4f} vs Base_real={b_real:.4f}",
            {"rft": r_real, "base": b_real},
        )

    for metric in ("persistence", "delta_r2", "mean_self_transition"):
        gap_real = None
        gap_shuf = None
        b_r = _metric_at_k(base_summary, k, "real", metric)
        r_r = _metric_at_k(rft_summary, k, "real", metric)
        b_s = _metric_at_k(base_summary, k, "shuffled", metric)
        r_s = _metric_at_k(rft_summary, k, "shuffled", metric)
        if None not in (b_r, r_r, b_s, r_s):
            gap_real = r_r - b_r
            gap_shuf = r_s - b_s
            if metric == "delta_r2":
                passed = gap_real > gap_shuf + 0.01
            else:
                passed = gap_real >= min_gap_shrink_ratio * (gap_shuf + epsilon)
            add_check(
                f"gap_real_gt_gap_shuffled_{metric}",
                passed,
                f"gap_real={gap_real:.4f} vs gap_shuffled={gap_shuf:.4f}",
                {"gap_real": gap_real, "gap_shuffled": gap_shuf},
            )
        else:
            add_check(f"gap_real_gt_gap_shuffled_{metric}", False, "missing metrics for gap comparison")

    all_passed = all(c["passed"] for c in checks)
    return {"K": k, "all_passed": all_passed, "checks": checks}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run base+RFT time-shuffle ablations and evaluate desired outcomes.",
    )
    p.add_argument("--base-features-path", type=Path, required=True)
    p.add_argument("--rft-features-path", type=Path, required=True)
    p.add_argument("--limit-problems", type=int, default=500)
    p.add_argument("--shuffle-mode", default="full", choices=["full", "block"])
    p.add_argument("--block-size", type=int, default=3)
    p.add_argument("--shuffle-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[4, 5, 6],
        help="K values to fit; checks run at --k-focus",
    )
    p.add_argument("--k-focus", type=int, default=5, help="K used for pass/fail criteria")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "compare_base_rft",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--skip-run",
        action="store_true",
        help="Only evaluate existing summaries under out-dir/run-name/{base,rft}/",
    )
    p.add_argument(
        "--min-real-vs-shuf-ratio",
        type=float,
        default=1.05,
        help="Real metric must be at least this times shuffled (per model)",
    )
    p.add_argument(
        "--min-gap-shrink-ratio",
        type=float,
        default=1.0,
        help="Require gap_real >= ratio * gap_shuffled (use 1.05 for strict)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_name
    base_out = run_dir / "base"
    rft_out = run_dir / "rft"

    if not args.skip_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        print("=== Base model (real + shuffled SDS) ===", flush=True)
        base_summary = execute_ablation(
            args.base_features_path,
            limit_problems=args.limit_problems,
            shuffle_mode=args.shuffle_mode,
            block_size=args.block_size,
            shuffle_seeds=args.shuffle_seeds,
            k_values=args.k_values,
            cebra_cache=run_dir / "cebra_cache_base.pkl",
            out_dir=base_out,
            model_tag="base",
        )
        print("\n=== RFT / reasoning model (real + shuffled SDS) ===", flush=True)
        rft_summary = execute_ablation(
            args.rft_features_path,
            limit_problems=args.limit_problems,
            shuffle_mode=args.shuffle_mode,
            block_size=args.block_size,
            shuffle_seeds=args.shuffle_seeds,
            k_values=args.k_values,
            cebra_cache=run_dir / "cebra_cache_rft.pkl",
            out_dir=rft_out,
            model_tag="rft",
        )
    else:
        with open(base_out / "summary.json", encoding="utf-8") as f:
            base_summary = json.load(f)
        with open(rft_out / "summary.json", encoding="utf-8") as f:
            rft_summary = json.load(f)

    if args.k_focus not in args.k_values:
        print(
            f"Warning: --k-focus {args.k_focus} not in --k-values {args.k_values}; "
            "evaluation may use missing K.",
            flush=True,
        )

    evaluation = evaluate_pair(
        base_summary,
        rft_summary,
        args.k_focus,
        min_real_vs_shuf_ratio=args.min_real_vs_shuf_ratio,
        min_gap_shrink_ratio=args.min_gap_shrink_ratio,
    )

    # Optional: evaluate all K in k_values for reporting
    by_k = {
        str(k): evaluate_pair(
            base_summary,
            rft_summary,
            k,
            min_real_vs_shuf_ratio=args.min_real_vs_shuf_ratio,
            min_gap_shrink_ratio=args.min_gap_shrink_ratio,
        )
        for k in args.k_values
    }

    report = {
        "run_name": run_name,
        "base_features_path": str(args.base_features_path),
        "rft_features_path": str(args.rft_features_path),
        "k_focus": args.k_focus,
        "k_values": args.k_values,
        "desired_outcome_summary": (
            "RFT_real > Base_real; both real >> shuffled; (RFT-Base)_real > (RFT-Base)_shuffled"
        ),
        "evaluation_at_k_focus": evaluation,
        "evaluation_by_k": by_k,
        "overall_pass": evaluation["all_passed"],
    }

    report_path = run_dir / "comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nWrote {report_path}", flush=True)
    print(f"\n=== Evaluation at K={args.k_focus} ===", flush=True)
    for c in evaluation["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['name']}: {c['detail']}", flush=True)

    print(
        f"\nOverall: {'PASS' if evaluation['all_passed'] else 'FAIL'} "
        f"(desired outcome {'met' if evaluation['all_passed'] else 'not met'})",
        flush=True,
    )
    return 0 if evaluation["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
