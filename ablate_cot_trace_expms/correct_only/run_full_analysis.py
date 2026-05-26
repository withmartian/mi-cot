#!/usr/bin/env python3
"""
Run base+RFT correct-trace ablations for all (or selected) model×dataset combos;
aggregate metrics into CSV + plot; report whether desired outcomes are met.

Example:
  python run_full_analysis.py --hf-cache-dir ./hf_cache --datasets gsm8k math500 \\
      --model-families qwen14 --k-values 5 --skip-run  # evaluate existing only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_SCRIPT = SCRIPT_DIR / "run_compare_base_rft.py"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from config import DEFAULT_LIMIT_PROBLEMS, HF_REPOS, MODEL_PAIRS, all_combo_keys  # noqa: E402
from evaluate import evaluate_compare_summaries  # noqa: E402
from pipeline import metric_at_k  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Full correct-trace ablation sweep + summary plot.")
    p.add_argument("--datasets", nargs="+", default=None, choices=list(HF_REPOS))
    p.add_argument("--model-families", nargs="+", default=None, choices=list(MODEL_PAIRS))
    p.add_argument("--hf-cache-dir", type=Path, default=Path(__file__).parent / "hf_cache")
    p.add_argument("--limit-problems", type=int, default=None)
    p.add_argument("--k-values", type=int, nargs="+", default=[5])
    p.add_argument("--k-focus", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results" / "full_analysis")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--skip-run", action="store_true", help="Only aggregate; expect compare runs under out-dir")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    p.add_argument(
        "--manifest-cache-dir",
        type=Path,
        default=None,
        help="Passed to run_compare_base_rft (default: correct_only/manifest_cache)",
    )
    p.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Force re-grade and refresh manifest cache for all compare runs.",
    )
    p.add_argument(
        "--no-use-cached-manifest",
        action="store_true",
        help="Disable manifest cache reads for compare subprocesses.",
    )
    return p.parse_args()


def _gap_row(
    dataset: str,
    model_family: str,
    subset: str,
    metric: str,
    base_summary: dict,
    rft_summary: dict,
    k: int,
) -> dict | None:
    b = metric_at_k(base_summary, k, metric)
    r = metric_at_k(rft_summary, k, metric)
    if b is None or r is None:
        return None
    return {
        "dataset": dataset,
        "model_family": model_family,
        "subset": subset,
        "metric": metric,
        "K": k,
        "base": b,
        "rft": r,
        "gap_rft_minus_base": r - b,
    }


def run_compare_subprocess(
    dataset: str,
    model_family: str,
    *,
    hf_cache_dir: Path,
    limit: int | None,
    k_values: list[int],
    k_focus: int,
    out_dir: Path,
    run_name: str,
    dry_run: bool,
    manifest_cache_dir: Path | None = None,
    rebuild_manifest: bool = False,
    use_cached_manifest: bool = True,
) -> int:
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--dataset",
        dataset,
        "--model-family",
        model_family,
        "--hf-cache-dir",
        str(hf_cache_dir),
        "--k-focus",
        str(k_focus),
        "--k-values",
        *[str(k) for k in k_values],
        "--out-dir",
        str(out_dir / "compare"),
        "--run-name",
        run_name,
    ]
    if manifest_cache_dir is not None:
        cmd.extend(["--manifest-cache-dir", str(manifest_cache_dir)])
    if rebuild_manifest:
        cmd.append("--rebuild-manifest")
    if not use_cached_manifest:
        cmd.append("--no-use-cached-manifest")
    if limit is not None:
        cmd.extend(["--limit-problems", str(limit)])
    print(" ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def collect_rows(compare_root: Path, k_focus: int) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    eval_reports: list[dict] = []

    for report_path in sorted(compare_root.rglob("comparison_report.json")):
        combo_dir = report_path.parent
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        eval_reports.append(report)

        dataset = report["dataset"]
        model_family = report["model_family"]

        for subset in ("all", "within_correct", "paired_correct"):
            base_p = combo_dir / "base" / subset / "summary.json"
            rft_p = combo_dir / "rft" / subset / "summary.json"
            if not base_p.is_file() or not rft_p.is_file():
                continue
            with open(base_p, encoding="utf-8") as f:
                base_s = json.load(f)
            with open(rft_p, encoding="utf-8") as f:
                rft_s = json.load(f)
            for metric in ("persistence", "mean_self_transition", "delta_r2", "K_eff"):
                row = _gap_row(dataset, model_family, subset, metric, base_s, rft_s, k_focus)
                if row:
                    rows.append(row)

    return pd.DataFrame(rows), eval_reports


def plot_gaps(df: pd.DataFrame, out_path: Path, k_focus: int) -> None:
    if df.empty:
        return
    plot_metrics = ["persistence", "delta_r2"]
    subdf = df[df["metric"].isin(plot_metrics) & (df["K"] == k_focus)]
    if subdf.empty:
        return

    combos = subdf.groupby(["dataset", "model_family"]).size().reset_index()[
        ["dataset", "model_family"]
    ]
    n = len(combos)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    subsets = ["all", "within_correct", "paired_correct"]
    colors = {"all": "#4C72B0", "within_correct": "#DD8452", "paired_correct": "#55A868"}

    for ax, (_, row) in zip(axes[0], combos.iterrows()):
        d, m = row["dataset"], row["model_family"]
        block = subdf[(subdf["dataset"] == d) & (subdf["model_family"] == m)]
        x = list(range(len(subsets)))
        for i, subset in enumerate(subsets):
            for metric, offset in zip(plot_metrics, (-0.15, 0.15)):
                val = block[(block["subset"] == subset) & (block["metric"] == metric)]
                if val.empty:
                    continue
                gap = float(val["gap_rft_minus_base"].iloc[0])
                ax.bar(
                    i + offset,
                    gap,
                    width=0.25,
                    color=colors.get(subset, "gray"),
                    alpha=0.85 if metric == "persistence" else 0.55,
                    label=f"{subset}" if metric == "persistence" else None,
                )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in subsets], fontsize=8)
        ax.set_title(f"{d}\n{m}")
        ax.set_ylabel("RFT − Base")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Correct-trace ablation: RFT−Base gap (K={k_focus})", y=1.08)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    datasets = args.datasets or list(HF_REPOS)
    families = args.model_families or list(MODEL_PAIRS)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    analysis_dir = args.out_dir / run_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        failures = 0
        for dataset, model_family in all_combo_keys():
            if dataset not in datasets or model_family not in families:
                continue
            rc = run_compare_subprocess(
                dataset,
                model_family,
                hf_cache_dir=args.hf_cache_dir,
                limit=args.limit_problems,
                k_values=args.k_values,
                k_focus=args.k_focus,
                out_dir=analysis_dir,
                run_name=run_name,
                dry_run=args.dry_run,
                manifest_cache_dir=args.manifest_cache_dir,
                rebuild_manifest=args.rebuild_manifest,
                use_cached_manifest=not args.no_use_cached_manifest,
            )
            if rc != 0:
                failures += 1
        if args.dry_run:
            return 0
        if failures:
            print(f"Warning: {failures} compare run(s) returned non-zero exit.", flush=True)

    compare_root = analysis_dir / "compare" / run_name
    df, eval_reports = collect_rows(compare_root, args.k_focus)

    csv_path = analysis_dir / "gap_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)", flush=True)

    plot_path = analysis_dir / "gap_by_subset.png"
    plot_gaps(df, plot_path, args.k_focus)
    if plot_path.is_file():
        print(f"Wrote {plot_path}", flush=True)

    summary_rows = []
    for report in eval_reports:
        ev = report.get("evaluation", {})
        summary_rows.append(
            {
                "combo": f"{report['dataset']}_{report['model_family']}",
                "primary_pass": report.get("overall_pass", False),
                "all_checks_pass": report.get("all_checks_pass", False),
                "base_acc": report.get("base_manifest_stats", {}).get("accuracy"),
                "rft_acc": report.get("rft_manifest_stats", {}).get("accuracy"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = analysis_dir / "outcome_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    master = {
        "run_name": run_name,
        "k_focus": args.k_focus,
        "n_combos": len(summary_rows),
        "n_primary_pass": int(summary_df["primary_pass"].sum()) if not summary_df.empty else 0,
        "desired_outcome": (
            "RFT > Base on paired_correct for persistence and delta_r2 at k_focus"
        ),
        "combos": summary_rows,
        "eval_reports": [
            {
                "combo": f"{r['dataset']}_{r['model_family']}",
                "overall_pass": r.get("overall_pass"),
                "checks": r.get("evaluation", {}).get("checks", []),
            }
            for r in eval_reports
        ],
    }
    master_path = analysis_dir / "master_report.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    print(f"\nWrote {summary_path}", flush=True)
    print(f"Wrote {master_path}", flush=True)

    if summary_df.empty:
        print("No comparison reports found.", flush=True)
        return 1

    all_primary = bool(summary_df["primary_pass"].all())
    print(
        f"\nPrimary outcome across combos: "
        f"{int(summary_df['primary_pass'].sum())}/{len(summary_df)} passed",
        flush=True,
    )
    print(summary_df.to_string(index=False), flush=True)
    return 0 if all_primary else 1


if __name__ == "__main__":
    sys.exit(main())
