#!/usr/bin/env python3
"""Aggregate comparison_report.json files into a summary table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def load_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--compare-dirs",
        nargs="+",
        type=Path,
        default=[
            SCRIPT_DIR / "results" / "compare" / "middle_layer_sweep",
            SCRIPT_DIR / "results" / "compare" / "qwen15_math500_first_test",
        ],
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=SCRIPT_DIR / "results" / "compare" / "middle_layer_sweep_summary.csv",
    )
    args = p.parse_args()

    rows = []
    for compare_dir in args.compare_dirs:
        if not compare_dir.is_dir():
            continue
        for report_path in sorted(compare_dir.glob("*/comparison_report.json")):
            r = load_report(report_path)
            ev = r.get("evaluation", {})
            checks = {c["name"]: c for c in ev.get("checks", [])}

            def metric_check(subset: str, metric: str) -> str | None:
                key = f"rft_gt_base_{subset}_{metric}"
                c = checks.get(key)
                if not c:
                    return None
                return "PASS" if c["passed"] else "FAIL"

            rows.append(
                {
                    "dataset": r["dataset"],
                    "model_family": r["model_family"],
                    "layer": r["layer"],
                    "run_dir": str(report_path.parent.name),
                    "base_acc": r.get("base_manifest_stats", {}).get("accuracy"),
                    "rft_acc": r.get("rft_manifest_stats", {}).get("accuracy"),
                    "primary_pass": ev.get("primary_pass"),
                    "all_pass": ev.get("all_passed"),
                    "paired_persist": metric_check("paired_correct", "persistence"),
                    "paired_self_trans": metric_check("paired_correct", "mean_self_transition"),
                    "paired_delta_r2": metric_check("paired_correct", "delta_r2"),
                    "paired_k_eff": metric_check("paired_correct", "K_eff"),
                    "gap_persist": metric_check("paired_correct", "persistence")
                    and checks.get("paired_retains_gap_persistence", {}).get("passed"),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "model_family"]).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {args.out_csv}")
    print(f"Primary pass: {df['primary_pass'].sum()}/{len(df)}")


if __name__ == "__main__":
    main()
