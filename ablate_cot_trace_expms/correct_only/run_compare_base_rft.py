#!/usr/bin/env python3
"""
Run correct-trace ablations for base + RFT on one dataset; evaluate desired outcomes.

Subsets per model:
  - all
  - within_correct
  - paired_correct (same problem_ids where both base and RFT are correct)

Example:
  python run_compare_base_rft.py --dataset gsm8k --model-family qwen14 \\
      --hf-cache-dir ./hf_cache
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

from config import DEFAULT_LIMIT_PROBLEMS, resolve_paths  # noqa: E402
from evaluate import evaluate_compare_summaries  # noqa: E402
from hf_utils import ensure_model_artifacts  # noqa: E402
from manifest import (  # noqa: E402
    DEFAULT_MANIFEST_CACHE_DIR,
    add_manifest_cache_args,
    copy_manifest_to_run_dir,
    get_or_build_manifest,
    get_or_build_paired_index,
)
from pipeline import execute_correct_subset  # noqa: E402
from subsets import resolve_subset_pids  # noqa: E402

COMPARE_SUBSETS = ("all", "within_correct", "paired_correct")


def parse_args():
    p = argparse.ArgumentParser(description="Base vs RFT correct-trace ablation.")
    p.add_argument("--dataset", required=True, choices=list(DEFAULT_LIMIT_PROBLEMS))
    p.add_argument("--model-family", required=True, choices=["llama8", "qwen14", "qwen1.5"])
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--base-features-path", type=Path, default=None)
    p.add_argument("--base-cot-data-path", type=Path, default=None)
    p.add_argument("--rft-features-path", type=Path, default=None)
    p.add_argument("--rft-cot-data-path", type=Path, default=None)
    p.add_argument("--hf-cache-dir", type=Path, default=None)
    p.add_argument("--limit-problems", type=int, default=None)
    p.add_argument("--k-values", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--k-focus", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results" / "compare")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--skip-run", action="store_true", help="Load existing summaries from out-dir")
    p.add_argument("--min-paired-gap-fraction", type=float, default=0.5)
    add_manifest_cache_args(p)
    return p.parse_args()


def _run_model_subsets(
    variant: str,
    features_path: Path,
    cot_path: Path,
    manifest: dict,
    base_manifest: dict,
    rft_manifest: dict,
    *,
    limit: int,
    k_values: list[int],
    run_dir: Path,
) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    for subset in COMPARE_SUBSETS:
        pids = resolve_subset_pids(
            subset,
            manifest=manifest,
            limit_problems=limit,
            base_manifest=base_manifest,
            rft_manifest=rft_manifest,
        )
        print(f"  [{variant}] subset={subset}: n_pids={len(pids)}", flush=True)
        if len(pids) < 3:
            print(f"    SKIP (too few)", flush=True)
            continue
        sub_out = run_dir / variant / subset
        summaries[subset] = execute_correct_subset(
            features_path,
            pids,
            subset_name=subset,
            limit_problems=limit,
            k_values=k_values,
            cebra_cache=sub_out / "cebra_cache.pkl",
            out_dir=sub_out,
            model_tag=variant,
        )
    return summaries


def main() -> int:
    args = parse_args()
    limit = args.limit_problems or DEFAULT_LIMIT_PROBLEMS[args.dataset]
    base_paths = resolve_paths(args.dataset, args.model_family, "base", layer=args.layer)
    rft_paths = resolve_paths(args.dataset, args.model_family, "rft", layer=args.layer)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_name / f"{args.dataset}_{args.model_family}"

    base_summaries: dict[str, dict] = {}
    rft_summaries: dict[str, dict] = {}
    cache_dir = args.manifest_cache_dir or DEFAULT_MANIFEST_CACHE_DIR
    base_cache_path: str | None = None
    rft_cache_path: str | None = None
    paired_cache_path: str | None = None

    if not args.skip_run:
        if args.base_features_path and args.base_cot_data_path:
            base_feat, base_cot = args.base_features_path, args.base_cot_data_path
        else:
            base_feat, base_cot = ensure_model_artifacts(base_paths, cache_dir=args.hf_cache_dir)

        if args.rft_features_path and args.rft_cot_data_path:
            rft_feat, rft_cot = args.rft_features_path, args.rft_cot_data_path
        else:
            rft_feat, rft_cot = ensure_model_artifacts(rft_paths, cache_dir=args.hf_cache_dir)

        run_dir.mkdir(parents=True, exist_ok=True)
        mkwargs = dict(
            cache_dir=cache_dir,
            use_cached=args.use_cached_manifest,
            rebuild=args.rebuild_manifest,
        )

        print("=== Base manifest ===", flush=True)
        base_manifest, base_cache, base_cached = get_or_build_manifest(
            base_cot,
            args.dataset,
            limit_problems=limit,
            model_folder=base_paths.folder,
            variant="base",
            layer=base_paths.layer,
            **mkwargs,
        )
        base_cache_path = str(base_cache)
        copy_manifest_to_run_dir(base_cache, run_dir / "base_manifest.json")
        print(
            f"  base accuracy={base_manifest['accuracy']:.3f} "
            f"({base_manifest['n_correct']}/{base_manifest['n_problems']}) "
            f"[{'cache' if base_cached else 'built'}]",
            flush=True,
        )

        print("=== RFT manifest ===", flush=True)
        rft_manifest, rft_cache, rft_cached = get_or_build_manifest(
            rft_cot,
            args.dataset,
            limit_problems=limit,
            model_folder=rft_paths.folder,
            variant="rft",
            layer=rft_paths.layer,
            **mkwargs,
        )
        rft_cache_path = str(rft_cache)
        copy_manifest_to_run_dir(rft_cache, run_dir / "rft_manifest.json")
        print(
            f"  rft accuracy={rft_manifest['accuracy']:.3f} "
            f"({rft_manifest['n_correct']}/{rft_manifest['n_problems']}) "
            f"[{'cache' if rft_cached else 'built'}]",
            flush=True,
        )

        paired_list, paired_cache, paired_cached = get_or_build_paired_index(
            base_manifest,
            rft_manifest,
            dataset_key=args.dataset,
            model_family=args.model_family,
            layer=base_paths.layer,
            limit_problems=limit,
            cache_dir=cache_dir,
            use_cached=args.use_cached_manifest,
            rebuild=args.rebuild_manifest,
        )
        paired_cache_path = str(paired_cache) if paired_cache else None
        print(
            f"  paired_correct n={len(paired_list)} "
            f"[{'cache' if paired_cached else 'built'} -> {paired_cache}]",
            flush=True,
        )

        print("\n=== Base model SDS ===", flush=True)
        base_summaries = _run_model_subsets(
            "base",
            base_feat,
            base_cot,
            base_manifest,
            base_manifest,
            rft_manifest,
            limit=limit,
            k_values=args.k_values,
            run_dir=run_dir,
        )

        print("\n=== RFT model SDS ===", flush=True)
        rft_summaries = _run_model_subsets(
            "rft",
            rft_feat,
            rft_cot,
            rft_manifest,
            base_manifest,
            rft_manifest,
            limit=limit,
            k_values=args.k_values,
            run_dir=run_dir,
        )
    else:
        with open(run_dir / "base_manifest.json", encoding="utf-8") as f:
            base_manifest = json.load(f)
        with open(run_dir / "rft_manifest.json", encoding="utf-8") as f:
            rft_manifest = json.load(f)
        for variant in ("base", "rft"):
            store = base_summaries if variant == "base" else rft_summaries
            for subset in COMPARE_SUBSETS:
                p = run_dir / variant / subset / "summary.json"
                if p.is_file():
                    with open(p, encoding="utf-8") as f:
                        store[subset] = json.load(f)

    evaluation = evaluate_compare_summaries(
        base_summaries,
        rft_summaries,
        args.k_focus,
        min_paired_gap_fraction=args.min_paired_gap_fraction,
    )

    report = {
        "dataset": args.dataset,
        "model_family": args.model_family,
        "layer": base_paths.layer,
        "limit_problems": limit,
        "k_focus": args.k_focus,
        "k_values": args.k_values,
        "base_manifest_stats": {
            "accuracy": base_manifest.get("accuracy"),
            "n_correct": base_manifest.get("n_correct"),
        },
        "rft_manifest_stats": {
            "accuracy": rft_manifest.get("accuracy"),
            "n_correct": rft_manifest.get("n_correct"),
        },
        "evaluation": evaluation,
        "overall_pass": evaluation["primary_pass"],
        "all_checks_pass": evaluation["all_passed"],
        "manifest_cache_dir": str(cache_dir),
        "base_manifest_cache": base_cache_path,
        "rft_manifest_cache": rft_cache_path,
        "paired_index_cache": paired_cache_path,
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
        f"\nPrimary outcome (paired_correct RFT>Base): "
        f"{'PASS' if evaluation['primary_pass'] else 'FAIL'}",
        flush=True,
    )
    print(
        f"All checks (incl. paired vs all gap): {'PASS' if evaluation['all_passed'] else 'FAIL'}",
        flush=True,
    )
    return 0 if evaluation["primary_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
