#!/usr/bin/env python3
"""
Run correct-trace ablation for one model variant on one dataset.

Builds a correctness manifest from cot_data.pkl, then runs CEBRA+SDS on
subsets: all, within_correct (and optionally others passed via --subsets).

Example:
  python run_single_model.py --dataset gsm8k --model-family qwen14 --variant base \\
      --features-path ./hf_cache/.../all_sentences_features.pkl \\
      --cot-data-path ./hf_cache/.../cot_data.pkl
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
from hf_utils import ensure_model_artifacts  # noqa: E402
from manifest import (  # noqa: E402
    DEFAULT_MANIFEST_CACHE_DIR,
    add_manifest_cache_args,
    copy_manifest_to_run_dir,
    get_or_build_manifest,
)
from pipeline import execute_correct_subset  # noqa: E402
from subsets import resolve_subset_pids  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Correct-trace SDS ablation (single model).")
    p.add_argument("--dataset", required=True, choices=list(DEFAULT_LIMIT_PROBLEMS))
    p.add_argument("--model-family", required=True, choices=["llama8", "qwen14", "qwen1.5"])
    p.add_argument("--variant", required=True, choices=["base", "rft"])
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--features-path", type=Path, default=None)
    p.add_argument("--cot-data-path", type=Path, default=None)
    p.add_argument("--hf-cache-dir", type=Path, default=None, help="Download HF files here if paths missing")
    p.add_argument("--limit-problems", type=int, default=None)
    p.add_argument(
        "--subsets",
        nargs="+",
        default=["all", "within_correct"],
        choices=["all", "within_correct"],
    )
    p.add_argument("--k-values", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "results" / "single_model")
    p.add_argument("--run-name", type=str, default=None)
    add_manifest_cache_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    limit = args.limit_problems or DEFAULT_LIMIT_PROBLEMS[args.dataset]
    paths = resolve_paths(args.dataset, args.model_family, args.variant, layer=args.layer)

    if args.features_path and args.cot_data_path:
        features_path = args.features_path
        cot_path = args.cot_data_path
    else:
        features_path, cot_path = ensure_model_artifacts(paths, cache_dir=args.hf_cache_dir)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_name / paths.model_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.manifest_cache_dir or DEFAULT_MANIFEST_CACHE_DIR
    print(f"Loading/building manifest for {cot_path}...", flush=True)
    manifest, cache_path, from_cache = get_or_build_manifest(
        cot_path,
        args.dataset,
        limit_problems=limit,
        model_folder=paths.folder,
        variant=args.variant,
        layer=paths.layer,
        cache_dir=cache_dir,
        use_cached=args.use_cached_manifest,
        rebuild=args.rebuild_manifest,
    )
    manifest_path = run_dir / "manifest.json"
    copy_manifest_to_run_dir(cache_path, manifest_path)
    print(
        f"  accuracy={manifest['accuracy']:.3f} "
        f"({manifest['n_correct']}/{manifest['n_problems']}) "
        f"[{'cache' if from_cache else 'built'} -> {cache_path}]",
        flush=True,
    )

    summaries = {}
    for subset in args.subsets:
        pids = resolve_subset_pids(
            subset,
            manifest=manifest,
            limit_problems=limit,
        )
        print(f"\n=== Subset '{subset}': {len(pids)} problems ===", flush=True)
        if len(pids) < 3:
            print(f"  SKIP: too few problems ({len(pids)})", flush=True)
            continue
        sub_out = run_dir / subset
        summary = execute_correct_subset(
            features_path,
            pids,
            subset_name=subset,
            limit_problems=limit,
            k_values=args.k_values,
            cebra_cache=sub_out / "cebra_cache.pkl",
            out_dir=sub_out,
            model_tag=paths.model_tag,
        )
        summaries[subset] = summary

    report = {
        "dataset": args.dataset,
        "model_family": args.model_family,
        "variant": args.variant,
        "layer": paths.layer,
        "limit_problems": limit,
        "manifest_path": str(manifest_path),
        "manifest_cache_path": str(cache_path),
        "manifest_from_cache": from_cache,
        "features_path": str(features_path),
        "subsets": summaries,
    }
    report_path = run_dir / "run_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
