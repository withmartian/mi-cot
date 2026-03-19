#!/usr/bin/env python3
"""
Loop over model/layer feature pickles on the Hugging Face dataset
https://huggingface.co/datasets/withmartian/SDS_train_gsm8k

For each ``.../all_sentences_features.pkl`` (excluding *_with_neutral* variants),
runs ``cebra_em_steering_inputDep.run_pipeline`` and writes artifacts under::

    OUTPUT_ROOT/<dataset_subpath_up_to_layer>/

e.g. ``qwen1.5b_reasoning/layer_27/all_sentences_features.pkl`` →
``OUTPUT_ROOT/qwen1.5b_reasoning/layer_27/``.

**What gets looped**

- By default: *every* qualifying pickle in the repo (all architectures and layers),
  unless you filter with ``--dataset-core``, regexes, or ``--skip-ckpt-sweeps``.
- Each combination of **dataset subtree** (e.g. ``qwen1.5b_reasoning``) and **layer**
  (e.g. ``layer_27``) is one run — so you automatically sweep **all layers**
  published under that folder.

**DeepSeek R1 Distill (HF) vs dataset paths**

The dataset does *not* use ``deepseek-ai/DeepSeek-R1-Distill-*`` as directory names.
Reasoning traces use distilled “reasoning” subtrees; base subtrees are non-distill
activations in the same scale family.
See ``sds_train_gsm8k_hf.HF_DATASET_MODEL_PATH_ROWS`` for the mapping used with
``--dataset-core``.

Uses HF_TOKEN from the environment when set (optional for public files).

Example (only the 6 core base+reasoning trees; all their layers):
  python exploration/batch_cebra_steering_hf_dataset.py \\
    --output-dir exploration/sds_steering_batch \\
    --num-samples 100 \\
    --dataset-core \\
    --skip-ckpt-sweeps
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_here = os.path.abspath(os.path.dirname(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from cebra_em_steering_inputDep import SteerConfig, run_pipeline
from sds_train_gsm8k_hf import (
    HF_DATASET_MODEL_PATH_ROWS,
    SDS_TRAIN_GSM8K_REPO_ID,
    hub_path_matches_core_prefix,
)


def list_hub_feature_pickles(repo_id: str, token: Optional[str]) -> List[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    out: List[str] = []
    for f in files:
        if not f.endswith("all_sentences_features.pkl"):
            continue
        if "_with_neutral" in f:
            continue
        out.append(f)
    return sorted(out)


def hf_path_to_save_dir(output_root: str, hub_filename: str) -> str:
    """
    Map Hub path to a stable output folder.
    e.g. qwen1.5b_reasoning/layer_27/all_sentences_features.pkl
         -> OUTPUT_ROOT/qwen1.5b_reasoning/layer_27
    e.g. Qwen_14B_ckpts/qwen14b_sft_ckpt_final/layer_47/all_sentences_features.pkl
         -> OUTPUT_ROOT/Qwen_14B_ckpts/qwen14b_sft_ckpt_final/layer_47
    """
    rel = hub_filename.replace("\\", "/")
    if rel.endswith("/all_sentences_features.pkl"):
        rel = rel[: -len("/all_sentences_features.pkl")]
    elif rel.endswith("all_sentences_features.pkl"):
        rel = rel[: -len("all_sentences_features.pkl")].rstrip("/")
    return os.path.join(output_root, *rel.split("/"))


def cache_name_for_hub_path(num_samples: int, hub_filename: str) -> str:
    base = hub_filename.replace("\\", "/")
    base = base.replace("/all_sentences_features.pkl", "").replace("/", "_")
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)[:120]
    return f"subset_{num_samples}_{base}.pkl"


def _ckpt_sweep_path(p: str) -> bool:
    return p.startswith("Llama_8B_ckpts/") or p.startswith("Qwen_14B_ckpts/")


def filter_paths(
    paths: List[str],
    include: Optional[re.Pattern],
    exclude: Optional[re.Pattern],
    skip_ckpt_sweeps: bool,
    dataset_core_only: bool,
) -> List[str]:
    out = []
    for p in paths:
        if skip_ckpt_sweeps and _ckpt_sweep_path(p):
            continue
        if dataset_core_only and not hub_path_matches_core_prefix(p):
            continue
        if include is not None and not include.search(p):
            continue
        if exclude is not None and exclude.search(p):
            continue
        out.append(p)
    return out


def run_one(
    hub_filename: str,
    *,
    repo_id: str,
    output_root: str,
    num_samples: int,
    enable_openai_judge: bool,
    base_cfg: SteerConfig,
) -> Dict[str, Any]:
    save_dir = hf_path_to_save_dir(output_root, hub_filename)
    os.makedirs(save_dir, exist_ok=True)
    cfg = replace(
        base_cfg,
        hf_dataset_repo=repo_id,
        hf_dataset_filename=hub_filename,
        hf_fallback_num_samples=num_samples,
        hf_fallback_cache_name=cache_name_for_hub_path(num_samples, hub_filename),
        save_dir=os.path.abspath(save_dir),
        enable_openai_judge=enable_openai_judge,
        hf_auto_download_if_missing=True,
    )
    t0 = time.perf_counter()
    artifacts_path, payload = run_pipeline(cfg)
    elapsed = time.perf_counter() - t0
    g = payload.get("steering_eval", {}).get("global", {})
    return {
        "hub_filename": hub_filename,
        "save_dir": save_dir,
        "artifacts_path": artifacts_path,
        "seconds": round(elapsed, 3),
        "steerability_pct": g.get("steerability_pct"),
        "count": g.get("count"),
        "status": "ok",
        "error": None,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch CEBRA-EM steering over SDS_train_gsm8k (HF dataset) model/layer pickles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Root folder for per-model/layer subfolders.",
    )
    p.add_argument(
        "--repo",
        default=SDS_TRAIN_GSM8K_REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    p.add_argument("--num-samples", type=int, default=100, help="Non-NEUTRAL rows per run after Hub download.")
    p.add_argument(
        "--openai-judge",
        action="store_true",
        help="Enable OpenAI judge (requires OPENAI_API_KEY). Default: off for batch.",
    )
    p.add_argument(
        "--skip-ckpt-sweeps",
        action="store_true",
        help="Exclude Llama_8B_ckpts/* and Qwen_14B_ckpts/* (SFT checkpoint grids).",
    )
    p.add_argument(
        "--dataset-core",
        action="store_true",
        dest="dataset_core",
        help=(
            "Only run the six base+reasoning subtrees from HF_DATASET_MODEL_PATH_ROWS "
            "(DeepSeek-R1-Distill 1.5B / 8B / 14B + base counterparts; all published layers). "
            "See sds_train_gsm8k_hf.HF_DATASET_MODEL_PATH_ROWS."
        ),
    )
    p.add_argument("--include-regex", type=str, default=None, help="Only paths matching this regex.")
    p.add_argument("--exclude-regex", type=str, default=None, help="Skip paths matching this regex.")
    p.add_argument("--max-runs", type=int, default=None, help="Stop after N successful starts (for testing).")
    p.add_argument("--dry-run", action="store_true", help="List runs only; do not execute pipeline.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get("HF_TOKEN", "").strip() or None

    include = re.compile(args.include_regex) if args.include_regex else None
    exclude = re.compile(args.exclude_regex) if args.exclude_regex else None

    print(f"Listing feature pickles in {args.repo!r} …", flush=True)
    all_paths = list_hub_feature_pickles(args.repo, token)
    paths = filter_paths(
        all_paths,
        include,
        exclude,
        args.skip_ckpt_sweeps,
        dataset_core_only=args.dataset_core,
    )
    print(f"Found {len(all_paths)} candidate file(s), {len(paths)} after filters.", flush=True)

    output_root = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_root, exist_ok=True)

    manifest_path = os.path.join(output_root, "batch_manifest.json")
    summary_lines: List[str] = []

    base_cfg = SteerConfig()

    results: List[Dict[str, Any]] = []
    n_ok = 0
    for i, hub_filename in enumerate(paths):
        if args.max_runs is not None and n_ok >= args.max_runs:
            print(f"Stopping early (--max-runs {args.max_runs}).", flush=True)
            break
        save_dir = hf_path_to_save_dir(output_root, hub_filename)
        line = f"[{i+1}/{len(paths)}] {hub_filename} -> {save_dir}"
        print(line, flush=True)
        summary_lines.append(line)
        if args.dry_run:
            results.append(
                {
                    "hub_filename": hub_filename,
                    "save_dir": save_dir,
                    "status": "dry_run",
                }
            )
            continue
        try:
            rec = run_one(
                hub_filename,
                repo_id=args.repo,
                output_root=output_root,
                num_samples=args.num_samples,
                enable_openai_judge=args.openai_judge,
                base_cfg=base_cfg,
            )
            results.append(rec)
            n_ok += 1
            print(
                f"    OK in {rec['seconds']}s | steerability_pct={rec.get('steerability_pct')}",
                flush=True,
            )
        except Exception as e:  # pragma: no cover
            err = f"{type(e).__name__}: {e}"
            print(f"    FAIL: {err}", flush=True)
            results.append(
                {
                    "hub_filename": hub_filename,
                    "save_dir": save_dir,
                    "status": "error",
                    "error": err,
                    "traceback": traceback.format_exc(),
                }
            )

    sweep = {
        "repo": args.repo,
        "output_root": output_root,
        "num_samples": args.num_samples,
        "openai_judge": args.openai_judge,
        "skip_ckpt_sweeps": args.skip_ckpt_sweeps,
        "dataset_core_subset": args.dataset_core,
        "hf_dataset_model_path_rows": HF_DATASET_MODEL_PATH_ROWS,
        "include_regex": args.include_regex,
        "exclude_regex": args.exclude_regex,
        "dry_run": args.dry_run,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": results,
        "n_listed": len(paths),
        "n_completed_ok": sum(1 for r in results if r.get("status") == "ok"),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)

    summary_txt = os.path.join(output_root, "batch_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Sweep: {args.repo}\n")
        f.write(f"Output: {output_root}\n")
        f.write(f"Manifest: {manifest_path}\n")
        f.write("\n".join(summary_lines))
        f.write("\n")

    print(f"Wrote {manifest_path}", flush=True)
    print(f"Wrote {summary_txt}", flush=True)
    if args.dry_run:
        return 0
    had_err = any(r.get("status") == "error" for r in results)
    return 1 if had_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
