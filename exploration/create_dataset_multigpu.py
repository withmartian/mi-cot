"""
Multi-GPU launcher for create_dataset: shards problems across N GPUs (e.g. 8x A100),
runs Phase 1 + Phase 2 per shard, then merges and writes final outputs.

Usage:
    python create_dataset_multigpu.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --base Qwen/Qwen2.5-14B \
        --dataset openai/gsm8k \
        --split train \
        --n 7473 \
        --layer 28 \
        --out ./gsm8k_output \
        --ngpus 8
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse
import subprocess
import tempfile

# Import from create_dataset for merge/flatten logic and validation
from create_dataset import (
    CLASSES_ORDERED,
    parse_args as base_parse_args,
    print_validation,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run create_dataset across multiple GPUs (shard by problem index)."
    )
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    p.add_argument("--base", default="Qwen/Qwen2.5-14B")
    p.add_argument("--clf", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    p.add_argument("--split", default="test")
    p.add_argument("--layer", type=int, default=28)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out", default="./dataset_output")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--validate", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument(
        "--ngpus",
        type=int,
        default=8,
        help="Number of GPUs to use (default 8 for 8x A100)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)

    # Load dataset once to get size and validate
    from datasets import load_dataset

    config = "default" if "MATH" in args.dataset else "main"
    ds = load_dataset(args.dataset, config, split=f"{args.split}[:{args.n}]")
    key = "problem" if "problem" in ds[0] else "question"
    n_problems = len(ds)
    print(f"Dataset: {args.dataset} {args.split}, {n_problems} problems")
    print(f"Sharding across {args.ngpus} GPUs\n")

    ngpus = min(args.ngpus, n_problems)
    if ngpus < args.ngpus:
        print(f"Using {ngpus} GPUs (n_problems={n_problems} < ngpus={args.ngpus})")

    # Shard boundaries: [0, s1), [s1, s2), ..., [s_{n-1}, n_problems)
    shard_size = (n_problems + ngpus - 1) // ngpus
    ranges = []
    for i in range(ngpus):
        start = i * shard_size
        end = min(start + shard_size, n_problems)
        if start < n_problems:
            ranges.append((i, start, end))

    # Args for workers (out must be absolute since workers run with cwd=script_dir)
    args_dict = {
        "out": os.path.abspath(args.out),
        "dataset": args.dataset,
        "split": args.split,
        "n": args.n,
        "base": args.base,
        "model": args.model,
        "clf": args.clf,
        "layer": args.layer,
        "batch": args.batch,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, "create_dataset_worker.py")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        args_pickle = f.name
    try:
        pickle.dump(args_dict, open(args_pickle, "wb"))

        procs = []
        for rank, start_idx, end_idx in ranges:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(rank)
            cmd = [
                sys.executable,
                worker_script,
                str(rank),
                str(start_idx),
                str(end_idx),
                args_pickle,
            ]
            p = subprocess.Popen(cmd, env=env, cwd=script_dir)
            procs.append((rank, p))

        for rank, p in procs:
            p.wait()
            if p.returncode != 0:
                raise RuntimeError(f"Worker rank {rank} exited with code {p.returncode}")

        print("\nAll workers finished. Merging shards...")
    finally:
        if os.path.exists(args_pickle):
            os.unlink(args_pickle)

    # Merge raw_extractions from all shards
    all_extractions = {}
    for rank, start_idx, end_idx in ranges:
        shard_raw = os.path.join(args.out, f"shard_{rank}", "raw_extractions.pkl")
        if not os.path.exists(shard_raw):
            continue
        shard_data = pickle.load(open(shard_raw, "rb"))
        for pid, data in shard_data.items():
            all_extractions[pid] = data

    # Flatten to final format (same as create_dataset)
    ckpt_features = os.path.join(args.out, "all_sentences_features.pkl")
    ckpt_with_neu = os.path.join(args.out, "all_sentences_features_with_neutral.pkl")
    ckpt_cot = os.path.join(args.out, "cot_data.pkl")
    ckpt_raw_merged = os.path.join(args.out, "raw_extractions.pkl")

    pickle.dump(all_extractions, open(ckpt_raw_merged, "wb"))

    all_features = []
    all_features_with_neutral = []
    for pid, data in all_extractions.items():
        for feat in data["sentence_features"]:
            if "stage" not in feat:
                continue
            entry = {
                "hidden_state": feat["hidden_state"],
                "hidden_state_last": feat["hidden_state_last"],
                "problem_id": pid,
                "sentence_idx": feat["sentence_idx"],
                "sentence": feat["sentence"],
                "stage": feat["stage"],
                "is_anchor": feat.get("is_anchor", feat["stage"] != "NEUTRAL"),
            }
            all_features_with_neutral.append(entry)
            if feat["stage"] != "NEUTRAL":
                all_features.append(entry)

    pickle.dump(all_features, open(ckpt_features, "wb"))
    pickle.dump(all_features_with_neutral, open(ckpt_with_neu, "wb"))
    pickle.dump(
        {
            pid: {
                "problem": d["problem"],
                "cot": d["cot"],
                "sentences": d["sentences"],
            }
            for pid, d in all_extractions.items()
        },
        open(ckpt_cot, "wb"),
    )

    print_validation(all_extractions, n=args.validate)

    stage_counts = {}
    for f in all_features_with_neutral:
        stage_counts[f["stage"]] = stage_counts.get(f["stage"], 0) + 1

    print(f"\n{'='*60}")
    print("DONE (multi-GPU merge)")
    print(f"  Non-neutral features : {len(all_features)}")
    print(f"  All features         : {len(all_features_with_neutral)}")
    print(f"  Output dir           : {args.out}")
    print(f"\nStage distribution:")
    for cls in CLASSES_ORDERED:
        print(f"  {cls:30s}: {stage_counts.get(cls, 0)}")
    print("="*60)


if __name__ == "__main__":
    main()
