"""
Worker for create_dataset_multigpu: runs Phase 1 + Phase 2 on a single GPU
for a slice of problem indices. Invoked by the launcher with
CUDA_VISIBLE_DEVICES set so this process sees only one GPU.

Usage (called by create_dataset_multigpu.py, not directly):
  CUDA_VISIBLE_DEVICES=<gpu_id> python create_dataset_worker.py <rank> <start_idx> <end_idx> <args_pickle>
"""

from __future__ import annotations

import os
import sys
import gc
import pickle
import argparse

# Set device before any torch/cuda import (launcher sets CUDA_VISIBLE_DEVICES when spawning)
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from create_dataset import (
    CLASSES_ORDERED,
    split_into_sentences,
    get_sentence_token_ranges,
    process_problem,
    get_classification_prompt,
    classify_sentences,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16


def parse_worker_args():
    p = argparse.ArgumentParser()
    p.add_argument("rank", type=int)
    p.add_argument("start_idx", type=int)
    p.add_argument("end_idx", type=int)
    p.add_argument("args_pickle", type=str)
    return p.parse_args()


def run_worker(rank: int, start_idx: int, end_idx: int, args_dict: dict):
    out = args_dict["out"]
    shard_dir = os.path.join(out, f"shard_{rank}")
    os.makedirs(shard_dir, exist_ok=True)
    ckpt_raw = os.path.join(shard_dir, "raw_extractions.pkl")

    from datasets import load_dataset

    dataset_name = args_dict["dataset"]
    split = args_dict["split"]
    n = args_dict["n"]
    config = "default" if "MATH" in dataset_name else "main"
    ds = load_dataset(dataset_name, config, split=f"{split}[:{n}]")
    key = "problem" if "problem" in ds[0] else "question"
    problems = [ds[i][key] for i in range(len(ds))]

    my_pids = list(range(start_idx, min(end_idx, len(problems))))
    if not my_pids:
        print(f"[Rank {rank}] No problems in range [{start_idx}, {end_idx})")
        return

    seed = args_dict.get("seed", 42)

    if os.path.exists(ckpt_raw):
        all_extractions = pickle.load(open(ckpt_raw, "rb"))
        done_pids = set(all_extractions.keys()) & set(my_pids)
    else:
        all_extractions = {}
        done_pids = set()

    remaining = [(pid, problems[pid]) for pid in my_pids if pid not in done_pids]
    print(f"[Rank {rank}] Problems {start_idx}-{end_idx}: {len(remaining)} remaining, {len(done_pids)} done")

    # Phase 1: generation + extraction
    if remaining:
        tokenizer = AutoTokenizer.from_pretrained(args_dict["base"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args_dict["model"],
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        model.eval()

        for pid, problem in tqdm(remaining, desc=f"Rank {rank} Phase1"):
            torch.manual_seed(seed + pid)
            np.random.seed(seed + pid)
            result = process_problem(problem, model, tokenizer, args_dict["layer"])
            if result:
                all_extractions[pid] = result
            if (pid + 1) % 25 == 0:
                pickle.dump(all_extractions, open(ckpt_raw, "wb"))
                torch.cuda.empty_cache()
                gc.collect()

        pickle.dump(all_extractions, open(ckpt_raw, "wb"))
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Phase 2: classification
    unclassified = []
    unclassified_refs = []
    for pid, data in all_extractions.items():
        for i, feat in enumerate(data["sentence_features"]):
            if "stage" not in feat:
                unclassified.append(feat["sentence"])
                unclassified_refs.append((pid, i))

    if unclassified:
        clf_tokenizer = AutoTokenizer.from_pretrained(args_dict["clf"], trust_remote_code=True)
        clf_tokenizer.pad_token = clf_tokenizer.eos_token
        clf_tokenizer.padding_side = "left"
        classifier = AutoModelForCausalLM.from_pretrained(
            args_dict["clf"],
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        classifier.eval()

        classifications = classify_sentences(
            unclassified,
            clf_tokenizer,
            classifier,
            batch_size=args_dict.get("batch", 4),
        )

        for (pid, feat_idx), cls in zip(unclassified_refs, classifications):
            all_extractions[pid]["sentence_features"][feat_idx]["stage"] = cls
            all_extractions[pid]["sentence_features"][feat_idx]["is_anchor"] = cls != "NEUTRAL"

        pickle.dump(all_extractions, open(ckpt_raw, "wb"))
        del classifier, clf_tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[Rank {rank}] Done. Extracted {len(all_extractions)} problems.")


def main():
    args = parse_worker_args()
    with open(args.args_pickle, "rb") as f:
        args_dict = pickle.load(f)
    run_worker(args.rank, args.start_idx, args.end_idx, args_dict)


if __name__ == "__main__":
    main()
