"""
Extract token-level and sentence-level entropy for each sentence in the SDS dataset.
Runs the reasoning model on each problem's full prompt and computes next-token
entropy at each position, then aggregates per sentence (mean and last-token).
Output order matches all_features so results align with SDS states.
"""
import argparse
import os
import pickle
import sys
import inspect

# Avoid filelock/huggingface_hub compatibility when loading tokenizer/model
os.environ.setdefault("HF_HUB_DISABLE_FILELOCK", "1")
try:
    import filelock
    if "mode" not in inspect.signature(filelock.FileLock.__init__).parameters:
        class FileLockCompat(filelock.FileLock):
            def __init__(self, lock_file, timeout=-1, **kwargs):
                super().__init__(lock_file, timeout=timeout)
        filelock.FileLock = FileLockCompat
        try:
            import huggingface_hub.utils._fixes as hf_fixes
            hf_fixes.FileLock = FileLockCompat
        except Exception:
            pass
except Exception:
    pass

# Pillow < 9.1 does not have Image.Resampling (required by transformers)
try:
    from PIL import Image
    if not hasattr(Image, "Resampling"):
        class _Resampling:
            NEAREST = Image.NEAREST
            BILINEAR = Image.BILINEAR
            BICUBIC = Image.BICUBIC
            LANCZOS = Image.LANCZOS
            HAMMING = Image.HAMMING
            BOX = Image.BOX
        Image.Resampling = _Resampling
except Exception:
    pass

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from data_utils import load_features_and_cot, ordered_sentence_keys, get_full_prompt_and_ranges


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) from logits, last dim is vocab."""
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return -(probs * log_probs).sum(dim=-1)


def extract_sentence_entropy(
    model,
    tokenizer,
    all_features: list,
    ckpt_cot: dict,
    device: str = "cuda",
    max_length: int = 2048,
) -> list:
    """
    For each (problem_id, sentence_idx) in all_features order, compute
    sent_entropy_mean and sent_entropy_last. Returns list of dicts in same order.
    """
    ordered = ordered_sentence_keys(all_features)
    # unique pids in order of first appearance (to match iteration over problems)
    seen = set()
    unique_pids = []
    for pid, _ in ordered:
        if pid not in seen:
            seen.add(pid)
            unique_pids.append(pid)

    # per-problem: run model once, get token entropies
    pid_to_entropies = {}
    for pid in tqdm(unique_pids, desc="Extracting entropy per problem"):
        full_prompt, sentences, ranges = get_full_prompt_and_ranges(pid, ckpt_cot, tokenizer)
        if full_prompt is None or not ranges:
            pid_to_entropies[pid] = None
            continue
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # (1, seq_len, vocab)
        # next-token entropy at each position: logits[i] predicts token i+1
        seq_len = logits.shape[1]
        # entropy at position i is for the next token (i+1)
        entropies = entropy_from_logits(logits[0]).cpu().numpy()  # (seq_len,)
        # build per-sentence (sentence_idx -> (mean_ent, last_ent))
        sent_entropy = []
        for (start, end) in ranges:
            if end > seq_len:
                end = seq_len
            if start >= end:
                sent_entropy.append((float("nan"), float("nan")))
                continue
            seg = entropies[start:end]
            mean_ent = float(np.nanmean(seg))
            last_ent = float(entropies[end - 1])
            sent_entropy.append((mean_ent, last_ent))
        pid_to_entropies[pid] = sent_entropy

    # build result in all_features order
    result = []
    for i, (pid, sent_idx) in enumerate(ordered):
        ent_list = pid_to_entropies.get(pid)
        if ent_list is None or sent_idx >= len(ent_list):
            result.append({
                "problem_id": pid,
                "sentence_idx": sent_idx,
                "sent_entropy_mean": float("nan"),
                "sent_entropy_last": float("nan"),
            })
        else:
            mean_ent, last_ent = ent_list[sent_idx]
            result.append({
                "problem_id": pid,
                "sentence_idx": sent_idx,
                "sent_entropy_mean": mean_ent,
                "sent_entropy_last": last_ent,
            })
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract sentence-level entropy for SDS alignment")
    parser.add_argument("--features", default="rpc_dataset/all_sentences_features.pkl", help="Path to all_sentences_features.pkl")
    parser.add_argument("--cot", default="rpc_dataset/cot_data.pkl", help="Path to cot_data.pkl")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", help="HuggingFace model id")
    parser.add_argument("--out", default="entropy_on_sds_output", help="Output directory")
    parser.add_argument("--limit-problems", type=int, default=None, help="Only use problem_id < N")
    parser.add_argument("--max-length", type=int, default=2048, help="Max token length per prompt")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.abspath(args.features) if os.path.isabs(args.features) else os.path.join(repo_root, args.features)
    cot_path = os.path.abspath(args.cot) if os.path.isabs(args.cot) else os.path.join(repo_root, args.cot)
    out_dir = os.path.abspath(args.out) if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_features, ckpt_cot = load_features_and_cot(features_path, cot_path, limit_problems=args.limit_problems)
    print(f"Loaded {len(all_features)} sentences, {len(ckpt_cot)} problems in cot_data")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    table = extract_sentence_entropy(model, tokenizer, all_features, ckpt_cot, device=device, max_length=args.max_length)

    out_path = os.path.join(out_dir, "sentence_entropy.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(table, f)
    print(f"Saved {len(table)} rows to {out_path}")

    # also CSV for inspection
    import csv
    csv_path = os.path.join(out_dir, "sentence_entropy.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem_id", "sentence_idx", "sent_entropy_mean", "sent_entropy_last"])
        w.writeheader()
        w.writerows(table)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
