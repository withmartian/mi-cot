#!/usr/bin/env python3
"""
Self-contained strict hard-set builder.

This replaces the two-stage:

  build_rlvr_cache.py -> build_strict_hard_set.py

with one resumable script. It directly checks:

  1. the reasoning model can solve the problem
  2. the base model gets zero correct out of base_k stochastic samples

and writes records compatible with the existing hard_*.json files.

The script still keeps an internal cache next to --out so a killed job can
resume, but that cache is produced by this script itself and is not an input
dependency.
"""

import argparse
import gc
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.ablations.passatk_colm_baselines import is_correct_generation


DATASET_CFG = {
    "math500": {
        "hf_id": "HuggingFaceH4/MATH-500",
        "config": "default",
        "split": "test",
        "type": "math",
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "type": "gsm8k",
    },
    "svamp": {
        "hf_id": "ChilleD/SVAMP",
        "config": None,
        "split": "test",
        "type": "svamp",
    },
    "mmlu_pro": {
        "hf_id": "TIGER-Lab/MMLU-Pro",
        "config": None,
        "split": "test",
        "type": "mmlu_pro",
    },
}


def load_hf_dataset(dataset: str, hf_cache_dir: Optional[str]) -> Dict[int, Dict[str, str]]:
    from datasets import load_dataset
    import tempfile

    cfg = DATASET_CFG[dataset]

    def _load(cache_dir=None, download_mode=None):
        kwargs: Dict[str, Any] = {"split": cfg["split"]}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if download_mode:
            kwargs["download_mode"] = download_mode
        if cfg["config"]:
            return load_dataset(cfg["hf_id"], cfg["config"], **kwargs)
        return load_dataset(cfg["hf_id"], **kwargs)

    try:
        ds = _load(cache_dir=hf_cache_dir)
    except Exception as e:
        print(f"HF cache load failed ({e}); retrying force_redownload.", flush=True)
        try:
            ds = _load(cache_dir=hf_cache_dir, download_mode="force_redownload")
        except Exception as e2:
            print(f"force_redownload failed ({e2}); retrying isolated tmp cache.", flush=True)
            ds = _load(cache_dir=tempfile.mkdtemp(prefix="hf_tmp_"))

    problems: Dict[int, Dict[str, str]] = {}
    for i, row in enumerate(ds):
        kind = cfg["type"]
        if kind == "svamp":
            body = row.get("Body", row.get("body", ""))
            q = row.get("Question", row.get("question", ""))
            problem = f"{body}\nQuestion: {q}" if body else q
            gt = str(row.get("Answer", row.get("answer", ""))).strip()
        elif kind == "mmlu_pro":
            question = row.get("question", "")
            options = row.get("options", [])
            opts = "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(options))
            problem = f"{question}\n\nOptions:\n{opts}\n\nAnswer with the correct option letter only."
            gt = str(row.get("answer", "")).strip().upper()
        elif kind == "gsm8k":
            problem = str(row.get("problem", row.get("question", "")))
            ans = row.get("answer", "")
            m = re.search(r"####\s*([\-\d,\.]+)", str(ans))
            gt = m.group(1).replace(",", "").strip() if m else str(ans)
        else:
            problem = str(row.get("problem", row.get("question", "")))
            gt = str(row.get("answer", "")).strip()
        problems[i] = {"problem": problem, "ground_truth": gt}
    return problems


def format_prompt(tokenizer, problem: str, style: str) -> str:
    if style == "plain":
        return problem
    if style == "cot":
        return f"Solve the following problem step by step.\n\nProblem: {problem}\n\nSolution:"
    if style in ("chat", "auto"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return f"Solve the following problem step by step.\n\nProblem: {problem}\n\nSolution:"
    raise ValueError(f"Unknown prompt style: {style}")


def load_model(model_id: str, hf_cache_dir: Optional[str]):
    print(f"Loading model: {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )
    model.eval()
    return tok, model


def unload_model(model, tokenizer) -> None:
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_samples(
    model,
    tokenizer,
    prompt: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> List[str]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"]
    t0 = prompt_ids.shape[1]
    batched = prompt_ids.expand(n, -1).to(model.device)
    do_sample = temperature > 0.0
    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
    with torch.no_grad():
        out = model.generate(batched, **kwargs)
    return [tokenizer.decode(out[i][t0:], skip_special_tokens=True) for i in range(n)]


def eval_k(
    model,
    tokenizer,
    prompt: str,
    ground_truth: str,
    dataset: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    early_stop_on_correct: bool,
    early_stop_batch: int,
) -> Tuple[int, List[bool], List[str]]:
    kind = DATASET_CFG[dataset]["type"]
    all_ok: List[bool] = []
    all_texts: List[str] = []

    first_n = min(k, early_stop_batch if early_stop_on_correct else k)
    texts = generate_samples(
        model, tokenizer, prompt, first_n, max_new_tokens, temperature, seed
    )
    for text in texts:
        ok = is_correct_generation(text, ground_truth, kind)
        all_ok.append(ok)
        all_texts.append(text)

    if early_stop_on_correct and any(all_ok):
        return sum(all_ok), all_ok, all_texts

    remaining = k - first_n
    if remaining > 0:
        texts2 = generate_samples(
            model,
            tokenizer,
            prompt,
            remaining,
            max_new_tokens,
            temperature,
            seed + 10007,
        )
        for text in texts2:
            ok = is_correct_generation(text, ground_truth, kind)
            all_ok.append(ok)
            all_texts.append(text)

    return sum(all_ok), all_ok, all_texts


def load_json_dict(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open() as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {int(x["problem_id"]): x for x in raw}
    return {int(k): v for k, v in raw.items()}


def save_cache(cache: Dict[int, Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in sorted(cache.items())}
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(serializable, f, indent=2)
    tmp.replace(path)


def save_hard_json(
    cache: Dict[int, Dict[str, Any]],
    hf_problems: Dict[int, Dict[str, str]],
    out_path: Path,
    args,
) -> None:
    records = []
    for pid, rec in sorted(cache.items()):
        if not rec.get("is_hard", False):
            continue
        prob = hf_problems[pid]
        records.append(
            {
                "problem_id": pid,
                "problem": prob["problem"],
                "ground_truth": prob["ground_truth"],
                "answer": prob["ground_truth"],
                "dataset": args.dataset,
                "base_model": args.base_model,
                "reasoning_model": args.reasoning_model,
                "base_correct": rec.get("base_correct", 0),
                "base_results": rec.get("base_results", []),
                "reasoning_correct": rec.get("reasoning_correct", 0),
                "reasoning_results": rec.get("reasoning_results", []),
                "hard_definition": (
                    f"reasoning_pass_at_{args.reasoning_k}_gt0_and_"
                    f"base_pass_at_{args.base_k}_0_"
                    f"{args.base_prompt_style}_temp{args.base_temperature}"
                ),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(records, f, indent=2)
    tmp.replace(out_path)


def count_hard(cache: Dict[int, Dict[str, Any]]) -> int:
    return sum(1 for v in cache.values() if v.get("is_hard", False))


def count_reasoning_candidates(cache: Dict[int, Dict[str, Any]]) -> int:
    return sum(1 for v in cache.values() if v.get("reasoning_ok", False))


def iter_problem_ids(
    problems: Dict[int, Dict[str, str]],
    seed: int,
    shuffle: bool,
    start_index: int,
    limit: int,
) -> List[int]:
    ids = sorted(problems)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ids)
    if start_index > 0:
        ids = ids[start_index:]
    if limit > 0:
        ids = ids[:limit]
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CFG))
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--reasoning-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-cache-dir", default="/scratch/abir19/huggingface")
    ap.add_argument("--cache-out", default=None)
    ap.add_argument("--n-target", type=int, default=100)
    ap.add_argument("--candidate-target", type=int, default=0,
                    help="Reasoning-correct candidates to collect before base filtering. 0 = max(4*n_target, n_target+50).")
    ap.add_argument("--base-k", type=int, default=8)
    ap.add_argument("--reasoning-k", type=int, default=1)
    ap.add_argument("--base-temperature", type=float, default=0.7)
    ap.add_argument("--reasoning-temperature", type=float, default=0.0)
    ap.add_argument("--base-prompt-style", default="chat", choices=["chat", "cot", "plain", "auto"])
    ap.add_argument("--reasoning-prompt-style", default="chat", choices=["chat", "cot", "plain", "auto"])
    ap.add_argument("--base-max-new-tokens", type=int, default=256)
    ap.add_argument("--reasoning-max-new-tokens", type=int, default=1024)
    ap.add_argument("--early-stop-batch", type=int, default=3)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-only", action="store_true",
                    help="Skip reasoning phase and only filter existing reasoning-ok candidates in the internal cache.")
    ap.add_argument("--reasoning-only", action="store_true",
                    help="Only populate reasoning candidates; do not load/evaluate base model.")
    args = ap.parse_args()

    out_path = Path(args.out)
    cache_path = Path(args.cache_out) if args.cache_out else out_path.with_suffix(".self_contained_cache.json")

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Output: {out_path}", flush=True)
    print(f"Internal cache: {cache_path}", flush=True)

    hf_problems = load_hf_dataset(args.dataset, args.hf_cache_dir)
    problem_ids = iter_problem_ids(
        hf_problems, args.seed, args.shuffle, args.start_index, args.limit
    )
    print(f"Loaded {len(hf_problems)} problems; active scan size={len(problem_ids)}", flush=True)

    cache = load_json_dict(cache_path)
    print(
        f"Resume cache: entries={len(cache)} reasoning_ok={count_reasoning_candidates(cache)} hard={count_hard(cache)}",
        flush=True,
    )

    candidate_target = args.candidate_target or max(4 * args.n_target, args.n_target + 50)

    if not args.base_only:
        print("\n=== Phase 1: reasoning-model candidate discovery ===", flush=True)
        print(f"Target reasoning-ok candidates: {candidate_target}", flush=True)
        r_tok, r_model = load_model(args.reasoning_model, args.hf_cache_dir)

        for n_seen, pid in enumerate(problem_ids, start=1):
            if count_reasoning_candidates(cache) >= candidate_target:
                print("Reached candidate target.", flush=True)
                break
            rec = cache.setdefault(pid, {})
            if "reasoning_ok" in rec:
                continue
            prob = hf_problems[pid]
            prompt = format_prompt(r_tok, prob["problem"], args.reasoning_prompt_style)
            n_correct, oks, texts = eval_k(
                r_model,
                r_tok,
                prompt,
                prob["ground_truth"],
                args.dataset,
                args.reasoning_k,
                args.reasoning_max_new_tokens,
                args.reasoning_temperature,
                args.seed + 17 * pid,
                early_stop_on_correct=True,
                early_stop_batch=max(1, min(args.reasoning_k, args.early_stop_batch)),
            )
            rec.update(
                {
                    "reasoning_ok": n_correct > 0,
                    "reasoning_correct": n_correct,
                    "reasoning_results": oks,
                    "reasoning_example": texts[0] if texts else "",
                }
            )
            status = "R_OK" if n_correct > 0 else "r_fail"
            print(
                f"[R {n_seen}/{len(problem_ids)}] pid={pid} {status} "
                f"candidates={count_reasoning_candidates(cache)} hard={count_hard(cache)}",
                flush=True,
            )
            if n_seen % args.save_every == 0:
                save_cache(cache, cache_path)
                save_hard_json(cache, hf_problems, out_path, args)

        save_cache(cache, cache_path)
        save_hard_json(cache, hf_problems, out_path, args)
        unload_model(r_model, r_tok)

    if args.reasoning_only:
        print("reasoning-only requested; stopping before base filtering.", flush=True)
        return

    print("\n=== Phase 2: base-model strict failure filter ===", flush=True)
    b_tok, b_model = load_model(args.base_model, args.hf_cache_dir)

    candidates = [
        pid
        for pid in problem_ids
        if cache.get(pid, {}).get("reasoning_ok", False)
        and "base_correct" not in cache.get(pid, {})
    ]
    print(f"Base candidates needing evaluation: {len(candidates)}", flush=True)

    for j, pid in enumerate(candidates, start=1):
        if count_hard(cache) >= args.n_target:
            print("Reached hard target.", flush=True)
            break
        rec = cache.setdefault(pid, {})
        prob = hf_problems[pid]
        prompt = format_prompt(b_tok, prob["problem"], args.base_prompt_style)
        n_correct, oks, texts = eval_k(
            b_model,
            b_tok,
            prompt,
            prob["ground_truth"],
            args.dataset,
            args.base_k,
            args.base_max_new_tokens,
            args.base_temperature,
            args.seed + 31 * pid,
            early_stop_on_correct=True,
            early_stop_batch=args.early_stop_batch,
        )
        rec.update(
            {
                "base_correct": n_correct,
                "base_results": oks,
                "base_example": texts[0] if texts else "",
                "is_hard": n_correct == 0,
            }
        )
        status = "HARD" if n_correct == 0 else "easy_base"
        print(
            f"[B {j}/{len(candidates)}] pid={pid} {status} "
            f"base={n_correct}/{len(oks)} hard={count_hard(cache)}",
            flush=True,
        )
        if j % args.save_every == 0:
            save_cache(cache, cache_path)
            save_hard_json(cache, hf_problems, out_path, args)

    save_cache(cache, cache_path)
    save_hard_json(cache, hf_problems, out_path, args)
    unload_model(b_model, b_tok)

    print("\n=== Summary ===", flush=True)
    print(f"Reasoning-ok candidates: {count_reasoning_candidates(cache)}", flush=True)
    print(f"Strict hard records: {count_hard(cache)}", flush=True)
    print(f"Saved hard set: {out_path}", flush=True)
    print(f"Saved internal cache: {cache_path}", flush=True)


if __name__ == "__main__":
    main()
