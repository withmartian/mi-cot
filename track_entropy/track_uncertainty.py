import argparse
import inspect
import json
import math
import os
import re
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_FILELOCK", "1")

import filelock

# Ensure compatibility with older filelock versions used by hf_hub.
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

# Pillow < 9.1 does not expose Image.Resampling; patch for transformers import.
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

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track token-level uncertainty during chain-of-thought reasoning."
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="GSM8K split to use (train/test)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=25,
        help="Number of GSM8K examples to evaluate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate for reasoning",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--output",
        default="uncertainty_results.jsonl",
        help="Path to write JSONL results",
    )
    return parser.parse_args()


def extract_final_number(text: str) -> str:
    matches = re.findall(r"[-+]?\d*\.?\d+", text)
    if not matches:
        return ""
    return matches[-1]


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -torch.sum(probs * log_probs).item()
    return entropy


def build_prompt(question: str) -> str:
    return (
        "Solve the problem. Show your reasoning step by step and finish with "
        "'Final answer: <number>'.\n\n"
        f"Question: {question}\n\n"
        "Let's think step by step.\n"
    )


def generate_with_uncertainty(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    stream_trace: bool,
) -> Tuple[str, List[Dict[str, float]]]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream_trace else None
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    uncertainty_steps: List[Dict[str, float]] = []
    for step, logits in enumerate(outputs.scores):
        step_entropy = entropy_from_logits(logits[0])
        uncertainty_steps.append(
            {
                "step": step,
                "entropy": step_entropy,
            }
        )
    return text, uncertainty_steps


def main() -> None:
    args = parse_args()

    dataset = load_dataset("gsm8k", "main", split=args.split)
    dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    try:
        import accelerate  # noqa: F401
        has_accelerate = True
    except Exception:
        has_accelerate = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if has_accelerate else None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    correct = 0

    for example in dataset:
        print(f"\n=== Example {len(results) + 1}/{len(dataset)} ===", flush=True)
        question = example["question"]
        gold = extract_final_number(example["answer"])
        prompt = build_prompt(question)
        generated, uncertainty = generate_with_uncertainty(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stream_trace=True,
        )
        pred = extract_final_number(generated)
        is_correct = pred == gold and pred != ""
        if is_correct:
            correct += 1

        results.append(
            {
                "question": question,
                "gold_answer": gold,
                "generated": generated,
                "predicted_answer": pred,
                "correct": is_correct,
                "uncertainty": uncertainty,
            }
        )

    accuracy = correct / len(results) if results else 0.0
    summary = {
        "model": args.model,
        "split": args.split,
        "max_examples": args.max_examples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "accuracy": accuracy,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary}) + "\n")
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(results)} examples to {args.output}")
    print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
