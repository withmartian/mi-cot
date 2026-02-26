"""
generate_dataset.py

Generates CoT reasoning dataset with hidden state extractions.
Supports resuming from existing files and appending new problems.

Usage:
    python generate_dataset.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --base Qwen/Qwen2.5-14B \
        --dataset HuggingFaceH4/MATH-500 \
        --layer 28 \
        --n 500 \
        --out /path/to/output \
        --validate 20
"""

import os
import re
import gc
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

CLASSES_ORDERED = [
    "PROBLEM_SETUP",
    "FACT_RETRIEVAL",
    "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT",
    "SELF_CHECKING",
    "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION",
    "FINAL_ANSWER_EMISSION"
]

# ── ARGS ────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    p.add_argument("--base",     default="Qwen/Qwen2.5-14B")
    p.add_argument("--clf",      default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset",  default="HuggingFaceH4/MATH-500")
    p.add_argument("--split",    default="test")
    p.add_argument("--layer",    type=int, default=28)
    p.add_argument("--n",        type=int, default=500)
    p.add_argument("--out",      default="./dataset_output")
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--validate", type=int, default=20, help="Print class sequences for first N problems")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    return p.parse_args()

# ── HELPERS ─────────────────────────────────────────────────

def split_into_sentences(text):
    # strip <think> tags common in DeepSeek outputs
    text = re.sub(r'<think>|</think>', '', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

@torch.no_grad()
def get_sentence_token_ranges(full_text, sentences, tokenizer):
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in full_ids]

    cumulative_chars = 0
    char_to_token = {}
    for i, tok in enumerate(tokens):
        for _ in range(len(tok)):
            char_to_token[cumulative_chars] = i
            cumulative_chars += 1

    ranges = []
    current_char = 0
    for sent in sentences:
        start_char = full_text.find(sent, current_char)
        if start_char == -1:
            start_char = current_char
        end_char = start_char + len(sent)
        start_tok = char_to_token.get(start_char, 0)
        end_tok = char_to_token.get(min(end_char - 1, max(char_to_token.keys())), len(full_ids) - 1) + 1
        ranges.append((start_tok, min(end_tok, len(full_ids))))
        current_char = end_char
    return ranges

# ── PHASE 1: CoT GENERATION + ACTIVATION EXTRACTION ─────────

@torch.no_grad()
def process_problem(problem, model, tokenizer, extract_layer):
    input_ids = tokenizer.encode(problem, return_tensors='pt').to(DEVICE)
    output_ids = model.generate(
        input_ids, max_new_tokens=1024, temperature=0.6,
        do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    cot = full_text[len(problem):].strip() if full_text.startswith(problem) else full_text
    sentences = split_into_sentences(cot)

    if len(sentences) < 2:
        return None

    full_prompt = problem + " " + cot
    prompt_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(DEVICE)
    seq_len = prompt_ids.size(1)

    hidden_states = {}
    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_states['h'] = h[0].float().cpu()  # [seq, d]

    handle = model.model.layers[extract_layer].register_forward_hook(hook_fn)
    model(prompt_ids)
    handle.remove()

    hidden = hidden_states['h']
    sentence_ranges = get_sentence_token_ranges(full_prompt, sentences, tokenizer)

    sentence_features = []
    for sent_idx, (start, end) in enumerate(sentence_ranges):
        if start >= seq_len:
            continue
        end = min(end, seq_len)
        if end <= 0:
            continue
        sentence_features.append({
            'sentence_idx': sent_idx,
            'sentence': sentences[sent_idx],
            'hidden_state': hidden[:end].mean(dim=0).numpy(),
            'hidden_state_last': hidden[end - 1].numpy(),
            'token_range': (start, end),
        })

    return {'problem': problem, 'cot': cot, 'sentences': sentences, 'sentence_features': sentence_features}

# ── PHASE 2: CLASSIFICATION ──────────────────────────────────

def get_classification_prompt(sentence, clf_tokenizer):
    messages = [
        {"role": "system", "content": "You are a classifier. Reply with ONLY the category name, nothing else."},
        {"role": "user", "content": f"""Classify this reasoning step into exactly one category:

PROBLEM_SETUP: Re-stating facts or identifying variables
FACT_RETRIEVAL: Recalling formulas or facts
PLAN_GENERATION: Stating what to do next
UNCERTAINTY_MANAGEMENT: Expressing confusion or re-evaluating
SELF_CHECKING: Verifying a previous step
RESULT_CONSOLIDATION: Summarizing intermediate steps
ACTIVE_COMPUTATION: Performing actual math
FINAL_ANSWER_EMISSION: The definitive final answer

Sentence: "{sentence[:300]}"

Reply with ONLY the category name."""}
    ]
    return clf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def classify_sentences(sentences, clf_tokenizer, classifier, batch_size=4):
    results = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="  Classifying"):
        batch = sentences[i:i + batch_size]
        prompts = [get_classification_prompt(s, clf_tokenizer) for s in batch]
        inputs = clf_tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(DEVICE)
        outputs = classifier.generate(
            inputs['input_ids'], attention_mask=inputs['attention_mask'],
            max_new_tokens=10, do_sample=False, pad_token_id=clf_tokenizer.pad_token_id
        )
        for out in outputs:
            response = clf_tokenizer.decode(
                out[inputs['input_ids'].shape[1]:], skip_special_tokens=True
            ).strip().upper()
            matched = "NEUTRAL"
            for cls in reversed(CLASSES_ORDERED):
                if cls in response:
                    matched = cls
                    break
            results.append(matched)
        if (i // batch_size) % 50 == 0:
            torch.cuda.empty_cache()
    return results

# ── VALIDATION PRINT ─────────────────────────────────────────

def print_validation(all_extractions, n=20):
    print(f"\n{'='*60}")
    print(f"VALIDATION: class sequences for first {n} problems")
    print('='*60)
    pids = sorted(all_extractions.keys())[:n]
    for pid in pids:
        data = all_extractions[pid]
        stages = [f['stage'] for f in data['sentence_features'] if 'stage' in f]
        print(f"\n[Problem {pid}] {data['problem'][:60]}...")
        print(f"  Sentences: {len(stages)}")
        print(f"  Classes:   {stages}")

# ── MAIN ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    ckpt_raw      = os.path.join(args.out, "raw_extractions.pkl")
    ckpt_features = os.path.join(args.out, "all_sentences_features.pkl")
    ckpt_with_neu = os.path.join(args.out, "all_sentences_features_with_neutral.pkl")
    ckpt_cot      = os.path.join(args.out, "cot_data.pkl")

    # ── load dataset ──
    from datasets import load_dataset
    print(f"Loading dataset {args.dataset}...")
    ds = load_dataset(args.dataset, 'default' if 'MATH' in args.dataset else 'main',
                      split=f"{args.split}[:{args.n}]")
    key = 'problem' if 'problem' in ds[0] else 'question'
    problems = [item[key] for item in ds]
    print(f"Loaded {len(problems)} problems\n")

    # ── resume: load existing extractions ──
    if os.path.exists(ckpt_raw):
        print(f"Resuming from {ckpt_raw}...")
        all_extractions = pickle.load(open(ckpt_raw, 'rb'))
        done_pids = set(all_extractions.keys())
        print(f"  Already done: {len(done_pids)} problems")
    else:
        all_extractions = {}
        done_pids = set()

    remaining = [(pid, prob) for pid, prob in enumerate(problems) if pid not in done_pids]
    print(f"  Remaining: {len(remaining)} problems\n")

    # ── phase 1: generation + extraction ──
    if remaining:
        print("Loading generation model...")
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(DEVICE)
        model.eval()
        print("Model loaded\n")

        print("="*60)
        print("PHASE 1: Generating CoTs + extracting activations")
        print("="*60)

        for pid, problem in tqdm(remaining, desc="Problems"):
            result = process_problem(problem, model, tokenizer, args.layer)
            if result:
                all_extractions[pid] = result
            if (pid + 1) % 25 == 0:
                pickle.dump(all_extractions, open(ckpt_raw, 'wb'))
                torch.cuda.empty_cache()
                gc.collect()

        pickle.dump(all_extractions, open(ckpt_raw, 'wb'))
        print(f"\nExtracted {len(all_extractions)} problems total\n")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # ── phase 2: classify unclassified sentences ──
    unclassified = []
    unclassified_refs = []
    for pid, data in all_extractions.items():
        for i, feat in enumerate(data['sentence_features']):
            if 'stage' not in feat:
                unclassified.append(feat['sentence'])
                unclassified_refs.append((pid, i))

    if unclassified:
        print("="*60)
        print(f"PHASE 2: Classifying {len(unclassified)} sentences")
        print("="*60)

        print("Loading classifier...")
        clf_tokenizer = AutoTokenizer.from_pretrained(args.clf, trust_remote_code=True)
        clf_tokenizer.pad_token = clf_tokenizer.eos_token
        clf_tokenizer.padding_side = "left"
        classifier = AutoModelForCausalLM.from_pretrained(
            args.clf, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(DEVICE)
        classifier.eval()
        print("Classifier loaded\n")

        classifications = classify_sentences(
            unclassified, clf_tokenizer, classifier, batch_size=args.batch
        )

        for (pid, feat_idx), cls in zip(unclassified_refs, classifications):
            all_extractions[pid]['sentence_features'][feat_idx]['stage'] = cls
            all_extractions[pid]['sentence_features'][feat_idx]['is_anchor'] = (cls != "NEUTRAL")

        pickle.dump(all_extractions, open(ckpt_raw, 'wb'))

        del classifier, clf_tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("All sentences already classified, skipping phase 2\n")

    # ── validation print ──
    print_validation(all_extractions, n=args.validate)

    # ── flatten to final format ──
    print("\nFlattening to final format...")
    all_features, all_features_with_neutral = [], []

    for pid, data in all_extractions.items():
        for feat in data['sentence_features']:
            if 'stage' not in feat:
                continue
            entry = {
                'hidden_state':      feat['hidden_state'],
                'hidden_state_last': feat['hidden_state_last'],
                'problem_id':        pid,
                'sentence_idx':      feat['sentence_idx'],
                'sentence':          feat['sentence'],
                'stage':             feat['stage'],
                'is_anchor':         feat.get('is_anchor', feat['stage'] != 'NEUTRAL'),
            }
            all_features_with_neutral.append(entry)
            if feat['stage'] != "NEUTRAL":
                all_features.append(entry)

    pickle.dump(all_features,              open(ckpt_features, 'wb'))
    pickle.dump(all_features_with_neutral, open(ckpt_with_neu, 'wb'))
    pickle.dump({pid: {'problem': d['problem'], 'cot': d['cot'], 'sentences': d['sentences']}
                 for pid, d in all_extractions.items()},
                open(ckpt_cot, 'wb'))

    # ── stats ──
    stage_counts = {}
    for f in all_features_with_neutral:
        stage_counts[f['stage']] = stage_counts.get(f['stage'], 0) + 1

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"  Non-neutral features : {len(all_features)}")
    print(f"  All features         : {len(all_features_with_neutral)}")
    print(f"  Output dir           : {args.out}")
    print(f"\nStage distribution:")
    for cls in CLASSES_ORDERED:
        print(f"  {cls:30s}: {stage_counts.get(cls, 0)}")
    print('='*60)

if __name__ == "__main__":
    main()