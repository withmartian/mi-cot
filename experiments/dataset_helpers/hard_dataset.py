"""
generate_hard_dataset.py

For each (model, dataset) combination, finds "hard examples" defined as:
    RLVR model answers correctly AND Base model answers incorrectly.

Saves 50 such examples per combo as a JSON file with:
  - problem text
  - ground truth answer
  - RLVR CoT + extracted answer
  - Base CoT + extracted answer

Usage:
    python generate_hard_dataset.py \
        --out /home/abir19/scratch/abir19/SDS_hard_datasets \
        --n_search 500 \
        --n_target 50
"""

import os, re, gc, json, argparse, random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16
print(f"[init] device={DEVICE}", flush=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── MODEL CONFIG ─────────────────────────────────────────────
MODELS = {
    "llama8b": {
        "rlvr":     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "base":     "meta-llama/Llama-3.1-8B-Instruct",
        "tok_base": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "qwen14b": {
        "rlvr":     "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "base":     "Qwen/Qwen2.5-14B-Instruct",
        "tok_base": "Qwen/Qwen2.5-14B-Instruct",
    },
    "qwen1.5b": {
        "rlvr":     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "base":     "Qwen/Qwen2.5-1.5B-Instruct",
        "tok_base": "Qwen/Qwen2.5-1.5B-Instruct",
    },
}

# ── DATASET CONFIG ────────────────────────────────────────────
DATASETS = {
    "gsm8k": {
        "hf_id":  "openai/gsm8k",
        "config": "main",
        "split":  "test",
        "type":   "gsm8k",
        # columns: question, answer (contains #### number)
    },
    "math500": {
        "hf_id":  "HuggingFaceH4/MATH-500",
        "config": "default",
        "split":  "test",
        "type":   "math",
        # columns: problem, solution, answer (clean), subject, level, unique_id
    },
}

# ── ANSWER EXTRACTION ─────────────────────────────────────────

def extract_boxed(text):
    """Extract content from \\boxed{...}, handling nested braces."""
    pattern = r'\\boxed\{'
    for m in re.finditer(pattern, text):
        start = m.end()
        depth = 1; i = start
        while i < len(text) and depth > 0:
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            i += 1
        if depth == 0:
            return text[start:i-1].strip()
    return None

def extract_number(text):
    """Extract final number from GSM8K/SVAMP style answers."""
    # look for #### answer pattern first
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m: return m.group(1).replace(',','').strip()
    # look for boxed
    b = extract_boxed(text)
    if b: return b.replace(',','').strip()
    # last number in text
    nums = re.findall(r'[\-]?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None



def get_ground_truth(row, ds_type, dataset_name):
    if ds_type == "gsm8k":
        # column: answer — format "... #### 72"
        ans = row.get("answer","")
        m = re.search(r'####\s*([\-\d,\.]+)', ans)
        return m.group(1).replace(',','').strip() if m else ans.strip()
    elif ds_type == "math":
        # column: answer — already clean e.g. "\\left( 3, \\frac{\\pi}{2} \\right)"
        return row.get("answer","").strip()
    return ""

def check_correct(pred, gt, ds_type):
    if pred is None or gt is None: return False
    pred = str(pred).strip()
    gt   = str(gt).strip()
    # exact match (covers math symbolic answers)
    if pred.lower() == gt.lower(): return True
    # strip LaTeX formatting for math comparison
    def strip_latex(s):
        s = re.sub(r'\\[a-zA-Z]+', ' ', s)
        s = re.sub(r'[{},\\s]', ' ', s)
        return s.strip().lower()
    if strip_latex(pred) == strip_latex(gt): return True
    # numeric comparison
    pred_clean = pred.replace(',','')
    gt_clean   = gt.replace(',','')
    try:
        return abs(float(pred_clean) - float(gt_clean)) < 1e-6
    except:
        return False

def extract_answer(cot, ds_type):
    if ds_type == "gsm8k": return extract_number(cot)
    elif ds_type == "math": return extract_boxed(cot) or extract_number(cot)
    return None

def get_problem(row, ds_type, dataset_name):
    # gsm8k: column = "question"
    # math500: column = "problem"
    if "problem"  in row: return row["problem"]
    if "question" in row: return row["question"]
    raise KeyError(f"No problem/question key in {list(row.keys())}")

# ── GENERATION ────────────────────────────────────────────────

def load_model(model_id, tok_base):
    print(f"  Loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(tok_base, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, low_cpu_mem_usage=True,
        device_map="auto")
    model.eval()
    return model, tok

def cleanup(model=None):
    if model is not None: del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

@torch.no_grad()
def generate_answer(problem, model, tok, max_new_tokens=1024):
    # use chat template if available, otherwise raw prompt
    try:
        messages = [{"role": "user", "content": problem}]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except:
        prompt = problem

    inputs = tok(prompt, return_tensors="pt",
                 truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy for reproducibility
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

# ── MAIN ──────────────────────────────────────────────────────

def run_one(model_key, dataset_key, args):
    mcfg  = MODELS[model_key]
    dcfg  = DATASETS[dataset_key]
    ds_type = dcfg["type"]

    out_path = os.path.join(args.out,
                            f"hard_{model_key}_{dataset_key}.json")
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        if len(existing) >= args.n_target:
            print(f"  [{model_key}/{dataset_key}] already {len(existing)} examples, skipping",
                  flush=True)
            return

    print(f"\n{'='*60}", flush=True)
    print(f"  {model_key} / {dataset_key}", flush=True)

    # load dataset
    ds = load_dataset(dcfg["hf_id"], dcfg["config"],
                      split=f"{dcfg['split']}[:{args.n_search}]")
    print(f"  {len(ds)} problems loaded", flush=True)

    # load existing partial results
    existing_results = []
    partial_path = out_path.replace(".json", "_partial.json")
    done_ids = set()
    if os.path.exists(partial_path):
        existing_results = json.load(open(partial_path))
        done_ids = {r["problem_id"] for r in existing_results}
        print(f"  Resuming from {len(existing_results)} partial results", flush=True)

    # ── run RLVR model ──
    rlvr_cache = os.path.join(args.out,
                              f"cache_{model_key}_{dataset_key}_rlvr.json")
    if os.path.exists(rlvr_cache):
        rlvr_results = json.load(open(rlvr_cache))
        print(f"  RLVR cache loaded ({len(rlvr_results)} results)", flush=True)
    else:
        print(f"  Running RLVR model...", flush=True)
        rlvr_model, rlvr_tok = load_model(mcfg["rlvr"], mcfg["tok_base"])
        rlvr_results = {}
        for i in tqdm(range(len(ds)), desc=f"  RLVR {model_key}/{dataset_key}"):
            row     = ds[i]
            problem = get_problem(row, ds_type, dcfg["hf_id"])
            gt      = get_ground_truth(row, ds_type, dcfg["hf_id"])
            cot     = generate_answer(problem, rlvr_model, rlvr_tok,
                                      args.max_tokens)
            pred    = extract_answer(cot, ds_type)
            correct = check_correct(pred, gt, ds_type)
            rlvr_results[str(i)] = {
                "cot": cot, "pred": pred, "gt": gt, "correct": correct
            }
            if (i+1) % 50 == 0:
                json.dump(rlvr_results, open(rlvr_cache,'w'), indent=2)
        json.dump(rlvr_results, open(rlvr_cache,'w'), indent=2)
        cleanup(rlvr_model)

    # filter: only keep problems RLVR got right
    rlvr_correct_ids = [int(k) for k,v in rlvr_results.items() if v["correct"]]
    print(f"  RLVR correct: {len(rlvr_correct_ids)}/{len(ds)}", flush=True)

    # ── run Base model only on RLVR-correct problems ──
    base_cache = os.path.join(args.out,
                              f"cache_{model_key}_{dataset_key}_base.json")
    if os.path.exists(base_cache):
        base_results = json.load(open(base_cache))
        print(f"  Base cache loaded ({len(base_results)} results)", flush=True)
    else:
        print(f"  Running Base model on {len(rlvr_correct_ids)} RLVR-correct problems...",
              flush=True)
        base_model, base_tok = load_model(mcfg["base"], mcfg["tok_base"])
        base_results = {}
        for i in tqdm(rlvr_correct_ids, desc=f"  Base {model_key}/{dataset_key}"):
            row     = ds[i]
            problem = get_problem(row, ds_type, dcfg["hf_id"])
            gt      = get_ground_truth(row, ds_type, dcfg["hf_id"])
            cot     = generate_answer(problem, base_model, base_tok,
                                      args.max_tokens)
            pred    = extract_answer(cot, ds_type)
            correct = check_correct(pred, gt, ds_type)
            base_results[str(i)] = {
                "cot": cot, "pred": pred, "gt": gt, "correct": correct
            }
            if len(base_results) % 50 == 0:
                json.dump(base_results, open(base_cache,'w'), indent=2)
        json.dump(base_results, open(base_cache,'w'), indent=2)
        cleanup(base_model)

    # ── find hard examples: RLVR correct, Base wrong ──
    hard = []
    for i in rlvr_correct_ids:
        if str(i) not in base_results: continue
        if base_results[str(i)]["correct"]: continue  # base also got it right

        row     = ds[i]
        problem = get_problem(row, ds_type, dcfg["hf_id"])
        hard.append({
            "problem_id":   i,
            "problem":      problem,
            "ground_truth": rlvr_results[str(i)]["gt"],
            "rlvr_cot":     rlvr_results[str(i)]["cot"],
            "rlvr_pred":    rlvr_results[str(i)]["pred"],
            "rlvr_correct": True,
            "base_cot":     base_results[str(i)]["cot"],
            "base_pred":    base_results[str(i)]["pred"],
            "base_correct": False,
            "model":        model_key,
            "dataset":      dataset_key,
        })
        if len(hard) >= args.n_target:
            break

    print(f"  Hard examples found: {len(hard)}", flush=True)
    json.dump(hard, open(out_path,'w'), indent=2)
    print(f"  Saved -> {out_path}", flush=True)

    # summary stats
    rlvr_acc = sum(v["correct"] for v in rlvr_results.values()) / len(rlvr_results)
    base_acc = sum(v["correct"] for v in base_results.values()) / len(base_results)
    print(f"  RLVR acc={rlvr_acc:.3f}  Base acc={base_acc:.3f}  "
          f"hard={len(hard)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",        default="/home/abir19/scratch/abir19/SDS_hard_datasets")
    parser.add_argument("--n_search",   type=int, default=500,
                        help="Problems to evaluate per dataset")
    parser.add_argument("--n_target",   type=int, default=50,
                        help="Hard examples to collect per combo")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--models",     nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--datasets",   nargs="+", default=list(DATASETS.keys()))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    runs = [(m, d) for m in args.models for d in args.datasets]
    print(f"[main] {len(runs)} combos: {args.models} × {args.datasets}")
    print(f"[main] n_search={args.n_search}  n_target={args.n_target}\n")

    for model_key, dataset_key in runs:
        try:
            run_one(model_key, dataset_key, args)
        except Exception as e:
            print(f"  FAILED {model_key}/{dataset_key}: {e}", flush=True)
        finally:
            # always clean up GPU between runs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # ── summary table ──
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for model_key in args.models:
        for dataset_key in args.datasets:
            out_path = os.path.join(args.out,
                                    f"hard_{model_key}_{dataset_key}.json")
            if os.path.exists(out_path):
                n = len(json.load(open(out_path)))
                print(f"  {model_key:12s} {dataset_key:10s}: {n} hard examples")
            else:
                print(f"  {model_key:12s} {dataset_key:10s}: not found")

if __name__ == "__main__":
    main()