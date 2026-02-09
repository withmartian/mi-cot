import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import re
import gc
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "rpc_dataset"  
os.makedirs(checkpoint_dir, exist_ok=True)
dtype = torch.bfloat16
SKIP_CACHE = False

model_ft_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" 
model_base_name = "Qwen/Qwen2.5-14B"
dataset_path = "openai/gsm8k"
number_cots = 100

ANCHOR_CLASSES = {
    "PROBLEM_SETUP": "Parsing or rephrasing the problem",
    "PLAN_GENERATION": "Stating or deciding on a plan of action",
    "FACT_RETRIEVAL": "Recalling facts, formulas, problem details",
    "ACTIVE_COMPUTATION": "Algebra, calculations, or manipulations",
    "UNCERTAINTY_MANAGEMENT": "Expressing confusion, re-evaluating",
    "RESULT_CONSOLIDATION": "Aggregating intermediate results",
    "SELF_CHECKING": "Verifying previous steps, checking",
    "FINAL_ANSWER_EMISSION": "Explicitly stating the final answer"
}


print(f"Model: {model_ft_name}")
print(f"Checkpoint: {checkpoint_dir}")

print("PHASE 1: LOAD MODELS & GENERATE CoTs")

print("Loading BASE model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_ft_name, torch_dtype=dtype, device_map="cuda"
)
model_tl = HookedTransformer.from_pretrained_no_processing(
    model_base_name, hf_model=hf_model, device=device, dtype=dtype
)
del hf_model
torch.cuda.empty_cache()
gc.collect()
print("Finetuned Model loaded\n")

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

def get_sentence_token_positions(text, sentences, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    token_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    positions = []
    current_pos = 0
    token_offset = 0
    for sent in sentences:
        start = token_text.find(sent, current_pos)
        if start == -1:
            positions.append(set())
            continue
        sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
        token_positions = set(range(token_offset, token_offset + len(sent_tokens)))
        positions.append(token_positions)
        token_offset += len(sent_tokens)
        current_pos = start + len(sent)
    return positions

def kl_divergence_batch(logits_p, logits_q):
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = F.softmax(logits_p, dim=-1)
    kl = torch.sum(p * (log_p - log_q), dim=-1)
    return kl.mean().item()

def get_causal_matrix(model, tokenizer, cot, sentences, problem=""):
    M = len(sentences)
    full_text = problem + " " + cot if problem else cot
    input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    token_positions = get_sentence_token_positions(full_text, sentences, tokenizer)
    seq_len = input_ids.size(1)
    
    causal_matrix = np.zeros((M, M))
    with torch.no_grad():
        logits_base, _ = model.run_with_cache(input_ids)
        logits_base = logits_base.cpu().float()
        torch.cuda.empty_cache()
        gc.collect()
    
    for source_idx in range(M):
        source_positions = [p for p in token_positions[source_idx] if p < seq_len]
        if not source_positions:
            continue
        
        def mask_attn(pattern, hook, positions=source_positions):
            masked = pattern.clone()
            masked[..., positions] = 0
            return masked / (masked.sum(dim=-1, keepdim=True) + 1e-8)
        
        with torch.no_grad():
            logits_masked = model.run_with_hooks(
                input_ids,
                fwd_hooks=[(f"blocks.{i}.attn.hook_pattern", mask_attn) for i in range(len(model.blocks))]
            )
            logits_masked = logits_masked.cpu().float()
            torch.cuda.empty_cache()
            gc.collect()
        
        for target_idx in range(source_idx + 1, M):
            target_positions = [p for p in token_positions[target_idx] if p < seq_len]
            if not target_positions:
                continue
            kl = kl_divergence_batch(
                logits_base[0, target_positions, :],
                logits_masked[0, target_positions, :]
            )
            causal_matrix[source_idx, target_idx] = kl
        
        del logits_masked
        torch.cuda.empty_cache()
        gc.collect()
    
    for target_idx in range(M):
        prior = causal_matrix[0:target_idx, target_idx]
        if len(prior) > 0:
            causal_matrix[0:target_idx, target_idx] -= np.mean(prior)
    
    return causal_matrix

print("LOADING DATASET & GENERATING CoTs")

dataset = load_dataset(dataset_path, 'main', split=f"train[:{number_cots}]")
problems = [item['question'] for item in dataset]
print(f"Loaded {len(problems)} problems\n")

ckpt_cot = f"{checkpoint_dir}/cot_data.pkl"
if os.path.exists(ckpt_cot) and not SKIP_CACHE:
    print(f"Loading cached CoT data...")
    all_cot_data_full = pickle.load(open(ckpt_cot, 'rb'))
    all_cot_data = {k: v for k, v in all_cot_data_full.items() if k < number_cots}
    print(f"Loaded {len(all_cot_data)} cached problems (limited to {number_cots})\n")
else:
    all_cot_data = {}
    for pid, problem in enumerate(problems):
        print(f"[{pid}] Generating CoT...", flush=True)
        input_ids = tokenizer.encode(problem, return_tensors='pt').to(device)
        with torch.no_grad():
            output_ids = model_tl.generate(
                input_ids, max_new_tokens=500, temperature=0.6,
                do_sample=True, top_p=0.9
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        cot = full_text[len(problem):].strip() if full_text.startswith(problem) else full_text
        sentences = split_into_sentences(cot)
        
        if len(sentences) < 2:
            print(f"  ✗ Only {len(sentences)} sentences, skipping")
            del output_ids, input_ids
            continue
        
        print(f"  {len(sentences)} sentences, computing causal matrix...")
        causal_matrix = get_causal_matrix(model_tl, tokenizer, cot, sentences, problem)
        all_cot_data[pid] = {
            'problem': problem,
            'cot': cot,
            'sentences': sentences,
            'causal_matrix': causal_matrix
        }
        del output_ids, input_ids
        torch.cuda.empty_cache()
        gc.collect()
    
    pickle.dump(all_cot_data, open(ckpt_cot, 'wb'))
    print(f"\n Saved CoT data for {len(all_cot_data)} problems\n")

print("PHASE 2: IDENTIFY ANCHORS & CLASSIFY SENTENCES")

del model_tl
torch.cuda.empty_cache()
gc.collect()

print("Loading base model for classification...")
model_base = AutoModelForCausalLM.from_pretrained(
    model_base_name, torch_dtype=dtype
).to(device)
print(" Model loaded\n")

def classify_sentence(sentence):
    classes_list = "\n".join([f"- {k}: {v}" for k, v in ANCHOR_CLASSES.items()])
    prompt = f"""Classify this reasoning step into one of these categories:

{classes_list}

Sentence: "{sentence}"

Category:"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model_base.generate(
            input_ids, max_new_tokens=5, temperature=0.1, do_sample=False
        )
    response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    category = response.strip().upper().split()[0]
    del input_ids, output_ids
    if category in ANCHOR_CLASSES:
        return category
    return "PLAN_GENERATION"

all_sentence_data = []
anchor_classes = {}

for pid, data in all_cot_data.items():
    sentences = data['sentences']
    causal_matrix = data['causal_matrix']
    
    print(f"[Problem {pid}] Processing {len(sentences)} sentences...", flush=True)
    
    outgoing = np.sum(np.abs(causal_matrix), axis=1)
    importance = outgoing
    threshold = np.percentile(importance[importance > 0], 50) if np.any(importance > 0) else 0
    
    for sent_idx, sentence in enumerate(sentences):
        is_anchor = importance[sent_idx] > threshold
        
        if is_anchor:
            stage = classify_sentence(sentence)
        else:
            stage = "NEUTRAL"
        
        all_sentence_data.append({
            'problem_id': pid,
            'sentence_idx': sent_idx,
            'sentence': sentence,
            'stage': stage,
            'is_anchor': is_anchor,
            'importance': importance[sent_idx],
        })
        
        if stage != "NEUTRAL":
            anchor_classes[stage] = anchor_classes.get(stage, 0) + 1
    
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n Processed {len(all_sentence_data)} total sentences\n")

print("Reasoning Stage Distribution (anchors only):")
for stage in sorted(ANCHOR_CLASSES.keys()):
    count = anchor_classes.get(stage, 0)
    print(f"  {stage:25s}: {count}")

del model_base
torch.cuda.empty_cache()
gc.collect()

print("PHASE 3: EXTRACT MEAN-POOLED ACTIVATIONS")

print("Reloading models for activation extraction...")
tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    model_ft_name, torch_dtype=dtype, device_map="cuda"
)
model_tl = HookedTransformer.from_pretrained_no_processing(
    model_base_name, hf_model=hf_model, device=device, dtype=dtype
)
del hf_model
torch.cuda.empty_cache()
gc.collect()
print("Models loaded\n")

def get_sentence_activation(problem, sentences, sent_idx, layer=-1):
    ctx = problem + " " + " ".join(sentences[:sent_idx])
    input_ids = tokenizer.encode(ctx, return_tensors="pt").to(device)
    
    with torch.no_grad():
        _, cache = model_tl.run_with_cache(input_ids)
        hidden = cache["resid_post", layer][0, :, :].float().cpu()
        hidden = hidden.mean(dim=0).numpy()
    
    del cache, input_ids
    torch.cuda.empty_cache()
    gc.collect()
    return hidden
