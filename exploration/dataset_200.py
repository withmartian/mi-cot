# import torch
# import torch.nn.functional as F
# import numpy as np
# import pickle
# import os
# import re
# import gc
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformer_lens import HookedTransformer
# from datasets import load_dataset
# from tqdm import tqdm

# device = "cuda" if torch.cuda.is_available() else "cpu"
# checkpoint_dir = "rpc_dataset_layer28_200"
# os.makedirs(checkpoint_dir, exist_ok=True)
# dtype = torch.bfloat16

# model_ft_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
# model_base_name = "Qwen/Qwen2.5-14B"
# dataset_path = "openai/gsm8k"
# number_cots = 200
# EXTRACT_LAYER = 28
# BATCH_SIZE_CLASSIFY = 4

# CLASSES_ORDERED = [
#     "NEUTRAL",
#     "PROBLEM_SETUP", 
#     "FACT_RETRIEVAL",
#     "PLAN_GENERATION",
#     "UNCERTAINTY_MANAGEMENT",
#     "SELF_CHECKING",
#     "RESULT_CONSOLIDATION",
#     "ACTIVE_COMPUTATION",
#     "FINAL_ANSWER_EMISSION"
# ]

# print(f"Model: {model_ft_name}")
# print(f"Target: {number_cots} problems")

# # --- LOAD MODEL ONCE ---
# print("Loading model...")
# tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# hf_model = AutoModelForCausalLM.from_pretrained(model_ft_name, torch_dtype=dtype, device_map="cuda")
# model = HookedTransformer.from_pretrained_no_processing(model_base_name, hf_model=hf_model, device=device, dtype=dtype)
# del hf_model
# torch.cuda.empty_cache()
# print("Model loaded\n")

# def split_into_sentences(text):
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

# @torch.no_grad()
# def get_sentence_token_ranges(full_text, sentences, tokenizer):
#     full_ids = tokenizer.encode(full_text, add_special_tokens=False)
#     ranges = []
    
#     tokens = [tokenizer.decode([tid]) for tid in full_ids]
#     cumulative_chars = 0
#     char_to_token = {}
#     for i, tok in enumerate(tokens):
#         for _ in range(len(tok)):
#             char_to_token[cumulative_chars] = i
#             cumulative_chars += 1
    
#     current_char = 0
#     for sent in sentences:
#         start_char = full_text.find(sent, current_char)
#         if start_char == -1:
#             start_char = full_text.find(sent.strip(), current_char)
#         if start_char == -1:
#             start_char = current_char
        
#         end_char = start_char + len(sent)
#         start_tok = char_to_token.get(start_char, 0)
#         end_tok = char_to_token.get(min(end_char - 1, max(char_to_token.keys()) if char_to_token else 0), len(full_ids) - 1) + 1
        
#         ranges.append((start_tok, min(end_tok, len(full_ids))))
#         current_char = end_char
    
#     return ranges

# @torch.no_grad()
# def process_single_problem(problem, model, tokenizer, extract_layer=28):
#     input_ids = tokenizer.encode(problem, return_tensors='pt').to(device)
    
#     output_ids = model.generate(input_ids, max_new_tokens=500, temperature=0.6, do_sample=True, top_p=0.9)
#     full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     cot = full_text[len(problem):].strip() if full_text.startswith(problem) else full_text
#     sentences = split_into_sentences(cot)
    
#     if len(sentences) < 2:
#         return None
    
#     full_prompt = problem + " " + cot
#     prompt_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
#     seq_len = prompt_ids.size(1)
    
#     _, cache = model.run_with_cache(prompt_ids, names_filter=lambda n: f"blocks.{extract_layer}" in n and "resid_post" in n)
    
#     hidden = cache[f"blocks.{extract_layer}.hook_resid_post"][0].float().cpu()
    
#     del cache
#     torch.cuda.empty_cache()
    
#     sentence_ranges = get_sentence_token_ranges(full_prompt, sentences, tokenizer)
    
#     sentence_features = []
#     for sent_idx, (start, end) in enumerate(sentence_ranges):
#         if start >= seq_len:
#             continue
#         end = min(end, seq_len)
#         ctx_end = end
#         if ctx_end <= 0:
#             continue
            
#         mean_pooled = hidden[:ctx_end].mean(dim=0).numpy()
#         last_token = hidden[ctx_end - 1].numpy()
        
#         sentence_features.append({
#             'sentence_idx': sent_idx,
#             'sentence': sentences[sent_idx],
#             'hidden_state': mean_pooled,
#             'hidden_state_last': last_token,
#             'token_range': (start, end),
#         })
    
#     return {
#         'problem': problem,
#         'cot': cot,
#         'sentences': sentences,
#         'sentence_features': sentence_features,
#     }

# def get_classification_prompt(sentence):
#     return f"""Classify this reasoning step.
# If multiple categories apply, choose the one lowest in this list (the most 'active' one).

# 1. NEUTRAL: Filler, conversational bridges, or irrelevant text.
# 2. PROBLEM_SETUP: Re-stating facts or identifying variables.
# 3. FACT_RETRIEVAL: Recalling formulas (e.g., "Area is pi*r^2").
# 4. PLAN_GENERATION: Stating what to do next.
# 5. UNCERTAINTY_MANAGEMENT: Expressing confusion or re-evaluating.
# 6. SELF_CHECKING: Verifying if a previous step was correct.
# 7. RESULT_CONSOLIDATION: Summarizing intermediate steps.
# 8. ACTIVE_COMPUTATION: Performing actual math or algebraic manipulation.
# 9. FINAL_ANSWER_EMISSION: The definitive final answer statement.

# Sentence: "{sentence[:300]}"
# Category:"""

# @torch.no_grad()
# def classify_all_sentences(sentences, batch_size=4):
#     from transformers import AutoModelForCausalLM, AutoTokenizer as AT
    
#     print("  Loading Qwen-7B classifier...")
#     clf_name = "Qwen/Qwen2.5-7B-Instruct"
#     clf_tokenizer = AT.from_pretrained(clf_name, trust_remote_code=True)
#     clf_tokenizer.pad_token = clf_tokenizer.eos_token
#     classifier = AutoModelForCausalLM.from_pretrained(clf_name, torch_dtype=dtype, device_map="cuda", trust_remote_code=True)
#     classifier.eval()
    
#     results = []
#     for i in tqdm(range(0, len(sentences), batch_size), desc="  Classifying"):
#         batch = sentences[i:i + batch_size]
#         prompts = [get_classification_prompt(s) for s in batch]
        
#         inputs = clf_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#         outputs = classifier.generate(
#             inputs['input_ids'], 
#             attention_mask=inputs['attention_mask'],
#             max_new_tokens=5, 
#             do_sample=False, 
#             pad_token_id=clf_tokenizer.pad_token_id
#         )
        
#         for out in outputs:
#             response = clf_tokenizer.decode(out[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
#             response_clean = response.strip().upper()
            
#             matched = "NEUTRAL"
#             for cls in reversed(CLASSES_ORDERED):
#                 if cls in response_clean:
#                     matched = cls
#                     break
#             results.append(matched)
        
#         if (i // batch_size) % 50 == 0:
#             torch.cuda.empty_cache()
    
#     del classifier, clf_tokenizer
#     torch.cuda.empty_cache()
#     gc.collect()
#     return results

# # --- MAIN EXTRACTION ---
# print("Loading dataset...")
# dataset = load_dataset(dataset_path, 'main', split=f"train[:{number_cots}]")
# problems = [item['question'] for item in dataset]
# print(f"Loaded {len(problems)} problems\n")

# # Phase 1: Generate CoT and extract activations
# print("="*60)
# print("PHASE 1: Generating CoTs and extracting activations...")
# print("="*60)

# ckpt_raw = f"{checkpoint_dir}/raw_extractions.pkl"
# if os.path.exists(ckpt_raw):
#     print("Loading cached extractions...")
#     all_extractions = pickle.load(open(ckpt_raw, 'rb'))
#     print(f"Loaded {len(all_extractions)} cached problems\n")
# else:
#     all_extractions = {}
#     for pid, problem in enumerate(tqdm(problems, desc="Processing")):
#         result = process_single_problem(problem, model, tokenizer, extract_layer=EXTRACT_LAYER)
#         if result:
#             all_extractions[pid] = result
        
#         if (pid + 1) % 50 == 0:
#             pickle.dump(all_extractions, open(ckpt_raw, 'wb'))
#             torch.cuda.empty_cache()
#             gc.collect()
    
#     pickle.dump(all_extractions, open(ckpt_raw, 'wb'))
#     print(f"\nExtracted {len(all_extractions)} problems\n")

# # Free VRAM before classification
# del model
# torch.cuda.empty_cache()
# gc.collect()

# # Phase 2: Classify ALL sentences with semantic LLM
# print("="*60)
# print("PHASE 2: Semantic classification of ALL sentences...")
# print("="*60)

# all_sentences = []
# sentence_refs = []
# for pid, data in all_extractions.items():
#     for i, feat in enumerate(data['sentence_features']):
#         all_sentences.append(feat['sentence'])
#         sentence_refs.append((pid, i))

# print(f"Total sentences to classify: {len(all_sentences)}")

# classifications = classify_all_sentences(all_sentences, batch_size=BATCH_SIZE_CLASSIFY)

# # Apply classifications
# for (pid, feat_idx), cls in zip(sentence_refs, classifications):
#     all_extractions[pid]['sentence_features'][feat_idx]['stage'] = cls
#     all_extractions[pid]['sentence_features'][feat_idx]['is_anchor'] = (cls != "NEUTRAL")

# # Save updated extractions
# pickle.dump(all_extractions, open(ckpt_raw, 'wb'))

# # Flatten to features
# print("\nFlattening to final format...")
# all_features = []
# all_features_with_neutral = []

# for pid, data in all_extractions.items():
#     for feat in data['sentence_features']:
#         entry = {
#             'hidden_state': feat['hidden_state'],
#             'hidden_state_last': feat['hidden_state_last'],
#             'problem_id': pid,
#             'sentence_idx': feat['sentence_idx'],
#             'sentence': feat['sentence'],
#             'text': feat['sentence'],
#             'stage': feat['stage'],
#             'is_anchor': feat['is_anchor'],
#         }
#         all_features_with_neutral.append(entry)
#         if feat['stage'] != "NEUTRAL":
#             all_features.append(entry)

# # Save
# ckpt_features = f"{checkpoint_dir}/all_sentences_features.pkl"
# pickle.dump(all_features, open(ckpt_features, 'wb'))
# pickle.dump(all_features_with_neutral, open(f"{checkpoint_dir}/all_sentences_features_with_neutral.pkl", 'wb'))

# cot_data = {pid: {'problem': d['problem'], 'cot': d['cot'], 'sentences': d['sentences']} 
#             for pid, d in all_extractions.items()}
# pickle.dump(cot_data, open(f"{checkpoint_dir}/cot_data.pkl", 'wb'))

# print(f"\n{'='*60}")
# print(f"DONE!")
# print(f"  Non-neutral features: {len(all_features)} -> all_sentences_features.pkl")
# print(f"  All features: {len(all_features_with_neutral)} -> all_sentences_features_with_neutral.pkl")
# print(f"{'='*60}")

# # Stats
# stage_counts = {}
# for cls in classifications:
#     stage_counts[cls] = stage_counts.get(cls, 0) + 1

# print("\nStage distribution:")
# for cls in CLASSES_ORDERED:
#     print(f"  {cls}: {stage_counts.get(cls, 0)}")



import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import re
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "rpc_dataset_math500_layer_final_500"
os.makedirs(checkpoint_dir, exist_ok=True)
dtype = torch.bfloat16

model_ft_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model_base_name = "Qwen/Qwen2.5-14B"
dataset_path = "HuggingFaceH4/MATH-500" #"openai/gsm8k" 
number_cots = 500
EXTRACT_LAYER = -1
BATCH_SIZE_CLASSIFY = 4

CLASSES_ORDERED = [
    "NEUTRAL",
    "PROBLEM_SETUP", 
    "FACT_RETRIEVAL",
    "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT",
    "SELF_CHECKING",
    "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION",
    "FINAL_ANSWER_EMISSION"
]

print(f"Model: {model_ft_name}")
print(f"Target: {number_cots} problems")

# --- LOAD MODEL ONCE ---
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

hf_model = AutoModelForCausalLM.from_pretrained(model_ft_name, torch_dtype=dtype, device_map="cuda")
model = HookedTransformer.from_pretrained_no_processing(model_base_name, hf_model=hf_model, device=device, dtype=dtype)
del hf_model
torch.cuda.empty_cache()
print("Model loaded\n")

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

@torch.no_grad()
def get_sentence_token_ranges(full_text, sentences, tokenizer):
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    ranges = []
    
    tokens = [tokenizer.decode([tid]) for tid in full_ids]
    cumulative_chars = 0
    char_to_token = {}
    for i, tok in enumerate(tokens):
        for _ in range(len(tok)):
            char_to_token[cumulative_chars] = i
            cumulative_chars += 1
    
    current_char = 0
    for sent in sentences:
        start_char = full_text.find(sent, current_char)
        if start_char == -1:
            start_char = full_text.find(sent.strip(), current_char)
        if start_char == -1:
            start_char = current_char
        
        end_char = start_char + len(sent)
        start_tok = char_to_token.get(start_char, 0)
        end_tok = char_to_token.get(min(end_char - 1, max(char_to_token.keys()) if char_to_token else 0), len(full_ids) - 1) + 1
        
        ranges.append((start_tok, min(end_tok, len(full_ids))))
        current_char = end_char
    
    return ranges

@torch.no_grad()
def process_single_problem(problem, model, tokenizer, extract_layer=28):
    input_ids = tokenizer.encode(problem, return_tensors='pt').to(device)
    
    output_ids = model.generate(input_ids, max_new_tokens=500, temperature=0.6, do_sample=True, top_p=0.9)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    cot = full_text[len(problem):].strip() if full_text.startswith(problem) else full_text
    sentences = split_into_sentences(cot)
    
    if len(sentences) < 2:
        return None
    
    full_prompt = problem + " " + cot
    prompt_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    seq_len = prompt_ids.size(1)
    
    _, cache = model.run_with_cache(prompt_ids, names_filter=lambda n: f"blocks.{extract_layer}" in n and "resid_post" in n)
    
    hidden = cache[f"blocks.{extract_layer}.hook_resid_post"][0].float().cpu()
    
    del cache
    torch.cuda.empty_cache()
    
    sentence_ranges = get_sentence_token_ranges(full_prompt, sentences, tokenizer)
    
    sentence_features = []
    for sent_idx, (start, end) in enumerate(sentence_ranges):
        if start >= seq_len:
            continue
        end = min(end, seq_len)
        ctx_end = end
        if ctx_end <= 0:
            continue
            
        mean_pooled = hidden[:ctx_end].mean(dim=0).numpy()
        last_token = hidden[ctx_end - 1].numpy()
        
        sentence_features.append({
            'sentence_idx': sent_idx,
            'sentence': sentences[sent_idx],
            'hidden_state': mean_pooled,
            'hidden_state_last': last_token,
            'token_range': (start, end),
        })
    
    return {
        'problem': problem,
        'cot': cot,
        'sentences': sentences,
        'sentence_features': sentence_features,
    }

def get_classification_prompt(sentence):
    return f"""Classify this reasoning step.
If multiple categories apply, choose the one lowest in this list (the most 'active' one).

1. NEUTRAL: Filler, conversational bridges, or irrelevant text.
2. PROBLEM_SETUP: Re-stating facts or identifying variables.
3. FACT_RETRIEVAL: Recalling formulas (e.g., "Area is pi*r^2").
4. PLAN_GENERATION: Stating what to do next.
5. UNCERTAINTY_MANAGEMENT: Expressing confusion or re-evaluating.
6. SELF_CHECKING: Verifying if a previous step was correct.
7. RESULT_CONSOLIDATION: Summarizing intermediate steps.
8. ACTIVE_COMPUTATION: Performing actual math or algebraic manipulation.
9. FINAL_ANSWER_EMISSION: The definitive final answer statement.

Sentence: "{sentence[:300]}"
Category:"""

@torch.no_grad()
def classify_all_sentences(sentences, batch_size=4):
    from transformers import AutoModelForCausalLM, AutoTokenizer as AT
    
    print("  Loading Qwen-7B classifier...")
    clf_name = "Qwen/Qwen2.5-7B-Instruct"
    clf_tokenizer = AT.from_pretrained(clf_name, trust_remote_code=True)
    clf_tokenizer.pad_token = clf_tokenizer.eos_token
    classifier = AutoModelForCausalLM.from_pretrained(clf_name, torch_dtype=dtype, device_map="cuda", trust_remote_code=True)
    classifier.eval()
    
    results = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="  Classifying"):
        batch = sentences[i:i + batch_size]
        prompts = [get_classification_prompt(s) for s in batch]
        
        inputs = clf_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = classifier.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_new_tokens=5, 
            do_sample=False, 
            pad_token_id=clf_tokenizer.pad_token_id
        )
        
        for out in outputs:
            response = clf_tokenizer.decode(out[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response_clean = response.strip().upper()
            
            matched = "NEUTRAL"
            for cls in reversed(CLASSES_ORDERED):
                if cls in response_clean:
                    matched = cls
                    break
            results.append(matched)
        
        if (i // batch_size) % 50 == 0:
            torch.cuda.empty_cache()
    
    del classifier, clf_tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return results

# --- MAIN EXTRACTION ---
print("Loading dataset...")
dataset = load_dataset(dataset_path, 'main', split=f"train[:{number_cots}]")
problems = [item['problem'] for item in dataset]
print(f"Loaded {len(problems)} problems\n")

# Phase 1: Generate CoT and extract activations
print("="*60)
print("PHASE 1: Generating CoTs and extracting activations...")
print("="*60)

ckpt_raw = f"{checkpoint_dir}/raw_extractions.pkl"
if os.path.exists(ckpt_raw):
    print("Loading cached extractions...")
    all_extractions = pickle.load(open(ckpt_raw, 'rb'))
    print(f"Loaded {len(all_extractions)} cached problems\n")
else:
    all_extractions = {}
    for pid, problem in enumerate(tqdm(problems, desc="Processing")):
        result = process_single_problem(problem, model, tokenizer, extract_layer=EXTRACT_LAYER)
        if result:
            all_extractions[pid] = result
        
        if (pid + 1) % 50 == 0:
            pickle.dump(all_extractions, open(ckpt_raw, 'wb'))
            torch.cuda.empty_cache()
            gc.collect()
    
    pickle.dump(all_extractions, open(ckpt_raw, 'wb'))
    print(f"\nExtracted {len(all_extractions)} problems\n")

# Free VRAM before classification
del model
torch.cuda.empty_cache()
gc.collect()

# Phase 2: Classify ALL sentences with semantic LLM
print("="*60)
print("PHASE 2: Semantic classification of ALL sentences...")
print("="*60)

all_sentences = []
sentence_refs = []
for pid, data in all_extractions.items():
    for i, feat in enumerate(data['sentence_features']):
        all_sentences.append(feat['sentence'])
        sentence_refs.append((pid, i))

print(f"Total sentences to classify: {len(all_sentences)}")

classifications = classify_all_sentences(all_sentences, batch_size=BATCH_SIZE_CLASSIFY)

# Apply classifications
for (pid, feat_idx), cls in zip(sentence_refs, classifications):
    all_extractions[pid]['sentence_features'][feat_idx]['stage'] = cls
    all_extractions[pid]['sentence_features'][feat_idx]['is_anchor'] = (cls != "NEUTRAL")

# Save updated extractions
pickle.dump(all_extractions, open(ckpt_raw, 'wb'))

# Flatten to features
print("\nFlattening to final format...")
all_features = []
all_features_with_neutral = []

for pid, data in all_extractions.items():
    for feat in data['sentence_features']:
        entry = {
            'hidden_state': feat['hidden_state'],
            'hidden_state_last': feat['hidden_state_last'],
            'problem_id': pid,
            'sentence_idx': feat['sentence_idx'],
            'sentence': feat['sentence'],
            'text': feat['sentence'],
            'stage': feat['stage'],
            'is_anchor': feat['is_anchor'],
        }
        all_features_with_neutral.append(entry)
        if feat['stage'] != "NEUTRAL":
            all_features.append(entry)

# Save
ckpt_features = f"{checkpoint_dir}/all_sentences_features.pkl"
pickle.dump(all_features, open(ckpt_features, 'wb'))
pickle.dump(all_features_with_neutral, open(f"{checkpoint_dir}/all_sentences_features_with_neutral.pkl", 'wb'))

cot_data = {pid: {'problem': d['problem'], 'cot': d['cot'], 'sentences': d['sentences']} 
            for pid, d in all_extractions.items()}
pickle.dump(cot_data, open(f"{checkpoint_dir}/cot_data.pkl", 'wb'))

print(f"\n{'='*60}")
print(f"DONE!")
print(f"  Non-neutral features: {len(all_features)} -> all_sentences_features.pkl")
print(f"  All features: {len(all_features_with_neutral)} -> all_sentences_features_with_neutral.pkl")
print(f"{'='*60}")

# Stats
stage_counts = {}
for cls in classifications:
    stage_counts[cls] = stage_counts.get(cls, 0) + 1

print("\nStage distribution:")
for cls in CLASSES_ORDERED:
    print(f"  {cls}: {stage_counts.get(cls, 0)}")