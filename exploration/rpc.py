from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import pickle
import os
import re
import gc


device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "checkpoints_causal_anchors_full_500"
os.makedirs(checkpoint_dir, exist_ok=True)
dtype = torch.bfloat16

model_ft_name = "deepseek-ai/deepseek-r1-distill-qwen-14b"
model_base_name = "Qwen/Qwen2.5-14B"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)

hf_model = AutoModelForCausalLM.from_pretrained(
    model_ft_name,
    torch_dtype=dtype,
    device_map="cuda",     
)

model_tl = HookedTransformer.from_pretrained_no_processing(
    model_base_name,
    hf_model=hf_model,
    device=device,
    dtype=dtype,
)
del hf_model
torch.cuda.empty_cache()
gc.collect()
print("TL model loaded.\n", flush=True)


ANCHOR_CLASSES = {
    "PROBLEM_SETUP": "Parsing or rephrasing the problem",
    "PLAN_GENERATION": "Stating or deciding on a plan of action, meta-reasoning",
    "FACT_RETRIEVAL": "Recalling facts, formulas, problem details without computation",
    "ACTIVE_COMPUTATION": "Algebra, calculations, or other manipulations toward the answer",
    "UNCERTAINTY_MANAGEMENT": "Expressing confusion, re-evaluating, including backtracking",
    "RESULT_CONSOLIDATION": "Aggregating intermediate results, summarizing, or preparing",
    "SELF_CHECKING": "Verifying previous steps, checking calculations, and re-confirmations",
    "FINAL_ANSWER_EMISSION": "Explicitly stating the final answer"
}

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
    
    for source_idx in tqdm(range(M), desc="  Masking sentences", leave=False):
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

def get_hidden_state(text, layer=-1):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        _, cache = model_tl.run_with_cache(input_ids)
        hidden = cache["resid_post", layer][0, -1, :].float().cpu().numpy()
    del cache, input_ids
    torch.cuda.empty_cache()
    gc.collect()
    return hidden

# LOAD DATASET
print("="*80)
print("LOADING DATASET")
print("="*80 + "\n")
dataset = load_dataset("openai/gsm8k", 'main', split="train[:500]")
problems = [item['question'] for item in dataset]
print(f"✓ Loaded {len(problems)} problems\n")

# GENERATE CoTs & CAUSAL MATRICES
print("="*80)
print("PHASE 1: GENERATE CoTs & COMPUTE CAUSAL MATRICES")
print("="*80 + "\n")

ckpt = f"{checkpoint_dir}/causal_data.pkl"
if os.path.exists(ckpt):
    print("Loading cached causal data...")
    all_data = pickle.load(open(ckpt, 'rb'))
    print(f" Loaded {len(all_data)} cached problems\n")
else:
    all_data = {}
    for pid, problem in enumerate(problems):
        print(f"[Problem {pid}] {problem[:70]}...", flush=True)
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

        print(f"  CoT: {len(sentences)} sentences, {len(cot.split())} tokens")
        causal_matrix = get_causal_matrix(model_tl, tokenizer, cot, sentences, problem)
        all_data[pid] = {
            'problem': problem,
            'cot': cot,
            'sentences': sentences,
            'causal_matrix': causal_matrix
        }
        del output_ids, input_ids
        torch.cuda.empty_cache()
        gc.collect()

    with open(ckpt, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"\n Saved causal data for {len(all_data)} problems\n")

# DELETE TL MODEL & LOAD BASE MODEL FOR CLASSIFICATION
print("="*80)
print("PHASE 2: CLASSIFY SENTENCES (DELETE TL, LOAD BASE)")
print("="*80 + "\n")

del model_tl
torch.cuda.empty_cache()
gc.collect()

print("Loading base model for classification...")
model_base = AutoModelForCausalLM.from_pretrained(model_base_name, trust_remote_code=True, torch_dtype=dtype).to(device)
print(" Base model loaded.\n")

# CLASSIFICATION FUNCTION

def classify_sentence(sentence):
    classes_list = "\n".join([f"- {k}: {v}" for k, v in ANCHOR_CLASSES.items()])
    prompt = f"""Classify this reasoning sentence into one of these categories:

{classes_list}

Sentence: "{sentence}"

Category:"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model_base.generate(input_ids, max_new_tokens=5, temperature=0.1, do_sample=False)
    response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    category = response.strip().upper().split()[0]
    del input_ids, output_ids
    if category in ANCHOR_CLASSES: return category
    return "PLAN_GENERATION"

# EXTRACT ANCHORS

print("="*80)
print("PHASE 2: EXTRACTING THOUGHT ANCHORS")
print("="*80 + "\n")

all_anchors = {}
total_anchors = 0
class_dist = {}

for pid, data in all_data.items():
    sentences = data['sentences']
    causal_matrix = data['causal_matrix']
    print(f"[Problem {pid}] Classifying {len(sentences)} sentences...", flush=True)
    
    classifications = [classify_sentence(s) for s in sentences]
    outgoing = np.sum(np.abs(causal_matrix), axis=1)
    importance = outgoing
    threshold = np.percentile(importance[importance > 0], 50) if np.any(importance > 0) else 0
    
    anchors = []
    for idx, imp in enumerate(importance):
        if imp > threshold:
            anchor_class = classifications[idx]
            anchors.append({
                'idx': idx,
                'text': sentences[idx],
                'class': anchor_class,
                'importance': imp,
                'outgoing': outgoing[idx]
            })
            class_dist[anchor_class] = class_dist.get(anchor_class, 0) + 1
            total_anchors += 1
    
    all_anchors[pid] = anchors
    print(f"   {len(anchors)}/{len(sentences)} anchors selected (threshold={threshold:.4f})")
    for anchor in anchors:
        print(f"    POS[{anchor['idx']:2d}] {anchor['class']:25s} | imp={anchor['importance']:.4f} | {anchor['text'][:60]}...")
    
    del classifications
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*80}")
print(f"TOTAL ANCHORS EXTRACTED: {total_anchors}")
print(f"{'='*80}")
print("\nAnchor Class Distribution:")
for cls, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
    pct = 100 * count / total_anchors if total_anchors > 0 else 0
    bar = "█" * int(pct / 5)
    print(f"  {cls:25s}: {count:3d} ({pct:5.1f}%) {bar}")

del model_base
torch.cuda.empty_cache()
gc.collect()

# RELOAD TL FOR FEATURE EXTRACTION
print("\n" + "="*80)
print("PHASE 3: RELOAD TL FOR FEATURE EXTRACTION")
print("="*80 + "\n")

print("Loading tokenizer and reasoning model for features...")
tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)

hf_model = AutoModelForCausalLM.from_pretrained(
    model_ft_name,
    torch_dtype=dtype,
    device_map="cuda",     
)

model_tl = HookedTransformer.from_pretrained_no_processing(
    model_base_name,
    hf_model=hf_model,
    device=device,
    dtype=dtype,
)
del hf_model
torch.cuda.empty_cache()
gc.collect()
print(" TL model loaded for feature extraction.\n")

# EXTRACT FEATURES

ckpt_features = f"{checkpoint_dir}/features.pkl"
if os.path.exists(ckpt_features):
    print("Loading cached features...")
    all_features = pickle.load(open(ckpt_features, 'rb'))
    print(f" Loaded {len(all_features)} cached features\n")
else:
    all_features = []
    for pid, anchors in tqdm(all_anchors.items(), desc="Extracting features"):
        data = all_data[pid]
        problem = data['problem']
        sentences = data['sentences']
        causal_matrix = data['causal_matrix']
        
        for anchor in anchors:
            idx = anchor['idx']
            text_before = problem + " " + " ".join(sentences[:idx])
            hidden_state = get_hidden_state(text_before)
            outgoing_feature = np.sum(np.abs(causal_matrix[idx, :]))
            all_features.append({
                'hidden_state': hidden_state,
                'class': anchor['class'],
                'sentence': anchor['text'],
                'problem_id': pid,
                'sentence_idx': idx,
                'importance': anchor['importance'],
                'outgoing': outgoing_feature
            })
    
    pickle.dump(all_features, open(ckpt_features, 'wb'))
    print(f"\n Extracted {len(all_features)} features\n")

del model_tl
torch.cuda.empty_cache()
gc.collect()

# TRAIN CLASSIFIERS
print("="*80)
print("PHASE 4: FILTER CLASSES & TRAIN CLASSIFIERS")
print("="*80 + "\n")

# Filter rare classes
class_counts = {}
for f in all_features:
    cls = f['class']
    class_counts[cls] = class_counts.get(cls, 0) + 1

print("Feature Class Distribution (before filtering):")
for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    pct = 100 * count / len(all_features)
    bar = "█" * int(pct / 5)
    print(f"  {cls:25s}: {count:3d} ({pct:5.1f}%) {bar}")

min_samples = 2
valid_classes = {cls for cls, count in class_counts.items() if count >= min_samples}
removed_classes = {cls for cls, count in class_counts.items() if count < min_samples}

if removed_classes:
    print(f"\nRemoving classes with < {min_samples} samples:")
    for cls in removed_classes:
        print(f"  {cls}: {class_counts[cls]} samples")

all_features = [f for f in all_features if f['class'] in valid_classes]

print(f"\n Remaining features: {len(all_features)}")
print("\nFeature Class Distribution (after filtering):")
new_class_counts = {}
for f in all_features:
    cls = f['class']
    new_class_counts[cls] = new_class_counts.get(cls, 0) + 1

for cls, count in sorted(new_class_counts.items(), key=lambda x: x[1], reverse=True):
    pct = 100 * count / len(all_features)
    bar = "█" * int(pct / 5)
    print(f"  {cls:25s}: {count:3d} ({pct:5.1f}%) {bar}")

X = np.array([f['hidden_state'] for f in all_features])
class_to_idx = {c: i for i, c in enumerate(sorted(valid_classes))}
y = np.array([class_to_idx[f['class']] for f in all_features])

print(f"\nFeature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Number of classes: {len(valid_classes)}\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}\n")

print("Training Logistic Regression...")
clf_lr = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42)
clf_lr.fit(X_train, y_train)
train_acc_lr = clf_lr.score(X_train, y_train)
test_acc_lr = clf_lr.score(X_test, y_test)
print(f"  Train accuracy: {train_acc_lr:.4f}")
print(f"  Test accuracy:  {test_acc_lr:.4f}\n")

print("Training MLP...")
clf_mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
clf_mlp.fit(X_train, y_train)
train_acc_mlp = clf_mlp.score(X_train, y_train)
test_acc_mlp = clf_mlp.score(X_test, y_test)
print(f"  Train accuracy: {train_acc_mlp:.4f}")
print(f"  Test accuracy:  {test_acc_mlp:.4f}\n")

# CLASSIFICATION REPORTS

print("="*80)
print("DETAILED CLASSIFICATION REPORTS")
print("="*80 + "\n")

idx_to_class = {v: k for k, v in class_to_idx.items()}
y_pred_lr = clf_lr.predict(X_test)
y_pred_mlp = clf_mlp.predict(X_test)

print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr, target_names=[idx_to_class[i] for i in range(len(idx_to_class))], zero_division=0))

print("\nMLP (256-128):")
print(classification_report(y_test, y_pred_mlp, target_names=[idx_to_class[i] for i in range(len(idx_to_class))], zero_division=0))

# SAVE MODELS
print("="*80)
print("SAVING MODELS")
print("="*80 + "\n")

with open(f"{checkpoint_dir}/classifier_lr.pkl", 'wb') as f:
    pickle.dump(clf_lr, f)
with open(f"{checkpoint_dir}/classifier_mlp.pkl", 'wb') as f:
    pickle.dump(clf_mlp, f)
with open(f"{checkpoint_dir}/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
with open(f"{checkpoint_dir}/class_to_idx.pkl", 'wb') as f:
    pickle.dump(class_to_idx, f)

print(f" Models saved to {checkpoint_dir}/")
print(f"  - classifier_lr.pkl")
print(f"  - classifier_mlp.pkl")
print(f"  - scaler.pkl")
print(f"  - class_to_idx.pkl")

# SUMMARY
print("\n" + "="*80)
print("PIPELINE COMPLETE - SUMMARY")
print("="*80)
print(f"\nDataset:")
print(f"  Problems processed: {len(all_data)}")
print(f"  Total anchors extracted: {total_anchors}")
print(f"  Features for training: {len(all_features)}")

print(f"\nClassifiers:")
print(f"  Logistic Regression:")
print(f"    - Train accuracy: {train_acc_lr:.4f}")
print(f"    - Test accuracy:  {test_acc_lr:.4f}")
print(f"  MLP (256-128):")
print(f"    - Train accuracy: {train_acc_mlp:.4f}")
print(f"    - Test accuracy:  {test_acc_mlp:.4f}")

print(f"\nClasses ({len(valid_classes)}):")
for cls in sorted(valid_classes):
    count = new_class_counts.get(cls, 0)
    print(f"  - {cls:25s}: {count:3d} samples")

print(f"\n{'='*80}")
print("PIPELINE COMPLETE!")
print(f"{'='*80}")