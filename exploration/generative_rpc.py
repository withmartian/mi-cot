import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import re
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "rpc_dataset"
test_checkpoint_dir = "rpc_test_dataset"
os.makedirs(checkpoint_dir, exist_ok=True)
dtype = torch.bfloat16

model_ft_name = "deepseek-ai/deepseek-r1-distill-qwen-14b"
model_base_name = "Qwen/Qwen2.5-14B"

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

# ============================================================================
# PHASE 1: GENERATE CoTs & CAUSAL MATRICES
# ============================================================================

print("="*80)
print("PHASE 1: LOAD MODELS & GENERATE CoTs")
print("="*80 + "\n")

print("Loading models...", flush=True)
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
print(" Models loaded\n")

def split_into_sentences(text):
    """Split text into sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

def get_sentence_token_positions(text, sentences, tokenizer):
    """Get token positions for each sentence"""
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
    """KL divergence between two logit distributions"""
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = F.softmax(logits_p, dim=-1)
    kl = torch.sum(p * (log_p - log_q), dim=-1)
    return kl.mean().item()

def get_causal_matrix(model, tokenizer, cot, sentences, problem=""):
    """Compute causal importance matrix via attention masking"""
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

print("="*80)
print("LOADING DATASET & GENERATING CoTs")
print("="*80 + "\n")

dataset = load_dataset("openai/gsm8k", 'main', split="train[:500]")
problems = [item['question'] for item in dataset]
print(f" Loaded {len(problems)} problems\n")

ckpt_cot = f"{checkpoint_dir}/cot_data.pkl"
if os.path.exists(ckpt_cot):
    print(f"Loading cached CoT data...")
    all_cot_data = pickle.load(open(ckpt_cot, 'rb'))
    print(f" Loaded {len(all_cot_data)} cached problems\n")
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

# ============================================================================
# PHASE 2: IDENTIFY THOUGHT ANCHORS & CLASSIFY
# ============================================================================

print("="*80)
print("PHASE 2: IDENTIFY ANCHORS & CLASSIFY SENTENCES")
print("="*80 + "\n")

del model_tl
torch.cuda.empty_cache()
gc.collect()

print("Loading base model for classification...")
model_base = AutoModelForCausalLM.from_pretrained(
    model_base_name, torch_dtype=dtype
).to(device)
print(" Model loaded\n")

def classify_sentence(sentence):
    """Classify a sentence into one of the reasoning stages"""
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

# Identify anchors (high importance) and classify all sentences
all_sentence_data = []  # Will store ALL sentences
anchor_classes = {}

for pid, data in all_cot_data.items():
    sentences = data['sentences']
    causal_matrix = data['causal_matrix']
    
    print(f"[Problem {pid}] Processing {len(sentences)} sentences...", flush=True)
    
    # Compute importance for each sentence
    outgoing = np.sum(np.abs(causal_matrix), axis=1)
    importance = outgoing
    threshold = np.percentile(importance[importance > 0], 50) if np.any(importance > 0) else 0
    
    # Classify each sentence
    for sent_idx, sentence in enumerate(sentences):
        is_anchor = importance[sent_idx] > threshold
        
        if is_anchor:
            # Classify anchor
            stage = classify_sentence(sentence)
        else:
            # Non-anchor: mark as NEUTRAL
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

# ============================================================================
# PHASE 3: EXTRACT ACTIVATIONS FOR ALL SENTENCES
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: EXTRACT MEAN-POOLED ACTIVATIONS")
print("="*80 + "\n")

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
print(" Models loaded\n")

def get_sentence_activation(problem, sentences, sent_idx, layer=-1):
    """
    Extract mean-pooled activation for tokens in sentence sent_idx.
    NOT accumulated context - just the sentence itself.
    """

    sentence = sentences[sent_idx]
    ctx = problem + " " + " ".join(sentences[:sent_idx+1])
    input_ids = tokenizer.encode(ctx, return_tensors="pt").to(device)
    sent_len = len(tokenizer.encode(sentence, add_special_tokens=False))
    sent_start = input_ids.size(1) - sent_len    
    with torch.no_grad():
        ouputs = model_tl.run_with_cache(input_ids)
        hidden = cache["resid_post", layer][0, sent_start:sent_start+sent_len, :].cpu().float()
        hidden = hidden.mean(dim=0).numpy()
    del input_ids, cache
    torch.cuda.empty_cache()
    gc.collect()
    return hidden 

ckpt_features = f"{checkpoint_dir}/all_sentences_features.pkl"
if os.path.exists(ckpt_features):
    print(f"Loading cached features...")
    all_features = pickle.load(open(ckpt_features, 'rb'))
    print(f" Loaded {len(all_features)} cached features\n")
else:
    all_features = []
    
    for item in tqdm(all_sentence_data, desc="Extracting activations"):
        pid = item['problem_id']
        sent_idx = item['sentence_idx']
        
        problem = all_cot_data[pid]['problem']
        sentences = all_cot_data[pid]['sentences']
        
        hidden_state = get_sentence_activation(problem, sentences, sent_idx, layer=-1)
        
        all_features.append({
            'hidden_state': hidden_state,
            'problem_id': pid,
            'sentence_idx': sent_idx,
            'sentence': item['sentence'],
            'stage': item['stage'],
            'is_anchor': item['is_anchor'],
            'importance': item['importance'],
        })
    
    pickle.dump(all_features, open(ckpt_features, 'wb'))
    print(f"\n Extracted and saved {len(all_features)} features\n")

del model_tl
torch.cuda.empty_cache()
gc.collect()

# ============================================================================
# PHASE 4: PREPROCESSING (NORMALIZE + PCA)
# ============================================================================

print("="*80)
print("PHASE 4: PREPROCESSING")
print("="*80 + "\n")

X_all = np.array([f['hidden_state'] for f in all_features])
print(f"Raw activation shape: {X_all.shape}")
print(f"Raw activation range: [{X_all.min():.4f}, {X_all.max():.4f}]")

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_all)
print(f"After normalization - range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_normalized)
d = X_pca.shape[1]
print(f"After PCA - shape: {X_pca.shape}, explained variance: {pca.explained_variance_ratio_.sum():.4f}\n")

# Create class mapping: reasoning stages + NEUTRAL
stage_to_idx = {stage: i for i, stage in enumerate(sorted(set(f['stage'] for f in all_features)))}
idx_to_stage = {v: k for k, v in stage_to_idx.items()}
num_classes = len(stage_to_idx)

print("Stage mapping:")
for stage, idx in sorted(stage_to_idx.items()):
    count = sum(1 for f in all_features if f['stage'] == stage)
    print(f"  {idx}: {stage:25s} ({count} sentences)")

# ============================================================================
# PHASE 5: BUILD TRAINING PAIRS (2-SENTENCE WINDOWS)
# ============================================================================

print("\n" + "="*80)
print("PHASE 5: BUILD TRAINING PAIRS (CONSECUTIVE SENTENCES)")
print("="*80 + "\n")

# Group by problem
problem_to_indices = defaultdict(list)
for i, f in enumerate(all_features):
    problem_to_indices[f['problem_id']].append(i)

sequences = []
labels = []

for pid in sorted(problem_to_indices.keys()):
    indices = problem_to_indices[pid]
    
    # Create 2-sentence windows for each consecutive pair
    for t in range(len(indices) - 1):
        idx_t_minus_1 = indices[t]
        idx_t = indices[t + 1]
        
        # Sequence: [z_{t-1}, z_t] with labels [y_{t-1}, y_t]
        z_pair = torch.from_numpy(X_pca[[idx_t_minus_1, idx_t]]).float().to(device)
        y_pair = torch.tensor(
            [stage_to_idx[all_features[idx_t_minus_1]['stage']],
             stage_to_idx[all_features[idx_t]['stage']]],
            dtype=torch.long
        ).to(device)
        
        sequences.append(z_pair)
        labels.append(y_pair)

print(f" Built {len(sequences)} training pairs (2-sentence windows)")
print(f"  Each pair: [z_{{t-1}}, z_t] with [y_{{t-1}}, y_t]\n")

# ============================================================================
# PHASE 6: MODEL DEFINITION
# ============================================================================

print("="*80)
print("PHASE 6: MODEL DEFINITION")
print("="*80 + "\n")

K = 4
hidden_dim = 128

class Encoder(nn.Module):
    """Encoder: q(s_t | s_{t-1}, z_{t-1})"""
    def __init__(self, d, K, hidden_dim):
        super().__init__()
        self.K = K
        self.state_embed = nn.Embedding(K, 32)
        self.net = nn.Sequential(
            nn.Linear(d + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K)
        )
    
    def forward(self, z, s_prev):
        s_emb = self.state_embed(s_prev)
        logits = self.net(torch.cat([z, s_emb], dim=-1))
        return logits

class Decoder(nn.Module):
    """Decoder: p(y_t | z_{t-1}, s_t)"""
    def __init__(self, d, K, num_classes, hidden_dim):
        super().__init__()
        self.state_embed = nn.Embedding(K, 32)
        self.net = nn.Sequential(
            nn.Linear(d + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, z, s):
        s_emb = self.state_embed(s)
        logits = self.net(torch.cat([z, s_emb], dim=-1))
        return logits

encoder = Encoder(d, K, hidden_dim).to(device)
decoder = Decoder(d, K, num_classes, hidden_dim).to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

print(f"Encoder params: {sum(p.numel() for p in encoder.parameters())}")
print(f"Decoder params: {sum(p.numel() for p in decoder.parameters())}")
print(f"Total: {sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())}\n")

# ============================================================================
# PHASE 7: TRAINING
# ============================================================================

print("="*80)
print("PHASE 7: TRAINING")
print("="*80 + "\n")

num_epochs = 30
beta_entropy = 5.0
lambda_persist = 0.1

print(f"HYPERPARAMETERS:")
print(f"  K (latent states): {K}")
print(f"  d (PCA dims): {d}")
print(f"  Beta (entropy): {beta_entropy}")
print(f"  Lambda (persist): {lambda_persist}")
print(f"  Epochs: {num_epochs}\n")

loss_history = []

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_pred = 0
    epoch_ent = 0
    epoch_pers = 0
    total_steps = 0
    
    for Z, Y in zip(sequences, labels):
        T = Z.shape[0]
        s_prev = torch.randint(0, K, (1,), dtype=torch.long).to(device)
        
        for t in range(1, T):
            z_prev = Z[t-1:t]
            
            # Encoder
            logits_state = encoder(z_prev, s_prev)
            s_dist = F.softmax(logits_state, dim=-1)
            s_t = torch.argmax(logits_state, dim=-1)
            
            # Decoder
            logits_pred = decoder(z_prev, s_t)
            
            # Loss
            L_pred = F.cross_entropy(logits_pred, Y[t:t+1])
            L_ent = -(s_dist * torch.log(s_dist + 1e-8)).sum()
            L_pers = (s_t != s_prev).float().sum()
            
            loss = L_pred + beta_entropy * L_ent + lambda_persist * L_pers
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0
            )
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_pred += L_pred.item()
            epoch_ent += L_ent.item()
            epoch_pers += L_pers.item()
            total_steps += 1
            
            s_prev = s_t.detach()
    
    epoch_loss /= total_steps
    epoch_pred /= total_steps
    epoch_ent /= total_steps
    epoch_pers /= total_steps
    loss_history.append(epoch_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {epoch_loss:.4f} "
              f"(pred={epoch_pred:.4f}, ent={epoch_ent:.4f}, pers={epoch_pers:.4f})")
        
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_states = defaultdict(int)
                for Z, _ in zip(sequences[:20], labels[:20]):
                    s_prev = torch.randint(0, K, (1,), dtype=torch.long).to(device)
                    for t in range(1, Z.shape[0]):
                        z_prev = Z[t-1:t]
                        logits = encoder(z_prev, s_prev)
                        s_prev = torch.argmax(logits, dim=-1)
                        sample_states[int(s_prev.item())] += 1
                state_dist = [sample_states[k] for k in range(K)]
                print(f"           State check: {state_dist}")

# ============================================================================
# PHASE 8: INFERENCE & VALIDATION
# ============================================================================

print("\n" + "="*80)
print("PHASE 8: INFERENCE & VALIDATION")
print("="*80 + "\n")

encoder.eval()
decoder.eval()

state_sequences = {}
state_persistence = []
state_counts = defaultdict(int)
state_transitions = defaultdict(int)
pred_correct = 0
pred_total = 0

with torch.no_grad():
    for pid, (Z, Y) in enumerate(zip(sequences, labels)):
        s_seq = []
        s_prev = torch.randint(0, K, (1,), dtype=torch.long).to(device)
        
        for t in range(1, Z.shape[0]):
            z_prev = Z[t-1:t]
            logits = encoder(z_prev, s_prev)
            s_t = torch.argmax(logits, dim=-1)
            s_seq.append(int(s_t.item()))
            state_counts[int(s_t.item())] += 1
            
            # Prediction
            logits_pred = decoder(z_prev, s_t)
            pred = torch.argmax(logits_pred, dim=-1).item()
            pred_total += 1
            if pred == int(Y[t].item()):
                pred_correct += 1
            
            s_prev = s_t
        
        state_sequences[pid] = s_seq
        
        # Persistence
        changes = sum(1 for i in range(1, len(s_seq)) if s_seq[i] != s_seq[i-1])
        persistence = 1.0 - (changes / max(len(s_seq) - 1, 1))
        state_persistence.append(persistence)
        
        # Transitions
        for i in range(len(s_seq) - 1):
            state_transitions[(s_seq[i], s_seq[i+1])] += 1

print("RESULTS")
print("-" * 80)

print("\nPERSISTENCE:")
print(f"  Mean: {np.mean(state_persistence):.3f}")
print(f"  Std:  {np.std(state_persistence):.3f}")

print("\nSTATE USAGE:")
total_states = sum(state_counts.values())
for k in range(K):
    count = state_counts[k]
    pct = 100 * count / total_states
    bar = "█" * int(pct / 5)
    print(f"  State {k}: {count:5d} ({pct:5.1f}%) {bar}")

print("\nPREDICTION ACCURACY:")
if pred_total > 0:
    acc = pred_correct / pred_total
    baseline = 1.0 / num_classes
    improvement = (acc - baseline) / baseline * 100
    print(f"  Accuracy: {acc:.3f}")
    print(f"  Baseline: {baseline:.3f}")
    print(f"  Improvement: +{improvement:.1f}%")

print("\nEXAMPLE STATE SEQUENCES:")
for i in range(min(3, len(state_sequences))):
    print(f"  Problem {i}: {state_sequences[i]}")

print("\nCONVERGENCE:")
checks = 0
if np.mean(state_persistence) > 0.6:
    print(f" Persistent (mean={np.mean(state_persistence):.3f})")
    checks += 1
else:
    print(f"  ✗ Not persistent (mean={np.mean(state_persistence):.3f})")

if pred_total > 0 and 'improvement' in locals() and improvement > 20:
    print(f" Strong prediction (+{improvement:.1f}%)")
    checks += 1
else:
    print(f"  ✗ Weak prediction")

if sum(1 for c in state_counts.values() if c > 0) >= K * 0.5:
    print(f" Multiple states ({sum(1 for c in state_counts.values() if c > 0)}/{K})")
    checks += 1
else:
    print(f"  ✗ Limited diversity")

print(f"\n{checks}/3 checks passed")
print("\n" + "="*80)