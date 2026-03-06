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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
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
print("Base Model loaded\n")

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

dataset = load_dataset("openai/gsm8k", 'main', split="train[:100]")
problems = [item['question'] for item in dataset]
print(f"Loaded {len(problems)} problems\n")

ckpt_cot = f"{checkpoint_dir}/cot_data.pkl"
if os.path.exists(ckpt_cot) and not SKIP_CACHE:
    print(f"Loading cached CoT data...")
    all_cot_data_full = pickle.load(open(ckpt_cot, 'rb'))
    all_cot_data = {k: v for k, v in all_cot_data_full.items() if k < 100}
    print(f"Loaded {len(all_cot_data)} cached problems (limited to 100)\n")
else:
    all_cot_data = {}
    for pid, problem in enumerate(problems):
        print(f"[{pid}] Generating CoT (BASE)...", flush=True)
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

ckpt_features = f"{checkpoint_dir}/all_sentences_features.pkl"
if os.path.exists(ckpt_features) and not SKIP_CACHE:
    print(f"Loading cached features...")
    all_features_full = pickle.load(open(ckpt_features, 'rb'))
    all_features = [f for f in all_features_full if f['problem_id'] < 100]
    print(f"Loaded {len(all_features)} cached features (limited to first 100 problems)\n")
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
    print(f"\nExtracted and saved {len(all_features)} features\n")

del model_tl
torch.cuda.empty_cache()
gc.collect()

print("PHASE 4: PREPROCESSING (PCA)")

X_all = np.array([f['hidden_state'] for f in all_features])
print(f"Raw activation shape: {X_all.shape}")

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_all)

pca_50 = PCA(n_components=0.6732)
X_pca = pca_50.fit_transform(X_norm)
d = X_pca.shape[1]
variance_50 = pca_50.explained_variance_ratio_.sum()

pca_100 = PCA(n_components=100)
X_pca_100 = pca_100.fit_transform(X_norm)
variance_100 = pca_100.explained_variance_ratio_.sum()

print(f"PCA shape: {X_pca.shape}")
print(f"\n*** DIMENSIONALITY ANALYSIS ***")
print(f"Top 50 components explain: {variance_50:.4f}")
print(f"Top 100 components explain: {variance_100:.4f}")


stage_to_idx = {stage: i for i, stage in enumerate(sorted(set(f['stage'] for f in all_features)))}
idx_to_stage = {v: k for k, v in stage_to_idx.items()}

for f in all_features:
    f['stage_idx'] = stage_to_idx[f['stage']]

num_stages = len(stage_to_idx)

print("PHASE 5: BUILD SEQUENCES")

problem_to_indices = defaultdict(list)
for i, f in enumerate(all_features):
    problem_to_indices[f['problem_id']].append(i)

sequences = []
for pid in sorted(problem_to_indices.keys()):
    indices = problem_to_indices[pid]
    if len(indices) < 5:
        continue
    z_seq = torch.from_numpy(X_pca[indices]).float().to(device)
    sequences.append((z_seq, pid))

print(f"Built {len(sequences)} sequences")
lengths = [s[0].shape[0] for s in sequences]
print(f"  Avg length: {np.mean(lengths):.1f}, min: {min(lengths)}, max: {max(lengths)}\n")

print("PHASE 6: SDS ARCHITECTURE")

K = 4
hidden_dim = 128

class SDSSwitch(nn.Module):
    """Gating network: decide which dynamical regime is active."""
    def __init__(self, d, K, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LayerNorm(hidden),
            nn.Softplus(),
            nn.Linear(hidden, K, bias=False)
        )
    
    def forward(self, z):
        return self.net(z)

class SDSRegime(nn.Module):
    """A dynamical rule: z_{t+1} = f(z_t)"""
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Softplus(),
            nn.Linear(hidden, d)
        )
    
    def forward(self, z):
        return self.net(z)

switch = SDSSwitch(d, K, hidden_dim).to(device)
regimes = nn.ModuleList([SDSRegime(d, hidden_dim) for _ in range(K)]).to(device)
optimizer = optim.Adam(list(switch.parameters()) + list(regimes.parameters()), lr=1e-3)

print(f"Switch params: {sum(p.numel() for p in switch.parameters())}")
print(f"Regimes params: {sum(p.numel() for p in regimes.parameters())}")
print(f"Total: {sum(p.numel() for p in switch.parameters()) + sum(p.numel() for p in regimes.parameters())}\n")

print("PHASE 7: TRAINING REGIME SWITCHING (SDS)")


num_epochs = 100
beta_diversity = 15.0
persistence_weight = 25.0

for epoch in range(num_epochs):
    total_loss = 0
    batch_usage = torch.zeros(K).to(device)
    tau = max(0.5, 2.0 * (0.9 ** epoch))
    
    for z_seq, pid in sequences:
        z_curr = z_seq[:-1]
        z_next = z_seq[1:]
        
        logits_s = switch(z_curr)
        s_dist = F.gumbel_softmax(logits_s, tau=tau, hard=False)
        batch_usage += s_dist.detach().sum(dim=0)
        
        preds = torch.stack([r(z_curr) for r in regimes], dim=1)
        errors = torch.mean((preds - z_next.unsqueeze(1))**2, dim=-1)
        l_dynamics = (s_dist * errors).sum(dim=1).mean()
        
        avg_usage = s_dist.mean(dim=0)
        l_div = (avg_usage * torch.log(avg_usage + 1e-8)).sum()
        
        l_persistence = torch.abs(s_dist[1:] - s_dist[:-1]).mean()
        
        top2, _ = torch.topk(logits_s, 2, dim=-1)
        l_margin = torch.clamp(1.0 - (top2[:, 0] - top2[:, 1]), min=0).mean()
        
        loss = (10.0 * l_dynamics) + \
               (beta_diversity * l_div) + \
               (2.0 * l_margin) + \
               (persistence_weight * l_persistence)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        usage = (batch_usage / batch_usage.sum()) * 100
        usage_str = " | ".join([f"R{i}:{usage[i]:.1f}%" for i in range(K)])
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(sequences):.4f} | Usage: [{usage_str}]")

print("PHASE 8: MAPPING REGIMES TO HUMAN LABELS (BASE MODEL)")

regime_to_stage = defaultdict(Counter)
switch.train()

with torch.no_grad():
    for z_seq, pid in sequences:
        logits = switch(z_seq[:-1])
        states = torch.argmax(logits, dim=-1)
        
        indices = problem_to_indices[pid]
        
        for t, state_idx in enumerate(states):
            original_stage = all_features[indices[t]]['stage']
            if original_stage != "NEUTRAL":
                regime_to_stage[state_idx.item()][original_stage] += 1

for r in range(K):
    print(f"Regime {r} Signature (What this regime handles):")
    top_labels = regime_to_stage[r].most_common(3)
    if top_labels:
        for label, count in top_labels:
            print(f"  - {label}: {count} tokens")
    else:
        print(f" - No significant specialization")
    print()

print("INTERPRETATION: Regimes in MODEL")


print("PHASE 9: REGIME TIMELINE ANALYSIS ")

def analyze_regime_persistence(sequences, switch, problem_to_indices):
    """Analyze how stable regimes are (measure of state differentiation)."""
    switch.eval()
    
    persistence_scores = []
    regime_sequences = []
    
    with torch.no_grad():
        for z_seq, pid in sequences:
            logits = switch(z_seq[:-1])
            regimes = torch.argmax(logits, dim=-1).cpu().numpy()
            
            regime_sequences.append((pid, regimes))
            
            if len(regimes) > 1:
                switches = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
                persistence = 1.0 - (switches / (len(regimes) - 1))
                persistence_scores.append(persistence)
    
    mean_persistence = np.mean(persistence_scores) if persistence_scores else 0
    print(f"Regime Persistence Statistics:")
    print(f"  Mean persistence: {mean_persistence:.3f}")
    print(f"  Median persistence: {np.median(persistence_scores):.3f}")
    print(f"  Min persistence: {np.min(persistence_scores):.3f}")
    print(f"  Max persistence: {np.max(persistence_scores):.3f}")
    print(f"  Std persistence: {np.std(persistence_scores):.3f}\n")

    
    print(f"\nExample regime sequences (first 5 problems):")
    for i in range(min(5, len(regime_sequences))):
        pid, seq = regime_sequences[i]
        regime_str = " → ".join([f"R{r}" for r in seq])
        print(f"  Problem {pid}: {regime_str}")
    
    return mean_persistence, regime_sequences

def analyze_regime_transitions(regime_sequences, K):
    """Analyze which regime transitions are preferred by the model."""
    transition_matrix = np.zeros((K, K))
    
    for pid, regimes in regime_sequences:
        for i in range(len(regimes) - 1):
            current = int(regimes[i])
            next_state = int(regimes[i + 1])
            transition_matrix[current, next_state] += 1
    
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_prob = np.divide(transition_matrix, row_sums, where=row_sums != 0, out=np.zeros_like(transition_matrix))
    
    print("Regime Transition Probabilities:")
    print("       ", " ".join([f"→R{i}" for i in range(K)]))
    for i in range(K):
        row = transition_prob[i]
        print(f"R{i}:   ", " ".join([f"{p:5.2f}" for p in row]))
    
    print(f"\nMost Common Transitions:")
    transitions = []
    for i in range(K):
        for j in range(K):
            if i != j:
                count = transition_matrix[i, j]
                if count > 0:
                    transitions.append((i, j, count))
    
    transitions.sort(key=lambda x: -x[2])
    for i, (src, dst, count) in enumerate(transitions[:5]):
        print(f"  {i+1}. R{src} → R{dst}: {int(count)} times")
    
    return transition_matrix, transition_prob

mean_persistence, regime_sequences = analyze_regime_persistence(sequences, switch, problem_to_indices)

print("REGIME TRANSITION MATRIX ")

transition_matrix, transition_prob = analyze_regime_transitions(regime_sequences, K)

