import os
import gc
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from tqdm import tqdm
from contrastive_gen.models import SDSSwitch, SDSRegime

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "rpc_dataset"  
os.makedirs(checkpoint_dir, exist_ok=True)
dtype = torch.bfloat16
SKIP_CACHE = False

model_ft_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" 
model_base_name = "Qwen/Qwen2.5-14B"
dataset_path = "openai/gsm8k"
number_cots = 100

K = 4
hidden_dim = 128


hf_model = AutoModelForCausalLM.from_pretrained(
    model_ft_name, torch_dtype=dtype, device_map="cuda"
)
model_tl = HookedTransformer.from_pretrained_no_processing(
    model_base_name, hf_model=hf_model, device=device, dtype=dtype
)
del hf_model
torch.cuda.empty_cache()
gc.collect()
print(" Finetuned Model loaded\n")



ckpt_features = f"{checkpoint_dir}/all_sentences_features.pkl"
if os.path.exists(ckpt_features) and not SKIP_CACHE:
    print(f"Loading cached features...")
    all_features_full = pickle.load(open(ckpt_features, 'rb'))
    all_features = [f for f in all_features_full if f['problem_id'] < number_cots]
    print(f"Loaded {len(all_features)} cached features (limited to first {number_cots} problems)\n")
else:
    raise RuntimeError("No cached features found. Please run create_dataset.py to generate features before running this script.")


print("BUILD SEQUENCES")

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

print("SDS ARCHITECTURE")




switch = SDSSwitch(d, K, hidden_dim).to(device)
regimes = nn.ModuleList([SDSRegime(d, hidden_dim) for _ in range(K)]).to(device)
optimizer = optim.Adam(list(switch.parameters()) + list(regimes.parameters()), lr=1e-3)

print(f"Switch params: {sum(p.numel() for p in switch.parameters())}")
print(f"Regimes params: {sum(p.numel() for p in regimes.parameters())}")
print(f"Total: {sum(p.numel() for p in switch.parameters()) + sum(p.numel() for p in regimes.parameters())}\n")

print("TRAINING REGIME SWITCHING (SDS)")


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

print("MAPPING REGIMES TO HUMAN LABELS (BASE MODEL)")

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


print("REGIME TIMELINE ANALYSIS ")

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

