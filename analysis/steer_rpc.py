import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
K = 4 # Number of regimes to discover
D_LATENT = 32
STEER_STRENGTH = 8 
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "rpc_dataset_layer28_200"
checkpoint_save = "rpc_full_suite"
os.makedirs(checkpoint_save, exist_ok=True)

# --- MODEL DEFINITIONS ---

class RPC_Encoder(nn.Module):
    def __init__(self, d_in, d_h, K):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sigma", torch.ones(d_in))
        
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, d_h)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_h, 128), nn.LayerNorm(128), nn.Softplus(),
            nn.Linear(128, K, bias=False)
        )

    def set_scaling(self, scaler):
        self.mu.copy_(torch.from_numpy(scaler.mean_).float())
        self.sigma.copy_(torch.from_numpy(scaler.scale_).float())

    def forward(self, x, temp=1.0, scaled=True):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if not scaled:
            x = (x - self.mu) / (self.sigma + 1e-8)
        h = F.normalize(self.encoder(x), dim=1)
        logits = self.gate(h)
        s = F.gumbel_softmax(logits, tau=temp, hard=False)
        return h, s

class MoEDynamics(nn.Module):
    def __init__(self, K, dim):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, dim)) 
            for _ in range(K)
        ])

    def forward(self, h, s):
        preds = torch.stack([m(h) for m in self.experts], dim=1)
        return torch.sum(s.unsqueeze(-1) * preds, dim=1)

def nce_loss(z, p, n, temp=0.05):
    pos = (z * p).sum(dim=1, keepdim=True)
    neg = (z * n).sum(dim=1, keepdim=True)
    return F.cross_entropy(torch.cat([pos, neg], dim=1) / temp, torch.zeros(len(z), dtype=torch.long, device=device))

# --- STEERING UTILITIES ---

def get_manifold_steering_vector(x_raw, target_regime, model, dyn, centroids, blend_alpha=0.3):
    model.eval()
    dyn.eval()
    
    if x_raw.dim() == 1:
        x_raw = x_raw.unsqueeze(0)

    with torch.no_grad():
        h_curr, _ = model(x_raw, temp=0.01, scaled=False)
        
        s_target = torch.zeros((1, K), device=device)
        s_target[0, target_regime] = 1.0
        h_ideal_next = dyn(h_curr, s_target)
        h_ideal_next = F.normalize(h_ideal_next, dim=1)
        
        target_point = (1 - blend_alpha) * h_ideal_next + blend_alpha * centroids[target_regime]
        target_point = F.normalize(target_point, dim=1)
        
        latent_delta = target_point - h_curr
        return latent_delta

def apply_orthogonal_steering(x_residual, v_steer_raw, alpha=0.1):
    if x_residual.dim() == 1:
        x_residual = x_residual.unsqueeze(0)
    
    x_unit = x_residual / (x_residual.norm() + 1e-8)
    proj = torch.sum(v_steer_raw * x_unit, dim=1, keepdim=True) * x_unit
    v_perp = v_steer_raw - proj
    
    orig_norm = x_residual.norm()
    x_steered = x_residual + alpha * v_perp
    x_final = x_steered * (orig_norm / x_steered.norm())
    
    return x_final

# --- EXECUTION ---

print("Loading features...")
with open(f"{checkpoint_dir}/all_sentences_features.pkl", 'rb') as f:
    features = pickle.load(f)

# Filter out NEUTRAL (already done in the file, but just in case)
features = [f for f in features if f.get('stage', 'NEUTRAL') != 'NEUTRAL']
print(f"Loaded {len(features)} non-neutral features")

# Stats
stage_counts = {}
for f in features:
    stage_counts[f['stage']] = stage_counts.get(f['stage'], 0) + 1
print("Stage distribution:")
for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
    print(f"  {stage}: {count}")

# Use last token activations for better steering
USE_LAST_TOKEN = True
if USE_LAST_TOKEN and 'hidden_state_last' in features[0]:
    print("\nUsing last-token activations")
    X_raw = np.array([f['hidden_state_last'] for f in features])
else:
    print("\nUsing mean-pooled activations")
    X_raw = np.array([f['hidden_state'] for f in features])

scaler = StandardScaler()
X_scaled = torch.from_numpy(scaler.fit_transform(X_raw)).float().to(device)

# Setup Triplets
p_map = defaultdict(list)
for i, f in enumerate(features): 
    p_map[f['problem_id']].append(i)

triplets = []
for pid, idxs in p_map.items():
    if len(idxs) < 2: continue
    for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, 15), replace=False):
        triplets.append((idxs[t], idxs[t+1], np.random.randint(len(features))))

print(f"Created {len(triplets)} triplets")

# Training Loop
model = RPC_Encoder(X_raw.shape[1], D_LATENT, K).to(device)
model.set_scaling(scaler)
dyn = MoEDynamics(K, D_LATENT).to(device)
optimizer = optim.AdamW(list(model.parameters()) + list(dyn.parameters()), lr=1e-3)

print(f"\nTraining Discovery Engine for K={K} regimes...")
for epoch in range(60):
    tau = max(0.1, 1.5 * (0.92 ** epoch))
    w_div = min(30.0, (epoch / 20.0) * 30.0)
    indices = np.random.permutation(len(triplets))
    for b in range(0, len(triplets), 128):
        b_idx = indices[b:b+128]
        i_t, p_t, n_t = [torch.tensor([triplets[x][idx] for x in b_idx], device=device) for idx in range(3)]
        h_i, s_i = model(X_scaled[i_t], temp=tau)
        h_p, s_p = model(X_scaled[p_t], temp=tau)
        h_n, _ = model(X_scaled[n_t], temp=tau)
        h_pred = dyn(h_i, s_i)
        loss = nce_loss(h_pred, h_p, h_n) + (10.0 * F.mse_loss(h_pred, h_p)) + \
               (w_div * (s_i.mean(0) * torch.log(s_i.mean(0) + 1e-8)).sum()) + \
               (15.0 * torch.abs(s_i - s_p).mean())
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# Compute centroids
print("\nComputing regime centroids...")
model.eval()
with torch.no_grad():
    h_all, s_all = model(X_scaled)
    states = s_all.argmax(1)
    centroids = {}
    for k in range(K):
        mask = states == k
        if mask.sum() > 0:
            centroids[k] = h_all[mask].mean(dim=0, keepdim=True)
        else:
            centroids[k] = torch.zeros(1, D_LATENT, device=device)

# Map discovered regimes to semantic labels
print("\nRegime -> Semantic class mapping:")
regime_to_stage = {}
for k in range(K):
    regime_mask = states == k
    regime_stages = [features[i]['stage'] for i in range(len(features)) if regime_mask[i]]
    if regime_stages:
        most_common = max(set(regime_stages), key=regime_stages.count)
        regime_to_stage[k] = most_common
        print(f"  Regime {k}: {most_common} ({regime_stages.count(most_common)}/{len(regime_stages)})")

regime_names = {k: regime_to_stage.get(k, f"Regime_{k}") for k in range(K)}

# --- STEERING EXAMPLES ---
print("\n" + "="*80)
print("STEERING EXAMPLES: Before vs After (Full Text)")
print("="*80)

n_examples = 5
example_indices = []
for k in range(K):
    regime_samples = (states == k).nonzero(as_tuple=True)[0]
    if len(regime_samples) > 0:
        idx = regime_samples[np.random.randint(len(regime_samples))].item()
        example_indices.append(idx)

with torch.no_grad():
    for i, idx in enumerate(example_indices[:n_examples]):
        sample_x = torch.from_numpy(X_raw[idx]).float().to(device)
        text = features[idx].get('text', features[idx].get('sentence', f'[Sample {idx}]'))
        problem_id = features[idx].get('problem_id', 'N/A')
        semantic_stage = features[idx].get('stage', 'Unknown')
        
        _, s_orig = model(sample_x.unsqueeze(0), temp=0.01, scaled=False)
        orig_regime = s_orig.argmax(1).item()
        
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1} | Problem ID: {problem_id} | Semantic: {semantic_stage}")
        print(f"{'='*80}")
        print(f"\n[TEXT]: {text}")
        print(f"\n[DISCOVERED REGIME]: {orig_regime} ({regime_names.get(orig_regime, '?')})")
        print(f"[PROBS]: {dict(zip([regime_names[k] for k in range(K)], s_orig[0].cpu().numpy().round(3)))}")
        
        target_k = (orig_regime + 2) % K
        
        v_latent = get_manifold_steering_vector(sample_x, target_k, model, dyn, centroids, blend_alpha=0.3)
        v_512 = torch.matmul(v_latent, model.encoder[3].weight)
        v_act = torch.matmul(v_512, model.encoder[0].weight) * model.sigma
        steered_x = apply_orthogonal_steering(sample_x, v_act, alpha=STEER_STRENGTH)
        
        _, s_steered = model(steered_x, temp=0.01, scaled=False)
        new_regime = s_steered.argmax(1).item()
        
        delta = s_steered[0, target_k].item() - s_orig[0, target_k].item()
        success = "SUCCESS ✓" if new_regime == target_k else "PARTIAL ○"
        print(f"\n[TARGET]: {target_k} ({regime_names.get(target_k, '?')})")
        print(f"[RESULT]: {success} | Δprob = {delta:+.3f}")
        print(f"[NEW PROBS]: {dict(zip([regime_names[k] for k in range(K)], s_steered[0].cpu().numpy().round(3)))}")

# --- STEERING SUCCESS RATE ---
print("\n" + "="*80)
print("STEERING SUCCESS RATE")
print("="*80)

n_test = min(200, len(features))
test_indices = np.random.choice(len(features), n_test, replace=False)
success_matrix = np.zeros((K, K))
count_matrix = np.zeros((K, K))

with torch.no_grad():
    for idx in test_indices:
        sample_x = torch.from_numpy(X_raw[idx]).float().to(device)
        _, s_orig = model(sample_x.unsqueeze(0), temp=0.01, scaled=False)
        orig_regime = s_orig.argmax(1).item()
        
        for target_k in range(K):
            if target_k == orig_regime:
                continue
            
            v_latent = get_manifold_steering_vector(sample_x, target_k, model, dyn, centroids, blend_alpha=0.3)
            v_512 = torch.matmul(v_latent, model.encoder[3].weight)
            v_act = torch.matmul(v_512, model.encoder[0].weight) * model.sigma
            steered_x = apply_orthogonal_steering(sample_x, v_act, alpha=STEER_STRENGTH)
            
            _, s_steered = model(steered_x, temp=0.01, scaled=False)
            new_regime = s_steered.argmax(1).item()
            
            count_matrix[orig_regime, target_k] += 1
            if new_regime == target_k:
                success_matrix[orig_regime, target_k] += 1

print(f"\nSuccess rate by (original -> target):")
for orig_k in range(K):
    for target_k in range(K):
        if count_matrix[orig_k, target_k] > 0:
            rate = success_matrix[orig_k, target_k] / count_matrix[orig_k, target_k]
            print(f"  {regime_names[orig_k][:12]:12s} -> {regime_names[target_k][:12]:12s}: {rate:.1%}")

overall = success_matrix.sum() / count_matrix.sum() if count_matrix.sum() > 0 else 0
print(f"\nOverall steering success rate: {overall:.1%}")

# --- LLM CONTINUATION TEST ---
print("\n" + "="*80)
print("LLM CONTINUATION TEST: Behavioral Effect of Steering")
print("="*80)

from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
print(f"\nLoading {LLM_MODEL}...")
tokenizer_llm = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
)
llm.eval()

# Inject at layer 28 (where features were extracted)
HOOK_LAYER = 28
print(f"Injecting at layer {HOOK_LAYER}")

injection_mode = None
steering_delta = None

def steering_hook(module, input, output):
    global injection_mode, steering_delta
    if steering_delta is None:
        return output
    
    hidden = output[0] if isinstance(output, tuple) else output
    
    if injection_mode == "last_token":
        hidden[:, -1, :] = hidden[:, -1, :] + steering_delta
    elif injection_mode == "broadcast":
        hidden = hidden + steering_delta.unsqueeze(0).unsqueeze(0)
    
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

hook_handle = llm.model.layers[HOOK_LAYER].register_forward_hook(steering_hook)

def generate_continuation(prompt, max_new_tokens=60):
    inputs = tokenizer_llm(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer_llm.eos_token_id
        )
    return tokenizer_llm.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

# Load cot_data for context
cot_data_path = f"{checkpoint_dir}/cot_data.pkl"
cot_data = {}
if os.path.exists(cot_data_path):
    with open(cot_data_path, 'rb') as f:
        cot_data = pickle.load(f)

n_tests = 3
test_indices = np.random.choice(len(features), n_tests, replace=False)

for i, idx in enumerate(test_indices):
    feat = features[idx]
    text = feat.get('text', feat.get('sentence', ''))
    problem_id = feat.get('problem_id', 'N/A')
    semantic_stage = feat.get('stage', 'Unknown')
    
    # Build context
    if problem_id in cot_data:
        problem = cot_data[problem_id]['problem']
        sentences = cot_data[problem_id]['sentences']
        sent_idx = feat.get('sentence_idx', 0)
        context = problem + " " + " ".join(sentences[:sent_idx+1])
    else:
        context = text
    
    sample_x = torch.from_numpy(X_raw[idx]).float().to(device)
    
    with torch.no_grad():
        _, s_orig = model(sample_x.unsqueeze(0), temp=0.01, scaled=False)
    orig_regime = s_orig.argmax(1).item()
    target_k = (orig_regime + 2) % K
    
    # Compute steering delta
    v_latent = get_manifold_steering_vector(sample_x, target_k, model, dyn, centroids, blend_alpha=0.3)
    v_512 = torch.matmul(v_latent, model.encoder[3].weight)
    v_act = torch.matmul(v_512, model.encoder[0].weight) * model.sigma
    steered_x = apply_orthogonal_steering(sample_x, v_act, alpha=STEER_STRENGTH)
    delta = (steered_x - sample_x).squeeze(0).to(torch.bfloat16)
    
    print(f"\n{'='*80}")
    print(f"TEST {i+1} | Problem: {problem_id} | Semantic: {semantic_stage}")
    print(f"{'='*80}")
    print(f"\n[CONTEXT]: {context[:300]}..." if len(context) > 300 else f"\n[CONTEXT]: {context}")
    print(f"\n[ORIG REGIME]: {orig_regime} ({regime_names.get(orig_regime, '?')})")
    print(f"[TARGET]:      {target_k} ({regime_names.get(target_k, '?')})")
    
    prompt = context + "\n"
    
    # Baseline
    injection_mode = None
    steering_delta = None
    baseline = generate_continuation(prompt)
    
    # Last token injection
    injection_mode = "last_token"
    steering_delta = delta
    last_token_cont = generate_continuation(prompt)
    
    # Broadcast injection
    injection_mode = "broadcast"
    steering_delta = delta
    broadcast_cont = generate_continuation(prompt)
    
    injection_mode = None
    steering_delta = None
    
    print(f"\n[BASELINE]:")
    print(f"  {baseline.strip()[:250]}")
    print(f"\n[LAST TOKEN -> {regime_names.get(target_k, '?')}]:")
    print(f"  {last_token_cont.strip()[:250]}")
    print(f"\n[BROADCAST -> {regime_names.get(target_k, '?')}]:")
    print(f"  {broadcast_cont.strip()[:250]}")

hook_handle.remove()
print("\n" + "="*80)
print("Hook removed.")

# Save
torch.save({
    'model': model.state_dict(), 
    'dyn': dyn.state_dict(),
    'regime_names': regime_names,
    'regime_to_stage': regime_to_stage,
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
}, f"{checkpoint_save}/rpc_meta_suite.pt")
print(f"\nSaved artifacts to {checkpoint_save}")