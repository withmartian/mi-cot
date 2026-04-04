
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from contrastive_gen.models import CEBRA_MoE_Encoder, DynamicsMoE
from contrastive_gen.losses import nce_loss
from contrastive_gen.data_utils import load_and_balance_data
# --- SETTINGS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_save = "rpc_sweep_results_base"
checkpoint_dir = "rpc_dataset_base" 
os.makedirs(checkpoint_save, exist_ok=True)



# --- THE SWEEP ENGINE ---
def train_and_eval(K_val, features, X_torch, triplets, epochs=50):
    print(f"\n--- Testing K={K_val} ---", flush=True)
    d_h = 32
    model = CEBRA_MoE_Encoder(X_torch.shape[1], d_h, K_val).to(device)
    dyn = DynamicsMoE(K_val, d_h).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(dyn.parameters()), lr=1e-3)

    for epoch in range(epochs):
        tau = max(0.2, 1.5 * (0.92 ** epoch))
        div_w = min(30.0, (epoch / 15.0) * 30.0)
        for b in range(0, len(triplets), 128):
            b_idx = range(b, min(b+128, len(triplets)))
            i_t, p_t, n_t = [torch.tensor([triplets[x][idx] for x in b_idx], device=device) for idx in range(3)]
            h_i, s_i = model(X_torch[i_t], temp=tau)
            h_p, s_p = model(X_torch[p_t], temp=tau)
            h_n, _ = model(X_torch[n_t], temp=tau)
            h_pred = dyn(h_i, s_i)
            loss = nce_loss(h_pred, h_p, h_n) + 10*F.mse_loss(h_pred, h_p) + div_w*(s_i.mean(0)*torch.log(s_i.mean(0)+1e-8)).sum() + 10*torch.abs(s_i-s_p).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        h_final, s_final = model(X_torch, temp=0.01)
        states = s_final.argmax(1).cpu().numpy()
    
    # Structural Analysis: Geometry (Blur) vs Dynamics (Entropy)
    centroids = torch.stack([h_final[states == k].mean(0) if (states == k).any() else torch.zeros(d_h).to(device) for k in range(K_val)])
    sim_matrix = F.cosine_similarity(centroids.unsqueeze(1), centroids.unsqueeze(0), dim=-1)
    mean_sim = sim_matrix[~torch.eye(K_val, dtype=torch.bool)].mean().item()

    p_map = defaultdict(list)
    for i, p_id in enumerate([f['problem_id'] for f in features]): p_map[p_id].append(states[i])
    t_matrix = np.zeros((K_val, K_val))
    for seq in p_map.values():
        for t in range(len(seq)-1): t_matrix[seq[t], seq[t+1]] += 1
    t_matrix_norm = np.divide(t_matrix, t_matrix.sum(axis=1, keepdims=True), where=t_matrix.sum(axis=1, keepdims=True) != 0)
    
    trans_entropy = np.mean([-np.sum(row[row>0] * np.log2(row[row>0])) for row in t_matrix_norm if row.sum() > 0])
    persistence = np.mean([np.sum(np.array(seq[1:]) == np.array(seq[:-1])) / (len(seq)-1) for seq in p_map.values() if len(seq) > 1])
    
    return persistence, trans_entropy, mean_sim, t_matrix_norm

# --- EXECUTION ---
all_features, triplets = load_and_balance_data(f"{checkpoint_dir}/all_sentences_features.pkl")
X_torch = torch.from_numpy(StandardScaler().fit_transform(np.array([f['hidden_state'] for f in all_features]))).float().to(device)

results = []
for k in [2, 3, 4, 5, 6, 8]:
    p, ent, sim, mat = train_and_eval(k, all_features, X_torch, triplets)
    results.append((k, p, ent, sim))

# --- FINAL PROOF PLOTTING ---
k_list, p_list, ent_list, sim_list = zip(*results)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(k_list, sim_list, 'r-o', label='Centroid Similarity (Blur)')
ax1.set_title("Geometric Blur Proof")
ax1.set_xlabel("K Regimes"); ax1.set_ylabel("Mean Cosine Similarity"); ax1.grid(True)

ax2.plot(k_list, ent_list, 'g-s', label='Transition Entropy (Chaos)')
ax2.set_title("Logical Fragmentation Proof")
ax2.set_xlabel("K Regimes"); ax2.set_ylabel("Entropy (bits)"); ax2.grid(True)

plt.tight_layout(); plt.savefig(f"{checkpoint_save}/_proof.png")
print(f"saved to {checkpoint_save}/undeniable_proof.png")