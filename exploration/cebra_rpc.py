import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
from contrastive_gen.models import CEBRA_MoE_Encoder, DynamicsMoE

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_save = "rpc_sweep_results"
checkpoint_dir = "rpc_dataset"

os.makedirs(checkpoint_save, exist_ok=True)

#--- DATA LOADING & SKEW CORRECTION ---
def load_and_balance_data(path, limit_problems=500, max_triplets_per_pid=20):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}.")
    all_features = pickle.load(open(path, 'rb'))
    all_features = [f for f in all_features if f['problem_id'] < limit_problems]
    p_map = defaultdict(list)
    for i, f in enumerate(all_features): p_map[f['problem_id']].append(i)
    triplets = []
    pids = list(p_map.keys())
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: 
            continue
        num_samples = min(len(idxs)-1, max_triplets_per_pid)
        for t in np.random.choice(len(idxs)-1, num_samples, replace=False):
            triplets.append((idxs[t], idxs[t+1], np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])))
    return all_features, triplets

# def load_and_balance_data(path, limit_problems=500, max_triplets_per_pid=20):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Could not find {path}.")
#     all_features = pickle.load(open(path, 'rb'))
#     all_features = [f for f in all_features if f['problem_id'] < limit_problems]
#     p_map = defaultdict(list)
#     for i, f in enumerate(all_features): p_map[f['problem_id']].append(i)
#     triplets = []
#     for pid, idxs in p_map.items():
#         if len(idxs) < 2: continue
#         for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_triplets_per_pid), replace=False):
#             triplets.append((idxs[t], idxs[t+1], np.random.randint(len(all_features))))
#     return all_features, triplets

    
class CEBRA_MoE_Encoder(nn.Module):
    def __init__(self, d_in, d_h, K):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, d_h)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_h, 128),
            nn.LayerNorm(128),
            nn.Softplus(),
            nn.Linear(128, K, bias=False)
        )

    def forward(self, x, temp=1.0):
        h = F.normalize(self.encoder(x), dim=1)
        logits = self.gate(h)
        s = F.gumbel_softmax(logits, tau=temp, hard=False)
        return h, s, logits


class DynamicsMoE(nn.Module):
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
    pos = torch.sum(z * p, dim=1, keepdim=True)
    neg = torch.sum(z * n, dim=1, keepdim=True)
    logits = torch.cat([pos, neg], dim=1) / temp
    return F.cross_entropy(logits, torch.zeros(len(z), dtype=torch.long, device=z.device))


def train_and_eval_k(K_val, features, X_torch, triplets, epochs=50):
    print(f"\n--- Evaluating K={K_val} ---", flush=True)
    d_h = 32
    model = CEBRA_MoE_Encoder(X_torch.shape[1], d_h, K_val).to(device)
    dyn = DynamicsMoE(K_val, d_h).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(dyn.parameters()), lr=1e-3)

    for epoch in range(epochs):
        tau = max(0.2, 1.5 * (0.92 ** epoch))
        curr_w_div = min(30.0, (epoch / 15.0) * 30.0)
        
        indices = np.random.permutation(len(triplets))
        for b in range(0, len(triplets), 128):
            b_idx = indices[b:b+128]
            i_t = torch.tensor([triplets[x][0] for x in b_idx], device=device)
            p_t = torch.tensor([triplets[x][1] for x in b_idx], device=device)
            n_t = torch.tensor([triplets[x][2] for x in b_idx], device=device)

            h_i, s_i, _ = model(X_torch[i_t], temp=tau)
            h_p, s_p, _ = model(X_torch[p_t], temp=tau)
            h_n, _, _ = model(X_torch[n_t], temp=tau)
            
            h_pred = dyn(h_i, s_i)
            
            l_nce = nce_loss(h_pred, h_p, h_n)
            l_mse = F.mse_loss(h_pred, h_p)
            l_div = (s_i.mean(0) * torch.log(s_i.mean(0) + 1e-8)).sum()
            l_pers = torch.abs(s_i - s_p).mean()

            loss = l_nce + (10.0 * l_mse) + (curr_w_div * l_div) + (10.0 * l_pers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _, s_final, _ = model(X_torch, temp=0.01)
        states = s_final.argmax(1).cpu().numpy()
    
    pids = [f['problem_id'] for f in features]
    p_scores = []
    p_map = defaultdict(list)
    for i, p in enumerate(pids):
        p_map[p].append(states[i])
    for p, seq in p_map.items():
        if len(seq) > 1:
            p_scores.append(1.0 - (np.sum(np.array(seq[1:]) != np.array(seq[:-1])) / (len(seq)-1)))
    
    return np.mean(p_scores), l_mse.item()


# --- EXECUTION ---

all_features, triplets = load_and_balance_data(f"{checkpoint_dir}/all_sentences_features.pkl")
X_raw = np.array([f['hidden_state'] for f in all_features])
X_torch = torch.from_numpy(StandardScaler().fit_transform(X_raw)).float().to(device)

k_values = [2, 3, 4, 5, 6, 8]
results = []

for k in k_values:
    persistence, final_mse = train_and_eval_k(k, all_features, X_torch, triplets)
    results.append((k, persistence, final_mse))
    print(f"K={k} Result -> Persistence: {persistence:.2%}, Dynamics MSE: {final_mse:.6f}", flush=True)

k_list, p_list, m_list = zip(*results)
fig, ax1 = plt.subplots()

ax1.set_xlabel('Number of Regimes (K)')
ax1.set_ylabel('Persistence (Stability)', color='tab:blue')
ax1.plot(k_list, p_list, marker='o', color='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Dynamics MSE (Accuracy)', color='tab:red')
ax2.plot(k_list, m_list, marker='s', color='tab:red')

plt.title("RPC State Sweep: Identifying Optimal Regime Count")
plt.savefig(f"{checkpoint_save}/k_sweep_analysis.png")
print(f"\n Sweep complete. Saved at '{checkpoint_save}/k_sweep_analysis.png'", flush=True)