
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




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_save = "rpc_sweep_results_abir"
checkpoint_dir = "/home/abir19/scratch/abir19/gsm8k_qwen14b_trajs"
os.makedirs(checkpoint_save, exist_ok=True)

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
        if len(idxs) < 2: continue
        for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_triplets_per_pid), replace=False):
            triplets.append((idxs[t], idxs[t+1], np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])))
    return all_features, triplets


class CEBRA_MoE_Encoder(nn.Module):
    def __init__(self, d_in, d_h, K):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, d_h))
        self.gate = nn.Sequential(
            nn.Linear(d_h, 128), nn.LayerNorm(128), nn.Softplus(), nn.Linear(128, K, bias=False))

    def forward(self, x, temp=1.0):
        h = F.normalize(self.encoder(x), dim=1)
        s = F.gumbel_softmax(self.gate(h), tau=temp, hard=False)
        return h, s


class DynamicsMoE(nn.Module):
    def __init__(self, K, dim):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, dim))
            for _ in range(K)])

    def forward(self, h, s):
        preds = torch.stack([m(h) for m in self.experts], dim=1)
        return torch.sum(s.unsqueeze(-1) * preds, dim=1)


def nce_loss(z, p, n, temp=0.05):
    logits = torch.cat([torch.sum(z*p,1,keepdim=True), torch.sum(z*n,1,keepdim=True)], dim=1) / temp
    return F.cross_entropy(logits, torch.zeros(len(z), dtype=torch.long, device=z.device))


def compute_transition_matrix(states_per_problem, K):
    T = np.zeros((K, K))
    for seq in states_per_problem:
        for a, b in zip(seq[:-1], seq[1:]):
            T[a, b] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    return T / np.where(row_sums == 0, 1, row_sums)


def plot_transition_matrix(T, K, title, save_path):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(T, cmap='YlOrRd', vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{T[i,j]:.2f}", ha='center', va='center', fontsize=9,
                    color='black' if T[i,j] < 0.6 else 'white')
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xlabel('Next State'); ax.set_ylabel('Current State')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_and_eval_k(K_val, features, X_torch, triplets, epochs=50):
    print(f"\n--- Evaluating K={K_val} ---", flush=True)
    d_h = 32
    model = CEBRA_MoE_Encoder(X_torch.shape[1], d_h, K_val).to(device)
    dyn   = DynamicsMoE(K_val, d_h).to(device)
    opt   = optim.AdamW(list(model.parameters()) + list(dyn.parameters()), lr=1e-3)

    for epoch in range(epochs):
        tau        = max(0.2, 1.5 * (0.92 ** epoch))
        curr_w_div = min(30.0, (epoch / 15.0) * 30.0)
        indices    = np.random.permutation(len(triplets))
        for b in range(0, len(triplets), 128):
            b_idx = indices[b:b+128]
            i_t = torch.tensor([triplets[x][0] for x in b_idx], device=device)
            p_t = torch.tensor([triplets[x][1] for x in b_idx], device=device)
            n_t = torch.tensor([triplets[x][2] for x in b_idx], device=device)

            h_i, s_i = model(X_torch[i_t], temp=tau)
            h_p, s_p = model(X_torch[p_t], temp=tau)
            h_n, _   = model(X_torch[n_t], temp=tau)
            h_pred   = dyn(h_i, s_i)

            l_nce  = nce_loss(h_pred, h_p, h_n)
            l_mse  = F.mse_loss(h_pred, h_p)
            l_div  = (s_i.mean(0) * torch.log(s_i.mean(0) + 1e-8)).sum()
            l_pers = torch.abs(s_i - s_p).mean()

            loss = l_nce + 10.0*l_mse + curr_w_div*l_div + 1.0*l_pers
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        _, s_final = model(X_torch, temp=0.01)
        states = s_final.argmax(1).cpu().numpy()

    # per-problem state sequences — sorted by sentence_idx
    p_map = defaultdict(list)
    for i, f in enumerate(features):
        p_map[f["problem_id"]].append((f["sentence_idx"], states[i]))
    seqs = []
    for pid in sorted(p_map.keys()):
        seq = [s for _, s in sorted(p_map[pid])]
        if len(seq) > 1:
            seqs.append(seq)

    persistence = np.mean([1.0 - np.sum(np.array(s[1:]) != np.array(s[:-1])) / (len(s)-1) for s in seqs])
    T_matrix    = compute_transition_matrix(seqs, K_val)

    # print transition matrix
    print(f"\n  Transition matrix (K={K_val}):", flush=True)
    header = "      " + "  ".join(f"→{j}" for j in range(K_val))
    print(f"  {header}", flush=True)
    for i in range(K_val):
        row = "  ".join(f"{T_matrix[i,j]:.2f}" for j in range(K_val))
        print(f"  {i}  [ {row} ]", flush=True)

    return persistence, l_mse.item(), T_matrix


# --- EXECUTION ---

all_features, triplets = load_and_balance_data(f"{checkpoint_dir}/all_sentences_features.pkl")
X_torch = torch.from_numpy(StandardScaler().fit_transform(
    np.array([f['hidden_state'] for f in all_features]))).float().to(device)

k_values = [2, 3, 4, 5, 6, 8]
results  = []

for k in k_values:
    persistence, final_mse, T = train_and_eval_k(k, all_features, X_torch, triplets)
    results.append((k, persistence, final_mse, T))
    print(f"K={k} -> Persistence: {persistence:.2%}, MSE: {final_mse:.6f}", flush=True)
    plot_transition_matrix(T, k, f"Transitions (K={k})", f"{checkpoint_save}/transition_K{k}.png")

# --- SUMMARY PLOT ---
k_list, p_list, m_list = zip(*[(r[0], r[1], r[2]) for r in results])

fig, ax1 = plt.subplots()
ax1.set_xlabel('Number of Regimes (K)')
ax1.set_ylabel('Persistence (Stability)', color='tab:blue')
ax1.plot(k_list, p_list, marker='o', color='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Dynamics MSE (Accuracy)', color='tab:red')
ax2.plot(k_list, m_list, marker='s', color='tab:red')
plt.title("RPC State Sweep: Identifying Optimal Regime Count")
plt.savefig(f"{checkpoint_save}/k_sweep_analysis.png")
plt.close()

# --- GRID OF ALL TRANSITION MATRICES ---
n = len(k_values)
fig, axes = plt.subplots(2, (n+1)//2, figsize=(4*((n+1)//2), 8))
axes = axes.flatten()
for idx, (k, _, _, T) in enumerate(results):
    ax = axes[idx]
    im = ax.imshow(T, cmap='YlOrRd', vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{T[i,j]:.2f}", ha='center', va='center', fontsize=8,
                    color='black' if T[i,j] < 0.6 else 'white')
    ax.set_title(f"K={k}"); ax.set_xlabel('Next State'); ax.set_ylabel('Current State')
for ax in axes[len(results):]: ax.axis('off')
plt.suptitle("Regime Transition Probabilities", fontsize=13)
plt.tight_layout()
plt.savefig(f"{checkpoint_save}/transition_grid.png", dpi=150)
plt.close()

print(f"\nDone. Results saved to '{checkpoint_save}/'", flush=True)