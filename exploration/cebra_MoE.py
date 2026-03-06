import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

PCA_DIM = 40
K_SWEEP = range(1, 13)

STAGES = [
    "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
]

def load_data(path, limit_problems=500, max_triplets=20):
    all_features = pickle.load(open(path, 'rb'))
    all_features = [f for f in all_features if f['problem_id'] < limit_problems]
    p_map = defaultdict(list)
    for i, f in enumerate(all_features): p_map[f['problem_id']].append(i)
    triplets = []
    pids = list(p_map.keys())
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: continue
        for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_triplets), replace=False):
            neg_pid = np.random.choice([p for p in pids if p != pid])
            triplets.append((idxs[t], idxs[t+1], np.random.choice(p_map[neg_pid])))
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


def hard_transition_matrix(state_seqs, K):
    T = np.zeros((K, K))
    for seq in state_seqs:
        for a, b in zip(seq[:-1], seq[1:]): T[a, b] += 1
    row_sums = T.sum(1, keepdims=True)
    return T / np.where(row_sums == 0, 1, row_sums)

def print_transition_matrix(T, K):
    header = "      " + " ".join(f"  →{j}" for j in range(K))
    print(f"  {header}", flush=True)
    for i in range(K):
        row = " ".join(f"{T[i,j]:5.2f}" for j in range(K))
        print(f"  {i} [ {row} ]  self={T[i,i]:.2f}", flush=True)
    print(f"  mean self-trans: {np.diag(T).mean():.3f}", flush=True)

def sss(persist, mean_self_trans, T, K, K_eff):
    if K == 1: return 0.0
    row_ents = -np.sum(T * np.log(T + 1e-12), axis=1) / np.log(K)
    return (K_eff / K) * mean_self_trans * np.log(persist + 1) * (1 - np.mean(row_ents))

def linear_ar_r2(pca_seqs):
    X_in  = np.vstack([s[:-1] for s in pca_seqs])
    X_out = np.vstack([s[1:]  for s in pca_seqs])
    X_aug = np.hstack([X_in, np.ones((len(X_in), 1))])
    coef, *_ = np.linalg.lstsq(X_aug, X_out, rcond=None)
    pred = X_aug @ coef
    ss_res = np.sum((X_out - pred)**2)
    ss_tot = np.sum((X_out - X_out.mean(0))**2)
    return 1 - ss_res / ss_tot

def regime_r2_on_pca(state_seqs, pca_seqs):
    K = max(s.max() for s in state_seqs) + 1
    X_in_k  = [[] for _ in range(K)]
    X_out_k = [[] for _ in range(K)]
    for s_seq, p_seq in zip(state_seqs, pca_seqs):
        for t in range(len(s_seq) - 1):
            X_in_k[s_seq[t]].append(p_seq[t])
            X_out_k[s_seq[t]].append(p_seq[t+1])
    coefs = []
    for k in range(K):
        if len(X_in_k[k]) < PCA_DIM + 2: coefs.append(None); continue
        Xi = np.array(X_in_k[k]); Xo = np.array(X_out_k[k])
        X_aug = np.hstack([Xi, np.ones((len(Xi), 1))])
        c, *_ = np.linalg.lstsq(X_aug, Xo, rcond=None)
        coefs.append(c)
    all_pred, all_true = [], []
    for s_seq, p_seq in zip(state_seqs, pca_seqs):
        for t in range(len(s_seq) - 1):
            k = s_seq[t]
            if coefs[k] is None: continue
            all_pred.append(np.append(p_seq[t], 1.0) @ coefs[k])
            all_true.append(p_seq[t+1])
    all_pred = np.array(all_pred); all_true = np.array(all_true)
    ss_res = np.sum((all_true - all_pred)**2)
    ss_tot = np.sum((all_true - all_true.mean(0))**2)
    return 1 - ss_res / ss_tot


def train_and_eval(K, X_torch, all_features, pca_seqs, triplets, epochs=75):
    d_h = 32
    model = CEBRA_MoE_Encoder(X_torch.shape[1], d_h, K).to(device)
    dyn   = DynamicsMoE(K, d_h).to(device)
    opt   = optim.AdamW(list(model.parameters()) + list(dyn.parameters()), lr=1e-3)

    for epoch in range(epochs):
        tau       = max(0.2, 1.5 * (0.92 ** epoch))
        w_div     = min(30.0, (epoch / 15.0) * 30.0)
        indices   = np.random.permutation(len(triplets))
        for b in range(0, len(triplets), 128):
            idx = indices[b:b+128]
            i_t = torch.tensor([triplets[x][0] for x in idx], device=device)
            p_t = torch.tensor([triplets[x][1] for x in idx], device=device)
            n_t = torch.tensor([triplets[x][2] for x in idx], device=device)
            h_i, s_i = model(X_torch[i_t], temp=tau)
            h_p, s_p = model(X_torch[p_t], temp=tau)
            h_n, _   = model(X_torch[n_t], temp=tau)
            h_pred   = dyn(h_i, s_i)
            loss = (nce_loss(h_pred, h_p, h_n)
                    + 10.0 * F.mse_loss(h_pred, h_p)
                    + w_div * (s_i.mean(0) * torch.log(s_i.mean(0) + 1e-8)).sum()
                    + 1.0  * torch.abs(s_i - s_p).mean())
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        _, s_final = model(X_torch, temp=0.01)
        states = s_final.argmax(1).cpu().numpy()

    # build per-problem state sequences
    p_map = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map[f['problem_id']].append((f['sentence_idx'], states[i]))
    state_seqs = []
    pid_order  = []
    for pid in sorted(p_map.keys()):
        seq = [s for _, s in sorted(p_map[pid])]
        if len(seq) > 1:
            state_seqs.append(np.array(seq))
            pid_order.append(pid)

    # align pca_seqs to same pid order
    pid_to_pca = {pid: pca_seqs[i] for i, pid in enumerate(sorted(p_map.keys())) if len(p_map[pid]) > 1}
    aligned_pca = [pid_to_pca[pid] for pid in pid_order if pid in pid_to_pca]
    # trim state_seqs to match aligned_pca length
    state_seqs  = state_seqs[:len(aligned_pca)]

    all_states = np.concatenate(state_seqs)
    counts     = np.bincount(all_states, minlength=K)
    K_eff      = int(np.sum(counts / len(all_states) > 0.01))
    persist    = np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs])

    confusion = np.zeros((K, len(STAGES)))
    for s_seq, f in zip(state_seqs, all_features):
        pass  # labels need separate alignment — done below
    # rebuild label alignment
    p_map_l = defaultdict(list)
    for f in all_features: p_map_l[f['problem_id']].append(f.get('stage', 'NEUTRAL'))
    confusion = np.zeros((K, len(STAGES)))
    for pid, s_seq in zip(pid_order, state_seqs):
        if pid not in pid_to_pca: continue
        l_seq = p_map_l[pid]
        for s, l in zip(s_seq, l_seq):
            if l in STAGES: confusion[s, STAGES.index(l)] += 1
    conf_norm = confusion / (confusion.sum(1, keepdims=True) + 1e-9)
    spec      = np.mean(np.max(conf_norm, axis=1))

    T_hard  = hard_transition_matrix(state_seqs, K)
    mean_st = np.diag(T_hard).mean()
    score   = sss(persist, mean_st, T_hard, K, K_eff)
    r2      = regime_r2_on_pca(state_seqs, aligned_pca)

    return persist, mean_st, spec, conf_norm, T_hard, score, K_eff, r2


if __name__ == "__main__":
    path = "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_reasoning/layer_28/all_sentences_features.pkl"

    print("Loading data...", flush=True)
    all_features, triplets = load_data(path)

    X_raw    = np.array([f['hidden_state_last'] for f in all_features])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca      = PCA(n_components=PCA_DIM, random_state=42)
    X_pca    = pca.fit_transform(X_scaled)
    X_torch  = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    # build pca_seqs per problem (sorted by sentence_idx)
    p_map_pca = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map_pca[f['problem_id']].append((f['sentence_idx'], X_pca[i]))
    pca_seqs = [np.array([x for _, x in sorted(p_map_pca[pid])])
                for pid in sorted(p_map_pca) if len(p_map_pca[pid]) > 1]

    ar_r2 = linear_ar_r2(pca_seqs)
    print(f"\n  Linear AR baseline R² (PCA space): {ar_r2:.4f}", flush=True)

    print(f"\n{'K':<4} | {'K_eff':<6} | {'Persist':<10} | {'Self-Trans':<10} | {'Spec':<10} | {'SSS':<8} | {'R²':<8} | {'ΔR²':<8}", flush=True)
    print("-" * 80, flush=True)

    sss_scores = {}
    for k in K_SWEEP:
        p, st, s, C_mat, T_hard, score, k_eff, r2 = train_and_eval(k, X_torch, all_features, pca_seqs, triplets)
        sss_scores[k] = score
        delta_r2 = r2 - ar_r2
        print(f"{k:<4} | {k_eff:<6} | {p:<10.2f} | {st:<10.3f} | {s:<10.4f} | {score:<8.4f} | {r2:<8.4f} | {delta_r2:+.4f}", flush=True)

        print(f"\n  Transition matrix (K={k}):", flush=True)
        print_transition_matrix(T_hard, k)

        if k >= 4:
            print(f"\n  Dominant stages (K={k}):", flush=True)
            for i in range(k):
                idx = np.argmax(C_mat[i])
                print(f"    Mode {i}: {STAGES[idx]} ({C_mat[i,idx]:.1%})", flush=True)
        print()

    best_k = max(sss_scores, key=sss_scores.get)
    print(f"\n  Best K by SSS: K={best_k}  (SSS={sss_scores[best_k]:.4f})", flush=True)
    print(f"  SSS profile: " + "  ".join([f"K{k}={v:.3f}" for k, v in sss_scores.items()]), flush=True)