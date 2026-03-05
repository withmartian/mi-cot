# import os
# import numpy as np
# import pickle
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from collections import defaultdict
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

# K_SWEEP = range(1, 13)
# D = 40
# PCA_DIM = 40
# CEBRA_EPOCHS = 100
# BATCH_SIZE = 1024
# KAPPA = 10.0
# N_ITERS_SLDS = 50

# STAGES = [
#     "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
#     "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
#     "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
# ]

# def load_and_prepare_cebra(path, mode='temporal', limit_problems=500, max_triplets=25):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Could not find {path}.")
#     print(f"Loading data for CEBRA-{mode}...", flush=True)
#     with open(path, 'rb') as f:
#         all_features = pickle.load(f)
#     all_features = [f for f in all_features if f['problem_id'] < limit_problems]
#     p_map = defaultdict(list)
#     for i, f in enumerate(all_features):
#         p_map[f['problem_id']].append(i)
#     triplets = []
#     pids = list(p_map.keys())
#     for pid in pids:
#         idxs = p_map[pid]
#         if len(idxs) < 2: continue
#         for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_triplets), replace=False):
#             anchor, positive = idxs[t], idxs[t+1]
#             if mode == 'temporal':
#                 negative = np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])
#             else:
#                 a_stage = all_features[anchor].get('stage', 'NEUTRAL')
#                 neg_pool = [i for i in idxs if all_features[i].get('stage', 'NEUTRAL') != a_stage]
#                 negative = np.random.choice(neg_pool) if neg_pool else \
#                            np.random.choice(p_map[np.random.choice(pids)])
#             triplets.append((anchor, positive, negative))
#     return all_features, triplets


# class CEBRANet(nn.Module):
#     def __init__(self, d_in, d_out):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_in, 512), nn.GELU(),
#             nn.Linear(512, 256), nn.GELU(),
#             nn.Linear(256, d_out))
#     def forward(self, x):
#         return F.normalize(self.net(x), p=2, dim=1)

# def train_cebra_projection(all_features, triplets, d_out=D):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     X_raw = np.array([f['hidden_state_last'] for f in all_features])
#     scaler = StandardScaler()
#     X_scaled_np = scaler.fit_transform(X_raw)

#     # Fit PCA once, carry alongside CEBRA embeddings
#     pca = PCA(n_components=PCA_DIM, random_state=42)
#     X_pca = pca.fit_transform(X_scaled_np)

#     X_scaled = torch.tensor(X_scaled_np, dtype=torch.float32).to(device)
#     model = CEBRANet(X_raw.shape[1], d_out).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     triplets_arr = np.array(triplets)
#     for epoch in range(CEBRA_EPOCHS):
#         indices = np.random.permutation(len(triplets_arr))
#         for i in range(0, len(triplets_arr), BATCH_SIZE):
#             batch = triplets_arr[indices[i:i+BATCH_SIZE]]
#             za = model(X_scaled[batch[:,0]])
#             zp = model(X_scaled[batch[:,1]])
#             zn = model(X_scaled[batch[:,2]])
#             sim_p = torch.sum(za*zp, dim=1) / 0.1
#             sim_n = torch.sum(za*zn, dim=1) / 0.1
#             loss = -torch.log(torch.exp(sim_p) / (torch.exp(sim_p) + torch.exp(sim_n))).mean()
#             optimizer.zero_grad(); loss.backward(); optimizer.step()

#     model.eval()
#     with torch.no_grad():
#         Z = model(X_scaled).cpu().numpy()

#     p_map_z, p_map_p, p_map_l = defaultdict(list), defaultdict(list), defaultdict(list)
#     for i, f in enumerate(all_features):
#         pid = f['problem_id']
#         p_map_z[pid].append(Z[i])
#         p_map_p[pid].append(X_pca[i])
#         p_map_l[pid].append(f.get('stage', 'NEUTRAL'))

#     pids_sorted = sorted(p for p in p_map_z if len(p_map_z[p]) >= 3)
#     cebra_seqs = [np.array(p_map_z[p]) for p in pids_sorted]
#     pca_seqs   = [np.array(p_map_p[p]) for p in pids_sorted]
#     labels     = [p_map_l[p]           for p in pids_sorted]
#     return cebra_seqs, pca_seqs, labels


# def linear_ar_r2(pca_seqs):
#     """Global linear AR baseline: x_{t+1} = M x_t + b, fitted across all sequences."""
#     X_in, X_out = [], []
#     for seq in pca_seqs:
#         X_in.append(seq[:-1]); X_out.append(seq[1:])
#     X_in  = np.vstack(X_in)
#     X_out = np.vstack(X_out)
#     X_aug = np.hstack([X_in, np.ones((len(X_in), 1))])
#     coef, *_ = np.linalg.lstsq(X_aug, X_out, rcond=None)
#     pred = X_aug @ coef
#     ss_res = np.sum((X_out - pred)**2)
#     ss_tot = np.sum((X_out - X_out.mean(0))**2)
#     return 1 - ss_res / ss_tot


# def regime_r2_on_pca(state_seqs, pca_seqs):
#     """Fit one linear model per regime on PCA space, evaluate R² on same data."""
#     K = max(s.max() for s in state_seqs) + 1
#     # collect per-regime (x_t, x_{t+1}) pairs
#     X_in_k  = [[] for _ in range(K)]
#     X_out_k = [[] for _ in range(K)]
#     for s_seq, p_seq in zip(state_seqs, pca_seqs):
#         for t in range(len(s_seq) - 1):
#             k = s_seq[t]
#             X_in_k[k].append(p_seq[t])
#             X_out_k[k].append(p_seq[t+1])

#     # fit per-regime linear models
#     coefs = []
#     for k in range(K):
#         if len(X_in_k[k]) < PCA_DIM + 2:
#             coefs.append(None); continue
#         Xi = np.array(X_in_k[k])
#         Xo = np.array(X_out_k[k])
#         X_aug = np.hstack([Xi, np.ones((len(Xi), 1))])
#         c, *_ = np.linalg.lstsq(X_aug, Xo, rcond=None)
#         coefs.append(c)

#     # evaluate globally
#     all_pred, all_true = [], []
#     for s_seq, p_seq in zip(state_seqs, pca_seqs):
#         for t in range(len(s_seq) - 1):
#             k = s_seq[t]
#             if coefs[k] is None: continue
#             x_aug = np.append(p_seq[t], 1.0)
#             all_pred.append(x_aug @ coefs[k])
#             all_true.append(p_seq[t+1])

#     all_pred = np.array(all_pred)
#     all_true = np.array(all_true)
#     ss_res = np.sum((all_true - all_pred)**2)
#     ss_tot = np.sum((all_true - all_true.mean(0))**2)
#     return 1 - ss_res / ss_tot


# def init_params(sequences, K, D):
#     X_in, X_out = [], []
#     for seq in sequences:
#         for t in range(len(seq)-1):
#             X_in.append(seq[t]); X_out.append(seq[t+1] - seq[t])
#     X_in, X_out = np.array(X_in), np.array(X_out)
#     labels = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X_out)
#     dM, db, dCov = np.zeros((K,D,D)), np.zeros((K,D)), np.array([np.eye(D)]*K)
#     for k in range(K):
#         mask = labels == k
#         if mask.sum() < D+2: dM[k] = 0.1*np.eye(D); continue
#         W, *_ = np.linalg.lstsq(np.hstack([X_in[mask], np.ones((mask.sum(),1))]), X_out[mask], rcond=None)
#         dM[k], db[k] = W[:D].T, W[D]
#         res = X_out[mask] - (X_in[mask] @ dM[k].T + db[k])
#         dCov[k] = np.cov(res.T) + 1e-3*np.eye(D)
#     return np.ones(K)/K, np.eye(K)*0.7+0.3/K, dM, db, dCov

# def get_log_emissions(seq, K, dM, db, dCov):
#     T, D_dim = seq.shape
#     log_emit = np.zeros((T, K))
#     for k in range(K):
#         _, logdet = np.linalg.slogdet(dCov[k])
#         inv_cov = np.linalg.inv(dCov[k])
#         means = np.vstack([db[k], seq[:-1] @ dM[k].T + db[k]])
#         diffs = seq - means
#         log_emit[:, k] = -0.5*(D_dim*np.log(2*np.pi) + logdet + np.sum((diffs @ inv_cov)*diffs, axis=1))
#     return log_emit

# def forward_backward(seq, pi, A, dM, db, dCov, K):
#     T = len(seq)
#     log_emit = get_log_emissions(seq, K, dM, db, dCov)
#     log_A, log_pi = np.log(A+1e-12), np.log(pi+1e-12)
#     la = np.zeros((T,K)); la[0] = log_pi + log_emit[0]
#     for t in range(1,T):
#         la[t] = log_emit[t] + np.logaddexp.reduce(la[t-1][:,None]+log_A, axis=0)
#     lb = np.zeros((T,K))
#     for t in range(T-2,-1,-1):
#         lb[t] = np.logaddexp.reduce(log_A + log_emit[t+1] + lb[t+1], axis=1)
#     lg = la+lb; lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
#     lxi = np.zeros((T-1,K,K))
#     for t in range(T-1):
#         lxi[t] = la[t][:,None] + log_A + log_emit[t+1] + lb[t+1]
#         lxi[t] -= np.logaddexp.reduce(lxi[t].ravel())
#     return np.exp(lg), np.exp(lxi)

# def m_step(sequences, gammas, xis, K, D_dim):
#     xi_sum = sum(xi.sum(0) for xi in xis) + np.eye(K)*KAPPA + 1e-8
#     A_new  = xi_sum / xi_sum.sum(1, keepdims=True)
#     pi_new = np.maximum(np.mean([g[0] for g in gammas], axis=0), 1e-8)
#     pi_new /= pi_new.sum()
#     dyn_M, dyn_b, dyn_cov = np.zeros((K,D_dim,D_dim)), np.zeros((K,D_dim)), np.zeros((K,D_dim,D_dim))
#     for k in range(K):
#         W_sum, WY_sum = np.zeros((D_dim+1,D_dim+1)), np.zeros((D_dim+1,D_dim))
#         for seq, gamma in zip(sequences, gammas):
#             X_aug = np.hstack([np.vstack([np.zeros(D_dim), seq[:-1]]), np.ones((len(seq),1))])
#             w = gamma[:,k]
#             W_sum  += (X_aug*w[:,None]).T @ X_aug
#             WY_sum += (X_aug*w[:,None]).T @ seq
#         coef = np.linalg.solve(W_sum + 1e-4*np.eye(D_dim+1), WY_sum)
#         dyn_M[k], dyn_b[k] = coef[:D_dim].T, coef[D_dim]
#         num, den = np.zeros((D_dim,D_dim)), 1e-9
#         for seq, gamma in zip(sequences, gammas):
#             err = seq - (np.vstack([np.zeros(D_dim), seq[:-1]]) @ dyn_M[k].T + dyn_b[k])
#             num += (err*gamma[:,k][:,None]).T @ err; den += gamma[:,k].sum()
#         dyn_cov[k] = num/den + 1e-4*np.eye(D_dim)
#     return pi_new, A_new, dyn_M, dyn_b, dyn_cov

# def hard_transition_matrix(state_seqs, K):
#     T = np.zeros((K,K))
#     for seq in state_seqs:
#         for a, b in zip(seq[:-1], seq[1:]): T[a,b] += 1
#     row_sums = T.sum(1, keepdims=True)
#     return T / np.where(row_sums==0, 1, row_sums)

# def print_transition_matrix(T, K):
#     header = "      " + " ".join(f"  →{j}" for j in range(K))
#     print(f"  {header}", flush=True)
#     for i in range(K):
#         row = " ".join(f"{T[i,j]:5.2f}" for j in range(K))
#         print(f"  {i} [ {row} ]  self={T[i,i]:.2f}", flush=True)
#     print(f"  mean self-trans: {np.diag(T).mean():.3f}", flush=True)

# def sss(persist, mean_self_trans, T, K, K_eff):
#     if K == 1: return 0.0
#     row_ents = -np.sum(T * np.log(T + 1e-12), axis=1) / np.log(K)
#     return (K_eff / K) * mean_self_trans * np.log(persist + 1) * (1 - np.mean(row_ents))

# def fit_and_evaluate(cebra_seqs, pca_seqs, labels, K):
#     pi, A, dM, db, dCov = init_params(cebra_seqs, K, D)
#     for _ in range(N_ITERS_SLDS):
#         gammas, xis = [], []
#         for seq in cebra_seqs:
#             g, x = forward_backward(seq, pi, A, dM, db, dCov, K)
#             gammas.append(g); xis.append(x)
#         pi, A, dM, db, dCov = m_step(cebra_seqs, gammas, xis, K, D)

#     state_seqs = [np.argmax(g, axis=1) for g in gammas]

#     all_states = np.concatenate(state_seqs)
#     counts = np.bincount(all_states, minlength=K)
#     K_eff = int(np.sum(counts / len(all_states) > 0.01))

#     persist = np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs])

#     confusion = np.zeros((K, len(STAGES)))
#     for s_seq, l_seq in zip(state_seqs, labels):
#         for s, l in zip(s_seq, l_seq):
#             if l in STAGES: confusion[s, STAGES.index(l)] += 1
#     conf_norm = confusion / (confusion.sum(1, keepdims=True) + 1e-9)
#     spec = np.mean(np.max(conf_norm, axis=1))

#     T_hard  = hard_transition_matrix(state_seqs, K)
#     mean_st = np.diag(T_hard).mean()
#     score   = sss(persist, mean_st, T_hard, K, K_eff)
#     r2      = regime_r2_on_pca(state_seqs, pca_seqs)

#     return persist, mean_st, spec, conf_norm, T_hard, score, K_eff, r2


# if __name__ == "__main__":
#     path = "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_reasoning/layer_28/all_sentences_features.pkl"

#     for mode in ['temporal']:
#         print(f"\n{'='*50}\nRUNNING CEBRA-{mode.upper()}\n{'='*50}", flush=True)
#         all_f, triplets = load_and_prepare_cebra(path, mode=mode)
#         cebra_seqs, pca_seqs, labels = train_cebra_projection(all_f, triplets)

#         ar_r2 = linear_ar_r2(pca_seqs)
#         print(f"\n  Linear AR baseline R² (PCA space): {ar_r2:.4f}", flush=True)

#         print(f"\n{'K':<4} | {'K_eff':<6} | {'Persist':<10} | {'Self-Trans':<10} | {'Spec':<10} | {'SSS':<8} | {'R²':<8} | {'ΔR²':<8}", flush=True)
#         print("-" * 80, flush=True)

#         sss_scores = {}
#         for k in K_SWEEP:
#             p, st, s, C_mat, T_hard, score, k_eff, r2 = fit_and_evaluate(cebra_seqs, pca_seqs, labels, k)
#             sss_scores[k] = score
#             delta_r2 = r2 - ar_r2
#             print(f"{k:<4} | {k_eff:<6} | {p:<10.2f} | {st:<10.3f} | {s:<10.4f} | {score:<8.4f} | {r2:<8.4f} | {delta_r2:+.4f}", flush=True)

#             print(f"\n  Transition matrix (K={k}):", flush=True)
#             print_transition_matrix(T_hard, k)

#             if k >= 4:
#                 print(f"\n  Dominant stages (K={k}):", flush=True)
#                 for i in range(k):
#                     idx = np.argmax(C_mat[i])
#                     print(f"    Mode {i}: {STAGES[idx]} ({C_mat[i,idx]:.1%})", flush=True)
#             print()

#         best_k = max(sss_scores, key=sss_scores.get)
#         print(f"\n  Best K by SSS: K={best_k}  (SSS={sss_scores[best_k]:.4f})", flush=True)
#         print(f"  SSS profile: " + "  ".join([f"K{k}={v:.3f}" for k, v in sss_scores.items()]), flush=True)



import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

K_SWEEP = range(1, 13)
D = 40
PCA_DIM = 40
CEBRA_EPOCHS = 100
BATCH_SIZE = 1024
KAPPA = 1.0
N_ITERS_SLDS = 50

STAGES = [
    "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
]

def load_and_prepare_cebra(path, mode='temporal', limit_problems=500, max_triplets=25):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}.")
    print(f"Loading data for CEBRA-{mode}...", flush=True)
    with open(path, 'rb') as f:
        all_features = pickle.load(f)
    all_features = [f for f in all_features if f['problem_id'] < limit_problems]
    p_map = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map[f['problem_id']].append(i)
    triplets = []
    pids = list(p_map.keys())
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: continue
        for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_triplets), replace=False):
            anchor, positive = idxs[t], idxs[t+1]
            if mode == 'temporal':
                negative = np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])
            else:
                a_stage = all_features[anchor].get('stage', 'NEUTRAL')
                neg_pool = [i for i in idxs if all_features[i].get('stage', 'NEUTRAL') != a_stage]
                negative = np.random.choice(neg_pool) if neg_pool else \
                           np.random.choice(p_map[np.random.choice(pids)])
            triplets.append((anchor, positive, negative))
    return all_features, triplets


class CEBRANet(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, d_out))
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

def train_cebra_projection(all_features, triplets, d_out=D):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_raw = np.array([f['hidden_state_last'] for f in all_features])
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X_raw)
    pca = PCA(n_components=PCA_DIM, random_state=42)
    X_pca = pca.fit_transform(X_scaled_np)
    X_scaled = torch.tensor(X_scaled_np, dtype=torch.float32).to(device)
    model = CEBRANet(X_raw.shape[1], d_out).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    triplets_arr = np.array(triplets)
    for epoch in range(CEBRA_EPOCHS):
        indices = np.random.permutation(len(triplets_arr))
        for i in range(0, len(triplets_arr), BATCH_SIZE):
            batch = triplets_arr[indices[i:i+BATCH_SIZE]]
            za = model(X_scaled[batch[:,0]])
            zp = model(X_scaled[batch[:,1]])
            zn = model(X_scaled[batch[:,2]])
            sim_p = torch.sum(za*zp, dim=1) / 0.1
            sim_n = torch.sum(za*zn, dim=1) / 0.1
            loss = -torch.log(torch.exp(sim_p) / (torch.exp(sim_p) + torch.exp(sim_n))).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        Z = model(X_scaled).cpu().numpy()
    p_map_z, p_map_p, p_map_l = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, f in enumerate(all_features):
        pid = f['problem_id']
        p_map_z[pid].append(Z[i])
        p_map_p[pid].append(X_pca[i])
        p_map_l[pid].append(f.get('stage', 'NEUTRAL'))
    pids_sorted = sorted(p for p in p_map_z if len(p_map_z[p]) >= 3)
    cebra_seqs = [np.array(p_map_z[p]) for p in pids_sorted]
    pca_seqs   = [np.array(p_map_p[p]) for p in pids_sorted]
    labels     = [p_map_l[p]           for p in pids_sorted]
    return cebra_seqs, pca_seqs, labels


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
        c, *_ = np.linalg.lstsq(np.hstack([Xi, np.ones((len(Xi),1))]), Xo, rcond=None)
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


def init_params(sequences, K, D):
    X_in, X_out = [], []
    for seq in sequences:
        for t in range(len(seq)-1):
            X_in.append(seq[t]); X_out.append(seq[t+1] - seq[t])
    X_in, X_out = np.array(X_in), np.array(X_out)
    labels = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X_out)
    dM, db, dCov = np.zeros((K,D,D)), np.zeros((K,D)), np.array([np.eye(D)]*K)
    for k in range(K):
        mask = labels == k
        if mask.sum() < D+2: dM[k] = 0.1*np.eye(D); continue
        W, *_ = np.linalg.lstsq(np.hstack([X_in[mask], np.ones((mask.sum(),1))]), X_out[mask], rcond=None)
        dM[k], db[k] = W[:D].T, W[D]
        res = X_out[mask] - (X_in[mask] @ dM[k].T + db[k])
        dCov[k] = np.cov(res.T) + 1e-3*np.eye(D)
    return np.ones(K)/K, np.eye(K)*0.7+0.3/K, dM, db, dCov

def get_log_emissions(seq, K, dM, db, dCov):
    T, D_dim = seq.shape
    log_emit = np.zeros((T, K))
    for k in range(K):
        _, logdet = np.linalg.slogdet(dCov[k])
        inv_cov = np.linalg.inv(dCov[k])
        means = np.vstack([db[k], seq[:-1] @ dM[k].T + db[k]])
        diffs = seq - means
        log_emit[:, k] = -0.5*(D_dim*np.log(2*np.pi) + logdet + np.sum((diffs @ inv_cov)*diffs, axis=1))
    return log_emit

def forward_backward(seq, pi, A, dM, db, dCov, K):
    T = len(seq)
    log_emit = get_log_emissions(seq, K, dM, db, dCov)
    log_A, log_pi = np.log(A+1e-12), np.log(pi+1e-12)
    la = np.zeros((T,K)); la[0] = log_pi + log_emit[0]
    for t in range(1,T):
        la[t] = log_emit[t] + np.logaddexp.reduce(la[t-1][:,None]+log_A, axis=0)
    lb = np.zeros((T,K))
    for t in range(T-2,-1,-1):
        lb[t] = np.logaddexp.reduce(log_A + log_emit[t+1] + lb[t+1], axis=1)
    lg = la+lb; lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
    lxi = np.zeros((T-1,K,K))
    for t in range(T-1):
        lxi[t] = la[t][:,None] + log_A + log_emit[t+1] + lb[t+1]
        lxi[t] -= np.logaddexp.reduce(lxi[t].ravel())
    seq_ll = np.logaddexp.reduce(la[-1])  # log p(x_{1:T})
    return np.exp(lg), np.exp(lxi), seq_ll

def m_step(sequences, gammas, xis, K, D_dim):
    xi_sum = sum(xi.sum(0) for xi in xis) + np.eye(K)*KAPPA + 1e-8
    A_new  = xi_sum / xi_sum.sum(1, keepdims=True)
    pi_new = np.maximum(np.mean([g[0] for g in gammas], axis=0), 1e-8)
    pi_new /= pi_new.sum()
    dyn_M, dyn_b, dyn_cov = np.zeros((K,D_dim,D_dim)), np.zeros((K,D_dim)), np.zeros((K,D_dim,D_dim))
    for k in range(K):
        W_sum, WY_sum = np.zeros((D_dim+1,D_dim+1)), np.zeros((D_dim+1,D_dim))
        for seq, gamma in zip(sequences, gammas):
            X_aug = np.hstack([np.vstack([np.zeros(D_dim), seq[:-1]]), np.ones((len(seq),1))])
            w = gamma[:,k]
            W_sum  += (X_aug*w[:,None]).T @ X_aug
            WY_sum += (X_aug*w[:,None]).T @ seq
        coef = np.linalg.solve(W_sum + 1e-4*np.eye(D_dim+1), WY_sum)
        dyn_M[k], dyn_b[k] = coef[:D_dim].T, coef[D_dim]
        num, den = np.zeros((D_dim,D_dim)), 1e-9
        for seq, gamma in zip(sequences, gammas):
            err = seq - (np.vstack([np.zeros(D_dim), seq[:-1]]) @ dyn_M[k].T + dyn_b[k])
            num += (err*gamma[:,k][:,None]).T @ err; den += gamma[:,k].sum()
        dyn_cov[k] = num/den + 1e-4*np.eye(D_dim)
    return pi_new, A_new, dyn_M, dyn_b, dyn_cov

def hard_transition_matrix(state_seqs, K):
    T = np.zeros((K,K))
    for seq in state_seqs:
        for a, b in zip(seq[:-1], seq[1:]): T[a,b] += 1
    row_sums = T.sum(1, keepdims=True)
    return T / np.where(row_sums==0, 1, row_sums)

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

def compute_bic(total_ll, K, D, N):
    # transition matrix: K*(K-1) free params
    # per-regime: D*D dynamics + D bias + D*(D+1)/2 covariance (symmetric)
    n_params = K*(K-1) + K*(D*D + D + D*(D+1)//2)
    return -2 * total_ll + n_params * np.log(N)

def fit_and_evaluate(cebra_seqs, pca_seqs, labels, K):
    pi, A, dM, db, dCov = init_params(cebra_seqs, K, D)
    for _ in range(N_ITERS_SLDS):
        gammas, xis, lls = [], [], []
        for seq in cebra_seqs:
            g, x, ll = forward_backward(seq, pi, A, dM, db, dCov, K)
            gammas.append(g); xis.append(x); lls.append(ll)
        pi, A, dM, db, dCov = m_step(cebra_seqs, gammas, xis, K, D)

    total_ll = sum(lls)
    N = sum(len(s) for s in cebra_seqs)
    bic = compute_bic(total_ll, K, D, N)

    state_seqs = [np.argmax(g, axis=1) for g in gammas]
    all_states = np.concatenate(state_seqs)
    counts = np.bincount(all_states, minlength=K)
    K_eff = int(np.sum(counts / len(all_states) > 0.01))

    persist = np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs])

    confusion = np.zeros((K, len(STAGES)))
    for s_seq, l_seq in zip(state_seqs, labels):
        for s, l in zip(s_seq, l_seq):
            if l in STAGES: confusion[s, STAGES.index(l)] += 1
    conf_norm = confusion / (confusion.sum(1, keepdims=True) + 1e-9)
    spec = np.mean(np.max(conf_norm, axis=1))

    T_hard  = hard_transition_matrix(state_seqs, K)
    mean_st = np.diag(T_hard).mean()
    score   = sss(persist, mean_st, T_hard, K, K_eff)
    r2      = regime_r2_on_pca(state_seqs, pca_seqs)

    return persist, mean_st, spec, conf_norm, T_hard, score, K_eff, r2, bic


if __name__ == "__main__":
    path = "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_reasoning/layer_28/all_sentences_features.pkl"

    for mode in ['temporal']:
        print(f"\n{'='*50}\nRUNNING CEBRA-{mode.upper()}\n{'='*50}", flush=True)
        all_f, triplets = load_and_prepare_cebra(path, mode=mode)
        cebra_seqs, pca_seqs, labels = train_cebra_projection(all_f, triplets)

        ar_r2 = linear_ar_r2(pca_seqs)
        print(f"\n  Linear AR baseline R² (PCA space): {ar_r2:.4f}", flush=True)

        print(f"\n{'K':<4} | {'K_eff':<6} | {'Persist':<10} | {'Self-Trans':<10} | {'Spec':<10} | {'SSS':<8} | {'R²':<8} | {'ΔR²':<8} | {'BIC':<14}", flush=True)
        print("-" * 100, flush=True)

        sss_scores, bic_scores = {}, {}
        for k in K_SWEEP:
            p, st, s, C_mat, T_hard, score, k_eff, r2, bic = fit_and_evaluate(cebra_seqs, pca_seqs, labels, k)
            sss_scores[k] = score
            bic_scores[k] = bic
            delta_r2 = r2 - ar_r2
            print(f"{k:<4} | {k_eff:<6} | {p:<10.2f} | {st:<10.3f} | {s:<10.4f} | {score:<8.4f} | {r2:<8.4f} | {delta_r2:+.4f} | {bic:<14.1f}", flush=True)

            print(f"\n  Transition matrix (K={k}):", flush=True)
            print_transition_matrix(T_hard, k)

            if k >= 4:
                print(f"\n  Dominant stages (K={k}):", flush=True)
                for i in range(k):
                    idx = np.argmax(C_mat[i])
                    print(f"    Mode {i}: {STAGES[idx]} ({C_mat[i,idx]:.1%})", flush=True)
            print()

        best_k_sss = max(sss_scores, key=sss_scores.get)
        best_k_bic = min(bic_scores, key=bic_scores.get)
        print(f"\n  Best K by SSS: K={best_k_sss}  (SSS={sss_scores[best_k_sss]:.4f})", flush=True)
        print(f"  Best K by BIC: K={best_k_bic}  (BIC={bic_scores[best_k_bic]:.1f})", flush=True)
        print(f"  SSS profile: " + "  ".join([f"K{k}={v:.3f}" for k, v in sss_scores.items()]), flush=True)
        print(f"  BIC profile: " + "  ".join([f"K{k}={v:.0f}" for k, v in bic_scores.items()]), flush=True)