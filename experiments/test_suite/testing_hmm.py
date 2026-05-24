# # # import numpy as np
# # # import pickle
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # import torch.optim as optim
# # # from collections import defaultdict
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.decomposition import PCA
# # # from sklearn.cluster import KMeans

# # # # --- Config ---
# # # K_EVAL = [1, 4, 7]
# # # D = 40          # PCA rank k (paper uses 40)
# # # KAPPA = 10.0
# # # N_ITERS_SLDS = 50
# # # MLP_EPOCHS = 200
# # # BATCH_SIZE = 1024

# # # STAGES = [
# # #     "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
# # #     "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
# # #     "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
# # # ]

# # # # ── DATA LOADING ─────────────────────────────────────────────

# # # def load_sequences(path, hidden_key='hidden_state_last'):
# # #     with open(path, 'rb') as f:
# # #         features = pickle.load(f)
# # #     p_map = defaultdict(list)
# # #     for feat in features:
# # #         p_map[feat['problem_id']].append(feat)
# # #     sequences, label_seqs = [], []
# # #     for pid in sorted(p_map.keys()):
# # #         feats = sorted(p_map[pid], key=lambda x: x['sentence_idx'])
# # #         if len(feats) >= 3:
# # #             sequences.append(np.array([f[hidden_key] for f in feats]))
# # #             label_seqs.append([f['stage'] for f in feats])
# # #     return sequences, label_seqs

# # # def fit_pca(sequences, n_components=D):
# # #     """Fit PCA on absolute h_t. Returns scaler, pca, and Vk (D_raw x k)."""
# # #     X_all = np.concatenate(sequences, axis=0)
# # #     scaler = StandardScaler()
# # #     X_scaled = scaler.fit_transform(X_all)
# # #     pca = PCA(n_components=n_components)
# # #     pca.fit(X_scaled)
# # #     var_explained = pca.explained_variance_ratio_.sum()
# # #     print(f"  PCA rank-{n_components} explains {var_explained:.1%} of variance", flush=True)
# # #     return scaler, pca

# # # def project(sequences, scaler, pca):
# # #     """Project sequences into k-dim PCA space: x_t = V_k^T h_t (standardized)."""
# # #     projected = []
# # #     for seq in sequences:
# # #         x = scaler.transform(seq)
# # #         projected.append(pca.transform(x))
# # #     return projected

# # # def get_deltas(proj_sequences):
# # #     """Δx'_t = x_{t+1} - x_t in PCA space."""
# # #     return [np.diff(s, axis=0) for s in proj_sequences]

# # # # ── SLDS (paper formulation) ──────────────────────────────────
# # # # Dynamics: Δx'_t = M_k * x_t + b_k + ε,  all in k-dim PCA space
# # # # Emission is on Δx'_t (the projected delta), not on x_{t+1} directly.
# # # # Prediction: x̂_{t+1} = x_t + Σ_k γ_{t,k} (M_k x_t + b_k)

# # # def init_params(proj_seqs, K):
# # #     """Init on projected deltas Δx'."""
# # #     k = proj_seqs[0].shape[1]
# # #     X_in, dX = [], []
# # #     for seq in proj_seqs:
# # #         for t in range(len(seq)-1):
# # #             X_in.append(seq[t]); dX.append(seq[t+1] - seq[t])
# # #     X_in, dX = np.array(X_in), np.array(dX)
# # #     labels = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(dX)
# # #     M = np.zeros((K, k, k))
# # #     b = np.zeros((K, k))
# # #     Cov = np.array([np.eye(k) for _ in range(K)])
# # #     for j in range(K):
# # #         mask = labels == j
# # #         if mask.sum() < k + 2: M[j] = 0.1 * np.eye(k); continue
# # #         W, *_ = np.linalg.lstsq(
# # #             np.hstack([X_in[mask], np.ones((mask.sum(), 1))]), dX[mask], rcond=None)
# # #         M[j], b[j] = W[:k].T, W[k]
# # #         res = dX[mask] - (X_in[mask] @ M[j].T + b[j])
# # #         Cov[j] = np.cov(res.T) + 1e-3 * np.eye(k)
# # #     return np.ones(K)/K, np.eye(K)*0.7 + 0.3/K, M, b, Cov

# # # def get_log_emissions(proj_seq, K, M, b, Cov):
# # #     """Emission on Δx'_t = x_{t+1} - x_t."""
# # #     T, k = proj_seq.shape
# # #     dX = np.diff(proj_seq, axis=0)          # (T-1, k)
# # #     log_emit = np.full((T, K), -1e9)
# # #     for j in range(K):
# # #         _, logdet = np.linalg.slogdet(Cov[j])
# # #         inv_cov = np.linalg.inv(Cov[j])
# # #         mean_j = proj_seq[:-1] @ M[j].T + b[j]  # (T-1, k)
# # #         diff = dX - mean_j
# # #         mahal = np.sum((diff @ inv_cov) * diff, axis=1)
# # #         log_emit[1:, j] = -0.5 * (k * np.log(2*np.pi) + logdet + mahal)
# # #     log_emit[0] = 0.0  # uniform at t=0
# # #     return log_emit

# # # def forward_backward(proj_seq, pi, A, M, b, Cov, K):
# # #     T = len(proj_seq)
# # #     log_emit = get_log_emissions(proj_seq, K, M, b, Cov)
# # #     log_A, log_pi = np.log(A + 1e-12), np.log(pi + 1e-12)
# # #     l_alpha = np.zeros((T, K))
# # #     l_alpha[0] = log_pi + log_emit[0]
# # #     for t in range(1, T):
# # #         l_alpha[t] = log_emit[t] + np.logaddexp.reduce(l_alpha[t-1][:, None] + log_A, axis=0)
# # #     l_beta = np.zeros((T, K))
# # #     for t in range(T-2, -1, -1):
# # #         l_beta[t] = np.logaddexp.reduce(log_A + log_emit[t+1] + l_beta[t+1], axis=1)
# # #     l_gamma = l_alpha + l_beta
# # #     l_gamma -= np.logaddexp.reduce(l_gamma, axis=1, keepdims=True)
# # #     l_xi = np.zeros((T-1, K, K))
# # #     for t in range(T-1):
# # #         l_xi[t] = l_alpha[t][:, None] + log_A + log_emit[t+1] + l_beta[t+1]
# # #         l_xi[t] -= np.logaddexp.reduce(l_xi[t].ravel())
# # #     return np.exp(l_gamma), np.exp(l_xi)

# # # def m_step(proj_seqs, gammas, xis, K):
# # #     k = proj_seqs[0].shape[1]
# # #     xi_sum = sum(xi.sum(axis=0) for xi in xis) + np.eye(K) * KAPPA + 1e-8
# # #     A_new = xi_sum / xi_sum.sum(axis=1, keepdims=True)
# # #     pi_new = np.maximum(np.mean([g[0] for g in gammas], axis=0), 1e-8)
# # #     pi_new /= pi_new.sum()
# # #     M_new = np.zeros((K, k, k))
# # #     b_new = np.zeros((K, k))
# # #     Cov_new = np.zeros((K, k, k))
# # #     for j in range(K):
# # #         W_sum  = np.zeros((k+1, k+1))
# # #         WY_sum = np.zeros((k+1, k))
# # #         for seq, gamma in zip(proj_seqs, gammas):
# # #             dX = np.diff(seq, axis=0)       # (T-1, k)
# # #             X_aug = np.hstack([seq[:-1], np.ones((len(seq)-1, 1))])  # (T-1, k+1)
# # #             w = gamma[1:, j]                # weight at t+1 per paper's ξ indexing
# # #             W_sum  += (X_aug * w[:, None]).T @ X_aug
# # #             WY_sum += (X_aug * w[:, None]).T @ dX
# # #         coef = np.linalg.solve(W_sum + 1e-4 * np.eye(k+1), WY_sum)
# # #         M_new[j], b_new[j] = coef[:k].T, coef[k]
# # #         num, den = np.zeros((k, k)), 1e-9
# # #         for seq, gamma in zip(proj_seqs, gammas):
# # #             dX = np.diff(seq, axis=0)
# # #             err = dX - (seq[:-1] @ M_new[j].T + b_new[j])
# # #             w = gamma[1:, j]
# # #             num += (err * w[:, None]).T @ err
# # #             den += w.sum()
# # #         Cov_new[j] = num / den + 1e-4 * np.eye(k)
# # #     return pi_new, A_new, M_new, b_new, Cov_new

# # # def fit_slds(proj_seqs, K):
# # #     pi, A, M, b, Cov = init_params(proj_seqs, K)
# # #     for _ in range(N_ITERS_SLDS):
# # #         gammas, xis = [], []
# # #         for seq in proj_seqs:
# # #             g, x = forward_backward(seq, pi, A, M, b, Cov, K)
# # #             gammas.append(g); xis.append(x)
# # #         pi, A, M, b, Cov = m_step(proj_seqs, gammas, xis, K)
# # #     return pi, A, M, b, Cov, gammas

# # # def slds_predict(proj_seq, pi, A, M, b, Cov, K):
# # #     """x̂_{t+1} = x_t + Σ_k γ_{t,k} (M_k x_t + b_k)  — eq. 10 from paper."""
# # #     gamma, _ = forward_backward(proj_seq, pi, A, M, b, Cov, K)
# # #     preds = np.zeros_like(proj_seq)
# # #     preds[0] = proj_seq[0]
# # #     for t in range(len(proj_seq)-1):
# # #         drift = sum(gamma[t, j] * (proj_seq[t] @ M[j].T + b[j]) for j in range(K))
# # #         preds[t+1] = proj_seq[t] + drift
# # #     return preds

# # # # ── BASELINES ────────────────────────────────────────────────

# # # def fit_linear_ar(proj_seqs):
# # #     """Global linear: Δx'_t = A x_t + c  (matches paper eq. 5-6)."""
# # #     X_in, dX = [], []
# # #     for seq in proj_seqs:
# # #         X_in.append(seq[:-1]); dX.append(np.diff(seq, axis=0))
# # #     X_in = np.concatenate(X_in, axis=0)
# # #     dX   = np.concatenate(dX,   axis=0)
# # #     W, *_ = np.linalg.lstsq(
# # #         np.hstack([X_in, np.ones((len(X_in), 1))]), dX, rcond=None)
# # #     return W[:-1].T, W[-1]   # M, b  s.t. Δx' ≈ M x + b

# # # def predict_linear_ar(proj_seq, M, b):
# # #     preds = np.zeros_like(proj_seq)
# # #     preds[0] = proj_seq[0]
# # #     for t in range(len(proj_seq)-1):
# # #         preds[t+1] = proj_seq[t] + proj_seq[t] @ M.T + b
# # #     return preds

# # # class ARNet(nn.Module):
# # #     def __init__(self, k):
# # #         super().__init__()
# # #         self.net = nn.Sequential(
# # #             nn.Linear(k, 256), nn.GELU(),
# # #             nn.Linear(256, 256), nn.GELU(),
# # #             nn.Linear(256, k)
# # #         )
# # #     def forward(self, x): return self.net(x)

# # # def fit_mlp_ar(proj_seqs, device='cpu'):
# # #     """MLP predicts Δx'_t from x_t."""
# # #     X_in, dX = [], []
# # #     for seq in proj_seqs:
# # #         X_in.append(seq[:-1]); dX.append(np.diff(seq, axis=0))
# # #     X_in = torch.tensor(np.concatenate(X_in), dtype=torch.float32).to(device)
# # #     dX   = torch.tensor(np.concatenate(dX),   dtype=torch.float32).to(device)
# # #     k = X_in.shape[1]
# # #     model = ARNet(k).to(device)
# # #     opt = optim.Adam(model.parameters(), lr=1e-3)
# # #     for ep in range(MLP_EPOCHS):
# # #         idx = torch.randperm(len(X_in))
# # #         for i in range(0, len(X_in), BATCH_SIZE):
# # #             bi = idx[i:i+BATCH_SIZE]
# # #             loss = F.mse_loss(model(X_in[bi]), dX[bi])
# # #             opt.zero_grad(); loss.backward(); opt.step()
# # #         if (ep+1) % 50 == 0:
# # #             print(f"    MLP epoch {ep+1}/{MLP_EPOCHS} | loss={loss.item():.5f}", flush=True)
# # #     model.eval()
# # #     return model

# # # def predict_mlp(proj_seq, model, device='cpu'):
# # #     preds = np.zeros_like(proj_seq)
# # #     preds[0] = proj_seq[0]
# # #     with torch.no_grad():
# # #         x = torch.tensor(proj_seq[:-1], dtype=torch.float32).to(device)
# # #         delta = model(x).cpu().numpy()
# # #     preds[1:] = proj_seq[:-1] + delta
# # #     return preds

# # # # ── METRICS ──────────────────────────────────────────────────

# # # def r2_score(seqs, preds_list):
# # #     """R² on x_{t+1} prediction (matches paper metric)."""
# # #     all_true, all_pred = [], []
# # #     for seq, pred in zip(seqs, preds_list):
# # #         all_true.append(seq[1:])
# # #         all_pred.append(pred[1:])
# # #     y_true = np.concatenate(all_true)
# # #     y_pred = np.concatenate(all_pred)
# # #     ss_res = np.sum((y_true - y_pred)**2)
# # #     ss_tot = np.sum((y_true - y_true.mean(0))**2)
# # #     return 1 - ss_res / ss_tot

# # # def nll_score(seqs, pi, A, M, b, Cov, K):
# # #     """Average NLL per transition under the fitted SLDS."""
# # #     total_nll, total_n = 0.0, 0
# # #     for seq in seqs:
# # #         log_emit = get_log_emissions(seq, K, M, b, Cov)
# # #         log_A_mat, log_pi = np.log(A + 1e-12), np.log(pi + 1e-12)
# # #         # forward pass to get log p(h_1,...,h_T)
# # #         log_alpha = np.zeros((len(seq), K))
# # #         log_alpha[0] = log_pi + log_emit[0]
# # #         for t in range(1, len(seq)):
# # #             log_alpha[t] = log_emit[t] + np.logaddexp.reduce(
# # #                 log_alpha[t-1][:, None] + log_A_mat, axis=0)
# # #         log_likelihood = np.logaddexp.reduce(log_alpha[-1])
# # #         total_nll += -log_likelihood
# # #         total_n   += len(seq) - 1
# # #     return total_nll / total_n

# # # # ── MODEL COMPARISON ─────────────────────────────────────────

# # # def run_model_comparison(proj_seqs, labels, tag=""):
# # #     device = "cuda" if torch.cuda.is_available() else "cpu"
# # #     n = len(proj_seqs)
# # #     idx = np.random.permutation(n)
# # #     train_idx, test_idx = idx[:int(0.8*n)], idx[int(0.8*n):]
# # #     train_seqs = [proj_seqs[i] for i in train_idx]
# # #     test_seqs  = [proj_seqs[i] for i in test_idx]
# # #     train_labs = [labels[i]    for i in train_idx]

# # #     print(f"\n{'='*60}\nMODEL COMPARISON — {tag}\n{'='*60}", flush=True)
# # #     print(f"  train={len(train_seqs)}, test={len(test_seqs)}", flush=True)

# # #     # Linear AR
# # #     print("\nFitting Linear AR...", flush=True)
# # #     M_ar, b_ar = fit_linear_ar(train_seqs)
# # #     ar_preds = [predict_linear_ar(s, M_ar, b_ar) for s in test_seqs]
# # #     ar_r2 = r2_score(test_seqs, ar_preds)
# # #     print(f"  Linear AR   R²: {ar_r2:.4f}", flush=True)

# # #     # MLP
# # #     print("\nFitting MLP AR...", flush=True)
# # #     mlp = fit_mlp_ar(train_seqs, device)
# # #     mlp_preds = [predict_mlp(s, mlp, device) for s in test_seqs]
# # #     mlp_r2 = r2_score(test_seqs, mlp_preds)
# # #     print(f"  MLP AR      R²: {mlp_r2:.4f}", flush=True)

# # #     # SLDS
# # #     print(f"\n{'K':<4} | {'R²':<8} | {'NLL':<8} | {'vs LinAR':10} | {'vs MLP':10} | {'Spec':<8} | {'Persist'}", flush=True)
# # #     print('-'*72, flush=True)
# # #     for K in K_EVAL:
# # #         print(f"\nFitting SLDS K={K}...", flush=True)
# # #         pi, A, M, b, Cov, gammas = fit_slds(train_seqs, K)
# # #         slds_preds = [slds_predict(s, pi, A, M, b, Cov, K) for s in test_seqs]
# # #         slds_r2  = r2_score(test_seqs, slds_preds)
# # #         slds_nll = nll_score(test_seqs, pi, A, M, b, Cov, K)

# # #         state_seqs = [np.argmax(g, axis=1) for g in gammas]
# # #         persist = np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs])
# # #         confusion = np.zeros((K, len(STAGES)))
# # #         for s_seq, l_seq in zip(state_seqs, train_labs):
# # #             for s, l in zip(s_seq, l_seq):
# # #                 if l in STAGES: confusion[s, STAGES.index(l)] += 1
# # #         fwd_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-9)
# # #         spec = np.mean(np.max(fwd_norm, axis=1))

# # #         delta_ar  = slds_r2 - ar_r2
# # #         delta_mlp = slds_r2 - mlp_r2
# # #         print(f"  K={K:<2} | {slds_r2:<8.4f} | {slds_nll:<8.2f} | {delta_ar:+.4f}{'':4} | {delta_mlp:+.4f}{'':4} | {spec:.3f}   | {persist:.2f}", flush=True)

# # #     print(f"\n  Summary:", flush=True)
# # #     print(f"    Linear AR : R²={ar_r2:.4f}", flush=True)
# # #     print(f"    MLP AR    : R²={mlp_r2:.4f}", flush=True)

# # # # ── MAIN ─────────────────────────────────────────────────────

# # # if __name__ == "__main__":
# # #     path = "/home/abir19/scratch/abir19/new_rpc_math500_layer28_qwen14/all_sentences_features.pkl"

# # #     for hidden_key, tag in [('hidden_state_last', 'last-token'), ('hidden_state', 'mean-pool')]:
# # #         print(f"\n{'='*60}\nLoading {tag}...\n{'='*60}", flush=True)
# # #         sequences, labels = load_sequences(path, hidden_key)
# # #         print(f"  {len(sequences)} sequences", flush=True)

# # #         # fit PCA on absolute h_t (paper approach)
# # #         scaler, pca = fit_pca(sequences, n_components=D)
# # #         proj_seqs = project(sequences, scaler, pca)

# # #         run_model_comparison(proj_seqs, labels, tag=f"{tag} | PCA-{D}")



# # import os
# # import numpy as np
# # import pickle
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # from collections import defaultdict
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.cluster import KMeans

# # # --- Config ---
# # K_SWEEP     = range(1, 13)
# # D           = 40
# # CEBRA_EPOCHS = 100
# # BATCH_SIZE  = 1024
# # KAPPA       = 10.0
# # N_ITERS_SLDS = 50

# # STAGES = [
# #     "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
# #     "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
# #     "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
# # ]

# # # ── DATA ─────────────────────────────────────────────────────

# # def load_and_prepare_cebra(path, limit_problems=500, max_triplets=25):
# #     with open(path, 'rb') as f:
# #         all_features = pickle.load(f)
# #     all_features = [f for f in all_features if f['problem_id'] < limit_problems]

# #     p_map = defaultdict(list)
# #     for i, f in enumerate(all_features):
# #         p_map[f['problem_id']].append(i)

# #     # compute inverse-frequency weights per stage
# #     stage_counts = defaultdict(int)
# #     for f in all_features:
# #         stage_counts[f.get('stage', 'NEUTRAL')] += 1
# #     total = sum(stage_counts.values())
# #     stage_weight = {s: total / (len(stage_counts) * c) for s, c in stage_counts.items()}
# #     print("  Stage weights (inverse freq):", flush=True)
# #     for s, w in sorted(stage_weight.items(), key=lambda x: -x[1]):
# #         print(f"    {s}: {w:.2f}  (n={stage_counts[s]})", flush=True)

# #     triplets = []
# #     pids = list(p_map.keys())

# #     for pid in pids:
# #         idxs = p_map[pid]
# #         if len(idxs) < 2:
# #             continue

# #         # weight each transition by the anchor's stage frequency
# #         transitions = list(range(len(idxs) - 1))
# #         weights = np.array([stage_weight.get(all_features[idxs[t]].get('stage', 'NEUTRAL'), 1.0)
# #                             for t in transitions])
# #         weights /= weights.sum()

# #         num_samples = min(len(transitions), max_triplets)
# #         sampled = np.random.choice(transitions, size=num_samples, replace=False, p=weights)

# #         for t in sampled:
# #             anchor   = idxs[t]
# #             positive = idxs[t + 1]
# #             a_stage  = all_features[anchor].get('stage', 'NEUTRAL')

# #             # hard negative: same problem, different stage
# #             hard_pool = [i for i in idxs if all_features[i].get('stage', 'NEUTRAL') != a_stage]
# #             if hard_pool:
# #                 negative = np.random.choice(hard_pool)
# #             else:
# #                 # fallback: cross-problem
# #                 other_pid = np.random.choice([p for p in pids if p != pid])
# #                 negative  = np.random.choice(p_map[other_pid])

# #             triplets.append((anchor, positive, negative))

# #     print(f"  {len(triplets)} triplets from {len(pids)} problems", flush=True)
# #     return all_features, triplets

# # # ── CEBRA MODEL ──────────────────────────────────────────────

# # class CEBRANet(nn.Module):
# #     def __init__(self, d_in, d_out):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(d_in, 512), nn.GELU(),
# #             nn.Linear(512, 256), nn.GELU(),
# #             nn.Linear(256, d_out)
# #         )
# #     def forward(self, x):
# #         return F.normalize(self.net(x), p=2, dim=1)

# # def train_cebra_projection(all_features, triplets, d_out=D):
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     X_raw  = np.array([f['hidden_state_last'] for f in all_features])
# #     scaler = StandardScaler()
# #     X_sc   = torch.tensor(scaler.fit_transform(X_raw), dtype=torch.float32).to(device)
# #     model  = CEBRANet(X_raw.shape[1], d_out).to(device)
# #     opt    = optim.Adam(model.parameters(), lr=1e-3)
# #     trips  = np.array(triplets)

# #     for epoch in range(CEBRA_EPOCHS):
# #         idx = np.random.permutation(len(trips))
# #         for i in range(0, len(trips), BATCH_SIZE):
# #             b = trips[idx[i:i+BATCH_SIZE]]
# #             za = model(X_sc[b[:, 0]])
# #             zp = model(X_sc[b[:, 1]])
# #             zn = model(X_sc[b[:, 2]])
# #             pos = torch.sum(za * zp, dim=1) / 0.1
# #             neg = torch.sum(za * zn, dim=1) / 0.1
# #             loss = -torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg))).mean()
# #             opt.zero_grad(); loss.backward(); opt.step()
# #         if (epoch + 1) % 25 == 0:
# #             print(f"  epoch {epoch+1}/{CEBRA_EPOCHS} | loss={loss.item():.4f}", flush=True)

# #     model.eval()
# #     with torch.no_grad():
# #         Z = model(X_sc).cpu().numpy()

# #     p_map_z, p_map_l = defaultdict(list), defaultdict(list)
# #     for i, f in enumerate(all_features):
# #         p_map_z[f['problem_id']].append(Z[i])
# #         p_map_l[f['problem_id']].append(f.get('stage', 'NEUTRAL'))

# #     seqs   = [np.array(p_map_z[p]) for p in sorted(p_map_z) if len(p_map_z[p]) >= 3]
# #     labels = [p_map_l[p]           for p in sorted(p_map_z) if len(p_map_z[p]) >= 3]
# #     return seqs, labels

# # # ── SLDS EM ───────────────────────────────────────────────────

# # def init_params(sequences, K, d):
# #     X_in = np.concatenate([s[:-1] for s in sequences])
# #     dX   = np.concatenate([np.diff(s, axis=0) for s in sequences])
# #     lbl  = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(dX)
# #     M = np.zeros((K, d, d)); b = np.zeros((K, d))
# #     Cov = np.array([np.eye(d)] * K)
# #     for k in range(K):
# #         mask = lbl == k
# #         if mask.sum() < d + 2: M[k] = 0.1 * np.eye(d); continue
# #         W, *_ = np.linalg.lstsq(np.hstack([X_in[mask], np.ones((mask.sum(),1))]), dX[mask], rcond=None)
# #         M[k], b[k] = W[:d].T, W[d]
# #         res = dX[mask] - (X_in[mask] @ M[k].T + b[k])
# #         Cov[k] = np.cov(res.T) + 1e-3 * np.eye(d)
# #     return np.ones(K)/K, np.eye(K)*0.7 + 0.3/K, M, b, Cov

# # def get_log_emissions(seq, K, M, b, Cov):
# #     T, d = seq.shape
# #     dX = np.diff(seq, axis=0)
# #     log_emit = np.full((T, K), -1e9)
# #     for k in range(K):
# #         _, logdet = np.linalg.slogdet(Cov[k])
# #         inv_cov   = np.linalg.inv(Cov[k])
# #         mean_k    = seq[:-1] @ M[k].T + b[k]
# #         diff      = dX - mean_k
# #         mahal     = np.sum((diff @ inv_cov) * diff, axis=1)
# #         log_emit[1:, k] = -0.5 * (d * np.log(2*np.pi) + logdet + mahal)
# #     log_emit[0] = 0.0
# #     return log_emit

# # def forward_backward(seq, pi, A, M, b, Cov, K):
# #     T = len(seq)
# #     log_emit = get_log_emissions(seq, K, M, b, Cov)
# #     log_A, log_pi = np.log(A + 1e-12), np.log(pi + 1e-12)
# #     la = np.zeros((T, K)); la[0] = log_pi + log_emit[0]
# #     for t in range(1, T):
# #         la[t] = log_emit[t] + np.logaddexp.reduce(la[t-1][:, None] + log_A, axis=0)
# #     lb = np.zeros((T, K))
# #     for t in range(T-2, -1, -1):
# #         lb[t] = np.logaddexp.reduce(log_A + log_emit[t+1] + lb[t+1], axis=1)
# #     lg = la + lb; lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
# #     lxi = np.zeros((T-1, K, K))
# #     for t in range(T-1):
# #         lxi[t] = la[t][:, None] + log_A + log_emit[t+1] + lb[t+1]
# #         lxi[t] -= np.logaddexp.reduce(lxi[t].ravel())
# #     return np.exp(lg), np.exp(lxi)

# # def m_step(sequences, gammas, xis, K, d):
# #     xi_sum = sum(xi.sum(0) for xi in xis) + np.eye(K) * KAPPA + 1e-8
# #     A_new  = xi_sum / xi_sum.sum(axis=1, keepdims=True)
# #     pi_new = np.maximum(np.mean([g[0] for g in gammas], axis=0), 1e-8)
# #     pi_new /= pi_new.sum()
# #     M_new  = np.zeros((K, d, d)); b_new = np.zeros((K, d)); Cov_new = np.zeros((K, d, d))
# #     for k in range(K):
# #         W_sum, WY_sum = np.zeros((d+1, d+1)), np.zeros((d+1, d))
# #         for seq, gamma in zip(sequences, gammas):
# #             dX    = np.diff(seq, axis=0)
# #             X_aug = np.hstack([seq[:-1], np.ones((len(seq)-1, 1))])
# #             w     = gamma[1:, k]
# #             W_sum  += (X_aug * w[:, None]).T @ X_aug
# #             WY_sum += (X_aug * w[:, None]).T @ dX
# #         coef = np.linalg.solve(W_sum + 1e-4 * np.eye(d+1), WY_sum)
# #         M_new[k], b_new[k] = coef[:d].T, coef[d]
# #         num, den = np.zeros((d, d)), 1e-9
# #         for seq, gamma in zip(sequences, gammas):
# #             dX  = np.diff(seq, axis=0)
# #             err = dX - (seq[:-1] @ M_new[k].T + b_new[k])
# #             w   = gamma[1:, k]
# #             num += (err * w[:, None]).T @ err; den += w.sum()
# #         Cov_new[k] = num / den + 1e-4 * np.eye(d)
# #     return pi_new, A_new, M_new, b_new, Cov_new

# # def fit_and_evaluate(sequences, labels, K):
# #     d = sequences[0].shape[1]
# #     pi, A, M, b, Cov = init_params(sequences, K, d)
# #     for _ in range(N_ITERS_SLDS):
# #         gammas, xis = [], []
# #         for seq in sequences:
# #             g, x = forward_backward(seq, pi, A, M, b, Cov, K)
# #             gammas.append(g); xis.append(x)
# #         pi, A, M, b, Cov = m_step(sequences, gammas, xis, K, d)

# #     state_seqs = [np.argmax(g, axis=1) for g in gammas]
# #     persist    = np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs])
# #     confusion  = np.zeros((K, len(STAGES)))
# #     for s_seq, l_seq in zip(state_seqs, labels):
# #         for s, l in zip(s_seq, l_seq):
# #             if l in STAGES: confusion[s, STAGES.index(l)] += 1
# #     fwd = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-9)
# #     rev = confusion / (confusion.sum(axis=0, keepdims=True) + 1e-9)
# #     spec = np.mean(np.max(fwd, axis=1))
# #     return persist, np.mean(np.diag(A)), spec, A, fwd, rev

# # # ── MAIN ─────────────────────────────────────────────────────

# # if __name__ == "__main__":
# #     path = "/home/abir19/scratch/abir19/rpc_dataset_math500_layer_final_500_qwen_14/all_sentences_features.pkl"

# #     print("Loading & balancing data...", flush=True)
# #     all_f, triplets = load_and_prepare_cebra(path)
# #     print("Training CEBRA projection...", flush=True)
# #     seqs, labels = train_cebra_projection(all_f, triplets)

# #     print(f"\n{'K':<4} | {'Persist':<10} | {'Self-Trans':<10} | {'Spec'}", flush=True)
# #     for k in K_SWEEP:
# #         p, st, s, A_mat, F_mat, R_mat = fit_and_evaluate(seqs, labels, k)
# #         print(f"{k:<4} | {p:<10.2f} | {st:<10.3f} | {s:.4f}", flush=True)

# #         if k == 7:
# #             header = " ".join([f"{s[:5]:>7}" for s in STAGES])
# #             print(f"\nMode → Stage (rows sum to 1):", flush=True)
# #             print(f"{'':>10} {header}", flush=True)
# #             for i in range(k):
# #                 row = " ".join([f"{F_mat[i,j]:7.2f}" for j in range(len(STAGES))])
# #                 print(f"  Mode {i:<3}: {row}", flush=True)
# #             print(f"\nDominant stage per mode:", flush=True)
# #             for i in range(k):
# #                 idx = np.argmax(F_mat[i])
# #                 print(f"  Mode {i}: {STAGES[idx]} ({F_mat[i,idx]:.1%})", flush=True)
                


# # import os
# # import numpy as np
# # import pickle
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # from collections import defaultdict
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.cluster import KMeans

# # # --- Config ---
# # K_SWEEP      = range(1, 13)
# # D            = 40
# # CEBRA_EPOCHS = 100
# # MOE_EPOCHS   = 100
# # BATCH_SIZE   = 1024
# # KAPPA        = 10.0
# # N_ITERS_SLDS = 50

# # STAGES = [
# #     "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
# #     "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
# #     "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
# # ]

# # # ── DATA ─────────────────────────────────────────────────────

# # def load_and_prepare(path, limit_problems=500, max_triplets=25):
# #     with open(path, 'rb') as f:
# #         all_features = pickle.load(f)
# #     all_features = [f for f in all_features if f['problem_id'] < limit_problems]

# #     p_map = defaultdict(list)
# #     for i, f in enumerate(all_features):
# #         p_map[f['problem_id']].append(i)

# #     stage_counts = defaultdict(int)
# #     for f in all_features:
# #         stage_counts[f.get('stage', 'NEUTRAL')] += 1
# #     total = sum(stage_counts.values())
# #     stage_weight = {s: total / (len(stage_counts) * c) for s, c in stage_counts.items()}

# #     print("  Stage weights (inverse freq):", flush=True)
# #     for s, w in sorted(stage_weight.items(), key=lambda x: -x[1]):
# #         print(f"    {s}: {w:.2f}  (n={stage_counts[s]})", flush=True)

# #     triplets, pids = [], list(p_map.keys())
# #     for pid in pids:
# #         idxs = p_map[pid]
# #         if len(idxs) < 2: continue
# #         transitions = list(range(len(idxs) - 1))
# #         weights = np.array([stage_weight.get(all_features[idxs[t]].get('stage','NEUTRAL'), 1.0)
# #                             for t in transitions])
# #         weights /= weights.sum()
# #         sampled = np.random.choice(transitions, size=min(len(transitions), max_triplets),
# #                                    replace=False, p=weights)
# #         for t in sampled:
# #             anchor, positive = idxs[t], idxs[t+1]
# #             a_stage = all_features[anchor].get('stage', 'NEUTRAL')
# #             hard_pool = [i for i in idxs if all_features[i].get('stage','NEUTRAL') != a_stage]
# #             negative = np.random.choice(hard_pool) if hard_pool else \
# #                        np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])
# #             triplets.append((anchor, positive, negative))

# #     print(f"  {len(triplets)} triplets from {len(pids)} problems", flush=True)
# #     return all_features, triplets

# # # ── ENCODER ──────────────────────────────────────────────────

# # class Encoder(nn.Module):
# #     def __init__(self, d_in, d_out):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(d_in, 512), nn.GELU(),
# #             nn.Linear(512, 256), nn.GELU(),
# #             nn.Linear(256, d_out))
# #     def forward(self, x):
# #         return F.normalize(self.net(x), p=2, dim=1)

# # # ── APPROACH 1: CEBRA only ────────────────────────────────────

# # def train_cebra(all_features, triplets, d_out=D):
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     X_raw  = np.array([f['hidden_state_last'] for f in all_features])
# #     X_sc   = torch.tensor(StandardScaler().fit_transform(X_raw), dtype=torch.float32).to(device)
# #     model  = Encoder(X_raw.shape[1], d_out).to(device)
# #     opt    = optim.Adam(model.parameters(), lr=1e-3)
# #     trips  = np.array(triplets)

# #     for epoch in range(CEBRA_EPOCHS):
# #         idx = np.random.permutation(len(trips))
# #         for i in range(0, len(trips), BATCH_SIZE):
# #             b  = trips[idx[i:i+BATCH_SIZE]]
# #             za, zp, zn = model(X_sc[b[:,0]]), model(X_sc[b[:,1]]), model(X_sc[b[:,2]])
# #             pos = torch.sum(za*zp, dim=1) / 0.1
# #             neg = torch.sum(za*zn, dim=1) / 0.1
# #             loss = -torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg))).mean()
# #             opt.zero_grad(); loss.backward(); opt.step()
# #         if (epoch+1) % 25 == 0:
# #             print(f"    epoch {epoch+1}/{CEBRA_EPOCHS} | loss={loss.item():.4f}", flush=True)

# #     model.eval()
# #     with torch.no_grad():
# #         Z = model(X_sc).cpu().numpy()
# #     return Z

# # # ── APPROACH 2: CEBRA-MoE (paper approach) ───────────────────

# # class GatingHead(nn.Module):
# #     def __init__(self, d, K):
# #         super().__init__()
# #         self.net = nn.Sequential(nn.Linear(d,128), nn.LayerNorm(128),
# #                                  nn.Softplus(), nn.Linear(128, K, bias=False))
# #     def forward(self, z, temp=1.0):
# #         return F.gumbel_softmax(self.net(z), tau=temp, hard=False)

# # class DynamicsExpert(nn.Module):
# #     def __init__(self, d, K):
# #         super().__init__()
# #         self.experts = nn.ModuleList([
# #             nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Linear(128,d))
# #             for _ in range(K)])
# #     def forward(self, z, s):
# #         preds = torch.stack([m(z) for m in self.experts], dim=1)
# #         return (s.unsqueeze(-1) * preds).sum(dim=1)

# # def train_cebra_moe(all_features, triplets, K, d_out=D):
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     X_raw  = np.array([f['hidden_state_last'] for f in all_features])
# #     X_sc   = torch.tensor(StandardScaler().fit_transform(X_raw), dtype=torch.float32).to(device)
# #     trips  = np.array(triplets)

# #     encoder = Encoder(X_raw.shape[1], d_out).to(device)
# #     gate    = GatingHead(d_out, K).to(device)
# #     dyn     = DynamicsExpert(d_out, K).to(device)
# #     opt     = optim.AdamW(list(encoder.parameters()) + list(gate.parameters()) +
# #                           list(dyn.parameters()), lr=1e-3)

# #     for epoch in range(MOE_EPOCHS):
# #         tau   = max(0.2, 1.5 * (0.92 ** epoch))
# #         w_div = min(30.0, (epoch / 15.0) * 30.0)
# #         idx   = np.random.permutation(len(trips))
# #         for i in range(0, len(trips), BATCH_SIZE):
# #             b      = trips[idx[i:i+BATCH_SIZE]]
# #             z_a    = encoder(X_sc[b[:,0]])
# #             z_p    = encoder(X_sc[b[:,1]])
# #             z_n    = encoder(X_sc[b[:,2]])
# #             s_a    = gate(z_a, tau)
# #             s_p    = gate(z_p, tau)
# #             z_pred = dyn(z_a, s_a)
# #             # z_pred in NCE instead of z_a — paper's key trick
# #             pos    = torch.sum(z_pred * z_p, dim=1) / 0.05
# #             neg    = torch.sum(z_pred * z_n, dim=1) / 0.05
# #             l_nce  = -torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg))).mean()
# #             l_mse  = F.mse_loss(z_pred, z_p)
# #             l_pers = torch.abs(s_a - s_p).mean()
# #             l_div  = (s_a.mean(0) * torch.log(s_a.mean(0) + 1e-8)).sum()
# #             loss   = l_nce + 10.0*l_mse + 1.0*l_pers + w_div*l_div
# #             opt.zero_grad(); loss.backward(); opt.step()

# #         if (epoch+1) % 25 == 0:
# #             print(f"    epoch {epoch+1}/{MOE_EPOCHS} | nce={l_nce.item():.4f} "
# #                   f"mse={l_mse.item():.4f} pers={l_pers.item():.4f}", flush=True)

# #     encoder.eval(); gate.eval()
# #     with torch.no_grad():
# #         Z      = encoder(X_sc).cpu().numpy()
# #         states = gate(encoder(X_sc), tau=0.01).argmax(1).cpu().numpy()
# #     return Z, states

# # # ── SEQUENCES ────────────────────────────────────────────────

# # def make_sequences(Z, all_features):
# #     p_map_z, p_map_l = defaultdict(list), defaultdict(list)
# #     for i, f in enumerate(all_features):
# #         p_map_z[f['problem_id']].append(Z[i])
# #         p_map_l[f['problem_id']].append(f.get('stage', 'NEUTRAL'))
# #     seqs   = [np.array(p_map_z[p]) for p in sorted(p_map_z) if len(p_map_z[p]) >= 3]
# #     labels = [p_map_l[p]           for p in sorted(p_map_z) if len(p_map_z[p]) >= 3]
# #     return seqs, labels

# # def make_state_sequences(states, all_features):
# #     p_map_s, p_map_l = defaultdict(list), defaultdict(list)
# #     for i, f in enumerate(all_features):
# #         p_map_s[f['problem_id']].append(int(states[i]))
# #         p_map_l[f['problem_id']].append(f.get('stage', 'NEUTRAL'))
# #     state_seqs = [p_map_s[p] for p in sorted(p_map_s) if len(p_map_s[p]) >= 3]
# #     label_seqs = [p_map_l[p] for p in sorted(p_map_s) if len(p_map_s[p]) >= 3]
# #     return state_seqs, label_seqs

# # # ── SLDS EM ───────────────────────────────────────────────────

# # def init_params(sequences, K, d):
# #     X_in = np.concatenate([s[:-1] for s in sequences])
# #     dX   = np.concatenate([np.diff(s, axis=0) for s in sequences])
# #     lbl  = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(dX)
# #     M = np.zeros((K,d,d)); b = np.zeros((K,d)); Cov = np.array([np.eye(d)]*K)
# #     for k in range(K):
# #         mask = lbl == k
# #         if mask.sum() < d+2: M[k] = 0.1*np.eye(d); continue
# #         W, *_ = np.linalg.lstsq(np.hstack([X_in[mask], np.ones((mask.sum(),1))]),
# #                                  dX[mask], rcond=None)
# #         M[k], b[k] = W[:d].T, W[d]
# #         res = dX[mask] - (X_in[mask] @ M[k].T + b[k])
# #         Cov[k] = np.cov(res.T) + 1e-3*np.eye(d)
# #     return np.ones(K)/K, np.eye(K)*0.7+0.3/K, M, b, Cov

# # def get_log_emissions(seq, K, M, b, Cov):
# #     T, d = seq.shape
# #     dX = np.diff(seq, axis=0)
# #     log_emit = np.full((T, K), -1e9)
# #     for k in range(K):
# #         _, logdet = np.linalg.slogdet(Cov[k])
# #         inv_cov   = np.linalg.inv(Cov[k])
# #         diff      = dX - (seq[:-1] @ M[k].T + b[k])
# #         mahal     = np.sum((diff @ inv_cov) * diff, axis=1)
# #         log_emit[1:,k] = -0.5*(d*np.log(2*np.pi) + logdet + mahal)
# #     log_emit[0] = 0.0
# #     return log_emit

# # def forward_backward(seq, pi, A, M, b, Cov, K):
# #     T = len(seq)
# #     log_emit = get_log_emissions(seq, K, M, b, Cov)
# #     log_A, log_pi = np.log(A+1e-12), np.log(pi+1e-12)
# #     la = np.zeros((T,K)); la[0] = log_pi + log_emit[0]
# #     for t in range(1,T):
# #         la[t] = log_emit[t] + np.logaddexp.reduce(la[t-1][:,None]+log_A, axis=0)
# #     lb = np.zeros((T,K))
# #     for t in range(T-2,-1,-1):
# #         lb[t] = np.logaddexp.reduce(log_A + log_emit[t+1] + lb[t+1], axis=1)
# #     lg = la+lb; lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
# #     lxi = np.zeros((T-1,K,K))
# #     for t in range(T-1):
# #         lxi[t] = la[t][:,None] + log_A + log_emit[t+1] + lb[t+1]
# #         lxi[t] -= np.logaddexp.reduce(lxi[t].ravel())
# #     return np.exp(lg), np.exp(lxi)

# # def m_step(sequences, gammas, xis, K, d):
# #     xi_sum = sum(xi.sum(0) for xi in xis) + np.eye(K)*KAPPA + 1e-8
# #     A_new  = xi_sum / xi_sum.sum(axis=1, keepdims=True)
# #     pi_new = np.maximum(np.mean([g[0] for g in gammas], axis=0), 1e-8)
# #     pi_new /= pi_new.sum()
# #     M_new = np.zeros((K,d,d)); b_new = np.zeros((K,d)); Cov_new = np.zeros((K,d,d))
# #     for k in range(K):
# #         W_sum, WY_sum = np.zeros((d+1,d+1)), np.zeros((d+1,d))
# #         for seq, gamma in zip(sequences, gammas):
# #             dX    = np.diff(seq, axis=0)
# #             X_aug = np.hstack([seq[:-1], np.ones((len(seq)-1,1))])
# #             w     = gamma[1:,k]
# #             W_sum  += (X_aug*w[:,None]).T @ X_aug
# #             WY_sum += (X_aug*w[:,None]).T @ dX
# #         coef = np.linalg.solve(W_sum + 1e-4*np.eye(d+1), WY_sum)
# #         M_new[k], b_new[k] = coef[:d].T, coef[d]
# #         num, den = np.zeros((d,d)), 1e-9
# #         for seq, gamma in zip(sequences, gammas):
# #             dX  = np.diff(seq, axis=0)
# #             err = dX - (seq[:-1] @ M_new[k].T + b_new[k])
# #             w   = gamma[1:,k]
# #             num += (err*w[:,None]).T @ err; den += w.sum()
# #         Cov_new[k] = num/den + 1e-4*np.eye(d)
# #     return pi_new, A_new, M_new, b_new, Cov_new

# # def run_slds(sequences, labels, K):
# #     d = sequences[0].shape[1]
# #     pi, A, M, b, Cov = init_params(sequences, K, d)
# #     for _ in range(N_ITERS_SLDS):
# #         gammas, xis = [], []
# #         for seq in sequences:
# #             g, x = forward_backward(seq, pi, A, M, b, Cov, K)
# #             gammas.append(g); xis.append(x)
# #         pi, A, M, b, Cov = m_step(sequences, gammas, xis, K, d)
# #     state_seqs = [np.argmax(g, axis=1) for g in gammas]
# #     return compute_metrics(state_seqs, labels, K, A)

# # # ── METRICS ──────────────────────────────────────────────────

# # def compute_metrics(state_seqs, label_seqs, K, A):
# #     persist    = np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs])
# #     self_trans = np.mean(np.diag(A))
# #     confusion  = np.zeros((K, len(STAGES)))
# #     for s_seq, l_seq in zip(state_seqs, label_seqs):
# #         for s, l in zip(s_seq, l_seq):
# #             if l in STAGES: confusion[s, STAGES.index(l)] += 1
# #     fwd = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-9)  # mode→stage, rows sum to 1
# #     rev = confusion / (confusion.sum(axis=0, keepdims=True) + 1e-9)  # stage→mode, cols sum to 1
# #     spec = np.mean(np.max(fwd, axis=1))
# #     return persist, self_trans, spec, A, fwd, rev

# # def moe_metrics(states, all_features, K):
# #     state_seqs, label_seqs = make_state_sequences(states, all_features)
# #     A = np.zeros((K,K)) + 1e-8
# #     for seq in state_seqs:
# #         for t in range(len(seq)-1):
# #             A[seq[t], seq[t+1]] += 1
# #     A /= A.sum(axis=1, keepdims=True)
# #     return compute_metrics(state_seqs, label_seqs, K, A)

# # def print_analysis(fwd, rev, K):
# #     header = " ".join([f"{s[:5]:>7}" for s in STAGES])
# #     print(f"\n  Mode→Stage (rows sum to 1 — what's inside each mode):", flush=True)
# #     print(f"  {'':>10} {header}", flush=True)
# #     for i in range(K):
# #         row = " ".join([f"{fwd[i,j]:7.2f}" for j in range(len(STAGES))])
# #         print(f"    Mode {i:<3}: {row}", flush=True)
# #     print(f"\n  Stage→Mode (cols sum to 1 — where does each stage live):", flush=True)
# #     print(f"  {'':>10} {header}", flush=True)
# #     for i in range(K):
# #         row = " ".join([f"{rev[i,j]:7.2f}" for j in range(len(STAGES))])
# #         print(f"    Mode {i:<3}: {row}", flush=True)
# #     print(f"\n  Dominant stage per mode:", flush=True)
# #     for i in range(K):
# #         idx = np.argmax(fwd[i])
# #         print(f"    Mode {i}: {STAGES[idx]} ({fwd[i,idx]:.1%})", flush=True)

# # # ── MAIN ─────────────────────────────────────────────────────

# # if __name__ == "__main__":
# #     path = "/home/abir19/scratch/abir19/new_rpc_math500_layer_20_qwen1_5/all_sentences_features.pkl"

# #     print("Loading & balancing data...", flush=True)
# #     all_f, triplets = load_and_prepare(path)

# #     # ── CEBRA + SLDS ─────────────────────────────────────────
# #     print(f"\n{'='*60}\nCEBRA + SLDS\n{'='*60}", flush=True)
# #     Z = train_cebra(all_f, triplets)
# #     seqs, labels = make_sequences(Z, all_f)
# #     print(f"\n{'K':<4} | {'Persist':<10} | {'Self-Trans':<10} | {'Spec'}", flush=True)
# #     for k in K_SWEEP:
# #         p, st, s, A_mat, F_mat, R_mat = run_slds(seqs, labels, k)
# #         print(f"{k:<4} | {p:<10.2f} | {st:<10.3f} | {s:.4f}", flush=True)
# #         if k == 4:
# #             print_analysis(F_mat, R_mat, k)

# #     # ── CEBRA-MoE K-sweep ────────────────────────────────────
# #     print(f"\n{'='*60}\nCEBRA-MoE K-sweep\n{'='*60}", flush=True)
# #     print(f"\n{'K':<4} | {'Persist':<10} | {'Self-Trans':<10} | {'Spec'}", flush=True)
# #     for k in K_SWEEP:
# #         print(f"  Training MoE K={k}...", flush=True)
# #         _, states = train_cebra_moe(all_f, triplets, K=k)
# #         p, st, s, A_mat, F_mat, R_mat = moe_metrics(states, all_f, k)
# #         print(f"{k:<4} | {p:<10.2f} | {st:<10.3f} | {s:.4f}", flush=True)
# #         if k == 4:
# #             print_analysis(F_mat, R_mat, k)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import pickle
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import json
# from collections import defaultdict
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix

# try:
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     from transformer_lens import HookedTransformer
# except ImportError:
#     print("Warning: transformers/transformer_lens not installed.")
#     AutoTokenizer = AutoModelForCausalLM = HookedTransformer = None

# device = "cuda" if torch.cuda.is_available() else "cpu"
# checkpoint_save = "rpc_final_pipeline"
# checkpoint_dir = "/home/abir19/scratch/abir19/gsm8k_qwen14b_trajs"
# os.makedirs(checkpoint_save, exist_ok=True)

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


# class CEBRA_MoE_Encoder(nn.Module):
#     def __init__(self, d_in, d_h, K):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(d_in, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, d_h))
#         self.gate = nn.Sequential(
#             nn.Linear(d_h, 128), nn.LayerNorm(128), nn.Softplus(), nn.Linear(128, K, bias=False))
#     def forward(self, x, temp=1.0):
#         h = F.normalize(self.encoder(x), dim=1)
#         s = F.gumbel_softmax(self.gate(h), tau=temp, hard=False)
#         return h, s, self.gate(h)


# class DynamicsMoE(nn.Module):
#     def __init__(self, K, dim):
#         super().__init__()
#         self.experts = nn.ModuleList([
#             nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, dim))
#             for _ in range(K)])
#     def forward(self, h, s):
#         return (torch.stack([m(h) for m in self.experts], dim=1) * s.unsqueeze(-1)).sum(1)


# def nce_loss(z, p, n, temp=0.05):
#     logits = torch.cat([torch.sum(z*p,1,keepdim=True), torch.sum(z*n,1,keepdim=True)],1) / temp
#     return F.cross_entropy(logits, torch.zeros(len(z), dtype=torch.long, device=z.device))


# def compute_transition_matrix(state_seqs, K):
#     T = np.zeros((K, K))
#     for seq in state_seqs:
#         for a, b in zip(seq[:-1], seq[1:]): T[a, b] += 1
#     row_sums = T.sum(1, keepdims=True)
#     return T / np.where(row_sums == 0, 1, row_sums)


# def print_transition_matrix(T, K):
#     header = "       " + "  ".join(f"→{j}" for j in range(K))
#     print(f"  {header}")
#     for i in range(K):
#         row = "  ".join(f"{T[i,j]:.2f}" for j in range(K))
#         print(f"  {i}  [ {row} ]  self={T[i,i]:.2f}")
#     diag = np.diag(T)
#     print(f"  mean self-transition: {diag.mean():.3f}  min: {diag.min():.3f}  max: {diag.max():.3f}")


# def train_and_eval_k(K_val, features, X_torch, triplets, epochs=50):
#     print(f"\n--- K={K_val} ---", flush=True)
#     d_h = 32
#     model = CEBRA_MoE_Encoder(X_torch.shape[1], d_h, K_val).to(device)
#     dyn   = DynamicsMoE(K_val, d_h).to(device)
#     opt   = optim.AdamW(list(model.parameters()) + list(dyn.parameters()), lr=1e-3)

#     for epoch in range(epochs):
#         tau      = max(0.2, 1.5 * (0.92 ** epoch))
#         w_div    = min(30.0, (epoch / 15.0) * 30.0)
#         indices  = np.random.permutation(len(triplets))
#         for b in range(0, len(triplets), 128):
#             b_idx = indices[b:b+128]
#             i_t = torch.tensor([triplets[x][0] for x in b_idx], device=device)
#             p_t = torch.tensor([triplets[x][1] for x in b_idx], device=device)
#             n_t = torch.tensor([triplets[x][2] for x in b_idx], device=device)
#             h_i, s_i, _ = model(X_torch[i_t], temp=tau)
#             h_p, s_p, _ = model(X_torch[p_t], temp=tau)
#             h_n, _,   _ = model(X_torch[n_t], temp=tau)
#             h_pred = dyn(h_i, s_i)
#             l_nce  = nce_loss(h_pred, h_p, h_n)
#             l_mse  = F.mse_loss(h_pred, h_p)
#             l_div  = (s_i.mean(0) * torch.log(s_i.mean(0) + 1e-8)).sum()
#             l_pers = torch.abs(s_i - s_p).mean()
#             loss   = l_nce + 10.0*l_mse + w_div*l_div + 1.0*l_pers
#             opt.zero_grad(); loss.backward(); opt.step()

#     model.eval()
#     with torch.no_grad():
#         _, s_final, _ = model(X_torch, temp=0.01)
#         states = s_final.argmax(1).cpu().numpy()

#     # per-problem state sequences sorted by sentence_idx
#     p_map = defaultdict(list)
#     for i, f in enumerate(features):
#         p_map[f['problem_id']].append((f.get('sentence_idx', i), states[i]))
#     state_seqs = [[s for _, s in sorted(v)] for v in p_map.values() if len(v) > 1]

#     persistence = np.mean([
#         1.0 - np.sum(np.array(s[1:]) != np.array(s[:-1])) / (len(s)-1)
#         for s in state_seqs])

#     T_matrix = compute_transition_matrix(state_seqs, K_val)

#     print(f"  Persistence: {persistence:.2%}")
#     print(f"  Transition matrix:")
#     print_transition_matrix(T_matrix, K_val)

#     return persistence, l_mse.item(), T_matrix, model, dyn, states


# def apply_logit_lens(centroid, W_U, tokenizer, top_k=10):
#     if W_U is None or tokenizer is None: return []
#     centroid = centroid.to(W_U.device).to(W_U.dtype).unsqueeze(0)
#     logits   = centroid @ W_U.t()
#     top_logits, top_indices = torch.topk(logits[0], top_k)
#     probs    = F.softmax(logits[0], dim=-1)
#     return [(tokenizer.decode([idx.item()]).strip(), l.item(), probs[idx].item())
#             for l, idx in zip(top_logits, top_indices)]


# def build_semantic_alignment_matrix(features, states, stage_key='stage'):
#     print("\n" + "="*60 + "\nSEMANTIC ALIGNMENT MATRIX\n" + "="*60)
#     if stage_key not in features[0]:
#         print(f"Warning: No '{stage_key}' field. Skipping."); return None
#     human_stages = [f[stage_key] for f in features]
#     alignment_df   = pd.crosstab(pd.Series(states, name='Regime'), pd.Series(human_stages, name='Stage'), margins=True)
#     alignment_norm = pd.crosstab(pd.Series(states, name='Regime'), pd.Series(human_stages, name='Stage'), normalize='index')
#     print(alignment_df)
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#     sns.heatmap(alignment_df.iloc[:-1,:-1], annot=True, fmt='d', cmap='Blues', ax=axes[0])
#     sns.heatmap(alignment_norm, annot=True, fmt='.2%', cmap='YlOrRd', ax=axes[1])
#     axes[0].set_title('Count'); axes[1].set_title('Specialization')
#     plt.tight_layout()
#     plt.savefig(f"{checkpoint_save}/alignment_matrix.png", dpi=150, bbox_inches='tight')
#     plt.close()
#     alignment_df.to_csv(f"{checkpoint_save}/alignment_matrix.csv")
#     alignment_norm.to_csv(f"{checkpoint_save}/alignment_matrix_normalized.csv")
#     return alignment_df, alignment_norm


# def analyze_regime_semantics(model, X_torch, states, W_U=None, tokenizer=None):
#     print("\n" + "="*60 + "\nSEMANTIC SIGNATURE ANALYSIS\n" + "="*60)
#     regime_semantics = {}
#     for regime_id in sorted(np.unique(states)):
#         idx = np.where(states == regime_id)[0]
#         print(f"\n--- Regime {regime_id} ({len(idx)} samples) ---")
#         centroid = X_torch[idx].mean(0)
#         regime_semantics[regime_id] = {}
#         if W_U is not None and tokenizer is not None:
#             tokens = apply_logit_lens(centroid, W_U, tokenizer)
#             regime_semantics[regime_id]['tokens'] = tokens
#             print("  Top tokens:", [(t, f"{p:.4f}") for t, _, p in tokens])
#         regime_semantics[regime_id]['stats'] = {
#             'mean': X_torch[idx].mean().item(),
#             'std':  X_torch[idx].std().item(),
#             'n':    len(idx)}
#     with open(f"{checkpoint_save}/regime_semantics.json", 'w') as f:
#         json.dump({str(k): {
#             'tokens': [(t, float(l), float(p)) for t, l, p in v.get('tokens', [])],
#             'stats': v.get('stats', {})} for k, v in regime_semantics.items()}, f, indent=2)
#     return regime_semantics


# def visualize_persistence_cliff(k_values, persistence_scores):
#     diffs = np.diff(persistence_scores)
#     cliff_idx = np.argmax(np.abs(diffs))
#     cliff_k   = k_values[cliff_idx]
#     print(f"\nPersistence cliff at K={cliff_k}: {persistence_scores[cliff_idx]:.2%} → {persistence_scores[cliff_idx+1]:.2%}")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(k_values, persistence_scores, marker='o', linewidth=2, color='darkblue')
#     ax.fill_between(k_values, persistence_scores, alpha=0.3, color='lightblue')
#     ax.axvline(x=cliff_k, color='red', linestyle='--', alpha=0.7, label=f'Cliff at K={cliff_k}')
#     ax.set_xlabel('K'); ax.set_ylabel('Persistence'); ax.set_title('Persistence Cliff')
#     ax.legend(); ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f"{checkpoint_save}/persistence_cliff.png", dpi=150)
#     plt.close()


# def perform_causal_intervention(tl_model, tokenizer, regime_centroid, prompt,
#                                 layer_idx=None, intervention_type='injection',
#                                 mode='hard', alpha=0.5, max_new_tokens=20):
#     if layer_idx is None: layer_idx = len(tl_model.blocks) - 1
#     print(f"\n--- Causal {intervention_type} ({mode}, α={alpha}) @ layer {layer_idx} ---")
#     print(f"Prompt: '{prompt}'")
#     residual_dim = tl_model.cfg.d_model
#     centroid_dim = regime_centroid.shape[0]
#     if centroid_dim != residual_dim:
#         if centroid_dim < residual_dim:
#             regime_centroid = torch.cat([regime_centroid,
#                 torch.zeros(residual_dim-centroid_dim, device=regime_centroid.device, dtype=regime_centroid.dtype)])
#         else:
#             regime_centroid = regime_centroid[:residual_dim]
#     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

#     def hook_fn(module, input, output):
#         out = output.clone()
#         c   = regime_centroid.to(out.device).to(out.dtype)
#         if mode == 'hard':
#             out[0,-1,:] = c if intervention_type == 'injection' else -c
#         else:
#             out[0,-1,:] += (alpha if intervention_type == 'injection' else -alpha) * c
#         return out

#     with torch.no_grad():
#         base_out  = tl_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
#         base_text = tokenizer.decode(base_out[0][input_ids.shape[1]:], skip_special_tokens=True)
#     handle = tl_model.blocks[layer_idx].hook_resid_post.register_forward_hook(hook_fn)
#     with torch.no_grad():
#         int_out  = tl_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
#         int_text = tokenizer.decode(int_out[0][input_ids.shape[1]:], skip_special_tokens=True)
#     handle.remove()
#     print(f"  Baseline:     {base_text}")
#     print(f"  Intervention: {int_text}")
#     return {'baseline': base_text, 'intervention': int_text}


# # ── MAIN ─────────────────────────────────────────────────────

# print("Loading data...")
# all_features, triplets = load_and_balance_data(f"{checkpoint_dir}/all_sentences_features.pkl")
# X_raw   = np.array([f['hidden_state'] for f in all_features])
# X_torch = torch.from_numpy(StandardScaler().fit_transform(X_raw)).float().to(device)

# print("Running K-sweep...")
# k_values = list(range(2, 20, 2))
# results  = []
# best_k = best_model = best_dyn = best_states = None

# for k in k_values:
#     persistence, final_mse, T_matrix, model, dyn, states = train_and_eval_k(
#         k, all_features, X_torch, triplets)
#     results.append((k, persistence, final_mse, T_matrix))
#     print(f"K={k}: Persistence={persistence:.2%}, MSE={final_mse:.6f}", flush=True)
#     print(f"Transition matrix for K={k}:\n{T_matrix}\n", flush=True)
#     if best_k is None or (persistence > 0.5 and final_mse < 0.1):
#         best_k, best_model, best_dyn, best_states = k, model, dyn, states

# k_list, p_list, m_list, t_list = zip(*results)

# fig, ax1 = plt.subplots()
# ax1.set_xlabel('K'); ax1.set_ylabel('Persistence', color='tab:blue')
# ax1.plot(k_list, p_list, marker='o', color='tab:blue')
# ax2 = ax1.twinx(); ax2.set_ylabel('MSE', color='tab:red')
# ax2.plot(k_list, m_list, marker='s', color='tab:red')
# plt.title("K-Sweep"); plt.savefig(f"{checkpoint_save}/k_sweep.png"); plt.close()

# print(f"\n✓ Best K: {best_k}")
# visualize_persistence_cliff(list(k_list), list(p_list))


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

# ── Config ────────────────────────────────────────────────────
PATH         = "/home/abir19/scratch/abir19/gsm8k_qwen14b_trajs"
HIDDEN_KEY   = "hidden_state_last"   # try "hidden_state" for mean-pool

K_EVAL       = [1, 4, 7]
PCA_RANKS    = [10, 40]      # skip 5 and 20, keep endpoints
CEBRA_D      = 40
N_ITERS_SLDS = 30            # converges well before 50
CEBRA_EPOCHS = 75
MLP_EPOCHS   = 100           # loss plateaus early
BATCH_SIZE   = 2048
KAPPA        = 10.0

STAGES = [
    "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
]

# ── DATA ─────────────────────────────────────────────────────

def load_data(path, hidden_key=HIDDEN_KEY):
    if os.path.isdir(path):
        pkls = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pkl')]
        assert pkls, f"No pkl files in {path}"
        path = sorted(pkls)[0]
        print(f"  Using: {path}", flush=True)

    with open(path, 'rb') as f:
        data = pickle.load(f)

    # check available keys
    sample_keys = set(data[0].keys())
    print(f"  Keys in dataset: {sample_keys}", flush=True)
    if hidden_key not in sample_keys:
        fallback = next(k for k in sample_keys if 'hidden' in k.lower())
        print(f"  '{hidden_key}' not found, using '{fallback}'", flush=True)
        hidden_key = fallback

    p_map = defaultdict(list)
    for f in data:
        p_map[f['problem_id']].append(f)

    sequences, labels, all_features = [], [], []
    for pid in sorted(p_map):
        feats = p_map[pid]
        if 'sentence_idx' in feats[0]:
            feats = sorted(feats, key=lambda x: x['sentence_idx'])
        if len(feats) < 3: continue
        sequences.append(np.array([f[hidden_key] for f in feats]))
        labels.append([f.get('stage', 'NEUTRAL') for f in feats])
        all_features.extend(feats)

    return sequences, labels, all_features, hidden_key

def dataset_diagnostics(sequences, labels):
    lengths = [len(s) for s in sequences]
    print(f"\n── Dataset ──")
    print(f"  sequences   : {len(sequences)}")
    print(f"  seq lengths : min={min(lengths)} mean={np.mean(lengths):.1f} "
          f"median={np.median(lengths):.0f} max={max(lengths)}")
    print(f"  total steps : {sum(lengths)}")
    print(f"  transitions : {sum(lengths) - len(lengths)}")
    print(f"  hidden dim  : {sequences[0].shape[1]}")
    stage_counts = defaultdict(int)
    for l in labels:
        for s in l: stage_counts[s] += 1
    total = sum(stage_counts.values())
    print(f"  stages:")
    for s, c in sorted(stage_counts.items(), key=lambda x: -x[1]):
        print(f"    {s:<35} {c:>6}  ({c/total:.1%})")

def train_test_split_seqs(sequences, labels, ratio=0.8):
    n = len(sequences)
    idx = np.random.permutation(n)
    cut = int(ratio * n)
    tr, te = idx[:cut], idx[cut:]
    return ([sequences[i] for i in tr], [labels[i] for i in tr],
            [sequences[i] for i in te], [labels[i] for i in te])

# ── BASELINES ────────────────────────────────────────────────

def fit_linear_ar(seqs):
    X_in  = np.concatenate([s[:-1] for s in seqs])
    X_out = np.concatenate([s[1:]  for s in seqs])
    W, *_ = np.linalg.lstsq(np.hstack([X_in, np.ones((len(X_in),1))]),
                             X_out, rcond=None)
    return W[:-1].T, W[-1]

def predict_linear_ar(seq, M, b):
    p = np.zeros_like(seq); p[0] = seq[0]
    for t in range(len(seq)-1): p[t+1] = seq[t] @ M.T + b
    return p

class ARNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,256), nn.GELU(),
                                 nn.Linear(256,256), nn.GELU(),
                                 nn.Linear(256,d))
    def forward(self, x): return self.net(x)

def fit_mlp_ar(seqs, device):
    X_in  = torch.tensor(np.concatenate([s[:-1] for s in seqs]), dtype=torch.float32).to(device)
    X_out = torch.tensor(np.concatenate([s[1:]  for s in seqs]), dtype=torch.float32).to(device)
    model = ARNet(X_in.shape[1]).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(MLP_EPOCHS):
        idx = torch.randperm(len(X_in))
        for i in range(0, len(X_in), BATCH_SIZE):
            b = idx[i:i+BATCH_SIZE]
            loss = F.mse_loss(model(X_in[b]), X_out[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 50 == 0:
            print(f"    MLP epoch {ep+1}/{MLP_EPOCHS} | loss={loss.item():.4f}", flush=True)
    model.eval()
    return model

def predict_mlp(seq, model, device):
    p = np.zeros_like(seq); p[0] = seq[0]
    with torch.no_grad():
        p[1:] = model(torch.tensor(seq[:-1], dtype=torch.float32).to(device)).cpu().numpy()
    return p

def r2_score(seqs, preds):
    y_true = np.concatenate([s[1:] for s in seqs])
    y_pred = np.concatenate([p[1:] for p in preds])
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean(0))**2)
    return 1 - ss_res / ss_tot

# ── SLDS ─────────────────────────────────────────────────────

def init_params(seqs, K, d):
    X_in  = np.concatenate([s[:-1] for s in seqs])
    dX    = np.concatenate([np.diff(s, axis=0) for s in seqs])
    lbl   = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(dX)
    M = np.zeros((K,d,d)); b = np.zeros((K,d)); Cov = np.array([np.eye(d)]*K)
    for k in range(K):
        mask = lbl == k
        if mask.sum() < d+2: M[k] = 0.1*np.eye(d); continue
        W, *_ = np.linalg.lstsq(np.hstack([X_in[mask], np.ones((mask.sum(),1))]),
                                 dX[mask], rcond=None)
        M[k], b[k] = W[:d].T, W[d]
        res = dX[mask] - (X_in[mask] @ M[k].T + b[k])
        Cov[k] = np.cov(res.T) + 1e-2*np.eye(d)
    return np.ones(K)/K, np.eye(K)*0.7+0.3/K, M, b, Cov

def get_log_emissions(seq, K, M, b, Cov):
    T, d  = seq.shape
    dX    = np.diff(seq, axis=0)
    log_e = np.full((T, K), -1e9)
    for k in range(K):
        _, logdet = np.linalg.slogdet(Cov[k])
        inv       = np.linalg.inv(Cov[k])
        diff      = dX - (seq[:-1] @ M[k].T + b[k])
        mahal     = np.sum((diff @ inv) * diff, axis=1)
        log_e[1:,k] = -0.5*(d*np.log(2*np.pi) + logdet + mahal)
    log_e[0] = 0.0
    return log_e

def forward_backward(seq, pi, A, M, b, Cov, K):
    T = len(seq)
    le = get_log_emissions(seq, K, M, b, Cov)
    lA, lpi = np.log(A+1e-12), np.log(pi+1e-12)
    la = np.zeros((T,K)); la[0] = lpi + le[0]
    for t in range(1,T):
        la[t] = le[t] + np.logaddexp.reduce(la[t-1][:,None]+lA, axis=0)
    lb = np.zeros((T,K))
    for t in range(T-2,-1,-1):
        lb[t] = np.logaddexp.reduce(lA + le[t+1] + lb[t+1], axis=1)
    lg = la+lb; lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
    lxi = np.zeros((T-1,K,K))
    for t in range(T-1):
        lxi[t] = la[t][:,None] + lA + le[t+1] + lb[t+1]
        lxi[t] -= np.logaddexp.reduce(lxi[t].ravel())
    return np.exp(lg), np.exp(lxi)

def m_step(seqs, gammas, xis, K, d):
    xi_sum = sum(xi.sum(0) for xi in xis) + np.eye(K)*KAPPA + 1e-8
    A_new  = xi_sum / xi_sum.sum(1, keepdims=True)
    pi_new = np.maximum(np.mean([g[0] for g in gammas], axis=0), 1e-8)
    pi_new /= pi_new.sum()
    M_new = np.zeros((K,d,d)); b_new = np.zeros((K,d)); Cov_new = np.zeros((K,d,d))
    for k in range(K):
        Ws, WY = np.zeros((d+1,d+1)), np.zeros((d+1,d))
        for seq, gamma in zip(seqs, gammas):
            dX    = np.diff(seq, axis=0)
            X_aug = np.hstack([seq[:-1], np.ones((len(seq)-1,1))])
            w     = gamma[1:,k]
            Ws  += (X_aug*w[:,None]).T @ X_aug
            WY  += (X_aug*w[:,None]).T @ dX
        coef = np.linalg.solve(Ws + 1e-4*np.eye(d+1), WY)
        M_new[k], b_new[k] = coef[:d].T, coef[d]
        num, den = np.zeros((d,d)), 1e-9
        for seq, gamma in zip(seqs, gammas):
            dX  = np.diff(seq, axis=0)
            err = dX - (seq[:-1] @ M_new[k].T + b_new[k])
            w   = gamma[1:,k]
            num += (err*w[:,None]).T @ err; den += w.sum()
        Cov_new[k] = num/den + 1e-2*np.eye(d)
    return pi_new, A_new, M_new, b_new, Cov_new

def fit_slds(seqs, K):
    d = seqs[0].shape[1]
    pi, A, M, b, Cov = init_params(seqs, K, d)
    for it in range(N_ITERS_SLDS):
        gammas, xis = [], []
        for seq in seqs:
            g, x = forward_backward(seq, pi, A, M, b, Cov, K)
            gammas.append(g); xis.append(x)
        pi, A, M, b, Cov = m_step(seqs, gammas, xis, K, d)
        if (it+1) % 10 == 0:
            usage = np.mean([g.mean(0) for g in gammas], axis=0)
            print(f"    EM iter {it+1} | usage={np.round(usage,2)}", flush=True)
    return pi, A, M, b, Cov, gammas

def slds_predict(seq, pi, A, M, b, Cov, K):
    gamma, _ = forward_backward(seq, pi, A, M, b, Cov, K)
    p = np.zeros_like(seq); p[0] = seq[0]
    for t in range(len(seq)-1):
        p[t+1] = seq[t] + sum(gamma[t,k] * (seq[t] @ M[k].T + b[k]) for k in range(K))
    return p

def persist_and_spec(gammas, labels, K):
    state_seqs = [np.argmax(g,1) for g in gammas]
    persist    = np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs])
    confusion  = np.zeros((K, len(STAGES)))
    for s_seq, l_seq in zip(state_seqs, labels):
        for s, l in zip(s_seq, l_seq):
            if l in STAGES: confusion[s, STAGES.index(l)] += 1
    fwd  = confusion / (confusion.sum(1, keepdims=True) + 1e-9)
    spec = np.mean(np.max(fwd, axis=1))
    return persist, spec, fwd

# ── CEBRA ────────────────────────────────────────────────────

class CEBRANet(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in,512), nn.GELU(),
                                 nn.Linear(512,256), nn.GELU(),
                                 nn.Linear(256,d_out))
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

def train_cebra(all_features, d_out=CEBRA_D):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # inverse-freq stage weights + hard negatives
    stage_counts = defaultdict(int)
    for f in all_features: stage_counts[f.get('stage','NEUTRAL')] += 1
    total = sum(stage_counts.values())
    sw = {s: total/(len(stage_counts)*c) for s,c in stage_counts.items()}

    p_map = defaultdict(list)
    for i, f in enumerate(all_features): p_map[f['problem_id']].append(i)
    pids = list(p_map.keys())

    triplets = []
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: continue
        trans = list(range(len(idxs)-1))
        w = np.array([sw.get(all_features[idxs[t]].get('stage','NEUTRAL'),1.0) for t in trans])
        w /= w.sum()
        for t in np.random.choice(trans, size=min(len(trans),25), replace=False, p=w):
            anchor, pos = idxs[t], idxs[t+1]
            a_stage  = all_features[anchor].get('stage','NEUTRAL')
            hard     = [i for i in idxs if all_features[i].get('stage','NEUTRAL') != a_stage]
            neg      = np.random.choice(hard) if hard else \
                       np.random.choice(p_map[np.random.choice([p for p in pids if p!=pid])])
            triplets.append((anchor, pos, neg))

    hidden_key = next(k for k in all_features[0] if 'hidden' in k.lower())
    X_raw = np.array([f[hidden_key] for f in all_features])
    X_sc  = torch.tensor(StandardScaler().fit_transform(X_raw), dtype=torch.float32).to(device)
    model = CEBRANet(X_raw.shape[1], d_out).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    trips = np.array(triplets)

    for epoch in range(CEBRA_EPOCHS):
        idx = np.random.permutation(len(trips))
        for i in range(0, len(trips), BATCH_SIZE):
            b  = trips[idx[i:i+BATCH_SIZE]]
            za, zp, zn = model(X_sc[b[:,0]]), model(X_sc[b[:,1]]), model(X_sc[b[:,2]])
            pos = torch.sum(za*zp,1)/0.1; neg = torch.sum(za*zn,1)/0.1
            loss = -torch.log(torch.exp(pos)/(torch.exp(pos)+torch.exp(neg))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 25 == 0:
            print(f"    CEBRA epoch {epoch+1}/{CEBRA_EPOCHS} | loss={loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        Z = model(X_sc).cpu().numpy()

    p_map_z, p_map_l = defaultdict(list), defaultdict(list)
    for i, f in enumerate(all_features):
        p_map_z[f['problem_id']].append(Z[i])
        p_map_l[f['problem_id']].append(f.get('stage','NEUTRAL'))
    seqs   = [np.array(p_map_z[p]) for p in sorted(p_map_z) if len(p_map_z[p]) >= 3]
    labels = [p_map_l[p]           for p in sorted(p_map_z) if len(p_map_z[p]) >= 3]
    return seqs, labels

# ── EXPERIMENTS ──────────────────────────────────────────────

def print_transition_matrix(T, K):
    header = "      " + " ".join(f"  →{j}" for j in range(K))
    print(f"  {header}")
    for i in range(K):
        row = " ".join(f"{T[i,j]:5.2f}" for j in range(K))
        print(f"  {i} [ {row} ]  self={T[i,i]:.2f}")
    print(f"  mean self-trans: {np.diag(T).mean():.3f}")

def hard_transition_matrix(gammas, K):
    state_seqs = [np.argmax(g, 1) for g in gammas]
    T = np.zeros((K, K))
    for seq in state_seqs:
        for a, b in zip(seq[:-1], seq[1:]): T[a, b] += 1
    row_sums = T.sum(1, keepdims=True)
    return T / np.where(row_sums == 0, 1, row_sums)


    """SLDS vs linear AR on a given embedding space."""
    tr_seqs, tr_labs, te_seqs, te_labs = train_test_split_seqs(seqs, labels)

    M_ar, b_ar = fit_linear_ar(tr_seqs)
    ar_r2 = r2_score(te_seqs, [predict_linear_ar(s, M_ar, b_ar) for s in te_seqs])

    print(f"\n{'='*60}\n{tag}\n{'='*60}")
    print(f"  Linear AR R²: {ar_r2:.4f}")
    print(f"\n  {'K':<4} {'R²':<10} {'Δ linAR':<10} {'persist':<10} {'spec'}")
    print(f"  {'-'*45}")

    for K in K_EVAL:
        print(f"\n  Fitting SLDS K={K}...", flush=True)
        pi, A, M, b, Cov, gammas = fit_slds(tr_seqs, K)
        slds_r2 = r2_score(te_seqs, [slds_predict(s, pi, A, M, b, Cov, K) for s in te_seqs])
        p, sp, fwd = persist_and_spec(gammas, tr_labs, K)
        print(f"  {K:<4} {slds_r2:<10.4f} {slds_r2-ar_r2:+.4f}    {p:<10.2f} {sp:.3f}")
        T = hard_transition_matrix(gammas, K)
        print(f"  Transition matrix (from hard state sequences):")
        print_transition_matrix(T, K)

        if K == 4:
            print(f"\n  Mode→Stage (K=4):")
            hdr = " ".join([f"{s[:5]:>7}" for s in STAGES])
            print(f"    {'':>8} {hdr}")
            for i in range(K):
                row = " ".join([f"{fwd[i,j]:7.2f}" for j in range(len(STAGES))])
                print(f"    Mode {i}: {row}  ← {STAGES[np.argmax(fwd[i])]}")

def pca_rank_sweep(seqs, labels):
    """Quick check: linear AR R² at each rank. We already know the gap exists
    from the first run — this just confirms which rank to use for SLDS."""
    tr_seqs, tr_labs, te_seqs, _ = train_test_split_seqs(seqs, labels)

    print(f"\n{'='*60}\nPCA RANK SWEEP — linear AR only (fast)\n{'='*60}")
    print(f"  {'D':<5} {'var%':<8} {'linAR R²'}")
    print(f"  {'-'*25}")

    for rank in PCA_RANKS:
        X_tr = np.concatenate(tr_seqs)
        sc = StandardScaler(); pca = PCA(n_components=rank)
        X_tr_sc = sc.fit_transform(X_tr)
        pca.fit(X_tr_sc)
        var     = pca.explained_variance_ratio_.sum()
        tr_proj = [pca.transform(sc.transform(s)) for s in tr_seqs]
        te_proj = [pca.transform(sc.transform(s)) for s in te_seqs]
        M_ar, b_ar = fit_linear_ar(tr_proj)
        ar_r2 = r2_score(te_proj, [predict_linear_ar(s, M_ar, b_ar) for s in te_proj])
        print(f"  {rank:<5} {var:<8.1%} {ar_r2:.4f}", flush=True)

# ── CEBRA-MoE ────────────────────────────────────────────────

class MoEEncoder(nn.Module):
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
    def __init__(self, K, d_h):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_h, 128), nn.ReLU(), nn.Linear(128, d_h))
            for _ in range(K)])
    def forward(self, h, s):
        preds = torch.stack([m(h) for m in self.experts], dim=1)
        return torch.sum(s.unsqueeze(-1) * preds, dim=1)

def train_cebra_moe(all_features, K, d_h=32, epochs=75):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # inverse-freq stage weights + hard negatives (same as CEBRA)
    stage_counts = defaultdict(int)
    for f in all_features: stage_counts[f.get('stage','NEUTRAL')] += 1
    total = sum(stage_counts.values())
    sw = {s: total/(len(stage_counts)*c) for s,c in stage_counts.items()}

    p_map = defaultdict(list)
    for i, f in enumerate(all_features): p_map[f['problem_id']].append(i)
    pids = list(p_map.keys())

    triplets = []
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: continue
        trans = list(range(len(idxs)-1))
        w = np.array([sw.get(all_features[idxs[t]].get('stage','NEUTRAL'),1.0) for t in trans])
        w /= w.sum()
        for t in np.random.choice(trans, size=min(len(trans),20), replace=False, p=w):
            anchor, pos = idxs[t], idxs[t+1]
            a_stage = all_features[anchor].get('stage','NEUTRAL')
            hard    = [i for i in idxs if all_features[i].get('stage','NEUTRAL') != a_stage]
            neg     = np.random.choice(hard) if hard else \
                      np.random.choice(p_map[np.random.choice([p for p in pids if p!=pid])])
            triplets.append((anchor, pos, neg))

    hidden_key = next(k for k in all_features[0] if 'hidden' in k.lower())
    X_raw = np.array([f[hidden_key] for f in all_features])
    X_sc  = torch.tensor(StandardScaler().fit_transform(X_raw), dtype=torch.float32).to(device)

    model = MoEEncoder(X_raw.shape[1], d_h, K).to(device)
    dyn   = DynamicsMoE(K, d_h).to(device)
    opt   = optim.AdamW(list(model.parameters()) + list(dyn.parameters()), lr=1e-3)
    trips = np.array(triplets)

    for epoch in range(epochs):
        tau      = max(0.2, 1.5 * (0.92 ** epoch))
        w_div    = min(30.0, (epoch / 15.0) * 30.0)
        idx      = np.random.permutation(len(trips))
        for i in range(0, len(trips), BATCH_SIZE):
            b  = trips[idx[i:i+BATCH_SIZE]]
            it = torch.tensor(b[:,0], device=device)
            pt = torch.tensor(b[:,1], device=device)
            nt = torch.tensor(b[:,2], device=device)
            h_i, s_i = model(X_sc[it], temp=tau)
            h_p, _   = model(X_sc[pt], temp=tau)
            h_n, _   = model(X_sc[nt], temp=tau)
            h_pred   = dyn(h_i, s_i)
            _, s_p   = model(X_sc[pt], temp=tau)
            l_nce  = F.cross_entropy(
                torch.cat([torch.sum(h_pred*h_p,1,keepdim=True),
                           torch.sum(h_pred*h_n,1,keepdim=True)],1) / 0.05,
                torch.zeros(len(b), dtype=torch.long, device=device))
            l_mse  = F.mse_loss(h_pred, h_p)
            l_div  = (s_i.mean(0) * torch.log(s_i.mean(0)+1e-8)).sum()
            l_pers = torch.abs(s_i - s_p).mean()
            loss   = l_nce + 10.0*l_mse + w_div*l_div + 1.0*l_pers
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 25 == 0:
            print(f"    MoE epoch {epoch+1}/{epochs} | nce={l_nce.item():.4f} "
                  f"mse={l_mse.item():.4f} pers={l_pers.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        _, s_all = model(X_sc, temp=0.01)
        states   = s_all.argmax(1).cpu().numpy()

    # rebuild per-problem sequences of state assignments
    p_map_s, p_map_l = defaultdict(list), defaultdict(list)
    for i, f in enumerate(all_features):
        p_map_s[f['problem_id']].append((f.get('sentence_idx', i), states[i]))
        p_map_l[f['problem_id']].append((f.get('sentence_idx', i), f.get('stage','NEUTRAL')))

    state_seqs = []
    label_seqs = []
    for pid in sorted(p_map_s):
        state_seqs.append([s for _,s in sorted(p_map_s[pid])])
        label_seqs.append([l for _,l in sorted(p_map_l[pid])])

    return state_seqs, label_seqs

def guided_r2_on_pca(pca_seqs, assignment_seqs, labels, K, tr_idx, te_idx):
    """Fit K linear models on PCA space using assignment_seqs to partition transitions.
    Evaluate R² on held-out PCA sequences. This is the fair comparison metric."""
    tr_pca  = [pca_seqs[i]        for i in tr_idx]
    te_pca  = [pca_seqs[i]        for i in te_idx]
    tr_asgn = [assignment_seqs[i] for i in tr_idx]
    te_asgn = [assignment_seqs[i] for i in te_idx]
    tr_labs = [labels[i]          for i in tr_idx]

    d = tr_pca[0].shape[1]
    M_k = np.zeros((K,d,d)); b_k = np.zeros((K,d))
    for k in range(K):
        Ws, WY = np.zeros((d+1,d+1)), np.zeros((d+1,d))
        for seq_pca, asgn in zip(tr_pca, tr_asgn):
            dX    = np.diff(seq_pca, axis=0)
            X_aug = np.hstack([seq_pca[:-1], np.ones((len(seq_pca)-1,1))])
            w     = (np.array(asgn)[:-1] == k).astype(float)
            if w.sum() == 0: continue
            Ws += (X_aug*w[:,None]).T @ X_aug
            WY += (X_aug*w[:,None]).T @ dX
        coef = np.linalg.solve(Ws + 1e-4*np.eye(d+1), WY)
        M_k[k], b_k[k] = coef[:d].T, coef[d]

    preds = []
    for seq_pca, asgn in zip(te_pca, te_asgn):
        p = np.zeros_like(seq_pca); p[0] = seq_pca[0]
        for t in range(len(seq_pca)-1):
            p[t+1] = seq_pca[t] + seq_pca[t] @ M_k[asgn[t]].T + b_k[asgn[t]]
        preds.append(p)

    r2   = r2_score(te_pca, preds)
    pers = np.mean([len(s)/(np.count_nonzero(np.diff(s))+1)
                    for s in [np.array(a) for a in te_asgn]])

    # specialization
    confusion = np.zeros((K, len(STAGES)))
    for asgn, l_seq in zip(tr_asgn, tr_labs):
        for s, l in zip(asgn, l_seq):
            if l in STAGES: confusion[s, STAGES.index(l)] += 1
    fwd  = confusion / (confusion.sum(1, keepdims=True) + 1e-9)
    spec = np.mean(np.max(fwd, axis=1))
    return r2, pers, spec, fwd

# ── MAIN ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading data from {PATH}...", flush=True)
    sequences, labels, all_features, hidden_key = load_data(PATH)
    dataset_diagnostics(sequences, labels)

    # ── Exp 1: PCA rank sweep on raw representations ──────────
    # Tells us: is the nonlinear/switching structure accessible at any rank?
    pca_rank_sweep(sequences, labels)

    # ── Exp 2: SLDS directly on best PCA projection ───────────
    # Run SLDS at the rank where gap was largest (default D=40)
    print(f"\n{'='*60}\nSLDS ON RAW PCA (D=40)\n{'='*60}", flush=True)
    X_all = np.concatenate(sequences)
    sc = StandardScaler(); pca = PCA(n_components=40)
    X_sc = sc.fit_transform(X_all)
    pca.fit(X_sc)
    pca_seqs = [pca.transform(sc.transform(s)) for s in sequences]
    run_on_space(pca_seqs, labels, tag="SLDS on PCA D=40")

    # ── Exp 3: CEBRA state assignments → linear dynamics on PCA ──
    # CEBRA discovers regime assignments; we evaluate whether those
    # regimes have better linear dynamics in the original PCA space.
    # This is the only fair comparison: baselines also evaluated on PCA.
    print(f"\n{'='*60}\nCEBRA REGIMES → LINEAR DYNAMICS ON PCA D=40\n{'='*60}", flush=True)
    print("Training CEBRA...", flush=True)
    cebra_seqs, cebra_labels = train_cebra(all_features, d_out=CEBRA_D)

    # get soft CEBRA state assignments per sample
    # cebra_seqs are normalized embeddings — use argmax as hard assignments
    # then fit one linear model per regime on PCA space
    # use same indices for both pca and cebra splits
    n = len(pca_seqs)
    idx = np.random.permutation(n)
    cut = int(0.8 * n)
    tr_idx, te_idx = idx[:cut], idx[cut:]

    tr_seqs_pca   = [pca_seqs[i]    for i in tr_idx]
    te_seqs_pca   = [pca_seqs[i]    for i in te_idx]
    tr_seqs_cebra = [cebra_seqs[i]  for i in tr_idx]
    te_seqs_cebra = [cebra_seqs[i]  for i in te_idx]
    tr_labs       = [labels[i]       for i in tr_idx]

    M_ar, b_ar = fit_linear_ar(tr_seqs_pca)
    ar_r2 = r2_score(te_seqs_pca, [predict_linear_ar(s, M_ar, b_ar) for s in te_seqs_pca])
    print(f"  Linear AR on PCA R²: {ar_r2:.4f}  (baseline)")

    for K in K_EVAL[1:]:  # skip K=1, same as linear AR
        Z_tr = np.concatenate(tr_seqs_cebra)
        km   = KMeans(n_clusters=K, n_init=10, random_state=42).fit(Z_tr)

        # per-sequence hard assignments from CEBRA clusters (length T, not T-1)
        assignments_tr = [km.predict(s) for s in tr_seqs_cebra]

        # fit linear dynamics per regime on PCA transitions weighted by assignment
        d = tr_seqs_pca[0].shape[1]
        M_k = np.zeros((K,d,d)); b_k = np.zeros((K,d))
        for k in range(K):
            Ws, WY = np.zeros((d+1,d+1)), np.zeros((d+1,d))
            for seq_pca, asgn in zip(tr_seqs_pca, assignments_tr):
                dX    = np.diff(seq_pca, axis=0)          # T-1
                X_aug = np.hstack([seq_pca[:-1],
                                   np.ones((len(seq_pca)-1, 1))])  # T-1
                w     = (asgn[:-1] == k).astype(float)    # T-1, hard weight
                if w.sum() == 0: continue
                Ws  += (X_aug * w[:,None]).T @ X_aug
                WY  += (X_aug * w[:,None]).T @ dX
            coef    = np.linalg.solve(Ws + 1e-4*np.eye(d+1), WY)
            M_k[k], b_k[k] = coef[:d].T, coef[d]

        # predict on test using CEBRA assignments on test sequences
        preds = []
        for seq_pca, seq_cebra in zip(te_seqs_pca, te_seqs_cebra):
            asgn = km.predict(seq_cebra)
            p = np.zeros_like(seq_pca); p[0] = seq_pca[0]
            for t in range(len(seq_pca)-1):
                p[t+1] = seq_pca[t] + seq_pca[t] @ M_k[asgn[t]].T + b_k[asgn[t]]
            preds.append(p)

        cebra_r2 = r2_score(te_seqs_pca, preds)
        persist  = np.mean([len(s)/(np.count_nonzero(np.diff(s))+1)
                            for s in [km.predict(s) for s in te_seqs_cebra]])
        print(f"  K={K}  CEBRA-guided R²: {cebra_r2:.4f}  Δ vs linAR: {cebra_r2-ar_r2:+.4f}  persist: {persist:.2f}", flush=True)

    # ── Exp 4: CEBRA-MoE → linear dynamics on PCA ─────────────
    # MoE gate assignments used to partition transitions,
    # linear dynamics fitted and evaluated on PCA space.
    # Same tr_idx/te_idx as Exp 3 for direct comparability.
    print(f"\n{'='*60}\nCEBRA-MoE REGIMES → LINEAR DYNAMICS ON PCA D=40\n{'='*60}", flush=True)
    print(f"\n  {'K':<4} {'R² on PCA':<12} {'Δ linAR':<10} {'persist':<10} {'spec'}")
    print(f"  {'-'*50}")

    for K in K_EVAL[1:]:
        print(f"\n  Training CEBRA-MoE K={K}...", flush=True)
        moe_state_seqs, moe_label_seqs = train_cebra_moe(all_features, K=K)
        r2, pers, spec, fwd = guided_r2_on_pca(
            pca_seqs, moe_state_seqs, labels, K, tr_idx, te_idx)
        print(f"  {K:<4} {r2:<12.4f} {r2-ar_r2:+.4f}    {pers:<10.2f} {spec:.3f}")
        print(f"\n  Mode→Stage (K={K}):")
        hdr = " ".join([f"{s[:5]:>7}" for s in STAGES])
        print(f"    {'':>8} {hdr}")
        for i in range(K):
            row = " ".join([f"{fwd[i,j]:7.2f}" for j in range(len(STAGES))])
            print(f"    Mode {i}: {row}  ← {STAGES[np.argmax(fwd[i])]}")

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{'='*60}\nSUMMARY (all R² evaluated on PCA D=40)\n{'='*60}")
    print(f"  Linear AR (null)       R²={ar_r2:.4f}  persist={40.3:.1f}")
    print(f"  EM-SLDS K=4            R²=0.1907  persist=1.71")
    print(f"  CEBRA+KMeans K=4       R²=0.4223  persist=4.39")
    print(f"  CEBRA-MoE K=4         → see above")