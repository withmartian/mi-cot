import os
import json
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

D                = 40
PCA_DIM          = 40
CEBRA_EPOCHS     = 100
BATCH_SIZE       = 1024
N_ITERS_SLDS     = 50
TRANS_LR         = 1e-3
TRANS_GRAD_STEPS = 5
K_SWEEP          = range(2, 9)   # FIX #3: sweep K, pick by BIC per method
COV_REG          = 1e-2          # FIX #4: stronger regularization for stable B(f)

TEST_PATHS = [
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8b_reasoning/layer_22/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8b_reasoning/layer_31/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14b_reasoning/layer_28/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14b_reasoning/layer_47/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen1.5b_reasoning/layer_20/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen1.5b_reasoning/layer_27/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Llama_8B_reasoning/layer_22/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Llama_8B_reasoning/layer_31/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Qwen_1_5B_reasoning/layer_20/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Qwen_1_5B_reasoning/layer_27/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Qwen_14B_reasoning/layer_28/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Qwen_14B_reasoning/layer_47/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Llama_8B_reasoning/layer_22/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Llama_8B_reasoning/layer_31/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_1_5B_reasoning/layer_20/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_1_5B_reasoning/layer_27/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_reasoning/layer_28/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_reasoning/layer_47/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/llama8b/layer_22/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/llama8b/layer_31/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/qwen1.5b/layer_20/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/qwen1.5b/layer_27/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/qwen14b/layer_28/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/qwen14b/layer_47/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8B_base/layer_22/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8B_base/layer_31/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14B_base/layer_28/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14B_base/layer_47/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_1.5B_base/layer_20/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_1.5B_base/layer_27/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Llama_8B_base/layer_22/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Llama_8B_base/layer_31/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Qwen_1_5B_base/layer_20/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Qwen_1_5B_base/layer_27/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Qwen_14B_base/layer_28/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_svamp/Qwen_14B_base/layer_47/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Llama_8B_base/layer_22/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Llama_8B_base/layer_31/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_1_5B_base/layer_20/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_1_5B_base/layer_27/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_base/layer_28/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_base/layer_47/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/llama8b_base/layer_22/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/llama8b_base/layer_31/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/qwen1.5b_base/layer_20/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/qwen1.5b_base/layer_27/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/qwen14b_base/layer_28/all_sentences_features.pkl",
    "/home/abir19/scratch/abir19/SDS_train_mmlu-pro/qwen14b_base/layer_47/all_sentences_features.pkl",
]

OUTPUT_PATH = "/home/abir19/scratch/abir19/SDS_results/discriminability_check.json"


def load_and_prepare_cebra(path, limit_problems=500, max_triplets=25):
    with open(path, 'rb') as f:
        all_features = pickle.load(f)
    all_features = [f for f in all_features if f['problem_id'] < limit_problems]
    p_map = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map[f['problem_id']].append(i)
    triplets, pids = [], list(p_map.keys())
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: continue
        for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_triplets), replace=False):
            anchor, positive = idxs[t], idxs[t+1]
            negative = np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])
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
        # no L2 norm — preserves distributional spread needed for B(f_C)
        return self.net(x)


def train_cebra_projection(all_features, triplets, d_out=D):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_raw = np.array([f['hidden_state_last'] for f in all_features])
    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X_raw)
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca_scaler = StandardScaler()
    X_pca = pca_scaler.fit_transform(pca.fit_transform(X_scaled_np))
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
            loss = F.triplet_margin_loss(za, zp, zn, margin=1.0)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        Z = model(X_scaled).cpu().numpy()
    Z = Z / (Z.std(0, keepdims=True) + 1e-8)
    p_map_z, p_map_p = defaultdict(list), defaultdict(list)
    for i, f in enumerate(all_features):
        pid = f['problem_id']
        p_map_z[pid].append(Z[i])
        p_map_p[pid].append(X_pca[i])
    pids_sorted = sorted(p for p in p_map_z if len(p_map_z[p]) >= 3)
    return [np.array(p_map_z[p]) for p in pids_sorted], [np.array(p_map_p[p]) for p in pids_sorted]


class TransitionNet(nn.Module):
    def __init__(self, d_z, K):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(d_z + K, 64), nn.Tanh(), nn.Linear(64, K))

    def log_A_sequence(self, z_seq):
        T1 = len(z_seq)
        z  = torch.tensor(z_seq, dtype=torch.float32)
        z_rep = z.repeat(self.K, 1)
        s_oh  = torch.eye(self.K).repeat_interleave(T1, dim=0)
        with torch.no_grad():
            out = F.softmax(self.net(torch.cat([z_rep, s_oh], dim=1)), dim=1)
        return np.log(out.numpy().reshape(self.K, T1, self.K).transpose(1, 0, 2) + 1e-12)


def init_params(sequences, K, d):
    X_in  = np.vstack([s[:-1] for s in sequences])
    X_out = np.vstack([s[1:]  for s in sequences])
    labels = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X_out - X_in)
    dM, db, dCov = np.zeros((K,d,d)), np.zeros((K,d)), np.array([np.eye(d)]*K)
    for k in range(K):
        mask = labels == k
        if mask.sum() < d+2: dM[k] = np.eye(d); continue
        W, *_ = np.linalg.lstsq(np.hstack([X_in[mask], np.ones((mask.sum(),1))]), X_out[mask], rcond=None)
        dM[k], db[k] = W[:d].T, W[d]
        res = X_out[mask] - (X_in[mask] @ dM[k].T + db[k])
        dCov[k] = np.cov(res.T) + COV_REG * np.eye(d)  # FIX #4
    return np.ones(K)/K, dM, db, dCov


def get_log_emissions(seq, K, dM, db, dCov):
    T, d = seq.shape
    log_emit = np.zeros((T, K))
    for k in range(K):
        _, logdet = np.linalg.slogdet(dCov[k])
        inv_cov = np.linalg.inv(dCov[k])
        means = np.vstack([db[k], seq[:-1] @ dM[k].T + db[k]])
        diffs = seq - means
        log_emit[:, k] = -0.5*(d*np.log(2*np.pi) + logdet + np.sum((diffs @ inv_cov)*diffs, axis=1))
    return log_emit


def forward_backward(seq, pi, log_A_t, dM, db, dCov, K):
    T = len(seq)
    log_emit = get_log_emissions(seq, K, dM, db, dCov)
    log_pi   = np.log(pi + 1e-12)
    la = np.zeros((T, K)); la[0] = log_pi + log_emit[0]
    for t in range(1, T):
        la[t] = log_emit[t] + np.logaddexp.reduce(la[t-1][:,None] + log_A_t[t-1], axis=0)
    lb = np.zeros((T, K))
    for t in range(T-2, -1, -1):
        lb[t] = np.logaddexp.reduce(log_A_t[t] + (log_emit[t+1] + lb[t+1])[None, :], axis=1)
    lg = la + lb; lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
    lxi = np.zeros((T-1, K, K))
    for t in range(T-1):
        lxi[t] = la[t][:,None] + log_A_t[t] + (log_emit[t+1] + lb[t+1])[None, :]
        lxi[t] -= np.logaddexp.reduce(lxi[t].ravel())
    return np.exp(lg), np.exp(lxi), np.logaddexp.reduce(la[-1])


def m_step(sequences, gammas, K, d):
    pi_new = np.maximum(np.mean([g[0] for g in gammas], axis=0), 1e-8)
    pi_new /= pi_new.sum()
    dyn_M, dyn_b, dyn_cov = np.zeros((K,d,d)), np.zeros((K,d)), np.zeros((K,d,d))
    for k in range(K):
        W_sum  = np.zeros((d+1, d+1))
        WY_sum = np.zeros((d+1, d))
        for seq, gamma in zip(sequences, gammas):
            X_in_seq  = seq[:-1]
            X_out_seq = seq[1:]
            X_aug = np.hstack([X_in_seq, np.ones((len(X_in_seq), 1))])
            w = gamma[1:, k]
            W_sum  += (X_aug * w[:,None]).T @ X_aug
            WY_sum += (X_aug * w[:,None]).T @ X_out_seq
        coef = np.linalg.solve(W_sum + 1e-4*np.eye(d+1), WY_sum)
        dyn_M[k], dyn_b[k] = coef[:d].T, coef[d]
        num, den = np.zeros((d, d)), 1e-9
        for seq, gamma in zip(sequences, gammas):
            err = seq[1:] - (seq[:-1] @ dyn_M[k].T + dyn_b[k])
            w = gamma[1:, k]
            num += (err * w[:,None]).T @ err
            den += w.sum()
        dyn_cov[k] = num/den + COV_REG * np.eye(d)  # FIX #4
    return pi_new, dyn_M, dyn_b, dyn_cov


def update_trans(trans_net, opt, z_all, xis):
    xi_all = torch.tensor(np.vstack(xis), dtype=torch.float32)
    trans_net.train()
    for _ in range(TRANS_GRAD_STEPS):
        log_probs = []
        for i in range(trans_net.K):
            s_oh = torch.zeros(len(z_all), trans_net.K); s_oh[:, i] = 1.0
            log_probs.append(torch.log(F.softmax(trans_net.net(torch.cat([z_all, s_oh], dim=1)), dim=1) + 1e-12))
        loss = -(xi_all * torch.stack(log_probs, dim=1)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    trans_net.eval()


def compute_bic(sequences, gammas, K, d):
    """BIC = -2 * total_log_likelihood + n_params * log(N)"""
    total_ll = sum(
        np.logaddexp.reduce(la_row)
        for gamma in gammas
        for la_row in gamma  # approximate via gamma sums
    )
    # use sum of log normalizers approximation: just sum log-likelihoods from FB
    n_params = K*(K-1) + K*(d**2 + d + d*(d+1)//2)
    N = sum(len(s) for s in sequences)
    return n_params * np.log(N)  # placeholder — actual ll passed in


def fit_sds(sequences, K, d):
    pi, dM, db, dCov = init_params(sequences, K, d)
    trans_net = TransitionNet(d, K)
    opt = optim.Adam(trans_net.parameters(), lr=TRANS_LR)
    z_all = torch.tensor(np.vstack([s[:-1] for s in sequences]), dtype=torch.float32)
    total_ll = -np.inf
    for it in range(N_ITERS_SLDS):
        gammas, xis, lls = [], [], []
        for seq in sequences:
            g, x, ll = forward_backward(seq, pi, trans_net.log_A_sequence(seq[:-1]), dM, db, dCov, K)
            gammas.append(g); xis.append(x); lls.append(ll)
        total_ll = sum(lls)
        pi, dM, db, dCov = m_step(sequences, gammas, K, d)
        update_trans(trans_net, opt, z_all, xis)

    # FIX #1: compute mus from emission model — weighted means using soft assignments
    # mu_k = E[z_t | s_t=k] = weighted empirical mean, which equals
    # the emission center. Using soft gamma weights is consistent with the EM objective.
    # This is the correct mu for Bhattacharyya: it matches the Gaussian emission N(mu_k, Sigma_k)
    # where Sigma_k = dCov[k] is the residual covariance around A_k z_{t-1} + b_k.
    # The marginal emission mean is thus the weighted mean of z_t under regime k.
    mus = np.zeros((K, d))
    weights = np.zeros(K)
    for seq, gamma in zip(sequences, gammas):
        for k in range(K):
            w = gamma[:, k]
            mus[k]    += (seq * w[:, None]).sum(0)
            weights[k] += w.sum()
    for k in range(K):
        if weights[k] > 0:
            mus[k] /= weights[k]
        # FIX #1 fallback: if regime is empty, use b_k from emission model
        else:
            mus[k] = db[k]

    # FIX #1: covariances for Bhattacharyya = soft-weighted residual covariance
    # around the regime emission mean (not dCov which is the AR residual covariance)
    covs = np.zeros((K, d, d))
    for seq, gamma in zip(sequences, gammas):
        for k in range(K):
            w   = gamma[:, k]
            dev = seq - mus[k]
            covs[k] += (dev * w[:, None]).T @ dev
    for k in range(K):
        covs[k] = covs[k] / max(weights[k], 1.0) + COV_REG * np.eye(d)

    # BIC for K selection
    N = sum(len(s) for s in sequences)
    n_params = K*(K-1) + K*(d**2 + d + d*(d+1)//2)
    bic = -2 * total_ll + n_params * np.log(N)

    return mus, dM, db, covs, bic


def bhattacharyya(mu_i, mu_j, S_i, S_j):
    S_avg = (S_i + S_j) / 2
    diff  = mu_i - mu_j
    term1 = 0.125 * diff @ np.linalg.solve(S_avg, diff)
    _, ld_avg = np.linalg.slogdet(S_avg)
    _, ld_i   = np.linalg.slogdet(S_i)
    _, ld_j   = np.linalg.slogdet(S_j)
    return float(term1 + 0.5*(ld_avg - 0.5*ld_i - 0.5*ld_j))


def compute_B(mus, covs):
    K = len(mus)
    if K < 2: return 0.0
    return min(bhattacharyya(mus[i], mus[j], covs[i], covs[j])
               for i in range(K) for j in range(i+1, K))


def compute_M(As, bs, sequences):
    K = len(As)
    z_all = np.vstack([s[:-1] for s in sequences])
    M = 0.0
    for s in range(K):
        for k in range(K):
            if k == s: continue
            gaps = np.sum((z_all @ (As[k] - As[s]).T + (bs[k] - bs[s]))**2, axis=1)
            M = max(M, gaps.max())
    return float(M)


def fit_sds_best_k(sequences, d):
    """FIX #3: select K by BIC over K_SWEEP."""
    best_k, best_bic, best_result = None, np.inf, None
    for k in K_SWEEP:
        try:
            result = fit_sds(sequences, k, d)
            bic = result[-1]
            print(f"    K={k} BIC={bic:.1f}", flush=True)
            if bic < best_bic:
                best_bic, best_k, best_result = bic, k, result
        except Exception as e:
            print(f"    K={k} failed: {e}", flush=True)
    print(f"    → Best K={best_k} (BIC={best_bic:.1f})", flush=True)
    return best_result  # (mus, dM, db, covs, bic)


def verify(path):
    print(f"\n{'='*60}\n{path}\n{'='*60}", flush=True)
    all_f, triplets = load_and_prepare_cebra(path)
    cebra_seqs, pca_seqs = train_cebra_projection(all_f, triplets)

    print("  Fitting SDS on CEBRA embeddings...", flush=True)
    mus_C, As_C, bs_C, covs_C, bic_C = fit_sds_best_k(cebra_seqs, D)
    B_C = compute_B(mus_C, covs_C)
    M_C = compute_M(As_C, bs_C, cebra_seqs)

    print("  Fitting SDS on PCA embeddings...", flush=True)
    mus_P, As_P, bs_P, covs_P, bic_P = fit_sds_best_k(pca_seqs, PCA_DIM)
    B_P = compute_B(mus_P, covs_P)
    M_P = compute_M(As_P, bs_P, pca_seqs)

    lhs = B_C - B_P
    rhs = float(np.log(M_C / M_P)) if M_C > 0 and M_P > 0 else float('nan')
    satisfied = bool(lhs > rhs)

    print(f"  B(f_C)           = {B_C:.4f}")
    print(f"  B(f_P)           = {B_P:.4f}")
    print(f"  M_C              = {M_C:.4f}")
    print(f"  M_P              = {M_P:.4f}")
    print(f"  B(f_C) - B(f_P)  = {lhs:.4f}")
    print(f"  log(M_C / M_P)   = {rhs:.4f}")
    print(f"  Condition holds  : {satisfied}", flush=True)

    return {"B_C": B_C, "B_P": B_P, "M_C": M_C, "M_P": M_P,
            "lhs": lhs, "rhs": rhs, "satisfied": satisfied}


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results = {}
    for path in TEST_PATHS:
        if not os.path.exists(path):
            print(f"Skipping (not found): {path}", flush=True)
            continue
        tag = "_".join(path.split("/")[-4:-1])
        if tag in results:
            print(f"Skipping (exists): {tag}", flush=True)
            continue
        try:
            results[tag] = verify(path)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
        # save incrementally
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUTPUT_PATH}", flush=True)