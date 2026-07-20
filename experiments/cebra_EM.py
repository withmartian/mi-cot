import os
import json
import random
from datetime import datetime
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

# ── REPRODUCIBILITY ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

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

DATASETS = [
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

OUTPUT_DIR = "/home/abir19/scratch/abir19/SDS_results/cebra_em"


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
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled_np)
    X_scaled = torch.tensor(X_scaled_np, dtype=torch.float32).to(device)
    torch.manual_seed(SEED)
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
    # return cebra_seqs, pca_seqs, labels
    return cebra_seqs, pca_seqs, labels, scaler, model


def linear_ar_r2(pca_seqs):
    X_in  = np.vstack([s[:-1] for s in pca_seqs])
    X_out = np.vstack([s[1:]  for s in pca_seqs])
    X_aug = np.hstack([X_in, np.ones((len(X_in), 1))])
    coef, *_ = np.linalg.lstsq(X_aug, X_out, rcond=None)
    pred = X_aug @ coef
    return 1 - np.sum((X_out - pred)**2) / np.sum((X_out - X_out.mean(0))**2)


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
    return 1 - np.sum((all_true - all_pred)**2) / np.sum((all_true - all_true.mean(0))**2)


def init_params(sequences, K, D):
    X_in, X_out = [], []
    for seq in sequences:
        for t in range(len(seq)-1):
            X_in.append(seq[t]); X_out.append(seq[t+1] - seq[t])
    X_in, X_out = np.array(X_in), np.array(X_out)
    labels = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X_out)
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
    return np.exp(lg), np.exp(lxi), np.logaddexp.reduce(la[-1])


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


# ── METRICS ──────────────────────────────────────────────────

def tvd_from_uniform(T):
    K = len(T); u = np.ones((K,K))/K
    return float(np.mean(0.5*np.abs(T-u).sum(axis=1)))

def spectral_gap(T):
    K = len(T)
    if K <= 1: return 0.0
    return float(1 - np.sort(np.abs(np.linalg.eigvals(T)))[::-1][1])

def kl_from_uniform(T):
    K = len(T); u = np.ones((K,K))/K
    return float(np.mean(np.sum(T*np.log(T/(u+1e-12)+1e-12), axis=1)))

def stationary_entropy(T):
    ev, evec = np.linalg.eig(T.T)
    idx  = np.argmin(np.abs(ev - 1.0))
    stat = np.abs(evec[:,idx].real); stat /= stat.sum()
    return float(-np.sum(stat*np.log(stat+1e-12)))

def tqs(T, K):
    if K == 1: return 0.0
    if np.min(np.diag(T)) < 0.3: return 0.0
    H_mean = np.mean(-np.sum(T*np.log(T+1e-12), axis=1) / np.log(K))
    return (1 - np.trace(T)/K) * (1 - H_mean)

def sss(persist, mean_self_trans, T, K, K_eff):
    if K == 1: return 0.0
    row_ents = -np.sum(T * np.log(T + 1e-12), axis=1) / np.log(K)
    return (K_eff / K) * mean_self_trans * np.log(persist + 1) * (1 - np.mean(row_ents))

def compute_bic(total_ll, K, D, N):
    n_params = K*(K-1) + K*(D*D + D + D*(D+1)//2)
    return -2 * total_ll + n_params * np.log(N)


def fit_and_evaluate(cebra_seqs, pca_seqs, labels, K):
    np.random.seed(SEED + K)  # deterministic per K
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
    K_eff  = int(np.sum(counts / len(all_states) > 0.01))
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
    tqs_sc  = tqs(T_hard, K)
    r2      = regime_r2_on_pca(state_seqs, pca_seqs)

    return (persist, mean_st, spec, conf_norm, T_hard, score, tqs_sc, K_eff, r2, bic,
            tvd_from_uniform(T_hard), spectral_gap(T_hard),
            kl_from_uniform(T_hard), stationary_entropy(T_hard))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path in DATASETS:
        parts    = path.split("/")
        tag      = f"{parts[-4]}_{parts[-3]}_{parts[-2]}"
        out_path = os.path.join(OUTPUT_DIR, f"{tag}.json")

        if os.path.exists(out_path):
            print(f"  Skipping - already exists: {out_path}", flush=True)
            continue
        if not os.path.exists(path):
            print(f"  Skipping - data not found: {path}", flush=True)
            continue

        print(f"\n{'='*50}\nProcessing: {tag}\n{'='*50}", flush=True)
        try:
            all_f, triplets = load_and_prepare_cebra(path, mode='temporal')
            cebra_seqs, pca_seqs, labels = train_cebra_projection(all_f, triplets)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True); continue

        ar_r2 = linear_ar_r2(pca_seqs)
        print(f"  Linear AR baseline R²: {ar_r2:.4f}", flush=True)

        print(f"\n{'K':<4} | {'K_eff':<6} | {'Persist':<10} | {'Self-Trans':<10} | {'Spec':<10} | "
              f"{'SSS':<8} | {'TQS':<8} | {'TVD':<8} | {'SpGap':<8} | {'R²':<8} | {'ΔR²':<8} | {'BIC':<14}",
              flush=True)
        print("-" * 130, flush=True)

        sweep = {}
        sss_scores, tqs_scores, bic_scores = {}, {}, {}

        for k in K_SWEEP:
            try:
                (p, st, s, C_mat, T_hard, score, tqs_sc, k_eff, r2, bic,
                 tvd, sg, kl, se) = fit_and_evaluate(cebra_seqs, pca_seqs, labels, k)
            except Exception as e:
                print(f"  K={k} FAILED: {e}", flush=True); continue

            sss_scores[k] = score; tqs_scores[k] = tqs_sc; bic_scores[k] = bic
            delta_r2 = r2 - ar_r2
            print(f"{k:<4} | {k_eff:<6} | {p:<10.2f} | {st:<10.3f} | {s:<10.4f} | "
                  f"{score:<8.4f} | {tqs_sc:<8.4f} | {tvd:<8.4f} | {sg:<8.4f} | "
                  f"{r2:<8.4f} | {delta_r2:+.4f} | {bic:<14.1f}", flush=True)

            print(f"\n  Transition matrix (K={k}):", flush=True)
            print_transition_matrix(T_hard, k)

            if k >= 4:
                print(f"\n  Dominant stages (K={k}):", flush=True)
                for i in range(k):
                    idx = np.argmax(C_mat[i])
                    print(f"    Mode {i}: {STAGES[idx]} ({C_mat[i,idx]:.1%})", flush=True)
            print()

            dominant = [
                {"mode": i, "stage": STAGES[np.argmax(C_mat[i])],
                 "score": float(np.max(C_mat[i]))}
                for i in range(k)
            ] if k >= 2 else []

            sweep[str(k)] = {
                "k_eff":             int(k_eff),
                "persist":           float(p),
                "mean_self_trans":   float(st),
                "spec":              float(s),
                "sss":               float(score),
                "tqs":               float(tqs_sc),
                "r2":                float(r2),
                "delta_r2":          float(delta_r2),
                "bic":               float(bic),
                "tvd":               float(tvd),
                "spectral_gap":      float(sg),
                "kl_uniform":        float(kl),
                "stationary_entropy":float(se),
                "transition_matrix": T_hard.tolist(),
                "dominant_stages":   dominant,
            }

        if not bic_scores:
            print(f"  No valid K, skipping {tag}", flush=True); continue

        best_k_sss = max(sss_scores, key=sss_scores.get)
        best_k_tqs = max(tqs_scores, key=tqs_scores.get)
        best_k_bic = min(bic_scores, key=bic_scores.get)
        print(f"\n  Best K by SSS: K={best_k_sss}  (SSS={sss_scores[best_k_sss]:.4f})", flush=True)
        print(f"  Best K by TQS: K={best_k_tqs}  (TQS={tqs_scores[best_k_tqs]:.4f})", flush=True)
        print(f"  Best K by BIC: K={best_k_bic}  (BIC={bic_scores[best_k_bic]:.1f})", flush=True)

        result = {
            "metadata": {
                "data_path": path, "method": "cebra_slds",
                "timestamp": datetime.now().isoformat(),
                "seed": SEED,
                "config": {"k_sweep": list(K_SWEEP), "pca_dim": PCA_DIM,
                           "cebra_epochs": CEBRA_EPOCHS, "n_iters_slds": N_ITERS_SLDS},
            },
            "baseline": {"linear_ar_r2": float(ar_r2)},
            "best_k": {
                "by_bic": int(best_k_bic),
                "by_sss": int(best_k_sss),
                "by_tqs": int(best_k_tqs),
            },
            "sweep": sweep,
        }

        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved -> {out_path}", flush=True)