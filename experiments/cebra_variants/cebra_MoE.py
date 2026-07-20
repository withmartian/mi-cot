import json
import random
from datetime import datetime
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

# ── REPRODUCIBILITY ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

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


# ── METRICS ─────────────────────────────────────────────────

def tvd_from_uniform(T):
    K = len(T)
    uniform = np.ones((K,K)) / K
    return float(np.mean(0.5 * np.abs(T - uniform).sum(axis=1)))

def spectral_gap(T):
    K = len(T)
    if K <= 1: return 0.0
    eigs = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    return float(1 - eigs[1])

def stationary_entropy(T):
    ev, evec = np.linalg.eig(T.T)
    idx  = np.argmin(np.abs(ev - 1.0))
    stat = np.abs(evec[:,idx].real); stat /= stat.sum()
    return float(-np.sum(stat * np.log(stat + 1e-12)))

def kl_from_uniform(T):
    K = len(T)
    uniform = np.ones((K,K)) / K
    return float(np.mean(np.sum(T * np.log(T / (uniform + 1e-12) + 1e-12), axis=1)))

def hard_transition_matrix(state_seqs, K):
    T = np.zeros((K, K))
    for seq in state_seqs:
        for a, b in zip(seq[:-1], seq[1:]): T[a, b] += 1
    row_sums = T.sum(1, keepdims=True)
    return T / np.where(row_sums == 0, 1, row_sums)

def tqs(T, K):
    if K == 1: return 0.0
    if np.min(np.diag(T)) < 0.3: return 0.0
    H_mean = np.mean(-np.sum(T * np.log(T + 1e-12), axis=1) / np.log(K))
    return (1 - np.trace(T) / K) * (1 - H_mean)

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
    return 1 - np.sum((X_out - pred)**2) / np.sum((X_out - X_out.mean(0))**2)

def regime_r2_on_pca(state_seqs, pca_seqs):
    K = max(s.max() for s in state_seqs) + 1
    X_in_k  = [[] for _ in range(K)]; X_out_k = [[] for _ in range(K)]
    for s_seq, p_seq in zip(state_seqs, pca_seqs):
        for t in range(len(s_seq) - 1):
            X_in_k[s_seq[t]].append(p_seq[t]); X_out_k[s_seq[t]].append(p_seq[t+1])
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
            all_pred.append(np.append(p_seq[t], 1.0) @ coefs[k]); all_true.append(p_seq[t+1])
    all_pred = np.array(all_pred); all_true = np.array(all_true)
    return 1 - np.sum((all_true-all_pred)**2) / np.sum((all_true-all_true.mean(0))**2)

# ── TRAINING ─────────────────────────────────────────────────

def train_and_eval(K, X_torch, all_features, pca_seqs, triplets, epochs=75):
    torch.manual_seed(SEED + K)  # deterministic per K
    np.random.seed(SEED + K)
    d_h = 32
    model = CEBRA_MoE_Encoder(X_torch.shape[1], d_h, K).to(device)
    dyn   = DynamicsMoE(K, d_h).to(device)
    opt   = optim.AdamW(list(model.parameters()) + list(dyn.parameters()), lr=1e-3)

    for epoch in range(epochs):
        tau   = max(0.2, 1.5 * (0.92 ** epoch))
        w_div = min(30.0, (epoch / 15.0) * 30.0)
        indices = np.random.permutation(len(triplets))
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

    p_map = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map[f['problem_id']].append((f['sentence_idx'], states[i]))
    state_seqs, pid_order = [], []
    for pid in sorted(p_map.keys()):
        seq = [s for _, s in sorted(p_map[pid])]
        if len(seq) > 1:
            state_seqs.append(np.array(seq)); pid_order.append(pid)

    pid_to_pca  = {pid: pca_seqs[i] for i, pid in enumerate(sorted(p_map.keys())) if len(p_map[pid]) > 1}
    aligned_pca = [pid_to_pca[pid] for pid in pid_order if pid in pid_to_pca]
    state_seqs  = state_seqs[:len(aligned_pca)]

    all_states = np.concatenate(state_seqs)
    counts     = np.bincount(all_states, minlength=K)
    K_eff      = int(np.sum(counts / len(all_states) > 0.01))
    persist    = np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs])

    p_map_l = defaultdict(list)
    for f in all_features: p_map_l[f['problem_id']].append(f.get('stage', 'NEUTRAL'))
    confusion = np.zeros((K, len(STAGES)))
    for pid, s_seq in zip(pid_order, state_seqs):
        if pid not in pid_to_pca: continue
        for s, l in zip(s_seq, p_map_l[pid]):
            if l in STAGES: confusion[s, STAGES.index(l)] += 1
    conf_norm = confusion / (confusion.sum(1, keepdims=True) + 1e-9)
    spec      = np.mean(np.max(conf_norm, axis=1))

    T_hard    = hard_transition_matrix(state_seqs, K)
    mean_st   = np.diag(T_hard).mean()
    score     = sss(persist, mean_st, T_hard, K, K_eff)
    tqs_score = tqs(T_hard, K)
    r2        = regime_r2_on_pca(state_seqs, aligned_pca)
    tvd       = tvd_from_uniform(T_hard)
    sg        = spectral_gap(T_hard)
    se        = stationary_entropy(T_hard)
    kl        = kl_from_uniform(T_hard)
    unique_st = len(set(
        STAGES[np.argmax(conf_norm[i])] for i in range(K)
        if conf_norm[i].max() > 0
    ))

    return persist, mean_st, spec, conf_norm, T_hard, score, tqs_score, K_eff, r2, tvd, sg, se, kl, unique_st


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
]

OUTPUT_DIR = "/home/abir19/scratch/abir19/SDS_results/cebra_moe"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path in DATASETS:
        parts = path.split("/")
        tag = f"{parts[-4]}_{parts[-3]}_{parts[-2]}"
        out_path = os.path.join(OUTPUT_DIR, f"{tag}.json")

        if os.path.exists(out_path):
            print(f"  Skipping - already exists: {out_path}", flush=True)
            continue

        if not os.path.exists(path):
            print(f"  Skipping - data not found: {path}", flush=True)
            continue

        print(f"\n{'='*50}\nProcessing: {tag}\n{'='*50}", flush=True)

        try:
            all_features, triplets = load_data(path)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            continue

        X = np.array([f['hidden_state_last'] for f in all_features])
        scaler = StandardScaler(); X = scaler.fit_transform(X)
        pca = PCA(n_components=PCA_DIM, random_state=SEED); X_pca = pca.fit_transform(X)
        X_torch = torch.tensor(X, dtype=torch.float32).to(device)

        p_map = defaultdict(list)
        for i, feat in enumerate(all_features):
            p_map[feat['problem_id']].append((feat['sentence_idx'], X_pca[i]))
        pca_seqs = [np.array([x for _, x in sorted(p_map[pid])]) for pid in sorted(p_map.keys()) if len(p_map[pid]) > 1]

        ar_r2 = linear_ar_r2(pca_seqs)
        print(f"  Linear AR baseline R²: {ar_r2:.4f}", flush=True)

        sweep = {}
        sss_scores, tqs_scores = {}, {}

        for k in K_SWEEP:
            p, st, s, C_mat, T_hard, score, tqs_score, k_eff, r2, tvd, sg, se, kl, unique_st = \
                train_and_eval(k, X_torch, all_features, pca_seqs, triplets)
            sss_scores[k] = score
            tqs_scores[k] = tqs_score

            dominant = [
                {"mode": i, "stage": STAGES[np.argmax(C_mat[i])], "score": float(np.max(C_mat[i]))}
                for i in range(k)
            ] if k >= 2 else []

            sweep[str(k)] = {
                "k_eff":             int(k_eff),
                "persist":           float(p),
                "mean_self_trans":   float(st),
                "spec":              float(s),
                "sss":               float(score),
                "tqs":               float(tqs_score),
                "r2":                float(r2),
                "delta_r2":          float(r2 - ar_r2),
                "tvd":               float(tvd),
                "spectral_gap":      float(sg),
                "stationary_entropy":float(se),
                "kl_uniform":        float(kl),
                "unique_stages":     int(unique_st),
                "transition_matrix": T_hard.tolist(),
                "dominant_stages":   dominant,
            }
            print(f"  K={k}: SSS={score:.4f} TQS={tqs_score:.4f} TVD={tvd:.4f} "
                  f"SpGap={sg:.4f} R²={r2:.4f}", flush=True)

        result = {
            "metadata": {
                "data_path": path,
                "method": "cebra_moe",
                "timestamp": datetime.now().isoformat(),
                "seed": SEED,
                "config": {"k_sweep": list(K_SWEEP), "pca_dim": PCA_DIM}
            },
            "baseline": {"linear_ar_r2": float(ar_r2)},
            "best_k": {
                "by_sss": int(max(sss_scores, key=sss_scores.get)),
                "by_tqs": int(max(tqs_scores, key=tqs_scores.get)),
            },
            "sweep": sweep,
        }

        with open(out_path, 'w') as f_out:
            json.dump(result, f_out, indent=2)
        print(f"  Saved -> {out_path}", flush=True)