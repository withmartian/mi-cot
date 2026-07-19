import json
import random
from datetime import datetime
import os
import numpy as np
import pickle
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ── REPRODUCIBILITY ──────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

K_SWEEP = range(1, 13)
PCA_DIM = 40
KAPPA   = 10.0
N_ITERS = 50
STAGES = [
    "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
]

def load_data(path, limit_problems=500):
    with open(path, 'rb') as f:
        all_features = pickle.load(f)
    all_features = [f for f in all_features if f['problem_id'] < limit_problems]
    X_raw = np.array([f['hidden_state_last'] for f in all_features])
    X_pca = PCA(n_components=PCA_DIM, random_state=SEED).fit_transform(
                StandardScaler().fit_transform(X_raw))
    p_map_x, p_map_l = defaultdict(list), defaultdict(list)
    for i, f in enumerate(all_features):
        p_map_x[f['problem_id']].append((f['sentence_idx'], X_pca[i]))
        p_map_l[f['problem_id']].append((f['sentence_idx'], f.get('stage', 'NEUTRAL')))
    pids = sorted(p for p in p_map_x if len(p_map_x[p]) >= 3)
    seqs   = [np.array([x for _, x in sorted(p_map_x[p])]) for p in pids]
    labels = [[l for _, l in sorted(p_map_l[p])]            for p in pids]
    return seqs, labels

def linear_ar_r2(seqs):
    X_in  = np.vstack([s[:-1] for s in seqs])
    X_out = np.vstack([s[1:]  for s in seqs])
    X_aug = np.hstack([X_in, np.ones((len(X_in), 1))])
    coef, *_ = np.linalg.lstsq(X_aug, X_out, rcond=None)
    pred  = X_aug @ coef
    return 1 - np.sum((X_out - pred)**2) / np.sum((X_out - X_out.mean(0))**2)

def regime_r2(state_seqs, seqs):
    K = max(s.max() for s in state_seqs) + 1
    X_in_k = [[] for _ in range(K)]; X_out_k = [[] for _ in range(K)]
    for s_seq, seq in zip(state_seqs, seqs):
        for t in range(len(s_seq)-1):
            X_in_k[s_seq[t]].append(seq[t]); X_out_k[s_seq[t]].append(seq[t+1])
    coefs = []
    for k in range(K):
        if len(X_in_k[k]) < PCA_DIM+2: coefs.append(None); continue
        Xi = np.array(X_in_k[k]); Xo = np.array(X_out_k[k])
        c, *_ = np.linalg.lstsq(np.hstack([Xi, np.ones((len(Xi),1))]), Xo, rcond=None)
        coefs.append(c)
    pred, true = [], []
    for s_seq, seq in zip(state_seqs, seqs):
        for t in range(len(s_seq)-1):
            k = s_seq[t]
            if coefs[k] is None: continue
            pred.append(np.append(seq[t], 1.0) @ coefs[k]); true.append(seq[t+1])
    pred = np.array(pred); true = np.array(true)
    return 1 - np.sum((true-pred)**2) / np.sum((true-true.mean(0))**2)

def init_params(seqs, K):
    X_in, X_out = [], []
    for s in seqs:
        X_in.append(s[:-1]); X_out.append(s[1:]-s[:-1])
    X_in = np.vstack(X_in); X_out = np.vstack(X_out)
    labs = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X_out)
    dM, db, dCov = np.zeros((K,PCA_DIM,PCA_DIM)), np.zeros((K,PCA_DIM)), np.array([np.eye(PCA_DIM)]*K)
    for k in range(K):
        m = labs==k
        if m.sum() < PCA_DIM+2: dM[k]=0.1*np.eye(PCA_DIM); continue
        W,*_ = np.linalg.lstsq(np.hstack([X_in[m],np.ones((m.sum(),1))]), X_out[m], rcond=None)
        dM[k],db[k] = W[:PCA_DIM].T, W[PCA_DIM]
        res = X_out[m] - (X_in[m]@dM[k].T + db[k])
        dCov[k] = np.cov(res.T) + 1e-3*np.eye(PCA_DIM)
    return np.ones(K)/K, np.eye(K)*0.7+0.3/K, dM, db, dCov

def forward_backward(seq, pi, A, dM, db, dCov, K):
    T = len(seq); D = PCA_DIM
    log_emit = np.zeros((T,K))
    for k in range(K):
        _,logdet = np.linalg.slogdet(dCov[k])
        inv_cov  = np.linalg.inv(dCov[k])
        means    = np.vstack([db[k], seq[:-1]@dM[k].T+db[k]])
        diffs    = seq - means
        log_emit[:,k] = -0.5*(D*np.log(2*np.pi)+logdet+np.sum((diffs@inv_cov)*diffs,1))
    log_A,log_pi = np.log(A+1e-12), np.log(pi+1e-12)
    la = np.zeros((T,K)); la[0] = log_pi+log_emit[0]
    for t in range(1,T):
        la[t] = log_emit[t]+np.logaddexp.reduce(la[t-1][:,None]+log_A,0)
    lb = np.zeros((T,K))
    for t in range(T-2,-1,-1):
        lb[t] = np.logaddexp.reduce(log_A+log_emit[t+1]+lb[t+1],1)
    lg = la+lb; lg -= np.logaddexp.reduce(lg,1,keepdims=True)
    lxi = np.zeros((T-1,K,K))
    for t in range(T-1):
        lxi[t] = la[t][:,None]+log_A+log_emit[t+1]+lb[t+1]
        lxi[t] -= np.logaddexp.reduce(lxi[t].ravel())
    seq_ll = np.logaddexp.reduce(la[-1])
    return np.exp(lg), np.exp(lxi), seq_ll

def m_step(seqs, gammas, xis, K):
    D = PCA_DIM
    xi_sum = sum(x.sum(0) for x in xis) + np.eye(K)*KAPPA + 1e-8
    A_new  = xi_sum / xi_sum.sum(1,keepdims=True)
    pi_new = np.maximum(np.mean([g[0] for g in gammas],0), 1e-8); pi_new/=pi_new.sum()
    dM,db,dCov = np.zeros((K,D,D)),np.zeros((K,D)),np.zeros((K,D,D))
    for k in range(K):
        Ws,WY = np.zeros((D+1,D+1)),np.zeros((D+1,D))
        for seq,g in zip(seqs,gammas):
            Xa = np.hstack([np.vstack([np.zeros(D),seq[:-1]]),np.ones((len(seq),1))])
            w  = g[:,k]
            Ws += (Xa*w[:,None]).T@Xa; WY += (Xa*w[:,None]).T@seq
        c = np.linalg.solve(Ws+1e-4*np.eye(D+1), WY)
        dM[k],db[k] = c[:D].T, c[D]
        num,den = np.zeros((D,D)),1e-9
        for seq,g in zip(seqs,gammas):
            err = seq-(np.vstack([np.zeros(D),seq[:-1]])@dM[k].T+db[k])
            num+=(err*g[:,k][:,None]).T@err; den+=g[:,k].sum()
        dCov[k] = num/den + 1e-4*np.eye(D)
    return pi_new, A_new, dM, db, dCov

def hard_tm(state_seqs, K):
    T = np.zeros((K,K))
    for s in state_seqs:
        for a,b in zip(s[:-1],s[1:]): T[a,b]+=1
    r = T.sum(1,keepdims=True)
    return T/np.where(r==0,1,r)

def print_tm(T, K):
    print("  "+"      "+" ".join(f"  →{j}" for j in range(K)), flush=True)
    for i in range(K):
        print(f"  {i} [ "+" ".join(f"{T[i,j]:5.2f}" for j in range(K))+f" ]  self={T[i,i]:.2f}", flush=True)
    print(f"  mean self-trans: {np.diag(T).mean():.3f}", flush=True)

def tqs(T, K):
    if K == 1: return 0.0
    if np.min(np.diag(T)) < 0.3: return 0.0
    H_mean = np.mean(-np.sum(T * np.log(T + 1e-12), axis=1) / np.log(K))
    return (1 - np.trace(T) / K) * (1 - H_mean)

def sss(persist, mean_st, T, K, K_eff):
    if K==1: return 0.0
    row_ents = -np.sum(T*np.log(T+1e-12),1)/np.log(K)
    return (K_eff/K)*mean_st*np.log(persist+1)*(1-np.mean(row_ents))

def compute_bic(total_ll, K, N):
    n_params = K*(K-1) + K*(PCA_DIM*PCA_DIM + PCA_DIM + PCA_DIM*(PCA_DIM+1)//2)
    return -2 * total_ll + n_params * np.log(N)

# ── NEW METRICS ───────────────────────────────────────────────

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

def fit_and_evaluate(seqs, labels, K):
    np.random.seed(SEED + K)  # deterministic per K
    pi,A,dM,db,dCov = init_params(seqs, K)
    for _ in range(N_ITERS):
        gammas,xis,lls = [],[],[]
        for s in seqs:
            g,x,ll = forward_backward(s,pi,A,dM,db,dCov,K)
            gammas.append(g); xis.append(x); lls.append(ll)
        pi,A,dM,db,dCov = m_step(seqs,gammas,xis,K)
    total_ll = sum(lls)
    N = sum(len(s) for s in seqs)
    bic = compute_bic(total_ll, K, N)
    state_seqs = [np.argmax(g,1) for g in gammas]
    all_s  = np.concatenate(state_seqs)
    counts = np.bincount(all_s, minlength=K)
    K_eff  = int(np.sum(counts/len(all_s)>0.01))
    persist= np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs])
    confusion = np.zeros((K,len(STAGES)))
    for s_seq,l_seq in zip(state_seqs,labels):
        for s,l in zip(s_seq,l_seq):
            if l in STAGES: confusion[s,STAGES.index(l)]+=1
    conf_norm = confusion/(confusion.sum(1,keepdims=True)+1e-9)
    spec      = np.mean(np.max(conf_norm,1))
    T_hard    = hard_tm(state_seqs, K)
    mean_st   = np.diag(T_hard).mean()
    score     = sss(persist, mean_st, T_hard, K, K_eff)
    tqs_score = tqs(T_hard, K)
    r2        = regime_r2(state_seqs, seqs)
    return (persist, mean_st, spec, conf_norm, T_hard, score, tqs_score, K_eff, r2, bic,
            tvd_from_uniform(T_hard), spectral_gap(T_hard),
            kl_from_uniform(T_hard), stationary_entropy(T_hard))


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

OUTPUT_DIR = "/home/abir19/scratch/abir19/SDS_results/pca_slds"

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
            seqs, labels = load_data(path)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            continue

        ar_r2 = linear_ar_r2(seqs)
        print(f"  Linear AR baseline R²: {ar_r2:.4f}", flush=True)

        sweep = {}
        sss_scores, tqs_scores, bic_scores = {}, {}, {}

        for k in K_SWEEP:
            (p, st, s, C_mat, T_hard, score, tqs_score, k_eff, r2, bic,
             tvd, sg, kl, se) = fit_and_evaluate(seqs, labels, k)
            sss_scores[k] = score
            tqs_scores[k] = tqs_score
            bic_scores[k] = bic

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
                "bic":               float(bic),
                "tvd":               float(tvd),
                "spectral_gap":      float(sg),
                "kl_uniform":        float(kl),
                "stationary_entropy":float(se),
                "transition_matrix": T_hard.tolist(),
                "dominant_stages":   dominant,
            }
            print(f"  K={k}: SSS={score:.4f} TQS={tqs_score:.4f} TVD={tvd:.4f} "
                  f"SpGap={sg:.4f} BIC={bic:.1f} R²={r2:.4f}", flush=True)

        result = {
            "metadata": {
                "data_path": path,
                "method": "pca_slds",
                "timestamp": datetime.now().isoformat(),
                "seed": SEED,
                "config": {"k_sweep": list(K_SWEEP), "n_iters_slds": N_ITERS}
            },
            "baseline": {"linear_ar_r2": float(ar_r2)},
            "best_k": {
                "by_bic": int(min(bic_scores, key=bic_scores.get)),
                "by_sss": int(max(sss_scores, key=sss_scores.get)),
                "by_tqs": int(max(tqs_scores, key=tqs_scores.get)),
            },
            "sweep": sweep,
        }

        with open(out_path, 'w') as f_out:
            json.dump(result, f_out, indent=2)
        print(f"  Saved -> {out_path}", flush=True)