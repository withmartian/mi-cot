import os
import numpy as np
import pickle
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
    X_pca = PCA(n_components=PCA_DIM, random_state=42).fit_transform(
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
    labs = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X_out)
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
    return np.exp(lg), np.exp(lxi)

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

def sss(persist, mean_st, T, K, K_eff):
    if K==1: return 0.0
    row_ents = -np.sum(T*np.log(T+1e-12),1)/np.log(K)
    return (K_eff/K)*mean_st*np.log(persist+1)*(1-np.mean(row_ents))

def fit_and_evaluate(seqs, labels, K):
    pi,A,dM,db,dCov = init_params(seqs, K)
    for _ in range(N_ITERS):
        gammas,xis = [],[]
        for s in seqs:
            g,x = forward_backward(s,pi,A,dM,db,dCov,K)
            gammas.append(g); xis.append(x)
        pi,A,dM,db,dCov = m_step(seqs,gammas,xis,K)
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
    spec   = np.mean(np.max(conf_norm,1))
    T_hard = hard_tm(state_seqs, K)
    mean_st= np.diag(T_hard).mean()
    score  = sss(persist, mean_st, T_hard, K, K_eff)
    r2     = regime_r2(state_seqs, seqs)
    return persist, mean_st, spec, conf_norm, T_hard, score, K_eff, r2


if __name__ == "__main__":
    path = "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_reasoning/layer_28/all_sentences_features.pkl"

    print("Loading and projecting data...", flush=True)
    seqs, labels = load_data(path)

    ar_r2 = linear_ar_r2(seqs)
    print(f"\n  Linear AR baseline R² (PCA space): {ar_r2:.4f}", flush=True)
    print(f"\n{'K':<4} | {'K_eff':<6} | {'Persist':<10} | {'Self-Trans':<10} | {'Spec':<10} | {'SSS':<8} | {'R²':<8} | {'ΔR²':<8}", flush=True)
    print("-"*80, flush=True)

    sss_scores = {}
    for k in K_SWEEP:
        p,st,s,C_mat,T_hard,score,k_eff,r2 = fit_and_evaluate(seqs, labels, k)
        sss_scores[k] = score
        print(f"{k:<4} | {k_eff:<6} | {p:<10.2f} | {st:<10.3f} | {s:<10.4f} | {score:<8.4f} | {r2:<8.4f} | {r2-ar_r2:+.4f}", flush=True)
        print(f"\n  Transition matrix (K={k}):", flush=True)
        print_tm(T_hard, k)
        if k >= 4:
            print(f"\n  Dominant stages (K={k}):", flush=True)
            for i in range(k):
                idx = np.argmax(C_mat[i])
                print(f"    Mode {i}: {STAGES[idx]} ({C_mat[i,idx]:.1%})", flush=True)
        print()

    best_k = max(sss_scores, key=sss_scores.get)
    print(f"\n  Best K by SSS: K={best_k}  (SSS={sss_scores[best_k]:.4f})", flush=True)
    print(f"  SSS profile: "+"  ".join([f"K{k}={v:.3f}" for k,v in sss_scores.items()]), flush=True)