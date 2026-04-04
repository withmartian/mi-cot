"""
Loads existing cebra_em JSON results, re-runs regime R2 with random state
permutations at best_k (by BIC), and reports the drop vs identity.
"""

import os, json, pickle, numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

PCA_DIM   = 40
KAPPA     = 10.0
N_ITERS   = 50
N_RANDOM  = 50

CEBRA_EM_DIR = "/home/abir19/scratch/abir19/SDS_results/cebra_em"
DATA_ROOT    = "/home/abir19/scratch/abir19"
OUTPUT_DIR   = "/home/abir19/scratch/abir19/SDS_results/state_swap"

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
device = "cuda" if torch.cuda.is_available() else "cpu"

class CEBRANet(nn.Module):
    def __init__(self, d_in, d_out=PCA_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, d_out))
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

def load_and_triplets(path, limit=500, max_t=25):
    all_f = pickle.load(open(path, 'rb'))
    all_f = [f for f in all_f if f['problem_id'] < limit]
    p_map = defaultdict(list)
    for i, f in enumerate(all_f): p_map[f['problem_id']].append(i)
    pids, trips = list(p_map.keys()), []
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: continue
        for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_t), replace=False):
            neg = np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])
            trips.append((idxs[t], idxs[t+1], neg))
    return all_f, trips

def train_cebra(all_features, triplets, epochs=100, batch=1024):
    X_raw = np.array([f['hidden_state_last'] for f in all_features])
    X_sc  = StandardScaler().fit_transform(X_raw)
    X_pca = PCA(n_components=PCA_DIM, random_state=42).fit_transform(X_sc)
    X_t   = torch.tensor(X_sc, dtype=torch.float32).to(device)
    model = CEBRANet(X_raw.shape[1]).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    tarr  = np.array(triplets)
    for _ in range(epochs):
        idx = np.random.permutation(len(tarr))
        for i in range(0, len(tarr), batch):
            b  = tarr[idx[i:i+batch]]
            za = model(X_t[b[:,0]]); zp = model(X_t[b[:,1]]); zn = model(X_t[b[:,2]])
            sp = torch.sum(za*zp,1)/0.1; sn = torch.sum(za*zn,1)/0.1
            loss = -torch.log(torch.exp(sp)/(torch.exp(sp)+torch.exp(sn))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Z = model(X_t).cpu().numpy()
    p_map_z, p_map_p = defaultdict(list), defaultdict(list)
    for i, f in enumerate(all_features):
        pid = f['problem_id']
        p_map_z[pid].append(Z[i]); p_map_p[pid].append(X_pca[i])
    pids = sorted(p for p in p_map_z if len(p_map_z[p]) >= 3)
    return ([np.array(p_map_z[p]) for p in pids],
            [np.array(p_map_p[p]) for p in pids])

def init_params(seqs, K):
    X_in  = np.vstack([s[:-1] for s in seqs])
    X_out = np.vstack([s[1:]-s[:-1] for s in seqs])
    labs  = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X_out)
    D = PCA_DIM
    dM, db, dCov = np.zeros((K,D,D)), np.zeros((K,D)), np.array([np.eye(D)]*K)
    for k in range(K):
        m = labs==k
        if m.sum() < D+2: dM[k]=0.1*np.eye(D); continue
        W,*_ = np.linalg.lstsq(np.hstack([X_in[m],np.ones((m.sum(),1))]), X_out[m], rcond=None)
        dM[k],db[k] = W[:D].T, W[D]
        res = X_out[m]-(X_in[m]@dM[k].T+db[k])
        dCov[k] = np.cov(res.T)+1e-3*np.eye(D)
    return np.ones(K)/K, np.eye(K)*0.7+0.3/K, dM, db, dCov

def forward_backward(seq, pi, A, dM, db, dCov, K):
    T, D = len(seq), PCA_DIM
    log_emit = np.zeros((T,K))
    for k in range(K):
        _,ld = np.linalg.slogdet(dCov[k]); ic = np.linalg.inv(dCov[k])
        mns  = np.vstack([db[k], seq[:-1]@dM[k].T+db[k]])
        df   = seq-mns
        log_emit[:,k] = -0.5*(D*np.log(2*np.pi)+ld+np.sum((df@ic)*df,1))
    lA,lp = np.log(A+1e-12), np.log(pi+1e-12)
    la = np.zeros((T,K)); la[0] = lp+log_emit[0]
    for t in range(1,T):
        la[t] = log_emit[t]+np.logaddexp.reduce(la[t-1][:,None]+lA,0)
    lb = np.zeros((T,K))
    for t in range(T-2,-1,-1):
        lb[t] = np.logaddexp.reduce(lA+log_emit[t+1]+lb[t+1],1)
    lg = la+lb; lg -= np.logaddexp.reduce(lg,1,keepdims=True)
    lxi = np.zeros((T-1,K,K))
    for t in range(T-1):
        lxi[t] = la[t][:,None]+lA+log_emit[t+1]+lb[t+1]
        lxi[t] -= np.logaddexp.reduce(lxi[t].ravel())
    return np.exp(lg), np.exp(lxi), np.logaddexp.reduce(la[-1])

def m_step(seqs, gammas, xis, K):
    D = PCA_DIM
    xi_sum = sum(x.sum(0) for x in xis)+np.eye(K)*KAPPA+1e-8
    A_new  = xi_sum/xi_sum.sum(1,keepdims=True)
    pi_new = np.maximum(np.mean([g[0] for g in gammas],0),1e-8); pi_new/=pi_new.sum()
    dM,db,dCov = np.zeros((K,D,D)),np.zeros((K,D)),np.zeros((K,D,D))
    for k in range(K):
        Ws,WY = np.zeros((D+1,D+1)),np.zeros((D+1,D))
        for seq,g in zip(seqs,gammas):
            Xa = np.hstack([np.vstack([np.zeros(D),seq[:-1]]),np.ones((len(seq),1))])
            w  = g[:,k]
            Ws += (Xa*w[:,None]).T@Xa; WY += (Xa*w[:,None]).T@seq
        c = np.linalg.solve(Ws+1e-4*np.eye(D+1),WY)
        dM[k],db[k] = c[:D].T, c[D]
        num,den = np.zeros((D,D)),1e-9
        for seq,g in zip(seqs,gammas):
            err = seq-(np.vstack([np.zeros(D),seq[:-1]])@dM[k].T+db[k])
            num+=(err*g[:,k][:,None]).T@err; den+=g[:,k].sum()
        dCov[k] = num/den+1e-4*np.eye(D)
    return pi_new, A_new, dM, db, dCov

def fit_slds(seqs, K):
    pi,A,dM,db,dCov = init_params(seqs, K)
    for _ in range(N_ITERS):
        gammas,xis,lls = [],[],[]
        for s in seqs:
            g,x,ll = forward_backward(s,pi,A,dM,db,dCov,K)
            gammas.append(g); xis.append(x); lls.append(ll)
        pi,A,dM,db,dCov = m_step(seqs,gammas,xis,K)
    return [np.argmax(g,1) for g in gammas]

def fit_regime_coefs(state_seqs, pca_seqs, K):
    """Fit per-regime AR coefs on true state assignments. Returns list of coefs."""
    X_in_k  = [[] for _ in range(K)]; X_out_k = [[] for _ in range(K)]
    for s_seq, p_seq in zip(state_seqs, pca_seqs):
        for t in range(len(s_seq)-1):
            X_in_k[s_seq[t]].append(p_seq[t]); X_out_k[s_seq[t]].append(p_seq[t+1])
    coefs = []
    for k in range(K):
        if len(X_in_k[k]) < PCA_DIM+2: coefs.append(None); continue
        Xi = np.array(X_in_k[k]); Xo = np.array(X_out_k[k])
        c,*_ = np.linalg.lstsq(np.hstack([Xi,np.ones((len(Xi),1))]), Xo, rcond=None)
        coefs.append(c)
    return coefs

def eval_r2(state_seqs, pca_seqs, coefs, perm=None):
    """Evaluate R2 using fixed coefs. If perm given, state i uses coef perm[i]."""
    pred, true = [], []
    for s_seq, p_seq in zip(state_seqs, pca_seqs):
        for t in range(len(s_seq)-1):
            k = perm[s_seq[t]] if perm is not None else s_seq[t]
            if coefs[k] is None: continue
            pred.append(np.append(p_seq[t],1.0)@coefs[k]); true.append(p_seq[t+1])
    if not pred: return float('nan')
    pred = np.array(pred); true = np.array(true)
    return float(1-np.sum((true-pred)**2)/np.sum((true-true.mean(0))**2))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    for fname in sorted(os.listdir(CEBRA_EM_DIR)):
        if not fname.endswith(".json"): continue
        tag      = fname[:-5]
        out_path = os.path.join(OUTPUT_DIR, fname)

        if os.path.exists(out_path):
            print(f"  Skipping - already exists: {tag}", flush=True)
            continue

        em_result = json.load(open(os.path.join(CEBRA_EM_DIR, fname)))
        data_path = em_result["metadata"]["data_path"]
        best_k    = em_result["best_k"]["by_bic"]

        if not os.path.exists(data_path):
            print(f"  Skipping - data not found: {data_path}", flush=True)
            continue

        print(f"\n{'='*50}\n{tag}  K={best_k}\n{'='*50}", flush=True)

        try:
            all_f, trips = load_and_triplets(data_path)
            cebra_seqs, pca_seqs = train_cebra(all_f, trips)
            state_seqs = fit_slds(cebra_seqs, best_k)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            continue

        coefs = fit_regime_coefs(state_seqs, pca_seqs, best_k)
        r2_identity = eval_r2(state_seqs, pca_seqs, coefs)

        rand_r2s = []
        for _ in range(N_RANDOM):
            perm = rng.permutation(best_k).tolist()
            rand_r2s.append(eval_r2(state_seqs, pca_seqs, coefs, perm))






        r2_rand_mean = float(np.mean(rand_r2s))
        r2_rand_std  = float(np.std(rand_r2s))

        print(f"  R2 identity={r2_identity:.4f}  rand_mean={r2_rand_mean:.4f}  "
              f"drop={r2_identity-r2_rand_mean:.4f}", flush=True)

        result = {
            "metadata":      em_result["metadata"],
            "best_k":        best_k,
            "r2_identity":   r2_identity,
            "r2_rand_mean":  r2_rand_mean,
            "r2_rand_std":   r2_rand_std,
            "r2_drop":       r2_identity - r2_rand_mean,
            "n_permutations": N_RANDOM,
            "timestamp":     datetime.now().isoformat(),
        }
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved -> {out_path}", flush=True)