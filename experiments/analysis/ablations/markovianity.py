"""
markovianity_test.py

Tests Markovianity of fitted SLDS regime sequences using:
  1. Chapman-Kolmogorov test: ||T^2 - T_empirical(2)||_F
  2. Order-2 BIC test: BIC(order-1) vs BIC(order-2 Markov)

Both tests require raw state sequences — refit CEBRA+EM at best_k
then run the tests on the decoded state sequences.

Saves results to OUTPUT_PATH.
Usage: python markovianity_test.py
"""

import os, json, random, pickle
from datetime import datetime
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── SEED ─────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

D            = 40
PCA_DIM      = 40
CEBRA_EPOCHS = 100
BATCH_SIZE   = 1024
KAPPA        = 1.0
N_ITERS      = 50

OUTPUT_PATH = "/home/abir19/scratch/abir19/SDS_results/markovianity.json"

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

# ── CEBRA + SLDS (same as cebra_slds_sweep.py) ───────────────

class CEBRANet(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, d_out))
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

def load_and_embed(path, limit=500, max_triplets=25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_features = pickle.load(open(path, 'rb'))
    all_features = [f for f in all_features if f['problem_id'] < limit]
    p_map = defaultdict(list)
    for i, f in enumerate(all_features): p_map[f['problem_id']].append(i)
    triplets, pids = [], list(p_map.keys())
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: continue
        for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_triplets), replace=False):
            neg = np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])
            triplets.append((idxs[t], idxs[t+1], neg))
    X_raw = np.array([f['hidden_state_last'] for f in all_features])
    X_sc  = StandardScaler().fit_transform(X_raw)
    X_t   = torch.tensor(X_sc, dtype=torch.float32).to(device)
    torch.manual_seed(SEED)
    model = CEBRANet(X_raw.shape[1], D).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    tarr  = np.array(triplets)
    for _ in range(CEBRA_EPOCHS):
        idx = np.random.permutation(len(tarr))
        for i in range(0, len(tarr), BATCH_SIZE):
            b = tarr[idx[i:i+BATCH_SIZE]]
            za = model(X_t[b[:,0]]); zp = model(X_t[b[:,1]]); zn = model(X_t[b[:,2]])
            sp = torch.sum(za*zp,1)/0.1; sn = torch.sum(za*zn,1)/0.1
            loss = -torch.log(torch.exp(sp)/(torch.exp(sp)+torch.exp(sn))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Z = model(X_t).cpu().numpy()
    p_map_z = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map_z[f['problem_id']].append(Z[i])
    pids = sorted(p for p in p_map_z if len(p_map_z[p]) >= 3)
    return [np.array(p_map_z[p]) for p in pids]

def fit_slds_get_states(seqs, K):
    X_in  = np.vstack([s[:-1] for s in seqs])
    X_out = np.vstack([s[1:]-s[:-1] for s in seqs])
    labs  = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X_out)
    dM, db, dCov = np.zeros((K,D,D)), np.zeros((K,D)), np.array([np.eye(D)]*K)
    for k in range(K):
        m = labs==k
        if m.sum() < D+2: dM[k]=0.1*np.eye(D); continue
        W,*_ = np.linalg.lstsq(np.hstack([X_in[m],np.ones((m.sum(),1))]),X_out[m],rcond=None)
        dM[k],db[k] = W[:D].T,W[D]
        res = X_out[m]-(X_in[m]@dM[k].T+db[k])
        dCov[k] = np.cov(res.T)+1e-3*np.eye(D)
    pi = np.ones(K)/K; A = np.eye(K)*0.7+0.3/K

    def fwdbwd(seq):
        T = len(seq)
        le = np.zeros((T,K))
        for k in range(K):
            _,ld = np.linalg.slogdet(dCov[k]); ic=np.linalg.inv(dCov[k])
            mn = np.vstack([db[k],seq[:-1]@dM[k].T+db[k]]); df=seq-mn
            le[:,k] = -0.5*(D*np.log(2*np.pi)+ld+np.sum((df@ic)*df,1))
        lA=np.log(A+1e-12); lp=np.log(pi+1e-12)
        la=np.zeros((T,K)); la[0]=lp+le[0]
        for t in range(1,T):
            la[t]=le[t]+np.logaddexp.reduce(la[t-1][:,None]+lA,0)
        lb=np.zeros((T,K))
        for t in range(T-2,-1,-1):
            lb[t]=np.logaddexp.reduce(lA+le[t+1]+lb[t+1],1)
        lg=la+lb; lg-=np.logaddexp.reduce(lg,1,keepdims=True)
        lxi=np.zeros((T-1,K,K))
        for t in range(T-1):
            lxi[t]=la[t][:,None]+lA+le[t+1]+lb[t+1]
            lxi[t]-=np.logaddexp.reduce(lxi[t].ravel())
        return np.exp(lg),np.exp(lxi),np.logaddexp.reduce(la[-1])

    for _ in range(N_ITERS):
        gammas,xis = [],[]
        for s in seqs:
            g,x,_ = fwdbwd(s); gammas.append(g); xis.append(x)
        xi_sum = sum(x.sum(0) for x in xis)+np.eye(K)*1.0+1e-8
        A = xi_sum/xi_sum.sum(1,keepdims=True)
        pi = np.maximum(np.mean([g[0] for g in gammas],0),1e-8); pi/=pi.sum()
        for k in range(K):
            Ws,WY = np.zeros((D+1,D+1)),np.zeros((D+1,D))
            for seq,g in zip(seqs,gammas):
                Xa=np.hstack([np.vstack([np.zeros(D),seq[:-1]]),np.ones((len(seq),1))])
                w=g[:,k]; Ws+=(Xa*w[:,None]).T@Xa; WY+=(Xa*w[:,None]).T@seq
            c=np.linalg.solve(Ws+1e-4*np.eye(D+1),WY)
            dM[k],db[k]=c[:D].T,c[D]
            num,den=np.zeros((D,D)),1e-9
            for seq,g in zip(seqs,gammas):
                err=seq-(np.vstack([np.zeros(D),seq[:-1]])@dM[k].T+db[k])
                num+=(err*g[:,k][:,None]).T@err; den+=g[:,k].sum()
            dCov[k]=num/den+1e-4*np.eye(D)
    state_seqs = [np.argmax(g,1) for g in gammas]
    return state_seqs, A  # A = fitted 1-step transition matrix


# ── MARKOVIANITY TESTS ────────────────────────────────────────

def empirical_2step_tm(state_seqs, K):
    """Empirical 2-step transition matrix T(2)[i,j] = P(s_{t+2}=j | s_t=i)."""
    T2 = np.zeros((K, K))
    for seq in state_seqs:
        for t in range(len(seq)-2):
            T2[seq[t], seq[t+2]] += 1
    rs = T2.sum(1, keepdims=True)
    return T2 / np.where(rs==0, 1, rs)

def chapman_kolmogorov_test(state_seqs, T1, K):
    """
    CK test: for a Markov chain, T^2 should equal T(2).
    Residual = ||T^2 - T_empirical(2)||_F
    Normalized by ||T_empirical(2)||_F.
    Also compute p-value via bootstrap permutation.
    """
    T2_pred     = T1 @ T1
    T2_empirical = empirical_2step_tm(state_seqs, K)
    residual     = np.linalg.norm(T2_pred - T2_empirical, 'fro')
    normalizer   = np.linalg.norm(T2_empirical, 'fro') + 1e-12
    ck_residual  = float(residual / normalizer)

    # bootstrap: permute transitions within each sequence, recompute T(2)
    # null = what CK residual looks like if transitions are shuffled
    n_boot = 200
    boot_residuals = []
    for _ in range(n_boot):
        shuffled = [np.random.permutation(seq) for seq in state_seqs]
        T1_boot  = empirical_tm(shuffled, K)
        T2_boot  = empirical_2step_tm(shuffled, K)
        T2_pred_boot = T1_boot @ T1_boot
        r = np.linalg.norm(T2_pred_boot - T2_boot, 'fro') / (np.linalg.norm(T2_boot,'fro')+1e-12)
        boot_residuals.append(r)
    # p-value: fraction of bootstrap residuals LARGER than observed
    # (smaller residual = more Markovian, so p = fraction where null > observed)
    p_value = float(np.mean(np.array(boot_residuals) > ck_residual))
    return ck_residual, p_value, float(np.mean(boot_residuals))

def empirical_tm(state_seqs, K):
    T = np.zeros((K,K))
    for seq in state_seqs:
        for a,b in zip(seq[:-1],seq[1:]): T[a,b]+=1
    rs = T.sum(1,keepdims=True)
    return T/np.where(rs==0,1,rs)

def order2_bic_test(state_seqs, K):
    """
    Order-2 vs order-1 Markov test via BIC.
    Order-1: K*(K-1) free params, log-lik from T1
    Order-2: K^2*(K-1) free params, log-lik from T2 (3D transition tensor)
    If BIC(order-1) < BIC(order-2): chain is order-1 Markov.
    """
    # build 1-step and 2-step counts
    C1 = np.zeros((K,K))       # C1[i,j] = count of i->j
    C2 = np.zeros((K,K,K))     # C2[i,j,k] = count of i->j->k
    for seq in state_seqs:
        for t in range(len(seq)-1):
            C1[seq[t], seq[t+1]] += 1
        for t in range(len(seq)-2):
            C2[seq[t], seq[t+1], seq[t+2]] += 1

    N1 = C1.sum()  # total 1-step transitions
    N2 = C2.sum()  # total 2-step transitions

    # order-1 log-likelihood
    T1 = C1 / (C1.sum(1,keepdims=True)+1e-12)
    ll1 = float(np.sum(C1 * np.log(T1 + 1e-12)))

    # order-2 log-likelihood
    C2_sum = C2.sum(2, keepdims=True) + 1e-12
    T2 = C2 / C2_sum  # T2[i,j,k] = P(s_{t+2}=k | s_{t+1}=j, s_t=i)
    ll2 = float(np.sum(C2 * np.log(T2 + 1e-12)))

    # BIC
    p1 = K*(K-1)       # order-1 free params
    p2 = K*K*(K-1)     # order-2 free params
    N  = N1
    bic1 = -2*ll1 + p1*np.log(N+1e-12)
    bic2 = -2*ll2 + p2*np.log(N+1e-12)
    delta_bic = float(bic2 - bic1)  # positive = order-1 preferred

    # likelihood ratio test
    lr_stat = 2*(ll2 - ll1)
    df      = p2 - p1
    p_lrt   = float(1 - stats.chi2.cdf(lr_stat, df=df)) if df > 0 else 1.0

    return {
        "bic_order1":  float(bic1),
        "bic_order2":  float(bic2),
        "delta_bic":   delta_bic,      # >0 means order-1 Markov preferred
        "order1_preferred": delta_bic > 0,
        "lr_stat":     float(lr_stat),
        "lr_df":       int(df),
        "p_lrt":       p_lrt,          # p<0.05 means order-2 significantly better
        "markovian":   p_lrt > 0.05,   # fail to reject = Markovian
    }


# ── MAIN ─────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # load existing results to skip
    if os.path.exists(OUTPUT_PATH):
        results = json.load(open(OUTPUT_PATH))
    else:
        results = {}

    for path in DATASETS:
        if not os.path.exists(path):
            print(f"  Skipping - not found: {path}", flush=True); continue
        parts = path.split("/")
        tag   = f"{parts[-4]}_{parts[-3]}_{parts[-2]}"
        if tag in results:
            print(f"  Skipping - exists: {tag}", flush=True); continue

        print(f"\n{'='*50}\n{tag}\n{'='*50}", flush=True)
        np.random.seed(SEED); torch.manual_seed(SEED)

        try:
            seqs = load_and_embed(path)
        except Exception as e:
            print(f"  FAILED embed: {e}", flush=True); continue

        # get best K from pre-existing cebra_em results if available
        em_dir = "/home/abir19/scratch/abir19/SDS_results/cebra_em"
        json_path = os.path.join(em_dir, f"{tag}.json")
        if os.path.exists(json_path):
            K = json.load(open(json_path))["best_k"]["by_bic"]
            print(f"  Using best K={K} from existing results", flush=True)
        else:
            K = 5  # fallback
            print(f"  Using fallback K={K}", flush=True)

        try:
            np.random.seed(SEED + K)
            state_seqs, T1 = fit_slds_get_states(seqs, K)
        except Exception as e:
            print(f"  FAILED slds: {e}", flush=True); continue

        print(f"  Running CK test...", flush=True)
        ck_res, ck_p, ck_null_mean = chapman_kolmogorov_test(state_seqs, T1, K)

        print(f"  Running order-2 BIC test...", flush=True)
        o2 = order2_bic_test(state_seqs, K)

        result = {
            "tag": tag, "K": K,
            "ck_residual":      ck_res,
            "ck_p_value":       ck_p,
            "ck_null_mean":     ck_null_mean,
            "ck_markovian":     ck_p > 0.05,  # large residual relative to null -> non-Markovian
            **o2,
            "timestamp": datetime.now().isoformat(),
        }
        results[tag] = result

        print(f"  CK residual={ck_res:.4f} (null={ck_null_mean:.4f}) p={ck_p:.3f} "
              f"Markovian={result['ck_markovian']}", flush=True)
        print(f"  ΔBIC={o2['delta_bic']:.1f} order1_preferred={o2['order1_preferred']} "
              f"p_lrt={o2['p_lrt']:.4f} Markovian={o2['markovian']}", flush=True)

        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    # summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    rlvr = {k:v for k,v in results.items() if "base" not in k.lower() and "_Base" not in k}
    base = {k:v for k,v in results.items() if "base" in k.lower() or "_Base" in k}

    for cond, group in [("RLVR", rlvr), ("Base", base)]:
        if not group: continue
        ck_markov = np.mean([v["ck_markovian"] for v in group.values()])
        o2_markov = np.mean([v["markovian"]    for v in group.values()])
        ck_res    = np.mean([v["ck_residual"]  for v in group.values()])
        dbic      = np.mean([v["delta_bic"]    for v in group.values()])
        print(f"  {cond} (n={len(group)}): "
              f"CK Markovian={ck_markov:.0%}  Order2 Markovian={o2_markov:.0%}  "
              f"mean CK residual={ck_res:.4f}  mean ΔBIC={dbic:.1f}")