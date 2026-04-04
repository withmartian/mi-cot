"""
continuation_margin.py  —  optimized version
"""

import os, json, pickle, argparse, sys, random
import numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[init] device={device}", flush=True)

# ── SPEED LEVERS ─────────────────────────────────────────────
SEED         = 42
PCA_DIM      = 40
KAPPA        = 10.0
N_ITERS      = 50
CEBRA_EPOCHS = 50    # FIX: was 100, 50 is enough for embedding quality
BATCH_SIZE   = 2048  # FIX: larger batch = fewer steps
GAMMA_0      = 0.0
BETA         = 1.0
HORIZON      = 5
DISCOUNT     = 0.9

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

RLVR_DATASETS = [
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8b_reasoning/layer_22/all_sentences_features.pkl",  "rlvr", "llama8b",  "gsm8k",  "L22"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8b_reasoning/layer_31/all_sentences_features.pkl",  "rlvr", "llama8b",  "gsm8k",  "L31"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14b_reasoning/layer_28/all_sentences_features.pkl",  "rlvr", "qwen14b",  "gsm8k",  "L28"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14b_reasoning/layer_47/all_sentences_features.pkl",  "rlvr", "qwen14b",  "gsm8k",  "L47"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen1.5b_reasoning/layer_20/all_sentences_features.pkl",  "rlvr", "qwen1.5b", "gsm8k",  "L20"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen1.5b_reasoning/layer_27/all_sentences_features.pkl",  "rlvr", "qwen1.5b", "gsm8k",  "L27"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Llama_8B_reasoning/layer_22/all_sentences_features.pkl", "rlvr", "llama8b",  "math500","L22"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Llama_8B_reasoning/layer_31/all_sentences_features.pkl", "rlvr", "llama8b",  "math500","L31"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_reasoning/layer_28/all_sentences_features.pkl", "rlvr", "qwen14b",  "math500","L28"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_reasoning/layer_47/all_sentences_features.pkl", "rlvr", "qwen14b",  "math500","L47"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Qwen_1_5B_reasoning/layer_20/all_sentences_features.pkl","rlvr", "qwen1.5b", "math500","L20"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Qwen_1_5B_reasoning/layer_27/all_sentences_features.pkl","rlvr", "qwen1.5b", "math500","L27"),
]
BASE_DATASETS = [
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8B_base/layer_22/all_sentences_features.pkl",       "base", "llama8b",  "gsm8k",  "L22"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8B_base/layer_31/all_sentences_features.pkl",       "base", "llama8b",  "gsm8k",  "L31"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14B_base/layer_28/all_sentences_features.pkl",       "base", "qwen14b",  "gsm8k",  "L28"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14B_base/layer_47/all_sentences_features.pkl",       "base", "qwen14b",  "gsm8k",  "L47"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_1.5B_base/layer_20/all_sentences_features.pkl",      "base", "qwen1.5b", "gsm8k",  "L20"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_1.5B_base/layer_27/all_sentences_features.pkl",      "base", "qwen1.5b", "gsm8k",  "L27"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Llama_8B_base/layer_22/all_sentences_features.pkl",      "base", "llama8b",  "math500","L22"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Llama_8B_base/layer_31/all_sentences_features.pkl",      "base", "llama8b",  "math500","L31"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_base/layer_28/all_sentences_features.pkl",      "base", "qwen14b",  "math500","L28"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_base/layer_47/all_sentences_features.pkl",      "base", "qwen14b",  "math500","L47"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Qwen_1_5B_base/layer_20/all_sentences_features.pkl",     "base", "qwen1.5b", "math500","L20"),
    ("/home/abir19/scratch/abir19/SDS_math500_test/Qwen_1_5B_base/layer_27/all_sentences_features.pkl",     "base", "qwen1.5b", "math500","L27"),
]

# ── CEBRA ────────────────────────────────────────────────────

class CEBRANet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, PCA_DIM))
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

def make_triplets(all_features, max_t=25):
    p_map = defaultdict(list)
    for i, f in enumerate(all_features): p_map[f['problem_id']].append(i)
    pids  = list(p_map.keys())
    trips = []
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: continue
        for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_t), replace=False):
            neg = np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])
            trips.append((idxs[t], idxs[t+1], neg))
    return trips

def train_cebra(all_features, triplets):
    X_raw = np.array([f['hidden_state_last'] for f in all_features])
    print(f"    [cebra] X_raw={X_raw.shape}, triplets={len(triplets)}", flush=True)
    X_sc  = StandardScaler().fit_transform(X_raw)
    X_t   = torch.tensor(X_sc, dtype=torch.float32).to(device)
    torch.manual_seed(SEED)
    model = CEBRANet(X_raw.shape[1]).to(device)
    # FIX: use OneCycleLR for faster convergence
    opt   = optim.Adam(model.parameters(), lr=3e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CEBRA_EPOCHS)
    tarr  = np.array(triplets)
    for ep in range(CEBRA_EPOCHS):
        idx = np.random.permutation(len(tarr))
        for i in range(0, len(tarr), BATCH_SIZE):
            b  = tarr[idx[i:i+BATCH_SIZE]]
            za = model(X_t[b[:,0]]); zp = model(X_t[b[:,1]]); zn = model(X_t[b[:,2]])
            sp = torch.sum(za*zp,1)/0.1; sn = torch.sum(za*zn,1)/0.1
            loss = -torch.log(torch.exp(sp)/(torch.exp(sp)+torch.exp(sn))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep+1) % 10 == 0:
            print(f"    [cebra] ep {ep+1}/{CEBRA_EPOCHS} loss={loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        Z = model(X_t).cpu().numpy()
    p_map_z = defaultdict(list)
    for i, f in enumerate(all_features): p_map_z[f['problem_id']].append(Z[i])
    pids = sorted(p for p in p_map_z if len(p_map_z[p]) >= 3)
    print(f"    [cebra] done. {len(pids)} problems", flush=True)
    return [np.array(p_map_z[p]) for p in pids], pids

# ── SLDS EM ──────────────────────────────────────────────────

def init_params(seqs, K):
    X_in  = np.vstack([s[:-1] for s in seqs])
    X_out = np.vstack([s[1:]  for s in seqs])
    labs  = KMeans(n_clusters=K, n_init=5, random_state=SEED).fit_predict(X_out)
    D = PCA_DIM
    dM, db, dCov = np.zeros((K,D,D)), np.zeros((K,D)), np.array([np.eye(D)]*K)
    for k in range(K):
        m = labs==k
        if m.sum() < D+2: dM[k]=0.1*np.eye(D); continue
        W,*_ = np.linalg.lstsq(np.hstack([X_in[m],np.ones((m.sum(),1))]),X_out[m],rcond=None)
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
        lb[t] = np.logaddexp.reduce(lA+(log_emit[t+1]+lb[t+1])[None,:],1)
    lg = la+lb; lg -= np.logaddexp.reduce(lg,1,keepdims=True)
    lxi = np.zeros((T-1,K,K))
    for t in range(T-1):
        lxi[t] = la[t][:,None]+lA+(log_emit[t+1]+lb[t+1])[None,:]
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

def compute_bic(total_ll, K, N):
    n_params = K*(K-1)+K*(PCA_DIM**2+PCA_DIM+PCA_DIM*(PCA_DIM+1)//2)
    return -2*total_ll+n_params*np.log(N)

def fit_slds(seqs, K):
    print(f"    [slds] K={K} ...", flush=True)
    pi,A,dM,db,dCov = init_params(seqs, K)
    for it in range(N_ITERS):
        gammas,xis,lls = [],[],[]
        for s in seqs:
            g,x,ll = forward_backward(s,pi,A,dM,db,dCov,K)
            gammas.append(g); xis.append(x); lls.append(ll)
        pi,A,dM,db,dCov = m_step(seqs,gammas,xis,K)
    bic = compute_bic(sum(lls), K, sum(len(s) for s in seqs))
    state_seqs = [np.argmax(g,1) for g in gammas]
    print(f"    [slds] K={K} bic={bic:.2f}", flush=True)
    return state_seqs, pi, A, dM, db, dCov, bic

# ── VALUE FUNCTION (vectorized) ──────────────────────────────

def precompute_rollouts(dM, db, K, horizon):
    """
    FIX: precompute regime dynamics matrices for multi-step rollout.
    M_k^h = dM[k]^h,  b_k^h = sum_{i=0}^{h-1} dM[k]^i @ db[k]
    Returns: Ms[k,h] = dM[k]^h,  bs[k,h] = accumulated bias
    """
    D = PCA_DIM
    Ms = np.zeros((K, horizon+1, D, D))
    bs = np.zeros((K, horizon+1, D))
    for k in range(K):
        Ms[k,0] = np.eye(D); bs[k,0] = np.zeros(D)
        for h in range(1, horizon+1):
            Ms[k,h] = dM[k] @ Ms[k,h-1]
            bs[k,h] = dM[k] @ bs[k,h-1] + db[k]
    return Ms, bs  # Ms[k,h] @ z + bs[k,h] = predicted z at step h

def compute_continuation_margins_fast(seqs, state_seqs, dM, db, K,
                                       horizon=HORIZON, discount=DISCOUNT):
    """
    FIX: fully vectorized margin computation.
    Instead of Python loops over timesteps, stack all (z_t, future) pairs
    and compute V_k for all k at once via matrix ops.
    """
    # precompute rollout matrices
    Ms, bs = precompute_rollouts(dM, db, K, horizon)
    # discount weights per horizon step
    disc = np.array([discount**(h+1) for h in range(horizon)])  # (horizon,)

    all_gammas = []
    for seq, s_seq in zip(seqs, state_seqs):
        T = len(seq)
        if T < 2: continue

        # for each t, compute V_k(z_t) = -sum_h disc^h ||z_{t+h} - M_k^h z_t - b_k^h||^2
        for t in range(T - 1):
            H = min(horizon, T - t - 1)
            if H == 0: continue
            z_t    = seq[t]          # (D,)
            future = seq[t+1:t+1+H]  # (H, D)

            # predicted trajectories for all K regimes: (K, H, D)
            preds = Ms[:, 1:H+1] @ z_t + bs[:, 1:H+1]  # (K, H, D)

            # errors: (K, H, D)
            errs = future[None, :, :] - preds  # broadcast future over K

            # squared errors summed over D: (K, H)
            sq_errs = np.sum(errs**2, axis=2)

            # discounted value: (K,)
            V = -np.sum(disc[:H][None,:] * sq_errs, axis=1)

            i = s_seq[t]
            V_others = np.delete(V, i)
            gamma = float(V[i] - V_others.max()) if len(V_others) > 0 else 0.0
            all_gammas.append(gamma)

    return np.array(all_gammas)

# ── PERSISTENCE ──────────────────────────────────────────────

def actual_persistence(state_seqs):
    same, total = 0, 0
    for s in state_seqs:
        for a, b in zip(s[:-1], s[1:]):
            total += 1
            if a == b: same += 1
    return same / total if total > 0 else 0.0

def persistence_bound(gamma_bar, delta, K, beta=BETA):
    if K <= 1: return 1.0
    return (1 - delta) / (1 + (K-1) * np.exp(-gamma_bar / beta))

# ── MAIN EXPERIMENT ──────────────────────────────────────────

def run_one(path, condition, model_name, dataset, layer, out_dir):
    tag      = f"{condition}_{model_name}_{dataset}_{layer}"
    out_path = os.path.join(out_dir, f"{tag}.json")

    print(f"\n{'='*50}\n  {tag}\n{'='*50}", flush=True)

    if os.path.exists(out_path):
        print(f"  Skipping - exists", flush=True)
        return json.load(open(out_path))
    if not os.path.exists(path):
        print(f"  Skipping - data not found", flush=True)
        return None

    try:
        all_features = pickle.load(open(path, 'rb'))
        print(f"  {len(all_features)} features loaded", flush=True)
        triplets     = make_triplets(all_features)
        cebra_seqs, pids = train_cebra(all_features, triplets)
    except Exception as e:
        print(f"  FAILED embed: {e}", flush=True); return None

    # FIX: use best_k from existing cebra_em results to skip K sweep
    em_results_dir = "/home/abir19/scratch/abir19/SDS_results/cebra_em"
    parts = path.split("/")
    json_tag  = f"{parts[-4]}_{parts[-3]}_{parts[-2]}"
    json_path = os.path.join(em_results_dir, f"{json_tag}.json")
    if os.path.exists(json_path):
        K = json.load(open(json_path))["best_k"]["by_bic"]
        print(f"  Using cached best K={K}", flush=True)
        try:
            np.random.seed(SEED + K)
            state_seqs, pi, A, dM, db, dCov, bic = fit_slds(cebra_seqs, K)
        except Exception as e:
            print(f"  FAILED slds: {e}", flush=True); return None
    else:
        # FIX: only sweep K=3,4,5,6 instead of 2..7
        print(f"  No cached K — sweeping K=3..6", flush=True)
        bic_scores, fits = {}, {}
        for k in [3, 4, 5, 6]:
            try:
                np.random.seed(SEED + k)
                ss, pi, A, dM, db, dCov, bic = fit_slds(cebra_seqs, k)
                bic_scores[k] = bic; fits[k] = (ss, pi, A, dM, db, dCov)
            except Exception as e:
                print(f"  K={k} failed: {e}", flush=True)
        if not bic_scores:
            print(f"  FAILED: no valid K", flush=True); return None
        K = min(bic_scores, key=bic_scores.get)
        state_seqs, pi, A, dM, db, dCov = fits[K]
        print(f"  Best K={K}", flush=True)

    try:
        print(f"  Computing margins (vectorized)...", flush=True)
        margin_vals = compute_continuation_margins_fast(
            cebra_seqs, state_seqs, dM, db, K)
        print(f"  {len(margin_vals)} margins computed", flush=True)
    except Exception as e:
        print(f"  FAILED margins: {e}", flush=True); return None

    gamma_bar  = float(margin_vals.mean())
    delta      = float((margin_vals < GAMMA_0).mean())
    act_pers   = actual_persistence(state_seqs)
    pers_bound = persistence_bound(gamma_bar, delta, K)

    print(f"  γ̄={gamma_bar:.4f}  δ={delta:.4f}  "
          f"bound={pers_bound:.4f}  actual={act_pers:.4f}", flush=True)

    result = {
        "tag": tag, "condition": condition, "model": model_name,
        "dataset": dataset, "layer": layer, "K": K,
        "gamma_bar":          gamma_bar,
        "delta":              delta,
        "persistence_bound":  pers_bound,
        "actual_persistence": act_pers,
        "gamma_0": GAMMA_0, "beta": BETA, "horizon": HORIZON, "discount": DISCOUNT,
        "n_timesteps": len(margin_vals),
        "gamma_percentiles": {
            "p10": float(np.percentile(margin_vals, 10)),
            "p25": float(np.percentile(margin_vals, 25)),
            "p50": float(np.percentile(margin_vals, 50)),
            "p75": float(np.percentile(margin_vals, 75)),
            "p90": float(np.percentile(margin_vals, 90)),
        },
        "gamma_values": margin_vals.tolist(),
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_path}", flush=True)
    return result

# ── FIGURE ───────────────────────────────────────────────────

def plot_gamma_distributions(results_dir, out_dir):
    RLVR_C = '#ff563f'; BASE_C = '#55c89f'
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    by_model = defaultdict(lambda: {"rlvr": [], "base": []})
    for fname in files:
        d = json.load(open(f"{results_dir}/{fname}"))
        if "gamma_values" not in d: continue
        by_model[d["model"]][d["condition"]].extend(d["gamma_values"])
    models = sorted(by_model.keys())
    if not models: return
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    if len(models) == 1: axes = [axes]
    fig.patch.set_facecolor("white")
    for ax, model in zip(axes, models):
        ax.set_facecolor("white")
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5dfdf'); ax.spines['bottom'].set_color('#e5dfdf')
        for cond, color, label in [("rlvr", RLVR_C, "RLVR"), ("base", BASE_C, "Base")]:
            vals = np.array(by_model[model][cond])
            if len(vals) == 0: continue
            vals_clip = np.clip(vals, np.percentile(vals,1), np.percentile(vals,99))
            ax.hist(vals_clip, bins=60, color=color, alpha=0.6,
                    label=f"{label} (μ={vals.mean():.3f})", density=True, zorder=3)
            ax.axvline(vals.mean(), color=color, linewidth=2, linestyle='--', zorder=4)
        ax.axvline(0, color='#928e8b', linewidth=1, zorder=2)
        ax.set_xlabel("γ (continuation margin)", fontsize=14, fontweight='bold', color='#0c0c0c')
        ax.set_ylabel("Density", fontsize=14, fontweight='bold', color='#0c0c0c')
        ax.set_title(model, fontsize=15, fontweight='bold', color='#0c0c0c')
        ax.tick_params(labelsize=11, colors='#0c0c0c')
        ax.legend(fontsize=11, frameon=False)
        ax.grid(axis='y', color='#f5f5f5', linewidth=0.8, zorder=0)
    fig.suptitle("Continuation Margin: RLVR vs Base",
                 fontsize=13, fontweight='bold', color='#0c0c0c', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig_gamma_distribution.png", dpi=180,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved fig_gamma_distribution.png", flush=True)

def print_summary_table(results_dir):
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".json"))
    print(f"\n{'='*80}")
    print(f"{'Tag':<35} {'γ̄':>8} {'δ':>8} {'bound':>8} {'actual':>8}")
    print("-"*80)
    by_cond = {"rlvr": [], "base": []}
    for fname in files:
        d = json.load(open(f"{results_dir}/{fname}"))
        if "gamma_bar" not in d: continue
        print(f"  {d['tag']:<33} {d['gamma_bar']:>8.4f} {d['delta']:>8.4f} "
              f"{d['persistence_bound']:>8.4f} {d['actual_persistence']:>8.4f}")
        by_cond[d["condition"]].append(d)
    print("\n── Aggregate ──")
    for cond in ["rlvr", "base"]:
        rs = by_cond[cond]
        if not rs: continue
        print(f"  {cond.upper():5s}  "
              f"γ̄={np.mean([r['gamma_bar'] for r in rs]):.4f}  "
              f"δ={np.mean([r['delta'] for r in rs]):.4f}  "
              f"bound={np.mean([r['persistence_bound'] for r in rs]):.4f}  "
              f"actual={np.mean([r['actual_persistence'] for r in rs]):.4f}")

# ── MAIN ─────────────────────────────────────────────────────

def main():
    global HORIZON, DISCOUNT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/abir19/scratch/abir19/SDS_results/continuation")
    parser.add_argument("--horizon",  type=int,   default=HORIZON)
    parser.add_argument("--discount", type=float, default=DISCOUNT)
    args    = parser.parse_args()
    HORIZON  = args.horizon
    DISCOUNT = args.discount
    os.makedirs(args.out, exist_ok=True)
    print(f"[main] out={args.out}  horizon={HORIZON}  discount={DISCOUNT}", flush=True)

    for entry in RLVR_DATASETS + BASE_DATASETS:
        try:
            run_one(*entry, args.out)
        except Exception as e:
            print(f"  UNHANDLED: {e}", flush=True)

    print_summary_table(args.out)
    plot_gamma_distributions(args.out, args.out)
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()