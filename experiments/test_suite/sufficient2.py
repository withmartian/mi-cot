"""
membership_matrix.py

For each dataset, builds the stage->state membership matrix:
    M[stage, state] = P(state | stage)  -- normalized over states
and its transpose:
    M[state, stage] = P(stage | state)  -- what does each regime correspond to

Uses the same CEBRA+SLDS pipeline as cebra_slds_sweep.py.
Saves results + heatmap figure.
"""

import os, json, pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

PCA_DIM  = 40
KAPPA    = 10.0
N_ITERS  = 50
CEBRA_EPOCHS = 100
BATCH_SIZE   = 1024

STAGES = [
    "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
]
STAGE_SHORT = ["SETUP", "RETRIEVAL", "PLAN", "UNCERT", "CHECK", "CONSOL", "COMPUTE", "ANSWER"]

DATASETS = [
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8b_reasoning/layer_22/all_sentences_features.pkl",  "rlvr_llama8b_L22"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8b_reasoning/layer_31/all_sentences_features.pkl",  "rlvr_llama8b_L31"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14b_reasoning/layer_28/all_sentences_features.pkl",  "rlvr_qwen14b_L28"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14b_reasoning/layer_47/all_sentences_features.pkl",  "rlvr_qwen14b_L47"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen1.5b_reasoning/layer_20/all_sentences_features.pkl",  "rlvr_qwen1.5b_L20"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen1.5b_reasoning/layer_27/all_sentences_features.pkl",  "rlvr_qwen1.5b_L27"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8B_base/layer_22/all_sentences_features.pkl",       "base_llama8b_L22"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8B_base/layer_31/all_sentences_features.pkl",       "base_llama8b_L31"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14B_base/layer_28/all_sentences_features.pkl",       "base_qwen14b_L28"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14B_base/layer_47/all_sentences_features.pkl",       "base_qwen14b_L47"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_1.5B_base/layer_20/all_sentences_features.pkl",      "base_qwen1.5b_L20"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_1.5B_base/layer_27/all_sentences_features.pkl",      "base_qwen1.5b_L27"),
]

OUTPUT_DIR = "/home/abir19/scratch/abir19/SDS_results/membership"

# ── CEBRA + SLDS (minimal, same as other scripts) ────────────

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
    pids, trips = list(p_map.keys()), []
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: continue
        for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_t), replace=False):
            neg = np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])
            trips.append((idxs[t], idxs[t+1], neg))
    return trips

def train_cebra(all_features, triplets):
    X_raw = np.array([f['hidden_state_last'] for f in all_features])
    X_sc  = StandardScaler().fit_transform(X_raw)
    X_t   = torch.tensor(X_sc, dtype=torch.float32).to(device)
    model = CEBRANet(X_raw.shape[1]).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    tarr  = np.array(triplets)
    for _ in range(CEBRA_EPOCHS):
        idx = np.random.permutation(len(tarr))
        for i in range(0, len(tarr), BATCH_SIZE):
            b  = tarr[idx[i:i+BATCH_SIZE]]
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
    return [np.array(p_map_z[p]) for p in pids], pids

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

def fit_slds(seqs, K):
    pi,A,dM,db,dCov = init_params(seqs, K)
    for _ in range(N_ITERS):
        gammas,xis,lls = [],[],[]
        for s in seqs:
            g,x,ll = forward_backward(s,pi,A,dM,db,dCov,K)
            gammas.append(g); xis.append(x); lls.append(ll)
        pi,A,dM,db,dCov = m_step(seqs,gammas,xis,K)
    return [np.argmax(g,1) for g in gammas]

# ── MEMBERSHIP MATRIX ────────────────────────────────────────

def build_membership(all_features, state_seqs, pids, K):
    """
    M_state_stage[k, s] = count(sentences in state k with stage s)
    Returns both P(stage|state) and P(state|stage), normalized.
    """
    pid_to_states = {pid: ss for pid, ss in zip(pids, state_seqs)}
    # build pid -> ordered list of (sentence_idx, stage)
    pid_stages = defaultdict(list)
    for f in all_features:
        if f['stage'] == 'NEUTRAL': continue
        pid_stages[f['problem_id']].append((f['sentence_idx'], f['stage']))

    M = np.zeros((K, len(STAGES)))  # [state, stage]
    for pid, entries in pid_stages.items():
        if pid not in pid_to_states: continue
        ss = pid_to_states[pid]
        entries_sorted = sorted(entries, key=lambda x: x[0])
        for i, (_, stage) in enumerate(entries_sorted):
            if i >= len(ss): break
            if stage not in STAGES: continue
            M[ss[i], STAGES.index(stage)] += 1

    # P(stage | state) — each row sums to 1
    p_stage_given_state = M / (M.sum(1, keepdims=True) + 1e-9)
    # P(state | stage) — each column sums to 1
    p_state_given_stage = M / (M.sum(0, keepdims=True) + 1e-9)

    return M, p_stage_given_state, p_state_given_stage

def plot_membership(p_stage_given_state, tag, K, out_path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(p_stage_given_state, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(STAGES))); ax.set_xticklabels(STAGE_SHORT, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"R{k}" for k in range(K)], fontsize=9)
    ax.set_xlabel("Reasoning Stage", fontsize=10)
    ax.set_ylabel("Regime", fontsize=10)
    ax.set_title(f"P(stage | regime) — {tag}", fontsize=11)
    for i in range(K):
        for j in range(len(STAGES)):
            v = p_stage_given_state[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                        fontsize=7, color='white' if v > 0.5 else 'black')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

# ── MAIN ─────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path, tag in DATASETS:
        out_json = os.path.join(OUTPUT_DIR, f"{tag}.json")
        out_fig  = os.path.join(OUTPUT_DIR, f"{tag}.png")

        if os.path.exists(out_json):
            print(f"  Skipping - already exists: {tag}", flush=True)
            continue
        if not os.path.exists(path):
            print(f"  Skipping - data not found: {path}", flush=True)
            continue

        print(f"\n{'='*50}\n{tag}\n{'='*50}", flush=True)

        all_features = [f for f in pickle.load(open(path,'rb'))
                        if f['problem_id'] < 500]

        K = 5 if tag.startswith("rlvr") else 3
        triplets  = make_triplets(all_features)
        cebra_seqs, pids = train_cebra(all_features, triplets)
        state_seqs = fit_slds(cebra_seqs, K)

        M, p_sg_st, p_st_sg = build_membership(all_features, state_seqs, pids, K)

        # print dominant stage per regime
        print(f"  P(stage | regime):", flush=True)
        for k in range(K):
            top = np.argmax(p_sg_st[k])
            print(f"    R{k}: {STAGES[top]:<30} ({p_sg_st[k,top]:.3f})", flush=True)

        plot_membership(p_sg_st, tag, K, out_fig)

        result = {
            "tag": tag, "K": K,
            "count_matrix":        M.tolist(),
            "p_stage_given_state": p_sg_st.tolist(),
            "p_state_given_stage": p_st_sg.tolist(),
            "dominant_stage_per_regime": [
                {"regime": k, "stage": STAGES[int(np.argmax(p_sg_st[k]))],
                 "prob": float(np.max(p_sg_st[k]))}
                for k in range(K)
            ],
        }
        with open(out_json, 'w') as f:
            import json; json.dump(result, f, indent=2)
        print(f"  Saved -> {out_json}", flush=True)