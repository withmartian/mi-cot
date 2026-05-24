"""
sufficiency_gsm8k.py

Loads all_sentences_features.pkl + cot_data.pkl per dataset,
extracts per-problem switching metrics from CEBRA+SLDS,
matches CoT answers against GSM8K ground truth,
and correlates switching structure with correctness — RLVR vs base.

Usage:
    python sufficiency_gsm8k.py --out /path/to/output
"""

import os, re, json, pickle, argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

N_PROBLEMS = 200  # set higher for full run
FIXED_K_RLVR = 5
FIXED_K_BASE = 3

PCA_DIM      = 40
KAPPA        = 10.0
N_ITERS      = 50
CEBRA_EPOCHS = 100
BATCH_SIZE   = 1024

# (features_pkl_dir, tag, condition)
DATASETS = [
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8b_reasoning/layer_22",  "llama8b_L22",  "rlvr"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8b_reasoning/layer_31",  "llama8b_L31",  "rlvr"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14b_reasoning/layer_28",  "qwen14b_L28",  "rlvr"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14b_reasoning/layer_47",  "qwen14b_L47",  "rlvr"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen1.5b_reasoning/layer_20",  "qwen1.5b_L20", "rlvr"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen1.5b_reasoning/layer_27",  "qwen1.5b_L27", "rlvr"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8B_base/layer_22",       "llama8b_L22",  "base"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8B_base/layer_31",       "llama8b_L31",  "base"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14B_base/layer_28",       "qwen14b_L28",  "base"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14B_base/layer_47",       "qwen14b_L47",  "base"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_1.5B_base/layer_20",      "qwen1.5b_L20", "base"),
    ("/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_1.5B_base/layer_27",      "qwen1.5b_L27", "base"),
]

# ── ANSWER PARSING ───────────────────────────────────────────

def extract_number(text):
    text = text.replace(",", "")
    m = re.search(r'####\s*([-\d\.]+)', text)
    if m: return m.group(1).strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else None

def is_correct(cot, gt_answer):
    pred = extract_number(cot)
    gt   = extract_number(gt_answer)
    if pred is None or gt is None: return None
    try:
        return abs(float(pred) - float(gt)) < 1e-3
    except:
        return None

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
    X_pca = PCA(n_components=PCA_DIM, random_state=42).fit_transform(X_sc)
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
    p_map_z, p_map_p = defaultdict(list), defaultdict(list)
    for i, f in enumerate(all_features):
        pid = f['problem_id']
        p_map_z[pid].append(Z[i]); p_map_p[pid].append(X_pca[i])
    pids = sorted(p for p in p_map_z if len(p_map_z[p]) >= 3)
    return [np.array(p_map_z[p]) for p in pids], pids

# ── SLDS ─────────────────────────────────────────────────────

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

def compute_bic(total_ll, K, N):
    n_params = K*(K-1)+K*(PCA_DIM**2+PCA_DIM+PCA_DIM*(PCA_DIM+1)//2)
    return -2*total_ll+n_params*np.log(N)

def fit_slds(seqs, K):
    pi,A,dM,db,dCov = init_params(seqs, K)
    for _ in range(N_ITERS):
        gammas,xis,lls = [],[],[]
        for s in seqs:
            g,x,ll = forward_backward(s,pi,A,dM,db,dCov,K)
            gammas.append(g); xis.append(x); lls.append(ll)
        pi,A,dM,db,dCov = m_step(seqs,gammas,xis,K)
    bic = compute_bic(sum(lls), K, sum(len(s) for s in seqs))
    return [np.argmax(g,1) for g in gammas], bic

# ── PER-PROBLEM METRICS ──────────────────────────────────────

STAGES = [
    "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
]

def problem_metrics(state_seq, K, n_sentences):
    n_sw   = int(np.count_nonzero(np.diff(state_seq)))
    counts = np.bincount(state_seq, minlength=K)
    probs  = counts / counts.sum()
    return {
        "persistence":       float(len(state_seq) / (n_sw + 1)),
        "k_eff":             int(np.sum(probs > 0.01)),
        "n_switches":        n_sw,
        "switch_rate":       float(n_sw / max(n_sentences - 1, 1)),  # switches per transition
        "entropy":           float(-np.sum(probs * np.log(probs + 1e-12))),
        "n_sentences":       n_sentences,
    }

def regime_sequence_analysis(state_seqs, stage_seqs, correct_labels, K, window=3):
    """
    For each problem, extract the regime sequence in the last `window` steps
    before FINAL_ANSWER_EMISSION. Compare correct vs incorrect.
    Also compute: which regime is most common in correct vs incorrect problems.
    """
    pre_answer_correct   = []
    pre_answer_incorrect = []
    last_regime_correct   = []
    last_regime_incorrect = []

    for s_seq, l_seq, correct in zip(state_seqs, stage_seqs, correct_labels):
        # find last FINAL_ANSWER_EMISSION
        fa_idx = None
        for i in range(len(l_seq)-1, -1, -1):
            if l_seq[i] == "FINAL_ANSWER_EMISSION":
                fa_idx = i; break
        if fa_idx is None or fa_idx < window: continue

        pre = list(s_seq[max(0, fa_idx-window):fa_idx])
        last = int(s_seq[min(fa_idx, len(s_seq)-1)])
        if correct:
            pre_answer_correct.append(pre)
            last_regime_correct.append(last)
        else:
            pre_answer_incorrect.append(pre)
            last_regime_incorrect.append(last)

    def regime_dist(seqs, K):
        counts = np.zeros(K)
        for s in seqs: counts[s] += 1
        return (counts / counts.sum()).tolist() if counts.sum() > 0 else [0]*K

    return {
        "pre_answer_regime_dist_correct":   regime_dist([s for seq in pre_answer_correct   for s in seq], K),
        "pre_answer_regime_dist_incorrect": regime_dist([s for seq in pre_answer_incorrect for s in seq], K),
        "last_regime_dist_correct":         regime_dist(last_regime_correct,   K),
        "last_regime_dist_incorrect":       regime_dist(last_regime_incorrect, K),
        "n_correct_with_fa":   len(pre_answer_correct),
        "n_incorrect_with_fa": len(pre_answer_incorrect),
    }

# ── MAIN ─────────────────────────────────────────────────────

def run_one(data_dir, tag, condition, gsm8k_gt, out_dir):
    out_path = os.path.join(out_dir, f"{condition}_{tag}.json")
    if os.path.exists(out_path):
        print(f"  Skipping - already exists: {out_path}", flush=True)
        return

    feat_path = os.path.join(data_dir, "all_sentences_features.pkl")
    cot_path  = os.path.join(data_dir, "cot_data.pkl")

    if not os.path.exists(feat_path) or not os.path.exists(cot_path):
        print(f"  Skipping - missing files in {data_dir}", flush=True)
        return

    print(f"\n{'='*50}\n{condition} {tag}\n{'='*50}", flush=True)

    all_features = pickle.load(open(feat_path, 'rb'))
    cot_data     = pickle.load(open(cot_path,  'rb'))
    # limit to N_PROBLEMS
    pids_keep    = sorted(set(f['problem_id'] for f in all_features))[:N_PROBLEMS]
    pids_keep    = set(pids_keep)
    all_features = [f for f in all_features if f['problem_id'] in pids_keep]
    cot_data     = {pid: d for pid, d in cot_data.items() if pid in pids_keep}
    # cot_data: {pid: {problem, cot, sentences}}

    # build correctness per pid
    pid_to_correct = {}
    for pid, d in cot_data.items():
        problem_text = d['problem'].strip()
        gt = gsm8k_gt.get(problem_text) or gsm8k_gt.get(problem_text[:120])
        if gt is None: continue
        correct = is_correct(d['cot'], gt)
        if correct is not None:
            pid_to_correct[pid] = int(correct)

    n_matched = len(pid_to_correct)
    print(f"  Matched {n_matched} / {len(cot_data)} problems to GSM8K", flush=True)
    if n_matched < max(5, N_PROBLEMS // 2):
        print(f"  Too few matches, skipping", flush=True)
        return

    acc = np.mean(list(pid_to_correct.values()))
    print(f"  Accuracy: {acc:.3f}", flush=True)

    # CEBRA + SLDS
    triplets = make_triplets(all_features)
    cebra_seqs, pids = train_cebra(all_features, triplets)

    fixed_k = FIXED_K_RLVR if condition == "rlvr" else FIXED_K_BASE
    if fixed_k is not None:
        best_k     = fixed_k
        state_seqs, _ = fit_slds(cebra_seqs, best_k)
        print(f"  Fixed K={best_k}", flush=True)
    else:
        bic_scores, state_seqs_by_k = {}, {}
        for k in range(2, 10):
            ss, bic = fit_slds(cebra_seqs, k)
            bic_scores[k] = bic; state_seqs_by_k[k] = ss
        best_k     = int(min(bic_scores, key=bic_scores.get))
        state_seqs = state_seqs_by_k[best_k]
        print(f"  Best K by BIC={best_k}", flush=True)

    # per-problem metrics for matched problems only
    records = []
    for i, pid in enumerate(pids):
        if pid not in pid_to_correct: continue
        n_sent = len(state_seqs[i])
        m = problem_metrics(state_seqs[i], best_k, n_sent)
        m['correct'] = pid_to_correct[pid]
        # store stage sequence for regime analysis (from all_features)
        m['state_seq'] = state_seqs[i].tolist()
        pid_stages = [f['stage'] for f in all_features if f['problem_id'] == pid]
        m['stage_seq'] = pid_stages
        records.append(m)

    if len(records) < max(5, N_PROBLEMS // 2):
        print(f"  Too few matched problems after SLDS ({len(records)}), skipping", flush=True)
        return

    correct     = np.array([r['correct']      for r in records])
    persist     = np.array([r['persistence']  for r in records])
    keff        = np.array([r['k_eff']        for r in records])
    switches    = np.array([r['n_switches']   for r in records])
    switch_rate = np.array([r['switch_rate']  for r in records])
    entropy     = np.array([r['entropy']      for r in records])
    n_sents     = np.array([r['n_sentences']  for r in records])

    def pb_corr(x):
        r, p = stats.pointbiserialr(correct, x)
        return float(r), float(p)

    def partial_corr_control_length(x):
        """Partial point-biserial: correlate x with correct controlling for n_sentences."""
        # residualize both x and correct (as float) on n_sentences
        from numpy.linalg import lstsq
        L = n_sents.reshape(-1,1)
        x_res = x - (L @ lstsq(L, x, rcond=None)[0])
        c_res = correct.astype(float) - (L @ lstsq(L, correct.astype(float), rcond=None)[0])
        r, p  = stats.pearsonr(x_res, c_res)
        return float(r), float(p)

    def group_means(x):
        return float(x[correct==1].mean()), float(x[correct==0].mean())

    # regime sequence analysis
    state_seqs_matched = [np.array(r['state_seq']) for r in records]
    stage_seqs_matched = [r['stage_seq']            for r in records]
    reg_analysis = regime_sequence_analysis(
        state_seqs_matched, stage_seqs_matched, correct, best_k)

    result = {
        "condition": condition, "tag": tag,
        "best_k": best_k, "n": len(records), "accuracy": float(acc),
        "correlations": {
            "persistence":  {"r": pb_corr(persist)[0],      "p": pb_corr(persist)[1]},
            "k_eff":        {"r": pb_corr(keff)[0],         "p": pb_corr(keff)[1]},
            "n_switches":   {"r": pb_corr(switches)[0],     "p": pb_corr(switches)[1]},
            "switch_rate":  {"r": pb_corr(switch_rate)[0],  "p": pb_corr(switch_rate)[1]},
            "entropy":      {"r": pb_corr(entropy)[0],      "p": pb_corr(entropy)[1]},
        },
        "partial_correlations_ctrl_length": {
            "n_switches":  {"r": partial_corr_control_length(switches)[0],
                            "p": partial_corr_control_length(switches)[1]},
            "switch_rate": {"r": partial_corr_control_length(switch_rate)[0],
                            "p": partial_corr_control_length(switch_rate)[1]},
            "k_eff":       {"r": partial_corr_control_length(keff)[0],
                            "p": partial_corr_control_length(keff)[1]},
        },
        "means_correct_vs_incorrect": {
            "persistence":  group_means(persist),
            "k_eff":        group_means(keff),
            "n_switches":   group_means(switches),
            "switch_rate":  group_means(switch_rate),
            "n_sentences":  group_means(n_sents),
            "entropy":      group_means(entropy),
        },
        "regime_sequence_analysis": reg_analysis,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"  n_correct={correct.sum()}  n_incorrect={(1-correct).sum()}", flush=True)
    print(f"  CoT length: correct={n_sents[correct==1].mean():.1f}  incorrect={n_sents[correct==0].mean():.1f}", flush=True)
    for m, x in [("persistence", persist), ("k_eff", keff), ("n_switches", switches),
                 ("switch_rate", switch_rate), ("entropy", entropy)]:
        r, p   = pb_corr(x)
        rp, pp = partial_corr_control_length(x)
        mc, mi = group_means(x)
        print(f"    {m:<14} correct={mc:.3f} incorrect={mi:.3f}  r={r:+.3f} p={p:.4f}  "
              f"partial_r={rp:+.3f} p={pp:.4f}", flush=True)
    print(f"  Regime seq analysis: n_correct_with_FA={reg_analysis['n_correct_with_fa']}  "
          f"n_incorrect_with_FA={reg_analysis['n_incorrect_with_fa']}", flush=True)

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/abir19/scratch/abir19/SDS_results/sufficiency")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset
    print("Loading GSM8K ground truth...", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    # two lookup keys: full text and 120-char prefix
    gsm8k_gt = {}
    for row in ds:
        q = row['question'].strip()
        gsm8k_gt[q]       = row['answer']
        gsm8k_gt[q[:120]] = row['answer']
    print(f"  {len(ds)} problems loaded\n", flush=True)

    for data_dir, tag, condition in DATASETS:
        run_one(data_dir, tag, condition, gsm8k_gt, args.out)

    # aggregate
    print(f"\n{'='*60}\nAGGREGATE\n{'='*60}", flush=True)
    for condition in ["rlvr", "base"]:
        files = [f for f in os.listdir(args.out)
                 if f.startswith(condition) and f.endswith(".json")]
        if not files: continue
        rs = {}
        for fname in files:
            d = json.load(open(os.path.join(args.out, fname)))
            for m, v in d["correlations"].items():
                if m not in rs: rs[m] = []
                rs[m].append(v["r"])
            if "partial_correlations_ctrl_length" in d:
                for m, v in d["partial_correlations_ctrl_length"].items():
                    key = f"{m}_partial"
                    if key not in rs: rs[key] = []
                    rs[key].append(v["r"])
        print(f"  {condition.upper()}")
        for m, vals in rs.items():
            print(f"    {m:<14} mean r={np.mean(vals):+.3f}  std={np.std(vals):.3f}  "
                  f"n_pos={sum(v>0 for v in vals)}/{len(vals)}")

if __name__ == "__main__":
    main()