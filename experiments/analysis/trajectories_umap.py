"""
fig_cebra_trajectories.py  — one figure per (model, dataset, condition)

Each figure shows ONE trajectory on the UMAP background.
Clean, focused, paper-ready.
"""

import os, json, pickle, argparse, random, gc
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

device  = "cuda" if torch.cuda.is_available() else "cpu"
PCA_DIM = 40; CEBRA_EPOCHS = 100; BATCH_SIZE = 1024; KAPPA = 1.0; N_ITERS = 50
MAX_UMAP = 20000

REGIME_COLORS = ['#ff563f','#6e73ff','#ffccc5','#848484','#55c89f',
                 '#277a63','#bec9ff','#ffd24d','#c07d20','#ff9e80']
RLVR_C = '#ff563f'; BASE_C = '#55c89f'; TEXT_C = '#0c0c0c'
MODEL_LABELS = {"llama8b": "Llama-8B", "qwen14b": "Qwen-14B", "qwen1.5b": "Qwen-1.5B"}

RUNS = [
    ("rlvr","llama8b",  "gsm8k",  "L31", "/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8b_reasoning/layer_31/all_sentences_features.pkl"),
    ("rlvr","qwen14b",  "gsm8k",  "L47", "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14b_reasoning/layer_47/all_sentences_features.pkl"),
    ("rlvr","qwen1.5b", "gsm8k",  "L27", "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen1.5b_reasoning/layer_27/all_sentences_features.pkl"),
    ("rlvr","llama8b",  "math500","L31", "/home/abir19/scratch/abir19/SDS_math500_test/Llama_8B_reasoning/layer_31/all_sentences_features.pkl"),
    ("rlvr","qwen14b",  "math500","L47", "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_reasoning/layer_47/all_sentences_features.pkl"),
    ("rlvr","qwen1.5b", "math500","L27", "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_1_5B_reasoning/layer_27/all_sentences_features.pkl"),
    ("base","llama8b",  "gsm8k",  "L31", "/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8B_base/layer_31/all_sentences_features.pkl"),
    ("base","qwen14b",  "gsm8k",  "L47", "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14B_base/layer_47/all_sentences_features.pkl"),
    ("base","qwen1.5b", "gsm8k",  "L27", "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_1.5B_base/layer_27/all_sentences_features.pkl"),
    ("base","llama8b",  "math500","L31", "/home/abir19/scratch/abir19/SDS_math500_test/Llama_8B_base/layer_31/all_sentences_features.pkl"),
    ("base","qwen14b",  "math500","L47", "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_14B_base/layer_47/all_sentences_features.pkl"),
    ("base","qwen1.5b", "math500","L27", "/home/abir19/scratch/abir19/SDS_math500_test/Qwen_1_5B_base/layer_27/all_sentences_features.pkl"),
]

# ── CEBRA ─────────────────────────────────────────────────────

class CEBRANet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, PCA_DIM))
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

def embed_cebra(all_features, limit=500, max_triplets=25):
    all_features = [f for f in all_features if f['problem_id'] < limit]
    p_map = defaultdict(list)
    for i, f in enumerate(all_features): p_map[f['problem_id']].append(i)
    pids  = list(p_map.keys())
    trips = []
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2: continue
        for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_triplets), replace=False):
            neg = np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])
            trips.append((idxs[t], idxs[t+1], neg))
    X_raw = np.array([f['hidden_state_last'] for f in all_features])
    X_sc  = StandardScaler().fit_transform(X_raw)
    X_t   = torch.tensor(X_sc, dtype=torch.float32).to(device)
    torch.manual_seed(SEED)
    model = CEBRANet(X_raw.shape[1]).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    tarr  = np.array(trips)
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
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    p_map_seq = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map_seq[f['problem_id']].append((f['sentence_idx'], i))
    pids_valid = sorted(p for p in p_map_seq if len(p_map_seq[p]) >= 3)
    seqs_Z, seqs_idx = [], []
    for pid in pids_valid:
        ordered = sorted(p_map_seq[pid], key=lambda x: x[0])
        seqs_idx.append([i for _, i in ordered])
        seqs_Z.append(Z[seqs_idx[-1]])
    return Z, seqs_Z, seqs_idx

def fit_states(seqs_Z, K):
    X_in  = np.vstack([s[:-1] for s in seqs_Z])
    X_out = np.vstack([s[1:]-s[:-1] for s in seqs_Z])
    labs  = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(X_out)
    D = PCA_DIM
    dM, db, dCov = np.zeros((K,D,D)), np.zeros((K,D)), np.array([np.eye(D)]*K)
    for k in range(K):
        m = labs==k
        if m.sum() < D+2: dM[k]=0.1*np.eye(D); continue
        W,*_ = np.linalg.lstsq(np.hstack([X_in[m],np.ones((m.sum(),1))]),X_out[m],rcond=None)
        dM[k],db[k] = W[:D].T, W[D]
        res = X_out[m]-(X_in[m]@dM[k].T+db[k])
        dCov[k] = np.cov(res.T)+1e-3*np.eye(D)
    pi = np.ones(K)/K; A = np.eye(K)*0.7+0.3/K

    def fb(seq):
        T = len(seq); le = np.zeros((T,K))
        for k in range(K):
            _,ld=np.linalg.slogdet(dCov[k]); ic=np.linalg.inv(dCov[k])
            mn=np.vstack([db[k],seq[:-1]@dM[k].T+db[k]]); df=seq-mn
            le[:,k]=-0.5*(D*np.log(2*np.pi)+ld+np.sum((df@ic)*df,1))
        lA=np.log(A+1e-12); lp=np.log(pi+1e-12)
        la=np.zeros((T,K)); la[0]=lp+le[0]
        for t in range(1,T): la[t]=le[t]+np.logaddexp.reduce(la[t-1][:,None]+lA,0)
        lb=np.zeros((T,K))
        for t in range(T-2,-1,-1): lb[t]=np.logaddexp.reduce(lA+le[t+1]+lb[t+1],1)
        lg=la+lb; lg-=np.logaddexp.reduce(lg,1,keepdims=True)
        lxi=np.zeros((T-1,K,K))
        for t in range(T-1):
            lxi[t]=la[t][:,None]+lA+le[t+1]+lb[t+1]
            lxi[t]-=np.logaddexp.reduce(lxi[t].ravel())
        return np.exp(lg), np.exp(lxi)

    for _ in range(N_ITERS):
        gammas,xis = [],[]
        for s in seqs_Z:
            g,x = fb(s); gammas.append(g); xis.append(x)
        xi_sum = sum(x.sum(0) for x in xis)+np.eye(K)*KAPPA+1e-8
        A = xi_sum/xi_sum.sum(1,keepdims=True)
        pi = np.maximum(np.mean([g[0] for g in gammas],0),1e-8); pi/=pi.sum()
        for k in range(K):
            Ws,WY = np.zeros((D+1,D+1)),np.zeros((D+1,D))
            for seq,g in zip(seqs_Z,gammas):
                Xa=np.hstack([np.vstack([np.zeros(D),seq[:-1]]),np.ones((len(seq),1))])
                w=g[:,k]; Ws+=(Xa*w[:,None]).T@Xa; WY+=(Xa*w[:,None]).T@seq
            c=np.linalg.solve(Ws+1e-4*np.eye(D+1),WY)
            dM[k],db[k]=c[:D].T,c[D]
            num,den=np.zeros((D,D)),1e-9
            for seq,g in zip(seqs_Z,gammas):
                err=seq-(np.vstack([np.zeros(D),seq[:-1]])@dM[k].T+db[k])
                num+=(err*g[:,k][:,None]).T@err; den+=g[:,k].sum()
            dCov[k]=num/den+1e-4*np.eye(D)
    return [np.argmax(g, axis=1) for g in gammas]

def pick_best_problem(state_seqs, min_sw=3):
    """Pick the single most interesting trajectory: most switches, diverse regimes."""
    for min_sw in [5, 4, 3, 2, 1, 0]:
        candidates = [
            (i, int(np.count_nonzero(np.diff(st))), len(set(st.tolist())))
            for i, st in enumerate(state_seqs)
            if np.count_nonzero(np.diff(st)) >= min_sw
            and 8 <= len(st) <= 50
        ]
        if candidates:
            candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
            return candidates[0][0]
    return 0

def process_run(run, results_dir, out_dir, limit=500):
    cond, model, dataset, layer, path = run
    tag   = f"{cond}_{model}_{dataset}_{layer}"
    cache = os.path.join(out_dir, f"{tag}_single_cache.npz")

    if os.path.exists(cache):
        c = np.load(cache, allow_pickle=True)
        return (c["umap2d"], c["all_states"],
                list(c["seqs_idx"]), int(c["best_id"]), int(c["K"]))

    print(f"  [{tag}] computing...", flush=True)
    all_features = pickle.load(open(path,'rb'))

    em_dir = os.path.join(results_dir, "cebra_em" if cond=="rlvr" else "cebra_em_base")
    K = 5
    if os.path.exists(em_dir):
        for jf in os.listdir(em_dir):
            jl = jf.lower()
            if (model.replace(".","") in jl and
                dataset.replace("-","").replace("500","") in jl and
                layer.lower().replace("l","layer_") in jl):
                try: K = json.load(open(f"{em_dir}/{jf}"))["best_k"]["by_bic"]; break
                except: pass

    np.random.seed(SEED); torch.manual_seed(SEED)
    Z, seqs_Z, seqs_idx = embed_cebra(all_features, limit=limit)
    state_seqs = fit_states(seqs_Z, K)
    best_id    = pick_best_problem(state_seqs)

    if len(Z) > MAX_UMAP:
        sub = np.random.choice(len(Z), MAX_UMAP, replace=False)
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
        reducer.fit(Z[sub])
        chunk = 10000
        umap2d = np.zeros((len(Z), 2), dtype=np.float32)
        for ci in range(0, len(Z), chunk):
            umap2d[ci:ci+chunk] = reducer.transform(Z[ci:ci+chunk])
        del reducer
    else:
        from umap import UMAP
        umap2d = UMAP(n_components=2, random_state=SEED,
                      n_neighbors=30, min_dist=0.1).fit_transform(Z)

    all_states = np.zeros(len(Z), dtype=int)
    for idx_list, st in zip(seqs_idx, state_seqs):
        for gi, s in zip(idx_list, st): all_states[gi] = s

    np.savez(cache, umap2d=umap2d, all_states=all_states,
             seqs_idx=np.array(seqs_idx, dtype=object),
             best_id=best_id, K=K)
    del Z, seqs_Z; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  [{tag}] K={K} best_id={best_id} n_switches={np.count_nonzero(np.diff(state_seqs[best_id]))}", flush=True)
    return umap2d, all_states, seqs_idx, best_id, K

# ── SINGLE FIGURE ─────────────────────────────────────────────

def make_figure(umap2d, all_states, seqs_idx, best_id, K, run):
    cond, model, dataset, layer, _ = run
    pts = umap2d[seqs_idx[best_id]]
    st  = all_states[seqs_idx[best_id]]
    T   = len(pts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.08})
    fig.patch.set_facecolor('white')

    # ── LEFT: UMAP + trajectory ──
    ax = axes[0]
    ax.set_facecolor('white')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

    # background: all points, lightly colored by regime
    for k in range(K):
        mask = all_states == k
        ax.scatter(umap2d[mask,0], umap2d[mask,1],
                   c=REGIME_COLORS[k % len(REGIME_COLORS)],
                   s=3, alpha=0.12, linewidths=0, zorder=1)

    # zoom to trajectory + generous padding
    pad_x = (pts[:,0].max()-pts[:,0].min()) * 0.6 + 1.5
    pad_y = (pts[:,1].max()-pts[:,1].min()) * 0.6 + 1.5
    ax.set_xlim(pts[:,0].min()-pad_x, pts[:,0].max()+pad_x)
    ax.set_ylim(pts[:,1].min()-pad_y, pts[:,1].max()+pad_y)

    # ── build regime runs ──
    runs = []  # (regime, start_t, end_t, mean_x, mean_y)
    t = 0
    while t < T:
        k = int(st[t]); t2 = t
        while t2 < T and int(st[t2]) == k: t2 += 1
        xs = pts[t:t2, 0].mean(); ys = pts[t:t2, 1].mean()
        runs.append((k, t, t2-1, xs, ys))
        t = t2

    # ── anchor nodes only: first run, last run, + transition points ──
    # always include first and last; for middle ones only keep if
    # the spatial jump from previous anchor is large enough to be visible
    anchors = [0]
    for i in range(1, len(runs)-1):
        _, _, _, x_prev, y_prev = runs[anchors[-1]]
        _, _, _, x_cur,  y_cur  = runs[i]
        dist = np.sqrt((x_cur-x_prev)**2 + (y_cur-y_prev)**2)
        if dist > 0.8:   # only keep if meaningfully far from last anchor
            anchors.append(i)
    if len(runs) > 1:
        anchors.append(len(runs)-1)
    anchors = sorted(set(anchors))

    anchor_pts = [(runs[i][0], runs[i][1], runs[i][2],
                   runs[i][3], runs[i][4]) for i in anchors]

    # ── draw curved arrows between anchors ──
    # use connectionstyle arc3,rad to curve arrows so crossings are visible
    import matplotlib.patches as mpatches2
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patheffects as pfx

    for idx in range(len(anchor_pts)-1):
        k0, ts0, te0, x0, y0 = anchor_pts[idx]
        k1, ts1, te1, x1, y1 = anchor_pts[idx+1]
        col = REGIME_COLORS[k0 % len(REGIME_COLORS)]
        # alternate curve direction to untangle crossings
        rad = 0.25 * (1 if idx % 2 == 0 else -1)
        ax.annotate("",
                    xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=col,
                        lw=2.2,
                        mutation_scale=16,
                        alpha=0.88,
                        shrinkA=10, shrinkB=10,
                        connectionstyle=f"arc3,rad={rad}",
                    ),
                    zorder=3)
        # order label at midpoint of arrow
        mx = (x0+x1)/2; my = (y0+y1)/2
        ax.text(mx, my, str(idx+1), fontsize=6, ha='center', va='center',
                color='white', fontweight='bold', zorder=6,
                bbox=dict(boxstyle='circle,pad=0.15', fc='#333',
                          ec='none', alpha=0.7))

    # ── draw anchor nodes — fixed size, labeled with regime + step range ──
    NODE_SIZE = 320
    for idx, (k, t_start, t_end, x, y) in enumerate(anchor_pts):
        col     = REGIME_COLORS[k % len(REGIME_COLORS)]
        run_len = t_end - t_start + 1
        label   = f"R{k}" if run_len == 1 else f"R{k}\n{t_start+1}–{t_end+1}"
        ax.scatter(x, y, s=NODE_SIZE, c=col, alpha=0.95,
                   linewidths=1.8, edgecolors='white', zorder=4)
        ax.text(x, y, label, fontsize=6, ha='center', va='center',
                color='white', fontweight='bold', zorder=5,
                multialignment='center')

    # START/END use regime color of their respective step
    col_start = REGIME_COLORS[int(st[0])  % len(REGIME_COLORS)]
    col_end   = REGIME_COLORS[int(st[-1]) % len(REGIME_COLORS)]
    ax.scatter(*pts[0],  s=420, marker='o', facecolors=col_start,
               edgecolors='white', linewidths=2.5, zorder=7)
    ax.scatter(*pts[-1], s=480, marker='*', facecolors=col_end,
               edgecolors='white', linewidths=2, zorder=7)
    ax.text(pts[0,0],  pts[0,1]-0.9,  "START", fontsize=8,
            ha='center', color=col_start, fontweight='bold', zorder=8)
    ax.text(pts[-1,0], pts[-1,1]-0.9, "END",   fontsize=8,
            ha='center', color=col_end, fontweight='bold', zorder=8)

    # no border — clean look
    for sp in ax.spines.values(): sp.set_visible(False)

    n_sw = int(np.count_nonzero(np.diff(st)))
    ax.set_title(f"CEBRA Embedding — {MODEL_LABELS[model]} · "
                 f"{'RLVR' if cond=='rlvr' else 'Base'} · {dataset.upper()} · {layer}\n"
                 f"{T} reasoning steps · {n_sw} regime switches",
                 fontsize=12, fontweight='bold', color=TEXT_C, pad=8)

    # regime legend inside plot
    handles = [mpatches.Patch(facecolor=REGIME_COLORS[k], edgecolor='white',
                               linewidth=1, label=f"Regime {k}")
               for k in range(K)]
    ax.legend(handles=handles, fontsize=9, frameon=False,
              loc='lower left', ncol=2)

    # ── RIGHT: regime sequence strip ──
    ax2 = axes[1]
    ax2.set_facecolor('white')
    ax2.set_xlim(-0.3, 1.3)
    ax2.set_ylim(-0.5, T-0.5)
    ax2.set_xticks([]); ax2.invert_yaxis()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_color('#ddd')
    ax2.set_ylabel("Reasoning step", fontsize=10, color='#555')

    for t in range(T):
        col  = REGIME_COLORS[int(st[t]) % len(REGIME_COLORS)]
        rect = mpatches.FancyBboxPatch(
            (0.05, t-0.38), 0.9, 0.76,
            boxstyle="round,pad=0.04",
            facecolor=col, edgecolor='white', linewidth=1.2, zorder=2)
        ax2.add_patch(rect)
        ax2.text(0.5, t, f"R{st[t]}", fontsize=8,
                 ha='center', va='center', color='white',
                 fontweight='bold', zorder=3)
        ax2.text(1.15, t, str(t+1), fontsize=7,
                 ha='left', va='center', color='#888')
        # mark switches
        if t > 0 and st[t] != st[t-1]:
            ax2.axhline(t-0.5, color='#333', lw=1.5, zorder=4,
                        xmin=0.05, xmax=0.95)

    ax2.set_title("Regime\nsequence", fontsize=10,
                  fontweight='bold', color=TEXT_C, pad=6)

    return fig

# ── MAIN ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="/home/abir19/scratch/abir19/SDS_results")
    parser.add_argument("--out",         default="./figures/single_trajectories")
    parser.add_argument("--limit",       type=int, default=500)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for run in RUNS:
        cond, model, dataset, layer, path = run
        if not os.path.exists(path):
            print(f"  Skipping {path}", flush=True); continue

        tag      = f"{cond}_{model}_{dataset}_{layer}"
        out_path = os.path.join(args.out, f"traj_{tag}.png")
        if os.path.exists(out_path):
            print(f"  Skipping {tag} — exists", flush=True); continue

        try:
            umap2d, all_states, seqs_idx, best_id, K = process_run(
                run, args.results_dir, args.out, args.limit)
            fig = make_figure(umap2d, all_states, seqs_idx, best_id, K, run)
            fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  Saved {out_path}", flush=True)
        except Exception as e:
            print(f"  FAILED {tag}: {e}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    main()