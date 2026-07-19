"""
generate_all_figures.py

Generates all paper figures:
  1. table_rlvr_vs_base.html + .tex   — per-model metric table
  2. method_<metric>.png (x9)          — CEBRA+EM vs MoE vs PCA per metric
  3. fig_swap_big_layers.png           — swap ablation late layers
  4. fig_swap_small_layers.png         — swap ablation early layers
  5. fig_sufficiency_big_layers.png    — sufficiency late layers
  6. fig_sufficiency_small_layers.png  — sufficiency early layers
  7. membership_<tag>.png (x12)        — membership heatmaps per model

Usage:
    python generate_all_figures.py \
        --results_dir /home/abir19/scratch/abir19/SDS_results \
        --out ./figures
"""

import os, json, argparse
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── PALETTE ─────────────────────────────────────────────────
RLVR_C  = '#ff563f'
BASE_C  = '#55c89f'
MOE_C   = '#55c89f'
PCA_C   = '#ffd24d'
EM_C    = '#ff563f'
TEXT_C  = '#0c0c0c'
SUB_C   = '#928e8b'
GRID_C  = '#f5f5f5'
AXIS_C  = '#e5dfdf'
FONT    = "DejaVu Sans"

STAGES_SHORT = ["SETUP","RETRIEVAL","PLAN","UNCERT","CHECK","CONSOL","COMPUTE","ANSWER"]

# ── HELPERS ──────────────────────────────────────────────────

def ax_style(ax):
    ax.set_facecolor("white")
    ax.grid(axis='y', color=GRID_C, linewidth=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(AXIS_C)
    ax.spines['bottom'].set_color(AXIS_C)
    ax.tick_params(colors=TEXT_C, labelsize=11)

def pval_str(p):
    if p < 0.0001: return "p<0.0001"
    if p < 0.001:  return "p<0.001"
    if p < 0.01:   return f"p={p:.3f}"
    return f"p={p:.3f}"

def normalize(fname):
    s = fname.replace(".json","").lower()
    for r in ["_reasoning","_base","_Base"]: s = s.replace(r,"")
    return s

def short_label(key):
    s = key.replace("sds_","").replace("train_","").replace("test_","")
    s = s.replace("_layer_","·L").replace("math500","m500")
    return (s.replace("llama_8b","L8B").replace("qwen_14b","Q14B")
             .replace("qwen_1_5b","Q1.5B").replace("qwen14b","Q14B")
             .replace("qwen1.5b","Q1.5B").replace("llama8b","L8B")
             .replace("llama_8B","L8B").replace("Llama_8B","L8B")
             .replace("Qwen_14B","Q14B").replace("Qwen_1_5B","Q1.5B"))

def clean_swap_tag(tag):
    tag = tag.replace("SDS_","").replace("_reasoning","")
    tag = tag.replace("train_","").replace("test_","")
    return short_label(tag)

def is_big_layer(s):
    return any(l in s for l in ["layer_31","layer_47","layer_27","L31","L47","L27"])

def load_dir(path):
    return {normalize(f): json.load(open(f"{path}/{f}"))
            for f in os.listdir(path) if f.endswith(".json")}

def get_model(key):
    k = key.lower()
    if "llama_8b" in k or "llama8b" in k: return "Llama-8B"
    if "qwen_14b" in k or "qwen14b" in k: return "Qwen-14B"
    if "qwen_1_5b" in k or "qwen1.5b" in k: return "Qwen-1.5B"
    return None

def compute_metrics(data, method="em"):
    k = str(data["best_k"].get("by_bic", data["best_k"].get("by_sss"))) \
        if method != "moe" else str(data["best_k"].get("by_sss"))
    s = data["sweep"][k]
    T = np.array(s["transition_matrix"]); K = len(T)
    uniform = np.ones((K,K))/K
    tvd  = float(np.mean(0.5*np.abs(T-uniform).sum(axis=1)))
    eigs = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
    sg   = float(1-eigs[1]) if K > 1 else 0.0
    ev, evec = np.linalg.eig(T.T)
    idx  = np.argmin(np.abs(ev-1.0))
    stat = np.abs(evec[:,idx].real); stat /= stat.sum()
    se   = float(-np.sum(stat*np.log(stat+1e-12)))
    ds   = s.get("dominant_stages",[])
    return {
        "K":              int(k),
        "k_eff":          s["k_eff"],
        "persistence":    s["persist"],
        "tvd":            tvd,
        "spectral_gap":   sg,
        "stat_entropy":   se,
        "unique_stages":  len(set(d["stage"] for d in ds)) if ds else 0,
        "sss":            s["sss"],
        "tqs":            s["tqs"],
        "delta_r2":       s["delta_r2"],
    }


# ══════════════════════════════════════════════════════════════
# FIG SET 1: RLVR vs BASE TABLE
# ══════════════════════════════════════════════════════════════

METRICS_TABLE = [
    ("k_eff",         "K_eff",          True),
    ("persistence",   "Persistence",    True),
    ("tvd",           "TVD",            True),
    ("spectral_gap",  "Spec. Gap",      False),
    ("unique_stages", "Stages",         True),
    ("stat_entropy",  "Entropy",        True),
]

def fig_rlvr_vs_base_table(em_dir, base_dir, out_dir):
    rlvr_files = {normalize(f): f for f in os.listdir(em_dir)   if f.endswith(".json")}
    base_files = {normalize(f): f for f in os.listdir(base_dir) if f.endswith(".json")}
    common = sorted(set(rlvr_files) & set(base_files))

    model_rlvr = defaultdict(list)
    model_base = defaultdict(list)
    for key in common:
        model = get_model(key)
        if model is None: continue
        model_rlvr[model].append(compute_metrics(json.load(open(f"{em_dir}/{rlvr_files[key]}"))))
        model_base[model].append(compute_metrics(json.load(open(f"{base_dir}/{base_files[key]}"))))

    MODELS = ["Llama-8B", "Qwen-14B", "Qwen-1.5B"]
    col_labels = [f"{m[1]} {'↑' if m[2] else '↓'}" for m in METRICS_TABLE]
    n_metrics = len(METRICS_TABLE)

    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor("white"); ax.axis('off')

    # build table data
    header = ["Model"] + [f"{l} Base/RLVR" for l in col_labels]
    rows = []
    cell_colors = []
    for model in MODELS:
        rv = model_rlvr[model]; bv = model_base[model]
        if not rv: continue
        row = [model]
        colors = [["white"]]
        for key, _, higher in METRICS_TABLE:
            bm = np.mean([d[key] for d in bv])
            rm = np.mean([d[key] for d in rv])
            r_better = (rm > bm) if higher else (rm < bm)
            row.append(f"{bm:.3f}   {rm:.3f}")
            bc_cell = "#ffccc5" if not r_better else "white"
            rc_cell = "#ffccc5" if r_better     else "white"
            # use single cell color — highlight the better one
            colors.append([rc_cell if r_better else bc_cell])
        rows.append(row)
        cell_colors.append(colors)

    # one subplot per metric, 3 grouped bars (models) x 2 (base/rlvr)
    fig, axes = plt.subplots(1, n_metrics, figsize=(22, 4))
    fig.patch.set_facecolor("white")
    x = np.arange(len(MODELS)); w = 0.38

    for ax, (key, label, higher) in zip(axes, METRICS_TABLE):
        ax_style(ax)
        bv = [np.mean([d[key] for d in model_base[m]]) if model_base[m] else 0 for m in MODELS]
        rv = [np.mean([d[key] for d in model_rlvr[m]]) if model_rlvr[m] else 0 for m in MODELS]
        ax.bar(x-w/2, bv, w, color=BASE_C,  alpha=0.85, label="Base", zorder=3)
        ax.bar(x+w/2, rv, w, color=RLVR_C, alpha=0.85, label="RLVR", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=25, ha='right', fontsize=11, color=TEXT_C)
        arrow = "↑" if higher else "↓"
        ax.set_ylabel(f"{label} {arrow}", fontsize=12, fontweight='bold', color=TEXT_C)
        ax.tick_params(axis='y', labelsize=10, colors=TEXT_C)

    handles = [plt.Rectangle((0,0),1,1,color=BASE_C,alpha=0.85),
               plt.Rectangle((0,0),1,1,color=RLVR_C,alpha=0.85)]
    fig.legend(handles, ["Base","RLVR"], fontsize=13, frameon=False,
               loc='upper center', bbox_to_anchor=(0.5,1.0), ncol=2,
               bbox_transform=fig.transFigure)
    fig.suptitle("Structural Switching Metrics: Base vs RLVR per Model",
                 fontsize=14, fontweight='bold', color=TEXT_C, y=1.08)
    plt.tight_layout(); plt.subplots_adjust(top=0.82, wspace=0.5)
    plt.savefig(f"{out_dir}/table_rlvr_vs_base.png", dpi=180,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved table_rlvr_vs_base.png")


# ══════════════════════════════════════════════════════════════
# FIG SET 2: METHOD COMPARISON (one PNG per metric)
# ══════════════════════════════════════════════════════════════

METHOD_METRICS = [
    ("delta_r2",    "ΔR² over Linear AR",       True),
    ("tvd",         "TVD from Uniform",           True),
    ("spectral_gap","Spectral Gap",               False),
    ("k_eff",       "Effective Regimes (K_eff)",  True),
    ("persistence", "Persistence",                True),
    ("stat_entropy","Stationary Entropy",         True),
    ("sss",         "SSS (appendix)",             True),
    ("tqs",         "TQS (appendix)",             True),
]

def fig_method_comparison(em_dir, moe_dir, pca_dir, out_dir):
    em  = load_dir(em_dir); moe = load_dir(moe_dir); pca = load_dir(pca_dir)
    common = sorted(set(em) & set(moe) & set(pca))
    labels = [short_label(k) for k in common]

    def get(data, method):
        k = str(data["best_k"].get("by_bic", data["best_k"].get("by_sss"))) \
            if method != "moe" else str(data["best_k"].get("by_sss"))
        s = data["sweep"][k]
        T = np.array(s["transition_matrix"]); K = len(T)
        u = np.ones((K,K))/K
        tvd = float(np.mean(0.5*np.abs(T-u).sum(axis=1)))
        eigs = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
        sg = float(1-eigs[1]) if K>1 else 0.0
        ev,evec = np.linalg.eig(T.T); idx=np.argmin(np.abs(ev-1.0))
        stat=np.abs(evec[:,idx].real); stat/=stat.sum()
        se=float(-np.sum(stat*np.log(stat+1e-12)))
        return {"delta_r2":s["delta_r2"],"tvd":tvd,"spectral_gap":sg,
                "k_eff":s["k_eff"],"persistence":s["persist"],
                "stat_entropy":se,"sss":s["sss"],"tqs":s["tqs"]}

    em_m  = [get(em[k],  "em")  for k in common]
    moe_m = [get(moe[k], "moe") for k in common]
    pca_m = [get(pca[k], "pca") for k in common]

    for metric, display, higher in METHOD_METRICS:
        ev=[v[metric] for v in em_m]; mv=[v[metric] for v in moe_m]; pv=[v[metric] for v in pca_m]
        def wp(a,b):
            try: return stats.wilcoxon(a,b).pvalue
            except: return float('nan')
        p_em_moe=wp(ev,mv); p_em_pca=wp(ev,pv)
        arrow="↑" if higher else "↓"

        fig,ax=plt.subplots(figsize=(18,5)); fig.patch.set_facecolor("white"); ax_style(ax)
        x=np.arange(len(common)); w=0.26
        for i,(vals,col,name) in enumerate(zip([ev,mv,pv],[EM_C,MOE_C,PCA_C],
                                               ["CEBRA+EM","CEBRA-MoE","PCA+SLDS"])):
            ax.bar(x+(i-1)*w, vals, w, color=col, alpha=0.85, label=name, zorder=3)
            ax.axhline(np.mean(vals), color=col, linestyle='--', linewidth=1.5,
                       alpha=0.75, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=13, color=TEXT_C)
        ax.set_ylabel(f"{display} {arrow}", fontsize=18, fontweight='bold', color=TEXT_C)
        ax.set_xlabel("Dataset · Model · Layer", fontsize=16, fontweight='bold', color=TEXT_C)
        ax.set_title(f"CEBRA+EM vs MoE: {pval_str(p_em_moe)}   ·   CEBRA+EM vs PCA+SLDS: {pval_str(p_em_pca)}",
                     fontsize=12, color=SUB_C, pad=8)
        ax.legend(fontsize=13, frameon=False, loc='upper center',
                  bbox_to_anchor=(0.5,1.0), ncol=3, bbox_transform=fig.transFigure)
        plt.tight_layout(); plt.subplots_adjust(top=0.82)
        plt.savefig(f"{out_dir}/method_{metric}.png", dpi=180,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved method_{metric}.png")


# ══════════════════════════════════════════════════════════════
# FIG SET 3: SWAP ABLATION (big + small layers)
# ══════════════════════════════════════════════════════════════

def _draw_swap(records, title, fname, out_dir):
    records.sort(key=lambda x: x["drop"], reverse=True)
    fig,ax=plt.subplots(figsize=(16,5)); fig.patch.set_facecolor("white"); ax_style(ax)
    x=np.arange(len(records)); w=0.38; tags=[r["tag"] for r in records]
    ax.bar(x-w/2,[r["r2_id"]   for r in records],w,color=RLVR_C,alpha=0.85,label="R² (identity)",zorder=3)
    ax.bar(x+w/2,[r["r2_rand"] for r in records],w,color=BASE_C, alpha=0.85,label="R² (random perm)",zorder=3)
    for i,r in enumerate(records):
        ax.annotate(f"−{r['drop']:.2f}", xy=(i,max(r["r2_id"],r["r2_rand"])+0.01),
                    ha='center', va='bottom', fontsize=9, color=SUB_C)
    mean_drop=np.mean([r["drop"] for r in records])
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha='right', fontsize=13, color=TEXT_C)
    ax.set_ylabel("R²", fontsize=18, fontweight='bold', color=TEXT_C)
    ax.set_xlabel("Dataset · Model · Layer", fontsize=16, fontweight='bold', color=TEXT_C)
    ax.set_title(f"{title}  ·  Mean drop = {mean_drop:.3f}  (n={len(records)})",
                 fontsize=12, color=SUB_C, pad=8)
    ax.legend(fontsize=13, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5,1.0), ncol=2, bbox_transform=fig.transFigure)
    plt.tight_layout(); plt.subplots_adjust(top=0.82)
    plt.savefig(f"{out_dir}/{fname}", dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  Saved {fname}")

def fig_swap(swap_dir, out_dir):
    big_rec, small_rec = [], []
    for f in sorted(os.listdir(swap_dir)):
        if not f.endswith(".json"): continue
        d=json.load(open(f"{swap_dir}/{f}"))
        tag=clean_swap_tag(f.replace(".json",""))
        rec={"tag":tag,"r2_id":d["r2_identity"],"r2_rand":d["r2_rand_mean"],"drop":d["r2_drop"]}
        (big_rec if is_big_layer(f) else small_rec).append(rec)
    _draw_swap(big_rec,   "State Swap Ablation — Late Layers",  "fig_swap_big_layers.png",   out_dir)
    _draw_swap(small_rec, "State Swap Ablation — Early Layers", "fig_swap_small_layers.png", out_dir)


# ══════════════════════════════════════════════════════════════
# FIG SET 4: SUFFICIENCY (big + small layers)
# ══════════════════════════════════════════════════════════════

SUF_PANELS = [
    ("correlations",                    "n_switches",  "n_switches (raw r)"),
    ("partial_correlations_ctrl_length","k_eff",       "k_eff (partial r)"),
    ("partial_correlations_ctrl_length","switch_rate", "switch_rate (partial r)"),
    ("partial_correlations_ctrl_length","n_switches",  "n_switches (partial r)"),
]

def _draw_sufficiency(rlvr_list, base_list, title, fname, out_dir):
    fig,axes=plt.subplots(1,4,figsize=(22,5)); fig.patch.set_facecolor("white")
    for ax,(section,key,label) in zip(axes,SUF_PANELS):
        ax_style(ax)
        def get_r(d):
            try: return d[section][key]["r"]
            except: return 0.0
        rv=[get_r(d) for d in rlvr_list]; bv=[get_r(d) for d in base_list]
        tags=[d["tag"] for d in rlvr_list]
        x=np.arange(len(tags)); w=0.38
        ax.bar(x-w/2,rv,w,color=RLVR_C,alpha=0.85,label="RLVR",zorder=3)
        ax.bar(x+w/2,bv,w,color=BASE_C, alpha=0.85,label="Base",zorder=3)
        ax.axhline(0,color=SUB_C,linewidth=0.8,zorder=2)
        ax.axhline(np.mean(rv),color=RLVR_C,linewidth=1.5,linestyle='--',alpha=0.75,zorder=2)
        ax.axhline(np.mean(bv),color=BASE_C, linewidth=1.5,linestyle='--',alpha=0.75,zorder=2)
        try: _,p=stats.wilcoxon(rv,bv)
        except: p=float('nan')
        ax.set_xticks(x)
        ax.set_xticklabels(tags,rotation=45,ha='right',fontsize=11,color=TEXT_C)
        ax.set_ylabel(label,fontsize=14,fontweight='bold',color=TEXT_C)
        ax.set_title(f"RLVR μ={np.mean(rv):.3f}  Base μ={np.mean(bv):.3f}\n{pval_str(p)}",
                     fontsize=10,color=SUB_C)
    handles=[plt.Rectangle((0,0),1,1,color=RLVR_C,alpha=0.85),
             plt.Rectangle((0,0),1,1,color=BASE_C, alpha=0.85)]
    fig.legend(handles,["RLVR","Base"],fontsize=13,frameon=False,
               loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=2,bbox_transform=fig.transFigure)
    fig.suptitle(title,fontsize=13,fontweight='bold',color=TEXT_C,y=1.06)
    plt.tight_layout(); plt.subplots_adjust(top=0.82,wspace=0.35)
    plt.savefig(f"{out_dir}/{fname}",dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(); print(f"  Saved {fname}")

def fig_sufficiency(suf_dir, out_dir):
    rlvr=sorted([json.load(open(f"{suf_dir}/{f}")) for f in os.listdir(suf_dir) if f.startswith("rlvr")],
                key=lambda x:x["tag"])
    base=sorted([json.load(open(f"{suf_dir}/{f}")) for f in os.listdir(suf_dir) if f.startswith("base")],
                key=lambda x:x["tag"])
    def big(d): return is_big_layer(d["tag"])
    _draw_sufficiency([d for d in rlvr if     big(d)],[d for d in base if     big(d)],
                      "Sufficiency — Late Layers",  "fig_sufficiency_big_layers.png",  out_dir)
    _draw_sufficiency([d for d in rlvr if not big(d)],[d for d in base if not big(d)],
                      "Sufficiency — Early Layers", "fig_sufficiency_small_layers.png", out_dir)


# ══════════════════════════════════════════════════════════════
# FIG SET 5: MEMBERSHIP HEATMAPS (one per model)
# ══════════════════════════════════════════════════════════════

def fig_membership(mem_dir, out_dir):
    all_tags=[f.replace(".json","") for f in os.listdir(mem_dir) if f.endswith(".json")]
    for tag in sorted(all_tags):
        d=json.load(open(f"{mem_dir}/{tag}.json"))
        M=np.array(d["p_stage_given_state"]); K=d["K"]
        is_rlvr=tag.startswith("rlvr")
        cell_color=RLVR_C if is_rlvr else BASE_C
        accent=RLVR_C if is_rlvr else BASE_C
        cond="RLVR" if is_rlvr else "Base"
        model_part=tag.replace("rlvr_","").replace("base_","")

        fig,ax=plt.subplots(figsize=(11,4)); fig.patch.set_facecolor("white"); ax.set_facecolor("white")
        base_rgb=np.array(mcolors.to_rgb(cell_color))
        for i in range(K):
            for j in range(8):
                v=M[i,j]; alpha=v/max(M.max(),1e-6)
                blended=1-(1-base_rgb)*alpha
                ax.add_patch(plt.Rectangle([j-0.5,i-0.5],1,1,color=blended,zorder=1))
                if v>0.07:
                    tc="white" if alpha>0.6 else TEXT_C
                    ax.text(j,i,f"{v:.2f}",ha='center',va='center',
                            fontsize=10,color=tc,fontweight='bold',zorder=2)
        ax.set_xlim(-0.5,7.5); ax.set_ylim(-0.5,K-0.5); ax.invert_yaxis()
        ax.set_xticks(range(8))
        ax.set_xticklabels(STAGES_SHORT,rotation=40,ha='right',fontsize=13,color=TEXT_C)
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"R{k}" for k in range(K)],fontsize=13,color=TEXT_C)
        ax.set_ylabel("Regime",fontsize=14,fontweight='bold',color=TEXT_C)
        ax.set_xlabel("Reasoning Stage",fontsize=14,fontweight='bold',color=TEXT_C)
        for i in range(K+1): ax.axhline(i-0.5,color='white',linewidth=1.5,zorder=3)
        for j in range(9):   ax.axvline(j-0.5,color='white',linewidth=1.5,zorder=3)
        for spine in ax.spines.values(): spine.set_edgecolor(accent); spine.set_linewidth(2)
        ax.set_title(f"{cond} · {model_part}  —  P(stage | regime)",
                     fontsize=15,fontweight='bold',color=TEXT_C,pad=10)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/membership_{tag}.png",dpi=180,bbox_inches='tight',facecolor='white')
        plt.close(); print(f"  Saved membership_{tag}.png")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="/home/abir19/scratch/abir19/SDS_results")
    parser.add_argument("--out", default="./figures")
    args=parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R=args.results_dir; O=args.out

    print("\n── RLVR vs Base Table ──")
    fig_rlvr_vs_base_table(f"{R}/cebra_em", f"{R}/cebra_em_base", O)

    print("\n── Method Comparison ──")
    fig_method_comparison(f"{R}/cebra_em", f"{R}/cebra_moe", f"{R}/pca_slds", O)

    print("\n── Swap Ablation ──")
    fig_swap(f"{R}/state_swap", O)

    print("\n── Sufficiency ──")
    fig_sufficiency(f"{R}/sufficiency", O)

    print("\n── Membership Heatmaps ──")
    fig_membership(f"{R}/membership", O)

    print("\nAll figures done.")

if __name__ == "__main__":
    main()