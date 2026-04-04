import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cebra_EM as cem

STAGES = [
    "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
]
STAGE_SHORT = ["SETUP", "FACTS", "PLAN", "UNCERT", "CHECK", "CONSOL", "COMPUTE", "EMIT"]

MODELS = {
    "qwen14b_L47": {
        "pkl": "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen_14b_reasoning/layer_47/all_sentences_features.pkl",
        "label": "Qwen-14B (L47)",
    },
    "llama8b_L31": {
        "pkl": "/home/abir19/scratch/abir19/SDS_train_gsm8k/llama_8b_reasoning/layer_31/all_sentences_features.pkl",
        "label": "Llama-8B (L31)",
    },
    "qwen15b_L27": {
        "pkl": "/home/abir19/scratch/abir19/SDS_train_gsm8k/qwen1.5b_reasoning/layer_27/all_sentences_features.pkl",
        "label": "Qwen-1.5B (L27)",
    },
}


def _fit_sds_one_k(cebra_seqs, K, cebra_dim, em_iters):
    pi, A, dM, db, dCov = cem.init_params(cebra_seqs, K, cebra_dim)
    for _ in range(em_iters):
        gammas, xis, lls = [], [], []
        for seq in cebra_seqs:
            g, x, ll = cem.forward_backward(seq, pi, A, dM, db, dCov, K)
            gammas.append(g); xis.append(x); lls.append(ll)
        pi, A, dM, db, dCov = cem.m_step(cebra_seqs, gammas, xis, K, cebra_dim)
    total_ll = sum(lls)
    N = sum(len(s) for s in cebra_seqs)
    n_params = K*(K-1) + K*(cebra_dim**2 + cebra_dim + cebra_dim*(cebra_dim+1)//2)
    bic = -2 * total_ll + n_params * np.log(N)
    return pi, A, dM, db, dCov, gammas, bic


def fit_and_get_membership(pkl_path, K_fixed, k_max, cebra_dim, em_iters,
                           limit_problems, max_triplets, seed):
    import random, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    all_features, triplets = cem.load_and_prepare_cebra(
        pkl_path, mode='temporal',
        limit_problems=limit_problems, max_triplets=max_triplets,
    )
    cebra_seqs = cem.train_cebra_projection(all_features, triplets, d_out=cebra_dim)[0]

    if K_fixed is not None:
        K = K_fixed
        print(f"    Fitting K={K} (fixed)", flush=True)
        _, A, dM, db, dCov, gammas, bic = _fit_sds_one_k(cebra_seqs, K, cebra_dim, em_iters)
        bic_scores = {K: bic}
    else:
        print(f"    BIC sweep K=1..{k_max}", flush=True)
        bic_scores = {}
        best = (None, np.inf, None)
        for k in range(1, k_max + 1):
            result = _fit_sds_one_k(cebra_seqs, k, cebra_dim, em_iters)
            bic_scores[k] = result[6]
            print(f"      K={k}  BIC={result[6]:.1f}", flush=True)
            if result[6] < best[1]:
                best = (k, result[6], result)
        K = best[0]
        _, A, dM, db, dCov, gammas, bic = best[2]
        print(f"    BIC-optimal K={K}  (BIC={bic:.1f})", flush=True)

    state_seqs = [np.argmax(g, axis=1) for g in gammas]
    T_hard = cem.hard_transition_matrix(state_seqs, K)

    # build p_map aligned with cebra_seqs ordering
    p_map = {}
    for i, f in enumerate(all_features):
        p_map.setdefault(int(f['problem_id']), []).append(i)
    pids_sorted = sorted(p for p in p_map if len(p_map[p]) >= 3)

    counts_hard = np.zeros((len(STAGES), K))
    counts_soft = np.zeros((len(STAGES), K))

    for pid_idx, pid in enumerate(pids_sorted):
        idxs = p_map[pid]
        gamma = gammas[pid_idx]
        for t, global_idx in enumerate(idxs):
            stage = all_features[global_idx].get('stage', 'NEUTRAL')
            if stage not in STAGES:
                continue
            s_idx = STAGES.index(stage)
            counts_hard[s_idx, int(state_seqs[pid_idx][t])] += 1
            counts_soft[s_idx] += gamma[t]

    row_sum_h = counts_hard.sum(1, keepdims=True)
    row_sum_s = counts_soft.sum(1, keepdims=True)
    membership_hard = counts_hard / np.where(row_sum_h == 0, 1, row_sum_h)
    membership_soft = counts_soft / np.where(row_sum_s == 0, 1, row_sum_s)
    dominant = [f"R{np.argmax(membership_hard[s])}" for s in range(len(STAGES))]
    persist = float(np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs]))

    return dict(
        membership_hard=membership_hard, membership_soft=membership_soft,
        counts_hard=counts_hard, T_hard=T_hard,
        dominant_stages=dominant, persistence=persist,
        mean_self_trans=float(np.diag(T_hard).mean()),
        K=K, bic_scores=bic_scores,
    )


def plot_membership(membership, title, out_path, K):
    fig, ax = plt.subplots(figsize=(max(5, K * 1.2), 6))
    sns.heatmap(
        membership, ax=ax,
        xticklabels=[f"R{k}" for k in range(K)],
        yticklabels=STAGE_SHORT,
        annot=True, fmt=".2f", cmap="YlOrRd",
        vmin=0, vmax=1.0,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'P(regime | stage)'},
    )
    ax.set_xlabel("Discovered Regime", fontsize=11)
    ax.set_ylabel("Reasoning Stage", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=None,
                    help='Fix K for all models. Omit to use --bic-k.')
    ap.add_argument('--bic-k', action='store_true',
                    help='Select K per model via BIC sweep (default if --K not given).')
    ap.add_argument('--k-max', type=int, default=10)
    ap.add_argument('--cebra-dim', type=int, default=40)
    ap.add_argument('--em-iters', type=int, default=50)
    ap.add_argument('--limit-problems', type=int, default=500)
    ap.add_argument('--max-triplets', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', type=str, default='membership_results')
    ap.add_argument('--models', nargs='+', default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    args = ap.parse_args()

    # if neither flag given, default to BIC
    use_bic = (args.K is None)

    import os; os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}

    for model_key in args.models:
        cfg = MODELS[model_key]
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {cfg['label']}", flush=True)

        res = fit_and_get_membership(
            cfg['pkl'],
            K_fixed=args.K if not use_bic else None,
            k_max=args.k_max,
            cebra_dim=args.cebra_dim,
            em_iters=args.em_iters,
            limit_problems=args.limit_problems,
            max_triplets=args.max_triplets,
            seed=args.seed,
        )
        K = res['K']

        print(f"\n  K={K}  Persistence={res['persistence']:.2f}  "
              f"mean_self_trans={res['mean_self_trans']:.3f}", flush=True)

        print(f"\n  P(regime | stage):", flush=True)
        for s_idx, stage in enumerate(STAGES):
            row = res['membership_hard'][s_idx]
            top2 = sorted(range(K), key=lambda k: -row[k])[:2]
            print(f"    {stage:<30} → R{top2[0]} ({row[top2[0]]:.1%})  "
                  f"R{top2[1]} ({row[top2[1]]:.1%})", flush=True)

        regime_cols = "".join(f"{'R'+str(k):>8}" for k in range(K))
        print(f"\n  {'Stage':<30}{regime_cols}", flush=True)
        for s_idx, stage in enumerate(STAGES):
            row = "".join(f"{v:8.3f}" for v in res['membership_hard'][s_idx])
            print(f"  {stage:<30}{row}", flush=True)

        plot_membership(
            res['membership_hard'],
            title=f"P(Regime | Stage): {cfg['label']} (K={K}, GSM8K)",
            out_path=os.path.join(args.out_dir, f"{model_key}_K{K}_membership.png"),
            K=K,
        )

        all_results[model_key] = {
            'label': cfg['label'], 'K': K,
            'persistence': res['persistence'],
            'mean_self_trans': res['mean_self_trans'],
            'dominant_regime_per_stage': res['dominant_stages'],
            'bic_scores': {str(k): v for k, v in res['bic_scores'].items()},
            'membership_hard': res['membership_hard'].tolist(),
            'membership_soft': res['membership_soft'].tolist(),
            'counts_hard': res['counts_hard'].tolist(),
            'T_hard': res['T_hard'].tolist(),
        }

    out_json = os.path.join(args.out_dir, 'membership_all_models.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_json}", flush=True)


if __name__ == '__main__':
    main()