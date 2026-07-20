"""
Cross-dataset SDS consistency for a fixed model/layer.

Trains CEBRA once on the first dataset (or pooled), then embeds all other
datasets with the same frozen encoder — spaces are directly comparable.

Modes:
  --fixed-k K   fit all datasets at K regimes
  --bic-k       fit each at its own BIC-optimal K

Metrics (all meaningful because spaces are aligned):
  1. Transition matrix similarity (Frobenius after Hungarian)
  2. Centroid cosine similarity (after Procrustes)
  3. Linear CKA between pooled CEBRA embeddings
  4. Cross-fit ΔR²: SDS from A predicts PCA trajectories of B
"""

import argparse, itertools, json, random, pickle
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import defaultdict

import cebra_EM as cem


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def hungarian_align(T1, T2):
    K = T1.shape[0]
    cost = np.array([[np.linalg.norm(T1[i] - T2[j]) for j in range(K)] for i in range(K)])
    _, col = linear_sum_assignment(cost)
    return T2[np.ix_(col, col)], col

def transition_similarity(T1, T2):
    K = T1.shape[0]
    T2p, _ = hungarian_align(T1, T2)
    return float(1.0 - np.linalg.norm(T1 - T2p, 'fro') / np.sqrt(2 * K))

def centroid_cosine_sim(C1, C2):
    K1, K2, D = len(C1), len(C2), C1.shape[1]
    K = max(K1, K2)
    pad = lambda C, k: np.vstack([C, np.zeros((k - len(C), D))]) if len(C) < k else C
    C1p, C2p = pad(C1, K), pad(C2, K)
    R, _ = orthogonal_procrustes(C2p, C1p)
    C2a = C2p @ R
    sims = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            n1, n2 = np.linalg.norm(C1p[i]), np.linalg.norm(C2a[j])
            if n1 > 1e-8 and n2 > 1e-8:
                sims[i, j] = np.dot(C1p[i], C2a[j]) / (n1 * n2)
    row, col = linear_sum_assignment(-sims)
    valid = [(r, c) for r, c in zip(row, col) if r < K1 and c < K2]
    return float(np.mean([sims[r, c] for r, c in valid])) if valid else float('nan')

def linear_cka(X, Y):
    def hsic(A, B):
        n = A.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return np.trace(A @ A.T @ H @ B @ B.T @ H) / (n - 1) ** 2
    return float(hsic(X, Y) / (np.sqrt(hsic(X, X) * hsic(Y, Y)) + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# embed a pkl with a *frozen* scaler + encoder (no retraining)
# ─────────────────────────────────────────────────────────────────────────────

def embed_with_frozen_encoder(pkl_path, scaler, encoder, pca_dim, limit_problems, device):
    """
    Load features, apply the frozen scaler + CEBRA encoder, group into trajectories.
    Returns cebra_seqs, pca_seqs, labels — same format as train_cebra_projection.
    """
    with open(pkl_path, 'rb') as f:
        all_features = pickle.load(f)
    all_features = [ft for ft in all_features if ft['problem_id'] < limit_problems]

    X_raw = np.array([ft['hidden_state_last'] for ft in all_features])
    X_scaled = scaler.transform(X_raw)

    # PCA in the same scaler space (refit PCA on this dataset — or reuse reference PCA)
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # CEBRA embed with frozen encoder
    import torch.nn.functional as F
    encoder.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        Z = F.normalize(encoder.net(X_t), p=2, dim=1).cpu().numpy()

    # group into per-problem trajectories
    p_map_z, p_map_p, p_map_l = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, ft in enumerate(all_features):
        pid = ft['problem_id']
        p_map_z[pid].append(Z[i])
        p_map_p[pid].append(X_pca[i])
        p_map_l[pid].append(ft.get('stage', 'NEUTRAL'))

    pids_sorted = sorted(p for p in p_map_z if len(p_map_z[p]) >= 3)
    cebra_seqs = [np.array(p_map_z[p]) for p in pids_sorted]
    pca_seqs   = [np.array(p_map_p[p]) for p in pids_sorted]
    labels     = [p_map_l[p]           for p in pids_sorted]
    return cebra_seqs, pca_seqs, labels


# ─────────────────────────────────────────────────────────────────────────────
# fit SDS on pre-embedded sequences
# ─────────────────────────────────────────────────────────────────────────────

def fit_sds(cebra_seqs, K, D, em_iters, seed):
    set_seed(seed)
    pi, A, dM, db, dCov = cem.init_params(cebra_seqs, K, D)
    for _ in range(em_iters):
        gammas, xis, lls = [], [], []
        for seq in cebra_seqs:
            g, x, ll = cem.forward_backward(seq, pi, A, dM, db, dCov, K)
            gammas.append(g); xis.append(x); lls.append(ll)
        pi, A, dM, db, dCov = cem.m_step(cebra_seqs, gammas, xis, K, D)
    total_ll = sum(lls)
    return pi, A, dM, db, dCov, gammas, total_ll

def bic(total_ll, K, D, N):
    n_params = K * (K - 1) + K * (D * D + D + D * (D + 1) // 2)
    return -2 * total_ll + n_params * np.log(N)

def select_k_bic(cebra_seqs, k_range, D, em_iters, seed):
    results = {}
    best = (None, np.inf, None)
    for K in k_range:
        pi, A, dM, db, dCov, gammas, total_ll = fit_sds(cebra_seqs, K, D, em_iters, seed)
        N = sum(len(s) for s in cebra_seqs)
        b = bic(total_ll, K, D, N)
        results[K] = b
        if b < best[1]:
            best = (K, b, (pi, A, dM, db, dCov, gammas))
    return best[0], results, best[2]

def build_fit(cebra_seqs, pca_seqs, labels, K, args):
    pi, A, dM, db, dCov, gammas, _ = fit_sds(cebra_seqs, K, args.cebra_dim, args.em_iters, args.seed)
    state_seqs = [np.argmax(g, axis=1) for g in gammas]
    T_hard = cem.hard_transition_matrix(state_seqs, K)
    centroids = []
    for k in range(K):
        vecs = [cebra_seqs[i][state_seqs[i] == k]
                for i in range(len(cebra_seqs)) if np.any(state_seqs[i] == k)]
        centroids.append(np.vstack(vecs).mean(0) if vecs else np.zeros(args.cebra_dim))
    return dict(cebra_seqs=cebra_seqs, pca_seqs=pca_seqs, labels=labels,
                pi=pi, A=A, dM=dM, db=db, dCov=dCov,
                state_seqs=state_seqs, centroids=np.array(centroids),
                T_hard=T_hard, K=K)


# ─────────────────────────────────────────────────────────────────────────────
# cross-fit R²
# ─────────────────────────────────────────────────────────────────────────────

def cross_fit_r2(src, tgt):
    K_src = src['K']
    state_seqs_cross = []
    for seq in tgt['cebra_seqs']:
        g, _, _ = cem.forward_backward(seq, src['pi'], src['A'], src['dM'], src['db'], src['dCov'], K_src)
        state_seqs_cross.append(np.argmax(g, axis=1))
    r2 = cem.regime_r2_on_pca(state_seqs_cross, tgt['pca_seqs'])
    ar2 = cem.linear_ar_r2(tgt['pca_seqs'])
    return float(r2), float(ar2)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkls', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', required=True)
    ap.add_argument('--fixed-k', type=int, default=None)
    ap.add_argument('--bic-k', action='store_true')
    ap.add_argument('--k-max', type=int, default=10)
    ap.add_argument('--cebra-dim', type=int, default=40)
    ap.add_argument('--em-iters', type=int, default=50)
    ap.add_argument('--limit-problems', type=int, default=500)
    ap.add_argument('--max-triplets', type=int, default=25)
    ap.add_argument('--cka-subsample', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='cross_dataset_consistency.json')
    args = ap.parse_args()

    assert len(args.pkls) == len(args.names)
    assert (args.fixed_k is not None) ^ args.bic_k, "specify exactly one of --fixed-k / --bic-k"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── step 1: train CEBRA on the first dataset, freeze encoder ─────────────
    print(f"Training reference CEBRA encoder on: {args.names[0]}", flush=True)
    set_seed(args.seed)
    all_features_ref, triplets_ref = cem.load_and_prepare_cebra(
        args.pkls[0], mode='temporal',
        limit_problems=args.limit_problems, max_triplets=args.max_triplets,
    )
    # train_cebra_projection must return (cebra_seqs, pca_seqs, labels, scaler, encoder)
    # after the one-line change to cebra_EM.py
    result = cem.train_cebra_projection(all_features_ref, triplets_ref, d_out=args.cebra_dim)
    assert len(result) == 5, (
        "train_cebra_projection must return 5 values: cebra_seqs, pca_seqs, labels, scaler, encoder.\n"
        "Add ', scaler, model' to the return statement in cebra_EM.py"
    )
    cebra_seqs_ref, pca_seqs_ref, labels_ref, ref_scaler, ref_encoder = result
    ref_encoder.eval()
    print(f"  Reference encoder trained. Embedding all datasets with frozen encoder...", flush=True)

    # ── step 2: embed every dataset with the frozen encoder ───────────────────
    embedded = {args.names[0]: (cebra_seqs_ref, pca_seqs_ref, labels_ref)}
    for name, pkl in zip(args.names[1:], args.pkls[1:]):
        print(f"  Embedding {name}...", flush=True)
        cebra_seqs_i, pca_seqs_i, labels_i = embed_with_frozen_encoder(
            pkl, ref_scaler, ref_encoder, args.cebra_dim, args.limit_problems, device
        )
        embedded[name] = (cebra_seqs_i, pca_seqs_i, labels_i)

    # ── step 3: fit SDS per dataset in the shared embedding space ────────────
    K_fixed = args.fixed_k
    fits = {}
    for name in args.names:
        cebra_seqs_i, pca_seqs_i, labels_i = embedded[name]
        print(f"  Fitting SDS on {name}...", flush=True)
        if args.bic_k:
            K, bics, (pi, A, dM, db, dCov, gammas) = select_k_bic(
                cebra_seqs_i, range(1, args.k_max + 1), args.cebra_dim, args.em_iters, args.seed
            )
            print(f"    BIC-optimal K={K}  " + "  ".join(f"K{k}={v:.0f}" for k, v in bics.items()), flush=True)
            state_seqs = [np.argmax(g, axis=1) for g in gammas]
            T_hard = cem.hard_transition_matrix(state_seqs, K)
            centroids = []
            for k in range(K):
                vecs = [cebra_seqs_i[i][state_seqs[i] == k]
                        for i in range(len(cebra_seqs_i)) if np.any(state_seqs[i] == k)]
                centroids.append(np.vstack(vecs).mean(0) if vecs else np.zeros(args.cebra_dim))
            fits[name] = dict(cebra_seqs=cebra_seqs_i, pca_seqs=pca_seqs_i, labels=labels_i,
                              pi=pi, A=A, dM=dM, db=db, dCov=dCov,
                              state_seqs=state_seqs, centroids=np.array(centroids),
                              T_hard=T_hard, K=K)
        else:
            fits[name] = build_fit(cebra_seqs_i, pca_seqs_i, labels_i, K_fixed, args)

        f = fits[name]
        print(f"    K={f['K']}  mean_self_trans={np.diag(f['T_hard']).mean():.3f}", flush=True)

    # ── step 4: metrics ───────────────────────────────────────────────────────
    results = {'config': vars(args), 'per_dataset': {}, 'pairwise': {}}

    for name, f in fits.items():
        ar2 = cem.linear_ar_r2(f['pca_seqs'])
        r2  = cem.regime_r2_on_pca(f['state_seqs'], f['pca_seqs'])
        persist = float(np.mean([len(s) / (np.count_nonzero(np.diff(s)) + 1) for s in f['state_seqs']]))
        results['per_dataset'][name] = dict(
            K=int(f['K']), ar_r2=float(ar2), regime_r2=float(r2),
            delta_r2=float(r2 - ar2),
            mean_self_trans=float(np.diag(f['T_hard']).mean()),
            persistence=persist,
        )

    pairs = list(itertools.combinations(args.names, 2))
    hdr = f"{'Pair':<25} | {'T_sim':>7} | {'Cent_cos':>9} | {'CKA':>7} | {'dR2_AonB':>9} | {'dR2_BonA':>9}"
    print(f"\n{hdr}\n" + "-" * len(hdr), flush=True)

    for n1, n2 in pairs:
        f1, f2 = fits[n1], fits[n2]

        t_sim = transition_similarity(f1['T_hard'], f2['T_hard']) if f1['K'] == f2['K'] else float('nan')
        c_sim = centroid_cosine_sim(f1['centroids'], f2['centroids'])

        z1 = np.vstack(f1['cebra_seqs']); z2 = np.vstack(f2['cebra_seqs'])
        n = min(len(z1), len(z2), args.cka_subsample)
        cka = linear_cka(z1[np.random.choice(len(z1), n, replace=False)],
                         z2[np.random.choice(len(z2), n, replace=False)])

        r2_1on2, ar2_2 = cross_fit_r2(f1, f2)
        r2_2on1, ar2_1 = cross_fit_r2(f2, f1)

        key = f"{n1}|{n2}"
        results['pairwise'][key] = dict(
            K_n1=int(f1['K']), K_n2=int(f2['K']),
            transition_sim=t_sim, centroid_cos=c_sim, cka=cka,
            r2_src1_on_tgt2=r2_1on2, delta_r2_src1_on_tgt2=float(r2_1on2 - ar2_2),
            r2_src2_on_tgt1=r2_2on1, delta_r2_src2_on_tgt1=float(r2_2on1 - ar2_1),
        )
        print(f"{key:<25} | {t_sim:7.4f} | {c_sim:9.4f} | {cka:7.4f} | "
              f"{r2_1on2 - ar2_2:+9.4f} | {r2_2on1 - ar2_1:+9.4f}", flush=True)

    with open(args.out, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nSaved → {args.out}", flush=True)


if __name__ == '__main__':
    main()