# import numpy as np
# import pickle
# from collections import defaultdict
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# K_SWEEP = range(1, 9)
# D = 40
# N_ITERS = 50
# KAPPA = 10.0

# STAGES = [
#     "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
#     "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
#     "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
# ]

# # ── LOAD ─────────────────────────────────────────────────────

# def load_raw_sequences(path, hidden_key='hidden_state_last'):
#     with open(path, 'rb') as f:
#         features = pickle.load(f)
#     p_map = defaultdict(list)
#     for feat in features:
#         if feat.get('stage', 'NEUTRAL') != 'NEUTRAL':
#             p_map[feat['problem_id']].append(feat)
#     sequences, label_seqs = [], []
#     for pid in sorted(p_map.keys()):
#         feats = sorted(p_map[pid], key=lambda x: x['sentence_idx'])
#         if len(feats) >= 3:
#             sequences.append(np.array([f[hidden_key] for f in feats]))
#             label_seqs.append([f['stage'] for f in feats])
#     return sequences, label_seqs

# # ── REPRESENTATIONS ───────────────────────────────────────────

# def make_representation(sequences, label_seqs=None, mode='delta',
#                         reducer='pca', n_components=D):
#     if mode == 'centered':
#         seqs = [s - s.mean(0, keepdims=True) for s in sequences]
#         labels_aligned = label_seqs
#     elif mode == 'delta':
#         seqs = [np.diff(s, axis=0) for s in sequences]
#         labels_aligned = [l[1:] for l in label_seqs] if label_seqs else None
#     else:
#         seqs = sequences
#         labels_aligned = label_seqs

#     X_all = np.concatenate(seqs, axis=0)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_all)

#     if reducer == 'lda' and labels_aligned is not None:
#         y_all = np.concatenate([[STAGES.index(l) if l in STAGES else 0
#                                  for l in ls] for ls in labels_aligned])
#         n_comp = min(n_components, len(STAGES) - 1)
#         red = LinearDiscriminantAnalysis(n_components=n_comp)
#         X_red = red.fit_transform(X_scaled, y_all)
#     else:
#         red = PCA(n_components=n_components)
#         X_red = red.fit_transform(X_scaled)

#     reduced, idx = [], 0
#     for s in seqs:
#         n = len(s)
#         reduced.append(X_red[idx:idx+n])
#         idx += n

#     return reduced, labels_aligned, scaler, red

# # ── DIAGNOSTIC ────────────────────────────────────────────────

# def run_diagnostic(sequences, label_seqs, tag=""):
#     print(f"\n── Diagnostic {tag} ──")
#     configs = [
#         (mode, reducer)
#         for mode in ['absolute', 'centered', 'delta']
#         for reducer in ['pca', 'lda']
#     ]
#     results = []
#     for mode, reducer in configs:
#         try:
#             reduced, _, _, _ = make_representation(
#                 sequences, label_seqs, mode=mode,
#                 reducer=reducer, n_components=min(D, len(STAGES)-1)
#             )
#             X_all = np.concatenate(reduced, axis=0)
#             km_labels = KMeans(n_clusters=4, n_init=10,
#                                random_state=42).fit_predict(X_all)
#             sil = silhouette_score(X_all, km_labels,
#                                    sample_size=min(2000, len(X_all)))
#             print(f"  {mode:10s} | {reducer:3s}: silhouette={sil:.3f}  n={len(X_all)}")
#             results.append((sil, mode, reducer))
#         except Exception as e:
#             print(f"  {mode:10s} | {reducer:3s}: ERROR {e}")

#     best = max(results, key=lambda x: x[0])
#     print(f"\n  Best: mode={best[1]}, reducer={best[2]} (silhouette={best[0]:.3f})")
#     return best[1], best[2]

# # ── SLDS ──────────────────────────────────────────────────────

# def init_params(sequences, K, D):
#     X_in, X_out = [], []
#     for seq in sequences:
#         for t in range(len(seq)-1):
#             X_in.append(seq[t])
#             X_out.append(seq[t+1] - seq[t])
#     X_in  = np.array(X_in)
#     X_out = np.array(X_out)

#     km = KMeans(n_clusters=K, n_init=10, random_state=42)
#     labels = km.fit_predict(X_out)

#     dynamics_M = np.zeros((K, D, D))
#     dynamics_b = np.zeros((K, D))
#     noise_cov  = np.array([np.eye(D) for _ in range(K)])

#     for k in range(K):
#         mask = labels == k
#         if mask.sum() < D + 2:
#             dynamics_M[k] = 0.1 * np.eye(D)
#             continue
#         Xk_in  = X_in[mask]
#         Xk_out = X_out[mask]
#         X_aug  = np.hstack([Xk_in, np.ones((len(Xk_in), 1))])
#         W, *_  = np.linalg.lstsq(X_aug, Xk_out, rcond=None)
#         dynamics_M[k] = W[:D].T
#         dynamics_b[k] = W[D]
#         res = Xk_out - (Xk_in @ dynamics_M[k].T + dynamics_b[k])
#         noise_cov[k] = np.cov(res.T) + 1e-3 * np.eye(D)

#     pi = np.ones(K) / K
#     A  = np.eye(K) * 0.7 + 0.3 / K
#     return pi, A, dynamics_M, dynamics_b, noise_cov

# def get_log_emissions(seq, K, dM, db, dCov):
#     T, D = seq.shape
#     log_emit = np.zeros((T, K))
#     for k in range(K):
#         sign, logdet = np.linalg.slogdet(dCov[k])
#         inv_cov = np.linalg.inv(dCov[k])
#         means = np.vstack([db[k], seq[:-1] @ dM[k].T + db[k]])
#         diffs = seq - means
#         mahal = np.sum((diffs @ inv_cov) * diffs, axis=1)
#         log_emit[:, k] = -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)
#     return log_emit

# def forward_backward(seq, pi, A, dM, db, dCov, K):
#     T = len(seq)
#     log_emit = get_log_emissions(seq, K, dM, db, dCov)
#     log_A    = np.log(A  + 1e-12)
#     log_pi   = np.log(pi + 1e-12)

#     log_alpha = np.zeros((T, K))
#     log_alpha[0] = log_pi + log_emit[0]
#     for t in range(1, T):
#         log_alpha[t] = log_emit[t] + np.logaddexp.reduce(
#             log_alpha[t-1][:, None] + log_A, axis=0
#         )

#     log_beta = np.zeros((T, K))
#     for t in range(T-2, -1, -1):
#         log_beta[t] = np.logaddexp.reduce(
#             log_A + log_emit[t+1] + log_beta[t+1], axis=1
#         )

#     log_gamma = log_alpha + log_beta
#     log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
#     gamma = np.exp(log_gamma)

#     log_xi = np.zeros((T-1, K, K))
#     for t in range(T-1):
#         log_xi[t] = (log_alpha[t][:, None] + log_A +
#                      log_emit[t+1] + log_beta[t+1])
#         log_xi[t] -= np.logaddexp.reduce(log_xi[t].ravel())
#     xi = np.exp(log_xi)

#     return gamma, xi

# def m_step(sequences, gammas, xis, K, D):
#     xi_sum  = sum(xi.sum(axis=0) for xi in xis) + np.eye(K) * KAPPA + 1e-8
#     A_new   = xi_sum / xi_sum.sum(axis=1, keepdims=True)
#     pi_new  = np.maximum(np.mean([g[0] for g in gammas], axis=0), 1e-8)
#     pi_new /= pi_new.sum()

#     dyn_M   = np.zeros((K, D, D))
#     dyn_b   = np.zeros((K, D))
#     dyn_cov = np.zeros((K, D, D))

#     for k in range(K):
#         W_sum  = np.zeros((D+1, D+1))
#         WY_sum = np.zeros((D+1, D))
#         for seq, gamma in zip(sequences, gammas):
#             X     = np.vstack([np.zeros(D), seq[:-1]])
#             dH    = seq
#             w     = gamma[:, k]
#             X_aug = np.hstack([X, np.ones((len(X), 1))])
#             W_sum  += (X_aug * w[:, None]).T @ X_aug
#             WY_sum += (X_aug * w[:, None]).T @ dH
#         coef = np.linalg.solve(W_sum + 1e-4 * np.eye(D+1), WY_sum)
#         dyn_M[k] = coef[:D].T
#         dyn_b[k] = coef[D]

#         num, den = np.zeros((D, D)), 1e-9
#         for seq, gamma in zip(sequences, gammas):
#             X   = np.vstack([np.zeros(D), seq[:-1]])
#             dH  = seq
#             err = dH - (X @ dyn_M[k].T + dyn_b[k])
#             w   = gamma[:, k]
#             num += (err * w[:, None]).T @ err
#             den += w.sum()
#         dyn_cov[k] = num / den + 1e-4 * np.eye(D)

#     return pi_new, A_new, dyn_M, dyn_b, dyn_cov

# def fit_slds(sequences, K=4, D=40, n_iters=50, verbose=False):
#     pi, A, dM, db, dCov = init_params(sequences, K, D)
#     for it in range(n_iters):
#         gammas, xis = [], []
#         for seq in sequences:
#             g, x = forward_backward(seq, pi, A, dM, db, dCov, K)
#             gammas.append(g)
#             xis.append(x)
#         pi, A, dM, db, dCov = m_step(sequences, gammas, xis, K, D)
#         if verbose and (it+1) % 10 == 0:
#             usage = np.mean([g.mean(0) for g in gammas], axis=0)
#             print(f"    iter {it+1:3d} | diag(A): {np.round(np.diag(A), 2)} "
#                   f"| usage: {np.round(usage, 2)}", flush=True)
#     return pi, A, dM, db, dCov, gammas

# # ── ANALYZE ───────────────────────────────────────────────────

# def analyze(A, gammas, label_seqs, K, verbose=True):
#     state_seqs = [np.argmax(g, axis=1) for g in gammas]

#     persistence = []
#     for s in state_seqs:
#         if len(s) < 2: continue
#         sw = np.count_nonzero(np.diff(s))
#         persistence.append(len(s) / (sw + 1))
#     avg_persistence = np.mean(persistence)

#     if verbose:
#         print(f"\n  Transition Matrix:")
#         print(np.round(A, 3))
#         print(f"  Self-transition: {np.round(np.diag(A), 3)}")
#         print(f"  Average persistence: {avg_persistence:.2f}")

#     confusion = np.zeros((K, len(STAGES)))
#     if label_seqs:
#         for state_seq, label_seq in zip(state_seqs, label_seqs):
#             for mode, label in zip(state_seq, label_seq[:len(state_seq)]):
#                 if label in STAGES:
#                     confusion[mode, STAGES.index(label)] += 1
#     confusion_norm = confusion / (confusion.sum(1, keepdims=True) + 1e-8)

#     if verbose and label_seqs:
#         print(f"\n  Mode → Stage Distribution:")
#         header = " ".join(f"{s[:5]:>7}" for s in STAGES)
#         print(f"  {'':>6} {header}")
#         for k in range(K):
#             row = " ".join(f"{confusion_norm[k,j]:7.2f}" for j in range(len(STAGES)))
#             print(f"  Mode {k}: {row}")
#         print(f"\n  Dominant stage per mode:")
#         for k in range(K):
#             dom = STAGES[np.argmax(confusion_norm[k])]
#             pct = confusion_norm[k].max()
#             print(f"    Mode {k}: {dom} ({pct:.1%})")

#     # summary metrics
#     avg_self_trans = np.mean(np.diag(A))
#     # mode specialization: average max probability per mode
#     avg_specialization = np.mean(confusion_norm.max(axis=1))

#     return avg_persistence, avg_self_trans, avg_specialization

# # ── SWEEP ────────────────────────────────────────────────────

# def sweep_K(sequences_reduced, labels_aligned, actual_D, config_name):
#     print(f"\n{'='*60}")
#     print(f"K SWEEP — {config_name}")
#     print('='*60)

#     summary = []
#     for k in K_SWEEP:
#         print(f"\n── K={k} ──")
#         pi, A, dM, db, dCov, gammas = fit_slds(
#             sequences_reduced, K=k, D=actual_D, n_iters=N_ITERS, verbose=(k==4)
#         )
#         avg_pers, avg_self, avg_spec = analyze(
#             A, gammas, labels_aligned, k, verbose=True
#         )
#         summary.append((k, avg_pers, avg_self, avg_spec))

#     print(f"\n── Summary for {config_name} ──")
#     print(f"  {'K':>3}  {'persistence':>12}  {'self-trans':>10}  {'specialization':>14}")
#     for k, p, s, sp in summary:
#         print(f"  {k:>3}  {p:>12.2f}  {s:>10.3f}  {sp:>14.3f}")

# # ── MAIN ──────────────────────────────────────────────────────

# if __name__ == "__main__":
#     path = "/home/abir19/scratch/abir19/new_rpc_math500_layer28_qwen14/all_sentences_features.pkl"

#     print("Loading last-token hidden states...")
#     sequences_last, label_seqs = load_raw_sequences(path, 'hidden_state_last')
#     print(f"Loaded {len(sequences_last)} sequences")

#     print("\nLoading mean-pooled hidden states...")
#     sequences_mean, _ = load_raw_sequences(path, 'hidden_state')

#     print("\n=== Last-token diagnostic ===")
#     run_diagnostic(sequences_last, label_seqs, tag="last-token")

#     print("\n=== Mean-pooled diagnostic ===")
#     run_diagnostic(sequences_mean, label_seqs, tag="mean-pooled")

#     CONFIGS = [
#         ("last-token | absolute | lda", sequences_last, label_seqs, 'absolute', 'lda'),
#         ("last-token | absolute | pca", sequences_last, label_seqs, 'absolute', 'pca'),
#         ("mean-pool  | delta    | pca", sequences_mean, label_seqs, 'delta',    'pca'),
#     ]

#     for config_name, sequences, lbls, mode, reducer in CONFIGS:
#         sequences_reduced, labels_aligned, scaler, red = make_representation(
#             sequences, lbls, mode=mode, reducer=reducer, n_components=D
#         )
#         actual_D = sequences_reduced[0].shape[1]
#         print(f"\nConfig: {config_name} | reduced dim: {actual_D}")
#         sweep_K(sequences_reduced, labels_aligned, actual_D, config_name)





import numpy as np
import pickle
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

K_SWEEP = range(1, 13)
D = 40
N_ITERS = 50
KAPPA = 10.0

STAGES = [
    "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
]

# ── LOAD ─────────────────────────────────────────────────────

def load_raw_sequences(path, hidden_key='hidden_state_last'):
    with open(path, 'rb') as f:
        features = pickle.load(f)
    p_map = defaultdict(list)
    for feat in features:
        if feat.get('stage', 'NEUTRAL') != 'NEUTRAL':
            p_map[feat['problem_id']].append(feat)
    sequences, label_seqs = [], []
    for pid in sorted(p_map.keys()):
        feats = sorted(p_map[pid], key=lambda x: x['sentence_idx'])
        if len(feats) >= 3:
            sequences.append(np.array([f[hidden_key] for f in feats]))
            label_seqs.append([f['stage'] for f in feats])
    return sequences, label_seqs

# ── CEBRA ─────────────────────────────────────────────────────

def _build_cebra_pairs(sequences, tau=1, n_neg=5):
    """Build (anchor, positive, [negatives]) index lists."""
    # flatten to per-sequence offsets
    all_vecs = np.concatenate(sequences, axis=0)
    offsets = np.cumsum([0] + [len(s) for s in sequences])

    anchors, positives, negatives = [], [], []
    for seq_i, seq in enumerate(sequences):
        T = len(seq)
        start = offsets[seq_i]
        for t in range(T - tau):
            anchors.append(start + t)
            positives.append(start + t + tau)
            # negatives: random from other positions (outside [t-tau, t+tau])
            forbidden = set(range(max(0, start + t - tau),
                                  min(len(all_vecs), start + t + tau + 1)))
            pool = [i for i in range(len(all_vecs)) if i not in forbidden]
            neg_idx = np.random.choice(pool, size=min(n_neg, len(pool)), replace=False)
            negatives.append(neg_idx)

    return all_vecs, np.array(anchors), np.array(positives), negatives


def fit_cebra(sequences, n_components=40, n_iters=500, lr=1e-3,
              tau=1, n_neg=5, temperature=0.1, hidden_dim=256, verbose=False):
    """
    Train a 2-layer MLP projection via InfoNCE contrastive loss.
    Temporal positives: sentences tau steps apart in the same sequence.
    Negatives: random sentences from anywhere else.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch required for CEBRA: pip install torch")

    all_vecs, anchors, positives, negatives = _build_cebra_pairs(
        sequences, tau=tau, n_neg=n_neg)

    in_dim = all_vecs.shape[1]
    scaler = StandardScaler()
    X = scaler.fit_transform(all_vecs).astype(np.float32)

    X_t = torch.from_numpy(X)

    net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, n_components),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    anchors_t   = torch.tensor(anchors, dtype=torch.long)
    positives_t = torch.tensor(positives, dtype=torch.long)

    for it in range(n_iters):
        net.train()
        Z = net(X_t)  # (N, n_components)
        Z = nn.functional.normalize(Z, dim=1)

        za = Z[anchors_t]       # (M, d)
        zp = Z[positives_t]     # (M, d)

        # build neg matrix (M, n_neg, d)
        neg_idx = torch.tensor(
            np.array([n[:n_neg] for n in negatives]), dtype=torch.long)
        zn = Z[neg_idx]         # (M, n_neg, d)

        pos_sim = (za * zp).sum(1, keepdim=True) / temperature   # (M, 1)
        neg_sim = torch.einsum('md,mnd->mn', za, zn) / temperature  # (M, n_neg)

        logits = torch.cat([pos_sim, neg_sim], dim=1)  # (M, 1+n_neg)
        labels = torch.zeros(len(logits), dtype=torch.long)
        loss = nn.functional.cross_entropy(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and (it + 1) % 100 == 0:
            print(f"    CEBRA iter {it+1:4d} | loss={loss.item():.4f}")

    net.eval()
    with torch.no_grad():
        Z_np = net(X_t).numpy()

    # split back into per-sequence arrays
    reduced, idx = [], 0
    for seq in sequences:
        n = len(seq)
        reduced.append(Z_np[idx:idx+n])
        idx += n

    return reduced, scaler, net


# ── REPRESENTATIONS ───────────────────────────────────────────

def make_representation(sequences, label_seqs=None, mode='delta',
                        reducer='pca', n_components=D):
    if mode == 'centered':
        seqs = [s - s.mean(0, keepdims=True) for s in sequences]
        labels_aligned = label_seqs
    elif mode == 'delta':
        seqs = [np.diff(s, axis=0) for s in sequences]
        labels_aligned = [l[1:] for l in label_seqs] if label_seqs else None
    else:
        seqs = sequences
        labels_aligned = label_seqs

    X_all = np.concatenate(seqs, axis=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    if reducer == 'lda' and labels_aligned is not None:
        y_all = np.concatenate([[STAGES.index(l) if l in STAGES else 0
                                 for l in ls] for ls in labels_aligned])
        n_comp = min(n_components, len(STAGES) - 1)
        red = LinearDiscriminantAnalysis(n_components=n_comp)
        X_red = red.fit_transform(X_scaled, y_all)
    else:
        red = PCA(n_components=n_components)
        X_red = red.fit_transform(X_scaled)

    reduced, idx = [], 0
    for s in seqs:
        n = len(s)
        reduced.append(X_red[idx:idx+n])
        idx += n

    return reduced, labels_aligned, scaler, red

# ── DIAGNOSTIC ────────────────────────────────────────────────

def run_diagnostic(sequences, label_seqs, tag=""):
    print(f"\n── Diagnostic {tag} ──")
    configs = [
        (mode, reducer)
        for mode in ['absolute', 'centered', 'delta']
        for reducer in ['pca', 'lda']
    ]
    results = []
    for mode, reducer in configs:
        try:
            reduced, _, _, _ = make_representation(
                sequences, label_seqs, mode=mode,
                reducer=reducer, n_components=min(D, len(STAGES)-1)
            )
            X_all = np.concatenate(reduced, axis=0)
            km_labels = KMeans(n_clusters=4, n_init=10,
                               random_state=42).fit_predict(X_all)
            sil = silhouette_score(X_all, km_labels,
                                   sample_size=min(2000, len(X_all)))
            print(f"  {mode:10s} | {reducer:3s}: silhouette={sil:.3f}  n={len(X_all)}")
            results.append((sil, mode, reducer))
        except Exception as e:
            print(f"  {mode:10s} | {reducer:3s}: ERROR {e}")

    best = max(results, key=lambda x: x[0])
    print(f"\n  Best: mode={best[1]}, reducer={best[2]} (silhouette={best[0]:.3f})")
    return best[1], best[2]

# ── SLDS ──────────────────────────────────────────────────────

def init_params(sequences, K, D):
    X_in, X_out = [], []
    for seq in sequences:
        for t in range(len(seq)-1):
            X_in.append(seq[t])
            X_out.append(seq[t+1] - seq[t])
    X_in  = np.array(X_in)
    X_out = np.array(X_out)

    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = km.fit_predict(X_out)

    dynamics_M = np.zeros((K, D, D))
    dynamics_b = np.zeros((K, D))
    noise_cov  = np.array([np.eye(D) for _ in range(K)])

    for k in range(K):
        mask = labels == k
        if mask.sum() < D + 2:
            dynamics_M[k] = 0.1 * np.eye(D)
            continue
        Xk_in  = X_in[mask]
        Xk_out = X_out[mask]
        X_aug  = np.hstack([Xk_in, np.ones((len(Xk_in), 1))])
        W, *_  = np.linalg.lstsq(X_aug, Xk_out, rcond=None)
        dynamics_M[k] = W[:D].T
        dynamics_b[k] = W[D]
        res = Xk_out - (Xk_in @ dynamics_M[k].T + dynamics_b[k])
        noise_cov[k] = np.cov(res.T) + 1e-3 * np.eye(D)

    pi = np.ones(K) / K
    A  = np.eye(K) * 0.7 + 0.3 / K
    return pi, A, dynamics_M, dynamics_b, noise_cov

def get_log_emissions(seq, K, dM, db, dCov):
    T, D = seq.shape
    log_emit = np.zeros((T, K))
    for k in range(K):
        sign, logdet = np.linalg.slogdet(dCov[k])
        inv_cov = np.linalg.inv(dCov[k])
        means = np.vstack([db[k], seq[:-1] @ dM[k].T + db[k]])
        diffs = seq - means
        mahal = np.sum((diffs @ inv_cov) * diffs, axis=1)
        log_emit[:, k] = -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)
    return log_emit

def forward_backward(seq, pi, A, dM, db, dCov, K):
    T = len(seq)
    log_emit = get_log_emissions(seq, K, dM, db, dCov)
    log_A    = np.log(A  + 1e-12)
    log_pi   = np.log(pi + 1e-12)

    log_alpha = np.zeros((T, K))
    log_alpha[0] = log_pi + log_emit[0]
    for t in range(1, T):
        log_alpha[t] = log_emit[t] + np.logaddexp.reduce(
            log_alpha[t-1][:, None] + log_A, axis=0
        )

    log_beta = np.zeros((T, K))
    for t in range(T-2, -1, -1):
        log_beta[t] = np.logaddexp.reduce(
            log_A + log_emit[t+1] + log_beta[t+1], axis=1
        )

    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    log_xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        log_xi[t] = (log_alpha[t][:, None] + log_A +
                     log_emit[t+1] + log_beta[t+1])
        log_xi[t] -= np.logaddexp.reduce(log_xi[t].ravel())
    xi = np.exp(log_xi)

    return gamma, xi

def m_step(sequences, gammas, xis, K, D):
    xi_sum  = sum(xi.sum(axis=0) for xi in xis) + np.eye(K) * KAPPA + 1e-8
    A_new   = xi_sum / xi_sum.sum(axis=1, keepdims=True)
    pi_new  = np.maximum(np.mean([g[0] for g in gammas], axis=0), 1e-8)
    pi_new /= pi_new.sum()

    dyn_M   = np.zeros((K, D, D))
    dyn_b   = np.zeros((K, D))
    dyn_cov = np.zeros((K, D, D))

    for k in range(K):
        W_sum  = np.zeros((D+1, D+1))
        WY_sum = np.zeros((D+1, D))
        for seq, gamma in zip(sequences, gammas):
            X     = np.vstack([np.zeros(D), seq[:-1]])
            dH    = seq
            w     = gamma[:, k]
            X_aug = np.hstack([X, np.ones((len(X), 1))])
            W_sum  += (X_aug * w[:, None]).T @ X_aug
            WY_sum += (X_aug * w[:, None]).T @ dH
        coef = np.linalg.solve(W_sum + 1e-4 * np.eye(D+1), WY_sum)
        dyn_M[k] = coef[:D].T
        dyn_b[k] = coef[D]

        num, den = np.zeros((D, D)), 1e-9
        for seq, gamma in zip(sequences, gammas):
            X   = np.vstack([np.zeros(D), seq[:-1]])
            dH  = seq
            err = dH - (X @ dyn_M[k].T + dyn_b[k])
            w   = gamma[:, k]
            num += (err * w[:, None]).T @ err
            den += w.sum()
        dyn_cov[k] = num / den + 1e-4 * np.eye(D)

    return pi_new, A_new, dyn_M, dyn_b, dyn_cov

def fit_slds(sequences, K=4, D=40, n_iters=50, verbose=False):
    pi, A, dM, db, dCov = init_params(sequences, K, D)
    for it in range(n_iters):
        gammas, xis = [], []
        for seq in sequences:
            g, x = forward_backward(seq, pi, A, dM, db, dCov, K)
            gammas.append(g)
            xis.append(x)
        pi, A, dM, db, dCov = m_step(sequences, gammas, xis, K, D)
        if verbose and (it+1) % 10 == 0:
            usage = np.mean([g.mean(0) for g in gammas], axis=0)
            print(f"    iter {it+1:3d} | diag(A): {np.round(np.diag(A), 2)} "
                  f"| usage: {np.round(usage, 2)}", flush=True)
    return pi, A, dM, db, dCov, gammas

# ── ANALYZE ───────────────────────────────────────────────────

def analyze(A, gammas, label_seqs, K, verbose=True):
    state_seqs = [np.argmax(g, axis=1) for g in gammas]

    persistence = []
    for s in state_seqs:
        if len(s) < 2: continue
        sw = np.count_nonzero(np.diff(s))
        persistence.append(len(s) / (sw + 1))
    avg_persistence = np.mean(persistence)

    if verbose:
        print(f"\n  Transition Matrix:")
        print(np.round(A, 3))
        print(f"  Self-transition: {np.round(np.diag(A), 3)}")
        print(f"  Average persistence: {avg_persistence:.2f}")

    confusion = np.zeros((K, len(STAGES)))
    if label_seqs:
        for state_seq, label_seq in zip(state_seqs, label_seqs):
            for mode, label in zip(state_seq, label_seq[:len(state_seq)]):
                if label in STAGES:
                    confusion[mode, STAGES.index(label)] += 1
    confusion_norm = confusion / (confusion.sum(1, keepdims=True) + 1e-8)

    if verbose and label_seqs:
        print(f"\n  Mode → Stage Distribution:")
        header = " ".join(f"{s[:5]:>7}" for s in STAGES)
        print(f"  {'':>6} {header}")
        for k in range(K):
            row = " ".join(f"{confusion_norm[k,j]:7.2f}" for j in range(len(STAGES)))
            print(f"  Mode {k}: {row}")
        print(f"\n  Dominant stage per mode:")
        for k in range(K):
            dom = STAGES[np.argmax(confusion_norm[k])]
            pct = confusion_norm[k].max()
            print(f"    Mode {k}: {dom} ({pct:.1%})")

    avg_self_trans = np.mean(np.diag(A))
    avg_specialization = np.mean(confusion_norm.max(axis=1))

    return avg_persistence, avg_self_trans, avg_specialization

# ── SWEEP ────────────────────────────────────────────────────

def sweep_K(sequences_reduced, labels_aligned, actual_D, config_name):
    print(f"\n{'='*60}")
    print(f"K SWEEP — {config_name}")
    print('='*60)

    summary = []
    for k in K_SWEEP:
        print(f"\n── K={k} ──")
        pi, A, dM, db, dCov, gammas = fit_slds(
            sequences_reduced, K=k, D=actual_D, n_iters=N_ITERS, verbose=(k==4)
        )
        avg_pers, avg_self, avg_spec = analyze(
            A, gammas, labels_aligned, k, verbose=True
        )
        summary.append((k, avg_pers, avg_self, avg_spec))

    print(f"\n── Summary for {config_name} ──")
    print(f"  {'K':>3}  {'persistence':>12}  {'self-trans':>10}  {'specialization':>14}")
    for k, p, s, sp in summary:
        print(f"  {k:>3}  {p:>12.2f}  {s:>10.3f}  {sp:>14.3f}")

# ── MAIN ──────────────────────────────────────────────────────

if __name__ == "__main__":
    path = "/home/abir19/scratch/abir19/new_rpc_math500_layer28_qwen14/all_sentences_features.pkl"

    print("Loading last-token hidden states...")
    sequences_last, label_seqs = load_raw_sequences(path, 'hidden_state_last')
    print(f"Loaded {len(sequences_last)} sequences")

    print("\nLoading mean-pooled hidden states...")
    sequences_mean, _ = load_raw_sequences(path, 'hidden_state')

    print("\n=== Last-token diagnostic ===")
    run_diagnostic(sequences_last, label_seqs, tag="last-token")

    print("\n=== Mean-pooled diagnostic ===")
    run_diagnostic(sequences_mean, label_seqs, tag="mean-pooled")

    # ── CEBRA projections ────────────────────────────────────
    print("\n=== Fitting CEBRA (last-token) ===")
    cebra_last, _, _ = fit_cebra(
        sequences_last, n_components=D, n_iters=500,
        tau=1, n_neg=5, temperature=0.1, verbose=True
    )

    print("\n=== Fitting CEBRA (mean-pooled) ===")
    cebra_mean, _, _ = fit_cebra(
        sequences_mean, n_components=D, n_iters=500,
        tau=1, n_neg=5, temperature=0.1, verbose=True
    )

    # silhouette check for CEBRA
    for tag, cebra_seqs in [("last-token", cebra_last), ("mean-pooled", cebra_mean)]:
        X_all = np.concatenate(cebra_seqs, axis=0)
        km_labels = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X_all)
        sil = silhouette_score(X_all, km_labels, sample_size=min(2000, len(X_all)))
        print(f"  CEBRA {tag}: silhouette={sil:.3f}  n={len(X_all)}")

    CONFIGS = [
        ("last-token | absolute | lda",  sequences_last, label_seqs, 'absolute', 'lda'),
        ("last-token | absolute | pca",  sequences_last, label_seqs, 'absolute', 'pca'),
        ("last-token | delta    | pca",  sequences_last, label_seqs, 'delta',    'pca'),
        ("mean-pool  | delta    | pca",  sequences_mean, label_seqs, 'delta',    'pca'),
    ]

    for config_name, sequences, lbls, mode, reducer in CONFIGS:
        sequences_reduced, labels_aligned, scaler, red = make_representation(
            sequences, lbls, mode=mode, reducer=reducer, n_components=D
        )
        actual_D = sequences_reduced[0].shape[1]
        print(f"\nConfig: {config_name} | reduced dim: {actual_D}")
        sweep_K(sequences_reduced, labels_aligned, actual_D, config_name)

    # ── CEBRA sweeps ─────────────────────────────────────────
    sweep_K(cebra_last, label_seqs, D, "last-token | CEBRA")
    sweep_K(cebra_mean, label_seqs, D, "mean-pool  | CEBRA")