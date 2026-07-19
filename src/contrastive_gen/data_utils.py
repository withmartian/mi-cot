import os
import pickle
from collections import defaultdict
import numpy as np


def load_and_balance_data(path: str, limit_problems: int = 500, max_triplets_per_pid: int = 20):
    """Load activation features and build balanced triplets for contrastive training."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}.")

    all_features = pickle.load(open(path, 'rb'))
    all_features = [f for f in all_features if f['problem_id'] < limit_problems]

    p_map = defaultdict(list)
    for idx, feature in enumerate(all_features):
        p_map[feature['problem_id']].append(idx)

    rng = np.random.default_rng()
    triplets = []
    pids = list(p_map.keys())

    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2:
            continue

        num_samples = min(len(idxs) - 1, max_triplets_per_pid)
        for sample_idx in rng.choice(len(idxs) - 1, num_samples, replace=False):
            anchor_idx = idxs[sample_idx]
            positive_idx = idxs[sample_idx + 1]
            negative_pid = rng.choice([other for other in pids if other != pid])
            negative_idx = rng.choice(p_map[negative_pid])
            triplets.append((anchor_idx, positive_idx, negative_idx))

    return all_features, triplets



# def load_and_balance_data(path, limit_problems=500, max_triplets_per_pid=20):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Could not find {path}.")
#     all_features = pickle.load(open(path, 'rb'))
#     all_features = [f for f in all_features if f['problem_id'] < limit_problems]
#     p_map = defaultdict(list)
#     for i, f in enumerate(all_features): p_map[f['problem_id']].append(i)
#     triplets = []
#     pids = list(p_map.keys())
#     for pid in pids:
#         idxs = p_map[pid]
#         if len(idxs) < 2: 
#             continue
#         num_samples = min(len(idxs)-1, max_triplets_per_pid)
#         for t in np.random.choice(len(idxs)-1, num_samples, replace=False):
#             triplets.append((idxs[t], idxs[t+1], np.random.choice(p_map[np.random.choice([p for p in pids if p != pid])])))
#     return all_features, triplets
