import os
import pickle
from collections import defaultdict
import numpy as np

def load_and_balance_data(path, limit_problems=500, max_triplets_per_pid=20):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}.")
    all_features = pickle.load(open(path, 'rb'))
    all_features = [f for f in all_features if f['problem_id'] < limit_problems]
    p_map = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map[f['problem_id']].append(i)
    triplets = []
    for pid, idxs in p_map.items():
        if len(idxs) < 2:
            continue
        for t in np.random.choice(len(idxs)-1, min(len(idxs)-1, max_triplets_per_pid), replace=False):
            triplets.append((idxs[t], idxs[t+1], np.random.randint(len(all_features))))
    return all_features, triplets