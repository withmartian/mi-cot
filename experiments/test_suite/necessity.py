import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 35
BATCH_SIZE = 64
W_VAL = 5
ROLLOUT_STEPS = 3
K_SWEEP = list(range(1, 11))
NOISE_STD = 0.1
VAL_SPLIT = 0.1

PAIRS = [
    {"name": "Llama8B_GSM8K_L22",
     "rlvr": "/home/abir19/scratch/abir19/rpc_dataset_gsm8k_layer_22_llama8b",
     "base": "/home/abir19/scratch/abir19/rpc_base_gsm8k_layer_22_llama8b"},
    {"name": "Qwen14B_Math500_L28",
     "rlvr": "/home/abir19/scratch/abir19/rpc_dataset_math500_layer28_500_qwen_14",
     "base": "/home/abir19/rpc_base_math500_layer_28_qwen14"},
    {"name": "Qwen1.5_GSM8K_L20",
     "rlvr": "/home/abir19/scratch/abir19/rpc_dataset_gsm8k_layer_20_qwen1_5",
     "base": "/home/abir19/rpc_base_gsm8k_layer_20_qwen1_5B"},
]

# ── DATA ────────────────────────────────────────────────────

def load_and_preprocess(path, scaler=None):
    f_path = os.path.join(path, "all_sentences_features.pkl")
    if not os.path.exists(f_path):
        print(f"No features file: {f_path}"); return None, None, None

    with open(f_path, "rb") as f:
        features = pickle.load(f)
    if not features:
        print(f"Empty features: {f_path}"); return None, None, None

    X = np.array([f['hidden_state_last'] for f in features])
    if scaler is None:
        scaler = StandardScaler().fit(X)
    X_scaled = torch.from_numpy(scaler.transform(X)).float().to(DEVICE)

    p_map = defaultdict(list)
    for i, f in enumerate(features):
        p_map[f.get('problem_id', 'def')].append(i)

    return X_scaled, p_map, scaler


def build_rollout_dataset(X, p_map, window=W_VAL, rollout=ROLLOUT_STEPS):
    data = []
    for idxs in p_map.values():
        if len(idxs) <= window + rollout:
            continue
        for t in range(window - 1, len(idxs) - rollout):
            w = torch.cat([X[idxs[t - i]] for i in range(window)])
            y = torch.stack([X[idxs[t + j]] for j in range(1, rollout + 1)])  # [R, d]
            data.append({'h_w': w, 'h_c': X[idxs[t]], 'd_t': y})
    np.random.shuffle(data)
    return data

# ── MODELS ──────────────────────────────────────────────────

class SDSRouter(nn.Module):
    def __init__(self, d_in, K, window_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in * window_size, 256), nn.ReLU(), nn.Linear(256, K)
        )

    def forward(self, x_w, temp=1.0):
        logits = self.net(x_w)
        return F.gumbel_softmax(logits, tau=temp, hard=True), logits


class SDSExperts(nn.Module):
    def __init__(self, K, d_in):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d_in, d_in, bias=False) for _ in range(K)])

    def forward(self, h_c, s, shuffle=False, noise=False):
        B = h_c.size(0)
        if shuffle:
            perms = torch.stack([torch.randperm(s.size(1), device=s.device) for _ in range(B)])
            s = s[torch.arange(B, device=s.device).unsqueeze(1), perms]

        expert_outputs = torch.stack([m(h_c) for m in self.experts], dim=1)  # [B, K, d_in]
        if noise:
            expert_outputs += torch.randn_like(expert_outputs) * NOISE_STD
        return torch.sum(s.unsqueeze(-1) * expert_outputs, dim=1)  # [B, d_in]

# ── TRAINING ────────────────────────────────────────────────

def train_sds(data, d_in, K):
    n_val = max(1, int(len(data) * VAL_SPLIT))
    train_data, val_data = data[n_val:], data[:n_val]

    router = SDSRouter(d_in, K, W_VAL).to(DEVICE)
    experts = SDSExperts(K, d_in).to(DEVICE)
    opt = torch.optim.Adam(list(router.parameters()) + list(experts.parameters()), lr=1e-4)

    for epoch in range(EPOCHS):
        router.train(); experts.train()
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE]
            bw = torch.stack([x['h_w'] for x in batch])
            bc = torch.stack([x['h_c'] for x in batch])
            dt = torch.stack([x['d_t'] for x in batch])  # [B, R, d_in]

            s, _ = router(bw)
            loss = sum(
                F.mse_loss(experts(bc, s), dt[:, step, :])
                for step in range(ROLLOUT_STEPS)
            )
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 5 == 0:
            router.eval(); experts.eval()
            with torch.no_grad():
                vw = torch.stack([x['h_w'] for x in val_data])
                vc = torch.stack([x['h_c'] for x in val_data])
                vt = torch.stack([x['d_t'] for x in val_data])
                vs, _ = router(vw, temp=0.1)
                val_loss = sum(
                    F.mse_loss(experts(vc, vs), vt[:, step, :])
                    for step in range(ROLLOUT_STEPS)
                )
            print(f"  epoch {epoch+1} | val_loss={val_loss.item():.4f}")

    return router, experts, val_data

# ── EVALUATION ──────────────────────────────────────────────

def evaluate_sds(router, experts, data, mode='baseline'):
    router.eval(); experts.eval()
    cos_steered, cos_trivial = [], []

    with torch.no_grad():
        for x in data:
            h_w    = x['h_w'].unsqueeze(0)
            base   = x['h_c'].unsqueeze(0)   # [1, d_in]
            target = x['d_t']                # [R, d_in]

            s, _ = router(h_w, temp=0.1)

            for step in range(ROLLOUT_STEPS):
                t_step  = target[step].unsqueeze(0)
                steered = experts(base, s, shuffle=(mode == 'shuffle'), noise=(mode == 'noise'))

                cos_steered.append(F.cosine_similarity(steered, t_step).item())
                cos_trivial.append(F.cosine_similarity(base, t_step).item())

    mean_steered = np.mean(cos_steered)
    mean_trivial = np.mean(cos_trivial)
    return mean_steered, mean_trivial, mean_steered - mean_trivial

# ── RUN ─────────────────────────────────────────────────────

def run_pair(pair):
    print(f"\n>>> {pair['name']}")
    X_rlvr, p_rlvr, scaler = load_and_preprocess(pair['rlvr'])
    X_base, p_base, _      = load_and_preprocess(pair['base'], scaler=scaler)
    if X_rlvr is None or X_base is None:
        print(f"Skipping {pair['name']} — missing data."); return

    dataset = build_rollout_dataset(X_rlvr, p_rlvr)
    if not dataset:
        print(f"Skipping {pair['name']} — empty rollout dataset."); return

    print(f"Dataset size: {len(dataset)}")
    d_in = X_rlvr.shape[1]

    for K in K_SWEEP:
        print(f"\nTraining K={K}")
        router, experts, val_data = train_sds(dataset, d_in, K)

        for mode in ['baseline', 'shuffle', 'noise']:
            cos_s, cos_t, delta = evaluate_sds(router, experts, val_data, mode)
            print(f"  K={K} [{mode:8s}] | steered={cos_s:.4f} | trivial={cos_t:.4f} | Δ={delta:+.4f}")


def main():
    for pair in PAIRS:
        run_pair(pair)

if __name__ == "__main__":
    main()