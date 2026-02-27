"""
Infer SDS (Switching Dynamical System) states for each sentence.
Uses CEBRA-style encoder + MoE dynamics; trains if no checkpoint provided.
Output state per sentence in the same order as all_features.
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Import SDS components from mi-cot
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from contrastive_gen.models import CEBRA_MoE_Encoder, DynamicsMoE


def nce_loss(z, p, n, temp=0.05, device="cuda"):
    pos = (z * p).sum(dim=1, keepdim=True)
    neg = (z * n).sum(dim=1, keepdim=True)
    logits = torch.cat([pos, neg], dim=1) / temp
    return F.cross_entropy(logits, torch.zeros(len(z), dtype=torch.long, device=device))


def load_and_balance_data(path: str, limit_problems: int = None, max_triplets_per_pid: int = 20):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    all_features = pickle.load(open(path, "rb"))
    if limit_problems is not None:
        all_features = [f for f in all_features if f["problem_id"] < limit_problems]
    p_map = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map[f["problem_id"]].append(i)
    triplets = []
    pids = list(p_map.keys())
    for pid in pids:
        idxs = p_map[pid]
        if len(idxs) < 2:
            continue
        num_samples = min(len(idxs) - 1, max_triplets_per_pid)
        for _ in range(num_samples):
            t = np.random.randint(0, len(idxs) - 1)
            neg_pid = np.random.choice([p for p in pids if p != pid])
            neg_idx = np.random.choice(p_map[neg_pid])
            triplets.append((idxs[t], idxs[t + 1], neg_idx))
    return all_features, triplets


def train_and_infer_states(
    all_features: list,
    triplets: list,
    K: int = 4,
    d_h: int = 32,
    epochs: int = 50,
    device: str = "cuda",
) -> np.ndarray:
    """Train CEBRA MoE + dynamics and return state per sample in all_features order."""
    X_raw = np.array([f["hidden_state"] for f in all_features])
    scaler = StandardScaler()
    X_torch = torch.from_numpy(scaler.fit_transform(X_raw)).float().to(device)
    d_in = X_torch.shape[1]

    model = CEBRA_MoE_Encoder(d_in, d_h, K).to(device)
    dyn = DynamicsMoE(K, d_h).to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(dyn.parameters()), lr=1e-3)

    for epoch in range(epochs):
        tau = max(0.2, 1.5 * (0.92 ** epoch))
        curr_w_div = min(30.0, (epoch / 15.0) * 30.0)
        indices = np.random.permutation(len(triplets))
        for b in range(0, len(triplets), 128):
            b_idx = indices[b : b + 128]
            i_t = torch.tensor([triplets[x][0] for x in b_idx], device=device)
            p_t = torch.tensor([triplets[x][1] for x in b_idx], device=device)
            n_t = torch.tensor([triplets[x][2] for x in b_idx], device=device)
            h_i, s_i, _ = model(X_torch[i_t], temp=tau)
            h_p, s_p, _ = model(X_torch[p_t], temp=tau)
            h_n, _, _ = model(X_torch[n_t], temp=tau)
            h_pred = dyn(h_i, s_i)
            l_nce = nce_loss(h_pred, h_p, h_n, device=device)
            l_mse = F.mse_loss(h_pred, h_p)
            l_div = (s_i.mean(0) * torch.log(s_i.mean(0) + 1e-8)).sum()
            l_pers = torch.abs(s_i - s_p).mean()
            loss = l_nce + (10.0 * l_mse) + (curr_w_div * l_div) + (10.0 * l_pers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        _, s_final, _ = model(X_torch, temp=0.01)
        states = s_final.argmax(1).cpu().numpy()

    return states, scaler, model, dyn


def main():
    parser = argparse.ArgumentParser(description="Infer SDS states for each sentence")
    parser.add_argument("--features", default="rpc_dataset/all_sentences_features.pkl", help="Path to all_sentences_features.pkl")
    parser.add_argument("--out", default="entropy_on_sds_output", help="Output directory")
    parser.add_argument("--limit-problems", type=int, default=None, help="Only use problem_id < N")
    parser.add_argument("--K", type=int, default=4, help="Number of regimes")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--save-model", action="store_true", help="Save SDS model state dict and scaler")
    args = parser.parse_args()

    # Resolve relative --out against cwd so output dir is consistent with run_pipeline
    out_dir = os.path.abspath(args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_path = args.features if os.path.isabs(args.features) else os.path.join(REPO_ROOT, args.features)

    all_features, triplets = load_and_balance_data(features_path, limit_problems=args.limit_problems)
    print(f"Loaded {len(all_features)} sentences, {len(triplets)} triplets")

    states, scaler, model, dyn = train_and_infer_states(
        all_features, triplets, K=args.K, epochs=args.epochs, device=device
    )

    # Build table in same order as all_features
    result = [
        {"problem_id": f["problem_id"], "sentence_idx": f["sentence_idx"], "state": int(states[i])}
        for i, f in enumerate(all_features)
    ]

    out_path = os.path.join(out_dir, "sds_states.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"states": states, "result": result, "K": args.K}, f)
    print(f"Saved states (K={args.K}) to {out_path}")

    if args.save_model:
        torch.save({
            "encoder": model.state_dict(),
            "dynamics": dyn.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "K": args.K,
            "d_in": len(scaler.mean_),
        }, os.path.join(out_dir, "sds_model.pt"))
        print("Saved SDS model to sds_model.pt")


if __name__ == "__main__":
    main()
