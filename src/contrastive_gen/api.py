import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from contrastive_gen.data_utils import load_and_balance_data
from contrastive_gen.models import CEBRA_MoE_Encoder, DynamicsMoE
from contrastive_gen.train import train_and_eval_k


def load_features(path: str, limit_problems: int = 500, max_triplets_per_pid: int = 20):
    """Load activation features and generate balanced training triplets."""
    return load_and_balance_data(path, limit_problems=limit_problems, max_triplets_per_pid=max_triplets_per_pid)


def standardize_hidden_states(features: list):
    """Convert feature hidden states to a standardized PyTorch tensor."""
    X_raw = np.array([f["hidden_state"] for f in features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_torch = torch.from_numpy(X_scaled).float()
    return X_torch, scaler


def build_state_sequences(features: list, states: np.ndarray):
    """Return a per-problem list of state assignments for downstream analysis."""
    sequences = defaultdict(list)
    for idx, state in enumerate(states):
        problem_id = features[idx]["problem_id"]
        sequences[problem_id].append(state)
    return dict(sequences)


def compute_persistence(sequences: dict):
    """Compute persistence scores from state sequences."""
    scores = []
    for seq in sequences.values():
        if len(seq) < 2:
            continue
        scores.append(1.0 - (np.sum(np.array(seq[1:]) != np.array(seq[:-1])) / (len(seq) - 1)))
    return float(np.mean(scores)) if scores else 0.0


def train_k_regime_model(
    K: int,
    features: list,
    triplets: list,
    epochs: int = 50,
    batch_size: int = 128,
    device: str = None,
):
    """Train a K-regime CEBRA model and return state assignments."""
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    X_torch, _ = standardize_hidden_states(features)
    persistence, final_mse, model, dyn, states = train_and_eval_k(
        K,
        features,
        X_torch.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        triplets,
        epochs=epochs,
        batch_size=batch_size,
    )

    return {
        "persistence": persistence,
        "final_mse": final_mse,
        "encoder": model,
        "dynamics": dyn,
        "states": states,
        "sequences": build_state_sequences(features, states),
    }


def save_model_artifacts(path: str, artifacts: dict):
    """Save model artifacts and derived data to disk."""
    os.makedirs(path, exist_ok=True)
    if "encoder" in artifacts:
        torch.save(artifacts["encoder"].state_dict(), os.path.join(path, "encoder.pt"))
    if "dynamics" in artifacts:
        torch.save(artifacts["dynamics"].state_dict(), os.path.join(path, "dynamics.pt"))
    if "states" in artifacts:
        with open(os.path.join(path, "states.pkl"), "wb") as f:
            pickle.dump(artifacts["states"], f)
    if "sequences" in artifacts:
        with open(os.path.join(path, "state_sequences.pkl"), "wb") as f:
            pickle.dump(artifacts["sequences"], f)
