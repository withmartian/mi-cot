# Reasoning Policy Controllers in Fine-Tuned Models

![Figure 1: Transplanting reasoning policy onto a base model](figure_page2_combined.png)

This repository supports the COLM 2026 paper "Reasoning Fine-Tuning Induces Persistent Latent Policy States" by providing code for: expressive reasoning model comparisons, discrete latent policy discovery, and controlled policy transplantation.

## Key ideas
- Fine-tuning for reasoning can change not only output quality but also the internal policy dynamics.
- We model chain-of-thought activations as a Switching Dynamical System (SDS).
- The framework recovers discrete latent policy states, measures persistence and transition structure, and tests causal relevance by transplanting the policy onto a base model.

## Repository structure
- `contrastive_gen/`: core CEBRA contrastive training code and API helpers
- `analysis/`: reasoning analysis scripts for causal tracing, policy extraction, and steering
- `generate_data/`: dataset creation and generation utilities
- `huggingface_scripts/`: HF repo management utilities
- `Locating_Reasoning_Policies.pdf`: paper PDF

## Installation
```bash
git clone https://github.com/withmartian/mi-cot.git
cd mi-cot
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Quickstart API
Use the package API for the cleanest workflow.

```python
from contrastive_gen import load_features, standardize_hidden_states, train_k_regime_model, save_model_artifacts

features, triplets = load_features("rpc_dataset/all_sentences_features.pkl")
X_torch, scaler = standardize_hidden_states(features)
result = train_k_regime_model(K=4, features=features, triplets=triplets, epochs=50)
print("Persistence:", result["persistence"])
print("State sequences for problem 0:", result["sequences"][0])
save_model_artifacts("rpc_results", result)
```

## Usage patterns
### 1. Load hidden-state features
The canonical dataset is expected at `rpc_dataset/all_sentences_features.pkl`.

```python
from contrastive_gen import load_features
features, triplets = load_features("rpc_dataset/all_sentences_features.pkl")
```

### 2. Standardize and train
```python
from contrastive_gen import standardize_hidden_states, train_k_regime_model
X_torch, scaler = standardize_hidden_states(features)
result = train_k_regime_model(K=4, features=features, triplets=triplets, epochs=60)
```

### 3. Inspect latent states
```python
sequences = result["sequences"]
persistence = result["persistence"]
print(f"Recovered {len(sequences)} problem traces with persistence {persistence:.3f}")
```

## Model loading and activation extraction
If you want to extract activations from a reasoning model and build your own dataset, use the scripts under `analysis/` and `generate_data/`.

Example:
- `analysis/gen_rpc_base.py` — generate causal matrices and sentence-level activation traces for base models
- `analysis/rpc.py` — extract reasoning anchors and label sentence types
- `analysis/cebra_rpc.py` — perform regime discovery with CEBRA and dynamics analysis

## Appendix: Practical workflow
1. Generate or locate sentence-level hidden-state data.
2. Standardize hidden-state vectors with `contrastive_gen.standardize_hidden_states()`.
3. Build triplets from adjacent same-problem points and random negatives using `load_features()`.
4. Train a discrete policy encoder via `train_k_regime_model()`.
5. Recover latent state sequences and analyze persistence, mixing, and transition structure.

## Good practices for COLM-style repos
- Prefer reusable APIs over one-off scripts.
- Keep dataset and model paths explicit.
- Document expected inputs, outputs, and the high-level analysis pipeline.
- Avoid verbose inline print debugging in production modules.

## Notes
- The main figure is extracted from `Locating_Reasoning_Policies.pdf`.
- This repo is organized to support reproducible reasoning policy discovery and analysis.
