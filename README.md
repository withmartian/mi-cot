# Reasoning Policy Controllers in Fine-Tuned Models

## Research Question
How do fine-tuned reasoning models decide *when* to apply reasoning behaviors like verification and backtracking?
We investigate whether these decision-making mechanisms — **Reasoning Policy Controllers (RPCs)** — are learnable, isolable components that fine-tuning creates.

## Approach
We compare base and fine-tuned models to identify where they diverge, then analyze the internal states at these divergence points to understand the mechanisms driving reasoning behavior selection.

## Scope
- **Models**: DeepSeek-R1, Qwen, and related reasoning models
- **Datasets**: Mathematical reasoning (MATH, GSM8K), general knowledge (MMLU-Pro), visual reasoning (ARC-AGI), tool-use tasks
- **Methods**: Mechanistic interpretability techniques including crosscoders, attention analysis, activation patching, and feature decomposition

## Project Structure

```
mi-cot/
├── src/                       # Reusable library code
│   ├── utils/                # Utility functions and helpers
│   ├── models/               # Reusable model architectures
│   └── data/                 # Data processing and loading
├── experiments/              # Core implementations and variants
│   ├── cebra_EM.py          # Main EM-based CEBRA implementation
│   ├── rpc.py               # Main RPC implementation
│   ├── sds.py               # Semantic Distributional Steering
│   ├── core/                # Shared training utilities
│   ├── cebra_variants/      # Alternative CEBRA implementations
│   ├── rpc_variants/        # RPC variants (steering, generative, etc.)
│   ├── dataset_helpers/     # Dataset creation utilities
│   ├── analysis/            # Mechanistic analysis scripts
│   └── test_suite/          # Test and validation scripts
├── notebooks/               # Jupyter notebooks (exploration, visualization)
├── configs/                 # Configuration files (JSON/YAML)
├── scripts/                 # Utility scripts (HF integration, etc.)
├── data/                    # Pre-extracted features and datasets
├── results/                 # Experiment results and figures
├── outputs/                 # Generated logs and metrics
├── checkpoints/             # Model checkpoints
├── models/                  # Additional model files
├── docs/                    # Project documentation
├── requirements.txt
└── README.md

```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running Core Implementations
```bash
# CEBRA-EM
python experiments/cebra_EM.py

# RPC
python experiments/rpc.py

# SDS
python experiments/sds.py
```

### Dataset Preparation
```bash
python experiments/dataset_helpers/create_dataset.py
```

### Analysis
```bash
python experiments/analysis/causal_study.py
```

## Key Files

| File | Purpose |
|------|---------|
| `experiments/cebra_EM.py` | Core EM implementation for CEBRA |
| `experiments/rpc.py` | Main Reasoning Policy Controller |
| `experiments/sds.py` | Semantic distributional steering method |
| `experiments/pca_EM.py` | PCA-based alternative approach |
| `experiments/cebra_MoE.py` | Mixture of Experts variant |

## Directory Guidelines

- **`src/`** - Reusable library code; import with `from src.utils import ...`
- **`experiments/`** - Reproducible experiments; each script is standalone executable
- **`notebooks/`** - Exploratory and visualization notebooks
- **`configs/`** - Hyperparameter configs referenced by experiment scripts
- **`results/`** - Final results and figures (version controlled)
- **`outputs/`** - Temporary outputs, logs, metrics (add to `.gitignore`)
- **`checkpoints/`** - Model weights (add to `.gitignore`)
- **`docs/`** - Method descriptions and documentation (Markdown)

## Status

Active research. Details and findings will be shared as work progresses.
