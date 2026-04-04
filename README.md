# Reasoning Policy Controllers in Fine-Tuned Models

## Research Question
How do fine-tuned reasoning models decide *when* to apply reasoning behaviors like verification and backtracking?
We investigate whether these decision-making mechanisms — **Reasoning Policy Controllers (RPCs)** — are learnable, isolable components that fine-tuning creates.

## Approach
We compare base and fine-tuned models to identify where they diverge, then analyze the internal states at these divergence points to understand the mechanisms driving reasoning behavior selection.

## Scope
- **Models**: DeepSeek-R1, Qwen, Llama, and related reasoning models
- **Datasets**: Mathematical reasoning (MATH, GSM8K, SVAMP), general knowledge (MMLU-Pro), visual reasoning (ARC-AGI)
- **Methods**: Mechanistic interpretability techniques including CEBRA representation learning, RPC isolation, causal intervention analysis, and pattern embedding studies

## Project Structure

```
mi-cot/
├── src/                       # Reusable library code
│   ├── utils/                # Utility functions and helpers
│   ├── models/               # Reusable model architectures
│   └── data/                 # Data processing and loading
├── experiments/              # Core implementations and experiments
│   ├── cebra_EM.py          # Main CEBRA EM implementation (reasoning stage discovery)
│   ├── rpc.py               # Main RPC implementation
│   ├── sds.py               # Semantic Distributional Steering
│   ├── core/                # Shared training utilities, models, losses
│   ├── cebra_variants/      # CEBRA variants (MoE, PCA, causal, steering)
│   ├── rpc_variants/        # RPC variants (generative, steering, uncertainty)
│   ├── dataset_helpers/     # Feature extraction from reasoning models
│   ├── analysis/            # Mechanistic interpretability analysis
│   └── test_suite/          # Validation and testing
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

## The Experiments Folder

The `experiments/` directory contains the core implementations of the project:

### Core Implementations (Root Level)

#### **`cebra_EM.py`** (477 lines)
Main EM-based CEBRA implementation for discovering semantic decision states (SDS) in LLM hidden activations.
- **Input**: Per-sentence activation features from intermediate layers of reasoning models (Llama 8B, Qwen 1.5B/14B) during CoT problem-solving
- **Reasoning stages discovered**: PROBLEM_SETUP, FACT_RETRIEVAL, PLAN_GENERATION, UNCERTAINTY_MANAGEMENT, SELF_CHECKING, RESULT_CONSOLIDATION, ACTIVE_COMPUTATION, FINAL_ANSWER_EMISSION
- **Process**:
  - Uses contractive learning (CEBRA) to encode trajectory sequences into low-dimensional representations
  - Fits HMM (K=1-12 regimes) with EM to discover which internal states correspond to reasoning stages
  - Learns transition matrix and state distributions
- **Output**: SDS embeddings, regime transitions, semantic alignment to labeled reasoning stages
- **Datasets**: GSM8K (Llama 8B, Qwen 1.5B/14B), SVAMP problem-solving traces

#### **`cebra_MoE.py`** - Mixture of Experts CEBRA Variant
Experimental MoE approach where different expert pools specialize in modeling specific reasoning stages.
- Allows per-stage heterogeneous dynamics

#### **`pca_EM.py`** - PCA-based EM Alternative
Reduces dimensionality via PCA instead of contrastive learning, then fits HMM regimes.
- Simpler baseline for comparing representation learning approaches

**Note:** Other variant implementations (`cebra_steering.py`, `cebra_rpc.py`, `rpc_variants/`) are older implementations and experimental explorations.

### Dataset Helpers

#### **`dataset_helpers/`** - CoT Activation Feature Extraction
Builds datasets of activations indexed per sentence in each CoT problem trajectory.
- **`create_dataset.py`**: Main pipeline extracting hidden states from intermediate model layers during reasoning
  - Processes problems one by one
  - Stores per-sentence activations with problem_id, sentence_idx, true/false reasoning stage labels
  - Output format: `[{problem_id, sentence_idx, hidden_state_last, stage, ...}, ...]` pickled to `.pkl`

- **`data_llama.py`**: Llama-specific feature extraction (batch processing)

- **`hard_dataset.py`**: Creates hard-vs-easy problem subsets
  - Identifies problems where **base model fails but reasoning model succeeds**
  - Provides targeted evaluation on problems requiring sophisticated reasoning
  - Used by `transplant_passatk.py` to measure improvement on truly difficult cases

- **`create_dataset_multigpu.py`**, **`create_dataset_worker.py`**: Parallel processing utilities for faster extraction

### Analysis & Mechanistic Interpretability

#### **`transplant_passatk.py`** - SDS Transplantation to Base Model
Transfers learned Semantic Decision States (SDS) from a reasoning model onto a base model to improve reasoning ability.
- **Process**:
  1. Learns SDS (CEBRA + HMM regimes) on reasoning model activations
  2. Fits linear decoder to map SDS back to activation space
  3. On base model, applies steering: nudges activations towards reasoning-model SDS
  4. Uses KL-regularized steering to avoid excessive deviation (maintains fluency)
- **Metric**: Pass@K improvement on hard problem set (problems base fails but reasoning model solves)
- **Finding**: Controllers transfer partially, suggesting universal reasoning patterns

#### **`trajectories_umap.py`** - Geometric Trajectory Visualization
Visualizes the learned SDS space and how reasoning unfolding moves through it.
- **Process**:
  1. Learns CEBRA embeddings per problem trajectory
  2. Fits UMAP on pooled embeddings to 2D for visualization
  3. Plots single trajectories against UMAP background, color-coded by SDS regime
- **Output**: Per-(model, dataset) figures showing trajectory geometry and cluster separation quality
- **Reveals**: Whether discovered regimes form coherent, separable clusters in representation space

#### **`cross_data_consistent.py`** - SDS Generalization Across Datasets
Validates that learned SDSs generalize across different benchmark datasets for a fixed model.
- **Process**:
  1. Train CEBRA + HMM once on dataset A
  2. Freeze encoder, embed all datasets (A, B, C, ...) with same frozen encoder
  3. Compare via multiple metrics:
     - Transition matrix similarity (Hungarian alignment, Frobenius norm)
     - Centroid cosine similarity (Procrustes-aligned)
     - Linear CKA between embeddings
     - Cross-fit ΔR²: SDS learned from dataset A predicts PCA trajectories in dataset B
- **Output**: Consistency scores across datasets, heatmaps showing alignment
- **Finding**: High consistency suggests SDS are dataset-generic model properties, not task-specific

#### **`membership.py`** - Semantic Attribution of SDS to Reasoning Behaviors
Attributes discovered SDS regimes to specific reasoning stages by measuring alignment.
- **Process**:
  1. Fit CEBRA embeddings and HMM regimes
  2. Hard assignment: map each sentence to its most-likely regime
  3. Cross-tabulation: measure which regimes fire during which human-annotated reasoning stages
  4. Soft assignment: use posterior inference probabilities from HMM
- **Output**: Alignment matrix, heatmaps, regime→stage confidence
- **Finding**: Validates that latent SDS regimes correspond to meaningful reasoning stages

#### **`causal_study.py`** - Necessity & Sufficiency Testing (Not directly shown, but referenced)
While detailed analysis is in ablations, this includes causal intervention on SDS:
- Remove or corrupt specific SDS regimes mid-trajectory
- Measure performance degradation to test **necessity**
- Transplant SDS modes across problems to test **sufficiency**

#### **Ablation Studies** (`analysis/ablations/`)
Validate modeling assumptions and justify architectural choices:
- **`layers_ablation.py`**: Tests which model layers contain learnable SDS
  - Compares K-regime fit quality across layers
  - Finds optimal layer depth for each model

- **`discriminability.py`**: Validates that discovered regimes are actually distinct
  - Measures separation quality between clusters
  - Compares to null models and shuffled regimes

- **`markovianity.py`**: Validates Markovian assumption (state transitions depend only on current state)
  - Compares HMM fit vs longer-range dependency models
  - Measures residual temporal correlations

- **`gamma_fit.py`**: Validates distributional assumptions of HMM states
  - Tests if state durations follow expected gamma distributions
  - Diagnostic for model fit quality

- **`swap_states.py`**: State swapping experiments to test sufficiency
  - Swaps SDS state sequences between different trajectories
  - Measures if problem can still be solved after swapping

- **`discriminability.py`**: Validates latent regime cluster separation quality

### Core Utilities

#### **`core/`** - Shared Training Infrastructure
- `train.py`: Training loops, optimization, evaluation
- `models.py`: Neural architectures used across experiments
- `data_utils.py`: Data loading and preprocessing
- `losses.py`: Loss functions for CEBRA and HMM training

### Test & Validation

#### **`test_suite/`** - Experiment Validation (Older Implementations)

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Dataset Extraction
First, extract features from model activations:
```bash
python experiments/dataset_helpers/create_dataset.py
```

### Running Core Experiments
```bash
# Discover reasoning stages via CEBRA-EM
python experiments/cebra_EM.py

# Train reasoning policy controllers
python experiments/rpc.py

# Apply steering-based control
python experiments/sds.py
```

### Analysis
```bash
# Causal intervention testing
python experiments/analysis/causal_study.py

# Cross-dataset consistency checks
python experiments/analysis/cross_data_consistent.py

# Membership inference tests
python experiments/analysis/membership.py

# Transplantation analysis
python experiments/analysis/transplant_passatk.py

# Generate figures
python experiments/analysis/figures.py
```

## Key Findings Summary

- **Semantic Decision States (SDS) discovered**: 6-8 consistent regimes in intermediate model layers corresponding to labeled reasoning stages
- **Cross-dataset consistency**: SDS learned from one dataset (GSM8K) remain aligned when embedded on others (SVAMP, MATH500)
- **Transferability**: SDS can be transplanted to base models via steering, improving Pass@K on hard problems
- **Necessity validated**: Causal interventions show removing key SDS regimes degrades performance
- **Markovian dynamics**: SDS transitions follow HMM assumptions without strong long-range dependencies
- **Layer locality**: Optimal SDS emerge in deep intermediate layers (L22-L31 for Llama, L47 for Qwen 14B)

## Key Files Reference

| File | Purpose | Key Output |
|------|---------|-----------|
| `experiments/cebra_EM.py` | SDS discovery via contrastive learning + HMM | Regime transitions, semantic alignment, embeddings |
| `experiments/analysis/transplant_passatk.py` | SDS steering to base model | Pass@K improvement, decoder mapping |
| `experiments/analysis/cross_data_consistent.py` | Cross-dataset generalization validation | Consistency scores, aligned heatmaps |
| `experiments/analysis/trajectories_umap.py` | Geometric visualization of SDS space | Per-model UMAP trajectory plots |
| `experiments/analysis/membership.py` | Attribution of SDS to reasoning stages | Stage-regime alignment matrix, confidence |
| `experiments/analysis/ablations/` | Model validation and assumptions | Layer importance, distributional fit, causality |
| `experiments/dataset_helpers/create_dataset.py` | CoT activation extraction | Per-sentence activation `.pkl` files |
| `experiments/dataset_helpers/hard_dataset.py` | Hard problem identification | Base-hard / reasoning-easy problem subsets |

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
