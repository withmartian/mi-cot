# Experiments - Reasoning Policy Controllers Research

This directory contains all experimental code, implementations, and analysis for the RPC research project.

## Directory Structure

### Core Implementations (Root)
- **`cebra_EM.py`** - Core EM-based CEBRA implementation
- **`rpc.py`** - Main Reasoning Policy Controller implementation
- **`sds.py`** - Semantic Distributional Steering approach

### Subdirectories

#### `core/`
Shared training utilities and models used across experiments.
- `data_utils.py` - Data loading and preprocessing
- `models.py` - Model architectures
- `train.py` - Training loop utilities
- `losses.py` - Loss functions

#### `cebra_variants/`
Alternative CEBRA implementations and experimental variants.
- `cebra_base.py` - Basic CEBRA architecture
- `cebra_EM_base.py` - Base EM implementation
- `cebra_MoE.py` - Mixture of Experts variant of CEBRA
- `pca_EM.py` - PCA-based EM approach
- `cebra_causal.py` - Causal analysis variant
- `cebra_rpc.py` - CEBRA combined with RPC
- `cebra_rpc_old.py` - Earlier RPC variant (archived)
- `cebra_steering.py` - Steering-based variant
- `cebra_umap.py` - UMAP visualization variant

#### `rpc_variants/`
Alternative RPC implementations and variants.
- `gen_rpc_base.py` - Generative RPC base
- `generative_rpc.py` - Full generative RPC
- `steer_rpc.py` - Steering-based RPC
- `rpc_uncertainty.py` - Uncertainty quantification variant

#### `dataset_helpers/`
Dataset creation, loading, and preprocessing utilities.
- `create_dataset.py` - Main dataset creation script
- `create_dataset_multigpu.py` - Multi-GPU dataset creation
- `create_dataset_worker.py` - Worker process for parallel creation
- `data_llama.py` - Llama model data extraction
- `hard_dataset.py` - Hard example dataset creation

#### `analysis/`
Analysis scripts for mechanistic interpretability and probing.
- `causal_study.py` - Causal intervention analysis
- `cross_data_consistent.py` - Cross-dataset consistency checks
- `discriminability.py` - Discriminability analysis
- `gamma_fit.py` - Gamma distribution fitting
- `markovianity.py` - Markovian property analysis
- `membership.py` - Membership inference tests
- `swap_states.py` - State swapping experiments
- `transplant_passatk.py` - Pass@K transplantation analysis
- `figures.py` - Visualization utilities for analysis plots

#### `test_suite/`
Test scripts and experimental validations.
- `sufficiency.py` - Sufficiency testing
- `sufficient.py` - Alternative sufficiency test
- `sufficient2.py` - Another sufficiency variant
- `necessity.py` - Necessity testing
- `testing_hmm.py` - HMM testing
- Various ad-hoc test files

## Usage

All scripts assume you're in the repo root directory. To run a core implementation:

```bash
python experiments/cebra_EM.py
python experiments/rpc.py
```

For dataset creation:

```bash
python experiments/dataset_helpers/create_dataset.py
```

## Development Guidelines

- Core implementations (root level) are production-ready
- Variants are experimental - use them to test new ideas
- Add new experiments to appropriate subdirectory (or create new one if needed)
- Maintain imports from `core/` for shared utilities
