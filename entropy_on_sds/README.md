# Entropy on SDS States

This folder implements experiments that **track token-level and sentence-level entropy across SDS (Switching Dynamical System) regimes**, to test whether entropy correlates with certain states or state transitions (e.g. "high-entropy transition state" in the SDS writeups).

## Prerequisites

- **Dataset**: Run `exploration/create_dataset.py` first so that:
  - `rpc_dataset/all_sentences_features.pkl` exists (or set `--features` / `--cot` to your paths)
  - `rpc_dataset/cot_data.pkl` exists
- **Python**: Same environment as the rest of mi-cot (transformers, torch, sklearn, pandas, matplotlib, seaborn).

## Pipeline Overview

1. **Extract sentence entropy** (`extract_sentence_entropy.py`)  
   For each problem, runs the reasoning model on the full prompt, computes next-token entropy at each position, and aggregates per sentence (mean and last-token entropy). Output order matches `all_sentences_features.pkl`.

2. **Infer SDS states** (`infer_sds_states.py`)  
   Loads the same features, trains a CEBRA-style encoder + MoE dynamics (or loads a saved SDS), and infers one discrete state per sentence. Saves `sds_states.pkl` and optionally `sds_model.pt`.

3. **Analyze** (`analyze_entropy_states.py`)  
   Builds a single table (sentence entropy + state + stage + is_transition) and runs:
   - **Exp 1**: Entropy by regime (distribution, mean, boxplot)
   - **Exp 2**: Mean entropy at transition vs non-transition steps
   - **Exp 3**: Mean entropy by transition type (from state i → to state j)
   - **Exp 4**: SDS transition entropy (per regime) vs mean sentence entropy in that regime
   - **Exp 5**: Heatmap stage × regime with color = mean sentence entropy
   - **Exp 6**: Trajectory plots (sentence index vs entropy, colored by state) for a few problems

## Quick Run

From the **mi-cot** repo root:

```bash
cd entropy_on_sds
python run_pipeline.py --features ../rpc_dataset/all_sentences_features.pkl --cot ../rpc_dataset/cot_data.pkl
```

For a faster test (fewer problems):

```bash
python run_pipeline.py --limit-problems 100
```

Outputs go to `entropy_on_sds_output/` by default (or `--out DIR`).

## Running Steps Individually

```bash
# 1. Extract entropy (needs GPU and the reasoning model)
python extract_sentence_entropy.py --features ../rpc_dataset/all_sentences_features.pkl --cot ../rpc_dataset/cot_data.pkl --out entropy_on_sds_output

# 2. Infer SDS states (trains CEBRA-style SDS)
python infer_sds_states.py --features ../rpc_dataset/all_sentences_features.pkl --out entropy_on_sds_output --K 4 --save-model

# 3. Analyze
python analyze_entropy_states.py --out entropy_on_sds_output --features ../rpc_dataset/all_sentences_features.pkl
```

## Output Files

| File | Description |
|------|-------------|
| `sentence_entropy.pkl` / `.csv` | Per-sentence entropy (mean, last-token) in features order |
| `sds_states.pkl` | Inferred state per sentence; optional `sds_model.pt` |
| `exp1_entropy_by_regime.csv` / `.png` | Entropy statistics and boxplot by regime |
| `exp2_transition_vs_non.csv` / `.png` | Entropy at transition vs non-transition |
| `exp3_entropy_by_transition_type.csv` / `.png` | Mean entropy by (from_state, to_state) |
| `exp4_*.csv` / `.png` | Transition entropy vs sentence entropy per regime |
| `exp5_stage_regime_entropy.csv` / `.png` | Stage × regime heatmap (if stage labels loaded) |
| `exp6_trajectory_pid_*.png` | Entropy and state over time for sample problems |
| `analysis_summary.json` | Summary metrics for all experiments |

## Dashboard

After the pipeline has produced the output above, build an HTML dashboard:

```bash
cd entropy_on_sds
python build_entropy_sds_dashboard.py --out entropy_on_sds_output
```

Optional: pass `--features ../rpc_dataset_small/all_sentences_features.pkl` (or your features path) to include stage labels in the per-sentence table.

Open `entropy_on_sds_output/dashboard_entropy_sds.html` in a browser. If images do not load (e.g. when opening the file via `file://`), serve the folder:

```bash
python -m http.server 8000 --directory entropy_on_sds_output
```

Then open `http://localhost:8000/dashboard_entropy_sds.html`.

## Commands summary

| Goal | Command |
|------|--------|
| **Generate pipeline output** | `cd mi-cot/entropy_on_sds && python run_pipeline.py --features ../rpc_dataset_small/all_sentences_features.pkl --cot ../rpc_dataset_small/cot_data.pkl` |
| **Generate dashboard** | `cd mi-cot/entropy_on_sds && python build_entropy_sds_dashboard.py --out entropy_on_sds_output` |

Use `../rpc_dataset/...` if your dataset is in `mi-cot/rpc_dataset/` instead of `rpc_dataset_small`.

## Related Work

- **SDS early results / Locating_Reasoning_Policies**: Regime 1 as "High-Entropy Transition State"; these experiments test whether **measured** next-token entropy is higher in that regime or at transitions.
- **Tie_all_hypotheses**: Tracks 1 (uncertainty) and 2 (SDS); linking high-entropy forks to regime transitions.
- **Qwen "High-entropy minority tokens"**: Token-level critical forks; we aggregate to sentence-level and align to SDS states.
