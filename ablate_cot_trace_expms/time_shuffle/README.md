# Ablation (2): Time-shuffled trajectories before SDS

**Goal:** Sanity-check that SDS metrics depend on real temporal order. CEBRA is trained on **real** CoT order; only the **sequences passed to EM** are permuted.

**Good outcome:** On real order, RFT beats base on persistence, self-transition, ΔR², and similar metrics). 

Under shuffle, both models’ metrics drop sharply, and the RFT − base gap is smaller than on real order.

This means that "not any order gives the same sticky regimes and transitions". This requires the "logical order" to give the desired gap between base and RFT.

## EXAMPLE: Non-shuffled vs shuffled trajectory (same problem)

One GSM8K-style problem produces four sentence-level steps. CEBRA embeddings are written as `z_t`; SDS infers a discrete regime `s_t` at each timestep (example regimes shown for intuition only).

### Non-shuffled (real CoT order) — input to SDS

| Timestep | Sentence in original CoT | Embedding row | Example regime (SDS decode) |
|----------|--------------------------|---------------|-----------------------------|
| 0 | "Let me define the variables." | `z₀` | PLAN |
| 1 | "Substituting into the equation…" | `z₁` | COMPUTE |
| 2 | "Wait, let me check that step." | `z₂` | VERIFY |
| 3 | "So the final answer is 42." | `z₃` | ANSWER |

**Trajectory passed to EM:** `[z₀, z₁, z₂, z₃]`  
**Timeline matches how the model actually wrote the trace** — plan → compute → verify → answer.

SDS can learn transitions like PLAN→COMPUTE→VERIFY→ANSWER and long runs in the same regime when the reasoning stays in one mode for several sentences.

### Shuffled (same vectors, permuted order) — input to SDS

Same four sentences and the **same** `z₀…z₃` (each row still belongs to its original sentence). Only the **order of rows** changes, e.g. `perm = [2, 0, 3, 1]`:

| Timestep | Sentence (still tied to that row) | Embedding row | Example regime (SDS decode) |
|----------|-----------------------------------|---------------|-----------------------------|
| 0 | "Wait, let me check that step." | `z₂` | (varies) |
| 1 | "Let me define the variables." | `z₀` | (varies) |
| 2 | "So the final answer is 42." | `z₃` | (varies) |
| 3 | "Substituting into the equation…" | `z₁` | (varies) |

**Trajectory passed to EM:** `[z₂, z₀, z₃, z₁]`  
**Timeline does not match the model’s reasoning flow** — verify appears before setup; answer before substitution.

CEBRA was **not** retrained on this order; only SDS is refit on the scrambled sequence.

### Side-by-side summary

```
Non-shuffled (real):   z₀ → z₁ → z₂ → z₃     (setup → compute → verify → answer)
Shuffled (example):    z₂ → z₀ → z₃ → z₁     (verify → setup → answer → compute)
                       └── same four vectors, different positions ──┘
```

## What gets shuffled (what happens in code)

Each problem yields one trajectory from `all_sentences_features.pkl`. Each timestep is one **sentence-aligned step**:

| Index | Sentence (from CoT) | Stored in pickle | In trajectory |
|-------|---------------------|------------------|---------------|
| 0 | "Let me set up the equation." | `hidden_state_last`₀, `stage`₀ | row 0 of `z_seq` |
| 1 | "We substitute x = 3." | `hidden_state_last`₁, `stage`₁ | row 1 |
| 2 | "Therefore the answer is 12." | `hidden_state_last`₂, `stage`₂ | row 2 |

`shuffle_sequence_lists` applies one `perm` to `cebra_seqs`, `pca_seqs`, and `labels` in parallel:

```
Real:      z = [z₀, z₁, z₂]     labels = [L₀, L₁, L₂]
Shuffled:  z = [z₂, z₀, z₁]     labels = [L₂, L₀, L₁]    (perm = [2, 0, 1])
```

### Block shuffle (optional)

With `block_size=2` on the four-step example, blocks are `[0,1]` and `[2,3]`. If block order is reversed:

```
Non-shuffled:  [z₀, z₁, z₂, z₃]
Block-shuffled: [z₂, z₃, z₀, z₁]   # (compute+verify) block before (setup) block
```

Local order inside each block is preserved; global story order is still wrong.

## What this is meant to test

| Question | If real ≫ shuffled | If real ≈ shuffled |
|----------|-------------------|-------------------|
| Does SDS need sequential structure? | Yes — readout uses time order | Suspect order-invariant artifacts |
| Is persistence just static clustering? | Less likely | Investigate further |


## Design (pipeline)

1. Load `all_sentences_features.pkl` (from `generate_data/create_dataset.py`).
2. Train CEBRA → `cebra_seqs`, `pca_seqs`, `labels` (real order).
3. **Real order:** fit SDS (EM); report persistence, \(K_{\mathrm{eff}}\), self-transition, \(\Delta R^2\), BIC.
4. **Shuffled order:** `shuffle_sequence_lists(...)` then refit SDS (same \(K\), multiple shuffle seeds).
5. Compare real vs shuffled in `summary.json` / `summary.csv`.

Run the script separately on base and reasoning feature pickles and compare gaps.

## Usage

### Single model (real + shuffled in one run)

From repo root:

```bash
python ablate_cot_length_expms/time_shuffle/run_time_shuffle.py \
  --features-path /path/to/all_sentences_features.pkl \
  --limit-problems 500 \
  --shuffle-seeds 0 1 2 3 4 \
  --out-dir ablate_cot_length_expms/time_shuffle/results
```

### Base + RFT: four conditions, auto pass/fail

Runs base and RFT pickles (each: real-order SDS + shuffled-order SDS), then checks:

- Real ≫ shuffled per model (persistence, self-transition, ΔR²)
- RFT_real > Base_real
- (RFT − Base)_real > (RFT − Base)_shuffled

```bash
python ablate_cot_length_expms/time_shuffle/run_compare_base_rft.py \
  --base-features-path /path/to/base/all_sentences_features.pkl \
  --rft-features-path /path/to/rft/all_sentences_features.pkl \
  --limit-problems 500 \
  --k-values 4 5 6 \
  --k-focus 5 \
  --out-dir ablate_cot_length_expms/time_shuffle/results/compare_base_rft
```

Writes `results/compare_base_rft/<run>/base/summary.json`, `rft/summary.json`, and `comparison_report.json`. Exit code **0** if all checks pass at `--k-focus`, **1** otherwise.

Re-evaluate without retraining:

```bash
python ablate_cot_length_expms/time_shuffle/run_compare_base_rft.py \
  --base-features-path ... --rft-features-path ... \
  --skip-run --run-name <existing_run_folder_name> \
  --out-dir ablate_cot_length_expms/time_shuffle/results/compare_base_rft
```

Optional: cache CEBRA embeddings to skip retraining:

```bash
python ablate_cot_length_expms/time_shuffle/run_time_shuffle.py \
  --features-path /path/to/all_sentences_features.pkl \
  --cebra-cache ablate_cot_length_expms/time_shuffle/results/cebra_cache.pkl
```

## Outputs

- `results/<run_name>/summary.json` — per-\(K\) metrics for `real` and `shuffled` (mean/std over shuffle seeds)
- `results/<run_name>/summary.csv` — flat table for plotting

## Shuffle modes

- `full` (default): uniform random permutation of all sentence indices in a trajectory.
- `block`: permute blocks of `block_size` consecutive sentences (preserves local adjacency).
