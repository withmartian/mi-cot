# Ablation: correct-trace subsets only

Tests whether the base vs RFT SDS gap survives when restricting to **correct** trajectories—especially **paired correct** (same problems where **both** base and RFT are graded correct).

No length matching or incorrect-trace controls.

## Subsets

| Subset | Definition |
|--------|------------|
| `all` | All problems in the manifest (up to `--limit-problems`) |
| `within_correct` | Per-model: only trajectories with a correct final answer |
| `paired_correct` | `problem_id` where **base and RFT** are both correct |

CEBRA is retrained **within each subset** (only allowed `problem_id`s enter triplets and trajectories).

## Manifest cache (correctness labels only)

Grading runs once per `(dataset, model folder, variant, layer, cot_data hash)` and is saved under **`manifest_cache/`** (default). No activations are copied—only JSON with `problem_id`, `is_correct`, answers, etc.

| Flag | Meaning |
|------|---------|
| `--manifest-cache-dir PATH` | Cache directory (default: `correct_only/manifest_cache`) |
| `--use-cached-manifest` / `--no-use-cached-manifest` | Read cache when valid (default: use) |
| `--rebuild-manifest` | Force re-grade and overwrite cache |

Paired-correct `problem_id` lists are cached under `manifest_cache/paired/`. Each run still copies manifests into its `results/...` folder for reproducibility.

## Scripts

| Script | Role |
|--------|------|
| `run_single_model.py` | One variant (`base` or `rft`) + one dataset → manifest + SDS metrics |
| `run_compare_base_rft.py` | Base + RFT for one dataset → all subsets + `comparison_report.json` |
| `run_full_analysis.py` | Sweep combos → `gap_summary.csv`, `gap_by_subset.png`, `outcome_summary.csv` |

## Quick start

```bash
python ablate_cot_trace_expms/correct_only/run_compare_base_rft.py \
  --dataset gsm8k --model-family qwen14 \
  --hf-cache-dir ./hf_cache --k-values 5 --k-focus 5

python ablate_cot_trace_expms/correct_only/run_full_analysis.py \
  --hf-cache-dir ./hf_cache --datasets gsm8k --model-families qwen14
```

## Desired outcome (pass/fail)

At `--k-focus` (default **K=5**), **primary pass** requires on **`paired_correct`**:

- RFT > Base for `persistence`, `mean_self_transition`, `delta_r2`, `K_eff`
- Paired RFT−Base gap retains ≥50% of the `all`-subset gap (persistence & ΔR²)

Exit code **0** when primary pass holds (`run_compare_base_rft.py`, `run_full_analysis.py`).

## Outputs

```
results/compare/<run_name>/<dataset>_<model_family>/
  base_manifest.json
  rft_manifest.json
  base/{all,within_correct,paired_correct}/summary.json
  rft/...
  comparison_report.json
```

`gap_by_subset.png` compares RFT−Base gaps for `all`, `within_correct`, and `paired_correct`.
