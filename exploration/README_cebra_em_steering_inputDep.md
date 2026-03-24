# `cebra_em_steering_inputDep.py`

End-to-end **CEBRA embedding + SLDS EM** on sentence-level activations, then **KL-regularized steering** in latent space mapped into activation edits. Implementation lives in [`cebra_em_steering_inputDep.py`](./cebra_em_steering_inputDep.py); math utilities come from [`cebra_EM.py`](./cebra_EM.py).

## Command-line usage

Run from the `exploration/` directory (the script adds this folder to `sys.path`), or from the repo root:

```bash
python exploration/cebra_em_steering_inputDep.py --help
```

## Data sources and subset size

You can use any **compatible** Hugging Face dataset or local pickle. “Compatible” means the pickled list-of-dicts format expected by `cebra_EM.load_and_prepare_cebra` (e.g. `hidden_state_last`, `problem_id`, `stage`, and fields needed for temporal triplets). Random tabular or HDF5 data is not supported without converting to that layout.

| Flag | Role |
|------|------|
| `--local-data PATH` | Use this features `.pkl` directly. |
| `--dataset REPO_OR_URL_OR_PATH` | HF dataset id (`org/name`), `hf://…`, `https://huggingface.co/datasets/…`, or local `.pkl` / directory containing `all_sentences_features.pkl` (`hf_steering_data_io.resolve_cli_data_path`). |
| `--model SUBPATH` | Path inside the Hub dataset to features (e.g. `qwen1.5b_reasoning/layer_27`), or a preset from `sds_train_gsm8k_hf` (including HF model ids like `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`). If the value has no `.pkl` suffix, `all_sentences_features.pkl` is appended under that subpath. |
| `--num-samples` / `--hf-samples N` | When downloading from Hub, keep the first *N* rows with `stage != "NEUTRAL"` and write a cache under `--save-dir`. |
| `--limit-problems` | Passed to `load_and_prepare_cebra` when `data_path` is an **existing** local file. |
| `--no-hf-fallback` | Fail if the configured `data_path` is missing instead of downloading. |
| `--save-dir DIR` | Output directory and location of the HF subset cache. |
| `--no-openai-judge` | Skip the LLM judge (no `OPENAI_API_KEY`). |

**Auth:** `HF_TOKEN` for gated Hub files; `OPENAI_API_KEY` for the judge.

Related modules: `hf_steering_data_io.py` (CLI + download/subset), `sds_train_gsm8k_hf.py` (presets and Hub paths).

## Examples

```bash
python cebra_em_steering_inputDep.py --num-samples 500 --no-openai-judge --save-dir out_run
python cebra_em_steering_inputDep.py --model qwen1.5b_reasoning/layer_27 --num-samples 500 --save-dir out_run
python cebra_em_steering_inputDep.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --num-samples 200
python cebra_em_steering_inputDep.py --dataset your-org/your-dataset --model your_subpath/layer_27 --num-samples 300 --save-dir out_hf
python cebra_em_steering_inputDep.py --local-data path/to/all_sentences_features.pkl --limit-problems 500
```

If `SteerConfig.data_path` is missing and `hf_auto_download_if_missing` is true (default), `run_pipeline` calls `hf_steering_data_io.resolve_hf_subset_data_path` to download and cache from the configured Hub repo and filename.

## Outputs (under `--save-dir`)

| Artifact | Description |
|----------|-------------|
| `steer_cebra_em_imported_artifacts.pkl` | Main bundle: SLDS parameters (`pi`, `A`, `dM`, `db`, `dCov`), `per_sample_state`, `included_indices`, scaler and `decoder_W`, per-row `steering_cache`, and copies of eval/judge reports (`steering_eval`, `steering_logit_lens_eval`, `state_behavior_profiles`, `steering_judge_eval`). |
| `steering_eval_summary.json` | Machine-readable aggregates from `compute_steering_reports` (`global`, `by_target`, `by_transfer`, `rows`). |
| `steering_eval_report.txt` | Human-readable steering metrics with short “what they mean” and “how to analyze” sections. |
| `steering_logit_lens_summary.json` | Logit-lens evaluation dict (enabled flag, metrics, per-row rows when available). |
| `steering_logit_lens_report.txt` | Short text summary of logit-lens results. |
| `steering_judge_summary.json` | LLM judge output (rows + summary rates). |
| `steering_judge_report.txt` | Human-readable judge report with interpretation notes. |
| `sds_hf{N}_*.pkl` (optional) | Cached subset written when data is fetched from Hub (name depends on `--model` / sample count). |

## How to analyze

1. **Start with `steering_eval_report.txt`** — global `steerability_pct`, `target_prob_lift`, `kl_q_p`, `coherence_penalty`, and slices by target regime and `source->target` transfer. Use `steering_eval_summary.json` for programmatic plots or tables.

2. **Interpretation (from the report text):**
   - Higher **steerability_pct** / **target_prob_lift** usually means the tilted policy `q*` is doing what you asked in discrete regime space.
   - **coherence_penalty** is relative activation edit size ‖Δx‖/‖x‖; keep it bounded if you care about “small” nudges.
   - **kl_q_p** and **l1_policy_shift** measure how far `q*` moved from the natural `p`.

3. **Logit-lens** — If `steering_logit_lens_report.txt` shows `enabled: true`, read target logit lift and margin lift; if disabled, check `reason` in the JSON.

4. **LLM judge** — Compare `steering_judge_report.txt` (behavior vs state vs coherence) with the numeric metrics. Mismatches (high policy steerability but low judge state success, etc.) are called out in the judge report’s analysis section.

5. **Repro / custom analysis** — Load `steer_cebra_em_imported_artifacts.pkl` in Python; keys mirror the payload built in `run_pipeline` (config, EM parameters, `steering_cache` per index, and nested eval dicts).

## Pipeline hook

Import `SteerConfig`, `run_pipeline`, and helpers from this module in other scripts; the `__main__` block only wires CLI → config → `run_pipeline` and prints the artifact path.
