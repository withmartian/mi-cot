# Run CEBRA-EM steering on 100 samples (SDS / Qwen 1.5B reasoning)

Hugging Face dataset **[SDS_train_gsm8k](https://huggingface.co/datasets/withmartian/SDS_train_gsm8k)** — subtree **qwen1.5b_reasoning**, layer 27.

## 1. Install

```bash
pip install huggingface_hub numpy torch scikit-learn
```

## 2. Set token and run

**Linux / macOS**

```bash
export HF_TOKEN="hf_..."  # your token; do not commit
cd /path/to/mi-cot
python exploration/run_steer_sds_hf.py
```

**Windows (PowerShell)**

```powershell
$env:HF_TOKEN = "hf_..."  # your token; do not commit
cd D:\Documents\_prog\repos_wlg1_shared\mi-cot
python exploration/run_steer_sds_hf.py
```

Or double‑click `exploration/run_steer_sds_hf.bat` (after setting Python in PATH).

## 3. Outputs

- **`rpc_cebra_em_sds100/sds_qwen15b_layer27_100.pkl`** — 100-sentence feature subset (cached after first download).
- **`rpc_cebra_em_sds100/steer_rpc_cebra_em_artifacts.pkl`** — Full pipeline artifacts (CEBRA model, EM params, steering cache, scaler, etc.).
- **`rpc_cebra_em_sds100/results_summary.txt`** — Human-readable summary: transition matrix **A**, regime counts, improvement notes.

Console will print the transition matrix and regime counts after the run.

## 4. LLM judge (`cebra_em_steering_inputDep.py`)

If you run the import-based pipeline with `SteerConfig(enable_openai_judge=True)`, set an API key **only via environment variable** (do not commit it):

```bash
export OPENAI_API_KEY="sk-..."
```

Outputs (under `save_dir`): `steering_judge_summary.json`, `steering_judge_report.txt`.
