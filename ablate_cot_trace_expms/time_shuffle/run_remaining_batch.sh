#!/usr/bin/env bash
# Run remaining model×dataset time-shuffle comparisons; cleanup pkls/caches after each.
set -euo pipefail
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN}"
cd /workspace/mi-cot
PY=python3
RUNNER=ablate_cot_trace_expms/time_shuffle/run_compare_base_rft.py
OUT_ROOT=ablate_cot_trace_expms/time_shuffle/results

cleanup_data() {
  rm -rf /workspace/data/SDS_train_gsm8k /workspace/data/SDS_train_svamp \
    /workspace/data/SDS_train_mmlu-pro /workspace/data/SDS_math500_test 2>/dev/null || true
}

cleanup_run_artifacts() {
  local run_dir="$1"
  find "$run_dir" -name 'cebra_cache_*.pkl' -delete 2>/dev/null || true
}

download_pair() {
  local repo="$1" base_f="$2" rft_f="$3" local_name="$4"
  $PY << PY
import os
from huggingface_hub import hf_hub_download
token = os.environ["HF_TOKEN"]
repo = "$repo"
local_dir = "/workspace/data/$local_name"
for f in ["$base_f", "$rft_f"]:
    p = hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", token=token, local_dir=local_dir)
    print("downloaded", p)
PY
}

run_job() {
  local tag="$1" repo="$2" local_name="$3" base_f="$4" rft_f="$5" limit="$6" out_name="$7"
  echo ""
  echo "========== JOB: $tag =========="
  cleanup_data
  download_pair "$repo" "$base_f" "$rft_f" "$local_name"
  local base_p="/workspace/data/${local_name}/${base_f}"
  local rft_p="/workspace/data/${local_name}/${rft_f}"
  local out_dir="${OUT_ROOT}/${out_name}"
  mkdir -p "$out_dir"
  set +e
  $PY "$RUNNER" \
    --base-features-path "$base_p" \
    --rft-features-path "$rft_p" \
    --limit-problems "$limit" \
    --k-values 4 5 6 \
    --k-focus 5 \
    --shuffle-seeds 0 1 2 3 4 \
    --out-dir "$out_dir"
  local ec=$?
  set -e
  local run_dir
  run_dir=$(ls -td "${out_dir}"/*/ 2>/dev/null | head -1)
  if [[ -n "$run_dir" ]]; then
    cleanup_run_artifacts "$run_dir"
  fi
  cleanup_data
  echo "JOB $tag exit_code=$ec run_dir=$run_dir"
  return $ec
}

# --- Remaining jobs (skip completed: qwen15 math500+gsm8k, qwen14/llama8 math500) ---

# GSM8K
run_job "qwen14_gsm8k" "withmartian/SDS_train_gsm8k" "SDS_train_gsm8k" \
  "qwen_14B_base/layer_28/all_sentences_features.pkl" \
  "qwen_14b_reasoning/layer_28/all_sentences_features.pkl" \
  2000 "compare_qwen14_gsm8k" || true

run_job "llama8_gsm8k" "withmartian/SDS_train_gsm8k" "SDS_train_gsm8k" \
  "llama_8B_base/layer_22/all_sentences_features.pkl" \
  "llama_8b_reasoning/layer_22/all_sentences_features.pkl" \
  2000 "compare_llama8_gsm8k" || true

# SVAMP
run_job "qwen15_svamp" "withmartian/SDS_train_svamp" "SDS_train_svamp" \
  "Qwen_1_5B_base/layer_20/all_sentences_features.pkl" \
  "Qwen_1_5B_reasoning/layer_20/all_sentences_features.pkl" \
  800 "compare_qwen15_svamp" || true

run_job "qwen14_svamp" "withmartian/SDS_train_svamp" "SDS_train_svamp" \
  "Qwen_14B_base/layer_28/all_sentences_features.pkl" \
  "Qwen_14B_reasoning/layer_28/all_sentences_features.pkl" \
  800 "compare_qwen14_svamp" || true

run_job "llama8_svamp" "withmartian/SDS_train_svamp" "SDS_train_svamp" \
  "Llama_8B_base/layer_22/all_sentences_features.pkl" \
  "Llama_8B_reasoning/layer_22/all_sentences_features.pkl" \
  800 "compare_llama8_svamp" || true

# MMLU-Pro
run_job "qwen15_mmlu" "withmartian/SDS_train_mmlu-pro" "SDS_train_mmlu-pro" \
  "qwen1.5b_base/layer_20/all_sentences_features.pkl" \
  "qwen1.5b/layer_20/all_sentences_features.pkl" \
  500 "compare_qwen15_mmlu-pro" || true

run_job "qwen14_mmlu" "withmartian/SDS_train_mmlu-pro" "SDS_train_mmlu-pro" \
  "qwen14b_base/layer_28/all_sentences_features.pkl" \
  "qwen14b/layer_28/all_sentences_features.pkl" \
  500 "compare_qwen14_mmlu-pro" || true

run_job "llama8_mmlu" "withmartian/SDS_train_mmlu-pro" "SDS_train_mmlu-pro" \
  "llama8b_base/layer_22/all_sentences_features.pkl" \
  "llama8b/layer_22/all_sentences_features.pkl" \
  500 "compare_llama8_mmlu-pro" || true

echo ""
echo "========== BATCH DONE =========="
