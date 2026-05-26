#!/usr/bin/env bash
# Run remaining model×dataset combos (middle layers); cleanup large pkls after each run.
set -euo pipefail

MI_COT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CORRECT_ONLY="$(cd "$(dirname "$0")" && pwd)"
cd "$MI_COT_ROOT"

RUN_NAME="${RUN_NAME:-middle_layer_sweep}"
HF_CACHE="${HF_CACHE:-$MI_COT_ROOT/hf_cache}"
RESULTS="$CORRECT_ONLY/results/compare/$RUN_NAME"

# dataset model_family layer
COMBOS=(
  "math500 qwen14 28"
  "math500 llama8 22"
  "gsm8k qwen14 28"
  "gsm8k qwen1.5 20"
  "gsm8k llama8 22"
  "svamp qwen14 28"
  "svamp qwen1.5 20"
  "svamp llama8 22"
  "mmlu-pro qwen14 28"
  "mmlu-pro qwen1.5 20"
  "mmlu-pro llama8 22"
)

cleanup_combo() {
  local dataset="$1"
  local family="$2"
  local layer="$3"
  local run_subdir="$RESULTS/${dataset}_${family}"

  # CEBRA caches (reproducible from summaries)
  find "$run_subdir" -name 'cebra_cache.pkl' -delete 2>/dev/null || true

  # HF activation pickles for this combo's folders
  local -a folders=()
  case "$dataset" in
    math500|svamp)
      case "$family" in
        qwen14) folders=(Qwen_14B_base Qwen_14B_reasoning) ;;
        qwen1.5) folders=(Qwen_1_5B_base Qwen_1_5B_reasoning) ;;
        llama8) folders=(Llama_8B_base Llama_8B_reasoning) ;;
      esac
      ;;
    gsm8k)
      case "$family" in
        qwen14) folders=(qwen_14B_base qwen_14b_reasoning) ;;
        qwen1.5) folders=(qwen_1.5B_base qwen1.5b_reasoning) ;;
        llama8) folders=(llama_8B_base llama_8b_reasoning) ;;
      esac
      ;;
    mmlu-pro)
      case "$family" in
        qwen14) folders=(qwen14b_base qwen14b) ;;
        qwen1.5) folders=(qwen1.5b_base qwen1.5b) ;;
        llama8) folders=(llama8b_base llama8b) ;;
      esac
      ;;
  esac

  for folder in "${folders[@]}"; do
    rm -rf "$HF_CACHE/$folder/layer_${layer}" 2>/dev/null || true
    rm -rf "$HF_CACHE/withmartian__"*/"$folder"/"layer_${layer}" 2>/dev/null || true
  done

  # HuggingFace hub blob cache (safe to drop between combos)
  rm -rf "$HF_CACHE/.cache" 2>/dev/null || true

  echo "  [cleanup] freed pkls for ${dataset}/${family} layer_${layer}"
}

for spec in "${COMBOS[@]}"; do
  read -r dataset family layer <<< "$spec"
  echo ""
  echo "========== ${dataset} / ${family} (layer ${layer}) =========="
  if [[ -f "$RESULTS/${dataset}_${family}/comparison_report.json" ]]; then
    echo "  Skipping (comparison_report.json exists)"
    cleanup_combo "$dataset" "$family" "$layer"
    continue
  fi

  python3 ablate_cot_trace_expms/correct_only/run_compare_base_rft.py \
    --dataset "$dataset" \
    --model-family "$family" \
    --layer "$layer" \
    --hf-cache-dir "$HF_CACHE" \
    --k-values 5 \
    --k-focus 5 \
    --run-name "$RUN_NAME" \
    || true

  if [[ ! -f "$RESULTS/${dataset}_${family}/comparison_report.json" ]]; then
    echo "ERROR: no comparison_report.json for ${dataset} ${family}"
    exit 1
  fi

  cleanup_combo "$dataset" "$family" "$layer"
done

echo ""
echo "========== SWEEP DONE =========="
