# PrefixGuard

PrefixGuard checks a reasoning trace while it is still being generated. It
uses SDS/CEBRA states learned from labeled sentence activations, then scores new
traces according to the states they visit and the transitions between them.

## Contents

- `scripts/reasoning_sds_rerank.py` generates several answers for each problem
  and ranks them with the learned state scores. It also reports simpler
  baselines such as log-probability, answer majority, and length.
- `scripts/analyze_prefixes.py` reruns the scoring at different points in each
  saved trace and writes the tables and plots used in the analysis.
- `scripts/baselines.py` holds the SDS fitting, prompt formatting, and answer
  checking shared by both scripts.
- `results/` has the small summary files kept with this release. The large raw
  outputs and activation files are not included.

## Inputs

You need a Hugging Face causal language model and three data files:

- a pickle of sentence records with `problem_id`, `sentence_idx`, and
  `hidden_state_last`;
- a JSON file mapping problem IDs to correctness labels; and
- a JSON problem set with `problem`, `problem_id`, and either `answer` or
  `ground_truth`.

By default, evaluation problems are left out when the state scores are fitted.

## Run the verifier

Run commands from the repository root:

```bash
python -m prefixguard.scripts.reasoning_sds_rerank \
  --reasoning-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --reasoning-pkl /path/to/all_sentences_features.pkl \
  --rlvr-cache /path/to/correctness_cache.json \
  --hard-json /path/to/problems.json \
  --dataset math500 \
  --layer-idx 20 \
  --n-problems 60 \
  --k-samples 8 \
  --out outputs/prefixguard/rerank.json
```

## Analyze prefixes

If you only want the aggregate analysis, use `--skip-early-exit`. This does not
load the language model:

```bash
python -m prefixguard.scripts.analyze_prefixes \
  --result-json outputs/prefixguard/rerank.json \
  --out-dir outputs/prefixguard/analysis \
  --skip-early-exit
```

To score prefixes, leave out `--skip-early-exit` and pass the model, activation
pickle, correctness labels, and layer used for the original run. The default
checkpoints are 25%, 50%, 75%, and 100% of the trace. The script works from the
saved generations; it does not sample them again.
