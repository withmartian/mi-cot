# PrefixGuard: prefix-time reasoning monitoring

This directory contains the paper implementation for using SDS/CEBRA policy
states as a verifier over reasoning-model traces. It fits latent policy states
from labeled sentence activations, scores sampled reasoning traces by their
state trajectories, and tests how early those scores can distinguish useful
continuations.

## Contents

- `scripts/reasoning_sds_rerank.py` samples candidate traces and compares
  state-, CEBRA-, transition-, log-probability-, length-, majority-, and random
  selection rules.
- `scripts/analyze_prefixes.py` produces grouped robustness metrics,
  candidate AUCs, interpretable state summaries, and prefix-time evaluations at
  configurable generation fractions.
- `scripts/baselines.py` contains shared SDS fitting, answer checking, and
  prompt utilities.
- `results/` contains compact aggregate summaries. Raw generations, activation
  caches, checkpoints, and cluster logs are omitted.

## Inputs

The reranker expects a Hugging Face causal language model, a pickle of sentence
records containing `problem_id`, `sentence_idx`, and `hidden_state_last`, a JSON
mapping problem IDs to terminal correctness labels, and a JSON problem set with
`problem`, `problem_id`, and an `answer` or `ground_truth` field. Evaluation IDs
are excluded from the state-utility labels by default.

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

Analysis can run without loading a language model when prefix rescoring is
skipped:

```bash
python -m prefixguard.scripts.analyze_prefixes \
  --result-json outputs/prefixguard/rerank.json \
  --out-dir outputs/prefixguard/analysis \
  --skip-early-exit
```

For prefix-time scoring, omit `--skip-early-exit` and supply the same model,
activation pickle, correctness cache, and layer settings used for generation.
The defaults evaluate 25%, 50%, 75%, and 100% of each saved generation without
regenerating traces.

## Released result snapshot

On the representative 60-problem Qwen 1.5B MATH-500 run, the CEBRA verifier
reached candidate AUC 0.748, versus 0.593 for average log-probability. Selection
accuracy was 0.167 at 25% of the trace and 0.250 on the full trace, versus 0.133
and 0.217 for average log-probability. See
`results/qwen1_5b_math500_summary.md` for the full slice table and
`results/early_continue_policy_summary.json` for compact multi-seed results.

These are experimental selection results, not calibrated deployment-risk
guarantees.
