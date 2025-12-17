# Reasoning Policy Controllers in Fine-Tuned Models

## Research Question
How do fine-tuned reasoning models decide *when* to apply reasoning behaviors like verification and backtracking?
We investigate whether these decision-making mechanisms — **Reasoning Policy Controllers (RPCs)** — are learnable, isolable components that fine-tuning creates.

## Approach
We compare base and fine-tuned models to identify where they diverge, then analyze the internal states at these divergence points to understand the mechanisms driving reasoning behavior selection.

## Scope
- **Models**: DeepSeek-R1, Qwen, and related reasoning models
- **Datasets**: Mathematical reasoning (MATH, GSM8K), general knowledge (MMLU-Pro), visual reasoning (ARC-AGI), tool-use tasks
- **Methods**: Mechanistic interpretability techniques including crosscoders, attention analysis, activation patching, and feature decomposition

## Goals
1. Identify how fine-tuning modifies internal mechanisms for reasoning
2. Extract policy controllers and study their structure
3. Apply findings to advanced reasoning tasks

## Status
Active research. Details and findings will be shared as work progresses.
