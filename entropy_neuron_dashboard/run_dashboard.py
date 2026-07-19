#!/usr/bin/env python3
"""
Entropy + confidence-neuron dashboard using the EXACT same dashboard code as
track_entropy/plot_uncertainty.py, with added: model toggle (Base | Reasoning),
per-neuron activation plot (one neuron at a time), and Compare base vs reasoning.

- Imports and uses: build_prompt, split_sentences, map_sentence_ends_to_token_indices,
  build_word_spans, map_tokens_to_words, build_token_values, is_wait_token,
  write_html_heatmap, plot_entropy_series, build_sentence_legend, plot_distribution.
- For each model (base, reasoning): finds entropy + token-freq neurons, runs forward
  on each example to get entropies, then builds the SAME trace/plot/legend as
  plot_uncertainty (Token/Word heatmap, entropy timeseries, sentence legend).
- Writes base_{idx}_uncertainty_trace.html, base_{idx}_uncertainty_timeseries.png, etc.
  and reasoning_{idx}_*; and base_aggregate_*.png, reasoning_aggregate_*.png.
- Dashboard HTML is the exact same structure as write_dashboard_html_static, with
  only: (1) Model toggle at top of sidebar, (2) data entries with base_* and
  reasoning_* fields, (3) Confidence neuron activation section, (4) Compare section.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

# PIL compatibility for older Pillow (transformers)
try:
    from PIL import Image
    if not hasattr(Image, "Resampling"):
        _R = Image
        _resample_values = (("NEAREST", 0), ("LANCZOS", 1), ("BILINEAR", 2), ("BICUBIC", 3), ("BOX", 4), ("HAMMING", 5))
        class _Resampling:
            pass
        for name, default in _resample_values:
            setattr(_Resampling, name, getattr(_R, name, default))
        Image.Resampling = _Resampling
except Exception:
    pass

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MI_COT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(MI_COT, "track_entropy"))
sys.path.insert(0, os.path.join(MI_COT, "entropy_steering"))

# Exact same code from track_entropy/plot_uncertainty
from plot_uncertainty import (
    load_results,
    build_prompt,
    split_sentences,
    map_sentence_ends_to_token_indices,
    build_word_spans,
    map_tokens_to_words,
    build_token_values,
    is_wait_token,
    write_html_heatmap,
    plot_entropy_series,
    build_sentence_legend,
    plot_distribution,
)

from entropy_neuron_utils import (
    get_layers,
    get_mlp_down_proj,
    get_unembedding_weight,
    locate_entropy_neurons_with_likelihood,
    locate_token_frequency_neurons,
)


def build_p_freq_from_gsm8k(tokenizer: AutoTokenizer, max_samples: int = 3000, device: str = "cpu") -> torch.Tensor:
    from collections import Counter
    dataset = load_dataset("gsm8k", "main", split="train")
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None and hasattr(tokenizer, "get_vocab"):
        vocab_size = len(tokenizer.get_vocab())
    if vocab_size is None:
        vocab_size = 32000
    counts = Counter()
    for ex in dataset:
        text = (ex.get("question") or "") + " " + (ex.get("answer") or "")
        ids = tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)
        counts.update(ids)
    p = torch.zeros(vocab_size, dtype=torch.float32, device=device) + 1e-9
    for idx, c in counts.items():
        if idx < vocab_size:
            p[idx] = c
    return p / p.sum()


def entropy_from_logits(logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return (-(probs * log_probs).sum(dim=-1)).cpu().numpy()


def forward_and_track_with_prompt_len(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    prompt_len: int,
    entropy_indices: List[int],
    tokfreq_indices: List[int],
    device: torch.device,
) -> Tuple[List[float], Dict[str, List[float]]]:
    layers = get_layers(model)
    down_proj = get_mlp_down_proj(layers[-1])
    captured = {}

    def hook(_m, inp, _out):
        captured["mlp_input"] = inp[0].detach()

    handle = down_proj.register_forward_hook(hook)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids.to(device))
    finally:
        handle.remove()

    logits = outputs.logits
    mlp_input = captured["mlp_input"]
    gen_len = input_ids.shape[1] - prompt_len
    if gen_len <= 0:
        return [], {}

    logits_gen = logits[0, prompt_len - 1 : prompt_len - 1 + gen_len]
    mlp_gen = mlp_input[0, prompt_len - 1 : prompt_len - 1 + gen_len]
    entropies = entropy_from_logits(logits_gen).flatten().tolist()
    all_indices = list(entropy_indices) + list(tokfreq_indices)
    names = [f"entropy_{i}" for i in range(len(entropy_indices))] + [f"token_freq_{i}" for i in range(len(tokfreq_indices))]
    neuron_activations = {}
    if all_indices:
        idx_t = torch.tensor(all_indices, device=mlp_gen.device, dtype=torch.long)
        acts = mlp_gen[:, idx_t].float().cpu().numpy()
        for i, name in enumerate(names):
            neuron_activations[name] = acts[:, i].tolist()
    return entropies, neuron_activations


def find_neurons_and_run_examples(
    model_name: str,
    p_freq: torch.Tensor,
    examples: List[Any],
    tokenizer: AutoTokenizer,
    device: torch.device,
    top_n_entropy: int,
    top_n_tokfreq: int,
    max_tokens: int,
) -> Tuple[List[int], List[int], Dict[int, Dict[str, Any]]]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model_vocab = get_unembedding_weight(model).shape[0]
    if p_freq.shape[0] != model_vocab:
        p_resized = torch.zeros(model_vocab, dtype=p_freq.dtype, device=p_freq.device) + 1e-9
        p_resized[: min(p_freq.shape[0], model_vocab)] = p_freq[: min(p_freq.shape[0], model_vocab)].to(p_resized.device)
        p_freq_use = p_resized / p_resized.sum()
    else:
        p_freq_use = p_freq
    p_freq_use = p_freq_use.to(device)

    ent_idx, _, _ = locate_entropy_neurons_with_likelihood(
        model, layer_idx=-1, top_n=top_n_entropy, min_norm_quantile=0.90, null_k_frac=0.01, device=str(device)
    )
    tf_idx, _, _ = locate_token_frequency_neurons(model, p_freq=p_freq_use, layer_idx=-1, top_n=top_n_tokfreq, device=str(device))
    entropy_indices = ent_idx.tolist()
    tokfreq_indices = tf_idx.tolist()

    per_example: Dict[int, Dict[str, Any]] = {}
    for idx, ex in enumerate(examples):
        prompt_text = build_prompt(ex.question)
        full_text = prompt_text + ex.generated
        inp_full = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        inp_prompt = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        input_ids = inp_full["input_ids"]
        prompt_len = inp_prompt["input_ids"].shape[1]
        prompt_len = min(prompt_len, input_ids.shape[1] - 1)
        gen_len = input_ids.shape[1] - prompt_len
        if gen_len <= 0:
            per_example[idx] = {"entropies": [], "neuron_activations": {}, "overall_entropy_avg": 0, "wait_prev_entropy_avg": 0, "overall_entropy_count": 0, "wait_prev_entropy_count": 0}
            continue
        entropies, neuron_activations = forward_and_track_with_prompt_len(
            model, tokenizer, input_ids, prompt_len, entropy_indices, tokfreq_indices, device
        )
        entropies = entropies[:max_tokens]
        for k in neuron_activations:
            neuron_activations[k] = neuron_activations[k][:max_tokens]
        per_example[idx] = {
            "entropies": entropies,
            "neuron_activations": neuron_activations,
            "overall_entropy_avg": float(np.mean(entropies)) if entropies else 0.0,
            "wait_prev_entropy_avg": 0.0,
            "overall_entropy_count": len(entropies),
            "wait_prev_entropy_count": 0,
        }
    del model
    torch.cuda.empty_cache()
    return entropy_indices, tokfreq_indices, per_example


def run_one_model_plots(
    model_key: str,
    examples: List[Any],
    tokenizer: AutoTokenizer,
    per_example_entropies: Dict[int, List[float]],
    output_dir: str,
    max_tokens: int,
) -> Tuple[Dict[int, Dict[str, float]], List[float], List[float], List[float], List[float], Dict[int, List[str]]]:
    """
    For each example, use the EXACT same pipeline as plot_uncertainty main():
    tokenize generated, offsets, split_sentences, map_sentence_ends_to_token_indices,
    build_word_spans, map_tokens_to_words, build_token_values, write_html_heatmap,
    plot_entropy_series. Compute wait_prev stats and distributions.
    Returns also token_strings_by_idx for neuron heatmap (token text per position).
    """
    per_example_stats: Dict[int, Dict[str, float]] = {}
    distribution_overall: List[float] = []
    distribution_wait_prev: List[float] = []
    distribution_overall_wait: List[float] = []
    distribution_wait_prev_wait: List[float] = []
    token_strings_by_idx: Dict[int, List[str]] = {}

    for idx, ex in enumerate(examples):
        entropies = per_example_entropies.get(idx, [])
        tokenized = tokenizer(ex.generated, add_special_tokens=False, return_offsets_mapping=True)
        offsets = tokenized["offset_mapping"]
        if len(entropies) != len(offsets):
            min_len = min(len(entropies), len(offsets))
            entropies = entropies[:min_len]
            offsets = offsets[:min_len]
        entropies = entropies[:max_tokens]
        offsets = offsets[:max_tokens]

        sentences = split_sentences(ex.generated)
        sentence_ends = [end for _, end in sentences]
        sentence_ticks = map_sentence_ends_to_token_indices(offsets, sentence_ends)
        sentence_labels = [text for text, _ in sentences[: len(sentence_ticks)]]
        wait_sentence_indices = [i for i, text in enumerate(sentence_labels) if text.lstrip().startswith("Wait")]

        word_spans = build_word_spans(ex.generated)
        word_values = map_tokens_to_words(word_spans, offsets, entropies)
        token_values = build_token_values(ex.generated, offsets, entropies)

        wait_prev_indices = set()
        for token_idx, (token_text, _) in enumerate(token_values):
            if token_idx > 0 and is_wait_token(token_text):
                wait_prev_indices.add(token_idx - 1)
        overall_sum = sum(e for i, e in enumerate(entropies) if i not in wait_prev_indices)
        overall_count = len(entropies) - len(wait_prev_indices)
        wait_prev_sum = sum(entropies[i] for i in wait_prev_indices)
        wait_prev_count = len(wait_prev_indices)
        overall_avg = overall_sum / overall_count if overall_count else 0.0
        wait_prev_avg = wait_prev_sum / wait_prev_count if wait_prev_count else 0.0

        distribution_overall.append(overall_avg)
        distribution_wait_prev.append(wait_prev_avg)
        if wait_prev_count:
            distribution_overall_wait.append(overall_avg)
            distribution_wait_prev_wait.append(wait_prev_avg)
        per_example_stats[idx] = {
            "overall_entropy_avg": overall_avg,
            "overall_entropy_count": overall_count,
            "wait_prev_entropy_avg": wait_prev_avg,
            "wait_prev_entropy_count": wait_prev_count,
        }

        html_path = os.path.join(output_dir, f"{model_key}_{idx}_uncertainty_trace.html")
        plot_path = os.path.join(output_dir, f"{model_key}_{idx}_uncertainty_timeseries.png")
        legend_path = os.path.join(output_dir, f"{model_key}_{idx}_uncertainty_timeseries_legend.txt")
        write_html_heatmap(word_values, token_values, html_path)
        plot_entropy_series(entropies, sentence_ticks, sentence_labels, plot_path, legend_path, wait_sentence_indices)
        token_strings_by_idx[idx] = [t for t, _ in token_values]

    return per_example_stats, distribution_overall, distribution_wait_prev, distribution_overall_wait, distribution_wait_prev_wait, token_strings_by_idx


def collect_dual_assets(output_dir: str, indices: List[int], model_key_base: str = "base", model_key_reasoning: str = "reasoning") -> List[Dict[str, Any]]:
    entries = []
    for idx in indices:
        item = {"index": str(idx)}
        for key in (model_key_base, model_key_reasoning):
            trace_name = f"{key}_{idx}_uncertainty_trace.html"
            plot_name = f"{key}_{idx}_uncertainty_timeseries.png"
            legend_name = f"{key}_{idx}_uncertainty_timeseries_legend.txt"
            if os.path.isfile(os.path.join(output_dir, trace_name)):
                item[f"{key}_trace"] = trace_name
            if os.path.isfile(os.path.join(output_dir, plot_name)):
                item[f"{key}_plot"] = plot_name
            if os.path.isfile(os.path.join(output_dir, legend_name)):
                try:
                    with open(os.path.join(output_dir, legend_name), "r", encoding="utf-8") as f:
                        item[f"{key}_legend_text"] = f.read()
                except Exception:
                    item[f"{key}_legend_text"] = ""
            else:
                item[f"{key}_legend_text"] = ""
        entries.append(item)
    return entries


def write_entropy_neuron_dashboard_static(
    output_dir: str,
    output_name: str,
    entries: List[Dict],
    base_stats: Dict[str, Any],
    reasoning_stats: Dict[str, Any],
    neuron_data: Dict[str, Any],
) -> None:
    """Exact same dashboard as write_dashboard_html_static, plus model toggle, neuron activation, compare."""
    payload = json.dumps(entries)
    mean_overall_base = base_stats.get("mean_overall", 0.0)
    mean_wait_base = base_stats.get("mean_wait_prev", 0.0)
    mean_overall_wait_base = base_stats.get("mean_overall_wait", 0.0)
    mean_wait_wait_base = base_stats.get("mean_wait_prev_wait", 0.0)
    mean_overall_reasoning = reasoning_stats.get("mean_overall", 0.0)
    mean_wait_reasoning = reasoning_stats.get("mean_wait_prev", 0.0)
    mean_overall_wait_reasoning = reasoning_stats.get("mean_overall_wait", 0.0)
    mean_wait_wait_reasoning = reasoning_stats.get("mean_wait_prev_wait", 0.0)
    base_neuron_opts = json.dumps(neuron_data.get("base_neuron_options", []))
    reasoning_neuron_opts = json.dumps(neuron_data.get("reasoning_neuron_options", []))

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
body {{
  font-family: sans-serif;
  margin: 0;
  padding: 0;
}}
.container {{
  display: grid;
  grid-template-columns: 240px 1fr;
  min-height: 100vh;
}}
.sidebar {{
  border-right: 1px solid #ddd;
  padding: 12px;
  overflow-y: auto;
}}
.sidebar button {{
  display: block;
  width: 100%;
  text-align: left;
  margin-bottom: 6px;
  padding: 6px 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fafafa;
  cursor: pointer;
}}
.sidebar .wait-only-example {{
  background: #fff8e6;
  border-color: #e6c200;
}}
.sidebar button.active {{
  background: #eef5ff;
  border-color: #aac7ff;
}}
.content {{
  padding: 16px;
}}
.section {{
  margin-bottom: 16px;
}}
.tabs button {{
  margin-right: 8px;
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fafafa;
  cursor: pointer;
}}
.tabs button.active {{
  background: #eef5ff;
  border-color: #aac7ff;
}}
.legend {{
  white-space: pre-wrap;
  background: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 10px;
}}
.plot {{
  max-width: 100%;
  border: 1px solid #eee;
}}
.trace {{
  width: 100%;
  height: 500px;
  border: 1px solid #ddd;
}}
.model-toggle-group {{
  border: 2px solid #888;
  border-radius: 6px;
  padding: 8px 10px;
  margin-bottom: 14px;
  background: #f9f9f9;
}}
.model-toggle-group .model-toggle-btn {{
  margin-bottom: 4px;
}}
.model-toggle-group .model-toggle-btn:last-child {{
  margin-bottom: 0;
}}
.neuron-heatmap {{
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
}}
.neuron-token {{
  display: inline-block;
  margin: 2px;
  padding: 2px 4px;
  border-radius: 3px;
  position: relative;
}}
.neuron-token .val {{
  font-size: 0.65rem;
  text-align: center;
  color: #111;
}}
.neuron-token .bar {{
  width: 4px;
  margin: 0 auto 2px;
  background: rgba(0,0,0,0.3);
  border-radius: 1px;
}}
</style>
</head>
<body>
<div class="container">
  <div class="sidebar">
    <div id="sidebar" class="section"></div>
  </div>
  <div class="content">
    <h2 id="title">Select an example</h2>
    <div id="tab-example-content">
      <div class="section">
        <h3>Prompt</h3>
        <div id="prompt" class="legend">No prompt available.</div>
      </div>
      <div class="section">
        <h3>Example stats</h3>
        <div class="legend">
          Average entropy (all tokens, excluding token before Wait): <span id="overall-avg">0.0000</span>
          (<span id="overall-count">0</span> tokens)
          <br/>
          Average entropy (token before Wait): <span id="wait-avg">0.0000</span>
          (<span id="wait-count">0</span> occurrences)
        </div>
      </div>
      <div class="section">
        <h3>Entropy Plot</h3>
        <img id="plot" class="plot" alt="Entropy plot"/>
      </div>
      <div class="section">
        <h3>Sentence legend (S# -> sentence)</h3>
        <div id="legend" class="legend">No legend available.</div>
      </div>
      <div class="section">
        <h3>Trace (HTML)</h3>
        <p class="legend">Token/word heatmap; Wait tokens are highlighted with an orange-yellow border.</p>
        <iframe id="trace" class="trace"></iframe>
      </div>
      <div class="section">
        <h3>Confidence neuron activation (one neuron)</h3>
        <p class="legend">Select a neuron to see its activation per token (color spectrum: red = high, blue = low).</p>
        <select id="neuron-select"><option value="">Select neuron</option></select>
        <div id="neuron-heatmap" class="neuron-heatmap"></div>
      </div>
    </div>
    <div id="tab-dist-content" style="display:none;">
      <div class="section">
        <h3>Aggregate distributions (per-example averages)</h3>
        <div class="legend" id="dist-legend">
          Mean overall entropy (excluding token before Wait): <span id="dist-mean-overall">0.0000</span>
          <br/>
          Mean entropy (token before Wait): <span id="dist-mean-wait">0.0000</span>
        </div>
      </div>
      <div class="section">
        <h4>Overall entropy averages</h4>
        <img id="dist-overall-img" class="plot" alt="Overall entropy distribution"/>
      </div>
      <div class="section">
        <h4>Token-before-Wait averages</h4>
        <img id="dist-wait-img" class="plot" alt="Wait entropy distribution"/>
      </div>
    </div>
    <div id="tab-dist-wait-content" style="display:none;">
      <div class="section">
        <h3>Aggregate (Wait-only examples)</h3>
        <div class="legend" id="dist-wait-legend">
          Mean overall entropy (excluding token before Wait): <span id="dist-mean-overall-wait">0.0000</span>
          <br/>
          Mean entropy (token before Wait): <span id="dist-mean-wait-wait">0.0000</span>
        </div>
      </div>
      <div class="section">
        <h4>Overall entropy averages (Wait-only)</h4>
        <img id="dist-overall-wait-img" class="plot" alt="Overall entropy distribution (Wait-only)"/>
      </div>
      <div class="section">
        <h4>Token-before-Wait averages (Wait-only)</h4>
        <img id="dist-wait-wait-img" class="plot" alt="Wait entropy distribution (Wait-only)"/>
      </div>
    </div>
    <div id="tab-compare-content" style="display:none;">
      <div class="section">
        <h3>Compare: Base vs Reasoning (average entropies only)</h3>
        <table class="legend" style="border-collapse: collapse; min-width: 320px;">
          <tr><th style="text-align:left; padding: 6px 12px;">Metric</th><th style="padding: 6px 12px;">Base</th><th style="padding: 6px 12px;">Reasoning</th></tr>
          <tr><td style="padding: 6px 12px;">Mean overall entropy (excl. token before Wait)</td><td style="padding: 6px 12px;" id="compare-base-overall">—</td><td style="padding: 6px 12px;" id="compare-reasoning-overall">—</td></tr>
          <tr><td style="padding: 6px 12px;">Mean entropy (token before Wait)</td><td style="padding: 6px 12px;" id="compare-base-wait">—</td><td style="padding: 6px 12px;" id="compare-reasoning-wait">—</td></tr>
          <tr><td style="padding: 6px 12px;">Mean overall (Wait-only examples)</td><td style="padding: 6px 12px;" id="compare-base-overall-wait">—</td><td style="padding: 6px 12px;" id="compare-reasoning-overall-wait">—</td></tr>
          <tr><td style="padding: 6px 12px;">Mean token-before-Wait (Wait-only)</td><td style="padding: 6px 12px;" id="compare-base-wait-wait">—</td><td style="padding: 6px 12px;" id="compare-reasoning-wait-wait">—</td></tr>
        </table>
      </div>
    </div>
  </div>
</div>
<script>
const data = {payload};
const neuronData = {json.dumps(neuron_data)};
const baseNeuronOptions = {base_neuron_opts};
const reasoningNeuronOptions = {reasoning_neuron_opts};
const meanBase = {{ overall: {mean_overall_base}, wait: {mean_wait_base}, overallWait: {mean_overall_wait_base}, waitWait: {mean_wait_wait_base} }};
const meanReasoning = {{ overall: {mean_overall_reasoning}, wait: {mean_wait_reasoning}, overallWait: {mean_overall_wait_reasoning}, waitWait: {mean_wait_wait_reasoning} }};
let currentModel = 'base';
const sidebar = document.getElementById('sidebar');
const title = document.getElementById('title');
const plot = document.getElementById('plot');
const legend = document.getElementById('legend');
const trace = document.getElementById('trace');
const distOverallImg = document.getElementById('dist-overall-img');
const distWaitImg = document.getElementById('dist-wait-img');
const distOverallWaitImg = document.getElementById('dist-overall-wait-img');
const distWaitWaitImg = document.getElementById('dist-wait-wait-img');

function setActive(btn) {{
  const buttons = sidebar.querySelectorAll('button');
  buttons.forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
}}

function showTab(kind) {{
  const exampleContent = document.getElementById('tab-example-content');
  const distContent = document.getElementById('tab-dist-content');
  const distWaitContent = document.getElementById('tab-dist-wait-content');
  const compareContent = document.getElementById('tab-compare-content');
  exampleContent.style.display = (kind === 'example') ? 'block' : 'none';
  distContent.style.display = (kind === 'dist') ? 'block' : 'none';
  distWaitContent.style.display = (kind === 'dist-wait') ? 'block' : 'none';
  compareContent.style.display = (kind === 'compare') ? 'block' : 'none';
  if (kind === 'dist' || kind === 'dist-wait') {{
    const m = currentModel === 'base' ? meanBase : meanReasoning;
    const prefix = currentModel + '_aggregate_';
    distOverallImg.src = prefix + 'overall_entropy.png';
    distWaitImg.src = prefix + 'wait_prev_entropy.png';
    distOverallWaitImg.src = prefix + 'wait_overall_entropy.png';
    distWaitWaitImg.src = prefix + 'wait_prev_entropy_wait.png';
    document.getElementById('dist-mean-overall').textContent = (kind === 'dist' ? m.overall : m.overallWait).toFixed(4);
    document.getElementById('dist-mean-wait').textContent = (kind === 'dist' ? m.wait : m.waitWait).toFixed(4);
    document.getElementById('dist-mean-overall-wait').textContent = m.overallWait.toFixed(4);
    document.getElementById('dist-mean-wait-wait').textContent = m.waitWait.toFixed(4);
  }}
  if (kind === 'compare') {{
    document.getElementById('compare-base-overall').textContent = meanBase.overall.toFixed(4);
    document.getElementById('compare-reasoning-overall').textContent = meanReasoning.overall.toFixed(4);
    document.getElementById('compare-base-wait').textContent = meanBase.wait.toFixed(4);
    document.getElementById('compare-reasoning-wait').textContent = meanReasoning.wait.toFixed(4);
    document.getElementById('compare-base-overall-wait').textContent = meanBase.overallWait.toFixed(4);
    document.getElementById('compare-reasoning-overall-wait').textContent = meanReasoning.overallWait.toFixed(4);
    document.getElementById('compare-base-wait-wait').textContent = meanBase.waitWait.toFixed(4);
    document.getElementById('compare-reasoning-wait-wait').textContent = meanReasoning.waitWait.toFixed(4);
  }}
}}

function colorForValue(norm) {{
  const r = Math.round(255 * norm);
  const g = Math.round(64 * (1 - norm));
  const b = Math.round(255 * (1 - norm));
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}}

function escapeHtml(s) {{
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}}

function renderNeuronHeatmap(tokens, values) {{
  const container = document.getElementById('neuron-heatmap');
  container.innerHTML = '';
  if (!tokens || !values || tokens.length === 0 || values.length === 0) return;
  const len = Math.min(tokens.length, values.length);
  const min = Math.min(...values.slice(0, len));
  const max = Math.max(...values.slice(0, len)) || 1;
  const range = max - min || 1;
  for (let i = 0; i < len; i++) {{
    const norm = range > 0 ? (values[i] - min) / range : 0;
    const color = colorForValue(norm);
    const barH = Math.max(2, 4 + norm * 24);
    const tokenText = tokens[i] === '' || tokens[i] === ' ' ? '\\u00a0' : tokens[i];
    const span = document.createElement('span');
    span.className = 'neuron-token';
    span.style.background = color;
    span.innerHTML = '<div class="bar" style="height:' + barH + 'px;"></div><div>' + escapeHtml(tokenText) + '</div><div class="val">' + Number(values[i]).toFixed(2) + '</div>';
    container.appendChild(span);
  }}
}}

function updateNeuronSelect() {{
  const sel = document.getElementById('neuron-select');
  const opts = currentModel === 'base' ? baseNeuronOptions : reasoningNeuronOptions;
  sel.innerHTML = '<option value="">Select neuron</option>' + opts.map(o => '<option value="'+o+'">'+o+'</option>').join('');
  sel.onchange = function() {{
    const key = currentModel + '_per_example';
    const tokenKey = currentModel + '_token_strings';
    const ex = neuronData[key] && neuronData[key][String(currentExampleIndex)];
    const tokens = (neuronData[tokenKey] && neuronData[tokenKey][String(currentExampleIndex)]) || [];
    const acts = (ex && ex.neuron_activations) || {{}};
    const val = sel.value;
    if (val && acts[val]) renderNeuronHeatmap(tokens, acts[val]);
    else document.getElementById('neuron-heatmap').innerHTML = '';
  }};
}}

var currentExampleIndex = 0;

function showExample(item, btn) {{
  if (btn) setActive(btn);
  currentExampleIndex = parseInt(item.index, 10);
  const prompt = document.getElementById('prompt');
  title.textContent = 'Example ' + item.index;
  const plotKey = currentModel + '_plot';
  const traceKey = currentModel + '_trace';
  const legendKey = currentModel + '_legend_text';
  const avgKey = currentModel + '_overall_entropy_avg';
  const countKey = currentModel + '_overall_entropy_count';
  const waitAvgKey = currentModel + '_wait_prev_entropy_avg';
  const waitCountKey = currentModel + '_wait_prev_entropy_count';
  plot.src = item[plotKey] || '';
  plot.style.display = item[plotKey] ? 'block' : 'none';
  legend.textContent = item[legendKey] || 'No legend available.';
  trace.src = item[traceKey] || '';
  trace.style.display = item[traceKey] ? 'block' : 'none';
  prompt.textContent = item.prompt || 'No prompt available.';
  document.getElementById('overall-avg').textContent = Number(item[avgKey] ?? 0).toFixed(4);
  document.getElementById('overall-count').textContent = item[countKey] ?? 0;
  document.getElementById('wait-avg').textContent = Number(item[waitAvgKey] ?? 0).toFixed(4);
  document.getElementById('wait-count').textContent = item[waitCountKey] ?? 0;
  updateNeuronSelect();
  const sel = document.getElementById('neuron-select');
  if (sel && sel.value) {{
    const key = currentModel + '_per_example';
    const tokenKey = currentModel + '_token_strings';
    const ex = neuronData[key] && neuronData[key][String(currentExampleIndex)];
    const tokens = (neuronData[tokenKey] && neuronData[tokenKey][String(currentExampleIndex)]) || [];
    const acts = (ex && ex.neuron_activations) || {{}};
    if (acts[sel.value]) renderNeuronHeatmap(tokens, acts[sel.value]);
  }}
}}

function setModel(model) {{
  currentModel = model;
  document.querySelectorAll('.model-toggle-btn').forEach(b => b.classList.toggle('active', b.dataset.model === model));
  const item = data.find(e => String(e.index) === String(currentExampleIndex));
  const exampleContent = document.getElementById('tab-example-content');
  const distContent = document.getElementById('tab-dist-content');
  const compareContent = document.getElementById('tab-compare-content');
  if (exampleContent.style.display !== 'none') {{
    if (item) showExample(item, null);
  }} else if (compareContent.style.display !== 'none') {{
    showTab('compare');
  }} else {{
    const kind = distContent.style.display !== 'none' ? 'dist' : 'dist-wait';
    showTab(kind);
  }}
}}

const modelToggleGroup = document.createElement('div');
modelToggleGroup.className = 'model-toggle-group';
const btnBase = document.createElement('button');
btnBase.className = 'model-toggle-btn';
btnBase.dataset.model = 'base';
btnBase.textContent = 'Base';
btnBase.onclick = () => setModel('base');
modelToggleGroup.appendChild(btnBase);
const btnReasoning = document.createElement('button');
btnReasoning.className = 'model-toggle-btn';
btnReasoning.dataset.model = 'reasoning';
btnReasoning.textContent = 'Reasoning';
btnReasoning.onclick = () => setModel('reasoning');
modelToggleGroup.appendChild(btnReasoning);
sidebar.appendChild(modelToggleGroup);

const aggregateButton = document.createElement('button');
aggregateButton.textContent = 'Aggregate';
aggregateButton.addEventListener('click', () => {{
  setActive(aggregateButton);
  showTab('dist');
}});
sidebar.appendChild(aggregateButton);
const aggregateWaitButton = document.createElement('button');
aggregateWaitButton.textContent = 'Aggregate (Wait-only)';
aggregateWaitButton.addEventListener('click', () => {{
  setActive(aggregateWaitButton);
  showTab('dist-wait');
}});
sidebar.appendChild(aggregateWaitButton);

const compareButton = document.createElement('button');
compareButton.textContent = 'Compare';
compareButton.addEventListener('click', () => {{
  setActive(compareButton);
  showTab('compare');
}});
sidebar.appendChild(compareButton);

data.forEach((item) => {{
  const btn = document.createElement('button');
  btn.textContent = 'Example ' + item.index;
  const hasWait = (Number(item.base_wait_prev_entropy_count) || 0) + (Number(item.reasoning_wait_prev_entropy_count) || 0) > 0;
  if (hasWait) btn.classList.add('wait-only-example');
  btn.addEventListener('click', () => {{
    showTab('example');
    showExample(item, btn);
  }});
  sidebar.appendChild(btn);
}});

if (data.length > 0) {{
  const firstExampleButton = sidebar.querySelectorAll('button')[4];
  if (firstExampleButton) {{
    showTab('example');
    setModel('base');
    setActive(firstExampleButton);
    showExample(data[0], firstExampleButton);
  }}
}}
document.getElementById('neuron-heatmap').innerHTML = '';
</script>
</body>
</html>"""
    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Entropy + confidence neuron dashboard (same as track_entropy dashboard + model toggle + neurons + compare)")
    parser.add_argument("--input", default="results.jsonl", help="JSONL from track_uncertainty")
    parser.add_argument("--output-dir", default="entropy_neuron_dashboard_out", help="Output directory")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--reasoning_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--top_n_entropy", type=int, default=16)
    parser.add_argument("--top_n_tokfreq", type=int, default=16)
    parser.add_argument("--gsm8k_max_samples", type=int, default=2000)
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--no_dashboard", action="store_true")
    args = parser.parse_args()

    summary, examples = load_results(args.input)
    if not examples:
        raise SystemExit("No examples in JSONL")
    examples = examples[: args.max_examples]
    indices = list(range(len(examples)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_base = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer_base.pad_token is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token
    print("Building p_freq from GSM8K...")
    p_freq = build_p_freq_from_gsm8k(tokenizer_base, max_samples=args.gsm8k_max_samples)

    print("Base model: find neurons and run forward...")
    base_ent, base_tf, base_per_example = find_neurons_and_run_examples(
        args.base_model, p_freq, examples, tokenizer_base, device,
        args.top_n_entropy, args.top_n_tokfreq, args.max_tokens,
    )
    base_entropies = {idx: base_per_example[idx].get("entropies", []) for idx in indices}
    print("Base model: write trace/plot/legend (same as plot_uncertainty)...")
    base_stats_dict, base_dist_overall, base_dist_wait, base_dist_overall_wait, base_dist_wait_wait, base_token_strings = run_one_model_plots(
        "base", examples, tokenizer_base, base_entropies, args.output_dir, args.max_tokens,
    )
    base_per_example_prompts = {idx: build_prompt(ex.question) for idx, ex in enumerate(examples)}
    mean_overall_base = sum(base_dist_overall) / len(base_dist_overall) if base_dist_overall else 0.0
    mean_wait_base = sum(base_dist_wait) / len(base_dist_wait) if base_dist_wait else 0.0
    mean_overall_wait_base = sum(base_dist_overall_wait) / len(base_dist_overall_wait) if base_dist_overall_wait else 0.0
    mean_wait_wait_base = sum(base_dist_wait_wait) / len(base_dist_wait_wait) if base_dist_wait_wait else 0.0
    plot_distribution(base_dist_overall, "Overall entropy (excluding token before Wait)", os.path.join(args.output_dir, "base_aggregate_overall_entropy.png"))
    plot_distribution(base_dist_wait, "Entropy (token before Wait)", os.path.join(args.output_dir, "base_aggregate_wait_prev_entropy.png"))
    plot_distribution(base_dist_overall_wait, "Overall entropy (Wait-only)", os.path.join(args.output_dir, "base_aggregate_wait_overall_entropy.png"))
    plot_distribution(base_dist_wait_wait, "Entropy (token before Wait, Wait-only)", os.path.join(args.output_dir, "base_aggregate_wait_prev_entropy_wait.png"))

    print("Reasoning model: find neurons and run forward...")
    tokenizer_r = AutoTokenizer.from_pretrained(args.reasoning_model, use_fast=True, trust_remote_code=True)
    if getattr(tokenizer_r, "vocab_size", 0) != p_freq.shape[0]:
        p_freq_r = build_p_freq_from_gsm8k(tokenizer_r, max_samples=args.gsm8k_max_samples)
    else:
        p_freq_r = p_freq
    reasoning_ent, reasoning_tf, reasoning_per_example = find_neurons_and_run_examples(
        args.reasoning_model, p_freq_r, examples, tokenizer_r, device,
        args.top_n_entropy, args.top_n_tokfreq, args.max_tokens,
    )
    reasoning_entropies = {idx: reasoning_per_example[idx].get("entropies", []) for idx in indices}
    print("Reasoning model: write trace/plot/legend (same as plot_uncertainty)...")
    reasoning_stats_dict, reasoning_dist_overall, reasoning_dist_wait, reasoning_dist_overall_wait, reasoning_dist_wait_wait, reasoning_token_strings = run_one_model_plots(
        "reasoning", examples, tokenizer_r, reasoning_entropies, args.output_dir, args.max_tokens,
    )
    mean_overall_r = sum(reasoning_dist_overall) / len(reasoning_dist_overall) if reasoning_dist_overall else 0.0
    mean_wait_r = sum(reasoning_dist_wait) / len(reasoning_dist_wait) if reasoning_dist_wait else 0.0
    mean_overall_wait_r = sum(reasoning_dist_overall_wait) / len(reasoning_dist_overall_wait) if reasoning_dist_overall_wait else 0.0
    mean_wait_wait_r = sum(reasoning_dist_wait_wait) / len(reasoning_dist_wait_wait) if reasoning_dist_wait_wait else 0.0
    plot_distribution(reasoning_dist_overall, "Overall entropy (excluding token before Wait)", os.path.join(args.output_dir, "reasoning_aggregate_overall_entropy.png"))
    plot_distribution(reasoning_dist_wait, "Entropy (token before Wait)", os.path.join(args.output_dir, "reasoning_aggregate_wait_prev_entropy.png"))
    plot_distribution(reasoning_dist_overall_wait, "Overall entropy (Wait-only)", os.path.join(args.output_dir, "reasoning_aggregate_wait_overall_entropy.png"))
    plot_distribution(reasoning_dist_wait_wait, "Entropy (token before Wait, Wait-only)", os.path.join(args.output_dir, "reasoning_aggregate_wait_prev_entropy_wait.png"))

    os.makedirs(args.output_dir, exist_ok=True)
    entries = collect_dual_assets(args.output_dir, indices)
    for e in entries:
        idx = int(e["index"])
        e["prompt"] = build_prompt(examples[idx].question)
        e["base_overall_entropy_avg"] = base_stats_dict.get(idx, {}).get("overall_entropy_avg", 0)
        e["base_overall_entropy_count"] = base_stats_dict.get(idx, {}).get("overall_entropy_count", 0)
        e["base_wait_prev_entropy_avg"] = base_stats_dict.get(idx, {}).get("wait_prev_entropy_avg", 0)
        e["base_wait_prev_entropy_count"] = base_stats_dict.get(idx, {}).get("wait_prev_entropy_count", 0)
        e["reasoning_overall_entropy_avg"] = reasoning_stats_dict.get(idx, {}).get("overall_entropy_avg", 0)
        e["reasoning_overall_entropy_count"] = reasoning_stats_dict.get(idx, {}).get("overall_entropy_count", 0)
        e["reasoning_wait_prev_entropy_avg"] = reasoning_stats_dict.get(idx, {}).get("wait_prev_entropy_avg", 0)
        e["reasoning_wait_prev_entropy_count"] = reasoning_stats_dict.get(idx, {}).get("wait_prev_entropy_count", 0)

    neuron_data = {
        "base_neuron_options": [f"entropy_{i}" for i in range(len(base_ent))] + [f"token_freq_{i}" for i in range(len(base_tf))],
        "reasoning_neuron_options": [f"entropy_{i}" for i in range(len(reasoning_ent))] + [f"token_freq_{i}" for i in range(len(reasoning_tf))],
        "base_per_example": {str(idx): {"entropies": base_per_example.get(idx, {}).get("entropies", []), "neuron_activations": base_per_example.get(idx, {}).get("neuron_activations", {})} for idx in indices},
        "reasoning_per_example": {str(idx): {"entropies": reasoning_per_example.get(idx, {}).get("entropies", []), "neuron_activations": reasoning_per_example.get(idx, {}).get("neuron_activations", {})} for idx in indices},
        "base_token_strings": {str(idx): base_token_strings.get(idx, []) for idx in indices},
        "reasoning_token_strings": {str(idx): reasoning_token_strings.get(idx, []) for idx in indices},
    }

    if not args.no_dashboard:
        write_entropy_neuron_dashboard_static(
            args.output_dir,
            "dashboard_static_entropy_neurons.html",
            entries,
            {"mean_overall": mean_overall_base, "mean_wait_prev": mean_wait_base, "mean_overall_wait": mean_overall_wait_base, "mean_wait_prev_wait": mean_wait_wait_base},
            {"mean_overall": mean_overall_r, "mean_wait_prev": mean_wait_r, "mean_overall_wait": mean_overall_wait_r, "mean_wait_prev_wait": mean_wait_wait_r},
            neuron_data,
        )
        print("Wrote dashboard to", os.path.join(args.output_dir, "dashboard_static_entropy_neurons.html"))
    print("Done. Output dir:", args.output_dir)


if __name__ == "__main__":
    main()
