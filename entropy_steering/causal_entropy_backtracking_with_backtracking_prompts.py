"""
causal_entropy_backtracking.py

Causal tests: do entropy-related components cause backtracking markers (Wait/Hmm)?

Two intervention families:
(A) Layer-10 entropy direction (from repo-style regression in entropy_neurons.py).
(B) Final-layer "entropy neurons" (Confidence Regulation Neurons): select neurons
    with high ρ (projection of neuron output weights onto unembedding nullspace),
    then mean-ablate or amplify their activations.

Outputs a CSV with per-prompt backtracking metrics under each condition.

Dependencies:
  pip install nnsight transformers torch tqdm pandas numpy

Notes:
- Model paths are Llama-like. If module names differ for your checkpoint, adjust
  get_layers() and get_mlp_down_proj().
"""

from __future__ import annotations

import argparse
import os
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import json

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from nnsight import LanguageModel


# -----------------------------
# Utilities: model/module access
# -----------------------------
def get_layers(model: LanguageModel):
    # DeepSeek-R1-Distill-Llama-8B is Llama-like in HF: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Some other arch: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Could not find transformer layers on this model.")


def get_unembedding_weight(model: LanguageModel) -> torch.Tensor:
    # Usually lm_head.weight: [vocab, d_model]
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    raise AttributeError("Could not find lm_head.weight (unembedding) on this model.")


def get_mlp_down_proj(layer) -> torch.nn.Module:
    # HF Llama: layer.mlp.down_proj
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    raise AttributeError("Could not find layer.mlp.down_proj. Adjust for your architecture.")


# -----------------------------
# Prompts + scoring
# -----------------------------
BACKTRACK_PAT = re.compile(r"\b(wait|hmm)\b", flags=re.IGNORECASE)


def extract_think_block(decoded: str) -> str:
    # DeepSeek R1-style chat often includes <think> ... </think>
    # If absent, just use full text.
    if "</think>" in decoded:
        return decoded.split("</think>")[0]
    return decoded


def backtracking_rate(text: str) -> float:
    """
    Simple proxy used in the repo/paper: fraction of words that are backtracking markers.
    Ward et al. use keyword-based metrics and validate them as a proxy.
    """
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in ("wait", "hmm"))
    return hits / len(words)


def backtracking_count(text: str) -> int:
    return len(BACKTRACK_PAT.findall(text))




# -----------------------------
# Prompt loading (latent-backtracking style) + prefiltering
# -----------------------------
def _extract_prompt_from_record(obj) -> Optional[str]:
    """Best-effort extraction of a single 'prompt' string from a JSON record."""
    if obj is None:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        return s or None
    if isinstance(obj, dict):
        # Common keys across task datasets
        for k in ("prompt", "problem", "question", "input", "instruction", "query", "task", "text"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Chat-style format: {"messages":[{"role":"user","content":...}, ...]}
        msgs = obj.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                    s = m["content"].strip()
                    if s:
                        return s
    if isinstance(obj, list):
        # A list might be a list of messages, or a singleton wrapper
        if len(obj) == 1:
            return _extract_prompt_from_record(obj[0])
        # If it looks like messages, grab the first user message
        if all(isinstance(x, dict) for x in obj):
            for m in obj:
                if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip():
                    return m["content"].strip()
    return None


def load_prompts_from_path(path: str) -> List[str]:
    """Load prompts from a .json, .jsonl, .txt, or a directory containing those."""
    if os.path.isdir(path):
        prompts: List[str] = []
        for root, _, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith((".json", ".jsonl", ".txt")):
                    prompts.extend(load_prompts_from_path(os.path.join(root, fn)))
        # De-dup while preserving order
        seen = set()
        out = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    # JSON / JSONL
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    if path.lower().endswith(".jsonl"):
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            p = _extract_prompt_from_record(obj)
            if p:
                prompts.append(p)
        return prompts

    # .json
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    if isinstance(obj, list):
        for item in obj:
            p = _extract_prompt_from_record(item)
            if p:
                prompts.append(p)
    else:
        p = _extract_prompt_from_record(obj)
        if p:
            prompts.append(p)
    return prompts


def load_latent_backtracking_prompts(repo_dir: str) -> List[str]:
    """Load prompt strings from a local clone of jnward/latent-backtracking.

    The repo's tasks live under tasks/ and new_tasks/. This loader just walks those
    directories and extracts a prompt/problem/question field from JSON/JSONL.
    """
    candidates: List[str] = []
    for sub in ("tasks", "new_tasks"):
        d = os.path.join(repo_dir, sub)
        if os.path.isdir(d):
            candidates.extend(load_prompts_from_path(d))
    # De-dup while preserving order
    seen = set()
    out = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


@torch.no_grad()
def prefilter_prompts_that_backtrack(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    *,
    min_backtrack_count: int = 1,
    max_candidates: int = 200,
    keep_n: int = 50,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 0,
    device: str = "cuda",
) -> Tuple[List[str], Dict[str, str]]:
    """Run a quick baseline pass and keep prompts that actually elicit 'wait'/'hmm'.

    Returns (selected_prompts, baseline_cache). The cache maps prompt->decoded output for
    prompts we evaluated during prefiltering.
    """
    selected: List[str] = []
    cache: Dict[str, str] = {}
    for p in tqdm(prompts[:max_candidates], desc="Prefilter: finding prompts that backtrack"):
        out = generate_with_optional_interventions(
            model, tokenizer, p,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            device=device,
        )
        cache[p] = out
        think = extract_think_block(out)
        if backtracking_count(think) >= min_backtrack_count:
            selected.append(p)
            if len(selected) >= keep_n:
                break
    return selected, cache


# -----------------------------
# Annotated chain loading (entropy_neurons.py style)
# -----------------------------
def load_annotated_chains(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def format_annotated_chain(chain: dict, tokenizer: AutoTokenizer) -> str:
    # Matches entropy_neurons.py:
    # messages = [{"role":"user","content": problem}, {"role":"assistant","content": reasoning_chain}]
    messages = [
        {"role": "user", "content": chain["problem"]},
        {"role": "assistant", "content": chain["reasoning_chain"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def load_formatted_annotated_chains(
    path: str,
    tokenizer: AutoTokenizer,
    max_chains: int = 100,
    seed: int = 0,
    shuffle: bool = True,
) -> List[str]:
    chains = load_annotated_chains(path)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(chains)
    chains = chains[:max_chains]
    return [format_annotated_chain(c, tokenizer) for c in chains]


# -----------------------------
# Direction loading / computing
# -----------------------------
@torch.no_grad()
def maybe_load_direction(path: str, device: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if os.path.exists(path):
        v = torch.load(path, map_location="cpu")
        v = v.to(device=device, dtype=dtype)
        v = v / (v.norm() + 1e-12)
        return v
    return None


@torch.no_grad()
def compute_layer_entropy_direction_from_annotated_chains(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    annotated_chains_path: str,
    n_chains: int = 100,
    seed: int = 0,
    shuffle: bool = True,
    layer_idx: int = 10,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute the entropy direction the same way as entropy_neurons.py:
    - Load annotated reasoning chains from JSON (problem + reasoning_chain)
    - Format as a chat with both user and assistant turns
    - Trace the full text to collect layer activations and logits
    - Compute token entropy from logits
    - Fit least-squares direction from activations -> entropy
    """
    layers = get_layers(model)

    formatted_chains = load_formatted_annotated_chains(
        annotated_chains_path,
        tokenizer,
        max_chains=n_chains,
        seed=seed,
        shuffle=shuffle,
    )

    acts_all: List[torch.Tensor] = []
    entropy_all: List[torch.Tensor] = []

    for text in tqdm(formatted_chains, desc="Collecting acts/entropy from annotated chains"):
        with model.trace(text) as tracer:
            a = layers[layer_idx].output[0].save()  # [1, seq, d_model]
            logits = model.output[0].save()         # [1, seq, vocab]

        probs = torch.softmax(logits.float(), dim=-1)
        ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # [1, seq]

        acts_all.append(a.squeeze(0).cpu())       # [seq, d_model]
        entropy_all.append(ent.squeeze(0).cpu())  # [seq]

    X = torch.cat(acts_all, dim=0).float()        # [N, d_model]
    y = torch.cat(entropy_all, dim=0).float()     # [N]

    X = X - X.mean(dim=0, keepdim=True)
    y = y - y.mean()

    w = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
    w = w / (w.norm() + 1e-12)
    return w.to(device=device, dtype=torch.bfloat16)


@torch.no_grad()
def compute_layer10_entropy_direction(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer_idx: int = 10,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Reproduces the idea in entropy_neurons.py: fit a least-squares direction from layer activations -> entropy.
    The repo does this on annotated reasoning chains and uses layer 10 output. 
    """
    layers = get_layers(model)

    acts_all = []
    entropy_all = []

    for p in tqdm(prompts, desc="Collecting acts/entropy for regression"):
        msgs = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)

        # Generate one trace to get a realistic reasoning-style continuation
        with model.generate(input_ids, max_new_tokens=max_new_tokens) as gen:
            gen_out = model.generator.output.save()
        decoded = tokenizer.decode(gen_out[0])

        # Forward pass on the produced text to get logits + layer activations aligned per token
        # (This is cheaper and simpler than instrumenting per-step logits inside generation.)
        with model.trace(decoded) as tracer:
            a = layers[layer_idx].output[0].save()   # [1, seq, d_model] or similar
            logits = model.output[0].save()          # [1, seq, vocab]

        probs = torch.softmax(logits.float(), dim=-1)
        ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # [1, seq]
        acts_all.append(a.squeeze(0).cpu())
        entropy_all.append(ent.squeeze(0).cpu())

    X = torch.cat(acts_all, dim=0).float()     # [N, d_model]
    y = torch.cat(entropy_all, dim=0).float()  # [N]

    X = X - X.mean(dim=0, keepdim=True)
    y = y - y.mean()

    # Least squares direction: argmin ||X w - y||^2
    w = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)
    w = w / (w.norm() + 1e-12)
    return w.to(device=device, dtype=torch.bfloat16)


# -----------------------------
# Final-layer entropy-neuron selection (ρ metric)
# -----------------------------
@torch.no_grad()
def compute_gram_unembed(WU: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """
    Compute G = WU^T WU in chunks to avoid huge memory spikes.
    WU: [vocab, d_model]
    Returns: [d_model, d_model]
    """
    vocab, d_model = WU.shape
    G = torch.zeros((d_model, d_model), device=WU.device, dtype=torch.float32)
    for i in tqdm(range(0, vocab, chunk), desc="Computing WU^T WU (chunked)"):
        Wc = WU[i : i + chunk].float()  # [chunk, d_model]
        G += Wc.T @ Wc
    return G


@torch.no_grad()
def get_unembed_nullspace_basis(
    model: LanguageModel,
    k: int = 256,
    device: str = "cuda",
    chunk: int = 8192,
) -> torch.Tensor:
    """
    Effective null space basis V0 from bottom-k eigenvectors of WU^T WU.
    Confidence Regulation Neurons motivates an "effective null space" in WU. 
    """
    # Retrieve the unembedding weight. It may be a meta tensor if the model
    # hasn't been materialized yet. Attempt to copy it to the target device,
    # falling back to a warm-up trace if necessary.
    WU = get_unembedding_weight(model)
    try:
        WU = WU.to(device=device)
    except NotImplementedError:
        # materialize meta weights by running a dummy forward pass
        with model.trace("Hello!") as tracer:
            _ = model.output.save()
        WU = get_unembedding_weight(model).to(device=device)
    G = compute_gram_unembed(WU, chunk=chunk)  # [d_model, d_model]
    # eigh gives ascending eigenvalues; bottom-k eigenvectors approximate null directions
    evals, evecs = torch.linalg.eigh(G)        # evecs: [d_model, d_model]
    V0 = evecs[:, :k].contiguous()             # [d_model, k]
    return V0


@torch.no_grad()
def select_entropy_neurons_by_rho(
    model: LanguageModel,
    null_basis_V0: torch.Tensor,   # [d_model, k]
    layer_idx: int = -1,
    top_n: int = 64,
    min_norm_quantile: float = 0.90,
    device: str = "cuda",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute rho for each MLP neuron (column of down_proj.weight) in a layer:
      rho_j = || V0^T w_out_j || / || w_out_j ||
    Then select high rho among high-norm neurons.

    This follows the paper's identification idea. 
    """
    layers = get_layers(model)
    layer = layers[layer_idx]
    down_proj = get_mlp_down_proj(layer)

    W_out = down_proj.weight.to(device=device).float()  # [d_model, d_ff] in HF Llama
    # Projection into null space
    proj = null_basis_V0.T.float() @ W_out              # [k, d_ff]
    proj_norm = proj.norm(dim=0)                        # [d_ff]
    out_norm = W_out.norm(dim=0) + 1e-12                # [d_ff]
    rho = proj_norm / out_norm                          # [d_ff]

    # Restrict to high-norm neurons first
    thresh = torch.quantile(out_norm, torch.tensor(min_norm_quantile, device=device))
    mask = out_norm >= thresh

    # Score = rho among the masked set
    rho_masked = rho.clone()
    rho_masked[~mask] = -1.0

    top_idx = torch.topk(rho_masked, k=top_n).indices

    stats = {
        "rho": rho.detach().cpu(),
        "out_norm": out_norm.detach().cpu(),
        "selected_rho": rho[top_idx].detach().cpu(),
        "selected_out_norm": out_norm[top_idx].detach().cpu(),
    }
    return top_idx.detach().cpu(), stats


# -----------------------------
# Interventions + generation
# -----------------------------
@dataclass
class Condition:
    name: str


@torch.no_grad()
def generate_with_optional_interventions(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    layer10_add: Optional[Tuple[int, torch.Tensor, float]] = None,  # (layer_idx, direction, alpha)
    final_neuron_ablate: Optional[Tuple[int, torch.Tensor, torch.Tensor]] = None,  # (layer_idx, neuron_idx, mean_vals)
    final_neuron_scale: Optional[Tuple[int, torch.Tensor, float]] = None,  # (layer_idx, neuron_idx, scale)
    device: str = "cuda",
) -> str:
    """
    Generate a response to the given prompt while optionally applying interventions.

    Interventions include adding a direction in a specific layer (layer10_add) and
    modifying selected neurons in the final layer via mean ablation or scaling.

    To avoid NNsight OutOfOrder errors, this function saves the generation output
    before applying any interventions. It also skips layer-10 interventions when
    the alpha value is effectively zero and wraps neuron modifications in try/except
    blocks to bypass steps that cannot be applied in early tokens.
    """
    torch.manual_seed(seed)

    # Format the prompt as chat with generation prompt; fall back to raw prompt on failure
    msgs = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        formatted = prompt

    input_ids = tokenizer.encode(
        formatted,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    layers = get_layers(model)
    out = None  # ensure defined even if generation fails

    with model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature) as tracer:
        # Save generation output immediately; ensures `out` is defined even if interventions error
        out = model.generator.output.save()

        # Apply layer-10 direction if provided and alpha is non-zero
        if layer10_add is not None:
            lidx, v, alpha = layer10_add
            if abs(alpha) > 1e-12:
                # Ensure the direction is on the correct device
                v = v.to(device=device)
                # Broadcast to [batch, seq, d_model]
                delta = (alpha * v).view(1, 1, -1)
                try:
                    with tracer.all():
                        layers[lidx].output[0] = layers[lidx].output[0] + delta
                except Exception:
                    # Skip if layer modification fails on early tokens
                    pass

        # Apply mean ablation to selected final-layer neurons
        if final_neuron_ablate is not None:
            lidx, neuron_idx_cpu, mean_vals_cpu = final_neuron_ablate
            # Move indices and means to target device
            neuron_idx = neuron_idx_cpu.to(device)
            mean_vals = mean_vals_cpu.to(device)
            try:
                down_proj = get_mlp_down_proj(layers[lidx])
                with tracer.all():
                    x = down_proj.input[0]
                    # set selected neurons at current token position
                    if x.ndim == 3:
                        x[:, -1, neuron_idx] = mean_vals[neuron_idx]
                    elif x.ndim == 2:
                        x[-1, neuron_idx] = mean_vals[neuron_idx]
                    else:
                        raise RuntimeError(f"Unexpected down_proj.input shape: {tuple(x.shape)}")
            except Exception:
                # Some positions may not have the input available; skip silently
                pass

        # Apply scaling to selected final-layer neurons
        if final_neuron_scale is not None:
            lidx, neuron_idx_cpu, scale = final_neuron_scale
            neuron_idx = neuron_idx_cpu.to(device)
            try:
                down_proj = get_mlp_down_proj(layers[lidx])
                with tracer.all():
                    x = down_proj.input[0]
                    if x.ndim == 3:
                        x[:, -1, neuron_idx] = x[:, -1, neuron_idx] * scale
                    elif x.ndim == 2:
                        x[-1, neuron_idx] = x[-1, neuron_idx] * scale
                    else:
                        raise RuntimeError(f"Unexpected down_proj.input shape: {tuple(x.shape)}")
            except Exception:
                pass

    # After generation completes, ensure `out` is defined
    if out is None:
        raise RuntimeError("Generation finished but no output was captured (out is None).")
    return tokenizer.decode(out[0])


@torch.no_grad()
def estimate_mlp_mean_activations(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer_idx: int = -1,
    max_new_tokens: int = 128,
    device: str = "cuda",
    max_samples: int = 32,
) -> torch.Tensor:
    """
    Estimate mean activation of final-layer MLP hidden units (down_proj.input) across some samples.
    This is the reference mean needed for mean ablation.
    """
    prompts = prompts[:max_samples]
    layers = get_layers(model)
    down_proj = get_mlp_down_proj(layers[layer_idx])

    sum_vec = None
    count = 0

    for p in tqdm(prompts, desc="Estimating mean MLP hidden activations"):
        msgs = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer.encode(formatted, add_special_tokens=False, return_tensors="pt").to(device)

        with model.generate(input_ids, max_new_tokens=max_new_tokens) as gen:
            out = model.generator.output.save()
        decoded = tokenizer.decode(out[0])

        with model.trace(decoded) as tracer:
            x = down_proj.input[0].save()  # [1, seq, d_ff] in many cases

        x = x.squeeze(0).float()          # [seq, d_ff]
        sum_here = x.sum(dim=0)           # [d_ff]
        cnt_here = x.shape[0]

        if sum_vec is None:
            sum_vec = sum_here
        else:
            sum_vec += sum_here
        count += cnt_here

    mean_vals = sum_vec / max(count, 1)
    return mean_vals.detach().cpu()


# -----------------------------
# Main experiment
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--layer_entropy_dir_path", type=str, default="l10_entropy_direction.pt")
    ap.add_argument("--compute_entropy_dir_if_missing", action="store_true")

    # ap.add_argument("--annotated_chains_path", type=str, default="reasoning_chains/all_reasoning_chains.json")
    ap.add_argument("--annotated_chains_path", type=str, default="all_reasoning_chains.json")
    ap.add_argument("--annotated_chains_n", type=int, default=100)
    ap.add_argument("--annotated_chains_seed", type=int, default=0)
    ap.add_argument("--annotated_chains_no_shuffle", action="store_true")

    # Prompt source (to ensure we evaluate prompts that actually backtrack)
    ap.add_argument(
        "--latent_backtracking_repo_dir",
        type=str,
        default="",
        help="Path to a local clone of https://github.com/jnward/latent-backtracking (loads tasks/ + new_tasks/).",
    )
    ap.add_argument(
        "--prompts_path",
        type=str,
        default="",
        help="Path to a .json/.jsonl/.txt file (or directory) containing prompts.",
    )
    ap.add_argument(
        "--use_annotated_chain_problems_as_prompts",
        action="store_true",
        help="Use the 'problem' fields from annotated chains JSON as prompts.",
    )
    ap.add_argument(
        "--prefilter_backtracking",
        action="store_true",
        help="Run a baseline pass and keep only prompts where the model outputs at least one backtracking marker.",
    )
    ap.add_argument("--prefilter_min_backtrack_count", type=int, default=1)
    ap.add_argument("--prefilter_max_candidates", type=int, default=200)
    ap.add_argument("--prefilter_keep_n", type=int, default=50)
    ap.add_argument("--prefilter_max_new_tokens", type=int, default=0,
                    help="0 => use --max_new_tokens for prefiltering.")
    ap.add_argument("--prefilter_temperature", type=float, default=float('nan'),
                    help="NaN => use --temperature for prefiltering.")

    ap.add_argument("--use_final_entropy_neurons", action="store_true")
    ap.add_argument("--null_k", type=int, default=256)
    ap.add_argument("--entropy_neurons_top_n", type=int, default=64)

    # ap.add_argument("--alpha_list", type=str, default="-6,-3,0,3,6")  # strengths for layer-10 direction
    ap.add_argument("--alpha_list", type=str, default="3")  # strengths for layer-10 direction
    ap.add_argument("--out_csv", type=str, default="causal_entropy_backtracking_results.csv")

    args = ap.parse_args()

    device = args.device
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Load the model on a concrete device map. Passing a string like "cuda"
    # explicitly ensures that no weights remain as meta tensors. Adjust for CPU fallback.
    map_val: str
    if device and isinstance(device, str) and device.lower().startswith("cuda"):
        map_val = "cuda"
        device = "cuda"  # ensure consistency for encoding
    elif device and isinstance(device, str) and device.lower().startswith("cpu"):
        map_val = "cpu"
        device = "cpu"
    else:
        # Default to auto device mapping
        map_val = "auto"
    model = LanguageModel(args.model, device_map=map_val, dtype=torch.bfloat16)
    # Warm-up trace to ensure all lazy/meta weights are materialized
    try:
        with model.trace("Hello!") as tracer:
            _ = model.output.save()
    except Exception:
        pass

    # Candidate prompt pool (priority: latent-backtracking repo -> custom prompts file/dir -> annotated chain problems -> fallback)
    candidate_prompts: List[str] = []

    if args.latent_backtracking_repo_dir:
        candidate_prompts = load_latent_backtracking_prompts(args.latent_backtracking_repo_dir)
        if not candidate_prompts:
            print(f"Warning: no prompts found under {args.latent_backtracking_repo_dir}/(tasks|new_tasks).")
    if (not candidate_prompts) and args.prompts_path:
        candidate_prompts = load_prompts_from_path(args.prompts_path)
        if not candidate_prompts:
            print(f"Warning: no prompts loaded from {args.prompts_path}.")
    if (not candidate_prompts) and args.use_annotated_chain_problems_as_prompts and os.path.exists(args.annotated_chains_path):
        chains = load_annotated_chains(args.annotated_chains_path)
        candidate_prompts = [c.get("problem", "") for c in chains if isinstance(c, dict) and isinstance(c.get("problem"), str)]
        candidate_prompts = [p.strip() for p in candidate_prompts if p.strip()]
        if not candidate_prompts:
            print(f"Warning: annotated chains found but no 'problem' strings were extracted from {args.annotated_chains_path}.")

    if not candidate_prompts:
        # Fallback: prompts that often trigger self-correction ("wait"/"hmm") in reasoning models
        candidate_prompts = [
            "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball? Explain your reasoning.",
            "How many months have 28 days? Explain carefully.",
            "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Explain carefully.",
            "A farmer has 15 sheep and all but 8 die. How many are left? Explain carefully.",
            "Compute 48 divided by 1/2. Double-check your answer.",
            "Is 111 a prime number? Show your work and sanity-check the result.",
            "If x+y=10 and x-y=4, what is x? Verify by substitution.",
        ]

    # Optionally prefilter to keep only prompts that *actually* elicit backtracking markers for this model
    baseline_cache: Dict[str, str] = {}
    prompts: List[str] = candidate_prompts

    if args.prefilter_backtracking:
        pf_tokens = args.prefilter_max_new_tokens if args.prefilter_max_new_tokens > 0 else args.max_new_tokens
        pf_temp = args.prefilter_temperature
        if isinstance(pf_temp, float) and math.isnan(pf_temp):
            pf_temp = args.temperature

        prompts, baseline_cache = prefilter_prompts_that_backtrack(
            model=model,
            tokenizer=tokenizer,
            prompts=candidate_prompts,
            min_backtrack_count=args.prefilter_min_backtrack_count,
            max_candidates=args.prefilter_max_candidates,
            keep_n=args.prefilter_keep_n,
            max_new_tokens=pf_tokens,
            temperature=float(pf_temp),
            seed=args.seed,
            device=device,
        )
        if not prompts:
            print(
                "Warning: prefiltering found no backtracking prompts. "
                "Falling back to the first few candidates."
            )
            prompts = candidate_prompts[: max(1, min(len(candidate_prompts), args.prefilter_keep_n))]


    # -----------------------------
    # (A) Layer-10 entropy direction
    # -----------------------------
    # Parse the list of alphas and determine if any intervention is non-zero
    alpha_list = [float(x.strip()) for x in args.alpha_list.split(",")]
    do_direction = any(abs(a) > 1e-12 for a in alpha_list)
    entropy_dir: Optional[torch.Tensor] = None
    # if do_direction:
    #     entropy_dir = maybe_load_direction(args.layer_entropy_dir_path, device=device, dtype=torch.bfloat16)
    #     if entropy_dir is None:
    #         if not args.compute_entropy_dir_if_missing:
    #             raise FileNotFoundError(
    #                 f"Missing {args.layer_entropy_dir_path}. "
    #                 f"Either place it (from entropy_neurons.py) or pass --compute_entropy_dir_if_missing."
    #             )
    #         # Fallback: compute the direction on default prompts
    #         entropy_dir = compute_layer10_entropy_direction(
    #             model=model,
    #             tokenizer=tokenizer,
    #             prompts=prompts,
    #             layer_idx=10,
    #             device=device,
    #         )
    #         torch.save(entropy_dir.detach().cpu().float(), args.layer_entropy_dir_path)

    if os.path.exists(args.annotated_chains_path):
        entropy_dir = compute_layer_entropy_direction_from_annotated_chains(
            model=model,
            tokenizer=tokenizer,
            annotated_chains_path=args.annotated_chains_path,
            n_chains=args.annotated_chains_n,
            seed=args.annotated_chains_seed,
            shuffle=(not args.annotated_chains_no_shuffle),
            layer_idx=10,
            device=device,
        )
    else:
        print(
            f"Warning: annotated chains file not found at {args.annotated_chains_path}. "
            f"Falling back to prompt-generated direction."
        )

    # -----------------------------
    # (B) Final-layer entropy neurons
    # -----------------------------
    final_entropy_neuron_idx = None
    mean_mlp_vals = None

    if args.use_final_entropy_neurons:
        V0 = get_unembed_nullspace_basis(model, k=args.null_k, device=device)  # [d_model, k]
        final_entropy_neuron_idx, stats = select_entropy_neurons_by_rho(
            model=model,
            null_basis_V0=V0,
            layer_idx=-1,
            top_n=args.entropy_neurons_top_n,
            device=device,
        )
        # Reference means for mean-ablation
        mean_mlp_vals = estimate_mlp_mean_activations(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            layer_idx=-1,
            device=device,
        )

    rows = []

    # Baseline + directional interventions
    for prompt in tqdm(prompts, desc="Running causal interventions"):
        # Baseline
        if prompt in baseline_cache:
            baseline = baseline_cache[prompt]
        else:
            baseline = generate_with_optional_interventions(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
                device=device,
            )
        base_think = extract_think_block(baseline)
        rows.append({
            "prompt": prompt,
            "condition": "baseline",
            "backtrack_count": backtracking_count(base_think),
            "backtrack_rate": backtracking_rate(base_think),
            "text": baseline,
        })

        # (A) Add layer-10 entropy direction at different strengths
        # Only run directional interventions when at least one alpha is non-zero
        if do_direction:
            for alpha in alpha_list:
                out = generate_with_optional_interventions(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seed=args.seed,
                    layer10_add=(10, entropy_dir, alpha),
                    device=device,
                )
                think = extract_think_block(out)
                rows.append({
                    "prompt": prompt,
                    "condition": f"layer10_entropy_dir_alpha={alpha}",
                    "backtrack_count": backtracking_count(think),
                    "backtrack_rate": backtracking_rate(think),
                    "text": out,
                })

        # (B) Mean-ablate or amplify final-layer entropy neurons
        if args.use_final_entropy_neurons and final_entropy_neuron_idx is not None:
            out_ablate = generate_with_optional_interventions(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
                final_neuron_ablate=(-1, final_entropy_neuron_idx, mean_mlp_vals),
                device=device,
            )
            think_a = extract_think_block(out_ablate)
            rows.append({
                "prompt": prompt,
                "condition": f"final_entropy_neurons_mean_ablated_top{args.entropy_neurons_top_n}",
                "backtrack_count": backtracking_count(think_a),
                "backtrack_rate": backtracking_rate(think_a),
                "text": out_ablate,
            })

            # Simple amplification test (scale > 1)
            out_amp = generate_with_optional_interventions(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
                final_neuron_scale=(-1, final_entropy_neuron_idx, 1.5),
                device=device,
            )
            think_s = extract_think_block(out_amp)
            rows.append({
                "prompt": prompt,
                "condition": f"final_entropy_neurons_scaled_1.5_top{args.entropy_neurons_top_n}",
                "backtrack_count": backtracking_count(think_s),
                "backtrack_rate": backtracking_rate(think_s),
                "text": out_amp,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    # Print a tiny summary
    summary = df.groupby("condition")[["backtrack_count", "backtrack_rate"]].mean().sort_values(
        "backtrack_rate", ascending=False
    )
    print("\nMean backtracking metrics by condition:")
    print(summary)
    print(f"\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()
