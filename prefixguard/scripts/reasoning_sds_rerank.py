#!/usr/bin/env python3
"""Use SDS/CEBRA as a verifier for reasoning-model rollouts.

This is the reviewer-facing alternative to activation transplantation:
sample k traces from the reasoning model itself, score each trace by how much
its hidden-state trajectory resembles successful reasoning trajectories, and
ask whether SDS/CEBRA can choose a correct trace more often than simple
selection controls.
"""

import argparse
import json
import math
import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from prefixguard.scripts import baselines as base


def as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def load_hard(path: str, n: int) -> List[Dict[str, Any]]:
    rows = json.load(open(path))
    out = []
    for r in rows:
        ans = r.get("ground_truth", r.get("answer"))
        out.append({
            "problem_id": as_int(r.get("problem_id")),
            "problem": r["problem"],
            "answer": ans,
        })
    return out[:n] if n > 0 else out


def load_cache_labels(path: str) -> Dict[int, bool]:
    raw = json.load(open(path))
    return {int(k): bool(v["correct"]) for k, v in raw.items()}


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: str,
    seed: int,
) -> Tuple[str, int, float]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    generated = inputs["input_ids"].clone()
    logps = []
    model.eval()
    with torch.no_grad():
        past = None
        for _ in range(max_new_tokens):
            out = model(
                input_ids=generated[:, -1:] if past is not None else generated,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = torch.nan_to_num(out.logits[:, -1, :].float(), nan=0.0, posinf=1e4, neginf=-1e4)
            if temperature <= 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1).clamp(min=0)
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
                nxt = torch.multinomial(probs, 1)
            logps.append(float(F.log_softmax(logits, dim=-1)[0, nxt.item()].item()))
            generated = torch.cat([generated, nxt], dim=1)
            if tokenizer.eos_token_id is not None and nxt.item() == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(generated[0, input_len:], skip_special_tokens=True)
    return text, int(generated.shape[1] - input_len), float(np.mean(logps) if logps else 0.0)


def continuation_positions(input_len: int, total_len: int, stride: int) -> List[int]:
    start = min(total_len - 1, input_len)
    if total_len <= start:
        return []
    pos = list(range(start, total_len, max(1, stride)))
    if pos[-1] != total_len - 1:
        pos.append(total_len - 1)
    return pos


def load_training_z(
    pkl_path: str,
    sds: Dict[str, Any],
    labels: Dict[int, bool],
    eval_ids: set,
    exclude_eval: bool,
):
    import pickle

    rows = pickle.load(open(pkl_path, "rb"))
    zs, ys, states = [], [], []
    state_counts = np.zeros((sds["K"], 2), dtype=float)
    for r in rows:
        pid = as_int(r.get("problem_id"))
        if pid is None or pid not in labels:
            continue
        if exclude_eval and pid in eval_ids:
            continue
        y = int(labels[pid])
        h = np.asarray(r["hidden_state_last"], dtype=np.float32)
        z = base.embed_hidden(h, sds)
        k = base.infer_regime(z, sds)
        zs.append(z)
        ys.append(y)
        states.append(k)
        state_counts[k, y] += 1
    zs = np.asarray(zs, dtype=np.float32)
    ys = np.asarray(ys, dtype=int)
    state_p = (state_counts[:, 1] + 1.0) / (state_counts.sum(1) + 2.0)
    state_logit = np.log(state_p / (1.0 - state_p))
    state_z = (state_logit - state_logit.mean()) / (state_logit.std() + 1e-8)
    return zs, ys, np.asarray(states, dtype=int), state_p, state_z, state_counts


def load_training_transition_values(
    pkl_path: str,
    sds: Dict[str, Any],
    labels: Dict[int, bool],
    eval_ids: set,
    exclude_eval: bool,
):
    import pickle

    rows = pickle.load(open(pkl_path, "rb"))
    grouped = defaultdict(list)
    for r in rows:
        pid = as_int(r.get("problem_id"))
        sid = as_int(r.get("sentence_idx"))
        if pid is None or sid is None or pid not in labels:
            continue
        if exclude_eval and pid in eval_ids:
            continue
        h = np.asarray(r["hidden_state_last"], dtype=np.float32)
        z = base.embed_hidden(h, sds)
        grouped[pid].append((sid, base.infer_regime(z, sds)))

    counts = np.zeros((sds["K"], sds["K"], 2), dtype=float)
    for pid, vals in grouped.items():
        vals = sorted(vals, key=lambda x: x[0])
        states = [k for _, k in vals]
        if len(states) < 2:
            continue
        y = int(labels[pid])
        for a, b in zip(states[:-1], states[1:]):
            counts[int(a), int(b), y] += 1.0

    p_correct = (counts[:, :, 1] + 1.0) / (counts.sum(2) + 2.0)
    logits = np.log(p_correct / (1.0 - p_correct))
    support = counts.sum(2)
    center_vals = logits[support >= 3] if np.any(support >= 3) else logits.reshape(-1)
    trans_z = (logits - float(center_vals.mean())) / (float(center_vals.std()) + 1e-8)
    return trans_z, p_correct, counts


def fit_cebra_success_probe(zs, ys, seed: int):
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(zs, ys)
    train_prob = clf.predict_proba(zs)[:, 1]
    try:
        auc = float(roc_auc_score(ys, train_prob))
    except Exception:
        auc = float("nan")
    return clf, auc


def score_trace(
    model,
    tokenizer,
    prompt: str,
    generation: str,
    sds: Dict[str, Any],
    state_z: np.ndarray,
    shuffled_state_z: np.ndarray,
    trans_z: np.ndarray,
    shuffled_trans_z: np.ndarray,
    clf,
    layer_idx: int,
    token_stride: int,
    max_score_tokens: int,
    device: str,
) -> Dict[str, Any]:
    full_text = prompt + generation
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_score_tokens).to(device)
    full = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_score_tokens).to(device)
    input_len = min(prompt_ids["input_ids"].shape[1], full["input_ids"].shape[1])
    total_len = full["input_ids"].shape[1]
    pos = continuation_positions(input_len, total_len, token_stride)
    if not pos:
        return {
            "state_score": -1e9,
            "shuffled_state_score": -1e9,
            "cebra_score": -1e9,
            "transition_value": -1e9,
            "shuffled_transition_value": -1e9,
            "transition_ll": -1e9,
            "ctrls_state": -1e9,
            "ctrls_cebra": -1e9,
            "ctrls_shuffled_transition": -1e9,
            "detour_frac": 1.0,
            "states": [],
            "n_states": 0,
        }
    with torch.no_grad():
        out = model(**full, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[min(layer_idx + 1, len(out.hidden_states) - 1)][0]
    states, zs = [], []
    for p in pos:
        h = hs[p].detach().float().cpu().numpy().astype(np.float32)
        z = base.embed_hidden(h, sds)
        zs.append(z)
        states.append(base.infer_regime(z, sds))
    states = np.asarray(states, dtype=int)
    zs = np.asarray(zs, dtype=np.float32)
    cebra_prob = clf.predict_proba(zs)[:, 1]
    state_score = float(np.mean(state_z[states]))
    shuffled_state_score = float(np.mean(shuffled_state_z[states]))
    cebra_score = float(np.mean(np.log(cebra_prob + 1e-8) - np.log(1.0 - cebra_prob + 1e-8)))
    detour_frac = float(np.mean(state_z[states] < 0.0))
    if len(states) >= 2:
        pairs = [(int(a), int(b)) for a, b in zip(states[:-1], states[1:])]
        transition_ll = float(np.mean([math.log(float(sds["A"][a, b]) + 1e-12) for a, b in pairs]))
        transition_value = float(np.mean([trans_z[a, b] for a, b in pairs]))
        shuffled_transition_value = float(np.mean([shuffled_trans_z[a, b] for a, b in pairs]))
    else:
        transition_ll = -12.0
        transition_value = -1e9
        shuffled_transition_value = -1e9
    return {
        "state_score": state_score,
        "shuffled_state_score": shuffled_state_score,
        "cebra_score": cebra_score,
        "transition_ll": transition_ll,
        "transition_value": transition_value,
        "shuffled_transition_value": shuffled_transition_value,
        "joint_state": float(state_score + 0.05 * transition_ll - 0.25 * detour_frac),
        "joint_cebra": float(cebra_score + 0.05 * transition_ll - 0.25 * detour_frac),
        "ctrls_state": float(state_score + 0.50 * transition_value + 0.05 * transition_ll - 0.25 * detour_frac),
        "ctrls_cebra": float(cebra_score + 0.50 * transition_value + 0.05 * transition_ll - 0.25 * detour_frac),
        "ctrls_shuffled_transition": float(cebra_score + 0.50 * shuffled_transition_value + 0.05 * transition_ll - 0.25 * detour_frac),
        "detour_frac": detour_frac,
        "states": states.tolist(),
        "n_states": int(len(states)),
    }


def answer_key(text: str) -> str:
    cands = base.extract_answer_candidates(text)
    if cands:
        key = base.simple_normalize(cands[0])
        if key:
            return key
    return "__none__"


def argmax(items, key):
    return int(np.argmax([x[key] for x in items]))


def argmin(items, key):
    return int(np.argmin([x[key] for x in items]))


def majority_pick(items):
    counts = Counter(answer_key(x["generation"]) for x in items)
    best_count = max(counts.values())
    winners = {k for k, v in counts.items() if v == best_count}
    for i, x in enumerate(items):
        if answer_key(x["generation"]) in winners:
            return i
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning-model", required=True)
    ap.add_argument("--reasoning-pkl", required=True)
    ap.add_argument("--rlvr-cache", required=True)
    ap.add_argument("--hard-json", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--prompt-style", default="chat")
    ap.add_argument("--layer-idx", type=int, default=20)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--cebra-dim", type=int, default=16)
    ap.add_argument("--em-iters", type=int, default=30)
    ap.add_argument("--limit-problems", type=int, default=500)
    ap.add_argument("--max-triplets", type=int, default=25)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--k-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-score-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--token-stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-eval-in-labels", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    problems = load_hard(args.hard_json, args.n_problems)
    eval_ids = {p["problem_id"] for p in problems if p["problem_id"] is not None}
    labels = load_cache_labels(args.rlvr_cache)

    print("Fitting reasoning SDS...", flush=True)
    sds = base.fit_sds_and_decoder(
        args.reasoning_pkl,
        args.K,
        args.cebra_dim,
        args.em_iters,
        args.limit_problems,
        args.max_triplets,
        args.seed,
    )
    zs, ys, train_states, state_p, state_z, state_counts = load_training_z(
        args.reasoning_pkl,
        sds,
        labels,
        eval_ids,
        exclude_eval=not args.include_eval_in_labels,
    )
    trans_z, trans_p, trans_counts = load_training_transition_values(
        args.reasoning_pkl,
        sds,
        labels,
        eval_ids,
        exclude_eval=not args.include_eval_in_labels,
    )
    clf, train_auc = fit_cebra_success_probe(zs, ys, args.seed)
    shuffled_state_z = rng.permutation(state_z)
    shuffled_trans_z = rng.permutation(trans_z.reshape(-1)).reshape(trans_z.shape)
    print(f"Training states: n={len(ys)} positives={int(ys.sum())} train_auc={train_auc:.3f}", flush=True)
    print("State P(correct):", " ".join(f"S{k}={p:.3f}" for k, p in enumerate(state_p)), flush=True)
    print("State utility z:", " ".join(f"S{k}={v:+.3f}" for k, v in enumerate(state_z)), flush=True)
    flat = [(i, j, trans_p[i, j], trans_counts[i, j].sum()) for i in range(sds["K"]) for j in range(sds["K"])]
    flat = sorted(flat, key=lambda x: (x[2], x[3]), reverse=True)[:8]
    print("Top transition values:", " ".join(f"S{i}->S{j}:p={pc:.2f},n={int(n)}" for i, j, pc, n in flat), flush=True)

    print(f"Loading reasoning model: {args.reasoning_model}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.reasoning_model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.reasoning_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    choices = defaultdict(list)
    rows = []
    score_keys = [
        "state_score", "shuffled_state_score", "cebra_score",
        "transition_value", "shuffled_transition_value", "transition_ll",
        "joint_state", "joint_cebra",
        "ctrls_state", "ctrls_cebra", "ctrls_shuffled_transition",
    ]
    candidate_scores = defaultdict(list)
    candidate_labels = []

    for i, prob in enumerate(problems):
        prompt = base.format_prompt(tokenizer, prob["problem"], args.prompt_style)
        cands = []
        for s in range(args.k_samples):
            seed = args.seed + 100003 * (i + 1) + 1009 * s
            text, gen_len, avg_lp = generate_one(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
                device,
                seed,
            )
            correct = base.is_correct_generation(text, prob["answer"], args.dataset)
            scores = score_trace(
                model,
                tokenizer,
                prompt,
                text,
                sds,
                state_z,
                shuffled_state_z,
                trans_z,
                shuffled_trans_z,
                clf,
                args.layer_idx,
                args.token_stride,
                args.max_score_tokens,
                device,
            )
            cand = {
                "sample_idx": s,
                "generation": text,
                "correct": bool(correct),
                "gen_len": gen_len,
                "avg_lp": avg_lp,
                "length_chars": len(text),
                "length_tokens": int(tokenizer(text, return_tensors="pt")["input_ids"].shape[1]),
                "answer_key": answer_key(text),
                **scores,
            }
            cands.append(cand)
            for key in score_keys:
                candidate_scores[key].append(float(cand[key]))
            candidate_scores["avg_lp"].append(float(avg_lp))
            candidate_labels.append(int(correct))

        picks = {
            "first": 0,
            "random": int(rng.integers(0, len(cands))),
            "majority_answer": majority_pick(cands),
            "longest": argmax(cands, "length_tokens"),
            "shortest": argmin(cands, "length_tokens"),
            "avg_lp": argmax(cands, "avg_lp"),
            "state_score": argmax(cands, "state_score"),
            "shuffled_state_score": argmax(cands, "shuffled_state_score"),
            "cebra_score": argmax(cands, "cebra_score"),
            "joint_state": argmax(cands, "joint_state"),
            "joint_cebra": argmax(cands, "joint_cebra"),
            "transition_value": argmax(cands, "transition_value"),
            "ctrls_state": argmax(cands, "ctrls_state"),
            "ctrls_cebra": argmax(cands, "ctrls_cebra"),
            "ctrls_shuffled_transition": argmax(cands, "ctrls_shuffled_transition"),
            "transition_ll": argmax(cands, "transition_ll"),
        }
        for name, idx in picks.items():
            choices[name].append(bool(cands[idx]["correct"]))
        choices["oracle"].append(any(c["correct"] for c in cands))
        rows.append({
            "problem_id": prob["problem_id"],
            "problem": prob["problem"],
            "answer": prob["answer"],
            "count_correct": int(sum(c["correct"] for c in cands)),
            "oracle": bool(any(c["correct"] for c in cands)),
            "picks": picks,
            "pick_correct": {name: bool(cands[idx]["correct"]) for name, idx in picks.items()},
            "candidates": cands,
        })
        print(
            f"  {i+1}/{len(problems)} first={np.mean(choices['first']):.3f} "
            f"joint_cebra={np.mean(choices['joint_cebra']):.3f} "
            f"ctrls_cebra={np.mean(choices['ctrls_cebra']):.3f} "
            f"state={np.mean(choices['state_score']):.3f} "
            f"majority={np.mean(choices['majority_answer']):.3f} "
            f"oracle={np.mean(choices['oracle']):.3f}",
            flush=True,
        )

    aucs = {}
    labels_arr = np.asarray(candidate_labels, dtype=int)
    for key, vals in candidate_scores.items():
        try:
            aucs[key] = float(roc_auc_score(labels_arr, np.asarray(vals, dtype=float)))
        except Exception:
            aucs[key] = float("nan")

    summary = {
        "config": vars(args),
        "n_problems": len(rows),
        "state_p_correct": state_p.tolist(),
        "state_utility_z": state_z.tolist(),
        "state_counts_wrong_correct": state_counts.tolist(),
        "transition_p_correct": trans_p.tolist(),
        "transition_counts_wrong_correct": trans_counts.tolist(),
        "cebra_probe_train_auc": train_auc,
        "accuracy": {k: float(np.mean(v)) for k, v in choices.items()},
        "candidate_auc": aucs,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.write_text(json.dumps(summary, indent=2))
    print("Accuracy:", json.dumps(summary["accuracy"], indent=2), flush=True)
    print("Candidate AUC:", json.dumps(summary["candidate_auc"], indent=2), flush=True)
    print(f"Saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
