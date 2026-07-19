#!/usr/bin/env python3
"""Analysis for SDS/CEBRA prefix-verifier results.

This script turns the reasoning rollout reranker JSON into reviewer-facing
evidence:

  1. robustness tables across simple distribution-shift slices;
  2. residual/information-gap analysis versus average log-probability;
  3. an interpretable case where log-prob selects a wrong trace but SDS/CEBRA
     selects a correct one;
  4. optional early-exit scoring by re-running the model on saved generation
     prefixes at 25/50/75/100%.

The early-exit mode is intentionally read-only: it does not regenerate samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from prefixguard.scripts import baselines as base


SCORE_KEYS = [
    "avg_lp",
    "state_score",
    "cebra_score",
    "transition_ll",
    "joint_cebra",
]


def as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def load_cache_labels(path: str) -> Dict[int, bool]:
    raw = json.load(open(path))
    return {int(k): bool(v["correct"]) for k, v in raw.items()}


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


def quantile_bins(values: List[float], names=("short", "middle", "long")):
    vals = np.asarray(values, dtype=float)
    q1, q2 = np.quantile(vals, [1.0 / 3.0, 2.0 / 3.0])
    out = []
    for v in vals:
        if v <= q1:
            out.append(names[0])
        elif v <= q2:
            out.append(names[1])
        else:
            out.append(names[2])
    return out


def answer_form(answer: Any) -> str:
    s = str(answer).strip()
    if re.fullmatch(r"-?\d+", s):
        return "integer"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return "decimal"
    if "\\frac" in s or "/" in s:
        return "fraction"
    return "symbolic"


def row_slices(rows: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    problem_words = [len(str(r.get("problem", "")).split()) for r in rows]
    answer_lens = [len(str(r.get("answer", ""))) for r in rows]
    problem_bins = quantile_bins(problem_words)
    answer_bins = quantile_bins(answer_lens, names=("short_answer", "mid_answer", "long_answer"))

    slices = defaultdict(list)
    slices["all"] = list(range(len(rows)))
    for i, b in enumerate(problem_bins):
        slices[f"problem_len:{b}"].append(i)
    for i, b in enumerate(answer_bins):
        slices[f"answer_len:{b}"].append(i)
    for i, r in enumerate(rows):
        slices[f"answer_form:{answer_form(r.get('answer'))}"].append(i)
        slices[f"oracle:{bool(r.get('oracle'))}"].append(i)
    return dict(slices)


def safe_auc(labels: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def pick_accuracy(rows: List[Dict[str, Any]], idxs: List[int], key: str) -> float:
    vals = []
    for i in idxs:
        cands = rows[i]["candidates"]
        if not cands:
            continue
        j = int(np.argmax([float(c.get(key, -1e9)) for c in cands]))
        vals.append(bool(cands[j].get("correct", False)))
    return float(np.mean(vals)) if vals else float("nan")


def candidate_auc(rows: List[Dict[str, Any]], idxs: List[int], key: str) -> float:
    labels, scores = [], []
    for i in idxs:
        for c in rows[i]["candidates"]:
            labels.append(int(bool(c.get("correct", False))))
            scores.append(float(c.get(key, -1e9)))
    return safe_auc(labels, scores)


def robustness_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    slices = row_slices(rows)
    table = []
    for name, idxs in sorted(slices.items()):
        if len(idxs) < 3:
            continue
        rec: Dict[str, Any] = {
            "slice": name,
            "n_problems": len(idxs),
            "oracle": float(np.mean([bool(rows[i].get("oracle", False)) for i in idxs])),
        }
        for key in SCORE_KEYS:
            rec[f"pick_acc:{key}"] = pick_accuracy(rows, idxs, key)
            rec[f"candidate_auc:{key}"] = candidate_auc(rows, idxs, key)
        table.append(rec)
    return table


def flatten_candidates(rows: List[Dict[str, Any]]):
    xs = []
    labels = []
    groups = []
    meta = []
    for i, r in enumerate(rows):
        group = as_int(r.get("problem_id"))
        if group is None:
            group = i
        for j, c in enumerate(r["candidates"]):
            xs.append(
                {
                    "avg_lp": float(c.get("avg_lp", 0.0)),
                    "cebra_score": float(c.get("cebra_score", 0.0)),
                    "state_score": float(c.get("state_score", 0.0)),
                    "transition_ll": float(c.get("transition_ll", 0.0)),
                    "joint_cebra": float(c.get("joint_cebra", 0.0)),
                }
            )
            labels.append(int(bool(c.get("correct", False))))
            groups.append(group)
            meta.append((i, j))
    return xs, np.asarray(labels, dtype=int), np.asarray(groups), meta


def residual_analysis(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    xs, y, groups, meta = flatten_candidates(rows)
    feature_sets = {
        "logprob": ["avg_lp"],
        "cebra": ["cebra_score"],
        "logprob_plus_cebra": ["avg_lp", "cebra_score"],
        "logprob_plus_state_transition": ["avg_lp", "state_score", "transition_ll"],
        "all_sds": ["cebra_score", "state_score", "transition_ll"],
        "all": ["avg_lp", "cebra_score", "state_score", "transition_ll"],
    }
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    results = {}
    for name, keys in feature_sets.items():
        X = np.asarray([[x[k] for k in keys] for x in xs], dtype=float)
        probs = np.full(len(y), np.nan, dtype=float)
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in splitter.split(X, y, groups):
            if len(set(y[train_idx].tolist())) < 2:
                continue
            scaler = StandardScaler().fit(X[train_idx])
            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )
            clf.fit(scaler.transform(X[train_idx]), y[train_idx])
            probs[test_idx] = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]

        valid = ~np.isnan(probs)
        auc = safe_auc(y[valid], probs[valid]) if np.any(valid) else float("nan")
        by_problem = defaultdict(list)
        for k, (i, j) in enumerate(meta):
            if not np.isnan(probs[k]):
                by_problem[i].append((j, float(probs[k])))
        pick_correct = []
        for i, vals in by_problem.items():
            j = max(vals, key=lambda x: x[1])[0]
            pick_correct.append(bool(rows[i]["candidates"][j].get("correct", False)))
        results[name] = {
            "features": keys,
            "candidate_auc_group_cv": auc,
            "pick_accuracy_group_cv": float(np.mean(pick_correct)) if pick_correct else float("nan"),
            "n_scored_candidates": int(np.sum(valid)),
        }

    lp = np.asarray([x["avg_lp"] for x in xs], dtype=float)
    cebra = np.asarray([x["cebra_score"] for x in xs], dtype=float)
    corr = float(np.corrcoef(lp, cebra)[0, 1]) if len(lp) > 1 else float("nan")
    results["feature_correlation"] = {"avg_lp_vs_cebra_score": corr}
    return results


def rle_states(states: List[int]) -> List[List[int]]:
    if not states:
        return []
    out = []
    cur = states[0]
    count = 1
    for s in states[1:]:
        if s == cur:
            count += 1
        else:
            out.append([int(cur), int(count)])
            cur = s
            count = 1
    out.append([int(cur), int(count)])
    return out


def state_summary(states: List[int], state_utility: List[float]) -> Dict[str, Any]:
    if not states:
        return {}
    vals = np.asarray([state_utility[int(s)] for s in states], dtype=float)
    return {
        "n_states": len(states),
        "mean_state_utility": float(np.mean(vals)),
        "risk_state_frac": float(np.mean(vals < 0.0)),
        "state_histogram": {str(k): int(v) for k, v in Counter(states).items()},
        "state_rle": rle_states(states)[:40],
    }


def find_interpretability_example(rows: List[Dict[str, Any]], state_utility: List[float]) -> Dict[str, Any]:
    candidates = []
    for i, r in enumerate(rows):
        cands = r["candidates"]
        if not cands:
            continue
        lp_j = int(np.argmax([float(c.get("avg_lp", -1e9)) for c in cands]))
        ce_j = int(np.argmax([float(c.get("cebra_score", -1e9)) for c in cands]))
        joint_j = int(np.argmax([float(c.get("joint_cebra", -1e9)) for c in cands]))
        if not cands[lp_j].get("correct", False) and cands[ce_j].get("correct", False):
            candidates.append((i, lp_j, ce_j, "cebra_score"))
        elif not cands[lp_j].get("correct", False) and cands[joint_j].get("correct", False):
            candidates.append((i, lp_j, joint_j, "joint_cebra"))
    if not candidates:
        return {"found": False}

    i, lp_j, sds_j, score_name = candidates[0]
    row = rows[i]

    def pack(j):
        c = row["candidates"][j]
        text = str(c.get("generation", ""))
        return {
            "sample_idx": int(c.get("sample_idx", j)),
            "correct": bool(c.get("correct", False)),
            "answer_key": c.get("answer_key"),
            "avg_lp": float(c.get("avg_lp", float("nan"))),
            "cebra_score": float(c.get("cebra_score", float("nan"))),
            "joint_cebra": float(c.get("joint_cebra", float("nan"))),
            "transition_ll": float(c.get("transition_ll", float("nan"))),
            "detour_frac": float(c.get("detour_frac", float("nan"))),
            "state_summary": state_summary(c.get("states", []), state_utility),
            "generation_excerpt": text[:900],
        }

    return {
        "found": True,
        "problem_id": row.get("problem_id"),
        "problem": row.get("problem"),
        "answer": row.get("answer"),
        "score_that_fixed_logprob": score_name,
        "logprob_pick": pack(lp_j),
        "sds_pick": pack(sds_j),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def score_prefix(
    model,
    tokenizer,
    prompt: str,
    generation: str,
    sds: Dict[str, Any],
    clf,
    layer_idx: int,
    token_stride: int,
    fraction: float,
    device: str,
) -> Dict[str, Any]:
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    gen_ids = tokenizer(generation, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(gen_ids) == 0:
        return {
            "n_prefix_tokens": 0,
            "avg_lp_prefix": -1e9,
            "cebra_score_prefix": -1e9,
            "state_score_prefix": -1e9,
            "transition_ll_prefix": -1e9,
            "n_states_prefix": 0,
        }
    n = max(1, min(len(gen_ids), int(math.ceil(float(fraction) * len(gen_ids)))))
    input_ids = torch.cat([prompt_ids, gen_ids[:n]], dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    input_len = int(len(prompt_ids))
    total_len = int(input_ids.shape[1])
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    logits = out.logits.float()
    target = input_ids[:, input_len:total_len]
    pred_logits = logits[:, input_len - 1:total_len - 1, :]
    lp = F.log_softmax(pred_logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    avg_lp = float(lp.mean().item()) if lp.numel() else -1e9

    pos = continuation_positions(input_len, total_len, token_stride)
    if not pos:
        return {
            "n_prefix_tokens": int(n),
            "avg_lp_prefix": avg_lp,
            "cebra_score_prefix": -1e9,
            "state_score_prefix": -1e9,
            "transition_ll_prefix": -1e9,
            "n_states_prefix": 0,
        }

    hs = out.hidden_states[min(layer_idx + 1, len(out.hidden_states) - 1)][0]
    states, zs = [], []
    for p in pos:
        h = hs[p].detach().float().cpu().numpy().astype(np.float32)
        z = base.embed_hidden(h, sds)
        zs.append(z)
        states.append(base.infer_regime(z, sds))
    states_np = np.asarray(states, dtype=int)
    zs_np = np.asarray(zs, dtype=np.float32)
    cebra_prob = clf.predict_proba(zs_np)[:, 1]
    cebra_score = float(np.mean(np.log(cebra_prob + 1e-8) - np.log(1.0 - cebra_prob + 1e-8)))
    state_p = np.asarray(sds.get("state_utility_z_for_prefix_analysis"), dtype=float)
    state_score = float(np.mean(state_p[states_np]))
    if len(states_np) >= 2:
        transition_ll = float(np.mean([
            math.log(float(sds["A"][int(a), int(b)]) + 1e-12)
            for a, b in zip(states_np[:-1], states_np[1:])
        ]))
    else:
        transition_ll = -12.0
    return {
        "n_prefix_tokens": int(n),
        "avg_lp_prefix": avg_lp,
        "cebra_score_prefix": cebra_score,
        "state_score_prefix": state_score,
        "transition_ll_prefix": transition_ll,
        "n_states_prefix": int(len(states_np)),
    }


def early_exit_analysis(args, result: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    labels = load_cache_labels(args.rlvr_cache)
    eval_ids = {
        as_int(r.get("problem_id"))
        for r in result["rows"]
        if as_int(r.get("problem_id")) is not None
    }
    print("Fitting SDS/CEBRA probe for early-exit analysis...", flush=True)
    sds = base.fit_sds_and_decoder(
        args.reasoning_pkl,
        args.K,
        args.cebra_dim,
        args.em_iters,
        args.limit_problems,
        args.max_triplets,
        args.seed,
    )
    zs, ys, _, state_p, state_z, _ = load_training_z(
        args.reasoning_pkl,
        sds,
        labels,
        eval_ids,
        exclude_eval=not args.include_eval_in_labels,
    )
    clf, train_auc = fit_cebra_success_probe(zs, ys, args.seed)
    sds["state_utility_z_for_prefix_analysis"] = state_z

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    fractions = [float(x) for x in args.early_fractions.split(",")]
    rows = result["rows"][: args.n_problems if args.n_problems > 0 else None]
    early_rows = []
    score_keys = ["avg_lp_prefix", "cebra_score_prefix", "state_score_prefix"]
    for i, row in enumerate(rows):
        prompt = base.format_prompt(tokenizer, row["problem"], args.prompt_style)
        cands = row["candidates"][: args.k_samples if args.k_samples > 0 else None]
        per_fraction = {f: [] for f in fractions}
        for j, cand in enumerate(cands):
            for f in fractions:
                scores = score_prefix(
                    model,
                    tokenizer,
                    prompt,
                    cand.get("generation", ""),
                    sds,
                    clf,
                    args.layer_idx,
                    args.token_stride,
                    f,
                    device,
                )
                rec = {
                    "problem_id": row.get("problem_id"),
                    "problem_index": i,
                    "sample_idx": int(cand.get("sample_idx", j)),
                    "fraction": f,
                    "correct": bool(cand.get("correct", False)),
                    **scores,
                }
                early_rows.append(rec)
                per_fraction[f].append(rec)
        msg = []
        for f in fractions:
            accs = []
            for key in score_keys:
                vals = per_fraction[f]
                if vals:
                    pick = max(vals, key=lambda x: x[key])
                    accs.append(f"{key.replace('_prefix','')}={int(pick['correct'])}")
            msg.append(f"{f:.2f}:" + ",".join(accs))
        print(f"  early {i+1}/{len(rows)} " + " ".join(msg), flush=True)

    summary = {"cebra_probe_train_auc": train_auc, "fractions": {}}
    for f in fractions:
        vals = [r for r in early_rows if abs(r["fraction"] - f) < 1e-9]
        by_problem = defaultdict(list)
        for r in vals:
            by_problem[r["problem_index"]].append(r)
        fsum = {}
        for key in score_keys:
            picks = []
            labels_, scores_ = [], []
            for rs in by_problem.values():
                pick = max(rs, key=lambda x: x[key])
                picks.append(bool(pick["correct"]))
                labels_.extend([int(bool(x["correct"])) for x in rs])
                scores_.extend([float(x[key]) for x in rs])
            fsum[key] = {
                "pick_accuracy": float(np.mean(picks)) if picks else float("nan"),
                "candidate_auc": safe_auc(labels_, scores_),
            }
        summary["fractions"][str(f)] = fsum

    write_csv(out_dir / "early_exit_candidate_scores.csv", early_rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = fractions
        plt.figure(figsize=(7, 4))
        for key, label in [
            ("avg_lp_prefix", "avg log-prob"),
            ("cebra_score_prefix", "CEBRA verifier"),
            ("state_score_prefix", "SDS state score"),
        ]:
            ys_plot = [summary["fractions"][str(f)][key]["pick_accuracy"] for f in xs]
            plt.plot(xs, ys_plot, marker="o", label=label)
        plt.xlabel("Fraction of generation observed")
        plt.ylabel("Selection accuracy")
        plt.ylim(0.0, 1.0)
        plt.title("Early-exit verifier accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "early_exit_accuracy.png", dpi=200)
        plt.close()
    except Exception as e:
        summary["plot_error"] = repr(e)
    return summary


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    lines = []
    lines.append("# SDS/CEBRA Prefix-Verifier Analysis")
    lines.append("")
    lines.append("## Core Selection Accuracy")
    for k, v in result["base_accuracy"].items():
        lines.append(f"- `{k}`: {v:.3f}")
    lines.append("")
    lines.append("## Information Gap")
    for k, v in result["residual_analysis"].items():
        if not isinstance(v, dict) or "candidate_auc_group_cv" not in v:
            continue
        lines.append(
            f"- `{k}`: candidate AUC {v['candidate_auc_group_cv']:.3f}, "
            f"pick accuracy {v['pick_accuracy_group_cv']:.3f}"
        )
    corr = result["residual_analysis"].get("feature_correlation", {}).get("avg_lp_vs_cebra_score")
    if corr is not None:
        lines.append(f"- `corr(avg_lp, cebra_score)`: {corr:.3f}")
    lines.append("")
    lines.append("## Robustness Slices")
    for r in result["robustness_table"][:20]:
        lines.append(
            f"- `{r['slice']}` n={r['n_problems']} oracle={r['oracle']:.3f}: "
            f"avg_lp={r.get('pick_acc:avg_lp', float('nan')):.3f}, "
            f"cebra={r.get('pick_acc:cebra_score', float('nan')):.3f}, "
            f"joint={r.get('pick_acc:joint_cebra', float('nan')):.3f}"
        )
    if "early_exit" in result:
        lines.append("")
        lines.append("## Early Exit")
        for f, vals in result["early_exit"]["fractions"].items():
            lines.append(
                f"- `{f}`: avg_lp={vals['avg_lp_prefix']['pick_accuracy']:.3f}, "
                f"cebra={vals['cebra_score_prefix']['pick_accuracy']:.3f}, "
                f"state={vals['state_score_prefix']['pick_accuracy']:.3f}"
            )
    ex = result.get("interpretability_example", {})
    lines.append("")
    lines.append("## Interpretability Example")
    if not ex.get("found"):
        lines.append("No logprob-wrong / SDS-correct example found in this run.")
    else:
        lines.append(f"- Problem id: `{ex['problem_id']}`")
        lines.append(f"- Answer: `{ex['answer']}`")
        lines.append(f"- SDS score that fixed log-prob: `{ex['score_that_fixed_logprob']}`")
        lp = ex["logprob_pick"]
        sds = ex["sds_pick"]
        lines.append(
            f"- Log-prob pick: correct={lp['correct']}, avg_lp={lp['avg_lp']:.3f}, "
            f"cebra={lp['cebra_score']:.3f}, detour={lp['detour_frac']:.3f}"
        )
        lines.append(
            f"- SDS pick: correct={sds['correct']}, avg_lp={sds['avg_lp']:.3f}, "
            f"cebra={sds['cebra_score']:.3f}, detour={sds['detour_frac']:.3f}"
        )
        lines.append("- Log-prob state summary:")
        lines.append("```json")
        lines.append(json.dumps(lp["state_summary"], indent=2)[:2500])
        lines.append("```")
        lines.append("- SDS state summary:")
        lines.append("```json")
        lines.append(json.dumps(sds["state_summary"], indent=2)[:2500])
        lines.append("```")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-early-exit", action="store_true")
    ap.add_argument("--reasoning-model")
    ap.add_argument("--reasoning-pkl")
    ap.add_argument("--rlvr-cache")
    ap.add_argument("--prompt-style", default="chat")
    ap.add_argument("--layer-idx", type=int, default=20)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--cebra-dim", type=int, default=16)
    ap.add_argument("--em-iters", type=int, default=30)
    ap.add_argument("--limit-problems", type=int, default=500)
    ap.add_argument("--max-triplets", type=int, default=25)
    ap.add_argument("--token-stride", type=int, default=8)
    ap.add_argument("--early-fractions", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--n-problems", type=int, default=0)
    ap.add_argument("--k-samples", type=int, default=0)
    ap.add_argument("--include-eval-in-labels", action="store_true")
    args = ap.parse_args()

    result = json.load(open(args.result_json))
    rows = result["rows"][: args.n_problems if args.n_problems > 0 else None]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_utility = result.get("state_utility_z", result.get("state_p_correct", []))
    analysis = {
        "config": vars(args),
        "source_result": args.result_json,
        "base_accuracy": result.get("accuracy", {}),
        "base_candidate_auc": result.get("candidate_auc", {}),
        "robustness_table": robustness_table(rows),
        "residual_analysis": residual_analysis(rows, args.seed),
        "interpretability_example": find_interpretability_example(rows, state_utility),
    }
    write_csv(out_dir / "robustness_table.csv", analysis["robustness_table"])

    if not args.skip_early_exit:
        missing = [
            name for name in ["reasoning_model", "reasoning_pkl", "rlvr_cache"]
            if not getattr(args, name)
        ]
        if missing:
            raise ValueError(f"Early-exit mode requires: {missing}")
        analysis["early_exit"] = early_exit_analysis(args, result, out_dir)

    (out_dir / "prefix_analysis.json").write_text(json.dumps(analysis, indent=2))
    write_markdown(out_dir / "prefix_summary.md", analysis)
    print(json.dumps({
        "base_accuracy": analysis["base_accuracy"],
        "residual_analysis": analysis["residual_analysis"],
        "early_exit": analysis.get("early_exit", {}).get("fractions", {}),
        "out_dir": str(out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
