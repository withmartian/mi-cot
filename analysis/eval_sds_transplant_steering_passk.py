"""
Behavioral SDS-transplant evaluation with pass@k.

This is separate from transplant_sds_into_base.py:
- transplant_sds_into_base.py: predictive transfer metrics (R^2 / NLL / timing proxy)
- this script: generation-time behavioral eval (pass@k) on base vs SDS-steered base

Steering policy:
- SLDS is fit on reasoning-model features (via cebra_em_steering_inputDep artifacts)
- During base-model generation, infer current state from base hidden state
- Use reasoning SLDS transition row to decide target regime
- Apply steering vector learned from reasoning artifacts at the chosen layer
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from cebra_em_steering_inputDep import SteerConfig, run_pipeline
from sds_train_gsm8k_hf import SDS_TRAIN_GSM8K_REPO_ID


DEFAULT_SOURCE_RELPATH = "qwen1.5b_reasoning/layer_27/all_sentences_features.pkl"
DEFAULT_TARGET_RELPATH = "qwen_1.5B_base/layer_27/all_sentences_features.pkl"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_pickle_list(path: str) -> List[dict]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"Expected list pickle at {path!r}, got {type(obj)}")
    return obj


def _download_features(repo_id: str, relpath: str, token: Optional[str]) -> str:
    return hf_hub_download(repo_id=repo_id, filename=relpath, token=token, repo_type="dataset")


def _infer_layer_from_relpath(relpath: str) -> int:
    m = re.search(r"layer_(\d+)", relpath.replace("\\", "/"))
    if not m:
        raise ValueError(f"Could not infer layer from relpath: {relpath}")
    return int(m.group(1))


def _extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not nums:
        return None
    return nums[-1].replace(",", "")


def _normalize_answer(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def _gsm8k_gold_answer(ans: str) -> str:
    if "####" in ans:
        return ans.split("####")[-1].strip()
    return ans.strip()


def _openai_chat_json_object(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: int,
    max_retries: int,
) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = str(e)
            last_err = f"HTTP {e.code}: {err_body[:500]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.0 + 0.5 * attempt)
    return {"error": last_err or "openai_request_failed"}


def _judge_correctness(
    *,
    api_key: str,
    judge_model: str,
    question: str,
    gold_answer: str,
    model_answer: str,
) -> Tuple[bool, Dict[str, Any]]:
    system = (
        "You are a strict math QA grader. Determine if the model answer is mathematically equivalent "
        "to the gold answer. Return strict JSON only."
    )
    user = f"""Question:
{question}

Gold answer:
{gold_answer}

Model answer:
{model_answer}

Return:
{{
  "correct": <true|false>,
  "confidence": <0..1>,
  "reason": "<short>"
}}"""
    out = _openai_chat_json_object(
        api_key=api_key,
        model=judge_model,
        system_prompt=system,
        user_prompt=user,
        timeout_s=60,
        max_retries=3,
    )
    if "error" in out:
        return False, out
    return bool(out.get("correct", False)), out


@dataclass
class SDSSteerer:
    a: np.ndarray
    state_centroids: Dict[int, np.ndarray]
    vec_by_source_state: Dict[int, np.ndarray]
    alpha: float
    steer_threshold: float = 0.25
    # high_margin: only steer when off-diagonal mass exceeds threshold (can skip all steps).
    # argmax_offdiag: always treat SDS "next regime" as argmax transition off the diagonal (more interventions).
    steer_next_mode: str = "high_margin"

    def infer_state(self, h: np.ndarray) -> int:
        best_s = None
        best_d = None
        for s, c in self.state_centroids.items():
            d = float(np.sum((h - c) ** 2))
            if best_d is None or d < best_d:
                best_d = d
                best_s = s
        if best_s is None:
            return 0
        return int(best_s)

    def marginal_argmax_next(self, s: int) -> int:
        """Most likely next regime including self-transition, P(s'|s)."""
        row = self.a[s].astype(np.float64)
        return int(np.argmax(row))

    def argmax_offdiag_next(self, s: int) -> Optional[int]:
        if s < 0 or s >= self.a.shape[0]:
            return None
        row = self.a[s].astype(np.float64).copy()
        row[s] = -np.inf
        t = int(np.argmax(row))
        if not np.isfinite(row[t]):
            return None
        return int(t)

    def choose_target(self, s: int) -> Optional[int]:
        if s < 0 or s >= self.a.shape[0]:
            return None
        if self.steer_next_mode == "argmax_offdiag":
            return self.argmax_offdiag_next(s)
        row = self.a[s].astype(np.float64).copy()
        self_prob = float(row[s])
        row[s] = -np.inf
        t = int(np.argmax(row))
        if not np.isfinite(row[t]):
            return None
        if float(row[t]) < float(self.steer_threshold) or float(row[t]) <= self_prob:
            return None
        return int(t)

    def vector_for_state(self, s: int) -> Optional[np.ndarray]:
        v = self.vec_by_source_state.get(int(s))
        if v is None:
            return None
        return (self.alpha * v).astype(np.float32)


def _build_steerer_from_payload(
    payload: Dict[str, Any],
    source_features: List[dict],
    layer: int,
    alpha: float,
    steer_threshold: float,
    steer_next_mode: str,
    include_steering_vectors: bool = True,
) -> SDSSteerer:
    _ = layer  # layer is explicit in CLI for generation hook; vectors are from same-layer features.
    included = payload["included_indices"]
    per_sample_state = np.asarray(payload["per_sample_state"], dtype=np.int64)
    steering_cache: Dict[int, dict] = payload["steering_cache"]
    a = np.asarray(payload["A"], dtype=np.float64)

    x = np.array([r["hidden_state_last"] for r in source_features], dtype=np.float32)
    state_centroids: Dict[int, np.ndarray] = {}
    for s in sorted(set(int(per_sample_state[i]) for i in included)):
        idxs = [i for i in included if int(per_sample_state[i]) == s]
        if idxs:
            state_centroids[s] = np.mean(x[idxs], axis=0).astype(np.float32)

    vec_by_source_state: Dict[int, np.ndarray] = {}
    if include_steering_vectors:
        grouped: Dict[int, List[np.ndarray]] = {}
        for idx, row in steering_cache.items():
            s = int(row["source_state"])
            grouped.setdefault(s, []).append(np.asarray(row["delta_x_raw"], dtype=np.float32))
        for s, rows in grouped.items():
            if rows:
                vec_by_source_state[s] = np.mean(np.stack(rows, axis=0), axis=0).astype(np.float32)

    return SDSSteerer(
        a=a,
        state_centroids=state_centroids,
        vec_by_source_state=vec_by_source_state,
        alpha=float(alpha),
        steer_threshold=float(steer_threshold),
        steer_next_mode=str(steer_next_mode),
    )


def _compare_sds_predictions_on_hidden(
    h: np.ndarray, steer_reas: SDSSteerer, steer_base: SDSSteerer
) -> Dict[str, Any]:
    """Sanity: regime assignment and one-step SDS prediction should differ across fits when models differ."""
    s_r = steer_reas.infer_state(h)
    s_b = steer_base.infer_state(h)
    n_marg_r = steer_reas.marginal_argmax_next(s_r)
    n_marg_b = steer_base.marginal_argmax_next(s_b)
    n_off_r = steer_reas.argmax_offdiag_next(s_r)
    n_off_b = steer_base.argmax_offdiag_next(s_b)
    return {
        "s_reasoning": int(s_r),
        "s_base": int(s_b),
        "marginal_next_reasoning": int(n_marg_r),
        "marginal_next_base": int(n_marg_b),
        "offdiag_next_reasoning": None if n_off_r is None else int(n_off_r),
        "offdiag_next_base": None if n_off_b is None else int(n_off_b),
        "same_assigned_regime": bool(s_r == s_b),
        "same_marginal_next": bool(n_marg_r == n_marg_b),
        "same_offdiag_next": bool(n_off_r == n_off_b),
    }


def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    next_local = torch.multinomial(sorted_probs, num_samples=1)
    return torch.gather(sorted_idx, -1, next_local)


def _generate_one(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    layer_idx: int,
    steerer: Optional[SDSSteerer],
    device: torch.device,
    steer_base_for_diag: Optional[SDSSteerer] = None,
    sds_diag_max_steps: int = 64,
) -> Tuple[str, Dict[str, Any]]:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    eos = tokenizer.eos_token_id
    n_steered = 0
    state_trace: List[int] = []
    generated: List[int] = []
    past_key_values = None
    cur_input = input_ids
    vec_for_step: Optional[np.ndarray] = None

    # Fast path: no steering hook/state inference needed.
    if steerer is None:
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=bool(temperature > 0),
                temperature=(temperature if temperature > 0 else None),
                top_p=(top_p if temperature > 0 else None),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(gen[0, input_ids.shape[1] :], skip_special_tokens=True)
        return text, {"n_steered_steps": 0, "state_trace_prefix": []}

    diag_rows: List[Dict[str, Any]] = []
    n_diag = 0
    same_s = 0
    same_m = 0
    same_o = 0

    for step_i in range(max_new_tokens):
        hook_handle = None
        if vec_for_step is not None:
            vec = torch.from_numpy(vec_for_step).to(device=device)

            def _hook(_module, _inp, out):
                if isinstance(out, tuple):
                    hs = out[0]
                    hs[:, -1, :] = hs[:, -1, :] + vec.to(dtype=hs.dtype)
                    return (hs,) + out[1:]
                out[:, -1, :] = out[:, -1, :] + vec.to(dtype=out.dtype)
                return out

            hook_handle = model.model.layers[layer_idx].register_forward_hook(_hook)

        with torch.no_grad():
            out = model(
                input_ids=cur_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        if hook_handle is not None:
            hook_handle.remove()
            n_steered += 1

        past_key_values = out.past_key_values
        h_last = out.hidden_states[layer_idx + 1][0, -1, :].detach().float().cpu().numpy()

        vec_for_step = None
        s = steerer.infer_state(h_last)
        state_trace.append(int(s))
        t = steerer.choose_target(s)
        if t is not None:
            vec_for_step = steerer.vector_for_state(s)

        if steer_base_for_diag is not None:
            cmp_d = _compare_sds_predictions_on_hidden(h_last, steerer, steer_base_for_diag)
            n_diag += 1
            if cmp_d["same_assigned_regime"]:
                same_s += 1
            if cmp_d["same_marginal_next"]:
                same_m += 1
            if cmp_d["same_offdiag_next"]:
                same_o += 1
            if step_i < int(sds_diag_max_steps):
                cmp_d["step"] = int(step_i)
                diag_rows.append(cmp_d)

        next_tok = _sample_next_token(out.logits[:, -1, :], temperature=temperature, top_p=top_p)
        tok_id = int(next_tok.item())
        generated.append(tok_id)
        cur_input = next_tok
        if eos is not None and tok_id == int(eos):
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    meta: Dict[str, Any] = {
        "n_steered_steps": int(n_steered),
        "state_trace_prefix": state_trace[:64],
    }
    if steer_base_for_diag is not None and n_diag > 0:
        meta["sds_compare_steps_sample"] = diag_rows
        meta["sds_compare_aggregate"] = {
            "n_steps": int(n_diag),
            "frac_same_assigned_regime": float(same_s / n_diag),
            "frac_same_marginal_next": float(same_m / n_diag),
            "frac_same_offdiag_next": float(same_o / n_diag),
        }
    return text, meta


def _load_eval_dataset(name: str) -> Tuple[List[dict], str, str]:
    n = name.lower()
    if n == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [ds[i] for i in range(len(ds))], "question", "answer"
    if n == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        qkey = "problem" if "problem" in ds.column_names else "question"
        akey = "answer" if "answer" in ds.column_names else "solution"
        return [ds[i] for i in range(len(ds))], qkey, akey
    raise ValueError(f"Unsupported dataset: {name}")


def _load_custom_hf_eval_dataset(dataset_id: str, split: str) -> Tuple[List[dict], str, str]:
    ds = load_dataset(dataset_id, split=split)
    cols = set(ds.column_names)
    qkey = "problem" if "problem" in cols else ("question" if "question" in cols else None)
    akey = "answer" if "answer" in cols else ("solution" if "solution" in cols else None)
    if qkey is None or akey is None:
        raise ValueError(
            f"Could not infer question/answer columns from {dataset_id}::{split}. Columns={list(ds.column_names)}"
        )
    return [ds[i] for i in range(len(ds))], qkey, akey


def _is_correct(
    *,
    dataset: str,
    question: str,
    gold: str,
    pred: str,
    use_judge: bool,
    judge_model: str,
    api_key: str,
    openai_judge_primary: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    if use_judge and api_key and openai_judge_primary:
        ok, judge_raw = _judge_correctness(
            api_key=api_key,
            judge_model=judge_model,
            question=question,
            gold_answer=gold,
            model_answer=pred,
        )
        return bool(ok), {"method": "openai_judge_primary", "judge": judge_raw}

    if dataset in {"gsm8k", "custom"}:
        gold_num = _extract_last_number(_gsm8k_gold_answer(gold))
        pred_num = _extract_last_number(pred)
        if gold_num is not None and pred_num is not None:
            return bool(gold_num == pred_num), {"method": "regex_numeric", "gold_num": gold_num, "pred_num": pred_num}
        if use_judge and api_key:
            ok, judge_raw = _judge_correctness(
                api_key=api_key,
                judge_model=judge_model,
                question=question,
                gold_answer=gold,
                model_answer=pred,
            )
            return bool(ok), {
                "method": "openai_judge_fallback_nonnumeric",
                "gold_num": gold_num,
                "pred_num": pred_num,
                "judge": judge_raw,
            }

    if not use_judge or not api_key:
        return bool(_normalize_answer(pred) == _normalize_answer(gold)), {"method": "string_fallback"}

    ok, judge_raw = _judge_correctness(
        api_key=api_key,
        judge_model=judge_model,
        question=question,
        gold_answer=gold,
        model_answer=pred,
    )
    return bool(ok), {"method": "openai_judge", "judge": judge_raw}


def _pass_at_k(rows: Sequence[dict]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([1.0 if any(r["correct_list"]) else 0.0 for r in rows]))


def _merge_sds_compare_aggregates(metas: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    acc: Dict[str, float] = {}
    n_tot = 0
    for m in metas:
        agg = (m or {}).get("sds_compare_aggregate")
        if not agg:
            continue
        n = int(agg.get("n_steps", 0))
        if n <= 0:
            continue
        n_tot += n
        for key in ("frac_same_assigned_regime", "frac_same_marginal_next", "frac_same_offdiag_next"):
            acc[key] = acc.get(key, 0.0) + float(agg[key]) * n
    if n_tot <= 0:
        return None
    return {
        "n_token_steps_total": n_tot,
        "mean_frac_same_assigned_regime": float(acc["frac_same_assigned_regime"] / n_tot),
        "mean_frac_same_marginal_next": float(acc["frac_same_marginal_next"] / n_tot),
        "mean_frac_same_offdiag_next": float(acc["frac_same_offdiag_next"] / n_tot),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SDS-guided steering behavioral evaluation with pass@k.")
    ap.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500"])
    ap.add_argument("--custom-eval-dataset-id", type=str, default="")
    ap.add_argument("--custom-eval-split", type=str, default="")
    ap.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--dataset-repo", type=str, default=SDS_TRAIN_GSM8K_REPO_ID)
    ap.add_argument("--source-relpath", type=str, default=DEFAULT_SOURCE_RELPATH)
    ap.add_argument("--target-relpath", type=str, default=DEFAULT_TARGET_RELPATH)
    ap.add_argument("--artifacts-path", type=str, default="")
    ap.add_argument("--out-root", type=str, default="transplant_sds_artifacts")
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument(
        "--eval-offset",
        type=int,
        default=0,
        help="Skip this many problems from the start of the eval split (e.g. 15 for a fresh slice after an n=50 run).",
    )
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--steer-alpha", type=float, default=1.0)
    ap.add_argument("--steer-threshold", type=float, default=0.25)
    ap.add_argument(
        "--steer-next-mode",
        type=str,
        default="high_margin",
        choices=["high_margin", "argmax_offdiag"],
        help="How reasoning SDS picks a next regime for steering. argmax_offdiag steers more often.",
    )
    ap.add_argument(
        "--base-sds-artifacts",
        type=str,
        default="",
        help="Optional pickle from run_pipeline fit on *base* features (same schema as reasoning artifacts).",
    )
    ap.add_argument(
        "--build-base-sds",
        action="store_true",
        help="If set and --base-sds-artifacts missing, fit SLDS on --target-relpath features into run_dir/base_sds_fit.",
    )
    ap.add_argument("--sds-diag-max-steps", type=int, default=64)
    ap.add_argument("--cebra-epochs", type=int, default=5)
    ap.add_argument("--em-iters", type=int, default=5)
    ap.add_argument("--hf-num-samples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--use-openai-judge", action="store_true")
    ap.add_argument(
        "--openai-judge-primary",
        action="store_true",
        help="If set with --use-openai-judge, grade only via API (no regex short-circuit). More expensive.",
    )
    ap.add_argument("--judge-model", type=str, default="gpt-4.1-mini")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    layer_idx = _infer_layer_from_relpath(args.source_relpath)
    os.makedirs(args.out_root, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    uniq = secrets.token_hex(4)
    dataset_tag = args.dataset
    if args.custom_eval_dataset_id.strip():
        dataset_tag = "custom"
    off = int(args.eval_offset)
    run_name = (
        f"steerpk_{dataset_tag}_l{layer_idx}_o{off}_n{args.n_problems}_k{args.k}_{run_id}_{uniq}"
    )
    run_dir = os.path.join(args.out_root, run_name)
    while os.path.exists(run_dir):
        uniq = secrets.token_hex(4)
        run_name = (
            f"steerpk_{dataset_tag}_l{layer_idx}_o{off}_n{args.n_problems}_k{args.k}_{run_id}_{uniq}"
        )
        run_dir = os.path.join(args.out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    token = args.hf_token or None
    source_path = _download_features(args.dataset_repo, args.source_relpath, token)
    source_features = _load_pickle_list(source_path)

    artifacts_path = args.artifacts_path.strip()
    if artifacts_path and os.path.exists(artifacts_path):
        with open(artifacts_path, "rb") as f:
            payload = pickle.load(f)
    else:
        cfg = SteerConfig(
            data_path=source_path,
            limit_problems=int(args.hf_num_samples),
            cebra_epochs=int(args.cebra_epochs),
            em_iters=int(args.em_iters),
            save_dir=run_dir,
            enable_openai_judge=False,
            hf_auto_download_if_missing=False,
        )
        artifacts_path, payload = run_pipeline(cfg)

    steerer = _build_steerer_from_payload(
        payload=payload,
        source_features=source_features,
        layer=layer_idx,
        alpha=float(args.steer_alpha),
        steer_threshold=float(args.steer_threshold),
        steer_next_mode=str(args.steer_next_mode),
    )

    steer_base: Optional[SDSSteerer] = None
    base_artifacts_path = ""
    b_art = args.base_sds_artifacts.strip()
    need_base_sds = bool(b_art and os.path.exists(b_art)) or bool(args.build_base_sds)
    target_features: List[dict] = []
    base_path = ""
    if need_base_sds:
        base_path = _download_features(args.dataset_repo, args.target_relpath, token)
        target_features = _load_pickle_list(base_path)
    if b_art and os.path.exists(b_art):
        base_artifacts_path = b_art
        with open(base_artifacts_path, "rb") as f:
            payload_base = pickle.load(f)
    elif args.build_base_sds:
        if not base_path:
            base_path = _download_features(args.dataset_repo, args.target_relpath, token)
            target_features = _load_pickle_list(base_path)
        base_fit_dir = os.path.join(run_dir, "base_sds_fit")
        cfg_b = SteerConfig(
            data_path=base_path,
            limit_problems=int(args.hf_num_samples),
            cebra_epochs=int(args.cebra_epochs),
            em_iters=int(args.em_iters),
            save_dir=base_fit_dir,
            enable_openai_judge=False,
            hf_auto_download_if_missing=False,
        )
        base_artifacts_path, payload_base = run_pipeline(cfg_b)
    if base_artifacts_path:
        steer_base = _build_steerer_from_payload(
            payload=payload_base,
            source_features=target_features,
            layer=layer_idx,
            alpha=float(args.steer_alpha),
            steer_threshold=float(args.steer_threshold),
            steer_next_mode=str(args.steer_next_mode),
            include_steering_vectors=False,
        )

    if args.custom_eval_dataset_id.strip():
        eval_rows, q_key, a_key = _load_custom_hf_eval_dataset(
            args.custom_eval_dataset_id.strip(), args.custom_eval_split.strip()
        )
    else:
        eval_rows, q_key, a_key = _load_eval_dataset(args.dataset)
    eval_rows = eval_rows[off : off + int(args.n_problems)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
        trust_remote_code=True,
    ).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    use_judge = bool(args.use_openai_judge and api_key)

    results: List[dict] = []
    for i, item in enumerate(eval_rows):
        q = str(item[q_key])
        gold = str(item[a_key])

        row = {
            "problem_index": i,
            "question": q,
            "gold_answer": gold,
            "baseline": {"samples": [], "correct_list": []},
            "steered": {"samples": [], "correct_list": []},
        }

        for _ in range(int(args.k)):
            b_text, b_meta = _generate_one(
                model=model,
                tokenizer=tok,
                prompt=q,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                layer_idx=int(layer_idx),
                steerer=None,
                device=device,
            )
            b_ok, b_j = _is_correct(
                dataset=("custom" if args.custom_eval_dataset_id.strip() else args.dataset),
                question=q,
                gold=gold,
                pred=b_text,
                use_judge=use_judge,
                judge_model=args.judge_model,
                api_key=api_key,
                openai_judge_primary=bool(args.openai_judge_primary),
            )
            row["baseline"]["samples"].append({"text": b_text, "meta": b_meta, "judge": b_j})
            row["baseline"]["correct_list"].append(bool(b_ok))

            s_text, s_meta = _generate_one(
                model=model,
                tokenizer=tok,
                prompt=q,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                layer_idx=int(layer_idx),
                steerer=steerer,
                device=device,
                steer_base_for_diag=steer_base,
                sds_diag_max_steps=int(args.sds_diag_max_steps),
            )
            s_ok, s_j = _is_correct(
                dataset=("custom" if args.custom_eval_dataset_id.strip() else args.dataset),
                question=q,
                gold=gold,
                pred=s_text,
                use_judge=use_judge,
                judge_model=args.judge_model,
                api_key=api_key,
                openai_judge_primary=bool(args.openai_judge_primary),
            )
            row["steered"]["samples"].append({"text": s_text, "meta": s_meta, "judge": s_j})
            row["steered"]["correct_list"].append(bool(s_ok))

        results.append(row)
        print(
            f"[{i + 1}/{len(eval_rows)}] base_any={any(row['baseline']['correct_list'])} "
            f"steer_any={any(row['steered']['correct_list'])}",
            flush=True,
        )

    baseline_pk = _pass_at_k([{"correct_list": r["baseline"]["correct_list"]} for r in results])
    steered_pk = _pass_at_k([{"correct_list": r["steered"]["correct_list"]} for r in results])
    delta_pk = float(steered_pk - baseline_pk)

    steered_metas_first = [
        r["steered"]["samples"][0]["meta"]
        for r in results
        if r.get("steered", {}).get("samples")
    ]
    n_diff_text = sum(
        1
        for r in results
        if r["baseline"]["samples"]
        and r["steered"]["samples"]
        and r["baseline"]["samples"][0]["text"] != r["steered"]["samples"][0]["text"]
    )
    summary = {
        "dataset": args.dataset,
        "custom_eval_dataset_id": args.custom_eval_dataset_id.strip(),
        "custom_eval_split": args.custom_eval_split.strip(),
        "eval_offset": int(args.eval_offset),
        "steer_next_mode": str(args.steer_next_mode),
        "base_model": args.base_model,
        "dataset_repo": args.dataset_repo,
        "source_relpath": args.source_relpath,
        "target_relpath": args.target_relpath,
        "layer": int(layer_idx),
        "n_problems": int(args.n_problems),
        "k": int(args.k),
        "use_openai_judge": bool(use_judge),
        "openai_judge_primary": bool(use_judge and args.openai_judge_primary),
        "judge_model": args.judge_model if use_judge else "",
        "pass_at_k_baseline": baseline_pk,
        "pass_at_k_steered": steered_pk,
        "delta_pass_at_k": delta_pk,
        "n_problems_text_changed": int(n_diff_text),
        "reasoning_sds_artifacts_path": artifacts_path,
        "base_sds_artifacts_path": base_artifacts_path,
        "sds_reasoning_vs_base_rollup": _merge_sds_compare_aggregates(steered_metas_first),
        "run_dir": run_dir,
    }

    with open(os.path.join(run_dir, "steering_passk_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, "steering_passk_rows.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

