"""Grade CoT traces against benchmark ground truth (self-contained, no external imports)."""

from __future__ import annotations

import re
from typing import Any

DATASET_CFG: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "type": "gsm8k",
    },
    "svamp": {
        "hf_id": "garrethlee/svamp",
        "config": "default",
        "split": "train",
        "type": "svamp",
    },
    "mmlu-pro": {
        "hf_id": "TIGER-Lab/MMLU-Pro",
        "config": "default",
        "split": "test",
        "type": "mmlu_pro",
    },
    "math500": {
        "hf_id": "HuggingFaceH4/MATH-500",
        "config": "default",
        "split": "test",
        "type": "math",
    },
}


def _normalize_number(s: str) -> str:
    s = str(s).strip().replace(",", "")
    s = re.sub(r"^\$|\$$", "", s)
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return str(v)
    except ValueError:
        return s.strip().lower()


def _extract_answer_from_cot(cot: str, kind: str) -> str:
    text = cot.strip()
    if kind == "gsm8k":
        matches = re.findall(r"####\s*([\-\d,\.]+)", text)
        if matches:
            return _normalize_number(matches[-1])
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        return _normalize_number(nums[-1]) if nums else ""
    if kind == "mmlu_pro":
        for line in reversed(text.splitlines()):
            line = line.strip()
            m = re.search(r"\b([A-J])\b\s*$", line, re.I)
            if m:
                return m.group(1).upper()
            m = re.search(r"(?:answer|option)\s*[:is]?\s*([A-J])\b", line, re.I)
            if m:
                return m.group(1).upper()
        m = re.search(r"\b([A-J])\b", text[-200:], re.I)
        return m.group(1).upper() if m else ""
    if kind == "math":
        boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxed:
            inner = boxed[-1]
            inner = re.sub(r"\\[a-zA-Z]+", "", inner)
            inner = inner.replace("{", "").replace("}", "").strip()
            return _normalize_number(inner) if inner else ""
        nums = re.findall(r"[-+]?\d*\.?\d+", text[-400:])
        return _normalize_number(nums[-1]) if nums else ""
    # svamp / generic numeric
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return _normalize_number(nums[-1]) if nums else ""


def is_correct_answer(predicted: str, ground_truth: str, kind: str) -> bool:
    pred = (predicted or "").strip()
    gold = (ground_truth or "").strip()
    if not pred or not gold:
        return False
    if kind == "mmlu_pro":
        return pred.upper() == gold.upper()
    if kind in ("gsm8k", "svamp", "math"):
        return _normalize_number(pred) == _normalize_number(gold)
    return pred.lower() == gold.lower()


def ground_truth_from_row(row: dict, kind: str) -> str:
    if kind == "svamp":
        return str(row.get("Answer", row.get("answer", ""))).strip()
    if kind == "mmlu_pro":
        return str(row.get("answer", "")).strip().upper()
    if kind == "gsm8k":
        ans = row.get("answer", "")
        m = re.search(r"####\s*([\-\d,\.]+)", str(ans))
        if m:
            return _normalize_number(m.group(1))
        return str(ans).strip()
    return str(row.get("answer", row.get("solution", ""))).strip()


def load_benchmark_rows(
    dataset_key: str,
    *,
    limit: int | None = None,
    hf_cache_dir: str | None = None,
) -> dict[int, str]:
    """problem_id -> ground_truth string (aligned with create_dataset row order)."""
    from datasets import load_dataset

    cfg = DATASET_CFG[dataset_key]
    base_split = cfg["split"].split("[")[0]
    kwargs: dict[str, Any] = {
        "split": f"{base_split}[:{limit}]" if limit is not None else base_split,
    }
    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir

    if cfg.get("config"):
        ds = load_dataset(cfg["hf_id"], cfg["config"], **kwargs)
    else:
        ds = load_dataset(cfg["hf_id"], **kwargs)

    kind = cfg["type"]
    out: dict[int, str] = {}
    for i, row in enumerate(ds):
        out[i] = ground_truth_from_row(row, kind)
    return out
