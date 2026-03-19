"""
Shared Hugging Face layout for the **SDS_train_gsm8k** activation dataset.

Repo: https://huggingface.co/datasets/withmartian/SDS_train_gsm8k

Used by ``cebra_em_steering_inputDep`` (single runs) and
``batch_cebra_steering_hf_dataset`` (sweeps).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Canonical Hub dataset id (org/name on Hugging Face Hub).
SDS_TRAIN_GSM8K_REPO_ID = "withmartian/SDS_train_gsm8k"
SDS_TRAIN_FEATURES_FILENAME = "all_sentences_features.pkl"

# Public HF model checkpoints ↔ top-level paths inside the dataset repo.
HF_DATASET_MODEL_PATH_ROWS: List[Dict[str, str]] = [
    {
        "family": "qwen_1.5b",
        "role": "reasoning",
        "hf_model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "dataset_path_prefix": "qwen1.5b_reasoning/",
    },
    {
        "family": "qwen_1.5b",
        "role": "base",
        "hf_model_id": "Qwen/Qwen2.5-1.5B",
        "dataset_path_prefix": "qwen_1.5B_base/",
    },
    {
        "family": "llama_8b",
        "role": "reasoning",
        "hf_model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "dataset_path_prefix": "llama_8b_reasoning/",
    },
    {
        "family": "llama_8b",
        "role": "base",
        "hf_model_id": "meta-llama/Llama-3.1-8B",
        "dataset_path_prefix": "llama_8B_base/",
    },
    {
        "family": "qwen_14b",
        "role": "reasoning",
        "hf_model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "dataset_path_prefix": "qwen_14b_reasoning/",
    },
    {
        "family": "qwen_14b",
        "role": "base",
        "hf_model_id": "Qwen/Qwen2.5-14B",
        "dataset_path_prefix": "qwen_14B_base/",
    },
]

DEFAULT_LAYER_BY_FAMILY: Dict[str, str] = {
    "qwen_1.5b": "layer_27",
    "llama_8b": "layer_31",
    "qwen_14b": "layer_47",
}

HF_DATASET_CORE_PATH_PREFIXES: Tuple[str, ...] = tuple(
    row["dataset_path_prefix"] for row in HF_DATASET_MODEL_PATH_ROWS
)


def default_hub_features_relpath(
    family: str = "qwen_1.5b",
    role: str = "reasoning",
) -> str:
    """Relative path inside the dataset repo to a default features pickle."""
    layer = DEFAULT_LAYER_BY_FAMILY[family]
    prefix = next(
        r["dataset_path_prefix"].rstrip("/")
        for r in HF_DATASET_MODEL_PATH_ROWS
        if r["family"] == family and r["role"] == role
    )
    return f"{prefix}/{layer}/{SDS_TRAIN_FEATURES_FILENAME}"


DEFAULT_STEERING_HF_FEATURES_RELPATH = default_hub_features_relpath("qwen_1.5b", "reasoning")


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "", s.lower())


def _build_model_preset_map() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for row in HF_DATASET_MODEL_PATH_ROWS:
        fam = row["family"]
        role = row["role"]
        layer = DEFAULT_LAYER_BY_FAMILY[fam]
        prefix = row["dataset_path_prefix"].rstrip("/")
        sub = f"{prefix}/{layer}"
        hf_id = row["hf_model_id"]
        m[_norm_key(hf_id)] = sub
        m[_norm_key(hf_id.split("/")[-1])] = sub
        m[_norm_key(f"{fam}-{role}")] = sub
        m[_norm_key(f"{fam}_{role}")] = sub
    # Shorthands (published tree uses qwen1.5b_reasoning/, not qwen2.5_* hub paths)
    m[_norm_key("qwen2.5-1.5b-instruct")] = m[_norm_key("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")]
    m[_norm_key("qwen2.5_1.5b_reasoning")] = m[_norm_key("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")]
    m[_norm_key("qwen1.5b-reasoning")] = m[_norm_key("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")]
    m[_norm_key("qwen1.5b_reasoning")] = m[_norm_key("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")]
    m[_norm_key("qwen-1.5b-base")] = m[_norm_key("Qwen/Qwen2.5-1.5B")]
    m[_norm_key("llama-8b-reasoning")] = m[_norm_key("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")]
    m[_norm_key("llama-8b-base")] = m[_norm_key("meta-llama/Llama-3.1-8B")]
    m[_norm_key("qwen-14b-reasoning")] = m[_norm_key("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")]
    m[_norm_key("qwen-14b-base")] = m[_norm_key("Qwen/Qwen2.5-14B")]
    return m


_MODEL_PRESET_MAP: Dict[str, str] = _build_model_preset_map()


def expand_steering_model_argument(model: str) -> str:
    """
    Turn a CLI ``--model`` value into a dataset-relative path prefix or full ``*.pkl`` path.
    """
    raw = model.strip()
    if not raw:
        return raw
    if raw.lower().endswith(".pkl"):
        return raw
    if "layer_" in raw.replace("\\", "/"):
        return raw.strip().strip("/")
    mapped = _MODEL_PRESET_MAP.get(_norm_key(raw))
    if mapped is not None:
        return mapped
    return raw


def normalize_hub_features_filename(model_or_path: str) -> str:
    """Path inside the Hub dataset repo to the features pickle."""
    m = expand_steering_model_argument(model_or_path)
    m = m.strip().strip("/")
    if m.lower().endswith(".pkl"):
        return m
    return f"{m}/{SDS_TRAIN_FEATURES_FILENAME}"


def hub_path_matches_core_prefix(hub_filename: str) -> bool:
    """True if path starts with one of the six base+reasoning subtree prefixes in ``HF_DATASET_CORE_PATH_PREFIXES``."""
    p = hub_filename.replace("\\", "/")
    return any(p.startswith(pref) for pref in HF_DATASET_CORE_PATH_PREFIXES)
