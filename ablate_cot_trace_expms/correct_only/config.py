"""Paths and defaults for correct-trace ablations on SDS HF datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# HF dataset repo id -> benchmark loader key (see grading.DATASET_CFG)
HF_REPOS: dict[str, str] = {
    "gsm8k": "withmartian/SDS_train_gsm8k",
    "svamp": "withmartian/SDS_train_svamp",
    "mmlu-pro": "withmartian/SDS_train_mmlu-pro",
    "math500": "withmartian/SDS_math500_test",
}

# Short model family key -> (base folder, rft folder) under train HF repos
MODEL_PAIRS: dict[str, tuple[str, str]] = {
    "llama8": ("llama_8B_base", "llama_8b_reasoning"),
    "qwen14": ("qwen_14B_base", "qwen_14b_reasoning"),
    "qwen1.5": ("qwen_1.5B_base", "qwen1.5b_reasoning"),
}

# Per-repo folder names (train repos differ from SDS_math500_test layout)
MODEL_PAIRS_BY_REPO: dict[str, dict[str, tuple[str, str]]] = {
    "withmartian/SDS_math500_test": {
        "llama8": ("Llama_8B_base", "Llama_8B_reasoning"),
        "qwen14": ("Qwen_14B_base", "Qwen_14B_reasoning"),
        "qwen1.5": ("Qwen_1_5B_base", "Qwen_1_5B_reasoning"),
    },
    "withmartian/SDS_train_svamp": {
        "llama8": ("Llama_8B_base", "Llama_8B_reasoning"),
        "qwen14": ("Qwen_14B_base", "Qwen_14B_reasoning"),
        "qwen1.5": ("Qwen_1_5B_base", "Qwen_1_5B_reasoning"),
    },
    "withmartian/SDS_train_mmlu-pro": {
        "llama8": ("llama8b_base", "llama8b"),
        "qwen14": ("qwen14b_base", "qwen14b"),
        "qwen1.5": ("qwen1.5b_base", "qwen1.5b"),
    },
}


def model_pair_folders(
    model_family: str,
    *,
    hf_repo: str,
) -> tuple[str, str]:
    pairs = MODEL_PAIRS_BY_REPO.get(hf_repo, MODEL_PAIRS)
    if model_family not in pairs:
        raise KeyError(f"Unknown model_family {model_family!r} for repo {hf_repo!r}")
    return pairs[model_family]

DEFAULT_LAYER: dict[str, int] = {
    "llama8": 22,
    "qwen14": 28,
    "qwen1.5": 20,
}

# Default problem cap (match exploration scripts / paper subsets)
DEFAULT_LIMIT_PROBLEMS: dict[str, int] = {
    "gsm8k": 2000,
    "svamp": 800,
    "mmlu-pro": 500,
    "math500": 500,
}

SUBSET_NAMES = (
    "all",
    "within_correct",
    "paired_correct",
)


@dataclass(frozen=True)
class ModelDatasetPaths:
    dataset_key: str
    model_family: str
    variant: str  # "base" | "rft"
    layer: int
    hf_repo: str
    folder: str
    features_path: Path
    cot_data_path: Path

    @property
    def model_tag(self) -> str:
        return f"{self.model_family}_{self.variant}"


def resolve_paths(
    dataset_key: str,
    model_family: str,
    variant: str,
    *,
    layer: int | None = None,
    hf_root: Path | None = None,
    hf_repo: str | None = None,
) -> ModelDatasetPaths:
    if dataset_key not in HF_REPOS:
        raise KeyError(f"Unknown dataset_key {dataset_key!r}; choose from {list(HF_REPOS)}")
    if model_family not in MODEL_PAIRS:
        raise KeyError(f"Unknown model_family {model_family!r}; choose from {list(MODEL_PAIRS)}")
    if variant not in ("base", "rft"):
        raise ValueError("variant must be 'base' or 'rft'")

    layer = layer if layer is not None else DEFAULT_LAYER[model_family]
    repo = hf_repo or HF_REPOS[dataset_key]
    base_folder, rft_folder = model_pair_folders(model_family, hf_repo=repo)
    folder = base_folder if variant == "base" else rft_folder
    layer_dir = f"layer_{layer}"
    root = hf_root if hf_root is not None else Path(repo.replace("/", "__"))

    base = root / folder / layer_dir
    return ModelDatasetPaths(
        dataset_key=dataset_key,
        model_family=model_family,
        variant=variant,
        layer=layer,
        hf_repo=repo,
        folder=folder,
        features_path=base / "all_sentences_features.pkl",
        cot_data_path=base / "cot_data.pkl",
    )


def all_combo_keys() -> list[tuple[str, str]]:
    return [(d, m) for d in HF_REPOS for m in MODEL_PAIRS]
