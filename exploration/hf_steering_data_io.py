"""
Hugging Face dataset resolution for steering pipelines (not steering math itself).

- Parse Hub repo ids / URLs vs local paths
- Download a features pickle, subset non-NEUTRAL rows, cache locally
- CLI helpers: ``add_hf_dataset_cli_args``, ``apply_hf_dataset_cli_config``

Layout presets for SDS_train_gsm8k live in ``sds_train_gsm8k_hf``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from typing import Any, List, Protocol, Tuple

from sds_train_gsm8k_hf import normalize_hub_features_filename


class _SteeringDataSourceCfg(Protocol):
    """Minimal config surface for HF download / subset (structurally satisfied by ``SteerConfig``)."""

    data_path: str
    limit_problems: int
    save_dir: str
    hf_auto_download_if_missing: bool
    hf_dataset_repo: str
    hf_dataset_filename: str
    hf_fallback_num_samples: int
    hf_fallback_cache_name: str


def parse_hf_dataset_repo(value: str) -> str:
    """
    Accepts:
      - org/name
      - https://huggingface.co/datasets/org/name
      - https://huggingface.co/datasets/org/name/tree/main
      - hf://org/name
    """
    v = value.strip().rstrip("/")
    if not v:
        raise ValueError("Empty --dataset value.")
    if v.lower().startswith("hf://"):
        return v[5:].strip().strip("/")
    if "huggingface.co/datasets/" in v:
        after = v.split("huggingface.co/datasets/", 1)[1]
        after = after.split("/tree/")[0].split("/blob/")[0].split("?")[0]
        return after.strip("/")
    return v


def is_probably_hf_dataset_ref(s: str) -> bool:
    """True for Hub repo id, hf:// URL, or huggingface.co/datasets/... URL (not a local path)."""
    t = s.strip()
    if not t:
        return False
    if t.lower().startswith("hf://"):
        return True
    if "huggingface.co/datasets/" in t:
        return True
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", t):
        return True
    return False


def resolve_cli_data_path(dataset_arg: str) -> str:
    """
    Resolve ``--dataset`` when treated as local: file, or directory containing
    ``all_sentences_features.pkl``; else abspath of the argument (may not exist yet).
    """
    raw = dataset_arg.strip()
    p = os.path.abspath(os.path.expanduser(raw))
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        cand = os.path.join(p, "all_sentences_features.pkl")
        if os.path.isfile(cand):
            return cand
    return p


def safe_hf_cache_slug(model_path: str, max_len: int = 72) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_path.strip().strip("/"))
    return s[:max_len] if s else "default"


def resolve_hf_subset_data_path(cfg: _SteeringDataSourceCfg) -> Tuple[str, int]:
    """
    Return ``(absolute path to features pickle, limit_problems)`` for ``load_and_prepare_cebra``.

    If ``cfg.data_path`` exists as a file, return it with ``cfg.limit_problems``.
    Otherwise, when ``cfg.hf_auto_download_if_missing``, download from Hub, keep the first
    ``hf_fallback_num_samples`` non-NEUTRAL rows, write under ``cfg.save_dir``, and return
    ``limit_problems = max(problem_id)+1`` over the subset.
    """
    path = os.path.abspath(os.path.expanduser(cfg.data_path))
    if os.path.isfile(path):
        return path, cfg.limit_problems

    if not cfg.hf_auto_download_if_missing:
        raise FileNotFoundError(
            f"Steering data not found: {path!r}. "
            "Point data_path to an existing pickle, or set hf_auto_download_if_missing=True."
        )

    token = os.environ.get("HF_TOKEN", "").strip() or None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for HF fallback. Install with: pip install huggingface_hub"
        ) from e

    os.makedirs(cfg.save_dir, exist_ok=True)
    print(
        f"[data] Local file not found ({path!r}). Downloading from Hugging Face "
        f"{cfg.hf_dataset_repo} / {cfg.hf_dataset_filename!r} …",
        flush=True,
    )
    try:
        remote = hf_hub_download(
            repo_id=cfg.hf_dataset_repo,
            filename=cfg.hf_dataset_filename,
            token=token,
            repo_type="dataset",
        )
    except Exception as e:
        hint = (
            " If the repo is gated, set HF_TOKEN (env var only)."
            if token is None
            else ""
        )
        raise FileNotFoundError(
            f"Could not download {cfg.hf_dataset_repo!r} / {cfg.hf_dataset_filename!r}:{hint}"
        ) from e
    with open(remote, "rb") as f:
        features: List[dict] = pickle.load(f)

    features = [feat for feat in features if feat.get("stage", "NEUTRAL") != "NEUTRAL"]
    n = cfg.hf_fallback_num_samples
    if len(features) < n:
        raise ValueError(
            f"Only {len(features)} non-NEUTRAL samples after filter; need >= {n}. "
            "Try another hf_dataset_filename or smaller hf_fallback_num_samples."
        )
    subset = features[:n]
    limit_problems = max(int(f["problem_id"]) for f in subset) + 1

    cache_path = os.path.join(cfg.save_dir, cfg.hf_fallback_cache_name)
    with open(cache_path, "wb") as f:
        pickle.dump(subset, f)

    print(
        f"[data] Cached {len(subset)} samples to {cache_path} (limit_problems={limit_problems}).",
        flush=True,
    )
    return os.path.abspath(cache_path), limit_problems


def add_hf_dataset_cli_args(p: argparse.ArgumentParser) -> None:
    """Register CLI flags for Hub repo, local path, model subpath, and subset size."""
    p.add_argument(
        "--local-data",
        type=str,
        default=None,
        metavar="PATH",
        help="Local features .pkl (e.g. all_sentences_features.pkl). If it exists, used directly.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="REPO_OR_URL_OR_PATH",
        help=(
            "Hugging Face dataset repo id (org/name), full datasets URL, hf://org/name, "
            "or a local .pkl path / directory containing all_sentences_features.pkl."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="SUBPATH",
        help=(
            "Path inside the Hub dataset, e.g. qwen1.5b_reasoning/layer_27, or a preset from "
            "sds_train_gsm8k_hf (HF model id, shorthands, …)."
        ),
    )
    p.add_argument(
        "--hf-samples",
        "--num-samples",
        type=int,
        default=None,
        dest="hf_samples",
        metavar="N",
        help="When downloading from Hub, keep this many non-NEUTRAL rows (alias: --num-samples).",
    )
    p.add_argument(
        "--no-hf-fallback",
        action="store_true",
        help="If local data_path is missing, fail instead of downloading from Hugging Face.",
    )


def apply_hf_dataset_cli_config(cfg: Any, args: argparse.Namespace) -> Any:
    """
    Merge HF-related CLI flags into a config object (e.g. ``SteerConfig``).
    Uses ``dataclasses.replace``; cfg must be a dataclass instance.
    """
    from dataclasses import replace

    c = cfg

    if args.hf_samples is not None:
        c = replace(c, hf_fallback_num_samples=int(args.hf_samples))

    if args.local_data:
        c = replace(c, data_path=os.path.abspath(os.path.expanduser(args.local_data.strip())))
    elif args.dataset:
        if is_probably_hf_dataset_ref(args.dataset):
            c = replace(c, hf_dataset_repo=parse_hf_dataset_repo(args.dataset))
        else:
            c = replace(c, data_path=resolve_cli_data_path(args.dataset))

    if args.model:
        rel = normalize_hub_features_filename(args.model)
        c = replace(c, hf_dataset_filename=rel)
        slug = safe_hf_cache_slug(rel.replace(".pkl", ""))
        c = replace(
            c,
            hf_fallback_cache_name=f"sds_hf{c.hf_fallback_num_samples}_{slug}.pkl",
        )

    return c
