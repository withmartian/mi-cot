"""Build, cache, and load per-model correctness manifests (no activation duplication)."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

from grading import DATASET_CFG, _extract_answer_from_cot, is_correct_answer, load_benchmark_rows

DEFAULT_MANIFEST_CACHE_DIR = Path(__file__).resolve().parent / "manifest_cache"


def add_manifest_cache_args(parser) -> None:
    """Register CLI flags shared by run_single_model and run_compare_base_rft."""
    parser.add_argument(
        "--manifest-cache-dir",
        type=Path,
        default=DEFAULT_MANIFEST_CACHE_DIR,
        help="Directory for reusable correctness manifests (JSON, no activations).",
    )
    parser.add_argument(
        "--use-cached-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load manifest from cache when cot_data fingerprint matches (default: true).",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Force re-grade cot_data and overwrite cached manifest.",
    )


def trajectory_length(cot_entry: dict) -> int:
    sents = cot_entry.get("sentences") or []
    if sents:
        return len(sents)
    cot = cot_entry.get("cot") or ""
    parts = [p for p in cot.replace("\n", " ").split(". ") if len(p.strip()) > 10]
    return max(len(parts), 1)


def cot_data_fingerprint(cot_data_path: Path) -> str:
    """SHA-256 of cot_data.pkl; cache invalidates when the file changes."""
    h = hashlib.sha256()
    with open(cot_data_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_cache_path(
    cache_dir: Path,
    *,
    dataset_key: str,
    model_folder: str,
    variant: str,
    layer: int,
    limit_problems: int,
    cot_data_path: Path,
) -> Path:
    fp = cot_data_fingerprint(cot_data_path)
    name = (
        f"{dataset_key}__{model_folder}__{variant}__layer{layer}"
        f"__n{limit_problems}__{fp[:16]}.json"
    )
    return cache_dir / name


def _manifest_matches_cache(
    manifest: dict[str, Any],
    *,
    dataset_key: str,
    limit_problems: int,
    cot_data_path: Path,
    model_folder: str,
    variant: str,
    layer: int,
) -> bool:
    fp = cot_data_fingerprint(cot_data_path)
    return (
        manifest.get("dataset_key") == dataset_key
        and manifest.get("limit_problems") == limit_problems
        and manifest.get("cot_data_fingerprint") == fp
        and manifest.get("cot_data_path") == str(cot_data_path.resolve())
        and manifest.get("model_folder") == model_folder
        and manifest.get("variant") == variant
        and manifest.get("layer") == layer
        and "records" in manifest
    )


def build_manifest(
    cot_data_path: Path,
    dataset_key: str,
    *,
    limit_problems: int,
    hf_cache_dir: str | None = None,
    model_folder: str | None = None,
    variant: str | None = None,
    layer: int | None = None,
) -> dict[str, Any]:
    cot_data_path = cot_data_path.resolve()
    with open(cot_data_path, "rb") as f:
        cot_data: dict = pickle.load(f)

    gold = load_benchmark_rows(dataset_key, limit=limit_problems, hf_cache_dir=hf_cache_dir)
    kind = DATASET_CFG[dataset_key]["type"]

    records = []
    n_correct = 0
    for pid in sorted(cot_data.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
        pid_int = int(pid)
        if pid_int >= limit_problems:
            continue
        entry = cot_data[pid]
        cot = entry.get("cot", "")
        pred = _extract_answer_from_cot(cot, kind)
        gt = gold.get(pid_int, "")
        ok = is_correct_answer(pred, gt, kind)
        if ok:
            n_correct += 1
        records.append(
            {
                "problem_id": pid_int,
                "is_correct": ok,
                "T": trajectory_length(entry),
                "extracted_answer": pred,
                "ground_truth": gt,
            }
        )

    manifest: dict[str, Any] = {
        "dataset_key": dataset_key,
        "cot_data_path": str(cot_data_path),
        "cot_data_fingerprint": cot_data_fingerprint(cot_data_path),
        "limit_problems": limit_problems,
        "n_problems": len(records),
        "n_correct": n_correct,
        "accuracy": n_correct / len(records) if records else 0.0,
        "records": records,
    }
    if model_folder is not None:
        manifest["model_folder"] = model_folder
    if variant is not None:
        manifest["variant"] = variant
    if layer is not None:
        manifest["layer"] = layer
    return manifest


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_or_build_manifest(
    cot_data_path: Path,
    dataset_key: str,
    *,
    limit_problems: int,
    model_folder: str,
    variant: str,
    layer: int,
    cache_dir: Path | None = None,
    hf_cache_dir: str | None = None,
    use_cached: bool = True,
    rebuild: bool = False,
) -> tuple[dict[str, Any], Path, bool]:
    """
    Load a cached correctness manifest or build and save one.

    Returns (manifest, cache_path, from_cache).
  """
    cache_dir = cache_dir or DEFAULT_MANIFEST_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cot_data_path = cot_data_path.resolve()
    cache_path = manifest_cache_path(
        cache_dir,
        dataset_key=dataset_key,
        model_folder=model_folder,
        variant=variant,
        layer=layer,
        limit_problems=limit_problems,
        cot_data_path=cot_data_path,
    )

    if use_cached and not rebuild and cache_path.is_file():
        manifest = load_manifest(cache_path)
        if _manifest_matches_cache(
            manifest,
            dataset_key=dataset_key,
            limit_problems=limit_problems,
            cot_data_path=cot_data_path,
            model_folder=model_folder,
            variant=variant,
            layer=layer,
        ):
            return manifest, cache_path, True
        print(
            f"  Manifest cache stale or mismatched ({cache_path.name}); rebuilding.",
            flush=True,
        )

    manifest = build_manifest(
        cot_data_path,
        dataset_key,
        limit_problems=limit_problems,
        hf_cache_dir=hf_cache_dir,
        model_folder=model_folder,
        variant=variant,
        layer=layer,
    )
    save_manifest(manifest, cache_path)
    return manifest, cache_path, False


def copy_manifest_to_run_dir(cache_path: Path, run_path: Path) -> None:
    """Copy cached manifest into a run folder for reproducibility."""
    run_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_path, run_path)


def records_by_pid(manifest: dict[str, Any]) -> dict[int, dict]:
    return {r["problem_id"]: r for r in manifest["records"]}


def get_or_build_paired_index(
    base_manifest: dict[str, Any],
    rft_manifest: dict[str, Any],
    *,
    dataset_key: str,
    model_family: str,
    layer: int,
    limit_problems: int,
    cache_dir: Path | None = None,
    use_cached: bool = True,
    rebuild: bool = False,
) -> tuple[list[int], Path | None, bool]:
    """
    Cache the list of problem_ids where both models are correct.

    Returns (problem_ids, cache_path or None, from_cache).
    """
    from subsets import pids_paired_correct

    cache_dir = cache_dir or DEFAULT_MANIFEST_CACHE_DIR
    paired_dir = cache_dir / "paired"
    paired_dir.mkdir(parents=True, exist_ok=True)

    base_fp = base_manifest["cot_data_fingerprint"][:8]
    rft_fp = rft_manifest["cot_data_fingerprint"][:8]
    cache_path = paired_dir / (
        f"{dataset_key}__{model_family}__layer{layer}__n{limit_problems}"
        f"__base_{base_fp}__rft_{rft_fp}.json"
    )

    if use_cached and not rebuild and cache_path.is_file():
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("base_fingerprint", "").startswith(base_fp) and data.get(
            "rft_fingerprint", ""
        ).startswith(rft_fp):
            return data["problem_ids"], cache_path, True

    pids = sorted(pids_paired_correct(base_manifest, rft_manifest))
    payload = {
        "dataset_key": dataset_key,
        "model_family": model_family,
        "layer": layer,
        "limit_problems": limit_problems,
        "base_fingerprint": base_manifest["cot_data_fingerprint"],
        "rft_fingerprint": rft_manifest["cot_data_fingerprint"],
        "n_paired": len(pids),
        "problem_ids": pids,
    }
    save_manifest(payload, cache_path)
    return pids, cache_path, False
