"""Download SDS artifacts from Hugging Face dataset repos."""

from __future__ import annotations

from pathlib import Path


def ensure_hf_file(
    repo_id: str,
    filename: str,
    local_dir: Path | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir) if local_dir else None,
    )
    return Path(path)


def ensure_model_artifacts(
    paths,
    *,
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Return (features_path, cot_data_path), downloading from HF if missing locally."""
    features = paths.features_path
    cot = paths.cot_data_path
    if features.is_file() and cot.is_file():
        return features, cot

    layer_dir = f"layer_{paths.layer}"
    prefix = f"{paths.folder}/{layer_dir}"
    repo = paths.hf_repo
    root = cache_dir or Path("hf_cache") / repo.replace("/", "__")

    if not cot.is_file():
        cot = ensure_hf_file(repo, f"{prefix}/cot_data.pkl", local_dir=root)
    if not features.is_file():
        features = ensure_hf_file(repo, f"{prefix}/all_sentences_features.pkl", local_dir=root)
    return Path(features), Path(cot)
