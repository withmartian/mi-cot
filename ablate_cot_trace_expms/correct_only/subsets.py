"""Select problem_id sets for correct-trace ablation subsets."""

from __future__ import annotations


def pids_within_correct(manifest: dict) -> set[int]:
    return {r["problem_id"] for r in manifest["records"] if r["is_correct"]}


def pids_all(manifest: dict, limit_problems: int) -> set[int]:
    return {r["problem_id"] for r in manifest["records"] if r["problem_id"] < limit_problems}


def pids_paired_correct(base_manifest: dict, rft_manifest: dict) -> set[int]:
    b = pids_within_correct(base_manifest)
    r = pids_within_correct(rft_manifest)
    return b & r


def resolve_subset_pids(
    subset: str,
    *,
    manifest: dict,
    limit_problems: int,
    base_manifest: dict | None = None,
    rft_manifest: dict | None = None,
) -> set[int]:
    if subset == "all":
        return pids_all(manifest, limit_problems)
    if subset == "within_correct":
        return pids_within_correct(manifest)
    if subset == "paired_correct":
        if base_manifest is None or rft_manifest is None:
            raise ValueError("paired_correct requires base_manifest and rft_manifest")
        return pids_paired_correct(base_manifest, rft_manifest)
    raise ValueError(f"Unknown subset {subset!r}; use all, within_correct, or paired_correct")
