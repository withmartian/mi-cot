#!/usr/bin/env python3
"""
Rename a folder (path) in a Hugging Face Hub repository.

Moves all files from old_path to new_path in the same repo:
- Lists files under old_path
- Downloads each, uploads to new_path, deletes the original

Requires: pip install huggingface_hub

Authentication:
  - Run `huggingface-cli login` or set HF_TOKEN in the environment.
  - Your token must have write access to the repo.
"""

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def _norm_path(p: str) -> str:
    """Normalize path: no leading slash, ensure trailing slash for directory."""
    p = p.strip("/").replace("\\", "/")
    return p + "/" if p else ""


def rename_folder(
    repo_id: str,
    old_path: str,
    new_path: str,
    repo_type: str = "model",
    revision: str = "main",
    token: str = None,
    dry_run: bool = False,
):
    """
    Move all files from old_path to new_path inside the same repo.

    Args:
        repo_id: "username/repo-name".
        old_path: Current folder path in the repo (e.g. "data/raw" or "data/raw/").
        new_path: Target folder path (e.g. "data/processed").
        repo_type: "model", "dataset", or "space".
        revision: Branch to work on (default: main).
        token: HF token (default: HF_TOKEN env or huggingface-cli login).
        dry_run: If True, only print what would be moved; do not upload or delete.
    """
    api = HfApi(token=token)
    old_prefix = _norm_path(old_path)
    new_prefix = _norm_path(new_path)

    if old_prefix == new_prefix:
        print("Old and new path are the same. Nothing to do.")
        return

    all_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    to_move = [f for f in all_files if f == old_prefix.rstrip("/") or f.startswith(old_prefix)]

    if not to_move:
        print(f"No files found under '{old_path}'. Check the path and repo_type.")
        return

    print(f"Found {len(to_move)} file(s) under '{old_path}' to move to '{new_path}'.")
    if dry_run:
        for f in to_move:
            rel = f[len(old_prefix):] if f.startswith(old_prefix) else ""
            print(f"  {f} -> {new_prefix.rstrip('/') + '/' + rel}")
        return

    with tempfile.TemporaryDirectory(prefix="hf_rename_") as tmpdir:
        for path_in_repo in to_move:
            if path_in_repo.endswith("/"):
                continue
            if path_in_repo.startswith(old_prefix):
                rel = path_in_repo[len(old_prefix):]
            else:
                rel = Path(path_in_repo).name
            new_file_path = (new_prefix + rel).rstrip("/")

            print(f"  {path_in_repo} -> {new_file_path}")
            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type=repo_type,
                revision=revision,
                token=token,
                local_dir=tmpdir,
                local_dir_use_symlinks=False,
                force_download=True,
            )
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo=new_file_path,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
                commit_message=f"Move {path_in_repo} to {new_file_path}",
            )
            api.delete_file(
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
                commit_message=f"Remove old path {path_in_repo}",
            )

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Rename (move) a folder in a Hugging Face repo.",
    )
    parser.add_argument(
        "repo_id",
        help='Repo id, e.g. "username/repo-name"',
    )
    parser.add_argument(
        "old_path",
        help='Current folder path in the repo (e.g. data/raw)',
    )
    parser.add_argument(
        "new_path",
        help='Target folder path (e.g. data/processed)',
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="model",
        help="Repo type (default: model)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch to work on (default: main)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (default: HF_TOKEN env or huggingface-cli login)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list what would be moved; do not upload or delete",
    )
    args = parser.parse_args()

    rename_folder(
        repo_id=args.repo_id,
        old_path=args.old_path,
        new_path=args.new_path,
        repo_type=args.repo_type,
        revision=args.revision,
        token=args.token,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
