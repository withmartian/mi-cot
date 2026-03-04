#!/usr/bin/env python3
"""
Transfer files from one Hugging Face Hub repository to another.

Requires: pip install huggingface_hub

Authentication:
  - Run `huggingface-cli login` or set HF_TOKEN in the environment.
  - Your token must have read access to the source repo and write access to the destination.
"""

import argparse
import tempfile
import shutil
from pathlib import Path

from huggingface_hub import (
    HfApi,
    snapshot_download,
)


def transfer_repo(
    source_repo: str,
    dest_repo: str,
    repo_type: str = "model",
    source_revision: str = None,
    dest_revision: str = "main",
    path_in_repo: str = None,
    create_dest: bool = True,
    token: str = None,
    allow_patterns: list = None,
    ignore_patterns: list = None,
):
    """
    Download all files from source_repo and upload them to dest_repo.

    Args:
        source_repo: "username/repo-name" of the source repo.
        dest_repo: "username/repo-name" of the destination repo.
        repo_type: "model", "dataset", or "space".
        source_revision: Branch/tag/commit of source (default: main).
        dest_revision: Branch to push to (default: main).
        path_in_repo: Subfolder in dest repo to upload into (e.g. "models/v1"). None = repo root.
        create_dest: If True, create destination repo if it doesn't exist.
        token: HF token (default: from HF_TOKEN env or huggingface-cli login).
        allow_patterns: Only transfer files matching these globs (e.g. ["*.safetensors"]).
        ignore_patterns: Skip files matching these globs.
    """
    api = HfApi(token=token)
    source_revision = source_revision or "main"

    with tempfile.TemporaryDirectory(prefix="hf_transfer_") as tmpdir:
        local_path = Path(tmpdir) / "repo"
        print(f"Downloading {source_repo} ({repo_type}) -> {local_path}")
        snapshot_download(
            repo_id=source_repo,
            repo_type=repo_type,
            revision=source_revision,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        if create_dest:
            try:
                api.create_repo(
                    repo_id=dest_repo,
                    repo_type=repo_type,
                    exist_ok=True,
                )
                print(f"Destination repo {dest_repo} ready (created or already exists).")
            except Exception as e:
                print(f"Note: create_repo: {e}")

        dest_path_msg = f" -> {path_in_repo}/" if path_in_repo else ""
        print(f"Uploading {local_path} -> {dest_repo}{dest_path_msg} ({repo_type})")
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=dest_repo,
            repo_type=repo_type,
            revision=dest_revision,
            path_in_repo=path_in_repo,
            token=token,
            commit_message=f"Transfer from {source_repo}",
        )

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer files from one Hugging Face repo to another."
    )
    parser.add_argument(
        "source_repo",
        help='Source repo id, e.g. "username/source-repo"',
    )
    parser.add_argument(
        "dest_repo",
        help='Destination repo id, e.g. "other-user/dest-repo"',
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="model",
        help="Repo type (default: model)",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Source branch/tag/commit (default: main)",
    )
    parser.add_argument(
        "--dest-revision",
        default="main",
        help="Destination branch to push to (default: main)",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        metavar="PATH",
        help="Subfolder in destination repo to upload into (e.g. models/v1). Default: repo root",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Do not create destination repo if it does not exist",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token (default: HF_TOKEN env or huggingface-cli login)",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="+",
        default=None,
        help="Only transfer files matching these globs, e.g. --allow-patterns '*.safetensors' '*.json'",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="+",
        default=None,
        help="Skip files matching these globs",
    )
    args = parser.parse_args()

    transfer_repo(
        source_repo=args.source_repo,
        dest_repo=args.dest_repo,
        repo_type=args.repo_type,
        source_revision=args.source_revision,
        dest_revision=args.dest_revision,
        path_in_repo=args.path_in_repo,
        create_dest=not args.no_create,
        token=args.token,
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )


if __name__ == "__main__":
    main()
