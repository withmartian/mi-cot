#!/usr/bin/env python3
"""
Upload create_dataset output (e.g. gsm8k_output) to a Hugging Face dataset repo.

Usage:
    python upload_to_hf.py
    python upload_to_hf.py --folder ./gsm8k_output --repo [yourname]/[yourrepo] --path-in-repo layer_20 

Requires: pip install huggingface_hub, then login via:
    python -c "from huggingface_hub import login; login()"
or set HF_TOKEN in the environment.
"""

import argparse
import os


def main():
    p = argparse.ArgumentParser(description="Upload dataset output to Hugging Face")
    p.add_argument(
        "--folder",
        default="./gsm8k_output",
        help="Local folder containing raw_extractions.pkl, all_sentences_features.pkl, etc.",
    )
    p.add_argument(
        "--repo",
        default="withmartian/gsm8k_qwen14b_SDS_traindata",
        help="Hugging Face dataset repo id (e.g. username/repo_name)",
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Upload only the 4 merged files (no shard_* dirs)",
    )
    p.add_argument(
        "--path-in-repo",
        default=".",
        help="Path inside the repo where to upload (e.g. 'layer_27' to put files in repo/layer_27/)",
    )
    args = p.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    if args.merge_only:
        files = [
            "raw_extractions.pkl",
            "all_sentences_features.pkl",
            "all_sentences_features_with_neutral.pkl",
            "cot_data.pkl",
        ]
        for f in files:
            path = os.path.join(folder, f)
            if not os.path.isfile(path):
                print(f"Skip (missing): {path}")
                continue
            print(f"Uploading {f} ...")
            dest = os.path.join(args.path_in_repo, f).replace("\\", "/")
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=dest,
                repo_id=args.repo,
                repo_type="dataset",
                token=token,
            )
        print("Done (merge-only).")
    else:
        print(f"Uploading folder {folder} to {args.repo} ({args.path_in_repo}) ...")
        api.upload_folder(
            folder_path=folder,
            repo_id=args.repo,
            repo_type="dataset",
            path_in_repo=args.path_in_repo,
            token=token,
        )
        print("Done.")


if __name__ == "__main__":
    main()
