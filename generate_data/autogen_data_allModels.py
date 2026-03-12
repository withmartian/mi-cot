#!/usr/bin/env python3
"""
Run create_dataset for all models/layers in the SDS_train_gsm8k repo layout,
then upload each output to Hugging Face. Supports single-GPU or multi-GPU.

Requires:
  - A valid Hugging Face access token (HF_TOKEN from env, .env, or interactive prompt).
  - The token must have gated access to any models used (e.g. Llama, Qwen) and write
    access to the destination repo when not using --skip-upload.
  - A valid Hugging Face repo ID (--repo) for uploads.
  - One of the supported dataset options: math500, gsm8k, humaneval, svamp, mmlu-pro, hotpotqa

Usage:
  export HF_TOKEN=your_token
  python autogen_data_allModels.py --repo withmartian/SDS_train_gsm8k --dataset gsm8k --n 2000
  python autogen_data_allModels.py --repo withmartian/SDS_train_gsm8k --dataset gsm8k --ngpus 1   # single-GPU
  python autogen_data_allModels.py --repo withmartian/SDS_train_gsm8k --dataset math500 --split test --n 500 --skip-upload
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Paths relative to this script (create_dataset scripts live in this folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
UPLOAD_SCRIPT = os.path.join(REPO_ROOT, "huggingface_scripts", "upload_to_hf.py")
CREATE_SCRIPT_MULTIGPU = os.path.join(SCRIPT_DIR, "create_dataset_multigpu.py")
CREATE_SCRIPT_SINGLE = os.path.join(SCRIPT_DIR, "create_dataset.py")

# All models in the repo: (repo_folder_name, model_hf_id, base_hf_id, layers)
# For "reasoning" variants, model is the reasoning model and base is the base; adjust if your setup differs.
MODEL_CONFIGS = [
    ("llama3.1_8b_base", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B", [22, 31]),
    ("llama3.1_8b_reasoning", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B", [22, 31]),
    ("qwen2.5_1.5b_base", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B", [20, 27]),
    ("qwen2.5_1.5b_reasoning", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B", [20, 27]),
    ("qwen2.5_14b_base", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-14B", [28, 47]),
    ("qwen2.5_14b_reasoning", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B", [28, 47]),
]

# Dataset option -> (hf_dataset_id, split, default_n)
# Formats: humaneval uses "prompt"; svamp uses "question" (train); MMLU-Pro uses "question" + "options"
DATASET_OPTIONS = {
    "math500": ("HuggingFaceH4/MATH-500", "test", 500),
    "gsm8k": ("openai/gsm8k", "train", 2000),
    "humaneval": ("openai/openai_humaneval", "test", 164),
    "svamp": ("garrethlee/svamp", "train", 800),
    "mmlu-pro": ("TIGER-Lab/MMLU-Pro", "test", 500),
    "hotpotqa": ("hotpot_qa", "train", 1000),
}


def _load_dotenv(path: str) -> None:
    """Load HF_TOKEN from a .env file if present. Sets os.environ only for HF_TOKEN."""
    if not os.path.isfile(path):
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    val = line.split("=", 1)[1].strip()
                    if val.startswith('"') and val.endswith('"'):
                        val = val[1:-1].replace('\\"', '"')
                    elif val.startswith("'") and val.endswith("'"):
                        val = val[1:-1].replace("\\'", "'")
                    if val:
                        os.environ.setdefault("HF_TOKEN", val)
                    return
    except OSError:
        pass


def ensure_hf_token() -> None:
    """Ensure HF_TOKEN is set: check env, then .env in cwd and script dir, else prompt. Exit if not provided."""
    if os.environ.get("HF_TOKEN", "").strip():
        print("Using HF_TOKEN from environment. Ensure it has gated access to the models used (e.g. meta-llama/*, Qwen/*).")
        return
    _load_dotenv(os.path.join(os.getcwd(), ".env"))
    if os.environ.get("HF_TOKEN", "").strip():
        print("Using HF_TOKEN from .env. Ensure it has gated access to the models used (e.g. meta-llama/*, Qwen/*).")
        return
    _load_dotenv(os.path.join(SCRIPT_DIR, ".env"))
    if os.environ.get("HF_TOKEN", "").strip():
        print("Using HF_TOKEN from .env. Ensure it has gated access to the models used (e.g. meta-llama/*, Qwen/*).")
        return
    print(
        "Hugging Face token is required for model downloads and for uploading results.",
        file=sys.stderr,
    )
    print(
        "Make sure your token has gated access to the models used (e.g. Llama, Qwen) and "
        "write access to the destination repo.",
        file=sys.stderr,
    )
    try:
        token = input("Enter your Hugging Face token (or press Enter to exit): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nNo token provided. Exiting.", file=sys.stderr)
        sys.exit(1)
    if not token:
        print("No token provided. Exiting.", file=sys.stderr)
        sys.exit(1)
    os.environ["HF_TOKEN"] = token
    print("Token set. Ensure it has gated access to the relevant models (e.g. meta-llama/*, Qwen/*).")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run create_dataset (multi-GPU) for all repo models/layers, then upload to HF."
    )
    p.add_argument(
        "--repo",
        required=True,
        help="Hugging Face dataset repo id (e.g. withmartian/SDS_train_gsm8k). Required.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_OPTIONS.keys()),
        help="Dataset to run: one of math500, gsm8k, humaneval, svamp, mmlu-pro, hotpotqa",
    )
    p.add_argument(
        "--out-root",
        default="./dataset_all_output",
        help="Root directory for outputs (default: ./dataset_all_output)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of examples (default: dataset-specific)",
    )
    p.add_argument(
        "--split",
        default=None,
        help="Dataset split (default: dataset-specific)",
    )
    p.add_argument(
        "--ngpus",
        type=int,
        default=8,
        help="Number of GPUs: 1 = single-GPU (create_dataset.py), >1 = multi-GPU (create_dataset_multigpu.py). Default: 8",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Only run create_dataset, do not upload to HF",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="If set, only run these repo folder names (e.g. qwen_14B_base llama_8B_base)",
    )
    p.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="If set, only run these layer numbers (e.g. 28 47)",
    )
    return p.parse_args()


def run_create_dataset(
    model_hf: str,
    base_hf: str,
    dataset_hf: str,
    split: str,
    n: int,
    layer: int,
    out_dir: str,
    ngpus: int,
) -> bool:
    use_single = ngpus == 1
    script = CREATE_SCRIPT_SINGLE if use_single else CREATE_SCRIPT_MULTIGPU
    cmd = [
        sys.executable,
        script,
        "--model",
        model_hf,
        "--base",
        base_hf,
        "--dataset",
        dataset_hf,
        "--split",
        split,
        "--n",
        str(n),
        "--layer",
        str(layer),
        "--out",
        out_dir,
    ]
    if not use_single:
        cmd.extend(["--ngpus", str(ngpus)])
    try:
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)
        return True
    except subprocess.CalledProcessError as e:
        script_name = "create_dataset" if use_single else "create_dataset_multigpu"
        print(f"{script_name} failed: {e}", file=sys.stderr)
        return False


def run_upload(folder: str, repo: str, path_in_repo: str) -> bool:
    if not os.path.isdir(folder):
        print(f"Skip upload (missing dir): {folder}", file=sys.stderr)
        return False
    cmd = [
        sys.executable,
        UPLOAD_SCRIPT,
        "--folder",
        folder,
        "--repo",
        repo,
        "--path-in-repo",
        path_in_repo,
        "--merge-only",
    ]
    try:
        subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"upload_to_hf failed: {e}", file=sys.stderr)
        return False


def main():
    args = parse_args()

    ensure_hf_token()

    if not os.path.isfile(CREATE_SCRIPT_SINGLE):
        print(f"Missing script: {CREATE_SCRIPT_SINGLE}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(CREATE_SCRIPT_MULTIGPU):
        print(f"Missing script: {CREATE_SCRIPT_MULTIGPU}", file=sys.stderr)
        sys.exit(1)
    if not args.skip_upload and not os.path.isfile(UPLOAD_SCRIPT):
        print(f"Missing script: {UPLOAD_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    ds_id, split_default, n_default = DATASET_OPTIONS[args.dataset]
    split = args.split if args.split is not None else split_default
    n = args.n if args.n is not None else n_default

    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    configs = MODEL_CONFIGS
    if args.models:
        configs = [c for c in configs if c[0] in args.models]
        if not configs:
            print("No matching models for --models", file=sys.stderr)
            sys.exit(1)

    total = 0
    failed = []
    for repo_folder, model_hf, base_hf, layers in configs:
        layer_list = layers if args.layers is None else [l for l in layers if l in args.layers]
        if args.layers is not None and not layer_list:
            continue
        for layer in layer_list:
            path_in_repo = f"{repo_folder}/layer_{layer}"
            out_dir = os.path.join(out_root, args.dataset, repo_folder, f"layer_{layer}")
            total += 1
            print(f"\n{'='*60}")
            mode = "single-GPU" if args.ngpus == 1 else f"multi-GPU (ngpus={args.ngpus})"
            print(f"[{total}] {repo_folder} layer_{layer} | dataset={args.dataset} ({ds_id}) n={n} split={split} | {mode}")
            print("="*60)
            if not run_create_dataset(
                model_hf=model_hf,
                base_hf=base_hf,
                dataset_hf=ds_id,
                split=split,
                n=n,
                layer=layer,
                out_dir=out_dir,
                ngpus=args.ngpus,
            ):
                failed.append(f"{repo_folder}/layer_{layer}")
                continue
            if not args.skip_upload:
                run_upload(folder=out_dir, repo=args.repo, path_in_repo=path_in_repo)

    print(f"\nDone. Ran {total} jobs.")
    if failed:
        print(f"Failed: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
