"""
Run the full entropy-on-SDS pipeline in order:
  1. Extract sentence-level entropy (requires model + features + cot_data)
  2. Infer SDS states (requires features; trains CEBRA-style SDS)
  3. Analyze: entropy by regime, at transitions, by transition type, etc.

Usage:
  cd mi-cot/entropy_on_sds
  python run_pipeline.py --features ../rpc_dataset/all_sentences_features.pkl --cot ../rpc_dataset/cot_data.pkl
  # Base vs fine-tuned is on by default (--base-model Qwen/Qwen2.5-14B). Use --no-base to skip.
  # Optional: --limit-problems N
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Run full entropy-on-SDS pipeline")
    parser.add_argument("--features", default="rpc_dataset/all_sentences_features.pkl", help="Path to all_sentences_features.pkl")
    parser.add_argument("--cot", default="rpc_dataset/cot_data.pkl", help="Path to cot_data.pkl")
    parser.add_argument("--out", default="entropy_on_sds_output", help="Output directory")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", help="Model for entropy extraction (fine-tuned/CoT)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-14B", help="Base model for base vs fine-tuned entropy comparison")
    parser.add_argument("--no-base", action="store_true", help="Skip base model extraction (fine-tuned only)")
    parser.add_argument("--limit-problems", type=int, default=None, help="Use only problem_id < N (faster for testing)")
    parser.add_argument("--K", type=int, default=4, help="Number of SDS regimes")
    parser.add_argument("--skip-extract", action="store_true", help="Skip step 1 (use existing sentence_entropy.pkl)")
    parser.add_argument("--skip-sds", action="store_true", help="Skip step 2 (use existing sds_states.pkl)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # Resolve relative paths against cwd (script_dir) so ../rpc_dataset_small works
    out_abs = os.path.abspath(args.out) if not os.path.isabs(args.out) else args.out
    features_abs = os.path.abspath(args.features) if not os.path.isabs(args.features) else args.features
    cot_abs = os.path.abspath(args.cot) if not os.path.isabs(args.cot) else args.cot
    os.makedirs(out_abs, exist_ok=True)
    # Write marker so the output folder is visible and path is explicit
    out_abs_resolved = os.path.abspath(out_abs)
    marker = os.path.join(out_abs_resolved, "OUTPUT_DIR.txt")
    with open(marker, "w") as f:
        f.write("Entropy-on-SDS pipeline output\n")
        f.write("Full path: " + out_abs_resolved + "\n")
        f.write("\nIn the file tree, this folder is: mi-cot/entropy_on_sds/entropy_on_sds_output\n")
    print("Output directory:", out_abs_resolved)

    # Step 1: Extract sentence entropy (and optionally base model entropy)
    if not args.skip_extract:
        cmd = [
            sys.executable, "extract_sentence_entropy.py",
            "--features", features_abs,
            "--cot", cot_abs,
            "--out", out_abs,
            "--model", args.model,
        ]
        if not args.no_base and args.base_model and args.base_model.strip():
            cmd += ["--base-model", args.base_model.strip()]
        if args.limit_problems is not None:
            cmd += ["--limit-problems", str(args.limit_problems)]
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("extract_sentence_entropy.py failed")
            sys.exit(r.returncode)
    else:
        print("Skipping step 1 (extract sentence entropy)")

    # Step 2: Infer SDS states
    if not args.skip_sds:
        cmd = [
            sys.executable, "infer_sds_states.py",
            "--features", features_abs,
            "--out", out_abs,
            "--K", str(args.K),
            "--save-model",
        ]
        if args.limit_problems is not None:
            cmd += ["--limit-problems", str(args.limit_problems)]
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("infer_sds_states.py failed")
            sys.exit(r.returncode)
    else:
        print("Skipping step 2 (infer SDS states)")

    # Step 3: Analyze
    cmd = [
        sys.executable, "analyze_entropy_states.py",
        "--out", out_abs,
        "--features", features_abs,
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("analyze_entropy_states.py failed")
        sys.exit(r.returncode)

    # Step 4: Build dashboard
    cmd = [sys.executable, "build_entropy_sds_dashboard.py", "--out", out_abs, "--features", features_abs]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("build_entropy_sds_dashboard.py failed (non-fatal)")
    else:
        print("Dashboard written to", os.path.join(out_abs, "dashboard_entropy_sds.html"))

    print("Pipeline complete.")
    print("Outputs written to:", os.path.abspath(out_abs))
    print("  Open this folder in your file explorer or IDE to see the files.")


if __name__ == "__main__":
    main()
