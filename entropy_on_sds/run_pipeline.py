"""
Run the full entropy-on-SDS pipeline in order:
  1. Extract sentence-level entropy (requires model + features + cot_data)
  2. Infer SDS states (requires features; trains CEBRA-style SDS)
  3. Analyze: entropy by regime, at transitions, by transition type, etc.

Usage:
  cd mi-cot/entropy_on_sds
  python run_pipeline.py --features ../rpc_dataset/all_sentences_features.pkl --cot ../rpc_dataset/cot_data.pkl [--limit-problems 100]
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
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", help="Model for entropy extraction")
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

    # Step 1: Extract sentence entropy
    if not args.skip_extract:
        cmd = [
            sys.executable, "extract_sentence_entropy.py",
            "--features", features_abs,
            "--cot", cot_abs,
            "--out", out_abs,
            "--model", args.model,
        ]
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

    print("Pipeline complete. Outputs in", out_abs)


if __name__ == "__main__":
    main()
