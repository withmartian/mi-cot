#!/usr/bin/env python3
"""
Run causal-lens CEBRA-EM judge pipeline for reasoning+base counterparts:
  - qwen1.5b
  - llama3 (Llama-3.1-8B)
  - qwen14b
across:
  - SDS_train_gsm8k
  - SDS_math500_test
  - SDS_train_mmlu-pro
  - SDS_train_svamp

Writes per-run artifacts plus an incrementally-updated LaTeX summary after each
model+dataset combo (both sides attempted).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "cebra_em_steering_inputDep_logitlens_judge_causal_lens.py"

# Logical compromise: enough judged rows for stable aggregate estimates, but
# materially less runtime/cost than judging all steered rows.
JUDGE_CAP = 120
TARGET_SAMPLES = 1000
EM_ITERS = 40
LENS_DEVICE = "cpu"

DATASETS: List[Tuple[str, str]] = [
    ("gsm8k", "withmartian/SDS_train_gsm8k"),
    ("math500", "withmartian/SDS_math500_test"),
    ("mmlu-pro", "withmartian/SDS_train_mmlu-pro"),
    ("svamp", "withmartian/SDS_train_svamp"),
]

MODEL_PAIRS: Dict[str, Dict[str, str]] = {
    "qwen15b": {
        "reasoning_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "reasoning_lens": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "base_model": "Qwen/Qwen2.5-1.5B",
        "base_lens": "Qwen/Qwen2.5-1.5B",
        "tex": "Qwen-1.5B / Qwen2.5-1.5B",
    },
    "llama8b": {
        "reasoning_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "reasoning_lens": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "base_model": "meta-llama/Llama-3.1-8B",
        "base_lens": "meta-llama/Llama-3.1-8B",
        "tex": "Llama-3.1-8B (reasoning/base)",
    },
    "qwen14b": {
        "reasoning_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "reasoning_lens": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "base_model": "Qwen/Qwen2.5-14B",
        "base_lens": "Qwen/Qwen2.5-14B",
        "tex": "Qwen-14B / Qwen2.5-14B",
    },
}

# Dataset repos do not share a single naming convention for feature subpaths.
# Use explicit per-dataset paths to avoid 404s from automatic model-id mapping.
MODEL_FEATURE_SUBPATHS: Dict[str, Dict[str, Dict[str, str]]] = {
    "gsm8k": {
        "qwen15b": {"reasoning": "qwen1.5b_reasoning/layer_27", "base": "qwen_1.5B_base/layer_27"},
        "llama8b": {"reasoning": "llama_8b_reasoning/layer_31", "base": "llama_8B_base/layer_31"},
        "qwen14b": {"reasoning": "qwen_14b_reasoning/layer_47", "base": "qwen_14B_base/layer_47"},
    },
    "mmlu-pro": {
        "qwen15b": {"reasoning": "qwen1.5b/layer_27", "base": "qwen1.5b_base/layer_27"},
        "llama8b": {"reasoning": "llama8b/layer_31", "base": "llama8b_base/layer_31"},
        "qwen14b": {"reasoning": "qwen14b/layer_47", "base": "qwen14b_base/layer_47"},
    },
    "svamp": {
        "qwen15b": {"reasoning": "Qwen_1_5B_reasoning/layer_27", "base": "Qwen_1_5B_base/layer_27"},
        "llama8b": {"reasoning": "Llama_8B_reasoning/layer_31", "base": "Llama_8B_base/layer_31"},
        "qwen14b": {"reasoning": "Qwen_14B_reasoning/layer_47", "base": "Qwen_14B_base/layer_47"},
    },
    "math500": {
        "qwen15b": {"reasoning": "Qwen_1_5B_reasoning/layer_27", "base": "Qwen_1_5B_base/layer_27"},
        "llama8b": {"reasoning": "Llama_8B_reasoning/layer_31", "base": "Llama_8B_base/layer_31"},
        "qwen14b": {"reasoning": "Qwen_14B_reasoning/layer_47", "base": "Qwen_14B_base/layer_47"},
    },
}

DATASET_TEX = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "mmlu-pro": "MMLU-Pro",
    "svamp": "SVAMP",
}


def _tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def parse_only_n(stderr: str) -> Optional[int]:
    m = re.search(r"Only (\d+) non-NEUTRAL samples", stderr)
    return int(m.group(1)) if m else None


def build_cmd(
    save_dir: Path,
    dataset_repo: str,
    model_arg: str,
    lens_id: str,
    hf_samples: int,
) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--dataset",
        dataset_repo,
        "--model",
        model_arg,
        "--logit-lens-model-id",
        lens_id,
        "--logit-lens-device",
        LENS_DEVICE,
        "--hf-samples",
        str(hf_samples),
        "--judge-max-samples",
        str(JUDGE_CAP),
        "--em-iters",
        str(EM_ITERS),
        "--save-dir",
        str(save_dir),
    ]


def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(HERE),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, out


def run_one_side(
    *,
    combo_dir: Path,
    dataset_repo: str,
    model_slug: str,
    ds_slug: str,
    side: str,
    model_arg: str,
    lens_id: str,
) -> Dict[str, Any]:
    save_dir = combo_dir / f"{model_slug}_{side}_{ds_slug}"
    save_dir.mkdir(parents=True, exist_ok=True)
    n_try = TARGET_SAMPLES
    cmd = build_cmd(save_dir, dataset_repo, model_arg, lens_id, n_try)
    code, combined = run_cmd(cmd)
    if code != 0:
        n_avail = parse_only_n(combined)
        if n_avail is not None and n_avail < n_try:
            cmd = build_cmd(save_dir, dataset_repo, model_arg, lens_id, n_avail)
            code, combined = run_cmd(cmd)
            n_try = n_avail
    ok = code == 0
    if not ok:
        (save_dir / "batch_error.txt").write_text(combined[-12000:], encoding="utf-8")
        return {
            "ok": False,
            "save_dir": str(save_dir),
            "hf_used": n_try,
            "n_judge": 0,
            "steer": 0.0,
            "coh": 0.0,
            "conf": 0.0,
        }
    js = save_dir / "judge_logitlens_summary.json"
    summ = {}
    if js.is_file():
        with open(js, "r", encoding="utf-8") as f:
            data = json.load(f)
        summ = (data.get("judge_report") or {}).get("summary") or {}
    return {
        "ok": True,
        "save_dir": str(save_dir),
        "hf_used": n_try,
        "n_judge": int(summ.get("valid_count", summ.get("count", 0))),
        "steer": float(summ.get("judge_steering_toward_target_pct", 0.0)),
        "coh": float(summ.get("judge_coherence_success_pct", 0.0)),
        "conf": float(summ.get("mean_confidence", 0.0)),
    }


def write_progress_summary(root: Path, records: List[Dict[str, Any]]) -> None:
    rows: List[str] = []
    for r in records:
        rr = r.get("reasoning", {})
        bb = r.get("base", {})
        if not rr.get("ok") or not bb.get("ok"):
            rows.append(
                f"{_tex_escape(r['model_tex'])} & {_tex_escape(r['ds_tex'])} & "
                f"\\emph{{partial/fail}} & & & & & \\\\"
            )
            continue
        ds = float(bb["steer"]) - float(rr["steer"])
        dc = float(bb["coh"]) - float(rr["coh"])
        rows.append(
            f"{_tex_escape(r['model_tex'])} & {_tex_escape(r['ds_tex'])} & "
            f"{rr['steer']:.1f} & {bb['steer']:.1f} & {ds:+.1f} & "
            f"{rr['coh']:.1f} & {bb['coh']:.1f} & {dc:+.1f} \\\\"
        )
    body = "\n".join(rows)
    text = f"""\\label{{sec:causal-lens-reasoning-vs-base-progress}}

\\paragraph{{Batch status.}}
Incremental summary refreshed after each completed model+dataset combo.
Protocol: \\texttt{{cebra\\_em\\_steering\\_inputDep\\_logitlens\\_judge\\_causal\\_lens.py}},
$K{{=}}4$, target samples up to {TARGET_SAMPLES}, judge cap {JUDGE_CAP}, EM iters {EM_ITERS}.

\\begin{{table}}[t]
\\centering
\\scriptsize
\\caption{{Reasoning vs. base judged metrics (\\textbf{{progressive}}; $\\Delta$ = base minus reasoning).}}
\\begin{{tabular}}{{llrrrrrr}}
\\hline
Family & Dataset & Steer$_{{\\mathrm{{R}}}}$ & Steer$_{{\\mathrm{{B}}}}$ & $\\Delta$Steer & Coh$_{{\\mathrm{{R}}}}$ & Coh$_{{\\mathrm{{B}}}}$ & $\\Delta$Coh \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}

\\paragraph{{Output root.}} \\texttt{{{_tex_escape(root.name)}}}
"""
    (root / "master_reasoning_vs_base_progress.tex").write_text(text, encoding="utf-8")


def write_final_analysis(root: Path, records: List[Dict[str, Any]]) -> None:
    good = [r for r in records if r.get("reasoning", {}).get("ok") and r.get("base", {}).get("ok")]
    if not good:
        (root / "master_reasoning_vs_base_final_analysis.tex").write_text(
            "No completed pairs.\n", encoding="utf-8"
        )
        return
    n_reason_higher = 0
    n_base_higher = 0
    n_tie = 0
    lines = []
    for r in good:
        rr = r["reasoning"]
        bb = r["base"]
        ds = float(bb["steer"]) - float(rr["steer"])
        if abs(ds) < 1.0:
            n_tie += 1
        elif ds > 0:
            n_base_higher += 1
        else:
            n_reason_higher += 1
        lines.append(
            f"{_tex_escape(r['model_tex'])} ({_tex_escape(r['ds_tex'])}): "
            f"Steer {rr['steer']:.1f} vs {bb['steer']:.1f}, "
            f"Coh {rr['coh']:.1f} vs {bb['coh']:.1f}."
        )
    txt = (
        "\\section*{Final analysis}\n"
        f"Completed comparable cells: {len(good)}.\n\n"
        "\\paragraph{Steering direction counts.}\n"
        f"Reasoning higher in {n_reason_higher}, base higher in {n_base_higher}, tie in {n_tie}.\n\n"
        "\\paragraph{Per-cell notes.}\n"
        + "\n".join([f"- {x}" for x in lines])
        + "\n"
    )
    (root / "master_reasoning_vs_base_final_analysis.tex").write_text(txt, encoding="utf-8")


def main() -> None:
    if not SCRIPT.is_file():
        sys.exit(f"Missing script: {SCRIPT}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = HERE / f"cebra_causal_lens_reasoning_base_4ds_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STARTING BATCH (3 models × 4 datasets × 2 sides = 24 runs)")
    print(f"Judge cap: {JUDGE_CAP}; target samples: {TARGET_SAMPLES}; EM iters: {EM_ITERS}")
    print("Estimated wall time: ~6-12 hours total (sequential, API/network dependent).")
    print("=" * 72)

    records: List[Dict[str, Any]] = []
    total_combo = len(MODEL_PAIRS) * len(DATASETS)
    idx_combo = 0
    for model_slug in ("qwen15b", "llama8b", "qwen14b"):
        spec = MODEL_PAIRS[model_slug]
        for ds_slug, dataset_repo in DATASETS:
            idx_combo += 1
            print(f"starting [new run {model_slug}+{ds_slug}] ({idx_combo}/{total_combo})", flush=True)
            combo_dir = root / f"{model_slug}_{ds_slug}"
            combo_dir.mkdir(parents=True, exist_ok=True)

            rec: Dict[str, Any] = {
                "model_slug": model_slug,
                "ds_slug": ds_slug,
                "model_tex": spec["tex"],
                "ds_tex": DATASET_TEX[ds_slug],
            }

            rec["reasoning"] = run_one_side(
                combo_dir=combo_dir,
                dataset_repo=dataset_repo,
                model_slug=model_slug,
                ds_slug=ds_slug,
                side="reasoning",
                model_arg=MODEL_FEATURE_SUBPATHS[ds_slug][model_slug]["reasoning"],
                lens_id=spec["reasoning_lens"],
            )
            rec["base"] = run_one_side(
                combo_dir=combo_dir,
                dataset_repo=dataset_repo,
                model_slug=model_slug,
                ds_slug=ds_slug,
                side="base",
                model_arg=MODEL_FEATURE_SUBPATHS[ds_slug][model_slug]["base"],
                lens_id=spec["base_lens"],
            )
            records.append(rec)

            write_progress_summary(root, records)
            print(f"finished run {model_slug}+{ds_slug}", flush=True)

    write_final_analysis(root, records)
    print(f"Batch complete. Root: {root}", flush=True)


if __name__ == "__main__":
    main()

