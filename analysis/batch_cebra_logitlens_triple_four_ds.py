#!/usr/bin/env python3
"""
Batch CEBRA-EM logit-lens + judge: Qwen-1.5B, Llama-3.1-8B R1-distill, Qwen-14B
× GSM8K, MATH-500, MMLU-Pro, SVAMP.

- Target up to 1000 non-NEUTRAL rows per run; if the pickle has fewer, retries with that count.
- Each model×dataset gets its own folder and analysis_run.tex as soon as the run finishes.
- Final master_summary_triple_model_four_ds.tex aggregates all runs.

Usage (from mi-cot/exploration):
  python batch_cebra_logitlens_triple_four_ds.py
Requires OPENAI_API_KEY, HF_TOKEN recommended.
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
SCRIPT = HERE / "cebra_em_steering_inputDep_logitlens_judge_only.py"
TARGET_SAMPLES = 1000
JUDGE_CAP = 300

DATASETS: List[Tuple[str, str]] = [
    ("gsm8k", "withmartian/SDS_train_gsm8k"),
    ("math500", "withmartian/SDS_math500_test"),
    ("mmlu-pro", "withmartian/SDS_train_mmlu-pro"),
    ("svamp", "withmartian/SDS_train_svamp"),
]

# (slug, --model value, --logit-lens-model-id)
MODEL_SPECS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "qwen15b": {
        "gsm8k": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        ),
        "math500": (
            "Qwen_1_5B_reasoning/layer_27",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        ),
        "mmlu-pro": (
            "qwen1.5b/layer_27",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        ),
        "svamp": (
            "Qwen_1_5B_reasoning/layer_27",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        ),
    },
    "llama8b": {
        "gsm8k": (
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ),
        "math500": (
            "Llama_8B_reasoning/layer_31",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ),
        "mmlu-pro": (
            "llama8b/layer_31",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ),
        "svamp": (
            "Llama_8B_reasoning/layer_31",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ),
    },
    "qwen14b": {
        "gsm8k": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        ),
        "math500": (
            "Qwen_14B_reasoning/layer_47",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        ),
        "mmlu-pro": (
            "qwen14b/layer_47",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        ),
        "svamp": (
            "Qwen_14B_reasoning/layer_47",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        ),
    },
}

MODEL_LABEL_TEX = {
    "qwen15b": "Qwen-1.5B (DeepSeek R1-distill)",
    "llama8b": "Llama-3.1-8B (DeepSeek R1-distill)",
    "qwen14b": "Qwen-14B (DeepSeek R1-distill)",
}

DATASET_LABEL_TEX = {
    "gsm8k": "GSM8K (\\texttt{SDS\\_train\\_gsm8k})",
    "math500": "MATH-500 (\\texttt{SDS\\_math500\\_test})",
    "mmlu-pro": "MMLU-Pro (\\texttt{SDS\\_train\\_mmlu-pro})",
    "svamp": "SVAMP (\\texttt{SDS\\_train\\_svamp})",
}


def _tex_escape(s: str, max_len: int = 0) -> str:
    t = (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )
    if max_len and len(t) > max_len:
        t = t[: max_len - 3] + "..."
    return t


def build_cmd(
    save_dir: Path,
    dataset_repo: str,
    model_arg: str,
    lens_id: str,
    hf_samples: int,
    judge_max: int,
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
        "auto",
        "--hf-samples",
        str(hf_samples),
        "--judge-max-samples",
        str(judge_max),
        "--save-dir",
        str(save_dir),
    ]


def replace_hf_samples(cmd: List[str], new_n: int) -> List[str]:
    out = cmd.copy()
    i = out.index("--hf-samples")
    out[i + 1] = str(new_n)
    return out


def run_one_subprocess(cmd: List[str]) -> Tuple[int, str, str]:
    env = os.environ.copy()
    p = subprocess.run(
        cmd,
        cwd=str(HERE),
        env=env,
        capture_output=True,
        text=True,
    )
    return p.returncode, p.stdout, p.stderr


def parse_only_n(stderr: str) -> Optional[int]:
    m = re.search(r"Only (\d+) non-NEUTRAL samples", stderr)
    if m:
        return int(m.group(1))
    return None


def write_analysis_run_tex(
    run_dir: Path,
    slug: str,
    model_slug: str,
    ds_slug: str,
    hf_used: int,
    stdout_tail: str,
) -> None:
    """Per-run LaTeX with metrics + short analysis (survives if batch stops later)."""
    js_path = run_dir / "judge_logitlens_summary.json"
    if not js_path.is_file():
        (run_dir / "analysis_run.tex").write_text(
            f"% analysis_run.tex — run failed or JSON missing for {slug}\n", encoding="utf-8"
        )
        return
    with open(js_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rep = data.get("judge_report", {})
    summ = rep.get("summary", {})
    n_j = int(summ.get("valid_count", summ.get("count", 0)))
    steer = float(summ.get("judge_steering_toward_target_pct", 0.0))
    coh = float(summ.get("judge_coherence_success_pct", 0.0))
    conf = float(summ.get("mean_confidence", 0.0))
    ann = data.get("state_annotations", {})
    audit = data.get("regime_annotation_audit", {})
    n_regimes = len(audit) if audit else len(ann)

    lines: List[str] = [
        f"% Auto-written after successful pipeline: {slug}",
        f"% HF non-NEUTRAL target/actual rows (CEBRA fit): {hf_used}",
        "",
        r"\paragraph{Run: " + _tex_escape(MODEL_LABEL_TEX[model_slug]) + r" on " + DATASET_LABEL_TEX[ds_slug] + r".}",
        (
            f"We fit CEBRA-EM ($K{{=}}4$) on up to {hf_used} sentence features, "
            f"annotated regimes from centroid logit lens via \\texttt{{gpt-4.1-mini}}, "
            f"then judged up to {JUDGE_CAP} steered timesteps (post-activation lens). "
            f"Artifacts: \\texttt{{{_tex_escape(str(run_dir.name))}/}}."
        ),
        "",
        r"\begin{table}[h]",
        r"\centering\small",
        rf"\caption{{Aggregate judge metrics ({_tex_escape(slug)}).}}",
        rf"\label{{tab:run-{_tex_escape(slug.replace('_', '-'))}}}",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Metric & $N_{\mathrm{judge}}$ & Value \\",
        r"\hline",
        rf"Steering toward target (\%) & {n_j} & {steer:.1f} \\",
        rf"Coherence (\%) & {n_j} & {coh:.1f} \\",
        rf"Mean judge confidence & {n_j} & {conf:.2f} \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\paragraph{Interpretation.}",
    ]
    if n_j == 0:
        lines.append("No successful judge rows; check API key and logs.")
    elif steer >= 70 and coh >= 70:
        lines.append(
            "Under this protocol, both steering and coherence are relatively high: the judge often sees "
            "a plausible shift toward the target regime readout without internal contradiction."
        )
    elif steer < 40 and coh >= 80:
        lines.append(
            "\\textbf{High coherence, low steering:} readouts stay self-consistent, but the post-steering "
            "activation lens rarely moves toward the target centroid pattern—weak evidence for controllability "
            "at this layer under cyclic soft steering."
        )
    elif coh < 60:
        lines.append(
            "Coherence is relatively low; inspect \\texttt{judge\\_report.rows} in JSON for ambiguous or contradictory readouts."
        )
    else:
        lines.append(
            "Mixed steering and coherence; see transition breakdown \\texttt{judge\\_transition\\_breakdown\\_table.tex} "
            "and per-row reasons in \\texttt{judge\\_logitlens\\_summary.json}."
        )
    lines.extend(
        [
            "",
            rf"\paragraph{{Regimes.}} {n_regimes} regime(s) with centroid lens + annotations in "
            r"\texttt{state\_annotations\_with\_lens\_table.tex} and \texttt{regime\_annotation\_audit} (JSON).",
            "",
        ]
    )
    (run_dir / "analysis_run.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Save a small log snippet for debugging interrupted batches
    log_path = run_dir / "batch_subprocess_tail.txt"
    tail = (stdout_tail or "")[-8000:]
    log_path.write_text(tail, encoding="utf-8")


def write_master_summary(root: Path, records: List[Dict[str, Any]]) -> None:
    rows = []
    for r in records:
        if not r.get("ok"):
            rows.append(
                f"{r['model_tex']} & {r['ds_tex']} & --- & --- & --- & --- & "
                f"\\emph{{failed}} \\\\"
            )
            continue
        rows.append(
            f"{r['model_tex']} & {r['ds_tex']} & {r['hf_used']} & {r['n_judge']} & "
            f"{r['steer']:.1f} & {r['coh']:.1f} & {r['conf']:.2f} \\\\"
        )
    out = root / "master_summary_triple_model_four_ds.tex"
    body = "\n".join(rows)
    text = r"""% Master summary: all model × dataset logit-lens judge batches.
% Generated by batch_cebra_logitlens_triple_four_ds.py
\label{sec:master-logitlens-triple-four}

\paragraph{Procedure.}
Each cell is an independent run of \texttt{cebra\_em\_steering\_inputDep\_logitlens\_judge\_only.py} with up to 1000 non-NEUTRAL Hub rows (or all available if fewer), $K{=}4$ EM states, centroid logit-lens regime labels (\texttt{gpt-4.1-mini}), and up to 300 post-steering judge calls per run.
See each subfolder's \texttt{analysis\_run.tex} for per-run narrative.

\begin{table}[t]
\centering
\footnotesize
\caption{Aggregate post-activation logit-lens judge metrics (all model--dataset pairs).}
\label{tab:master-triple-four-aggregate}
\begin{tabular}{llrrrrr}
\hline
Model & Dataset & HF rows & $N_{\mathrm{judge}}$ & Steer (\%) & Coh. (\%) & Mean conf. \\
\hline
""" + body + r"""
\hline
\end{tabular}
\end{table}

\paragraph{Cross-cutting notes.}
Compare \textbf{Steer (\%)} across the same dataset column: differences are driven by model family and hidden-state geometry, not the benchmark alone.
Pairs with high coherence but low steering suggest the intervention rarely moves the vocabulary-facing readout toward the target regime pattern even when logits look plausible—\textbf{not} strong causal sufficiency for SDS-style claims without stronger interventions or larger $N$.
Failed rows indicate subprocess error; inspect that folder's terminal log or rerun.

\paragraph{Per-run detail.}
Each subdirectory contains \texttt{analysis\_run.tex}, full \texttt{judge\_logitlens\_summary.json}, and lens-audit tables.
"""
    out.write_text(text, encoding="utf-8")
    print(f"Wrote master summary: {out}")


def main() -> None:
    if not SCRIPT.is_file():
        sys.exit(f"Missing {SCRIPT}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = HERE / f"cebra_logitlens_batch_triple_4ds_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("TIME ESTIMATE (sequential, order-of-magnitude)")
    print("=" * 72)
    print("12 runs total (3 models × 4 datasets).")
    print("GPU: Qwen-14B unembedding + EM is slowest (~several min/run); 1.5B/8B faster.")
    print(f"API: up to {JUDGE_CAP} judge calls × 12 runs ≈ {JUDGE_CAP * 12} OpenAI requests → often 45–120+ min.")
    print("TOTAL WALL CLOCK (typical): roughly 2–5 hours if nothing fails; plan for interruptions—each run writes its own folder + analysis_run.tex.")
    print("=" * 72 + "\n")

    order: List[Tuple[str, str, str, str, str, str]] = []
    for model_slug in ("qwen15b", "llama8b", "qwen14b"):
        for ds_slug, repo in DATASETS:
            model_arg, lens = MODEL_SPECS[model_slug][ds_slug]
            folder = f"{model_slug}_{ds_slug}"
            order.append((folder, model_slug, ds_slug, repo, model_arg, lens))

    records: List[Dict[str, Any]] = []

    for i, (folder, model_slug, ds_slug, repo, model_arg, lens) in enumerate(order):
        slug = folder
        print(f"\n>>> starting new run {model_slug}+{ds_slug} ({i+1}/12) → {folder}/")
        run_dir = root / folder
        run_dir.mkdir(parents=True, exist_ok=True)

        hf_try = TARGET_SAMPLES
        cmd = build_cmd(run_dir, repo, model_arg, lens, hf_try, JUDGE_CAP)
        code, out, err = run_one_subprocess(cmd)
        combined = out + "\n" + err

        if code != 0:
            n_avail = parse_only_n(combined)
            if n_avail is not None and n_avail < hf_try:
                print(f"    retry with hf-samples={n_avail} (fewer non-NEUTRAL rows)")
                cmd2 = replace_hf_samples(cmd, n_avail)
                code, out, err = run_one_subprocess(cmd2)
                combined = out + "\n" + err
                hf_try = n_avail

        ok = code == 0
        if not ok:
            print(f"FAILED {slug} rc={code}")
            (run_dir / "batch_error.txt").write_text(combined[-12000:], encoding="utf-8")
            write_analysis_run_tex(run_dir, slug, model_slug, ds_slug, hf_try, combined)
            records.append(
                {
                    "ok": False,
                    "slug": slug,
                    "model_tex": MODEL_LABEL_TEX[model_slug],
                    "ds_tex": DATASET_LABEL_TEX[ds_slug],
                }
            )
        else:
            print(f"finished run {model_slug}+{ds_slug}")
            write_analysis_run_tex(run_dir, slug, model_slug, ds_slug, hf_try, combined)
            js = run_dir / "judge_logitlens_summary.json"
            summ = {}
            if js.is_file():
                with open(js, "r", encoding="utf-8") as f:
                    data = json.load(f)
                summ = (data.get("judge_report") or {}).get("summary") or {}
            records.append(
                {
                    "ok": True,
                    "slug": slug,
                    "model_tex": MODEL_LABEL_TEX[model_slug],
                    "ds_tex": DATASET_LABEL_TEX[ds_slug],
                    "hf_used": hf_try,
                    "n_judge": int(summ.get("valid_count", summ.get("count", 0))),
                    "steer": float(summ.get("judge_steering_toward_target_pct", 0.0)),
                    "coh": float(summ.get("judge_coherence_success_pct", 0.0)),
                    "conf": float(summ.get("mean_confidence", 0.0)),
                }
            )

    write_master_summary(root, records)
    print(f"\nBatch root: {root}")


if __name__ == "__main__":
    main()
