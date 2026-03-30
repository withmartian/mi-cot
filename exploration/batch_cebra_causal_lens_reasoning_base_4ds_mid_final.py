#!/usr/bin/env python3
"""
Layer-group sweep for input-dependent CEBRA-EM + causal logit lens.

Runs both reasoning/base counterparts for:
  - qwen15b
  - llama8b
  - qwen14b
across:
  - gsm8k
  - math500
  - mmlu-pro
  - svamp
for BOTH layer groups:
  - middle: qwen15b=20, llama8b=22, qwen14b=28
  - final:  qwen15b=27, llama8b=31, qwen14b=47

Writes progressive LaTeX summary after each model+dataset+layer_group combo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "cebra_em_steering_inputDep_logitlens_judge_causal_lens.py"

JUDGE_CAP = 120
TARGET_SAMPLES = 1000
EM_ITERS = 40
LENS_DEVICE = "auto"

DATASETS: List[Tuple[str, str]] = [
    ("gsm8k", "withmartian/SDS_train_gsm8k"),
    ("math500", "withmartian/SDS_math500_test"),
    ("mmlu-pro", "withmartian/SDS_train_mmlu-pro"),
    ("svamp", "withmartian/SDS_train_svamp"),
]

MODEL_PAIRS: Dict[str, Dict[str, str]] = {
    "qwen15b": {
        "reasoning_lens": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "base_lens": "Qwen/Qwen2.5-1.5B",
        "tex": "Qwen-1.5B / Qwen2.5-1.5B",
    },
    "llama8b": {
        "reasoning_lens": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "base_lens": "meta-llama/Llama-3.1-8B",
        "tex": "Llama-3.1-8B (reasoning/base)",
    },
    "qwen14b": {
        "reasoning_lens": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "base_lens": "Qwen/Qwen2.5-14B",
        "tex": "Qwen-14B / Qwen2.5-14B",
    },
}

LAYER_GROUPS: Dict[str, Dict[str, int]] = {
    "middle": {"qwen15b": 20, "llama8b": 22, "qwen14b": 28},
    "final": {"qwen15b": 27, "llama8b": 31, "qwen14b": 47},
}

MODEL_PREFIX_BY_DATASET: Dict[str, Dict[str, Dict[str, str]]] = {
    "gsm8k": {
        "qwen15b": {"reasoning": "qwen1.5b_reasoning", "base": "qwen_1.5B_base"},
        "llama8b": {"reasoning": "llama_8b_reasoning", "base": "llama_8B_base"},
        "qwen14b": {"reasoning": "qwen_14b_reasoning", "base": "qwen_14B_base"},
    },
    "mmlu-pro": {
        "qwen15b": {"reasoning": "qwen1.5b", "base": "qwen1.5b_base"},
        "llama8b": {"reasoning": "llama8b", "base": "llama8b_base"},
        "qwen14b": {"reasoning": "qwen14b", "base": "qwen14b_base"},
    },
    "svamp": {
        "qwen15b": {"reasoning": "Qwen_1_5B_reasoning", "base": "Qwen_1_5B_base"},
        "llama8b": {"reasoning": "Llama_8B_reasoning", "base": "Llama_8B_base"},
        "qwen14b": {"reasoning": "Qwen_14B_reasoning", "base": "Qwen_14B_base"},
    },
    "math500": {
        "qwen15b": {"reasoning": "Qwen_1_5B_reasoning", "base": "Qwen_1_5B_base"},
        "llama8b": {"reasoning": "Llama_8B_reasoning", "base": "Llama_8B_base"},
        "qwen14b": {"reasoning": "Qwen_14B_reasoning", "base": "Qwen_14B_base"},
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


def parse_only_n(output: str) -> Optional[int]:
    m = re.search(r"Only (\d+) non-NEUTRAL samples", output)
    return int(m.group(1)) if m else None


def _delete_pkls_under(path: Path) -> int:
    """Remove *.pkl under path (recursive). Returns bytes freed (best effort)."""
    freed = 0
    if not path.is_dir():
        return 0
    for p in path.rglob("*.pkl"):
        try:
            freed += p.stat().st_size
            p.unlink()
        except OSError:
            pass
    return freed


def _safe_write_batch_error(save_dir: Path, text: str) -> None:
    tail = text[-12000:] if text else ""
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "batch_error.txt").write_text(tail, encoding="utf-8")
    except OSError:
        try:
            fd, tmp = tempfile.mkstemp(prefix="batch_error_", suffix=".txt", text=True)
            os.close(fd)
            Path(tmp).write_text(tail, encoding="utf-8")
            print(f"[warn] Could not write batch_error.txt under {save_dir}; wrote {tmp}", flush=True)
        except OSError:
            print(f"[warn] batch_error (tail): {tail[:2000]}", flush=True)


def _try_load_completed_side(save_dir: Path) -> Optional[Dict[str, Any]]:
    js = save_dir / "judge_logitlens_summary.json"
    if not js.is_file():
        return None
    with open(js, "r", encoding="utf-8") as f:
        data = json.load(f)
    summ = (data.get("judge_report") or {}).get("summary") or {}
    return {
        "ok": True,
        "save_dir": str(save_dir),
        "n_judge": int(summ.get("valid_count", summ.get("count", 0))),
        "steer": float(summ.get("judge_steering_toward_target_pct", 0.0)),
        "coh": float(summ.get("judge_coherence_success_pct", 0.0)),
        "conf": float(summ.get("mean_confidence", 0.0)),
    }


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


def build_model_subpath(ds_slug: str, model_slug: str, side: str, layer: int) -> str:
    prefix = MODEL_PREFIX_BY_DATASET[ds_slug][model_slug][side]
    return f"{prefix}/layer_{layer}"


def build_cmd(
    save_dir: Path,
    dataset_repo: str,
    model_subpath: str,
    lens_id: str,
    hf_samples: int,
) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--dataset",
        dataset_repo,
        "--model",
        model_subpath,
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
        "--no-save-steer-artifacts-pkl",
    ]


def run_one_side(
    *,
    combo_dir: Path,
    dataset_repo: str,
    model_slug: str,
    ds_slug: str,
    layer_group: str,
    side: str,
    lens_id: str,
    resume: bool,
) -> Dict[str, Any]:
    layer = int(LAYER_GROUPS[layer_group][model_slug])
    model_subpath = build_model_subpath(ds_slug, model_slug, side, layer)
    save_dir = combo_dir / f"{model_slug}_{side}_{ds_slug}"
    save_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        done = _try_load_completed_side(save_dir)
        if done is not None:
            done["layer"] = layer
            done["model_subpath"] = model_subpath
            _delete_pkls_under(save_dir)
            return done

    n_try = TARGET_SAMPLES
    cmd = build_cmd(save_dir, dataset_repo, model_subpath, lens_id, n_try)
    code, combined = run_cmd(cmd)
    if code != 0:
        n_avail = parse_only_n(combined)
        if n_avail is not None and n_avail < n_try:
            cmd = build_cmd(save_dir, dataset_repo, model_subpath, lens_id, n_avail)
            code, combined = run_cmd(cmd)
            n_try = n_avail

    if code != 0:
        _safe_write_batch_error(save_dir, combined)
        _delete_pkls_under(save_dir)
        return {
            "ok": False,
            "save_dir": str(save_dir),
            "layer": layer,
            "model_subpath": model_subpath,
            "n_judge": 0,
            "steer": 0.0,
            "coh": 0.0,
            "conf": 0.0,
        }

    _delete_pkls_under(save_dir)

    js = save_dir / "judge_logitlens_summary.json"
    summ: Dict[str, Any] = {}
    if js.is_file():
        with open(js, "r", encoding="utf-8") as f:
            data = json.load(f)
        summ = (data.get("judge_report") or {}).get("summary") or {}

    return {
        "ok": True,
        "save_dir": str(save_dir),
        "layer": layer,
        "model_subpath": model_subpath,
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
        fam = _tex_escape(r["model_tex"])
        ds = _tex_escape(r["ds_tex"])
        lg = _tex_escape(r["layer_group"])
        if not rr.get("ok") or not bb.get("ok"):
            rows.append(f"{lg} & {fam} & {ds} & \\emph{{partial/fail}} & & & & & \\\\")
            continue
        ds_steer = float(bb["steer"]) - float(rr["steer"])
        ds_coh = float(bb["coh"]) - float(rr["coh"])
        rows.append(
            f"{lg} & {fam} & {ds} & {rr['steer']:.1f} & {bb['steer']:.1f} & {ds_steer:+.1f} & "
            f"{rr['coh']:.1f} & {bb['coh']:.1f} & {ds_coh:+.1f} \\\\"
        )
    body = "\n".join(rows)
    text = f"""\\label{{sec:causal-lens-mid-final-progress}}

\\paragraph{{Batch status.}}
Refreshed after each completed model+dataset+layer-group combo.
Protocol: \\texttt{{cebra\\_em\\_steering\\_inputDep\\_logitlens\\_judge\\_causal\\_lens.py}},
$K{{=}}4$, target samples up to {TARGET_SAMPLES}, judge cap {JUDGE_CAP}, EM iters {EM_ITERS}.
Layer groups: middle (20/22/28) and final (27/31/47).

\\begin{{table}}[t]
\\centering
\\scriptsize
\\caption{{Reasoning vs base by layer group (\\textbf{{progressive}}; $\\Delta$ = base minus reasoning).}}
\\begin{{tabular}}{{lllrrrrrr}}
\\hline
Group & Family & Dataset & Steer$_{{\\mathrm{{R}}}}$ & Steer$_{{\\mathrm{{B}}}}$ & $\\Delta$Steer & Coh$_{{\\mathrm{{R}}}}$ & Coh$_{{\\mathrm{{B}}}}$ & $\\Delta$Coh \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}

\\paragraph{{Output root.}} \\texttt{{{_tex_escape(root.name)}}}
"""
    (root / "master_reasoning_vs_base_mid_final_progress.tex").write_text(text, encoding="utf-8")


def write_final_analysis(root: Path, records: List[Dict[str, Any]]) -> None:
    good = [r for r in records if r.get("reasoning", {}).get("ok") and r.get("base", {}).get("ok")]
    if not good:
        (root / "master_reasoning_vs_base_mid_final_analysis.tex").write_text(
            "No completed comparable cells.\n", encoding="utf-8"
        )
        return

    per_group: Dict[str, Dict[str, int]] = {}
    lines: List[str] = []
    for r in good:
        lg = str(r["layer_group"])
        rr = r["reasoning"]
        bb = r["base"]
        ds = float(bb["steer"]) - float(rr["steer"])
        g = per_group.setdefault(lg, {"reasoning_higher": 0, "base_higher": 0, "tie": 0})
        if abs(ds) < 1.0:
            g["tie"] += 1
        elif ds > 0:
            g["base_higher"] += 1
        else:
            g["reasoning_higher"] += 1
        lines.append(
            f"{r['layer_group']} / {r['model_tex']} ({r['ds_tex']}): "
            f"Steer {rr['steer']:.1f} vs {bb['steer']:.1f}, Coh {rr['coh']:.1f} vs {bb['coh']:.1f}."
        )

    counts_txt = []
    for lg in ("middle", "final"):
        g = per_group.get(lg, {"reasoning_higher": 0, "base_higher": 0, "tie": 0})
        counts_txt.append(
            f"{lg}: reasoning higher={g['reasoning_higher']}, base higher={g['base_higher']}, tie={g['tie']}"
        )

    text = (
        "\\section*{Final analysis (middle + final layer sweep)}\n"
        f"Comparable cells: {len(good)}.\n\n"
        "\\paragraph{Steering direction counts by layer group.}\n"
        + "\n".join(f"- {x}" for x in counts_txt)
        + "\n\n"
        "\\paragraph{Per-cell notes.}\n"
        + "\n".join(f"- {x}" for x in lines)
        + "\n"
    )
    (root / "master_reasoning_vs_base_mid_final_analysis.tex").write_text(text, encoding="utf-8")


def main() -> None:
    if not SCRIPT.is_file():
        raise SystemExit(f"Missing script: {SCRIPT}")

    ap = argparse.ArgumentParser(description="Middle/final layer sweep with optional resume into existing root.")
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Existing batch output directory to resume (skip completed sides; reuses same root).",
    )
    ap.add_argument(
        "--no-clean-pkls-on-resume",
        action="store_true",
        help="Do not delete all .pkl under --root once at resume start (frees disk from prior crash).",
    )
    args = ap.parse_args()

    if args.root is not None:
        root = args.root.resolve()
        if not root.is_dir():
            raise SystemExit(f"--root is not a directory: {root}")
        resume = True
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = HERE / f"cebra_causal_lens_reasoning_base_4ds_mid_final_{stamp}"
        root.mkdir(parents=True, exist_ok=True)
        resume = False

    if resume and not args.no_clean_pkls_on_resume:
        freed = _delete_pkls_under(root)
        print(f"[resume] Removed .pkl caches under root (~{freed / 1e6:.1f} MB).", flush=True)

    print("=" * 72)
    print("STARTING MID+FINAL BATCH (2 groups × 3 models × 4 datasets × 2 sides = 48 runs)")
    print(f"Judge cap: {JUDGE_CAP}; target samples: {TARGET_SAMPLES}; EM iters: {EM_ITERS}")
    print(f"Resume: {resume}; root: {root}")
    print("Estimated wall time: ~12-24 hours total (sequential, API/network dependent).")
    print("=" * 72)

    records: List[Dict[str, Any]] = []
    total_combo = len(LAYER_GROUPS) * len(MODEL_PAIRS) * len(DATASETS)
    idx_combo = 0

    for layer_group in ("middle", "final"):
        for model_slug in ("qwen15b", "llama8b", "qwen14b"):
            spec = MODEL_PAIRS[model_slug]
            for ds_slug, dataset_repo in DATASETS:
                idx_combo += 1
                print(
                    f"starting [new run {layer_group}:{model_slug}+{ds_slug}] ({idx_combo}/{total_combo})",
                    flush=True,
                )
                combo_dir = root / f"{layer_group}_{model_slug}_{ds_slug}"
                combo_dir.mkdir(parents=True, exist_ok=True)
                rec: Dict[str, Any] = {
                    "layer_group": layer_group,
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
                    layer_group=layer_group,
                    side="reasoning",
                    lens_id=spec["reasoning_lens"],
                    resume=resume,
                )
                rec["base"] = run_one_side(
                    combo_dir=combo_dir,
                    dataset_repo=dataset_repo,
                    model_slug=model_slug,
                    ds_slug=ds_slug,
                    layer_group=layer_group,
                    side="base",
                    lens_id=spec["base_lens"],
                    resume=resume,
                )
                _delete_pkls_under(combo_dir)
                records.append(rec)
                write_progress_summary(root, records)
                print(f"finished run [{layer_group}:{model_slug}+{ds_slug}]", flush=True)

    write_final_analysis(root, records)
    print(f"Batch complete. Root: {root}", flush=True)


if __name__ == "__main__":
    main()

