"""
Bidirectional SDS transplant

This script supports fitting one SDS per model-family split (reasoning/base),
then evaluating each fitted SDS on both reasoning and base target trajectories.
The resulting 2x2 matrix of regime R^2 values mirrors a "Table 1"-style
native-vs-transplant comparison:
  - reasoning SDS -> reasoning target   (native)
  - reasoning SDS -> base target        (transplant)
  - base SDS -> base target             (native)
  - base SDS -> reasoning target        (transplant)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import transplant_sds_into_base as base_mod


def _clone_rows_with_uid(rows: List[dict], *, direction: str, run_id: str) -> List[dict]:
    out: List[dict] = []
    for i, row in enumerate(rows):
        pid = int(row.get("problem_id", -1))
        sidx = int(row.get("sentence_idx", -1))
        uid = f"{run_id}::{direction}::pid{pid}::sid{sidx}::row{i}"
        r = dict(row)
        r["transplant_input_uid"] = uid
        out.append(r)
    return out


def _fit_source_bundle(
    *,
    source_key: str,
    args: argparse.Namespace,
    out_dir: str,
    source_rows: List[dict],
    run_id: str,
) -> Dict[str, object]:
    source_rows = _clone_rows_with_uid(
        source_rows, direction=f"{source_key}_source", run_id=run_id
    )
    if not source_rows:
        raise RuntimeError(f"{source_key}: no source rows after filtering.")
    source_subset_path = os.path.join(out_dir, f"{source_key}_source_subset.pkl")
    base_mod._save_pickle_list(source_subset_path, source_rows)
    z_source, _pca_source, scaler, pca, encoder = base_mod._train_cebra_source(
        source_subset_path,
        limit_problems=10**9 if args.all_problems else int(args.limit_problems),
        max_triplets_per_pid=int(args.max_triplets_per_pid),
        cebra_dim=int(args.cebra_dim),
        cebra_epochs=int(args.cebra_epochs),
    )
    source_grouped = base_mod._group_sequences(source_rows, min_len=3)
    if not source_grouped.idx_seqs:
        raise RuntimeError(f"{source_key}: source has no trajectories of length >= 3.")
    z_source_seqs = base_mod._build_sequences_from_rows(z_source, source_grouped.idx_seqs)
    pi, a, d_m, d_b, d_cov = base_mod.cebra_mod.fit_slds_em_iters(
        z_source_seqs,
        int(args.k_regimes),
        int(args.cebra_dim),
        int(args.em_iters),
    )
    return {
        "source_key": source_key,
        "source_rows": source_rows,
        "source_subset_path": source_subset_path,
        "source_trajectories": int(len(z_source_seqs)),
        "scaler": scaler,
        "pca": pca,
        "encoder": encoder,
        "pi": pi,
        "a": a,
        "d_m": d_m,
        "d_b": d_b,
        "d_cov": d_cov,
    }


def _evaluate_bundle_on_target(
    *,
    bundle: Dict[str, object],
    target_key: str,
    target_rows: List[dict],
    args: argparse.Namespace,
    out_dir: str,
    run_id: str,
) -> Dict[str, object]:
    eval_key = f"{bundle['source_key']}_sds_on_{target_key}"
    target_rows = _clone_rows_with_uid(target_rows, direction=eval_key, run_id=run_id)
    if not target_rows:
        raise RuntimeError(f"{eval_key}: no target rows after filtering.")
    target_subset_path = os.path.join(out_dir, f"{eval_key}_target_subset.pkl")
    base_mod._save_pickle_list(target_subset_path, target_rows)
    z_target, pca_target = base_mod.cebra_mod.embed_cebra_with_fitted_transforms(
        target_rows, bundle["scaler"], bundle["pca"], bundle["encoder"]
    )
    target_grouped = base_mod._group_sequences(target_rows, min_len=3)
    if not target_grouped.idx_seqs:
        raise RuntimeError(f"{eval_key}: target has no trajectories of length >= 3.")
    z_target_seqs = base_mod._build_sequences_from_rows(z_target, target_grouped.idx_seqs)
    pca_target_seqs = base_mod._build_sequences_from_rows(pca_target, target_grouped.idx_seqs)
    state_seqs_target, lls_target = base_mod.cebra_mod.infer_gamma_argmax_states(
        z_target_seqs,
        bundle["pi"],
        bundle["a"],
        bundle["d_m"],
        bundle["d_b"],
        bundle["d_cov"],
        int(args.k_regimes),
    )

    ar_r2 = float(base_mod.cebra_mod.linear_ar_r2(pca_target_seqs))
    regime_r2 = float(base_mod.cebra_mod.regime_r2_on_pca(state_seqs_target, pca_target_seqs))
    delta_r2 = float(regime_r2 - ar_r2)
    nll = base_mod.cebra_mod.slds_nll_per_transition(lls_target, z_target_seqs)
    by_traj = base_mod._collect_pred_true_by_traj(
        pca_target_seqs=pca_target_seqs,
        state_seqs_target=state_seqs_target,
        pca_dim=int(args.cebra_dim),
    )
    delta_r2_boot = base_mod._bootstrap_delta_r2(
        by_traj,
        n_boot=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    timing = base_mod._bootstrap_timing_lift(
        target_features=target_rows,
        idx_seqs=target_grouped.idx_seqs,
        state_seqs_target=state_seqs_target,
        n_boot=int(args.n_bootstrap),
        seed=int(args.seed) + 17,
    )
    timing_judge = base_mod._judge_timing_predictions(
        target_features=target_rows,
        idx_seqs=target_grouped.idx_seqs,
        state_seqs_target=state_seqs_target,
        judge_samples=int(args.judge_samples),
        judge_model=str(args.judge_model),
        seed=int(args.seed) + 31,
    )

    result: Dict[str, object] = {
        "evaluation_key": eval_key,
        "source_sds": str(bundle["source_key"]),
        "target_domain": target_key,
        "is_native_fit": bool(bundle["source_key"] == target_key),
        "source_features_subset": str(bundle["source_subset_path"]),
        "target_features_subset": target_subset_path,
        "counts": {
            "source_rows": int(len(bundle["source_rows"])),
            "target_rows": int(len(target_rows)),
            "source_trajectories": int(bundle["source_trajectories"]),
            "target_trajectories": int(len(z_target_seqs)),
            "target_transitions": int(sum(max(0, len(s) - 1) for s in z_target_seqs)),
        },
        "metrics": {
            "linear_ar_r2": ar_r2,
            "regime_r2": regime_r2,
            "delta_r2": delta_r2,
            "slds_target_nll_per_transition": nll,
            "delta_r2_bootstrap": delta_r2_boot,
            "timing_metrics": timing,
            "timing_judge": timing_judge,
        },
    }
    out_json = os.path.join(out_dir, f"{eval_key}_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["summary_json"] = out_json
    return result


def _build_section_5_1_like_analysis(
    table: Dict[str, Dict[str, Dict[str, object]]],
) -> Dict[str, object]:
    rr = float(table["reasoning"]["reasoning"]["regime_r2"])
    rb = float(table["reasoning"]["base"]["regime_r2"])
    bb = float(table["base"]["base"]["regime_r2"])
    br = float(table["base"]["reasoning"]["regime_r2"])
    rr_gap = float(rr - rb)
    bb_gap = float(bb - br)
    rr_rel_drop = float(rr_gap / max(1e-12, abs(rr)))
    bb_rel_drop = float(bb_gap / max(1e-12, abs(bb)))
    avg_cross = float((rb + br) / 2.0)
    avg_native = float((rr + bb) / 2.0)

    if avg_cross < avg_native:
        headline = "Cross-model transplant underperforms native SDS fits on average."
    elif avg_cross > avg_native:
        headline = "Cross-model transplant outperforms native SDS fits on average."
    else:
        headline = "Cross-model and native SDS fits are tied on average."

    if rr_gap > bb_gap:
        asym = "Reasoning SDS is less portable to base than base SDS is to reasoning."
    elif rr_gap < bb_gap:
        asym = "Base SDS is less portable to reasoning than reasoning SDS is to base."
    else:
        asym = "Both SDS directions show matched portability gaps."

    return {
        "table_1_like_regime_r2": {
            "source_reasoning": {
                "target_reasoning_native_r2": rr,
                "target_base_transplant_r2": rb,
                "native_minus_cross_gap": rr_gap,
                "relative_drop_from_native": rr_rel_drop,
            },
            "source_base": {
                "target_base_native_r2": bb,
                "target_reasoning_transplant_r2": br,
                "native_minus_cross_gap": bb_gap,
                "relative_drop_from_native": bb_rel_drop,
            },
        },
        "aggregate": {
            "avg_native_r2": avg_native,
            "avg_cross_r2": avg_cross,
            "cross_minus_native": float(avg_cross - avg_native),
        },
        "section_5_1_style_interpretation": [headline, asym],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run SDS transplant in one or both directions and compare outcomes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math500", "mmlu-pro", "mmlu_pro", "svamp"],
    )
    ap.add_argument("--dataset-repo", type=str, default=None)
    ap.add_argument("--reasoning-relpath", type=str, default=None)
    ap.add_argument("--base-relpath", type=str, default=None)
    ap.add_argument("--reasoning-features-pkl", type=str, default=None)
    ap.add_argument("--base-features-pkl", type=str, default=None)
    ap.add_argument("--model-size", type=str, default="1.5b", choices=["1.5b", "14b", "llama8b"])
    ap.add_argument(
        "--run-directions",
        type=str,
        default="both",
        choices=["both", "reasoning_to_base", "base_to_reasoning", "all_pairs"],
    )
    ap.add_argument("--limit-problems", type=int, default=8)
    ap.add_argument("--all-problems", action="store_true")
    ap.add_argument("--max-triplets-per-pid", type=int, default=10)
    ap.add_argument("--cebra-dim", type=int, default=40)
    ap.add_argument("--cebra-epochs", type=int, default=3)
    ap.add_argument("--k-regimes", type=int, default=4)
    ap.add_argument("--em-iters", type=int, default=3)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--judge-samples", type=int, default=0)
    ap.add_argument("--judge-model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="transplant_sds_artifacts")
    ap.add_argument("--out-name", type=str, default="bidirectional_transplant_summary.json")
    ap.add_argument(
        "--only-problem-ids",
        type=str,
        default=None,
        help="Comma-separated problem_id ints to keep (after non-NEUTRAL filter). Both reasoning and base rows are restricted to this set.",
    )
    args = ap.parse_args()

    base_mod._seed_everything(int(args.seed))
    default_repo, default_reasoning_rel, default_base_rel = base_mod._dataset_defaults(args.dataset, args.model_size)
    dataset_repo = args.dataset_repo or default_repo
    reasoning_rel = args.reasoning_relpath or default_reasoning_rel
    base_rel = args.base_relpath or default_base_rel

    token = os.environ.get("HF_TOKEN", "").strip() or None
    reasoning_path = os.path.abspath(args.reasoning_features_pkl) if args.reasoning_features_pkl else base_mod._download_features(dataset_repo, reasoning_rel, token)
    base_path = os.path.abspath(args.base_features_pkl) if args.base_features_pkl else base_mod._download_features(dataset_repo, base_rel, token)

    limit_problems: Optional[int] = None if args.all_problems else int(args.limit_problems)
    reasoning_rows = base_mod._subset_non_neutral_by_problem(base_mod._load_pickle_list(reasoning_path), limit_problems)
    base_rows = base_mod._subset_non_neutral_by_problem(base_mod._load_pickle_list(base_path), limit_problems)
    if args.only_problem_ids:
        allowed = {int(x.strip()) for x in str(args.only_problem_ids).split(",") if x.strip()}
        reasoning_rows = [r for r in reasoning_rows if int(r["problem_id"]) in allowed]
        base_rows = [r for r in base_rows if int(r["problem_id"]) in allowed]

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    run_id = f"bi_{args.dataset}_{args.model_size.replace('.', '')}_{stamp}"
    out_dir = os.path.abspath(os.path.join(args.out_dir, run_id))
    os.makedirs(out_dir, exist_ok=True)

    bundles = {
        "reasoning": _fit_source_bundle(
            source_key="reasoning",
            args=args,
            out_dir=out_dir,
            source_rows=reasoning_rows,
            run_id=run_id,
        ),
        "base": _fit_source_bundle(
            source_key="base",
            args=args,
            out_dir=out_dir,
            source_rows=base_rows,
            run_id=run_id,
        ),
    }
    eval_plan = []
    if args.run_directions in {"both", "all_pairs"}:
        eval_plan = [
            ("reasoning", "reasoning"),
            ("reasoning", "base"),
            ("base", "base"),
            ("base", "reasoning"),
        ]
    elif args.run_directions == "reasoning_to_base":
        eval_plan = [("reasoning", "base")]
    else:
        eval_plan = [("base", "reasoning")]

    domain_rows = {"reasoning": reasoning_rows, "base": base_rows}
    outputs: Dict[str, Dict[str, object]] = {}
    for source_key, target_key in eval_plan:
        out = _evaluate_bundle_on_target(
            bundle=bundles[source_key],
            target_key=target_key,
            target_rows=domain_rows[target_key],
            args=args,
            out_dir=out_dir,
            run_id=run_id,
        )
        outputs[out["evaluation_key"]] = out

    comparison = None
    if (
        "reasoning_sds_on_reasoning" in outputs
        and "reasoning_sds_on_base" in outputs
        and "base_sds_on_base" in outputs
        and "base_sds_on_reasoning" in outputs
    ):
        table = {
            "reasoning": {
                "reasoning": outputs["reasoning_sds_on_reasoning"]["metrics"],
                "base": outputs["reasoning_sds_on_base"]["metrics"],
            },
            "base": {
                "reasoning": outputs["base_sds_on_reasoning"]["metrics"],
                "base": outputs["base_sds_on_base"]["metrics"],
            },
        }
        comparison = _build_section_5_1_like_analysis(table)
        with open(os.path.join(out_dir, "comparison_interpretation.json"), "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

    summary = {
        "mode": "cebra_em_transfer_bidirectional",
        "run_id": run_id,
        "run_dir": out_dir,
        "dataset": args.dataset,
        "dataset_repo": dataset_repo,
        "config": {
            "model_size": args.model_size,
            "only_problem_ids": args.only_problem_ids,
            "run_directions": args.run_directions,
            "all_problems": bool(args.all_problems),
            "limit_problems": None if args.all_problems else int(args.limit_problems),
            "max_triplets_per_pid": int(args.max_triplets_per_pid),
            "cebra_dim": int(args.cebra_dim),
            "cebra_epochs": int(args.cebra_epochs),
            "k_regimes": int(args.k_regimes),
            "em_iters": int(args.em_iters),
            "n_bootstrap": int(args.n_bootstrap),
            "judge_samples": int(args.judge_samples),
            "judge_model": str(args.judge_model),
            "seed": int(args.seed),
        },
        "reasoning_source_relpath": reasoning_rel,
        "base_source_relpath": base_rel,
        "reasoning_features_original": reasoning_path,
        "base_features_original": base_path,
        "outputs": outputs,
        "comparison_interpretation": comparison,
    }
    out_path = os.path.join(out_dir, args.out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Bidirectional SDS Transplant Evaluation ===")
    for d in sorted(outputs):
        m = outputs[d]["metrics"]
        print(
            f"{d}: regime_r2={m['regime_r2']:.6f}, "
            f"linear_ar_r2={m['linear_ar_r2']:.6f}, "
            f"delta_r2={m['delta_r2']:.6f}, "
            f"nll={m['slds_target_nll_per_transition']:.6f}, "
            f"timing_lift={m['timing_metrics']['timing_lift']:.6f}"
        )
    if comparison is not None:
        print("Comparison saved:", os.path.join(out_dir, "comparison_interpretation.json"))
    print("Saved:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
