"""Pass/fail criteria for correct-trace ablations."""

from __future__ import annotations

from pipeline import metric_at_k

METRICS_RFT_SHOULD_WIN = ("persistence", "mean_self_transition", "delta_r2", "K_eff")


def evaluate_compare_summaries(
    base_summaries: dict[str, dict],
    rft_summaries: dict[str, dict],
    k_focus: int,
    *,
    min_paired_gap_fraction: float = 0.5,
) -> dict:
    """
    Desired outcome:
      - On paired_correct: RFT > base on key metrics
      - paired_correct gap retains a substantial fraction of the all-trace gap
    """
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str, values: dict | None = None) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail, "values": values or {}})

    for subset in ("paired_correct", "all"):
        if subset not in base_summaries or subset not in rft_summaries:
            continue
        for metric in METRICS_RFT_SHOULD_WIN:
            b = metric_at_k(base_summaries[subset], k_focus, metric)
            r = metric_at_k(rft_summaries[subset], k_focus, metric)
            if b is None or r is None:
                add(f"rft_gt_base_{subset}_{metric}", False, f"missing K={k_focus}")
                continue
            add(
                f"rft_gt_base_{subset}_{metric}",
                r > b,
                f"{subset}: RFT={r:.4f} vs Base={b:.4f}",
                {"base": b, "rft": r, "subset": subset},
            )

    for metric in ("persistence", "delta_r2"):
        if not all(s in base_summaries for s in ("all", "paired_correct")):
            continue
        if not all(s in rft_summaries for s in ("all", "paired_correct")):
            continue
        gap_all = metric_at_k(rft_summaries["all"], k_focus, metric) - metric_at_k(
            base_summaries["all"], k_focus, metric
        )
        gap_paired = metric_at_k(rft_summaries["paired_correct"], k_focus, metric) - metric_at_k(
            base_summaries["paired_correct"], k_focus, metric
        )
        if gap_all <= 0:
            add(
                f"paired_retains_gap_{metric}",
                gap_paired > 0,
                f"all gap non-positive ({gap_all:.4f}); require paired gap>0 ({gap_paired:.4f})",
            )
        else:
            frac = gap_paired / gap_all if gap_all else 0.0
            add(
                f"paired_retains_gap_{metric}",
                gap_paired >= min_paired_gap_fraction * gap_all,
                f"paired_gap={gap_paired:.4f} all_gap={gap_all:.4f} frac={frac:.2f}",
                {"gap_paired": gap_paired, "gap_all": gap_all},
            )

    primary = [
        c for c in checks
        if "paired_correct" in c["name"] and c["name"].startswith("rft_gt_base")
    ]
    primary_pass = all(c["passed"] for c in primary) if primary else False
    all_passed = all(c["passed"] for c in checks)

    return {
        "k_focus": k_focus,
        "checks": checks,
        "all_passed": all_passed,
        "primary_pass": primary_pass,
        "desired_outcome": (
            "RFT > Base on paired_correct; paired RFT−Base gap retains "
            f"≥{min_paired_gap_fraction:.0%} of all-trace gap (persistence, ΔR²)"
        ),
    }
