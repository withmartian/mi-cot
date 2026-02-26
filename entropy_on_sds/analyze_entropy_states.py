"""
Analyze entropy across SDS states: by regime, at transitions, by transition type,
stage x regime heatmap, and trajectory plots.
Expects sentence_entropy.pkl and sds_states.pkl (and optionally features for stage).
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def load_data(out_dir: str, features_path: str = None):
    """Load sentence entropy, SDS states, and optionally features for stage."""
    with open(os.path.join(out_dir, "sentence_entropy.pkl"), "rb") as f:
        entropy_table = pickle.load(f)
    with open(os.path.join(out_dir, "sds_states.pkl"), "rb") as f:
        sds_data = pickle.load(f)
    states = sds_data["states"]
    K = sds_data["K"]

    # Build dataframe in same order (all tables are aligned by index)
    df = pd.DataFrame(entropy_table)
    df["state"] = states

    # is_transition: state != previous state within same problem
    df["is_transition"] = False
    prev_state = None
    prev_pid = None
    for i in range(len(df)):
        pid = df.loc[i, "problem_id"]
        s = df.loc[i, "state"]
        if prev_pid == pid and prev_state is not None and s != prev_state:
            df.loc[i, "is_transition"] = True
        prev_state = s
        prev_pid = pid

    # Add stage from features if available
    if features_path and os.path.exists(features_path):
        all_features = pickle.load(open(features_path, "rb"))
        if len(all_features) == len(df):
            df["stage"] = [f.get("stage", "UNK") for f in all_features]
        else:
            df["stage"] = "UNK"
    else:
        df["stage"] = "UNK"

    return df, K


def transition_matrix_and_entropy(df: pd.DataFrame, K: int):
    """Transition matrix and per-regime mean entropy (for Exp 4)."""
    trans = np.zeros((K, K))
    entropy_by_regime = defaultdict(list)
    for _, row in df.iterrows():
        entropy_by_regime[row["state"]].append(row["sent_entropy_mean"])
    # transition counts
    for pid in df["problem_id"].unique():
        sub = df[df["problem_id"] == pid].sort_values("sentence_idx")
        states = sub["state"].values
        for i in range(len(states) - 1):
            trans[int(states[i]), int(states[i + 1])] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_prob = np.divide(trans, row_sums, where=row_sums != 0, out=np.zeros_like(trans))
    # transition entropy per regime (row entropy)
    trans_entropy = np.array([
        -np.sum(row[row > 0] * np.log2(row[row > 0])) for row in trans_prob
    ])
    mean_sent_entropy_by_regime = np.array([
        np.nanmean(entropy_by_regime[k]) if entropy_by_regime[k] else np.nan for k in range(K)
    ])
    return trans_prob, trans_entropy, mean_sent_entropy_by_regime


def run_experiments(df: pd.DataFrame, K: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available; skipping plots")
        plt = None
        sns = None

    # Filter valid entropy for stats
    valid = df["sent_entropy_mean"].notna() & np.isfinite(df["sent_entropy_mean"])

    # ----- Exp 1: Entropy by regime -----
    summary_by_state = df[valid].groupby("state")["sent_entropy_mean"].agg(["mean", "median", "std", "count"])
    summary_by_state.to_csv(os.path.join(out_dir, "exp1_entropy_by_regime.csv"))
    print("Exp 1: Entropy by regime")
    print(summary_by_state.to_string())

    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        df[valid].boxplot(column="sent_entropy_mean", by="state", ax=ax)
        ax.set_xlabel("Regime")
        ax.set_ylabel("Sentence entropy (mean over tokens)")
        ax.set_title("Entropy by SDS regime")
        plt.suptitle("")
        plt.savefig(os.path.join(out_dir, "exp1_entropy_by_regime.png"), dpi=150)
        plt.close()

    # ----- Exp 2: Entropy at transition vs non-transition -----
    trans_ent = df.loc[df["is_transition"] & valid, "sent_entropy_mean"].mean()
    non_trans_ent = df.loc[~df["is_transition"] & valid, "sent_entropy_mean"].mean()
    exp2 = {"mean_entropy_at_transition": trans_ent, "mean_entropy_non_transition": non_trans_ent}
    with open(os.path.join(out_dir, "exp2_transition_vs_non.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in exp2.items():
            f.write(f"{k},{v}\n")
    print("Exp 2: Mean entropy at transition vs non-transition:", exp2)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Transition", "Non-transition"], [trans_ent, non_trans_ent], color=["#e74c3c", "#3498db"], alpha=0.8)
        ax.set_ylabel("Mean sentence entropy")
        ax.set_title("Entropy at state transition vs same-state")
        plt.savefig(os.path.join(out_dir, "exp2_transition_vs_non.png"), dpi=150)
        plt.close()

    # ----- Exp 3: Mean entropy by transition type (i -> j) -----
    # For each transition (s_{t-1}, s_t), take entropy at step t (the sentence in state j)
    trans_type_entropy = defaultdict(list)
    for pid in df["problem_id"].unique():
        sub = df[df["problem_id"] == pid].sort_values("sentence_idx")
        for i in range(1, len(sub)):
            row = sub.iloc[i]
            key = (int(sub.iloc[i - 1]["state"]), int(row["state"]))
            if np.isfinite(row["sent_entropy_mean"]):
                trans_type_entropy[key].append(row["sent_entropy_mean"])
    exp3_rows = []
    for (i, j), vals in sorted(trans_type_entropy.items()):
        exp3_rows.append({"from_state": i, "to_state": j, "mean_entropy": np.mean(vals), "count": len(vals)})
    pd.DataFrame(exp3_rows).to_csv(os.path.join(out_dir, "exp3_entropy_by_transition_type.csv"), index=False)

    if plt is not None and exp3_rows:
        exp3_df = pd.DataFrame(exp3_rows)
        pivot = exp3_df.pivot(index="from_state", columns="to_state", values="mean_entropy")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", ax=ax, cmap="YlOrRd")
        ax.set_title("Mean sentence entropy by transition (from → to)")
        ax.set_xlabel("To regime")
        ax.set_ylabel("From regime")
        plt.savefig(os.path.join(out_dir, "exp3_entropy_by_transition_type.png"), dpi=150)
        plt.close()

    # ----- Exp 4: SDS transition entropy vs mean sentence entropy per regime -----
    trans_prob, trans_entropy, mean_sent_entropy = transition_matrix_and_entropy(df, K)
    exp4 = pd.DataFrame({
        "regime": range(K),
        "transition_entropy_bits": trans_entropy,
        "mean_sentence_entropy": mean_sent_entropy,
    })
    exp4.to_csv(os.path.join(out_dir, "exp4_regime_transition_entropy_vs_sentence_entropy.csv"), index=False)
    corr = np.corrcoef(trans_entropy, mean_sent_entropy)[0, 1] if K > 1 else float("nan")
    print("Exp 4: Correlation (transition entropy vs mean sentence entropy across regimes):", corr)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(trans_entropy, mean_sent_entropy, s=80)
        for k in range(K):
            ax.annotate(f"R{k}", (trans_entropy[k], mean_sent_entropy[k]), fontsize=10)
        ax.set_xlabel("SDS transition entropy (bits)")
        ax.set_ylabel("Mean sentence entropy in regime")
        ax.set_title("Regime: transition entropy vs sentence entropy")
        plt.savefig(os.path.join(out_dir, "exp4_regime_entropy_correlation.png"), dpi=150)
        plt.close()

    # ----- Exp 5: Stage x regime heatmap (color = mean entropy) -----
    if "stage" in df.columns and df["stage"].nunique() > 1:
        stage_regime = df[valid].groupby(["stage", "state"])["sent_entropy_mean"].mean().unstack(fill_value=np.nan)
        stage_regime.to_csv(os.path.join(out_dir, "exp5_stage_regime_entropy.csv"))
        if plt is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(stage_regime, annot=True, fmt=".2f", ax=ax, cmap="viridis")
            ax.set_title("Mean sentence entropy: stage × regime")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "exp5_stage_regime_entropy.png"), dpi=150)
            plt.close()

    # ----- Exp 6: Trajectory plots (entropy + state over time) for a few problems -----
    if plt is not None:
        pids = df["problem_id"].unique()[:5]
        for pid in pids:
            sub = df[df["problem_id"] == pid].sort_values("sentence_idx").reset_index(drop=True)
            if len(sub) < 2:
                continue
            fig, ax1 = plt.subplots(figsize=(12, 4))
            x = np.arange(len(sub))
            ax1.plot(x, sub["sent_entropy_mean"].values, "b-o", label="Sentence entropy", markersize=4)
            ax1.set_xlabel("Sentence index")
            ax1.set_ylabel("Sentence entropy", color="b")
            ax2 = ax1.twinx()
            ax2.scatter(x, sub["state"].values, color="orange", s=20, alpha=0.7, label="State")
            ax2.set_ylabel("SDS state", color="orange")
            ax2.set_ylim(-0.5, K - 0.5)
            ax1.set_title(f"Problem {pid}: entropy and regime over time")
            fig.legend(loc="upper right")
            plt.savefig(os.path.join(out_dir, f"exp6_trajectory_pid_{pid}.png"), dpi=150)
            plt.close()

    # Summary
    def _convert(obj):
        if hasattr(obj, "item") and callable(obj.item):
            return obj.item()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return obj

    summary = {
        "n_sentences": len(df),
        "K": K,
        "exp1_entropy_by_regime": _convert(summary_by_state.to_dict()),
        "exp2": exp2,
        "exp4_correlation": float(corr),
    }
    import json
    with open(os.path.join(out_dir, "analysis_summary.json"), "w") as f:
        json.dump(_convert(summary), f, indent=2)
    print("Analysis complete. Outputs in", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Analyze entropy vs SDS states")
    parser.add_argument("--out", default="entropy_on_sds_output", help="Directory with sentence_entropy.pkl and sds_states.pkl")
    parser.add_argument("--features", default=None, help="Path to all_sentences_features.pkl for stage labels")
    args = parser.parse_args()

    # Resolve relative paths against cwd so --out entropy_on_sds_output works from entropy_on_sds/
    out_dir = os.path.abspath(args.out) if not os.path.isabs(args.out) else args.out
    features_path = args.features
    if features_path and not os.path.isabs(features_path):
        features_path = os.path.abspath(features_path) if not os.path.isabs(features_path) else features_path
    if not features_path:
        features_path = os.path.join(REPO_ROOT, "rpc_dataset", "all_sentences_features.pkl")

    df, K = load_data(out_dir, features_path=features_path)
    print(f"Loaded {len(df)} rows, K={K}")

    run_experiments(df, K, out_dir)


if __name__ == "__main__":
    main()
