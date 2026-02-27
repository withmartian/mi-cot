"""
Build an HTML dashboard from entropy_on_sds pipeline output.
Reads sentence_entropy.pkl, sds_states.pkl, analysis_summary.json, and exp PNGs
from the output directory and writes a single static HTML dashboard.

Usage:
  python build_entropy_sds_dashboard.py --out entropy_on_sds_output
  # Then open entropy_on_sds_output/dashboard_entropy_sds.html in a browser.
  # If images don't load, serve the folder: python -m http.server 8000 --directory entropy_on_sds_output
"""
import argparse
import json
import os
import pickle

import pandas as pd


def load_data(out_dir: str, features_path: str = None):
    """Load sentence entropy, SDS states, and optionally features for stage."""
    with open(os.path.join(out_dir, "sentence_entropy.pkl"), "rb") as f:
        entropy_table = pickle.load(f)
    with open(os.path.join(out_dir, "sds_states.pkl"), "rb") as f:
        sds_data = pickle.load(f)
    states = sds_data["states"]
    K = sds_data["K"]

    df = pd.DataFrame(entropy_table)
    df["state"] = states
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

    if features_path and os.path.exists(features_path):
        all_features = pickle.load(open(features_path, "rb"))
        if len(all_features) == len(df):
            df["stage"] = [f.get("stage", "UNK") for f in all_features]
        else:
            df["stage"] = "UNK"
    else:
        df["stage"] = "UNK"

    return df, K


def build_dashboard(out_dir: str, features_path: str = None) -> None:
    df, K = load_data(out_dir, features_path=features_path)

    # Per-problem entries for sidebar and detail view
    problems = []
    for pid in df["problem_id"].unique():
        sub = df[df["problem_id"] == pid].sort_values("sentence_idx")
        trajectory = []
        for _, row in sub.iterrows():
            trajectory.append({
                "sentence_idx": int(row["sentence_idx"]),
                "entropy_mean": round(float(row["sent_entropy_mean"]), 4) if pd.notna(row["sent_entropy_mean"]) else None,
                "entropy_last": round(float(row["sent_entropy_last"]), 4) if pd.notna(row["sent_entropy_last"]) else None,
                "state": int(row["state"]),
                "is_transition": bool(row["is_transition"]),
                "stage": str(row["stage"]),
            })
        trajectory_plot = f"exp6_trajectory_pid_{pid}.png"
        if not os.path.exists(os.path.join(out_dir, trajectory_plot)):
            trajectory_plot = None
        problems.append({
            "problem_id": int(pid),
            "n_sentences": len(trajectory),
            "trajectory": trajectory,
            "trajectory_plot": trajectory_plot,
            "mean_entropy": round(sub["sent_entropy_mean"].mean(), 4) if sub["sent_entropy_mean"].notna().any() else None,
        })

    # Summary stats
    summary_path = os.path.join(out_dir, "analysis_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = {"n_sentences": len(df), "K": K}

    # Which exp images exist
    exp_images = []
    for name in [
        "exp1_entropy_by_regime.png",
        "exp2_transition_vs_non.png",
        "exp3_entropy_by_transition_type.png",
        "exp4_regime_entropy_correlation.png",
        "exp5_stage_regime_entropy.png",
        "exp7_scatter_base_vs_finetuned.png",
        "exp7_entropy_reduction_by_regime.png",
    ]:
        if os.path.exists(os.path.join(out_dir, name)):
            exp_images.append({"name": name.replace(".png", "").replace("_", " ").title(), "file": name})

    index_data = {"problems": problems, "K": K, "n_sentences": len(df)}
    stats_data = {"summary": summary, "exp_images": exp_images}

    # Write JSON index/stats for optional fetch-based dashboard
    with open(os.path.join(out_dir, "dashboard_entropy_sds_index.json"), "w") as f:
        json.dump(index_data, f, indent=2)
    with open(os.path.join(out_dir, "dashboard_entropy_sds_stats.json"), "w") as f:
        json.dump(stats_data, f, indent=2)

    # Single-file static HTML with embedded data
    problems_json = json.dumps(problems)
    summary_json = json.dumps(summary)
    exp_images_json = json.dumps(exp_images)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Entropy on SDS Dashboard</title>
<style>
body {{ font-family: sans-serif; margin: 0; padding: 0; }}
.container {{ display: grid; grid-template-columns: 220px 1fr; min-height: 100vh; }}
.sidebar {{ border-right: 1px solid #ddd; padding: 12px; overflow-y: auto; background: #fafafa; }}
.sidebar h3 {{ margin: 0 0 10px 0; font-size: 14px; }}
.sidebar button {{ display: block; width: 100%; text-align: left; margin-bottom: 6px; padding: 8px 10px; border: 1px solid #ddd; border-radius: 4px; background: #fff; cursor: pointer; }}
.sidebar button:hover {{ background: #eef5ff; border-color: #aac7ff; }}
.sidebar button.active {{ background: #eef5ff; border-color: #357abd; font-weight: 600; }}
.content {{ padding: 16px; overflow-y: auto; }}
.section {{ margin-bottom: 20px; }}
.section h3 {{ margin-top: 0; }}
.plot {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
tr.transition {{ background: #fff8e6; }}
.legend {{ background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px; padding: 10px; margin-bottom: 12px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }}
.summary-card {{ background: #f7f7f7; border: 1px solid #ddd; border-radius: 4px; padding: 10px; }}
.exp-img {{ margin-bottom: 16px; }}
.exp-img img {{ max-width: 100%; }}
</style>
</head>
<body>
<div class="container">
  <div class="sidebar">
    <h3>Summary</h3>
    <button id="btn-summary" class="active">Overview</button>
    <h3>Problems</h3>
    <div id="sidebar-problems"></div>
  </div>
  <div class="content">
    <div id="panel-summary" class="section">
      <h2>Entropy on SDS – Overview</h2>
      <div class="legend" id="summary-legend">Loading…</div>
      <div class="section" id="summary-exp-images"></div>
    </div>
    <div id="panel-problem" class="section" style="display:none;">
      <h2 id="problem-title">Problem</h2>
      <div class="legend" id="problem-stats"></div>
      <div class="section">
        <img id="problem-trajectory-plot" class="plot" alt="Trajectory" style="max-width: 800px;"/>
      </div>
      <div class="section">
        <h3>Per-sentence entropy & state</h3>
        <table>
          <thead><tr><th>#</th><th>Entropy (mean)</th><th>Entropy (last)</th><th>State</th><th>Transition</th><th>Stage</th></tr></thead>
          <tbody id="problem-table"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>
<script>
const problems = {problems_json};
const summary = {summary_json};
const expImages = {exp_images_json};

function showSummary() {{
  document.getElementById('btn-summary').classList.add('active');
  document.querySelectorAll('.sidebar button:not(#btn-summary)').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-summary').style.display = 'block';
  document.getElementById('panel-problem').style.display = 'none';
  const el = document.getElementById('summary-legend');
  let exp7Text = '';
  if (summary.exp7_base_vs_finetuned && summary.exp7_base_vs_finetuned.mean_entropy_reduction_overall != null) {{
    exp7Text = 'Base vs fine-tuned: mean entropy reduction (base − finetuned) = ' + Number(summary.exp7_base_vs_finetuned.mean_entropy_reduction_overall).toFixed(4);
    if (summary.exp7_base_vs_finetuned.entropy_reduction_by_regime && summary.exp7_base_vs_finetuned.entropy_reduction_by_regime.length) {{
      exp7Text += ' | By regime: [' + summary.exp7_base_vs_finetuned.entropy_reduction_by_regime.map(function(x) {{ return Number(x).toFixed(3); }}).join(', ') + ']';
    }}
    exp7Text += '<br/>';
  }}
  el.innerHTML = 'K = ' + (summary.K || '') + ' regimes | n = ' + (summary.n_sentences || '') + ' sentences<br/>' +
    (summary.exp2 ? 'Mean entropy at transition: ' + Number(summary.exp2.mean_entropy_at_transition).toFixed(4) + '<br/>Mean entropy (non-transition): ' + Number(summary.exp2.mean_entropy_non_transition).toFixed(4) + '<br/>' : '') +
    (summary.exp4_correlation != null ? 'Correlation (transition entropy vs sentence entropy): ' + Number(summary.exp4_correlation).toFixed(4) + '<br/>' : '') +
    exp7Text;
  const container = document.getElementById('summary-exp-images');
  container.innerHTML = '';
  expImages.forEach(exp => {{
    const div = document.createElement('div');
    div.className = 'exp-img';
    div.innerHTML = '<h4>' + exp.name + '</h4><img src="' + exp.file + '" alt="' + exp.name + '"/>';
    container.appendChild(div);
  }});
}}

function showProblem(problem, btn) {{
  document.getElementById('btn-summary').classList.remove('active');
  document.querySelectorAll('.sidebar button').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  document.getElementById('panel-summary').style.display = 'none';
  document.getElementById('panel-problem').style.display = 'block';
  document.getElementById('problem-title').textContent = 'Problem ' + problem.problem_id;
  document.getElementById('problem-stats').textContent = problem.n_sentences + ' sentences | mean entropy: ' + (problem.mean_entropy != null ? problem.mean_entropy : '—');
  const img = document.getElementById('problem-trajectory-plot');
  if (problem.trajectory_plot) {{ img.src = problem.trajectory_plot; img.style.display = 'block'; }} else {{ img.style.display = 'none'; }}
  const tbody = document.getElementById('problem-table');
  tbody.innerHTML = '';
  problem.trajectory.forEach((row, i) => {{
    const tr = document.createElement('tr');
    if (row.is_transition) tr.classList.add('transition');
    tr.innerHTML = '<td>' + (i + 1) + '</td><td>' + (row.entropy_mean != null ? row.entropy_mean : '—') + '</td><td>' + (row.entropy_last != null ? row.entropy_last : '—') + '</td><td>R' + row.state + '</td><td>' + (row.is_transition ? 'yes' : '') + '</td><td>' + (row.stage || '') + '</td>';
    tbody.appendChild(tr);
  }});
}}

// Sidebar: Summary button
document.getElementById('btn-summary').addEventListener('click', () => showSummary());

// Sidebar: problem buttons
const container = document.getElementById('sidebar-problems');
problems.forEach((p, idx) => {{
  const btn = document.createElement('button');
  btn.textContent = 'Problem ' + p.problem_id + ' (' + p.n_sentences + ')';
  btn.addEventListener('click', () => showProblem(p, btn));
  container.appendChild(btn);
}});

showSummary();
</script>
</body>
</html>
"""
    out_path = os.path.join(out_dir, "dashboard_entropy_sds.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote dashboard to {out_path}")
    print(f"  Index: {os.path.join(out_dir, 'dashboard_entropy_sds_index.json')}")
    print(f"  Stats: {os.path.join(out_dir, 'dashboard_entropy_sds_stats.json')}")
    print("  Open the HTML in a browser. If images do not load, run: python -m http.server 8000 --directory " + os.path.abspath(out_dir))


def main():
    parser = argparse.ArgumentParser(description="Build HTML dashboard from entropy_on_sds output")
    parser.add_argument("--out", default="entropy_on_sds_output", help="Directory containing sentence_entropy.pkl, sds_states.pkl, etc.")
    parser.add_argument("--features", default=None, help="Path to all_sentences_features.pkl for stage labels")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out) if not os.path.isabs(args.out) else args.out
    if not os.path.exists(os.path.join(out_dir, "sentence_entropy.pkl")):
        raise SystemExit("Missing sentence_entropy.pkl in " + out_dir + ". Run the entropy_on_sds pipeline first.")
    features_path = os.path.abspath(args.features) if args.features and not os.path.isabs(args.features) else args.features
    build_dashboard(out_dir, features_path=features_path)


if __name__ == "__main__":
    main()
