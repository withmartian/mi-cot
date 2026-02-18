import argparse
import inspect
import json
import math
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_FILELOCK", "1")

import filelock

# Ensure compatibility with older filelock versions used by hf_hub.
if "mode" not in inspect.signature(filelock.FileLock.__init__).parameters:
    class FileLockCompat(filelock.FileLock):
        def __init__(self, lock_file, timeout=-1, **kwargs):
            super().__init__(lock_file, timeout=timeout)

    filelock.FileLock = FileLockCompat
    try:
        import huggingface_hub.utils._fixes as hf_fixes

        hf_fixes.FileLock = FileLockCompat
    except Exception:
        pass

import matplotlib.pyplot as plt
from transformers import AutoTokenizer


@dataclass
class ExampleData:
    question: str
    generated: str
    uncertainty: List[Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render uncertainty visuals from JSONL results."
    )
    parser.add_argument(
        "--input",
        default="uncertainty_results.jsonl",
        help="Path to JSONL output from track_uncertainty.py",
    )
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Which example to visualize (0-based)",
    )
    parser.add_argument(
        "--example-indices",
        default="",
        help="Comma-separated list of example indices to visualize",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all examples",
    )
    parser.add_argument(
        "--output-dir",
        default="uncertainty_plots",
        help="Directory to write plots/html",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to display in plots",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable writing the dashboard HTML output",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Only build the dashboard (assumes plots already exist)",
    )
    return parser.parse_args()


def load_results(path: str) -> Tuple[Dict[str, str], List[ExampleData]]:
    summary: Dict[str, str] = {}
    examples: List[ExampleData] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            payload = json.loads(line)
            if line_idx == 0 and "summary" in payload:
                summary = payload["summary"]
                continue
            examples.append(
                ExampleData(
                    question=payload.get("question", ""),
                    generated=payload["generated"],
                    uncertainty=payload["uncertainty"],
                )
            )
    return summary, examples


def normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    span = max(max_val - min_val, 1e-12)
    return [(val - min_val) / span for val in values]


def color_for_value(value: float) -> str:
    # Blue -> red gradient.
    r = int(255 * value)
    g = int(64 * (1 - value))
    b = int(255 * (1 - value))
    return f"rgb({r},{g},{b})"


def split_sentences(text: str) -> List[Tuple[str, int]]:
    sentences: List[Tuple[str, int]] = []
    last_end = 0
    for match in re.finditer(r"[.!?]", text):
        end = match.end()
        sentence = text[last_end:end].strip()
        if sentence:
            sentences.append((sentence, end))
        last_end = end
    remainder = text[last_end:].strip()
    if remainder:
        sentences.append((remainder, len(text)))
    return sentences


def build_prompt(question: str) -> str:
    return (
        "Solve the problem. Show your reasoning step by step and finish with "
        "'Final answer: <number>'.\n\n"
        f"Question: {question}\n\n"
        "Let's think step by step.\n"
    )


def map_sentence_ends_to_token_indices(
    offsets: List[Tuple[int, int]], ends: List[int]
) -> List[int]:
    indices = []
    for end in ends:
        idx = None
        for token_idx, (_, token_end) in enumerate(offsets):
            if token_end <= end:
                idx = token_idx
            else:
                break
        if idx is not None:
            indices.append(idx)
    return indices


def build_word_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


def map_tokens_to_words(
    word_spans: List[Tuple[int, int, str]],
    offsets: List[Tuple[int, int]],
    entropies: List[float],
) -> List[Tuple[str, float]]:
    word_values: List[Tuple[str, float]] = []
    for start, end, word in word_spans:
        token_values = []
        for (tok_start, tok_end), entropy in zip(offsets, entropies):
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            token_values.append(entropy)
        if token_values:
            word_values.append((word, sum(token_values) / len(token_values)))
        else:
            word_values.append((word, 0.0))
    return word_values


def is_wait_token(word: str) -> bool:
    return word.lstrip().startswith("Wait")


def build_token_values(
    text: str,
    offsets: List[Tuple[int, int]],
    entropies: List[float],
) -> List[Tuple[str, float]]:
    values: List[Tuple[str, float]] = []
    for (start, end), entropy in zip(offsets, entropies):
        token_text = text[start:end] or " "
        values.append((token_text, entropy))
    return values


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_heatmap_spans(values: List[Tuple[str, float]]) -> str:
    entropies = [value for _, value in values]
    normalized = normalize(entropies)
    parts = []
    for (token, entropy), norm in zip(values, normalized):
        color = color_for_value(norm)
        wait_class = " wait-token" if is_wait_token(token) else ""
        rendered = html_escape(token).replace(" ", "&nbsp;")
        parts.append(
            f"<span class='token{wait_class}' style='background:{color}'>"
            f"{rendered}<div class='entropy'>{entropy:.2f}</div></span>"
        )
    return "".join(parts)


def write_html_heatmap(
    word_values: List[Tuple[str, float]],
    token_values: List[Tuple[str, float]],
    output_path: str,
) -> None:
    word_parts = render_heatmap_spans(word_values)
    token_parts = render_heatmap_spans(token_values)
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
body {{
  font-family: sans-serif;
  line-height: 1.4;
}}
.toolbar {{
  margin-bottom: 10px;
}}
.heatmap {{
  white-space: pre-wrap;
}}
.token {{
  display: inline-block;
  margin: 2px 2px;
  padding: 2px 4px;
  border-radius: 3px;
}}
.wait-token {{
  border: 4px solid #ff8c00;
  box-shadow: 0 0 0 2px #ff8c00 inset;
}}
.entropy {{
  font-size: 0.6rem;
  text-align: center;
  color: #111;
}}
</style>
</head>
<body>
<div class="toolbar">
  <button onclick="showHeatmap('token')">Token</button>
  <button onclick="showHeatmap('word')">Word</button>
</div>
<div id="token-heatmap" class="heatmap">
{token_parts}
</div>
<div id="word-heatmap" class="heatmap" style="display:none;">
{word_parts}
</div>
<script>
function showHeatmap(kind) {{
  var token = document.getElementById('token-heatmap');
  var word = document.getElementById('word-heatmap');
  if (kind === 'word') {{
    token.style.display = 'none';
    word.style.display = 'block';
  }} else {{
    word.style.display = 'none';
    token.style.display = 'block';
  }}
}}
</script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def collect_example_assets(output_dir: str) -> List[Dict[str, str]]:
    assets: Dict[int, Dict[str, str]] = {}
    for name in os.listdir(output_dir):
        trace_match = re.match(r"(\d+)_uncertainty_trace\.html$", name)
        plot_match = re.match(r"(\d+)_uncertainty_timeseries\.png$", name)
        legend_match = re.match(r"(\d+)_uncertainty_timeseries_legend\.txt$", name)
        if trace_match:
            idx = int(trace_match.group(1))
            assets.setdefault(idx, {})["trace"] = name
        elif plot_match:
            idx = int(plot_match.group(1))
            assets.setdefault(idx, {})["plot"] = name
        elif legend_match:
            idx = int(legend_match.group(1))
            assets.setdefault(idx, {})["legend"] = name

    entries: List[Dict[str, str]] = []
    for idx in sorted(assets.keys()):
        item = {"index": str(idx)}
        item.update(assets[idx])
        legend_name = assets[idx].get("legend")
        if legend_name:
            legend_path = os.path.join(output_dir, legend_name)
            try:
                with open(legend_path, "r", encoding="utf-8") as f:
                    item["legend_text"] = f.read()
            except Exception:
                item["legend_text"] = ""
        else:
            item["legend_text"] = ""
        entries.append(item)
    return entries


def sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"


def write_dashboard_index(output_dir: str) -> str:
    entries = collect_example_assets(output_dir)
    output_path = os.path.join(output_dir, "dashboard_index.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return output_path


def write_dashboard_stats(
    output_dir: str,
    per_example_stats: Dict[int, Dict[str, float]],
    per_example_prompts: Dict[int, str],
    distribution_overall: List[float],
    distribution_wait_prev: List[float],
    mean_overall: float,
    mean_wait_prev: float,
    distribution_overall_wait: List[float],
    distribution_wait_prev_wait: List[float],
    mean_overall_wait: float,
    mean_wait_prev_wait: float,
) -> str:
    per_example_payload = {}
    for idx, stats in per_example_stats.items():
        per_example_payload[str(idx)] = {
            **stats,
            "prompt": per_example_prompts.get(idx, ""),
        }
    payload = {
        "overall_distribution": distribution_overall,
        "wait_prev_distribution": distribution_wait_prev,
        "overall_mean": mean_overall,
        "wait_prev_mean": mean_wait_prev,
        "overall_distribution_plot": "aggregate_overall_entropy.png",
        "wait_prev_distribution_plot": "aggregate_wait_prev_entropy.png",
        "overall_distribution_wait_plot": "aggregate_wait_overall_entropy.png",
        "wait_prev_distribution_wait_plot": "aggregate_wait_prev_entropy_wait.png",
        "overall_distribution_wait": distribution_overall_wait,
        "wait_prev_distribution_wait": distribution_wait_prev_wait,
        "overall_mean_wait": mean_overall_wait,
        "wait_prev_mean_wait": mean_wait_prev_wait,
        "per_example": per_example_payload,
    }
    output_path = os.path.join(output_dir, "dashboard_stats.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def plot_distribution(
    values: List[float],
    title: str,
    output_path: str,
) -> None:
    if not values:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=20, color="#4c78a8", alpha=0.8, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Average entropy")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_dashboard_html_static(
    output_dir: str,
    output_name: str,
    per_example_stats: Dict[int, Dict[str, float]],
    per_example_prompts: Dict[int, str],
    distribution_overall: List[float],
    distribution_wait_prev: List[float],
    mean_overall: float,
    mean_wait_prev: float,
    distribution_overall_wait: List[float],
    distribution_wait_prev_wait: List[float],
    mean_overall_wait: float,
    mean_wait_prev_wait: float,
) -> None:
    entries = collect_example_assets(output_dir)
    for entry in entries:
        try:
            idx = int(entry.get("index", "-1"))
        except ValueError:
            idx = -1
        stats = per_example_stats.get(idx, {})
        entry["overall_entropy_avg"] = stats.get("overall_entropy_avg", 0.0)
        entry["overall_entropy_count"] = stats.get("overall_entropy_count", 0)
        entry["wait_prev_entropy_avg"] = stats.get("wait_prev_entropy_avg", 0.0)
        entry["wait_prev_entropy_count"] = stats.get("wait_prev_entropy_count", 0)
        entry["prompt"] = per_example_prompts.get(idx, "")
    payload = json.dumps(entries)
    overall_payload = json.dumps(distribution_overall)
    wait_payload = json.dumps(distribution_wait_prev)
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
body {{
  font-family: sans-serif;
  margin: 0;
  padding: 0;
}}
.container {{
  display: grid;
  grid-template-columns: 240px 1fr;
  min-height: 100vh;
}}
.sidebar {{
  border-right: 1px solid #ddd;
  padding: 12px;
  overflow-y: auto;
}}
.sidebar button {{
  display: block;
  width: 100%;
  text-align: left;
  margin-bottom: 6px;
  padding: 6px 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fafafa;
  cursor: pointer;
}}
.sidebar button.active {{
  background: #eef5ff;
  border-color: #aac7ff;
}}
.content {{
  padding: 16px;
}}
.section {{
  margin-bottom: 16px;
}}
.tabs {{
  margin-bottom: 12px;
}}
.tabs button {{
  margin-right: 8px;
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fafafa;
  cursor: pointer;
}}
.tabs button.active {{
  background: #eef5ff;
  border-color: #aac7ff;
}}
.legend {{
  white-space: pre-wrap;
  background: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 10px;
}}
.plot {{
  max-width: 100%;
  border: 1px solid #eee;
}}
.trace {{
  width: 100%;
  height: 500px;
  border: 1px solid #ddd;
}}
</style>
</head>
<body>
<div class="container">
  <div class="sidebar">
    <div id="sidebar" class="section"></div>
  </div>
  <div class="content">
    <h2 id="title">Select an example</h2>
    <div id="tab-example-content">
      <div class="section">
        <h3>Prompt</h3>
        <div id="prompt" class="legend">No prompt available.</div>
      </div>
      <div class="section">
        <h3>Example stats</h3>
        <div class="legend">
          Average entropy (all tokens, excluding token before Wait): <span id="overall-avg">0.0000</span>
          (<span id="overall-count">0</span> tokens)
          <br/>
          Average entropy (token before Wait): <span id="wait-avg">0.0000</span>
          (<span id="wait-count">0</span> occurrences)
        </div>
      </div>
      <div class="section">
        <h3>Entropy Plot</h3>
        <img id="plot" class="plot" alt="Entropy plot"/>
      </div>
      <div class="section">
        <h3>Sentence legend (S# -> sentence)</h3>
        <div id="legend" class="legend">No legend available.</div>
      </div>
      <div class="section">
        <h3>Trace (HTML)</h3>
        <iframe id="trace" class="trace"></iframe>
      </div>
    </div>
    <div id="tab-dist-content" style="display:none;">
      <div class="section">
        <h3>Aggregate distributions (per-example averages)</h3>
        <div class="legend">
          Mean overall entropy (excluding token before Wait): {mean_overall:.4f}
          <br/>
          Mean entropy (token before Wait): {mean_wait_prev:.4f}
        </div>
      </div>
      <div class="section">
        <h4>Overall entropy averages</h4>
        <img id="dist-overall-img" class="plot" alt="Overall entropy distribution"/>
      </div>
      <div class="section">
        <h4>Token-before-Wait averages</h4>
        <img id="dist-wait-img" class="plot" alt="Wait entropy distribution"/>
      </div>
    </div>
    <div id="tab-dist-wait-content" style="display:none;">
      <div class="section">
        <h3>Aggregate (Wait-only examples)</h3>
        <div class="legend">
          Mean overall entropy (excluding token before Wait): {mean_overall_wait:.4f}
          <br/>
          Mean entropy (token before Wait): {mean_wait_prev_wait:.4f}
        </div>
      </div>
      <div class="section">
        <h4>Overall entropy averages (Wait-only)</h4>
        <img id="dist-overall-wait-img" class="plot" alt="Overall entropy distribution (Wait-only)"/>
      </div>
      <div class="section">
        <h4>Token-before-Wait averages (Wait-only)</h4>
        <img id="dist-wait-wait-img" class="plot" alt="Wait entropy distribution (Wait-only)"/>
      </div>
    </div>
  </div>
</div>
<script>
const data = {payload};
const sidebar = document.getElementById('sidebar');
const title = document.getElementById('title');
const plot = document.getElementById('plot');
const legend = document.getElementById('legend');
const trace = document.getElementById('trace');
const distOverallImg = document.getElementById('dist-overall-img');
const distWaitImg = document.getElementById('dist-wait-img');
const distOverallWaitImg = document.getElementById('dist-overall-wait-img');
const distWaitWaitImg = document.getElementById('dist-wait-wait-img');

function setActive(btn) {{
  const buttons = sidebar.querySelectorAll('button');
  buttons.forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}}

function showTab(kind) {{
  const exampleContent = document.getElementById('tab-example-content');
  const distContent = document.getElementById('tab-dist-content');
  const distWaitContent = document.getElementById('tab-dist-wait-content');
  if (kind === 'dist') {{
    exampleContent.style.display = 'none';
    distContent.style.display = 'block';
    distWaitContent.style.display = 'none';
  }} else if (kind === 'dist-wait') {{
    exampleContent.style.display = 'none';
    distContent.style.display = 'none';
    distWaitContent.style.display = 'block';
  }} else {{
    distContent.style.display = 'none';
    distWaitContent.style.display = 'none';
    exampleContent.style.display = 'block';
  }}
}}

function showExample(item, btn) {{
  if (btn) {{
    setActive(btn);
  }}
  const prompt = document.getElementById('prompt');
  title.textContent = 'Example ' + item.index;
  plot.src = item.plot || '';
  plot.style.display = item.plot ? 'block' : 'none';
  legend.textContent = item.legend_text ? item.legend_text : 'No legend available.';
  trace.src = item.trace || '';
  trace.style.display = item.trace ? 'block' : 'none';
  prompt.textContent = item.prompt ? item.prompt : 'No prompt available.';
  document.getElementById('overall-avg').textContent = Number(item.overall_entropy_avg || 0).toFixed(4);
  document.getElementById('overall-count').textContent = item.overall_entropy_count || 0;
  document.getElementById('wait-avg').textContent = Number(item.wait_prev_entropy_avg || 0).toFixed(4);
  document.getElementById('wait-count').textContent = item.wait_prev_entropy_count || 0;
}}

const aggregateButton = document.createElement('button');
aggregateButton.textContent = 'Aggregate';
aggregateButton.addEventListener('click', () => {{
  setActive(aggregateButton);
  showTab('dist');
}});
sidebar.appendChild(aggregateButton);
const aggregateWaitButton = document.createElement('button');
aggregateWaitButton.textContent = 'Aggregate (Wait-only)';
aggregateWaitButton.addEventListener('click', () => {{
  setActive(aggregateWaitButton);
  showTab('dist-wait');
}});
sidebar.appendChild(aggregateWaitButton);

data.forEach((item) => {{
  const btn = document.createElement('button');
  btn.textContent = 'Example ' + item.index;
  btn.addEventListener('click', () => {{
    showTab('example');
    showExample(item, btn);
  }});
  sidebar.appendChild(btn);
}});

if (data.length > 0) {{
  const firstExampleButton = sidebar.querySelectorAll('button')[2];
  if (firstExampleButton) {{
    showTab('example');
    showExample(data[0], firstExampleButton);
  }}
}}
distOverallImg.src = 'aggregate_overall_entropy.png';
distWaitImg.src = 'aggregate_wait_prev_entropy.png';
distOverallWaitImg.src = 'aggregate_wait_overall_entropy.png';
distWaitWaitImg.src = 'aggregate_wait_prev_entropy_wait.png';
</script>
</body>
</html>
"""
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def write_dashboard_html(
    output_dir: str,
    output_name: str,
) -> None:
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
body {{
  font-family: sans-serif;
  margin: 0;
  padding: 0;
}}
.container {{
  display: grid;
  grid-template-columns: 240px 1fr;
  min-height: 100vh;
}}
.sidebar {{
  border-right: 1px solid #ddd;
  padding: 12px;
  overflow-y: auto;
}}
.sidebar button {{
  display: block;
  width: 100%;
  text-align: left;
  margin-bottom: 6px;
  padding: 6px 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fafafa;
  cursor: pointer;
}}
.sidebar button.active {{
  background: #eef5ff;
  border-color: #aac7ff;
}}
.content {{
  padding: 16px;
}}
.section {{
  margin-bottom: 16px;
}}
.tabs {{
  margin-bottom: 12px;
}}
.tabs button {{
  margin-right: 8px;
  padding: 6px 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fafafa;
  cursor: pointer;
}}
.tabs button.active {{
  background: #eef5ff;
  border-color: #aac7ff;
}}
.legend {{
  white-space: pre-wrap;
  background: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 10px;
}}
.plot {{
  max-width: 100%;
  border: 1px solid #eee;
}}
.trace {{
  width: 100%;
  height: 500px;
  border: 1px solid #ddd;
}}
</style>
</head>
<body>
<div class="container">
  <div class="sidebar">
    <div id="sidebar" class="section"></div>
  </div>
  <div class="content">
    <h2 id="title">Select an example</h2>
    <div id="tab-example-content">
      <div class="section">
        <h3>Prompt</h3>
        <div id="prompt" class="legend">No prompt available.</div>
      </div>
      <div class="section">
        <h3>Example stats</h3>
        <div class="legend">
          Average entropy (all tokens, excluding token before Wait): <span id="overall-avg">0.0000</span>
          (<span id="overall-count">0</span> tokens)
          <br/>
          Average entropy (token before Wait): <span id="wait-avg">0.0000</span>
          (<span id="wait-count">0</span> occurrences)
        </div>
      </div>
      <div class="section">
        <h3>Entropy Plot</h3>
        <img id="plot" class="plot" alt="Entropy plot"/>
      </div>
      <div class="section">
        <h3>Sentence legend (S# -> sentence)</h3>
        <div id="legend" class="legend">No legend available.</div>
      </div>
      <div class="section">
        <h3>Trace (HTML)</h3>
        <iframe id="trace" class="trace"></iframe>
      </div>
    </div>
    <div id="tab-dist-content" style="display:none;">
      <div class="section">
        <h3>Aggregate distributions (per-example averages)</h3>
        <div class="legend">
          Mean overall entropy (excluding token before Wait): <span id="mean-overall">0.0000</span>
          <br/>
          Mean entropy (token before Wait): <span id="mean-wait">0.0000</span>
        </div>
      </div>
      <div class="section">
        <h4>Overall entropy averages</h4>
        <img id="dist-overall-img" class="plot" alt="Overall entropy distribution"/>
      </div>
      <div class="section">
        <h4>Token-before-Wait averages</h4>
        <img id="dist-wait-img" class="plot" alt="Wait entropy distribution"/>
      </div>
    </div>
    <div id="tab-dist-wait-content" style="display:none;">
      <div class="section">
        <h3>Aggregate (Wait-only examples)</h3>
        <div class="legend">
          Mean overall entropy (excluding token before Wait): <span id="mean-overall-wait">0.0000</span>
          <br/>
          Mean entropy (token before Wait): <span id="mean-wait-wait">0.0000</span>
        </div>
      </div>
      <div class="section">
        <h4>Overall entropy averages (Wait-only)</h4>
        <img id="dist-overall-wait-img" class="plot" alt="Overall entropy distribution (Wait-only)"/>
      </div>
      <div class="section">
        <h4>Token-before-Wait averages (Wait-only)</h4>
        <img id="dist-wait-wait-img" class="plot" alt="Wait entropy distribution (Wait-only)"/>
      </div>
    </div>
  </div>
</div>
<script>
const INDEX_FILE = 'dashboard_index.json';
const STATS_FILE = 'dashboard_stats.json';
const sidebar = document.getElementById('sidebar');
const title = document.getElementById('title');
const plot = document.getElementById('plot');
const legend = document.getElementById('legend');
const trace = document.getElementById('trace');
const distOverallImg = document.getElementById('dist-overall-img');
const distWaitImg = document.getElementById('dist-wait-img');
const distOverallWaitImg = document.getElementById('dist-overall-wait-img');
const distWaitWaitImg = document.getElementById('dist-wait-wait-img');
const meanOverallEl = document.getElementById('mean-overall');
const meanWaitEl = document.getElementById('mean-wait');
const meanOverallWaitEl = document.getElementById('mean-overall-wait');
const meanWaitWaitEl = document.getElementById('mean-wait-wait');
let statsByExample = {{}};

function setActive(btn) {{
  const buttons = sidebar.querySelectorAll('button');
  buttons.forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}}

function showTab(kind) {{
  const exampleContent = document.getElementById('tab-example-content');
  const distContent = document.getElementById('tab-dist-content');
  const distWaitContent = document.getElementById('tab-dist-wait-content');
  if (kind === 'dist') {{
    exampleContent.style.display = 'none';
    distContent.style.display = 'block';
    distWaitContent.style.display = 'none';
  }} else if (kind === 'dist-wait') {{
    exampleContent.style.display = 'none';
    distContent.style.display = 'none';
    distWaitContent.style.display = 'block';
  }} else {{
    distContent.style.display = 'none';
    distWaitContent.style.display = 'none';
    exampleContent.style.display = 'block';
  }}
}}

function showExample(item, btn) {{
  if (btn) {{
    setActive(btn);
  }}
  const prompt = document.getElementById('prompt');
  title.textContent = 'Example ' + item.index;
  plot.src = item.plot || '';
  plot.style.display = item.plot ? 'block' : 'none';
  if (item.legend) {{
    fetch(item.legend).then(r => r.text()).then(t => {{
      legend.textContent = t || 'No legend available.';
    }}).catch(() => {{
      legend.textContent = 'No legend available.';
    }});
  }} else {{
    legend.textContent = 'No legend available.';
  }}
  trace.src = item.trace || '';
  trace.style.display = item.trace ? 'block' : 'none';
  const stats = statsByExample[String(item.index)] || {{}};
  prompt.textContent = stats.prompt ? stats.prompt : 'No prompt available.';
  document.getElementById('overall-avg').textContent = Number(stats.overall_entropy_avg || 0).toFixed(4);
  document.getElementById('overall-count').textContent = stats.overall_entropy_count || 0;
  document.getElementById('wait-avg').textContent = Number(stats.wait_prev_entropy_avg || 0).toFixed(4);
  document.getElementById('wait-count').textContent = stats.wait_prev_entropy_count || 0;
}}

Promise.all([
  fetch(INDEX_FILE).then(r => r.json()),
  fetch(STATS_FILE).then(r => r.json())
]).then(([indexData, statsData]) => {{
  statsByExample = statsData.per_example || {{}};
  const aggregateButton = document.createElement('button');
  aggregateButton.textContent = 'Aggregate';
  aggregateButton.addEventListener('click', () => {{
    setActive(aggregateButton);
    showTab('dist');
  }});
  sidebar.appendChild(aggregateButton);
  const aggregateWaitButton = document.createElement('button');
  aggregateWaitButton.textContent = 'Aggregate (Wait-only)';
  aggregateWaitButton.addEventListener('click', () => {{
    setActive(aggregateWaitButton);
    showTab('dist-wait');
  }});
  sidebar.appendChild(aggregateWaitButton);
  indexData.forEach((item) => {{
    const btn = document.createElement('button');
    btn.textContent = 'Example ' + item.index;
    btn.addEventListener('click', () => {{
      showTab('example');
      showExample(item, btn);
    }});
    sidebar.appendChild(btn);
  }});
  if (indexData.length > 0) {{
    const firstExampleButton = sidebar.querySelectorAll('button')[2];
    if (firstExampleButton) {{
      showTab('example');
      showExample(indexData[0], firstExampleButton);
    }}
  }}
  meanOverallEl.textContent = Number(statsData.overall_mean || 0).toFixed(4);
  meanWaitEl.textContent = Number(statsData.wait_prev_mean || 0).toFixed(4);
  meanOverallWaitEl.textContent = Number(statsData.overall_mean_wait || 0).toFixed(4);
  meanWaitWaitEl.textContent = Number(statsData.wait_prev_mean_wait || 0).toFixed(4);
  distOverallImg.src = statsData.overall_distribution_plot || 'aggregate_overall_entropy.png';
  distWaitImg.src = statsData.wait_prev_distribution_plot || 'aggregate_wait_prev_entropy.png';
  distOverallWaitImg.src = statsData.overall_distribution_wait_plot || 'aggregate_wait_overall_entropy.png';
  distWaitWaitImg.src = statsData.wait_prev_distribution_wait_plot || 'aggregate_wait_prev_entropy_wait.png';
}}).catch(() => {{
  title.textContent = 'Failed to load dashboard data. Use a local web server.';
}});
</script>
</body>
</html>
"""
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def build_sentence_legend(
    sentence_labels: List[str],
    wait_sentence_indices: List[int],
    wrap_width: int = 80,
) -> Tuple[str, int]:
    entries = []
    line_count = 0
    wait_set = set(wait_sentence_indices)
    for idx, label in enumerate(sentence_labels):
        safe_label = label.replace("$", r"\$")
        wrapped = textwrap.fill(safe_label, width=wrap_width)
        if idx in wait_set:
            entries.append(f"S{idx + 1} (Backtracking): {wrapped}")
        else:
            entries.append(f"S{idx + 1}: {wrapped}")
        line_count += max(1, len(wrapped.splitlines()))
    if entries:
        line_count += len(entries) - 1
    return "\n\n".join(entries), line_count


def plot_entropy_series(
    entropies: List[float],
    sentence_ticks: List[int],
    sentence_labels: List[str],
    output_path: str,
    legend_path: str,
    wait_sentence_indices: List[int],
) -> bool:
    xs = list(range(len(entropies)))
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(xs, entropies, linewidth=1.2)
    wait_set = set(wait_sentence_indices)
    for idx, tick in enumerate(sentence_ticks):
        if idx in wait_set:
            ax.axvline(tick, color="darkorange", linestyle="--", linewidth=1.2, alpha=0.8)
        else:
            ax.axvline(tick, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.xlabel("Token index (reasoning trace length)")
    plt.ylabel("Shannon entropy")
    if sentence_ticks:
        xticklabels = [f"S{idx+1}" for idx in range(len(sentence_ticks))]
        plt.xticks(sentence_ticks, xticklabels)
        for idx, label in enumerate(ax.get_xticklabels()):
            if idx in wait_set:
                label.set_color("darkorange")
                label.set_fontweight("bold")
    font_size = 8
    legend_text, legend_lines = build_sentence_legend(
        sentence_labels,
        wait_sentence_indices,
    )
    legend_written = False
    if legend_text:
        fig_height = fig.get_size_inches()[1]
        line_height_in = (font_size / 72) * 1.3
        required_bottom = (legend_lines * line_height_in) / fig_height + 0.02
        if required_bottom <= 0.32:
            fig.text(
                0.01,
                0.01,
                legend_text,
                ha="left",
                va="bottom",
                fontsize=font_size,
                wrap=True,
            )
            plt.tight_layout(rect=[0, required_bottom, 1, 1])
        else:
            with open(legend_path, "w", encoding="utf-8") as f:
                f.write(legend_text + "\n")
            legend_written = True
            plt.tight_layout(rect=[0, 0.12, 1, 1])
    else:
        plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return legend_written


def main() -> None:
    args = parse_args()
    summary, examples = load_results(args.input)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    if args.output_dir == "uncertainty_plots":
        args.output_dir = f"plots_{base_name}"

    if not examples:
        raise SystemExit("No examples found in JSONL file.")
    model_id = summary.get("model")
    if not model_id:
        raise SystemExit("Missing model id in summary line.")

    selected_indices: List[int]
    if args.all:
        selected_indices = list(range(len(examples)))
    elif args.example_indices.strip():
        selected_indices = [
            int(item)
            for item in args.example_indices.split(",")
            if item.strip()
        ]
    else:
        selected_indices = [args.example_index]

    for idx in selected_indices:
        if idx < 0 or idx >= len(examples):
            raise SystemExit(f"example-index out of range: {idx}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    os.makedirs(args.output_dir, exist_ok=True)

    per_example_stats: Dict[int, Dict[str, float]] = {}
    per_example_prompts: Dict[int, str] = {}
    distribution_overall: List[float] = []
    distribution_wait_prev: List[float] = []
    distribution_overall_wait: List[float] = []
    distribution_wait_prev_wait: List[float] = []

    for idx in selected_indices:
        example = examples[idx]
        tokenized = tokenizer(
            example.generated,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = tokenized["offset_mapping"]

        entropies = [step["entropy"] for step in example.uncertainty]
        if len(entropies) != len(offsets):
            min_len = min(len(entropies), len(offsets))
            entropies = entropies[:min_len]
            offsets = offsets[:min_len]

        entropies = entropies[: args.max_tokens]
        offsets = offsets[: args.max_tokens]

        sentences = split_sentences(example.generated)
        sentence_ends = [end for _, end in sentences]
        sentence_ticks = map_sentence_ends_to_token_indices(offsets, sentence_ends)
        sentence_labels = [text for text, _ in sentences[: len(sentence_ticks)]]
        wait_sentence_indices = [
            idx
            for idx, text in enumerate(sentence_labels)
            if text.lstrip().startswith("Wait")
        ]

        prompt_text = build_prompt(example.question) if example.question else ""
        per_example_prompts[idx] = prompt_text
        word_spans = build_word_spans(example.generated)
        word_values = map_tokens_to_words(word_spans, offsets, entropies)
        token_values = build_token_values(example.generated, offsets, entropies)

        wait_prev_indices = set()
        for token_idx, (token_text, _) in enumerate(token_values):
            if token_idx > 0 and is_wait_token(token_text):
                wait_prev_indices.add(token_idx - 1)

        overall_sum = 0.0
        overall_count = 0
        wait_prev_sum = 0.0
        wait_prev_count = 0
        for idx_entropy, entropy in enumerate(entropies):
            if idx_entropy in wait_prev_indices:
                wait_prev_sum += entropy
                wait_prev_count += 1
            else:
                overall_sum += entropy
                overall_count += 1

        overall_avg = (overall_sum / overall_count) if overall_count else 0.0
        wait_prev_avg = (wait_prev_sum / wait_prev_count) if wait_prev_count else 0.0

        distribution_overall.append(overall_avg)
        distribution_wait_prev.append(wait_prev_avg)
        if wait_prev_count:
            distribution_overall_wait.append(overall_avg)
            distribution_wait_prev_wait.append(wait_prev_avg)
        per_example_stats[idx] = {
            "overall_entropy_avg": overall_avg,
            "overall_entropy_count": overall_count,
            "wait_prev_entropy_avg": wait_prev_avg,
            "wait_prev_entropy_count": wait_prev_count,
        }

        if not args.dashboard_only:
            html_path = os.path.join(args.output_dir, f"{idx}_uncertainty_trace.html")
            plot_path = os.path.join(
                args.output_dir, f"{idx}_uncertainty_timeseries.png"
            )

            write_html_heatmap(word_values, token_values, html_path)
            legend_path = os.path.join(
                args.output_dir, f"{idx}_uncertainty_timeseries_legend.txt"
            )
            legend_written = plot_entropy_series(
                entropies,
                sentence_ticks,
                sentence_labels,
                plot_path,
                legend_path,
                wait_sentence_indices,
            )

            print(f"[{idx}] Wrote HTML trace to {html_path}")
            print(f"[{idx}] Wrote entropy plot to {plot_path}")
            if legend_written:
                print(f"[{idx}] Wrote sentence legend to {legend_path}")

    if not args.no_dashboard:
        dataset_label = "gsm8k" if "gsm8k" in base_name.lower() else base_name
        safe_model = sanitize_label(model_id)
        safe_dataset = sanitize_label(dataset_label)
        dashboard_name = f"dashboard_{safe_model}_{safe_dataset}.html"
        dashboard_static_name = f"dashboard_static_{safe_model}_{safe_dataset}.html"
        mean_overall = (
            sum(distribution_overall) / len(distribution_overall)
            if distribution_overall
            else 0.0
        )
        mean_wait_prev = (
            sum(distribution_wait_prev) / len(distribution_wait_prev)
            if distribution_wait_prev
            else 0.0
        )
        mean_overall_wait = (
            sum(distribution_overall_wait) / len(distribution_overall_wait)
            if distribution_overall_wait
            else 0.0
        )
        mean_wait_prev_wait = (
            sum(distribution_wait_prev_wait) / len(distribution_wait_prev_wait)
            if distribution_wait_prev_wait
            else 0.0
        )
        overall_plot_name = "aggregate_overall_entropy.png"
        wait_plot_name = "aggregate_wait_prev_entropy.png"
        overall_wait_plot_name = "aggregate_wait_overall_entropy.png"
        wait_wait_plot_name = "aggregate_wait_prev_entropy_wait.png"
        plot_distribution(
            distribution_overall,
            "Overall entropy (excluding token before Wait)",
            os.path.join(args.output_dir, overall_plot_name),
        )
        plot_distribution(
            distribution_wait_prev,
            "Entropy (token before Wait)",
            os.path.join(args.output_dir, wait_plot_name),
        )
        plot_distribution(
            distribution_overall_wait,
            "Overall entropy (Wait-only examples)",
            os.path.join(args.output_dir, overall_wait_plot_name),
        )
        plot_distribution(
            distribution_wait_prev_wait,
            "Entropy (token before Wait, Wait-only examples)",
            os.path.join(args.output_dir, wait_wait_plot_name),
        )
        index_path = write_dashboard_index(args.output_dir)
        stats_path = write_dashboard_stats(
            args.output_dir,
            per_example_stats,
            per_example_prompts,
            distribution_overall,
            distribution_wait_prev,
            mean_overall,
            mean_wait_prev,
            distribution_overall_wait,
            distribution_wait_prev_wait,
            mean_overall_wait,
            mean_wait_prev_wait,
        )
        write_dashboard_html(
            args.output_dir,
            dashboard_name,
        )
        write_dashboard_html_static(
            args.output_dir,
            dashboard_static_name,
            per_example_stats,
            per_example_prompts,
            distribution_overall,
            distribution_wait_prev,
            mean_overall,
            mean_wait_prev,
            distribution_overall_wait,
            distribution_wait_prev_wait,
            mean_overall_wait,
            mean_wait_prev_wait,
        )
        print(f"Wrote dashboard to {os.path.join(args.output_dir, dashboard_name)}")
        print(
            f"Wrote static dashboard to {os.path.join(args.output_dir, dashboard_static_name)}"
        )
        print(f"Wrote dashboard index to {index_path}")
        print(f"Wrote dashboard stats to {stats_path}")


if __name__ == "__main__":
    main()
