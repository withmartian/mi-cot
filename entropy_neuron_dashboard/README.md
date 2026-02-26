# Entropy + Confidence Neuron Dashboard

Dashboard that tracks **token-level entropy** and **entropy / token-frequency neuron activations** for **base** and **reasoning** models on the same examples, with a static HTML dashboard that includes:

- **Left panel:** Model toggle (Base | Reasoning), Aggregate, and one button per example (Example 0, 1, …).
- **Content:** Prompt, example stats, entropy plot, **confidence neuron activation** (one neuron at a time via dropdown), trace iframe, and **Compare: Base vs Reasoning** (entropy for the same example from both models).
- **Aggregate tab:** Distribution of per-example mean entropy for Base vs Reasoning.

## Requirements

- Python 3.8+
- `torch`, `transformers`, `datasets`, `numpy`
- `matplotlib` (for entropy timeseries PNGs)
- **mi-cot** layout: `track_entropy/` and `entropy_steering/` at the same level as `entropy_neuron_dashboard/` (so the script can import from them). The script uses `entropy_steering/entropy_neuron_utils` for finding entropy and token-frequency neurons.

## Input

A **JSONL** file produced by the track_uncertainty pipeline (e.g. from `track_entropy`), with:

- First line (optional): `{"summary": {"model": "model-id"}}`
- Remaining lines: one JSON object per example with at least `"question"` and `"generated"` (and optionally `"uncertainty"`). The script uses `question` + `generated` to run a forward pass on both models and compute entropies and neuron activations.

## How to run

1. **From the repo root** (e.g. `mi-cot`), or from any directory with `mi-cot` as a sibling:

   ```bash
   cd /path/to/mi-cot
   python entropy_neuron_dashboard/run_dashboard.py \
     --input track_entropy/results_qwen1.5_gsm8k.jsonl \
     --output-dir entropy_neuron_dashboard_out \
     --base_model Qwen/Qwen2.5-1.5B \
     --reasoning_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
     --max_examples 10
   ```

2. **Arguments:**

   - `--input` – JSONL path (default: `results.jsonl`).
   - `--output-dir` – Where to write dashboard files (default: `entropy_neuron_dashboard_out`).
   - `--base_model` – Base model name (default: `Qwen/Qwen2.5-1.5B`).
   - `--reasoning_model` – Reasoning / fine-tuned model (default: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`).
   - `--top_n_entropy` – Number of entropy neurons to use (default: 16).
   - `--top_n_tokfreq` – Number of token-frequency neurons (default: 16).
   - `--gsm8k_max_samples` – GSM8K samples for token-frequency distribution (default: 2000).
   - `--max_examples` – Max examples to process (default: 10).
   - `--max_tokens` – Max tokens per example for plots (default: 512).
   - `--no_dashboard` – Skip writing the HTML dashboard (still writes index + stats + PNGs).

3. **Output:**

   - `dashboard_index.json` – List of examples and asset filenames (base/reasoning plot and trace per example).
   - `dashboard_stats.json` – Per-model entropy indices, token-freq indices, and per-example entropies + neuron activations.
   - `base_{i}_uncertainty_timeseries.png`, `reasoning_{i}_uncertainty_timeseries.png` – Entropy timeseries per example.
   - `base_{i}_uncertainty_trace.html`, `reasoning_{i}_uncertainty_trace.html` – Minimal trace HTML per example.
   - `dashboard_static_entropy_neurons.html` – Static dashboard (open in a browser; use a local server if loading JSON/images fails due to CORS).

4. **View the dashboard:**

   Open `output_dir/dashboard_static_entropy_neurons.html` in a browser. If the page fails to load assets (e.g. images or JSON), serve the output directory with a local HTTP server:

   ```bash
   cd entropy_neuron_dashboard_out
   python -m http.server 8080
   ```

   Then open `http://localhost:8080/dashboard_static_entropy_neurons.html`.

## Behaviour

- **Neuron finding:** For each model, the script finds entropy neurons and token-frequency neurons (using GSM8K-based `p_freq` and the logic in `entropy_steering/entropy_neuron_utils`).
- **Per-example forward:** For each example, the full sequence `prompt + generated` is run through each model; entropies and final-layer MLP activations for the selected neurons are recorded for the generated tokens.
- **Dashboard:** The left panel lets you switch between **Base** and **Reasoning**. The main content shows the selected model’s entropy plot and trace, and a **Confidence neuron activation** plot: choose a neuron from the dropdown to see its activation over tokens. The **Compare** section shows base vs reasoning entropy for the same example. **Aggregate** shows the distribution of per-example mean entropy for both models.
