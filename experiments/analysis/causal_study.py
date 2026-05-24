import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

from contrastive_gen.data_utils import load_and_balance_data
from contrastive_gen.train import train_and_eval_k


device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_save = "rpc_final_pipeline"
checkpoint_dir = "rpc_dataset"

os.makedirs(checkpoint_save, exist_ok=True)

def apply_logit_lens(centroid, W_U, tokenizer, top_k=10):
    if W_U is None or tokenizer is None:
        return []
    centroid = centroid.to(W_U.device).to(W_U.dtype).unsqueeze(0)
    logits = centroid @ W_U.t()

    top_logits, top_indices = torch.topk(logits[0], top_k)
    probs = F.softmax(logits[0], dim=-1)
    top_probs = torch.gather(probs, 0, top_indices)

    tokens = []
    for logit, idx, prob in zip(top_logits, top_indices, top_probs):
        token_str = tokenizer.decode([idx.item()])
        tokens.append((token_str, logit.item(), prob.item()))
    return tokens


def build_semantic_alignment_matrix(features, states, stage_key='stage'):
    print("SEMANTIC ALIGNMENT MATRIX")

    
    if not hasattr(features[0], '__getitem__') or stage_key not in features[0]:
        print(f"Warning: No '{stage_key}' field in features. Skipping alignment matrix.")
        return None
    
    human_stages = [f[stage_key] for f in features]
    
    alignment_df = pd.crosstab(
        pd.Series(states, name='Latent_Regime'),
        pd.Series(human_stages, name='Human_Stage'),
        margins=True
    )
    
    print("\nContingency Table:")
    print(alignment_df)
    
    alignment_norm = pd.crosstab(
        pd.Series(states, name='Latent_Regime'),
        pd.Series(human_stages, name='Human_Stage'),
        normalize='columns'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(alignment_df.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Semantic Alignment: Count (Regimes × Stages)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Human Reasoning Stage', fontsize=11)
    axes[0].set_ylabel('Latent Regime', fontsize=11)
    
    sns.heatmap(alignment_norm, annot=True, fmt='.2%', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title('Semantic Alignment: Stage Composition (% within Stage)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Human Reasoning Stage', fontsize=11)
    axes[1].set_ylabel('Latent Regime', fontsize=11)
    
    plt.suptitle('RPC Semantic Alignment: State-Stage Correspondence', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{checkpoint_save}/alignment_matrix.png", dpi=600, bbox_inches='tight')
    print(f"Alignment matrix visualization saved to alignment_matrix.png")
    plt.close()
    
    
    for stage in alignment_norm.columns:
        stage_regimes = alignment_norm[stage]
        max_regime = stage_regimes.idxmax()
        max_prob = stage_regimes[max_regime]
        
        print(f"\nStage '{stage}':")
        print(f"  Dominant Regime: {max_regime} ({max_prob:.1%})")
        print(f"  Top 3 Regimes:")
        for regime, prob in stage_regimes.nlargest(3).items():
            print(f"    - Regime {regime}: {prob:.1%}")
    
    alignment_df.to_csv(f"{checkpoint_save}/alignment_matrix.csv")
    alignment_norm.to_csv(f"{checkpoint_save}/alignment_matrix_normalized.csv")
    
    return alignment_df, alignment_norm


def analyze_regime_semantics(model, X_torch, states, W_U=None, tokenizer=None):
    print("SEMANTIC SIGNATURE ANALYSIS (Multi-Token Logit Lens)")
    
    regime_semantics = {}
    
    for regime_id in sorted(np.unique(states)):
        regime_idx = np.where(states == regime_id)[0]
        if len(regime_idx) == 0:
            continue
        
        print(f"\n--- Regime {regime_id} ({len(regime_idx)} samples) ---")
        
        centroid = X_torch[regime_idx].mean(0)
        regime_semantics[regime_id] = {}
        
        if W_U is not None and tokenizer is not None:
            tokens = apply_logit_lens(centroid, W_U, tokenizer, top_k=10)
            regime_semantics[regime_id]['tokens'] = tokens
            
            print("  Top 10 Tokens (Logit Lens):")
            total_prob = sum(t[2] for t in tokens)
            for token, logit, prob in tokens:
                print(f"    '{token}': logit={logit:.4f}, prob={prob:.4f}")
            print(f"  Total Probability Mass: {total_prob:.4f}")
            regime_semantics[regime_id]['total_prob_mass'] = total_prob
        
        regime_data = X_torch[regime_idx]
        variance = regime_data.var(0)
        top_dims = variance.topk(5)[1].cpu().numpy()
        print(f"  High-Variance Dims: {top_dims}")
        
        mean_val = regime_data.mean()
        std_val = regime_data.std()
        print(f"  Activation Stats: mean={mean_val:.4f}, std={std_val:.4f}")
        regime_semantics[regime_id]['stats'] = {
            'mean': mean_val.item(),
            'std': std_val.item(),
            'n_samples': len(regime_idx)
        }
    
    serializable = {}
    for k, v in regime_semantics.items():
        serializable[str(k)] = {
            'tokens': [(t, float(l), float(p)) for t, l, p in v.get('tokens', [])],
            'stats': v.get('stats', {}),
            'total_prob_mass': float(v.get('total_prob_mass', 0))
        }
    
    with open(f"{checkpoint_save}/regime_semantics.json", 'w') as f:
        json.dump(serializable, f, indent=2)
    
    return regime_semantics

def visualize_persistence_cliff(k_values, persistence_scores):
    print("PERSISTENCE CLIFF ANALYSIS")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, persistence_scores, marker='o', linewidth=2, markersize=8, color='darkblue')
    ax.fill_between(k_values, persistence_scores, alpha=0.3, color='lightblue')
    
    ax.set_xlabel('Number of Regimes (K)', fontsize=12)
    ax.set_ylabel('Mean Persistence', fontsize=12)
    ax.set_title('The Persistence Cliff: Effective Dimensionality of RPC', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    max_diff_idx = np.argmax(np.diff(persistence_scores))
    cliff_k = k_values[max_diff_idx]
    print(f"Persistence Cliff at K={cliff_k}")
    print(f"  Drop: {persistence_scores[max_diff_idx]:.2%} -> {persistence_scores[max_diff_idx+1]:.2%}")
    
    ax.axvline(x=cliff_k, color='red', linestyle='--', alpha=0.7, label=f'Cliff at K={cliff_k}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{checkpoint_save}/persistence_cliff.png", dpi=150)
    plt.close()


print("Loading data...")
all_features, triplets = load_and_balance_data(f"{checkpoint_dir}/all_sentences_features.pkl")
X_raw = np.array([f['hidden_state'] for f in all_features])
X_torch = torch.from_numpy(StandardScaler().fit_transform(X_raw)).float().to(device)

print("Running K-sweep...")
k_values = [2, 3, 4, 5, 6, 8]
results = []
best_k = None
best_model = None
best_dyn = None
best_states = None

for k in k_values:
    persistence, final_mse, model, dyn, states = train_and_eval_k(k, all_features, X_torch, triplets)
    results.append((k, persistence, final_mse))
    print(f"K={k}: Persistence={persistence:.2%}, MSE={final_mse:.6f}")
    
    if best_k is None or (persistence > 0.5 and final_mse < 0.1):
        best_k = k
        best_model = model
        best_dyn = dyn
        best_states = states

k_list, p_list, m_list = zip(*results)

fig, ax1 = plt.subplots()
ax1.set_xlabel('Number of Regimes (K)')
ax1.set_ylabel('Persistence (Stability)', color='tab:blue')
ax1.plot(k_list, p_list, marker='o', color='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Dynamics MSE (Accuracy)', color='tab:red')
ax2.plot(k_list, m_list, marker='s', color='tab:red')
plt.title("K-Sweep: Identifying Optimal Regime Count")
plt.savefig(f"{checkpoint_save}/k_sweep.png")
plt.close()

print(f"\n Best K: {best_k}")

visualize_persistence_cliff(list(k_list), list(p_list))

print("\nLoading DeepSeek model for logit lens...")
ds_model = None
try:
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", trust_remote_code=True)
    ds_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", torch_dtype=torch.float16, device_map="auto"
    )
    W_U = ds_model.lm_head.weight.data.detach().clone()
    print(f"  Loaded unembedding: {W_U.shape}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    W_U = None
    tokenizer = None

analyze_regime_semantics(best_model, X_torch, best_states, W_U=W_U, tokenizer=tokenizer)

if ds_model is not None:
    del ds_model
    torch.cuda.empty_cache()

if all_features and hasattr(all_features[0], '__getitem__'):
    possible_stage_keys = ['stage', 'reasoning_stage', 'phase', 'step_type']
    stage_key = None
    for key in possible_stage_keys:
        if key in all_features[0]:
            stage_key = key
            break
    
    if stage_key:
        alignment_df, alignment_norm = build_semantic_alignment_matrix(all_features, best_states, stage_key)
        
        alignment_json = {
            'raw_counts': alignment_df.iloc[:-1, :-1].to_dict(),
            'normalized': alignment_norm.to_dict(),
            'analysis': {
                f'Stage_{stage}': {
                    'dominant_regime': int(alignment_norm[stage].idxmax()),
                    'dominance_score': float(alignment_norm[stage].max()),
                    'regime_composition': {int(r): float(p) for r, p in alignment_norm[stage].items()}
                }
                for stage in alignment_norm.columns
            }
        }
        with open(f"{checkpoint_save}/alignment_matrix_summary.json", 'w') as f:
            json.dump(alignment_json, f, indent=2)
        print(f"\n Alignment matrix saved to alignment_matrix_summary.json")
    else:
        print("\nWarning: No reasoning stage labels found in features. Skipping alignment matrix.")

print("\n" + "-"*60)
print("Cleaning GPU memory for TransformerLens...")
print("-"*60)
del X_raw, all_features, triplets, results, k_list, p_list, m_list
del best_model, best_dyn
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
import gc
gc.collect()
print("GPU memory cleaned and reset")

