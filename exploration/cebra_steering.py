import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
from contrastive_gen.models import CEBRA_MoE_Encoder, DynamicsMoE
from contrastive_gen.data_utils import load_and_balance_data
from contrastive_gen.losses import nce_loss
from contrastive_gen.train import train_and_eval_k

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_save = "rpc_final_pipeline"
checkpoint_dir = "rpc_dataset"

os.makedirs(checkpoint_save, exist_ok=True)



print("="*60, flush=True)
print("PHASE 1: LOAD DATA & TRAIN ENCODER", flush=True)
print("="*60, flush=True)

print("Loading data...", flush=True)
all_features, triplets = load_and_balance_data(f"{checkpoint_dir}/all_sentences_features.pkl")
print(f"Loaded {len(all_features)} features, {len(triplets)} triplets", flush=True)
X_raw = np.array([f['hidden_state'] for f in all_features])
X_torch = torch.from_numpy(StandardScaler().fit_transform(X_raw)).float().to(device)
print(f"Data shape: {X_torch.shape}", flush=True)

print("\nTraining encoder with K=4...", flush=True)
K = 4
persistence, final_mse, best_model, best_dyn, best_states = train_and_eval_k(K, all_features, X_torch, triplets)
print(f"K=4: Persistence={persistence:.2%}, MSE={final_mse:.6f}", flush=True)

del X_raw, triplets
torch.cuda.empty_cache()
gc.collect()

print("\n" + "="*60, flush=True)
print("PHASE 2: STATE ASSIGNMENT", flush=True)
print("="*60, flush=True)

best_model.eval()
with torch.no_grad():
    X_torch_eval = torch.from_numpy(StandardScaler().fit_transform(
        np.array([f['hidden_state'] for f in all_features])
    )).float().to(device)
    _, s_final, _ = best_model(X_torch_eval)
    state_assignments = s_final.argmax(1).cpu().numpy()

for i, feat in enumerate(all_features):
    feat['assigned_state'] = state_assignments[i]

state_to_indices = defaultdict(list)
for i, feat in enumerate(all_features):
    state_to_indices[feat['assigned_state']].append(i)

print(f"State distribution:", flush=True)
for state_id in sorted(state_to_indices.keys()):
    count = len(state_to_indices[state_id])
    print(f"  State {state_id}: {count} sentences", flush=True)

del X_torch_eval, best_model, best_dyn, best_states
torch.cuda.empty_cache()
gc.collect()

print("\n" + "="*60, flush=True)
print("PHASE 2.5: LOGIT LENS - INTERPRET STATE MEANINGS", flush=True)
print("="*60, flush=True)

print(f"Loading model for logit lens...", flush=True)
model_ft_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model_base_name = "Qwen/Qwen2.5-14B"
dtype = torch.bfloat16

hf_model = AutoModelForCausalLM.from_pretrained(model_ft_name, torch_dtype=dtype, device_map="cuda")
model_tl = HookedTransformer.from_pretrained_no_processing(model_base_name, hf_model=hf_model, device=device, dtype=dtype)
W_U = model_tl.W_U
logit_tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
del hf_model
gc.collect()
torch.cuda.empty_cache()
print(f"Model loaded", flush=True)

def apply_logit_lens(centroid, W_U, tokenizer, top_k=10):
    if W_U is None or tokenizer is None:
        return []
    centroid = centroid.to(W_U.device).to(W_U.dtype).unsqueeze(0)
    logits = centroid @ W_U
    
    top_logits, top_indices = torch.topk(logits[0], top_k)
    probs = F.softmax(logits[0], dim=-1)
    top_probs = torch.gather(probs, 0, top_indices)

    tokens = []
    for logit, idx, prob in zip(top_logits, top_indices, top_probs):
        token_str = tokenizer.decode([idx.item()])
        tokens.append((token_str, logit.item(), prob.item()))
    return tokens

print(f"\nNote: Will analyze states after computing centroids (states may permute on retrain)", flush=True)

del model_tl, W_U
torch.cuda.empty_cache()
gc.collect()

print("\n" + "="*60, flush=True)
print("PHASE 3: ACTIVATION EXTRACTION", flush=True)
print("="*60, flush=True)

centroids_cache = f"{checkpoint_save}/centroids_multilayer.pkl"

if os.path.exists(centroids_cache):
    print("Loading cached centroids...", flush=True)
    centroids = pickle.load(open(centroids_cache, 'rb'))
    print(f"Loaded {len(centroids)} state centroids", flush=True)
else:
    running_stats = defaultdict(lambda: defaultdict(lambda: {"sum": None, "count": 0}))
    
    model_ft_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    model_base_name = "Qwen/Qwen2.5-14B"
    dtype = torch.bfloat16

    print(f"Loading models...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(model_ft_name, torch_dtype=dtype, device_map="cuda")
    model_tl = HookedTransformer.from_pretrained_no_processing(model_base_name, hf_model=hf_model, device=device, dtype=dtype)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✓ Models loaded", flush=True)

    dataset = load_dataset("openai/gsm8k", 'main', split="train")
    dataset_dict = {i: dataset[i]['question'] for i in range(len(dataset))}
    layers_to_extract = [0, 5, 10, 15, 20, 25, 31]

    def get_token_positions_in_output(problem, sentence, tokenizer, model, device):
        input_ids = tokenizer.encode(problem, return_tensors="pt").to(device)
        with torch.no_grad():
            logits, cache = model.run_with_cache(input_ids)
        full_output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        full_tokens = tokenizer.encode(full_output, add_special_tokens=False)
        positions = []
        for i in range(len(full_tokens) - len(sentence_tokens) + 1):
            if full_tokens[i:i+len(sentence_tokens)] == sentence_tokens:
                positions = list(range(i, i + len(sentence_tokens)))
                break
        return positions, cache, input_ids.shape[1]

    print(f"Extracting activations...", flush=True)
    for state_id in sorted(state_to_indices.keys()):
        indices = state_to_indices[state_id][:100]
        print(f"  State {state_id}: {len(indices)} samples", flush=True)
        
        for idx in tqdm(indices, desc=f"State {state_id}", disable=False):
            feat = all_features[idx]
            try:
                problem = dataset_dict[feat['problem_id']]
                sentence = feat['sentence']
                
                positions, cache, _ = get_token_positions_in_output(problem, sentence, tokenizer, model_tl, device)
                
                if positions:
                    for layer in layers_to_extract:
                        if layer < len(model_tl.blocks):
                            act = cache["resid_post", layer][0, positions, :].mean(0).float().cpu()
                            
                            if running_stats[state_id][layer]["sum"] is None:
                                running_stats[state_id][layer]["sum"] = act
                            else:
                                running_stats[state_id][layer]["sum"] += act
                            running_stats[state_id][layer]["count"] += 1
                
                del cache
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error: {e}", flush=True)
                continue

    print(f"Extraction complete", flush=True)

    del model_tl
    torch.cuda.empty_cache()
    gc.collect()

    print("\n" + "="*60, flush=True)
    print("PHASE 4: COMPUTE CENTROIDS", flush=True)
    print("="*60, flush=True)

    centroids = {}
    for sid in sorted(running_stats.keys()):
        centroids[sid] = {}
        for lyr in sorted(running_stats[sid].keys()):
            stat = running_stats[sid][lyr]
            if stat["count"] > 0:
                centroid = (stat["sum"] / stat["count"]).to(device)
                centroids[sid][lyr] = centroid
                print(f"  State {sid}, Layer {lyr}: {stat['count']} samples, shape {centroid.shape}", flush=True)

    pickle.dump(centroids, open(centroids_cache, 'wb'))
    print(f"Saved centroids to {centroids_cache}", flush=True)

    del running_stats
    torch.cuda.empty_cache()
    gc.collect()

print("\n" + "="*60, flush=True)
print("PHASE 2.5 (continued): LOGIT LENS ANALYSIS ON CENTROIDS", flush=True)
print("="*60, flush=True)

print(f"Loading model for logit lens analysis...", flush=True)
hf_model = AutoModelForCausalLM.from_pretrained(model_ft_name, torch_dtype=dtype, device_map="cuda")
model_tl = HookedTransformer.from_pretrained_no_processing(model_base_name, hf_model=hf_model, device=device, dtype=dtype)
W_U = model_tl.W_U
del hf_model
gc.collect()
torch.cuda.empty_cache()
print(f"Model loaded, W_U shape: {W_U.shape}", flush=True)

state_interpretations = {}

print(f"\nAnalyzing state representations (logit lens on final layer):", flush=True)
for state_id in sorted(centroids.keys()):
    print(f"\n  State {state_id}:", flush=True)
    
    centroid = centroids[state_id][31] if 31 in centroids[state_id] else centroids[state_id][25]
    top_tokens = apply_logit_lens(centroid, W_U, logit_tokenizer, top_k=10)
    state_interpretations[state_id] = top_tokens
    
    print(f"    Top predicted tokens:", flush=True)
    for i, (token, logit, prob) in enumerate(top_tokens[:5], 1):
        print(f"      {i}. '{token}' (logit={logit:.3f}, prob={prob:.3%})", flush=True)

pickle.dump(state_interpretations, open(f"{checkpoint_save}/state_logit_lens.pkl", 'wb'))
print(f"\nSaved state interpretations to {checkpoint_save}/state_logit_lens.pkl", flush=True)

del model_tl, W_U
torch.cuda.empty_cache()
gc.collect()

# === PHASE 5: CROSS-STATE STEERING ===
print("\n" + "="*60, flush=True)
print("PHASE 5: ENHANCED STEERING (Diff-in-Means + Sentence-Aware)", flush=True)
print("="*60, flush=True)

print(f"Loading transformer for steering session...", flush=True)
model_ft_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model_base_name = "Qwen/Qwen2.5-14B"
dtype = torch.bfloat16

hf_model = AutoModelForCausalLM.from_pretrained(
    model_ft_name, torch_dtype=dtype, device_map="cuda"
)
model_tl = HookedTransformer.from_pretrained_no_processing(
    model_base_name, hf_model=hf_model, device=device, dtype=dtype
)
del hf_model
torch.cuda.empty_cache()
gc.collect()
print(f"Transformer loaded", flush=True)

from functools import partial

tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
dataset = load_dataset("openai/gsm8k", 'main', split="train")

layers_to_test = [5, 10, 15, 20, 25]
steering_alphas = [0.3, 0.5, 0.7, 1.0]
steering_results = []

output_log = open(f"{checkpoint_save}/steering_outputs.txt", 'w')

def steering_hook_sentence_aware(output, inject_start_pos, inject_end_pos, steering_vector, coeff):
    """Inject steering vector across sentence token range"""
    if inject_start_pos < output.shape[1]:
        end = min(inject_end_pos, output.shape[1])
        output[0, inject_start_pos:end, :] += coeff * steering_vector.to(output.dtype)
    return output

for prob_idx in range(5):
    problem = dataset[prob_idx]['question']
    input_ids = tokenizer.encode(problem, return_tensors="pt").to(device)
    
    print(f"\n{'='*60}", flush=True)
    print(f"PROBLEM {prob_idx}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Q: {problem[:80]}...", flush=True)
    
    output_log.write(f"\n{'='*60}\n")
    output_log.write(f"PROBLEM {prob_idx}\n")
    output_log.write(f"{'='*60}\n")
    output_log.write(f"Q: {problem}\n\n")
    
    with torch.no_grad():
        baseline_output = model_tl.generate(
            input_ids, max_new_tokens=150, temperature=0.6,
            do_sample=True, top_p=0.9
        )
    baseline_text = tokenizer.decode(baseline_output[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nBASELINE:\n{baseline_text[:250]}...\n", flush=True)
    output_log.write(f"BASELINE:\n{baseline_text}\n\n")
    
    problem_features = [f for f in all_features if f['problem_id'] == prob_idx]
    if problem_features:
        native_state = np.argmax(np.bincount([f['assigned_state'] for f in problem_features]))
        print(f"Native state: {native_state}", flush=True)
    else:
        native_state = 0
    
    positions_choices = [
        (0, "BEGINNING"),
        (input_ids.shape[1] // 2, "MIDDLE"),
        (max(0, input_ids.shape[1] - 1), "END")
    ]
    
    for start_pos, loc_name in positions_choices:
        end_pos = input_ids.shape[1]
        print(f"  --- {loc_name} (tokens {start_pos}:{end_pos}, native state {native_state}) ---", flush=True)
        output_log.write(f"\n  --- {loc_name} (tokens {start_pos}:{end_pos}, native state {native_state}) ---\n")
        
        use_diff_in_means = native_state in centroids
        if not use_diff_in_means:
            print(f"  ⚠️  Native state {native_state} not in centroids, using absolute steering instead", flush=True)
            output_log.write(f"  ⚠️  Native state {native_state} not in centroids, using absolute steering instead\n")
        
        for inject_state in sorted(centroids.keys()):
            if inject_state == native_state:
                continue
            
            method_name = "Diff-in-Means" if use_diff_in_means else "Absolute"
            print(f"    Inject State {inject_state} ({method_name}):", flush=True)
            output_log.write(f"    Inject State {inject_state} ({method_name}):\n")
            
            for layer in layers_to_test:
                if layer not in centroids[inject_state]:
                    continue
                if use_diff_in_means and layer not in centroids[native_state]:
                    continue
                
                cent_inject = centroids[inject_state][layer]
                steering_vec = cent_inject - centroids[native_state][layer] if use_diff_in_means else cent_inject
                
                for alpha in steering_alphas:
                    hook_fn = partial(
                        steering_hook_sentence_aware, 
                        inject_start_pos=start_pos, 
                        inject_end_pos=end_pos,
                        steering_vector=steering_vec, 
                        coeff=alpha
                    )
                    
                    handle = model_tl.blocks[layer].hook_resid_post.register_forward_hook(
                        lambda m, i, o: hook_fn(o)
                    )
                    
                    with torch.no_grad():
                        steered_output = model_tl.generate(
                            input_ids, max_new_tokens=150, temperature=0.6,
                            do_sample=True, top_p=0.9
                        )
                    steered_text = tokenizer.decode(steered_output[0][input_ids.shape[1]:], skip_special_tokens=True)
                    
                    handle.remove()
                    torch.cuda.empty_cache()
                    
                    is_different = baseline_text[:120] != steered_text[:120]
                    marker = "✓" if is_different else "—"
                    
                    steering_results.append({
                        'problem_idx': prob_idx,
                        'position': loc_name,
                        'native_state': native_state,
                        'inject_state': inject_state,
                        'layer': layer,
                        'alpha': alpha,
                        'baseline': baseline_text,
                        'steered': steered_text,
                        'changed': is_different,
                        'steering_method': method_name
                    })
                    
                    print(f"      L{layer} α={alpha}: {marker}", end="  ", flush=True)
                    output_log.write(f"      L{layer} α={alpha}:\n{steered_text}\n\n")
                    del steered_output
                
                print(flush=True)
                gc.collect()

output_log.close()

print(f"\nCleaning up transformer...", flush=True)
del model_tl
torch.cuda.empty_cache()
gc.collect()

pickle.dump(steering_results, open(f"{checkpoint_save}/cross_state_steering.pkl", 'wb'))

print(f"\n{'='*60}", flush=True)
print("CROSS-STATE STEERING SUMMARY", flush=True)
print(f"{'='*60}", flush=True)

changes = sum(1 for r in steering_results if r['changed'])
total = len(steering_results)
print(f"\nTotal experiments: {total}", flush=True)
print(f"Generation changed: {changes} ({100*changes/total:.1f}%)", flush=True)

print(f"\n--- By Layer ---", flush=True)
for layer in sorted(set(r['layer'] for r in steering_results)):
    matches = [r for r in steering_results if r['layer'] == layer]
    changed = sum(1 for r in matches if r['changed'])
    print(f"  Layer {layer}: {changed}/{len(matches)} ({100*changed/len(matches):.1f}%)", flush=True)

print(f"\n--- By Steering Coefficient ---", flush=True)
for alpha in sorted(set(r['alpha'] for r in steering_results)):
    matches = [r for r in steering_results if r['alpha'] == alpha]
    changed = sum(1 for r in matches if r['changed'])
    print(f"  α={alpha}: {changed}/{len(matches)} ({100*changed/len(matches):.1f}%)", flush=True)

print(f"\n--- By Position ---", flush=True)
for pos in ["BEGINNING", "MIDDLE", "END"]:
    matches = [r for r in steering_results if r['position'] == pos]
    if matches:
        changed = sum(1 for r in matches if r['changed'])
        print(f"  {pos}: {changed}/{len(matches)} ({100*changed/len(matches):.1f}%)", flush=True)

print(f"\n--- By State Transfer ---", flush=True)
for native_state in sorted(set(r['native_state'] for r in steering_results)):
    for inject_state in sorted(set(r['inject_state'] for r in steering_results)):
        matches = [r for r in steering_results 
                  if r['native_state'] == native_state and r['inject_state'] == inject_state]
        if matches:
            changed = sum(1 for r in matches if r['changed'])
            print(f"  State {native_state} → State {inject_state}: {changed}/{len(matches)} ({100*changed/len(matches):.1f}%)", flush=True)

print(f"\nPipeline complete.")
print(f"  Results saved to: {checkpoint_save}/cross_state_steering.pkl")
print(f"  Outputs saved to: {checkpoint_save}/steering_outputs.txt")
print(f"  Centroids saved to: {centroids_cache}")
print(f"  State interpretations saved to: {checkpoint_save}/state_logit_lens.pkl", flush=True)

print(f"\n" + "="*60, flush=True)
print("STATE INTERPRETATIONS (LOGIT LENS)", flush=True)
print("="*60, flush=True)

logit_lens_cache = f"{checkpoint_save}/state_logit_lens.pkl"
if os.path.exists(logit_lens_cache):
    state_interp = pickle.load(open(logit_lens_cache, 'rb'))
    for state_id in sorted(state_interp.keys()):
        print(f"\nState {state_id}:", flush=True)
        top_5 = state_interp[state_id][:5]
        for i, (token, logit, prob) in enumerate(top_5, 1):
            print(f"  {i}. '{token}' (prob={prob:.3%})", flush=True)