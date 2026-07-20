import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

CHECKPOINT_DIR = "/home/abir19/scratch/abir19/rpc_dataset_math500_layer28_500_qwen_14"
MODEL_FT_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
CLASSIFIER_NAME = "Qwen/Qwen2.5-7B-Instruct"
EXTRACT_LAYER = 28
W_VAL = 5
EPOCHS = 35
K_SWEEP = [5]
BATCH_SIZE = 4
STEERING_ALPHA = 2.0
N_SENTENCES = 20  # how many sentences to evaluate across

CLASSES_ORDERED = [
    "NEUTRAL", "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
]

# ── MODELS ──────────────────────────────────────────────────

class SDSRouter(nn.Module):
    def __init__(self, d_in, K, window_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in * window_size, 256), nn.ReLU(), nn.Linear(256, K)
        )
    def forward(self, x_w, temp=1.0):
        logits = self.net(x_w)
        return F.gumbel_softmax(logits, tau=temp, hard=True), logits

class SDSExperts(nn.Module):
    def __init__(self, K, d_in):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d_in, d_in, bias=False) for _ in range(K)])
    def forward(self, h_c, s):
        expert_outputs = torch.stack([m(h_c) for m in self.experts], dim=1)
        return torch.sum(s.unsqueeze(-1) * expert_outputs, dim=1)

# ── HELPERS ─────────────────────────────────────────────────

def load_features(path):
    with open(os.path.join(path, "all_sentences_features.pkl"), "rb") as f:
        features = pickle.load(f)
    X = torch.from_numpy(np.array([feat['hidden_state_last'] for feat in features])).float().to(DEVICE)
    return X, features

def build_rollout_dataset(X, window=W_VAL):
    data = []
    for t in range(window - 1, len(X) - 1):
        w = torch.cat([X[t - i] for i in range(window)])
        data.append({'h_w': w, 'h_c': X[t], 'd_t': X[t + 1]})
    return data

def train_sds(data, d_in, K):
    router = SDSRouter(d_in, K, W_VAL).to(DEVICE)
    experts = SDSExperts(K, d_in).to(DEVICE)
    opt = torch.optim.Adam(list(router.parameters()) + list(experts.parameters()), lr=1e-4)
    for _ in range(EPOCHS):
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            bw = torch.stack([x['h_w'] for x in batch])
            bc = torch.stack([x['h_c'] for x in batch])
            dt = torch.stack([x['d_t'] for x in batch])
            s, _ = router(bw)
            loss = F.mse_loss(experts(bc, s), dt)
            opt.zero_grad(); loss.backward(); opt.step()
    return router, experts

def classify_sentence_llm(sentence, classifier, tokenizer):
    cats = "\n".join(f"- {c}" for c in CLASSES_ORDERED)
    prompt = f"""You are classifying reasoning steps in a math problem solution.

Categories:
{cats}

Rules:
- FACT_RETRIEVAL: recalling a formula or fact
- PLAN_GENERATION: deciding what to do next  
- UNCERTAINTY_MANAGEMENT: expressing doubt or hedging
- ACTIVE_COMPUTATION: performing a calculation
- SELF_CHECKING: verifying a result
- PROBLEM_SETUP: restating the problem
- RESULT_CONSOLIDATION: summarizing findings
- FINAL_ANSWER_EMISSION: stating the final answer
- NEUTRAL: none of the above

Sentence: "{sentence[:300]}"

Reply with ONLY the category name, nothing else."""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out_ids = classifier.generate(
        inputs['input_ids'], attention_mask=inputs['attention_mask'],
        max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(out_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
    for cls in reversed(CLASSES_ORDERED):
        if cls in response:
            return cls
    return "NEUTRAL"

def generate_with_steering(model, tokenizer, input_ids, steered_activation, layer_idx, alpha=STEERING_ALPHA):
    fired = [False]

    def hook_fn(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()
        if not fired[0]:
            fired[0] = True
            orig = hidden[:, -1, :] if hidden.dim() == 3 else hidden[-1, :]
            steered = steered_activation.to(hidden.dtype)
            steered = steered / (steered.norm() + 1e-8) * (orig.norm() + 1e-8)
            blended = (1 - alpha) * orig + alpha * steered
            if hidden.dim() == 3:
                hidden[:, -1, :] = blended
            else:
                hidden[-1, :] = blended
        return (hidden,) + output[1:] if is_tuple else hidden

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    handle.remove()
    return out_ids

def print_transition_matrix(matrix, K):
    print("\n── Transition Matrix (original_stage -> expert -> steered_stage) ──")
    for orig_stage, expert_counts in sorted(matrix.items()):
        print(f"\n  [{orig_stage}]")
        for expert_idx in range(K):
            counts = expert_counts[expert_idx]
            if counts:
                top = sorted(counts.items(), key=lambda x: -x[1])
                top_str = ", ".join(f"{s}:{n}" for s, n in top)
                print(f"    Expert {expert_idx}: {top_str}")

# ── MAIN ─────────────────────────────────────────────────────

def run_steering_experiment():
    X, features = load_features(CHECKPOINT_DIR)
    print(f"Loaded {len(features)} sentence features")

    dataset = build_rollout_dataset(X)
    router, experts = train_sds(dataset, X.shape[1], K=K_SWEEP[0])
    print("Trained SDS model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_FT_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading main model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_FT_NAME, dtype=dtype, low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()

    print("Loading classifier...")
    clf_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_NAME, trust_remote_code=True)
    if clf_tokenizer.pad_token is None:
        clf_tokenizer.pad_token = clf_tokenizer.eos_token
    classifier = AutoModelForCausalLM.from_pretrained(
        CLASSIFIER_NAME, dtype=dtype, low_cpu_mem_usage=True
    ).to(DEVICE)
    classifier.eval()

    # sample evenly across the features
    K = K_SWEEP[0]
    indices = np.linspace(0, len(features) - 1, N_SENTENCES, dtype=int)

    # matrix[orig_stage][expert_idx][steered_stage] = count
    matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for i, idx in enumerate(indices):
        feat = features[idx]
        sentence = feat['sentence']
        hidden_state = torch.from_numpy(np.array(feat['hidden_state_last'])).float().to(DEVICE)

        orig_stage = classify_sentence_llm(sentence, classifier, clf_tokenizer)
        print(f"\n[{i+1}/{N_SENTENCES}] orig={orig_stage} | {sentence[:80]}...")

        for expert_idx in range(K):
            s = torch.zeros((1, K), device=DEVICE)
            s[0, expert_idx] = 1.0
            steered_activation = experts(hidden_state.unsqueeze(0), s).squeeze(0)

            input_ids = tokenizer.encode(sentence, return_tensors="pt").to(DEVICE)
            out_ids = generate_with_steering(model, tokenizer, input_ids, steered_activation, EXTRACT_LAYER)
            next_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)[len(sentence):].strip()
            steered_stage = classify_sentence_llm(next_text, classifier, clf_tokenizer)

            matrix[orig_stage][expert_idx][steered_stage] += 1
            print(f"  E{expert_idx} -> {steered_stage} | {next_text[:100]}")

    print_transition_matrix(matrix, K)

    del model, classifier
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    run_steering_experiment()