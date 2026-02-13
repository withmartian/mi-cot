# # # #!/usr/bin/env python
# # # """
# # # RPC Contrastive Learning — Minimalist: InfoNCE + L1 Logit Sparsity

# # # No annealing. Just:
# # # 1. Hard STE for discrete state selection
# # # 2. L1 penalty on raw logits (not softmax)
# # # 3. LayerNorm at output for fair initialization
# # # """

# # # import pickle
# # # from collections import defaultdict

# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # import torch.optim as optim
# # # from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
# # # from sklearn.preprocessing import StandardScaler

# # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # torch.manual_seed(0)
# # # np.random.seed(0)


# # # def gumbel_softmax_ste(logits, tau=1.0, eps=1e-20):
# # #     """Hard one-hot assignment via Gumbel-Softmax + Straight-Through Estimator"""
# # #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
# # #     y = logits + gumbel_noise
# # #     y_soft = F.softmax(y / tau, dim=-1)
    
# # #     y_hard = torch.zeros_like(y_soft).scatter_(
# # #         1, y_soft.argmax(dim=1, keepdim=True), 1.0
# # #     )
# # #     return y_hard - y_soft.detach() + y_soft


# # # class TopKEncoder(nn.Module):
# # #     def __init__(self, d_input, num_states=8):
# # #         super().__init__()
# # #         self.num_states = num_states
# # #         self.feature_extractor = nn.Sequential(
# # #             nn.Linear(d_input, 1024),
# # #             nn.LayerNorm(1024),
# # #             nn.ReLU(),
# # #             nn.Linear(1024, 32),
# # #             nn.LayerNorm(32),
# # #             nn.ReLU()
# # #         )
# # #         self.state_head = nn.Sequential(
# # #             nn.Linear(32, num_states),
# # #             nn.LayerNorm(num_states)
# # #         )

# # #     def forward(self, x, temp=0.1):
# # #         latent = self.feature_extractor(x)
# # #         logits = self.state_head(latent)
# # #         z = gumbel_softmax_ste(logits, tau=temp)
# # #         return z, logits


# # # def info_nce_loss(z_a, z_p, z_n, temperature=0.1):
# # #     pos_sim = torch.sum(z_a * z_p, dim=-1, keepdim=True)
# # #     neg_sim = torch.matmul(z_a.unsqueeze(1), z_n.transpose(-1, -2)).squeeze(1)
# # #     logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
# # #     labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
# # #     return F.cross_entropy(logits, labels)


# # # def logit_sparsity_loss(logits):
# # #     """L1 penalty on raw logits to encourage silence in unused states"""
# # #     return torch.mean(torch.abs(logits))


# # # # === DATA ===

# # # print("=" * 80, flush=True)
# # # print("LOADING DATA", flush=True)
# # # print("=" * 80, flush=True)

# # # ckpt = "rpc_dataset/all_sentences_features.pkl"
# # # try:
# # #     all_features_full = pickle.load(open(ckpt, "rb"))
# # #     all_features = [f for f in all_features_full if f["problem_id"] < 100]
# # #     print(f"✓ Loaded {len(all_features)} features", flush=True)
# # # except FileNotFoundError:
# # #     print("⚠ Data file not found", flush=True)

# # # X = np.array([f["hidden_state"] for f in all_features])
# # # X = StandardScaler().fit_transform(X)
# # # X_torch = torch.tensor(X, dtype=torch.float32).to(device)

# # # stage_to_idx = {s: i for i, s in enumerate(sorted({f["stage"] for f in all_features}))}
# # # for f in all_features:
# # #     f["stage_idx"] = stage_to_idx[f["stage"]]
# # #     f["is_labeled"] = f["stage"] != "NEUTRAL"

# # # print(f"Data shape: {X.shape}", flush=True)
# # # print(f"Labeled: {sum(f['is_labeled'] for f in all_features)} / {len(all_features)}", flush=True)


# # # # === TRIPLETS ===

# # # def build_triplets(all_features, time_window=3, hard_lag=6,
# # #                    num_negatives=8, repeat=10):
# # #     pid_to_idx = defaultdict(list)
# # #     for i, f in enumerate(all_features):
# # #         pid_to_idx[f["problem_id"]].append(i)

# # #     valid_idx = [i for i, f in enumerate(all_features) if f["is_labeled"]]
# # #     triplets = []

# # #     for _ in range(repeat):
# # #         for anchor in valid_idx:
# # #             pid = all_features[anchor]["problem_id"]
# # #             local = pid_to_idx[pid]
# # #             t = local.index(anchor)

# # #             pos = []
# # #             for dt in range(1, time_window + 1):
# # #                 if t + dt < len(local):
# # #                     pos.append(local[t + dt])
# # #                 if t - dt >= 0:
# # #                     pos.append(local[t - dt])

# # #             if not pos:
# # #                 continue
# # #             pos = np.random.choice(pos)

# # #             negs = []
# # #             for dt in range(hard_lag, len(local) - t):
# # #                 if t + dt < len(local):
# # #                     negs.append(local[t + dt])

# # #             other_pids = [p for p in pid_to_idx if p != pid]
# # #             while len(negs) < num_negatives:
# # #                 negs.append(np.random.choice(pid_to_idx[np.random.choice(other_pids)]))

# # #             triplets.append((anchor, pos, negs[:num_negatives]))

# # #     return triplets


# # # triplets = build_triplets(all_features, time_window=3, hard_lag=6, repeat=15)
# # # print(f"✓ Built {len(triplets)} triplets", flush=True)


# # # # === TRAINING ===

# # # encoder = TopKEncoder(X.shape[1], num_states=4).to(device)
# # # optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

# # # print("\nTraining — InfoNCE + L1 Logit Sparsity (No Annealing)\n", flush=True)

# # # epochs = 200
# # # batch_size = 64
# # # temp = 0.1
# # # sparsity_weight = 0.2

# # # for epoch in range(epochs):
# # #     np.random.shuffle(triplets)
# # #     total_loss = 0
    
# # #     for i in range(0, len(triplets), batch_size):
# # #         batch = triplets[i:i + batch_size]
# # #         a = torch.tensor([t[0] for t in batch]).to(device)
# # #         p = torch.tensor([t[1] for t in batch]).to(device)
# # #         n = torch.tensor([t[2] for t in batch]).to(device)

# # #         z_a, logits_a = encoder(X_torch[a], temp=temp)
# # #         z_p, _ = encoder(X_torch[p], temp=temp)

# # #         n_flat = n.view(-1)
# # #         z_n_flat, _ = encoder(X_torch[n_flat], temp=temp)
# # #         z_n = z_n_flat.view(n.shape[0], n.shape[1], -1)

# # #         loss_nce = info_nce_loss(z_a, z_p, z_n)
# # #         loss_sparse = logit_sparsity_loss(logits_a)
        
# # #         loss = loss_nce + (sparsity_weight * loss_sparse)

# # #         optimizer.zero_grad()
# # #         loss.backward()
# # #         optimizer.step()

# # #         total_loss += loss.item()

# # #     if (epoch + 1) % 10 == 0:
# # #         avg_loss = total_loss / (len(triplets) / batch_size)
# # #         print(f"Epoch {epoch+1:3d} | Loss {avg_loss:.4f}", flush=True)


# # # # === INFERENCE ===

# # # encoder.eval()

# # # def check_geometry(encoder, all_features, X_torch):
# # #     with torch.no_grad():
# # #         z_all = []
# # #         for i in range(0, len(X_torch), 256):
# # #             z_batch, _ = encoder(X_torch[i:i+256], temp=0.05)
# # #             z_all.append(z_batch)
# # #         z_all = torch.cat(z_all, dim=0)

# # #     pid_to_idx = defaultdict(list)
# # #     for i, f in enumerate(all_features):
# # #         pid_to_idx[f["problem_id"]].append(i)

# # #     close_sims, far_sims = [], []
# # #     pids = list(pid_to_idx.keys())
# # #     np.random.shuffle(pids)

# # #     count = 0
# # #     for pid in pids:
# # #         if count >= 1000:
# # #             break
# # #         idxs = pid_to_idx[pid]
# # #         if len(idxs) < 10:
# # #             continue

# # #         t = np.random.randint(2, len(idxs) - 2)
# # #         a = idxs[t]
# # #         p = idxs[t + 1]
# # #         close_sims.append(torch.dot(z_all[a], z_all[p]).item())

# # #         other_pid = np.random.choice([x for x in pids if x != pid])
# # #         n = np.random.choice(pid_to_idx[other_pid])
# # #         far_sims.append(torch.dot(z_all[a], z_all[n]).item())

# # #         count += 1

# # #     print("\n--- GEOMETRY CHECK ---", flush=True)
# # #     print(f"Neighbor (same problem): {np.mean(close_sims):.4f}", flush=True)
# # #     print(f"Random (diff problem):   {np.mean(far_sims):.4f}", flush=True)
# # #     print(f"Gap:                     {np.mean(close_sims) - np.mean(far_sims):.4f}", flush=True)


# # # check_geometry(encoder, all_features, X_torch)


# # # # === STATE ANALYSIS ===

# # # with torch.no_grad():
# # #     z_all = []
# # #     for i in range(0, len(X_torch), 256):
# # #         z_batch, _ = encoder(X_torch[i:i+256], temp=0.05)
# # #         z_all.append(z_batch)
# # #     z_all = torch.cat(z_all, dim=0)

# # # states = torch.argmax(z_all, dim=1).cpu().numpy()
# # # usage = np.bincount(states, minlength=encoder.num_states)

# # # print("\n--- STATE USAGE ---", flush=True)
# # # for i, c in enumerate(usage):
# # #     pct = 100 * c / len(states)
# # #     status = "DEAD" if c == 0 else ""
# # #     print(f"State {i}: {c:5d} ({pct:5.1f}%) {status}", flush=True)


# # # # === LABEL CORRELATION ===

# # # mask = np.array([f["is_labeled"] for f in all_features])
# # # true = np.array([f["stage_idx"] for f in all_features])[mask]
# # # pred = states[mask]

# # # stage_names = sorted(stage_to_idx.keys())
# # # state_to_labels = defaultdict(lambda: defaultdict(int))

# # # for t, s in zip(true, pred):
# # #     state_to_labels[s][t] += 1

# # # print("\n--- TOP LABELS PER STATE ---", flush=True)
# # # for s in range(encoder.num_states):
# # #     labels = state_to_labels[s]
# # #     if not labels:
# # #         print(f"State {s}: (unused)", flush=True)
# # #         continue
# # #     top = sorted(labels.items(), key=lambda x: x[1], reverse=True)[:3]
# # #     print(f"State {s}: {', '.join(f'{stage_names[l]}({c})' for l, c in top)}", flush=True)


# # # # === METRICS ===

# # # ami = adjusted_mutual_info_score(true, pred)
# # # nmi = normalized_mutual_info_score(true, pred)
# # # print(f"\nAMI: {ami:.4f} | NMI: {nmi:.4f}", flush=True)


# # # # === PERSISTENCE ===

# # # pid_to_idx = defaultdict(list)
# # # for i, f in enumerate(all_features):
# # #     pid_to_idx[f["problem_id"]].append(i)

# # # persistence_scores = []
# # # for pid, idxs in pid_to_idx.items():
# # #     if len(idxs) < 2:
# # #         continue
# # #     seq = states[idxs]
# # #     switches = sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))
# # #     persistence = 1 - switches / (len(seq) - 1)
# # #     persistence_scores.append(persistence)

# # # print("\n--- PERSISTENCE ---", flush=True)
# # # print(f"Mean:   {np.mean(persistence_scores):.3f}", flush=True)
# # # print(f"Median: {np.median(persistence_scores):.3f}", flush=True)
# # # print(f"Std:    {np.std(persistence_scores):.3f}", flush=True)


# # # # === STATE DURATIONS ===

# # # state_durations = defaultdict(list)
# # # for pid, idxs in pid_to_idx.items():
# # #     seq = states[idxs]
# # #     current = seq[0]
# # #     dur = 1
# # #     for s in seq[1:]:
# # #         if s == current:
# # #             dur += 1
# # #         else:
# # #             state_durations[current].append(dur)
# # #             current = s
# # #             dur = 1
# # #     state_durations[current].append(dur)

# # # print("\n--- STATE DURATIONS ---", flush=True)
# # # for s in range(encoder.num_states):
# # #     durs = state_durations[s]
# # #     if durs:
# # #         print(f"State {s}: mean={np.mean(durs):.1f}, min={np.min(durs)}, max={np.max(durs)}, n={len(durs)}", flush=True)
# # #     else:
# # #         print(f"State {s}: (never occurred)", flush=True)


# # # # === TRANSITIONS ===

# # # transitions = np.zeros((encoder.num_states, encoder.num_states))
# # # for i in range(len(states) - 1):
# # #     if all_features[i]["problem_id"] == all_features[i+1]["problem_id"]:
# # #         transitions[states[i], states[i + 1]] += 1

# # # row_sums = transitions.sum(axis=1, keepdims=True)
# # # trans_probs = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums != 0)

# # # fig, ax = plt.subplots(figsize=(10, 8))
# # # im = ax.imshow(trans_probs, cmap="Greens", vmin=0, vmax=1, aspect="auto")
# # # ax.set_title("State Transition Probability Matrix")
# # # ax.set_xlabel("Next State")
# # # ax.set_ylabel("Current State")
# # # ax.set_xticks(range(encoder.num_states))
# # # ax.set_yticks(range(encoder.num_states))
# # # fig.colorbar(im, ax=ax)
# # # plt.tight_layout()
# # # plt.show()


# # #!/usr/bin/env python3
# # """
# # RPC-CEBRA: Complete Unified Pipeline
# # - Phase 1: Generate CoTs and causal matrices
# # - Phase 2: Extract token-level activations with windowed sliding
# # - Phase 3: Balanced triplet sampling and training
# # """

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # import numpy as np
# # import pickle
# # import os
# # import re
# # import gc
# # import matplotlib.pyplot as plt
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
# # from collections import defaultdict, Counter
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # from transformer_lens import HookedTransformer
# # from datasets import load_dataset
# # from tqdm import tqdm

# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # torch.manual_seed(0)
# # np.random.seed(0)

# # checkpoint_dir = "rpc_cebra_window"
# # os.makedirs(checkpoint_dir, exist_ok=True)
# # dtype = torch.bfloat16
# # SKIP_CACHE = False

# # model_ft_name = "deepseek-ai/deepseek-r1-distill-qwen-14b"
# # model_base_name = "Qwen/Qwen2.5-14B"

# # ANCHOR_CLASSES = {
# #     "PROBLEM_SETUP": "Parsing or rephrasing the problem",
# #     "PLAN_GENERATION": "Stating or deciding on a plan of action",
# #     "FACT_RETRIEVAL": "Recalling facts, formulas, problem details",
# #     "ACTIVE_COMPUTATION": "Algebra, calculations, or manipulations",
# #     "UNCERTAINTY_MANAGEMENT": "Expressing confusion, re-evaluating",
# #     "RESULT_CONSOLIDATION": "Aggregating intermediate results",
# #     "SELF_CHECKING": "Verifying previous steps, checking",
# #     "FINAL_ANSWER_EMISSION": "Explicitly stating the final answer"
# # }

# # # === PHASE 1: LOAD MODELS & GENERATE CoTs ===

# # print("=" * 80, flush=True)
# # print("PHASE 1: LOAD MODELS & GENERATE CoTs", flush=True)
# # print("=" * 80 + "\n", flush=True)

# # print("Loading models...", flush=True)
# # tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
# # hf_model = AutoModelForCausalLM.from_pretrained(
# #     model_ft_name, torch_dtype=dtype, device_map="cuda"
# # )
# # model_tl = HookedTransformer.from_pretrained_no_processing(
# #     model_base_name, hf_model=hf_model, device=device, dtype=dtype
# # )
# # del hf_model
# # torch.cuda.empty_cache()
# # gc.collect()
# # print("✓ Models loaded\n", flush=True)

# # def split_into_sentences(text):
# #     sentences = re.split(r'(?<=[.!?])\s+', text)
# #     return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

# # def get_sentence_token_positions(text, sentences, tokenizer):
# #     input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
# #     token_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
# #     positions = []
# #     current_pos = 0
# #     token_offset = 0
# #     for sent in sentences:
# #         start = token_text.find(sent, current_pos)
# #         if start == -1:
# #             positions.append(set())
# #             continue
# #         sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
# #         token_positions = set(range(token_offset, token_offset + len(sent_tokens)))
# #         positions.append(token_positions)
# #         token_offset += len(sent_tokens)
# #         current_pos = start + len(sent)
# #     return positions



# # print("Loading dataset...", flush=True)
# # dataset = load_dataset("openai/gsm8k", 'main', split="train[:100]")
# # problems = [item['question'] for item in dataset]
# # print(f"✓ Loaded {len(problems)} problems\n", flush=True)

# # ckpt_cot = f"{checkpoint_dir}/cot_data.pkl"
# # if os.path.exists(ckpt_cot) and not SKIP_CACHE:
# #     print(f"Loading cached CoT data...", flush=True)
# #     all_cot_data = pickle.load(open(ckpt_cot, 'rb'))
# #     print(f"✓ Loaded {len(all_cot_data)} cached problems\n", flush=True)
# # else:
# #     all_cot_data = {}
# #     for pid, problem in enumerate(problems):
# #         print(f"[{pid}] Generating CoT...", flush=True)
# #         input_ids = tokenizer.encode(problem, return_tensors='pt').to(device)
# #         with torch.no_grad():
# #             output_ids = model_tl.generate(
# #                 input_ids, max_new_tokens=500, temperature=0.6,
# #                 do_sample=True, top_p=0.9
# #             )
# #         full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# #         cot = full_text[len(problem):].strip() if full_text.startswith(problem) else full_text
# #         sentences = split_into_sentences(cot)
        
# #         if len(sentences) < 2:
# #             print(f"  ✗ Only {len(sentences)} sentences, skipping", flush=True)
# #             del output_ids, input_ids
# #             continue
        
# #         print(f"  {len(sentences)} sentences", flush=True)
# #         all_cot_data[pid] = {
# #             'problem': problem,
# #             'cot': cot,
# #             'sentences': sentences,
# #         }
# #         del output_ids, input_ids
# #         torch.cuda.empty_cache()
# #         gc.collect()
    
# #     pickle.dump(all_cot_data, open(ckpt_cot, 'wb'))
# #     print(f"\n✓ Saved CoT data for {len(all_cot_data)} problems\n", flush=True)

# # # === PHASE 2: CLASSIFY & EXTRACT TOKEN-LEVEL ACTIVATIONS ===

# # print("=" * 80, flush=True)
# # print("PHASE 2: CLASSIFY & EXTRACT TOKEN-LEVEL WINDOWED ACTIVATIONS", flush=True)
# # print("=" * 80 + "\n", flush=True)

# # del model_tl
# # torch.cuda.empty_cache()
# # gc.collect()

# # print("Loading base model for classification...", flush=True)
# # model_base = AutoModelForCausalLM.from_pretrained(
# #     model_base_name, torch_dtype=dtype
# # ).to(device)
# # print("✓ Model loaded\n", flush=True)

# # def classify_sentence(sentence):
# #     classes_list = "\n".join([f"- {k}: {v}" for k, v in ANCHOR_CLASSES.items()])
# #     prompt = f"""Classify this reasoning step into one of these categories:

# # {classes_list}

# # Sentence: "{sentence}"

# # Category:"""
# #     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
# #     with torch.no_grad():
# #         output_ids = model_base.generate(
# #             input_ids, max_new_tokens=5, temperature=0.1, do_sample=False
# #         )
# #     response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
# #     category = response.strip().upper().split()[0]
# #     del input_ids, output_ids
# #     if category in ANCHOR_CLASSES:
# #         return category
# #     return "PLAN_GENERATION"

# # all_sentence_data = []
# # anchor_classes = {}

# # for pid, data in all_cot_data.items():
# #     sentences = data['sentences']
    
# #     print(f"[Problem {pid}] Processing {len(sentences)} sentences...", flush=True)
    
# #     for sent_idx, sentence in enumerate(sentences):
# #         stage = classify_sentence(sentence)
# #         is_anchor = stage != "NEUTRAL"
        
# #         all_sentence_data.append({
# #             'problem_id': pid,
# #             'sentence_idx': sent_idx,
# #             'sentence': sentence,
# #             'stage': stage,
# #             'is_anchor': is_anchor,
# #         })
        
# #         if is_anchor:
# #             anchor_classes[stage] = anchor_classes.get(stage, 0) + 1
    
# #     torch.cuda.empty_cache()
# #     gc.collect()

# # print(f"\n✓ Processed {len(all_sentence_data)} total sentences\n", flush=True)

# # print("Reasoning Stage Distribution (anchors only):", flush=True)
# # for stage in sorted(ANCHOR_CLASSES.keys()):
# #     count = anchor_classes.get(stage, 0)
# #     print(f"  {stage:25s}: {count}", flush=True)
# # print()

# # del model_base
# # torch.cuda.empty_cache()
# # gc.collect()

# # print("Reloading models for token-level activation extraction...", flush=True)
# # tokenizer = AutoTokenizer.from_pretrained(model_base_name, trust_remote_code=True)
# # hf_model = AutoModelForCausalLM.from_pretrained(
# #     model_ft_name, torch_dtype=dtype, device_map="cuda"
# # )
# # model_tl = HookedTransformer.from_pretrained_no_processing(
# #     model_base_name, hf_model=hf_model, device=device, dtype=dtype
# # )
# # del hf_model
# # torch.cuda.empty_cache()
# # gc.collect()
# # print("✓ Models loaded\n", flush=True)

# # def get_token_activations(problem, sentences, sent_idx, layer=-1):
# #     """Extract token-level activations for a sentence"""
# #     ctx = problem + " " + " ".join(sentences[:sent_idx])
# #     input_ids = tokenizer.encode(ctx, return_tensors="pt").to(device)
    
# #     with torch.no_grad():
# #         _, cache = model_tl.run_with_cache(input_ids)
# #         hidden = cache["resid_post", layer][0, :, :].float().cpu().numpy()
    
# #     del cache, input_ids
# #     torch.cuda.empty_cache()
# #     gc.collect()
# #     return hidden

# # def get_windowed_embeddings(token_embeddings, window_size=10, stride=5):
# #     """Extract windowed embeddings from token sequence"""
# #     if len(token_embeddings) < window_size:
# #         return np.array([token_embeddings.mean(axis=0)])
    
# #     windows = []
# #     for i in range(0, len(token_embeddings) - window_size + 1, stride):
# #         window = token_embeddings[i : i + window_size]
# #         windows.append(window.mean(axis=0))
    
# #     return np.array(windows)

# # ckpt_features = f"{checkpoint_dir}/windowed_features.pkl"
# # if os.path.exists(ckpt_features) and not SKIP_CACHE:
# #     print(f"Loading cached windowed features...", flush=True)
# #     all_features = pickle.load(open(ckpt_features, 'rb'))
# #     print(f"✓ Loaded {len(all_features)} cached windowed embeddings\n", flush=True)
# # else:
# #     all_features = []
    
# #     for item in tqdm(all_sentence_data, desc="Extracting token activations"):
# #         pid = item['problem_id']
# #         sent_idx = item['sentence_idx']
        
# #         problem = all_cot_data[pid]['problem']
# #         sentences = all_cot_data[pid]['sentences']
        
# #         token_hidden = get_token_activations(problem, sentences, sent_idx, layer=-1)
# #         windows = get_windowed_embeddings(token_hidden, window_size=10, stride=5)
        
# #         for w_idx, window in enumerate(windows):
# #             all_features.append({
# #                 'hidden_state': window,
# #                 'problem_id': pid,
# #                 'sentence_idx': sent_idx,
# #                 'window_idx': w_idx,
# #                 'sentence': item['sentence'],
# #                 'stage': item['stage'],
# #                 'is_anchor': item['is_anchor'],
# #             })
    
# #     pickle.dump(all_features, open(ckpt_features, 'wb'))
# #     print(f"\n✓ Extracted and saved {len(all_features)} windowed embeddings\n", flush=True)

# # del model_tl
# # torch.cuda.empty_cache()
# # gc.collect()

# # # === PHASE 3: PREPARE DATA FOR TRAINING ===

# # print("=" * 80, flush=True)
# # print("PHASE 3: DATA PREPARATION & TRAINING", flush=True)
# # print("=" * 80 + "\n", flush=True)

# # X = np.array([f["hidden_state"] for f in all_features])
# # X = StandardScaler().fit_transform(X)
# # X_torch = torch.tensor(X, dtype=torch.float32).to(device)

# # stage_to_idx = {s: i for i, s in enumerate(sorted({f["stage"] for f in all_features}))}
# # for f in all_features:
# #     f["stage_idx"] = stage_to_idx[f["stage"]]

# # print(f"Data shape: {X.shape}", flush=True)
# # print(f"Labeled: {sum(f['is_anchor'] for f in all_features)} / {len(all_features)}", flush=True)

# # stage_dist = Counter([f['stage'] for f in all_features if f['is_anchor']])
# # print("\nStage Distribution (windowed):", flush=True)
# # for stage, count in sorted(stage_dist.items()):
# #     pct = 100 * count / sum(stage_dist.values()) if stage_dist else 0
# #     print(f"  {stage:25s}: {count:5d} ({pct:5.1f}%)", flush=True)
# # print()

# # # === BALANCED TRIPLET BUILDING ===

# # def build_balanced_triplets(all_features, time_window=3, hard_lag=6,
# #                             num_negatives=8, repeat=10):
# #     """Build triplets with balanced stage sampling"""
# #     pid_to_idx = defaultdict(list)
    
# #     for i, f in enumerate(all_features):
# #         pid_to_idx[f["problem_id"]].append(i)
    
# #     valid_idx = [i for i, f in enumerate(all_features) if f["is_anchor"]]
# #     triplets = []
    
# #     for _ in range(repeat):
# #         for anchor in valid_idx:
# #             pid = all_features[anchor]["problem_id"]
# #             local = pid_to_idx[pid]
# #             t = local.index(anchor)
            
# #             pos = []
# #             for dt in range(1, time_window + 1):
# #                 if t + dt < len(local):
# #                     pos.append(local[t + dt])
# #                 if t - dt >= 0:
# #                     pos.append(local[t - dt])
            
# #             if not pos:
# #                 continue
# #             pos = np.random.choice(pos)
            
# #             negs = []
# #             for dt in range(hard_lag, len(local) - t):
# #                 if t + dt < len(local):
# #                     negs.append(local[t + dt])
            
# #             other_pids = [p for p in pid_to_idx if p != pid]
# #             while len(negs) < num_negatives:
# #                 other_pid = np.random.choice(other_pids)
# #                 negs.append(np.random.choice(pid_to_idx[other_pid]))
            
# #             triplets.append((anchor, pos, negs[:num_negatives]))
    
# #     return triplets

# # triplets = build_balanced_triplets(all_features, time_window=3, hard_lag=6, repeat=15)
# # print(f"✓ Built {len(triplets)} balanced triplets\n", flush=True)

# # # === MODEL & TRAINING ===

# # def gumbel_softmax_ste(logits, tau=1.0, eps=1e-20):
# #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
# #     y = logits + gumbel_noise
# #     y_soft = F.softmax(y / tau, dim=-1)
# #     y_hard = torch.zeros_like(y_soft).scatter_(
# #         1, y_soft.argmax(dim=1, keepdim=True), 1.0
# #     )
# #     return y_hard - y_soft.detach() + y_soft

# # class TopKEncoder(nn.Module):
# #     def __init__(self, d_input, num_states=4):
# #         super().__init__()
# #         self.num_states = num_states
# #         self.feature_extractor = nn.Sequential(
# #             nn.Linear(d_input, 1024),
# #             nn.LayerNorm(1024),
# #             nn.ReLU(),
# #             nn.Linear(1024, 32),
# #             nn.LayerNorm(32),
# #             nn.ReLU()
# #         )
# #         self.state_head = nn.Sequential(
# #             nn.Linear(32, num_states),
# #             nn.LayerNorm(num_states)
# #         )

# #     def forward(self, x, temp=0.1):
# #         latent = self.feature_extractor(x)
# #         logits = self.state_head(latent)
# #         z = gumbel_softmax_ste(logits, tau=temp)
# #         return z, logits

# # def info_nce_loss(z_a, z_p, z_n, temperature=0.1):
# #     pos_sim = torch.sum(z_a * z_p, dim=-1, keepdim=True)
# #     neg_sim = torch.matmul(z_a.unsqueeze(1), z_n.transpose(-1, -2)).squeeze(1)
# #     logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
# #     labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
# #     return F.cross_entropy(logits, labels)

# # def logit_sparsity_loss(logits):
# #     return torch.mean(torch.abs(logits))

# # encoder = TopKEncoder(X.shape[1], num_states=4).to(device)
# # optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

# # epochs = 200
# # batch_size = 64
# # temp = 0.1
# # sparsity_weight = 0.2

# # print(f"Training Config: temp={temp}, sparsity_weight={sparsity_weight}\n", flush=True)

# # for epoch in range(epochs):
# #     np.random.shuffle(triplets)
# #     total_loss = 0
    
# #     for i in range(0, len(triplets), batch_size):
# #         batch = triplets[i:i + batch_size]
# #         a = torch.tensor([t[0] for t in batch]).to(device)
# #         p = torch.tensor([t[1] for t in batch]).to(device)
# #         n = torch.tensor([t[2] for t in batch]).to(device)

# #         z_a, logits_a = encoder(X_torch[a], temp=temp)
# #         z_p, _ = encoder(X_torch[p], temp=temp)

# #         n_flat = n.view(-1)
# #         z_n_flat, _ = encoder(X_torch[n_flat], temp=temp)
# #         z_n = z_n_flat.view(n.shape[0], n.shape[1], -1)

# #         loss_nce = info_nce_loss(z_a, z_p, z_n)
# #         loss_sparse = logit_sparsity_loss(logits_a)
# #         loss = loss_nce + (sparsity_weight * loss_sparse)

# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()

# #     if (epoch + 1) % 10 == 0:
# #         avg_loss = total_loss / (len(triplets) / batch_size)
# #         print(f"Epoch {epoch+1:3d} | Loss {avg_loss:.4f}", flush=True)

# # print()

# # # === INFERENCE & ANALYSIS ===

# # print("=" * 80, flush=True)
# # print("PHASE 4: INFERENCE & ANALYSIS", flush=True)
# # print("=" * 80 + "\n", flush=True)

# # encoder.eval()

# # with torch.no_grad():
# #     z_all = []
# #     for i in range(0, len(X_torch), 256):
# #         z_batch, _ = encoder(X_torch[i:i+256], temp=0.05)
# #         z_all.append(z_batch)
# #     z_all = torch.cat(z_all, dim=0)

# # states = torch.argmax(z_all, dim=1).cpu().numpy()
# # usage = np.bincount(states, minlength=encoder.num_states)

# # print("--- STATE USAGE ---", flush=True)
# # for i, c in enumerate(usage):
# #     pct = 100 * c / len(states)
# #     status = "DEAD" if c == 0 else ""
# #     print(f"State {i}: {c:5d} ({pct:5.1f}%) {status}", flush=True)

# # mask = np.array([f["is_anchor"] for f in all_features])
# # true = np.array([f["stage_idx"] for f in all_features])[mask]
# # pred = states[mask]

# # stage_names = sorted(stage_to_idx.keys())
# # state_to_labels = defaultdict(lambda: defaultdict(int))

# # for t, s in zip(true, pred):
# #     state_to_labels[s][t] += 1

# # print("\n--- TOP LABELS PER STATE ---", flush=True)
# # for s in range(encoder.num_states):
# #     labels = state_to_labels[s]
# #     if not labels:
# #         print(f"State {s}: (unused)", flush=True)
# #         continue
# #     top = sorted(labels.items(), key=lambda x: x[1], reverse=True)[:3]
# #     print(f"State {s}: {', '.join(f'{stage_names[l]}({c})' for l, c in top)}", flush=True)

# # ami = adjusted_mutual_info_score(true, pred)
# # nmi = normalized_mutual_info_score(true, pred)
# # print(f"\nAMI: {ami:.4f} | NMI: {nmi:.4f}", flush=True)

# # pid_to_idx = defaultdict(list)
# # for i, f in enumerate(all_features):
# #     pid_to_idx[f["problem_id"]].append(i)

# # persistence_scores = []
# # for pid, idxs in pid_to_idx.items():
# #     if len(idxs) < 2:
# #         continue
# #     seq = states[idxs]
# #     switches = sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))
# #     persistence = 1 - switches / (len(seq) - 1)
# #     persistence_scores.append(persistence)

# # print("\n--- PERSISTENCE ---", flush=True)
# # print(f"Mean:   {np.mean(persistence_scores):.3f}", flush=True)
# # print(f"Median: {np.median(persistence_scores):.3f}", flush=True)
# # print(f"Std:    {np.std(persistence_scores):.3f}", flush=True)

# # state_durations = defaultdict(list)
# # for pid, idxs in pid_to_idx.items():
# #     seq = states[idxs]
# #     current = seq[0]
# #     dur = 1
# #     for s in seq[1:]:
# #         if s == current:
# #             dur += 1
# #         else:
# #             state_durations[current].append(dur)
# #             current = s
# #             dur = 1
# #     state_durations[current].append(dur)

# # print("\n--- STATE DURATIONS ---", flush=True)
# # for s in range(encoder.num_states):
# #     durs = state_durations[s]
# #     if durs:
# #         print(f"State {s}: mean={np.mean(durs):.1f}, min={np.min(durs)}, max={np.max(durs)}, n={len(durs)}", flush=True)
# #     else:
# #         print(f"State {s}: (never occurred)", flush=True)

# # transitions = np.zeros((encoder.num_states, encoder.num_states))
# # for i in range(len(states) - 1):
# #     if all_features[i]["problem_id"] == all_features[i+1]["problem_id"]:
# #         transitions[states[i], states[i + 1]] += 1

# # row_sums = transitions.sum(axis=1, keepdims=True)
# # trans_probs = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums != 0)

# # fig, ax = plt.subplots(figsize=(10, 8))
# # im = ax.imshow(trans_probs, cmap="Greens", vmin=0, vmax=1, aspect="auto")
# # ax.set_title("State Transition Probability Matrix")
# # ax.set_xlabel("Next State")
# # ax.set_ylabel("Current State")
# # ax.set_xticks(range(encoder.num_states))
# # ax.set_yticks(range(encoder.num_states))
# # fig.colorbar(im, ax=ax)
# # plt.tight_layout()
# # plt.show()

# # print("\n✓ RPC-CEBRA pipeline complete", flush=True)




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import pickle
# import os
# import gc
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
# from collections import defaultdict, Counter
# from tqdm import tqdm

# device = "cuda" if torch.cuda.is_available() else "cpu"
# torch.manual_seed(0)
# np.random.seed(0)

# checkpoint_dir = "rpc_cebra_window"

# print("=" * 80, flush=True)
# print("PHASE 1: LOAD CACHED WINDOWED FEATURES", flush=True)
# print("=" * 80 + "\n", flush=True)

# ckpt_features = f"{checkpoint_dir}/windowed_features.pkl"
# if not os.path.exists(ckpt_features):
#     print(f"⚠ Feature file not found: {ckpt_features}", flush=True)
#     exit(1)

# print(f"Loading cached windowed features...", flush=True)
# all_features = pickle.load(open(ckpt_features, 'rb'))
# print(f"✓ Loaded {len(all_features)} windowed embeddings\n", flush=True)

# X = np.array([f["hidden_state"] for f in all_features])
# X = StandardScaler().fit_transform(X)
# X_torch = torch.tensor(X, dtype=torch.float32).to(device)

# stage_to_idx = {s: i for i, s in enumerate(sorted({f["stage"] for f in all_features}))}
# for f in all_features:
#     f["stage_idx"] = stage_to_idx[f["stage"]]
#     f["is_labeled"] = f["stage"] != "NEUTRAL"

# print(f"Data shape: {X.shape}", flush=True)
# print(f"Labeled: {sum(f['is_labeled'] for f in all_features)} / {len(all_features)}", flush=True)

# stage_dist = Counter([f['stage'] for f in all_features if f['is_labeled']])
# print("\nStage Distribution (windowed):", flush=True)
# for stage in sorted(stage_dist.keys()):
#     count = stage_dist[stage]
#     pct = 100 * count / sum(stage_dist.values())
#     print(f"  {stage:25s}: {count:5d} ({pct:5.1f}%)", flush=True)
# print()

# # === BUILD TRIPLETS ===

# print("=" * 80, flush=True)
# print("PHASE 2: BUILD TRIPLETS", flush=True)
# print("=" * 80 + "\n", flush=True)

# def build_balanced_triplets(all_features, time_window=2, hard_lag=4,
#                             num_negatives=8, repeat=10):
#     """Build triplets with temporal locality"""
#     pid_to_idx = defaultdict(list)
    
#     for i, f in enumerate(all_features):
#         pid_to_idx[f["problem_id"]].append(i)
    
#     valid_idx = [i for i, f in enumerate(all_features) if f["is_labeled"]]
#     triplets = []
    
#     for _ in range(repeat):
#         for anchor in valid_idx:
#             pid = all_features[anchor]["problem_id"]
#             local = pid_to_idx[pid]
#             t = local.index(anchor)
            
#             pos = []
#             for dt in range(1, time_window + 1):
#                 if t + dt < len(local):
#                     pos.append(local[t + dt])
#                 if t - dt >= 0:
#                     pos.append(local[t - dt])
            
#             if not pos:
#                 continue
#             pos = np.random.choice(pos)
            
#             negs = []
#             for dt in range(hard_lag, len(local) - t):
#                 if t + dt < len(local):
#                     negs.append(local[t + dt])
            
#             other_pids = [p for p in pid_to_idx if p != pid]
#             while len(negs) < num_negatives:
#                 other_pid = np.random.choice(other_pids)
#                 negs.append(np.random.choice(pid_to_idx[other_pid]))
            
#             triplets.append((anchor, pos, negs[:num_negatives]))
    
#     return triplets

# triplets = build_balanced_triplets(all_features, time_window=2, hard_lag=4, repeat=15)
# print(f"✓ Built {len(triplets)} triplets\n", flush=True)

# # === MODEL ===

# def gumbel_softmax_ste(logits, tau=1.0, eps=1e-20):
#     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
#     y = logits + gumbel_noise
#     y_soft = F.softmax(y / tau, dim=-1)
#     y_hard = torch.zeros_like(y_soft).scatter_(
#         1, y_soft.argmax(dim=1, keepdim=True), 1.0
#     )
#     return y_hard - y_soft.detach() + y_soft

# # class TopKEncoder(nn.Module):
# #     def __init__(self, d_input, num_states=4):
# #         super().__init__()
# #         self.num_states = num_states
# #         self.feature_extractor = nn.Sequential(
# #             nn.Linear(d_input, 1024),
# #             nn.LayerNorm(1024),
# #             nn.ReLU(),
# #             nn.Linear(1024, 32),
# #             nn.LayerNorm(32),
# #             nn.ReLU()
# #         )
# #         self.state_head = nn.Sequential(
# #             nn.Linear(32, num_states),
# #             nn.LayerNorm(num_states)
# #         )

# #     def forward(self, x, temp=0.1):
# #         latent = self.feature_extractor(x)
# #         logits = self.state_head(latent)
# #         z = gumbel_softmax_ste(logits, tau=temp)
# #         return z, logits


# class VoronoiEncoder(nn.Module):
#     def __init__(self, d_input, num_states=4):
#         super().__init__()
#         self.num_states = num_states

#         self.embed = nn.Sequential(
#             nn.Linear(d_input, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64)
#         )

#         # One centroid per state (learned)
#         self.centroids = nn.Parameter(torch.randn(num_states, 64))

#     def forward(self, x, temp=0.1):
#         h = F.normalize(self.embed(x), dim=1)
#         c = F.normalize(self.centroids, dim=1)

#         # squared distances to centroids
#         d2 = torch.cdist(h, c, p=2)**2

#         logits = -d2
#         z = gumbel_softmax_ste(logits, tau=temp)

#         return z, logits, h



# def info_nce_loss(z_a, z_p, z_n, temperature=0.1):
#     pos_sim = torch.sum(z_a * z_p, dim=-1, keepdim=True)
#     neg_sim = torch.matmul(z_a.unsqueeze(1), z_n.transpose(-1, -2)).squeeze(1)
#     logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
#     labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
#     return F.cross_entropy(logits, labels)

# def logit_sparsity_loss(logits):
#     return torch.mean(torch.abs(logits))

# # === TRAINING ===

# print("=" * 80, flush=True)
# print("PHASE 3: TRAINING", flush=True)
# print("=" * 80 + "\n", flush=True)

# # encoder = TopKEncoder(X.shape[1], num_states=4).to(device)
# encoder = VoronoiEncoder(X.shape[1], num_states=4).to(device)

# optimizer = optim.Adam(encoder.parameters(), lr=1e-4)

# epochs = 200
# batch_size = 128
# temp = 0.1
# sparsity_weight = 0.5

# print(f"Training Config:", flush=True)
# print(f"  Temp: {temp}", flush=True)
# print(f"  Sparsity Weight: {sparsity_weight}", flush=True)
# print(f"  Batch Size: {batch_size}", flush=True)
# print(f"  Learning Rate: 1e-4\n", flush=True)

# for epoch in range(epochs):
#     np.random.shuffle(triplets)
#     total_loss_nce = 0
#     total_loss_sparse = 0
#     total_loss = 0
#     num_batches = 0
    
#     for i in range(0, len(triplets), batch_size):
#         batch = triplets[i:i + batch_size]
#         a = torch.tensor([t[0] for t in batch]).to(device)
#         p = torch.tensor([t[1] for t in batch]).to(device)
#         n = torch.tensor([t[2] for t in batch]).to(device)

#         z_a, logits_a = encoder(X_torch[a], temp=temp)
#         z_p, _ = encoder(X_torch[p], temp=temp)

#         n_flat = n.view(-1)
#         z_n_flat, _ = encoder(X_torch[n_flat], temp=temp)
#         z_n = z_n_flat.view(n.shape[0], n.shape[1], -1)

#         loss_nce = info_nce_loss(z_a, z_p, z_n, temperature=0.1)
#         loss_sparse = logit_sparsity_loss(logits_a)
#         loss = loss_nce + (sparsity_weight * loss_sparse)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss_nce += loss_nce.item()
#         total_loss_sparse += loss_sparse.item()
#         total_loss += loss.item()
#         num_batches += 1

#     if (epoch + 1) % 10 == 0:
#         avg_loss = total_loss / num_batches
#         avg_nce = total_loss_nce / num_batches
#         avg_sparse = total_loss_sparse / num_batches
#         print(f"Epoch {epoch+1:3d} | Loss {avg_loss:.4f} (NCE: {avg_nce:.4f}, Sparse: {avg_sparse:.4f})", flush=True)

# print()

# # === INFERENCE & ANALYSIS ===

# print("=" * 80, flush=True)
# print("PHASE 4: INFERENCE & ANALYSIS", flush=True)
# print("=" * 80 + "\n", flush=True)

# encoder.eval()

# with torch.no_grad():
#     z_all = []
#     for i in range(0, len(X_torch), 256):
#         z_batch, _ = encoder(X_torch[i:i+256], temp=0.05)
#         z_all.append(z_batch)
#     z_all = torch.cat(z_all, dim=0)

# states = torch.argmax(z_all, dim=1).cpu().numpy()
# usage = np.bincount(states, minlength=encoder.num_states)

# print("--- STATE USAGE ---", flush=True)
# for i, c in enumerate(usage):
#     pct = 100 * c / len(states)
#     status = "DEAD" if c == 0 else ""
#     print(f"State {i}: {c:6d} ({pct:5.1f}%) {status}", flush=True)

# mask = np.array([f["is_labeled"] for f in all_features])
# true = np.array([f["stage_idx"] for f in all_features])[mask]
# pred = states[mask]

# stage_names = sorted(stage_to_idx.keys())
# state_to_labels = defaultdict(lambda: defaultdict(int))

# for t, s in zip(true, pred):
#     state_to_labels[s][t] += 1

# print("\n--- TOP LABELS PER STATE ---", flush=True)
# for s in range(encoder.num_states):
#     labels = state_to_labels[s]
#     if not labels:
#         print(f"State {s}: (unused)", flush=True)
#         continue
#     top = sorted(labels.items(), key=lambda x: x[1], reverse=True)[:3]
#     print(f"State {s}: {', '.join(f'{stage_names[l]}({c})' for l, c in top)}", flush=True)

# ami = adjusted_mutual_info_score(true, pred)
# nmi = normalized_mutual_info_score(true, pred)
# print(f"\nAMI: {ami:.4f} | NMI: {nmi:.4f}", flush=True)

# pid_to_idx = defaultdict(list)
# for i, f in enumerate(all_features):
#     pid_to_idx[f["problem_id"]].append(i)

# persistence_scores = []
# for pid, idxs in pid_to_idx.items():
#     if len(idxs) < 2:
#         continue
#     seq = states[idxs]
#     switches = sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))
#     persistence = 1 - switches / (len(seq) - 1)
#     persistence_scores.append(persistence)

# print("\n--- PERSISTENCE ---", flush=True)
# print(f"Mean:   {np.mean(persistence_scores):.3f}", flush=True)
# print(f"Median: {np.median(persistence_scores):.3f}", flush=True)
# print(f"Std:    {np.std(persistence_scores):.3f}", flush=True)

# state_durations = defaultdict(list)
# for pid, idxs in pid_to_idx.items():
#     seq = states[idxs]
#     current = seq[0]
#     dur = 1
#     for s in seq[1:]:
#         if s == current:
#             dur += 1
#         else:
#             state_durations[current].append(dur)
#             current = s
#             dur = 1
#     state_durations[current].append(dur)

# print("\n--- STATE DURATIONS ---", flush=True)
# for s in range(encoder.num_states):
#     durs = state_durations[s]
#     if durs:
#         print(f"State {s}: mean={np.mean(durs):.1f}, min={np.min(durs)}, max={np.max(durs)}, n={len(durs)}", flush=True)
#     else:
#         print(f"State {s}: (never occurred)", flush=True)

# transitions = np.zeros((encoder.num_states, encoder.num_states))
# for i in range(len(states) - 1):
#     if all_features[i]["problem_id"] == all_features[i+1]["problem_id"]:
#         transitions[states[i], states[i + 1]] += 1

# row_sums = transitions.sum(axis=1, keepdims=True)
# trans_probs = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums != 0)

# fig, ax = plt.subplots(figsize=(10, 8))
# im = ax.imshow(trans_probs, cmap="Greens", vmin=0, vmax=1, aspect="auto")
# ax.set_title("State Transition Probability Matrix")
# ax.set_xlabel("Next State")
# ax.set_ylabel("Current State")
# ax.set_xticks(range(encoder.num_states))
# ax.set_yticks(range(encoder.num_states))
# fig.colorbar(im, ax=ax)
# plt.tight_layout()
# plt.show()

# print("\n✓ RPC-CEBRA training complete", flush=True)
