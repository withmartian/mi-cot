"""
layer_ablation_generate.py

Sweeps all models × datasets × layer ablation schedules automatically.
For each (model, dataset), generates CoTs, extracts hidden states across
the layer schedule, classifies sentences, and computes all SLDS metrics.

Layer schedules (from main experiment):
  Llama-8B:  every 3rd layer (0,3,6,...,30,31)
  Qwen-14B:  every 3rd layer (0,3,6,...,45,47)
  Qwen-1.5B: every 2nd layer (0,2,4,...,26,27)

Usage:
    python layer_ablation_generate.py \
        --out /home/abir19/scratch/abir19/SDS_layer_ablation \
        --n 200
"""

import os, re, gc, pickle, argparse, json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16
print(f"[init] device={DEVICE}", flush=True)

PCA_DIM = 40
KAPPA   = 10.0
N_ITERS = 50
K_EVAL  = 5

CLASSES_ORDERED = [
    "PROBLEM_SETUP", "FACT_RETRIEVAL", "PLAN_GENERATION",
    "UNCERTAINTY_MANAGEMENT", "SELF_CHECKING", "RESULT_CONSOLIDATION",
    "ACTIVE_COMPUTATION", "FINAL_ANSWER_EMISSION"
]

# ── SWEEP CONFIG ─────────────────────────────────────────────
# (model_key, rlvr_hf_id, base_hf_id, tokenizer_base, layer_schedule)
MODELS = {
    # "llama8b": {
    #     "rlvr":     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    #     "base":     "meta-llama/Llama-3.1-8B",
    #     "tok_base": "meta-llama/Llama-3.1-8B",
    #     "layers":   list(range(0, 31, 3)) + [31],
    #     "model_type": "llama",
    # },
    "qwen14b": {
        "rlvr":     "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "base":     "Qwen/Qwen2.5-14B",
        "tok_base": "Qwen/Qwen2.5-14B",
        "layers":   list(range(0, 48, 3)) + [47],
        "model_type": "qwen14",
    },
    "qwen1.5b": {
        "rlvr":     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "base":     "Qwen/Qwen2.5-1.5B",
        "tok_base": "Qwen/Qwen2.5-1.5B",
        "layers":   list(range(0, 28, 2)) + [27],
        "model_type": "qwen1",
    },
}

DATASETS = {
    "gsm8k":    {"hf_id": "openai/gsm8k",             "config": "main",    "split": "train"},
    "math500":  {"hf_id": "HuggingFaceH4/MATH-500",   "config": "default", "split": "test"},
    "svamp":    {"hf_id": "ChilleD/SVAMP",             "config": "default", "split": "train"},
    "mmlu-pro": {"hf_id": "TIGER-Lab/MMLU-Pro",        "config": "default", "split": "test"},
}

CLF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── ARGS ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",     default="/home/abir19/scratch/abir19/SDS_layer_ablation")
    p.add_argument("--n",       type=int, default=200)
    p.add_argument("--batch",   type=int, default=4)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--models",  nargs="+", default=list(MODELS.keys()),
                   help="Subset of models to run (default: all)")
    p.add_argument("--datasets",nargs="+", default=list(DATASETS.keys()),
                   help="Subset of datasets to run (default: all)")
    p.add_argument("--conditions", nargs="+", default=["rlvr", "base"],
                   choices=["rlvr", "base"])
    return p.parse_args()

# ── DATASET HELPERS ──────────────────────────────────────────

def get_problem_from_row(row, dataset_name):
    name = dataset_name.lower()
    if "mmlu" in name and "question" in row:
        q = row["question"]
        if "options" in row and row["options"]:
            opts = row["options"]
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            parts = [f"{letters[i]}. {opts[i]}" for i in range(min(len(letters), len(opts)))]
            return q + "\n\n" + "\n".join(parts)
        return q
    if "problem"  in row: return row["problem"]
    if "question" in row: return row["question"]
    raise KeyError(f"No problem/question column in {list(row.keys())}")

def load_problems(ds_cfg, n):
    from datasets import load_dataset
    ds = load_dataset(ds_cfg["hf_id"], ds_cfg["config"],
                      split=f"{ds_cfg['split']}[:{n}]")
    return [get_problem_from_row(ds[i], ds_cfg["hf_id"]) for i in range(len(ds))]

# ── GENERATION + EXTRACTION ──────────────────────────────────

def split_into_sentences(text):
    text = re.sub(r'<think>|</think>', '', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

@torch.no_grad()
def generate_cot(problem, model, tokenizer):
    input_ids = tokenizer.encode(problem, return_tensors='pt').to(DEVICE)
    output_ids = model.generate(
        input_ids, max_new_tokens=1024, temperature=0.6,
        do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    cot = full_text[len(problem):].strip() if full_text.startswith(problem) else full_text
    return cot

@torch.no_grad()
def extract_layer(problem, cot, model, tokenizer, layer_idx):
    full_prompt = problem + " " + cot
    prompt_ids  = tokenizer.encode(full_prompt, return_tensors="pt").to(DEVICE)
    seq_len     = prompt_ids.size(1)
    sentences   = split_into_sentences(cot)
    if len(sentences) < 2: return None

    hidden_states = {}
    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_states['h'] = h[0].float().cpu()
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    model(prompt_ids)
    handle.remove()

    hidden   = hidden_states['h']
    full_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    tokens   = [tokenizer.decode([tid]) for tid in full_ids]
    cumchars, c2t = 0, {}
    for i, tok in enumerate(tokens):
        for _ in range(len(tok)): c2t[cumchars] = i; cumchars += 1

    feats, cur = [], 0
    for si, sent in enumerate(sentences):
        sc = full_prompt.find(sent, cur)
        if sc == -1: sc = cur
        ec = sc + len(sent)
        st = c2t.get(sc, 0)
        et = c2t.get(min(ec-1, max(c2t.keys())), len(full_ids)-1) + 1
        et = min(et, seq_len)
        if st >= seq_len or et <= 0: continue
        feats.append({
            'sentence_idx':      si,
            'sentence':          sent,
            'hidden_state_last': hidden[et-1].numpy(),
        })
        cur = ec
    return feats

# ── CLASSIFICATION ────────────────────────────────────────────

def get_clf_prompt(sentence, clf_tok):
    messages = [
        {"role": "system", "content": "You are a classifier. Reply with ONLY the category name, nothing else."},
        {"role": "user",   "content": f"""Classify this reasoning step:

PROBLEM_SETUP / FACT_RETRIEVAL / PLAN_GENERATION / UNCERTAINTY_MANAGEMENT /
SELF_CHECKING / RESULT_CONSOLIDATION / ACTIVE_COMPUTATION / FINAL_ANSWER_EMISSION

Sentence: "{sentence[:300]}"

Reply with ONLY the category name."""}
    ]
    return clf_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def classify_sentences(sentences, clf_tok, clf_model, batch_size=4):
    results = []
    for i in range(0, len(sentences), batch_size):
        batch   = sentences[i:i+batch_size]
        prompts = [get_clf_prompt(s, clf_tok) for s in batch]
        inputs  = clf_tok(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(DEVICE)
        outputs = clf_model.generate(
            inputs['input_ids'], attention_mask=inputs['attention_mask'],
            max_new_tokens=10, do_sample=False, pad_token_id=clf_tok.pad_token_id
        )
        for out in outputs:
            resp = clf_tok.decode(out[inputs['input_ids'].shape[1]:],
                                  skip_special_tokens=True).strip().upper()
            matched = "NEUTRAL"
            for cls in reversed(CLASSES_ORDERED):
                if cls in resp: matched = cls; break
            results.append(matched)
        if (i // batch_size) % 50 == 0:
            torch.cuda.empty_cache()
    return results

# ── SLDS + METRICS ────────────────────────────────────────────

def linear_ar_r2(seqs):
    X_in  = np.vstack([s[:-1] for s in seqs])
    X_out = np.vstack([s[1:]  for s in seqs])
    X_aug = np.hstack([X_in, np.ones((len(X_in), 1))])
    coef, *_ = np.linalg.lstsq(X_aug, X_out, rcond=None)
    pred = X_aug @ coef
    return 1 - np.sum((X_out-pred)**2) / np.sum((X_out-X_out.mean(0))**2)

def init_params(seqs, K):
    X_in  = np.vstack([s[:-1] for s in seqs])
    X_out = np.vstack([s[1:]-s[:-1] for s in seqs])
    labs  = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X_out)
    D     = PCA_DIM
    dM, db, dCov = np.zeros((K,D,D)), np.zeros((K,D)), np.array([np.eye(D)]*K)
    for k in range(K):
        m = labs==k
        if m.sum() < D+2: dM[k]=0.1*np.eye(D); continue
        W,*_ = np.linalg.lstsq(np.hstack([X_in[m],np.ones((m.sum(),1))]),X_out[m],rcond=None)
        dM[k],db[k] = W[:D].T, W[D]
        res = X_out[m]-(X_in[m]@dM[k].T+db[k])
        dCov[k] = np.cov(res.T)+1e-3*np.eye(D)
    return np.ones(K)/K, np.eye(K)*0.7+0.3/K, dM, db, dCov

def forward_backward(seq, pi, A, dM, db, dCov, K):
    T, D = len(seq), PCA_DIM
    log_emit = np.zeros((T,K))
    for k in range(K):
        _,logdet = np.linalg.slogdet(dCov[k])
        inv_cov  = np.linalg.inv(dCov[k])
        means    = np.vstack([db[k], seq[:-1]@dM[k].T+db[k]])
        diffs    = seq - means
        log_emit[:,k] = -0.5*(D*np.log(2*np.pi)+logdet+np.sum((diffs@inv_cov)*diffs,1))
    log_A, log_pi = np.log(A+1e-12), np.log(pi+1e-12)
    la = np.zeros((T,K)); la[0] = log_pi+log_emit[0]
    for t in range(1,T):
        la[t] = log_emit[t]+np.logaddexp.reduce(la[t-1][:,None]+log_A,0)
    lb = np.zeros((T,K))
    for t in range(T-2,-1,-1):
        lb[t] = np.logaddexp.reduce(log_A+log_emit[t+1]+lb[t+1],1)
    lg = la+lb; lg -= np.logaddexp.reduce(lg,1,keepdims=True)
    lxi = np.zeros((T-1,K,K))
    for t in range(T-1):
        lxi[t] = la[t][:,None]+log_A+log_emit[t+1]+lb[t+1]
        lxi[t] -= np.logaddexp.reduce(lxi[t].ravel())
    return np.exp(lg), np.exp(lxi), np.logaddexp.reduce(la[-1])

def m_step(seqs, gammas, xis, K):
    D = PCA_DIM
    xi_sum = sum(x.sum(0) for x in xis)+np.eye(K)*KAPPA+1e-8
    A_new  = xi_sum/xi_sum.sum(1,keepdims=True)
    pi_new = np.maximum(np.mean([g[0] for g in gammas],0),1e-8); pi_new/=pi_new.sum()
    dM,db,dCov = np.zeros((K,D,D)),np.zeros((K,D)),np.zeros((K,D,D))
    for k in range(K):
        Ws,WY = np.zeros((D+1,D+1)),np.zeros((D+1,D))
        for seq,g in zip(seqs,gammas):
            Xa = np.hstack([np.vstack([np.zeros(D),seq[:-1]]),np.ones((len(seq),1))])
            w  = g[:,k]
            Ws += (Xa*w[:,None]).T@Xa; WY += (Xa*w[:,None]).T@seq
        c = np.linalg.solve(Ws+1e-4*np.eye(D+1),WY)
        dM[k],db[k] = c[:D].T, c[D]
        num,den = np.zeros((D,D)),1e-9
        for seq,g in zip(seqs,gammas):
            err = seq-(np.vstack([np.zeros(D),seq[:-1]])@dM[k].T+db[k])
            num+=(err*g[:,k][:,None]).T@err; den+=g[:,k].sum()
        dCov[k] = num/den+1e-4*np.eye(D)
    return pi_new, A_new, dM, db, dCov

def hard_tm(state_seqs, K):
    T = np.zeros((K,K))
    for s in state_seqs:
        for a,b in zip(s[:-1],s[1:]): T[a,b]+=1
    r = T.sum(1,keepdims=True)
    return T/np.where(r==0,1,r)

def tqs(T, K):
    if K==1: return 0.0
    if np.min(np.diag(T)) < 0.3: return 0.0
    H_mean = np.mean(-np.sum(T*np.log(T+1e-12),axis=1)/np.log(K))
    return (1-np.trace(T)/K)*(1-H_mean)

def sss(persist, mean_st, T, K, K_eff):
    if K==1: return 0.0
    row_ents = -np.sum(T*np.log(T+1e-12),1)/np.log(K)
    return (K_eff/K)*mean_st*np.log(persist+1)*(1-np.mean(row_ents))

def compute_bic(total_ll, K, N):
    n_params = K*(K-1)+K*(PCA_DIM*PCA_DIM+PCA_DIM+PCA_DIM*(PCA_DIM+1)//2)
    return -2*total_ll+n_params*np.log(N)

def tvd_from_uniform(T):
    K = len(T); u = np.ones((K,K))/K
    return float(np.mean(0.5*np.abs(T-u).sum(axis=1)))

def spectral_gap(T):
    K = len(T)
    if K <= 1: return 0.0
    return float(1-np.sort(np.abs(np.linalg.eigvals(T)))[::-1][1])

def kl_from_uniform(T):
    K = len(T); u = np.ones((K,K))/K
    return float(np.mean(np.sum(T*np.log(T/(u+1e-12)+1e-12),axis=1)))

def stationary_entropy(T):
    ev,evec = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(ev-1.0))
    stat = np.abs(evec[:,idx].real); stat/=stat.sum()
    return float(-np.sum(stat*np.log(stat+1e-12)))

def compute_metrics_for_layer(features):
    p_map = defaultdict(list)
    for f in features:
        p_map[f['problem_id']].append((f['sentence_idx'], f['hidden_state_last']))
    pids  = sorted(p for p in p_map if len(p_map[p]) >= 3)
    X_all = np.vstack([v for feats in p_map.values() for _, v in feats])
    X_pca = PCA(n_components=PCA_DIM, random_state=42).fit_transform(
                StandardScaler().fit_transform(X_all))
    idx = 0; pca_map = {}
    for pid in sorted(p_map.keys()):
        n = len(p_map[pid]); pca_map[pid] = X_pca[idx:idx+n]; idx += n
    seqs = [np.array([pca_map[pid][i]
                      for i,_ in enumerate(sorted(p_map[pid], key=lambda x:x[0]))])
            for pid in pids]

    ar_r2 = linear_ar_r2(seqs)
    K = K_EVAL
    pi,A,dM,db,dCov = init_params(seqs, K)
    for _ in range(N_ITERS):
        gammas,xis,lls = [],[],[]
        for s in seqs:
            g,x,ll = forward_backward(s,pi,A,dM,db,dCov,K)
            gammas.append(g); xis.append(x); lls.append(ll)
        pi,A,dM,db,dCov = m_step(seqs,gammas,xis,K)
    total_ll = sum(lls)
    N   = sum(len(s) for s in seqs)
    bic = compute_bic(total_ll, K, N)
    state_seqs = [np.argmax(g,1) for g in gammas]
    all_s  = np.concatenate(state_seqs)
    counts = np.bincount(all_s, minlength=K)
    K_eff  = int(np.sum(counts/len(all_s)>0.01))
    persist= np.mean([len(s)/(np.count_nonzero(np.diff(s))+1) for s in state_seqs])
    T_hard = hard_tm(state_seqs, K)
    mean_st= np.diag(T_hard).mean()
    return {
        "sss":               float(sss(persist, mean_st, T_hard, K, K_eff)),
        "tqs":               float(tqs(T_hard, K)),
        "bic":               float(bic),
        "persist":           float(persist),
        "mean_self_trans":   float(mean_st),
        "k_eff":             K_eff,
        "ar_r2":             float(ar_r2),
        "tvd":               tvd_from_uniform(T_hard),
        "spectral_gap":      spectral_gap(T_hard),
        "kl_uniform":        kl_from_uniform(T_hard),
        "stationary_entropy":stationary_entropy(T_hard),
    }

# ── PER-RUN PIPELINE ─────────────────────────────────────────

def run_one(model_key, condition, ds_key, out_base, args):
    """
    Runs the full pipeline for one (model, condition, dataset) triple.
    out_dir = out_base / {model_key}_{condition} / {ds_key}
    Skips any layer already extracted.
    """
    mcfg = MODELS[model_key]
    dcfg = DATASETS[ds_key]
    hf_id = mcfg[condition]  # "rlvr" or "base"
    layers = mcfg["layers"]

    out_dir = os.path.join(out_base, f"{model_key}_{condition}", ds_key)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "layer_metrics.json")
    if os.path.exists(metrics_path):
        existing = json.load(open(metrics_path))
        if len(existing.get("layers", {})) == len(layers):
            print(f"  [{model_key}/{condition}/{ds_key}] already complete, skipping", flush=True)
            return

    # ── problems ──
    print(f"\n{'='*60}", flush=True)
    print(f"  {model_key} | {condition} | {ds_key}", flush=True)
    print(f"  hf_id={hf_id}  layers={layers}", flush=True)
    problems = load_problems(dcfg, args.n)
    print(f"  {len(problems)} problems loaded", flush=True)

    # ── CoTs ──
    cot_path = os.path.join(out_dir, "cots.pkl")
    if os.path.exists(cot_path):
        cots = pickle.load(open(cot_path, 'rb'))
        print(f"  CoTs loaded ({len(cots)})", flush=True)
    else:
        print(f"  Generating CoTs...", flush=True)
        tok = AutoTokenizer.from_pretrained(mcfg["tok_base"], trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        gen_model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE)
        gen_model.eval()
        cots = {}
        for pid, prob in tqdm(enumerate(problems), total=len(problems)):
            cots[pid] = generate_cot(prob, gen_model, tok)
            if (pid+1) % 25 == 0:
                pickle.dump(cots, open(cot_path, 'wb'))
                torch.cuda.empty_cache(); gc.collect()
        pickle.dump(cots, open(cot_path, 'wb'))
        del gen_model; torch.cuda.empty_cache(); gc.collect()

    # ── hidden state extraction ──
    layers_to_extract = [l for l in layers
                         if not os.path.exists(os.path.join(out_dir, f"layer_{l}_raw.pkl"))]
    if layers_to_extract:
        print(f"  Extracting layers {layers_to_extract}...", flush=True)
        tok = AutoTokenizer.from_pretrained(mcfg["tok_base"], trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        gen_model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE)
        gen_model.eval()
        for layer_idx in layers_to_extract:
            print(f"    Layer {layer_idx}...", flush=True)
            layer_feats = {}
            for pid, prob in tqdm(enumerate(problems), total=len(problems)):
                if pid not in cots: continue
                feats = extract_layer(prob, cots[pid], gen_model, tok, layer_idx)
                if feats: layer_feats[pid] = feats
            pickle.dump(layer_feats,
                        open(os.path.join(out_dir, f"layer_{layer_idx}_raw.pkl"), 'wb'))
        del gen_model; torch.cuda.empty_cache(); gc.collect()

    # ── classification (once, shared across layers) ──
    clf_path = os.path.join(out_dir, "classifications.pkl")
    if os.path.exists(clf_path):
        classifications = pickle.load(open(clf_path, 'rb'))
        print(f"  Classifications loaded ({len(classifications)})", flush=True)
    else:
        print(f"  Classifying sentences...", flush=True)
        clf_tok = AutoTokenizer.from_pretrained(CLF_MODEL, trust_remote_code=True)
        clf_tok.pad_token = clf_tok.eos_token; clf_tok.padding_side = "left"
        clf_model = AutoModelForCausalLM.from_pretrained(
            CLF_MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE)
        clf_model.eval()
        ref_data = pickle.load(open(os.path.join(out_dir, f"layer_{layers[0]}_raw.pkl"), 'rb'))
        sents, refs = [], []
        for pid, feats in ref_data.items():
            for fi, f in enumerate(feats):
                sents.append(f['sentence']); refs.append((pid, fi))
        labels = classify_sentences(sents, clf_tok, clf_model, batch_size=args.batch)
        classifications = {(pid, fi): lbl for (pid, fi), lbl in zip(refs, labels)}
        pickle.dump(classifications, open(clf_path, 'wb'))
        del clf_model; torch.cuda.empty_cache(); gc.collect()

    # ── compute metrics per layer ──
    print(f"  Computing metrics...", flush=True)
    results = {}
    for layer_idx in layers:
        lp = os.path.join(out_dir, f"layer_{layer_idx}_raw.pkl")
        if not os.path.exists(lp): continue
        layer_data = pickle.load(open(lp, 'rb'))
        flat = []
        for pid, feats in layer_data.items():
            for fi, f in enumerate(feats):
                stage = classifications.get((pid, fi), "NEUTRAL")
                if stage == "NEUTRAL": continue
                flat.append({'problem_id': pid, 'sentence_idx': f['sentence_idx'],
                             'hidden_state_last': f['hidden_state_last'], 'stage': stage})
        if len(flat) < 50:
            print(f"    Layer {layer_idx}: too few features ({len(flat)}), skip"); continue
        m = compute_metrics_for_layer(flat)
        results[layer_idx] = m
        print(f"    L{layer_idx}: SSS={m['sss']:.4f} TVD={m['tvd']:.4f} "
              f"SpGap={m['spectral_gap']:.4f} BIC={m['bic']:.1f}", flush=True)

    with open(metrics_path, 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "model": model_key, "condition": condition,
                   "hf_id": hf_id, "dataset": ds_key,
                   "n_problems": args.n, "k_eval": K_EVAL,
                   "layers": results}, f, indent=2)
    print(f"  Saved -> {metrics_path}", flush=True)

# ── MAIN ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    runs = [(m, c, d)
            for m in args.models
            for c in args.conditions
            for d in args.datasets]

    print(f"[main] {len(runs)} runs: {args.models} × {args.conditions} × {args.datasets}")
    print(f"[main] out={args.out}  n={args.n}\n")

    for model_key, condition, ds_key in runs:
        try:
            run_one(model_key, condition, ds_key, args.out, args)
        except Exception as e:
            print(f"  FAILED {model_key}/{condition}/{ds_key}: {e}", flush=True)

    print("\nAll runs complete.")

if __name__ == "__main__":
    main()