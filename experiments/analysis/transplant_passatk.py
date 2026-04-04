"""
Pass@k evaluation of SDS-steered generation on base model.

Steering math is identical to cebra_em_steering_inputDep.py (Yonedo):
  p(k)       = A[s_t]                          (natural next-regime from transition matrix)
  q*(k)      ∝ p(k) exp(β r(k))               (KL-regularised tilt toward target_k)
  Δz_t       = Σ_k [q*(k)-p(k)] (A_k z_t+b_k) (input-dependent latent nudge)
  Δx_scaled  = Δz_t @ W_dec                    (linear decoder, same least-squares fit)
  Δx_raw     = Δx_scaled * scaler.scale_        (un-standardise)
  x_edit     = x + α * Δx_raw                  (residual stream patch via hook)
"""

import argparse, json, os, pickle, re, random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

import cebra_EM as cem


def extract_number(text):
    nums = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    return nums[-1] if nums else None

def extract_letter(text):
    m = re.search(r'(?:answer is|answer:|boxed\{)\s*([A-E])', text, re.IGNORECASE)
    if m: return m.group(1).upper()
    letters = re.findall(r'\b([A-E])\b', text)
    return letters[-1].upper() if letters else None

def is_correct(pred, gold, dataset=None):
    gold = str(gold).strip()
    if gold.upper() in 'ABCDE' and len(gold) == 1:
        return extract_letter(pred) == gold.upper()
    p, g = extract_number(pred), extract_number(gold)
    if p is None or g is None: return False
    try: return abs(float(p) - float(g)) < 1e-4
    except ValueError: return False

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))

def bootstrap_ci(counts_s, counts_b, n, k, n_boot=2000, seed=42):
    """Bootstrap CI over problems for Δpass@k on truly-hard subset."""
    rng = np.random.default_rng(seed)
    m = len(counts_s)
    if m == 0:
        return float('nan'), float('nan'), float('nan')
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        pk_s = np.mean([pass_at_k(n, counts_s[i], k) for i in idx])
        pk_b = np.mean([pass_at_k(n, counts_b[i], k) for i in idx])
        deltas.append(pk_s - pk_b)
    arr = np.array(deltas)
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def load_hard_problems(model_key, dataset_name):
    from datasets import load_dataset
    split = f"{model_key}_{dataset_name}"
    ds = load_dataset('abir-hr196/rlvr-hard-examples', split=split)
    n_wrong = sum(1 for r in ds if not r['base_correct'])
    print(f"  Loaded {len(ds)} hard examples ({n_wrong} base_correct=False) from split '{split}'", flush=True)
    return [{'q': r['problem'], 'a': r['answer']} for r in ds]


def fit_sds_and_decoder(pkl_path, K, cebra_dim, em_iters, limit_problems, max_triplets):
    all_features, triplets = cem.load_and_prepare_cebra(
        pkl_path, mode='temporal',
        limit_problems=limit_problems, max_triplets=max_triplets,
    )
    cebra_seqs, _, _, _, _ = cem.train_cebra_projection(all_features, triplets, d_out=cebra_dim)

    p_map = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map[int(f['problem_id'])].append(i)
    pids_sorted = sorted(p for p in p_map if len(p_map[p]) >= 3)
    idx_seqs = [p_map[pid] for pid in pids_sorted]

    pi, A, dM, db, dCov = cem.init_params(cebra_seqs, K, cebra_dim)
    for it in range(em_iters):
        gammas, xis, lls = [], [], []
        for seq in cebra_seqs:
            g, x, ll = cem.forward_backward(seq, pi, A, dM, db, dCov, K)
            gammas.append(g); xis.append(x); lls.append(ll)
        pi, A, dM, db, dCov = cem.m_step(cebra_seqs, gammas, xis, K, cebra_dim)
        if (it + 1) % 10 == 0:
            print(f"  EM iter {it+1}/{em_iters}", flush=True)

    X_raw = np.array([f['hidden_state_last'] for f in all_features], dtype=np.float32)
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    z_flat = np.concatenate(cebra_seqs, axis=0)
    x_flat = np.concatenate([X_scaled[idxs] for idxs in idx_seqs], axis=0)
    z_aug = np.hstack([z_flat, np.ones((len(z_flat), 1))])
    coef, *_ = np.linalg.lstsq(z_aug, x_flat, rcond=None)
    W_dec = coef[:-1]

    state_seqs = [np.argmax(g, axis=1) for g in gammas]
    centroids = []
    for k in range(K):
        vecs = [cebra_seqs[i][state_seqs[i] == k]
                for i in range(len(cebra_seqs)) if np.any(state_seqs[i] == k)]
        centroids.append(np.vstack(vecs).mean(0) if vecs else np.zeros(cebra_dim))

    return dict(pi=pi, A=A, dM=dM, db=db, dCov=dCov,
                W_dec=W_dec, scaler=scaler, K=K, cebra_dim=cebra_dim,
                cebra_centroids=np.array(centroids),
                W_dec_pinv=np.linalg.pinv(W_dec))


def kl_tilt(p, target_k, beta):
    p = p / (p.sum() + 1e-12)
    g = np.zeros_like(p); g[target_k] = 1.0
    q = p * np.exp(beta * g)
    return q / q.sum()

def steering_delta(z_t, s_t, target_k, sds, beta):
    p = sds['A'][s_t].copy(); p /= p.sum() + 1e-12
    q = kl_tilt(p, target_k, beta)
    f_k = np.stack([sds['dM'][k] @ z_t + sds['db'][k] for k in range(sds['K'])])
    mu_orig    = (p[:, None] * f_k).sum(0)
    mu_steered = (q[:, None] * f_k).sum(0)
    return mu_steered - mu_orig

def embed_hidden(h_np, sds):
    h_scaled = sds['scaler'].transform(h_np.reshape(1, -1))[0]
    return h_scaled @ sds['W_dec_pinv']

def infer_regime(z, sds):
    dists = [np.linalg.norm(z - sds['cebra_centroids'][k]) for k in range(sds['K'])]
    return int(np.argmin(dists))


def generate_steered(model, tokenizer, prompt, sds, layer_idx, alpha, beta,
                     max_new_tokens, device, steering_stride, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_len = inputs['input_ids'].shape[1]
    state = {'h': None, 'delta': None, 'steps': 0, 'regime': int(np.argmax(sds['pi']))}

    def hook_fn(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        state['h'] = h[:, -1, :].detach().float().cpu().numpy()[0]
        if state['delta'] is not None:
            delta = state['delta'].to(h.dtype).to(h.device)
            patched = h + delta
            state['delta'] = None
            return (patched,) + output[1:] if isinstance(output, tuple) else patched
        return output

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    generated_ids = inputs['input_ids'].clone()

    try:
        model.eval()
        with torch.no_grad():
            past_kv = None
            for _ in range(max_new_tokens):
                out = model(
                    input_ids=generated_ids[:, -1:] if past_kv is not None else generated_ids,
                    past_key_values=past_kv, use_cache=True,
                )
                past_kv = out.past_key_values
                state['steps'] += 1

                if state['steps'] % steering_stride == 0 and state['h'] is not None:
                    z_t = embed_hidden(state['h'], sds)
                    s_t = infer_regime(z_t, sds)
                    target_k = int(np.argmax(sds['A'][s_t]))
                    dz = steering_delta(z_t, s_t, target_k, sds, beta)
                    dx_scaled = dz @ sds['W_dec']
                    dx_raw = dx_scaled * sds['scaler'].scale_
                    dx_norm = dx_raw / (np.linalg.norm(dx_raw) + 1e-8)
                    h_norm = float(np.linalg.norm(state['h']))
                    effective_alpha = min(alpha, 0.1 * h_norm)
                    delta_t = torch.tensor(effective_alpha * dx_norm, dtype=torch.float32)
                    state['delta'] = delta_t.unsqueeze(0).unsqueeze(0)
                    state['regime'] = s_t

                logits = out.logits[:, -1, :].float()
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                probs = F.softmax(logits / temperature, dim=-1)
                probs = torch.clamp(probs, min=0.0)
                probs = probs / (probs.sum() + 1e-10)
                next_tok = torch.multinomial(probs, 1)
                generated_ids = torch.cat([generated_ids, next_tok], dim=1)
                if next_tok.item() == tokenizer.eos_token_id:
                    break
    finally:
        hook.remove()

    return tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True)


def generate_baseline(model, tokenizer, prompt, max_new_tokens, device, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=temperature,
                             pad_token_id=tokenizer.eos_token_id,
                             repetition_penalty=1.1)
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True)


def report_results(steered_counts, baseline_counts, n, ks, label="all", n_boot=2000):
    """Print pass@k table with truly-hard subset and bootstrap CIs."""
    hard = [i for i, c in enumerate(baseline_counts) if pass_at_k(n, c, 8) == 0.0]

    print(f"\n--- pass@k results ({label}, n={len(steered_counts)}) ---", flush=True)
    print(f"{'k':<6} {'steered':>9} {'baseline':>9} {'delta':>9}", flush=True)
    for k in ks:
        if k > n: continue
        pk_s = float(np.mean([pass_at_k(n, c, k) for c in steered_counts]))
        pk_b = float(np.mean([pass_at_k(n, c, k) for c in baseline_counts]))
        print(f"pass@{k:<2} {pk_s:>9.4f} {pk_b:>9.4f} {pk_s-pk_b:>+9.4f}", flush=True)

    if hard:
        sc_h = [steered_counts[i] for i in hard]
        bc_h = [baseline_counts[i] for i in hard]
        print(f"\n--- truly hard subset (baseline pass@8=0, n={len(hard)}) ---", flush=True)
        print(f"{'k':<6} {'steered':>9} {'baseline':>9} {'delta':>9} {'95% CI':>20}", flush=True)
        for k in ks:
            if k > n: continue
            pk_s = float(np.mean([pass_at_k(n, c, k) for c in sc_h]))
            pk_b = float(np.mean([pass_at_k(n, c, k) for c in bc_h]))
            _, lo, hi = bootstrap_ci(sc_h, bc_h, n, k, n_boot=n_boot)
            print(f"pass@{k:<2} {pk_s:>9.4f} {pk_b:>9.4f} {pk_s-pk_b:>+9.4f} "
                  f"  [{lo:+.4f}, {hi:+.4f}]", flush=True)
    else:
        print(f"\n  (no truly hard problems in this split)", flush=True)

    return hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reasoning-pkl', required=True)
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--model-key', required=True, choices=['qwen1_5b', 'qwen14b', 'llama8b'])
    ap.add_argument('--dataset', required=True, choices=['gsm8k', 'svamp', 'math500', 'mmlu_pro'])
    ap.add_argument('--k-samples', type=int, default=8)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--cebra-dim', type=int, default=40)
    ap.add_argument('--em-iters', type=int, default=50)
    ap.add_argument('--limit-problems', type=int, default=500)
    ap.add_argument('--max-triplets', type=int, default=25)
    ap.add_argument('--layer-idx', type=int, default=27,
                    help='27=Qwen1.5B  31=Llama8B  47=Qwen14B')
    ap.add_argument('--alpha', type=float, default=8.0)
    ap.add_argument('--beta', type=float, default=8.0)
    ap.add_argument('--steering-stride', type=int, default=20)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--ks', nargs='+', type=int, default=[1, 3, 5, 8])
    ap.add_argument('--n-problems', type=int, default=0,
                    help='Cap number of problems (0 = use all)')
    ap.add_argument('--n-boot', type=int, default=2000,
                    help='Bootstrap resamples for CI')
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--hf-token', default=None)
    ap.add_argument('--out', default='passatk_results.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)

    print("Fitting reasoning SDS + decoder...", flush=True)
    sds = fit_sds_and_decoder(
        args.reasoning_pkl, args.K, args.cebra_dim, args.em_iters,
        args.limit_problems, args.max_triplets,
    )
    print(f"  K={args.K}  W_dec={sds['W_dec'].shape}  centroids={sds['cebra_centroids'].shape}", flush=True)

    print(f"Loading base model: {args.base_model}", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    kw = {'trust_remote_code': True}
    if args.hf_token: kw['token'] = args.hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map='auto', attn_implementation='eager', **kw
    )
    model.eval()

    raw_dim = sds['W_dec'].shape[1]
    model_dim = model.config.hidden_size
    if raw_dim != model_dim:
        print(f"WARNING: reasoning hidden dim {raw_dim} != base model dim {model_dim}.", flush=True)

    problems = load_hard_problems(args.model_key, args.dataset)
    if args.n_problems > 0:
        problems = problems[:args.n_problems]
    print(f"Evaluating on {len(problems)} problems", flush=True)

    prompt_tmpl = "Solve the following problem step by step.\n\nProblem: {q}\n\nSolution:"

    print("Running baseline pass...", flush=True)
    baseline_counts = []
    for i, prob in enumerate(problems):
        prompt = prompt_tmpl.format(q=prob['q'])
        bc = sum(
            is_correct(generate_baseline(model, tokenizer, prompt,
                                         args.max_new_tokens, device, args.temperature),
                       prob['a'], args.dataset)
            for _ in range(args.k_samples)
        )
        baseline_counts.append(bc)
        if (i + 1) % 5 == 0:
            print(f"  baseline [{i+1}/{len(problems)}] "
                  f"pass@1={np.mean([c > 0 for c in baseline_counts]):.3f}", flush=True)

    print("Running steered pass...", flush=True)
    steered_counts = []
    for i, prob in enumerate(problems):
        prompt = prompt_tmpl.format(q=prob['q'])
        sc = sum(
            is_correct(generate_steered(
                model, tokenizer, prompt, sds,
                layer_idx=args.layer_idx, alpha=args.alpha, beta=args.beta,
                max_new_tokens=args.max_new_tokens, device=device,
                steering_stride=args.steering_stride, temperature=args.temperature,
            ), prob['a'], args.dataset)
            for _ in range(args.k_samples)
        )
        steered_counts.append(sc)
        if (i + 1) % 5 == 0:
            print(f"  steered  [{i+1}/{len(problems)}] "
                  f"pass@1={np.mean([c > 0 for c in steered_counts]):.3f}", flush=True)

    n = args.k_samples
    hard_indices = report_results(steered_counts, baseline_counts, n, args.ks,
                                   label="all problems", n_boot=args.n_boot)

    # build summary with both full and truly-hard metrics + CIs
    summary = {'steered': {}, 'baseline': {}, 'truly_hard': {}, 'config': vars(args)}
    for k in args.ks:
        if k > n: continue
        pk_s = float(np.mean([pass_at_k(n, c, k) for c in steered_counts]))
        pk_b = float(np.mean([pass_at_k(n, c, k) for c in baseline_counts]))
        summary['steered'][f'pass@{k}'] = pk_s
        summary['baseline'][f'pass@{k}'] = pk_b

    if hard_indices:
        sc_h = [steered_counts[i] for i in hard_indices]
        bc_h = [baseline_counts[i] for i in hard_indices]
        summary['truly_hard']['n'] = len(hard_indices)
        for k in args.ks:
            if k > n: continue
            pk_s = float(np.mean([pass_at_k(n, c, k) for c in sc_h]))
            mean_d, lo, hi = bootstrap_ci(sc_h, bc_h, n, k, n_boot=args.n_boot)
            summary['truly_hard'][f'pass@{k}'] = {
                'steered': pk_s, 'baseline': 0.0,
                'delta': pk_s, 'ci95_low': lo, 'ci95_high': hi
            }

    summary['steered_counts'] = steered_counts
    summary['baseline_counts'] = baseline_counts
    summary['hard_indices'] = hard_indices

    with open(args.out, 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSaved → {args.out}", flush=True)


if __name__ == '__main__':
    main()