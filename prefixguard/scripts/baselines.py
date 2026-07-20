#!/usr/bin/env python
"""Shared SDS fitting, prompt formatting, and answer-checking helpers."""

import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments import cebra_EM as cem

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    SYMPY_OK = True
    TRANSFORMS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
except Exception:
    SYMPY_OK = False
    sp = None
    parse_expr = None
    TRANSFORMS = None


# ---------------------------------------------------------------------
# Answer extraction / checking
# ---------------------------------------------------------------------

def extract_boxed_all(text: str) -> List[str]:
    if text is None:
        return []
    text = str(text)
    outs = []
    pattern = r"\\boxed\{"
    for m in re.finditer(pattern, text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            outs.append(text[start:i - 1].strip())
    return outs


def extract_boxed(text: str) -> Optional[str]:
    xs = extract_boxed_all(text)
    return xs[-1] if xs else None


def clean_candidate(s: Any) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()

    s = s.replace("$", "")
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    s = s.strip()

    if "=" in s:
        # Usually the rightmost expression is the final answer in answer lines.
        s = s.split("=")[-1].strip()

    s = re.sub(
        r"\b(units?|inches?|ways?|degrees?|dollars?|meters?|feet?|miles?)\b.*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = s.strip().strip(".").strip()
    return s if s else None


def get_tail(text: str, n_lines: int = 12) -> str:
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    return "\n".join(lines[-n_lines:])


def extract_answer_candidates(cot: str) -> List[str]:
    """
    Final-answer-aware candidates. We avoid arbitrary "last number" unless
    no final-answer cues exist.
    """
    cot = str(cot)
    candidates = []

    boxed = extract_boxed_all(cot)
    candidates.extend(reversed(boxed))

    tail_lines = [l.strip() for l in get_tail(cot, 14).splitlines() if l.strip()]
    trigger_words = [
        "final answer",
        "the answer",
        "answer is",
        "therefore",
        "thus",
        "hence",
        "so,",
        "so ",
        "resulting complex number",
        "greatest average speed",
        "simplified form",
        "distance between",
        "perimeter",
        "value of",
    ]

    for line in reversed(tail_lines):
        low = line.lower()

        for b in reversed(extract_boxed_all(line)):
            candidates.append(b)

        if any(t in low for t in trigger_words):
            if ":" in line and "answer" in low:
                candidates.append(line.split(":")[-1])
            ms = list(re.finditer(r"\bis\b", line, flags=re.IGNORECASE))
            if ms:
                candidates.append(line[ms[-1].end():])
            if "=" in line:
                candidates.append(line.split("=")[-1])
            candidates.append(line)

    # Fallback: last few numeric-looking things, low priority.
    nums = re.findall(r"-?\d+(?:\.\d+)?", cot.replace(",", ""))
    if nums:
        candidates.append(nums[-1])

    out, seen = [], set()
    for c in candidates:
        cc = clean_candidate(c)
        if not cc:
            continue
        key = cc.lower().replace(" ", "")
        if key not in seen:
            seen.add(key)
            out.append(cc)
    return out


def text_target(gt: Any) -> Optional[str]:
    gt = str(gt).strip()
    m = re.fullmatch(r"\\text\{([^{}]+)\}", gt)
    if m:
        return m.group(1).strip()
    if re.fullmatch(r"[A-Za-z ]+", gt):
        return gt.strip()
    return None


def normalize_words(s: Any) -> str:
    s = str(s).lower()
    s = re.sub(r"\\text\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"[^a-z]+", "", s)
    return s


def latex_frac_to_plain(s: str) -> str:
    pattern = r"\\(?:dfrac|frac)\{([^{}]+)\}\{([^{}]+)\}"
    while re.search(pattern, s):
        s = re.sub(pattern, r"((\1)/(\2))", s)
    return s


def simple_normalize(s: Any) -> Optional[str]:
    s = clean_candidate(s)
    if s is None:
        return None
    s = str(s).lower()
    s = s.replace("$", "")
    s = s.replace(" ", "")
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("√", "\\sqrt")
    s = re.sub(r"\\text\{([^{}]+)\}", r"\1", s)
    s = s.replace("\\sqrt{", "\\sqrt").replace("}", "")
    s = s.replace("\\,", "").replace("\\!", "")
    return s


def to_sympy_string(s: Any) -> Optional[str]:
    s = clean_candidate(s)
    if s is None:
        return None

    s = str(s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("−", "-")
    s = s.replace("√", "sqrt")
    s = s.replace("^", "**")
    s = re.sub(r"\\text\{([^{}]+)\}", r"\1", s)
    s = latex_frac_to_plain(s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*([A-Za-z0-9]+)", r"sqrt(\1)", s)
    s = re.sub(r"sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"sqrt\s*([0-9]+)", r"sqrt(\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("$", "")

    # Complex i: 7i -> 7*I, standalone i -> I.
    s = re.sub(r"(?<=\d)i\b", "*I", s)
    s = re.sub(r"\bi\b", "I", s)

    return s.strip()


def is_reasonable_math_candidate(s: Any) -> bool:
    if s is None:
        return False
    s = str(s).strip()
    if not s or len(s) > 140:
        return False
    if s.count(",") >= 2:
        return False
    bad_tokens = ["draw(", "label(", "dot(", "[asy]", "[/asy]", "step "]
    if any(tok in s.lower() for tok in bad_tokens):
        return False
    words = re.findall(r"[A-Za-z]+", s)
    math_tokens = ["\\sqrt", "\\frac", "sqrt", "sin", "cos", "tan", "cot", "sec", "i"]
    if len(words) > 10 and not any(tok in s for tok in math_tokens):
        return False
    return True


def math_equiv(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False

    aa = simple_normalize(a)
    bb = simple_normalize(b)
    if aa == bb:
        return True

    target = text_target(b)
    if target is not None:
        return normalize_words(target) in normalize_words(a)

    if not is_reasonable_math_candidate(a):
        return False

    if not SYMPY_OK:
        return False

    sa = to_sympy_string(a)
    sb = to_sympy_string(b)
    if not sa or not sb:
        return False

    if not is_reasonable_math_candidate(sa):
        return False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ea = parse_expr(sa, transformations=TRANSFORMS, evaluate=True)
            eb = parse_expr(sb, transformations=TRANSFORMS, evaluate=True)
        return bool(sp.simplify(ea - eb) == 0)
    except Exception:
        return False


def cot_matches_math_gt(cot: str, gt: Any) -> Tuple[bool, Optional[str]]:
    target = text_target(gt)
    if target is not None:
        tail = get_tail(cot, 8)
        if normalize_words(target) in normalize_words(tail):
            return True, target

    candidates = extract_answer_candidates(cot)
    for c in candidates:
        if math_equiv(c, gt):
            return True, c
    return False, candidates[0] if candidates else None


def extract_letter(text: Any) -> Optional[str]:
    if text is None:
        return None
    t = str(text).strip().upper()
    patterns = [
        r"(?:FINAL ANSWER|ANSWER|CORRECT OPTION|OPTION|THE ANSWER IS)\s*[:\-]?\s*\(?([A-J])\)?",
        r"\\boxed\{([A-J])\}",
        r"\(([A-J])\)",
        r"\b([A-J])\b",
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1)
    return None


def extract_number_for_numeric(text: Any) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    boxed = extract_boxed(text)
    if boxed is not None:
        frac = re.search(r"\\(?:dfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", boxed)
        if frac:
            return f"{frac.group(1)}/{frac.group(2)}"
        nums = re.findall(r"-?\d+(?:\.\d+)?", boxed.replace(",", ""))
        if nums:
            return nums[-1]

    candidates = extract_answer_candidates(text)
    for c in candidates:
        frac = re.search(r"\\(?:dfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", c)
        if frac:
            return f"{frac.group(1)}/{frac.group(2)}"
        nums = re.findall(r"-?\d+(?:\.\d+)?", str(c).replace(",", ""))
        if nums:
            return nums[-1]

    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def to_float_maybe(x: Any) -> Optional[float]:
    if x is None:
        return None
    x = str(x).strip().replace(",", "")
    x = latex_frac_to_plain(x)
    x = x.replace("$", "")
    try:
        if SYMPY_OK:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return float(parse_expr(to_sympy_string(x), transformations=TRANSFORMS, evaluate=True))
        return float(x)
    except Exception:
        return None


def is_correct_generation(generation: str, gold: Any, dataset: str) -> bool:
    dataset = dataset.lower()
    gold_s = str(gold).strip()

    if dataset in {"mmlu_pro", "mmlu"}:
        return extract_letter(generation) == gold_s.upper()

    if dataset in {"math500", "math"}:
        ok, _ = cot_matches_math_gt(generation, gold_s)
        return ok

    # Numeric datasets such as GSM8K/SVAMP.
    pred = extract_number_for_numeric(generation)
    if pred is None:
        return False

    # Gold may contain a GSM8K "####" string.
    m = re.search(r"####\s*([-\d,.]+)", gold_s)
    if m:
        gold_s = m.group(1).replace(",", "")

    pn = to_float_maybe(pred)
    gn = to_float_maybe(gold_s)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-4

    return simple_normalize(pred) == simple_normalize(gold_s)


# ---------------------------------------------------------------------
# Data loading / prompt formatting
# ---------------------------------------------------------------------

def format_prompt(tokenizer, problem: str, style: str = "chat") -> str:
    if style == "plain":
        return f"Solve the following problem step by step.\n\nProblem: {problem}\n\nSolution:"

    messages = [{"role": "user", "content": problem}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"Solve the following problem step by step.\n\nProblem: {problem}\n\nSolution:"


# ---------------------------------------------------------------------
# SDS fitting and CAA direction
# ---------------------------------------------------------------------

def fit_sds_and_decoder(pkl_path, K, cebra_dim, em_iters, limit_problems, max_triplets, seed):
    all_features, triplets = cem.load_and_prepare_cebra(
        pkl_path, mode="temporal", limit_problems=limit_problems,
        max_triplets=max_triplets, seed=seed,
    )
    cebra_seqs, _, _, _, _ = cem.train_cebra_projection(
        all_features, triplets, d_out=cebra_dim, seed=seed,
    )

    p_map = defaultdict(list)
    for i, f in enumerate(all_features):
        p_map[int(f["problem_id"])].append(i)
    pids_sorted = sorted(p for p in p_map if len(p_map[p]) >= 3)
    idx_seqs = [p_map[pid] for pid in pids_sorted]

    pi, A, dM, db, dCov = cem.init_params(cebra_seqs, K, cebra_dim, seed=seed)
    for it in range(em_iters):
        gammas, xis, lls = [], [], []
        for seq in cebra_seqs:
            g, x, ll = cem.forward_backward(seq, pi, A, dM, db, dCov, K)
            gammas.append(g)
            xis.append(x)
            lls.append(ll)
        pi, A, dM, db, dCov = cem.m_step(cebra_seqs, gammas, xis, K, cebra_dim)
        if (it + 1) % 10 == 0:
            print(f"  EM iter {it+1}/{em_iters}", flush=True)

    X_raw = np.array([f["hidden_state_last"] for f in all_features], dtype=np.float32)
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
        vecs = [
            cebra_seqs[i][state_seqs[i] == k]
            for i in range(len(cebra_seqs))
            if np.any(state_seqs[i] == k)
        ]
        centroids.append(np.vstack(vecs).mean(0) if vecs else np.zeros(cebra_dim))

    print("  Learned A:", flush=True)
    for k in range(K):
        second = np.argsort(A[k])[::-1][1]
        print(f"    regime {k}: p_stay={A[k,k]:.3f}, top transition -> {second} ({A[k,second]:.3f})", flush=True)

    pca = PCA(n_components=min(10, X_raw.shape[0], X_raw.shape[1]), random_state=seed)
    pca.fit(X_raw)

    return dict(
        pi=pi, A=A, dM=dM, db=db, dCov=dCov,
        W_dec=W_dec, scaler=scaler, K=K, cebra_dim=cebra_dim,
        cebra_centroids=np.array(centroids),
        W_dec_pinv=np.linalg.pinv(W_dec),
        pca_directions=pca.components_,
        mean_activation=X_raw.mean(0),
        pca_explained_var=pca.explained_variance_ratio_,
        raw_dim=X_raw.shape[1],
    )


def embed_hidden(h_np: np.ndarray, sds: Dict[str, Any]) -> np.ndarray:
    h_scaled = sds["scaler"].transform(h_np.reshape(1, -1))[0]
    return h_scaled @ sds["W_dec_pinv"]


def infer_regime(z: np.ndarray, sds: Dict[str, Any]) -> int:
    dists = [np.linalg.norm(z - sds["cebra_centroids"][k]) for k in range(sds["K"])]
    return int(np.argmin(dists))
