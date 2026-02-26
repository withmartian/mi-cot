"""
Data loading and alignment utilities for entropy-on-SDS pipeline.
Loads all_sentences_features.pkl and cot_data.pkl; provides sentence token ranges.
"""
import os
import re
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np


def get_sentence_token_ranges(full_text: str, sentences: List[str], tokenizer) -> List[Tuple[int, int]]:
    """Map each sentence to (start_tok, end_tok) in the tokenized full text. Same logic as create_dataset."""
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in full_ids]
    cumulative_chars = 0
    char_to_token = {}
    for i, tok in enumerate(tokens):
        for _ in range(len(tok)):
            char_to_token[cumulative_chars] = i
            cumulative_chars += 1
    ranges = []
    current_char = 0
    for sent in sentences:
        start_char = full_text.find(sent, current_char)
        if start_char == -1:
            start_char = current_char
        end_char = start_char + len(sent)
        start_tok = char_to_token.get(start_char, 0)
        end_tok = char_to_token.get(min(end_char - 1, max(char_to_token.keys())), len(full_ids) - 1) + 1
        ranges.append((start_tok, min(end_tok, len(full_ids))))
        current_char = end_char
    return ranges


def load_features_and_cot(
    features_path: str,
    cot_path: str,
    limit_problems: int = None,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Load all_sentences_features.pkl and cot_data.pkl.
    Returns (all_features list, ckpt_cot dict by problem_id).
    If limit_problems is set, filter features to problem_id < limit_problems.
    """
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features not found: {features_path}")
    if not os.path.exists(cot_path):
        raise FileNotFoundError(f"COT data not found: {cot_path}")
    all_features = pickle.load(open(features_path, "rb"))
    ckpt_cot = pickle.load(open(cot_path, "rb"))
    if limit_problems is not None:
        all_features = [f for f in all_features if f["problem_id"] < limit_problems]
    return all_features, ckpt_cot


def ordered_sentence_keys(all_features: List[Dict]) -> List[Tuple[int, int]]:
    """Return (problem_id, sentence_idx) in the exact order of all_features."""
    return [(f["problem_id"], f["sentence_idx"]) for f in all_features]


def get_full_prompt_and_ranges(pid: int, ckpt_cot: Dict, tokenizer) -> Tuple[str, List[str], List[Tuple[int, int]]]:
    """For a problem, return full_prompt, sentences, and token ranges per sentence."""
    data = ckpt_cot.get(pid)
    if not data:
        return None, None, None
    problem = data["problem"]
    cot = data["cot"]
    sentences = data["sentences"]
    full_prompt = problem + " " + cot
    ranges = get_sentence_token_ranges(full_prompt, sentences, tokenizer)
    return full_prompt, sentences, ranges
