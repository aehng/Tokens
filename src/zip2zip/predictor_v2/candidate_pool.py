"""Fixed High-Recall Prompt-Only Candidate Pool & Feature Extractor for Predictor V2.

All candidate architectures MUST evaluate on the exact same candidate pool per prompt.
Candidate generation is prompt-only (strictly causal before response generation):
1. Prompt n-grams (lengths 2..4)
2. Token associations from index
3. Domain-general background bank

Each candidate record stores:
- Deterministic 21 handcrafted features
- Ground-truth objective labels derived from canonical Vanilla Phi continuation
"""

from __future__ import annotations

import math
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from transformers import AutoTokenizer

from src.zip2zip.predictor_v2.vanilla_labels import VanillaContinuationRecord

CACHED_PREDICTOR_PATH = "experiments/checkpoints/cached_predictor.pkl"

GRAMMATICAL_GLUE = {
    " of the", " to the", " in the", " for the", " on the", " with the", " by the",
    " at the", " from the", " into the", " as well", " as well as", " is a", " was a",
    " to be", " can be", " will be", " would be", " should be", " has been", " have been",
    " that the", " that is", " there is", " there are", " it is", " in order", " in order to",
    " such as", " based on", " due to", " according to", " in addition", " for example",
    " as a", " with a", " of a", " in a", " to a",
}

CODE_SYNTAX_FRAGMENTS = {
    "):\n", "): \n", "):", "):    ", "]:", "):\n    ", "\n    return ", "    return",
    "():", "[]", "{}", "()", "len(", "range(", "int(", "str(", "float(", "list(", "dict(",
}

FEATURE_NAMES = [
    "log_raw_weight",
    "exact_in_prompt",
    "prompt_count",
    "word_overlap_ratio",
    "phrase_len_2",
    "phrase_len_3",
    "phrase_len_4",
    "char_len",
    "leading_space",
    "trailing_space",
    "mid_word_start",
    "is_numeric",
    "numeric_grounded",
    "numeric_ungrounded",
    "is_function_name",
    "function_grounded",
    "syntax_hazard",
    "is_grammatical_glue",
    "domain_code",
    "domain_reasoning",
    "domain_instruction",
]


def is_bare_punctuation(tokens: Tuple[int, ...], tokenizer: AutoTokenizer) -> bool:
    text = tokenizer.decode(list(tokens)).strip()
    punct = set(".,!?:;\"'()[]{}<>-=_+*&^%$#@~`|\\/")
    return len(text) > 0 and all(c in punct for c in text)


def extract_handcrafted_features(
    phrase_text: str,
    phrase_tokens: Tuple[int, ...],
    raw_weight: float,
    prompt_text: str,
    prompt_tokens: Sequence[int],
    domain: str,
) -> np.ndarray:
    """Extracts the 21 handcrafted features for candidate phrase given prompt."""
    p_len = len(phrase_tokens)
    exact_in_p = 1.0 if (phrase_text in prompt_text or phrase_text.strip() in prompt_text) else 0.0

    prompt_words = set(re.findall(r"\b\w+\b", prompt_text.lower()))
    phrase_words = re.findall(r"\b\w+\b", phrase_text.lower())
    overlap = (sum(1 for w in phrase_words if w in prompt_words) / len(phrase_words)) if phrase_words else 0.0

    leading_space = 1.0 if (phrase_text.startswith(" ") or phrase_text.startswith("\n")) else 0.0
    trailing_space = 1.0 if (phrase_text.endswith(" ") or phrase_text.endswith("\t")) else 0.0
    mid_word = 1.0 if (len(phrase_text) > 0 and phrase_text[0].isalnum() and not phrase_text.startswith(" ")) else 0.0

    digits = re.findall(r"\d+", phrase_text)
    is_num = 1.0 if digits else 0.0
    num_grounded = 0.0
    num_ungrounded = 0.0
    if digits:
        p_digits = set(re.findall(r"\d+", prompt_text))
        if all(d in p_digits for d in digits) or exact_in_p > 0:
            num_grounded = 1.0
        else:
            num_ungrounded = 1.0

    fn_def = re.search(r"def\s+([a-zA-Z_]\w*)", phrase_text)
    fn_call = re.search(r"([a-zA-Z_]\w*)\s*\(", phrase_text)
    fn_name = None
    if fn_def:
        fn_name = fn_def.group(1)
    elif fn_call and len(fn_call.group(1)) > 1:
        fn_name = fn_call.group(1)

    is_fn = 1.0 if fn_name else 0.0
    fn_grounded = 1.0 if (fn_name and fn_name.lower() in prompt_text.lower()) else 0.0

    unbalanced = False
    for ob, cb in [("(", ")"), ("[", "]"), ("{", "}")]:
        if phrase_text.count(ob) != phrase_text.count(cb):
            unbalanced = True
            break
    syntax_hazard = 1.0 if (unbalanced or phrase_text.strip() in CODE_SYNTAX_FRAGMENTS) else 0.0

    clean_p = phrase_text.strip()
    is_glue = 1.0 if (phrase_text in GRAMMATICAL_GLUE or (" " + clean_p) in GRAMMATICAL_GLUE) else 0.0

    vec = [
        float(np.log1p(max(0.0, raw_weight))),
        exact_in_p,
        float(prompt_text.count(phrase_text)),
        float(overlap),
        1.0 if p_len == 2 else 0.0,
        1.0 if p_len == 3 else 0.0,
        1.0 if p_len == 4 else 0.0,
        float(len(phrase_text)),
        leading_space,
        trailing_space,
        mid_word,
        is_num,
        num_grounded,
        num_ungrounded,
        is_fn,
        fn_grounded,
        syntax_hazard,
        is_glue,
        1.0 if domain == "code" else 0.0,
        1.0 if domain == "reasoning" else 0.0,
        1.0 if domain == "instruction" else 0.0,
    ]
    return np.array(vec, dtype=np.float32)


@dataclass
class CandidateRecord:
    prompt_id: str
    tokens: Tuple[int, ...]
    text: str
    length: int
    sources: List[str]
    raw_association_weight: float
    features: np.ndarray
    occurs_in_vanilla: bool
    occurrence_count: int
    first_occurrence_index: int
    first_occurrence_bucket: int  # 0: [0, 31], 1: [32, 127], 2: [128, 255], 3: [256+], 4: Never
    isolated_steps_saved: int
    candidate_pool_oracle_k8: bool = False
    candidate_pool_oracle_k16: bool = False
    candidate_pool_oracle_k32: bool = False
    marginal_dp_saved: int = 0


def get_first_occurrence_bucket(first_idx: int) -> int:
    if first_idx < 0:
        return 4  # Never
    elif first_idx < 32:
        return 0  # 0..31
    elif first_idx < 128:
        return 1  # 32..127
    elif first_idx < 256:
        return 2  # 128..255
    else:
        return 3  # 256+


class PromptCandidateGenerator:
    """Generates the shared, fixed prompt-only candidate pool for any prompt."""

    def __init__(
        self,
        predictor_index: Any,
        tokenizer: AutoTokenizer,
        max_subtokens: int = 4,
        top_associations_per_token: int = 24,
        num_background: int = 96,
        target_pool_size: Tuple[int, int] = (256, 512),
    ):
        self.tokenizer = tokenizer
        self.max_subtokens = max_subtokens
        self.top_assoc = top_associations_per_token
        self.num_background = num_background
        self.target_pool_min, self.target_pool_max = target_pool_size

        idx = getattr(predictor_index, "index", predictor_index)
        self.token_associations = getattr(idx, "token_associations", {})
        self.precomputed_global_static = getattr(idx, "precomputed_global_static", [])
        self.disabled_ids = set(getattr(idx, "disabled_ids", [0, 1, 2] + list(range(32000, 32011))))

    def generate_candidate_pool(
        self,
        prompt_ids: Sequence[int],
        prompt_text: str,
        domain: str = "general",
    ) -> Dict[Tuple[int, ...], Dict[str, Any]]:
        """Gathers unique prompt-only candidate phrases and their initial weights and sources."""
        cands: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        p_ids = [t for t in prompt_ids if t not in self.disabled_ids]
        n = len(p_ids)

        # 1. Prompt n-grams (len 2..4)
        for l in range(2, min(self.max_subtokens + 1, n + 1)):
            for i in range(n - l + 1):
                gram = tuple(p_ids[i : i + l])
                if gram not in cands:
                    cands[gram] = {"sources": set(), "weight": 0.0}
                cands[gram]["sources"].add("prompt_ngram")
                cands[gram]["weight"] += 8.0

        # 2. Token associations
        for tok in set(p_ids):
            assoc_list = self.token_associations.get(tok, [])
            for gram, w in assoc_list[: self.top_assoc]:
                if any(t in self.disabled_ids for t in gram):
                    continue
                if len(gram) < 2 or len(gram) > self.max_subtokens:
                    continue
                if gram not in cands:
                    cands[gram] = {"sources": set(), "weight": 0.0}
                cands[gram]["sources"].add("token_association")
                cands[gram]["weight"] += float(w)

        # 3. Domain-general background bank
        for gram, bg_w in self.precomputed_global_static[: self.num_background]:
            if any(t in self.disabled_ids for t in gram):
                continue
            if len(gram) < 2 or len(gram) > self.max_subtokens:
                continue
            if gram not in cands:
                cands[gram] = {"sources": set(), "weight": 0.0}
            cands[gram]["sources"].add("background_bank")
            cands[gram]["weight"] += float(bg_w) * 0.2

        # Filter bare punctuation and trailing whitespace
        filtered: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for gram, data in cands.items():
            if is_bare_punctuation(gram, self.tokenizer):
                continue
            text = self.tokenizer.decode(list(gram))
            if text.endswith(" ") or text.endswith("\t"):
                continue
            filtered[gram] = data

        return filtered

    def build_candidate_records(
        self,
        record: VanillaContinuationRecord,
    ) -> List[CandidateRecord]:
        """Builds CandidateRecord objects with features and objective labels from continuation."""
        cand_dict = self.generate_candidate_pool(
            record.prompt_token_ids,
            record.prompt_text,
            record.domain,
        )

        continuation_tokens = record.continuation_token_ids
        n_cont = len(continuation_tokens)

        # Build occurrence index for fast lookup
        # Count and first occurrence position in continuation
        cont_counts: Counter[Tuple[int, ...]] = Counter()
        first_idx_map: Dict[Tuple[int, ...], int] = {}

        for l in range(2, min(self.max_subtokens + 1, n_cont + 1)):
            for i in range(n_cont - l + 1):
                g = tuple(continuation_tokens[i : i + l])
                cont_counts[g] += 1
                if g not in first_idx_map:
                    first_idx_map[g] = i

        cand_records: List[CandidateRecord] = []
        for gram, meta in cand_dict.items():
            text = self.tokenizer.decode(list(gram))
            p_len = len(gram)
            raw_w = meta["weight"]
            sources = sorted(list(meta["sources"]))

            count = cont_counts.get(gram, 0)
            occurs = count > 0
            first_idx = first_idx_map.get(gram, -1)
            bucket = get_first_occurrence_bucket(first_idx)
            isolated_saved = (p_len - 1) * count

            feats = extract_handcrafted_features(
                phrase_text=text,
                phrase_tokens=gram,
                raw_weight=raw_w,
                prompt_text=record.prompt_text,
                prompt_tokens=record.prompt_token_ids,
                domain=record.domain,
            )

            cand_records.append(
                CandidateRecord(
                    prompt_id=record.prompt_id,
                    tokens=gram,
                    text=text,
                    length=p_len,
                    sources=sources,
                    raw_association_weight=raw_w,
                    features=feats,
                    occurs_in_vanilla=occurs,
                    occurrence_count=count,
                    first_occurrence_index=first_idx,
                    first_occurrence_bucket=bucket,
                    isolated_steps_saved=isolated_saved,
                )
            )

        return cand_records
