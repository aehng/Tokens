"""Predictor & Codebook Policy with Category Caps and Diversity Constraints.

Implements Phase 2 policy:
- Budget K=32
- Prompt-conditioned only (sees prompt ONLY, strictly causal)
- Preserves:
    * prompt-conditioned content phrases
    * repeated prompt phrases
    * newlines / indentation
    * useful numeric structures
- Removes:
    * global filler list
    * bare punctuation with no measured incremental value
- Caps:
    * <= 8 structural / newline / numeric slots
    * >= 24 prompt-conditioned content / repeated multi-token phrases
"""

from __future__ import annotations

import heapq
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

from transformers import PreTrainedTokenizerBase


def is_structural(phrase_tokens: Tuple[int, ...], tokenizer: PreTrainedTokenizerBase) -> bool:
    """Classify whether a phrase is structural (newlines, indentation, tabs, whitespace)."""
    text = tokenizer.decode(list(phrase_tokens))
    # Newline / indentation / whitespace or empty
    if text.strip() == "" or all(c in " \t\r\n" for c in text):
        return True
    # Contains newlines/tabs with no alphanumeric tokens (e.g. '\n{\n', '\t#')
    if any(c in "\n\r\t" for c in text) and not any(c.isalnum() for c in text):
        return True
    return False


def is_numeric(phrase_tokens: Tuple[int, ...], tokenizer: PreTrainedTokenizerBase) -> bool:
    """Classify whether a phrase is numeric (pure numbers, currency, percents, decimals, ratios)."""
    text = tokenizer.decode(list(phrase_tokens)).strip()
    if not text:
        return False
    # Pure numbers or numbers with basic symbols (e.g. 100, $100, 120/80, 3.1415, 95%)
    if re.match(r"^[\$\€\£\¥]?\s*[-+]?\d+([\.,/:\-]\d+)*\s*\%?$", text):
        return True
    return False


def is_bare_punctuation(phrase_tokens: Tuple[int, ...], tokenizer: PreTrainedTokenizerBase) -> bool:
    """Detect uninformative bare punctuation chains."""
    text = tokenizer.decode(list(phrase_tokens)).strip()
    punct_chars = set(".,!?:;\"'()[]{}<>-=_+*&^%$#@~`|\\/")
    return len(text) > 0 and all(c in punct_chars for c in text)


def classify_phrase(phrase_tokens: Tuple[int, ...], tokenizer: PreTrainedTokenizerBase) -> str:
    """Classify phrase into one of: 'structural', 'numeric', 'bare_punct', 'content'."""
    if is_structural(phrase_tokens, tokenizer):
        return "structural"
    if is_bare_punctuation(phrase_tokens, tokenizer):
        return "bare_punct"
    if is_numeric(phrase_tokens, tokenizer):
        return "numeric"
    return "content"


def is_structural_or_numeric(phrase_tokens: Tuple[int, ...], tokenizer: PreTrainedTokenizerBase) -> bool:
    """Backward compatibility helper."""
    return is_structural(phrase_tokens, tokenizer) or is_numeric(phrase_tokens, tokenizer)


class CappedPredictorPolicy:
    """Selects K=32 prompt-conditioned phrases subject to category caps."""

    def __init__(
        self,
        predictor_index: Any,
        tokenizer: PreTrainedTokenizerBase,
        budget: int = 32,
        max_structural_slots: int = 0,
        allow_numeric: bool = True,
        max_numeric_slots: Optional[int] = None,
        filter_bare_punctuation: bool = True,
        initial_vocab_size: int = 32011,
    ) -> None:
        self.index = predictor_index
        self.tokenizer = tokenizer
        self.budget = budget
        self.max_structural_slots = max_structural_slots
        self.allow_numeric = allow_numeric
        self.max_numeric_slots = max_numeric_slots
        self.filter_bare_punctuation = filter_bare_punctuation
        self.initial_vocab_size = initial_vocab_size
        self.disabled_ids = set(getattr(predictor_index, "disabled_ids", []))
        self.max_subtokens = getattr(predictor_index, "max_subtokens", 4)

    def extract_prompt_ngrams(self, prompt_ids: Sequence[int]) -> Dict[Tuple[int, ...], int]:
        """Extract valid n-grams directly from the prompt."""
        ngrams: Counter[Tuple[int, ...]] = Counter()
        n = len(prompt_ids)
        for length in range(2, min(self.max_subtokens + 1, n + 1)):
            for i in range(n - length + 1):
                gram = tuple(prompt_ids[i : i + length])
                if any(tok in self.disabled_ids for tok in gram):
                    continue
                ngrams[gram] += 1
        return ngrams

    def select_codebook(
        self, prompt_ids: Sequence[int]
    ) -> Tuple[Dict[Tuple[int, ...], int], Dict[str, Any]]:
        """Select K phrases using prompt information only, enforcing category caps.
        
        Returns:
            codebook: dict mapping phrase_tuple -> hypertoken_id (starting at initial_vocab_size)
            meta: diagnostic info (latency, category counts, phrase strings)
        """
        t0 = time.perf_counter()
        scores: Dict[Tuple[int, ...], float] = defaultdict(float)

        # 1. Repetition prior from prompt
        p_ngrams = self.extract_prompt_ngrams(prompt_ids)
        for gram, cnt in p_ngrams.items():
            savings = len(gram) - 1
            scores[gram] += cnt * savings * 8.0

        # 2. Association prior from prompt tokens via offline index
        p_tokens = set(prompt_ids) - self.disabled_ids
        for p_tok in p_tokens:
            cands = getattr(self.index, "token_associations", {}).get(p_tok, [])
            for gram, weight in cands:
                scores[gram] += weight

        # Filter bare punctuation chains
        filtered_candidates: List[Tuple[Tuple[int, ...], float]] = []
        for gram, sc in scores.items():
            if self.filter_bare_punctuation and is_bare_punctuation(gram, self.tokenizer):
                continue
            filtered_candidates.append((gram, sc))

        # Rank all candidates deterministically: descending score, tie-break by token tuple
        ranked = sorted(filtered_candidates, key=lambda x: (-x[1], x[0]))

        # Apply category partition
        structural_cands: List[Tuple[Tuple[int, ...], float]] = []
        numeric_cands: List[Tuple[Tuple[int, ...], float]] = []
        content_cands: List[Tuple[Tuple[int, ...], float]] = []

        for gram, sc in ranked:
            cat = classify_phrase(gram, self.tokenizer)
            if cat == "structural":
                structural_cands.append((gram, sc))
            elif cat == "numeric":
                if self.allow_numeric:
                    numeric_cands.append((gram, sc))
            elif cat == "content":
                content_cands.append((gram, sc))

        # Cap structural slots
        accepted_structural = structural_cands[: self.max_structural_slots]

        # Cap numeric slots if configured
        if self.max_numeric_slots is not None:
            accepted_numeric = numeric_cands[: self.max_numeric_slots]
        else:
            accepted_numeric = numeric_cands

        # Primary pool: content + allowed numeric, sorted by score
        primary_pool = sorted(content_cands + accepted_numeric, key=lambda x: (-x[1], x[0]))

        # Allocate budget: primary pool gets budget - structural slots
        target_primary_count = max(0, self.budget - len(accepted_structural))
        selected: List[Tuple[int, ...]] = [gram for gram, _ in primary_pool[:target_primary_count]]

        # Add structural phrases
        for gram, _ in accepted_structural:
            if len(selected) < self.budget and gram not in selected:
                selected.append(gram)

        # If still room, fill with remainder of primary pool
        if len(selected) < self.budget:
            for gram, _ in primary_pool[target_primary_count:]:
                if gram not in selected:
                    selected.append(gram)
                    if len(selected) >= self.budget:
                        break

        # Map to hypertoken IDs (preserving deterministic ranked order)
        codebook = {
            gram: self.initial_vocab_size + i
            for i, gram in enumerate(selected)
        }

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        num_structural = sum(1 for g in codebook if is_structural(g, self.tokenizer))
        num_numeric = sum(1 for g in codebook if is_numeric(g, self.tokenizer))
        num_content = len(codebook) - num_structural - num_numeric

        meta = {
            "latency_ms": round(latency_ms, 2),
            "total_phrases": len(codebook),
            "content_phrases": num_content,
            "numeric_phrases": num_numeric,
            "structural_phrases": num_structural,
            "phrases": [
                {
                    "id": hid,
                    "text": self.tokenizer.decode(list(gram)),
                    "tokens": list(gram),
                    "category": classify_phrase(gram, self.tokenizer),
                }
                for gram, hid in codebook.items()
            ],
        }
        return codebook, meta
