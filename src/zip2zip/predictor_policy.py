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


def is_structural_or_numeric(phrase_tokens: Tuple[int, ...], tokenizer: PreTrainedTokenizerBase) -> bool:
    """Classify whether a phrase is structural (newlines/whitespace) or numeric."""
    text = tokenizer.decode(list(phrase_tokens))
    # Newline / indentation / whitespace
    if text.strip() == "" or all(c in " \t\r\n" for c in text):
        return True
    # Pure numbers or numbers with basic symbols (e.g. $100, 120/80)
    if re.match(r"^[\$\€\£]?\s*\d+([\.,/]\d+)*\s*\%?$", text.strip()):
        return True
    return False


def is_bare_punctuation(phrase_tokens: Tuple[int, ...], tokenizer: PreTrainedTokenizerBase) -> bool:
    """Detect uninformative bare punctuation chains."""
    text = tokenizer.decode(list(phrase_tokens))
    punct_chars = set(".,!?:;\"'()[]{}<>-=_+*&^%$#@~`|\\/")
    # If phrase consists solely of punctuation without any alphanumeric or newline characters
    return len(text.strip()) > 0 and all(c in punct_chars for c in text.strip())


class CappedPredictorPolicy:
    """Selects K=32 prompt-conditioned phrases subject to category caps."""

    def __init__(
        self,
        predictor_index: Any,
        tokenizer: PreTrainedTokenizerBase,
        budget: int = 32,
        max_structural_slots: int = 8,
        initial_vocab_size: int = 32011,
    ) -> None:
        self.index = predictor_index
        self.tokenizer = tokenizer
        self.budget = budget
        self.max_structural_slots = max_structural_slots
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
            # Give higher weight to repeated phrases in the prompt
            scores[gram] += cnt * savings * 8.0

        # 2. Association prior from prompt tokens via offline index
        p_tokens = set(prompt_ids) - self.disabled_ids
        for p_tok in p_tokens:
            cands = getattr(self.index, "token_associations", {}).get(p_tok, [])
            for gram, weight in cands:
                scores[gram] += weight

        # Filter out bare punctuation chains
        filtered_candidates: List[Tuple[Tuple[int, ...], float]] = []
        for gram, sc in scores.items():
            if is_bare_punctuation(gram, self.tokenizer):
                continue
            filtered_candidates.append((gram, sc))

        # Rank all candidates by score
        ranked = sorted(filtered_candidates, key=lambda x: x[1], reverse=True)

        # Apply category caps
        structural_slots: List[Tuple[Tuple[int, ...], float]] = []
        content_slots: List[Tuple[Tuple[int, ...], float]] = []

        for gram, sc in ranked:
            if is_structural_or_numeric(gram, self.tokenizer):
                if len(structural_slots) < self.max_structural_slots:
                    structural_slots.append((gram, sc))
            else:
                content_slots.append((gram, sc))

        # Combine: prioritize content, cap structural
        selected: List[Tuple[int, ...]] = []
        # Add top content phrases first up to budget - len(structural_slots)
        target_content_count = max(0, self.budget - len(structural_slots))
        for gram, _ in content_slots[:target_content_count]:
            selected.append(gram)

        # Add structural phrases
        for gram, _ in structural_slots:
            if len(selected) < self.budget:
                selected.append(gram)

        # If still room, fill with more content phrases
        if len(selected) < self.budget:
            for gram, _ in content_slots[target_content_count:]:
                if gram not in selected:
                    selected.append(gram)
                    if len(selected) >= self.budget:
                        break

        # Map to hypertoken IDs
        codebook = {
            gram: self.initial_vocab_size + i
            for i, gram in enumerate(selected)
        }

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        num_structural = sum(1 for g in codebook if is_structural_or_numeric(g, self.tokenizer))
        num_content = len(codebook) - num_structural

        meta = {
            "latency_ms": round(latency_ms, 2),
            "total_phrases": len(codebook),
            "content_phrases": num_content,
            "structural_phrases": num_structural,
            "phrases": [
                {
                    "id": hid,
                    "text": self.tokenizer.decode(list(gram)),
                    "tokens": list(gram),
                    "is_structural": is_structural_or_numeric(gram, self.tokenizer),
                }
                for gram, hid in codebook.items()
            ],
        }
        return codebook, meta
