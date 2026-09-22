"""Evidence-Aware Predictive Codebook Selector.

Reranks candidate hypertokens based on:
1. Expected value: P(emitted | prompt) * tokens_saved * safety_prior
2. Prompt provenance bonus (grounded numbers, identifiers, entities boosted)
3. Structural risk penalty (bare punctuation, dead structural patterns, isolated syntax penalized)
4. Diversity mechanism (suppressing redundant variants of identical patterns)
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from transformers import PreTrainedTokenizerBase
from zip2zip.predictor_policy import classify_phrase, is_bare_punctuation, is_numeric, is_structural

# Known dead structural phrases that consume capacity without empirical utilization
DEAD_STRUCTURAL_PHRASES = {
    ". The",
    ".  The",
    "\n    return",
    "\nreturn",
    "\n2.",
    "\n3.",
    "\n4.",
    "\n5.",
    ":\n    return",
    " \n   ",
}

# Isolated code syntax fragments that are toxic without exact prompt match
ISOLATED_SYNTAX_FRAGMENTS = {
    "):\n",
    "]:",
    "[]",
    "()",
    "len(",
    "print(",
    "def ",
    "import ",
    "assert ",
    " == ",
    " != ",
}


class EvidenceAwareSelector:
    """Predictive codebook selector with evidence-grounded scoring, structural risk penalties, and diversity constraints."""

    def __init__(
        self,
        predictor_index: Any,
        tokenizer: PreTrainedTokenizerBase,
        budget: int = 32,
        initial_vocab_size: int = 32011,
        provenance_bonus: float = 8.0,
        grounded_numeric_bonus: float = 6.0,
        ungrounded_numeric_penalty: float = 60.0,
        dead_structural_penalty: float = 60.0,
        isolated_syntax_penalty: float = 40.0,
        boundary_penalty: float = 5.0,
        min_score_threshold: Optional[float] = None,
    ) -> None:
        self.index = predictor_index
        self.tokenizer = tokenizer
        self.budget = budget
        self.initial_vocab_size = initial_vocab_size
        self.disabled_ids: Set[int] = set(getattr(predictor_index, "disabled_ids", []))
        self.max_subtokens: int = getattr(predictor_index, "max_subtokens", 4)
        
        self.provenance_bonus = provenance_bonus
        self.grounded_numeric_bonus = grounded_numeric_bonus
        self.ungrounded_numeric_penalty = ungrounded_numeric_penalty
        self.dead_structural_penalty = dead_structural_penalty
        self.isolated_syntax_penalty = isolated_syntax_penalty
        self.boundary_penalty = boundary_penalty
        self.min_score_threshold = min_score_threshold

        # Cache token piece string representations
        self._token_str_cache: Dict[int, str] = {}

    def _get_token_piece(self, tok_id: int) -> str:
        if tok_id not in self._token_str_cache:
            try:
                self._token_str_cache[tok_id] = self.tokenizer.convert_ids_to_tokens(tok_id)
            except Exception:
                self._token_str_cache[tok_id] = ""
        return self._token_str_cache[tok_id]

    def extract_prompt_ngrams(self, prompt_ids: Sequence[int]) -> Dict[Tuple[int, ...], int]:
        """Extract valid n-grams directly from prompt token sequence."""
        ngrams: Counter[Tuple[int, ...]] = Counter()
        n = len(prompt_ids)
        for length in range(2, min(self.max_subtokens + 1, n + 1)):
            for i in range(n - length + 1):
                gram = tuple(prompt_ids[i : i + length])
                if any(tok in self.disabled_ids for tok in gram):
                    continue
                ngrams[gram] += 1
        return dict(ngrams)

    def score_candidate(
        self,
        phrase: Tuple[int, ...],
        base_weight: float,
        prompt_ids_set: Set[int],
        prompt_ngrams: Dict[Tuple[int, ...], int],
        prompt_text: str,
        prompt_numbers: Set[str],
        prompt_words: Set[str],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute evidence-aware score for a single candidate phrase."""
        phrase_len = len(phrase)
        tokens_saved = phrase_len - 1
        meta: Dict[str, Any] = {"base_weight": base_weight, "tokens_saved": tokens_saved}

        phrase_str = self.tokenizer.decode(list(phrase))
        clean_phrase = phrase_str.strip()
        first_piece = self._get_token_piece(phrase[0])

        # 1. Base Expected Value Prior
        # Length prior: len 2 is empirically safer than len 3 (65.3% vs 92.5% failure)
        len_safety_factor = 1.0 if phrase_len == 2 else 0.65
        score = base_weight * tokens_saved * len_safety_factor

        # 2. Prompt Provenance
        # Exact token sequence match
        in_prompt_ngrams = phrase in prompt_ngrams
        in_prompt_text = (phrase_str in prompt_text) or (clean_phrase in prompt_text if clean_phrase else False)
        exact_grounded = in_prompt_ngrams or in_prompt_text
        meta["exact_grounded"] = exact_grounded

        if exact_grounded:
            occ_count = prompt_ngrams.get(phrase, prompt_text.count(clean_phrase) if clean_phrase else 1)
            score += self.provenance_bonus * math.log1p(occ_count)
            meta["provenance_bonus"] = self.provenance_bonus

        # Entity / Word Overlap
        phrase_words = set(re.findall(r"[a-zA-Z_]\w*", phrase_str.lower()))
        if phrase_words:
            overlap = len(phrase_words & prompt_words) / len(phrase_words)
            if overlap > 0:
                score += overlap * 3.0
                meta["word_overlap"] = overlap

        # 3. Numeric Grounding Verification
        phrase_nums = re.findall(r"\b\d+(?:\.\d+)?\b", phrase_str)
        has_digits = any(c.isdigit() for c in phrase_str)
        meta["has_digits"] = has_digits

        if has_digits:
            if phrase_nums:
                all_nums_in_prompt = all(n in prompt_numbers for n in phrase_nums)
            else:
                digits = [c for c in phrase_str if c.isdigit()]
                all_nums_in_prompt = all(d in prompt_text for d in digits)

            if all_nums_in_prompt or exact_grounded:
                # Prompt-present numeric phrase: safe and valuable
                score += self.grounded_numeric_bonus
                meta["numeric_status"] = "grounded"
            else:
                # Novel / inferred numeric phrase: empirically catastrophic (88.9% error rate)
                score -= self.ungrounded_numeric_penalty
                meta["numeric_status"] = "ungrounded_penalized"

        # 4. Structural & Dead Phrase Risk Penalty
        # Reject bare punctuation immediately
        if is_bare_punctuation(phrase, self.tokenizer):
            score -= 100.0
            meta["bare_punct"] = True

        # Check known dead structural phrases
        if any(dead in phrase_str for dead in DEAD_STRUCTURAL_PHRASES) and not exact_grounded:
            score -= self.dead_structural_penalty
            meta["dead_structural"] = True

        # Check isolated code syntax fragments
        if any(syn in phrase_str for syn in ISOLATED_SYNTAX_FRAGMENTS) and not exact_grounded:
            score -= self.isolated_syntax_penalty
            meta["isolated_syntax"] = True

        # 5. Boundary Alignment
        is_space_start = (phrase[0] == 29871 or first_piece.startswith("\u2581") or phrase_str.startswith((" ", "\t")))
        is_newline_start = phrase_str.startswith("\n") or first_piece.startswith("<0x0A>")
        is_punct_start = any(phrase_str.startswith(c) for c in ".,;:()[]{}<>\"'=")

        if is_space_start:
            score += 2.0  # Safe word boundary
            meta["boundary"] = "space"
        elif is_newline_start:
            if not exact_grounded:
                score -= self.boundary_penalty * 0.5
            meta["boundary"] = "newline"
        elif is_punct_start:
            if not exact_grounded:
                score -= self.boundary_penalty
            meta["boundary"] = "punct"
        else:
            # Mid-word or unspaced fragment (91.8% failure rate empirically)
            if not exact_grounded:
                score -= self.boundary_penalty * 1.5
            meta["boundary"] = "mid_word"

        meta["final_score"] = score
        return score, meta

    def select_codebook(
        self,
        prompt_ids: Sequence[int],
        prompt_text: Optional[str] = None,
        budget: Optional[int] = None,
        min_score_threshold: Optional[float] = None,
    ) -> Tuple[Dict[Tuple[int, ...], int], Dict[str, Any]]:
        """Select top evidence-grounded phrases subject to safety penalties and diversity constraints."""
        t0 = time.perf_counter()
        target_budget = budget if budget is not None else self.budget
        threshold = min_score_threshold if min_score_threshold is not None else self.min_score_threshold

        if prompt_text is None:
            prompt_text = self.tokenizer.decode(list(prompt_ids))

        prompt_ids_set = set(prompt_ids) - self.disabled_ids
        prompt_ngrams = self.extract_prompt_ngrams(prompt_ids)
        prompt_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", prompt_text))
        prompt_words = set(re.findall(r"[a-zA-Z_]\w*", prompt_text.lower()))

        candidate_raw_weights: Dict[Tuple[int, ...], float] = defaultdict(float)

        # 1. Repetition candidates directly from prompt
        for gram, cnt in prompt_ngrams.items():
            candidate_raw_weights[gram] += cnt * 6.0

        # 2. Token associations from index
        token_associations = getattr(self.index, "token_associations", {})
        for p_tok in prompt_ids_set:
            cands = token_associations.get(p_tok, [])
            for gram, weight in cands:
                candidate_raw_weights[gram] += weight

        # 3. Global static background (top candidates only)
        bg = getattr(self.index, "precomputed_global_static", [])
        for gram, bg_weight in bg[:target_budget]:
            candidate_raw_weights[gram] += bg_weight * 0.2

        # 4. Score all candidates with evidence and safety rules
        scored_candidates: List[Tuple[Tuple[int, ...], float, Dict[str, Any]]] = []
        for gram, base_w in candidate_raw_weights.items():
            if is_bare_punctuation(gram, self.tokenizer):
                continue
            sc, meta = self.score_candidate(
                gram,
                base_w,
                prompt_ids_set,
                prompt_ngrams,
                prompt_text,
                prompt_numbers,
                prompt_words,
            )
            # Apply threshold if specified
            if threshold is not None and sc < threshold:
                continue
            scored_candidates.append((gram, sc, meta))

        # Rank candidates deterministically
        ranked = sorted(scored_candidates, key=lambda x: (-x[1], x[0]))

        # 5. Diversity Mechanism: suppress redundant variants
        selected_phrases: List[Tuple[int, ...]] = []
        accepted_stems: Dict[str, int] = defaultdict(int)

        for gram, sc, meta in ranked:
            phrase_str = self.tokenizer.decode(list(gram)).strip()
            
            # Extract dominant semantic/alphanumeric stem
            stem_chars = "".join(c.lower() for c in phrase_str if c.isalnum())
            is_digit_variant = any(c.isdigit() for c in phrase_str) and len(phrase_str) <= 3

            # If heavily duplicated small numeric variant (e.g. 5th number in codebook)
            if is_digit_variant and accepted_stems["_digit_variant_"] >= 4 and not meta.get("exact_grounded", False):
                continue

            # Stem redundancy check
            if stem_chars and len(stem_chars) >= 3:
                if accepted_stems[stem_chars] >= 2 and not meta.get("exact_grounded", False):
                    continue

            selected_phrases.append(gram)
            if stem_chars:
                accepted_stems[stem_chars] += 1
            if is_digit_variant:
                accepted_stems["_digit_variant_"] += 1

            if len(selected_phrases) >= target_budget:
                break

        codebook = {
            gram: self.initial_vocab_size + idx
            for idx, gram in enumerate(selected_phrases)
        }

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        metadata = {
            "latency_ms": latency_ms,
            "target_budget": target_budget,
            "selected_count": len(selected_phrases),
            "total_considered": len(scored_candidates),
            "phrases": [self.tokenizer.decode(list(p)) for p in selected_phrases],
        }

        return codebook, metadata
