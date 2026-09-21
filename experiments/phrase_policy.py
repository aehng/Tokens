"""Prompt-side phrase policies for exact-match verification.

A phrase is useful here only if the model was already going to say it. Emitting
it is then a compression of the real continuation, so punctuation is allowed
when it actually occurs and banned when it is only a global filler.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from experiments.heldout_predictor_benchmark import extract_ngrams

Phrase = Tuple[int, ...]

FUNCTION_WORDS = {
    "the", "of", "and", "to", "in", "a", "for", "is", "on", "with", "that",
    "by", "from", "as", "at", "be", "this", "it", "or", "an", "are", "was",
    "were", "if", "then", "so", "not", "we", "you", "he", "she", "they",
}


def phrase_text(tokenizer, phrase: Phrase) -> str:
    return tokenizer.decode(list(phrase))


def phrase_kind(tokenizer, phrase: Phrase) -> str:
    text = phrase_text(tokenizer, phrase)
    if any(ch in text for ch in "\n\r\t"):
        return "structural"
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    if letters == 0 and digits == 0:
        return "punct"
    if digits > letters:
        return "numeric"
    words = [w.strip(".,:;\"'`()[]{}").lower() for w in text.split()]
    words = [w for w in words if w]
    if words and all(w in FUNCTION_WORDS for w in words):
        return "function"
    if letters >= 3:
        return "content"
    return "other"


def _score_candidates(predictor, prompt_ids: Sequence[int], use_global: bool, budget: int):
    index = predictor.index
    scores: Dict[Phrase, float] = defaultdict(float)
    repeated = extract_ngrams(
        prompt_ids, index.disabled_ids, min_len=2, max_len=index.max_subtokens
    )
    repeated_set = set(repeated)
    for gram, count in repeated.items():
        scores[gram] += count * (len(gram) - 1) * 6.0
    for tok in set(prompt_ids) - index.disabled_ids:
        for gram, weight in index.token_associations.get(tok, []):
            scores[gram] += weight
    if use_global:
        for gram, bg_score in index.precomputed_global_static[: budget * 2]:
            scores[gram] += bg_score
    return scores, repeated_set


def select_phrases(
    predictor,
    tokenizer,
    prompt_ids: Sequence[int],
    policy: str,
    budget: int = 32,
    domain: str = "",
) -> List[Phrase]:
    """Return up to `budget` phrases, highest score first."""
    use_global = policy == "raw"
    scores, repeated = _score_candidates(predictor, prompt_ids, use_global, budget)

    def kind(phrase: Phrase) -> str:
        return phrase_kind(tokenizer, phrase)

    def keep(phrase: Phrase) -> bool:
        k = kind(phrase)
        in_prompt = phrase in repeated
        if policy == "raw":
            return True
        if policy == "no_global":
            return True
        if policy == "no_structural":
            return k != "structural"
        if policy == "no_structural_no_punct":
            return k not in {"structural", "punct"}
        if policy == "content":
            return k == "content"
        if policy == "prompt_only":
            return in_prompt
        if policy == "prompt_content":
            return in_prompt and k == "content"
        if policy == "topic":
            # Code repeats indentation. Math repeats numerals. Instructions
            # repeat the words in the request, not a global comma code.
            if domain == "code":
                return k in {"content", "structural", "function"} or in_prompt
            if domain == "reasoning":
                return k in {"content", "numeric", "function"} or (in_prompt and k != "punct")
            return k in {"content", "function"} or (in_prompt and k not in {"punct", "structural"})
        raise ValueError(f"unknown policy {policy}")

    ranked = [item for item in scores.items() if keep(item[0])]
    top = heapq.nlargest(budget, ranked, key=lambda item: item[1])
    return [phrase for phrase, _ in top]


# no_global won the coverage comparison. Those percentages are offline availability:
# how many tokens of an already-written answer could be tiled by the phrases.
# They are not realized decode savings. Real savings require the model to select
# the hypertoken during generation and skip the later forwards.
DEFAULT_POLICY = "no_global"

POLICIES = (
    "raw",
    "no_global",
    "no_structural",
    "no_structural_no_punct",
    "content",
    "prompt_only",
    "prompt_content",
    "topic",
)


def kind_counts(tokenizer, phrases: Iterable[Phrase]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for phrase in phrases:
        counts[phrase_kind(tokenizer, phrase)] += 1
    return dict(counts)
