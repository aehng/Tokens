"""
Builds phrase banks and optimized predictor indices strictly from training data.
"""

from collections import Counter, defaultdict
import json
import math
import time
from typing import Dict, List, Set, Tuple

from experiments.heldout_predictor_benchmark import extract_ngrams, StrictHeldOutPhraseBank
from experiments.optimized_predictor import OptimizedPhraseIndex, FastPredictor


def build_phrase_bank_from_records(
    train_records: List[Dict],
    max_subtokens: int = 3,
    disabled_ids: Set[int] = None,
    candidate_pool_size: int = 8000,
    max_prompt_salient_tokens: int = 64,
) -> StrictHeldOutPhraseBank:
    """
    Mine strict co-occurrences and frequency statistics strictly from train records
    using two-pass candidate filtering to guarantee sub-minute runtime and bounded memory
    even on documents with thousands of tokens.
    """
    disabled = disabled_ids or {0, 1, 32000, 32001}
    bank = StrictHeldOutPhraseBank(max_subtokens=max_subtokens, disabled_ids=disabled)

    print(f"Mining phrase statistics from {len(train_records)} train records...")
    t0 = time.time()

    # Pass 1: Global and per-domain n-gram frequencies
    for rec in train_records:
        r_ngrams = extract_ngrams(rec["response_token_ids"], disabled, min_len=2, max_len=max_subtokens)
        dom = rec.get("domain", "general")
        for gram, cnt in r_ngrams.items():
            bank.global_counts[gram] += cnt
            bank.domain_counts[dom][gram] += cnt

    t_pass1 = time.time()
    print(f"Pass 1 (Global & Domain Counts) finished in {t_pass1 - t0:.2f}s. Unique phrases: {len(bank.global_counts):,}")

    # Identify candidate pool: top frequent global and domain phrases
    top_candidates = set(gram for gram, _ in bank.global_counts.most_common(candidate_pool_size))
    for dom, d_counts in bank.domain_counts.items():
        top_candidates.update(gram for gram, _ in d_counts.most_common(candidate_pool_size // 2))
    print(f"Candidate pool selected: {len(top_candidates):,} phrases.")

    # Pass 2: Association mapping prompt tokens -> candidate phrases
    for rec in train_records:
        p_ids = rec["prompt_token_ids"]
        p_tokens = set(p_ids) - disabled
        if len(p_tokens) > max_prompt_salient_tokens:
            # Keep first and last segments (most informative in instructions/conversations)
            half = max_prompt_salient_tokens // 2
            p_tokens = (set(p_ids[:half]) | set(p_ids[-half:])) - disabled

        r_ngrams = extract_ngrams(rec["response_token_ids"], disabled, min_len=2, max_len=max_subtokens)
        r_cands = {g: cnt for g, cnt in r_ngrams.items() if g in top_candidates}

        for p_tok in p_tokens:
            bank.prompt_token_totals[p_tok] += 1
            d = bank.prompt_to_phrase[p_tok]
            for gram, cnt in r_cands.items():
                d[gram] += cnt

    t1 = time.time()
    print(f"Pass 2 (Association) finished in {t1 - t_pass1:.2f}s. Total mining time: {t1 - t0:.2f}s.")
    print(f"Domains indexed: {list(bank.domain_counts.keys())}")
    return bank


def build_fast_predictor_from_records(
    train_records: List[Dict],
    max_subtokens: int = 3,
    top_candidates_per_token: int = 64,
    min_cooccurrence: int = 2,
    initial_vocab_size: int = 32011,
) -> FastPredictor:
    """Build complete FastPredictor from train records."""
    bank = build_phrase_bank_from_records(train_records, max_subtokens=max_subtokens)
    index = OptimizedPhraseIndex.build_from_phrase_bank(
        bank,
        top_candidates_per_token=top_candidates_per_token,
        min_cooccurrence=min_cooccurrence,
    )
    return FastPredictor(index, initial_vocab_size=initial_vocab_size)
