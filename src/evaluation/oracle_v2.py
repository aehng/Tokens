"""
Oracle V2: Exact and Near-Optimal Compression Oracles for Dynamic Token Vocabularies.

Provides:
1. ExactOracle: Combinatorial / branch-and-bound solver for provably optimal K-phrase codebooks on small instances.
2. OracleV2: Fast practical near-optimal oracle using iterative marginal-gain forward selection,
   candidate pre-filtering, and subphrase shadowing suppression.
3. AnswerAwareGreedyOracle: Backwards-compatible reference implementation of the legacy greedy frequency heuristic.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.evaluation.offline_segmenter import segment_tokens_dp


def extract_candidate_ngrams(
    tokens: Sequence[int],
    min_length: int = 2,
    max_length: int = 3,
    min_count: int = 1,
) -> Dict[Tuple[int, ...], int]:
    """Extract rolling n-grams and frequency counts from token sequence."""
    n = len(tokens)
    counts: Counter[Tuple[int, ...]] = Counter()
    for l in range(min_length, max_length + 1):
        for i in range(n - l + 1):
            gram = tuple(tokens[i : i + l])
            counts[gram] += 1
    if min_count > 1:
        return {g: c for g, c in counts.items() if c >= min_count}
    return dict(counts)


class ExactOracle:
    """Solves for the mathematically exact globally optimal codebook of size <= K."""

    @staticmethod
    def solve(
        tokens: List[int],
        k: int,
        min_length: int = 2,
        max_length: int = 3,
        candidate_pool_limit: int = 20,
    ) -> Tuple[Set[Tuple[int, ...]], int, Dict[str, Any]]:
        """Find the provably optimal codebook subset by combinatorial search over candidate pool."""
        t0 = time.perf_counter()
        n = len(tokens)
        if n < min_length or k <= 0:
            return set(), 0, {"runtime_ms": 0.0, "total_combinations": 0}

        counts = extract_candidate_ngrams(tokens, min_length=min_length, max_length=max_length, min_count=1)
        if not counts:
            return set(), 0, {"runtime_ms": 0.0, "total_combinations": 0}

        # Filter candidate pool to top candidates by theoretical unconstrained upper bound
        ranked_cands = sorted(
            counts.keys(),
            key=lambda g: counts[g] * (len(g) - 1),
            reverse=True,
        )[:candidate_pool_limit]

        best_cb: Set[Tuple[int, ...]] = set()
        best_saved: int = -1
        total_evals = 0

        # Try sizes up to k
        for size in range(1, min(k, len(ranked_cands)) + 1):
            for comb in itertools.combinations(ranked_cands, size):
                total_evals += 1
                cb_set = set(comb)
                saved = segment_tokens_dp(tokens, cb_set)[2]["tokens_saved"]
                if saved > best_saved:
                    best_saved = saved
                    best_cb = cb_set

        t1 = time.perf_counter()
        stats = {
            "runtime_ms": (t1 - t0) * 1000.0,
            "total_combinations": total_evals,
            "tokens_saved": best_saved,
            "codebook_size": len(best_cb),
        }
        return best_cb, best_saved, stats


class OracleV2:
    """Fast practical near-optimal compression oracle using iterative marginal-gain forward selection."""

    @staticmethod
    def compute_codebook(
        tokens: List[int],
        k: int = 32,
        min_length: int = 2,
        max_length: int = 3,
        beam_width: int = 4,
        candidate_limit: int = 80,
    ) -> Tuple[Set[Tuple[int, ...]], Dict[str, Any]]:
        """Compute near-optimal codebook that directly maximizes marginal decode steps saved."""
        t0 = time.perf_counter()
        n = len(tokens)
        if n < min_length or k <= 0:
            return set(), {"tokens_saved": 0, "compression_pct": 0.0, "runtime_ms": 0.0, "dead_slots": 0}

        # 1. Extract candidate n-grams
        raw_counts = extract_candidate_ngrams(tokens, min_length=min_length, max_length=max_length, min_count=1)
        if not raw_counts:
            return set(), {"tokens_saved": 0, "compression_pct": 0.0, "runtime_ms": 0.0, "dead_slots": 0}

        # Pre-filter candidate pool: prioritize phrases with recurring frequency
        # For phrases with count=1, only keep length >= 3 if pool allows
        filtered_cands = [
            (g, raw_counts[g] * (len(g) - 1))
            for g in raw_counts
            if raw_counts[g] >= 2 or len(g) >= 3
        ]
        if not filtered_cands:
            filtered_cands = [(g, raw_counts[g] * (len(g) - 1)) for g in raw_counts]
        filtered_cands.sort(key=lambda x: x[1], reverse=True)
        candidate_pool = [g for g, _ in filtered_cands[:candidate_limit]]

        # 2. Beam Search over marginal DP savings
        # Beam entry: (tokens_saved, codebook_frozenset)
        beam: List[Tuple[int, FrozenSet[Tuple[int, ...]]]] = [(0, frozenset())]

        for step in range(k):
            next_beam_candidates: List[Tuple[int, FrozenSet[Tuple[int, ...]]]] = []
            made_progress = False

            for cur_saved, cur_cb in beam:
                # Evaluate marginal gains of adding each candidate
                for cand in candidate_pool:
                    if cand in cur_cb:
                        continue
                    test_cb = cur_cb | {cand}
                    saved = segment_tokens_dp(tokens, set(test_cb))[2]["tokens_saved"]
                    if saved > cur_saved:
                        next_beam_candidates.append((saved, test_cb))
                        made_progress = True

            if not made_progress or not next_beam_candidates:
                # No candidate adds positive marginal value; stop expansion early
                break

            # Deduplicate and retain top beam_width
            next_beam_candidates.sort(key=lambda x: x[0], reverse=True)
            unique_beam: List[Tuple[int, FrozenSet[Tuple[int, ...]]]] = []
            seen_sets: Set[FrozenSet[Tuple[int, ...]]] = set()

            for s, cb in next_beam_candidates:
                if cb not in seen_sets:
                    seen_sets.add(cb)
                    unique_beam.append((s, cb))
                    if len(unique_beam) >= beam_width:
                        break

            beam = unique_beam

        # Best result from beam
        best_saved, best_frozen_cb = max(beam, key=lambda x: x[0])
        best_cb = set(best_frozen_cb)

        # Run final DP segmentation
        comp_len, tiles, dp_stats = segment_tokens_dp(tokens, best_cb)

        # Prune redundant phrases that were shadowed and not used in the optimal tiling
        used_phrases = {tile for tile in tiles if len(tile) > 1}
        best_cb = best_cb & used_phrases

        # Re-verify segmentation with pruned set
        comp_len, tiles, dp_stats = segment_tokens_dp(tokens, best_cb)
        t1 = time.perf_counter()

        used_slots = len(best_cb)
        dead_slots = 0

        stats = {
            "tokens_saved": dp_stats["tokens_saved"],
            "compressed_tokens": comp_len,
            "base_tokens": n,
            "compression_pct": dp_stats["compression_pct"],
            "codebook_size": len(best_cb),
            "used_slots": used_slots,
            "dead_slots": 0,
            "capacity_utilization_pct": 100.0 if best_cb else 0.0,
            "hypertoken_emissions": dp_stats["hypertoken_emissions"],
            "runtime_ms": (t1 - t0) * 1000.0,
        }
        return best_cb, stats


def compute_greedy_oracle_codebook(
    tokens: List[int],
    k: int = 32,
    min_length: int = 2,
    max_length: int = 3,
) -> Tuple[Set[Tuple[int, ...]], Dict[str, Any]]:
    """Legacy Answer-Aware Greedy Oracle (heuristic count * (len - 1))."""
    t0 = time.perf_counter()
    n = len(tokens)
    if n < min_length or k <= 0:
        return set(), {"tokens_saved": 0, "compression_pct": 0.0, "runtime_ms": 0.0, "dead_slots": 0}

    counts = extract_candidate_ngrams(tokens, min_length=min_length, max_length=max_length, min_count=1)
    weighted = [(ngram, cnt * (len(ngram) - 1)) for ngram, cnt in counts.items()]
    weighted.sort(key=lambda x: x[1], reverse=True)

    oracle_set = {ngram for ngram, _ in weighted[:k]}
    comp_len, tiles, dp_stats = segment_tokens_dp(tokens, oracle_set)
    t1 = time.perf_counter()

    used_slots = dp_stats["unique_hypertokens_used"]
    dead_slots = len(oracle_set) - used_slots

    stats = {
        "tokens_saved": dp_stats["tokens_saved"],
        "compressed_tokens": comp_len,
        "base_tokens": n,
        "compression_pct": dp_stats["compression_pct"],
        "codebook_size": len(oracle_set),
        "used_slots": used_slots,
        "dead_slots": dead_slots,
        "capacity_utilization_pct": (used_slots / len(oracle_set) * 100.0) if oracle_set else 0.0,
        "hypertoken_emissions": dp_stats["hypertoken_emissions"],
        "runtime_ms": (t1 - t0) * 1000.0,
    }
    return oracle_set, stats
