"""Oracle A: Global Occurrence Oracle for Predictor V2.

Computes the unrestricted physical ceiling of decode steps that could be saved
if the future token stream were known perfectly.
Does NOT depend on prompt candidate generator, predictor, or safety heuristics.
"""

from __future__ import annotations

import heapq
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from src.evaluation.offline_segmenter import segment_tokens_dp


@dataclass
class OracleResult:
    k: int
    selected_phrases: List[Tuple[int, ...]]
    selected_texts: List[str]
    steps_saved: int
    compressed_length: int
    emissions: int
    unique_phrases_used: int
    codebook_utilization: float
    runtime_ms: float
    is_exact: bool
    candidate_count_considered: int
    optimality_gap_upper_bound: int
    optimization_method: str


class GlobalOccurrenceOracle:
    """Computes the theoretical compression ceiling over all occurring n-grams (len 2..4)."""

    def __init__(
        self,
        min_len: int = 2,
        max_len: int = 4,
    ):
        self.min_len = min_len
        self.max_len = max_len

    def extract_occurring_ngrams(
        self,
        tokens: Sequence[int],
    ) -> Counter[Tuple[int, ...]]:
        """Extracts every valid occurring n-gram of length min_len..max_len."""
        counts: Counter[Tuple[int, ...]] = Counter()
        n = len(tokens)
        for l in range(self.min_len, min(self.max_len + 1, n + 1)):
            for i in range(n - l + 1):
                gram = tuple(tokens[i : i + l])
                counts[gram] += 1
        return counts

    def solve(
        self,
        tokens: Sequence[int],
        k: int,
        tokenizer: Optional[Any] = None,
    ) -> OracleResult:
        """Finds the optimal or near-optimal <= K subset of all occurring n-grams."""
        t0 = time.perf_counter()
        n = len(tokens)
        if n < self.min_len or k <= 0:
            return OracleResult(
                k=k,
                selected_phrases=[],
                selected_texts=[],
                steps_saved=0,
                compressed_length=n,
                emissions=0,
                unique_phrases_used=0,
                codebook_utilization=0.0,
                runtime_ms=0.0,
                is_exact=True,
                candidate_count_considered=0,
                optimality_gap_upper_bound=0,
                optimization_method="trivial_empty",
            )

        ngram_counts = self.extract_occurring_ngrams(tokens)
        candidate_count = len(ngram_counts)

        # 1. Compute singleton DP savings for each occurring candidate
        # Candidates that save 0 steps alone cannot improve an empty codebook.
        scored_candidates: List[Tuple[int, Tuple[int, ...]]] = []
        for gram, count in ngram_counts.items():
            isolated = (len(gram) - 1) * count
            # Quick upper bound check: if isolated == 0, skip
            if isolated <= 0:
                continue
            _, _, st = segment_tokens_dp(list(tokens), {gram})
            singleton_saved = st["tokens_saved"]
            if singleton_saved > 0:
                scored_candidates.append((singleton_saved, gram))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        active_candidates = [g for s, g in scored_candidates]

        # Case 1: Active candidates <= K. All can be included; result is mathematically exact!
        if len(active_candidates) <= k:
            selected_set = set(active_candidates)
            comp_len, _, st = segment_tokens_dp(list(tokens), selected_set)
            t1 = time.perf_counter()
            texts = [tokenizer.decode(list(p)) if tokenizer else str(p) for p in active_candidates]
            return OracleResult(
                k=k,
                selected_phrases=active_candidates,
                selected_texts=texts,
                steps_saved=st["tokens_saved"],
                compressed_length=st["compressed_tokens"],
                emissions=st["hypertoken_emissions"],
                unique_phrases_used=st["unique_hypertokens_used"],
                codebook_utilization=st["codebook_utilization"],
                runtime_ms=(t1 - t0) * 1000.0,
                is_exact=True,
                candidate_count_considered=candidate_count,
                optimality_gap_upper_bound=0,
                optimization_method="exact_all_active_fit_in_k",
            )

        # Case 2: Accelerated Lazy Greedy Forward Selection with heap bounds
        # Maintains heap of upper bounds on marginal gain
        # heap entry: (-upper_bound, -singleton_gain, gram)
        heap: List[Tuple[int, int, Tuple[int, ...]]] = []
        for s_saved, gram in scored_candidates:
            heapq.heappush(heap, (-s_saved, -s_saved, gram))

        selected: List[Tuple[int, ...]] = []
        selected_set: Set[Tuple[int, ...]] = set()
        curr_saved = 0

        # Track top unselected upper bounds to bound the optimality gap
        last_marginal_gains = []

        for step in range(k):
            found = False
            while heap:
                neg_ub, neg_single, best_g = heapq.heappop(heap)
                # Compute exact marginal gain of adding best_g to selected_set
                _, _, st = segment_tokens_dp(list(tokens), selected_set | {best_g})
                marginal_gain = st["tokens_saved"] - curr_saved

                if marginal_gain <= 0:
                    continue

                # Submodular lazy check: if marginal_gain >= top of heap, best_g is guaranteed best
                if not heap or marginal_gain >= -heap[0][0]:
                    selected.append(best_g)
                    selected_set.add(best_g)
                    curr_saved += marginal_gain
                    last_marginal_gains.append(marginal_gain)
                    found = True
                    break
                else:
                    # Reinsert with updated tighter upper bound
                    heapq.heappush(heap, (-marginal_gain, neg_single, best_g))

            if not found or not heap:
                break

        # Compute remaining upper bound over unselected items
        remaining_ub_sum = 0
        temp_heap = heap.copy()
        remaining_slots = k - len(selected)
        for _ in range(remaining_slots):
            if temp_heap:
                neg_ub, _, _ = heapq.heappop(temp_heap)
                remaining_ub_sum += (-neg_ub)

        # Final DP evaluation
        comp_len, _, final_st = segment_tokens_dp(list(tokens), selected_set)
        t1 = time.perf_counter()

        texts = [tokenizer.decode(list(p)) if tokenizer else str(p) for p in selected]
        is_exact = (len(heap) == 0 or remaining_ub_sum == 0)

        return OracleResult(
            k=k,
            selected_phrases=selected,
            selected_texts=texts,
            steps_saved=final_st["tokens_saved"],
            compressed_length=final_st["compressed_tokens"],
            emissions=final_st["hypertoken_emissions"],
            unique_phrases_used=final_st["unique_hypertokens_used"],
            codebook_utilization=final_st["codebook_utilization"],
            runtime_ms=(t1 - t0) * 1000.0,
            is_exact=is_exact,
            candidate_count_considered=candidate_count,
            optimality_gap_upper_bound=remaining_ub_sum if not is_exact else 0,
            optimization_method="lazy_greedy_dp_submodular",
        )
