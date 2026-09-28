"""Oracle B: Fixed Candidate-Pool Oracle for Predictor V2.

Evaluates the exact ceiling of opportunity reachable from the SHARED prompt-only candidate pool.
Cheats using the Vanilla continuation to select the <= K candidates that maximize realized DP savings.
Directly quantifies candidate-generation loss:
    candidate_generation_capture = candidate_pool_oracle_steps / global_occurrence_oracle_steps
"""

from __future__ import annotations

import heapq
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.evaluation.offline_segmenter import segment_tokens_dp
from src.zip2zip.predictor_v2.candidate_pool import CandidateRecord
from src.zip2zip.predictor_v2.oracle_global import OracleResult


@dataclass
class CandidatePoolOracleResult:
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
    candidate_generation_capture: float
    opportunity_lost_steps: int
    optimization_method: str


class CandidatePoolOracle:
    """Computes the maximum realized compression achievable from the fixed candidate pool."""

    def __init__(self, tokenizer: Optional[Any] = None):
        self.tokenizer = tokenizer

    def solve(
        self,
        candidate_records: Sequence[CandidateRecord],
        continuation_tokens: Sequence[int],
        k: int,
        global_oracle_steps: Optional[int] = None,
    ) -> CandidatePoolOracleResult:
        """Selects the best <= K candidates from candidate_records that maximize DP savings."""
        t0 = time.perf_counter()
        n = len(continuation_tokens)
        if n < 2 or k <= 0 or not candidate_records:
            return CandidatePoolOracleResult(
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
                candidate_count_considered=len(candidate_records),
                candidate_generation_capture=0.0,
                opportunity_lost_steps=global_oracle_steps or 0,
                optimization_method="trivial_empty",
            )

        # 1. Filter candidates to those that actually occur in the continuation
        occurring_cands = [c for c in candidate_records if c.occurs_in_vanilla]
        candidate_count = len(occurring_cands)

        if not occurring_cands:
            t1 = time.perf_counter()
            return CandidatePoolOracleResult(
                k=k,
                selected_phrases=[],
                selected_texts=[],
                steps_saved=0,
                compressed_length=n,
                emissions=0,
                unique_phrases_used=0,
                codebook_utilization=0.0,
                runtime_ms=(t1 - t0) * 1000.0,
                is_exact=True,
                candidate_count_considered=len(candidate_records),
                candidate_generation_capture=0.0,
                opportunity_lost_steps=global_oracle_steps or 0,
                optimization_method="no_occurring_candidates",
            )

        # 2. Evaluate singleton DP savings for occurring candidates
        scored: List[Tuple[int, CandidateRecord]] = []
        for c in occurring_cands:
            _, _, st = segment_tokens_dp(list(continuation_tokens), {c.tokens})
            saved = st["tokens_saved"]
            if saved > 0:
                scored.append((saved, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        active = [c for s, c in scored]

        # Case 1: Active candidates <= K. All can be included!
        if len(active) <= k:
            sel_phrases = [c.tokens for c in active]
            comp_len, _, st = segment_tokens_dp(list(continuation_tokens), set(sel_phrases))
            t1 = time.perf_counter()
            steps = st["tokens_saved"]
            cap = (steps / global_oracle_steps) if (global_oracle_steps and global_oracle_steps > 0) else 1.0
            lost = max(0, (global_oracle_steps or steps) - steps)

            # Mark candidates
            for c in active:
                if k == 8:
                    c.candidate_pool_oracle_k8 = True
                elif k == 16:
                    c.candidate_pool_oracle_k16 = True
                elif k == 32:
                    c.candidate_pool_oracle_k32 = True

            return CandidatePoolOracleResult(
                k=k,
                selected_phrases=sel_phrases,
                selected_texts=[c.text for c in active],
                steps_saved=steps,
                compressed_length=st["compressed_tokens"],
                emissions=st["hypertoken_emissions"],
                unique_phrases_used=st["unique_hypertokens_used"],
                codebook_utilization=st["codebook_utilization"],
                runtime_ms=(t1 - t0) * 1000.0,
                is_exact=True,
                candidate_count_considered=len(candidate_records),
                candidate_generation_capture=round(cap, 4),
                opportunity_lost_steps=lost,
                optimization_method="exact_all_occurring_fit_in_k",
            )

        # Case 2: Lazy greedy forward selection with heap
        heap: List[Tuple[int, int, CandidateRecord]] = []
        counter = 0
        for saved, cand in scored:
            counter += 1
            heapq.heappush(heap, (-saved, counter, cand))

        selected: List[CandidateRecord] = []
        selected_set: Set[Tuple[int, ...]] = set()
        curr_saved = 0

        for step in range(k):
            found = False
            while heap:
                neg_gain, _, best_cand = heapq.heappop(heap)
                # Compute marginal gain
                _, _, st = segment_tokens_dp(list(continuation_tokens), selected_set | {best_cand.tokens})
                marginal = st["tokens_saved"] - curr_saved

                if marginal <= 0:
                    continue

                if not heap or marginal >= -heap[0][0]:
                    selected.append(best_cand)
                    selected_set.add(best_cand.tokens)
                    curr_saved += marginal
                    best_cand.marginal_dp_saved = marginal
                    if k == 8:
                        best_cand.candidate_pool_oracle_k8 = True
                    elif k == 16:
                        best_cand.candidate_pool_oracle_k16 = True
                    elif k == 32:
                        best_cand.candidate_pool_oracle_k32 = True
                    found = True
                    break
                else:
                    counter += 1
                    heapq.heappush(heap, (-marginal, counter, best_cand))

            if not found or not heap:
                break

        comp_len, _, final_st = segment_tokens_dp(list(continuation_tokens), selected_set)
        t1 = time.perf_counter()
        steps = final_st["tokens_saved"]
        cap = (steps / global_oracle_steps) if (global_oracle_steps and global_oracle_steps > 0) else 1.0
        lost = max(0, (global_oracle_steps or steps) - steps)

        return CandidatePoolOracleResult(
            k=k,
            selected_phrases=[c.tokens for c in selected],
            selected_texts=[c.text for c in selected],
            steps_saved=steps,
            compressed_length=final_st["compressed_tokens"],
            emissions=final_st["hypertoken_emissions"],
            unique_phrases_used=final_st["unique_hypertokens_used"],
            codebook_utilization=final_st["codebook_utilization"],
            runtime_ms=(t1 - t0) * 1000.0,
            is_exact=(len(heap) == 0),
            candidate_count_considered=len(candidate_records),
            candidate_generation_capture=round(cap, 4),
            opportunity_lost_steps=lost,
            optimization_method="lazy_greedy_dp_submodular",
        )
