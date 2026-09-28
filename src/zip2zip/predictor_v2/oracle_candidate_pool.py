"""Oracle B: Fixed Candidate-Pool Oracle for Predictor V2.

Evaluates the exact ceiling of opportunity reachable from the SHARED prompt-only candidate pool.
Cheats using the Vanilla continuation to select the <= K candidates that maximize realized DP savings.
Uses mathematically certified exact 0-1 ILP CP-SAT solver.
Directly quantifies candidate-generation loss:
    candidate_generation_capture = candidate_pool_oracle_steps / global_occurrence_oracle_steps
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.evaluation.offline_segmenter import segment_tokens_dp
from src.zip2zip.predictor_v2.candidate_pool import CandidateRecord
from src.zip2zip.predictor_v2.oracle_exact import ExactHypertokenOracle


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
    solver_status: str = "OPTIMAL"
    optimality_gap: float = 0.0


class CandidatePoolOracle:
    """Computes the maximum realized compression achievable from the fixed candidate pool."""

    def __init__(self, tokenizer: Optional[Any] = None, time_limit_seconds: float = 10.0):
        self.tokenizer = tokenizer
        self.exact_solver = ExactHypertokenOracle(
            min_len=2,
            max_len=4,
            time_limit_seconds=time_limit_seconds,
        )

    def solve(
        self,
        candidate_records: Sequence[CandidateRecord],
        continuation_tokens: Sequence[int],
        k: int,
        global_oracle_steps: Optional[int] = None,
    ) -> CandidatePoolOracleResult:
        """Selects the best <= K candidates from candidate_records using exact CP-SAT."""
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
                solver_status="TRIVIAL_EMPTY",
                optimality_gap=0.0,
            )

        cand_set = {c.tokens for c in candidate_records}
        cand_map = {c.tokens: c for c in candidate_records}

        exact_res = self.exact_solver.solve(
            tokens=continuation_tokens,
            k=k,
            allowed_phrases=cand_set,
            tokenizer=self.tokenizer,
        )

        steps = exact_res.steps_saved
        cap = (steps / global_oracle_steps) if (global_oracle_steps and global_oracle_steps > 0) else 1.0
        lost = max(0, (global_oracle_steps or steps) - steps)

        # Mark candidate records
        for p in exact_res.selected_phrases:
            rec = cand_map.get(p)
            if rec is not None:
                if k == 8:
                    rec.candidate_pool_oracle_k8 = True
                elif k == 16:
                    rec.candidate_pool_oracle_k16 = True
                elif k == 32:
                    rec.candidate_pool_oracle_k32 = True

        return CandidatePoolOracleResult(
            k=k,
            selected_phrases=exact_res.selected_phrases,
            selected_texts=exact_res.selected_texts,
            steps_saved=steps,
            compressed_length=exact_res.compressed_length,
            emissions=exact_res.emissions,
            unique_phrases_used=exact_res.unique_phrases_used,
            codebook_utilization=exact_res.codebook_utilization,
            runtime_ms=exact_res.runtime_ms,
            is_exact=exact_res.is_exact,
            candidate_count_considered=len(candidate_records),
            candidate_generation_capture=round(cap, 4),
            opportunity_lost_steps=lost,
            optimization_method="exact_cpsat_01_ilp",
            solver_status=exact_res.solver_status,
            optimality_gap=exact_res.optimality_gap,
        )
