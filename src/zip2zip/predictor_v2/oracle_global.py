"""Oracle A: Global Occurrence Oracle for Predictor V2.

Computes the unrestricted physical ceiling of decode steps that could be saved
if the future token stream were known perfectly.
Uses mathematically certified exact 0-1 ILP CP-SAT solver.
Does NOT depend on prompt candidate generator, predictor, or safety heuristics.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from src.evaluation.offline_segmenter import segment_tokens_dp
from src.zip2zip.predictor_v2.oracle_exact import ExactHypertokenOracle


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
    optimality_gap_upper_bound: float
    optimization_method: str
    solver_status: str = "OPTIMAL"
    steps_saved_lower_bound: int = 0
    steps_saved_upper_bound: int = 0


class GlobalOccurrenceOracle:
    """Computes the theoretical compression ceiling over all occurring n-grams (len 2..4)."""

    def __init__(
        self,
        min_len: int = 2,
        max_len: int = 4,
        time_limit_seconds: float = 10.0,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.exact_solver = ExactHypertokenOracle(
            min_len=min_len,
            max_len=max_len,
            time_limit_seconds=time_limit_seconds,
        )

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
        """Finds the mathematically certified optimal <= K subset of all occurring n-grams."""
        exact_res = self.exact_solver.solve(
            tokens=tokens,
            k=k,
            allowed_phrases=None,
            tokenizer=tokenizer,
        )

        return OracleResult(
            k=k,
            selected_phrases=exact_res.selected_phrases,
            selected_texts=exact_res.selected_texts,
            steps_saved=exact_res.steps_saved,
            compressed_length=exact_res.compressed_length,
            emissions=exact_res.emissions,
            unique_phrases_used=exact_res.unique_phrases_used,
            codebook_utilization=exact_res.codebook_utilization,
            runtime_ms=exact_res.runtime_ms,
            is_exact=exact_res.is_exact,
            candidate_count_considered=exact_res.candidate_count_considered,
            optimality_gap_upper_bound=exact_res.optimality_gap,
            optimization_method="exact_cpsat_01_ilp",
            solver_status=exact_res.solver_status,
            steps_saved_lower_bound=exact_res.steps_saved,
            steps_saved_upper_bound=exact_res.objective_upper_bound or exact_res.steps_saved,
        )
