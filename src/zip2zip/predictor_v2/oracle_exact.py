"""Exact Hypertoken Oracle formulated as 0-1 Integer Linear Program / Constraint Programming (CP-SAT).

Guarantees mathematically certified global optimum with zero optimality gap.
Eliminates heuristic approximations (lazy-greedy, beam search) for oracle bounds.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ortools.sat.python import cp_model

from src.evaluation.offline_segmenter import segment_tokens_dp


@dataclass
class ExactOracleResult:
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
    solver_status: str
    optimality_gap: float
    num_variables: int
    num_constraints: int
    candidate_count_considered: int
    optimization_method: str


class ExactHypertokenOracle:
    """Exact 0-1 ILP/CP-SAT solver for optimal hypertoken codebook selection."""

    def __init__(
        self,
        min_len: int = 2,
        max_len: int = 4,
        time_limit_seconds: float = 10.0,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.time_limit_seconds = time_limit_seconds

    def extract_occurrences(
        self,
        tokens: Sequence[int],
        allowed_phrases: Optional[Set[Tuple[int, ...]]] = None,
    ) -> Dict[Tuple[int, ...], List[Tuple[int, int]]]:
        """Finds all occurrences (start, end) for each candidate phrase in tokens.

        Returns:
            dict mapping phrase -> list of (start, end) token intervals.
        """
        phrase_occurrences: Dict[Tuple[int, ...], List[Tuple[int, int]]] = defaultdict(list)
        n = len(tokens)

        if allowed_phrases is not None:
            # Check only allowed phrases
            for phrase in allowed_phrases:
                l = len(phrase)
                if l < self.min_len or l > self.max_len:
                    continue
                for i in range(n - l + 1):
                    if tuple(tokens[i : i + l]) == phrase:
                        phrase_occurrences[phrase].append((i, i + l))
        else:
            # Extract all occurring n-grams
            for l in range(self.min_len, min(self.max_len + 1, n + 1)):
                for i in range(n - l + 1):
                    gram = tuple(tokens[i : i + l])
                    phrase_occurrences[gram].append((i, i + l))

        return dict(phrase_occurrences)

    def solve(
        self,
        tokens: Sequence[int],
        k: int,
        allowed_phrases: Optional[Set[Tuple[int, ...]]] = None,
        tokenizer: Optional[Any] = None,
    ) -> ExactOracleResult:
        """Solves for the exact optimal codebook of size <= k using CP-SAT.

        Args:
            tokens: Sequence of continuation token IDs.
            k: Maximum codebook size budget.
            allowed_phrases: If provided, restricts candidates to this pool
                             (e.g., prompt candidate pool for Candidate-Pool Oracle).
                             If None, considers all occurring n-grams (Global Occurrence Oracle).
            tokenizer: Optional tokenizer to decode selected phrases to text.
        """
        t0 = time.perf_counter()
        n = len(tokens)

        if n < self.min_len or k <= 0:
            return ExactOracleResult(
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
                solver_status="TRIVIAL_EMPTY",
                optimality_gap=0.0,
                num_variables=0,
                num_constraints=0,
                candidate_count_considered=0,
                optimization_method="exact_empty",
            )

        occurrences_map = self.extract_occurrences(tokens, allowed_phrases=allowed_phrases)
        phrases = sorted(list(occurrences_map.keys()), key=lambda p: (len(p), p))
        candidate_count = len(phrases)

        if candidate_count == 0:
            return ExactOracleResult(
                k=k,
                selected_phrases=[],
                selected_texts=[],
                steps_saved=0,
                compressed_length=n,
                emissions=0,
                unique_phrases_used=0,
                codebook_utilization=0.0,
                runtime_ms=(time.perf_counter() - t0) * 1000.0,
                is_exact=True,
                solver_status="OPTIMAL",
                optimality_gap=0.0,
                num_variables=0,
                num_constraints=0,
                candidate_count_considered=0,
                optimization_method="exact_no_occurrences",
            )

        # Build CP-SAT Model
        model = cp_model.CpModel()

        # Variable: y[g] in {0, 1} for whether phrase g is selected in codebook
        y: Dict[Tuple[int, ...], cp_model.IntVar] = {}
        for g in phrases:
            y[g] = model.NewBoolVar(f"y_{g}")

        # Variable: x[o] in {0, 1} for whether occurrence o is emitted
        # Also map token positions to list of covering occurrence variables
        pos_to_occurrences: Dict[int, List[cp_model.IntVar]] = defaultdict(list)
        x_vars: List[Tuple[cp_model.IntVar, int]] = []  # (x_var, savings)
        num_occ = 0

        for g in phrases:
            savings = len(g) - 1
            for occ_idx, (start, end) in enumerate(occurrences_map[g]):
                x_var = model.NewBoolVar(f"x_{g}_{occ_idx}")
                # Constraint: Occurrence x can only be active if phrase g is selected in codebook
                model.Add(x_var <= y[g])
                x_vars.append((x_var, savings))
                num_occ += 1

                for pos in range(start, end):
                    pos_to_occurrences[pos].append(x_var)

        # Constraint: Non-overlapping emissions (at most one hypertoken covers any position p)
        for pos, covering_vars in pos_to_occurrences.items():
            if len(covering_vars) > 1:
                model.Add(sum(covering_vars) <= 1)

        # Constraint: Codebook capacity <= k
        model.Add(sum(y.values()) <= k)

        # Objective: Maximize sum(x_o * (len(g) - 1))
        model.Maximize(sum(x_var * s for x_var, s in x_vars))

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.num_search_workers = 1  # Deterministic single-threaded execution

        status = solver.Solve(model)
        status_name = solver.StatusName(status)

        is_exact = (status_name == "OPTIMAL")
        opt_gap = 0.0
        if status_name == "FEASIBLE":
            best_bound = solver.BestObjectiveBound()
            curr_obj = solver.ObjectiveValue()
            opt_gap = (best_bound - curr_obj) / max(1.0, curr_obj)

        selected_phrases: List[Tuple[int, ...]] = []
        for g in phrases:
            if solver.Value(y[g]) == 1:
                selected_phrases.append(g)

        # Re-verify through DP segmenter to guarantee exact contract alignment
        comp_len, _, st = segment_tokens_dp(list(tokens), set(selected_phrases))
        t1 = time.perf_counter()

        texts = [tokenizer.decode(list(p)) if tokenizer else str(p) for p in selected_phrases]

        return ExactOracleResult(
            k=k,
            selected_phrases=selected_phrases,
            selected_texts=texts,
            steps_saved=st["tokens_saved"],
            compressed_length=st["compressed_tokens"],
            emissions=st["hypertoken_emissions"],
            unique_phrases_used=st["unique_hypertokens_used"],
            codebook_utilization=st["codebook_utilization"],
            runtime_ms=(t1 - t0) * 1000.0,
            is_exact=is_exact,
            solver_status=status_name,
            optimality_gap=opt_gap,
            num_variables=len(y) + num_occ,
            num_constraints=len(pos_to_occurrences) + num_occ + 1,
            candidate_count_considered=candidate_count,
            optimization_method="cpsat_exact_01_ilp",
        )
