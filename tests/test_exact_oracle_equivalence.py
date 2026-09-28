"""Equivalence test verifying ExactHypertokenOracle against brute-force search.

Tests >= 100 randomized test cases across varied token sequence lengths,
vocabularies, repeated patterns, nested substrings, and budget K.
Guarantees 0-1 ILP CP-SAT solver finds mathematically certified global optimum.
"""

from __future__ import annotations

import itertools
import random
import unittest
from typing import List, Set, Tuple

from src.evaluation.offline_segmenter import segment_tokens_dp
from src.zip2zip.predictor_v2.oracle_exact import ExactHypertokenOracle


def brute_force_optimal_savings(
    tokens: List[int],
    k: int,
    min_len: int = 2,
    max_len: int = 4,
    allowed_phrases: Set[Tuple[int, ...]] | None = None,
) -> int:
    """Computes exact optimal savings by exhaustive enumeration of all subsets of size <= k."""
    n = len(tokens)
    candidates: Set[Tuple[int, ...]] = set()

    if allowed_phrases is not None:
        for p in allowed_phrases:
            l = len(p)
            if min_len <= l <= max_len:
                for i in range(n - l + 1):
                    if tuple(tokens[i : i + l]) == p:
                        candidates.add(p)
    else:
        for l in range(min_len, min(max_len + 1, n + 1)):
            for i in range(n - l + 1):
                candidates.add(tuple(tokens[i : i + l]))

    cand_list = sorted(list(candidates))
    if not cand_list or k <= 0:
        return 0

    best_saved = 0
    # Search all combinations of size 1..min(k, len(cand_list))
    for r in range(1, min(k, len(cand_list)) + 1):
        for combo in itertools.combinations(cand_list, r):
            _, _, st = segment_tokens_dp(tokens, set(combo))
            if st["tokens_saved"] > best_saved:
                best_saved = st["tokens_saved"]

    return best_saved


class TestExactOracleEquivalence(unittest.TestCase):
    def setUp(self):
        self.oracle = ExactHypertokenOracle(min_len=2, max_len=4, time_limit_seconds=5.0)

    def test_brute_force_equivalence_100_random_cases(self):
        """Generates 100+ randomized cases with overlapping, nested, and repetitive sequences."""
        rng = random.Random(42)
        total_cases = 120

        for case_idx in range(total_cases):
            # Vary sequence lengths (6 to 16 tokens so brute-force combination search stays tractable)
            seq_len = rng.randint(6, 15)
            # Vary alphabet size (smaller alphabet = more frequent repeated substrings & overlaps)
            alphabet_size = rng.choice([3, 4, 6, 10])
            tokens = [rng.randint(1, alphabet_size) for _ in range(seq_len)]

            # Sometimes inject intentional repeated motifs / nested patterns
            if rng.random() < 0.5:
                motif = [rng.randint(1, alphabet_size) for _ in range(rng.randint(2, 4))]
                # inject motif at 2 random spots
                pos1 = rng.randint(0, max(0, seq_len - len(motif)))
                pos2 = rng.randint(0, max(0, seq_len - len(motif)))
                tokens[pos1 : pos1 + len(motif)] = motif
                tokens[pos2 : pos2 + len(motif)] = motif

            k = rng.randint(1, 3)

            # Test either global oracle (allowed_phrases=None) or candidate pool subset
            use_allowed = (rng.random() < 0.4)
            allowed_phrases = None
            if use_allowed:
                # generate random pool of candidate phrases
                allowed_phrases = set()
                for _ in range(rng.randint(2, 6)):
                    p_len = rng.randint(2, 4)
                    p = tuple(rng.randint(1, alphabet_size) for _ in range(p_len))
                    allowed_phrases.add(p)

            # 1. Brute-force reference
            brute_saved = brute_force_optimal_savings(
                tokens,
                k=k,
                min_len=2,
                max_len=4,
                allowed_phrases=allowed_phrases,
            )

            # 2. CP-SAT Exact solver
            res = self.oracle.solve(tokens, k=k, allowed_phrases=allowed_phrases)

            # Verification assertions
            self.assertEqual(
                res.solver_status,
                "OPTIMAL",
                f"Case {case_idx}: Solver did not report OPTIMAL status",
            )
            self.assertEqual(
                res.optimality_gap,
                0.0,
                f"Case {case_idx}: Optimality gap is {res.optimality_gap}, expected 0.0",
            )
            self.assertEqual(
                res.steps_saved,
                brute_saved,
                f"Case {case_idx} mismatch! Tokens: {tokens}, k={k}, "
                f"Allowed: {allowed_phrases}. "
                f"CP-SAT saved {res.steps_saved} ({res.selected_phrases}), "
                f"Brute-force saved {brute_saved}",
            )


if __name__ == "__main__":
    unittest.main()
