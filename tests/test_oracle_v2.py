"""Unit tests for Oracle V2: Exact and Near-Optimal Compression Oracles."""

import unittest
from typing import List, Set, Tuple

from src.evaluation.oracle_v2 import (
    ExactOracle,
    OracleV2,
    compute_greedy_oracle_codebook,
)
from src.evaluation.offline_segmenter import segment_tokens_dp


class TestOracleV2(unittest.TestCase):
    def setUp(self):
        # Canonical counterexample where greedy oracle fails due to subphrase shadowing
        self.counterexample_tokens = [1, 2, 3, 99, 1, 2, 3, 88, 1, 2, 3, 77, 4, 5, 66, 4, 5]
        # Realistic Python-like code snippet
        self.code_tokens = [
            10, 20, 30, 40, 50, 10, 20, 30, 60, 70,
            10, 20, 30, 40, 50, 80, 90, 10, 20, 30,
            100, 110, 40, 50, 120, 40, 50,
        ]

    def test_synthetic_counterexample_v2_beats_greedy(self):
        """Oracle V2 must strictly outperform the legacy greedy oracle on overlapping patterns."""
        cb_greedy, stats_greedy = compute_greedy_oracle_codebook(self.counterexample_tokens, k=3)
        cb_v2, stats_v2 = OracleV2.compute_codebook(self.counterexample_tokens, k=3, beam_width=4)

        self.assertGreater(stats_v2["tokens_saved"], stats_greedy["tokens_saved"])
        self.assertEqual(stats_greedy["tokens_saved"], 6)
        self.assertEqual(stats_v2["tokens_saved"], 10)
        self.assertEqual(stats_v2["dead_slots"], 0, "Oracle V2 should not carry dead slots")

    def test_exact_oracle_matches_ground_truth(self):
        """ExactOracle must find the provable global maximum on small instances."""
        cb_exact, saved_exact, stats_exact = ExactOracle.solve(
            self.counterexample_tokens, k=3, min_length=2, max_length=3
        )
        self.assertEqual(saved_exact, 10)
        self.assertLessEqual(len(cb_exact), 3)

    def test_oracle_v2_matches_exact_oracle(self):
        """Oracle V2 beam search must match or achieve >= 95% of exact oracle savings."""
        _, saved_exact, _ = ExactOracle.solve(self.code_tokens, k=3, min_length=2, max_length=3)
        _, stats_v2 = OracleV2.compute_codebook(self.code_tokens, k=3, beam_width=4)

        self.assertGreaterEqual(stats_v2["tokens_saved"], int(0.95 * saved_exact))
        self.assertGreaterEqual(stats_v2["tokens_saved"], saved_exact - 1)

    def test_oracle_v2_early_stopping_no_dead_slots(self):
        """Oracle V2 must terminate when marginal gains drop to 0 rather than padding with dead slots."""
        # A simple sequence with only one repeating pair [1, 2]
        simple_tokens = [1, 2, 99, 1, 2, 88, 77]
        # Request k=10
        cb, stats = OracleV2.compute_codebook(simple_tokens, k=10)
        self.assertLessEqual(len(cb), 2, "Should terminate early (<=2 slots) instead of filling all 10 slots")
        self.assertEqual(stats["dead_slots"], 0)
        self.assertEqual(stats["used_slots"], len(cb))

    def test_max_length_4_supported(self):
        """Oracle V2 correctly identifies length-4 phrases when configured."""
        tokens_with_4gram = [1, 2, 3, 4, 99, 1, 2, 3, 4, 88, 1, 2, 3, 4]
        cb, stats = OracleV2.compute_codebook(tokens_with_4gram, k=2, min_length=2, max_length=4)
        self.assertIn((1, 2, 3, 4), cb)
        self.assertEqual(stats["tokens_saved"], 9)  # 3 occurrences * 3 saved = 9

    def test_oracle_v2_determinism(self):
        """Oracle V2 must be strictly deterministic across repeated runs."""
        cb1, _ = OracleV2.compute_codebook(self.code_tokens, k=5, beam_width=4)
        cb2, _ = OracleV2.compute_codebook(self.code_tokens, k=5, beam_width=4)
        self.assertEqual(cb1, cb2)

    def test_product_token_constraints_exclude_disabled_and_out_of_range_ids(self):
        """Product-constrained ceilings cannot select special/disabled token IDs."""
        tokens = [32007, 4, 32007, 4, 32007, 4, 32012, 5, 32012, 5, 32012, 5, 8, 9, 8, 9, 8, 9]
        disabled = {0, 1, 2, 32000, 32007}
        codebook, _ = OracleV2.compute_codebook(
            tokens,
            k=32,
            min_length=2,
            max_length=3,
            disabled_token_ids=disabled,
            valid_token_min=0,
            valid_token_max_exclusive=32011,
        )
        self.assertTrue(codebook)
        self.assertTrue(all(0 <= token < 32011 and token not in disabled for phrase in codebook for token in phrase))
        self.assertFalse(any(32007 in phrase for phrase in codebook))
        self.assertFalse(any(token >= 32011 for phrase in codebook for token in phrase))


if __name__ == "__main__":
    unittest.main()
