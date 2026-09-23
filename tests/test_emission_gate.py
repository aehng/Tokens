"""Unit tests for ContextualEmissionGate."""

import torch
import unittest
from zip2zip.static_codebook import StaticCodebookManager
from zip2zip.emission_gate import ContextualEmissionGate


class TestContextualEmissionGate(unittest.TestCase):
    def setUp(self):
        self.initial_vocab = 100
        self.max_codebook_size = 8
        self.static_mgr = StaticCodebookManager(
            initial_vocab_size=self.initial_vocab,
            max_codebook_size=self.max_codebook_size,
            max_subtokens=4,
            embedding_dim=64,
            pad_token_id=0,
        )
        # Seed 2 hypertokens:
        # Hypertoken 100 (index 0): [10, 20] -> first token 10
        # Hypertoken 101 (index 1): [50, 60] -> first token 50
        dictionary = {
            (10, 20): 100,
            (50, 60): 101,
        }
        self.static_mgr.set_seeded_codebook(dictionary, batch_size=1)

    def test_top_n_filtering(self):
        """Only hypertoken whose first token is in top-N should remain unmasked."""
        gate = ContextualEmissionGate(self.static_mgr, top_n=5, enabled=True)

        scores = torch.zeros(1, self.initial_vocab + self.max_codebook_size)
        # Make token 10 rank 1 (score 100.0)
        scores[0, 10] = 100.0
        # Make token 50 low rank (score 0.0)
        scores[0, 50] = 0.0
        # Fill other tokens 1..4 with high scores
        for t in range(1, 5):
            scores[0, t] = 50.0

        # Hypertoken logits
        scores[0, 100] = 5.0
        scores[0, 101] = 5.0

        input_ids = torch.tensor([[1, 2, 3]])
        filtered = gate(input_ids, scores)

        # Hypertoken 100 has first token 10 (which is in top 5) -> preserved
        self.assertEqual(filtered[0, 100].item(), 5.0)
        # Hypertoken 101 has first token 50 (not in top 5) -> masked to -inf
        self.assertEqual(filtered[0, 101].item(), float("-inf"))

        stats = gate.get_stats()
        self.assertEqual(stats["positions_evaluated"], 1)
        self.assertEqual(stats["total_candidates_considered"], 2)
        self.assertEqual(stats["total_candidates_gated_out"], 1)
        self.assertEqual(stats["total_candidates_permitted"], 1)

    def test_disabled_is_noop(self):
        """When disabled=False, logits should not be masked."""
        gate = ContextualEmissionGate(self.static_mgr, top_n=1, enabled=False)
        scores = torch.zeros(1, self.initial_vocab + self.max_codebook_size)
        scores[0, 100] = 5.0
        scores[0, 101] = 5.0

        input_ids = torch.tensor([[1, 2, 3]])
        filtered = gate(input_ids, scores)
        self.assertEqual(filtered[0, 100].item(), 5.0)
        self.assertEqual(filtered[0, 101].item(), 5.0)

    def test_probability_threshold(self):
        """Hypertokens whose first token probability is below min_prob are masked."""
        gate = ContextualEmissionGate(self.static_mgr, top_n=None, min_prob=0.2, enabled=True)
        scores = torch.zeros(1, self.initial_vocab + self.max_codebook_size)
        # Put mass on token 10 so P(10) > 0.5
        scores[0, 10] = 10.0
        scores[0, 50] = 0.0

        scores[0, 100] = 2.0
        scores[0, 101] = 2.0

        input_ids = torch.tensor([[1]])
        filtered = gate(input_ids, scores)

        self.assertEqual(filtered[0, 100].item(), 2.0)
        self.assertEqual(filtered[0, 101].item(), float("-inf"))


if __name__ == "__main__":
    unittest.main()
