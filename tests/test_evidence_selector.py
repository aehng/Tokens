"""Unit tests for EvidenceAwareSelector."""

import os
import time
import unittest
from typing import Tuple

from transformers import AutoTokenizer
from zip2zip.evidence_selector import EvidenceAwareSelector
from experiments.load_oracle_predictor import load_oracle_predictor

PREDICTOR_PATH = "experiments/checkpoints/oracle_guided_predictor.pkl"


class TestEvidenceAwareSelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        if os.path.exists(PREDICTOR_PATH):
            raw_pred = load_oracle_predictor(PREDICTOR_PATH)
            cls.index = getattr(raw_pred, "index", raw_pred)
        else:
            # Mock index for testing environment
            class MockIndex:
                disabled_ids = [0, 1, 2]
                max_subtokens = 4
                token_associations = {}
                precomputed_global_static = []
            cls.index = MockIndex()

        cls.selector = EvidenceAwareSelector(
            predictor_index=cls.index,
            tokenizer=cls.tokenizer,
            budget=32,
        )

    def test_prompt_present_vs_absent_numbers(self):
        """Prompt-present numbers must score higher than prompt-absent numbers."""
        prompt = "There are 42 apples in the basket and 7 people sharing them."
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Tokens for " 42" and " 99"
        tok_42 = tuple(self.tokenizer.encode(" 42", add_special_tokens=False))
        tok_99 = tuple(self.tokenizer.encode(" 99", add_special_tokens=False))

        # Score both directly with equal base weights
        score_42, meta_42 = self.selector.score_candidate(
            phrase=tok_42,
            base_weight=10.0,
            prompt_ids_set=set(prompt_ids),
            prompt_ngrams=self.selector.extract_prompt_ngrams(prompt_ids),
            prompt_text=prompt,
            prompt_numbers={"42", "7"},
            prompt_words={"apples", "basket", "people", "sharing"},
        )

        score_99, meta_99 = self.selector.score_candidate(
            phrase=tok_99,
            base_weight=10.0,
            prompt_ids_set=set(prompt_ids),
            prompt_ngrams=self.selector.extract_prompt_ngrams(prompt_ids),
            prompt_text=prompt,
            prompt_numbers={"42", "7"},
            prompt_words={"apples", "basket", "people", "sharing"},
        )

        self.assertEqual(meta_42.get("numeric_status"), "grounded")
        self.assertEqual(meta_99.get("numeric_status"), "ungrounded_penalized")
        self.assertGreater(score_42, score_99 + 40.0, "Grounded number should substantially outscore ungrounded number")

    def test_dead_structural_phrases_heavily_discounted(self):
        """Dead structural patterns like '. The' or '\n    return' must be penalized when not in prompt."""
        prompt = "def add(a, b):\n    # compute sum\n    "
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        dead_phrase = tuple(self.tokenizer.encode(". The", add_special_tokens=False))
        score_dead, meta_dead = self.selector.score_candidate(
            phrase=dead_phrase,
            base_weight=20.0,
            prompt_ids_set=set(prompt_ids),
            prompt_ngrams=self.selector.extract_prompt_ngrams(prompt_ids),
            prompt_text=prompt,
            prompt_numbers=set(),
            prompt_words={"def", "add", "compute", "sum"},
        )

        self.assertTrue(meta_dead.get("dead_structural", False))
        self.assertLess(score_dead, 0.0, "Dead structural phrase should receive a heavy negative penalty")

    def test_code_syntax_fragments_without_context_rejected(self):
        """Isolated code syntax fragments like '):\\n' without prompt grounding must be penalized."""
        prompt = "Explain why the sky is blue and how Rayleigh scattering works."
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        frag = tuple(self.tokenizer.encode("):\n", add_special_tokens=False))
        score_frag, meta_frag = self.selector.score_candidate(
            phrase=frag,
            base_weight=15.0,
            prompt_ids_set=set(prompt_ids),
            prompt_ngrams=self.selector.extract_prompt_ngrams(prompt_ids),
            prompt_text=prompt,
            prompt_numbers=set(),
            prompt_words={"explain", "sky", "blue", "rayleigh", "scattering", "works"},
        )

        self.assertTrue(meta_frag.get("isolated_syntax", False))
        self.assertLess(score_frag, 0.0, "Isolated syntax fragment without context should be penalized")

    def test_latency_under_50ms(self):
        """Selector must execute in well under 50ms per prompt."""
        prompt = "Write a python function to compute the Fibonacci sequence up to n terms and return the list of values."
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Warmup
        self.selector.select_codebook(prompt_ids, prompt_text=prompt)

        # Benchmark 10 runs
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            _, meta = self.selector.select_codebook(prompt_ids, prompt_text=prompt)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_lat = sum(latencies) / len(latencies)
        self.assertLess(mean_lat, 50.0, f"Mean latency ({mean_lat:.2f}ms) exceeds 50ms limit")
        self.assertLess(mean_lat, 20.0, f"Target latency is under 20ms, got {mean_lat:.2f}ms")

    def test_deterministic_output(self):
        """Selector output must be strictly deterministic across repeated runs."""
        prompt = "Calculate 15 * 8 - 30 and explain each arithmetic step."
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        cb1, meta1 = self.selector.select_codebook(prompt_ids, prompt_text=prompt)
        cb2, meta2 = self.selector.select_codebook(prompt_ids, prompt_text=prompt)

        self.assertEqual(cb1, cb2, "Codebooks from identical inputs must be identical")
        self.assertEqual(meta1["phrases"], meta2["phrases"])


if __name__ == "__main__":
    unittest.main()
