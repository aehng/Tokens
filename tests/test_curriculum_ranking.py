"""Test deterministic curriculum ranking and category preservation."""

import unittest
import pickle
from transformers import AutoTokenizer

from zip2zip.predictor_policy import (
    CappedPredictorPolicy,
    is_structural,
    is_numeric,
    is_bare_punctuation,
    classify_phrase,
)
from zip2zip.predictive_pipeline import PredictivePipeline

PREDICTOR_PATH = "experiments/checkpoints/cached_predictor.pkl"


class TestCurriculumRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        with open(PREDICTOR_PATH, "rb") as f:
            raw_predictor = pickle.load(f)
        cls.p_index = getattr(raw_predictor, "index", raw_predictor)

    def test_01_category_separation(self):
        """Verify numeric is distinguished from structural."""
        # Structural: whitespace / newlines
        newline_tokens = tuple(self.tokenizer.encode("\n\n", add_special_tokens=False))
        indent_tokens = tuple(self.tokenizer.encode("    ", add_special_tokens=False))
        self.assertTrue(is_structural(newline_tokens, self.tokenizer))
        self.assertTrue(is_structural(indent_tokens, self.tokenizer))
        self.assertFalse(is_numeric(newline_tokens, self.tokenizer))

        # Numeric: numbers / percentages / currency
        num_tokens = tuple(self.tokenizer.encode(" 100", add_special_tokens=False))
        dollar_tokens = tuple(self.tokenizer.encode(" $50", add_special_tokens=False))
        self.assertTrue(is_numeric(num_tokens, self.tokenizer))
        self.assertTrue(is_numeric(dollar_tokens, self.tokenizer))
        self.assertFalse(is_structural(num_tokens, self.tokenizer))
        self.assertFalse(is_structural(dollar_tokens, self.tokenizer))

        # Bare punctuation
        punct_tokens = tuple(self.tokenizer.encode("...", add_special_tokens=False))
        self.assertTrue(is_bare_punctuation(punct_tokens, self.tokenizer))
        self.assertEqual(classify_phrase(punct_tokens, self.tokenizer), "bare_punct")

    def test_02_deterministic_ranking(self):
        """Verify policy ranking is 100% deterministic across repeated calls."""
        policy = CappedPredictorPolicy(self.p_index, self.tokenizer, budget=32, max_structural_slots=0)
        prompt = "def calculate_average(values):\n    total = sum(values)\n    count = len(values)\n    return total / count"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        codebook1, _ = policy.select_codebook(prompt_ids)
        codebook2, _ = policy.select_codebook(prompt_ids)

        self.assertEqual(list(codebook1.keys()), list(codebook2.keys()))
        self.assertEqual(list(codebook1.values()), list(codebook2.values()))

    def test_03_curriculum_prefix_preservation(self):
        """Verify lower curriculum density produces a strict prefix of higher density."""
        policy = CappedPredictorPolicy(self.p_index, self.tokenizer, budget=32, max_structural_slots=0)
        pipeline = PredictivePipeline(policy, self.tokenizer, max_codebook_size=32)

        prompt = "Instruction: Given a list of numbers [10, 20, 30], compute the mean and standard deviation.\nAnswer:"
        response = "The mean is 20 and standard deviation is 8.16."

        s_25 = pipeline.process_sample(prompt, response, curriculum_density=0.25)
        s_50 = pipeline.process_sample(prompt, response, curriculum_density=0.50)
        s_75 = pipeline.process_sample(prompt, response, curriculum_density=0.75)
        s_100 = pipeline.process_sample(prompt, response, curriculum_density=1.00)

        p_25 = list(s_25["codebook_dict"].keys())
        p_50 = list(s_50["codebook_dict"].keys())
        p_75 = list(s_75["codebook_dict"].keys())
        p_100 = list(s_100["codebook_dict"].keys())

        # Check monotonic length increase
        self.assertLessEqual(len(p_25), len(p_50))
        self.assertLessEqual(len(p_50), len(p_75))
        self.assertLessEqual(len(p_75), len(p_100))

        # Check strict prefix preservation
        self.assertEqual(p_25, p_50[: len(p_25)], "Density 0.25 must be strict prefix of 0.50")
        self.assertEqual(p_50, p_75[: len(p_50)], "Density 0.50 must be strict prefix of 0.75")
        self.assertEqual(p_75, p_100[: len(p_75)], "Density 0.75 must be strict prefix of 1.00")

    def test_04_no_structural_tokens_when_capped_zero(self):
        """Verify zero structural phrases when max_structural_slots = 0."""
        policy = CappedPredictorPolicy(self.p_index, self.tokenizer, budget=32, max_structural_slots=0)
        prompt = "def foo():\n\n    x = 1\n    return x\n\n"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        codebook, meta = policy.select_codebook(prompt_ids)
        self.assertEqual(meta["structural_phrases"], 0)
        for gram in codebook.keys():
            self.assertFalse(
                is_structural(gram, self.tokenizer),
                f"Phrase {gram} ({self.tokenizer.decode(list(gram))}) should not be structural",
            )


if __name__ == "__main__":
    unittest.main()
