"""Unit tests verifying the predictive training data pipeline.

Verifies:
1. 100% exact lossless round-trip token reconstruction on code, math, and instruction.
2. Capped predictor policy strictly obeys K=32 and <=8 structural slots.
3. Training labels are masked to -100 on the prompt portion and active on response.
4. Codebook tensor and spans are formatted properly for encoder consumption.
"""

import unittest
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip.predictor_policy import CappedPredictorPolicy
from zip2zip.predictive_pipeline import PredictivePipeline
from experiments.load_oracle_predictor import load_oracle_predictor

PREDICTOR_PATH = "experiments/checkpoints/oracle_guided_predictor.pkl"


class TestPredictivePipelineRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        raw_predictor = load_oracle_predictor(PREDICTOR_PATH)
        p_index = getattr(raw_predictor, "index", raw_predictor)
        cls.policy = CappedPredictorPolicy(
            p_index, cls.tokenizer, budget=32, max_structural_slots=8
        )
        cls.pipeline = PredictivePipeline(cls.policy, cls.tokenizer, max_codebook_size=32)

    def test_01_roundtrip_code_domain(self):
        """Test exact lossless reconstruction on Python code."""
        prompt = "def binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n"
        response = "    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1"
        sample = self.pipeline.process_sample(prompt, response, domain="code")

        # Verify exact token reconstruction
        self.assertEqual(
            self.pipeline.decode_sequence(sample["compressed_prompt_ids"], self.policy_to_dict(sample)),
            sample["original_prompt_ids"],
        )
        self.assertEqual(
            self.pipeline.decode_sequence(sample["compressed_response_ids"], self.policy_to_dict(sample)),
            sample["original_response_ids"],
        )

    def test_02_roundtrip_math_reasoning_domain(self):
        """Test exact lossless reconstruction on GSM8k math reasoning."""
        prompt = "Solve step by step:\nA store sells 12 apples for $6. How much do 30 apples cost?\n"
        response = "1. Price per apple = 6 / 12 = $0.50.\n2. Total cost for 30 apples = 30 * 0.50 = $15.00.\n#### 15"
        sample = self.pipeline.process_sample(prompt, response, domain="reasoning")

        self.assertEqual(
            self.pipeline.decode_sequence(sample["compressed_prompt_ids"], self.policy_to_dict(sample)),
            sample["original_prompt_ids"],
        )
        self.assertEqual(
            self.pipeline.decode_sequence(sample["compressed_response_ids"], self.policy_to_dict(sample)),
            sample["original_response_ids"],
        )

    def test_03_roundtrip_instruction_domain(self):
        """Test exact lossless reconstruction on conversational instruction."""
        prompt = "Instruction: Outline three major advantages of electric vehicles over gas cars.\nAnswer:"
        response = "1. Lower operating and maintenance costs.\n2. Zero tailpipe emissions.\n3. Instant torque and quieter ride."
        sample = self.pipeline.process_sample(prompt, response, domain="instruction")

        self.assertEqual(
            self.pipeline.decode_sequence(sample["compressed_prompt_ids"], self.policy_to_dict(sample)),
            sample["original_prompt_ids"],
        )
        self.assertEqual(
            self.pipeline.decode_sequence(sample["compressed_response_ids"], self.policy_to_dict(sample)),
            sample["original_response_ids"],
        )

    def test_04_category_caps_and_label_masking(self):
        """Test that structural phrases are capped <= 8 and prompt labels are -100."""
        prompt = "for i in range(10):\n    for j in range(10):\n        print(i, j)\n"
        response = "    return sum(range(100))\n"
        sample = self.pipeline.process_sample(prompt, response, domain="code")

        meta = sample["policy_meta"]
        self.assertLessEqual(meta["structural_phrases"], 8)
        self.assertLessEqual(meta["total_phrases"], 32)

        # Labels: prompt portion must be -100
        prompt_len = len(sample["compressed_prompt_ids"])
        self.assertEqual(sample["labels"][:prompt_len], [-100] * prompt_len)
        # Response portion must match compressed response tokens
        self.assertEqual(sample["labels"][prompt_len:], sample["compressed_response_ids"])

    def test_05_prompt_only_causality(self):
        """Test that codebook selection depends strictly on the prompt and is invariant to the response."""
        prompt = "def compute_factorial(n):\n    if n <= 1:\n        return 1\n"
        response_a = "    return n * compute_factorial(n - 1)"
        response_b = "    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"

        sample_a = self.pipeline.process_sample(prompt, response_a, domain="code")
        sample_b = self.pipeline.process_sample(prompt, response_b, domain="code")

        # The codebook dictionary must be 100% identical regardless of response
        self.assertEqual(
            sample_a["codebook_dict"],
            sample_b["codebook_dict"],
            "Causality violation: changing the response altered the predicted codebook!",
        )

    def policy_to_dict(self, sample):
        return {eval(k) if isinstance(k, str) else k: v for k, v in sample["codebook_dict"].items()}


if __name__ == "__main__":
    unittest.main()
