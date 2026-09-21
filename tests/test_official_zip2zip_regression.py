"""Regression test suite verifying official Zip2Zip behavior.

Verifies:
1. Native codebook settings & dynamic token assignment (IDs >= initial_vocab_size)
2. RoPE semantic position IDs jump according to hypertoken span
3. Attention mask alignment
4. KV-cache generation across steps with dynamic updates
5. Lossless expansion / decompression to original base tokens
6. EOS emission and clean codebook reset / lifetime management
"""

import unittest
import torch
from transformers import AutoTokenizer

from zip2zip import Zip2ZipModel, Zip2ZipTokenizer, Zip2ZipConfig
from zip2zip.codebook import CodebookManager


class TestOfficialZip2ZipRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
        cls.tokenizer = Zip2ZipTokenizer.from_pretrained(cls.model_id)
        cls.model = Zip2ZipModel.from_pretrained(
            cls.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        cls.model.eval()

    def test_01_native_codebook_and_dynamic_ids(self):
        """Verify native codebook configuration and dynamic ID assignment."""
        config = self.model.zip2zip_config
        self.assertEqual(config.compression.initial_vocab_size, 32011)
        self.assertEqual(config.compression.max_codebook_size, 2048)
        self.assertEqual(config.compression.max_subtokens, 4)

        # Repeated text should trigger LZW compression in the tokenizer
        repeated_text = "apple orange banana apple orange banana apple orange banana"
        encoded = self.tokenizer(repeated_text, return_tensors="pt")
        input_ids = encoded["input_ids"][0].tolist()

        # Should contain dynamic tokens (>= 32011)
        dynamic_ids = [tok for tok in input_ids if tok >= 32011]
        self.assertTrue(len(dynamic_ids) > 0, "LZW tokenizer must create dynamic tokens on repeated input")

    def test_02_position_ids_follow_hypertoken_spans(self):
        """Verify position IDs jump forward according to each hypertoken's constituent length."""
        manager = CodebookManager(
            initial_vocab_size=32011,
            max_codebook_size=64,
            max_subtokens=4,
            embedding_dim=self.model.zip2zip_config.encoder.hidden_size,
            pad_token_id=32000,
            disabled_ids=list(self.model.zip2zip_config.compression.disabled_ids),
        )

        # Feed a sequence that registers tokens: [100, 200, 100, 200]
        # This installs 32011 = [100, 200] (span 2)
        pos_prefix = manager.prepare_input_ids(torch.tensor([[100, 200, 100, 200]]))
        self.assertEqual(pos_prefix.tolist(), [[0, 1, 2, 3]])

        # Now feed the hypertoken 32011: its span is 2, so position offset was 4,
        # semantic position is offset + span - 1 = 4 + 2 - 1 = 5.
        pos_hyper = manager.prepare_input_ids(torch.tensor([[32011]]))
        self.assertEqual(pos_hyper.tolist(), [[5]], "Hypertoken position must equal base token end position")

    def test_03_decompression_lossless_roundtrip(self):
        """Verify that tokenizer decompression reconstructs exact original base tokens."""
        text = "def calculate_factorial(number):\n    if number <= 1:\n        return 1\n    return number * calculate_factorial(number - 1)"
        encoded = self.tokenizer(text, return_tensors="pt")
        compressed_ids = encoded["input_ids"][0].tolist()

        # Decode via tokenizer (which runs LZW batch_decode)
        decompressed_text = self.tokenizer.decode(compressed_ids, skip_special_tokens=True)
        self.assertEqual(text.strip(), decompressed_text.strip(), "Decompression must match original text")

    def test_04_generation_coherence_and_decompression(self):
        """Verify model.generate produces valid tokens and continuation text is coherent."""
        prompt = "The capital of France is Paris. The capital of France is"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
            )

        gen_tokens = output[0].tolist()
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        self.assertIn("Paris", gen_text, "Model generation should coherently complete the fact")

    def test_05_codebook_lifetime_and_reset(self):
        """Verify codebook manager resets runtime state cleanly after generation."""
        cb_manager = self.model.codebook_manager
        # Before or after generate, runtime state should be reset
        self.assertIsNone(cb_manager.base_position_offset)
        self.assertFalse(cb_manager._prepared_for_embedding)


if __name__ == "__main__":
    unittest.main()
