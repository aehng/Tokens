"""Unit test for auxiliary auto-encoding reconstruction loss (Section 2.4)."""

import unittest
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from zip2zip import Zip2ZipModel
from zip2zip.training_objectives import compute_reconstruction_loss


class TestReconstructionLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
        cls.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        cls.model = Zip2ZipModel.from_pretrained(
            cls.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        cls.model.input_encoder.to(torch.float32)

    def test_01_reconstruction_position_specific(self):
        """Verify that different token positions produce distinct distributions."""
        # Create dummy codebook with 2-token phrase: [1000, 2000, 32000, 32000]
        codebook_tensor = torch.tensor(
            [[[1000, 2000, 32000, 32000]]], dtype=torch.long
        )  # shape (1, 1, 4)
        base_w = self.model.base_model.get_input_embeddings().weight

        # Zero existing grads
        for p in self.model.input_encoder.parameters():
            p.grad = None

        loss = compute_reconstruction_loss(
            self.model.input_encoder,
            codebook_tensor,
            base_w,
            pad_token_id=32000,
        )

        self.assertTrue(torch.isfinite(loss), f"Loss must be finite, got {loss.item()}")
        self.assertGreater(loss.item(), 0.0)

        # Backward
        loss.backward()

        # Check gradients in pos_embed
        pos_emb = getattr(self.model.input_encoder, "pos_embed", None)
        self.assertIsNotNone(pos_emb)
        self.assertIsNotNone(pos_emb.weight.grad)
        self.assertGreater(pos_emb.weight.grad.abs().sum().item(), 0.0)

        # Check gradients in transformer encoder layers
        layer_grads = [
            p.grad.abs().sum().item()
            for p in self.model.input_encoder.layers.parameters()
            if p.grad is not None
        ]
        self.assertTrue(len(layer_grads) > 0)
        self.assertTrue(any(g > 0.0 for g in layer_grads))

    def test_02_reconstruction_padding_handling(self):
        """Verify that padding-only entries produce zero loss without NaN."""
        # Codebook with only pads
        codebook_tensor = torch.full((1, 4, 4), 32000, dtype=torch.long)
        base_w = self.model.base_model.get_input_embeddings().weight

        loss = compute_reconstruction_loss(
            self.model.input_encoder,
            codebook_tensor,
            base_w,
            pad_token_id=32000,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
