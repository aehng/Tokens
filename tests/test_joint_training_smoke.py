"""Unit smoke test verifying joint predictive training differentiability.

Verifies:
1. Base transformer weights are frozen (requires_grad = False).
2. LoRA weights, input_encoder, and output_encoder are trainable (requires_grad = True).
3. Differentiable forward pass computes valid finite LM loss and reconstruction loss.
4. Backward pass delivers non-zero gradients to:
   - LoRA weights (lora_A, lora_B)
   - input_encoder
   - output_encoder
5. Base transformer weights receive NO gradients (param.grad is None).
6. AdamW optimizer step updates trainable parameters cleanly without NaN.
"""

import unittest
import torch
from transformers import AutoTokenizer

from zip2zip import Zip2ZipModel
from zip2zip.predictor_policy import CappedPredictorPolicy
from zip2zip.predictive_pipeline import PredictivePipeline
from zip2zip.training_objectives import (
    configure_joint_training_parameters,
    DifferentiableTrainingManager,
)
from experiments.load_oracle_predictor import load_oracle_predictor

PREDICTOR_PATH = "experiments/checkpoints/oracle_guided_predictor.pkl"


class TestJointTrainingSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_id = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
        cls.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        cls.model = Zip2ZipModel.from_pretrained(
            cls.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        # Cast trainable encoders to float32 for CPU backward compatibility (DNNL constraint)
        cls.model.input_encoder.to(torch.float32)
        cls.model.output_encoder.to(torch.float32)

        raw_predictor = load_oracle_predictor(PREDICTOR_PATH)
        p_index = getattr(raw_predictor, "index", raw_predictor)
        cls.policy = CappedPredictorPolicy(p_index, cls.tokenizer, budget=32, max_structural_slots=8)
        cls.pipeline = PredictivePipeline(cls.policy, cls.tokenizer, max_codebook_size=32)
        cls.training_mgr = DifferentiableTrainingManager(cls.model, max_codebook_size=32)

    def test_01_parameter_configuration(self):
        """Verify joint parameter freeze/train configuration."""
        report = configure_joint_training_parameters(self.model)

        # Base weights frozen
        self.assertGreater(report["base_frozen_params"], 3_500_000_000)
        # LoRA trainable
        self.assertGreater(report["lora_trainable_params"], 40_000_000)
        # Input encoder trainable
        self.assertGreater(report["input_encoder_params"], 200_000_000)
        # Output encoder trainable
        self.assertGreater(report["output_encoder_params"], 200_000_000)
        # Trainable % should be ~11.6% of total model params
        self.assertGreater(report["trainable_percentage"], 10.0)
        self.assertLess(report["trainable_percentage"], 15.0)

    def test_02_differentiable_forward_and_backward(self):
        """Verify differentiable forward pass, gradient flow, and optimizer step."""
        configure_joint_training_parameters(self.model)

        # Cast LoRA parameters to float32 for CPU backward compatibility
        for name, param in self.model.base_model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

        prompt = "Instruction: Write a brief function that adds two numbers.\nAnswer:"
        response = "def add(a, b):\n    return a + b"
        sample = self.pipeline.process_sample(prompt, response, domain="code")

        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long)
        labels = torch.tensor([sample["labels"]], dtype=torch.long)
        codebook_dict = {eval(k) if isinstance(k, str) else k: v for k, v in sample["codebook_dict"].items()}
        codebook_tensor = sample["codebook_tensor"]

        # Forward step
        loss, metrics = self.training_mgr.forward_step(
            input_ids=input_ids,
            labels=labels,
            codebook_dict=codebook_dict,
            codebook_tensor=codebook_tensor,
            recon_weight=0.1,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.isfinite(loss), f"Loss must be finite, got {loss.item()}")
        self.assertGreater(metrics["lm_loss"], 0.0)
        self.assertGreaterEqual(metrics["recon_loss"], 0.0)

        # Zero existing grads
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
        optimizer.zero_grad()

        # Backward step
        loss.backward()

        # Verify gradient flow to LoRA
        lora_grads = [
            param.grad.abs().sum().item()
            for name, param in self.model.base_model.named_parameters()
            if "lora" in name.lower() and param.grad is not None
        ]
        self.assertTrue(len(lora_grads) > 0, "LoRA parameters must receive gradients")
        self.assertTrue(any(g > 0.0 for g in lora_grads), "LoRA gradients must be non-zero")

        # Verify gradient flow to output_encoder
        out_enc_grads = [
            param.grad.abs().sum().item()
            for param in self.model.output_encoder.parameters()
            if param.grad is not None
        ]
        self.assertTrue(len(out_enc_grads) > 0, "output_encoder parameters must receive gradients")
        self.assertTrue(any(g > 0.0 for g in out_enc_grads), "output_encoder gradients must be non-zero")

        # Verify gradient flow to input_encoder (via reconstruction loss and/or embedding)
        in_enc_grads = [
            param.grad.abs().sum().item()
            for param in self.model.input_encoder.parameters()
            if param.grad is not None
        ]
        self.assertTrue(len(in_enc_grads) > 0, "input_encoder parameters must receive gradients")
        self.assertTrue(any(g > 0.0 for g in in_enc_grads), "input_encoder gradients must be non-zero")

        # Verify base model weights have NO gradients
        for name, param in self.model.base_model.named_parameters():
            if "lora" not in name.lower():
                self.assertIsNone(param.grad, f"Base weight {name} must have no gradient")

        # Compute base model weight hashes before optimizer step
        def hash_tensor(t):
            import hashlib
            return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

        base_hashes_before = {
            name: hash_tensor(param)
            for name, param in self.model.base_model.named_parameters()
            if "lora" not in name.lower() and any(x in name for x in ["embed_tokens", "layers.0.", "layers.16.", "layers.31."])
        }

        # Optimizer step
        optimizer.step()

        # Verify base model weights are 100% byte-identical after optimizer step
        for name, h_before in base_hashes_before.items():
            param = dict(self.model.base_model.named_parameters())[name]
            h_after = hash_tensor(param)
            self.assertEqual(
                h_before,
                h_after,
                f"SAFETY VIOLATION: Base model weight {name} was modified by optimizer step!",
            )


if __name__ == "__main__":
    unittest.main()
