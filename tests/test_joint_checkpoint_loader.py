import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from experiments.load_joint_checkpoint import (
    CHECKPOINT_LOADER_ID,
    accept_cached_generation,
    is_historical_provisional_baseline,
    load_joint_checkpoint,
    mark_historical_baseline,
    stamp_generation_record,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_encoder = nn.Linear(2, 2, bias=False)
        self.output_encoder = nn.Linear(2, 2, bias=False)
        self.base_model = nn.Module()
        self.base_model.embed_wrapper = nn.Module()
        # Mirrors Zip2Zip's hyper-module alias inside its embedding wrappers.
        self.base_model.embed_wrapper.input_encoder = self.input_encoder
        self.base_model.output_wrapper = nn.Module()
        self.base_model.output_wrapper.output_encoder = self.output_encoder
        self.base_model.backbone = nn.Linear(2, 2, bias=False)
        self.base_model.lora_A = nn.Linear(2, 2, bias=False)
        self.base_model.lora_B = nn.Linear(2, 2, bias=False)


class UnsupportedCheckpointObject:
    pass


def make_checkpoint(model):
    return {
        "step": 100,
        "trainable_mode": "joint",
        "config": {"model": {"name_or_path": "test-zip2zip-model"}, "device": "cpu", "enabled": True},
        "base_hashes": {},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "lora_state_dict": {
            name: torch.full_like(param, 0.25)
            for name, param in model.base_model.named_parameters()
            if "lora" in name.lower()
        },
        "input_encoder_state_dict": {
            name: torch.full_like(value, 0.5)
            for name, value in model.input_encoder.state_dict().items()
        },
        "output_encoder_state_dict": {
            name: torch.full_like(value, 0.75)
            for name, value in model.output_encoder.state_dict().items()
        },
    }


class JointCheckpointLoaderTests(unittest.TestCase):
    def test_loads_nested_weights_exactly_and_ignores_encoder_aliases_in_base_guard(self):
        model = TinyModel()
        checkpoint = make_checkpoint(model)
        base_before = model.base_model.backbone.weight.detach().clone()

        with patch.object(
            model,
            "load_state_dict",
            side_effect=AssertionError("the outer checkpoint must never be loaded as a model state_dict"),
        ) as flat_loader:
            report = load_joint_checkpoint(model, checkpoint)
            flat_loader.assert_not_called()

        self.assertEqual(report["step"], 100)
        self.assertEqual(report["model_id"], "test-zip2zip-model")
        self.assertEqual(report["base_hash_status"], "missing")
        self.assertEqual(report["checkpoint_loader"], CHECKPOINT_LOADER_ID)
        self.assertGreater(report["changed_tensor_count"], 0)
        for component in report["components"].values():
            self.assertEqual(component["missing_keys"], [])
            self.assertEqual(component["unexpected_keys"], [])
            self.assertNotEqual(component["before_sha256"], component["after_sha256"])
            self.assertGreater(component["after_l2_norm"], 0)
            self.assertEqual(set(component["changed_parameter_names"]), set(component["changed_parameter_shapes"]))
            self.assertTrue(component["changed_parameter_names"])
        self.assertTrue(report["frozen_base_unchanged_during_load"])
        self.assertTrue(report["frozen_base_parameter_sha256"])
        self.assertTrue(torch.equal(model.base_model.backbone.weight, base_before))
        self.assertTrue(torch.all(model.base_model.lora_A.weight == 0.25))
        self.assertTrue(torch.all(model.input_encoder.weight == 0.5))
        self.assertTrue(torch.all(model.output_encoder.weight == 0.75))

    def test_rejects_checkpoint_when_no_trainable_tensor_changes(self):
        model = TinyModel()
        checkpoint = make_checkpoint(model)
        parameters = dict(model.base_model.named_parameters())
        for name, value in checkpoint["lora_state_dict"].items():
            parameters[name].data.copy_(value)
        model.input_encoder.load_state_dict(checkpoint["input_encoder_state_dict"])
        model.output_encoder.load_state_dict(checkpoint["output_encoder_state_dict"])

        with self.assertRaisesRegex(RuntimeError, "did not change any expected parameter"):
            load_joint_checkpoint(model, checkpoint)

    def test_requires_checkpoint_training_step(self):
        model = TinyModel()
        checkpoint = make_checkpoint(model)
        checkpoint.pop("step")
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            load_joint_checkpoint(model, checkpoint)

    def test_rejects_a_valid_but_wrong_training_step(self):
        model = TinyModel()
        checkpoint = make_checkpoint(model)
        with self.assertRaisesRegex(ValueError, "expected 150, got 100"):
            load_joint_checkpoint(model, checkpoint, expected_step=150)

    def test_rejects_checkpoint_for_a_different_model_id(self):
        model = TinyModel()
        checkpoint = make_checkpoint(model)
        with self.assertRaisesRegex(ValueError, "wrong joint checkpoint model ID"):
            load_joint_checkpoint(model, checkpoint, expected_model_id="another-model")

    def test_preserves_checkpoint_dtype_exactly(self):
        model = TinyModel()
        checkpoint = make_checkpoint(model)
        checkpoint["lora_state_dict"] = {
            key: value.to(torch.float64) for key, value in checkpoint["lora_state_dict"].items()
        }
        report = load_joint_checkpoint(model, checkpoint)
        self.assertEqual(model.base_model.lora_A.weight.dtype, torch.float64)
        self.assertTrue(torch.all(model.base_model.lora_A.weight == 0.25))
        self.assertEqual(report["lora_tensors"], 2)

    def test_rejects_missing_and_unexpected_lora_keys(self):
        model = TinyModel()
        checkpoint = make_checkpoint(model)
        checkpoint["lora_state_dict"].pop(next(iter(checkpoint["lora_state_dict"])))
        with self.assertRaisesRegex(RuntimeError, "LoRA key mismatch"):
            load_joint_checkpoint(model, checkpoint)

        checkpoint = make_checkpoint(model)
        checkpoint["lora_state_dict"]["unexpected.lora"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            load_joint_checkpoint(model, checkpoint)

    def test_memory_mapped_weights_only_checkpoint_round_trip_and_explicit_safe_failure(self):
        model = TinyModel()
        checkpoint = make_checkpoint(model)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "safe.pt"
            torch.save(checkpoint, path)
            with patch("experiments.load_joint_checkpoint.torch.load", wraps=torch.load) as load:
                load_joint_checkpoint(TinyModel(), path)
            self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
            self.assertIs(load.call_args.kwargs["mmap"], True)
            self.assertIs(load.call_args.kwargs["weights_only"], True)

            bad_path = Path(temp_dir) / "unsupported.pt"

            torch.save({**checkpoint, "unsupported": UnsupportedCheckpointObject()}, bad_path)
            with self.assertRaisesRegex(RuntimeError, "weights_only=True"):
                load_joint_checkpoint(TinyModel(), bad_path)

    def test_mmap_unsupported_fails_without_non_mmap_retry(self):
        with patch("experiments.load_joint_checkpoint.torch.load", side_effect=TypeError("no mmap support")) as load:
            with self.assertRaisesRegex(RuntimeError, "refusing a non-memory-mapped fallback"):
                load_joint_checkpoint(TinyModel(), "unused.pt")
        load.assert_called_once()
        self.assertIs(load.call_args.kwargs["mmap"], True)
        self.assertIs(load.call_args.kwargs["weights_only"], True)

    def test_historical_baseline_is_provisional_not_cacheable(self):
        old = {"condition": "cond_b_evidence_k32"}
        self.assertFalse(accept_cached_generation(old))
        baseline = {"condition": "cond_a_baseline_k32"}
        mark_historical_baseline(baseline)
        self.assertTrue(is_historical_provisional_baseline(baseline))
        self.assertFalse(accept_cached_generation(baseline))
        self.assertFalse(is_historical_provisional_baseline({"condition": "cond_a_baseline_k32"}))
        self.assertEqual(stamp_generation_record(old)["checkpoint_loader"], CHECKPOINT_LOADER_ID)
        self.assertTrue(accept_cached_generation(old))


if __name__ == "__main__":
    unittest.main()
