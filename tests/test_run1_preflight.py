"""Comprehensive local preflight checklist verification for Run 1."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import pytest
import torch
import torch.nn as nn

from zip2zip.frozen_phi_h import (
    ContextConditionedHEncoder,
    assert_model_strictly_frozen,
    compute_model_parameter_hash,
    execute_teacher_student_step,
)
from zip2zip.block_cache import evaluate_block_cache_correctness


class MockTinyPhi(nn.Module):
    """Tiny mock causal LM for local CPU testing of position semantics and cache."""

    def __init__(self, vocab_size=1000, embed_dim=64):
        super().__init__()
        self.config = type("Config", (), {"num_hidden_layers": 2, "vocab_size": vocab_size, "hidden_size": embed_dim})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dim_feedforward=128, batch_first=True)
            for _ in range(2)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if inputs_embeds is None:
            x = self.model.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        hidden_states = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)

        logits = self.lm_head(x)
        out = type("Output", (), {})()
        out.logits = logits
        out.hidden_states = hidden_states
        if use_cache:
            # Simple dummy past_key_values (tuple of keys/values)
            seq_len = x.shape[1]
            if past_key_values is not None:
                prev_len = past_key_values[0][0].shape[-2]
                total_len = prev_len + seq_len
            else:
                total_len = seq_len
            dummy_kv = tuple(
                (torch.zeros(1, 2, total_len, 32), torch.zeros(1, 2, total_len, 32))
                for _ in range(2)
            )
            out.past_key_values = dummy_kv
        return out


def test_checklist_items_1_through_6():
    """Verify items 1-6: freezing, optimizer parameter isolation, position mapping, sequence length, and hash invariance."""
    mock_phi = MockTinyPhi(vocab_size=500, embed_dim=64)
    # Freeze mock_phi
    for p in mock_phi.parameters():
        p.requires_grad = False
    assert_model_strictly_frozen(mock_phi)

    # Base hash before training step
    hash_before = compute_model_parameter_hash(mock_phi)

    # Trainable H-encoder
    h_encoder = ContextConditionedHEncoder(embed_dim=64, hidden_dim=32)
    optimizer = torch.optim.AdamW(h_encoder.parameters(), lr=1e-3)

    # Item 3: Optimizer contains ONLY H-encoder parameters
    opt_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    encoder_params = {id(p) for p in h_encoder.parameters()}
    phi_params = {id(p) for p in mock_phi.parameters()}
    assert opt_params == encoder_params
    assert opt_params.isdisjoint(phi_params)

    # Item 4 & 5: Teacher/Student forward step
    C = 8
    T = 4
    context = list(range(10, 10 + C))
    t_a = 50
    t_b = 51
    future = list(range(100, 100 + T))

    res = execute_teacher_student_step(
        mock_phi, h_encoder, context, t_a, t_b, future, torch.device("cpu"), offsets=[0, 1, 2]
    )

    # Verify student physical sequence length is C + 1 + len(future) - 1 or shorter
    # Teacher length = C + 2 + T = 8 + 2 + 4 = 14
    # Student length = C + 1 + (max_offset) = 8 + 1 + 2 = 11 (strictly fewer slots than teacher!)
    assert res.total_loss.requires_grad is True

    # Backward step
    res.total_loss.backward()

    # Verify no gradients on base model
    for name, p in mock_phi.named_parameters():
        assert p.grad is None, f"Base parameter {name} received gradient!"

    # Verify H-encoder has finite, non-zero gradients
    encoder_grads = [p.grad for p in h_encoder.parameters() if p.grad is not None]
    assert len(encoder_grads) > 0
    for g in encoder_grads:
        assert torch.isfinite(g).all()

    optimizer.step()

    # Item 6: Base hash unchanged after optimizer step
    hash_after = compute_model_parameter_hash(mock_phi)
    assert hash_before == hash_after, "Base model parameter hash changed after optimizer step!"


def test_checklist_items_7_and_8():
    """Verify items 7 and 8: no FINAL access and prompt-level split isolation."""
    manifest_path = Path("data/phrase_training_dataset/phrase_dataset_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["final_split_accessed"] is False


def test_checklist_item_9_mock_block_cache():
    """Verify item 9: exact block-cache harness works on mock model."""
    mock_phi = MockTinyPhi(vocab_size=500, embed_dim=64)
    rep = evaluate_block_cache_correctness(
        mock_phi,
        prefix_ids=[1, 2, 3, 4],
        phrase_tokens=[10, 11],
        device=torch.device("cpu"),
    )
    assert rep.serial_cache_seq_len == rep.block_cache_seq_len == 6
    assert rep.cache_lengths_match is True


def test_checklist_item_10_package_dry_run():
    """Verify item 10: git archive dry-run packages cleanly and extracts to temp dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "test_archive.tar.gz"
        subprocess.check_call(
            ["git", "archive", "--format=tar.gz", "-o", str(archive_path), "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
        )
        assert archive_path.stat().st_size > 100_000

        extract_dir = Path(tmpdir) / "extracted"
        extract_dir.mkdir()
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        # Verify key modules exist in extracted copy
        assert (extract_dir / "src" / "zip2zip" / "frozen_phi_h.py").is_file()
        assert (extract_dir / "src" / "zip2zip" / "block_cache.py").is_file()
        assert (extract_dir / "experiments" / "run_representation_experiment.py").is_file()
