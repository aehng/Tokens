"""CPU regression tests for explicit PEFT adapter construction."""

from __future__ import annotations

import os
import subprocess
import sys
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import src.zip2zip.model as model_module
from src.zip2zip.config import CompressionConfig, Zip2ZipConfig
from src.zip2zip.model import Zip2ZipModel
from src.zip2zip.static_codebook import StaticCodebookManager
from src.zip2zip.nn.encoders.config import AttentionEncoderConfig, EncoderType
from experiments.run_phi_attribution_benchmark import base_phi_weight_sha256


class TinyEncoder(nn.Module):
    def get_encoder_fn(self):
        return lambda codebook, embeddings, pad_token_id: embeddings[:, :1]


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(6, 4)
        self.lm_head = nn.Linear(4, 6, bias=False)
        self.config = SimpleNamespace(vocab_size=6, hidden_size=4)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs


def _tiny_zip2zip_config() -> Zip2ZipConfig:
    return Zip2ZipConfig(
        base_model_name_or_path="tiny-pinned-phi",
        encoder_type=EncoderType.ATTENTION,
        encoder=AttentionEncoderConfig(
            hidden_size=4,
            tie_encoders=True,
            num_heads=1,
        ),
        compression=CompressionConfig(
            initial_vocab_size=6,
            max_codebook_size=2,
            max_subtokens=4,
            disabled_ids=[0],
        ),
        codebook_backend="static",
    )


def test_b0_from_pretrained_never_calls_peft_adapter_loader(monkeypatch):
    from peft import PeftModel
    import src.zip2zip.model as model_module

    config = _tiny_zip2zip_config()
    base_model = TinyCausalLM()
    manager = StaticCodebookManager(
        initial_vocab_size=6,
        max_codebook_size=2,
        max_subtokens=4,
        embedding_dim=4,
        pad_token_id=0,
    )
    calls = []

    def forbidden_peft_load(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("B0 must not call PeftModel.from_pretrained")

    monkeypatch.setattr(PeftModel, "from_pretrained", forbidden_peft_load)
    monkeypatch.setattr(
        model_module.Zip2ZipConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: config),
    )
    monkeypatch.setattr(
        Zip2ZipModel,
        "_align_encoder_flags_with_training_args",
        staticmethod(lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        Zip2ZipModel,
        "load_pretrained_hyper_encoders",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        model_module.BaseEncoder,
        "from_config",
        staticmethod(lambda *args, **kwargs: TinyEncoder()),
    )
    monkeypatch.setitem(sys.modules, "torchao", None)

    wrapped = Zip2ZipModel.from_pretrained(
        "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
        base_model=base_model,
        codebook_backend="static",
        codebook_manager=manager,
        load_peft_adapter=False,
    )

    assert calls == []
    assert wrapped.base_model is base_model
    assert getattr(wrapped.base_model, "peft_config", None) is None
    assert manager.num_seeded == 0
    assert isinstance(wrapped.base_model.get_input_embeddings(), nn.Embedding)
    assert isinstance(wrapped.base_model.get_output_embeddings(), nn.Linear)
    assert base_phi_weight_sha256(base_model) == base_phi_weight_sha256(wrapped)


def test_zip2zip_model_import_does_not_require_peft_or_torchao():
    repo_root = Path(__file__).resolve().parents[1]
    code = r"""
import builtins
real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "peft" or name.startswith("peft.") or name == "torchao" or name.startswith("torchao."):
        raise ImportError("blocked optional adapter dependency: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import zip2zip.model
"""
    env = dict(os.environ)
    source_path = str(repo_root / "src")
    env["PYTHONPATH"] = source_path + (";" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_span_position_hook_can_be_removed_without_a_wrapper_base_cycle():
    from zip2zip.config import CompressionConfig as CoreCompressionConfig
    from zip2zip.config import Zip2ZipConfig as CoreZip2ZipConfig
    from zip2zip.nn.encoders.config import AttentionEncoderConfig as CoreAttentionEncoderConfig
    from zip2zip.nn.encoders.config import EncoderType as CoreEncoderType

    config = CoreZip2ZipConfig(
        base_model_name_or_path="tiny-pinned-phi",
        encoder_type=CoreEncoderType.ATTENTION,
        encoder=CoreAttentionEncoderConfig(hidden_size=4, tie_encoders=True, num_heads=1),
        compression=CoreCompressionConfig(
            initial_vocab_size=6,
            max_codebook_size=2,
            max_subtokens=4,
            disabled_ids=[0],
        ),
        codebook_backend="static",
    )
    base_model = TinyCausalLM()
    manager = StaticCodebookManager(
        initial_vocab_size=6,
        max_codebook_size=2,
        max_subtokens=4,
        embedding_dim=4,
        pad_token_id=0,
    )
    wrapped = Zip2ZipModel(config, base_model=base_model, codebook_manager=manager)
    wrapped.enable_base_token_positions()
    wrapped_ref = weakref.ref(wrapped)
    base_ref = weakref.ref(base_model)

    assert "prepare_inputs_for_generation" in base_model.__dict__
    assert wrapped.remove_base_position_generation_hook() is True
    assert "prepare_inputs_for_generation" not in base_model.__dict__

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        del wrapped
        del base_model
        assert wrapped_ref() is None
        assert base_ref() is None
    finally:
        if was_enabled:
            gc.enable()


def test_b1_construction_loads_upstream_peft_without_checkpoint(monkeypatch, tmp_path):
    import experiments.run_phi_attribution_benchmark as benchmark

    peft_loads = []
    checkpoint_loads = []

    base_model = TinyCausalLM()
    config = _tiny_zip2zip_config()
    manager = StaticCodebookManager(
        initial_vocab_size=6,
        max_codebook_size=2,
        max_subtokens=4,
        embedding_dim=4,
        pad_token_id=0,
    )

    class DummyPeftModel(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.peft_config = {"default": SimpleNamespace(modules_to_save=[])}
            self.active_adapter = "default"
            self.lora_A = nn.Parameter(torch.zeros(2, 4))

        def get_base_model(self):
            return self.base

        def get_input_embeddings(self):
            return self.base.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.base.set_input_embeddings(value)

        def get_output_embeddings(self):
            return self.base.get_output_embeddings()

        def set_output_embeddings(self, value):
            self.base.set_output_embeddings(value)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return self.base.prepare_inputs_for_generation(*args, **kwargs)

    def mock_zip2zip_from_pretrained(*args, **kwargs):
        load_peft = kwargs.get("load_peft_adapter", False)
        base = kwargs.get("base_model")
        if load_peft:
            peft_loads.append((args, kwargs))
            wrapped_base = DummyPeftModel(base)
        else:
            wrapped_base = base
        m = Zip2ZipModel(config, base_model=wrapped_base, codebook_manager=manager)
        return m

    def mock_load_joint_checkpoint(*args, **kwargs):
        checkpoint_loads.append((args, kwargs))
        return {"step": 100, "checkpoint_applied": True}

    monkeypatch.setattr(benchmark.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(benchmark.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: base_model)
    monkeypatch.setattr(benchmark.Zip2ZipModel, "from_pretrained", mock_zip2zip_from_pretrained)
    monkeypatch.setattr(benchmark, "load_joint_checkpoint", mock_load_joint_checkpoint)

    model, tokenizer, report = benchmark.load_predictive_bundle(
        tmp_path / "chk_step_100.pt",
        "cpu",
        load_peft_adapter=True,
        apply_joint_checkpoint=False,
    )

    assert len(peft_loads) == 1
    assert len(checkpoint_loads) == 0
    assert report["checkpoint_applied"] is False
    assert report["step100_h_encoder_checkpoint_loaded"] is False
    assert report["upstream_adapter_id"] == benchmark.CANONICAL_ZIP2ZIP_ID
    assert report["upstream_adapter_revision"] == benchmark.CANONICAL_ZIP2ZIP_REVISION
    assert report["active_lora_during_generation"] is True
    assert report["adapter_param_count"] > 0


def test_b2_construction_loads_peft_and_step100_checkpoint(monkeypatch, tmp_path):
    import experiments.run_phi_attribution_benchmark as benchmark

    peft_loads = []
    checkpoint_loads = []

    base_model = TinyCausalLM()
    config = _tiny_zip2zip_config()
    manager = StaticCodebookManager(
        initial_vocab_size=6,
        max_codebook_size=2,
        max_subtokens=4,
        embedding_dim=4,
        pad_token_id=0,
    )

    class DummyPeftModel(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.peft_config = {"default": SimpleNamespace(modules_to_save=[])}
            self.active_adapter = "default"
            self.lora_A = nn.Parameter(torch.zeros(2, 4))

        def get_base_model(self):
            return self.base

        def get_input_embeddings(self):
            return self.base.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.base.set_input_embeddings(value)

        def get_output_embeddings(self):
            return self.base.get_output_embeddings()

        def set_output_embeddings(self, value):
            self.base.set_output_embeddings(value)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return self.base.prepare_inputs_for_generation(*args, **kwargs)

    def mock_zip2zip_from_pretrained(*args, **kwargs):
        peft_loads.append((args, kwargs))
        return Zip2ZipModel(config, base_model=DummyPeftModel(kwargs.get("base_model")), codebook_manager=manager)

    def mock_load_joint_checkpoint(*args, **kwargs):
        checkpoint_loads.append((args, kwargs))
        return {"step": 100, "checkpoint_applied": True, "step100_h_encoder_checkpoint_loaded": True}

    monkeypatch.setattr(benchmark.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(benchmark.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: base_model)
    monkeypatch.setattr(benchmark.Zip2ZipModel, "from_pretrained", mock_zip2zip_from_pretrained)
    monkeypatch.setattr(benchmark, "load_joint_checkpoint", mock_load_joint_checkpoint)

    model, tokenizer, report = benchmark.load_predictive_bundle(
        tmp_path / "chk_step_100.pt",
        "cpu",
        load_peft_adapter=True,
        apply_joint_checkpoint=True,
    )

    assert len(peft_loads) == 1
    assert len(checkpoint_loads) == 1
    assert report["checkpoint_applied"] is True
    assert checkpoint_loads[0][1]["expected_step"] == 100


def test_b1_h_disabled_cannot_emit_hypertokens(monkeypatch):
    import torch
    import experiments.run_phi_attribution_benchmark as benchmark
    from experiments.run_phi_attribution_benchmark import (
        verify_empty_static_h_mask,
        prepare_empty_b0_manager,
    )

    monkeypatch.setattr(benchmark, "INITIAL_VOCAB_SIZE", 6)

    config = _tiny_zip2zip_config()
    base_model = TinyCausalLM()
    manager = StaticCodebookManager(
        initial_vocab_size=6,
        max_codebook_size=2,
        max_subtokens=4,
        embedding_dim=4,
        pad_token_id=0,
    )
    wrapped = Zip2ZipModel(config, base_model=base_model, codebook_manager=manager)

    prepared_mgr = prepare_empty_b0_manager(wrapped, k=2, device=torch.device("cpu"))
    assert prepared_mgr.num_seeded == 0
    assert prepared_mgr.hyper_to_subtokens == {}
    assert prepared_mgr.subtokens_to_hyper == {}
    assert verify_empty_static_h_mask(prepared_mgr, torch.device("cpu")) is True


def test_b1_b0_preserve_same_frozen_base_fingerprint():
    base_model = TinyCausalLM()
    config = _tiny_zip2zip_config()
    manager_b0 = StaticCodebookManager(
        initial_vocab_size=6,
        max_codebook_size=2,
        max_subtokens=4,
        embedding_dim=4,
        pad_token_id=0,
    )
    b0_wrapped = Zip2ZipModel(config, base_model=base_model, codebook_manager=manager_b0)

    class DummyPeftModel(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.peft_config = {"default": SimpleNamespace(modules_to_save=[])}
            self.active_adapter = "default"
            self.lora_A = nn.Parameter(torch.zeros(2, 4))

        def get_base_model(self):
            return self.base

        def get_input_embeddings(self):
            return self.base.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.base.set_input_embeddings(value)

        def get_output_embeddings(self):
            return self.base.get_output_embeddings()

        def set_output_embeddings(self, value):
            self.base.set_output_embeddings(value)

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return self.base.prepare_inputs_for_generation(*args, **kwargs)

    manager_b1 = StaticCodebookManager(
        initial_vocab_size=6,
        max_codebook_size=2,
        max_subtokens=4,
        embedding_dim=4,
        pad_token_id=0,
    )
    b1_wrapped = Zip2ZipModel(config, base_model=DummyPeftModel(base_model), codebook_manager=manager_b1)

    hash_base = base_phi_weight_sha256(base_model)
    hash_b0 = base_phi_weight_sha256(b0_wrapped)
    hash_b1 = base_phi_weight_sha256(b1_wrapped)

    assert hash_b0 == hash_base
    assert hash_b1 == hash_base


def test_disable_adapter_context_restores_base_model_logits_on_wrapped_model():
    """Verify Option B offline: disabling PEFT adapter restores exact base-model logits."""
    from peft import LoraConfig, get_peft_model

    base_lm = TinyCausalLM()
    # Give non-trivial weights
    with torch.no_grad():
        base_lm.embed_tokens.weight.copy_(torch.randn_like(base_lm.embed_tokens.weight))
        base_lm.lm_head.weight.copy_(torch.randn_like(base_lm.lm_head.weight))

    x = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        vanilla_tokens = base_lm.embed_tokens(x)
        vanilla_logits = base_lm.lm_head(vanilla_tokens)

    peft_cfg = LoraConfig(r=2, lora_alpha=2, target_modules=["lm_head"], init_lora_weights=False)
    peft_lm = get_peft_model(base_lm, peft_cfg)

    # Put arbitrary non-zero values in lora weights
    with torch.no_grad():
        for n, p in peft_lm.named_parameters():
            if "lora" in n:
                p.copy_(torch.randn_like(p) * 0.5)

    with torch.no_grad():
        # Active adapter alters logits
        adapted_tokens = peft_lm.get_input_embeddings()(x)
        adapted_logits = peft_lm.get_output_embeddings()(adapted_tokens)
        shift = (adapted_logits - vanilla_logits).abs().max().item()
        assert shift > 1e-4, "Active LoRA must change logits"

        # Disabled adapter restores exact base logits
        with peft_lm.disable_adapter():
            restored_tokens = peft_lm.get_input_embeddings()(x)
            restored_logits = peft_lm.get_output_embeddings()(restored_tokens)
            restored_shift = (restored_logits - vanilla_logits).abs().max().item()
            assert restored_shift == 0.0, "Disabled adapter must produce identical logits to base model"

