import random
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from zip2zip import (
    CompressionConfig,
    StaticCodebookManager,
    Zip2ZipConfig,
    Zip2ZipModel,
    prepare_model_for_inference,
)
from zip2zip.codebook import CodebookManager
from zip2zip.nn.encoders.config import ResLatentAttnConfig
from zip2zip.nn.embedding import HyperEmbedding
from zip2zip.nn.linear import HyperLinear
from zip2zip.static_codebook import estimate_effective_table_memory


class MeanTokenEncoder:
    """Small deterministic stand-in for the production encoder contract."""

    def get_encoder_fn(self):
        def encode(updates, base_weight, pad_token_id):
            valid = updates.ne(pad_token_id)
            embedded = F.embedding(updates, base_weight)
            summed = (embedded * valid.unsqueeze(-1)).sum(dim=-2)
            return summed / valid.sum(dim=-1, keepdim=True).clamp_min(1)

        return encode


class TinyCausalModel(nn.Module):
    def __init__(self, embedding, output_head):
        super().__init__()
        self.embedding = embedding
        self.output_head = output_head

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.output_head


class TinyPredictiveModel(nn.Module):
    def __init__(self, manager, embedding, output_head):
        super().__init__()
        self.base_model = TinyCausalModel(embedding, output_head)
        self.input_encoder = MeanTokenEncoder()
        self.output_encoder = MeanTokenEncoder()
        self.codebook_manager = manager


def make_manager(*, vocab=12, codebook_size=4, dim=5):
    manager = StaticCodebookManager(
        initial_vocab_size=vocab,
        max_codebook_size=codebook_size,
        max_subtokens=3,
        embedding_dim=dim,
        pad_token_id=0,
    )
    manager.set_seeded_codebook([[2, 3], [4, 5, 6]], batch_size=1)
    return manager


def make_embedding(manager, *, vocab=12, tail=3, dim=5):
    base = nn.Embedding(vocab + tail, dim, padding_idx=0)
    with torch.no_grad():
        base.weight.copy_(torch.arange((vocab + tail) * dim).view(vocab + tail, dim) / 17)
    layer = HyperEmbedding(
        config=None,
        encoder=MeanTokenEncoder(),
        num_embeddings=vocab + tail,
        embedding_dim=dim,
        padding_idx=0,
        device=torch.device("cpu"),
        dtype=base.weight.dtype,
        initial_vocab_size=vocab,
        codebook_manager=manager,
    )
    layer.weight = base.weight
    return layer


def make_linear(manager, *, vocab=12, tail=3, dim=5, bias=True):
    base = nn.Linear(dim, vocab + tail, bias=bias)
    with torch.no_grad():
        base.weight.copy_(torch.arange((vocab + tail) * dim).view(vocab + tail, dim) / 23)
        if base.bias is not None:
            base.bias.copy_(torch.arange(vocab + tail) / 31)
    layer = HyperLinear(
        config=None,
        encoder=MeanTokenEncoder(),
        in_features=dim,
        out_features=vocab + tail,
        bias=bias,
        device=torch.device("cpu"),
        dtype=base.weight.dtype,
        initial_vocab_size=vocab,
        codebook_manager=manager,
    )
    layer.weight = base.weight
    if base.bias is not None:
        layer.bias = base.bias
    return layer


def expected_position_chunk(ids, spans_by_hyper, offsets, attention_mask=None):
    positions = torch.zeros_like(ids)
    next_offsets = offsets.clone()
    for row in range(ids.shape[0]):
        cursor = int(offsets[row, 0])
        for col in range(ids.shape[1]):
            valid = attention_mask is None or bool(attention_mask[row, col])
            if not valid:
                positions[row, col] = 0
                continue
            token_id = int(ids[row, col])
            span = spans_by_hyper.get(token_id, 1)
            cursor += span
            positions[row, col] = cursor - 1
        next_offsets[row, 0] = cursor
    return positions, next_offsets


def test_legacy_embedding_matches_base_h_and_mixed_reference():
    manager = make_manager()
    layer = make_embedding(manager)
    ids = torch.tensor([[1, 12, 7, 13]])

    actual = layer(ids)
    h_weights = manager.hyper_embedding_weight_cache[0]
    expected = torch.stack(
        [layer.weight[1], h_weights[0], layer.weight[7], h_weights[1]]
    ).unsqueeze(0)

    assert torch.equal(actual, expected)
    assert manager.input_encoder_calls == 1


def test_legacy_output_projection_preserves_base_h_insertion_and_tail():
    manager = make_manager()
    layer = make_linear(manager)
    hidden = torch.arange(2 * 5, dtype=torch.float32).view(1, 2, 5) / 7

    actual = layer(hidden)
    base_logits = F.linear(hidden, layer.weight, layer.bias)
    h_weights = manager.hyper_linear_weight_cache[0]
    h_logits = torch.matmul(hidden, h_weights.transpose(0, 1))
    expected = torch.cat(
        [base_logits[..., :12], h_logits, base_logits[..., 12:]], dim=-1
    )

    assert torch.equal(actual, expected)
    assert actual.shape[-1] == 12 + 4 + 3
    assert torch.equal(actual[..., 16:], base_logits[..., 12:])
    assert torch.equal(torch.topk(actual, k=5, dim=-1).indices,
                       torch.topk(expected, k=5, dim=-1).indices)
    assert manager.output_encoder_calls == 1


def test_unprepared_legacy_wrappers_remain_differentiable_for_training():
    manager = make_manager()
    embedding = make_embedding(manager)
    head = make_linear(manager)

    embedded = embedding(torch.tensor([[1, 12, 7, 13]]))
    logits = head(embedded)
    logits.square().mean().backward()

    assert not manager.fast_inference_ready
    assert embedding.weight.grad is not None
    assert head.weight.grad is not None
    assert manager.input_encoder_calls == 1
    assert manager.output_encoder_calls == 1


def test_legacy_embedding_does_not_map_shifted_original_tail_ids():
    """Record the existing mismatch: output tail IDs are shifted, input IDs are not."""
    manager = make_manager()
    layer = make_embedding(manager)
    # The first original tail row is emitted at V + K. Legacy embedding treats
    # it as H slot K, which is outside its K-row H table.
    with pytest.raises(IndexError):
        layer(torch.tensor([[12 + 4]]))


def test_legacy_positions_cover_h2_h3_masks_and_incremental_offsets():
    manager = make_manager()
    ids = torch.tensor([[1, 12, 7, 13]])
    positions = manager.prepare_input_ids(ids)
    assert positions.tolist() == [[0, 2, 3, 6]]

    # Continue as cached generation would: H2 spans the next two base positions.
    assert manager.prepare_input_ids(torch.tensor([[12]])).tolist() == [[8]]

    masked_manager = make_manager()
    masked_ids = torch.tensor([[0, 1, 12, 2], [3, 13, 4, 5]])
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 1]])
    assert masked_manager.prepare_input_ids(masked_ids, mask).tolist() == [
        [0, 0, 2, 3],
        [0, 3, 0, 4],
    ]
    assert masked_manager.base_position_offset.tolist() == [[4], [5]]


def test_legacy_positions_match_python_reference_for_random_mixed_chunks():
    rng = random.Random(7321)
    manager = make_manager()
    spans = {12: 2, 13: 3}
    offsets = torch.zeros((2, 1), dtype=torch.long)

    for length in range(1, 10):
        values = [rng.choice([1, 2, 3, 4, 12, 13]) for _ in range(2 * length)]
        ids = torch.tensor(values, dtype=torch.long).view(2, length)
        mask = torch.tensor(
            [[rng.randrange(2) for _ in range(length)] for _ in range(2)],
            dtype=torch.long,
        )
        expected, offsets = expected_position_chunk(ids, spans, offsets, mask)
        actual = manager.prepare_input_ids(ids, attention_mask=mask)
        assert torch.equal(actual, expected)
        assert torch.equal(manager.base_position_offset, offsets)


def test_trusted_fast_positions_match_legacy_across_masked_chunks():
    legacy = make_manager()
    fast_managers = [make_manager(), make_manager()]
    # Request setup will set this only after validating the codebook and
    # preparing the inference tables. Set it directly here to isolate the
    # branchless position calculation.
    for fast in fast_managers:
        fast.fast_inference_ready = True
    rng = random.Random(9917)

    for length in range(1, 13):
        ids = torch.tensor(
            [rng.choice([1, 2, 3, 4, 12, 13]) for _ in range(2 * length)]
        ).view(2, length)
        mask = torch.tensor(
            [[rng.randrange(2) for _ in range(length)] for _ in range(2)]
        )
        legacy_positions = legacy.prepare_input_ids(ids, attention_mask=mask)
        for row, fast in enumerate(fast_managers):
            fast_positions = fast.prepare_input_ids(
                ids[row : row + 1], attention_mask=mask[row : row + 1]
            )
            assert torch.equal(fast_positions, legacy_positions[row : row + 1])
            assert torch.equal(
                fast.base_position_offset, legacy.base_position_offset[row : row + 1]
            )


def test_prepared_position_path_has_no_any_or_python_tensor_branch():
    manager = make_manager()
    manager.fast_inference_ready = True
    ids = torch.tensor([[1, 12, 7, 13]])
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        manager.prepare_input_ids(ids)
    operator_counts = {event.key: event.count for event in prof.key_averages()}
    assert operator_counts.get("aten::any", 0) == 0

    source = textwrap.dedent(inspect.getsource(StaticCodebookManager.prepare_input_ids))
    fast_branch = source.split("if self.fast_inference_ready:", 1)[1].split(
        "elif is_hyper.any():", 1
    )[0]
    assert ".item(" not in fast_branch
    assert "bool(" not in fast_branch
    assert "elif is_hyper.any():" in source


def test_seeded_codebook_validation_rejects_invalid_or_sparse_ids_atomically():
    manager = StaticCodebookManager(
        initial_vocab_size=12,
        max_codebook_size=4,
        max_subtokens=3,
        embedding_dim=5,
        pad_token_id=0,
        disabled_ids=[9],
    )
    manager.set_seeded_codebook([[2, 3]])
    original = dict(manager.hyper_to_subtokens)

    invalid_definitions = [
        [[-1, 2]],       # negative base token
        [[2, 12]],       # nested/out-of-base-vocabulary token
        [[2, 9]],        # disabled base token
        [[2, 3], [2, 3]],  # duplicate phrase
        {1: [2, 3]},     # sparse slot; mask assumes packed slots
        {16: [2, 3]},    # absolute ID outside [V, V + K)
    ]
    for definition in invalid_definitions:
        with pytest.raises(ValueError):
            manager.set_seeded_codebook(definition)
        assert manager.hyper_to_subtokens == original


def test_new_codebook_invalidates_fast_inference_readiness_and_vectors():
    manager = make_manager()
    manager.fast_inference_ready = True
    manager.hyper_embedding_weight_cache = torch.ones(1, 4, 5)
    manager.hyper_linear_weight_cache = torch.ones(1, 4, 5)

    manager.set_seeded_codebook([[3, 4]])

    assert not manager.fast_inference_ready
    assert manager.hyper_embedding_weight_cache is None
    assert manager.hyper_linear_weight_cache is None


def test_prepared_inference_rejects_batches_before_mutating_state():
    manager = make_manager()
    embedding = make_embedding(manager)
    head = make_linear(manager)
    model = TinyPredictiveModel(manager, embedding, head).eval()
    manager.attach_to_model(model)
    initial = (
        manager.inference_tables_build_count,
        manager.inference_tables_version,
        manager.input_encoder_calls,
        manager.output_encoder_calls,
        manager.effective_embedding_weight_cache,
        manager.effective_linear_weight_cache,
        manager.base_position_offset,
    )

    for kwargs in (
        {"batch_size": 2},
        {"batch_size": 1, "dummy_input_ids": torch.zeros((2, 3), dtype=torch.long)},
    ):
        with pytest.raises(NotImplementedError, match="batch_size=1"):
            manager.prepare_inference_tables(model, **kwargs)
        assert (
            manager.inference_tables_build_count,
            manager.inference_tables_version,
            manager.input_encoder_calls,
            manager.output_encoder_calls,
            manager.effective_embedding_weight_cache,
            manager.effective_linear_weight_cache,
            manager.base_position_offset,
        ) == initial

    manager.prepare_inference_tables(model, batch_size=1)
    assert manager.fast_inference_ready
    assert manager.inference_tables_build_count == 1
    spans_before = manager.hyper_token_spans.clone()
    runtime_batch_before = manager.runtime_batch_size
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        manager.prepare_input_ids(torch.zeros((2, 2), dtype=torch.long))
    assert manager.runtime_batch_size == runtime_batch_before
    assert torch.equal(manager.hyper_token_spans, spans_before)


@pytest.mark.parametrize(
    "generation_config,model_config,expected_error",
    [
        (SimpleNamespace(eos_token_id=11, pad_token_id=0), None, None),
        (SimpleNamespace(eos_token_id=12, pad_token_id=0), None, "eos_token_id"),
        (SimpleNamespace(eos_token_id=11, pad_token_id=15), None, "pad_token_id"),
        (SimpleNamespace(eos_token_id=None, pad_token_id=None), SimpleNamespace(eos_token_id=None), None),
        (SimpleNamespace(eos_token_id=[11, 12], pad_token_id=None), None, "eos_token_id"),
    ],
)
def test_generation_special_token_space_guard(
    generation_config, model_config, expected_error
):
    manager = make_manager()
    embedding = make_embedding(manager)
    head = make_linear(manager)
    model = TinyPredictiveModel(manager, embedding, head).eval()
    model.base_model.generation_config = generation_config
    if model_config is not None:
        model.base_model.config = model_config
    manager.attach_to_model(model)

    if expected_error is None:
        manager.prepare_inference_tables(model)
        assert manager.fast_inference_ready
    else:
        with pytest.raises(ValueError, match=expected_error):
            manager.prepare_inference_tables(model)
        assert not manager.fast_inference_ready
        assert manager.inference_tables_build_count == 0
        assert manager.input_encoder_calls == 0
        assert manager.output_encoder_calls == 0


def test_fast_inference_guards_padding_index_at_hypertoken_boundary():
    manager = make_manager()
    embedding = make_embedding(manager)
    embedding.padding_idx = manager.initial_vocab_size
    model = TinyPredictiveModel(manager, embedding, make_linear(manager)).eval()
    manager.attach_to_model(model)

    with pytest.raises(ValueError, match="padding_idx"):
        manager.prepare_inference_tables(model)
    assert manager.inference_tables_build_count == 0


def test_real_phi_effective_table_memory_estimator_uses_shape_and_dtype_only():
    report = estimate_effective_table_memory(
        (32064, 3072),
        torch.float16,
        (32064, 3072),
        torch.float16,
        codebook_size=32,
        output_bias_shape=(32064,),
        output_bias_dtype=torch.float16,
    )
    expected_embedding = (32064 + 32) * 3072 * 2
    expected_head = (32064 + 32) * 3072 * 2 + (32064 + 32) * 2
    assert report["effective_input_embedding_bytes"] == expected_embedding
    assert report["effective_output_head_bytes"] == expected_head
    assert report["additional_bytes"] == expected_embedding + expected_head


def test_prepared_tables_match_legacy_and_cover_shifted_input_tail():
    legacy_manager = make_manager()
    legacy_embedding = make_embedding(legacy_manager)
    legacy_head = make_linear(legacy_manager)

    fast_manager = make_manager()
    fast_embedding = make_embedding(fast_manager)
    fast_head = make_linear(fast_manager)
    with torch.no_grad():
        fast_embedding.weight.copy_(legacy_embedding.weight)
        fast_head.weight.copy_(legacy_head.weight)
        fast_head.bias.copy_(legacy_head.bias)
    original_input = fast_embedding.weight.detach().clone()
    original_output = fast_head.weight.detach().clone()
    original_bias = fast_head.bias.detach().clone()

    model = TinyPredictiveModel(fast_manager, fast_embedding, fast_head).eval()
    fast_manager.attach_to_model(model)
    memory = fast_manager.prepare_inference_tables(model, batch_size=1)
    assert fast_manager.base_position_offset is None
    assert not fast_manager._prepared_for_embedding

    effective_input = fast_manager.effective_embedding_weight_cache
    effective_output = fast_manager.effective_linear_weight_cache
    effective_bias = fast_manager.effective_linear_bias_cache
    vocab, codebook, original_rows = 12, 4, 15
    assert effective_input.shape == (original_rows + codebook, 5)
    assert effective_output.shape == (original_rows + codebook, 5)
    assert effective_bias.shape == (original_rows + codebook,)
    assert torch.equal(effective_input[:vocab], original_input[:vocab])
    assert torch.equal(effective_output[:vocab], original_output[:vocab])
    assert torch.equal(effective_bias[:vocab], original_bias[:vocab])
    expected_input_h = torch.zeros((codebook, 5))
    expected_output_h = torch.zeros((codebook, 5))
    for slot, phrase in enumerate(fast_manager.hyper_to_subtokens.values()):
        phrase_ids = torch.tensor(phrase)
        expected_input_h[slot] = original_input[phrase_ids].mean(dim=0)
        expected_output_h[slot] = original_output[phrase_ids].mean(dim=0)
    assert torch.equal(effective_input[vocab : vocab + codebook], expected_input_h)
    assert torch.equal(effective_output[vocab : vocab + codebook], expected_output_h)
    assert torch.equal(effective_bias[vocab : vocab + codebook], torch.zeros(codebook))
    assert torch.equal(effective_input[vocab + codebook :], original_input[vocab:])
    assert torch.equal(effective_output[vocab + codebook :], original_output[vocab:])
    assert torch.equal(effective_bias[vocab + codebook :], original_bias[vocab:])

    prompt_ids = torch.tensor([[1, 12, 7, 13]])
    legacy_embedded = legacy_embedding(prompt_ids)
    fast_embedded = fast_embedding(prompt_ids)
    assert torch.equal(fast_embedded, legacy_embedded)

    hidden = torch.arange(2 * 5, dtype=torch.float32).view(1, 2, 5) / 7
    legacy_logits = legacy_head(hidden)
    fast_logits = fast_head(hidden)
    assert torch.allclose(fast_logits, legacy_logits, atol=1e-6, rtol=1e-6)
    assert torch.equal(
        torch.topk(fast_logits, k=5, dim=-1).indices,
        torch.topk(legacy_logits, k=5, dim=-1).indices,
    )
    def greedy_ids(embedding, head, manager, prompt, steps):
        sequence = prompt.clone()
        generated = []
        for _ in range(steps):
            hidden_step = embedding(sequence)[:, -1:, :]
            step_logits = head(hidden_step)[:, -1, :].clone()
            step_logits[..., 16:] = float("-inf")  # keep this fixture in the common ID space
            step_logits = manager.mask_unused_logits(step_logits)
            next_id = step_logits.argmax(dim=-1, keepdim=True)
            generated.append(next_id)
            sequence = torch.cat((sequence, next_id), dim=-1)
        return torch.cat(generated, dim=-1)

    legacy_greedy = greedy_ids(
        legacy_embedding, legacy_head, legacy_manager, torch.tensor([[1, 2]]), 5
    )
    fast_greedy = greedy_ids(
        fast_embedding, fast_head, fast_manager, torch.tensor([[1, 2]]), 5
    )
    assert torch.equal(fast_greedy, legacy_greedy)
    assert torch.count_nonzero(fast_manager.effective_linear_bias_cache[12:16]) == 0
    assert torch.equal(fast_logits[..., 16:], legacy_logits[..., 16:])

    shifted_tail_ids = torch.tensor([[16, 18]])
    expected_tail = torch.stack(
        [legacy_embedding.weight[12], legacy_embedding.weight[14]]
    ).unsqueeze(0)
    assert torch.equal(fast_embedding(shifted_tail_ids), expected_tail)
    assert fast_manager.decode_sequence([12, 13, 16, 18]) == [2, 3, 4, 5, 6, 12, 14]
    assert fast_manager.segment_sequence([2, 3, 4, 5, 6]) == [12, 13]
    assert fast_manager.prepare_input_sequence([12, 14], compress=False) == [16, 18]
    assert fast_manager.prepare_input_sequence([2, 3, 12, 14]) == [12, 16, 18]
    assert fast_manager.decode_sequence(
        fast_manager.prepare_input_sequence([2, 3, 12, 14])
    ) == [2, 3, 12, 14]
    assert torch.equal(
        fast_embedding(
            torch.tensor([fast_manager.prepare_input_sequence([12, 14], compress=False)])
        ),
        torch.stack([legacy_embedding.weight[12], legacy_embedding.weight[14]]).unsqueeze(0),
    )

    batch_legacy_manager = StaticCodebookManager(
        initial_vocab_size=12,
        max_codebook_size=4,
        max_subtokens=3,
        embedding_dim=5,
        pad_token_id=0,
    )
    batch_legacy_manager.set_seeded_codebook([[2, 3], [4, 5, 6]], batch_size=2)
    batch_legacy_embedding = make_embedding(batch_legacy_manager)
    batch_legacy_head = make_linear(batch_legacy_manager)
    with torch.no_grad():
        batch_legacy_embedding.weight.copy_(legacy_embedding.weight)
        batch_legacy_head.weight.copy_(legacy_head.weight)
        batch_legacy_head.bias.copy_(legacy_head.bias)
    batch_ids = torch.tensor([[1, 12, 7, 13], [7, 13, 1, 12]])
    assert torch.equal(fast_embedding(batch_ids), batch_legacy_embedding(batch_ids))
    batch_hidden = hidden.expand(2, -1, -1)
    assert torch.allclose(
        fast_head(batch_hidden), batch_legacy_head(batch_hidden), atol=1e-6, rtol=1e-6
    )

    assert fast_manager.input_encoder_calls == 1
    assert fast_manager.output_encoder_calls == 1
    assert fast_manager.inference_tables_build_count == 1
    assert memory["effective_input_embedding_bytes"] == 19 * 5 * 4
    assert memory["effective_output_head_bytes"] == 19 * 5 * 4 + 19 * 4
    assert memory["additional_cpu_ram_bytes"] == memory[
        "effective_input_embedding_bytes"
    ] + memory["effective_output_head_bytes"]
    assert {
        "h_vector_synthesis_ms",
        "effective_input_table_build_ms",
        "effective_output_table_build_ms",
        "total_table_preparation_ms",
    } <= fast_manager.inference_timing_report.keys()
    assert all(value >= 0 for value in fast_manager.inference_timing_report.values())

    for _ in range(3):
        fast_embedding(prompt_ids)
        fast_head(hidden)
    assert fast_manager.input_encoder_calls == 1
    assert fast_manager.output_encoder_calls == 1
    assert fast_manager.inference_tables_build_count == 1

    fast_manager.reset(clear_dictionary=False)
    fast_manager.prepare_inference_tables(model, batch_size=1)
    assert fast_manager.inference_tables_build_count == 1
    fast_manager.reset(clear_caches=True)
    assert not fast_manager.fast_inference_ready
    assert fast_manager.effective_embedding_weight_cache is None
    fast_manager.prepare_inference_tables(model, batch_size=1)
    assert fast_manager.inference_tables_build_count == 2
    assert fast_manager.input_encoder_calls == 2
    assert fast_manager.output_encoder_calls == 2


def test_prepared_output_projection_supports_biasless_head():
    manager = make_manager()
    embedding = make_embedding(manager)
    head = make_linear(manager, bias=False)
    model = TinyPredictiveModel(manager, embedding, head).eval()
    manager.attach_to_model(model)
    manager.prepare_inference_tables(model)

    assert manager.effective_linear_bias_cache is None
    hidden = torch.arange(10, dtype=torch.float32).view(1, 2, 5) / 9
    actual = head(hidden)
    base_logits = F.linear(hidden, head.weight)
    h_vectors = torch.zeros(4, 5)
    h_vectors[0] = head.weight[[2, 3]].mean(0)
    h_vectors[1] = head.weight[[4, 5, 6]].mean(0)
    h_logits = hidden @ h_vectors.T
    expected = torch.cat(
        [base_logits[..., :12], h_logits, base_logits[..., 12:]], dim=-1
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_lora_merge_preserves_wrappers_outputs_and_one_position_hook(monkeypatch):
    vocab, hidden = 12, 16
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            bos_token_id=1,
            eos_token_id=vocab - 1,
            pad_token_id=0,
        )
    )
    peft_base = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        ),
    )
    manager = StaticCodebookManager(
        initial_vocab_size=vocab,
        max_codebook_size=4,
        max_subtokens=3,
        embedding_dim=hidden,
        pad_token_id=0,
    )
    config = Zip2ZipConfig(
        format_version=2,
        base_model_name_or_path="unused",
        position_mode="base_token_end",
        encoder_type="res_latent_attn",
        encoder=ResLatentAttnConfig(
            hidden_size=hidden,
            model_hidden_size=None,
            num_hidden_layers=1,
            intermediate_size=32,
            num_heads=4,
            causal=False,
            residual=True,
            tie_encoders=False,
            position_encoding=None,
        ),
        compression=CompressionConfig(
            initial_vocab_size=vocab,
            max_codebook_size=4,
            max_subtokens=3,
            disabled_ids=[0],
        ),
    )
    monkeypatch.setattr(
        CodebookManager,
        "from_config",
        classmethod(lambda cls, config: manager),
    )
    model = Zip2ZipModel(config, base_model=peft_base).eval()
    manager.set_seeded_codebook([[2, 3]])
    manager.attach_to_model(model)
    manager.prepare_inference_tables(model)
    assert manager.fast_inference_ready
    assert manager.inference_tables_build_count == 1
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(NotImplementedError, match="batch_size=1"):
        model.generate(input_ids=ids.expand(2, -1))
    assert manager.fast_inference_ready
    with pytest.raises(ValueError, match="generation arguments.eos_token_id"):
        model.generate(input_ids=ids, eos_token_id=vocab)
    assert manager.fast_inference_ready
    pre_merge_base = model.base_model
    pre_merge_hook = pre_merge_base.prepare_inputs_for_generation.__func__
    with torch.no_grad():
        before = model.base_model(input_ids=ids).logits

    prepared = prepare_model_for_inference(model, merge_lora=True)
    assert prepared is model
    assert model.base_model is not pre_merge_base
    assert not manager.fast_inference_ready
    assert manager.effective_embedding_weight_cache is None
    assert manager.effective_linear_weight_cache is None
    assert isinstance(model.base_model.get_input_embeddings(), HyperEmbedding)
    assert isinstance(model.base_model.get_output_embeddings(), HyperLinear)
    assert not model.base_model.__class__.__name__.startswith("Peft")
    with torch.no_grad():
        after = model.base_model(input_ids=ids).logits
    assert torch.allclose(before, after, atol=2e-5, rtol=2e-5)

    hooked_method = model.base_model.prepare_inputs_for_generation
    hooked_func = hooked_method.__func__
    assert getattr(hooked_func, "_zip2zip_position_hook_owner", None) == id(model)
    assert hooked_func is not pre_merge_hook
    assert model._position_hook_base_model is model.base_model
    manager.prepare_inference_tables(model)
    assert manager.fast_inference_ready
    assert manager.inference_tables_build_count == 2
    merged_input = model.base_model.get_input_embeddings().weight
    merged_output = model.base_model.get_output_embeddings().weight
    assert torch.equal(manager.effective_embedding_weight_cache[:vocab], merged_input[:vocab])
    assert torch.equal(manager.effective_linear_weight_cache[:vocab], merged_output[:vocab])
    prepare_model_for_inference(model, merge_lora=True)
    assert model.base_model.prepare_inputs_for_generation.__func__ is hooked_func
    assert manager.fast_inference_ready
    assert manager.inference_tables_build_count == 2
