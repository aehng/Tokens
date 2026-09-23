import random

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from zip2zip import StaticCodebookManager
from zip2zip.nn.embedding import HyperEmbedding
from zip2zip.nn.linear import HyperLinear


class MeanTokenEncoder:
    """Small deterministic stand-in for the production encoder contract."""

    def get_encoder_fn(self):
        def encode(updates, base_weight, pad_token_id):
            valid = updates.ne(pad_token_id)
            embedded = F.embedding(updates, base_weight)
            summed = (embedded * valid.unsqueeze(-1)).sum(dim=-2)
            return summed / valid.sum(dim=-1, keepdim=True).clamp_min(1)

        return encode


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
