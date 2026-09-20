import pytest
import torch
import torch.nn as nn
from zip2zip import StaticCodebookManager, CompressionConfig, AttentionEncoderConfig
from zip2zip.nn.embedding import HyperEmbedding
from zip2zip.nn.linear import HyperLinear
from zip2zip.nn.encoders.attention import AttentionEncoder
from zip2zip.config import Zip2ZipConfig


def test_static_codebook_seeding_and_decoding():
    initial_vocab_size = 1000
    max_codebook_size = 128
    max_subtokens = 3
    dim = 64
    pad_id = 0

    mgr = StaticCodebookManager(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=max_codebook_size,
        max_subtokens=max_subtokens,
        embedding_dim=dim,
        pad_token_id=pad_id,
    )

    # Seed 3 hypertokens
    # H0: [51, 492] (len 2)
    # H1: [912, 18, 73] (len 3)
    # H2: [31, 605] (len 2)
    seeded = {
        0: [51, 492],
        1: [912, 18, 73],
        2: [31, 605],
    }
    mgr.set_seeded_codebook(seeded, batch_size=1)

    assert mgr.num_seeded == 3
    h0 = initial_vocab_size + 0
    h1 = initial_vocab_size + 1
    h2 = initial_vocab_size + 2

    assert mgr.hyper_to_subtokens[h0] == [51, 492]
    assert mgr.hyper_to_subtokens[h1] == [912, 18, 73]
    assert mgr.hyper_to_subtokens[h2] == [31, 605]

    # Test exact decoding
    mixed_sequence = [10, h0, 99, h1, h2, 500]
    expanded = mgr.decode_sequence(mixed_sequence)
    expected = [10, 51, 492, 99, 912, 18, 73, 31, 605, 500]
    assert expanded == expected


def test_static_codebook_spans_and_rope_positions():
    initial_vocab_size = 1000
    mgr = StaticCodebookManager(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=64,
        max_subtokens=3,
        embedding_dim=32,
        pad_token_id=0,
    )

    # H0 has 2 tokens, H1 has 3 tokens
    mgr.set_seeded_codebook([[10, 20], [30, 40, 50]], batch_size=1)

    h0 = initial_vocab_size
    h1 = initial_vocab_size + 1

    # Sequence: [base_5, H0, base_8, H1]
    # Spans should be: [1, 2, 1, 3]
    # Cumulative positions should be: [0, 0+2=2, 2+1=3, 3+3=6]
    input_ids = torch.tensor([[5, h0, 8, h1]], dtype=torch.long)
    positions = mgr.prepare_input_ids(input_ids)

    expected_positions = torch.tensor([[0, 2, 3, 6]], dtype=torch.long)
    assert torch.equal(positions, expected_positions)

    # Autoregressive generation step (single token appended: H0)
    # Next cumulative position should start from 6 + 1 + (2 - 1) = 8
    next_token = torch.tensor([[h0]], dtype=torch.long)
    next_pos = mgr.prepare_input_ids(next_token)
    assert next_pos.item() == 8


def test_static_codebook_with_hyper_embedding_and_linear():
    initial_vocab_size = 500
    max_codebook_size = 64
    max_subtokens = 3
    dim = 32
    pad_id = 0

    compression_config = CompressionConfig(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=max_codebook_size,
        max_subtokens=max_subtokens,
        disabled_ids=[0],
    )
    encoder_config = AttentionEncoderConfig(
        hidden_size=dim,
        num_heads=4,
    )
    encoder = AttentionEncoder(encoder_config, compression_config)

    mgr = StaticCodebookManager(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=max_codebook_size,
        max_subtokens=max_subtokens,
        embedding_dim=dim,
        pad_token_id=pad_id,
    )
    mgr.set_seeded_codebook([[10, 20], [30, 40, 50]], batch_size=1)

    # Mock base embedding
    base_embed = nn.Embedding(initial_vocab_size, dim)
    hyper_embed = HyperEmbedding(
        config=None,
        encoder=encoder,
        num_embeddings=initial_vocab_size,
        embedding_dim=dim,
        padding_idx=pad_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
        initial_vocab_size=initial_vocab_size,
        codebook_manager=mgr,
    )
    hyper_embed.weight = base_embed.weight

    # Test HyperEmbedding forward with both base tokens and hypertokens
    h0 = initial_vocab_size
    h1 = initial_vocab_size + 1
    input_ids = torch.tensor([[5, h0, 8, h1]], dtype=torch.long)

    out = hyper_embed(input_ids)
    assert out.shape == (1, 4, dim)
    assert not torch.isnan(out).any()

    # Mock base linear (LM head)
    base_linear = nn.Linear(dim, initial_vocab_size, bias=False)
    hyper_linear = HyperLinear(
        config=None,
        encoder=encoder,
        in_features=dim,
        out_features=initial_vocab_size,
        bias=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        initial_vocab_size=initial_vocab_size,
        codebook_manager=mgr,
    )
    hyper_linear.weight = base_linear.weight

    hidden_states = torch.randn(1, 4, dim)
    logits = hyper_linear(hidden_states)
    # Logits shape should be: (batch, seq, initial_vocab_size + max_codebook_size)
    assert logits.shape == (1, 4, initial_vocab_size + max_codebook_size)

    # Test logits masking for unseeded slots
    # Slots 0 and 1 are seeded. Slots 2..63 are unseeded.
    warped_logits = mgr.mask_unused_logits(logits.clone())
    unseeded_start = initial_vocab_size + 2
    unseeded_end = initial_vocab_size + max_codebook_size

    assert torch.isneginf(warped_logits[..., unseeded_start:unseeded_end]).all()
    # Seeded slots should NOT be -inf
    assert not torch.isneginf(warped_logits[..., initial_vocab_size : unseeded_start]).any()


def test_static_codebook_reset():
    mgr = StaticCodebookManager(
        initial_vocab_size=100,
        max_codebook_size=32,
        max_subtokens=3,
        embedding_dim=16,
        pad_token_id=0,
    )
    mgr.set_seeded_codebook([[1, 2], [3, 4]])
    assert mgr.num_seeded == 2

    # Normal reset (between steps/requests) preserves seeded dictionary
    mgr.prepare_input_ids(torch.tensor([[100]]))
    mgr.reset(clear_dictionary=False)
    assert mgr.num_seeded == 2
    assert mgr.base_position_offset is None

    # Full reset clears dictionary
    mgr.reset(clear_dictionary=True)
    assert mgr.num_seeded == 0
    assert len(mgr.hyper_to_subtokens) == 0
