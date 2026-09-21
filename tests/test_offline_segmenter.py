import pytest
from src.evaluation.offline_segmenter import segment_tokens_dp, compute_oracle_codebook


def test_segment_tokens_dp_empty():
    compressed_len, tiles, stats = segment_tokens_dp([], set())
    assert compressed_len == 0
    assert tiles == []
    assert stats["base_tokens"] == 0
    assert stats["compressed_tokens"] == 0


def test_segment_tokens_dp_no_codebook():
    tokens = [101, 102, 103, 104, 105]
    compressed_len, tiles, stats = segment_tokens_dp(tokens, set())
    assert compressed_len == 5
    assert len(tiles) == 5
    assert all(len(t) == 1 for t in tiles)
    assert stats["tokens_saved"] == 0
    assert stats["compression_pct"] == 0.0


def test_segment_tokens_dp_length2_and_length3():
    # tokens: [1, 2, 3, 4, 5, 6, 7]
    # codebook: (2, 3) and (4, 5, 6)
    tokens = [1, 2, 3, 4, 5, 6, 7]
    codebook = {(2, 3), (4, 5, 6)}
    compressed_len, tiles, stats = segment_tokens_dp(tokens, codebook)

    # Expected tiling: (1,), (2, 3), (4, 5, 6), (7,) -> 4 tiles
    assert compressed_len == 4
    assert tiles == [(1,), (2, 3), (4, 5, 6), (7,)]
    assert stats["base_tokens"] == 7
    assert stats["compressed_tokens"] == 4
    assert stats["tokens_saved"] == 3
    assert stats["hypertoken_emissions"] == 2
    assert stats["unique_hypertokens_used"] == 2
    assert stats["codebook_utilization"] == 1.0


def test_segment_tokens_dp_optimal_choice_over_suboptimal():
    # tokens: [10, 20, 30]
    # codebook has both (10, 20) and (10, 20, 30)
    tokens = [10, 20, 30]
    codebook = {(10, 20), (10, 20, 30)}
    compressed_len, tiles, stats = segment_tokens_dp(tokens, codebook)

    # (10, 20, 30) gives 1 token, while (10, 20) + (30,) gives 2 tokens
    assert compressed_len == 1
    assert tiles == [(10, 20, 30)]
    assert stats["tokens_saved"] == 2


def test_segment_tokens_dp_reconstruction_integrity():
    tokens = [5, 12, 19, 23, 5, 12, 19, 44, 99, 100, 101, 102]
    codebook = {(5, 12), (5, 12, 19), (99, 100)}
    compressed_len, tiles, stats = segment_tokens_dp(tokens, codebook)

    # Flatten tiles and verify exact match
    flattened = [tok for t in tiles for tok in t]
    assert flattened == tokens
    assert compressed_len == len(tiles)
    assert stats["base_tokens"] == len(tokens)
    assert stats["compressed_tokens"] == len(tiles)


def test_compute_oracle_codebook():
    # Repeated pattern [10, 20, 30] occurs 4 times
    tokens = [10, 20, 30, 99, 10, 20, 30, 88, 10, 20, 30, 77, 10, 20, 30]
    oracle_codebook = compute_oracle_codebook(tokens, k=2)
    assert (10, 20, 30) in oracle_codebook or (10, 20) in oracle_codebook
    compressed_len, _, stats = segment_tokens_dp(tokens, oracle_codebook)
    assert stats["tokens_saved"] > 0
