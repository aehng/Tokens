import pytest
from zip2zip.segmenter import DynamicSegmenter, segment_tokens_with_dictionary
from zip2zip import StaticCodebookManager


def test_user_spec_example():
    """Test the exact example from the user prompt:
    Base sequence: [12, 44, 98, 61, 72, 91]
    Dictionary: H0 = [12, 44], H1 = [61, 72, 91]
    Output: [H0, 98, H1]
    """
    h0 = 32000
    h1 = 32001
    dict_map = {
        (12, 44): h0,
        (61, 72, 91): h1,
    }
    seq = [12, 44, 98, 61, 72, 91]
    segmented = segment_tokens_with_dictionary(seq, dict_map)
    assert segmented == [h0, 98, h1]


def test_token_count_minimization_preference():
    """DP should choose longer matching hypertoken over shorter ones to minimize token count."""
    h_short = 32000  # [1, 2]
    h_long = 32001   # [1, 2, 3]

    dict_map = {
        (1, 2): h_short,
        (1, 2, 3): h_long,
    }

    # Sequence [1, 2, 3] could be [h_short, 3] (len 2) or [h_long] (len 1)
    seq = [1, 2, 3]
    segmented = segment_tokens_with_dictionary(seq, dict_map)
    assert segmented == [h_long]


def test_overlapping_patterns_optimality():
    """DP finds the globally optimal segmentation when candidates overlap."""
    h0 = 32000  # [1, 2] -> saves 1 token
    h1 = 32001  # [2, 3, 4] -> saves 2 tokens

    dict_map = {
        (1, 2): h0,
        (2, 3, 4): h1,
    }

    # Sequence [1, 2, 3, 4]
    # Choice A: H0 at [0:2] -> [H0, 3, 4] (len 3)
    # Choice B: H1 at [1:4] -> [1, H1] (len 2)
    # Optimal: Choice B
    seq = [1, 2, 3, 4]
    segmented = segment_tokens_with_dictionary(seq, dict_map)
    assert segmented == [1, h1]


def test_special_token_boundary_preservation():
    """Segmenter must never merge across or include special/disabled tokens."""
    sep_id = 999  # special token
    h0 = 32000    # [10, 20]
    h1 = 32001    # [20, 30]

    dict_map = {
        (10, 20): h0,
        (20, 30): h1,
    }

    # Sequence with special token between 10 and 20: [10, sep_id, 20]
    seq = [10, sep_id, 20]
    segmented = segment_tokens_with_dictionary(seq, dict_map, disabled_ids={sep_id})
    assert segmented == [10, sep_id, 20]

    # Sequence where 20 is separated from 30: [10, 20, sep_id, 20, 30]
    # Expected: [h0, sep_id, h1]
    seq2 = [10, 20, sep_id, 20, 30]
    segmented2 = segment_tokens_with_dictionary(seq2, dict_map, disabled_ids={sep_id})
    assert segmented2 == [h0, sep_id, h1]


def test_roundtrip_invertibility():
    """Verify exact lossless reconstruction for complex token sequences."""
    mgr = StaticCodebookManager(
        initial_vocab_size=1000,
        max_codebook_size=64,
        max_subtokens=3,
        embedding_dim=32,
        pad_token_id=0,
        disabled_ids=[0, 999],
    )

    seeded = [
        [10, 20],        # H0
        [30, 40, 50],    # H1
        [60, 70],        # H2
        [80, 90, 100],   # H3
    ]
    mgr.set_seeded_codebook(seeded)

    original = [
        5, 10, 20, 7, 30, 40, 50, 999, 60, 70, 80, 90, 100, 10, 20, 15
    ]
    compressed = mgr.segment_sequence(original)

    # Check that compression actually occurred
    assert len(compressed) < len(original)
    # Check that special token 999 is preserved
    assert 999 in compressed

    # Lossless roundtrip decode
    recovered = mgr.decode_sequence(compressed)
    assert recovered == original
