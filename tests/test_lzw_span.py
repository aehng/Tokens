"""Verification of Span-Based LZW Response Attribution against actual decompression."""

import pytest
from zip2zip_compression import LZWCompressor


def compute_lzw_span_compression(
    prompt_ids: list[int],
    response_ids: list[int],
    budget: int = 128,
    initial_vocab_size: int = 32011,
    disabled_ids: list[int] = None,
):
    disabled = disabled_ids or [0, 1]
    compressor = LZWCompressor(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=budget,
        max_subtokens=4,
        pad_token_id=0,
        disabled_ids=disabled,
    )
    full_seq = prompt_ids + response_ids
    encoded, _, codebook = compressor.encode(full_seq)
    cb_dict = codebook.to_dict()

    # Decompress and verify identity with full_seq
    decompressed_tuple = compressor.decode(encoded)
    decompressed_ids = decompressed_tuple[0]
    assert decompressed_ids == full_seq, "Decompression must exactly match original sequence"

    # Track token spans
    P = len(prompt_ids)
    R = len(response_ids)
    N = len(full_seq)

    pos = 0
    h_prompt = 0.0
    h_response = 0.0
    straddle_count = 0

    for token in encoded:
        if token < initial_vocab_size:
            subtokens = [token]
        else:
            subtokens = cb_dict[token]

        span_len = len(subtokens)
        span_start = pos
        span_end = pos + span_len
        pos = span_end

        if span_end <= P:
            # Entirely in prompt
            h_prompt += 1.0
        elif span_start >= P:
            # Entirely in response
            h_response += 1.0
        else:
            # Straddles boundary
            straddle_count += 1
            k_p = P - span_start
            k_r = span_end - P
            h_prompt += k_p / span_len
            h_response += k_r / span_len

    assert pos == N, f"Final position {pos} must equal total sequence length {N}"
    assert abs((h_prompt + h_response) - len(encoded)) < 1e-6, "Sum of attributed tokens must equal total encoded tokens"

    p_comp = (1.0 - h_prompt / P) * 100.0 if P > 0 else 0.0
    r_comp = (1.0 - h_response / R) * 100.0 if R > 0 else 0.0
    tot_comp = (1.0 - len(encoded) / N) * 100.0 if N > 0 else 0.0

    return {
        "base_prompt": P,
        "base_response": R,
        "encoded_total": len(encoded),
        "h_prompt": h_prompt,
        "h_response": h_response,
        "straddle_count": straddle_count,
        "prompt_comp_pct": p_comp,
        "response_comp_pct": r_comp,
        "total_comp_pct": tot_comp,
    }


def test_hand_checkable_lzw_no_straddle():
    """Prompt has repeating phrase, response has another repeating phrase."""
    # Pattern: [10, 20] repeated in prompt, [30, 40] repeated in response
    prompt = [10, 20, 10, 20, 10, 20]  # len = 6
    response = [30, 40, 30, 40, 30, 40]  # len = 6
    res = compute_lzw_span_compression(prompt, response, budget=32)

    assert res["straddle_count"] == 0
    assert res["h_prompt"] < 6, "Prompt must be compressed"
    assert res["h_response"] < 6, "Response must be compressed"
    assert abs(res["h_prompt"] + res["h_response"] - res["encoded_total"]) < 1e-6


def test_hand_checkable_lzw_with_straddle():
    """Pattern crosses boundary between prompt and response."""
    # Pattern [50, 60] occurs in prompt, and again at the boundary: [..., 50] [60, ...]
    prompt = [50, 60, 100, 50]  # ends with 50 (len=4)
    response = [60, 200, 50, 60]  # starts with 60 (len=4)
    res = compute_lzw_span_compression(prompt, response, budget=32)

    # Total encoded tokens should equal h_prompt + h_response
    assert abs(res["h_prompt"] + res["h_response"] - res["encoded_total"]) < 1e-6
    # Total compression should equal exact formula
    expected_tot = (1.0 - res["encoded_total"] / 8) * 100.0
    assert abs(res["total_comp_pct"] - expected_tot) < 1e-6


def test_unpacking_compressor_decode():
    """Verify compressor.decode return type unpacking."""
    comp = LZWCompressor(
        initial_vocab_size=32011,
        max_codebook_size=32,
        max_subtokens=4,
        pad_token_id=0,
        disabled_ids=[0, 1],
    )
    seq = [10, 20, 30, 10, 20, 30]
    enc, _, _ = comp.encode(seq)
    res = comp.decode(enc)
    assert isinstance(res, tuple), "decode() must return a tuple"
    assert len(res) == 2, "tuple length must be 2: (base_tokens, codebook)"
    base_tokens, cb = res
    assert base_tokens == seq, "Unpacked base_tokens must match original"
