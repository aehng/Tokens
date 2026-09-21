"""
Span-Aware LZW Simulator for Zip2Zip.
Tracks exact token spans to calculate prefix-stable, scientifically valid
prompt, response, and total compression percentages without boundary artifacts.
"""

from typing import Dict, List, Optional
from zip2zip_compression import LZWCompressor


def compute_lzw_span_compression(
    prompt_ids: List[int],
    response_ids: List[int],
    budget: int = 128,
    initial_vocab_size: int = 32011,
    disabled_ids: Optional[List[int]] = None,
) -> Dict:
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

    P = len(prompt_ids)
    R = len(response_ids)
    N = len(full_seq)

    pos = 0
    h_prompt = 0.0
    h_response = 0.0
    straddle_count = 0
    codebook_used = set()

    for token in encoded:
        if token < initial_vocab_size:
            subtokens = [token]
        else:
            subtokens = cb_dict[token]
            codebook_used.add(token)

        span_len = len(subtokens)
        span_start = pos
        span_end = pos + span_len
        pos = span_end

        if span_end <= P:
            h_prompt += 1.0
        elif span_start >= P:
            h_response += 1.0
        else:
            straddle_count += 1
            k_p = P - span_start
            k_r = span_end - P
            h_prompt += k_p / span_len
            h_response += k_r / span_len

    p_comp = (1.0 - h_prompt / P) * 100.0 if P > 0 else 0.0
    r_comp = (1.0 - h_response / R) * 100.0 if R > 0 else 0.0
    tot_comp = (1.0 - len(encoded) / N) * 100.0 if N > 0 else 0.0

    return {
        "base_prompt": P,
        "base_response": R,
        "encoded_total": len(encoded),
        "h_prompt": h_prompt,
        "h_response": h_response,
        "compressed_tokens": int(round(h_response)),
        "straddle_count": straddle_count,
        "prompt_comp_pct": p_comp,
        "response_comp_pct": r_comp,
        "total_comp_pct": tot_comp,
        "codebook_utilization": len(codebook_used) / budget if budget > 0 else 0.0,
        "codebook_size_used": len(codebook_used),
    }
