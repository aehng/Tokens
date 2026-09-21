"""
Span-Aware LZW Simulator for Zip2Zip.
Configured directly from the official checkpoint compression parameters.
Supports:
1. Standard reactive LZW with exact span attribution.
2. Apples-to-apples constraint matching (e.g. max_subtokens=3 vs max_subtokens=4).
3. True sequential pre-seeded hybrid LZW simulation.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple
from zip2zip_compression import LZWCompressor
from zip2zip.config import Zip2ZipConfig

# Official checkpoint defaults loaded dynamically
_DEFAULT_CHECKPOINT = "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1"
try:
    _cfg = Zip2ZipConfig.from_pretrained(_DEFAULT_CHECKPOINT)
    OFFICIAL_INITIAL_VOCAB_SIZE = _cfg.compression.initial_vocab_size  # 32011
    OFFICIAL_DISABLED_IDS = list(_cfg.compression.disabled_ids)  # [0, 1, 2, 32000..32010]
    OFFICIAL_MAX_SUBTOKENS = _cfg.compression.max_subtokens  # 4
except Exception:
    OFFICIAL_INITIAL_VOCAB_SIZE = 32011
    OFFICIAL_DISABLED_IDS = [0, 1, 2] + list(range(32000, 32011))
    OFFICIAL_MAX_SUBTOKENS = 4


def compute_lzw_span_compression(
    prompt_ids: List[int],
    response_ids: List[int],
    budget: int = 128,
    initial_vocab_size: int = OFFICIAL_INITIAL_VOCAB_SIZE,
    max_subtokens: int = 3,  # Strict apples-to-apples default matching predictive codebook (2-3 tokens)
    disabled_ids: Optional[List[int]] = None,
    pad_token_id: int = 32000,
) -> Dict:
    """
    Computes exact span-attributed LZW compression matching checkpoint configuration.
    """
    disabled = disabled_ids if disabled_ids is not None else OFFICIAL_DISABLED_IDS
    compressor = LZWCompressor(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=budget,
        max_subtokens=max_subtokens,
        pad_token_id=pad_token_id,
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


def compute_preseeded_hybrid_lzw(
    prompt_ids: List[int],
    response_ids: List[int],
    total_budget: int = 128,
    preseeded_phrases: Optional[Sequence[Tuple[int, ...]]] = None,
    reactive_budget: int = 0,
    max_subtokens: int = 3,
    initial_vocab_size: int = OFFICIAL_INITIAL_VOCAB_SIZE,
    disabled_ids: Optional[List[int]] = None,
) -> Dict:
    """
    True sequential simulation of a hybrid codebook:
    1. Pre-seeds N static dictionary entries from step 0.
    2. Runs sequential LZW greedy parsing on (prompt + response).
    3. Dynamically learns up to reactive_budget new reactive entries from observed context.
    4. Attributes compressed spans to prompt vs. response.
    """
    disabled = set(disabled_ids if disabled_ids is not None else OFFICIAL_DISABLED_IDS)
    preseeded = preseeded_phrases or []

    # Initialize dictionary
    vocab = {}  # tuple -> id
    reverse_vocab = {}  # id -> tuple
    next_id = initial_vocab_size

    for phrase in preseeded:
        p_tuple = tuple(phrase)
        if len(p_tuple) <= max_subtokens and p_tuple not in vocab:
            vocab[p_tuple] = next_id
            reverse_vocab[next_id] = p_tuple
            next_id += 1

    preseeded_count = len(vocab)
    max_id = next_id + reactive_budget

    full_seq = prompt_ids + response_ids
    encoded = []
    w = []

    for c in full_seq:
        wc = tuple(w + [c])
        is_in_vocab = (len(wc) == 1) or (wc in vocab)

        if is_in_vocab:
            w.append(c)
        else:
            # Emit w
            if len(w) == 1:
                encoded.append(w[0])
            else:
                encoded.append(vocab[tuple(w)])

            # Dynamically learn wc if capacity permits
            if next_id < max_id and len(wc) <= max_subtokens:
                if not any(t in disabled for t in wc):
                    vocab[wc] = next_id
                    reverse_vocab[next_id] = wc
                    next_id += 1
            w = [c]

    if w:
        if len(w) == 1:
            encoded.append(w[0])
        else:
            encoded.append(vocab[tuple(w)])

    # Span attribution
    P = len(prompt_ids)
    R = len(response_ids)
    N = len(full_seq)

    pos = 0
    h_prompt = 0.0
    h_response = 0.0
    used_hypertokens = set()

    for token in encoded:
        if token < initial_vocab_size:
            subtokens = [token]
        else:
            subtokens = list(reverse_vocab[token])
            used_hypertokens.add(token)

        span_len = len(subtokens)
        span_start = pos
        span_end = pos + span_len
        pos = span_end

        if span_end <= P:
            h_prompt += 1.0
        elif span_start >= P:
            h_response += 1.0
        else:
            k_p = P - span_start
            k_r = span_end - P
            h_prompt += k_p / span_len
            h_response += k_r / span_len

    r_comp = (1.0 - h_response / R) * 100.0 if R > 0 else 0.0
    tot_comp = (1.0 - len(encoded) / N) * 100.0 if N > 0 else 0.0

    return {
        "preseeded_count": preseeded_count,
        "reactive_learned": next_id - initial_vocab_size - preseeded_count,
        "total_codebook_size": next_id - initial_vocab_size,
        "encoded_total": len(encoded),
        "h_response": h_response,
        "response_comp_pct": r_comp,
        "total_comp_pct": tot_comp,
        "codebook_utilization": len(used_hypertokens) / (next_id - initial_vocab_size) if (next_id > initial_vocab_size) else 0.0,
    }
