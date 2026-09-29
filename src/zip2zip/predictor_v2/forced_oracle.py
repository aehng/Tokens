"""Model-free forced H substitution and exact expansion for CF fixtures."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from src.evaluation.offline_segmenter import segment_tokens_dp


def force_oracle_substitutions(
    base_token_ids: Sequence[int],
    phrase_to_h_id: Mapping[Sequence[int], int],
) -> Dict[str, object]:
    """DP-tile a known sequence, substitute only exact supplied phrases, then round-trip.

    This validates the forced-token schedule and expansion independently of a
    model. It does not claim live representation or continuation-state safety.
    """
    normalized = {tuple(int(token) for token in phrase): int(h_id) for phrase, h_id in phrase_to_h_id.items()}
    if len(normalized) != len(phrase_to_h_id):
        raise ValueError("Duplicate phrase after integer normalization")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("Each phrase must have a unique H ID")
    for phrase, h_id in normalized.items():
        if len(phrase) < 2 or h_id < 0:
            raise ValueError("Forced phrases must span at least two base tokens and use nonnegative H IDs")

    _, tiles, dp_stats = segment_tokens_dp(list(base_token_ids), set(normalized))
    compressed: List[int] = []
    emissions: List[Dict[str, object]] = []
    expanded: List[int] = []
    cursor = 0
    for tile in tiles:
        phrase = tuple(int(token) for token in tile)
        if len(phrase) > 1:
            h_id = normalized.get(phrase)
            if h_id is None:
                raise AssertionError(f"DP emitted an unavailable phrase tile: {phrase}")
            schedule_index = len(compressed)
            compressed.append(h_id)
            emissions.append({
                "id": h_id,
                "subtokens": list(phrase),
                "base_start": cursor,
                "span": len(phrase),
                "schedule_index": schedule_index,
            })
            expanded.extend(phrase)
        else:
            compressed.append(phrase[0])
            expanded.append(phrase[0])
        cursor += len(phrase)

    expected = [int(token) for token in base_token_ids]
    if expanded != expected:
        raise AssertionError("Forced H expansion did not recreate the intended base-token sequence")
    return {
        "base_token_ids": expected,
        "forced_generation_ids": compressed,
        "expanded_token_ids": expanded,
        "h_emissions": emissions,
        "base_token_count": len(expected),
        "forced_decode_calls": len(compressed),
        "tokens_saved": len(expected) - len(compressed),
        "dp_stats": dp_stats,
    }
