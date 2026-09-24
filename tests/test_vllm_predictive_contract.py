"""CPU checks for the logical vocabulary and semantic-position contract."""

import torch

from tokens_vllm.contract import (
    BASE_VOCAB_SIZE,
    CODEBOOK_SIZE,
    H_END,
    H_START,
    INITIAL_VOCAB_SIZE,
    LOGICAL_VOCAB_SIZE,
    codebook_sha256,
    expand_logical_id,
    insert_hypertoken_logits,
    logical_kind,
    pending_advances,
    physical_embed_id,
    reconstruct_semantic_offset,
    semantic_positions,
    spans_for_logical_ids,
    token_request_indices,
    validate_codebook,
)


def test_boundary_ids_match_the_insertion_layout():
    cases = {
        0: ("base", 0),
        32010: ("base", 32010),
        32011: ("hypertoken", 0),
        32042: ("hypertoken", 0),
        32043: ("shifted_tail", 32011),
        32095: ("shifted_tail", 32063),
    }
    for logical_id, (kind, physical) in cases.items():
        assert logical_kind(logical_id) == kind
        assert physical_embed_id(logical_id) == physical
        assert physical < BASE_VOCAB_SIZE


def test_hypertoken_ids_never_reach_the_physical_table():
    for logical_id in range(H_START, H_END):
        assert physical_embed_id(logical_id) == 0
    for logical_id in range(LOGICAL_VOCAB_SIZE):
        assert physical_embed_id(logical_id) < BASE_VOCAB_SIZE


def test_tail_shift_round_trips_through_expansion():
    phrases = [(10, 11)] * CODEBOOK_SIZE
    assert expand_logical_id(32010, phrases) == [32010]
    assert expand_logical_id(32011, phrases) == [10, 11]
    assert expand_logical_id(32043, phrases) == [32011]
    assert expand_logical_id(32095, phrases) == [32063]
    assert 32095 - CODEBOOK_SIZE == BASE_VOCAB_SIZE - 1


def test_logit_insertion_matches_hyperlinear_order():
    base = torch.arange(BASE_VOCAB_SIZE, dtype=torch.float32).view(1, -1)
    h_logits = torch.arange(1000, 1000 + CODEBOOK_SIZE, dtype=torch.float32).view(1, -1)
    merged = insert_hypertoken_logits(base, h_logits)

    assert merged.shape == (1, LOGICAL_VOCAB_SIZE)
    assert torch.equal(merged[0, :INITIAL_VOCAB_SIZE], base[0, :INITIAL_VOCAB_SIZE])
    assert torch.equal(merged[0, H_START:H_END], h_logits[0])
    assert torch.equal(merged[0, H_END:], base[0, INITIAL_VOCAB_SIZE:])


def test_same_hypertoken_id_keeps_separate_request_owners():
    # Batch order is B then A. Slot ids stay with the request, not the row.
    idx_mapping = [7, 4]
    query_start_loc = [0, 1, 2]
    assert token_request_indices(idx_mapping, query_start_loc) == [7, 4]

    h_spans = torch.zeros(8, CODEBOOK_SIZE, dtype=torch.long)
    h_spans[4, 5] = 2
    h_spans[7, 5] = 3
    logical_ids = [H_START + 5, H_START + 5]
    spans = spans_for_logical_ids(logical_ids, [7, 4], h_spans)
    assert spans == [3, 2]


def test_semantic_positions_use_per_request_offsets_and_do_not_commit():
    # Request 4: offset 10, spans 1 then 3 -> positions 10, 13. Advance 4.
    # Request 9: offset 0, span 2 -> position 1. Advance 2.
    spans = [1, 2, 3]
    req_indices = [4, 9, 4]
    offsets = [0] * 10
    offsets[4] = 10
    positions = semantic_positions(spans, req_indices, offsets)
    assert positions == [10, 1, 13]
    assert offsets[4] == 10

    semantic, physical = pending_advances(spans, req_indices, num_reqs=10)
    assert semantic[4] == 4
    assert physical[4] == 2
    assert semantic[9] == 2
    assert physical[9] == 1


def test_resume_reconstructs_the_semantic_offset_from_history():
    spans = [1] * CODEBOOK_SIZE
    spans[3] = 3
    history = [10, H_START + 3, 32043]
    assert reconstruct_semantic_offset(history, spans) == 1 + 3 + 1


def test_codebook_rejects_bad_phrases_and_checks_the_hash():
    phrases = [[slot, slot + 1] for slot in range(CODEBOOK_SIZE)]
    phrases[0] = [4, 5, 6]
    payload = {
        "version": 1,
        "k": CODEBOOK_SIZE,
        "phrases": phrases,
        "sha256": codebook_sha256(phrases),
    }
    codebook = validate_codebook(payload, disabled_ids={99})
    assert codebook.spans[0] == 3
    assert codebook.spans[1] == 2

    broken = dict(payload)
    broken["phrases"] = [list(phrase) for phrase in phrases]
    broken["phrases"][2] = [1]
    try:
        validate_codebook(broken)
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("expected a short phrase to be rejected")

    disabled = dict(payload)
    disabled["phrases"] = [list(phrase) for phrase in phrases]
    disabled["phrases"][1] = [99, 100]
    disabled["sha256"] = codebook_sha256(disabled["phrases"])
    try:
        validate_codebook(disabled, disabled_ids={99})
    except ValueError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("expected a disabled id to be rejected")
