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
    expected_hypertoken_spans,
    expand_logical_id,
    insert_hypertoken_logits,
    logical_kind,
    logical_output_to_base_ids,
    pending_advances,
    physical_embed_id,
    reconstruct_semantic_offset,
    semantic_positions,
    spans_for_logical_ids,
    token_request_indices,
    validate_codebook,
    validate_position_mode,
)
from tokens_vllm.proof_harness import (
    MAX_ENGINE_STEPS,
    assert_lora_merged_and_unloaded,
    engine_step_decision,
    preemption_block_budget,
    preemption_cycle,
    preempted_request_ids,
)


def test_tensor_remap_matches_the_boundary_table():
    from tokens_vllm.remap import remap_logical_ids

    logical = torch.tensor([0, 32010, 32011, 32042, 32043, 32095])
    physical, is_h, slots = remap_logical_ids(logical)
    assert physical.tolist() == [0, 32010, 0, 0, 32011, 32063]
    assert is_h.tolist() == [False, False, True, True, False, False]
    assert slots[2].item() == 0
    assert slots[3].item() == 31
    assert int(physical.max()) < 32064


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
    positions = semantic_positions(
        spans, req_indices, offsets, position_mode="base_token_end"
    )
    assert positions == [10, 1, 13]
    assert offsets[4] == 10

    semantic, physical = pending_advances(
        spans, req_indices, num_reqs=10, position_mode="base_token_end"
    )
    assert semantic[4] == 4
    assert physical[4] == 2
    assert semantic[9] == 2
    assert physical[9] == 1


def test_semantic_positions_compressed_mode():
    # In compressed mode, each token advances by 1 regardless of hypertoken span.
    # Request 4: offset 10, tokens 0 then 2 -> positions 10, 11. Advance 2.
    # Request 9: offset 0, token 1 -> position 0. Advance 1.
    spans = [1, 2, 3]
    req_indices = [4, 9, 4]
    offsets = [0] * 10
    offsets[4] = 10
    positions = semantic_positions(
        spans, req_indices, offsets, position_mode="compressed"
    )
    assert positions == [10, 0, 11]
    assert offsets[4] == 10

    semantic, physical = pending_advances(
        spans, req_indices, num_reqs=10, position_mode="compressed"
    )
    assert semantic[4] == 2
    assert physical[4] == 2
    assert semantic[9] == 1
    assert physical[9] == 1


def test_resume_reconstructs_the_semantic_offset_from_history():
    spans = [1] * CODEBOOK_SIZE
    spans[3] = 3
    history = [10, H_START + 3, 32043]
    assert reconstruct_semantic_offset(history, spans, position_mode="base_token_end") == 1 + 3 + 1
    assert reconstruct_semantic_offset(history, spans, position_mode="compressed") == len(history)


def test_4_token_span_contract_and_reconstruction():
    h_spans = torch.zeros(2, CODEBOOK_SIZE, dtype=torch.long)
    h_spans[0, 5] = 4
    logical_ids = [H_START + 5]
    spans = spans_for_logical_ids(logical_ids, [0], h_spans, max_subtokens=4)
    assert spans == [4]

    row = [1] * CODEBOOK_SIZE
    row[5] = 4
    assert reconstruct_semantic_offset([H_START + 5], row, position_mode="base_token_end", max_subtokens=4) == 4
    assert reconstruct_semantic_offset([H_START + 5], row, position_mode="compressed") == 1


def test_validate_position_mode():
    assert validate_position_mode("compressed") == "compressed"
    assert validate_position_mode("base_token_end") == "base_token_end"
    try:
        validate_position_mode("invalid_mode")
    except ValueError as exc:
        assert "unknown position_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid position_mode")


def test_assert_lora_merged_and_unloaded():
    import torch.nn as nn

    # Clean model passes
    clean_model = nn.Linear(10, 10)
    assert_lora_merged_and_unloaded(clean_model)

    # PeftModel name fails
    class MockPeftModel(nn.Module):
        pass
    try:
        assert_lora_merged_and_unloaded(MockPeftModel())
    except AssertionError as exc:
        assert "PEFT model" in str(exc)
    else:
        raise AssertionError("expected AssertionError for PeftModel")

    # peft_config attribute fails
    class MockModelWithConfig(nn.Module):
        def __init__(self):
            super().__init__()
            self.peft_config = {"default": {}}
    try:
        assert_lora_merged_and_unloaded(MockModelWithConfig())
    except AssertionError as exc:
        assert "peft_config" in str(exc)
    else:
        raise AssertionError("expected AssertionError for peft_config")

    # Submodule with LoRA class name fails
    class MockLoraLinear(nn.Module):
        pass
    class MockModelWithLoraName(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = MockLoraLinear()
    try:
        assert_lora_merged_and_unloaded(MockModelWithLoraName())
    except AssertionError as exc:
        assert "LoRA layer" in str(exc)
    else:
        raise AssertionError("expected AssertionError for LoRA class name")

    # Submodule with active lora_A fails
    class MockCustomLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.Parameter(torch.randn(2, 2))
    class MockModelWithLora(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = MockCustomLayer()
    try:
        assert_lora_merged_and_unloaded(MockModelWithLora())
    except AssertionError as exc:
        assert "lora_A" in str(exc)
    else:
        raise AssertionError("expected AssertionError for active lora_A")



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


def test_disabled_h_outputs_map_back_to_stock_phi_ids():
    # Logical tail 32043..32095 is base Phi 32011..32063. H ids are not base tokens.
    logical = [10, 32010, 32043, 32095]
    base = logical_output_to_base_ids(logical)
    stock = [10, 32010, 32011, 32063]
    assert base == stock
    assert base == logical_output_to_base_ids(logical)
    try:
        logical_output_to_base_ids([32011])
    except ValueError as exc:
        assert "hypertoken" in str(exc)
    else:
        raise AssertionError("an H id must not compare as a base Phi id")


def test_expected_h_spans_follow_the_reference_phrases():
    phrases = [[slot, slot + 1] for slot in range(CODEBOOK_SIZE)]
    phrases[0] = [4, 5, 6]
    phrases[4] = [7, 8, 9, 10]
    assert expected_hypertoken_spans(phrases) == [len(phrase) for phrase in phrases]
    assert expected_hypertoken_spans(phrases)[0] == 3
    assert expected_hypertoken_spans(phrases)[4] == 4


def test_engine_step_budget_stops_an_unfinished_loop():
    assert engine_step_decision(0, MAX_ENGINE_STEPS, True) == "continue"
    assert engine_step_decision(MAX_ENGINE_STEPS - 1, MAX_ENGINE_STEPS, True) == "continue"
    assert engine_step_decision(MAX_ENGINE_STEPS, MAX_ENGINE_STEPS, True) == "budget_exceeded"
    assert engine_step_decision(3, MAX_ENGINE_STEPS, False) == "stop"


def test_preemption_budget_fits_each_prompt_and_not_the_pair():
    budget = preemption_block_budget(20, 32, block_size=16)
    assert budget["a_prompt_blocks"] == 2
    assert budget["b_prompt_blocks"] == 2
    assert budget["usable_blocks"] == 4
    assert budget["num_gpu_blocks"] == 5
    assert budget["a_full_blocks"] <= budget["usable_blocks"]
    assert budget["b_full_blocks"] <= budget["usable_blocks"]
    assert budget["combined_full_blocks"] > budget["usable_blocks"]
    assert budget["a_prompt_blocks"] + budget["b_prompt_blocks"] <= budget["usable_blocks"]
    assert budget["max_new_tokens"] == 13


def test_preemption_cycle_requires_add_remove_readd():
    events = [
        {"event": "add", "req_id": "pre-A"},
        {"event": "remove", "req_id": "pre-A"},
        {"event": "add", "req_id": "pre-B"},
        {"event": "remove", "req_id": "pre-B"},
        {"event": "add", "req_id": "pre-A"},
    ]
    assert preemption_cycle(events, "pre-A")
    assert not preemption_cycle(events, "pre-B")
    assert preempted_request_ids(events) == ["pre-A"]
