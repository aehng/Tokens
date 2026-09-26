"""CPU checks for the logical vocabulary and semantic-position contract."""

import hashlib
from types import SimpleNamespace

import torch

from tokens_vllm.contract import (
    BASE_VOCAB_SIZE,
    CODEBOOK_SIZE,
    H_END,
    H_START,
    INITIAL_VOCAB_SIZE,
    LOGICAL_VOCAB_SIZE,
    chunk_positions_from_computed,
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
    blocks_for_tokens,
    engine_step_decision,
    max_scheduler_preemptions,
    preemption_block_budget,
    preemption_cycle,
    preemption_rebuild_report,
    preempted_request_ids,
    position_values_for_stage,
    request_prefill_trace,
    resume_position_contract_report,
    semantic_kv_rope_contract_report,
    two_codebook_isolation_passes,
)
from tokens_vllm.state import RequestScopedPositionTrace, write_semantic_positions


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


def _unit_spans():
    return [1] * CODEBOOK_SIZE


def test_compressed_chunks_follow_num_computed_without_postprocess_state():
    # Phase 8 shape: 16 + 16 + 10. A shadow offset left at 0 would restart
    # every chunk. The computed-token count is the origin instead.
    logical = list(range(42))
    got: list[int] = []
    for origin, length in ((0, 16), (16, 16), (32, 10)):
        chunk = logical[origin : origin + length]
        got.extend(
            chunk_positions_from_computed(
                chunk,
                [4] * length,
                {4: origin},
                position_mode="compressed",
            )
        )
    assert got == list(range(42))


def test_request_scoped_position_trace_resets_and_rejects_other_requests():
    trace = RequestScopedPositionTrace()
    trace.begin("old-request")
    old = trace.capture({"req_ids": ["old-request"]})
    assert old is not None

    trace.begin("phase8-chunk")
    assert trace.records == []
    assert trace.active_record is None
    assert trace.capture({"req_ids": ["warmup-request"]}) is None
    target = trace.capture({"req_ids": ["phase8-chunk"]})
    assert target is not None
    assert target["target_request_id"] == "phase8-chunk"
    assert target["trace_id"] == "phase8-chunk:1"
    assert trace.records == [target]


def test_request_scoped_position_trace_captures_both_potential_victims():
    trace = RequestScopedPositionTrace()
    trace.begin(("pre-A", "pre-B"))
    captured = trace.capture(
        {
            "req_ids": ["pre-A", "pre-B"],
            "request_snapshots": {
                "pre-A": {"slot": 1, "target_query_slice": [0, 2]},
                "pre-B": {"slot": 3, "target_query_slice": [2, 5]},
            },
        }
    )
    assert captured["target_request_id"] == "pre-A"
    assert trace.active_record is None
    assert [record["target_request_id"] for record in trace.active_records] == [
        "pre-A",
        "pre-B",
    ]
    assert [record["slot"] for record in trace.records] == [1, 3]


def test_semantic_model_positions_do_not_mutate_physical_kv_positions():
    physical_kv_positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    physical_before = physical_kv_positions.clone()
    model_position_buffer = torch.tensor([90, 91, 92, 93], dtype=torch.int64)
    stale_default_positions = model_position_buffer.tolist()
    semantic_positions = torch.tensor([12, 13, 14], dtype=torch.int64)

    model_positions = write_semantic_positions(
        model_position_buffer,
        semantic_positions,
        num_tokens=3,
        num_tokens_after_padding=4,
    )

    assert model_positions.tolist() == [12, 13, 14, 0]
    assert model_positions.tolist()[:3] != stale_default_positions[:3]
    assert physical_kv_positions.tolist() == physical_before.tolist()

    bte_buffer = torch.full((3,), -1, dtype=torch.int64)
    bte_positions = write_semantic_positions(
        bte_buffer,
        torch.tensor([2, 5, 8], dtype=torch.int64),
        num_tokens=3,
        num_tokens_after_padding=3,
    )
    assert bte_positions.tolist() == [2, 5, 8]
    assert physical_kv_positions.tolist() == physical_before.tolist()


def test_request_prefill_trace_and_stage_values_ignore_unrelated_and_decode_events():
    positions = list(range(16))
    unrelated = {
        "target_request_id": "old-request",
        "requests": [{"request_id": "old-request", "num_computed_tokens": 0}],
        "handoff": {"F": [{"target_values": [700]}]},
    }
    first = {
        "target_request_id": "phase8-chunk",
        "requests": [{"request_id": "phase8-chunk", "num_computed_tokens": 0}],
        "handoff": {"F": [{"target_values": positions}]},
    }
    second = {
        "target_request_id": "phase8-chunk",
        "requests": [{"request_id": "phase8-chunk", "num_computed_tokens": 16}],
        "handoff": {"F": [{"target_values": list(range(16, 32))}]},
    }
    decode = {
        "target_request_id": "phase8-chunk",
        "requests": [{"request_id": "phase8-chunk", "num_computed_tokens": 42}],
        "handoff": {"F": [{"target_values": [42]}]},
    }

    prefill = request_prefill_trace(
        [unrelated, first, second, decode], "phase8-chunk", prefill_token_count=42
    )

    assert prefill == [first, second]
    assert position_values_for_stage(prefill, "F") == list(range(32))


def test_fake_model_handoff_passes_semantic_override_through_layer0_rope():
    from experiments.kaggle.run_vllm_predictive_proof import (
        _consume_debug,
        _install_debug_hooks,
    )

    class Rotary(torch.nn.Module):
        def forward(self, positions, query, key):
            return query, key

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = Rotary()

        def forward(self, positions):
            self.rotary_emb(positions, torch.zeros(1), torch.zeros(1))
            return positions

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

        def forward(self, positions):
            return self.self_attn(positions)

    class LlamaModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

        def __call__(self, *args, **kwargs):
            # Match vLLM 0.30's eager support_torch_compile path, which calls
            # self.forward directly and therefore bypasses Module call hooks.
            return self.forward(*args, **kwargs)

        def forward(self, input_ids, positions):
            del input_ids
            return self.layers[0](positions).unsqueeze(0)

    class PredictiveModel(torch.nn.Module):
        def __init__(self, state):
            super().__init__()
            self.model = LlamaModel()
            self.predictive_state = state

        def compute_logits(self, hidden):
            return hidden

        def forward(self, input_ids, positions):
            return self.compute_logits(self.model(input_ids, positions))

    trace = RequestScopedPositionTrace()
    state = SimpleNamespace(
        position_trace=trace,
        admission_log=[],
        setup_ms={},
    )
    model = PredictiveModel(state)
    physical_positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    original_physical = physical_positions.clone()
    model_position_buffer = torch.tensor([90, 91, 92], dtype=torch.int64)
    semantic_positions = torch.tensor([12, 13, 14], dtype=torch.int64)

    _install_debug_hooks(model, "phase8-chunk")
    module_hook_calls = []
    skipped_module_hook = model.model.register_forward_pre_hook(
        lambda *_args: module_hook_calls.append(True)
    )
    returned_positions = write_semantic_positions(
        model_position_buffer,
        semantic_positions,
        num_tokens=3,
        num_tokens_after_padding=3,
    )
    trace.capture(
        {
            "req_ids": ["phase8-chunk"],
            "requests": [{"request_id": "phase8-chunk", "num_computed_tokens": 0}],
            "target_query_slice": [0, 3],
            "stock_input_batch_positions": {"target_values": physical_positions.tolist()},
            "calculated_semantic_positions": {"target_values": semantic_positions.tolist()},
            "returned_positions": {"target_values": returned_positions.tolist()},
        }
    )
    model_inputs = {"input_ids": torch.tensor([1, 2, 3]), "positions": physical_positions}
    model_inputs.update({"positions": returned_positions})
    model(**model_inputs)
    captured = _consume_debug(model)

    assert physical_positions.tolist() == original_physical.tolist()
    assert position_values_for_stage(captured["position_trace"], "A") == [0, 1, 2]
    assert position_values_for_stage(captured["position_trace"], "B") == [12, 13, 14]
    assert position_values_for_stage(captured["position_trace"], "C") == [12, 13, 14]
    assert captured["position_trace"][0]["handoff"]["D"][0]["target_values"] == [12, 13, 14]
    assert captured["position_trace"][0]["handoff"]["E"][0]["target_values"] == [12, 13, 14]
    assert captured["position_trace"][0]["handoff"]["F"][0]["target_values"] == [12, 13, 14]
    assert module_hook_calls == []
    skipped_module_hook.remove()
    assert "forward" not in vars(model.model)
    assert (
        captured["position_trace"][0]["handoff"]["E"][0]["data_ptr"]
        == returned_positions.data_ptr()
    )


def test_phase8_report_requires_request_scoped_chunks_and_all_handoff_stages():
    from experiments.kaggle.run_vllm_predictive_proof import _phase8_report

    prompt_positions = list(range(42))
    events = []
    for index, (computed, length) in enumerate(((0, 16), (16, 16), (32, 10))):
        chunk_positions = prompt_positions[computed : computed + length]
        snapshots = {
            "stock_input_batch_positions": {"target_values": chunk_positions},
            "calculated_semantic_positions": {"target_values": chunk_positions},
            "returned_positions": {"target_values": chunk_positions},
            "handoff": {
                stage: [{"target_values": chunk_positions}]
                for stage in ("D", "E", "F")
            },
        }
        events.append(
            {
                "target_request_id": "phase8-chunk",
                "requests": [
                    {
                        "request_id": "phase8-chunk",
                        "num_computed_tokens": computed,
                        "num_scheduled_tokens": length,
                    }
                ],
                **snapshots,
                "trace_id": f"phase8-chunk:{index + 1}",
            }
        )
    events.append(
        {
            "target_request_id": "phase8-chunk",
            "requests": [{"request_id": "phase8-chunk", "num_computed_tokens": 42}],
            "handoff": {"F": [{"target_values": [42]}]},
        }
    )

    report = _phase8_report(
        {"token_ids": [99, 100], "position_trace": events},
        {
            "compressed_ids": list(range(42)),
            "positions": prompt_positions,
            "new_ids": [99, 100],
        },
    )

    assert report["status"] == "PASS"
    assert report["scheduler_progress"] == [0, 16, 32]
    assert report["stage_matches_reference"] == {stage: True for stage in "ABCDEF"}
    assert report["chunk_positions"] == prompt_positions


def test_base_token_end_chunks_span_an_h_on_each_side_of_the_boundary():
    spans = _unit_spans()
    spans[1] = 3
    spans[2] = 4
    before = [10, 11, H_START + 1]
    after = [H_START + 2, 12]
    history = before
    first = chunk_positions_from_computed(
        before,
        [1] * len(before),
        {1: 0},
        {1: spans},
        {1: []},
        position_mode="base_token_end",
        use_hypertoken_spans=True,
    )
    # 10, 11, then H3 ends at position 4.
    assert first == [0, 1, 4]
    second = chunk_positions_from_computed(
        after,
        [1] * len(after),
        {1: len(history)},
        {1: spans},
        {1: history},
        position_mode="base_token_end",
        use_hypertoken_spans=True,
    )
    # Prefix is 1+1+3=5. H4 ends at 8. The next base token is 9.
    assert second == [8, 9]


def test_resume_starts_at_nonzero_num_computed():
    positions = chunk_positions_from_computed(
        [7, 8, 9],
        [2, 2, 2],
        {2: 20},
        position_mode="compressed",
    )
    assert positions == [20, 21, 22]


def test_h4_advance_is_four_positions_inside_one_chunk():
    spans = _unit_spans()
    spans[5] = 4
    positions = chunk_positions_from_computed(
        [3, H_START + 5, 4],
        [0, 0, 0],
        {0: 0},
        {0: spans},
        {0: []},
        position_mode="base_token_end",
        use_hypertoken_spans=True,
    )
    assert positions == [0, 4, 5]


def test_slot_reuse_reads_the_new_request_computed_count():
    first = chunk_positions_from_computed([1, 2], [6, 6], {6: 8}, position_mode="compressed")
    second = chunk_positions_from_computed([1, 2], [6, 6], {6: 0}, position_mode="compressed")
    assert first == [8, 9]
    assert second == [0, 1]


def test_two_requests_keep_separate_computed_origins():
    positions = chunk_positions_from_computed(
        [1, 2, 3, 4],
        [0, 1, 0, 1],
        {0: 10, 1: 0},
        position_mode="compressed",
    )
    assert positions == [10, 0, 11, 1]


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
    assert budget["a_peak_kv_blocks"] <= budget["usable_blocks"]
    assert budget["b_peak_kv_blocks"] <= budget["usable_blocks"]
    assert budget["combined_peak_kv_blocks"] > budget["usable_blocks"]
    assert budget["a_prompt_blocks"] + budget["b_prompt_blocks"] <= budget["usable_blocks"]
    assert budget["max_new_tokens"] == 14


def test_preemption_budget_does_not_count_the_final_output_as_cached():
    budget = preemption_block_budget(42, 74, block_size=16)

    assert budget["max_new_tokens"] == 8
    assert budget["a_peak_kv_tokens"] == 49
    assert budget["b_peak_kv_tokens"] == 81
    assert budget["a_peak_kv_blocks"] == 4
    assert budget["b_peak_kv_blocks"] == 6
    assert budget["usable_blocks"] == 8
    assert budget["num_gpu_blocks"] == 9
    assert budget["combined_peak_kv_blocks"] == 10
    assert (
        blocks_for_tokens(42 + 7 - 1, 16)
        + blocks_for_tokens(74 + 7 - 1, 16)
        == budget["usable_blocks"]
    )
    assert (
        blocks_for_tokens(42 + 8 - 1, 16)
        + blocks_for_tokens(74 + 8 - 1, 16)
        == budget["combined_peak_kv_blocks"]
    )


def test_scheduler_preemption_counts_keep_the_maximum_seen_per_request():
    snapshots = [
        {
            "running": [
                {"request_id": "pre-A", "num_preemptions": 0},
                {"request_id": "pre-B", "num_preemptions": 0},
            ],
            "waiting": [],
        },
        {
            "running": [{"request_id": "pre-B", "num_preemptions": 0}],
            "waiting": [{"request_id": "pre-A", "num_preemptions": 1}],
        },
        {
            "running": [],
            "waiting": [{"request_id": "pre-A", "num_preemptions": 2}],
        },
    ]

    assert max_scheduler_preemptions(snapshots, ("pre-A", "pre-B")) == {
        "pre-A": 2,
        "pre-B": 0,
    }


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


def _phase9_token_hash(token_ids):
    payload = b"".join(int(token).to_bytes(8, "little", signed=True) for token in token_ids)
    return hashlib.sha256(payload).hexdigest()


def _phase9_rebuild_events(*, resumed_count=0, dirty_resumed_slot=False, changes=None):
    identity = {
        "h_enabled": True,
        "position_mode": "compressed",
        "sha256": "codebook-sha",
        "h_input_hash": "h-input-hash",
        "h_output_hash": "h-output-hash",
        "spans": [1, 3],
        "h_active_after_synthesis": True,
    }
    initial = {
        "event": "add",
        "req_id": "pre-A",
        "req_index": 2,
        "was_clear": True,
        "already": 0,
        "semantic_offset": 0,
        "admission_generation": 1,
        "prefill_token_count": 32,
        "prefill_token_ids_sha256": _phase9_token_hash(range(32)),
        "slot_clear_before_synthesis": {"valid": True},
        **identity,
    }
    resumed = {
        **initial,
        "already": resumed_count,
        "semantic_offset": resumed_count,
        "admission_generation": 2,
        "was_clear": not dirty_resumed_slot,
        "slot_clear_before_synthesis": {"valid": not dirty_resumed_slot},
    }
    for field, value in (changes or {}).items():
        resumed[field] = value
    events = [
        initial,
        {
            "event": "remove",
            "req_id": "pre-A",
            "req_index": 2,
            "admission_generation": 1,
            "slot_clear_after_remove": {"valid": True},
        },
        resumed,
    ]
    return events


def _phase9_resume_trace(*, chunks=((0, 16), (16, 16)), generation=2, corrupt_stage=None):
    trace = []
    for computed, scheduled in chunks:
        expected = list(range(computed, computed + scheduled))
        stage_positions = list(expected)
        event = {
            "target_request_id": "pre-A",
            "request_id": "pre-A",
            "admission_generation": generation,
            "slot": 2,
            "num_computed_tokens_on_batch": computed,
            "num_computed_tokens_source": "input_batch.num_computed_tokens_np[row]",
            "target_query_slice": [0, scheduled],
            "target_token_ids": list(range(computed, computed + scheduled)),
            "position_mode": "compressed",
            "requests": [
                {
                    "request_id": "pre-A",
                    "slot": 2,
                    "num_computed_tokens": computed,
                    "num_computed_tokens_source": "input_batch.num_computed_tokens_np[row]",
                    "num_scheduled_tokens": scheduled,
                    "query_token_slice": [0, scheduled],
                    "admission_generation": generation,
                }
            ],
            "stock_input_batch_positions": {"target_values": list(stage_positions)},
            "calculated_semantic_positions": {"target_values": list(stage_positions)},
            "returned_positions": {"target_values": list(stage_positions)},
            "handoff": {
                stage: [{"target_values": list(stage_positions)}]
                for stage in ("D", "E", "F")
            },
        }
        if corrupt_stage == "restart" and computed > 0:
            event["stock_input_batch_positions"]["target_values"] = list(range(scheduled))
            event["calculated_semantic_positions"]["target_values"] = list(range(scheduled))
            event["returned_positions"]["target_values"] = list(range(scheduled))
            event["handoff"] = {
                stage: [{"target_values": list(range(scheduled))}]
                for stage in ("D", "E", "F")
            }
        trace.append(event)
    return trace


def _phase9_position_report(*, chunks=((0, 16), (16, 16)), expected_count=0, corrupt_stage=None):
    return resume_position_contract_report(
        _phase9_resume_trace(chunks=chunks, corrupt_stage=corrupt_stage),
        "pre-A",
        admission_generation=2,
        prefill_token_count=32,
        position_mode="compressed",
        expected_readmission_count=expected_count,
        expected_prefill_token_ids_sha256=_phase9_token_hash(range(32)) if expected_count == 0 else None,
    )


def test_preemption_rebuild_report_accepts_zero_count_recompute_resume():
    events = _phase9_rebuild_events(resumed_count=0)
    position_resume = _phase9_position_report()

    report = preemption_rebuild_report(
        events,
        "pre-A",
        scheduler_preemption_count=1,
        position_resume=position_resume,
        trajectory_match=True,
        expected_readmission_count=0,
    )

    assert report["add_remove_add_cycle"]
    assert report["codebook_identity_recorded"]
    assert report["resume_strategy"] == "recompute_from_zero"
    assert report["num_computed_tokens_on_readmission"] == 0
    assert report["predictive_rebuild"]["valid"]
    assert report["resumed_slot_was_clear"]
    assert report["position_resume"]["trace_complete"]
    assert report["position_resume"]["all_model_positions_match_expected"]
    assert report["state_rebuilt"]


def test_preemption_rebuild_rejects_predictive_identity_mismatch():
    for field, bad_value in (
        ("sha256", "different-codebook"),
        ("h_input_hash", "different-input"),
        ("h_output_hash", "different-output"),
        ("spans", [1, 4]),
    ):
        report = preemption_rebuild_report(
            _phase9_rebuild_events(resumed_count=0, changes={field: bad_value}),
            "pre-A",
            scheduler_preemption_count=1,
            position_resume=_phase9_position_report(),
            trajectory_match=True,
            expected_readmission_count=0,
        )
        assert not report["state_rebuilt"], field


def test_preemption_rebuild_rejects_dirty_resumed_slot():
    report = preemption_rebuild_report(
        _phase9_rebuild_events(resumed_count=0, dirty_resumed_slot=True),
        "pre-A",
        scheduler_preemption_count=1,
        position_resume=_phase9_position_report(),
        trajectory_match=True,
        expected_readmission_count=0,
    )
    assert not report["slot_clear"]["valid"]
    assert not report["state_rebuilt"]


def test_preemption_rebuild_requires_scheduler_preemption_count():
    report = preemption_rebuild_report(
        _phase9_rebuild_events(resumed_count=0),
        "pre-A",
        scheduler_preemption_count=0,
        position_resume=_phase9_position_report(),
        trajectory_match=True,
        expected_readmission_count=0,
    )
    assert not report["scheduler_preemption_confirmed"]
    assert not report["state_rebuilt"]


def test_resume_position_contract_accepts_contiguous_recompute_chunks():
    report = _phase9_position_report()
    assert report["resume_strategy"] == "recompute_from_zero"
    assert report["valid"]
    assert report["trace_complete"]
    assert report["all_model_positions_match_expected"]


def test_resume_position_contract_rejects_positions_restarting_after_first_chunk():
    report = _phase9_position_report(corrupt_stage="restart")
    assert not report["valid"]
    assert not report["all_model_positions_match_expected"]


def test_resume_position_contract_represents_cached_prefix_origin():
    report = resume_position_contract_report(
        _phase9_resume_trace(chunks=((16, 16),)),
        "pre-A",
        admission_generation=2,
        prefill_token_count=32,
        position_mode="compressed",
        expected_readmission_count=None,
    )
    assert report["resume_strategy"] == "cached_prefix"
    assert report["num_computed_tokens_on_readmission"] == 16
    assert report["valid"]


def test_two_codebook_isolation_requires_distinct_matching_rows():
    assert two_codebook_isolation_passes(
        a_match=True,
        b_match=True,
        row0_match=True,
        row1_match=True,
        rows_differ=True,
    )
    assert not two_codebook_isolation_passes(
        a_match=True,
        b_match=True,
        row0_match=True,
        row1_match=True,
        rows_differ=False,
    )


def test_semantic_rope_positions_use_the_rope_bound_not_the_kv_limit():
    report = semantic_kv_rope_contract_report(
        physical_positions=[0, 1, 2, 3, 4, 5],
        semantic_positions=[2, 5, 8, 11, 14, 17],
        expected_semantic_positions=[2, 5, 8, 11, 14, 17],
        physical_token_count=6,
        max_new_tokens=1,
        max_model_len=8,
        max_rope_position=131072,
    )
    assert report["status"] == "PASS"
    assert report["checks"]["physical_positions_within_max_model_len"]
    assert report["checks"]["physical_prompt_plus_generation_fits_max_model_len"]
    assert report["checks"]["semantic_positions_exceed_max_model_len"]
    assert report["checks"]["semantic_positions_within_rope_limit"]

    bad_physical = semantic_kv_rope_contract_report(
        physical_positions=[0, 1, 2, 3, 4, 8],
        semantic_positions=[2, 5, 8, 11, 14, 17],
        expected_semantic_positions=[2, 5, 8, 11, 14, 17],
        physical_token_count=6,
        max_new_tokens=1,
        max_model_len=8,
        max_rope_position=131072,
    )
    assert bad_physical["status"] == "FAIL"
