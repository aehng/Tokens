from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn
import experiments.run_phi_attribution_benchmark as benchmark

from experiments.run_phi_attribution_benchmark import (
    base_phi_weight_sha256,
    disabled_adapter_context,
    capture_prefix_logits,
    ForcedScheduleLogitsProcessor,
    generation_policy,
    generation_policy_sha256,
    run_a_b1_logit_fidelity,
)
from src.zip2zip.predictor_v2.forced_oracle import force_oracle_substitutions, h_vs_base_prefix_pair
from src.zip2zip.predictor_v2.attribution_harness import (
    CANONICAL_EOS_TOKEN_IDS,
    PAD_TOKEN_ID,
    STRATIFIED_DEV12_PROMPT_IDS,
    select_stratified_dev_prompts,
)
from src.zip2zip.predictor_v2.ablation_gates import (
    checkpoint_isolation_gate,
    forced_h_representation_gates,
    logit_parity_metrics,
    normalize_wrapper_logits,
    token_equivalence_gate,
)

from src.zip2zip.predictor_v2.canonical_dataset import CanonicalContinuation
from src.zip2zip.static_codebook import StaticCodebookManager


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 4)
        self.lm_head = nn.Linear(4, 11, bias=False)

    def get_input_embeddings(self):
        return self.embed


class TinyPrefixLogitModel(nn.Module):
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        rows = torch.arange(7, dtype=torch.float32).view(1, 1, 7)
        return type("TinyOutput", (), {"logits": rows + input_ids.unsqueeze(-1).float()})()


class TinyLogitCore(nn.Module):
    def __init__(self, vocab_size=5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 4)
        self.lm_head = nn.Linear(4, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


class TinyExpandedLogitModel(nn.Module):
    def __init__(self, *, insert_h_rows=False, change_base_logits=False):
        super().__init__()
        self.base_model = TinyLogitCore(vocab_size=5)
        self.config = SimpleNamespace(hidden_size=4)
        self.insert_h_rows = insert_h_rows
        self.change_base_logits = change_base_logits

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.base_model.get_input_embeddings()(input_ids)
        logits = self.base_model.get_output_embeddings()(hidden)
        if self.change_base_logits:
            logits = logits.clone()
            logits[..., 0] += 0.5
        if self.insert_h_rows:
            logits = torch.cat((logits, logits.new_zeros((*logits.shape[:-1], 2))), dim=-1)
        return SimpleNamespace(logits=logits)


class TinyPeftModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.ones(2, 2))
        self.peft_config = {"default": object()}
        self.adapter_enabled = True

    def get_base_model(self):
        return self.base

    @contextmanager
    def disable_adapter(self):
        prior = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = prior


class TinyZip2Zip(nn.Module):
    def __init__(self, peft_model):
        super().__init__()
        self.base_model = peft_model


def test_generation_policy_pins_greedy_eos_padding_and_budget():
    policy = generation_policy(77)
    assert policy == {
        "max_new_tokens": 77,
        "do_sample": False,
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": list(CANONICAL_EOS_TOKEN_IDS),
    }
    assert generation_policy_sha256(77) == generation_policy_sha256(77)
    assert generation_policy_sha256(77) != generation_policy_sha256(78)


def test_native_and_wrapper_base_phi_fingerprints_match_and_cover_lm_head():
    torch.manual_seed(5)
    vanilla = TinyCausalLM()
    torch.manual_seed(5)
    wrapped = TinyZip2Zip(TinyPeftModel(TinyCausalLM()))
    assert base_phi_weight_sha256(vanilla) == base_phi_weight_sha256(wrapped)
    with torch.no_grad():
        vanilla.lm_head.weight[0, 0] += 1.0
    assert base_phi_weight_sha256(vanilla) != base_phi_weight_sha256(wrapped)


def test_b0_adapter_is_disabled_only_inside_each_context_and_restored():
    model = TinyZip2Zip(TinyPeftModel(TinyCausalLM()))
    assert model.base_model.adapter_enabled is True
    for _ in range(2):
        with disabled_adapter_context(model):
            assert model.base_model.adapter_enabled is False
        assert model.base_model.adapter_enabled is True


def test_b0_refuses_peft_adapter_that_cannot_be_disabled():
    class UndisableablePeft(nn.Module):
        def __init__(self):
            super().__init__()
            self.peft_config = {"default": object()}

    model = TinyZip2Zip(UndisableablePeft())
    try:
        disabled_adapter_context(model)
    except RuntimeError as exc:
        assert "disable_adapter is unavailable" in str(exc)
    else:
        raise AssertionError("B0 must fail closed when a present adapter cannot be disabled")


def test_forced_oracle_substitution_expands_exactly_and_advances_semantic_positions():
    phrases = {(10, 11): 32011, (20, 21, 22): 32012}
    intended = [99, 10, 11, 20, 21, 22, 10, 11, 99]
    forced = force_oracle_substitutions(intended, phrases)
    assert forced["expanded_token_ids"] == intended
    assert forced["forced_generation_ids"] == [99, 32011, 32012, 32011, 99]
    assert [entry["id"] for entry in forced["h_emissions"]] == [32011, 32012, 32011]
    assert forced["tokens_saved"] == 4
    assert [entry["schedule_index"] for entry in forced["h_emissions"]] == [1, 2, 3]

    manager = StaticCodebookManager(
        initial_vocab_size=32011,
        max_codebook_size=2,
        max_subtokens=4,
        embedding_dim=4,
        pad_token_id=32000,
    )
    manager.set_seeded_codebook(phrases, batch_size=1, device="cpu")
    positions = manager.prepare_input_ids(torch.tensor([forced["forced_generation_ids"]]))
    assert positions.tolist() == [[0, 2, 5, 7, 8]]
    assert manager.base_position_offset.tolist() == [[9]]


def test_forced_oracle_does_not_substitute_nonmatching_spans():
    forced = force_oracle_substitutions([10, 12, 11], {(10, 11): 32011})
    assert forced["forced_generation_ids"] == [10, 12, 11]
    assert forced["h_emissions"] == []
    assert forced["expanded_token_ids"] == [10, 12, 11]


def test_logit_capture_uses_same_token_prefix_indices_and_stops_before_eos():
    model = TinyPrefixLogitModel()
    captured = capture_prefix_logits(
        model,
        [1, 2],
        [3, 4, 32007, 6],
        device=torch.device("cpu"),
        inserted_h_count=0,
    )
    assert set(captured) == {0, 1}
    assert captured[0].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert captured[1].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_forced_schedule_processor_uses_call_count_when_input_ids_are_cached():
    processor = ForcedScheduleLogitsProcessor(100, [7, 8, 9])
    scores = torch.zeros((1, 20))
    out = processor(torch.tensor([[5]]), scores.clone())
    assert int(out[0].argmax()) == 7
    out = processor(torch.tensor([[7]]), scores.clone())
    assert int(out[0].argmax()) == 8
    assert processor.unforced_top1 == [0, 0]


def test_h_vs_base_prefix_pair_ends_on_equivalent_semantic_content():
    forced = force_oracle_substitutions([10, 11, 12], {(10, 11): 32011})
    prefixes = h_vs_base_prefix_pair([1, 2], forced, forced["h_emissions"][0])
    assert prefixes["base_prefix_ids"] == [1, 2, 10, 11]
    assert prefixes["h_prefix_ids"] == [1, 2, 32011]
    assert prefixes["expected_next_base_token_id"] == 12
    assert prefixes["expected_next_compressed_id"] == 12


def test_pinned_dev12_ids_are_four_per_domain_and_stable():
    assert len(STRATIFIED_DEV12_PROMPT_IDS) == 12
    assert STRATIFIED_DEV12_PROMPT_IDS[:4] == ("mbpp_113", "mbpp_168", "mbpp_217", "mbpp_225")
    assert STRATIFIED_DEV12_PROMPT_IDS[4:8] == ("gsm_2032", "gsm_2044", "gsm_2353", "gsm_2491")
    assert STRATIFIED_DEV12_PROMPT_IDS[8:] == ("alpaca_1", "alpaca_1024", "alpaca_1029", "alpaca_1132")
    records = [
        {"prompt_id": prompt_id, "domain": "code" if prompt_id.startswith("mbpp_") else ("reasoning" if prompt_id.startswith("gsm_") else "instruction")}
        for prompt_id in (*STRATIFIED_DEV12_PROMPT_IDS, "mbpp_999", "gsm_999", "alpaca_999")
    ]
    selected = select_stratified_dev_prompts(records, limit=12)
    assert [row["prompt_id"] for row in selected] == list(STRATIFIED_DEV12_PROMPT_IDS)


def test_forced_schedule_processor_records_h_semantic_position_and_offset():
    manager = StaticCodebookManager(
        initial_vocab_size=32011,
        max_codebook_size=2,
        max_subtokens=4,
        embedding_dim=4,
        pad_token_id=32000,
    )
    manager.set_seeded_codebook({(10, 11): 32011}, batch_size=1, device="cpu")
    manager.prepare_input_ids(torch.tensor([[5, 6]]))
    processor = ForcedScheduleLogitsProcessor(
        2,
        [32011, 7],
        static_mgr=manager,
        h_position_expectations={0: {"expected_position": 3, "expected_next_offset": 4}},
    )
    scores = torch.zeros((1, 32020))
    processor(torch.tensor([[5, 6]]), scores.clone())
    manager.prepare_input_ids(torch.tensor([[32011]]))
    processor(torch.tensor([[32011]]), scores.clone())
    check = processor.semantic_position_checks[0]
    assert check["observed_position"] == check["expected_position"] == 3
    assert check["observed_next_offset"] == check["expected_next_offset"] == 4


def test_a_b0_token_gate_requires_exact_output_and_matching_generation_contract():
    common = {
        "base_phi_weight_sha256": "base",
        "generation_policy_sha256": "policy",
        "input_token_ids_sha256": "prompt",
        "torch_dtype": "torch.float32",
        "tokenizer_id": "model",
        "tokenizer_revision": "rev",
        "tokenizer_chat_template_sha256": "template",
        "attention_implementation": "transformers_default",
        "rendered_prompt_sha256": "rendered",
        "eos_token_ids": [32007, 32001, 32000],
        "max_new_tokens": 77,
        "generation_config": {"do_sample": False, "max_new_tokens": 77},
        "checkpoint_applied": False,
        "active_lora_during_generation": False,
        "adapter_disabled_context_used": True,
        "h_seeded_slots": 0,
        "h_logits_masked_by_static_manager": True,
    }
    records = [
        {"prompt_id": "p", "condition": "A_vanilla", "generated_token_ids": [1, 2, 32007],
         "termination_reason": "eos", "termination_token_id": 32007, "runtime": common},
        {"prompt_id": "p", "condition": "B0_tokens_vanilla_weights", "generated_token_ids": [1, 2, 32007],
         "termination_reason": "eos", "termination_token_id": 32007, "runtime": dict(common)},
    ]
    assert token_equivalence_gate(records)["status"] == "PASS"
    records[1]["generated_token_ids"] = [1, 3, 32007]
    assert token_equivalence_gate(records)["status"] == "FAIL"


def test_logit_gate_reports_all_requested_metrics_and_fails_top1_drift():
    a = torch.tensor([[4.0, 2.0, 1.0, -1.0], [3.0, 1.0, 0.0, -2.0]])
    b = a.clone()
    equal = logit_parity_metrics(a, b, eos_token_ids=[3], special_token_ids=[2], top_k=2, atol=1e-6)
    assert equal["status"] == "PASS"
    assert equal["max_abs_logit_difference"] == 0.0
    assert equal["mean_kl_reference_to_candidate_nats"] == 0.0
    assert equal["top1_all_positions_agree"] is True
    assert equal["eos_special_logit_differences"]["3"]["max_abs_difference"] == 0.0
    b[1, 1] = 4.0
    changed = logit_parity_metrics(a, b, eos_token_ids=[3], special_token_ids=[2], top_k=2, atol=1e-6)
    assert changed["status"] == "FAIL"
    assert changed["top1_agreement_rate"] == 0.5
    assert "mean_abs_top1_margin_difference" in changed
    assert changed["mean_kl_reference_to_candidate_nats"] > 0.0
    assert torch.isfinite(torch.tensor(changed["max_kl_reference_to_candidate_nats"]))


def test_a_b1_fidelity_compares_identical_vanilla_prefixes_and_reports_kl(monkeypatch):
    sample = CanonicalContinuation(
        prompt_id="dev-p1",
        domain="instruction",
        split="DEV",
        task_prompt_text="prompt",
        rendered_prompt_text="<|user|>test",
        continuation_text="answer",
        continuation_token_ids=(3, 4),
        generated_token_count=2,
        termination_reason="max_new_tokens",
        termination_token_id=None,
        model_id="phi",
        model_revision="rev",
        tokenizer_id="phi",
        tokenizer_revision="rev",
        generation_config_json="{}",
        generation_metadata_json="{}",
    )

    class TinyTokenizer:
        all_special_ids = []

        @staticmethod
        def encode(text, add_special_tokens=False):
            assert text == "<|user|>test"
            return [1, 2]

    tokenizer = TinyTokenizer()
    vanilla = TinyExpandedLogitModel()
    b1 = TinyExpandedLogitModel(insert_h_rows=True, change_base_logits=True)
    monkeypatch.setattr(benchmark, "INITIAL_VOCAB_SIZE", 5)
    monkeypatch.setattr(benchmark, "load_vanilla_model_and_tokenizer", lambda device: (vanilla, tokenizer))

    prompt_hash = benchmark.token_ids_sha256([1, 2])
    records = [
        {"prompt_id": "dev-p1", "condition": "A_vanilla", "generated_token_ids": [3, 4],
         "runtime": {"input_token_ids_sha256": prompt_hash}},
        {"prompt_id": "dev-p1", "condition": "B0_tokens_vanilla_weights",
         "runtime": {"input_token_ids_sha256": prompt_hash}},
    ]
    diagnostic = run_a_b1_logit_fidelity([sample], records, b1, tokenizer, "cpu", k=2)

    assert diagnostic["status"] == "MEASURED"
    assert diagnostic["checked_state_count"] == 2
    assert diagnostic["aggregate"]["checked_state_count"] == 2
    assert diagnostic["aggregate"]["mean_kl_reference_to_candidate_nats"] > 0.0
    assert diagnostic["per_prompt"][0]["prefix_results"][0]["continuation_prefix_length"] == 0


def test_wrapper_logit_normalization_removes_only_inserted_h_rows():
    base = torch.arange(10, dtype=torch.float32)
    wrapped = torch.cat((base[:4], torch.tensor([99.0, 98.0]), base[4:]))
    assert torch.equal(
        normalize_wrapper_logits(wrapped, base_vocab_size=10, initial_vocab_size=4, inserted_h_count=2),
        base,
    )


def test_b0_b1_isolation_gate_requires_exact_changed_parameter_inventory():
    changes = {
        "lora": {"names": ["lora_A"], "shapes": {"lora_A": [2, 2]}, "changed_tensor_count": 1},
        "input_encoder": {"names": ["input.weight"], "shapes": {"input.weight": [2, 2]}, "changed_tensor_count": 1},
        "output_encoder": {"names": ["output.weight"], "shapes": {"output.weight": [2, 2]}, "changed_tensor_count": 1},
    }
    records = [
        {"prompt_id": "p", "condition": "B0_tokens_vanilla_weights", "runtime": {
            "base_phi_weight_sha256": "base", "checkpoint_applied": False,
            "step100_h_encoder_checkpoint_loaded": False,
            "active_lora_during_generation": False, "adapter_disabled_context_used": True,
            "h_seeded_slots": 0, "h_logits_masked_by_static_manager": True}},
        {"prompt_id": "p", "condition": "B_h_disabled", "runtime": {
            "base_phi_weight_sha256": "base", "lora_delta_verified": True,
            "active_lora_during_generation": True,
            "checkpoint_changed_parameters": changes}},
    ]
    assert checkpoint_isolation_gate(records)["status"] == "PASS"
    records[1]["runtime"]["checkpoint_changed_parameters"]["lora"]["shapes"] = {}
    assert checkpoint_isolation_gate(records)["status"] == "FAIL"


def test_forced_h_gate_keeps_three_independent_acceptance_checks():
    record = {
        "condition": "CF_forced_oracle",
        "forced_oracle_roundtrip_ok": True,
        "cf_semantic_positions_ok": True,
        "cf_continuation_stable": True,
        "runtime": {"cf_semantic_position_checks": [{"observed_position": 5}], "cf_continuation_checks": [{"top1_matches": True}]},
    }
    gates = forced_h_representation_gates([record])
    assert gates["status"] == "PASS"
    record["cf_continuation_stable"] = False
    assert forced_h_representation_gates([record])["status"] == "FAIL"
