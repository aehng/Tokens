"""CPU checks that quality-benchmark termination follows the tokenizer EOS id."""

import torch

from zip2zip import StaticCodebookManager
from experiments.run_quality_benchmark import (
    prepare_prompt_input_ids,
    run_condition_predictive,
    sequence_reached_eos,
)

EOS_ID = 32000
MAX_NEW_TOKENS = 8


def test_eos_reached_uses_final_token_not_length():
    assert sequence_reached_eos([11, 12, EOS_ID], EOS_ID) is True
    assert sequence_reached_eos([11, 12, 13], EOS_ID) is False
    assert len([11, 12, 13]) < MAX_NEW_TOKENS
    assert sequence_reached_eos([7] * MAX_NEW_TOKENS, EOS_ID) is False
    assert sequence_reached_eos([], EOS_ID) is False


def test_predictive_prompt_compression_roundtrips_and_preserves_raw_option():
    manager = StaticCodebookManager(
        initial_vocab_size=100,
        max_codebook_size=4,
        max_subtokens=4,
        embedding_dim=8,
        pad_token_id=0,
    )
    manager.set_seeded_codebook({(11, 12): 100}, device=torch.device("cpu"))
    raw_ids = [11, 12, 13]

    assert prepare_prompt_input_ids(raw_ids, manager, compress_prompt=False) == raw_ids
    compressed_ids = prepare_prompt_input_ids(raw_ids, manager, compress_prompt=True)
    assert compressed_ids == [100, 13]
    assert manager.decode_sequence(compressed_ids) == raw_ids


def test_fully_cached_predictive_condition_does_not_load_checkpoint(tmp_path):
    records = run_condition_predictive(
        checkpoint_path=str(tmp_path / "intentionally_missing.pt"),
        condition_name="predictive_step_100",
        samples=[{"id": "prompt_1"}],
        raw_results_path=str(tmp_path / "raw.jsonl"),
        completed_keys={("prompt_1", "predictive_step_100")},
    )
    assert records == []
from experiments.run_quality_benchmark import PREDICTOR_PATH


def test_quality_benchmark_uses_the_canonical_oracle_guided_predictor():
    assert PREDICTOR_PATH.endswith("oracle_guided_predictor.pkl")
