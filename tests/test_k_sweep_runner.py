import pytest

from experiments.run_k_sweep import (
    DEFAULT_PREDICTOR,
    DEFAULT_PHI_REVISION,
    DEFAULT_ZIP2ZIP_REVISION,
    _validate_parameters,
    codebook_hash,
    domain_scoring_summary,
)


def test_k_sweep_parameters_accept_planned_fixed_k_set():
    _validate_parameters([4, 8, 16, 24, 32], [], max_new_tokens=300)


def test_k_sweep_defaults_to_the_canonical_oracle_guided_predictor():
    assert DEFAULT_PREDICTOR.name == "oracle_guided_predictor.pkl"


def test_k_sweep_pins_both_model_revisions():
    assert len(DEFAULT_PHI_REVISION) == 40
    assert len(DEFAULT_ZIP2ZIP_REVISION) == 40


@pytest.mark.parametrize(
    "k_values",
    [[], [0], [33], [4, 4], [4.5], [True]],
)
def test_k_sweep_rejects_invalid_k_values(k_values):
    with pytest.raises(ValueError, match="k_values"):
        _validate_parameters(k_values, [], max_new_tokens=300)


def test_codebook_hash_covers_all_mapping_entries():
    first = {(1, 2): 32011, (4, 5, 6): 32012}
    assert codebook_hash(first) == codebook_hash(dict(reversed(list(first.items()))))
    assert codebook_hash(first) != codebook_hash({(1, 2): 32011})


def test_k_sweep_reports_instruction_as_mechanical_and_separately():
    scores = domain_scoring_summary(
        [
            {"domain": "code", "problem_pass": True},
            {"domain": "reasoning", "exact_correct": False},
            {"domain": "instruction", "mechanical_instruction_pass": True},
            {"domain": "instruction", "mechanical_instruction_pass": False},
        ]
    )

    assert scores["code"]["measurement"] == "MBPP pass@1"
    assert scores["code"]["rate_pct"] == 100.0
    assert scores["reasoning"]["measurement"] == "GSM8K exact answer"
    assert scores["reasoning"]["rate_pct"] == 0.0
    assert scores["instruction"]["measurement"] == "Alpaca mechanical checks"
    assert scores["instruction"]["passed"] == 1
    assert scores["instruction"]["semantic_adherence_available"] is False
