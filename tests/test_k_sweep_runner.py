import pytest

from experiments.run_k_sweep import DEFAULT_PREDICTOR, _validate_parameters, codebook_hash


def test_k_sweep_parameters_accept_planned_fixed_k_set():
    _validate_parameters([4, 8, 16, 24, 32], [], max_new_tokens=300)


def test_k_sweep_defaults_to_the_canonical_oracle_guided_predictor():
    assert DEFAULT_PREDICTOR.name == "oracle_guided_predictor.pkl"


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
