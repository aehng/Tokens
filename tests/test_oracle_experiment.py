import pytest
from experiments.oracle_compression import (
    extract_candidate_ngrams,
    select_oracle_codebook,
    evaluate_sample_oracle,
)


def test_oracle_candidate_extraction():
    tokens = [1, 2, 3, 4, 1, 2, 3]
    disabled = {0}
    counts = extract_candidate_ngrams(tokens, disabled, min_len=2, max_len=3)

    assert counts[(1, 2)] == 2
    assert counts[(2, 3)] == 2
    assert counts[(1, 2, 3)] == 2
    assert counts[(3, 4)] == 1


def test_oracle_codebook_selection_and_compression():
    # Synthetic repetitive sequence
    prompt = [10, 20, 30, 40]
    # Response contains repeated 2-token and 3-token sequences
    response = [10, 20, 10, 20, 10, 20, 50, 60, 70, 50, 60, 70]
    disabled = {0}

    metrics = evaluate_sample_oracle(
        prompt_ids=prompt,
        response_ids=response,
        budget=16,
        disabled_ids=disabled,
        initial_vocab_size=1000,
    )

    # Base response tokens = 12
    # [10, 20] appears 3 times (saves 3 tokens)
    # [50, 60, 70] appears 2 times (saves 4 tokens)
    # Total saved in response = 7 tokens -> response length drops to 5
    assert metrics.base_response_tokens == 12
    assert metrics.hyper_response_tokens < metrics.base_response_tokens
    assert metrics.response_compression_pct > 40.0
