"""CPU checks that quality-benchmark termination follows the tokenizer EOS id."""

from experiments.run_quality_benchmark import sequence_reached_eos

EOS_ID = 32000
MAX_NEW_TOKENS = 8


def test_eos_reached_uses_final_token_not_length():
    assert sequence_reached_eos([11, 12, EOS_ID], EOS_ID) is True
    assert sequence_reached_eos([11, 12, 13], EOS_ID) is False
    assert len([11, 12, 13]) < MAX_NEW_TOKENS
    assert sequence_reached_eos([7] * MAX_NEW_TOKENS, EOS_ID) is False
    assert sequence_reached_eos([], EOS_ID) is False
