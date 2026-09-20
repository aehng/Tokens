import pytest
from experiments.predictor import (
    PhraseBank,
    PredictiveVocabularyOptimizer,
    run_predictor_experiment,
)


def test_phrase_bank_mining():
    disabled = {0}
    bank = PhraseBank(disabled_ids=disabled)
    corpus = [
        ("code", [10, 20, 30], [20, 30, 40, 20, 30]),
        ("code", [10, 50], [20, 30, 60]),
    ]
    bank.train_on_corpus(corpus)

    assert (20, 30) in bank.global_counts
    assert bank.global_counts[(20, 30)] >= 3
    assert bank.prompt_to_phrase_cooccurrence[10][(20, 30)] >= 3


def test_predictive_vocabulary_optimizer():
    disabled = {0}
    bank = PhraseBank(disabled_ids=disabled)
    corpus = [
        ("code", [1, 2], [10, 20, 30, 40]),
        ("math", [3, 4], [50, 60, 70, 80]),
    ]
    bank.train_on_corpus(corpus)

    optimizer = PredictiveVocabularyOptimizer(bank, initial_vocab_size=1000)

    # Global static
    cb_global = optimizer.select_global_static(budget=4)
    assert len(cb_global) > 0
    assert all(idx >= 1000 for idx in cb_global.values())

    # Prompt conditioned with prompt [1, 2] should favor [10, 20] over [50, 60]
    cb_prompt = optimizer.select_prompt_conditioned(prompt_ids=[1, 2], budget=4)
    assert (10, 20) in cb_prompt
