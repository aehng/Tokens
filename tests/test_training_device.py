import pytest
import torch
from torch import nn

from experiments.train_predictive_zip2zip import (
    estimate_minimum_training_memory_bytes,
    move_training_model_to_device,
)


def test_training_model_is_moved_to_the_requested_cpu_device():
    model = nn.Linear(3, 2)
    returned = move_training_model_to_device(model, "cpu")

    assert returned is model
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}


def test_training_model_fails_before_move_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        move_training_model_to_device(nn.Linear(3, 2), "cuda:0")


def test_gpu_memory_lower_bound_counts_frozen_trainable_gradient_and_adam_storage():
    model = nn.Module()
    model.frozen = nn.Parameter(torch.zeros(2, dtype=torch.float16), requires_grad=False)
    model.trainable = nn.Parameter(torch.zeros(3, dtype=torch.float32), requires_grad=True)

    estimate = estimate_minimum_training_memory_bytes(model)

    assert estimate == {
        "frozen_parameter_bytes": 4,
        "trainable_parameter_bytes": 12,
        "gradient_bytes": 12,
        "adamw_state_bytes": 24,
        "static_minimum_bytes": 52,
    }
