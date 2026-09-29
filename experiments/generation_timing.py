"""Generation timing helpers shared by attribution and quality runners.

Kept separate from ``run_quality_benchmark`` so the predictive attribution
stack can import timing without loading the legacy Zip2Zip tokenizer.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

import torch
from transformers import LogitsProcessor


class TimingLogitsProcessor(LogitsProcessor):
    """Measure TTFT and decode steps, and mask unseeded hypertoken logits."""

    def __init__(self, t_start: float, static_mgr: Any = None):
        self.t_start = t_start
        self.static_mgr = static_mgr
        self.ttft: Optional[float] = None
        self.step_count = 0
        self.decode_step_intervals_s: List[float] = []
        self._last_callback_time: Optional[float] = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        callback_time = time.perf_counter()
        if self.ttft is None:
            if scores.is_cuda:
                torch.cuda.synchronize(scores.device)
                callback_time = time.perf_counter()
            self.ttft = callback_time - self.t_start
        elif not scores.is_cuda and self._last_callback_time is not None:
            self.decode_step_intervals_s.append(callback_time - self._last_callback_time)
        self._last_callback_time = callback_time
        self.step_count += 1
        if self.static_mgr is not None:
            scores = self.static_mgr.mask_unused_logits(scores)
        return scores


def synchronize_device(device: torch.device) -> None:
    """Finish queued CUDA work before taking wall-clock timestamps."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
