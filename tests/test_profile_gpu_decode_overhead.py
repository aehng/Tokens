"""Unit tests for pure helper logic in profile_gpu_decode_overhead."""

import unittest
import torch
from experiments.profile_gpu_decode_overhead import (
    _compute_stats,
    _clone_kv_cache,
    _get_kv_cache_seq_len,
    PrepareInputIdsCallCounter,
)


class DummyManager:
    def __init__(self):
        self.call_count = 0

    def prepare_input_ids(self, *args, **kwargs):
        self.call_count += 1
        return "prepared"


class TestProfileGpuDecodeOverheadPureHelpers(unittest.TestCase):
    def test_compute_stats_empty(self):
        res = _compute_stats([])
        self.assertEqual(res["mean"], 0.0)
        self.assertEqual(res["samples_count"], 0)

    def test_compute_stats_single(self):
        res = _compute_stats([10.5])
        self.assertEqual(res["mean"], 10.5)
        self.assertEqual(res["median"], 10.5)
        self.assertEqual(res["stddev"], 0.0)
        self.assertEqual(res["samples_count"], 1)

    def test_compute_stats_multiple(self):
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        res = _compute_stats(samples)
        self.assertEqual(res["mean"], 30.0)
        self.assertEqual(res["median"], 30.0)
        self.assertEqual(res["min"], 10.0)
        self.assertEqual(res["max"], 50.0)
        self.assertEqual(res["samples_count"], 5)

    def test_clone_kv_cache_and_seq_len(self):
        t1 = torch.randn(1, 4, 64, 32)
        t2 = torch.randn(1, 4, 64, 32)
        kv = ((t1, t2),)
        seq_len = _get_kv_cache_seq_len(kv)
        self.assertEqual(seq_len, 64)

        cloned_kv = _clone_kv_cache(kv)
        self.assertEqual(_get_kv_cache_seq_len(cloned_kv), 64)
        # Verify independence
        self.assertFalse(cloned_kv[0][0].data_ptr() == t1.data_ptr())
        self.assertTrue(torch.equal(cloned_kv[0][0], t1))

    def test_prepare_input_ids_counter(self):
        mgr = DummyManager()
        with PrepareInputIdsCallCounter(mgr) as counter:
            mgr.prepare_input_ids()
            mgr.prepare_input_ids()
            self.assertEqual(counter.call_count, 2)
        # After exit, counter unhooks
        mgr.prepare_input_ids()
        self.assertEqual(counter.call_count, 2)


if __name__ == "__main__":
    unittest.main()
