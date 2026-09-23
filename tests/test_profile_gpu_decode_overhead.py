"""Unit tests for pure helper logic in profile_gpu_decode_overhead."""

import unittest
from experiments.profile_gpu_decode_overhead import _compute_stats


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


if __name__ == "__main__":
    unittest.main()
