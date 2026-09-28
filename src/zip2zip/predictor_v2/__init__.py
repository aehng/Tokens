"""Predictor V2 Architecture Bake-Off package.

Contains:
- candidate_pool: Fixed high-recall prompt-only candidate generator and feature extractor.
- vanilla_labels: Canonical Vanilla Phi continuation data contract and label extractor.
- dataset: Train/Dev/Frozen Test data structures, deterministic partitioning, and split manifest.
- oracle_global: Oracle A - Global Occurrence Oracle (unrestricted DP ceiling).
- oracle_candidate_pool: Oracle B - Fixed Candidate-Pool Oracle.
- safety_labels: Oracle C - Empirical continuation-safety probe harness & heuristic baseline audit.
- interfaces: PredictorScorer base class and evaluation protocol.
- utility: Dynamic programming utility calculation and ranking scores.
- metrics: Ranking, calibration, and classification metrics.
- profiling: CPU/GPU latency and compute profiler.
- models: Model implementations (Ridge, Pooled MLP, CNN, GRU, Transformer).
"""

__version__ = "2.0.0"
