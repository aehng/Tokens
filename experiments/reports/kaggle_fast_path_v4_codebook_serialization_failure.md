# Kaggle Fast-Path Validation: Version 4 Partial Result

## Run status

The run passed Step-100 mount and provenance checks, checkpoint and predictor
SHA-256 verification, TorchAO normalization, model-stack imports, and one-T4
verification. It completed the Vanilla behavioral smoke and fixed-active-KV
measurement, then stopped before either predictive condition ran.

This is a **valid Vanilla-only partial result**. It is not a predictive speed
result and does not establish a predictive speedup.

## Vanilla partial measurements

- Prompt: `gsm_2956`
- Behavioral smoke used a static cache with capacity 256; the active context
  grew during generation, so these timings are not fixed-KV measurements.
- Behavioral TTFT: approximately 0.068 seconds
- Behavioral generation: EOS was not emitted; generation reached the
  `max_new_tokens=101` limit.
- Fixed-KV protocol: active KV length 256, 20 warmups, 100 measured forwards.
- Reference cache length: 256 before and after timing.
- Fixed-KV median: 36.5557 ms per step (36.56 ms rounded).

Use a Vanilla measurement from the same future run as the baseline for any
predictive comparison. Do not treat historical V16 timing as an equivalent
measurement environment.

## Failure cause

After selecting the predictive codebook, the harness failed in
`_serialize_codebook()` because it treated mapping keys as H IDs and called
`int()` on a tuple. The canonical `CappedPredictorPolicy` result is
`Mapping[Tuple[int, ...], int]`: phrase tuple to absolute hypertoken ID. The
serialization assumption was backwards. No Predictive Legacy/Fast generation
or fixed-KV measurement ran, and there is no predictive result from this run.

The policy result and runtime codebook orientation are unchanged. The harness
fix validates that contract and serializes the phrase-to-H-ID mapping in H-ID
order before it is used by the manager.
