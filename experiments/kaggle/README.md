# Single-T4 Kaggle validation package

This package validates the committed Phi-3.5 predictive stack on one visible
T4 (`cuda:0`). It does not train, tune K, or run the full Tier-1 suite.

## What the kernel runs

1. Checks the private artifact dataset and pinned source commit.
2. Installs the inference dependencies without replacing Kaggle's CUDA-enabled
   PyTorch build.
3. Runs exactly three validation prompts (MBPP `mbpp_542`, GSM8K `gsm_2956`,
   Alpaca `alpaca_183`) with Vanilla Phi and Step-100 `compressed_prompt`.
   Phase 5 is complete; its committed result selects the no-gate condition.
4. Compares each output with its CPU reference, checks checkpoint/predictor
   hashes, CUDA placement, utilization samples, VRAM, and wall time.
5. If the smoke gates pass, uses the smoke as the warm-up and runs two timing
   repeats on the same three prompts.
6. If timing remains faster than the paired CPU references, analyzes the
   12-prompt validation-only predictor/oracle funnel and runs a small
   stratified continuation-safety probe.

Any smoke stop condition prevents all later stages. Kaggle's
`NvidiaTeslaT4` allocation can expose two independent GPUs; this kernel masks
the others and asserts that exactly one CUDA device is visible. There is no
sharding or shared-VRAM assumption.

## Private artifacts

The dataset is private and contains the 6 GB Step-100 checkpoint, the trusted
Oracle-Guided predictor, the three fixed smoke IDs, and selected CPU reference
records. Build the upload folder outside the repository so the checkpoint and
outputs never enter Git. Its `artifact_manifest.json` must include the exact
branch commit, byte sizes, and SHA256 values. The script verifies every hash
before loading a model.

Use `dataset-metadata.json` from this folder with the real Kaggle account slug
and the private dataset slug. Its license is `other`; the description states
that upstream model terms apply and the dataset is not intended for
redistribution. Do not change it to public.

## Reproduction

Check installed CLI help and quota first. `kaggle kernels push` runs the
kernel, so only use it for this already-authorized GPU validation. The
installed CLI's push help is authoritative for flags and accelerator names.

1. Create a private artifact folder outside the repository. The staging script
   requires the checkpoint and CPU references as input and checks their pinned
   hashes before copying them. For example, from the repository root:

   ```powershell
   python -m experiments.kaggle.stage_private_artifacts `
     --checkpoint C:\path\to\checkpoint_step_100.pt `
     --predictor experiments\checkpoints\oracle_guided_predictor.pkl `
     --vanilla-raw-results experiments\checkpoints\quality_benchmark\tier1_runs\81fc94a1eb26e970\raw_results.jsonl `
     --vanilla-run-manifest experiments\checkpoints\quality_benchmark\tier1_runs\81fc94a1eb26e970\run_manifest.json `
     --phase5-raw-results experiments\checkpoints\quality_benchmark\tier1_runs\3de046a1b858dc7a\raw_results.jsonl `
     --phase5-manifest experiments\checkpoints\quality_benchmark\tier1_runs\3de046a1b858dc7a\run_manifest.json `
     --output-dir C:\path\outside\the\repo\tokens-step100-gpu-smoke
   ```

2. Create the private dataset: `kaggle datasets create -p C:\path\outside\the\repo\tokens-step100-gpu-smoke`.
   The staging command writes `dataset-metadata.json` with the configured
   owner and private dataset slug.
3. Put its `owner/dataset-slug` in `kernel/kernel-metadata.json` if the owner
   differs from the configured account.
4. Confirm the metadata says private, GPU enabled, internet enabled, and
   `NvidiaTeslaT4`.
5. Check quota again, then push `kernel/` with the T4 accelerator and a bounded
   runtime.
6. Monitor the exact `owner/kernel-slug`, download outputs, then stop using
   the accelerator. The completed run should leave no active session.

The runtime writes a complete `environment_manifest.json` and the following
reports under `/kaggle/working/tokens-kaggle-output/`:

- `smoke_results.json` and `.md`
- `timing_sanity.json` and `.md`
- `predictor_oracle_funnel.json` and `.md`
- `continuation_safety_gpu.json` and `.md`
- run manifests and raw generation records for the smoke and timing repeats

Copy the reports into `experiments/checkpoints/gpu_smoke/` and update
`RESEARCH_LOG.md` and `experiments/QUALITY_BENCHMARK_METHODOLOGY.md` with CPU
and GPU measurements labeled separately. Never add the private checkpoint or
predictor upload folder to Git.

## Pinned model revisions

- Phi-3.5 base and tokenizer: `2fe192450127e6a83f7441aef6e3ca586c338b77`
- Zip2Zip: `11c461733a79d2a5de6b814585c3361ca2aacbe7`
- Predictive checkpoint step: `100`
- Predictor policy: capped Oracle-Guided predictor, K=32, no structural slots,
  numeric phrases allowed, bare punctuation filtered
- Prompt representation: `predictive_codebook_dp_segmented`

The Kaggle environment's actual Torch, Transformers, CUDA, driver, GPU, and
artifact hashes are written into the output manifest rather than assumed.
