# A/B0 generation-fidelity result

**Result: 12/12 exact greedy generations matched.** This establishes generation-level fidelity on the pinned smoke set. The separate matched-prefix logit comparison did not complete, so full distribution-level wrapper fidelity remains unverified.

## Conditions and provenance

- **A:** native `microsoft/Phi-3.5-mini-instruct`, revision `2fe192450127e6a83f7441aef6e3ca586c338b77`.
- **B0:** Tokens wrapper around that same pinned Phi revision and weights, with the PEFT adapter not loaded, the Step-100 checkpoint not applied, an empty static codebook, and H logits masked.
- **Tokenizer:** same model ID and revision as A. Chat-template SHA-256: `78d976a442bcde2f0be15aafbb8e3050e1104f86732266f68403251b89982a90`.
- **Runtime:** `torch.float16`; attention `Phi3Attention;config=sdpa`; EOS IDs `[32007, 32001, 32000]`; greedy generation (`do_sample=false`), `max_new_tokens=1024`, pad ID `32000`.
- **Base weight fingerprint:** `f4d2caf4c236b50267ef2299cf6f71a4cf605e73928c4504274b7989260b1b25`, equal between A and B0 for every prompt.
- **Source:** commit `f59ef6319a1b9989c1154fd7349a9f6f6a41bb05`, branch `grok/predictive-fidelity-audit`, dirty worktree as recorded by the run; source archive SHA-256 `f78731bbe8a17f29dce037b544a909a14967c7e7ca116bfeac26b32c50367e7c`.
- **Archive recovery caveat:** the source archive itself was not present in the recovered outputs. Kaggle's run records report the archive hash and successful extracted-file hash verification, but the archive hash could not be independently recomputed from the recovered files.
- **Dataset provenance:** manifest SHA-256 `f05966cbe456ff7b8da52693cad4f74d8a7a87c99e62c96d37d39663a1ac7ba8`; canonical dataset SHA-256 `0d3bea57de3ecac982da9ec65db97850cdfbaf5a57c0ace5def05f686ccf4c68`.
- **GPU:** Kaggle NVIDIA Tesla T4 x2; the runner used `cuda:0`. The two T4 devices are separate 14.56 GiB GPUs, not a shared memory pool.

The machine-readable evidence file contains the exact rendered prompt strings, exact prompt token-ID arrays, generation settings, and A/B0 generated token arrays for every row: [`offline_equivalence_evidence.json`](../experiments/results/phi_ab0_fidelity_dev12_20260929/offline_equivalence_evidence.json). The original 24 incremental records, completion manifest, and the two distinct log captures (Version 3 OOM and an earlier TorchAO/PEFT preflight failure) are preserved alongside it. The captured logs establish that all 12 A and B0 generations completed before the later logit-stage OOM; they do not identify the owner of the retained GPU tensors.

## Per-prompt result

| Prompt | Rendered prompt SHA-256 | Prompt token-ID SHA-256 | Generated tokens | A/B0 generated IDs | Termination |
|---|---|---|---:|---|---|
| `mbpp_113` | `7e2a37dec9495dcd…` | `9725a26a2999b142…` | 449 | Exact | `eos`, `32007` |
| `mbpp_168` | `f9a61184d3198583…` | `86a4f1220df5383d…` | 543 | Exact | `eos`, `32007` |
| `mbpp_217` | `0ef71f6a09cf77b9…` | `6c81a28a395ac299…` | 304 | Exact | `eos`, `32007` |
| `mbpp_225` | `71ce9d4132457d31…` | `d8309fc005adb5ef…` | 545 | Exact | `eos`, `32007` |
| `gsm_2032` | `af7fb0e767b02296…` | `389234327cf71109…` | 200 | Exact | `eos`, `32007` |
| `gsm_2044` | `5e21cbcf9db0ce47…` | `f1c934ea1d093f6c…` | 378 | Exact | `eos`, `32007` |
| `gsm_2353` | `fef54acf3a5da424…` | `82fc87f171b5feee…` | 340 | Exact | `eos`, `32007` |
| `gsm_2491` | `6c40390adda19457…` | `64b9db530f38137b…` | 489 | Exact | `eos`, `32007` |
| `alpaca_1` | `8caaa217944d606b…` | `d36b05b5440e7044…` | 147 | Exact | `eos`, `32007` |
| `alpaca_1024` | `b317c02389cd9f17…` | `e0b0593397849263…` | 428 | Exact | `eos`, `32007` |
| `alpaca_1029` | `d378a57baa5e0ca0…` | `46bd0a5a66daacf6…` | 226 | Exact | `eos`, `32007` |
| `alpaca_1132` | `0299d71b97c55509…` | `a08232fe20712f25…` | 542 | Exact | `eos`, `32007` |

Across all 12 prompts, rendered text and input token IDs match exactly. Generated IDs, expanded IDs, token counts, EOS-reached flags, termination reasons, and termination token IDs match exactly. The records report `adapter_load_requested=false`, `adapter_present_in_wrapper=false`, no active adapter names, `checkpoint_applied=false`, `step100_h_encoder_checkpoint_loaded=false`, `h_seeded_slots=0`, static codebook backend, and H-logit masking enabled.

## Logit-stage failure and interpretation

The run finished A and B0 generation for all 12 prompts, then failed while the matched-prefix routine attempted to load Vanilla Phi again. Kaggle reported an allocation request of 96 MiB with 76.81 MiB free and 14.35 GiB already allocated by PyTorch on GPU 0. No matched-prefix logit vectors or model-teardown snapshots were written.

The log proves that the GPU was still full when the second Vanilla load began. It does **not** identify which live object retained the memory. In particular, it does not prove that a `StaticCodebookManager._previous_manager` chain caused the retention. The CPU implementation now reuses one empty B0 manager and includes teardown guards, but this historical run did not capture the diagnostics needed to attribute its allocation to a specific object.

Therefore these metrics are **not available** for this run: maximum/mean logit error, KL divergence, top-1 or top-k logit agreement, top-1 margin difference, EOS/special-token logit differences, and numerical tolerance assessment. The A/B0 token gate passed; the stricter logit gate remains incomplete.
