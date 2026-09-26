# vLLM predictive proof, kernel v12

**Result: Phase 8 failed its instrumentation completeness gate.** The actual
rotary call received the expected positions and generated IDs matched the
reference. Stage E was empty because its forward pre-hook did not run.

## Provenance

- Source commit: `03261da32e81981ef596403dd0f38e80f8fea45d`
- Source archive SHA-256: `29a5e11775d51a1b6532c5b52a7ccd446604ec2833eb79b9140daf70c6ac4265`
- Runtime: vLLM `0.30.0`, PyTorch `2.13.0+cu130`, CUDA `13.0`, Tesla T4
- Mode: targeted Phase 8 only; diagnostic parity disabled

The JSON files in this directory are the unedited Kaggle reports for this run.

## Observed result

For request `phase8-chunk`, scheduler progress was `0, 16, 32`. Calculated
semantic positions, the returned model positions, the predictive wrapper
(stage D), and the actual layer-0 RoPE call (stage F) each concatenated to
`0..41`. Generated token IDs matched the HF reference `16/16`. All captured
events belonged to the target request.

The report is correctly marked **FAIL** because stage E, intended to capture
`LlamaModel.forward`, was empty. This run does not count as a Phase 8 pass.

## Root cause and follow-up

The vLLM `v0.30.0` `LlamaModel` is decorated with
`support_torch_compile`. Its eager `do_not_compile` path calls
`self.forward(...)` directly, bypassing `nn.Module.__call__` and its
registered forward hooks. The pinned sources show this in
[`decorators.py`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/compilation/decorators.py#L465-L470)
and show `LlamaModel.forward` passing the same positions to each decoder
layer in [`llama.py`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/llama.py#L379-L410).

This was a diagnostic-hook miss, not an observed position substitution in the
runtime. The follow-up wraps the nested model's `forward` method itself for
stage E, restores the original method after each request, and adds a CPU test
that reproduces vLLM's direct-forward call path. No production position
calculation or KV-cache position handling is changed by that correction.
