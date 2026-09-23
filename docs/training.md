# Training a `zip2zip` model

> **Scope note:** The TRL example below is an upstream Zip2Zip-style finetuning example. It is not the project's prompt-only predictive training pipeline and is not an instruction to full-finetune the customer model. Follow the [canonical research roadmap](../experiments/RESEARCH_ROADMAP.md) for the current phase order. Qwen/vLLM work is a later-stage target documented in [`QWEN3_VLLM_PRODUCTION_VALIDATION.md`](QWEN3_VLLM_PRODUCTION_VALIDATION.md).

## Project predictive-training data contract

For the project's predictive path (as distinct from the upstream example below):

- The predictor sees the prompt only. Response content may be used to construct training targets or determine whether already-selected phrases occur, but must not leak into prediction features.
- Training input uses the prompt representation intended for inference. If a compressed prompt is used for calibration, serving must provide that same representation; verify this with a raw-versus-compressed A/B.
- Targets are the compressed response **followed by EOS**. Assert that EOS is present in the labels and evaluate normal stopping, not just loss or decoded text.
- Verify exact response expansion and continuation equivalence, including KL/top-k/next-token agreement and coherent multi-token continuation after a hypertoken.
- Keep training, policy-tuning validation, and fresh holdout data separate. Do not tune against the final holdout.

## Planned Qwen port constraints

Retokenize with the pinned Qwen tokenizer and use task-valid formatting with a meaningful, balanced set of code, reasoning, and instruction/general examples. Do not judge the architecture from roughly 100 examples. Each training sequence uses the compressed prompt as context followed by compressed response and EOS; mask prompt labels as appropriate, and ensure response content never enters predictor features. Prefer a frozen Qwen base with small PEFT/LoRA and hypermodules plus short calibration; do not full-finetune Qwen3-8B for this validation. Before scaling, gate on forward/backward, gradient flow, checkpoint save/reload, unchanged base hashes, compressed-prompt use, EOS, at least one emitted predictive hypertoken, and coherent continuation. Do not assume Phi modules or representations transfer. The vLLM integration and benchmark remain planned, not verified; follow the canonical research roadmap for sequencing and the historical production-validation snapshot for Qwen-specific compatibility requirements.

## Finetuning using [TRL](https://github.com/huggingface/trl)

```python
from zip2zip import (
    Zip2ZipModel,
    Zip2ZipTokenizer,
    Zip2ZipConfig,
    EncoderType,
    TransformerEncoderConfig,
    CompressionConfig,
)

import torch
from datasets import load_dataset
from accelerate import Accelerator
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType

current_device = Accelerator().process_index

train_dataset = load_dataset(
    "epfl-dlab/zip2zip-1B", name="default", split="train"
).take(300_000)
eval_dataset = load_dataset(
    "epfl-dlab/zip2zip-1B", name="default", split="validation"
).take(250)

config = Zip2ZipConfig(
    "microsoft/Phi-3.5-mini-instruct",
    encoder_type=EncoderType.TRANSFORMER,
    encoder=TransformerEncoderConfig(
        hidden_size=3072,
        tie_encoders=True,
        num_hidden_layers=2,
        intermediate_size=12288,
        num_heads=32,
    ),
    compression=CompressionConfig(
        initial_vocab_size=32011,
        max_codebook_size=2048,
        max_subtokens=4,
    ),
)

model = Zip2ZipModel(
    config,
    peft_config=LoraConfig(
        r=32,
        lora_alpha=32,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "qkv_proj",
            "o_proj",
            "qkv_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
    ),
    device_map={"": current_device},
    torch_dtype=torch.bfloat16,
)

tokenizer = Zip2ZipTokenizer(config)

trainer_args = SFTConfig(
    max_length=2048,
    output_dir="zip2zip-train-debug/",
    packing=False,
    torch_compile=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    eval_strategy="steps",
    eval_steps=100,
    data_seed=42,
    max_steps=18_000,
    learning_rate=1e-5,
    warmup_steps=1_000,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 1e-6},
)

trainer = SFTTrainer(
    model,
    args=trainer_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()

trainer.save_model("zip2zip-train-debug-final/")

```
