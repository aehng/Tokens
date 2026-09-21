"""
Pre-Training Gradient Audit Script.
Verifies the exact computation graph and gradient flow during training:
1. Verifies that teacher-forced sequences containing seeded hypertoken IDs
   produce valid cross-entropy loss.
2. Checks gradient propagation to:
   - input_encoder parameters
   - output_encoder parameters
   - base_model transformer layers / LM head
3. Tests whether StaticCodebookManager caches interfere with backprop
   and whether slice-assignments in get_hyper_linear_weights / get_hyper_embedding_weights
   maintain requires_grad and allow gradients to reach encoder weights.
4. Audits the dynamic logit output path:
   logits_hyper = x @ W_hyper^T
   Verifies whether hypertoken logits can receive gradients to calibrate emission probability.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from zip2zip import Zip2ZipModel, StaticCodebookManager


def run_gradient_audit():
    print("=" * 80)
    print("PRE-TRAINING GRADIENT FLOW AUDIT")
    print("=" * 80)

    device = "cpu"
    max_k = 32
    initial_vocab_size = 32011

    # 1. Load model in float32 for training audit on CPU
    print("\n1. Loading Zip2Zip model on CPU...")
    model = Zip2ZipModel.from_pretrained(
        "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
        max_codebook_size=max_k,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    # Enable gradients on encoders and base model
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    print(f"Model loaded. input_encoder: {type(model.input_encoder)}")
    print(f"output_encoder: {type(model.output_encoder)}")
    print(f"Tied encoders: {model.zip2zip_config.encoder.tie_encoders}")

    inp_enc_params = list(model.input_encoder.parameters())
    out_enc_params = list(model.output_encoder.parameters()) if model.output_encoder else []
    print(f"input_encoder param count: {sum(p.numel() for p in inp_enc_params):,}")
    print(f"output_encoder param count: {sum(p.numel() for p in out_enc_params):,}")

    # 2. Setup StaticCodebookManager with 4 dummy hypertokens
    # e.g. phrase 1: [100, 200] -> ID 32011
    # phrase 2: [300, 400, 500] -> ID 32012
    # phrase 3: [600, 700] -> ID 32013
    seeded_dict = {
        (100, 200): 32011,
        (300, 400, 500): 32012,
        (600, 700): 32013,
    }
    dim = model.zip2zip_config.encoder.hidden_size
    pad_id = 32000
    disabled_ids = list(model.zip2zip_config.compression.disabled_ids)

    static_mgr = StaticCodebookManager(
        initial_vocab_size=initial_vocab_size,
        max_codebook_size=max_k,
        max_subtokens=3,
        embedding_dim=dim,
        pad_token_id=pad_id,
        disabled_ids=disabled_ids,
    )
    static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device(device))
    static_mgr.attach_to_model(model)

    print("\n2. Attached StaticCodebookManager to model.")
    print(f"Num seeded: {static_mgr.num_seeded}")

    # 3. Create a synthetic training sequence:
    # prompt tokens followed by response tokens containing hypertoken 32011 and 32012
    # Input: [10, 20, 30, 32011, 40, 32012, 50]
    # Target labels: [-100, -100, -100, 32011, 40, 32012, 50]
    input_ids = torch.tensor([[10, 20, 30, 32011, 40, 32012, 50]], dtype=torch.long, device=device)
    labels = torch.tensor([[-100, -100, -100, 32011, 40, 32012, 50]], dtype=torch.long, device=device)

    print("\n3. Testing forward pass with labels...")
    # Invalidate any cached weights so fresh graph is built
    static_mgr.hyper_embedding_weight_cache = None
    static_mgr.hyper_linear_weight_cache = None

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    print(f"Forward pass successful! Loss: {loss.item():.4f}")

    # 4. Backward pass
    print("\n4. Running backward pass...")
    model.zero_grad()
    loss.backward()

    # 5. Check gradient norms
    inp_grad_norms = [p.grad.norm().item() for p in inp_enc_params if p.grad is not None]
    out_grad_norms = [p.grad.norm().item() for p in out_enc_params if p.grad is not None]

    print("\n" + "=" * 80)
    print("GRADIENT AUDIT RESULTS")
    print("=" * 80)
    print(f"input_encoder params with gradients: {len(inp_grad_norms)} / {len(inp_enc_params)}")
    if inp_grad_norms:
        print(f"  Mean grad norm: {sum(inp_grad_norms)/len(inp_grad_norms):.6f}")
        print(f"  Max grad norm:  {max(inp_grad_norms):.6f}")
    else:
        print("  WARNING: NO GRADIENTS reached input_encoder!")

    print(f"output_encoder params with gradients: {len(out_grad_norms)} / {len(out_enc_params)}")
    if out_grad_norms:
        print(f"  Mean grad norm: {sum(out_grad_norms)/len(out_grad_norms):.6f}")
        print(f"  Max grad norm:  {max(out_grad_norms):.6f}")
    else:
        print("  WARNING: NO GRADIENTS reached output_encoder!")

    # Check base model lm_head / layers
    lm_head_weight = model.base_model.get_output_embeddings().weight
    print(f"LM Head base weight grad norm: {lm_head_weight.grad.norm().item() if lm_head_weight.grad is not None else 'None'}")

    embed_tokens_weight = model.base_model.get_input_embeddings().weight
    print(f"Embed tokens base weight grad norm: {embed_tokens_weight.grad.norm().item() if embed_tokens_weight.grad is not None else 'None'}")

    # 6. Test second iteration (simulate training loop step 2 with clear_caches=True)
    print("\n5. Testing iteration 2 (checking cache reset behavior with clear_caches=True)...")
    static_mgr.reset(clear_caches=True)
    outputs2 = model(input_ids=input_ids, labels=labels)
    loss2 = outputs2.loss
    print(f"Iteration 2 Loss: {loss2.item():.4f}")
    model.zero_grad()
    loss2.backward()
    inp_grad2 = [p.grad.norm().item() for p in inp_enc_params if p.grad is not None]
    out_grad2 = [p.grad.norm().item() for p in out_enc_params if p.grad is not None]
    print(f"Iteration 2 input_encoder grads: {len(inp_grad2)} / {len(inp_enc_params)}")
    print(f"Iteration 2 output_encoder grads: {len(out_grad2)} / {len(out_enc_params)}")
    print("SUCCESS: Backward pass succeeded without graph reuse error!")

    # 7. Test training speed with frozen base model (hyper-encoders only)
    print("\n6. Testing gradient flow with frozen base model (tuning output_encoder only)...")
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.output_encoder.parameters():
        p.requires_grad = True

    static_mgr.reset(clear_caches=True)
    model.zero_grad()
    t_start = time.perf_counter()
    out3 = model(input_ids=input_ids, labels=labels)
    loss3 = out3.loss
    loss3.backward()
    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    out_grad3 = [p.grad.norm().item() for p in out_enc_params if p.grad is not None]
    print(f"Frozen base step took: {t_elapsed_ms:.1f}ms")
    print(f"output_encoder params with grad: {len(out_grad3)} / {len(out_enc_params)}")
    if out_grad3:
        print(f"  Mean grad norm: {sum(out_grad3)/len(out_grad3):.6f}")

    static_mgr.detach_from_model(model)
    print("\nAudit complete.")


if __name__ == "__main__":
    run_gradient_audit()
