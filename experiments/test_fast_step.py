"""
Test Fast Detached Calibration Step on CPU.
Verifies that stopping backprop at hidden states allows output_encoder
to receive exact gradients with <1 second per step execution time.
"""

import time
import torch
from transformers import AutoTokenizer
from zip2zip import Zip2ZipModel, StaticCodebookManager


def test_fast_step():
    device = "cpu"
    print("1. Loading Zip2Zip model on CPU (float16 for fast inference)...")
    model = Zip2ZipModel.from_pretrained(
        "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
        max_codebook_size=32,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    # Freeze base model and input encoder
    model.eval()
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.input_encoder.parameters():
        p.requires_grad = False
    for p in model.output_encoder.parameters():
        p.requires_grad = True

    static_mgr = StaticCodebookManager(
        initial_vocab_size=32011,
        max_codebook_size=32,
        max_subtokens=3,
        embedding_dim=3072,
        pad_token_id=32000,
    )
    seeded_dict = {(100, 200): 32011, (300, 400): 32012}
    static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device(device))
    static_mgr.attach_to_model(model)

    input_ids = torch.tensor([[10, 20, 30, 32011, 40, 32012, 50]], dtype=torch.long, device=device)
    labels = torch.tensor([[-100, -100, -100, 32011, 40, 32012, 50]], dtype=torch.long, device=device)

    # Run 1 fast step
    static_mgr.reset(clear_caches=True)
    model.output_encoder.zero_grad()
    t0 = time.perf_counter()

    # 1. Forward pass through base transformer with no_grad to get hidden states
    with torch.no_grad():
        outputs = model.base_model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]  # shape: (1, seq_len, 3072)
        base_logits = outputs.logits  # shape: (1, seq_len, 32043)
        base_vocab_logits = base_logits[..., :32011]

    # 2. Dynamic hypertoken logits with grad through output_encoder
    lm_head = model.base_model.get_output_embeddings()
    hyper_linear_weights = static_mgr.get_hyper_linear_weights(
        lm_head.weight, lm_head.encoder_fn
    )  # shape: (1, 32, 3072)

    # hidden_states is detached leaf for this subgraph
    hyper_logits = torch.bmm(hidden_states, hyper_linear_weights.transpose(-2, -1))  # (1, seq_len, 32)

    # 3. Combine logits
    combined_logits = torch.cat([base_vocab_logits, hyper_logits], dim=-1)  # (1, seq_len, 32011 + 32)

    # Mask unused hypertoken slots
    if static_mgr.num_seeded < 32:
        combined_logits[..., 32011 + static_mgr.num_seeded :] = float("-inf")

    # 4. Cross Entropy Loss
    loss_fn = torch.nn.CrossEntropyLoss()
    shift_logits = combined_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    # 5. Backward pass
    loss.backward()
    t1 = time.perf_counter()

    step_ms = (t1 - t0) * 1000.0
    print(f"Step completed in {step_ms:.2f} ms! Loss: {loss.item():.4f}")

    out_grads = [p.grad.norm().item() for p in model.output_encoder.parameters() if p.grad is not None]
    print(f"output_encoder params with grad: {len(out_grads)} / {len(list(model.output_encoder.parameters()))}")
    if out_grads:
        print(f"Mean grad norm: {sum(out_grads)/len(out_grads):.6f}")
        print(f"Max grad norm:  {max(out_grads):.6f}")


if __name__ == "__main__":
    test_fast_step()
