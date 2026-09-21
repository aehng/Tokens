import time
import torch
from zip2zip import Zip2ZipModel, StaticCodebookManager

print("Testing step speed when input_encoder and base_model are frozen...")
model = Zip2ZipModel.from_pretrained(
    "epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1",
    max_codebook_size=32,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
).to("cpu")

# Freeze both base_model and input_encoder
for p in model.base_model.parameters():
    p.requires_grad = False
for p in model.input_encoder.parameters():
    p.requires_grad = False
for p in model.output_encoder.parameters():
    p.requires_grad = True

model.base_model.lm_head.detach_input = True

static_mgr = StaticCodebookManager(
    initial_vocab_size=32011,
    max_codebook_size=32,
    max_subtokens=3,
    embedding_dim=3072,
    pad_token_id=32000,
)
seeded_dict = {(100, 200): 32011, (300, 400): 32012}
static_mgr.set_seeded_codebook(seeded_dict, batch_size=1, device=torch.device("cpu"))
static_mgr.attach_to_model(model)

input_ids = torch.tensor([[10, 20, 30, 32011, 40, 32012, 50]], dtype=torch.long)
labels = torch.tensor([[-100, -100, -100, 32011, 40, 32012, 50]], dtype=torch.long)

# Run 3 steps to measure real step time
for step in range(3):
    static_mgr.reset(clear_caches=True)
    model.zero_grad()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    loss.backward()
    t1 = time.perf_counter()
    out_grads = [p.grad.norm().item() for p in model.output_encoder.parameters() if p.grad is not None]
    print(f"Step {step+1}: {(t1-t0)*1000:.1f} ms | Loss: {loss.item():.4f} | Grads: {len(out_grads)}/21 | Mean Grad: {sum(out_grads)/len(out_grads):.4f}")
