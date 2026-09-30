"""Context-Conditioned Hypertoken Representation Engine for Frozen Vanilla Phi-3.5.

Enforces:
  1. Base model is 100% frozen: zero gradient updates, zero LoRA, zero PEFT.
  2. Single-slot physical cache: student has exactly ONE physical cache entry
     for H(A, B) at semantic position C+1.
  3. Stable initialization: delta_H initialized to zero so H initially equals
     mean(embedding(A), embedding(B)).
  4. Teacher/student continuation loss comparing logits at matched semantic
     positions: immediately after H (offset 0), +1, +2, +4, +8, +16.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_model_parameter_hash(model: nn.Module) -> str:
    """Compute SHA256 digest of all parameter tensors in model."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        hasher.update(name.encode("utf-8"))
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def assert_model_strictly_frozen(model: nn.Module) -> None:
    """Assert that every parameter in model has requires_grad=False."""
    unfrozen = [name for name, param in model.named_parameters() if param.requires_grad]
    if unfrozen:
        raise AssertionError(
            f"Base model contains unfrozen parameters: {unfrozen[:5]} (total {len(unfrozen)})"
        )


class ContextConditionedHEncoder(nn.Module):
    """Encodes a 2-token phrase [A, B] into a single hypertoken embedding H.

    Conditioned on:
      - h_ctx: Context-final hidden state from the last layer of Phi (dim D)
      - e_A: Token A base embedding (dim D)
      - e_B: Token B base embedding (dim D)

    Architecture:
      H_base = 0.5 * (e_A + e_B)
      delta_H = tanh(gate) * MLP([LayerNorm(h_ctx, e_A, e_B)])
      H = H_base + delta_H

    Initial state:
      gate is initialized to 0.0, so at step 0: delta_H = 0 and H == H_base.
    """

    def __init__(
        self,
        embed_dim: int = 3072,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Input dimension is 3 * embed_dim: [h_ctx, e_A, e_B]
        in_dim = 3 * embed_dim
        self.input_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

        # Zero-initialized scalar gate for residual learning
        self.gate = nn.Parameter(torch.zeros(1))

        # Small projection initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        h_ctx: torch.Tensor,
        embed_a: torch.Tensor,
        embed_b: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass producing hypertoken embedding H.

        Args:
          h_ctx: [batch, embed_dim] context-final hidden state
          embed_a: [batch, embed_dim] embedding of token A
          embed_b: [batch, embed_dim] embedding of token B

        Returns:
          H: [batch, embed_dim] hypertoken embedding
        """
        if h_ctx.dim() == 1:
            h_ctx = h_ctx.unsqueeze(0)
            embed_a = embed_a.unsqueeze(0)
            embed_b = embed_b.unsqueeze(0)

        in_dtype = h_ctx.dtype
        enc_dtype = self.input_norm.weight.dtype

        # Base combination
        h_base = 0.5 * (embed_a + embed_b)

        # Context-conditioned delta
        x = torch.cat([h_ctx, embed_a, embed_b], dim=-1).to(enc_dtype)
        x_norm = self.input_norm(x)
        hidden = self.dropout(self.act(self.fc1(x_norm)))
        delta = self.fc2(hidden)

        # Gated residual addition
        gate_val = torch.tanh(self.gate)
        h = h_base + (gate_val * delta).to(in_dtype)
        return h.to(in_dtype)


# Semantic continuation evaluation offsets
EVAL_OFFSETS: Tuple[int, ...] = (0, 1, 2, 4, 8, 16)
EVAL_OFFSET_WEIGHTS: Dict[int, float] = {
    0: 1.0,  # Immediate next token after H (predicts y_0)
    1: 0.8,  # +1 token (predicts y_1)
    2: 0.6,  # +2 tokens (predicts y_2)
    4: 0.5,  # +4 tokens
    8: 0.4,  # +8 tokens (substantial long-range weight)
    16: 0.3, # +16 tokens
}


def continuation_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid_offsets: Sequence[int],
    weights: Optional[Mapping[int, float]] = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """Compute weighted KL(teacher || student) across semantic continuation offsets.

    Args:
      teacher_logits: [num_offsets, vocab_size] teacher logits at offsets
      student_logits: [num_offsets, vocab_size] student logits at offsets
      valid_offsets: List of integer offset indices present in the tensors
      weights: Mapping from offset to loss weight
      temperature: Softmax temperature

    Returns:
      total_weighted_kl: scalar tensor loss
      per_offset_kl: dict mapping offset to float KL value in nats
    """
    if weights is None:
        weights = EVAL_OFFSET_WEIGHTS

    loss_terms: List[torch.Tensor] = []
    weight_terms: List[float] = []
    per_offset_kl: Dict[int, float] = {}

    for idx, offset in enumerate(valid_offsets):
        w = float(weights.get(offset, 0.5))
        t_vec = teacher_logits[idx].float() / temperature
        s_vec = student_logits[idx].float() / temperature

        # Numerical stability: log_softmax and softmax in fp32
        p_teacher = F.softmax(t_vec, dim=-1)
        log_p_student = F.log_softmax(s_vec, dim=-1)

        # Exact KL(P || Q) = sum(P * (log P - log Q)) in nats
        kl_nats = F.kl_div(log_p_student, p_teacher, reduction="sum", log_target=False)
        per_offset_kl[offset] = float(kl_nats.detach().item())

        loss_terms.append(w * kl_nats)
        weight_terms.append(w)

    if not loss_terms:
        return torch.tensor(0.0, device=teacher_logits.device, requires_grad=True), {}

    total_weight = sum(weight_terms)
    total_loss = sum(loss_terms) / max(total_weight, 1e-6)
    return total_loss, per_offset_kl


@dataclass
class TeacherStudentForwardResult:
    total_loss: torch.Tensor
    kl_loss: torch.Tensor
    state_loss: torch.Tensor
    per_offset_kl: Dict[int, float]
    per_offset_top1_match: Dict[int, bool]
    immediate_top1_match: bool
    immediate_kl_nats: float
    teacher_logits: Dict[int, torch.Tensor]
    student_logits: Dict[int, torch.Tensor]
    h_embedding: torch.Tensor
    context_length: int
    phrase_tokens: Tuple[int, int]
    future_tokens: List[int]


def execute_teacher_student_step(
    phi_model: Any,
    h_encoder: ContextConditionedHEncoder,
    context_ids: Sequence[int],
    token_a: int,
    token_b: int,
    future_token_ids: Sequence[int],
    device: torch.device,
    *,
    offsets: Sequence[int] = EVAL_OFFSETS,
    offset_weights: Optional[Mapping[int, float]] = None,
    state_preservation_weight: float = 0.1,
) -> TeacherStudentForwardResult:
    """Execute one full Teacher/Student forward step on frozen Phi-3.5.

    Position Semantics:
      Context length C: tokens at 0 .. C-1
      Teacher:
        A at position C
        B at position C+1
        Future y_0 .. y_{T-1} at positions C+2 .. C+1+T
        Teacher sequence length: C + 2 + T
        Prediction of y_k is at position C + 1 + k
      Student:
        H at semantic position C+1 (physical index C)
        Future y_0 .. y_{T-2} at semantic positions C+2 .. C+T (physical indices C+1 .. C+T-1)
        Student sequence length: C + 1 + T (exactly 1 physical slot shorter!)
        Prediction of y_k is at physical index C + k (semantic position C + 1 + k)
    """
    C = len(context_ids)
    T = len(future_token_ids)
    if T < 1:
        raise ValueError("future_token_ids must contain at least 1 token")

    # Filter requested offsets that fall within future length
    valid_offsets = [k for k in offsets if k < T]
    if not valid_offsets:
        valid_offsets = [0]

    # Embeddings layer
    embed_tokens = phi_model.model.embed_tokens

    # 1. TEACHER FORWARD (strictly no_grad)
    teacher_seq = list(context_ids) + [token_a, token_b] + list(future_token_ids)
    teacher_input_ids = torch.tensor([teacher_seq], dtype=torch.long, device=device)
    teacher_positions = torch.arange(0, len(teacher_seq), dtype=torch.long, device=device).unsqueeze(0)
    teacher_mask = torch.ones_like(teacher_input_ids)

    with torch.no_grad():
        t_out = phi_model(
            input_ids=teacher_input_ids,
            position_ids=teacher_positions,
            attention_mask=teacher_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        t_logits = t_out.logits[0]  # [C + 2 + T, vocab_size]
        t_hidden = t_out.hidden_states[-1][0]  # [C + 2 + T, embed_dim]

        # Extract teacher logits and hidden states at matched offsets
        # Offset k predicts y_k at position C + 1 + k
        t_logits_at_offsets = torch.stack([t_logits[C + 1 + k] for k in valid_offsets])
        t_hidden_at_offsets = torch.stack([t_hidden[C + 1 + k] for k in valid_offsets])

        # Extract context-final hidden state and token embeddings for H encoder
        h_ctx = t_hidden[C - 1].unsqueeze(0)  # [1, embed_dim]
        embed_a = embed_tokens(torch.tensor([token_a], device=device))  # [1, embed_dim]
        embed_b = embed_tokens(torch.tensor([token_b], device=device))  # [1, embed_dim]

    # 2. STUDENT FORWARD (Gradients flow ONLY through H into h_encoder)
    # Generate hypertoken embedding
    H = h_encoder(h_ctx, embed_a, embed_b)  # [1, embed_dim]

    # Build student inputs_embeds
    with torch.no_grad():
        ctx_embeds = embed_tokens(torch.tensor([context_ids], device=device))  # [1, C, embed_dim]
        if T > 1:
            # Future tokens fed as inputs are y_0 .. y_{T-2} (predicting y_1 .. y_{T-1})
            fut_input_ids = list(future_token_ids[:-1])
            fut_embeds = embed_tokens(torch.tensor([fut_input_ids], device=device))  # [1, T-1, embed_dim]
        else:
            fut_embeds = torch.empty((1, 0, h_encoder.embed_dim), device=device)

    # Student inputs_embeds: [ctx, H, y_0 .. y_{T-2}] -> length C + 1 + (T - 1) = C + T
    # Wait, student physical length:
    # If student inputs [ctx, H, y_0, ..., y_{T-2}], it predicts at indices:
    # index C (H): predicts y_0 (offset 0)
    # index C+1 (y_0): predicts y_1 (offset 1)
    # index C+k (y_{k-1}): predicts y_k (offset k)
    # So to evaluate offset k, student needs input token at index C + k - 1!
    # For offset 0: input is H at index C -> predicts y_0.
    # For max_offset K: student needs inputs up to index C + K - 1 -> requires future tokens y_0 .. y_{K-1}.
    max_k = max(valid_offsets)
    needed_future_inputs = list(future_token_ids[:max_k])
    if needed_future_inputs:
        with torch.no_grad():
            fut_input_embeds = embed_tokens(torch.tensor([needed_future_inputs], device=device))
        student_embeds = torch.cat([ctx_embeds, H.unsqueeze(1), fut_input_embeds], dim=1)
    else:
        student_embeds = torch.cat([ctx_embeds, H.unsqueeze(1)], dim=1)

    # Student explicit semantic position IDs:
    # context: 0 .. C-1
    # H: C+1 (semantic position matching token B in teacher)
    # future tokens y_0 .. y_{max_k-1}: C+2 .. C+1+max_k
    ctx_pos = torch.arange(0, C, dtype=torch.long, device=device)
    h_pos = torch.tensor([C + 1], dtype=torch.long, device=device)
    fut_pos = torch.arange(C + 2, C + 2 + len(needed_future_inputs), dtype=torch.long, device=device)
    student_positions = torch.cat([ctx_pos, h_pos, fut_pos]).unsqueeze(0)
    student_mask = torch.ones((1, student_embeds.shape[1]), dtype=torch.long, device=device)

    # Forward through frozen Phi
    s_out = phi_model(
        inputs_embeds=student_embeds,
        position_ids=student_positions,
        attention_mask=student_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    s_logits = s_out.logits[0]  # [student_len, vocab_size]
    s_hidden = s_out.hidden_states[-1][0]  # [student_len, embed_dim]

    # Student logits and hidden states at matched offsets:
    # Offset k is predicted at physical index C + k
    s_logits_at_offsets = torch.stack([s_logits[C + k] for k in valid_offsets])
    s_hidden_at_offsets = torch.stack([s_hidden[C + k] for k in valid_offsets])

    # 3. LOSS COMPUTATION
    kl_loss, per_offset_kl = continuation_kl_loss(
        t_logits_at_offsets, s_logits_at_offsets, valid_offsets, weights=offset_weights
    )

    # State preservation loss: 1 - cosine_similarity(t_hidden, s_hidden)
    cos_sim = F.cosine_similarity(t_hidden_at_offsets.float(), s_hidden_at_offsets.float(), dim=-1)
    state_loss = (1.0 - cos_sim).mean()

    total_loss = kl_loss + state_preservation_weight * state_loss

    # Top-1 agreement metrics
    per_offset_top1: Dict[int, bool] = {}
    for idx, offset in enumerate(valid_offsets):
        top1_t = int(t_logits_at_offsets[idx].argmax().item())
        top1_s = int(s_logits_at_offsets[idx].argmax().item())
        per_offset_top1[offset] = bool(top1_t == top1_s)

    t_logits_dict = {
        offset: t_logits_at_offsets[idx].detach().cpu()
        for idx, offset in enumerate(valid_offsets)
    }
    s_logits_dict = {
        offset: s_logits_at_offsets[idx].detach().cpu()
        for idx, offset in enumerate(valid_offsets)
    }

    return TeacherStudentForwardResult(
        total_loss=total_loss,
        kl_loss=kl_loss.detach(),
        state_loss=state_loss.detach(),
        per_offset_kl=per_offset_kl,
        per_offset_top1_match=per_offset_top1,
        immediate_top1_match=per_offset_top1.get(0, False),
        immediate_kl_nats=per_offset_kl.get(0, 0.0),
        teacher_logits=t_logits_dict,
        student_logits=s_logits_dict,
        h_embedding=H.detach(),
        context_length=C,
        phrase_tokens=(token_a, token_b),
        future_tokens=list(future_token_ids),
    )
