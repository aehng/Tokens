"""Oracle Per-Example H Representation Engine (Phase 1).

Scientific Question:
For a specific fixed context and specific two-token phrase [A, B], does there
exist ANY single 3072-dimensional input embedding H such that completely
frozen Vanilla Phi behaves approximately as though it had processed A and B
normally?

This engine removes encoder architecture, dataset generalization, and predictor
quality completely by optimizing H directly as an independent 3072-D trainable vector
per example.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from zip2zip.frozen_phi_h import (
    EVAL_OFFSETS,
    assert_model_strictly_frozen,
    compute_model_parameter_hash,
)

# Continuation offset loss weights strongly protecting Offset 0
ORACLE_STAGE_B_WEIGHTS: Dict[int, float] = {
    0: 8.0,   # Immediate next token after H (predicts y_0) - strongly protected
    1: 4.0,   # +1 token (predicts y_1)
    2: 2.0,   # +2 tokens (predicts y_2)
    4: 1.0,   # +4 tokens
    8: 0.5,   # +8 tokens
    16: 0.25, # +16 tokens
}


@dataclass
class TeacherReferenceData:
    context_ids: List[int]
    token_a: int
    token_b: int
    future_ids: List[int]
    logits_at_offsets: Dict[int, torch.Tensor]  # offset -> logit tensor [vocab_size]
    hidden_at_offsets: Dict[int, torch.Tensor]  # offset -> hidden tensor [embed_dim]
    top1_at_offsets: Dict[int, int]             # offset -> top-1 token id
    vanilla_greedy_32: List[int]                # 32 greedy tokens generated after context + A + B


def compute_teacher_reference(
    phi_model: Any,
    context_ids: Sequence[int],
    token_a: int,
    token_b: int,
    future_token_ids: Sequence[int],
    device: torch.device,
    offsets: Sequence[int] = EVAL_OFFSETS,
    max_rollout: int = 32,
) -> TeacherReferenceData:
    """Precompute Teacher outputs and greedy rollout on frozen Vanilla Phi."""
    phi_model.eval()
    embed_tokens = phi_model.model.embed_tokens
    C = len(context_ids)
    max_k = max(offsets)
    needed_future = list(future_token_ids[:max_k])
    
    # Teacher input: [context, A, B, future_0 .. future_{max_k-1}]
    teacher_ids = list(context_ids) + [token_a, token_b] + needed_future
    t_tensor = torch.tensor([teacher_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        out = phi_model(
            input_ids=t_tensor,
            output_hidden_states=True,
            use_cache=False,
        )
        logits = out.logits[0]
        hidden = out.hidden_states[-1][0]
        
        logits_dict: Dict[int, torch.Tensor] = {}
        hidden_dict: Dict[int, torch.Tensor] = {}
        top1_dict: Dict[int, int] = {}
        
        # In teacher:
        # index C is token A -> predicts B
        # index C+1 is token B -> predicts y_0 (Offset 0)
        # index C+1+k is token y_{k-1} -> predicts y_k (Offset k)
        for k in offsets:
            phys_idx = C + 1 + k
            logits_dict[k] = logits[phys_idx].detach().cpu()
            hidden_dict[k] = hidden[phys_idx].detach().cpu()
            top1_dict[k] = int(logits[phys_idx].argmax().item())

    # Precompute Teacher Vanilla greedy rollout of length max_rollout
    # Teacher context is [context, A, B]
    prefix_ids = list(context_ids) + [token_a, token_b]
    vanilla_rollout = generate_greedy_rollout(
        phi_model, prefix_ids, device, max_tokens=max_rollout
    )

    return TeacherReferenceData(
        context_ids=list(context_ids),
        token_a=token_a,
        token_b=token_b,
        future_ids=list(future_token_ids),
        logits_at_offsets=logits_dict,
        hidden_at_offsets=hidden_dict,
        top1_at_offsets=top1_dict,
        vanilla_greedy_32=vanilla_rollout,
    )


def student_forward_with_oracle_h(
    phi_model: Any,
    context_ids: Sequence[int],
    H_param: torch.Tensor,
    device: torch.device,
    future_token_ids: Optional[Sequence[int]] = None,
    offsets: Sequence[int] = (0,),
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Forward Student with free trainable parameter H occupying ONE physical slot.

    Semantic positions:
      context: 0 .. C-1
      H: C+1 (matching semantic position of token B in teacher)
      future: C+2 .. C+1+K
    Physical indices:
      context: 0 .. C-1
      H: C (strictly 1 slot!)
      future: C+1 .. C+K
    """
    C = len(context_ids)
    embed_tokens = phi_model.model.embed_tokens
    embed_dtype = embed_tokens.weight.dtype
    
    with torch.no_grad():
        ctx_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
        ctx_embeds = embed_tokens(ctx_tensor)  # [1, C, D]

    # Ensure H maintains gradients, cast to model dtype for embedding tensor
    H_vec = H_param.to(embed_dtype)
    if H_vec.dim() == 1:
        H_embed = H_vec.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
    elif H_vec.dim() == 2:
        H_embed = H_vec.unsqueeze(0)  # [1, 1, D]
    else:
        H_embed = H_vec

    max_k = max(offsets)
    if max_k > 0 and future_token_ids:
        needed_future = list(future_token_ids[:max_k])
        with torch.no_grad():
            fut_tensor = torch.tensor([needed_future], dtype=torch.long, device=device)
            fut_embeds = embed_tokens(fut_tensor)  # [1, len(needed), D]
        student_embeds = torch.cat([ctx_embeds, H_embed, fut_embeds], dim=1)
        fut_len = len(needed_future)
    else:
        student_embeds = torch.cat([ctx_embeds, H_embed], dim=1)
        fut_len = 0

    # Semantic position IDs:
    ctx_pos = torch.arange(0, C, dtype=torch.long, device=device)
    h_pos = torch.tensor([C + 1], dtype=torch.long, device=device)
    if fut_len > 0:
        fut_pos = torch.arange(C + 2, C + 2 + fut_len, dtype=torch.long, device=device)
        student_positions = torch.cat([ctx_pos, h_pos, fut_pos]).unsqueeze(0)
    else:
        student_positions = torch.cat([ctx_pos, h_pos]).unsqueeze(0)

    student_mask = torch.ones((1, student_embeds.shape[1]), dtype=torch.long, device=device)

    s_out = phi_model(
        inputs_embeds=student_embeds,
        position_ids=student_positions,
        attention_mask=student_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    s_logits = s_out.logits[0]
    s_hidden = s_out.hidden_states[-1][0]

    # Student predictions:
    # index C (H) predicts y_0 (Offset 0)
    # index C+k (y_{k-1}) predicts y_k (Offset k)
    logits_dict: Dict[int, torch.Tensor] = {}
    hidden_dict: Dict[int, torch.Tensor] = {}
    for k in offsets:
        phys_idx = C + k
        logits_dict[k] = s_logits[phys_idx]
        hidden_dict[k] = s_hidden[phys_idx]

    return logits_dict, hidden_dict


def generate_greedy_rollout(
    phi_model: Any,
    initial_ids: Sequence[int],
    device: torch.device,
    max_tokens: int = 32,
    eos_token_id: int = 32007,
) -> List[int]:
    """Perform greedy autoregressive generation for reference comparison."""
    phi_model.eval()
    curr_ids = list(initial_ids)
    inp = torch.tensor([curr_ids], dtype=torch.long, device=device)
    pos = torch.arange(0, len(curr_ids), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        out = phi_model(input_ids=inp, position_ids=pos, use_cache=True)
        past_kv = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax().item())

    generated = [next_tok]
    if next_tok == eos_token_id:
        return generated

    for _ in range(max_tokens - 1):
        inp = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        pos = torch.tensor([[len(curr_ids) + len(generated) - 1]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = phi_model(input_ids=inp, position_ids=pos, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = int(out.logits[0, -1].argmax().item())
        generated.append(next_tok)
        if next_tok == eos_token_id:
            break

    return generated


def generate_student_rollout_from_h(
    phi_model: Any,
    context_ids: Sequence[int],
    H_vec: torch.Tensor,
    device: torch.device,
    max_tokens: int = 32,
    eos_token_id: int = 32007,
) -> List[int]:
    """Perform autoregressive greedy generation starting immediately after [context, H]."""
    phi_model.eval()
    C = len(context_ids)
    embed_tokens = phi_model.model.embed_tokens
    embed_dtype = embed_tokens.weight.dtype

    with torch.no_grad():
        ctx_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
        ctx_embeds = embed_tokens(ctx_tensor)
        h_embed = H_vec.to(device=device, dtype=embed_dtype).unsqueeze(0).unsqueeze(0)
        init_embeds = torch.cat([ctx_embeds, h_embed], dim=1)

        ctx_pos = torch.arange(0, C, dtype=torch.long, device=device)
        h_pos = torch.tensor([C + 1], dtype=torch.long, device=device)
        init_pos = torch.cat([ctx_pos, h_pos]).unsqueeze(0)

        out = phi_model(inputs_embeds=init_embeds, position_ids=init_pos, use_cache=True)
        past_kv = out.past_key_values
        next_tok = int(out.logits[0, -1].argmax().item())

    generated = [next_tok]
    if next_tok == eos_token_id:
        return generated

    # Autoregressive generation step by step
    for step_idx in range(max_tokens - 1):
        inp = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        pos = torch.tensor([[C + 2 + step_idx]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = phi_model(input_ids=inp, position_ids=pos, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = int(out.logits[0, -1].argmax().item())
        generated.append(next_tok)
        if next_tok == eos_token_id:
            break

    return generated


def compute_kl_and_topk(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, float, bool, float, float, float]:
    """Compute KL divergence, Top-1 match, Top-5 overlap, and logit differences."""
    t_f = teacher_logits.float() / temperature
    s_f = student_logits.float() / temperature

    p_t = F.softmax(t_f, dim=-1)
    log_s = F.log_softmax(s_f, dim=-1)

    # Exact sum reduction in nats
    kl_tensor = F.kl_div(log_s, p_t, reduction="sum", log_target=False)
    kl_nats = float(kl_tensor.detach().cpu().item())

    top1_t = int(t_f.argmax().item())
    top1_s = int(s_f.argmax().item())
    top1_match = bool(top1_t == top1_s)

    top5_t = set(torch.topk(t_f, k=5).indices.cpu().tolist())
    top5_s = set(torch.topk(s_f, k=5).indices.cpu().tolist())
    top5_overlap = len(top5_t.intersection(top5_s)) / 5.0

    diff = (t_f - s_f).abs()
    max_logit_diff = float(diff.max().item())
    mean_logit_diff = float(diff.mean().item())

    return kl_tensor, kl_nats, top1_match, top5_overlap, max_logit_diff, mean_logit_diff


def get_oracle_initializations(
    embed_tokens: Any,
    token_a: int,
    token_b: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Generate the 4 required independent initializations for H."""
    with torch.no_grad():
        e_a = embed_tokens(torch.tensor([token_a], device=device))[0].float()
        e_b = embed_tokens(torch.tensor([token_b], device=device))[0].float()

        # Init A: midpoint
        init_a = 0.5 * (e_a + e_b)

        # Init B: token B embedding
        init_b = e_b.clone()

        # Init C: midpoint + small Gaussian perturbation
        init_c = init_a + torch.randn_like(init_a) * 0.02

        # Init D: random vector matching token embedding empirical distribution
        # Sample using empirical mean and std of full embedding matrix
        all_weights = embed_tokens.weight.float()
        mean_emb = all_weights.mean(dim=0)
        std_emb = all_weights.std(dim=0)
        init_d = mean_emb + torch.randn_like(mean_emb) * std_emb

    return {
        "midpoint": init_a.detach().cpu(),
        "token_b": init_b.detach().cpu(),
        "perturbed_midpoint": init_c.detach().cpu(),
        "empirical_random": init_d.detach().cpu(),
    }


@dataclass
class SingleInitOptimizationResult:
    init_name: str
    stage_a_initial_kl: float
    stage_a_best_kl: float
    stage_a_best_top1: bool
    stage_a_best_top5_overlap: float
    stage_a_max_logit_diff: float
    stage_a_steps_run: int
    stage_b_initial_loss: float
    stage_b_best_loss: float
    stage_b_best_offset0_kl: float
    stage_b_best_multi_offset_kl: float
    stage_b_best_top1_rates: Dict[int, bool]
    stage_b_per_offset_kl: Dict[int, float]
    stage_b_steps_run: int
    best_H_stage_a: torch.Tensor
    best_H_stage_b: torch.Tensor
    h_norm_a: float
    h_norm_b: float
    cos_sim_a_with_tokens: Tuple[float, float]
    cos_sim_b_with_tokens: Tuple[float, float]
    rollout_agreement_stage_a: Dict[int, float]
    rollout_agreement_stage_b: Dict[int, float]
    vanilla_rollout_32: List[int]
    student_rollout_32_stage_a: List[int]
    student_rollout_32_stage_b: List[int]
    divergence_index_stage_a: int
    divergence_index_stage_b: int


def optimize_oracle_h_for_example(
    phi_model: Any,
    teacher_ref: TeacherReferenceData,
    device: torch.device,
    *,
    stage_a_max_steps: int = 300,
    stage_b_max_steps: int = 300,
    stage_a_lr: float = 1e-2,
    stage_b_lr: float = 5e-3,
    state_loss_weight: float = 0.05,
    patience: int = 40,
) -> Dict[str, Any]:
    """Execute complete 4-initialization, two-stage optimization on one DEV example."""
    embed_tokens = phi_model.model.embed_tokens
    inits = get_oracle_initializations(embed_tokens, teacher_ref.token_a, teacher_ref.token_b, device)
    
    e_a_float = embed_tokens(torch.tensor([teacher_ref.token_a], device=device))[0].float()
    e_b_float = embed_tokens(torch.tensor([teacher_ref.token_b], device=device))[0].float()

    init_results: Dict[str, SingleInitOptimizationResult] = {}

    for init_name, init_tensor in inits.items():
        # =====================================================================
        # STAGE A: IMMEDIATE FIT (Offset 0 only)
        # =====================================================================
        H_a = nn.Parameter(init_tensor.clone().to(device=device, dtype=torch.float32))
        opt_a = torch.optim.Adam([H_a], lr=stage_a_lr)
        sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=stage_a_max_steps, eta_min=1e-4)

        t_logits_0 = teacher_ref.logits_at_offsets[0].to(device)
        t_hidden_0 = teacher_ref.hidden_at_offsets[0].to(device)

        best_kl_a = float("inf")
        best_H_a_state = H_a.detach().clone().cpu()
        best_metrics_a = {}
        no_improve_a = 0
        initial_kl_a = float("inf")

        for step in range(1, stage_a_max_steps + 1):
            opt_a.zero_grad()
            s_logits, s_hidden = student_forward_with_oracle_h(
                phi_model, teacher_ref.context_ids, H_a, device, offsets=(0,)
            )

            kl_loss, kl_nats, top1_m, top5_o, max_l, mean_l = compute_kl_and_topk(
                t_logits_0, s_logits[0]
            )

            if step == 1:
                initial_kl_a = kl_nats

            # Weak state similarity auxiliary term
            cos_sim = F.cosine_similarity(t_hidden_0.float(), s_hidden[0].float(), dim=-1)
            state_loss = (1.0 - cos_sim)
            total_loss_a = kl_loss + state_loss_weight * state_loss

            total_loss_a.backward()
            torch.nn.utils.clip_grad_norm_([H_a], max_norm=1.0)
            opt_a.step()
            sched_a.step()

            if kl_nats < best_kl_a - 1e-4:
                best_kl_a = kl_nats
                best_H_a_state = H_a.detach().clone().cpu()
                best_metrics_a = {
                    "top1_match": top1_m,
                    "top5_overlap": top5_o,
                    "max_logit_diff": max_l,
                    "mean_logit_diff": mean_l,
                }
                no_improve_a = 0
            else:
                no_improve_a += 1
                if no_improve_a >= patience and step >= 60:
                    break

        steps_run_a = step

        # =====================================================================
        # STAGE B: CONTINUATION FIT (Offsets 0, 1, 2, 4, 8, 16)
        # =====================================================================
        # Initialize from best Stage A result
        H_b = nn.Parameter(best_H_a_state.clone().to(device=device, dtype=torch.float32))
        opt_b = torch.optim.Adam([H_b], lr=stage_b_lr)
        sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=stage_b_max_steps, eta_min=1e-5)

        t_logits_all = {k: v.to(device) for k, v in teacher_ref.logits_at_offsets.items()}
        t_hidden_all = {k: v.to(device) for k, v in teacher_ref.hidden_at_offsets.items()}

        best_loss_b = float("inf")
        best_H_b_state = H_b.detach().clone().cpu()
        best_b_offset0_kl = float("inf")
        best_b_multi_kl = float("inf")
        best_b_top1s = {}
        best_b_per_kl = {}
        no_improve_b = 0
        initial_loss_b = float("inf")

        for step in range(1, stage_b_max_steps + 1):
            opt_b.zero_grad()
            s_logits, s_hidden = student_forward_with_oracle_h(
                phi_model,
                teacher_ref.context_ids,
                H_b,
                device,
                future_token_ids=teacher_ref.future_ids,
                offsets=EVAL_OFFSETS,
            )

            loss_terms = []
            weight_terms = []
            cur_per_kl = {}
            cur_top1s = {}

            for off in EVAL_OFFSETS:
                w = ORACLE_STAGE_B_WEIGHTS.get(off, 1.0)
                kl_t, kl_v, top1_m, _, _, _ = compute_kl_and_topk(
                    t_logits_all[off], s_logits[off]
                )
                loss_terms.append(w * kl_t)
                weight_terms.append(w)
                cur_per_kl[off] = kl_v
                cur_top1s[off] = top1_m

            total_kl_b = sum(loss_terms) / sum(weight_terms)

            # Cosine state loss
            state_losses = [
                (1.0 - F.cosine_similarity(t_hidden_all[off].float(), s_hidden[off].float(), dim=-1))
                for off in EVAL_OFFSETS
            ]
            avg_state_loss = sum(state_losses) / len(state_losses)

            total_loss_b = total_kl_b + state_loss_weight * avg_state_loss

            if step == 1:
                initial_loss_b = float(total_loss_b.item())

            total_loss_b.backward()
            torch.nn.utils.clip_grad_norm_([H_b], max_norm=1.0)
            opt_b.step()
            sched_b.step()

            loss_b_val = float(total_loss_b.item())
            if loss_b_val < best_loss_b - 1e-4:
                best_loss_b = loss_b_val
                best_H_b_state = H_b.detach().clone().cpu()
                best_b_offset0_kl = cur_per_kl[0]
                best_b_multi_kl = float(np.mean(list(cur_per_kl.values())))
                best_b_top1s = cur_top1s
                best_b_per_kl = cur_per_kl
                no_improve_b = 0
            else:
                no_improve_b += 1
                if no_improve_b >= patience and step >= 60:
                    break

        steps_run_b = step

        # =====================================================================
        # FREE-ROLLOUT EVALUATION (Stage A best vs Stage B best)
        # =====================================================================
        rollout_a = generate_student_rollout_from_h(
            phi_model, teacher_ref.context_ids, best_H_a_state, device, max_tokens=32
        )
        rollout_b = generate_student_rollout_from_h(
            phi_model, teacher_ref.context_ids, best_H_b_state, device, max_tokens=32
        )

        vanilla_32 = teacher_ref.vanilla_greedy_32

        # Compute token agreement rates at 8, 16, 32
        def calc_rollout_metrics(student_seq: List[int]) -> Tuple[Dict[int, float], int]:
            agreements = {}
            for length in (8, 16, 32):
                v_slice = vanilla_32[:length]
                s_slice = student_seq[:length]
                min_len = min(len(v_slice), len(s_slice))
                if min_len == 0:
                    agreements[length] = 0.0
                else:
                    matches = sum(1 for i in range(min_len) if v_slice[i] == s_slice[i])
                    agreements[length] = matches / float(length)

            # First divergence index
            div_idx = 32
            for idx in range(min(len(vanilla_32), len(student_seq))):
                if vanilla_32[idx] != student_seq[idx]:
                    div_idx = idx
                    break
            return agreements, div_idx

        agree_a, div_a = calc_rollout_metrics(rollout_a)
        agree_b, div_b = calc_rollout_metrics(rollout_b)

        # Norms and cosine similarities with A and B embeddings
        h_a_dev = best_H_a_state.to(device)
        h_b_dev = best_H_b_state.to(device)

        sim_a_with_a = float(F.cosine_similarity(h_a_dev, e_a_float, dim=-1).item())
        sim_a_with_b = float(F.cosine_similarity(h_a_dev, e_b_float, dim=-1).item())
        sim_b_with_a = float(F.cosine_similarity(h_b_dev, e_a_float, dim=-1).item())
        sim_b_with_b = float(F.cosine_similarity(h_b_dev, e_b_float, dim=-1).item())

        init_results[init_name] = SingleInitOptimizationResult(
            init_name=init_name,
            stage_a_initial_kl=initial_kl_a,
            stage_a_best_kl=best_kl_a,
            stage_a_best_top1=best_metrics_a.get("top1_match", False),
            stage_a_best_top5_overlap=best_metrics_a.get("top5_overlap", 0.0),
            stage_a_max_logit_diff=best_metrics_a.get("max_logit_diff", 0.0),
            stage_a_steps_run=steps_run_a,
            stage_b_initial_loss=initial_loss_b,
            stage_b_best_loss=best_loss_b,
            stage_b_best_offset0_kl=best_b_offset0_kl,
            stage_b_best_multi_offset_kl=best_b_multi_kl,
            stage_b_best_top1_rates=best_b_top1s,
            stage_b_per_offset_kl=best_b_per_kl,
            stage_b_steps_run=steps_run_b,
            best_H_stage_a=best_H_a_state,
            best_H_stage_b=best_H_b_state,
            h_norm_a=float(best_H_a_state.norm().item()),
            h_norm_b=float(best_H_b_state.norm().item()),
            cos_sim_a_with_tokens=(sim_a_with_a, sim_a_with_b),
            cos_sim_b_with_tokens=(sim_b_with_a, sim_b_with_b),
            rollout_agreement_stage_a=agree_a,
            rollout_agreement_stage_b=agree_b,
            vanilla_rollout_32=vanilla_32,
            student_rollout_32_stage_a=rollout_a,
            student_rollout_32_stage_b=rollout_b,
            divergence_index_stage_a=div_a,
            divergence_index_stage_b=div_b,
        )

    # Determine best overall initialization based on lowest Stage-B loss
    best_init_name = min(init_results.keys(), key=lambda k: init_results[k].stage_b_best_loss)
    best_res = init_results[best_init_name]

    return {
        "best_init_name": best_init_name,
        "best_result": best_res,
        "all_init_results": init_results,
    }
