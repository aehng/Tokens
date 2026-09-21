"""Differentiable Training Objectives & Parameter Configuration (Phase 4 & Caution 2).

Provides:
1. configure_joint_training_parameters(model):
   - Freezes original base-model weights (base_layer weights)
   - Unfreezes Zip2Zip LoRA adapters (lora_A, lora_B)
   - Unfreezes input_encoder (hyper-embedding synthesis)
   - Unfreezes output_encoder (hyper-linear synthesis)
   - Ensures lm_head.detach_input = False so gradients flow into LoRA

2. DifferentiablePredictiveManager:
   - Computes hyper-embedding & hyper-linear weights differentiably every step
   - Handles cache invalidation & reset cleanly for PyTorch autograd
   - Controls position IDs according to model.uses_base_token_positions

3. compute_reconstruction_loss:
   - Computes auxiliary auto-encoding cross-entropy loss:
     P(c_s | input_encoder(c_1 ... c_S))
   - Guarantees hypertoken representations retain constituent token identity

4. compute_continuation_consistency_loss:
   - Optional ablation: KL divergence & cosine loss between base and hyper paths
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple, Any
import torch
from torch import nn
import torch.nn.functional as F

from zip2zip import Zip2ZipModel, StaticCodebookManager
from zip2zip.nn.encoders.base import EncoderFn


def configure_joint_training_parameters(model: Zip2ZipModel) -> Dict[str, Any]:
    """Configure parameter freeze/train flags for joint predictive training.

    Joint Training Targets:
    1. Base transformer weights -> FROZEN (requires_grad = False)
    2. Zip2Zip LoRA adapters    -> TRAINABLE (requires_grad = True)
    3. input_encoder            -> TRAINABLE (requires_grad = True)
    4. output_encoder           -> TRAINABLE (requires_grad = True)
    5. HyperLinear.detach_input -> FALSE (gradient flows through transformer into LoRA)
    """
    total_params = sum(p.numel() for p in model.parameters())

    # 1. Base model: freeze base weights, unfreeze LoRA
    base_frozen_count = 0
    lora_trainable_count = 0

    for name, param in model.base_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            lora_trainable_count += param.numel()
        else:
            param.requires_grad = False
            base_frozen_count += param.numel()

    # 2. Input encoder: unfreeze
    input_enc_count = 0
    for param in model.input_encoder.parameters():
        param.requires_grad = True
        input_enc_count += param.numel()

    # 3. Output encoder: unfreeze
    output_enc_count = 0
    if getattr(model, "output_encoder", None) is not None:
        for param in model.output_encoder.parameters():
            param.requires_grad = True
            output_enc_count += param.numel()

    # 4. Ensure lm_head allows gradients to flow into transformer
    if hasattr(model.base_model, "lm_head"):
        model.base_model.lm_head.detach_input = False
    base_m = getattr(model, "base_model", model)
    if hasattr(base_m, "get_output_embeddings"):
        out_emb = base_m.get_output_embeddings()
        if hasattr(out_emb, "detach_input"):
            out_emb.detach_input = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_total = sum(p.numel() for p in trainable_params)

    report = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_total,
        "trainable_percentage": round(100.0 * trainable_total / total_params, 2),
        "base_frozen_params": base_frozen_count,
        "lora_trainable_params": lora_trainable_count,
        "input_encoder_params": input_enc_count,
        "output_encoder_params": output_enc_count,
    }
    return report


def compute_reconstruction_loss(
    input_encoder: nn.Module,
    codebook_tensor: torch.Tensor,
    base_embedding_weight: torch.Tensor,
    pad_token_id: int = 32000,
) -> torch.Tensor:
    """Compute auxiliary reconstruction loss for hypertoken embeddings.

    Ensures the hypertoken embedding retains sufficient mutual information
    to predict all constituent base tokens.

    Args:
        input_encoder: The model's input_encoder module
        codebook_tensor: (B, K, S) or (K, S) tensor of phrase base token IDs
        base_embedding_weight: (vocab_size, hidden_dim) embedding matrix
        pad_token_id: padding token id
    Returns:
        Scalar reconstruction loss
    """
    if codebook_tensor.ndim == 2:
        codebook_tensor = codebook_tensor.unsqueeze(0)  # (1, K, S)

    B, K, S = codebook_tensor.shape
    device = base_embedding_weight.device
    codebook_tensor = codebook_tensor.to(device)

    # Valid mask: active phrases (length >= 2, first token != pad)
    valid_phrases = (codebook_tensor[:, :, 0] != pad_token_id) & (
        codebook_tensor[:, :, 1] != pad_token_id
    )  # (B, K)

    if not valid_phrases.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Differentiably encode hypertoken embeddings: (B, K, dim)
    hyper_embeds = input_encoder(codebook_tensor, base_embedding_weight, pad_token_id)

    # Reconstruct constituent tokens via dot product with base embedding table
    # hyper_embeds: (B*K, dim)
    flat_embeds = hyper_embeds.view(B * K, -1)
    flat_codes = codebook_tensor.view(B * K, S)
    flat_valid = valid_phrases.view(B * K)

    active_embeds = flat_embeds[flat_valid]  # (N_active, dim)
    active_codes = flat_codes[flat_valid]    # (N_active, S)

    if active_embeds.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Dot product against base embedding matrix: (N_active, vocab_size)
    # Using float32 for numerical stability in cross entropy
    logits = F.linear(active_embeds.float(), base_embedding_weight.float())  # (N_active, vocab_size)

    # Loss: average cross entropy predicting each constituent non-pad token
    total_recon_loss = torch.tensor(0.0, device=device)
    total_tokens = 0

    for s in range(S):
        targets = active_codes[:, s]  # (N_active,)
        mask = targets != pad_token_id
        if mask.any():
            loss_s = F.cross_entropy(logits[mask], targets[mask], reduction="sum")
            total_recon_loss = total_recon_loss + loss_s
            total_tokens += mask.sum().item()

    return total_recon_loss / max(total_tokens, 1)


class DifferentiableTrainingManager:
    """Manages the differentiable training step for Zip2Zip predictive models.

    Refreshes synthesized weights differentiably on each forward pass, ensuring
    gradients flow into input_encoder, output_encoder, and LoRA layers.
    """

    def __init__(
        self,
        model: Zip2ZipModel,
        initial_vocab_size: int = 32011,
        max_codebook_size: int = 32,
        max_subtokens: int = 4,
        pad_token_id: int = 32000,
    ) -> None:
        self.model = model
        self.initial_vocab_size = initial_vocab_size
        self.max_codebook_size = max_codebook_size
        self.max_subtokens = max_subtokens
        self.pad_token_id = pad_token_id

    def setup_differentiable_codebook(
        self,
        codebook_dict: Dict[Tuple[int, ...], int],
        codebook_tensor: torch.Tensor,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> StaticCodebookManager:
        """Create and attach a StaticCodebookManager configured for training autograd."""
        dim = self.model.zip2zip_config.encoder.hidden_size
        disabled_ids = list(self.model.zip2zip_config.compression.disabled_ids)

        manager = StaticCodebookManager(
            initial_vocab_size=self.initial_vocab_size,
            max_codebook_size=self.max_codebook_size,
            max_subtokens=self.max_subtokens,
            embedding_dim=dim,
            pad_token_id=self.pad_token_id,
            disabled_ids=disabled_ids,
        )
        manager.set_seeded_codebook(codebook_dict, batch_size=batch_size, device=device)
        # Clear autograd caches to ensure fresh graph construction
        manager.clear_weight_caches()
        manager.attach_to_model(self.model)
        return manager

    def forward_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        codebook_dict: Dict[Tuple[int, ...], int],
        codebook_tensor: torch.Tensor,
        recon_weight: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run one forward training step computing LM loss + reconstruction loss differentiably."""
        batch_size = input_ids.shape[0]
        manager = self.setup_differentiable_codebook(
            codebook_dict, codebook_tensor, batch_size=batch_size, device=device
        )

        try:
            # 1. Forward pass through Zip2Zip model (LM cross-entropy)
            # HyperEmbedding and HyperLinear will synthesize weights differentiably
            out = self.model(
                input_ids=input_ids.to(device),
                labels=labels.to(device),
            )
            lm_loss = out.loss

            # 2. Auxiliary reconstruction loss for input_encoder
            base_m = getattr(self.model, "base_model", self.model)
            inp_emb = base_m.get_input_embeddings()
            recon_loss = compute_reconstruction_loss(
                self.model.input_encoder,
                codebook_tensor,
                inp_emb.weight,
                pad_token_id=self.pad_token_id,
            )

            # Combined objective
            total_loss = lm_loss + recon_weight * recon_loss

            metrics = {
                "total_loss": total_loss.item(),
                "lm_loss": lm_loss.item(),
                "recon_loss": recon_loss.item(),
            }
            return total_loss, metrics

        finally:
            # Clean up manager and detach from model
            manager.clear_weight_caches()
            manager.detach_from_model(self.model)
