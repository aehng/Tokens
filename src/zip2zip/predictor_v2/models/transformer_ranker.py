"""Architecture E: 1-Layer Lightweight Transformer Ranker for Predictor V2.

Gives global multi-head self-attention a fair representative:
- 1-Layer TransformerEncoder (hidden=128, 4 heads, FFN=256).
- Learned positional embeddings.
- Mean & max pooling over attended prompt states.
- Candidate representation + multi-task MLP heads.
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.zip2zip.predictor_v2.candidate_pool import CandidateRecord
from src.zip2zip.predictor_v2.interfaces import MultiTaskPredictions, PredictorScorer
from src.zip2zip.predictor_v2.utility import compute_expected_occurrence_value

VOCAB_SIZE = 32064
D_MODEL = 128
MAX_POS = 1024


class TransformerRankerModule(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        nhead: int = 4,
        dim_ff: int = 256,
        feat_dim: int = 21,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=32000)
        self.pos_embed = nn.Embedding(MAX_POS, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # In dim:
        # prompt mean: d_model (128)
        # prompt max: d_model (128)
        # cand mean: d_model (128)
        # dot prod: 1
        # handcrafted: 21
        # Total = 128 + 128 + 128 + 1 + 21 = 406
        in_dim = 3 * d_model + 1 + feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.ReLU(),
        )
        self.head_occur = nn.Linear(96, 1)
        self.head_count = nn.Linear(96, 1)
        self.head_horizon = nn.Linear(96, 5)

    def encode_prompt(self, prompt_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        L = min(prompt_tokens.size(1), MAX_POS)
        p_toks = prompt_tokens[:, :L]
        positions = torch.arange(L, device=prompt_tokens.device).unsqueeze(0)

        emb = self.embed(p_toks) + self.pos_embed(positions)  # (1, L, D)
        out = self.transformer(emb)  # (1, L, D)

        mean_h = out.mean(dim=1)  # (1, D)
        max_h = out.max(dim=1).values  # (1, D)
        return mean_h, max_h

    def forward(
        self,
        mean_h: torch.Tensor,
        max_h: torch.Tensor,
        candidate_tokens: torch.Tensor,  # (B, 4)
        handcrafted: torch.Tensor,  # (B, 21)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = candidate_tokens.size(0)
        cand_emb = self.embed(candidate_tokens).mean(dim=1)  # (B, D)

        u_mean = mean_h.expand(B, -1)
        u_max = max_h.expand(B, -1)

        dot_prod = (u_mean * cand_emb).sum(dim=1, keepdim=True)

        features = torch.cat([u_mean, u_max, cand_emb, dot_prod, handcrafted], dim=1)
        hidden = self.mlp(features)

        logits_occur = self.head_occur(hidden).squeeze(-1)
        logits_count = F.softplus(self.head_count(hidden).squeeze(-1))
        logits_horizon = self.head_horizon(hidden)

        return logits_occur, logits_count, logits_horizon


class TransformerRanker(PredictorScorer):
    """Architecture E: 1-Layer Lightweight Transformer Ranker."""

    def __init__(self, d_model: int = D_MODEL, device: str = "cpu"):
        self.d_model = d_model
        self.device = torch.device(device)
        self.model = TransformerRankerModule(d_model=d_model).to(self.device)

    def fit(
        self,
        train_records: Sequence[Any],
        train_candidates: Dict[str, List[CandidateRecord]],
        dev_records: Optional[Sequence[Any]] = None,
        dev_candidates: Optional[Dict[str, List[CandidateRecord]]] = None,
        epochs: int = 15,
        lr: float = 1e-3,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        best_dev_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            for r in train_records:
                cands = train_candidates[r.prompt_id]
                if not cands:
                    continue

                p_tensor = torch.tensor([r.prompt_token_ids], dtype=torch.long, device=self.device)
                cand_pad = np.full((len(cands), 4), 32000, dtype=np.int64)
                for idx, c in enumerate(cands):
                    cand_pad[idx, : len(c.tokens)] = c.tokens

                cand_tensor = torch.tensor(cand_pad, dtype=torch.long, device=self.device)
                hand_tensor = torch.tensor(np.array([c.features for c in cands]), dtype=torch.float32, device=self.device)

                y_occur = torch.tensor([1.0 if c.occurs_in_vanilla else 0.0 for c in cands], dtype=torch.float32, device=self.device)
                y_count = torch.tensor([float(c.occurrence_count) for c in cands], dtype=torch.float32, device=self.device)
                y_horizon = torch.tensor([c.first_occurrence_bucket for c in cands], dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                mean_h, max_h = self.model.encode_prompt(p_tensor)
                logits_occ, pred_cnt, logits_hor = self.model(mean_h, max_h, cand_tensor, hand_tensor)

                loss_occ = F.binary_cross_entropy_with_logits(logits_occ, y_occur)
                loss_cnt = F.smooth_l1_loss(torch.log1p(pred_cnt), torch.log1p(y_count))
                loss_hor = F.cross_entropy(logits_hor, y_horizon)

                total_loss = loss_occ + 0.5 * loss_cnt + 0.3 * loss_hor
                total_loss.backward()
                optimizer.step()

            if dev_records and dev_candidates:
                dev_loss = self._eval_dev_loss(dev_records, dev_candidates)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_state = io.BytesIO()
                    torch.save(self.model.state_dict(), best_state)

        if best_state is not None:
            best_state.seek(0)
            self.model.load_state_dict(torch.load(best_state))

        self.model.eval()
        elapsed = time.perf_counter() - t0
        return {
            "model": "TransformerRanker",
            "epochs": epochs,
            "train_time_sec": round(elapsed, 4),
            "best_dev_loss": round(best_dev_loss, 4) if best_dev_loss < float("inf") else None,
        }

    def _eval_dev_loss(self, records: Sequence[Any], candidates: Dict[str, List[CandidateRecord]]) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for r in records:
                cands = candidates[r.prompt_id]
                if not cands:
                    continue
                p_tensor = torch.tensor([r.prompt_token_ids], dtype=torch.long, device=self.device)
                cand_pad = np.full((len(cands), 4), 32000, dtype=np.int64)
                for idx, c in enumerate(cands):
                    cand_pad[idx, : len(c.tokens)] = c.tokens

                cand_tensor = torch.tensor(cand_pad, dtype=torch.long, device=self.device)
                hand_tensor = torch.tensor(np.array([c.features for c in cands]), dtype=torch.float32, device=self.device)
                y_occur = torch.tensor([1.0 if c.occurs_in_vanilla else 0.0 for c in cands], dtype=torch.float32, device=self.device)
                y_count = torch.tensor([float(c.occurrence_count) for c in cands], dtype=torch.float32, device=self.device)
                y_horizon = torch.tensor([c.first_occurrence_bucket for c in cands], dtype=torch.long, device=self.device)

                mean_h, max_h = self.model.encode_prompt(p_tensor)
                logits_occ, pred_cnt, logits_hor = self.model(mean_h, max_h, cand_tensor, hand_tensor)

                loss = (
                    F.binary_cross_entropy_with_logits(logits_occ, y_occur)
                    + 0.5 * F.smooth_l1_loss(torch.log1p(pred_cnt), torch.log1p(y_count))
                    + 0.3 * F.cross_entropy(logits_hor, y_horizon)
                )
                total_loss += loss.item()
                n += 1
        self.model.train()
        return total_loss / max(n, 1)

    def score_candidates(
        self,
        prompt_ids: Sequence[int],
        candidates: Sequence[CandidateRecord],
        domain: str = "general",
    ) -> MultiTaskPredictions:
        if not candidates:
            return MultiTaskPredictions(
                p_occurs=np.array([]),
                expected_count=np.array([]),
                horizon_logits=np.zeros((0, 5)),
                p_safe=np.array([]),
                ranking_scores=np.array([]),
            )

        self.model.eval()
        with torch.no_grad():
            p_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            cand_pad = np.full((len(candidates), 4), 32000, dtype=np.int64)
            for idx, c in enumerate(candidates):
                cand_pad[idx, : len(c.tokens)] = c.tokens

            cand_tensor = torch.tensor(cand_pad, dtype=torch.long, device=self.device)
            hand_tensor = torch.tensor(np.array([c.features for c in candidates]), dtype=torch.float32, device=self.device)

            mean_h, max_h = self.model.encode_prompt(p_tensor)
            logits_occ, pred_cnt, logits_hor = self.model(mean_h, max_h, cand_tensor, hand_tensor)

            p_occurs = torch.sigmoid(logits_occ).cpu().numpy()
            expected_count = pred_cnt.cpu().numpy()
            horizon_logits = logits_hor.cpu().numpy()
            p_safe = np.ones(len(candidates), dtype=np.float32)

            lens = [c.length for c in candidates]
            ranking_scores = compute_expected_occurrence_value(p_occurs, expected_count, lens)

        return MultiTaskPredictions(
            p_occurs=p_occurs,
            expected_count=expected_count,
            horizon_logits=horizon_logits,
            p_safe=p_safe,
            ranking_scores=ranking_scores,
        )

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def get_model_size_bytes(self) -> int:
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return len(buffer.getvalue())
