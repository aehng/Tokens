"""Architecture C: CNN + Suffix Model Ranker for Predictor V2.

Captures cheap local-order sequence information:
- 1D Convolutions with kernels 2, 3, 5 over prompt token embeddings.
- Suffix pool over the last 32 prompt tokens.
- Candidate 1D Conv / pooling representation.
- Multi-task MLP heads.
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
EMBED_DIM = 96


class CNNRankerModule(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = EMBED_DIM, feat_dim: int = 21):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=32000)

        # 1D Convolutions over prompt
        self.conv2 = nn.Conv1d(embed_dim, 32, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, 32, kernel_size=5, padding=2)

        # Candidate Conv
        self.cand_conv = nn.Conv1d(embed_dim, 48, kernel_size=2, padding=1)

        # Suffix projection (last 32 tokens)
        self.suffix_dim = embed_dim

        # In dim:
        # prompt convs max-pool: 32 + 32 + 32 = 96
        # prompt suffix mean/max: 96 + 96 = 192
        # cand conv max-pool: 48
        # cand embed mean: 96
        # interaction dot: 1
        # handcrafted: 21
        # Total = 96 + 192 + 48 + 96 + 1 + 21 = 454
        in_dim = 96 + 192 + 48 + 96 + 1 + feat_dim
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

    def encode_prompt(self, prompt_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embed(prompt_tokens)  # (1, L, D)
        L = emb.size(1)

        # Conv features: (1, D, L)
        emb_t = emb.transpose(1, 2)
        c2 = F.relu(self.conv2(emb_t)).max(dim=2).values  # (1, 32)
        c3 = F.relu(self.conv3(emb_t)).max(dim=2).values  # (1, 32)
        c5 = F.relu(self.conv5(emb_t)).max(dim=2).values  # (1, 32)
        conv_feats = torch.cat([c2, c3, c5], dim=1)  # (1, 96)

        # Suffix features (last 32 tokens)
        suffix_len = min(32, L)
        suffix = emb[:, -suffix_len:, :]
        suffix_mean = suffix.mean(dim=1)  # (1, D)
        suffix_max = suffix.max(dim=1).values  # (1, D)
        suffix_feats = torch.cat([suffix_mean, suffix_max], dim=1)  # (1, 192)

        return conv_feats, suffix_feats, suffix_mean

    def forward(
        self,
        conv_p: torch.Tensor,
        suffix_p: torch.Tensor,
        suffix_mean_p: torch.Tensor,
        candidate_tokens: torch.Tensor,  # (B, 4)
        handcrafted: torch.Tensor,  # (B, 21)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = candidate_tokens.size(0)
        cand_emb = self.embed(candidate_tokens)  # (B, 4, D)
        cand_mean = cand_emb.mean(dim=1)  # (B, D)

        cand_emb_t = cand_emb.transpose(1, 2)  # (B, D, 4)
        cand_c = F.relu(self.cand_conv(cand_emb_t)).max(dim=2).values  # (B, 48)

        u_conv = conv_p.expand(B, -1)
        u_suffix = suffix_p.expand(B, -1)
        u_suffix_mean = suffix_mean_p.expand(B, -1)

        dot_prod = (u_suffix_mean * cand_mean).sum(dim=1, keepdim=True)

        features = torch.cat([u_conv, u_suffix, cand_c, cand_mean, dot_prod, handcrafted], dim=1)
        hidden = self.mlp(features)

        logits_occur = self.head_occur(hidden).squeeze(-1)
        logits_count = F.softplus(self.head_count(hidden).squeeze(-1))
        logits_horizon = self.head_horizon(hidden)

        return logits_occur, logits_count, logits_horizon


class CNNRanker(PredictorScorer):
    """Architecture C: CNN + Suffix Model Ranker."""

    def __init__(self, embed_dim: int = EMBED_DIM, device: str = "cpu"):
        self.embed_dim = embed_dim
        self.device = torch.device(device)
        self.model = CNNRankerModule(embed_dim=embed_dim).to(self.device)

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
                conv_p, suffix_p, suf_mean = self.model.encode_prompt(p_tensor)
                logits_occ, pred_cnt, logits_hor = self.model(conv_p, suffix_p, suf_mean, cand_tensor, hand_tensor)

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
            "model": "CNNRanker",
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

                conv_p, suffix_p, suf_mean = self.model.encode_prompt(p_tensor)
                logits_occ, pred_cnt, logits_hor = self.model(conv_p, suffix_p, suf_mean, cand_tensor, hand_tensor)

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

            conv_p, suffix_p, suf_mean = self.model.encode_prompt(p_tensor)
            logits_occ, pred_cnt, logits_hor = self.model(conv_p, suffix_p, suf_mean, cand_tensor, hand_tensor)

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
