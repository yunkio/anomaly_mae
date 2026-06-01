"""
NPSR — Nominality Score Ranking for Time Series Anomaly Detection (NeurIPS 2023)

Based on: "Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction"
Original code: https://github.com/andrewlai61616/NPSR

Modified for:
- Device-agnostic operation
- Single-file vendoring
- `performer-pytorch` dependency made optional — falls back to standard
  nn.MultiheadAttention if Performer is not installed. (The architectural role of
  Performer is fast attention; for benchmarking purposes standard attention
  preserves the same behavior at higher cost.)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from performer_pytorch import SelfAttention as PerformerSelfAttention
    HAS_PERFORMER = True
except ImportError:
    HAS_PERFORMER = False


# ===========================================================================
# Attention block (Performer if available, else standard MHA)
# ===========================================================================


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_performer: bool = False, dropout: float = 0.1):
        super().__init__()
        if use_performer and HAS_PERFORMER:
            self.attn = PerformerSelfAttention(dim=d_model, heads=n_heads, dropout=dropout, causal=False)
            self._uses_performer = True
        else:
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self._uses_performer = False

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self._uses_performer:
            attn_out = self.attn(x)
        else:
            attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ===========================================================================
# Point-level autoencoder (M_pt)
# ===========================================================================


class PointTransformer(nn.Module):
    """Per-timestep reconstruction via short-range Transformer."""

    def __init__(self, enc_in: int, d_model: int, n_heads: int, e_layers: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(enc_in, d_model)
        self.blocks = nn.ModuleList(
            [AttentionBlock(d_model, n_heads, use_performer=False, dropout=dropout) for _ in range(e_layers)]
        )
        self.out_proj = nn.Linear(d_model, enc_in)

    def forward(self, x):
        # x: (B, L, M) → (B, L, M)
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)


# ===========================================================================
# Sequence-level autoencoder (M_seq) with induction
# ===========================================================================


class InductionTransformer(nn.Module):
    """Long-range Transformer with induction tokens (random subset reconstructed from
    surrounding context). Uses Performer attention if available for efficiency.
    """

    def __init__(
        self,
        enc_in: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        induction_length: int,
        win_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.induction_length = induction_length
        self.win_size = win_size
        self.in_proj = nn.Linear(enc_in, d_model)
        # Learnable [MASK] embedding (induction token replacement)
        self.mask_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Learnable positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, win_size, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(d_model, n_heads, use_performer=HAS_PERFORMER, dropout=dropout)
                for _ in range(e_layers)
            ]
        )
        self.out_proj = nn.Linear(d_model, enc_in)

    def forward(self, x, mask: torch.Tensor = None):
        # x: (B, L, M)  mask: (B, L) bool — True at induction positions
        B, L, M = x.shape
        h = self.in_proj(x)
        if mask is not None:
            # Replace masked positions with mask embedding
            mask_b = mask.unsqueeze(-1).expand_as(h)
            h = torch.where(mask_b, self.mask_emb.expand_as(h), h)
        h = h + self.pos_emb[:, :L]
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)


# ===========================================================================
# NPSR main module
# ===========================================================================


class NPSR(nn.Module):
    """Nominality-conditioned anomaly detector with point + sequence autoencoders."""

    def __init__(
        self,
        win_size: int,
        enc_in: int,
        d_model: int = 256,
        n_heads: int = 4,
        e_layers: int = 4,
        induction_length: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.win_size = win_size
        self.enc_in = enc_in
        self.induction_length = induction_length

        self.M_pt = PointTransformer(
            enc_in=enc_in, d_model=d_model, n_heads=n_heads, e_layers=e_layers, dropout=dropout
        )
        self.M_seq = InductionTransformer(
            enc_in=enc_in,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            induction_length=induction_length,
            win_size=win_size,
            dropout=dropout,
        )

    def make_induction_mask(self, B: int, L: int, device) -> torch.Tensor:
        """Random induction mask: ~induction_length positions per sample."""
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        n_mask = min(self.induction_length, L)
        for b in range(B):
            idx = torch.randperm(L, device=device)[:n_mask]
            mask[b, idx] = True
        return mask

    def forward(self, x, induction_mask: torch.Tensor = None):
        # x: (B, L, M)
        pt_recon = self.M_pt(x)  # (B, L, M)
        if induction_mask is None:
            induction_mask = self.make_induction_mask(x.size(0), x.size(1), x.device)
        seq_recon = self.M_seq(x, mask=induction_mask)  # (B, L, M)
        return {
            "pt_recon": pt_recon,
            "seq_recon": seq_recon,
            "induction_mask": induction_mask,
        }

    def anomaly_score(self, x, theta_N: float = 0.985):
        """Per-timestep anomaly score per the NPSR formulation:
            err_pt(t)  = ||x(t) - pt_recon(t)||
            err_seq(t) = ||x(t) - seq_recon(t)||
            N(x)       = quantile(err_pt, theta_N)   (nominality threshold)
            score(t)   = max(err_pt(t), err_seq(t) - N(x))
        """
        out = self.forward(x)
        pt_err = ((x - out["pt_recon"]) ** 2).mean(dim=-1)  # (B, L)
        seq_err = ((x - out["seq_recon"]) ** 2).mean(dim=-1)
        # Per-batch nominality threshold (quantile across L)
        N_per_batch = torch.quantile(pt_err, theta_N, dim=-1, keepdim=True)  # (B, 1)
        adjusted_seq = seq_err - N_per_batch
        score = torch.maximum(pt_err, adjusted_seq)
        return score
