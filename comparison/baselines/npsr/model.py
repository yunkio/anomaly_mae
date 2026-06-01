"""
NPSR — Nominality Score Conditioned Time Series Anomaly Detection
       by Point/Sequential Reconstruction (Lai et al., NeurIPS 2023).

Paper-faithful reimplementation following the upstream reference:
    https://github.com/andrewlai61616/NPSR
(no LICENSE upstream — research use only; this file is a paraphrased
re-implementation, NOT verbatim copy. Identifiers are renamed for clarity.)

Components:
- ``FixedPositionalEmbedding``: sinusoidal positional encoding shared by both
  sub-models (used both for token-level and per-head position encoding feeds).
- ``NPSRPointAE`` (= reference ``PerformerAEPositionalEncoding``): the point
  reconstructor M_pt. Performer encoder → linear bottleneck to ``z_dim`` →
  linear → Performer decoder → ``tanh``. The ``z_dim`` bottleneck is the
  defining feature of M_pt; without it the autoencoder degenerates to identity.
- ``NPSRSeqReducer`` (= reference ``PerfPredSqz``): the sequence reconstructor
  M_seq. Iteratively shrinks the sequence axis from ``win_in`` to ``win_out``
  via a geometric schedule of per-step ``Linear(Ws[i], Ws[i+1])`` between
  single-depth Performer blocks. Output range stabilised via ``tanh``.
- ``NPSR``: thin holder bundling both sub-models for joint instantiation,
  plus helpers to construct M_seq induction batches.

If ``performer-pytorch`` is not importable, we fall back to a
``nn.MultiheadAttention``-based stand-in so the module is constructible at
all times. The fallback documents itself in ``HAS_PERFORMER = False``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


try:  # pragma: no cover - import gating
    from performer_pytorch import Performer as _PerformerImpl

    HAS_PERFORMER = True
except Exception:  # pragma: no cover - keep import-safe
    _PerformerImpl = None
    HAS_PERFORMER = False


# ---------------------------------------------------------------------------
# Fallback Performer — only used when performer-pytorch is unavailable.
# Provides a (depth)-layer self-attention stack with the same call shape:
#     y = block(x, pos_enc=optional)  # `pos_enc` ignored in fallback
# ---------------------------------------------------------------------------


class _MHAFallbackPerformer(nn.Module):
    """O(N^2) MultiheadAttention fallback for environments without performer_pytorch.

    The depth/heads/ff_mult API mirrors `performer_pytorch.Performer` closely
    enough to be a drop-in replacement for the two NPSR sub-models. The
    ``pos_enc`` keyword argument is accepted and ignored.
    """

    def __init__(self, dim: int, depth: int, heads: int, ff_mult: int = 4,
                 dropout: float = 0.0, **_unused) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            ff = nn.Sequential(
                nn.Linear(dim, ff_mult * dim),
                nn.GELU(),
                nn.Linear(ff_mult * dim, dim),
            )
            n1 = nn.LayerNorm(dim)
            n2 = nn.LayerNorm(dim)
            self.layers.append(nn.ModuleList([attn, ff, n1, n2]))

    def forward(self, x: torch.Tensor, pos_enc: Optional[torch.Tensor] = None,
                **_unused) -> torch.Tensor:
        for attn, ff, n1, n2 in self.layers:
            h = n1(x)
            a, _ = attn(h, h, h, need_weights=False)
            x = x + a
            x = x + ff(n2(x))
        return x


def _make_performer(dim: int, depth: int, heads: int, ff_mult: int = 4) -> nn.Module:
    """Construct a Performer or fallback with reference-matching args."""
    if HAS_PERFORMER:
        return _PerformerImpl(
            dim=dim,
            depth=depth,
            heads=heads,
            causal=False,
            feature_redraw_interval=1,
            dim_head=None,
            ff_mult=ff_mult,
        )
    return _MHAFallbackPerformer(dim=dim, depth=depth, heads=heads, ff_mult=ff_mult)


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding (verbatim-equivalent to reference).
# ---------------------------------------------------------------------------


class FixedPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding of shape ``(max_seq_len, dim)``.

    Returns ``emb[None, :L, :].to(x)`` on forward — broadcasts over batch.
    Used by both M_pt (token-level pos_enc) and M_seq (token + per-head).
    """

    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        # ensure even dim for sin/cos pairing — pad up by 1 if odd
        if dim % 2 != 0:
            dim_eff = dim + 1
        else:
            dim_eff = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_eff, 2, dtype=torch.float) / dim_eff))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        # trim back to requested dim (handles odd dims gracefully)
        emb = emb[:, :dim]
        self.register_buffer("emb", emb, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, *) — return (1, L, dim)
        return self.emb[None, : x.shape[1], :].to(x)


# ---------------------------------------------------------------------------
# NPSRPointAE — M_pt (point reconstructor with z_dim latent bottleneck)
# ---------------------------------------------------------------------------


class NPSRPointAE(nn.Module):
    """Point-level reconstruction autoencoder (a.k.a. M_pt).

    Reference: ``PerformerAEPositionalEncoding`` in ``models/NPSR.py``.

    Pipeline:
        x  -> token_emb(x) + pos_enc(x)                 # (B, L, D)
           -> enc_perf(x, pos_enc=layer_pos_enc(x))     # Performer encoder
           -> enc_lin = Linear(D, z_dim) + GELU         # ← latent bottleneck
           -> dec_lin = Linear(z_dim, D)
           -> dec_perf(x, pos_enc=layer_pos_enc(x))     # Performer decoder
           -> tanh                                       # match MinMax(-1,1)

    The ``z_dim`` bottleneck enforces information compression on a per-timestep
    basis: a normal timestep's content lies in a low-dim manifold and is well
    reconstructed; an anomaly cannot be compressed → larger Δ_xp.
    """

    def __init__(
        self,
        win_size: int,
        d_model: int,
        n_heads: int,
        depth: int = 4,
        z_dim: int = 4,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.model_name = "M_pt"
        self.win_size = win_size
        self.d_model = d_model
        self.z_dim = z_dim

        self.token_emb = nn.Linear(d_model, d_model)
        self.pos_enc = FixedPositionalEmbedding(dim=d_model, max_seq_len=win_size)
        self.layer_pos_enc = FixedPositionalEmbedding(dim=max(d_model // n_heads, 1),
                                                     max_seq_len=win_size)

        self.enc_perf = _make_performer(dim=d_model, depth=depth, heads=n_heads, ff_mult=ff_mult)
        self.enc_lin = nn.Sequential(nn.Linear(d_model, z_dim), nn.GELU())
        self.dec_lin = nn.Linear(z_dim, d_model)
        self.dec_perf = _make_performer(dim=d_model, depth=depth, heads=n_heads, ff_mult=ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        h = self.token_emb(x) + self.pos_enc(x)[:, :, : x.shape[-1]]
        h = self.enc_perf(h, pos_enc=self.layer_pos_enc(h))
        z = self.enc_lin(h)
        h = self.dec_lin(z)
        h = self.dec_perf(h, pos_enc=self.layer_pos_enc(h))
        return torch.tanh(h)


# ---------------------------------------------------------------------------
# NPSRSeqReducer — M_seq (sequence reconstructor with length squeezer)
# ---------------------------------------------------------------------------


class NPSRSeqReducer(nn.Module):
    """Sequence-axis reducer for M_seq (a.k.a. PerfPredSqz).

    Reference: ``PerfPredSqz`` in ``models/NPSR.py``.

    Pipeline (per layer i in [0..depth-1]):
        if i == 0:
            x  = token_emb(x) + pos_enc(x)
            x  = enc_perf[0](x, pos_enc=layer_pos_enc(x))
        else:
            x  = enc_perf[i](x)
        x = transpose(-1,-2)               # (B, D, L_i)
        x = enc_lin[i](x)                  # (B, D, L_{i+1}) — sequence shrink
        x = transpose(-1,-2)               # (B, L_{i+1}, D)
        if i+1 < depth: x = gelu(x)
    Final activation: tanh.

    Ws schedule (geometric):
        Ws = round(win_in * (win_out / win_in) ** linspace(0, 1, depth+1))
    """

    def __init__(
        self,
        win_in: int,
        win_out: int,
        d_model: int,
        n_heads: int,
        depth: int = 4,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.model_name = "M_seq"
        self.win_in = win_in
        self.win_out = win_out
        self.d_model = d_model

        # geometric sequence-length schedule (reference utils — np.round)
        Ws = np.round(
            win_in * (win_out / win_in) ** np.linspace(0.0, 1.0, depth + 1)
        ).astype(int)
        # guard against degenerate zero/negative lengths
        Ws = np.maximum(Ws, 1)
        self.Ws = Ws.tolist()

        self.token_emb = nn.Linear(d_model, d_model)
        self.pos_enc = FixedPositionalEmbedding(dim=d_model, max_seq_len=win_in)
        self.layer_pos_enc = FixedPositionalEmbedding(dim=max(d_model // n_heads, 1),
                                                     max_seq_len=win_in)

        enc_perf = []
        enc_lin = []
        for i in range(depth):
            enc_perf.append(_make_performer(dim=d_model, depth=1, heads=n_heads, ff_mult=ff_mult))
            enc_lin.append(nn.Linear(self.Ws[i], self.Ws[i + 1]))
        self.enc_perf = nn.ModuleList(enc_perf)
        self.enc_lin = nn.ModuleList(enc_lin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, win_in, D)
        for i in range(len(self.enc_perf)):
            if i == 0:
                x = self.token_emb(x) + self.pos_enc(x)[:, :, : x.shape[-1]]
                x = self.enc_perf[0](x, pos_enc=self.layer_pos_enc(x))
            else:
                x = self.enc_perf[i](x)
            x = x.transpose(-1, -2)        # (B, D, L_i)
            x = self.enc_lin[i](x)         # (B, D, L_{i+1})
            x = x.transpose(-1, -2)        # (B, L_{i+1}, D)
            if i + 1 < len(self.enc_perf):
                x = F.gelu(x)
        return torch.tanh(x)


# ---------------------------------------------------------------------------
# NPSR composition module — holds both sub-models for joint serialization.
# ---------------------------------------------------------------------------


def _pick_heads(d_model: int, max_heads: int) -> int:
    """Return the largest h ≤ max_heads such that ``d_model % h == 0`` (≥ 1)."""
    for h in range(min(d_model, max_heads), 0, -1):
        if d_model % h == 0:
            return h
    return 1


class NPSR(nn.Module):
    """Container module: M_pt (NPSRPointAE) + M_seq (NPSRSeqReducer).

    ``forward`` returns a dict with keys ``pt_recon`` (M_pt output, same shape
    as M_pt input) and ``seq_recon`` (M_seq output, shape ``(B, delta, D)``).

    This is a holder — the wrapper drives training (two-optimizer alternating)
    and the inference-time nominality + induced-score computation lives in
    ``wrapper.py`` (it operates over the full train/test reconstruction errors).
    """

    def __init__(
        self,
        win_size: int,
        enc_in: int,
        pred_dl: int = 100,
        delta: int = 20,
        n_heads: int = 4,
        enc_depth: int = 4,
        pred_depth: int = 4,
        z_dim: int = 4,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.enc_in = enc_in
        self.pred_dl = pred_dl
        self.delta = delta
        # Reference uses D = channel; mirror that here. Pick heads that divides D.
        d_model = enc_in
        heads = _pick_heads(d_model, n_heads)
        self.n_heads_effective = heads

        self.M_pt = NPSRPointAE(
            win_size=win_size,
            d_model=d_model,
            n_heads=heads,
            depth=enc_depth,
            z_dim=z_dim,
            ff_mult=ff_mult,
        )
        # M_seq input length = pred_dl (left+right halves concatenated)
        # M_seq output length = delta (middle target)
        self.M_seq = NPSRSeqReducer(
            win_in=pred_dl,
            win_out=delta,
            d_model=d_model,
            n_heads=heads,
            depth=pred_depth,
            ff_mult=ff_mult,
        )

    def forward(self, x_pt: torch.Tensor, x_seq: Optional[torch.Tensor] = None) -> dict:
        """Forward both sub-models.

        Parameters
        ----------
        x_pt : (B, win_size, D)
            Input batch for the point autoencoder.
        x_seq : Optional[(B, pred_dl, D)]
            Concatenated left+right halves for the sequence reducer. If None,
            only ``pt_recon`` is returned (used by tests that exercise just M_pt).
        """
        pt_recon = self.M_pt(x_pt)
        out = {"pt_recon": pt_recon}
        if x_seq is not None:
            out["seq_recon"] = self.M_seq(x_seq)
        return out
