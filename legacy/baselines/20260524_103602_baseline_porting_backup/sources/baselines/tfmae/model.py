"""
TFMAE Model — Temporal-Frequency Masked Autoencoders

Based on: "Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection"
Paper: ICDE 2024, https://ieeexplore.ieee.org/document/10597757
Original code: https://github.com/LMissher/TFMAE (MIT License)

Modified for:
- Device-agnostic operation (removed global `device` variable; uses tensor's device)
- Integration with comparison framework (sliding-window wrapper interface)
- Single-file vendoring (merged model/MTFAE.py + model/attn.py + model/embed.py)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Embedding (vendored from model/embed.py)
# ===========================================================================


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, data=None, idx=None):
        if data is not None:
            p = self.pe[:data].unsqueeze(0)
        else:
            p = self.pe.unsqueeze(0).repeat(idx.shape[0], 1, 1)[
                torch.arange(idx.shape[0])[:, None], idx, :
            ]
        return p


class TokenEmbedding(nn.Module):
    """Conv1d-based token embedding for time series."""

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    """Combined value + positional embedding (matches upstream)."""

    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(data=x.shape[1])
        return self.dropout(x)


# ===========================================================================
# Attention (vendored from model/attn.py)
# ===========================================================================


class AttentionLayer(nn.Module):
    """Single-head full self-attention with residual + LayerNorm."""

    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        queries = self.query_projection(x)
        keys = self.key_projection(x).transpose(1, 2)
        values = self.value_projection(x)

        attn = torch.softmax(torch.matmul(queries, keys) / math.sqrt(D), dim=-1)

        out = torch.matmul(attn, values) + x

        return self.out_projection(self.norm(out)) + out, attn


# ===========================================================================
# MTFA model (vendored from model/MTFAE.py)
# ===========================================================================


class Encoder(nn.Module):
    """Stack of AttentionLayer with optional LayerNorm. Returns (x, list of attn maps)."""

    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attlist = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attlist.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attlist


class FreEnc(nn.Module):
    """Frequency-domain masked autoencoder.

    Amplitude-based masking in FFT domain: lowest `fr` quantile magnitudes
    are replaced with a learnable complex mask token, then iFFT back to time
    domain and encoded.
    """

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, fr):
        super(FreEnc, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
        self.enc = Encoder(
            [AttentionLayer(d_model) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, d_model, 1, dtype=torch.cfloat))
        self.fr = fr

    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x)  # [B, T, D]
        # FFT along time axis: [B, D, T] -> rfft -> [B, D, T/2+1]
        cx = torch.fft.rfft(ex.transpose(1, 2))
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2)  # [B, D, Mag]

        quantile = torch.quantile(mag, self.fr, dim=2, keepdim=True)
        idx = torch.argwhere(mag < quantile)
        cx[mag < quantile] = self.mask_token.repeat(ex.shape[0], 1, mag.shape[-1])[
            idx[:, 0], idx[:, 1], idx[:, 2]
        ]

        ix = torch.fft.irfft(cx).transpose(1, 2)
        dx, att = self.enc(ix)
        rec = self.pro(dx)
        att.append(rec)
        return att  # list of [B, T, T] (L layers) + recon


class TemEnc(nn.Module):
    """Temporal-domain masked autoencoder.

    Window-based variance scoring identifies top-tr fraction of timesteps as
    "masked" (high variance regions). Unmasked tokens are encoded; masked tokens
    are replaced with learnable token + positional embedding; full sequence is
    decoded.
    """

    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, tr):
        super(TemEnc, self).__init__()
        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.enc = Encoder(
            [AttentionLayer(d_model) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.dec = Encoder(
            [AttentionLayer(d_model) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.tr = int(tr * win_size)
        self.seq_size = seq_size

    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x)  # [B, T, D]
        dev = ex.device
        filters = torch.ones(1, 1, self.seq_size, device=dev)
        ex2 = ex ** 2

        ltr = F.conv1d(
            ex.transpose(1, 2).reshape(-1, ex.shape[1]).unsqueeze(1),
            filters,
            padding=self.seq_size - 1,
        )
        ltr[:, :, : self.seq_size - 1] /= torch.arange(1, self.seq_size, device=dev)
        ltr[:, :, self.seq_size - 1 :] /= self.seq_size
        ltr2 = F.conv1d(
            ex2.transpose(1, 2).reshape(-1, ex.shape[1]).unsqueeze(1),
            filters,
            padding=self.seq_size - 1,
        )
        ltr2[:, :, : self.seq_size - 1] /= torch.arange(1, self.seq_size, device=dev)
        ltr2[:, :, self.seq_size - 1 :] /= self.seq_size

        ltrd = (
            (ltr2 - ltr ** 2)[:, :, : ltr.shape[-1] - self.seq_size + 1]
            .squeeze(1)
            .reshape(ex.shape[0], ex.shape[-1], -1)
            .transpose(1, 2)
        )
        ltrm = (
            ltr[:, :, : ltr.shape[-1] - self.seq_size + 1]
            .squeeze(1)
            .reshape(ex.shape[0], ex.shape[-1], -1)
            .transpose(1, 2)
        )
        score = ltrd.sum(-1) / ltrm.sum(-1)

        masked_idx, unmasked_idx = (
            score.topk(self.tr, dim=1, sorted=False)[1],
            (-1 * score).topk(x.shape[1] - self.tr, dim=1, sorted=False)[1],
        )
        unmasked_tokens = ex[torch.arange(ex.shape[0])[:, None], unmasked_idx, :]
        ux, _ = self.enc(unmasked_tokens)
        masked_tokens = self.mask_token.repeat(
            ex.shape[0], masked_idx.shape[1], 1
        ) + self.pos_emb(idx=masked_idx)

        tokens = torch.zeros(ex.shape, device=dev)
        tokens[torch.arange(ex.shape[0])[:, None], unmasked_idx, :] = ux
        tokens[torch.arange(ex.shape[0])[:, None], masked_idx, :] = masked_tokens

        dx, att = self.dec(tokens)
        rec = self.pro(dx)
        att.append(rec)
        return att  # list of [B, T, T] (L layers) + recon


class MTFA(nn.Module):
    """Temporal-Frequency Masked Autoencoder dual model.

    forward(x) returns (tematt, freatt): two lists of length L+1 (encoder layer
    attention maps + final reconstruction).
    """

    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5):
        super(MTFA, self).__init__()
        self.tem = TemEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, tr)
        self.fre = FreEnc(c_in, c_out, d_model, e_layers, win_size, fr)

    def forward(self, x):
        tematt = self.tem(x)
        freatt = self.fre(x)
        return tematt, freatt


# ===========================================================================
# KL divergence helper (vendored from solver.py)
# ===========================================================================


def my_kl_loss(p, q):
    """Per-row KL divergence: returns shape [..., T] (last dim summed)."""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)
