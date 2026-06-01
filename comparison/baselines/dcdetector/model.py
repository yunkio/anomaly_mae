"""
DCdetector Model — Dual Attention Contrastive Representation Learning

Based on: "DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection"
Paper: KDD 2023, https://arxiv.org/abs/2306.10347
Original code: https://github.com/DAMO-DI-ML/KDD2023-DCdetector (no LICENSE — attribution only)

Modified for:
- Removed tkinter._flatten dependency (replaced with itertools-based flatten)
- Removed hardcoded `cuda:0` in RevIN (uses input tensor's device)
- Device-agnostic operation
- Single-file vendoring (merged model/{DCdetector,attn,embed,RevIN}.py)
"""

import math
from math import sqrt
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


# ===========================================================================
# Helpers
# ===========================================================================


def _flatten(nested):
    """Flatten nested list of arbitrary depth (replaces tkinter._flatten)."""
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


# ===========================================================================
# RevIN (vendored from model/RevIN.py, device-agnostic)
# ===========================================================================


class RevIN(nn.Module):
    """Reversible Instance Normalization (instance-level normalize / denormalize)."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            # Use buffers so they move with the module via .to(device)
            self.register_buffer("affine_weight", torch.ones(self.num_features))
            self.register_buffer("affine_bias", torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unknown RevIN mode: {mode}")
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


# ===========================================================================
# Embedding (vendored from model/embed.py)
# ===========================================================================


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
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
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# ===========================================================================
# Dual Attention Structure (vendored from model/attn.py)
# ===========================================================================


class DAC_structure(nn.Module):
    """Dual-Attention Contrastive structure — produces patch-wise + in-patch attention maps."""

    def __init__(
        self,
        win_size,
        patch_size,
        channel,
        mask_flag=True,
        scale=None,
        attention_dropout=0.05,
        output_attention=False,
    ):
        super(DAC_structure, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def forward(
        self,
        queries_patch_size,
        queries_patch_num,
        keys_patch_size,
        keys_patch_num,
        values,
        patch_index,
        attn_mask,
    ):
        # Patch-wise attention
        B, L, H, E = queries_patch_size.shape
        scale_patch_size = self.scale or 1.0 / sqrt(E)
        scores_patch_size = torch.einsum(
            "blhe,bshe->bhls", queries_patch_size, keys_patch_size
        )
        attn_patch_size = scale_patch_size * scores_patch_size
        series_patch_size = self.dropout(torch.softmax(attn_patch_size, dim=-1))

        # In-patch attention
        B, L, H, E = queries_patch_num.shape
        scale_patch_num = self.scale or 1.0 / sqrt(E)
        scores_patch_num = torch.einsum(
            "blhe,bshe->bhls", queries_patch_num, keys_patch_num
        )
        attn_patch_num = scale_patch_num * scores_patch_num
        series_patch_num = self.dropout(torch.softmax(attn_patch_num, dim=-1))

        # Upsampling to win_size × win_size
        series_patch_size = repeat(
            series_patch_size,
            "b l m n -> b l (m repeat_m) (n repeat_n)",
            repeat_m=self.patch_size[patch_index],
            repeat_n=self.patch_size[patch_index],
        )
        series_patch_num = series_patch_num.repeat(
            1,
            1,
            self.window_size // self.patch_size[patch_index],
            self.window_size // self.patch_size[patch_index],
        )
        series_patch_size = reduce(
            series_patch_size, "(b reduce_b) l m n -> b l m n", "mean", reduce_b=self.channel
        )
        series_patch_num = reduce(
            series_patch_num, "(b reduce_b) l m n -> b l m n", "mean", reduce_b=self.channel
        )

        if self.output_attention:
            return series_patch_size, series_patch_num
        return None


class AttentionLayer(nn.Module):
    """Wraps DAC_structure with Q/K projection."""

    def __init__(
        self,
        attention,
        d_model,
        patch_size,
        channel,
        n_heads,
        win_size,
        d_keys=None,
        d_values=None,
    ):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.window_size = win_size
        self.n_heads = n_heads

        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask):
        # patch_size
        B, L, _ = x_patch_size.shape
        H = self.n_heads
        queries_patch_size = self.patch_query_projection(x_patch_size).view(B, L, H, -1)
        keys_patch_size = self.patch_key_projection(x_patch_size).view(B, L, H, -1)

        # patch_num
        B, L, _ = x_patch_num.shape
        queries_patch_num = self.patch_query_projection(x_patch_num).view(B, L, H, -1)
        keys_patch_num = self.patch_key_projection(x_patch_num).view(B, L, H, -1)

        # x_ori → values (unused in DAC_structure but kept for API symmetry)
        B, L, _ = x_ori.shape
        values = self.value_projection(x_ori).view(B, L, H, -1)

        series, prior = self.inner_attention(
            queries_patch_size,
            queries_patch_num,
            keys_patch_size,
            keys_patch_num,
            values,
            patch_index,
            attn_mask,
        )
        return series, prior


# ===========================================================================
# DCdetector main model (vendored from model/DCdetector.py)
# ===========================================================================


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(
                x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask
            )
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list


class DCdetector(nn.Module):
    """Multi-scale dual-attention contrastive representation model.

    Output: (series_patch_mean, prior_patch_mean) — two lists of attention maps
    (over multi-scale patches × encoder layers).
    """

    def __init__(
        self,
        win_size,
        enc_in,
        c_out,
        n_heads=1,
        d_model=256,
        e_layers=3,
        patch_size=(3, 5, 7),
        channel=55,
        d_ff=512,
        dropout=0.0,
        activation="gelu",
        output_attention=True,
    ):
        super(DCdetector, self).__init__()
        self.output_attention = output_attention
        self.patch_size = list(patch_size)
        self.channel = channel
        self.win_size = win_size

        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, d_model, dropout))
            self.embedding_patch_num.append(
                DataEmbedding(self.win_size // patchsize, d_model, dropout)
            )

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(
                        win_size,
                        self.patch_size,
                        channel,
                        False,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    self.patch_size,
                    channel,
                    n_heads,
                    win_size,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        # Store channel for later RevIN re-instantiation in forward (matches upstream pattern)
        self._enc_in = enc_in

    def forward(self, x):
        B, L, M = x.shape  # Batch, win_size, channel

        # Create RevIN on the fly to match the input feature dim and device
        # (upstream does the same; this is to support variable n_features at inference time)
        revin_layer = RevIN(num_features=M).to(x.device)

        series_patch_mean = []
        prior_patch_mean = []

        # Instance Normalization
        x = revin_layer(x, "norm")
        x_ori = self.embedding_window_size(x)

        # Multi-scale patching
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, "b l m -> b m l")
            x_patch_num = rearrange(x_patch_num, "b l m -> b m l")

            x_patch_size = rearrange(x_patch_size, "b m (n p) -> (b m) n p", p=patchsize)
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            x_patch_num = rearrange(x_patch_num, "b m (p n) -> (b m) p n", p=patchsize)
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)

            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.append(series)
            prior_patch_mean.append(prior)

        series_patch_mean = list(_flatten(series_patch_mean))
        prior_patch_mean = list(_flatten(prior_patch_mean))

        if self.output_attention:
            return series_patch_mean, prior_patch_mean
        return None


def my_kl_loss(p, q):
    """Per-row KL divergence used by DCdetector loss."""
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)
