"""
ModernTCN — A Modern Pure Convolution Structure for General Time Series Analysis

Based on: "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis"
Paper: ICLR 2024 Spotlight, https://openreview.net/forum?id=vpJMJerXHU
Original code: https://github.com/luodhhh/ModernTCN (MIT License)

Vendored verbatim from upstream `ModernTCN-detection/`:
    - models/ModernTCN.py        (LayerNorm, ReparamLargeKernelConv, Block, Stage, ModernTCN)
    - models/ModernTCN_Layer.py  (Flatten_Head)
    - layers/RevIN.py            (RevIN — device-agnostic via register_buffer)

Modifications (minimum necessary):
    1. `nn.Layernorm` → `nn.LayerNorm` (upstream typo fix; class is unused in detection path).
    2. RevIN: `register_buffer` for affine params (device-agnostic, matches upstream functionality).
    3. `from layers.RevIN ...` and `from models.ModernTCN_Layer ...` removed; all in-line.
    4. Block.forward: `if N != self.large_kernel_size...` adjusted only when input length needs padding
       (upstream check unchanged here).

Architecture and forward path are preserved verbatim — no reshuffling of layers/ops/init.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# RevIN (vendored from upstream layers/RevIN.py, device-agnostic via buffer)
# ===========================================================================


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # upstream uses plain tensors + manual `.cuda()`; we use Parameters for
        # device-agnostic operation AND optional learning. With affine=True we follow
        # upstream's intent of learnable scale/shift. With affine=False (SWaT default
        # per upstream argparse `--affine 0`) these are not created.
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
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
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ===========================================================================
# Flatten_Head (vendored from upstream ModernTCN_Layer.py)
# ===========================================================================


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


# ===========================================================================
# ModernTCN core (vendored verbatim from upstream models/ModernTCN.py)
# ===========================================================================


class LayerNorm(nn.Module):
    """upstream had `nn.Layernorm` (typo); class is unused in detection forward path."""

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=small_kernel,
                                          stride=stride, padding=small_kernel // 2, groups=groups, dilation=1, bias=False)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):
        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left)
            pad_right = torch.zeros(D_out, D_in, pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left, x], dims=-1)
        x = torch.cat([x, pad_right], dims=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super().__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff // dmodel

    def forward(self, x):
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars,
                 small_kernel_merged=False, drop=0.1):
        super().__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars,
                        small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)
        self.blocks = nn.ModuleList(blks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ModernTCN(nn.Module):
    """ModernTCN core model — verbatim from upstream ModernTCN-detection/models/ModernTCN.py.

    Wrapper-facing constructor signature: supports BOTH:
    (a) upstream-style positional/keyword args (task_name, patch_size, ..., nvars, ...) — used by
        `Model(configs)` wrapper.
    (b) baseline-comparison friendly kwargs (seq_len, enc_in, c_out, patch_size, ..., use_revin,
        affine, subtract_last) — used by `ModernTCNBaseline` wrapper. Internally aliased.
    """

    def __init__(
        self,
        # ---- baseline-wrapper friendly (defaults match upstream SWaT.sh + argparse) ----
        seq_len: int = 100,
        enc_in: int = None,
        c_out: int = None,
        patch_size: int = 8,
        patch_stride: int = 4,
        dims=None,
        num_blocks=None,
        large_size=None,
        small_size=None,
        ffn_ratio: int = 1,
        dropout: float = 0.1,             # backbone_dropout
        head_dropout: float = 0.0,
        use_revin: bool = True,
        affine: bool = False,             # upstream argparse default --affine 0
        subtract_last: bool = False,      # upstream argparse default --subtract_last 0
        # ---- upstream knobs (SWaT defaults / argparse defaults) ----
        task_name: str = 'anomaly_detection',
        stem_ratio: int = 6,              # upstream argparse default
        downsample_ratio: int = 2,        # upstream argparse default
        dw_dims=None,                     # upstream argparse default [256,256,256,256]; here = dims if None
        small_kernel_merged: bool = False,
        use_multi_scale: bool = False,    # SWaT.sh: False
        freq=None,
        individual: bool = False,
        target_window: int = None,        # for anomaly: defaults to seq_len
        # ---- alias for upstream-style positional invocation ----
        nvars: int = None,
        c_in: int = None,
        backbone_dropout: float = None,
        # ---- forecasting-only (no-op in anomaly task) ----
        class_drop: float = 0.,
        class_num: int = 10,
    ):
        super().__init__()
        # Aliases: upstream uses (nvars, c_in); wrapper uses (enc_in, c_out).
        if nvars is None:
            nvars = enc_in
        if c_in is None:
            c_in = enc_in
        if backbone_dropout is not None:
            dropout = backbone_dropout
        if target_window is None:
            target_window = seq_len
        if dims is None:
            dims = [128]
        if num_blocks is None:
            num_blocks = [3]
        if large_size is None:
            large_size = [51]
        if small_size is None:
            small_size = [5]
        if dw_dims is None:
            dw_dims = list(dims)  # match dims length

        # Sanity: stage lists must share length
        assert len(dims) == len(num_blocks) == len(large_size) == len(small_size), \
            "dims / num_blocks / large_size / small_size must all share length"
        # dw_dims length = dims length (use first len(dims) elements if upstream-style [256,...] passed)
        if len(dw_dims) < len(dims):
            dw_dims = list(dw_dims) + [dw_dims[-1]] * (len(dims) - len(dw_dims))
        dw_dims = list(dw_dims)[:len(dims)]

        self.task_name = task_name
        self.class_drop = class_drop
        self.class_num = class_num
        self.seq_len = seq_len

        # RevIN
        self.revin = use_revin
        if self.revin:
            self.revin_layer = RevIN(nvars, affine=affine, subtract_last=subtract_last)

        # stem layer & down sampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Linear(patch_size, dims[0])
        self.downsample_layers.append(stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # backbone
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx],
                          dmodel=dims[stage_idx], dw_model=dw_dims[stage_idx], nvars=nvars,
                          small_kernel_merged=small_kernel_merged, drop=dropout)
            self.stages.append(layer)

        # head
        patch_num = seq_len // patch_stride
        self.n_vars = nvars
        self.individual = individual
        d_model = dims[self.num_stage - 1]

        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)
        else:
            if patch_num % pow(downsample_ratio, (self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio, (self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1)) + 1)
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)

        if self.task_name == 'anomaly_detection':
            self.head_dection1 = nn.Linear(d_model, self.patch_size)

    def forward_feature(self, x, te=None):
        B, M, L = x.shape
        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)

            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
                x = x.reshape(B, M, 1, -1).squeeze(-2)
                x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
                x = self.downsample_layers[i](x)
                x = x.permute(0, 1, 3, 2)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
                    x = self.downsample_layers[i](x)
                    _, D_, N_ = x.shape
                    x = x.reshape(B, M, D_, N_)
                else:
                    x = self.downsample_layers[i](x)
                    _, D_, N_ = x.shape
                    x = x.reshape(B, M, D_, N_)

            x = self.stages[i](x)
        return x

    def detection(self, x):
        # x: (B, L, M) on entry from baseline wrapper. Upstream Model.forward does
        # `x = x.permute(0, 2, 1)` before calling ModernTCN, so x inside detection
        # starts as (B, M, L). We replicate that here.
        x = x.permute(0, 2, 1)  # (B, L, M) → (B, M, L)

        if self.revin:
            x = x.permute(0, 2, 1)            # (B, M, L) → (B, L, M)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)            # back to (B, M, L)

        x = self.forward_feature(x, te=None)   # (B, M, D, N)
        x = x.permute(0, 1, 3, 2)              # (B, M, N, D)
        x = self.head_dection1(x)              # (B, M, N, patch_size)
        B, M, _, _ = x.shape
        x = x.reshape(B, M, -1)                # (B, M, N*patch_size)
        x = x[:, :, :self.seq_len]             # (B, M, seq_len)
        x = x.permute(0, 2, 1)                 # (B, seq_len, M)

        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x

    def forward(self, x, te=None):
        """Baseline-wrapper-facing forward: x is (B, L, M); returns (B, L, M).

        For anomaly_detection task, dispatches to detection() (mirrors upstream).
        """
        if self.task_name == 'anomaly_detection':
            x = self.detection(x)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


__all__ = ["ModernTCN", "RevIN", "Flatten_Head"]
