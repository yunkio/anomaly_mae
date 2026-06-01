"""
CATCH — Channel-Aware feature regularizaTion with frequency-Compositional Head

Upstream source (vendored verbatim, structurally unchanged):
    https://github.com/decisionintelligence/CATCH
    (TAB framework: ts_benchmark/baselines/catch/)

Vendored from upstream files (commit-level fidelity):
    - ts_benchmark/baselines/catch/CATCH.py                 (DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS)
    - ts_benchmark/baselines/catch/models/CATCH_model.py    (CATCHModel, Flatten_Head)
    - ts_benchmark/baselines/catch/layers/RevIN.py          (RevIN)
    - ts_benchmark/baselines/catch/layers/channel_mask.py   (channel_mask_generator)
    - ts_benchmark/baselines/catch/layers/cross_channel_Transformer.py
                                                            (PreNorm, FeedForward, c_Attention,
                                                             c_Transformer, Trans_C)
    - ts_benchmark/baselines/catch/utils/ch_discover_loss.py
                                                            (DynamicalContrastiveLoss)
    - ts_benchmark/baselines/catch/utils/fre_rec_loss.py    (frequency_loss, frequency_criterion)

Modifications (minimum necessary for standalone use):
    1. TAB-framework `from ts_benchmark...` imports removed; all dependencies in-lined.
    2. Device-agnostic — no `.cuda()` hardcoding. Tensors follow input device.
    3. `CATCH` exported as a thin alias for `CATCHModel` (same forward signature).
    4. `TransformerConfig` retained from upstream `CATCH.py` so wrapper can pass kwargs.

Non-standard dependencies: `einops` only (already in dc_vis env).

Architecture is preserved verbatim — no reshuffling of layers, ops, or parameter
init. The forward signature returns `(recon, complex_z, dcloss)` exactly as in
upstream `CATCHModel.forward`.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from torch.nn.functional import gumbel_softmax


# =============================================================================
# Default hyperparameters (verbatim from upstream CATCH.py)
# =============================================================================

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "lr": 0.0001,
    "Mlr": 0.00001,
    "e_layers": 3,
    "n_heads": 2,
    "cf_dim": 64,
    "d_ff": 256,
    "d_model": 128,
    "head_dim": 64,
    "individual": 0,
    "dropout": 0.2,
    "head_dropout": 0.1,
    "auxi_loss": "MAE",
    "auxi_type": "complex",
    "auxi_mode": "fft",
    "auxi_lambda": 0.005,
    "score_lambda": 0.05,
    "regular_lambda": 0.5,
    "temperature": 0.07,
    "patch_stride": 8,
    "patch_size": 16,
    "inference_patch_stride": 1,
    "inference_patch_size": 32,
    "dc_lambda": 0.005,
    "module_first": True,
    "mask": False,
    "pretrained_model": None,
    "num_epochs": 3,
    "batch_size": 128,
    "patience": 3,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "seq_len": 192,
    "pct_start": 0.3,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "lradj": "type1",
}


class TransformerConfig:
    """Vendored from upstream CATCH.py — config object passed to CATCHModel.

    Stores defaults + kwargs as plain attributes; supports `.pred_len` /
    `.learning_rate` properties used elsewhere in upstream training loop.
    """

    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr


# =============================================================================
# RevIN (verbatim from upstream layers/RevIN.py)
# =============================================================================


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
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
        elif mode == 'transform':
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

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


# =============================================================================
# channel_mask_generator (verbatim from upstream layers/channel_mask.py)
# =============================================================================


class channel_mask_generator(torch.nn.Module):
    def __init__(self, input_size, n_vars):
        super(channel_mask_generator, self).__init__()
        self.generator = nn.Sequential(
            torch.nn.Linear(input_size * 2, n_vars, bias=False),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            self.generator[0].weight.zero_()
        self.n_vars = n_vars

    def forward(self, x):  # x: [(bs x patch_num) x n_vars x patch_size]
        distribution_matrix = self.generator(x)
        resample_matrix = self._bernoulli_gumbel_rsample(distribution_matrix)

        inverse_eye = 1 - torch.eye(self.n_vars, device=x.device)
        diag = torch.eye(self.n_vars, device=x.device)

        resample_matrix = torch.einsum("bcd,cd->bcd", resample_matrix, inverse_eye) + diag
        return resample_matrix

    def _bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape
        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)
        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix


# =============================================================================
# DynamicalContrastiveLoss (verbatim from upstream utils/ch_discover_loss.py)
# =============================================================================


class DynamicalContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, k=0.3):
        super(DynamicalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.k = k

    def _stable_scores(self, scores):
        max_scores = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        stable_scores = scores - max_scores
        return stable_scores

    def forward(self, scores, attn_mask, norm_matrix):
        b = scores.shape[0]
        n_vars = scores.shape[-1]

        cosine = (scores / norm_matrix).mean(1)
        pos_scores = torch.exp(cosine / self.temperature) * attn_mask
        all_scores = torch.exp(cosine / self.temperature)

        clustering_loss = -torch.log(pos_scores.sum(dim=-1) / all_scores.sum(dim=-1))

        eye = torch.eye(attn_mask.shape[-1], device=attn_mask.device).unsqueeze(0).repeat(b, 1, 1)
        regular_loss = 1 / (n_vars * (n_vars - 1)) * torch.norm(
            eye.reshape(b, -1) - attn_mask.reshape((b, -1)), p=1, dim=-1
        )
        loss = clustering_loss.mean(1) + self.k * regular_loss
        mean_loss = loss.mean()
        return mean_loss


# =============================================================================
# Cross-Channel Transformer (verbatim from upstream layers/cross_channel_Transformer.py)
# =============================================================================


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class c_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.d_k = math.sqrt(self.dim_head)
        inner_dim = dim_head * heads
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.dynamicalContranstiveLoss = DynamicalContrastiveLoss(
            k=regular_lambda, temperature=temperature
        )

    def forward(self, x, attn_mask=None):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        scale = 1 / self.d_k

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dynamical_contrastive_loss = None

        scores = einsum('b h i d, b h j d -> b h i j', q, k)

        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        norm_matrix = torch.einsum('bhid,bhjd->bhij', q_norm, k_norm)
        if attn_mask is not None:
            def _mask(scores, attn_mask):
                large_negative = -math.log(1e10)
                attention_mask = torch.where(attn_mask == 0, large_negative, 0)
                scores = scores * attn_mask.unsqueeze(1) + attention_mask.unsqueeze(1)
                return scores

            masked_scores = _mask(scores, attn_mask)
            dynamical_contrastive_loss = self.dynamicalContranstiveLoss(scores, attn_mask, norm_matrix)
        else:
            masked_scores = scores

        attn = self.attend(masked_scores * scale)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn, dynamical_contrastive_loss


class c_Transformer(nn.Module):
    """Register the blocks into whole network (verbatim upstream)."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.8, regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,
                        c_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                    regular_lambda=regular_lambda, temperature=temperature)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x, attn_mask=None):
        total_loss = 0
        for attn, ff in self.layers:
            x_n, attn, dcloss = attn(x, attn_mask=attn_mask)
            total_loss += dcloss
            x = x_n + x
            x = ff(x) + x
        dcloss = total_loss / len(self.layers)
        return x, attn, dcloss


class Trans_C(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head, dropout, patch_dim, horizon, d_model,
                 regular_lambda=0.3, temperature=0.1):
        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.transformer = c_Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,
            regular_lambda=regular_lambda, temperature=temperature,
        )
        self.mlp_head = nn.Linear(dim, d_model)

    def forward(self, x, attn_mask=None):
        x = self.to_patch_embedding(x)
        x, attn, dcloss = self.transformer(x, attn_mask)
        x = self.dropout(x)
        x = self.mlp_head(x).squeeze()
        return x, dcloss


# =============================================================================
# Frequency loss / criterion (verbatim from upstream utils/fre_rec_loss.py)
# =============================================================================


class frequency_loss(torch.nn.Module):
    def __init__(self, configs, keep_dim=False, dim=None):
        super(frequency_loss, self).__init__()
        self.keep_dim = keep_dim
        self.dim = dim
        if configs.auxi_mode == "fft":
            self.fft = torch.fft.fft
        elif configs.auxi_mode == "rfft":
            self.fft = torch.fft.rfft
        else:
            raise NotImplementedError
        self.configs = configs
        if configs.mask:
            self._generate_mask()
        else:
            self.mask = None

    def _generate_mask(self):
        if self.configs.add_noise and self.configs.noise_amp > 0:
            seq_len = self.configs.pred_len
            cutoff_freq_percentage = self.configs.noise_freq_percentage
            if self.configs.auxi_mode == "rfft":
                cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            elif self.configs.auxi_mode == "fft":
                cutoff_freq = int((seq_len) * cutoff_freq_percentage)
                low_pass_mask = torch.ones(seq_len)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1)
        else:
            self.mask = None

    def forward(self, outputs, batch_y):
        if outputs.is_complex():
            frequency_outputs = outputs
        else:
            frequency_outputs = self.fft(outputs, dim=1)
        # fft shape: [B, P, D]
        if self.configs.auxi_type == 'complex':
            loss_auxi = frequency_outputs - self.fft(batch_y, dim=1)
        elif self.configs.auxi_type == 'complex-phase':
            loss_auxi = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
        elif self.configs.auxi_type == 'complex-mag-phase':
            loss_auxi_mag = (frequency_outputs - self.fft(batch_y, dim=1)).abs()
            loss_auxi_phase = (frequency_outputs - self.fft(batch_y, dim=1)).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        elif self.configs.auxi_type == 'phase':
            loss_auxi = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
        elif self.configs.auxi_type == 'mag':
            loss_auxi = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
        elif self.configs.auxi_type == 'mag-phase':
            loss_auxi_mag = frequency_outputs.abs() - self.fft(batch_y, dim=1).abs()
            loss_auxi_phase = frequency_outputs.angle() - self.fft(batch_y, dim=1).angle()
            loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
        else:
            raise NotImplementedError

        if self.mask is not None:
            loss_auxi *= self.mask

        if self.configs.auxi_loss == "MAE":
            loss_auxi = loss_auxi.abs().mean(dim=self.dim, keepdim=self.keep_dim) if self.configs.module_first \
                else loss_auxi.mean(dim=self.dim, keepdim=self.keep_dim).abs()
        elif self.configs.auxi_loss == "MSE":
            loss_auxi = (loss_auxi.abs() ** 2).mean(dim=self.dim, keepdim=self.keep_dim) if self.configs.module_first \
                else (loss_auxi ** 2).mean(dim=self.dim, keepdim=self.keep_dim).abs()
        else:
            raise NotImplementedError
        return loss_auxi


class frequency_criterion(torch.nn.Module):
    def __init__(self, configs):
        super(frequency_criterion, self).__init__()
        self.metric = frequency_loss(configs, dim=1, keep_dim=True)
        self.patch_size = configs.inference_patch_size
        self.patch_stride = configs.inference_patch_stride
        self.win_size = configs.seq_len
        self.patch_num = int((self.win_size - self.patch_size) / self.patch_stride + 1)
        self.padding_length = self.win_size - (self.patch_size + (self.patch_num - 1) * self.patch_stride)

    def forward(self, outputs, batch_y):
        output_patch = outputs.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        b, n, c, p = output_patch.shape
        output_patch = rearrange(output_patch, 'b n c p -> (b n) p c')
        y_patch = batch_y.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        y_patch = rearrange(y_patch, 'b n c p -> (b n) p c')

        main_part_loss = self.metric(output_patch, y_patch)
        main_part_loss = main_part_loss.repeat(1, self.patch_size, 1)
        main_part_loss = rearrange(main_part_loss, '(b n) p c -> b n p c', b=b)

        end_point = self.patch_size + (self.patch_num - 1) * self.patch_stride - 1
        start_indices = np.array(range(0, end_point, self.patch_stride))
        end_indices = start_indices + self.patch_size

        indices = torch.tensor(
            [range(start_indices[i], end_indices[i]) for i in range(n)]
        ).unsqueeze(0).unsqueeze(-1)
        indices = indices.repeat(b, 1, 1, c).to(main_part_loss.device)
        main_loss = torch.zeros((b, n, self.win_size - self.padding_length, c), device=main_part_loss.device)
        main_loss.scatter_(dim=2, index=indices, src=main_part_loss)

        non_zero_cnt = torch.count_nonzero(main_loss, dim=1)
        main_loss = main_loss.sum(1) / non_zero_cnt

        if self.padding_length > 0:
            padding_loss = self.metric(outputs[:, -self.padding_length:, :], batch_y[:, -self.padding_length:, :])
            padding_loss = padding_loss.repeat(1, self.padding_length, 1)
            total_loss = torch.concat([main_loss, padding_loss], dim=1)
        else:
            total_loss = main_loss
        return total_loss


# =============================================================================
# CATCHModel + Flatten_Head (verbatim from upstream models/CATCH_model.py)
# =============================================================================


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, seq_len, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, seq_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, seq_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears1[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
        return x


class CATCHModel(nn.Module):
    """CATCH backbone (verbatim from upstream models/CATCH_model.py).

    forward(z): z [bs x seq_len x n_vars] → returns
        (recon [bs x seq_len x n_vars],
         complex_z [bs x n_vars x (seq_len*2)] (complex),
         dcloss (scalar))
    """

    def __init__(self, configs, **kwargs):
        super(CATCHModel, self).__init__()

        self.revin_layer = RevIN(configs.c_in, affine=configs.affine, subtract_last=configs.subtract_last)
        # Patching
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.horizon = self.seq_len
        patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 1)
        self.norm = nn.LayerNorm(self.patch_size)
        # Backbone
        self.re_attn = True
        self.mask_generator = channel_mask_generator(input_size=configs.patch_size, n_vars=configs.c_in)
        self.frequency_transformer = Trans_C(
            dim=configs.cf_dim, depth=configs.e_layers, heads=configs.n_heads,
            mlp_dim=configs.d_ff, dim_head=configs.head_dim, dropout=configs.dropout,
            patch_dim=configs.patch_size * 2, horizon=self.horizon * 2,
            d_model=configs.d_model * 2,
            regular_lambda=configs.regular_lambda, temperature=configs.temperature,
        )

        # Head
        self.head_nf_f = configs.d_model * 2 * patch_num
        self.n_vars = configs.c_in
        self.individual = configs.individual
        self.head_f1 = Flatten_Head(
            self.individual, self.n_vars, self.head_nf_f, configs.seq_len,
            head_dropout=configs.head_dropout,
        )
        self.head_f2 = Flatten_Head(
            self.individual, self.n_vars, self.head_nf_f, configs.seq_len,
            head_dropout=configs.head_dropout,
        )

        self.ircom = nn.Linear(self.seq_len * 2, self.seq_len)
        self.rfftlayer = nn.Linear(self.seq_len * 2 - 2, self.seq_len)
        self.final = nn.Linear(self.seq_len * 2, self.seq_len)

        # break up R&I:
        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)

    def forward(self, z):  # z: [bs x seq_len x n_vars]
        z = self.revin_layer(z, 'norm')

        z = z.permute(0, 2, 1)
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        # do patching
        z1 = z1.unfold(dimension=-1, size=self.patch_size,
                       step=self.patch_stride)  # z1: [bs x nvars x patch_num x patch_size]
        z2 = z2.unfold(dimension=-1, size=self.patch_size,
                       step=self.patch_stride)

        # for channel-wise_1
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_size = z1.shape[3]

        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))
        z_cat = torch.cat((z1, z2), -1)

        channel_mask = self.mask_generator(z_cat)

        z, dcloss = self.frequency_transformer(z_cat, channel_mask)
        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)  # z1: [bs, nvars, patch_num, horizon]
        z2 = z2.permute(0, 2, 1, 3)

        z1 = self.head_f1(z1)  # z: [bs x nvars x seq_len]
        z2 = self.head_f2(z2)

        complex_z = torch.complex(z1, z2)

        z = torch.fft.ifft(complex_z)
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        # denorm
        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')

        return z, complex_z.permute(0, 2, 1), dcloss


# =============================================================================
# Public alias — preserve upstream class name + a shorter export name
# =============================================================================

# `CATCH` is the public symbol the wrapper imports. It is identical to
# upstream `CATCHModel` — no architectural change.
CATCH = CATCHModel


__all__ = [
    "CATCH",
    "CATCHModel",
    "TransformerConfig",
    "DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS",
    "RevIN",
    "channel_mask_generator",
    "DynamicalContrastiveLoss",
    "Trans_C",
    "c_Transformer",
    "c_Attention",
    "PreNorm",
    "FeedForward",
    "Flatten_Head",
    "frequency_loss",
    "frequency_criterion",
]
