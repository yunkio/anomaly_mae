"""
Self-Distilled MAE Model for Multivariate Time Series Anomaly Detection

Supports two patchify modes:
- 'patch_cnn': Patchify first, then CNN per patch (no cross-patch leakage)
- 'linear': Patchify then linear embedding (MAE original style, no CNN)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import Config


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., ICLR'22).

    Per-window per-feature mean/std normalization with optional learnable affine.
    Statistics are detached — no gradient through mean/std.

    Upstream pattern (matches ModernTCN/CATCH/TimesNet/PatchTST):
        normalize at model entry (before patch embedding) →
        encoder/decoder operate in normalized space →
        denormalize at decoder output (before loss / scoring).

    Storage: mean/stdev are stored on self per forward call (single-GPU pattern).
    Suitable for serial forward passes; not safe for nn.DataParallel.

    Args:
        num_features: per-feature affine dimension
        eps: variance floor for numerical stability
        affine: enable learnable γ, β (per-feature)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        # Per-forward statistics (overwritten each normalize() call)
        self.mean: Optional[torch.Tensor] = None
        self.stdev: Optional[torch.Tensor] = None

    def normalize(
        self,
        x: torch.Tensor,
        point_labels: Optional[torch.Tensor] = None,
        visible_only: bool = False,
    ) -> torch.Tensor:
        """Compute and apply per-window normalization. Stores stats on self.

        Args:
            x: (B, L, M) input
            point_labels: (B, L) — 1=anomaly, 0=normal. Required if visible_only=True.
            visible_only: if True + point_labels given, exclude anomaly positions from stats
        Returns:
            x_norm: (B, L, M) normalized
        """
        if visible_only and point_labels is not None:
            # Keep non-anomaly positions for stats (point_labels==0 → keep=1)
            keep = (1.0 - point_labels.float()).unsqueeze(-1).to(x.dtype)  # (B, L, 1)
            cnt = keep.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1, 1)
            mean = (x * keep).sum(dim=1, keepdim=True) / cnt
            var = ((x - mean) ** 2 * keep).sum(dim=1, keepdim=True) / cnt
        else:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)

        # Detach: no gradient through statistics
        self.mean = mean.detach()
        self.stdev = torch.sqrt(var + self.eps).detach()

        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of the most recent normalize() call."""
        if self.mean is None or self.stdev is None:
            raise RuntimeError("RevIN.denormalize() called before normalize()")
        if self.affine:
            # eps**2 in denominator matches upstream RevIN repo for numerical guard
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


class PatchDiscriminator(nn.Module):
    """1D CNN Discriminator for patch-level realism assessment.

    Determines whether a patch is real (from original data) or fake (student-generated).
    Uses Spectral Normalization on all learnable layers for Lipschitz constraint.

    Input:  (batch, num_features, patch_size) — features as channels, time as spatial dim
    Output: (batch, 1) — real/fake logit (pre-sigmoid)
    """

    def __init__(self, num_features: int, patch_size: int, channels: tuple = (64, 32)):
        super().__init__()
        c1, c2 = channels
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv1d(num_features, c1, kernel_size=3)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv1d(c1, c2, kernel_size=3)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(c2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features, patch_size)
        Returns:
            logits: (batch, 1) — real/fake logit
        """
        return self.net(x)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer: identity forward, negate gradient backward."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class AnomalyClassifierHead(nn.Module):
    """Patch-level anomaly classifier with GRL for adversarial feature suppression.

    Input:  (seq_len, batch, d_model) — student decoder hidden states
    Output: (seq_len, batch, 1) — anomaly logits per patch
    """

    def __init__(self, d_model: int, hidden_dim: int = 0, arch: str = 'default'):
        super().__init__()
        if arch == 'dann':
            # DANN-style: 3-layer MLP, ReLU, Dropout(0.5), wider (d_model*2)
            # Matches Ganin et al. 2016 discriminator architecture
            h = d_model * 2
            self.classifier = nn.Sequential(
                nn.Linear(d_model, h),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(h, 1),
            )
        elif arch == '2layer':
            # 2-layer MLP: intermediate between default (1-layer) and dann (3-layer)
            hidden_dim = hidden_dim if hidden_dim > 0 else d_model
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            # Default: 1-layer MLP with LayerNorm
            hidden_dim = hidden_dim if hidden_dim > 0 else d_model // 2
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, hidden: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        reversed_hidden = GradientReversalFunction.apply(hidden, lambda_grl)
        return self.classifier(reversed_hidden)


class ScadProjectionHead(nn.Module):
    """SCAD projection head: maps student hidden to L2-normalized embedding.

    Training-only — not used at inference (SimCLR convention).
    Input:  (seq_len, batch, d_model) — student decoder hidden
    Output: (seq_len, batch, d_proj) — L2-normalized projection in contrastive space
    """

    def __init__(self, d_model: int, d_proj: int = 128, arch: str = 'default'):
        super().__init__()
        if arch == 'linear':
            # 1-layer linear (mitigates projection-head absorption)
            self.proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_proj),
            )
        elif arch == 'deep':
            # 3-layer (more expressive, but higher absorption risk)
            self.proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_proj),
            )
        else:  # default: 2-layer MLP (SimCLR convention)
            self.proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_proj),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.proj(h)
        return F.normalize(z, p=2, dim=-1)


class WassersteinCritic(nn.Module):
    """Wasserstein critic for WDGRL-style domain adaptation.

    No GRL needed — minimax is handled by separate optimizer in trainer.
    No sigmoid — output is unbounded scalar (Wasserstein distance estimate).

    Input:  (N, d_model) — flattened patch features
    Output: (N, 1) — critic scores
    """

    def __init__(self, d_model: int, hidden_dim: int = 0):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim > 0 else d_model // 2
        self.critic = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (N, d_model)
        Returns:
            scores: (N, 1)
        """
        return self.critic(features)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return x


class SelfDistilledMAEMultivariate(nn.Module):
    """Self-Distilled MAE for Multivariate Time Series

    Supports two patchify modes:
    - 'patch_cnn': Patchify first, then CNN per patch (no cross-patch leakage)
    - 'linear': Patchify then linear embedding (MAE original style, no CNN)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patchify_mode = config.patchify_mode

        # Patch configuration (always defined for both strategies)
        # Hard assert: seq_length must be divisible by patch_size. This is also
        # checked in make_config() and trainer.py, but kept here as a final
        # safeguard for direct model instantiation paths.
        assert config.seq_length % config.patch_size == 0, (
            f"seq_length ({config.seq_length}) must be divisible by "
            f"patch_size ({config.patch_size})"
        )
        self.num_patches = config.seq_length // config.patch_size
        self.patch_size = config.patch_size

        # Both strategies use patch-based processing
        self.use_patch = True
        self.effective_seq_len = self.num_patches

        # RevIN (per-window instance normalization) — optional
        # Active path: normalize at forward() entry, denormalize at teacher/student output
        self.use_revin = getattr(config, 'use_revin', False)
        self.revin_visible_only = getattr(config, 'revin_visible_only', False)
        if self.use_revin:
            # Enforce upstream pattern (PatchTST / ModernTCN / CATCH / TimesNet):
            # loader MUST be 'zscore' (global StandardScaler fit on train).
            # With minmax (default for 271), the loader+RevIN combination is non-standard
            # and clipping to [0,1] before per-window normalization can cause numerical
            # issues (very small per-window stdev → 1/stdev amplifies noise under AMP).
            norm_mode = getattr(config, 'normalize_mode', 'zscore')
            if norm_mode != 'zscore':
                raise ValueError(
                    f"use_revin=True requires normalize_mode='zscore' (got '{norm_mode}'). "
                    "RevIN is designed for raw or zscore-normalized inputs (PatchTST/ModernTCN/"
                    "CATCH/TimesNet standard). With minmax, the [0,1] clipping followed by "
                    "per-window normalization is non-standard and may cause numerical instability."
                )
            self.revin = RevIN(
                num_features=config.num_features,
                eps=getattr(config, 'revin_eps', 1e-5),
                affine=getattr(config, 'revin_affine', True),
            )
        else:
            self.revin = None

        # Build embedding layers based on patchify_mode
        self._build_embedding_layers(config)

        # Positional encoding (for encoder)
        self.pos_encoder = PositionalEncoding(config.d_model)

        # === [신규 2026-06-15] Asymmetric decoder width (MAE-style) ===
        # decoder_half_dim=True → decoder(teacher/student/shared)·mask token·output proj·
        # GRL/SCAD head 를 d_model//2, dim_feedforward 를 (d_model//2)*4, nhead 재사용.
        # encoder 는 d_model 유지. decoder_embed(Linear d_model→d_model//2)가 latent 를 좁힘.
        # False → Identity + _dec=d_model → byte-identical. dynamic d_model 도 //2 자동 반영.
        self._decoder_half = getattr(config, 'decoder_half_dim', False)
        if self._decoder_half:
            if not config.mask_after_encoder:
                raise ValueError("decoder_half_dim requires mask_after_encoder=True (271 setup).")
            if not config.use_transformer_encoder_decoder:
                raise ValueError("decoder_half_dim requires use_transformer_encoder_decoder=True.")
            if getattr(config, 'use_teacher_output_ema', False):
                raise ValueError("decoder_half_dim is incompatible with use_teacher_output_ema.")
            if config.d_model % 2 != 0:
                raise ValueError(f"decoder_half_dim requires even d_model, got {config.d_model}.")
            _dec = config.d_model // 2
            if _dec % config.nhead != 0:
                raise ValueError(
                    f"decoder_half_dim: d_model//2={_dec} not divisible by nhead={config.nhead}.")
            _dec_ff = _dec * 4
        else:
            _dec = config.d_model
            _dec_ff = config.dim_feedforward
        self._decoder_d_model = _dec
        self.decoder_embed = nn.Linear(config.d_model, _dec) if self._decoder_half else nn.Identity()

        # Decoder positional encoding (only for TransformerEncoder decoder)
        if config.use_transformer_encoder_decoder:
            self.decoder_pos_encoder = PositionalEncoding(_dec)
        else:
            self.decoder_pos_encoder = None

        # Encoder (Pre-Norm + GELU + eps=1e-6, matching original MAE)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,
            norm_first=True,
            activation='gelu',
            layer_norm_eps=1e-6,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.d_model, eps=1e-6),
        )

        # Shared decoder (optional, trained with teacher)
        # Option 2: Use TransformerEncoder (self-attention only, MAE-style)
        self.num_shared_decoder_layers = config.num_shared_decoder_layers
        self.shared_decoder = None
        if self.num_shared_decoder_layers > 0:
            if config.use_transformer_encoder_decoder:
                # TransformerEncoder (self-attention only)
                shared_decoder_layer = nn.TransformerEncoderLayer(
                    d_model=_dec,
                    nhead=config.nhead,
                    dim_feedforward=_dec_ff,
                    dropout=config.dropout,
                    batch_first=False,
                    norm_first=True,
                    activation='gelu',
                    layer_norm_eps=1e-6,
                )
                self.shared_decoder = nn.TransformerEncoder(
                    shared_decoder_layer,
                    num_layers=self.num_shared_decoder_layers,
                    norm=nn.LayerNorm(_dec, eps=1e-6),
                )
            else:
                # TransformerDecoder (cross-attention with encoder memory)
                shared_decoder_layer = nn.TransformerDecoderLayer(
                    d_model=_dec,
                    nhead=config.nhead,
                    dim_feedforward=_dec_ff,
                    dropout=config.dropout,
                    batch_first=False,
                    norm_first=True,
                    activation='gelu',
                    layer_norm_eps=1e-6,
                )
                self.shared_decoder = nn.TransformerDecoder(
                    shared_decoder_layer,
                    num_layers=self.num_shared_decoder_layers,
                    norm=nn.LayerNorm(_dec, eps=1e-6),
                )

        # Teacher decoder
        self.teacher_decoder = None
        if config.use_teacher:
            if config.use_transformer_encoder_decoder:
                # TransformerEncoder (self-attention only, MAE-style)
                teacher_decoder_layer = nn.TransformerEncoderLayer(
                    d_model=_dec,
                    nhead=config.nhead,
                    dim_feedforward=_dec_ff,
                    dropout=config.dropout,
                    batch_first=False,
                    norm_first=True,
                    activation='gelu',
                    layer_norm_eps=1e-6,
                )
                self.teacher_decoder = nn.TransformerEncoder(
                    teacher_decoder_layer,
                    num_layers=config.num_teacher_decoder_layers,
                    norm=nn.LayerNorm(_dec, eps=1e-6),
                )
            else:
                # TransformerDecoder (cross-attention with encoder output)
                teacher_decoder_layer = nn.TransformerDecoderLayer(
                    d_model=_dec,
                    nhead=config.nhead,
                    dim_feedforward=_dec_ff,
                    dropout=config.dropout,
                    batch_first=False,
                    norm_first=True,
                    activation='gelu',
                    layer_norm_eps=1e-6,
                )
                self.teacher_decoder = nn.TransformerDecoder(
                    teacher_decoder_layer,
                    num_layers=config.num_teacher_decoder_layers,
                    norm=nn.LayerNorm(_dec, eps=1e-6),
                )

        # Student decoder
        self.student_decoder = None
        if config.use_student:
            if config.use_transformer_encoder_decoder:
                # TransformerEncoder (self-attention only, MAE-style)
                student_decoder_layer = nn.TransformerEncoderLayer(
                    d_model=_dec,
                    nhead=config.nhead,
                    dim_feedforward=_dec_ff,
                    dropout=config.dropout,
                    batch_first=False,
                    norm_first=True,
                    activation='gelu',
                    layer_norm_eps=1e-6,
                )
                self.student_decoder = nn.TransformerEncoder(
                    student_decoder_layer,
                    num_layers=config.num_student_decoder_layers,
                    norm=nn.LayerNorm(_dec, eps=1e-6),
                )
            else:
                # TransformerDecoder (cross-attention with encoder output)
                student_decoder_layer = nn.TransformerDecoderLayer(
                    d_model=_dec,
                    nhead=config.nhead,
                    dim_feedforward=_dec_ff,
                    dropout=config.dropout,
                    batch_first=False,
                    norm_first=True,
                    activation='gelu',
                    layer_norm_eps=1e-6,
                )
                self.student_decoder = nn.TransformerDecoder(
                    student_decoder_layer,
                    num_layers=config.num_student_decoder_layers,
                    norm=nn.LayerNorm(_dec, eps=1e-6),
                )

        # Output projections
        if self.use_patch:
            output_dim = config.patch_size * config.num_features
        else:
            output_dim = config.num_features

        if config.use_teacher:
            self.teacher_output_projection = nn.Linear(_dec, output_dim)
        if config.use_student:
            self.student_output_projection = nn.Linear(_dec, output_dim)

        # Mask token(s) - shared or separate for teacher/student
        self.shared_mask_token = config.shared_mask_token
        self.mask_after_encoder = config.mask_after_encoder

        if self.shared_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, _dec))
            nn.init.normal_(self.mask_token, std=0.02)
        else:
            # Separate mask tokens for teacher and student
            if config.use_teacher:
                self.teacher_mask_token = nn.Parameter(torch.zeros(1, 1, _dec))
                nn.init.normal_(self.teacher_mask_token, std=0.02)
            if config.use_student:
                self.student_mask_token = nn.Parameter(torch.zeros(1, 1, _dec))
                nn.init.normal_(self.student_mask_token, std=0.02)

        # === [신규 2026-06-01] Teacher output EMA modules ===
        # use_teacher_output_ema=True 일 때만 생성. teacher_decoder + teacher_output_projection
        # (+ shared_decoder, + teacher mask token) 의 weight-EMA 사본 (encoder 제외). backprop 대상 아님.
        # student 의 output discrepancy 표적 산출 전용. _ema_active 는 trainer 가 매 epoch 설정
        # (post-warmup & freeze_teacher_after_warmup=False 일 때만 True → 그때만 ema forward 수행).
        self._ema_active = False
        self._ema_teacher_output = None
        self._has_teacher_output_ema = bool(getattr(config, 'use_teacher_output_ema', False)) and config.use_teacher
        if self._has_teacher_output_ema:
            import copy as _copy
            self.ema_teacher_decoder = _copy.deepcopy(self.teacher_decoder)
            self.ema_teacher_output_projection = _copy.deepcopy(self.teacher_output_projection)
            self.ema_shared_decoder = (_copy.deepcopy(self.shared_decoder)
                                       if getattr(self, 'shared_decoder', None) is not None else None)
            _src_mask = self.mask_token if self.shared_mask_token else self.teacher_mask_token
            self.register_buffer('ema_teacher_mask_token', _src_mask.detach().clone())
            for _m in (self.ema_teacher_decoder, self.ema_teacher_output_projection, self.ema_shared_decoder):
                if _m is not None:
                    _m.eval()
                    for _p in _m.parameters():
                        _p.requires_grad_(False)

        # GRL / WDGRL (use_grl=True일 때만)
        if getattr(config, 'use_grl', False):
            _grl_mode = getattr(config, 'grl_mode', 'classifier')
            if _grl_mode == 'wdgrl':
                self.wasserstein_critic = WassersteinCritic(
                    _dec, getattr(config, 'grl_cls_hidden', 0))
            else:
                self.anomaly_classifier = AnomalyClassifierHead(
                    _dec, getattr(config, 'grl_cls_hidden', 0),
                    arch=getattr(config, 'grl_cls_arch', 'default'))

        # SCAD (use_scad=True일 때만) — replaces GRL classifier.
        # H-SCAD-C (scad_apply_space='hidden_final') applies the repulsion directly on the
        # final student_hidden via a parameter-free LayerNorm+L2norm, so it needs NO
        # projection head. Build scad_head only for the 'projection' path (default), which
        # keeps the projection-SCAD-C model byte-identical.
        if getattr(config, 'use_scad', False):
            self._scad_apply_space = getattr(config, 'scad_apply_space', 'projection')
            if self._scad_apply_space == 'projection':
                self.scad_head = ScadProjectionHead(
                    d_model=_dec,
                    d_proj=getattr(config, 'scad_d_proj', 128),
                    arch=getattr(config, 'scad_proj_head_arch', 'default'),
                )
            if getattr(config, 'scad_use_memory_bank', False):
                _q_size = getattr(config, 'scad_memory_bank_size', 1024)
                _d_proj = getattr(config, 'scad_d_proj', 128)
                self.register_buffer(
                    'scad_anom_queue',
                    torch.zeros(_q_size, _d_proj),
                )
                self.register_buffer(
                    'scad_queue_ptr', torch.zeros(1, dtype=torch.long)
                )
                self.scad_queue_initialized = False

        # Weight initialization (matching original MAE: xavier_uniform for Linear, constant for LN)
        self.apply(self._init_weights)

        # Override: CNN projection with reduced gain for stability (after global init)
        if self.patchify_mode == 'patch_cnn' and config.use_flatten_linear_embedding:
            if self.cnn_flatten_proj is not None:
                nn.init.xavier_uniform_(self.cnn_flatten_proj[0].weight, gain=0.5)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_embedding_layers(self, config: Config):
        """Build embedding layers based on patchify_mode"""

        if self.patchify_mode == 'patch_cnn':
            # Patchify first, then CNN per patch
            # Each patch: (batch * num_patches, num_features, patch_size)
            # CNN processes each patch independently

            # Get CNN channels from config (default: d_model//2, d_model)
            cnn_channels = config.cnn_channels
            if cnn_channels is None:
                mid_channels = config.d_model // 2
                out_channels = config.d_model
            else:
                mid_channels, out_channels = cnn_channels

            k = config.cnn_kernel_size
            pad = k // 2  # same-padding
            self.patch_cnn = nn.Sequential(
                nn.Conv1d(config.num_features, mid_channels, kernel_size=k, padding=pad),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(),
                nn.Conv1d(mid_channels, out_channels, kernel_size=k, padding=pad),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )

            # Choose embedding method based on config
            if config.use_flatten_linear_embedding:
                # Flatten + Linear projection (preserves patch structure)
                # CNN output: (batch * num_patches, out_channels, patch_size)
                # After flatten: (batch * num_patches, out_channels * patch_size)
                # Linear: (out_channels * patch_size) -> d_model + LayerNorm for stability
                _proj_linear = nn.Linear(out_channels * config.patch_size, config.d_model)
                self.cnn_flatten_proj = nn.Sequential(
                    _proj_linear,
                    nn.LayerNorm(config.d_model)
                )
                self.cnn_projection = None
            else:
                # Mean pooling (simple averaging over patch dimension)
                # CNN output: (batch * num_patches, out_channels, patch_size)
                # After mean: (batch * num_patches, out_channels)
                # Linear: out_channels -> d_model
                self.cnn_flatten_proj = None
                self.cnn_projection = nn.Linear(out_channels, config.d_model)

        elif self.patchify_mode == 'linear':
            # Simple linear embedding (MAE original style)
            if self.use_patch:
                # Patch embedding: (patch_size * num_features) -> d_model
                self.patch_embed = nn.Linear(config.patch_size * config.num_features, config.d_model)
            else:
                # Token-based: num_features -> d_model
                self.input_projection = nn.Linear(config.num_features, config.d_model)
        else:
            raise ValueError(f"Unknown patchify_mode: {self.patchify_mode}")


    def _embed_input(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input based on patchify_mode

        Args:
            x: (batch, seq_length, num_features)
        Returns:
            x_embed: (seq_len, batch, d_model) where seq_len is num_patches or seq_length
        """
        batch_size, seq_length, num_features = x.shape

        if self.patchify_mode == 'patch_cnn':
            # Patchify first, then CNN per patch
            if self.use_patch:
                # (batch, seq, features) -> (batch, num_patches, patch_size, features)
                x_patches = x.reshape(batch_size, self.num_patches, self.patch_size, num_features)
                # (batch, num_patches, patch_size, features) -> (batch * num_patches, features, patch_size)
                x_patches = x_patches.reshape(batch_size * self.num_patches, self.patch_size, num_features)
                x_patches = x_patches.transpose(1, 2)  # (batch * num_patches, features, patch_size)

                # Apply CNN per patch
                x_cnn = self.patch_cnn(x_patches)  # (batch * num_patches, out_channels, patch_size)

                # Choose embedding method based on config
                if self.config.use_flatten_linear_embedding:
                    # Flatten + Linear (preserves patch structure)
                    x_flat = x_cnn.reshape(batch_size * self.num_patches, -1)  # (batch*num_patches, out_channels*patch_size)
                    x_embed = self.cnn_flatten_proj(x_flat)  # (batch * num_patches, d_model)
                else:
                    # Mean pooling (simple averaging)
                    x_mean = x_cnn.mean(dim=2)  # (batch * num_patches, out_channels)
                    x_embed = self.cnn_projection(x_mean)  # (batch * num_patches, d_model)

                # Reshape back: (batch * num_patches, d_model) -> (batch, num_patches, d_model)
                x_embed = x_embed.reshape(batch_size, self.num_patches, -1)
                x_embed = x_embed.transpose(0, 1)  # (num_patches, batch, d_model)
            else:
                # For token-based, apply CNN per time step (less common)
                x_cnn_input = x.transpose(1, 2)  # (batch, features, seq)
                x_cnn = self.patch_cnn(x_cnn_input)  # (batch, d_model, seq)
                x_embed = x_cnn.transpose(1, 2).transpose(0, 1)  # (seq, batch, d_model)

        elif self.patchify_mode == 'linear':
            if self.use_patch:
                # Patchify: (batch, seq, features) -> (batch, num_patches, patch_size * features)
                x_patches = self._patchify_input(x)
                # Embed: (batch, num_patches, patch_size * features) -> (batch, num_patches, d_model)
                x_embed = self.patch_embed(x_patches)
                x_embed = x_embed.transpose(0, 1)  # (num_patches, batch, d_model)
            else:
                # Token-based linear projection
                x_embed = self.input_projection(x)  # (batch, seq, d_model)
                x_embed = x_embed.transpose(0, 1)  # (seq, batch, d_model)
        else:
            raise ValueError(f"Unknown patchify_mode: {self.patchify_mode}")

        return x_embed

    def _patchify_input(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify raw input (for linear mode)
        Args:
            x: (batch, seq_length, num_features)
        Returns:
            patches: (batch, num_patches, patch_size * num_features)
        """
        batch_size, seq_length, num_features = x.shape
        # Reshape to patches
        x = x.reshape(batch_size, self.num_patches, self.patch_size * num_features)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to time series
        Args:
            x: (batch, num_patches, patch_size * num_features)
        Returns:
            time_series: (batch, seq_length, num_features)
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_patches * self.patch_size, self.config.num_features)
        return x

    def random_masking(
        self, x: torch.Tensor, masking_ratio: float, mask_only: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random masking for patches/tokens (patch strategy)

        Args:
            x: (seq_len, batch_size, d_model)
            masking_ratio: ratio of patches to mask
            mask_only: if True, skip x_masked computation and return x unchanged
                       (used when mask_after_encoder=True, where x_masked is not needed)
        Returns:
            x_masked: masked input (or original x if mask_only=True)
            mask: binary mask (1=keep, 0=masked) - (seq_len, batch_size)
        """
        # Handle masking_ratio as either scalar or tensor
        masking_val = masking_ratio.item() if torch.is_tensor(masking_ratio) else masking_ratio
        if not self.config.use_masking or masking_val == 0:
            return x, torch.ones(x.size(0), x.size(1), device=x.device)

        seq_len, batch_size, d_model = x.shape
        num_keep = round(seq_len * (1 - masking_val))

        # Random selection of patches to keep
        noise = torch.rand(seq_len, batch_size, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        # Create binary mask
        mask = torch.zeros(seq_len, batch_size, device=x.device)
        mask[:num_keep, :] = 1
        mask = torch.gather(mask, dim=0, index=ids_restore)

        if mask_only:
            return x, mask

        # Apply mask (use default mask token - teacher for consistency)
        mask_token = self._get_mask_token('teacher')
        mask_tokens = mask_token.repeat(seq_len, batch_size, 1)
        x_masked = x * mask.unsqueeze(-1) + mask_tokens * (1 - mask.unsqueeze(-1))

        return x_masked, mask

    def _get_mask_token(self, for_decoder: str = 'shared') -> torch.Tensor:
        """Get the appropriate mask token based on configuration.

        Args:
            for_decoder: 'teacher', 'student', or 'shared'
        Returns:
            mask_token: (1, 1, d_model)
        """
        if self.shared_mask_token:
            return self.mask_token
        else:
            if for_decoder == 'teacher':
                return self.teacher_mask_token if hasattr(self, 'teacher_mask_token') else self.mask_token
            elif for_decoder == 'student':
                return self.student_mask_token if hasattr(self, 'student_mask_token') else self.mask_token
            else:
                # Fallback to teacher token
                return self.teacher_mask_token if hasattr(self, 'teacher_mask_token') else self.mask_token

    def _encode_visible_only(
        self,
        x_embed: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode only visible patches (standard MAE style).

        Args:
            x_embed: (seq_len, batch, d_model) - embedded patches
            mask: (seq_len, batch) - binary mask (1=keep, 0=masked)
        Returns:
            latent: (num_visible, batch, d_model) - encoded visible patches
            ids_restore: (seq_len, batch) - indices to restore original order
        """
        seq_len, batch_size, d_model = x_embed.shape
        visible_counts = mask.sum(dim=0)  # (batch,)
        # num_keep = MAX visible across the batch. When masking is uniform (eval / default /
        # force_mask_all_anomaly with no over-budget sample) this equals the common count, so the
        # original path below runs byte-identical. When RAGGED (exp337 over-budget samples),
        # num_keep is the least-masked sample's visible count; over-masked samples get padded up.
        num_keep = int(visible_counts.max().item())
        _is_uniform = bool((visible_counts == num_keep).all())

        # Sort patches: visible first, then masked
        # noise gives same order for patches with same mask value, preserving relative order
        noise = torch.rand(seq_len, batch_size, device=x_embed.device)
        # Make visible patches (mask=1) have lower sorting values
        ids_shuffle = torch.argsort(mask * 1000 + noise, dim=0, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        # Keep only visible patches
        ids_keep = ids_shuffle[:num_keep, :]  # (num_keep, batch)

        # Gather visible patches
        ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, d_model)
        x_visible = torch.gather(x_embed, dim=0, index=ids_keep_expanded)  # (num_keep, batch, d_model)

        # Add positional encoding using ORIGINAL positions (not sequential 0,1,2,...)
        # ids_keep contains the original positions of visible patches
        # Gather the correct positional encodings for those positions
        pe_for_visible = torch.gather(
            self.pos_encoder.pe[:seq_len].expand(-1, batch_size, -1),  # (seq_len, batch, d_model)
            dim=0,
            index=ids_keep_expanded  # (num_keep, batch, d_model)
        )
        x_visible_with_pos = x_visible + pe_for_visible

        # Encode only visible patches. self._last_encode_kpm carries the padding mask to
        # _insert_mask_tokens_and_unshuffle (None in the uniform/eval/default path → that fn is a
        # no-op there, byte-identical).
        if _is_uniform:
            self._last_encode_kpm = None
            latent = self.encoder(x_visible_with_pos)  # (num_keep, batch, d_model)
        else:
            # RAGGED (exp337): rows >= visible_counts[b] of ids_keep are masked patches gathered
            # into the visible region. Mark them as key-padding so the encoder ignores them as
            # keys (real visible patches attend only to real visible keys; the seq_len-1 cap in the
            # force block guarantees >=1 real visible key per sample → no all-padding NaN).
            _pad = (torch.arange(num_keep, device=x_embed.device).unsqueeze(1)
                    >= visible_counts.unsqueeze(0))                       # (num_keep, batch) True=pad
            self._last_encode_kpm = _pad
            latent = self.encoder(x_visible_with_pos, src_key_padding_mask=_pad.transpose(0, 1))

        return latent, ids_restore

    def _insert_mask_tokens_and_unshuffle(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
        seq_len: int,
        mask_token: torch.Tensor
    ) -> torch.Tensor:
        """Insert mask tokens at masked positions and restore original order.

        Args:
            latent: (num_visible, batch, d_model) - encoded visible patches
            ids_restore: (seq_len, batch) - indices to restore original order
            seq_len: original sequence length (num_patches)
            mask_token: (1, 1, d_model) - mask token to insert
        Returns:
            latent_full: (seq_len, batch, d_model) - full sequence with mask tokens
        """
        num_visible, batch_size, d_model = latent.shape

        # [exp337] ragged path: rows flagged as padding by _encode_visible_only are masked patches
        # that got gathered into the visible region — overwrite them with the mask token so they
        # become mask tokens after unshuffle (visible/masked partition then exact per sample).
        # Uniform / eval / default path: _last_encode_kpm is None → skipped → byte-identical.
        _kpm = getattr(self, '_last_encode_kpm', None)
        if _kpm is not None and _kpm.shape[0] == num_visible:
            latent = torch.where(_kpm.unsqueeze(-1), mask_token.to(latent.dtype), latent)

        num_masked = seq_len - num_visible

        # Create mask tokens for masked positions
        mask_tokens = mask_token.expand(num_masked, batch_size, -1)  # (num_masked, batch, d_model)

        # Concatenate visible patches and mask tokens
        latent_with_masks = torch.cat([latent, mask_tokens], dim=0)  # (seq_len, batch, d_model)

        # Unshuffle to restore original order
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, d_model)
        latent_full = torch.gather(latent_with_masks, dim=0, index=ids_restore_expanded)

        return latent_full

    def update_teacher_output_ema(self, momentum: float):
        """[신규 2026-06-01] EMA teacher 모듈을 live teacher 가중치로 갱신(매 optimizer step 호출).
        θ_ema ← m·θ_ema + (1−m)·θ_live. 대상: teacher_decoder, teacher_output_projection,
        (shared_decoder), teacher mask token. encoder는 갱신 안 함(제외). use_teacher_output_ema=False면 no-op."""
        if not self._has_teacher_output_ema:
            return
        m = float(momentum)
        with torch.no_grad():
            _pairs = [(self.ema_teacher_decoder, self.teacher_decoder),
                      (self.ema_teacher_output_projection, self.teacher_output_projection)]
            if self.ema_shared_decoder is not None and getattr(self, 'shared_decoder', None) is not None:
                _pairs.append((self.ema_shared_decoder, self.shared_decoder))
            for _ema_mod, _live_mod in _pairs:
                for _pe, _pl in zip(_ema_mod.parameters(), _live_mod.parameters()):
                    _pe.mul_(m).add_(_pl.detach(), alpha=1.0 - m)
                for _be, _bl in zip(_ema_mod.buffers(), _live_mod.buffers()):
                    _be.mul_(m).add_(_bl.detach(), alpha=1.0 - m)
            _live_tok = self.mask_token if self.shared_mask_token else self.teacher_mask_token
            self.ema_teacher_mask_token.mul_(m).add_(_live_tok.detach(), alpha=1.0 - m)

    def reset_teacher_output_ema(self):
        """[신규 2026-06-01] EMA 모듈을 현재 live teacher 가중치로 재초기화(early-stop revert 시 사용)."""
        if not self._has_teacher_output_ema:
            return
        with torch.no_grad():
            self.ema_teacher_decoder.load_state_dict(self.teacher_decoder.state_dict())
            self.ema_teacher_output_projection.load_state_dict(self.teacher_output_projection.state_dict())
            if self.ema_shared_decoder is not None and getattr(self, 'shared_decoder', None) is not None:
                self.ema_shared_decoder.load_state_dict(self.shared_decoder.state_dict())
            _live_tok = self.mask_token if self.shared_mask_token else self.teacher_mask_token
            self.ema_teacher_mask_token.copy_(_live_tok.detach())

    def forward(
        self,
        x: torch.Tensor,
        masking_ratio: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        teacher_only: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_length, num_features)
            masking_ratio: ratio of patches to mask
            mask: optional pre-defined mask
            point_labels: (batch_size, seq_length) for force_mask_anomaly
            teacher_only: if True, skip student decoder + GRL classifier + SCAD head.
                          Saves ~22% of transformer forward compute during
                          teacher_only_warmup epochs. Returns student_output=None.
                          Side-effect attrs (self._student_hidden, _grl_cls_logits,
                          _scad_z) are set to None so consumers via getattr()
                          see a clean signal rather than stale prior-batch values.
                          Default False preserves backward compat for all callers
                          (evaluator, training_visualizer). Added 2026-05-29.
        Returns:
            teacher_output, student_output, mask
            student_output is None when teacher_only=True.
        """
        if masking_ratio is None:
            masking_ratio = self.config.masking_ratio

        _profiling = getattr(self, '_profiling', False)
        if _profiling:
            torch.cuda.synchronize()
            _t0 = time.time()

        batch_size, seq_length, num_features = x.shape

        # RevIN normalize (Step 6 in plan): per-window mean/std on top of loader-level zscore.
        # Computes stats from current window, stores on self.revin for later denormalize().
        # Upstream pattern matches ModernTCN/CATCH/TimesNet/PatchTST.
        if self.use_revin:
            x = self.revin.normalize(
                x,
                point_labels=point_labels if self.training else None,
                visible_only=self.revin_visible_only,
            )

        # Embed input based on patchify_mode
        x_embed = self._embed_input(x)  # (seq_len, batch, d_model)
        seq_len = x_embed.size(0)

        if _profiling:
            torch.cuda.synchronize()
            _t_embed = time.time()

        # Masking (patch strategy)
        mask_provided_externally = mask is not None
        x_masked = None  # Only computed when mask_after_encoder=False

        if mask is None:
            x_masked, mask = self.random_masking(
                x_embed, masking_ratio, mask_only=self.mask_after_encoder
            )
            # mask is (seq_len, batch) for patch strategy
            patch_mask = mask
        else:
            # Handle pre-defined mask (from external source like evaluator)
            # External mask is always (batch, seq_length) format
            if self.use_patch and mask.shape[1] != self.num_patches:
                mask_reshaped = mask.reshape(batch_size, self.num_patches, self.patch_size)
                mask = mask_reshaped[:, :, -1]

            mask = mask.transpose(0, 1)  # (seq_len, batch)
            patch_mask = mask
            if not self.mask_after_encoder:
                default_mask_token = self._get_mask_token('teacher')
                mask_tokens = default_mask_token.repeat(x_embed.size(0), x_embed.size(1), 1)
                x_masked = x_embed * mask.unsqueeze(-1) + mask_tokens * (1 - mask.unsqueeze(-1))

        # Force mask patches containing anomalies during training (fixed budget)
        # Masking budget is always exactly target_num_masked, with anomaly patches
        # prioritized. If anomaly patches exceed the budget, excess remain visible
        # as encoder context (acceptable trade-off to maintain uniform batch masking).
        if (self.training and self.config.force_mask_anomaly and
            point_labels is not None):
            current_seq_len = patch_mask.shape[0]

            if self.use_patch:
                patch_labels = point_labels.reshape(batch_size, self.num_patches, self.patch_size)
                anomaly_patches = (patch_labels.sum(dim=2) > 0).float()
                anomaly_patches = anomaly_patches.transpose(0, 1)  # (seq_len, batch)
            else:
                anomaly_patches = point_labels.transpose(0, 1).float()

            target_num_masked = round(current_seq_len * masking_ratio)

            # Priority-based masking: anomaly patches get high priority (1000+noise),
            # normal patches get low priority (0+noise). Top target_num_masked are masked.
            noise = torch.rand(current_seq_len, batch_size, device=patch_mask.device)
            masking_priority = anomaly_patches * 1000 + noise

            if (getattr(self.config, 'force_mask_all_anomaly', False)
                    and self.config.use_masking and masking_ratio > 0):
                # [exp337] per-sample variable budget: mask exactly K_s = min(max(budget, N_a),
                # seq_len-1) patches per sample. anomaly patches (priority ~1000) fill the top
                # ranks → ALL anomalies masked; normals fill the rest up to budget. Cap keeps >=1
                # visible. Produces a RAGGED mask (variable masked count per sample) — handled by
                # _encode_visible_only's padding/src_key_padding_mask path (mask_after_encoder=True).
                _n_anom = anomaly_patches.sum(dim=0)                                    # (batch,)
                _k_s = torch.clamp(torch.clamp(_n_anom, min=float(target_num_masked)),
                                   max=float(current_seq_len - 1))                      # (batch,)
                _rank = torch.argsort(torch.argsort(masking_priority, dim=0, descending=True), dim=0)
                patch_mask = (_rank >= _k_s.unsqueeze(0)).to(masking_priority.dtype)    # 1=keep, 0=mask top-K_s
                mask = patch_mask
            else:
                # === LEGACY (byte-identical to pre-2026-06-22): fixed uniform budget ===
                ids_sorted = torch.argsort(masking_priority, dim=0, descending=True)
                patch_mask = torch.ones(current_seq_len, batch_size, device=patch_mask.device)
                patch_mask.scatter_(0, ids_sorted[:target_num_masked, :], 0.0)
                mask = patch_mask

            # Re-apply masking for non-mask_after_encoder mode
            if not self.mask_after_encoder:
                reapply_mask_token = self._get_mask_token('teacher')
                mask_tokens = reapply_mask_token.repeat(x_embed.size(0), x_embed.size(1), 1)
                x_masked = x_embed * patch_mask.unsqueeze(-1) + mask_tokens * (1 - patch_mask.unsqueeze(-1))

        if _profiling:
            torch.cuda.synchronize()
            _t_mask = time.time()

        # === Encoding ===
        if self.mask_after_encoder:
            # Standard MAE: encode visible patches only, insert mask tokens before decoder
            latent_visible, ids_restore = self._encode_visible_only(x_embed, patch_mask)
        else:
            # Current behavior: mask tokens go through encoder
            x_masked = self.pos_encoder(x_masked)
            latent = self.encoder(x_masked)

        if _profiling:
            torch.cuda.synchronize()
            _t_encode = time.time()

        # === Decoding ===
        # Option 2: No separate tgt - decoder uses self-attention on full sequence with PE

        teacher_output = None
        student_output = None

        # [신규 2026-06-15] decoder_embed: encoder latent 을 decoder 폭으로 좁힘 (off면 Identity).
        # teacher 쪽에서 1회 계산(grad→decoder_embed+encoder); student 는 .detach()로 받음
        # (현재 encoder↔student detach 비대칭과 동일). mask_after_encoder=True 경로 전용.
        if self.mask_after_encoder:
            _lat_dec = self.decoder_embed(latent_visible)

        if self.config.use_teacher and self.teacher_decoder is not None:
            if self.mask_after_encoder:
                # Insert mask tokens before teacher decoder
                teacher_mask_token = self._get_mask_token('teacher')
                teacher_latent = self._insert_mask_tokens_and_unshuffle(
                    _lat_dec, ids_restore, seq_len, teacher_mask_token
                )
                # Add decoder positional encoding (only for TransformerEncoder)
                if self.config.use_transformer_encoder_decoder:
                    teacher_latent = teacher_latent + self.decoder_pos_encoder.pe[:seq_len]
            else:
                teacher_latent = latent
                # Add decoder positional encoding (only for TransformerEncoder)
                if self.config.use_transformer_encoder_decoder:
                    teacher_latent = teacher_latent + self.decoder_pos_encoder.pe[:seq_len]

            # Apply shared decoder first if exists (trained with teacher)
            if self.shared_decoder is not None:
                if self.config.use_transformer_encoder_decoder:
                    # TransformerEncoder: single input
                    teacher_latent = self.shared_decoder(teacher_latent)
                else:
                    # TransformerDecoder: (tgt, memory)
                    teacher_latent = self.shared_decoder(teacher_latent, latent_visible)

            # Apply teacher decoder
            if self.config.use_transformer_encoder_decoder:
                # TransformerEncoder: self-attention only
                teacher_hidden = self.teacher_decoder(teacher_latent)
            else:
                # TransformerDecoder: cross-attention with encoder output
                teacher_hidden = self.teacher_decoder(teacher_latent, latent_visible)

            # Store for FM distance computation (evaluator/loss)
            self._teacher_hidden = teacher_hidden  # (num_patches, batch, d_model)

            teacher_output = self.teacher_output_projection(teacher_hidden)
            teacher_output = teacher_output.transpose(0, 1)

            if self.use_patch:
                teacher_output = self.unpatchify(teacher_output)

            # RevIN denormalize (Step 8 in plan): inverse per-window normalization.
            # Output is now in the same scale as the loader-zscored input,
            # so loss vs original_input and anomaly scoring operate in that consistent space.
            if self.use_revin:
                teacher_output = self.revin.denormalize(teacher_output)

            # === [신규 2026-06-01] EMA teacher 출력 (student discrepancy 표적 전용) ===
            # _ema_active(post-warmup & freeze=False, trainer가 매 epoch 설정)이고 teacher_only가
            # 아닐 때만, EMA 사본 모듈로 한 번 더 decode. encoder latent(latent_visible)는 live 재사용
            # (encoder 제외). no_grad로 backprop 차단. 결과는 loss의 discrepancy 표적으로만 사용되며
            # 추론/scoring/dynamic_margin은 live teacher_output을 그대로 쓴다.
            self._ema_teacher_output = None
            if (self._has_teacher_output_ema and self._ema_active and not teacher_only
                    and self.training):
                with torch.no_grad():
                    # 부모 model.train()이 자식 EMA 모듈도 train 모드로 되돌리므로, 매 EMA forward
                    # 직전 eval 재확정 → dropout off의 결정론적 평탄 표적 보장(불안정성 완화 의도).
                    self.ema_teacher_decoder.eval()
                    self.ema_teacher_output_projection.eval()
                    if self.ema_shared_decoder is not None:
                        self.ema_shared_decoder.eval()
                    if self.mask_after_encoder:
                        _ema_latent = self._insert_mask_tokens_and_unshuffle(
                            latent_visible, ids_restore, seq_len, self.ema_teacher_mask_token)
                        if self.config.use_transformer_encoder_decoder:
                            _ema_latent = _ema_latent + self.decoder_pos_encoder.pe[:seq_len]
                    else:
                        _ema_latent = latent
                        if self.config.use_transformer_encoder_decoder:
                            _ema_latent = _ema_latent + self.decoder_pos_encoder.pe[:seq_len]
                    if self.ema_shared_decoder is not None:
                        if self.config.use_transformer_encoder_decoder:
                            _ema_latent = self.ema_shared_decoder(_ema_latent)
                        else:
                            _ema_latent = self.ema_shared_decoder(_ema_latent, latent_visible)
                    if self.config.use_transformer_encoder_decoder:
                        _ema_hidden = self.ema_teacher_decoder(_ema_latent)
                    else:
                        _ema_hidden = self.ema_teacher_decoder(_ema_latent, latent_visible)
                    _ema_out = self.ema_teacher_output_projection(_ema_hidden).transpose(0, 1)
                    if self.use_patch:
                        _ema_out = self.unpatchify(_ema_out)
                    if self.use_revin:
                        _ema_out = self.revin.denormalize(_ema_out)
                    self._ema_teacher_output = _ema_out

        if _profiling:
            torch.cuda.synchronize()
            _t_teacher = time.time()

        if self.config.use_student and self.student_decoder is not None and not teacher_only:
            if self.mask_after_encoder:
                # Insert mask tokens before student decoder
                # Detach encoder output to prevent encoder updates from student loss
                student_mask_token = self._get_mask_token('student')
                student_latent = self._insert_mask_tokens_and_unshuffle(
                    _lat_dec.detach(), ids_restore, seq_len, student_mask_token
                )
                # Add decoder positional encoding (only for TransformerEncoder)
                if self.config.use_transformer_encoder_decoder:
                    student_latent = student_latent + self.decoder_pos_encoder.pe[:seq_len]
            else:
                # Detach encoder output to prevent encoder updates from student loss
                student_latent = latent.detach()
                # Add decoder positional encoding (only for TransformerEncoder)
                if self.config.use_transformer_encoder_decoder:
                    student_latent = student_latent + self.decoder_pos_encoder.pe[:seq_len]

            # Student uses its own decoder directly (no shared decoder)
            # [신규 2026-06-15] grl_attach_layer='first' → GRL classifier 가 student decoder 의
            # FIRST-layer 출력(h1)을 읽음. reconstruction/FM 는 그대로 최종 hidden(h2) 사용.
            # 'last'(기본)는 기존 호출 그대로 → byte-identical.
            _grl_first = (self.training and getattr(self.config, 'grl_attach_layer', 'last') == 'first'
                          and hasattr(self, 'anomaly_classifier'))
            if self.config.use_transformer_encoder_decoder:
                # TransformerEncoder: self-attention only
                if _grl_first and len(self.student_decoder.layers) > 1:
                    # 수동 layer loop 로 layer-0 출력(h1) 포착. 최종 norm 까지 적용해 h2 는
                    # self.student_decoder(student_latent) 와 동일(복원/FM 불변).
                    _h = student_latent
                    _grl_src = None
                    for _i, _layer in enumerate(self.student_decoder.layers):
                        _h = _layer(_h)
                        if _i == 0:
                            _grl_src = _h
                    if self.student_decoder.norm is not None:
                        _h = self.student_decoder.norm(_h)
                    student_hidden = _h
                else:
                    student_hidden = self.student_decoder(student_latent)
                    _grl_src = student_hidden
            else:
                # TransformerDecoder: cross-attention with encoder output
                # Use detached encoder output for memory
                student_hidden = self.student_decoder(student_latent, latent_visible.detach())
                _grl_src = student_hidden

            # Store for FM distance computation (evaluator/loss)
            self._student_hidden = student_hidden  # (num_patches, batch, d_model)

            # GRL / WDGRL: adversarial training on student hidden (training only)
            if self.training and hasattr(self, 'anomaly_classifier'):
                # Classifier mode (DANN-style GRL); _grl_src = h1 if grl_attach_layer='first' else h2
                lambda_grl = getattr(self, '_grl_lambda', 0.0)
                cls_logits = self.anomaly_classifier(_grl_src, lambda_grl)
                self._grl_cls_logits = cls_logits.squeeze(-1).transpose(0, 1)
            else:
                self._grl_cls_logits = None
            # WDGRL: critic scores are computed in trainer.py (separate forward pass)

            # SCAD: build the L2-normalized embedding the repulsion is measured on (training only).
            # Note: stored as (num_patches, batch, d) — same layout as student_hidden.
            if self.training and getattr(self, '_scad_apply_space', 'projection') == 'hidden_final':
                # H-SCAD-C: repulsion DIRECTLY on the final student_hidden via a parameter-free
                # LayerNorm (removes per-token mean / DC) + L2 normalize — no projection head.
                # Better aligned with the score path (student_hidden → output projection → discrepancy).
                _d_h = student_hidden.size(-1)
                self._scad_z = F.normalize(F.layer_norm(student_hidden, (_d_h,)), p=2, dim=-1)
            elif self.training and hasattr(self, 'scad_head'):
                self._scad_z = self.scad_head(student_hidden)
            else:
                self._scad_z = None

            student_output = self.student_output_projection(student_hidden)
            student_output = student_output.transpose(0, 1)

            if self.use_patch:
                student_output = self.unpatchify(student_output)

            # RevIN denormalize: inverse per-window normalization.
            # Uses same stats stored by self.revin during normalize() at forward entry.
            if self.use_revin:
                student_output = self.revin.denormalize(student_output)
        else:
            # teacher_only mode (warmup) OR use_student=False / no student_decoder.
            # Explicitly clear side-effect attrs so downstream getattr() reads None
            # rather than stale prior-batch values. student_output stays None.
            # Saves student-decoder forward + projection + GRL classifier + SCAD head.
            self._student_hidden = None
            self._grl_cls_logits = None
            self._scad_z = None

        if _profiling:
            torch.cuda.synchronize()
            _t_student = time.time()
            self._forward_timing = {
                'embed_input_ms': (_t_embed - _t0) * 1000,
                'masking_ms': (_t_mask - _t_embed) * 1000,
                'encoder_ms': (_t_encode - _t_mask) * 1000,
                'teacher_decoder_ms': (_t_teacher - _t_encode) * 1000,
                'student_decoder_ms': (_t_student - _t_teacher) * 1000,
                'forward_total_ms': (_t_student - _t0) * 1000,
            }

        # Handle single decoder case
        if not self.config.use_teacher and self.config.use_student:
            teacher_output = student_output
        elif self.config.use_teacher and not self.config.use_student:
            student_output = teacher_output

        # Convert mask back to (batch, seq_length) format
        if mask_provided_externally:
            # External mask was already (batch, seq_length), now is (seq_len, batch) after transpose
            # Just transpose back
            mask = mask.transpose(0, 1)  # (batch, seq_len)
            if self.use_patch and mask.shape[1] == self.num_patches:
                mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
                mask = mask.reshape(batch_size, seq_length)
        else:
            # patch strategy: mask is (seq_len, batch)
            mask = mask.transpose(0, 1)  # (batch, seq_len)

            if self.use_patch:
                mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
                mask = mask.reshape(batch_size, seq_length)

        return teacher_output, student_output, mask
