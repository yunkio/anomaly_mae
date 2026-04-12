"""
Loss functions for Self-Distilled MAE Anomaly Detection

Supports:
- Patch-level and window-level discrepancy loss
- Multiple margin types: hinge, softplus, dynamic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SelfDistillationLoss(nn.Module):
    """Loss function for self-distilled MAE with configurable discrepancy loss granularity

    Supports two modes:
    - patch_level_loss=True: Compute loss at patch level (each patch has its own gradient direction)
    - patch_level_loss=False: Compute loss at window level (sample-level, original behavior)

    Supports four margin types:
    - 'hinge': relu(margin - discrepancy) - original, hard cutoff at margin
    - 'softplus': log(1 + exp(margin - discrepancy)) - soft version, always has gradient
    - 'dynamic': margin = mu + k*sigma based on normal patches/samples in batch
    - 'none': -discrepancy (no margin, unbounded maximization)
    """

    def __init__(self, config):
        super().__init__()
        self.margin = config.margin
        self.margin_type = config.margin_type
        self.dynamic_margin_k = config.dynamic_margin_k
        self.use_discrepancy = config.use_discrepancy_loss
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.patch_level_loss = config.patch_level_loss

        # GRL
        self.use_grl = getattr(config, 'use_grl', False)
        self.grl_disable_anomaly_loss = getattr(config, 'grl_disable_anomaly_loss', True)

        # Feature Matching
        self.use_feature_matching = getattr(config, 'use_feature_matching', False)
        self.use_output_discrepancy = getattr(config, 'use_output_discrepancy', True)
        self.fm_loss_weight = getattr(config, 'fm_loss_weight', 1.0)
        self.fm_adaptive_lambda = getattr(config, 'fm_adaptive_lambda', False)
        self.fm_distance_metric = getattr(config, 'fm_distance_metric', 'cosine')
        self.anomaly_loss_direction = getattr(config, 'anomaly_loss_direction', 'maximize')
        self.grl_target_mode = getattr(config, 'grl_target_mode', 'patch')

        # Discrepancy loss weights
        self.anomaly_loss_weight = config.anomaly_loss_weight
        self.normal_loss_weight = getattr(config, 'normal_loss_weight', 1.0)

        # GRL pos_weight: dataset-level fixed (set by run_base_experiments.py from actual data)
        self.grl_pos_weight = getattr(config, 'grl_pos_weight', 19.0)
        self.grl_balanced_sampling = getattr(config, 'grl_balanced_sampling', False)
        self.grl_use_focal = getattr(config, 'grl_use_focal', True)

    def _compute_anomaly_loss(
        self,
        discrepancy: torch.Tensor,
        anomaly_mask: torch.Tensor,
        normal_mask: torch.Tensor,
        margin: float
    ) -> torch.Tensor:
        """Compute anomaly loss based on margin_type (for window-level mode)"""
        if self.margin_type == 'none':
            per_sample_loss = -discrepancy
        elif self.margin_type == 'hinge':
            per_sample_loss = F.relu(margin - discrepancy)
        elif self.margin_type == 'softplus':
            per_sample_loss = F.softplus(margin - discrepancy)
        elif self.margin_type == 'dynamic':
            if normal_mask.sum() > 1:
                normal_disc = discrepancy[normal_mask.bool()]
                mu = normal_disc.mean()
                sigma = normal_disc.std() + 1e-4
                dynamic_margin = mu + self.dynamic_margin_k * sigma
            else:
                dynamic_margin = margin
            per_sample_loss = F.relu(dynamic_margin - discrepancy)
        else:
            raise ValueError(f"Unknown margin_type: {self.margin_type}")
        return per_sample_loss

    def _compute_patch_anomaly_loss(
        self,
        patch_discrepancy: torch.Tensor,
        patch_anomaly_mask: torch.Tensor,
        patch_normal_mask: torch.Tensor,
        margin: float
    ) -> torch.Tensor:
        """Compute anomaly loss for each patch based on margin_type

        Args:
            patch_discrepancy: (batch, num_patches) discrepancy per patch
            patch_anomaly_mask: (batch, num_patches) 1 if patch contains anomaly
            patch_normal_mask: (batch, num_patches) 1 if patch is normal
            margin: margin threshold

        Returns:
            per_patch_loss: (batch, num_patches) loss per patch
        """
        if self.margin_type == 'none':
            per_patch_loss = -patch_discrepancy
        elif self.margin_type == 'hinge':
            per_patch_loss = F.relu(margin - patch_discrepancy)
        elif self.margin_type == 'softplus':
            per_patch_loss = F.softplus(margin - patch_discrepancy)
        elif self.margin_type == 'dynamic':
            normal_patches_flat = patch_discrepancy[patch_normal_mask.bool()]
            if normal_patches_flat.numel() > 1:
                mu = normal_patches_flat.mean()
                sigma = normal_patches_flat.std() + 1e-4
                dynamic_margin = mu + self.dynamic_margin_k * sigma
            else:
                dynamic_margin = margin
            per_patch_loss = F.relu(dynamic_margin - patch_discrepancy)
        else:
            raise ValueError(f"Unknown margin_type: {self.margin_type}")

        return per_patch_loss

    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        original_input: torch.Tensor,
        mask: torch.Tensor,
        point_labels: torch.Tensor,
        warmup_factor: float = 1.0,
        teacher_only: bool = False,
        grl_cls_logits: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
        student_hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Args:
            teacher_output: (batch, seq_length, num_features)
            student_output: (batch, seq_length, num_features)
            original_input: (batch, seq_length, num_features)
            mask: (batch, seq_length) - 1=keep, 0=masked
            point_labels: (batch, seq_length) - 1=anomaly, 0=normal
            warmup_factor: factor for anomaly loss warmup
            teacher_only: if True, only compute teacher reconstruction loss (for warm-up epochs)
        """
        batch_size = teacher_output.size(0)
        mask_expanded = mask.unsqueeze(-1)

        # Determine which samples have anomaly in masked region (used for multiple loss computations)
        masked_point_labels = point_labels * (1 - mask)
        has_anomaly_sample = (masked_point_labels.sum(dim=1) > 0).float()  # (batch,)
        is_normal_sample = 1 - has_anomaly_sample

        # Teacher reconstruction loss (total)
        teacher_recon_full = F.mse_loss(
            teacher_output * (1 - mask_expanded),
            original_input * (1 - mask_expanded),
            reduction='none'
        )
        num_features = teacher_recon_full.size(-1)
        teacher_recon_per_sample = teacher_recon_full.sum(dim=(1, 2)) / ((1 - mask_expanded).sum(dim=(1, 2)) * num_features + 1e-4)
        reconstruction_loss = teacher_recon_per_sample.mean()

        # Teacher reconstruction loss by sample type
        teacher_recon_normal = (is_normal_sample * teacher_recon_per_sample).sum() / (is_normal_sample.sum() + 1e-4)
        teacher_recon_anomaly = (has_anomaly_sample * teacher_recon_per_sample).sum() / (has_anomaly_sample.sum() + 1e-4)

        # Student reconstruction loss (for tracking, always computed)
        student_recon_full = F.mse_loss(
            student_output * (1 - mask_expanded),
            original_input * (1 - mask_expanded),
            reduction='none'
        )
        student_recon_per_sample = student_recon_full.sum(dim=(1, 2)) / ((1 - mask_expanded).sum(dim=(1, 2)) * num_features + 1e-4)
        student_recon_normal_metric = (is_normal_sample * student_recon_per_sample).sum() / (is_normal_sample.sum() + 1e-4)
        student_recon_anomaly_metric = (has_anomaly_sample * student_recon_per_sample).sum() / (has_anomaly_sample.sum() + 1e-4)

        # Feature-level stats from teacher_recon_full (B, L, F) — masked positions only
        # Uses existing tensor, single reduction per stat: ~0.03ms/batch overhead
        _mask_count_per_feature = (1 - mask_expanded).sum(dim=(0, 1)) + 1e-4  # (F,)
        _recon_masked = teacher_recon_full  # already masked at computation (line 137-138)
        feature_recon_mean = (_recon_masked.sum(dim=(0, 1)) / _mask_count_per_feature).detach()  # (F,)
        feature_recon_max = _recon_masked.max(dim=0).values.max(dim=0).values.detach()  # (F,)

        if self.use_discrepancy and not teacher_only:
            # Compute per-position discrepancy: (batch, seq_length, num_features)
            discrepancy_full = (teacher_output.detach() - student_output) ** 2

            # Feature-level discrepancy stats (B, L, F) → (F,) — masked positions only
            _disc_masked = discrepancy_full * (1 - mask_expanded)
            feature_disc_mean = (_disc_masked.sum(dim=(0, 1)) / _mask_count_per_feature).detach()  # (F,)
            feature_disc_max = _disc_masked.max(dim=0).values.max(dim=0).values.detach()  # (F,)

            if self.patch_level_loss:
                # ============== PATCH-LEVEL LOSS ==============
                # Each patch gets its own gradient direction based on its anomaly status

                # Reshape to (batch, num_patches, patch_size, num_features)
                discrepancy_patches = discrepancy_full.reshape(
                    batch_size, self.num_patches, self.patch_size, -1
                )
                mask_patches = mask.reshape(batch_size, self.num_patches, self.patch_size)
                point_labels_patches = point_labels.reshape(batch_size, self.num_patches, self.patch_size)

                # Per-patch discrepancy (only on masked positions)
                mask_inverse_patches = 1 - mask_patches
                mask_inverse_expanded = mask_inverse_patches.unsqueeze(-1)

                patch_discrepancy_sum = (discrepancy_patches * mask_inverse_expanded).sum(dim=(2, 3))
                patch_mask_count = mask_inverse_patches.sum(dim=2) * discrepancy_full.size(-1) + 1e-4
                patch_discrepancy = patch_discrepancy_sum / patch_mask_count

                # Determine patch-level anomaly status
                masked_anomaly_patches = point_labels_patches * mask_inverse_patches
                patch_has_anomaly = (masked_anomaly_patches.sum(dim=2) > 0).float()
                patch_is_normal = 1 - patch_has_anomaly
                patch_has_masked = (mask_inverse_patches.sum(dim=2) > 0).float()

                # Normal patch loss
                normal_patch_mask = patch_is_normal * patch_has_masked
                anomaly_patch_mask = patch_has_anomaly * patch_has_masked

                if self.use_output_discrepancy:
                    normal_loss = (normal_patch_mask * patch_discrepancy).sum() / (normal_patch_mask.sum() + 1e-4)
                    normal_loss = normal_loss * self.normal_loss_weight

                    # Anomaly patch loss (with weight multiplier)
                    if self.use_grl and self.grl_disable_anomaly_loss:
                        # GRL handles anomaly disc generation → anomaly_loss disabled
                        anomaly_loss = torch.tensor(0.0, device=teacher_output.device)
                    elif self.anomaly_loss_direction == 'minimize':
                        # Same direction as normal: minimize discrepancy on anomaly patches too
                        anomaly_loss = (anomaly_patch_mask * patch_discrepancy).sum() / (anomaly_patch_mask.sum() + 1e-4)
                        anomaly_loss = warmup_factor * anomaly_loss * self.anomaly_loss_weight
                    else:
                        # Default: maximize discrepancy on anomaly patches
                        per_patch_anomaly_loss = self._compute_patch_anomaly_loss(
                            patch_discrepancy, anomaly_patch_mask, normal_patch_mask, self.margin
                        )
                        anomaly_loss = (anomaly_patch_mask * per_patch_anomaly_loss).sum() / (anomaly_patch_mask.sum() + 1e-4)
                        anomaly_loss = warmup_factor * anomaly_loss * self.anomaly_loss_weight
                else:
                    # OD disabled — only FM contributes to discrepancy_loss
                    normal_loss = torch.tensor(0.0, device=teacher_output.device)
                    anomaly_loss = torch.tensor(0.0, device=teacher_output.device)

                # Forward-direction discrepancy on anomaly patches (always computed for metrics)
                anomaly_disc_forward = (anomaly_patch_mask * patch_discrepancy).sum() / (anomaly_patch_mask.sum() + 1e-4)

                # GRL classifier loss (reuses patch_has_anomaly, patch_has_masked)
                if grl_cls_logits is not None:
                    valid = patch_has_masked.bool()
                    valid_logits = grl_cls_logits[valid]    # (N,)
                    if self.grl_target_mode == 'window':
                        # Window-level label: all patches in anomaly window get target=1
                        _window_label = has_anomaly_sample.unsqueeze(1).expand_as(patch_has_anomaly)
                        valid_targets = _window_label[valid]  # (N,)
                    else:
                        # Patch-level label (default)
                        valid_targets = patch_has_anomaly[valid]  # (N,)

                    _pos_count = valid_targets.sum()
                    if _pos_count == 0:
                        # No anomaly in this batch → skip GRL loss
                        _grl_results = {
                            'grl_cls_loss_tensor': None,
                            'grl_cls_loss': 0.0,
                            'grl_balanced_acc': 0.5,
                            'grl_anomaly_acc': 0.0,
                            'grl_normal_acc': 1.0,
                        }
                    else:
                        # Balanced accuracy on ALL patches (monitoring, no grad)
                        with torch.no_grad():
                            _pos_m_all = valid_targets > 0.5
                            _neg_m_all = ~_pos_m_all
                            _tpr = (valid_logits[_pos_m_all] > 0).float().mean() if _pos_m_all.any() else torch.tensor(0.5, device=teacher_output.device)
                            _tnr = (valid_logits[_neg_m_all] <= 0).float().mean() if _neg_m_all.any() else torch.tensor(0.5, device=teacher_output.device)
                            _balanced_acc = (_tpr + _tnr) / 2

                        # Select subset for loss computation
                        if self.grl_balanced_sampling:
                            # Downsample normal to match anomaly count → 1:1 balanced
                            _n_pos = int(_pos_count.item())
                            _pos_idx = _pos_m_all.nonzero(as_tuple=True)[0]
                            _neg_idx = _neg_m_all.nonzero(as_tuple=True)[0]
                            _n_neg_keep = min(_n_pos, len(_neg_idx))
                            _neg_keep = _neg_idx[torch.randperm(len(_neg_idx), device=_neg_idx.device)[:_n_neg_keep]]
                            _keep = torch.cat([_pos_idx, _neg_keep])
                            _loss_logits = valid_logits[_keep]
                            _loss_targets = valid_targets[_keep]
                            # No pos_weight needed (already 1:1)
                            _bce = F.binary_cross_entropy_with_logits(
                                _loss_logits, _loss_targets, reduction='none')
                        else:
                            # Original: all patches with pos_weight
                            _loss_logits = valid_logits
                            _loss_targets = valid_targets
                            _pos_weight = torch.tensor(
                                self.grl_pos_weight, device=teacher_output.device
                            ).expand_as(_loss_logits)
                            _bce = F.binary_cross_entropy_with_logits(
                                _loss_logits, _loss_targets,
                                pos_weight=_pos_weight, reduction='none')

                        if self.grl_use_focal:
                            _p_t = torch.exp(-_bce)
                            _focal = ((1 - _p_t) ** 2.0) * _bce
                            grl_cls_loss = _focal.mean()
                        else:
                            grl_cls_loss = _bce.mean()

                        _grl_results = {
                            'grl_cls_loss_tensor': grl_cls_loss,
                            'grl_cls_loss': grl_cls_loss.item(),
                            'grl_balanced_acc': _balanced_acc.item(),
                            'grl_anomaly_acc': _tpr.item(),
                            'grl_normal_acc': _tnr.item(),
                        }
                else:
                    _grl_results = None

                sample_discrepancy = (patch_discrepancy * patch_has_masked).sum(dim=1) / (patch_has_masked.sum(dim=1) + 1e-4)

            else:
                # ============== WINDOW-LEVEL LOSS ==============
                # Original behavior: sample-level classification

                discrepancy_masked = discrepancy_full * (1 - mask_expanded)
                sample_discrepancy = discrepancy_masked.sum(dim=(1, 2)) / ((1 - mask_expanded).sum(dim=(1, 2)) * discrepancy_full.size(-1) + 1e-4)

                # Sample has anomaly if ANY masked position has anomaly
                masked_point_labels = point_labels * (1 - mask)
                has_anomaly_in_masked = (masked_point_labels.sum(dim=1) > 0).float()

                normal_mask = (1 - has_anomaly_in_masked)
                anomaly_mask = has_anomaly_in_masked

                if self.use_output_discrepancy:
                    # Normal loss: minimize discrepancy for normal samples
                    normal_loss = (normal_mask * sample_discrepancy).sum() / (normal_mask.sum() + 1e-4)
                    normal_loss = normal_loss * self.normal_loss_weight

                    # Anomaly loss: encourage discrepancy to be large (with weight multiplier)
                    per_sample_anomaly_loss = self._compute_anomaly_loss(
                        sample_discrepancy, anomaly_mask, normal_mask, self.margin
                    )
                    anomaly_loss = (anomaly_mask * per_sample_anomaly_loss).sum() / (anomaly_mask.sum() + 1e-4)
                    anomaly_loss = warmup_factor * anomaly_loss * self.anomaly_loss_weight
                else:
                    # OD disabled — only FM contributes to discrepancy_loss
                    normal_loss = torch.tensor(0.0, device=teacher_output.device)
                    anomaly_loss = torch.tensor(0.0, device=teacher_output.device)

                # Forward-direction discrepancy on anomaly samples (always computed for metrics)
                anomaly_disc_forward = (anomaly_mask * sample_discrepancy).sum() / (anomaly_mask.sum() + 1e-4)

            # Feature matching loss (cosine distance on masked normal patches)
            fm_loss = torch.tensor(0.0, device=teacher_output.device)
            if (self.use_feature_matching and teacher_hidden is not None
                    and student_hidden is not None and not teacher_only):
                # teacher_hidden, student_hidden: (num_patches, batch, d_model)
                # Compute per-patch distance
                if self.fm_distance_metric == 'l2':
                    _fm_dist = ((teacher_hidden.detach() - student_hidden) ** 2).mean(dim=-1)  # (num_patches, batch)
                else:  # cosine (default)
                    _cos_sim = F.cosine_similarity(
                        teacher_hidden.detach(), student_hidden, dim=-1)  # (num_patches, batch)
                    _fm_dist = 1 - _cos_sim  # (num_patches, batch)
                _fm_dist = _fm_dist.transpose(0, 1)  # (batch, num_patches)

                if self.patch_level_loss:
                    # Normal masked patches only
                    _fm_normal_mask = patch_is_normal * patch_has_masked
                    fm_loss = (_fm_normal_mask * _fm_dist).sum() / (_fm_normal_mask.sum() + 1e-4)
                else:
                    fm_loss = _fm_dist.mean()

            if self.fm_adaptive_lambda:
                # FM excluded from total — trainer will add with adaptive weight
                discrepancy_loss = normal_loss + anomaly_loss
            else:
                discrepancy_loss = normal_loss + anomaly_loss + self.fm_loss_weight * fm_loss
            total_loss = reconstruction_loss + discrepancy_loss
        else:
            discrepancy_loss = torch.tensor(0.0, device=teacher_output.device)
            normal_loss = torch.tensor(0.0, device=teacher_output.device)
            anomaly_loss = torch.tensor(0.0, device=teacher_output.device)
            anomaly_disc_forward = torch.tensor(0.0, device=teacher_output.device)
            fm_loss = torch.tensor(0.0, device=teacher_output.device)
            sample_discrepancy = torch.zeros(batch_size, device=teacher_output.device)
            feature_disc_mean = None
            feature_disc_max = None
            total_loss = reconstruction_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'discrepancy_loss': discrepancy_loss.item() if isinstance(discrepancy_loss, torch.Tensor) else discrepancy_loss,
            'normal_loss': normal_loss.item() if isinstance(normal_loss, torch.Tensor) else normal_loss,
            'anomaly_loss': anomaly_loss.item() if isinstance(anomaly_loss, torch.Tensor) else anomaly_loss,
            'mean_discrepancy': sample_discrepancy.mean().item(),
            # Detailed metrics for visualization
            'teacher_recon_normal': teacher_recon_normal.item(),
            'teacher_recon_anomaly': teacher_recon_anomaly.item(),
            'student_recon_normal': student_recon_normal_metric.item(),
            'student_recon_anomaly': student_recon_anomaly_metric.item(),
            # Feature-level stats: (num_features,) numpy arrays or None
            'feature_recon_mean': feature_recon_mean.cpu().numpy(),
            'feature_recon_max': feature_recon_max.cpu().numpy(),
            'feature_disc_mean': feature_disc_mean.cpu().numpy() if feature_disc_mean is not None else None,
            'feature_disc_max': feature_disc_max.cpu().numpy() if feature_disc_max is not None else None,
            'fm_loss': fm_loss.item() if isinstance(fm_loss, torch.Tensor) else fm_loss,
        }

        # Tensors with gradients retained for adversarial training
        loss_tensors = {
            'normal_loss': normal_loss,
            'anomaly_loss': anomaly_loss,
            'anomaly_disc_forward': anomaly_disc_forward,
            'reconstruction_loss': reconstruction_loss,
            'fm_loss': fm_loss,
        }
        # Patch-level masks for WDGRL critic (no grad needed)
        if self.patch_level_loss:
            _pll = locals()
            if 'patch_has_masked' in _pll:
                loss_tensors['patch_has_masked'] = _pll['patch_has_masked']
            if 'patch_has_anomaly' in _pll:
                loss_tensors['patch_has_anomaly'] = _pll['patch_has_anomaly']

        # GRL metrics (from patch_level_loss block)
        _grl = locals().get('_grl_results')
        if _grl is not None:
            loss_dict['grl_cls_loss'] = _grl['grl_cls_loss']
            loss_dict['grl_balanced_acc'] = _grl['grl_balanced_acc']
            loss_dict['grl_anomaly_acc'] = _grl['grl_anomaly_acc']
            loss_dict['grl_normal_acc'] = _grl['grl_normal_acc']
            if _grl['grl_cls_loss_tensor'] is not None:
                loss_tensors['grl_cls_loss'] = _grl['grl_cls_loss_tensor']

        return total_loss, loss_dict, loss_tensors


# ============================================================
# Adversarial Discriminator Loss Functions
# ============================================================

def compute_discriminator_loss(
    discriminator: nn.Module,
    real_patches: torch.Tensor,
    fake_patches: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    """Compute discriminator loss on ALL patches (normal + anomaly).

    D distinguishes real (original data) vs fake (student output).
    This is NOT normal vs anomaly classification.

    Args:
        discriminator: PatchDiscriminator module
        real_patches: (N, num_features, patch_size) original data patches
        fake_patches: (N, num_features, patch_size) student output patches (detached)

    Returns:
        d_loss: scalar loss tensor
        d_real_acc: accuracy on real patches (fraction classified as real)
        d_fake_acc: accuracy on fake patches (fraction classified as fake)
    """
    real_logits = discriminator(real_patches)
    fake_logits = discriminator(fake_patches.detach())

    d_loss_real = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)
    )
    d_loss_fake = F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )
    d_loss = d_loss_real + d_loss_fake

    with torch.no_grad():
        d_real_acc = (real_logits > 0).float().mean().item()
        d_fake_acc = (fake_logits < 0).float().mean().item()

    return d_loss, d_real_acc, d_fake_acc


def compute_student_adversarial_loss(
    discriminator: nn.Module,
    fake_patches: torch.Tensor,
) -> torch.Tensor:
    """Adversarial loss for student: fool D by making fake patches look real.

    BCE(D(fake), 1) — gradient flows through student decoder.
    Applied only to anomaly patches in the trainer.

    Args:
        discriminator: PatchDiscriminator module
        fake_patches: (N, num_features, patch_size) student output patches (with grad)

    Returns:
        adv_loss: scalar loss tensor
    """
    fake_logits = discriminator(fake_patches)
    adv_loss = F.binary_cross_entropy_with_logits(
        fake_logits, torch.ones_like(fake_logits)
    )
    return adv_loss


def compute_adaptive_lambda(
    last_weight: torch.Tensor,
    normal_loss: torch.Tensor,
    anomaly_disc_forward: torch.Tensor,
    adv_loss: torch.Tensor,
    delta: float = 1e-4,
) -> torch.Tensor:
    """VQGAN-style adaptive λ via gradient magnitude balancing.

    λ = (||∇_w normal_loss|| + ||∇_w anomaly_disc_forward||) / (||∇_w adv_loss|| + δ)

    Uses sum of individual gradient norms (not norm of sum) to prevent
    partial gradient cancellation between normal and anomaly directions.

    Args:
        last_weight: Parameter of last student decoder layer (gradient anchor)
        normal_loss: Normal discrepancy loss tensor (minimize direction)
        anomaly_disc_forward: Anomaly discrepancy in forward direction (no reversal)
        adv_loss: Student adversarial loss tensor
        delta: Stability epsilon

    Returns:
        lambda_val: Adaptive scaling factor (detached, clamped to [0, 10])
    """
    normal_grads = torch.autograd.grad(
        normal_loss, last_weight,
        retain_graph=True, allow_unused=True
    )[0]
    anomaly_grads = torch.autograd.grad(
        anomaly_disc_forward, last_weight,
        retain_graph=True, allow_unused=True
    )[0]
    adv_grads = torch.autograd.grad(
        adv_loss, last_weight,
        retain_graph=True, allow_unused=True
    )[0]

    if adv_grads is None:
        return torch.tensor(1.0, device=adv_loss.device)

    normal_norm = normal_grads.norm() if normal_grads is not None else 0.0
    anomaly_norm = anomaly_grads.norm() if anomaly_grads is not None else 0.0

    lambda_val = (normal_norm + anomaly_norm) / (adv_grads.norm() + delta)
    lambda_val = torch.clamp(lambda_val, 0.0, 10.0)
    return lambda_val.detach()


def compute_wdgrl_gradient_penalty(
    critic: nn.Module,
    normal_features: torch.Tensor,
    anomaly_features: torch.Tensor,
) -> torch.Tensor:
    """Gradient penalty for WDGRL (1-Lipschitz constraint on critic).

    Penalizes ||∇_x critic(x̂)||₂ - 1)² on interpolated points between
    normal and anomaly features (WGAN-GP style, Gulrajani et al. 2017).

    Args:
        critic: WassersteinCritic module
        normal_features: (N, d_model) features from normal patches
        anomaly_features: (M, d_model) features from anomaly patches

    Returns:
        gradient_penalty: scalar tensor
    """
    # Match sizes: sample min(N, M) from the larger set
    n = min(normal_features.size(0), anomaly_features.size(0))
    if n == 0:
        return torch.tensor(0.0, device=normal_features.device)
    f_n = normal_features[:n]
    f_a = anomaly_features[:n]

    # Interpolate between normal and anomaly
    alpha = torch.rand(n, 1, device=f_n.device)
    interpolated = (alpha * f_n + (1 - alpha) * f_a).requires_grad_(True)

    # Critic scores on interpolated points
    scores = critic(interpolated)

    # Compute gradients w.r.t. interpolated input
    gradients = torch.autograd.grad(
        outputs=scores, inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0]

    # Gradient penalty: (||grad||₂ - 1)²
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1.0) ** 2).mean()
    return penalty
