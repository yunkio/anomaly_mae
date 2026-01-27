"""
Loss functions for Self-Distilled MAE Anomaly Detection

Supports:
- Patch-level and window-level discrepancy loss
- Multiple margin types: hinge, softplus, dynamic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SelfDistillationLoss(nn.Module):
    """Loss function for self-distilled MAE with configurable discrepancy loss granularity

    Supports two modes:
    - patch_level_loss=True: Compute loss at patch level (each patch has its own gradient direction)
    - patch_level_loss=False: Compute loss at window level (sample-level, original behavior)

    Supports three margin types:
    - 'hinge': relu(margin - discrepancy) - original, hard cutoff at margin
    - 'softplus': log(1 + exp(margin - discrepancy)) - soft version, always has gradient
    - 'dynamic': margin = mu + k*sigma based on normal patches/samples in batch
    """

    def __init__(self, config):
        super().__init__()
        self.margin = config.margin
        self.margin_type = getattr(config, 'margin_type', 'hinge')
        self.dynamic_margin_k = getattr(config, 'dynamic_margin_k', 3.0)
        self.use_discrepancy = getattr(config, 'use_discrepancy_loss', True)
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.patch_level_loss = getattr(config, 'patch_level_loss', True)

        # Anomaly loss weight
        self.anomaly_loss_weight = getattr(config, 'anomaly_loss_weight', 1.0)

    def _compute_anomaly_loss(
        self,
        discrepancy: torch.Tensor,
        anomaly_mask: torch.Tensor,
        normal_mask: torch.Tensor,
        margin: float
    ) -> torch.Tensor:
        """Compute anomaly loss based on margin_type (for window-level mode)"""
        if self.margin_type == 'hinge':
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
        if self.margin_type == 'hinge':
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
        teacher_only: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        teacher_recon_per_sample = teacher_recon_full.sum(dim=(1, 2)) / ((1 - mask_expanded).sum(dim=(1, 2)) + 1e-4)
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
        student_recon_per_sample = student_recon_full.sum(dim=(1, 2)) / ((1 - mask_expanded).sum(dim=(1, 2)) + 1e-4)
        student_recon_normal_metric = (is_normal_sample * student_recon_per_sample).sum() / (is_normal_sample.sum() + 1e-4)
        student_recon_anomaly_metric = (has_anomaly_sample * student_recon_per_sample).sum() / (has_anomaly_sample.sum() + 1e-4)

        if self.use_discrepancy and not teacher_only:
            # Compute per-position discrepancy: (batch, seq_length, num_features)
            discrepancy_full = (teacher_output.detach() - student_output) ** 2

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
                normal_loss = (normal_patch_mask * patch_discrepancy).sum() / (normal_patch_mask.sum() + 1e-4)

                # Anomaly patch loss (with weight multiplier)
                anomaly_patch_mask = patch_has_anomaly * patch_has_masked
                per_patch_anomaly_loss = self._compute_patch_anomaly_loss(
                    patch_discrepancy, anomaly_patch_mask, normal_patch_mask, self.margin
                )
                anomaly_loss = (anomaly_patch_mask * per_patch_anomaly_loss).sum() / (anomaly_patch_mask.sum() + 1e-4)
                anomaly_loss = warmup_factor * anomaly_loss * self.anomaly_loss_weight

                sample_discrepancy = (patch_discrepancy * patch_has_masked).sum(dim=1) / (patch_has_masked.sum(dim=1) + 1e-4)

            else:
                # ============== WINDOW-LEVEL LOSS ==============
                # Original behavior: sample-level classification

                discrepancy_masked = discrepancy_full * (1 - mask_expanded)
                sample_discrepancy = discrepancy_masked.sum(dim=(1, 2)) / ((1 - mask_expanded).sum(dim=(1, 2)) + 1e-4)

                # Sample has anomaly if ANY masked position has anomaly
                masked_point_labels = point_labels * (1 - mask)
                has_anomaly_in_masked = (masked_point_labels.sum(dim=1) > 0).float()

                normal_mask = (1 - has_anomaly_in_masked)
                anomaly_mask = has_anomaly_in_masked

                # Normal loss: minimize discrepancy for normal samples
                normal_loss = (normal_mask * sample_discrepancy).sum() / (normal_mask.sum() + 1e-4)

                # Anomaly loss: encourage discrepancy to be large (with weight multiplier)
                per_sample_anomaly_loss = self._compute_anomaly_loss(
                    sample_discrepancy, anomaly_mask, normal_mask, self.margin
                )
                anomaly_loss = (anomaly_mask * per_sample_anomaly_loss).sum() / (anomaly_mask.sum() + 1e-4)
                anomaly_loss = warmup_factor * anomaly_loss * self.anomaly_loss_weight

            discrepancy_loss = normal_loss + anomaly_loss
            total_loss = reconstruction_loss + discrepancy_loss
        else:
            discrepancy_loss = torch.tensor(0.0, device=teacher_output.device)
            normal_loss = torch.tensor(0.0, device=teacher_output.device)
            anomaly_loss = torch.tensor(0.0, device=teacher_output.device)
            sample_discrepancy = torch.zeros(batch_size, device=teacher_output.device)
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
        }

        return total_loss, loss_dict
