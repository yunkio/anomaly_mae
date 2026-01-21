"""
Self-Distilled MAE Model for Multivariate Time Series Anomaly Detection

Supports three patchify modes:
- 'cnn_first': CNN on full sequence, then patchify (information leakage across patches)
- 'patch_cnn': Patchify first, then CNN per patch (no cross-patch leakage)
- 'linear': Patchify then linear embedding (MAE original style, no CNN)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .config import Config


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

    Supports three patchify modes:
    - 'cnn_first': CNN on full sequence, then patchify (information leakage across patches)
    - 'patch_cnn': Patchify first, then CNN per patch (no cross-patch leakage)
    - 'linear': Patchify then linear embedding (MAE original style, no CNN)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patchify_mode = getattr(config, 'patchify_mode', 'linear')

        # Patch configuration (always defined for both strategies)
        self.num_patches = config.seq_length // config.patch_size
        self.patch_size = config.patch_size

        # Both strategies use patch-based processing
        self.use_patch = True
        self.effective_seq_len = self.num_patches

        # Build embedding layers based on patchify_mode
        self._build_embedding_layers(config)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        # Teacher decoder
        self.teacher_decoder = None
        if config.use_teacher:
            teacher_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=False
            )
            self.teacher_decoder = nn.TransformerDecoder(
                teacher_decoder_layer,
                num_layers=config.num_teacher_decoder_layers
            )

        # Student decoder
        self.student_decoder = None
        if config.use_student:
            student_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=False
            )
            self.student_decoder = nn.TransformerDecoder(
                student_decoder_layer,
                num_layers=config.num_student_decoder_layers
            )

        # Output projections
        if self.use_patch:
            output_dim = config.patch_size * config.num_features
        else:
            output_dim = config.num_features

        if config.use_teacher:
            self.teacher_output_projection = nn.Linear(config.d_model, output_dim)
        if config.use_student:
            self.student_output_projection = nn.Linear(config.d_model, output_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_model))

    def _build_embedding_layers(self, config: Config):
        """Build embedding layers based on patchify_mode"""

        if self.patchify_mode == 'cnn_first':
            # CNN on full sequence, then patchify
            # Input: (batch, num_features, seq_length)
            # Output: (batch, d_model, seq_length)
            self.cnn_layers = nn.Sequential(
                nn.Conv1d(config.num_features, config.d_model // 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(config.d_model // 2),
                nn.ReLU(),
                nn.Conv1d(config.d_model // 2, config.d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(config.d_model),
                nn.ReLU()
            )
            if self.use_patch:
                # After CNN: (batch, d_model, seq) -> (batch, num_patches, d_model * patch_size)
                self.patch_embed = nn.Linear(config.d_model * config.patch_size, config.d_model)

        elif self.patchify_mode == 'patch_cnn':
            # Patchify first, then CNN per patch
            # Each patch: (batch * num_patches, num_features, patch_size)
            # CNN processes each patch independently
            self.patch_cnn = nn.Sequential(
                nn.Conv1d(config.num_features, config.d_model // 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(config.d_model // 2),
                nn.ReLU(),
                nn.Conv1d(config.d_model // 2, config.d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(config.d_model),
                nn.ReLU()
            )
            # Pool/flatten CNN output to get d_model embedding
            # CNN output: (batch * num_patches, d_model, patch_size)
            # After mean pooling: (batch * num_patches, d_model)

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

        if self.patchify_mode == 'cnn_first':
            # CNN on full sequence first
            # (batch, seq, features) -> (batch, features, seq)
            x_cnn_input = x.transpose(1, 2)
            # (batch, d_model, seq)
            x_cnn = self.cnn_layers(x_cnn_input)

            if self.use_patch:
                # Patchify CNN features: (batch, d_model, seq) -> (batch, num_patches, d_model*patch_size)
                x_patches = self._patchify_cnn_output(x_cnn)
                # Embed: (batch, num_patches, d_model*patch_size) -> (batch, num_patches, d_model)
                x_embed = self.patch_embed(x_patches)
                x_embed = x_embed.transpose(0, 1)  # (num_patches, batch, d_model)
            else:
                # Token-based: (batch, d_model, seq) -> (seq, batch, d_model)
                x_embed = x_cnn.transpose(1, 2).transpose(0, 1)

        elif self.patchify_mode == 'patch_cnn':
            # Patchify first, then CNN per patch
            if self.use_patch:
                # (batch, seq, features) -> (batch, num_patches, patch_size, features)
                x_patches = x.reshape(batch_size, self.num_patches, self.patch_size, num_features)
                # (batch, num_patches, patch_size, features) -> (batch * num_patches, features, patch_size)
                x_patches = x_patches.reshape(batch_size * self.num_patches, self.patch_size, num_features)
                x_patches = x_patches.transpose(1, 2)  # (batch * num_patches, features, patch_size)

                # Apply CNN per patch
                x_cnn = self.patch_cnn(x_patches)  # (batch * num_patches, d_model, patch_size)

                # Mean pooling over patch_size dimension
                x_embed = x_cnn.mean(dim=2)  # (batch * num_patches, d_model)

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

    def _patchify_cnn_output(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify CNN output (for cnn_first mode)
        Args:
            x: (batch, d_model, seq_length)
        Returns:
            patches: (batch, num_patches, d_model * patch_size)
        """
        batch_size, d_model, seq_length = x.shape
        # Reshape: (batch, d_model, num_patches, patch_size)
        x = x.reshape(batch_size, d_model, self.num_patches, self.patch_size)
        # Permute: (batch, num_patches, d_model, patch_size)
        x = x.permute(0, 2, 1, 3)
        # Flatten: (batch, num_patches, d_model * patch_size)
        x = x.reshape(batch_size, self.num_patches, d_model * self.patch_size)
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

    def random_masking(self, x: torch.Tensor, masking_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random masking for patches/tokens (patch strategy)

        Args:
            x: (seq_len, batch_size, d_model)
            masking_ratio: ratio of patches to mask
        Returns:
            x_masked: masked input
            mask: binary mask (1=keep, 0=masked) - (seq_len, batch_size)
        """
        if not self.config.use_masking or masking_ratio == 0:
            return x, torch.ones(x.size(0), x.size(1), device=x.device)

        seq_len, batch_size, d_model = x.shape
        num_keep = round(seq_len * (1 - masking_ratio))

        # Random selection of patches to keep
        noise = torch.rand(seq_len, batch_size, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        # Create binary mask
        mask = torch.zeros(seq_len, batch_size, device=x.device)
        mask[:num_keep, :] = 1
        mask = torch.gather(mask, dim=0, index=ids_restore)

        # Apply mask
        mask_tokens = self.mask_token.repeat(seq_len, batch_size, 1)
        x_masked = x * mask.unsqueeze(-1) + mask_tokens * (1 - mask.unsqueeze(-1))

        return x_masked, mask

    def feature_wise_masking(
        self,
        x: torch.Tensor,
        x_raw: torch.Tensor,
        masking_ratio: float,
        point_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Feature-wise masking: mask each feature independently at different PATCHES

        Each feature is masked at different patch positions independently.
        Example with 5 patches:
            F0: [██, ░░, ██, ░░, ██]  (patches 1,3 masked)
            F1: [░░, ██, ░░, ██, ██]  (patches 0,2 masked)
            F2: [██, ██, ░░, ░░, ██]  (patches 2,3 masked)

        Args:
            x: (seq_len, batch_size, d_model) - embedded input (seq_len = num_patches)
            x_raw: (batch_size, seq_length, num_features) - raw input
            masking_ratio: ratio of patches to mask per feature
            point_labels: (batch_size, seq_length) for force_mask_anomaly
        Returns:
            x_masked: masked input (seq_len, batch_size, d_model)
            mask: binary mask (batch_size, num_patches, num_features) - 1=keep, 0=masked
        """
        batch_size, seq_length, num_features = x_raw.shape
        seq_len = x.size(0)  # num_patches if using patch mode

        if not self.config.use_masking or masking_ratio == 0:
            return x, torch.ones(batch_size, self.num_patches, num_features, device=x.device)

        num_mask_per_feature = round(self.num_patches * masking_ratio)

        # Vectorized feature-wise masking
        # Generate random indices for all batch x feature combinations at once
        # Shape: (batch_size, num_features, num_patches)
        rand_vals = torch.rand(batch_size, num_features, self.num_patches, device=x.device)

        # Force mask anomaly patches if enabled
        if self.training and self.config.force_mask_anomaly and point_labels is not None:
            # point_labels: (batch, seq_length) -> (batch, num_patches)
            patch_labels = point_labels.reshape(batch_size, self.num_patches, self.patch_size)
            anomaly_patches = (patch_labels.sum(dim=2) > 0).float()  # (batch, num_patches)
            # Give anomaly patches very low random values so they get selected first
            # Expand to (batch, num_features, num_patches)
            anomaly_boost = anomaly_patches.unsqueeze(1).expand(-1, num_features, -1)
            rand_vals = rand_vals - anomaly_boost * 10  # Anomaly patches get negative values

        # Get indices that would sort rand_vals (ascending)
        # Smallest values (including anomaly-boosted) come first
        sorted_indices = rand_vals.argsort(dim=2)  # (batch, num_features, num_patches)

        # Select first num_mask_per_feature indices for masking
        mask_indices = sorted_indices[:, :, :num_mask_per_feature]  # (batch, num_features, num_mask)

        # Create mask using scatter
        mask = torch.ones(batch_size, num_features, self.num_patches, device=x.device)
        mask.scatter_(2, mask_indices, 0)

        # Transpose to (batch, num_patches, num_features)
        mask = mask.transpose(1, 2)

        # For embedded input, we need to determine which patches to mask
        # A patch in embedded space is masked if ANY feature at that patch is masked
        patch_mask = mask.min(dim=2)[0]  # (batch, num_patches)
        patch_mask = patch_mask.transpose(0, 1)  # (num_patches, batch)

        # Apply patch-level masking to embedded input
        mask_tokens = self.mask_token.repeat(seq_len, batch_size, 1)
        x_masked = x * patch_mask.unsqueeze(-1) + mask_tokens * (1 - patch_mask.unsqueeze(-1))

        return x_masked, mask

    def forward(
        self,
        x: torch.Tensor,
        masking_ratio: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_length, num_features)
            masking_ratio: ratio of patches to mask
            mask: optional pre-defined mask
            point_labels: (batch_size, seq_length) for force_mask_anomaly
        Returns:
            teacher_output, student_output, mask
        """
        if masking_ratio is None:
            masking_ratio = self.config.masking_ratio

        batch_size, seq_length, num_features = x.shape

        # Embed input based on patchify_mode
        x_embed = self._embed_input(x)  # (seq_len, batch, d_model)

        # Masking based on strategy
        mask_provided_externally = mask is not None
        if mask is None:
            if self.config.masking_strategy == 'feature_wise':
                # feature_wise handles force_mask_anomaly internally
                x_masked, mask = self.feature_wise_masking(x_embed, x, masking_ratio, point_labels)
                # mask is (batch, num_patches, num_features) for feature_wise
            else:  # 'patch' strategy (default)
                x_masked, mask = self.random_masking(x_embed, masking_ratio)
                # mask is (seq_len, batch) for patch strategy
        else:
            # Handle pre-defined mask (from external source like evaluator)
            # External mask is always (batch, seq_length) format
            if self.use_patch and mask.shape[1] != self.num_patches:
                mask_reshaped = mask.reshape(batch_size, self.num_patches, self.patch_size)
                mask = mask_reshaped[:, :, -1]

            mask = mask.transpose(0, 1)  # (seq_len, batch)
            mask_tokens = self.mask_token.repeat(x_embed.size(0), x_embed.size(1), 1)
            x_masked = x_embed * mask.unsqueeze(-1) + mask_tokens * (1 - mask.unsqueeze(-1))

        # Force mask patches containing anomalies during training (patch strategy only)
        # Note: feature_wise strategy handles force_mask_anomaly in feature_wise_masking()
        if (self.training and self.config.force_mask_anomaly and
            point_labels is not None and self.config.masking_strategy == 'patch'):
            seq_len = mask.shape[0]

            if self.use_patch:
                patch_labels = point_labels.reshape(batch_size, self.num_patches, self.patch_size)
                anomaly_patches = (patch_labels.sum(dim=2) > 0).float()
                anomaly_patches = anomaly_patches.transpose(0, 1)
            else:
                anomaly_patches = point_labels.transpose(0, 1).float()

            # Force anomaly patches to be masked
            mask = mask * (1 - anomaly_patches)

            # Add random masking if needed to maintain masking ratio
            target_num_masked = round(seq_len * masking_ratio)
            current_num_masked = (1 - mask).sum(dim=0)

            for b in range(batch_size):
                additional_needed = int(target_num_masked - current_num_masked[b].item())
                if additional_needed > 0:
                    available_mask = (mask[:, b] == 1) & (anomaly_patches[:, b] == 0)
                    available_indices = torch.where(available_mask)[0]

                    if len(available_indices) > 0:
                        num_to_mask = min(additional_needed, len(available_indices))
                        perm = torch.randperm(len(available_indices), device=mask.device)[:num_to_mask]
                        indices_to_mask = available_indices[perm]
                        mask[indices_to_mask, b] = 0

            # Re-apply masking
            mask_tokens = self.mask_token.repeat(x_embed.size(0), x_embed.size(1), 1)
            x_masked = x_embed * mask.unsqueeze(-1) + mask_tokens * (1 - mask.unsqueeze(-1))

        # Positional encoding
        x_masked = self.pos_encoder(x_masked)

        # Encode
        latent = self.encoder(x_masked)

        # Decode
        tgt = self.pos_encoder(torch.zeros_like(x_embed))

        teacher_output = None
        student_output = None

        if self.config.use_teacher and self.teacher_decoder is not None:
            teacher_hidden = self.teacher_decoder(tgt, latent)
            teacher_output = self.teacher_output_projection(teacher_hidden)
            teacher_output = teacher_output.transpose(0, 1)

            if self.use_patch:
                teacher_output = self.unpatchify(teacher_output)

        if self.config.use_student and self.student_decoder is not None:
            student_hidden = self.student_decoder(tgt, latent)
            student_output = self.student_output_projection(student_hidden)
            student_output = student_output.transpose(0, 1)

            if self.use_patch:
                student_output = self.unpatchify(student_output)

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
        elif self.config.masking_strategy == 'feature_wise':
            # mask is (batch, num_patches, num_features)
            # Convert to (batch, seq_length) by:
            # 1. Take min across features (position masked if ANY feature masked)
            # 2. Expand patches to time steps
            patch_mask = mask.min(dim=2)[0]  # (batch, num_patches)
            if self.use_patch:
                patch_mask = patch_mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
                mask = patch_mask.reshape(batch_size, seq_length)
            else:
                mask = patch_mask
        else:
            # patch strategy: mask is (seq_len, batch)
            mask = mask.transpose(0, 1)  # (batch, seq_len)

            if self.use_patch:
                mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
                mask = mask.reshape(batch_size, seq_length)

        return teacher_output, student_output, mask
