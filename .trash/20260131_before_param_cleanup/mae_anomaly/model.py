"""
Self-Distilled MAE Model for Multivariate Time Series Anomaly Detection

Supports two patchify modes:
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

    Supports two patchify modes:
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

        # Shared decoder (optional, trained with teacher)
        self.num_shared_decoder_layers = getattr(config, 'num_shared_decoder_layers', 0)
        self.shared_decoder = None
        if self.num_shared_decoder_layers > 0:
            shared_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=False
            )
            self.shared_decoder = nn.TransformerDecoder(
                shared_decoder_layer,
                num_layers=self.num_shared_decoder_layers
            )

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

        # Mask token(s) - shared or separate for teacher/student
        self.shared_mask_token = getattr(config, 'shared_mask_token', True)
        self.mask_after_encoder = getattr(config, 'mask_after_encoder', False)

        if self.shared_mask_token:
            self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        else:
            # Separate mask tokens for teacher and student
            if config.use_teacher:
                self.teacher_mask_token = nn.Parameter(torch.randn(1, 1, config.d_model))
            if config.use_student:
                self.student_mask_token = nn.Parameter(torch.randn(1, 1, config.d_model))

    def _build_embedding_layers(self, config: Config):
        """Build embedding layers based on patchify_mode"""

        if self.patchify_mode == 'patch_cnn':
            # Patchify first, then CNN per patch
            # Each patch: (batch * num_patches, num_features, patch_size)
            # CNN processes each patch independently

            # Get CNN channels from config (default: d_model//2, d_model)
            cnn_channels = getattr(config, 'cnn_channels', None)
            if cnn_channels is None:
                mid_channels = config.d_model // 2
                out_channels = config.d_model
            else:
                mid_channels, out_channels = cnn_channels

            self.patch_cnn = nn.Sequential(
                nn.Conv1d(config.num_features, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(),
                nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )

            # If out_channels != d_model, need projection
            if out_channels != config.d_model:
                self.cnn_projection = nn.Linear(out_channels, config.d_model)
            else:
                self.cnn_projection = None
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

                # Mean pooling over patch_size dimension
                x_embed = x_cnn.mean(dim=2)  # (batch * num_patches, out_channels)

                # Apply projection if needed (when out_channels != d_model)
                if self.cnn_projection is not None:
                    x_embed = self.cnn_projection(x_embed)  # (batch * num_patches, d_model)

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

        # Apply mask (use default mask token - teacher for consistency)
        mask_token = self._get_mask_token('teacher')
        mask_tokens = mask_token.repeat(seq_len, batch_size, 1)
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

        # Apply patch-level masking to embedded input (use default mask token)
        mask_token = self._get_mask_token('teacher')
        mask_tokens = mask_token.repeat(seq_len, batch_size, 1)
        x_masked = x * patch_mask.unsqueeze(-1) + mask_tokens * (1 - patch_mask.unsqueeze(-1))

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
        num_keep = int(mask.sum(dim=0)[0].item())  # Assumes uniform masking across batch

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

        # Encode only visible patches
        latent = self.encoder(x_visible_with_pos)  # (num_keep, batch, d_model)

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
        num_masked = seq_len - num_visible

        # Create mask tokens for masked positions
        mask_tokens = mask_token.expand(num_masked, batch_size, -1)  # (num_masked, batch, d_model)

        # Concatenate visible patches and mask tokens
        latent_with_masks = torch.cat([latent, mask_tokens], dim=0)  # (seq_len, batch, d_model)

        # Unshuffle to restore original order
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, d_model)
        latent_full = torch.gather(latent_with_masks, dim=0, index=ids_restore_expanded)

        return latent_full

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
        seq_len = x_embed.size(0)

        # Masking based on strategy
        mask_provided_externally = mask is not None
        is_feature_wise = self.config.masking_strategy == 'feature_wise'

        # Get default mask token for pre-masking (used when mask_after_encoder=False)
        default_mask_token = self._get_mask_token('teacher')

        if mask is None:
            if is_feature_wise:
                # feature_wise handles force_mask_anomaly internally
                x_masked, mask = self.feature_wise_masking(x_embed, x, masking_ratio, point_labels)
                # mask is (batch, num_patches, num_features) for feature_wise
                # Convert to patch-level mask for encoding
                patch_mask = mask.min(dim=2)[0].transpose(0, 1)  # (seq_len, batch)
            else:  # 'patch' strategy (default)
                x_masked, mask = self.random_masking(x_embed, masking_ratio)
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
            mask_tokens = default_mask_token.repeat(x_embed.size(0), x_embed.size(1), 1)
            x_masked = x_embed * mask.unsqueeze(-1) + mask_tokens * (1 - mask.unsqueeze(-1))

        # Force mask patches containing anomalies during training (patch strategy only)
        # Note: feature_wise strategy handles force_mask_anomaly in feature_wise_masking()
        if (self.training and self.config.force_mask_anomaly and
            point_labels is not None and self.config.masking_strategy == 'patch'):
            current_seq_len = patch_mask.shape[0]

            if self.use_patch:
                patch_labels = point_labels.reshape(batch_size, self.num_patches, self.patch_size)
                anomaly_patches = (patch_labels.sum(dim=2) > 0).float()
                anomaly_patches = anomaly_patches.transpose(0, 1)
            else:
                anomaly_patches = point_labels.transpose(0, 1).float()

            # Force anomaly patches to be masked
            patch_mask = patch_mask * (1 - anomaly_patches)
            mask = patch_mask  # Update mask for patch strategy

            # Add random masking if needed to maintain masking ratio
            target_num_masked = round(current_seq_len * masking_ratio)
            current_num_masked = (1 - patch_mask).sum(dim=0)

            for b in range(batch_size):
                additional_needed = int(target_num_masked - current_num_masked[b].item())
                if additional_needed > 0:
                    available_mask = (patch_mask[:, b] == 1) & (anomaly_patches[:, b] == 0)
                    available_indices = torch.where(available_mask)[0]

                    if len(available_indices) > 0:
                        num_to_mask = min(additional_needed, len(available_indices))
                        perm = torch.randperm(len(available_indices), device=patch_mask.device)[:num_to_mask]
                        indices_to_mask = available_indices[perm]
                        patch_mask[indices_to_mask, b] = 0
                        mask = patch_mask

            # Re-apply masking for non-mask_after_encoder mode
            if not self.mask_after_encoder:
                mask_tokens = default_mask_token.repeat(x_embed.size(0), x_embed.size(1), 1)
                x_masked = x_embed * patch_mask.unsqueeze(-1) + mask_tokens * (1 - patch_mask.unsqueeze(-1))

        # === Encoding ===
        if self.mask_after_encoder:
            # Standard MAE: encode visible patches only, insert mask tokens before decoder
            latent_visible, ids_restore = self._encode_visible_only(x_embed, patch_mask)
        else:
            # Current behavior: mask tokens go through encoder
            x_masked = self.pos_encoder(x_masked)
            latent = self.encoder(x_masked)

        # === Decoding ===
        tgt = self.pos_encoder(torch.zeros_like(x_embed))

        teacher_output = None
        student_output = None

        if self.config.use_teacher and self.teacher_decoder is not None:
            if self.mask_after_encoder:
                # Insert mask tokens before teacher decoder
                teacher_mask_token = self._get_mask_token('teacher')
                teacher_latent = self._insert_mask_tokens_and_unshuffle(
                    latent_visible, ids_restore, seq_len, teacher_mask_token
                )
            else:
                teacher_latent = latent

            # Apply shared decoder first if exists (trained with teacher)
            if self.shared_decoder is not None:
                teacher_latent = self.shared_decoder(tgt, teacher_latent)

            teacher_hidden = self.teacher_decoder(tgt, teacher_latent)
            teacher_output = self.teacher_output_projection(teacher_hidden)
            teacher_output = teacher_output.transpose(0, 1)

            if self.use_patch:
                teacher_output = self.unpatchify(teacher_output)

        if self.config.use_student and self.student_decoder is not None:
            if self.mask_after_encoder:
                # Insert mask tokens before student decoder
                # Detach encoder output to prevent encoder updates from student loss
                student_mask_token = self._get_mask_token('student')
                student_latent = self._insert_mask_tokens_and_unshuffle(
                    latent_visible.detach(), ids_restore, seq_len, student_mask_token
                )
            else:
                # Detach encoder output to prevent encoder updates from student loss
                student_latent = latent.detach()

            # Student uses its own decoder directly (no shared decoder)
            student_hidden = self.student_decoder(tgt, student_latent)
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
        elif is_feature_wise:
            # mask is (batch, num_patches, num_features)
            # Convert to (batch, seq_length) by:
            # 1. Take min across features (position masked if ANY feature masked)
            # 2. Expand patches to time steps
            patch_mask_out = mask.min(dim=2)[0]  # (batch, num_patches)
            if self.use_patch:
                patch_mask_out = patch_mask_out.unsqueeze(-1).repeat(1, 1, self.patch_size)
                mask = patch_mask_out.reshape(batch_size, seq_length)
            else:
                mask = patch_mask_out
        else:
            # patch strategy: mask is (seq_len, batch)
            mask = mask.transpose(0, 1)  # (batch, seq_len)

            if self.use_patch:
                mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
                mask = mask.reshape(batch_size, seq_length)

        return teacher_output, student_output, mask
