"""
Self-Distilled MAE Model for Multivariate Time Series Anomaly Detection
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
    """Self-Distilled MAE for Multivariate Time Series"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Determine if using patch-based or token-based
        self.use_patch = (config.masking_strategy == 'patch')

        if self.use_patch:
            # Patch embedding: (patch_size * num_features) -> d_model
            self.num_patches = config.seq_length // config.patch_size
            self.patch_size = config.patch_size
            self.patch_embed = nn.Linear(config.patch_size * config.num_features, config.d_model)
            self.effective_seq_len = self.num_patches
        else:
            # Token-based: num_features -> d_model
            self.input_projection = nn.Linear(config.num_features, config.d_model)
            self.effective_seq_len = config.seq_length

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

        # Output projection
        if self.use_patch:
            # Patch reconstruction: d_model -> (patch_size * num_features)
            self.output_projection = nn.Linear(config.d_model, config.patch_size * config.num_features)
        else:
            # Token reconstruction: d_model -> num_features
            self.output_projection = nn.Linear(config.d_model, config.num_features)

        # Mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.d_model))

    def random_masking(self, x: torch.Tensor, masking_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random masking with support for patch, token, temporal, and feature-wise strategies

        Args:
            x: (seq_len, batch_size, d_model)
            masking_ratio: ratio of tokens/patches to mask

        Returns:
            x_masked: masked input
            mask: binary mask (1=keep, 0=masked) - (seq_len, batch_size)
        """
        if not self.config.use_masking or masking_ratio == 0:
            # No masking
            return x, torch.ones(x.size(0), x.size(1), device=x.device)

        seq_len, batch_size, d_model = x.shape

        if self.config.masking_strategy == 'patch':
            # Patch-based masking: mask entire patches (contiguous blocks)
            # seq_len should be num_patches here
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

        elif self.config.masking_strategy == 'token':
            # Token-level masking (BERT style): randomly mask individual tokens
            # Each position in the sequence is masked independently
            num_elements = seq_len * batch_size
            num_keep = int(num_elements * (1 - masking_ratio))

            # Create 2D noise for all positions
            noise = torch.rand(seq_len, batch_size, device=x.device)

            # Flatten, sort, and create mask
            noise_flat = noise.flatten()
            ids_shuffle_flat = torch.argsort(noise_flat)
            ids_restore_flat = torch.argsort(ids_shuffle_flat)

            mask_flat = torch.zeros(num_elements, device=x.device)
            mask_flat[:num_keep] = 1
            mask_flat = torch.gather(mask_flat, dim=0, index=ids_restore_flat)

            # Reshape back to 2D
            mask = mask_flat.reshape(seq_len, batch_size)

            # Apply mask
            mask_tokens = self.mask_token.repeat(seq_len, batch_size, 1)
            x_masked = x * mask.unsqueeze(-1) + mask_tokens * (1 - mask.unsqueeze(-1))

            return x_masked, mask

        elif self.config.masking_strategy == 'temporal':
            # Temporal masking: mask all features at same time steps
            num_keep = int(seq_len * (1 - masking_ratio))

            noise = torch.rand(seq_len, batch_size, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=0)
            ids_restore = torch.argsort(ids_shuffle, dim=0)

            mask = torch.zeros(seq_len, batch_size, device=x.device)
            mask[:num_keep, :] = 1
            mask = torch.gather(mask, dim=0, index=ids_restore)

            mask_tokens = self.mask_token.repeat(seq_len, batch_size, 1)
            x_masked = x * mask.unsqueeze(-1) + mask_tokens * (1 - mask.unsqueeze(-1))

            return x_masked, mask

        elif self.config.masking_strategy == 'feature_wise':
            # Feature-wise masking: mask each time-feature pair independently
            # This is more flexible and doesn't require d_model to be divisible by num_features

            len_keep = int(seq_len * (1 - masking_ratio))

            # Create a 3D mask: (seq_len, batch_size, d_model)
            # But apply it based on feature groups
            num_features = self.config.num_features
            feature_dim = d_model // num_features  # This might not be exact

            # Create separate masks for different feature dimensions
            mask_3d = torch.zeros(seq_len, batch_size, d_model, device=x.device)

            # Divide d_model into num_features chunks (as evenly as possible)
            chunk_size = d_model // num_features
            remainder = d_model % num_features

            start_idx = 0
            for feat_idx in range(num_features):
                # Some features get one extra dimension if there's a remainder
                current_chunk = chunk_size + (1 if feat_idx < remainder else 0)
                end_idx = start_idx + current_chunk

                # Create independent mask for this feature
                noise = torch.rand(seq_len, batch_size, device=x.device)
                ids_shuffle = torch.argsort(noise, dim=0)
                ids_restore = torch.argsort(ids_shuffle, dim=0)

                feat_mask = torch.zeros(seq_len, batch_size, device=x.device)
                feat_mask[:len_keep, :] = 1
                feat_mask = torch.gather(feat_mask, dim=0, index=ids_restore)

                # Apply to this feature's dimensions
                mask_3d[:, :, start_idx:end_idx] = feat_mask.unsqueeze(-1)

                start_idx = end_idx

            # Apply mask
            mask_tokens = self.mask_token.repeat(seq_len, batch_size, 1)
            x_masked = x * mask_3d + mask_tokens * (1 - mask_3d)

            # Return 2D mask for compatibility (average across d_model)
            mask_2d = mask_3d.mean(dim=2)  # (seq_len, batch_size)

            return x_masked, mask_2d

        else:
            raise ValueError(f"Unknown masking strategy: {self.config.masking_strategy}")

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to patches
        Args:
            x: (batch, seq_length, num_features)
        Returns:
            patches: (batch, num_patches, patch_size * num_features)
        """
        batch_size, seq_length, num_features = x.shape
        assert seq_length % self.patch_size == 0, f"seq_length {seq_length} must be divisible by patch_size {self.patch_size}"

        # Reshape to patches
        x = x.reshape(batch_size, self.num_patches, self.patch_size * num_features)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to time series
        Args:
            x: (batch, num_patches, patch_size * num_features)
        Returns:
            time_series: (batch, seq_length, num_features)
        """
        batch_size = x.shape[0]
        # Reshape back to time series
        x = x.reshape(batch_size, self.num_patches * self.patch_size, self.config.num_features)
        return x

    def forward(
        self,
        x: torch.Tensor,
        masking_ratio: Optional[float] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_length, num_features)
        Returns:
            teacher_output, student_output, mask
        """
        if masking_ratio is None:
            masking_ratio = self.config.masking_ratio

        batch_size, seq_length, num_features = x.shape

        # Input projection: different for patch vs token
        if self.use_patch:
            # Patchify: (batch, seq, features) -> (batch, num_patches, patch_size*features)
            x_patches = self.patchify(x)
            # Embed patches: (batch, num_patches, patch_size*features) -> (batch, num_patches, d_model)
            x_embed = self.patch_embed(x_patches)
            x_embed = x_embed.transpose(0, 1)  # (num_patches, batch, d_model)
        else:
            # Token-based projection
            x_embed = self.input_projection(x)  # (batch, seq, d_model)
            x_embed = x_embed.transpose(0, 1)  # (seq, batch, d_model)

        # Masking
        if mask is None:
            x_masked, mask = self.random_masking(x_embed, masking_ratio)
        else:
            # For patch mode, mask should be at patch level
            if self.use_patch and mask.shape[1] != self.num_patches:
                # Convert time-step mask to patch mask (take last token of each patch)
                mask_reshaped = mask.reshape(batch_size, self.num_patches, self.patch_size)
                mask = mask_reshaped[:, :, -1]  # Use last time step of each patch

            mask = mask.transpose(0, 1)
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
            teacher_output = self.output_projection(teacher_hidden)
            teacher_output = teacher_output.transpose(0, 1)  # (batch, num_patches/seq, patch_size*features or features)

            # Unpatchify if using patch mode
            if self.use_patch:
                teacher_output = self.unpatchify(teacher_output)  # (batch, seq_length, num_features)

        if self.config.use_student and self.student_decoder is not None:
            student_hidden = self.student_decoder(tgt, latent)
            student_output = self.output_projection(student_hidden)
            student_output = student_output.transpose(0, 1)

            # Unpatchify if using patch mode
            if self.use_patch:
                student_output = self.unpatchify(student_output)  # (batch, seq_length, num_features)

        # If only one decoder, use it for both outputs (for compatibility)
        if not self.config.use_teacher and self.config.use_student:
            teacher_output = student_output
        elif self.config.use_teacher and not self.config.use_student:
            student_output = teacher_output

        # Convert mask back to original shape
        mask = mask.transpose(0, 1)  # (batch, num_patches) or (batch, seq_length)

        # For patch mode, expand mask to match original sequence length
        if self.use_patch:
            # Expand patch mask to time-step mask: (batch, num_patches) -> (batch, seq_length)
            mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_size)  # (batch, num_patches, patch_size)
            mask = mask.reshape(batch_size, seq_length)  # (batch, seq_length)

        return teacher_output, student_output, mask


class SelfDistillationLoss(nn.Module):
    """Self-distillation loss with reconstruction and discrepancy components"""

    def __init__(self, margin: float = 1.0, lambda_disc: float = 0.5):
        super().__init__()
        self.margin = margin
        self.lambda_disc = lambda_disc
        self.mse = nn.MSELoss(reduction='none')

    def forward(
        self,
        teacher_out: torch.Tensor,
        student_out: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        use_discrepancy: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            teacher_out: (batch, seq_length, num_features)
            student_out: (batch, seq_length, num_features)
            target: (batch, seq_length, num_features)
            mask: (batch, seq_length) - 1=kept, 0=masked
            use_discrepancy: whether to use discrepancy loss

        Returns:
            total_loss, recon_loss, disc_loss
        """
        # Reconstruction loss (MSE on masked regions)
        recon_error_teacher = self.mse(teacher_out, target)  # (batch, seq, features)
        recon_error_student = self.mse(student_out, target)

        # Average over features, keep sequence dimension
        recon_error_teacher = recon_error_teacher.mean(dim=2)  # (batch, seq)
        recon_error_student = recon_error_student.mean(dim=2)  # (batch, seq)

        # Apply mask (focus on masked regions)
        mask_inverse = 1 - mask  # 0=kept, 1=masked
        masked_recon_teacher = (recon_error_teacher * mask_inverse).sum() / (mask_inverse.sum() + 1e-8)
        masked_recon_student = (recon_error_student * mask_inverse).sum() / (mask_inverse.sum() + 1e-8)

        recon_loss = masked_recon_teacher + masked_recon_student

        # Discrepancy loss (hinge loss)
        disc_loss = torch.tensor(0.0, device=teacher_out.device)
        if use_discrepancy:
            # Compute discrepancy on masked regions
            discrepancy = torch.abs(recon_error_teacher - recon_error_student)  # (batch, seq)
            masked_discrepancy = (discrepancy * mask_inverse).sum() / (mask_inverse.sum() + 1e-8)

            # Hinge loss: encourage discrepancy to be at least margin
            disc_loss = torch.relu(self.margin - masked_discrepancy)

        total_loss = recon_loss + self.lambda_disc * disc_loss

        return total_loss, recon_loss, disc_loss
