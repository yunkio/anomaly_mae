"""
Comprehensive Experiments for Self-Distilled MAE on Multivariate Time Series
- Multivariate data generation
- Hyperparameter tuning experiments
- Ablation studies
- Comprehensive visualizations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import pandas as pd
import random
from tqdm import tqdm
import warnings
import json
import os
from copy import deepcopy
from datetime import datetime
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Configuration for experiments"""
    # Data parameters
    seq_length: int = 100
    num_features: int = 5  # Multivariate: 5 features
    num_train_samples: int = 2000
    num_test_samples: int = 500
    train_anomaly_ratio: float = 0.05
    test_anomaly_ratio: float = 0.25

    # Model parameters
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 3
    num_teacher_decoder_layers: int = 4
    num_student_decoder_layers: int = 1
    dim_feedforward: int = 256
    dropout: float = 0.1
    masking_ratio: float = 0.6
    masking_strategy: str = 'patch'  # 'patch', 'token', 'temporal', or 'feature_wise'
    patch_size: int = 10  # Number of time steps per patch (for patch-based masking)

    # Loss parameters
    margin: float = 1.0
    lambda_disc: float = 0.5

    # Training parameters
    batch_size: int = 32
    num_epochs: int = 50  # Reduced for faster experiments
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 10

    # Inference parameters
    mask_last_n: int = 10

    # Ablation flags
    use_discrepancy_loss: bool = True
    use_teacher: bool = True
    use_student: bool = True
    use_masking: bool = True

    # Reproducibility
    random_seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Multivariate Time Series Dataset
# ============================================================================

class MultivariateTimeSeriesDataset(Dataset):
    """
    Multivariate time series dataset simulating server monitoring data
    Features: CPU, Memory, Disk I/O, Network, Response Time
    """

    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        num_features: int,
        anomaly_ratio: float,
        is_train: bool = True,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.anomaly_ratio = anomaly_ratio
        self.is_train = is_train

        if seed is not None:
            np.random.seed(seed)

        self.feature_names = ['CPU', 'Memory', 'DiskIO', 'Network', 'ResponseTime'][:num_features]
        self.data, self.seq_labels, self.point_labels = self._generate_data()

    def _generate_normal_multivariate(self) -> np.ndarray:
        """
        Generate normal multivariate time series with correlations
        Simulating server monitoring data
        """
        t = np.linspace(0, 4 * np.pi, self.seq_length)
        signals = np.zeros((self.seq_length, self.num_features))

        # Feature 0: CPU usage (base pattern)
        cpu_freq = np.random.uniform(0.5, 2.0)
        cpu_amp = np.random.uniform(0.3, 0.7)
        cpu_base = np.random.uniform(0.2, 0.4)
        signals[:, 0] = cpu_base + cpu_amp * np.sin(cpu_freq * t) + np.random.normal(0, 0.05, self.seq_length)

        # Feature 1: Memory usage (correlated with CPU, slower variation)
        if self.num_features > 1:
            mem_base = np.random.uniform(0.4, 0.6)
            signals[:, 1] = mem_base + 0.3 * signals[:, 0] + 0.2 * np.sin(cpu_freq * 0.5 * t) + np.random.normal(0, 0.03, self.seq_length)

        # Feature 2: Disk I/O (spiky, correlated with memory)
        if self.num_features > 2:
            disk_base = np.random.uniform(0.1, 0.3)
            disk_spikes = np.random.poisson(0.1, self.seq_length) * 0.2
            signals[:, 2] = disk_base + 0.2 * signals[:, 1] + disk_spikes + np.random.normal(0, 0.05, self.seq_length)

        # Feature 3: Network traffic (bursty)
        if self.num_features > 3:
            net_freq = np.random.uniform(1.0, 3.0)
            net_amp = np.random.uniform(0.2, 0.5)
            signals[:, 3] = net_amp * np.abs(np.sin(net_freq * t)) + np.random.normal(0, 0.05, self.seq_length)

        # Feature 4: Response time (correlated with CPU and Network)
        if self.num_features > 4:
            resp_base = np.random.uniform(0.1, 0.3)
            signals[:, 4] = resp_base + 0.3 * signals[:, 0] + 0.2 * signals[:, 3] + np.random.normal(0, 0.05, self.seq_length)

        # Clip to [0, 1] range
        signals = np.clip(signals, 0, 1)
        return signals

    def _inject_multivariate_spike(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU and Network spike (DDoS attack scenario)
        Returns: (modified_signals, point_labels) where point_labels is (seq_length,) binary array
        """
        margin = min(20, self.seq_length // 4)
        spike_pos = np.random.randint(margin, max(margin + 1, self.seq_length - margin))
        spike_width = np.random.randint(3, min(10, self.seq_length // 10 + 1))

        # Point-level labels
        point_labels = np.zeros(self.seq_length, dtype=np.int64)
        point_labels[spike_pos:spike_pos + spike_width] = 1

        # CPU spike
        signals[spike_pos:spike_pos + spike_width, 0] += np.random.uniform(0.3, 0.5)
        # Network spike (if exists)
        if self.num_features > 3:
            signals[spike_pos:spike_pos + spike_width, 3] += np.random.uniform(0.4, 0.6)
        # Response time spike (if exists)
        if self.num_features > 4:
            signals[spike_pos:spike_pos + spike_width, 4] += np.random.uniform(0.3, 0.5)

        return np.clip(signals, 0, 1), point_labels

    def _inject_multivariate_memory_leak(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Memory gradual increase (memory leak scenario)
        Returns: (modified_signals, point_labels)
        """
        point_labels = np.zeros(self.seq_length, dtype=np.int64)

        if self.num_features < 2:
            return signals, point_labels

        margin = min(20, self.seq_length // 5)
        start_pos = np.random.randint(margin, max(margin + 1, self.seq_length // 2))
        leak_length = self.seq_length - start_pos

        # Point-level labels
        point_labels[start_pos:] = 1

        # Gradual memory increase
        leak = np.linspace(0, np.random.uniform(0.3, 0.5), leak_length)
        signals[start_pos:, 1] += leak

        # Disk I/O increases due to swapping
        if self.num_features > 2:
            signals[start_pos:, 2] += leak * 0.5

        return np.clip(signals, 0, 1), point_labels

    def _inject_multivariate_noise(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Increased noise across all features
        Returns: (modified_signals, point_labels)
        """
        margin = min(20, self.seq_length // 5)
        max_length = min(40, self.seq_length // 2)
        min_length = min(20, self.seq_length // 4)

        noise_start = np.random.randint(margin, max(margin + 1, self.seq_length - max_length))
        noise_length = np.random.randint(min_length, max(min_length + 1, max_length))

        # Point-level labels
        point_labels = np.zeros(self.seq_length, dtype=np.int64)
        point_labels[noise_start:noise_start + noise_length] = 1

        # Add noise to all features
        for i in range(self.num_features):
            noise = np.random.normal(0, 0.2, noise_length)
            signals[noise_start:noise_start + noise_length, i] += noise

        return np.clip(signals, 0, 1), point_labels

    def _inject_multivariate_drift(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gradual drift in CPU and response time
        Returns: (modified_signals, point_labels)
        """
        margin = min(20, self.seq_length // 5)
        max_length = min(50, self.seq_length // 2)
        min_length = min(30, self.seq_length // 3)

        drift_start = np.random.randint(margin, max(margin + 1, self.seq_length - max_length))
        drift_length = np.random.randint(min_length, max(min_length + 1, max_length))

        # Point-level labels
        point_labels = np.zeros(self.seq_length, dtype=np.int64)
        point_labels[drift_start:drift_start + drift_length] = 1

        drift = np.linspace(0, np.random.uniform(0.2, 0.4), drift_length)

        # CPU drift
        signals[drift_start:drift_start + drift_length, 0] += drift
        # Response time drift
        if self.num_features > 4:
            signals[drift_start:drift_start + drift_length, 4] += drift * 0.5

        return np.clip(signals, 0, 1), point_labels

    def _inject_multivariate_network_congestion(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Network congestion with response time spike
        Returns: (modified_signals, point_labels)
        """
        point_labels = np.zeros(self.seq_length, dtype=np.int64)

        if self.num_features < 4:
            return signals, point_labels

        margin = min(30, self.seq_length // 3)
        change_point = np.random.randint(margin, max(margin + 1, self.seq_length - margin))

        # Point-level labels
        point_labels[change_point:] = 1

        # High network traffic after change point
        signals[change_point:, 3] += np.random.uniform(0.3, 0.5)

        # Response time increases
        if self.num_features > 4:
            signals[change_point:, 4] += np.random.uniform(0.2, 0.4)

        return np.clip(signals, 0, 1), point_labels

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate complete multivariate dataset
        Returns: (data, sequence_labels, point_labels)
            data: (num_samples, seq_length, num_features)
            sequence_labels: (num_samples,) - binary sequence-level labels
            point_labels: (num_samples, seq_length) - binary point-level labels
        """
        data = []
        seq_labels = []
        point_labels = []

        num_anomalies = int(self.num_samples * self.anomaly_ratio)
        num_normal = self.num_samples - num_anomalies

        # Generate normal sequences
        for _ in range(num_normal):
            signals = self._generate_normal_multivariate()
            data.append(signals)
            seq_labels.append(0)
            point_labels.append(np.zeros(self.seq_length, dtype=np.int64))

        # Generate anomalous sequences
        anomaly_funcs = [
            self._inject_multivariate_spike,
            self._inject_multivariate_memory_leak,
            self._inject_multivariate_noise,
            self._inject_multivariate_drift,
            self._inject_multivariate_network_congestion
        ]

        for _ in range(num_anomalies):
            signals = self._generate_normal_multivariate()
            anomaly_func = np.random.choice(anomaly_funcs)
            signals, point_label = anomaly_func(signals)
            data.append(signals)
            seq_labels.append(1)
            point_labels.append(point_label)

        # Shuffle
        data = np.array(data, dtype=np.float32)
        seq_labels = np.array(seq_labels, dtype=np.int64)
        point_labels = np.array(point_labels, dtype=np.int64)

        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        data = data[indices]
        seq_labels = seq_labels[indices]
        point_labels = point_labels[indices]

        return data, seq_labels, point_labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (sequence, seq_label, point_labels)
        sequence: (seq_length, num_features)
        seq_label: scalar - sequence-level label
        point_labels: (seq_length,) - point-level labels
        """
        sequence = torch.from_numpy(self.data[idx])  # (seq_length, num_features)
        seq_label = torch.tensor(self.seq_labels[idx], dtype=torch.long)
        point_label = torch.from_numpy(self.point_labels[idx])  # (seq_length,)
        return sequence, seq_label, point_label


# ============================================================================
# Model Architecture (Modified for Multivariate)
# ============================================================================

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


# ============================================================================
# Loss Function
# ============================================================================

class SelfDistillationLoss(nn.Module):
    """Custom loss for self-distilled MAE"""

    def __init__(self, config: Config):
        super().__init__()
        self.margin = config.margin
        self.lambda_disc = config.lambda_disc if config.use_discrepancy_loss else 0.0
        self.use_discrepancy = config.use_discrepancy_loss

    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        original_input: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        warmup_factor: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss"""
        # Reconstruction loss
        mask_expanded = mask.unsqueeze(-1)
        reconstruction_loss = F.mse_loss(
            teacher_output * (1 - mask_expanded),
            original_input * (1 - mask_expanded),
            reduction='sum'
        ) / ((1 - mask_expanded).sum() + 1e-8)

        # Discrepancy loss
        if self.use_discrepancy and self.lambda_disc > 0:
            discrepancy = torch.mean((teacher_output - student_output) ** 2, dim=(1, 2))
            normal_mask = (labels == 0).float()
            anomaly_mask = (labels == 1).float()

            normal_loss = (normal_mask * discrepancy).sum() / (normal_mask.sum() + 1e-8)
            anomaly_loss = (anomaly_mask * F.relu(self.margin - discrepancy)).sum() / (anomaly_mask.sum() + 1e-8)
            anomaly_loss = warmup_factor * anomaly_loss

            discrepancy_loss = normal_loss + anomaly_loss
            total_loss = reconstruction_loss + self.lambda_disc * discrepancy_loss
        else:
            discrepancy_loss = torch.tensor(0.0)
            normal_loss = torch.tensor(0.0)
            anomaly_loss = torch.tensor(0.0)
            discrepancy = torch.zeros(teacher_output.size(0), device=teacher_output.device)
            total_loss = reconstruction_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'discrepancy_loss': discrepancy_loss.item() if isinstance(discrepancy_loss, torch.Tensor) else discrepancy_loss,
            'normal_loss': normal_loss.item() if isinstance(normal_loss, torch.Tensor) else normal_loss,
            'anomaly_loss': anomaly_loss.item() if isinstance(anomaly_loss, torch.Tensor) else anomaly_loss,
            'mean_discrepancy': discrepancy.mean().item()
        }

        return total_loss, loss_dict


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """Trainer class"""

    def __init__(
        self,
        model: SelfDistilledMAEMultivariate,
        config: Config,
        train_loader: DataLoader,
        test_loader: DataLoader,
        verbose: bool = True
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.verbose = verbose

        self.criterion = SelfDistillationLoss(config)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        self.model = self.model.to(config.device)

        self.history = {
            'train_loss': [],
            'train_rec_loss': [],
            'train_disc_loss': [],
            'epoch': []
        }

    def _compute_warmup_factor(self, epoch: int) -> float:
        if epoch < self.config.warmup_epochs:
            return epoch / self.config.warmup_epochs
        return 1.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'discrepancy_loss': 0.0,
            'normal_loss': 0.0,
            'anomaly_loss': 0.0,
            'mean_discrepancy': 0.0
        }

        warmup_factor = self._compute_warmup_factor(epoch)

        iterator = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}', disable=not self.verbose)

        for sequences, seq_labels, point_labels in iterator:
            sequences = sequences.to(self.config.device)
            seq_labels = seq_labels.to(self.config.device)
            point_labels = point_labels.to(self.config.device)  # Not used in training, but loaded

            teacher_output, student_output, mask = self.model(sequences)

            loss, loss_dict = self.criterion(
                teacher_output, student_output, sequences, mask, seq_labels, warmup_factor
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key]

        for key in epoch_losses.keys():
            epoch_losses[key] /= len(self.train_loader)

        return epoch_losses

    def train(self) -> None:
        for epoch in range(self.config.num_epochs):
            epoch_losses = self.train_epoch(epoch)
            self.scheduler.step()

            self.history['train_loss'].append(epoch_losses['total_loss'])
            self.history['train_rec_loss'].append(epoch_losses['reconstruction_loss'])
            self.history['train_disc_loss'].append(epoch_losses['discrepancy_loss'])
            self.history['epoch'].append(epoch + 1)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={epoch_losses['total_loss']:.4f}, "
                      f"Rec={epoch_losses['reconstruction_loss']:.4f}, "
                      f"Disc={epoch_losses['discrepancy_loss']:.4f}")


# ============================================================================
# Evaluator
# ============================================================================

class Evaluator:
    """Evaluator for anomaly detection"""

    def __init__(
        self,
        model: SelfDistilledMAEMultivariate,
        config: Config,
        test_loader: DataLoader
    ):
        self.model = model
        self.config = config
        self.test_loader = test_loader
        self.model.eval()

    def compute_anomaly_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute anomaly scores for both sequence-level and point-level

        Returns:
            seq_scores: (num_samples,) - sequence-level anomaly scores
            seq_labels: (num_samples,) - sequence-level ground truth labels
            point_scores: (num_samples * seq_length,) - point-level anomaly scores (flattened)
            point_labels: (num_samples * seq_length,) - point-level ground truth labels (flattened)
        """
        all_seq_scores = []
        all_seq_labels = []
        all_point_scores = []
        all_point_labels = []

        with torch.no_grad():
            for sequences, seq_labels, point_labels in self.test_loader:
                sequences = sequences.to(self.config.device)
                batch_size, seq_length, num_features = sequences.shape

                # Mask last N time steps for prediction
                mask = torch.ones(batch_size, seq_length, device=self.config.device)
                mask[:, -self.config.mask_last_n:] = 0

                teacher_output, student_output, _ = self.model(sequences, masking_ratio=0.0, mask=mask)

                # Compute error at each time step
                if not self.config.use_student and self.config.use_teacher:
                    # TeacherOnly ablation: use reconstruction error
                    error = ((teacher_output - sequences) ** 2).mean(dim=2)  # (batch, seq_length)
                elif self.config.use_student and not self.config.use_teacher:
                    # StudentOnly ablation: use reconstruction error
                    error = ((student_output - sequences) ** 2).mean(dim=2)  # (batch, seq_length)
                else:
                    # Normal case: use teacher-student discrepancy
                    error = ((teacher_output - student_output) ** 2).mean(dim=2)  # (batch, seq_length)

                # Sequence-level score: average over masked positions (last N steps)
                masked_positions = (mask == 0)  # (batch, seq_length)
                seq_scores = (error * masked_positions).sum(dim=1) / (masked_positions.sum(dim=1) + 1e-8)

                # Point-level scores: error at each time step
                point_scores = error  # (batch, seq_length)

                all_seq_scores.append(seq_scores.cpu().numpy())
                all_seq_labels.append(seq_labels.cpu().numpy())
                all_point_scores.append(point_scores.cpu().numpy())
                all_point_labels.append(point_labels.cpu().numpy())

        # Concatenate and flatten point-level arrays
        seq_scores = np.concatenate(all_seq_scores)
        seq_labels = np.concatenate(all_seq_labels)
        point_scores = np.concatenate(all_point_scores).flatten()
        point_labels = np.concatenate(all_point_labels).flatten()

        return seq_scores, seq_labels, point_scores, point_labels

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Evaluate and return metrics for sequence-level, point-level, and combined

        Returns:
            Dictionary with three keys: 'sequence', 'point', 'combined'
            Each containing: roc_auc, precision, recall, f1_score, optimal_threshold
        """
        seq_scores, seq_labels, point_scores, point_labels = self.compute_anomaly_scores()

        results = {}

        # 1. Sequence-level evaluation
        if len(np.unique(seq_labels)) > 1:  # Check if we have both classes
            seq_roc_auc = roc_auc_score(seq_labels, seq_scores)
            fpr, tpr, thresholds = roc_curve(seq_labels, seq_scores)
            optimal_idx = np.argmax(tpr - fpr)
            seq_threshold = thresholds[optimal_idx]

            seq_predictions = (seq_scores > seq_threshold).astype(int)
            seq_precision = precision_score(seq_labels, seq_predictions, zero_division=0)
            seq_recall = recall_score(seq_labels, seq_predictions, zero_division=0)
            seq_f1 = f1_score(seq_labels, seq_predictions, zero_division=0)

            results['sequence'] = {
                'roc_auc': seq_roc_auc,
                'precision': seq_precision,
                'recall': seq_recall,
                'f1_score': seq_f1,
                'optimal_threshold': seq_threshold
            }
        else:
            results['sequence'] = {
                'roc_auc': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'optimal_threshold': 0.0
            }

        # 2. Point-level evaluation
        if len(np.unique(point_labels)) > 1:  # Check if we have both classes
            point_roc_auc = roc_auc_score(point_labels, point_scores)
            fpr, tpr, thresholds = roc_curve(point_labels, point_scores)
            optimal_idx = np.argmax(tpr - fpr)
            point_threshold = thresholds[optimal_idx]

            point_predictions = (point_scores > point_threshold).astype(int)
            point_precision = precision_score(point_labels, point_predictions, zero_division=0)
            point_recall = recall_score(point_labels, point_predictions, zero_division=0)
            point_f1 = f1_score(point_labels, point_predictions, zero_division=0)

            results['point'] = {
                'roc_auc': point_roc_auc,
                'precision': point_precision,
                'recall': point_recall,
                'f1_score': point_f1,
                'optimal_threshold': point_threshold
            }
        else:
            results['point'] = {
                'roc_auc': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'optimal_threshold': 0.0
            }

        # 3. Combined evaluation (weighted average)
        # Weight by F1 scores, or equal weight if both are 0
        seq_f1 = results['sequence']['f1_score']
        point_f1 = results['point']['f1_score']
        total_weight = seq_f1 + point_f1

        if total_weight > 0:
            seq_weight = seq_f1 / total_weight
            point_weight = point_f1 / total_weight
        else:
            seq_weight = 0.5
            point_weight = 0.5

        results['combined'] = {
            'roc_auc': seq_weight * results['sequence']['roc_auc'] + point_weight * results['point']['roc_auc'],
            'precision': seq_weight * results['sequence']['precision'] + point_weight * results['point']['precision'],
            'recall': seq_weight * results['sequence']['recall'] + point_weight * results['point']['recall'],
            'f1_score': seq_weight * results['sequence']['f1_score'] + point_weight * results['point']['f1_score'],
            'seq_weight': seq_weight,
            'point_weight': point_weight
        }

        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"\n{'Metric':<20} {'Sequence-Level':<20} {'Point-Level':<20} {'Combined':<20}")
        print("-"*80)
        print(f"{'ROC-AUC':<20} {results['sequence']['roc_auc']:<20.4f} {results['point']['roc_auc']:<20.4f} {results['combined']['roc_auc']:<20.4f}")
        print(f"{'Precision':<20} {results['sequence']['precision']:<20.4f} {results['point']['precision']:<20.4f} {results['combined']['precision']:<20.4f}")
        print(f"{'Recall':<20} {results['sequence']['recall']:<20.4f} {results['point']['recall']:<20.4f} {results['combined']['recall']:<20.4f}")
        print(f"{'F1-Score':<20} {results['sequence']['f1_score']:<20.4f} {results['point']['f1_score']:<20.4f} {results['combined']['f1_score']:<20.4f}")
        print("="*80)
        print(f"Combined weights: Sequence={seq_weight:.2f}, Point={point_weight:.2f}")
        print("="*80)

        return results


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """Run comprehensive experiments"""

    def __init__(self, base_config: Config, output_dir: str = './experiment_results'):
        self.base_config = base_config

        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"\nExperiment results will be saved to: {self.output_dir}")

        self.results = []

    def run_single_experiment(
        self,
        config: Config,
        experiment_name: str,
        train_dataset: Dataset,
        test_dataset: Dataset
    ) -> Dict:
        """Run a single experiment"""
        print(f"\n{'='*80}")
        print(f"Running: {experiment_name}")
        print(f"{'='*80}")

        set_seed(config.random_seed)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

        model = SelfDistilledMAEMultivariate(config)
        trainer = Trainer(model, config, train_loader, test_loader, verbose=True)
        trainer.train()

        evaluator = Evaluator(model, config, test_loader)
        metrics = evaluator.evaluate()

        result = {
            'experiment_name': experiment_name,
            'config': asdict(config),
            'metrics': metrics,
            'history': trainer.history
        }

        self.results.append(result)
        print(f"\nResults: ROC-AUC={metrics['combined']['roc_auc']:.4f}, F1={metrics['combined']['f1_score']:.4f}")
        print(f"  Sequence-Level: ROC-AUC={metrics['sequence']['roc_auc']:.4f}, F1={metrics['sequence']['f1_score']:.4f}")
        print(f"  Point-Level: ROC-AUC={metrics['point']['roc_auc']:.4f}, F1={metrics['point']['f1_score']:.4f}")

        return result

    def run_hyperparameter_experiments(self, train_dataset: Dataset, test_dataset: Dataset):
        """Run hyperparameter tuning experiments"""
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING EXPERIMENTS")
        print("="*80)

        # Baseline
        baseline_config = deepcopy(self.base_config)
        self.run_single_experiment(baseline_config, "Baseline", train_dataset, test_dataset)

        # Masking ratio
        for ratio in [0.4, 0.75]:
            config = deepcopy(self.base_config)
            config.masking_ratio = ratio
            self.run_single_experiment(config, f"MaskingRatio_{ratio}", train_dataset, test_dataset)

        # Lambda discrepancy
        for lam in [0.1, 1.0]:
            config = deepcopy(self.base_config)
            config.lambda_disc = lam
            self.run_single_experiment(config, f"LambdaDisc_{lam}", train_dataset, test_dataset)

        # Margin
        for margin in [0.5, 2.0]:
            config = deepcopy(self.base_config)
            config.margin = margin
            self.run_single_experiment(config, f"Margin_{margin}", train_dataset, test_dataset)

        # D_model
        for d in [32, 128]:
            config = deepcopy(self.base_config)
            config.d_model = d
            self.run_single_experiment(config, f"DModel_{d}", train_dataset, test_dataset)

    def run_ablation_experiments(self, train_dataset: Dataset, test_dataset: Dataset):
        """Run ablation studies"""
        print("\n" + "="*80)
        print("ABLATION STUDIES")
        print("="*80)

        # No discrepancy loss
        config = deepcopy(self.base_config)
        config.use_discrepancy_loss = False
        self.run_single_experiment(config, "Ablation_NoDiscrepancy", train_dataset, test_dataset)

        # Teacher only (use reconstruction error for anomaly detection)
        config = deepcopy(self.base_config)
        config.use_student = False
        config.use_discrepancy_loss = False
        self.run_single_experiment(config, "Ablation_TeacherOnly", train_dataset, test_dataset)

        # Student only (use reconstruction error for anomaly detection)
        config = deepcopy(self.base_config)
        config.use_teacher = False
        config.use_discrepancy_loss = False
        self.run_single_experiment(config, "Ablation_StudentOnly", train_dataset, test_dataset)

        # No masking
        config = deepcopy(self.base_config)
        config.use_masking = False
        self.run_single_experiment(config, "Ablation_NoMasking", train_dataset, test_dataset)

    def run_masking_strategy_experiments(self, train_dataset: Dataset, test_dataset: Dataset):
        """Compare different masking strategies"""
        print("\n" + "="*80)
        print("MASKING STRATEGY COMPARISON")
        print("="*80)

        # Patch-based masking (MAE/ViT style - default)
        config = deepcopy(self.base_config)
        config.masking_strategy = 'patch'
        self.run_single_experiment(config, "Masking_Patch", train_dataset, test_dataset)

        # Token-level masking (BERT style - for comparison)
        config = deepcopy(self.base_config)
        config.masking_strategy = 'token'
        self.run_single_experiment(config, "Masking_Token", train_dataset, test_dataset)

        # Temporal masking (all features masked together)
        config = deepcopy(self.base_config)
        config.masking_strategy = 'temporal'
        self.run_single_experiment(config, "Masking_Temporal", train_dataset, test_dataset)

        # Feature-wise masking (each feature independently masked)
        config = deepcopy(self.base_config)
        config.masking_strategy = 'feature_wise'
        self.run_single_experiment(config, "Masking_FeatureWise", train_dataset, test_dataset)

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def save_results(self):
        """Save results to JSON"""
        filepath = os.path.join(self.output_dir, 'experiment_results.json')
        # Convert numpy types to native Python types
        serializable_results = self._convert_to_serializable(self.results)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nResults saved to {filepath}")

    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # 1. Hyperparameter comparison
        self._plot_hyperparameter_comparison()

        # 2. Ablation study comparison
        self._plot_ablation_comparison()

        # 3. Training curves
        self._plot_training_curves()

        # 4. ROC curves comparison
        self._plot_roc_comparison()

        # 5. Performance heatmap
        self._plot_performance_heatmap()

    def _plot_hyperparameter_comparison(self):
        """Plot hyperparameter tuning results"""
        hyperparameter_results = [r for r in self.results if not r['experiment_name'].startswith('Ablation')]

        metrics_to_plot = ['roc_auc', 'f1_score', 'precision', 'recall']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            names = [r['experiment_name'] for r in hyperparameter_results]
            values = [r['metrics']['combined'][metric] for r in hyperparameter_results]

            axes[idx].barh(names, values, color=sns.color_palette("husl", len(names)))
            axes[idx].set_xlabel(metric.upper().replace('_', ' '))
            axes[idx].set_title(f'Hyperparameter Comparison: {metric.upper()} (Combined)')
            axes[idx].set_xlim([0, 1])

            for i, v in enumerate(values):
                axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'hyperparameter_comparison.png'), dpi=150, bbox_inches='tight')
        print("✓ Saved hyperparameter_comparison.png")
        plt.close()

    def _plot_ablation_comparison(self):
        """Plot ablation study results"""
        ablation_results = [r for r in self.results if r['experiment_name'].startswith('Ablation') or r['experiment_name'] == 'Baseline']

        metrics_to_plot = ['roc_auc', 'f1_score', 'precision', 'recall']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            names = [r['experiment_name'] for r in ablation_results]
            values = [r['metrics']['combined'][metric] for r in ablation_results]

            colors = ['green' if 'Baseline' in n else 'coral' for n in names]
            axes[idx].barh(names, values, color=colors)
            axes[idx].set_xlabel(metric.upper().replace('_', ' '))
            axes[idx].set_title(f'Ablation Study: {metric.upper()} (Combined)')
            axes[idx].set_xlim([0, 1])

            for i, v in enumerate(values):
                axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ablation_comparison.png'), dpi=150, bbox_inches='tight')
        print("✓ Saved ablation_comparison.png")
        plt.close()

    def _plot_training_curves(self):
        """Plot training curves for all experiments"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for result in self.results[:10]:  # Plot first 10 to avoid clutter
            name = result['experiment_name']
            history = result['history']

            axes[0].plot(history['epoch'], history['train_loss'], label=name, alpha=0.7)
            axes[1].plot(history['epoch'], history['train_rec_loss'], label=name, alpha=0.7)
            axes[2].plot(history['epoch'], history['train_disc_loss'], label=name, alpha=0.7)

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Discrepancy Loss')
        axes[2].set_title('Discrepancy Loss')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        print("✓ Saved training_curves.png")
        plt.close()

    def _plot_roc_comparison(self):
        """Plot ROC curves comparison (would need actual FPR/TPR data)"""
        # Placeholder - would need to store FPR/TPR during evaluation
        print("✓ ROC comparison (requires FPR/TPR data - skipped for now)")

    def _plot_performance_heatmap(self):
        """Plot performance heatmap"""
        # Create dataframe
        data = []
        for result in self.results:
            row = {
                'Experiment': result['experiment_name'][:20],  # Truncate name
                'ROC-AUC': result['metrics']['combined']['roc_auc'],
                'F1': result['metrics']['combined']['f1_score'],
                'Precision': result['metrics']['combined']['precision'],
                'Recall': result['metrics']['combined']['recall']
            }
            data.append(row)

        df = pd.DataFrame(data)
        df = df.set_index('Experiment')

        plt.figure(figsize=(10, max(8, len(data) * 0.5)))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Score'})
        plt.title('Performance Heatmap Across All Experiments (Combined Metrics)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_heatmap.png'), dpi=150, bbox_inches='tight')
        print("✓ Saved performance_heatmap.png")
        plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run comprehensive experiments"""
    print("="*80)
    print("COMPREHENSIVE EXPERIMENTS: Self-Distilled MAE on Multivariate Time Series")
    print("="*80)

    # Base configuration
    base_config = Config()
    base_config.num_epochs = 100  # Full experiments
    base_config.num_train_samples = 1000
    base_config.num_test_samples = 300

    set_seed(base_config.random_seed)

    # Create datasets
    print("\nGenerating multivariate datasets...")
    train_dataset = MultivariateTimeSeriesDataset(
        num_samples=base_config.num_train_samples,
        seq_length=base_config.seq_length,
        num_features=base_config.num_features,
        anomaly_ratio=base_config.train_anomaly_ratio,
        is_train=True,
        seed=base_config.random_seed
    )

    test_dataset = MultivariateTimeSeriesDataset(
        num_samples=base_config.num_test_samples,
        seq_length=base_config.seq_length,
        num_features=base_config.num_features,
        anomaly_ratio=base_config.test_anomaly_ratio,
        is_train=False,
        seed=base_config.random_seed + 1
    )

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Features: {train_dataset.feature_names}")

    # Run experiments
    runner = ExperimentRunner(base_config)

    # 1. Hyperparameter experiments
    runner.run_hyperparameter_experiments(train_dataset, test_dataset)

    # 2. Masking strategy comparison
    runner.run_masking_strategy_experiments(train_dataset, test_dataset)

    # 3. Ablation studies
    runner.run_ablation_experiments(train_dataset, test_dataset)

    # 4. Save results
    runner.save_results()

    # 5. Generate visualizations
    runner.generate_visualizations()

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved to: {runner.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
