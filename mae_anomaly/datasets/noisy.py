"""Noisy label dataset wrapper for label noise experiments."""

import numpy as np
from ..dataset_sliding import SlidingWindowDataset


class NoisyLabelSlidingWindowDataset(SlidingWindowDataset):
    """SlidingWindowDataset with noisy labels (e.g., 50% anomalies relabeled as normal).

    This wrapper modifies point_labels to simulate label noise while keeping
    the actual data unchanged. During training, it returns noisy labels; during
    testing, it returns original labels for proper evaluation.
    """

    def __init__(
        self,
        signals: np.ndarray,
        point_labels: np.ndarray,
        noisy_point_labels: np.ndarray,  # Labels with noise
        anomaly_regions: list,
        window_size: int,
        stride: int,
        mask_last_n: int = 0,
        split: str = 'train',
        train_ratio: float = 0.8,
        seed: int = 42,
        run_boundaries: list = None,
        normalize_mode: str = 'zscore',
    ):
        """Initialize noisy label dataset.

        Args:
            signals: Time series data (N, num_features)
            point_labels: Original ground truth labels (N,)
            noisy_point_labels: Labels with added noise (N,)
            anomaly_regions: List of AnomalyRegion objects
            window_size: Size of sliding window
            stride: Stride for sliding window
            mask_last_n: Number of last timesteps to mask
            split: 'train' or 'test'
            train_ratio: Proportion of data for training
            seed: Random seed
            run_boundaries: Positions where independent runs end
            normalize_mode: 'zscore' or 'minmax'
        """
        # Store noisy labels for training
        self.noisy_point_labels = noisy_point_labels
        self.use_noisy_labels = (split == 'train')

        super().__init__(
            signals=signals,
            point_labels=point_labels,  # Original labels for proper window classification
            anomaly_regions=anomaly_regions,
            window_size=window_size,
            stride=stride,
            mask_last_n=mask_last_n,
            split=split,
            train_ratio=train_ratio,
            seed=seed,
            run_boundaries=run_boundaries,
            normalize_mode=normalize_mode,
        )

    def __getitem__(self, idx):
        """Override to return noisy labels during training."""
        item = super().__getitem__(idx)

        if self.use_noisy_labels:
            # Replace point_labels with noisy version
            window_start = self.window_start_indices[idx]
            window_end = window_start + self.window_size
            noisy_window_labels = self.noisy_point_labels[window_start:window_end].astype(np.float32)

            # item = (signals, window_label, point_labels, sample_type, anomaly_type)
            return (item[0], item[1], noisy_window_labels, item[3], item[4])

        return item
