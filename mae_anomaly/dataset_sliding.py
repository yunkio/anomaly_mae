"""
Sliding Window Time Series Dataset for Anomaly Detection

This module generates a long continuous time series with anomalies injected,
then extracts samples using sliding windows. This approach:
1. Creates more realistic temporal continuity
2. Ensures patch-size-independent data generation
3. Naturally creates disturbing normal samples (anomaly before last patch)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


# Feature names (8 features for server monitoring)
FEATURE_NAMES = [
    'CPU',           # 0: Base CPU usage
    'Memory',        # 1: Memory usage (correlated with CPU)
    'DiskIO',        # 2: Disk I/O (correlated with Memory, spiky)
    'Network',       # 3: Network traffic (bursty)
    'ResponseTime',  # 4: Response time (correlated with CPU, Network)
    'ThreadCount',   # 5: Active threads (correlated with CPU)
    'ErrorRate',     # 6: Error rate (correlated with ResponseTime)
    'QueueLength',   # 7: Request queue (correlated with CPU, ThreadCount)
]

# Anomaly type constants (7 types + normal)
ANOMALY_TYPE_NAMES = [
    'normal',              # 0
    'spike',               # 1: Traffic spike / DDoS
    'memory_leak',         # 2: Gradual memory increase
    'cpu_saturation',      # 3: CPU stuck at high level
    'network_congestion',  # 4: Network bottleneck
    'cascading_failure',   # 5: Error propagation
    'resource_contention', # 6: Thread/queue competition
    'point_spike',         # 7: Point anomaly (3-5 timesteps, 2+ features)
]
ANOMALY_TYPES = {name: idx for idx, name in enumerate(ANOMALY_TYPE_NAMES)}
NUM_FEATURES = len(FEATURE_NAMES)
NUM_ANOMALY_TYPES = len(ANOMALY_TYPE_NAMES) - 1  # Exclude 'normal'

# Per-anomaly-type configuration: (length_min, length_max, interval_mean)
# Designed based on realistic characteristics of each anomaly type
ANOMALY_TYPE_CONFIGS = {
    # spike: Short and sudden (DDoS, traffic burst)
    1: {'length_range': (10, 25), 'interval_mean': 3500},
    # memory_leak: Long and gradual (slow accumulation)
    2: {'length_range': (80, 150), 'interval_mean': 7000},
    # cpu_saturation: Medium duration (sustained high load)
    3: {'length_range': (40, 80), 'interval_mean': 4500},
    # network_congestion: Medium duration
    4: {'length_range': (30, 60), 'interval_mean': 4000},
    # cascading_failure: Long and propagating (chain reaction)
    5: {'length_range': (60, 120), 'interval_mean': 6500},
    # resource_contention: Medium with oscillation
    6: {'length_range': (35, 65), 'interval_mean': 4500},
    # point_spike: Very short point anomaly (3-5 timesteps)
    7: {'length_range': (3, 5), 'interval_mean': 4000},
}


@dataclass
class AnomalyRegion:
    """Represents an anomaly region in the time series"""
    start: int
    end: int
    anomaly_type: int  # 1-7 (not 0, which is normal)


class SlidingWindowTimeSeriesGenerator:
    """
    Generates a long continuous time series with anomalies injected.

    The time series simulates server monitoring data with 8 correlated features.
    Anomalies are injected at random intervals throughout the series.
    """

    def __init__(
        self,
        total_length: int,
        num_features: int = 8,
        anomaly_type_configs: Optional[Dict] = None,  # Per-type configs
        interval_scale: float = 1.0,  # Scale factor for all intervals (for tuning)
        seed: Optional[int] = None
    ):
        self.total_length = total_length
        self.num_features = min(num_features, NUM_FEATURES)
        self.interval_scale = interval_scale
        self.seed = seed

        # Use default configs or provided ones
        self.anomaly_type_configs = anomaly_type_configs or ANOMALY_TYPE_CONFIGS

        if seed is not None:
            np.random.seed(seed)

    def _generate_normal_series(self) -> np.ndarray:
        """
        Generate a long normal time series with 8 correlated features.
        Uses consistent parameters throughout (no segments).
        """
        t = np.linspace(0, self.total_length * 0.1, self.total_length)
        signals = np.zeros((self.total_length, self.num_features), dtype=np.float32)

        # Base parameters (consistent throughout)
        cpu_freq = np.random.uniform(0.5, 1.5)
        cpu_amp = np.random.uniform(0.2, 0.4)
        cpu_base = np.random.uniform(0.3, 0.5)

        # Feature 0: CPU usage (base pattern with daily/weekly cycles)
        signals[:, 0] = (
            cpu_base +
            cpu_amp * np.sin(cpu_freq * t) +
            0.1 * np.sin(cpu_freq * 0.3 * t) +  # Slower cycle
            np.random.normal(0, 0.03, self.total_length)
        )

        # Feature 1: Memory usage (correlated with CPU, slower variation)
        if self.num_features > 1:
            mem_base = np.random.uniform(0.4, 0.6)
            signals[:, 1] = (
                mem_base +
                0.25 * signals[:, 0] +
                0.15 * np.sin(cpu_freq * 0.4 * t) +
                np.random.normal(0, 0.02, self.total_length)
            )

        # Feature 2: Disk I/O (spiky, correlated with memory)
        if self.num_features > 2:
            disk_base = np.random.uniform(0.1, 0.25)
            disk_spikes = np.random.poisson(0.05, self.total_length) * 0.15
            signals[:, 2] = (
                disk_base +
                0.15 * signals[:, 1] +
                disk_spikes +
                np.random.normal(0, 0.03, self.total_length)
            )

        # Feature 3: Network traffic (bursty pattern)
        if self.num_features > 3:
            net_freq = np.random.uniform(0.8, 1.5)
            net_amp = np.random.uniform(0.15, 0.35)
            signals[:, 3] = (
                net_amp * np.abs(np.sin(net_freq * t)) +
                0.1 * np.random.exponential(0.1, self.total_length) +
                np.random.normal(0, 0.03, self.total_length)
            )

        # Feature 4: Response time (correlated with CPU and Network)
        if self.num_features > 4:
            resp_base = np.random.uniform(0.1, 0.25)
            signals[:, 4] = (
                resp_base +
                0.25 * signals[:, 0] +
                0.15 * signals[:, 3] +
                np.random.normal(0, 0.03, self.total_length)
            )

        # Feature 5: Thread count (correlated with CPU, smoother)
        if self.num_features > 5:
            thread_base = np.random.uniform(0.3, 0.5)
            # Smooth version of CPU
            from scipy.ndimage import gaussian_filter1d
            cpu_smooth = gaussian_filter1d(signals[:, 0], sigma=10)
            signals[:, 5] = (
                thread_base +
                0.3 * cpu_smooth +
                np.random.normal(0, 0.02, self.total_length)
            )

        # Feature 6: Error rate (low base, correlated with response time)
        if self.num_features > 6:
            error_base = np.random.uniform(0.02, 0.08)
            signals[:, 6] = (
                error_base +
                0.1 * np.maximum(0, signals[:, 4] - 0.3) +  # Errors increase with high response time
                np.random.exponential(0.02, self.total_length)
            )

        # Feature 7: Queue length (correlated with CPU and threads)
        if self.num_features > 7:
            queue_base = np.random.uniform(0.1, 0.25)
            signals[:, 7] = (
                queue_base +
                0.2 * signals[:, 0] +
                0.15 * signals[:, 5] +
                np.random.normal(0, 0.03, self.total_length)
            )

        # Clip to [0, 1] range
        signals = np.clip(signals, 0, 1)
        return signals

    def _inject_spike(self, signals: np.ndarray, start: int, end: int) -> None:
        """Inject traffic spike / DDoS anomaly"""
        length = end - start

        # CPU spike
        signals[start:end, 0] += np.random.uniform(0.3, 0.5)

        # Network spike
        if self.num_features > 3:
            signals[start:end, 3] += np.random.uniform(0.4, 0.6)

        # Response time spike
        if self.num_features > 4:
            signals[start:end, 4] += np.random.uniform(0.3, 0.5)

        # Error rate spike
        if self.num_features > 6:
            signals[start:end, 6] += np.random.uniform(0.2, 0.4)

        # Queue spike
        if self.num_features > 7:
            signals[start:end, 7] += np.random.uniform(0.3, 0.5)

    def _inject_memory_leak(self, signals: np.ndarray, start: int, end: int) -> None:
        """Inject memory leak anomaly (gradual increase)"""
        length = end - start
        leak_curve = np.linspace(0, 1, length) ** 0.7  # Gradual curve

        # Memory gradual increase
        if self.num_features > 1:
            signals[start:end, 1] += leak_curve * np.random.uniform(0.3, 0.5)

        # Disk I/O increases (swapping)
        if self.num_features > 2:
            signals[start:end, 2] += leak_curve * np.random.uniform(0.2, 0.4)

        # Thread count affected
        if self.num_features > 5:
            signals[start:end, 5] += leak_curve * np.random.uniform(0.1, 0.3)

    def _inject_cpu_saturation(self, signals: np.ndarray, start: int, end: int) -> None:
        """Inject CPU saturation anomaly"""
        length = end - start

        # CPU stuck at high level
        signals[start:end, 0] = np.clip(
            signals[start:end, 0] + np.random.uniform(0.4, 0.6),
            0.7, 1.0  # Saturated
        )

        # Thread count high
        if self.num_features > 5:
            signals[start:end, 5] += np.random.uniform(0.3, 0.5)

        # Queue builds up
        if self.num_features > 7:
            buildup = np.linspace(0, 1, length)
            signals[start:end, 7] += buildup * np.random.uniform(0.3, 0.5)

        # Response time degrades
        if self.num_features > 4:
            signals[start:end, 4] += np.random.uniform(0.2, 0.4)

    def _inject_network_congestion(self, signals: np.ndarray, start: int, end: int) -> None:
        """Inject network congestion anomaly"""
        length = end - start

        # Network saturated
        if self.num_features > 3:
            signals[start:end, 3] += np.random.uniform(0.4, 0.6)

        # Response time high
        if self.num_features > 4:
            signals[start:end, 4] += np.random.uniform(0.3, 0.5)

        # Error rate increases
        if self.num_features > 6:
            signals[start:end, 6] += np.random.uniform(0.15, 0.35)

        # Queue increases
        if self.num_features > 7:
            signals[start:end, 7] += np.random.uniform(0.2, 0.4)

    def _inject_cascading_failure(self, signals: np.ndarray, start: int, end: int) -> None:
        """Inject cascading failure anomaly (propagating errors)"""
        length = end - start
        cascade = np.linspace(0, 1, length) ** 0.5  # Fast initial increase

        # Error rate spikes first
        if self.num_features > 6:
            signals[start:end, 6] += cascade * np.random.uniform(0.4, 0.6)

        # Response time follows
        if self.num_features > 4:
            delayed_cascade = np.roll(cascade, length // 4)
            delayed_cascade[:length // 4] = 0
            signals[start:end, 4] += delayed_cascade * np.random.uniform(0.3, 0.5)

        # Queue builds up
        if self.num_features > 7:
            delayed_queue = np.roll(cascade, length // 3)
            delayed_queue[:length // 3] = 0
            signals[start:end, 7] += delayed_queue * np.random.uniform(0.4, 0.6)

        # CPU increases due to retries
        signals[start:end, 0] += cascade * np.random.uniform(0.2, 0.4)

    def _inject_resource_contention(self, signals: np.ndarray, start: int, end: int) -> None:
        """Inject resource contention anomaly (threads competing)"""
        length = end - start

        # Oscillating pattern (contention)
        oscillation = 0.5 + 0.5 * np.sin(np.linspace(0, 6 * np.pi, length))

        # Thread count spikes
        if self.num_features > 5:
            signals[start:end, 5] += oscillation * np.random.uniform(0.3, 0.5)

        # Queue oscillates
        if self.num_features > 7:
            signals[start:end, 7] += oscillation * np.random.uniform(0.3, 0.5)

        # CPU contention
        signals[start:end, 0] += oscillation * np.random.uniform(0.2, 0.4)

        # Memory pressure
        if self.num_features > 1:
            signals[start:end, 1] += np.random.uniform(0.15, 0.3)

    def _inject_point_spike(self, signals: np.ndarray, start: int, end: int) -> None:
        """Inject point spike anomaly (3-5 timesteps, 2+ random features)

        This is a true point anomaly - very short duration with random features spiking.
        Unlike other anomalies, which features spike is random (but at least 2).
        """
        # Select 2 or more random features to spike
        num_features_to_spike = np.random.randint(2, self.num_features + 1)
        features_to_spike = np.random.choice(
            self.num_features,
            size=num_features_to_spike,
            replace=False
        )

        # Apply spike to selected features
        for feat_idx in features_to_spike:
            spike_magnitude = np.random.uniform(0.3, 0.6)
            signals[start:end, feat_idx] += spike_magnitude

    def _distribute_anomalies(self) -> List[AnomalyRegion]:
        """
        Distribute anomalies throughout the time series.
        Each anomaly type has its own length range and interval.
        All types are distributed across the entire time series.
        Returns list of AnomalyRegion objects sorted by start position.
        """
        all_anomalies = []

        # Generate anomalies for each type independently
        for anomaly_type in range(1, NUM_ANOMALY_TYPES + 1):
            config = self.anomaly_type_configs[anomaly_type]
            length_range = config['length_range']
            interval_mean = config['interval_mean'] * self.interval_scale

            # Start at random position
            current_pos = np.random.randint(50, 200)

            while current_pos < self.total_length - length_range[1]:
                # Random anomaly length for this type
                length = np.random.randint(length_range[0], length_range[1] + 1)

                all_anomalies.append(AnomalyRegion(
                    start=current_pos,
                    end=current_pos + length,
                    anomaly_type=anomaly_type
                ))

                # Move to next position with type-specific interval
                interval = int(np.random.exponential(interval_mean))
                interval = max(interval, length_range[1] + 50)  # Minimum gap
                current_pos += length + interval

        # Sort by start position
        all_anomalies.sort(key=lambda x: x.start)

        # Remove overlapping anomalies (keep earlier one)
        non_overlapping = []
        last_end = 0
        for anomaly in all_anomalies:
            if anomaly.start >= last_end:
                non_overlapping.append(anomaly)
                last_end = anomaly.end

        return non_overlapping

    def generate(self) -> Tuple[np.ndarray, np.ndarray, List[AnomalyRegion]]:
        """
        Generate the complete time series with anomalies.

        Returns:
            signals: (total_length, num_features) normalized time series
            point_labels: (total_length,) binary labels (1 = anomaly point)
            anomaly_regions: List of AnomalyRegion objects
        """
        # Generate normal base series
        signals = self._generate_normal_series()

        # Initialize point labels
        point_labels = np.zeros(self.total_length, dtype=np.int64)

        # Distribute and inject anomalies
        anomaly_regions = self._distribute_anomalies()

        inject_funcs = {
            1: self._inject_spike,
            2: self._inject_memory_leak,
            3: self._inject_cpu_saturation,
            4: self._inject_network_congestion,
            5: self._inject_cascading_failure,
            6: self._inject_resource_contention,
            7: self._inject_point_spike,
        }

        for region in anomaly_regions:
            inject_funcs[region.anomaly_type](signals, region.start, region.end)
            point_labels[region.start:region.end] = 1

        # Clip to [0, 1]
        signals = np.clip(signals, 0, 1).astype(np.float32)

        return signals, point_labels, anomaly_regions


class SlidingWindowDataset(Dataset):
    """
    Dataset that extracts sliding windows from a long time series.

    Sample Types:
        0: pure_normal - no anomaly in the window
        1: disturbing_normal - anomaly exists but NOT in last mask_last_n (label=0)
        2: anomaly - anomaly exists in last mask_last_n (label=1)
    """

    def __init__(
        self,
        signals: np.ndarray,
        point_labels: np.ndarray,
        anomaly_regions: List[AnomalyRegion],
        window_size: int,
        stride: int,
        mask_last_n: int,
        split: str,  # 'train' or 'test'
        train_ratio: float = 0.5,
        target_anomaly_ratio: Optional[float] = None,  # For test set downsampling (legacy)
        target_counts: Optional[Dict[str, int]] = None,  # Explicit counts per sample type
        seed: Optional[int] = None
    ):
        self.window_size = window_size
        self.stride = stride
        self.mask_last_n = mask_last_n
        self.split = split
        self.target_counts = target_counts

        if seed is not None:
            np.random.seed(seed)

        # Split the time series
        total_length = len(signals)
        train_end = int(total_length * train_ratio)

        # Ensure clean split (no window crosses boundary)
        train_end = (train_end // stride) * stride

        if split == 'train':
            self.signals = signals[:train_end]
            self.point_labels = point_labels[:train_end]
            offset = 0
        else:  # test
            self.signals = signals[train_end:]
            self.point_labels = point_labels[train_end:]
            offset = train_end

        # Filter anomaly regions for this split
        self.anomaly_regions = []
        for region in anomaly_regions:
            if split == 'train':
                if region.end <= train_end:
                    self.anomaly_regions.append(region)
            else:
                if region.start >= train_end:
                    self.anomaly_regions.append(AnomalyRegion(
                        start=region.start - offset,
                        end=region.end - offset,
                        anomaly_type=region.anomaly_type
                    ))

        # Extract windows
        self._extract_windows()

        # Downsample for target ratio/counts if specified (for test set)
        if split == 'test' and (target_counts is not None or target_anomaly_ratio is not None):
            self._downsample_for_ratio(
                target_counts=target_counts,
                target_anomaly_ratio=target_anomaly_ratio
            )

    def _extract_windows(self):
        """Extract all windows and compute labels"""
        self.windows = []
        self.seq_labels = []  # 0 or 1 (based on last mask_last_n)
        self.window_point_labels = []
        self.sample_types = []  # 0: pure_normal, 1: disturbing_normal, 2: anomaly
        self.anomaly_type_labels = []  # Which anomaly type (0 for normal)

        series_length = len(self.signals)

        for start in range(0, series_length - self.window_size + 1, self.stride):
            end = start + self.window_size

            window = self.signals[start:end]
            window_pl = self.point_labels[start:end]

            # Check last mask_last_n region
            last_region_start = end - self.mask_last_n
            has_anomaly_in_last = window_pl[-self.mask_last_n:].sum() > 0
            has_anomaly_in_window = window_pl.sum() > 0

            # Determine sample type and label
            if has_anomaly_in_last:
                seq_label = 1
                sample_type = 2  # anomaly
            elif has_anomaly_in_window:
                seq_label = 0
                sample_type = 1  # disturbing_normal
            else:
                seq_label = 0
                sample_type = 0  # pure_normal

            # Find anomaly type for this window
            anomaly_type = 0  # normal
            for region in self.anomaly_regions:
                # Check if this region overlaps with window's last region
                if has_anomaly_in_last:
                    region_in_window_start = max(region.start, last_region_start)
                    region_in_window_end = min(region.end, end)
                    if region_in_window_start < region_in_window_end:
                        anomaly_type = region.anomaly_type
                        break

            self.windows.append(window)
            self.seq_labels.append(seq_label)
            self.window_point_labels.append(window_pl)
            self.sample_types.append(sample_type)
            self.anomaly_type_labels.append(anomaly_type)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.seq_labels = np.array(self.seq_labels, dtype=np.int64)
        self.window_point_labels = np.array(self.window_point_labels, dtype=np.int64)
        self.sample_types = np.array(self.sample_types, dtype=np.int64)
        self.anomaly_type_labels = np.array(self.anomaly_type_labels, dtype=np.int64)

    def _downsample_for_ratio(
        self,
        target_counts: Optional[Dict[str, int]] = None,
        target_anomaly_ratio: Optional[float] = None
    ):
        """
        Downsample to achieve target sample counts or ratio.

        Args:
            target_counts: Dict with keys 'pure_normal', 'disturbing_normal', 'anomaly'
                          e.g., {'pure_normal': 1200, 'disturbing_normal': 300, 'anomaly': 500}
            target_anomaly_ratio: Alternative - just specify anomaly ratio (legacy)
        """
        # Get current counts
        pure_indices = np.where(self.sample_types == 0)[0]
        disturb_indices = np.where(self.sample_types == 1)[0]
        anomaly_indices = np.where(self.sample_types == 2)[0]

        if target_counts is not None:
            # Use explicit target counts
            target_pure = target_counts.get('pure_normal', len(pure_indices))
            target_disturb = target_counts.get('disturbing_normal', len(disturb_indices))
            target_anomaly = target_counts.get('anomaly', len(anomaly_indices))

            # Check if we have enough samples
            if len(anomaly_indices) < target_anomaly:
                print(f"Warning: Not enough anomaly samples ({len(anomaly_indices)} < {target_anomaly})")
                target_anomaly = len(anomaly_indices)
            if len(disturb_indices) < target_disturb:
                print(f"Warning: Not enough disturbing samples ({len(disturb_indices)} < {target_disturb})")
                target_disturb = len(disturb_indices)
            if len(pure_indices) < target_pure:
                print(f"Warning: Not enough pure normal samples ({len(pure_indices)} < {target_pure})")
                target_pure = len(pure_indices)

            # Sample from each category
            keep_pure = np.random.choice(pure_indices, size=target_pure, replace=False)
            keep_disturb = np.random.choice(disturb_indices, size=target_disturb, replace=False)
            keep_anomaly = np.random.choice(anomaly_indices, size=target_anomaly, replace=False)

            keep_indices = np.sort(np.concatenate([keep_pure, keep_disturb, keep_anomaly]))

        elif target_anomaly_ratio is not None:
            # Legacy: use anomaly ratio
            num_anomaly = len(anomaly_indices)
            if num_anomaly == 0:
                return

            target_num_normal = int(num_anomaly * (1 - target_anomaly_ratio) / target_anomaly_ratio)
            normal_indices = np.where(self.sample_types != 2)[0]

            if target_num_normal >= len(normal_indices):
                return

            keep_normal = np.random.choice(normal_indices, size=target_num_normal, replace=False)
            keep_indices = np.sort(np.concatenate([anomaly_indices, keep_normal]))
        else:
            return

        # Apply downsampling
        self.windows = self.windows[keep_indices]
        self.seq_labels = self.seq_labels[keep_indices]
        self.window_point_labels = self.window_point_labels[keep_indices]
        self.sample_types = self.sample_types[keep_indices]
        self.anomaly_type_labels = self.anomaly_type_labels[keep_indices]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (sequence, label, point_labels, sample_type, anomaly_type)
        """
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.seq_labels[idx], dtype=torch.long),
            torch.from_numpy(self.window_point_labels[idx]),
            torch.tensor(self.sample_types[idx], dtype=torch.long),
            torch.tensor(self.anomaly_type_labels[idx], dtype=torch.long),
        )

    def get_ratios(self) -> Dict[str, float]:
        """Get sample type ratios"""
        total = len(self.sample_types)
        if total == 0:
            return {'pure_normal': 0, 'disturbing_normal': 0, 'anomaly': 0}

        pure = (self.sample_types == 0).sum() / total
        disturbing = (self.sample_types == 1).sum() / total
        anomaly = (self.sample_types == 2).sum() / total

        return {
            'pure_normal': pure,
            'disturbing_normal': disturbing,
            'anomaly': anomaly,
            'total_samples': total
        }

    def get_anomaly_type_distribution(self) -> Dict[str, int]:
        """Get count of each anomaly type"""
        dist = {}
        for name in ANOMALY_TYPE_NAMES:
            idx = ANOMALY_TYPES[name]
            dist[name] = (self.anomaly_type_labels == idx).sum()
        return dist


def analyze_and_tune_parameters(
    target_train_ratios: Dict[str, float],
    window_size: int = 100,
    mask_last_n: int = 10,
    initial_interval: float = 300.0,
    max_iterations: int = 10,
    tolerance: float = 0.02,
    seed: int = 42
) -> Dict:
    """
    Iteratively tune anomaly_interval to match target ratios.

    Args:
        target_train_ratios: Target ratios for train set
            e.g., {'pure_normal': 0.72, 'disturbing_normal': 0.18, 'anomaly': 0.10}
        window_size: Size of sliding window
        mask_last_n: Last n timesteps for anomaly labeling
        initial_interval: Starting anomaly interval
        max_iterations: Maximum tuning iterations
        tolerance: Acceptable deviation from target
        seed: Random seed

    Returns:
        Dict with final parameters and achieved ratios
    """
    # Calculate total length based on target sample counts
    # Assume ~5000 train samples, ~1500 test samples
    total_samples = 6500
    total_length = total_samples * window_size * 2  # 2x for margin

    current_interval = initial_interval
    best_interval = initial_interval
    best_error = float('inf')

    results = []

    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1} ===")
        print(f"Testing anomaly_interval = {current_interval:.1f}")

        # Generate data
        generator = SlidingWindowTimeSeriesGenerator(
            total_length=total_length,
            num_features=8,
            anomaly_interval_mean=current_interval,
            anomaly_length_range=(30, 80),
            seed=seed + iteration
        )

        signals, point_labels, anomaly_regions = generator.generate()

        # Create train dataset
        train_dataset = SlidingWindowDataset(
            signals=signals,
            point_labels=point_labels,
            anomaly_regions=anomaly_regions,
            window_size=window_size,
            stride=window_size,  # No overlap
            mask_last_n=mask_last_n,
            split='train',
            train_ratio=0.5,
            seed=seed
        )

        ratios = train_dataset.get_ratios()
        print(f"  Pure Normal: {ratios['pure_normal']:.1%}")
        print(f"  Disturbing Normal: {ratios['disturbing_normal']:.1%}")
        print(f"  Anomaly: {ratios['anomaly']:.1%}")
        print(f"  Total samples: {ratios['total_samples']}")

        # Calculate error
        error = (
            abs(ratios['pure_normal'] - target_train_ratios['pure_normal']) +
            abs(ratios['disturbing_normal'] - target_train_ratios['disturbing_normal']) +
            abs(ratios['anomaly'] - target_train_ratios['anomaly'])
        )

        results.append({
            'interval': current_interval,
            'ratios': ratios,
            'error': error
        })

        if error < best_error:
            best_error = error
            best_interval = current_interval

        if error < tolerance:
            print(f"\nTarget achieved! Error = {error:.4f}")
            break

        # Adjust interval based on anomaly ratio
        actual_anomaly = ratios['anomaly']
        target_anomaly = target_train_ratios['anomaly']

        if actual_anomaly > target_anomaly:
            # Too many anomalies, increase interval
            adjustment = 1 + (actual_anomaly - target_anomaly) * 3
            current_interval *= adjustment
        else:
            # Too few anomalies, decrease interval
            adjustment = 1 - (target_anomaly - actual_anomaly) * 3
            current_interval *= max(0.5, adjustment)

        current_interval = max(100, min(1000, current_interval))

    return {
        'best_interval': best_interval,
        'best_error': best_error,
        'all_results': results
    }


if __name__ == '__main__':
    # Test the dataset generation
    print("=" * 60)
    print("Testing Sliding Window Dataset Generation")
    print("=" * 60)

    # Target ratios (from current config)
    target_train = {
        'pure_normal': 0.72,  # 80% normal * 90% pure
        'disturbing_normal': 0.18,  # 80% normal * 10% disturbing (adjusted)
        'anomaly': 0.10
    }

    print("\nTarget Train Ratios:")
    print(f"  Pure Normal: {target_train['pure_normal']:.1%}")
    print(f"  Disturbing Normal: {target_train['disturbing_normal']:.1%}")
    print(f"  Anomaly: {target_train['anomaly']:.1%}")

    # Tune parameters
    result = analyze_and_tune_parameters(
        target_train_ratios=target_train,
        window_size=100,
        mask_last_n=10,
        initial_interval=300.0,
        max_iterations=10,
        seed=42
    )

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Best anomaly_interval: {result['best_interval']:.1f}")
    print(f"Best error: {result['best_error']:.4f}")
