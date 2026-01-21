"""
Multivariate Time Series Dataset for Anomaly Detection
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional

# Anomaly type constants
ANOMALY_TYPE_NAMES = ['normal', 'spike', 'memory_leak', 'noise', 'drift', 'network_congestion']
ANOMALY_TYPES = {name: idx for idx, name in enumerate(ANOMALY_TYPE_NAMES)}


class MultivariateTimeSeriesDataset(Dataset):
    """
    Multivariate time series dataset simulating server monitoring data
    Features: CPU, Memory, Disk I/O, Network, Response Time

    Sample Types:
        0: pure_normal - completely normal sequence
        1: disturbing_normal - anomaly exists but NOT in last patch (label=0)
        2: anomaly - anomaly exists in last patch (label=1)
    """

    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        num_features: int,
        anomaly_ratio: float,
        is_train: bool = True,
        seed: Optional[int] = None,
        mask_last_n: int = 5,  # Number of time steps in last patch (for label computation)
        disturbing_ratio: float = 0.2  # Ratio of "disturbing normal" among normal samples
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_features = num_features
        self.anomaly_ratio = anomaly_ratio
        self.is_train = is_train
        self.mask_last_n = mask_last_n  # Used for computing last_patch_label
        self.disturbing_ratio = disturbing_ratio  # Ratio of normal samples with anomaly outside last patch

        if seed is not None:
            np.random.seed(seed)

        self.feature_names = ['CPU', 'Memory', 'DiskIO', 'Network', 'ResponseTime'][:num_features]
        self.data, self.seq_labels, self.point_labels, self.sample_types, self.anomaly_types = self._generate_data()

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

        Note: Spike is positioned to ALWAYS include the last patch (last mask_last_n time steps)
        """
        spike_width = np.random.randint(3, min(10, self.seq_length // 10 + 1))

        # Ensure spike overlaps with last patch:
        # spike covers [spike_pos, spike_pos + spike_width)
        # last patch covers [seq_length - mask_last_n, seq_length)
        # For overlap: spike_pos + spike_width > seq_length - mask_last_n
        # So: spike_pos > seq_length - mask_last_n - spike_width
        # Minimum start: seq_length - mask_last_n - spike_width + 1
        min_start = max(0, self.seq_length - self.mask_last_n - spike_width + 1)
        max_start = self.seq_length - spike_width  # spike ends at seq_length
        spike_pos = np.random.randint(min_start, max(min_start + 1, max_start + 1))

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

        Note: Memory leak continues to end of sequence, so always includes last patch
        """
        point_labels = np.zeros(self.seq_length, dtype=np.int64)

        if self.num_features < 2:
            # Even if no memory feature, mark last patch as anomaly for consistency
            point_labels[-self.mask_last_n:] = 1
            return signals, point_labels

        margin = min(20, self.seq_length // 5)
        start_pos = np.random.randint(margin, max(margin + 1, self.seq_length // 2))
        leak_length = self.seq_length - start_pos

        # Point-level labels (leak continues to end, so always includes last patch)
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

        Note: Noise region must include the last patch
        """
        min_length = min(20, self.seq_length // 4)
        max_length = min(40, self.seq_length // 2)

        noise_length = np.random.randint(min_length, max(min_length + 1, max_length))

        # Ensure noise overlaps with last patch:
        # noise covers [noise_start, noise_start + noise_length)
        # last patch covers [seq_length - mask_last_n, seq_length)
        # For overlap: noise_start + noise_length > seq_length - mask_last_n
        # Minimum start: seq_length - mask_last_n - noise_length + 1
        min_start = max(0, self.seq_length - self.mask_last_n - noise_length + 1)
        max_start = self.seq_length - noise_length
        noise_start = np.random.randint(min_start, max(min_start + 1, max_start + 1))

        # Adjust noise_length to not exceed sequence
        noise_end = min(noise_start + noise_length, self.seq_length)
        actual_noise_length = noise_end - noise_start

        # Point-level labels
        point_labels = np.zeros(self.seq_length, dtype=np.int64)
        point_labels[noise_start:noise_end] = 1

        # Add noise to all features
        for i in range(self.num_features):
            noise = np.random.normal(0, 0.2, actual_noise_length)
            signals[noise_start:noise_end, i] += noise

        return np.clip(signals, 0, 1), point_labels

    def _inject_multivariate_drift(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gradual drift in CPU and response time
        Returns: (modified_signals, point_labels)

        Note: Drift region must include the last patch
        """
        min_length = min(30, self.seq_length // 3)
        max_length = min(50, self.seq_length // 2)

        drift_length = np.random.randint(min_length, max(min_length + 1, max_length))

        # Ensure drift overlaps with last patch:
        # Minimum start: seq_length - mask_last_n - drift_length + 1
        min_start = max(0, self.seq_length - self.mask_last_n - drift_length + 1)
        max_start = self.seq_length - drift_length
        drift_start = np.random.randint(min_start, max(min_start + 1, max_start + 1))

        # Adjust drift_length to not exceed sequence
        drift_end = min(drift_start + drift_length, self.seq_length)
        actual_drift_length = drift_end - drift_start

        # Point-level labels
        point_labels = np.zeros(self.seq_length, dtype=np.int64)
        point_labels[drift_start:drift_end] = 1

        drift = np.linspace(0, np.random.uniform(0.2, 0.4), actual_drift_length)

        # CPU drift
        signals[drift_start:drift_end, 0] += drift
        # Response time drift
        if self.num_features > 4:
            signals[drift_start:drift_end, 4] += drift * 0.5

        return np.clip(signals, 0, 1), point_labels

    def _inject_multivariate_network_congestion(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Network congestion with response time spike
        Returns: (modified_signals, point_labels)

        Note: Congestion continues to end of sequence, so always includes last patch
        """
        point_labels = np.zeros(self.seq_length, dtype=np.int64)

        if self.num_features < 4:
            # Even if no network feature, mark last patch as anomaly for consistency
            point_labels[-self.mask_last_n:] = 1
            return signals, point_labels

        margin = min(30, self.seq_length // 3)
        change_point = np.random.randint(margin, max(margin + 1, self.seq_length - margin))

        # Point-level labels (congestion continues to end, so always includes last patch)
        point_labels[change_point:] = 1

        # High network traffic after change point
        signals[change_point:, 3] += np.random.uniform(0.3, 0.5)

        # Response time increases
        if self.num_features > 4:
            signals[change_point:, 4] += np.random.uniform(0.2, 0.4)

        return np.clip(signals, 0, 1), point_labels

    def _inject_anomaly_outside_last_patch(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inject anomaly ONLY in the region BEFORE the last patch.
        This creates a "disturbing normal" case where:
        - The sequence contains an anomaly
        - But the last patch (evaluation region) is normal
        - Used to test if anomalies in earlier parts affect reconstruction of last patch

        Returns: (modified_signals, point_labels)
        """
        # Available region: [0, seq_length - mask_last_n)
        available_length = self.seq_length - self.mask_last_n

        if available_length < 10:
            # Not enough space for anomaly
            return signals, np.zeros(self.seq_length, dtype=np.int64)

        # Randomly choose anomaly type
        anomaly_type = np.random.choice(['spike', 'noise', 'drift'])

        point_labels = np.zeros(self.seq_length, dtype=np.int64)

        if anomaly_type == 'spike':
            # Spike in early region
            spike_width = np.random.randint(3, min(10, available_length // 4 + 1))
            spike_pos = np.random.randint(5, available_length - spike_width)

            point_labels[spike_pos:spike_pos + spike_width] = 1

            # CPU spike
            signals[spike_pos:spike_pos + spike_width, 0] += np.random.uniform(0.3, 0.5)
            if self.num_features > 3:
                signals[spike_pos:spike_pos + spike_width, 3] += np.random.uniform(0.4, 0.6)
            if self.num_features > 4:
                signals[spike_pos:spike_pos + spike_width, 4] += np.random.uniform(0.3, 0.5)

        elif anomaly_type == 'noise':
            # Noise burst in early/middle region
            noise_length = np.random.randint(10, min(30, available_length // 2))
            noise_start = np.random.randint(5, available_length - noise_length)
            noise_end = noise_start + noise_length

            point_labels[noise_start:noise_end] = 1

            for i in range(self.num_features):
                noise = np.random.normal(0, 0.2, noise_length)
                signals[noise_start:noise_end, i] += noise

        else:  # drift
            # Drift that ends BEFORE last patch
            drift_length = np.random.randint(15, min(40, available_length // 2))
            drift_start = np.random.randint(5, available_length - drift_length)
            drift_end = drift_start + drift_length

            point_labels[drift_start:drift_end] = 1

            drift = np.linspace(0, np.random.uniform(0.2, 0.4), drift_length)
            signals[drift_start:drift_end, 0] += drift
            if self.num_features > 4:
                signals[drift_start:drift_end, 4] += drift * 0.5

        return np.clip(signals, 0, 1), point_labels

    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate complete multivariate dataset
        Returns: (data, sequence_labels, point_labels, sample_types, anomaly_types)
            data: (num_samples, seq_length, num_features)
            sequence_labels: (num_samples,) - binary sequence-level labels (based on last patch)
            point_labels: (num_samples, seq_length) - binary point-level labels
            sample_types: (num_samples,) - 0: pure_normal, 1: disturbing_normal, 2: anomaly
            anomaly_types: (num_samples,) - 0: normal, 1: spike, 2: memory_leak, 3: noise, 4: drift, 5: network_congestion
        """
        data = []
        seq_labels = []
        point_labels = []
        sample_types = []  # 0: pure_normal, 1: disturbing_normal, 2: anomaly
        anomaly_types = []  # 0: normal, 1-5: anomaly types

        num_anomalies = int(self.num_samples * self.anomaly_ratio)
        num_normal = self.num_samples - num_anomalies

        # Split normal samples into pure normal and disturbing normal
        num_disturbing = int(num_normal * self.disturbing_ratio)
        num_pure_normal = num_normal - num_disturbing

        # Generate pure normal sequences (completely normal)
        for _ in range(num_pure_normal):
            signals = self._generate_normal_multivariate()
            data.append(signals)
            seq_labels.append(0)  # last patch is normal
            point_labels.append(np.zeros(self.seq_length, dtype=np.int64))
            sample_types.append(0)  # pure_normal
            anomaly_types.append(ANOMALY_TYPES['normal'])  # 0

        # Generate disturbing normal sequences (anomaly outside last patch)
        for _ in range(num_disturbing):
            signals = self._generate_normal_multivariate()
            signals, point_label = self._inject_anomaly_outside_last_patch(signals)
            data.append(signals)
            seq_labels.append(0)  # last patch is still normal!
            point_labels.append(point_label)
            sample_types.append(1)  # disturbing_normal
            anomaly_types.append(ANOMALY_TYPES['normal'])  # 0 (last patch is normal)

        # Generate anomalous sequences (anomaly includes last patch)
        # Map function to anomaly type
        anomaly_funcs_with_types = [
            (self._inject_multivariate_spike, ANOMALY_TYPES['spike']),
            (self._inject_multivariate_memory_leak, ANOMALY_TYPES['memory_leak']),
            (self._inject_multivariate_noise, ANOMALY_TYPES['noise']),
            (self._inject_multivariate_drift, ANOMALY_TYPES['drift']),
            (self._inject_multivariate_network_congestion, ANOMALY_TYPES['network_congestion'])
        ]

        for _ in range(num_anomalies):
            signals = self._generate_normal_multivariate()
            func_idx = np.random.randint(0, len(anomaly_funcs_with_types))
            anomaly_func, anomaly_type = anomaly_funcs_with_types[func_idx]
            signals, point_label = anomaly_func(signals)
            data.append(signals)
            seq_labels.append(1)  # last patch has anomaly
            point_labels.append(point_label)
            sample_types.append(2)  # anomaly
            anomaly_types.append(anomaly_type)

        # Shuffle
        data = np.array(data, dtype=np.float32)
        seq_labels = np.array(seq_labels, dtype=np.int64)
        point_labels = np.array(point_labels, dtype=np.int64)
        sample_types = np.array(sample_types, dtype=np.int64)
        anomaly_types = np.array(anomaly_types, dtype=np.int64)

        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        data = data[indices]
        seq_labels = seq_labels[indices]
        point_labels = point_labels[indices]
        sample_types = sample_types[indices]
        anomaly_types = anomaly_types[indices]

        return data, seq_labels, point_labels, sample_types, anomaly_types

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (sequence, last_patch_label, point_labels, sample_type, anomaly_type)
        sequence: (seq_length, num_features)
        last_patch_label: scalar - 1 if last patch contains anomaly, 0 otherwise
        point_labels: (seq_length,) - point-level labels
        sample_type: scalar - 0: pure_normal, 1: disturbing_normal, 2: anomaly
        anomaly_type: scalar - 0: normal, 1: spike, 2: memory_leak, 3: noise, 4: drift, 5: network_congestion
        """
        sequence = torch.from_numpy(self.data[idx])  # (seq_length, num_features)
        point_label = torch.from_numpy(self.point_labels[idx])  # (seq_length,)
        sample_type = torch.tensor(self.sample_types[idx], dtype=torch.long)
        anomaly_type = torch.tensor(self.anomaly_types[idx], dtype=torch.long)

        # Compute label based on whether last patch contains anomaly
        # Last patch = last mask_last_n time steps
        last_patch_label = torch.tensor(
            1 if point_label[-self.mask_last_n:].sum() > 0 else 0,
            dtype=torch.long
        )
        return sequence, last_patch_label, point_label, sample_type, anomaly_type
