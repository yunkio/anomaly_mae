"""
Multivariate Time Series Dataset for Anomaly Detection
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


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
