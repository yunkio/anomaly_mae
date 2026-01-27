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
from scipy.ndimage import gaussian_filter1d


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

# Anomaly type constants (9 types + normal)
# Types 1-6: Value-based anomalies (values deviate from normal range)
# Types 7-9: Pattern-based anomalies (values within normal range, patterns differ)
ANOMALY_TYPE_NAMES = [
    'normal',              # 0
    # Value-based anomalies (type 1-6)
    'spike',               # 1: Traffic spike / DDoS
    'memory_leak',         # 2: Gradual memory increase
    'cpu_saturation',      # 3: CPU stuck at high level
    'network_congestion',  # 4: Network bottleneck
    'cascading_failure',   # 5: Error propagation
    'resource_contention', # 6: Thread/queue competition
    # Pattern-based anomalies (type 7-9) - values in normal range, patterns differ
    'correlation_inversion',  # 7: CPU-Memory correlation breaks (inverse)
    'temporal_flatline',      # 8: Values freeze (stuck sensor)
    'frequency_shift',        # 9: Unusual periodicity in values
]
ANOMALY_TYPES = {name: idx for idx, name in enumerate(ANOMALY_TYPE_NAMES)}
NUM_FEATURES = len(FEATURE_NAMES)
NUM_ANOMALY_TYPES = len(ANOMALY_TYPE_NAMES) - 1  # Exclude 'normal'

# Anomaly category: 'value' (value range differs) or 'pattern' (only pattern differs)
# This allows comparing model's ability to detect value vs pattern anomalies
ANOMALY_CATEGORY = {
    1: 'value', 2: 'value', 3: 'value', 4: 'value',
    5: 'value', 6: 'value',
    7: 'pattern', 8: 'pattern', 9: 'pattern'
}

# Per-anomaly-type configuration: (length_min, length_max, interval_mean)
# Designed based on realistic characteristics of each anomaly type
ANOMALY_TYPE_CONFIGS = {
    # === Value-based anomalies (types 1-6) ===
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
    # === Pattern-based anomalies (types 7-9) ===
    # correlation_inversion: Medium duration to show pattern change
    7: {'length_range': (50, 100), 'interval_mean': 5000},
    # temporal_flatline: Sudden freeze, medium duration
    8: {'length_range': (30, 60), 'interval_mean': 4500},
    # frequency_shift: Needs enough length to show frequency change
    9: {'length_range': (60, 100), 'interval_mean': 5500},
}


def _normalize_per_feature(signals: np.ndarray) -> np.ndarray:
    """Per-feature min-max normalization to [0, 1] range.

    This is preferred over clipping because:
    1. Preserves relative magnitude of anomalies (spikes won't be capped)
    2. No artificial saturation at boundaries
    3. More realistic simulation of real-world data preprocessing

    Args:
        signals: (total_length, num_features) array

    Returns:
        Normalized signals with each feature scaled to [0, 1]
    """
    signals = signals.copy()  # Don't modify original
    for f in range(signals.shape[1]):
        min_val = signals[:, f].min()
        max_val = signals[:, f].max()
        if max_val - min_val > 1e-8:  # Avoid division by zero
            signals[:, f] = (signals[:, f] - min_val) / (max_val - min_val)
        else:
            signals[:, f] = 0.5  # Constant signal -> set to middle
    return signals.astype(np.float32)


@dataclass
class AnomalyRegion:
    """Represents an anomaly region in the time series"""
    start: int
    end: int
    anomaly_type: int  # 1-7 (not 0, which is normal)


@dataclass
class NormalDataComplexity:
    """Configuration for normal data complexity features.

    All features are designed to add realistic variation to normal data
    WITHOUT being confused with anomaly patterns. Key safety constraints:
    - All transitions >= 1000 timesteps (gradual changes)
    - Values stay in [0.05, 0.70] range (below anomaly thresholds)
    - Bump magnitudes << anomaly magnitudes
    """
    # Master switch
    enable_complexity: bool = True

    # 1. Regime Switching - Different operational states
    enable_regime_switching: bool = True
    regime_duration_range: tuple = (8000, 25000)  # Duration of each regime
    regime_transition_length: int = 1500  # Smooth transition over 1500 timesteps

    # 2. Multi-Scale Periodicity - Multiple overlapping cycles
    enable_multi_scale_periodicity: bool = True
    # freq1: fast (hourly-like), freq2: medium (daily-like), freq3: slow (weekly-like)

    # 3. Heteroscedastic Noise - Load-dependent noise variance
    enable_heteroscedastic_noise: bool = True
    base_noise: float = 0.025
    noise_load_sensitivity: float = 0.8  # noise = base * (1 + sensitivity * load)

    # 4. Time-Varying Correlations - Feature correlations that slowly change
    enable_varying_correlations: bool = True
    correlation_variation_period: int = 15000  # Period of correlation change
    correlation_variation_amplitude: float = 0.08  # ±0.08 variation in correlation

    # 5. Bounded Drift (Ornstein-Uhlenbeck) - Mean-reverting random walk
    enable_drift: bool = True
    drift_theta: float = 0.002  # Mean reversion speed (higher = faster reversion)
    drift_sigma: float = 0.025  # Volatility
    drift_max: float = 0.08  # Maximum drift from base (clipped)

    # 6. Normal Bumps - Small, gradual load increases (NOT anomalies)
    enable_normal_bumps: bool = True
    bump_interval_range: tuple = (6000, 15000)  # Interval between bumps
    bump_duration_range: tuple = (100, 300)  # Duration (much longer than spike's 10-25)
    bump_magnitude_max: float = 0.10  # Max magnitude (much less than anomaly's 0.3+)
    bump_features_affected: int = 2  # Only 1-2 features affected (not correlated degradation)

    # 7. Phase Jitter - Slowly varying phase to break strict periodicity
    enable_phase_jitter: bool = True
    phase_jitter_sigma: float = 0.002  # Random walk step size for phase
    phase_jitter_smoothing: int = 500  # Smoothing window for phase changes


class SlidingWindowTimeSeriesGenerator:
    """
    Generates a long continuous time series with anomalies injected.

    The time series simulates server monitoring data with 8 correlated features.
    Anomalies are injected at random intervals throughout the series.

    Normal data can include various complexity features (regime switching,
    multi-scale periodicity, heteroscedastic noise, etc.) controlled by
    the `complexity` parameter.
    """

    def __init__(
        self,
        total_length: int,
        num_features: int = 8,
        anomaly_type_configs: Optional[Dict] = None,  # Per-type configs
        interval_scale: float = 1.0,  # Scale factor for all intervals (for tuning)
        complexity: Optional[NormalDataComplexity] = None,  # Normal data complexity config
        seed: Optional[int] = None
    ):
        self.total_length = total_length
        self.num_features = min(num_features, NUM_FEATURES)
        self.interval_scale = interval_scale
        self.seed = seed

        # Normal data complexity configuration
        self.complexity = complexity or NormalDataComplexity()

        # Use default configs or provided ones
        self.anomaly_type_configs = anomaly_type_configs or ANOMALY_TYPE_CONFIGS

        if seed is not None:
            np.random.seed(seed)

    def _generate_normal_series(self) -> np.ndarray:
        """
        Generate a long normal time series with 8 correlated features.

        With complexity enabled, includes:
        1. Regime switching (different operational states)
        2. Multi-scale periodicity (overlapping cycles)
        3. Heteroscedastic noise (load-dependent variance)
        4. Time-varying correlations
        5. Bounded drift (Ornstein-Uhlenbeck process)
        6. Normal bumps (small, gradual load increases)
        """
        signals = np.zeros((self.total_length, self.num_features), dtype=np.float32)
        c = self.complexity  # Shorthand

        # If complexity is disabled, use simple generation
        if not c.enable_complexity:
            return self._generate_simple_normal_series()

        # === Step 1: Generate regime schedule ===
        if c.enable_regime_switching:
            regimes = self._generate_regimes()
        else:
            regimes = [{'start': 0, 'end': self.total_length, 'params': self._random_regime_params()}]

        # === Step 2: Generate base CPU signal with regime-aware multi-scale periodicity ===
        # Generate phase jitter for the entire series (breaks strict periodicity)
        if c.enable_phase_jitter:
            phase_jitter = self._generate_phase_jitter(self.total_length)
        else:
            phase_jitter = np.zeros(self.total_length)

        for regime in regimes:
            start, end = regime['start'], regime['end']
            params = regime['params']
            length = end - start
            t = np.arange(length) * 0.01  # Time scale
            jitter = phase_jitter[start:end]  # Phase jitter for this segment

            if c.enable_multi_scale_periodicity:
                # Three overlapping frequencies with phase jitter
                # Irrational frequency ratios + phase jitter = non-repeating patterns
                signals[start:end, 0] = (
                    params['cpu_base'] +
                    params['cpu_amp1'] * np.sin(params['freq1'] * t + jitter) +  # Fast (hourly-like)
                    params['cpu_amp2'] * np.sin(params['freq2'] * t + jitter * 0.7) +  # Medium (daily-like)
                    params['cpu_amp3'] * np.sin(params['freq3'] * t + jitter * 0.4)    # Slow (weekly-like)
                )
            else:
                signals[start:end, 0] = (
                    params['cpu_base'] +
                    params['cpu_amp1'] * np.sin(params['freq1'] * t + jitter)
                )

        # === Step 3: Apply smooth regime transitions ===
        if c.enable_regime_switching and len(regimes) > 1:
            signals[:, 0] = self._apply_regime_transitions(signals[:, 0], regimes)

        # === Step 4: Generate correlated features with time-varying correlations ===
        signals = self._generate_correlated_features(signals, regimes)

        # === Step 5: Add bounded drift (Ornstein-Uhlenbeck process) ===
        if c.enable_drift:
            drift = self._generate_ou_drift()
            for i in range(self.num_features):
                signals[:, i] += drift[:, i]

        # === Step 6: Add heteroscedastic noise ===
        if c.enable_heteroscedastic_noise:
            signals = self._add_heteroscedastic_noise(signals)
        else:
            for i in range(self.num_features):
                signals[:, i] += np.random.normal(0, 0.03, self.total_length)

        # === Step 7: Add normal bumps ===
        if c.enable_normal_bumps:
            signals = self._add_normal_bumps(signals)

        # No hard clipping - let normal data have natural range
        # This makes the task more challenging and realistic

        return signals.astype(np.float32)

    def _generate_simple_normal_series(self) -> np.ndarray:
        """Simple normal series generation (when complexity is disabled)."""
        t = np.linspace(0, self.total_length * 0.1, self.total_length)
        signals = np.zeros((self.total_length, self.num_features), dtype=np.float32)

        cpu_freq = np.random.uniform(0.5, 1.5)
        cpu_amp = np.random.uniform(0.2, 0.4)
        cpu_base = np.random.uniform(0.3, 0.5)

        signals[:, 0] = (
            cpu_base + cpu_amp * np.sin(cpu_freq * t) +
            0.1 * np.sin(cpu_freq * 0.3 * t) +
            np.random.normal(0, 0.03, self.total_length)
        )

        if self.num_features > 1:
            signals[:, 1] = 0.5 + 0.25 * signals[:, 0] + np.random.normal(0, 0.02, self.total_length)
        if self.num_features > 2:
            signals[:, 2] = 0.2 + 0.15 * signals[:, 1] + np.random.normal(0, 0.03, self.total_length)
        if self.num_features > 3:
            signals[:, 3] = 0.25 * np.abs(np.sin(cpu_freq * 1.2 * t)) + np.random.normal(0, 0.03, self.total_length)
        if self.num_features > 4:
            signals[:, 4] = 0.2 + 0.25 * signals[:, 0] + 0.15 * signals[:, 3] + np.random.normal(0, 0.03, self.total_length)
        if self.num_features > 5:
            cpu_smooth = gaussian_filter1d(signals[:, 0], sigma=10)
            signals[:, 5] = 0.4 + 0.3 * cpu_smooth + np.random.normal(0, 0.02, self.total_length)
        if self.num_features > 6:
            signals[:, 6] = 0.05 + 0.1 * np.maximum(0, signals[:, 4] - 0.3) + np.random.exponential(0.02, self.total_length)
        if self.num_features > 7:
            signals[:, 7] = 0.2 + 0.2 * signals[:, 0] + 0.15 * signals[:, 5] + np.random.normal(0, 0.03, self.total_length)

        return _normalize_per_feature(signals)

    def _random_regime_params(self) -> Dict:
        """Generate random parameters for a single regime.

        Uses irrational frequency ratios (π, √2, φ) to prevent
        beat pattern repetition and reduce strict periodicity.
        """
        # Base fast frequency
        freq1 = np.random.uniform(0.8, 1.5)

        # Use irrational ratios to prevent beat pattern repetition
        # π ≈ 3.14159, √2 ≈ 1.41421, φ (golden ratio) ≈ 1.61803
        PI = np.pi
        SQRT2 = np.sqrt(2)
        PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

        # Medium frequency: divide by π with small random variation
        freq2 = freq1 / (PI * np.random.uniform(2.8, 3.5))  # ~1/9 to 1/11
        # Slow frequency: divide by π² with variation
        freq3 = freq1 / (PI * PI * np.random.uniform(1.5, 2.5))  # ~1/15 to 1/25

        return {
            # Base values (kept in safe range 0.25-0.55)
            'cpu_base': np.random.uniform(0.28, 0.48),
            'mem_base': np.random.uniform(0.35, 0.55),
            'disk_base': np.random.uniform(0.12, 0.25),
            'net_base': np.random.uniform(0.15, 0.35),
            'resp_base': np.random.uniform(0.12, 0.25),
            'thread_base': np.random.uniform(0.30, 0.48),
            'error_base': np.random.uniform(0.03, 0.08),
            'queue_base': np.random.uniform(0.12, 0.25),
            # Multi-scale periodicity parameters with irrational ratios
            'cpu_amp1': np.random.uniform(0.06, 0.12),  # Fast cycle amplitude
            'cpu_amp2': np.random.uniform(0.04, 0.08),  # Medium cycle amplitude
            'cpu_amp3': np.random.uniform(0.02, 0.05),  # Slow cycle amplitude
            'freq1': freq1,       # Fast frequency
            'freq2': freq2,       # Medium frequency (irrational ratio)
            'freq3': freq3,       # Slow frequency (irrational ratio)
            # Correlation strengths (will vary over time if enabled)
            'corr_cpu_mem': np.random.uniform(0.20, 0.30),
            'corr_cpu_resp': np.random.uniform(0.20, 0.30),
            'corr_cpu_thread': np.random.uniform(0.25, 0.35),
            'corr_mem_disk': np.random.uniform(0.12, 0.20),
        }

    def _generate_regimes(self) -> List[Dict]:
        """Generate regime schedule with smooth transitions."""
        c = self.complexity
        regimes = []
        current_pos = 0

        while current_pos < self.total_length:
            # Random regime duration
            duration = np.random.randint(c.regime_duration_range[0], c.regime_duration_range[1])
            end_pos = min(current_pos + duration, self.total_length)

            regimes.append({
                'start': current_pos,
                'end': end_pos,
                'params': self._random_regime_params()
            })

            current_pos = end_pos

        return regimes

    def _generate_phase_jitter(self, length: int) -> np.ndarray:
        """Generate smoothly varying phase jitter using random walk.

        This breaks strict periodicity by adding a slowly-changing phase offset
        to sinusoidal components, making patterns less predictable.

        Returns:
            Array of phase offsets (in radians) for each timestep.
        """
        c = self.complexity

        # Generate random walk
        random_steps = np.random.normal(0, c.phase_jitter_sigma, length)
        phase_walk = np.cumsum(random_steps)

        # Smooth the random walk to avoid sudden jumps
        smoothed_phase = gaussian_filter1d(phase_walk, sigma=c.phase_jitter_smoothing)

        return smoothed_phase

    def _apply_regime_transitions(self, signal: np.ndarray, regimes: List[Dict]) -> np.ndarray:
        """Apply smooth sigmoid transitions between regimes."""
        c = self.complexity
        transition_len = c.regime_transition_length

        for i in range(1, len(regimes)):
            prev_regime = regimes[i - 1]
            curr_regime = regimes[i]
            transition_start = curr_regime['start']

            # Create smooth transition using sigmoid
            if transition_start + transition_len <= self.total_length:
                t = np.arange(transition_len)
                # Sigmoid from 0 to 1
                sigmoid = 1 / (1 + np.exp(-10 * (t / transition_len - 0.5)))

                # Blend values
                old_vals = signal[transition_start:transition_start + transition_len].copy()
                # Recalculate with new regime params would be complex,
                # so we just smooth the existing transition
                smoothed = gaussian_filter1d(signal[max(0, transition_start - 500):
                                                    min(self.total_length, transition_start + transition_len + 500)],
                                             sigma=transition_len // 6)
                # Apply smoothed values to transition region
                smooth_start = transition_start - max(0, transition_start - 500)
                signal[transition_start:transition_start + transition_len] = \
                    smoothed[smooth_start:smooth_start + transition_len]

        return signal

    def _generate_correlated_features(self, signals: np.ndarray, regimes: List[Dict]) -> np.ndarray:
        """Generate correlated features with optional time-varying correlations."""
        c = self.complexity

        # Time-varying correlation modifier
        if c.enable_varying_correlations:
            t = np.arange(self.total_length)
            corr_modifier = c.correlation_variation_amplitude * np.sin(
                2 * np.pi * t / c.correlation_variation_period
            )
        else:
            corr_modifier = np.zeros(self.total_length)

        # Get base correlation from first regime (will be modified)
        base_params = regimes[0]['params']

        # Feature 1: Memory (correlated with CPU)
        if self.num_features > 1:
            corr = base_params['corr_cpu_mem'] + corr_modifier
            for regime in regimes:
                start, end = regime['start'], regime['end']
                p = regime['params']
                t = np.arange(end - start) * 0.01
                signals[start:end, 1] = (
                    p['mem_base'] +
                    corr[start:end] * signals[start:end, 0] +
                    0.05 * np.sin(p['freq2'] * t * 1.1)
                )

        # Feature 2: Disk I/O (correlated with Memory)
        if self.num_features > 2:
            corr = base_params['corr_mem_disk'] + corr_modifier * 0.5
            for regime in regimes:
                start, end = regime['start'], regime['end']
                p = regime['params']
                # Small random spikes (normal disk activity)
                disk_activity = np.random.poisson(0.03, end - start) * 0.08
                signals[start:end, 2] = (
                    p['disk_base'] +
                    corr[start:end] * signals[start:end, 1] +
                    disk_activity
                )

        # Feature 3: Network (semi-independent, bursty)
        if self.num_features > 3:
            for regime in regimes:
                start, end = regime['start'], regime['end']
                p = regime['params']
                t = np.arange(end - start) * 0.01
                signals[start:end, 3] = (
                    p['net_base'] +
                    0.08 * np.abs(np.sin(p['freq1'] * t * 0.8)) +
                    0.04 * np.abs(np.sin(p['freq2'] * t * 1.3))
                )

        # Feature 4: Response Time (correlated with CPU and Network)
        if self.num_features > 4:
            corr = base_params['corr_cpu_resp'] + corr_modifier
            for regime in regimes:
                start, end = regime['start'], regime['end']
                p = regime['params']
                signals[start:end, 4] = (
                    p['resp_base'] +
                    corr[start:end] * signals[start:end, 0] +
                    0.10 * signals[start:end, 3]
                )

        # Feature 5: Thread Count (correlated with CPU, smoother)
        if self.num_features > 5:
            corr = base_params['corr_cpu_thread'] + corr_modifier
            cpu_smooth = gaussian_filter1d(signals[:, 0], sigma=15)
            for regime in regimes:
                start, end = regime['start'], regime['end']
                p = regime['params']
                signals[start:end, 5] = (
                    p['thread_base'] +
                    corr[start:end] * cpu_smooth[start:end]
                )

        # Feature 6: Error Rate (low, slightly correlated with response time)
        if self.num_features > 6:
            for regime in regimes:
                start, end = regime['start'], regime['end']
                p = regime['params']
                # Errors slightly increase when response time is higher (but stay low in normal)
                signals[start:end, 6] = (
                    p['error_base'] +
                    0.05 * np.maximum(0, signals[start:end, 4] - 0.25)
                )

        # Feature 7: Queue Length (correlated with CPU and Thread)
        if self.num_features > 7:
            for regime in regimes:
                start, end = regime['start'], regime['end']
                p = regime['params']
                signals[start:end, 7] = (
                    p['queue_base'] +
                    0.15 * signals[start:end, 0] +
                    0.10 * signals[start:end, 5]
                )

        return signals

    def _generate_ou_drift(self) -> np.ndarray:
        """Generate Ornstein-Uhlenbeck (mean-reverting) drift for each feature."""
        c = self.complexity
        drift = np.zeros((self.total_length, self.num_features), dtype=np.float32)

        for feat in range(self.num_features):
            x = 0.0
            for t in range(self.total_length):
                # O-U process: dx = theta * (mu - x) * dt + sigma * dW
                # mu = 0 (drift around zero)
                dx = -c.drift_theta * x + c.drift_sigma * np.random.randn()
                x += dx
                # Clip to maximum drift
                x = np.clip(x, -c.drift_max, c.drift_max)
                drift[t, feat] = x

        return drift

    def _add_heteroscedastic_noise(self, signals: np.ndarray) -> np.ndarray:
        """Add load-dependent noise (higher load = more variance)."""
        c = self.complexity

        # Use CPU (feature 0) as load proxy
        load = signals[:, 0]

        for i in range(self.num_features):
            # Noise scale increases with load
            noise_scale = c.base_noise * (1 + c.noise_load_sensitivity * load)
            noise = np.random.randn(self.total_length) * noise_scale
            signals[:, i] += noise

        return signals

    def _add_normal_bumps(self, signals: np.ndarray) -> np.ndarray:
        """Add small, gradual load increases (normal operational bumps).

        These are NOT anomalies - they represent:
        - Scheduled batch jobs
        - Normal traffic variations
        - Planned maintenance activities

        Key differences from anomaly spikes:
        - Much longer duration (100-300 vs 10-25 timesteps)
        - Much smaller magnitude (max 0.10 vs 0.3-0.6)
        - Smooth Gaussian shape (not sudden)
        - Only 1-2 features affected (no correlated degradation)
        - NO error rate increase
        """
        c = self.complexity
        current_pos = np.random.randint(1000, 3000)  # Start after initial period

        while current_pos < self.total_length - c.bump_duration_range[1]:
            # Random bump parameters
            duration = np.random.randint(c.bump_duration_range[0], c.bump_duration_range[1])
            magnitude = np.random.uniform(0.03, c.bump_magnitude_max)

            # Select 1-2 features to affect (exclude error rate - feature 6)
            safe_features = [0, 1, 3, 5, 7]  # CPU, Memory, Network, Thread, Queue
            if self.num_features <= 6:
                safe_features = [i for i in safe_features if i < self.num_features]
            num_features = np.random.randint(1, min(3, len(safe_features) + 1))
            affected_features = np.random.choice(safe_features, size=num_features, replace=False)

            # Gaussian envelope (smooth bump shape)
            t = np.arange(duration)
            envelope = np.exp(-0.5 * ((t - duration / 2) / (duration / 4)) ** 2)
            envelope *= magnitude

            # Apply bump to selected features
            for feat in affected_features:
                end_pos = min(current_pos + duration, self.total_length)
                actual_len = end_pos - current_pos
                signals[current_pos:end_pos, feat] += envelope[:actual_len]

            # Move to next bump position
            interval = np.random.randint(c.bump_interval_range[0], c.bump_interval_range[1])
            current_pos += duration + interval

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

    # =========================================================================
    # Pattern-based anomalies (types 7-9)
    # These maintain normal value ranges but break temporal/correlation patterns
    # =========================================================================

    def _inject_correlation_inversion(self, signals: np.ndarray, start: int, end: int) -> None:
        """Inject correlation inversion anomaly (CPU-Memory correlation breaks)

        Real-world scenario: Cache misconfiguration, where Memory decreases when CPU increases
        (opposite of normal positive correlation).

        Values stay within normal range, but cross-feature correlation is inverted.
        """
        # Get local statistics for CPU (feature 0) and Memory (feature 1)
        cpu_local_mean = signals[start:end, 0].mean()
        mem_local_mean = signals[start:end, 1].mean() if self.num_features > 1 else 0.5

        if self.num_features > 1:
            # Invert CPU-Memory correlation
            # When CPU deviates from mean, Memory deviates in opposite direction
            cpu_deviation = signals[start:end, 0] - cpu_local_mean
            new_memory = mem_local_mean - cpu_deviation * np.random.uniform(0.6, 0.9)
            # Clip to stay within reasonable range (not too extreme)
            signals[start:end, 1] = np.clip(new_memory, 0.15, 0.85)

        # Also invert ThreadCount-CPU correlation (normally positive)
        if self.num_features > 5:
            thread_local_mean = signals[start:end, 5].mean()
            cpu_deviation = signals[start:end, 0] - cpu_local_mean
            new_thread = thread_local_mean - cpu_deviation * np.random.uniform(0.5, 0.8)
            signals[start:end, 5] = np.clip(new_thread, 0.15, 0.85)

    def _inject_temporal_flatline(self, signals: np.ndarray, start: int, end: int) -> None:
        """Inject temporal flatline anomaly (values freeze / stuck sensor)

        Real-world scenario: Metric collection failure, sensor stuck at last reading.
        Values stay within normal range but temporal variation disappears.
        """
        # Select 3-5 features to freeze (not all, to be realistic)
        num_features_to_freeze = min(np.random.randint(3, 6), self.num_features)
        features_to_freeze = np.random.choice(
            self.num_features,
            size=num_features_to_freeze,
            replace=False
        )

        # Freeze each selected feature at its value just before the anomaly
        for feat in features_to_freeze:
            frozen_value = signals[start, feat]
            signals[start:end, feat] = frozen_value

    def _inject_frequency_shift(self, signals: np.ndarray, start: int, end: int) -> None:
        """Inject frequency shift anomaly (unusual periodicity)

        Real-world scenario: Wrong cron interval, abnormal scheduling pattern.
        Values stay in normal range but oscillate at different frequency.
        """
        length = end - start

        # Apply to first 3-4 features
        features_to_shift = min(np.random.randint(3, 5), self.num_features)

        for feat in range(features_to_shift):
            # Get local statistics to maintain normal value range
            local_mean = signals[start:end, feat].mean()
            local_std = signals[start:end, feat].std()
            local_std = max(local_std, 0.05)  # Ensure some variation

            # Create higher frequency oscillation (2-4x normal)
            freq_multiplier = np.random.uniform(2.5, 4.0)
            phase = np.random.uniform(0, 2 * np.pi)
            t = np.linspace(0, freq_multiplier * np.pi, length)

            # Replace with frequency-shifted signal (same amplitude, different frequency)
            new_signal = local_mean + local_std * np.sin(t + phase)
            signals[start:end, feat] = np.clip(new_signal, 0.15, 0.85)

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

    def generate(self, inject_anomalies: bool = True) -> Tuple[np.ndarray, np.ndarray, List[AnomalyRegion]]:
        """
        Generate the complete time series with optional anomalies.

        Args:
            inject_anomalies: If True (default), inject anomalies into the data.
                              If False, return pure normal data without anomalies.

        Returns:
            signals: (total_length, num_features) normalized time series
            point_labels: (total_length,) binary labels (1 = anomaly point)
            anomaly_regions: List of AnomalyRegion objects (empty if inject_anomalies=False)
        """
        # Generate normal base series
        signals = self._generate_normal_series()

        # Initialize point labels
        point_labels = np.zeros(self.total_length, dtype=np.int64)
        anomaly_regions = []

        if inject_anomalies:
            # Distribute and inject anomalies
            anomaly_regions = self._distribute_anomalies()

            inject_funcs = {
                # Value-based anomalies (types 1-6)
                1: self._inject_spike,
                2: self._inject_memory_leak,
                3: self._inject_cpu_saturation,
                4: self._inject_network_congestion,
                5: self._inject_cascading_failure,
                6: self._inject_resource_contention,
                # Pattern-based anomalies (types 7-9)
                7: self._inject_correlation_inversion,
                8: self._inject_temporal_flatline,
                9: self._inject_frequency_shift,
            }

            for region in anomaly_regions:
                inject_funcs[region.anomaly_type](signals, region.start, region.end)
                point_labels[region.start:region.end] = 1

        # Normalize each feature to [0, 1] range
        signals = _normalize_per_feature(signals)

        return signals, point_labels, anomaly_regions


class SlidingWindowDataset(Dataset):
    """
    Dataset that extracts sliding windows from a long time series.

    Sample Types:
        0: pure_normal - no anomaly in the window
        1: disturbing_normal - anomaly exists but NOT in last mask_last_n (label=0)
        2: anomaly - anomaly exists in last mask_last_n (label=1)

    Note:
        For test split, stride is always forced to 1 for proper point-level PA%K evaluation.
        Downsampling is disabled by default for test set to preserve full sliding window coverage.
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
        target_anomaly_ratio: Optional[float] = None,  # For test set downsampling (legacy, disabled by default)
        target_counts: Optional[Dict[str, int]] = None,  # Explicit counts per sample type (disabled by default)
        seed: Optional[int] = None,
        force_stride_1_for_test: bool = True  # Force stride=1 for test split (for point-level PA%K)
    ):
        self.window_size = window_size
        self.mask_last_n = mask_last_n
        self.split = split
        self.target_counts = target_counts

        # For test split, force stride=1 for proper point-level PA%K evaluation
        if split == 'test' and force_stride_1_for_test:
            self.stride = 1
        else:
            self.stride = stride

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
        self.window_start_indices = []  # Start index of each window (for point-level aggregation)

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
            self.window_start_indices.append(start)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.seq_labels = np.array(self.seq_labels, dtype=np.int64)
        self.window_point_labels = np.array(self.window_point_labels, dtype=np.int64)
        self.sample_types = np.array(self.sample_types, dtype=np.int64)
        self.anomaly_type_labels = np.array(self.anomaly_type_labels, dtype=np.int64)
        self.window_start_indices = np.array(self.window_start_indices, dtype=np.int64)

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

            # Adjust targets if not enough samples available
            if len(anomaly_indices) < target_anomaly:
                target_anomaly = len(anomaly_indices)
            if len(disturb_indices) < target_disturb:
                target_disturb = len(disturb_indices)
            if len(pure_indices) < target_pure:
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
        self.window_start_indices = self.window_start_indices[keep_indices]

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
