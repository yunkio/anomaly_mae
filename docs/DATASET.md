# Dataset Documentation

**Last Updated**: 2026-01-24

---

## Overview

This project uses a **Sliding Window Time Series Dataset** that simulates server monitoring data. The dataset is designed to be:

1. **Patch-size independent**: Anomaly positions are fixed regardless of model configuration
2. **Temporally continuous**: Samples come from a single long time series
3. **Realistic**: Train/test split by time (no data leakage)

---

## Dataset Generation Process

```
1. Generate Long Time Series (2.2M timesteps)
   ├── 8 correlated features (server metrics)
   └── Inject anomalies at random intervals

2. Sliding Window Extraction
   ├── Window size: 100 timesteps
   ├── Train stride: Configurable (default 10)
   ├── Test stride: Always 1 (for point-level PA%K)
   └── Total windows: ~220,000 (train), ~110,000 (test)

3. Train/Test Split
   ├── Train: First 50% (no downsampling, ~5% anomaly)
   └── Test: Last 50% (stride=1, no downsampling by default)

4. Labeling
   ├── Check last 10 timesteps (mask_last_n)
   └── Classify as: pure_normal, disturbing_normal, or anomaly
```

---

## Sample Types

| Type | Label | Description |
|------|-------|-------------|
| **Pure Normal** | 0 | No anomaly anywhere in the window |
| **Disturbing Normal** | 0 | Anomaly exists but NOT in last 10 timesteps |
| **Anomaly** | 1 | Anomaly exists in last 10 timesteps |

### Why "Disturbing Normal"?

This represents a challenging case where:
- The window contains anomalous patterns earlier in the sequence
- But the evaluation region (last 10 timesteps) is normal
- Tests if the model correctly ignores past anomalies when predicting the masked region

---

## Dataset Statistics

### Train Set (Natural Distribution)

| Sample Type | Approx. Ratio |
|-------------|---------------|
| Pure Normal | ~88% |
| Disturbing Normal | ~7% |
| Anomaly | ~5% |

### Test Set (Full, Stride=1)

> **Note**: As of the latest update, test set uses stride=1 and no downsampling by default.
> This ensures proper point-level PA%K evaluation with overlapping windows.

| Aspect | Value |
|--------|-------|
| Stride | 1 (forced) |
| Downsampling | Disabled by default |
| Total windows | ~110,000 (half of time series) |

**Legacy mode** (downsampled, for backwards compatibility):

| Sample Type | Count | Ratio |
|-------------|-------|-------|
| Pure Normal | 1,200 | 60% |
| Disturbing Normal | 300 | 15% |
| Anomaly | 500 | 25% |
| **Total** | **2,000** | 100% |

---

## Point-Level PA%K Evaluation

With stride=1 sliding windows, each timestep is covered by multiple windows' last patches.
The evaluation aggregates window-level scores to point-level using one of three methods:

### Aggregation Methods

| Method | Description | Formula |
|--------|-------------|---------|
| **Voting** (default) | Majority vote of binary predictions | `1 if votes > n/2 else 0` |
| **Mean** | Average of window scores | `mean(scores)` |
| **Median** | Median of window scores | `median(scores)` |

### Window Coverage

For `seq_length=100` and `patch_size=10`:

```
Window w's last patch: timesteps [w+90, w+99]

Timestep t is covered by windows where:
  w+90 ≤ t ≤ w+99
  → w ∈ [t-99, t-90]
  → Up to 10 windows per timestep

Coverage by position:
  - Timesteps 0-89: Not covered (not in any last patch)
  - Timesteps 90-99: 1-10 windows
  - Timesteps 100+: 10 windows (full coverage)
```

### Sample-Level vs Point-Level Metrics

| Metric Type | Level | Description |
|-------------|-------|-------------|
| ROC-AUC, F1, Precision, Recall | Sample (window) | Each window = one sample |
| PA%K F1, PA%K ROC-AUC | Point (timestep) | Aggregated to timestep level |

---

## Features (8 Server Metrics)

| Index | Name | Description | Correlation |
|-------|------|-------------|-------------|
| 0 | CPU | CPU usage (0-1) | Base signal |
| 1 | Memory | Memory usage | Correlated with CPU |
| 2 | DiskIO | Disk I/O operations | Correlated with Memory, spiky |
| 3 | Network | Network traffic | Bursty pattern |
| 4 | ResponseTime | Response latency | Correlated with CPU, Network |
| 5 | ThreadCount | Active thread count | Smoothed CPU correlation |
| 6 | ErrorRate | Error rate | Correlated with ResponseTime |
| 7 | QueueLength | Request queue length | Correlated with CPU, ThreadCount |

### Feature Generation

```python
# Base CPU pattern (periodic + noise)
CPU = base + amp * sin(freq * t) + noise

# Memory (correlated with CPU)
Memory = base + 0.25 * CPU + 0.15 * sin(slower_freq * t) + noise

# DiskIO (Memory-correlated + spikes)
DiskIO = base + 0.15 * Memory + poisson_spikes + noise

# Network (bursty)
Network = amp * |sin(freq * t)| + exponential_bursts + noise

# ResponseTime (CPU + Network correlated)
ResponseTime = base + 0.25 * CPU + 0.15 * Network + noise

# ThreadCount (smoothed CPU)
ThreadCount = base + 0.3 * gaussian_smooth(CPU) + noise

# ErrorRate (ResponseTime threshold)
ErrorRate = base + 0.1 * max(0, ResponseTime - 0.3) + exponential

# QueueLength (CPU + ThreadCount)
QueueLength = base + 0.2 * CPU + 0.15 * ThreadCount + noise
```

### Data Normalization

After signal generation (including anomaly injection), each feature is **independently normalized** to the [0, 1] range using **per-feature min-max normalization**:

```python
def _normalize_per_feature(signals: np.ndarray) -> np.ndarray:
    """Per-feature min-max normalization to [0, 1] range."""
    for f in range(signals.shape[1]):
        min_val = signals[:, f].min()
        max_val = signals[:, f].max()
        if max_val - min_val > 1e-8:
            signals[:, f] = (signals[:, f] - min_val) / (max_val - min_val)
        else:
            signals[:, f] = 0.5  # Constant signal -> set to middle
    return signals
```

**Why normalization instead of clipping?**

| Aspect | Clipping (`np.clip(signals, 0, 1)`) | Min-Max Normalization |
|--------|-------------------------------------|----------------------|
| Anomaly spikes | Capped at 1.0 (info loss) | Preserved proportionally |
| Boundary artifacts | Flat regions at 0 or 1 | No artificial saturation |
| Relative magnitudes | Distorted near boundaries | Preserved exactly |
| Real-world similarity | Less realistic | Matches standard preprocessing |

---

## Normal Data Complexity Features

The dataset includes **configurable complexity features** to make normal data more realistic and challenging for anomaly detection models. **All features are designed to NOT be confused with anomaly patterns.**

### Quick Configuration

```python
from mae_anomaly.dataset_sliding import NormalDataComplexity, SlidingWindowTimeSeriesGenerator

# Create complexity config (all features enabled by default)
complexity = NormalDataComplexity(
    enable_complexity=True,           # Master switch
    enable_regime_switching=True,     # Different operational states
    enable_multi_scale_periodicity=True,  # Overlapping cycles
    enable_heteroscedastic_noise=True,    # Load-dependent noise
    enable_varying_correlations=True,     # Time-varying correlations
    enable_drift=True,                    # O-U mean-reverting drift
    enable_normal_bumps=True,             # Small load bumps
    enable_phase_jitter=True,             # Break strict periodicity
)

# Create generator with complexity
generator = SlidingWindowTimeSeriesGenerator(
    total_length=440000,
    complexity=complexity,
    seed=42
)
```

### Disabling Complexity (Simple Mode)

```python
# Disable all complexity (original behavior)
complexity = NormalDataComplexity(enable_complexity=False)

# Or disable specific features
complexity = NormalDataComplexity(
    enable_regime_switching=False,
    enable_normal_bumps=False,
)
```

---

### 1. Regime Switching

**Purpose**: Simulate different operational states (low load, normal, high load).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_regime_switching` | True | On/off switch |
| `regime_duration_range` | (8000, 25000) | Duration of each regime |
| `regime_transition_length` | 1500 | Smooth transition period |

**How it works**:
- Time series is divided into regimes of 8000-25000 timesteps
- Each regime has different base values, amplitudes, and frequencies
- Transitions use sigmoid smoothing over 1500+ timesteps

**Why NOT confused with anomalies**:
- Transitions take 1500+ timesteps (anomalies are 3-150 ts)
- Values stay in normal range (0.28-0.48 vs anomaly's 0.7+)
- Changes are bidirectional (can go up or down)

---

### 2. Multi-Scale Periodicity

**Purpose**: Add realistic overlapping cycles (hourly, daily, weekly patterns).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_multi_scale_periodicity` | True | On/off switch |

**Frequency scales** (using irrational ratios to prevent beat pattern repetition):
```
freq1: 0.8-1.5                      # Fast (hourly-like)
freq2: freq1 / (π * [2.8-3.5])      # Medium (~1/9 to 1/11 of freq1)
freq3: freq1 / (π² * [1.5-2.5])     # Slow (~1/15 to 1/25 of freq1)

signal = base + amp1*sin(freq1*t + jitter) + amp2*sin(freq2*t + jitter*0.7) + amp3*sin(freq3*t + jitter*0.4)
```

**Irrational frequency ratios**: Using π-based ratios (instead of integer ratios like 1:10:50) ensures that beat patterns never repeat exactly, making the signal more realistic.

**Why NOT confused with anomalies**:
- All patterns are smooth sinusoids (no sudden changes)
- Total amplitude stays bounded (sum of amps < 0.25)
- Patterns are continuous (no discontinuities)

---

### 3. Heteroscedastic Noise

**Purpose**: Realistic load-dependent noise variance (busier = noisier).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_heteroscedastic_noise` | True | On/off switch |
| `base_noise` | 0.025 | Base noise level |
| `noise_load_sensitivity` | 0.8 | Load multiplier |

**Formula**:
```python
noise_scale = base_noise * (1 + noise_load_sensitivity * cpu_load)
# CPU at 0.3 → noise ≈ 0.031
# CPU at 0.5 → noise ≈ 0.035
```

**Why NOT confused with anomalies**:
- Noise is symmetric (anomaly spikes are always upward)
- Maximum amplitude ≈ 0.08 (3σ), anomalies are 0.3+
- No sustained bias in any direction

---

### 4. Time-Varying Correlations

**Purpose**: Feature correlations that slowly change over time.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_varying_correlations` | True | On/off switch |
| `correlation_variation_period` | 15000 | Period of change |
| `correlation_variation_amplitude` | 0.08 | ±variation |

**Formula**:
```python
corr_modifier = 0.08 * sin(2π * t / 15000)
effective_corr = base_corr + corr_modifier
# CPU-Memory correlation varies between 0.12-0.38
```

**Why NOT confused with anomalies**:
- Changes are extremely gradual (period = 15000 timesteps)
- No sudden correlation changes
- Correlations stay positive and bounded

---

### 5. Bounded Drift (Ornstein-Uhlenbeck Process)

**Purpose**: Mean-reverting random walk for realistic baseline drift.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_drift` | True | On/off switch |
| `drift_theta` | 0.002 | Mean reversion speed |
| `drift_sigma` | 0.025 | Volatility |
| `drift_max` | 0.08 | Maximum drift magnitude |

**O-U Process**:
```python
dx = -theta * x * dt + sigma * dW
x = clip(x + dx, -drift_max, drift_max)
```

**Why NOT confused with anomalies**:
- Bidirectional (goes up AND down), memory_leak is monotonic
- Maximum magnitude 0.08 << leak's 0.3-0.5 increase
- Mean-reverting (always returns to baseline)

---

### 6. Normal Bumps

**Purpose**: Small, gradual load increases representing normal operations (batch jobs, traffic variations).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_normal_bumps` | True | On/off switch |
| `bump_interval_range` | (6000, 15000) | Interval between bumps |
| `bump_duration_range` | (100, 300) | Duration per bump |
| `bump_magnitude_max` | 0.10 | Maximum magnitude |
| `bump_features_affected` | 2 | Max features affected |

**Comparison with Anomaly Spike**:

| Aspect | Normal Bump | Anomaly Spike |
|--------|-------------|---------------|
| Duration | 100-300 ts | 10-25 ts |
| Magnitude | max 0.10 | 0.3-0.6 |
| Shape | Smooth Gaussian | Sudden |
| Features | 1-2 only | 5+ simultaneous |
| Error rate | NOT affected | Increases |

---

### 7. Phase Jitter

**Purpose**: Break strict periodicity by adding slowly-varying phase offsets to sinusoidal components.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_phase_jitter` | True | On/off switch |
| `phase_jitter_sigma` | 0.002 | Random walk step size |
| `phase_jitter_smoothing` | 500 | Smoothing window for phase |

**How it works**:
- Generates a smoothed random walk as phase offset
- Applied to each frequency component with decreasing weight (1.0, 0.7, 0.4)
- Combined with irrational frequency ratios, ensures patterns never repeat exactly

**Why NOT confused with anomalies**:
- Phase changes are extremely gradual (smoothed over 500 timesteps)
- Does not change amplitude or value range
- Only affects timing of peaks/valleys, not their magnitude

---

### Safety Constraints Summary

| Constraint | Value | Reason |
|------------|-------|--------|
| Transition time | >= 1000 ts | Anomalies are much shorter |
| Value range | Per-feature [0,1] normalized | Relative magnitudes preserved |
| Drift magnitude | max ±0.08 | Memory leak grows 0.3-0.5 |
| Bump magnitude | max 0.10 | Spike adds 0.3-0.6 |
| Bump duration | 100-300 ts | Spike is 10-25 ts |

---

### Tuning Difficulty

```python
# Easier (less complexity)
complexity = NormalDataComplexity(
    enable_regime_switching=False,
    enable_normal_bumps=False,
)

# Harder (more variation)
complexity = NormalDataComplexity(
    regime_duration_range=(5000, 15000),  # More frequent regime changes
    bump_interval_range=(4000, 10000),    # More frequent bumps
    bump_magnitude_max=0.12,              # Slightly larger bumps
)
```

---

## Anomaly Types

The dataset includes **9 distinct anomaly types** divided into two categories:

- **Value-based anomalies (Types 1-6)**: Values deviate from normal range (ADDITIVE injection)
- **Pattern-based anomalies (Types 7-9)**: Values stay within normal range, patterns differ

This distinction allows evaluating whether the model detects anomalies based on unusual VALUES (trivial) or unusual PATTERNS (meaningful).

### Anomaly Type Summary

| ID | Name | Category | Duration | Interval | Description |
|----|------|----------|----------|----------|-------------|
| 0 | Normal | - | - | - | No anomaly (baseline) |
| 1 | Spike | **Value** | Short (10-25) | 3500 | Traffic spike / DDoS attack |
| 2 | Memory Leak | **Value** | Long (80-150) | 7000 | Gradual memory accumulation |
| 3 | CPU Saturation | **Value** | Medium (40-80) | 4500 | Sustained high CPU load |
| 4 | Network Congestion | **Value** | Medium (30-60) | 4000 | Network bottleneck |
| 5 | Cascading Failure | **Value** | Long (60-120) | 6500 | Error propagation chain |
| 6 | Resource Contention | **Value** | Medium (35-65) | 4500 | Thread/queue competition |
| 7 | Correlation Inversion | **Pattern** | Medium (50-100) | 5000 | CPU-Memory correlation breaks |
| 8 | Temporal Flatline | **Pattern** | Medium (30-60) | 4500 | Values freeze (stuck sensor) |
| 9 | Frequency Shift | **Pattern** | Medium (60-100) | 5500 | Unusual oscillation frequency |

> **Note**: Duration is in timesteps. Interval is the mean number of timesteps between occurrences (before applying `interval_scale`).

---

### 1. Spike (Traffic Spike / DDoS)

**Real-world scenario**: A sudden surge in traffic, such as a DDoS attack or flash crowd event.

**Characteristics**:
- **Duration**: Short (10-25 timesteps)
- **Onset**: Immediate
- **Recovery**: Rapid

**Affected Features**:

| Feature | Effect | Magnitude |
|---------|--------|-----------|
| CPU | Spike | +0.3 to +0.5 |
| Network | Spike | +0.4 to +0.6 |
| ResponseTime | Spike | +0.3 to +0.5 |
| ErrorRate | Spike | +0.2 to +0.4 |
| QueueLength | Spike | +0.3 to +0.5 |

---

### 2. Memory Leak

**Real-world scenario**: A software bug causing gradual memory accumulation without proper deallocation.

**Characteristics**:
- **Duration**: Long (80-150 timesteps)
- **Onset**: Gradual (follows curve: t^0.7)
- **Recovery**: Requires intervention (restart)

**Affected Features**:

| Feature | Effect | Magnitude |
|---------|--------|-----------|
| Memory | Gradual increase | +0.3 to +0.5 (peak) |
| DiskIO | Gradual increase | +0.2 to +0.4 (swapping) |
| ThreadCount | Gradual increase | +0.1 to +0.3 |

---

### 3. CPU Saturation

**Real-world scenario**: A compute-intensive process or runaway thread consuming all available CPU resources.

**Characteristics**:
- **Duration**: Medium (40-80 timesteps)
- **Onset**: Rapid transition to saturated state
- **Recovery**: Gradual (after process completion)

**Affected Features**:

| Feature | Effect | Magnitude |
|---------|--------|-----------|
| CPU | Saturated (0.7-1.0) | +0.4 to +0.6 |
| ThreadCount | Elevated | +0.3 to +0.5 |
| QueueLength | Building up | Linear increase (+0.3 to +0.5 peak) |
| ResponseTime | Elevated | +0.2 to +0.4 |

---

### 4. Network Congestion

**Real-world scenario**: Network bandwidth saturation, packet loss, or upstream provider issues.

**Characteristics**:
- **Duration**: Medium (30-60 timesteps)
- **Onset**: Can be gradual or sudden
- **Recovery**: After congestion clears

**Affected Features**:

| Feature | Effect | Magnitude |
|---------|--------|-----------|
| Network | Saturated | +0.4 to +0.6 |
| ResponseTime | Elevated | +0.3 to +0.5 |
| ErrorRate | Elevated | +0.15 to +0.35 |
| QueueLength | Elevated | +0.2 to +0.4 |

---

### 5. Cascading Failure

**Real-world scenario**: A failure in one component triggers failures in dependent components, creating a chain reaction.

**Characteristics**:
- **Duration**: Long (60-120 timesteps)
- **Onset**: Rapid initial increase, then propagation
- **Recovery**: Requires systematic intervention

**Affected Features** (with temporal propagation):

| Feature | Effect | Timing |
|---------|--------|--------|
| ErrorRate | Spike (cascade curve) | Immediate |
| ResponseTime | Spike | Delayed (1/4 duration) |
| QueueLength | Building up | Delayed (1/3 duration) |
| CPU | Elevated (retries) | Gradual with cascade |

---

### 6. Resource Contention

**Real-world scenario**: Multiple threads or processes competing for shared resources (locks, connections, memory).

**Characteristics**:
- **Duration**: Medium (35-65 timesteps)
- **Onset**: Oscillating pattern
- **Recovery**: After contention resolves

**Affected Features**:

| Feature | Effect | Magnitude |
|---------|--------|-----------|
| ThreadCount | Oscillating | +0.3 to +0.5 (amplitude) |
| QueueLength | Oscillating | +0.3 to +0.5 (amplitude) |
| CPU | Oscillating | +0.2 to +0.4 (amplitude) |
| Memory | Elevated | +0.15 to +0.3 |

---

## Pattern-Based Anomalies (Types 7-9)

These anomalies **maintain normal value ranges** (0.15-0.85) but break temporal or correlation patterns. They help evaluate if the model is detecting based on unusual PATTERNS rather than just unusual VALUES.

### 7. Correlation Inversion

**Real-world scenario**: Database query cache misconfiguration causing Memory to decrease when CPU increases (opposite of normal positive correlation).

**Characteristics**:
- **Duration**: Medium (50-100 timesteps)
- **Onset**: Gradual correlation shift
- **Recovery**: When configuration is fixed
- **Value Range**: Stays within 0.15-0.85

**Pattern Break**:

| Feature Pair | Normal Correlation | Anomaly Correlation |
|--------------|-------------------|---------------------|
| CPU ↔ Memory | Positive (+) | Inverted (-) |
| CPU ↔ ThreadCount | Positive (+) | Inverted (-) |

**Implementation**:
```python
# Invert CPU-Memory correlation
cpu_deviation = signals[start:end, 0] - local_mean
signals[start:end, 1] = local_mean - cpu_deviation * 0.8
```

---

### 8. Temporal Flatline

**Real-world scenario**: Metric collection failure or stuck sensor where values freeze at last reading.

**Characteristics**:
- **Duration**: Medium (30-60 timesteps)
- **Onset**: Instantaneous freeze
- **Recovery**: When sensor/collection is fixed
- **Value Range**: Stays within normal range (frozen at pre-anomaly value)

**Pattern Break**:

| Aspect | Normal | Anomaly |
|--------|--------|---------|
| Temporal variance | Present | Zero (flat) |
| Features affected | - | 3-5 random features |

**Implementation**:
```python
# Freeze selected features at their start values
for feat in features_to_freeze:
    signals[start:end, feat] = signals[start, feat]
```

---

### 9. Frequency Shift

**Real-world scenario**: Wrong cron interval or abnormal scheduling causing unusual periodicity in values.

**Characteristics**:
- **Duration**: Medium (60-100 timesteps)
- **Onset**: Gradual transition
- **Recovery**: When scheduling is corrected
- **Value Range**: Stays within normal range

**Pattern Break**:

| Aspect | Normal | Anomaly |
|--------|--------|---------|
| Oscillation frequency | Normal (low) | 2.5-4x higher |
| Amplitude | Normal | Same |
| Phase | Consistent | Random shift |

**Implementation**:
```python
# Replace with higher frequency oscillation
freq_multiplier = random.uniform(2.5, 4.0)
t = np.linspace(0, freq_multiplier * np.pi, length)
signals[start:end, feat] = local_mean + local_std * np.sin(t + phase)
```

---

### Feature Impact Matrix

#### Value-Based Anomalies (Types 1-6)

| Feature | Spike | MemLeak | CPUSat | NetCong | Cascade | Contention |
|---------|:-----:|:-------:|:------:|:-------:|:-------:|:----------:|
| CPU | +++ | - | ++++ | - | ++ | ++ |
| Memory | - | ++++ | - | - | - | ++ |
| DiskIO | - | +++ | - | - | - | - |
| Network | ++++ | - | - | ++++ | - | - |
| ResponseTime | +++ | - | ++ | +++ | +++ | - |
| ThreadCount | - | ++ | +++ | - | - | +++ |
| ErrorRate | ++ | - | - | ++ | ++++ | - |
| QueueLength | +++ | - | +++ | ++ | +++ | +++ |

Legend: `-` = not affected, `+` = slight, `++` = moderate, `+++` = strong, `++++` = severe

#### Pattern-Based Anomalies (Types 7-9)

| Feature | CorrInversion | Flatline | FreqShift |
|---------|:-------------:|:--------:|:---------:|
| CPU | ◇ | ○ | ∿ |
| Memory | ◇ | ○ | ∿ |
| DiskIO | - | ○ | ∿ |
| Network | - | ○ | ∿ |
| ResponseTime | - | ○ | - |
| ThreadCount | ◇ | ○ | - |
| ErrorRate | - | ○ | - |
| QueueLength | - | ○ | - |

Legend: `-` = not affected, `◇` = correlation inverted, `○` = frozen (3-5 random), `∿` = frequency shifted

---

## Configuration Parameters

```python
from mae_anomaly import Config

config = Config()

# Key dataset parameters
config.seq_length = 100                    # Window size
config.num_features = 8                    # Number of features
config.sliding_window_total_length = 2200000  # Total time series length
config.sliding_window_stride = 10          # Stride (90% overlap)
config.anomaly_interval_scale = 1.5        # Controls anomaly density
config.mask_last_n = 10                    # Last N timesteps for labeling

# Test set target ratios (for downsampling)
config.test_ratio_pure_normal = 0.65      # 65%
config.test_ratio_disturbing_normal = 0.15  # 15%
config.test_ratio_anomaly = 0.25           # 25%
```

### Anomaly Type Configurations

```python
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

    # === Pattern-based anomalies (types 7-9) ===
    # correlation_inversion: Medium duration to show pattern change
    7: {'length_range': (50, 100), 'interval_mean': 5000},

    # temporal_flatline: Sudden freeze, medium duration
    8: {'length_range': (30, 60), 'interval_mean': 4500},

    # frequency_shift: Needs enough length to show frequency change
    9: {'length_range': (60, 100), 'interval_mean': 5500},
}
```

The `interval_scale` parameter (default: 1.5) globally scales all intervals:
```
effective_interval = interval_mean * interval_scale
```

---

## Usage

### Basic Usage

```python
from mae_anomaly import (
    Config,
    SlidingWindowTimeSeriesGenerator,
    SlidingWindowDataset
)

config = Config()

# Generate long time series
generator = SlidingWindowTimeSeriesGenerator(
    total_length=config.sliding_window_total_length,
    num_features=config.num_features,
    interval_scale=config.anomaly_interval_scale,
    seed=config.random_seed
)
signals, point_labels, anomaly_regions = generator.generate()

# Create train dataset
train_dataset = SlidingWindowDataset(
    signals=signals,
    point_labels=point_labels,
    anomaly_regions=anomaly_regions,
    window_size=config.seq_length,
    stride=config.sliding_window_stride,
    mask_last_n=config.mask_last_n,
    split='train',
    train_ratio=0.5,
    seed=config.random_seed
)

# Create test dataset with target counts
test_dataset = SlidingWindowDataset(
    signals=signals,
    point_labels=point_labels,
    anomaly_regions=anomaly_regions,
    window_size=config.seq_length,
    stride=config.sliding_window_stride,
    mask_last_n=config.mask_last_n,
    split='test',
    train_ratio=0.5,
    target_counts={
        'pure_normal': int(num_test * config.test_ratio_pure_normal),
        'disturbing_normal': int(num_test * config.test_ratio_disturbing_normal),
        'anomaly': int(num_test * config.test_ratio_anomaly)
    },
    seed=config.random_seed
)
```

### DataLoader Usage

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for batch in train_loader:
    sequence, label, point_labels, sample_type, anomaly_type = batch
    # sequence: (batch, 100, 8)
    # label: (batch,) - 0 or 1
    # point_labels: (batch, 100) - per-timestep labels
    # sample_type: (batch,) - 0, 1, or 2
    # anomaly_type: (batch,) - 0-6
```

### Analyzing Anomaly Distribution

```python
# After creating dataset
distribution = dataset.get_anomaly_type_distribution()
for name, count in distribution.items():
    print(f"{name}: {count}")

ratios = dataset.get_ratios()
print(f"Pure Normal: {ratios['pure_normal']:.1%}")
print(f"Disturbing Normal: {ratios['disturbing_normal']:.1%}")
print(f"Anomaly: {ratios['anomaly']:.1%}")
```

---

## File Structure

```
mae_anomaly/
├── dataset_sliding.py   # Sliding window dataset
│   ├── FEATURE_NAMES           # 8 feature names
│   ├── ANOMALY_TYPE_NAMES      # 11 anomaly type names (0=normal, 1-7=value, 8-10=pattern)
│   ├── ANOMALY_CATEGORY        # Category mapping: 'value' or 'pattern'
│   ├── ANOMALY_TYPE_CONFIGS    # Per-type length/interval configs
│   ├── SlidingWindowTimeSeriesGenerator  # Long series generator
│   └── SlidingWindowDataset    # Window extraction + labeling
└── config.py            # Configuration with dataset parameters
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Model architecture documentation
- [ABLATION_STUDIES.md](ABLATION_STUDIES.md) - Experiment configurations
