# Dataset Documentation

**Last Updated**: 2026-05-30

---

## Overview

This project uses a **Sliding Window Time Series Dataset** that simulates server monitoring data. The dataset is designed to be:

1. **Patch-size independent**: Anomaly positions are fixed regardless of model configuration
2. **Temporally continuous**: Samples come from a single long time series
3. **Realistic**: Train/test split by time (no data leakage)

---

## Dataset Generation Process

```
1. Generate Long Time Series (275K timesteps, configurable)
   ├── 8 correlated features (server metrics)
   └── Inject anomalies at random intervals

2. Sliding Window Extraction
   ├── Window size: 500 timesteps (configurable)
   ├── Train stride: Configurable (default 21)
   ├── Test stride: Configurable (default 21)
   ├── Epoch offset: Random train window offset per epoch (default off)
   └── Total windows: varies based on window size and stride

3. Train/Test Split
   ├── Train: First 80% (220K timesteps, ~5% anomaly)
   └── Test: Last 20% (55K timesteps, stride=21, no downsampling)

4. Labeling
   ├── Check last patch_size timesteps
   └── Classify as: pure_normal, disturbing_normal, or anomaly
```

---

## Sample Types

| Type | Label | Description |
|------|-------|-------------|
| **Pure Normal** | 0 | No anomaly anywhere in the window |
| **Disturbing Normal** | 0 | Anomaly exists but NOT in last patch_size timesteps |
| **Anomaly** | 1 | Anomaly exists in last patch_size timesteps |

### Why "Disturbing Normal"?

This represents a challenging case where:
- The window contains anomalous patterns earlier in the sequence
- But the evaluation region (last patch_size timesteps, default 5) is normal
- Tests if the model correctly ignores past anomalies when predicting the masked region

---

## Dataset Statistics

### Train Set (Natural Distribution)

| Sample Type | Approx. Ratio |
|-------------|---------------|
| Pure Normal | ~88% |
| Disturbing Normal | ~7% |
| Anomaly | ~5% |

### Test Set (Full, Stride=21)

> **Note**: As of the latest update, test set uses stride=21 and no downsampling by default.
> This ensures proper point-level PA%K evaluation with overlapping windows while reducing compute.

| Aspect | Value |
|--------|-------|
| Stride | 21 (default) |
| Downsampling | Disabled by default |
| Total windows | ~55,000 (20% of time series) |

**Legacy mode** (downsampled, for backwards compatibility):

| Sample Type | Count | Ratio |
|-------------|-------|-------|
| Pure Normal | 1,200 | 60% |
| Disturbing Normal | 300 | 15% |
| Anomaly | 500 | 25% |
| **Total** | **2,000** | 100% |

---

## Point-Level PA%K Evaluation

With stride=21 sliding windows, each timestep is covered by multiple windows' last patches.
The evaluation aggregates window-level scores to point-level using one of four methods:

### Aggregation Methods

| Method | Description | Formula | Characteristics |
|--------|-------------|---------|-----------------|
| **Voting** (default) | Majority vote of binary predictions | `1 if votes > n/2 else 0` | Robust to outliers |
| **Mean** | Average of window scores | `mean(scores)` | Balanced aggregation |
| **Median** | Median of window scores | `median(scores)` | Robust to outliers |
| **Max** | Maximum of window scores | `max(scores)` | Most sensitive, catches any anomaly signal |

### Window Coverage

For `seq_length=500` and `patch_size=5`:

```
Window w's last patch: timesteps [w+495, w+499]

Timestep t is covered by windows where:
  w+495 ≤ t ≤ w+499
  → w ∈ [t-499, t-495]
  → Up to 5 windows per timestep

Coverage by position:
  - Timesteps 0-494: Not covered (not in any last patch)
  - Timesteps 495-499: 1-5 windows
  - Timesteps 500+: 5 windows (full coverage)
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

Normalization is performed by `SlidingWindowDataset` using **per-feature z-score standardization fitted on the train portion only**. This follows the standard practice in time series anomaly detection (Anomaly Transformer, TimesNet, etc.) and prevents data leakage from test into normalization statistics.

```python
def _standardize_per_feature(signals, train_end):
    """Per-feature z-score standardization fitted on train portion only."""
    train_signals = signals[:train_end]
    scaler_mean = train_signals.mean(axis=0)   # (num_features,)
    scaler_std = train_signals.std(axis=0)     # (num_features,)
    scaler_std[scaler_std < 1e-8] = 1.0        # Protect constant features
    return (signals - scaler_mean) / scaler_std
```

**Key design decisions:**

| Aspect | Previous (min-max [0,1]) | Current (z-score, train-only fit) |
|--------|--------------------------|----------------------------------|
| Fit data | Entire series (train+test) | Train portion only |
| Data leakage | Yes (test stats leak) | No |
| Output range | Bounded [0, 1] | Unbounded (mean=0, std=1) |
| Anomaly sensitivity | Compressed by outlier min/max | Naturally amplified (deviations in σ) |
| Swap experiment support | Re-normalize needed | Automatic (SlidingWindowDataset handles) |

**Why z-score over min-max?**
1. **No data leakage**: Scaler statistics are computed from train data only
2. **Anomaly amplification**: Anomalous values naturally produce large z-scores (many σ from mean)
3. **Model compatibility**: Linear output projection (unbounded) matches z-score's unbounded range
4. **Community standard**: Used by Anomaly Transformer (ICLR'22), TimesNet (ICLR'23)

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
| Value range | Per-feature z-score (train-only fit) | Relative magnitudes preserved |
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
config.seq_length = 500                    # Window size
config.num_features = 8                    # Number of features
config.sliding_window_total_length = 275000   # Total time series length (220K train + 55K test)
config.sliding_window_stride = 21             # Train stride
config.epoch_offset = False                    # Non-replacement random train window offset (True=enabled)
config.anomaly_interval_scale = 0.75       # Controls anomaly density (2x frequency, ~13% anomaly)
config.patch_size = 5                      # Patch size (also used for window labeling)

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
    mask_last_n=config.patch_size,
    split='train',
    train_ratio=config.sliding_window_train_ratio,
    seed=config.random_seed
)

# Create test dataset with target counts
test_dataset = SlidingWindowDataset(
    signals=signals,
    point_labels=point_labels,
    anomaly_regions=anomaly_regions,
    window_size=config.seq_length,
    stride=config.sliding_window_stride,
    mask_last_n=config.patch_size,
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

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

for batch in train_loader:
    sequence, label, point_labels, sample_type, anomaly_type = batch
    # sequence: (batch, 500, 8)
    # label: (batch,) - 0 or 1
    # point_labels: (batch, 500) - per-timestep labels
    # sample_type: (batch,) - 0, 1, or 2
    # anomaly_type: (batch,) - 0-9 (0=normal, 1-6=value, 7-9=pattern)
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
│   ├── ANOMALY_TYPE_NAMES      # 10 anomaly type names (0=normal, 1-6=value, 7-9=pattern)
│   ├── ANOMALY_CATEGORY        # Category mapping: 'value' or 'pattern'
│   ├── ANOMALY_TYPE_CONFIGS    # Per-type length/interval configs
│   ├── SlidingWindowTimeSeriesGenerator  # Long series generator
│   └── SlidingWindowDataset    # Window extraction + labeling
└── config.py            # Configuration with dataset parameters
```

---

## Real-World Datasets

In addition to the simulation dataset, the project supports several real-world time series anomaly detection datasets via the loader registry (`DATASET_LOADERS` in `mae_anomaly/datasets/loaders.py`).

### Available Dataset Loaders

| Key | Dataset | Description |
|-----|---------|-------------|
| `simulation` | Simulation | 8-feature server monitoring (no complexity) |
| `simulation_complex` | Simulation | Same with normal data complexity enabled |
| `swat_A1A2` | SWaT | A1 (normal) + A2 (attack), standard split |
| `swat_A1A2_swap` | SWaT | A1 + A2 with swapped halves |
| `wadi_14days_A1` | WaDi A1 | 14days normal + attack data |
| `wadi_14days_A2` | WaDi A2 | 14days normal + attack data |
| `tep` | TEP | All 20 fault types |
| `tep_faultN` | TEP | Single fault type N (1-20) |
| `smd` | SMD | All 28 server machines |
| `smd_machine-X-Y` | SMD | Single machine (e.g. `smd_machine-1-1`) |
| `PSM` | PSM | Pooled Server Metrics (eBay, KDD 2021), 25 features, single contiguous stream |

### TEP (Tennessee Eastman Process)

**Source**: Rieth et al. (2017), originally from Downs & Vogel (1993)

**Data Structure**:
- 52 process variables (41 measurements + 11 manipulated variables)
- 20 fault types (step changes, random variations, slow drifts, sticking valves)
- 500 independent simulation runs per condition
- Training runs: 500 samples/run (25 hours), Testing runs: 960 samples/run (48 hours)
- Fault onset: sample 160 in testing data (8 hours into simulation)

**Loader Design** (`load_tep()`):
- Training data: Fault-free testing runs (960 samples/run) → all normal
- Testing data: Faulty testing runs (960 samples/run) → fault onset at sample 160
- Configurable number of runs per condition (`n_train_runs`, `n_test_runs`, default 50)
- Per-fault-type evaluation via `anomaly_type` field in `AnomalyRegion`

**Run Boundary Handling**:

TEP consists of independent simulation runs, unlike continuous streams (SWaT/WaDi). Runs are concatenated into a single array but `run_boundaries` are tracked and passed to `SlidingWindowDataset` to prevent windows from crossing run boundaries.

```python
# Loader returns run_boundaries in data_info
signals, labels, regions, features, train_ratio, data_info = load_tep(
    fault_types=[1, 5],  # Specific faults
    n_train_runs=50,     # Fault-free runs for training
    n_test_runs=50,      # Faulty runs per fault type
)
run_boundaries = data_info['run_boundaries']

# SlidingWindowDataset respects boundaries
dataset = SlidingWindowDataset(
    signals=signals, point_labels=labels, anomaly_regions=regions,
    window_size=100, stride=50, mask_last_n=10, split='train',
    train_ratio=train_ratio, run_boundaries=run_boundaries,
)
```

### SMD (Server Machine Dataset)

**Source**: Su et al. (KDD 2019), OmniAnomaly

**Data Structure**:
- 28 independent server machines (machine-1-1 through machine-3-11)
- 38 features per machine (anonymous server metrics)
- Train/test split: ~50/50 per machine (train = all normal, test = with anomalies)
- Total: ~1.4M samples across all machines, ~4.16% test anomaly ratio
- Binary anomaly labels (no anomaly type distinction)

**Loader Design** (`load_smd()`):
- Data concatenated as `[all_train | all_test]` to match pipeline's train_ratio-based split
- `run_boundaries` track machine boundaries to prevent cross-machine windows
- Individual machine loading via `machines` parameter
- Constant column removal (z-score normalization handled by SlidingWindowDataset)

**Registry** (concat / simple, 2026-06-01):
- **`SMD_concat`** (`load_smd_concat`): all 28 machines concatenated, **per-machine test-cut** (orig_train + front-50% test → train, back-50% test → test) — matches SMAP/MSL/Exathlon concat. `run_boundaries` mark every machine boundary AND each orig_train/test_front seam. **Per-machine normalization (2026-06-02)**: emits `entity_norm_segments` so each machine is scaler-fit on its own train portion (not one whole-array fit over all 28 machines — see SMAP/MSL normalization note).
- **`SMD_simple_<machine>`** (`load_smd_simple`): one machine = one dataset (test-cut). 28 keys. (run_base `DATASETS` uses these; `results_subdir` stays `SMD/<machine>`.)
- Legacy: `smd` = all machines with the **ORIGINAL** train/test split (no test-cut); `smd_machine-X-Y` = single machine original split.

Boundary safety (verified): no sliding window crosses a machine boundary, an orig_train/test_front seam, or the train|test split — empirically **0 crossing windows** across train+test (see CHANGELOG 2026-06-01).

```python
# Load all 28 machines
signals, labels, regions, features, train_ratio, data_info = load_smd()

# Load specific machines
signals, labels, regions, features, train_ratio, data_info = load_smd(
    machines=['machine-1-1', 'machine-2-1']
)

# Per-machine evaluation (for rank averaging)
for machine_id in SMD_MACHINE_NAMES:
    loader = get_dataset_loader(f'smd_{machine_id}')
    signals, labels, regions, features, ratio, info = loader()
```

### PSM (Pooled Server Metrics, eBay)

**Source**: Abdulaal et al. (KDD 2021), eBay Application Performance Management

**Data Structure**:
- Single contiguous stream from anonymized eBay server pool (no per-machine separation, unlike SMD)
- 25 anonymized server metrics (`feature_0` ~ `feature_24`)
- Train file: 132,481 samples (all normal)
- Test file: 87,841 samples (test anomaly ratio ~27.76%, 72 anomaly regions)
- Sampling: 1 minute
- Anomaly regions: NOT individually documented (anonymous incidents)
- NaN values present in train (~4,195) — handled by forward/backward-fill

**Loader Design** (`load_psm()`):
- Same simple 50/50 split pattern as `load_smd_simple`:
  - Train = original train file (132,481, all normal) + front 50% of test file (43,920)
  - Test  = back 50% of test file (43,921)
- `run_boundaries = [len(train_data)]` — windows must not cross orig_train / test_front boundary
- Constant column removal (typically 0 columns removed for PSM)
- Train-only z-score normalization handled by `SlidingWindowDataset`

**Registry**:
- `PSM`: Single key in `DATASET_LOADERS` (no per-machine variants — PSM is a single stream)

```python
from mae_anomaly.datasets.loaders import load_psm

signals, labels, regions, features, train_ratio, data_info = load_psm()
# signals: (220322, 25) float32
# labels:  (220322,) int64, anom=24,381
# regions: 72 AnomalyRegion objects
# train_ratio: 0.8007  (176,401 train / 43,921 test)
# data_info['run_boundaries']: [132481]
```

**Data Files** (`dataset/PSM/`):
```
dataset/PSM/
├── train.csv          # timestamp_(min) + feature_0 ~ feature_24 (132,481 rows)
├── test.csv           # timestamp_(min) + feature_0 ~ feature_24 (87,841 rows)
├── test_label.csv     # timestamp_(min) + label                 (87,841 rows)
└── LICENSE            # BSD-3-Clause (eBay)
```

Source: [github.com/eBay/RANSynCoders](https://github.com/eBay/RANSynCoders) (official eBay release).

---

## Exathlon Dataset (Spark Cluster Anomaly Benchmark)

**Source**: Jacob et al., "Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series", VLDB 2021. [arXiv:2010.05073](https://arxiv.org/abs/2010.05073) · [GitHub](https://github.com/exathlonbenchmark/exathlon)

**Construction**: Real data traces from repeated executions of 10 distributed streaming applications on a 4-node Apache Spark cluster over 2.5 months, sampled at 1 Hz.

### Raw Dataset Specs

| Item | Value |
|------|-------|
| Total traces (after curation) | 93 |
| Total data points | 2,335,781 (≈ 649 hours) |
| Raw features per trace | 2,283 metrics |
| Anomaly types | 6 (T1 Bursty input, T2 Bursty until crash, T3 Stalled input, T4 CPU contention, T5 Driver failure, T6 Executor failure) |
| Total anomaly instances | 97 (main) + 12 unknown side anomalies |
| Raw data size | 24.6 GB |

### Feature Reduction: FScustom 19 features

This project uses **FScustom** (manually selected by domain knowledge in the Exathlon paper) — defined as "upper-bound performance" reference in the paper:

| Group | Count | Description |
|-------|:-:|-----|
| 1. Identity (raw) | 3 | Driver streaming delays: processingDelay, schedulingDelay, totalDelay |
| 2. 1-Difference | 8 | Driver counters (×4) + Driver memory + JVM heap + 4× node CPU idle |
| 3. Executor average + 1-Difference | 6 | 5-executor avg then diff for: hdfs_write_ops, cpuTime, runTime, shuffleRead, shuffleWritten, jvm_heap_used |
| **Total** | **19** | |

Preprocessing script: [`dataset/Exathlon/preprocess.py`](../dataset/Exathlon/preprocess.py)
- Downloads all 93 traces from GitHub (handles flat + split-zip layouts via 7z)
- Extracts 19 features per trace
- Generates point-level binary labels (anomaly = RCI ∪ EEI)
- Saves to `dataset/Exathlon/app{N}/{trace_name}.csv` (~175 MB total after reduction)

### App-based Evaluation Convention

Apps used: **{1, 2, 4, 5, 6, 9}** (6 apps, following TimeSeAD 6-app subset)
- App 7 excluded: no disturbed traces (test impossible)
- App 8 excluded: no undisturbed traces (train impossible)
- App 3, 10 retained (paper excludes for distributional shift, but kept here per user spec)

Per-app train/test split:
- **Train** = all undisturbed traces (concat by trace_id) + first `floor(N_dist/2)` disturbed traces
- **Test** = remaining disturbed traces

Sliding windows respect `run_boundaries` (trace boundaries within both train and test portions).

**Registry** (concat / simple, 2026-06-01):
- **`Exathlon_concat`** (`load_exathlon_concat`): all 6 apps concatenated into one stream. Each app is loaded via `load_exathlon(app)`, split into its train/test by `train_ratio`, then re-merged as `[all_app_train | all_app_test]`. `run_boundaries` merge every app's internal **trace** boundaries (mapped into the global train/test blocks) PLUS every app boundary. 19 features (all apps). **Per-app normalization (2026-06-02)**: emits `entity_norm_segments` so each app is scaler-fit on its own train portion (not one whole-array fit over all apps — see SMAP/MSL normalization note).
- **`Exathlon_simple_app<N>`** (`load_exathlon`): one app = one dataset. 6 keys. (run_base `DATASETS` uses these; `results_subdir` stays `Exathlon/app<N>`.) Legacy alias `exathlon_app<N>` retained.
- Boundary safety (verified): no window crosses a trace, app, or train|test boundary — **0 crossing windows** (see CHANGELOG 2026-06-01).

### Per-app Statistics (after split)

| App | Total Rows | Train Rows | Test Rows | Train Anom% | Test Anom% | #Anomaly Regions | #Train Traces | #Test Traces |
|:---:|:----------:|:----------:|:---------:|:-----------:|:----------:|:----------------:|:-------------:|:------------:|
| app1 | 90,897 | 44,192 | 46,705 | 5.24% | 13.24% | 9 | 7 | 2 |
| app2 | 164,950 | 118,230 | 46,720 | 3.89% | 26.46% | 9 | 5 | 2 |
| app4 | 340,994 | 337,373 | 3,621 | 4.77% | 17.34% | 11 | 7 | 1 |
| app5 | 322,775 | 269,387 | 53,388 | 5.26% | 7.29% | 21 | 11 | 4 |
| app6 | 399,102 | 348,832 | 50,270 | 1.39% | 8.84% | 11 | 13 | 2 |
| app9 | 375,594 | 326,571 | 49,023 | 2.25% | 12.47% | 14 | 8 | 2 |

### Usage

```python
from mae_anomaly.datasets.loaders import load_exathlon, EXATHLON_APP_IDS

# Load a single app
signals, labels, regions, features, train_ratio, data_info = load_exathlon(app=1)
# signals: (90897, 19) float32
# labels:  (90897,) int64
# regions: 9 AnomalyRegion objects
# train_ratio: 0.4862
# data_info['run_boundaries']: list of trace boundaries (7 for app1)
```

Comparison pipeline:
```python
from comparison.data.unified_loader import UnifiedLoader

loader = UnifiedLoader(
    dataset='exathlon',
    app=1,
    normalize_mode='minmax',  # or 'zscore'
    variant=None,             # or 'normalonly' (Q3/Q4)
).load()
```

DATASET_LOADERS keys: `exathlon_app1`, `exathlon_app2`, `exathlon_app4`, `exathlon_app5`, `exathlon_app6`, `exathlon_app9`.

### Evaluation Protocol

Each app evaluated independently. Final metrics reported as **mean across 6 apps** (following TimeSeAD's per-app aggregation pattern).

---

## NASA SMAP / MSL (Telemanom, Hundman et al. KDD 2018)

NASA Soil Moisture Active Passive (SMAP) satellite + Mars Science Laboratory (MSL) Curiosity rover telemetry anomaly benchmark — Telemanom dataset.

### Source / Citation

- **Paper**: Hundman, Kyle; Constantinou, Valentino; Laporte, Christopher; Colwell, Ian; Soderstrom, Tom. *"Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"*. Proc. 24th ACM SIGKDD KDD 2018, **pp. 387–395**. DOI [`10.1145/3219819.3219845`](https://doi.org/10.1145/3219819.3219845) · arXiv [`1802.04431`](https://arxiv.org/abs/1802.04431).
- **Official repo**: <https://github.com/khundman/telemanom> (Apache 2.0 for code; data are NASA-derived telemetry, repo does not explicitly state a data license; commonly redistributed as a public benchmark).
- **Canonical data URL**: `https://s3-us-west-2.amazonaws.com/telemanom/data.zip` (currently HTTP 403). The Telemanom README now points to a Kaggle mirror (`patrickfleith/nasa-anomaly-detection-dataset-smap-msl`).
- **Labels CSV**: `https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv`.
- **Used in this repo**: Wayback Machine snapshot `http://web.archive.org/web/20221016205142/http://s3-us-west-2.amazonaws.com/telemanom/data.zip` (2022-10-16, 85,899,803 bytes compressed → 272 MB / 417 npy entries).
- **Distinct from**: NSIDC's NASA SMAP science archive (L-band radiometer/radar L1–L4 products). Telemanom is the *engineering telemetry* benchmark, not raw mission science.

### Channels & dimensions

| Spacecraft | Unique channels | Feature dim | Notes |
|---|---|---|---|
| SMAP | **54** | **25** (1 telemetry + 24 commanded actions) | `labeled_anomalies.csv` has 55 rows — P-2 appears twice with different anomaly_sequences |
| MSL  | **27** | **55** | P-2 is a SMAP channel; absent from MSL data |

**P-2 duplicate handling** (4 known variants in the literature — see `temp/msl_smap_pattern_ab_0526/01_SOURCE_AND_LABEL_CHECK.md`):
- **This repo**: UNION of the two annotated intervals (`[5300, 6575]`) — conservative
- OmniAnomaly (KDD'19): explicit exclude
- TranAD (VLDB'22): silent overwrite (second row wins)
- QuoVadis (ICML'24): MSL `P-2_` removed; SMAP P-2 unspecified

### Split rule (both patterns)

Per channel:
- `train_portion = orig_train.npy (all normal) + test_front_50%` (chronological)
- `test_portion  = test_back_50%`
- 50% split point is pushed outside any anomaly region by ±10 timestamps (`_find_safe_cut_point`, reused from SMD). Verified: **0 boundary-straddling anomalies** across all 54 + 27 channels.

### Runnable dataset keys (registry, 2026-06-01)

`DATASET_LOADERS` (and run_base `--list`/`--dataset`) expose two patterns:
- **`SMAP_concat` / `MSL_concat`** — all channels time-concatenated into one multivariate stream; `run_boundaries` mark every channel + the train|test seam so windows never cross segments. SMAP=54ch×25feat, MSL=27ch×55feat.
- **`SMAP_simple_<ch>` / `MSL_simple_<ch>`** — one channel = one dataset (e.g. `SMAP_simple_A-1`, `MSL_simple_C-1`); SMD-style. 54 + 27 = 81 keys.

**Normalization is leakage-free AND per-entity (2026-06-02)**: each channel is normalized using statistics fitted on **its own train portion only** (`SlidingWindowDataset(entity_segments=...)` → `_normalize_per_entity`), then concatenated. The held-out `test_back_50%` (eval set) never enters any fit; `test_front_50%` is part of train *by design* (chronological prefix), not leakage. **Previously this was a single cross-channel (whole-array) min/max fit over all channels' pooled train portions — fixed 2026-06-02**: whole-array fit mixed channels of differing scale so the dominant normalized signal became "which channel" rather than "is this anomalous", crushing small-magnitude channels' intra-channel variation. The concat loaders now emit `data_info['entity_norm_segments']` (per-channel `(train_len, test_len)`) and `SlidingWindowDataset` fits each channel independently. z-score (`_standardize_per_feature`, computed in float64) and minmax behave identically per entity. Single-entity datasets (SWaT/WaDi/PSM) keep the whole-array path (one entity → equivalent).

### Pattern A — all-channels concat (legacy convention)

**Loaders**: `load_smap_combined()`, `load_msl_combined()`. **Entries**: `smap`, `smap_normalonly`, `msl`, `msl_normalonly` (4 total).

Channels are time-concatenated into a single 2D stream. `data_info['run_boundaries']` lists every discontinuity (intra-channel `orig_train ↔ test_front` junction + inter-channel boundaries). **As of 2026-06-02 the scaler is fit PER CHANNEL on each channel's own train portion** (`entity_norm_segments` → `_normalize_per_entity`), not a single cross-channel pooled fit. (The earlier whole-array fit followed the Anomaly-Transformer / TimesNet / DCdetector mirror but mixed channel scales — see the normalization note above.)

`run_boundaries` + `compute_segment_safe_window_indices` guarantee windows never cross channel boundaries. With per-entity normalization, a normalized value now has a consistent within-channel meaning (each channel is centered/scaled on itself).

| Spacecraft | Total | Train | Test | train_ratio | train anom% | test anom% | run_boundaries | safe-cut moved |
|---|---|---|---|---|---|---|---|---|
| SMAP | 573,830 | 355,905 | 217,925 | 0.6202 | 0.70% | 24.54% | 161 | 0/54 |
| MSL  | 132,046 |  95,271 |  36,775 | 0.7215 | 1.70% | 16.72% |  80 | 4/27 (D-16, M-1, M-2, S-2) |

### Pattern B — per-channel (SMD/Exathlon-style, OmniAnomaly/Telemanom entity-level)

**Loaders**: `load_smap_simple(channel)`, `load_msl_simple(channel)`. **Entries** (dynamic): `smap_{ch}` + `smap_{ch}_normalonly` × 54 + `msl_{ch}` + `msl_{ch}_normalonly` × 27 = **162 total** (SMD `smd_{machine}` pattern).

Each entry loads a single channel only. `UnifiedLoader` fits min-max / z-score **on this single channel's train portion** — per-entity scaler, matching the per-machine SMD and per-app Exathlon conventions, and the entity-level treatment in the original Telemanom and OmniAnomaly papers. `run_boundaries = [len(orig_train)]` (one intra-channel junction).

| Aggregate over per-channel sums | SMAP | MSL |
|---|---|---|
| total samples | 573,830 | 132,046 |
| train anomaly points | 2,499 | 1,616 |
| test anomaly points | 53,473 | 6,150 |
| channels with safe-cut moved | 0/54 | 4/27 |

Pattern A vs Pattern B use the **same raw npy + same 50/50 PSM-style split + same safe-cut margin** → samples are identical. Differences are limited to (a) scaler fit scope and (b) entry granularity / reporting unit.

### Reporting convention

- Pattern B: per-channel metrics aggregated as **mean across 54 SMAP channels / 27 MSL channels** (SMD/Exathlon convention).
- Pattern A: single global metric per spacecraft (channel mixing inherent).

### Boundary safety

- **Pattern A**: `_apply_normalonly` is aware of `run_boundaries` → anomaly-removal segments never straddle channel/recording joins. `compute_segment_safe_window_indices` drops any window that would cross a boundary (verified on SMAP normalonly: 11,385 / 353,307 = 3.22% windows dropped).
- **Pattern B**: single channel ⇒ no cross-channel risk. The one intra-channel junction at `orig_train ↔ test_front` is registered in `run_boundaries` and handled identically.

### Loaders / dispatch

- Raw loaders: `mae_anomaly/datasets/loaders.py` — `_load_smap_msl_combined` (A), `_load_smap_msl_simple_single` (B), `load_smap_combined`/`load_msl_combined`/`load_smap_simple`/`load_msl_simple` wrappers.
- Channel inventories: `SMAP_CHANNEL_NAMES` (54), `MSL_CHANNEL_NAMES` (27).
- `comparison/data/unified_loader.py` accepts `dataset='smap'|'msl'|'smap_simple'|'msl_simple'` (+ `channel` kwarg for `*_simple`).
- `comparison/experiment_configs.py` generates the 162 Pattern B entries via for-loop over channel name lists (mirrors SMD's `for _machine in SMD_MACHINE_NAMES`).
- **MAE side (registered 2026-06-02)**: SMAP/MSL are registered in `mae_anomaly/datasets/loaders.py:DATASET_LOADERS` — `SMAP_concat`/`MSL_concat` (all-channels stream via `load_smap_combined`/`load_msl_combined`) + `SMAP_simple_<ch>` ×54 / `MSL_simple_<ch>` ×27 (`load_smap_simple`/`load_msl_simple`). `scripts/run_base_experiments.py` registers `SMAP_concat`/`MSL_concat` in the base `DATASETS` list and the per-channel `*_simple` keys in `SMAP_MSL_SIMPLE_DATASETS` (`results_subdir` = `SMAP/concat`·`MSL/concat` and `SMAP/<ch>`·`MSL/<ch>`).

### Per-channel statistics

See `temp/msl_smap_pattern_ab_0526/07_DATASET_STATISTICS.md` for full per-channel rows (54 SMAP + 27 MSL) and `stats_pattern_b_smap.json` / `stats_pattern_b_msl.json` for machine-readable counts.

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Model architecture documentation
- [ABLATION_STUDIES.md](ABLATION_STUDIES.md) - Experiment configurations
