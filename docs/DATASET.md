# Dataset Documentation

**Last Updated**: 2026-01-23

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
   ├── Stride: 10 (90% overlap)
   └── Total windows: ~220,000

3. Train/Test Split
   ├── Train: First 50% (~110,000 samples)
   └── Test: Last 50% (downsampled to 2,000)

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

### Test Set (Downsampled)

| Sample Type | Count | Ratio |
|-------------|-------|-------|
| Pure Normal | 1,200 | 60% |
| Disturbing Normal | 300 | 15% |
| Anomaly | 500 | 25% |
| **Total** | **2,000** | 100% |

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
| Value range | Natural (no clipping) | Anomalies push to higher ranges |
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

The dataset includes **7 distinct anomaly types** commonly observed in server monitoring systems.

### Anomaly Type Summary

| ID | Name | Duration | Interval | Description |
|----|------|----------|----------|-------------|
| 0 | Normal | - | - | No anomaly (baseline) |
| 1 | Spike | Short (10-25) | 3500 | Traffic spike / DDoS attack |
| 2 | Memory Leak | Long (80-150) | 7000 | Gradual memory accumulation |
| 3 | CPU Saturation | Medium (40-80) | 4500 | Sustained high CPU load |
| 4 | Network Congestion | Medium (30-60) | 4000 | Network bottleneck |
| 5 | Cascading Failure | Long (60-120) | 6500 | Error propagation chain |
| 6 | Resource Contention | Medium (35-65) | 4500 | Thread/queue competition |
| 7 | **Point Spike** | **Very Short (3-5)** | 4000 | **True point anomaly** |

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

### 7. Point Spike (True Point Anomaly)

**Real-world scenario**: A brief sensor glitch, momentary hardware fault, or single-event upset.

**Characteristics**:
- **Duration**: Very short (3-5 timesteps)
- **Onset**: Instantaneous
- **Recovery**: Immediate
- **Unique**: 2+ random features spike simultaneously

**Affected Features**:

| Feature | Effect | Selection |
|---------|--------|-----------|
| 2+ Random Features | Spike | Random selection from all 8 features |
| Spike magnitude | +0.3 to +0.6 per feature | Same magnitude for all selected |

> **Note**: Unlike other anomaly types where specific features are always affected, Point Spike randomly selects 2 or more features to spike. This makes it harder to detect using simple threshold methods on individual features.

---

### Feature Impact Matrix

| Feature | Spike | MemLeak | CPUSat | NetCong | Cascade | Contention | PointSpike |
|---------|:-----:|:-------:|:------:|:-------:|:-------:|:----------:|:----------:|
| CPU | +++ | - | ++++ | - | ++ | ++ | ? |
| Memory | - | ++++ | - | - | - | ++ | ? |
| DiskIO | - | +++ | - | - | - | - | ? |
| Network | ++++ | - | - | ++++ | - | - | ? |
| ResponseTime | +++ | - | ++ | +++ | +++ | - | ? |
| ThreadCount | - | ++ | +++ | - | - | +++ | ? |
| ErrorRate | ++ | - | - | ++ | ++++ | - | ? |
| QueueLength | +++ | - | +++ | ++ | +++ | +++ | ? |

Legend: `-` = not affected, `+` = slight, `++` = moderate, `+++` = strong, `++++` = severe, `?` = random (2+ features)

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

# Test set target counts
config.test_target_pure_normal = 1200
config.test_target_disturbing_normal = 300
config.test_target_anomaly = 500
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

    # point_spike: Very short point anomaly (3-5 timesteps, 2+ random features)
    7: {'length_range': (3, 5), 'interval_mean': 4000},
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
        'pure_normal': config.test_target_pure_normal,
        'disturbing_normal': config.test_target_disturbing_normal,
        'anomaly': config.test_target_anomaly
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
│   ├── ANOMALY_TYPE_NAMES      # 7 anomaly type names (including 'normal')
│   ├── ANOMALY_TYPE_CONFIGS    # Per-type length/interval configs
│   ├── SlidingWindowTimeSeriesGenerator  # Long series generator
│   └── SlidingWindowDataset    # Window extraction + labeling
└── config.py            # Configuration with dataset parameters
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Model architecture documentation
- [ABLATION_STUDIES.md](ABLATION_STUDIES.md) - Experiment configurations
