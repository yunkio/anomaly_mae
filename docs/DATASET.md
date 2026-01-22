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

## Anomaly Types

The dataset includes **6 distinct anomaly types** commonly observed in server monitoring systems.

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

### Feature Impact Matrix

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
