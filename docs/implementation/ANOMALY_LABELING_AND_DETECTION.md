# Anomaly Labeling and Detection Mechanism Analysis

## Summary

**Labeling**: Sequence-level (window-based)
**Detection**: Last N time steps masking with discrepancy/reconstruction scoring

---

## 1. Anomaly Labeling Criteria

### Method: Sequence-Level Window-Based Labeling

The anomaly labeling is **sequence-level** (or window-based), not point-level.

### How It Works

From [multivariate_mae_experiments.py:256-295](multivariate_mae_experiments.py#L256-L295):

```python
def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
    """Generate complete multivariate dataset"""
    data = []
    labels = []

    num_anomalies = int(self.num_samples * self.anomaly_ratio)
    num_normal = self.num_samples - num_anomalies

    # Generate normal sequences
    for _ in range(num_normal):
        signals = self._generate_normal_multivariate()
        data.append(signals)
        labels.append(0)  # Entire sequence labeled as 0

    # Generate anomalous sequences
    for _ in range(num_anomalies):
        signals = self._generate_normal_multivariate()
        anomaly_func = np.random.choice(anomaly_funcs)
        signals = anomaly_func(signals)  # Inject anomaly somewhere in sequence
        data.append(signals)
        labels.append(1)  # Entire sequence labeled as 1
```

### Key Points

1. **Each sequence has a single binary label**: 0 (normal) or 1 (anomalous)
2. **The label applies to the entire sequence** of length 100 time steps
3. **If an anomaly is injected anywhere in the sequence**, the whole sequence is labeled as 1
4. **No point-level labels**: We don't track which specific time steps contain the anomaly

### Anomaly Injection Examples

Different anomaly types are injected at different positions:

#### Spike Anomaly ([lines 168-183](multivariate_mae_experiments.py#L168-L183))
```python
margin = min(20, self.seq_length // 4)
spike_pos = np.random.randint(margin, max(margin + 1, self.seq_length - margin))
spike_width = np.random.randint(3, min(10, self.seq_length // 10 + 1))
```
- Injected between time step ~20 and ~75
- Duration: 3-10 time steps
- **Whole sequence labeled as 1**

#### Memory Leak ([lines 185-202](multivariate_mae_experiments.py#L185-L202))
```python
start_pos = np.random.randint(margin, max(margin + 1, self.seq_length // 2))
leak_length = self.seq_length - start_pos
```
- Starts between time step ~20 and ~50
- Continues until end of sequence
- **Whole sequence labeled as 1**

#### Noise Burst ([lines 204-218](multivariate_mae_experiments.py#L204-L218))
```python
noise_start = np.random.randint(margin, max(margin + 1, self.seq_length - max_length))
noise_length = np.random.randint(min_length, max(min_length + 1, max_length))
```
- Duration: 20-40 time steps
- **Whole sequence labeled as 1**

#### Network Congestion ([lines 239-254](multivariate_mae_experiments.py#L239-L254))
```python
change_point = np.random.randint(margin, max(margin + 1, self.seq_length - margin))
signals[change_point:, 3] += np.random.uniform(0.3, 0.5)
```
- Starts between time step ~30 and ~67
- Continues until end
- **Whole sequence labeled as 1**

---

## 2. Anomaly Detection Mechanism

### Method: Last-N Time Steps Masking + Discrepancy/Reconstruction Scoring

From [multivariate_mae_experiments.py:810-849](multivariate_mae_experiments.py#L810-L849):

### How It Works

```python
def compute_anomaly_scores(self) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        for sequences, labels in self.test_loader:
            # Create mask: unmask first (seq_length - mask_last_n) steps,
            # mask last mask_last_n steps
            mask = torch.ones(batch_size, seq_length, device=self.config.device)
            mask[:, -self.config.mask_last_n:] = 0  # Mask last N steps

            # Forward pass with fixed mask (masking_ratio=0.0 disables random masking)
            teacher_output, student_output, _ = self.model(sequences, masking_ratio=0.0, mask=mask)

            masked_positions = (mask == 0).unsqueeze(-1)  # Last N time steps

            # Compute score on masked positions only
            if use_discrepancy:
                discrepancy = ((teacher_output - student_output) ** 2) * masked_positions
                scores = discrepancy.sum(dim=(1, 2)) / (masked_positions.sum(dim=(1, 2)) + 1e-8)
            else:
                reconstruction_error = ((output - sequences) ** 2) * masked_positions
                scores = reconstruction_error.sum(dim=(1, 2)) / (masked_positions.sum(dim=(1, 2)) + 1e-8)
```

### Step-by-Step Process

1. **Input**: Entire test sequence (100 time steps × 5 features)

2. **Masking**: Mask the last `mask_last_n` time steps (default: 10 steps)
   - First 90 steps: unmasked (visible to model)
   - Last 10 steps: masked (model must reconstruct)

3. **Forward Pass**:
   - Model sees first 90 steps
   - Reconstructs all 100 steps (including masked last 10)

4. **Scoring**: Compute error **only on the masked positions** (last 10 steps)
   - **Normal mode**: Teacher-student discrepancy
     ```
     score = mean((teacher_output - student_output)² on last 10 steps)
     ```
   - **Teacher-only ablation**: Reconstruction error
     ```
     score = mean((teacher_output - ground_truth)² on last 10 steps)
     ```
   - **Student-only ablation**: Reconstruction error
     ```
     score = mean((student_output - ground_truth)² on last 10 steps)
     ```

5. **Threshold**: Find optimal threshold using ROC curve
   ```python
   fpr, tpr, thresholds = roc_curve(labels, scores)
   optimal_idx = np.argmax(tpr - fpr)  # Maximize (TPR - FPR)
   optimal_threshold = thresholds[optimal_idx]
   ```

6. **Prediction**:
   ```python
   prediction = 1 if score > optimal_threshold else 0
   ```

### Configuration Parameters

From [multivariate_mae_experiments.py:48-58](multivariate_mae_experiments.py#L48-L58):

```python
class Config:
    # Detection parameters
    mask_last_n: int = 10  # Number of last time steps to mask during inference

    # Training parameters
    masking_ratio: float = 0.6  # 60% of patches/tokens masked during training
    use_masking: bool = True

    # Model architecture
    use_teacher: bool = True
    use_student: bool = True
    use_discrepancy_loss: bool = True
```

---

## 3. Why This Design?

### Sequence-Level Labeling

**Advantages**:
- Simple ground truth generation
- Matches real-world scenarios where we know "something is wrong" but not exactly where
- Easier to label real data (operators know a system was anomalous during a time window)

**Limitations**:
- Can't measure point-level precision (which exact time steps were detected)
- Anomaly could be at start, middle, or end of sequence

### Last-N Masking for Detection

**Why mask the last N steps?**

1. **Simulates online monitoring**: In real deployment, we have historical data (first 90 steps) and want to detect if current/recent behavior (last 10 steps) is anomalous

2. **Evaluates model's predictive ability**: Can the model accurately predict recent behavior from historical context?

3. **Consistent with training**: During training, model learns to reconstruct masked portions. During testing, we mask the last N to see if reconstruction/discrepancy is higher for anomalies

4. **Sequence-level score from point-level errors**: Averages error over last 10 steps to get a single score per sequence, matching the sequence-level labels

### Why Last 10 Out of 100?

- **10% of sequence**: Reasonable balance
- **Recent behavior focus**: Last 10 steps represent "current state"
- **Statistical reliability**: Averaging over 10 steps × 5 features = 50 values reduces noise

---

## 4. Alignment Between Labeling and Detection

### Current Implementation

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Masking** | Random 60% of patches/tokens | Fixed: last 10 time steps |
| **Objective** | Reconstruct masked patches | Reconstruct last 10 steps |
| **Label** | Sequence-level (0 or 1) | Sequence-level score |
| **Scoring** | MSE reconstruction + discrepancy loss | Discrepancy/reconstruction on last 10 steps |

### Potential Issues

1. **Temporal locality mismatch**:
   - Some anomalies (spike, noise burst) occur in the middle of the sequence
   - Detection masks the last 10 steps
   - If anomaly is at time steps 30-40, but we only score steps 90-100, we might miss it

2. **Training-inference gap**:
   - Training: random masking across entire sequence
   - Inference: fixed masking of last 10 steps
   - Model trained to handle any position, but only tested on last position

### Why It Still Works

1. **Sequence-level labels**: Since label is 1 for entire sequence, any detection method that finds the anomaly anywhere counts as correct

2. **Temporal correlations**: Multivariate time series features are correlated over time
   - Memory leak: starts early, affects last 10 steps
   - Network congestion: change point early, but persists to end
   - Even spike at t=40 may affect predictions at t=90-100 due to correlations

3. **Model sees context**: First 90 steps provide context. If there was a spike/leak/drift, the context will look different, making last 10 steps harder to predict accurately

---

## 5. Alternative Approaches (Not Currently Implemented)

### Option 1: Sliding Window Masking
- Mask different positions during inference
- Take max/mean anomaly score across positions
- Better handles anomalies at any position

### Option 2: Point-Level Labels
- Label each time step as 0 or 1
- Track exact positions of injected anomalies
- Evaluate point-level precision/recall
- More complex ground truth generation

### Option 3: Reconstruction of All Steps
- Don't use fixed masking during inference
- Score reconstruction error on entire sequence
- Similar to classic autoencoder anomaly detection

---

## 6. Verification Summary

✅ **Anomaly Labeling**:
- Sequence-level (window-based)
- Label = 1 if anomaly exists anywhere in 100-step sequence
- Label = 0 if entire sequence is normal

✅ **Anomaly Detection**:
- Masks last 10 time steps during inference
- Computes discrepancy (or reconstruction error) on those 10 steps
- Averages to get sequence-level anomaly score
- Threshold via ROC curve optimization

✅ **Alignment**:
- Both are sequence-level (one label, one score per sequence)
- Works because labels apply to full sequence and temporal correlations propagate anomalies

⚠️ **Limitation**:
- Anomalies in middle of sequence may be harder to detect if they don't affect last 10 steps
- Training-inference masking strategy differs (random vs fixed-last-N)

---

## 7. Code References

- **Anomaly injection**: [lines 168-254](multivariate_mae_experiments.py#L168-L254)
- **Label assignment**: [lines 265-284](multivariate_mae_experiments.py#L265-L284)
- **Anomaly scoring**: [lines 810-849](multivariate_mae_experiments.py#L810-L849)
- **Evaluation**: [lines 851-871](multivariate_mae_experiments.py#L851-L871)
- **Config**: [lines 48-98](multivariate_mae_experiments.py#L48-L98)

---

## 8. Recommendations

### Current Implementation: ✓ Acceptable

The current approach is **valid and reasonable** for sequence-level anomaly detection.

### Possible Improvements (Optional):

1. **Add point-level evaluation**: Track anomaly positions and measure point-level metrics
2. **Multi-position masking**: Mask different positions during training/inference
3. **Adaptive masking**: Learn which positions to mask based on data
4. **Full-sequence scoring**: Score reconstruction on all positions, not just last N

But for the current research goal (comparing patch vs token masking, ablation studies), the existing mechanism is sufficient.
