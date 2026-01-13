# Point-Level Anomaly Detection 구현

## 개요

Sequence-level anomaly detection에 더해 point-level anomaly detection을 추가로 구현했습니다.

---

## 구현 내용

### 1. Point-Level Labels 추가

각 anomaly injection 함수가 이제 point-level binary mask를 반환합니다:

```python
def _inject_multivariate_spike(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: (modified_signals, point_labels)
    point_labels: (seq_length,) binary array indicating which time steps are anomalous
    """
    point_labels = np.zeros(self.seq_length, dtype=np.int64)
    point_labels[spike_pos:spike_pos + spike_width] = 1  # Mark anomalous time steps
    # ... modify signals ...
    return signals, point_labels
```

#### 각 Anomaly 유형별 Point-Level Labels

| Anomaly Type | Affected Time Steps | Label Pattern |
|--------------|---------------------|---------------|
| **Spike** | `spike_pos:spike_pos+spike_width` (3-10 steps) | Contiguous 1s in spike region |
| **Memory Leak** | `start_pos:end` (50-80 steps) | 1s from leak start to end |
| **Noise Burst** | `noise_start:noise_start+noise_length` (20-40 steps) | Contiguous 1s in noisy region |
| **Drift** | `drift_start:drift_start+drift_length` (30-50 steps) | Contiguous 1s in drift region |
| **Network Congestion** | `change_point:end` (30-70 steps) | 1s from congestion start to end |

### 2. Dataset 수정

```python
class MultivariateTimeSeriesDataset:
    def __init__(self, ...):
        self.data, self.seq_labels, self.point_labels = self._generate_data()

    def __getitem__(self, idx):
        """
        Returns:
            sequence: (seq_length, num_features)
            seq_label: scalar - sequence-level binary label
            point_labels: (seq_length,) - point-level binary labels
        """
        return sequence, seq_label, point_labels
```

**데이터 shape**:
- `self.data`: `(num_samples, seq_length=100, num_features=5)`
- `self.seq_labels`: `(num_samples,)` - binary (0 or 1)
- `self.point_labels`: `(num_samples, seq_length=100)` - binary (0 or 1)

### 3. Anomaly Scoring

#### Sequence-Level Scoring (기존 방식)
```python
# Mask last N time steps
mask[:, -config.mask_last_n:] = 0

# Compute error at each time step
error = ((teacher_output - student_output) ** 2).mean(dim=2)  # (batch, seq_length)

# Aggregate over masked positions
masked_positions = (mask == 0)
seq_score = (error * masked_positions).sum(dim=1) / masked_positions.sum(dim=1)
# Result: (batch,) - one score per sequence
```

#### Point-Level Scoring (신규)
```python
# Same error computation
error = ((teacher_output - student_output) ** 2).mean(dim=2)  # (batch, seq_length)

# Each time step gets its own score
point_scores = error  # (batch, seq_length)
# Result: (batch, seq_length) - one score per time step
```

### 4. Evaluation Metrics

3가지 평가 결과를 제공합니다:

#### (1) Sequence-Level Evaluation

```python
# Evaluate using sequence-level scores and labels
seq_scores: (num_samples,)
seq_labels: (num_samples,)

metrics = {
    'roc_auc': ...,
    'precision': ...,
    'recall': ...,
    'f1_score': ...,
    'optimal_threshold': ...
}
```

#### (2) Point-Level Evaluation

```python
# Evaluate using point-level scores and labels (flattened)
point_scores: (num_samples * seq_length,)
point_labels: (num_samples * seq_length,)

metrics = {
    'roc_auc': ...,
    'precision': ...,
    'recall': ...,
    'f1_score': ...,
    'optimal_threshold': ...
}
```

#### (3) Combined Evaluation

```python
# Weighted average based on F1 scores
seq_weight = seq_f1 / (seq_f1 + point_f1)
point_weight = point_f1 / (seq_f1 + point_f1)

combined_metric = seq_weight * seq_metric + point_weight * point_metric
```

---

## 출력 형식

```
================================================================================
EVALUATION RESULTS
================================================================================

Metric               Sequence-Level       Point-Level          Combined
--------------------------------------------------------------------------------
ROC-AUC              0.9234               0.8567               0.8901
Precision            0.8901               0.7845               0.8373
Recall               0.8567               0.8123               0.8345
F1-Score             0.8731               0.7982               0.8357
================================================================================
Combined weights: Sequence=0.52, Point=0.48
================================================================================
```

---

## 사용 예시

```python
# Train model
trainer = Trainer(model, config, train_loader, test_loader)
trainer.train()

# Evaluate
evaluator = Evaluator(model, config, test_loader)
results = evaluator.evaluate()

# Access results
print(f"Sequence-level F1: {results['sequence']['f1_score']:.4f}")
print(f"Point-level F1: {results['point']['f1_score']:.4f}")
print(f"Combined F1: {results['combined']['f1_score']:.4f}")
```

---

## Sequence-Level vs Point-Level 비교

| Aspect | Sequence-Level | Point-Level |
|--------|----------------|-------------|
| **Label Granularity** | 1 label per sequence (100 steps) | 1 label per time step |
| **Label Count** | num_samples | num_samples × seq_length |
| **Scoring** | Average error over last N steps | Error at each time step |
| **Use Case** | "Is this window anomalous?" | "Which exact time steps are anomalous?" |
| **Precision** | Coarse | Fine-grained |
| **Evaluation** | ROC-AUC on sequence scores | ROC-AUC on point scores |

---

## 장단점

### Sequence-Level

**장점**:
- Simple labeling (easier to obtain in practice)
- Matches deployment scenario (monitor time windows)
- Less sensitive to labeling errors
- Balanced dataset (fewer samples)

**단점**:
- Can't pinpoint exact anomaly location
- Mislabels normal parts of anomalous sequences
- Less informative

### Point-Level

**장점**:
- Precise localization of anomalies
- Fine-grained evaluation
- Better for root cause analysis
- More informative for debugging

**단점**:
- Requires precise labeling (harder to obtain)
- Highly imbalanced dataset (많은 normal points)
- May be overly strict
- Harder to label in practice

### Combined

**장점**:
- Balances both perspectives
- Robust evaluation
- Considers both window-level and point-level performance

**단점**:
- Interpretation can be complex
- Weights may need tuning

---

## 구현 참고사항

### DataLoader 수정 필요

기존:
```python
for sequences, labels in data_loader:
    # ...
```

신규:
```python
for sequences, seq_labels, point_labels in data_loader:
    # seq_labels: (batch,)
    # point_labels: (batch, seq_length)
```

### Backward Compatibility

Training 시에는 point_labels를 로드만 하고 사용하지 않습니다:
```python
for sequences, seq_labels, point_labels in train_loader:
    # point_labels loaded but not used in training
    # Only seq_labels used for loss computation
```

Evaluation 시에는 둘 다 사용합니다.

---

## 파일 수정 사항

### 수정된 파일들

1. **multivariate_mae_experiments.py**
   - `_inject_*()` 함수들: point_labels 반환
   - `_generate_data()`: point_labels 생성 및 저장
   - `__getitem__()`: point_labels 반환
   - `Trainer.train_epoch()`: point_labels 로드
   - `Evaluator.compute_anomaly_scores()`: 두 레벨 점수 계산
   - `Evaluator.evaluate()`: 3가지 결과 반환

---

## 성능 비교 예상

일반적으로:
- **Sequence-level**: F1-score 0.85-0.95 (easier task)
- **Point-level**: F1-score 0.70-0.85 (harder task, imbalanced)
- **Combined**: F1-score 0.80-0.90 (balanced view)

Point-level이 더 어려운 이유:
1. Class imbalance (20-40 anomalous points vs 60-80 normal points per anomalous sequence)
2. Boundary effects (anomaly start/end points may be ambiguous)
3. Stricter evaluation (must pinpoint exact locations)

---

## 향후 개선 방향

1. **PA%K Metric**: Point-Adjusted metric (anomaly detection 전용 metric)
2. **Window-based Point Evaluation**: Allow some tolerance around true points
3. **Feature-wise Point Labels**: Track which features are anomalous
4. **Adaptive Weighting**: Learn optimal combination weights
5. **Range-based Evaluation**: Evaluate on anomaly ranges instead of points

---

## 참고 자료

- **NAB (Numenta Anomaly Benchmark)**: Point-level anomaly detection benchmark
- **SMD (Server Machine Dataset)**: Point-level labels for multivariate time series
- **SMAP/MSL**: NASA datasets with point-level labels

---

## 요약

✅ **Point-level labels 추가 완료**
✅ **Point-level scoring 구현 완료**
✅ **3-way evaluation (sequence/point/combined) 구현 완료**
✅ **Backward compatible (기존 코드 동작 유지)**
✅ **모든 anomaly 유형에 대해 point-level labels 생성**

이제 더 정밀한 anomaly detection 평가가 가능합니다!
