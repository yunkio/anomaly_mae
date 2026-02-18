# TEP (Tennessee Eastman Process) 실험 가이드

**작성일**: 2026-02-17

---

## 1. 현재 시스템 이해

### 1.1 모델 목적

Self-Distilled MAE는 이상 탐지 모델로, 데이터셋에 따라 학습 방식이 다르다.

**학습 방식**:
- `force_mask_anomaly=True`: 이상 패치는 항상 마스킹 → 이상값을 재구성 대상에서 제외
- `anomaly_loss_weight=2.0`: 이상 윈도우의 discrepancy loss 가중치 증폭
- Reconstruction loss: 마스킹된 패치의 재구성 품질 학습 (teacher 기준)

**탐지 원리 (추론 시)**:
- Teacher decoder: 복잡한 구조로 더 잘 재구성
- Student decoder: 얕은 구조로 근사 재구성 → 이상 시 teacher와 차이 발생
- **이상 점수**: `score = recon_loss + λ × discrepancy_loss`

**Dataset별 훈련 이상 포함 여부**:

| Dataset | 훈련 이상 비율 | 학습 방식 |
|---------|-------------|---------|
| Simulation | ~13% | Semi-supervised (force_mask_anomaly, anomaly_loss_weight 활성) |
| SWaT / WaDi / TEP / SMD | 0% | Unsupervised (이상 윈도우 없으므로 위 파라미터 발동 안 됨) |

### 1.2 실험 프레임워크 구조

```
Config 파일 (DATASET_TYPE, BASE_CONFIG, EXPERIMENTS)
    │
    ↓ run_ablation.py
DATASET_TYPE → get_dataset_loader()
    │
    ↓ loader()
(signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info)
    │
    ├── run_boundaries = data_info.get('run_boundaries')  # 런 경계 보호
    │
    ↓ SlidingWindowDataset
Train windows (stride=N) / Test windows (stride=1)
    │
    ↓ Trainer → Evaluator
PA%K, F1_T, AUC-ROC 지표
```

**평가**:
- Test stride=1로 생성된 모든 윈도우에 이상 점수 계산
- 각 timestep 점수 = 해당 timestep을 포함하는 모든 윈도우 점수의 집계 (voting/mean/max)
- 지표: PA%K, F1_T (Time-series F1), AUC-ROC

---

## 2. TEP 데이터셋 특성

### 2.1 데이터 파일

```
dataset/TEP/
├── TEP_FaultFree_Testing.RData   (46 MB)  ← 훈련용
├── TEP_Faulty_Testing.RData      (799 MB) ← 테스트용
├── TEP_FaultFree_Training.RData  (24 MB)  ← 미사용 (짧음)
└── TEP_Faulty_Training.RData     (472 MB) ← 미사용 (짧음)
```

### 2.2 런 구조

| 항목 | 값 |
|------|-----|
| Fault-free 런 수 | 500 runs |
| Faulty 런 수 | 500 runs × 20 fault types |
| 런당 샘플 수 | 960 samples |
| Fault onset | sample 160 (정상: 0-159, 이상: 160-959) |
| 이상 비율 per run | 800/960 ≈ 83.3% |
| Feature 수 | 52 (상수 열 자동 제거) |

### 2.3 로더 동작 방식

```python
# DATASET_TYPE 키
'tep'          # 전체 20 fault types
'tep_fault1'   # fault type 1만
...
'tep_fault20'  # fault type 20만
```

로더 내부 (`load_tep(fault_types=None, n_train_runs=50, n_test_runs=50)`):
1. Fault-free runs → n_train_runs개 선택 → 연결 (훈련용)
2. Faulty runs → 각 fault_type별 n_test_runs개 연결 (테스트용)
3. `run_boundaries` 자동 생성 (슬라이딩 윈도우 경계 보호)
4. 상수 열 제거, NaN 처리, min-max 정규화
5. `train_ratio = train_len / total_len` 자동 계산

### 2.4 데이터 규모

| 구성 | Train | Test | train_ratio |
|------|-------|------|------------|
| 전체 (n_train=50, n_test=50, 20 faults) | 48,000 | 960,000 | ≈ 0.048 |
| 단일 fault (n_train=50, n_test=50) | 48,000 | 48,000 | ≈ 0.50 |

---

## 3. 데이터셋 비교

| 특성 | SWaT / WaDi | TEP |
|------|------------|-----|
| 데이터 구조 | 단일 연속 시계열 | 독립 시뮬레이션 런 연결 |
| run_boundaries | 없음 | **자동 포함** (런 경계 보호) |
| train_ratio | 데이터셋 고정 | 런 수에 따라 동적 계산 |
| Anomaly type 수 | 1 (attack) | 20 (fault types) |
| 추천 seq_length | 500 | 160 (fault onset 기간 대응) |
| 추천 train stride | 11 | 5 |

---

## 4. 권장 설정

### seq_length

- **160 (권장)**: fault onset 기간(0-159)과 정확히 대응. `patch_size=8, num_patches=20`
- **100 (대안)**: 런당 더 많은 윈도우. `patch_size=5, num_patches=20`
- **500 (권장 안 함)**: 960-sample 런에서 disturbing_normal 과다

### 핵심 파라미터

```
sliding_window_stride=5      (train: 런당 ~161 windows)
sliding_window_test_stride=1 (test: PA%K 정확 계산 필수)
num_features                 → 자동 설정 (로더에서 상수 열 제거 후 결정)
train_ratio                  → 자동 설정 (data_info에서)
```

---

## 5. 실험 방향

- **단일 fault**: `DATASET_TYPE = 'tep_fault1'` ~ `'tep_fault20'`
  - train_ratio ≈ 0.50 (균형 잡힌 데이터셋)
  - fault별 탐지 난이도 분석에 적합
  - 알려진 어려운 fault: fault3, fault9, fault15

- **전체 20 faults**: `DATASET_TYPE = 'tep'`
  - train_ratio ≈ 0.048 (train 매우 작음, 정상 동작)
  - test 규모 큼 (~960K samples) → evaluation 오래 걸림

- **기존 config 참조**: `scripts/ablation/configs/swat_A1A2_test.py` 구조와 동일하게 작성
  - `DATASET_TYPE`, `BASE_CONFIG`, `EXPERIMENTS`, `SCORING_MODES` 정의

---

## 6. 주의사항

- `run_boundaries`: 로더가 자동 포함, run_ablation.py가 자동 처리 — 별도 설정 불필요
- `num_features`, `train_ratio`: BASE_CONFIG에 설정 불필요 — 로더가 자동 결정
- All faults (20×50 runs): ~210 MB RAM, test stride=1 시 GPU 4-8 GB 필요
- **fault20**: 탐지 불가 수준의 unknown fault — 결과 해석 시 주의

---

**참조**:
- [loaders.py - load_tep()](../mae_anomaly/datasets/loaders.py) (line 643)
- [swat_A1A2_test.py](../scripts/ablation/configs/swat_A1A2_test.py) (config 작성 참조)
