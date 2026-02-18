# TEP (Tennessee Eastman Process) 실험 가이드

**작성일**: 2026-02-17
**대상**: 현재 SWaT/WaDi/Simulation 실험 구조를 TEP 데이터셋에 적용하는 방법

---

## 목차

| 섹션 | 내용 |
|------|------|
| [1. 현재 시스템 이해](#1-현재-시스템-이해) | 모델 목적, 실험 구조, 데이터 흐름 |
| [2. TEP 데이터셋 특성](#2-tep-데이터셋-특성) | 구조, 로더 동작, 규모 계산 |
| [3. 데이터셋 비교](#3-데이터셋-비교) | SWaT / WaDi / TEP 특성 비교 |
| [4. 권장 하이퍼파라미터](#4-권장-하이퍼파라미터) | seq_length, stride, 모델 파라미터 |
| [5. 실험 시나리오](#5-실험-시나리오) | Quick Test / Single Fault / All Faults |
| [6. 실행 방법](#6-실행-방법) | 명령어, 출력 구조 |
| [7. 주의사항](#7-주의사항) | 알려진 제한사항, 트러블슈팅 |

---

## 1. 현재 시스템 이해

### 1.1 모델 목적

Self-Distilled MAE는 **비지도/반지도 이상 탐지** 모델이다.

- **학습**: 정상 데이터만 사용 (anomaly-free training)
- **탐지 원리**:
  - Masked Autoencoder가 정상 패턴의 재구성(reconstruction)을 학습
  - Teacher decoder: 복잡한 decoder로 정확하게 재구성
  - Student decoder: 얕은 decoder로 근사 재구성
  - **이상 신호**: teacher-student 불일치(discrepancy)가 정상 대비 크게 발생
- **이상 점수**: `score = recon_loss + λ × discrepancy_loss`

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

### 1.3 데이터 흐름 세부

**SlidingWindowDataset**:
- `train_ratio`로 train/test split 결정
- `run_boundaries` 지정 시: 슬라이딩 윈도우가 런 경계를 절대 넘지 않음
- 각 윈도우는 `sample_type`으로 분류됨:
  - `0` (pure_normal): 윈도우 전체가 정상
  - `1` (disturbing_normal): 이상 구간에 걸쳐 있지만 다수가 정상
  - `2` (anomaly): 윈도우의 일부 이상 포함 (훈련 시 discrepancy loss 강화)

**평가 (Evaluator)**:
- Test stride=1로 생성된 모든 윈도우에 이상 점수 계산
- 각 timestep의 점수 = 해당 timestep을 포함하는 모든 윈도우 점수의 집계 (voting/mean/max)
- 지표: PA%K (Point-Adjust@K%), F1_T (Time-series F1), AUC-ROC

---

## 2. TEP 데이터셋 특성

### 2.1 데이터 파일 위치

```
dataset/TEP/
├── TEP_FaultFree_Testing.RData   (46 MB)  ← 훈련용 (정상 런)
├── TEP_FaultFree_Training.RData  (24 MB)  ← 미사용 (짧음)
├── TEP_Faulty_Testing.RData      (799 MB) ← 테스트용 (fault 런)
└── TEP_Faulty_Training.RData     (472 MB) ← 미사용 (짧음)
```

### 2.2 런 구조

| 항목 | 값 |
|------|-----|
| Fault-free 런 수 | 500 runs |
| Faulty 런 수 | 500 runs × 20 fault types |
| 런당 샘플 수 | 960 samples |
| Fault onset | sample 160 (0-indexed: 159) |
| 정상 구간 | samples 0-159 (160 samples) |
| 이상 구간 | samples 160-959 (800 samples) |
| 이상 비율 per run | 800/960 ≈ 83.3% |
| Feature 수 | 52 (상수 열 자동 제거됨) |

### 2.3 로더 동작 방식

```python
# 사용 가능한 DATASET_TYPE
'tep'           # 전체 20 fault types
'tep_fault1'    # fault type 1만
...
'tep_fault20'   # fault type 20만
```

로더 내부 동작 (`load_tep(fault_types=None, n_train_runs=50, n_test_runs=50)`):

1. `TEP_FaultFree_Testing.RData` 로드 → `n_train_runs`개 런 랜덤 선택 → 연결
2. `TEP_Faulty_Testing.RData` 로드 → 각 fault_type별 `n_test_runs`개 런 연결
3. `run_boundaries`: 각 런의 끝 위치 목록 (슬라이딩 윈도우 경계 보호용)
4. 상수 열 제거, NaN 처리 (ffill+bfill), min-max 정규화
5. `train_ratio = train_len / total_len` 자동 계산

### 2.4 데이터 규모 계산

기본값 (`n_train_runs=50`, `n_test_runs=50`):

| 항목 | 계산 | 크기 |
|------|------|------|
| Train (fault-free) | 50 runs × 960 | 48,000 samples |
| Test (all 20 faults) | 20 × 50 × 960 | 960,000 samples |
| Total | - | 1,008,000 samples |
| train_ratio | 48K / 1008K | ≈ 0.0476 (4.8%) |
| 메모리 (float32) | 1008K × 52 × 4 bytes | ≈ 210 MB |

단일 fault (`n_test_runs=50`):

| 항목 | 크기 |
|------|------|
| Train | 48,000 samples |
| Test (1 fault) | 48,000 samples |
| Total | 96,000 samples |
| train_ratio | ≈ 0.50 |

> **Note**: `train_ratio`는 로더가 자동 계산하므로 BASE_CONFIG에 설정 불필요.

---

## 3. 데이터셋 비교

| 특성 | SWaT A1+A2 | WaDi 14days+A1 | TEP |
|------|-----------|----------------|-----|
| 데이터 구조 | 연속 시계열 | 연속 시계열 | 독립 시뮬레이션 런 |
| Train | Normal 기간 | Normal 기간 | Fault-free runs |
| Test | Attack 기간 | Attack 기간 | Faulty runs |
| run_boundaries | 없음 | 없음 | **필수** (런 경계 보호) |
| Feature 수 | 51 | 127 | 52 |
| Train 길이 | ~495K | ~118K | 48K (50 runs) |
| Test 길이 | ~449K | ~14K | 48K~960K (runs × faults) |
| train_ratio 계산 | 데이터셋 고정 | 데이터셋 고정 | 런 수에 따라 변동 |
| Anomaly type 수 | 1 (attack) | 1 (attack) | 20 (fault types) |
| 추천 stride (train) | 11 | 11 | 5 |

---

## 4. 권장 하이퍼파라미터

### 4.1 seq_length 결정

TEP 런: 960 samples, fault onset: sample 160

```
seq_length=160 (권장)
- fault onset 기간과 정확히 대응
- 정상 구간(0-159)에서 1개 완전한 정상 윈도우 추출 가능
- 이상 구간에서 완전한 이상 윈도우 다수 추출
- patch_size=8, num_patches=20

seq_length=100 (대안, 빠른 실험)
- 더 많은 정상 윈도우 (정상 구간에서 60개 추가)
- patch_size=5, num_patches=20
- 런당 더 많은 윈도우 생성 → 더 많은 훈련 샘플

seq_length=500 (권장 안 함)
- 960 샘플 런에서 비효율
- 대부분의 윈도우가 fault onset 포함 → disturbing_normal 과다
```

### 4.2 Stride 결정

```
sliding_window_stride=5 (train, 권장)
- seq_length=160: 런당 (960-160)/5 + 1 ≈ 161 windows
- 50 runs: ~8,050 training windows

sliding_window_test_stride=1 (test, 필수)
- PA%K 정확한 계산을 위해 반드시 1 사용
```

### 4.3 권장 설정 요약

```python
BASE_CONFIG = {
    # 아키텍처 (Phase 2 최적 모델과 동일)
    'seq_length': 160,
    'patch_size': 8,
    'num_patches': 20,        # seq_length / patch_size
    'd_model': 128,
    'nhead': 8,
    'num_encoder_layers': 2,
    'num_teacher_decoder_layers': 4,
    'num_student_decoder_layers': 1,
    'patchify_mode': 'patch_cnn',

    # 학습
    'num_epochs': 50,
    'learning_rate': 2e-3,
    'batch_size': 256,
    'warmup_epochs': 10,
    'teacher_only_warmup_epochs': 3,
    'use_amp': True,

    # 슬라이딩 윈도우
    'use_sliding_window_dataset': True,
    'sliding_window_stride': 5,
    'sliding_window_test_stride': 1,

    # num_features는 자동으로 설정됨 (loaders.py에서)
    # train_ratio도 자동으로 설정됨 (data_info['train_ratio'])
}
```

---

## 5. 실험 시나리오

### Scenario A: Quick Smoke Test

**목적**: 파이프라인 동작 확인 (1 epoch, 1 fault)
**파일**: `scripts/ablation/configs/tep_quick_test.py`

```
Train: 50 fault-free runs × 960 = 48,000 samples
Test:  fault1, 10 runs × 960 = 9,600 samples
Total: ~57,600 samples
시간: 약 3-5분
```

### Scenario B: Single Fault Full Study

**목적**: 특정 fault에 대한 탐지 성능 분석 (50 epochs)
**파일**: `scripts/ablation/configs/tep_single_fault.py`

- `tep_fault1` ~ `tep_fault20` 사용 가능
- 각 fault별 탐지 난이도 분석
- 알려진 어려운 fault: fault3, fault9, fault15 (산업 벤치마크 기준)

```
Train: 50 fault-free runs × 960 = 48,000 samples
Test:  1 fault × 50 runs × 960 = 48,000 samples
train_ratio ≈ 0.50
```

### Scenario C: All 20 Faults

**목적**: 전체 TEP 탐지 성능 평가
**파일**: `scripts/ablation/configs/tep_all_faults.py`

```
Train: 50 fault-free runs × 960 = 48,000 samples
Test:  20 faults × 50 runs × 960 = 960,000 samples
train_ratio ≈ 0.048
```

> **Note**: train_ratio가 매우 작지만 정상적으로 동작함. 로더에서 자동 계산됨.

---

## 6. 실행 방법

### 환경 설정

```bash
conda activate dc_vis
```

### Quick Test 실행

```bash
python scripts/ablation/run_ablation.py \
    --config scripts/ablation/configs/tep_quick_test.py
```

### 전체 실험 실행

```bash
# Single fault (fault1)
python scripts/ablation/run_ablation.py \
    --config scripts/ablation/configs/tep_single_fault.py

# All 20 faults
python scripts/ablation/run_ablation.py \
    --config scripts/ablation/configs/tep_all_faults.py
```

### 결과 구조

```
results/
└── tep_all_faults/
    ├── 000_ablation_info/
    │   ├── dataset.md          ← 데이터셋 통계
    │   └── ablation_config.json
    └── 000_tep_base/
        ├── best_config.json
        ├── best_metrics.json   ← PA%K, F1_T, AUC-ROC
        ├── best_model.pth
        └── best_model_detailed.csv
```

### 결과 지표 확인

```bash
cat results/tep_all_faults/000_tep_base/best_metrics.json
```

주요 지표:
- `test_pa_k_f1`: PA%K F1 score (주요 지표)
- `test_f1_t`: Time-series F1 (F1_T)
- `test_roc_auc`: AUC-ROC

---

## 7. 주의사항

### 7.1 run_boundaries 필수

TEP는 독립 시뮬레이션 런으로 구성됨. 슬라이딩 윈도우가 런 경계를 넘으면 의미 없는 패턴이 생성됨.
→ TEP 로더는 자동으로 `run_boundaries`를 `data_info`에 포함시킴.
→ `run_ablation.py`가 자동으로 처리함. 별도 설정 불필요.

### 7.2 num_features 자동 설정

TEP의 feature 수는 상수 열 제거 후 자동 결정됨 (일반적으로 52).
BASE_CONFIG에 `num_features`를 설정하지 않아도 됨. run_ablation.py가 자동으로 업데이트함.

### 7.3 train_ratio 자동 계산

TEP 로더가 `train_ratio`를 자동 계산하여 `data_info`에 포함.
BASE_CONFIG의 `sliding_window_train_ratio`는 무시됨 (simulation 전용 파라미터).

### 7.4 메모리 요구사항

- All faults (20 × 50 runs): ~210 MB RAM (float32)
- test stride=1이면 test windows 수가 많아 evaluation 시 GPU 메모리 주의
  - 960K test samples, stride=1: ~800K+ test windows
  - batch_size=256이면 GPU memory 약 4-8 GB 사용

### 7.5 TEP 데이터 통계적 특성

- **fault1**: Step change → 탐지 쉬움
- **fault3, fault9, fault15**: 미세한 변화 → 탐지 어려움 (산업 벤치마크 기준)
- **fault20**: 미지 fault (이상 없음처럼 보임) → 주의
- 정상 구간이 fault onset 이후에도 시스템이 안정화될 수 있음

### 7.6 DATASET_LOADERS 키 목록

```python
# loaders.py에 등록된 TEP 관련 키
'tep'           # 전체 20 faults
'tep_fault1'    # Fault type 1: 'A/C Feed Step (stream 4)'
'tep_fault2'    # Fault type 2: 'B Composition Step (stream 4)'
'tep_fault3'    # Fault type 3: 'D Feed Temperature Step'
...
'tep_fault20'   # Fault type 20: Unknown
```

### 7.7 기존 실험과의 차이점

| 특성 | SWaT/WaDi | TEP |
|------|-----------|-----|
| 데이터 연속성 | 단일 연속 시계열 | 독립 런 연결 |
| run_boundaries | 없음 | 자동 포함 |
| Anomaly type | 단일 (attack=1) | 20가지 (fault 1-20) |
| train_ratio | 고정 (70:30 등) | 동적 계산 |
| 추천 seq_length | 500 | 160 |
| 추천 stride | 11 | 5 |

---

**참조 파일**:
- [loaders.py - load_tep()](../mae_anomaly/datasets/loaders.py) (line 643)
- [run_ablation.py](../scripts/ablation/run_ablation.py)
- [swat_A1A2_test.py](../scripts/ablation/configs/swat_A1A2_test.py) (참조 config)
