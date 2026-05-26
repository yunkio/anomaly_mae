# Self-Distilled MAE for Multivariate Time Series Anomaly Detection

Self-Distilled Masked Autoencoder (MAE) 구현으로, 다변량 시계열 데이터의 이상 탐지를 수행합니다.

## 프로젝트 구조

```
.
├── README.md                 # 메인 문서
├── requirements.txt          # Python 의존성
├── setup.py                  # 패키지 설정
│
└── mae_anomaly/              # 메인 패키지
    ├── __init__.py
    ├── config.py             # 설정 클래스
    ├── dataset_sliding.py    # 슬라이딩 윈도우 데이터셋
    ├── model.py              # MAE 모델 아키텍처
    ├── loss.py               # Self-distillation loss
    ├── trainer.py            # 학습 로직
    ├── evaluator.py          # 평가 로직
    ├── datasets/             # 데이터셋 로더
    │   ├── loaders.py        # SWaT, WaDi, SMD, PSM, Simulation, TEP 데이터 로더
    │   └── noisy.py          # 노이즈 레이블 데이터셋
    └── utils/                # 유틸리티
        ├── system.py         # GPU 메모리 관리
        ├── experiment.py     # 실험 설정 헬퍼
        └── sampling.py       # 카테고리 기반 서브샘플링
```

## 주요 기능

- **1D-CNN + Transformer 하이브리드 아키텍처**: 로컬 feature 추출과 글로벌 의존성 캡처
- **2가지 Patchify 모드**:
  - `linear`: Linear embedding (MAE 원본 스타일)
  - `patch_cnn`: Patchify → CNN (패치별 독립 CNN, cross-patch leakage 방지)
- **Self-Distillation**: Teacher-student 아키텍처와 discrepancy loss
- **2-Stage 실험**: Quick Search (1 epoch)로 상위 조합 선별 후 Full Training

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 또는 패키지로 설치
pip install -e .
```

## Quick Start

### 기본 사용법

```python
from mae_anomaly import (
    Config, set_seed,
    SlidingWindowTimeSeriesGenerator, SlidingWindowDataset,
    SelfDistilledMAEMultivariate
)

# 설정 생성
set_seed(42)
config = Config()
config.patchify_mode = 'linear'  # 'linear', 'patch_cnn'

# 데이터셋 생성 (슬라이딩 윈도우 기반)
generator = SlidingWindowTimeSeriesGenerator(
    total_length=100000,
    num_features=config.num_features,
    interval_scale=config.anomaly_interval_scale,
    seed=config.random_seed
)
signals, point_labels, anomaly_regions = generator.generate()

dataset = SlidingWindowDataset(
    signals=signals,
    point_labels=point_labels,
    anomaly_regions=anomaly_regions,
    window_size=config.seq_length,
    stride=config.sliding_window_stride,
    mask_last_n=10,
    split='train',
    train_ratio=0.5,
    seed=config.random_seed
)

# 모델 생성
model = SelfDistilledMAEMultivariate(config)
```

## 설정

`Config` 클래스의 주요 파라미터:

```python
# 데이터 파라미터
seq_length: int = 500           # 시퀀스 길이
num_features: int = 8           # Feature 수 (8개 서버 메트릭)
sliding_window_total_length: int = 275000  # 전체 시계열 길이
sliding_window_stride: int = 3  # 윈도우 stride

# 모델 파라미터
d_model: int = 128              # 모델 차원
nhead: int = 8                  # Attention head 수
num_encoder_layers: int = 2     # Encoder 레이어 수
num_patches: int = 100          # 패치 수
patch_size: int = 5             # 패치 크기
patchify_mode: str = 'patch_cnn'  # Patchify 모드

# 마스킹 파라미터
masking_ratio: float = 0.15     # 마스킹 비율

# 학습 파라미터
batch_size: int = 256
num_epochs: int = 50
learning_rate: float = 2e-3
```

## Patchify 모드

### 1. Linear
- 패치화 후 linear embedding
- MAE 원본 논문 스타일
- 가장 단순한 구조

### 2. Patch CNN (기본값)
- 먼저 패치화 후 각 패치에 CNN 적용
- Cross-patch information leakage 방지
- 패치별 독립적인 feature 추출

## 요구사항

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- Scikit-learn >= 1.3.0
- tqdm >= 4.65.0

## License

MIT License

---

**마지막 업데이트**: 2026-02-17
