# Self-Distilled MAE for Multivariate Time Series Anomaly Detection

Self-Distilled Masked Autoencoder (MAE) 구현으로, 다변량 시계열 데이터의 이상 탐지를 수행합니다.

## 프로젝트 구조

```
.
├── README.md                 # 메인 문서
├── requirements.txt          # Python 의존성
├── setup.py                  # 패키지 설정
├── CLAUDE.md                 # Claude Code 작업 지침
├── ablation_guideline.md     # Ablation 실험 가이드
│
├── mae_anomaly/              # 메인 패키지
│   ├── __init__.py
│   ├── config.py             # 설정 클래스
│   ├── dataset_sliding.py    # 슬라이딩 윈도우 데이터셋
│   ├── model.py              # MAE 모델 아키텍처
│   ├── loss.py               # Self-distillation loss
│   ├── trainer.py            # 학습 로직
│   ├── evaluator.py          # 평가 로직
│   ├── datasets/             # 데이터셋 로더
│   │   ├── loaders.py        # SWaT, WaDi, Simulation 데이터 로더
│   │   └── noisy.py          # 노이즈 레이블 데이터셋
│   ├── utils/                # 유틸리티
│   │   ├── system.py         # GPU 메모리 관리
│   │   └── experiment.py     # 실험 설정 헬퍼
│   └── visualization/        # 시각화 모듈
│       ├── __init__.py
│       ├── base.py                      # 시각화 베이스 클래스
│       ├── data_visualizer.py           # 데이터 시각화
│       ├── architecture_visualizer.py   # 아키텍처 시각화
│       ├── experiment_visualizer.py     # 실험 결과 시각화
│       ├── stage2_visualizer.py         # Stage 2 시각화
│       ├── best_model_visualizer.py     # 최고 모델 분석
│       ├── training_visualizer.py       # 학습 진행 시각화
│       └── parallel.py                  # 병렬 처리 유틸
│
├── scripts/                  # 실행 스크립트
│   ├── visualize_all.py      # 통합 시각화 스크립트
│   └── ablation/
│       ├── run_ablation.py   # 통합 ablation 실험 실행기
│       └── configs/          # 실험 설정 파일
│           ├── README.md     # 설정 시스템 가이드
│           ├── simulation_test.py
│           ├── swat_A1A2_test.py
│           └── wadi_14days_A1_test.py
│
├── docs/                     # 문서
│   ├── ARCHITECTURE.md       # 모델 아키텍처 문서
│   ├── ABLATION_STUDIES.md   # Ablation study 설명
│   ├── ABLATION_EXPERIMENTS.md  # 실험 결과 문서
│   ├── CHANGELOG.md          # 변경 이력
│   ├── VISUALIZATIONS.md     # 시각화 가이드
│   ├── DATASET.md            # 데이터셋 문서
│   └── INFERENCE_MODES.md    # 추론 모드 문서
│
├── comparison/               # 모델 비교 (실험 결과)
│   ├── GUIDE.md              # 비교 가이드
│   ├── MODELS.md             # 모델 정의
│   └── results/              # 데이터셋별 비교 결과
│
└── results/                  # 실험 결과
    ├── experiments/          # Ablation 실험 결과
    ├── SWaT/                 # SWaT 데이터셋 결과
    └── WaDi/                 # WaDi 데이터셋 결과
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

### Ablation 실험 실행

**통합 설정 기반 시스템** (모든 데이터셋에 단일 진입점):

```bash
# Simulation 데이터 테스트
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/simulation_test.py

# SWaT A1+A2 실험
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/swat_A1A2_test.py

# WaDi 14days + A1 실험
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/wadi_14days_A1_test.py

# 시각화
python scripts/visualize_all.py --experiment-dir results/experiments/YYYYMMDD_HHMMSS
```

**설정 파일 생성**:
```bash
# 템플릿 복사
cp scripts/ablation/configs/simulation_test.py scripts/ablation/configs/my_experiment.py

# 설정 편집 (DATASET_TYPE, BASE_CONFIG, EXPERIMENTS)
# 실행
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/my_experiment.py
```

자세한 내용은 [ablation_guideline.md](ablation_guideline.md) 및 [scripts/ablation/configs/README.md](scripts/ablation/configs/README.md) 참조.

## 설정

`Config` 클래스의 주요 파라미터:

```python
# 데이터 파라미터
seq_length: int = 100           # 시퀀스 길이
num_features: int = 8           # Feature 수 (8개 서버 메트릭)
sliding_window_total_length: int = 440000  # 전체 시계열 길이
sliding_window_stride: int = 10  # 윈도우 stride

# 모델 파라미터
d_model: int = 64               # 모델 차원
nhead: int = 4                  # Attention head 수
num_encoder_layers: int = 3     # Encoder 레이어 수
num_patches: int = 25           # 패치 수
patch_size: int = 4             # 패치 크기
patchify_mode: str = 'linear'   # Patchify 모드

# 마스킹 파라미터
masking_ratio: float = 0.4      # 마스킹 비율
masking_strategy: str = 'patch' # 마스킹 전략 (patch 고정)

# 학습 파라미터
batch_size: int = 32
num_epochs: int = 50
learning_rate: float = 1e-3
```

## Patchify 모드

### 1. Linear (기본값)
- 패치화 후 linear embedding
- MAE 원본 논문 스타일
- 가장 단순한 구조

### 2. CNN First
- 전체 시퀀스에 2-layer 1D-CNN 적용
- CNN 출력을 패치화
- 로컬 feature를 먼저 추출

### 3. Patch CNN
- 먼저 패치화 후 각 패치에 CNN 적용
- Cross-patch information leakage 방지
- 패치별 독립적인 feature 추출

## 실험 결과

실험 실행 시 생성되는 결과물:

```
results/experiments/YYYYMMDD_HHMMSS/
├── quick_results.csv           # Quick search 결과 (1 epoch)
├── full_results.csv            # Full training 결과
├── best_model.pt               # 최고 성능 모델 checkpoint
└── visualization/              # 시각화 폴더
    ├── data/                   # 데이터 시각화
    ├── architecture/           # 아키텍처 시각화
    ├── stage1/                 # Quick search 결과
    ├── stage2/                 # Full training 결과
    └── best_model/             # Best model 분석
```

## 문서

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - 모델 아키텍처 상세 설명
- [ABLATION_STUDIES.md](docs/ABLATION_STUDIES.md) - Ablation study 설명
- [CHANGELOG.md](docs/CHANGELOG.md) - 변경 이력
- [VISUALIZATIONS.md](docs/VISUALIZATIONS.md) - 시각화 가이드
- [DATASET.md](docs/DATASET.md) - 슬라이딩 윈도우 데이터셋 문서

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

**마지막 업데이트**: 2026-02-15
