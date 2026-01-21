# Self-Distilled MAE for Multivariate Time Series Anomaly Detection

Self-Distilled Masked Autoencoder (MAE) 구현으로, 다변량 시계열 데이터의 이상 탐지를 수행합니다.

## 프로젝트 구조

```
.
├── README.md                 # 메인 문서
├── requirements.txt          # Python 의존성
├── setup.py                  # 패키지 설정
│
├── mae_anomaly/              # 메인 패키지
│   ├── __init__.py
│   ├── config.py             # 설정 클래스
│   ├── dataset.py            # 데이터셋 구현
│   ├── model.py              # MAE 모델 아키텍처
│   ├── loss.py               # Self-distillation loss
│   ├── trainer.py            # 학습 로직
│   └── evaluator.py          # 평가 로직
│
├── scripts/                  # 실행 스크립트
│   ├── run_experiments.py    # 2-stage 실험 (Quick Search → Full Training)
│   └── visualize_all.py      # 결과 시각화
│
├── examples/                 # 사용 예제
│   └── basic_usage.py
│
├── docs/                     # 문서
│   ├── ARCHITECTURE.md       # 모델 아키텍처 문서
│   ├── ABLATION_STUDIES.md   # Ablation study 설명
│   ├── CHANGELOG.md          # 변경 이력
│   └── VISUALIZATIONS.md     # 시각화 가이드
│
└── results/                  # 실험 결과
    └── experiments/          # 실험별 결과 저장
```

## 주요 기능

- **1D-CNN + Transformer 하이브리드 아키텍처**: 로컬 feature 추출과 글로벌 의존성 캡처
- **3가지 Patchify 모드**:
  - `linear`: Linear embedding (MAE 원본 스타일)
  - `cnn_first`: CNN → Patchify (전체 시퀀스에 CNN 적용 후 패치화)
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
from mae_anomaly import Config, MultivariateTimeSeriesDataset, SelfDistilledMAEMultivariate

# 설정 생성
config = Config()
config.patchify_mode = 'linear'  # 'linear', 'cnn_first', 'patch_cnn'

# 데이터셋 생성
dataset = MultivariateTimeSeriesDataset(
    num_samples=1000,
    seq_length=100,
    num_features=5,
    anomaly_ratio=0.1
)

# 모델 생성
model = SelfDistilledMAEMultivariate(config)
```

### 실험 실행

```bash
# 2-stage 실험 실행 (Quick Search → Full Training → Visualization)
python scripts/run_experiments.py

# 기존 결과에 대해 시각화만 실행
python scripts/visualize_all.py --experiment-dir results/experiments/YYYYMMDD_HHMMSS
```

## 설정

`Config` 클래스의 주요 파라미터:

```python
# 데이터 파라미터
seq_length: int = 100           # 시퀀스 길이
num_features: int = 5           # Feature 수
num_train_samples: int = 10000  # 학습 샘플 수
num_test_samples: int = 2500    # 테스트 샘플 수

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

**마지막 업데이트**: 2026-01-22
