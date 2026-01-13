# Self-Distilled MAE for Multivariate Time Series Anomaly Detection

깔끔하고 모듈화된 Self-Distilled Masked Autoencoder (MAE) 구현으로, 다변량 시계열 데이터의 이상 탐지를 수행합니다.

## 📁 프로젝트 구조

```
.
├── README.md                 # 메인 문서
├── requirements.txt          # Python 의존성
├── setup.py                  # 패키지 설정
│
├── mae_anomaly/              # 메인 패키지
│   ├── __init__.py
│   ├── config.py            # 설정 클래스
│   ├── dataset.py           # 데이터셋 구현
│   └── model.py             # MAE 모델 아키텍처
│
├── scripts/                  # 실행 스크립트
│   ├── run_full_experiments.py    # 전체 실험 스위트
│   ├── analyze_results.py         # 결과 분석
│   ├── generate_visualizations.py # 시각화 생성
│   └── verify/                    # 검증 스크립트
│       ├── verify_mask_patterns.py
│       ├── verify_patch_masking.py
│       └── verify_positional_encoding.py
│
├── tests/                    # 테스트 스위트
│   └── integration/          # 통합 테스트
│       ├── test_implementation.py
│       ├── test_masking_strategies.py
│       └── test_visualization_fix.py
│
├── examples/                 # 사용 예제
│   └── basic_usage.py
│
├── docs/                     # 문서
│   ├── bugfixes/            # 버그 수정 문서
│   ├── analysis/            # 실험 분석
│   └── implementation/      # 구현 상세
│
└── results/                  # 실험 결과
    └── archived/            # 보관된 결과
```

## ✨ 주요 기능

- **다중 마스킹 전략**: 4가지 마스킹 전략 지원
  - Patch-based (MAE-style)
  - Token-level (BERT-style)
  - Temporal (time-step masking)
  - Feature-wise (독립적 feature masking)

- **Self-Distillation**: Teacher-student 아키텍처와 discrepancy loss

- **이중 레벨 탐지**:
  - Sequence-level 이상 탐지
  - Point-level 이상 위치 파악

- **포괄적인 실험**:
  - Hyperparameter tuning
  - Ablation studies
  - Masking 전략 비교

## 🚀 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 또는 패키지로 설치
pip install -e .
```

## 💡 Quick Start

### 기본 사용법

```python
from mae_anomaly import Config, MultivariateTimeSeriesDataset, SelfDistilledMAEMultivariate

# 설정 생성
config = Config()

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
# 전체 실험 스위트 실행
python scripts/run_full_experiments.py

# 결과 분석
python scripts/analyze_results.py

# 마스킹 전략 검증
python scripts/verify/verify_mask_patterns.py
```

## ⚙️ 설정

`Config` 클래스의 주요 파라미터:

```python
# 데이터 파라미터
seq_length: int = 100           # 시퀀스 길이
num_features: int = 5           # Feature 수
num_train_samples: int = 2000   # 학습 샘플 수
num_test_samples: int = 500     # 테스트 샘플 수

# 모델 파라미터
d_model: int = 64               # 모델 차원
nhead: int = 4                  # Attention head 수
num_encoder_layers: int = 3     # Encoder 레이어 수
masking_ratio: float = 0.6      # 마스킹 비율
masking_strategy: str = 'patch' # 마스킹 전략

# 학습 파라미터
batch_size: int = 32
num_epochs: int = 50
learning_rate: float = 1e-3
```

## 🎭 마스킹 전략

### 1. Patch Masking (기본값)
- 연속된 시간 스텝 블록을 마스킹
- Vision Transformer (ViT) 패치와 유사
- 로컬 시간적 패턴 캡처에 최적

### 2. Token Masking
- 개별 시간 스텝을 무작위로 마스킹
- BERT 마스킹과 유사
- 전역 의존성 캡처에 효과적

### 3. Temporal Masking
- 모든 feature에 걸쳐 전체 시간 스텝 마스킹
- Feature 간 관계 보존
- 다변량 상관관계 학습에 유용

### 4. Feature-wise Masking
- 각 feature에 대해 독립적 마스킹
- 이질적 feature에 유연
- Feature별 패턴 학습에 효과적

## 🔬 실험

포괄적인 실험 지원:

### Hyperparameter Tuning
- Masking ratio: [0.5, 0.6, 0.75]
- Lambda (discrepancy 가중치): [0.3, 0.5, 0.7]
- Model dimension: [32, 64, 128]

### Ablation Studies
- Teacher-only
- Student-only
- No discrepancy loss
- No masking

### Masking 전략 비교
- Patch vs Token vs Temporal vs Feature-wise
- 다양한 데이터 타입에서의 성능

## 📊 결과

실험 결과 생성물:
- 모든 메트릭이 포함된 JSON 결과 파일
- 시각화 그래프:
  - Hyperparameter 비교
  - Ablation study 비교
  - Training curves
  - Performance heatmaps

결과는 `results/archived/YYYYMMDD_HHMMSS/`에 저장됩니다.

## 🧪 테스트

```bash
# 모든 테스트 실행
pytest tests/

# 특정 테스트 실행
python tests/integration/test_masking_strategies.py
```

## 📚 문서

`docs/`에서 포괄적인 문서 확인 가능:

- **Bug Fixes**: `docs/bugfixes/` - 모든 버그 수정 요약
- **Analysis**: `docs/analysis/` - 실험 분석
- **Implementation**: `docs/implementation/` - 상세 구현 문서

## 🆕 최근 업데이트

### 2025-01-09: 코드 리팩토링
- 프로젝트 구조 재구성
- 코드베이스 모듈화
- 문서 구조 개선
- 임시 파일 정리

### 2025-01-09: 마스킹 전략 수정
- Token vs Temporal masking 분리 (동일한 결과 생성 문제 해결)
- 각 전략의 독립적 구현
- 검증 테스트 추가

### 2024-12-30: 버그 수정
- JSON 직렬화 에러 수정
- Nested metrics 접근 KeyError 수정
- 모든 시각화 메서드 업데이트

## 📦 요구사항

- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- Matplotlib
- Scikit-learn
- tqdm
- pandas
- seaborn

## 📝 Citation

연구에 이 코드를 사용하시는 경우, 다음과 같이 인용해주세요:

```bibtex
@software{mae_anomaly_detection,
  title = {Self-Distilled MAE for Multivariate Time Series Anomaly Detection},
  year = {2025},
}
```

## 📄 License

MIT License

---

**상태**: ✅ 모든 기능 작동, 철저한 테스트 및 문서화 완료.

**마지막 업데이트**: 2025-01-09
