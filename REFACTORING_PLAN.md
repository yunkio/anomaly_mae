# 프로젝트 정리 및 리팩토링 계획

## 현재 상태 분석

### 파일 분류

#### 1. 메인 코드 (1,419 줄 - 모듈화 필요)
- `multivariate_mae_experiments.py` - 모든 기능이 하나의 파일에 집중됨
- `self_distilled_mae_anomaly_detection.py` - 구버전으로 보임 (삭제 예정)

#### 2. 테스트 파일 (5개)
- `test_experiment_fix.py`
- `test_implementation.py`
- `test_mae_quick.py`
- `test_masking_strategies.py`
- `test_visualization_fix.py`

#### 3. 검증/분석 스크립트 (4개)
- `analyze_results.py`
- `verify_mask_patterns.py`
- `verify_patch_masking.py`
- `verify_positional_encoding.py`

#### 4. 유틸리티 스크립트 (2개)
- `example_usage.py`
- `generate_visualizations.py`

#### 5. 문서 (13개 MD 파일)
**루트 레벨:**
- `README.md`
- `BUGFIX_SUMMARY.md`
- `ERROR_FIX_COMPLETE.md`
- `EXPERIMENT_ANALYSIS.md`
- `KEYERROR_FIX_SUMMARY.md`
- `MASKING_FIX_COMPLETE.md`
- `TEST_SUMMARY.md`

**description/ 폴더 (7개):**
- Various implementation and analysis documents

#### 6. 생성된 이미지 (11개 PNG - 루트 레벨)
- `input_projection_visualization.png`
- `mask_pattern_comparison.png`
- `patch_masking_verification.png`
- `patch_mode_visualization.png`
- `positional_encoding_effect.png`
- `positional_encoding_visualization.png`
- `roc_curve.png`
- `sample_reconstruction_0.png`
- `sample_reconstruction_1.png`
- `score_distribution.png`
- `training_history.png`

#### 7. 로그 파일 (2개 - 삭제 예정)
- `experiment_output.log`
- `experiment_output_v2.log`

#### 8. 실험 결과 (25개 폴더)
- `experiment_results/` - 많은 중간 실험 결과 포함

#### 9. 기타
- `requirements.txt`
- `__pycache__/` (자동 생성)
- `.claude/` (설정)
- `__marimo__/` (노트북)

---

## 문제점 분석

### 1. 코드 구조 문제
- **Monolithic Design**: 1,419줄의 단일 파일에 모든 기능 집중
  - Config, Dataset, Model, Trainer, Evaluator, ExperimentRunner 모두 포함
  - 유지보수 및 테스트 어려움
  - 재사용성 낮음

### 2. 파일 구조 문제
- **루트 디렉토리 오염**: 테스트, 스크립트, 이미지 모두 루트에 산재
- **일관성 없는 문서화**: 루트와 description/ 폴더에 분산
- **임시 파일 관리 부족**: 생성된 PNG, 로그 파일이 루트에 존재

### 3. 실험 결과 관리
- **과도한 중간 결과**: 25개의 실험 폴더 (대부분 테스트 중 생성)
- **보관 가치 낮은 결과**: 초기 디버깅 실험들

---

## 목표 구조

```
/home/ykio/notebooks/claude/
│
├── README.md                          # 메인 문서
├── requirements.txt                   # 의존성
├── setup.py                          # 패키지 설정 (NEW)
├── .gitignore                        # Git 무시 파일 (NEW)
│
├── mae_anomaly/                      # 메인 패키지 (NEW)
│   ├── __init__.py
│   ├── config.py                     # Configuration
│   ├── dataset.py                    # Dataset 클래스
│   ├── model.py                      # MAE 모델
│   ├── trainer.py                    # 학습 로직
│   ├── evaluator.py                  # 평가 로직
│   ├── experiment.py                 # ExperimentRunner
│   └── utils.py                      # 유틸리티 함수
│
├── scripts/                          # 실행 스크립트 (NEW)
│   ├── run_experiments.py           # 메인 실험 실행
│   ├── analyze_results.py           # 결과 분석
│   ├── generate_visualizations.py   # 시각화 생성
│   └── verify/                      # 검증 스크립트 (NEW)
│       ├── verify_mask_patterns.py
│       ├── verify_patch_masking.py
│       └── verify_positional_encoding.py
│
├── tests/                           # 테스트 파일 (NEW)
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_trainer.py
│   ├── test_evaluator.py
│   ├── test_experiment.py
│   └── integration/                 # 통합 테스트 (NEW)
│       ├── test_implementation.py
│       ├── test_masking_strategies.py
│       └── test_visualization_fix.py
│
├── examples/                        # 예제 코드 (NEW)
│   └── basic_usage.py              # example_usage.py 이동
│
├── docs/                           # 문서 (NEW)
│   ├── guides/                     # 사용 가이드
│   │   └── README.md
│   ├── bugfixes/                   # 버그 수정 히스토리
│   │   ├── BUGFIX_SUMMARY.md
│   │   ├── ERROR_FIX_COMPLETE.md
│   │   ├── KEYERROR_FIX_SUMMARY.md
│   │   └── MASKING_FIX_COMPLETE.md
│   ├── analysis/                   # 실험 분석
│   │   ├── EXPERIMENT_ANALYSIS.md
│   │   └── TEST_SUMMARY.md
│   └── implementation/             # 구현 상세 (description/ 이동)
│       └── [기존 description/ 내용]
│
└── results/                        # 실험 결과 (experiment_results/ 이름 변경)
    ├── latest/                     # 최신 결과 (심볼릭 링크)
    └── archived/                   # 보관할 결과만 (NEW)
        └── 20251230_021121/       # 최종 실험 결과만 보관
```

---

## 상세 실행 계획

### Phase 1: 분석 및 준비 (현재 진행 중)
- [x] 파일 목록 작성
- [x] 파일 분류 및 용도 파악
- [ ] 삭제할 파일 확정
- [ ] 모듈 분리 계획 수립

### Phase 2: 코드 모듈화 (가장 중요)
#### 2.1 multivariate_mae_experiments.py 분석
```bash
# 파일 구조 분석
- Lines 1-50: Imports and Config class
- Lines 51-150: Dataset class
- Lines 151-700: Model class
- Lines 701-900: Trainer class
- Lines 901-1000: Evaluator class
- Lines 1001-1419: ExperimentRunner class
```

#### 2.2 모듈 추출 계획
1. **config.py** (약 50줄)
   - `Config` 클래스
   - 모든 하이퍼파라미터 및 설정

2. **dataset.py** (약 100줄)
   - `MultivariateTimeSeriesDataset` 클래스
   - 데이터 생성 로직

3. **model.py** (약 550줄)
   - `SelfDistilledMAEMultivariate` 클래스
   - 모델 아키텍처
   - Masking 로직

4. **trainer.py** (약 200줄)
   - `Trainer` 클래스
   - 학습 루프
   - 손실 함수

5. **evaluator.py** (약 100줄)
   - `Evaluator` 클래스
   - 평가 메트릭 계산
   - 3-way evaluation

6. **experiment.py** (약 400줄)
   - `ExperimentRunner` 클래스
   - 실험 관리
   - 시각화

7. **utils.py** (약 20줄)
   - 공통 유틸리티 함수
   - JSON 직렬화 등

### Phase 3: 파일 정리
#### 3.1 삭제할 파일
```bash
# 로그 파일 (재생성 불필요)
- experiment_output.log
- experiment_output_v2.log

# 구버전 코드
- self_distilled_mae_anomaly_detection.py

# 루트 레벨 PNG (결과 폴더나 docs로 이동 가능)
- 모든 .png 파일 → docs/images/ 또는 삭제

# 불필요한 실험 결과 (최신 2-3개만 보관)
- experiment_results/20251229_* (초기 테스트)
- experiment_results/20251230_01* (디버깅 중)
```

#### 3.2 폴더 생성
```bash
mkdir -p mae_anomaly
mkdir -p scripts/verify
mkdir -p tests/integration
mkdir -p examples
mkdir -p docs/{guides,bugfixes,analysis,implementation,images}
mkdir -p results/archived
```

#### 3.3 파일 이동
```bash
# 문서 이동
mv BUGFIX_SUMMARY.md docs/bugfixes/
mv ERROR_FIX_COMPLETE.md docs/bugfixes/
mv KEYERROR_FIX_SUMMARY.md docs/bugfixes/
mv MASKING_FIX_COMPLETE.md docs/bugfixes/
mv EXPERIMENT_ANALYSIS.md docs/analysis/
mv TEST_SUMMARY.md docs/analysis/
mv description/* docs/implementation/

# 테스트 이동
mv test_*.py tests/integration/

# 스크립트 이동
mv analyze_results.py scripts/
mv generate_visualizations.py scripts/
mv verify_*.py scripts/verify/

# 예제 이동
mv example_usage.py examples/basic_usage.py

# 실험 결과 정리
mv experiment_results results
# 최신 결과만 archived로 이동
```

### Phase 4: 테스트 및 검증
#### 4.1 단위 테스트 작성
```python
# tests/test_config.py
# tests/test_dataset.py
# tests/test_model.py
# tests/test_trainer.py
# tests/test_evaluator.py
```

#### 4.2 통합 테스트 검증
```bash
# 기존 테스트가 여전히 작동하는지 확인
python -m pytest tests/
```

#### 4.3 예제 코드 검증
```bash
# 새로운 구조로 예제가 작동하는지 확인
python examples/basic_usage.py
```

### Phase 5: 문서화 업데이트
#### 5.1 README.md 업데이트
- 새로운 프로젝트 구조 설명
- 설치 방법
- 사용 예제
- 모듈별 설명

#### 5.2 setup.py 작성
```python
from setuptools import setup, find_packages

setup(
    name="mae-anomaly",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tqdm"
    ]
)
```

#### 5.3 .gitignore 작성
```
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
results/*/
!results/archived/
*.log
.claude/
```

---

## 리팩토링 원칙

### 코드 품질
1. **Single Responsibility**: 각 클래스는 하나의 책임만
2. **DRY (Don't Repeat Yourself)**: 중복 코드 제거
3. **Clear Naming**: 명확한 변수명 및 함수명
4. **Type Hints**: 타입 힌트 추가
5. **Docstrings**: 모든 클래스/함수에 문서화

### 파일 구조
1. **Logical Grouping**: 관련 기능을 같은 폴더에
2. **Flat is Better**: 너무 깊은 중첩 피하기
3. **Clear Separation**: 소스/테스트/문서/결과 명확히 분리

### 의존성
1. **Minimal Dependencies**: 필요한 것만 import
2. **No Circular Dependencies**: 순환 참조 방지
3. **Clear Hierarchy**: 상위 모듈은 하위 모듈만 import

---

## 예상 결과

### Before (현재)
```
파일 개수: 60+ 파일 (루트에 산재)
메인 코드: 1,419줄 단일 파일
테스트: 루트에 흩어져 있음
문서: 2곳에 분산
실험 결과: 25개 폴더 (정리 안됨)
```

### After (목표)
```
파일 구조: 체계적인 폴더 구조
메인 코드: 7개 모듈 (평균 100-200줄)
테스트: tests/ 폴더에 체계적으로 정리
문서: docs/ 폴더에 카테고리별 정리
실험 결과: 필요한 것만 보관
```

---

## 실행 순서

1. **분석 완료** (현재 단계)
   - 모든 파일 파악 완료
   - 분류 및 용도 확인 완료

2. **백업 생성**
   ```bash
   cp -r /home/ykio/notebooks/claude /home/ykio/notebooks/claude_backup
   ```

3. **모듈 추출** (가장 중요)
   - multivariate_mae_experiments.py를 7개 모듈로 분리
   - import 경로 업데이트

4. **파일 이동 및 삭제**
   - 불필요한 파일 삭제
   - 필요한 파일 적절한 위치로 이동

5. **테스트**
   - 모든 기능 정상 작동 확인
   - 테스트 코드 업데이트

6. **문서화**
   - README.md 업데이트
   - 각 모듈에 docstring 추가

---

## 다음 단계

사용자 확인 필요:
1. 이 계획에 동의하시나요?
2. 어떤 부분을 먼저 시작할까요?
   - 추천: Phase 2 (코드 모듈화)부터 시작
3. 삭제할 파일 목록에 추가/제외할 것이 있나요?
4. 폴더 구조에 수정할 부분이 있나요?
