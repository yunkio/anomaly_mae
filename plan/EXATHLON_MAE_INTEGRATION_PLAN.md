# Exathlon Dataset — MAE 파이프라인 통합 + Notion 정리 실행계획

**작성일**: 2026-05-18
**문서 목적**: PSM 통합 패턴과 동일하게 Exathlon을 MAE 학습 파이프라인에 통합 + Baseline Notion 페이지 Q2/Q4 정리.
**전제**: Comparison 파이프라인 통합은 이미 완료 (loader, dispatch, configs, queue files, MD docs).
**원칙**: 기존 데이터셋(SMD/PSM)과 100% 동일한 패턴 유지. 신규 컨벤션 도입 금지.

---

## 0. 현재 상태 (점검만, 변경 금지)

### 0.1 완료된 작업

| 항목 | 위치 | 상태 |
|------|------|------|
| Exathlon 원본 데이터 → 19 FScustom features 추출 | `dataset/Exathlon/preprocess.py` | ✅ 93/93 trace |
| Raw loader `load_exathlon(app)` | `mae_anomaly/datasets/loaders.py:1234` | ✅ |
| DATASET_LOADERS 등록 (`exathlon_app{1,2,4,5,6,9}`) | `loaders.py:1858` | ✅ |
| UnifiedLoader 통합 | `comparison/data/unified_loader.py:267` | ✅ |
| Experiment configs (12개: 6 apps × standard/normalonly) | `comparison/experiment_configs.py` | ✅ |
| Queue 파일 4개 | `configs/baseline_exathlon_*.json` | ✅ |
| `docs/DATASET.md` Exathlon 섹션 | `docs/DATASET.md` | ✅ |
| `docs/CHANGELOG.md` 항목 | `docs/CHANGELOG.md` | ✅ |
| Baseline Comparison Notion 페이지 §2, §3, §9 skeleton | Notion ID `32087856b2078112b500c81664181ee7` | ✅ skeleton |

### 0.2 진행 중 / 예정 작업

| 항목 | 상태 |
|------|------|
| **Baseline Q1 (minmax)** — 16 models × 6 apps = 96 실험 | ✅ **96/96 완료** |
| **Baseline Q2 (zscore)** — 사용자 결정으로 **스킵** | ⏸ Skipped |
| **Baseline Q3 (minmax+normalonly)** — 96 실험 | 🔄 **진행 중** (app1 9/16) |
| **Baseline Q4 (zscore+normalonly)** — 사용자 결정으로 **스킵** | ⏸ Skipped |
| Baseline 결과 aggregation (Q1+Q3 only) | ⏳ 대기 |
| Baseline Notion 페이지 Q2/Q4 섹션 삭제 | ⏳ 대기 |
| Baseline Notion 페이지 Exathlon 결과 컬럼 추가 (Q1, Q3만) | ⏳ Q3 완료 후 |
| **MAE-side 통합 (본 계획서의 주요 작업)** | ⏳ 대기 |

### 0.3 MAE-side 미적용 (본 문서의 대상)

| 항목 | 변경 위치 | 상태 |
|------|----------|------|
| MAE base experiment 등록 | `scripts/run_base_experiments.py` DATASETS | ❌ |
| MAE 결과 병합 매핑 | `comparison/add_mae_results.py` MAE_SOURCE_DIRS | ❌ |
| MAE 방침 문서 갱신 | `CLAUDE.md`, `set_guideline.md`, `ablation_guideline.md`, `docs/ABLATION_STUDIES.md` | ❌ |
| Notion subpage 생성 | "MAE for Anomaly Detection" 하위 | ❌ |
| CHANGELOG에 MAE-side 엔트리 | `docs/CHANGELOG.md` | ❌ |

---

## 1. 현재 파이프라인 구조 (변경하지 말 것)

### 1.1 MAE Base Experiments

`scripts/run_base_experiments.py:243-291`의 `DATASETS` list가 single source of truth.

**현재 등록 (33개 = 5 base + 28 SMD)**:
- simulation, SWaT_A1A2, WaDi_A1, WaDi_A2, PSM (정적)
- smd_machine-1-1 ~ smd_machine-3-11 (동적, 28개)

**각 entry 시그니처**:
```python
{
    'key': str,                  # 사용자 노출 ID
    'loader': str,               # DATASET_LOADERS dict의 키
    'train_stride': int,         # 모두 21
    'normal50': bool,            # Exathlon: False
    'results_subdir': str,       # 결과 저장 경로
}
```

### 1.2 Exathlon-specific 결정사항

**왜 Exathlon은 6개 entry로 등록하는가?**

| Dataset | Entry 수 | 이유 |
|---------|:--------:|------|
| PSM | 1 (단일 'PSM') | 단일 contiguous stream |
| SMD | 28 (per-machine) | 머신마다 별도 데이터 |
| **Exathlon** | **6 (per-app)** | **앱마다 별도 데이터 + per-app 평가가 표준** |

→ SMD pattern을 그대로 따르되, 6 apps × 1 entry = 6 entries.

### 1.3 결과 디렉토리 구조 (예정)

```
results/experiments/{N}_{timestamp}_{desc}/
├── simulation/simulation/
├── SWaT/A1A2/
├── WaDi/A1/, WaDi/A2/
├── PSM/
├── SMD/{machine-X-Y}/      # 28개
└── Exathlon/                # ⬅ 신규
    ├── app1/
    ├── app2/
    ├── app4/
    ├── app5/
    ├── app6/
    └── app9/
```

각 디렉토리 내부 (`Exathlon/app{N}/`):
```
{best_config, epoch_metrics, training_histories, best_epoch_train_scores,
 experiment_metadata, anomaly_type_metrics, batch_profiling}.json/npz/csv
checkpoints/
epoch_scores/
visualization/
```

---

## 2. 단계별 실행 계획 (Phases)

### Phase A: 진행 중인 Baseline 마무리 (Q3 완료 대기)

- [ ] **A1**: Q3 (minmax_normalonly) 96 실험 완료 대기 (~6시간)
  - 모니터링: `tail -f /tmp/exathlon_Q3.log`
  - 모든 trace 에러 없이 완료 확인
- [ ] **A2**: Q1+Q3 결과 aggregation (`comparison/scripts/aggregate_exathlon.py`)
  - Per-model 6 apps 평균 산출
  - `aggregated.csv` 저장
- [ ] **A3**: Sanity check — Q1 vs Q3 비교, 비정상값 (NaN, 동일값 collision) 없는지 검증

### Phase B: Baseline Notion 정리 — Q2/Q4 제거

**페이지**: `https://www.notion.so/Baseline-Comparison-16-Models-6-Datasets-4-Conditions-32087856b2078112b500c81664181ee7`

#### B.1 §3 (실험 그룹) — Q2, Q4 섹션 전체 삭제
- `## Queue 2: Z-Score Normalization` 전체 블록 제거
- `## Queue 4: Z-Score + NormalOnly` 전체 블록 제거
- Q1, Q3만 유지

#### B.2 §4 (Cross-Queue Comparison Summary) — Q1/Q3만 남게 수정
- 비교 표에서 Q2, Q4 행 제거
- 💡 비교 설계 의도 callout 텍스트 수정: "Q1 vs Q3: NormalOnly 학습이 미치는 효과"

#### B.3 §9 (실험 결과) — Q2, Q4 ranking 테이블 제거
- `## 9.2 Q2 (zscore)` 전체 블록 (PAK_AUC_F1 + PAK_AUC_PRC 2개 테이블) 삭제
- `## 9.4 Q4 (zscore_normalonly)` 전체 블록 삭제
- 9.1 → 9.1 (Q1 그대로), 9.3 → 9.2 (Q3), 9.5 → 9.3 (종합 분석) 으로 renumber
- 종합 분석 callout: "4 Queue → 2 Queue" 로 텍스트 갱신

#### B.4 §11 (구 §10, 11-Metric Analysis) — 영향 없음
- 본 섹션은 Q3 only로 작성되어 있어 변경 불필요

### Phase C: Baseline Notion 정리 — Exathlon 결과 컬럼 추가

**전제**: Phase A 완료 (Q1+Q3 결과 산출)

#### C.1 §9 ranking 테이블에 Exathlon 컬럼 추가
- `Q1 — PAK_AUC_F1`, `Q1 — PAK_AUC_PRC`, `Q3 — PAK_AUC_F1`, `Q3 — PAK_AUC_PRC` (총 4 테이블)
- 각 테이블에 `Exathlon` 컬럼 추가 (SMD 다음, PSM 이전)
- Rank Avg를 5-DS 평균으로 갱신 (SWaT excl22 + WaDi A1/A2 + SMD + Exathlon)
- PSM은 informational only 유지

#### C.2 §11 (11-Metric Analysis) ranking 테이블에 Exathlon 컬럼 추가
- 11개 metric × 1 (Q3) = 11 테이블에 모두 Exathlon 컬럼 추가
- §11.12 종합 분석 텍스트 갱신: "4-DS Rank Avg → 5-DS Rank Avg"

### Phase D: MAE-side 통합 — `scripts/run_base_experiments.py` 등록

**대상 파일**: `/home/ykio/notebooks/claude/scripts/run_base_experiments.py`

#### D.1 EXATHLON_APP_IDS import 추가
**위치**: line 70 근처 (`from mae_anomaly.datasets.loaders import SMD_MACHINE_NAMES` 직후)
```python
from mae_anomaly.datasets.loaders import SMD_MACHINE_NAMES, EXATHLON_APP_IDS
```

#### D.2 Exathlon DATASETS 동적 entry 추가
**삽입 위치**: SMD_DATASETS 정의 직후 (현재 line 294)
```python
# Exathlon Per-App datasets (6 apps × 1 entry each, TimeSeAD 6-app convention)
# Apps {1, 2, 4, 5, 6, 9}. Apps 7/8 excluded (no disturbed traces / no undisturbed traces).
EXATHLON_DATASETS = []
for _app in EXATHLON_APP_IDS:
    EXATHLON_DATASETS.append({
        'key': f'exathlon_app{_app}',
        'loader': f'exathlon_app{_app}',
        'train_stride': 21,
        'normal50': False,
        'results_subdir': f'Exathlon/app{_app}',
    })
del _app  # Clean up loop variable
```

#### D.3 DATASETS 합치기 — 기존 SMD_DATASETS와 동일 패턴
```python
DATASETS = DATASETS + SMD_DATASETS + EXATHLON_DATASETS
```
(기존 `DATASETS = DATASETS + SMD_DATASETS` 줄 수정)

#### D.4 활성 데이터셋 개수 주석 갱신
```
33 → 39 datasets (5 base + 28 SMD + 6 Exathlon)
```

### Phase E: D Phase 검증 (1-epoch dry-run)

```bash
conda activate dc_vis

# 1. 데이터셋 목록 확인
python scripts/run_base_experiments.py --list 2>&1 | grep -i exathlon
# 기대: exathlon_app1, exathlon_app2, exathlon_app4, exathlon_app5, exathlon_app6, exathlon_app9

# 2. Dry-run with set C, 1 epoch (app1 only)
python scripts/run_base_experiments.py --dataset exathlon_app1 --set C \
    --config-override num_epochs=1
# 기대:
#   Dataset: exathlon_app1
#   Signals: (90897, 19)
#   Train ratio: 0.4862
#   Dynamic d_model: ?? (raw=190, p=10)
#   1 epoch 학습 + eval 완료
#   results/.../Exathlon/app1/{epoch_metrics.json, best_config.json}
```

**완료 기준**:
- [ ] `--list`에 `exathlon_app{1,2,4,5,6,9}` 6개 포함
- [ ] 1-epoch dry-run 무에러 완료
- [ ] `Exathlon/app1/epoch_metrics.json` 정상 (pak_auc_f1 키 존재)
- [ ] `Exathlon/app1/best_config.json`의 `num_features: 19`

### Phase F: 60-Config Retrofit Training (PSM 패턴 적용)

**중요 변경 (2026-05-18 사용자 지시)**: 단일 base experiment 학습이 아니라 **PSM과 동일한 60-config retrofit** 방식으로 진행.

#### F.0 PSM 패턴 분석 (참조)

PSM 60-model retrofit (`temp/run_psm_60_models.py`):
1. 60개 ablation Exp의 기존 디렉토리 (`results/experiments/{N}_...`)에 PSM/ 서브디렉토리 추가
2. 각 exp의 `best_config.json`에서 hyperparameter 추출 → `--config-override`로 전달
3. **Set C** 사용 (모든 exp), `--dataset PSM`
4. 우선순위: **274 first**, then ascending exp number
5. 274만 `KEEP_BEST_CKPT=1`
6. `/tmp/psm_60_models.json` = `{exp_num: {"dir": "...", "overrides": {...}}}`

#### F.1 Exathlon 60-Config Retrofit 계획

**대상 60 exps** (PSM과 동일):
```
140, 150, 153, 155, 157, 159, 160, 161, 165, 166,
169, 172, 173, 179, 184, 187, 190, 191, 198, 203,
208, 209, 211, 212, 214, 217, 221, 222, 223, 224,
226, 228, 229, 230, 231, 234, 236, 245, 247, 248,
249, 254, 256, 264, 265, 266, 269, 270, 271, 272,
273, 274, 275, 276, 277, 278, 279, 282, 283, 284
```

**우선순위**: **274 first**, 나머지는 274 이후 차례대로 (ascending).

**핵심 차이 (PSM vs Exathlon)**:
| 측면 | PSM | Exathlon |
|------|-----|----------|
| Dataset key | 'PSM' (단일) | 'exathlon_app{1,2,4,5,6,9}' (6개) |
| Training runs per exp | 1 | **6 (per-app)** |
| Total runs (60 exps) | 60 | **360** |
| Result subdir | `PSM/` | `Exathlon/app{N}/` |

#### F.2 스크립트 작성

**파일**: `temp/run_exathlon_60_models.py` (PSM 스크립트 변형)

```python
"""Chain script: Exathlon 60 Top-RA models × 6 apps retrofit.

Priority: 274 (all 6 apps) first → ascending exp num × 6 apps.
"""
import json, os, subprocess, sys, time
from datetime import datetime
from mae_anomaly.datasets.loaders import EXATHLON_APP_IDS  # [1,2,4,5,6,9]

PLAN_PATH = '/tmp/exathlon_60_models.json'  # reuse psm_60_models.json format
LOG_DIR = '/home/ykio/notebooks/claude/temp'
WORKDIR = '/home/ykio/notebooks/claude'

MODELS_ALL = [140, 150, ..., 284]  # 60개
ORDER = [274] + [n for n in sorted(MODELS_ALL) if n != 274]

# For each exp, train all 6 apps before moving to next exp
for n in ORDER:
    for app in EXATHLON_APP_IDS:
        cmd = [
            'conda', 'run', '-n', 'dc_vis',
            'python', '-u', 'scripts/run_base_experiments.py',
            '--set', 'C', '--no-wait',
            '--dataset', f'exathlon_app{app}',
            '--output-base', exp_dir,
            '--config-override', override_str,
        ]
        # ...
```

#### F.3 단계별 실행

- [ ] **F.3.a**: `/tmp/exathlon_60_models.json` 생성 (PSM 동일 형식)
- [ ] **F.3.b**: `temp/run_exathlon_60_models.py` 작성
- [ ] **F.3.c**: **274 단독** 6 apps 학습 (smoke test) → ~3-6시간
- [ ] **F.3.d**: 274 결과 검증 (`Exathlon/app{1,2,4,5,6,9}/epoch_metrics.json` 모두 정상)
- [ ] **F.3.e**: 나머지 59 exp × 6 apps 학습 — chain script로 background 실행 → 매우 오래 (~150-300시간)
- [ ] **F.3.f**: 모니터링 + 실패 시 재시작 로직

**예상 d_model (Exathlon 19 features, Set C)**:
- patch_size=10, num_features=19, raw=190 → d_model=192 (`resolve_dynamic_d_model` 자동)
- 274 best_config는 d_model=256 — override 우선 적용 시 256 (PSM과 동일)

⚠️ **사용자 확인 요청**: F.3.e (나머지 59 × 6) 시점 — 274 검증 직후 즉시? 또는 별도 trigger?

### Phase G: `comparison/add_mae_results.py` 매핑 추가

**전제**: Phase F 완료 (MAE 결과 생성됨)

**대상 파일**: `/home/ykio/notebooks/claude/comparison/add_mae_results.py`

```python
MAE_SOURCE_DIRS = {
    # ...existing entries...
    "psm": "results/PSM",
    # Exathlon — per-app
    "exathlon_app1": "results/Exathlon/app1",
    "exathlon_app2": "results/Exathlon/app2",
    "exathlon_app4": "results/Exathlon/app4",
    "exathlon_app5": "results/Exathlon/app5",
    "exathlon_app6": "results/Exathlon/app6",
    "exathlon_app9": "results/Exathlon/app9",
}
```

### Phase H: MAE 방침 문서 일괄 갱신

#### H.1 `CLAUDE.md`
```diff
- # Run base experiments (5 base + 28 SMD = 33 datasets: simulation, SWaT, WaDi A1/A2, PSM, SMD ×28)
+ # Run base experiments (5 base + 28 SMD + 6 Exathlon = 39 datasets: simulation, SWaT, WaDi A1/A2, PSM, SMD ×28, Exathlon ×6)
```

#### H.2 `set_guideline.md`
- 데이터셋 표 헤더 갱신
- Exathlon 전용 서브섹션 추가 (PSM 섹션 뒤에)
- 결과 디렉토리 구조 그림에 Exathlon 추가

#### H.3 `ablation_guideline.md`
- Dataset Types 표에 Exathlon 6 entries 추가

#### H.4 `docs/ABLATION_STUDIES.md`
- `DATASET_TYPE` 옵션 코멘트에 `exathlon_app1` 등 추가

### Phase I: `docs/CHANGELOG.md` MAE-side 엔트리

```markdown
## 2026-MM-DD: Exathlon dataset MAE-side integration (base_experiments registry)

### Summary
Exathlon 데이터셋 MAE base experiments 파이프라인 통합. Comparison 통합(2026-05-18)에 이어 MAE 학습/평가도 지원.

### 주요 변경
- scripts/run_base_experiments.py: EXATHLON_DATASETS 동적 entry × 6 추가
- comparison/add_mae_results.py: MAE_SOURCE_DIRS에 exathlon_app{N} × 6 매핑
- CLAUDE.md, set_guideline.md, ablation_guideline.md, docs/ABLATION_STUDIES.md 갱신

### MAE 학습 결과 (Set A/B/C, 6 apps 평균)
- TBD (Phase F 실행 후 채워질 부분)
```

### Phase J: Notion subpage 생성 — MAE for Anomaly Detection 하위

**부모 페이지**: `https://www.notion.so/MAE-for-Anomaly-Detection-31687856b20780e29fbcd961d69773ea`

**페이지 제목**: `Exathlon Dataset — MAE 파이프라인 통합 실행계획`

**내용 구조** (PSM subpage 템플릿 그대로):
- 0. 사전 점검 (Comparison 통합 완료 상태 확인 명령어)
- 1. 현재 파이프라인 구조 (변경 금지) — SMD 패턴 따라 6-app 등록
- 2. 단계별 실행 계획 (Phase D ~ I 옮겨담기)
- 3. 위험 요소 및 주의사항 (긴 train_ratio 변동, 19 features 작은 차원)
- 4. Acceptance Criteria
- 5. 참조 파일 (Single Source of Truth)
- 6. Exathlon 데이터 통계 (6 apps 표)
- 7. 실행 순서 요약
- 8. FAQ (Why 6 entries / How is it different from PSM/SMD)

### Phase K: 통합 검증

```python
"""Exathlon 통합 일관성 검증 — MAE + Comparison 양쪽."""
import sys, os, json, numpy as np
sys.path.insert(0, '/home/ykio/notebooks/claude')

# 1. MAE loader
from mae_anomaly.datasets.loaders import load_exathlon, DATASET_LOADERS, EXATHLON_APP_IDS
assert EXATHLON_APP_IDS == [1, 2, 4, 5, 6, 9]
for app in EXATHLON_APP_IDS:
    assert f'exathlon_app{app}' in DATASET_LOADERS

# 2. UnifiedLoader (Comparison)
from comparison.data.unified_loader import UnifiedLoader
loader = UnifiedLoader(dataset='exathlon', app=1, normalize_mode='minmax').load()
assert loader.features.shape[1] == 19

# 3. run_base_experiments DATASETS
src = open("/home/ykio/notebooks/claude/scripts/run_base_experiments.py").read()
for app in EXATHLON_APP_IDS:
    assert f"'exathlon_app{app}'" in src or f'"exathlon_app{app}"' in src

# 4. add_mae_results.py MAE_SOURCE_DIRS (after Phase G)
src = open("/home/ykio/notebooks/claude/comparison/add_mae_results.py").read()
for app in EXATHLON_APP_IDS:
    assert f'"exathlon_app{app}"' in src or f"'exathlon_app{app}'" in src

# 5. Comparison experiment_configs
from comparison.experiment_configs import EXPERIMENT_CONFIGS
for app in EXATHLON_APP_IDS:
    assert f'exathlon_app{app}' in EXPERIMENT_CONFIGS
    assert f'exathlon_app{app}_normalonly' in EXPERIMENT_CONFIGS

print("✅ ALL CHECKS PASSED")
```

### Phase L: Git commit & push

```bash
git add scripts/run_base_experiments.py comparison/add_mae_results.py \
        CLAUDE.md set_guideline.md ablation_guideline.md \
        docs/ABLATION_STUDIES.md docs/CHANGELOG.md
git commit -m "Feat: Exathlon dataset MAE-side integration (6 apps, 19 FScustom features)"
git push
```

---

## 3. 위험 요소 및 주의사항

### 3.1 데이터 일관성

| 위험 | 영향 | 완화 |
|------|------|------|
| Exathlon `train_ratio`가 앱마다 다름 (0.49~0.99) | sliding window 통계가 앱마다 다름 | 정상 — 자동 계산이므로 코드 변경 불필요 |
| App 4 test_len = 3,621 → 매우 작음 | 평가 신뢰성 낮음 | Caveat 표기, 평균 계산 시 weight 고려 검토 |
| 19 features → Set A patch_size=5 → raw=95 | d_model 결정 자동 | `resolve_dynamic_d_model` 자동 처리 |
| Train portion에 disturbed trace 절반 포함 → anomaly mixed in train (5-8%) | semi-supervised 가정 일부 위반 | SWaT(5.2%), PSM(6.2%)와 유사 수준 → 동일 처리 |
| 19 features에 1-difference, executor-average 등 transformations 이미 적용 | 추가 정규화 시 distribution shift 가능 | 그대로 minmax 사용 (z-score는 outlier에 취약 — Comparison Q2 검증으로 확인) |

### 3.2 학습 시간 예상 (Phase F)

Exathlon 6 apps × 3 sets × 50 epochs:
- App별 train size 다름 (44K ~ 348K samples)
- 평균 sliding window 약 1,000~16,000 windows
- 1 app × 1 set 50 epochs: ~15-40분
- **총 추정**: 6 apps × 3 sets × ~25분 = ~7.5시간 (V100 기준)

### 3.3 Notion Exp 119-290 페이지 영향

PSM 통합 시와 마찬가지로 기존 157개 ablation exp에 Exathlon 컬럼 추가하려면 모든 exp 재학습 필요 → 별도 결정.

**권장**: 신규 실험만 Exathlon 포함, 기존 exp는 별도 결정.

---

## 4. Acceptance Criteria

- [ ] **A**: Phase A (Baseline Q3) 완료 — 96/96 results 정상
- [ ] **B**: Phase B (Baseline Notion Q2/Q4 제거) 완료 — Notion 페이지에 Q1, Q3만 존재
- [ ] **C**: Phase C (Baseline Notion Exathlon 컬럼) 완료 — 4 ranking 테이블 + 11-metric 테이블에 Exathlon 컬럼
- [ ] **D**: Phase D (`run_base_experiments.py` 등록) — 6 apps 추가
- [ ] **E**: Phase E (1-epoch dry-run) — 6 apps 모두 무에러 완료
- [ ] **F**: Phase F (옵션, 본 학습) — Set A/B/C 50-epoch 학습 완료, pak_auc_f1 > 0.3
- [ ] **G**: Phase G (add_mae_results 매핑) — 6 entries 추가
- [ ] **H**: Phase H (문서 갱신) — 4개 MAE 방침 문서 모두 Exathlon 언급
- [ ] **I**: Phase I (CHANGELOG) — MAE-side 엔트리 추가
- [ ] **J**: Phase J (Notion subpage) — "MAE for Anomaly Detection" 하위에 페이지 생성
- [ ] **K**: Phase K (통합 검증 스크립트) — `✅ ALL CHECKS PASSED` 출력
- [ ] **L**: Phase L (Git commit & push) — 완료

---

## 5. 실행 순서 요약

```
Phase A: Baseline Q3 완료 대기                       (~6시간, 현재 진행 중)
Phase B: Notion baseline Q2/Q4 제거                  (~30분)
Phase C: Notion baseline Exathlon 컬럼 추가          (~45분)
Phase D: run_base_experiments.py Exathlon entry      (10분)
Phase E: 1-epoch dry-run 검증                        (~5-10분)
Phase F: (옵션) 본 학습 Set A/B/C × 6 apps           (~7-12시간)
Phase G: add_mae_results.py 매핑                     (5분)
Phase H: MAE 방침 문서 4개 갱신                       (30분)
Phase I: CHANGELOG 엔트리                            (5분)
Phase J: Notion MAE subpage 생성                     (45분)
Phase K: 통합 검증                                   (10분)
Phase L: Git commit & push                           (5분)
```

**총 소요 (Phase F 제외)**: 약 3-4시간
**Phase F 포함 시**: 약 11-16시간

---

## 6. 사용자 답변 (2026-05-18)

| Q | 답 | 비고 |
|---|---|------|
| Q1 | 학습은 Q3 baseline 완료 후 시작. PSM/Exathlon이 default DATASETS에 포함. **60-config retrofit** 패턴 (PSM처럼): 274 first → 나머지 ascending | F.3.a~f 단계 |
| Q2 | A (MAE subpage standalone) | PSM 패턴 |
| Q3 | A (§4 Q1 vs Q3만, callout 간소화) | |
| Q4 | A (§9.5 종합 분석 Q1/Q3 텍스트 재작성) | |
| Q5 | A (Phase G는 Phase F 완료 후) | |
| Q6 | A (Phase B 지금, Phase C는 Q3 완료 후) | |

## 6.1 추가 확인 필요 사항 (60-config retrofit 관련) ⚠️

### NQ1. **60 exp × 6 apps = 360 학습 진행 방식**
- (A) 274 6 apps 완료 후 즉시 나머지 59 × 6 자동 진행 (single chain script)
- (B) 274 6 apps만 먼저 → 사용자 검증 → 별도 trigger로 나머지
- (C) 274 단일 app (예: app1)만 먼저 → 6 apps 완료 → 나머지
- **권장**: (B) — 274 결과 confirm 후 대량 실행

### NQ2. **각 exp에서 어떤 Set 사용?**
- PSM은 모든 exp가 Set C로 학습 (config-override로 specific hyperparam 전달)
- Exathlon도 동일하게 Set C?
- **권장**: PSM 패턴 그대로 Set C

### NQ3. **6 apps 학습 순서 (per exp)**
- (A) app1, app2, app4, app5, app6, app9 (ID 오름차순)
- (B) 가장 작은 app부터 (app1 90K → app2 165K → app4 341K → app5 323K → app6 399K → app9 376K)
- (C) Random / 사용자 지정
- **권장**: (A) — ID 오름차순으로 단순화

### NQ4. **Aggregation script 자동 실행?**
- 각 exp의 6 apps 결과 → 평균 (또는 Notion 표용 단일 값)
- (A) 학습 직후 자동 aggregate (`comparison/scripts/aggregate_exathlon.py` 호출)
- (B) 별도 수동 trigger
- **권장**: (A) 자동

### NQ5. **Notion Exp 119-290 페이지에 Exathlon 컬럼 추가?**
- PSM은 §4에 60-model PSM 테이블 별도 추가했음
- (A) §4를 6-DS로 확장 (PSM 컬럼 옆에 Exathlon 컬럼 추가) — Rank Avg는 6-DS 평균
- (B) 새 §5 "Exathlon-Included Subset Leaderboard" 별도 섹션 추가
- (C) Notion 작업 보류 (학습 완료 후 결정)
- **권장**: (A) — 6-DS Rank Avg가 의미 있음

### NQ6. **현재 Q3 baseline GPU 점유 — 274 Exathlon 학습 시작 시점**
- Q3는 `comparison/run_baseline_queue.py`로 별도 process, 274 학습은 `scripts/run_base_experiments.py`로 별도 process
- (A) Q3 완료까지 무조건 대기 (GPU 충돌 회피)
- (B) GPU 여유 보면서 274 1 app만 시작 (Q3와 병렬)
- **권장**: (A) — 안전. Q3 끝나는 약 ~5-7시간 후 274 시작

### NQ7. **274 KEEP_BEST_CKPT 적용 범위**
- 6 apps 전부에 `KEEP_BEST_CKPT=1` 적용? (PSM은 단일 디렉토리)
- (A) 6 apps 모두 적용 (각 app 디렉토리에 checkpoint 보존)
- (B) 첫 번째 app (app1)만
- **권장**: (A) — 6 apps 모두 보존 (디스크 여유 있음)

---

## 7. 참조 파일

| 용도 | 파일 |
|------|------|
| Loader 구현 | `mae_anomaly/datasets/loaders.py:1234` `load_exathlon()` |
| Loader 레지스트리 | `mae_anomaly/datasets/loaders.py:1858` (6 entries) |
| App ID list | `mae_anomaly/datasets/loaders.py:1232` `EXATHLON_APP_IDS` |
| MAE base exp registry | `scripts/run_base_experiments.py` `DATASETS` (TODO Phase D) |
| Comparison dispatch | `comparison/data/unified_loader.py:267` |
| Comparison configs | `comparison/experiment_configs.py` (12 configs) |
| Queue files | `configs/baseline_exathlon_*.json` (4개) |
| Preprocessing | `dataset/Exathlon/preprocess.py` |
| Data files | `dataset/Exathlon/app{1,2,3,4,5,6,7,8,9,10}/*.csv` (93 traces) |
| Aggregation script | `comparison/scripts/aggregate_exathlon.py` |

---

## 8. Exathlon 데이터 통계

| App | Total Rows | Train Rows | Test Rows | Train Anom% | Test Anom% | #Anomaly Regions | #Train Traces | #Test Traces |
|:---:|:----------:|:----------:|:---------:|:-----------:|:----------:|:----------------:|:-------------:|:------------:|
| app1 | 90,897 | 44,192 | 46,705 | 5.24% | 13.24% | 9 | 7 | 2 |
| app2 | 164,950 | 118,230 | 46,720 | 3.89% | 26.46% | 9 | 5 | 2 |
| app4 | 340,994 | 337,373 | 3,621 | 4.77% | 17.34% | 11 | 7 | 1 |
| app5 | 322,775 | 269,387 | 53,388 | 5.26% | 7.29% | 21 | 11 | 4 |
| app6 | 399,102 | 348,832 | 50,270 | 1.39% | 8.84% | 11 | 13 | 2 |
| app9 | 375,594 | 326,571 | 49,023 | 2.25% | 12.47% | 14 | 8 | 2 |
| **Total** | **1,694,312** | **1,444,585** | **249,727** | — | — | **75** | — | — |
