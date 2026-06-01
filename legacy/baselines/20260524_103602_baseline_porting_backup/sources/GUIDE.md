# Baseline Comparison Pipeline Guide

이 문서는 `comparison/` 디렉토리의 베이스라인 비교 실험 파이프라인을 설명합니다.

---

## 1. 디렉토리 구조

```
comparison/
├── run_baseline.py          # 통합 베이스라인 실행기
├── run_baseline_queue.py    # 큐 오케스트레이터 (다중 실험 순차 관리)
├── baseline_common.py       # 공용 유틸리티 (메트릭, 모델 팩토리, 결과 I/O)
├── experiment_configs.py    # 실험 설정 (base + SMD 28개 + Exathlon 6개 자동 생성)
├── visualization.py         # 시각화 (epoch_metrics + PRC curve)
├── add_mae_results.py       # MAE 결과를 비교 테이블에 병합
├── MODELS.md                # 모델 출처/논문 정리 (Notion Section 12 참조)
├── baselines/               # 활성 실험 22개 (BASELINE_MODELS) + 참고용 코드 3개 (AnomalyBERT, CAROTS, CrossAD; 17절 참조)
│   ├── neural_base.py       # Neural 모델 공통 base 클래스
│   ├── random/              # RandomBaseline (Simple)
│   ├── sensor_range/        # SensorRangeDeviation (Simple)
│   ├── pca_error/           # PCAError (Simple)
│   ├── l2_norm/             # L2Norm (Simple)
│   ├── nn_distance/         # NNDistance (Simple)
│   ├── mlp/                 # MLPBaseline (Neural)
│   ├── mlpmixer/            # MLPMixerBaseline (Neural)
│   ├── transformer/         # TransformerBaseline (Neural)
│   ├── gcn_lstm/            # GCNLSTMBaseline (SOTA)
│   ├── anomaly_transformer/ # AnomalyTransformerBaseline (SOTA)
│   ├── tranad/              # TranADBaseline (SOTA)
│   ├── usad/                # USADBaseline (SOTA)
│   ├── dagmm/               # DAGMMBaseline (SOTA)
│   ├── gdn/                 # GDNBaseline (SOTA)
│   ├── omnianomaly/         # OmniAnomalyBaseline (SOTA)
│   ├── tfmae/               # TFMAEBaseline (SOTA, ICDE'24, dual MAE) — 2026-05-19 batch
│   ├── timesnet/            # TimesNetBaseline (SOTA, ICLR'23, FFT 2D conv) — 2026-05-19 batch
│   ├── dcdetector/          # DCdetectorBaseline (SOTA, KDD'23, dual attn) — 2026-05-19 batch
│   ├── memto/               # MEMTOBaseline (SOTA, NeurIPS'23, memory module + K-means) — 2026-05-19 batch
│   ├── moderntcn/           # ModernTCNBaseline (SOTA, ICLR'24 Spot, large-kernel DW) — 2026-05-19 batch
│   ├── crossad/             # CrossADBaseline (SOTA, NeurIPS'25, cross-scale + query lib) — 2026-05-19 batch
│   ├── catch/               # CATCHBaseline (SOTA, ICLR'25, channel-aware freq) — 2026-05-19 batch
│   └── npsr/                # NPSRBaseline (SOTA, NeurIPS'23, nominality scoring) — 2026-05-19 batch
├── data/
│   ├── __init__.py
│   └── unified_loader.py    # 단일 데이터 로더 (MAE raw loaders + normalization)
├── scripts/
│   ├── aggregate_exathlon.py    # Exathlon 6 apps 결과 집계
│   ├── add_teacher_only.py
│   ├── calc_teacher_only.py
│   └── show_combined_results.py
└── results/
    └── experiments/             # 큐 실험 결과
        └── {N}_{timestamp}_{desc}/
            ├── queue_summary.json
            ├── simulation/simulation/{model}/
            ├── SWaT/A1A2_full/{model}/
            ├── SWaT/A1A2_excl22/{model}/    # excl22 후처리 자동 생성
            ├── WaDi/A1/{model}/
            ├── WaDi/A2/{model}/
            ├── SMD/                          # SMD 실험
            │   ├── {machine-id}/{model}/         # 28머신 × 16모델 (큐 실행 직후 생성)
            │   └── results/{model}/results.csv   # 28머신 평균 — **별도 집계 스크립트** (scripts/aggregate_smd_results.py) 실행 후에만 생성
            ├── PSM/{model}/                  # PSM 단일 stream × 16모델
            └── Exathlon/                     # Exathlon (minmax 큐 Q1/Q3에만 존재)
                ├── app{1,2,4,5,6,9}/{model}/     # 6 apps × 16모델
                └── aggregated.csv                # 6 apps 평균 (comparison/scripts/aggregate_exathlon.py)
```

### 모델별 결과 파일

```
{model}/
├── epoch_metrics.json       # 매 epoch 메트릭 (MAE 동일 형식)
├── epoch_metrics.json.bak   # [optional] 후처리 스크립트 (add_teacher_only.py 등)가 원본 백업 후 생성
├── scores.npz               # key: anomaly_score (1D float32)
├── metadata.json            # 모델 정보, 실행 시간
├── visualization/
│   ├── epoch_metrics/       # [DL only] epoch별 성능 추이
│   │   ├── epoch_dashboard.png
│   │   ├── epoch_prc_auc.png
│   │   ├── epoch_f1_t.png
│   │   ├── epoch_pa_k_f1.png
│   │   └── epoch_pak_auc.png
│   └── best_model/
│       └── best_model_prc_curve.png
├── epoch_scores/            # [DL only] epoch별 scores
└── model/                   # [DL only] 학습된 가중치
```

> `.bak` 파일은 `comparison/scripts/add_teacher_only.py` / `calc_teacher_only.py` 등 후처리 스크립트가 `epoch_metrics.json`을 갱신할 때 원본을 보존하기 위해 만든 것. 큐 실행 자체는 생성하지 않음.

## 2. 핵심 원칙

1. **MAE 코드를 직접 import** — 데이터 로딩, 전처리, 지표 계산 모두 MAE 원본 사용
2. **MAE 수정 시 자동 반영** — comparison에 중복 코드 없음
3. **결과 형식 MAE 동일** — `epoch_metrics.json`, `scores.npz` 동일 키
4. **디렉토리 구조 MAE 동일** — `SWaT/A1A2_full`, `WaDi/A1` 등 MAE experiment #40+ 구조 일치
5. **excl22 자동 후처리** — SWaT 실험 완료 후 `A1A2_excl22/` 자동 생성

## 3. 데이터셋 (7개; Notion: "6 Datasets"는 Simulation 제외 카운트)

MAE experiment #40+과 동일한 데이터셋 + SMD/PSM/Exathlon:

| Dataset | Loader | Features | Dir | Q 적용 | 비고 |
|---------|--------|----------|-----|--------|------|
| Simulation | `load_simulation()` | 8 | `simulation/simulation` | Q1, Q3 | 합성 데이터 |
| SWaT A1+A2 | `load_swat_combined()` | 51 | `SWaT/A1A2_full` + `SWaT/A1A2_excl22` | Q1, Q3 | excl22 자동 생성, train_stride=3 |
| WaDi A1 | `load_wadi_14days_raw('A1')` | 123 | `WaDi/A1` | Q1, Q3 | 14일 normal + attack day |
| WaDi A2 | `load_wadi_14days_raw('A2')` | 123 | `WaDi/A2` | Q1, Q3 | 14일 normal + attack day |
| SMD (28 machines) | `load_smd_simple(machine=...)` | 32-38 | `SMD/{machine}` | Q1, Q3 | 28 머신, simple split (50/50) |
| PSM | `load_psm()` | 25 | `PSM` | Q1, Q3 | eBay server metrics, 단일 stream, simple split |
| **Exathlon (6 apps)** | `load_exathlon(app=N)` | **19** (FScustom) | `Exathlon/app{N}` | Q1, Q3 | Spark 분산처리, apps {1,2,4,5,6,9}, per-app disturbed-split |

> **Q2/Q4 (zscore) 큐는 더 이상 실행하지 않음.** 본 가이드는 minmax 기반 Q1 (full) + Q3 (normalonly) 두 큐만 다룬다. 과거의 Q2/Q4 결과 디렉토리는 `comparison/results/experiments/`에 남아 있지만 신규 실험에는 사용되지 않으며, `configs/baseline_queue_zscore*.json` 파일은 삭제됨.

### SMD (Server Machine Dataset)

28개 서버 머신의 시계열 데이터. SWaT/WaDi와 동일한 방식으로 train/test 분할.

**데이터 처리 방식 (`load_smd_simple`)**:
- Train = 원본 train 파일 (전부 정상, ~24K) + test 파일 앞 50% (~12K)
- Test = test 파일 뒤 50% (~12K)
- 상수 컬럼 제거 (38 → 32~38 features)

**Experiment config 이름 규칙**:
- `smd_{machine}` — non-normalonly (Q1)
- `smd_{machine}_normalonly` — normalonly (Q3)

**결과 집계**: `scripts/aggregate_smd_results.py`로 28머신 평균 → `SMD/results/{model}/results.csv`

### PSM (Pooled Server Metrics, eBay)

eBay 서버 모니터링 시계열 데이터 (Abdulaal et al., KDD 2021). SMD/SWaT와 동일한 50/50 split 패턴.

**데이터 처리 방식 (`load_psm`)**:
- Train = 원본 train 파일 (132,481 전부 정상) + test 파일 앞 50% (43,920)
- Test = test 파일 뒤 50% (43,921)
- 25 features (anonymized `feature_0` ~ `feature_24`), 상수 컬럼 없음
- NaN ~4,195개 → forward/backward-fill 처리
- `run_boundaries = [132481]` (orig_train / test_front 경계 — window 가로지름 방지)

**특징**:
- 단일 연속 stream (SMD처럼 머신별 분할 없음 → `psm` 키 1개만)
- 이상 region 72개 (전체 데이터 기준), 길이 median ~5, max ~9,000
- 이상 region 설명 없음 (익명 server incidents)

**Experiment config 이름 규칙**:
- `psm` — non-normalonly (Q1)
- `psm_normalonly` — normalonly (Q3)

**데이터 파일** (`dataset/PSM/`): `train.csv`, `test.csv`, `test_label.csv` (출처: [github.com/eBay/RANSynCoders](https://github.com/eBay/RANSynCoders), BSD-3-Clause 코드 / CC BY 4.0 데이터)

### Exathlon (Spark 분산처리 모니터링)

Apache Spark 스트리밍 작업의 anomaly detection 벤치마크 (Jacob et al., **VLDB 2021**, PVLDB 14(11):2613-2626). 6개 apps `{1, 2, 4, 5, 6, 9}` 활용 (TimeSeAD 6-app convention, apps 7/8은 구조적으로 invalid).

**데이터 처리 방식 (`load_exathlon(app=N)`)** — Per-App Disturbed-Split:
1. App `N`의 모든 trace 메타데이터 로드 후 `trace_id` 오름차순 정렬
2. **Undisturbed trace 전부**는 Train에 포함 (정상 학습용)
3. **Disturbed trace** (총 `N_dist`개): 앞쪽 `floor(N_dist / 2)`개 → Train, 나머지 `ceil(N_dist / 2)`개 → Test
4. **Train** = 모든 undisturbed traces + first floor(N_dist/2) disturbed traces (concat)
5. **Test** = remaining disturbed traces (anomaly label 포함)
6. 모든 trace concat 시 `run_boundaries`로 trace 경계 표시 (window 가로지름 방지)
7. **19 FScustom features**: 3 driver streaming delays + 8 1-difference + 6 5-executor avg+diff (원본 2,283 metric 중 도메인 전문가 선정 표준 subset)

**Experiment config 이름 규칙**:
- `exathlon_app{N}` — non-normalonly (Q1), N ∈ {1, 2, 4, 5, 6, 9}
- `exathlon_app{N}_normalonly` — normalonly (Q3), N ∈ {1, 2, 4, 5, 6, 9}

**결과 집계**: `comparison/scripts/aggregate_exathlon.py`로 6개 app 평균 → `Exathlon/aggregated.csv`

**데이터 파일** (`dataset/Exathlon/`):
- `preprocess.py`: GitHub raw API에서 93 trace zip 다운로드 + 19 FScustom 추출 (한 번만 실행)
- `app{1,2,4,5,6,9}/{trace_name}.csv`: `t` (timestamp) + `label` (0/1) + `f0~f18` (19 features)
- `ground_truth.csv`: 87 disturbed trace 메타데이터 (root_cause_start, extended_effect_end)
- 출처: [github.com/exathlonbenchmark/exathlon](https://github.com/exathlonbenchmark/exathlon) (CC BY-NC-SA 4.0 데이터 / Apache-2.0 코드)

### Normalization Mode

`--normalize-mode` CLI 옵션으로 정규화 방식 선택:
- `minmax` (현재 base queue 표준): Min-max scaling to [0,1] (train-only fit, clipped)
- `zscore` (코드 default, 현재 미사용): Z-score standardization (train-only fit). 큐 JSON에서 `normalize_mode: "minmax"`를 명시하여 base queue는 항상 minmax를 사용. `zscore` 옵션은 CLI에서는 여전히 작동하지만, 신규 비교 실험에서는 사용하지 않음.

### NormalOnly Variant

`variant: normalonly` — 학습 데이터에서 anomaly region 제거, normal 데이터만으로 학습.
segment-aware windowing 사용하여 segment 경계를 넘지 않는 window 생성.

## 4. 사용법 (`run_baseline.py`)

```bash
conda activate dc_vis

# 실험 목록 보기
python comparison/run_baseline.py --list-experiments

# 전체 baseline 실행
python comparison/run_baseline.py --experiment simulation --model all

# 단일 모델 실행
python comparison/run_baseline.py --experiment swat_a1a2 --model random

# 현재 상태 확인
python comparison/run_baseline.py --experiment simulation --list

# Normalization mode 설정
python comparison/run_baseline.py --experiment simulation --model all --normalize-mode minmax

# 강제 재실행
python comparison/run_baseline.py --experiment simulation --model random --force

# 출력 디렉토리 명시 (큐 결과 통합 시 사용)
python comparison/run_baseline.py --experiment psm --model all --output-base {existing_queue_dir}

# 모델 카테고리 필터
python comparison/run_baseline.py --experiment simulation --model all --only-simple    # Simple 5만
python comparison/run_baseline.py --experiment simulation --model all --skip-sota      # SOTA 8 제외

# Epoch override
python comparison/run_baseline.py --experiment simulation --model all --neural-epochs 5
python comparison/run_baseline.py --experiment simulation --model anomaly_transformer --at-epochs 3
python comparison/run_baseline.py --experiment simulation --model usad --sota-epochs 5
```

### 전체 CLI 옵션
- `--experiment, -e STR`: experiment_configs.py의 실험명 (필수, 또는 `--list-experiments`)
- `--model, -m STR | "all"`: 모델명 또는 `all` (필수)
- `--output-base PATH`: 큐 결과 디렉토리에 통합 시 사용 (없으면 single experiment dir 자동 생성)
- `--normalize-mode {minmax|zscore}`: 정규화 방식 (default: `zscore`)
- `--neural-epochs N`: Neural 모델 epoch override (default: 10)
- `--sota-epochs N`: SOTA 모델 epoch override (default: 10)
- `--at-epochs N`: anomaly_transformer epoch override (default: 10)
- `--eval-interval N`: epoch eval 간격 (default: 1)
- `--only-simple`: Simple 5개만 실행
- `--skip-sota`: SOTA 8개 제외
- `--force`: 기존 결과 덮어쓰기
- `--list`: 현재 결과 status 표만 출력

## 5. SWaT excl22 처리

SWaT 실험에서는 full 메트릭 외에 excl22 메트릭도 자동 계산:

1. `run_baseline.py`가 각 모델 실행 시 full + excl22 메트릭을 `epoch_metrics.json`에 함께 저장
2. 모든 모델 완료 후 `generate_excl22_directory()`가 자동 호출:
   - `SWaT/A1A2_full/{model}/`에서 excl22 메트릭 추출
   - `SWaT/A1A2_excl22/{model}/epoch_metrics.json` 생성 (excl22_ prefix 제거)
   - best epoch을 `pak_auc_f1` 기준으로 재선정

## 6. 큐 실행 (다중 실험)

### 통합 base 큐 구조 (PSM + Exathlon 포함, 39 실험)

PSM·Exathlon은 별도 큐가 아니라 **base queue JSON에 직접 포함**되어 있어 단일 큐 실행으로 7개 데이터셋 전체가 처리됨.

| Queue 파일 | 정규화 | variant | 실험 개수 |
|------------|--------|---------|-----------|
| `comparison/configs/baseline_queue_q1_minmax.json` (Q1) | minmax | full | **39** (Sim 1 + SWaT 1 + WaDi 2 + SMD 28 + PSM 1 + Exathlon 6) |
| `comparison/configs/baseline_queue_q3_minmax_normalonly.json` (Q3) | minmax | normalonly | **39** (동일 구성, `_normalonly` suffix) |

### 표준 실행 순서 (2026-05-22 업데이트 — **Q3 우선** 정책)

신규 batch (2026-05-22) 부터는 **Q3 → Q1 순서** 로 실행한다. NormalOnly 결과를 먼저 확보하여 anomaly 오염 영향이 없는 baseline 비교를 확정한 뒤, Q1 (full) 로 anomaly 노출 시 성능 변화를 측정한다.

```bash
# Q3 (normalonly) — minmax 정규화, normal-only 학습 (먼저 실행)
python comparison/run_baseline_queue.py --queue comparison/configs/baseline_queue_q3_minmax_normalonly.json

# Q1 (full) — minmax 정규화, anomaly 포함 학습 (Q3 완료 후 실행)
python comparison/run_baseline_queue.py --queue comparison/configs/baseline_queue_q1_minmax.json
```

**신규 SOTA 10개만 따로 돌리려면** (기존 15개는 `1_20260312_041500_*`, `3_20260312_203923_*` 디렉토리에 결과 보존, 신규 model subdir만 추가) — `scripts/run_new_sota_22.py` driver 사용:

```bash
python scripts/run_new_sota_22.py --condition q3   # Q3 먼저
python scripts/run_new_sota_22.py --condition q1   # Q1 나중
```

driver는 7개 신규 모델 (TFMAE/NPSR/TimesNet/DCdetector/MEMTO/ModernTCN/CATCH) 을 39 dataset 전부에 sequential 실행. **AnomalyBERT + CAROTS + CrossAD는 2026-05-22 제외** — 단일 (dataset, model) 작업당 3-37시간 소요로 39 데이터셋 pipeline 비현실적. 자세한 사유와 보존된 코드 경로는 17절 참조.

> **Q2/Q4 (zscore) 큐는 폐기.** `configs/baseline_queue_zscore*.json`은 삭제되었음. 현 비교 실험은 Q1·Q3 두 큐만 사용한다.

<callout>

**⚠ `queue_summary.json` overwrite 동작**: `run_baseline_queue.py`가 큐 종료 시 `{output_base}/queue_summary.json`을 **덮어쓴다**. 통합 큐 구조에서는 단일 `--output-base`에 단일 큐만 출력되므로 (PSM·Exathlon이 base에 포함되어 별도 호출이 없음) 이 이슈는 발생하지 않는다. 단, 디버깅을 위해 단독 실험을 동일 `--output-base`에 추가 실행하면 `queue_summary.json`이 덮어쓰일 수 있으므로 주의.

</callout>

### 6.1 단독 실험 실행 (`run_baseline.py --experiment ...`)

큐를 거치지 않고 특정 데이터셋·모델만 빠르게 검증하거나 결과를 보완할 때 사용:

```bash
# 단일 데이터셋 × 전체 모델 (큐 없이 단독 결과 디렉토리 생성)
python comparison/run_baseline.py --experiment psm --model all --normalize-mode minmax

# 기존 큐 결과 디렉토리에 단독 결과 병합 (output-base 명시)
python comparison/run_baseline.py --experiment exathlon_app1 --model all \
    --normalize-mode minmax \
    --output-base comparison/results/experiments/1_20260312_041500_baseline_minmax

# 단일 모델만 (e.g., 새 모델 검증)
python comparison/run_baseline.py --experiment simulation --model random --normalize-mode minmax
```

**단독 실행 vs 큐 실행 차이**:
- **단독 실행 (`run_baseline.py`)**: `--output-base` 없으면 timestamp 기반 single-experiment 디렉토리 자동 생성. `queue_summary.json` 미생성. 단일 데이터셋·모델·정규화 조합.
- **큐 실행 (`run_baseline_queue.py`)**: 여러 실험을 JSON으로 묶어 순차 실행. `queue_summary.json` 생성. `--output-base` 없으면 `{N}_{timestamp}_{desc}/` 자동 생성.

**기존 큐 결과에 PSM/Exathlon만 추가하고 싶을 때** (e.g., 과거 디렉토리 보완):
```bash
# 기존 Q1 디렉토리에 PSM 추가 (큐 JSON 수정 없이)
python comparison/run_baseline.py --experiment psm --model all --normalize-mode minmax \
    --output-base comparison/results/experiments/{기존_Q1_dir}
```
이렇게 하면 같은 `--output-base` 안에 `PSM/{model}/` 서브디렉토리가 추가되며, 기존 다른 데이터셋 결과는 보존됨.

### `run_baseline_queue.py` CLI 옵션
- `--queue, -q PATH`: 큐 JSON 파일 (필수)
- `--output-base PATH`: 출력 디렉토리 (없으면 timestamp 기반 `{N}_{YYYYMMDD_HHMMSS}_{desc}/` 자동 생성)
- `--desc, -d STR`: 자동 생성 디렉토리 이름의 desc 부분
- `--dry-run`: 실제 실행 없이 큐 검증만
- `--monitor`: 리소스 사용량 주기 출력

### 큐 JSON 형식

```json
{
    "description": "baseline_minmax",
    "experiments": [
        {
            "name": "sim_minmax",
            "experiment": "simulation",
            "model": "all",
            "normalize_mode": "minmax",
            "eval_interval": 1
        }
    ]
}
```

**지원 필드**:

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `name` | str | - | 실험 식별자 |
| `experiment` | str | **필수** | experiment_configs.py의 실험명 |
| `model` | str | - | 모델명 또는 `"all"` |
| `normalize_mode` | str | - | `zscore` 또는 `minmax` |
| `eval_interval` | int | - | DL 모델 epoch eval 간격 |
| `force` | bool | - | 기존 결과 덮어쓰기 |

## 7. 모델 분류 (활성 22개; 2026-05-19 신규 10개 SOTA 통합, 2026-05-22 AnomalyBERT/CAROTS/CrossAD 제외 → 7개 활성 + 3개 참고용)

| 분류 | 개수 | 모델 | 특징 |
|------|------|------|------|
| **Simple** | 5 | `random`, `sensor_range`, `pca_error`, `l2_norm`, `nn_distance` | 학습 없음 — 통계적/거리 기반 |
| **Neural** | 3 | `mlp`, `mlpmixer`, `transformer` | 외부 training loop, per-epoch eval, **10 epoch 통일** (2026-05-22 변경 — 이전 20 epoch 정책 폐기) |
| **SOTA (legacy)** | 7 | `gcn_lstm`, `anomaly_transformer`, `tranad`, `usad`, `dagmm`, `gdn`, `omnianomaly` | 내부 training loop + epoch_callback, 10 epoch (default) |
| **SOTA (new, 2026-05-19)** | 7 | `tfmae` (ICDE'24), `timesnet` (ICLR'23), `dcdetector` (KDD'23), `memto` (NeurIPS'23), `moderntcn` (ICLR'24 Spot), `catch` (ICLR'25), `npsr` (NeurIPS'23) | 내부 training loop + epoch_callback, 10 epoch. 각 모델별 distinct objective — `MODELS.md` 16-22번 참조. |
| **참고용 (코드만 유지, 실험 제외)** | 3 | `anomalybert` (ICLR'23 WS), `carots` (ICML'25), `crossad` (NeurIPS'25) | timing outlier로 실험 queue 제외. 17절 참조. |

**주의 (신규 7개 활성 모델):**
- `npsr`은 `performer-pytorch` 의존 (2026-05-22 `--no-deps` 설치 완료, torch 2.4.1.post302 무변동). 미설치 환경에서는 `nn.MultiheadAttention` 자동 fallback.
- `crossad` / `catch`는 2026-05-22 **upstream 원본 코드 기반 재구현 완료** (이전 paper-architecture 추측 기반 구현은 `.trash/0522/` 백업). 원본 알고리즘 그대로 vendoring + 디바이스 비종속화만 적용. (CrossAD는 timing outlier로 실험 queue에서 제외, 17절 참조.)
- LLM/VLM/foundation LM 기반 baseline은 정책상 제외 (GPT4TS/LLM-TSAD 등). AnomalyBERT는 BERT 아키텍처(encoder-only Transformer)만 차용한 random-init self-supervised 모델로 LLM 아님 → 17절 timing outlier 사유로 제외.
- 상세 명세: `/plan/SOTA_BASELINE_10_INTEGRATION_PLAN.md`, 모델별 설명은 `comparison/MODELS.md` (sections 16–22 활성 + 23-25 참고용).

**핵심 하이퍼파라미터 (`default` preset, 전 데이터셋 동일; `baseline_common.py:_get_default_model_params`)**:
- 모든 Neural + SOTA 모델 **`epochs=10` (코드 default)**. 현재 `comparison/results/experiments/`의 결과도 10 epoch.
  - ⚠ Notion 페이지의 "20" 표기는 이전 설정 기준이며, **현재 코드 default는 10**. 두 값 중 신뢰 우선순위: **코드 > Notion**. `--neural-epochs N` / `--sota-epochs N` / `--at-epochs N` CLI 옵션 또는 `experiment_configs.py`의 `default_neural_epochs` 필드로 override 가능.
- `seq_len`: 대부분 모델 5-10, `anomaly_transformer`/`omnianomaly`는 100
- 특이 모델: `anomaly_transformer` (d_model=512, n_heads=8, e_layers=3), `omnianomaly` (hidden=100, z_dim=3)

**Random seed**:
- `random` 모델: `seed=42` 명시 (deterministic)
- Neural/SOTA: 명시적 `torch.manual_seed` / `np.random.seed` 호출 **없음** → 다중 실행 간 미세한 결과 변동 가능. Reproduction 시 동일 결과 보장하려면 코드 추가 필요.

`experiment_configs.py:STANDARD_BASELINES`가 단일 source of truth (활성 22개 리스트, 2026-05-19 batch 신규 7개 SOTA 포함; AnomalyBERT + CAROTS + CrossAD는 2026-05-22 timing outlier로 실험 queue에서 제외 — 17절 참조).

## 8. 결과 형식

### epoch_metrics.json

```json
{
  "eval_interval": 1,
  "epochs": [
    {
      "epoch": 1,
      "roc_auc": 0.82, "prc_auc": 0.66, "f1_t": 0.70,
      "pak_auc_f1": 0.66, "pak_auc_prc_auc": 0.65,
      "pa_0_f1": 0.96, "pa_50_f1": 0.82, "pa_100_f1": 0.27,
      "excl22_prc_auc": 0.58,
      "excl22_pak_auc_f1": 0.55
    }
  ]
}
```

### scores.npz

```python
np.load('scores.npz')['anomaly_score']  # (n_test_timesteps,) float32
```

## 9. 시각화

### DL 모델 (epoch 학습)
각 모델 완료 후 자동 생성:
- `visualization/epoch_metrics/`: epoch_prc_auc.png, epoch_f1_t.png, epoch_pa_k_f1.png, epoch_pak_auc.png, epoch_dashboard.png
- `visualization/best_model/`: best_model_prc_curve.png

### Simple 모델
- `visualization/best_model/`: best_model_prc_curve.png

## 10. 모니터링 (CRITICAL)

### 실행 원칙

> **⚠️ `nohup`, `&`, `conda run` 사용 금지.**
> 반드시 Bash tool의 `run_in_background=true` 파라미터로 실행하고, `TaskOutput`으로 출력 확인.

```bash
# 실험 시작 (run_in_background=true 사용)
# Bash tool에서:
#   command: /home/ykio/anaconda3/envs/dc_vis/bin/python comparison/run_baseline_queue.py --queue configs/baseline_queue_minmax.json
#   run_in_background: true
#   timeout: 600000

# 출력 확인 (TaskOutput tool 사용)
#   task_id: {task_id}
#   block: false
#   timeout: 5000
```

### 실시간 모니터링 방법

`TaskOutput`의 출력 파일 경로를 알고 있으므로, 아래 명령어로 실시간 확인:

```bash
# 최근 출력 확인
tail -50 /tmp/claude-1000/-home-ykio-notebooks-claude/tasks/{task_id}.output

# 완료/에러 상태
grep -E "completed|ERROR|FAILED|KILLED|EXIT CODE" /tmp/claude-1000/-home-ykio-notebooks-claude/tasks/{task_id}.output

# 에러 확인
grep -iE "error|exception|killed|OOM|SIGKILL|Traceback" /tmp/claude-1000/-home-ykio-notebooks-claude/tasks/{task_id}.output
```

### 모니터링 주기 및 TODO 관리

실험 실행 중 **능동 점검 10분 간격** (사용자 2026-05-22 지시 — ScheduleWakeup 600s) + **백그라운드 60초 폴링 로그** (`/tmp/0522_monitor.log` 형태) 병행. **TODO list를 생성**하여 각 큐의 진행 상태를 추적. Context가 잘려도 TODO list가 현재 상태의 유일한 진실 소스 역할을 해야 한다.

**TODO list 필수 항목** (현 Q3+Q1 2-queue 흐름, **Q3 우선** — 2026-05-22):
1. Q3 (`baseline_queue_q3_minmax_normalonly.json`, 39 실험) 실행 + 모니터링
2. Q3 완료 확인 → Q1 시작
3. Q1 (`baseline_queue_q1_minmax.json`, 39 실험) 실행 + 모니터링
4. Q1 완료 확인 → 최종 결과 검증 (집계 스크립트 + Notion 갱신)

### 기준 메트릭

| # | 항목 | 설명 |
|---|------|------|
| 1 | **PRC** | PRC-AUC |
| 2 | **PAK_AUC_F1** | PA%K F1 AUC — **best epoch 선정 기준** |
| 3 | **PAK_AUC_PRC** | PA%K PRC AUC |
| 4 | **F1_T** | Time-series F1 |

### 모니터링 체크리스트

```bash
# 1. GPU 상태
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# 2. RAM 상태
free -h | grep Mem

# 3. Python 프로세스 확인
ps aux | grep python | grep -v grep | grep -v vscode

# 4. 큐 진행 상태 (task output 파일 직접 확인)
tail -50 /tmp/claude-1000/-home-ykio-notebooks-claude/tasks/{task_id}.output

# 5. 완료된 모델/데이터셋 수
grep -E "completed" /tmp/claude-1000/-home-ykio-notebooks-claude/tasks/{task_id}.output | tail -10

# 6. 에러 확인
grep -iE "error|exception|killed|OOM|SIGKILL|Traceback|EXIT CODE" /tmp/claude-1000/-home-ykio-notebooks-claude/tasks/{task_id}.output
```

### 모니터링 보고 형식

```
**모니터링 #{N} — {timestamp}**
| 항목 | 상태 |
|------|------|
| 큐 | Q{N} ({name}) — {M}/4 |
| 데이터셋 | {dataset} {completed}/{total} |
| 현재 모델 | {model} Epoch {ep}/{total_ep} |
| GPU | {used} / {total} MiB ({util}%) |
| RAM | {used} / {total} |
| 에러 | {있음/없음} |
```

### 큐 완료 시 다음 큐 자동 시작

큐가 완료되면 즉시 다음 큐를 `run_in_background=true`로 시작:
- Q3 완료 → Q1: `--queue comparison/configs/baseline_queue_q1_minmax.json`
- Q1 완료 → 종료 (집계 스크립트 + Notion 갱신 단계로 진행)

### 이상 징후 및 대응

| 증상 | 원인 | 대응 |
|------|------|------|
| 프로세스 사라짐 (EXIT CODE -9) | OOM | GPU 메모리 확인 → batch_size 줄여서 재시작 |
| 특정 모델 이후 모든 모델 누락 | 이전 모델 crash로 프로세스 종료 | 로그에서 마지막 성공 모델 확인 → 재실행 |
| PRC가 계속 ~0.1 | 학습 실패 | loss 로그 확인 → config 점검 |
| GPU util 0%이지만 프로세스 살아있음 | CPU eval 중 또는 hang | tail로 마지막 출력 시간 확인 |

## 11. 실험 큐 구성 (Q1, Q3 — 현 active)

| Queue | 정규화 | 학습 데이터 | 실험 config suffix | 실험 수 | PSM | Exathlon |
|-------|--------|------------|-------------------|---------|-----|----------|
| Q1 | minmax | Full (anomaly 포함) | (none) | 39 | ✓ | ✓ |
| Q3 | minmax | NormalOnly (정상만) | `_normalonly` | 39 | ✓ | ✓ |

> Q2 (zscore full) / Q4 (zscore normalonly) 는 폐기. 큐 JSON 파일도 삭제. 현재 `comparison/results/experiments/` 에는 `2_*`/`4_*` 디렉토리도 존재하지 않으며 (이전 정리 시 제거됨), 향후 재실행 계획 없음.

각 큐별 결과 디렉토리 예시:
- `1_20260312_041500_baseline_minmax/` (Q1, **historical**)
- `3_20260312_203923_baseline_minmax_normalonly/` (Q3, **historical**)

**디렉토리 번호 매김 주의**: `make_numbered_baseline_dir()`는 `max(기존 N) + 1`로 자동 채번한다 ([baseline_common.py:1229](baseline_common.py)). 현 상태에서 Q1 큐를 다시 실행하면 디렉토리는 `4_*`로 생성됨 (N=1, 3 → max=3 → next=4). "Q1=`1_*`" 매핑은 historical convention이며 enforced되지 않는다. 같은 번호로 reproduce하려면 `--output-base comparison/results/experiments/1_{timestamp}_baseline_minmax` 형태로 직접 지정.

## 12. SMD / Exathlon 집계

### SMD (28 머신 평균)
SMD는 baseline queue JSON에 28개 머신 실험을 포함하여 실행. 결과 집계는 project root의 스크립트 사용:

```bash
# SMD 28 머신 평균 → SMD/results/{model}/results.csv
python scripts/aggregate_smd_results.py --queue-dir comparison/results/experiments/{queue_dir}
```

### Exathlon (6 apps 평균)
Exathlon은 6개 app별로 학습/평가 후 집계 (comparison/scripts 내):

```bash
# Exathlon 6 apps 평균 → Exathlon/aggregated.csv
python comparison/scripts/aggregate_exathlon.py --queue-dir comparison/results/experiments/{queue_dir}
```

## 13. MAE 결과 병합 (`add_mae_results.py`)

`run_baseline_queue.py`가 만든 베이스라인 결과 트리(`comparison/results/experiments/{N}_*/`)와는 **별도로**, MAE 자체 실험 결과를 비교용 `results.json` 형식으로 변환하여 통합한다. 결과 파일은 큐 디렉토리가 아닌 **`comparison/results/{dataset_dir}/results.json`** 에 쓰인다 (예: `comparison/results/WaDi/A1/results.json`).

### 기본 사용법 (auto mode)

```bash
# MAE_SOURCE_DIRS 에 등록된 experiment 이름이면 자동 모드 사용 가능
python comparison/add_mae_results.py --experiment wadi_14days_A1
python comparison/add_mae_results.py --experiment swat_a1a2
python comparison/add_mae_results.py --experiment psm

# 등록된 후보 MAE 실험 디렉토리 조회 (병합 X)
python comparison/add_mae_results.py --experiment wadi_14days_A1 --discover
```

자동 모드는 `MAE_SOURCE_DIRS` 매핑 + `BEST_MAE_CONFIGS` 하드코딩이 있으면 해당 best run을 가져오고, 없으면 PRC 기준 best run을 auto-discover.

### 수동 모드 (custom MAE run 지정)

```bash
python comparison/add_mae_results.py --experiment wadi_A1 \
    --mae-dir results/WaDi/A1/20260202_222231 \
    --name mae_custom \
    --scoring combined
```

| 옵션 | 필수 | 설명 |
|------|------|------|
| `--experiment, -e` | ✓ | `experiment_configs.py:EXPERIMENT_CONFIGS` 의 키 또는 `MAE_SOURCE_DIRS` 의 키 |
| `--mae-dir` | manual mode 시 ✓ | 특정 MAE 실험 디렉토리 (project root 기준 상대 경로 또는 절대 경로) |
| `--name` | `--mae-dir` 사용 시 ✓ | `results.json` 내 model 이름 (e.g. `mae_v2`, `mae_teacher_only`) |
| `--scoring` | - | `combined` (default) / `total` / `teacher` / `teacher_recon` / `disc` |
| `--discover` | - | 후보 MAE 디렉토리만 출력 (병합 X) |

### Scoring 종류

| `--scoring` 값 | 의미 |
|--------|------|
| `combined`, `total` | adaptive/total_loss scoring (기본 권장) |
| `teacher`, `teacher_recon` | reconstruction-only metric |
| `disc` | discrepancy-only metric (legacy) |

### 출력 위치 및 결과 형식

```
comparison/results/
└── {results_dir_name}/                     # experiment_configs.py 의 `results_dir_name`
    └── results.json                        # MAE + baseline 통합 metric 표 (모델별 dict)
```

`results.json` 내부 구조 (`models` 키에 모델별 metric dict):

```json
{
  "experiment": "wadi_14days_A1",
  "timestamp": "2026-...",
  "models": {
    "mae_default": {"prc_auc": ..., "pak_auc_f1": ...},
    "mae_teacher_only": {...}
  }
}
```

> baseline 결과를 동일 `results.json` 에 병합하려면 별도 스크립트 (`comparison/scripts/show_combined_results.py` 등) 사용 — `add_mae_results.py` 단독은 MAE 모델만 다룬다.

### 일관성 주의

- `--experiment` 값은 **MAE 실험 이름** (예: `wadi_14days_A1`) 이지 baseline queue 이름 (`wadi_A1_minmax`) 이 아님.
- `MAE_SOURCE_DIRS` (`comparison/add_mae_results.py` 상단) 가 단일 진실 소스. 미등록 experiment 은 manual mode 필수.
- 베이스라인 큐 디렉토리 (`comparison/results/experiments/{N}_*/`) 의 `epoch_metrics.json` 은 이 스크립트와 무관 (이쪽은 큐가 직접 작성).

## 14. 핵심 참조 파일

| 용도 | 파일 |
|------|------|
| 단일 실험 실행 | `comparison/run_baseline.py` |
| 큐 실행 (다중 실험) | `comparison/run_baseline_queue.py` |
| 실험 설정 (model + dataset configs) | `comparison/experiment_configs.py` |
| 공용 유틸리티 (메트릭, 모델 팩토리) | `comparison/baseline_common.py` |
| 시각화 | `comparison/visualization.py` |
| 데이터 로더 (UnifiedLoader) | `comparison/data/unified_loader.py` |
| MAE 결과 병합 | `comparison/add_mae_results.py` |
| 큐 JSON | `configs/baseline_queue_*.json` |
| 평가 (MAE 공유) | `mae_anomaly/evaluator.py` |
| SMD loader | `mae_anomaly/datasets/loaders.py` → `load_smd_simple()` |
| Exathlon loader | `mae_anomaly/datasets/loaders.py` → `load_exathlon(app=N)` |
| PSM loader | `mae_anomaly/datasets/loaders.py` → `load_psm()` |
| **SMD 결과 집계** | `scripts/aggregate_smd_results.py` (project root, comparison 외부) |
| **Exathlon 결과 집계** | `comparison/scripts/aggregate_exathlon.py` |
| 모델 출처/논문 | `comparison/MODELS.md` |

## 15. 새 모델/데이터셋 추가 절차

새 baseline 모델 또는 데이터셋을 추가할 때 수정해야 할 모든 위치를 명시. 한 곳이라도 누락하면 결과가 일관되지 않거나 실행이 실패한다.

### 15.1 새 baseline 모델 추가

수정 위치 6곳:

| # | 파일 | 작업 |
|---|------|------|
| 1 | `comparison/baselines/{model_name}/__init__.py` | 모델 클래스 export |
| 2 | `comparison/baselines/{model_name}/model.py` | 모델 구현 (Simple: `fit(train_x)`+`score(test_x)→1D scores` / Neural: `neural_base.NeuralBaselineBase` 상속 / SOTA: 자체 `fit(train_X, epoch_callback)`+`predict(test_X)→1D scores` 패턴, 입출력 sliding-window 호환 필수) |
| 3 | `comparison/baseline_common.py:_get_default_model_params()` | 모델 hyperparameter dict 등록 |
| 4 | `comparison/baseline_common.py:create_model()` | if-elif 체인에 모델 instantiation 추가 (Simple/Neural/SOTA 카테고리에 맞게) |
| 5 | `comparison/baseline_common.py:SIMPLE_MODELS` 또는 해당 카테고리 list | 모델명 추가 (category gating용) |
| 6 | `comparison/experiment_configs.py:STANDARD_BASELINES` | 23-list에 모델명 추가 |

**Neural 모델 인터페이스 요구**:
- `epoch_callback(epoch, metrics)`: per-epoch 메트릭 export
- `score(test_x) → np.ndarray (n_timesteps,)`: 1D anomaly score
- training_history 저장 (선택)

검증: 추가 후 `python comparison/run_baseline.py --experiment simulation --model {new_model}` 로 단독 실행하여 결과 디렉토리 구조 확인.

### 15.2 새 데이터셋 추가

수정 위치 5곳:

| # | 파일 | 작업 |
|---|------|------|
| 1 | `mae_anomaly/datasets/loaders.py` | `load_xxx()` 함수 신규 — `(signals, point_labels, anomaly_regions, feature_names, train_ratio, data_info)` 6-tuple 반환 |
| 2 | `mae_anomaly/datasets/loaders.py:DATASET_LOADERS` 레지스트리 | `{'xxx': load_xxx}` 등록 |
| 3 | `comparison/data/unified_loader.py:UnifiedLoader.load()` | if-elif chain에 dispatch 추가 (line ~217-281) |
| 4 | `comparison/experiment_configs.py:EXPERIMENT_CONFIGS` | `loader_kwargs`, `results_dir_name`, `dataset_name`, `model_preset`, `train_stride`, `has_excl22`, `segment_aware_training`, `all_models_list` 모두 명시 |
| 5 | `configs/baseline_queue_minmax.json` + `configs/baseline_queue_minmax_normalonly.json` (active 2개) | 두 큐에 새 실험 추가 — base queue가 통합 단일 소스. (별도 큐 JSON은 만들지 말 것) |

**필수 필드 (data_info)**:
- `run_boundaries`: sliding window가 cross 못 할 경계 list
- `train_attack_ratio`, `test_attack_ratio`, `n_features`

**normalonly variant 지원** (Q3 용):
- loader에 `variant='normalonly'` 인자 처리: anomaly region 제거 후 segment-aware concat
- `experiment_configs.py`에서 별도 entry로 `{name}_normalonly` 추가 (`segment_aware_training: True`)

**excl22 mode 지원** (SWaT 같은 단일 큰 anomaly):
- `experiment_configs.py`의 entry에 `has_excl22: True` 설정
- `run_baseline.py`가 자동으로 `find_swat_largest_region()` + `compute_metrics_with_exclusion()` 호출 후 별도 디렉토리 생성

검증: 추가 후 `python comparison/run_baseline.py --experiment {new_dataset} --model random` (가장 빠른 simple 모델)로 단독 실행하여 결과 디렉토리 + `scores.npz` + `epoch_metrics.json` 생성 확인.

### 15.3 일관성 유지 체크리스트

새 모델/데이터셋 추가 후 반드시 확인:
- [ ] **결과 디렉토리 구조** = `{output_base}/{dataset_dir}/{model}/{epoch_metrics.json, scores.npz, metadata.json, visualization/}` 형태 일치
- [ ] **`epoch_metrics.json` schema** = MAE 동일 (roc_auc, prc_auc, pak_auc_f1, pak_auc_prc_auc, pa_K_f1, f1_t, 등)
- [ ] **`scores.npz` key** = `anomaly_score` (1D float32, n_test_timesteps)
- [ ] **`STANDARD_BASELINES` 동기화** = `experiment_configs.py` + Notion Section 1 + `baselines/` 디렉토리 = 모두 동일 (Simple 5 + Neural 3 + Legacy SOTA 7 + 2026-05-19 batch 7 (AnomalyBERT + CAROTS + CrossAD 실험 제외 — 코드는 유지) = **활성 22개**)
- [ ] **Notion 페이지 업데이트** (Section 1/2/7-9의 모델·데이터셋 표 갱신)
- [ ] **신규 baseline의 input/output 형식 통일 검증** — 사용자 지시 (2026-05-22): "재구현할때 현재 내 baseline 실험 파이프라인에 완벽하게 호환되어야 하고, input과 output 형태도 완벽하게 통일해야 한다." 즉 `fit(train_X: np.ndarray, epoch_callback=None)` + `predict(test_X: np.ndarray) -> np.ndarray (N_test,)` 시그니처 엄수.

## 16. Notion 대조 참조

비교 실험의 결과 표·분석은 Notion 페이지 [Baseline Comparison](https://www.notion.so/Baseline-Comparison-16-Models-6-Datasets-4-Conditions-32087856b2078112b500c81664181ee7)에 유지. **2026-05-22 페이지 rewrite 완료** — 기존 페이지가 단일 거대 테이블로 망가져 있어, 24 모델 × 39 dataset × Q1/Q3 구조로 8-section rewrite 적용. 백업은 `.trash/0522/notion/baseline_comparison_page_pre_rewrite_57k.txt`.

본 GUIDE.md와 Notion 페이지의 일관성 규칙 (2026-05-22 업데이트):

- **Models**: Simple 5 + Neural 3 + Legacy SOTA 7 + 신규 SOTA 7 (2026-05-19 batch, AnomalyBERT + CAROTS + CrossAD는 2026-05-22 실험 제외 — 코드는 baselines/에 유지, 17절 참조) = **활성 22개**. `experiment_configs.py:STANDARD_BASELINES` 가 단일 source of truth.
- **Datasets**: Notion 제목의 "6 Datasets"는 Simulation 제외 카운트. 실제 base 실험은 **Simulation 포함 7개** (Simulation + SWaT + WaDi A1/A2 + SMD + PSM + Exathlon) — 총 39 dataset run/condition.
- **Conditions**: Notion 제목의 "4 Conditions"는 historical 4-Q 표기지만, **현 active 실험은 Q1 + Q3 두 조건만** (minmax × full / normalonly). Q2/Q4 (zscore) 는 폐기. **실행 순서는 Q3 → Q1** (2026-05-22 사용자 지시).
- **신규 10개 페이지 rewrite 필요 항목**:
  - 모델 분류 sub-table (Simple 5 / Neural 3 / Legacy SOTA 7 / New SOTA 10) — 4개 분리
  - 신규 10개 paper metadata (논문명/연도/학회/citation/링크/repo) — 정확성 critical
  - 결과 row placeholder: 실험 완료 전에는 "실험 진행 중 (2026-05-22 시작)" 표시
- **Notion Section 12**: Baseline 모델 출처 검증 ↔ `comparison/MODELS.md`와 동기화 (sections 16–22 활성 7개 + 23-25 참고용 3개 포함)

## 17. 실험 제외 모델 (구현/참고용 코드만 유지)

다음 3개 모델은 **코드 구현 + 논문 참고는 완료했으나, 실험 queue (Q1/Q3)에서 제외**되었다. `comparison/baselines/{anomalybert,carots,crossad}/` 디렉토리는 그대로 보존되며, `baseline_common.py`의 import / `MODEL_PRESETS` / `create_model()` 분기는 모두 유지되어 코드 호출은 가능. 단 `BASELINE_MODELS` 리스트와 `scripts/run_new_sota_22.py:NEW_SOTA_MODELS`에서만 빠져 있어, 표준 실험 파이프라인은 자동으로 학습을 건너뛴다.

| 모델 | 출처 | 제외 사유 | 코드 위치 | 결과 백업 |
|------|------|----------|----------|----------|
| **AnomalyBERT** | ICLR'23 ML4IoT Workshop (Self-supervised Pre-LN Transformer + 4-type degradation) | sim 1 task에 **4시간+** 소요. 39 데이터셋 전체로 환산하면 비현실적. | `comparison/baselines/anomalybert/` | `.trash/0522/comparison/results_anomalybert_q3/` |
| **CAROTS** | ICML'25 (Contrastive AD with positive/negative augmenters, CUTS_Plus 3-stage) | sim 1 task에 **3시간+** 소요. 3-stage pipeline이 long-tail. | `comparison/baselines/carots/` | `.trash/0522/comparison/results_carots_q3/` |
| **CrossAD** | NeurIPS'25 (Cross-scale + query library, Channel Independence) | Channel Independence (`model.py:558-559`)로 effective batch = `bs × c_features`. SWaT(c=45)에서 bs=64 → effective 2880, 12 GB GPU OOM swap으로 12 s/iter → **단일 epoch 37시간 예상**. bs=32 (effective 1440)로 축소해도 SWaT 1 모델당 23시간으로 여전히 infeasible. | `comparison/baselines/crossad/` | `.trash/0522/crossad_removal/results_crossad_q3/` |

**LLM/VLM 정책**: 위 3개와는 별개로, foundation LM (GPT4TS / LLM-TSAD / VLM4TS 등)은 자원/패러다임 차이로 **코드 구현조차 안 함** (메모리 `feedback_no_llm_baselines.md`).

**다시 실험에 포함시키려면**:
1. `comparison/baseline_common.py`의 `BASELINE_MODELS` 리스트에 모델 키 추가
2. `scripts/run_new_sota_22.py`의 `NEW_SOTA_MODELS` 리스트에 추가
3. 메모리/시간 outlier이므로, 데이터셋별로 batch_size를 줄여서 (CrossAD의 경우 SWaT bs=16 또는 8 등) 별도 preset 설정 필요
