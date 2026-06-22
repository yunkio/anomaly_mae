# TEP Type-Disjoint Generalization — MAE 실험 가이드

> TEP(Tennessee Eastman Process, Rieth 2017) **type-disjoint generalization** 실험을
> 기존 MAE 실험 코드(`official=True` 경로)로 그대로 돌리기 위한 코드 변경 + 실행 방법.
> 설계 근거: Notion "Table 4 — tab:tep_typegen" (조건 A/B/B0/D × 4 fold + noisy-label sweep; condition C는
> v4에서 제거 — supervised 분류 모델 부재). **조건은 model flag가 아니라 데이터/라벨 체제로 정의**(2026-06-22 수정).
>
> **상태**: 코드 구현 완료 + CPU 스모크 검증 완료(2026-06-22, noisy 포함). 실제 학습은 GPU 필요 →
> 아래 명령은 사용자가 GPU 가용 시 실행. (`run_base_experiments.py:2529`에서 `config.device='cuda'` 고정.)

---

## 1. 개요

| 축 | 내용 |
|---|---|
| 목적 | 한 fault family를 "seen contamination"으로 학습에 노출, 나머지 unseen fault에 대한 일반화 측정 |
| 모델 | MAE self-distilled Teacher–Student + GRL purging (`official=True` = CANON_271 recipe) |
| 정규화 | **minmax `0_1`** (다른 데이터셋과 동일; CANON_271 기본값, override 불필요) |
| Best epoch | `pak_auc_f1` (project 기본; `config.best_epoch_metric`) |
| Weight | **저장 안 함** (`--save-weights` 생략 = 기본 off, `official_keep_checkpoints=False`) |
| 결과/시각화 | 평소대로 전부 생성 (`epoch_metrics.json`, `training_histories.json`, `epoch_*_scores.npz`, `visualization/`) |

### Conditions — **데이터/라벨 체제**로 정의 (모델은 하나의 LASAD, full official config 공통)

조건은 model flag가 아니라 **train 데이터에 라벨을 다느냐 / 안 다느냐 / 아예 빼느냐**로 정의한다.
모델 config(use_grl=True, force_mask_anomaly=True 등)는 **모든 조건에서 동일**(full CANON_271).
라벨이 없으면 라벨-구동 컴포넌트가 **데이터에 의해 자동으로 inert**가 된다 — `use_grl=False` 같은
flag로 모델을 꺾을 필요가 없다(아래 "자동 inert 근거" 참조).

| 코드 | train 데이터 | 라벨 | 별도 학습 | 구현 (full config 공통) |
|---|---|---|---|---|
| **A** LASAD (ours) | contaminated | 달기 (100%) | 4 fold | 기본 (`TEP_typegen_<fold>`) |
| **B** label-blind | contaminated | **안 달기 (0%)** | 4 fold | `+ blind_train_labels=True` |
| **B0** clean ref | **clean (faulty 제외)** | — | 1 (fold 무관) | `TEP_typegen_ffonly` 데이터 |
| **D** recon-only | = A | — | **0 (A에서 파생)** | A run의 `teacher_pak_auc_f1` |
| **noisy** lab80/50/25/10 | contaminated | **일부만** (80/50/25/10% labeled) | 16 (추가실험) | `TEP_typegen_<fold>_{lab80,lab50,lab25,lab10}` 데이터 |
| **LOFO** (추가) | **3 family seen** | A/B처럼 | 8+ (추가실험) | `TEP_typegen_lofo_<heldout>[_cont]` 데이터 |

- **A vs B** = 데이터·모델 완전 동일, **오직 라벨 유무**만 차이 → 깨끗한 matched control (라벨의 순수 효과).
- **B vs B0** = 둘 다 무라벨, B는 오염 섞임 / B0는 오염 뺌 → 오염의 순수 피해(C_dmg).
- **noisy(부분 라벨)** = A(100% labeled) → lab80(80%) → lab50(50%) → lab25(25%) → lab10(10%) → B(0% labeled) 곡선 (태그=labeled %). seen-family
  faulty run 중 일부를 무라벨 오염으로 남김(per-fault 뒤쪽 run; #12 "앞쪽 k labeled" 규칙). **모델·config는 A와 동일** — labeled 부분에만 GRL 등이 작동.

> **자동 inert 근거 (코드 검증)**: 라벨(anomaly) 0이면 ① GRL classifier loss가 **명시적으로 skip**됨
> (`loss.py:310-323` `_pos_count==0 → grl_cls_loss_tensor=None`, encoder gradient 0), ② force_mask_anomaly는
> anomaly patch가 없어 **일반 random masking으로 fallback**(모든 normal-only 학습이 이미 그렇게 동작),
> ③ dynamic margin은 전부 normal로 처리(= 라벨 없을 때의 올바른 동작). 따라서 B(blind)·B0(ffonly)는
> **full config 그대로** 두면 라벨-구동 컴포넌트가 자동으로 꺼진다.

> **D가 무료인 이유** (코드 검증): student decoder는 **detached** encoder latent을 받고
> (`model.py` forward: *"student는 .detach()로 받음"*), discrepancy target도 `teacher_output.detach()`
> (`loss.py:233`), `student_recon_weight=0`. 즉 **student/discrepancy의 gradient가 encoder·teacher에
> 전혀 흐르지 않음** → A 모델의 teacher reconstruction = student 없는 모델의 reconstruction과 동일.
> 게다가 official 평가는 매 epoch **teacher-recon-only score에 대해 full metric set을 이미 산출·저장**
> (`run_base_experiments.py:714,728,839-846`) → D는 A run의 `epoch_metrics.json`에서
> `teacher_pak_auc_f1`(+`teacher_prc_auc`,`teacher_f1_t`,`teacher_pa_20_f1`)을 읽으면 끝. 재학습/재평가 0.

---

## 2. 데이터

위치: `scripts/TEP/data/` (frozen NPZ, `scripts/TEP/build_tep_data.py`로 생성). `run_len=960`,
fault onset = sample 161 (= 0-based index 160; 각 faulty run은 앞 160 정상 / 뒤 800 이상).

| 파일 | 용도 | shape (X) | 비고 |
|---|---|---|---|
| `train_ffonly.npz` | B0 clean ref | (230,400, 52) | 240 FF run, anomaly 0% |
| `train_f_step.npz` | fold F-STEP train | (288,000, 52) | seen faults {1,2,4,5,6,7} |
| `train_f_rand.npz` | fold F-RAND train | (288,000, 52) | seen = random family |
| `train_f_ds.npz`   | fold F-DS train   | (288,000, 52) | seen = drift+sticking {13,14} |
| `train_f_unk.npz`  | fold F-UNK train  | (288,000, 52) | seen faults {16,17,18,19,20} |
| `test_stream.npz`  | 공유 test (모든 fold 동일) | (422,400, 52) | 440 run, anomaly 75.76% (400 faulty + 40 FF), 400 anomaly regions |

NPZ 키: `X (N,52) float32`, `y (N,) int64` (0/1, 보정된 onset 그대로 사용),
`fault_id (N,) int16` (0=FF, 1–20), `run_boundaries (int64)`.

오염 fold는 train에 60개 seen-family faulty run이 섞여 train anomaly ≈ 16.67%.
train_ratio는 fold별로 자동 산출 (ffonly 0.3529, 오염 fold 0.4054 — test 길이는 공유, train 길이만 다름).

---

## 3. 코드 변경 내역 (2026-06-22)

> 백업: `.trash/260622/pre_tep_mae/{loaders.py, config.py, dataset_sliding.py, run_base_experiments.py}`.
> 모든 변경은 **additive · default-off** — 기존 데이터셋/실행 경로에 무영향.

### (1) `mae_anomaly/datasets/loaders.py`
- **`load_tep_typegen(fold)`** 신규 (load_tep 직후). frozen NPZ 2개(train_<fold> + test_stream)를
  표준 `[train|test]` 6-tuple로 조립 → 기존 SlidingWindowDataset 기계가 그대로 ingest.
  - run_boundaries = train-internal + train|test seam(train_len) + test-internal(+offset).
  - anomaly_regions = test의 연속 `y==1` 구간(각 800), `anomaly_type`=fault_id (per-fault partition metric용).
  - feature_names = `manifest.json`의 `feature_cols` (52) 또는 generic.
- **DATASET_LOADERS 등록**: `tep_typegen_{ffonly,fstep,frand,fds,funk}`.
- **noisy-label (부분 라벨, 2026-06-22; 태그 재정의 2026-06-23)**: `load_tep_typegen(fold, unlabeled_frac)` —
  seen-family faulty train run 중 per-fault 뒤쪽 `round(n*frac)` run의 라벨을 0으로(무라벨 오염으로 잔류). **데이터 레벨**
  연산(point_labels만 변경, signals/test 무손상). 등록 키: `tep_typegen_{fstep,frand,fds,funk}_{lab80,lab50,lab25,lab10}`
  — **태그 = labeled %** (`lab80`=80% labeled=`unlabeled_frac` 0.20, `lab50`=0.50, `lab25`=0.75, `lab10`=0.90).
  per-fault round 기준 실측 무라벨 run: lab80 ≈ 12/60, lab50 = 30/60, lab25 ≈ 45/60, lab10 ≈ 54/60.
- **LOFO (leave-one-family-out, 2026-06-22)**: `load_tep_typegen_lofo(held_out, contaminate_heldout)` —
  기존 frozen NPZ에서 **재시뮬 없이 조립**: FF 240 + seen 3 family faulty(각 60, labeled) [+ held-out 60 unlabeled].
  등록 키: `tep_typegen_lofo_{step,rand,ds,unk}`(held-out 제외) + `_cont`(held-out 무라벨 오염). train 420 runs
  (cont 480), labeled-anom ≈35.7%(cont 31.25%). 메인 16.67%보다 높음(3 family 동시 오염). seen/heldout fault set은 data_info에 기록.

### (2) `scripts/run_base_experiments.py`
- **`TEP_TYPEGEN_DATASETS`** 별도 리스트 신규 (SMAP/MSL simple 정의 뒤). 5 base 키(`TEP_typegen_<fold>`)
  + 16 noisy 키(`TEP_typegen_<fold>_{lab80,lab50,lab25,lab10}`). loader=동명, train_stride=1, results_subdir=`TEP/typegen_<...>`.
  - **`DATASETS`에 넣지 않음** → 기본 5-base sweep 오염 없음. `--dataset TEP_typegen_<...>`로만 접근.
- `all_datasets` 조립부에 `+ TEP_TYPEGEN_DATASETS` 추가.
- **train_dataset 생성부에 `blind_train_labels=getattr(config,'blind_train_labels',False)` 전달**.

### (3) `mae_anomaly/config.py`
- **`blind_train_labels: bool = False`** 필드 신규 (force_mask_anomaly 블록 뒤).

### (4) `mae_anomaly/dataset_sliding.py`
- `SlidingWindowDataset.__init__`에 **`blind_train_labels: bool = False`** 파라미터 추가.
- **train split 분기에서만** `if blind_train_labels: self.point_labels = np.zeros_like(...)`.
  단일 root-cause 차단 — 라벨을 소비하는 4지점(force_mask `model.py`, GRL/anomaly_loss `loss.py`,
  **dynamic-margin normal 선택 `loss.py:105`**)이 모두 라벨-free train을 봄. **test split·anomaly_regions(eval)은 무손상.**

### pos_weight 가드 — 불필요(이미 존재)
B0(ffonly)·B(blind)는 labeled anomaly=0 → patch ratio=0이지만 `run_base_experiments.py:2722`에
`_patch_ratio = max(_patch_ratio, 0.001)` 가드가 이미 있어 div-by-zero 없음. 게다가 GRL은 `_pos_count==0`이면
`loss.py:311`에서 통째로 skip되므로 pos_weight 자체가 안 쓰임. 별도 코드 변경 없음.

### CPU 스모크 검증 결과 (2026-06-22, GPU 미사용)
- loader 6-tuple/shape, **onset 정확** (첫 test 이상 = test-idx 160, 전 anomaly region 길이 800).
- fold-disjoint seen sets: F-STEP={1,2,4,5,6,7}, F-UNK={16,17,18,19,20}.
- `blind_train_labels`: train 라벨 합 0 (정상=2720), **test 라벨 보존**(blind 전달해도 18140).
- ffonly = clean(train anomaly 0%) + test 400 regions(eval용).
- official(minmax/d512/52feat) 모델 29.3M params, 실제 TEP batch forward → teacher/student recon finite.

---

## 4. 실행 방법

> 전제: `conda activate dc_vis`. **GPU 필요** (학습 경로는 cuda 고정). minmax는 CANON_271 기본이라 override 불필요.
> warmup은 `teacher_only_warmup_epochs`를 생략하면 official 분기가 자동으로 `num_epochs//2`로 설정
> (`run_base_experiments.py:2512`) → ep30→15, ep10→5.
> **`eval_interval=2` (2026-06-22)**: 모든 TEP run은 eval을 **2 epoch 간격**으로(+ 마지막 epoch 항상 평가). official 기본(매 epoch)은
> eval(~110s)≫학습(~36s)이라 **eval-bound로 wall-clock이 평가에 지배**됨 → 2 epoch 간격으로 평가 횟수 절반. `config.eval_interval` 필드(Config),
> `run_base_experiments.py`에서 override>0이면 우선. best-epoch는 평가된 epoch 중 선택(해상도 약간 거침).

### Phase 1 — 파일럿: B0 clean-ref, **epoch 30** (loss 추이 + baseline 파악)
```bash
conda activate dc_vis
TS=$(date +%Y%m%d_%H%M%S)
python scripts/run_base_experiments.py \
  --set A \
  --dataset TEP_typegen_ffonly \
  --output-base results/experiments/271_${TS}_30_42 \
  --config-override official=True num_epochs=30 eval_interval=2 official_keep_checkpoints=False random_seed=42
```
- `official=True` → CANON_271 base + official bundle(stride=1, epoch_offset off, eval 매 epoch).
- **flag 토글 없음** — ffonly엔 이상 라벨이 없어 GRL/force_mask가 자동 inert(데이터-체제, full config 그대로).
- weight 미저장(`--save-weights` 생략 + `official_keep_checkpoints=False`).
- 결과: `results/experiments/271_${TS}_30_42/TEP/typegen_ffonly/` →
  `training_histories.json`(loss 추이), `epoch_metrics.json`(`pak_auc_f1`·`teacher_pak_auc_f1`), `visualization/`.

### Phase 2 — 본 매트릭스: **epoch 10**, 9 runs (A×4 + B×4 + B0×1), D는 회수
> A와 B는 같은 dataset subdir(`TEP/typegen_<fold>`)를 쓰므로 **condition별로 `--output-base`를 분리**해야
> 덮어쓰기가 안 남(가장 안전). 디렉토리 코어는 `271_{ts}_{epoch}_{seed}`, 뒤에 condition 태그.

**A — LASAD full × 4 fold**
```bash
TS=$(date +%Y%m%d_%H%M%S)
python scripts/run_base_experiments.py --set A \
  --dataset TEP_typegen_fstep TEP_typegen_frand TEP_typegen_fds TEP_typegen_funk \
  --output-base results/experiments/271_${TS}_10_42_A \
  --config-override official=True num_epochs=10 eval_interval=2 official_keep_checkpoints=False random_seed=42
```

**B — label-blind control × 4 fold** (`blind_train_labels=True`만 추가, 다른 flag 없음)
```bash
python scripts/run_base_experiments.py --set A \
  --dataset TEP_typegen_fstep TEP_typegen_frand TEP_typegen_fds TEP_typegen_funk \
  --output-base results/experiments/271_${TS}_10_42_B \
  --config-override official=True num_epochs=10 eval_interval=2 blind_train_labels=True official_keep_checkpoints=False random_seed=42
```

**B0 — clean ref × 1 (matrix용, epoch 10)** (ffonly 데이터, flag 없음)
```bash
python scripts/run_base_experiments.py --set A \
  --dataset TEP_typegen_ffonly \
  --output-base results/experiments/271_${TS}_10_42_B0 \
  --config-override official=True num_epochs=10 eval_interval=2 official_keep_checkpoints=False random_seed=42
```

**D — recon-only (실행 없음)**
A의 각 fold `epoch_metrics.json`에서 best epoch의 **`teacher_pak_auc_f1`** 을 읽음
(보조: `teacher_prc_auc`, `teacher_f1_t`, `teacher_pa_20_f1`; raw point score는 `epoch_*_scores.npz`의
`teacher_recon_error`). 별도 학습/평가 불필요.

---

## ⊕ 추가 실험 (본 계획과 **별도** — 옵션)

> 기본 계획(Phase 1 pilot + Phase 2 A/B/B0/D)과 분리된 추가 실험. 동일 코드·config, dataset 키만 다름.
> 추가 학습 = noisy 16 + LOFO 8 = **24 runs (옵션)**.

### A. Noisy-label (부분 라벨) sweep, epoch 10
A(100% labeled)→lab80→lab50→lab25→lab10→B(0% labeled) 곡선 (태그=labeled %). **A와 동일 config**, dataset 키만 noisy(데이터에 부분 라벨).
```bash
# labeled % sweep: lab80(80%)→lab50(50%)→lab25(25%)→lab10(10%), 각 × 4 fold
for LAB in lab80 lab50 lab25 lab10; do
  python scripts/run_base_experiments.py --set A \
    --dataset TEP_typegen_fstep_${LAB} TEP_typegen_frand_${LAB} TEP_typegen_fds_${LAB} TEP_typegen_funk_${LAB} \
    --output-base results/experiments/271_${TS}_10_42_${LAB} \
    --config-override official=True num_epochs=10 eval_interval=2 official_keep_checkpoints=False random_seed=42
done
```

### B. LOFO (leave-one-family-out), epoch 10
3 family seen / 1 held-out unseen (메인의 반대 비율 — "거의 다 라벨 → 새 종류 탐지"). **A와 동일 config**,
dataset 키만 LOFO. 조건은 메인과 동일하게 LOFO-A(=as-is) / LOFO-B(=`blind_train_labels=True`). held-out 변형:
제외(기본) vs `_cont`(held-out 무라벨 오염).
```bash
# LOFO-A (3 seen family LABELED) × 4 held-out, held-out 제외
python scripts/run_base_experiments.py --set A \
  --dataset TEP_typegen_lofo_step TEP_typegen_lofo_rand TEP_typegen_lofo_ds TEP_typegen_lofo_unk \
  --output-base results/experiments/271_${TS}_10_42_lofoA \
  --config-override official=True num_epochs=10 eval_interval=2 official_keep_checkpoints=False random_seed=42
# LOFO-B (3 seen family UNLABELED) — 동일 dataset + blind_train_labels=True
python scripts/run_base_experiments.py --set A \
  --dataset TEP_typegen_lofo_step TEP_typegen_lofo_rand TEP_typegen_lofo_ds TEP_typegen_lofo_unk \
  --output-base results/experiments/271_${TS}_10_42_lofoB \
  --config-override official=True num_epochs=10 eval_interval=2 blind_train_labels=True official_keep_checkpoints=False random_seed=42
# (변형) held-out 무라벨 오염: --dataset 을 TEP_typegen_lofo_<ho>_cont 로 교체
```
판정: held-out family의 per-mode pak_auc_f1(unseen 타깃) — **Δ_unseen(LOFO) = LOFO-A(held-out) − LOFO-B(held-out)**.
seen 3 family = S, held-out = U(이 protocol의 관심 U).

> seed 여러 개로 반복하려면 `random_seed`와 디렉토리의 seed 부분을 바꿔 재실행 (예: `..._10_43_A`).
> **기본 계획 = 9 runs** (Phase 2: A4+B4+B0×1), D=0(파생), simple baseline=0(#12 재활용).
> **추가 실험(별도) = 24 runs** (noisy lab80/50/25/10 ×4 = 16 + LOFO lofoA×4+lofoB×4 = 8). held-out `_cont` 변형은 선택(+8).

### LOFO 설계 결정 (조정 가능)
- **오염 budget**: seen 3 family를 각각 **full 60 runs**(180 total, labeled-anom 35.7%)로 둠 — 각 family가 메인과
  동일하게 완전 표현됨. (대안: 60 total로 budget-match하려면 family당 20 subsample — fault 누락 발생.)
- **held-out 기본 = 제외**(순수 unseen). `_cont` = held-out를 무라벨 오염으로 둠(현실적 시나리오: 일부 type만 라벨).
- 위 둘 다 조정 원하면 알려주세요.

---

## 5. 평가/해석 (#12 설계 정합)
- **VUS 제외 (2026-06-23 사용자 지시, 영구)**: TEP 최종 분석에서 **VUS 계열(`vus_pr`/`vus_roc`)은 무시**한다. headline은 `pak_auc_f1` 그대로이고 PA%K/PRC/F1/Affiliation/R-F1는 모두 유지된다. 구현 메커니즘은 **`MAE_SKIP_VUS` 환경변수**다:
  - **자동 (TEP)**: `run_base_experiments.py`가 `key.startswith('TEP_typegen')`이면 `os.environ['MAE_SKIP_VUS']='1'`을 **자동 설정**(run_base:2763)하고, 이 env는 spawn되는 bg eval/viz/pool worker에 상속된다. 효과 세 곳 — ① `evaluator.compute_full_metric_set`가 `skip_vus=(lite or MAE_SKIP_VUS=='1')`로 **VUS만** 건너뜀, ② bg-worker 최종 eval이 중복 재계산 대신 `epoch_metrics[best]` read 경로 사용, ③ 사후 `_run_vus_sweep_on_saved_npz`도 skip → `vus_*`는 0/공백으로 남는다. per-epoch eval은 원래도 `lite=True`라 VUS를 계산하지 않는다. non-TEP은 env 미설정 → VUS를 기존대로 계산(default byte-identical).
  - **VUS 끄는 법 (임의 런)**: 실행 전 환경변수 `MAE_SKIP_VUS=1`을 주면 어떤 데이터셋이든 위와 동일하게 VUS가 꺼진다 — 예: `MAE_SKIP_VUS=1 python scripts/run_base_experiments.py --dataset <KEY>`. (TEP_typegen* 키는 run_base가 자동으로 set하므로 따로 지정할 필요 없다.)
  - **VUS 다시 켜는 법**: `MAE_SKIP_VUS`를 unset(또는 `0`)하면 된다. 단 TEP_typegen* 키는 run_base가 강제로 `1`을 set하므로, TEP에서 VUS를 켜려면 run_base:2763 분기를 수정해야 한다.
  - 향후 `scripts/TEP/build_table_d1.py`(미작성)·Notion "Table D.1"·headline 어디에도 `vus_pr`/`vus_roc`를 싣지 말 것.
- **셀 metric = macro per-mode `pak_auc_f1`** (= 표 "F1_PA%K"; per-mode 평가셋 = 그 mode 20 faulty run + 공유 40 FF run → positive rate ≈29.4%). **raw S/U를 그대로 읽지 말 것** — label-free엔 type-gen 격차 자체가 없고, 격차 대부분은 평가구성+오염 artifact(#12 insight 1).
- **판정량**: ① **Ĝ = G_model − G_control(B)** (G = seen − unseen; B 대비로만 해석), ② **Δ_unseen = A_U − B_U > 0** (H1 vs H2 결정), ③ **C_dmg = clean_seen − contaminated_seen** (오염 피해; A가 얼마나 회복하나).
- **D = `teacher_pak_auc_f1`** (teacher-recon score, A에서 파생).
- per-fault 분해: anomaly_regions의 `anomaly_type`(fault_id) + `tep_common.seen/unseen_faults`로 mode별/S·U 집계. (IDV 3·9·15 = excluded-hard, headline 제외.)
- 가설: **H1**(general purging: Δ_unseen>0, Ĝ≈0) vs **H2**(implicit classifier: unseen 붕괴). A vs B vs D vs B0 + noisy sweep으로 판정.

## 6. 주의/일관성
- 정규화는 **minmax `0_1`** (CANON_271 기본). `minmax_clamp_±4`는 `0_1` 경로에서 **dead** (neg1_1 전용), `0_1`은 [0,1] tight-clip만 활성 — 다른 데이터셋과 동일.
- run_boundaries가 train|test seam과 모든 run 경계를 보호 → 윈도가 run을 넘지 않음. (train_ratio float→`int(total*ratio)` 1-step 오차는 전 데이터셋 공통, seam 경계가 무결성 보장.)
- weight 미저장이므로 사후 재평가 불가 — 필요한 모든 diagnostic(teacher/disc score 포함)은 학습 중 이미 저장됨.
