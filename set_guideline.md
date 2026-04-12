# Experiment Set Guideline

Claude용 참조 문서. 실험 재현/새 실험 지시 시 사용.

## 실험 진행 원칙 (CRITICAL)

> **⚠️ 스크립트 실행 금지 패턴**: `conda run -n dc_vis python ...` 형태 **절대 금지**.
> conda run은 stdout을 버퍼링하여 실시간 모니터링이 불가능함.
> 반드시 `python scripts/...` 형태로 직접 실행하고, 백그라운드 실행은 Bash tool의 `run_in_background=true` 파라미터 사용.

실험을 진행할 때 반드시 **to-do list를 생성**하고, 각 단계마다 **to-do list를 업데이트**하여 맥락을 잃지 않도록 한다. 컨텍스트가 길어지면 이전 내용이 잘려나갈 수 있으므로, to-do list가 현재 상태의 유일한 진실 소스(single source of truth) 역할을 해야 한다.

**To-do list 필수 항목 예시**:
1. 스크립트/문서 수정사항 반영
2. 기존 실험 결과 삭제
3. Set A 실험 시작 + 모니터링
4. Set A 완료 확인
5. Set B 실험 시작 + 모니터링
6. Set B 완료 확인
7. 최종 결과 구조 검증

각 단계 완료 시 즉시 `completed`로 마킹하고, 다음 단계를 `in_progress`로 전환한다.

## Config Presets

| Param | Set A | Set B | Set C |
|-------|-------|-------|-------|
| `patch_size` | 5 | 20 | 10 |
| `num_patches` | 100 | 25 | 50 |
| `d_model` | 128 | 256 | **dynamic** |
| `dim_feedforward` | 512 | 1024 | **4 × d_model** (auto) |
| `cnn_kernel_size` | 3 | 5 | - (linear) |
| `patchify_mode` | patch_cnn | patch_cnn | **linear** |

공통: `seq_length=500, enc2, td4, sd1, epochs=50, lr=1e-3, batch_size=512, sliding_window_stride=21, sliding_window_test_stride=21, mask_after_encoder=True, anomaly_score_mode=adaptive, use_amp=True, use_discriminator=False`

### Set C: Dynamic d_model 규칙

`d_model = 'dynamic'`이면 데이터 로딩 후 `num_features`를 확인하여 자동 결정:
- `raw = patch_size × num_features = 10 × f`
- `d_model` = `[128, 192, 256, 384, 512]` 중 `raw` 이상인 최소값 (최대 512)
- `dim_feedforward` = `4 × d_model` (overrides에 없으면 자동 계산)
- `patchify_mode = 'linear'` (CNN 없이 Linear embedding)

함수: `mae_anomaly.utils.experiment.resolve_dynamic_d_model(num_features, patch_size)`

정확한 값: `scripts/run_base_experiments.py` 내 `CONFIG_PRESETS` dict 참조.

## 데이터셋 (9 + 56개)

`python scripts/run_base_experiments.py --set A --list` 로 확인 가능.

| Key | Loader | Normal50 | Subdir |
|-----|--------|----------|--------|
| simulation | simulation | No | simulation/simulation |
| simulation_normal50 | simulation | Yes | simulation/simulation_normal50 |
| simulation_complex | simulation_complex | No | simulation/simulation_complex |
| simulation_complex_normal50 | simulation_complex | Yes | simulation/simulation_complex_normal50 |
| SWaT_A1A2 | swat_A1A2 | No | SWaT/A1A2_full + SWaT/A1A2_excl22 |
| SWaT_A1A2_normal50 | swat_A1A2 | Yes | SWaT/A1A2_normal50_full + SWaT/A1A2_normal50_excl22 |
| SWaT_A1A2_swap | swat_A1A2_swap | No | SWaT/A1A2_swap |
| WaDi_A1 | WaDi_14days_A1 | No | WaDi/A1 |
| WaDi_A2 | WaDi_14days_A2 | No | WaDi/A2 |

모든 데이터셋 train stride = 21, test stride = 21.
Normal50 = 학습 데이터의 anomaly region 중 50%를 normal로 재라벨링 (seed=123).
Loader 레지스트리: `mae_anomaly/datasets/loaders.py` → `DATASET_LOADERS`

**데이터 파일 (raw CSV만 사용)**:
- SWaT: `dataset/SWaT/SWaT.A1 & A2_Dec 2015/SWaT_A1_normal_raw.csv`, `SWaT_A2_attack_raw.csv`
- WaDi A1: `dataset/WaDi/WADI.A1_9 Oct 2017/WADI_A1_14days_raw.csv` + `WADI_A1_attack_raw.csv`
- WaDi A2: `dataset/WaDi/WADI.A2_19 Nov 2019/WADI_A2_14days_raw.csv` + `WADI_A2_attack_raw.csv`
- Simulation: 코드 내 생성 (`load_simulation()`)

### SMD 데이터셋 (28머신)

SWaT/WaDi와 동일한 50/50 split 방식.

| Key 패턴 | Loader | Subdir |
|-----------|--------|--------|
| `smd_{machine_id}` | `smd_simple_{machine_id}` | `SMD/{machine}` |

- Train = 원본 train 파일 (전부 정상) + test 파일 앞 50%
- Test = test 파일 뒤 50%
- 집계: `aggregate_smd_results(experiment_dir)` → `SMD/results/results.csv`

**SMD 결과 디렉토리 구조**:
```
{experiment_dir}/
├── simulation/simulation/       # 기존 9개 데이터셋
├── SWaT/A1A2_full/
├── SWaT/A1A2_excl22/
├── WaDi/A1/
├── WaDi/A2/
└── SMD/                        # SMD 28머신
    ├── machine-1-1/            # epoch_metrics.json 등 직접 저장
    ├── machine-1-2/
    ├── ...
    └── results/results.csv     # 28머신 평균
```

## 실행

```bash
conda activate dc_vis

# 전체 실행 (기존 9개 데이터셋)
python scripts/run_base_experiments.py --set A
python scripts/run_base_experiments.py --set B
python scripts/run_base_experiments.py --set C

# 개별/재개
python scripts/run_base_experiments.py --set A --dataset SWaT_A1A2
python scripts/run_base_experiments.py --set A --start-from 5

# SMD 전체 실행 (28개, smd_all 단축키)
python scripts/run_base_experiments.py --set C --dataset smd_all --output-base results/experiments/44_...

# 출력 디렉토리 지정
python scripts/run_base_experiments.py --set A --output-base results/experiments/my_test

# 복수 데이터셋 지정
python scripts/run_base_experiments.py --set A --dataset simulation SWaT_A1A2 WaDi_A1

# config override (key=value 형식)
python scripts/run_base_experiments.py --set A --config-override force_mask_anomaly=False num_epochs=50

# Discriminator 활성화
python scripts/run_base_experiments.py --set A --config-override use_discriminator=True disc_warmup_epochs=10

# 기존 실험에 SMD 추가 (임시 스크립트 — config를 실험에서 자동 복원)
python scripts/run_smd_mae.py --experiment-dir results/experiments/102_... results/experiments/114_...
python scripts/run_smd_mae.py --experiment-dir results/experiments/102_... --skip-existing  # 이미 완료된 것 건너뛰기
python scripts/run_smd_mae.py --experiment-dir results/experiments/102_... --aggregate-only  # 집계만
python scripts/run_smd_mae.py --experiment-dir results/experiments/102_... --machines machine-1-1 machine-1-2  # 특정 머신만
```

**실행 규칙**: 항상 foreground에서 실행. `conda run`, `nohup`, `&` 사용 금지.
Bash tool의 `run_in_background` 파라미터로 백그라운드 실행하고, `TaskOutput`으로 출력 확인.

## 모니터링 (CRITICAL)

실험 실행 중 **주기적으로** 아래 항목을 모니터링해야 한다. OOM(Out of Memory)이 발생하면 에러 메시지 없이 프로세스가 무응답 상태로 멈추므로, 능동적인 모니터링이 필수.

### 기준 메트릭 (5+2+1)

**성능 메트릭 (5개)**:

| # | 항목 | 설명 |
|---|------|------|
| 1 | **PRC** | Adaptive anomaly score의 PRC-AUC |
| 2 | **PAK_AUC_F1** | Adaptive의 PA%K F1 AUC (K=0→100 sweep) — **best epoch 선정 기준** |
| 3 | **PAK_AUC_PRC** | Adaptive의 PA%K PRC AUC (K=0→100 sweep) |
| 4 | **F1_T** | Adaptive의 Time-series F1 (QuoVadisTAD) |
| 5 | **d_SNR** | Discrepancy loss의 SNR |

**학습 Loss (2+1개)**:

| 항목 | 설명 |
|------|------|
| **t_loss** | Teacher reconstruction loss (`train_rec_loss`) |
| **s_loss** | Student discrepancy loss (`train_disc_loss`) |
| **d_loss** | Discriminator loss — `use_discriminator=True`일 때만 출력 |

**SWaT 특수 처리**: SWaT 데이터셋은 region 22가 테스트 데이터의 ~16%를 차지하여 성능을 왜곡함. 따라서 SWaT의 모니터링 메트릭은 `metrics_excl_region22` (experiment_metadata.json)에서 추출한다.

출력 예시:
```
# discriminator OFF
[base_simulation] [1/4] COMPLETE: PRC=0.9550 PAK_AUC_F1=0.8225 PAK_AUC_PRC=0.8488 F1_T=0.8991 | d_SNR=1.23 t_loss=0.0312 s_loss=0.0045 | best_ep=30
# discriminator ON
[base_simulation] [1/4] COMPLETE: PRC=0.9550 PAK_AUC_F1=0.8225 PAK_AUC_PRC=0.8488 F1_T=0.8991 | d_SNR=1.23 t_loss=0.0312 s_loss=0.0045 d_loss=0.6543 | best_ep=30
```

- **epoch callback**: `PRC`, `PAK_F1`, `PAK_PRC`, `F1_T`, `d_SNR`, `t_loss`, `s_loss` (+`d_loss`) 출력
- **background COMPLETE**: 동일 메트릭 + 소요시간 출력

### 모니터링 명령어

```bash
# 1. 학습 진행 — epoch eval 결과 확인 (PRC, PAK_F1, PAK_PRC, F1_T, d_SNR)
grep -o '\[Epoch.*d_SNR=[0-9.-]*' /tmp/claude-1000/.../tasks/{task_id}.output

# 2. 데이터셋 진행 — 완료/에러 (5개 메트릭 출력됨)
grep -E 'Training complete|Spawning|Base Experiment:|Completed:|COMPLETE|ERROR' /tmp/claude-1000/.../tasks/{task_id}.output

# 3. GPU 상태 (메모리 + 사용률)
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# 4. CPU 메모리
free -h | grep Mem

# 5. 전체 python 프로세스 확인 (각 프로세스가 무엇을 하는지 파악)
ps aux | grep python | grep -v grep | grep -v vscode
```

### 프로세스별 역할 식별

`ps aux` 출력에서 프로세스를 식별하는 방법:

| cmdline 패턴 | 역할 | 상태 확인 |
|-------------|------|----------|
| `python scripts/run_base_experiments.py --set A` | 메인 학습 프로세스 (GPU) | CPU% 높으면 학습 중, 0%면 멈춤 의심 |
| `python -c from multiprocessing.spawn` | 백그라운드 eval+viz (CPU) | 여러 개 동시 실행 가능 (최대 10개) |

**확인 포인트**:
- 메인 프로세스 1개 + 백그라운드 0~10개가 정상
- 백그라운드 프로세스의 RAM 사용량 합계 주시 (각 1~3GB 소모 가능)
- 메인 프로세스 CPU%가 0%이면 OOM 또는 hang 의심 → 출력 로그 확인
- 백그라운드 완료 시 `[base_XXX] [N/15] COMPLETE` 메시지 출력됨

### 모니터링 주기

| 시점 | 확인 사항 |
|------|----------|
| 실험 시작 직후 (30초) | 첫 데이터셋 로딩 성공, GPU 메모리 할당 정상 |
| 첫 Epoch 5 출력 후 | PRC > 0.3 확인 (random baseline 이상), GPU 메모리 안정적 |
| 각 데이터셋 전환 시 | "Spawning background eval+viz" 메시지 확인, `[N/15]` 진행 확인 |
| 5번째 데이터셋 이후 | 백그라운드 프로세스 누적 수 확인 (최대 10개), `free -h` 로 RAM 확인 |
| 대형 데이터셋 (SWaT, WaDi) 진입 시 | GPU 메모리 충분한지, RAM 여유 있는지 확인 |

### 모니터링 출력 형식 (TABLE)

모니터링 시 완료된 데이터셋 결과를 **테이블 형식**으로 출력한다:

```
| Dataset    | PRC    | PAK_F1 | PAK_PRC | F1_T   | d_SNR | t_loss | s_loss | d_loss |
|------------|--------|--------|---------|--------|-------|--------|--------|--------|
| sim        | 0.9550 | 0.8225 |  0.8488 | 0.8991 |  1.23 | 0.0312 | 0.0045 |   —    |
| SWaT(ex22) | 0.3982 | 0.4550 |  0.3921 | 0.4690 |  0.85 | 0.0198 | 0.0067 |   —    |
| WaDi_A1    |  ...   |  ...   |   ...   |  ...   |  ...  |  ...   |  ...   |   —    |
| WaDi_A2    |  ...   |  ...   |   ...   |  ...   |  ...  |  ...   |  ...   |   —    |
```

- `d_loss` 컬럼: `use_discriminator=True`일 때만 값 표시, 아니면 `—`
- `SWaT(ex22)`: `metrics_excl_region22`에서 추출한 메트릭 사용
- **PAK_F1** (= PAK_AUC_F1)이 best epoch 선정 기준 → 가장 중요한 메트릭
- 테이블과 함께 **todo list도 같이 출력**하여 전체 진행 상황 파악

**데이터 추출 방법**:
```bash
# COMPLETE 메시지에서 메트릭 추출
grep 'COMPLETE' /tmp/claude-1000/.../tasks/{task_id}.output

# experiment_metadata.json에서 추출 (더 정확, SWaT은 excl_region22 사용)
python -c "
import json,sys; m=json.load(open(sys.argv[1]))
ad=m['metrics']; h=m.get('history',{})
print(f'PRC={ad[\"prc_auc\"]:.4f} PAK_F1={ad.get(\"pak_auc_f1\",0):.4f} PAK_PRC={ad.get(\"pak_auc_prc_auc\",0):.4f} F1_T={ad.get(\"f1_t\",0):.4f}')
" results/experiments/{dir}/{group}/{scenario}/experiment_metadata.json
```

### TODO List 운영 규칙 (CRITICAL)

- **TODO list의 임의 요약, 생략, 수정, 왜곡은 절대 금지**
- Context compact 시에도 **TODO list는 최우선으로 원문 그대로 보존**
- 모니터링 출력 형식 템플릿을 TODO list 최상단에 고정하여 compact 후에도 참조 가능하게 유지
- TODO list 항목 변경은 실제 작업 완료/추가 시에만 허용

### 정상 동작 판단 기준

- `[Epoch N]` 출력이 `EVAL_INTERVAL`(5) 간격으로 나옴
- PRC > 0.3 (random baseline 이상)
- GPU 메모리 안정적 (변동 < 1GB)
- 각 데이터셋 완료 후 "Spawning background eval+viz" 메시지 출력
- `free -h`에서 available 메모리 > 5GB

### 이상 징후 및 대응

| 증상 | 원인 | 대응 |
|------|------|------|
| epoch 출력 없이 멈춤 (2분 이상) | GPU OOM | `nvidia-smi` 확인 → 프로세스 kill → batch_size 줄여서 재시작 |
| 프로세스 갑자기 사라짐 (exit code 없음) | RAM OOM (kernel killed) | `dmesg \| tail -20` 확인 → 백그라운드 프로세스 수 줄여서 재시작 |
| 백그라운드 프로세스 10개 이상 쌓임 | eval+viz가 느림 | `free -h` 확인 → 완료될 때까지 대기 (자동 대기 로직 있음) |
| PRC가 계속 random 수준 (~0.1) | 학습 실패 | loss 로그 확인 → config 점검 |
| "Waiting for background processes" 반복 | 프로세스가 좀비 상태 | `ps aux \| grep spawn` 으로 상태 확인 |

### OOM으로 인한 재실험 절차

백그라운드 프로세스 동시 실행으로 RAM OOM 발생 시:

1. 현재 실험 중단
2. 좀비 프로세스 정리: `pkill -f run_base_experiments; pkill -f spawn`
3. `scripts/run_base_experiments.py`에서 `>= 10`를 `>= 3` 또는 `>= 2`로 줄임
4. 완료된 데이터셋 확인: `ls results/experiments/{dir}/` 로 이미 완료된 항목 파악
5. `--start-from N` 으로 미완료 데이터셋부터 재개
6. 재실험 후 정상 완료되면 프로세스 수를 원복 (`>= 10`)

## 결과 디렉토리 넘버링 규칙 (CRITICAL)

모든 실험 디렉토리에는 **순서 번호 prefix**가 붙는다:

```
results/experiments/{N}_{YYYYMMDD_HHMMSS}_{suffix}/
```

- `N`은 0부터 시작하는 정수 (기존 디렉토리의 max + 1로 자동 할당)
- `run_base_experiments.py`, `run_ablation.py` 모두 `--output-base`를 지정하지 않으면 자동 넘버링
- 유틸리티 함수: `mae_anomaly.utils.experiment.make_numbered_experiment_dir()`
- 수동으로 `--output-base`를 지정할 경우에는 넘버링이 적용되지 않으므로, 직접 prefix를 붙여야 한다

**예시**:
```
results/experiments/
├── 0_20260128_012500_phase1/
├── 1_20260131_225102_phase2/
├── ...
├── 8_20260222_222835_w500p20e2t4d1_d256k5/
├── 9_20260224_120000_phase3/          ← 다음 실험은 자동으로 9번
```

## 결과 디렉토리 구조

```
results/experiments/{N}_{YYYYMMDD_HHMMSS}_{suffix}/
├── summary.json                           # 전체 실험 요약 (각 데이터셋별 train_time, PRC-AUC, F1)
├── {DatasetGroup}/{Scenario}/             # 결과가 직접 저장됨 (timestamp 서브디렉토리 없음)
│   ├── best_model.pt                      # best epoch(PAK_F1 기준) 모델 weights + config + best_epoch (+D state if active)
│   ├── best_config.json                   # 전체 config dict (~61 fields)
│   ├── training_histories.json            # 학습 loss 이력 (epoch별 loss values, D 활성시 +5 keys)
│   ├── epoch_metrics.json                 # epoch별 point-level 평가 (아래 형식 참조)
│   ├── batch_profiling.json               # 첫 N배치 per-component + per-layer 타이밍 (batch 0 제외)
│   ├── batch_profiling.txt               # profiler 형태 요약 테이블 (layer 분해 포함) + per-batch 상세
│   ├── experiment_metadata.json           # 최종 평가 메트릭 + 세부 타이밍 (아래 형식 참조)
│   ├── best_model_detailed.csv            # per-window 상세 결과 (best epoch 모델 기준)
│   ├── anomaly_type_metrics.json          # anomaly type별 성능
│   ├── checkpoints/                       # 모델 체크포인트 (best + latest만 유지)
│   │   ├── best_checkpoint.pt             # best PAK_F1 epoch 모델 (+discriminator_state_dict if D active)
│   │   ├── best_checkpoint_excl22.pt      # (SWaT only) excl22 PAK_F1 best epoch 모델
│   │   └── latest_checkpoint.pt           # 마지막 평가 epoch 모델 (+discriminator_state_dict if D active)
│   ├── epoch_scores/                      # epoch별 point-level 스코어 스냅샷
│   │   └── epoch_{NNN}_scores.npz         # {adaptive_score, teacher_recon_error, discrepancy_error}
│   └── visualization/
│       ├── best_model/                    # 15+ PNGs (BestModelVisualizer.generate_all)
│       └── epoch_metrics/                 # 4-5 PNGs (학습 동태, D 활성시 +1)
│
├── SWaT/A1A2_full/                        # SWaT: full eval (region22 포함)
│   └── (위와 동일한 파일 구조)
├── SWaT/A1A2_excl22/                      # SWaT: excl22 eval (region22 제외)
│   ├── best_model.pt                      # excl22 best epoch 기준 모델
│   ├── experiment_metadata.json           # metrics = excl22 기준 (metrics_full = 전체 기준도 포함)
│   ├── visualization/best_model/          # excl22 기준 시각화
│   ├── checkpoints/ → (symlink to _full)  # 공유
│   └── epoch_scores/ → (symlink to _full) # 공유
```

suffix: 실제 config override에서 동적 생성. 형식: `w{seq}p{patch}e{enc}t{td}d{sd}[_dynamic][_linear][_k{val}]`. (예: `7_20260221_202606_w500p5e2t4d1`, `39_..._w500p10e2t4d1_dynamic_linear_k6`)
DatasetGroup/Scenario 매핑: `scripts/run_base_experiments.py` → `DATASETS` 리스트의 `results_subdir` 필드.

### experiment_metadata.json (핵심 결과 파일)

```json
{
  "experiment_name": "base_simulation",
  "scoring_mode": "adaptive",
  "train_time": 234.5,
  "inference_time": 45.2,
  "metrics": {
    "roc_auc": 0.9734, "prc_auc": 0.9216, "f1_score": 0.8891, "f1_t": 0.8329,
    "pa_20_roc_auc": ..., "pa_20_f1": ...,
    "pa_50_roc_auc": ..., "pa_80_roc_auc": ...,
    "pak_auc_prc_auc": 0.8225, "pak_auc_roc_auc": ..., "pak_auc_f1": ...,
    "pak_auc_f1_t": ..., "pak_auc_precision": ..., "pak_auc_recall": ...,
    "pak_auc_f1_raw": ..., "pak_auc_f1_t_raw": ..., "pak_auc_precision_raw": ..., "pak_auc_recall_raw": ...
  },
  "disc_metrics": { "roc_auc": ..., "f1_score": ... },
  "teacher_recon_metrics": { "roc_auc": ..., "f1_score": ..., "pak_auc_prc_auc": ..., ... },
  "student_recon_metrics": { "roc_auc": ..., "f1_score": ..., "pak_auc_prc_auc": ..., ... },
  "loss_stats": {
    "disc_SNR": ..., "disc_cohens_d_normal_vs_anomaly": ...,
    "disc_ratio": ...
  },
  "config": { ... }
}
```

### epoch_metrics.json (학습 동태 — point-level)

epoch callback에서 all-patches GPU inference + point-level evaluation을 직접 수행하여 생성.

```json
{
  "eval_interval": 5,
  "epochs": [
    {
      "epoch": 5,
      "prc_auc": 0.84, "f1_t": 0.76, "pa_20_f1": 0.72,
      "pak_auc_prc_auc": 0.72, "pak_auc_roc_auc": ..., "pak_auc_f1": ...,
      "teacher_prc_auc": 0.78, "teacher_f1_t": 0.71, "teacher_pa_20_f1": 0.68,
      "teacher_pak_auc_prc_auc": 0.65, "teacher_pak_auc_roc_auc": ...,
      "disc_snr": 1.23,
      "d_loss": 0.65, "d_real_acc": 0.82, "d_fake_acc": 0.78,
      "adv_loss": 0.43, "adaptive_lambda": 1.2,
      "callback_time": 55.2, "_inference_time": 48.1, "_eval_time": 7.1
    },
    ...
  ]
}
```

50 epochs / interval 5 = 10개 epoch entries (5, 10, 15, ..., 50).
각 entry에 소요시간(callback_time, _inference_time, _eval_time)도 포함.
`d_loss`, `d_real_acc`, `d_fake_acc`, `adv_loss`, `adaptive_lambda`는 `use_discriminator=True`일 때만 존재.

### visualization/ 디렉토리

**best_model/** (15+ PNGs): `BestModelVisualizer.generate_all()`이 생성. **Best epoch (PRC-AUC 기준) 모델로 평가한 결과.**
- 주요 차트: score distribution, ROC curve, PRC curve, confusion matrix, feature-wise heatmap, time-series overlay, anomaly type comparison, score timeline, reconstruction samples 등.

**epoch_metrics/** (4 PNGs): `plot_epoch_metrics()`이 생성.
- `epoch_prc_auc.png`: Adaptive + Teacher PRC-AUC 추이
- `epoch_f1_t.png`: Adaptive + Teacher F1_T 추이
- `epoch_pa_k_f1.png`: PA%K F1 추이 (PA0, PA20, PA50, PA100)
- `epoch_dashboard.png`: 4개 패널 종합 대시보드 (PRC, F1_T, PA%K, disc_SNR)
- `epoch_discriminator.png` (D 활성시만): D Loss & Accuracy, Adv Loss, Adaptive λ 3-subplot

## 파이프라인

`run_base_experiments.py` 내 `run_base_experiment()`:

1. 데이터 로딩 (`get_dataset_loader()`)
2. GPU 학습 (`Trainer.train(epoch_callback=, profile_n_batches=10)` — 매 5 epoch **all-patches** 평가 + 체크포인트 저장)
   - **에포크 1 직후 즉시 출력**: 첫 10배치 (batch 0 제외, CUDA warmup 왜곡 회피) per-component + per-layer `cuda.synchronize()` 타이밍 테이블 + 예상 잔여 학습시간 → `batch_profiling.json` + `batch_profiling.txt`
     - Batch level: data→GPU, model_forward, loss_compute, backward, optimizer_step
     - Layer level (model_forward 내부): embed_input(Patchify+CNN), masking, encoder, teacher_decoder, student_decoder
   - per-epoch 타이밍: `train_epoch` (forward/backward), `callback` → `history['epoch_timings']`
   - epoch callback: GPU all-patches inference + point-level evaluation 직접 수행 → `epoch_metrics.json`
4. `epoch_metrics.json` + epoch 시각화 저장
5. GPU 추론 — 단일 통합 pass (`inference_time`):
   - `evaluator._compute_patch_scores_all_patches(collect_detail=True)` — patch scores + timestep-level reconstruction 동시 수집
   - `derive_pred_data()` — evaluator 출력을 viz 포맷으로 변환 (pure numpy, GPU 불필요)
6. GPU 해제 (`free_gpu()`)
7. Background CPU 프로세스 (`_cpu_eval_viz_worker`) 스폰:
   - `Evaluator.set_precomputed_patch_scores()` → `evaluate()` → `cpu_eval_time`
   - `compute_loss_statistics()` (from `run_ablation.py`)
   - `BestModelVisualizer.generate_all()` → 15+ PNGs → `cpu_viz_time`

최대 동시 background 프로세스: eval+viz **10개**. GPU는 즉시 다음 데이터셋 학습 시작.

**중요**:
- Inference는 반드시 **all-patches masking** 방식을 사용. 각 patch를 하나씩 마스킹하여 N번 forward pass 수행 후 patch별 score를 집계. last-patch masking은 사용 금지.
- Point-level score 집계는 **mean aggregation** 방식 사용. 겹치는 윈도우의 patch score를 평균하여 timestep별 최종 score 산출. (`_build_aggregation_map` + `_aggregate_with_map(method='mean')`)
- PA%K AUC: K=0→100을 sweep하며 각 K에서의 PRC-AUC, ROC-AUC, F1 등을 계산한 뒤, trapezoidal rule로 적분하여 단일 스칼라로 요약. (`compute_pa_k_auc()` in `evaluator.py`)
  - `pak_auc_f1` (best_f1_w_pa): **per-K threshold re-optimization** — 각 K에서 PA%K 조정 후 최적 threshold를 다시 찾아 F1 계산 (Kim et al., AAAI 2022 tadpak 방식). Best epoch 선정 기준.
  - `pak_auc_f1_raw` (raw_f1_w_pa): **fixed threshold** — pre-PA F1-optimal threshold를 고정하여 PA%K 조정 후 F1 계산 (legacy, 비교용).
  - PRC-AUC/ROC-AUC는 이미 threshold sweep을 사용하므로 변경 없음.

## Experiment Queue (다중 실험 관리)

`scripts/run_queue.py` — 순수 orchestrator. ML 코드 import 없이 `run_base_experiments.py`를 subprocess로 호출.

### 핵심 원리

- **GPU idle 제거**: `--no-wait` 플래그로 각 실험의 background CPU 대기를 스킵. 실험 N의 GPU 완료 즉시 실험 N+1 시작.
- **CPU 병렬**: background eval+viz 프로세스는 최대 10개까지 동시 실행 (초과 시 자동 대기).
- **느슨한 결합**: `run_base_experiments.py`의 stdout을 best-effort 파싱. 스크립트 수정해도 실행에 영향 없음.

### 사용법

```bash
conda activate dc_vis

# 1. JSON 큐 파일로 실행
python scripts/run_queue.py --queue configs/queue_example.json

# 2. 인라인 실험 정의
python scripts/run_queue.py \
    --exp "set=C name=baseline" \
    --exp "set=C name=with_disc config_override='use_discriminator=True disc_warmup_epochs=10'"

# 3. Dry run (명령어만 확인, 실행 안 함)
python scripts/run_queue.py --queue configs/queue_example.json --dry-run

# 4. 리소스 모니터 (실험 없이 GPU/CPU/RAM 상태만 5초 간격 출력)
python scripts/run_queue.py --monitor

# 5. 백그라운드 worker 수 제한
python scripts/run_queue.py --queue configs/queue.json --max-bg-workers 5
```

### 큐 JSON 형식

```json
{
    "experiments": [
        {
            "name": "exp42_baseline",
            "set": "C",
            "config_override": "num_epochs=50"
        },
        {
            "name": "exp43_disc",
            "set": "C",
            "config_override": "num_epochs=50 use_discriminator=True disc_warmup_epochs=10"
        },
        {
            "name": "exp44_sim_only",
            "set": "C",
            "dataset": ["simulation"],
            "config_override": "num_epochs=10"
        }
    ]
}
```

**지원 필드**: `name` (식별자), `set` (필수, A/B/C), `config_override` (문자열), `dataset` (리스트), `start_from` (정수)

### 모니터링 출력

실행 중 자동으로 표시되는 정보:

```
######################################################################
# QUEUE [2/4] exp43_disc
# Command: python scripts/run_base_experiments.py --set C --no-wait ...
# Background workers: 3/10
# Resources: GPU 87% 9.2/12.0GB | CPU 34% | RAM 18.2/31.3GB
######################################################################
  [exp43_disc] # [1/4] simulation
  [exp43_disc] Epoch 23/50: ...
  ...

──────────────────────────────────────────────────────────────────────
  Queue progress: 2/4 GPU done | 5 background workers | Elapsed: 1:45:23
  Resources: GPU 0% 0.5/12.0GB | CPU 45% | RAM 22.1/31.3GB
  Background workers:
    [exp42_baseline] SWaT_A1A2 (PID 12345, 0:32:15)
    [exp42_baseline] WaDi_A1 (PID 12346, 0:28:40)
    [exp43_disc] simulation (PID 12400, 0:05:12)
──────────────────────────────────────────────────────────────────────
```

### 중단 및 재개

- `Ctrl+C` 1회: 현재 실험 완료 후 큐 중단 (background 프로세스는 독립 실행 계속)
- `Ctrl+C` 2회: 강제 종료
- 재개: 완료된 실험을 JSON에서 제거하고 다시 실행

### 결과 요약

큐 완료 시 `results/experiments/queue_summary.json`에 전체 요약 저장:
- 각 실험의 GPU 소요 시간, 상태, 실행 명령어
- 총 GPU 시간

## 핵심 참조 파일

| 용도 | 파일 |
|------|------|
| 실험 큐 관리 | `scripts/run_queue.py` |
| 실험 실행 | `scripts/run_base_experiments.py` |
| Config preset/defaults | `mae_anomaly/utils/experiment.py` (`make_config`) |
| Config 전체 필드 | `mae_anomaly/config.py` |
| 데이터셋 로더 | `mae_anomaly/datasets/loaders.py` (`DATASET_LOADERS`) |
| 평가 | `mae_anomaly/evaluator.py` |
| 시각화 | `mae_anomaly/visualization/best_model_visualizer.py` (`generate_all`) |
| Loss 통계 | `scripts/ablation/run_ablation.py` (`compute_loss_statistics`) |
| 기존 실험 point-level 평가 (임시) | `scripts/eval_epoch_pointlevel.py` |
| Baseline 비교 | `comparison/GUIDE.md` |
