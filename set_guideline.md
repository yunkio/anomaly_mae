# Experiment Set Guideline

Claude용 참조 문서. 실험 재현/새 실험 지시 시 사용.

## 실험 진행 원칙 (CRITICAL)

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

공통: `seq_length=500, enc2, td4, sd1, epochs=50, lr=1e-3, batch_size=512, sliding_window_stride=21, sliding_window_test_stride=21, mask_after_encoder=True, anomaly_score_mode=adaptive, use_amp=True`

### Set C: Dynamic d_model 규칙

`d_model = 'dynamic'`이면 데이터 로딩 후 `num_features`를 확인하여 자동 결정:
- `raw = patch_size × num_features = 10 × f`
- `d_model` = `[128, 192, 256, 384, 512]` 중 `raw` 이상인 최소값 (최대 512)
- `dim_feedforward` = `4 × d_model` (overrides에 없으면 자동 계산)
- `patchify_mode = 'linear'` (CNN 없이 Linear embedding)

함수: `mae_anomaly.utils.experiment.resolve_dynamic_d_model(num_features, patch_size)`

정확한 값: `scripts/run_base_experiments.py` 내 `CONFIG_PRESETS` dict 참조.

## 데이터셋 (15개)

`python scripts/run_base_experiments.py --set A --list` 로 확인 가능.

| Key | Loader | Normal50 | Subdir |
|-----|--------|----------|--------|
| simulation | simulation | No | simulation/simulation |
| simulation_normal50 | simulation | Yes | simulation/simulation_normal50 |
| simulation_complex | simulation_complex | No | simulation/simulation_complex |
| simulation_complex_normal50 | simulation_complex | Yes | simulation/simulation_complex_normal50 |
| SWaT_A1A2 | swat_A1A2 | No | SWaT/A1A2 |
| SWaT_A1A2_normal50 | swat_A1A2 | Yes | SWaT/A1A2_normal50 |
| SWaT_A1A2_swap | swat_A1A2_swap | No | SWaT/A1A2_swap |
| WaDi_A1 | wadi_A1 | No | WaDi/A1 |
| WaDi_A1_swap | wadi_A1_swap | No | WaDi/A1_swap |
| WaDi_A2 | wadi_A2 | No | WaDi/A2 |
| WaDi_A2_swap | wadi_A2_swap | No | WaDi/A2_swap |
| WaDi_A1_14days | wadi_14days_A1 | No | WaDi/A1_14days |
| WaDi_A1_14days_normal50 | wadi_14days_A1 | Yes | WaDi/A1_14days_normal50 |
| WaDi_A2_14days | wadi_14days_A2 | No | WaDi/A2_14days |
| WaDi_A2_14days_normal50 | wadi_14days_A2 | Yes | WaDi/A2_14days_normal50 |

모든 데이터셋 train stride = 21, test stride = 21.
Normal50 = 학습 데이터의 anomaly region 중 50%를 normal로 재라벨링 (seed=123).
Loader 레지스트리: `mae_anomaly/datasets/loaders.py` → `DATASET_LOADERS`

### SMD K=6 Block Split 데이터셋 (28머신 × 2 parity = 56개)

상세: `docs/SMD_BLOCK_SPLIT.md`

| Key 패턴 | Loader | Parity | 설명 |
|-----------|--------|--------|------|
| `smd_k6_{machine_id}` | `load_smd_block_split(machine, parity=0)` | 0 (even→train) | 짝수블록 학습 |
| `smd_k6_{machine_id}_swap` | `load_smd_block_split(machine, parity=1)` | 1 (odd→train) | 홀수블록 학습 |

- test 파일만 6블록으로 나누어 교차 배정 (이상 분배 ~50/50)
- 경계는 이상 영역에서 ±500 이상 떨어진 정상 구간에 배치
- 2회 실험(parity 0, 1) 후 메트릭 평균 = 최종 결과
- Test 이상=0 케이스: machine-1-1(parity=0만), machine-2-8(parity=1만)

## 실행

```bash
conda activate dc_vis

# 전체 실행
python scripts/run_base_experiments.py --set A
python scripts/run_base_experiments.py --set B
python scripts/run_base_experiments.py --set C

# 개별/재개
python scripts/run_base_experiments.py --set A --dataset SWaT_A1A2
python scripts/run_base_experiments.py --set A --start-from 5

# 출력 디렉토리 지정
python scripts/run_base_experiments.py --set A --output-base results/experiments/my_test

# 복수 데이터셋 지정
python scripts/run_base_experiments.py --set A --dataset simulation SWaT_A1A2 WaDi_A1

# config override (key=value 형식)
python scripts/run_base_experiments.py --set A --config-override force_mask_anomaly=False num_epochs=50
```

**실행 규칙**: 항상 foreground에서 실행. `conda run`, `nohup`, `&` 사용 금지.
Bash tool의 `run_in_background` 파라미터로 백그라운드 실행하고, `TaskOutput`으로 출력 확인.

## 모니터링 (CRITICAL)

실험 실행 중 **주기적으로** 아래 항목을 모니터링해야 한다. OOM(Out of Memory)이 발생하면 에러 메시지 없이 프로세스가 무응답 상태로 멈추므로, 능동적인 모니터링이 필수.

### 기준 메트릭 (9개 항목)

모니터링 및 성능 비교 시 아래 메트릭을 기준으로 한다. COMPLETE 메시지에 **9개 항목**이 출력된다:

| # | 항목 | 설명 |
|---|------|------|
| 1 | **PRC** | Adaptive anomaly score의 PRC-AUC |
| 2 | **F1_T** | Adaptive의 Time-series F1 (QuoVadisTAD) |
| 3 | **PA20** | Adaptive의 Point-Adjusted F1 (k=20) |
| 4 | **t_PRC** | Teacher-only reconstruction의 PRC-AUC |
| 5 | **t_F1_T** | Teacher-only의 Time-series F1 |
| 6 | **t_PA20** | Teacher-only의 Point-Adjusted F1 (k=20) |
| 7 | **d_SNR** | Discrepancy loss의 SNR = (disc_anomaly - disc_normal) / (disc_anomaly_std + disc_normal_std) |
| 8 | **소요시간** | 총 소요시간 (train + eval + viz) |
| 9 | **진행상황** | `[N/15]` 형식으로 데이터셋 진행 |

출력 예시:
```
[base_simulation] [1/15] COMPLETE (992s): PRC=0.9550 F1_T=0.8991 PA20=0.8879 | t_PRC=0.8234 t_F1_T=0.7654 t_PA20=0.7321 | d_SNR=1.23
```

- **epoch callback** (학습 중): all-patches GPU inference + point-level evaluation 직접 수행 → `PRC`, `F1_T`, `PA20`, `t_PRC`, `t_F1_T`, `d_SNR` 출력 + `(infer=Ns eval=Ns)` 소요시간 표시
- **background COMPLETE** (full eval 후): 위 9개 항목 모두 출력

### 모니터링 명령어

```bash
# 1. 학습 진행 — epoch eval 결과 확인 (PRC, F1, t_PRC, d_SNR)
grep -o '\[Epoch.*d_SNR=[0-9.-]*' /tmp/claude-1000/.../tasks/{task_id}.output

# 2. 데이터셋 진행 — 완료/에러 (8개 메트릭 출력됨)
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

모니터링 시 완료된 데이터셋 결과를 **테이블 형식**으로 출력한다. 9개 컬럼 + Dataset 이름:

```
| Dataset                    | PRC    | F1_T   | PA20   | t_PRC  | t_F1_T | t_PA20 | d_SNR | Time  | Progress |
|----------------------------|--------|--------|--------|--------|--------|--------|-------|-------|----------|
| simulation                 | 0.9550 | 0.8991 | 0.8879 | 0.8234 | 0.7654 | 0.7321 |  1.23 | 992s  | [1/15]   |
| simulation_normal50        | 0.8267 | 0.8858 | 0.8741 | 0.7891 | 0.7432 | 0.7198 |  1.05 | 972s  | [2/15]   |
| simulation_complex         |  ...   |  ...   |  ...   |  ...   |  ...   |  ...   |  ...  |  ...  | [3/15]   |
```

- 각 데이터셋의 `COMPLETE` 메시지에서 9개 값을 추출하여 테이블에 추가
- 학습 중인 데이터셋은 최신 epoch 결과를 별도로 표시:
  ```
  Training: simulation_complex [Epoch 25] PRC=0.7123 t_PRC=0.6543 d_SNR=0.85
  ```
- 테이블과 함께 **todo list도 같이 출력**하여 전체 진행 상황 파악

**데이터 추출 방법**:
```bash
# COMPLETE 메시지에서 메트릭 추출
grep 'COMPLETE' /tmp/claude-1000/.../tasks/{task_id}.output

# experiment_metadata.json에서 추출 (더 정확)
cat results/experiments/{dir}/{group}/{scenario}/{ts}_default/experiment_metadata.json | python -c "
import json,sys; m=json.load(sys.stdin)
ad=m['metrics']; tr=m.get('teacher_recon_metrics',{})
print(f'PRC={ad[\"prc_auc\"]:.4f} F1_T={ad.get(\"f1_t\",0):.4f} PA20={ad.get(\"pa_20_f1\",0):.4f}')
print(f't_PRC={tr.get(\"prc_auc\",0):.4f} t_F1_T={tr.get(\"f1_t\",0):.4f} t_PA20={tr.get(\"pa_20_f1\",0):.4f}')
"
```

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
├── {DatasetGroup}/{Scenario}/{YYYYMMDD_HHMMSS}_default/
│   ├── best_model.pt                      # 학습된 모델 weights + config dict
│   ├── best_config.json                   # 전체 config dict (~61 fields)
│   ├── training_histories.json            # 학습 loss 이력 (epoch별 loss values)
│   ├── epoch_metrics.json                 # epoch별 point-level 평가 (아래 형식 참조)
│   ├── batch_profiling.json               # 첫 N배치 per-component + per-layer 타이밍 (batch 0 제외)
│   ├── batch_profiling.txt               # profiler 형태 요약 테이블 (layer 분해 포함) + per-batch 상세
│   ├── experiment_metadata.json           # 최종 평가 메트릭 + 세부 타이밍 (아래 형식 참조)
│   ├── best_model_detailed.csv            # per-window 상세 결과
│   ├── anomaly_type_metrics.json          # anomaly type별 성능
│   ├── checkpoints/                       # epoch별 모델 체크포인트
│   │   ├── epoch_005.pt                   # {epoch, model_state_dict, config, metrics}
│   │   ├── epoch_010.pt
│   │   └── ...                            # 매 EVAL_INTERVAL(5) epoch마다 저장
│   └── visualization/
│       ├── best_model/                    # 15+ PNGs (BestModelVisualizer.generate_all)
│       └── epoch_metrics/                 # 6 PNGs (학습 동태)
```

suffix: Set A = `w500p5e2t4d1`, Set B = `w500p20e2t4d1_d256k5`, Set C = `w500p10e2t4d1_dynamic_linear` (번호는 자동 부여, 예: `7_20260221_202606_w500p5e2t4d1`)
DatasetGroup/Scenario 매핑: `scripts/run_base_experiments.py` → `DATASETS` 리스트의 `results_subdir` 필드.

### experiment_metadata.json (핵심 결과 파일)

```json
{
  "experiment_name": "base_simulation",
  "scoring_mode": "adaptive",
  "train_time": 234.5,
  "inference_time": 45.2,
  "metrics": {
    "roc_auc": 0.9734, "prc_auc": 0.9216, "f1_score": 0.8891,
    "pa_20_roc_auc": ..., "pa_20_f1": ...,
    "pa_50_roc_auc": ..., "pa_80_roc_auc": ...
  },
  "disc_metrics": { "roc_auc": ..., "f1_score": ... },
  "teacher_recon_metrics": { "roc_auc": ..., "f1_score": ... },
  "student_recon_metrics": { "roc_auc": ..., "f1_score": ... },
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
      "teacher_prc_auc": 0.78, "teacher_f1_t": 0.71, "teacher_pa_20_f1": 0.68,
      "disc_snr": 1.23,
      "callback_time": 55.2, "_inference_time": 48.1, "_eval_time": 7.1
    },
    ...
  ]
}
```

50 epochs / interval 5 = 10개 epoch entries (5, 10, 15, ..., 50).
각 entry에 소요시간(callback_time, _inference_time, _eval_time)도 포함.

### visualization/ 디렉토리

**best_model/** (15+ PNGs): `BestModelVisualizer.generate_all()`이 생성.
- 주요 차트: score distribution, ROC curve, PRC curve, confusion matrix, feature-wise heatmap, time-series overlay, anomaly type comparison, score timeline, reconstruction samples 등.

**epoch_metrics/** (4 PNGs): `plot_epoch_metrics()`이 생성.
- `epoch_prc_auc.png`: Adaptive + Teacher PRC-AUC 추이
- `epoch_f1_t.png`: Adaptive + Teacher F1_T 추이
- `epoch_pa_k_f1.png`: PA%K F1 추이 (PA0, PA20, PA50, PA100)
- `epoch_dashboard.png`: 4개 패널 종합 대시보드 (PRC, F1_T, PA%K, disc_SNR)

## 파이프라인

`run_base_experiments.py` 내 `run_base_experiment()`:

1. 데이터 로딩 (`get_dataset_loader()`)
2. GPU 학습 (`Trainer.train(epoch_callback=, profile_n_batches=10)` — 매 5 epoch **all-patches** 평가 + 체크포인트 저장)
   - **에포크 1 직후 즉시 출력**: 첫 10배치 (batch 0 제외, CUDA warmup 왜곡 회피) per-component + per-layer `cuda.synchronize()` 타이밍 테이블 + 예상 잔여 학습시간 → `batch_profiling.json` + `batch_profiling.txt`
     - Batch level: data→GPU, model_forward, loss_compute, backward, optimizer_step
     - Layer level (model_forward 내부): embed_input(Patchify+CNN), masking, encoder, teacher_decoder, student_decoder
   - per-epoch 타이밍: `train_epoch` (forward/backward), `contrib_ratios`, `callback` → `history['epoch_timings']`
   - epoch callback: GPU all-patches inference + point-level evaluation 직접 수행 → `epoch_metrics.json`
4. `epoch_metrics.json` + epoch 시각화 저장
5. GPU 추론 (patch_scores → `patch_scores_time`, viz_data → `viz_collect_time`)
6. GPU 해제 (`free_gpu()`)
7. Background CPU 프로세스 (`_cpu_eval_viz_worker`) 스폰:
   - `Evaluator.set_precomputed_patch_scores()` → `evaluate()` → `cpu_eval_time`
   - `compute_loss_statistics()` (from `run_ablation.py`)
   - `BestModelVisualizer.generate_all()` → 15+ PNGs → `cpu_viz_time`

최대 동시 background 프로세스: eval+viz **10개**. GPU는 즉시 다음 데이터셋 학습 시작.

**중요**: Inference는 반드시 **all-patches masking** 방식을 사용. 각 patch를 하나씩 마스킹하여 N번 forward pass 수행 후 patch별 score를 집계. last-patch masking은 사용 금지.

## 핵심 참조 파일

| 용도 | 파일 |
|------|------|
| 실험 실행 | `scripts/run_base_experiments.py` |
| Config preset/defaults | `mae_anomaly/utils/experiment.py` (`make_config`) |
| Config 전체 필드 | `mae_anomaly/config.py` |
| 데이터셋 로더 | `mae_anomaly/datasets/loaders.py` (`DATASET_LOADERS`) |
| 평가 | `mae_anomaly/evaluator.py` |
| 시각화 | `mae_anomaly/visualization/best_model_visualizer.py` (`generate_all`) |
| Loss 통계 | `scripts/ablation/run_ablation.py` (`compute_loss_statistics`) |
| 기존 실험 point-level 평가 (임시) | `scripts/eval_epoch_pointlevel.py` |
| Baseline 비교 | `comparison/GUIDE.md` |
