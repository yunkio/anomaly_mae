# 실험 지시 프롬프트 (다른 컴퓨터의 새 세션에서 paste용)

다음 내용을 새 Claude Code 세션에 그대로 복사하여 시작하세요.

**대상 머신**: 다른 컴퓨터 WSL Ubuntu-22.04
**작업 디렉토리**: `/home/ykio/notebooks/TSMAE` (WSL 내부 경로)
**Windows 경로**: `\\wsl.localhost\Ubuntu-22.04\home\ykio\notebooks\TSMAE`

---

# 작업 개요

GRL/SCAD ablation 실험 **21개 (Exp 285-305)** 를 처음부터 끝까지 실행합니다.

- **작업 디렉토리**: `/home/ykio/notebooks/TSMAE`
- **Queue file**: `/home/ykio/notebooks/TSMAE/configs/queue_exp285_305_4ds.json` (21 entries, 작성 완료)
- **Base**: Exp 271 ablation (e4t3d2, ep=500/w=250, FM+OD+GRL+L2+adp+minmax+window+slow_cls)
- **Datasets** (4개): `SWaT_A1A2` (full+excl22 dual eval 자동), `WaDi_A1`, `WaDi_A2`, `PSM`
- **각 실험**: 500 epochs, 4 datasets, 추정 ~12-15h
- **총 추정**: 21 × ~13h ≈ **11-13일**

## 환경 (CRITICAL)

- **Conda env**: `dc_vis` (이미 활성화되어 있어야 함, default python = `/home/ykio/anaconda3/envs/dc_vis/bin/python`)
  - **확인**: `which python` → `/home/ykio/anaconda3/envs/dc_vis/bin/python` 출력되어야 함
  - 아니면: `conda activate dc_vis` 후 재확인
- **절대 금지**: `conda run -n dc_vis python ...` 형태 (stdout 버퍼링됨, 실시간 모니터링 불가)
- **백그라운드 실행**: Bash tool의 `run_in_background=true` 파라미터 사용
- **참조 문서**:
  - `/home/ykio/notebooks/TSMAE/CLAUDE.md` (프로젝트 규칙)
  - `/home/ykio/notebooks/TSMAE/set_guideline.md` (실험 가이드라인)

## 실험 21개 분류

| Exp | Name | 271 대비 변경 |
|-----|------|------|
| 285 | 271_no_fm | use_feature_matching=False (FM 제거) |
| 286 | 271_minmax_neg11_clamp_pm4 | minmax_range=neg1_1, clamp±4 |
| 287 | 271_unmask | force_mask_anomaly=False |
| 288 | 271_no_focal | grl_use_focal=False |
| 289-291 | 271_w{300,200,100}_p10 | seq_length 변형 |
| 292-294 | 271_w{500,200,100}_p5 | patch_size=5 변형 |
| 295 | 271_target_patch | grl_target_mode=patch |
| 296 | 271_grl_w1 | grl_loss_weight=1.0 |
| 297 | 271_cls_lr1 | grl_cls_lr_ratio=1.0 |
| 298-300 | 271_revin{,_no_affine,_visible_only} | RevIN 3변형 (zscore 동반) |
| 301 | 271_noadv | use_grl=False, use_scad=False (Group Q baseline) |
| 302-305 | 271_scad{A_default,A_w10,B_default,B_w10} | SCAD 2×2 grid |

---

# 사전 체크 0: 머신 환경 검증 (실행 전 필수)

## 0-1. 작업 디렉토리 존재 확인

```bash
cd /home/ykio/notebooks/TSMAE
pwd
ls -la
```
**기대**: `/home/ykio/notebooks/TSMAE` 디렉토리 존재 + `mae_anomaly/`, `scripts/`, `configs/`, `dataset/` 등 폴더 있음.

## 0-2. Conda env 검증

```bash
which python
python -c "import torch; print(f'PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}, GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```
**기대**:
- `which python` → `/home/ykio/anaconda3/envs/dc_vis/bin/python`
- PyTorch + CUDA True + GPU 인식

만약 conda env 다르면: `conda activate dc_vis` 후 재시도.

## 0-3. 핵심 스크립트 존재 확인

```bash
ls -la /home/ykio/notebooks/TSMAE/scripts/run_queue.py \
       /home/ykio/notebooks/TSMAE/scripts/run_base_experiments.py \
       /home/ykio/notebooks/TSMAE/scripts/collect_telemetry.sh \
       /home/ykio/notebooks/TSMAE/scripts/plot_gpu_telemetry.py \
       /home/ykio/notebooks/TSMAE/configs/queue_exp285_305_4ds.json
```
**기대**: 5개 파일 모두 존재.

만약 `collect_telemetry.sh` 또는 `plot_gpu_telemetry.py` 없으면:
- 사용자에게 다른 컴퓨터에서 작성된 스크립트가 동기화되었는지 확인 요청

## 0-4. 데이터셋 존재 확인

```bash
ls /home/ykio/notebooks/TSMAE/dataset/SWaT/ \
   /home/ykio/notebooks/TSMAE/dataset/WaDi/ \
   /home/ykio/notebooks/TSMAE/dataset/PSM/ 2>&1 | head -30
```
**기대**: SWaT, WaDi, PSM raw data 파일 존재.

## 0-5. GPU 확인

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
```
**기대**: GPU 인식 + 메모리 정상 + idle 상태.

---

# 사전 체크 1: 디렉토리 + 넘버링 초기화 ⚠️ CRITICAL

⚠️ **새 머신은 `results/experiments/` 디렉토리가 없거나 비어있음**. run_base_experiments.py의 자동 numbering은 **`{N}_*` 디렉토리 중 max N + 1**을 사용. 그러므로 빈 디렉토리에서 시작하면 **0번부터** 시작됨.

**목표**: 첫 실험이 **285번**부터 시작되도록 초기 setup.

## 1-1. results 디렉토리 생성

```bash
mkdir -p /home/ykio/notebooks/TSMAE/results/experiments
ls /home/ykio/notebooks/TSMAE/results/experiments/
```
**기대**: 비어있음.

## 1-2. 넘버링 marker 생성 (284_marker)

`mae_anomaly/utils/experiment.py::get_next_experiment_number`는 **`{N}_*` 패턴의 max N + 1**을 반환. 즉 `284_marker/` 디렉토리 1개만 있으면 다음 자동 번호 = **285**.

```bash
mkdir -p /home/ykio/notebooks/TSMAE/results/experiments/284_NUMBERING_MARKER_DO_NOT_DELETE
echo "Purpose: numbering anchor — next auto-numbered exp will be 285_..." > \
     /home/ykio/notebooks/TSMAE/results/experiments/284_NUMBERING_MARKER_DO_NOT_DELETE/README.txt
ls /home/ykio/notebooks/TSMAE/results/experiments/
```
**기대**: `284_NUMBERING_MARKER_DO_NOT_DELETE/` 1개.

## 1-3. 넘버링 검증

```bash
cd /home/ykio/notebooks/TSMAE
python3 -c "
from mae_anomaly.utils.experiment import get_next_experiment_number
n = get_next_experiment_number('/home/ykio/notebooks/TSMAE/results/experiments')
print(f'Next experiment number: {n}')
assert n == 285, f'ERROR: Expected 285, got {n}'
print('OK — first experiment will be 285_...')
"
```
**기대**:
```
Next experiment number: 285
OK — first experiment will be 285_...
```

만약 285 아니면 → **즉시 사용자에게 보고** (잘못된 marker, 또는 다른 `{N}_*` 디렉토리 존재).

---

# 사전 체크 2: 프로세스 완전 정리 (PGID 단위 kill)

⚠️ **이전 세션에서 stall 5번 반복된 원인**: 이전 attempt의 좀비 pt_main_thread가 살아남아 CPU 100% 점유. `pkill -f pattern` 은 multi-thread spawn worker 일부를 놓침. **PGID 단위 `kill -9 -PGID`** 가 정답.

새 머신이라도 첫 시작 전에 확인:

```bash
# Step 1: 살아있는 process 확인
ps -eo pid,user,etime,pcpu,stat,comm | grep -E "pt_main_thread|python.*scripts/run|collect_telemetry" | grep -v grep
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
```
**기대**: 0개, GPU idle (< 2000 MiB, < 10% util).

만약 있으면:
```bash
# PGID 단위 kill
for PID in $(ps -eo pid,comm | awk '$2=="pt_main_thread" {print $1}'); do
  PGID=$(ps -o pgid= -p $PID 2>/dev/null | tr -d ' ')
  [ -n "$PGID" ] && kill -9 -$PGID 2>/dev/null && echo "Killed PGID $PGID"
done
for PID in $(pgrep -f "scripts/run_base_experiments|scripts/run_queue|collect_telemetry" 2>/dev/null); do
  PGID=$(ps -o pgid= -p $PID 2>/dev/null | tr -d ' ')
  [ -n "$PGID" ] && kill -9 -$PGID 2>/dev/null
done
sleep 3

# 재검증
ps -eo pid,comm | awk '$2=="pt_main_thread" || /python.*scripts/' | grep -v grep && echo "ERROR: STILL ALIVE" || echo "CLEAN"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

---

# 사전 체크 3: Queue file 검증

```bash
cd /home/ykio/notebooks/TSMAE
python3 -c "
import json
d = json.load(open('configs/queue_exp285_305_4ds.json'))
print(f'Total: {len(d[\"experiments\"])} experiments')
ds_sets = {tuple(e['dataset']) for e in d['experiments']}
print(f'Datasets uniform: {ds_sets}')
print(f'First: {d[\"experiments\"][0][\"name\"]}')
print(f'Last: {d[\"experiments\"][-1][\"name\"]}')
"
```

**기대 출력**:
```
Total: 21 experiments
Datasets uniform: {('SWaT_A1A2', 'WaDi_A1', 'WaDi_A2', 'PSM')}
First: exp285_271_no_fm
Last: exp305_271_scadB_w10
```

---

# 실행

## Plan 요약
- **Phase 1**: 285, 286 학습 + **GPU telemetry 수집** (10초 주기, 36 metrics)
- **Phase 2**: 286 끝나면 → 285, 286 각각 plot 4종 생성 → `./temp/`
- **Phase 3**: 287-305 queue 실행 (telemetry 없음)
- **Phase 4**: 최종 검증

## Phase 1 — 285, 286 학습 + telemetry

### A. Telemetry collector 가동 (백그라운드, 10초 주기)

```bash
mkdir -p /home/ykio/notebooks/TSMAE/temp
TELEMETRY_CSV=/home/ykio/notebooks/TSMAE/temp/telemetry_285_286_$(date +%Y%m%d_%H%M%S).csv

# Background (run_in_background=true)
bash /home/ykio/notebooks/TSMAE/scripts/collect_telemetry.sh $TELEMETRY_CSV
```
→ `run_in_background=true` 로 실행. Task ID + CSV path **반드시 기억**.

### B. Queue 285, 286만 학습 (백그라운드)

```bash
# 임시 queue file (285, 286 2개만)
python3 -c "
import json
d = json.load(open('/home/ykio/notebooks/TSMAE/configs/queue_exp285_305_4ds.json'))
d['experiments'] = d['experiments'][:2]
json.dump(d, open('/tmp/queue_285_286.json', 'w'), indent=2)
print('Saved /tmp/queue_285_286.json with 2 experiments')
"

cd /home/ykio/notebooks/TSMAE
python scripts/run_queue.py --queue /tmp/queue_285_286.json
```
→ `run_in_background=true`. **Task ID 기억** (학습 log path 사용).

### C. 모니터링 (10분 주기, stall 의심 시 30초 후 재확인)

10분 주기 health snapshot monitor:

```bash
TRAIN_LOG=<위 B에서 받은 task output path>
CSV=$TELEMETRY_CSV
PREV_EP=""
PREV_TS=0
while true; do
  TS=$(date '+%H:%M:%S')
  EP=$(grep -oE 'Epoch [0-9]+/500' $TRAIN_LOG 2>/dev/null | tail -1)
  N_SAMPLES=$(tail -n +2 $CSV 2>/dev/null | wc -l)
  LAST=$(tail -3 $CSV 2>/dev/null | awk -F, '$2 ~ /^[0-9]/ {printf "%s gpu=%dMiB/util%s%%/pwr%sW/T%s°C | RAM=%dMiB cpu(us%s id%s wa%s) load=%s; ", substr($1,12,8),$3,$6,$8,$10,$14,$32,$34,$35,$21}')
  STATUS="OK"
  NOW=$(date +%s)
  if [ -n "$EP" ] && [ "$EP" = "$PREV_EP" ] && [ "$PREV_TS" -gt 0 ]; then
    DIFF=$((NOW - PREV_TS))
    if [ "$DIFF" -gt 600 ]; then
      STATUS="STALL_${DIFF}s"
    fi
  else
    PREV_EP=$EP
    PREV_TS=$NOW
  fi
  echo "[$TS] ${EP:-init} samples=${N_SAMPLES} ${STATUS} | ${LAST}"
  sleep 600
done
```
→ Monitor tool로 가동 (persistent=true).

**stall 의심 시** (GPU util < 10%, pwr < 200W, epoch 변화 없음):
30초 후 재확인:
- Epoch 변화 있으면 → callback timing, 무시
- 변화 없으면 → STALL 확정 → 즉시 사용자에게 보고

## Phase 2 — 285, 286 학습 완료 후 Plot 생성

학습 완료 후 (run_queue.py 종료 알림 수신):

### A. Telemetry collector 종료

```bash
# Telemetry collector task의 TaskStop
```

### B. Dataset별 timestamp 추출

```bash
TRAIN_LOG=<task output>
# 각 dataset start/end 시각 추출
grep -nE "Base Experiment:|COMPLETE \([0-9]+s\)" $TRAIN_LOG
# COMPLETE 메시지 시각 = dataset 끝. Base Experiment: 메시지 = dataset 시작.
```

### C. Plot 생성 (각 실험 × 4 datasets × 4 categories = 32 plots)

```bash
# 285 (4 datasets × 4 categories = 16 plots)
for DS in swat wadi_a1 wadi_a2 psm; do
  # DS_START_UNIX, DS_END_UNIX는 log에서 추출
  python /home/ykio/notebooks/TSMAE/scripts/plot_gpu_telemetry.py \
    --csv $TELEMETRY_CSV \
    --out-dir /home/ykio/notebooks/TSMAE/temp \
    --prefix 285_${DS} \
    --title-suffix "exp285 271_no_fm — ${DS}" \
    --start-unix $DS_START_UNIX \
    --end-unix $DS_END_UNIX
done

# 286도 동일하게 (prefix 286_, title suffix 변경)
```

생성될 파일 (총 32개):
- `./temp/285_{swat,wadi_a1,wadi_a2,psm}_{main,gpu_details,memory,cpu_system}.png` (16개)
- `./temp/286_{swat,wadi_a1,wadi_a2,psm}_{main,gpu_details,memory,cpu_system}.png` (16개)

## Phase 3 — 287-305 queue 실행 (telemetry OFF)

```bash
# 287-305 queue (19 entries)
python3 -c "
import json
d = json.load(open('/home/ykio/notebooks/TSMAE/configs/queue_exp285_305_4ds.json'))
d['experiments'] = d['experiments'][2:]  # 287-305
json.dump(d, open('/tmp/queue_287_305.json', 'w'), indent=2)
print(f'Saved /tmp/queue_287_305.json with {len(d[\"experiments\"])} experiments')
"

cd /home/ykio/notebooks/TSMAE
python scripts/run_queue.py --queue /tmp/queue_287_305.json
```
→ `run_in_background=true`. 모니터링 (10분 주기, stall 시 30초 후 재확인) 동일.

추정 시간: 19 × ~13h ≈ **10일**.

## Phase 4 — 최종 검증

```bash
# 21개 실험 디렉토리 + 각 dataset 결과 점검
cd /home/ykio/notebooks/TSMAE
for N in $(seq 285 305); do
  DIR=$(ls -d /home/ykio/notebooks/TSMAE/results/experiments/${N}_*/ 2>/dev/null | head -1)
  if [ -z "$DIR" ]; then
    echo "Exp${N}: MISSING"
    continue
  fi
  MISSING=""
  for DS in SWaT/A1A2_full SWaT/A1A2_excl22 WaDi/A1 WaDi/A2 PSM; do
    [ ! -f "$DIR$DS/experiment_metadata.json" ] && MISSING="$MISSING $DS"
  done
  if [ -z "$MISSING" ]; then echo "Exp${N}: OK"
  else echo "Exp${N}: MISSING:$MISSING"
  fi
done
```

---

# 모니터링 요령 (CRITICAL)

## 모니터링 주기
- **10분 주기**: 자동 health snapshot (epoch 진행, GPU/RAM/CPU)
- **Dataset 전환 시점**: milestone 알림 (`# [N/4] DATASET` log line)
- **Epoch 변경 없으면 (>10분)**: STALL 감지 → 30초 후 재확인
  - 재확인에서 epoch 변경되면 → false alarm (callback timing)
  - 재확인에서도 변경 없으면 → 진짜 STALL → 사용자 보고

## Stall 판단 기준
- GPU util < 10% (지속)
- GPU power < 200W (정상 학습 320-380W의 절반)
- CPU us 100% saturated + load > 30 (좀비 multi-process 의심)
- 30초간 epoch 변화 없음

## Stall 진짜 발생 시 대응
1. 사용자에게 즉시 보고 (kill/대기 여부 결정)
2. Kill 결정 시 PGID 단위 (`kill -9 -PGID`)
3. Kill 후 GPU 메모리 < 2000 MiB 확인
4. 부분 결과 디렉토리 백업 → `/home/ykio/notebooks/TSMAE/.trash/$(date +%y%m%d)/`
5. 재학습 시 같은 명령 사용

## 정상 동작 판단
- `[Epoch N]` 출력이 5 epoch 간격으로 나옴
- PRC > 0.3 (random baseline 이상)
- GPU 메모리 안정적 (변동 < 1GB)
- GPU power 320-380W (학습 중)
- 각 dataset 완료 후 "Spawning background eval+viz" 메시지
- `free -h` available > 5GB

## 정상 학습 패턴 (참고)
- 1 epoch: ~11-21초 (dataset에 따라)
- callback eval: 5 epoch마다 ~50초 추가 (GPU inference + background eval+viz spawn)
- 1 dataset 학습 (500 epoch): ~2-4시간
- 1 실험 (4 datasets): ~10-15시간

---

# 핵심 참고 자료

| 용도 | 경로 |
|------|------|
| Queue 실행 | `/home/ykio/notebooks/TSMAE/scripts/run_queue.py` |
| 실험 실행 | `/home/ykio/notebooks/TSMAE/scripts/run_base_experiments.py` |
| Telemetry collector | `/home/ykio/notebooks/TSMAE/scripts/collect_telemetry.sh` |
| Plot 생성 | `/home/ykio/notebooks/TSMAE/scripts/plot_gpu_telemetry.py` |
| Queue config | `/home/ykio/notebooks/TSMAE/configs/queue_exp285_305_4ds.json` |
| 가이드라인 | `/home/ykio/notebooks/TSMAE/set_guideline.md` |
| 프로젝트 규칙 | `/home/ykio/notebooks/TSMAE/CLAUDE.md` |
| 백업 위치 | `/home/ykio/notebooks/TSMAE/.trash/$(date +%y%m%d)/` |
| 결과 위치 | `/home/ykio/notebooks/TSMAE/results/experiments/{N}_{timestamp}_*/` |

---

# 이전 세션에서 발견한 주요 issue 및 해결

1. **WaDi_A2 stall 5회 발생** → 원인: 이전 attempt의 좀비 pt_main_thread가 CPU 100% 점유. 새 학습 시작 전 반드시 PGID 단위 강제 kill 검증.

2. **SCAD bug fix** (`mae_anomaly/loss.py:328`) — `self.config` 참조 문제. `__init__`에 `self.use_scad`, `self.scad_form`, `self.scad_temperature`, `self.scad_margin`, `self.scad_patch_label_mode` 추가하여 fix됨 ✓ (이미 commit됨, 새 컴퓨터 코드에도 반영되어 있어야 함).

3. **background eval+viz hang** → 매우 가끔 발생 (특히 SWaT_excl22). 학습 후 weight 자동 삭제되므로 hang 시 viz 재생성 불가능. 발생하면 dataset 재학습 필요.

4. **Monitor 잘못된 file grep** → 파일 path 명시적으로 지정 (glob `*.output` 사용 금지).

5. **Kill 실패** → `pkill -f pattern` 은 spawn worker 일부 놓침. PGID 단위 `kill -9 -PGID` 가 정답.

6. **새 머신 numbering 초기화** → 빈 `results/experiments/`에서 시작하면 다음 자동 번호 = 0. `284_marker/` 더미 디렉토리 생성으로 next = 285 보장 (필수, 위 사전 체크 1-2).

---

# 진행 시 사용자 확인 사항

다음 사항은 사용자 확인 받고 진행:
- Stall 확정 후 kill/대기 결정
- 디렉토리 백업 후 삭제
- Phase 간 전환 (Phase 1 → Phase 2 → Phase 3)
- Queue 중단 결정

진행 상황은 적극적 모니터링 (10분 주기 + dataset 전환 milestone). Idle 대기 금지.

---

작업 시작 전 **사전 체크 0 → 1 → 2 → 3 순서**로 모두 통과한 후 Phase 1 실행.

특히 **사전 체크 1-3 (넘버링 검증)**에서 `Next experiment number: 285` 출력 확인 필수 — 아니면 첫 학습이 잘못된 번호로 진행됨.
