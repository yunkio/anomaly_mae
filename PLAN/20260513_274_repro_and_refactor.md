# 274 재실험 + 시각화/저장 리팩토링 — 실행 계획

작성: 2026-05-13
출처 요청: 사용자 메시지 5개 작업 (1)~(5)

---

## 0. 코드베이스 사전 조사 결과 (요약)

### 0-A. 디렉토리 카운터 메커니즘
- `mae_anomaly/utils/experiment.py:33` `get_next_experiment_number(dir)` = MAX(existing N) + 1
- `make_numbered_experiment_dir(experiments_dir, suffix)`로 디렉토리 생성
- `run_base_experiments.py:2459` 에서 `make_numbered_experiment_dir` 호출
- **`--output-base` 인자로 명시적 경로 지정 가능** (line 2453-2454)
- `run_queue.py:223` 가 `--output-base` 를 자식 프로세스에 전달함 → 큐에서도 명시 가능

### 0-B. best_checkpoint.pt 삭제 위치
- `run_base_experiments.py:2092` `os.remove(best_ckpt_path)` — best epoch 학습 종료 후 자동 삭제
- 274만 보존하려면 환경변수 또는 dataset_def 플래그 필요

### 0-C. epoch_scores npz 저장 위치 (현재)
- `run_base_experiments.py:1875-1885`
- 현재 저장 키: `adaptive_score`, `teacher_recon_error`, `discrepancy_error`, `point_labels`, (조건부) `fm_error` — 모두 point-level mean 집계 (shape=(T,))
- patch-level raw: `recon_p`, `disc_p`, `fm_p` 는 line 1821-1824에서 이미 정의됨 (shape=(n_windows, num_patches))
- `adaptive_patch` 는 line 1843-1846에서 정의됨 (shape=(n_windows, num_patches))
- `ws_indices` 는 line 1820에서 `np.array(test_dataset.window_start_indices)` 로 정의됨 (shape=(n_windows,))

### 0-D. anomaly_threshold.png 현재 상태 (BUG)
- 279/simulation 출력 PNG: anomaly 영역 빨간 음영 **없음**, detection ratio 라벨 **없음**, best_epoch=15 (의심)
- 코드 (`best_model_visualizer.py:2906-3128`) 는 `axvspan` + detection ratio 로직을 포함하지만, 실제 출력에서 누락
- 원인 추정: `test_dataset.point_labels` 가 비어있거나, `m=min(len(adaptive_score), len(point_labels))` 가 0이 되는 케이스
- 정상 동작 참조: `scripts/visualize_score_components.py:127-235` (`_plot_score_components_on_axes`)

### 0-E. 기존 실험 config 매트릭스 (264-278)
exp274 base config = `ep=500, warmup=250, balanced=True, fm=l2, fm_adapt=False, grl_w=0.2, cls_lr=0.1, cls_arch=default, focal=True, anomaly_w=2.0, normal_w=1.0`

| Exp | ep | balanced | fm_dist | fm_adapt | grl_w | focal | cls_lr | normal_w | 기타 |
|-----|-----|----------|---------|----------|-------|-------|--------|----------|------|
| 264 | 400 | True | cosine | False | 0.2 | T | 0.1 | 1.0 | base |
| 265 | 500 | True | cosine | False | 0.2 | T | 0.1 | 1.0 | base+ep500 |
| 266-268 | **300** | True | cosine | False | 0.2 | T | 0.1 | 1.0 | base+ep300 |
| 269 | 500 | False | cosine | False | 0.2 | T | 0.1 | 1.0 | 265−balanced |
| 270 | 500 | False | l2 | False | 0.2 | T | 0.1 | 1.0 | 269+fm_l2 |
| 271 | 500 | False | l2 | True | 0.2 | T | 0.1 | 1.0 | 270+fm_adapt |
| 272 | 200 | True | cosine | False | 0.2 | T | 0.1 | 1.0 | base+ep200 |
| 273 | 500 | False | cosine | False | 0.2 | T | 0.1 | 1.0 | =269? 중복 |
| 274 (삭제) | 500 | True | **l2** | False | 0.2 | T | 0.1 | 1.0 | 265+fm_l2 |
| 275 | 500 | True | cosine | False | 0.2 | **F** | 0.1 | 1.0 | 265+no_focal |
| 276 | 500 | True | cosine | False | **0.5** | T | 0.1 | 1.0 | 265+grl_w0.5 |
| 277 | 500 | True | cosine | False | 0.2 | T | 0.1 | 1.0 | =265? 중복 |
| 278 | (no meta) | — | — | — | — | — | — | — | 미완료/실패 |

### 0-F. 큐 (queue_exp278_290.json) 구성 (11개)
모두 base에 fm=l2 기반:
- **exp278_247_adapt_off_w005_ep500** = ep=500, balanced=True, fm=cosine, grl_w=0.05, **adapt=False** (특수)
- **exp279_274_no_balanced** = 274 − balanced (=270+balanced=False, 이미 270 존재)
- **exp280_274_fm_adaptive** = 274 + fm_adapt=True
- **exp281_271_fm_l2** = 271 (이미 존재) — 그러나 이미 271 dir 있어서 사실상 재실험
- **exp282_274_cls_lr_025** = 274 + cls_lr=0.25
- **exp283_274_cls_2layer** = 274 + cls_arch=2layer
- **exp284_274_ep750** = 274 + ep=750
- **exp286_274_normal_w3** = 274 + normal_w=3.0
- **exp287_274_no_focal** = 274 + focal=False
- **exp289_271_no_balanced** = 271은 이미 balanced=False → 중복?
- **exp290_274_ep300** = 274 + ep=300

---

## 1. 작업 5개 (사용자 요청)

### Task (1) — 274 재실험 (디렉토리 안 꼬이게)
- (a) 현재 279_ 디렉토리(잘못된 새 274) 학습 프로세스 중단
- (b) 279_ 디렉토리 백업/삭제
- (c) 디렉토리 강제로 274_*로 새로 시작
- (d) 274 한정으로 dataset마다 `best_checkpoint.pt` **삭제 금지** 옵션 적용

**기술 방안**:
- 디렉토리 강제: `--output-base /home/ykio/notebooks/claude/results/experiments/274_<TS>_w500p10e4t3d2_dynamic_linear_minmax_k6` 직접 지정 (run_queue.py가 전달)
- best.pt 보존: 환경변수 `KEEP_BEST_CKPT=1` 도입, `run_base_experiments.py:2092`에서 분기
  ```python
  if os.environ.get('KEEP_BEST_CKPT') != '1':
      os.remove(best_ckpt_path)
  ```
  체인 스크립트에서 274만 env로 plus

### Task (2) — anomaly_threshold.png 재구현
- 현재 best_model_visualizer.plot_anomaly_threshold가 anomaly 영역/라벨 누락하는 버그 수정
- visualize_score_components.py:127-235 (`_plot_score_components_on_axes`)를 참고하여 동일 외형/논리로 포트
- 핵심 누락 원인 디버그:
  - `test_dataset.point_labels` 가 npz와 길이/정합 맞는지
  - SWaT_full 의 경우 dual-eval 슬라이싱 차이
  - 결과 검증: 279 결과로 재생성해서 확인

### Task (3) — 274 + 279~ 실험 큐 시작
- 274 단독 (보존된 best.pt 사용) 학습 큐 새로 작성
- 279~ 부터 새 큐 작성 (디렉토리 카운터 충돌 없도록)
- 체인 스크립트로 일괄 실행

**디렉토리 안 꼬이게 조치**:
- 옵션 A: 각 실험마다 `--output-base` 명시
- 옵션 B: 자동 카운터에 맡기되 첫 실험 종료 후 다음 실험 시작 — race condition 없음 (run_queue.py가 순차 처리)
- 옵션 A가 더 안전 (병렬 실험 대비)

### Task (4) — patch-level raw arrays npz 저장
- `run_base_experiments.py:1875` 의 `save_dict`에 추가:
  - `patch_adaptive_score` = `np.nan_to_num(adaptive_patch).astype(np.float32)` shape=(n_windows, num_patches)
  - `patch_recon_error` = `recon_p.astype(np.float32)` shape=(n_windows, num_patches)
  - `patch_disc_error` = `disc_p.astype(np.float32)` shape=(n_windows, num_patches)
  - `patch_fm_error` (조건부) shape=(n_windows, num_patches)
  - `window_start_indices` = `ws_indices.astype(np.int32)` shape=(n_windows,)
- best_epoch_train_scores.npz 에도 동일 추가
- 추가 디스크 비용 추정: simulation 500 ep × 5 ep 간격 × n_windows × num_patches × float32 — 압축률 50% 가정시 약 100MB/실험 증가 (수용 가능)
- evaluator.py 변경 불필요 (이미 patch-level 변수 제공함 — `eval_data['recon_patches']` 등)

### Task (5) — 279~ 를 ep=300 으로 변경 시 중복 체크
- exp284 (ep=750) → ep=300 변경 시 **exp290 (=274+ep=300)와 완전 동일** ⚠️ 중복
- exp289 (=271 + ep=500, balanced=False) → ep=300 변경 시 **exp281 (=271 fm_l2 + ep=500)와 동일** ⚠️ 중복 (exp281도 ep=300이라 가정)
- exp290 (이미 ep=300) → 변경 사항 없음

**기존 dir과의 중복 (ep=300 가정시)**:
- 266/267/268 는 base=cosine, 새 큐는 base=l2 → 중복 없음
- 다른 ep=300 dir 없음

**결론**: 큐 변경시 **284 + 289 둘 중 하나 또는 둘 다 삭제** 필요. 또는 ep=300 적용 범위를 사용자가 명시.

---

## 2. 체크리스트

### 단계 A — 현재 실험 중단 + 정리
- [ ] A1: chain script PID 359834 kill
- [ ] A2: 274 학습 프로세스 PID 359455/359462/359467 kill
- [ ] A3: 279_ 디렉토리 → `.trash/0513/` 백업 이동
- [ ] A4: viz_after_274.log 등 기존 임시 로그 정리 (선택)

### 단계 B — 코드 변경
- [ ] B1: `run_base_experiments.py` save_dict에 patch_* + window_start_indices 추가 (epoch_scores npz)
- [ ] B2: best_epoch_train_scores.npz 에도 동일 추가
- [ ] B3: `run_base_experiments.py:2092` best.pt 삭제에 `KEEP_BEST_CKPT` 환경변수 분기 추가
- [ ] B4: `best_model_visualizer.plot_anomaly_threshold` 버그 수정 — anomaly region + detection ratio 표시 (visualize_score_components.py 참조)
- [ ] B5: 임시 PNG로 B4 검증 (279 데이터에 generate_all() 재실행)

### 단계 C — 큐 정리
- [ ] C1: 새 queue_exp274_only.json 작성 (디렉토리 명시: `274_<TS>_*`)
- [ ] C2: 새 queue_exp279_290.json 작성 (279부터 명시, 중복 제거, ep=300 적용 여부 반영)
- [ ] C3: 체인 스크립트 `run_274_then_queue.sh` 작성 (`KEEP_BEST_CKPT=1` for 274만)

### 단계 D — 검증
- [ ] D1: 274 simulation 종료 후 npz 키 확인 (`patch_adaptive_score` 등 5개 + 기존 5개)
- [ ] D2: 274 simulation best.pt 보존 확인
- [ ] D3: 274 simulation anomaly_threshold.png 정상 (anomaly region 빨강 + ratio 표시)
- [ ] D4: 279부터 best.pt 삭제됨 확인 (`KEEP_BEST_CKPT` 미적용)

### 단계 E — 실행
- [ ] E1: 274 시작 (foreground 또는 nohup)
- [ ] E2: 274 완료 후 자동으로 279~290 큐 시작
- [ ] E3: 진행 상황 모니터링

---

## 3. 애매한 부분 (사용자 확인 필요)

### Q1. 기존 278_ 디렉토리 처리
- 278_20260512_183536_* 은 old code 시점 생성, npz 새 포맷 미적용
- 옵션: (a) 그대로 유지하고 279부터 새 큐 / (b) 278도 삭제하고 278부터 재실행
- 사용자 발언 "279번~"은 (a)에 가까워 보임

### Q2. "274 best.pt 전부 유지"의 정확한 의미
- 옵션 A: 학습 종료 후 best_checkpoint.pt 삭제하지 않음 (한 dataset당 1개 파일 유지) ← 가장 자연스러운 해석
- 옵션 B: 매 epoch best 갱신마다 별도 파일(`best_epoch_N.pt`) 저장 (디스크 폭증)
- A로 진행 가정. 동의?

### Q3. 큐 (3) 의 범위와 ep
- 사용자 메시지: "279번~ 실험 다시 QUEUE 걸고 시작"
- "279번 실험부터는 epoch 300으로 하고 싶다"
- 옵션 1: 279~290 전부 ep=300 → exp284/289 중복 발생 → **284 (원래 ep=750)와 289 (271_no_balanced) 삭제**
- 옵션 2: ep=300 옵션을 일부에만 적용 (284, 290 같이 ep 자체를 다루는 실험은 원래 값 유지)
- 옵션 3: ep=300 으로 가는 게 전반적 추세이므로 284, 289 제거하고 나머지는 ep=300

### Q4. Notion 페이지 (Exp 119-290) 참조
- 사용자가 링크 제공 ("Mechanism / Depth / Epoch / Optimal Config / Ablation")
- 큐의 의도 (어떤 hypothesis인지) 를 더 명확히 알면 중복 판단 정확
- Notion MCP로 fetch해서 plan에 반영해도 될지

### Q5. 디렉토리 강제 274_<TS>_<SUFFIX> 의 SUFFIX 형식
- 현재 다른 디렉토리들은 `w500p10e4t3d2_dynamic_linear_minmax_k6` 식 자동 suffix 사용
- 274 강제할 때 동일 suffix 유지하면 됨 (`make_dynamic_suffix(overrides)` 결과)
- 단순히 `274_<TS>_w500...` 형식이 적절. 동의?

### Q6. anomaly_threshold.png 외형 — option5_best 그대로?
- 임시 score_components_*.png는 4 panel (anomaly_score / recon / scaled_disc / scaled_fm)
- 임시 option5_best 는 추가로 Gaussian smoothing 적용된 score
- best_model_visualizer.plot_anomaly_threshold 는 **default 모드** (raw adaptive_score)가 자연스러움 — Gaussian 옵션은 안 적용
- 동의?

---

## 4. 의존성 및 위험

### 영향 받는 파일
- `scripts/run_base_experiments.py` (B1, B2, B3)
- `mae_anomaly/visualization/best_model_visualizer.py` (B4)
- `configs/queue_exp274_only.json` (C1, rewrite)
- `configs/queue_exp279_290.json` (C2, new)
- `temp/run_274_then_queue.sh` (C3, new)

### 위험
- patch-level array 추가로 npz 크기 증가 — simulation 약 100MB/실험 → 11실험 × 5dataset × 100 epoch ≈ 비용 추정 필요. 28 SMD 머신 곱하면 큼.
- `KEEP_BEST_CKPT=1` 인 경우 디스크 보존, 큐 진행 후 누적될 수 있음 — 274만 적용이므로 영향 작음
- anomaly_threshold.png 버그 수정 시 다른 호출자(test_loader 형태)에 영향 가능

### 롤백 전략
- 코드 변경 전 `.trash/0513/` 에 백업 복사
- 단계별 commit (B1 → B4 각각)
