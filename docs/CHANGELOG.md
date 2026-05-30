# Changelog

## 2026-05-30: Resume record-consistency (score-contribution off-by-one + lost eval records)

### Problem (관측됨)
1. resume 후 완주한 run 의 `best_model_score_contribution.png` 누락. `training_histories.json` 의 per-epoch contribution-ratio 배열이 `epoch` 보다 1 짧음 (`epoch=500`, `epoch_recon_ratio_*=499`) → stackplot crash → `_safe_plot` swallow.
2. resume 된 dataset 의 `epoch_metrics.json` 에 eval epoch 구멍. `271_lr SWaT-full [285,290]`, `271_lr WaDi/A1 [275,280,350,355]` 누락. pause 직전 1–2 eval 이 영구 손실.

### Root causes (코드 단위)
1. **off-by-one**: 구 checkpoint 저장이 `epoch_callback`(mid-epoch, [trainer.py:1225](mae_anomaly/trainer.py#L1225)) 안에서 history 를 스냅샷 — `epoch` 은 append 됐지만 contribution-ratio 들 ([trainer.py:1245](mae_anomaly/trainer.py#L1245)) 은 아직 append 전 → len(contrib)=N-1 박제.
2. **lost evals**: per-epoch eval 이 background thread 에서 `torch.save` *뒤* (step D) 큐에 put 되고, 그 다음 eval-epoch thread 가 drain → checkpoint_N 이 ep N eval 을 못 담음. pause 가 그 사이에 떨어지면 큐 결과 영구 손실. 게다가 `epoch_metrics_list` 스냅샷이 build 시점(drain 전)에 동기적으로 떠짐.

### Fixes
- **[trainer.py] `post_epoch_callback` 추가** (tracked): epoch loop 의 가장 끝 (모든 per-epoch append 완료 후) 호출. checkpoint 저장을 이리로 이동 → history 항상 len==epoch.
- **[run_base_experiments.py, gitignored runner] eval-before-checkpoint 불변식**: `_run_bg_all` 을 `[join → eval 실행 → 기록 → checkpoint fold-in+save → best 복사]` 로 재배치. "checkpoint_N 존재 ⟺ ep N 까지 eval 기록 완료". result queue 폐기.
- **CPU-clone 스냅샷** (`_clone_state_to_cpu`/`_clone_optim_state`): live param 공유 race 제거.
- **strict resume normalization** (로드측): `epoch` 을 `1..N` 재구성 + per-epoch 키 len==ckpt_epoch 강제. `batch_profiling` 등 per-batch 키는 `_NON_PER_EPOCH` 로 제외.
- **[best_model_visualizer.py, gitignored] viz 방어**: stackplot 전 epoch 축과 ratio 배열 min-length 정렬.
- **완료 dataset gap backfill**: 유실 eval 의 `epoch_scores/epoch_NNN_scores.npz` 로 메트릭 재계산 → `epoch_metrics.json` 보충. npz 없는 진행중 dataset 은 깨끗한 재학습.

### Verification
`simulation` full pipeline kill@ep13 → resume → finish ep14: `epoch=[1..14]` 연속(skip/dup 0), per-epoch 43키 전부 len 14, `epoch_metrics evals=[5,10,14]`(누락 0), `batch_profiling=9` 보존. 상세: [docs/POST_MORTEMS/2026-05-30_resume_record_consistency.md](POST_MORTEMS/2026-05-30_resume_record_consistency.md).

### Note
핵심 수정 2 파일 (`scripts/run_base_experiments.py`, `mae_anomaly/visualization/best_model_visualizer.py`) 은 `.gitignore` (`scripts/run_*.py`, `mae_anomaly/visualization/`) 로 추적 제외 → 디스크에서 활성이나 commit 대상 아님. tracked 변경은 `trainer.py` + 본 문서 + post-mortem.

## 2026-05-29: Bg-worker CPU throttle + epoch dashboard VUS-completeness fix

### Problem (관측됨)
1. SWaT 종료 후 spawn 되는 2개 bg-worker (full + excl22) 가 16-core 시스템에서 **합산 13-14 cores 잠식** → 다음 dataset (WaDi/A1) main process 의 dataloader 가 starve → batch speed **4-6x 저속**. GPU util 1-12% 로 idle 상태 지속.
2. SWaT epoch_dashboard.png 에 **VUS-PR / VUS-ROC 컬럼이 비어있음**. 사용자가 dashboard 를 본 시점에 VUS sweep 이 아직 안 끝난 상태였기 때문.

### Root causes (코드 단위)
1. **잘못된 affinity hardcode** [run_base_experiments.py:1523](scripts/run_base_experiments.py#L1523) (기존):
   ```python
   os.sched_setaffinity(0, set(range(16, 24)))  # 24+ cores 가정
   ```
   16-core 시스템에서 `range(16, 24)` 가 존재 안 함 → Exception 발생 → `except: pass` 로 silent 실패 → bg-worker 가 16 cores 전체 사용.
2. **OMP_NUM_THREADS env 가 import 뒤에 설정** [L1517-1521] → matplotlib import 시점에 numpy/OpenBLAS 가 이미 16-thread pool 캐시 → env 무시.
3. **VUS sweep max_workers=4 hardcoded** [L1987], **EXCL22 pool default 8** [L1742] — 2 bg-worker 병렬 + 각 내부 4-8 sub-worker × OpenBLAS 16-thread = 무제한.
4. **L2702 main process 가 dashboard 렌더 → bg-worker 가 재렌더** — 2 render 사이에 user 가 보면 VUS 없음.

### Fixes
- [run_base_experiments.py:1511-1546] `_cpu_eval_viz_worker` 함수 첫 라인에서 OMP/MKL/OPENBLAS/NUMEXPR `NUM_THREADS=2` 설정, **matplotlib import 이전**. 이후 `torch.set_num_threads(2)`.
- [L1538-1545] Dynamic affinity: `n_bg = max(2, cpu_count() // 4)` → 16c → 4 cores, 24c → 6 cores, 32c → 8 cores. 최소 2 cores 보장.
- [L1765] `TSMAE_EXCL22_WORKERS` default 8 → 2.
- [L2020] `TSMAE_VUS_WORKERS` env 신설, default 2 (was hardcoded 4).
- [L2032-2065] VUS sweep `try-except-finally` 로 변경: 성공/실패 어느 경우든 finally block 에서 dashboard 1회 렌더링. `with VUS` / `VUS missing` 로 상태 log.
- [L2755-2758] Main process 의 `plot_epoch_metrics()` 호출 제거. feature_stats viz 만 유지. dashboard 는 bg-worker 가 단일 source.

### Effect (16c 기준 예상)
| 항목 | 이전 | 이후 |
|---|---|---|
| bg-worker affinity | 미적용 (16/16) | **4 cores (12-15)** |
| bg-worker 1 CPU | ~7 cores | **~4 cores** (max_workers=2 × OMP=2) |
| bg-worker 2 CPU (SWaT excl22) | ~6 cores | **~4 cores** (둘이 같은 4 core pool 공유) |
| main spare during bg-worker | 1-2 cores | **~12 cores** |
| Next dataset (WaDi/A1) speed | 5-6x baseline | **baseline 도달 (15-25 s/ep)** |
| bg-worker 완료 time | ~25 min | ~50-60 min (2-3x ↑, main 영향 0) |
| Dashboard with VUS | TIMING race (재렌더 전 보면 빈 VUS) | **VUS sweep 완료 후 단일 렌더 → 항상 VUS 포함** |

### Verification
- syntax + import 검증 ✓ ([scripts/run_base_experiments.py](scripts/run_base_experiments.py) parses)
- Smoke test (sim 5 ep): bg-worker PID `affinity=[12-15]` 정확 ✓, "epoch_dashboard.png will be rendered by bg-worker after VUS sweep" 로그 정확 ✓
- 큐 재시작 (TS=20260529_054522): SWaT_A1A2 ep 4 @ 5.7 it/s 정상 ✓
- 백업: `.trash/0529/cpu_throttle_053048/run_base_experiments.py.original`

### Env override (사용자 자원에 따른 fine-tune)
```bash
export TSMAE_VUS_WORKERS=4         # 빠른 host 에서 sweep 가속
export TSMAE_EXCL22_WORKERS=4      # 동상
export OMP_NUM_THREADS=4           # bg-worker per-process OpenBLAS thread cap (default 2 in bg-worker)
```

---

## 2026-05-29: Forward-skip warmup optimization + log-line recon/dis explicit fields

### Forward-skip optimization (`model.py`, `trainer.py`, `loss.py`)
- 새 파라미터 `teacher_only: bool = False` 를 `SelfDistilledMAEMultivariate.forward()` 에 추가. True 일 때 student decoder + student output projection + GRL classifier + SCAD head 의 forward pass 를 통째로 skip. 기존 동작상 이 출력들은 `if not teacher_only` gate (loss.py:196, trainer.py:597/620/704) 안에서만 소비되어 warmup 중에는 backward 에 기여하지 않는 dead compute 였음.
- `trainer.py:517` 가 train step 마다 `teacher_only=(epoch < warmup)` 로 전달 → warmup 50% 구간에서 transformer forward 약 22% 절감, **dataset 당 ~3.5-5 분 / queue 23 entry × 4 dataset 기준 총 5-8 시간 절감 예상**.
- `loss.py:179-187`: `student_output is None` 시 student recon metric 을 0.0 sentinel 로 반환. 다른 loss 경로 (FM/GRL/SCAD/discrepancy) 는 모두 이미 `not teacher_only` gate 가 있어 변경 불필요.
- `model.py:1078-1086`: skip 분기에서 `self._student_hidden / _grl_cls_logits / _scad_z` 를 명시적으로 None 으로 클리어 → 다음 batch 에서 stale 값 읽힘 방지.
- **검증**: smoke test (sim, 5 epoch, warmup=2) 통과. ep 1,2 (warmup): `recon_s=0, dis=0` (skip 모드 동작). ep 3,4,5 (post-warmup): `recon_s, dis` 정상 측정. 모든 viz 및 VUS sweep 정상 생성, COMPLETE 라인 정상 emit.
- Backward compat: default False 이므로 evaluator (eval mode 시 student 정상 forward) / training_visualizer / 외부 caller 영향 없음. 현재 진행 중인 entry 는 import 시점 모듈 사용 → 영향 없음. 274 entry 부터 자동 적용.
- 백업: `.trash/0529/forward_skip_warmup_041531/`

### Log-line recon/dis explicit fields (`run_base_experiments.py`)
- per-epoch eval 라인과 dataset COMPLETE 요약 라인에 신규 3 필드 추가:
  - `recon_t` = `train_teacher_recon_normal` (teacher 단독 recon, normal 샘플 평균)
  - `recon_s` = `train_student_recon_normal` (student 단독 recon — warmup 중 0)
  - `dis` = `train_disc_loss` (output discrepancy — `s_loss` 와 동일 값, 명확한 이름)
- 기존 `t_loss / s_loss` 는 misleading legacy alias 로 backward-compat 유지 (각각 joint recon, discrepancy 의미). 의미 명확화 위해 신규 코드/문서/모니터에서는 `recon_t / recon_s / dis` 사용 권장.
- 백업: `.trash/0529/log_line_change_035408/`

### Parser hardening (`monitor_status.py`)
- `RE_EVAL_NEW` regex 에 nan/inf 대응 atom (`NUM`) + 신규 3 필드 optional capture 추가. 단일 nan 값으로 인한 전체 eval 라인 drop 방지.
- 매칭 실패 시 `parse_warnings` stderr 출력 — 빈칸의 원인을 추적 가능.

### `recon_snr` + Option B disc_snr annotation
- 로그 라인에 `recon_SNR` 신규 필드 추가 (per-epoch eval + COMPLETE summary). Teacher-only recon 의 anomaly/normal 분리도 — `(recon_anomaly_mean − recon_normal_mean) / (σ_a + σ_n + ε)`. `disc_snr` 의 student-teacher 쌍 (Cohen's-d 형).
- Pre-warmup 시 `disc_snr` 은 학습 안된 student 의 측정값이라 anomaly detection signal 로 해석 misleading (구체적으로: discrepancy = (teacher − random_student)² ≈ teacher_output² → normal 의 teacher output 크기에 좌우). 음수가 나오는 게 정상.
- Option B 적용: pre-warmup row 의 `disc_snr` 값 옆에 ⚠️ + 표 하단 footnote 1회 (값은 그대로 표기 — student-joining 전환점 (음수→양수) 추적 보존).
- `recon_snr` 은 teacher-only 분리도라 pre-warmup 에서도 의미 그대로.
- `monitor_status.py` regex 에 optional `recon_SNR=({NUM})` 추가, group renumbering, legacy 호환.
- SKILL.md/set_guideline.md comparison table 12 → 13 columns (diag 1 → 2).

### Documentation
- `set_guideline.md`: training loss 표를 "정확한 의미" 컬럼 포함 6 행 표 (legacy + 신규 + d_loss) 로 정정. 보고용 dataset 결과 테이블의 `t_loss / s_loss` 컬럼을 `recon_t / recon_s / dis` 로 교체.
- `.claude/skills/training-status/SKILL.md`: Family table 에 Training loss row 추가, Comparison table 컬럼 13개 (recon_snr + loss 3개), SWaT dual-eval 시 2-table format 명시, no-blank rule + 5종 합법 사례 절차적 enforcement, Option B disc_snr annotation 룰.

---

## 2026-05-29: FM-score consistency refactor + KEEP_CHECKPOINT_DATASETS + 3x3 epoch dashboard + auto test stride

대규모 refactor — 2026-05-28 식별된 FM-score 누락 버그의 근본 원인 (인라인 9곳 중복 + 데이터 컨테이너 silent key drop + Optional default 패턴) 을 전수 차단. 자세한 post-mortem 은 `docs/POST_MORTEMS/2026-05-29_fm_score_omission.md` 참조.

### Phase 1 — Single-source scoring (`mae_anomaly/scoring.py` 신설)
- 모든 anomaly score 공식 (adaptive, default, ratio_weighted) 을 단일 모듈 `mae_anomaly/scoring.py` 로 통합. 이전 코드에 인라인 9곳 ({evaluator, run_base_experiments × 3, visualization/base × 2, visualization/best_model_visualizer × 3}) 에 흩어져 있던 식들을 모두 `compute_adaptive_components` / `compute_adaptive_point_score` / `compute_score` 단일 함수 호출로 교체.
- `ADAPTIVE_SCORE_EPSILON = 1e-4` 로 epsilon 통일 (이전엔 1e-4 와 1e-8 혼용 → 미세 결과 drift).
- `resolve_score_weights(config)` 가 `eval_disc_weight`, `eval_fm_weight`, `fm_loss_weight`, `use_output_discrepancy`, `use_feature_matching` 의 default 해석을 한 곳에서 처리. Config dataclass 와 dict (bg-worker spawn 후) 모두 지원.
- doctest 10/10 통과 + `temp/0529/test_scoring_equivalence.py` 10/10 통과 (canonical path byte-equal).

### Phase 2 — `PatchScoresBundle` dataclass (`mae_anomaly/types.py` 신설)
- patch-level 추론 출력의 단일 typed container. `fm` 은 명시적 `Optional[ndarray]` 필드로, 호출자가 누락할 수 없음 (2026-05-28 FM 누락 버그의 구조적 차단).
- `from_eval_data(eval_data)` / `from_patch_scores_dict(ps)` 두 classmethod 로 기존 dict 흐름 통합. `Evaluator.set_precomputed_patch_scores` 는 이제 bundle 만 받음 (Optional kwarg 없음). bg-worker pickle 호환 검증 (26 KB roundtrip OK).

### Phase 3 — API hygiene (kw-only required)
- `Evaluator.evaluate(*, lite, also_excl22=False)`, `Evaluator.evaluate_by_score_type(score_type, *, lite)`, `compute_full_metric_set(..., *, lite)`, `compute_extra_metrics(..., *, skip_vus)` 모두 `lite` 등을 keyword-only required 로 변경. positional 전달이나 default fallback 으로 silent 동작 분기 차단.

### Phase 4 — `KEEP_CHECKPOINT_DATASETS`
- `SWaT_A1A2`, `WaDi_A1`, `WaDi_A2`, `PSM` 4개 base dataset 에 한해 학습 종료 후 `best_model.pt`, `best_checkpoint.pt`, `latest_checkpoint.pt` 자동 보존. SWaT 의 excl22 best 도 함께. 다른 dataset (simulation, SMD × 28, Exathlon × 6) 은 기존 delete-after-inference 유지하여 디스크 사용 제한. 환경변수 `KEEP_BEST_CKPT=1` 도 그대로 동작.

### Phase 5 — 3x3 epoch_dashboard + bg-worker VUS sweep
- `plot_epoch_metrics` 의 dashboard 가 2x3 → 3x3 으로 확장. 새 패널 3개: VUS-PR, VUS-ROC, Range-based F1 (Affiliation-F1 + R-based F1).
- VUS 는 per-epoch 평가에서 계산 비용이 커서 학습 중에는 lite=True 로 skip. 학습 종료 후 bg-worker 가 모든 저장된 `epoch_NNN_scores.npz` 에 대해 `ProcessPoolExecutor(max_workers=4)` 병렬로 VUS sweep 후 `epoch_metrics.json` 갱신 및 dashboard 재렌더.
- 예상 sweep 시간 (병렬 4): SWaT 17min, WaDi 2min, PSM 4min. 메인 프로세스 GPU 학습과 무관 (CPU only) → 다음 dataset 학습 즉시 시작 가능. 실패 시 zero-filled bottom row 로 폴백.

### Phase 6 — 추론 stride 자동화
- `Config.sliding_window_test_stride` default 가 21 → −1 (sentinel) 로 변경. `mae_anomaly/utils/experiment.py:resolve_test_stride(config)` 가 sentinel 을 받으면 `num_patches − 1` 로 해석. 양수 override 는 그대로. Set C (num_patches=50) 기준 자동값 49.
- 4 call sites (run_base_experiments × 2, viz × 2) 모두 helper 경유로 통일.

### 위험·검증
- 매 Phase 끝 byte-equal 검증 (Phase 1: 0.0 diff for canonical / NPZ-save / dict-config 경로; Phase 2: bundle pickle roundtrip OK; Phase 3: kw-only required TypeError 강제 동작 확인).
- 회귀 test `temp/0529/test_scoring_equivalence.py` 가 모든 Phase 후에도 10/10 통과 유지.
- 검증 학습 (sim 5 ep + SWaT 10 ep) 으로 end-to-end 동작 확인 후 산출 dir 삭제.

## 2026-05-26: Comparison boundary-safe predicate 를 MAE-strict 형태로 통일 (effective span = window + target)

**Predicate 표기 통일 (numerical 결과 0 영향)** — `<= end` 표기를 `< end_eff` (where `end_eff = i + seq_len + 1`) 로 변경. `b <= X` ⇔ `b < X + 1` 정수 동치에 의해 algebraically 같으나, 표기 형태가 MAE 의 `start < b < end` 와 글자 그대로 동일해짐. Effective span 을 `(window, target)` 결합 = `seq_len + 1` 로 정의하여 next-step target boundary leak 까지 자동 차단.

**핵심**: window 자체 길이 `seq_len` 은 유지, "boundary check 단위" 만 `seq_len + 1` 로 확장. MAE 가 reconstruction-only (target 개념 없음) 라 window-only check 면 충분한 것과 자연스럽게 호환 — MAE 의 `window_size = SEQ + 1` 설정 시 comparison 과 정확히 같은 start indices 산출 (검증됨).

**변경 method**:
- `create_train_windows_boundary_safe` (standard): `for i in range(0, n - seq_len, stride): end_eff = i + seq_len + 1; if any(i < b < end_eff for b in bs): continue`
- `create_windows_from_segments` (normalonly): 위 + anomaly skip `if pl[i:end_eff].sum() > 0: continue`

**3-axis 검증 (CPU-only, all PASS)**:

### (1) Byte-identical to `<= end` (이전 버전)
14 entries (8 standard + 6 normalonly) — shape + 모든 값 완전 일치:
```
Standard:    smap/msl/smap_simple/msl_simple/swat/psm/smd_simple/exathlon → byte_identical=True
NormalOnly:  smap/msl/smap_simple/msl_simple/psm/smd_simple → byte_identical=True
```

### (2) MAE SlidingWindowDataset (window_size = seq_len+1) 와 start indices 완전 일치
같은 boundary set 에 대해 comparison `i < b < i+seq_len+1` 와 MAE `start < b < start+window_size` (with window_size = SEQ+1) 이 같은 인덱스 produce:
```
smap         {}: cmp=345,105 mae=345,105 identical=True
msl          {}: cmp= 89,871 mae= 89,871 identical=True
smap_simple  {A-1}: cmp=7,000 mae=7,000 identical=True
msl_simple   {C-1}: cmp=3,090 mae=3,090 identical=True
swat         {}: cmp=719,759 mae=719,759 identical=True
psm          {}: cmp=176,201 mae=176,201 identical=True
smd_simple   {m-1-1}: cmp=42,518 mae=42,518 identical=True
exathlon     {app=1}: cmp=43,492 mae=43,492 identical=True
```

### (3) Next-step prediction target safety (모든 dataset)
어떤 accepted window 의 target index 도 boundary 위에 떨어지지 않음 + normalonly 에서는 anomaly point 위에도 떨어지지 않음:
```
Standard variant:    8 dataset 모두 next-step target violations = 0
NormalOnly variant:  6 dataset 모두 boundary_viol=0, anomaly_target_viol=0
```

### (4) Pattern A/B regression
`verify_pattern_ab.py`: **23/23 PASS**

**의의**:
- MAE 와 comparison 양쪽이 같은 predicate 형태 (`start < b < end`) 사용 — strict identical
- 단 effective span 정의가 다름: MAE = window only, comparison = window + target (target convention 반영)
- 결과: 같은 boundary skip set 산출 (검증됨), 모든 baseline type (reconstruction + next-step forecasting + simple) 안전
- Numerical 결과는 직전 `<= end` 버전과 byte-identical → 진행 중/완료된 실험 영향 0

**Backup**: `/home/ykio/notebooks/claude/.trash/0526/smap_msl_pattern_b/comparison/data/unified_loader.py.strict_mae.pre`.

---

## 2026-05-26: Comparison 측 boundary-safe 구현을 MAE-style single-pass 로 통일

**구현 통일** — `comparison/data/unified_loader.py` 의 boundary-safe 윈도우 생성 두 method 를 MAE 측 (`mae_anomaly.dataset_sliding.SlidingWindowDataset._extract_windows`) 와 동일한 single-pass-with-skip 알고리즘으로 재작성. 이전 segment-split (각 segment 내부 sliding) → MAE-style (전체 train 1 회 pass + in-loop skip) 로 변경.

**변경 method**:
1. `create_train_windows_boundary_safe` (standard variant) — `self.train_features[:train_end]` 위에 single global pass, `i < b <= i+seq_len` 이면 skip
2. `create_windows_from_segments` (normalonly variant) — `self.features[:original_train_length]` (anomaly 보존 view) 위에 single global pass, (a) boundary skip (`i < b <= end`) + (b) anomaly point skip (`point_labels[i:end+1].sum() > 0`) 두 조건 모두 통과 시 사용. 즉 anomaly + boundary 처리를 한 pass 에 통합.

**MAE 와 차이점 (의도적)**:
- `_epoch_offset` 없음 (deterministic, 사용자 명시) — MAE 는 epoch 마다 random shift
- Skip predicate 가 `start < b <= end` (MAE 는 `start < b < end`) — comparison 의 next-step target 이 boundary 의 첫 timestep 에 떨어지는 case 도 차단. MAE 는 target 개념이 없어 strict less-than 만으로 충분.
- target/label 표현: MAE 는 reconstruction (window 자체), comparison 은 next-step (`train_X[end]`) — baseline task 특성 유지

**그 외 부분 (stride 정렬, last timestep 처리, single-pass 형태)**: MAE 와 동일.

**검증 (CPU-only)**:
- **Byte-identical** 검증: 14개 entry (8 standard + 6 normalonly) 의 새 single-pass output 이 이전 segment-split output 과 **shape + 모든 값 완전 일치**. `verify_mae_style.py` 결과:
  ```
  Standard:  smap, msl, smap_simple, msl_simple, swat, psm, smd_simple, exathlon — 모두 identical=True
  NormalOnly: smap, msl, smap_simple, msl_simple, psm, smd_simple — 모두 diff=+0
  ```
- `verify_pattern_ab.py` **23/23 PASS** (Pattern A/B regression 0)

**의의**: 같은 boundary-safe set 을 두 다른 알고리즘 (single-pass vs segment-split) 으로 생성해도 stride=1 일 때 윈도우가 정확히 일치함을 증명. 이제 MAE 와 comparison 양쪽이 **알고리즘 형태도 동일** (epoch_offset, target convention 제외).

**Backup**: `/home/ykio/notebooks/claude/.trash/0526/smap_msl_pattern_b/comparison/data/unified_loader.py.mae_style.pre`.

---

## 2026-05-26: Boundary-safe sliding windows — 모든 multi-segment dataset standard variant 적용 (확장)

**안전성 수정 (전 dataset 확장)** — `data_info['run_boundaries']` 가 비지 않은 모든 dataset 의 comparison/baseline standard variant 에서 boundary cross sliding window 자동 차단. MAE 메인 파이프라인 (`SlidingWindowDataset`) 이 이미 enforce 하는 것을 baseline 측에도 동일 적용.

**문제**: 이전 동작에서는 `is_segment_aware=False` (standard variant) 인 모든 entry 가 `model._create_windows(train_X)` / `model.fit(train_X)` 로 boundary 무시 sliding 을 수행 → 시간적으로 비연속한 두 recording 의 timestep 이 한 윈도우에 들어감.

| 경계 종류 | 해당 dataset | 한 윈도우에 mixed 가능했던 것 |
|---|---|---|
| Cross-channel concat | `smap` / `msl` (Pattern A) | 다른 telemetry channel 의 timestep |
| Cross-recording | `swat_a1a2`, `wadi_14days_A1`/`A2`, `psm`, `smd_*`, `smap_simple`, `msl_simple` | 다른 recording 시점의 timestep |
| Cross-trace | `exathlon_app*` | 다른 Spark application 의 timestep |

**적용 범위 (모든 multi-segment dataset 의 standard variant)**:

| Entry | run_boundaries | Dropped windows (seq_len=100, stride=1) | drop % |
|---|---|---|---|
| `smap` (Pattern A) | 161 | 10,700 | 3.01% |
| `msl` (Pattern A) | 80 | 5,300 | 5.57% |
| `smap_{ch}` (Pattern B) × 54 | 1 each | ~100 each | 1.4% |
| `msl_{ch}` (Pattern B) × 27 | 1 each | ~100 each | 3.1% |
| `swat_a1a2` | 1 | 100 | 0.01% |
| `wadi_14days_A1`/`A2` | 1 | 100 | <0.01% |
| `psm` | 1 | 100 | 0.06% |
| `smd_*` × 28 | 1 each | 100 each | ~0.25% |
| `exathlon_app*` × 6 | trace 별 6-8개 | 600-800 each | 0.18-1.36% |
| `simulation` | 0 | 0 (변화 없음) | — |
| 모든 `*_normalonly` | — | 변화 없음 (이미 처리) | — |

**구현 (2 files, additive)**:
- `comparison/data/unified_loader.py`: `UnifiedLoader.get_boundary_train_segments()` + `create_train_windows_boundary_safe(seq_len, stride)` 헬퍼 추가 (dataset-agnostic) — anomaly 보존 + run_boundaries 기준 segment 분리.
- `comparison/run_baseline.py`: NEURAL/SOTA baseline dispatch 에서 `has_run_boundaries = bool(loader.data_info.get('run_boundaries'))` 만 체크. `is_segment_aware=False` + `has_run_boundaries=True` 이면 boundary-safe path 사용. simulation 만 변화 없음 (run_boundaries 없음).

**의도된 numerical 영향**:
- 기존 SWaT/WaDi/PSM/SMD/Exathlon baseline 결과와 numerical 차이 발생 — boundary 가로지르는 학습 sample 들이 제거됨. 작지만 (대부분 < 1.5%) 재현성 비교 시 알려져야 함.
- SMAP/MSL Pattern A 가 가장 큰 영향 (3-5.6% drop) — 가장 부자연스러운 cross-channel mixing 제거.

**MAE pipeline 영향**: 없음 (`SlidingWindowDataset` 은 이미 `run_boundaries` 인식).

**검증 (CPU-only)**:
- Pattern A/B regression: `verify_pattern_ab.py` **23/23 PASS**
- 10-entry dispatch matrix: simulation (run_boundaries 없음) 만 `uses_bsafe=False`, 나머지 모두 `True`
- 각 dataset 의 segment 분할 정확 (Exathlon trace 별, SMD/PSM/SWaT/WaDi orig↔test_front, SMAP 108 sub-blocks = 54 ch × 2, MSL 54 sub-blocks = 27 × 2)

**Backup**: `/home/ykio/notebooks/claude/.trash/0526/smap_msl_pattern_b/comparison/{unified_loader.py.cross_channel_fix.pre, run_baseline.py.cross_channel_fix.pre}`.

---

## 2026-05-26: NASA SMAP + MSL (Telemanom) — Pattern A (concat) + Pattern B (per-channel) 통합

**기능 추가** — Hundman et al. *"Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"* (KDD 2018, DOI [`10.1145/3219819.3219845`](https://doi.org/10.1145/3219819.3219845), arXiv [`1802.04431`](https://arxiv.org/abs/1802.04431)) Telemanom benchmark 의 SMAP (54 channels × 25 features) / MSL (27 channels × 55 features) 두 dataset 을 baseline comparison pipeline 에 통합.

**Source**: `https://s3-us-west-2.amazonaws.com/telemanom/data.zip` (현재 HTTP 403 → Wayback Machine 2022-10-16 snapshot 사용; Telemanom README 는 최근 Kaggle 미러 안내). Labels: `https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv`.

**2 가지 통합 패턴 (additive, 둘 다 사용 가능)**:
- **Pattern A** (`smap` / `smap_normalonly` / `msl` / `msl_normalonly` — 4 entries): 모든 channel 을 시간축으로 concat 한 single stream. `run_boundaries` 가 channel 경계 + intra-channel `orig_train↔test_front` junction 모두 등록 (SMAP 161, MSL 80). UnifiedLoader 가 모든 채널의 train portion 에 single per-feature minmax fit — Anomaly Transformer/TimesNet/DCdetector 가 사용하는 OmniAnomaly preprocessed mirror 의 묵시적 가정과 일치.
- **Pattern B** (`smap_{ch}` / `smap_{ch}_normalonly` × 54 + `msl_{ch}` / `msl_{ch}_normalonly` × 27 — 162 entries): 각 channel 을 독립으로 처리. UnifiedLoader 가 해당 channel 의 train portion 만으로 minmax fit — **per-channel scaler**, OmniAnomaly (KDD'19) + 원본 Telemanom 의 entity-level 처리 관례 + 우리 pipeline 의 SMD per-machine / Exathlon per-app 패턴과 일관. SMD `smd_{machine}` 패턴 그대로 (162 entries dynamic for-loop 생성).

**Split rule (둘 다 공통)**:
- 각 channel 별로 `train = orig_train.npy (all normal) + test_front_50%`, `test = test_back_50%` (PSM convention)
- 50% cut 은 anomaly region 밖으로 ±10 timestamps 이동 (SMD `_find_safe_cut_point` 재사용)
- 검증: boundary-straddling anomaly = **0건** (SMAP 54/54, MSL 27/27). MSL 4 channels (D-16, M-1, M-2, S-2) 에서 cut 이동 발생.

**SMAP P-2 channel duplicate** — CSV 에 P-2 가 2 번 등장 (anomaly_sequences `[[5350,6575]]` vs `[[5300,6420]]`). 처리 정책 **UNION** (`[5300, 6575]` 가 anomaly) — 가장 conservative. 다른 baseline 처리 4 variants: OmniAnomaly excludes, TranAD silent overwrite, QuoVadis MSL `P-2_` 만 remove, 우리 = UNION.

**변경 사항** (6 source files, ~280 LoC, additive only — 기존 dataset 동작 영향 0):
- `mae_anomaly/datasets/loaders.py`: 이전 작업 (`_load_smap_msl_combined` + `load_smap_combined` + `load_msl_combined` Pattern A) 위에 `SMAP_CHANNEL_NAMES` (54), `MSL_CHANNEL_NAMES` (27), `_load_smap_msl_simple_single`, `load_smap_simple(channel)`, `load_msl_simple(channel)` Pattern B 추가
- `mae_anomaly/datasets/__init__.py`: 6 신규 식별자 exports
- `comparison/data/unified_loader.py`: `channel` parameter + `'smap_simple'` / `'msl_simple'` dispatch
- `comparison/experiment_configs.py`: SMD pattern dynamic for-loop 으로 162 entries 자동 생성
- `set_guideline.md` + `docs/DATASET.md`: SMAP/MSL Pattern A+B 섹션 + 출처/citation/통계

**Verification (CPU-only, 23/23 PASS)**:
- Pattern A regression 5/5 (shape, train_end, run_boundaries, normalonly NormalSegments 모두 일치)
- Pattern B sanity 6/6 (shape, train_end, run_boundaries=[orig_train_len], MSL 55 feats, normalonly)
- Per-channel scaler vs concat scaler 차이 증명 (max |min_A - min_B| = 1.999)
- Boundary-safety: SMAP 0/54 violations, MSL 0/27 violations
- experiment_configs entries: 4 Pattern A + 108 SMAP Pattern B + 54 MSL Pattern B
- Smoke load: SMAP 54/54 + MSL 27/27
- GPU 미사용 (`torch.cuda.is_initialized() == False`)

**미통합 (의도된 scope 분리, 별도 작업)**:
- `mae_anomaly/datasets/loaders.py:DATASET_LOADERS` 등록 — MAE 학습 통합
- `scripts/run_base_experiments.py:DATASETS` 추가 — MAE 학습 통합
- `comparison/configs/baseline_queue_*.json` 에 162 entries — automation queue

**Notion 업데이트**:
- "Baseline Comparison" 페이지 (`32087856b2078112b500c81664181ee7`): 13 곳 (title, snapshot callout, §2 dataset 표, §6.4/6.5/6.6/6.7 citation/license/refs/ack, §8 changelog) additive update, page +14.6%
- "0. MAE" 페이지 (`31387856b20781cd8d4ed14df7f65470`): 4 곳 (§1.2 데이터셋 카운트 self-inconsistency 정정 + SMAP/MSL callout, §5.2.2 향후 확장 row, §5.4 References [12] Hundman 2018)

**작업 로그**:
- `/home/ykio/notebooks/claude/temp/msl_smap_pattern_ab_0526/10_FINAL_REPORT.md` (메인 보고서)
- `/home/ykio/notebooks/claude/temp/smap_dataset_integration_0526/FINAL_REPORT.md` (이전 Pattern A 작업)

**Backup**: `/home/ykio/notebooks/claude/.trash/0526/smap_msl_pattern_b/` (6 파일 SHA-256 검증).

**Independent review (critical-reviewer agent)**: **ACCEPT** (10/10 axes PASS, 8 critical issues 모두 처리, 5 non-blocking + 3 missing insights documented).

---

## 2026-05-23: SCAD (Supervised Contrastive Anomaly Discrimination) 통합

**기능 추가** — Student decoder hidden 위에 작용하는 새로운 supervised contrastive loss. GRL의 대체 옵션으로 추가 (`use_scad: bool = False` 기본 OFF).

**Motivation**: 현재 GRL의 5가지 문제 (adversarial cat-and-mouse / L_FM과 모순 / window-mode noise / domain adaptation 부적절 차용 / FM에 압도) 해결. **4가지 엄격한 디자인 요구사항** — Anomaly anchor only, 단방향 push, no normal-normal attraction, no anomaly-anomaly attraction — 을 모두 만족하는 free-energy log-sum-exp 형태.

**핵심 수식 (Form A)**: L_SCAD = (1/|P_a|) Σ_{i∈P_a} log Σ_{n∈P_n} exp(z_i·z_n/τ)

**Reference Lineage (IEEE style — 4 핵심 ancestor)**:
- [1] PASCL (Wang et al., ICML 2022 Oral) — Asymmetric anchor philosophy
- [2] COBRA (Mirzaei et al., ICLR 2025) — Spurious negative pairs counterproductive
- [3] DevNet (Pang et al., KDD 2019) — 정상→이상 push spirit
- [4] Energy OOD (Liu et al., NeurIPS 2020) — Free energy log-sum-exp 수학 origin

**변경 사항** (6 source files, +440 LoC, backward-compatible):
- `mae_anomaly/config.py`: 11개 SCAD config 필드 추가 (default 비활성)
- `mae_anomaly/model.py`: `ScadProjectionHead` 클래스 + 조건부 `self.scad_head` 인스턴스화 + forward에서 `self._scad_z` 저장
- `mae_anomaly/loss.py`: `compute_scad_loss()` (Form A + Form B), `SelfDistillationLoss.forward()`에 `scad_z` 인자 + SCAD branch + 6 scalar metrics + loss_tensors['scad_loss']
- `mae_anomaly/trainer.py`: validation 3 rules (mutual exclusion), 12 history keys, adaptive λ + sigmoid/linear/none ramp-up, total_loss 추가
- `scripts/run_base_experiments.py`: `cb_metrics` SCAD 12 metrics attach, `plot_epoch_scad()` 블록 8 → `epoch_scad.png` (1×4)
- `mae_anomaly/visualization/best_model_visualizer.py`: `plot_scad_contribution_trend()` → `SCAD_contribution_trend.png` (1×3)

**Metric 수집 (12 keys, GRL pattern mirroring)**:
`scad_loss`, `scad_n_anom`, `scad_n_norm`, `scad_z_separation`, `scad_z_anom_var`, `scad_z_norm_var`, `scad_lambda`, `scad_adaptive_lambda`, `scad_ramp`, `scad_effective_weight`, `scad_grad_norm`, `scad_main_grad_norm`

**On/Off Switch (backward compat)**:
- Default `use_scad=False` → 모든 SCAD 코드 경로 비활성, 기존 baseline 동작 그대로
- `use_scad=True, use_grl=False`: SCAD 활성 (GRL 비활성)
- `use_scad=True, use_grl=True`: ValueError (mutual exclusion)
- `use_scad=True, patch_level_loss=False`: ValueError

**검증 통과** (8 unit tests, 4 integration tests): Config default, Model use_scad=False/True, ScadProjectionHead forward (L2 norm = 1.0), compute_scad_loss Form A/B, edge cases (empty P_a / P_n), mutual exclusion validation 3 케이스, default Trainer history dict.

**파일 백업**: `./temp/0523/{config,model,loss,trainer,run_base_experiments,best_model_visualizer}.py`
**구현 plan**: `./temp/scad_implementation_plan.md`, `./temp/scad_code_implementation_plan_0523.md`
**Notion page**: SCAD page (16 sections, 42 ablation configs, IEEE-style references in § 17)

## 2026-05-22: RevIN (Reversible Instance Normalization) 통합

**기능 추가** — Group P의 Exp 303 `271_revin` 준비 (학습 미실행, 코드만 구현).

**Reference 검토**: PatchTST (ICLR'23), ModernTCN (ICLR'24), CATCH (NeurIPS'24), TimesNet, DCdetector (KDD'23). 4개 baseline 모두 동일한 시점 패턴 확인:
- Step 2 (loader): `StandardScaler.fit(train)` → `transform(train+test)` (우리 `_standardize_per_feature()`)
- Step 6 (model forward 첫 줄): `revin.normalize(x)` per-window
- Step 7 (encoder/decoder): 정규화 공간에서 작동
- Step 8 (output 직후): `revin.denormalize(output)` per-window

**변경 사항**:
- `config.py`: `use_revin`, `revin_affine`, `revin_eps`, `revin_visible_only` 4개 필드 추가 (default 비활성)
- `model.py`: `RevIN` 클래스 추가 (per-feature learnable γ/β, detached stats, optional visible-only)
- `model.py:SelfDistilledMAEMultivariate.__init__()`: `self.revin = RevIN(...)` 조건부 인스턴스화
- `model.py:SelfDistilledMAEMultivariate.forward()`: 입력 시 `revin.normalize()`, teacher/student output 직후 `revin.denormalize()`
- 영향 받지 않음: encoder, decoder, FM hidden discrepancy, GRL classifier 입력 (모두 정규화 공간에서 작동), loss/evaluator (denorm 후 원본 공간)

**검증 통과**:
- Roundtrip (normalize → denormalize): max error < 1e-6
- Baseline regression (`use_revin=False`): 동일 동작 보존
- Train backward: affine γ/β gradient flow OK
- Inference fallback: `visible_only=True` + `point_labels=None` → full window stats 사용
- Visible-only stats: anomaly 위치 제외 정확 (test mean 1.0 vs 50.5)

**참고**: Exp 303 사용 시 `normalize_mode='zscore'` 함께 적용 (271은 minmax → zscore로 변경 필수). RevIN은 raw 또는 zscore된 입력에 적용되는 표준이므로 minmax + RevIN 조합은 비추천.

**파일 백업**: `./.trash/0522/{model,config,dataset_sliding,trainer,loss,evaluator}.py.bak`
**구현 plan**: `./temp/revin_plan.md` (선행연구 시점 분석 + 시점별 구현 매핑)

## 2026-05-21: Dynamic d_model 후보 확장 (64, 96 추가) + seq_length 일관성 검증

### Summary

Set C dynamic d_model의 candidate list를 확장하고, `seq_length / patch_size / num_patches` 의 silent inconsistency를 차단하는 명시적 ValueError 검증을 도입.

### 핵심 변경

- **`D_MODEL_CANDIDATES`**: `[128, 192, 256, 384, 512]` → **`[64, 96, 128, 192, 256, 384, 512]`**
  - low-F 데이터셋 (simulation F=8) 에서 raw=patch_size×F 가 80일 때, 기존엔 d_model=128 (최소 후보) 으로 over-provisioned. 이제 raw에 더 가까운 후보 선택 가능.
  - 모든 후보는 nhead=8 의 배수 (64/8=8, 96/8=12).
  - Cap은 512로 동일 유지.
- **`make_config()` 일관성 검증**: 함수 종료 직전 다음 두 검사 추가, 위반 시 `ValueError` 발생:
  1. `seq_length % patch_size != 0` → "must be divisible"
  2. `seq_length != patch_size * num_patches` → "Inconsistent patch configuration"
- **`Trainer.__init__` validation rule #7** 추가 (defense-in-depth, 동일 두 검사).
- **`SelfDistilledMAEMultivariate.__init__`** 에 `assert config.seq_length % config.patch_size == 0` 추가 (model 직접 생성 경로 보호).

### Impact

#### patch_size=10 (Set C / #274 baseline)

| 데이터셋 | F | 기존 d_model | 신규 d_model | 변화 |
|----------|----|-----------|------------|------|
| **Simulation** | 8 | 128 | **96** | ⚠️ 28% ↓ (raw=80 ≤ 96) |
| Exathlon | 19 | 192 | 192 | — |
| PSM | 25 | 256 | 256 | — |
| SMD | 38 | 384 | 384 | — |
| SWaT | 51 | 512 | 512 | — (cap) |
| WaDi A1 | 123 | 512 | 512 | — (cap) |
| WaDi A2 | 127 | 512 | 512 | — (cap) |

→ **Simulation 한 데이터셋만 d_model 변화 (128 → 96)**. 기존 simulation 실험 결과와는 baseline 비교 불가하므로, 신규 baseline 으로 재학습 필요.

#### patch_size=5 (Set A, dynamic 미사용이므로 영향 없음)

Set A는 `d_model=128` 으로 fixed. dynamic 경로 안 탐. 직접 dynamic 사용 시:
- F=8 sim: 40 → 64 (이전 128)
- F=13~19: 64 또는 96 (이전 128)
- F≥20: 변화 없음

#### Silent break 방지

- 기존: `seq_length=500, patch_size=7` 같은 inconsistent override 시 `model.py:179` 의 `self.num_patches = 500 // 7 = 71` 으로 silent 절단. timestep 3개는 model에서 reshape 안되어 runtime error 가능.
- 신규: `make_config()` 단계에서 즉시 `ValueError` 발생. 명확한 진단.

### 변경 파일

- `mae_anomaly/utils/experiment.py` — D_MODEL_CANDIDATES 갱신, `make_config` 검증 추가, `resolve_dynamic_d_model` docstring 갱신
- `mae_anomaly/trainer.py` — validation rule #7 추가
- `mae_anomaly/model.py` — `__init__` 에 assert 추가
- `docs/ARCHITECTURE.md` — D_MODEL_CANDIDATES 명시 갱신
- `temp/test_d_model_extension.py` (신규) — 10개 regression test (전부 통과)
- `temp/d_model_extension_plan.md` (신규) — 작업 계획

### Verification

```
$ python temp/test_d_model_extension.py
Test 9: D_MODEL_CANDIDATES constant verification             ✓
Test 1: patch_size=10 expected values (Option B)             ✓
Test 2: patch_size=5 new candidate effect (low-F datasets)   ✓
Test 3: patch_size=20 (unaffected by new candidates)         ✓
Test 4: make_config raises on indivisible seq_length         ✓
Test 5: make_config raises on inconsistent num_patches       ✓
Test 6: default Config() passes validation                   ✓
Test 7: Set A/B/C presets pass                               ✓
Test 8: model.py assertion on direct instantiation           ✓
Test 10: cap value (= 512)                                   ✓
============================================================
ALL REGRESSION TESTS PASSED
```

---

## 2026-05-21: Q3 v7 — Beyond Inference Investigation (P23-P26)

### Summary

Q3 v6 P20의 19% inverted anomaly subtype의 본질을 4 hypothesis (label noise, reverse learning, feature absence, training contamination)로 정량 검증. 4 새 실험 + 4 새 modules 작성, alignment 버그 수정 후 corrected statistics 발견.

### 핵심 발견

- **H1 (label noise) + H3 (feature absence) 둘 다 73% datasets에서 STRONGLY SUPPORTED** — median magnitude ratio 0.045
- **Inverted anomalies의 27%는 channel adversarial mixing** (raw distinct but adaptive_combine fails)
- **P24 raw feature subset이 5 hard dataset에서 274 model을 압도** (smd_machine-3-7: +0.184 with single feature)
- **P25 anomaly type별 274 model 성능 매우 다름** (quasi_normal win_rate 30.8% vs noise_burst 78.3%)
- **P26 train quality vs detection performance 약한 양의 상관 (r=+0.46)** — train ↔ test shift가 클수록 detection 쉬움 (H2 hypothesis falsified)

### 변경 — Code (new modules)

- `mae_anomaly/scripts/q3_exploration/core/inverted_signal_analysis.py` — 286 LOC, H1-H4 utilities
- `mae_anomaly/scripts/q3_exploration/core/feature_attribution.py` — 178 LOC, per-feature importance
- `mae_anomaly/scripts/q3_exploration/core/synthetic_anomaly.py` — 171 LOC, synthetic anomaly injection
- `mae_anomaly/scripts/q3_exploration/core/training_audit.py` — 160 LOC, train data quality metrics

### 변경 — Experiments

- `mae_anomaly/scripts/q3_exploration/experiments/exp_P23_inverted_signal_investigation.py` — 4 hypothesis tests, alignment fix
- `mae_anomaly/scripts/q3_exploration/experiments/exp_P24_per_feature_importance.py` — per-feature subset comparison vs 274 model
- `mae_anomaly/scripts/q3_exploration/experiments/exp_P25_anomaly_type_response.py` — 5-type anomaly classification + per-channel response
- `mae_anomaly/scripts/q3_exploration/experiments/exp_P26_training_distribution_audit.py` — train quality vs PAK correlation

### Bug Fix

- P23 초기 실행에서 raw signals (full=train+test)과 ds.regions (test indices) alignment 잘못 → metrics 일부 왜곡
- 모든 4 experiments에서 `get_raw_signals(alias, ds)` helper로 align (`test_signals = signals[full_len - test_len:]`)
- P23 v2 corrected 결과는 H1/H3 강력 지지 (median ratio 0.045)

### 변경 — Docs

- `mae_anomaly/scripts/q3_exploration/RESULTS_v7.md` — comprehensive report (450+ lines)

## 2026-05-19: 신규 SOTA 7개 통합 (TFMAE/TimesNet/DCdetector/MEMTO/ModernTCN/CATCH/NPSR)

### Summary

비교 baseline을 15 → 22 standard baselines로 확장. 2023-2025 frontier TS-AD 논문 7개를 단일 배치로 통합 (`comparison/baselines/<model>/{model.py, wrapper.py, __init__.py}` + 3곳 등록).

### 새로 추가된 모델 (7개)

| Phase | 모델 | 학회 | Distinct objective |
|---|---|---|---|
| 1 | TFMAE | ICDE'24 | Dual temporal+frequency MAE w/ adversarial KL |
| 1 | NPSR | NeurIPS'23 | Point + induction MSE, nominality-conditioned score |
| 2 | TimesNet | ICLR'23 | FFT top-k period → 2D Inception conv recon |
| 2 | DCdetector | KDD'23 | Patch + in-patch dual attention, symmetric KL |
| 3 | MEMTO | NeurIPS'23 | Memory module w/ K-means init, 2-phase training |
| 3 | ModernTCN | ICLR'24 Spot | Large-kernel DW + dual ConvFFN mixers |
| 5 | CATCH | ICLR'25 | Channel-mask + freq recon + channel discovery |

### 변경 — Code

- `comparison/baselines/<model>/` — 7개 신규 디렉토리, 모두 (model.py + wrapper.py + __init__.py).
- `comparison/baselines/__init__.py` — 7개 `XxxBaseline` import + `__all__` 갱신.
- `comparison/baseline_common.py` — 7개 `HAS_XXX` import-guard, `BASELINE_MODELS` / `SOTA_MODELS` / `SOTA_AVAILABILITY` 등록, `MODEL_PRESETS['default']` 등록 (각 모델 단일 preset, 전 데이터셋 동일), `create_model()` dispatch `elif` 분기.
- `comparison/experiment_configs.py` — `STANDARD_BASELINES` 리스트에 7개 key 추가.

### 변경 — Docs

- `comparison/MODELS.md` — 헤더 "17 baseline models" → "22 baseline models", New SOTA 섹션 확장 (16–22). 모델별 description + configuration table + reference.
- `comparison/GUIDE.md` — 디렉토리 트리에 신규 7개 엔트리, 모델 분류 표 갱신 (17개 → 22개), 신규 7개 모델 description.

### 검증

End-to-end import + create_model + 모델 forward smoke-test (7개 모두 통과):

| 모델 | Params | 비고 |
|---|---|---|
| tfmae | 0.67M | |
| timesnet | 7.03M | |
| dcdetector | 0.87M | |
| memto | 5.28M | 2-phase (random→K-means init) |
| moderntcn | 0.10M | |
| catch | 0.43M | 구조적 재구성 |
| npsr | 6.35M | Performer fallback → MHA |

**Note**: CATCH는 paper architecture + integration-plan hyperparameters 기준 structural reconstruction. NPSR은 `performer-pytorch` 의존을 optional로 만들어 미설치 시 `nn.MultiheadAttention` fallback.

### Pending (별도 작업)

- 신규 7개 모델 학습 실험 (사용자 요청: "실행만 빼고 코드구현 다 해봐").
- TFMAE/TimesNet/DCdetector는 이전 세션에서 동일 패턴 통합, 본 배치에서 코드 검증만 추가.

---

## 2026-05-19: Baseline 정리 — THOC 제거 (16 → 15 standard baselines)

### Summary

`THOC` (Shen et al., NeurIPS 2020) baseline을 비교 명단에서 완전 제거. 코드/문서/결과/Notion 페이지 모든 흔적 삭제 후 `.trash/thoc/`로 백업.

### 사유

1. **공식 코드 부재**: 원저자 (HKUST) 코드 미공개. 사용 가능한 모든 구현이 community reproduction.
2. **Reproduction 신뢰도 미검증**: 본 codebase가 사용한 [carrtesy/THOC-Pytorch](https://github.com/carrtesy/THOC-Pytorch) (Dongmin Kim, KAIST)는 README에서 "Unofficial Implementation" 명시 + SWaT/MSL/SMAP/NeurIPS-TS validation 표 빈 칸으로 reproduction 정확도 자체가 검증 안 됨.
3. **QuoVadisTAD 비교 명단에서도 제외**: 본 codebase의 baseline 셋의 기준이 되는 [QuoVadisTAD](https://arxiv.org/abs/2405.02678) (ICML 2024 Position Paper)도 THOC를 비교 SOTA에서 의도적으로 제외하고, PA(Point Adjustment) 평가 함정을 도입한 paper로만 인용.

### 변경 — Code

- `comparison/baselines/thoc/` (전체 디렉토리) → `.trash/thoc/code/baselines_thoc/`
- `comparison/baselines/__init__.py` — `THOCBaseline` import + `__all__` + docstring 제거
- `comparison/baseline_common.py` — HAS_THOC import-guard, BASELINE_MODELS/SOTA_MODELS/SOTA_AVAILABILITY 등록, MODEL_PRESETS hyperparameter dict, dispatch `elif model_name == 'thoc'` 분기 모두 제거
- `comparison/experiment_configs.py` — `STANDARD_BASELINES`에서 `'thoc'` 제거 + comment "16 baselines" → "15 baselines"
- `comparison/scripts/aggregate_exathlon.py` — BASELINE_MODELS list에서 'thoc' 제거

### 변경 — Docs

- `comparison/MODELS.md` — §16. THOC 섹션 전체 삭제, §17→§16 (TimesNet), §18→§17 (TFMAE) 번호 재정렬. 헤더 "18 baseline models" → "17 baseline models". "8 legacy SOTA" → "7 legacy SOTA"
- `comparison/GUIDE.md` — 디렉토리 트리 thoc 엔트리 제거, "SOTA (legacy) 8 → 7", `seq_len` 100 모델 목록에서 thoc 제거, 특이 모델 설명 thoc 제거, Notion 페이지 제목 "16 Models → 15 Models"
- `docs/CHANGELOG.md` — 이전 "2026-05-19 THOC docstring fix" 항목 제거 (THOC 제거로 무의미)

### 변경 — Results

- Q1 (`1_…_baseline_minmax`), Q3 (`3_…_baseline_minmax_normalonly`) 각 데이터셋 디렉토리 안 `thoc/` 서브폴더 (총 80개) → `.trash/thoc/results/` 로 이동, 원본 디렉토리 구조 보존

### 변경 — Notion

- 페이지 [Baseline Comparison](https://www.notion.so/Baseline-Comparison-16-Models-6-Datasets-4-Conditions-32087856b2078112b500c81664181ee7): THOC 행 + 관련 분석 내용 제거, "16 Models" → "15 Models" 카운트 갱신

### 백업 위치

- `.trash/thoc/code/baselines_thoc/` — THOC 구현 코드 (model.py, __init__.py)
- `.trash/thoc/code/original_files/` — 수정 대상 7개 파일의 수정 전 원본
- `.trash/thoc/plan_temp_originals/` — plan/temp markdown 문서 12개 수정 전 원본
- `.trash/thoc/results/` — 80개 thoc 결과 서브폴더

### 영향

- 향후 `--model all` 호출 시 자동으로 17개 모델 (Standard 15 + 2026-05-19 batch 2)만 실행
- THOC 관련 분석/그래프/순위 비교는 모두 무효화. 기존 보고서는 .trash 백업 참조

---

## 2026-05-19: Add 2 new SOTA baselines — TFMAE (ICDE'24) + TimesNet (ICLR'23) (Phase 1+2 of 10)

### Summary

신규 SOTA baseline 통합 작업 (총 10개 계획)의 Phase 1+2 완료. TFMAE (사용자 MAE 직접 경쟁모델, ICDE 2024)와 TimesNet (1티어 SOTA, ICLR 2023) 두 모델을 `comparison/baselines/`에 통합. 각 모델은 공식 repo에서 vendoring (단일 model.py)되었으며, 기존 baseline (anomaly_transformer 등)과 동일한 wrapper 패턴 (fit/predict/epoch_callback) 준수.

### 통합된 모델

**1. TFMAE** (Temporal-Frequency Masked Autoencoders, ICDE 2024)
- 출처: [LMissher/TFMAE](https://github.com/LMissher/TFMAE) (MIT License)
- 카테고리: 이중 MAE (temporal + frequency) + contrastive + adversarial KL
- 사용자 self-distilled MAE의 직접 경쟁 모델 → 비교 가치 최대
- Hyperparams: win_size=100, d_model=128, e_layers=3, lr=1e-4, epochs=10, batch_size=64

**2. TimesNet** (Temporal 2D-Variation Modeling, ICLR 2023)
- 출처: [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) (MIT License)
- 카테고리: FFT period detect + 2D Inception conv + reconstruction
- 종합 SOTA, 인용수 ~2,047 (Semantic Scholar, 2026-05)
- Hyperparams: win_size=100, d_model=64, e_layers=3, top_k=3, lr=1e-4, epochs=10, batch_size=128

### 변경 파일

**신규 생성 (gitignored 디렉토리, 로컬 변경)**:
- `comparison/baselines/tfmae/{__init__.py, model.py, wrapper.py}`
- `comparison/baselines/timesnet/{__init__.py, model.py, wrapper.py}`

**기존 수정**:
- `comparison/baselines/__init__.py`: 2 import + `__all__`
- `comparison/baseline_common.py`: 2 try/except + HAS_TFMAE/HAS_TIMESNET + BASELINE_MODELS/SOTA_MODELS/SOTA_AVAILABILITY + MODEL_PRESETS + `create_model` dispatch
- `comparison/experiment_configs.py`: `STANDARD_BASELINES` 리스트에 2 key 추가

**백업**: 모든 수정 파일의 원본은 `./.trash/260519/comparison/`에 보존.

### 검증

End-to-end import + 1 epoch fit + predict (tiny dummy data, T=500/200, D=8) 양 모델 통과:
- TFMAE: scores shape=(200,), dtype=float32, range=[0.000, 1.000] (softmax 정규화)
- TimesNet: scores shape=(200,), dtype=float32, range=[0.395, 3.068] (raw MSE)

### 다음 단계

Phase 1+2 외 5개 모델 (NPSR, DCdetector, MEMTO, ModernTCN, CATCH)은 별도 세션에서 진행. NPSR은 `performer-pytorch` 패키지 설치 필요 (blocker, 사용자 승인 대기).

Q1/Q3 전체 데이터셋 실행 계획은 별도 Notion subpage에 상세 작성 (실행 정책상 현재 미수행).

상세 계획: `/plan/SOTA_BASELINE_10_INTEGRATION_PLAN.md` + `/plan/SOTA_BASELINE_CHECKLIST.md` + [Notion subpage](https://www.notion.so/36487856b20781a29441e1ddf95900a0).

---

## 2026-05-18: Exathlon dataset MAE-side integration (base_experiments registry + 6 apps default)

### Summary

Exathlon 데이터셋을 MAE base experiments 파이프라인에 등록. Comparison 통합(같은 날 오전)에 이어 MAE 학습/평가 파이프라인도 6 apps × per-app evaluation으로 지원. PSM 통합 패턴과 동일한 retrofit 학습 계획 수립 (60-config × 6 apps).

### 주요 변경

**`scripts/run_base_experiments.py`**:
- `EXATHLON_APP_IDS` import 추가
- `EXATHLON_DATASETS` 동적 entry 추가 (6 apps × 1 entry each)
  - `key='exathlon_app{N}', loader='exathlon_app{N}', train_stride=21, normal50=False, results_subdir='Exathlon/app{N}'`
- `all_datasets = DATASETS + SMD_DATASETS + EXATHLON_DATASETS` (33 → 39 datasets)
- `aggregate_exathlon_results(experiment_dir)` 함수 추가 (SMD pattern mirror)
  - 6 apps의 epoch_metrics.json 읽어 best epoch 선정 (pak_auc_f1 기준)
  - Per-app + AVERAGE 행 → `Exathlon/results/results.csv`
- `--list` 출력에 39 datasets 표시

**문서 갱신**:
- `CLAUDE.md`: "5 base + 28 SMD + 6 Exathlon = 39 datasets"
- `set_guideline.md`: 데이터셋 표 + Exathlon 전용 서브섹션 (per-app statistics) + 결과 디렉토리 구조 갱신
- `ablation_guideline.md`: DATASET_TYPE 표에 Exathlon 6 entries 추가
- `docs/ABLATION_STUDIES.md`: DATASET_TYPE 코멘트에 `'exathlon_app1'` 추가

### 검증

- `python scripts/run_base_experiments.py --set C --list` 출력: 33-38번 줄에 `exathlon_app{1,2,4,5,6,9}` 6 entries 표시
- `Total: 5 base + 28 SMD + 6 Exathlon = 39` 정상 출력
- `aggregate_exathlon_results` import 성공

### 학습 예정 (Phase F — Q3 baseline 완료 후)

PSM과 동일하게 60-config retrofit:
- 대상: Exp 119-290 페이지 §4 PSM-target subset의 60 exps (140, 150, ..., 274, ..., 284)
- 우선순위: **274 first** (6 apps 모두) → 검증 후 → 나머지 59 exps × 6 apps
- 각 exp의 best_config.json 기반 `--config-override` 적용 (PSM 패턴)
- 결과: `results/experiments/{N}_*/Exathlon/app{1,2,4,5,6,9}/`
- 자동 aggregation: `aggregate_exathlon_results(exp_dir)`

---

## 2026-05-18: Exathlon dataset Comparison integration (raw loader + 19 FScustom features + per-app evaluation)

### Summary

Exathlon 데이터셋 (Jacob et al., VLDB 2021) 을 본 프로젝트 파이프라인에 통합. **TimeSeAD가 권장한 2개 dataset 중 하나**로, 실제 Apache Spark cluster trace 기반 설명 가능 이상 탐지 벤치마크. 6개 application으로 평가 (TimeSeAD 6-app convention), 각 app 별도 학습 후 평균.

### 주요 변경

**다운로드/전처리 (`dataset/Exathlon/`)**:
- `preprocess.py`: 93개 trace 자동 다운로드 + 19 FScustom features 추출 + binary label 생성
  - GitHub flat layout + nested split-zip layout 모두 처리 (7z multi-volume zip)
  - 원본 24.6 GB → 19 features로 축소 후 ~175 MB 저장
- 6 anomaly 종류: T1 bursty input, T2 bursty crash, T3 stalled, T4 CPU contention, T5 driver fail, T6 executor fail
- 라벨: RCI ∪ EEI (root cause + extended effect)

**`mae_anomaly/datasets/loaders.py`**:
- `load_exathlon(app)` 함수 추가
  - 입력: app ID ∈ {1, 2, 4, 5, 6, 9} (apps 7, 8은 구조적 결함으로 제외)
  - Train = all undisturbed + first `floor(N_dist/2)` disturbed (sorted by trace_id)
  - Test = remaining disturbed
  - `run_boundaries`로 trace 경계 보호
- `EXATHLON_APP_IDS = [1, 2, 4, 5, 6, 9]` 상수
- `DATASET_LOADERS`에 `exathlon_app{N}` × 6 키 추가

**`comparison/data/unified_loader.py`**:
- `dataset='exathlon', app=N` 분기 추가
- normalonly variant 지원 (Q3/Q4용)
- z-score/minmax 둘 다 호환

**`comparison/experiment_configs.py`**:
- 12개 config 등록: `exathlon_app{N}` + `exathlon_app{N}_normalonly` × 6 apps

**Queue 파일 추가 (`configs/`)**:
- `baseline_exathlon_minmax.json` (Q1)
- `baseline_exathlon_zscore.json` (Q2)
- `baseline_exathlon_minmax_normalonly.json` (Q3)
- `baseline_exathlon_zscore_normalonly.json` (Q4)

**문서**:
- `docs/DATASET.md`: Exathlon 섹션 추가 (스펙, 19 features 정의, app-level statistics, usage)

### 검증

- `load_exathlon(app=1)` smoke test 통과 (44K train, 46K test, 9 anomaly regions)
- 6개 앱 모두 정상 로드, train/test split 일관성 확인
- UnifiedLoader 4 conditions (Q1-Q4) 모두 정상 동작
- Random baseline × app1 × Q1 smoke test 통과 (PRC=0.129, PAK_F1=0.418)

### 평가 단위

각 app 별도 학습/평가 → 6 apps F1/PRC/AUC 평균 = Exathlon 종합 점수 (SMD per-machine pattern 동일).

---

## 2026-05-17: PSM dataset MAE-side integration (base_experiments registry + 60-model PSM run plan)

### Summary

PSM 데이터셋을 MAE base experiments 파이프라인에 등록. 60개 Top-RA 모델 (Exp 119-290 ablation 결과 기반) 대상으로 PSM 추가 학습 수행 예정. 결과는 각 기존 실험 디렉토리의 `PSM/` 서브디렉토리에 SMD/SWaT/WaDi와 동일 형식으로 저장.

### 주요 변경

**`scripts/run_base_experiments.py`**:
- DATASETS list에 PSM entry 추가 (WaDi_A2 직후)
  - `key='PSM', loader='PSM', train_stride=21, normal50=False, results_subdir='PSM'`
- Dynamic d_model: PSM(25 features) × patch_size=10 → d_model=256, dim_ff=1024
- **비활성 variant 정리**: `simulation_normal50`, `simulation_complex`, `simulation_complex_normal50`, `SWaT_A1A2_normal50`, `SWaT_A1A2_swap` 5개 entry를 default DATASETS에서 제외 (loader는 유지). 활성 데이터셋 = 5 base (simulation, SWaT_A1A2, WaDi_A1, WaDi_A2, PSM) + 28 SMD = **33 datasets**.
- Docstring 및 주석 갱신: "All 33 datasets" 등

**`comparison/add_mae_results.py`**:
- `MAE_SOURCE_DIRS["psm"] = "results/PSM"` 매핑 추가

**문서 갱신**:
- `CLAUDE.md`: "Run base experiments (5 base + 28 SMD = 33 datasets: simulation, SWaT, WaDi A1/A2, PSM, SMD ×28)"로 변경
- `set_guideline.md`: 데이터셋 표 재구성 (5 active datasets + 28 SMD), 비활성 variant 안내 추가, PSM 전용 서브섹션 추가
- `ablation_guideline.md`: Dataset Types 표에 PSM 추가
- `docs/ABLATION_STUDIES.md`: DATASET_TYPE 옵션 코멘트에 PSM 추가

### 검증

- Dry-run (`--dataset PSM --set C --config-override num_epochs=1`) 무에러 완료
- 생성 결과: `PSM/{epoch_metrics.json, best_config.json, training_histories.json, best_epoch_train_scores.npz, anomaly_type_metrics.json, batch_profiling.json, experiment_metadata.json, visualization/}` — SMD/SWaT와 동일 구조
- `epoch_metrics.json` 첫 entry 159 keys, `pak_auc_f1=0.585`, `pak_auc_prc_auc=0.564`
- `best_config.json`: `num_features=25`, `sliding_window_train_ratio=0.8007`, `d_model=256`, `patch_size=10`

### 백업

`/.trash/0517/` — 변경 전 8개 파일 (scripts/, comparison/, docs/, mae_anomaly/utils/) 백업

---

## 2026-05-15: Add PSM dataset integration (Comparison pipeline)

### Summary

PSM (Pooled Server Metrics, eBay) 데이터셋을 파이프라인에 추가. 우선 Comparison 파이프라인에 통합 (MAE base experiments 통합은 별도 작업 예정 — `temp/PSM_MAE_integration_plan.md` 참조). SMD/SWaT와 동일한 50/50 split 패턴, return 시그니처/문서 형식 완전 일관성 유지.

### 주요 변경

**`mae_anomaly/datasets/loaders.py`**:
- `load_psm()` 추가 — `load_smd_simple` 패턴 그대로 따름 (train = orig train + front 50% test, test = back 50% test)
- `DATASET_LOADERS['PSM'] = load_psm` 등록
- 25 features (`feature_0` ~ `feature_24`), 단일 연속 stream
- 4,195 NaN forward/backward-fill 처리, run_boundaries=[132481]

**`mae_anomaly/datasets/__init__.py`**:
- `load_psm` export 추가

**`comparison/data/unified_loader.py`**:
- `_call_raw_loader()` 에 `'psm'` dispatch 추가
- 에러 메시지 Available 목록에 `psm` 추가

**`comparison/experiment_configs.py`**:
- `'psm'` (standard) + `'psm_normalonly'` (normalonly variant) 엔트리 추가
- `results_dir_name: 'PSM'`, `model_preset: 'default'`, `all_models_list: STANDARD_BASELINES`

**문서**:
- `docs/DATASET.md`: DATASET_LOADERS 표 + PSM 별도 섹션 추가 (Abdulaal et al. KDD 2021, eBay)
- `comparison/GUIDE.md`: Section 3 "데이터셋 (5개 → 6개)" + PSM 처리 방식 상세
- `README.md`: line 22 dataset loader 목록에 SMD, PSM, TEP 추가
- `temp/PSM_integration_plan.md`: 통합 실행계획 (이번 작업의 source-of-truth)
- `temp/PSM_MAE_integration_plan.md`: 향후 MAE-side 통합 계획

**데이터**:
- `dataset/PSM/{train.csv, test.csv, test_label.csv, LICENSE}` 배치 (출처: github.com/eBay/RANSynCoders, BSD-3-Clause)
- shape: train (132,481×25 normal) + test (87,841×25, 27.76% anomaly)
- 50/50 split 후: train 176,401 (6.20% anom) / test 43,921 (30.63% anom)

### 미적용 (별도 작업)

- `scripts/run_base_experiments.py` DATASETS 엔트리 추가 (MAE-side 통합)
- `comparison/add_mae_results.py` MAE_SOURCE_DIRS 매핑 (MAE 결과 생성 후)
- `CLAUDE.md`, `set_guideline.md`, `ablation_guideline.md`, `docs/ABLATION_STUDIES.md` 문서 업데이트 (MAE 통합 시 일괄)

## 2026-03-05 (Update 69): Unified Post-Training Inference — 3-pass → 1-pass

### Summary

학습 후 GPU inference를 단일 evaluator pass로 통합. 기존 3개 독립 inference (evaluator patch_scores 37s + collect_predictions 84s + collect_detailed_data 84s ≈ 205s) → 1개 pass (~40s). AMP + 최적 batch_size 자동 적용.

### 주요 변경

**`mae_anomaly/evaluator.py`**:
- `_compute_patch_scores_all_patches(collect_detail=False)` — `collect_detail=True` 시 reconstruction 텐서 (feature 0) 및 timestep-level discrepancy를 기존 forward pass 중 수집하여 `self.detail_results`에 저장

**`mae_anomaly/visualization/base.py`**:
- `derive_pred_data()` 추가 — evaluator의 (N, num_patches) 출력을 BestModelVisualizer가 기대하는 pred_data dict로 변환 (pure numpy, GPU 불필요)
- `collect_predictions()`, `collect_detailed_data()`, `collect_all_visualization_data()` 삭제

**`scripts/run_base_experiments.py`**:
- 3-pass inference → 단일 `evaluator._compute_patch_scores_all_patches(collect_detail=True)` + `derive_pred_data()` 호출로 교체
- timing 키 단순화: `patch_scores_time`/`viz_collect_time` → `inference_time`

**삭제:**
- `scripts/run_reinference.py` (독립 재추론 스크립트, 통합으로 불필요)

## 2026-03-05 (Update 68): Pipeline Unification — Baseline ↔ MAE 완전 통합

### Summary

Baseline comparison 파이프라인을 MAE와 완전히 통합. 데이터 로딩, 전처리, 지표 계산 모두 MAE 코드를 직접 import하여 사용. 결과 형식도 MAE와 동일 (`epoch_metrics.json`, `scores.npz`).

### 주요 변경

**새로 작성:**
- `comparison/data/unified_loader.py`: MAE raw loaders + z-score를 직접 호출하는 단일 `UnifiedLoader` 클래스
- `comparison/experiment_configs.py`: 22개 → 11개 실험으로 정리 (swap/normal50 제거)
- `comparison/baseline_common.py`: MAE evaluator 함수 직접 사용, `epoch_metrics.json`/`scores.npz` MAE 형식 저장
- `comparison/run_baseline.py`: DL baseline epoch-level scoring + async CPU eval (ThreadPoolExecutor)

**삭제:**
- `comparison/data/` 14개 개별 로더 파일 (wadi_loader, swat_loader, simulation_loader 등)
- `comparison/baselines/evaluator.py` (mae_anomaly/evaluator로 대체)
- `comparison/results/` 전체 (정규화 방식 변경으로 무효화)
- swap/normal50 실험 11개

**기타:**
- Preprocessed CSV 파일 6개에 `(deprecated)_` 접두사 추가
- `comparison/baselines/__init__.py`에서 evaluator import 제거
- `comparison/GUIDE.md` 재작성

### 결과 형식 (MAE 동일)

```
comparison/results/{experiment_name}/{model_name}/
├── metadata.json
├── scores.npz              # key: anomaly_score (float32)
├── epoch_metrics.json      # MAE 동일 키 (teacher_* = null)
├── epoch_scores/           # [DL only] epoch별 scores
└── model/                  # [DL only] 학습된 가중치
```

## 2026-03-05 (Update 67): SWaT Dual-Eval + Directory Refactoring

### Summary

Major refactoring of experiment output structure:
1. **Removed redundant timestamp subdirectory**: `{YYYYMMDD_HHMMSS}_default/` intermediate dir removed. Results now stored directly under `results_subdir/`.
2. **SWaT dual-eval**: For SWaT datasets (non-swap), results split into `_full/` and `_excl22/` directories with independent best-epoch selection and evaluation.
3. **Comparison baselines**: Same SWaT dual directory structure applied to `comparison/` baselines.

### Changes

**`scripts/run_base_experiments.py`:**
- Removed `{timestamp}_default` subdirectory creation — `exp_dir = results_dir` directly
- Added `is_swat_dual` detection: SWaT non-swap datasets → `_full` + `_excl22` dirs
- Dual best-epoch tracking: `_best_ckpt_score_excl22` + `best_checkpoint_excl22.pt`
- Epoch callback: computes `excl22_pak_auc_f1` via `compute_metrics_with_exclusion()`
- After training: saves `best_model.pt` to both `_full/` and `_excl22/` dirs
- Spawns 2 background CPU eval+viz workers for SWaT (full + excl22)
- `_cpu_eval_viz_worker()`: new `swat_eval_mode` parameter; excl22 uses excl22 metrics as primary
- Shared files (checkpoints, epoch_scores) symlinked from `_full` to `_excl22`

**`scripts/run_reinference.py`:**
- `find_dataset_dirs()`: supports both old (`_default` subdir) and new (flat) structures
- `DATASET_SUBDIR_TO_LOADER`: added `SWaT/A1A2_full` and `SWaT/A1A2_excl22` entries

**`scripts/run_all_base.py`:**
- Updated docstring directory path (removed `_default`)

**`comparison/baseline_common.py`:**
- `run_single_baseline()`: new `results_dir_excl22` parameter for dual directory output

**`comparison/run_baseline.py`:**
- SWaT experiments: `results_dir` → `{name}_full`, `results_dir_excl22` → `{name}_excl22`

**`set_guideline.md`:**
- Updated dataset table (SWaT Subdir column shows `_full + _excl22`)
- Updated directory structure section (removed `_default`, added SWaT dual structure)

## 2026-03-04 (Update 66): Visualization y-axis unification

### Summary

Unified y-axis scales across subplots that display the same type of values, enabling fair visual comparison. Also removed hardcoded `ylim(0, 1)` in data_visualizer that became incorrect after z-score migration.

### Changes

**`mae_anomaly/visualization/best_model_visualizer.py`:**
- `learning_curve.png`: Unified y-axis for Row 0 cols 0-1 (Teacher/Student Recon) and Row 1 cols 0-1 (Normal/Anomaly T-vs-S)
- `best_model_reconstruction.png`: Unified y-axis within each row (cols 0-1 signal), col 2 (discrepancy) across rows
- `best_model_detection_examples.png`: Unified y-axis across all 4 subplots (TP/TN/FP/FN)
- `case_study_gallery.png`: Unified y-axis for col 0 (time series) and col 1 (discrepancy) across rows
- `hardest_samples.png`: Unified y-axis for col 0 (time series) and col 1 (discrepancy) across rows

**`mae_anomaly/visualization/training_visualizer.py`:**
- `score_evolution.png`: Unified x and y axes across all epoch subplots
- `sample_trajectories.png`: Unified y-axis across both trajectory plots
- `metrics_evolution.png`: Unified y-axis across all 4 metric subplots
- `late_bloomer_analysis.png`: Unified y-axis for Row 0 cols 0-2 (Anomaly Score trajectories)
- `reconstruction_evolution.png`: Unified y-axis across epoch columns per sample row
- `late_bloomer_case_studies.png`: Unified y-axis across epoch columns per sample row

**`mae_anomaly/visualization/data_visualizer.py`:**
- `complexity_comparison.png`: Removed hardcoded `ylim(0, 1)` (incorrect after z-score), auto-unified across 4 subplots
- `complexity_vs_anomaly.png`: Removed hardcoded `ylim(0, 1)`, auto-unified per row

### Already correctly handled (no changes needed)
- `performance_by_anomaly_type.png`: All Detection Rate (%) subplots already share ylim(0, 110)
- `score_distribution_by_type.png`: Already uses `sharey=True`
- `best_model_score_contribution.png`: Row 3-4 already share computed y_max
- `best_model_score_contribution_trends.png`: Already shares y-axis via manual computation

---

## 2026-03-04 (Update 65): Z-score standardization & data leakage fix

### Summary

Replaced min-max [0,1] normalization with per-feature z-score standardization (train-only fit). This follows the standard practice in time series anomaly detection literature (Anomaly Transformer, TimesNet) and fixes data leakage where test data statistics were leaking into normalization.

### Changes

**`mae_anomaly/dataset_sliding.py`:**
- Removed `_normalize_per_feature()` function (min-max [0,1])
- Added `_standardize_per_feature(signals, train_end)`: z-score fitted on train portion only
- `SlidingWindowDataset.__init__()`: now applies z-score normalization before train/test split
- Scaler statistics stored as `self.scaler_mean`, `self.scaler_std` for reproducibility
- `SlidingWindowTimeSeriesGenerator.generate()`: removed normalization call (returns raw signals)
- `_generate_simple_normal_series()`: removed normalization call (returns raw signals)

**`mae_anomaly/datasets/loaders.py`:**
- Removed min-max normalization from 6 loader functions:
  - `load_swat_combined()`, `load_swat_combined_swap()`
  - `load_wadi_14days_combined()`
  - `load_tep()`, `load_smd()`, `load_smd_block_split()`
- All loaders now return raw (unnormalized) signals
- Normalization is handled uniformly by `SlidingWindowDataset`

**`mae_anomaly/model.py`:**
- Removed `torch.clamp(student_output, 0.0, 1.0)` from student decoder
- Student output is now unbounded, matching z-score normalized input range

**Documentation:**
- `docs/DATASET.md`: Updated "Data Normalization" section
- `docs/SMD_BLOCK_SPLIT.md`: Updated return value description
- `docs/TEP_EXPERIMENT_GUIDE.md`: Updated preprocessing pipeline description

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Z-score over min-max | Unbounded range matches linear output projection; anomalies naturally amplified |
| Train-only fit | Prevents data leakage; follows community standard |
| Normalization in SlidingWindowDataset | Single point of truth; automatic adaptation to swap experiments |
| Remove student clamp | [0,1] clamp incompatible with z-score's unbounded range |

### Impact

- **All existing experiment results are invalidated** — re-training required with new normalization
- Dynamic margin (`margin_type='dynamic'`) auto-adjusts to z-score scale — no config change needed
- Fixed margin values (hinge/softplus) may need re-tuning for z-score scale

## 2026-03-04 (Update 64): PA%K AUC F1 per-K threshold re-optimization (tadpak best_f1_w_pa)

### Summary

Implemented per-K threshold re-optimization for PA%K AUC F1 metric, following the original tadpak method (Kim et al., AAAI 2022). Previously, `pak_auc_f1` used a fixed pre-PA threshold across all K values. Now it sweeps thresholds per-K after PA%K segment adjustment to find the true optimal F1 at each K.

### Changes

**`mae_anomaly/evaluator.py`:**
- `compute_pa_k_auc()`: Complete rewrite — computes both best (per-K optimized) and raw (fixed threshold) variants
- Added `precision_recall_curve` to sklearn imports for PRC-based threshold candidates
- New return keys: `pak_auc_f1_raw`, `pak_auc_f1_t_raw`, `pak_auc_precision_raw`, `pak_auc_recall_raw`
- `pak_auc_f1` now represents best_f1_w_pa (per-K re-optimized, primary metric)
- Updated `pak_auc_keys` in `evaluate()` and `evaluate_by_score_type()` zero_results

**`mae_anomaly/config.py`:**
- Updated `best_epoch_metric` comments to document best vs raw semantics

**`scripts/run_base_experiments.py`:**
- Teacher metric propagation now includes raw variant keys
- `plot_epoch_metrics()` chart #3: Added PAK AUC F1 best vs raw comparison lines

**`mae_anomaly/visualization/best_model_visualizer.py`:**
- `plot_pa_k_auc_summary()` bar chart: Shows both best and raw F1/Precision/Recall
- K-sweep curve legends: F1/Precision/Recall show both best and raw AUC values

### Metric Naming Convention

| Key | Meaning | Threshold |
|-----|---------|-----------|
| `pak_auc_f1` | best_f1_w_pa (primary) | Per-K re-optimized after PA%K adjustment |
| `pak_auc_f1_raw` | raw_f1_w_pa (legacy) | Fixed pre-PA F1-optimal threshold |

### Baseline comparison (comparison/)
- `comparison/baselines/evaluator.py`: `compute_pak_auc()` rewritten with same best/raw logic
- `comparison/baseline_common.py`: `_empty_excl_metrics()` zero dict updated with raw keys
- `comparison/GUIDE.md`: results.json schema updated with raw keys
- **All 320 baseline models across 22 experiments recomputed** with `--force`
- `comparison/compute_pak_auc.py`, `compute_pak_auc_parallel.py`: 일회성 사후 재계산 스크립트 → `trash/0304/`로 이동 (baseline 실행 시 `baseline_common.py`가 자동으로 pak_auc 포함)

### Non-PA%K metrics unchanged
Point-level F1, precision, recall, ROC-AUC, PRC-AUC keep existing roc_curve-based threshold determination.

---

## 2026-03-04 (Update 63): Comprehensive documentation sync (Notion + project docs)

### Summary

Full documentation audit: compared every parameter in Notion page and local docs against `config.py` source of truth. Fixed all discrepancies and added discriminator content throughout.

### Changes

**Notion page (0 MAE 프로젝트 개요):**
- Section 1.3: Added Adversarial Discriminator to innovations list
- Section 3.1: Fixed `weight_decay` (1e-5→1e-3), added `margin_type='none'`, added discriminator params
- Section 3.5: Added `margin_type='none'` to loss types, added Adversarial Discriminator Loss subsection
- Section 3.6: Added D optimizer (TTUR) and discriminator training flow
- Section 4.1: Complete hyperparameter table rewrite — added 15+ missing params, fixed `weight_decay`, `shared_mask_token` default, removed invalid `point_aggregation_method`
- Section 4.2: Fixed Set B (`p10/d128` → `p20/d256/k5`), Set C suffix (`w500p10_linear_dynamic` → `w500p10e2t4d1_dynamic_linear`), added CNN Kernel column
- Section 5: Added adversarial learning paragraph to conclusion

**`docs/ARCHITECTURE.md`:**
- Fixed `dropout` (0.1→0.15) in encoder, teacher decoder, and student decoder sections
- Fixed `learning_rate` (2e-3→1e-3), `weight_decay` (1e-5→1e-3) in config table
- Fixed `shared_mask_token` default label (True→False)
- Added `margin_type='none'` to margin type options
- Added `adv_loss_weight` to discriminator params in config table

**`docs/ABLATION_STUDIES.md`:**
- Fixed encoder layers (1→2), teacher decoder layers (2→4), student decoder layers (2→1)
- Fixed `margin_type` default (hinge→dynamic), added 'none' option
- Fixed `force_mask_anomaly` default (False→True)
- Fixed `shared_mask_token` default (True→False)
- Fixed `learning_rate` (2e-3→1e-3) in example config

---

## 2026-03-04 (Update 62): Add margin_type='none' (no margin, unbounded discrepancy)

### Summary

Adds `margin_type='none'` option that removes the margin entirely from anomaly loss. Anomaly loss becomes `-discrepancy`, pushing discrepancy higher without any cap. No unnecessary computation (no normal stats, no margin comparison).

### Changes

**`mae_anomaly/loss.py`:**
- `_compute_anomaly_loss`: Added `'none'` branch returning `-discrepancy` (checked first to skip all margin logic)
- `_compute_patch_anomaly_loss`: Same `'none'` branch for patch-level mode

**`mae_anomaly/config.py`:**
- `margin_type` comment updated to include `'none'` option

---

## 2026-03-04 (Update 61): Adaptive λ Formula Fix & adv_loss_weight Parameter

### Summary

Fixes `compute_adaptive_lambda` to match Notion-recommended formula: uses sum of individual gradient norms (`||∇normal|| + ||∇anomaly||`) instead of norm of sum (`||∇(normal + anomaly)||`), preventing partial gradient cancellation. Adds `adv_loss_weight` config parameter to control discrepancy:adversarial ratio (e.g., 0.5, 0.2, 0.1).

### Changes

**`mae_anomaly/loss.py`:**
- `compute_adaptive_lambda`: Changed from 2 `autograd.grad` calls (norm of sum) to 3 separate calls (sum of individual norms)
- Formula: `λ = (||∇_w normal_loss|| + ||∇_w anomaly_loss||) / (||∇_w adv_loss|| + δ)`

**`mae_anomaly/config.py`:**
- Added `adv_loss_weight: float = 1.0` — multiplier for adversarial loss after adaptive λ

**`mae_anomaly/trainer.py`:**
- Adversarial loss application: `loss + adv_loss_weight * λ_adv * adv_loss` (was `loss + λ_adv * adv_loss`)

---

## 2026-03-04 (Update 60): Discriminator LR Schedule & Metrics Tracking

### Summary

Adds CosineAnnealingLR scheduler for discriminator optimizer (matching main model pattern), D metrics propagation to `epoch_metrics.json`, and D metrics visualization in epoch-wise plots.

### Changes

**`mae_anomaly/trainer.py`:**
- Added `CosineAnnealingLR` scheduler for D optimizer (`d_scheduler`), active from `disc_warmup_epochs` to end
- D scheduler stepped in `train()` loop after `disc_warmup_epochs`

**`scripts/run_base_experiments.py`:**
- `_run_cpu_eval()`: Attaches D metrics (`d_loss`, `d_real_acc`, `d_fake_acc`, `adv_loss`, `adaptive_lambda`) from trainer.history to epoch_metrics entries
- `plot_epoch_metrics()`: New 5th PNG `epoch_discriminator.png` with 3 subplots (D Loss & Accuracy, Adv Loss, Adaptive λ) when D metrics present

---

## 2026-03-04 (Update 59): Adversarial Discriminator for Student Decoder

### Summary

Optional adversarial discriminator to prevent student decoder's "noise strategy". When enabled (`use_discriminator=True`), a 1D CNN PatchDiscriminator with Spectral Normalization trains alongside the model using TTUR (Two Time-scale Update Rule). The discriminator learns to distinguish real (original) patches from fake (student-generated) patches, and the adversarial loss forces the student to produce structurally different (not just noisy) reconstructions. Adaptive λ (VQGAN-style gradient magnitude balancing) automatically scales the adversarial loss contribution. Default is disabled — existing behavior is 100% preserved.

### Changes

**`mae_anomaly/config.py`:**
- Added 6 discriminator parameters: `use_discriminator`, `d_grad_student_layers`, `disc_lr_ratio`, `adaptive_lambda`, `disc_warmup_epochs`, `disc_channels`

**`mae_anomaly/model.py`:**
- Added `PatchDiscriminator` class (1D CNN + Spectral Normalization, independent from `SelfDistilledMAEMultivariate`)

**`mae_anomaly/loss.py`:**
- Extended `SelfDistillationLoss.forward()` return to 3-tuple: `(total_loss, loss_dict, loss_tensors)`
- `loss_tensors` includes `anomaly_disc_forward` (forward-direction discrepancy, no margin reversal — for adaptive λ)
- Added 3 module-level functions: `compute_discriminator_loss`, `compute_student_adversarial_loss`, `compute_adaptive_lambda`

**`mae_anomaly/trainer.py`:**
- Added D creation, D optimizer (TTUR: 4× LR, β1=0), `_extract_patches` helper
- `train_epoch()` integrates D step: D trains on all masked patches → student adversarial loss on anomaly patches → adaptive λ → combined backward
- 5 new history keys: `train_d_loss`, `train_d_real_acc`, `train_d_fake_acc`, `train_adv_loss`, `train_adaptive_lambda`

**`scripts/run_base_experiments.py`:**
- Checkpoint saves `discriminator_state_dict` when discriminator is active

**`mae_anomaly/visualization/best_model_visualizer.py`:**
- `plot_learning_curve()` expands from 2×3 to 3×3 when D metrics exist (D Loss/Accuracy, Adv Loss, Adaptive λ)

---

## 2026-03-03 (Update 58): Baseline PAK_AUC + Merlin Removal

### Summary

1. **Baseline PAK_AUC**: Computed PA%K AUC (K=0..100, trapezoidal integration) for all 15 baseline models across 22 experiments (320 models total). Results saved to each experiment's `results.json` under `pak_auc` key. Key names match MAE evaluator output for cross-system consistency.
2. **SWaT excl_r21 PAK_AUC**: For SWaT experiments with `has_excl_r21=True`, computed PAK_AUC excluding Region #21 (the disproportionately large anomaly segment). Saved under `excl_r21.pak_auc`.
3. **Auto-compute in future runs**: Updated `baseline_common.py` so `compute_metrics_for_results_json()` automatically computes and includes PAK_AUC for all new baseline experiments.
4. **Merlin removal**: Deleted merlin baseline model entirely — code (`baselines/merlin/`), results (`results/WaDi_A1/merlin/`), and all references in docs/scripts.

### Changes

**`comparison/baselines/evaluator.py`:**
- Added `compute_pak_auc()` — sweeps K=0..100, computes PRC-AUC/ROC-AUC/F1/F1_T/Precision/Recall per K, integrates via trapezoidal rule. Returns 6 scalars with same key names as MAE evaluator.

**`comparison/baseline_common.py`:**
- Added `compute_pak_auc` import from evaluator
- Updated `compute_metrics_for_results_json()` to automatically compute and include `pak_auc`
- Updated `_empty_excl_metrics()` to include empty `pak_auc` dict
- Updated `print_status()` table: replaced PA20_PRC/PA80_PRC columns with PAK_PRC/PAK_F1

**`comparison/baselines/__init__.py`:**
- Added `compute_pak_auc` to imports and `__all__`
- Removed merlin baseline (import, docstring, `__all__` entry)

**`comparison/compute_pak_auc.py`** (NEW):
- One-shot script to compute PAK_AUC for all existing baseline results
- Iterates 22 experiment configs, loads scores.npy + labels, updates results.json
- Handles SWaT excl_r21 via existing loader infrastructure
- CLI: `--experiment`, `--dry-run`, `--force`

**`comparison/GUIDE.md`:**
- Added `pak_auc` section to results.json structure documentation
- Updated `baseline_common.py` description to mention PAK_AUC computation

**Merlin deletion:**
- Deleted `comparison/baselines/merlin/` directory
- Deleted `comparison/results/WaDi_A1/merlin/` directory
- Updated `comparison/MODELS.md` (removed Section 16 + reference #9)
- Updated `comparison/_deprecated/run_new_models.py`, `run_missing_models.py`

## 2026-03-03 (Update 57): PA%K Mean-Based Refactor + PA%K AUC + Checkpoint Strategy

### Summary

Major pipeline refactoring with 6 changes:
1. **PA%K voting → mean**: Removed all voting-based PA%K code. All metrics (including PA%K) now use mean-aggregated point-level scores, eliminating unnecessary computation and unifying the aggregation method.
2. **PA%K AUC metric**: Added PA%K AUC — sweep K=0,1,...,100 for 6 metrics (PRC-AUC, ROC-AUC, F1, F1_T, Precision, Recall), integrate via trapezoidal rule. 12 new scalars per experiment (6 adaptive + 6 teacher).
3. **Checkpoint strategy**: Changed from saving all epoch checkpoints to saving only `best_checkpoint.pt` (best PRC-AUC) + `latest_checkpoint.pt`. Added `epoch_scores/` with point-level score snapshots (npz) per eval interval.
4. **Experiment directory suffix**: Experiment directories 20+ now use dynamic suffix from actual config overrides (`w{seq}p{patch}e{enc}t{td}d{sd}[_dynamic][_linear][_k{val}]`). Renamed 18 existing directories (Exp 22-39).
5. **PA%K AUC visualization**: New `pa_k_auc_summary.png` showing K-sweep curves and Adaptive vs Teacher comparison. Updated PA%K visualizations to use mean-based scoring.
6. **Config cleanup**: Removed `point_aggregation_method` field from Config dataclass.

### Changes

**`mae_anomaly/evaluator.py`:**
- Deleted voting functions: `precompute_point_score_indices()`, `vectorized_voting_for_all_thresholds()`, `_compute_single_pa_k_roc()`, `_compute_voted_point_predictions()`, `_compute_pa_k_f1_at_threshold()`
- Deleted method: `Evaluator._get_point_score_indices()`
- Removed voting branches from `_aggregate_with_map()` and `aggregate_patch_scores_to_point_level()`
- Added: `compute_pa_k_metrics_from_mean_scores()` — F1/Precision/Recall/F1_T at single threshold with PA%K adjustment
- Added: `compute_pa_k_roc_prc_from_mean_scores()` — ROC-AUC/PRC-AUC via threshold sweep with PA%K adjustment
- Added: `compute_pa_k_auc()` — sweep K=0..100, integrate 6 metrics → 6 AUC scalars
- Updated: `evaluate()`, `evaluate_by_score_type()`, `get_performance_by_anomaly_type()` to use mean-based PA%K

**`mae_anomaly/config.py`:**
- Removed `point_aggregation_method: str = 'voting'` field

**`scripts/run_base_experiments.py`:**
- Added `make_dynamic_suffix()` for config-aware experiment directory naming
- Checkpoint strategy: `best_checkpoint.pt` + `latest_checkpoint.pt` (no more per-epoch checkpoints)
- Added epoch point-level score saving: `epoch_scores/epoch_{NNN}_scores.npz`
- Added PA%K AUC extraction for teacher metrics in epoch callbacks and full evaluation

**`mae_anomaly/visualization/best_model_visualizer.py`:**
- Replaced all voting-based imports with mean-based equivalents
- Rewrote `plot_performance_by_pa_k()` using mean aggregation
- Rewrote `plot_roc_curve_pa80_comparison()` using mean aggregation
- Updated `plot_performance_by_anomaly_type()` to use mean-based PA%K
- Added `plot_pa_k_auc_summary()` visualization

### New Output Keys

| Key | Description |
|-----|-------------|
| `pak_auc_prc_auc` | PA%K AUC of PRC-AUC (adaptive) |
| `pak_auc_roc_auc` | PA%K AUC of ROC-AUC (adaptive) |
| `pak_auc_f1` | PA%K AUC of F1 (adaptive) |
| `pak_auc_f1_t` | PA%K AUC of F1_T (adaptive) |
| `pak_auc_precision` | PA%K AUC of Precision (adaptive) |
| `pak_auc_recall` | PA%K AUC of Recall (adaptive) |
| `teacher_pak_auc_{metric}` | Same 6 metrics for teacher-only scoring |

### Directory Structure Change

```
checkpoints/
├── best_checkpoint.pt   (best PRC-AUC epoch)
└── latest_checkpoint.pt (last evaluated epoch)
epoch_scores/
└── epoch_{NNN}_scores.npz  (adaptive_score, teacher_recon_error, discrepancy_error)
```

---

## 2026-02-27 (Update 56): Best Epoch Model Selection + MAE-Aligned Training

### Summary

Two major changes:
1. **Best epoch model selection**: `best_model.pt`, `best_model_detailed.csv`, and all `visualization/best_model/` outputs now use the **best epoch by PRC-AUC** instead of the last training epoch. After training completes, the best epoch is identified from `epoch_metrics`, its checkpoint is loaded, and all subsequent evaluation/visualization uses that model.
2. **MAE-aligned training** (from Update 55b, applied to Exp 18-20): 8 modifications to match original MAE paper settings — Pre-Norm, GELU, eps=1e-6, mask token init, xavier init, Adam beta2=0.95, bias/norm WD separation, LR warmup+cosine annealing.

### Changes

**`scripts/run_base_experiments.py`:**
- After training, finds best epoch from `epoch_metrics_list` by max PRC-AUC
- Loads best epoch checkpoint from `checkpoints/epoch_{N:03d}.pt` and replaces model state
- `best_model.pt` now includes `best_epoch` and `best_prc_auc` fields
- All GPU inference (patch scores, viz data collection) uses best epoch model
- Summary output shows `best_epoch` alongside final PRC and F1
- `experiment_metadata.json` timing dict includes `best_epoch` and `best_prc_auc`

**`mae_anomaly/model.py`** (MAE-aligned):
- All TransformerEncoderLayer/DecoderLayer: `norm_first=True`, `activation='gelu'`, `layer_norm_eps=1e-6`
- All TransformerEncoder/Decoder: Final `LayerNorm(eps=1e-6)` added
- Mask token init: `torch.zeros` + `nn.init.normal_(std=0.02)` (was `torch.randn`, std=1.0)
- Weight init: `xavier_uniform_` for Linear, `constant_` for LayerNorm (via `self.apply(_init_weights)`)

**`mae_anomaly/trainer.py`** (MAE-aligned):
- AdamW `betas=(0.9, 0.95)` (was PyTorch default 0.999)
- Bias/Norm weight decay separation: `param.ndim <= 1` → `weight_decay=0.0`
- LR warmup: `LinearLR(start_factor=1e-4)` → `CosineAnnealingLR` via `SequentialLR`

**Documentation:**
- `set_guideline.md`: Updated `best_model.pt` and `best_model_detailed.csv` descriptions
- `docs/VISUALIZATIONS.md`: Updated best_model section description

## 2026-02-26 (Update 55): patch_batch_size OOM Fix for Large d_model

### Summary

Reduced `patch_batch_size` from 4 to 2 when `d_model >= 512` to prevent GPU OOM during contribution ratio computation on large test sets (SWaT/WaDi with Set C).

### Changes

**`mae_anomaly/evaluator.py`:**
- `patch_batch_size`: Changed from unconditional `min(num_patches, 4)` to `min(num_patches, 2 if d_model >= 512 else 4)`. Prevents GPU OOM on SWaT (10,689 test windows) and WaDi (4,091 test windows) when d_model=512.

## 2026-02-25 (Update 54): Set C — Dynamic d_model + Linear Embedding + Auto dim_feedforward

### Summary

Added Set C experiment preset with per-dataset dynamic d_model selection and linear patch embedding. `dim_feedforward` is now auto-computed as `4 × d_model` when not explicitly overridden in `make_config()`.

### Changes

**`mae_anomaly/utils/experiment.py`:**
- `resolve_dynamic_d_model(num_features, patch_size)` (NEW): Selects smallest d_model from `[128, 192, 256, 384, 512]` that is ≥ `patch_size × num_features`. Caps at 512.
- `D_MODEL_CANDIDATES` (NEW): Candidate list `[128, 192, 256, 384, 512]`.
- `make_config()`: Auto-computes `dim_feedforward = 4 × d_model` when `'dim_feedforward'` is not in overrides dict. Existing presets (Set A/B) that explicitly pass `dim_feedforward` are unaffected.

**`scripts/run_base_experiments.py`:**
- Set C preset: `patch_size=10, num_patches=50, d_model='dynamic', patchify_mode='linear'`. Other params same as Set B.
- Dynamic resolution: After data loading, if `d_model='dynamic'`, calls `resolve_dynamic_d_model()` with the dataset's actual `num_features` to determine d_model before `make_config()`.
- `--set` argparse: Added `'C'` to choices.

**`set_guideline.md`:**
- Config Presets table: Added Set C column.
- Added "Set C: Dynamic d_model 규칙" subsection.

**`docs/ARCHITECTURE.md`:**
- Default Configuration table: Updated d_model and dim_feedforward descriptions.
- Added "Dynamic d_model (Set C)" subsection.

## 2026-02-25 (Update 53): SMD K=6 Block Split Loader + Epoch Offset Train Augmentation

### Summary

Added SMD per-machine K=6 block split loader for balanced anomaly distribution (~50/50 train/test). Added epoch offset feature that shifts train sliding window start positions each epoch for better generalization with large strides.

### Changes

**`mae_anomaly/datasets/loaders.py`:**
- `load_smd_block_split(machine, k_blocks, parity, margin)` (NEW): Splits a single SMD machine's test file into K blocks with safe boundary snapping (±margin from anomaly regions). Alternates blocks between train/test by parity.
- `_find_safe_cut_point`, `_get_anomaly_regions_local` (NEW): Helpers for boundary placement.
- Registry: `smd_k6_{machine_id}` (parity=0) and `smd_k6_{machine_id}_swap` (parity=1) for all 28 machines.

**`mae_anomaly/dataset_sliding.py`:**
- `SlidingWindowDataset.set_epoch_offset(offset)` (NEW): Shifts window start positions by `offset % stride`, re-extracts window metadata. Train-only; test stays at offset=0.

**`mae_anomaly/config.py`:**
- `epoch_offset: bool = False` (NEW): When True, Trainer applies non-replacement random offsets from `[0, stride)` each epoch. Over `stride` epochs, all positions are covered exactly once.

**`mae_anomaly/trainer.py`:**
- `train()`: When `epoch_offset=True`, pops a random offset from a permutation pool of `[0, stride)` each epoch and calls `train_dataset.set_epoch_offset()`.

**`docs/SMD_BLOCK_SPLIT.md`** (NEW): Experiment guide for K=6 block split methodology.

## 2026-02-24 (Update 52): Point-Level Epoch Eval + Detailed Timing + Layer-Level Batch Profiling

### Summary

Replaced window-level epoch monitoring with point-level metrics. Added comprehensive timing measurement across all pipeline stages. Added first-N-batch per-component + per-layer profiling (replaces PyTorch Profiler) with batch 0 skipped (CUDA warmup distortion).

### Changes

**`mae_anomaly/model.py`:**
- `forward`: Added layer-level profiling support via `_profiling` attribute. When `_profiling=True`, inserts `cuda.synchronize()` between 5 architectural sections (embed_input, masking, encoder, teacher_decoder, student_decoder). Results stored in `_forward_timing` dict.

**`mae_anomaly/trainer.py`:**
- `train_epoch`: Added per-epoch timing (forward_approx, backward_approx, epoch_total) with CUDA sync at epoch boundaries only (~1% overhead)
- `train_epoch`: Added `profile_batches` param — batches 1..N of epoch 1 get per-component `cuda.synchronize()` timing (batch 0 skipped to avoid CUDA warmup distortion)
  - Batch level: data→GPU, model_forward, loss_compute, backward, optimizer_step
  - Layer level (inside model_forward): embed_input, masking, encoder, teacher_decoder, student_decoder
- `train`: Accepts `profile_n_batches`, passes to epoch 0 only. Stores results in `history['batch_profiling']`
- `_print_batch_profiling`: Prints hierarchical profiler-like table (with layer breakdown under Model Forward) immediately after epoch 1, with estimated remaining training time
- `train`: Records per-epoch timing for train_epoch, contrib_ratios, callback → `history['epoch_timings']`

**`scripts/run_base_experiments.py`:**
- `compute_epoch_test_metrics`: Returns inference_time vs eval_time breakdown in metrics dict
- `save_batch_profiling` (NEW): Formats per-batch timing into profiler-like summary table + JSON. Saves `batch_profiling.json` + `batch_profiling.txt`
- Removed `run_profiling` (PyTorch Profiler) — replaced with in-training batch profiling
- Epoch callback: Logs `(infer=Ns eval=Ns)` per eval, accumulates callback_total_time
- Training timing: Separates `pure_train_time` (excludes callback), `contrib_ratios_time`, `epoch_eval_time`
- Final inference: Separates `patch_scores_time` vs `viz_collect_time`
- timing dict: Expanded with all phase timings (wall_time, pure_train_time, epoch_eval_time, etc.)
- Removed dead code: `_cpu_epoch_pointlevel_worker`, `_merge_epoch_pointlevel`, `_epoch_pl_processes`
- `plot_epoch_metrics`: Rewritten for point-level (4 PNGs: prc_auc, f1_t, pa_k_f1, dashboard)

**`mae_anomaly/dataset_sliding.py`:**
- Fixed `train_end` alignment bug (removed stride-dependent boundary shift)

**`set_guideline.md`:**
- Updated epoch callback, epoch_metrics.json format, pipeline, visualization descriptions

## 2026-02-23 (Update 51): Fix force_mask_anomaly Non-Uniform Masking Bug (A-1)

### Summary

Fixed critical bug where `force_mask_anomaly` broke the uniform masking assumption required by `_encode_visible_only` (standard MAE encoder). The old implementation forced ALL anomaly patches to be masked regardless of masking budget, causing variable `num_keep` across the batch. This led to masked patches (including anomaly data) leaking into the encoder for some samples.

### Problem

When `force_mask_anomaly=True` and a sample had anomaly patches that didn't overlap with the random mask:
1. All anomaly patches were force-masked, increasing total masked count beyond `target_num_masked`
2. Different samples in a batch had different numbers of visible patches
3. `_encode_visible_only` used `num_keep` from sample 0, causing:
   - Samples with fewer visible patches: masked patches leaked into encoder (anomaly information leakage)
   - Samples with more visible patches: visible patches incorrectly excluded from encoder

### Fix

Replaced the old force-then-patch approach with **fixed-budget priority-based masking**:
- Masking budget is always exactly `round(num_patches * masking_ratio)` per sample
- Anomaly patches are prioritized for masking within this budget
- If anomaly patches exceed the budget, excess remain visible as encoder context
- Fully vectorized implementation (no per-sample loop) using priority sorting + scatter

### Changes

**`mae_anomaly/model.py`:**
- Rewrote `force_mask_anomaly` section in `forward()` with vectorized priority-based masking
- Added assertion in `_encode_visible_only` to catch non-uniform masking (safety check)

**`mae_anomaly/config.py`:**
- Updated `force_mask_anomaly` description to reflect new priority-based behavior

**`docs/ARCHITECTURE.md`:**
- Updated Force Mask Anomaly section with detailed behavior description

**`docs/TEP_EXPERIMENT_GUIDE.md`:**
- Updated force_mask_anomaly description

**`docs/ABLATION_STUDIES.md`:**
- Updated Force Mask Anomaly experiment description

## 2026-02-17 (Update 50): TEP Experiment Guide + save_dataset_info Fix

### Summary

Added comprehensive TEP experiment guidelines and config files. Fixed `save_dataset_info` bug where fault types >= 10 caused IndexError. Three config files cover quick test, single fault, and all-faults scenarios.

### Changes

**Fixed `scripts/ablation/run_ablation.py`:**
- `save_dataset_info`: Added `_atype_to_name()` helper to safely convert anomaly_type
  - Fault types 1-9: maps to simulation type names (backward compatible)
  - Fault types 10+: maps to `fault_N` (supports TEP fault types 10-20)
- Replaced hardcoded `SLIDING_ANOMALY_TYPE_NAMES` indexing with dynamic type set from `anomaly_regions`

**New `docs/TEP_EXPERIMENT_GUIDE.md`:**
- Part 1: Current model/experiment framework understanding
- Part 2: TEP dataset structure (960 samples/run, fault onset at sample 160, 20 faults)
- Part 3: Dataset comparison (SWaT vs WaDi vs TEP)
- Part 4: Recommended hyperparameters (seq_length=160, stride=5)
- Part 5: Three experiment scenarios (Quick/Single/All)
- Part 6: Execution instructions and result structure

**New config files (`scripts/ablation/configs/`):**
- `tep_quick_test.py`: 1 epoch, fault1 only, stride=11 test (pipeline verification)
- `tep_single_fault.py`: 50 epochs, configurable fault type, full PA%K evaluation
- `tep_all_faults.py`: 50 epochs, all 20 faults, full evaluation

### Key Design Decisions

- `seq_length=160`: aligns with fault onset period (samples 0-159 normal, 160-959 anomalous)
- `patch_size=8, num_patches=20`: efficient patch granularity for 160-sample windows
- `sliding_window_stride=5` (train): ~161 windows per 960-sample run
- `run_boundaries` handled automatically by loader (data_info['run_boundaries'])

---

## 2026-02-17 (Update 49): SMD (Server Machine Dataset) Loader

### Summary

Added SMD dataset loader for 28 server machines with full pipeline compatibility. Uses the same `run_boundaries` mechanism as TEP for handling independent machine boundaries.

### Changes

**New in `mae_anomaly/datasets/loaders.py`:**
- `load_smd()` function: loads all 28 machines or specific subset
- `SMD_MACHINE_NAMES`: list of all 28 machine IDs
- Registry entries: `smd` (all machines) + `smd_machine-X-Y` (28 individual loaders)

**Modified `mae_anomaly/datasets/__init__.py`:**
- Added `load_smd` and `SMD_MACHINE_NAMES` to exports

### Dataset Stats
- 28 machines, 38 features each (37 after constant removal)
- Total: 1,416,825 samples (708,405 train + 708,420 test)
- Test anomaly ratio: 4.16% (29,444 anomaly points)
- 327 anomaly regions across all machines
- train_ratio: 0.5 (train = all normal, test = with anomalies)

---

## 2026-02-15 (Update 48): TEP Dataset Support + Run Boundary Handling

### Summary

Added TEP (Tennessee Eastman Process) dataset loader with support for 20 fault types and independent simulation run handling. Introduced `run_boundaries` mechanism to prevent sliding windows from crossing independent run boundaries.

### Changes

**New in `mae_anomaly/datasets/loaders.py`:**
- `load_tep()` function: loads TEP RData files, supports per-fault-type selection
- `TEP_FAULT_NAMES`: descriptive names for 20 TEP fault types
- Registry entries: `tep` (all faults) and `tep_fault1` through `tep_fault20`

**Modified `mae_anomaly/dataset_sliding.py`:**
- `SlidingWindowDataset.__init__`: new optional `run_boundaries` parameter
- `_extract_windows()`: skips windows that cross run boundaries

**Modified `mae_anomaly/evaluator.py`:**
- Per-type evaluation now discovers anomaly types dynamically from data (supports >9 types)

**Modified `mae_anomaly/trainer.py`:**
- Per-type score computation now handles arbitrary anomaly type indices

**Modified `scripts/ablation/run_ablation.py`:**
- Extracts `run_boundaries` from `data_info` and passes through entire pipeline
- All `SlidingWindowDataset` and `NoisyLabelSlidingWindowDataset` calls updated

**Modified `mae_anomaly/datasets/noisy.py`:**
- `NoisyLabelSlidingWindowDataset`: passes `run_boundaries` to parent class

**Dependencies:**
- `pyreadr` required for loading TEP RData files

---

## 2026-02-15 (Update 47): Script Consolidation — Unified Config System

### Summary

Consolidated 10 separate run_*.py scripts (4,742 lines) into a unified config-based system with single entry point. This eliminates code duplication, reduces maintenance burden, and enables easy experiment reproduction via config files.

### Changes

**New Modules:**
- `mae_anomaly/datasets/loaders.py` - Centralized dataset loaders with registry pattern
- `mae_anomaly/datasets/noisy.py` - NoisyLabelSlidingWindowDataset
- `mae_anomaly/utils/system.py` - GPU memory utilities (free_gpu, mem_status)
- `mae_anomaly/utils/experiment.py` - Config creation helper (make_config)

**Updated Scripts:**
- `scripts/ablation/run_ablation.py` - Extended to support multiple dataset types via DATASET_TYPE config parameter
- `scripts/run_base_experiments.py` - Updated imports to use new modules

**Config System:**
```bash
# Single entry point for all datasets
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/<config>.py
```

**Dataset Types:**
- `simulation` - Generated time series (default)
- `swat_A1A2` - SWaT A1+A2 combined
- `swat_A1A2_swap` - SWaT with swapped halves
- `wadi_14days_A1` - WaDi 14 days + A1
- `wadi_14days_A2` - WaDi 14 days + A2
- `wadi_A2` - WaDi A2 only

**Template Configs:**
- `scripts/ablation/configs/simulation_test.py`
- `scripts/ablation/configs/swat_A1A2_test.py`
- `scripts/ablation/configs/wadi_14days_A1_test.py`
- `scripts/ablation/configs/README.md` - Migration guide

### Files Modified

- `mae_anomaly/datasets/` - New module (loaders.py, noisy.py, __init__.py)
- `mae_anomaly/utils/` - New module (system.py, experiment.py, __init__.py)
- `scripts/ablation/run_ablation.py` - Dataset type support (lines 1337-1354, 1386-1425, 1106-1125)
- `scripts/run_base_experiments.py` - Import updates (lines 42-52, 635)
- `scripts/ablation/configs/` - New config templates and README
- `ablation_guideline.md` - Section 7 added (Unified Config System)

### Archived Scripts

Moved to `.trash/20260215_run_scripts/` (10 files, 4,742 lines):
- run_mae_baseline.py, run_mae_normal50.py
- run_swat_ablation.py, run_swat_A1A2_swap.py, run_swat_A1A2_normal50.py
- run_wadi_ablation.py, run_wadi_14days_ablation.py, run_wadi_14days_normal50.py, run_wadi.py
- run_base_experiments.py (old version)

### Benefits

- ✅ Single codebase eliminates ~4,700 duplicate lines
- ✅ Easy experiment reproduction (share config file)
- ✅ Centralized bug fixes and improvements
- ✅ Type-safe config validation
- ✅ Backward compatible (DATASET_TYPE defaults to 'simulation')

---

## 2026-02-09 (Update 46): Default Parameter Update — enc2, td4, sd1

### Summary

Updated default model parameters based on WaDi 14days ablation study results. The new defaults (enc=2, td=4, sd=1, p=5, d=128) showed +30.7% PRC-AUC improvement on A1_14days and +13.5% on A2_14days compared to original training data.

### New Default Parameters

| Parameter | New Default | Previous | Reason |
|-----------|-------------|----------|--------|
| num_encoder_layers | **2** | 1 | enc=2 +21.7% PRC on A1_14days |
| num_teacher_decoder_layers | **4** | 2 | td=4 optimal for reconstruction |
| num_student_decoder_layers | **1** | 2 | sd=1 creates better discrepancy signal |
| patch_size | 5 | 5 | Maintained (fine-grained best) |
| d_model | 128 | 128 | Maintained |

### Files Modified

- `mae_anomaly/config.py` - Core default values
- `scripts/run_mae_baseline.py`, `run_mae_normal50.py` - Training scripts
- `scripts/run_wadi_ablation.py`, `run_wadi_14days_ablation.py` - WaDi scripts
- `docs/ARCHITECTURE.md` - Architecture documentation
- `docs/PROJECT_SUMMARY.md` - Project summary

### Key Insight

With 14days normal training data, enc=2 gains +21.7% PRC-AUC for A1, making deeper encoders beneficial when more normal patterns are available. The shallow student (sd=1) with deep teacher (td=4) creates optimal capacity gap for discrepancy-based anomaly detection.

---

## 2026-02-05 (Update 45): WaDi A2 Ablation Study Complete

### Summary

Completed 40 ablation experiments on WaDi A2 dataset (172,803 timesteps, 96 features, 7 attack segments). Best configuration: w100_p5_td3_sd1 (F1=0.5728, ROC-AUC=0.9396). Key finding: td3_sd1 outperforms td4_sd1 on A2 (unlike A1), suggesting moderate teacher depth generalizes better for heterogeneous attack types.

### Key Results

| Metric | Best Value | Configuration |
|--------|------------|---------------|
| F1 Score | 0.5728 | w100_p5_td3_sd1 |
| ROC-AUC | 0.9396 | w100_p5_td3_sd1 |
| PRC-AUC | 0.6146 | w500_p5_td3_sd1 |

### A1 vs A2 Comparison

| Parameter | A1 Optimal | A2 Optimal |
|-----------|------------|------------|
| Teacher Decoder | 4 layers | 3 layers |
| Best F1 | 0.6065 | 0.5728 |

### New Files

| File | Description |
|------|-------------|
| `docs/ablation/WaDi/A2_ANALYSIS.md` | Comprehensive analysis document |
| `results/WaDi/A2/ablation_results.csv` | All experiment metrics in CSV format |

---

## 2026-01-31 (Update 44): Fix Phase 2 Defaults — enc1, lr=2e-3

### Summary

Diagnostic testing revealed Phase 2 defaults (enc2, lr=5e-3) cause catastrophic performance degradation at w500. `num_encoder_layers=2` collapses discrepancy signal (disc_d: 2.44→0.21), making teacher/student outputs nearly identical. Combined enc2+td4 drops roc from 0.9855 to 0.7592. `lr=5e-3` also degrades at w500 (-0.040 roc). Corrected defaults: `num_encoder_layers=1`, `learning_rate=2e-3`. Phase 2 config file updated.

### Corrected Parameters

| Parameter | Corrected | Previous | Reason |
|-----------|-----------|----------|--------|
| num_encoder_layers | 1 | 2 | enc2 collapses disc_d at w500 (0.21 vs 2.44) |
| learning_rate | 2e-3 | 5e-3 | lr=5e-3 too aggressive for w500+d128 |

## 2026-01-31 (Update 43): Phase 2 Experiment Plan & Default Parameter Update

### Summary

Updated model default parameters based on Phase 1 ablation analysis (1,014 evaluations). Created Phase 2 experiment plan with 150 configs (600 total evaluations). New defaults reflect Phase 1 optimal findings: larger window (500), larger model (d128/nh8), deeper decoder (td4), higher learning rate (0.005), lower masking ratio (0.15), and stronger discrepancy training (λ=2.0, k=2.0, alw=2).

### Default Parameter Changes

| Parameter | New | Old | Rationale |
|-----------|-----|-----|-----------|
| seq_length | 500 | 100 | Best disturbing-normal separation (H3) |
| d_model | 128 | 64 | Critical for w500 performance (H7) |
| nhead | 8 | 2 | Best mean roc_auc (0.9694) |
| dim_feedforward | 512 | 256 | d_model × 4 |
| num_encoder_layers | 2 | 1 | el=2-3 improves over el=1 |
| num_teacher_decoder_layers | 4 | 2 | Best overall by mean and max |
| patch_size | 20 | 10 | Optimal for w500 (25 patches) |
| masking_ratio | 0.15 | 0.2 | SNR sweet spot 0.08-0.15 |
| lambda_disc | 2.0 | 0.5 | Eliminates scoring mode gap for mask_after |
| dynamic_margin_k | 2.0 | 1.5 | Higher k helps mask_after disc_d |
| anomaly_loss_weight | 2.0 | 1.0 | Boosts mask_after disc_d +22% |
| dropout | 0.15 | 0.1 | Between 0.1 and 0.2 (phase1 best) |
| shared_mask_token | False | True | Separate mask tokens preferred |

### Code Changes

| Component | Changes |
|-----------|---------|
| `config.py` | Updated all default parameter values |

### Documentation Changes

| File | Changes |
|------|---------|
| `docs/ARCHITECTURE.md` | Updated Default Configuration table |
| `docs/ablation/phase2/PHASE2_PLAN.md` | New file: 150-config Phase 2 experiment plan |
| `docs/CHANGELOG.md` | This entry |

---

## 2026-01-30 (Update 42): Point-Level Evaluation Refactor

### Summary

Refactored all evaluation metrics from patch-level to point-level. Primary metrics (roc_auc, f1, precision, recall) now use point-level scores computed by mean-aggregating patch scores to physical timestamps. PA%K metrics use majority voting with the point-level threshold instead of independent threshold optimization per K.

### Code Changes

| Component | Changes |
|-----------|---------|
| `evaluator.py` | `evaluate()`, `evaluate_by_score_type()`, `get_performance_by_anomaly_type()` refactored to point-level; added `_compute_voted_point_predictions()` and `_compute_pa_k_f1_at_threshold()` helpers |
| `visualization/base.py` | `collect_predictions()` and `collect_all_visualization_data()` now return point-level scores/labels as primary, with patch-level data retained for loss stats and voting |
| `visualization/best_model_visualizer.py` | Updated ROC, threshold, detection examples, comparison plots to use point-level data; removed patch-level masked region highlighting from detection plots |
| `run_ablation.py` | No changes needed — CSV column mapping auto-adapts via `**metrics` unpacking |

### Documentation Changes

| File | Changes |
|------|---------|
| ARCHITECTURE.md | Added "Point-Level Aggregation" section, clarified inference metrics |
| VISUALIZATIONS.md | Updated inference mode description for point-level aggregation |
| ABLATION_STUDIES.md | Clarified point-level labeling in evaluation strategy |

## 2026-01-29 (Update 41): Remove last_patch inference mode

### Summary

Removed `last_patch` inference mode entirely. The system now exclusively uses `all_patches` (iterative per-patch masking with N forward passes). Removed `inference_mode` and `mask_last_n` from Config. Deleted 1020 last_patch result directories. Updated all code, configs, and documentation.

### Code Changes

| File | Changes |
|------|---------|
| `config.py` | Removed `inference_mode` and `mask_last_n` fields |
| `evaluator.py` | Deleted `aggregate_scores_to_point_level()`, `compute_point_level_pa_k()`, `_compute_raw_scores_last_patch()`, `_compute_raw_scores_all_patches()`; simplified all methods to remove branching |
| `trainer.py` | Renamed `last_patch_labels` → `window_labels`; `mask_last_n` → `patch_size` |
| `visualization/base.py` | Removed inference_mode branching in all collect functions |
| `visualization/best_model_visualizer.py` | Removed `self.inference_mode` and all conditional branches |
| `visualization/training_visualizer.py` | `last_patch_labels` → `window_labels`; `mask_last_n` → `patch_size` |
| `visualization/data_visualizer.py` | `config.mask_last_n` → `config.patch_size` |
| `visualization/stage2_visualizer.py` | Removed `mask_last_n` from hyperparameter display |
| `run_ablation.py` | Removed inference_mode loops, suffixes, and INFERENCE_MODES handling |
| `configs/phase1.py` | Removed `INFERENCE_MODES` list and `mask_last_n` from experiments |

### Documentation Changes

| File | Changes |
|------|---------|
| `INFERENCE_MODES.md` | Rewritten: single inference process (removed last_patch section and comparison) |
| `ABLATION_EXPERIMENTS.md` | Variants: 12 → 6 per experiment; Total: 2040 → 1020 |
| `ABLATION_STUDIES.md` | Removed inference modes section; updated variant counts |
| `ARCHITECTURE.md` | Simplified inference time description |
| `VISUALIZATIONS.md` | Removed inference mode handling table |
| `DATASET.md` | `config.mask_last_n` → `config.patch_size` |

### Results Cleanup

Deleted 1020 `*_last` directories from `results/experiments/20260128_012500_phase1/`.

---

## 2026-01-29 (Update 40): evaluate_by_score_type, Documentation Sync & Cleanup

### Summary

Implemented `evaluate_by_score_type()` in evaluator to populate 24 CSV columns (disc_only_*, teacher_recon_*, student_recon_*) that were previously always 0. Added student reconstruction error to evaluator cache. Comprehensive documentation sync across all docs. Removed obsolete scripts and analysis files.

### Evaluator Changes

| Change | Description |
|--------|-------------|
| `evaluate_by_score_type(score_type)` | NEW: Evaluate using individual score components ('disc', 'teacher_recon', 'student_recon') |
| Student recon in cache | `_compute_raw_scores_last_patch()` and `_compute_patch_scores_all_patches()` now return 6-tuple including student_recon |
| 24 CSV columns | disc_only_*, teacher_recon_*, student_recon_* now have real values |

### Documentation Sync

Fixed inconsistencies across all documentation files:

| Parameter | Old Doc Value | Corrected Value |
|-----------|--------------|-----------------|
| Patchify modes | linear/cnn_first/patch_cnn | linear/patch_cnn (cnn_first removed) |
| Default patchify_mode | linear | patch_cnn |
| sliding_window_total_length | 440K / 2.2M | 275K |
| anomaly_interval_scale | 1.5 | 0.75 |
| Scoring modes (ablation) | default/adaptive/disc_only | default/adaptive/normalized |
| Anomaly types | 6 / 11 names | 9 types (10 names including normal) |
| Train/test split | 50/50 | 80/20 |
| Default margin_type | hinge | dynamic |
| teacher_only_warmup_epochs | 1 | 3 |
| Feature count (design doc) | 5 | 8 |

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/evaluator.py` | Added `evaluate_by_score_type()`, student_recon in cache |
| `mae_anomaly/model.py` | Minor updates |
| `mae_anomaly/visualization/base.py` | Optimization updates |
| `mae_anomaly/visualization/best_model_visualizer.py` | ROC comparison methods |
| `CLAUDE.md` | Fixed patchify modes, dataset stats, added evaluator mapping |
| `docs/ARCHITECTURE.md` | Removed cnn_first, fixed defaults, added per-component scoring |
| `docs/DATASET.md` | Fixed total_length, interval_scale, anomaly type counts |
| `docs/ABLATION_STUDIES.md` | Fixed masking ratio, scoring modes, dataset size |
| `docs/INFERENCE_MODES.md` | Fixed scoring mode reference |
| `docs/VISUALIZATIONS.md` | Updated date, dataset sizes, removed CNN-First reference |
| `docs/CHANGELOG.md` | This entry |

### Files Removed

| File | Reason |
|------|--------|
| `scripts/ablation/configs/phase2.py` | Obsolete (merged into phase1) |
| `scripts/analyze_phase1_results.py` | One-off analysis script |
| `scripts/deep_analysis_phase1.py` | One-off analysis script |
| `scripts/generate_phase1_report.py` | One-off report generator |
| `docs/ablation_result/phase1/*` | Stale analysis results |
| `scripts/profile_*.py`, `scripts/benchmark_*.py`, `scripts/verify_*.py` | One-off profiling/verification scripts (moved to .trash/) |

---

## 2026-01-28 (Update 39): Segment-Based PA%K Fix & Documentation Update

### Summary

Fixed critical PA%K (Point-Adjust with K%) metric calculation to use proper segment-based detection rates instead of sample-level approximation. Updated all documentation to match current codebase.

### PA%K Metric Fix

**Problem Identified**:
- `plot_performance_by_anomaly_type_comparison()` was using sample-level `compute_pa_k_adjusted_predictions()` which breaks segment structure when filtering by anomaly_type
- This caused incorrect PA%K calculations in visualization

**Solution**:
- Added segment-based PA%K calculation using `compute_segment_pa_k_detection_rate()` in `best_model_visualizer.py`
- Pre-computes point-level scores for each scoring method
- Uses `test_dataset.anomaly_regions` for proper segment-based detection rate

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/visualization/best_model_visualizer.py` | Added segment-based PA%K support in `plot_performance_by_anomaly_type_comparison()` |
| `docs/ARCHITECTURE.md` | Fixed encoder/decoder layer counts, attention heads, masking_ratio (0.2), added missing parameters |
| `docs/DATASET.md` | Updated sliding_window_total_length (440K), stride (11), window counts |
| `docs/ABLATION_STUDIES.md` | Updated to reflect new ablation framework (run_ablation.py) |
| `docs/CHANGELOG.md` | Added this entry |

### Documentation Sync

Updated documentation to match current config.py defaults:

| Parameter | Old Doc Value | New Value |
|-----------|--------------|-----------|
| Encoder layers | 3 | 1 |
| Teacher decoder layers | 4 | 2 |
| Attention heads | 4 | 2 |
| masking_ratio | 0.4 | 0.2 |
| sliding_window_total_length | 2.2M | 440K |
| sliding_window_stride | 10 | 11 |

### Usage

PA%K metrics are now calculated correctly in all visualizations:
- `plot_performance_by_anomaly_type()` - reads from JSON (auto-benefits)
- `plot_performance_by_anomaly_type_comparison()` - now uses segment-based calculation

---

## 2026-01-27 (Update 38): Comprehensive Phase 1 Deep Analysis & Strategic Phase 2 Planning

### Summary

Ultra-deep analysis of 1,398 Phase 1 ablation experiments across 10 strategic focus areas. Generated comprehensive insights document and created 150 Phase 2 experiments organized into 8 targeted groups based on critical findings about balancing discrepancy and reconstruction objectives.

### Key Discoveries

1. **Balance Over Extremes**: High disc_ratio (>4.0) models achieve poor ROC-AUC (0.74-0.88) due to sacrificing reconstruction quality. Best models balance moderate disc_cohens_d (0.9-1.2) with high recon_cohens_d (2.5-3.8).

2. **Reconstruction Quality Dominates**: recon_cohens_d correlates more strongly with ROC-AUC (r=+0.518) than disc_cohens_d (r=+0.445), revealing reconstruction as the foundation for anomaly detection.

3. **Configuration Winners**:
   - Inference: all_patches (+5.9% over last_patch)
   - Scoring: default (best for tuned models)
   - Baseline: w500_p20, d_model=128
   - Teacher-Student: t4s1, t4s2 optimal

4. **Disturbing Normal Challenge**: Best disc_cohens_d_disturbing_vs_anomaly only 0.803 (vs 1.926 for pure normal), identifying this as the key frontier for improvement.

5. **Scarcity of Excellence**: Only 3 models achieved both high disc_d (>1.33) AND high recon_d (>1.73), averaging ROC-AUC 0.942 and PA%80 0.951.

### Analysis Framework (10 Focus Areas)

| Focus Area | Key Finding | Phase 2 Impact |
|------------|-------------|----------------|
| 1. High Disc Ratio | Top 50 models (disc_d 1.88-1.93) average only 0.860 ROC-AUC | GROUP 1: Optimize balance |
| 2. Disc+Recon Balance | Only 3 models meet criteria → GOLDEN ZONE | GROUP 1: Replicate success |
| 3. Modes & Windows | all_patches +0.047 ROC-AUC; w500_p20 strong baseline | GROUP 2: Scale windows |
| 4. Disturbing Separation | 009_w500_p20 achieves 0.803 (best) | GROUP 3: Push beyond 0.85 |
| 5. PA%80 + Disc Ratio | Rare combination, critical for deployment | GROUP 4: Systematic optimization |
| 6. Window-Depth-Masking | Relationships need systematic exploration | GROUP 2, 6, 7 |
| 7. Mask After Optimization | Most top models use mask_after=False | GROUP 8: Lambda tuning |
| 8. Mode Sensitivity | Same model: 0.956 (default) vs 0.928 (adaptive) | GROUP 4: Test systematically |
| 9. High Perf + Disturbing | Achieving both is rare but valuable | GROUP 3: Targeted approach |
| 10. Additional Insights | disc_ratio negatively correlated with ROC-AUC (r=-0.124) | All groups: Avoid extremes |

### Phase 2 Strategic Plan (150 Experiments)

| Group | Experiments | Goal | Strategy |
|-------|-------------|------|----------|
| **1: Balanced Disc+Recon** | 30 | disc_d > 1.2, recon_d > 2.8 | Build on 028_d128_nhead_16, vary masking/lambda |
| **2: Window & Capacity** | 25 | Scaling laws | Test w100/500/1000 with matched capacity |
| **3: Disturbing Separation** | 20 | disc_d_disturbing > 0.85 | Build on 009_w500_p20, vary k/lambda/weight |
| **4: PA%80 Optimization** | 20 | PA%80 > 0.970 | Large windows, high capacity, mode testing |
| **5: Teacher-Student Ratios** | 15 | Optimal T:S balance | Systematic t1s1 through t6s1, balanced ratios |
| **6: Masking Strategy** | 15 | Optimal ratios per d_model | d128: [0.05-0.35], d256: [0.60-0.90] |
| **7: Architecture Depth** | 15 | Optimal encoder-decoder | Systematic depth combinations |
| **8: Lambda Discrepancy** | 10 | Optimal loss weighting | Fine-grained [0.5-3.0] |

### Files Created

| File | Purpose |
|------|---------|
| `docs/ablation_result/PHASE1_COMPREHENSIVE_ANALYSIS.md` | 📄 Complete analysis report (13KB) |
| `docs/ablation_result/phase1_analysis_report.md` | 📊 Executive summary with tables |
| `docs/ablation_result/table1_top10_roc_auc.csv` | 🏆 Top 10 models by ROC-AUC |
| `docs/ablation_result/table2_top10_disc_ratio.csv` | 📈 Top 10 by discrepancy ratio |
| `docs/ablation_result/table3_top10_t_ratio.csv` | 🎯 Top 10 by teacher reconstruction ratio |
| `docs/ablation_result/all_experiments.csv` | 💾 All 1,398 results (1.2MB) |
| `scripts/ablation/configs/phase2.py` | ⚙️ 150 Phase 2 experiment configs |
| `scripts/analyze_phase1_results.py` | 🔧 Analysis script |
| `scripts/generate_phase1_report.py` | 📝 Report generator |
| `docs/CHANGELOG.md` | 📋 UPDATED (this entry) |

### Usage

```bash
# Review comprehensive analysis
cat docs/ablation_result/PHASE1_COMPREHENSIVE_ANALYSIS.md

# Review executive summary
cat docs/ablation_result/phase1_analysis_report.md

# Verify Phase 2 config
python scripts/ablation/configs/phase2.py

# Run Phase 2 experiments
python scripts/ablation/run_ablation.py --config configs/phase2.py
```

### Expected Phase 2 Outcomes

1. **10+ models with ROC-AUC > 0.960** (vs Phase 1 best: 0.9624)
2. **5+ models with disc_d > 1.2 AND recon_d > 2.8** (vs Phase 1: only 3)
3. **disc_cohens_d_disturbing_vs_anomaly > 0.85** (vs Phase 1 best: 0.803)
4. **PA%80 ROC-AUC > 0.970** (vs Phase 1 best: 0.965)
5. **Establish scaling laws** for window size vs model capacity
6. **Identify 2-3 production-ready configurations**

### Documentation Philosophy

- **Insight-Driven**: Each experiment group based on specific Phase 1 insight
- **Hypothesis-Testing**: Clear hypotheses with verification criteria
- **Balanced Approach**: Optimize for balance, not single metric extremes
- **Deployment-Ready**: Focus on PA%80 and disturbing normal separation

---

## 2026-01-27 (Update 37): Phase 1 Analysis and Phase 2 Experiment Planning

### Summary

Comprehensive analysis of 1,392 Phase 1 ablation experiments with deep-dive insights across 10 analysis points. Generated 150 Phase 2 experiment configurations organized into 7 thematic tracks based on Phase 1 findings.

### Key Findings

1. **Best Performance:** ROC-AUC=0.9624 with `mask_after=False`, `d_model=128`, `nhead=16`
2. **Highest Disc Ratio:** 4.26 with `mask_after=True`, `dynamic_margin_k=4.0` (but lower ROC-AUC)
3. **Trade-off Identified:** High disc_ratio negatively correlates with performance (-0.45 with recon_ratio)
4. **Window Size:** w500_p20 achieved ROC-AUC=0.9586 (2nd best), warrants further exploration
5. **Inference Mode:** `all_patches` outperforms `last_patch` by +0.046 ROC-AUC

### Phase 2 Experiment Tracks (150 total)

1. **Track 1 (30):** Balanced Performance Optimization - optimize mask_before configs
2. **Track 2 (25):** Window Size Exploration - systematically test w500/1000/1500
3. **Track 3 (25):** High Disc Ratio Optimization - improve ROC while maintaining high disc
4. **Track 4 (20):** Disturbing Normal Discrimination - optimize disturbing vs anomaly separation
5. **Track 5 (20):** Architectural Depth - systematic encoder-decoder depth exploration
6. **Track 6 (15):** Masking Ratio Fine-tuning - fine-grained search in 0.08-0.3 range
7. **Track 7 (15):** Lambda_disc Exploration - systematic lambda values

### Files

| File | Status |
|------|--------|
| `docs/ablation_result/phase1_top_models_tables.md` | NEW (top 10 models by 3 metrics) |
| `docs/ablation_result/phase1_deep_analysis.md` | NEW (10-point analysis, 22 tables) |
| `docs/ablation_result/PHASE1_SUMMARY_AND_PHASE2_PLAN.md` | NEW (executive summary) |
| `scripts/ablation/configs/phase2/20260127_141642_phase2.py` | NEW (150 phase2 configs) |
| `docs/CHANGELOG.md` | UPDATED |

### Usage

```bash
# View analysis results
cat docs/ablation_result/PHASE1_SUMMARY_AND_PHASE2_PLAN.md

# Run Phase 2 experiments
python scripts/ablation/run_ablation.py --config configs/phase2/20260127_141642_phase2.py
```

---

## 2026-01-27 (Update 36): Unified Ablation Study and Visualization Optimization

### Summary

Unified Phase 1 and Phase 2 ablation configs into single Phase 1 (170 experiments). Added parallel visualization support and optimized data collection with `collect_all_visualization_data()` function for ~2x speedup.

### Changes

1. **Unified Ablation Config** (`scripts/ablation/configs/20260127_052220_phase1.py`):
   - Combined 70 (Phase 1) + 100 (Phase 2) = **170 experiments**
   - Unified base config defaults: d_model=64, nhead=2, masking_ratio=0.2
   - Total expected results: 170 × 2 (mask) × 2 (inference) × 3 (scoring) = **2040**

2. **Visualization Optimization** (`mae_anomaly/visualization/base.py`):
   - Added `collect_all_visualization_data()` - merged function for ~2x speedup
   - Combines `collect_predictions()` and `collect_detailed_data()` into single pass
   - Reduces redundant forward passes

3. **Parallel Visualization** (`mae_anomaly/visualization/parallel.py`):
   - New `ParallelVisualizer` class for multiprocessing-based plot generation
   - New `generate_plots_parallel()` helper function
   - Uses file-based data passing to avoid IPC overhead

4. **Module Exports** (`mae_anomaly/visualization/__init__.py`):
   - Added `collect_all_visualization_data` export
   - Added `ParallelVisualizer`, `generate_plots_parallel` exports

### Usage

```bash
# Run unified Phase 1 (170 experiments × 12 variants = 2040 results)
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py
```

### Files

| File | Status |
|------|--------|
| `scripts/ablation/configs/20260127_052220_phase1.py` | NEW (unified) |
| `mae_anomaly/visualization/base.py` | MODIFIED (collect_all_visualization_data) |
| `mae_anomaly/visualization/parallel.py` | NEW |
| `mae_anomaly/visualization/__init__.py` | MODIFIED |
| `docs/ABLATION_EXPERIMENTS.md` | UPDATED |
| `docs/VISUALIZATIONS.md` | UPDATED |

---

## 2026-01-27: Ablation Study Framework Refactoring

### Summary

Refactored ablation study scripts into a unified, modular framework with separate config files.

### Changes

1. **Unified Runner** (`scripts/ablation/run_ablation.py`):
   - Single entry point for all ablation studies
   - Dynamic config loading from Python files
   - Background visualization with concurrency control
   - Skip-existing and experiment filtering support

2. **Config Files** (`scripts/ablation/configs/`):
   - Modular format for easy extension

3. **Visualization Fix** (`mae_anomaly/visualization/best_model_visualizer.py`):
   - Fixed `best_model_score_contribution_trends.png` for adaptive/normalized modes
   - Now correctly recalculates disc score weights from raw history values

### Usage

```bash
# Run unified Phase 1
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py

# Run specific experiments
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py \
    --experiments 001_default 002_window_200
```

### Files

| File | Status |
|------|--------|
| `scripts/ablation/run_ablation.py` | NEW |
| `scripts/ablation/configs/__init__.py` | NEW |
| `scripts/ablation/configs/20260127_052220_phase1.py` | NEW |
| `scripts/ablation/run_ablation_experiments_*.py` | DEPRECATED |
| `docs/ABLATION_EXPERIMENTS.md` | UPDATED |
| `mae_anomaly/visualization/best_model_visualizer.py` | FIXED |

---

## 2026-01-25 (Update 34): Mixed Precision Training (AMP) Support

### Summary

Added Automatic Mixed Precision (AMP) training support for faster training and reduced memory usage.

### Performance Impact (RTX 3080 Ti)

| Metric | No AMP | AMP | Improvement |
|--------|--------|-----|-------------|
| Training time | 2.89s | 2.40s | **1.20x** |
| Inference time | 4.32s | 3.52s | **1.23x** |
| Training memory | 449 MB | 272 MB | **40% ↓** |
| Inference memory | 1062 MB | 437 MB | **59% ↓** |

### Changes

1. **Config**: Added `use_amp: bool = True` option
2. **Epsilon values**: Changed all `1e-8` → `1e-4` for float16 numerical stability
3. **Trainer**: Added `autocast` and `GradScaler` for mixed precision training
4. **Evaluator**: Added `autocast` for mixed precision inference

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/config.py` | Added `use_amp` option |
| `mae_anomaly/loss.py` | 14x epsilon update |
| `mae_anomaly/trainer.py` | AMP support + 8x epsilon update |
| `mae_anomaly/evaluator.py` | AMP support + 6x epsilon update |

### Notes

- AMP is enabled by default (`use_amp=True`)
- Requires GPU with Tensor Cores (Volta+) for best speedup
- Accuracy is preserved (ROC-AUC difference < 0.01)

---

## 2026-01-25 (Update 33): Performance Optimization - Batched all_patches and Training Params

### Summary

Major performance improvements: batched `all_patches` inference (7x speedup), batch_size=1024, learning_rate=5e-3.

### Changes

1. **Batched all_patches Inference** (~7x speedup):
   - `evaluator.py`: `_compute_patch_scores_all_patches()` now processes all patches in single forward pass
   - `visualization/base.py`: `collect_predictions()` and `collect_detailed_data()` also optimized
   - Before: 10 forward passes per batch (one per patch)
   - After: 1 forward pass per batch (all patches expanded in batch dimension)

2. **Updated Training Parameters**:
   - `batch_size`: 32 → 1024 (better GPU utilization, ~0.6GB VRAM)
   - `learning_rate`: 2e-3 → 5e-3 (faster convergence with larger batch)

3. **Enabled cuDNN Benchmark**:
   - `cudnn.benchmark = True` for auto-tuned convolution algorithms
   - Additional ~20% training speedup

### Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| all_patches per batch | 23.87 ms | 3.43 ms | **7.0x** |
| Training throughput | - | ~3x faster | GPU utilization |

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/config.py` | batch_size=1024, learning_rate=5e-3, cudnn.benchmark=True |
| `mae_anomaly/evaluator.py` | Batched all_patches in `_compute_patch_scores_all_patches()` |
| `mae_anomaly/visualization/base.py` | Batched all_patches in `collect_predictions()`, `collect_detailed_data()` |
| `docs/DATASET.md` | Updated DataLoader example |
| `docs/ABLATION_EXPERIMENTS.md` | Updated learning_rate default |
| `docs/ARCHITECTURE.md` | Updated learning_rate default |

---

## 2026-01-25 (Update 32): Visualization Cleanup and all_patches Mode Fixes

### Summary

Removed redundant visualization functions, fixed `all_patches` mode visualizations, and added new score contribution epoch trends plot.

### Changes

1. **Removed Visualization Functions** (simplification):
   - `plot_score_distribution()` - redundant with score_contribution_analysis
   - `plot_score_components()` - redundant with score_contribution_analysis
   - `plot_teacher_student_comparison()` - not essential for analysis
   - `plot_hypothesis_verification()` - not essential for analysis
   - `plot_feature_contribution_analysis()` - not essential for analysis

2. **Fixed all_patches Mode Visualizations**:
   - Added `_patch_idx_to_window_idx()` helper for index conversion
   - Fixed `plot_detection_examples()`, `plot_case_study_gallery()`, `_plot_sample_detail()` to use window index
   - Fixed `plot_reconstruction_examples()` to skip masked region shading in all_patches mode

3. **Added New Visualization**:
   - `plot_score_contribution_epoch_trends()`: Stacked area plots showing recon/disc score contributions over epochs for each anomaly type (similar to J-L plots), with unified y-axis and starting from epoch 5

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/visualization/best_model_visualizer.py` | Removed 5 functions, added helper and new plot, fixed all_patches mode |
| `docs/VISUALIZATIONS.md` | Updated visualization list and API examples |

---

## 2026-01-25 (Update 31): Fix Dimension Mismatch in collect_detailed_data for all_patches Mode

### Summary

Bug fix: `collect_detailed_data()` now returns consistent window-level shapes for both inference modes.

### Problem

In `all_patches` mode, `collect_detailed_data()` returned mismatched shapes:
- `teacher_errors`, `student_errors`: (n_windows, seq_length) = (2625, 100)
- `labels`, `sample_types`: flattened to (n_windows × num_patches,) = (26250,)

This caused `IndexError` in visualization functions like `plot_teacher_student_comparison()` when using labels as boolean masks on errors.

### Solution

Changed `collect_detailed_data()` to keep window-level labels for `all_patches` mode:
- Use `last_patch_labels` instead of flattened `patch_labels`
- Keep `sample_types` at window level instead of expanding

Note: `collect_predictions()` correctly uses patch-level labels since it also returns patch-level scores for metrics.

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/visualization/base.py` | `collect_detailed_data()` uses window-level labels for all_patches mode |

### Impact

- All 18 visualization files now generate correctly for `all_patches` mode
- Previously only 5 files were generated before the error occurred

---

## 2026-01-25 (Update 30): Fix Visualization Functions to Respect inference_mode

### Summary

Bug fix: `collect_predictions()` and `collect_detailed_data()` in visualization/base.py now respect `config.inference_mode` setting instead of always using last_patch mode.

### Problem

Confusion matrices and other visualizations were identical for both `last_patch` and `all_patches` inference modes because visualization functions ignored the `inference_mode` config:
- Always masked only the last patch
- Always used `last_patch_labels`

### Solution

Updated both functions to handle `inference_mode`:

**For `all_patches` mode:**
- Mask each patch one at a time (N forward passes)
- Compute patch-level labels from `point_labels`
- Flatten scores/labels to (n_windows × num_patches,)

**For `last_patch` mode:**
- Original behavior preserved

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/visualization/base.py` | `collect_predictions()`, `collect_detailed_data()` now check `inference_mode` |
| `docs/ARCHITECTURE.md` | Added inference_mode documentation |
| `docs/VISUALIZATIONS.md` | Added inference_mode handling section |

### Impact

- Confusion matrices now correctly differ between inference modes
- ROC curves, score distributions match evaluation methodology
- Ablation experiments restarted with fix applied

---

## 2026-01-25 (Update 29): Point-Level PA%K with Stride=1 Sliding Window

### Summary

Major update to evaluation methodology: Test set now uses stride=1 sliding windows without downsampling, enabling proper point-level PA%K evaluation with window score aggregation.

### Key Changes

1. **Model Parameters Updated**
   - `patch_size`: 4 → **10** (larger patches for better context)
   - `num_patches`: 25 → **10** (seq_length / patch_size = 100/10)
   - `mask_last_n`: 4 → **10** (matches patch_size)

2. **Test Set Evaluation**
   - Stride forced to 1 for test split (each timestep covered by multiple windows)
   - Downsampling disabled by default (full sliding window coverage)
   - Window scores aggregated to point-level for PA%K

3. **Point-Level Aggregation Methods**
   - **Voting** (default): Majority vote of binary predictions
   - **Mean**: Average of window scores per timestep
   - **Median**: Median of window scores per timestep

### Window Coverage with Stride=1

```
Window w's last patch: [w+90, w+99] (10 timesteps)
Each timestep covered by up to 10 windows
```

### Metrics Separation

| Metric Type | Level | Notes |
|-------------|-------|-------|
| ROC-AUC, F1, Precision, Recall | Sample (window) | Unchanged |
| PA%K F1, PA%K ROC-AUC | **Point (timestep)** | Aggregated via voting |

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/config.py` | `patch_size=10`, `num_patches=10`, `mask_last_n=10`, `point_aggregation_method` |
| `mae_anomaly/dataset_sliding.py` | Stride=1 forced for test, `window_start_indices` added |
| `mae_anomaly/evaluator.py` | `aggregate_scores_to_point_level()`, `compute_point_level_pa_k()` |
| `scripts/run_experiments.py` | Pass `test_dataset` to Evaluator |
| `scripts/run_temp_experiments.py` | Pass `test_dataset` to Evaluator |
| `docs/ARCHITECTURE.md` | Updated patch dimensions |
| `docs/DATASET.md` | Added Point-Level PA%K section |

### Configuration

```python
# New config parameter
config.point_aggregation_method = 'voting'  # 'mean', 'median', 'voting'
```

### Backwards Compatibility

- Evaluator accepts optional `test_dataset` parameter
- Without `test_dataset`, falls back to sample-level PA%K
- Train set behavior unchanged (configurable stride)

---

## 2026-01-25 (Update 28): Comprehensive PA%K Metrics (K=10,20,50,80)

### Summary

Extended PA%K evaluation to compute both F1-score and ROC-AUC for K=10%, 20%, 50%, 80% (total 8 metrics). Added PA%K ROC-AUC computation that applies segment adjustment at each threshold level.

### Key Features

1. **PA%K ROC-AUC Algorithm**: For each threshold, binarize → apply PA%K adjustment → compute TPR/FPR → build ROC curve
2. **8 PA%K Metrics**: F1 + ROC-AUC for each K value (10, 20, 50, 80)
3. **9-Subplot Visualization**: Compare Point-wise, PA%10 (lenient), PA%80 (strict)

### New Metrics

| Metric | Description |
|--------|-------------|
| `pa_10_f1`, `pa_10_roc_auc` | PA%10 (very lenient, 10% segment detection) |
| `pa_20_f1`, `pa_20_roc_auc` | PA%20 (lenient) |
| `pa_50_f1`, `pa_50_roc_auc` | PA%50 (moderate) |
| `pa_80_f1`, `pa_80_roc_auc` | PA%80 (strict, 80% segment detection) |

### Changes

#### 1. Core Implementation (evaluator.py)
- Added `compute_pa_k_roc_auc()` function for threshold-aware PA%K ROC-AUC
- `evaluate()` returns 8 PA%K metrics (F1 + ROC-AUC × 4 K values)
- `get_performance_by_anomaly_type()` includes all 4 K values for detection rates

#### 2. Experiment Scripts
- `run_experiments.py` - Saves all 8 PA%K metrics to results
- `run_temp_experiments.py` - Displays PA%K table in console output

#### 3. Visualization (best_model_visualizer.py)
- New 3×3 grid (9 subplots) showing:
  - Row 1: Point-wise, PA%10, PA%80 detection rates
  - Row 2: All PA%K comparison, PA%10 vs PA%80, Mean scores
  - Row 3: Consistency gap, Sample distribution, Summary statistics

---

## 2026-01-25 (Update 27): PA%K (Point-Adjust with K%) Evaluation Metric

### Summary

Added PA%K evaluation metric (default K=20%) for more realistic time series anomaly detection evaluation. PA%K is a segment-level adjustment that considers an anomaly segment as "detected" if at least K% of its points are flagged.

### Motivation

Point-wise F1 score can be overly harsh for time series anomaly detection because:
- If a model detects 9 out of 10 anomaly points but misses 1, point-wise F1 penalizes heavily
- In practice, detecting ANY point within an anomaly segment is often sufficient for alerting
- PA%K provides a more realistic evaluation by giving credit for partial segment detection

### PA%K Algorithm

```
For each contiguous anomaly segment:
    if (detected_points / total_points) >= K%:
        All points in segment count as DETECTED (TP)
    else:
        All points in segment count as NOT DETECTED (FN)
```

With K=20% (PA%20):
- A segment of 100 points needs only 20 detected points to count as fully detected
- Balanced between leniency and rigor for real-world alerting scenarios

### Changes

#### 1. Core Implementation (evaluator.py)

- Added `compute_pa_k_adjusted_predictions()` function
- Added `compute_pa_k_metrics()` function returning precision, recall, F1
- Updated `evaluate()` to include `pa_k_f1`, `pa_k_precision`, `pa_k_recall`
- Updated `get_performance_by_anomaly_type()` to include `pa_k_detection_rate` per type

#### 2. Experiment Scripts

- `run_experiments.py` - Added PA%K columns to Stage 2 results
- `run_temp_experiments.py` - Added PA%K to summary display and console output

#### 3. Visualization (best_model_visualizer.py)

- Updated `plot_performance_by_anomaly_type()` to show side-by-side comparison:
  - Point-wise detection rate (lighter bars)
  - PA%20 detection rate (darker bars)

### New Metrics in Experiment Results

| Metric | Description |
|--------|-------------|
| `pa_k_f1` | PA%K F1 score (K=20%) |
| `pa_k_precision` | PA%K precision |
| `pa_k_recall` | PA%K recall |
| `pa_k_detection_rate` | Per-anomaly-type PA%K detection rate |

### Files Modified

- `mae_anomaly/evaluator.py` - Core PA%K implementation
- `mae_anomaly/visualization/best_model_visualizer.py` - Visualization update
- `scripts/run_experiments.py` - Result column additions
- `scripts/run_temp_experiments.py` - Display and column updates

---

## 2026-01-25 (Update 26): Remove point_spike Anomaly Type

### Summary

Removed `point_spike` (formerly type 7) from anomaly types. Pattern-based anomalies are renumbered from 8-10 to 7-9.

### Rationale

Point spike anomalies were:
1. Too similar to the existing `spike` anomaly type
2. Very short duration (3-5 timesteps) made them unrealistic for most real-world monitoring scenarios
3. Random feature selection made them inconsistent for systematic evaluation

### Changes

#### Before → After

| Category | Before | After |
|----------|--------|-------|
| Value-based | Types 1-7 | Types 1-6 |
| Pattern-based | Types 8-10 | Types 7-9 |
| Total | 10 types | 9 types |

#### Anomaly Type Renumbering

| Old ID | New ID | Name |
|--------|--------|------|
| 7 | (removed) | point_spike |
| 8 | 7 | correlation_inversion |
| 9 | 8 | temporal_flatline |
| 10 | 9 | frequency_shift |

### Files Modified

- `mae_anomaly/dataset_sliding.py` - Removed point_spike, renumbered types
- `mae_anomaly/visualization/base.py` - Updated anomaly info
- `mae_anomaly/visualization/data_visualizer.py` - Removed point_spike comment
- `docs/DATASET.md` - Updated all references

---

## 2026-01-25 (Update 25): Pattern-Only Anomalies for Meaningful Detection Validation

### Summary

Added 3 new pattern-based anomaly types that maintain normal value ranges but break temporal/correlation patterns. This allows distinguishing between trivial value-based detection (detecting unusual VALUES) and meaningful pattern-based detection (detecting unusual PATTERNS).

### Problem Statement

Previously, ALL anomaly types were ADDITIVE (values increase beyond normal range). This made it impossible to determine if the model was:
- Detecting anomalies because of unusual **VALUES** (trivial statistical detection)
- Detecting anomalies because of unusual **PATTERNS** (meaningful anomaly detection)

### Changes

#### 1. Added 3 Pattern-Only Anomaly Types (dataset_sliding.py)

| Type ID | Name | Description | Pattern Break |
|---------|------|-------------|---------------|
| 7 | correlation_inversion | CPU-Memory correlation breaks | Cross-feature correlation |
| 8 | temporal_flatline | Values freeze (stuck sensor) | Temporal continuity |
| 9 | frequency_shift | Unusual oscillation frequency | Normal periodicity |

All pattern-based anomalies use `np.clip(value, 0.15, 0.85)` to ensure values stay within normal range.

#### 2. Added ANOMALY_CATEGORY Metadata

```python
ANOMALY_CATEGORY = {
    1: 'value', 2: 'value', 3: 'value', 4: 'value',
    5: 'value', 6: 'value',
    7: 'pattern', 8: 'pattern', 9: 'pattern'
}
```

#### 3. Fixed Y-axis Unification in loss_by_anomaly_type (best_model_visualizer.py)

Applied unified y-axis limits across all subplots for fair visual comparison.

#### 4. Added Value vs Pattern Comparison Visualization

New `plot_value_vs_pattern_comparison()` method showing:
- Score distribution comparison (Normal vs Value-based vs Pattern-based)
- Box plot comparison
- Detection rate comparison
- Loss components comparison

#### 5. Distinct Colors for Pattern-Based Anomalies (base.py)

Pattern-based anomalies use cool colors (blue/purple) to visually distinguish from warm-colored value-based anomalies.

### Files Modified

- `mae_anomaly/dataset_sliding.py` - Added anomaly types, category, injection methods
- `mae_anomaly/__init__.py` - Exported ANOMALY_CATEGORY
- `mae_anomaly/visualization/base.py` - Updated get_anomaly_colors()
- `mae_anomaly/visualization/best_model_visualizer.py` - Added visualization, fixed y-axis

### Documentation Updates

- CHANGELOG.md - This entry

---

## 2026-01-24 (Update 24): Per-Feature Min-Max Normalization

### Summary

Replaced data clipping (`np.clip(signals, 0, 1)`) with per-feature min-max normalization. This preserves relative anomaly magnitudes and eliminates boundary artifacts.

### Changes

#### 1. Added Normalization Function

**Modified Files**:
- `mae_anomaly/dataset_sliding.py`

**New Function**:
```python
def _normalize_per_feature(signals: np.ndarray) -> np.ndarray:
    """Per-feature min-max normalization to [0, 1] range.

    This is preferred over clipping because:
    1. Preserves relative magnitude of anomalies (spikes won't be capped)
    2. No artificial saturation at boundaries
    3. More realistic simulation of real-world data preprocessing
    """
    signals = signals.copy()
    for f in range(signals.shape[1]):
        min_val = signals[:, f].min()
        max_val = signals[:, f].max()
        if max_val - min_val > 1e-8:
            signals[:, f] = (signals[:, f] - min_val) / (max_val - min_val)
        else:
            signals[:, f] = 0.5
    return signals.astype(np.float32)
```

---

#### 2. Replaced Clipping with Normalization

**Locations Changed**:

| Method | Before | After |
|--------|--------|-------|
| `_generate_simple_normal_series()` | `np.clip(signals, 0, 1)` | `_normalize_per_feature(signals)` |
| `generate()` | `np.clip(signals, 0, 1)` | `_normalize_per_feature(signals)` |

---

#### 3. Why This Change?

| Aspect | Clipping | Min-Max Normalization |
|--------|----------|----------------------|
| Spike anomalies | Capped at 1.0 (info loss) | Full magnitude preserved |
| Boundary behavior | Flat saturation | Natural distribution |
| Relative magnitudes | Distorted | Preserved exactly |
| Real-world similarity | Artificial | Matches preprocessing |

---

### Documentation Updates

- **DATASET.md**: Added "Data Normalization" section, updated Safety Constraints table
- **CHANGELOG.md**: This entry

---

## 2026-01-24 (Update 23): Dataset Visualization Improvements

### Summary

Improved dataset visualization quality by using dedicated datasets for plotting (without anomaly contamination), added before/after comparisons at same window positions, and cleaned up redundant/misleading visualizations.

### Changes

#### 1. Added `inject_anomalies` Parameter to Generator

**Modified Files**:
- `mae_anomaly/dataset_sliding.py`

**New Parameter**:
```python
def generate(self, inject_anomalies: bool = True) -> Tuple[...]:
    """
    Args:
        inject_anomalies: If True (default), inject anomalies.
                          If False, return pure normal data.
    """
```

This allows visualization code to generate clean normal data for complexity feature demonstrations.

---

#### 2. Improved Dataset Visualizations

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:

| Function | Change |
|----------|--------|
| `plot_anomaly_generation_rules()` | Show only 1 example per anomaly type (was 2) |
| `plot_normal_complexity_features()` | Uses `inject_anomalies=False` for clean comparison |
| `plot_complexity_comparison()` | Uses `inject_anomalies=False` for clean comparison |
| `plot_complexity_vs_anomaly()` | **Completely redesigned**: Before/after comparison at same window position |
| `plot_dataset_statistics()` | **Removed** (hardcoded values were misleading) |

**New `plot_complexity_vs_anomaly()` Design**:
- Row 1: Complexity features (gray=before, blue=after) at same window position
- Row 2: Anomaly injection (gray=before, red=after) at same window position
- Allows clear visualization of what each feature/anomaly actually changes

---

#### 3. Stage 1 Visualization Cleanup

**Modified Files**:
- `mae_anomaly/visualization/experiment_visualizer.py`

**Changes**:

| Function | Change |
|----------|--------|
| `plot_metric_correlations()` | **Removed** (not useful for hyperparameter analysis) |
| `plot_parallel_coordinates()` | **Added interpretation guide** panel explaining how to read the plot |

---

### Documentation Updates

- **VISUALIZATIONS.md**: Updated tables and usage examples
- **CHANGELOG.md**: This entry

---

## 2026-01-24 (Update 22): Comprehensive Visualization Style Consistency

### Summary

Extended VIS_COLORS with additional semantic color keys and applied consistent styling across ALL visualization files, eliminating hardcoded color values.

### Changes

#### 1. Extended VIS_COLORS Constants

**Modified Files**:
- `mae_anomaly/visualization/base.py`

**New Color Keys Added**:
```python
VIS_COLORS = {
    # Primary data types (existing)
    'normal': '#3498DB',
    'anomaly': '#E74C3C',
    'disturbing': '#F39C12',
    'teacher': '#27AE60',
    'student': '#9B59B6',
    'total': '#2ECC71',
    # Region highlighting (NEW)
    'anomaly_region': '#E74C3C',
    'masked_region': '#F1C40F',
    'normal_region': '#27AE60',
    # Darker variants (NEW)
    'normal_dark': '#2980B9',
    'anomaly_dark': '#C0392B',
    'student_dark': '#8E44AD',
    # Detection outcomes (NEW)
    'true_positive': '#27AE60',
    'true_negative': '#3498DB',
    'false_positive': '#F39C12',
    'false_negative': '#E74C3C',
    # General purpose (NEW)
    'baseline': 'black',
    'reference': 'gray',
    'threshold': '#27AE60',
}
```

---

#### 2. Applied VIS_COLORS Across All Visualizers

**Modified Files**:
- `mae_anomaly/visualization/best_model_visualizer.py`
- `mae_anomaly/visualization/experiment_visualizer.py`
- `mae_anomaly/visualization/stage2_visualizer.py`
- `mae_anomaly/visualization/training_visualizer.py`
- `mae_anomaly/visualization/data_visualizer.py`
- `mae_anomaly/visualization/architecture_visualizer.py`

**Changes**:
- Replaced ALL hardcoded hex color values (e.g., `'#3498DB'`) with `VIS_COLORS['normal']`
- Replaced ALL hardcoded color names (e.g., `'red'`, `'yellow'`) with `VIS_COLORS` keys
- Added VIS_COLORS import to files that were missing it
- Used semantic color keys (e.g., `'anomaly_region'` for highlighting anomalies)

---

### Documentation Updates

- **VISUALIZATIONS.md**: Updated VIS_COLORS table with all new keys
- **CHANGELOG.md**: This entry

---

## 2026-01-24 (Update 21): Self-Distillation Training Improvements

### Summary

Added encoder gradient detachment for student decoder, configurable warm-up epochs, detailed learning curve visualization, and consistent color/marker scheme across all visualizations.

### Changes

#### 1. Encoder Gradient Detachment for Student Decoder

**Modified Files**:
- `mae_anomaly/model.py`

**Changes**:
- Student decoder now receives `.detach()`ed encoder output
- Encoder is only updated by teacher reconstruction loss
- Prevents student's conflicting objectives from corrupting encoder representations

**Implementation**:
```python
# In forward():
if self.config.use_student:
    student_latent = latent.detach()  # Detach encoder output
    student_output = self.student_decoder(student_latent)
```

---

#### 2. Configurable Teacher-Only Warm-up Epochs

**Modified Files**:
- `mae_anomaly/config.py`
- `mae_anomaly/trainer.py`
- `mae_anomaly/loss.py`

**New Parameter**:
- `teacher_only_warmup_epochs: int = 1` (default)

**Changes**:
- First N epochs train only teacher model (no discrepancy/student loss)
- Added `teacher_only` parameter to loss function
- Allows teacher to learn basic reconstruction before introducing discrepancy

---

#### 3. Detailed Learning Curve Visualization

**Modified Files**:
- `mae_anomaly/loss.py`
- `mae_anomaly/trainer.py`
- `mae_anomaly/visualization/best_model_visualizer.py`
- `scripts/visualize_all.py`

**New Metrics Tracked**:
- `train_teacher_recon_normal`: Teacher recon loss on normal samples
- `train_teacher_recon_anomaly`: Teacher recon loss on anomaly samples
- `train_student_recon_normal`: Student recon loss on normal samples
- `train_student_recon_anomaly`: Student recon loss on anomaly samples

**New Visualization**: `learning_curve.png`
- 2x3 grid showing detailed loss breakdown:
  - Teacher Reconstruction (Normal vs Anomaly)
  - Student Reconstruction (Normal vs Anomaly)
  - Discrepancy Loss (Normal vs Anomaly)
  - Normal Data: Teacher vs Student
  - Anomaly Data: Teacher vs Student
  - All Losses Combined

---

#### 4. Consistent Visualization Color/Marker Scheme

**Modified Files**:
- `mae_anomaly/visualization/base.py`
- `mae_anomaly/visualization/__init__.py`
- `mae_anomaly/visualization/best_model_visualizer.py`

**New Style Constants** (in `base.py`):
```python
VIS_COLORS = {
    'normal': '#3498DB',      # Blue for normal data
    'anomaly': '#E74C3C',     # Red for anomaly data
    'disturbing': '#F39C12',  # Orange for disturbing normal
    'teacher': '#27AE60',     # Green for teacher model
    'student': '#9B59B6',     # Purple for student model
    'total': '#2ECC71',       # Green for totals
}

VIS_MARKERS = {
    'discrepancy': 's',       # Square for discrepancy loss
    'teacher_recon': 'o',     # Circle for teacher reconstruction
    'student_recon': '^',     # Triangle for student reconstruction
    'total': 'D',             # Diamond for total/combined
}
```

**Applied to**:
- `plot_learning_curve()`: Full color/marker scheme
- `plot_discrepancy_trend()`: Consistent colors
- `plot_pure_vs_disturbing_normal()`: Consistent colors for bar charts

---

### Documentation Updates

- **ARCHITECTURE.md**: Added encoder gradient detachment and warm-up epochs documentation
- **VISUALIZATIONS.md**: Added VIS_COLORS/VIS_MARKERS documentation and learning_curve.png
- **CHANGELOG.md**: This entry

---

## 2026-01-23 (Update 20): Quick Search Dataset Configuration

### Changes
- `quick_length`: 100,000 → 200,000 timesteps
- `quick_train_ratio`: 0.3 → 0.2 (20% train, 80% test)
- "Anomaly Types" → "Anomaly Types (samples)" for clarity
- Removed sample count warning messages

### Files Modified
- `scripts/run_experiments.py`
- `mae_anomaly/dataset_sliding.py`

---

## 2026-01-23 (Update 19): Enhanced Dataset Statistics Display

### Changes
- Now displays **3 dataset views**: Train Set (Raw), Test Set (Raw), Test Set (Downsampled)
- Each view shows **Anomaly Types** distribution (per sample, not per region)
- Clearer output format for experiment monitoring

### Output Format
```
[Quick Dataset - Train Set (Raw)]
  - Pure Normal: X,XXX (XX.X%)
  - Anomaly: XXX (X.X%)
  Anomaly Types:
    - spike: XX
    - memory_leak: XX
    ...

[Quick Dataset - Test Set (Raw)]
  ...

[Quick Dataset - Test Set (Downsampled to 65%:15%:25%)]
  ...
```

### Files Modified
- `scripts/run_experiments.py`

---

## 2026-01-23 (Update 18): Train/Test Set Composition Fix

### Problem
- Only test set statistics were displayed, train set was missing
- Test set ratios were hardcoded as absolute counts (1200:300:500)

### Changes

#### 1. Train/Test Statistics Display
- Now shows both **Train Set (Raw)** and **Test Set (Raw)** statistics
- Train set: no downsampling, natural distribution (~5% anomaly from interval_scale)
- Test set: shows raw distribution + target ratio info

#### 2. Test Set Ratio-Based Downsampling
- **Before**: Hardcoded counts (1200:300:500 = 60:15:25)
- **After**: Ratio-based (65:15:25) scaled to `num_test_samples`
- Config now uses `test_ratio_*` instead of `test_target_*`

#### 3. Dataset Composition
| Split | Pure Normal | Disturbing | Anomaly | Downsampling |
|-------|-------------|------------|---------|--------------|
| Train | Natural | Natural | ~5% | None |
| Test | 65% | 15% | 25% | Yes |

### Files Modified
- `mae_anomaly/config.py`
- `scripts/run_experiments.py`

---

## 2026-01-23 (Update 17): Fix Anomaly Ratio in Quick Search

### Problem
- Previous fix scaled interval proportionally: `quick_interval_scale = base * (quick/full)`
- This reduced interval → more frequent anomalies → 19% anomaly ratio (too high)

### Solution
- Use same `interval_scale` for both quick and full search
- Anomaly ratio determined by interval_scale, not data length
- Consistent ~5% anomaly ratio regardless of dataset size

### Files Modified
- `scripts/run_experiments.py`

---

## 2026-01-23 (Update 16): Quick Search Dataset Size Increase

### Changes
- `quick_length`: 66000 → 100000 (more data for quick search)
- Warning threshold: 200 → 300 (suppress warnings when samples >= 300)

### Files Modified
- `scripts/run_experiments.py`
- `mae_anomaly/dataset_sliding.py`

---

## 2026-01-23 (Update 15): Reduce Periodicity in Complex Normal Data

### Summary

Improved normal data generation to be less strictly periodic, making anomaly detection more challenging and realistic.

### Changes

#### 1. Remove Hard Clipping
- **Before**: Normal data was clipped to `[0.05, 0.70]` range
- **After**: No clipping - natural value distribution
- Reason: Hard clipping made normal data unrealistically bounded and easy to classify

#### 2. Irrational Frequency Ratios
- **Before**: `freq2 ≈ freq1/10`, `freq3 ≈ freq1/50` (integer-like ratios)
- **After**: `freq2 = freq1/(π×[2.8-3.5])`, `freq3 = freq1/(π²×[1.5-2.5])`
- Reason: Integer ratios cause beat patterns to repeat; irrational ratios (π-based) prevent exact repetition

#### 3. Phase Jitter
- **New feature**: Slowly-varying phase offset added to sinusoidal components
- Parameters: `enable_phase_jitter=True`, `phase_jitter_sigma=0.002`, `phase_jitter_smoothing=500`
- Applied with decreasing weight per frequency: fast (1.0), medium (0.7), slow (0.4)
- Result: Even with same frequencies, patterns drift over time

### New NormalDataComplexity Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_phase_jitter` | True | Enable phase jitter |
| `phase_jitter_sigma` | 0.002 | Random walk step size |
| `phase_jitter_smoothing` | 500 | Smoothing window |

### Files Modified
- `mae_anomaly/dataset_sliding.py`
- `docs/DATASET.md`
- `docs/CHANGELOG.md`

---

## 2026-01-23 (Update 14): Experiment Configuration Updates

### Summary

Simplified num_patches options, doubled full search dataset, and improved warning thresholds.

### Changes

#### 1. `num_patches` Grid Reduction
- **Before**: `[10, 25, 50]` (3 values)
- **After**: `[10, 25]` (2 values)
- Reason: 50 patches = 2 timesteps per patch, too granular for effective pattern learning

#### 2. Full Search Dataset Size Doubled
- **Before**: `full_length = 220000`
- **After**: `full_length = 440000`
- Provides more training data for Stage 2 full search

#### 3. Warning Threshold for Sample Count
- Warnings now only appear when sample count < 200 (previously: any shortage)
- Reduces noise during quick searches with limited data

#### 4. Grid Combinations
- **Before**: 2×2×3×3×2×2×2×2×2 = 1152 combinations
- **After**: 2×2×2×3×2×2×2×2×2 = 768 combinations

### Files Modified
- `scripts/run_experiments.py`
- `mae_anomaly/dataset_sliding.py`
- `docs/ABLATION_STUDIES.md`
- `docs/VISUALIZATIONS.md`

---

## 2026-01-23 (Update 13): MAE Architecture Enhancements

### Summary

Added two new architecture parameters for standard MAE masking and separate mask tokens, along with experiment infrastructure improvements.

### New Parameters

#### 1. `mask_after_encoder` (config.py)
- **False (default)**: Mask tokens go through encoder (current behavior)
- **True**: Standard MAE - encode visible patches only, insert mask tokens before decoder

**Implementation**:
- Added `_encode_visible_only()` method: Encodes only visible patches
- Added `_insert_mask_tokens_and_unshuffle()` method: Inserts mask tokens at correct positions
- Modified `forward()` to support both modes

#### 2. `shared_mask_token` (config.py)
- **True (default)**: Single mask token shared between teacher/student
- **False**: Separate learnable mask tokens for teacher and student decoders

**Implementation**:
- Added `_get_mask_token(for_decoder)` method to retrieve appropriate token
- Separate `teacher_mask_token` and `student_mask_token` when not shared

### Experiment Changes

**Modified Files**:
- `scripts/run_experiments.py`
- `scripts/visualize_all.py`

**Parameter Grid Updates**:
```python
DEFAULT_PARAM_GRID = {
    # ... existing parameters ...
    'mask_after_encoder': [False, True],
    'shared_mask_token': [True, False],
}
# Total combinations: 2*2*3*3*2*2*2*2*2 = 1152
```

**Dataset Size Changes**:
- `quick_length`: 200000 → 66000 (1/3 reduction)
- `full_length`: 440000 → 220000 (1/2 reduction)
- `full_epochs`: fixed at 2

**Stage 2 Selection Updates**:
- Added `mask_after_encoder` (top 5 per value)
- Added `shared_mask_token` (top 5 per value)

**Output Cleanup**:
- Removed "Train: X, Test: Y" from Stage 1/2 headers (values were outdated)

### Documentation Updates

**Modified Files**:
- `docs/ARCHITECTURE.md`: Added MAE Masking Architecture and Mask Token Configuration sections
- `docs/ABLATION_STUDIES.md`: Added sections 8 (Mask After Encoder) and 9 (Shared Mask Token)

---

## 2026-01-23 (Update 12.2): Complexity Visualization

### Summary

Added 3 new visualization functions to explain NormalDataComplexity features.

### Changes

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**New Visualizations**:
1. `plot_normal_complexity_features()` - Shows each of 6 complexity features individually
2. `plot_complexity_comparison()` - Simple vs Complex normal data side-by-side
3. `plot_complexity_vs_anomaly()` - Why complexity features don't resemble anomalies

**Output Files**:
- `normal_complexity_features.png` - 6-panel feature explanation
- `complexity_comparison.png` - Simple vs Complex comparison
- `complexity_vs_anomaly.png` - Complexity vs Anomaly discrimination

---

## 2026-01-23 (Update 12.1): Experiment Integration

### Summary

- Exported `NormalDataComplexity` from `mae_anomaly` package
- Updated `run_experiments.py` to use complexity features by default
- Added `--no-complexity` CLI flag to disable complexity features

### Changes

**Modified Files**:
- `mae_anomaly/__init__.py`: Export `NormalDataComplexity`
- `scripts/run_experiments.py`: Use complexity by default, add CLI flag

**Usage**:
```bash
# Default: with complexity (recommended)
python scripts/run_experiments.py

# Without complexity (simple patterns)
python scripts/run_experiments.py --no-complexity
```

---

## 2026-01-23 (Update 12): Normal Data Complexity Features

### Summary

Added 6 configurable complexity features to make normal data more realistic and challenging for anomaly detection models. All features are designed to NOT be confused with anomaly patterns.

### Changes

#### 1. NormalDataComplexity Configuration

**Modified Files**:
- `mae_anomaly/dataset_sliding.py`

**Added**:
- `NormalDataComplexity` dataclass with on/off switches for each feature
- All features enabled by default, individually toggleable

```python
@dataclass
class NormalDataComplexity:
    enable_complexity: bool = True
    enable_regime_switching: bool = True
    enable_multi_scale_periodicity: bool = True
    enable_heteroscedastic_noise: bool = True
    enable_varying_correlations: bool = True
    enable_drift: bool = True
    enable_normal_bumps: bool = True
    # ... detailed parameters for each
```

---

#### 2. Six Complexity Features Implemented

| Feature | Description | Transition Time |
|---------|-------------|-----------------|
| **Regime Switching** | Different operational states | 1500 timesteps |
| **Multi-Scale Periodicity** | 3 overlapping frequencies | Continuous |
| **Heteroscedastic Noise** | Load-dependent variance | Continuous |
| **Time-Varying Correlations** | Slowly changing correlations | Period 15000 ts |
| **Bounded Drift (O-U)** | Mean-reverting random walk | Continuous |
| **Normal Bumps** | Small, gradual load increases | Gaussian envelope |

---

#### 3. Safety Constraints

All complexity features enforce strict constraints to distinguish from anomalies:

| Constraint | Value | Reason |
|------------|-------|--------|
| Transition time | >= 1000 ts | Anomalies are 3-150 ts |
| Value range | [0.05, 0.70] | Anomalies push to 0.7-1.0 |
| Bump magnitude | max 0.10 | Spike adds 0.3-0.6 |
| Bump duration | 100-300 ts | Spike is 10-25 ts |

---

#### 4. Documentation Updated

**Modified Files**:
- `docs/DATASET.md`

**Added**:
- New section "Normal Data Complexity Features"
- Detailed documentation for each feature
- Configuration examples
- Safety constraints explanation

---

### Usage

```python
from mae_anomaly.dataset_sliding import NormalDataComplexity, SlidingWindowTimeSeriesGenerator

# Full complexity (default)
complexity = NormalDataComplexity()

# Simple mode
complexity = NormalDataComplexity(enable_complexity=False)

# Custom
complexity = NormalDataComplexity(
    enable_regime_switching=True,
    enable_normal_bumps=False,
)

generator = SlidingWindowTimeSeriesGenerator(
    total_length=440000,
    complexity=complexity,
    seed=42
)
```

---

## 2026-01-23 (Update 11): Visualization Quality Improvements

### Changes

#### 1. Removed Redundant anomaly_types Visualization

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- Removed `plot_anomaly_types()` from `generate_all()` - redundant with `plot_anomaly_generation_rules()`
- The `anomaly_generation_rules.png` provides more informative visualization using actual dataset samples

---

#### 2. Improved feature_examples Visualization

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- Now displays ALL 8 features (was hardcoded to 5)
- Uses actual `FEATURE_NAMES` for labels (CPU_Usage, Memory_Usage, etc.)
- Dynamic subplot layout based on feature count

---

#### 3. Improved sample_types Visualization with Diverse Sampling

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- Added `select_diverse()` function to randomly sample from shuffled data
- Prevents showing overlapping/similar samples due to stride=10
- Ensures visual diversity in sample type comparison

---

#### 4. Improved patchify_modes as Conceptual Flow Diagrams

**Modified Files**:
- `mae_anomaly/visualization/architecture_visualizer.py`

**Changes**:
- Complete rewrite of `plot_patchify_modes()`
- Now shows conceptual processing pipeline with boxes and arrows
- Three modes clearly differentiated:
  - **CNN-First**: Input → CNN → Patchify → Embed
  - **Patch-CNN**: Input → Patchify → CNN (per patch) → Embed
  - **Linear (MAE)**: Input → Patchify → Linear Projection
- Removed meaningless bar chart comparison

---

#### 5. Improved discrepancy_trend Visualization

**Modified Files**:
- `mae_anomaly/visualization/best_model_visualizer.py`

**Changes**:
- Added standard deviation bands (mean ± std shading)
- Added zoomed view of last patch region (masked region)
- Added box plots showing discrepancy distribution by sample type
- Added statistics text box with mean ± std values
- More informative for analyzing masked region behavior

---

#### 6. Fixed METRIC_COLUMNS in Stage2Visualizer

**Modified Files**:
- `mae_anomaly/visualization/stage2_visualizer.py`

**Changes**:
- Added missing metrics to `METRIC_COLUMNS`:
  - `disturbing_roc_auc`, `disturbing_f1`, `disturbing_precision`, `disturbing_recall`
  - `quick_roc_auc`, `quick_f1`, `quick_disturbing_roc_auc`
  - `roc_auc_improvement`, `selection_criterion`, `stage2_rank`
- Prevents metrics from being incorrectly treated as hyperparameters

---

### Benefits

1. **Cleaner visualizations**: Removed redundant plots, improved clarity
2. **More informative**: All features shown with proper names
3. **Better diversity**: Sample type visualization shows varied data
4. **Conceptual clarity**: Patchify modes now explain the processing pipeline
5. **Statistical rigor**: Discrepancy trend includes uncertainty bands
6. **Correct hyperparameter analysis**: Metrics no longer appear as hyperparameters in Stage 2 plots

---

## 2026-01-23 (Update 10): Dynamic Hyperparameter and Configuration Management

### Changes

#### 1. Dynamic param_keys in visualize_all.py

**Modified Files**:
- `scripts/visualize_all.py`

**Before**: Hardcoded list of hyperparameter keys
```python
param_keys = ['masking_ratio', 'masking_strategy', 'num_patches', ...]
```

**After**: Dynamically extracted from experiment metadata or results
```python
if exp_data['metadata'] and 'param_grid' in exp_data['metadata']:
    param_keys = list(exp_data['metadata']['param_grid'].keys())
else:
    # Fallback: extract from results DataFrame
    param_keys = [c for c in columns if c not in metric_cols]
```

---

#### 2. Dynamic Hyperparameter Lists in stage2_visualizer.py

**Modified Files**:
- `mae_anomaly/visualization/stage2_visualizer.py`

**Changes**:
- Added `METRIC_COLUMNS` class constant for known metric columns
- Added `_get_hyperparam_columns()` helper method
- `plot_all_hyperparameters()`: Now uses dynamic hyperparameter detection
- `plot_hyperparameter_interactions()`: Dynamically generates interaction pairs
- `plot_best_config_summary()`: Uses dynamic hyperparams with fallback descriptions

---

#### 3. Dynamic Categorical Parameters in experiment_visualizer.py

**Modified Files**:
- `mae_anomaly/visualization/experiment_visualizer.py`

**Changes**:
- Added `_get_categorical_params()` helper method
- `plot_summary_dashboard()`: Uses dynamically detected categorical params
- `generate_all()`: Uses dynamic categorical params for comparisons

---

#### 4. Robust get_anomaly_type_info in base.py

**Modified Files**:
- `mae_anomaly/visualization/base.py`

**Changes**:
- `get_anomaly_type_info()` now handles unknown anomaly types gracefully
- Auto-generates descriptions for new anomaly types not in known_info dict
- Always includes all types from `ANOMALY_TYPE_NAMES`

---

### Benefits

1. **No manual updates needed**: Adding new hyperparameters to `DEFAULT_PARAM_GRID` automatically includes them in visualizations
2. **No sync issues**: New anomaly types are automatically handled with auto-generated descriptions
3. **Reduced maintenance**: Less hardcoded values = fewer places to update when configuration changes
4. **Better error handling**: Fallback mechanisms prevent crashes from missing data

---

## 2026-01-23 (Update 9): Visualization Module Modularization

### Changes

#### 1. Modular Visualization Package

**New Directory Structure**:
```
mae_anomaly/
└── visualization/
    ├── __init__.py              # Module exports
    ├── base.py                  # Common utilities, colors, data loading
    ├── data_visualizer.py       # DataVisualizer class
    ├── architecture_visualizer.py  # ArchitectureVisualizer class
    ├── experiment_visualizer.py # ExperimentVisualizer (Stage 1)
    ├── stage2_visualizer.py     # Stage2Visualizer class
    ├── best_model_visualizer.py # BestModelVisualizer class
    └── training_visualizer.py   # TrainingProgressVisualizer class
```

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py): Reduced from ~4900 lines to ~166 lines
- [mae_anomaly/visualization/](../mae_anomaly/visualization/): New modular package

**Benefits**:
- Cleaner, more maintainable code structure
- Each visualizer class in its own file
- Common utilities centralized in `base.py`
- Easy to extend with new visualizers

---

#### 2. Dynamic Color Management

**Modified Files**:
- `mae_anomaly/visualization/base.py`
- `mae_anomaly/visualization/best_model_visualizer.py`
- `mae_anomaly/visualization/training_visualizer.py`

**Changes**:
- Created `get_anomaly_colors()` function that dynamically generates colors for all anomaly types
- Created `SAMPLE_TYPE_COLORS` and `SAMPLE_TYPE_NAMES` constants
- Replaced all hardcoded color dictionaries with dynamic functions
- Colors now automatically adapt when anomaly types are added/removed

**Before** (hardcoded):
```python
colors = {
    'normal': '#3498DB',
    'spike': '#E74C3C',
    # ... manually maintained
}
```

**After** (dynamic):
```python
from mae_anomaly.visualization import get_anomaly_colors
colors = get_anomaly_colors()  # Automatically includes all anomaly types
```

---

#### 3. Dynamic plot_anomaly_generation_rules

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- `plot_anomaly_generation_rules()` now dynamically generates visualizations based on `ANOMALY_TYPE_NAMES`
- Uses actual dataset examples instead of synthetic simulation
- Automatically adapts grid size based on number of anomaly types
- Gets anomaly info (length_range, characteristics) from `ANOMALY_TYPE_CONFIGS`

---

#### 4. Usage Update

**New Import Pattern**:
```python
# Old (from script)
from scripts.visualize_all import DataVisualizer, load_best_model

# New (from module)
from mae_anomaly.visualization import (
    DataVisualizer,
    ArchitectureVisualizer,
    ExperimentVisualizer,
    Stage2Visualizer,
    BestModelVisualizer,
    TrainingProgressVisualizer,
    setup_style,
    load_best_model,
    get_anomaly_colors,
)
```

**Running visualizations** (unchanged):
```bash
python scripts/visualize_all.py  # Still works the same way
```

---

## 2026-01-23 (Update 8): Point Spike Duration Change and Visualization Fixes

### Changes

#### 1. Point Spike Duration Change

**Modified Files**:
- [mae_anomaly/dataset_sliding.py](../mae_anomaly/dataset_sliding.py)
- [docs/DATASET.md](DATASET.md)

**Changes**:
- Point spike duration: (1, 3) → **(3, 5)** timesteps
- Still the shortest anomaly type, but more detectable

```python
# Before
7: {'length_range': (1, 3), 'interval_mean': 4000}

# After
7: {'length_range': (3, 5), 'interval_mean': 4000}
```

---

#### 2. Visualization Color Map Update

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- Updated `plot_loss_by_anomaly_type()` colors: Added `point_spike` color
- Updated `plot_loss_scatter_by_anomaly_type()` colors: Fixed outdated anomaly type names (`noise`, `drift` → actual types)

**Before** (incorrect):
```python
colors = {
    'normal': '#3498DB',
    'spike': '#E74C3C',
    'memory_leak': '#F39C12',
    'noise': '#9B59B6',        # ← Wrong
    'drift': '#1ABC9C',         # ← Wrong
    'network_congestion': '#E67E22'
}
```

**After** (correct):
```python
colors = {
    'normal': '#3498DB',
    'spike': '#E74C3C',
    'memory_leak': '#F39C12',
    'cpu_saturation': '#9B59B6',
    'network_congestion': '#E67E22',
    'cascading_failure': '#1ABC9C',
    'resource_contention': '#16A085',
    'point_spike': '#E91E63',
}
```

---

#### 3. Anomaly-Type Performance Comparison Verification

**Existing Functions (Best Model)**:
- `plot_loss_by_anomaly_type()`: Loss distribution per anomaly type ✓
- `plot_performance_by_anomaly_type()`: Detection rate & mean score per type ✓
- `plot_loss_scatter_by_anomaly_type()`: Loss scatter per type ✓
- `plot_anomaly_type_case_studies()`: TP/FN examples per type ✓

**Existing Functions (Training Progress)**:
- `plot_anomaly_type_learning()`: Detection rate over epochs per type ✓

**Stage 1/2**: Designed for hyperparameter comparison, not anomaly-type analysis (by design)

---

## 2026-01-23 (Update 7): Point Spike Anomaly and Dataset Statistics

### Changes

#### 1. New Anomaly Type: Point Spike

**Modified Files**:
- [mae_anomaly/dataset_sliding.py](../mae_anomaly/dataset_sliding.py)
- [docs/DATASET.md](DATASET.md)

**New Anomaly Type**:
- **point_spike** (type 7): True point anomaly lasting only 3-5 timesteps
- **Unique characteristic**: 2+ random features spike simultaneously
- Makes threshold-based detection on individual features less effective

```python
# Point spike configuration
7: {'length_range': (3, 5), 'interval_mean': 4000}

# Injection logic
def _inject_point_spike(self, signals, start, end):
    # Select 2+ random features
    num_features_to_spike = np.random.randint(2, self.num_features + 1)
    features_to_spike = np.random.choice(self.num_features, num_features_to_spike, replace=False)
    # Apply spike magnitude +0.3 to +0.6 to each selected feature
```

---

#### 2. Dataset Statistics Output

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**New Feature**: When running experiments, dataset statistics are now printed:

```
[Quick Dataset Statistics - Test Set (Raw)]
Sample Types:
  - Pure Normal:       XXXX (XX.X%)
  - Disturbing Normal: XXX (XX.X%)
  - Anomaly:           XXX (XX.X%)
  - Total:             XXXX

Anomaly Types (region count):
  - spike: XX
  - memory_leak: XX
  - cpu_saturation: XX
  - network_congestion: XX
  - cascading_failure: XX
  - resource_contention: XX
  - point_spike: XX
```

---

#### 3. Visualization Code Update

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- `plot_anomaly_type_case_studies()`: Now dynamically uses `ANOMALY_TYPE_NAMES` instead of hardcoded list
- `plot_anomaly_type_learning()`: Now dynamically uses `ANOMALY_TYPE_NAMES` instead of hardcoded list
- Handles any number of anomaly types automatically

---

## 2026-01-23 (Update 6): Reduce Full Search Epochs

### Changes

- Changed `full_epochs` default from **3 to 2** for faster experimentation
- Updated files:
  - [scripts/run_experiments.py](../scripts/run_experiments.py): Function parameter and argparse default
  - [README.md](../README.md): Experiment settings table
  - [docs/ABLATION_STUDIES.md](ABLATION_STUDIES.md): Stage 2 description
  - [docs/VISUALIZATIONS.md](VISUALIZATIONS.md): Settings table

---

## 2026-01-23 (Update 5): Threshold Fix and Hypothesis Verification

### Changes

#### 1. Disturbing Normal Evaluation Fix

**Modified Files**:
- [mae_anomaly/evaluator.py](../mae_anomaly/evaluator.py)

**Problem**:
- Disturbing normal evaluation was using a **separate threshold** calculated only from pure_normal and disturbing_normal samples
- This was incorrect - should use the **global threshold** from the entire dataset

**Fix**:
- Now uses the global optimal threshold (calculated from all samples) for disturbing normal evaluation
- ROC-AUC is threshold-free, so no change needed there
- Precision/Recall/F1 now use the same threshold as overall evaluation

**Before** (incorrect):
```python
d_fpr, d_tpr, d_thresholds = roc_curve(disturbing_labels, disturbing_scores)
d_optimal_idx = np.argmax(d_tpr - d_fpr)
d_threshold = d_thresholds[d_optimal_idx]  # Separate threshold!
d_predictions = (disturbing_scores > d_threshold).astype(int)
```

**After** (correct):
```python
# Use GLOBAL threshold (from entire dataset)
d_predictions = (disturbing_scores > threshold).astype(int)
```

---

#### 2. Hypothesis Verification Visualization

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)
- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md)

**New Visualization**: `hypothesis_verification.png`

Verifies 4 hypotheses about why disturbing normal might outperform pure normal:

1. **H1: Anomaly Hint** - Does anomaly in window increase score?
   - Scatter plot of anomaly ratio vs total score

2. **H2: Transition Effect** - Does recent anomaly affect last patch?
   - Scatter plot of distance from anomaly to last patch vs score

3. **H3: Variance Analysis** - Does pure normal have higher variance?
   - Violin plot comparing score distributions

4. **H4: Classification Rates** - How do FP/TP rates compare with global threshold?
   - Bar chart of classification rates

---

#### 3. Quick Search Epoch Reduction

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)
- [README.md](../README.md)
- [docs/ABLATION_STUDIES.md](../docs/ABLATION_STUDIES.md)
- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md)

**Changes**:
- Stage 1 (Quick Search) epochs: 2 → **1**

**Rationale**:
- Single epoch sufficient for quick screening of 432 combinations
- Significantly reduces experiment time while maintaining ranking quality

**Updated Settings**:
| Stage | Epochs |
|-------|--------|
| Stage 1 (Quick) | 1 |
| Stage 2 (Full) | 3 |

---

## 2026-01-23 (Update 4): Estimated Time Display

### Changes

#### Time Estimation Feature

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Added time estimation based on first model training time
- Displays estimated time for Quick Search, Full Search, and Total
- Considers dataset size, epochs, and model count differences

**Output Format**:
```
>>> Estimated Time (based on 1st model: X.Xs) <<<
  Quick Search: XX분 (432 models × X.Xs)
  Full Search:  XX분 (~60 models × X.Xs)
  Total:        XX분
  (Quick remaining: XX분)
```

**Calculation**:
- Quick Search: `first_model_time × n_models`
- Full Search: `first_model_time × (full_train/quick_train) × (full_epochs/quick_epochs) × n_stage2_models`
  - `full_train/quick_train = 22,000/6,000 ≈ 3.67`
  - `full_epochs/quick_epochs = 3/2 = 1.5`

---

## 2026-01-23 (Update 3): Stage 2 Selection Reduction and Epoch Fine-tuning

### Changes

#### 1. Quick Search Epoch Reduction

**Changes**:
- Stage 1 epochs: 5 → **3**

**Rationale**:
- Further speed up quick search screening
- 3 epochs sufficient to identify promising configurations

---

#### 2. Stage 2 Selection Criteria Reduction

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Per-parameter top models: 10 → **5**
- Overall ROC-AUC top models: 30 → **10**
- Disturbing ROC-AUC top models: 20 → **5**
- Expected Stage 2 models: ~150 → **~50-70** (after deduplication)

**Rationale**:
- Faster full training while maintaining diverse coverage
- Still covers all parameter values with representative models

---

#### 3. Stage 2 Model Count Display

**Changes**:
- Added print statement showing Stage 2 model count during experiment execution
- Format: `>>> Stage 2 will train {N} models (from {M} Stage 1 combinations) <<<`

---

## 2026-01-23 (Update 2): Two-Stage Dataset and Epoch Configuration

### Changes

#### 1. Separate Datasets for Quick/Full Search

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Stage 1 (Quick Search): 200,000 timesteps, train_ratio=0.3 → ~6,000 train, ~14,000 test
- Stage 2 (Full Search): 2,200,000 timesteps, train_ratio=0.5 → ~110,000 train, ~110,000 test
- Test set always uses target_counts 1200:300:500 (total 2,000)

**Rationale**:
- Quick search needs fast iteration (small train set)
- Test set composition should be consistent across stages for fair comparison

---

#### 2. Epoch Count Reduction

**Changes**:
- Stage 1 epochs: 15 → **5**
- Stage 2 epochs: 100 → **30**

**Rationale**:
- Faster experimentation while maintaining reasonable training quality
- Quick search only needs to identify promising configurations

---

## 2026-01-23: Dataset Migration and Hyperparameter Grid Cleanup

### Major Changes

#### 1. Dataset Migration to SlidingWindowDataset

**Modified Files**:
- [mae_anomaly/dataset.py](../mae_anomaly/dataset.py) → Deprecated
- [mae_anomaly/dataset_sliding.py](../mae_anomaly/dataset_sliding.py) → Primary dataset
- [scripts/run_experiments.py](../scripts/run_experiments.py)
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- Replaced `MultivariateTimeSeriesDataset` with `SlidingWindowTimeSeriesGenerator` and `SlidingWindowDataset`
- New dataset features:
  - Continuous sliding window extraction from long time series
  - 8 correlated server metrics (CPU, Memory, DiskIO, Network, ResponseTime, ThreadCount, ErrorRate, QueueLength)
  - 6 realistic anomaly types: spike, memory_leak, cpu_saturation, network_congestion, cascading_failure, resource_contention
  - Three sample types: pure_normal, disturbing_normal, anomaly
  - Train/test split by time (no data leakage)

---

#### 2. Fixed Hyperparameters (margin, lambda_disc)

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)
- [scripts/visualize_all.py](../scripts/visualize_all.py)
- [mae_anomaly/config.py](../mae_anomaly/config.py)

**Changes**:
- `margin` and `lambda_disc` are now fixed at 0.5 (not in hyperparameter grid)
- Reduced hyperparameter search space from 2592 to 288 combinations
- Grid now includes: `masking_ratio`, `masking_strategy`, `num_patches`, `margin_type`, `force_mask_anomaly`, `patch_level_loss`, `patchify_mode`

**Rationale**:
- Preliminary experiments showed margin=0.5 and lambda_disc=0.5 perform well across configurations
- Reducing search space allows more thorough exploration of other hyperparameters

---

#### 3. Stage 2 Selection Criteria Update

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- New selection criteria for Stage 2 (150 diverse candidates):
  - Per-parameter top 10 (e.g., best for each masking_ratio value)
  - Overall top 30 by ROC-AUC
  - Top 20 by disturbing normal ROC-AUC
- Added `masking_strategy` to selection criteria

---

#### 4. num_features Updated (5 → 8)

**Modified Files**:
- [mae_anomaly/config.py](../mae_anomaly/config.py)
- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
- [docs/DATASET.md](../docs/DATASET.md)

**Changes**:
- Default `num_features` changed from 5 to 8
- All documentation diagrams updated to reflect (batch, 100, 8) dimensions

---

#### 5. Visualization Bug Fixes

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Fixes**:
- Updated `param_keys` to remove `margin` and `lambda_disc`
- Added `ANOMALY_TYPE_NAMES` import for visualization
- Fixed `plot_loss_by_anomaly_type` subplot grid (2x3 → dynamic for 7 anomaly types)
- Updated multiple places where margin/lambda_disc were referenced

---

### Documentation Updates

- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md): Updated num_features (5→8), all dimension examples
- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md): Updated param_keys, CSV columns, removed margin/lambda_disc
- [docs/DATASET.md](../docs/DATASET.md): Complete documentation for SlidingWindowDataset
- [docs/CHANGELOG.md](../docs/CHANGELOG.md): This entry

---

## 2026-01-22 (Update 3): Qualitative Case Studies and Late Bloomer Fix

### Major Changes

#### 1. Late Bloomer Algorithm Fix

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Issue Found**:
- Late bloomer analysis used final epoch's threshold for all epochs
- At epoch 0, the model hasn't learned, so all scores are similar
- Using the final threshold at epoch 0 produces incorrect predictions

**Fixes**:
- Implemented per-epoch optimal threshold calculation
- Late bloomers now correctly identified as samples that changed from incorrect to correct classification
- Added two categories:
  - **Late Bloomer Anomalies (FN→TP)**: Missed at start, detected at end
  - **Late Bloomer Normals (FP→TN)**: False alarm at start, correct at end

---

#### 2. Reconstruction Evolution Enhancement

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- Added Student reconstruction alongside Teacher (was Teacher-only)
- Added discrepancy visualization (|Teacher - Student|)
- Shows both reconstruction and discrepancy evolution over epochs
- Key insight: Discrepancy should increase in masked anomaly regions as training progresses

---

#### 3. Qualitative Case Study Visualizations

**New Files** (in `best_model/`):
- `case_study_gallery.png`: Representative TP/TN/FP/FN examples with detailed analysis
- `anomaly_type_case_studies.png`: Per-anomaly-type TP vs FN comparison
- `feature_contribution_analysis.png`: Which features drive anomaly detection
- `hardest_samples.png`: Analysis of hardest-to-detect samples (lowest margin FN/FP)

**New Files** (in `training_progress/`):
- `late_bloomer_case_studies.png`: Detailed time series evolution for late bloomers

**New Methods**:
- `BestModelVisualizer.plot_case_study_gallery()`: Median examples for each outcome
- `BestModelVisualizer.plot_anomaly_type_case_studies()`: Per-type TP/FN comparison
- `BestModelVisualizer.plot_feature_contribution_analysis()`: Feature importance ranking
- `BestModelVisualizer.plot_hardest_samples()`: Hardest FN and FP analysis
- `TrainingProgressVisualizer.plot_late_bloomer_case_studies()`: Detailed late bloomer evolution

---

### Documentation Updates

- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md): Added new visualizations and updated descriptions
- [docs/CHANGELOG.md](../docs/CHANGELOG.md): This changelog entry

---

## 2026-01-22 (Update 2): Visualization Enhancements and Consistency Fixes

### Major Changes

#### 1. Visualization Data Consistency Fix

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Issue Found**:
- `visualize_all.py` used different evaluation settings than `run_experiments.py`:
  - `anomaly_ratio=0.3` instead of `config.test_anomaly_ratio=0.25`
  - Random masking instead of fixed last-patch masking
  - MAE (absolute error) instead of MSE (squared error)

**Fixes**:
- Changed `anomaly_ratio` to use `config.test_anomaly_ratio` (0.25)
- Changed `collect_predictions()` and `collect_detailed_data()` to use same evaluation as `evaluator.py`:
  - Fixed mask: `mask[:, -config.mask_last_n:] = 0`
  - Forward with `masking_ratio=0.0` and explicit mask
  - MSE computation: `((output - input) ** 2).mean(dim=2)`

---

#### 2. New Data Visualizations

**New Files**:
- `data/anomaly_generation_rules.png`: Detailed rules for each anomaly type
- `data/feature_correlations.png`: Feature correlation matrix and generation rules
- `data/experiment_settings.png`: Experiment settings summary (Stage 1/2)

**Changes**:
- Added `plot_anomaly_generation_rules()`: Shows how each anomaly type is generated
- Added `plot_feature_correlations()`: Shows inter-feature correlations
- Added `plot_experiment_settings()`: Summarizes experiment settings for reproducibility

---

#### 3. Stage 2 Per-Hyperparameter Visualizations

**New Files** (in `stage2/`):
- `hyperparam_masking_ratio.png`
- `hyperparam_num_patches.png`
- `hyperparam_margin.png`
- `hyperparam_lambda_disc.png`
- `hyperparam_margin_type.png`
- `hyperparam_force_mask_anomaly.png`
- `hyperparam_patch_level_loss.png`
- `hyperparam_patchify_mode.png`
- `hyperparameter_interactions.png`
- `best_config_summary.png`

**Changes**:
- Added `plot_hyperparameter_impact()`: Per-hyperparameter detailed analysis
- Added `plot_all_hyperparameters()`: Generate all per-hyperparameter plots
- Added `plot_hyperparameter_interactions()`: Interaction heatmaps
- Added `plot_best_config_summary()`: Best config with Korean descriptions

---

#### 4. Best Model Analysis Improvements

**New Files**:
- `best_model/pure_vs_disturbing_normal.png`: Pure Normal vs Disturbing Normal comparison
- `best_model/discrepancy_trend.png`: Discrepancy trend analysis across time steps

**Changes**:
- Added `plot_pure_vs_disturbing_normal()`: Detailed comparison of sample types
- Added `plot_discrepancy_trend()`: Time-step level discrepancy analysis

---

### Documentation Updates

- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md): Complete rewrite with all new visualizations

---

## 2026-01-22: Project Cleanup and Patchify Mode

### Major Changes

#### 1. Patchify Mode Feature

**Modified Files**:
- [mae_anomaly/model.py](../mae_anomaly/model.py)
- [mae_anomaly/config.py](../mae_anomaly/config.py)

**Changes**:
- Added `patchify_mode` configuration option with 2 modes:
  - `linear`: Direct patchify + linear projection (MAE original style)
  - `patch_cnn`: Patchify first, then CNN per patch (no cross-patch leakage)
- Updated model to support both patchify modes

**Benefits**:
- Flexibility to test different patchification strategies
- `patch_cnn` mode prevents information leakage across patches
- Better control over local feature extraction

---

#### 2. Visualization Refactoring

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py) (NEW)
- [scripts/run_experiments.py](../scripts/run_experiments.py) (refactored)

**Changes**:
- Moved all visualization code from `run_experiments.py` to dedicated `visualize_all.py`
- Created 5 visualization classes:
  - `DataVisualizer`: Data distribution and sample visualization
  - `ArchitectureVisualizer`: Model architecture diagrams
  - `ExperimentVisualizer`: Stage 1 (Quick Search) results
  - `Stage2Visualizer`: Stage 2 (Full Training) results
  - `BestModelVisualizer`: Best model analysis
- `run_experiments.py` now only handles training and saves results to CSV
- Fixed shape mismatch bugs when data has fewer than 10 rows

---

#### 3. Project Cleanup

**Deleted Files**:
- `tests/integration/` (obsolete test files using old module)
- `REFACTORING_COMPLETE.md`, `REFACTORING_PLAN.md`
- `docs/bugfixes/`, `docs/implementation/`, `docs/analysis/`

**Updated Files**:
- [README.md](../README.md) - Complete rewrite for current structure
- [examples/basic_usage.py](../examples/basic_usage.py) - Updated imports and examples
- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - Added patchify_mode documentation
- [docs/ABLATION_STUDIES.md](../docs/ABLATION_STUDIES.md) - Added patchify_mode experiments

---

### Documentation Updates

- README.md now reflects current project structure
- Added patchify mode examples in basic_usage.py
- Architecture documentation includes all 3 patchify modes
- Ablation studies documentation includes patchify_mode experiments

---

## 2026-01-14: Architecture and Training Updates

### Major Changes

#### 1. Architecture: Transformer → 1D-CNN + Transformer Hybrid

**Modified Files**:
- [mae_anomaly/model.py](../mae_anomaly/model.py)

**Changes**:
- Added 2-layer 1D-CNN before Transformer:
  - Conv1: num_features (5) → d_model//2 (32), kernel=3
  - Conv2: d_model//2 (32) → d_model (64), kernel=3
  - BatchNorm + ReLU after each layer
- Updated patch embedding to work with CNN features:
  - New method: `patchify_cnn()` for CNN output
  - Processes (batch, d_model, seq_length) → (batch, num_patches, d_model*patch_size)
- Updated forward pass:
  - Input → CNN → Patchify → Transformer
  - CNN adds ~6,912 parameters
  - Total parameters: ~513K (was ~505K)

**Benefits**:
- Better local feature extraction
- Combines CNN (local) + Transformer (global) strengths
- Improved representation learning

---

#### 2. Best Model Selection: Match Evaluation Criterion

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Best model selection now matches evaluation metric:
  - **Baseline**: Uses total loss (reconstruction + discrepancy)
  - **TeacherOnly**: Uses teacher reconstruction loss
  - **StudentOnly**: Uses student reconstruction loss
- Model selection based on ROC-AUC during grid search

**Rationale**:
- Previous: All experiments used reconstruction loss for model selection
- Issue: Baseline evaluation uses discrepancy, but model selected on reconstruction
- Fix: Model selection criterion now matches what we optimize for in each ablation

---

### Documentation Updates

#### Created Files:
1. [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
   - Complete architecture documentation
   - Component-by-component breakdown
   - Parameter counts and pipeline diagram
   - Design rationale and comparisons

#### Updated Files:
1. [docs/ABLATION_STUDIES.md](../docs/ABLATION_STUDIES.md)
   - Added architecture overview section
   - Updated best model selection notes
   - Clarified evaluation criteria for each ablation

---

---

## Previous Updates (2026-01-14)

### Data and Ablation Updates

1. **Data Size Increase** (5x):
   - Train: 1,000 → 5,000 samples
   - Test: 300 → 1,500 samples

2. **Best Model Checkpointing**:
   - Track training loss during epochs
   - Save model at lowest loss epoch
   - Restore best model after training

3. **Masking Strategy Ablation**:
   - Added patch masking (same-time across features)
   - Added feature-wise masking (independent per feature)
   - Tests importance of cross-feature temporal coherence

4. **Removed Redundant Ablations**:
   - Removed NoDiscrepancy (redundant with TeacherOnly)
   - Removed NoMasking (replaced with more informative experiments)

5. **Cleanup**:
   - Deleted old experiment results
   - Removed unused folders
   - Regenerated visualizations

---

## File Structure

```
mae_anomaly/
├── model.py            [MODIFIED] - Added 1D-CNN layers, patchify modes
├── config.py           [MODIFIED] - Added patchify_mode, margin_type options
├── loss.py             - Self-distillation loss
├── trainer.py          - Training loop
├── evaluator.py        - Evaluation metrics
└── dataset.py          - Synthetic dataset generation

scripts/
├── run_experiments.py  - Two-stage grid search experiment runner
└── visualize_all.py    - Comprehensive visualization generator

docs/
├── ARCHITECTURE.md     - Architecture documentation
├── ABLATION_STUDIES.md - Ablation study documentation
├── VISUALIZATIONS.md   - Visualization guide
└── CHANGELOG.md        - This file
```

---

## Summary

**Total Changes**:
1. ✅ Transformer → 1D-CNN + Transformer hybrid architecture
2. ✅ Best model selection matches evaluation criterion
3. ✅ Comprehensive architecture documentation
4. ✅ Testing and verification scripts

**Status**: All changes implemented, tested, and documented.

**Next Steps**: Run full experiments with updated architecture and training logic.
