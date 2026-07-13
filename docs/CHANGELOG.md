# Changelog

## 2026-07-13: 자원누적 근본원인 완결 — graceful backstop + startup self-heal reaper (재부팅 불필요화)

**증상**: 캠페인이 길어질수록 하드웨어 부담↑ → 반복 재부팅. **근본원인(4-에이전트 적대 워크플로우로 확정, high-conf)**:
- **1차(RAM staircase, 역사적 주범)**: launcher가 `run_base --no-wait` → run_base가 detached bg-worker를 join 없이 종료 → hang(특히 Timer가 못 깨는 GIL-holding C hang)하거나 sweep 중이던 워커가 **init로 reparent된 영구 고아**(각 0.5~1.7GB+~35스레드). throttle(:4452)은 run_base별 로컬 리스트라 이전 조건 고아를 못 셈. `use_reconsnr_es_halt`가 증폭기(조기종료→데이터셋 빨리끝→3코어에 bg-worker 몰림→N배 느려짐→trailing 고아↑). ~17개 쌓이면 swap-thrash. 12번 재부팅 중 대부분이 backstop(7/12) 이전 = 이게 역사적 driver.
- **2차(loky semaphore)**: 337e3cc backstop의 `os._exit`+SIGKILL이 **워커 자신의 resource_tracker까지 죽여** sem 정리를 막음 → 발동당 ~5 sem 누수(재현됨). 정상운영·graceful 종료는 누수 0(실증).

**수정(전부 post-training/startup 경로 → 학습 compute/결과/속도 무영향)**:
1. **graceful backstop** (`_bg_hang_backstop`): resource_tracker는 **살려두고**(EOF로 sem 정리) 나머지 descendant(손자 포함) pool 워커만 SIGTERM→grace(`MAE_BG_BACKSTOP_GRACE`=5s)→SIGKILL 후 os._exit. → 발동당 sem 누수 0 + 손자 고아 방지 + ppid==self race 해소.
2. **startup loky-sem reaper** (`_startup_self_heal`): `/dev/shm/sem.loky-<pid>-*` 중 owner pid가 죽은 것만 unlink(교차-캠페인 self-heal). mp sem(랜덤명)은 안 건드림.
3. **startup age-gated orphan reaper**: ppid==1 + spawn/bg-worker 시그니처 + age > `MAE_BG_WORKER_TIMEOUT`+300s인 고아 프로세스만 SIGTERM→SIGKILL. Timer가 못 잡는 GIL-hang 고아를 **외부에서** 회수. age gate로 직전 조건의 정상 trailing 워커(ppid=1이지만 수초~수분)는 보존.

**검증**: (1) graceful backstop 실제 bg-worker 발동 end-to-end — 종료·VUS pool 자식0·ppid1 고아0·**sem 누수0**(구 backstop=5 대비); 정상완료 경로도 clean(자식 정리, sem0). (2) age 계산이 `ps -o etimes`와 정확 일치 → fresh 워커 오살 없음. (3) loky reaper: dead-pid 삭제/live-pid 보존. (4) py_compile OK. 효과: 매 run_base startup이 직전 잔여물 self-heal → 캠페인 누적 불가.

## 2026-07-12: bg-worker HANG BACKSTOP — 고아 eval/viz 워커 무한누적 근본원인 수정 (정상 워커 byte-identical)

**근본원인**: 모든 launcher가 `run_base --no-wait`로 실행 → run_base가 background eval+viz+VUS-sweep 워커(`_cpu_eval_viz_worker`, dataset당 1~2개, `ctx.Process` **non-daemon**, ~35 OpenBLAS 스레드 + 큰 RSS)를 **join/terminate 없이 버리고 종료**. hung/deadlock 워커는 init로 재부모돼 **영원히 고아**로 남고, "max 10" throttle은 run_base별 로컬이라 이전 실험 고아를 세지 않음 → 긴 큐에서 누적 → **재부팅으로만 해소**(사용자 반복 재부팅 증상).

**수정 (`scripts/run_base_experiments.py` `_cpu_eval_viz_worker`)**: body를 감싸는 **hang-backstop daemon Timer** 추가. `MAE_BG_WORKER_TIMEOUT`(기본 **1800s=30min**, 정상 워커 ~2-10min의 3배↑ 여유) 초과 시 자식(VUS pool)까지 `SIGKILL` 후 `os._exit` → 고아 없이 자멸. body 완료(성공/실패) 즉시 `finally`에서 **cancel → 정상 워커는 타이머 무발동 → 출력 byte-identical**. `os.setsid` 미사용(=run_base 프로세스그룹 유지 → 수동 pause의 PGID-kill이 기존대로 워커까지 reap). Timer 스레드는 main이 C-ext(BLAS/torch)에 갇혀도 발동(GIL 짧게만 점유), `os._exit`는 어느 스레드에서도 종료.

**검증**: (1) standalone KILL-path — 타임아웃 시 자식 정리(고아 0) + 자기 종료 확인. (2) standalone CANCEL-path — 완료 시 cancel → 무발동·생존. (3) 정상 워커 무영향은 구조적 보장(cancelled Timer=무효과). (4) end-to-end 실파이프라인 확인. py_compile OK. 성능/속도 무영향(30min 백스톱은 정상 워커에 절대 도달 안 함).

## 2026-07-11: `use_reconsnr_es_halt` — POST-warmup recon_snr early-stop이 실제로 학습을 halt (기본 False = byte-identical)

**동기**: 논문의 recon_snr ES 기준(post-warmup·EMA α=0.2·best-so-far·patience=2)은 지금까지 30ep 완주 후 **사후 선택**이었다. 실제로 그 epoch에서 학습을 멈추게 하여 compute를 절약(ES=16 셀은 e18에서 정지, 12ep 절약)하고 논문이 주장하는 early-stop을 코드로 구현.

**변경 (additive · 기본 False = byte-identical)**:
- `mae_anomaly/config.py`: `use_reconsnr_es_halt: bool = False` + `reconsnr_es_halt_alpha: float = 0.2` + `reconsnr_es_halt_patience: int = 2`. 기존 `teacher_warmup_*` early-stop(=warmup 단축)과 별개.
- `mae_anomaly/trainer.py`: epoch 루프 끝(`post_epoch_callback` 뒤, 현재 epoch 산출물 저장 후)에 게이트 블록 추가. `self.history['train_recon_snr']` 위에서 사후 `es()`와 **동일한** 스트리밍 기준을 돌려 patience 소진 epoch에서 `break`. recon_snr None(train 이상라벨 없음: blind/excised)은 skip → 트리거 안 됨(완주). False면 블록 전체 미실행.

**왜 결과가 사후방식과 동일한가**: 사후 `es()`도 첫 patience-소진에서 break해 그 이후 epoch을 무시한다. halt는 바로 그 지점에서 학습을 멈추므로 ES epoch(best_ep)과 그 모델 상태·scores NPZ가 불변. 사후 VUS 스윕은 저장된 **전 epoch**(`glob('epoch_*_scores.npz')`)을 계산하므로 halt로 18ep만 저장돼도 ES epoch(16)의 VUS가 채워진다.

**검증**: (1) 스트리밍 기준이 사후 `es()`와 동일 ES epoch — 실데이터 5시드×4셀 20/20 일치. (2) **end-to-end 수용검증**: PSM seed40 baseline+halt 실행 → e18 halt(12ep skip), e16의 8개 지표(pak/vus_pr/vus_roc/aff/prc/f1_t/pa_0_f1/r_based_f1_ar) **전부 기존 30ep 런과 소수 8자리까지 bit-identical**. (3) flag off → 블록 미실행 byte-identical. py_compile OK.

## 2026-07-04: `train_label_mask_exclude` — 마스킹된 anomaly를 unlabeled로 두지 않고 학습에서 **제거** (기본 False = no-op)

**동기**: group-random 마스킹 스윕(unlab10r/25r/50r/75r)의 짝 실험. 마스킹은 선택된 anomaly 타임포인트의 **라벨만 0으로** 가리고 데이터는 unlabeled로 학습에 남긴다. 이 옵션은 **똑같이 선택된** 타임포인트를 아예 splice 제거해, "가린 이상치를 unlabeled 데이터로 유지" vs "완전 제거"를 동일 타임포인트에서 분리 비교한다. frac=1.0(전 그룹)은 전 anomaly 제거 ≡ 기존 `exclanom`(`train_exclude_anomaly_segments=True`)과 동일하므로 재실행 제외.

**변경 (additive · 기본 False = byte-identical)**:
- `mae_anomaly/config.py`: `train_label_mask_exclude: bool = False`. `train_label_mask_frac>0`과 함께일 때만 의미. `train_exclude_anomaly_segments`(frac 무관 전 anomaly 제거)와 상호배타.
- `mae_anomaly/dataset_sliding.py`: 기존 마스킹 블록을 `_sel`(선택 인덱스) 계산 후 분기하도록 리팩터 — exclude=False면 `point_labels[_sel]=0`(기존과 byte-identical), True면 `_mask_exclude_keep` 마스크를 만들어 run_boundaries 설정 뒤 splice(선택분 제거 + junction에 boundary 삽입 + 생존 anomaly_regions를 spliced 좌표로 remap). 미선택 그룹은 TRUE 라벨 유지 → recon_snr/GRL/anomaly_loss가 그대로 관측. 기존 `train_exclude_anomaly_segments`(전 anomaly 제거) 경로와 별개.
- `scripts/run_base_experiments.py`: 학습 train_dataset에 `train_label_mask_exclude` 전달(getattr default False).

**검증**: (1) **무회귀** — 실데이터 4개(PSM/WaDi_A1/A2/SWaT) 전 frac에서 리팩터 전후 마스킹 byte-identical. (2) **no-op** — exclude=True+frac=0 → baseline 완전 동일, test split 무영향. (3) **의도 동작** — 합성 SlidingWindowDataset 통합테스트로 선택분만 splice, 미선택 TRUE 라벨 유지, boundary 삽입, region remap 정확, frac=1.0≡전량 제거 확인. (4) **선택 동일성** — exclude가 제거하는 지점 = mask가 가리는 지점(동일 RandomState(42)). py_compile OK. official 큐 exclude_grouprandom(excl10r/25r/50r/75r; 100%=exclanom 제외)로 추가(seeds→discsnr→odofffeat 뒤).

## 2026-07-02: `train_label_mask_random` — TRAIN 라벨 마스킹을 시간순-마지막 대신 랜덤으로 (그룹-단위, 2026-07-03 개정; 기본 False = no-op)

**동기**: `train_label_mask_frac`(anomaly 타임포인트의 시간순 후반 frac unlabel)의 랜덤 대조군 — 후반부 편향 없이 무작위로 unlabel해 위치 효과와 분리. 산발적 point-단위 대신 **연속 구간(그룹) 단위**로 마스킹해야 의미가 있어 group-level로 개정.

**변경 (additive · 기본 False = byte-identical)**:
- `mae_anomaly/config.py`: `train_label_mask_random: bool = False` + `train_label_mask_group_size: int = 100`. `train_label_mask_frac>0`과 함께일 때만 의미.
- `mae_anomaly/dataset_sliding.py`: 기존 `frac>0` 블록 안에서 True면 anomaly 타임포인트를 `group_size`(=100)-ts 빈(`idx // group_size`)으로 묶고 `RandomState(42)`로 **랜덤 frac의 그룹**을 골라 그 그룹의 모든 anomaly point를 unlabel(고정 시드 → 재현 가능, 학습 RNG 미교란). False면 기존 시간순-마지막 경로 그대로. frac=1.0 ⇒ 전 그룹 ⇒ 전부(back과 동일).
- `scripts/run_base_experiments.py`: 학습 train_dataset에 `train_label_mask_random`/`train_label_mask_group_size` 전달(getattr default False/100).

**검증**: 기본 False + frac=0 이중 default-off → 기존 전 실험 byte-identical. py_compile OK. official 큐 unlab_random(frac 0.10/0.25/0.50/0.75; frac=1.0=unlab100과 동일이라 제외)으로 추가.

## 2026-06-24: `masking_strategy='feature_wise'` — feature-wise 마스킹 학습 어블레이션 (기본 'patch' = no-op)

**동기**: 기존 patch 마스킹(시간 패치 토큰을 통째로 마스킹) 대신, **patch-feature 셀 단위**로 raw 입력을 마스킹하고 그 마스킹된 feature 셀에서만 recon/discrepancy loss를 계산하는 학습 어블레이션.

**변경 (additive · 기본 'patch' = byte-identical)**:
- `mae_anomaly/config.py`: `masking_strategy: str = 'patch'` 추가 (`'patch'`|`'feature_wise'`).
- `mae_anomaly/model.py`: `feature_wise_masking()` 신규 + forward에 `feature_wise_training`(training ∧ feature_wise ∧ mask=None) 분기. **'patch'면 `feature_mask_token` 파라미터를 아예 생성하지 않음** → 모델 파라미터화 불변. 평가는 기존 patch leave-one-out 마스크 그대로 → **official scoring/inference 불변**.
- `mae_anomaly/loss.py`: `loss_mask`((B,L,F) keep 마스크) 인자 추가 → feature 셀 단위 loss/denominator. `loss_mask=None`(기존 모든 경로)이면 원래 식과 **동일(byte-identical)**.
- `mae_anomaly/trainer.py`: `masking_strategy` 검증(+`force_mask_all_anomaly`와 비호환 가드) + `_feature_loss_mask`를 loss에 전달.

**검증**: 기본 'patch' default no-op(파라미터·loss 모두 byte-identical), py_compile OK. official 큐 odoff_featurewise 어블레이션으로 추가.

## 2026-06-30: `train_label_mask_frac` 의미 수정 — 타임라인 위치 → anomaly 타임포인트 순위 기반

**버그**: 기존 구현은 train **타임라인(전체 timestep)의 뒤쪽 frac 구간** 라벨을 통째로 0으로 만들었다(`point_labels[int(len*(1-frac)):]=0`). 그러나 본 base4 데이터셋은 모두 `[전부-정상 prefix] + [attack 시작부]` 구조라 **train anomaly가 전부 타임라인 맨 뒤 few %에 몰려** 있다 → 가장 작은 frac(unlab10=뒤 10%)에서 이미 WaDi_A1은 100%, PSM은 ~전부의 anomaly 라벨이 제거되어 **unlab10≈unlab100(비트-동일)**. 의도(anomaly 타임포인트의 시간순 후반 N%만 unlabel)와 불일치.

**수정 (국소·기본 no-op)**:
- `mae_anomaly/dataset_sliding.py`: frac>0 블록을 **anomaly 순위 기반**으로 — `anom_idx=np.nonzero(point_labels)[0]`(시간순 train anomaly 타임포인트), `k=round(frac*len(anom_idx))`, `point_labels[anom_idx[-k:]]=0`(최신 k개 anomaly만 unlabel). normal 라벨은 미변경. `.copy()` 유지 → 공유배열/test/train-진단 TRUE 라벨 보존. frac=1.0 ≡ `blind_train_labels`.
- `mae_anomaly/config.py`: 주석을 "anomaly 타임포인트 후반 frac unlabel"로 동기화.

**효과**: 의도된 graded sweep 복원 — WaDi_A1 unlab10/25/50/75/100 = 잔존 anomaly 라벨 90/75/50/25/0%(검증). frac=0=byte-identical, frac=1.0=blind 동치, test/normal 미변경.

**실험**: 기존 unlab10~100 official 결과(버그 버전) 5개 삭제, `scripts/run_official_unlab_rankfix_after.py`로 현재 캠페인 종료 후 fresh 재실행 큐잉.

## 2026-06-24: `train_exclude_anomaly_segments` — TRAIN에서 anomaly 구간 제거(splice) 옵션

**동기**: "271 그대로지만 label 달린(anomaly) 데이터를 train에서 제외"하는 실험. 마스킹(label→0)이 아니라 **timestep 자체를 제거**하고, 제거로 생긴 이음새에 run boundary를 넣어 윈도가 서로 다른 시점을 잇지 않게.

**변경 (국소·기본 no-op)**:
- `mae_anomaly/config.py`: `train_exclude_anomaly_segments: bool = False`.
- `mae_anomaly/dataset_sliding.py`: train-split의 anomaly_regions 필터 직후, flag면 `keep=(point_labels==0)`로 anomaly timestep을 splice(`self.signals[keep]`/`point_labels[keep]` = **fancy-index 새 배열** → 공유 signals/point_labels 미변조). run boundaries = removed-gap junction(`np.diff(orig)!=1`) + 기존 boundary remap(`searchsorted`). anomaly_regions=[]. 정규화 scaler는 분할 전 fit이라 불변 → **test 정규화 무관**.
- `scripts/run_base_experiments.py`: 학습 train_dataset(2886) **한 곳에만** 전달.

**검증**: 기본 False=byte-identical. synthetic으로 splice(800→720, anomaly 0, boundary=junction+remap), 공유배열 미변조, test split no-op, official config 반영 확인. official 큐 8th(`train_exclude_anomaly_segments=True`)로 추가.

## 2026-06-24: `train_label_mask_frac` — TRAIN 라벨 부분 마스킹(unlabeled) 옵션

**동기**: "271 그대로지만 train 라벨을 unlabeled로 두는" 실험 + 마스킹 비율 선택. 100% 마스킹은 기존 `blind_train_labels=True`로 가능했으나(전부/전무 bool), **부분(%) 마스킹은 미구현**이었다.

**변경 (국소·기본 no-op)**:
- `mae_anomaly/config.py`: `train_label_mask_frac: float = 0.0` 추가 (0=off, 1.0=전부, 0.5=back 50%).
- `mae_anomaly/dataset_sliding.py`: `SlidingWindowDataset` train-split 블록에서 frac>0이면 train point_labels의 **뒤쪽 frac 구간만 0**으로. **`.copy()` 사용** → 공유 point_labels 배열 미변조 → test split + `best_epoch_train_scores`(train-추론 진단)는 **TRUE 라벨 유지**. frac=1.0 ≡ blind_train_labels.
- `scripts/run_base_experiments.py`: 학습용 train_dataset(2885) **한 곳에만** `train_label_mask_frac` 전달. test/train-진단/bg-worker/viz 경로 미전달 → default 0.

**의미**: 마스킹된 구간은 anomaly 라벨(1) 소멸 → GRL/anomaly 감독이 그 구간에서 비활성(라벨 0=정상 취급, label-free). frac=1.0이면 anomaly 표본 0 → train recon/disc SNR은 None(미정의).

**검증**: 기본값 0이라 기존 전 실험 byte-identical. synthetic 테스트로 frac 0/0.5/1.0 정확 + 입력배열 미변조 확인. official 큐 6th(`train_label_mask_frac=1.0`)·7th(`=0.5`)로 추가.

## 2026-06-24: train `disc_snr` per-epoch 로깅 추가 (recon_snr 미러링, early-stop 후보)

**동기**: recon_snr(teacher recon 분리도)는 train history에 있었으나 **disc_snr(discrepancy 분리도, 논문 핵심 기여)** 는 미저장이라 disc_snr 기반 early-stopping을 보려면 매번 재추론해야 했다. freezeenc 재추론 분석 결과 disc_snr early-stop(ema0.2/pat2)이 recon_snr보다 우수(mean regret 0.0039 vs 0.0087, 3/4 데이터셋 정확 적중)여서 상시 로깅 가치 확인됨.

**변경**:
- `mae_anomaly/loss.py`: `loss_tensors['es_disc_per_sample'] = sample_discrepancy.detach()` 노출 (recon `es_teacher_recon_per_sample`과 동일 패턴; teacher-only warmup엔 sample_discrepancy=0). detached → loss 미진입 → **학습 byte-identical**.
- `mae_anomaly/trainer.py`: disc 누적기(`_ds_sum/_ds_sumsq`, count는 `_es_cnt` 공유) → epoch 끝에서 `_train_disc_snr = (mean_a−mean_n)/(σ_a+σ_n+ε)` 계산 → `history['train_disc_snr']`에 append. recon_snr 계산을 그대로 미러링 (Cohen's-d, train data + train labels).

**범위**: 새 run_base 프로세스부터 적용(fresh import). 진행 중인 run(메모리 내 구코드)은 미적용 — 2026-06-23 official 재개 큐에서 sd1(실행 중)은 제외, enc1sd1·w100p5·nogrl(미시작)부터 `train_disc_snr` 로깅됨. recon_snr 미저장이던 기존 5모델은 freezeenc만 재추론으로 disc_snr 확보.

## 2026-06-23: best-epoch 선택 POST-WARMUP 무조건 강제 (checkpoint + metric + viz 일원화)

**문제**: 지금까지 post-warmup 강제는 **viz에만**, 그것도 `config.official` 게이트로만 적용됐다. best_checkpoint 저장(greedy)·최종 best_epoch·SWaT excl22 best는 **warmup 필터 없이** pak_auc_f1 전체 최대를 골라, student/discrepancy가 아직 안 배운 **pre-warmup epoch이 best_model·보고 메트릭으로 선택**될 수 있었다(예: WaDi_A1 metric-best=ep12 pre-warmup, viz=ep27 post-warmup으로 불일치).

**수정 — 단일 진실원**: `mae_anomaly/utils/experiment.py`에 `resolve_warmup_boundary(config)`(teacher_only_warmup_epochs, -1→num_epochs//2) + `select_best_epoch(records, metric_key, warm)`(epoch>warm 중 최대, 없으면 전체 fallback, **config 플래그 비게이트=무조건**) 추가. 네 지점이 전부 이 헬퍼를 사용:
- `scripts/run_base_experiments.py` **(A) greedy best_checkpoint**: `is_best = (ep > _warm_boundary) and (score > best)` — pre-warmup epoch은 best_checkpoint.pt가 되지 못함.
- `scripts/run_base_experiments.py` **(B) 최종 best_epoch**: `select_best_epoch(...)` — greedy와 동일 epoch 보장 → 로드되는 가중치·라벨·메트릭 일관.
- `scripts/run_base_experiments.py` **(excl22)**: SWaT excl22 best override도 `ep_num > _warm_excl` 가드(init이 이미 full-SWaT post-warmup best라 fallback 자동).
- `mae_anomaly/visualization/best_model_visualizer.py` `_select_best_epoch_for_viz`: `config.official` 게이트 제거 → **무조건** post-warmup, 위 헬퍼 공유.

**범위**: 새 run_base 프로세스부터 적용(실행 중인 freeze는 메모리 내 구코드라 영향 없음 — 의도된 "freezeenc 이후"). SMD/Exathlon 요약-집계 CLI(`run_base:4106/4210`)는 official 경로 아님·config 없음·display-only라 제외. warmup 없는 런(warm=0)은 no-op. 헬퍼 단위테스트 통과(WaDi_A1형 ep12 무시→ep27, fallback, auto-warm).

## 2026-06-23: official warmup score BUG FIX (force_recon_only) + TEP eval_interval/viz/bg-join 정리 + noisy-label 재정의(labeled %)

**핵심 버그 수정 — `mae_anomaly/scoring.py`**: `compute_official_causal_score(..., force_recon_only=False)` 추가. teacher-only warmup(`is_prewarmup_epoch`) 동안 student discrepancy는 **아직 학습 안 된 noise**라 anomaly score에 들어가면 안 되는데(이미 `compute_adaptive_components`는 같은 게이트 적용), official causal score는 이를 그대로 섞어 **warmup 구간에서 `official_score < teacher_recon`**이 되는 문제가 있었다. `force_recon_only=True`면 score를 teacher reconstruction-only로 환원(= `teacher_recon_error`와 bit-identical). `scripts/run_base_experiments.py`의 두 호출 지점(per-epoch npz 저장 `is_prewarmup_epoch(config, ep)`, parallel eval `evaluator._force_recon_only`)에 wiring.

**`scripts/run_base_experiments.py`**:
- **eval_interval**: `TEP_typegen` 경로 전용 default = **3** (explicit `config.eval_interval` override는 항상 우선; 비-TEP은 종전대로 official=1 / else `EVAL_INTERVAL=5`).
- **bg-worker join 무제한화**: 기존 `p.join(timeout=600)`+terminate가 느린 viz(예: win100 detailed 84만 샘플)를 렌더 도중 강제 종료해 **viz 파일이 누락**되던 것을 제거. 기본 = 무제한 대기(끝까지 완료). 진짜 hang backstop은 `MAE_BG_JOIN_TIMEOUT`(초) 명시 시에만 동작.
- **noisy-label 재정의**: 태그 = **LABELED %**(`lab80/lab50/lab25/lab10` = 100·(1−unlabeled_frac))로 변경, u25/u50 2-point → 4-point sweep. A(100% labeled)·B(0%)는 Phase-2 main matrix라 미포함.

**`mae_anomaly/datasets/loaders.py`**: noisy-label variant 등록 키 `tep_typegen_{fold}_{lab80,lab50,lab25,lab10}` (unlabeled_frac 0.20/0.50/0.75/0.90). 기존 `_u25/_u50` 대체.

**`mae_anomaly/config.py`**: 중복 `eval_interval` 정의 제거 — Training-parameters 그룹의 새 `-1`(auto) 정의를 아래쪽 잔존 `eval_interval: int = 5`가 가려 default가 5로 깨져 있던 것을 단일 정의로 정리.

**`mae_anomaly/visualization/best_model_visualizer.py`**: TEP type-gen run은 `anomaly_threshold` / `anomaly_threshold_test_event` 2개 event-timeline viz skip(test-event의 per-pred-region 렌더가 O(n_pred_regions) → TEP의 fragmented 예측에서 ~14min/plot 폭발). 경로의 `'typegen'` 마커로만 감지 → SWaT/WaDi/SMD 등 **타 데이터셋 무영향**.

## 2026-06-23: docs — TEP_MAE.md §5에 VUS 제외 규칙 + `MAE_SKIP_VUS` 끄는/켜는 법 명문화 (doc-only)

TEP 최종 분석의 **VUS 영구 제외**(`vus_pr`/`vus_roc` 무시, headline=`pak_auc_f1`)와 그 구현(`MAE_SKIP_VUS` 환경변수)을 `docs/TEP_MAE.md §5`에 서술. `TEP_typegen*` 키는 `run_base_experiments.py:2763`이 `MAE_SKIP_VUS=1`을 **자동 set**(spawn된 bg eval/viz/pool worker에 상속), 임의 런은 `MAE_SKIP_VUS=1 python … --dataset <KEY>`로 끄고 unset/`0`으로 켠다. 코드 변경 없음. (로컬 `config.skip_vus` 방식은 폐기되고 origin `MAE_SKIP_VUS`로 일원화됨.)

## 2026-06-23: TEP eval-bound 완화 — VUS/RF1 skip + eval_interval override + best-epoch 재계산 제거 + EXPERIMENT_INFO.md (base 100% 불변)

TEP-scale 테스트(422K points / 320K anomaly / 400 regions)에서 **wall-clock이 학습이 아니라 평가에 지배**되는 eval-bound 문제를 root-cause로 해소. cProfile 결과 final eval 104s 중 **R-based F1(`metric_RF1`)이 82.3s** — TSB_AD의 순수 Python `O(n_anomaly_points)` 루프 — 가 단일 최대 비용이며, per-epoch ×N + final + per-fault로 반복 지불됨. VUS(~40s/call)도 eval-tail 병목.

**핵심 가드: 모든 신규 동작은 `MAE_SKIP_VUS=1` env에서만 켜짐. base/non-TEP 데이터셋은 byte-identical(VUS·RF1 모두 종전대로 per-epoch+final 계산, npz@best misalignment finalize도 유지).** env는 TEP run에서만(`run_base_experiment`가 `TEP_typegen` key 감지 시) 설정되고 spawn된 bg eval/viz·pool worker에 상속됨.

**추가/변경**:
- `mae_anomaly/config.py`: `eval_interval: int = -1` 필드. `>0`이면 N epoch 간격으로만 eval(+ 마지막 epoch 항상) → 평가 횟수 절감. `-1`(default) = auto(official이면 1, 아니면 `EVAL_INTERVAL=5`).
- `mae_anomaly/evaluator.py`: `_compute_threshold_dependent(..., skip_rf1=False)` + `MAE_SKIP_VUS` 게이트로 RF1 skip. `compute_full_metric_set`의 VUS도 `lite OR MAE_SKIP_VUS`로 skip. **RF1/VUS는 보조 진단**(headline=`pak_auc_f1`)이라 TEP에서만 off.
- `scripts/run_base_experiments.py`:
  - `_read_best_epoch_metric_set` — VUS-off final eval은 재계산 대신 `epoch_metrics.json[best]`를 읽음(per-epoch eval이 이미 계산+저장 → best epoch을 SELECT한 바로 그 값; `pak_auc_f1`이 old 재계산과 5dp 일치 검증). adaptive + teacher 키 복원.
  - `_score_type_metrics_parallel` — per-epoch에 저장 안 되는 disc/student_recon만 `ProcessPoolExecutor`(spawn, `MAE_SKIP_VUS` 상속)로 병렬 계산, pool 실패 시 serial fallback.
  - bg-worker final eval을 `MAE_SKIP_VUS`로 분기: VUS-off → 무재계산 read 경로 / 그 외 → **기존 full final eval + npz@best finalize 그대로**.
  - post-training VUS sweep(`_run_vus_sweep_on_saved_npz`, epoch_NNN npz × 30)도 VUS-off일 때 skip(산출물=대시보드 VUS 행뿐인데 VUS off면 공백). 대시보드는 `epoch_metrics.json`에서 직접 렌더(VUS 행 공백).
  - `_write_tep_experiment_info` — TEP run마다 조건/데이터셋 구성/**라벨링**/설정을 담은 `EXPERIMENT_INFO.md` 자동 생성(self-documenting; best-effort, run에 예외 전파 안 함).
- `docs/TEP_MAE.md`: 모든 실행 예시에 `eval_interval=2` 추가 + eval-bound 설명 노트.

**무영향/안전**: 모든 변경이 `if config.official`/`MAE_SKIP_VUS=='1'` 가드 뒤 → official=False·non-TEP는 numeric/경로 불변. best-epoch 선정 기준(`pak_auc_f1`) 불변.

## 2026-06-22: TEP type-disjoint generalization — MAE 실험 (조건=데이터/라벨 체제) + noisy-label·LOFO 추가실험

TEP type-gen 실험을 기존 MAE 파이프라인(`official=True`)으로 돌리기 위한 loader/조건 구현. **조건은 model flag가 아니라 데이터/라벨 체제로 정의** (A=contaminated+라벨, B=contaminated+`blind_train_labels=True`, B0=clean ffonly). 라벨0이면 GRL이 `loss.py:310-323`(`_pos_count==0`)에서 자동 skip → `use_grl=False` 같은 flag 불필요(자동 inert).

**추가 (additive, default-off)**:
- `mae_anomaly/datasets/loaders.py`: `load_tep_typegen(fold, unlabeled_frac=0.0)` (frozen NPZ → `[train|test]` 6-tuple; noisy-label = per-fault 뒤쪽 일부 무라벨) + `load_tep_typegen_lofo(held_out, contaminate_heldout)` (3 seen / 1 held-out, 기존 NPZ 조립). DATASET_LOADERS 키 21개(base 5 + noisy 8 `_u25/_u50` + LOFO 8 `_lofo_<ho>[_cont]`).
- `mae_anomaly/config.py`: `blind_train_labels: bool=False` 필드.
- `mae_anomaly/dataset_sliding.py`: `SlidingWindowDataset(..., blind_train_labels=False)` — train split point_labels zero(단일 root-cause; test 무손상).
- `scripts/run_base_experiments.py`: `TEP_TYPEGEN_DATASETS`(DATASETS엔 미포함 → `--dataset`로만) + `all_datasets`에 추가 + train_dataset에 `blind_train_labels` 전달.
- `docs/TEP_MAE.md` 신규(실행 가이드), Notion "TEP Type-Disjoint Generalization (Table D.1)" 페이지.

**기본 계획** = Phase 1(B0 pilot ep30) + Phase 2(A/B/B0/D ep10, 9 runs, D=A에서 파생). **추가실험(별도)** = noisy-label(u25/u50) + LOFO(±cont). weight 미저장, minmax. **검증**: py_compile OK + CPU 스모크(onset 161, FF/test 무손상, blind/noisy/LOFO 라벨·shape, official 모델 forward finite). ⚠️ `scripts/TEP/data/*.npz`는 gitignore(`*.npz`) → 타 머신은 `build_tep_data.py` 재생성 필요.

## 2026-06-22: anomaly_threshold_test_event.png — AR-threshold Test Event Timeline (별도 파일)

**신규 메서드** `BestModelVisualizer.plot_anomaly_threshold_test_event` → `anomaly_threshold_test_event.png`. **anomaly score · recon · disc 각각**에 대해 image1-style **Test Event Timeline**(score 검정선 + threshold 점선 + ground-truth 음영) + **gt/pred/overlap 이벤트 트랙**(`broken_barh`, gt 빨강/pred 파랑/overlap 보라)을 그림 (총 6 패널 = 3 컴포넌트 × [timeline, tracks]).

**Threshold = TEST anomaly ratio 기반**: `ar = mean(point_labels==1)`, 컴포넌트별 `thr = quantile(score, 1-ar)` (정확히 `ar` 비율을 양성으로 플래그). `pred = score ≥ thr`, `overlap = pred ∩ gt`.

**컴포넌트 = 메트릭이 실제로 쓰는 점수 경로와 정확히 일치**:
- **official 런**: anomaly score = `official_score`(= recon + 0.25·disc·`s_t`, causal; `s_t=(R_tr+Σrecon)/(D_tr+Σdisc)` train-normal-seeded 누적비), disc 기여 = `official_score − recon`(= 0.25·disc·`s_t`, **정확값**). recon = raw teacher recon.
- **non-official 런**: anomaly score = `adaptive_score`, disc = `scaled_disc`(student_error, compute_adaptive_components).

**중요(scale 검증)**: adaptive의 scale-match `disc×(recon.mean/disc.mean)/4`는 official의 `0.25·disc·s_t`와 **다르다**(SWaT 실측: 평균 0.53×, 점별 corr 0.85, `s_t`는 1.652 상수가 아니라 0~1.384 변동). 둘 다 0.25(=/4=w)는 같지만 "비율 맞추는 작업"(scale-match)이 TEST 평균비(전역 상수) vs train-seed 누적비(점별)로 갈림. 초기엔 official 런에도 adaptive scaled_disc를 그려 실제 점수의 disc 기여를 잘못 표현했던 것을 정정 — official 런은 이제 `official_score`/`official_score−recon`을 사용. AR-threshold는 quantile 기반이라 pred/트랙은 단조변환에 불변, plot/threshold 스케일만 정합.

**색상/축**: test_event 선 색은 `anomaly_threshold.png`와 동일(anomaly score=black, recon=tab:blue, disc=tab:green). y축은 **panel별 독립(자동 스케일)** — 각 컴포넌트의 detail이 보이도록(y축 공통 통일도 시도했으나 disc가 과도하게 눌려 가독성이 떨어져 사용자 판단으로 미적용).

**`anomaly_threshold.png`도 official scale로 정합**: official 런에서 panel1 = `official_score`, disc panel = `official_score − recon`(= 0.25·disc·s_t), FPR도 official 기준. det_ratio/panels/FPR 모두 `score_plot`(official) 사용. non-official은 기존 adaptive 그대로. (anomaly_threshold.png의 per-panel 독립 y축 레이아웃은 det-ratio 주석 때문에 유지 — test_event만 y축 통일.)

**wiring**: `generate_all`의 `_safe_plot('anomaly_threshold', …)` 직후에 `_safe_plot('anomaly_threshold_test_event', …)` 추가 → 이후 모든 런이 자동 생성. npz 기반(추론 없음). 현재 official SWaT(`271_…_30ep_42`, full + excl22) 두 파일 모두 재생성 완료.

**(viz best-epoch — official은 post-warmup 강제)**: `_select_best_epoch_for_viz` 헬퍼 추가 — `pak_auc_f1` 최대 epoch을 고르되 `official=True`면 **post-warmup(epoch > teacher_only_warmup_epochs)으로 제한**(warmup 중엔 student/discrepancy 미학습이라 pre-warmup 'best'가 score/disc 시각화에 오해 소지). `plot_anomaly_threshold`·`plot_anomaly_threshold_test_event` 둘 다 사용. **VISUALIZATION 한정** — 학습/메트릭 best-epoch(`config.best_epoch_metric='pak_auc_f1'`, run_base 2810)는 불변. 예: official WaDi_A1 전체best=ep12(pre-warmup 0.8054)였으나 viz는 ep27(post-warmup 0.8010) 사용 — WaDi_A1 viz npz 기반 재생성 완료(학습 미중단: npz/CPU라 충돌 없음, official keep=False라 중단은 WaDi_A2 진행분 손실).

## 2026-06-22: baseline epoch_metrics.json — per-epoch `train_loss` 기록

각 baseline의 `epoch_metrics.json` 각 epoch 엔트리에 **평균 train loss(`train_loss`)**를 기록하도록 추가. 기존엔 모든 DL/SOTA wrapper가 epoch마다 loss를 계산해 `self.train_loss_history`에 적재하고 **stdout으로만 출력**했을 뿐, 결과 JSON엔 저장되지 않아 실행 후 소실됐다(8·10번 등 과거 전 실험 0/전체).

**Fix** — `comparison/baseline_common.py`에 `_attach_train_loss(metrics, model, ep)` 헬퍼 추가: wrapper들이 `epoch_callback` **호출 직전** 적재하는 `train_loss_history[ep-1]`을 epoch 엔트리에 기록. 호출 지점 = 4개 학습 함수의 eval 산출부 + fallback 분기: `run_dl_baseline_with_epoch_eval`(generic mlp/transformer/mlpmixer), `run_sota_baseline_with_epoch_eval`(14 SOTA), `run_weak_sota_baseline_with_epoch_eval`(deepmil/treemil/wetas/nrdetector), `run_segment_aware_dl_baseline`. `npsr`의 `(point, seq)` 튜플은 합산, loss 미노출/범위초과는 `None`. 비-DL(`run_simple_baseline`: random/pca_error 등)은 `train_loss=None`으로 키 일관성 유지. `_zero_metrics()`에도 `train_loss` 키 추가(degenerate epoch 스키마 일치). 기존 `run_dl_baseline`의 train_loss 주입 경로(이미 존재)와 정합.

**의미/주의**: 모델별 loss 함수가 다르므로(MSE / ELBO / BCE+DTW / l1+l2 …) **스케일 비교 불가** → 모델 내 epoch 추이 진단용. **2026-06-22 이후 실행분에만 존재**(과거 실험엔 backfill 불가 — 당시 stdout 로그에만 잔존).

**무영향/안전**: 순수 additive(새 키 1개), 학습/스코어/best-epoch(`pak_auc_f1`) 선정 numeric 불변. py_compile OK + 헬퍼 단위테스트(scalar/npsr튜플/누락/범위초과/nan) 통과. 원본 `comparison/.trash/260622/baseline_common.py.bak` 백업.

**과거 실험 backfill** — `comparison/backfill_train_loss.py`(신규)로 8번·10번(+9번 weak deepmil/wetas)의 기존 `epoch_metrics.json`에 동일 형태로 `train_loss` 주입(과거엔 stdout에만 출력돼 소실됐던 값을 캡처 로그에서 복구). 결과: **8번 627 DL run(12,175/12,360 엔트리 non-null) + 10번 456 DL run(10,470/10,655) 전수 채움**, 비-DL은 `null`. 핵심: **8번의 SMD/MSL/SMAP는 06-13+ 후속 run(`chain_10_11_12_resumed_0613.log`·`watcher_10_11.log`)이 8번 출력 디렉토리에 써넣은 것** — 해당 로그에서 복구(SMD 셀당 정확히 1회 기록=재실행 없음 확인). 귀속 정확성: 큐 job별 `[tag] Experiment: <DS>/<VAR>` 라인으로 결과경로 직접 매핑(태그는 `smd_1-2` 단축형이라 토큰 추측 불가→Experiment 라인 사용), baseline 17모델 필터로 exp9 분리, 로그 mtime 시간순 latest-wins. 포맷 7종(generic `Loss:`/`Epoch N: loss=`/memto 2-phase/`rec_loss`/dagmm `L1+L2`/npsr `M_pt+M_seq`/wetas) 커버. 값 검증: dagmm(0.051674+0.177649=0.2293)·npsr(0.074485+0.218612=0.2931)·anomaly_transformer(rec_loss 0.0426)·tranad-SMD(watcher 로그 0.055134) raw와 소수점까지 일치. 멱등(재실행 동일)·atomic write. 사전 전체 스냅샷 `comparison/.trash/260622/epoch_metrics_pre_trainloss_*.tgz`(1,455 files). **미복구 2건**: 8번 SMAP/{P-1,G-7}/dcdetector는 원본 epoch_metrics.json 공백 손상(기존 문제)이라 주입 불가.

## 2026-06-22: baseline metadata.json — 실제 사용 파라미터 자동 기록 + 8/9/10 fact-only backfill

각 baseline 모델의 `metadata.json`에 **학습·추론에 실제 사용된 모든 파라미터**를 기록하도록 추가. 핵심 요구: **추측 없이 fact만**, 특히 **per-dataset batch_size 변경(WaDi catch÷8/dcdetector÷2 등)까지 정확히 반영**.

**Forward(자동 수집)** — `comparison/baseline_common.py`에 `augment_run_metadata()` + 헬퍼(`introspect_model_attributes`/`_json_safe_scalar`/`_capture_environment`/`_git_commit_sha`) 추가. `comparison/run_baseline.py`는 매 모델 dispatch 직후(모든 분기 공통 지점, `torch.cuda.empty_cache()`와 viz 사이) 이를 1회 호출. **살아있는 model 객체에서 직접** `batch_size`(post-divisor)/`epochs`(post-override)/`lr` 등을 읽으므로 모든 override가 정확. 기록 키: `parameters.{configured_preset_hp, effective, all_model_attributes, batch_size_override, epoch_overrides, epochs_run, normalize_mode, n_features, train/test_shape, eval_interval}` + `environment.{git_commit, torch/cuda/python, gpu, conda_env}` (`metadata_schema_version: 2`).

**Backfill(과거 8/9/10)** — `comparison/backfill_metadata.py`(신규). live 객체가 없는 과거 run에는 **artifact·코드규칙·git에서 확인되는 사실만** `backfill_parameters`(별도 키, schema 1)로 기록하고 값마다 `provenance` 부착: `epochs_run`(epoch_metrics.json), `normalize_mode`(self-norm 규칙), WaDi catch/dcdetector `batch_size_override`(divisor 규칙 + **저장 timestamp로 git era 고정** — catch는 commit b7fc99e[2026-06-13]에서 ÷4→÷8 변경되어 8번=batch 32 / 10번=batch 16). 미기록 HP는 **날조 없이** note 명시. 적용 결과: 8번 814(810 ok + 손상 metadata 2 복구 + 손상 epoch_metrics 2) / 9번 185 / 10번 633. WaDi batch override = 8번 4셀 + 10번 4셀.

**무영향/안전(엄격)**: ① run_baseline.py 변경은 **metadata 키 추가만** — 실험 numeric 출력 불변. ② augment 함수 + 호출부 **이중 try/except** 가드 → metadata 실패가 run을 중단/변경 불가(outer try의 sys.exit(3) 경로 차단). ③ atomic write(tmp+os.replace). ④ 실행 중인 10번(chain 87747)은 구코드 subprocess 무영향, **신규 subprocess부터** 자동 적용. ⑤ backfill은 완료(scores.npz)·미-augmented(schema<2) dir만 처리, live worker dir은 metadata.json 부재로 자연 제외. **부수 발견**: 8번 SMD/machine-1-2/mlpmixer·SMD/machine-3-3/npsr의 metadata.json 및 SMAP/{P-1,G-7}/dcdetector의 epoch_metrics.json이 공백으로 손상(기존 문제) — 전자는 backfill로 복구, 후자는 epochs_run=unavailable 표기.

**검증**: 오프라인(mock model + 실제 metadata 복사본)으로 기존필드 보존·batch override·비스칼라 제외·atomic write assertion 통과. dry-run→디스크 검증(8번 catch×WaDi=32/div4, 10번=16/div8, dcdetector=64/div2). py_compile 3파일 OK. 독립 다중 에이전트 adversarial 검증.

## 2026-06-22: `official=True` — MAE-mode override bundle (separate code path, default-off)

**개요**: 신규 config `official: bool = False`. ON이면 **별도 경로**가 고정 번들을 적용하고 **충돌하는 모든 config을 무조건 override**한다. 언급 안 한 config은 **271(canonical)을 default**로 깔되, 기존 방식(`config_override`)으로 개별 override 가능. **official=False는 byte-identical**(모든 신규 동작이 `if config.official` 가드/단락 뒤).

**번들(강제, official-8)**:
1. **train offset 제거 + `sliding_window_stride=1`** — `apply_official_overrides` + run_base의 **로컬 `train_stride=1`** 강제(데이터셋은 config 필드가 아닌 로컬 train_stride를 읽으므로 둘 다 필요).
2. **`num_epochs` default 30 (override 가능)**, **`teacher_only_warmup_epochs` default `num_epochs//2` (override 가능)** + `use_teacher_warmup_early_stop=False`. 둘 다 run_base의 official 빌드에서 user merge **전** default로 깔려 사용자 `config_override` 값이 우선(예: `num_epochs=40`→warmup auto 20, `num_epochs=40 teacher_only_warmup_epochs=10`→10). apply_official_overrides는 이 둘을 **강제하지 않음**(epoch_offset/stride/early_stop만 강제).
3. **per-iteration LR** — MAE식(`util/lr_sched.py`: 0→선형 warmup over `w=teacher_only_warmup_epochs`, 이후 half-cosine→`min_lr`=0 over `[w, num_epochs)`), fractional epoch `e=epoch+batch/len(loader)`. group별 peak-LR 캡처 후 비율 스케일(GRL-cls lr 비율 보존). per-epoch `scheduler.step()`은 official일 때 skip.
4. **매 epoch model-only 저장** → `official_epochs/epoch_NNN.pt`(별도 namespace, best_checkpoint 미간섭). ⚠️ d_model=512에서 ~138MB×25 ≈ **3.3GB/dataset** 디스크. **체크포인트 보존 옵션**(`official_keep_checkpoints: bool=True` 전역 + `official_ckpt_overrides='key1:false,key2:true'` per-dataset, 미명시는 전역): `False`('저장 안함')면 official_epochs/ writes를 **skip**하고 eval+viz는 정상 수행, **끝나면** best_model/best_checkpoint/latest를 **삭제**(save_weights/KEEP_CHECKPOINT_DATASETS 무관) → 결과(metrics·npz·viz)만 남김(스모크: 3.3GB→**13MB**).
5. **`eval_interval=1`**(매 epoch eval) — per-experiment **로컬**(전역 `EVAL_INTERVAL=5` 미변경 → 큐 오염 0).
6. **신규 causal/online anomaly score** (`scoring.py` 단일출처): train-normal seed `R_tr=Σrecon_tr[정상]`, `D_tr=Σdisc_tr[정상]`; per test t: `s_t=(R_tr+cumsum recon)/(D_tr+cumsum disc+eps)`; `score_t=recon_t+0.25·disc_t·s_t`. cumsum=prefix-only → **미래·라벨 미사용**. best-epoch는 매 epoch **train-inference(R_tr/D_tr)** 후 causal score의 pak_auc_f1로 선택. 최종 metric·VUS·excl22·viz **전부 causal로 일관**(npz에 `official_score` key 추가, `adaptive_score` 보존).
7. **단일 전역 seed** — `set_seed_official`(+PYTHONHASHSEED + DataLoader generator/worker_init_fn), `cudnn.benchmark=True` 유지·`use_deterministic_algorithms` 미호출(속도 보존). 하나(`random_seed`) 바꾸면 전부 바뀜.

**271-base 레이어링**: `make_config`에서 official이면 `CANON_271`(완전한 271 dict) 베이스 ← 사용자 명시 `config_override` key ← `apply_official_overrides`(official-8 강제, 최종 writer). Set C preset 기하는 우회(271이 자체 공급).

**핵심 메커니즘**: `apply_official_overrides(config)`는 `if not official: return config`로 단락(official=False면 setattr 0). `make_config`(queue/CLI 양쪽 유일 깔때기)에서 merge loop 직후·dim_feedforward/검증 직전에 호출 → official이 preset/override-string/dataset_def를 last-writer로 이김.

**파일**: `config.py`(+82/−0: 필드·CANON_271·apply_official_overrides·set_seed_official), `utils/experiment.py`(+7/−0), `scoring.py`(+60/−0: causal 2함수), `trainer.py`(+36/−1: per-iter LR), `run_base_experiments.py`(+161/−11: 271-base 빌드·로컬 stride/eval_interval·매-epoch 저장·per-epoch train-inference→causal 치환·npz·viz·seeding). 삭제 전수감사 = 전부 의도된 교체, 비-official 경로 보존.

**검증**: ① **official=False byte-identity** — 가드/단락 + 삭제 12줄 전수감사 + `make_config(official=False)==hand-built` 단위테스트. ② 코어 단위테스트(config 깔때기·271-base·causal **미래미사용 경험증명**·per-iter LR MAE공식 일치) PASS. ③ **실제 GPU e2e 스모크**(MSL T-13, batch=64, 동시 실행 중인 exp336 무영향): official=True/ep25/warmup15/stride1/offset off/d_model512(271-base)·batch64(사용자 override) 확인, eval_interval=1·25 eval, `official_epochs/` 25 ckpt(model-only), npz `official_score`(causal, adaptive와 distinct, finite), best-epoch 10 causal-pak_auc_f1 선택, 전 파이프라인(train+eval+train-inference+VUS+viz) 57s 완주. 백업: `.trash/0622/`.

## 2026-06-22: `force_mask_all_anomaly` — per-sample mask ALL anomaly patches (Option C) + exp337

**문제**: `force_mask_anomaly`는 **고정 budget**(`round(num_patches*masking_ratio)`)만큼만 마스킹하고 anomaly 패치를 우선순위로 채운다. 한 윈도우의 anomaly 패치가 budget을 **초과**하면 초과분은 top-budget에 못 들어 **encoder에 그대로 보인다**([model.py:1004](mae_anomaly/model.py#L1004) 주석의 "excess remain visible").

**변경 (default-off, 271 byte-identical)**: 신규 플래그 `force_mask_all_anomaly`. ON이면 **per-sample**로 `K_s = min(max(budget, N_a), num_patches-1)`만큼 마스킹 → 모든 anomaly 패치를 가린다(encoder가 anomaly 미관측). 단일변수.

**핵심 — mask_after_encoder=True(MAE visible-only) 유지하며 ragged(가변 visible) 처리**: per-sample 마스킹 개수가 달라 visible 개수가 ragged가 되는데, MAE visible-only 인코더는 직사각 텐서를 요구한다. 이를 **padding + `src_key_padding_mask`**로 해결:
- `model.py` force 블록: flag-gated per-sample 가변 마스크(`(rank >= K_s)`). OFF는 기존 scatter 경로 **verbatim**(byte-identical).
- `_encode_visible_only`: `num_keep = visible.max()`, uniform이면 **원본 경로 그대로**, ragged면 over-masked 샘플을 max까지 padding하고 padding 위치를 `src_key_padding_mask`로 인코더가 무시. padding mask를 `self._last_encode_kpm`에 저장.
- `_insert_mask_tokens_and_unshuffle`: `_last_encode_kpm`이 있으면 padding 행 latent을 mask token으로 덮어쓰고 기존 `cat + ids_restore` 복원. None(uniform/eval/default)이면 **no-op → byte-identical**.
- `config.py`: `force_mask_all_anomaly: bool = False`. `trainer.py`: 검증(force_mask_anomaly=True + use_masking=True 요구, fully-assembled config 위치 — config.py는 `__post_init__`이 setattr override를 못 봄).
- **loss.py 무수정**: 반환 mask가 그대로 흘러가고 모든 소비처가 `+1e-4`/`n==0` 가드.

**eval/viz 무영향**: force 블록은 `self.training` 게이트 → 학습 forward(trainer.py:981) 1곳에서만. eval/scoring/viz(evaluator.py:1755/1818, training_visualizer.py:206)는 외부 uniform mask + `masking_ratio=0.0` + eval 모드라 force 블록 미진입.

**검증(GPU 미사용, CPU, HEAD 대비 실증) — 전부 PASS**: ① **OFF + eval byte-identity**: model.py만 HEAD로 stash해 동일 seed 학습-force-mask + eval-external-mask forward 시그니처 비교 → **완전 일치**. ② ON ragged: per-sample `n_masked==K_s`, **anomaly_left_visible=[0,0,0,0]**(이상 전부 마스킹). ③ unshuffle STRONG: mask token이 masked 위치에 **정확히** 일치(다른 곳 없음). ④ all-anomaly 윈도우: cap→1 visible, 무크래시·유한. ⑤ 271-path(GRL+FM+patch_level)+ON: forward·criterion·**backward 유한, grads 유한**. ⑥ config 검증 ValueError 발화.

**exp337**: 271 base + `force_mask_all_anomaly=True` (단일변수, queue 끝에 추가). 백업: `.trash/0622/`.

## 2026-06-20: Warmup early-stop `train_loss` peak_reversal mode + always-on `train_recon_snr` + exp330/331

**1) `train_recon_snr` 항상 계산·저장 (early-stop OFF 포함 전 실험)**: teacher train-recon SNR(`(recon_a−recon_n)/(σ_a+σ_n+ε)`, anomaly↔normal 분리도)은 warmup early-stop 메트릭으로 이미 계산됐으나 `use_teacher_warmup_early_stop=True`에서만, 그리고 history에 미저장이었음. 이제 `loss.py`의 per-sample teacher recon 노출을 **항상**으로(게이트 제거), `trainer.py` 누적/계산도 `_es_on`/`teacher_only` 게이트 제거 → **모든 실험·모든 epoch**에서 `_train_recon_snr` 계산 후 `history['train_recon_snr']`에 기록(미정의 epoch=None). detached/no-grad → 학습 byte-identical.

**2) Warmup early-stop `metric='train_loss'` (peak_reversal, spec pseudo-code 정확 구현)**: 기존 `recon_snr`(분리도 maximize, 무갱신-plateau) 외에 **`train_loss`**(teacher recon **minimize**, peak_reversal) 모드 추가. `teacher_warmup_early_stop_metric='train_loss'`로 선택. 규칙: `min_epoch=20`부터 `check_interval=5` epoch마다 epoch train_loss 확인 → 최저점 대비 `relative_threshold=1%` 초과 악화 check가 `patience_checks=2`회 연속이면 종료, **train_loss 최저 epoch**으로 model+optimizer+scheduler rollback 후 student 시작(`teacher_only_warmup_epochs`는 상한). 신규 config 4종(`teacher_warmup_es_check_interval/_patience_checks/_relative_threshold/_min_epoch`)이 pseudo 기본값. `recon_snr` 경로는 `elif`로 분리 → **byte-identical**. 미지원 metric은 trainer init에서 ValueError.

**파일**: `config.py`(metric enum 주석 + 4 신규 필드), `loss.py`(es-tensor 노출 항상), `trainer.py`(누적/계산 게이트 제거 + `train_recon_snr` history + train_loss 분기 + metric validation).

**검증(GPU 미사용, CPU, HEAD 대비 실증)**: ① **byte-identity**: 동일 seed GRL 학습의 per-epoch train/recon/disc/normal/anomaly/fm/grl loss가 **현재 vs git-HEAD 10자리 완전일치**(stash↔pop diff 공란) → anomaly_disc+train_recon_snr+train_loss 변경 전부 학습-중립. ② **recon_snr 경로**: ES=ON 현재 vs HEAD의 recon_snr+loss 시리즈 완전일치 → 기존 292/294 거동 불변. ③ **ES=OFF**: `train_recon_snr`가 실수값으로 기록(요청 기능). ④ **train_loss peak_reversal**: 실제 `Trainer.train()`에 스크립트 손실곡선 주입 → best-low @ep30, reversal 2/2 @ep35·40 → 트리거 @ep40, ep30 rollback, warmup=40 (pseudo 손계산과 정확히 일치). ⑤ import/문법/impact-surface(loss.py 다른 사용처 0, `_train_recon_snr`가 누적·평균 루프 뒤 추가) 점검.

**3) exp330/331 (Group V) + 332–336 renumber**: 신규 **330 warmup_es_trainloss**(271 + ES train_loss) · **331 warmup_es_trainloss_freezeenc**(330 + `freeze_encoder_only=True`, 동적 단축 warmup_end 직후 encoder freeze) 끼워넣고 기존 dual-balancer/capacity 330–334 → **332–336** renumber(queue 54개, 중복 없음, parse+metric-validation PASS). Notion Spec(Group V 신설 + Group U 332–336) + Results(4표 행 renumber+삽입, callout/footnote) 반영.

## 2026-06-20: Collect anomaly output-discrepancy in training_histories (all paths, incl. GRL/SCAD)

**문제**: anomaly-region output discrepancy(= teacher↔student 출력 MSE를 anomaly 패치에서 평균낸 forward/un-margined 값, `anomaly_disc_forward`)는 `loss.py:296/462`에서 **`disable_anomaly_loss` 게이트 바깥**, 즉 GRL/SCAD 경로에서도 **항상 계산**되지만 `loss_tensors`(adaptive-λ용)에만 있고 **`loss_dict`→`epoch_losses`→`history`(training_histories.json)로는 수집되지 않았음**. 결과적으로 GRL/SCAD 실험(271 포함, `anomaly_loss`=0)에서는 normal discrepancy(`train_normal_loss`)는 기록되는데 **anomaly discrepancy는 미기록**(비대칭). anomaly discrepancy가 곧 anomaly detection score의 학습-시점 신호인데도 로깅 파일에 남지 않았음.

**변경 (additive, 학습 byte-identical)**: 이미 계산된 `anomaly_disc_forward`를 실험 로깅 파일에 per-epoch 수집.
- `loss.py` — `loss_dict['anomaly_disc_forward']`에 detached `.item()` 노출(1 라인 + 주석). `loss`/gradient 미접촉.
- `trainer.py` — (1) `self.history['train_anomaly_disc_forward']` init, (2) `epoch_losses['anomaly_disc_forward']` init(기존 `for key in epoch_losses` 누적·평균 루프가 자동 처리 — SCAD/GRL 집계버그 회피 패턴), (3) history append. 총 3 지점, 전부 additive.

**의미**: GRL/SCAD 경로에서도(`anomaly_loss` disable 무관) anomaly discrepancy 신호를 training_histories에 기록 → normal(`train_normal_loss`)과 대칭 비교 가능. warmup/teacher_only·discrepancy off 구간은 `0.0` sentinel(기존 `dis`와 동일 규약). 동적 스키마 UI 대시보드는 신규 키 자동 인식.

**검증(GPU 미사용, CPU)**: ① GRL 경로 — `disable_anomaly_loss=True`·`anomaly_loss=0`인데 `anomaly_disc_forward=2.21` 실수집, `total_loss` 불변. ② SCAD 경로 동일(1.88). ③ warmup `teacher_only` → 0.0 sentinel. ④ 실제 `train_epoch`(누적 루프 포함) tiny end-to-end: `history['train_anomaly_disc_forward']=[0.0, 2.05, 1.25]`(warmup 0/post-warmup 실값), `anomaly_loss` 내내 0 → 학습 불변. ⑤ trainer import/문법 정상. diff는 loss.py 1 + trainer.py 3 지점 **additive-only**.

## 2026-06-17: GRL effect diagnostics (`grl_diagnostics/`) + 6 new GRL scalars

SCAD-C가 `scad_diagnostics/`로 효과를 검증하듯, **GRL(gradient-reversal adversarial classifier)의 효과 검증** 진단을 추가. GRL은 적대적 minimax라 "성공"이 직관 반대(classifier가 나빠짐=balanced_acc→0.5가 목표)인데, 그 0.5가 **starvation/class-collapse와 구분 불가** → 진단의 핵심은 **진짜 invariance vs 죽은 게임 구분**. (이 세션 분석에서 발견한 #1 실패모드 = adaptive-λ starvation `effective_weight→0`.)

**신규 per-epoch 스칼라 6개**(전부 cheap·대부분 이미 계산됨·**loss 미투입 → 학습 byte-identical**): `train_grl_ramp_lambda`(Ganin 반전강도 `model._grl_lambda`), `train_grl_pos_logit_mean`/`_neg_logit_mean`(연속 invariance — student가 classifier 속일수록 수렴), `train_grl_logit_margin`(mean|logit| 확신도), `train_grl_main_grad_norm`/`_adv_grad_norm`(절대 적대 압력, adaptive-λ grad 블록 재사용). 기존 7개(cls_loss/balanced_acc/anomaly_acc/normal_acc/acc_gap/lambda/effective_weight)에 추가. `loss.py`는 기존 no_grad 블록에서 logit 통계 계산, `trainer.py`는 epoch_losses init+accumulation+history append(SCAD 집계버그 회피 — init·loss_dict 양쪽 등록).

**`GrlDiagnosticsVisualizer`**(`mae_anomaly/visualization/grl_diagnostics_visualizer.py`) — 6 figure(scad 미러): `grl_adversarial_progress`(balanced_acc→0.5·실제 적대 압력=eff_w×‖grad_adv‖, ≈0=STARVED) / `grl_game_health`(TPR·TNR 동시궤적·acc_gap·phase plot으로 invariance vs class-collapse 구분) / `grl_optimization_signal`(λ ratio·‖grad‖ 절대값·starvation fraction) / `grl_detection_coupling`(invariance vs pak_f1, **post-warmup corr**) / `grl_transfer`(hidden invariance vs disc_snr, post-warmup corr — flat/음수=압력이 hidden에 갇힘) / `grl_diagnostics_summary` + JSON(verdict: STARVED/DEGENERATE/no-invariance/no-transfer/EFFECTIVE). 모든 corr은 **post-warmup 구간 한정**(pre-warmup 0-신호 artifact 제거).

**기존 GRL viz 이동**: `_relocate_existing()`가 `best_model/GRL_contribution_trend.png`·`epoch_metrics/epoch_grl.png`를 `grl_diagnostics/`로 **이동**(named 2개만, 삭제 없음, 없으면 no-op) → GRL 시각화 단일화.

**무영향**: `has_grl()` 가드 → `use_grl & grl_mode=='classifier'`일 때만 생성(SCAD/plain/WDGRL no-op). `run_base_experiments.py`(best_model+scad 뒤) + `visualize_all.py` + `__init__` export. 실행 중 exp324(구 코드)는 무영향, exp325+ 신규 subprocess부터 적용.

**검증(GPU 미사용)**: ① 4파일 diff **additive-only**(0 removed) + 신규 스칼라 no_grad/metric → 학습 numerics byte-identical. ② **실제 exp271 PSM** history로 6 figure+verdict("GRL DEGENERATE — balanced_acc 0.50→0.22 below-chance") 생성. ③ 합성 13-series(invariance+transfer)→verdict "EFFECTIVE"(transfer corr 1.00). ④ relocation/guard/logit-formula 테스트 PASS. ⑤ 적대적 다중 에이전트 검증.

## 2026-06-17: FM↔OD loss balancer (`fm_balance_mode`) + exp327–329

271의 FM **adaptive weight 계산 방식**을, 기존 grad-norm ratio 대신 multi-task loss balancer로 교체하는 옵션 추가. balance 대상은 **OD↔FM 쌍만**(reconstruction은 `total_loss`에 weight 1로 고정, balance 비관여). GRL은 **건드리지 않음**(자체 `loss_balance_mode` 유지, 271=`adaptive_lambda_legacy`). default `'none'`은 exp271과 **byte-identical**.

**왜 OD↔FM인가**: legacy FM λ(`‖∇_student OD‖/‖∇_student FM‖`)도 이미 OD↔FM 밸런서 — teacher reconstruction은 student decoder weight에 gradient가 0이라 상쇄됨(`config.py` line ~218 주석 "FM-OD gradient balancing"). 신규 4개 방법은 grad-norm이 아니라 **loss 값 기반**이라, balancer에 `recon+OD`가 아니라 **OD만** anchor로 넘겨 동일 의미를 값 기반으로 재현.

**`fm_balance_mode: str = 'none'`** (∈ none/relobralo/famo/uwso):
- `relobralo` — ReLoBRaLo loss-ratio softmax(Bischof&Kraus 2110.09813), `relobralo_*` 재사용. OD anchor, FM 상대가중, [0,10] clamp.
- `famo` — FAMO log-loss simplex(Liu NeurIPS2023 2306.03792), OD·FM 동시 재가중. `loss = recon + logcomb([OD,FM])` (recon은 `loss_tensors['reconstruction_loss']`로 명시 재구성 → loss 순서 불변식 의존 제거).
- `uwso` — UW-SO inverse-loss tempered softmax(Kirchdorfer 2408.07985)의 **MSE↔MSE scale-free 변형**: OD·FM 모두 작은 MSE(~1e-3)라 raw 1/L(~1e3)이 softmax를 saturate → 1/L를 평균으로 정규화해 temperature를 scale-free(손실 비율 의존)로. `fm_uwso_temperature=1.0`/`loss_floor=1e-4`/`ema_beta=0.9`, rel [0,10] clamp.
- `mse_norm_dann`은 **의도적 제외** — Ganin adversarial ramp가 협력적 MSE↔MSE에 무의미(원전 검토).

**파일**: `config.py`(`fm_balance_mode` + `fm_uwso_*`), `trainer.py`(validation·독립 RNG 상태·`_fm_balance_apply`+`_fm_relobralo`/`_fm_uwso`/`_fm_famo`·FM 블록 분기·`_lbm_state_dict`/`_load` resume 확장). **`loss.py` 무수정**(OD 텐서·FM 제외 로직 기존 활용). 요구: `fm_adaptive_lambda=True`+`use_feature_matching=True`, `use_scad` 배타.

**실험 327–329**(271 base, GRL=legacy 유지): **327** `fm_relobralo` · **328** `fm_famo` · **329** `fm_uwso`. 큐 44→47, v2f가 326 뒤 자동 실행.

**검증(GPU 미사용)**: ① `loss.py` sha256 **byte-identical** + 'none' 경로 legacy FM 계산 **content-identical**(들여쓰기만) → 진행 중 큐(exp324-326, 전부 default 'none') 무영향. ② 3 balancer CPU functional — finite·rel∈[0,10]·**recon grad==1.0(weight 1 불변)**·student-grad 흐름. ③ resume round-trip(famo w bit-exact·legacy ckpt back-compat). ④ 자체 발견·수정 2건: uwso saturation(1/L 평균정규화) + 메서드/상태 이름충돌(`_fm_famo`→상태 `_fm_famo_state`). ⑤ **적대적 다중 에이전트 검증(6 agent)** → SHIP, byte_identity_safe=True, must-fix 0; famo recon-isolation을 ordering-independent로 하드닝(should-fix 반영).

## 2026-06-15: H-SCAD-C (hidden-space repulsion) + transfer 진단 + scad_c 집계 버그 수정 + exp326

**H-SCAD-C** (`scad_apply_space='hidden_final'`): SCAD Form C one-sided repulsion 수식은 그대로, 측정·적용 위치를 projection `z=scad_head(student_hidden)` → **final student_hidden 직접**(`h=L2norm(LayerNorm(student_hidden))`, **parameter-free LN, 0 new params**, projection head 미생성). score path(student_hidden→student_output_projection→output discrepancy)와 정합 → projection-only absorption 위험↓. 6개 `scad_c_*` metric + 시각화 전부 공유(동일 모듈, `scad_form=='C'` 가드). `config.py: scad_apply_space='projection'`(default) `| 'hidden_final'`; `model.py` forward 분기 + `__init__`에서 projection일 때만 scad_head build(hidden은 0 param).

**Transfer 진단(behavior-neutral)**: hidden/projection 분리가 score가 쓰는 output discrepancy로 전이되는지 측정. `loss.py`에서 `scad_output_disc_gap = disc(A+)−disc(U)`, `scad_disc_anom_mean`, `scad_disc_u_mean`(U-drift watch) 계산(**detach, loss 미포함 → gradient·학습 무변경**). `trainer.py` history + viz `scad_c_transfer.png`(separation↑ vs gap↑ twin-axis · disc(A+)/disc(U) U-drift · sep-vs-gap corr) + verdict(`gap_transferred`/`u_drift_suspected`/`output_absorption_suspected`/`transfer_corr_sep_gap`/`transfer_success`). projection-SCAD-C(exp321)도 동일 진단 산출.

**버그 수정(root cause)**: `epoch_losses` 초기 dict(`trainer.py`)에 6개 `scad_c_*`가 누락 → per-batch 집계 루프(`for key in epoch_losses`)가 건너뛰어 history에 **전부 0.0**으로 기록되던 latent bug(직전 세션 Phase-4 누락). exp321(첫 Form-C)이 미실행이라 아직 데이터로 안 드러났을 뿐. init dict에 6 scad_c + 3 transfer 키 추가로 수정 — **학습 무변경(진단 metric만 올바르게 집계)**.

**시각화 태그**: `apply_space`를 summary JSON + figure suptitle에 표기(`H-SCAD-C [hidden_final]` / `SCAD-C [projection]`).

**실험 326**(271 base, exp321 단일변수 미러): **326** `hscadC_w10_hidden`(`scad_apply_space=hidden_final`). 큐 43→44, v2f가 325 뒤 자동 실행 → **E3(projection) vs E4(hidden) 비교**.

**검증(GPU 미사용, `.trash/0615` backup와 직접 대조)**: byte-identity — projection SCAD-C(94 param)·non-scad(88 param) **old=new bit-equal**(state_dict + forward teacher/student/scad_z sha) → 진행 중 큐(315·316-325) 무영향. hidden 경로 **0 new params**(88=non-scad), `_scad_z`=d_model-dim L2-normalized. 집계 버그 수정 확인. viz A+B 합성 산출(6 PNG + summary, transfer corr 0.98 "transfers to score"). 전 파일 `py_compile` OK.

## 2026-06-15: SCAD-C 진단 시각화 (`scad_diagnostics/` 신규 sub-dir)

`ScadDiagnosticsVisualizer`(`mae_anomaly/visualization/scad_diagnostics_visualizer.py`) — SCAD Form C(one-sided repulsion)의 6개 C-metric을 4개 렌즈("repulsion 작동 → collapse 없이 → 검출 전이")로 판정하는 전용 진단. 산출 `<exp>/<dataset>/visualization/scad_diagnostics/`: `scad_c_repulsion_progress`(mean_sim·active_frac·loss) / `scad_c_collapse_guard`(separation·cluster-var·위상궤적) / `scad_c_optimization_signal`(grad balance·weight·anchor 수) / `scad_c_detection_coupling`(mean_sim vs pak_f1) / `scad_c_summary` + `scad_c_diagnostics_summary.json`(자동 verdict). 데이터: `training_histories.json`+`epoch_metrics.json`, post-warmup만 verdict 계산. **무영향**: Form A/B/non-scad는 C-series 상수 0 → `has_scad_c()` False → 미생성. wiring: `run_base_experiments.py`(best_model viz 직후, guard `scad_form=='C'`) + `visualize_all.py` + `__init__.py` export. 그림 텍스트 영어(env에 CJK 폰트 없음).

## 2026-06-15: GRL 부착층(`grl_attach_layer`) + MAE식 비대칭 decoder 폭(`decoder_half_dim`) + exp324–325

두 구조 ablation 추가 (둘 다 default-off, **byte-identical**; `.trash/0615` 백업).

**(1) `grl_attach_layer: str = 'last'`** — `'first'`면 GRL classifier가 student decoder **1번째 층 출력(h1)** 을 읽음(reconstruction/FM은 최종 hidden h2 유지). adversarial invariance를 중간 표현에 걸고 마지막 층은 복원에 특화. `model.py` student forward에서 `'first'`일 때만 layer 수동 루프로 h1 포착(student depth≤1이면 'last' fallback). `'last'`는 기존 호출 그대로.

**(2) `decoder_half_dim: bool = False`** — `True`면 MAE식 비대칭 폭: teacher/student/shared decoder·mask token·output proj·GRL/SCAD head = **d_model//2**, decoder dim_feedforward = `(d_model//2)*4`, nhead 재사용, **encoder는 d_model 유지**. `decoder_embed=Linear(d_model, d_model//2)`가 latent를 좁힘(teacher side 학습, student는 detach — encoder↔student 비대칭 미러). dynamic d_model도 //2 자동. FM/discrepancy 무변경(양 decoder 동일 폭). guard: mask_after_encoder+transformer-enc-dec 필수, ema 비호환, nhead 비가분/홀수 d_model → ValueError. `False`면 `decoder_embed=Identity`.

**파일**: `config.py`(2 flag), `model.py`(decoder/proj/mask/head를 `_dec`/`_dec_ff`로, encoder 불변; forward에 decoder_embed + grl_first 분기). loss/trainer 무변경.

**실험 324·325** (271 base, 각 1개, v2f가 323 뒤 자동 실행): **324** `grl_first_layer` · **325** `dec_dmodel_half`. 큐 41→43. Notion Group R + Results 반영.

**검증(GPU 미사용, backup model.py와 직접 대조)**: **param-level byte-identity**(default 104 param 전부 bit-equal, 신규 key 0) + **forward byte-identity**(EVAL·TRAIN teacher·student·grl_logits bit-equal) → 실행 중 큐(315·316-323) 무영향. grl_first: forward finite + **h1≠h2 확인**(동일 weight서 first/last logits 상이). half: enc512/dec256·embed512→256·ff1024·FM 호환(양쪽 256). guard ValueError 2건. half+grl_first 결합 정상. 324·325 271 config 빌드 OK.

## 2026-06-15: SCAD Form C (one-sided thresholded negative repulsion) + exp321–323

**SCAD-C**: `L = mean max(0, cos(z_a, sg[z_u]) − γ)²` — anomaly anchor를 U(masked non-anomaly = background)에서 밀어내되 **U는 stop-gradient**(one_sided=True)라 anchor만 이동. 오염된 PU 세팅에서 U contamination에 robust + γ(기본 0)로 over-separation 회피. 수학적 관계: **C(one_sided=False, γ=−m) ≡ B(margin=m)** — C는 B에 (1)U detach (2)threshold reparametrize를 더한 변형.

**구현** (default-off, A/B byte-identical):
- `config.py`: `scad_gamma=0.0`, `scad_one_sided=True` (2필드).
- `loss.py compute_scad_loss`: `form='C'` branch + `gamma`/`one_sided` 인자 + 6개 C metric(`scad_c_mean_sim`·`active_pair_frac`·`active_sim_mean`·`gamma`·`n_anchor`·`n_u`) → `_scad_info`→`loss_dict`. `else: raise` 메시지에 'C' 추가. `MAEAnomalyLoss.__init__`에서 읽어 호출부 전달.
- `trainer.py`: 6개 C metric history init+append. 가중합/ramp/adaptive-λ/`disable_anomaly_loss` 게이트는 기존 재사용(무변경). `model.py` 무변경(`ScadProjectionHead` 재사용).

**실험 321–323** (전부 271 base, `linear` head, `w=1.0`, `patch` mode → A/B/C form 순수 비교):
- 321 `scadC_w10_linear`: Form C, γ=0.0, one_sided=True.
- 322 `scadA_w10_linear` / 323 `scadB_w10_linear`: 기존 A/B를 linear head·w=1.0로 재실행(312-315는 default head). v2f(FIRST_TORUN=316)가 320 뒤에 자동 실행. 큐 38→41.

**검증(GPU 미사용)**: import OK, config defaults, **A/B 연산 라인 unchanged**(git diff), C branch CPU smoke — **grad detach 확인**(anomaly>0, U=0), 6 metric finite, **C(os=False,γ=−m)≡B(m) allclose**, bad form raise, edge(no anchor) 안전; 321-323 mutex 3/3 PASS(use_grl=False·use_scad=True·patch_level_loss=True). smooth variant 미구현(hinge만).

## 2026-06-14: exp316–320 큐 추가 + `freeze_encoder_only` resume 버그 수정

**큐 추가**: `configs/queue_dedup_renumbered_v6.json`에 5개 신규 ablation(33→38개). 271 override string에 단일 토큰 append:
- **316 freeze_enc_after_warmup**: `freeze_encoder_only=True` — warmup(250) 이후 shared encoder(+patchify) freeze, decoder/GRL/FM는 학습 지속.
- **317–320 lbm_***: `loss_balance_mode=mse_norm_dann|relobralo|famo|uwso` — GRL BCE↔MSE discrepancy 스케일 정합 ablation(use_grl=True/classifier/use_scad=False 유지 → mutex 통과).

**launcher**: `scripts/resume_dedup_v2f.py`(FIRST_TORUN=316) 신규. 실행 중 v2e(311-315)는 큐를 시작 시 1회만 읽으므로 316-320을 무시(무중단) → v2e 완료 후 v2f로 실행. subprocess per-exp라 신규 코드 re-import됨.

**버그 수정 — `freeze_encoder_only` resume**: 트레이너의 encoder freeze 트리거는 `epoch==teacher_warmup`에서만 발화하는데, `_frozen_encoder_modules`가 checkpoint에 저장/복원되지 않아 warmup 이후 crash·resume 시 encoder가 조용히 un-freeze되던 잠재 버그를 수정(`run_base_experiments.py` save+restore, `_frozen_eval_modules` 패턴 미러링). **byte-identical when off**: 키는 freeze 발화 시에만 non-None → 그 외 실험(271 포함) 전부 무영향(구 ckpt no-op back-compat). exp316만 영향(정확성 회복).

**검증(GPU 미사용)**: queue JSON 유효성 + 316-320 override→Config mutex 재현(5/5 PASS), v2f syntax, freeze fix byte-identical-off 논리. Notion Spec(Group R)·Results(4 table 미실행 행) 반영.

## 2026-06-14: `loss_balance_mode` — GRL BCE ↔ MSE discrepancy 스케일 정합 6 선택지 (default-off, bit-identical)

**배경**: GRL classifier BCE 손실(O(1–20))과 자기증류 MSE discrepancy 손실(O(0.5→1e-4))은 차원·스케일이 근본적으로 다른 점수다. 기존 `grl_adaptive_lambda`(경사-노름 비 λ)는 main 경사가 줄면 λ→0으로 폭주해 GRL 신호를 굶긴다(adaptive runaway). 이를 대체할 **Axis-A 스케일 정합** 방법 4종을 공식 논문/repo line-by-line 기준으로 엄격 구현.

**구현**: 단일 배타 enum `loss_balance_mode ∈ {adaptive_lambda_legacy(기본), fixed, mse_norm_dann, relobralo, famo, uwso}` + 19개 지원 필드(전부 default-off). 각각:
- `mse_norm_dann`: BCE를 EMA(|BCE|)로 정규화→O(1) + Ganin λ ramp (Ganin et al. 2016).
- `relobralo`: 손실 비율 softmax + Bernoulli lookback + EMA (Bischof & Kraus, arXiv:2110.09813).
- `famo`: log-loss simplex balancer, O(1) streaming `w` 갱신 (Liu et al., NeurIPS 2023, arXiv:2306.03792).
- `uwso`: tempered-softmax over 1/L, closed-form σ (Kirchdorfer et al., arXiv:2408.07985, IJCV 2025).

**파일**: `config.py`(20 필드), `trainer.py`(mutex 검증 + dispatch 3-way[legacy byte-identical/fixed/`_lbm_apply`] + 헬퍼 4종 + state save/load), `run_base_experiments.py`(checkpoint `lbm_state` 저장/복원, 구 ckpt 호환).

**불변식**: recon은 절대 앵커(재정규화 X — BCE 항/disc 비만 재가중; `famo`만 두 task 재가중); weight는 stop-gradient; Ganin reversal ramp(`model._grl_lambda`)와 곱하지 않음(double-ramp 금지); 모든 default는 271(num_epochs=500, warmup=250) ep250→350 구간 calibration. 신규 4모드는 `use_grl=True`+`grl_mode='classifier'` 요구, `use_scad`와 배타(`ValueError`).

**검증(GPU 미사용 — 실험 진행 중)**: import OK(실행 중 큐의 다음 실험 re-import 안전), config defaults 정확, **legacy bit-identical**(`git diff` dispatch 내부 연산 라인 불변, indentation만 +4), 4모드 CPU smoke(정상 0.5·붕괴 1e-4 MSE × onset 250·mid 300에서 loss/weight/grad finite, NaN 0), mutex `ValueError` 4건, state round-trip(FAMO `w`/Adam-step/prev·RNG 복원 + 구 ckpt no-op back-compat). 상세 → Notion subpage "loss_balance_mode — GRL·MSE 손실 균형 6 선택지", `docs/ARCHITECTURE.md` GRL 섹션.

## 2026-06-09: (정정) 위 metadata 버그의 **진짜 원인 = finalize evaluate-path score divergence** (best_checkpoint는 정상)

2026-06-08 항목이 원인을 "async best_checkpoint 정렬불일치(뒤 epoch weights)"로 기술했으나 **틀렸다.** weight-level forensic 재실행으로 `best_checkpoint.pt` = **`model@best_epoch` (bit-identical)** 임을 4 cell(SMAP/P-4@255, MSL/C-2@295, SMD/machine-1-4@470, SMAP/T-3@470)에서 확인. 실제 원인은 finalize의 `evaluate(lite=False)` **anomaly-score 경로가 per-epoch npz / best-epoch 선택 / viz(`derive_pred_data`)와 다른 score**를 낸 것 (= FM-omission류 score-path 중복; 같은 `model@255`에서 evaluate=0.4337 vs npz/selection/viz=0.4858). `0.4337`이 우연히 `epoch_metrics@470`(0.4338)과 비슷해 "epoch 470 model을 썼다"고 오인했을 뿐이다.

**viz는 처음부터 정상**: best_model 그림은 정상 checkpoint(`model@best`)에서 `derive_pred_data`로 생성되고 그 score는 npz와 bit-equal(Δ≈1.8e-5). → **재학습/viz 재생성 불필요.** metadata block(별도 evaluate 호출)만 틀렸고 npz@best 재계산 fix로 이미 교정됨.

**검증(2026-06-09)**: **Audit A**(`scripts/reexp_comprehensive_audit.py`, 370 cell 전수, 재실행 없음) — `metadata.metrics == compute_full_metric_set(npz@best)` 전 metric + `best_epoch==argmax(epoch_metrics)` → **370/370 OK, 0 issue**. **Audit B**(`scripts/reexp_auditB_forensic.py`, 4 simple flip cell, 정확 config 재실행) — `best_checkpoint==model@best`(weight bit-identical) + npz bit-identical → **4/4 OK**. **결정론**: `best_config.json`의 정확 config로 재실행 시 bit-for-bit 재현 (이전 "재현 불가"는 Set-C preset drift였음: `d_model 512→256, batch_size 1024→512, dynamic_margin_k 6→2`. config는 항상 `best_config.json`에서 복원).

## 2026-06-08: Finalize metadata가 best-epoch score와 어긋나던 버그 수정 → npz@best 사용 (원인 설명은 위 2026-06-09 항목으로 정정)

**증상**: `experiment_metadata.json["metrics"]`가 saved best-epoch score와 불일치 (특히 small/simple dataset). `timing.best_epoch`·`epoch_metrics.json`·`epoch_scores/*.npz`·**best_model viz** 전부 정상인데 최종 metric block만 어긋남 — 최악 simple cell에서 **pak_auc_f1 ~0.05** 차. 예: `271/SMAP/P-4` `best_epoch=255`(`epoch_metrics@255`=0.4858)인데 `metrics.pak_auc_f1=0.4337`.

**탐지**: 재실험 Phase 4 strict-consistency 검사에서 non-excl22 FLIP cell **176/200 불일치**.

**수정**: (1) **코드** `run_base_experiments.py` `_bg_worker_body` — re-forward 후 primary/disc/teacher metadata를 **`npz@best_epoch`(선택·viz와 일관된 authoritative score)로 재계산**(excl22 block과 동일 패턴, try/except fallback). (2) **데이터**: 210 flip cell metadata를 `npz@best`로 재계산 → Phase 4 재검증 ALL CONSISTENT(360/360).

**영향**: **INFERENCE-only**(재학습 불필요). best epoch의 record-of-truth는 persisted artifact(npz)여야 하며 score를 내는 두 번째 evaluate 경로를 쓰면 안 됨(single-source `scoring.py` 유지). 상세 → `docs/POST_MORTEMS/2026-06-08_finalize_wrong_epoch_metadata.md`.

## 2026-06-03 (PM): Correction-of-the-correction — nrdetector classifier GATE is faithful; the no-gate `5cff9da` was a regression → **REVERTED**

직전 2026-06-03 (AM) 항목과 commit `5cff9da`("binary mean-gate 제거 → continuous ACTMAP emit")는 **틀렸으며 revert**되었다. nrdetector predict()는 **classifier-gated continuous ACTMAP** = `actmap × [seg_prob ≥ mean(seg_prob)]` 으로 **복원**되었다(= 2026-06-01 형태).

**증상 (실험에서 발견)**: 50 epoch의 saved score가 전부 **bit-identical**(ep1==ep25==ep50, SWaT·WaDi·PSM 모두) + 성능 급락(SWaT pak_auc_f1 0.84→0.44). 같은 run에서 deepmil/treemil/wetas는 epoch마다 변함 → nrdetector 고유 문제.

**근본 원인**: nrdetector의 encoder(DilatedCNN)는 **Stage 0에서 한 번만 학습 후 frozen**이고, per-epoch 학습 루프는 **PU-classifier만** 훈련한다. `5cff9da`가 predict()의 **classifier gate를 제거**하자 score = frozen-encoder actmap만 남아 → (1) epoch마다 불변(bit-identical) → best-epoch 선택 무의미, (2) raw per-window-minmax actmap이 normal-window 점들로 ranking을 오염시켜 붕괴.

**실제 upstream (live 재fetch + 독립 red-team `B_classifier_gated`, integral=yes)**: default `python main.py`(`--mode` default `'train'`)는 `solver.rank_test()`만 실행하며(main.py:44-48), 이것이 랭킹하는 `interested_instance`(solver.py:219)는 `save_instance_files`가 **classifier가 flag한 window(`instance_label[i]>0`, solver.py:198-203)의 actmap만** 모은 것이다. `instance_label` = PU-classifier의 per-window mean-gate(`test_outputs ≥ mean`, solver.py:164-171,185). 즉 **classifier gate가 ranking pool을 결정** — operating-point selector가 아니라 ranked object 자체의 일부. no-gate raw actmap은 upstream이 절대 랭킹하지 않는 점들을 주입하므로 NOT faithful.

**실측 (inference-only, 저장 weights, 재학습 없음)**: SWaT pak_auc_f1 **0.440(no-gate)→0.858(gated)**, roc_auc 0.365→0.883, prc_auc 0.156→0.808. PSM roc_auc 0.356→0.598, prc_auc 0.244→0.333(pak는 30.6% 이상치+PA 관대성으로 약간 낮으나 정직한 ranking 지표는 회복). harness 재현 검증: no-gate 재계산 = saved score(0.5443) 정확 일치. (`temp/verify_nrdetector_three_versions.py`, `temp/verify_swat.py`; canonical: `temp/baseline_faithfulness_audit/NRDETECTOR_CORRECTION_v2.md`)

**영향**: **INFERENCE-only**(retrain 불필요). prior no-gate run의 nrdetector/nrdetector_full score는 **STALE → 재-score 필수**. 코드에는 gate 제거 금지 경고(rank_test 추적 포함)를 docstring에 명시. boundary-safe windowing + per-file leak-free normalization 보존.

> ⚠ 아래 2026-06-03 (AM) 항목은 **이 항목으로 superseded** — "binary mean-gate가 버그"라는 기술은 오판이었다(gate는 faithful, 제거가 버그였음).

## 2026-06-03 (AM, SUPERSEDED — 위 PM 항목 참조): Correction — nrdetector test-time score (binary mean-gate was non-default → continuous ACTMAP restored)

직전(2026-06-02, `f94d7dc`) 항목에서 nrdetector의 test-time score를 upstream `solver.test()`의 **binary transductive segment-gate**(`seg ≥ mean + thre·(max−min)`, thre=0)로 고치고 "official 충실"로 기술했으나, **그 기술은 틀렸으며 commit `5cff9da`(2026-06-03)로 superseded**된다.

**무엇이 틀렸나 (prior "binary segment-gate fix")**: 우리 `predict()`가 test score를 `scores = actmap × binary_gate`로 산출했고, 이 gate는 `window_seg_prob ≥ mean(seg_prob) + anomaly_thre·(max−min)`(`anomaly_thre=0`)였다. 이는 upstream `solver.test()`에서 가져온 것이나, 해당 branch는 **commented-out, NON-DEFAULT `--mode test`** 경로다. hard-binarize 결과 일부 데이터셋에서 score의 **>80%가 exact-zero**가 되어 AUC/PA%K가 필요로 하는 continuous ranking이 붕괴했다.

**실제 upstream (live-fetched UCSC-REAL/NRdetector — main.py + solver.py)**: DEFAULT run(`python main.py`; `--mode` argparse default=`'train'`)은 `solver.rank_test()`를 호출한다(`solver.test()`는 train block에서 주석 처리됨). `rank_test()`는 **continuous per-point min-max ACTMAP**(`point_Score = interested_instance.reshape(-1)`)를 `np.argsort`로 랭킹해 상위 `anomaly_ratio`(default **0.65**) 비율을 flag하고, training-free **HOC** auto-rate re-threshold(`get_hoc_threshold`/`rank_hoc`)를 적용한다. 랭킹되는 연속 객체 = `interested_instance` = upstream `get_dpred`의 dense per-window min-max ACTMAP `(h−min)/max`(`h = fc(out)`). 따라서 prior binary-gate-as-final-score는 **두 upstream branch 어디와도 불일치하는 MAJOR_DEVIATION**(non-default intermediate를 hard-binarize + rank_test/HOC 누락)였다.

**수정 (commit `5cff9da`)**: `predict()`가 이제 **continuous per-window min-max ACTMAP를 직접 방출**한다(= upstream `get_dpred` verbatim, `rank_test`가 랭킹하는 바로 그 객체). binary mean-gate **제거**. 우리 ranking harness(ROC / PRC / pak_auc_f1 — 자체 operating-point 선택)에 충실한 객체이며, upstream의 single-operating-point selector(`anomaly_ratio=0.65` + HOC)는 hard label이 필요한 경우를 위해 in-code로 명시만 한다. `b043dd6`의 boundary-safe per-entity TEST windowing + per-entity leak-free normalization은 **보존**(gate만 제거). 검증: continuous(521/530 distinct, non-binary), `(T_test,)`/finite/higher=anomalous.

**영향**: **INFERENCE-only** 변경(retrain 불필요, weights valid). 다만 prior nrdetector / nrdetector_full score(binary, >80% zeros)는 **STALE → 재-score(re-inference) 필수**. nrdetector·nrdetector_full(동일 코드) 모두 해당. unrelated soft-DTW restoration note는 그대로 유효.

## 2026-06-03: 평가 코드 **decision-threshold convention 통일 + audit 확정 버그 3종 수정** (eval 정합성)

공유 평가 코드(`mae_anomaly/evaluator.py`, MAE·comparison 양쪽이 사용)에서 **F1과 F1_PA가 서로 다른 예측을 입력으로 받던** 정합성 버그를 근본 수정하고, 22-agent forensic audit로 추가 확정된 버그 2종을 함께 고쳤다.

**근본 원인 (F1_PA가 F1보다 작아지는 모순)**: raw F1·Aff는 `score >= thr`로 이진화하는데 point-adjust F1(`compute_pa_k_metrics_from_mean_scores`) 등은 `score > thr`(strict)였다. AR-quantile threshold가 **동률(plateau) 점수값에 걸리면**(예: random binary score는 thr=1.0=max) `>`는 0개를 flag → F1_PA=0인데 raw F1(`>=`)는 비0 → point-adjust 불변식 `F1_PA >= F1` 위반. random의 F1_PA=0이 그 증상이었다.
- **수정**: 단일 decision-threshold 이진화 전 지점을 `>=`로 통일 (evaluator.py 200/471/723/788/925/1109/1242/1986/2160 + 두 sweep 1050/554는 linspace라 direction-invariant이나 동일 통일). comparison 측 `baseline_common.py:585`의 raw F1도 `>` → `>=`. 연속 점수(MAE 포함)는 동률이 없어 값 불변 — 코드 일관성만 확보.

**Audit 확정 버그 (forensic 22-agent workflow)**:
1. **(critical) excl22가 VUS/Affiliation/AR에서 eval_mask 무시**: `compute_full_metric_set`가 VUS·Aff·AR을 full `point_*`로 계산해 SWaT(excl22) 값이 full과 **byte-identical**(region-22 = SWaT 이상의 84%가 그대로 누설). → masked `base_scores/base_labels` 사용 (evaluator.py 970-977).
2. **(major) K=0 point-adjust guard 누락**: `compute_pa_k_auc`·`compute_pa_k_roc_prc_from_mean_scores`가 K=0에서 `ratio>=0` 자동참 → zero-detection 구간도 검출 처리(do-nothing 모델이 F1@K=0=1.0). pak_auc_f1·**pak_auc_prc_auc(랭킹 정렬 키)**·pa_0_* inflation. → `sums>=1` guard 추가 (evaluator.py 563/1085/1115). K>0은 no-op.

**저장값 일괄 재계산 + 완전성 게이트**: 26 baseline + MAE **271/274/284** × 5 데이터셋 = **전 셀**을 saved score에서 고친 코드로 재계산 (silent skip 금지 — 누락 시 LOUD 실패; 사전검증 145/145 셀 존재). 영향 지표: F1_PAK·PRC_PAK(전체, K=0 guard), VUS_PR·VUS_ROC(excl22, mask). F1·F1_PA·Aff는 uniform `>=`·masked로 별도 재계산.

**검증**: 전 모델 `F1_PA >= F1` 위반 0; excl22 VUS가 full과 분리됨(pca 0.7707→0.3774); K=0 guard로 anti-correlated 모델 pak_auc_f1=0.18(≠1.0); F1_PAK sweep 변경은 영향 0 실측(random recomputed=stored=0.405516). **271/274/284 재판단: 고친 지표에서도 271이 최균형**(MeanRank 271=1.312 < 274=1.438 < 284=1.812; 271 전체 1/27위). 백업 `./.trash/0603/{evaluator,baseline_common}.py.bak_*`.


## 2026-06-02: Comparison **boundary-safe per-entity TEST windowing** (21 windowing baselines)

**문제 (test-time cross-entity window)**: 정규화는 per-entity로 고쳤으나, **test 추론의 sliding/non-overlap window는 여전히 전체 concat test를 경계 무시하고 분할**했다. multi-entity 데이터셋(SMD 28머신·SMAP 54채널·MSL 27채널·Exathlon 6앱)에서 entity 경계를 가로지르는 window가 두 entity 데이터를 섞어 그 경계 부근 score를 오염시킨다. harness는 **train**만 boundary-safe(`create_train_windows_boundary_safe`/`train_segments`)였고 **test-side 메커니즘은 전무**했다 (검증 결과: 21개 windowing 모델 전부 test-boundary-unsafe — `test_segments`는 정규화에만 쓰이고 windowing은 전체 분할).

**수정 (per-entity test windowing, side-effect-free)**:
- 공유 helper `comparison/baselines/_boundary_safe_window.py::per_entity_concat(test_X, test_segments, raw_fn)`: 각 entity의 test slice에서 **windowing+inference를 독립 수행** → 어떤 window도 경계를 안 넘음 → raw per-timestep score를 concat. **score 후처리(median-IQR/smooth/max/softmax-axis)는 concat 후 전체 test 기준 그대로** → granularity 불변. 단일 entity/단일파일/None → **1회 호출 = bit-identical NO-OP**.
- 21개 wrapper가 windowing+inference를 per-entity로 경유. **정규화는 클래스별로 전부 불변**: harness-norm(이미 per-entity), self-norm(기존 per-entity transform 유지 후 per-entity window), **6 RevIN SOTA**(timesnet/tfmae/memto/moderntcn/dcdetector/catch — whole-test StandardScaler를 slice 전 1회 적용, slice-invariant, **windowing만** per-entity). NEURAL은 `test_segments`를 `run_dl_baseline_with_epoch_eval`+`_predict_gated`로 plumbing.
- **official 충실**: upstream이 entity를 개별 처리함을 재확인 (npsr `evaluation.py` "only one entity should be input at a time", wetas/deepmil per-file chunking, omnianomaly single-entity). 즉 per-entity windowing이 곧 official 충실이고 기존 concat-naive가 deviation.

**train boundary-safety 재검증**: dispatch+`create_train_windows_boundary_safe`(실제 `start<b<end` skip) 추적 → **non-simple 모델은 전부 train-boundary-safe.** NEURAL(mlp/mlpmixer/transformer)은 pre-built boundary-safe window(`create_train_windows_boundary_safe`), SOTA 14·WEAK 4는 `fit(train_segments=...)`에서 `compute_segment_safe_window_indices`로 cross-boundary window를 drop. (초기 재검증에서 dagmm을 train-unsafe로 적었으나 — `dagmm.fit`의 multi-line 시그니처를 단일줄로 grep한 false-negative 아티팩트였고, 실제로는 `git 0dc76ce`부터 `train_segments`를 수용·사용함을 코드로 확인. 정정함.)

**검증**: 19/19 파일 adversarial gate **GREEN** — single-entity **bit-identical**(side-effect 없음), 경계 미crossing, 후처리·정규화 granularity 불변, contract, py_compile/import OK. **영향**: SMD/MSL/SMAP/Exathlon(multi-entity)은 재실행 시 corrected 수치; 단일파일(PSM/SWaT/WaDi)은 bit-identical.

## 2026-06-02: Comparison 파이프라인 **per-entity 정규화 일원화** + baseline faithfulness 재감사 fix

MAE 파이프라인 fix(아래 항목, `f94d7dc`)는 `comparison/` baseline 경로를 고치지 않았다(comparison은 자체
`unified_loader`에서 whole-array로 `_minmax_per_feature`를 직접 호출). 본 작업은 그 위에 comparison 측을 **동일한
`entity_norm_segments` 단일 소스**로 일원화한다.

**per-entity 정규화 (comparison)**:
- `comparison/data/unified_loader.py::get_file_norm_segments()`가 **`data_info['entity_norm_segments']`**(main이 emit)를
  단일 소스로 소비 → 4개 multi-entity concat(smap/msl/smd_concat/exathlon_concat) per-entity, 단일파일/`*_simple`은 no-op.
- Path A(harness minmax/zscore, 14모델): 신규 공유 커널 `comparison/baselines/_per_file_norm.py`로 entity별 leak-free
  fit(train)/transform(test). single-file은 legacy `_minmax_per_feature(clip=False)`/`_standardize_per_feature`와 **bit-identical**.
- self-norm 6모델(npsr·anomaly_transformer·wetas·deepmil·treemil·nrdetector): wrapper 내부에서 per-entity(scaler-pluggable,
  모델별 scaler 보존), `test_segments` plumbing은 `run_baseline.py`→`baseline_common.py`에서 `inspect.signature` sibling-gate.
- **이중정규화 0** 증명(none→wrapper vs harness minmax/zscore 상호배타); **6 untouchable SOTA**(timesnet/tfmae/memto/
  moderntcn/dcdetector/catch)는 upstream이 whole-array라 무수정. comparison 고유 차이(minmax **clip=False** paper-faithful,
  모델별 scaler identity)는 의도적으로 보존.
- `load_smd`('smd', multi-machine)에도 `entity_norm_segments` emit 추가 → per-entity가 **SMD 로더 setting 무관 일관**
  (smd_block_split은 단일 머신 K-block split → single-entity no-op이 정확).

**faithfulness fix (strict 재감사)**: moderntcn score-aggregation을 last-position-only(`err[:,-1]`)→**all-position
overlap-average**(upstream flatten 복원); nrdetector를 continuous `act×sigmoid(seg)`→**upstream `solver.test`의 binary
transductive gate**(`seg≥mean+thre·(max−min)`, thre=0) verbatim; wetas test 정규화를 transductive fit-on-test→**train-scaler
transform**(leak 제거). wetas의 방출 score `dscore`는 upstream `dauc`/`dauprc` 입력과 동일 = 본 실험 ranking metric
(pak_auc_f1/ROC/PRC) 기준 faithful (paper headline인 DTW-aligned F1/IoU는 본 실험 평가 대상 아님 → 미산출 유지).

**검증**: 26모델 최종 per-model verify(faithfulness + 파이프라인 trace + normalize-once + concat/simple 양쪽 + contract);
kernel 단위테스트 통과(per-entity 독립·leak-free·single-file bit-identical). **영향**: comparison SMD/MSL/SMAP/Exathlon +
moderntcn/nrdetector 결과는 코드 변경으로 무효 → 재실행 시 corrected 수치 생성(재실행 자체는 미수행).

## 2026-06-02: Fix — concat 멀티-entity 데이터셋 **per-entity 정규화** (whole-array → per-entity)

**문제 (whole-array fit)**: `SMAP_concat`/`MSL_concat`/`SMD_concat`/`Exathlon_concat`은 entity(채널/머신/app)를
`[e1_train..eN_train | e1_test..eN_test]`로 이어붙인 raw 배열을 반환했고, 하류 `SlidingWindowDataset`이
`signals[:train_end]` **전체(=통짜 concat train)에서 단 한 세트의 통계로 정규화**했다. entity마다 절대 스케일이
다를 때(예: A센서 100±5 vs B센서 5±0.5) 통짜 z-score는 A→≈+1·B→≈−1(entity별 상수)로 만들고, 각 entity 내부의
진짜 변동(std 5, 0.5)을 큰 전체 std(≈48)로 나눠 ≈0.1·0.01로 뭉갠다. 결과적으로 모델이 보는 주신호가 "이상 여부"가
아니라 "어느 entity 출신"이 되고, 절대값 작은 entity의 이상은 가려지며, 단일 threshold도 일그러진 분포 위에서 잡힌다.

**수정 (per-entity fit, leakage-free)**:
- 4개 concat 로더가 `data_info['entity_norm_segments']`(= entity별 `(train_len, test_len)`, concat 순서)를 emit.
- `SlidingWindowDataset(entity_segments=...)` → 각 entity를 **자기 train 구간만으로 fit**(`_normalize_per_entity`),
  config의 동일 mode/range/clamp(zscore / minmax 0_1 / neg1_1+clamp) 재사용(`_apply_normalization` 단일화).
  per-entity라 단일 global scaler 없음(`scaler_*=None`; 외부 소비처 없음). `entity_segments=None`(SWaT/WaDi/PSM
  단일 entity)이면 기존 whole-array 경로 그대로(=동작 불변).
- `_standardize_per_feature`의 통계 계산을 **float64**로 변경(near-constant·large-offset feature에서 float32
  axis-0 합산 오차가 작은 std로 나뉘며 정규화 train mean을 ~0.02 흔들던 정밀도 artifact 제거). minmax는 min/max라 무관.
- run_base 4개 생성부 + `reviz_one_best_model`/`backfill_score_contribution`/`NoisyLabel`에 `entity_segments` 전달.
  (viz/*.py·ablation은 run_boundaries조차 안 넘기는 simulation/legacy 경로라 concat 비대상.)

**검증(실측)**: per-entity zscore train mean ≈0 (MSL 5e-6, SMD 1.4e-4) / unit-std(err<7e-4) — whole-array는
entity 미중심(MSL 2.08, SMD 4.64)으로 버그 재현. 누수 0(test-region mean 자유). boundary 0-crossing 보존
(MSL/SMD train+test 전부 CROSS=0). minmax 0_1 per-entity(각 entity train min0/max1) True. 단일 entity(PSM) global 경로 불변.

**영향**: 기존 SMAP/MSL/SMD concat 결과(whole-array)는 무효 → 삭제(재실행 대상). SWaT/WaDi/PSM(단일 entity)은 영향 없음.

## 2026-06-01: 멀티-세그먼트 데이터셋 `concat` / `simple` 등록 (SMAP·MSL·SMD·Exathlon)

채널/머신/app 으로 나뉜 4개 데이터셋을 일관된 **`<DS>_concat`**(전 세그먼트를 한 스트림으로) /
**`<DS>_simple_<seg>`**(세그먼트 1개 = 1 데이터셋) 네이밍으로 정리·등록.

- **`SMAP_concat` / `MSL_concat`** — 전 채널 time-concat. 채널별 test safe-cut(front→train / back→test).
  SMAP=54ch×25feat, MSL=27ch×55feat. (로더는 있었으나 `DATASET_LOADERS` 미등록이라 실행 불가했던 것 등록)
- **`SMD_concat`** (신규 `load_smd_concat`) — 전 28머신 concat, 머신별 test-cut(orig_train + front50%test → train,
  back50%test → test). 38feat. (기존 `'smd'`는 원본 split 유지=legacy)
- **`Exathlon_concat`** (신규 `load_exathlon_concat`) — 전 app concat. app별 `load_exathlon` 결과를 train_ratio로
  분리 후 `[all_train | all_test]` 재병합. 19feat.
- **`SMAP_simple_<ch>` / `MSL_simple_<ch>` / `SMD_simple_<machine>` / `Exathlon_simple_app<N>`** — per-segment
  (test-cut/원-split). SMD/Exathlon run_base `DATASETS` 키도 이 일관 이름으로 변경(`results_subdir` 유지 →
  집계·기존 결과 dir·`smd_all` 자동 호환).

**경계 차단 검증(채널/머신/app/trace/orig-train·test-front seam + train|test)**: `run_boundaries`가 모든
세그먼트 불연속을 기록하고, `SlidingWindowDataset._extract_windows`가 윈도우 `[start,end)` 안에 경계 b가
`start<b<end`면 스킵. train|test 경계는 split을 별도 인스턴스(`signals[:train_end]`/`[train_end:]`)로 분리해
inherently 차단. 실측: 4 concat × (train+test) **총 ~72,000 윈도우 중 경계 교차 0건** (SMD 27501·MSL 1835·
SMAP 9864·Exathlon 33247). run_base는 train·test 양쪽 dataset 모두에 `run_boundaries` 전달.

**minmax leakage 없음**: `_minmax_per_feature(signals, train_end)`가 `signals[:train_end]`(train, train_ratio
경계)에서만 min/max fit 후 전체 적용 — test 부분(평가셋)은 fit 제외. test-front는 설계상 train(chronological
prefix)이라 누설 아님. z-score 동일.

등록 위치: `mae_anomaly/datasets/loaders.py`(`DATASET_LOADERS` + 동적 루프, `load_smd_concat`/`load_exathlon_concat` 신규),
`scripts/run_base_experiments.py`(`DATASETS` concat 4 + `SMAP_MSL_SIMPLE_DATASETS` + `SMD_DATASETS`/`EXATHLON_DATASETS` 일관 키).

## 2026-06-01: GRL `balanced_acc=0.5` 기만 — degeneracy gap `grl_acc_gap` 추가

### Problem
`grl_balanced_acc = (anomaly_acc + normal_acc)/2` 는 **완전 퇴화**(분류기가 모든 패치를
anomaly 로 → `anomaly_acc=1, normal_acc=0`)에서도 **0.50** 을 내어 "random 정상"으로
오독된다. exp287_unmask SWaT ep~340+ 에서 실제 발생.

### Fix (live pipeline)
- **수집(단일 파생점):** `trainer.py` history `train_grl_acc_gap` += `|normal−anomaly|`
  (이미 저장되는 epoch-mean 정확도에서 파생 — WGAN/WDGRL 계산 경로 미변경).
  `run_base_experiments.py` `cb_metrics['grl_acc_gap']` 파생 + excl22 복사 목록에 추가.
- **시각화:** `epoch_grl.png`(plot_epoch_metrics) + `GRL_contribution_trend.png`(C, best_model_visualizer)
  두 패널에 gap 곡선 + `gap≥0.8` 퇴화 음영. balanced_acc 는 유지(쌍으로 읽어야 degeneracy 판정).
- **백필:** 9 GRL-active run × 전 cell(45 파일, ~4490 ep) `epoch_metrics.json` 에 grl_acc_gap
  파생 추가 — 안전게이트(백업→검증→쓰기→재대조), bad=0.
- **재시각화:** `scripts/reviz_grl_gap.py` — epoch_grl 45/45 + GRL_contribution 45/45, 오류 0.

### 미변경 (의도)
WGAN/WDGRL 경로(score 기반), 과거 동결 ES 분석 스크립트(`early_stopping_analysis_v*`,
`build_es_notion_*` — balanced_acc 를 ES 신호로 사용; 기만 위험 post-mortem 에 기록만).
백업: 코드 `./.trash/0601/grl_gap_code/`, 데이터 `./.trash/0601/grl_gap_backfill/`,
PNG `./.trash/0601/grl_gap_viz_backup/`. 상세: `docs/POST_MORTEMS/2026-06-01_grl_balanced_acc_deceptive_degeneracy_gap.md`.

## 2026-06-01: SWaT excl22 epoch_metrics `recon_snr` + `fm_loss` always None (copy-list omission)

### Problem (관측됨)
SWaT/A1A2_excl22 의 `epoch_metrics.json` 에서 `recon_snr` 와 `fm_loss` 가 **100개 eval 전부
None**. 같은 자리의 `disc_snr` 는 full-SWaT 값이 byte 단위로 그대로 들어 있어, excl22 대상
recon/fm 분석이 조용히 None 을 읽었다 (6 모델 + 285/286/287 excl22 전부 해당).
full↔excl22 키셋 전수 비교로 dataset-wide 진단 누락은 이 둘 뿐임을 확인 (나머지 full-only
키는 excl22_* prefix 중복 / excl22-종속 counts / disturbing detection / 내부 _* viz 배열).

### Root cause (코드 단위)
excl22 worker (`run_base_experiments.py`, swat_eval_mode='excl22') 는 detection 지표를
`compute_metrics_with_exclusion` 로 계산하고(SNR 미포함), test-region 무관한 dataset-wide
진단(`disc_snr`, `d_*`, `grl_*`)만 full-SWaT epoch_metrics 에서 per-eval 복사한다.
`recon_snr` 은 2026-05-29 신규 추가 필드인데 이 **복사 목록(L1928)에 추가되지 않아** 누락
→ excl22 epoch_metrics 에 키 자체가 없어 None. (recon_snr 은 disc_snr 과 동일 성격의
dataset-wide 값 → 복사가 올바른 동작.)

### Fix
- **[run_base_experiments.py L1928]** excl22 per-epoch 복사 목록 맨 앞에 `recon_snr`, `fm_loss`
  추가 (`['disc_snr', 'recon_snr', 'fm_loss', 'd_loss', ...]`). `if key in full_epoch_data[ep_num]`
  가드로 legacy 안전. excl22 epoch_metrics.json 에만 영향 — disc_snr·detection·best-epoch 선정 불변.
  (`fm_loss` 도 dataset-wide 학습 진단 — 9 run 중 8 run 에서 post-warmup non-zero, max 0.10~0.43.)

### Verification
- py_compile OK. full-SWaT epoch_metrics 로 복사 로직 재현: recon_snr **100/100**, fm_loss
  **100/100 복사**(수정 전 0), disc_snr **100/100 무회귀**, mismatch 0. full↔excl22 키셋 전수
  비교로 copy-class 누락이 recon_snr·fm_loss 둘뿐임 확정. 원본 백업 `./.trash/0601/`.

### Note
forward-only fix. **이미 완료된 excl22 epoch_metrics 의 None 은 미수정** — 필요 시 대응
full-SWaT epoch_metrics 의 recon_snr 을 epoch 단위로 복사하는 backfill 로 보정 가능
(사용자 확인 후). 상세: `docs/POST_MORTEMS/2026-06-01_excl22_recon_snr_copy_omission.md`.

## 2026-06-01: Pre-warmup recon-only anomaly score (frozen-student disc/FM leak fix)

### Problem (관측됨)
`teacher_only_warmup_epochs > 0` 실험에서 teacher-only warmup 구간(eval epoch ≤ warmup)의
anomaly score 에 **frozen / random-init student 의 disc·FM 항이 혼입**. Warmup gate 가
학습(trainer/loss)만 막고 평가(evaluator)는 항상 full forward 를 돌려, student 가 아직
학습되지 않은 구간에서도 disc/FM 가 score 에 더해졌다. `w_disc=0` 만으로는 FM 항이 별도
분기로 남아 누수 지속.

### Root cause (코드 단위)
eval 경로(`evaluator._apply_scoring_formula` → `scoring.compute_adaptive_components`)에
warmup 개념이 없었음. recon-only 보장은 `w_disc=0` **AND** `fm_active=False` 동시 필요.

### Fixes
- **[scoring.py] `is_prewarmup_epoch(config, epoch)`** 단일 게이트 술어 추가
  (`0 < epoch ≤ teacher_only_warmup_epochs`; None/warmup0 → False). evaluator·npz·contrib·
  train-scoring·viz 가 공유.
- **[scoring.py] `force_recon_only: bool` required keyword-only** 를
  `compute_adaptive_components` / `compute_adaptive_point_score` / `compute_score` 에 추가.
  True → `w_disc=0` & `fm_active=False` → `score == recon` (정확). 누락 호출자는 즉시
  `TypeError` (API-change checklist 규칙 #2).
- **[evaluator.py] `set_eval_context(*, epoch)`** + `self._force_recon_only` —
  `_apply_scoring_formula` 가 adaptive 분기에 플래그 forward.
- **[run_base_experiments.py, gitignored runner]** per-epoch eval(`epoch=ep`),
  npz adaptive_score, contribution, best-epoch train scoring, best-model viz,
  final bg-worker eval(`epoch=timing['best_epoch']`) 게이트 연결. raw npz 배열은 항상 raw.
- **[base.py / best_model_visualizer.py, gitignored viz]** post-hoc viz 는 epoch 맥락 없음 →
  `force_recon_only=False` (legacy full score) 명시; `derive_pred_data` 에 게이트 kwarg 추가.

### Verification
doctest 12/12; `is_prewarmup_epoch` 경계(250=pre,251=post,None=post,warmup0=post);
모듈 `force_recon_only=True → score==recon` 정확 일치; evaluator 배선
(ep250=recon-only, ep251=full, None=full); 13개 게이트 호출부 전수 `force_recon_only` 전달;
편집 5파일 `py_compile` 통과. 상세: [docs/POST_MORTEMS/2026-06-01_prewarmup_student_score_leak.md](POST_MORTEMS/2026-06-01_prewarmup_student_score_leak.md).

### Note
tracked 변경: `mae_anomaly/scoring.py`(+게이트), `mae_anomaly/evaluator.py`,
`mae_anomaly/types.py`(단일소스 PatchScoresBundle), 본 문서, post-mortem, ARCHITECTURE.
런타임 fix 의 다수가 위치한 `scripts/run_base_experiments.py` 와
`mae_anomaly/visualization/*` 는 `.gitignore` 로 추적 제외(디스크 활성, commit 대상 아님).
완료 실험 소급 재계산(pre-warmup npz + epoch_metrics + best_epoch 재산정)은 별도 backfill.

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
## 2026-05-30: Weak-supervised baseline fidelity rework (normalization · DeepMIL encoder · NRdetector params · provenance gate)

Root-cause 분석 (`temp/ssl_official_baseline_porting_0529/rework_execution_after_root_cause/17_PROVENANCE_FAILURE_ROOT_CAUSE.md`) 후, 4 weak baseline (`deepmil`/`wetas`/`treemil`/`nrdetector`) 의 충실도 결함을 코드 레벨에서 수정. **기존 22개 unsupervised baseline·metric·output 코드 무수정.** 상태 = 구현 완료 · CPU dry-test 통과 · **GPU 미실행** (결과표 weak 행 수치 0개; Q3 = N/A 구조적 부적합, Q1 = pending).

**1) Normalization fidelity (핵심):** 이전엔 4 weak 모델이 pipeline **global MinMax scaler** 를 받아 원논문과 불일치(silent fidelity 결함)했음. `run_baseline.py:240` 에 `SELF_NORMALIZING_WEAK={wetas, treemil, nrdetector, deepmil}` 추가 → 4종 모두 **raw 데이터 수신**(`normalize_mode='none'`) 후 각 wrapper 가 **원논문 normalization 자체 적용**:
- WETAS = per-recording StandardScaler (z-score, `timeseries.py:35-40`)
- TreeMIL = per-file StandardScaler (z-score, `timeseries.py:53-55`) — 이전 deviation 이 문서에 silent 였음 → `MODELS.md §25` 에 명시
- DeepMIL = per-recording StandardScaler (WETAS-lineage)
- NRdetector = per-split z-score StandardScaler (`data_loader.py:50-55`)
- residual (결함 아님): `predict(test_X)` 가 test segment 경계를 안 받는 contract 제약 → per-recording test 불가, transductive whole-test z-score 로 대체 (train→test leakage 없음, scope gap = granularity 뿐).

**2) DeepMIL encoder → WETAS DiCNN + optimizer:** 이전 bespoke `TSSegmentEncoder` (NON_OFFICIAL) 제거 → encoder = **WETAS `DilatedCNN`** (DERIVATIVE_CITED, WETAS ICCV'21 p.7360 "DeepMIL employs the same model architecture with WETAS (i.e., DiCNN)"; input=F, hidden=out=128, kernel=2, n_layers=7, RF=128). **OFFICIAL 아님** 명시 (Sultani 원논문 = video/C3D, 학습형 TS encoder 부재). head+loss 는 Sultani CVPR'18 FAITHFUL 유지 (`D→512→32→1`, hinge margin 1.0, λ_smooth=λ_sparse=8e-5, L2=0.001). scoring = **dense per-timestep** (32-seg broadcast 은퇴; `n_segments` vestigial). **optimizer = Adam lr=1e-4** (WETAS `train_classifier.py:234` 출처) — Sultani 의 Adagrad lr=0.01 은 frozen-C3D shallow head 전용이라 deep DiCNN 과 joint 학습 시 발산(logits→-200/-440, score collapse)하므로 encoder 출처 optimizer 채택. preset 도 `optimizer='adam'`/`lr=1e-4` 로 갱신.

**3) NRdetector encoder_lr 파라미터화 + confound 문서화:** `encoder_lr` 을 preset 에 노출 (`encoder_lr=1e-3`, 이전 wrapper 하드코딩). `encoder_epochs=50`/`encoder_lr=1e-3` = **IMPL-INVENTED** — 공식은 encoder 학습 recipe 없이 pretrained `.pth` 로드만 (`modules/extractor.py:65`); 우리는 pth 미보유라 from-scratch BCE-only 학습 → 출처 없는 50/1e-3 은 confound 로 문서화. `prior=None` → 런타임 동적 추정 (PU class prior = intrinsic anomaly ratio, 데이터셋 의존 → estimate 가 맞음; 공식 고정 0.25 우회는 의도적). `noisy_rate=0.4` = reveal fraction (양성 라벨 40%만 공개, `selector.py:31-39`) = 실험 knob 이라 고정.

**4) Parameterization (R3b):** 4 weak 모델 고정값 전부 `baseline_common.py` preset 에 노출 (하드코딩 금지). 분류 구분 — fixed-param / runtime-estimated (`prior`) / normalization / impl-invented (`encoder_epochs`·`encoder_lr`). default 값은 `MODELS.md §23–26` 및 `GUIDE.md §20`.

**5) Provenance gate G1–G5 (재발방지, project-wide):** `GUIDE.md §7.1` + `MODELS.md` weak 섹션에 신설 — G1 모든 컴포넌트 `{FAITHFUL|DERIVATIVE_CITED|NON_OFFICIAL}`+source locus ("design choice" escape 금지) · G2 NON_OFFICIAL ⇒ ≥5-round source-chain · G3 in-project sibling cross-check · G4 provenance≠comparability routing · G5 vendored baseline VCS 가시성.

**문서 변경:** `comparison/GUIDE.md` (§7 분류표에 "Weakly Supervised (4)" 행 + 개수 주석, §7.1 provenance gate 신설, §20 normalization fidelity 표·DeepMIL DiCNN/optimizer·충실도 주의 갱신), `comparison/MODELS.md` (§23 DeepMIL encoder/optimizer/dense/normalization 재작성, §24 WETAS·§25 TreeMIL·§26 NRdetector normalization 명시, §26 encoder confound+verbatim 발췌+prior 동적+noisy_rate knob, weak intro 에 normalization fidelity·G1–G5). DeepMIL CVPR'18 DOI = `10.1109/CVPR.2018.00678` (검증).

**코드 변경 (참조):** `comparison/run_baseline.py` (`SELF_NORMALIZING_WEAK`), `comparison/baseline_common.py` (nrdetector `encoder_lr` 파라미터화 + deepmil `optimizer`/`lr` 갱신 + 4종 normalization/분류 주석), `comparison/baselines/{deepmil,wetas,treemil,nrdetector}/{model,wrapper}.py` (상세 work-log: `temp/ssl_official_baseline_porting_0529/rework_execution_after_root_cause/09_CODE_REWORK_LOG.md`).

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
