# Post-mortem — 2026-06-01 — Pre-warmup student disc/FM leaked into the anomaly score

## Summary

`teacher_only_warmup_epochs > 0` 으로 학습된 모든 실험에서, **teacher-only warmup
구간(eval epoch ≤ `teacher_only_warmup_epochs`)의 per-epoch 평가가 frozen / random-init
student decoder 의 output-discrepancy(disc) 및 feature-matching(FM) 신호를 anomaly score
에 포함**시켜 왔다. Warmup gate 는 **학습(trainer/loss)만** teacher-only 로 막을 뿐,
평가(evaluator)는 항상 full forward 를 수행하므로 student 가 아직 학습되지 않은 구간에서도
disc/FM 항이 score 에 더해졌다.

`teacher_only_warmup_epochs <= 0` (warmup 미사용) 실험은 영향 없음.
`anomaly_score_mode != 'adaptive'` 실험도 영향 없음(disc/FM 항이 adaptive 분기에만 존재).

## Impact

| 산출물 | warmup 구간 정확성 | 비고 |
| ---- | ---- | ---- |
| `epoch_metrics.json` 의 pre-warmup entry (`pak_auc_f1`, `prc_auc`, `f1_t`, `aff_f1`, …) | ❌ random-init student disc/FM 혼입 | per-epoch eval path |
| `epoch_NNN_scores.npz["adaptive_score"]` (ep ≤ warmup) | ❌ disc/FM 혼입 | npz save path |
| `experiment_metadata.json["metrics"]` (best_epoch 이 pre-warmup 일 때만) | ❌ | bg-worker final eval |
| `best_epoch` 선정 | ⚠️ 양방향으로 변동 가능 | gate 가 pre-warmup score 를 **올릴 수도 내릴 수도** 있음 (271 PSM ep100: 0.7930→0.7946) |
| `epoch_NNN_scores.npz["teacher_recon_error" / "discrepancy_error" / "fm_error"]` | ✅ 항상 raw | gate 무관, offline 재계산의 ground truth |
| post-warmup entry (ep > warmup) | ✅ 변화 없음 | gate=off → byte-for-byte 동일 |

영향 받은 실험: warmup 을 사용한 271–287 (288 은 SWaT-only paused 라 scope 제외).
각 cell 의 pre-warmup npz(`adaptive_score`)와 pre-warmup `epoch_metrics` row 가 재계산
대상. **raw 배열(teacher_recon_error 등)은 항상 정확**하므로 모델 재학습 없이 offline
재계산 가능(단, best_epoch 이 flip 되어 저장된 `best_checkpoint.pt` 의 epoch 과
달라지는 cell 의 best-model per-sample PNG / `best_epoch_train_scores.npz` 는
weight 부재로 재생성 불가 → scalar metric 만 재계산, 시각 산출물은 caveat 표기).

## Root cause

Warmup gate(`teacher_only_warmup_epochs`)는 **학습 경로에만** 적용된다:
`model.forward(teacher_only=True)` 가 student decoder/projection/GRL/SCAD 를 skip 하고
`loss.py` 가 disc/FM 를 0 sentinel 로 둔다. 그러나 평가 경로
(`evaluator._compute_patch_scores_all_patches` → `_apply_scoring_formula`)는
warmup 여부와 무관하게 항상 student 를 포함한 full forward 를 돌리고, adaptive score
공식(`scoring.compute_adaptive_components`)이 그 disc/FM 를 그대로 더했다. 즉 **eval 에는
warmup 개념이 없었다.**

추가로, `w_disc=0` 만으로는 누수가 막히지 않는다 — `scoring.py` 의 FM 항은 별도
분기(`fm_active and fm is not None`)라 `w_disc=0` 이어도 `scaled_fm` 이 남는다. recon-only
를 보장하려면 `w_disc=0` **그리고** `fm_active=False` 가 동시에 필요하다.

## Fix (2026-06-01)

단일 게이트를 `mae_anomaly/scoring.py` 에 도입:

1. **`is_prewarmup_epoch(config, epoch)`** — `0 < epoch ≤ teacher_only_warmup_epochs`
   판정의 **단일 소스**(정책 전용, score 수식 없음). `epoch=None` 또는 warmup 미사용 →
   `False`. evaluator / npz-save / contribution / train-scoring / best-model viz 가
   모두 이 함수를 공유.
2. **`force_recon_only: bool`** 을 `compute_adaptive_components` /
   `compute_adaptive_point_score` / `compute_score` 의 **required keyword-only** 인자로
   추가. `True` 면 `w_disc=0` **및** `fm_active=False` 를 동시에 적용 → `student_error=0`
   → `score == recon` (정확히 teacher reconstruction). API-change checklist 규칙 #2 에 따라
   `Optional=None` 기본값 대신 required-kw 로 만들어, 누락 호출자는 런타임 `TypeError` 로
   즉시 드러나도록 했다(2026-05-29 FM-omission 류 재발 방지).
3. **`Evaluator.set_eval_context(*, epoch)`** — 평가 직전 호출하여
   `self._force_recon_only = is_prewarmup_epoch(config, epoch)` 설정.
   `_apply_scoring_formula` 가 이 플래그를 adaptive 분기에 forward.
4. `run_base_experiments.py` 의 per-epoch eval(`compute_epoch_test_eval(epoch=ep)`),
   npz adaptive_score(`force_recon_only=is_prewarmup_epoch(config, ep)`),
   contribution(`compute_contrib_from_eval_data(epoch=ep)`),
   best-epoch train scoring(`is_prewarmup_epoch(config, best_epoch)`),
   best-model viz(`derive_pred_data(force_recon_only=…)`),
   final bg-worker eval(`set_eval_context(epoch=timing['best_epoch'])`) 모두 게이트 연결.
5. 순수 post-hoc viz 호출(`base.py`, `best_model_visualizer.py`)은 epoch 맥락이 없으므로
   `force_recon_only=False`(legacy full score) 명시.

raw 배열(teacher_recon_error/discrepancy_error/fm_error)은 게이트와 무관하게 항상 raw
저장 → SWaT excl22 path 는 게이트된 `adaptive_score` npz 를 그대로 읽어 자동으로 일관.

### 검증

- `scoring.py` doctest 12/12 통과; `is_prewarmup_epoch` 경계값(250=pre, 251=post,
  None=post, warmup0=post) 통과.
- 모듈 레벨: `force_recon_only=True → score == recon` **정확 일치**(`np.array_equal`),
  `False → recon + scaled_disc(+scaled_fm)`.
- Evaluator 배선: `set_eval_context(epoch=250)` → recon-only 정확, `epoch=251` → full,
  `epoch=None` → full.
- 13개 게이트 함수 호출부 전수 grep → 전부 `force_recon_only` 전달 확인.
- 편집 5개 파일 `py_compile` 통과.

## Offline recompute (영향 실험 소급 정정)

- **R1** pre-warmup npz `adaptive_score` 를 raw `teacher_recon_error` 로 덮어쓰기(recon-only).
- **R2/R3** pre-warmup `epoch_metrics` row 를 `compute_full_metric_set` 로 재계산
  (teacher_* 키 재사용 금지 — lite=True 라 VUS/Aff/R-F1 누락이므로 full 재계산 필수).
- **R4** corrected `best_epoch` 재산정; flip 되어 저장 ckpt epoch 과 다른 cell 은
  `experiment_metadata.metrics` scalar 만 npz 로 재계산(VUS 포함 — point score 만 필요),
  best-model per-sample PNG / train-score npz 는 weight 부재로 **STALE 표기**(재학습 없이
  재생성 불가).
- **R5** 모든 in-scope cell(45)의 pre-warmup npz coverage 완비 확인 → 재학습 0건으로
  scalar 정정 가능. (feasibility 한계는 누락 npz 가 아니라 누락 per-epoch model weight.)

## Execution outcome (2026-06-01, 실측)

`scripts/backfill_prewarmup_recon_only.py --apply` (OMP=1, 14 worker, 6114s) 로 45 cell 정정.

- **R1** pre-warmup npz `adaptive_score := teacher_recon_error`: 1450 파일 덮어씀
  (나머지는 직전 run 에서 이미 recon-only → idempotent skip). 전 npz `.tmp` 대신
  `.rebuild.npz` 로 원자적 교체 (np.savez_compressed 의 `.npz` 자동 접미사 회피).
- **R2/R3** pre-warmup epoch_metrics row: 2250 row (45×50) 를 `compute_full_metric_set`
  (SWaT 는 `compute_metrics_with_exclusion` 추가) 로 재계산. raw 배열은 불변.
- **R4** best_epoch 재선정 → **5 cell flip** (양방향, 예측과 정확히 일치):
  271canon/PSM 105→100, 271_lr/WaDi_A2 245→240, 271_lr/SWaT_full 370→180(post→pre,
  Δ=2e-5 동률), 285/PSM 95→490(pre→post), 286/WaDi_A1 225→165. flip + pre-warmup-best
  16 cell 의 `experiment_metadata.metrics` scalar 재계산.
- **백업**: `.trash/0531/backfill_backups/` (원본 npz+json), `.trash/0531/backfill_viz_backups/`
  (원본 PNG 107M).

### 독립 감사 (`scripts/audit_prewarmup_backfill.py` + 병렬 recompute)
- C1 pre-npz recon-only **45/45**, C2 post-npz 미변경 **45/45**, C4 post-warmup row
  byte-identical **45/45**, C5 best_epoch=argmax **45/45**.
- 기록된 pre-warmup row 와 독립 재계산 **정확 일치(max diff = 0.000e+00, 75 row 표본)**.
- 유일 flag 1건(285/WaDi_A2 ep5)은 backfill 오류가 **아니라** 원본 per-epoch eval 의
  `teacher_pak_auc_f1` 과 재계산 경로 간 **1.58e-6 PA%K 적분 jitter** (다른 epoch 은 diff 0).

### 재시각화
- epoch-metric curve: **45/45 cell** 재생성 (`scripts/reviz_prewarmup_backfill.py`).
- best_model: 29 CORRECT(post-warmup best, 영향 없음), 16 STALE → **Option A**
  (`scripts/reviz_flip_cells_optionA.py`): 정정 best-epoch npz 로 score 기반 5종
  (ROC/PRC/CM/score-dist/threshold) 재생성, 신호복원 7종은 STALE 마커.
  5 flip cell 은 weight 부재로 신호복원 재생성 불가(사용자 deprioritise),
  11 pre-warmup-best 는 weight 존재 → full pipeline 로 재생성 가능(deferred).
- 상세: `docs/POST_MORTEMS/2026-06-01_prewarmup_backfill_stale_viz.md`.

## Prevention

- 평가 경로에도 학습 스케줄(warmup) 맥락을 명시적으로 주입(`set_eval_context`)하는 패턴을
  표준화. "학습 gate 가 eval 에도 적용될 것" 이라는 암묵 가정 금지.
- score 컴포넌트 분기(disc, FM)는 항상 **AND 조건**으로 차단 — 가중치 0 만으로 끄지 말 것.
- 게이트 술어는 `scoring.py` 단일 소스(`is_prewarmup_epoch`)로만 — 인라인 `epoch<=warmup`
  복제 금지(2026-05-28 FM-omission 인라인 중복 교훈과 동일 class).

## Related

- `docs/POST_MORTEMS/2026-05-29_fm_score_omission.md` — 동일 class(score 컴포넌트 누락/혼입,
  required-kw 미적용으로 인한 silent divergence).
- `mae_anomaly/scoring.py` — 단일 소스 모듈.
- CLAUDE.md "API change checklist (2026-05-29)".
