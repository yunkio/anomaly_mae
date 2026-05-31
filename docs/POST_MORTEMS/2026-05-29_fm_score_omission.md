# Post-mortem — 2026-05-29 — FM-score omission across the evaluation and visualization pipeline

## Summary

`use_feature_matching=True` 설정으로 학습된 모든 실험에서, `epoch_metrics.json` 의
per-epoch metric (`pak_auc_f1`, `prc_auc`, `f1_t`, `affiliation_f1` 등) 과
`experiment_metadata.json` 의 최종 metric, 그리고 시각화 (`visualization/best_model/*.png`)
가 **FM-누락된 anomaly score** 로 계산되어 왔다. NPZ 에 저장된
`adaptive_score` 만 정확하게 FM 을 포함하고 있었으며, 두 score 의 불일치는
`best_epoch` 선정 오류 (잘못된 epoch 의 checkpoint 가 best 로 선정) 와 그
checkpoint 기반 viz 의 시각적 오해를 야기했다.

`use_feature_matching=False` 실험은 영향 없음 (FM-누락 분기와 정상 분기가
수학적으로 동치).

## Impact

| 산출물 | 정확성 | 비고 |
| ---- | ---- | ---- |
| `epoch_NNN_scores.npz["adaptive_score"]` | ✅ FM 포함 | 단일 정확 소스 |
| `epoch_metrics.json` 의 모든 entry | ❌ FM 누락 | per-epoch eval path |
| `experiment_metadata.json["metrics"]` | ❌ FM 누락 | bg-worker final eval path |
| `best_checkpoint.pt` 의 epoch 선택 | ❌ 잘못된 best_epoch | `epoch_metrics.json` 인용 |
| `visualization/best_model/best_model_score_contribution.png` | ✅ NPZ 직접 사용 | 유일하게 정확한 viz |
| `visualization/best_model/*` (PRC/ROC/CM/score timeline 등) | ❌ FM 누락 score | derive_pred_data 인용 |
| `summary.json`, `anomaly_type_metrics.json`, `best_model_detailed.csv` | ❌ | 위 source 들 인용 |

영향 받은 실험: `271_20260527_173025_...`, `274_20260528_175756_...`,
`274_20260513_055753_...` 및 5/26 이후 학습된 모든 ablation 실험 (queue
286–305 미실행, 진행 중 학습은 중단).

영향 받지 **않은** 실험: `271_20260508_094241_...` (5/26 commit ff4c4c3
이전 NPZ save 코드 자체에 FM 인라인이 없었기 때문에 평가 양쪽 path 가
같이 FM 을 누락 → 결과는 buggy 였지만 자체 정합성 유지).

## Root cause

`evaluator.set_precomputed_patch_scores` (commit ff4c4c3, 2026-05-26)
시그니처에 `fm_patches: Optional[np.ndarray] = None` 키워드 인자가
추가되었지만, 호출자 3 곳 (`scripts/run_base_experiments.py:633`,
`scripts/run_base_experiments.py:1473`, `scripts/ablation/run_ablation.py:1908`)
이 동시 업데이트되지 않아 모두 `fm_patches` 를 빠뜨린 채 호출했다.

결과적으로 evaluator 내부에서 `self.fm_patches` 가 `None` 으로 유지되고,
`_apply_scoring_formula` 가 `fm is None` 분기를 타며 `score = recon + scaled_disc`
공식 (FM 누락) 로 점수를 계산했다.

같은 시점 `scripts/run_base_experiments.py` 의 NPZ save 코드 (line 2257-2316)
는 `eval_data['fm_patches']` 를 직접 인라인으로 처리하면서 FM 을 포함한
정확한 공식 `recon + (scaled_disc + scaled_fm)/2` 로 `adaptive_score` 를 계산.

→ NPZ 의 score 와 per-epoch eval 의 score 가 서로 다른 입력을 보고 있었음.

## Why it was not caught

1. **Optional default=None 의 silent failure**: caller 가 인자를 빠뜨려도
   런타임 에러 없이 None 으로 fallback → 코드 review 와 정적 분석으로 잡기
   매우 어려움.
2. **공식의 인라인 중복**: 동일 adaptive 공식이 9 곳에 흩어져 있었고
   epsilon (1e-4 vs 1e-8) 까지 일부 달랐음 → 한쪽 path 만 업데이트해도
   다른 path 가 그대로 남는 패턴이 구조적으로 발생.
3. **자기-일관 비결정성**: 모든 metric 이 같은 buggy score 로 계산되어
   epoch_metrics.json 내부에서는 정합성 유지 (ep75 saved 0.7514 = NPZ recompute
   0.7511). 정확한 NPZ score 와 비교했을 때만 격차 발생.
4. **single source 주석의 오도**: 5/27 commit `5a76c54` 가 metric 계산을
   "single source pipeline (compute_full_metric_set)" 로 통합했지만, score
   생성은 여전히 두 path 인라인이라 검증 누락.
5. **학습 후 자동 sanity check 부재**: 학습 종료 시점에 NPZ adaptive_score
   와 epoch_metrics 의 metric 이 일치하는지 자동 비교가 어디에도 없었음.

## Detection

사용자가 `new_274 WaDi/A1` 의 metadata best_epoch=75 (`pak_auc_f1=0.7514`) 가
직관과 다르다는 의문을 제기 → 동일 epoch 의 NPZ score 로 외부 재계산한 결과
(`0.7511`) 가 metadata 와 거의 같으나, **ep325 에서는 NPZ recompute = 0.7650
vs metadata saved = 0.7494** 로 0.014 차이 → 동일 score 입력에 대해 두 path
가 다른 답을 내고 있다는 사실 발견. caller 별로 `set_precomputed_patch_scores`
호출을 grep 한 결과 3 곳 모두 `fm_patches` 가 빠진 것을 확인.

## Fix (Phase 1-6, 2026-05-29)

CHANGELOG.md 의 `2026-05-29` 항목 참조. 핵심:

1. `mae_anomaly/scoring.py` 모듈 신설로 score 공식의 단일 정답 source 확립.
2. `mae_anomaly/types.py:PatchScoresBundle` dataclass 로 patch-level data
   의 typed container 도입. `fm` 은 명시적 `Optional[ndarray]` 필드.
3. `Evaluator.set_precomputed_patch_scores` 가 bundle 만 받도록 변경 (Optional
   kwarg 제거). 이전 silent failure 패턴 구조적 차단.
4. `compute_full_metric_set`, `Evaluator.evaluate` 등의 `lite` 등 boolean
   인자를 keyword-only required 로 변경.

## Prevention

- **API 변경 → 호출자 동시 업데이트 체크리스트** (`CLAUDE.md` 추가).
- 새 Optional kwarg 를 추가할 때는 모든 호출 site 를 grep 하고, default 가
  silent 결과 분기를 만드는지 명시적으로 검토.
- 같은 의미의 공식은 인라인 중복 금지. 새 score component 추가 시 반드시
  `mae_anomaly/scoring.py` 만 수정 (set_guideline.md 추가).
- `_apply_scoring_formula` 는 `mae_anomaly.scoring` 의 thin wrapper 로 유지.

## Lessons

- "single source of truth" 라는 주석은 그 source 가 실제로 모든 path 의
  값을 결정한다는 검증 없이는 위로일 뿐이다.
- frozen dataclass + Optional 필드의 명시적 표기는 Optional kwarg + default
  None 보다 silent failure 확률을 크게 낮춘다.
- 학습 종료 시점에 "두 path 가 같은 값을 산출하는지" 자동 검증하는 작은
  invariant check 가 큰 가치가 있다 (향후 추가 가능).
