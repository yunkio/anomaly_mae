# Post-mortem — 2026-06-01 — GRL `balanced_acc=0.5` is deceptive (degeneracy gap added)

## Summary
GRL 도메인-적대 분류기의 `grl_balanced_acc` 는 **퇴화(degeneracy)를 숨긴다**.
`balanced_acc = (anomaly_acc + normal_acc) / 2` 이므로, 분류기가 **모든 패치를
anomaly 로** 찍는 완전 퇴화 상태(`anomaly_acc=1.00`, `normal_acc=0.00`)에서도
`balanced_acc = 0.50` 이 나온다 — "random 수준, GRL 정상 작동"으로 **오독**된다.
실제 exp287_unmask SWaT 에서 ep~340 이후 정확히 이 패턴이 발생(아래 그림).

## 진단 지표
`grl_acc_gap = |normal_acc − anomaly_acc|` (저장된 epoch-mean 정확도에서 파생):
| 상태 | normal | anomaly | balanced | **gap** | 해석 |
| ---- | ------ | ------- | -------- | ------- | ---- |
| 완전 퇴화(all-anomaly) | 0.00 | 1.00 | **0.50** | **1.00** | degenerate |
| 진짜 동전던지기 | 0.50 | 0.50 | 0.50 | 0.00 | uninformative |
| 진짜 좋은 분류기 | 0.90 | 0.90 | 0.90 | 0.00 | genuine |

**중요:** gap 단독도 "좋은 분류기(둘 다 0.9, gap=0)"와 "동전던지기(둘 다 0.5,
gap=0)"를 구분 못 한다. 정직한 퇴화 판정은 **(balanced_acc, gap) 쌍**:
`balanced≈0.5 AND gap≈1.0 ⟺ degenerate`. → balanced_acc 를 **유지**하고 gap 을
**추가**하며 `gap≥0.8` 구간을 음영 표시.

## 변경 (live pipeline 만)
- **계산/수집 (단일 파생점 = trainer history):**
  - `mae_anomaly/trainer.py` — history 에 `train_grl_acc_gap` 추가, append 시
    `abs(grl_normal_acc − grl_anomaly_acc)` (이미 저장되는 epoch-mean 정확도에서
    파생 → 모든 GRL 변형에 공통 적용, **WGAN/WDGRL 계산 경로는 미변경**).
  - `scripts/run_base_experiments.py` — `cb_metrics['grl_acc_gap']` 파생,
    SWaT excl22 복사 목록에 `grl_acc_gap` 추가.
- **시각화 (2곳 모두 gap 곡선 + `gap≥0.8` 퇴화 음영):**
  - `scripts/run_base_experiments.py` `plot_epoch_metrics` → `epoch_grl.png`
    accuracy 패널.
  - `mae_anomaly/visualization/best_model_visualizer.py`
    `plot_grl_contribution_trend` → `GRL_contribution_trend.png` (C) 패널.
- **데이터 백필:** 9 GRL-active run × 전 cell 의 기존 `epoch_metrics.json` 에
  `grl_acc_gap` 파생 추가 (45 파일, ~4490 epoch). normal/anomaly 가 모두 존재할
  때만 삽입(271_lr orphan ep285/290 = grl None → 미삽입). 안전게이트:
  백업→in-mem 검증(실패시 쓰기0)→쓰기→백업 재대조, 전 파일 bad=0.
- **재시각화:** `scripts/reviz_grl_gap.py` 로 9 run × 전 cell 의 `epoch_grl.png`
  + `GRL_contribution_trend.png` 재생성(둘 다 순수 data→PNG, GPU 불필요).
  45/45 + 45/45, 오류 0. 기존 PNG 백업 보존.

## 미변경 (의도적)
- **WGAN/WDGRL 경로** (`trainer.py:694`): 거기 `grl_balanced_acc=|norm−anom score|`
  이고 normal/anomaly 가 [0,1] 정확도가 아닌 critic **score** — 의미가 달라
  사용자 지시로 **그대로 둠**(현 9 run 모두 이 경로 미사용).
- **과거 동결 분석 스크립트** — `scripts/early_stopping_analysis_v2..v6.py`,
  `scripts/build_es_notion_*.py` 는 `train_grl_balanced_acc` 를 **early-stopping
  신호로 직접 사용**(`deriv_grl_balanced_acc_dW*`, `abs=0` plateau 등). 이는 바로
  이 기만 지표를 ES 기준으로 쓴 것 — 즉 그 ES 픽들은 **퇴화 구간의 balanced≈0.5
  평탄부를 "수렴"으로 오인했을 수 있음**. 일회성 동결 산출물이라 **소급 수정하지
  않고 본 문서에 기록만** 함(사용자 지시). 향후 ES 설계 시 `grl_acc_gap` 또는
  (balanced, gap) 쌍을 신호로 쓸 것.

## 관측 사례 (exp287_unmask / SWaT)
`GRL_contribution_trend.png` (C): ep~340+ 에서 `balanced_acc` 가 정확히 0.50 에
안착하는 동안 `normal_acc→0`, `anomaly_acc→1`, `gap→1.0` → 음영으로 degenerate
구간 표시. balanced_acc 단독 뷰에서는 "random regularizer 정상"으로 보였을 신호.

## Prevention
- 평균-기반 집계 지표(balanced/평균 정확도)는 **분해 성분의 격차(gap)와 함께**
  보고. degeneracy 는 평균이 숨긴다.
- 새 진단을 추가할 때 dataset-wide 파생값은 **단일 파생점**(여기선 trainer
  history)에서 계산하고 excl22 복사 목록 등 **병렬 유지 지점**을 같은 커밋에서
  갱신. (cf. `2026-06-01_excl22_recon_snr_copy_omission.md`)
