# Post-mortem — 2026-06-01 — SWaT excl22 `recon_snr` always None (parallel copy-list omission)

## Summary
SWaT 의 dual-eval (`A1A2_full` / `A1A2_excl22`) 중 **excl22** 의 `epoch_metrics.json` 에서
`recon_snr` **와 `fm_loss`** 가 **100 개 eval 전부 None**. 같은 row 의 `disc_snr` 는 full-SWaT
값이 그대로 들어 있어, "excl22 의 recon/fm 분석"은 무엇이든 조용히 None 을 읽었다. 6 개 모델
+ 285/286/287 의 excl22 전부 해당. (full↔excl22 키셋 전수 비교로 dataset-wide 진단 누락은
`recon_snr`·`fm_loss` 둘 뿐임을 확정 — 아래 분류 참조.)

## Impact
| 산출물 | 증상 |
| ---- | ---- |
| `SWaT/A1A2_excl22/epoch_metrics.json` 의 `recon_snr` (100 eval) | 전부 None |
| 같은 파일의 `disc_snr` | full-SWaT 값 그대로 (정상 — dataset-wide) |
| excl22 recon-SNR 추이/상관 분석 | 입력이 None → 무의미하거나 0 처리 |

탐지 지표(pak/prc/f1/aff/r_f1 등)·best-epoch 선정·`disc_snr` 은 **영향 없음**.

## Root cause
excl22 worker (`scripts/run_base_experiments.py`, `swat_eval_mode='excl22'`) 의 per-epoch
metric 조립 (`excl22_epoch_metrics_list` 구성):

1. detection 지표는 `compute_metrics_with_exclusion()` 로 **excl22 마스크 기준 재계산** —
   이 함수는 point-score 기반 탐지 지표만 내며 **SNR 진단은 내지 않는다**.
2. SNR 등 **test-region 과 무관한 dataset-wide 진단**(`disc_snr`, `d_*`, `grl_*`)은 full-SWaT
   `epoch_metrics.json`(spawn 시 excl22 dir 로 `shutil.copy2` 된 것)에서 **per-eval 복사**
   (`for key in [...]: if key in full_epoch_data[ep_num]: entry[key]=...`).

`recon_snr` 은 **2026-05-29** 에 신규 추가된 dataset-wide 진단(teacher recon 분리도)인데,
full-SWaT 저장 경로(L710)에는 추가됐지만 **이 excl22 복사 목록(L1928)에는 추가되지 않았다.**
→ excl22 entry 에 `recon_snr` 키가 아예 없어 None.

**핵심 결함 패턴:** *"동일 정보를 두 곳(원본 산출 + 병렬 복사 목록)에서 유지해야 하는데,
새 필드를 한 곳에만 추가"*. `disc_snr` 의 쌍둥이 필드 `recon_snr` 를 짝으로 추가하지 않은 것.

## Fix
- **[run_base_experiments.py L1928]** 복사 목록 맨 앞에 `recon_snr`, `fm_loss` 추가:
  `['disc_snr', 'recon_snr', 'fm_loss', 'd_loss', ...]`. `if key in full_epoch_data[ep_num]` 가드가
  legacy(필드 없는 옛 run)를 자동 보호. excl22 epoch_metrics.json 에만 영향.
  (`fm_loss` = `train_fm_loss[idx]` 학습 FM loss — dataset-wide, 9 run 중 8 run post-warmup non-zero.)

## full↔excl22 키셋 전수 비교 (누락 한정 확인)
full(329키) − excl22(166키) = full-only 163키. 분류:
| 분류 | 개수 | 처리 |
| ---- | ---- | ---- |
| `excl22_*` prefix (full 파일에 embed 된 excl22 ref; excl22 파일엔 unprefix 동일 데이터) | 145 | 누락 아님 |
| excl22-**종속** counts `n_anomaly`/`n_pure_normal`/`n_disturbing_normal` (region-22 제외로 값이 달라야 함 → 복사하면 틀림) | 3 | 미복사 정당 |
| `disturbing_*` (disturbing-normal subset detection — copy-class 아닌 test-종속 detection) | 4 | 미복사 정당 |
| 내부 `_*` viz/timing 배열 (full-worker 전용; excl22 는 `_filter_excl22_viz_data`) | 8 | 미복사 정당 |
| `teacher_pa_20_f1` (단일 teacher PA bucket; excl22 는 `teacher_pak_auc_*` 보유) | 1 | 미복사 정당 |
| **dataset-wide 진단 (copy-class) — 진짜 누락** | **2** | **`recon_snr`, `fm_loss` 추가 (FIX)** |
→ copy-class 누락은 `recon_snr`·`fm_loss` 둘뿐. 그 외는 설계상 정당한 차이.

## Verification
- `py_compile` OK.
- 복사 로직을 실제 286 full-SWaT `epoch_metrics.json` 으로 재현: `recon_snr` **100/100 복사**
  (수정 전 0), `disc_snr` **100/100 무회귀**, full↔excl22 값 mismatch 0.
- 원본 백업: `./.trash/0601/run_base_experiments.py.bak_<ts>`.

## Prevention
- **dataset-wide(=exclusion-independent) 진단을 새로 추가할 때는 두 곳을 한 커밋에서 같이
  수정**: (a) full 저장부 (L703-710 부근), (b) excl22 복사 목록 (L1928). `recon_snr`↔`disc_snr`
  처럼 쌍을 이루는 필드는 항상 같은 목록에 함께 둔다.
- 복사 목록을 하드코딩 대신 "full 에 있고 excl22 detection 결과에 없는 진단 키 전부"를
  자동 산출하는 방식도 고려 가능(추후 리팩터). 현재는 명시 목록 유지 + 본 post-mortem 으로
  체크리스트화.

## Backfill (이미 완료된 excl22 데이터) — 2026-06-01 완료
사용자 승인("정합성 높을 때만, 오류 0") 하에 9개 excl22 `epoch_metrics.json` 백필 완료.
**충전 조건(전부 충족시에만 채움):** ① full 에 동일 epoch 존재 ② full 에 `recon_snr`·`fm_loss`
유효 ③ **excl22 의 기존 `disc_snr` 가 full 과 byte 일치**(정렬 증명) — 이미 복사돼 있던
dataset-wide 진단으로 epoch 정렬·동일-run 임을 교차검증. `disc_snr` 바로 뒤에 두 키 삽입
(fresh-gen 순서 일치).

**결과:** 9 run 중 8 run 100/100 충전, `271_lr` 은 98/100(full-SWaT 가 ep285·ep290 eval 을
영구 누락 → orphan 2개는 source 없어 **None 유지, 미변경**). `285_no_fm` 은 `fm_loss`=0.0
(FM 비활성, full 도 0.0 — 정상). 안전 게이트: 백업→in-memory 빌드+검증(실패시 쓰기 0)→
쓰기→디스크 재대조. 전 파일 bad=0 (기존 키 전부 무변형, `recon_snr`·`fm_loss` 2키만 추가).
백업: `./.trash/0601/backfill/<run>__excl22_epoch_metrics.json.bak`.
