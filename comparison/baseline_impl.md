# Baseline Implementation — Faithfulness Pitfalls

원칙: 진실의 출처 = **원본 논문 + 저자가 "이 task"를 실제로 돌린 config**. 주석/문서/argparse-default를 액면 그대로 믿지 말 것. (새 주의사항은 아래에 짧게 append)

## P1. 잘못된 config 계층 (가장 빈번)
- 원인: repo의 **범용 argparse/class default**(다중-task lib에선 보통 forecasting용)를 가져옴 — 저자가 실제로 그 task에 쓴 **task/per-dataset 스크립트(.sh/.conf/Table)**가 이를 override하는데 그걸 안 봄.
- 결과: wrong-task/wrong-size HP인데 "default라 충실"해 보임 (예: TimesNet 512=forecasting vs anomaly ≤128).
- 규칙: **task 스크립트/논문 Table 값** 사용. 각 HP가 어느 계층 값인지 명시. canonical이 per-dataset면 그 사실 자체를 기록(단일값으로 silent collapse 금지).

## P2. stale 주석/문서/이전 판정
- 원인: docstring·문서·이전 audit을 실행 코드 대신 신뢰.
- 결과: 실제 동작과 반대 결론.
- 규칙: 모든 주장은 **실행 코드 경로 + 논문**으로 검증. 주석류는 미검증 claim 취급.

## P3. config 필드 ≠ runtime
- 원인: 런타임 추적 없이 config 필드만 읽음 (self-norm 모델은 무시; dead 필드 존재).
- 결과: 허위 주장(예: "전부 minmax"; dead yaml weight_decay를 실제 적용).
- 규칙: entry→dispatch→loader→model 추적, live/dead 구분, 가능하면 실행으로 확인.

## P4. 같은 작업의 코드 경로 이중화 (evaluate 등)
- 원인: MAE와 baseline이 **동일 작업**(per-epoch metric 계산)을 각자 **별도 함수**로 구현. 둘 다 "single source of truth"라 주석에 적어놨으나 실제로는 복제본 → 한쪽만 수정되면 silent divergence(예: point F1 threshold `>` vs `>=`).
- 결과: 같은 metric이 파이프라인마다 다른 값. 비교 무효 + 디버깅 지옥.
- 규칙: **공통 작업은 단일 함수로 통일**(기준 = MAE). baseline은 그 함수를 호출하는 **thin wrapper**만 두고 presentation 키(alias/None)만 추가. 새 metric은 그 한 함수에만 추가.
- 적용(2026-06-06): `comparison/baseline_common.py::compute_all_metrics`/`_zero_metrics` → MAE `compute_full_metric_set`/`_zero_metric_set` wrapper로 통일. 검증: 162개 키 byte-identical, 의도된 point p/r/f1만 변경.

## 체크리스트 (모델마다)
- [ ] HP = task/논문 config (범용 default 아님), 계층 명시
- [ ] per-dataset/sweep는 기록 (silent 단일값 금지)
- [ ] 주장은 실행코드+논문 근거 (주석/문서 아님)
- [ ] runtime 추적 + size/compute 상식 점검
