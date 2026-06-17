# IDV 3/9/15 (excluded-hard) 심층 검증

질문: "폐루프 제어가 완전 보상해 사실상 구분 불가"라는 설계 §2.2의 주장이 (L1) point 수준, (L2) run-집계 수준, (L3) 모델 score 수준에서 모두 성립하는가?  
대상: IDV [3, 9, 15] + 참조 anchor IDV [1, 16, 19] (1=쉬운 step, 16/19=usable 중 가장 subtle).

## L1. Point 수준 분리 한계 (per-feature ROC-AUC, post-onset faulty pts vs FF pts)

| fault | family | **best-feature AUC** | top-3 features (AUC) | mean-shift 최대 효과크기 d | std-ratio 최대 |
|---|---|---|---|---|---|
| IDV3 | EXCL-HARD | **0.568** | xmeas_21(0.568), xmv_10(0.539), xmeas_20(0.514) | 0.24 (xmeas_21) | 1.04 (xmeas_20) |
| IDV9 | EXCL-HARD | **0.514** | xmv_5(0.514), xmeas_20(0.514), xmeas_11(0.512) | 0.06 (xmv_5) | 1.04 (xmeas_21) |
| IDV15 | EXCL-HARD | **0.515** | xmv_5(0.515), xmeas_20(0.514), xmeas_13(0.513) | 0.06 (xmv_5) | 1.09 (xmeas_22) |
| IDV1 | step | **0.996** | xmv_3(0.996), xmeas_1(0.996), xmeas_4(0.987) | 16.30 (xmv_3) | 4.15 (xmeas_16) |
| IDV16 | unknown | **0.514** | xmv_5(0.514), xmeas_13(0.514), xmeas_11(0.514) | 0.11 (xmv_9) | 2.23 (xmeas_19) |
| IDV19 | unknown | **0.511** | xmeas_33(0.511), xmeas_13(0.511), xmeas_7(0.511) | 0.05 (xmv_5) | 2.50 (xmeas_5) |

## L2. Run-집계 수준 분리 한계 (run당 800-sample 집계 후 20 vs 40 runs AUC)

window 모델(W=500)이 쓸 수 있는 "시간 맥락 집계"의 상한 근사. mean-집계와 std-집계 각각의 best-feature run-level AUC.

| fault | best AUC (run-mean) | feature | best AUC (run-std) | feature | 판정 |
|---|---|---|---|---|---|
| IDV3 | 1.000 | xmeas_21 | 0.709 | xmeas_36 | 분리 가능 |
| IDV9 | 0.721 | xmeas_20 | 0.740 | xmeas_21 | 비식별 |
| IDV15 | 0.721 | xmeas_20 | 0.967 | xmeas_22 | 분리 가능 |
| IDV1 | 1.000 | xmeas_1 | 1.000 | xmeas_1 | 분리 가능 |
| IDV16 | 0.722 | xmeas_20 | 1.000 | xmeas_18 | 분리 가능 |
| IDV19 | 0.684 | xmeas_8 | 1.000 | xmeas_5 | 분리 가능 |

## L3. 모델 score 수준 (per-fault, 각 fault 20 runs + FF 40 runs; positive rate = 800x20/(800x20+38400) ≈ 29.4% point 기준)

| fault | model | roc_auc | prc_auc | pak_auc_f1 | 해석 |
|---|---|---|---|---|---|
| IDV3 | random@ffonly | 0.502 | 0.278 | 0.483 | random 동등 |
| IDV3 | sensor_range@ffonly | 0.501 | 0.278 | 0.005 | random 동등 |
| IDV3 | pca_error@ffonly | 0.510 | 0.284 | 0.609 | random 동등 |
| IDV3 | l2_norm@ffonly | 0.516 | 0.294 | 0.616 | random 동등 |
| IDV3 | nn_distance@ffonly | 0.518 | 0.297 | 0.604 | random 동등 |
| IDV9 | random@ffonly | 0.502 | 0.279 | 0.483 | random 동등 |
| IDV9 | sensor_range@ffonly | 0.500 | 0.278 | 0.004 | random 동등 |
| IDV9 | pca_error@ffonly | 0.513 | 0.286 | 0.612 | random 동등 |
| IDV9 | l2_norm@ffonly | 0.519 | 0.296 | 0.619 | random 동등 |
| IDV9 | nn_distance@ffonly | 0.519 | 0.298 | 0.606 | random 동등 |
| IDV15 | random@ffonly | 0.502 | 0.278 | 0.483 | random 동등 |
| IDV15 | sensor_range@ffonly | 0.501 | 0.279 | 0.010 | random 동등 |
| IDV15 | pca_error@ffonly | 0.512 | 0.287 | 0.609 | random 동등 |
| IDV15 | l2_norm@ffonly | 0.523 | 0.303 | 0.620 | random 동등 |
| IDV15 | nn_distance@ffonly | 0.532 | 0.311 | 0.615 | random 동등 |
| IDV1 | random@ffonly | 0.500 | 0.278 | 0.481 | random 동등 |
| IDV1 | sensor_range@ffonly | 0.997 | 0.995 | 1.000 | 식별 |
| IDV1 | pca_error@ffonly | 1.000 | 1.000 | 1.000 | 식별 |
| IDV1 | l2_norm@ffonly | 0.997 | 0.996 | 1.000 | 식별 |
| IDV1 | nn_distance@ffonly | 0.999 | 0.999 | 1.000 | 식별 |
| IDV16 | random@ffonly | 0.504 | 0.280 | 0.486 | random 동등 |
| IDV16 | sensor_range@ffonly | 0.544 | 0.341 | 0.246 | random 동등 |
| IDV16 | pca_error@ffonly | 0.966 | 0.954 | 0.986 | 식별 |
| IDV16 | l2_norm@ffonly | 0.577 | 0.385 | 0.663 | 약한 신호 |
| IDV16 | nn_distance@ffonly | 0.776 | 0.637 | 0.791 | 식별 |
| IDV19 | random@ffonly | 0.496 | 0.276 | 0.478 | random 동등 |
| IDV19 | sensor_range@ffonly | 0.534 | 0.326 | 0.188 | random 동등 |
| IDV19 | pca_error@ffonly | 0.992 | 0.985 | 0.993 | 식별 |
| IDV19 | l2_norm@ffonly | 0.555 | 0.339 | 0.641 | 약한 신호 |
| IDV19 | nn_distance@ffonly | 0.775 | 0.603 | 0.786 | 식별 |

## 종합 판정

- IDV3: **point 비식별 / run-집계 분리 가능** (L1 0.568 vs L2 1.000) — point-wise 방법은 못 잡지만 시간-맥락 모델(W=500)은 잡을 수 있는 후보. 설계의 "어떤 방법으로도 비식별" 문구를 "point-wise 비식별"로 한정 필요.
- IDV9: **완전 비식별** — point(best-feature AUC 0.514)과 run-집계(0.740) 모두 분리 불가. "어떤 방법으로도(window 집계 포함) 비식별" 주장 지지.
- IDV15: **point 비식별 / run-집계 분리 가능** (L1 0.515 vs L2 0.967) — point-wise 방법은 못 잡지만 시간-맥락 모델(W=500)은 잡을 수 있는 후보. 설계의 "어떤 방법으로도 비식별" 문구를 "point-wise 비식별"로 한정 필요.

Anchor 대조: IDV1 L1=0.996/L2=1.000, IDV16 L1=0.514/L2=1.000, IDV19 L1=0.511/L2=1.000 (subtle-usable도 run-집계에서는 분리됨이 정상).