# TEP Type-Generalization #12 — Simple Baselines 분석/검증 보고서

결과: `/home/ykio/notebooks/claude/temp/0610/TEP/results/12_20260610_211815_tep_typegen_simple`  
설계: `temp/tep_design/80_experiment_design_final.md` (사전 등록 2026-06-10)  
조건: pca_error·sensor_range × {4 contaminated folds + ffonly reference}, minmax(no-clip, train-fit), per-run boundary-safe scoring, 평가 = 기존 baseline 스택 (`compute_all_metrics`) 그대로.

## 1. 검증 게이트 (sanity checks)

| 검사 | 결과 | 상세 |
|---|---|---|
| test stream 크기 | PASS | 422,400 = 440 runs x 960 |
| test anomaly 수 | PASS | 320,000 = 400 faulty runs x 800 |
| train_f_step anomaly 비율 | PASS | 0.1667 (= 16.67% 설계값) |
| train_f_rand anomaly 비율 | PASS | 0.1667 (= 16.67% 설계값) |
| train_f_ds anomaly 비율 | PASS | 0.1667 (= 16.67% 설계값) |
| train_f_unk anomaly 비율 | PASS | 0.1667 (= 16.67% 설계값) |
| 상수 feature (FF train) | PASS | 없음 (denom-guard로 유지) |
| 전체 run 길이 960 균일 | PASS | 440 runs |
| run_boundaries 수 | PASS | 439 internal seams |
| f_step/random score 길이/유한성 | PASS | 422,400 |
| f_step/sensor_range score 길이/유한성 | PASS | 422,400 |
| f_step/sensor_range score는 {0,1} 이진 | PASS | unique=[0. 1.] |
| f_step/pca_error score 길이/유한성 | PASS | 422,400 |
| f_step/pca_error run별 선두 5pt = 0 (per-run smoothing 증거) | PASS | 440 runs x 5 pts |
| f_step/l2_norm score 길이/유한성 | PASS | 422,400 |
| f_step/nn_distance score 길이/유한성 | PASS | 422,400 |
| f_rand/random score 길이/유한성 | PASS | 422,400 |
| f_rand/sensor_range score 길이/유한성 | PASS | 422,400 |
| f_rand/sensor_range score는 {0,1} 이진 | PASS | unique=[0. 1.] |
| f_rand/pca_error score 길이/유한성 | PASS | 422,400 |
| f_rand/pca_error run별 선두 5pt = 0 (per-run smoothing 증거) | PASS | 440 runs x 5 pts |
| f_rand/l2_norm score 길이/유한성 | PASS | 422,400 |
| f_rand/nn_distance score 길이/유한성 | PASS | 422,400 |
| f_ds/random score 길이/유한성 | PASS | 422,400 |
| f_ds/sensor_range score 길이/유한성 | PASS | 422,400 |
| f_ds/sensor_range score는 {0,1} 이진 | PASS | unique=[0. 1.] |
| f_ds/pca_error score 길이/유한성 | PASS | 422,400 |
| f_ds/pca_error run별 선두 5pt = 0 (per-run smoothing 증거) | PASS | 440 runs x 5 pts |
| f_ds/l2_norm score 길이/유한성 | PASS | 422,400 |
| f_ds/nn_distance score 길이/유한성 | PASS | 422,400 |
| f_unk/random score 길이/유한성 | PASS | 422,400 |
| f_unk/sensor_range score 길이/유한성 | PASS | 422,400 |
| f_unk/sensor_range score는 {0,1} 이진 | PASS | unique=[0. 1.] |
| f_unk/pca_error score 길이/유한성 | PASS | 422,400 |
| f_unk/pca_error run별 선두 5pt = 0 (per-run smoothing 증거) | PASS | 440 runs x 5 pts |
| f_unk/l2_norm score 길이/유한성 | PASS | 422,400 |
| f_unk/nn_distance score 길이/유한성 | PASS | 422,400 |
| ffonly/random score 길이/유한성 | PASS | 422,400 |
| ffonly/sensor_range score 길이/유한성 | PASS | 422,400 |
| ffonly/sensor_range score는 {0,1} 이진 | PASS | unique=[0. 1.] |
| ffonly/pca_error score 길이/유한성 | PASS | 422,400 |
| ffonly/pca_error run별 선두 5pt = 0 (per-run smoothing 증거) | PASS | 440 runs x 5 pts |
| ffonly/l2_norm score 길이/유한성 | PASS | 422,400 |
| ffonly/nn_distance score 길이/유한성 | PASS | 422,400 |
| f_step partition 크기 (seen/unseen) | PASS | seen 153,600=6f, unseen 249,600=11f, 교집합 없음 + 3/9/15 헤드라인 제외 |
| f_rand partition 크기 (seen/unseen) | PASS | seen 115,200=4f, unseen 288,000=13f, 교집합 없음 + 3/9/15 헤드라인 제외 |
| f_ds partition 크기 (seen/unseen) | PASS | seen 76,800=2f, unseen 326,400=15f, 교집합 없음 + 3/9/15 헤드라인 제외 |
| f_unk partition 크기 (seen/unseen) | PASS | seen 134,400=5f, unseen 268,800=12f, 교집합 없음 + 3/9/15 헤드라인 제외 |

## 2. 주 결과표 — fold × model × partition

점수: 사전 등록된 threshold-robust 4지표. G = seen − unseen (label-blind 모델이므로 G = 순수 난이도 + train 오염 방향 효과; 설계의 G_ctrl 해석 — MAE 조건 A의 Ĝ 보정 기준선이 됨).

### pak_auc_f1

| fold | model | seen | unseen | **G=seen−unseen** | exclhard | full |
|---|---|---|---|---|---|---|
| f_step | random | 0.7127 | 0.7458 | **-0.0332** | 0.6506 | 0.7645 |
| f_step | sensor_range | 0.1842 | 0.6161 | **-0.4319** | 0.0007 | 0.4332 |
| f_step | pca_error | 0.8736 | 0.9251 | **-0.0515** | 0.7882 | 0.9230 |
| f_step | l2_norm | 0.8698 | 0.8920 | **-0.0221** | 0.7991 | 0.9153 |
| f_step | nn_distance | 0.8770 | 0.9279 | **-0.0509** | 0.7914 | 0.9246 |
| f_rand | random | 0.6814 | 0.7516 | **-0.0702** | 0.6508 | 0.7647 |
| f_rand | sensor_range | 0.0108 | 0.4553 | **-0.4445** | 0.0007 | 0.3240 |
| f_rand | pca_error | 0.9141 | 0.9387 | **-0.0246** | 0.7979 | 0.9257 |
| f_rand | l2_norm | 0.8340 | 0.9021 | **-0.0680** | 0.8006 | 0.9170 |
| f_rand | nn_distance | 0.8842 | 0.9189 | **-0.0347** | 0.7923 | 0.9245 |
| f_ds | random | 0.5966 | 0.7571 | **-0.1605** | 0.6501 | 0.7646 |
| f_ds | sensor_range | 0.0090 | 0.5817 | **-0.5726** | 0.0007 | 0.4713 |
| f_ds | pca_error | 0.7470 | 0.9511 | **-0.2041** | 0.7956 | 0.9241 |
| f_ds | l2_norm | 0.7862 | 0.9071 | **-0.1209** | 0.8007 | 0.9163 |
| f_ds | nn_distance | 0.9617 | 0.9314 | **0.0303** | 0.7902 | 0.9246 |
| f_unk | random | 0.7005 | 0.7493 | **-0.0488** | 0.6502 | 0.7651 |
| f_unk | sensor_range | 0.1329 | 0.3941 | **-0.2612** | 0.0000 | 0.2841 |
| f_unk | pca_error | 0.8583 | 0.9744 | **-0.1160** | 0.7964 | 0.9253 |
| f_unk | l2_norm | 0.8328 | 0.9055 | **-0.0727** | 0.8000 | 0.9124 |
| f_unk | nn_distance | 0.8854 | 0.9202 | **-0.0348** | 0.7911 | 0.9246 |
| ffonly | random | — | — | — | 0.6513 | 0.7647 |
| ffonly | sensor_range | — | — | — | 0.0065 | 0.8072 |
| ffonly | pca_error | — | — | — | 0.7908 | 0.9345 |
| ffonly | l2_norm | — | — | — | 0.7962 | 0.9159 |
| ffonly | nn_distance | — | — | — | 0.7880 | 0.9273 |

### pak_auc_prc_auc

| fold | model | seen | unseen | **G=seen−unseen** | exclhard | full |
|---|---|---|---|---|---|---|
| f_step | random | 0.5398 | 0.5894 | **-0.0497** | 0.4578 | 0.6196 |
| f_step | sensor_range | 0.1021 | 0.4525 | **-0.3504** | 0.0004 | 0.2796 |
| f_step | pca_error | 0.9432 | 0.9841 | **-0.0409** | 0.6089 | 0.9683 |
| f_step | l2_norm | 0.9103 | 0.9147 | **-0.0044** | 0.6029 | 0.9260 |
| f_step | nn_distance | 0.9486 | 0.9736 | **-0.0250** | 0.6222 | 0.9646 |
| f_rand | random | 0.4967 | 0.5987 | **-0.1020** | 0.4581 | 0.6199 |
| f_rand | sensor_range | 0.0058 | 0.2985 | **-0.2927** | 0.0004 | 0.1953 |
| f_rand | pca_error | 0.9740 | 0.9878 | **-0.0139** | 0.6185 | 0.9767 |
| f_rand | l2_norm | 0.8207 | 0.9227 | **-0.1020** | 0.6027 | 0.9183 |
| f_rand | nn_distance | 0.9435 | 0.9717 | **-0.0282** | 0.6168 | 0.9638 |
| f_ds | random | 0.3960 | 0.6073 | **-0.2113** | 0.4573 | 0.6198 |
| f_ds | sensor_range | 0.0050 | 0.4120 | **-0.4070** | 0.0004 | 0.3095 |
| f_ds | pca_error | 0.8343 | 0.9911 | **-0.1567** | 0.6250 | 0.9753 |
| f_ds | l2_norm | 0.8316 | 0.9353 | **-0.1038** | 0.6035 | 0.9303 |
| f_ds | nn_distance | 0.9856 | 0.9738 | **0.0118** | 0.6197 | 0.9669 |
| f_unk | random | 0.5223 | 0.5947 | **-0.0724** | 0.4573 | 0.6203 |
| f_unk | sensor_range | 0.0719 | 0.2496 | **-0.1777** | 0.0000 | 0.1677 |
| f_unk | pca_error | 0.9439 | 0.9960 | **-0.0521** | 0.6248 | 0.9773 |
| f_unk | l2_norm | 0.7844 | 0.9287 | **-0.1442** | 0.6045 | 0.9149 |
| f_unk | nn_distance | 0.9148 | 0.9751 | **-0.0603** | 0.6171 | 0.9620 |
| ffonly | random | — | — | — | 0.4586 | 0.6198 |
| ffonly | sensor_range | — | — | — | 0.0033 | 0.6789 |
| ffonly | pca_error | — | — | — | 0.6144 | 0.9861 |
| ffonly | l2_norm | — | — | — | 0.6309 | 0.9611 |
| ffonly | nn_distance | — | — | — | 0.6405 | 0.9815 |

### vus_pr

| fold | model | seen | unseen | **G=seen−unseen** | exclhard | full |
|---|---|---|---|---|---|---|
| f_step | random | 0.6582 | 0.7428 | **-0.0846** | 0.5269 | 0.7978 |
| f_step | sensor_range | 0.6708 | 0.8088 | **-0.1380** | 0.5265 | 0.8217 |
| f_step | pca_error | 0.9101 | 0.9714 | **-0.0613** | 0.5298 | 0.9545 |
| f_step | l2_norm | 0.8616 | 0.8677 | **-0.0061** | 0.5278 | 0.8952 |
| f_step | nn_distance | 0.9122 | 0.9529 | **-0.0407** | 0.5400 | 0.9464 |
| f_rand | random | 0.5854 | 0.7605 | **-0.1750** | 0.5266 | 0.7978 |
| f_rand | sensor_range | 0.5854 | 0.7937 | **-0.2083** | 0.5265 | 0.8118 |
| f_rand | pca_error | 0.9446 | 0.9799 | **-0.0353** | 0.5392 | 0.9667 |
| f_rand | l2_norm | 0.7293 | 0.8916 | **-0.1623** | 0.5287 | 0.8910 |
| f_rand | nn_distance | 0.9063 | 0.9547 | **-0.0484** | 0.5362 | 0.9478 |
| f_ds | random | 0.4384 | 0.7744 | **-0.3361** | 0.5265 | 0.7978 |
| f_ds | sensor_range | 0.4392 | 0.8278 | **-0.3886** | 0.5265 | 0.8269 |
| f_ds | pca_error | 0.7601 | 0.9832 | **-0.2231** | 0.5439 | 0.9646 |
| f_ds | l2_norm | 0.7022 | 0.9099 | **-0.2078** | 0.5293 | 0.9060 |
| f_ds | nn_distance | 0.9530 | 0.9583 | **-0.0053** | 0.5385 | 0.9520 |
| f_unk | random | 0.6271 | 0.7521 | **-0.1250** | 0.5262 | 0.7978 |
| f_unk | sensor_range | 0.6307 | 0.7849 | **-0.1542** | 0.5265 | 0.8098 |
| f_unk | pca_error | 0.8967 | 0.9907 | **-0.0940** | 0.5429 | 0.9664 |
| f_unk | l2_norm | 0.7180 | 0.8957 | **-0.1777** | 0.5296 | 0.8888 |
| f_unk | nn_distance | 0.8584 | 0.9602 | **-0.1018** | 0.5362 | 0.9447 |
| ffonly | random | — | — | — | 0.5268 | 0.7978 |
| ffonly | sensor_range | — | — | — | 0.5267 | 0.9077 |
| ffonly | pca_error | — | — | — | 0.5334 | 0.9812 |
| ffonly | l2_norm | — | — | — | 0.5468 | 0.9425 |
| ffonly | nn_distance | — | — | — | 0.5522 | 0.9698 |

### aff_f1

| fold | model | seen | unseen | **G=seen−unseen** | exclhard | full |
|---|---|---|---|---|---|---|
| f_step | random | 0.9145 | 0.9161 | **-0.0015** | 0.9129 | 0.9163 |
| f_step | sensor_range | 0.4378 | 0.8404 | **-0.4026** | 0.0380 | 0.6528 |
| f_step | pca_error | 0.9396 | 0.9747 | **-0.0350** | 0.9155 | 0.9254 |
| f_step | l2_norm | 0.9153 | 0.9162 | **-0.0009** | 0.9134 | 0.9167 |
| f_step | nn_distance | 0.9530 | 0.9464 | **0.0066** | 0.9141 | 0.9203 |
| f_rand | random | 0.9145 | 0.9162 | **-0.0017** | 0.9127 | 0.9165 |
| f_rand | sensor_range | 0.3180 | 0.6965 | **-0.3784** | 0.0347 | 0.5588 |
| f_rand | pca_error | 0.9793 | 0.9777 | **0.0016** | 0.9155 | 0.9319 |
| f_rand | l2_norm | 0.9144 | 0.9164 | **-0.0020** | 0.9134 | 0.9167 |
| f_rand | nn_distance | 0.9653 | 0.9394 | **0.0260** | 0.9144 | 0.9193 |
| f_ds | random | 0.9109 | 0.9162 | **-0.0053** | 0.9132 | 0.9164 |
| f_ds | sensor_range | 0.3305 | 0.7681 | **-0.4376** | 0.0347 | 0.6578 |
| f_ds | pca_error | 0.9473 | 0.9849 | **-0.0375** | 0.9155 | 0.9257 |
| f_ds | l2_norm | 0.9114 | 0.9165 | **-0.0051** | 0.9134 | 0.9167 |
| f_ds | nn_distance | 0.9859 | 0.9413 | **0.0446** | 0.9141 | 0.9239 |
| f_unk | random | 0.9151 | 0.9160 | **-0.0010** | 0.9129 | 0.9165 |
| f_unk | sensor_range | 0.2879 | 0.6599 | **-0.3720** | 0.0000 | 0.5046 |
| f_unk | pca_error | 0.9399 | 0.9902 | **-0.0503** | 0.9155 | 0.9392 |
| f_unk | l2_norm | 0.9150 | 0.9163 | **-0.0014** | 0.9134 | 0.9167 |
| f_unk | nn_distance | 0.9336 | 0.9548 | **-0.0212** | 0.9144 | 0.9189 |
| ffonly | random | — | — | — | 0.9137 | 0.9165 |
| ffonly | sensor_range | — | — | — | 0.2634 | 0.9227 |
| ffonly | pca_error | — | — | — | 0.9160 | 0.9826 |
| ffonly | l2_norm | — | — | — | 0.9134 | 0.9167 |
| ffonly | nn_distance | — | — | — | 0.9146 | 0.9447 |

## 3. 분해 분석

### 3a. G (seen−unseen) 요약 — pak_auc_f1

| model | f_step | f_rand | f_ds | f_unk | 4/4 부호 일치 |
|---|---|---|---|---|---|
| random | -0.0332 | -0.0702 | -0.1605 | -0.0488 | 예 |
| sensor_range | -0.4319 | -0.4445 | -0.5726 | -0.2612 | 예 |
| pca_error | -0.0515 | -0.0246 | -0.2041 | -0.1160 | 예 |
| l2_norm | -0.0221 | -0.0680 | -0.1209 | -0.0727 | 예 |
| nn_distance | -0.0509 | -0.0347 | 0.0303 | -0.0348 | 아니오 |

### 3b. C_dmg = ffonly − contaminated (오염 피해; 양수 = 오염이 성능을 깎음) — pak_auc_f1

| model | fold | C_dmg(seen) | C_dmg(unseen) | C_dmg(full) |
|---|---|---|---|---|
| random | f_step | 0.0002 | -0.0004 | 0.0002 |
| random | f_rand | -0.0018 | 0.0004 | -0.0000 |
| random | f_ds | 0.0015 | -0.0005 | 0.0000 |
| random | f_unk | -0.0011 | -0.0004 | -0.0004 |
| sensor_range | f_step | 0.7619 | 0.2337 | 0.3740 |
| sensor_range | f_rand | 0.8889 | 0.4262 | 0.4831 |
| sensor_range | f_ds | 0.9894 | 0.2871 | 0.3358 |
| sensor_range | f_unk | 0.5923 | 0.5468 | 0.5230 |
| pca_error | f_step | 0.1261 | 0.0686 | 0.0115 |
| pca_error | f_rand | 0.0796 | 0.0568 | 0.0087 |
| pca_error | f_ds | 0.2517 | 0.0441 | 0.0103 |
| pca_error | f_unk | 0.1319 | 0.0229 | 0.0092 |
| l2_norm | f_step | 0.0637 | -0.0010 | 0.0006 |
| l2_norm | f_rand | 0.0273 | -0.0015 | -0.0010 |
| l2_norm | f_ds | 0.1998 | -0.0001 | -0.0004 |
| l2_norm | f_unk | -0.0011 | 0.0249 | 0.0036 |
| nn_distance | f_step | 0.0809 | 0.0276 | 0.0027 |
| nn_distance | f_rand | 0.0693 | 0.0377 | 0.0028 |
| nn_distance | f_ds | 0.0370 | 0.0272 | 0.0027 |
| nn_distance | f_unk | 0.0408 | 0.0462 | 0.0027 |

### 3c. excluded-hard (IDV 3/9/15) 규칙 검증

설계 §2.2 예측: 폐루프 보상으로 모든 방법에서 거의 비식별 (낮은 pak_auc_f1, prc_auc ≈ positive rate 수준).

| 조건 | model | exclhard pak_auc_f1 | exclhard prc_auc | full 대비 |
|---|---|---|---|---|
| f_step | random | 0.6506 | 0.5003 | -0.1139 |
| f_step | sensor_range | 0.0007 | 0.5001 | -0.4324 |
| f_step | pca_error | 0.7882 | 0.5095 | -0.1348 |
| f_step | l2_norm | 0.7991 | 0.5030 | -0.1162 |
| f_step | nn_distance | 0.7914 | 0.5196 | -0.1332 |
| f_rand | random | 0.6508 | 0.5001 | -0.1139 |
| f_rand | sensor_range | 0.0007 | 0.5001 | -0.3233 |
| f_rand | pca_error | 0.7979 | 0.5175 | -0.1279 |
| f_rand | l2_norm | 0.8006 | 0.5033 | -0.1164 |
| f_rand | nn_distance | 0.7923 | 0.5154 | -0.1321 |
| f_ds | random | 0.6501 | 0.5001 | -0.1145 |
| f_ds | sensor_range | 0.0007 | 0.5001 | -0.4706 |
| f_ds | pca_error | 0.7956 | 0.5225 | -0.1286 |
| f_ds | l2_norm | 0.8007 | 0.5040 | -0.1156 |
| f_ds | nn_distance | 0.7902 | 0.5175 | -0.1344 |
| f_unk | random | 0.6502 | 0.4997 | -0.1149 |
| f_unk | sensor_range | 0.0000 | 0.5000 | -0.2841 |
| f_unk | pca_error | 0.7964 | 0.5216 | -0.1288 |
| f_unk | l2_norm | 0.8000 | 0.5047 | -0.1123 |
| f_unk | nn_distance | 0.7911 | 0.5154 | -0.1335 |
| ffonly | random | 0.6513 | 0.5005 | -0.1134 |
| ffonly | sensor_range | 0.0065 | 0.5007 | -0.8007 |
| ffonly | pca_error | 0.7908 | 0.5136 | -0.1437 |
| ffonly | l2_norm | 0.7962 | 0.5265 | -0.1198 |
| ffonly | nn_distance | 0.7880 | 0.5342 | -0.1393 |

## 4. Per-fault 분석

**데이터 통계 subtle-fault 랭킹** (post-onset 평균 max|z|, FF-train 기준; 설계 §2.2의 동결 후보): 하위 5 (usable 17 중) = **[16, 19, 10, 5, 20]** / 전체 하위 5 = [3, 9, 15, 16, 19]

| fault | family | mean max\|z\| | 비고 |
|---|---|---|---|
| IDV1 | step | 16.536 |  |
| IDV2 | step | 31.266 |  |
| IDV3 | EXCLUDED-HARD | 2.420 | 제외 규칙 대상 |
| IDV4 | step | 6.935 |  |
| IDV5 | step | 3.725 | subtle-5 |
| IDV6 | step | 80.895 |  |
| IDV7 | step | 12.911 |  |
| IDV8 | random | 11.457 |  |
| IDV9 | EXCLUDED-HARD | 2.421 | 제외 규칙 대상 |
| IDV10 | random | 3.489 | subtle-5 |
| IDV11 | random | 6.474 |  |
| IDV12 | random | 12.725 |  |
| IDV13 | drift_sticking | 15.130 |  |
| IDV14 | drift_sticking | 15.639 |  |
| IDV15 | EXCLUDED-HARD | 2.443 | 제외 규칙 대상 |
| IDV16 | unknown | 2.957 | subtle-5 |
| IDV17 | unknown | 30.758 |  |
| IDV18 | unknown | 70.271 |  |
| IDV19 | unknown | 2.993 | subtle-5 |
| IDV20 | unknown | 5.089 | subtle-5 |

### Per-fault pak_auc_f1 (각 fault 20 runs + FF 40 runs 기준, lite)

| fault | family | random@ffonly | sensor_range@ffonly | pca_error@ffonly | l2_norm@ffonly | nn_distance@ffonly | random@f_step | sensor_range@f_step | pca_error@f_step | l2_norm@f_step | nn_distance@f_step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IDV1 | step | 0.4814 | 0.9997 | 1.0000 | 0.9999 | 1.0000 | 0.4813 | 0.0057 | 0.7491 | 0.9385 | 0.9267 |
| IDV2 | step | 0.4848 | 0.9996 | 0.9998 | 0.9998 | 0.9999 | 0.4855 | 0.0069 | 0.8766 | 0.8074 | 0.9025 |
| IDV3 | EXCL | 0.4833 | 0.0052 | 0.6087 | 0.6156 | 0.6041 | 0.4835 | 0.0007 | 0.6045 | 0.6223 | 0.6138 |
| IDV4 | step | 0.4798 | 0.9997 | 1.0000 | 0.8456 | 0.9971 | 0.4869 | 0.0007 | 0.6188 | 0.6329 | 0.6208 |
| IDV5 | step | 0.4817 | 0.5081 | 1.0000 | 0.7386 | 0.7665 | 0.4780 | 0.0000 | 0.7815 | 0.6408 | 0.7061 |
| IDV6 | step | 0.4794 | 0.9998 | 1.0000 | 0.9999 | 1.0000 | 0.4811 | 0.7236 | 0.9993 | 0.9942 | 0.9753 |
| IDV7 | step | 0.4809 | 0.9998 | 1.0000 | 0.9960 | 1.0000 | 0.4819 | 0.0591 | 0.9996 | 0.7296 | 0.9307 |
| IDV8 | random | 0.4837 | 0.9983 | 0.9979 | 0.9861 | 0.9994 | 0.4824 | 0.7365 | 0.9974 | 0.7998 | 0.9961 |
| IDV9 | EXCL | 0.4833 | 0.0045 | 0.6121 | 0.6189 | 0.6056 | 0.4832 | 0.0007 | 0.6058 | 0.6251 | 0.6151 |
| IDV10 | random | 0.4835 | 0.5562 | 0.9785 | 0.6639 | 0.8473 | 0.4798 | 0.0007 | 0.9762 | 0.6308 | 0.7152 |
| IDV11 | random | 0.4807 | 0.9220 | 0.9956 | 0.7068 | 0.9470 | 0.4835 | 0.3392 | 0.7223 | 0.6477 | 0.7942 |
| IDV12 | random | 0.4855 | 0.9988 | 0.9996 | 0.9864 | 0.9997 | 0.4850 | 0.6306 | 0.9996 | 0.8183 | 0.9969 |
| IDV13 | drift_sticking | 0.4827 | 0.9968 | 0.9973 | 0.9908 | 0.9974 | 0.4801 | 0.7294 | 0.9971 | 0.8424 | 0.9955 |
| IDV14 | drift_sticking | 0.4784 | 0.9997 | 1.0000 | 0.9787 | 1.0000 | 0.4838 | 0.9991 | 1.0000 | 0.8531 | 1.0000 |
| IDV15 | EXCL | 0.4831 | 0.0095 | 0.6089 | 0.6199 | 0.6155 | 0.4836 | 0.0007 | 0.6125 | 0.6251 | 0.6194 |
| IDV16 | unknown | 0.4863 | 0.2458 | 0.9858 | 0.6627 | 0.7909 | 0.4826 | 0.0007 | 0.9826 | 0.6316 | 0.6845 |
| IDV17 | unknown | 0.4837 | 0.9894 | 0.9987 | 0.8938 | 0.9899 | 0.4843 | 0.9629 | 0.9985 | 0.7838 | 0.9827 |
| IDV18 | unknown | 0.4831 | 0.9973 | 0.9980 | 0.9954 | 0.9974 | 0.4841 | 0.9772 | 0.9980 | 0.9534 | 0.9968 |
| IDV19 | unknown | 0.4775 | 0.1882 | 0.9933 | 0.6407 | 0.7860 | 0.4850 | 0.1672 | 0.6464 | 0.6486 | 0.7684 |
| IDV20 | unknown | 0.4830 | 0.7247 | 0.9657 | 0.5790 | 0.8702 | 0.4798 | 0.0013 | 0.9158 | 0.5961 | 0.7222 |

near/far 쌍 (설계 §2.2): [(4, 11), (5, 12)] — MAE 실험에서 F-STEP fold의 unseen 11,12가 near-variable.

## 5. 게이트 판정

검증 게이트: **PASS** — 모든 sanity check 통과. 이 결과는 MAE 조건 A/B/B0 비교의 anchor로 사용 가능.


## 6. (추가 2026-06-11) Composition-등화 검증 — macro per-fault G

사용자 지적: "random이 G≠0이면 그냥 무작위 결과 아닌가? seen/unseen을 같은 조건으로 평가해야 하지 않나?" — 타당하며, 등화 방법이 존재한다.

**원인**: stream(micro) 평가는 partition 전체를 한 덩어리로 평가한다. seen/unseen partition은 fault 수가 달라 positive rate(41.7~62.5% vs 70.5~73.5%)와 region 수가 다르고, F1 계열 지표는 random scorer에 대해 positive rate의 증가함수다(precision ≈ positive rate). 즉 random의 micro G는 모델이 아니라 **지표-구성 artifact**다.

**등화 방법**: per-fault 평가는 모든 fault가 동일 구성(20 runs × 800 anomaly + 동일 FF 40 runs = positive rate 29.4%, 20 regions)이므로, fault별 지표를 구한 뒤 seen/unseen 그룹에서 **macro 평균**하면 평가 조건이 완전히 같아진다.

| model | fold | micro G | **macro G** |
|---|---|---|---|
| random | f_step | -0.033 | **-0.000** |
| random | f_rand | -0.070 | **+0.001** |
| random | f_ds | -0.160 | **-0.002** |
| random | f_unk | -0.049 | **+0.000** |
| sensor_range | f_step | -0.432 | **-0.371** |
| sensor_range | f_rand | -0.444 | **-0.322** |
| sensor_range | f_ds | -0.573 | **-0.428** |
| sensor_range | f_unk | -0.261 | **-0.179** |
| pca_error | f_step | -0.052 | **-0.093** |
| pca_error | f_rand | -0.025 | **-0.025** |
| pca_error | f_ds | -0.204 | **-0.161** |
| pca_error | f_unk | -0.116 | **-0.127** |
| l2_norm | f_step | -0.022 | **+0.045** |
| l2_norm | f_rand | -0.068 | **-0.070** |
| l2_norm | f_ds | -0.121 | **-0.030** |
| l2_norm | f_unk | -0.073 | **-0.099** |
| nn_distance | f_step | -0.051 | **-0.034** |
| nn_distance | f_rand | -0.035 | **-0.018** |
| nn_distance | f_ds | +0.030 | **+0.093** |
| nn_distance | f_unk | -0.035 | **-0.096** |

**판정**: (1) random의 macro G = −0.002~+0.001 ≈ 0 (per-fault 값 자체가 0.479~0.487로 사실상 상수, roc 0.498~0.505) → micro G −0.03~−0.16은 전액 composition artifact였음이 확정. (2) 실제 모델들의 오염 효과는 등화 후에도 살아남는다 (pca f_ds macro −0.161, sensor_range −0.18~−0.43, nn_distance f_ds **+0.093**). (3) 단 일부 micro 수치는 왜곡이 있었다 — l2_norm f_step은 micro −0.022가 macro **+0.045**로 부호 반전. **결론: MAE 본 실험의 seen/unseen 주 비교는 macro per-fault로 수행하고(평가 조건 완전 동일), stream micro는 보조로 강등한다. 이는 사전 등록 설계 §4.4(b)의 per-fault matched 분석을 co-primary로 둔 결정의 실증적 근거다.**
