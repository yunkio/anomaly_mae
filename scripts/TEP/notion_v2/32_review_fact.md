# 40_final_r1.md 사실 검증 결과 (fact-check r1)

검증일: 2026-06-11 · 검증자: fact-checker subagent
대상: `temp/0610/TEP/notion_v2/40_final_r1.md`
원천: `results/12_20260610_211815_tep_typegen_simple/{analysis_report.md, idv_hard_report.md, */*/epoch_metrics.json, */*/per_fault_metrics.json, */random/metadata.json, sweep/**}`, `data/manifest.json`, `mae_anomaly/evaluator.py`

## 판정: **ALL VERIFIED**

표 셀 전수 + 본문 수치 전수를 python3로 원천 JSON에서 재계산하여 대조한 결과, 불일치 없음.

## 검증 범위 (전수 대조 항목)

1. **§1.2 데이터 구성 표** — manifest.json 대조: train 288,000 samples/300 runs/anomaly 16.67% (4 fold 모두 0.16667), ffonly 230,400/240 runs, test 422,400 = 440 runs × 960, 라벨 등화 F-STEP [1,2,4,5,6,7]×10 / F-RAND [8,10,11,12]×15 / F-DS [13,14]×30 / F-UNK [16..20]×12, onset sample 161 (manifest `fault_onset_idx_0based`=160), region 400개 (metadata `n_regions`=400), internal seam 439개, test 구성 "faults 1..20 × runs 441..460 + FF 461..500" (manifest `order` 문자열 일치). 모두 일치.
2. **§1.2 fold 표** — seen fault 집합 4개 모두 manifest와 일치, unseen 수 11/13/15/12종 = metadata partition fault 목록과 일치 (excl 3/9/15 제외 산술 정확).
3. **§1.3 매트릭스** — ① 25 runs (5모델×5조건 디렉토리 실재), random n_runs=5 (metadata `"n_runs": 5` 확인), ② 20 runs (sweep 2모델×2fold×5label 실재), 검증 게이트 항목들 (16.67%, sample 161, 440 runs 경계, 5-box per-run 재시작) = analysis_report §1 일치.
4. **§3.1 macro G 표 (25셀)** — per_fault_metrics.json에서 재계산. contaminated 16셀: random −0.0003/+0.0014/−0.0018/+0.0003, pca −0.0929/−0.0245/−0.1607/−0.1271, l2 +0.0446/−0.0702/−0.0296/−0.0990, nn −0.0338/−0.0178/+0.0925/−0.0957, sensor −0.3714/−0.3222/−0.4278/−0.1793 → 표기값과 3자리 반올림 일치 (nn f_ds 정확값 0.092515 → +0.093 정당). ffonly macro G 범위 5셀: random [−0.0018,+0.0016], pca [−0.0091,+0.0081], l2 [−0.1534,+0.1384], nn [−0.0760,+0.0660], sensor [−0.2858,+0.1898] → 표기 범위와 모두 일치. 색상 규칙(0.02/0.10)과 굵게(양수 G) 적용도 전 셀 일관.
5. **§3.1 본문** — seen positive rate 41.7~62.5% (f_ds 32,000/76,800=41.67% ~ f_step 96,000/153,600=62.5%), unseen 70.5~73.5% (f_step 176,000/249,600=70.51% ~ f_ds 240,000/326,400=73.53%), per-fault 29.4% (16,000/54,400=29.41%), run당 800 anomaly (960−160), FF 40 runs — 모두 일치. 판독1·2: random micro −0.03~−0.16 ✓, l2 f_step micro −0.022→macro +0.045 부호반전 ✓, pca f_ds −0.161 ✓, sensor 전 fold −0.18~−0.43 ✓, clean pca |G|<0.010 (max 0.0091) ✓, sensor ffonly −0.286~+0.190 ✓, F-UNK seen에 subtle 16/19/20 포함 ✓, nn f_ds +0.093 최대 양수 ✓.
6. **§3.2 C_dmg 표 (20셀, per-fault macro)** — 재계산: random −0.0011/+0.0005/−0.0009/+0.0005, pca 0.1625/0.0740/0.2092/0.1439, l2 0.1394/0.1408/0.2307/0.0790, nn 0.1169/0.0911/0.0356/0.0945, sensor 0.7851/0.8584/0.9894/0.5259 → 표기값과 반올림 일치. 색상 규칙(0.05/0.15) 전 셀 일관. 본문: IDV11 spillover 0.996→0.722 (0.9956→0.7223) ✓, sensor C_dmg 0.53~0.99 ✓, nn F-DS 0.036 ✓.
7. **§3.3 sweep 표 (20셀 + 잔류 run 수 5개)** — sweep epoch_metrics.json 대조: pca@F-STEP 0.8736/0.8794/0.9427/0.9748/0.99968 → 0.874/0.879/0.943/0.975/0.9997 ✓, pca@F-DS 0.7470/0.7481/0.7566/0.8356/0.9986 → 0.747/0.748/0.757/0.836/0.999 ✓, sensor@F-STEP 0.1842/0.2002/0.2119/0.3035/0.9461 ✓, sensor@F-DS 0.0090/0.0090/0.00947/0.0208/0.9984 → 0.009/0.009/0.009/0.021/0.998 ✓. kept_faulty_runs 60/48/30/12/0 (sweep_summary) ✓. 본문 0.836 vs 0.999 ✓.
8. **§3.4 3단 검증 표 (15셀)** — idv_hard_report.md와 전 셀 일치: L1 0.568/0.514/0.515/0.996/0.514·0.511, L2 1.000(xmeas_21 run-mean)/0.740/0.967(xmeas_22 run-std)/1.000/1.000·1.000, L3 pca@ffonly roc 0.510/0.513/0.512/1.000/0.966·0.992. 각주: AUC null std ≈0.08 (해석적으로 √(61/9600)=0.0797 ✓), 104회 = 52 feature × 2 집계 ✓ (manifest feature_cols 52개).
9. **§3.5 + 인사이트 + §7** — random full 0.765 (0.7645~0.7651) ✓, positive rate 75.8% (320,000/422,400=0.7576) ✓, per-fault random 0.48 (0.4775~0.4863) ✓, exclhard pca 0.79 (0.7882~0.7979) ✓, exclhard prc_auc 0.51 (0.5095~0.5225) vs positive rate 50% (48,000/96,000, metadata exclhard 96,000 pts) ✓, pca macro C_dmg 0.07~0.21 (0.0740~0.2092) ✓, 인사이트1 micro −0.16 ✓, 인사이트4 1.000/0.967 ✓.
10. **§4 해석** — pca@ffonly usable 17종 중 ≥0.99가 14종, 나머지 3종 (IDV10 0.9785, IDV16 0.9858, IDV20 0.9657) 최저 0.966 ✓ (per_fault JSON 재계산으로 확정).
11. **부록 A micro G (20셀)** — epoch_metrics.json seen/unseen 재계산과 전 셀 반올림 일치 (analysis_report §2와도 일치).
12. **평가 코드 명칭** — `compute_full_metric_set`: mae_anomaly/evaluator.py:864에 실재, TEP 스크립트는 그 thin wrapper(`compute_all_metrics`) 사용 → 표기 정당.

## NOTE (불일치 아님, 참고)

- **N-1 (§3.4 각주 "우연만으로 0.75 수준")**: 이 값은 원천 파일에 없는 파생 통계 추정치. 독립 시뮬레이션(Mann-Whitney null, 20 vs 40 runs, 104회 folded max, 1,000 trials) 결과 E[max] ≈ 0.715 (95th pct 0.770). "0.75"는 Gumbel 상한 근사(0.5+3.05σ≈0.744)에 해당하여 기대값으로는 ~0.03 과대. 단, IDV9의 0.740은 null max 분포의 81번째 백분위(우연만으로 19% 확률로 초과)로 "잡음과 구분 불가" 결론 자체는 유효. "수준"이라는 표현 범위 내로 판단, 불일치로 처리하지 않음.
- **N-2 ("FaultFree runs 1~240")**: manifest는 `n_runs: 240`만 기록하고 run 번호 범위는 명시하지 않음. 설계 문서 유래 표기로 manifest와 모순 없음.
- **N-3 (경계 반올림 셀)**: nn f_ds macro G 0.092515→+0.093, random C_dmg ±0.0005~0.0011→±0.001, sensor@F-DS 50% 0.00947→0.009 등 반올림 경계 셀들은 모두 정상 반올림 범위 내.
