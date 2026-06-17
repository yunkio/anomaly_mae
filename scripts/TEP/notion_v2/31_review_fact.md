# 사실 검증 보고 — 20_draft.md (검증일 2026-06-11)

검증자: fact-checker subagent. 원천: `results/12_20260610_211815_tep_typegen_simple/` (analysis_report.md, idv_hard_report.md, 5 fold × 5 model epoch_metrics.json + per_fault_metrics.json, sweep/*/​*/labeled_*/epoch_metrics.json, sweep_summary_*.json), `data/manifest.json`. 모든 macro 값은 per_fault_metrics.json에서 python3로 재계산.

## 판정: 불일치 5건 (모두 마지막 자리 rounding 수준, 결론 영향 없음)

### FACT-1. §3.2 C_dmg 표 — pca_error @ F-STEP
- Draft: **0.163**
- 실제: ffonly macro_seen − f_step macro_seen = **0.16249** → 올바른 3자리 반올림은 **0.162**
- 색상(red, >0.15)·해석은 그대로 유효.

### FACT-2. §3.2 C_dmg 표 — l2_norm @ F-STEP
- Draft: **0.140**
- 실제: **0.13945** → 올바른 3자리 반올림은 **0.139**
- 색상(orange)·해석은 그대로 유효.

### FACT-3. §3.3 sweep 표 — sensor_range @ F-DS, labeled 50%
- Draft: **0.010**
- 실제: seen_pak_auc_f1 = **0.009473** → 올바른 3자리 반올림은 **0.009**
- "100% 직전까지 사실상 0" 서사는 그대로 유효 (오히려 더 강화됨).

### FACT-4 (minor). §3.2 C_dmg 표 — random @ F-RAND, @ F-UNK
- Draft: **+0.000** (두 셀)
- 실제: F-RAND **+0.00052**, F-UNK **+0.00053** → round-half-up 3자리는 **+0.001**
- |값| < 0.001 수준의 표기 선택 문제. 음성 대조(≈0) 결론에 영향 없음.

### FACT-5 (borderline). 판독 2 및 §4.1 본문 — "pca_error ffonly macro G 네 fold 전부 절대값 0.009 이내"
- 실제 4 fold 값: +0.00810 / −0.00236 / +0.00447 / **−0.00910** (f_unk)
- max |G| = 0.00910으로 raw 값은 0.009를 미세 초과 (표시 정밀도 3자리에서는 0.009로 성립).
- §3.1 표의 범위 표기 "−0.009 ~ +0.008"은 정확. "0.010 이내" 또는 "≈0.009"로 완화 권장.

## 일치 확인된 항목 (전수 검증)

- **§1.2 데이터 표**: train contaminated 288,000 samples·anomaly 16.67%·FF 240 runs + faulty 60 runs, 라벨 등화 6×10/4×15/2×30/5×12, ffonly 230,400, test 440 runs = 422,400 samples (fault 20종 × runs 441~460 + FF 461~500), onset sample 161 (manifest 0-based 160), region 400개, internal seam 439개 — 모두 manifest.json·analysis_report §1과 일치.
- **§1.2 fold 표**: seen faults (1,2,4,5,6,7 / 8,10,11,12 / 13,14 / 16~20) manifest와 일치, unseen 11/13/15/12종 산술 일치.
- **§1.3 매트릭스**: 25 runs (5모델×5조건), random n_runs=5 (metadata.json) ✓, sweep 20 runs ✓.
- **§3.1 micro positive rate**: seen 41.7~62.5% (0.4167~0.625), unseen 70.5~73.5% (0.7051~0.7353), per-fault 29.4%, full 75.8% — epoch_metrics anomaly_ratio와 일치.
- **§3.1 macro G 표 20셀 전부**: random −0.000/+0.001/−0.002/+0.000, pca −0.093/−0.025/−0.161/−0.127, l2 +0.045/−0.070/−0.030/−0.099, nn −0.034/−0.018/+0.093/−0.096, sensor −0.371/−0.322/−0.428/−0.179 — per_fault 재계산과 일치.
- **§3.1 ffonly macro G 범위 5셀 전부**: random −0.002~+0.002, pca −0.009~+0.008, l2 −0.153~+0.138, nn −0.076~+0.066, sensor −0.286~+0.190 — 재계산과 일치.
- **§3.2 C_dmg**: FACT-1/2/4 외 나머지 셀 (sensor 0.785/0.858/0.989/0.526, pca 0.074/0.209/0.144, l2 0.141/0.231/0.079, nn 0.117/0.091/0.036/0.095, random −0.001/−0.001) 일치. 본문 "0.53~0.99", "0.07~0.21"(§7), "피해 0.036" 일치.
- **§3.2 spillover**: IDV11 pca 0.9956→0.7223 ("0.996에서 0.722") 일치.
- **§3.3 sweep**: FACT-3 외 17셀 전부 일치 (pca@F-STEP 0.874/0.879/0.943/0.975/0.9997, pca@F-DS 0.747/0.748/0.757/0.836/0.999, sensor@F-STEP 0.184/0.200/0.212/0.304/0.946, sensor@F-DS 0.009/0.009/0.021/0.998). 잔류 run 수 60/48/30/12/0 = sweep_summary kept_faulty_runs와 일치.
- **§3.4 3단 검증 표**: L1 (0.568/0.514/0.515/0.996/0.514/0.511), L2 (1.000 xmeas_21 run-mean / 0.740 / 0.967 xmeas_22 run-std / 1.000 / 1.000·1.000), L3 pca@ffonly roc (0.510/0.513/0.512/1.000/0.966/0.992) — idv_hard_report.md와 전부 일치. 52 features = manifest feature_cols 52개 ✓.
- **§3.5**: random full 0.765 (0.7645~0.7651), per-fault random 0.48 (0.4775~0.4863), exclhard pca 0.79 (0.7882~0.7979), exclhard prc_auc 0.51 (0.5095~0.5225), exclhard positive rate 50% (0.5) — 일치.
- **§4**: pca@ffonly usable 17종 중 14종 ≥0.99, 나머지 3종 (IDV10 0.9785, IDV16 0.9858, IDV20 0.9657) 최저 0.966 — per_fault 재계산과 일치.
- **§6.5**: subtle-set {16,19,10,5,20} = analysis_report §4 동결 후보와 일치.
- **부록 A micro G 표 20셀 전부**: epoch_metrics seen−unseen 재계산과 일치 (borderline 셀 −0.160=−0.16048, −0.444=−0.44450, −0.052=−0.05151 포함 모두 정확한 반올림).
- **검증 게이트 callout**: 16.67%, onset 161, 440 runs×5pt smoothing 경계 — analysis_report §1과 일치.

## 권장 수정

1. §3.2 표: 0.163→0.162, 0.140→0.139, (선택) random +0.000→+0.001 두 셀.
2. §3.3 표: sensor_range@F-DS 50% 0.010→0.009.
3. 판독 2·§4.1 본문: "절대값 0.009 이내"→"절대값 0.010 미만" 또는 "≈0.009".
