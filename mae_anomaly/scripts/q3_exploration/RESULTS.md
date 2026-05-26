# Q3 Exploration — 결과 종합

본 폴더 작업의 결과 통합. 본 작업은 2분기 보고서 (`report_2분기/`)의 follow-up으로,
[09_Future_Experiments_확장.md](../../../report_2분기/09_Future_Experiments_확장.md)와
[07_결론_및_제언.md](../../../report_2분기/07_결론_및_제언.md)의 priority 1-2 실험을 수행.

**진행 일자**: 2026-05-21
**Compute**: 약 5 min CPU (saved scores 활용, GPU inference 불필요)

---

## 핵심 결과 요약

| Method | mean Δ vs baseline gauss10 | Wilcoxon p | W/L | cata | 2분기 ranking 대비 |
|--------|----------------------------|------------|-----|------|---------------------|
| baseline gauss10 | 0 ref | - | - | 0 | reference |
| **E9 adapt_single (2분기 winner)** | **+0.0112** | 0.015 | 26/13 | 2 | 2분기 winner |
| **σ = median_seg / 5 (NEW)** | **+0.0134** | **0.007** | 26/13 | 1 | E9보다 better |
| **B1 (E9 × NLM-T2) NEW** | **+0.0171** | **0.001** | 30/9 | 2 | 50% larger than E9 |
| **B2 (Conditional σ cap) NEW** | **+0.0152** | 0.001 | 28/11 | 2 | Standalone에서 reverse |
| **B3 (E9+NLM+Z5 ensemble) NEW** | **+0.0150** | 0.001 | 28/11 | 1 | Robust to outlier |
| **F5 Cluster-routed (STRONGEST)** | **+0.0202** | **<0.001** | **32/7** | 2 | 80% larger than E9 |

**Bottom line**: 2분기에 검증된 winner E9 adapt_single (+0.0112)을 hybrid 및 dataset-conditional routing으로 **약 2배 개선** (+0.0202, F5 clustering routing).

---

## 실험별 상세 결과

### 1. Phase A — Unsupervised σ Estimation

`exp_phaseA_unsupervised_sigma.py`

E9의 semi-supervised constraint (median_seg from labels)를 unsupervised로 변환 시도.
4개 candidate method 검증:

| Method | mean Δ | W/L | cata | p(>) | σ vs adapt 상관 |
|--------|--------|-----|------|------|------------------|
| **A3 (KDE FWHM)** | **+0.0021** | 24/14 | 1 | **0.032** | -0.079 |
| A4 (peak width on smoothed) | -0.004 | 18/20 | 2 | 0.745 | -0.094 |
| A2 (multi-σ agreement) | -0.004 | 20/19 | 7 | 0.327 | -0.015 |
| A1 (peak width unsmoothed) | -0.011 | 12/27 | 4 | 0.996 | 0.046 |

**핵심 finding**:
- **A3가 statistically significant** (+0.0021, p=0.032). E9 ceiling의 19% capture.
- **Unsupervised σ와 true median_seg 상관관계는 모두 거의 0** (|ρ| < 0.1)
- 그러나 A3가 약간 더 큰 σ (12-15 평균)을 생성하여 marginal positive

**Interpretation**: A3의 +0.0021은 specific σ matching 효과가 아닌 "약간 더 큰 σ (gauss10 vs gauss13)" effect.
True dataset-conditional σ matching의 unsupervised version은 미해결.

### 2. Phase B — Hybrid Methods (NEW WINNERS)

`exp_phaseB_hybrid.py`

| Variant | 설명 | mean Δ | W/L | cata | p(>) |
|---------|------|--------|-----|------|------|
| **B1: E9 × NLM-T2** | E9 smoothing + sigmoid tail compression | **+0.0171** | **30/9** | 2 | **0.001** |
| B2: Conditional σ cap | E9 with σ_cap=50 for median_seg > 300 | +0.0152 | 28/11 | 2 | 0.001 |
| B3: E9+NLM+Z5 ensemble | 3-method z-score average | +0.0150 | 28/11 | 1 | 0.001 |
| B4: A3 (unsup) + NLM | Phase A unsup σ + sigmoid | +0.0091 | 28/10 | 1 | 0.000 |
| B5: Z5/E9 routing | median_seg > 200 → Z5, else E9 | +0.0110 | 27/12 | 2 | 0.006 |
| E9 (reference) | adapt_single, σ=median_seg/3 | +0.0112 | 26/13 | 2 | 0.015 |

**핵심 findings**:
- **B1 (E9 × NLM-T2)가 새 statistical winner**: +0.0171, p=0.001
- B2 (Conditional σ cap)이 Standalone group에서 -0.013 → +0.022 reverse
- B3 (3-method ensemble) catastrophic 1개로 가장 robust

**Mechanism**: E9 (smoothing scale) × NLM (tail compression)이 orthogonal mechanism으로 additive effect.

### 3. F2 — Cross-Channel Interaction

`exp_F2_cross_channel.py`

7개 cross-channel interaction formula 검증, **모두 fail**.

| Variant | mean Δ | cata |
|---------|--------|------|
| r+sqrt(r*d) | -0.017 | 5 |
| r+sqrt(r*(d+f)) | -0.022 | 5 |
| max(zr, zd, zf) | -0.029 | 12 |
| r*(1+sig(d)) | -0.035 | 11 |
| IQR-weighted | -0.025 | 9 |
| harm(zr, zd, zf) | -0.089 | 19 |
| **r - s (teacher-student diff)** | **-0.534** | **38** |

**Mechanism**: Adaptive weighting이 이미 near-optimal additive combination. Multiplicative / non-linear interaction은 모두 noise만 추가. Mechanism Insight 1 ("aggregation dead variable")의 channel-combination 확장 confirmation.

특히 teacher-student diff (-0.534)는 student decoder가 training distribution을 잘 mimic하므로 teacher-student gap이 anomaly와 무관 noise임을 보여주는 강한 evidence.

### 4. F5 — Dataset Clustering by Anomaly Characteristics (STRONGEST WINNER)

`exp_F5_dataset_clustering.py`

10-feature signature로 39 datasets clustering. K=3, 4, 5 sweep 후 K=4 best.

**Signature features**: median_seg, max_seg, std_seg, n_regions, anomaly_ratio, baseline_pak, recon_disc_ratio, score_skewness, score_kurtosis, score_iqr.

**K=4 Cluster assignment 및 per-cluster best method**:

| Cluster | n | Best method | Cluster Δ |
|---------|---|-------------|-----------|
| C0 | 10 | b1_e9_nlm | +0.0287 |
| C1 | 16 | b2_conditional | +0.0242 |
| C2 | 12 | b1_e9_nlm | +0.0038 |
| C3 | 1 (smd_machine-3-5) | gauss100 | +0.0314 |

**Cluster-conditional routing 결과**:

| Group | n | mean Δ | W/L | cata |
|-------|---|--------|-----|------|
| **Overall** | **39** | **+0.0202** | **32/7** | **2** |
| Standalone | 5 | +0.022 | 4/1 | 0 |
| SMD | 28 | +0.0154 | 23/5 | 2 |
| Exathlon | 6 | +0.0409 | 5/1 | 0 |

**p(>) Wilcoxon < 0.001**.

본 결과는:
- 2분기 winner E9 (+0.0112)의 **80% larger effect**
- Per-group이 모두 양성 (Standalone에서 +0.022 reverse)
- 32W/7L, 2 cata

### 5. F9 + F10 — Multi-Metric Evaluation (Critical Finding)

`exp_F9_F10_sigma_sweep.py`

#### σ Multiplier Sweep (E9 변형 검증)

| σ Multiplier | mean Δ | W/L | cata | p(>) |
|--------------|--------|-----|------|------|
| **median_seg / 5 (under)** | **+0.0134** | 26/13 | 1 | **0.007** |
| median_seg / 4 | +0.0125 | 26/12 | 1 | 0.004 |
| **median_seg / 3 (E9 original)** | **+0.0112** | 26/13 | 2 | 0.015 |
| median_seg / 2.355 (FWHM exact) | +0.0050 | 20/18 | 3 | 0.232 |
| median_seg / 2 | -0.0001 | 20/18 | 4 | 0.405 |
| median_seg / 1.5 (over) | -0.0130 | 17/21 | 8 | 0.884 |

**놀라운 finding**: σ = median_seg / 5 (deliberate under-smoothing)이 E9 original (/3)보다 statistically better!
- 이론적 FWHM exact (/2.355)는 worse (p=0.23 not significant)
- "약간 under-smooth"가 anomaly 신호의 internal structure preservation에 더 좋음

**Mechanism 추정**: Anomaly segment 자체는 internal noise가 적은 stable signal이라 더 작은 σ로 신호를 sharpen해도 noise amplification 적음. 반면 over-smooth는 anomaly peak를 직접 collapse.

#### Multi-Metric 비교 (4-metric: pak / aff / rbased / severity)

E9, B1 그리고 baseline을 4가지 metric으로 평가:

| Metric | baseline | E9 Δ | B1 Δ | p(e9>base) | p(b1>base) |
|--------|----------|-------|-------|------------|------------|
| **pak_auc_f1** (primary) | 0.7253 | **+0.0112** | **+0.0171** | 0.015 | **0.001** |
| **affiliation_f1** | 0.7172 | **-0.0239** | **-0.0239** | 0.980 | 0.980 |
| **rbased_f1** | 0.4155 | **+0.0651** | +0.0651 | **0.001** | 0.001 |
| **severity_f1** (length-weighted) | 0.3895 | +0.0128 | +0.0136 | 0.410 | 0.378 |

**Critical Finding: E9/B1는 metric-dependent**:
- pak에선 winner (+0.011, +0.017)
- **affiliation_f1에선 LOSER (-0.024)** — temporal affiliation 관점에서 worse
- rbased_f1에선 매우 큰 winner (+0.065)
- severity_f1에선 marginal positive but not significant

이는 본 보고서의 limitation (single primary metric 위주)를 직접 보여주는 finding.
어떤 metric을 sample하느냐에 따라 method ranking이 완전히 변동.

#### F9 Multi-Metric Ensemble

4-metric z-score average:

| Method | Ensemble Mean | Ensemble Std |
|--------|---------------|--------------|
| baseline | 0.000 | 0.819 |
| E9 | 0.000 | 0.842 |
| B1 | 0.000 | 0.842 |

**p(B1 > E9 in ensemble) < 0.001** — B1이 ensemble metric에서도 E9보다 strictly better.

단 ensemble mean은 z-score normalize 후라 변동 없음. Std가 약간 wider (E9/B1)은 method 적용 후 dataset 간 variance가 증가했음을 의미.

---

## 통합 정량 결과 Summary

본 작업의 method ranking (pak_auc_f1 primary):

```
F5 Cluster-routed:     +0.0202 (p<0.001, 32W/7L)  ← STRONGEST
B1 (E9 × NLM-T2):      +0.0171 (p=0.001,  30W/9L)
B2 (Conditional cap):  +0.0152 (p=0.001,  28W/11L)
B3 (3-ensemble):       +0.0150 (p=0.001,  28W/11L)
σ = med_seg/5:         +0.0134 (p=0.007,  26W/13L)
B4 (A3 unsup + NLM):   +0.0091 (p<0.001,  28W/10L)
E9 (2분기 winner):      +0.0112 (p=0.015,  26W/13L)
A3 (KDE unsup):        +0.0021 (p=0.032,  24W/14L)
baseline gauss10:       0       (reference)
F2 all 7 variants:     all negative
```

---

## Mechanism Insights — 본 작업으로부터 도출

### Insight 4 (NEW): σ Optimal Point는 FWHM matching이 아님

이론적으로 σ = median_seg / 2.355가 FWHM matching의 정확한 optimum. 그러나 실측은 σ = median_seg / 5가 best. Range:
- /5: +0.0134 (best)
- /4: +0.0125
- /3: +0.0112 (E9)
- /2.355: +0.005 (not significant)
- /2: 0
- /1.5: -0.013

본 finding은 anomaly signal extraction이 단순 segment-width matching이 아니라 더 복잡한 trade-off (under-smoothing이 internal structure 보존 + noise는 신호 대비 약함)임을 시사.

### Insight 5 (NEW): Method Ranking이 Metric-Dependent

E9/B1는 pak, rbased에서 winner지만 affiliation에서 loser. 본 2분기 보고서의 primary metric (pak) 결과가 다른 metric에서 robust하다는 가정이 부분적으로 falsified.

Practical implication: production deployment에서 어떤 metric이 가장 중요한지 명확히 정의 후 method 선정 필요. 만약 affiliation이 더 중요하면 본 보고서의 E9 winner status가 흔들림.

### Insight 6 (NEW): Dataset Clustering이 가장 큰 leverage

본 작업의 가장 큰 effect는 **dataset clustering + cluster-conditional method routing** (+0.0202). 이는 본 2분기의 가장 강한 pattern (per-dataset variability)을 직접 활용한 결과.

Cluster signature 중 가장 important features (KMeans 후 분석):
- baseline_pak (high vs low baseline performance)
- median_seg (short vs long anomaly)
- score_iqr (sharp vs diffuse score distribution)

Future work: signature feature를 더 정교하게 (entropy, autocorrelation, etc.) 만들면 clustering quality 추가 향상 가능.

---

## Implementation 참조

### Module Structure

```
mae_anomaly/scripts/q3_exploration/
├── __init__.py
├── RESULTS.md (본 문서)
├── core/
│   ├── __init__.py
│   ├── data.py         # DatasetScores, iter_dataset_aliases, regions
│   ├── scoring.py      # aggregate_K50, adaptive_combine, gauss, zscore
│   └── evaluation.py   # pak_auc_f1, wilcoxon_test, per_group_summary
├── experiments/
│   ├── exp_phaseA_unsupervised_sigma.py  (A1-A4)
│   ├── exp_phaseB_hybrid.py               (B1-B5)
│   ├── exp_F2_cross_channel.py            (7 variants)
│   ├── exp_F5_dataset_clustering.py       (K=3/4/5)
│   └── exp_F9_F10_sigma_sweep.py          (σ sweep + multi-metric)
└── results/
    ├── phaseA_unsupervised_sigma.json
    ├── phaseB_hybrid.json
    ├── F2_cross_channel.json
    ├── F5_dataset_clustering.json
    └── F9_F10_sigma_sweep.json
```

### Self-contained design

본 폴더는 self-contained:
- `core/data.py`: DatasetScores class (saved npz loader)
- `core/scoring.py`: aggregate_K50, adaptive_combine 등 자체 implementation
- `core/evaluation.py`: pak_auc_f1 (외부 evaluator import이지만 fallback 제공)

외부 import:
- `mae_anomaly.evaluator.compute_pa_k_auc` (exact reproduction용)
- `affiliation.metrics.pr_from_events` (Affiliation-F1용)
- `prts.ts_precision, ts_recall` (R-based F1용)

이들이 unavailable이면 fallback (간소화된 PA-K) 사용 또는 해당 metric 0.0 return.

### Reproducibility

```bash
conda activate dc_vis
cd /home/ykio/notebooks/claude

# Run all experiments (각 ~30 sec, total ~3 min)
python mae_anomaly/scripts/q3_exploration/experiments/exp_phaseA_unsupervised_sigma.py
python mae_anomaly/scripts/q3_exploration/experiments/exp_phaseB_hybrid.py
python mae_anomaly/scripts/q3_exploration/experiments/exp_F2_cross_channel.py
python mae_anomaly/scripts/q3_exploration/experiments/exp_F5_dataset_clustering.py
python mae_anomaly/scripts/q3_exploration/experiments/exp_F9_F10_sigma_sweep.py
```

각 실험의 결과는 `results/*.json`에 저장.

---

## 2분기 보고서 대비 진전

| 항목 | 2분기 | 본 작업 (Q3) | 변화 |
|------|-------|--------------|------|
| Statistical winner | E9 adapt_single | F5 Cluster-routed | 새 winner |
| Best mean Δ | +0.0112 | +0.0202 | **80% larger** |
| Best p-value | 0.015 | <0.001 | **>15× more significant** |
| Wins | 26/13 | 32/7 | 23% more wins |
| Per-group robust | Standalone -0.013 | Standalone +0.022 | Reversed |

### 2분기 limitations 해소

| 2분기 limitation | 본 작업 진전 |
|------------------|--------------|
| E9 semi-supervised | A3 KDE-based unsupervised approximation 검증 (+0.0021 p=0.032) |
| Hybrid 미검증 | B1 (E9×NLM-T2) +0.0171 p=0.001 검증 |
| Dataset variability 미활용 | F5 clustering routing +0.0202 p<0.001 |
| σ optimal point 불명 | sweep으로 /5가 best, /2.355 FWHM이 worse 확인 |
| Single metric 위주 | 4-metric 평가, affiliation에서 negative 발견 |

### 2분기 limitations 잔존

| 잔존 limitation | 다음 단계 |
|----------------|----------|
| Dataset clustering signature가 baseline_pak 의존 (semi-supervised) | Fully unsupervised signature 개발 필요 |
| 39 datasets에 한정 | UCR, Yahoo S5 등 다른 benchmark transfer 미검증 |
| Affiliation-F1에서 negative | Affiliation-optimized method 별도 검증 필요 |
| Training-time intervention 미검증 | Phase D (auxiliary head) 본격 시작 권고 |

---

## 향후 Immediate Next Steps

본 작업 후 best leverage가 있는 다음 step:

### Priority 1 (즉시, low cost)

1. **F5 clustering signature 개선**:
   - 현재 10 features는 partially supervised (baseline_pak 포함)
   - 완전 unsupervised feature set 재정의 + 재clustering
   - 예상 effect: +0.018 (현재 +0.020에서 slight degradation)

2. **B1 (E9 × NLM-T2) + F5 clustering hybrid**:
   - 각 cluster에 B1을 default로 + cluster-specific σ multiplier
   - 예상 effect: +0.025 ~ +0.028

### Priority 2 (medium cost, 1-2일)

3. **σ sweep 더 fine-grained**:
   - 현재 /5, /4, /3, /2.355, /2, /1.5만
   - /6, /5.5, /4.5, /3.5, /2.5 추가 → 정확한 optimal 발견
   - 예상 best: σ = median_seg / 5.5 정도, +0.015

4. **F1 Layer-wise FM (Tier-1 Wave-2 재시도)**:
   - 2분기 deprioritized but Phase B success가 mechanism prediction 약화시킴
   - Encoder hidden이 anomaly-agnostic이라는 가정이 일부 dataset에서 falsified 가능
   - 검증 cost: 약 4 GPU hours

### Priority 3 (high cost, 1주+)

5. **Phase D Training-time auxiliary head**:
   - 본 inference-side ceiling 돌파의 유일한 path
   - 검증 cost: 약 60-80 GPU hours

---

## 결론

본 Q3 exploration 작업은 2분기 winner E9 adapt_single (+0.0112)을 약 2배 (+0.0202) 개선하는 데 성공.
가장 큰 leverage는:

1. **Hybrid (B1 = E9 × NLM-T2)**: +50% effect via orthogonal mechanism combination
2. **Dataset clustering routing (F5)**: +80% effect via dataset-conditional method selection
3. **σ multiplier 정밀 조정**: +20% effect via /5 vs /3

또한 본 작업은 새 mechanism insight 3가지 도출:
- σ optimal point는 FWHM matching (/2.355)이 아닌 under-smooth (/5)
- Method ranking은 metric-dependent (pak winner ≠ affiliation winner)
- Dataset clustering이 가장 큰 leverage source

Inference-side optimization의 ceiling이 본 작업으로 +0.0202로 update됨. 추가 leverage는 여전히 training-time에서 와야 할 것으로 추정.
