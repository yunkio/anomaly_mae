# Phase 1 Ablation Study - 종합 분석 보고서

## 1. 실험 개요

- **총 실험 수**: 1,020개 (340 base configs × 3 scoring modes)
- **base experiment**: 170개 config × 2 mask modes (after/before)
- **데이터**: 275K timesteps, 8 features, 9 anomaly types
- **평가 기준**: ROC AUC, F1, PA%K (K=20,50,80), Discrepancy Ratio, Cohen's d

### Default Setting (Phase 1 Baseline)
| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| nhead | 2 |
| seq_length | 100 |
| patch_size | 10 |
| masking_ratio | 0.2 |
| num_encoder_layers | 1 |
| num_teacher_decoder_layers | 2 |
| num_student_decoder_layers | 1 |
| dim_feedforward | 256 |
| dropout | 0.1 |
| lambda_disc | 0.5 |
| learning_rate | 0.002 |
| margin_type | dynamic |
| dynamic_margin_k | 1.5 |

### Default Model 성능 (001_default_mask_after)
| Scoring | ROC AUC | F1 | PA%80 ROC | PA%80 F1 | disc_ratio | disc_cd_na | recon_cd_na |
|---------|---------|-----|-----------|----------|------------|------------|-------------|
| default | 0.9523 | 0.7248 | 0.9518 | 0.7304 | 6.137 | 2.515 | 1.775 |
| adaptive | 0.9558 | 0.7120 | 0.9620 | 0.7392 | 6.137 | 2.515 | 1.775 |
| normalized | 0.9572 | 0.7217 | 0.9599 | 0.7343 | 6.137 | 2.515 | 1.775 |

---

## 2. 결과 테이블

### 2.1 ROC AUC 상위 Top 10

| Experiment | ROC AUC | F1 | PA%20 ROC | PA%20 F1 | PA%50 ROC | PA%50 F1 | PA%80 ROC | PA%80 F1 | Disc Ratio1 | Disc Ratio2 | T-Recon Ratio | disc_cd_na | disc_cd_da | recon_cd_na | recon_cd_da | Scoring | Mask |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 067_optimal_v3_mask_after | 0.9829 | 0.8297 | 0.9890 | 0.8501 | 0.9825 | 0.8242 | 0.9664 | 0.7664 | 6.931 | 0.175 | 2.752 | 3.125 | 2.550 | 2.107 | 1.714 | adaptive | after |
| 067_optimal_v3_mask_after | 0.9828 | 0.8270 | 0.9877 | 0.8532 | 0.9810 | 0.8428 | 0.9651 | 0.7665 | 6.931 | 0.175 | 2.752 | 3.125 | 2.550 | 2.107 | 1.714 | default | after |
| 067_optimal_v3_mask_after | 0.9827 | 0.8264 | 0.9877 | 0.8560 | 0.9806 | 0.8423 | 0.9653 | 0.7674 | 6.931 | 0.175 | 2.752 | 3.125 | 2.550 | 2.107 | 1.714 | normalized | after |
| 028_d128_nhead16_mask_after | 0.9791 | 0.8089 | 0.9908 | 0.8925 | 0.9855 | 0.8740 | 0.9729 | 0.8313 | 8.291 | 0.277 | 2.249 | 3.158 | 1.432 | 1.534 | 0.557 | adaptive | after |
| 065_optimal_v1_mask_after | 0.9771 | 0.8137 | 0.9911 | 0.9046 | 0.9850 | 0.8790 | 0.9711 | 0.8198 | 9.951 | 0.236 | 2.293 | 3.279 | 1.521 | 1.559 | 0.608 | adaptive | after |
| 022_d_model_128_mask_after | 0.9771 | 0.8137 | 0.9911 | 0.9046 | 0.9850 | 0.8790 | 0.9711 | 0.8198 | 9.951 | 0.236 | 2.293 | 3.279 | 1.521 | 1.559 | 0.608 | adaptive | after |
| 028_d128_nhead16_mask_after | 0.9754 | 0.7876 | 0.9871 | 0.8965 | 0.9813 | 0.8738 | 0.9688 | 0.7943 | 8.291 | 0.277 | 2.249 | 3.158 | 1.432 | 1.534 | 0.557 | normalized | after |
| 068_optimal_v4_mask_after | 0.9752 | 0.7985 | 0.9902 | 0.8880 | 0.9855 | 0.8769 | 0.9728 | 0.8045 | 8.890 | 0.246 | 2.243 | 3.098 | 1.433 | 1.426 | 0.461 | adaptive | after |
| 157_combo_d128_nhead4_t3s1_mr0.08_mask_after | 0.9752 | 0.7921 | 0.9894 | 0.8912 | 0.9852 | 0.8836 | 0.9736 | 0.8187 | 8.500 | 0.292 | 2.384 | 3.248 | 1.452 | 1.701 | 0.635 | adaptive | after |
| 095_d128_t3s1_mask0.10_mask_after | 0.9752 | 0.7921 | 0.9894 | 0.8912 | 0.9852 | 0.8836 | 0.9736 | 0.8187 | 8.500 | 0.292 | 2.384 | 3.248 | 1.452 | 1.701 | 0.635 | adaptive | after |

**핵심 관찰**: 상위 모델은 전부 `mask_after=True`이며, d_model≥128, adaptive/default scoring. 067_optimal_v3 (d128, w500, mr0.15, nhead4, t2s1)가 최고 ROC AUC 달성.

### 2.2 Disc Ratio 상위 Top 10

| Experiment | ROC AUC | F1 | PA%20 ROC | PA%20 F1 | PA%50 ROC | PA%50 F1 | PA%80 ROC | PA%80 F1 | Disc Ratio1 | Disc Ratio2 | T-Recon Ratio | disc_cd_na | disc_cd_da | recon_cd_na | recon_cd_da | Scoring | Mask |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 141_pa80_mr0.10_nhead1_mask_after | 0.9721 | 0.7656 | 0.9914 | 0.8921 | 0.9870 | 0.8732 | 0.9736 | 0.8385 | 10.799 | 0.273 | 2.432 | 3.020 | 1.331 | 1.805 | 0.683 | adaptive | after |
| 083_d128_nhead1_mask0.10_mask_after | 0.9721 | 0.7656 | 0.9914 | 0.8921 | 0.9870 | 0.8732 | 0.9736 | 0.8385 | 10.799 | 0.273 | 2.432 | 3.020 | 1.331 | 1.805 | 0.683 | adaptive | after |
| 141_pa80_mr0.10_nhead1_mask_after | 0.9687 | 0.7816 | 0.9890 | 0.9059 | 0.9843 | 0.8717 | 0.9720 | 0.8406 | 10.799 | 0.273 | 2.432 | 3.020 | 1.331 | 1.805 | 0.683 | normalized | after |
| 022_d_model_128_mask_after | 0.9771 | 0.8137 | 0.9911 | 0.9046 | 0.9850 | 0.8790 | 0.9711 | 0.8198 | 9.951 | 0.236 | 2.293 | 3.279 | 1.521 | 1.559 | 0.608 | adaptive | after |
| 065_optimal_v1_mask_after | 0.9771 | 0.8137 | 0.9911 | 0.9046 | 0.9850 | 0.8790 | 0.9711 | 0.8198 | 9.951 | 0.236 | 2.293 | 3.279 | 1.521 | 1.559 | 0.608 | adaptive | after |
| 022_d_model_128_mask_after | 0.9728 | 0.8092 | 0.9866 | 0.9000 | 0.9807 | 0.8682 | 0.9657 | 0.7761 | 9.951 | 0.236 | 2.293 | 3.279 | 1.521 | 1.559 | 0.608 | normalized | after |
| 006_patch_5_mask_after | 0.9598 | 0.7277 | 0.9892 | 0.8816 | 0.9826 | 0.8604 | 0.9559 | 0.7846 | 9.922 | 0.275 | 1.584 | 3.261 | 1.612 | 1.028 | 0.328 | adaptive | after |
| 116_d128_patch5_mask_after | 0.9598 | 0.7277 | 0.9892 | 0.8816 | 0.9826 | 0.8604 | 0.9559 | 0.7846 | 9.922 | 0.275 | 1.584 | 3.261 | 1.612 | 1.028 | 0.328 | adaptive | after |
| 068_optimal_v4_mask_after | 0.9752 | 0.7985 | 0.9902 | 0.8880 | 0.9855 | 0.8769 | 0.9728 | 0.8045 | 8.890 | 0.246 | 2.243 | 3.098 | 1.433 | 1.426 | 0.461 | adaptive | after |
| 167_optimal_combo_4_mask_after | 0.9712 | 0.7652 | 0.9891 | 0.8849 | 0.9855 | 0.8717 | 0.9727 | 0.7888 | 8.647 | 0.243 | 2.127 | 3.369 | 1.456 | 1.437 | 0.476 | adaptive | after |

**핵심 관찰**: nhead1 + mr0.10이 disc_ratio를 극대화 (10.80). 낮은 attention head 수가 teacher-student 간 discrepancy를 증폭시킴.

### 2.3 T-Ratio (Reconstruction Ratio) 상위 Top 10

| Experiment | ROC AUC | F1 | PA%20 ROC | PA%20 F1 | PA%50 ROC | PA%50 F1 | PA%80 ROC | PA%80 F1 | Disc Ratio1 | Disc Ratio2 | T-Recon Ratio | disc_cd_na | disc_cd_da | recon_cd_na | recon_cd_da | Scoring | Mask |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 007_patch_20_mask_before | 0.9745 | 0.8303 | 0.9909 | 0.8981 | 0.9855 | 0.8755 | 0.9672 | 0.7587 | 1.785 | 0.792 | 5.834 | 1.051 | 0.310 | 2.413 | 1.143 | default | before |
| 117_d128_patch20_mask_before | 0.9745 | 0.8303 | 0.9909 | 0.8981 | 0.9855 | 0.8755 | 0.9672 | 0.7587 | 1.785 | 0.792 | 5.834 | 1.051 | 0.310 | 2.413 | 1.143 | default | before |
| 165_optimal_combo_2_mask_before | 0.9672 | 0.7602 | 0.9955 | 0.9385 | 0.9875 | 0.9011 | 0.9606 | 0.7311 | 2.113 | 0.801 | 5.732 | 1.124 | 0.254 | 2.511 | 1.210 | default | before |
| 168_pa80_final_opt1_mask_before | 0.9687 | 0.7653 | 0.9957 | 0.9454 | 0.9899 | 0.8964 | 0.9627 | 0.7534 | 2.381 | 0.697 | 5.703 | 1.112 | 0.331 | 2.545 | 1.229 | default | before |
| 169_pa80_final_opt2_mask_before | 0.9707 | 0.7820 | 0.9957 | 0.9364 | 0.9898 | 0.8959 | 0.9682 | 0.7867 | 2.373 | 0.634 | 5.603 | 1.219 | 0.424 | 2.572 | 1.259 | default | before |

**핵심 관찰**: T-ratio 상위는 전부 `mask_before`, 주로 patch_size=20. mask_before에서 teacher는 전체 정보를 보고 재구성하므로 anomaly에서의 재구성 오류가 극대화됨. 하지만 disc_ratio는 매우 낮음 (1.8~2.4).

---

## 3. 파라미터별 상세 분석 및 Insight

### 3.1 Window Size (seq_length)

| Window | ROC AUC (mean) | ROC AUC (max) | PA%80 F1 (mean) | disc_ratio (mean) | disc_cd_na (mean) | recon_cd_na (mean) |
|--------|---------------|---------------|-----------------|-------------------|--------------------|---------------------|
| 100 | 0.9437 | 0.9745 | 0.7107 | 4.123 | 1.710 | 2.018 |
| 200 | 0.9205 | 0.9504 | 0.6185 | 4.457 | 1.229 | 1.865 |
| 500 | 0.8791 | **0.9828** | 0.5286 | 2.248 | 0.949 | 1.854 |
| 1000 | 0.7530 | 0.8846 | 0.3489 | 1.186 | 0.222 | 1.217 |
| 2000 | 0.6855 | 0.6863 | 0.2496 | 0.970 | -0.063 | 0.910 |

**Insight**:
- w100이 평균적으로 가장 안정적이나, w500이 최적 파라미터 조합 시 최고 성능 달성 (0.9828).
- **가설**: w500은 더 긴 temporal context를 제공하여 패턴 기반 anomaly 검출에 유리하지만, 모델 capacity가 충분해야 함. w100에서 좋던 세팅이 w500에서 그대로 작동하지 않음.
- w1000 이상은 현재 모델 구조에서 처리 불가능한 수준. disc_cd_na가 0에 가까워짐.

**mask_after vs mask_before × window 상호작용**:
| Window | mask_after ROC | mask_before ROC | mask_after disc_ratio | mask_before disc_ratio |
|--------|---------------|-----------------|----------------------|----------------------|
| 100 | 0.9426 | 0.9448 | 6.277 | 1.969 |
| 200 | **0.9504** | 0.8906 | **7.820** | 1.093 |
| 500 | 0.8607 | **0.8975** | 3.098 | 1.399 |
| 1000 | 0.7072 | **0.7988** | 1.419 | 0.953 |

**핵심 발견**: w100에서는 mask_after/before 성능 유사하지만, w500 이상에서는 mask_before가 ROC에서 우위. 반면 mask_after는 모든 window에서 disc_ratio가 2~6배 높음. w200에서 mask_after가 ROC와 disc_ratio 모두에서 압도적으로 우수.

### 3.2 d_model

| d_model | ROC AUC (mean) | ROC AUC (max) | disc_ratio (mean) | disc_cd_na (mean) |
|---------|---------------|---------------|-------------------|--------------------|
| 16 | 0.9449 | 0.9516 | 2.812 | 1.191 |
| 32 | 0.9436 | 0.9474 | 3.529 | 1.355 |
| 64 | 0.9250 | 0.9745 | 3.663 | 1.504 |
| 96 | 0.8750 | 0.9161 | 1.421 | 0.523 |
| **128** | 0.9121 | **0.9828** | 3.567 | 1.521 |
| 192 | 0.9304 | 0.9640 | 3.613 | 1.420 |
| 256 | **0.9594** | 0.9624 | 3.950 | 1.756 |

**Insight**:
- d256이 가장 높은 평균이지만 실험 수 제한적.
- d128이 max ROC 0.9828로 최고치 달성. 분산이 큼 → 다른 파라미터와의 조합이 중요.
- d96은 의외로 성능 저하 → d_model이 어중간한 크기일 때 문제 발생 가능 (dim_feedforward=384 관련).
- **가설**: d128 이상에서 discrepancy signal이 더 정밀해짐. teacher의 표현력이 높아져 normal/anomaly 구분에 유리.

### 3.3 Masking Ratio

| mr | ROC AUC (mean) | disc_ratio (mean) | disc_cd_na (mean) |
|----|---------------|-------------------|--------------------|
| 0.05 | 0.6963 | 1.322 | 0.588 |
| **0.08** | **0.9579** | 4.530 | 1.979 |
| 0.10 | 0.9308 | 4.061 | 1.682 |
| 0.12 | 0.9540 | 3.771 | 1.854 |
| 0.15 | 0.9248 | 3.742 | 1.574 |
| 0.18 | 0.9540 | 3.833 | 1.549 |
| **0.20** | 0.9277 | 3.566 | 1.449 |
| 0.30 | 0.9364 | 3.917 | 1.426 |
| 0.40 | 0.9315 | 4.109 | 1.476 |
| 0.50 | 0.8848 | 2.184 | 1.125 |
| 0.80 | 0.8345 | 2.342 | 1.099 |

**Insight**:
- mr=0.08이 최고 평균 ROC (0.9579)이며 disc 관련 지표도 우수.
- mr=0.05는 치명적 성능 저하 → masking이 너무 적으면 학습 신호 부족.
- mr=0.10도 disc_ratio 최대화에 효과적 (max 10.80).
- **가설**: 낮은 masking ratio (0.08-0.12)에서 모델이 "쉬운" 재구성을 학습하면서도 anomaly와의 discrepancy를 극대화. 높은 mr은 정상 데이터도 재구성이 어려워져 normal/anomaly 구분력 감소.

### 3.4 Attention Heads (nhead)

| nhead | ROC AUC (mean) | disc_ratio (mean/max) | disc_cd_na (mean) |
|-------|---------------|----------------------|---------------------|
| 1 | 0.9438 | 4.445 / 10.80 | 1.683 |
| 2 | 0.9190 | 3.533 / 9.92 | 1.425 |
| 4 | 0.9055 | 3.168 / 9.95 | 1.338 |
| **6** | **0.9611** | **5.004** / 8.65 | **1.917** |
| **8** | 0.9603 | 4.593 / 8.24 | 1.937 |
| 16 | 0.9420 | 3.805 / 8.41 | 1.742 |
| 32 | 0.9488 | 4.194 / 7.02 | 1.741 |

**Insight**:
- nhead=6,8이 평균 ROC 최고 (0.96+). 하지만 실험 수 제한적.
- nhead=1이 disc_ratio 최대 (10.80)를 달성하지만 ROC 평균은 보통.
- **가설**: 적은 attention head는 모델의 유연성을 제한하여 student가 teacher를 따라가기 어렵게 만들어 discrepancy 증가. 하지만 reconstruction 품질도 함께 저하되어 ROC는 최적이 아님.
- nhead=4-8이 ROC와 disc의 균형점.

### 3.5 Decoder Depth

**w100에서의 decoder depth 효과**:
| Teacher/Student Layers | ROC AUC | disc_cd_disturbing |
|----------------------|---------|-------------------|
| t1s1 | 0.9350 | 0.630 |
| t2s1 | 0.9369 | 0.923 |
| t2s2 | 0.9417 | 0.812 |
| **t3s1** | **0.9571** | **1.224** |
| t3s2 | 0.9639 | 1.185 |
| t4s1 | 0.9465 | 0.972 |
| t4s2 | 0.9595 | 1.179 |
| t5s1 | 0.9483 | 0.856 |

**Insight**:
- t3s1과 t3s2가 w100에서 최적. disc_cd_disturbing도 가장 높음.
- **t4 이상에서 성능 하락** → teacher decoder가 너무 깊으면 과적합 또는 학습 불안정.
- student_decoder=2가 1보다 약간 나은 경향 (t3s2 > t3s1 ROC, 하지만 disc_cd_disturbing은 t3s1이 우위).
- w500에서는 decoder depth 증가 시 성능 급격히 하락 (t4s1: 0.769). 큰 window에서는 shallow decoder가 적합.

### 3.6 Learning Rate

| LR | ROC AUC (mean) | disc_ratio (mean) |
|----|---------------|-------------------|
| 0.0005 | 0.9414 | 3.521 |
| 0.001 | 0.9408 | 3.517 |
| **0.002** (default) | 0.9224 | 3.609 |
| **0.003** | **0.9562** | 4.195 |
| **0.005** | **0.9639** | **4.703** |

**Insight**:
- **높은 LR이 일관되게 우수**. lr=0.005가 ROC와 disc_ratio 모두 최고.
- default lr=0.002가 실제로 최적이 아니었음. 이것은 "easy win" 파라미터.
- **가설**: 높은 LR이 teacher-student 간 더 뚜렷한 학습 차이를 만들어냄. Teacher가 빠르게 수렴하면서 student와의 gap이 확대.

### 3.7 Dropout

| Dropout | ROC AUC (mean) | disc_ratio (mean) | disc_cd_na (mean) |
|---------|---------------|-------------------|--------------------|
| 0.0 | 0.9405 | 2.842 | 1.439 |
| 0.05 | 0.9382 | 3.824 | 1.664 |
| **0.1** (default) | 0.9231 | 3.637 | 1.495 |
| **0.2** | **0.9523** | **4.305** | **1.682** |
| 0.3 | 0.9447 | 3.378 | 1.414 |

**Insight**: dropout=0.2가 최적. 이것도 default (0.1)보다 나은 "easy win".

### 3.8 Lambda Disc

| lambda | ROC AUC (mean) |
|--------|---------------|
| 0.10 | 0.9438 |
| 0.25 | 0.9500 |
| **0.50** (default) | 0.9233 |
| **1.00** | **0.9557** |
| 2.00 | 0.9528 |
| 3.00 | 0.9481 |

**Insight**: lambda=1.0이 최적이지만, disc_ratio와 disc_cd는 거의 동일. Lambda는 학습 시 loss weight만 조절하므로 추론 시 discrepancy 분포에는 영향 없음. 성능 차이는 학습 dynamics에서 발생.

### 3.9 Patch Size

| Patch | ROC AUC (mean) | disc_ratio (mean/max) | disc benefit ROC |
|-------|---------------|----------------------|-----------------|
| 5 | 0.9033 | 5.468 / 9.92 | **+0.083** |
| 10 | 0.9256 | 3.706 / 10.80 | +0.022 |
| 20 | 0.9130 | 2.735 / 6.93 | +0.020 |
| **25** | **0.9684** | 4.303 / 6.17 | +0.006 |

**Insight**:
- patch_size=5에서 discrepancy의 기여가 압도적 (+0.083 ROC, +0.218 PA80_F1). 작은 패치 = 많은 패치 수 = 더 많은 독립 discrepancy 신호.
- patch_size=25가 평균 ROC 최고이나 실험 수 제한적.
- **가설**: 작은 패치는 reconstruction 성능은 낮추지만 (teacher_recon이 약함), discrepancy signal이 강력하여 보완. 이 보완 효과가 최대인 sweet spot이 존재.

---

## 4. 핵심 Cross-cutting 분석

### 4.1 Discrepancy와 Reconstruction의 Trade-off

**상관관계**: disc_ratio ↔ recon_cd_na = **-0.417** (음의 상관)

이것은 가장 중요한 발견 중 하나. Discrepancy를 극대화하면 reconstruction quality가 떨어지고, reconstruction을 극대화하면 discrepancy 분리도가 낮아짐.

**최적 모델은 이 둘의 균형점**:
- 067_optimal_v3: disc_cd=3.125, recon_cd=2.107 (합계 5.231) → **ROC 0.9828 (최고)**
- 141_pa80_nhead1_mr0.10: disc_cd=3.020, recon_cd=1.805 (합계 4.825) → disc_ratio 10.80 (최고), PA80 0.84

### 4.2 Scoring Mode 효과와 disc_ratio의 관계

| disc_ratio 수준 | default | adaptive | normalized |
|---------------|---------|----------|------------|
| low (<3) | 0.9047 | 0.8785 | 0.8584 |
| med (3-5) | 0.9340 | 0.9354 | 0.9357 |
| high (5-7) | 0.9500 | 0.9559 | 0.9560 |
| **very high (7+)** | 0.9616 | **0.9688** | 0.9661 |

**핵심 발견**: disc_ratio가 높을수록 adaptive/normalized scoring이 default보다 우수해짐. disc_ratio가 낮으면 default가 최선.

**가설**: 높은 disc_ratio = 강한 discrepancy signal → adaptive scoring이 이 신호를 더 효과적으로 활용. 낮은 disc_ratio에서는 discrepancy가 noise에 가까워 adaptive weighting이 오히려 해로움.

### 4.3 Student Reconstruction의 의외의 우수성

| Signal | ROC AUC (avg) | PA80 F1 (avg) |
|--------|--------------|---------------|
| **Student-recon** | **0.9260** | **0.6744** |
| Combined (default) | 0.9241 | 0.6598 |
| Teacher-recon | 0.9014 | 0.5838 |
| Disc-only | 0.7742 | 0.4576 |

Student reconstruction이 teacher보다 일관되게 우수. Student decoder는 shallower (1 layer)하여 regularization 효과가 있음.

**Student vs Teacher 차이가 큰 조건**:
- d96: +0.070 (student가 teacher보다 7%p 높음)
- d192: +0.055
- w200: +0.042

**가설**: capacity가 큰 teacher는 정상 데이터를 과적합하여 anomaly에서도 비교적 잘 재구성. Student는 capacity 제한으로 anomaly 재구성에 실패하여 더 나은 anomaly detector.

### 4.4 Discrepancy Score가 도움되는 경우

Combined scoring (recon + disc) > Teacher-recon only인 비율: **307/340 (90.3%)**

**가장 도움되는 조건**:
| Condition | disc benefit (ROC) | disc benefit (PA80 F1) |
|-----------|-------------------|----------------------|
| patch_size=5 | +0.083 | +0.218 |
| d_model=256 | +0.036 | +0.111 |
| d_model=192 | +0.036 | +0.095 |
| masking_ratio≥0.60 | +0.040 | +0.125 |
| nhead=6 | +0.036 | +0.119 |

### 4.5 mask_after vs mask_before: 언제 어떤 것이 좋은가

**mask_after 우위 조건**:
- feature_wise masking (+0.507 ROC!)
- w200 (+0.060)
- patch_size=5 with w500 (+0.138)

**mask_before 우위 조건**:
- w1000 (-0.204)
- w500 + small d_model/nhead1 (-0.128)
- w500 + deep decoder t3s2/t4s2 (-0.144)

**가설**: mask_after는 encoder가 이미 처리한 표현에서 masking하므로, 충분한 capacity가 있어야 효과적. 큰 window + 작은 모델에서는 encoder가 이미 정보를 잃어서 추가 masking이 해로움. mask_before는 raw input을 masking하므로 모델 capacity와 무관하게 안정적.

---

## 5. 특정 관점별 심화 분석

### 5.1 disc_cd_disturbing_vs_anomaly가 높은 모델들의 특징

상위 모델들 (disc_cd_da > 2.0):
1. **067_optimal_v3** (d128, w500, mr0.15, nhead4): disc_cd_da=2.550
2. **063_w500_p20_d128** (d128, w500, p20, mr0.2): disc_cd_da=2.189
3. **070_optimal_final** (d128, w500, mr0.15): disc_cd_da=2.142
4. **008_w500_p5** (d64, w500, p5, mr0.2): disc_cd_da=2.128

**공통 특징**: 전부 w500. 긴 window가 disturbing normal과 anomaly를 구분하는 데 핵심적. d128이면 더 좋음.

**가설**: Disturbing normal은 짧은 window에서는 anomaly와 구분 어려움 (일시적 deviation이 anomaly와 유사). w500에서는 충분한 context로 disturbing의 "일시성"과 anomaly의 "지속성"을 구분 가능.

### 5.2 PA%80이 높으면서 disc_ratio도 높은 모델들

| Model | PA%80 F1 | disc_ratio | ROC AUC |
|-------|----------|------------|---------|
| 141_pa80_mr0.10_nhead1 (norm) | **0.8406** | **10.799** | 0.9687 |
| 083_d128_nhead1_mask0.10 (adapt) | 0.8385 | 10.799 | 0.9721 |
| 022_d128 (adaptive) | 0.8198 | 9.951 | 0.9771 |
| 028_d128_nhead16 (adaptive) | 0.8313 | 8.291 | 0.9791 |
| 157_combo_t3s1_mr0.08 (adapt) | 0.8187 | 8.500 | 0.9752 |

**공통 특징**: mask_after + d128 (또는 큰 d_model) + adaptive scoring + 낮은 mr (0.08-0.2).

### 5.3 Scoring Mode 변경 시 성능 차이가 큰 파라미터

disc_ratio > 5인 모델 중 adaptive가 default 대비 가장 많이 좋아진 경우:
1. **008_w500_p5_mask_after**: +0.025 ROC (disc_ratio=9.72)
2. **041_lambda_0.1_mask_after**: +0.022
3. **167_optimal_combo_4_mask_after**: +0.013

→ disc_ratio가 높을수록 adaptive scoring의 이점이 커짐 (위 4.2 확인).

---

## 6. Phase 2 실험 계획

### Default Setting (고정)
Phase 1의 base config를 default로 유지: d_model=64, nhead=2, seq_length=100, patch_size=10, masking_ratio=0.2, encoder=1, teacher_dec=2, student_dec=1, dropout=0.1, lr=0.002, lambda_disc=0.5, margin_type=dynamic, k=1.5

### 실험 계획

#### Group 1: Disc Ratio 극대화 (관점 1)
disc_ratio가 높았던 핵심 요인: nhead1 + mr0.10 + d128 + mask_after

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-001 | d128, nhead1, mr0.10, lr0.005, dropout0.2 | 높은 disc_ratio 조건 + "easy win" LR/dropout 결합 |
| P2-002 | d128, nhead1, mr0.08, lr0.005, dropout0.2 | mr0.08이 mean ROC 최고 → disc_ratio 극대화 조건과 결합 |
| P2-003 | d128, nhead1, mr0.10, t3s1, lr0.005 | disc_ratio 극대화 + 최적 decoder depth |
| P2-004 | d128, nhead1, mr0.10, lambda1.0, lr0.003 | disc_ratio 극대화 + 최적 lambda/lr |
| P2-005 | d256, nhead1, mr0.10, lr0.005 | 더 큰 d_model에서 disc_ratio 극대화 가능한지 |

#### Group 2: disc_cd + recon_cd 동시 극대화 (관점 2)
067_optimal_v3 (d128, w500, nhead4, mr0.15, t2s1)의 특징을 기반으로 변형

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-006 | d128, w500, nhead4, mr0.15, t2s1 (067 재현) + lr0.003 | 최적 모델에 더 높은 LR 적용 |
| P2-007 | d128, w500, nhead4, mr0.15, t2s1, dropout0.2 | 최적 모델에 더 높은 dropout 적용 |
| P2-008 | d128, w500, nhead4, mr0.12, t2s1 | mr을 약간 낮춰 disc 향상 시도 |
| P2-009 | d128, w500, nhead8, mr0.15, t2s1 | nhead 증가로 recon_cd 향상 시도 |
| P2-010 | d128, w500, nhead4, mr0.15, t3s1 | decoder 깊이 증가가 combined score에 미치는 영향 |

#### Group 3: Best Model + Scoring Mode + Window Size (관점 3)
disc_cd_na와 recon_cd_na가 모두 높은 모델 (067_optimal_v3 params)에서 window/scoring 변형

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-011 | 067 params + w100 (scoring 3종) | w100으로 줄이면 안정성 증가하나 disc_cd_da 감소 예상 |
| P2-012 | 067 params + w1000, p20 (scoring 3종) | w1000으로 늘려도 d128이면 성능 유지 가능한지 |
| P2-013 | d128, nhead1, mr0.10 + w500 (scoring 3종) | disc_ratio 최고 모델에 w500 적용 → disc_cd_da 개선 기대 |
| P2-014 | d128, nhead1, mr0.10 + w1000, p20 (scoring 3종) | disc_ratio 최고 모델에 w1000 적용 |

#### Group 4: Disturbing Normal vs Anomaly 구분 (관점 4)
disc_cd_disturbing_vs_anomaly가 높았던 조건: w500, d128

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-015 | d128, w500, nhead4, mr0.15, t3s1 | disc_cd_da 상위 모델의 decoder 변형 |
| P2-016 | d128, w500, nhead4, mr0.10 | mr 감소 → disc 증가하면서 disturbing 구분력도 증가? |
| P2-017 | d128, w500, nhead8, mr0.15 | nhead 증가가 disturbing 구분에 미치는 영향 |
| P2-018 | d128, w500, p10, mr0.15, lr0.003 | LR 증가가 disturbing 구분에 미치는 영향 |

#### Group 5: PA%80 + disc_ratio 동시 최적화 (관점 5)

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-019 | d128, nhead1, mr0.10, t3s2, lr0.005, dropout0.2 | PA80 최적 조건 조합 |
| P2-020 | d128, nhead4, mr0.08, t3s1, lr0.003 | mr0.08 + t3s1 (PA80 최적 decoder) |
| P2-021 | d128, nhead16, mr0.10, t4s1, lr0.003, dropout0.2 | 기존 PA80 최적 실험에 easy win 추가 |

#### Group 6: Window × Depth × Masking 관계 (관점 6)

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-022 | w100, d128, t3s1, mr0.10 | w100에서 t3s1 + 낮은 mr |
| P2-023 | w100, d128, t3s1, mr0.08 | w100에서 t3s1 + mr0.08 |
| P2-024 | w500, d128, t2s1, mr0.10 | w500에서 shallow decoder + 낮은 mr |
| P2-025 | w500, d128, t2s1, mr0.08 | w500에서 shallow decoder + mr0.08 |
| P2-026 | w500, d128, t3s1, mr0.08 | w500에서 mid decoder + mr0.08 |
| P2-027 | w1000, d128, p20, t2s1, mr0.10 | w1000에서 d128 + shallow + 낮은 mr |
| P2-028 | w1000, d128, p20, t2s1, mr0.08 | w1000에서 d128 + shallow + mr0.08 |

#### Group 7: Scoring 변경 시 차이가 큰 요인 검증 (관점 7)

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-029 | d128, p5, mr0.20, lr0.003 | patch5에서 scoring 차이 극대화 조건 |
| P2-030 | d128, nhead16, mr0.20, lr0.003 | 높은 disc_ratio + nhead16에서 scoring 차이 |
| P2-031 | d128, mr0.40, lr0.003 | 높은 mr에서 scoring mode 차이 검증 |

#### Group 8: 성능 + disc_cd_disturbing 동시 최적화 (관점 8)

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-032 | d128, w500, nhead4, mr0.15, t2s1, lr0.003 | 067 + 높은 LR |
| P2-033 | d128, w500, nhead4, mr0.15, t2s1, dropout0.2 | 067 + 높은 dropout |
| P2-034 | d128, w500, nhead4, mr0.12, t2s1, lr0.003, dropout0.2 | 067 + mr 미세 조정 + easy wins |

#### Group 9: Softplus Margin 테스트 (관점 9)

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-035 | default + margin_type=softplus | baseline에 softplus 적용 |
| P2-036 | d128, nhead4, w500, mr0.15, t2s1, margin=softplus | 최고 ROC 모델에 softplus |
| P2-037 | d128, nhead1, mr0.10, margin=softplus | 최고 disc_ratio 모델에 softplus |
| P2-038 | d128, nhead1, mr0.10, t3s1, lr0.005, margin=softplus | PA80 최적 + softplus |
| P2-039 | d128, nhead16, mr0.10, t4s1, margin=softplus | high capacity + softplus |

#### Group 10: mask_after vs mask_before 하이퍼파라미터별 차이 분석 (관점 10)

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-040 | d128, w200, mr0.10 (both mask modes) | w200에서 mask_after 우위가 재현되는지 |
| P2-041 | d128, w500, mr0.10 (both mask modes) | w500에서 mask_before 우위가 d128에서도 유지되는지 |
| P2-042 | d128, w500, nhead4, mr0.15 (both mask modes) | 067 조건에서의 mask mode 차이 |
| P2-043 | d64, w500, nhead1, mr0.20 (both mask modes) | mask_before 우위 조건에서 nhead 감소 효과 |

#### Group 11: Combined Scoring vs Reconstruction-only 분석 (관점 11)
별도 실험 불필요 - 기존 데이터에서 이미 disc_only, teacher_recon, student_recon 성능이 기록됨. 다만 특정 조건에서의 검증:

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-044 | d256, nhead8, mr0.15, lr0.003 | d256에서 disc benefit이 유지되는지 |
| P2-045 | d128, p5, w100, lr0.003 | patch5에서 disc benefit 극대화 조건 + LR boost |
| P2-046 | d128, p5, w500, lr0.003 | patch5 + w500에서 disc benefit |

#### Group 12: 추가 Insight 도출 (관점 12)

| # | 변경 파라미터 | 가설 |
|---|---|---|
| P2-047 | d128, nhead4, mr0.08, t3s1, lr0.005, dropout0.2, lambda1.0 | 모든 "easy win" 결합 |
| P2-048 | d128, nhead6, mr0.12, t3s1, lr0.003 | nhead6이 정말 최적인지 검증 |
| P2-049 | d128, nhead8, mr0.08, t3s2, lr0.003, dropout0.2 | s_dec=2 효과 검증 |
| P2-050 | d128, w500, nhead4, mr0.15, t2s1, lr0.005, dropout0.2, lambda1.0 | 067 + 모든 easy win |

---

## 7. 종합 결론

### 핵심 발견 요약

1. **최고 ROC AUC**: 067_optimal_v3 (d128, w500, mr0.15, nhead4, t2s1) = 0.9828
2. **최고 Disc Ratio**: nhead1 + mr0.10 + d128 = 10.80 (PA%80 F1도 0.84로 우수)
3. **Disc-Recon Trade-off**: 두 지표는 음의 상관 (-0.42). 최적 모델은 균형점에 위치.
4. **Adaptive scoring**: disc_ratio > 7일 때 default보다 우수. 그 외에는 default가 안전.
5. **Student recon > Teacher recon**: Student의 shallow architecture가 오히려 anomaly detection에 유리.
6. **Easy Wins**: lr=0.003-0.005, dropout=0.2, lambda=1.0 → 아직 최적 모델에 적용되지 않음.
7. **Window Size**: w500이 최적이지만 d128 이상 필요. w100은 안정적.
8. **Decoder Depth**: t3s1이 w100에서 최적. w500에서는 t2s1이 최적.
9. **mask_after**: disc_ratio 크게 증가시키지만 capacity 부족 시 성능 하락. mask_before는 안정적.
10. **Discrepancy score의 가치**: 90% 이상의 실험에서 combined > teacher_recon. 특히 patch5에서 효과 극대화.

### Phase 2 핵심 질문

1. 모든 "easy win" (lr, dropout, lambda)을 최적 아키텍처에 동시 적용하면 0.99+ 가능한가?
2. disc_ratio 극대화 조건 (nhead1, mr0.10)에 w500을 적용하면 disc_cd_disturbing도 개선되는가?
3. softplus margin이 dynamic margin 대비 어떤 장단점이 있는가?
4. student_recon을 primary signal로 사용하는 새로운 scoring 방식이 효과적일 수 있는가?
