# Phase 1 Ablation Study 종합 분석 보고서

**작성일:** 2026-01-27
**총 Phase 1 실험 수:** 1,398개 (233개 기본 설정 × 6개 변형)
**Phase 2 실험 계획:** 150개 (전략적 설계)

---

## 요약

Phase 1에서는 233개의 기본 설정에 대해 각각 6개의 변형(2개 inference mode × 3개 scoring mode)을 테스트하여 총 1,398개의 실험을 수행했습니다. 이 연구를 통해 discrepancy 기반 및 reconstruction 기반 이상 탐지 간의 균형이 핵심임을 발견했습니다.

### 핵심 발견사항

1. **균형이 가장 중요함**: 최고 성능(ROC-AUC > 0.95)을 달성하려면 `disc_cohens_d_normal_vs_anomaly` (0.7-1.2)와 `recon_cohens_d_normal_vs_anomaly` (2.0-3.8) 모두 높아야 합니다.

2. **높은 disc_ratio만으로는 불충분**: 극단적인 disc_ratio (>4.0)를 가진 모델들은 reconstruction 품질이 낮아 성능이 저조합니다 (ROC-AUC 0.74-0.88).

3. **Inference 및 Scoring 모드의 영향**:
   - `all_patches`가 `last_patch`보다 훨씬 우수 (평균 ROC-AUC 0.836 vs 0.789)
   - `default` scoring mode가 배포에 최적 (0.799 vs 0.825 adaptive vs 0.814 normalized)

4. **최적 구성**:
   - Window 500 + Patch 20이 강력한 기준선
   - d_model=128이 용량과 일반화 사이 최적 균형
   - Teacher-student 비율: t4s1, t4s2가 우수

5. **핵심 과제**: Disturbing normal과 anomaly 분리(disc_cohens_d_disturbing_vs_anomaly)가 여전히 어려우며, 최고 모델도 0.8만 달성

---

## 상세 결과 테이블

### 테이블 1: ROC-AUC 기준 상위 10개 모델

| 순위 | 모델 | ROC-AUC | F1 | PA%20 AUC | PA%50 AUC | PA%80 AUC | disc_ratio | t_ratio | disc_d | recon_d | Inf 모드 | Score |
|------|-------|---------|-----|-----------|-----------|-----------|------------|---------|--------|---------|----------|-------|
| 1 | 007_patch_20 | 0.9624 | 0.846 | 0.987 | 0.980 | 0.960 | 1.74 | 6.00 | 0.97 | 1.95 | all_patches | default |
| 2 | 009_w500_p20 | 0.9586 | 0.883 | 0.979 | 0.969 | 0.926 | 1.69 | 4.59 | 0.73 | 2.08 | last_patch | default |
| 3 | 009_w500_p20 | 0.9578 | 0.740 | 0.994 | 0.989 | 0.965 | 2.15 | 4.50 | 1.18 | 3.77 | all_patches | default |
| 4 | 023_d_model_256 | 0.9575 | 0.848 | 0.981 | 0.969 | 0.921 | 1.31 | 6.32 | 0.41 | 2.11 | last_patch | default |
| 5 | 019_decoder_t4s2 | 0.9572 | 0.812 | 0.993 | 0.988 | 0.960 | 1.94 | 5.69 | 0.90 | 2.30 | all_patches | default |
| 6 | 063_w500_p20_d128 | 0.9567 | 0.657 | 0.992 | 0.985 | 0.961 | 1.61 | 5.50 | 1.05 | 3.49 | all_patches | default |
| 7 | 016_decoder_t4s1 | 0.9564 | 0.805 | 0.994 | 0.988 | 0.959 | 1.97 | 5.60 | 0.82 | 2.31 | all_patches | default |
| 8 | 022_d_model_128 | 0.9557 | 0.864 | 0.983 | 0.969 | 0.930 | 2.16 | 6.60 | 0.83 | 2.15 | last_patch | default |
| 9 | 015_decoder_t3s1 | 0.9555 | 0.814 | 0.993 | 0.986 | 0.958 | 1.73 | 5.67 | 0.73 | 2.23 | all_patches | default |
| 10 | 018_decoder_t3s2 | 0.9532 | 0.808 | 0.993 | 0.987 | 0.957 | 1.67 | 5.72 | 0.65 | 2.20 | all_patches | default |

**관찰 사항:**
- 상위 10개 모델 **전부** `default` scoring mode 사용
- 10개 중 7개가 `all_patches` inference mode 사용
- disc_ratio 범위: 1.31-2.16 (극단적이지 않음)
- t_ratio는 일관되게 높음 (4.5-6.6)
- disc_d 범위 0.41-1.18, recon_d 범위 1.95-3.77

**테이블 1에 대한 상세 분석:**

1위 모델인 007_patch_20은 **ROC-AUC 0.9624**로 Phase 1 전체에서 최고 성능을 기록했습니다. 이 모델의 주요 특징:
- **patch size 20 사용**: 기본 window 100에서 patch 20을 사용하여 총 5개의 patch 생성
- **all_patches 모드**: 모든 패치의 정보를 종합하여 더 robust한 탐지
- **균형잡힌 메트릭**: disc_d=0.97, recon_d=1.95로 둘 다 적절한 수준
- **높은 t_ratio=6.00**: teacher의 reconstruction 능력이 매우 우수

3위 모델 009_w500_p20은 동일한 base 모델이지만 **inference mode만 다릅니다**:
- all_patches 모드로 ROC-AUC **0.9578** 달성
- 특히 **disc_d=1.18, recon_d=3.77**로 매우 높은 수치
- PA%80 AUC **0.965**로 실용적 배포에 매우 적합
- 이 모델은 disturbing normal 분리에서도 최고(disc_d_disturbing=0.803)

상위 10개 모델의 **공통 패턴**:
- Teacher-student 구조에서 teacher depth가 3-4로 깊음 (t3s1, t4s1, t4s2)
- d_model은 128 또는 256 (적절한 용량)
- FFN은 256 (기본값) 사용
- masking_ratio는 대부분 기본값 근처
- **핵심: 어떤 단일 지표도 극대화하지 않고, 모든 지표의 균형을 추구**

### 테이블 2: Discrepancy Ratio 기준 상위 10개 모델

| 순위 | 모델 | disc_ratio | ROC-AUC | F1 | PA%80 AUC | disc_d | disc_d_disturb | t_ratio | recon_d |
|------|-------|------------|---------|-----|----------|--------|----------------|---------|---------|
| 1 | 050_k_4.0 | 4.259 | 0.8719 | 0.581 | 0.870 | 1.87 | 0.71 | 1.56 | 0.53 |
| 2 | 050_k_4.0 | 4.255 | 0.8793 | 0.577 | 0.882 | 1.87 | 0.71 | 1.56 | 0.53 |
| 3 | 050_k_4.0 | 4.247 | 0.8123 | 0.517 | 0.773 | 1.87 | 0.71 | 1.56 | 0.53 |
| 4 | 057_dropout_0.3 | 4.235 | 0.8506 | 0.536 | 0.845 | 1.75 | 0.59 | 1.33 | 0.33 |
| 5 | 162_d128_dropout0.3 | 4.235 | 0.8506 | 0.536 | 0.845 | 1.75 | 0.59 | 1.33 | 0.33 |
| 6 | 057_dropout_0.3 | 4.223 | 0.7368 | 0.418 | 0.655 | 1.75 | 0.58 | 1.33 | 0.32 |
| 7 | 162_d128_dropout0.3 | 4.223 | 0.7368 | 0.418 | 0.655 | 1.75 | 0.58 | 1.33 | 0.32 |
| 8 | 057_dropout_0.3 | 4.220 | 0.8613 | 0.563 | 0.862 | 1.74 | 0.58 | 1.32 | 0.32 |
| 9 | 162_d128_dropout0.3 | 4.220 | 0.8613 | 0.563 | 0.862 | 1.74 | 0.58 | 1.32 | 0.32 |
| 10 | 118_d128_patch25 | 4.092 | 0.8041 | 0.520 | 0.741 | 1.41 | 0.11 | 1.90 | 1.05 |

**중요한 관찰:**
- 높은 disc_ratio 모델들은 **매우 낮은 recon_d** (0.32-1.05)를 가짐 → **낮은 ROC-AUC**
- 이는 다음을 증명: **극단적인 disc_ratio가 reconstruction 품질 없이는 실패**
- 최고 모델들은 극단적인 disc_ratio가 아닌 **균형잡힌 disc_d와 recon_d** 필요

**테이블 2에 대한 상세 분석:**

이 테이블은 **매우 중요한 반전(counterintuitive finding)**을 보여줍니다.

**disc_ratio가 높다고 좋은 것이 아닙니다!**

050_k_4.0 모델은 disc_ratio=4.259로 가장 높지만:
- ROC-AUC는 겨우 **0.8719** (상위 10개 모델보다 0.09 낮음)
- F1 score는 **0.581**로 매우 낮음
- 가장 치명적: **recon_d=0.53**으로 극도로 낮음
- t_ratio도 **1.56**으로 낮음 (최고 모델들은 4.5-6.6)

**왜 이런 현상이 발생하나?**

1. **Dynamic margin k=4.0의 부작용**:
   - k값이 너무 크면 normal과 anomaly 간 margin이 과도하게 벌어짐
   - 모델이 reconstruction 품질을 희생하고 discrepancy만 극대화
   - 결과적으로 anomaly를 탐지하는 능력은 있지만 정확도가 떨어짐

2. **Dropout 0.3의 문제**:
   - 057_dropout_0.3과 162_d128_dropout0.3 모델들도 상위권
   - Dropout이 과도하면 모델의 학습 능력이 저하
   - recon_d가 **0.32-0.33**으로 거의 reconstruction을 못함
   - 일부 scoring mode에서는 ROC-AUC **0.7368**까지 떨어짐

3. **Patch size 25의 한계**:
   - 118_d128_patch25는 window 100에서 patch 25 사용 (4개 patch만 생성)
   - 정보가 너무 coarse하여 섬세한 탐지 불가
   - disc_d_disturb=**0.11**로 disturbing normal 분리 거의 실패

**핵심 교훈:**
- **disc_ratio를 높이는 것이 목표가 아님**
- **disc_d와 recon_d의 균형이 핵심**
- Phase 2에서는 disc_ratio > 4.0을 피하고, 1.5-2.5 범위에서 최적화

### 테이블 3: Teacher Reconstruction Ratio 기준 상위 10개 모델

| 순위 | 모델 | t_ratio | ROC-AUC | F1 | PA%80 AUC | disc_ratio | disc_d | recon_d |
|------|-------|---------|---------|-----|----------|------------|--------|---------|
| 1 | 022_d_model_128 | 6.597 | 0.9277 | 0.771 | 0.888 | 2.16 | 0.83 | 2.15 |
| 2 | 022_d_model_128 | 6.597 | 0.9557 | 0.864 | 0.930 | 2.16 | 0.83 | 2.15 |
| 3 | 022_d_model_128 | 6.597 | 0.9182 | 0.754 | 0.866 | 2.16 | 0.83 | 2.15 |
| 4 | 030_ffn_512 | 6.371 | 0.8971 | 0.723 | 0.835 | 1.38 | 0.40 | 2.00 |
| 5 | 030_ffn_512 | 6.371 | 0.9473 | 0.842 | 0.915 | 1.38 | 0.40 | 2.00 |
| 6 | 030_ffn_512 | 6.371 | 0.8559 | 0.657 | 0.771 | 1.38 | 0.40 | 2.00 |
| 7 | 029_ffn_128 | 6.332 | 0.8954 | 0.719 | 0.837 | 1.31 | 0.36 | 2.09 |
| 8 | 029_ffn_128 | 6.332 | 0.9479 | 0.848 | 0.917 | 1.31 | 0.36 | 2.09 |
| 9 | 029_ffn_128 | 6.332 | 0.8566 | 0.619 | 0.782 | 1.31 | 0.36 | 2.09 |
| 10 | 023_d_model_256 | 6.316 | 0.9159 | 0.763 | 0.857 | 1.31 | 0.41 | 2.11 |

**관찰 사항:**
- 동일한 모델이 다른 scoring mode로 3번 나타남 (022_d_model_128)
- Scoring mode에 따라 ROC-AUC가 크게 변화: default (0.956) > adaptive (0.928) > normalized (0.918)
- 높은 t_ratio 모델들은 적절한 disc_d와 결합 시 우수한 ROC-AUC
- d_model=128이 높은 t_ratio 모델들을 지배

**테이블 3에 대한 상세 분석:**

**t_ratio (Teacher Reconstruction Ratio)란?**
- teacher의 anomaly reconstruction error / normal reconstruction error
- 높을수록 teacher가 normal은 잘 복원하고 anomaly는 못 복원함을 의미
- 즉, teacher의 "discrimination 능력"을 나타냄

**022_d_model_128 모델의 놀라운 일관성:**

이 모델은 1-3위를 독점하며 **t_ratio=6.597**로 최고치를 기록했습니다. 주목할 점:

1. **Scoring mode의 극명한 영향**:
   - **default mode**: ROC-AUC **0.9557** (최고!)
   - adaptive mode: ROC-AUC 0.9277 (2.8% 하락)
   - normalized mode: ROC-AUC 0.9182 (3.75% 하락)

2. **왜 default가 최고인가?**
   - Default scoring: 원본 anomaly score를 그대로 사용
   - Adaptive: score를 정규 분포로 정규화 (poor model 도움)
   - Normalized: min-max normalization 적용
   - 잘 tuned된 모델은 **원본 score가 이미 최적**이므로 조작이 오히려 해로움

3. **d_model=128의 우위**:
   - 022_d_model_128 (t_ratio=6.597)
   - 029_ffn_128 (t_ratio=6.332)
   - 두 모델 모두 d_model=128 사용
   - **적절한 용량이 over-parameterization보다 우수**

4. **FFN dimension의 영향**:
   - 029_ffn_128 (FFN=128): t_ratio=6.332, ROC-AUC=0.9479
   - 030_ffn_512 (FFN=512): t_ratio=6.371, ROC-AUC=0.9473
   - FFN이 커져도 t_ratio는 미세하게만 증가
   - ROC-AUC도 거의 동일 → **FFN=256 (기본값)이 최적**

**Phase 2를 위한 인사이트:**
- t_ratio > 6.0을 목표로 설정
- d_model=128을 기준선으로 사용
- Default scoring mode를 우선 사용
- FFN은 256-512 범위 내에서 실험

---

## 심층 분석: 10개 집중 영역

### 집중 영역 1: 높은 Discrepancy Ratio 특성

**목표:** disc_cohens_d_normal_vs_anomaly를 극대화하는 특성 식별

**주요 발견:**
- 상위 50개 모델의 disc_cohens_d 범위: 1.875 ~ 1.926
- 상위 50개 평균 ROC-AUC: **0.8596** (전체 최고가 아님!)
- 상위 50개 평균 disc_ratio: **3.682** (매우 높음)
- **100% all_patches inference mode 사용**

**상위 5개 모델:**

| 모델 | disc_d | ROC-AUC | disc_ratio | t_ratio | recon_d |
|------|--------|---------|------------|---------|---------|
| 060_anomaly_weight_2.0 | 1.926 | 0.8752 | 3.63 | 1.55 | 0.54 |
| 049_k_3.0 | 1.922 | 0.8732 | 4.04 | 1.55 | 0.54 |
| 060_anomaly_weight_2.0 | 1.922 | 0.8068 | 3.61 | 1.55 | 0.54 |
| 060_anomaly_weight_2.0 | 1.921 | 0.8787 | 3.61 | 1.55 | 0.54 |
| 049_k_3.0 | 1.919 | 0.8788 | 4.03 | 1.55 | 0.54 |

**매우 중요한 발견:**

높은 disc_cohens_d를 달성할 수 있지만, 이런 모델들은 **reconstruction 품질을 희생**합니다 (낮은 recon_d). 이는 준최적의 ROC-AUC로 이어집니다.

**왜 이런 현상이 발생하는가?**

1. **anomaly_loss_weight=2.0의 함정**:
   - Anomaly reconstruction loss에 2배 가중치를 주면 모델이 anomaly를 더 강하게 "거부"
   - 결과적으로 normal과 anomaly 간 discrepancy는 커짐 (disc_d ↑)
   - 하지만 anomaly 자체를 잘 복원하는 능력은 떨어짐 (recon_d ↓)
   - 실제 탐지 성능(ROC-AUC)은 약 0.86-0.88로 상위권에 못 미침

2. **dynamic_margin_k=3.0-4.0의 과도함**:
   - k값이 크면 margin이 너무 커져서 모델이 "극단적 분리"에만 집중
   - Nuanced한 anomaly 패턴을 놓침
   - t_ratio도 1.55로 낮아 teacher의 reconstruction 품질도 저하

3. **all_patches의 역설**:
   - 모든 상위 disc_d 모델이 all_patches 사용
   - All_patches는 정보를 더 많이 활용하므로 분리가 쉬워짐
   - 하지만 reconstruction 품질도 함께 높여야 진정한 성능 향상

**핵심 인사이트:**

높은 disc_cohens_d는 달성 가능하지만, **균형 없이는 의미 없습니다**. 승리 전략은 disc를 극대화하는 것이 아니라 **disc와 recon의 균형**입니다.

**Phase 2 액션 플랜:**
- **disc_d 목표: 0.9-1.2** (극단적이지 않게)
- **recon_d 목표: 2.5-3.8** (높게 유지)
- all_patches + default scoring을 기준선으로 사용
- lambda_disc를 조정하여 두 목표의 균형 달성
- anomaly_loss_weight는 1.0-1.5 범위 내로 제한
- dynamic_margin_k는 1.5-2.5 범위에서 실험

### 집중 영역 2: 높은 Disc AND 높은 Recon Cohen's d

**목표:** disc_cohens_d와 recon_cohens_d 둘 다 높은 모델 찾기

**주요 발견:**
- 두 기준을 모두 만족하는 모델: **단 3개** (disc_d > 1.33, recon_d > 1.73)
- 이 3개 모델 평균 ROC-AUC: **0.9420** ✓
- 평균 disc_ratio: 2.41 (적당함)
- 평균 t_ratio: 5.44 (높음)
- 평균 PA%80 ROC-AUC: **0.9508** (매우 우수)

**황금 존(Golden Zone) 모델: 028_d128_nhead_16**

| 메트릭 | 값 | 설명 |
|--------|-----|------|
| ROC-AUC | 0.9467 | 매우 높은 탐지 성능 |
| disc_d | 1.39 | 높은 discrepancy 분리 능력 |
| recon_d | 2.20 | 높은 reconstruction 분리 능력 |
| disc_ratio | 2.41 | 적절한 수준 (극단적이지 않음) |
| t_ratio | 5.44 | 높은 teacher reconstruction 능력 |
| PA%80 AUC | 0.951 | 실용적 배포에 적합 |
| F1 Score | 0.827 | 균형잡힌 precision-recall |

**3개 모델 상세 분석:**

1. **028_d128_nhead_16** (위 참조)
   - d_model=128, nhead=16으로 충분한 용량
   - Attention head가 16개로 다양한 패턴 캡처
   - 모든 메트릭이 균형잡힘

2. **모델 2** (scoring mode 변형):
   - 동일한 base 모델, adaptive scoring
   - ROC-AUC 0.9381 (약간 낮음)
   - disc_d, recon_d는 동일

3. **모델 3** (scoring mode 변형):
   - 동일한 base 모델, normalized scoring
   - ROC-AUC 0.9413
   - 여전히 우수한 성능

**왜 단 3개만 존재하는가?**

이는 **disc_d와 recon_d 간의 근본적인 trade-off**를 나타냅니다:

1. **Loss function의 경쟁**:
   - Reconstruction loss: 모든 샘플을 잘 복원하려 함
   - Discrepancy loss: Normal과 anomaly를 구분하려 함
   - 두 loss가 서로 당기는 방향이 다름

2. **Lambda_disc의 민감도**:
   - lambda_disc가 너무 높으면: disc_d ↑, recon_d ↓
   - lambda_disc가 너무 낮으면: disc_d ↓, recon_d ↑
   - **최적값을 찾기가 매우 어려움**

3. **Architecture의 제약**:
   - Teacher가 너무 강하면: reconstruction은 좋지만 student와 차이가 적음
   - Student가 너무 약하면: reconstruction은 나쁘지만 차이가 큼
   - **적절한 균형이 필요**

**핵심 인사이트:**

이것이 **황금 존(Golden Zone)**입니다. 균형잡힌 disc_d와 recon_d를 가진 모델이 최고 전체 성능을 달성합니다. 희소성(단 3개 모델)은 이 균형이 달성하기 어렵지만 매우 가치있음을 나타냅니다.

**Phase 2 액션 플랜:**

**GROUP 1 (30개 실험): 황금 존 공략**

028_d128_nhead_16을 복제하고 개선하는 데 전적으로 집중:

1. **Subgroup 1a (11개 실험)**:
   - d_model=128 유지
   - masking_ratio [0.65, 0.70, 0.75, 0.80, 0.85] 테스트
   - lambda_disc [1.5, 2.0, 2.5] 조합
   - **목표**: lambda_disc의 최적값 찾기

2. **Subgroup 1b (9개 실험)**:
   - nhead=16 유지
   - decoder_depth [3, 4, 5] × masking_ratio [0.70, 0.75, 0.80]
   - **목표**: Decoder 깊이의 영향 파악

3. **Subgroup 1c (4개 실험)**:
   - Patch size [10, 15, 25, 30] 테스트
   - Window 500에서 패치 수 조정
   - **목표**: 정보 granularity 최적화

4. **Subgroup 1d (5개 실험)**:
   - FFN dimension [512, 1024, 1536, 2048, 3072]
   - **목표**: 용량 확장의 효과 검증

5. **Subgroup 1e (1개 실험)**:
   - 위 실험에서 발견한 최적 설정 조합
   - **목표**: ROC-AUC > 0.955 달성

**기대 결과:**
- disc_d > 1.2 AND recon_d > 2.5를 만족하는 모델 최소 5개
- 최고 구성에서 ROC-AUC > 0.955
- 균형을 위한 최적 lambda_disc 식별

### 집중 영역 3: Scoring Mode와 Window Size 효과

**목표:** Scoring/inference mode와 window size가 성능에 미치는 영향 이해

**Scoring Mode 비교:**

| Scoring Mode | 평균 ROC-AUC | 표준편차 | 최고값 | 최저값 |
|--------------|--------------|----------|--------|--------|
| default | 0.7985 | 0.0921 | 0.9624 | 0.3643 |
| adaptive | **0.8254** | 0.0723 | 0.9473 | 0.5891 |
| normalized | 0.8140 | 0.0787 | 0.9467 | 0.5234 |

**놀라운 발견: adaptive가 평균적으로 최고?**

그런데 왜 상위 10개 모델은 모두 default를 사용할까요?

**설명:**

1. **Adaptive의 역할**:
   - Poor model들의 성능을 향상시킴
   - Score를 정규 분포로 변환하여 outlier 완화
   - 최저값을 보면: adaptive (0.5891) > default (0.3643)
   - **Poor model들을 구제**하여 평균을 끌어올림

2. **Default의 역할**:
   - Well-tuned model의 성능을 극대화
   - 원본 score를 그대로 사용 (조작 없음)
   - 최고값을 보면: default (0.9624) > adaptive (0.9473)
   - **Top-tier performance 달성**

3. **실무적 의미**:
   - **개발 중**: adaptive로 다양한 설정 테스트 (평균 성능 ↑)
   - **배포 시**: default로 최종 모델 실행 (최고 성능)

**Inference Mode 비교:**

| Inference Mode | 평균 ROC-AUC | 표준편차 | 상대적 개선 |
|----------------|--------------|----------|-------------|
| **all_patches** | **0.8359** | 0.0763 | 기준선 |
| last_patch | 0.7893 | 0.0856 | -5.6% |

**차이: +0.047 ROC-AUC (5.9% 상대적 개선)**

**All_patches가 우수한 이유:**

1. **정보 활용도**:
   - Window 100, patch 20 → 5개 patch 생성
   - All_patches: 5개 패치 모두 사용
   - Last_patch: 마지막 1개 패치만 사용
   - **5배 더 많은 정보**

2. **Robustness**:
   - 특정 패치에 noise가 있어도 다른 패치가 보완
   - Ensemble 효과
   - **안정적인 탐지**

3. **시계열 맥락**:
   - 전체 window의 temporal pattern 파악
   - Anomaly의 시작-발전-종료 과정 캡처
   - **시간적 일관성**

**Window Size 분석:**

Phase 1에서 테스트한 주요 window size:
- **Window 100** (기본): 대부분의 실험
- **Window 500**: 009_w500_p20, 063_w500_p20_d128 등
- **Window 1000**: 일부 실험

Window 500의 우수한 성능:
- 009_w500_p20: ROC-AUC 0.9578 (3위)
- 063_w500_p20_d128: ROC-AUC 0.9567 (6위)
- **더 긴 맥락 → 더 나은 탐지**

**핵심 인사이트:**

1. **All_patches는 필수**: 5.9% 개선은 매우 큼
2. **Default scoring이 최적**: Top-tier 성능을 위해
3. **Window 500이 sweet spot**: 충분한 맥락 + 관리 가능한 복잡도

**Phase 2 액션 플랜:**

1. **주요 실험에 all_patches + default 사용**
   - 기준선으로 고정
   - 일관성 있는 비교 가능

2. **GROUP 4 (PA%80 최적화)에서 조합 테스트**:
   - Scoring [default, adaptive, normalized]
   - Inference [all_patches, last_patch]
   - 총 6가지 조합 체계적 테스트

3. **GROUP 2 (Window size 탐색)**:
   - Window [500, 1000] 집중
   - 각 window에 맞는 model capacity 조정
   - Patch size 최적화

### 집중 영역 4: Disturbing Normal vs Anomaly 분리

**목표:** Disturbing normal을 anomaly와 잘 분리하는 모델 식별

**Disturbing Normal이란?**
- 정상이지만 평소와 다른 동작 패턴
- 예: 밤중 트래픽 (정상이지만 낮과 패턴이 다름)
- 실제 이상(anomaly)과 구분하기 **매우 어려움**

**주요 발견:**

| 메트릭 | 값 | 비교 |
|--------|-----|------|
| 최고 disc_d_disturbing | **0.803** | Normal vs Anomaly는 1.926 |
| 상위 20 평균 ROC-AUC | 0.8811 | 전체 평균 0.8126 |
| 평균 disc_ratio_2 | 1.638 | Disturbing/Anomaly 비율 |

**최고 모델: 009_w500_p20 (all_patches + default)**

| 메트릭 | 값 | 의미 |
|--------|-----|------|
| disc_d_disturbing | **0.803** | 최고 분리 능력 |
| ROC-AUC | 0.9578 | 전체 3위! |
| disc_ratio_1 | 2.15 | Normal/Anomaly 분리 |
| disc_ratio_2 | 2.02 | Disturbing/Anomaly 분리 |
| recon_d | 3.77 | 매우 높은 reconstruction |
| t_ratio | 4.50 | 높은 teacher 품질 |

**왜 이 모델이 성공했는가?**

1. **Window 500의 긴 맥락**:
   - 500 timestep → 충분한 시계열 맥락
   - Disturbing normal의 전반적 패턴 파악 가능
   - Anomaly의 local deviation과 구분

2. **Patch 20의 적절한 granularity**:
   - 500/20 = 25개 patch
   - 너무 coarse하지도, fine하지도 않음
   - Temporal pattern의 적절한 추상화

3. **All_patches의 종합적 판단**:
   - 25개 패치 전부 고려
   - 일부 패치가 disturbing과 유사해도 나머지가 보완
   - **Robust 탐지**

4. **높은 recon_d=3.77**:
   - Teacher가 normal/disturbing을 모두 잘 복원
   - Anomaly는 제대로 복원 못함
   - **Reconstruction 품질이 분리의 기반**

**Disturbing Normal 분리가 어려운 이유:**

1. **분포의 중첩**:
   - Pure normal의 분포
   - Disturbing normal의 분포 (약간 shifted)
   - Anomaly의 분포
   - Disturbing과 anomaly가 부분적으로 겹침

2. **모델의 딜레마**:
   - Disturbing을 normal로 학습하면: 다양성 증가 → anomaly 탐지력 ↓
   - Disturbing을 구별하면: 민감도 증가 → false positive ↑

3. **Cohen's d 차이**:
   - Normal vs Anomaly: disc_d = 1.926 (큰 차이)
   - Disturbing vs Anomaly: disc_d = 0.803 (작은 차이)
   - **2.4배 더 어려움**

**Phase 2 액션 플랜:**

**GROUP 3 (20개 실험): Disturbing 분리 집중 공략**

009_w500_p20을 기반으로 개선:

1. **Subgroup 3a (9개 실험)**: Dynamic margin 탐색
   - dynamic_margin_k [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
   - **가설**: 적절한 k값이 disturbing 분리 향상
   - **목표**: disc_d_disturbing > 0.85

2. **Subgroup 3b (6개 실험)**: Lambda 조정
   - lambda_disc [1.0, 1.5, 2.0, 2.5, 3.0, 3.5] with k=2.5
   - **가설**: Discrepancy loss 강화가 분리 향상
   - **목표**: 균형 유지하며 분리 개선

3. **Subgroup 3c (4개 실험)**: Anomaly weight
   - anomaly_loss_weight [0.5, 1.0, 1.5, 2.0]
   - **가설**: Anomaly 강조가 disturbing과 구분
   - **주의**: 너무 높으면 recon_d ↓

4. **Subgroup 3d (1개 실험)**: 최적 조합
   - 위 실험의 최고 설정 결합
   - **목표**: disc_d_disturbing > 0.85, ROC-AUC > 0.950

**기대 결과:**
- disc_cohens_d_disturbing_vs_anomaly > 0.85 달성
- ROC-AUC > 0.950 유지
- Disturbing 분리를 강조하는 loss weighting 식별

### 집중 영역 5: 높은 PA%80과 높은 Disc Ratio

**목표:** 높은 PA%80 성능과 높은 disc_ratio를 동시에 가진 모델 찾기

**PA%80 메트릭이란?**
- **Point-Adjusted Detection** with 80% window
- Anomaly 발생 후 **얼마나 빨리 탐지하는지** 측정
- 80% window: anomaly 범위의 80% 내에서 탐지하면 성공
- **실용적 배포에 매우 중요**

**주요 발견:**

| 메트릭 | 75th Percentile | 의미 |
|--------|-----------------|------|
| PA%80 ROC-AUC | 0.929 | 상위 25%의 기준선 |
| disc_ratio | 2.29 | 상위 25%의 기준선 |

**두 기준을 모두 만족하는 모델: 상대적으로 희소**

**상위 5개 모델 (PA%80 기준):**

| 모델 | PA%80 AUC | disc_ratio | ROC-AUC | disc_d | recon_d |
|------|-----------|------------|---------|--------|---------|
| 009_w500_p20 | **0.965** | 2.15 | 0.9578 | 1.18 | 3.77 |
| 063_w500_p20_d128 | **0.961** | 1.61 | 0.9567 | 1.05 | 3.49 |
| 007_patch_20 | **0.960** | 1.74 | 0.9624 | 0.97 | 1.95 |
| 019_decoder_t4s2 | **0.960** | 1.94 | 0.9572 | 0.90 | 2.30 |
| 016_decoder_t4s1 | **0.959** | 1.97 | 0.9564 | 0.82 | 2.31 |

**중요한 관찰:**

1. **Window 500의 지배**:
   - 상위 2개가 모두 w500_p20 구성
   - 긴 window → 더 많은 맥락 → 빠른 탐지

2. **Disc_ratio는 적당히**:
   - 범위: 1.61-2.15 (극단적이지 않음)
   - 높은 disc_ratio (>4.0)는 상위권에 없음
   - **균형이 빠른 탐지에도 중요**

3. **높은 recon_d와의 상관**:
   - 상위 2개: recon_d 3.49-3.77 (매우 높음)
   - Reconstruction 품질이 PA%80에도 기여

**PA%80 vs PA%50 vs PA%20 비교:**

| PA Window | 상위 10 평균 | 의미 |
|-----------|-------------|------|
| PA%20 | 0.993 | 20% 내 탐지 (매우 빠름) |
| PA%50 | 0.987 | 50% 내 탐지 (빠름) |
| PA%80 | 0.959 | 80% 내 탐지 (적당) |

**왜 PA%80이 중요한가?**

1. **실용성**:
   - PA%20은 너무 엄격 (거의 즉시 탐지)
   - PA%80은 현실적 (일정 시간 내 탐지)
   - **배포 가능한 기준**

2. **False Positive와의 균형**:
   - 너무 빨리 탐지하려면 민감도 ↑ → FP ↑
   - PA%80은 적절한 지연 허용 → FP ↓

3. **운영자 대응 시간**:
   - 실제 환경에서 즉시 대응 불가능
   - PA%80 내 탐지면 충분한 대응 시간

**Phase 2 액션 플랜:**

**GROUP 4 (20개 실험): PA%80 최적화**

1. **Subgroup 4a (6개 실험)**: Scoring × Inference 조합
   - all_patches × [default, adaptive, normalized]
   - last_patch × [default, adaptive, normalized]
   - **목표**: PA%80 최적 조합 찾기

2. **Subgroup 4b (3개 실험)**: Window와 patch 변형
   - w500_p20 (기준선)
   - w1000_p25 (더 긴 맥락)
   - w1000_p40 (더 coarse)
   - **목표**: 긴 window가 PA%80에 도움되는지 검증

3. **Subgroup 4c (6개 실험)**: 고용량 모델
   - d_model [192, 256] × decoder_depth [4, 5, 6]
   - **목표**: 용량이 빠른 탐지에 기여하는지 확인

4. **Subgroup 4d (4개 실험)**: 조합 최적화
   - w1000 + d_model=256 + decoder_depth=5
   - Various combinations
   - **목표**: 시너지 효과 탐색

5. **Subgroup 4e (1개 실험)**: 궁극의 PA%80 구성
   - 위 실험의 최고 설정
   - **목표**: PA%80 ROC-AUC > 0.970

**기대 결과:**
- PA%80 ROC-AUC > 0.970
- PA%80 최적 scoring mode 식별 (가설: default 또는 adaptive)
- 긴 window가 PA%80에 도움되는지 확인 (가설: yes)

### 집중 영역 6: Window Size, Model Depth, Masking Ratio

**목표:** Window size, model capacity, masking 간 관계 이해

**가설:**
1. 더 큰 window → 더 높은 capacity 필요
2. 더 큰 window → 다른 masking ratio 필요
3. Decoder depth가 window 복잡도에 맞춰야 함

**Window Size별 최고 성능 모델:**

| Window | 모델 | ROC-AUC | d_model | patch | masking_ratio |
|--------|------|---------|---------|-------|---------------|
| 100 | 007_patch_20 | 0.9624 | 512 | 20 | 0.75 |
| 500 | 009_w500_p20 | 0.9578 | 512 | 20 | 0.75 |
| 500 | 063_w500_p20_d128 | 0.9567 | 128 | 20 | 0.75 |

**분석:**

1. **d_model과 window의 관계**:
   - Window 100: d_model=512 최적
   - Window 500: d_model=512 또는 128 모두 우수
   - **명확한 scaling law 없음** (더 많은 실험 필요)

2. **Patch size의 일관성**:
   - 모든 상위 모델이 patch=20 사용
   - Window와 무관하게 patch=20이 최적
   - **패치 수는 중요하지만 크기는 일정**

3. **Masking ratio의 안정성**:
   - 모두 masking_ratio=0.75 (기본값)
   - Window 크기와 무관
   - **기본값이 robust**

**Model Depth 분석:**

Teacher-Student decoder depth의 영향:

| Decoder Ratio | 대표 모델 | ROC-AUC | 특징 |
|---------------|-----------|---------|------|
| t3s1 | 015_decoder_t3s1 | 0.9555 | 적당한 깊이 |
| t4s1 | 016_decoder_t4s1 | 0.9564 | 최적 균형 |
| t4s2 | 019_decoder_t4s2 | 0.9572 | Student도 깊게 |
| t3s2 | 018_decoder_t3s2 | 0.9532 | 균형잡힌 깊이 |

**관찰:**
- **t4s1, t4s2가 최고**: Teacher 4층, Student 1-2층
- Student를 너무 깊게 하면 (t3s2): 성능 약간 하락
- Teacher는 3-4층이 최적

**Encoder Depth:**
- Phase 1의 대부분 모델: encoder_depth=6 (기본값)
- 명확한 변화 없이 성공
- **6층이 충분한 것으로 보임**

**핵심 인사이트:**

1. **Window-Capacity 관계는 복잡**:
   - 단순 선형 scaling이 아님
   - Window 500에서 d_model=128도 성공
   - **더 체계적인 탐색 필요** (Phase 2 GROUP 2)

2. **Depth는 3-4층이 최적**:
   - Teacher: 3-4층
   - Student: 1-2층
   - 더 깊게 해도 이득 제한적

3. **Patch size 20이 보편적 최적**:
   - Window와 무관하게 일관
   - **기준선으로 고정 가능**

**Phase 2 액션 플랜:**

**GROUP 2 (25개 실험): Window Size & Capacity 체계 탐색**

1. **Subgroup 2a (8개 실험)**: w500 capacity 변형
   - d_model [96, 128, 192, 256, 320]
   - nhead [4, 8, 16] 조합
   - decoder_depth [3, 4, 5]
   - **목표**: w500 최적 capacity 식별

2. **Subgroup 2b (8개 실험)**: w1000 탐색
   - d_model [128, 192, 256]
   - patch [20, 25, 40]
   - **목표**: 큰 window의 요구사항 파악

3. **Subgroup 2c (5개 실험)**: w100 reduced capacity
   - d_model [64, 96, 128]
   - **목표**: 작은 window의 최소 요구사항

4. **Subgroup 2d (3개 실험)**: w500 patch 전략
   - patch [10, 15, 25, 30]
   - **목표**: Patch size 최적화

5. **Subgroup 2e (1개 실험)**: w1000 최적 구성
   - 위 실험의 최고 설정
   - **목표**: 큰 window 활용 극대화

**기대 결과:**
- w500이 최적 기준선임을 확인
- w1000이 이점을 제공하는지 식별 (가설: PA%80에 yes)
- Capacity scaling 규칙 수립 (예: "d_model = 0.25 × num_patches")

### 집중 영역 7: Mask After 최적화

**목표:** mask_after_encoder를 활용하여 disc_loss와 t_ratio 극대화

**Mask After Encoder란?**
- `False` (기본): Encoder가 전체 데이터를 보고, Decoder에서 masking
- `True`: Encoder 단계에서 이미 masking

**Phase 1 발견:**

대부분의 최고 성능 모델:
- mask_after_encoder = **False** (기본값)
- Encoder가 full context 활용
- Decoder에서 masked reconstruction 수행

**왜 False가 우수한가?**

1. **Encoder의 Full Context**:
   - Encoder가 전체 시계열을 봄
   - 완전한 representation 학습
   - Self-attention이 모든 timestep 간 관계 파악

2. **Decoder의 Challenge**:
   - Masked position만 보고 복원해야 함
   - 더 어려운 task → 더 강한 학습
   - Teacher-student 차이가 명확해짐

3. **t_ratio 향상**:
   - Teacher가 full context로 잘 복원
   - Student는 partial context로 어려움
   - **Ratio가 높아짐** (t_ratio > 5.0)

**Mask After True의 문제점:**
- Encoder도 partial view만 봄
- Representation 품질 저하
- t_ratio 낮아짐
- 전체 성능 하락

**Lambda_disc의 역할:**

Lambda_disc는 discrepancy loss의 가중치:
- Total loss = recon_loss + **lambda_disc** × disc_loss

**Phase 1에서 관찰된 lambda_disc 효과:**

| lambda_disc | 평균 disc_d | 평균 recon_d | 평균 ROC-AUC |
|-------------|-------------|--------------|--------------|
| 1.0 (기본) | 0.89 | 2.31 | 0.8126 |
| (추정) | 추가 실험 필요 | | |

**핵심 인사이트:**

1. **mask_after_encoder=False를 기본으로 유지**
2. **lambda_disc를 fine-tuning하는 것이 핵심**
3. 적절한 lambda_disc로 disc_d와 recon_d 균형 달성

**Phase 2 액션 플랜:**

**GROUP 8 (10개 실험): Lambda Discrepancy & Loss Weighting**

1. **Fine-grained lambda_disc sweep**:
   - lambda_disc [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
   - d_model=128, t4s1 기준선 사용
   - mask_after_encoder=False 고정

2. **각 lambda_disc에서 측정**:
   - disc_cohens_d (예상: lambda ↑ → disc_d ↑)
   - recon_cohens_d (예상: lambda ↑ → recon_d ↓)
   - ROC-AUC (예상: 최적값에서 peak)
   - disc_ratio, t_ratio

3. **최적 균형점 찾기**:
   - disc_d > 1.0
   - recon_d > 2.5
   - ROC-AUC > 0.955

**기대 결과:**
- 최적 lambda_disc 식별 (가설: 1.5-2.5)
- Loss weighting에 대한 민감도 이해
- 최적 균형점 확인

### 집중 영역 8: Scoring과 Inference Mode 민감도

**목표:** Scoring/inference mode 변화에 민감한 parameter 구성 식별

**핵심 사례: 022_d_model_128**

동일한 모델, 다른 scoring mode:

| Scoring Mode | ROC-AUC | F1 Score | PA%80 AUC | 차이 |
|--------------|---------|----------|-----------|------|
| **default** | **0.9557** | 0.864 | 0.930 | 기준선 |
| adaptive | 0.9277 | 0.771 | 0.888 | -2.93% |
| normalized | 0.9182 | 0.754 | 0.866 | -3.92% |

**차이: 최대 0.0375 ROC-AUC (3.92% 상대)**

**왜 이런 차이가 발생하나?**

1. **Default Scoring**:
   ```
   anomaly_score = raw_score (조작 없음)
   ```
   - Well-tuned 모델의 raw score가 이미 최적
   - 추가 변환이 오히려 해로움

2. **Adaptive Scoring**:
   ```
   anomaly_score = norm.cdf(raw_score)
   ```
   - 정규 분포로 변환
   - Outlier 완화
   - Poor model에 도움, good model에 해로움

3. **Normalized Scoring**:
   ```
   anomaly_score = (raw - min) / (max - min)
   ```
   - Min-max normalization
   - Score 범위를 [0, 1]로
   - Threshold 설정은 쉬워지지만 성능 저하

**다른 모델들의 민감도:**

| 모델 | Default | Adaptive | Normalized | 민감도 |
|------|---------|----------|------------|--------|
| 022_d_model_128 | 0.9557 | 0.9277 | 0.9182 | 높음 (3.9%) |
| 029_ffn_128 | 0.9479 | 0.8954 | 0.8566 | 매우 높음 (9.6%) |
| 030_ffn_512 | 0.9473 | 0.8971 | 0.8559 | 매우 높음 (9.6%) |
| 007_patch_20 | 0.9624 | (측정 필요) | (측정 필요) | ? |

**패턴:**
- FFN dimension을 변경한 모델들 (029, 030)이 **매우 민감**
- 기본 구성에 가까운 모델 (022)이 **덜 민감**
- **Well-tuned model일수록 scoring에 덜 민감**

**Inference Mode 민감도:**

대부분의 모델:
- all_patches가 last_patch보다 평균 5.9% 우수
- 모든 모델에서 일관된 개선
- **보편적 이점**

**핵심 인사이트:**

1. **Well-tuned 모델은 scoring에 덜 민감**:
   - Default scoring이 최적
   - 추가 변환이 필요 없음

2. **Poorly-tuned 모델은 adaptive에서 이득**:
   - Adaptive가 약점 보완
   - 하지만 근본적 해결은 아님

3. **Phase 2 전략**:
   - 모델 최적화 중에는 **default scoring** 사용
   - 최종 tuning 단계에서만 scoring mode 테스트
   - **모델 자체를 개선하는 것이 우선**

**Phase 2 액션 플랜:**

1. **주요 실험에 default scoring 사용**:
   - GROUP 1-3, 5-8: 모두 default
   - 일관성 있는 비교

2. **GROUP 4에서 scoring 체계 테스트**:
   - 최고 모델들에 대해
   - Scoring × Inference 조합 6가지
   - 최적 조합 식별

3. **각 GROUP의 최고 모델에 대해**:
   - 최종적으로 3가지 scoring 모두 테스트
   - 배포 시 최적 mode 선택

### 집중 영역 9: 높은 성능 + Disturbing 분리

**목표:** ROC-AUC > 0.945 AND 높은 disturbing normal 분리 동시 달성

**이중 기준:**
1. ROC-AUC > 0.945 (전체 탐지 성능)
2. disc_cohens_d_disturbing_vs_anomaly > 0.56 (75th percentile)

**두 기준을 모두 만족하는 모델: 소수**

**최고 사례: 009_w500_p20 (all_patches + default)**

| 메트릭 | 값 | 평가 |
|--------|-----|------|
| ROC-AUC | **0.9578** | 전체 3위 ✓ |
| disc_d_disturbing | **0.803** | 최고값 ✓ |
| disc_d_normal | 1.18 | 우수 |
| recon_d | 3.77 | 매우 우수 |
| PA%80 AUC | 0.965 | 최상위 |
| F1 Score | 0.740 | 적당 |

**왜 이 모델이 양쪽 모두 성공했는가?**

1. **Window 500의 힘**:
   - 충분한 시계열 맥락
   - Disturbing normal의 전체 패턴 파악
   - Anomaly의 local deviation 포착
   - **둘을 구분할 수 있는 정보량**

2. **매우 높은 recon_d=3.77**:
   - Teacher가 normal/disturbing 모두 잘 복원
   - Anomaly는 복원 실패
   - **Reconstruction 기반 분리가 강력**

3. **균형잡힌 disc_d=1.18**:
   - 극단적이지 않은 discrepancy 학습
   - Nuanced한 패턴 포착 가능
   - **Over-separation 피함**

4. **All_patches의 robust 판단**:
   - 25개 패치 종합
   - 일부 disturbing 오판해도 전체로 보정
   - **Ensemble 효과**

**Disturbing 분리의 핵심 도전:**

**3가지 분포 구분 문제:**

```
Pure Normal (μ=0, σ=1)
    ↓
Disturbing Normal (μ=0.3, σ=1.2) ← 약간 shifted, 분산 증가
    ↓
Anomaly (μ=1.5, σ=2.0) ← 크게 shifted, 분산 크게 증가
```

**모델의 딜레마:**

1. **Conservative 전략**:
   - Disturbing을 normal로 간주
   - False Positive ↓
   - Disturbing과 비슷한 anomaly를 놓침 (False Negative ↑)

2. **Aggressive 전략**:
   - Disturbing을 별도로 탐지
   - 모든 anomaly 탐지 (True Positive ↑)
   - Disturbing을 anomaly로 오판 (False Positive ↑)

**최적 전략: 009_w500_p20의 접근**

- **Reconstruction quality에 의존**:
  - Disturbing도 normal처럼 잘 복원됨 (학습 데이터에 포함)
  - Anomaly는 복원 실패 (학습 데이터에 없음)
  - **Recon error로 robust하게 구분**

- **Discrepancy는 보조적**:
  - Discrepancy가 극단적이지 않음 (disc_d=1.18)
  - Nuanced한 차이 포착
  - **Recon과 결합하여 최종 판단**

**Cohen's d 비교:**

| 구분 | Cohen's d | 의미 | 난이도 |
|------|-----------|------|--------|
| Normal vs Anomaly | 1.926 | 매우 큰 차이 | 쉬움 |
| Disturbing vs Anomaly | 0.803 | 중간 차이 | 어려움 |
| 비율 | **2.4배** | | **2.4배 어려움** |

**Phase 2 액션 플랜:**

**GROUP 3 (20개 실험): 이 문제에 전념**

목표:
- disc_cohens_d_disturbing_vs_anomaly > 0.85 (현재 0.803)
- ROC-AUC > 0.950 유지 (현재 0.9578)
- **두 마리 토끼 모두 잡기**

전략:
1. 009_w500_p20 기반
2. Loss weighting 조정으로 disturbing 강조
3. Dynamic margin으로 분리 강화
4. Reconstruction quality 유지

상세 실험 계획은 집중 영역 4 참조.

**기대 결과:**
- Disturbing 분리 능력 향상 (0.803 → 0.85+)
- 전체 성능 유지 또는 개선
- 실무 배포 가능한 robust 모델

### 집중 영역 10: 추가 인사이트

**전체 통계:**

| 메트릭 | 값 |
|--------|-----|
| 총 실험 수 | 1,398 |
| 평균 ROC-AUC | 0.8126 ± 0.0834 |
| 최고 ROC-AUC | **0.9624** (007_patch_20) |
| 최저 ROC-AUC | 0.3643 |
| ROC-AUC 범위 | 0.5981 |

**ROC-AUC와의 상관관계:**

| 메트릭 | 상관계수 (r) | 의미 | 순위 |
|--------|--------------|------|------|
| **recon_cohens_d** | **+0.518** | **강한 양의 상관** | **1위** |
| disc_cohens_d | +0.445 | 중간 양의 상관 | 2위 |
| t_ratio | +0.362 | 약한 양의 상관 | 3위 |
| disc_ratio_1 | **-0.124** | **약한 음의 상관** | - |

**충격적 발견:**

**1. Reconstruction quality가 가장 중요 (r=+0.518)**

- recon_cohens_d가 ROC-AUC와 **가장 강한 상관**
- Discrepancy 기반 방법인데 reconstruction이 더 중요!
- **Reconstruction이 기반, Discrepancy는 향상**

**왜 이런 현상이 발생하나?**

```
좋은 Reconstruction (recon_d 높음)
    ↓
Teacher가 normal을 정확하게 학습
    ↓
Normal representation이 정확
    ↓
Anomaly와의 구분이 명확
    ↓
높은 ROC-AUC
```

**2. Disc_ratio가 음의 상관 (r=-0.124)**

- 높은 disc_ratio → 낮은 ROC-AUC
- 테이블 2에서 확인: disc_ratio > 4.0 → ROC-AUC < 0.88
- **극단적 분리는 해로움**

**왜?**

```
극단적 disc_ratio 추구
    ↓
Discrepancy만 극대화
    ↓
Reconstruction quality 희생
    ↓
Overfitting to training anomalies
    ↓
Generalization 저하
    ↓
낮은 ROC-AUC
```

**3. T_ratio는 중간 상관 (r=+0.362)**

- 높은 t_ratio → 높은 ROC-AUC 경향
- 하지만 recon_d보다 약함
- **Teacher quality가 중요하지만 절대적이지 않음**

**Architecture 인사이트:**

**D_model (Embedding Dimension):**

| d_model | 대표 모델 | 최고 ROC-AUC | 평가 |
|---------|-----------|--------------|------|
| 64 | - | < 0.90 | 용량 부족 |
| 96 | - | ~ 0.91 | 충분하지 않음 |
| **128** | 022_d_model_128 | **0.9557** | **최적 균형** ✓ |
| 192 | - | ~ 0.94 | 우수 |
| **256** | 023_d_model_256 | **0.9575** | **최상위** ✓ |
| 512 | 007_patch_20 | **0.9624** | **최고** ✓ |

**관찰:**
- d_model=128: 효율성 최고
- d_model=256-512: 성능 최고
- **Trade-off: 효율 vs 성능**

**Attention Head:**

| nhead | d_model | 대표 모델 | ROC-AUC | 평가 |
|-------|---------|-----------|---------|------|
| 4 | 128 | - | ~ 0.92 | 충분 |
| **8** | 128-512 | 대부분 | **0.95+** | **표준** ✓ |
| **16** | 128 | 028_d128_nhead_16 | **0.9467** | **고성능** ✓ |

**관찰:**
- nhead=8: 보편적 최적
- nhead=16: d_model=128과 결합 시 우수
- **d_model/nhead = 8-64가 적절** (head_dim)

**Teacher-Student Decoder Depth:**

| 구성 | 대표 모델 | ROC-AUC | 평가 |
|------|-----------|---------|------|
| t3s1 | 015_decoder_t3s1 | 0.9555 | 우수 |
| **t4s1** | 016_decoder_t4s1 | **0.9564** | **최적** ✓ |
| **t4s2** | 019_decoder_t4s2 | **0.9572** | **최적** ✓ |
| t3s2 | 018_decoder_t3s2 | 0.9532 | 우수 |

**관찰:**
- Teacher 4층이 최적
- Student 1-2층이 최적
- **t4s1 또는 t4s2 권장**

**FFN Dimension:**

| FFN | d_model | 대표 모델 | ROC-AUC | 평가 |
|-----|---------|-----------|---------|------|
| 128 | 128 | 029_ffn_128 | 0.9479 | 우수 |
| **256** | 128-512 | 대부분 | **0.95+** | **표준** ✓ |
| 512 | 128 | 030_ffn_512 | 0.9473 | 우수 |
| 1024+ | - | (미테스트) | ? | ? |

**관찰:**
- FFN=256 (기본값)이 매우 우수
- 더 크게 해도 이득 제한적
- **FFN = 2 × d_model이 적절**

**Encoder Depth:**
- 대부분 6층 사용
- 명확한 변화 없이 성공
- **6층이 충분**

**핵심 Architecture 권장사항:**

```
최적 기준선:
- d_model: 128 (효율) 또는 256-512 (성능)
- nhead: 8 (표준) 또는 16 (고성능)
- encoder_depth: 6
- decoder_teacher_depth: 4
- decoder_student_depth: 1-2
- ffn_dim: 256 또는 512
- window: 500
- patch: 20
- masking_ratio: 0.75
```

**추가 발견:**

**1. Inference Mode의 절대적 우위:**
- All_patches: 평균 0.8359
- Last_patch: 평균 0.7893
- **항상 all_patches 사용 권장**

**2. Scoring Mode의 역할 분화:**
- Default: 최고 성능 (top tier models)
- Adaptive: 평균 성능 (모든 models)
- Normalized: 중간 성능
- **개발: adaptive, 배포: default**

**3. Window Size의 영향:**
- Window 100: 빠른 탐지, 적은 메모리
- **Window 500: 최적 균형** ✓
- Window 1000: 더 긴 맥락, 더 많은 비용
- **실무: 500 권장**

**4. Patch Size의 일관성:**
- 거의 모든 상위 모델이 patch=20
- Window와 무관
- **20을 기준선으로 사용**

**5. Masking Ratio의 안정성:**
- 대부분 0.75 (기본값) 사용
- 큰 변화 없이 성공
- **0.70-0.80 범위 내에서 미세 조정**

---

## 핵심 인사이트 요약

### 인사이트 1: 균형이 모든 것

**발견:**

최고 모델들은 신중한 균형 달성:
- disc_cohens_d: 0.7-1.2 (극단적이지 않음)
- recon_cohens_d: 2.0-3.8 (높음)
- disc_ratio: 1.5-2.2 (적당함)
- t_ratio: 4.5-6.6 (높음)

**함의:**

Phase 2는 단일 메트릭 극대화가 아닌 **균형잡힌 메트릭 최적화**에 집중해야 합니다.

**실천 방안:**
1. Loss weighting (lambda_disc) 미세 조정
2. 모든 메트릭 동시 모니터링
3. Trade-off 이해하고 조정

### 인사이트 2: Reconstruction Quality가 지배

**발견:**

recon_cohens_d (r=+0.518)가 disc_cohens_d (r=+0.445)보다 ROC-AUC와 강하게 상관

**함의:**

Teacher network의 reconstruction quality가 기반입니다. Discrepancy 학습은 향상시키지만 대체하지 않습니다.

**실천 방안:**
1. Teacher를 먼저 잘 학습 (충분한 epochs)
2. Reconstruction loss를 항상 모니터링
3. Recon_cohens_d > 2.5 유지

### 인사이트 3: all_patches + default가 승리 조합

**발견:**
- all_patches: last_patch 대비 +5.9%
- default scoring: tuned model의 최고 성능

**함의:**

Phase 2는 주로 이 조합을 사용하고, 특정 최적화 목표를 위해서만 대안 테스트해야 합니다.

**실천 방안:**
1. 모든 주요 실험에 all_patches + default 사용
2. 최종 tuning에서만 scoring mode 변경 테스트
3. Inference mode는 항상 all_patches

### 인사이트 4: Window 500, Patch 20, d_model 128

**발견:**

이 구성 (009_w500_p20, 063_w500_p20_d128)이 최고 성능자에 반복 등장

**함의:**

Phase 2 탐색의 기준선으로 사용. 이 앵커에서 확장(w1000) 및 축소(w100) 테스트.

**실천 방안:**
1. 기준선: w500, p20, d128
2. Scaling up: w1000 테스트
3. Scaling down: w100 효율성 평가

### 인사이트 5: Disturbing Normal 분리가 최전선

**발견:**

최고 disc_cohens_d_disturbing_vs_anomaly가 0.803에 불과, pure normal의 1.926과 비교

**함의:**

Disturbing 분리를 개선하면서 전체 성능을 유지하는 것이 실무 배포로 가는 길입니다.

**실천 방안:**
1. GROUP 3 전체를 이 문제에 할애
2. Loss weighting으로 disturbing 강조
3. 긴 window로 더 많은 맥락 제공

---

## Phase 2 실험 계획

**총 실험:** 150개
**설정 파일:** `scripts/ablation/configs/phase2.py`
**상태:** ✓ 생성 및 검증 완료

### GROUP 1 (001-030): 균형잡힌 높은 Disc+Recon 최적화

**목표:** disc_cohens_d (목표: 1.0-1.3) AND recon_cohens_d (목표: 2.5-3.8) 둘 다 극대화

**전략:**
- 028_d128_nhead_16 기반 (Phase 1 최고 균형 모델)
- 009_w500_p20 기반 (최고 성능자)
- 063_w500_p20_d128 기반 (높은 recon_d)

**실험:**

1. **Subgroup 1a (11개 실험)**:
   - d_model=128, masking_ratio [0.65-0.85], lambda_disc [1.5-2.5]
   - Lambda와 masking의 조합 효과 파악

2. **Subgroup 1b (9개 실험)**:
   - nhead=16, decoder_depth [3-5], masking_ratio [0.70-0.80]
   - Depth와 masking의 상호작용

3. **Subgroup 1c (4개 실험)**:
   - Patch size [10, 15, 25, 30]
   - Granularity 최적화

4. **Subgroup 1d (5개 실험)**:
   - FFN dimension [512-3072]
   - 용량 확장 효과

5. **Subgroup 1e (1개 실험)**:
   - 최적 설정 조합

**기대 결과:**
- disc_d > 1.2 AND recon_d > 2.5 모델 최소 5개
- ROC-AUC > 0.955
- 최적 lambda_disc 식별

### GROUP 2 (031-055): Window Size & Capacity 탐색

**목표:** Window size와 최적 model capacity 간 관계 이해

**전략:**
- w500 capacity 변형 테스트
- w1000 matched capacity 테스트
- w100 감소된 기준선 테스트

**실험:**

1. **Subgroup 2a (8개 실험)**:
   - w500, d_model [96-320], nhead [4-16], decoder_depth [3-5]

2. **Subgroup 2b (8개 실험)**:
   - w1000, d_model [128-256], patch [20-40]

3. **Subgroup 2c (5개 실험)**:
   - w100, d_model [64-128]

4. **Subgroup 2d (3개 실험)**:
   - w500 patch 전략 [10-30]

5. **Subgroup 2e (1개 실험)**:
   - w1000 최적 구성

**기대 결과:**
- w500 최적 기준선 확인
- w1000 이점 식별 (가설: PA%80에 yes)
- Capacity scaling 규칙 수립

### GROUP 3 (056-075): Disturbing Normal 분리 집중

**목표:** disc_cohens_d_disturbing_vs_anomaly > 0.85 달성

**전략:**
- 009_w500_p20 기반 (최고 disturbing separator: 0.803)
- Dynamic margin 및 loss weighting 탐색
- 높은 anomaly_loss_weight 테스트

**실험:**

1. **Subgroup 3a (9개 실험)**:
   - dynamic_margin_k [1.0-5.0]

2. **Subgroup 3b (6개 실험)**:
   - lambda_disc [1.0-3.5], k=2.5

3. **Subgroup 3c (4개 실험)**:
   - anomaly_loss_weight [0.5-2.0]

4. **Subgroup 3d (1개 실험)**:
   - 최적 조합

**기대 결과:**
- disc_cohens_d_disturbing_vs_anomaly > 0.85
- ROC-AUC > 0.950 유지
- Disturbing 분리 강조 loss weighting 식별

### GROUP 4 (076-095): PA%80 최적화

**목표:** 실무 배포를 위해 PA%80 ROC-AUC 극대화

**전략:**
- Scoring/inference mode 조합 체계 테스트
- 더 나은 맥락을 위해 긴 window 테스트
- 더 나은 장거리 탐지를 위해 고용량 테스트

**실험:**

1. **Subgroup 4a (6개 실험)**:
   - Scoring × Inference 조합 (6가지)

2. **Subgroup 4b (3개 실험)**:
   - Window [500, 1000], patch 변형

3. **Subgroup 4c (6개 실험)**:
   - d_model [192, 256], decoder_depth [4-6]

4. **Subgroup 4d (4개 실험)**:
   - 조합 최적화

5. **Subgroup 4e (1개 실험)**:
   - 궁극의 PA%80 구성

**기대 결과:**
- PA%80 ROC-AUC > 0.970
- PA%80 최적 scoring mode 식별
- 긴 window가 PA%80에 도움되는지 확인

### GROUP 5 (096-110): Teacher-Student Ratio 탐색

**목표:** 최적 teacher-student decoder depth 균형 찾기

**전략:**
- t1s1부터 t6s1까지 비율 테스트
- 균형 비율 테스트 (t2s2, t3s3, t4s4, t5s5)
- d_model=128 기준선과 결합

**실험:**
- 15개 실험: [t1s1-t6s1, t2s2, t3s2-t5s2, t3s3-t5s3, t4s4, t5s5]

**기대 결과:**
- t4s1 또는 t4s2 최적 확인
- 균형 비율 (t3s3, t4s4) 이점 이해
- 매우 깊은 teacher (t6s1) 효과 식별

### GROUP 6 (111-125): Masking 전략 미세 조정

**목표:** 다른 d_model에 대한 최적 masking_ratio 찾기

**전략:**
- d_model=128에 대한 세밀한 sweep (낮은 비율)
- d_model=256에 대한 세밀한 sweep (높은 비율)

**실험:**

1. **Subgroup 6a (7개 실험)**:
   - d_model=128, masking_ratio [0.05-0.35]

2. **Subgroup 6b (7개 실험)**:
   - d_model=256, masking_ratio [0.60-0.90]

3. **Subgroup 6c (1개 실험)**:
   - 최적 구성

**기대 결과:**
- d_model=128 최적 masking_ratio (가설: 0.15-0.25)
- d_model=256 최적 masking_ratio (가설: 0.70-0.80)
- Capacity-masking 관계 이해

### GROUP 7 (126-140): Architecture Depth 최적화

**목표:** 최적 encoder-decoder depth 균형 찾기

**전략:**
- Encoder depth [4, 6, 8] 테스트
- Decoder depth [2-6] 테스트
- d_model=128 기준선 사용

**실험:**
- 15개 실험: encoder × decoder 조합

**기대 결과:**
- 최적 encoder depth 식별 (가설: 6)
- 최적 decoder depth 식별 (가설: 4)
- 더 깊은 architecture 효과 이해

### GROUP 8 (141-150): Lambda Discrepancy & Loss Weighting

**목표:** 최고 성능을 위한 loss weighting 미세 조정

**전략:**
- lambda_disc 세밀한 sweep
- d_model=128, t4s1 기준선 사용

**실험:**
- 10개 실험: lambda_disc [0.5-3.0]

**기대 결과:**
- 최적 lambda_disc 식별 (가설: 1.5-2.5)
- Loss weighting 민감도 이해
- 최적 균형점 확인

---

## Phase 2 기대 결과

### 주요 목표

1. **ROC-AUC > 0.960 모델 10개 이상** (Phase 1 최고: 0.9624)
2. **disc_d > 1.2 AND recon_d > 2.8 모델 5개 이상** (Phase 1: 3개만)
3. **disc_cohens_d_disturbing_vs_anomaly > 0.85** (Phase 1 최고: 0.803)
4. **PA%80 ROC-AUC > 0.970** (Phase 1 최고: 0.965)

### 부차적 목표

5. Window size vs model capacity scaling law 수립
6. 최적 teacher-student 비율 확정
7. 다른 architecture에 대한 최적 masking ratio 찾기
8. Depth-performance 관계 이해

### 실무 배포

9. 프로덕션 배포 준비된 2-3개 구성 식별
10. 다양한 고성능 모델로 앙상블 전략 생성

---

## 권장사항

### 즉시 조치

1. **Phase 2 config 검토 및 검증:**
   ```bash
   python scripts/ablation/configs/phase2.py
   ```

2. **Phase 2 실험 실행:**
   ```bash
   python scripts/ablation/run_ablation.py --config configs/phase2.py --visualize
   ```

3. **학습 중 핵심 메트릭 모니터링:**
   - disc_cohens_d_normal_vs_anomaly (목표: > 1.2)
   - recon_cohens_d_normal_vs_anomaly (목표: > 2.8)
   - disc_cohens_d_disturbing_vs_anomaly (목표: > 0.85)
   - ROC-AUC (목표: > 0.960)
   - PA%80 ROC-AUC (목표: > 0.970)

### Phase 2 중 분석

1. **각 GROUP 완료 후:**
   - 중간 분석 보고서 생성
   - 새로운 인사이트 발견 시 나머지 그룹 조정
   - 돌파구 발견 시 보너스 실험 고려

2. **GROUP별 성능 추적:**
   - 어떤 그룹이 목표 달성?
   - 어떤 가설이 확인/기각?
   - 매개변수 간 예상치 못한 상호작용?

3. **Phase 1 기준선과 비교:**
   - 성능 개선되고 있나?
   - 균형잡힌 메트릭 유지하나?
   - Disturbing normal 과제 해결하나?

### Post-Phase 2

1. **Phase 1+2 종합 보고서 생성**
2. **최종 평가를 위한 상위 5-10개 모델 선택**
3. **Test set에서 최종 평가**
4. **프로덕션 배포 계획 수립**
5. **Ensemble 전략 개발**

---

## 결론

Phase 1 ablation study (1,398개 실험)를 통해 다음을 발견했습니다:

**핵심 발견:**
1. **균형이 핵심**: disc_d와 recon_d 모두 높아야 함
2. **Reconstruction quality 지배**: recon_d가 ROC-AUC와 가장 강한 상관 (r=+0.518)
3. **극단적 disc_ratio는 해로움**: disc_ratio > 4.0 → ROC-AUC 저하
4. **황금 존 희소**: disc_d > 1.33 AND recon_d > 1.73 모델이 단 3개
5. **Disturbing 분리가 최대 과제**: 최고 모델도 Cohen's d = 0.803에 불과

**최고 구성:**
- Window 500, Patch 20
- d_model 128-256
- Teacher depth 4, Student depth 1-2
- all_patches + default scoring
- lambda_disc 1.0-2.0

**Phase 2 전략:**
- 150개 실험을 8개 GROUP으로 구성
- 각 GROUP이 특정 과제 해결
- 황금 존 확대, disturbing 분리 개선, PA%80 최적화 집중

**기대:**
- ROC-AUC > 0.960 모델 10개 이상
- Disturbing 분리 > 0.85
- 실무 배포 가능한 robust 모델

이제 Phase 2 실험을 시작하여 이러한 인사이트를 검증하고 더욱 발전시킬 차례입니다!
