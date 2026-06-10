---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: lee2021wetas
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [pages_added]
card_grade: FULL
---
# Weakly Supervised Temporal Anomaly Segmentation with Dynamic Time Warping

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: Dongha Lee, Sehun Yu, Hyunjun Ju, Hwanjo Yu
Venue: ICCV 2021 (IEEE/CVF International Conference on Computer Vision)
연도: 2021
DOI: 10.1109/ICCV48922.2021.00726 [A1 확인 — DBLP 일치]
Pages: 7335–7344 [A1 추가 — DBLP 확인]
arXiv: 2108.06816
공식 URL: https://arxiv.org/abs/2108.06816 (ICCV 2021 Open Access: https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_Weakly_Supervised_Temporal_Anomaly_Segmentation_With_Dynamic_Time_Warping_ICCV_2021_paper.pdf)

## Abstract 전문 (verbatim)
"Most recent studies on detecting and localizing temporal anomalies have mainly employed deep neural networks to learn the normal patterns of temporal data in an unsupervised manner. Unlike them, the goal of our work is to fully utilize instance-level (or weak) anomaly labels, which only indicate whether any anomalous events occurred or not in each instance of temporal data. In this paper, we present WETAS, a novel framework that effectively identifies anomalous temporal segments (i.e., consecutive time points) in an input instance. WETAS learns discriminative features from the instance-level labels so that it infers the sequential order of normal and anomalous segments within each instance, which can be used as a rough segmentation mask. Based on the dynamic time warping (DTW) alignment between the input instance and its segmentation mask, WETAS obtains the result of temporal segmentation, and simultaneously, it further enhances itself by using the mask as additional supervision."

## 핵심 발췌 (verbatim, 섹션/위치 표기)

> "the goal of our work is to fully utilize instance-level (or weak) anomaly labels, which only indicate whether any anomalous events occurred or not in each instance of temporal data." (Abstract)

커버 claim: C-023, C-071
활용 맥락: §2.2에서 weakly-supervised 계열을 소개할 때. instance-level label = "이상이 발생했는가" 여부만 아는 약한 지도 신호.

---

> "WETAS learns discriminative features from the instance-level labels so that it infers the sequential order of normal and anomalous segments within each instance." (Abstract)

커버 claim: C-023
활용 맥락: §2.2에서 WETAS의 학습 방식을 설명. weak label이 분류 목적함수의 지도 신호로 직접 사용됨 — 우리의 masked-reconstruction self-distillation pretext와 구조적으로 다름.

---

> "WETAS considers two different types of losses: the classification loss for correctly classifying an input instance as its instance-level anomaly label, and the alignment loss for matching the input instance with the sequential anomaly label" (§ 방법론 — 2차 소스 기반 요약, 정확한 section 번호는 verifier 확인 필요)

커버 claim: C-023
활용 맥락: §2.2에서 WETAS의 손실 함수 구조를 설명하며 차별화. WETAS는 classification + alignment loss로 학습 → 표현 학습 pretext가 없음. 우리 모델은 self-supervised masking pretext + adversarial GRL로 구분.

---

> "Based on the dynamic time warping (DTW) alignment between the input instance and its segmentation mask, WETAS obtains the result of temporal segmentation." (Abstract)

커버 claim: C-023, C-071
활용 맥락: DTW alignment를 통한 temporal segmentation — 이 방법론은 시계열 패턴 정렬에 의존. 우리 패치 기반 접근과의 차이를 §2.2에서 1문장으로 정리.

## 우리 논문에서의 활용

커버 claim: C-023, C-071

- **§2.2 Related Work (weakly-supervised 계열)**: C-023 — DeepMIL, WETAS, TreeMIL을 "weak label이 분류/정렬 목적함수의 지도 신호로 직접 사용되는 계열"로 묶어 소개. WETAS는 instance-level label + DTW alignment loss.
- **§4.1.4 Baselines**: C-071 — baseline 출처 인용.

## 주의사항
- WETAS는 비디오/visual 시계열 도메인에서 제안됨 — 범용 multivariate TSAD와 직접 비교 시 도메인 차이 명시. 우리 논문에서는 "temporal weakly-supervised 계열의 대표 사례"로 소개하되 도메인 차이를 한 줄 부기.
- venue 정정 기록: ICML 2021 추정 → ICCV 2021 (scout 2026-06-11 정정; arXiv 페이지 "accepted to ICCV 2021" 명시 확인). DOI 10.1109/ICCV48922.2021.00726은 R26 truth — verifier 재확인 필요.
- WETAS의 차별점 서술: "self-supervised pretext 없이 label이 목적함수" — 이것이 우리 TSMAE와의 핵심 구분. 유사하게 label을 쓰지만 방식이 다름.
