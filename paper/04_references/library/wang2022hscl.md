---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: wang2022hscl
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
card_grade: LIGHT
corrections:
  - field: pages
    wrong: missing
    correct: "110–128"
    severity: MINOR
    confirmed_by: "DBLP conf/eccv/WangZWSN22"
  - field: venue_detail
    wrong: "ECCV 2022 계열 [scout 목록상 verifier-TODO]"
    correct: "ECCV 2022, LNCS 13685 (Part XXV)"
    severity: MINOR
    confirmed_by: "DBLP conf/eccv/WangZWSN22; DOI 10.1007/978-3-031-19806-9_7"
---
# Hierarchical Semi-Supervised Contrastive Learning for Contamination-Resistant Anomaly Detection
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Gaoang Wang, Yibing Zhan, Xinchao Wang, Mingli Song, Klara Nahrstedt
- Venue: ECCV 2022, LNCS 13685 (Part XXV), pp. 110–128 [VERIFIED_A: DOI + pages confirmed via DBLP]
- Springer DOI: 10.1007/978-3-031-19806-9_7
- arXiv: 2207.11789 (v1 2022-07-24)
- fetch한 페이지: https://arxiv.org/abs/2207.11789 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim)
Anomaly detection aims at identifying deviant samples from the normal data distribution. Contrastive learning has provided a successful way to sample representation that enables effective discrimination on anomalies. However, when contaminated with unlabeled abnormal samples in training set under semi-supervised settings, current contrastive-based methods generally 1) ignore the comprehensive relation between training data, leading to suboptimal performance, and 2) require fine-tuning, resulting in low efficiency. To address the above two issues, in this paper, we propose a novel hierarchical semi-supervised contrastive learning (HSCL) framework, for contamination-resistant anomaly detection. Specifically, HSCL hierarchically regulates three complementary relations: sample-to-sample, sample-to-prototype, and normal-to-abnormal relations, enlarging the discrimination between normal and abnormal samples with a comprehensive exploration of the contaminated data. Besides, HSCL is an end-to-end learning approach that can efficiently learn discriminative representations without fine-tuning. HSCL achieves state-of-the-art performance in multiple scenarios, such as one-class classification and cross-dataset detection. Extensive ablation studies further verify the effectiveness of each considered relation.

## 역할 (커버 claim)
- C-032 (선택): §3.1 — "contaminated semi-supervised" 신조어 정의 각주에서 인접 용어 "contamination-resistant"(HSCL, 이미지 도메인)와의 구분 괄호 인용 (LIGHT-optional).

## 비고
- 모델 약칭: HSCL. 이미지 도메인 — 용어 구분 각주 전용, 비교 실험 대상 아님. ECCV 2022 수록 여부·쪽수는 verifier가 Springer 페이지에서 확정.
