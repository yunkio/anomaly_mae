---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: deng2021gdn
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: []
card_grade: LIGHT
---
# Graph Neural Network-Based Anomaly Detection in Multivariate Time Series
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Ailin Deng (National University of Singapore), Bryan Hooi (National University of Singapore)
- Venue: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35, No. 5 (AAAI-21 Technical Tracks 5), pp.4027–4035; 게재일 2021-05-18
- DOI: 10.1609/aaai.v35i5.16523
- fetch한 페이지: https://ojs.aaai.org/index.php/AAAI/article/view/16523 (AAAI 공식, 2026-06-11)

## Abstract 전문 (verbatim)
Given high-dimensional time series data (e.g., sensor data), how can we detect anomalous events, such as system faults and attacks? More challengingly, how can we do this in a way that captures complex inter-sensor relationships, and detects and explains anomalies which deviate from these relationships? Recently, deep learning approaches have enabled improvements in anomaly detection in high-dimensional datasets; however, existing methods do not explicitly learn the structure of existing relationships between variables, or use them to predict the expected behavior of time series. Our approach combines a structure learning approach with graph neural networks, additionally using attention weights to provide explainability for the detected anomalies. Experiments on two real-world sensor datasets with ground truth anomalies show that our method detects anomalies more accurately than baseline approaches, accurately captures correlations between sensors, and allows users to deduce the root cause of a detected anomaly.

## 역할 (커버 claim)
- C-061: §4.1.4 SOTA Legacy baseline 표 출처 (GDN).
- C-004 / C-013: §1·§2.1 예측 기반 TSAD 계보 괄호 클러스터.

## 비고
- 모델 약칭: GDN (Graph Deviation Network). 우리 실험에서 Legacy SOTA 비교군.
