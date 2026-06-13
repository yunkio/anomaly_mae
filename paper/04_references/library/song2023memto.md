---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: song2023memto
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via arXiv + OpenReview UFW67uduJd + DBLP. NeurIPS 2023 confirmed. Abstract verbatim confirmed."
card_grade: LIGHT
---
# MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho
- Venue: NeurIPS 2023
- 식별자: arXiv 2312.02530 (v1 2023-12-05); papers.nips.cc hash b4c898eb… (scout 목록); OpenReview UFW67uduJd (R26 truth [B12])
- fetch한 페이지: https://arxiv.org/abs/2312.02530 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim)
Detecting anomalies in real-world multivariate time series data is challenging due to complex temporal dependencies and inter-variable correlations. Recently, reconstruction-based deep models have been widely used to solve the problem. However, these methods still suffer from an over-generalization issue and fail to deliver consistently high performance. To address this issue, we propose the MEMTO, a memory-guided Transformer using a reconstruction-based approach. It is designed to incorporate a novel memory module that can learn the degree to which each memory item should be updated in response to the input data. To stabilize the training procedure, we use a two-phase training paradigm which involves using K-means clustering for initializing memory items. Additionally, we introduce a bi-dimensional deviation-based detection criterion that calculates anomaly scores considering both input space and latent space. We evaluate our proposed method on five real-world datasets from diverse domains, and it achieves an average anomaly detection F1-score of 95.74%, significantly outperforming the previous state-of-the-art methods. We also conduct extensive experiments to empirically validate the effectiveness of our proposed model's key components.

## 역할 (커버 claim)
- C-067: §4.1.4 SOTA New baseline 표 출처 (MEMTO).
- C-004 / C-012: §1·§2.1 재구성 기반 TSAD 계보 괄호 클러스터.

## 비고
- 모델 약칭: MEMTO. 우리 실험에서 New SOTA 비교군.
- arXiv 페이지 제목 표기는 "Memory-guided" (소문자 g) — papers.nips.cc 최종 표기와의 대소문자 차이는 verifier 확정.
