---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: lai2023npsr
verification_status: VERIFIED_2CH
verified_by_A: 2026-06-11
verified_by_B: 2026-06-11 (blind export — DBLP conf/nips/LaiSGLB23)
corrected_fields: [authors_Lang_middle_initial]
card_grade: LIGHT
correction_history: |
  2026-06-11 assembler (P4_DIFF_REPORT 해소 ③): 4th author "Jeffrey Lang" → "Jeffrey H. Lang".
  근거: DBLP conf/nips/LaiSGLB23 + NeurIPS proceedings 정본. 본 card의 종전 표기는
  OpenReview API 축약 표기를 그대로 전사한 오류 (A1은 OpenReview 기반으로 '정정 필드 없음'
  판정 — B채널 blind export가 검출). refs.bib 정본은 "Jeffrey H. Lang" / "Duane S. Boning".
---
# Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (2채널 검증 완료 — 2026-06-11)
- 저자 (정본 — DBLP/NeurIPS proceedings): Chih-Yu Lai, Fan-Keng Sun, Zhengqi Gao, **Jeffrey H. Lang**, Duane S. Boning
  - [정정 2026-06-11, diff 해소 ③] 종전 표기 "Jeffrey Lang, Duane S Boning"은 OpenReview API 축약 표기 전사 — DBLP conf/nips/LaiSGLB23 + NeurIPS 정본 기준으로 정정 (P4_DIFF_REPORT.md §3 ③)
- Venue: NeurIPS 2023 poster (venueid NeurIPS.cc/2023/Conference)
- 식별자: OpenReview forum ljgM3vNqfQ; submission number 5605
- fetch한 페이지: https://api2.openreview.net/notes?id=ljgM3vNqfQ (OpenReview 공식 API, 2026-06-11)

## Abstract 전문 (verbatim)
Time series anomaly detection is challenging due to the complexity and variety of patterns that can occur. One major difficulty arises from modeling time-dependent relationships to find contextual anomalies while maintaining detection accuracy for point anomalies. In this paper, we propose a framework for unsupervised time series anomaly detection that utilizes point-based and sequence-based reconstruction models. The point-based model attempts to quantify point anomalies, and the sequence-based model attempts to quantify both point and contextual anomalies. Under the formulation that the observed time point is a two-stage deviated value from a nominal time point, we introduce a nominality score calculated from the ratio of a combined value of the reconstruction errors. We derive an induced anomaly score by further integrating the nominality score and anomaly score, then theoretically prove the superiority of the induced anomaly score over the original anomaly score under certain conditions. Extensive studies conducted on several public datasets show that the proposed framework outperforms most state-of-the-art baselines for time series anomaly detection.

## 역할 (커버 claim)
- C-064: §4.1.4 SOTA New baseline 표 출처 (NPSR).

## 비고
- 모델 약칭: NPSR. 우리 실험에서 New SOTA 비교군.
- 정식 수록: papers.nips.cc hash f1cf02ce09757f57c3b93c0db83181e0 (refs.bib url 필드 — B채널 DBLP export로 확보).
- 인용 시 저자 표기는 refs.bib 정본("Jeffrey H. Lang") 사용 — 본 card 종전 표기 사용 금지.
