---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: yang2023dcdetector
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via arXiv + DBLP. KDD 2023, pages 3033-3045, DOI 10.1145/3580305.3599295 confirmed. Abstract verbatim confirmed."
card_grade: LIGHT
---
# DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, Liang Sun
- Venue: KDD 2023 (arXiv comments: "Accepted by ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD 2023)")
- DOI: 10.1145/3580305.3599295 (arXiv 페이지 related DOI로 표시)
- arXiv: 2306.10347 (v1 2023-06-17, v2 2023-10-11)
- fetch한 페이지: https://arxiv.org/abs/2306.10347 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim)
Time series anomaly detection is critical for a wide range of applications. It aims to identify deviant samples from the normal sample distribution in time series. The most fundamental challenge for this task is to learn a representation map that enables effective discrimination of anomalies. Reconstruction-based methods still dominate, but the representation learning with anomalies might hurt the performance with its large abnormal loss. On the other hand, contrastive learning aims to find a representation that can clearly distinguish any instance from the others, which can bring a more natural and promising representation for time series anomaly detection. In this paper, we propose DCdetector, a multi-scale dual attention contrastive representation learning model. DCdetector utilizes a novel dual attention asymmetric design to create the permutated environment and pure contrastive loss to guide the learning process, thus learning a permutation invariant representation with superior discrimination abilities. Extensive experiments show that DCdetector achieves state-of-the-art results on multiple time series anomaly detection benchmark datasets. Code is publicly available at https://github.com/DAMO-DI-ML/KDD2023-DCdetector.

## 역할 (커버 claim)
- C-066: §4.1.4 SOTA New baseline 표 출처 (DCdetector).
- C-004 / C-014 / C-016: §1·§2.1 연관/대조 기반 TSAD 계보 괄호 클러스터.
- (보조) C-053: AR threshold 관행 추종 사례 — 단 R30 보류 중 (발췌 확보 전 사용 금지, FULL 측 Anomaly Transformer 카드 참조).

## 비고
- 모델 약칭: DCdetector. 우리 실험에서 New SOTA 비교군.
