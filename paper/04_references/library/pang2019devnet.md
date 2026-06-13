---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: pang2019devnet
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed. Pages 353-362 and DOI 10.1145/3292500.3330871 added (were verifier-TODO). Abstract confirmed from arXiv."
card_grade: LIGHT
---
# Deep Anomaly Detection with Deviation Networks
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Guansong Pang, Chunhua Shen, Anton van den Hengel
- Venue: KDD 2019 (arXiv comments: "10 Pages, Published in KDD19")
- Pages: 353–362 [VERIFIED_A: confirmed via DBLP]
- 식별자: arXiv 1911.08623 (v1 2019-11-19); ACM DOI 10.1145/3292500.3330871 [VERIFIED_A: confirmed via DBLP]
- fetch한 페이지: https://arxiv.org/abs/1911.08623 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim)
Although deep learning has been applied to successfully address many data mining problems, relatively limited work has been done on deep learning for anomaly detection. Existing deep anomaly detection methods, which focus on learning new feature representations to enable downstream anomaly detection methods, perform indirect optimization of anomaly scores, leading to data-inefficient learning and suboptimal anomaly scoring. Also, they are typically designed as unsupervised learning due to the lack of large-scale labeled anomaly data. As a result, they are difficult to leverage prior knowledge (e.g., a few labeled anomalies) when such information is available as in many real-world anomaly detection applications. This paper introduces a novel anomaly detection framework and its instantiation to address these problems. Instead of representation learning, our method fulfills an end-to-end learning of anomaly scores by a neural deviation learning, in which we leverage a few (e.g., multiple to dozens) labeled anomalies and a prior probability to enforce statistically significant deviations of the anomaly scores of anomalies from that of normal data objects in the upper tail. Extensive results show that our method can be trained substantially more data-efficiently and achieves significantly better anomaly scoring than state-of-the-art competing methods.

## 역할 (커버 claim)
- C-021: §2.2 단락 2 — 비시계열 영역의 few-labeled-anomaly(semi-supervised) AD 적용 사례 괄호 인용 (+C-011/025 최초성 차별화 보조).

## 비고
- 모델 약칭: DevNet. 이미지/표 형식 데이터 도메인 — 시계열 아님(차별화 맥락 유지).
