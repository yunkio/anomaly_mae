---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: bergmann2020uninformed
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [pages, doi]
card_grade: LIGHT
---
# Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger
- Venue: CVPR 2020, pp.4182–4191
- 식별자: arXiv 1911.02357 (v1 2019-11-06, v2 2020-03-18); DOI: 10.1109/CVPR42600.2020.00424
- [A1 추가] pages 4182–4191, DOI 10.1109/CVPR42600.2020.00424 (DBLP 확인)
- fetch한 페이지: https://arxiv.org/abs/1911.02357 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim)
We introduce a powerful student-teacher framework for the challenging problem of unsupervised anomaly detection and pixel-precise anomaly segmentation in high-resolution images. Student networks are trained to regress the output of a descriptive teacher network that was pretrained on a large dataset of patches from natural images. This circumvents the need for prior data annotation. Anomalies are detected when the outputs of the student networks differ from that of the teacher network. This happens when they fail to generalize outside the manifold of anomaly-free training data. The intrinsic uncertainty in the student networks is used as an additional scoring function that indicates anomalies. We compare our method to a large number of existing deep learning based methods for unsupervised anomaly detection. Our experiments demonstrate improvements over state-of-the-art methods on a number of real-world datasets, including the recently introduced MVTec Anomaly Detection dataset that was specifically designed to benchmark anomaly segmentation algorithms.

## 역할 (커버 claim)
- C-027: §2.3 단락 2 — knowledge distillation 기반 AD 계보(teacher-student 격차를 이상 신호로 쓰는 계열) 괄호 인용.

## 비고
- 통칭: Uninformed Students (US). 이미지 도메인 — 우리 self-distillation 구도와의 차별화 맥락에서 인용.
