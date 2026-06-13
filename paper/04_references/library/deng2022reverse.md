---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: deng2022reverse
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [pages, doi]
card_grade: LIGHT
---
# Anomaly Detection via Reverse Distillation from One-Class Embedding
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Hanqiu Deng, Xingyu Li
- Venue: CVPR 2022, pp.9727–9736
- 식별자: arXiv 2201.10703 (v1 2022-01-26, v2 2022-03-23); DOI: 10.1109/CVPR52688.2022.00951
- [A1 추가] pages 9727–9736, DOI 10.1109/CVPR52688.2022.00951 (DBLP 확인)
- fetch한 페이지: https://arxiv.org/abs/2201.10703 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim)
Knowledge distillation (KD) achieves promising results on the challenging problem of unsupervised anomaly detection (AD). The representation discrepancy of anomalies in the teacher-student (T-S) model provides essential evidence for AD. However, using similar or identical architectures to build the teacher and student models in previous studies hinders the diversity of anomalous representations. To tackle this problem, we propose a novel T-S model consisting of a teacher encoder and a student decoder and introduce a simple yet effective "reverse distillation" paradigm accordingly. Instead of receiving raw images directly, the student network takes teacher model's one-class embedding as input and targets to restore the teacher's multiscale representations. Inherently, knowledge distillation in this study starts from abstract, high-level presentations to low-level features. In addition, we introduce a trainable one-class bottleneck embedding (OCBE) module in our T-S model. The obtained compact embedding effectively preserves essential information on normal patterns, but abandons anomaly perturbations. Extensive experimentation on AD and one-class novelty detection benchmarks shows that our method surpasses SOTA performance, demonstrating our proposed approach's effectiveness and generalizability.

## 역할 (커버 claim)
- C-027: §2.3 단락 2 — knowledge distillation 기반 AD 계보 괄호 인용 (Bergmann et al.과 클러스터).

## 비고
- 통칭: Reverse Distillation (RD4AD). SCOUT §D 오류 정정 ④의 "Wang et al. CVPR 2021" 대체재로 채택된 후보.
