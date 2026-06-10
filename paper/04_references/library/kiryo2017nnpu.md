---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: kiryo2017nnpu
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [pages_added]
card_grade: LIGHT
---
# Positive-Unlabeled Learning with Non-Negative Risk Estimator
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Ryuichi Kiryo, Gang Niu, Marthinus C. du Plessis, Masashi Sugiyama
- Venue: NIPS 2017, pp.1675–1685 (Oral)
  - [A1 추가] pages 1675–1685 (DBLP 확인)
- 식별자: arXiv 1703.00593 (v1 2017-03-02, v2 2017-11-04); proceedings.neurips.cc hash 7cce53cf…
- fetch한 페이지: https://arxiv.org/abs/1703.00593 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim)
From only positive (P) and unlabeled (U) data, a binary classifier could be trained with PU learning, in which the state of the art is unbiased PU learning. However, if its model is very flexible, empirical risks on training data will go negative, and we will suffer from serious overfitting. In this paper, we propose a non-negative risk estimator for PU learning: when getting minimized, it is more robust against overfitting, and thus we are able to use very flexible models (such as deep neural networks) given limited P data. Moreover, we analyze the bias, consistency, and mean-squared-error reduction of the proposed risk estimator, and bound the estimation error of the resulting empirical risk minimizer. Experiments demonstrate that our risk estimator fixes the overfitting problem of its unbiased counterparts.

## 역할 (커버 claim)
- C-020: §2.2 단락 1 — PU Learning 비용민감형(Non-negative Risk Estimator) 계열 대표 인용.
- C-019: PU 정의 보조.

## 비고
- 통칭: nnPU. NIPS 2017 Oral.
