---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: duplessis2014pu
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [pages_added]
card_grade: LIGHT
---
# Analysis of Learning from Positive and Unlabeled Data
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Marthinus C. du Plessis, Gang Niu, Masashi Sugiyama
- Venue: Advances in Neural Information Processing Systems 27 (NIPS 2014), pp.703–711
- 식별자: papers.nips.cc/paper/5509; DBLP conf/nips/PlessisNS14 (DOI 없음)
- [A1 추가] pages 703–711 (DBLP 확인)
- fetch한 페이지: https://papers.nips.cc/paper/5509-analysis-of-learning-from-positive-and-unlabeled-data (NeurIPS 공식 proceedings, 2026-06-11)

## Abstract 전문 (verbatim)
Learning a classifier from positive and unlabeled data is an important class of classification problems that are conceivable in many practical applications. In this paper, we first show that this problem can be solved by cost-sensitive learning between positive and unlabeled data. We then show that convex surrogate loss functions such as the hinge loss may lead to a wrong classification boundary due to an intrinsic bias, but the problem can be avoided by using non-convex loss functions such as the ramp loss. We next analyze the excess risk when the class prior is estimated from data, and show that the classification accuracy is not sensitive to class prior estimation if the unlabeled data is dominated by the positive data (this is naturally satisfied in inlier-based outlier detection because inliers are dominant in the unlabeled dataset). Finally, we provide generalization error bounds and show that, for an equal number of labeled and unlabeled samples, the generalization error of learning only from positive and unlabeled samples is no worse than $2\sqrt{2}$ times the fully supervised case. These theoretical findings are also validated through experiments.

## 역할 (커버 claim)
- C-019: §2.2 단락 1 — PU Learning 일반 정의의 원류 인용.
- C-020: §2.2 단락 1 — PU 계보(비용민감 계열 이론 토대) 괄호 인용.

## 비고
- abstract 내 `$2\sqrt{2}$`는 공식 페이지의 LaTeX 표기 그대로.
