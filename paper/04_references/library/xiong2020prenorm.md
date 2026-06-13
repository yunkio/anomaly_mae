---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: xiong2020prenorm
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via PMLR v119:10524-10533. Author name 'Tieyan Liu' confirmed as PMLR official spelling (no hyphen). Abstract verbatim confirmed."
card_grade: LIGHT
---
# On Layer Normalization in the Transformer Architecture
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tieyan Liu
- Venue: ICML 2020 — Proceedings of the 37th International Conference on Machine Learning, PMLR 119:10524–10533, 2020
- 식별자: proceedings.mlr.press/v119/xiong20b.html; arXiv 2002.04745
- fetch한 페이지: https://proceedings.mlr.press/v119/xiong20b.html (PMLR 공식, 2026-06-11)

## Abstract 전문 (verbatim)
The Transformer is widely used in natural language processing tasks. To train a Transformer however, one usually needs a carefully designed learning rate warm-up stage, which is shown to be crucial to the final performance but will slow down the optimization and bring more hyper-parameter tunings. In this paper, we first study theoretically why the learning rate warm-up stage is essential and show that the location of layer normalization matters. Specifically, we prove with mean field theory that at initialization, for the original-designed Post-LN Transformer, which places the layer normalization between the residual blocks, the expected gradients of the parameters near the output layer are large. Therefore, using a large learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful for avoiding this problem. On the other hand, our theory also shows that if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization. This motivates us to remove the warm-up stage for the training of Pre-LN Transformers. We show in our experiments that Pre-LN Transformers without the warm-up stage can reach comparable results with baselines while requiring significantly less training time and hyper-parameter tuning on a wide range of applications.

## 역할 (커버 claim)
- C-039 / C-085: §3.4 — Transformer Encoder의 Pre-Norm(Pre-LN) 학습 안정성 일반 근거 (방법론 섹션 기법 선례).

## 비고
- CLAIM_CITATION_MAP 주의사항 준수: 원논문은 NLP/일반 Transformer 대상 — "시계열 한정" 서술 금지, "Pre-LN의 학습 안정성" 일반 근거로만 인용.
- 저자명 PMLR 표기는 "Tieyan Liu" (하이픈 없음) — 최종 표기는 verifier 확정.
