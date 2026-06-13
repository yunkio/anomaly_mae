---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: ganin2016dann
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [lambda_schedule_excerpt_resolved, grl_formula_excerpt_resolved]
card_grade: FULL
excerpt_access: abstract_only
---
# Domain-Adversarial Training of Neural Networks

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, Victor Lempitsky
Venue: Journal of Machine Learning Research (JMLR)
연도: 2016
권호: 17(59):1–35
공식 URL: https://jmlr.org/papers/v17/15-239.html
arXiv: 1505.07818

## Abstract 전문 (verbatim)
"We introduce a new representation learning approach for domain adaptation, in which data at training and test time come from similar but different distributions. Our approach is directly inspired by the theory on domain adaptation suggesting that, for effective domain transfer to be achieved, predictions must be made based on features that cannot discriminate between the training (source) and test (target) domains. The approach implements this idea in the context of neural network architectures that are trained on labeled data from the source domain and unlabeled data from the target domain (no labeled target-domain data is necessary). As the training progresses, the approach promotes the emergence of features that are (i) discriminative for the main learning task on the source domain and (ii) indiscriminate with respect to the shift between the domains. We show that this adaptation behaviour can be achieved in almost any feed-forward model by augmenting it with few standard layers and a new gradient reversal layer. The resulting augmented architecture can be trained using standard backpropagation and stochastic gradient descent, and can thus be implemented with little effort using any of the deep learning packages. We demonstrate the success of our approach for two distinct classification problems (document sentiment analysis and image classification), where state-of-the-art domain adaptation performance on standard benchmarks is achieved. We also validate the approach for descriptor learning task in the context of person re-identification application."

## 핵심 발췌 (verbatim, 섹션/위치 표기 — EXCERPT_UNVERIFIED: JMLR PDF 바이너리 디코딩 불가, 이하 발췌는 abstract에서만 추출; 본문 공식 발췌는 verifier 필요)

> "this adaptation behaviour can be achieved in almost any feed-forward model by augmenting it with few standard layers and a new gradient reversal layer." (Abstract)

커버 claim: C-036
활용 맥락: §3.5(C)에서 GRL 메커니즘의 원류를 밝힐 때. "gradient reversal layer는 Ganin et al. (JMLR 2016)에서 도입" 1문장 인용.

---

> "predictions must be made based on features that cannot discriminate between the training (source) and test (target) domains." (Abstract)

커버 claim: C-036
활용 맥락: GRL의 핵심 직관(feature가 domain을 구별하지 못하도록 gradient를 역전)을 설명할 때. 우리 논문에서는 이 domain-discrimination 억제 메커니즘을 anomaly-discrimination 억제에 전용(repurpose)했음을 서술.

---

**[A1 EXCERPT_RESOLVED — JMLR PDF 직접 다운로드 후 발췌 확보 (2026-06-11).]**

**GRL 공식화 (§4.2 / pp.11-12 of PDF):**
> "Mathematically, we can formally treat the gradient reversal layer as a 'pseudo-function' R(x) defined by two (incompatible) equations describing its forward and backpropagation behaviour: R(x) = x (Eq.16), dR/dx = −I (Eq.17), where I is an identity matrix." (§4.2, Eq.16-17)

커버 claim: C-036 (GRL pseudo-function 정의, §4.2 Eq.16-17에서 확보)

---

**λ schedule (§5.2):**
> "The domain adaptation parameter λ is initiated at 0 and is gradually changed to 1 using the following schedule: λ_p = 2/(1+exp(−γ·p)) − 1, where γ was set to 10 in all experiments." (§5.2)

커버 claim: C-036 (λ schedule 수식, §5.2에서 확보; "Ganin-style sigmoid schedule" 표현의 수식 근거)

## 우리 논문에서의 활용

커버 claim: C-036 (+ C-076 동반)

- **§3.5(C) Methodology (GRL)**: C-036 — gradient reversal의 λ_rev schedule "Ganin-style sigmoid schedule 2/(1+exp(−10p))−1" 의 원류 인용 필수. 1문장 + equation에서 [(Ganin et al., 2016)] 괄호 인용.
- **§4.3 Ablation (w/o GRL)**: C-076 동반 — GRL의 능동 억제 효과 ablation 논의 시 원류 인용.

## 주의사항
- GRL은 원래 domain adaptation 목적으로 설계됨 — "GRL을 anomaly detection에 처음 사용"이라는 서술은 사실이 아님. AEGR(Soft Computing 2021, 비지도 network AD) 및 기타 domain-adversarial 계열이 선례로 존재. 우리 논문에서는 "우리는 GRL을 masked-reconstruction self-distillation의 anomaly-overlook 억제에 적용"으로 서술을 제한.
- JMLR PDF 바이너리로 λ schedule 정확한 equation 번호 미확보 — verifier가 논문 §3 또는 §4에서 해당 수식 직접 발췌·equation 번호 확정 필요. 이 card의 schedule 수식은 2차 소스 기반으로 EXCERPT_UNVERIFIED.
- 저자 표기 주의: 8인 저자 중 "Mario March" → "Mario Marchand"가 정확한 이름일 가능성 — verifier 확인.
