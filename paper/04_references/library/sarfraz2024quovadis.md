---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: sarfraz2024quovadis
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via PMLR v235:43461-43476 + arXiv 2405.02678. Abstract verbatim confirmed. arXiv and PMLR abstracts match."
card_grade: FULL
---
# Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: M. Saquib Sarfraz, Mei-Yen Chen, Lukas Layer, Kunyu Peng, Marios Koulakis
Venue: ICML 2024 (Proceedings of Machine Learning Research)
연도: 2024
권호: PMLR v235:43461–43476
공식 URL: https://proceedings.mlr.press/v235/sarfraz24a.html
arXiv: 2405.02678
GitHub: https://github.com/ssarfraz/QuoVadisTAD

## Abstract 전문 (verbatim)
"The current state of machine learning scholarship in Timeseries Anomaly Detection (TAD) is plagued by the persistent use of flawed evaluation metrics, inconsistent benchmarking practices, and a lack of proper justification for the choices made in novel deep learning-based model designs. Our paper presents a critical analysis of the status quo in TAD, revealing the misleading track of current research and highlighting problematic methods, and evaluation practices. Our position advocates for a shift in focus from solely pursuing novel model designs to improving benchmarking practices, creating non-trivial datasets, and critically evaluating the utility of complex methods against simpler baselines. Our findings demonstrate the need for rigorous evaluation protocols, the creation of simple baselines, and the revelation that state-of-the-art deep anomaly detection models effectively learn linear mappings. These findings suggest the need for more exploration and development of simple and interpretable TAD methods. The increment of model complexity in the state-of-the-art deep-learning based models unfortunately offers very little improvement. We offer insights and suggestions for the field to move forward."

## 핵심 발췌 (verbatim, 섹션/위치 표기)

> "The current state of machine learning scholarship in Timeseries Anomaly Detection (TAD) is plagued by the persistent use of flawed evaluation metrics, inconsistent benchmarking practices, and a lack of proper justification for the choices made in novel deep learning-based model designs." (Abstract)

커버 claim: C-054, C-055, C-056
활용 맥락: §4.1.4 baseline 출처로 QuoVadisTAD를 인용할 때 이 논문의 동기를 1문장 설명.

---

> "state-of-the-art deep anomaly detection models effectively learn linear mappings." (Abstract)

커버 claim: C-054, C-055
활용 맥락: 복잡한 deep TAD 모델들이 결국 선형 매핑에 수렴한다는 이 논문의 핵심 발견. §4.1.4에서 simple/neural baseline을 포함한 비교 실험의 중요성을 정당화할 때.

---

> "The increment of model complexity in the state-of-the-art deep-learning based models unfortunately offers very little improvement." (Abstract)

커버 claim: C-054, C-055
활용 맥락: 단순 baseline 대비 복잡 모델의 개선폭이 미미함을 지적. §4.1.4에서 simple baseline 포함 이유 설명 시.

---

> "Our position advocates for a shift in focus from solely pursuing novel model designs to improving benchmarking practices, creating non-trivial datasets, and critically evaluating the utility of complex methods against simpler baselines." (Abstract)

커버 claim: C-054, C-055, C-056
활용 맥락: 우리 논문의 baseline 비교 설계(simple 5종 + neural 3종 + GCN-LSTM 포함)가 이 입장 논문의 권고를 따른다는 맥락에서 인용 가능.

## 우리 논문에서의 활용

커버 claim: C-054, C-055, C-056

- **§4.1.4 Baselines**: C-054, C-055, C-056 — simple 5종(random, sensor_range, pca_error, l2_norm, nn_distance), neural 3종(MLP, MLPMixer, Transformer), GCN-LSTM 모두 이 논문에서 도입. baseline 표 각주 또는 §4.1.4 본문에서 "following Sarfraz et al. (ICML 2024)" 표기.
- 주의: GCN-LSTM은 별도 원논문 없음(QuoVadisTAD-introduced) — 이 논문 + repo `ssarfraz/QuoVadisTAD` 표기.

## 주의사항
- 이 논문은 "Position paper" — 실험적 주장이 position/advocacy 형식으로 제시됨. 우리 논문에서 사실 주장(fact claim)처럼 인용하지 말고, baseline 출처 인용 또는 TAD 분야 비판적 관점 소개 수준으로 제한.
- simple baseline과 neural baseline 각각의 정확한 정의(수식 수준)는 본문 §X에 있으나 PMLR HTML 접근 시 세부 발췌 불가 — verifier가 PMLR 또는 arXiv 2405.02678에서 baseline 정의 섹션 발췌 보강 필요.
- arXiv 2405.02678이 PMLR 게재본과 동일한지 확인 필요 (position paper는 arXiv와 proceedings 버전이 다를 수 있음).
