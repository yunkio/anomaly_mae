---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: huang2022slavae
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [abstract_obtained]
card_grade: FULL
excerpt_access: abstract_only
---
# A Semi-Supervised VAE Based Active Anomaly Detection Framework in Multivariate Time Series for Online Systems

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: Tao Huang, Pengfei Chen, Ruipeng Li
Venue: WWW 2022 (The Web Conference 2022)
연도: 2022
쪽수: pp.1797–1806
DOI: 10.1145/3485447.3511984
DBLP: conf/www/HuangCL22
공식 URL: https://dl.acm.org/doi/10.1145/3485447.3511984
GitHub: https://github.com/shendu-ht/SLA-VAE

## Abstract 전문 (verbatim — Semantic Scholar API 2026-06-11, doi 질의)
[A1 RESOLVED — abstract 확보. ACM DL 403 대체: Semantic Scholar API (DOI 10.1145/3485447.3511984 질의)]

"Nowadays, the large online systems are constructed on the basis of microservice architecture. A failure in this architecture may cause a series of failures due to the fault propagation. Thus, the large online systems need to be monitored comprehensively to ensure the service quality. Even though many anomaly detection techniques have been proposed, few of them can be directly applied to a given microservice or cloud server in industrial environment. To settle these challenges, this paper presents SLA-VAE, a semi-supervised learning based active anomaly detection framework using variational auto-encoder. SLA-VAE first defines anomalies based on feature extraction module, introduces semi-supervised VAE to identify anomalies in multivariate time series, and employs active learning to update the online model via a small number of uncertain samples. We conduct experiments on the cloud server data from two different types of game business in Tencent. The results show that SLA-VAE significantly outperforms other state-of-the-art methods and is suitable for wide deployment in large online business system."

C-011/C-025 관련 판정: SLA-VAE는 active learning 루프 기반으로 labeled anomaly를 활용하며, 표현 학습 gradient에 labeled anomaly가 직접 adversarial 방식으로 개입하는 메커니즘은 abstract에서 확인되지 않음. "semi-supervised VAE" + "active learning" 조합으로 운용 — 우리의 masked-reconstruction + GRL adversarial 통합과 구조적으로 다름. 반증 성립 약함.

## 핵심 발췌 (verbatim — EXCERPT_UNVERIFIED: 본문/abstract 접근 불가)

지지 발췌 없음.

C-011/C-025(최초성 차별화) 관련:
- SLA-VAE가 labeled anomaly를 VAE 학습에 직접 통합하는 방식의 구체적 메커니즘 미확인. active learning 루프 + semi-supervised VAE 조합으로 알려져 있으나, 표현 학습(representation learning) gradient에 label이 직접 개입하는지는 verifier 정독 필요.
- C-011/C-025 최초성 반증 후보 ②로 지목된 논문. 반증 성립 여부는 verifier 확인 필수.

## 우리 논문에서의 활용

커버 claim: C-011, C-025 (최초성 차별화 — 반증 후보)

- **§2.2 Related Work (포지셔닝)**: C-011, C-025 — 우리의 최초성 주장("masked-reconstruction self-distillation + GRL adversarial 통합")을 좁힐 때 SLA-VAE와의 차별화 인용. "SLA-VAE는 active learning 루프를 통해 labeled anomaly를 활용하지만, 표현 학습 gradient에 직접 adversarial 통합하지 않는다"는 방향으로 차별화 서술 (verifier 정독 후 확정).

## 주의사항
- SLA-VAE는 online systems / KPI 모니터링 도메인 — 우리의 일반적 MTSAD 벤치마크와 다른 응용 영역. 차별화 서술 시 도메인 차이도 한 줄 부기.
- active learning 루프 의존이라는 중요한 구조적 차이: SLA-VAE는 반복적 인간 라벨링 루프를 전제 → 우리 모델의 fixed labeled-anomaly 설정과 다름. 이 차이가 반증을 약화시키는 핵심 논거일 수 있음.
- abstract 전문 미확보 — EXCERPT_UNVERIFIED 상태. verifier 필수 작업.
- C-011/C-025 최초성 주장 재서술 전까지 이 논문을 "반증 후보"로 명시하고 단독 인용 금지. 차별화 인용으로만 활용.
