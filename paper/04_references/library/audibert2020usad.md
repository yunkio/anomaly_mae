---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: audibert2020usad
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [authors_full_name_Guyard]
card_grade: LIGHT
abstract_source: semanticscholar-mirror (공식 dl.acm.org 403 차단)
---
# USAD: UnSupervised Anomaly Detection on Multivariate Time Series
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Julien Audibert, Pietro Michiardi, Frédéric Guyard, Sébastien Marti, Maria A. Zuluaga
  - [A1 정정] S2 미러의 "F. Guyard" → Crossref/DBLP 공식 "Frédéric Guyard" 확정.
- Venue: KDD 2020, pp.3395–3404 (쪽수는 scout 목록/DBLP 기준)
- DOI: 10.1145/3394486.3403392
- DBLP: conf/kdd/AudibertMGMZ20 / arXiv 없음 (KDD 본이 공식본)
- fetch한 페이지: api.semanticscholar.org (DOI 질의, 2026-06-11) — dl.acm.org/doi/10.1145/3394486.3403392 는 403

## Abstract 전문 (verbatim — S2 미러 기준; 공식 페이지 대조는 verifier)
The automatic supervision of IT systems is a current challenge at Orange. Given the size and complexity reached by its IT operations, the number of sensors needed to obtain measurements over time, used to infer normal and abnormal behaviors, has increased dramatically making traditional expert-based supervision methods slow or prone to errors. In this paper, we propose a fast and stable method called UnSupervised Anomaly Detection for multivariate time series (USAD) based on adversely trained autoencoders. Its autoencoder architecture makes it capable of learning in an unsupervised way. The use of adversarial training and its architecture allows it to isolate anomalies while providing fast training. We study the properties of our methods through experiments on five public datasets, thus demonstrating its robustness, training speed and high anomaly detection performance. Through a feasibility study using Orange's proprietary data we have been able to validate Orange's requirements on scalability, stability, robustness, training speed and high performance.

## 역할 (커버 claim)
- C-059: §4.1.4 SOTA Legacy baseline 표 출처 (USAD).
- C-004 / C-012: §1·§2.1 재구성 기반 TSAD 계보 괄호 클러스터.

## 비고
- 모델 약칭: USAD. 우리 실험에서 Legacy SOTA 비교군.
- 미러 abstract 내 "adversely trained"는 원문 표기 가능성 높음(원논문에도 동일 오기로 알려짐) — verifier가 ACM 원문과 철자 단위 대조.
