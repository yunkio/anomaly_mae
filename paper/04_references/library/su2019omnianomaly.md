---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: su2019omnianomaly
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "All fields confirmed via DBLP + Semantic Scholar. Pages 2828-2837, DOI confirmed. 'signicantly' artifact confirmed as real PDF extraction issue — not a curator error. Correct word is 'significantly'."
card_grade: LIGHT
abstract_source: semanticscholar-mirror (공식 dl.acm.org 403 차단)
---
# Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, Dan Pei
- Venue: KDD 2019, pp.2828–2837 (쪽수는 scout 목록 기준)
- DOI: 10.1145/3292500.3330672
- DBLP: conf/kdd/SuZNLSP19
- fetch한 페이지: api.semanticscholar.org (DOI 질의, 2026-06-11) — dl.acm.org/doi/10.1145/3292500.3330672 는 403

## Abstract 전문 (verbatim — S2 미러 기준; 공식 페이지 대조는 verifier)
Industry devices (i.e., entities) such as server machines, spacecrafts, engines, etc., are typically monitored with multivariate time series, whose anomaly detection is critical for an entity's service quality management. However, due to the complex temporal dependence and stochasticity of multivariate time series, their anomaly detection remains a big challenge. This paper proposes OmniAnomaly, a stochastic recurrent neural network for multivariate time series anomaly detection that works well robustly for various devices. Its core idea is to capture the normal patterns of multivariate time series by learning their robust representations with key techniques such as stochastic variable connection and planar normalizing flow, reconstruct input data by the representations, and use the reconstruction probabilities to determine anomalies. Moreover, for a detected entity anomaly, OmniAnomaly can provide interpretations based on the reconstruction probabilities of its constituent univariate time series. The evaluation experiments are conducted on two public datasets from aerospace and a new server machine dataset (collected and released by us) from an Internet company. OmniAnomaly achieves an overall F1-Score of 0.86 in three real-world datasets, signicantly outperforming the best performing baseline method by 0.09. The interpretation accuracy for OmniAnomaly is up to 0.89.

## 역할 (커버 claim)
- C-062: §4.1.4 SOTA Legacy baseline 표 출처 (OmniAnomaly).
- C-042: §4.1.1 Table 1 — SMD(Server Machine Dataset) 데이터셋 출처 (baseline+dataset 겸용).
- C-004 / C-012: §1·§2.1 재구성 기반 TSAD 계보 괄호 클러스터.

## 비고
- 모델 약칭: OmniAnomaly. SMD 데이터셋 공개 논문 — 우리 실험은 SMD 28개 machine entity 사용.
- 미러 텍스트의 "signicantly"는 fi-합자 소실 가능성 있는 미러 아티팩트("significantly"가 원문일 가능성) — verifier가 ACM 원문과 철자 단위 대조 필요.
