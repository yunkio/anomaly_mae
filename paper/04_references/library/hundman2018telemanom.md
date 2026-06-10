---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: hundman2018telemanom
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: [pages_added, author_diacritical]
card_grade: LIGHT
---
# Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, Tom Söderström
  - [A1 정정] "Soderstrom" → "Söderström" (DBLP 공식 표기)
- Venue: KDD 2018, pp.387–395
  - [A1 추가] pages 387–395 (Crossref/DBLP 확인)
- DOI: 10.1145/3219819.3219845
- arXiv: 1802.04431 (v1 2018-02-13, v3 2018-06-06)
- fetch한 페이지: https://arxiv.org/abs/1802.04431 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim)
As spacecraft send back increasing amounts of telemetry data, improved anomaly detection systems are needed to lessen the monitoring burden placed on operations engineers and reduce operational risk. Current spacecraft monitoring systems only target a subset of anomaly types and often require costly expert knowledge to develop and maintain due to challenges involving scale and complexity. We demonstrate the effectiveness of Long Short-Term Memory (LSTMs) networks, a type of Recurrent Neural Network (RNN), in overcoming these issues using expert-labeled telemetry anomaly data from the Soil Moisture Active Passive (SMAP) satellite and the Mars Science Laboratory (MSL) rover, Curiosity. We also propose a complementary unsupervised and nonparametric anomaly thresholding approach developed during a pilot implementation of an anomaly detection system for SMAP, and offer false positive mitigation strategies along with other key improvements and lessons learned during development.

## 역할 (커버 claim)
- C-044: §4.1.1 Table 1 — SMAP·MSL 데이터셋 출처 (실험 섹션 전용).
- (보조) C-001: 우주 telemetry 응용 중요성 보강 인용 가능.

## 비고
- 통칭: Telemanom (LSTM-NDT). SMAP/MSL 데이터셋 공개 논문 — 데이터셋 표 인용 전용.
