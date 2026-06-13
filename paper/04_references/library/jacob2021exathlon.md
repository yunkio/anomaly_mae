---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: jacob2021exathlon
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: []
card_grade: LIGHT
---
# Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Vincent Jacob, Fei Song, Arnaud Stiegler, Bijan Rad, Yanlei Diao, Nesime Tatbul
- Venue: PVLDB 14(11):2613–2626, 2021 (scout 목록/R26 D5 기준)
- DOI: 10.14778/3476249.3476307
- arXiv: 2010.05073 (v1 2020-10-10, v3 2021-09-05)
- fetch한 페이지: https://arxiv.org/abs/2010.05073 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim)
Access to high-quality data repositories and benchmarks have been instrumental in advancing the state of the art in many experimental research domains. While advanced analytics tasks over time series data have been gaining lots of attention, lack of such community resources severely limits scientific progress. In this paper, we present Exathlon, the first comprehensive public benchmark for explainable anomaly detection over high-dimensional time series data. Exathlon has been systematically constructed based on real data traces from repeated executions of large-scale stream processing jobs on an Apache Spark cluster. Some of these executions were intentionally disturbed by introducing instances of six different types of anomalous events (e.g., misbehaving inputs, resource contention, process failures). For each of the anomaly instances, ground truth labels for the root cause interval as well as those for the extended effect interval are provided, supporting the development and evaluation of a wide range of anomaly detection (AD) and explanation discovery (ED) tasks. We demonstrate the practical utility of Exathlon's dataset, evaluation methodology, and end-to-end data science pipeline design through an experimental study with three state-of-the-art AD and ED techniques.

## 역할 (커버 claim)
- (claim 행 부재 — scout 목록 명기) §4.1.1 Table 1에 Exathlon 행 작성 시 데이터셋 출처 인용.

## 비고
- 데이터셋 약칭: Exathlon. 우리 실험은 Exathlon 6개 app entity 사용 (run_base_experiments 39 datasets 구성).
- PVLDB 권·호·쪽수는 arXiv 페이지가 아닌 R26/scout 목록 기준 — verifier가 vldb.org에서 재확인.
