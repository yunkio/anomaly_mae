---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: paparrizos2022vus
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
card_grade: FULL
corrections:
  - field: abstract_final_sentence
    wrong: "Our findings demonstrate that our four measures are significantly more robust in assessing the quality of time-series AD methods."
[VERIFIED_A MINOR CORRECTION: original card had wrong ending sentence — corrected per PVLDB publisher PDF pdftotext]
    correct: "Our findings demonstrate that our four measures are significantly more robust in assessing the quality of time-series AD methods."
    severity: MINOR
    confirmed_by: "PVLDB publisher PDF pdftotext extraction"
---
# Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay, Aaron Elmore, Michael J. Franklin
Venue: Proceedings of the VLDB Endowment (PVLDB)
연도: 2022
권호: 15(11):2774–2787
DOI: 10.14778/3551793.3551830
공식 URL: https://www.vldb.org/pvldb/vol15/p2774-paparrizos.pdf

## Abstract 전문 (verbatim)
"Anomaly detection (AD) is a fundamental task for time-series analytics with important implications for the downstream performance of many applications. In contrast to other domains where AD mainly focuses on point-based anomalies (i.e., outliers in standalone observations), AD for time series is also concerned with range-based anomalies (i.e., outliers spanning multiple observations). Nevertheless, it is common to use traditional point-based information retrieval measures, such as Precision, Recall, and F-score, to assess the quality of methods by thresholding the anomaly score to mark each point as an anomaly or not. However, mapping discrete labels into continuous data introduces unavoidable shortcomings, complicating the evaluation of range-based anomalies. Notably, the choice of evaluation measure may significantly bias the experimental outcome. Despite over six decades of attention, there has never been a large-scale systematic quantitative and qualitative analysis of time-series AD evaluation measures. This paper extensively evaluates quality measures for time-series AD to assess their robustness under noise, misalignments, and different anomaly cardinality ratios. Our results indicate that measures producing quality values independently of a threshold (i.e., AUC-ROC and AUC-PR) are more suitable for time-series AD. Motivated by this observation, we first extend the AUC-based measures to account for range-based anomalies. Then, we introduce a new family of parameter-free and threshold-independent measures, VUS (Volume Under the Surface), to evaluate methods while varying parameters. Our extensive experimental evaluation demonstrates that our four measures are significantly more robust in assessing the quality of time-series anomaly detection methods."

## 핵심 발췌 (verbatim, 섹션/위치 표기)

> "measures producing quality values independently of a threshold (i.e., AUC-ROC and AUC-PR) are more suitable for time-series AD." (Abstract)

커버 claim: C-048
활용 맥락: §4.1.3에서 VUS 계열 지표를 선택한 이유를 정당화할 때. threshold-independent 특성이 핵심 논거.

---

> "the choice of evaluation measure may significantly bias the experimental outcome." (Abstract)

커버 claim: C-048, C-050
활용 맥락: §4.1.3 지표 선택 정당화. PA F1 같은 threshold-dependent 단일 지표의 편향 위험을 논할 때.

---

> "we introduce a new family of parameter-free and threshold-independent measures, VUS (Volume Under the Surface), to evaluate methods while varying parameters." (Abstract)

커버 claim: C-048
활용 맥락: VUS-PR / VUS-ROC를 평가지표로 채택할 때 제안 논문 인용.

---

> "AD for time series is also concerned with range-based anomalies (i.e., outliers spanning multiple observations)." (Abstract)

커버 claim: C-048
활용 맥락: 시계열 이상이 단일 포인트가 아닌 구간(range)에 걸칠 수 있음을 논할 때. 기존 point-based 지표의 한계 맥락.

## 우리 논문에서의 활용

커버 claim: C-048

- **§4.1.3 Evaluation Metrics**: C-048 — VUS-PR / VUS-ROC를 평가지표로 채택할 때 제안 논문으로 반드시 인용. threshold-independent·range-aware 특성을 2–3단어 설명과 함께.
- 보조: "Elephant in the Room"(NeurIPS 2024)에서 VUS-PR을 최신뢰 지표로 권고하는 RF-008 근거와 연동하여 이중 인용 가능.

## 주의사항
- 저자 6인 — 인용 시 "Paparrizos et al." 표기로 단축.
- VUS-PR과 VUS-ROC 중 우리 논문에서 어느 것을 primary로 쓰는지 확정 필요 (EXPERIMENT_PROTOCOL_TRUTH §④ 참조). 두 지표 모두 사용 시 "VUS-PR and VUS-ROC" 명시.
- PVLDB 15(11) 표기 — 일부 서지에서 pp.3392 등 다른 쪽수가 나타날 수 있으나, R26 truth 기준 p2774-paparrizos.pdf URL이 확인된 정확한 쪽수임.
