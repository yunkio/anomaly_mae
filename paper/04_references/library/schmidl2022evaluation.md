---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: schmidl2022evaluation
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "EXCERPT_UNVERIFIED RESOLVED. Abstract obtained verbatim from VLDB publisher PDF. Both card excerpts confirmed verbatim. Papenbrock spelling confirmed correct (not Papenbrook). All bibliographic fields confirmed."
card_grade: FULL
---
# Anomaly Detection in Time Series: A Comprehensive Evaluation

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: Sebastian Schmidl, Phillip Wenig, Thorsten Papenbrock
Venue: Proceedings of the VLDB Endowment (PVLDB)
연도: 2022
권호: 15(9):1779–1797
DOI: 10.14778/3538598.3538602
공식 URL: http://vldb.org/pvldb/vol15/p1779-wenig.pdf

## Abstract 전문 (verbatim) [VERIFIED_A — confirmed from VLDB publisher PDF pdftotext]

"Detecting anomalous subsequences in time series data is an important task in areas ranging from manufacturing processes over finance applications to health care monitoring. An anomaly can indicate important events, such as production faults, delivery bottlenecks, system defects, or heart flicker, and is therefore of central interest. Because time series are often large and exhibit complex patterns, data scientists have developed various specialized algorithms for the automatic detection of such anomalous patterns. The number and variety of anomaly detection algorithms has grown significantly in the past and, because many of these solutions have been developed independently and by different research communities, there is no comprehensive study that systematically evaluates and compares the different approaches. For this reason, choosing the best detection technique for a given anomaly detection task is a difficult challenge. This comprehensive, scientific study carefully evaluates most state-of-the-art anomaly detection algorithms. We collected and re-implemented 71 anomaly detection algorithms from different domains and evaluated them on 976 time series datasets. The algorithms have been selected from different algorithm families and detection approaches to represent the entire spectrum of anomaly detection techniques. In the paper, we provide a concise overview of the techniques and their commonalities; we evaluate their individual strengths and weaknesses and, thereby, consider factors, such as effectiveness, efficiency, and robustness. Our experimental results should ease the algorithm selection problem and open up new research directions."

## 핵심 발췌 (verbatim, 섹션/위치 표기)

> "many of these solutions have been developed independently and by different research communities, there is no comprehensive study that systematically evaluates and compares the different approaches." (§1 Introduction — 2차 소스 기반, EXCERPT_UNVERIFIED)

커버 claim: C-001
활용 맥락: §1 도입부에서 TSAD 연구의 파편화를 지적하고 종합 평가 필요성을 논할 때.

---

> "choosing the best detection technique for a given anomaly detection task is a difficult challenge." (§1 Introduction 근처 — 2차 소스 기반, EXCERPT_UNVERIFIED)

커버 claim: C-001
활용 맥락: TSAD 방법 선택의 어려움을 인정하며 우리 논문의 체계적 비교 실험 필요성을 정당화.

---

**[EXCERPT_UNVERIFIED: 이하는 vldb.org PDF에서 확인된 주요 발견이며 정확한 verbatim이 아님. verifier 확인 필요.]**

주요 발견 (2차 소스):
- 71개 알고리즘 × 976개 시계열 데이터셋 — 이 규모의 TSAD 평가가 최초.
- "Every anomaly detection family can be effective and there is no clear winner."
- Reconstruction 기반 방법들이 대체로 낮은 AUC-ROC(~0.5)를 기록 — EncDec-AD, Donut 예외.
- Distance 및 forecasting 계열이 강한 성능.
- 도메인·이상 유형에 따라 최적 알고리즘이 달라짐.

## 우리 논문에서의 활용

커버 claim: C-001 (+ C-009, C-045 보조)

- **§1 Introduction (Para 1)**: C-001 — "다변량 시계열 이상탐지가 산업·안전 응용에서 중요하다" 도입부 클러스터 인용. Blázquez-García et al. (CSUR 2022)과 함께 괄호 클러스터 인용.
- **§4.1.3 또는 §4.1.1 보조**: C-009, C-045 — benchmark 관행 평가 맥락에서 보조 인용 가능.

## 주의사항
- 이 논문은 주로 univariate 이상탐지 알고리즘도 포함하므로 "multivariate TSAD" 특화 주장의 근거로만 인용하면 범위 왜곡이 생길 수 있음. §1 도입부의 "TSAD의 중요성과 다양성" 수준 인용이 적절.
- abstract 전문 미확보 — EXCERPT_UNVERIFIED 상태. verifier 필수 작업.
- vldb.org PDF URL의 저자 순서 주의: URL이 "p1779-wenig.pdf"로 Wenig이 앞에 있으나, 논문 저자 순서는 Schmidl, Wenig, Papenbrock. 인용 시 "Schmidl et al." 사용.
