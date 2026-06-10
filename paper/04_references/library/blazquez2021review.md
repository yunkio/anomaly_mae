---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: blazquez2021review
verification_status: VERIFIED_2CH
verified_by_A: 2026-06-11
verified_by_B: 2026-06-11 (blind export — DBLP journals/csur/Blazquez-Garcia21)
corrected_fields: [year_clarified, pages_added, year_final_2022]
card_grade: LIGHT
correction_history: |
  2026-06-11 assembler (P4_DIFF_REPORT 해소 ①): 인용 연도 최종 채택 = 2022
  (ACM CSUR 인쇄판/DBLP year 필드 기준 — vol.54 no.3, 2022-04 인쇄).
  A1의 2021 판정은 Crossref 온라인 게재일(2021-04-17) 기준으로 사실 자체는 정확하나,
  인용 연도는 인쇄판 기준으로 통일. refs.bib 정본 year=2022. key는 blazquez2021review 유지
  (DBLP key 접미사와 동일 관례 — key≠인용 연도).
---
# A Review on Outlier/Anomaly Detection in Time Series Data
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자: Ane Blázquez-García, Angel Conde, Usue Mori, Jose A. Lozano
- Venue: ACM Computing Surveys, vol.54, no.3, Article 56, pp.56:1–56:33
- 연도: **2022 [최종 채택 — assembler 2026-06-11, diff 해소 ①]** — ACM CSUR 인쇄판/DBLP 기준 (vol.54 no.3, 2022-04). Crossref online 게재일은 2021-04-17 (A1의 2021 판정 근거 — 온라인 게재일로서는 정확하나 인용 연도는 인쇄판 기준 통일). refs.bib year=2022. DOI: 10.1145/3444690
- DOI: 10.1145/3444690
- arXiv: 2002.04236 (v1 2020-02-11; comments: "32 pages, 21 figures, submitted to ACM Computing Surveys (CSUR)")
- DBLP: journals/csur/Blazquez-Garcia21
- fetch한 페이지: https://arxiv.org/abs/2002.04236 (2026-06-11 직접 열람)

## Abstract 전문 (verbatim — arXiv v1 기준)
Recent advances in technology have brought major breakthroughs in data collection, enabling a large amount of data to be gathered over time and thus generating time series. Mining this data has become an important task for researchers and practitioners in the past few years, including the detection of outliers or anomalies that may represent errors or events of interest. This review aims to provide a structured and comprehensive state-of-the-art on outlier detection techniques in the context of time series. To this end, a taxonomy is presented based on the main aspects that characterize an outlier detection technique.

## 역할 (커버 claim)
- C-001: §1 Para 1 — 시계열 이상탐지의 도메인 중요성 survey 괄호 클러스터 (Schmidl et al. PVLDB 2022와 동반 인용).

## 비고
- arXiv abs 페이지의 abstract는 arXiv 판 기준 — CSUR 게재본 abstract와 문구 차이 가능 (verifier가 dl.acm.org/10.1145/3444690 대조; 본 fetch 시점 ACM DL은 403 차단 경향).
- 인용 표기는 CSUR 본(권·호·Article 번호) 우선, **연도는 2022** (인쇄판 — refs.bib 정본과 일치; P4_DIFF_REPORT.md §3 ①).
