---
phase: 4
agent: excerpt-curator-3
directives: [T4, R19]
last_modified: 2026-06-11
key: ahmed2017wadi
verification_status: VERIFIED_A
verified_by_A: 2026-06-11
corrected_fields: []
card_grade: LIGHT
abstract_access: failed
abstract_access_reason: |
  공식 dl.acm.org/doi/10.1145/3055366.3055375 → HTTP 403;
  Semantic Scholar API → "publisher 보류"로 abstract 필드 미제공;
  Crossref API → abstract 필드 부재;
  CySWATER 2017 워크숍 사이트(FORTH) PDF → 발표 슬라이드본(abstract 없음);
  dokumen.pub 재호스팅 → 사이트 점검 중.
  아래 abstract는 OpenAlex abstract_inverted_index 재구성본 — verbatim 보증 불가 (검증/대조에 그대로 사용 금지).
---
# WADI: A Water Distribution Testbed for Research in the Design of Secure Cyber Physical Systems
**경고: abstract verbatim은 검증/표절 대조 전용 — 본문 복사 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
- 저자 (Crossref 표기): Chuadhry Mujeeb Ahmed, Venkata Reddy Palleti, Aditya P. Mathur
- Venue: CySWATER '17 — Proceedings of the 3rd International Workshop on Cyber-Physical Systems for Smart Water Networks (CPS Week 2017, Pittsburgh, Pennsylvania), pp.25–28, 게재 2017-04-21
- DOI: 10.1145/3055366.3055375
- DBLP: conf/cpsweek/AhmedPM17
- fetch한 페이지: api.crossref.org + api.openalex.org + api.semanticscholar.org (DOI 질의, 2026-06-11); 공식 ACM 페이지 접근 실패 (403)

## Abstract 전문 — **비-verbatim 참고용** (OpenAlex inverted-index 재구성; 원문 대조 필수)
The architecture of a water distribution testbed (WADI), and on-going research in the design of secure system is presented. WADI consists of three stages controlled by Programmable Logic Controllers (PLCs) and two via Remote Terminal Units (RTUs). Each PLC and RTU uses sensors to estimate state and actuators to effect control. WADI is currently used to (a) conduct security analysis for water networks, (b) experimentally assess detection mechanisms for potential cyber and physical attacks, (c) understand how the impact of an attack on one CPS could cascade to other connected CPSs. The cascading effects of attacks can be studied through WADI's connection to testbeds, namely water treatment and power generation and distribution.

## 역할 (커버 claim)
- C-041: §4.1.1 Table 1 — WaDi 데이터셋 출처 (실험 섹션 전용).

## 비고
- 데이터셋 표기: WaDi (우리 실험 A1/A2 두 버전 사용; 원논문 표기는 WADI).
- verifier 액션: ACM 공식 페이지(또는 iTrust SUTD 원문 PDF)에서 abstract verbatim 확보 후 본 카드 교체 필요. 서지(저자 전원·쪽수 25–28)는 Crossref 공식 메타데이터 기준으로 신뢰도 높음.
