---
phase: 4
agent: assembler
directives: [T4]
last_modified: 2026-06-11
inputs:
  - VERIFICATION_LEDGER_A1.md (source-verifier-A1: 카드 1–25 + xu2018kpivae/xu2022anomalytransformer 추가 패스)
  - VERIFICATION_LEDGER_A2.md (source-verifier-A2: 카드 26–49)
  - refs_B1.bib / VERIFICATION_LEDGER_B1.md (source-verifier-B1: blind export 1–25)
  - refs_B2.bib / VERIFICATION_LEDGER_B2.md (source-verifier-B2: blind export 26–49)
  - VERIFIER_B_SEED.md (orchestrator seed — key+제목)
  - orchestrator 기계 diff (필드 단위, 2026-06-11)
output: refs.bib (49항목 병합 정본 — 본 보고서가 그 병합 근거 기록)
---

# P4_DIFF_REPORT — 2채널 독립 검증 기계 diff 결과 (Phase 4)

> **방법론**: A채널(source-verifier-A1/A2)은 reference card를 공식 소스(Crossref/DBLP/OpenReview/PMLR/publisher PDF/arXiv)와 대조 검증. B채널(source-verifier-B1/B2)은 card·scout 산출물을 일절 보지 않는 **blind 규약**으로 seed(key+제목)만 받아 공식 BibTeX를 신규 export. orchestrator가 양 채널 결과를 필드 단위로 기계 diff. 본 보고서는 그 diff 결과의 정식 기록이며, 모든 내용은 위 입력 문서들에 이미 존재하는 검증 기록에서만 취합했다 (신규 사실 창작 없음).

---

## 1. 총괄 결과

| 구분 | 건수 | 비율 |
|------|------|------|
| 완전 일치 | 33 / 49 | 67.3% |
| 표기 관례 차이 (실질 일치) | 10 / 49 | 20.4% |
| 실질 충돌 → 해소 | 6 / 49 | 12.2% |
| **QUARANTINE** | **0 / 49** | **0%** |

**49편 전부 검증 통과.** 병합 정본은 `refs.bib` (B채널 공식 export 기반 + 아래 §3 해소 6건 반영).

---

## 2. 표기 관례 차이 10건 (실질 일치)

양 채널이 같은 논문·같은 실질 서지를 가리키나 표기 수준에서만 갈린 경우. 어느 쪽도 오류 아님.

| 유형 | 해당 | 내용 |
|------|------|------|
| pages·doi 부재 — venue 표준 | ICLR(zong2018dagmm, ruff2020deepsad, xu2022anomalytransformer, wu2023timesnet, luo2024moderntcn, wu2025catch) / NeurIPS(song2023memto, liu2024elephant 등) / PMLR(sarfraz2024quovadis, xiong2020prenorm) 계열 | OpenReview/proceedings 기반 venue는 pages·DOI가 공식적으로 부재 — **양측 합의** (A: "N/A — 표준" / B: "not assigned"). 충돌 아님 |
| diacritic 정규화 아티팩트 | ganin2016dann | B export의 cedilla 정규화 아티팩트 (Fran\c{c}ois Laviolette 인코딩 차) — 동일 인물, DBLP 정본 표기 채택 |
| DOI escaping | goh2016swat | DBLP export의 `10.1007/978-3-319-71368-7\_8` LaTeX escaping vs A의 plain DOI — 동일 DOI |
| article-number 표기 | xu2023rosas | Elsevier article-number 저널: A "vol.60, issue 5, article 103459" vs B `pages = {103459}` — 동일 식별 (IPM 60(5) Art. 103459) |

> 키 단위의 '완전 일치 33 / 표기 관례 10' 버킷 구분은 orchestrator 기계 diff 산출 기준. 본 절은 그중 명시 기록된 차이 유형을 정리한 것이며, 모든 10건이 실질 일치로 판정되었다.

---

## 3. 실질 충돌 해소 6건

| # | key | 충돌 내용 | 해소 (refs.bib 반영) | 근거 |
|---|-----|----------|---------------------|------|
| ① | blazquez2021review | year: A=2021 (Crossref 온라인 게재 2021-04-17 기준) vs B=2022 (DBLP year 필드) | **2022 채택** | ACM CSUR **인쇄판/DBLP 기준** (vol.54 no.3, 2022-04). A의 2021은 온라인 게재일 — 오류는 아니나 인용 연도는 인쇄판 기준으로 통일. card에 채택 주석 추가 |
| ② | darban2024dacad | 인용 단위: A=TKDE 2025 본판 (arXiv journal-ref에서 확인) vs B=arXiv 2404.11269 preprint (2024 기록 우선 판단; "TKDE 판은 저자 상이" 플래그) | **TKDE 2025 본판 채택** — IEEE TKDE 37(8):4485–4496, DOI 10.1109/TKDE.2025.3569909 | DBLP `journals/tkde/DarbanYWAWPS25` **재export**; 동일 제목 확인. peer-reviewed 본판 우선 원칙. (저자 7인 — A채널 검증과 일치) |
| ③ | lai2023npsr | 4th author: A card="Jeffrey Lang" (OpenReview 표기) vs B="Jeffrey H. Lang" (DBLP) | **"Jeffrey H. Lang" 채택** | DBLP `conf/nips/LaiSGLB23` + NeurIPS proceedings 정본. **A측 card가 오류** (OpenReview 축약 표기를 그대로 전사) → card 정정 수행 (본 보고서 §5) |
| ④ | zhang2022selfdistill | B 최초 export가 **SdAE (Chen et al., ECCV 2022)** 를 오매칭 — seed에 제목 누락(orchestrator 결함)이 원인. B2가 KEY MISMATCH를 정당하게 플래그 | **DBLP `journals/pami/ZhangBM22` 재export로 해소** — Zhang, Bao & Ma, "Self-Distillation: Towards Efficient and Compact Neural Networks", IEEE TPAMI 44(8):4388–4403 | card는 처음부터 정확 (A2 검증 일치). 재export는 blind 규약 유지 (A 산출물 미참조, DBLP key 직접 지정) |
| ⑤ | sultani2018deepmil | B export에서 IEEE Xplore 차단(418)으로 DOI 확보 경로 불안정 | **doi 10.1109/CVPR.2018.00678 추가** | A채널 검증 (DBLP conf/cvpr/SultaniCS18 + arXiv) — IEEE Xplore 문서 8578776 |
| ⑥ | wang2025nrdetector | pages: 양 채널 모두 VERIFY_REQUIRED (ACM DL 403; DBLP KDD 2025 미색인) | **pages 1551–1562 확정** | orchestrator의 **Crossref DOI 질의** (10.1145/3690624.3709257) |

---

## 4. 양 ledger CRITICAL 정정 요약

A채널 검증에서 드러난 card 원본의 중대 오류 (전부 정정 완료; 상세는 각 ledger):

### A1 (VERIFICATION_LEDGER_A1.md)

| 카드 | 정정 | 심각도 |
|------|------|--------|
| xu2018kpivae | **저자 24인 → 13인** — card의 11인이 spurious (공식 기록 미지원, 제거). pages 187–196 추가 | **CRITICAL** |
| (그 외) | pages/DOI 누락 보강 등 21개 필드 MINOR 정정 (총 22개 필드, A1 ledger §6) | MINOR |

### A2 (VERIFICATION_LEDGER_A2.md)

| 카드 | 정정 | 심각도 |
|------|------|--------|
| liu2024treemil | 4th author "Jiming Li" → **"Shizhong Li"** (DBLP + arXiv) | **CRITICAL** |
| xu2023rosas | 5th author "Ninghui Liu" → **"Ning Liu"** (별개 인물 오기) | **CRITICAL** |
| xue2022fewpositive | 양 저자 모두 오류 "Yifan Xue, Yijie Yan" → **"Feng Xue, Weizhong Yan"** (심각한 hallucination — 두 이름 전부 오기) | **CRITICAL** |
| xu2018kpivae | A1과 동일 발견 (24→13인) — 양 검증자 독립 일치 | **CRITICAL** |
| paparrizos2022vus / ristea2024sdmae / wang2022hscl | abstract 말미 문장 정정 / DOI·pages 보강 / pages·LNCS 권 보강 | MINOR |

**교차 입증**: 위 CRITICAL 저자 정정들은 B채널 blind export(treemil=Shizhong Li, rosas=Ning Liu, xue=Feng Xue·Weizhong Yan, kpivae=13인)와 전부 일치 — 2채널 독립 수렴으로 정정의 정당성이 기계적으로 입증됨.

**역방향 검출**: lai2023npsr는 반대로 **B채널이 A측 card 오류를 검출**한 사례 (해소 ③) — A1은 OpenReview 기반으로 "정정 필드 없음" 판정했으나 DBLP/NeurIPS 정본 대조에서 middle initial 누락이 드러남. 2채널 설계의 존재 이유를 보여주는 사례.

---

## 5. Seed 결함 11건과 그 영향

`VERIFIER_B_SEED.md`에서 제목이 누락("(제목 누락)")된 행 11건 — orchestrator 결함:

| # | key | B 추론 결과 | diff 판정 |
|---|-----|------------|----------|
| 7 | darban2024dacad | DACAD 정확 식별 (논문 정체성 정확; 출판 단계 충돌은 별건 — §3 ②로 해소) | 정확 |
| 17 | huet2022affiliation | "affiliation" metric → Huet KDD 2022 단일 매치 | 정확 (A와 일치) |
| 20 | kim2022rigorous | "rigorous"+Kim+2022 → AAAI 2022 단일 매치 | 정확 (A와 일치) |
| 24 | lin2017focal | "focal"+Lin+2017 → ICCV 2017 RetinaNet 단일 매치 | 정확 (A와 일치) |
| 25 | liu2024elephant | Liu & Paparrizos NeurIPS 2024 D&B (TSB-AD) 단일 매치 | 정확 (A와 일치) |
| 26 | liu2024treemil | ICASSP 2024 TreeMIL | 정확 (A와 일치) |
| 30 | ristea2024sdmae | CVPR 2024 Self-Distilled MAE | 정확 (A와 일치) |
| 31 | ruff2020deepsad | ICLR 2020 Deep SAD | 정확 (A와 일치) |
| 36 | sultani2018deepmil | CVPR 2018 Real-World Anomaly Detection | 정확 (A와 일치) |
| 46 | xue2022fewpositive | IJCNN 2022 Xue & Yan | 정확 (A와 일치) |
| 48 | zhang2022selfdistill | **오매칭** — "selfdistill" → SdAE ECCV 2022 (Chen et al.)로 export. B2가 KEY MISMATCH(first author≠Zhang) 플래그 | **유일한 오매칭** → §3 ④ 재export로 해소 |

**영향 평가**: 11건 중 **오매칭 1건(zhang2022selfdistill)만 발생**, 나머지 10건은 B의 key-휴리스틱 추론이 전부 정확했음이 diff로 입증됨. 단 zhang 사례는 seed 제목이 blind 검증의 단일 실패점임을 보여줌 — 향후 blind seed에는 제목 전수 포함 필수 (orchestrator 프로세스 교훈). B2의 conflict 플래그 규약이 오매칭을 자체 검출한 점은 프로토콜이 의도대로 작동한 증거.

---

## 6. 잔존 제한 사항 (병합 후)

- **EXCERPT_UNVERIFIED 잔존 3건** (서지는 49편 전부 검증 완료 — 발췌/원문 대조만 제한): 상세는 `VERIFICATION_LEDGER.md` §3.
- **wang2025nrdetector 본문 발췌**: arXiv preprint 기준 확보 — KDD 2025 최종 게재본과 다를 수 있음 (A2 기록).
- **luo2024moderntcn "Spotlight" 표기**: A2는 공식 소스 미확인(DBLP poster 표기) / B2는 iclr.cc 가상 사이트에서 Spotlight 확인 — 원고에 Spotlight를 표기할 경우 Phase 7에서 iclr.cc 기준 재확인 권장 (인용 자체에는 영향 없음 — ICLR venue 표기만으로 충분).

---

## 정오 부기 (게이트 감사 r1 반영, 2026-06-11 orchestrator)

- **GB-1 해소**: `refs.bib`·`refs_B2.bib`의 `wang2025nrdetector` 구문 결함(doi 후행 콤마 누락 + 항목 내 % 코멘트)을 신규 DBLP 색인 `conf/kdd/Wang0XWJSZZ025` verbatim export로 전체 교체. 49/49 bibtexparser 파싱 확인.
- **MAJOR-1 (A1 요약 통계 자체모순)**: `VERIFICATION_LEDGER_A1.md` 상단 요약의 정정 건수 집계(18↔22, 7↔11)는 부기 오류 — **행 단위 상세 기록이 정본**이며 게이트 감사의 기계 집계가 이를 재확인함. 요약 수치는 게이트 보고서(`99_reviews/p4_coverage_gate_r1.md`) 기준으로 정정 해석할 것.
- **MAJOR-2 (diff 버킷 비재현)**: 본 보고서의 "완전 일치 33 / 표기 관례 10 / 해소 6" 버킷은 1차 diff 시점 분류 — 게이트 감사의 재분류와 경계 차이가 있으나(표기 관례 vs 일치의 경계), **49편 전체가 '실질 불일치 0'이라는 결론은 양쪽 동일**. 항목 단위 기록이 정본.
