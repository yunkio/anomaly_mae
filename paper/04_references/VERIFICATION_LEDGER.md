---
phase: 4
agent: assembler
directives: [T4]
last_modified: 2026-06-11
status: FINAL — 49/49 VERIFIED (2-channel), QUARANTINE 0
detail_pointers:
  - VERIFICATION_LEDGER_A1.md (A채널 카드 1–25 + xu2018kpivae/xu2022anomalytransformer 추가 패스, 발췌 해소 11건)
  - VERIFICATION_LEDGER_A2.md (A채널 카드 26–49, CRITICAL 정정 4건)
  - VERIFICATION_LEDGER_B1.md / refs_B1.bib (B채널 blind export 1–25)
  - VERIFICATION_LEDGER_B2.md / refs_B2.bib (B채널 blind export 26–49)
  - P4_DIFF_REPORT.md (기계 diff 결과 + 해소 6건 + seed 결함 영향)
---

# VERIFICATION LEDGER — 통합 마스터 (Phase 4)

> 49편 전부 **VERIFIED (2-channel)**: A채널(card↔공식소스 대조) + B채널(blind 공식 BibTeX export) 독립 수렴 + orchestrator 기계 diff 통과. 정본 서지는 `refs.bib`. 본 표는 요약이며 키별 상세 로그는 frontmatter의 포인터 문서 참조.

## 1. 판정 총괄

| 항목 | 수치 |
|------|------|
| 총 reference | 49 |
| VERIFIED (2-channel) | **49** |
| QUARANTINE | **0** |
| diff: 완전 일치 / 표기 관례(실질 일치) / 해소 | 33 / 10 / 6 |
| EXCERPT_UNVERIFIED 잔존 (발췌만 제한 — §3) | 3 |

## 2. 49편 통합 판정표

A 검증 일시는 전 건 2026-06-11. diff 결과 열의 ①–⑥은 `P4_DIFF_REPORT.md` §3의 해소 번호.

| # | key | 판정 | A 검증자 | B export 소스 | diff 결과 | 비고 |
|---|-----|------|---------|--------------|----------|------|
| 1 | abdulaal2021psm | VERIFIED (2ch) | A1 | DBLP conf/kdd/AbdulaalLL21 | 일치 | pages 2485–2494 (A 보강) |
| 2 | ahmed2017wadi | VERIFIED (2ch) | A1 | DBLP conf/cpsweek/AhmedPM17 | 일치 | |
| 3 | audibert2020usad | VERIFIED (2ch) | A1 | DBLP conf/kdd/AudibertMGMZ20 | 일치 | Frédéric Guyard 전체 이름 (A 정정) |
| 4 | bekker2020pusurvey | VERIFIED (2ch) | A1 | DBLP journals/ml/BekkerD20 | 일치 | SCAR·two-step 발췌 확보 (A1) |
| 5 | bergmann2020uninformed | VERIFIED (2ch) | A1 | DBLP conf/cvpr/BergmannFSS20 | 일치 | pages/DOI 보강 (A) |
| 6 | blazquez2021review | VERIFIED (2ch) | A1 | DBLP journals/csur/Blazquez-Garcia21 | **해소①** | year **2022 채택** (인쇄판/DBLP; A의 2021=온라인 게재일) — card 주석 |
| 7 | darban2024dacad | VERIFIED (2ch) | A1 | DBLP journals/tkde/DarbanYWAWPS25 (재export) | **해소②** | **TKDE 2025 본판 채택** (37(8):4485–4496); seed 제목 누락이었으나 B 정체성 식별 정확 |
| 8 | deng2021gdn | VERIFIED (2ch) | A1 | DBLP conf/aaai/DengH21 | 일치 | |
| 9 | deng2022reverse | VERIFIED (2ch) | A1 | DBLP conf/cvpr/DengL22 | 일치 | pages/DOI 보강 (A) |
| 10 | duplessis2014pu | VERIFIED (2ch) | A1 | DBLP conf/nips/PlessisNS14 | 일치 (doi 부재 — NeurIPS 표준 합의) | |
| 11 | elkan2008pu | VERIFIED (2ch) | A1 | DBLP conf/kdd/ElkanN08 | 일치 | 제목 소문자 공식 표기 (A 확정) |
| 12 | fang2024tfmae | VERIFIED (2ch) | A1 | DBLP conf/icde/FangXZ0G024 | 일치 | arXiv 없음 (ICDE 본이 유일 공식본) |
| 13 | ganin2016dann | VERIFIED (2ch) | A1 | DBLP journals/jmlr/GaninUAGLLML16 | 표기 관례 (cedilla 정규화 아티팩트) | GRL Eq.16-17·λ schedule 발췌 확보 (A1) |
| 14 | goh2016swat | VERIFIED (2ch) | A1 | DBLP conf/critis/GohAJM16 | 표기 관례 (DOI escaping) | LNCS vol.10242 |
| 15 | he2022mae | VERIFIED (2ch) | A1 | DBLP conf/cvpr/HeCXLDG22 | 일치 | linear-patchify 발췌 확보 (A1); B의 구판 pages 16000–16009 검색 결과는 DBLP export로 기각 |
| 16 | huang2022slavae | VERIFIED (2ch) | A1 | DBLP conf/www/HuangCL22 | 일치 | abstract 확보 (S2 API) — C-011/025 반증 판정 자료 |
| 17 | huet2022affiliation | VERIFIED (2ch) | A1 | DBLP conf/kdd/HuetNR22 | 일치 | seed 제목 누락 — B 추론 정확 |
| 18 | hundman2018telemanom | VERIFIED (2ch) | A1 | DBLP conf/kdd/HundmanCLCS18 | 일치 | Söderström diacritic (A 정정) |
| 19 | jacob2021exathlon | VERIFIED (2ch) | A1 | DBLP journals/pvldb/JacobSSRDT21 | 일치 | |
| 20 | kim2022rigorous | VERIFIED (2ch) | A1 | DBLP conf/aaai/KimCCLY22 | 일치 | seed 제목 누락 — B 추론 정확; PA%K 발췌 확보 (A1) |
| 21 | kiryo2017nnpu | VERIFIED (2ch) | A1 | DBLP conf/nips/KiryoNPS17 | 일치 (doi 부재 — NeurIPS 표준 합의) | |
| 22 | lai2023npsr | VERIFIED (2ch) | A1 | DBLP conf/nips/LaiSGLB23 | **해소③** | **"Jeffrey H. Lang" 채택** (DBLP/NeurIPS 정본) — **A측 card 오류 → 정정 완료** |
| 23 | lee2021wetas | VERIFIED (2ch) | A1 | DBLP conf/iccv/LeeYJY21 | 일치 | DOI R26 truth 일치 확인 |
| 24 | lin2017focal | VERIFIED (2ch) | A1 | DBLP conf/iccv/LinGGHD17 | 일치 | seed 제목 누락 — B 추론 정확; p_t·FL 수식 발췌 확보 (A1) |
| 25 | liu2024elephant | VERIFIED (2ch) | A1 | DBLP conf/nips/LiuP24a | 일치 (pages/doi 부재 — NeurIPS 표준 합의) | seed 제목 누락 — B 추론 정확 |
| 26 | liu2024treemil | VERIFIED (2ch) | A2 | DBLP conf/icassp/LiuHLL24 | 일치 | **CRITICAL 정정**: "Jiming Li"→"Shizhong Li" (A2) — B export와 교차 입증; seed 제목 누락 — B 추론 정확 |
| 27 | luo2024moderntcn | VERIFIED (2ch) | A2 | DBLP conf/iclr/LuoW24 | 일치 (pages/doi 부재 — ICLR 표준 합의) | Spotlight 표기는 Phase 7 재확인 권장 (A2 미확인 / B2 iclr.cc 확인) |
| 28 | pang2019devnet | VERIFIED (2ch) | A2 | DBLP conf/kdd/PangSH19 | 일치 | pages/DOI 보강 (A2) |
| 29 | paparrizos2022vus | VERIFIED (2ch) | A2 | ACM DL/VLDB PDF + DBLP journals/pvldb/PaparrizosB0TFE22 | 일치 | abstract 말미 문장 MINOR 정정 (A2, 게재 PDF 대조) |
| 30 | ristea2024sdmae | VERIFIED (2ch) | A2 | DBLP conf/cvpr/RisteaCIPKS24 | 일치 | DOI/pages 보강 (A2); seed 제목 누락 — B 추론 정확 |
| 31 | ruff2020deepsad | VERIFIED (2ch) | A2 | DBLP conf/iclr/RuffVGBMMK20 | 일치 (pages/doi 부재 — ICLR 표준 합의) | seed 제목 누락 — B 추론 정확; SAD loss 발췌 잔존 (§3) |
| 32 | sarfraz2024quovadis | VERIFIED (2ch) | A2 | PMLR v235/sarfraz24a (공식; DBLP .bib fetch 실패) | 일치 (doi 부재 — PMLR 표준 합의) | |
| 33 | schmidl2022evaluation | VERIFIED (2ch) | A2 | DBLP journals/pvldb/SchmidlWP22 | 일치 | abstract verbatim 확보 (A2, VLDB PDF) — Papenbrock 철자 확정 |
| 34 | song2023memto | VERIFIED (2ch) | A2 | DBLP conf/nips/SongKOC23 | 일치 (pages/doi 부재 — NeurIPS 표준 합의) | |
| 35 | su2019omnianomaly | VERIFIED (2ch) | A2 | DBLP conf/kdd/SuZNLSP19 | 일치 | abstract "signicantly" = S2/ACM ligature 아티팩트 (기록 완료) |
| 36 | sultani2018deepmil | VERIFIED (2ch) | A2 | DBLP conf/cvpr/SultaniCS18 (IEEE 418 — 다중 소스 triangulation) | **해소⑤** | **doi 10.1109/CVPR.2018.00678 추가** (A 검증); seed 제목 누락 — B 추론 정확 |
| 37 | tuli2022tranad | VERIFIED (2ch) | A2 | DBLP journals/pvldb/TuliCJ22 | 일치 | |
| 38 | wang2022hscl | VERIFIED (2ch) | A2 | DBLP conf/eccv/WangZWSN22 | 일치 | pages 110–128·LNCS Part XXV 보강 (A2) |
| 39 | wang2025nrdetector | VERIFIED (2ch) | A2 | ACM DL DOI 10.1145/3690624.3709257 + arXiv PDF (DBLP 미색인) | **해소⑥** | **pages 1551–1562** (orchestrator Crossref 질의 — 양측 VERIFY_REQUIRED 해소); 본문 발췌는 preprint 기준 (caveat) |
| 40 | wu2023timesnet | VERIFIED (2ch) | A2 | DBLP conf/iclr/WuHLZ0L23 | 일치 (pages/doi 부재 — ICLR 표준 합의) | poster (spotlight 아님 — A2 확인) |
| 41 | wu2025catch | VERIFIED (2ch) | A2 | DBLP conf/iclr/WuQL0HGXY25 | 일치 (pages/doi 부재 — ICLR 표준 합의) | |
| 42 | xiong2020prenorm | VERIFIED (2ch) | A2 | DBLP conf/icml/XiongYHZZXZLWL20 + PMLR v119/xiong20b | 일치 (PMLR doi 부재 채택; B의 ACM DL 10.5555 미채택) | PMLR 공식 표기 "Tieyan Liu" vs DBLP "Tie-Yan Liu" — refs.bib는 DBLP export 표기 |
| 43 | xu2018kpivae | VERIFIED (2ch) | A2 + A1(추가 발췌 패스) | DBLP conf/www/XuCZLBLLZPFCWQ18 | 일치 | **CRITICAL 정정**: 저자 24→**13인** (A1·A2 독립 일치, B export와 교차 입증); PA 프로토콜·abstract 발췌 확보 (A1, arXiv PDF); 게재본 대조 잔존 (§3) |
| 44 | xu2022anomalytransformer | VERIFIED (2ch) | A2 + A1(추가 발췌 패스) | DBLP conf/iclr/XuWWL22 | 일치 (pages/doi 부재 — ICLR 표준 합의) | **AR-threshold 발췌 확보 (A1, arXiv PDF §4) → R30 보류 해제**; ICLR 2022 Spotlight 확인 |
| 45 | xu2023rosas | VERIFIED (2ch) | A2 | ScienceDirect DOI 10.1016/j.ipm.2023.103459 + DBLP journals/ipm/XuWPJLW23 | 표기 관례 (article-number 표기) | **CRITICAL 정정**: "Ninghui Liu"→"Ning Liu" (A2) — B export와 교차 입증 |
| 46 | xue2022fewpositive | VERIFIED (2ch) | A2 | DBLP conf/ijcnn/XueY22 | 일치 | **CRITICAL 정정**: "Yifan Xue, Yijie Yan"→**"Feng Xue, Weizhong Yan"** (A2) — B export와 교차 입증; seed 제목 누락 — B 추론 정확 |
| 47 | yang2023dcdetector | VERIFIED (2ch) | A2 | ACM DL DOI 10.1145/3580305.3599295 (DBLP .bib fetch 실패) | 일치 | |
| 48 | zhang2022selfdistill | VERIFIED (2ch) | A2 | DBLP journals/pami/ZhangBM22 (재export) | **해소④** | B 최초 export가 **SdAE ECCV 오매칭** (seed 제목 누락 — orchestrator 결함; B2 KEY MISMATCH 플래그 정당) → 재export 해소. **card는 처음부터 정확**; abstract 발췌 잔존 (§3) |
| 49 | zong2018dagmm | VERIFIED (2ch) | A2 | DBLP conf/iclr/ZongSMCLCC18 | 일치 (pages/doi 부재 — ICLR 표준 합의) | "Daeki Cho"(OpenReview/card) vs "Dae-ki Cho"(DBLP) 양형 attested (A2 기록) — refs.bib는 DBLP export 표기 |

---

## 3. EXCERPT_UNVERIFIED 잔존 목록

**서지는 49편 전부 2채널 검증 완료. 아래 항목은 발췌(원문 verbatim 대조)만 제한**된 상태다.

| key | 잔존 범위 | 사유 | 서지 상태 |
|-----|----------|------|----------|
| zhang2022selfdistill | abstract·본문 전체 | IEEE Xplore paywall; arXiv preprint 부재 (A2 검색 확인) | VERIFIED (DBLP) — "self-distillation" 용어 원류의 1–2문장 괄호 인용 용도로는 서지만으로 충분 (A2 판정) |
| xu2018kpivae | ACM **게재본** abstract·본문 대조 | ACM DL 403. A1이 arXiv PDF(1802.03903)에서 abstract·PA 프로토콜(§4.2) verbatim 확보 — 단 preprint 기준이며 게재본 직접 대조는 미완 | VERIFIED (DBLP+Crossref, 13인 저자) |
| ruff2020deepsad | SAD loss objective 수식 (§3 본문) | §3 본문 미접근 (abstract만 확보) | VERIFIED (DBLP+OpenReview+arXiv) |

부수 caveat (EXCERPT_UNVERIFIED는 아님): **wang2025nrdetector** 본문 발췌 6건은 arXiv preprint 기준 확보 — KDD 2025 최종 게재본과 다를 수 있음 (A2 기록).

### 2단계 격리 규칙 (인용 시 적용)

> EXCERPT_UNVERIFIED 잔존 항목에 대해: **직접 인용(따옴표)·verbatim 활용 금지. reference로서의 인용(서지 인용)은 가능.**
> 즉 위 3편은 괄호 인용·서지 인용에는 제약이 없으나, 해당 잔존 범위의 원문 문구를 원고에 따오거나 verbatim으로 근거 삼는 것은 게재본 발췌 확보 전까지 금지한다.

### 해소 완료된 발췌 (참고 — 상세 A1 §5 / A2)

A1 발췌 해소 11건: bekker2020pusurvey(SCAR §3.1.1·two-step §5), ganin2016dann(GRL Eq.16-17 §4.2·λ schedule §5.2), he2022mae(linear patchify §3), kim2022rigorous(PA%K §4+Fig.6), lin2017focal(p_t Eq.2·FL Eq.4), xu2018kpivae(abstract·PA 프로토콜 §4.2 — arXiv 판), xu2022anomalytransformer(**AR threshold §4 — R30 보류 해제**), huang2022slavae(abstract — S2 API).
A2 발췌 해소 2건: schmidl2022evaluation(abstract — VLDB PDF), paparrizos2022vus(abstract — PVLDB PDF, 말미 문장 정정 포함).

---

## 4. 후속 단계 인계 사항

1. 정본 서지는 **refs.bib만** 사용 (손 타이핑 금지). IEEE 잠정 정리본: `REFERENCES_IEEE.md` (최종 번호·스타일은 Phase 7 elsarticle bibliography가 결정).
2. claim별 인용 배치: `CLAIM_CITATION_MAP.md` (assembler 갱신 — CANDIDATE→VERIFIED 반영, §6 갱신 로그).
3. card 색인: `REFERENCE_LIBRARY_INDEX.md`.
4. C-011/C-025 최초성 서술은 **D-008 스코핑 축소(재서술) 유지** — A 검증으로 반증 후보의 반증 강도가 약화(SLA-VAE)·정밀화(Xue & Yan 저자 정정)되었으나 보수적 재서술 권고는 변경하지 않음.
