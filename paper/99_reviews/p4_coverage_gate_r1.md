---
phase: 4
agent: gate-auditor
directives: [T4, M10]
last_modified: 2026-06-11
verdict: "FAIL (조건부) — BLOCKER 1건: refs.bib wang2025nrdetector 항목 BibTeX 구문 결함(파싱 시 항목 전체 탈락). 그 외 전수 재감사·무작위 재검증 16편·QUARANTINE 무결·격리 규칙 전부 통과. 1줄 수정 후 재확인 시 PASS 가능"
inputs:
  - VERIFICATION_LEDGER.md (통합) / VERIFICATION_LEDGER_{A1,A2,B1,B2}.md
  - P4_DIFF_REPORT.md / refs.bib / refs_B1.bib / refs_B2.bib
  - REFERENCES_IEEE.md / REFERENCE_LIBRARY_INDEX.md / CLAIM_CITATION_MAP.md
  - library/ card 49 + VERIFIER_B_SEED.md
method: |
  ① 통합 ledger 49행 전수 재감사 (표본 아님): A1/A2 스냅샷 ↔ B1/B2 기록 ↔ refs.bib 필드 ↔ card frontmatter
     — 필드 비교는 스크립트(bibtexparser + 정규화 비교)로 기계 수행, 판정·해소 기록은 수동 대조.
  ② 무작위 재검증 라운드: python random.seed(42); random.sample(sorted(49 keys), 10) + 충돌 해소 6편
     = 16 슬롯(고유 14편; blazquez·darban 중복) — DBLP .bib 직접 fetch 13건 + Crossref DOI 질의 1건으로
     저자 전원·제목·venue·연도·pages·DOI 전 필드 재검증.
  ③ QUARANTINE 무결: 전 문서 grep + CLAIM_CITATION_MAP DOI 전수 추출 ↔ refs.bib 대조.
  ④ refs.bib 형식: bibtexparser 1.4.4 + 실제 bibtex 0.99d (TeX Live) 양쪽으로 파싱 시험.
  ⑤ EXCERPT_UNVERIFIED 3건 격리 마킹 grep 확인.
---

# P4 Coverage Gate 감사 r1 — VERIFICATION_LEDGER 전수 재감사 + 무작위 재검증

> 게이트 규정: "독립 리뷰어가 VERIFICATION_LEDGER를 전수로 재감사 + 무작위 재검증 라운드 1회.
> QUARANTINE 항목이 인용 목록에 없는지 + CLAIM_CITATION_MAP의 어떤 행도 QUARANTINE key를 가리키지 않는지 확인."

## 0. 판정 총괄

| 감사 항목 | 결과 |
|----------|------|
| ① ledger 전수 재감사 (49행) | **통과** — A↔B↔refs.bib↔card 필드 모순 0건 (문서화된 해소 6건 제외 시 기계 비교 mismatch 0) |
| ② 무작위 재검증 16편 (고유 14) | **전건 서지 일치** — DBLP/Crossref 직접 재fetch, 실질 필드 불일치 0건 (sultani booktitle/publisher 표기 일탈 1건 — MINOR) |
| ③ QUARANTINE 무결 | **통과** — QUARANTINE 0 사실 확인; CLAIM_CITATION_MAP 비검증 인용 경로 0; refs.bib/REFERENCES_IEEE 비검증 항목 0 |
| ④ refs.bib 형식 무결 | **실패 (BLOCKER 1)** — wang2025nrdetector 항목 구문 결함으로 bibtex·bibtexparser 모두 해당 항목 탈락 (48/49) |
| ⑤ EXCERPT_UNVERIFIED 격리 | **통과** — 3건 전부 ledger·card에 verbatim 금지 마킹 확인 |
| **종합** | **FAIL (조건부)** — BLOCKER 1건 수정(사실상 1줄) 후 PASS 전환 가능 |

---

## 1. ledger 전수 재감사 (49행 — 표본 아님)

검증 방식: 통합 ledger의 각 행에 대해 (a) A채널 스냅샷 존재·판정 VERIFIED_A, (b) B채널 기록 존재·소스 일치,
(c) refs.bib 필드(author/title/year/pages/doi/volume/number/venue) 일치 — 기계 비교, (d) card frontmatter
verification_status 확인. 기계 비교 결과: **문서화된 해소 6건(①–⑥)을 제외한 43행에서 필드 mismatch 0건**,
해소 6건은 전부 P4_DIFF_REPORT §3의 채택 근거대로 refs.bib에 반영 확인.

| # | key | A기록 | B기록 | refs.bib | card | 비고 |
|---|-----|-------|-------|----------|------|------|
| 1 | abdulaal2021psm | ✓ | ✓ | ✓ | VERIFIED_A | pages 2485–2494 일치 |
| 2 | ahmed2017wadi | ✓ | ✓ | ✓ | VERIFIED_A | (무작위 재검증 — §2 #3) |
| 3 | audibert2020usad | ✓ | ✓ | ✓ | VERIFIED_A | Frédéric Guyard 전체 이름 일치 |
| 4 | bekker2020pusurvey | ✓ | ✓ | ✓ | VERIFIED_A | DOI 대문자 S는 DBLP export 관례 |
| 5 | bergmann2020uninformed | ✓ | ✓ | ✓ | VERIFIED_A | |
| 6 | blazquez2021review | ✓(2021) | ✓(2022) | ✓ year=2022 | **VERIFIED_2CH** | 해소① 반영 확인; card 주석 존재 (무작위 재검증 — §2 #10) |
| 7 | darban2024dacad | ✓ | ✓(재export) | ✓ TKDE 2025 | VERIFIED_A | 해소② 반영 확인 (무작위 재검증 — §2 #8) |
| 8 | deng2021gdn | ✓ | ✓ | ✓ | VERIFIED_A | (무작위 재검증 — §2 #2) |
| 9 | deng2022reverse | ✓ | ✓ | ✓ | VERIFIED_A | (무작위 재검증 — §2 #7) |
| 10 | duplessis2014pu | ✓ | ✓ | ✓ | VERIFIED_A | doi 부재 = NeurIPS 표준 (양 채널 합의) |
| 11 | elkan2008pu | ✓ | ✓ | ✓ | VERIFIED_A | 제목 소문자 공식 표기 |
| 12 | fang2024tfmae | ✓ | ✓ | ✓ | VERIFIED_A | |
| 13 | ganin2016dann | ✓ | ✓ | ✓ | VERIFIED_A | cedilla 정규화 아티팩트 — 실질 일치 |
| 14 | goh2016swat | ✓ | ✓ | ✓ | VERIFIED_A | LNCS volume 필드는 DBLP export 자체에 부재 — 직접 fetch로 확인 (결락 아님) |
| 15 | he2022mae | ✓ | ✓ | ✓ | VERIFIED_A | (무작위 재검증 — §2 #6) |
| 16 | huang2022slavae | ✓ | ✓ | ✓ | VERIFIED_A | (무작위 재검증 — §2 #5) |
| 17 | huet2022affiliation | ✓ | ✓(seed 결함·추론 정확) | ✓ | VERIFIED_A | |
| 18 | hundman2018telemanom | ✓ | ✓ | ✓ | VERIFIED_A | Söderström diacritic 일치 (무작위 재검증 — §2 #4) |
| 19 | jacob2021exathlon | ✓ | ✓ | ✓ | VERIFIED_A | claim 행 부재는 인계 사항으로 기록됨 (결함 아님) |
| 20 | kim2022rigorous | ✓ | ✓ | ✓ | VERIFIED_A | |
| 21 | kiryo2017nnpu | ✓ | ✓ | ✓ | VERIFIED_A | |
| 22 | lai2023npsr | ✓ | ✓ | ✓ Jeffrey H. Lang | **VERIFIED_2CH** | 해소③ 반영·card 정정 확인 (충돌 재검증 — §2 #11) |
| 23 | lee2021wetas | ✓ | ✓ | ✓ | VERIFIED_A | DOI R26 truth 일치 기록 확인 |
| 24 | lin2017focal | ✓ | ✓ | ✓ | VERIFIED_A | |
| 25 | liu2024elephant | ✓ | ✓ | ✓ | VERIFIED_A | |
| 26 | liu2024treemil | ✓(CRITICAL 정정) | ✓ | ✓ Shizhong Li | VERIFIED_A | card 정정 반영 확인 |
| 27 | luo2024moderntcn | ✓ | ✓ | ✓ | VERIFIED_A | Spotlight 표기 Phase 7 재확인 caveat 유지 — §5 MINOR gm-5 참조 |
| 28 | pang2019devnet | ✓ | ✓ | ✓ | VERIFIED_A | |
| 29 | paparrizos2022vus | ✓ | ✓ | ✓ | VERIFIED_A | abstract 말미 정정 기록 일치 |
| 30 | ristea2024sdmae | ✓ | ✓ | ✓ | VERIFIED_A | |
| 31 | ruff2020deepsad | ✓ | ✓ | ✓ | VERIFIED_A | EXCERPT 잔존 — 격리 마킹 확인 (§4) |
| 32 | sarfraz2024quovadis | ✓ | ✓(PMLR) | ✓ | VERIFIED_A | doi 부재 = PMLR 표준 |
| 33 | schmidl2022evaluation | ✓ | ✓ | ✓ | VERIFIED_A | Papenbrock 철자 일치 |
| 34 | song2023memto | ✓ | ✓ | ✓ | VERIFIED_A | |
| 35 | su2019omnianomaly | ✓ | ✓ | ✓ | VERIFIED_A | (무작위 재검증 — §2 #9) |
| 36 | sultani2018deepmil | ✓ | ✓(triangulation) | ✓ 실질 필드 | VERIFIED_A | 해소⑤ DOI 일치; booktitle/publisher 손조립 표기 일탈 — §5 MINOR gm-3 (충돌 재검증 — §2 #13) |
| 37 | tuli2022tranad | ✓ | ✓ | ✓ | VERIFIED_A | |
| 38 | wang2022hscl | ✓ | ✓ | ✓ | VERIFIED_A | LNCS volume 필드는 DBLP export 자체에 부재 — 직접 fetch로 확인 (결락 아님) |
| 39 | wang2025nrdetector | ✓ | ✓(pages VERIFY_REQUIRED) | ✓ 필드 / **✗ 구문** | VERIFIED_A | 해소⑥ pages 1551–1562 — Crossref+DBLP 재확인 일치. **단 bib 항목 구문 결함 — §3 BLOCKER** (충돌 재검증 — §2 #14) |
| 40 | wu2023timesnet | ✓ | ✓ | ✓ | VERIFIED_A | poster 확인 기록 일치 |
| 41 | wu2025catch | ✓ | ✓ | ✓ | VERIFIED_A | (무작위 재검증 — §2 #1) |
| 42 | xiong2020prenorm | ✓ | ✓ | ✓ Tie-Yan Liu(DBLP 표기) | VERIFIED_A | B의 ACM 10.5555 미채택 — refs.bib doi 부재 일치 |
| 43 | xu2018kpivae | ✓(A1+A2 독립 일치, 13인) | ✓(13인) | ✓ 13인 | VERIFIED_A | CRITICAL 정정 card 반영 확인; EXCERPT 잔존 격리 마킹 확인 (§4) |
| 44 | xu2022anomalytransformer | ✓(A1 추가 패스 — R30 해제) | ✓ | ✓ | VERIFIED_A | A2의 "R30 hold" 기록은 A1 추가 패스로 시간순 해소 — 모순 아님 |
| 45 | xu2023rosas | ✓(CRITICAL 정정) | ✓ | ✓ Ning Liu | VERIFIED_A | card 정정 반영 확인 |
| 46 | xue2022fewpositive | ✓(CRITICAL 정정) | ✓ | ✓ Feng Xue·Weizhong Yan | VERIFIED_A | card 정정 반영 확인 |
| 47 | yang2023dcdetector | ✓ | ✓ | ✓ | VERIFIED_A | |
| 48 | zhang2022selfdistill | ✓ | △(최초 SdAE 오매칭→재export) | ✓ TPAMI ZhangBM22 | VERIFIED_A | 해소④ 반영 확인; B2 ledger 본문은 미갱신 — §5 MINOR gm-4 (충돌 재검증 — §2 #12) |
| 49 | zong2018dagmm | ✓ | ✓ | ✓ Dae-ki Cho(DBLP 표기) | VERIFIED_A | 양형 attested 기록 일치 |

**card frontmatter 비고**: 49장 전수 확인 — 47장 `VERIFIED_A`, 2장 `VERIFIED_2CH`(blazquez, lai).
표기 비균일이나 의미 모순은 아님 (card는 A채널 산출물; 2채널 판정의 정본은 통합 ledger) — §5 MINOR gm-1.

---

## 2. 무작위 재검증 라운드 (16편 슬롯 / 고유 14편)

재현 절차 (검증 가능):

```python
import random
keys = sorted(<refs.bib 49 key 전체>)   # 알파벳순
random.seed(42)
sample = random.sample(keys, 10)
# → ['wu2025catch','deng2021gdn','ahmed2017wadi','hundman2018telemanom','huang2022slavae',
#    'he2022mae','deng2022reverse','darban2024dacad','su2019omnianomaly','blazquez2021review']
```

충돌 해소 6편(지정): blazquez2021review, darban2024dacad, lai2023npsr, zhang2022selfdistill,
sultani2018deepmil, wang2025nrdetector. **blazquez·darban이 무작위 표본과 중복 → 고유 14편.**

방법: DBLP `.bib` export 직접 curl(13편) + Crossref API DOI 질의(wang2025nrdetector) — 2026-06-11 게이트 감사 시점
신규 fetch. refs.bib 항목과 **author(전원)/title/year/pages/doi/volume/number/journal·booktitle/publisher/series
전 필드 기계 비교**.

| # | 구분 | key | 공식 소스 (직접 fetch URL) | 결과 |
|---|------|-----|---------------------------|------|
| 1 | 무작위 | wu2025catch | https://dblp.org/rec/conf/iclr/WuQL0HGXY25.bib | **전 필드 일치** (저자 8인; pages/doi 부재 = ICLR 표준) |
| 2 | 무작위 | deng2021gdn | https://dblp.org/rec/conf/aaai/DengH21.bib | **전 필드 일치** (pp.4027–4035, DOI 10.1609/AAAI.V35I5.16523) |
| 3 | 무작위 | ahmed2017wadi | https://dblp.org/rec/conf/cpsweek/AhmedPM17.bib | **전 필드 일치** (저자 3인, pp.25–28, DOI 10.1145/3055366.3055375) |
| 4 | 무작위 | hundman2018telemanom | https://dblp.org/rec/conf/kdd/HundmanCLCS18.bib | **전 필드 일치** (Söderström diacritic 포함, pp.387–395) |
| 5 | 무작위 | huang2022slavae | https://dblp.org/rec/conf/www/HuangCL22.bib | **전 필드 일치** (pp.1797–1806, DOI 10.1145/3485447.3511984) |
| 6 | 무작위 | he2022mae | https://dblp.org/rec/conf/cvpr/HeCXLDG22.bib | **전 필드 일치** (Piotr Dollár 포함, pp.15979–15988 — B의 구판 16000–16009 기각 판단 재확인) |
| 7 | 무작위 | deng2022reverse | https://dblp.org/rec/conf/cvpr/DengL22.bib | **전 필드 일치** (pp.9727–9736, DOI 10.1109/CVPR52688.2022.00951) |
| 8 | 무작위+충돌② | darban2024dacad | https://dblp.org/rec/journals/tkde/DarbanYWAWPS25.bib | **전 필드 일치** — TKDE 37(8):4485–4496, 2025, DOI 10.1109/TKDE.2025.3569909, 저자 7인 (해소② 채택 정당) |
| 9 | 무작위 | su2019omnianomaly | https://dblp.org/rec/conf/kdd/SuZNLSP19.bib | **전 필드 일치** (저자 6인, pp.2828–2837) |
| 10 | 무작위+충돌① | blazquez2021review | https://dblp.org/rec/journals/csur/Blazquez-Garcia21.bib | **전 필드 일치** — DBLP year=2022 (해소① "인쇄판 2022 채택" 정당; A의 2021=Crossref 온라인 게재일도 사실로 확인) |
| 11 | 충돌③ | lai2023npsr | https://dblp.org/rec/conf/nips/LaiSGLB23.bib | **전 필드 일치** — 4th author **"Jeffrey H. Lang"**, "Duane S. Boning" (해소③ 정당 — A측 card 오류였음을 재확인; card 정정 반영 확인) |
| 12 | 충돌④ | zhang2022selfdistill | https://dblp.org/rec/journals/pami/ZhangBM22.bib | **전 필드 일치** — Zhang/Bao/Ma, TPAMI 44(8):4388–4403, DOI 10.1109/TPAMI.2021.3067100 (재export 해소④ 정당) |
| 13 | 충돌⑤ | sultani2018deepmil | https://dblp.org/rec/conf/cvpr/SultaniCS18.bib | **실질 필드(저자·제목·연도·pages 6479–6488·DOI 10.1109/CVPR.2018.00678) 전부 일치** — 해소⑤ DOI 정당. 단 booktitle("2018 IEEE Conference…" vs refs.bib "Proceedings of the IEEE/CVF…")·publisher("Computer Vision Foundation / IEEE Computer Society" vs "IEEE") 표기 일탈 — 손조립 흔적 (MINOR gm-3). 참고: 현재 DBLP export에 DOI 직접 포함 — verbatim 교체 가능 |
| 14 | 충돌⑥ | wang2025nrdetector | https://api.crossref.org/works/10.1145/3690624.3709257 + https://dblp.org/rec/conf/kdd/Wang0XWJSZZ025.bib | **서지 전 필드 일치** — Crossref: 저자 9인 전원·제목·KDD '25 V.1·**pages 1551–1562**·DOI (해소⑥ 정당). **추가 발견: DBLP가 현재 색인 완료** (key conf/kdd/Wang0XWJSZZ025, pages 1551–1562 — 3중 수렴; B2 VERIFY_REQUIRED 2건 폐쇄 가능). 단 refs.bib 항목 자체는 구문 결함 — §3 BLOCKER |
| 15 | (중복 슬롯) | blazquez2021review | — #10과 동일 | 동일 |
| 16 | (중복 슬롯) | darban2024dacad | — #8과 동일 | 동일 |

**재검증 실패(필드 불일치) 0건. 16/16 슬롯(고유 14편) 서지 확인.**

---

## 3. refs.bib 형식 무결성 — **BLOCKER 1건**

| 항목 | 결과 |
|------|------|
| 항목 수 | 49 (raw key 추출 기준) — REFERENCES_IEEE.md 49항목과 일치 |
| key 중복 | 0 |
| 필수 필드 (author/title/year + venue류) | 전 항목 존재 (파싱 가능한 48항목 기계 확인 + wang2025 수동 확인) |
| 공식 export 출처 코멘트 | 전 항목 존재 (`% source:` 50개 — darban은 구판+재export 2중 코멘트) |
| **파싱** | **48/49 — wang2025nrdetector 탈락** |

### GB-1 (BLOCKER) — wang2025nrdetector 항목 파싱 불가

`refs.bib` 887–890행:

```
  doi          = {10.1145/3690624.3709257}          ← 후행 콤마 누락
  % pages: VERIFY_REQUIRED (ACL DL returned 403; exact pages unconfirmed),   ← 항목 내부 % 코멘트 (BibTeX는 항목 내부 %를 코멘트로 취급하지 않음)
  pages        = {1551--1562}
```

실측 (2026-06-11):
- **실제 bibtex 0.99d (TeX Live)**: `I was expecting a ',' or a '}'---line 889 of file refs.bib` → **"I'm skipping whatever remains of this entry"** — 항목 전체 탈락.
- **bibtexparser 1.4.4**: 무경고 silent drop — 48개만 파싱.

영향: wang2025nrdetector는 **최다 인용 reference**(CLAIM_CITATION_MAP 14개 claim 커버 — C-003, 005, 006, 007, 010,
017, 022, 024, 046, 052, 073, 074, 078, 079)다. "정본 서지는 refs.bib만 사용" 인계 규칙(VERIFICATION_LEDGER §4-1)
하에서 Phase 7 빌드 시 이 인용 전부가 미해결 인용으로 깨진다. 동일 결함이 `refs_B2.bib` 312–313행에도 존재(원천).

수정 권고 (둘 중 하나):
1. (권장) 현재 DBLP가 색인 완료했으므로 **verbatim export로 교체**: `https://dblp.org/rec/conf/kdd/Wang0XWJSZZ025.bib`
   (pages 1551–1562 포함 — 본 감사에서 fetch·대조 완료, refs.bib 내용과 서지 동일). "공식 export 기반" 규칙에도 부합.
2. (최소) doi 행 끝에 `,` 추가 + 889행 `%` 코멘트를 항목 밖(@inproceedings 위)으로 이동.

---

## 4. QUARANTINE 무결 + EXCERPT_UNVERIFIED 격리

### QUARANTINE — 통과
- 통합 ledger·A1·A2·B1·B2·diff report 전수 grep: QUARANTINE 판정 항목 **0건 사실 확인** (A1 §7 "없음", A2 표 0, diff 0/49).
- refs.bib 49 key = 통합 ledger 49 key = REFERENCES_IEEE 49항목 = card 49장: **비검증 항목 혼입 0**.
- CLAIM_CITATION_MAP DOI 전수 추출(기계) ↔ refs.bib 대조: refs.bib 외 DOI는 2건뿐이며 **둘 다 인용 경로 아님** —
  ① Dist-PU(10.1109/CVPR52688.2022.01406): C-019 행 내 **"미채택" 경고 주석** (인용 후보 아님).
  ② Gao TSMAE(10.1109/TNSE.2022.3163144): §5.4 모델명 충돌 보고 (인용 아님).
  그 외 비검증 언급(AEGR Soft Computing 2021 — C-025 서술 금지 주석, Takahashi OpenReview — C-032 "인용 부적격" 명시)도
  전부 부정·경고 용도로만 존재. **어떤 claim 행도 비검증/QUARANTINE 항목을 채택 reference로 가리키지 않음.**
- claim 매트릭스 기계 집계: 85행 전수 = VERIFIED(2채널) 78 + NOT_FOUND 1(C-032 — 의도된 부재) + 인용 불필요 6 —
  문서 말미 상태 선언과 정확히 일치.

### EXCERPT_UNVERIFIED 잔존 3건 격리 — 통과
| key | ledger §3 등재 | card 마킹 (확인 위치) |
|-----|---------------|----------------------|
| zhang2022selfdistill | ✓ | `INTERNAL_NOTE_NOT_FOR_MANUSCRIPT` + "원고에 그대로 복사 금지" 경고 + verified_note에 EXCERPT_UNVERIFIED 명기 + "내용 인용 금지" (53행) |
| xu2018kpivae | ✓ (ACM 게재본 대조 잔존) | 27행 "verbatim 발췌·abstract는 검증/문체 대조 전용 — 복사·근접 의역 절대 금지 (A2)" |
| ruff2020deepsad | ✓ (SAD loss §3) | `INTERNAL_NOTE_NOT_FOR_MANUSCRIPT` + verified_note에 §3 EXCERPT_UNVERIFIED 명기 + 복사 금지 표현 목록 |

2단계 격리 규칙(서지 인용 가능 / verbatim 금지)이 통합 ledger §3, REFERENCE_LIBRARY_INDEX(V2* 표기),
REFERENCES_IEEE 말미 주석에 일관 전파됨. wang2025nrdetector preprint-기준 발췌 caveat도 3개 문서에서 일관 유지.

---

## 5. 발견 사항 (rubric별)

### BLOCKER (1)
- **GB-1**: refs.bib wang2025nrdetector 항목 파싱 불가 (§3 상세). 게이트 통과 차단 항목.

### MAJOR (2)
- **GM-1 (기록 불완전 — A1 ledger 내부 모순)**: A1 §1 요약표 "정정 필드 18개(9개 카드 major)" / "EXCERPT_UNVERIFIED
  해소 7건" ↔ 같은 문서 §6 "총 정정 필드 22개(CRITICAL 1)" / §5 "해소 11건". 상세 절(§5·§6)이 정본이고 통합
  ledger·diff report는 상세 절 수치(11건·22개)를 인용하므로 하류 오염은 없으나, 요약 통계가 자체 상세와 불일치.
- **GM-2 (기록 불완전 — diff 버킷 집계 비재현)**: "완전 일치 33 / 표기 관례 10 / 해소 6"이 통합 ledger 표의 행별
  주석으로 재현 불가 — 행별 표기는 일치 28 / 일치(venue 표준 부재 합의) 12 / 표기 관례 3 / 해소 6. P4_DIFF_REPORT
  §2도 "표기 관례 10건" 제하에 13개 이상 key를 열거. §2 말미에 "버킷 구분은 orchestrator 기계 diff 산출 기준"
  caveat가 있으나 산출 기준 자체가 기록되지 않아 감사 불가. (실질 검증 결과에는 영향 없음 — 6건 해소·0 QUARANTINE은
  행 단위로 전부 재확인됨.)

### MINOR (6)
- **gm-1**: card verification_status 표기 비균일 — 47장 `VERIFIED_A` vs 2장 `VERIFIED_2CH`(blazquez, lai).
  INDEX는 전 49편 V2로 집계. 의미 모순은 아니나 표기 통일 권장.
- **gm-2**: refs.bib darban2024dacad 항목 위에 구판 B1 코멘트("No peer-reviewed 2024 venue found … arXiv preprint
  used")가 잔존 — 바로 아래 재export 코멘트와 상충하는 인상. 구판 코멘트 삭제 권장.
- **gm-3**: sultani2018deepmil refs.bib 항목이 verbatim DBLP export가 아님(손조립 — booktitle·publisher 일탈; 항목
  코멘트에 자인). 실질 필드는 본 감사 직접 fetch로 전부 정확 확인. 파일 헤더의 "손 타이핑 금지" 규칙과 형식 불일치 —
  현 DBLP export(DOI 포함)로 verbatim 교체 권장.
- **gm-4**: VERIFICATION_LEDGER_B2.md의 zhang2022selfdistill 본문이 재export 후에도 SdAE 오매칭 기록 그대로
  (재export 기록은 master/diff/refs_B2.bib 코멘트에만 존재). B2 ledger에 1줄 포인터 추가 권장.
- **gm-5**: luo2024moderntcn Spotlight — CLAIM_CITATION_MAP C-068 셀은 scout 시점 "Spotlight — OpenReview 직접 확인"
  서술 잔존, A2는 공식 확인 불가(DBLP poster) 판정, B2는 iclr.cc로 확인. master/INDEX의 "Phase 7 재확인" caveat가
  정본이나 C-068 셀이 이를 반영하지 않아 원고 작성자 오인 소지.
- **gm-6**: CLAIM_CITATION_MAP §4 수요 통계(인용 필요 72 / 필수 52 / 권장 20)가 현 매트릭스 실측(인용 강도 보유
  79행 = 필수 61 / 권장 18)과 불일치 — scout r2 시점 집계 잔존으로 추정. 핵심 상태 통계(VERIFIED 78 / NOT_FOUND 1 /
  불필요 6)는 실측 일치.

### 폐쇄 가능 항목 (감사 중 신규 확인)
- wang2025nrdetector의 DBLP 색인 완료 (key `conf/kdd/Wang0XWJSZZ025`, pages 1551–1562) — B2 VERIFY_REQUIRED 2건
  (pages, DBLP key) 및 A2 "DBLP 미색인" 기록을 공식적으로 폐쇄할 수 있음. GB-1 수정 시 이 export를 그대로 쓰면 일거양득.

---

## 6. Phase 4 Directive 충족 근거 (제안 문자열)

- **T4 (reference 탐색·발췌·검증·IEEE 정리)**: "49/49 reference 2채널 독립 검증(A: card↔공식소스 / B: blind 공식
  BibTeX export) + orchestrator 기계 diff 통과, QUARANTINE 0; 발췌 13건 해소·잔존 3건은 2단계 격리(서지 인용 가능·
  verbatim 금지) 마킹 완료; 정본 refs.bib(공식 export 기반 49항목·key 중복 0) + REFERENCES_IEEE.md 49항목 정리.
  게이트 감사(r1)에서 ledger 전수 재감사 모순 0 + 무작위·충돌 16편 공식 소스 재검증 전건 일치 — 단 refs.bib
  wang2025nrdetector 1항목 BibTeX 구문 결함(GB-1) 수정 조건부."
- **R26 (Notion truth 활용 + 공식 재확인 — venue 정정 4건 포함)**: "데이터셋 5종(C-040~C-044)·baseline 22+4종
  (C-054~C-073)·지표 4종(C-047~C-051) 인용을 R26 truth(NOTION_DIGEST §II-2·II-3 → EXPERIMENT_PROTOCOL_TRUTH §①·④)
  에서 출발해 전건 공식 소스(DBLP/Crossref/OpenReview/PMLR/publisher PDF)로 재확인 — venue 정정 4건 포함(WETAS
  ICML 추정→**ICCV 2021**, TreeMIL ICML/NeurIPS 추정→**ICASSP 2024**, Dist-PU AAAI 표기→**CVPR 2022**(미채택 처리),
  DACAD arXiv→**TKDE 2025 본판**); lee2021wetas·liu2024treemil DOI의 R26 truth 일치 재확인 기록(ledger 행 23·26)."
- **R36 1차 (수요 매핑 체계 가동)**: "CLAIM_CITATION_MAP이 PAPER_BLUEPRINT r3 전 섹션 전수 추출로 claim 지점
  C-001~C-085 85행의 인용 수요를 매핑 — 행별 블루프린트 위치·근거 유형·후보 reference·인용 강도(필수/권장)·R19
  분류·검증 상태를 부여하고, 수요→scout 후보(OPEN 31→CANDIDATE 30+NOT_FOUND 1)→2채널 검증(VERIFIED 78)→배치
  지정의 전 주기를 §4 수요 통계와 §6 갱신 로그로 추적 — 수요 매핑 체계 가동 입증 (게이트 감사 기계 집계: 85행 =
  VERIFIED 78 + NOT_FOUND 1 + 인용 불필요 6, 비검증 인용 경로 0)."

---

## 7. 종합 판정

**FAIL (조건부)** — rubric상 BLOCKER 1건(GB-1: 정본 refs.bib의 wang2025nrdetector 항목 파싱 불가 — 실제 bibtex가
항목 전체를 폐기, 14개 claim의 인용이 빌드에서 소실될 결함)이 존재하므로 게이트를 통과시킬 수 없다.

단, 검증의 **실질**은 건전하다: ① ledger 전수 재감사에서 A↔B↔refs.bib↔card 간 검증 기록 모순 0건, ② 무작위+충돌
16편 슬롯 공식 소스 재검증에서 필드 불일치 0건(해소 6건의 채택 근거 전부 독립 재확인 — lai의 "Jeffrey H. Lang",
blazquez year=2022, darban TKDE 2025, zhang TPAMI 재export, sultani DOI, wang pages 1551–1562), ③ QUARANTINE 0 +
비검증 인용 경로 0, ④ 격리 규칙 마킹 완전.

**재게이트 조건**: GB-1 수정(권장: DBLP `conf/kdd/Wang0XWJSZZ025` verbatim export 교체 — 본 감사에서 서지 동일성
확인 완료) + `bibtex`/`bibtexparser` 49/49 파싱 확인. MAJOR 2건(GM-1·GM-2)은 요약 통계 정정으로 동시 처리 권장,
MINOR 6건은 Phase 7 전 일괄 처리 가능.
