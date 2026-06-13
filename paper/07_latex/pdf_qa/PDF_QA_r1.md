---
phase: 7
agent: pdf-qa-reviewer
directives: [T7, R6, R7]
last_modified: 2026-06-11
inputs:
  - main.pdf (preprint,12pt; 48 PDF pages = Highlights + printed 1–47; letter)
  - main_5p_measure.pdf (final,5p,times,twocolumn; 18 PDF pages = Highlights + printed 1–17; A4)
  - main.log / main_5p_measure.log (overfull: 26 / 17건)
  - 05_manuscript/PLACEHOLDER_REGISTRY.md (float 크기 가정 대조)
method: 전 페이지 시각 검수 (Read pages 파라미터로 두 PDF의 모든 페이지를 직접 열람)
  + pdftotext '??' 스캔 + 로그의 overfull→소스파일 매핑 + .tex/.bib 원인 확인(읽기 전용)
---

# PDF QA r1 — 컴파일 결과 시각 검수 (T7) + 분량 분석 (R6/D-012) + Appendix 구성 (R7)

## 0. 판정 요약

| 빌드 | 판정 | 핵심 근거 |
|---|---|---|
| `main_5p_measure.pdf` (게이트 판형) | **FAIL (BLOCKER)** | p.6·p.8·p.15·p.17에서 float–본문 텍스트 겹침(overprint) — Table 2(sideways)가 2단 모드에서 본문/Table 3 위에 인쇄됨. 분량 실측 **본문 ≈ 9.2p** (>9p, 단 깨진 레이아웃 위의 측정이라 신뢰도 제한) |
| `main.pdf` (배포 기준) | **CONDITIONAL FAIL (MAJOR 다수, 겹침 없음)** | 페이지 겹침은 없으나 표 4개(1, A.4, A.9, C.16)에서 우측 마진 밖 내용 잘림(내용 손실), `(author?)` 깨진 인용 2회, Fig 2–4가 참조 위치에서 ~20–30페이지 떨어진 문서 말미로 표류 |

미해석 참조(`??`)는 **양 빌드 0건** (pdftotext 전수 스캔). 인용 번호 [1]–[48] 정상, 식 (1)–(6)/(C.1)–(C.5) 정상.

---

## 1. main_5p_measure.pdf — 페이지 단위 검수표 (전 18p)

표기: PDF p = 파일 페이지, 인쇄 p = 하단 페이지 번호 (PDF p1 = Highlights, 무번호; 인쇄 p = PDF p − 1).

| PDF p | 인쇄 p | 내용 | 상태 | 비고 |
|---|---|---|---|---|
| 1 | — | Highlights | OK | elsarticle 규약대로 별도 페이지, 본문에 끼지 않음. bullet 5에 `[N]` placeholder (NUM-003, 예상됨) |
| 2 | 1 | Title/저자/Abstract/Keywords + §1 시작 | OK | 저자·소속·저널명 placeholder 정상 렌더. 소속 줄에 잔여 쉼표 ", , , , ," (MINOR-5). "Appendix Appendix A.3" 중복 첫 출현 (MAJOR-4) |
| 3 | 2 | FIG-1 placeholder + §1 기여/§2 시작 | OK | placeholder 박스+캡션 정상, 1단 폭 ~0.26p |
| 4 | 3 | §2.2–§3.1, 각주 1–2 | OK | 100% 산문 |
| 5 | 4 | FIG-2 placeholder + §3.2–3.4 | OK | 박스 ~0.30p |
| 6 | 5 | §3.4–3.6, Eq (1)–(5) | OK | 100% 산문/수식 |
| 7 | 6 | **Table 1 + §4.1.1** | **BLOCKER-1** | Table 1(table*)이 우측 폭 초과(overfull 166pt ≈ 5.8cm)로 **우측 컬럼 본문 텍스트와 겹쳐 인쇄** — "Test AR" 열 영역에 §4.1.1 산문이 overprint되어 양쪽 모두 판독 불가 |
| 8 | 7 | §4.1.2–4.2 | OK | 100% 산문 |
| 9 | 8 | **Table 2(sideways) + Table 3 + §4.2–4.4** | **BLOCKER-2** | `sidewaystable`(비-* 환경)이 twocolumn에서 컬럼 float로 처리되어 **회전된 Table 2 전체가 Table 3 및 §4.2–4.3 본문 위에 겹쳐 인쇄** — 페이지의 절반 이상이 이중 인쇄로 판독 불가. 본 빌드 최대 결함 |
| 10 | 9 | FIG-3 + FIG-4 placeholder + §4.4–4.5 + §5 시작 | OK | 두 placeholder 각 1단 폭, 합 ~0.30p |
| 11 | 10 | **§5 결론 종료(좌측 컬럼 ~42% 지점)** + References 시작 | OK | **본문 종점 = 인쇄 p10의 21% 지점 → 본문 실측 9.2p** |
| 12–13 | 11–12 | References [9]–[48] | OK | 2단 참고문헌 정상 |
| 14 | 13 | Refs 종료 + Appendix A 시작 + Table A.4/A.5 | **MAJOR-1** | Table A.4 우측 마진 초과·**페이지 밖 잘림**: "d_model = 512…", "Linear(512→256)→…→Li", "(Eq. C.4…" 등 셀 내용 절단 (overfull 133pt) |
| 15 | 14 | Table A.6/A.7 + §A.2–A.3 | **MAJOR-2** | Table A.6 "Key parameters" 열이 마진 침범(절단 직전), Table A.7 우측 가장자리 도달 |
| 16 | 15 | §A.4–B.3 + Table A.8/A.9/A.10 | **BLOCKER-3** | **Table A.8(좌측 컬럼)과 Table A.9/A.10(우측, 좌측으로 돌출)이 상호 겹침**; A.9/A.10 우측 열("excl22" 그룹)은 페이지 밖으로 절단 (overfull 232pt ≈ 8.2cm) |
| 17 | 16 | Table A.11/B.12/B.13/B.14 + Appendix C.1 | OK | 이 페이지 자체는 정상 |
| 18 | 17 | FIG-B1 placeholder + Table B.15 + **Algorithm 1** + Table C.16 + §C.2–C.4 | **BLOCKER-4** | **Algorithm 1(우측 컬럼)이 Table B.15 우측 열 및 Table C.16과 겹쳐 인쇄** — 알고리즘 본문과 표 셀이 상호 overprint. Table C.16 "Derivation" 열도 절단 |
| (18) | (17 하단) | §C.3/C.4 헤더 + Table C.17 | MINOR | Table C.17(notation)은 다음 영역에 단독 배치, 상단 대형 여백 |

---

## 2. main.pdf (preprint,12pt) — 페이지 단위 검수표 (전 48p)

| PDF p | 인쇄 p | 내용 | 상태 | 비고 |
|---|---|---|---|---|
| 1 | — | Highlights | OK | `[N]` placeholder 1건 (예상) |
| 2 | 1 | Title/Abstract | OK | 소속 줄 잔여 쉼표 (MINOR-5). Abstract가 p2로 이월 — preprint 규약상 정상 |
| 3 | 2 | Abstract 끝/Keywords/§1 | OK | "Appendix Appendix A.3" 중복 (MAJOR-4) |
| 4–5 | 3–4 | §1 기여 + FIG-1 placeholder | OK | FIG-1은 §1 참조 위치 부근(p4)에 정상 배치 |
| 6–10 | 5–9 | §2–§3.4 | OK | 100% 산문, 각주 정상 |
| 11 | 10 | §3.4 GRL dual-λ | **MAJOR-3** | "**(author?)** [36]" — `\citet{ganin2016dann}`이 숫자 스타일에서 깨짐 (시각적으로 노출) |
| 12–13 | 11–12 | §3.5–3.6, Eq (1)–(5) | OK | |
| 14 | 13 | **Table 1** + Eq (6) + §4.1.1 | **BLOCKER-A** | Table 1 "Test AR (%)" 열이 우측 마진 밖 **절단**: "19.05 (full) / 3.68 (e…"에서 잘림 — excl22 수치 정보 손실 (overfull 140pt) |
| 15–18 | 14–17 | §4.1.2–4.2 | OK | |
| 19 | 18 | **Table 2 (sideways, 전용 페이지)** | **MAJOR-6** | 회전 배치 자체는 성공했으나 표가 회전 방향으로도 과폭(overfull 148pt): **MSL avg 열 그룹이 페이지 가장자리에서 절단** ("V[US]…", "[X.X…" 잘림) |
| 20 | 19 | Table 3 + §4.2 결론부/§4.3 | OK | Table 3 정상 (캡션 완전) |
| 21–22 | 20–21 | §4.4–§5 | OK | |
| 23 | 22 | §5 종료(상단 2줄) + References 시작 | OK | 본문 = 인쇄 21.05p (preprint 환산; 게이트 지표 아님) |
| 23–31 | 22–30 | References [1]–[48] | OK | ref [36] Ganin 항목 자체는 정상 렌더. 긴 URL 줄바꿈 정상 |
| 31–32 | 30–31 | Appendix A 시작 + **Table A.4** | **BLOCKER-B** | Table A.4 GRL head 행 절단: "2-layer MLP: LayerNorm→Linear(512→256)→GELU→Dropout(0.1)→Li" 에서 **페이지 밖으로 잘림** (overfull 133pt) |
| 33–35 | 32–34 | Table A.5 + §A.1–A.4 | OK | Table A.5 정상 |
| 36 | 35 | §A.5/A.6(헤더만)/§B.1–B.3 | MINOR-7 | A.5·A.6이 본문 없는 빈 소절(헤더 연속) |
| 37 | 36 | §B.4/B.5 + **Appendix C.1** | **MAJOR-3** | "(author?)" 2번째 출현 (C.1 reversal-coefficient 문단) |
| 38 | 37 | Eq (C.3)–(C.5) + §C.2–C.4 헤더 | MINOR | C.3/C.4 헤더가 페이지 하단에 본문 없이 고립 |
| 39 | 38 | **FIG-2 placeholder** | **MAJOR-8** | §3.2(인쇄 p8)에서 참조되는 그림이 **30페이지 떨어진 문서 말미로 표류** |
| 40 | 39 | **FIG-3 placeholder** | MAJOR-8 | §4.4(p20) 참조 → p39 배치 (19p 거리) |
| 41 | 40 | **FIG-4 + FIG-B1 placeholder** | MAJOR-8 | §4.5(p21) 참조 → p40. FIG-B1 캡션이 "**Figure B.5**"로 렌더 (MAJOR-5 번호 체계) |
| 42 | 41 | Table A.6 | OK | preprint 폭에서는 정상 수용 |
| 43 | 42 | Table A.7/A.8/**A.9** | **BLOCKER-C** | Table A.9 "excl22 condition" 열 그룹이 우측 가장자리 **절단** ("VUS-R…", "[X.X…" 잘림; overfull 181pt). A.7 "Source" 열도 마진 끝 도달 |
| 44 | 43 | Table A.10/A.11 | OK | |
| 45 | 44 | Table B.12/B.13/B.14 | OK | |
| 46 | 45 | Table B.15 + **Table C.16** | **MAJOR-9** | C.16 "Derivation" 열 절단: "…{P202, P401, P404, P502, P601," 에서 잘림 (overfull 153pt) |
| 47 | 46 | **Algorithm 1** | **MAJOR-10** | 알고리즘 박스가 하단 마진 초과 — 페이지 번호 "46"이 알고리즘 24–25행 사이에 **겹쳐 인쇄**, return 행이 푸터 영역 침범 |
| 48 | 47 | Table C.17 (notation) | OK | 정상. 문서 종료 |

---

## 3. 발견 사항 (severity별 통합)

### BLOCKER (페이지 깨짐 / 내용 손실)
| ID | 빌드 | 위치 | 내용 | 근본 원인 (소스 확인) |
|---|---|---|---|---|
| BLOCKER-2 | 5p | 인쇄 p7 | sideways Table 2가 Table 3·본문 위에 overprint — 페이지 절반 판독 불가 | `sec4_experiments.tex:193` `\begin{sidewaystable}` — rotating 패키지의 비-* 환경은 twocolumn에서 컬럼 float로 취급되어 배치 파탄. `sidewaystable*` 또는 직립 `table*` 필요 |
| BLOCKER-1 | 5p | 인쇄 p6 | Table 1이 우측 컬럼 본문과 겹침 | 표 자연폭 > \textwidth (overfull 166pt). Test AR 열 축약 필요 |
| BLOCKER-3 | 5p | 인쇄 p15 | Table A.8 ↔ A.9/A.10 상호 겹침 + A.9/A.10 우측 절단 | A.9(27열)·A.10이 컬럼 폭을 8cm 초과 (overfull 232pt) — table* + \scriptsize 필요 |
| BLOCKER-4 | 5p | 인쇄 p17 | Algorithm 1 ↔ Table B.15/C.16 겹침 | algorithm2e 박스가 컬럼 폭/높이 초과 |
| BLOCKER-A | preprint | 인쇄 p13 | Table 1 Test AR 열 페이지 밖 절단 (excl22 수치 손실) | 동일 폭 초과 |
| BLOCKER-B | preprint | 인쇄 p31 | Table A.4 GRL head 행 절단 (아키텍처 사양 손실) | Value 열 미줄바꿈 — p{} 열 지정 필요 |
| BLOCKER-C | preprint | 인쇄 p42 | Table A.9 excl22 열 그룹 절단 (full/excl22 이중 조건의 절반 손실) | 2조건×5지표 = 11열이 본문 폭 초과 |

### MAJOR (overflow / 배치 / 참조 체계)
| ID | 빌드 | 내용 | 근본 원인 |
|---|---|---|---|
| MAJOR-3 | 양쪽 | "**(author?)** [36]" 가시적 깨진 인용 ×2 (§3.4 본문, §C.1) | `\citet{ganin2016dann}` (sec3_method.tex:165, appendix_C.tex:12) — elsarticle-num(숫자) 스타일에서 \citet의 저자명 미생성 (natbib 경고 "Author undefined" 2건과 일치). "Ganin et al.~\cite{…}"로 교체 필요 |
| MAJOR-4 | 양쪽 | "Appendix **Appendix** A.3" 중복 표기 — 본문·캡션 전반 ~20회 | 소스의 `Appendix~\ref{…}` 패턴 (예: sec1_intro.tex:19) — elsarticle은 appendix 섹션 \ref가 이미 "Appendix A.3"을 산출 |
| MAJOR-5 | 양쪽 | **Appendix float 번호 체계 깨짐**: Table A.1→"A.4", A.3→"A.6", B.1→"**B.12**", C.1→"**C.16**"; Figure B.1→"**B.5**" — 본문 카운터가 appendix에서 리셋되지 않고 연속 | `\setcounter{table}{0}` + `\renewcommand{\thetable}{A.\arabic{table}}` 부재 (main.tex에 카운터 리셋 없음 확인). REGISTRY 명명(A.1…)과 전면 불일치 |
| MAJOR-6 | preprint | Table 2가 회전 후에도 과폭 → MSL 열 절단 | 7패밀리×2지표=15열 + 그룹 라벨. 약어/\tabcolsep 축소 필요 (fallback ladder (a)) |
| MAJOR-8 | preprint | Fig 2/3/4가 참조 위치에서 19–30페이지 떨어진 말미(p38–40)로 표류 — refs 뒤·appendix 한가운데 본문 그림이 출현 | 대형 placeholder parbox가 float 배치 제약을 못 넘겨 문서 끝까지 누적 표류 |
| MAJOR-9/10 | preprint | Table C.16 Derivation 열 절단 / Algorithm 1이 푸터·페이지번호와 겹침 | 폭 미지정 열 / 알고리즘 박스 높이 > \textheight 잔여분 |
| MAJOR-11 | 양쪽 | **Algorithm 1 내부의 스테일 참조**: 입력부가 "configuration **Table A.1**" — 실제 렌더 번호는 Table A.4 | `appendix_C.tex:113`에 하드코딩된 "Table~A.1" (\ref 미사용). MAJOR-5 수정 시에도 \ref로 교체해야 안전 |

### MINOR (미관)
- MINOR-5: 타이틀 소속 줄 "[AFFILIATION — to be filled], , , , ," 잔여 쉼표 (빈 주소 필드들).
- MINOR-7: §A.5/A.6이 헤더만 있는 빈 소절 (표 포인터 한 줄이라도 권장).
- MINOR-12: §C.3/C.4 헤더가 내용과 분리되어 페이지 하단 고립 (양 빌드).
- MINOR-13: 5p 인쇄 p17 상단(=Table C.17 페이지) 대형 공백 — float 단독 페이지화.
- (참고) placeholder 박스 자체의 렌더 품질은 양호: 회색 박스 + 캡션 + 크기 가정 명시, FIG/TAB/ALG 전 항목 식별 가능.

빌드 로그의 overfull 26건(main) / 17건(5p) 중 **>30pt 대형 건은 전부 위 표의 표/알고리즘 폭 초과와 1:1 대응**함을 확인 (소스파일 매핑: sec4_experiments 2–6건, appendix_A 8건, appendix_B 3–4건, appendix_C 2건). 10pt 미만 잔여 건은 시각적으로 무해.

---

## 4. 분량 분석 (5p 빌드 — D-012 / R6)

### 4.1 본문 실측
- 본문 종점: **§5 Conclusion이 인쇄 p10 좌측 컬럼 약 42% 지점에서 종료** ("Code is available at [URL] …acceptance).") → p10 기여분 = 0.5(컬럼) × 0.42 ≈ 0.21p.
- **본문 실측 = 9 + 0.21 ≈ 9.2p** (Highlights 페이지 제외; References p10 잔여~p12, Appendix p13–17).
- 판정 기준 ≤9p & ≥8.5p 대비 **+0.2p 초과** — 단, 아래 주의 필요:
  - **측정 신뢰도 경고**: p6·p8의 float 겹침으로 Table 1·Table 2가 본문 공간을 "공짜로" 점유 중. Table 2를 규정대로 배치하면(회전 전용 float 페이지) 본문은 **~9.6p까지 증가**할 수 있음. 즉 현 9.2p는 하한치.

### 4.2 페이지별 구성 비율 (본문 p1–p10)
| 인쇄 p | 산문/수식 | float | 비고 |
|---|---|---|---|
| 1 | 0.38 | — (frontmatter 0.62) | title/abstract/keywords |
| 2 | 0.74 | 0.26 (FIG-1) | |
| 3 | 1.00 | — | |
| 4 | 0.70 | 0.30 (FIG-2) | |
| 5 | 1.00 | — | |
| 6 | 0.73 | 0.27 (TAB-1) | 겹침으로 비율 왜곡 |
| 7 | 1.00 | — | |
| 8 | ~0.45 | ~0.55 (TAB-2+TAB-3) | 겹침 — 측정 불가 영역 |
| 9 | 0.70 | 0.30 (FIG-3+FIG-4) | |
| 10 | 0.21 | — (이후 refs) | 본문 종점 |
| 합 | ≈ 6.9p 산문 | ≈ 1.7p float + 0.6p frontmatter | |

### 4.3 REGISTRY 크기 가정 대비 — 초과 float 식별
| Float | REGISTRY 가정 | 실측(5p) | 판정 | 권고 |
|---|---|---|---|---|
| FIG-1 | 0.40p | 0.26p | 여유 | 실그림도 1단 폭이면 가정 내 |
| FIG-2 | 0.40p | 0.30p | 여유 | |
| TAB-1 | 0.25p | 0.27p | **폭 초과(질적)** | 면적은 가정 내, 폭이 5.8cm 초과 — "19.05 (full) / 3.68 (excl22)" → "19.05 / 3.68†" 각주화, #Train/Test 천단위 축약 |
| **TAB-2** | **0.55p (sideways)** | 측정 불가 (겹침); 올바른 sideways 배치 시 **사실상 1.0p** (회전 전용 페이지) | **유일한 실질 초과 float (+0.45p)** | fallback ladder (a) 실행: 직립 `table*` + \scriptsize + \tabcolsep 2–3pt + 헤더 약어(F1/VP) + 메서드명 축약 → 0.55–0.6p 목표. preprint에서도 회전 후 절단되는 현 27행×15열은 어떤 판형에서도 무가공 수용 불가 |
| TAB-3 | 0.20p | 0.13p | 여유 | |
| FIG-3 | 0.33p | 0.25p | 여유 | |
| FIG-4 | 0.30p | 0.28p | 가정 내 | |
| 합계 | 2.43p | ≈1.7p (placeholder) / 위험 시나리오 ≈2.3p | | |

### 4.4 회수 가능 분량 추정
- TAB-2를 회전 전용 페이지(1.0p) 대신 직립 압축(0.55p)으로: **약 0.4–0.45p 회수**.
- TAB-1 폭 수정: 분량 영향 0 (겹침 해소 목적).
- 종합: 겹침을 모두 올바르게 풀었을 때의 기대 본문 ≈ 9.6p → TAB-2 직립 압축 적용 시 **≈ 9.1–9.2p** → 추가로 본문 산문 ~0.2p 압축(예: §2 관련연구 0.1p + §4.4 "Why graceful degradation" 문단 0.1p) 또는 FIG-3·FIG-4의 실그림 높이를 가정 하한(각 −0.05p)으로 잡으면 **8.8–9.0p 도달 → R6 게이트 통과 가능**. 대규모 본문 절제 불필요.

---

## 5. Appendix 구성 (R7)
- 구분 자체는 적절: Appendix A(셋업+전체 결과) / B(추가 분석) / C(방법 상세)가 §헤더로 명확히 분리되고, 본문→appendix 참조가 모두 해석됨.
- **"Appendix C 12p" 질문에 대한 의견: 과대 아님 — 착시임.** preprint에서 C가 인쇄 p36–47에 걸쳐 보이지만, p38–41은 **표류한 본문 그림(Fig 2/3/4, Fig B.5)과 Table A.6**, p42–44는 Appendix A/B의 표들이 차지. 순수 C 콘텐츠는 §C.1 산문+수식(~1.2p) + C.16 표(~0.3p) + Algorithm 1(~1p) + C.17 notation(~1p) = **약 3.5–4p**로 R7 취지(재현성 상세)에 합당. float 표류(MAJOR-8)를 고치면 외형도 정상화됨.
- 구조적 흠: A.5/A.6 빈 소절(MINOR-7), appendix float 번호 체계(MAJOR-5)가 REGISTRY의 A.1/B.1/C.1 명명과 어긋나 Phase 6 산출물과의 대조 작업을 방해 — 카운터 리셋이 R7 마감 전 필수.

---

## 6. 종합 판정
1. **5p 게이트 빌드: FAIL.** BLOCKER 4건(전부 float 폭/배치) 해소 전에는 D-012 분량 판정을 확정할 수 없음. 현 실측 9.2p는 하한치이며, 정상 배치 시 9.6p까지 증가 가능 → TAB-2 직립 압축이 게이트 통과의 핵심 (≈0.4p 회수).
2. **preprint 빌드: CONDITIONAL FAIL.** 구조·흐름·참조는 건전하나 내용 절단 4건(BLOCKER-A/B/C + C.16)과 (author?)×2, 그림 표류는 배포 전 필수 수정.
3. 수정 우선순위 (전부 .tex 단계, 산문 수술 불요): ① sidewaystable→table* 직립 압축(TAB-2) ② 광폭 표 6개에 \small/\scriptsize+p{}열+약어 ③ \citet→Ganin et al.~\cite ④ "Appendix~\ref" 중복 제거(일괄 치환) ⑤ appendix 카운터 리셋 + Algorithm의 하드코딩 "Table A.1"을 \ref로 ⑥ float 배치 옵션([!t], \FloatBarrier)으로 그림 표류 해소.
