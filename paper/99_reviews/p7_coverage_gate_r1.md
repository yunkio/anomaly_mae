---
phase: 7
agent: coverage-auditor
directives: [M10]
last_modified: 2026-06-11
inputs:
  - 07_latex/main.pdf (배포, 46p letter) / main_5p_measure.pdf (게이트 판형, 19p A4) / main_3p_measure.pdf (25p)
  - 07_latex/overleaf_package.zip (12 files) + main.log / main_5p_measure.log
  - 07_latex/TEMPLATE_REQUIREMENTS.md v2, PROSE_DIFF_LOG.md (§5.6 측정 무결성 포함)
  - 07_latex/pdf_qa/PDF_QA_r1.md, FIX_REPORT_r2.md
  - 99_reviews/p7_prose_miniaudit_r1.md; 05_manuscript/PLACEHOLDER_REGISTRY.md (v3-r1)
method: |
  전 판정의 독립 재현 — (a) 두 PDF 시각 검수: 5p 전 19페이지 + preprint 44/46페이지(1–23, 31–46)
  Read pages로 직접 열람, (b) 빌드 로그 경고 전수 추출 vs 기록 대조, (c) pdftotext 무결성 grep
  (??/(author?)/Appendix 중복), (d) pdftotext -bbox 본문 종점 좌표 실측, (e) zip 추출→파일별 cmp
  →/tmp에서 latexmk 단독 재컴파일 1회, (f) placeholder 7건 REGISTRY 캡션/스펙 대조.
verdict: "GATE PASS — 조건 5/5 충족. BLOCKER 0 / MAJOR 0 / MINOR 2 + 핸드오프 경고 1."
---

# P7 Coverage Gate 감사 r1 — 게이트 5조건 + Directive 4종 판정

## 0. 판정 요약

| # | 조건 | 판정 | 핵심 근거 |
|---|---|---|---|
| ① | 컴파일 무경고 (불가피 경고 사유 기록) | **PASS** (MINOR 1) | 오류 0·undefined 0; overfull/underfull/dest 중복 모두 기록 일치; `h`→`ht` 경고만 미기록 (무해) |
| ② | pdf-qa 시각 검수 통과 | **PASS** | r1 BLOCKER 7건 전 위치 시각 재검증 — 겹침·절단·깨짐 잔존 0 (5p 전 19p + preprint 44p 직접 열람) |
| ③ | 페이지 예산 (본문 8.5–9.0p, D-012 5p 판형) | **PASS** (경계) | 독립 실측 본문 **8.997p** — §5 종점 = printed p9 우측 컬럼 yMax 762.8/766.8pt (99.4%) |
| ④ | 산문 변경분 미니 감사 통과 | **PASS** | p7_prose_miniaudit_r1 verdict PASS (B0/M0/MINOR7); M-1·M-2 후속 처리 기록 확인 |
| ⑤ | zip 단독 컴파일 성공 | **PASS** | /tmp 추출→latexmk exit 0, 46p, 내용 동일 (타임스탬프만 상이); 내용물 마스터 요구와 1:1 일치 |

---

## 1. 조건 ① — 컴파일 경고 전수 대조

빌드 로그 직접 추출 (main.log / main_5p_measure.log, 2026-06-11 15:57–58 빌드):

| 잔존 경고 | 실측 | 기록 여부 | 판정 |
|---|---|---|---|
| Overfull hbox (preprint) | **10건**, 전부 산문 행, 최대 16.55pt (appendix_B 48–52행) | FIX_REPORT §2 + PROSE_DIFF_LOG §5.7 "10건 ≤16.5pt" + TEMPLATE_REQ §8 — 일치 | 기록됨 |
| Overfull hbox (5p) | **1건**, 1.90pt, output routine | FIX_REPORT §2 / §5.7 — 일치 | 기록됨 |
| pdfTeX dest `page.1` 중복 | 양 빌드 각 1건 | TEMPLATE_REQ §8 + FIX_REPORT §2 (elsarticle/hyperref 무해 아티팩트) | 기록됨 |
| Underfull hbox/vbox | placeholder 골격 표 다수 | TEMPLATE_REQ §8 포괄 기록 ("draft 허용") | 기록됨 |
| **`h' float specifier changed to `ht'** | preprint ×2 / 5p ×1 — 출처: `appendix_A.tex:372 \begin{table}[h]`, `appendix_B.tex:110 \begin{figure}[h]` | **미기록** (TEMPLATE_REQ §8·FIX_REPORT §2 모두 부재) | **MINOR-G1** |
| 오류 / Citation undefined / Reference undefined | 0 / 0 / 0 (최종 pass) | — | 클린 |

- **MINOR-G1**: `h`→`ht`는 LaTeX이 자동 보정하는 정보성 경고로 출력 영향 0 — 단, "잔존 경고는 사유 기록" 규정상 기록 누락. 처방: TEMPLATE_REQ §8에 1행 추가 또는 두 소스의 `[h]`를 `[ht]`로 (게이트 비차단).
- pdftotext 무결성 grep 재확인: `??` 0 / `(author?)` 0 / `Appendix Appendix` 0 — **양 빌드 모두 0** (보고치 재현).

## 2. 조건 ② — 최종 PDF 시각 spot (직접 열람 기록)

### 2.1 main_5p_measure.pdf — 전 19페이지 열람
- **r1 BLOCKER-1 (Table 1 ↔ 본문 overprint)**: printed p6 — TAB-1이 단일 컬럼(D-013 §5.5 형태)으로 좌상단 배치, dagger 셀 "19.05/3.68†" 컬럼 폭 내 완전 수용, 우측 컬럼 본문과 겹침 없음. **해소.**
- **r1 BLOCKER-2 (sideways Table 2 overprint)**: printed p8 — 직립 `table*` 전폭 상단, 7패밀리×2지표 = 14 데이터 열 + 27행 + 하단 protocol-effect 블록(3행) 전부 가시, Table 3(좌하단 단일컬럼)·본문과 겹침 없음. **해소.**
- **r1 BLOCKER-3 (A.8↔A.9/A.10 상호 겹침)**: printed p15 — Table A.4–A.7 전폭 float 페이지, 절단·겹침 없음; A.6(swat dual) full/excl22 양 조건 열군 가시. **해소.**
- **r1 BLOCKER-4 (Algorithm ↔ 표 겹침)**: printed p18 — Algorithm C.1 전폭 상단, Table C.2와 분리, 겹침 없음. **해소.**
- 본문 p1–p9: 2단 겹침 0. frontmatter(printed p1) 정상 — 저자/소속/저널 placeholder 렌더, 소속 줄 잔여 쉼표 1개(기록된 MINOR-5 잔존). Highlights 별면(PDF p1, 무번호) 정상.
- Appendix 번호 체계: **Table A.1–A.8 / B.1–B.4 / C.1–C.2, Figure B.1, Algorithm C.1** — REGISTRY 명명과 일치 (r1 MAJOR-5 해소 확인).
- 잔존 미관(기록됨, 비차단): A.5/A.6/B.4 헤더-only 소절(MINOR-7 — 산문 생성 금지로 보류 결정 기록), §C.3/C.4 헤더와 float 분리(MINOR-12).

### 2.2 main.pdf (배포 46p) — 44페이지 열람 (1–23, 31–46; 비열람 24–30은 References 연속부, p30–31 표본 정상)
- **r1 BLOCKER-A (Table 1 절단)**: printed p14 — 마진 내 완전 수용, dagger 표기. **해소.**
- **r1 MAJOR-6 (Table 2 회전-과폭)**: printed p18 — 직립 전폭, MSL avg 열군까지 전부 가시. **해소.**
- **r1 BLOCKER-B (Table A.4→현 A.1 GRL행 절단)**: printed p32 — "LayerNorm→Linear(512→256)→GELU→Dropout(0.1)→Linear(256→1)" 2줄 개행 완전 표시. **해소.**
- **r1 BLOCKER-C (swat dual 표 절단)**: printed p37 (현 Table A.6) — full/excl22 10열 전부 마진 내. **해소.**
- **r1 MAJOR-9 (C.16 Derivation 절단)**: printed p43 (현 Table C.1) — "{P202, P401, P404, P502, P601, P603}" 개행 수용. **해소.**
- **r1 MAJOR-10 (Algorithm 푸터 침범)**: printed p44 — 박스 마진 내, 페이지 번호와 분리; 입력부 "configuration Table A.1"이 \ref로 정합 (MAJOR-11 해소). **해소.**
- **r1 MAJOR-8 (그림 표류)**: FIG-1 printed p3 (§1 인접), FIG-2 p8 (§3.2 직전), FIG-3 p21 (§4.4 동일면), FIG-4 p22 (§4.5 동일면) — 표류 0. **해소.**
- **r1 MAJOR-3 ((author?))**: printed p10 "sigmoid schedule of Ganin et al. [36]" 정상; grep 0. **해소.**
- 무작위 페이지 (printed 5–7, 11–13, 15–17, 20, 30–31, 33–36, 39–42, 45): 겹침·절단·폰트 깨짐 0. 인용 [1]–[48]·식 (1)–(6)/(C.1)–(C.5) 정상.

## 3. 조건 ③ — 5p 분량 직접 재현 (PROSE_DIFF_LOG §5.6 판정 검증)

- 페이지 구조 실측: PDF p1 = Highlights 별면(무번호, **미산입**) → printed p1(타이틀/초록)–p9 = 본문, References는 printed p10 1행부터 (pdftotext per-page 확인).
- 본문 종점 좌표 (pdftotext -bbox, printed p9): 좌측 컬럼 마지막 행 "…fails to replicate the Teacher." yMax 762.84pt; §5 종점 "ceptance)." **우측 컬럼 yMin 753.95 / yMax 762.84pt** — 본문영역 84.8–766.8pt의 **99.4%**.
- **본문 = 8 + 0.5 + 0.5×0.994 = 8.997p** → **8.5 ≤ 8.997 ≤ 9.0 PASS.** PROSE_DIFF_LOG §5.7 보고치 정확 재현.
- **MINOR-G2 (기록 정밀성)**: §5.6은 종점을 "우측 컬럼 ~97%"로, §5.7은 "99.4%"로 기재 — §5.7(좌표 병기)이 정확, 판정엔 무영향.
- **핸드오프 경고 (비차단)**: 상한 여유 ≈0.003p — 마지막 행이 p9 최종 행. Phase 8에서 실그림 투입·[X.XX]→실수치 치환 시 단 1행 증가로도 9p 초과 가능. **Phase 8 게이트에서 분량 재측정 필수** (FIX_REPORT §3.3의 placeholder 텍스트 인플레이션 소멸 ~0.05–0.1p 자연 회수 기대가 완충이나 보장 아님). PROSE_DIFF_LOG §5.6의 SMD 셀 "(\S A.3)" 하드코딩 재환원도 Phase 8 핸드오프 등재 확인.

## 4. 조건 ④ — 산문 미니 감사

- `p7_prose_miniaudit_r1.md` verdict: **PASS — BLOCKER 0 / MAJOR 0 / MINOR 7** (ai-phrasing 3, plagiarism 0, method-truth 4); 무손상 검증(\cite 48 키 집합 diff 0, PH 마커 31 NUM/4 TXT/5 FIG/11 TAB/1 ALG) PASS.
- 후속 처리 확인: **M-1**(TAB-1 Source 열 제거 미기록) → PROSE_DIFF_LOG §5.5 보완 기재로 해소; **M-2**(\S A.3 하드코딩) → §5.6에 측정 무결성 사유로 의도적 보류 + Phase 8 노트 등재 — 권고 처리 경로 모두 문서화됨. **조건 ④ PASS.**

## 5. 조건 ⑤ — zip 단독 컴파일 + 내용물 대조

- 내용물 (12 files): `main.tex`, `refs.bib`, `elsarticle-num.bst`, `elsarticle.cls`, `sections/{sec1_intro, sec2_related, sec3_method, sec4_experiments, sec5_conclusion, appendix_A, appendix_B, appendix_C}.tex` — 마스터 요구(루트 main.tex / sections / refs.bib / .bst / .cls 사본)와 1:1; build 부산물(.aux/.log/.pdf)·내부 문서(TEMPLATE_REQ, PROSE_DIFF_LOG)·measure 빌드 .tex **포함 0** (요구대로 제외).
- 파일별 `cmp`: 12/12 **작업본과 byte-identical**.
- 독립 재컴파일: /tmp/p7_zip_audit에서 `latexmk -pdf` 1회 — **exit 0, 46페이지**, `??` 0 / `(author?)` 0 / 최종 pass undefined 0; 산출 PDF는 배포본과 내용 동일 (차이 = 임베디드 타임스탬프 영역만). **PASS.**
- (관찰, 비차단) TEMPLATE_REQUIREMENTS §7 말미 "Submission zip must bundle: adjustbox + collectbox, placeins, algorithm2e"는 과기재 — 표준 TeX 배포 패키지로 zip 동봉 불요함을 본 단독 컴파일이 입증. 문서 문구 정정 권장.

## 6. Directive 근거 (4종)

- **T7**: elsarticle(num) 템플릿 준수 변환 — TEMPLATE_REQUIREMENTS v2 체크리스트(frontmatter/float/bib/appendix 규약) + placeholder 17종 LaTeX 실체화 + PDF 확인 루프(QA r1 → FIX r2 → 본 감사 시각 재검수)로 3빌드 무오류·겹침 0 도달.
- **R3 (P7분)**: FIG-1..4·FIG-B1 placeholder 박스, TAB 골격 11종, ALG-C1이 PLACEHOLDER_REGISTRY의 **완성형 캡션·크기 가정대로 렌더** — spot 7건(FIG-1/2/3/4, TAB-1/2, ALG-C1) 캡션 문안 대조 일치, 실측 크기 전부 가정 이하, 번호 체계 A.x/B.x/C.x REGISTRY 정합.
- **R6**: 본문 분량 게이트 — D-013 압축(−219 words) 후 5p 판형 본문 8.997p로 8.5–9.0p 창 충족; 본 감사가 bbox 좌표로 독립 재현 (상한 여유 0.003p는 Phase 8 재측정 조건부).
- **R7**: Appendix 3그룹 구성 유지 — A(셋업+전체 결과: A.1–A.6 / Table A.1–A.8) / B(추가 분석: B.1–B.5 / Table B.1–B.4, Fig B.1) / C(방법 상세: C.1–C.4 / Table C.1–C.2, Algorithm C.1); 본문→appendix \ref 전부 해석, 순수 C 분량 ~4p로 재현성 취지 합당 (QA r1 §5 판단 유지).

## 7. 종합

**게이트 PASS (5/5).** Phase 7 산출물은 다음 잔여 항목과 함께 Phase 8로 인계 가능:
1. (MINOR-G1) `h`→`ht` 경고 기록 누락 — TEMPLATE_REQ §8 1행 추가 또는 `[h]`→`[ht]` 수정.
2. (MINOR-G2) PROSE_DIFF_LOG §5.6 "~97%" ↔ §5.7 "99.4%" 표기 불일치 — §5.7이 정본.
3. (핸드오프 경고) 분량 상한 여유 0.003p — Phase 8 실데이터 투입 후 재측정 필수; SMD 셀 "(\S A.3)" 하드코딩 갱신 의무 동반.
4. (관찰) TEMPLATE_REQ §7 zip 동봉 문구 과기재 정정 권장.
