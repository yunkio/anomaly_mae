---
phase: 7
agent: journal-format-researcher
directives: [T7]
last_modified: 2026-06-11
target_journal: Knowledge-Based Systems (Elsevier, ISSN 0950-7051)
---

# KBS (Knowledge-Based Systems) 투고 형식 요구사항 체크리스트

> 타깃 저널 확정: **Knowledge-Based Systems (Elsevier)**. 본 문서는 공식 Guide for Authors(GFA)
> 및 Elsevier 공식 LaTeX 문서를 근거로 한 항목별 요구사항 + 현 패키지(`paper/07_latex/`) 갭 판정.

## 0. 출처 및 검증 방법 (소스 신뢰도 고지)

| 출처 | URL | 비고 |
|---|---|---|
| KBS Guide for Authors (공식, 현행 canonical URL) | https://www.sciencedirect.com/journal/knowledge-based-systems/publish/guide-for-authors | 구 URL `elsevier.com/journals/knowledge-based-systems/0950-7051/guide-for-authors` 는 이 주소로 301 리다이렉트 확인 (2026-06-11) |
| KBS GFA 전문 (Wayback 스냅샷 2024-01-22) | http://web.archive.org/web/20240122174622/https://www.sciencedirect.com/journal/knowledge-based-systems/publish/guide-for-authors | **본 문서 인용문의 1차 텍스트 소스** |
| Elsevier LaTeX instructions (공식) | https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions | elsarticle 권장, 소스파일 제출 규정 |
| elsarticle 공식 문서 elsdoc.pdf v3.5 (2026-01-09) | http://mirrors.ctan.org/macros/latex/contrib/elsarticle/doc/elsdoc.pdf (패키지: https://ctan.org/pkg/elsarticle) | documentclass 옵션·.bst 정의 |
| 게재본 판형 실측 #1 | https://openreview.net/pdf/b77f9a495046cd509608ff0920bffb8b75f6fb60.pdf — Mei et al., "A feature reuse framework with texture-adaptive aggregation for reference-based super-resolution", **Knowledge-Based Systems 314 (2025) 113201** (VoR 미러) | 2단(two-column) 확인 |
| 게재본 판형 실측 #2 | http://weihuaxu.com/papers/2025/2025-KBS-Xu-Li.pdf — Xu & Li, "Enhancing information fusion and feature selection efficiency via the PROMETHEE method…", KBS 2025 (VoR 미러) | 2단(two-column) 확인 |

**검증 한계 고지**: 현행 ScienceDirect GFA 페이지는 자동 접근에 403(봇 차단)을 반환하여 라이브 전문을
직접 확보하지 못함. 대신 (a) Wayback 2024-01-22 전문 스냅샷, (b) 2026-06-11 시점 검색엔진 인덱스
스니펫(현행 페이지 기준)을 교차 검증 — highlights 85자, 키워드 최대 6개, "20 double line spaced
manuscript pages" 문구가 모두 일치함. 아래 인용문은 2024-01-22 스냅샷 원문이며, 투고 직전 브라우저로
현행 GFA 1회 재확인 권장(아래 각 표의 "투고 전 재확인" 표기 참조).

---

## 1. 제출 방식 (Your Paper Your Way)

| 항목 | 요구사항 | 출처 | 현 상태 판정 |
|---|---|---|---|
| YPYW 명시 여부 | GFA에 "Your Paper Your Way" 명칭 **없음** (전문 grep 0건) | KBS GFA | — |
| 참고문헌 형식 (1차 투고) | 자유: "There are no strict requirements on reference formatting at submission. References can be in any style or format as long as the style is consistent." 저널 스타일은 게재 확정 후 proof 단계에서 Elsevier가 적용 | KBS GFA §Reference formatting | **적합** (이미 elsarticle-num으로 저널 스타일 선적용 — 무해) |
| 원고 구조 | 단, 본문 구조 규정은 적용: numbered sections (1, 1.1, 1.1.1…), abstract는 섹션 번호 제외 | KBS GFA §Subdivision - numbered sections | **적합** (elsarticle 기본) |
| 개정(revision) 시 | "Please ensure you provide all relevant editable source files at submission **and every revision**." — 매 라운드 editable 소스(LaTeX) 필수 | KBS GFA §LaTeX | **적합** (LaTeX 소스 보유) |

## 2. LaTeX 템플릿 / 판형

| 항목 | 요구사항 | 출처 | 현 상태 판정 |
|---|---|---|---|
| 템플릿 | "For LaTeX submissions we encourage authors to use our LaTeX Template" — elsarticle 권장(강제 아님) | KBS GFA §LaTeX; Elsevier LaTeX instructions | **적합** (elsarticle.cls 사용) |
| 투고용 documentclass 옵션 | elsdoc: `preprint` = "default option which format the document for submission to Elsevier journals" (기본 로드 옵션: a4paper, 10pt, oneside, onecolumn, preprint) | elsdoc.pdf v3.5 §4 Usage | **적합** (`\documentclass[preprint,12pt]{elsarticle}` — 12pt는 리뷰 가독성용, 허용 범위) |
| 게재본 판형 (실측) | **2단(double column)** — 2025년 게재 VoR PDF 2편 실측: 페이지 595×794pt, 본문 블록 x0=38/307pt, 컬럼폭 ~251pt의 2단 구성 (KBS 314 (2025) 113201; Xu & Li 2025) | 게재본 실측 #1, #2 | **참고** — 분량 추정은 `5p,twocolumn` 기준이 정확 (`main_5p_measure.tex` 보유, 적합) |
| 최종 판형 옵션 매핑 | elsdoc: `5p` = "formats for model 5+ journals. This is always of two column style." (`3p`+`twocolumn`도 동일 계열) | elsdoc.pdf v3.5 §4 Usage | **적합** (5p 측정 파일 존재) |
| 소스 제출 형식 | 모든 제출 파일(그림·표·스타일·bst 포함)은 **하위 폴더 없이 단일 레벨**로 묶어 'LaTeX source files' 항목으로 업로드 + 컴파일된 PDF를 'Manuscript' 항목으로 첨부 | Elsevier LaTeX instructions | **수정 필요** — 현재 `sections/` 하위 폴더 구조 → 제출용 zip은 flat 구조로 재구성 필요 (`overleaf_package.zip` 점검) |

## 3. Reference 스타일

| 항목 | 요구사항 | 출처 | 현 상태 판정 |
|---|---|---|---|
| 인용 방식 | **Numbered**: "Indicate references by number(s) in square brackets in line with the text." 예: '..... as demonstrated [3,6]. Barnaby and Jones [8] obtained a different result ....' | KBS GFA §Reference style | **적합** |
| 목록 정렬 | "Number the references (numbers in square brackets) in the list **in the order in which they appear in the text**." (출현 순서, 알파벳순 아님) | KBS GFA §Reference style | **적합** — `elsarticle-num.bst` = "numbered scheme" (elsdoc §Bibliography: "Three bibliographic style files (*.bst) are provided — elsarticle-num.bst, elsarticle-num-names.bst and elsarticle-harv.bst") |
| 저널명 약어 | "Journal names should be abbreviated according to the List of Title Word Abbreviations (LTWA)." | KBS GFA §Journal abbreviations source | **확인 필요** — refs.bib 저널명 약어 일관성 점검 (단, 1차 투고는 자유 형식이므로 비차단) |
| DOI | "Use of the DOI is highly encouraged." | KBS GFA §Reference formatting | **확인 필요** — refs.bib DOI 필드 보강 권장 |

## 4. Highlights

| 항목 | 요구사항 (현행 공식 문구 원문) | 출처 | 현 상태 판정 |
|---|---|---|---|
| 필수 여부 | "Highlights are **optional yet highly encouraged** for this journal" | KBS GFA §Highlights | 선택이나 제출 권장 |
| 개수·글자수 | "Please use 'Highlights' in the file name and include **3 to 5 bullet points (maximum 85 characters, including spaces, per bullet point)**." — **85자** (125자 아님; 2026-06 검색 인덱스 스니펫으로 현행 일치 재확인) | KBS GFA §Highlights | **수정 필요** — 현 5개 모두 초과: 실측 109/120/120/121/125자 → 전부 ≤85자로 재작성 |
| 제출 형태 | "Highlights should be submitted in a **separate editable file** in the online submission system." | KBS GFA §Highlights | **수정 필요** — 현재 main.tex 내 inline `highlights` 환경 → 제출 시 별도 파일(예: `Highlights.docx`/`.tex`) 분리 |

## 5. Abstract / Keywords

| 항목 | 요구사항 | 출처 | 현 상태 판정 |
|---|---|---|---|
| Abstract 단어수 | **명시적 단어수 제한 없음**. "A concise and factual abstract is required. … it must be able to stand alone. … References should be avoided" (인용·비표준 약어 회피) | KBS GFA §Abstract | **적합** (실측 237 words, placeholder 포함 — 제한 없음이나 'concise' 권고상 현 수준 유지/압축 권장) |
| Keywords 개수 | "provide a **maximum of 6 keywords**, using **American spelling** and avoiding general and plural terms and multiple concepts" | KBS GFA §Keywords | **수정 필요** — 현재 **7개** → 6개 이하로 축소. American spelling 점검 병행 |

## 6. Graphical abstract

| 항목 | 요구사항 | 출처 | 현 상태 판정 |
|---|---|---|---|
| 필수 여부 | **선택(optional)**. 현행 GFA 본문에 전용 섹션 없음; 제출 체크리스트에만 "Graphical Abstracts / Highlights files (**where applicable**)" | KBS GFA §Submission checklist | **적합** (미제출 가능 — 제출 시 가산점 차원 검토만) |

## 7. 필수 선언 섹션

| 섹션 | 필수 여부·위치 | 출처 | 현 상태 판정 |
|---|---|---|---|
| CRediT authorship contribution statement | **필수**: "we **require** corresponding authors to provide co-author contributions to the manuscript using the relevant CRediT roles" (14개 역할 분류; 원고 내 기재) | KBS GFA §Author contributions | **수정 필요** — 원고에 부재 → 본문 말미(References 앞)에 섹션 추가 |
| Declaration of competing interest | **필수**: "Corresponding authors, on behalf of all the authors of a submission, **must disclose** any financial and personal relationships…" + 체크리스트 "A competing interests statement is provided, **even if the authors have no competing interests to declare**". Elsevier declaration tool로 .docx 생성 후 Attach Files 단계 업로드(다른 파일형식 변환 금지, 서명 불요) | KBS GFA §Declaration of competing interest, §Submission checklist | **수정 필요** — 원고 내 섹션 + 제출 시 tool 생성 .docx 둘 다 준비 |
| Data availability | **필수**: "we **require** you to state the availability of your data in your submission" (제출 과정에서 Data Statement 작성; 게재 시 논문과 함께 표시) | KBS GFA §Data statement | **수정 필요** — 원고 내 Data availability 섹션 추가 + 제출 폼 입력 준비 (코드 공개 계획 [URL] 연동) |
| Funding / Acknowledgements | Acknowledgements: 별도 섹션, **본문 끝 References 앞** (제목 페이지·각주 금지). Funding 표준 문구 형식 지정; 무펀딩 시 권장 문구: "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors." | KBS GFA §Acknowledgements, §Formatting of funding sources, §Role of the funding source | **수정 필요** — 원고에 부재 → 해당 여부 확인 후 섹션 추가 |
| Generative AI 사용 고지 | 집필 과정에 생성형 AI 사용 시 **필수**: 원고 말미 References 앞에 'Declaration of Generative AI and AI-assisted technologies in the writing process' 섹션 신설. 지정 문구: "During the preparation of this work the author(s) used [NAME TOOL / SERVICE] in order to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication." (가독성·언어 개선 용도만 허용; 문법·맞춤법 도구는 비대상; 고지할 것 없으면 문구 불요; AI를 저자로 기재 금지) | KBS GFA §Declaration of generative AI in scientific writing | **수정 필요** — 본 프로젝트는 AI 집필 지원 사용 → 섹션 추가 + 사용 범위가 정책(가독성/언어 개선) 내인지 저자 판단·기록 필요 |

## 8. 분량 제한

| 항목 | 요구사항 | 출처 | 현 상태 판정 |
|---|---|---|---|
| 연구 논문 | "Original high-quality research and review papers (**preferably no more than 20 double line spaced manuscript pages, including tables and figures**)" — 'preferably' = 권고(soft limit) | KBS GFA §Types of paper | **확인 필요** — preprint(double-spaced 유사) 기준 현 분량 측정 후 비교; 초과 시에도 hard reject 규정은 아니나 압축 검토 |
| Short communication | "no more than 10 double line spaced manuscript pages" | KBS GFA §Types of paper | 해당 없음 |

## 9. 기타 규정

| 항목 | 요구사항 | 출처 | 현 상태 판정 |
|---|---|---|---|
| 제목 페이지 | Title: 간결, 약어·수식 회피. 저자: given+family name, 소문자 위첨자 알파벳으로 소속 연결, 소속에 **국가명 포함 전체 주소 + (가능 시) 각 저자 e-mail**, corresponding author 명시, Present/Permanent address는 숫자 위첨자 각주 | KBS GFA §Essential title page information | **확인 필요** — main.tex frontmatter의 소속 주소/국가/email/corresponding 표기 점검 |
| ORCID | GFA에 **언급 없음** (전문 grep 0건). Editorial Manager 제출 단계에서 별도 요구 가능성 → **확인 불가(투고 시 확인)** | KBS GFA 전문 | 비차단 |
| 수식 | "Please submit math equations as **editable text and not as images**." 단순 수식은 본문 인라인 + solidus(/) 사용, 변수 이탤릭, e 거듭제곱은 exp 권장, 별도 표시 수식은 연속 번호 | KBS GFA §Math formulae | **적합** (LaTeX 네이티브) |
| 그림 | 벡터: EPS/PDF(폰트 임베드); 사진(halftone): TIFF/JPEG ≥300dpi; 선화: ≥1000dpi; 혼합: ≥500dpi. 폰트는 Arial/Courier/Times/Symbol 계열. 캡션은 그림과 분리 제출. **색각 이상자 접근성 보장** ("Ensure that color images are accessible to all, including those with impaired color vision") | KBS GFA §Electronic artwork | **확인 필요** — 그림 파이프라인(현 PDF 벡터?) 및 색상 팔레트 접근성 점검 |
| 표 | "submit tables as editable text and not as images" + "avoid using vertical rules and shading in table cells" | KBS GFA §Tables | **적합** (booktabs 스타일이면 통과; 세로줄 사용 여부만 확인) |
| 부록 | Appendix A, B…; 수식 Eq. (A.1), 표 Table A.1, 그림 Fig. A.1 별도 번호 체계 | KBS GFA §Appendices | **적합** (appendix counter reset 구현됨 — TEMPLATE_REQUIREMENTS.md r2) |
| 피어리뷰 방식 | Single anonymized (이중맹검 아님 → `doubleblind` 옵션 불요, 제목 페이지에 저자 노출 정상) | KBS GFA §Peer review | **적합** |
| 데이터 인용 | 데이터셋 인용 시 reference list에 `[dataset]` 마커 + 영구 식별자 권장 | KBS GFA §Data references | **확인 필요** — 벤치마크 데이터셋(SWaT/WADI/PSM/SMD 등) 인용 형식 검토 (권장사항) |

## 10. 갭 요약 — 수정 필요 항목 (우선순위순)

| # | 항목 | 현 상태 | 조치 |
|---|---|---|---|
| 1 | **Highlights 글자수** | 5개 모두 85자 초과 (109–125자) | 전부 ≤85자(공백 포함)로 재작성 + 별도 파일 분리 |
| 2 | **Keywords 개수** | 7개 | ≤6개로 축소 (후보 병합: 예. 'Contaminated benchmark' 통합 검토) + American spelling 확인 |
| 3 | **선언 섹션 일괄 부재** | CRediT / Competing interest / Data availability / (Funding·Ack) / Generative AI 모두 없음 | References 앞에 섹션 블록 추가 + competing interest는 Elsevier tool .docx 별도 생성 |
| 4 | **\journal{} 미설정** | `[JOURNAL NAME --- to be filled]` | `\journal{Knowledge-Based Systems}` |
| 5 | **제출 패키지 구조** | `sections/` 하위 폴더 | 제출 zip은 단일 폴더 레벨(flat)로 재구성 |
| 6 | 분량 | 미측정 (soft limit 20p double-spaced) | preprint 기준 측정·기록, 초과 시 압축 판단 |
| 7 | 제목 페이지/그림/bib 세부 | 미점검 | 주소·국가·email 표기, 그림 dpi·색 접근성, LTWA 약어·DOI 점검 |

**적합 확인 (변경 불요)**: elsarticle + `[preprint,12pt]` (투고), elsarticle-num.bst (numbered, 출현순),
abstract (제한 없음), graphical abstract (선택 — 미제출 가능), 1차 투고 참고문헌 자유 형식,
single anonymized review (저자 노출 정상), 게재본 판형 = **2단** (분량 추정은 5p 기준 사용).
