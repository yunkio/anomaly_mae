---
phase: 7 (re-entry)
agent: latex-engineer
directives: [D-015]
date: 2026-06-11
input: KBS_FORMAT_REQUIREMENTS.md (gap items #1–#5)
---

# KBS 정합화 작업 보고 (D-015)

LaTeX 패키지를 Knowledge-Based Systems(Elsevier) 투고 규정에 정합화. 본문 산문(§1–§5,
Appendix A–C) 무접촉 — 변경은 highlights/keywords/선언 섹션/구조/메타에 한정.
산문 신설분 전문은 `PROSE_DIFF_LOG.md` §7 수록(§7.3 미니 감사 대상).

## 작업 8건 결과

| # | 작업 | 결과 |
|---|---|---|
| 1 | Highlights ≤85자 재작성 + `highlights.txt` | **DONE** — 5/5 재작성, python `len()` 실측 **79/81/84/84/83자** (구: 109/120/120/121/125). 의미 보존 5축(설정 정식화/라벨 3경로/비대칭 T–S/오염 벤치마크/실험·강건성). 별도 plain-text 파일 `highlights.txt` 생성(동일 5 bullet, 전부 ≤85 검증) |
| 2 | Keywords 7→6 | **DONE** — "Contaminated benchmark" 제거(프로토콜 자기 지칭 — 색인 정보량 최소; 'Semi-supervised learning'이 포섭). 잔존 6개 American spelling 점검 통과 |
| 3 | 선언 섹션 5종 신설 | **DONE** — Conclusion 뒤·References 앞 순서: CRediT(2저자 템플릿+14역할 주석) → Competing interest(Elsevier 표준 문구+[TO BE CONFIRMED]) → Generative AI 고지(Elsevier 지정 양식+[TO BE CONFIRMED — 도구·범위]) → Data availability([public benchmarks; code at [URL]] placeholder, PH:TXT-002 동기화) → Funding/Acknowledgements([TO BE FILLED]+무펀딩 권장 문구 주석) |
| 4 | `\journal{Knowledge-Based Systems}` | **DONE** — main.tex / main_3p_measure.tex / main_5p_measure.tex 3종 전부 |
| 5 | flat 구조 재구성 | **DONE** — `sections/*.tex` 8종 루트로 이동(git rename `R` 기록 8건 확인), `\input` 경로 3종 main 전부 갱신. Overleaf·Editorial Manager(단일 레벨 규정) 양쪽 호환 |
| 6 | 재컴파일 3종 + 5p 재측정 | **PASS** — 3종 전부 `!` 오류 0, undefined ref 0, 렌더 "??" 0. **5p 본문 8.997p 유지**: §5 종점 "ceptance)." printed p.9(PDF p.10) 우측 컬럼 yMax 762.842847pt — D-014 기록과 bit-identical 좌표. 선언 섹션은 PDF p.11부터 렌더(측정 종점 이후, D-015 ⑥ 측정 제외 확인). 총 페이지: preprint 46→47, 5p 19→21 (선언 섹션 추가분 — 측정 제외 영역) |
| 7 | zip 재패키징 + 단독 컴파일 검증 | **PASS** — `overleaf_package.zip` flat 13파일(main.tex, sec1–5, appendix A–C, refs.bib, elsarticle-num.bst, elsarticle.cls, highlights.txt; measure 빌드·부산물 제외). 임시 폴더 추출 후 단독 `latexmk -pdf`: exit 0, 오류 0, "??" 0, 47p |
| 8 | PROSE_DIFF_LOG §7 | **DONE** — highlights 신구 5쌍(글자수 병기), keyword 변경 근거, 선언 섹션 영문 전문(미니 감사 대상 명시), 구조·재측정 기록 |

## 신 Highlights (글자수 실측)

| # | 글자수 | 내용 |
|---|---|---|
| H1 | 79 | We formalize the contaminated semi-supervised MTSAD setting with sparse labels. |
| H2 | 81 | Three label paths: anomaly-priority masking, loss bifurcation, gradient reversal. |
| H3 | 84 | CSMAD's asymmetric Teacher--Student masked autoencoder amplifies anomaly discrepancy. (en-dash 렌더 기준; 소스 문자 그대로도 85) |
| H4 | 84 | A contaminated benchmark adds test prefixes to training, exposing labeled anomalies. |
| H5 | 83 | Competitive on [N] datasets under five metrics; graceful decay with label sparsity. (PH:NUM-003 유지) |

## 잔여 항목 (제출 직전 — 코드/원고 외부)

- KBS_FORMAT_REQUIREMENTS.md §10-6/7 비차단 항목: refs.bib LTWA 약어·DOI 보강, 그림 색 접근성,
  제목 페이지 저자/소속/국가/email (저자 확정 후).
- Competing interest는 원고 내 섹션 외에 Elsevier declaration tool 생성 `.docx`를 제출 시 별도 첨부.
- 투고 직전 현행 GFA 브라우저 1회 재확인 권장(소스 문서의 검증 한계 고지 참조).
