---
phase: 7
agent: latex-engineer (r2 — pdf-qa fix round)
directives: [T7, R6, R7]
last_modified: 2026-06-11
inputs: PDF_QA_r1.md (전수 발견), PLACEHOLDER_REGISTRY.md (float 크기 가정)
outputs: 수정된 main.tex / main_3p_measure.tex / main_5p_measure.tex / sections/*.tex,
         재컴파일 PDF 3종, TEMPLATE_REQUIREMENTS.md v2, PROSE_DIFF_LOG.md §3b
prose_changes: "ZERO — 전 수정이 LaTeX/float/카운터 레벨. 유일한 렌더 텍스트 변화는
                깨진 \\citet 복구('(author?)'→'Ganin et al.')와 'Appendix Appendix' 중복
                제거 — PROSE_DIFF_LOG §3b 기록."
---

# FIX REPORT r2 — PDF QA r1 발견 전수 수정 + 재컴파일·재측정

## 0. 결과 요약

| 빌드 | 컴파일 | Overfull (r1→r2) | 페이지 (PDF) | 비고 |
|---|---|---|---|---|
| `main.pdf` (preprint,12pt — 배포) | latexmk 무오류 | 26 → **11** (전부 산문 행, 최대 16.5pt; >30pt 표/알고리즘 건 **0**) | 48 → 49 | 그림 표류 해소로 본문 그림이 본문 안으로 복귀 |
| `main_3p_measure.pdf` | latexmk 무오류 | — → **4** (최대 11.5pt) | 26 | 측정용 |
| `main_5p_measure.pdf` (게이트 판형) | latexmk 무오류 | 17 → **1** (1.9pt, output routine) | 18 → 21 | 겹침(overprint) **전부 해소**; 본문 실측 9.3p (아래 §3) |

**5p 게이트 판정: 본문 9.30p — ≤9.0p 기준 +0.3p 초과로 미달 (정직 보고; §3.3).**
**산문 변경: 없음** (불가피 산문 수정 0건; PROSE_DIFF_LOG §3b에 감사 기록).

---

## 1. 발견별 처리 (전수)

### BLOCKER (7건 — 전부 해소, 시각 검증 완료)

| ID | 내용 | 처리 | 검증 |
|---|---|---|---|
| BLOCKER-1 (5p) | Table 1이 우측 컬럼 본문과 overprint | `table`→`table*[t]` + `\footnotesize` + `adjustbox{max width=\linewidth}` | 5p 인쇄 p7 상단 전폭 배치, 마진 내, 겹침 없음 |
| BLOCKER-2 (5p) | sideways Table 2가 Table 3·본문 위에 overprint | **`sidewaystable`→직립 `table*[t]`** + `\scriptsize` + `\tabcolsep` 2pt + adjustbox (QA fallback ladder (a)) | 5p 인쇄 p8 상단 ~0.5p, 16열 전부 가시, 겹침 없음 — **~0.45p 회수** (회전 전용 1.0p 대비) |
| BLOCKER-3 (5p) | Table A.8↔A.9/A.10 상호 겹침 + 우측 절단 | swat_dual/full_metrics/per_entity 등 광폭 appendix 표 전부 `table*[tp]` + `\footnotesize` + adjustbox | 5p 인쇄 p15 (A.4–A.7) float 페이지, 절단·겹침 없음 |
| BLOCKER-4 (5p) | Algorithm 1 ↔ Table B.15/C.16 겹침 | `algorithm`→**`algorithm*[t]`** (algorithm2e 2단 전폭 float) + `\footnotesize` | 5p 인쇄 p19 페이지 상단 전폭, 겹침 없음 |
| BLOCKER-A (preprint) | Table 1 Test AR 열 절단 | BLOCKER-1과 동일 수단 (adjustbox가 468pt 폭에 자동 축소) | preprint 인쇄 p15 마진 내 완전 수용 |
| BLOCKER-B (preprint) | Table A.4(config) GRL head 행 절단 | p{} 셀 내 unbreakable 체인에 `\allowbreak` 삽입 (`LayerNorm→\allowbreak Linear…`) + `table*` + adjustbox | 5p 인쇄 p13: 행이 2줄로 정상 개행; preprint overfull 133pt→0 |
| BLOCKER-C (preprint) | Table A.9(swat_dual) excl22 열군 절단 | `table*[tp]` + `\footnotesize` + `\tabcolsep` 3pt + adjustbox | preprint 인쇄 p40: full/excl22 양 조건 11열 전부 가시 |

### MAJOR (전부 해소)

| ID | 처리 | 검증 |
|---|---|---|
| MAJOR-3 | `\citet{ganin2016dann}` ×2 → `Ganin et al.~\cite{ganin2016dann}` (sec3_method.tex §3.4, appendix_C.tex §C.1) | pdftotext 전수 스캔: "(author?)" 0건; "Ganin et al. [36]" 정상 렌더 (양 빌드) |
| MAJOR-4 | `Appendix~\ref{…}` 패턴 29곳 일괄 `\ref{…}`로 (elsarticle이 "Appendix A.x"를 자체 산출) | "Appendix Appendix" 0건 (3빌드 전수 스캔) |
| MAJOR-5 | main.tex `\appendix` 직후 카운터 블록: `\setcounter{table/figure/algocf}{0}` + `\@addtoreset{…}{section}` + `\thealgocf` 접두 + `\theHtable/\theHfigure` 앵커 충돌 방지 | 렌더 번호: **Table A.1–A.8, B.1–B.4, C.1–C.2, Figure B.1, Algorithm C.1** — REGISTRY 명명과 전면 일치; hyperref 중복 dest 경고 없음 |
| MAJOR-6 | TAB-2 직립 전환으로 회전-과폭 문제 자체 소멸 | preprint 인쇄 p23 직립 마진 내 |
| MAJOR-8 | ① 그림 `[t]`→`[tbp]` ② `placeins` + `\BodyFloatBarrier`(§1/§3/§4 말미; **1단 빌드 전용** — twocolumn에서는 no-op으로 5p 데드스페이스 방지) | preprint: FIG-2 인쇄 p10 (§3.2 p9 직후; r1은 30p 표류), FIG-3 p21 (§4.4 동일면), FIG-4 p22 (§4.5 인접). 3p: FIG-3 p12/FIG-4·결론 p13 인접 |
| MAJOR-9 | Table C.16(dimensionality) → `table*[tp]` + Derivation 열 `p{0.55\linewidth}` + adjustbox | 5p 인쇄 p19/preprint p46: SWaT 상수열 목록 개행 수용, 절단 없음 |
| MAJOR-10 | algorithm*[t] 전환 + `\footnotesize` (preprint "Float too large by 49pt" 해소) | preprint 인쇄 p47: 박스 마진 내, 페이지 번호 겹침 없음 |
| MAJOR-11 | 하드코딩 "Table~A.1" → `Table~\ref{tab:csmad_config}`; 추가로 같은 부류 스테일 4곳 정리 (sec4의 "Table A.4"→`\ref{tab:per_entity}`, "Table A.3"→`\ref{tab:baseline_hparams}`, "Table B.4"×2→`\ref{tab:extended_ablations}`) | 카운터 리셋 후에도 \ref로 항상 정합 |

### MINOR

| ID | 처리 |
|---|---|
| MINOR-5 | `\affiliation`의 빈 주소 필드(addressline/city/postcode/state/country) 제거 → ", , , , ," 해소. **잔존**: organization 뒤 쉼표 1개 (elsarticle 내부 포매팅; 실제 소속 기입 시 자연 해소) |
| MINOR-7 | **미수정** — A.5/A.6 빈 소절에 포인터 문장을 넣는 것은 산문 생성이라 금지 범위. orchestrator 결정 필요 |
| MINOR-12/13 | 부분 잔존 — §C 소절 헤더와 해당 float(C.1/C.2 표, Algorithm)가 [tp] float 페이지로 분리 (5p 인쇄 p17 우측 컬럼·p18 하단 여백). appendix 미관 사안, 본문 게이트 무관 |

### 추가 수정 (QA 로그 >30pt overfull 전수 대응 — 발견표 외)
- Table 3(ablation): `\scriptsize`+`\tabcolsep` 3pt+adjustbox로 **단일 컬럼 복귀** (REGISTRY 형태 0.20p; r1 overfull 68/111pt 해소).
- tab:budgets, tab:baseline_hparams, tab:per_entity, tab:split_shifts, tab:contaminated, tab:epoch_sensitivity, tab:compute, tab:extended_ablations, tab:notation → `table*[tp]` + `\footnotesize` + adjustbox (각각 r1에서 16–130pt overfull).
- §4.4 Design 문단의 `$\{1.0, 0.75, …\}$` 수식 리스트에 `\allowbreak` (preprint 88.6pt overfull 해소; 단어 변경 없음).

---

## 2. 잔존 경고 (전수)

| 빌드 | 잔존 | 상세 |
|---|---|---|
| main (preprint) | Overfull 11건 | 전부 산문 행 하이픈 한계, 2.6–16.5pt (QA r1 기준 "시각적 무해" 범주). 위치: sec1 ×2, sec3 ×3, sec4 ×2, appendix_A ×4 부근 |
| main_3p | Overfull 4건 | 1.9–11.5pt 산문 행 |
| main_5p | Overfull 1건 | 1.9pt (output routine — 헤더) |
| 3빌드 공통 | pdfTeX dest `page.1` 중복 1건 | Highlights 무번호 페이지와 인쇄 p1의 앵커 공유 — r1 이전부터 존재한 elsarticle/hyperref 무해 아티팩트 |
| 3빌드 공통 | 미해석 참조 `??` 0건, undefined citation 0건 | pdftotext 전수 스캔 |

---

## 3. 재측정 (5p 게이트 판형) — D-012/R6

### 3.1 본문 종점 (좌표 실측, pdftotext -bbox)
- §5 Conclusion 종료 "…(to be released upon acceptance)." = **인쇄 p10, 좌측 컬럼, 컬럼 높이 59.2% 지점** (yMax 488.2pt / 본문영역 84.8–766.8pt). References 헤더는 동일 컬럼 61.5% 지점에서 시작.
- p10 기여분 = 0.5(컬럼) × 0.592 ≈ **0.30p**.
- **본문 실측 = 9 + 0.30 = 9.30p** (Highlights 제외).
- References: 인쇄 p10 (좌컬럼 61.5%) – p12 초반. Appendix: 인쇄 p12 – p20 (총 인쇄 20p).

### 3.2 조치 이력 (측정 경로)
| 단계 | 본문 | 비고 |
|---|---|---|
| r1 (겹침 깨진 레이아웃) | 9.2p (하한; QA는 정상 배치 시 ~9.6p 경고) | 측정 신뢰도 제한 |
| r2 ① BLOCKER 전부 해소 + TAB-2 직립 압축 | 9.5p | TAB-1/TAB-3 table* 승격의 전폭 밴드 비용 포함 |
| r2 ② FIG-2/3/4 높이를 REGISTRY 가정(하한)으로: 5.0/4.0/3.5cm | 9.5p (변화 미미) | placeholder 박스는 내부 설명 텍스트가 높이를 지배 — min-height 축소 실효 제한. 실측 FIG-3 0.22p/FIG-4 0.25p로 **이미 가정(0.33/0.30p) 이하** |
| r2 ③ TAB-3 단일 컬럼 복귀 + `\BodyFloatBarrier` twocolumn no-op화 (§4말 barrier가 만들던 ~0.3p 데드스페이스 제거) | **9.30p** | 최종 |

### 3.3 게이트 판정 — **FAIL (+0.30p)**, 정직 보고
- 기준: ≤9.0p & ≥8.5p. 실측 9.30p.
- 지시된 fallback(FIG-3/FIG-4 REGISTRY 하한 조정)을 적용했으나 두 그림은 이미 가정 이하 크기로 렌더되고 있어 회수 여력이 없었음 — **초과의 원인은 그림이 아니라 산문+표 총량**.
- 남은 회수 수단 (전부 orchestrator 결정 필요):
  1. **산문 압축 ~0.3p** — QA r1 §4.4가 지목한 후보: §2 관련연구 ~0.1p + §4.4 "Why graceful degradation" 문단 ~0.1p (+α). 본 라운드에서는 산문 수술 금지라 미실행.
  2. 실그림 투입 시 placeholder 설명 텍스트 인플레이션 소멸 → ~0.05–0.1p 자연 회수 기대 (기대 본문 ~9.2p, 여전히 +0.2p).
  3. (참고) TAB-1을 셀 축약("19.05 / 3.68†" 각주화 등)으로 단일 컬럼화하면 ~0.1p 추가 회수 가능하나 표 내용 표기 변경이라 보류.

---

## 4. 파일 변경 목록
- `main.tex` / `main_3p_measure.tex` / `main_5p_measure.tex`: 패키지(rotating 제거, adjustbox·placeins 추가), `\BodyFloatBarrier` 조건 매크로, affiliation 빈 필드 제거, `\appendix` 카운터 리셋 블록. (measure 2종은 main.tex에서 documentclass만 치환해 재생성 — \input 구조 공유 확인)
- `sections/sec1_intro.tex`: Appendix~\ref ×2, FIG-1 [tbp], \BodyFloatBarrier
- `sections/sec2_related.tex`: 변경 없음 (Appendix~\ref 패턴 없음)
- `sections/sec3_method.tex`: \citet 복구, Appendix~\ref ×3, FIG-2 [tbp]+5.0cm, \BodyFloatBarrier
- `sections/sec4_experiments.tex`: TAB-1 table*화, TAB-2 sidewaystable→table*, TAB-3 단일컬럼 \scriptsize, FIG-3/4 [tbp]+높이 조정, Appendix~\ref ×20, 스테일 표번호→\ref ×4, 수식 리스트 \allowbreak, \BodyFloatBarrier
- `sections/appendix_A.tex`: 표 7개 table*[tp]+adjustbox, GRL행 \allowbreak, Appendix~\ref ×1
- `sections/appendix_B.tex`: 표 4개 table*[tp]+adjustbox, Appendix~\ref ×1
- `sections/appendix_C.tex`: \citet 복구, dimensionality p{}열, notation table*, algorithm*[t]+\footnotesize, Table~A.1→\ref
- `TEMPLATE_REQUIREMENTS.md` v2 (패키지 목록·float 규약·경고표 갱신), `PROSE_DIFF_LOG.md` §3b (r2 무산문 감사 기록)
