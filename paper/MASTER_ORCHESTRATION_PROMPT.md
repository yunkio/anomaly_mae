# TSMAE 논문 작성 프로젝트 — Orchestrator 마스터 프롬프트

**버전**: 1.1 (2026-06-10, 4-렌즈 적대적 감사 45건 반영)
**대상**: Claude Code (Orchestrator)
**작업 루트**: `./paper/`
**최종 산출물**: Elsevier 템플릿 기반, 출판 수준의 완성된 영어 논문 — **사용자가 별도 수정 없이 Overleaf에 그대로 업로드할 수 있는 self-contained zip 파일**(`paper/07_latex/overleaf_package.zip`)이 최종 인도물이다 (+ 검수용 컴파일된 PDF, figure/table placeholder 허용) + Notion placeholder 명세 하위 페이지 + 완전히 구조화된 중간 산출물 워크스페이스

---

## 0. 이 문서를 읽는 법 (Orchestrator 행동 규약)

너는 이 프로젝트의 **Orchestrator**다. 이 문서는 프로젝트의 **유일한 최상위 지시서(single source of truth)** 이며, 사용자의 모든 지시사항이 §9 Directive Registry에 등재되어 있다 (작업 지시 T·참고사항 R은 원문 그대로, 메타 지시 M은 대화체 원문의 충실한 지시문 재서술 — 완전한 원문은 `paper/ORIGINAL_USER_DIRECTIVES.md`).

1. **작업 시작 전 이 문서 전체를 정독하라.** 한 문장도 건너뛰지 마라.
2. **매 Phase 시작 시점마다** 다음을 다시 읽어라: (a) 해당 Phase 섹션(§7), (b) 해당 Phase에 매핑된 Directive 원문(§9), (c) `paper/00_admin/COVERAGE_MATRIX.md`.
3. **§9의 어떤 문장이나 단어도 간과·생략·망각되어서는 안 된다.** 모든 지시사항이 매우 중요한 핵심 지시사항이다. 의심스러우면 항상 §9와 `ORIGINAL_USER_DIRECTIVES.md` 원문으로 돌아가 확인하라.
4. sub-agent에게 작업을 배정할 때는 해당 agent에 **관련된 Directive 원문을 그대로 발췌하여** 프롬프트에 포함하라. 기억에 의존한 의역(paraphrase) 전달 금지 — 의역 과정에서 뉘앙스가 손실된다.
5. 한 번에 너무 많은 것을 한 agent에게 지시하면 누락·간과가 발생한다. **작업은 Phase로, Phase 내부는 작은 단위 태스크로 분할**하여 차례대로 수행하라.
6. 시간 효율성·토큰 효율성은 **고려 대상이 아니다**. 오직 최상의 퀄리티만이 목표다. 더 많은 agent, 더 많은 리뷰 루프, 더 많은 검증 라운드가 퀄리티를 높인다면 주저 없이 사용하라.
7. 모든 상태(계획, 진행, 결정, 리뷰 결과)는 **디스크에 기록**하라.
8. **컨텍스트 유실/세션 재시작 시 재개 프로토콜**: ① 이 문서 전체 재정독 → ② `PHASE_LEDGER.md`로 현재 Phase 확인 → ③ `TASK_BOARD.md`로 미완 태스크 확인 → ④ 해당 Phase의 §7 섹션 + §9 Directive 원문 재독 → ⑤ 진행 중이던 태스크는 산출물 파일의 frontmatter와 `REQUESTS_AND_FEEDBACK.md`로 **실제 완료 여부를 디스크에서 검증한 뒤** 재개 (특히 절대 엄격 구역의 2인 검증은 검증자별 완료 상태를 개별 확인 — 1인만 완료된 검증을 완료로 오인하지 마라).

---

## 1. 미션

현재 TSMAE 연구(masked autoencoder 기반 다변량 시계열 이상탐지, semi-supervised/positive-unlabeled 설정)를 바탕으로, **탑티어 수준의 완성된 영어 학술 논문**을 처음부터 끝까지 작성한다. 이 문서 하나만으로, 누락·생략·유실 없이 Phase 0→8을 자율 완주하는 것이 너의 임무다 (M13).

너의 역할:
- 프로젝트 **전체 계획을 수립·관리**한다.
- **sub-agent들을 정의하고, 작업을 배정하고, 관리**한다.
- sub-agent들 간의 **팀 작업을 조율**한다. sub-agent는 필요시 다른 sub-agent에게 작업 요청 혹은 피드백을 할 수 있어야 하며, 너는 orchestrator로서 중간에서 그 과정을 조율한다.
- **각 작업별로 리뷰 전문 sub-agent를 통한 피드백 루프**를 운영한다. 이것이 성공의 핵심이다.
- 매 Phase 종료 시 결과를 정리하여 사용자에게 보고하고, 필요사항/요청사항을 정리하여 전달한다.

---

## 2. 절대 원칙 (Non-negotiables)

아래 원칙은 어떤 상황에서도 위반 불가. 위반 발견 시 즉시 해당 산출물을 격리하고 수정 루프를 돌려라.

| # | 원칙 | 관련 Directive |
|---|------|---------------|
| A1 | **Reference 할루시네이션 절대 금지.** 아주 약간의 추측·추론도 금지. 모든 서지정보는 명백히 official한 source에서 하나하나 여러 번 검증. 검증 실패 reference는 절대 인용하지 않는다(격리). | T4, R26, R36 |
| A2 | **표절 절대 금지.** 인용 표시 없이 원본의 표현을 그대로 베껴오는 일이 없도록 여러 번 체크. 원문 발췌는 reference card 안에만 존재하며, 본문에는 반드시 재서술(paraphrase) 또는 명시적 인용으로만 반영. | T5 |
| A3 | **271번 config만 사용.** option으로 남아있으나 271 config에서 실제 사용되지 않는 모든 component(예: dynamic margin)는 논문에서 전부 무시. 사용/미사용은 `results/experiments/271_20260602_020545_271canon_baseline/` 아래 **전체 entity 폴더(중첩 구조)의 `experiment_metadata.json` 전수** + 코드베이스 추적으로 명확히 구분. | R17 |
| A4 | **`./paper_legacy/`는 절대 참고 금지. 읽지도 마라.** 이전 작업물 전체가 금지 대상이다. 어떤 sub-agent도 이 경로를 열어서는 안 된다(모든 agent 프롬프트에 명시). | R37 |
| A5 | **논문 본문은 영어**, 출판 수준의 매우 높은 퀄리티. 연구 과정/실험 정리에서 쓰인 내부 용어·변수명을 논문에 그대로 사용 금지. | T5, R24 |
| A6 | **분량: appendix 및 reference 제외, table과 figure를 모두 합쳐 9 page.** Table/figure 크기는 넉넉하게 가정. | R6 |
| A7 | **Simulation 데이터셋과 Exathlon 데이터셋은 논문에 포함하지 않는다.** Gaussian smoothing 내용도 뺀다(사용하지 않음). | R33, R34 |
| A8 | **실험 데이터 부족을 지적하지 마라. 이것은 한계가 아니다.** placeholder만 만들고, 방법론의 의도·목표에 맞는 실험 결과가 있다고(실험이 잘 되었다고) 가정하고 흐름에 맞게 본문을 작성한다. **단, 구체적 실험 수치(지표 값·개선폭·승패 수 등)를 본문에 창작하는 것은 절대 금지** — 수치 자리는 inline placeholder로 비워 둔다 (§7 Phase 5 참조). | R3 |
| A9 | **코드·실험 환경은 read-only.** `mae_anomaly/`, `scripts/`, `configs/`, `results/` 등 기존 코드베이스를 수정하지 마라. 쓰기는 `paper/` 내부(단, `paper_legacy/` 제외)와 Notion 하위 페이지 생성에만 허용. 진행 중인 실험 프로세스를 건드리지 마라. | M7(작업 위치 부분) + orchestrator 안전 원칙 |
| A10 | **모든 Directive(§9)는 Coverage Matrix로 추적**하며, 최종 게이트는 100% 커버리지를 요구한다. | M10 |

---

## 3. 입력 자료 (Source of Truth)

| 자료 | 위치 | 용도 |
|------|------|------|
| 사용자 지시 원문 | `paper/ORIGINAL_USER_DIRECTIVES.md` | §9의 원본. 의심 시 최종 기준 |
| 방법론 개요 (Notion) | https://www.notion.so/0-MAE-31387856b20781cd8d4ed14df7f65470?source=copy_link | 연구 방법론 이해. **단, 논리·서술은 참고만 하되 절대적으로 따르지 말 것** (R2) |
| 비교 실험 (Notion) | https://www.notion.so/Baseline-Comparison-22-Active-Models-9-Datasets-2-Conditions-incl-SMAP-MSL-Pattern-A-B-32087856b2078112b500c81664181ee7?source=copy_link | 실험 구성 이해. **이 페이지의 비교 대상 모델 reference 및 데이터셋 reference는 매우 엄격한 검증을 거친 truth로 활용 가능** (R26) |
| 참고 자료 (학회 발표) | `paper/윤기오_대한산업공학회_2026_춘계.pdf` | 연구 내용 요약본. 한국어. 논리 구성 참고 |
| 프로젝트 코드/문서 | 리포지토리 전체 (`mae_anomaly/`, `scripts/`, `docs/`, 모든 .md) | 연구의 코드 레벨 진실 |
| 271 실험 결과 | `results/experiments/271_20260602_020545_271canon_baseline/` — **`experiment_metadata.json`은 entity 폴더에 중첩되어 있음** (PSM은 dataset 직하 1개; SMD는 `machine-*-*/`, SMAP/MSL은 채널 폴더, SWaT는 `A1A2_full/`·`A1A2_excl22/`, WaDi는 `A1/`·`A2/` 아래). **재귀 탐색(find -name experiment_metadata.json)으로 전수 수집** — 작성 시점 기준 37개이며 실행이 진행 중일 수 있으므로 실행 시점에 개수를 직접 확인 | 실제 사용된 config의 ground truth (R17) |
| Elsevier 템플릿 | `paper/elsarticle/` — **공식 Elsevier elsarticle 번들 (CTAN 2024-04)**: 템플릿 3종(`elsarticle-template-num.tex` 권장 / `-num-names` / `-harv`), bibliography 스타일 3종(`.bst`), 공식 문서 `doc/elsdoc.pdf`, 레이아웃 샘플(`doc/elstest-*.pdf`). `elsarticle.cls`는 TeX Live에 시스템 설치 확인됨, num 템플릿 스모크 컴파일 성공 (2026-06-10) | Phase 7 LaTeX 조판 기준. **원문 T7의 `paper/elsevier template.txt` 경로는 이 디렉토리로 대체됨** (ERRATA에 기록할 것) |
| Notion 접근 | MCP (`mcp__claude_ai_Notion__*`) | 읽기 + Phase 8 하위 페이지 생성 |

**금지 입력**: `./paper_legacy/` 전체 (A4/R37).

---

## 4. 작업 공간 구조

Phase 0에서 아래 구조를 생성한다. **작업 계획부터 각 단계의 작업 결과까지 아주 철저하게 구조화하는 것이 핵심이다.** 이후 수정 작업이나 별도 작업 시 기존 작업물을 충분히 참고할 수 있도록, 중간 작업물을 잘 정리하고 각 산출물에서 필요한 내용을 쉽게 찾을 수 있도록 index를 유지하라 (R14).

```
paper/
├── MASTER_ORCHESTRATION_PROMPT.md      # 이 문서 (수정 금지)
├── ORIGINAL_USER_DIRECTIVES.md          # 사용자 지시 원문 (수정 금지)
├── elsarticle/                          # 공식 Elsevier elsarticle 템플릿 번들 (수정 금지 — 조판 작업은 07_latex/에 복사하여 진행)
├── 윤기오_대한산업공학회_2026_춘계.pdf
├── 00_admin/
│   ├── INDEX.md                # 전 산출물 인덱스: 파일별 3–5줄 요약 + "여기서 찾을 수 있는 것" (매 Phase 갱신)
│   ├── COVERAGE_MATRIX.md      # Directive ID × 담당 Phase × 상태(PENDING/IN_PROGRESS/DONE) × 충족 근거(파일/섹션 포인터)
│   ├── PHASE_LEDGER.md         # Phase별 시작/종료, 게이트 결과, 반복 횟수, 회귀(re-entry) round 기록
│   ├── TASK_BOARD.md           # 태스크 단위 진행 상황 — dispatch 시점과 완료 시점에 즉시 갱신 (게이트 시점이 아님). 엄격 구역 검증은 검증자별 개별 행으로 기록
│   ├── DECISION_LOG.md         # 모든 중요 결정 + 근거 (예: contribution 구조 채택/기각, 모델명, placeholder 위치)
│   ├── ERRATA.md               # 이 문서 자체의 오류/모호 발견 시 정오 기록 (문서는 수정하지 않고 여기에 기록)
│   ├── AGENT_ROSTER.md         # 정의한 sub-agent 명세 (역할/입력/산출물/리뷰 상대)
│   ├── REQUESTS_AND_FEEDBACK.md# agent 간 요청·피드백 라우팅 테이블 (요청자→대상, 내용, 상태)
│   └── PHASE_REPORTS/          # phase0_report.md ~ phase8_report.md
├── 01_research_understanding/  # Phase 1 산출물
├── 02_venue_study/             # Phase 2 산출물
├── 03_blueprint/               # Phase 3 산출물
├── 04_references/
│   ├── library/                # reference card (논문 1편당 1개 md 파일)
│   ├── REFERENCE_LIBRARY_INDEX.md
│   ├── CLAIM_CITATION_MAP.md
│   ├── VERIFICATION_LEDGER.md  # 서지 검증 기록 (검증자/소스/일시/판정)
│   ├── REFERENCES_IEEE.md      # IEEE 스타일 정리본
│   └── refs.bib
├── 05_manuscript/
│   ├── sections/               # 섹션별 드래프트 + 리비전 히스토리
│   ├── MANUSCRIPT_v1.md, v2.md, ...
│   └── PLACEHOLDER_REGISTRY.md # figure/table/inline-number placeholder 전수 목록
├── 06_style_audit/
├── 07_latex/                   # Overleaf-ready LaTeX 프로젝트 + build/ + pdf_qa/
├── 08_final_audit/
└── 99_reviews/                 # 모든 리뷰 산출물: {phase}_{artifact}_{round}.md
```

규칙:
- 모든 산출물 파일 상단에 frontmatter: 생성 Phase, 작성 agent, 충족하는 Directive ID 목록, 최종 수정일.
- `INDEX.md`는 매 Phase 게이트 통과 시 갱신 — 나중에 "X에 대한 내용이 어디 있지?"라는 질문에 INDEX만 보고 답할 수 있어야 한다.
- 각 Phase 게이트 통과 시 git commit (paper/ 디렉토리 파일만 선택적으로 staging, 커밋 메시지 예: `Paper: Phase 3 blueprint complete (gate passed)`).

---

## 5. Agent 팀 운영 규약

### 5.1 팀 설계 원칙
- **작업 agent와 리뷰 agent를 분리**한다. 자기 산출물을 자기가 리뷰하는 것은 금지. 이 분리는 모든 산출물 경로에 예외 없이 적용된다 (Phase 8의 Notion 명세 포함).
- 각 agent에게는 **그 작업에 필요한 것만** 정확히 준다: 관련 Directive 원문(§9에서 그대로 발췌), 입력 파일 경로, 산출물 파일 경로, 금지사항(`paper_legacy/` 접근 금지, 코드 수정 금지 등). 전체 지시사항 57개를 모든 agent에게 쏟아붓지 마라 — 과부하는 누락을 낳는다.
- agent가 작업 중 다른 agent의 산출물이 필요하거나 의문이 있으면, 산출물에 `REQUEST:` / `FEEDBACK:` 블록을 남기게 하라. 너는 `00_admin/REQUESTS_AND_FEEDBACK.md`에 등록하고 해당 agent에게 라우팅한 뒤 결과를 회신시켜라.
- 이 환경에는 과거 프로젝트용으로 정의된 agent 페르소나(`fresh-paper-*`, `paper-*` 등)가 다수 존재한다. **역할이 맞으면 재사용해도 좋다.** 단:
  - 정의에 박힌 과거 가정(IEEE 템플릿, 12페이지 등)은 **이 문서가 전부 override한다** — Elsevier 템플릿, 9페이지가 진실이다.
  - **페르소나 정의에 하드코딩된 모든 파일 경로는 무효로 선언**하고, 프롬프트에 이번 태스크의 입출력 경로를 명시적 전체 목록으로 제공하라. 과거 `paper/` 경로의 산출물은 현재 `paper_legacy/` 아래로 이동했으므로, 정의된 경로를 탐색하다 `paper_legacy/`로 진입하는 것이 A4 위반의 전형적 경로다 — 이 경고를 재사용 프롬프트에 반드시 포함하라.
- 병렬화 가능한 독립 작업은 병렬로 dispatch하라. 단, 동일 파일을 두 agent가 동시에 쓰는 일이 없도록 산출물 경로를 분리하라.

### 5.2 권장 페르소나 로스터 (Phase 0에서 확정하여 AGENT_ROSTER.md에 기록)

| 페르소나 | 역할 | 주요 Phase |
|----------|------|-----------|
| research-archaeologist | 코드베이스/문서 정독, 연구 이해 | 1 |
| config-forensics | 271 config 사용/미사용 component 코드 추적 | 1 |
| notion-analyst | Notion 2개 페이지 정독·구조화 (MCP) | 1 |
| venue-scout | 탑티어 학회/논문 조사, 구조 패턴 분석, 문장 표본 corpus 수집 | 2 |
| anchor-paper-analyst | Self-Distilled MAE 논문·NRdetector 심층 분석 | 2 |
| narrative-architect | 논문 블루프린트·contribution 설계 | 3 |
| outline-red-teamer | 블루프린트 적대적 비판 | 3 |
| claim-citation-mapper | 블루프린트의 주장→필요 근거(인용 수요) 매핑 | 4 |
| reference-scout | 후보 reference 탐색 | 4 |
| excerpt-curator | 원문 발췌 + 활용 맥락 카드 작성 | 4 |
| source-verifier ×2 (독립) | 서지정보 official source 교차 검증 | 4 |
| section-drafter (섹션별) | 영어 본문 작성 | 5 |
| method-truth-auditor | 본문 ↔ Phase 1 진실 대조 + notation 검증 + 미등재 수치 색출 | 5, 6(spot), 7(diff) |
| plagiarism-guardian | 표절·근접 의역 검사 | 5, 6, 7(diff) |
| claim-citation-auditor | 주장↔인용 양방향 정합성 검사, 인용 필요 부분 탐지 | 5 |
| style-auditor ×2 / ai-phrasing-detector | 문장 단위 학술 문체·AI 티 검출 | 6, 7(diff) |
| terminology-normalizer | 분야 표준 용어·notation 표기 정합성 | 6 |
| latex-engineer | Elsevier LaTeX 조판 | 7 |
| pdf-qa-reviewer | 컴파일된 PDF 페이지 단위 시각 검수 | 7 |
| adversarial-reviewer (범용) | 모든 Phase 게이트의 독립 리뷰 | 전체 |
| coverage-auditor | Directive 커버리지 감사 | 0, 각 게이트, 8 |
| notion-publisher | Notion 하위 페이지 생성 | 8 |

필요하면 페르소나를 추가·세분화하라. 핵심은 **모든 산출물에 독립 리뷰어가 붙는 것**이다.

### 5.3 리뷰 피드백 루프 (모든 산출물에 의무 적용)

```
작업 agent 산출 → 독립 adversarial 리뷰 (99_reviews/에 기록)
  → 발견사항 분류: BLOCKER / MAJOR / MINOR
  → 작업 agent(또는 신규 agent) 수정 → 수정분 재리뷰
  → BLOCKER=0, MAJOR=0 이 될 때까지 반복 (MINOR는 해소 또는 DECISION_LOG에 waive 사유 기록)
```

- **심각도 기준 (모든 리뷰어 프롬프트에 포함)**: BLOCKER = Directive/절대 원칙 위반, 사실 오류, 무결성(인용·표절·진실 정합) 위험. MAJOR = 의미 손실 또는 명백한 품질 저하 위험. MINOR = 다듬기 수준. **waive는 MINOR에만 허용되며, A1/A2/A3 관련 발견은 어떤 등급이든 waive 불가.**
- 리뷰어 프롬프트에는 위 rubric과 **해당 산출물에 적용되는 Directive 원문**을 반드시 포함하고, "통과시키는 것이 아니라 깨뜨리는 것이 너의 임무"임을 명시하라.
- 모든 게이트는 **최소 1회의 실제 리뷰 라운드**를 요구한다(첫 시도에 깨끗해 보여도 리뷰는 수행하고 기록을 남긴다).
- **절대 엄격 구역**(각 Phase에 명시: reference 검증, 표절, 최종 감사 등)은 강화 프로토콜: 서로 다른 관점의 **독립 리뷰어 2인 이상** + 수정 후 **전체 재검증 1라운드 추가**.

---

## 6. 품질 게이트 & Coverage 추적

### 6.1 Phase 게이트
각 Phase는 아래를 모두 만족해야 종료된다:
1. 모든 산출물이 §5.3 리뷰 루프 통과.
2. **coverage-auditor가 해당 Phase에 매핑된 Directive(§9.4) 각각에 대해 "충족 근거(산출물 파일+섹션)"를 확인**하고 COVERAGE_MATRIX.md를 갱신. 근거를 못 대는 Directive가 있으면 게이트 실패 → 해당 작업 재개.
3. **정합성 우선순위**: §9.4 매핑 표가 게이트 판정의 정본(canonical)이다. §7의 "적용 Directive" 목록과 §9.4가 불일치하는 것을 발견하면, 게이트 진행 전 `00_admin/ERRATA.md`에 기록하고 **두 목록의 합집합 기준으로 감사**하라.
4. INDEX.md / PHASE_LEDGER.md / TASK_BOARD.md 갱신, git commit.
5. Phase 보고 (§8).

### 6.2 Coverage Matrix
- Phase 0에서 §9의 모든 Directive(T1–T7, R1–R37, M1–M13 — **총 57행**)를 행으로 등재. 전사 후 §9.1–9.3의 ID를 1번부터 순차 열거 대조하여 **행 수 일치(57)를 기계적으로 확인**하라 — §9.4 표 자체에 오류가 있더라도 §9.1–9.3 전수가 기준이다.
- 상태 전이는 반드시 근거 포인터와 함께. "했음"이라는 자기 선언은 근거가 아니다.
- 한 Directive가 여러 Phase에 걸치면 Phase별 부분 충족을 각각 기록, 전부 충족 시 DONE.
- **Phase 8 최종 게이트: 모든 행이 DONE + 근거 유효성 재검증.** 단 하나의 누락도 허용되지 않는다.

### 6.3 Phase 회귀(re-entry) 프로토콜
후속 Phase(특히 Phase 8 최종 감사)에서 미달 판정으로 이전 Phase N으로 되돌아갈 때:
1. PHASE_LEDGER에 재진입 round를 기록한다.
2. 영향받는 COVERAGE_MATRIX 행을 IN_PROGRESS로 강등한다.
3. 수정 후, N+1..7의 영향 게이트를 **축약 재실행**한다. 최소 의무: 표절 재검사(plagiarism-guardian), 변경 문장 문체 spot 검사(ai-phrasing-detector), method/experiments 변경 시 method-truth 대조, LaTeX 재컴파일 + pdf-qa + 9페이지 예산 재확인.
4. manuscript는 v4, v5, ...로 버전을 증가시킨 후 Phase 8 재감사로 복귀한다.

---

## 7. Phase 실행 계획

> 공통: 각 Phase 시작 시 §0-2 수행. 각 Phase의 "적용 Directive"에 나열된 ID의 **원문을 §9에서 다시 읽고** 착수하라. 모든 sub-agent 프롬프트에 A4(paper_legacy 금지)와 A9(코드 read-only)를 포함하라.

---

### Phase 0 — 셋업 & 지시사항 내재화

**목적**: 워크스페이스·추적 체계·agent 팀을 구축하고, 지시사항 누락 0을 구조적으로 보장하는 장치를 가동한다.

**절차**:
1. §4 구조 생성. `00_admin/` 관리 파일 초기화.
2. §9 전체를 기반으로 COVERAGE_MATRIX.md 작성 (§9.4의 초기 매핑 표 사용). **전사 후 §9.1–9.3을 기준으로 T1–T7, R1–R37, M1–M13을 순차 열거 대조하여 총 57행임을 기계적으로 확인** — §9.4 표 자체의 누락/오류도 검출 대상이며, 발견 시 ERRATA.md에 기록하고 §9.1–9.3 기준으로 보정한다.
3. **독립 coverage-auditor agent**를 띄워, 이 문서 §9와 `ORIGINAL_USER_DIRECTIVES.md`를 처음부터 다시 읽고 (a) Registry에 원문 대비 등재 누락·왜곡이 없는지, (b) Matrix에 등재 누락된 문장/요구가 없는지 교차 검증시켜라 (이 문서 자체에 대한 감사 — §9.4 표의 오류도 잡아야 한다).
4. AGENT_ROSTER.md 확정 (§5.2 기반, 가감 가능).
5. **사전 점검(pre-flight)**:
   - (a) `paper/elsarticle/` 템플릿 번들 존재 확인 + `kpsewhich elsarticle.cls`로 클래스 설치 확인 (2026-06-10 기준 확보·검증 완료: 번들 존재, cls 시스템 설치, num 템플릿 스모크 컴파일 성공. 원문 T7이 지칭한 `paper/elsevier template.txt`는 이 번들로 대체됨 — ERRATA.md에 기록).
   - (b) Notion MCP로 두 페이지 접근 가능 확인.
   - (c) `paper/윤기오_대한산업공학회_2026_춘계.pdf` 읽기 가능 확인.
   - (d) LaTeX→PDF 변환 도구(latexmk/pdflatex) 동작 확인.
   - (e) 271 결과 폴더 확인: 재귀 탐색으로 `experiment_metadata.json` 전수 개수 확인 (작성 시점 37개 — 진행 중 실험으로 늘 수 있음).
   - (f) **웹 접근 확인**: WebSearch 동작 + WebFetch로 DBLP/arXiv/OpenReview/publisher 대표 URL 각 1건 fetch 성공 확인 (Phase 2 venue 조사와 Phase 4 서지 검증의 절대 전제). sub-agent에서도 동일 도구 사용 가능 여부 확인. 실패 시 BLOCKER로 보고.
6. PHASE_LEDGER/TASK_BOARD에 Phase 1–8 계획 등재.

**산출물**: §4의 골격 전체, COVERAGE_MATRIX.md, AGENT_ROSTER.md, phase0_report.md
**게이트**: coverage-auditor의 "Matrix 등재 누락 0" 판정 + pre-flight 결과 기록.
**적용 Directive**: M1–M13, R14, R37, A9

---

### Phase 1 — 연구 완전 이해 (절대 엄격 구역: 271 config 진실)

**목적**: 현재 연구에 대해 완벽하게 이해한다 (T1). 이후 모든 Phase가 의존하는 "연구의 진실"을 문서화한다.

**절차**:
1. **research-archaeologist**: 프로젝트의 모든 스크립트와 문서(md 파일)를 정독. `mae_anomaly/` 전 모듈, `scripts/`, `docs/` (ARCHITECTURE.md, DATASET.md, ABLATION_STUDIES.md 등). → `01_research_understanding/CODEBASE_UNDERSTANDING.md`
2. **notion-analyst**: 방법론 개요 페이지 + 비교 실험 페이지를 MCP로 정독·구조화. **R2 적용: 논리·서술을 그대로 받아들이지 말고, "Notion의 주장"과 "검증된 사실"을 구분하여 기록** (예: Notion에 정리된 contribution이 있다면, 그것이 적절한지/그 구조를 따라갈지는 Phase 3에서 별도 판단할 사안임을 명시). → `01_research_understanding/NOTION_DIGEST.md`
3. **config-forensics**: `find results/experiments/271_20260602_020545_271canon_baseline -name experiment_metadata.json`으로 **전체 metadata를 재귀 수집** (entity 중첩 구조: PSM 직하, SMD `machine-*-*/`, SMAP/MSL 채널 폴더, SWaT `A1A2_full/`·`A1A2_excl22/`, WaDi `A1/`·`A2/`; 작성 시점 37개 — 개수 직접 확인). **전 파일의 config 블록을 교차 대조하여 공통(canonical) config와 dataset별 차이를 분리 기록**하고, 불일치 발견 시 BLOCKER로 보고. SWaT의 full/excl22 이중 조건이 R28(22번 이상 영역 제외 지표)의 데이터 근거임을 명시. 이후 코드베이스(`mae_anomaly/config.py`, `model.py`, `loss.py`, `scoring.py` 등)를 구체적으로 추적하여 **실제로 쓰인 component와 안 쓰인 component를 명확히 구분**한 표 작성 (각 항목에 metadata 필드 + 코드 file:line 근거). dynamic margin 등 미사용 옵션을 명시적으로 "논문 제외" 목록에 등재. Gaussian smoothing도 제외 목록에 등재 (R34). → `01_research_understanding/271_CONFIG_TRUTH.md`
4. **실험 프로토콜 진실 문서화**: 다음을 코드/Notion/metadata에서 확인하여 정리 → `01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md`
   - 데이터셋 구성: 논문 포함 대상은 SWaT, WaDi, PSM, SMD, SMAP, MSL 계열. **Simulation과 Exathlon은 제외** (R33).
   - Main 실험 split: 테스트 데이터를 길이 기준 반반으로 나눠 앞 50%를 train에 포함시키는 프로토콜. 시간적으로 더 뒤쪽 데이터를 test로 사용하기 위함 + 실험 공정성을 위해 취사선택 없이 모든 데이터셋에 통일 적용 (R13).
   - Unsupervised 비교군의 label 활용 방식: 알려진 이상을 학습 데이터에서 제거하여 순도 높은 정상 데이터로 학습 (R12, R31).
   - 평가지표: vus_roc, vus_pr, pak_auc_f1, pak_auc_pr, affiliated-f1 (+ PA-F1은 제시하되 비판) — 각 지표의 **정식 학술 명칭**을 확인해 두라 (R29; 내부 변수명을 논문에 옮기지 않기 위한 사전 작업, R24).
   - Threshold: test 데이터 anomaly 비율 사용 프로토콜 (R30).
   - SWaT '22번 이상 영역' 제외 지표의 배경 (R28).
   - 라벨 희소화(label sparsification) sweep 실험 계획 (R32).
5. **참고 자료 정독**: `paper/윤기오_대한산업공학회_2026_춘계.pdf` 전체를 읽고 연구의 핵심 논리 요약 → `01_research_understanding/CONFERENCE_PDF_DIGEST.md`
6. 위 산출물을 종합한 `01_research_understanding/RESEARCH_SYNTHESIS.md` 작성: 방법론의 각 component, 그 component가 다변량 시계열에서 갖는 의미(R10의 원재료), semi-supervised/PU 설정의 정의(R11), 코드 공개 계획(git, R25) 등.

**절대 엄격 구역**: 271_CONFIG_TRUTH.md — 모든 사용/미사용 판정에 코드 근거(file:line) 필수. 추측 금지. 독립 리뷰어가 근거를 하나하나 재추적하여 검증.
**게이트**: adversarial-reviewer가 산출물 전체의 정확성·근거를 검증 + config-forensics 산출물은 강화 프로토콜(리뷰어 2인).
**적용 Directive**: T1, R2, R10(원재료 수집), R11(정의 문서화), R12, R13, R17, R24(정식 명칭 확인), R25, R26(인지), R28, R29, R30, R31, R32, R33, R34, M8

---

### Phase 2 — 탑티어 논문 구조 연구

**목적**: 본격적인 논문 내용 구성 전에, 퀄리티 높은 논문의 논리적 흐름·포함 내용·구성을 파악하고 틀을 잡기 위한 정보를 얻는다 (T2).

**절차**:
1. **venue-scout**: 최근 3년(2024–2026)의 탑티어 AI 학회를 리스트업 (NeurIPS, ICML, ICLR, KDD, AAAI, IJCAI, WWW, VLDB 등 + 시계열/이상탐지 강세 venue). 각 학회에서 높은 평가를 받는 논문들(수상작, oral/spotlight, 고인용)을 조사하되, **시계열 이상탐지 모델 논문을 반드시 포함**. → `02_venue_study/VENUE_AND_PAPER_LIST.md`
2. 선정 논문들의 **논리적 흐름과 구성**(섹션 구조, intro의 논증 전개, contribution 제시 방식, related work 조직법, 실험 설계 서술법)을 분석. **어떤 plot/figure/table이 들어가는지** 유형별로 정리 (architecture diagram, 성능 비교 표, ablation 표, sensitivity plot, qualitative 시각화 등 — 각각 어느 섹션에 어떤 크기로). 추가로 **실제 탑티어 시계열 이상탐지 논문의 문장 표본(섹션별 5–10문장, 출처 표기)** 을 수집하라 — Phase 6 문체 검사의 기준 corpus가 된다. → `02_venue_study/STRUCTURE_AND_FIGURE_PATTERNS.md`, `02_venue_study/SENTENCE_CORPUS.md`
3. **anchor-paper-analyst**: 두 핵심 논문 심층 분석. **dossier 안에서 원문을 인용할 때는 reference card와 동일한 규약(따옴표 + 출처 표기 + "본문 복사 금지" 경고문)을 적용**하라 — dossier 문장이 본문으로 흘러가는 무자각 표절 경로를 차단한다.
   - *Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors*: 구조 분석 + **해당 논문이 왜 그 구조를 'self-distilled'라고 부르는지 확인** (R21의 방어논리 원재료). 우리 방법론과의 유사/차이 지점을 내부 문서로 정리(논문에 차이점 나열식으로 쓰기 위함이 아니라, R9의 포지셔닝 전략 수립용). → `02_venue_study/ANCHOR_SDMAE_DOSSIER.md`
   - *NRdetector* (시계열 이상탐지의 거의 유일한 semi-supervised learning): **실험 구성과 논리를 상세 분석** (R16) — baseline reference 처리 방식(실험 섹션에서만 인용하는 방식, R19 근거), 실험 프로토콜 서술법, PU 설정 정당화 논리. 공통점보다 **차이점 위주로 강조할 재료** 추출 (R20). → `02_venue_study/NRDETECTOR_DOSSIER.md`

**게이트**: adversarial-reviewer — 분석의 깊이(표면적 요약이 아닌, Phase 3에서 바로 쓸 수 있는 실행 가능한 패턴인지), 시계열 이상탐지 논문 포함 여부, 문장 corpus 확보 여부, 두 dossier의 완성도.
**적용 Directive**: T2, R9(준비), R16(준비), R19(근거 수집), R20(준비), R21(방어논리 확인)

---

### Phase 3 — 논문 블루프린트 (틀 잡기)

**목적**: 연구 내용을 바탕으로 논문의 전체 개요·틀을 잡는다 (T3). 전체 구성 → 섹션 구성 → 각 섹션에 포함될 내용 → 필요한 근거를 대략적으로 확정한다.

**절차**:
1. **narrative-architect**가 Phase 1·2 산출물을 입력으로 블루프린트 작성 → `03_blueprint/PAPER_BLUEPRINT.md`:
   - 전체 논문 구성, 섹션·서브섹션 구조, 각 섹션에 들어갈 내용과 논증 흐름, 각 주장에 필요한 근거(→ Phase 4의 수요 목록). **Elsevier 논문의 필수 요소(abstract, keywords, 그리고 템플릿이 요구할 경우 highlights)도 섹션 계획에 포함** — 이들도 Phase 5에서 작성되고 Phase 6 감사를 거쳐야 한다.
   - **Related work, contribution, 실험 부분은 MECE하게 구성** (R1).
   - **Contribution 설계 (R8 — 매우 핵심)**: novelty를 충분히 탐색하고 충분히 강조하는 구조. Notion의 contribution 정리가 있다면 각 contribution이 적절한지, 그 구조를 그대로 따라갈지 **먼저 판단** 후 채택/수정/기각을 DECISION_LOG에 기록 (R2).
   - **Semi-supervised / positive-unlabeled 설정을 논문의 중심 환경으로** (R11): 훈련 데이터 대부분이 unlabeled(정상인지 이상인지 모름)이지만 일부는 실제 고장 발생 등으로 이상 label이 있는 상황. 기존 unsupervised 이상탐지는 대량의 unlabeled 데이터로 전체 분포는 학습하지만, **소수지만 매우 중요한 labeled 데이터를 활용하지 못한다**는 것이 핵심 motivation.
   - **각 component에 대해 "왜 다변량 시계열 데이터에서 이 방법을 적용해야만 하는가"가 충분히 설명되는 구조** (R10). '다변량 시계열'이라는 도메인 자체가 핵심임을 반영. 이 질문에 답이 안 되는 component가 있다면 치열하게 고민하여 논리를 만들고 블루프린트에 명시.
   - **SSL/PU related work 전략** (R20): 기존 방법론의 목표를 충분히 언급하되, **시계열에서는 해당 연구가 거의 존재하지 않음을 충분히 강조**. NRdetector는 공통점보다 차이점 위주로.
   - **Anchor 논문 포지셔닝** (R9): Self-Distilled MAE는 핵심 인용이지만 지나치게 유사하게 느껴지지 않도록. **"차이점 나열 방식" 금지** — 그 차이점 말고는 매우 유사하다고 받아들여져 novelty가 적어 보임. 숨기지는 않되 자연스럽게 언급하고 넘어가는 방식으로, 지나치게 강조하지 않는다. self-distillation 용어는 해당 논문의 선례를 방어논리로 (R21).
   - **Patch/masking의 계보** (R22): 오직 vision의 Masked Autoencoder에서 영향을 받음. 시계열의 patch/masking 연구들은 영향·계승 관계가 아니라 단지 유사점이 있을 뿐 — 혼동 금지.
   - **Baseline 언급 정책** (R19): 단순 성능 비교용 모델들은 실험 섹션 인용으로 충분. 핵심 계승 요소가 있거나 직접 비교 대상인 경우만 related work에서 설명 (NRdetector 논문 방식 참고).
   - **Notation 설계 방침** (R5): 오류 없되 최대한 일반적이고 이해하기 쉬운 방식 (참고 자료의 notation은 참고만).
   - **페이지 예산** (R6): appendix·reference 제외 9페이지, table/figure 크기 넉넉히 가정한 섹션별 분량 배분 → `03_blueprint/PAGE_BUDGET.md`
   - **Appendix 구성 계획** (R7).
   - **Figure/Table 계획**: Phase 2 패턴 기반으로 어떤 figure/table이 어느 섹션에 필요한지 초안 (placeholder 명세의 시작점).
2. **모델명/제목 결정** (R15): 불필요한 축약어를 새로 정의하지 않되, 논문 제목·모델 이름·모델 축약어는 novelty가 뛰어나 보이는 방향으로 후보 3–5개 생성 + 장단점 → DECISION_LOG에 선정 기록.
3. **outline-red-teamer**: 블루프린트 적대적 비판 — novelty가 충분히 부각되는가? reviewer가 reject할 약점은? MECE 위반은? anchor 논문 대비 derivative하게 보이지 않는가? PU 설정의 motivation이 설득력 있는가? → 수정 루프.

**게이트**: outline-red-teamer + adversarial-reviewer 2중 리뷰, 모든 BLOCKER/MAJOR 해소.
**적용 Directive**: T3, R1, R2, R5, R6, R7, R8, R9, R10, R11, R15, R16, R19, R20, R21, R22, R32(실험 구성 반영)

---

### Phase 4 — Reference 확보 & 절대 검증 (절대 엄격 구역: 할루시네이션 0)

**목적**: 블루프린트의 각 부분을 채우는 데 필요한 reference를 탐색·발췌·검증하여 논문에 활용할 근거자료를 확보한다 (T4).

**절차**:
1. **claim-citation-mapper**: 블루프린트에서 인용이 필요한 모든 주장 지점을 추출 → `04_references/CLAIM_CITATION_MAP.md` (주장 → 필요한 근거 유형 → 후보 reference → 이후 "주장↔근거 발췌 포인터"로 발전).
2. **reference-scout**: 각 수요에 대해 reference 탐색. **되도록 퀄리티 높은 논문** (탑티어 학회 또는 고인용). 비교 실험 Notion 페이지의 baseline 모델 reference와 데이터셋 reference는 **엄격 검증을 거친 truth로 활용 가능** (R26) — 단, 서지 메타데이터의 최종 표기는 그래도 공식 소스로 재확인.
3. **excerpt-curator**: 논문 1편당 1개의 reference card 작성 → `04_references/library/{key}.md`. **매번 다시 논문을 읽을 필요가 없도록**, 각 논문에서 내용을 발췌하여 **원본의 표현 그대로(verbatim, 섹션/페이지 표기)** + **내 논문에서 어떤 맥락으로 쓰일 수 있는지**를 함께 정리. **card에는 제목과 abstract 전문도 저장** (표절 검사 corpus 용도). (verbatim 발췌는 card 내부 전용 — 본문에 무단 복사되면 표절. card에 경고문 포함.)
4. **서지 검증 (강화 프로토콜)**: 두 명의 **독립 source-verifier**가 각 reference를 검증하되, **독립성을 구조적으로 보장**한다:
   - verifier A: card의 메타데이터를 공식 소스(학회/저널 공식 proceedings, publisher 페이지, DOI, OpenReview, arXiv 공식 페이지, DBLP 교차)에서 **하나하나 여러 번 검증**.
   - verifier B: **card 메타데이터를 보지 않고**, 공식 소스(DBLP/publisher DOI)에서 BibTeX를 새로 export. orchestrator가 양쪽을 **필드 단위로 기계적 diff**. `refs.bib` 항목은 손 타이핑 금지 — 공식 BibTeX export 기반으로만 생성.
   - 두 verifier는 서로의 ledger 기록을 보지 않고 병렬 작업한다.
   - **아주 약간의 추측이나 추론도 절대 금지.** 서지 필드 검증 실패가 하나라도 있으면 그 reference는 `QUARANTINE` 처리 — 절대 인용 목록에 들어가지 않는다.
   - **2단계 격리**: 서지 필드 검증 실패 = reference 전체 QUARANTINE. 전문(full text) 접근 불가로 **발췌만** 원문 대조가 불가한 경우 = 해당 발췌에 `EXCERPT_UNVERIFIED` 마킹 (직접 인용·verbatim 활용 금지; reference 자체는 사용 가능). 접근 가능한 경우 verbatim 발췌가 실제 원문에 존재하는지 재확인.
   - **QUARANTINE 발생 시**: CLAIM_CITATION_MAP의 해당 수요를 OPEN으로 되돌리고 reference-scout가 대체 후보 탐색 → 동일 검증 파이프라인.
   - 모든 검증 기록(검증자, 확인한 소스 URL, 일시, 판정)을 `VERIFICATION_LEDGER.md`에 남김.
5. **IEEE 스타일 reference 목록 정리** → `REFERENCES_IEEE.md` + `refs.bib` 생성. (주: 최종 LaTeX의 bibliography 스타일은 Phase 7에서 Elsevier 템플릿이 요구하는 형식을 따른다. REFERENCES_IEEE.md는 중간 산출물의 표준 정리본이고 refs.bib는 스타일 중립적 데이터다.)
6. `REFERENCE_LIBRARY_INDEX.md` 갱신: key, 제목, venue/연도, 검증 상태, 활용 예정 위치.

**절대 엄격 구역**: 4번 전체. **여기서 할루시네이션이 발생하면 절대로 안 된다.**
**게이트**: 독립 리뷰어가 VERIFICATION_LEDGER를 표본이 아닌 **전수**로 재감사 + 무작위 재검증 라운드 1회 추가. QUARANTINE 항목이 인용 목록에 없는지 + CLAIM_CITATION_MAP의 어떤 행도 QUARANTINE key를 가리키지 않는지 확인.
**적용 Directive**: T4, R26, R36(1차 — 이후 Phase 5에서 추가 수요 발생 시 이 Phase의 파이프라인을 미니 사이클로 재가동)

---

### Phase 5 — 영어 본문 작성 (절대 엄격 구역: 표절 0, 방법론 진실 정합, 수치 창작 0)

**목적**: 블루프린트(P3)와 근거자료(P4)를 활용해 논문 본문을 완성한다 (T5). 영어로, 완벽하게 완성된 형태로.

**절차**:
1. **섹션별 section-drafter** 운영 (abstract / intro / related work / method / experiments / conclusion 등 블루프린트 구조대로 — abstract·keywords·highlights도 이 Phase에서 작성). 각 drafter에게: 블루프린트 해당 섹션, **PAGE_BUDGET.md의 해당 섹션 분량**, 관련 reference card, Phase 1 진실 문서, 해당 섹션에 적용되는 Directive 원문(**R4 원문 포함 — 초안부터 자연스러운 학술 문체로 작성**)을 제공. 섹션별 드래프트 → `05_manuscript/sections/`.
2. 본문 작성 시 반드시 반영할 내용 (각 섹션 drafter 프롬프트에 해당분 포함):
   - **Method**: 271 config의 실제 사용 component만 (R17, A3). 각 component마다 "왜 다변량 시계열에서 이렇게 해야만 하는가" 설명 (R10). 구현 방식을 지나치게 구체적으로 하나하나 나열하지 말 것 — 필요한 정보, 핵심 정보만 (R27). hyperparameter는 꼭 필요한 것만 구체값을 쓰고 주로 일반적 서술 사용 (R23). notation은 일반적이고 이해하기 쉽게 (R5). 내부 변수명·연구과정 용어 사용 금지 (R24). 너무 지엽적인 것은 생략 (R35). self-distillation 용어는 SDMAE 선례 기반으로 자연스럽게 (R21), anchor 논문은 자연스럽게 언급하고 넘어가기 (R9), patch/masking은 vision MAE 계보로 서술 (R22).
   - **Experiments**: NRdetector dossier의 프로토콜 서술법을 참조하라 (R16). main 실험 프로토콜은 **동기부터 서술**: 기존 시계열 이상탐지 벤치마크는 훈련 데이터에 anomaly가 포함되지 않은 경우가 대부분이므로, 테스트 데이터에 포함된 anomaly를 학습 단계에 반영하기 위해 테스트 데이터를 길이 기준 반반 분할하여 앞 50%를 train에 포함 — 시간적으로 더 뒤쪽 데이터를 test로 사용하기 위해 + 공정성을 위해 취사선택 없이 전 데이터셋 통일 적용 강조 (R13). unsupervised 비교군은 알려진 이상을 학습 데이터에서 제거하는 방식으로 label 활용 — 이것이 unsupervised가 label을 활용하는 최선의 방법이며, **이상 데이터를 포함하면 성능이 하락**하기 때문 (R12, R13). **공정성 방어 논리** (R31): label을 활용하는 기존 시계열 이상탐지 모델 자체가 부족하다는 점 + unsupervised의 label 활용 방식 제공 — 설득력 있게. 평가지표 5종(VUS-ROC, VUS-PR, PA%K 기반 AUC-F1/AUC-PR, Affiliation-F1 — 정식 명칭은 Phase 1 확인 결과 사용)이 **각각 어떤 다른 관점을 평가하며 어떻게 상호보완적인지** 설명, 실험 결과는 매우 긍정적이므로 **모든 관점에서 뛰어난 성능**임을 강조. **PA-F1은 전체적인 관점에서도 좋은 성능을 보이기에 제시하되**, 해당 지표의 challenge·문제점을 지적하여 주 지표로 참고하지 않음을 명시 (R29). Threshold는 test 데이터 anomaly 비율 사용 — test label을 쓰지만 (1) threshold 무관 지표들을 같이 제시해 보완하고 (2) 평가 protocol일 뿐임을 설득력 있게, cherry-picking으로 보이지 않게 서술 (R30). SWaT의 '22번 이상 영역'이 지배적으로 거대해 포함 시 비교가 무의미해지므로 제외 지표를 별도 제시 — 충분히 설명 (R28). 라벨 희소화 sweep 실험 포함 + anomaly가 unlabeled 상태로 훈련 데이터에 섞여도 모델이 강건한 이유에 대한 논리적·설득력 있는 설명 (R32). Simulation/Exathlon 미포함 (R33), Gaussian smoothing 미언급 (R34). baseline 인용은 실험 섹션에서 (R19).
   - **수치 창작 절대 금지**: 산문 내 모든 정량 결과값(지표 값, 개선폭, win/loss 수, p-value 등)은 직접 기입하지 말고 `[X.XX]` 형태의 **inline placeholder**로만 쓰고 PLACEHOLDER_REGISTRY에 유형 `inline-number`로 등재. 성능 우수성은 정성적 서술로 표현 (A8).
   - **Figure/Table**: 어디에 어떤 형태로 삽입될지 본문에 포함. 준비되지 않았으므로 `[placeholder]`로 두되 어떤 내용이 들어갈지 서술. **제목(캡션)과 설명은 placeholder여도 완성된 형태로 작성** (R3). 모든 placeholder를 `PLACEHOLDER_REGISTRY.md`에 등재 (ID, 유형 figure/table/inline-number, 위치, 캡션, 들어갈 내용, 크기 가정).
   - **실험이 잘 되었다고 가정**하고 흐름에 맞게 작성. 실험 데이터 부족 지적 금지 (R3, A8).
   - **코드 공개**: git으로 공개 예정 — 논문에 넣는 게 자연스러우면 넣고, 아니면 넣지 않는다 (R25). 판단을 DECISION_LOG에 기록.
   - Related work / contribution / experiments **MECE** (R1). SSL/PU related work는 R20 전략대로.
3. 섹션 통합 → `MANUSCRIPT_v1.md` (frontmatter에 버전·날짜). 통합 후 전체 흐름·일관성 패스 1회 (transition, 용어 통일, 중복 제거) + **단어수 기반 페이지 추정 체크 1회** (PAGE_BUDGET 대비 — Phase 7에서의 분량 충격을 줄인다).
4. **인용 보강 루프 (R36)**: claim-citation-auditor가 전체 본문을 훑어 "일반적 서술이 아니라 뒷받침 근거가 필요한데 인용이 없는 부분"을 전수 탐지 → 해당 부분을 인용할 수 있는 논문을 Phase 4 파이프라인(탐색→발췌→2인 검증→ledger)으로 확보하여 reference를 단다. 신규 reference도 예외 없이 동일한 검증 강도 (T4 보강 사이클).
5. **검증 루프 (병렬)**:
   - **method-truth-auditor**: 본문 전체 ↔ `271_CONFIG_TRUTH.md`/`EXPERIMENT_PROTOCOL_TRUTH.md` 대조. 미사용 component 언급, 사실과 다른 서술 전수 색출. **추가 임무 2건**: (a) **notation 검증** — 모든 수식 기호가 첫 사용 전 정의되는지, 기호 충돌·재정의가 없는지, 수식이 271_CONFIG_TRUTH의 실제 계산과 일치하는지 (R5), (b) **미등재 수치 색출** — Experiments 산문에서 PLACEHOLDER_REGISTRY에 등재되지 않은 구체 수치 주장을 grep 수준으로 전수 색출 (A8).
   - **plagiarism-guardian (강화 프로토콜)**: 본문을 (a) 모든 reference card의 verbatim 발췌 + **abstract 전문**, (b) Phase 2 dossier들, (c) anchor 논문들과 대조. **인용 표시 없이 원본 표현을 그대로 베껴온 부분이 없는지 여러 번 체크** — 연속 어구 일치, 근접 의역(구조만 바꾼 복사) 모두 검출. 추가로 related work·정의 문장 등 **고위험 구절에서 8단어 이상의 특징적 n-gram을 추출해 웹 검색으로 spot-check** (corpus에 없는 원천에서의 복사 검출). 발견 시 재서술.
   - **claim-citation-auditor (양방향)**: R36 탐지(절차 4)에 더해, **본문의 모든 기존 인용을 전수 순회하며 주장 내용이 해당 reference card의 verbatim 발췌로 실제로 뒷받침되는지 역방향 검증**. card 발췌로 뒷받침되지 않으면 (a) 원문을 fetch해 확인 후 card에 발췌 추가(2인 검증 경유), 또는 (b) 주장 재서술/인용 교체. 결과를 CLAIM_CITATION_MAP.md에 "주장↔근거 발췌 포인터"로 기록 — 실존 논문에 허위 귀속하는 할루시네이션을 차단한다.
   - **adversarial-reviewer**: 논증 품질, contribution 부각(R8), MECE(R1), anchor 포지셔닝(R9) 준수 여부.
6. 모든 루프 통과본 → `MANUSCRIPT_v2.md`.

**절대 엄격 구역**: 표절 검사, 인용 정합성(양방향), 방법론 진실 정합, 수치 창작 금지.
**게이트**: 위 감사 전부 BLOCKER/MAJOR 0 + coverage-auditor의 Phase 5 Directive 전수 확인.
**적용 Directive**: T4(보강 사이클), T5, R1, R3, R4(초안 단계 예방), R5, R8–R13, R15, R16(실험 서술 참조), R17, R19–R25, R27–R36, A2, A3, A5, A8

---

### Phase 6 — 학술 문체 정밀 검증 (문장 단위)

**목적**: AI·딥러닝·시계열 이상탐지 분야에서 잘 쓰이는, 충분히 academic하고 전문적이며 자연스러운 어휘·표현·어구를 사용하는지 아주 철저하게 검증한다 (T6).

**절차**:
1. **ai-phrasing-detector**: **문장 단위 전수 검사** — 'AI가 작성한 티'가 나는 문구, 일반적으로 논문 작성에서 쓰이지 않는 표현을 철저하게 검출 (R4). **판정 기준**: Phase 2의 `SENTENCE_CORPUS.md`(실제 탑티어 논문 문장 표본)를 기준 corpus로 삼고, 구체적 금지 패턴 목록(예: delve, comprehensive 남발, 형식적 3연 병렬 구문, 과도한 전환 부사)을 LEDGER에 유지하며 대조 판정 — 주관적 인상만으로 판정하지 마라. 검출 항목마다: 원문장 / 문제 유형 / 수정안. → `06_style_audit/AI_PHRASING_LEDGER.md`
2. **style-auditor (독립 2인)**: 동일 본문을 서로 다른 관점(학술 자연스러움 / 분야 관용 표현)으로 문장 단위 재검사. 한 명이 통과시킨 문장도 다른 한 명이 잡을 수 있다.
3. **terminology-normalizer**: 분야 표준 용어 정합성 — 해당 도메인에서 일반적으로 쓰이는 표현인지 확인. 확신이 없는 표현은 실제 논문 용례를 검색하여 근거를 남긴다. **notation이 분야에서 일반적이고 이해하기 쉬운 표기인지 최종 확인** (R5). 불필요한 신규 축약어가 정의되지 않았는지 확인 (R15). 내부 용어·변수명 잔존 여부 재확인 (R24). 너무 지엽적인 서술 잔존 확인 (R35).
4. 수정 반영 → 수정분 재검사(수정이 새로운 어색함을 만들지 않았는지) → `MANUSCRIPT_v3.md`.
5. **회귀 검사 2종**: (a) **plagiarism-guardian 재가동** — 문체 수정 과정에서 reference 원문 표현으로 회귀한 부분이 없는지 최종 1회 검사. (b) **method-truth spot-check** — method/experiments 섹션에서 변경된 문장에 한해 method-truth-auditor가 진실 문서 대조 1회 수행 (용어 교체가 기술적 의미를 바꾸지 않았는지).

**게이트**: 3종 검사 모두 잔존 지적 0 (또는 DECISION_LOG에 waive 사유 — MINOR만 가능) + 회귀 검사 2종 통과.
**적용 Directive**: T6, R4, R5(notation 최종 점검), R15, R24, R35, A2

---

### Phase 7 — LaTeX 조판 (Elsevier) & PDF 시각 검증

**목적**: 완성된 논문의 format을 맞춘다 (T7). Overleaf에서 쓸 수 있는 LaTeX로, 템플릿을 철저하게 지켜 하나의 완성된 논문을 만든다.

**절차**:
1. **latex-engineer**: `paper/elsarticle/` 번들을 정독하고 요구사항을 체크리스트화 → `07_latex/TEMPLATE_REQUIREMENTS.md`. 정독 대상: 공식 문서 `doc/elsdoc.pdf`(documentclass 옵션 — 1p/3p/5p 레이아웃, review/final 모드, 패키지, frontmatter 명령, 섹션/표/그림/수식/참고문헌 규칙), 템플릿 `elsarticle-template-num.tex`(기본 채택 — numbered citation이 Elsevier 표준; `-num-names`/`-harv` 변형 중 다른 선택을 하면 DECISION_LOG에 사유 기록), 대응 `.bst` 스타일, 레이아웃 샘플 `doc/elstest-*.pdf`. `elsarticle.cls`는 TeX Live에 시스템 설치되어 있음(번들 원본은 수정 금지, 작업 사본은 `07_latex/`에).
2. `MANUSCRIPT_v3.md` → `07_latex/` 아래 Overleaf-ready LaTeX 프로젝트로 변환. bibliography는 템플릿이 요구하는 스타일로 `refs.bib` 연동. **이 시점부터 `.tex`가 정본(canonical)이며 MANUSCRIPT_v3.md는 동결**한다.
3. **산문 변경 통제**: 레이아웃 수준을 넘는 산문 변경(문장 삭제·압축·신규 작성 — 분량 조정 포함)이 발생하면, 변경분(diff)에 대해 **ai-phrasing-detector + plagiarism-guardian + method-truth-auditor 미니 감사**를 게이트 전에 의무 수행한다 — 감사를 거치지 않은 새 산문이 최종본에 들어가서는 안 된다.
4. **Figure/table placeholder**: 적절한 위치에 적절한 크기로 너(latex-engineer)가 판단해서 삽입 (PLACEHOLDER_REGISTRY 기반, 캡션은 완성된 형태). 크기는 **REGISTRY에 기록된 가정값(높이/폭)을 그대로 사용**하고, 넉넉하게 가정 (R6). 실제 figure 삽입 시 재조정이 필요할 수 있음을 핸드오프 보고에 명시.
5. **컴파일 → PDF 직접 확인 루프**: LaTeX→PDF 변환 도구는 이미 설치되어 있음. 컴파일 후 **pdf-qa-reviewer**가 PDF를 페이지 단위로 직접 열어 확인: 페이지 깨짐, 표/그림 overflow, 수식 줄넘침, 캡션·참조 정합, 전체 구성 적절성, **페이지 예산 판정 규칙 — 본문(appendix·reference 제외, table/figure 포함)이 9페이지를 초과하지 않으며 8.5페이지 이상을 채울 것** (R6), appendix 구성 적절성 (R7). 발견사항 수정 → 재컴파일 → 재검수, 깨끗해질 때까지 반복. QA 기록 → `07_latex/pdf_qa/`
6. **Overleaf 직업로드 패키징 (최종 인도물)**: 사용자는 이 zip을 **그대로 Overleaf에 업로드**한다 — 어떤 수동 수정/추가도 전제하지 마라. `07_latex/overleaf_package.zip` 요구사항:
   - **구조**: zip 루트에 main `.tex` 배치 (Overleaf가 main 문서를 자동 인식하도록; 섹션 분할 시 하위 tex 파일들 포함).
   - **포함**: main tex(+분할 tex), `refs.bib`, 선택한 `elsarticle-*.bst`, `elsarticle.cls` 사본(Overleaf에 elsarticle이 기본 포함되어 있으나 버전 고정을 위해 동봉), 사용한 모든 `.sty`, figure placeholder가 외부 그림 파일을 참조한다면 해당 파일 전부 (권장: placeholder는 TikZ/framebox 등 외부 파일 의존 없는 방식으로 작성).
   - **제외**: build 부산물(.aux/.log/.bbl/.synctex 등), pdf_qa 기록, 내부 작업 문서.
   - **Self-contained 검증 (게이트 조건)**: zip을 임시 폴더에 압축 해제하고 **그 폴더 안의 파일만으로** `latexmk -pdf` 컴파일이 성공해야 한다 — 누락 파일이 하나라도 있으면 게이트 실패.
   - 최종 산출: 컴파일 성공한 LaTeX 프로젝트 + 검수용 PDF + `overleaf_package.zip`.

**게이트**: 컴파일 무경고(불가피한 경고는 사유 기록) + pdf-qa-reviewer 시각 검수 통과 + 페이지 예산 충족 + 산문 변경분 미니 감사 통과 + **zip 압축 해제 단독 컴파일 성공 (self-contained 검증)**.
**적용 Directive**: T7, R3(placeholder 배치), R6, R7

---

### Phase 8 — 최종 감사, Notion Placeholder 페이지, 핸드오프

**목적**: 출판 수준 완성도 최종 점검 (R18) + placeholder 명세의 Notion 문서화 (R3) + 사용자 핸드오프.

**절차**:
1. **최종 감사 (절대 엄격 구역)**: 신규 adversarial-reviewer 2인이 최종 PDF/LaTeX를 처음 보는 reviewer의 눈으로 전체 검토 — **"이것이 정말 출판된 level의 하나의 완성된 논문인가?"** (placeholder는 허용). **실제 학회/저널 리뷰 양식으로 작성**: novelty/soundness/clarity/significance 점수 + 강점/약점 + accept-reject 판정 + "내가 reject한다면 그 사유". reject-사유급 약점은 BLOCKER로 분류. 검토 항목에 **related work/contribution/experiments의 MECE 구조(R1) 재확인**, 논증 완결성, contribution 설득력, 실험 서술 완성도, 표절·인용 무결성 스팟 재검증 포함. → `08_final_audit/FINAL_AUDIT_REPORT.md`. 미달 판정 시 **§6.3 회귀 프로토콜**에 따라 해당 Phase로 되돌아가 수정 루프 (하류 게이트 축약 재실행 의무 포함).
2. **coverage-auditor 최종 전수 감사**: COVERAGE_MATRIX의 모든 행(T1–T7, R1–R37, M1–M13 — 57행)이 DONE이고 근거 포인터가 유효한지 하나하나 재확인. **단 하나의 누락도 불허.** 누락 발견 시 즉시 해당 작업 수행 후 재감사.
3. **Notion placeholder 명세 — 2단계 발행**:
   - (a) **명세 초안 작성**: `08_final_audit/NOTION_PLACEHOLDER_SPECS.md`에 먼저 작성 (입력: PLACEHOLDER_REGISTRY + EXPERIMENT_PROTOCOL_TRUTH + 271_CONFIG_TRUTH). PLACEHOLDER_REGISTRY의 **각 figure 및 table별로** (inline-number placeholder 포함), 어떤 실험 혹은 figure가 들어가야 하는지를 **아주 구체적이고, 퀄리티 높은 자연스러운 한국어 문장**으로 정리 (R3) — 어떤 실험을 어떤 설정으로 돌려 어떤 형태(축, 계열, 비교 대상)로 시각화/표화해야 하는지 **재현 가능한 수준**으로. → 독립 리뷰어가 §5.3 루프로 검증: 한국어 품질, 실행 가능성, REGISTRY 전수 일치.
   - (b) **발행**: 검수 통과 후에만 notion-publisher가 발행. 원문 R3의 "[노션 페이지]" 지칭은 모호하므로 **비교 실험 페이지 하위 단일 명세 페이지(figure/table별 섹션)를 기본값**으로 하되, 이것이 해석임을 가능한 가장 이른 Phase 보고(늦어도 Phase 5 보고)의 ⑤항에 사용자 확인 질문으로 등재하고, 응답이 없으면 기본값으로 진행 + DECISION_LOG 기록. 발행은 markdown 전체를 **한 번의 페이지 생성 호출**로 (긴 내용을 update 계열로 넣으면 렌더링이 깨진다). 발행 후 re-fetch로 렌더링 확인.
4. **워크스페이스 마감**: INDEX.md 최종 갱신(전 산출물 + "무엇을 어디서 찾는가"), 전체 git commit.
5. **최종 핸드오프 보고** (§8 형식): **최종 인도물 `paper/07_latex/overleaf_package.zip`의 경로와 "그대로 Overleaf에 업로드하면 됨"을 첫 줄에 명시**, 산출물 전체 목록, 주요 결정사항(모델명, contribution 구조, placeholder 목록), 사용자 액션 필요 항목(실험 수행→placeholder 채우기, figure 크기 재조정 가능성 등), 미해결 항목.

**게이트**: 최종 감사 PASS (모의 피어리뷰에서 reject-사유급 약점 0) + 커버리지 100% (57행) + Notion 검수 통과본 발행 + 렌더링 확인.
**적용 Directive**: R1(MECE 최종 확인), R3, R14, R18, M9, M10, M13 + 전 Directive 최종 확인

---

## 8. 보고 및 블로커 프로토콜

- **매 Phase 종료 시**: `00_admin/PHASE_REPORTS/phase{N}_report.md` 작성 + 동일 내용을 사용자에게 채팅으로 보고. 형식: ① 수행 내용 요약 ② 산출물 목록(경로) ③ 게이트/리뷰 결과(반복 횟수 포함) ④ 주요 결정사항 ⑤ **사용자 확인이 필요하거나 요청할 사항** ⑥ 다음 Phase 예고.
- 보고 후 **기본적으로 다음 Phase를 자율 진행**한다 (M13). 단, 다음의 경우에만 진행을 멈추고 사용자 응답을 기다린다: (a) `paper/elsarticle/` 템플릿 번들 유실·손상 상태로 Phase 7 진입 시점 도달, (b) Notion 접근 불가, (c) 연구 내용에 대한 상충 정보로 사용자만이 판단 가능한 사안, (d) 파괴적·되돌리기 어려운 작업, (e) **웹 접근 불가로 reference 검증(Phase 4) 수행 불능**. 그 외 질문은 보고의 ⑤항에 모아서 전달하되 작업은 계속한다 (보수적 기본값을 선택하고 DECISION_LOG에 기록).
- 모든 작업이 끝나면 Phase 8의 최종 핸드오프 보고로 마무리한다.

---

## 9. Directive Registry — 사용자 지시사항 전체

> **이 섹션이 이 프로젝트의 헌법이다.** 아래 항목의 어떤 문장이나 단어도 간과되거나 생략되어서는 안 된다. 모든 항목이 매우 중요한 핵심 지시사항이다.
> 표기: §9.1(T)·§9.2(R)는 **원문 그대로**다 — 단, 영문 철자 교정 2건(Latex→LaTeX, Self-distilation→Self-distillation)과 '작업 :'/'목적 :'의 콜론 앞 공백 제거 외 내용·단어 변경 없음 (사용자 대화 원문의 인코딩 깨짐 문자 ç/ƒ는 `ORIGINAL_USER_DIRECTIVES.md` 작성 시 이미 c/f로 정규화되어 있음). §9.3(M)은 대화체 원문의 **충실한 지시문 재서술**이다 — 완전한 원문은 `paper/ORIGINAL_USER_DIRECTIVES.md` 참조.

### 9.1 작업 지시 (T1–T7)

**[T1]** 작업: 현재 프로젝트의 모든 스크립트, 문서 (md파일), 그리고 위 노션 페이지를 정독해서 현재 연구에 대해 완벽하게 이해해. / 목적: 현재 연구에 대해 완벽하게 이해하는 것

**[T2]** 작업: 최근 3년 (현재 2026년) 간의 탑-티어 AI 학회들을 리스트업하고, 해당 학회들의 높은 평가를 받는 논문들을 주로 살펴봐서 논문의 논리적 흐름, 구성에 대해 파악해. 특히, 시계열 이상탐지 모델은 포함되어야 해. 또한 어떤 plot 혹은 figure, table이 들어가는지도 잘 파악해. / 목적: 본격적인 논문 내용을 구성하기 전 퀄리티 높은 논문의 논리적 흐름이나 포함되어야 하는 내용, 구성 등을 파악하고, 틀을 잡기 위한 정보를 얻는 것.

**[T3]** 작업: 내 연구 내용을 바탕으로, 논문의 전체적인 개요 및 틀을 잡아줘. 전체적인 논문의 구성부터 시작해서, 섹션 구성, 각 섹션에 포함되어야 하는 내용들, 필요한 근거 등을 우선 대략적으로 잡으면 돼. / 목적: 논문 틀 잡기

**[T4]** 작업: 위 작업 (3)에서 대략적으로 잡은 틀을 바탕으로, 각 부분의 내용을 채우기 위해 필요한 reference 탐색에 들어가서 필요한 논문들을 탐색해줘. reference는 되도록 퀄리티 높은 (top-tier 학회나 피인용횟수가 높은 논문들) 논문으로 해야해. 또한, 매번 다시 논문을 읽을 필요가 없도록, 각 논문에서 내용을 발췌하여, 내 논문에서 어떤 맥락으로 쓰일 수 있는지를 원본의 표현 그대로와 함께 정리해. 더해서, 여기서 찾은 논문은 직접적으로 reference로 활용할거야. 따라서 각 논문의 reference가 정말 확실한지 아주 철저하게 검증할 필요가 있어. 여기에서 할루시네이션이 발생하면 절대로 안돼. 아주 약간의 추측이나 추론은 절대 안되고, 명백히 official한 source에서 하나하나 여러번 검증해야해. 마무리로 IEEE 스타일의 reference 인용 표시로 정리해. / 목적: 논문에 활용할 근거자료 확보

**[T5]** 작업: 작업 (3)에서 잡은 틀 및 작업 (4)에서 얻은 근거자료를 활용해서 논문 본문을 작성해. 영어로 작성해야해. 완벽하게 완성된 형태로 작성하면 돼. figure 및 table의 경우 어디에 어떤 형태로 삽입되면 될지도 포함하고, figure 및 table은 우선 준비되지 않았으니 [placeholder] 형태로 두되, 어떤 내용이 들어가면 될지 서술해. 표절은 절대 금지야. 인용 표시 없이 원본의 표현을 그대로 베껴오는 일이 없도록 여러번 체크해. (4)에서 찾은 reference를 통해 뒷받침해. / 목적: 논문 본문 완성

**[T6]** 작업: AI, 딥러닝, 그리고 시계열 이상탐지 분야에서 잘 쓰이는, 충분히 academic하고 전문적인, 자연스러운 어휘, 표현, 어구를 사용하고 있는지 아주 철저하게 검증할 필요가 있어. AI로 생성한듯한 문구나, 일반적으로 논문 작성에서 쓰이지 않는 표현 등을 쓰는지를 문장 단위로 철저하게 검사하는게 매우 중요해.

**[T7]** 완성된 논문의 format을 맞춰야해. 일반적인 overleaf를 활용해서 LaTeX로 작성할거야. 템플릿에 대한 정보는 paper/elsevier template.txt에도 있어. LaTeX 형태로, 템플릿을 철저하게 지켜서 하나의 완성된 논문을 만드는 것이 목표야. figure나 table의 placeholder의 경우 적절한 위치에 적절한 크기로 너가 판단해서 집어넣어. LaTeX를 pdf로 변환하는 tool은 이미 설치되어있으므로, pdf로 변환하여 결과를 직접 확인하여 페이지가 깨지는지, 구성은 적절한지, 테이블 및 피규어 구성도 적절한지 등을 확인하는 과정도 거쳐야해.

### 9.2 참고사항 (R1–R37)

**[R1]** Related work와 contribution, 실험 부분은 MECE하게 구성할 것.

**[R2]** 참고자료 (특히 notion page)의 논리 및 서술은 참고하되, 절대적으로 따르지는 말 것. 충분한 판단 후에 활용할 것. (가령 contribution이 정리되어 있는데, 각 contribution이 적절한지, 해당 contribution 구조를 그대로 따라갈지에 대해서는 판단이 먼저 있어야 함)

**[R3]** Figure 및 실험 부분은, 현재 실험 데이터 및 figure가 없더라도, placeholder로 미리 틀만 만들어서 비워놓고, 실험의 경우에는 '해당 실험이 잘 되었다고' 가정하고 글을 작성하고, 각 placeholder 에 어떤 실험 혹은 figure가 들어가야 할 것인지에 대해서는 별도의 [노션 페이지]에 하위 페이지를 만들어서 각 Figure 및 table 별로 아주 구체적이고, 퀄리티 높은 자연스러운 한국어 문장으로 정리해놓을것. 현재 실험데이터가 부족한건 지적하지말고, 이건 한계도 아니므로, placeholder만 만들고, 내 방법론의 의도와 목표에 맞는 실험결과가 있다고 가정하고 본문을 흐름에 맞게 작성할것. 테이블 및 figure의 제목 및 설명은, placeholder더라도, 완성된 형태로 작성해놓을것.

**[R4]** 문장이 'AI가 작성한 티' 가 나거나, 해당 도메인에서 일반적으로 쓰이지 않는 표현, 연구논문에서 일반적으로 쓰이지 않는 표현인지 엄격하게 검증할 것.

**[R5]** 수식에 쓰이는 Notation 등은 오류가 없되, 최대한 일반적이고 이해하기 쉬운 방식으로 정리할 것. (참고 자료는 참고만 할것)

**[R6]** 전체 분량은 appendix 및 reference 제외, table과 figure를 모두 합쳐 9page로 구성할 것. Table과 figure 크기는 넉넉하게 가정할 것.

**[R7]** Appendix 구성에도 주의할 것.

**[R8]** Contribution 강조가 매우 핵심이라는 점을 잊지 말 것. Novelty를 충분히 탐색하고, 해당 Novelty를 충분히 강조할것.

**[R9]** Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors는 핵심 인용이지만, 지나치게 유사하게 느껴지지 않도록 주의할것. 특히, "저 논문과의 차이점을 나열하는 방식" 은 오히려 해당 차이점 말고는 매우 유사한 방법론이라고 받아들여져 novelty가 적어보일 수 있음. 숨기지는 않되, 자연스럽게 언급하고 넘어가는 방식으로, 해당 논문을 지나치게 강조하지 않도록 주의할것.

**[R10]** 방법론을 구성하는 각 component에 대해, 왜 '다변량 시계열 데이터에서 이러한 방법을 적용하였는지' 에 대해서도, '왜 이렇게 해야만 하는가' 가 충분히 설명되도록, 충분한 설명이 되고 있는지에 대해서 치열하게 고민하고 논문에도 반영할 것. '다변량 시계열 데이터' 라는 도메인도 중요한 핵심임.

**[R11]** 'semi-supervised learning', 혹은 'positive-unlabeled learning' 이라는 환경에 집중할 것. 훈련 데이터셋이 대부분 이상인지 아닌지 모르는 unlabeled 상태지만, 그 중 일부는 실제 고장상황 발생 등으로 이상 label이 되어 있는 상황을 가정하고 있으며, 기존 unsupervised learning 기반의 이상탐지는 대량의 unlabel 데이터로부터 전체 데이터 분포는 학습할 수 있지만, 소수 존재하지만 매우 중요한 labeled 데이터를 활용하지 못하는 것이 매우 핵심임.

**[R12]** 실험 부분에서, Unsupervised learning의 경우, label을 활용하는 최선의 방법이, 학습데이터에서 제거하여 순도 높은 정상데이터로 학습시키는 것이라는 점을 참고할 것.

**[R13]** Main 실험은, 기존의 시계열 이상탐지 벤치마크는 훈련 데이터에 anomaly가 포함되어있지 않은 경우가 대부분임. 본 실험에서는 테스트 데이터에 포함되어 있는 anomaly를 학습 단계에 반영하기 위해, 테스트 데이터를 길이 기준으로 반반 나눠서, 앞의 50%를 train data로 포함시킨 실험임. 이러한 상황에서는 기존 unsupervised learning에서는, 학습 데이터에 포함된 알려진 이상데이터를 제거하여 순도 높은 정상으로만 이루어진 학습 데이터를 구성하는 것임. (이상 데이터 포함시 성능 하락) 시간적으로 더 뒤쪽의 데이터를 test로 사용하기 위해, 그리고 실험 공정성을 위해 취사선택 없이 모든 데이터셋에 대해서 통일된 방법으로 나눈다는 점도 강조해.

**[R14]** 이후에 수정 작업이나 별도 작업을 할때 기존 작업물을 충분히 참고할 수 있도록, 중간 작업물들도 잘 정리해놓고, 각 중간 작업물들에서 필요한 내용을 쉽게 찾을 수 있도록 index를 비롯하여 잘 정리해놓을것.

**[R15]** 불필요한 축약어를 새로 정의하지 말것, 단 논문 제목이나 모델의 이름, 모델 축약어는 novelty가 뛰어나보이는 방향으로 정할것.

**[R16]** NRDetector 논문의 실험 구성이나 논리를 참고할 것. 방향성이 비슷한 점이 있음. 거의 유일한 시계열 이상탐지에서의 semi-supervised learning임.

**[R17]** 271번의 config에 대해서만 사용하고, option으로 남겨져있으나 271번 config에서 실제로 사용되고 있지 않은 모든 부분들은 전부 무시해. 가령 dynamic margin 등은 271번 config에서는 활용하고 있지 않아. 271번 실험 결과 폴더 내부의 metadata 파일에 구체적인 config이 기록되어 있으니, 코드베이스의 코드를 구체적으로 추적하여서 실제로 쓰인 component와 안 쓰인 component를 명확히 구분해서, 불필요한 내용이 들어가지 않도록 해.

**[R18]** 모든 작업이 끝나면, 이게 정말 출판된 level의 하나의 완성된 논문인지 점검해. (placeholder는 허용)

**[R19]** Baseline에 쓰인 모든 논문을 related work 등에 언급할 필요는 없음. 단순히 성능 비교를 위한 비교 모델들은, 실험 부분에서 reference를 다는 것으로 충분할 수 있음. (nrdetector 논문 참고) 내 방법론의 핵심적인 계승 요소가 있거나 직접적인 비교 대상이 될 수 있는 등의 경우에만 설명할것.

**[R20]** SSL 혹은 PU-learning 관련 related work에는, 기존의 방법론의 목표 등을 충분히 언급하되, 시계열에서는 해당 연구가 거의 존재하지 않는다는 점을 충분히 강조할 필요가 있음. 따라서, nrdetector에 대해서도, 공통점보다는 차이점 위주로 강조할것.

**[R21]** Self-distillation이라는 표현이 기존 용례와 다를 수 있음. 이는, Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors에서, 유사한 구조를 이미 self-distilled라는 용어를 사용했다는 것을 방어논리로 삼을것. 해당 논문에서 왜 이 구조를 그렇게 부르는지 확인할 필요도 있음.

**[R22]** 본 연구의 patch 및 masking 방식은, 오직 vision 의 Masked Autoencoder에서 영향을 받음. 시계열에서도 patch 및 masking을 활용한 경우가 있지만, 이것은 내가 해당 논문들에 영향을 받았거나 계승한 것이 아니라 그냥 방법론에 유사한 점이 있다는 점에 그친다는 것을 명심하고 혼동하지 말것.

**[R23]** 방법론 설명할때 hyperparameter 하나하나를 구체적으로 언급할 필요는 없음. 각 하이퍼 파라미터의 구체적인 값은 꼭 필요한 것만 설명하고, 일반적인 서술을 주로 사용할 것.

**[R24]** 내가 연구 과정, 혹은 실험 정리 과정에서 쓰인 용어를, 적합하지 않음에도 불구하고 그대로 사용하지 말것. 특히 변수명을 사용하거나 하는 것은 금지. 출판 level의 매우 높은 퀄리티의 논문 작성이 목표라는 것을 명심할 것.

**[R25]** 코드는 git으로 공개할 예정임. 논문에 넣는게 자연스러우면 넣고, 아니면 넣지 말것.

**[R26]** [노션 페이지] 의 비교 대상 모델 reference 및 데이터셋 reference는 매우 엄격한 검증을 거친 truth로 활용할 수 있음.

**[R27]** 방법론의 구현방식을 지나치게 구체적으로 하나하나 전부 나열할 필요는 없음. 필요한 정보, 핵심적인 정보를 담도록 주의할것.

**[R28]** SWaT의 경우, '22번 이상 영역' 이 지배적으로 거대해서 해당 영역을 포함하여 평가 metric을 계산할 경우, 비교가 무의미해진다는 문제가 있어, 해당 영역을 제외한 성능지표를 별개로 제시함. 이 부분에 대해서도 충분한 설명이 필요함.

**[R29]** 평가지표의 경우 vus_roc, vus_pr, pak_auc_f1, pak_auc_pr, affiliated-f1을 사용할 것임. 해당 지표들이 모델의 각각 다른 관점들을 평가하며, 해당 관점들이 어떻게 상호보완적인지 설명할 필요가 있음. 실험 결과는 매우 긍정적임. 따라서 각 지표가 평가하는 방법에서 전부 뛰어난 성능을 보인다는 점을 강조할 필요가 있음. 전체적인 관점에서 좋은 성능을 보여 Pa_f1도 제시할거지만, 해당 지표의 challenge 및 문제점을 지적해서 참고하지 않을 것이라는 점을 명시.

**[R30]** Threshold의 경우, test 데이터의 anomaly 비율을 사용할 것임. Test data의 label을 활용하긴 하지만, 첫째로 threshold랑 무관한 지표들을 같이 제시하여 보완하며, 둘째로 평가를 위한 protocol 일 뿐임. 이러한 점을 설득력 있게, cherry-picking으로 보이지 않게 설명할것.

**[R31]** 실험의 경우, SSL 등 알려진 이상을 활용하는 기존의 방법론이 없어, 불공정하게 느껴질 수 있음. 1차적으로는 unsupervised 방법론의 경우 알려진 이상을 훈련데이터에서 제외하는 방식으로 레이블을 활용하며, 해당 애초에 비교할 수 있는 레이블을 활용하는 시계열 이상탐지 모델이 부족하다는 점과 함께, 이를 방어할 수 있는 충분한 논리를 포함할것. 설득력이 있어야 함.

**[R32]** 라벨 희소화 sweep 실험도 포함할 것. 진행할 예정임. 또한, anomaly가 unlabeled인 상태로 훈련 데이터에 섞이더라도 이 모델이 강건한 이유에 대한 논리적이고 설득력 있는 설명도 포함할 것.

**[R33]** Simulation 데이터셋 및 exathlon 데이터셋은 포함되지 않을 예정임.

**[R34]** Gaussian smoothing에 대한 내용은 빼. 안 쓸거야.

**[R35]** 너무 지엽적인건 생략해.

**[R36]** 일반적인 서술이라기보단 뒷받침하는 의견이 필요하기 때문에 인용을 하는 것이 좋은 부분인데, 실제로 인용이 되어있지 않은 경우에는, 해당 부분을 인용할 수 있는 논문을 찾아서 레퍼런스를 달아줘야 해.

**[R37]** ./paper_legacy 에 있는 작업물 (이전 작업물) 은 절대 참고하지 말 것. 무시할 것.

### 9.3 메타 지시 (M1–M13)

> M 항목은 대화체 원문의 충실한 지시문 재서술이다. 원문 그대로의 표현이 필요하면 `paper/ORIGINAL_USER_DIRECTIVES.md`의 "메타 지시"/"마무리 지시" 절을 직접 발췌하라.

**[M1]** Claude Code는 프로젝트 전체적인 계획을 수립 및 관리하며, 특히 sub-agent들을 정의, 작업 배정, 관리하고, sub-agent들끼리의 팀 작업을 하는 역할을 해야 함.

**[M2]** 각 작업별로 리뷰 전문 sub-agent를 통한 피드백 루프는 성공의 핵심임.

**[M3]** 필요시에 다른 sub-agent에게 작업 요청 혹은 피드백을 할 수 있어야 하고, Claude Code는 orchestrator로서 중간에서 그 과정을 잘 조율해야 함.

**[M4]** 작업의 퀄리티를 높일 수 있는 모든 테크닉을 활용할 것. 토큰은 매우 많이 사용할 수 있으므로, 시간적 효율성이나 토큰 효율성 등을 따지지 않고 오직 최상의 퀄리티를 목적으로 할 것.

**[M5]** 한꺼번에 너무 많은 지시를 내리면 누락되거나 간과될 수 있으므로, 적절한 시점에서 끊어서 차례대로 지시할 수 있도록 phase를 나눌 것.

**[M6]** 각 phase에서 절대적인 엄격함이 필요한 파트들이 있음 (예: reference 표기, 표절 방지 등).

**[M7]** 작업은 ./paper/ 디렉토리에서 진행. 작업 계획부터 시작해서 각 단계의 작업 결과 등을 아주 철저하게 구조화하는 게 핵심임.

**[M8]** Notion 페이지는 MCP로 접근 가능함.

**[M9]** 매 phase마다 해당 phase의 내용을 정리하여 보고하고, 필요하거나 요청사항이 있으면 사용자에게 정리해서 알려줄 것.

**[M10]** 지시사항 및 참고사항의 그 어떤 문장이나 단어도 간과되거나 생략되어서는 안 됨. 모든 지시사항이 매우 중요한 핵심 지시사항임. 절대로 누락되는 게 있어서는 안 됨.

**[M11]** 전체 작업 과정을 아주 깊게 고민해서, 나열된 지시사항·참고사항을 단순히 순서대로 따르지 말고 최대한 효과적인 process와 그에 따르는 phase로 재구성하되, 모든 내용이 포함되도록 할 것. 첫 phase부터 잘 구성할 것. 프롬프트(지시)의 퀄리티가 결과물의 퀄리티를 좌우함.

**[M12]** 입력 자료: 방법론 개요 Notion 페이지, 비교 실험 Notion 페이지 (§3의 URL), 참고 자료 `paper/윤기오_대한산업공학회_2026_춘계.pdf`.

**[M13]** Orchestrator는 이 전체 내용이 포함된 하나의 프롬프트를 한꺼번에 받아서, 절대 누락되거나 생략하거나 잃지 않고 작업을 처음부터 끝까지(Phase 0→8) 완료할 것 (§8의 자율 진행 규칙이 이행 수단).

### 9.4 초기 Coverage 매핑 (Phase 0에서 COVERAGE_MATRIX.md로 전사 — 총 57행: T 7 + R 37 + M 13)

> 이 표는 게이트 판정의 정본이다. 단, 전사 시 §9.1–9.3의 ID 전수를 1번부터 순차 대조하여 표 자체의 누락·오류를 검증하라 (§6.2).

| Directive | 담당 Phase | Directive | 담당 Phase |
|-----------|-----------|-----------|-----------|
| T1 | 1 | R19 | 2(근거 수집), 3, 5 |
| T2 | 2 | R20 | 2(준비), 3, 5 |
| T3 | 3 | R21 | 2, 3, 5 |
| T4 | 4 (+5 보강) | R22 | 3, 5 |
| T5 | 5 | R23 | 5 |
| T6 | 6 | R24 | 1(명칭 확인), 5, 6 |
| T7 | 7 | R25 | 1, 5 |
| R1 | 3, 5, 8 | R26 | 1, 4 |
| R2 | 1, 3 | R27 | 5 |
| R3 | 5, 7, 8 | R28 | 1, 5 |
| R4 | 5, 6 | R29 | 1, 5 |
| R5 | 3, 5, 6 | R30 | 1, 5 |
| R6 | 3, 7 | R31 | 1, 5 |
| R7 | 3, 7 | R32 | 1, 3, 5 |
| R8 | 3, 5 | R33 | 1, 5 |
| R9 | 2, 3, 5 | R34 | 1, 5 |
| R10 | 1, 3, 5 | R35 | 5, 6 |
| R11 | 1, 3, 5 | R36 | 4, 5 |
| R12 | 1, 5 | R37 | 0 (전 Phase 상시) |
| R13 | 1, 5 | M1–M7 | 0 (전 Phase 상시) |
| R14 | 0, 8 (전 Phase 상시) | M9 | 매 Phase 종료 시 |
| R15 | 3, 5, 6 | M10 | 0, 8 (전 게이트) |
| R16 | 2, 3, 5 | M11–M12 | 0 |
| R17 | 1, 5 | M13 | 전 Phase 상시, 8 |
| R18 | 8 | M8 | 0, 1 (전 Phase 상시) |

---

## 10. 최종 완료 정의 (Definition of Done)

아래 전부를 만족해야 프로젝트 완료다:

1. `paper/07_latex/overleaf_package.zip` — **압축 해제만으로 단독 컴파일되는, Overleaf에 그대로 업로드 가능한 self-contained zip** (검증 기록 포함) + 컴파일 성공한 LaTeX 프로젝트 + 검수용 PDF. 본문은 appendix·reference 제외 9페이지 이내(8.5페이지 이상, table/figure 포함, 크기 넉넉히), appendix 구성 완료.
2. 모든 reference가 VERIFICATION_LEDGER에서 2인 독립 검증 PASS. QUARANTINE 항목 인용 0건. refs.bib는 공식 BibTeX export 기반.
3. 표절 검사·문체 검사·방법론 진실 검사·양방향 인용 정합성 검사 모두 최종본 기준 PASS 기록 존재. 본문에 창작된 실험 수치 0건 (모든 수치는 inline placeholder).
4. 모든 figure/table/inline-number placeholder가 PLACEHOLDER_REGISTRY에 등재되고, 검수를 통과한 한국어 명세가 Notion 하위 페이지로 발행되어 렌더링 확인 완료.
5. COVERAGE_MATRIX의 모든 Directive(T1–T7, R1–R37, M1–M13 — 57행)가 근거 포인터와 함께 DONE.
6. Phase 0–8 보고서 전부 존재, INDEX.md 최신, ERRATA.md 반영 완료, 전 산출물 git commit 완료.
7. 최종 감사(모의 피어리뷰 양식) 판정: "출판된 level의 하나의 완성된 논문" (placeholder 허용) — PASS, reject-사유급 약점 0.

**시작하라. Phase 0부터.**
