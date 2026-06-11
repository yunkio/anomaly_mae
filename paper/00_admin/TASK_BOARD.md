---
phase: 0
agent: orchestrator
directives: [M5, M7, R14]
last_modified: 2026-06-10
---

# Task Board

> 태스크 단위 진행 상황. **dispatch 시점과 완료 시점에 즉시 갱신** (게이트 시점이 아님).
> 절대 엄격 구역의 2인 검증은 **검증자별 개별 행**으로 기록.
> 상태: PLANNED / DISPATCHED / DONE / BLOCKED / REWORK.

## Phase 0 — 셋업 & 지시사항 내재화

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| P0-1 워크스페이스 구조 생성 (§4) | orchestrator | DONE | 2026-06-10 |
| P0-2 COVERAGE_MATRIX.md 작성 + 57행 기계 대조 | orchestrator | DONE | T7+R37+M13=57 확인 |
| P0-3a 감사 A — Registry 원문 충실성 | coverage-auditor-A (독립) | DONE | PASS (B0/M0/m2 — ERRATA E-002·E-003 기록). 웹 도구 sub-agent 가용 확인 포함 |
| P0-3b 감사 B — Matrix 완전성·매핑 정합 (r1) | coverage-auditor-B (독립) | DONE | 조건부 FAIL (B0/M1/m6) → Matrix 6행 보정 |
| P0-3c 수정분 재리뷰 (r2) | coverage-auditor-r2 (독립 신규) | DONE | PASS (B0/M0/m1 → R19 표기 즉시 보정) |
| P0-4 AGENT_ROSTER.md 확정 | orchestrator | DONE | |
| P0-5 pre-flight (a) elsarticle 번들 + cls 설치 | orchestrator | DONE | kpsewhich PASS |
| P0-5 pre-flight (b) Notion MCP 2페이지 접근 | orchestrator | DONE | 방법론 78,956자 / 비교실험 112,046자 수신 |
| P0-5 pre-flight (c) 학회 PDF 읽기 | orchestrator | DONE | p.1–2 정상 렌더 |
| P0-5 pre-flight (d) latexmk/pdflatex | orchestrator | DONE | latexmk 4.76 |
| P0-5 pre-flight (e) 271 metadata 전수 개수 | orchestrator | DONE | **37개** (2026-06-10 실측) |
| P0-5 pre-flight (f) WebSearch+WebFetch (DBLP/arXiv) | orchestrator | DONE | arXiv·DBLP fetch PASS; sub-agent 가용성은 P0-3 dispatch에서 확인 |
| P0-6 PHASE_LEDGER/TASK_BOARD Phase 1–8 계획 등재 | orchestrator | DONE | 본 파일 하단 |
| P0-7 phase0_report.md + git commit | orchestrator | DONE | 게이트 PASS — `Paper: Phase 0 setup complete (gate passed)` |

## Phase 1 — 연구 완전 이해 (엄격 구역: 271_CONFIG_TRUTH)

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| P1-1 CODEBASE_UNDERSTANDING.md | research-archaeologist | DONE | r3 (reconciler+fixer-2 정정) |
| P1-2 NOTION_DIGEST.md (R2 적용) | notion-analyst | DONE | r2 (fixer-3 정정) |
| P1-3 271_CONFIG_TRUTH.md (엄격) | config-forensics | DONE | r3 (reconciler+fixer-1+fixer-5 정정) |
| P1-4 EXPERIMENT_PROTOCOL_TRUTH.md | protocol-truth-writer | DONE | r3 (fixer-4+fixer-5 정정) |
| P1-5 CONFERENCE_PDF_DIGEST.md | pdf-digest | DONE | r2 (fixer-3 정정) |
| P1-5b reconciliation (P1-1↔P1-3 모순 20건) | reconciler | DONE | `p1_reconciliation_r1.md` |
| P1-6 RESEARCH_SYNTHESIS.md | synthesis-writer | DONE | r2 + CG-1·α-m3 패치 |
| P1-7a/b/c adversarial 리뷰 (코드이해·종합 / digest 2종 / 프로토콜) | 리뷰어 3인 | DONE | r1: B5+B3+B1, 전건 해소 |
| P1-8 271_CONFIG_TRUTH 강화 검증 — 검증자 1 | verifier-1 (재추적 관점) | DONE | r1: B1/M4 → 해소 |
| P1-9 271_CONFIG_TRUTH 강화 검증 — 검증자 2 | verifier-2 (완전성 관점) | DONE | r1: B6/M3 → 해소 |
| P1-9b 재리뷰 r2 (α: truth 3종 / β: digest+프로토콜) | 재리뷰어 2인 | DONE | 잔존 B4 적발 → fixer-5 r3 해소 |
| P1-10 coverage-auditor Phase 1 게이트 감사 | coverage-auditor | DONE | PASS 18/18 + spot 4/4 (`p1_coverage_gate_r1.md`) |

## Phase 2 — 탑티어 논문 구조 연구

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| P2-1 VENUE_AND_PAPER_LIST + STRUCTURE_AND_FIGURE_PATTERNS | venue-scout | DONE | 14편 TSAD 포함 + Elsevier 관례 절 |
| P2-2 SENTENCE_CORPUS (Phase 6 기준 corpus) | corpus-collector | DONE | 11편·92엔트리·금지 패턴 시드 |
| P2-3 ANCHOR_SDMAE_DOSSIER (R21 명명 근거) | anchor-paper-analyst | DONE | self-distilled 명명 원문 확보 |
| P2-4 NRDETECTOR_DOSSIER (R16/R19/R20) | nrdetector-analyst | DONE | 사용자 중단 시점에 파일 완결 확인 (완결성 리뷰 추가 검증 예정) |
| P2-5 adversarial 리뷰 (A: venue·구조·corpus / B: dossier 2종) + 수정 루프 | 리뷰어 2인 + fixer | DONE | r1 B0/M4/m15 → fixer 26건 전수 처리 (R21 계보 강화: Zhang TPAMI 2022) |
| P2-6 coverage 게이트 감사 | coverage-auditor | DONE | PASS 6/6 + spot 7/7 (`p2_coverage_gate_r1.md`) |

## Phase 3 — 논문 블루프린트

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| P3-1 PAPER_BLUEPRINT + PAGE_BUDGET | narrative-architect | DONE | r1 |
| P3-2 red-team 비판 | outline-red-teamer | DONE | B3/M10 |
| P3-3 adversarial 리뷰 | adversarial-reviewer | DONE | B5/M12 |
| P3-4 개정 r2 (49건 전수) | blueprint-reviser | DONE | 8 BLOCKER 해소 |
| P3-5 재리뷰 r2 (양 관점) | 재리뷰어 2인 | DONE | RT: PASS_WITH_CONDITIONS / ADV: 신규 B2 적발 (GRL 이중 λ, warmup forward skip) |
| P3-6 P1 정본 회귀 보강 + r3 | fixer | DONE | 271_CONFIG_TRUTH r4 + CODEBASE r4 + SYNTHESIS r3 + 블루프린트 r3 |
| P3-7 모델명·제목 선정 (R15) | orchestrator | DONE | D-007: CSMAD + "Label-Aware Masked Autoencoding with Gradient Reversal…" |
| P3-8 coverage 게이트 감사 | coverage-auditor | DONE | PASS (spot 6/6, `p3_coverage_gate_r1.md`) + NOTE 2건 orchestrator 패치 |

## Phase 4 — Reference 확보 & 절대 검증 (엄격 구역: 서지 검증 전체)

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| P4-1 CLAIM_CITATION_MAP (claim 85, 인용 필요 72) | claim-citation-mapper | DONE | |
| P4-2 reference 후보 탐색 (OPEN 31 해소) | reference-scout | DONE | 중대 발견: 최초성 반증 2편 → D-008 스코핑 축소; venue 정정 4건 |
| P4-3 reference card 49편 (FULL 22 + LIGHT 27) | excerpt-curator ×3 | DONE | |
| P4-4a 서지 검증 A1 (1–25) | source-verifier A1 | DONE | VERIFIED 25, CRITICAL 정정 1 (kpivae 저자 24→13) |
| P4-4a 서지 검증 A2 (26–49) | source-verifier A2 | DONE | VERIFIED 24, CRITICAL 정정 3 (treemil·rosas·xue 저자) |
| P4-4b 서지 검증 B1 (blind export 1–25) | source-verifier B1 | DONE | 25/25 공식 export |
| P4-4b 서지 검증 B2 (blind export 26–49) | source-verifier B2 | DONE | 24/24 + zhang seed 결함 플래그 (정당) |
| P4-5 기계 diff (33 일치/10 표기/6 해소) + QUARANTINE 0 | orchestrator | DONE | `P4_DIFF_REPORT.md` |
| P4-6 refs.bib(49) + REFERENCES_IEEE + LIBRARY_INDEX + 통합 ledger | assembler | DONE | |
| P4-7 게이트: 전수 재감사 + 무작위 16편 재검증 | gate-auditor | DONE | 조건부 FAIL(GB-1 bib 구문) → 수정 + 49/49 파싱 → PASS |

## Phase 5 — 영어 본문 작성 (엄격 구역: 표절·진실 정합·수치 창작)

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| P5-1 섹션 drafter 4인 (front/related/method/experiments) | section-drafter ×4 | DONE | |
| P5-2 v1 통합 + PLACEHOLDER_REGISTRY + 인용 key 검증 | integrator | DONE | 분량 11.8p 적발 |
| P5-3 분량 수술 1·2차 + Appendix 작성 | budget-surgeon ×2 | DONE | D-009/D-010; 10.42p (Phase 7 실측 인계) |
| P5-4 검증 5종 병렬 (R36/truth/표절/인용 역방향/adversarial) | 감사 5인 | DONE | 발견 99건 (B17·M다수; placeholder 정책 충돌 2건은 기각) |
| P5-5 종합 수정 → MANUSCRIPT_v2 | comprehensive fixer | DONE | 94건 처리 + 정본 errata 2건 |
| P5-6 coverage 게이트 (마감+재추적+A8 스윕+Directive 32) | coverage-auditor | DONE | 조건부 FAIL(F-1 1문장) → orchestrator 정정 → PASS |

## Phase 6 — 학술 문체 정밀 검증

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| P6-1 ai-phrasing 전수 (corpus 기준) | ai-phrasing-detector | DONE | 187문장, MUST 11 |
| P6-2 style 감사 A (자연스러움) | style-auditor-A | DONE | 중단 1회 재dispatch; 214문장, MUST 25 |
| P6-3 style 감사 B (분야 관용) | style-auditor-B | DONE | 67건, Moderate 3 |
| P6-4 terminology 정합 | terminology-normalizer | DONE | Q1/Q3 11곳 판정 (매핑 정확 확인) |
| P6-5 fixer → MANUSCRIPT_v3 | style-fixer | DONE | 전수 처리 + 의미 보존 거부 16건 (audit 사실 오류 3건 교정) |
| P6-6 수정분 재검사 + 회귀 2종 (표절·truth) | 재검사 3인 | DONE | 회귀 0·truth PASS·신규 MAJOR 3 → orchestrator touch-up |
| P6-7 coverage 게이트 | coverage-auditor | DONE | PASS (`p6_coverage_gate_r1.md`); MINOR 4 waive (D-011) |

## Phase 7 — LaTeX 조판 (Elsevier) & PDF 검증

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| P7-1 TEMPLATE_REQUIREMENTS + v3→LaTeX 변환 + 컴파일 | latex-engineer | DONE | preprint 빌드; .tex 정본화 |
| P7-1b 판형 실측 (3p/5p) + 기준 판형 결정 | orchestrator | DONE | D-012: final,5p,twocolumn |
| P7-2 PDF 시각 검수 r1 | pdf-qa-reviewer | DONE | B4(5p 겹침)+B/M 다수 적발 |
| P7-3 LaTeX 수정 r2 (B7/M8 전수) | latex-engineer r2 | DONE | 산문 변경 0 |
| P7-4 D-013 한정 산문 압축 (−219w) + float 보정 | prose-compressor | DONE | 본문 8.997p 달성 |
| P7-5 산문 diff 미니 감사 3종 (§7-3 의무) | prose-miniauditor | DONE | 3종 PASS (B0/M0) |
| P7-6 overleaf_package.zip + self-contained 검증 | orchestrator | DONE | 12파일, 독립 컴파일 PASS ×2 |
| P7-7 coverage 게이트 (5조건+시각 spot) | coverage-auditor | DONE | PASS 5/5 (`p7_coverage_gate_r1.md`) |

## Phase 8 — 최종 감사 + Notion + 핸드오프

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| P8-1 모의 피어리뷰 ×2 (신규 리뷰어, 학회 양식) | final-reviewer 1·2 | DONE | Major Revision → D-014 triage (placeholder-본질 기각, 채택 2건) |
| P8-2 NOTION_PLACEHOLDER_SPECS 작성 | placeholder-spec-writer | DONE | placeholder 전수 + 신규 실행 11건 |
| P8-3 명세 독립 검수 → r2 | spec-reviewer + fixer | DONE | B1(R-PROBE)·M1(w/o OD 전제) 정정 |
| P8-4 D-014(a) Appendix B.2 보강 + 미니 감사 3종 | fixer | DONE | 본문 좌표 단위 무영향 |
| P8-5 Notion 발행 + re-fetch 검증 | notion-publisher | DONE | 단일 create-pages, 렌더링 무결 |
| P8-6 최종 coverage 전수 감사 (57행 + DoD 7항목) | final-coverage-auditor | DONE | PASS (`p8_final_coverage_r1.md`) |
| P8-7 마감 commit + 최종 핸드오프 보고 | orchestrator | DONE | |

## 재진입 (2026-06-11 — 일시 중단 중, RESUME_STATE.md 참조)

| Task | 담당 | 상태 | 비고 |
|------|------|------|------|
| RE-1 KBS 규정 조사 | journal-format-researcher | DONE | 85자 highlights·2단 판형·선언 5종 등 |
| RE-2 KBS 적용 8건 (highlights/keywords/선언/journal/flat/재컴파일/재측정/zip) | latex-engineer | DONE | 본문 8.997p 유지, zip flat 13파일 PASS |
| RE-3 KBS 산문 미니 감사 | kbs-miniauditor | DONE | PASS 4/4, NOTE 3건 핸드오프 예정 |
| RE-4 Notion 확장판 작성 (B1 본문 / B2 부록+OVERVIEW) | spec-enricher ×2 | DONE | 18페이지 분량 + 대시보드 자료 |
| RE-5 정제 (한국어·구성 통일 → NOTION_FINAL_PAGES) | notion-polisher-C | **중단됨 (미시작)** | 재개 1순위 — RESUME_STATE §3-1 |
| RE-6 독립 검수 | 검수자 D | PLANNED | |
| RE-7 발행 (MAE for AD 하위, 부모+하위 ~19장) + 구 페이지 중립화 | notion-expert | PLANNED | |
| RE-8 마감 (matrix/ledger/보고/commit) | orchestrator | PLANNED | |

## Phase 1–8 계획 (개요 등재 — 상세 태스크는 각 Phase 시작 시 전개)

| Phase | 핵심 태스크 (마스터 §7) | 엄격 구역 |
|-------|------------------------|----------|
| 1 | CODEBASE_UNDERSTANDING / NOTION_DIGEST / 271_CONFIG_TRUTH / EXPERIMENT_PROTOCOL_TRUTH / CONFERENCE_PDF_DIGEST / RESEARCH_SYNTHESIS + 리뷰 루프 | 271_CONFIG_TRUTH (코드 근거 file:line, 리뷰어 2인) |
| 2 | VENUE_AND_PAPER_LIST / STRUCTURE_AND_FIGURE_PATTERNS / SENTENCE_CORPUS / ANCHOR_SDMAE_DOSSIER / NRDETECTOR_DOSSIER + 리뷰 | — |
| 3 | PAPER_BLUEPRINT / PAGE_BUDGET / 모델명·제목 후보 / red-team 2중 리뷰 | — |
| 4 | CLAIM_CITATION_MAP / reference 탐색·card / 2인 독립 서지 검증 / refs.bib / REFERENCES_IEEE | 서지 검증 전체 (할루시네이션 0) |
| 5 | 섹션별 드래프트 → v1 → 인용 보강(R36) → 검증 루프 4종 → v2 / PLACEHOLDER_REGISTRY | 표절·인용 정합·진실 정합·수치 창작 0 |
| 6 | AI_PHRASING_LEDGER / style 2인 / terminology / 회귀 검사 2종 → v3 | — |
| 7 | TEMPLATE_REQUIREMENTS / LaTeX 변환 / placeholder 삽입 / 컴파일·pdf-qa 루프 / overleaf_package.zip self-contained 검증 | — |
| 8 | FINAL_AUDIT(모의 피어리뷰 2인) / coverage 최종 전수(57행) / NOTION_PLACEHOLDER_SPECS → 검수 → 발행 / 핸드오프 | 최종 감사 |
