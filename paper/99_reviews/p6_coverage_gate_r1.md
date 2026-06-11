---
phase: 6
agent: coverage-auditor
directives: [M10]
round: r1
last_modified: 2026-06-11
target: paper/05_manuscript/MANUSCRIPT_v3.md (orchestrator touch-up 적용본)
inputs:
  - paper/99_reviews/p6_recheck_r2.md (MAJOR 3 + MINOR 5 권고)
  - paper/99_reviews/p6_plagiarism_regression_r1.md
  - paper/99_reviews/p6_truth_spot_r1.md
  - paper/99_reviews/p6_style_fixlog_r1.md
  - paper/06_style_audit/ 4종 (AI_PHRASING_LEDGER, STYLE_AUDIT_A/B, TERMINOLOGY_AUDIT)
  - MASTER_ORCHESTRATION_PROMPT.md §7 Phase 6 / §9 (T6, R4, R5, R15, R24, R35, A2)
  - paper/00_admin/DECISION_LOG.md (D-011)
method: v3 본문 직접 grep/계수 실측 (주장 재인용 아님) + 산출물 포인터 대조
verdict: "PASS — touch-up 4/4 적용 실측, 신규 문제 0, 마커 무손상 전 항목 일치, Directive 7종 근거 확정, 게이트 조건 충족"
---

# Phase 6 Coverage Gate r1 — touch-up 확인 + Directive 근거 확정

## 1. Orchestrator touch-up 4건 — v3 실측 확인 (전건 적용)

| 건 | 권고 (p6_recheck_r2) | v3 실측 (grep) | 판정 |
|---|---|---|---|
| R2-M1 | "Exploiting … is designed to" 행위주성 불일치 → "The design exploits all three simultaneously to amplify …" | **L174**: "The design exploits all three simultaneously to amplify both the reconstruction error and the Teacher–Student discrepancy at anomalous regions (Section 4.3)." — 재검사자 권고 문안 그대로; 구 문형 0건 | **적용** |
| R2-M2 | §A.1 baseline 열거 dash-pair 파싱 붕괴 → 내부 분해 괄호 + 외부 semicolon 복원 | **L590**: "… nine detectors adopted from the protocol study of \cite{sarfraz2024quovadis} (five simple detectors: …; three lightweight neural detectors: …; and a GCN-LSTM detector); six established deep MTSAD systems (…); and seven recent methods (…)" — 권고 구조 그대로; \cite 보존 | **적용** |
| R2-M3 | Abstract introduce/introducing echo → "thereby exposing" 복원 | **L117**: "… we introduce a contaminated benchmark protocol …, thereby exposing labeled anomalies …" — "thereby introducing" 0건, 문장 내 introduce 1회만 | **적용** |
| R2-m1 | labelled → labeled 통일 | "labelled" **0건** (L434 "labeled (oracle)", L591 'labeled "DAGMM (simplified)"' — 신규분과 v2 기존분 모두 통일) | **적용** |

**신규 문제 도입 점검 (해당 4문장 한정, 1회)**: M1·M3은 재검사자 제안 문안 문자 그대로 채택(자작 변형 없음), M2는 제안 구두점 구조 그대로 — 괄호/semicolon 중첩이 정상 파싱되고 외부 3항(nine/six/seven) 경계 명확, 인용·고유명사 무변경. m1은 철자만 변경. **신규 어색함·의미 변화·마커 영향 없음.**

## 2. 마커 무손상 재확인 (frontmatter 제외 본문, 기계 계수)

| 항목 | recheck r2 §3 기준치 | 본 게이트 실측 | 판정 |
|---|---|---|---|
| `\cite`/`\citet` 명령 / key 연인원 | 89 / 131 | 89 / 131 | 일치 |
| PH:NUM | 31 (고유 ID 31) | 31 / 고유 31 | 일치 |
| PH:TXT / PH:FIG / PH:TAB / PH:ALG | 4 / 5 (1,2,3,4,B1) / 11 / 1 (C1) | 4 / 5 (동일 ID) / 11 (동일 ID) / 1 (ALG-C1) | 일치 |
| `[X.XX]` / `[N]` | 20 / 13 | 20 / 13 | 일치 |
| 수식 `\tag` / `$$` 블록 | 11 (1–6, C.1–C.5) / 11 | 11 (동일 번호·순서) / 11 | 일치 |

→ **touch-up이 마커·인용·수식에 영향 없음 확정.**

## 3. Directive 근거 확정 (7종 — 산출물 포인터 + 근거 문자열)

| Directive | 근거 문자열 (1줄) |
|---|---|
| **T6** (문장 단위 철저 검증) | 검사 4종 전수 수행 — `06_style_audit/`: AI_PHRASING_LEDGER(52 entries, MUST 11), STYLE_AUDIT_A("sentences_inspected: 214", 88 entries), STYLE_AUDIT_B(67 findings, "Independent audit; … not consulted"), TERMINOLOGY_AUDIT(18 issues) + 재검사 `p6_recheck_r2.md` method "101 hunks, 229 changed v3 sentences, 전수 검토" verdict PASS. |
| **R4** (AI-티 검출, corpus 판정) | `AI_PHRASING_LEDGER.md` frontmatter "corpus: paper/02_venue_study/SENTENCE_CORPUS.md" + §I "Applicable Prohibition Patterns (corpus-derived)"; 금지 패턴 잔존 0 = `p6_plagiarism_regression_r1.md` Pass 5 "delve/pivotal/seamlessly/… ABSENT … No AI-phrasing regressions detected". |
| **R5** (notation 최종) | `TERMINOLOGY_AUDIT.md` §4 Notation Audit → v3 반영 실측: `d_\text{model}` 0건·`d_{\mathrm{model}}` 통일, Table C.2 +4행(r̄/d̄, ε=10⁻⁴, c=4) — `p6_truth_spot_r1.md` C6 "신규 행 수치 정본 일치"(CONFIG_TRUTH ε=1e-4, score_recon_disc_ratio=4.0) + recheck §1.4 Eq.4/Eq.5/Table A.1 교차 일치. |
| **R15** (불필요 신규 축약어 0) | `TERMINOLOGY_AUDIT.md` §5 "Defined abbreviations — full inventory"(약어 전수표) — 유일 지적 TSAD(미정의, MEDIUM)은 해소: recheck §1.4 "bare TSAD 0건; MTSAD 통일 + 일반-분야 주장 2곳 spell-out"; 그 외 전부 justified/standard → **신규 불필요 축약어 0**. |
| **R24** (내부 용어·변수명 본문 0) | `TERMINOLOGY_AUDIT.md` §2 VERDICT(Q1/Q3 leak, HIGH) → 본 게이트 직접 grep: v3 본문 `\bQ[13]\b` **0건**(frontmatter 메타데이터만 잔존); `p6_truth_spot_r1.md` §2-① 11개 출현 전수 정방향 치환("anomaly-excised"/"contaminated-training", 반전 0); excl22는 최초 사용 L395 정의 후행. |
| **R35** (지엽 생략) | `TERMINOLOGY_AUDIT.md` §8 Overly Granular Prose 2건(전부 LOW·flag-only) — `p6_style_fixlog_r1.md` 처리: §A.1 SWaT note 증거-선행 재배열(PARTIAL), §A.3 boundary-aware KEPT(audit 원문 "no removal recommended" — 재현성 필수 공시) → **잔존 actionable 0**. |
| **A2** (표절 회귀 0) | `p6_plagiarism_regression_r1.md` 5-pass 결과 "TOTAL **0 regressions** BLOCKER 0 / MAJOR 0 / MINOR 0 … verdict: PASSED" + Phase 5 MAJOR 4건(F1/F2/F3/SC-06) "confirmed fixed and maintained". |

## 4. 게이트 조건 판정

**조건**: 3종 검사 잔존 지적 0 (waive는 MINOR만, DECISION_LOG 기록) + 회귀 검사 2종 통과.

1. **3종 검사 잔존 지적**: MUST급 전건 실해소(recheck §1.1–1.4: AI 11/11 + STYLE_A 17/17 + STYLE_B 3/3 + TERM 전건) + 재검사 MAJOR 3건·철자 1건은 본 게이트 §1에서 **적용 실측 확인** → 잔존 0. MINOR 4건(R2-m2~m5)은 **D-011**(`00_admin/DECISION_LOG.md` L21, 2026-06-11)로 Phase 7 polish 이월 waive — MINOR-only 규정(§5.3) 준수. → **충족**
2. **회귀 검사 2종**: (a) plagiarism-guardian 재가동 — PASSED, 회귀 0 (§3 A2 행); (b) method-truth spot — `p6_truth_spot_r1.md` "PASS — BLOCKER 0 · MAJOR 0 · MINOR 3 (전부 기록성; 수정 불요)", 의미 반전·정본 모순 0. → **충족**

### 최종 판정: **PASS** — Phase 6 게이트 통과. MANUSCRIPT_v3.md(touch-up 적용본)는 Phase 7 진행 가능.
