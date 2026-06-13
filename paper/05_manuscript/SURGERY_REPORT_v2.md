---
phase: 5
agent: budget-surgeon + appendix-drafter (r1) / budget-surgeon r2 — 표적 축소 (§8)
version: v2-r2
directives: [T5, R6, R7, R35, D-009, D-010]
last_modified: 2026-06-11
inputs:
  - 05_manuscript/MANUSCRIPT_v1.md + PLACEHOLDER_REGISTRY.md (v1) + INTEGRATION_REPORT_v1.md
  - 03_blueprint/PAPER_BLUEPRINT.md §8 + PAGE_BUDGET.md §5
  - 01_research_understanding/271_CONFIG_TRUTH.md (r4), EXPERIMENT_PROTOCOL_TRUTH.md (r3)
  - (r2) D-010 승인 항목 (본 문서 §4 사다리 1/4/5 + §3 tighten 재가)
outputs:
  - 05_manuscript/MANUSCRIPT_v2_draft.md (r1: 수술 + Appendix A/B/C 초안; r2: D-010 표적 축소, rev v2-draft-r2)
  - 05_manuscript/PLACEHOLDER_REGISTRY.md (r1: v2 갱신; r2: v2-r2 갱신)
  - 본 문서 (§1–7 r1; §8 r2 추가)
---

# SURGERY REPORT — MANUSCRIPT v2 (D-009 Appendix 이관 수술)

## 1. 작업 범위 요약

v1을 복사해 D-009 이관 수술 적용: ① §3 보조 수식 이관 (본문 display 수식 12 → **6**, 7±1 충족; 재번호 (1)–(6)), ② §4 지표 형식 정의·excl22 유도 세부·구현/baseline 세부·TAB-1 per-entity 세부 이관 (이관부마다 본문 1문장 포인터), ③ Appendix A(설정·전수 결과)/B(보조 분석)/C(방법 세부) 초안 작성 — 블루프린트 §8 잔여 항목 포함, 전 appendix float R3 규약 (placeholder + 완성 캡션 + 내용 명세, registry §7), ④ 분량 재계산 후 ±10% 미달 확인 → **추가 압축 1회 (R35 지엽 위주) 적용 완료**, ⑤ registry v2 갱신.
수치 창작 0건 (A8): appendix 실수치 표는 전부 271_CONFIG_TRUTH r4 / EXPERIMENT_PROTOCOL_TRUTH r3에서 전사; 실험 결과 셀은 전부 placeholder 유지.

## 2. 이관 목록 (원위치 → 새위치)

### §3 → Appendix C.1 (D-009 ②)

| # | 항목 | 원위치 (v1) | 새위치 (v2) | 본문 잔류물 |
|---|---|---|---|---|
| 1 | λ_rev sigmoid schedule 식 | §3.4 Eq (4) | §C.1 Eq (C.1) | 일반 서술 (Ganin schedule, ≈0.02→≈1, student-phase) + 포인터 |
| 2 | GRL backward 항등식 | §3.5 Eq (8) | §C.1 Eq (C.2) | "identity forward / negated, λ_rev-scaled backward" 산문 + 합성 gradient −λ_rev·λ_GRL 서술 |
| 3 | focal 변형 정확식 + p_t 정의 | §3.5 Eq (7) + 2문장 | §C.1 Eq (C.3) | 일반 서술 + 표준 focal과의 차이 1문장 (ADV MAJ-004 충족) + 포인터 |
| 4 | argtopk 마스킹 선택식 | §3.3 Eq (2) | §C.1 Eq (C.5) | π_i inline + 산문 선택 규칙 |
| 5 | stop-gradient 식 | §3.4 Eq (3) | §3.4 **inline 수학** (이관 아님) | — |
| 6 | patch embedding 식 | §3.3 Eq (1) | §3.3 **inline 수학** (추가 압축 패스) | — |
| 7 | adaptive λ grad-ratio 규칙 | §3.4/3.5 산문 (식 없음) | §C.1 Eq (C.4)로 **정식화 (신규)** — β_GRL=0.2, β_FM=1.0, clamp [0,10] (271_CONFIG_TRUTH r4) | 기존 일반 서술 유지 |

본문 잔류 수식 (재번호): (1) L_OD, (2) L_FM, (3) L_total, (4) adaptive scaling, (5) σ_i, (6) s_t. Appendix 수식: (C.1)–(C.5).

### §4 → Appendix (D-009 ①)

| # | 항목 | 원위치 (v1) | 새위치 (v2) | 본문 잔류물 (의무) |
|---|---|---|---|---|
| 8 | 지표 형식 정의·계산 세부 (PA/PA%K formal, per-K re-opt, VUS tolerance·구현, Affiliation 정식, AR threshold 정식, PA F1 oracle, 점수 집계) | §4.1.3 | §A.2 | 5지표 1절 요약 + **상호보완성 문단 (R29)** + **PA-F1 비판·주지표 불참고 (R29)** + threshold 방어는 §4.1.2 잔류 (R30) |
| 9 | excl22 수치 유도 세부 (region 좌표 [2,869, 38,769), 35,900 pts, 15.96%, 결정적 식별 규칙, eval-mask 구현, 독립 best-epoch) | §4.1.1 | §A.4 | **핵심 근거 (R28)**: 83.75% 질량 → 단일 사건 지배 → dual 보고·excl22 랭킹·mask-only·전 baseline 동일 |
| 10 | 구현 세부 (optimizer/LR schedule/bf16/GRL lr/dropout/ffn/nhead, SWaT 45-feature 재현성 주의) | §4.1.2 | §A.1 (Table A.1 + prose) | 핵심 상수 1문장 (patch 10/window 500/4L–3L–2L/d512/500ep/250 warmup/batch 1024/seed 42 단일 run) + **epoch 비대칭 공개 (의무)** |
| 11 | Hardware/code 문장 (TXT-001, TXT-002) | §4.1.2 | §A.1 Environment | — (registry §4 갱신) |
| 12 | baseline 세부 (simple 5/neural 3/GCN-LSTM 열거, DAGMM simplified variant 주기, 원구현 충실·random 5-run) | §4.1.4 | §A.1 + TAB-A3 | 계층 요약 + 인용 클러스터 (lai2023npsr·luo2024moderntcn 등 인용 보존) + **R31 공정성·양적 비대칭 전체 잔류** |
| 13 | TAB-1 per-entity 세부 (WaDi A1/A2 분리 행, SMD per-machine) | §4.1.1 Table 1 (7행) | §A.3 Table A.4 (per-entity; SMD 셀 placeholder) | TAB-1 = **6계열 요약** (registry 스펙 개정) |
| 14 | SMAP/MSL 경계 이동 실측 세부 (7.58% 등 채널별 수치) | §4.1.1 | §A.3 Table A.5 (실측 4채널 표) | 메커니즘 + "4/81 채널, 최대 +166 steps" 1문장 (프로토콜 투명성) |

### 신규 작성 (이관 아님 — 블루프린트 §8 + 임무 지정)

§A.3 (train-label semantics, boundary windowing), §A.5 (전수 다지표, TAB-A7), §A.6 (per-entity 결과, TAB-A8), §B.1 (Q1 전체 비교, **TAB-B1 + 완성 캡션**), §B.2 (epoch-sensitivity placeholder, TAB-B2), §B.3 (계산 비용, TAB-B3 + **NUM-031** — §5 "approximately 50×" sync 조건), §B.4 (parameter sensitivity, FIG-B1), §B.5 (확장·conditional ablation 수용처, TAB-B4), §C.2 (입력 차원 표, 실수치), §C.3 (pseudocode, ALG-C1), §C.4 (notation 표, 실수치).

### 본문 cross-reference 재지정 (v1 → v2)

구 §A.2(Q1)→§B.1 / 구 §A.3(전수 지표)→§A.5 / 구 §A.4(per-entity)→§A.6 / 구 §A.5(SWaT full)→§A.4 / 구 §B.1(conditional ablation)→§B.5 / 구 §B.4(epoch sens.)→§B.2 / 구 §C.1(입력 차원)→§C.2. §A.1·§B.3은 번호 유지. 스캔 검증: 본문·appendix의 모든 "Appendix §X.Y" 참조가 실재 절에 매핑 (orphan 0).

## 3. 압축 통계 (동일 스크립트: prose 단어, 주석·display 수식·float 마커·표 제외)

| 섹션 | v1 단어 (수식) | v2 단어 (수식) | Δ |
|---|---:|---:|---:|
| §1 | 750 | 750 | 0 (무변경) |
| §2 | 736 | 736 | 0 (무변경) |
| §3 | 1,523 (12 eq) | 1,464 (6 eq) | −59 w, **−6 eq** |
| §4 | 2,288 | 1,907 | **−381 w (−17%)** |
| §5 | 222 | 222 | 0 (무변경) |
| **본문 계** | **5,519** | **5,079** | **−440 (−8.0%)** |
| Appendix (신규) | — | 1,658 (+5 eq) + 실수치 표 6 + placeholder float 10 | — |

참고: §4 이관 절대량은 ~600–700 단어 상당이나, D-009가 요구한 본문 1문장 포인터 9개(+신규 연결부)가 부분 상쇄. §4.3/§4.5의 [Conditional]/[Note] 괄호는 인쇄되지 않는 HTML 주석으로 전환 (제작 노트이며 원고 산문 아님 — 정보 무손실).

## 4. 분량 재계산 (R6 — INTEGRATION_REPORT 방식 그대로: 675 w/p, 수식 0.05p, float = PAGE_BUDGET §3 사양, 헤더 §3 0.18p/§4 0.20p)

스크립트 기준선: v1 합계 11.99p (INTEGRATION 수기 11.8p와 +0.2p 차이 — 동일 방식 내 일관 비교를 위해 스크립트 수치로 양쪽 산정).

| 섹션 | v1 | **v2 (수술+추가 압축 1회 후)** | 예산 | Δ vs 예산 | 판정 |
|---|---:|---:|---:|---:|---|
| §1 Introduction | 1.56 | **1.56** | 1.6 | −0.04 | ✓ |
| §2 Related Work | 1.18 | **1.18** | 1.1 | +0.08 | ✓ |
| §3 Methodology | 3.44 | **3.05** (text 2.17 + eq 0.30 + Fig.2 0.40 + hdr 0.18) | 2.7 | **+0.35** | ✗ |
| §4 Experiments | 5.45 | **4.86** (text 2.83 + floats 1.83 + hdr 0.20) | 3.3 | **+1.56** | ✗ |
| §5 Conclusion | 0.36 | **0.36** | 0.3 | +0.06 | ✓ |
| **본문 합계** | **11.99** | **11.00** | **9.0 (허용 8.1–9.9)** | **+2.00** | **미달성** |

(§4 float 내역: TAB-1 0.25 [6행 축소] + TAB-2 0.50 [landscape] + TAB-4 0.20 + TAB-3 0.25 + FIG-3 0.33 + FIG-4 0.30 = 1.83p. Appendix 예상 볼륨: prose 2.46p + 실수치 표 ~1.3p + 수식 0.25p + placeholder float 충전 시 ~4.4p ≈ **8p 내외** — 본문 9p 카운트 외이나 PAGE_BUDGET §5 추정(4–5p)을 상회함을 명기; D-009 이관분이 가산된 결과.)

### 정직 판정과 원인 분석

**추가 압축 1회 적용 후에도 9.0p ±10% 미달성 (11.0p, +2.0p).** 순감 −1.0p (v1 대비 −8.3%). 원인은 구조적이며 INTEGRATION_REPORT §5의 결론("directive 의무 서술의 구조적 하한이 §8 단어 예산과 양립 불가")이 이관 후에도 유지됨:

1. **§4 산술 하한**: floats 1.83p + 헤더 0.20p = 2.03p가 예산 3.3p의 62%를 선점 → 잔여 텍스트 예산 857단어. 그러나 D-009 불가침 의무 서술(아래 §6 체크리스트 전 항목)을 모두 담은 §4 prose 실측 하한이 ≈1,900단어 (모든 비의무 세부는 이미 이관·압축됨 — 잔여 문장은 의무 공개/방어, placeholder 주장문, 1문장 포인터뿐).
2. **§3 하한**: GRL 필요성 논증 + dual-λ + warmup 공개 + capacity-gap 논증 + R10 다변량 동기 + 수식 6개 유지 시 ~1,460단어.
3. 이관 자체의 상쇄: 이관부마다 의무화된 본문 포인터 1문장(9개)이 절감분을 부분 잠식.

### 잔여 해소 사다리 (정량 — 적용 권한 구분)

| # | 수단 | 절감 | 권한 |
|---|---|---:|---|
| 1 | Table 4를 Table 2 하단 블록으로 흡수 (PAGE_BUDGET 전략 2) | −0.15p | 재가 불요 (Phase 7 조판 시; 가독성 우선이면 보류) |
| 2 | Phase 7 LaTeX 1차 조판 실측 — 단어환산은 ±10% 오차 (텍스트 질량 ~7.6p 기준 ±0.76p) | ±0.76p | 자동 (실측 선행 의무) |
| 3 | PAGE_BUDGET 섹션 재배분 (§3 2.7→3.0, §4 3.3→4.4; Appendix 상쇄) — INTEGRATION §5-4와 동일 안 | +격차 해소 | **재가 필요** (ADV BLK-001: blueprint-reviser 경유) |
| 4 | 의무 서술 범위 재해석 추가 이관 (예: §4.1.3 5지표 나열 전체 §A.2 이관, 상호보완성 문단만 잔류; §4.1.1 precedent 문장 이관) | −0.3~−0.4p | **재가 필요** (D-009 현행 문언상 불가 — orchestrator 재결정) |
| 5 | Table 3 conditional rows 5/7 미완 확정 시 §B.5 강등 (행+문단 2개) | −0.15p | 실험 결과 의존 (Phase 6) |
| 6 | Table 2 landscape 미지원 판명 시 **역방향 위험** (RT V1 fallback) | +0.2p | Phase 7 플래그 유지 |

사다리 1+4+5 동시 적용 시 ≈ −0.7p → 10.3p; 9.9p 진입에는 #2 실측 호전 또는 #3 재배분이 추가로 필요. **#3(예산 재배분)이 근본 해법**임을 권고 — 의무 서술 불가침 전제에서 9.0p는 산술적으로 불능.

## 5. 발견 사실 플래그 (Phase 6 검증 인계 — 수술 중 발견)

1. **[모순 — 본문 정정] test stride**: v1 §4.1.2 "test stride 1"은 정본 271_CONFIG_TRUTH **r4**의 `resolve_test_stride = W//10 − 1 = 49`와 모순 (코드 재확인: `mae_anomaly/utils/experiment.py:20–42`; EXPERIMENT_PROTOCOL_TRUTH r3 §④-2의 "test stride=1" 문구가 stale — 같은 문서가 window/patch 정본을 271_CONFIG_TRUTH로 위임). 조치: 본문에서 stride 주장 제거 (의미 불변 — leave-one-out 서술은 stride 무관), 정본값 49는 Table A.1에만 기재 + 주석 플래그. **두 정본 문서 간 모순이므로 Phase 6에서 EXPERIMENT_PROTOCOL_TRUTH r4 정정 필요.**
2. **[표기 모순 — 본문 유지] PA%K grid**: 271_CONFIG_TRUTH r4 §VIII "k=0 to k=100 in steps of 1"은 부정확 — 코드 정본 `evaluator.py:831 PA_K_VALUES = list(range(0, 101, 5))` (직접 확인). 본문·§A.2의 {0, 5, …, 100} 유지가 옳음. 정본 문서 정정 인계.
3. 작업트리의 `mae_anomaly/utils/experiment.py` 미커밋 diff는 d_model 후보 목록(768 추가) 변경으로 271 고정 d_model=512와 무관 — 원고 영향 없음.

## 6. 의무 서술 보존 체크리스트 (D-009 ③ — 삭제 0건, 전부 본문 잔류·압축만)

| 의무 항목 | v2 본문 위치 | 상태 |
|---|---|---|
| **R13** 동기-우선 프로토콜 서술 (원본 split 구조적 무라벨 → re-split 필연 + 50% 규칙 + no-lookahead + redefinition 논증 + NRdetector 선례 + 한계 인정) | §4.1.1 "Contaminated benchmark protocol" 문단 전체 | ✅ 보존 (자구 압축만) |
| **R29** 5지표 상호보완성 개념 | §4.1.3 "The five metrics span three orthogonal perspectives … going undetected" | ✅ 보존 (원문 그대로) |
| **R29** PA-F1 비판 + 주지표 불참고 명시 | §4.1.3 말미 "(oracle) … never used for ranking: even a random score …" | ✅ 보존 |
| **R30** threshold 방어 ((1−r) quantile 정의 + 전 모델 동일 + 학습 불개입 + threshold-free 지표 무관) | §4.1.2 "Inference and threshold" | ✅ 보존 (원문 사실상 그대로) |
| **R28** excl22 핵심 설명 (83.75% 단일 사건 지배 → dual 보고 + excl22 랭킹 + 3.68 vs 19.05 + eval-mask only + 전 baseline 동일) | §4.1.1 "SWaT dual evaluation" | ✅ 보존 (유도 세부만 §A.4) |
| **R31** 공정성 방어 (Q3 = unsupervised의 최선 + 동일 split/평가) + 양적 비대칭 인정 (0.52–6.20% 절제 + Table 4/§B.1 보완) | §4.1.4 "Comparison conditions" | ✅ 보존 |
| **R32** 강건성 논리 (3-성질 (i)(ii)(iii) + 연속 감쇠 + NRdetector sweep 축 구분) | §4.4 "Why graceful degradation is expected" | ✅ 보존 (자구 압축만) |
| **공개 ①** epoch 비대칭 사실 (500/10/50 + eval 5 vs 1 + batch 1024 vs 512 + no early stopping + 수렴 특성 방어 + §B.2 민감도) | §4.1.2 "Epoch asymmetry disclosure" | ✅ 보존 (D-009 명시 본문 잔류) |
| **공개 ②** test-set model selection (전 모델 동일 + validation split 부재 + 낙관 편향 한계 인정) | §4.1.2 "Test-set model selection" | ✅ 보존 (원문 그대로) |
| GRL 필요성 논증 (R23 — bifurcation 너머 memorization 경로 차단) | §3.5 "Why gradient reversal is necessary beyond loss bifurcation" | ✅ 보존 (원문 그대로) |
| SDMAE 계층 구분 1문장 (target/loss space vs gradient space — R21 연계) | §3.5 첫 문장 | ✅ 보존 |
| Teacher-only warmup 공개 (contribution 아님 명시) | §3.4 "Teacher-only warmup" | ✅ 보존 (원문 그대로) |
| dual-λ 구조 (λ_GRL ↔ λ_rev 구분 — 단일 λ 합침 금지) | §3.4 (일반 서술) + §C.1 (정확식) | ✅ 보존 |
| focal 변형 차이 1문장 (ADV MAJ-004) + "본 설계" 명시 (ADV NOTE-002) | §3.5 1문장 + §C.1 (정확식·p_t 정의·설계 귀속) | ✅ 보존 |
| complementary masking "구현됐으나 미사용" 수식어 (ADV MAJ-012) | §5 (무변경) | ✅ 보존 |

## 7. Phase 6/7 인계

1. **분량 격차 +2.0p (최대 이슈)** — §4 사다리 (본 문서 §4) 중 #3(예산 재배분)·#4(의무 범위 재해석)는 orchestrator/blueprint-reviser 재가 필요. Phase 7 LaTeX 1차 조판 실측이 모든 후속 조정의 선행 조건.
2. Appendix placeholder 10건 (TAB-A3/A6/A7/A8, TAB-B1–B4, FIG-B1, ALG-C1) + NUM-031 — registry §7 명세대로 충전. TAB-A3는 comparison 파이프라인 설정에서 **전사**(발명 금지).
3. Table A.4의 SMD per-machine 셀 (부분 placeholder) — per-machine 통계 추출 후 충전.
4. test stride·PA%K grid 정본 문서 모순 2건 정정 (본 문서 §5).
5. v1 인계 항목 승계: NUM sync group A/B, conditional ablation rows (이제 §B.5 강등 규칙), §2.2 "to our knowledge" 스코핑, [^sd-fn] 각주 LaTeX 처리, TXT-001 metadata 충전 (§A.1로 위치 이동), 미사용 bib key 5건.
6. Appendix 볼륨 ~8p (충전 후) — PAGE_BUDGET §5 추정(4–5p) 상회. supplementary 처리 가능 범위이나 Phase 7에서 저널 규정 확인 권장.

---

## 8. 2차 표적 축소 (D-010 — v2-draft-r2, 2026-06-11)

D-010 승인 4개 항목만 정밀 적용 (그 외 변경 0건). §1–7은 r1(D-009) 기록으로 보존; 본 절이 r2의 전체 기록이다.

### 8.1 항목별 조치와 절감 실적

| # | D-010 항목 | 조치 | 승인 추정 | **실측 절감** |
|---|---|---|---:|---:|
| ① | Table 4 → Table 2 흡수 | `[TAB-4]` 마커 제거; TAB-2에 하단 protocol-effect row-group 병합 (캡션·행구조·standard-split run 의존성 registry TAB-2 항목으로 통합); §4.2 protocol-effect 서술 전량 유지, 표 참조 4곳 갱신 (§4.1.4, §4.2 ×2, §4.4) | −0.15p | **−0.13p** (float −0.20+0.05; 참조 어구 +11w = +0.02p) |
| ② | Conditional ablation rows 강등 | Table 3 → 확정 4행 (Full / w/o GRL / w/o 마스킹 / w/o OD); 구 5/6/7행(FM·warmup·symmetric)을 TAB-B4로, Row-5/7 문단 2개(NUM-024/025 포함)를 §B.5 산문으로 이동; 본문은 1문장 포인터("Extended variants.") | −0.15p | **−0.11p** (§4 prose −42w = −0.06p; TAB-3 0.25→0.20p) |
| ③ | §4.1 setup 압축 | (a) dataset 열거 산문 → TAB-1 위임 + 핵심 2문장 (도메인 괄호 세부·entity 수 → Table 1/§A.3); (b) per-dataset 라벨 의미 열거 → §A.3 포인터 (§A.3 "Training-label semantics"에 verbatim 중복 존재 — 정보 무손실); (c) baseline 가족별 열거 → 인용 클러스터 2문장 + §A.1 위임 포인터 (R19; 방법명 목록은 Table 2 행·§A.1 잔존) | −0.3~−0.4p | **−0.06p** (−38w) — 미달 원인 §8.4 |
| ④ | §3 tighten | 중복·우회 표현 압축 17개소 (§3.1 표기 압축, §3.2 개관 압축 + §3.6 중복 문장 1개 제거, §3.3 선택규칙 재서술·blind-spot 중복 절 압축 + mask-token 삽입 3중 서술→§3.4 단일화, §3.4 dual-λ·warmup·Teacher/Student 정의 압축, §3.5 focal·GRL 역학·OD/FM 압축, §3.6 leave-one-out·scaling 근거 압축) | −0.35p (~240w) | **−0.28p** (−190w) — 의무 하한, §8.4 |
| | **합계** | | **−0.95~−1.05p** | **−0.58p** |

부수 효과: Appendix §B.5 +90w (+0.13p) — 이동 수용분; 충전 후 Appendix 예상 ~8.1p (본문 카운트 외).

### 8.2 분량 재계산 (§4와 동일 산정 방식: 675 w/p, display 수식 0.05p, float PAGE_BUDGET §3 사양, 헤더 §1 0.05 / §2 0.09 / §3 0.18 / §4 0.20 / §5 0.03; 단어 = prose만, 주석·수식·마커·표 제외)

| 섹션 | v2 (r1) | **v2-r2** | 예산 | Δ vs 예산 | 판정 |
|---|---:|---:|---:|---:|---|
| §1 Introduction | 1.56 | **1.56** (무변경) | 1.6 | −0.04 | ✓ |
| §2 Related Work | 1.18 | **1.18** (무변경) | 1.1 | +0.08 | ✓ |
| §3 Methodology | 3.05 | **2.77** (text 1,274w=1.89 + eq 0.30 + Fig.2 0.40 + hdr 0.18) | 2.7 | **+0.07** | ≈✓ |
| §4 Experiments | 4.86 | **4.55** (text 1,838w=2.72 + floats 1.63 + hdr 0.20) | 3.3 | +1.25 | ✗ |
| §5 Conclusion | 0.36 | **0.36** (무변경) | 0.3 | +0.06 | ✓ |
| **본문 합계** | **11.00** | **10.42** | 9.0 / **D-010 목표 ≤10.0** | **+0.42 (목표 대비)** | **미달성 — §8.4** |

§4 float 내역 (r2): TAB-1 0.25 + TAB-2 0.55 (흡수 블록 +0.05 포함) + TAB-3 0.20 (4행) + FIG-3 0.33 + FIG-4 0.30 = **1.63p** (r1 1.83p, −0.20p).
단어 검산: §4 Δ = ① +11 ② −42 ③ −38 = −69w (스크립트 실측 일치); §3 Δ = −190w (1,464 → 1,274); body display 수식 6개 (1)–(6) 불변 확인 (tag 스캔).

### 8.3 의무 서술 보존 재확인 (삭제 0건 — §6 체크리스트 전 항목 재검)

| 의무 항목 | r2 상태 | r2 변화 내역 |
|---|---|---|
| **R13** 프로토콜 동기 (구조적 무라벨 → re-split 필연 + 50% 규칙 + no-lookahead + redefinition + NRdetector 선례 + 한계 인정) | ✅ 보존 | per-dataset 라벨 의미 열거만 §A.3 포인터로 위임 (§A.3에 동일 내용 verbatim 존재); 핵심 주장·나머지 요소 원문 그대로 |
| **R29** 5지표 상호보완성 문단 / **R29** PA-F1 비판·랭킹 불참고 | ✅ 보존 | 무변경 (§4.1.3 열거 문장 포함 전체 무변경 — D-010 문언 외 범위) |
| **R30** threshold 방어 | ✅ 보존 | 무변경 |
| **R28** excl22 핵심 설명 (83.75% → dual 보고 + excl22 랭킹 + mask-only + 전 baseline 동일) | ✅ 보존 | 무변경 |
| **R31** Q3 공정성 + 양적 비대칭 (0.52–6.20% + 보완 분석) | ✅ 보존 | 무변경 (표 참조만 "protocol-effect block of Table 2"로 갱신); 약지도 Q1-only 사유 문장 잔류 |
| **R32** 3-성질 강건성 논리 | ✅ 보존 | 무변경 (Table 4 참조 1곳만 갱신) |
| **공개 ①** epoch 비대칭 / **공개 ②** test-set model selection | ✅ 보존 | 무변경 (불가침 지정 항목) |
| **R23** GRL 필요성 논증 (§3.5) | ✅ 보존 | 원문 그대로 (verbatim) |
| **R21** SDMAE 계층 구분 1문장 (§3.5) + §2.3 각주 [^sd-fn] | ✅ 보존 | 원문 그대로 (불가침 지정 항목) |
| **R10** 다변량 동기 (§3.1 말미) | ✅ 보존 | 원문 그대로 (불가침 지정 항목) |
| Teacher-only warmup 공개 (contribution 아님 명시) | ✅ 보존 | 자구 압축만 ("the loss is computed on the Teacher branch alone"→"only the Teacher branch is trained"); 두 요소(고정 초기 epoch + not a contribution) 그대로 |
| dual-λ 구조 (λ_GRL ↔ λ_rev 구분) | ✅ 보존 | 자구 압축만; 두 양·adaptive 규칙·sigmoid schedule·≈0.02→≈1·§C.1 포인터 전부 잔류 |
| focal 변형 차이 1문장 (ADV MAJ-004) | ✅ 보존 | 자구 압축 ("unlike the standard focal loss, whose modulating factor derives from the raw prediction, here it derives from the class-prior-weighted cross-entropy itself"); 설계 귀속·정확식 §C.1 잔류 |
| complementary masking "구현됐으나 미사용" (§5, ADV MAJ-012) | ✅ 보존 | 무변경 |
| 수식 6개 (1)–(6) | ✅ 보존 | 무변경 (tag 스캔 검증) |

비고: §3.2 말미 "At inference the GRL branch is inactive; scores derive from …" 문장은 §3.6와의 중복으로 제거 — training-only 공개는 §3.2 첫 문장("training-only label-guided module")과 Fig. 2 캡션("training only" 명시)에 잔존, GRL 추론 미사용 사실은 §3.6에 잔존 (정보 무손실).

### 8.4 정직 판정 — D-010 목표 ≤10.0p 대비 +0.42p 잔여

1. **③의 추정-실측 격차가 주원인** (−0.3~0.4p 추정 vs −0.06p 실측). 사다리 #4의 −0.3~0.4p는 "§4.1.3 5지표 열거 전체 §A.2 이관 + §4.1.1 precedent 문장 이관"을 전제한 수치였으나, D-010 최종 승인 문언은 baseline/dataset 열거 2건만 지정 (R29/R13 보호로 해석) — 1차 수술 후 해당 2건에 남은 산문 슬랙은 ~100단어(실측 −38w 순절감)에 불과.
2. **④는 의무 하한 도달** (−190w/−240w): R10·R21·R23 원문 유지 + dual-λ/warmup/capacity-gap/focal 의무 요소 + 수식 6개 전제에서, 잔여 §3 문장은 정의·논증·공개뿐 — 240단어 도달은 의무 서술 침범 없이는 불가 (1차 보고 §4-2의 §3 하한 분석과 일치).
3. **잔여 +0.42p 처리 (D-010 ⑤)**: Phase 7 LaTeX 1차 조판 실측이 선행 (단어환산 ±10% ≈ ±0.76p — 잔여가 오차범위 내). 실측 후에도 초과 시 잔여 수단: (a) 사다리 #3 PAGE_BUDGET 재배분 (blueprint-reviser 재가 — 근본 해법 권고 유지), (b) §4.1.3 지표 열거의 자구 압축 또는 §A.2 이관 (orchestrator 재결정 필요; R29 의무 문단 2개는 어느 경우든 본문 잔류 가능).

### 8.5 registry/문서 동기화 (D-010 ⑥)

- **PLACEHOLDER_REGISTRY v2-r2**: TAB-2 병합 스펙 (캡션 병합, 하단 블록 행구조, 0.55p, fallback (b) 소진 표기, standard-split 의존성 승계) / TAB-4 → ABSORBED 감사 항목 (본문 마커 0) / TAB-3 4행·0.20p·강등 노트 / TAB-B4 확정 호스트 캡션 / NUM-014..019 위치 재지정, NUM-024/025 → §B.5 / v2-r2 완전성 스캔: NUM 31/31, TXT 2 ID(4 occ), 본문 TAB 3 + appendix TAB 8, FIG 4+1, ALG 1 — orphan 0.
- **원고 frontmatter**: rev `v2-draft-r2`, D-010 notes 추가. r1 frontmatter의 stale 표기 "display equations 12 → 7 / (1)–(7)"을 6 / (1)–(6)으로 정정 (본 보고 §2 기록·실제 tag와 일치 — 단순 표기 오류였음).

### 8.6 Phase 6/7 인계 (r2 추가분 — §7 승계 위에)

1. §7-1 분량 격차 갱신: +2.0p → **+1.42p** (9.0p 예산 기준) / **+0.42p** (D-010 목표 10.0p 기준). Phase 7 실측 선행 의무 불변.
2. TAB-2 landscape 실패 시 fallback은 (c) 단일 지표 열만 잔존 — (b)는 본 수술로 소진; 역방향 위험은 흡수 블록 포함 +0.2p 이상으로 상향.
3. 구 Table 3 rows 5–7의 "run 미완 시 강등" 조건부 규칙은 해소 (상시 §B.5) — 단, NUM-024 미충전 시 contribution bullet 3을 설계 원리로 완화하는 Phase 6 규칙은 유지 (TAB-3 registry 노트).
4. §B.5에 이동한 NUM-024/025 문단의 결과 충전은 기존 ablation 큐와 동일 경로.
