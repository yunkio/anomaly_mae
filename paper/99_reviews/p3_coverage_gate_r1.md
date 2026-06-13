---
phase: 3
agent: coverage-auditor
directives: [M10]
last_modified: 2026-06-11
inputs:
  - paper/MASTER_ORCHESTRATION_PROMPT.md §9 (Phase 3 매핑: T3, R1, R2, R5, R6, R7, R8, R9, R10, R11, R15, R16, R19, R20, R21, R22, R32 — §9.4 표와 17종 일치 확인)
  - paper/03_blueprint/PAPER_BLUEPRINT.md (r3)
  - paper/03_blueprint/PAGE_BUDGET.md (r3)
  - paper/99_reviews/p3_rereview_redteam_r2.md, p3_rereview_adversarial_r2.md, p3_fixlog_r3.md
  - paper/01_research_understanding/271_CONFIG_TRUTH.md (r4), CODEBASE_UNDERSTANDING.md (r4), RESEARCH_SYNTHESIS.md (r3)
  - paper/00_admin/DECISION_LOG.md (D-005~D-007)
verification_basis: |
  코드 1차 소스 직접 재실측 (mae_anomaly/, read-only; 2026-06-11): trainer.py 1195–1215 / 740–770 /
  185–195 / 520–540 / 635–655 / 1310–1322, model.py 125–142 / 1110–1130 / 1145–1170,
  loss.py 185–220 / 288–305, evaluator.py 806–816 + grep 전수(`_grl_lambda` 대입/소비처).
  271 metadata 독립 재추출: results/experiments/271_20260602_020545_271canon_baseline/PSM/
  experiment_metadata.json (use_grl=True, grl_adaptive_lambda=True, grl_loss_weight=0.2,
  fm_adaptive_lambda=True, fm_loss_weight=1.0, teacher_only_warmup_epochs=250, num_epochs=500,
  grl_target_mode='window', d_model=512, dim_feedforward=2048, grl_disable_anomaly_loss=True
  — fixlog r3 verification_basis와 전건 일치). 정본 3종 grep 교차 대조.
verdict: PASS (BLOCKER 0 / MAJOR 0 / 미마감 발견 0 / Directive 17/17 근거 확보; NOTE 3 — 비차단)
---

# Phase 3 Coverage Gate (r1) — 블루프린트 r3 + PAGE_BUDGET r3

## 1. r3 수정분 spot 재검증 (코드 1차 소스, 6건 — 요구 5건 초과)

### Spot ① — GRL 이중 λ 구조 (NEW-B1 수정의 사실 검증) — **일치 (PASS)**

코드 직접 재실측 (2026-06-11):
- `trainer.py:1201` 주석 "GRL lambda: set BEFORE train_epoch", `:1202` 게이트 `if getattr(self.config, 'use_grl', False):` — **게이트는 use_grl뿐**, 271 metadata `use_grl=True` 본 감사 독립 재추출로 확인 → **271 활성**.
- `:1204–1207` `p = clip((epoch − 250 + 1)/250, 0, 1)` (`_student_total = max(500−250,1)=250`), `:1208–1209` warmup 중 `model._grl_lambda = 0.0`, `:1211` `2.0/(1.0 + math.exp(−10.0·_p)) − 1.0`. 수치 재계산: 0-based epoch 250에서 p=0.004 → λ_rev=tanh(0.02)≈**0.0200**, epoch 499에서 p=1.0 → ≈**0.99991** — 정본·블루프린트 기재값과 일치.
- `model._grl_lambda` **대입 지점 grep 전수 = trainer.py:1209/1211 뿐** (그 외 매치 :319/:1292/:1294는 history/로깅), **소비처 = model.py:1152–1153 단일** (`anomaly_classifier(student_hidden, lambda_grl)`).
- `model.py:129–139` `GradientReversalFunction`: forward `x.clone()`, backward `return −ctx.lambda_ · grad_output` — backward 곱셈 계수 = **λ_rev(sigmoid)** 맞음.
- 손실 가중치 경로: `trainer.py:746` elif 게이트(`use_grl and not teacher_only and 'grl_cls_loss' in loss_tensors`), `:749` `grl_loss_weight`(metadata 0.2), `:760` `(‖∇L_main‖/(‖∇L_GRL‖+1e-4)).clamp(0.0, 10.0)`, `:762–763` `_grl_effective = _prev_epoch_grl_lambda × _grl_w; loss += _grl_effective × _grl_cls_loss` — **×0.2 실재, 직전 epoch 값 적용** (`:190` 초기값 1.0, `:1317–1319` prev-epoch 갱신) — 전부 확인.
- FM 경로 `trainer.py:639–653` 직접 확인: grad-ratio clamp[0,10] × `fm_loss_weight`(1.0), prev-epoch — **model-level sigmoid/ramp 부재** → "sigmoid는 GRL 반전 계수 전용" 주장 정확.

→ 271_CONFIG_TRUTH r4 §VIII(이중 λ 구조 / Reversal coefficient λ_rev / Student-hidden 도달 gradient 3행 신설, Lambda balancing 행 "손실 가중치 λ_GRL" 명칭 명확화), 블루프린트 r3 §5.5 이원 서술·§5.6(C) λ_rev 계수 정정·§9.1 λ_rev 행·§9.2 교체 조항·§15 GRL 행 — **전건 코드와 일치, 불일치 0**.

### Spot ② — warmup 중 student forward skip (NEW-B2 수정의 사실 검증) — **일치 (PASS)**

- `trainer.py:526–535`: 주석 "2026-05-29: propagate teacher_only so model can skip student decoder / GRL classifier / SCAD head forward during warmup" + "Evaluator and visualizer paths leave teacher_only at default False" → `self.model(..., teacher_only=teacher_only)` 호출 확인.
- `model.py:1119`: `if self.config.use_student and self.student_decoder is not None and not teacher_only:` — 학습 경로 warmup 중 student forward **자체 생략** 확인.
- `loss.py:193` `if student_output is None:` sentinel + `:213` `if self.use_discrepancy and not teacher_only:` 이중 방어 확인.
- 시점 정합: 코드 주석 2026-05-29 변경, 271 실행 디렉토리 timestamp 2026-06-02 → "271 실행 이전 반영" 서술 정확. student 학습 개시 = 0-based epoch 250.

→ 블루프린트 r3 §5.5의 역전 교체 서술("학습 경로 forward 자체 생략 / 평가 경로 full forward / 'forward 수행+gradient 차단' 서술 금지")·capacity-gap 충돌 없음 재점검, 정본 3종 Training/표A 행 — **전건 코드와 일치**.

### Spot ③ — Table 4 실행 사양의 `loss.py:293–302` 인용 정확성 — **정확 (PASS)**

직접 확인: `loss.py:293` `_pos_count = valid_targets.sum()`, `:294` `if _pos_count == 0:`, `:295` 주석 "No anomaly in this batch → skip GRL loss", `:296–302` `_grl_results = {'grl_cls_loss_tensor': None, ...}` (닫는 중괄호 :302). 블루프린트 §6.6 실행 사양의 인용 범위·내용("`loss.py:293–302` `_pos_count == 0 → grl_cls_loss_tensor=None` skip") **정확**. use_grl=True 유지 + baseline=Q3 명시도 §6.6에 반영 확인.

### Spot ④ — 정본 3종 간 GRL·warmup 서술 모순 0 (grep 교차) — **모순 0 (PASS)**

`λ_rev|sigmoid|forward.*skip|frozen|teacher_only` grep 전수:
- **271_CONFIG_TRUTH r4**: §VIII Training(warmup forward-skip + 학습 개시 epoch + 평가 경로 구분), Student-loss 행("ramp 없음"을 손실 항 한정으로 정밀화 + λ_rev 별도 ramp 병기), Loss Components GRL 행(`-lambda × grad`의 lambda = λ_rev 명시), GRL Details(이중 λ/λ_rev/도달 gradient 3행) — 일관.
- **CODEBASE_UNDERSTANDING r4**: §1 GRL bullet(λ_GRL "손실 항 가중치" 주의 + λ_rev bullet :106–107), §2.5(:225 anomaly-ramp 271 no-op + 이원 서술), §2.6(:254 "별개 메커니즘 — adaptive λ 3경로와 독립"), §5.3 구역(:403 λ_rev warmup 0.0) — 일관.
- **RESEARCH_SYNTHESIS r3**: §③-3(:57 λ_rev 병기), 표A warmup 행(:95 forward-skip + 이원 서술 + anomaly-ramp 삭제), 표A GRL 행(:97 −λ_rev×grad + 정확 공식 + 이중 λ), 정정 이력(:319–320) — 일관.
- 세 문서 간 공식·수치(2/(1+e^{−10p})−1, p 정의, 0.020→0.9999, ×0.2, file:line) **상호 모순 0**. (정본 외 NOTION_DIGEST 잔존 stale 1건은 §5 NOTE-2.)

### Spot ⑤ — R2-MAJ-01 처리의 3개소 일관성 — **일관 (PASS)**

- 블루프린트 §0.4: "Ablation suite (Table 3 행 2–5·7) 미실행 — Phase 5 진입 전 실행 필수" bullet 신설, 최소 행 2·7 필수 + 행 7 = bullet 3 load-bearing + 271 canon config + 행 2 anomaly-OD 제외 유지 설계 조건 — 등재 확인.
- §6.7: 행 5(FM) "미실행 placeholder + 행 6과 동일 conditional 규칙", 행 7(symmetric) "미실행 + load-bearing + 필수 실행 + 미완 시 bullet 3 주장 강도 하향('intended to provide') 지침" — 확인.
- PAGE_BUDGET §3 Table 3 행: conditional을 행 5·7 포함으로 확장 + bullet 3 load-bearing + §0.4 상호 참조; r3 정정 이력 2항 — 확인.
- fixlog §3 EXPERIMENT_EXECUTION_TODO 항목 6 신설(구 fixlog r2 §7의 8항목 대체·확장 선언 명시) — 확인. **3개소 + TODO 집계 일관, 누락 0**.

### Spot ⑥ (보너스) — NEW-n1 라인 정정 + metadata 재실측 — **정확 (PASS)**

- `evaluator.py:811–813` 직접 확인: `:811` affiliation_precision_ar, `:812` affiliation_recall_ar, **`:813` `out['affiliation_f1_ar'] = ...` (키 할당)** — 블루프린트 §6.4 "811–813 (키 할당 :813)" 정확.
- PSM experiment_metadata.json 독립 재추출 — fixlog r3 verification_basis 기재 8개 키 전건 일치 (+d_model=512, dim_feedforward=2048, grl_disable_anomaly_loss=True 추가 확인).

## 2. r2 재리뷰 발견 전수 마감 대조 (11/11 닫힘)

| 출처 | ID | 등급 | fixlog 처리 | 문서 반영 확인 위치 | 판정 |
|------|----|------|------------|-------------------|------|
| ADV r2 | NEW-B1 | BLOCKER | §2-2 + 정본 escalation | BLUEPRINT §5.5(이원 서술)·§5.6(C)(λ_rev 정정)·§9.1(λ_rev 행)·§9.2(조항 교체)·§15(방어 재료)·헤더 r3/r2 취소선; 271truth r4 §VIII 3행 + CODEBASE r4 + SYNTHESIS r3 | **닫힘** (Spot ①·④) |
| ADV r2 | NEW-B2 | BLOCKER | §2-2 | BLUEPRINT §5.5 역전 교체 + 평가 경로 구분 + 서술 금지 지침 + capacity-gap 재점검; 정본 Training/표A 행 | **닫힘** (Spot ②) |
| ADV r2 | NEW-m1 | MINOR | §2-2 | §0.4 "학습 단위 36/113 + 평가 단위 37/114" 병기·혼용 금지; §6.2 완주 주석 동일 통일 | **닫힘** |
| ADV r2 | NEW-n1 | NOTE | §2-2 | §6.4 "811–813 (키 할당 :813)" | **닫힘** (Spot ⑥) |
| RT r2 | R2-MAJ-01 | MAJOR | §2-2 + §3 항목 6 | §0.4 bullet + §6.7 행 5·7 + PAGE_BUDGET §3 + 헤더 | **닫힘** (Spot ⑤) |
| RT r2 | R2-MIN-01 | MINOR | §2-2 | §14 논거 ② "가장 직접적인" + synthetic injection 병기 + "유일한" 표기 금지 | **닫힘** |
| RT r2 | R2-MIN-02 | MINOR | §2-2 | PAGE_BUDGET §2 전략 1 fallback 사다리(fontsize→Table 4 흡수→1열화 최후 수단+V3 재결정 필요) + §7 + r3 이력 1 | **닫힘** |
| RT r2 | R2-MIN-03 | MINOR | §2-2 | §6.6 실행 사양 신설(use_grl=True 유지 + loss.py:293–302 인용 + use_grl=False 금지 + baseline=Q3) + TODO 항목 3 설계 조건 | **닫힘** (Spot ③) |
| RT r2 | R2-MIN-04 | MINOR | §2-2 | §6.5 "실측 완료 데이터셋 기준 0.5–6.2%; SMD per-machine 확정 대기" — §5.2 어법 통일 | **닫힘** |
| RT r2 | R2-NOTE-01 | NOTE | §2-2 + §3 항목 7 격상 | §6.3 권고 단락(REQUEST-4 (iii) 소형 실험 + baseline epoch 1점 + §B.4 실측 격상) | **닫힘** |
| RT r2 | R2-NOTE-02 | NOTE | §2-2 | §3.1 Para 3 스코핑("the standard MTSAD benchmarks we evaluate on") + §14 배치 지침 동일 | **닫힘** |

발견 수 대조: redteam r2 = MAJOR 1 + MINOR 4 + NOTE 2 = 7건, adversarial r2 = BLOCKER 2 + MINOR 1 + NOTE 1 = 4건 → **총 11건, fixlog §2 처리표와 1:1, 미마감 0**. NEW-B1의 "r2 서술 보존 범위"(손실 항 즉시 투입 + grad-ratio×0.2 유지) 판단도 코드 재확인 결과 타당.

## 3. Directive 17종 충족 근거 판정표

§9.4 매핑 표 대조: Phase 3가 담당 Phase에 포함된 ID = T3, R1, R2, R5, R6, R7, R8, R9, R10, R11, R15, R16, R19, R20, R21, R22, R32 — **프롬프트 17종과 일치, §7/§9.4 불일치 없음 (ERRATA 불요)**. 블루프린트 frontmatter directives 목록도 동일 17종.

| ID | 판정 | 충족 근거 (산출물 + 섹션) |
|----|------|--------------------------|
| T3 | **충족** | BLUEPRINT §2(섹션 구조 전체 개요) + §3–§8(섹션별 내용·필요 근거·figure/table 계획·Appendix) — 논문 전체 개요·틀 완성; ADV r2의 T3 "부분 충족" 잔존 사유(코드-모순 2건)는 r3에서 해소 (본 게이트 Spot ①·② 코드 재검증) |
| R1 | **충족** | §4.1(Related Work 3소절 MECE 논거) + §6.1(Experiments 소절 MECE 논거) + §11 결정 ① MECE 검증문(bullet 2/3 경계 "라벨 신호 주입 vs 구조적 기판") |
| R2 | **충족** | §11 결정 ①(Notion C1–C4 채택/수정/기각 판정표 + 사유) + §5.3/§5.4/§5.5(NOTION I-3/I-4 stale 판정 — 코드 정본 우선 채택) |
| R5 | **충족** | §9.1(기호 체계 표 — λ_GRL/λ_rev 역할 분리 포함, 코드 정합) + §9.2(수식 금지 사항 — 재정의 금지·표기 통일) |
| R6 | **충족** | PAGE_BUDGET §1–§9(단일 정본 선언, 9.0p 배분, table/figure 넉넉 가정, 압축 사다리·체크포인트) + BLUEPRINT §2(전사 + 충돌 시 PAGE_BUDGET 우선) |
| R7 | **충족** | BLUEPRINT §8(Appendix A/B/C 구성 계획 + 위임 전략) + PAGE_BUDGET §5(Appendix 소절별 분량 ~4–5p) |
| R8 | **충족** | §0.1–§0.3(Thesis·포지셔닝·차별점 3축) + §11 결정 ① contribution 4-bullet 재설계(novelty 중심) + §10 명명 원칙(novelty 부각) |
| R9 | **충족** | §4.4 옵션 C("adapt this architectural paradigm", sibling 포지셔닝, 차이 나열 금지) + §11 결정 ⑤ + §15 SDMAE 유사 방어 행 + D-007(SDMAE 유사감 최대 후보 1·5 기각 사유) |
| R10 | **충족** | §12 R10 논증 배치 전수표(10 component × 강도·위치·요지) + §5.2/§5.4/§5.5/§5.6/§5.7/§6.2 개별 "왜 다변량 시계열인가" 논증 |
| R11 | **충족** | §5.2(contaminated semi-supervised 정의 + ②-1/②-2/②-3 3단 구조) + §3.1 Para 3 + §11 결정 ②(명칭 확정 사유) + §6.8(일반 케이스 검증 연결) |
| R15 | **충족** | §10.1 모델명 후보 4종 + §10.2 제목 후보 5종(각 장단점 표 — "3개 이상+장단점" 충족) + DECISION_LOG D-007 선정 기록(모델명 CSMAD·제목 후보 2 + 기각 사유) + §10 명명 원칙(불필요 신규 약어 금지) |
| R16 | **충족** | §14 논거 ⑤(NRdetector 재분할 선례 — 단정 인용 금지 주의 포함) + §6.4(동일 평가 철학 인용) + §6.8(label-noise sweep 패턴 참고) + §4.3(실험·논리 차이점 구분) |
| R19 | **충족** | §4.2 인용 정책(괄호 클러스터 인용, baseline은 §4 Experiments에서 이름+인용 최초 결합) + §6.5(계열명 1문장 + 대표 인용, 개별 소개 금지) |
| R20 | **충족** | §4.3(시계열 PU/SSL 희소성 정밀 스코핑 강조 + NRdetector 공통점 3개 짧게·차이 D1/D3/D5 중심) |
| R21 | **충족** | §4.4 용어 계보(Zhang TPAMI 2022 → SDMAE → 본 논문) + §11 결정 ⑤ 각주 초안(용어 계보 + branch-off vs 독립 decoder 구조 차이 방어) |
| R22 | **충족** | §4.4(vision MAE = 직접 계보 vs 시계열 patch 연구 = 독립 수렴 명시) + §5.4 R22 원칙(He et al. 2022 도입 1문장 명시) |
| R32 | **충족** | §6.8 Label Sparsity Analysis(p∈{1.0,…,0.1} sweep 설계 + Fig. 3 + 강건성 선험 논리 4단) + fixlog §3 TODO 항목 4(실행 등재) |

**17/17 근거 확보 — 근거 불가 Directive 0.**

### DECISION_LOG D-005~D-007 정합

- **D-005**(결정 ①–⑥ 채택) = BLUEPRINT §11 결정 ①–⑥과 항목·내용 일치 (4-bullet/명칭/0.62899/Q3+Table 4 격상/옵션 C/코드 조건부). 정합.
- **D-006**(⑦ DAGMM 표기, ⑧ TS-SDMAE 제외, ⑨ PAGE_BUDGET 단일 정본, ⑩ d_model 512, ⑪ Phase 1 정본 회귀 보강) = 결정 ⑦/⑧ + PAGE_BUDGET frontmatter + §5.4 + r3 escalation 기록과 일치. 정합.
- **D-007**(R15 선정: 모델명 **CSMAD**, 제목 **후보 2**) — 블루프린트 §10이 "orchestrator DECISION_LOG에서 확정"으로 위임한 사안을 확정한 것으로 정합. 선정이 §10.2 "권장(옵션 1 또는 5)"과 다르나, 기각 사유(후보 1·5의 "Self-Distilled MAE" 전면 배치 = R9 위험 최대)가 §10.2 표 자체의 단점 기재("SDMAE 유사감")·§15 방어 행과 일치하므로 **모순 아님** (NOTE-3 참조).

## 4. frontmatter·정정 이력 적합성

- **PAPER_BLUEPRINT.md**: phase 3 / agent blueprint-reviser / directives 17종(매핑과 정확 일치) / revision r3(fixlog 참조 명시) / last_modified 2026-06-11 / authority(정본 우선순위·placeholder 정책·PAGE_BUDGET 전사 선언) — 적합. 헤더 r3·r2 정정 요약 + 말미 r3·r2 정정 이력 부록(처리 ID·코드 근거 file:line 포함) — 적합. r2 헤더의 폐기 서술(BLK-004 단정)은 취소선 + r3 정정 포인터로 보존 — 이력 추적성 양호.
- **PAGE_BUDGET.md**: phase 3 / directives [R6] / revision r3(R2-MIN-02 + R2-MAJ-01 파급 명시) / last_modified 2026-06-11 / 단일 정본 선언 — 적합. r3·r2 정정 이력 부록 — 적합.
- **p3_fixlog_r3.md**: phase 3 / agent fixer / inputs·outputs 전수 / verification_basis(metadata 재실측 명시) — 적합.

## 5. 비차단 NOTE (게이트 판정에 영향 없음)

1. **[NOTE-1] fixlog r3 §1 A-3 라인 표기 off-by-one**: "trainer.py:747 elif use_grl…" — 실제 elif는 **:746** (:747은 주석). 블루프린트·정본이 인용하는 범위(:751–765, :760, :762–763)는 전부 정확하므로 산출물 영향 0 — fixlog 내부 표기만. (동류: A-2 "model.py:1149-1150"의 if문은 :1150 — 범위 내라 무해.)
2. **[NOTE-2] NOTION_DIGEST.md:168 (I-4) stale 서술 잔존**: "Student 디코더 forward는 수행되지만 … 손실 비활성" — NEW-B2로 코드와 반대임이 확정된 서술이 원천 digest에 무표기 잔존. 원천 충실 전사가 digest의 역할이므로 수정 의무는 아니나, "[검증된 사실 후보]" 라벨 + "논문 학습 절차 서술에 필요한 메커니즘" 문구가 Phase 5 drafter 오용 경로가 될 수 있음 → 해당 항목에 "⚠️ 코드와 모순 — 271_CONFIG_TRUTH §VIII r4 참조" 1줄 cross-ref 권고 (블루프린트 §5.5의 stale 명시가 1차 방어로 이미 존재 — 비차단).
3. **[NOTE-3] 블루프린트 §10 "권장" 행 stale**: §10.1 "권장: CSMAD 또는 SemiMAD"·§10.2 "권장: 옵션 1 또는 5"는 D-007 확정(CSMAD + 제목 후보 2) 이후 참고용 — 블루프린트가 확정 권한을 DECISION_LOG에 위임했으므로 모순은 아니나, Phase 5 drafter는 **D-007 기준** 준수 (필요 시 §10에 D-007 포인터 1줄 추가 권고).

## 6. 종합 판정

**PASS.**

- r3 spot 재검증 6건(요구 5건 초과) 전건 코드 1차 소스와 일치 — 불일치 0. 271 metadata 게이트 사실(use_grl 등 11키)도 본 감사가 독립 재추출로 확인.
- 재리뷰 r2 발견 11건(RT 7 + ADV 4) **전수 마감** — fixlog 처리 기록과 문서 반영 1:1, 미마감 0. NEW-B1/B2의 근본 원인(Phase 1 정본 누락)은 escalation으로 정본 3종 동기화 완료, 모순 0 (PHASE_LEDGER Phase 1 재진입 round 1 기록과 정합).
- Phase 3 매핑 Directive **17/17 충족 근거 확보** (§3 판정표). DECISION_LOG D-005~D-007 정합 — R15는 후보 3개 이상 + 장단점 + 선정 기록 3요소 전부 충족.
- frontmatter·정정 이력 적합. NOTE 3건은 전부 비차단(차기 라운드/orchestrator 전달 사항).
