---
phase: 8
agent: spec-fixer (r2)
directives: [R3]
last_modified: 2026-06-11
inputs: |
  paper/99_reviews/p8_notion_spec_review_r1.md (발견 F-1..F-5 + OBS-1/2),
  paper/00_admin/DECISION_LOG.md D-014 (a)(b),
  코드 직접 재확인(read-only): mae_anomaly/scoring.py (resolve_score_weights :85-108, score 분기 :247-256),
  mae_anomaly/trainer.py:1200-1210, configs/queue_dedup_renumbered_v5.json (295-303 전수),
  results/experiments/271_*/PSM + 298_*/299_* metadata (num_evals/eval_interval 실측),
  paper/01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md r4 §④-3
outputs: |
  paper/08_final_audit/NOTION_PLACEHOLDER_SPECS.md (r2),
  paper/07_latex/sections/appendix_B.tex (B.2 +2문장),
  paper/07_latex/PROSE_DIFF_LOG.md §6,
  paper/07_latex/{main.pdf, main_5p_measure.pdf} 재빌드,
  paper/07_latex/overleaf_package.zip 재패키징
verdict: 발견 7/7 처리 (BLOCKER 1, MAJOR 1, MINOR 3, 관찰 2) + D-014 (a) 원고 보강 완료 — 본문 무영향 검증 PASS
---

# P8 fixlog r2 — 명세 r2 (검수 전수 반영) + D-014 (a) Appendix B.2 보강

## 1. 작업 A — NOTION_PLACEHOLDER_SPECS.md r2 (p8_notion_spec_review_r1 전수 반영)

모든 정정은 코드·큐·metadata **직접 재확인** 후 적용 (리뷰 인용 line은 본 fixer가 재실측한 행 번호로 기재).

| 발견 | Severity | 처리 | 검증 |
|---|---|---|---|
| **F-1** | BLOCKER | **권고 실험 R-PROBE 등재** — 신설 §6R "권고 실험 (rebuttal 대비, 원고 비반영) — D-014 (b)": [271c] 대표 entity best checkpoint 동결 + Student decoder final-layer hidden(output projection 직전 — GRL 부착 지점 동일, FIG-2 ③ⓒ) vs Teacher 동일 위치 hidden 위에 소형 probe(LayerNorm+Linear 1층) 학습 → probe AUC 비교 (기대: Student ≪ Teacher); w/o GRL run 대조군 확장(exp290 복합 각주 의무) 포함. `[신규 측정]` 분류, §7.4 표 1행 추가, §8 커버리지에 REGISTRY-외 항목으로 명기 | 리뷰 §6 권고안 그대로 채택 — 원고 placeholder 무관(원고 무변경) |
| **F-2** | MAJOR | **TAB-3 행4 (w/o OD) 전제 정정** — 구판 "OD 학습 제거 후에도 추론 score는 disc 성분 포함(adaptive 식)"은 코드와 **반대**라 폐기. 재서술: 기본 동작 = **자동 recon-only** — `scoring.py:105-106` `resolve_score_weights`가 `use_output_discrepancy=False`면 `w_disc=0` 강제 → `scoring.py:249-253` `if w_disc > 0 …` else `student_error=np.zeros_like(recon)` → score = Teacher recon만. "자동 recon-only 동작을 표 각주로 명시 + disc 잔류 변형은 별도 채점 경로 필요"로 방침 확정. §7.3 #6도 동일 정정 | `resolve_score_weights` 직접 재확인 (docstring "If use_output_discrepancy is False, w_disc is forced to 0" + :105-106 구현 + :249-253 분기 — 본 fixer 실측) |
| **F-3** | MINOR | TAB-B4 ④ 캡션 전사 정정: "(Teacher 2L\,/\,2L)" → "(Teacher 2L\,/\,Student 2L)" | `appendix_B.tex:157` 원문 직접 대조 — tex 그대로 |
| **F-4** | MINOR | [CMP-Q3] 서술 정정 ×4개소 (§0 약칭 정의, TAB-2 ② 2항, §7.3 #1, FIG-3 ⑤ⓑ): "SMD/SMAP/MSL STALE 재실행" → "`6_20260526_*`에는 SWaT/WaDi/PSM만 존재; SMD normalonly = 구버전 `3_20260312_*`(per-entity 정규화 이전 — 폐기 대상), SMAP/MSL normalonly = 어느 폴더에도 부재(미실행) → SMD 구버전 폐기+재실행 / SMAP·MSL 미실행분 신규 실행". 실행 결론(전 entity 실행) 불변 | 리뷰 실측 기재 그대로 (실행 지침·의존 placeholder 무변경) |
| **F-5** | MINOR | FIG-B1 ⑤ⓒ 큐 범위 정정: "295–303은 window/patch 크기 sweep" → "295/296/300–303 = window/patch sweep, **297 = dynamic d_model, 298/299 = epoch-budget 변형**" — masking-ratio 항목 없음(전 32항목 override 0건) 결론 불변 | 큐 v5 295–303 `config_override` 전수 직접 실측 (297 `d_model=dynamic`, 298 `num_epochs=300 warmup=150`, 299 `num_epochs=200 warmup=100`) |
| OBS-1 | 관찰 | ALG-C1 ④ τ식에 epoch 규약 연동 표기 추가: τ=clip((e−250)/250)는 **1-based e 규약에서만** 코드와 일치 — 3항의 규약 명시(각주/KwIn)가 이 식에도 적용됨을 한 줄 연동 (off-by-one 재발 차단) | `trainer.py:1205-1207` 직접 재확인 (`_student_total = max(num_epochs − _student_start, 1)`, `_p = (_student_epoch + 1)/_student_total` — 0-based) |
| OBS-2 | 관찰 | TAB-3 행3(exp287 재사용)에 큐 `force_mask_anomaly` 키 중복(True→False, last-wins) 경고 추가 — 신규 큐 항목 작성 시 답습 금지 | 리뷰 실측 인용 (metadata 단독 diff 확정은 기존 기재 유지) |

frontmatter `revision: r2 (p8 spec-fixer)` + `review_applied` 갱신, 신설 §9 정정 이력에 발견별 기록. 기존 절 번호(§7/§8)는 문서 내 상호참조 보존을 위해 유지 — 신설 절은 §6R/§9.

## 2. 작업 B — D-014 (a): Appendix B.2 선택-기회 비대칭 명시 공개

### 2.1 추가 문장 (appendix_B.tex §B.2 lead 문단 — 신규 2문장, appendix 한정)

> **S1** (기존 budget 문장 직후): "Because every method is reported at its best evaluated epoch, the budget asymmetry also entails an asymmetry in selection opportunities: under the evaluation cadence of Section~\ref{sec:impl}, CSMAD is evaluated at 100 checkpoints (every 5 of 500 epochs), versus 50 and 10 for the weakly supervised and unsupervised baselines (every epoch)."
>
> **S2** (기존 "To assess …" 문장 직후, 문단 말미): "These runs keep the evaluation cadence fixed, so the number of evaluated checkpoints scales with each budget and the sweep probes the selection-frequency effect together with the training-length effect."

**수치 유도 (전부 프로토콜 상수 — 발명 0건)**: CSMAD 100회 = [271c] metadata `timing.num_evals=100` 실측 (= `num_epochs=500` ÷ `eval_interval=5`); weak 50회 = 50ep × 매 epoch eval; unsup 10회 = 10ep × 매 epoch eval (EXPERIMENT_PROTOCOL_TRUTH r4 §④-3 ①②; `baseline_common.py:943` `eval_interval=1`). S2의 "cadence fixed + 비례 스케일"은 exp298/299 실측으로 입증 (`eval_interval=5` 유지, `num_evals` 60/40 — budget 비례) + 명세 TAB-B2 ⑤(baseline 50/100 run도 매 epoch eval 의무)와 정합. D-014 (a)의 "≈100 vs 10회"를 정확 수치(100/50/10)로 구체화.

### 2.2 미니 감사 3종 — **전부 PASS** (상세: PROSE_DIFF_LOG.md §6.2)

| 검사 | 판정 | 요지 |
|---|---|---|
| ① ai-phrasing | PASS | SENTENCE_CORPUS 부록 B 금지/자제 패턴 0건 (em-dash 0, 전환부사 0, 의인화 0 — 스캔 히트 2건은 인접 LaTeX 주석 `% ---- … ----`로 산문 아님). 수치 결합 선언문 — 양성 신호 부합 |
| ② plagiarism | PASS | 변별 n-gram 8종 × corpus(105문장+dossier 2종) + library 52 cards 전체 — 일치 0건. "best evaluated epoch"는 본문 §4.1.2 자기 표현 재사용(용어 일관성) |
| ③ method-truth | PASS | 100/50/10·cadence 5 vs 1·비례 스케일 전부 정본+metadata 실측 일치 (위 2.1) |

### 2.3 재컴파일 + 본문 무영향 검증 — **PASS**

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 빌드 (latexmk ×2: main.tex, main_5p_measure.tex) | — | exit 0 ×2, `!` 오류 0, undefined ref 0, 렌더 "??" 0 |
| main.pdf (preprint) | 46p | **46p (불변)** |
| main_5p_measure.pdf | 19p | **19p (불변)** |
| **5p 본문 종점** (§5 "…(to be released upon acceptance).") | printed p.9 우측 컬럼 yMax 762.8pt (PROSE_DIFF_LOG §5.7) | printed p.9 (PDF p.10) 우측 컬럼, 종점 단어 "ceptance)." **yMax 762.842847pt — 동일 좌표** → 본문 8.997p 보존, R6 게이트 상태 불변 |
| 신규 문장 렌더 | — | PDF p.15 (§B.2 — 변경 전과 동일 페이지; appendix 내부 흡수, appendix 총량도 불변) |
| Overfull | 5p 1건 / preprint 10건 | 5p 1건 / preprint 10건 — 회귀 없음 |

### 2.4 zip 재패키징 + 단독 컴파일 재검증 — **PASS**

- `overleaf_package.zip`에 `sections/appendix_B.tex`만 갱신 (12파일 구성 유지). zip 전 12파일 ↔ 07_latex 정본 `cmp` 전수 대조 — **12/12 MATCH** (stale 파일 없음).
- 임시 폴더 단독 컴파일: exit 0, 오류 0, 46p (정본 preprint와 동일), "??" 0, 신규 문장 렌더 확인 (1 hit).

## 3. 잔여/이관 사항

- 없음 — 리뷰 r1의 "위 5건 반영 후 재검수 없이 발행 가능 (r2 fixlog로 갈음)" 조건 충족. OBS 2건도 반영 완료.
- R-PROBE는 원고 placeholder와 무관한 권고 실험이므로 REGISTRY·커버리지 산식 불변 (§8에 REGISTRY-외 명기).
