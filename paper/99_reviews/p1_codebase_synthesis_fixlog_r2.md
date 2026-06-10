---
phase: 1
agent: fixer-2
directives: [T1, R10, R11, R17]
review_input: p1_codebase_synthesis_r1.md
targets:
  - paper/01_research_understanding/CODEBASE_UNDERSTANDING.md (rev r3)
  - paper/01_research_understanding/RESEARCH_SYNTHESIS.md (rev r2)
last_modified: 2026-06-10
result: 22/22 처리 (FIXED 22 / REJECTED 0; 4건은 리뷰 주장 일부를 정밀화·부분 반박하며 FIXED)
---

# Fix Log — Phase 1 CODEBASE_UNDERSTANDING + RESEARCH_SYNTHESIS (r2)

모든 항목은 수정 전 코드 직접 재검증(file:line)을 거쳤다. "재확인" 열은 fixer-2가 코드에서 독립적으로 확인한 사실.

## BLOCKER (5/5 FIXED)

| ID | 판정 | 재확인 (코드 근거) | 처리 내용 |
|----|------|-------------------|----------|
| BLK-001 | **FIXED** | `compute_adaptive_lambda`(`loss.py:683–728`)의 유일 호출처 = discriminator 경로 `trainer.py:610` (조건 `use_discriminator` + `config.adaptive_lambda`, `trainer.py:608–615`). GRL λ inline 공식 `(_main_g.norm()/(_grl_g.norm()+1e-4)).clamp(0,10)` `trainer.py:760` (블록 746–765); FM λ inline 공식 동형 `trainer.py:647` (블록 639–655). 둘 다 `compute_adaptive_lambda` 미호출. 적용값은 직전 epoch 집계값 (`_prev_epoch_grl_lambda`/`_prev_epoch_fm_lambda`, `trainer.py:189–190, 652, 762–763, 1298–1306`) | CODEBASE §2.6 전면 재작성 — Discriminator λ(VQGAN-style, 271 비활성) / GRL λ / FM λ **3경로 분리** + 각 공식·호출처·prev-epoch 적용 명시. §1 GRL bullet, §2.3, §2.5도 동기화. RESEARCH_SYNTHESIS 표A GRL/FM 행에 λ_GRL ≠ λ_FM + line ref 명시. (표A FM 행의 기존 ratio 공식 자체는 코드와 일치했음 — 오류는 CODEBASE 쪽 귀속 서술) |
| BLK-002 | **FIXED** | `_compute_patch_scores_all_patches` def `evaluator.py:1647`, docstring "Optimized: All patches processed in a single forward pass by expanding batch dimension" `evaluator.py:1650`. 구현: `sequences.unsqueeze(1).expand(...)` batch 확장 `evaluator.py:1807–1808`, forward `1818`. `patch_batch_size=2` HARD-LOCK 주석 "does NOT affect numerical results" `evaluator.py:1703–1717` | CODEBASE §3.1 재작성: "each patch is masked in turn" → 50개 마스킹 패턴 batch-차원 확장 병렬 forward (pb=2 분할은 메모리 관리, 수치 무영향). FLOPs ~50× 사실은 유효함을 병기(발표 p13 정합). RESEARCH_SYNTHESIS 표A Patch→Point 행 동일 정정 |
| BLK-003 | **FIXED** | BLK-001과 동일 근거 — "Used for both discriminator and FM adaptive weighting" 문장은 코드와 불일치 | CODEBASE §2.6에서 해당 문장 삭제, "discriminator 전용(271 비활성); GRL/FM은 별도 inline 공식"으로 대체 |
| BLK-004 | **FIXED** (리뷰 논거 일부 정밀화) | `loss.py:337–340`: `_p_t = exp(−_bce)`, `_focal = (1−_p_t)²×_bce`. 단 리뷰의 "exp(−BCE) ≠ σ(logit)" 일반 주장은 부정확 — **pos_weight 없는** BCE에서는 `exp(−BCE) = p_t` 성립. 비표준성의 정확한 근거는 **pos_weight 내장**(`loss.py:330–336`; exp271 `grl_balanced_sampling=False`로 pos_weight 경로 활성, dataset별 자동값 — SWaT metadata `grl_pos_weight=59.18`) → positive 샘플에서 `exp(−BCE_w)=p_t^w ≠ p_t` | CODEBASE §2.4: 코드 식 그대로 기술 + "표준 focal loss(Lin et al. 2017) 아님 / 'standard focal loss' 표기 금지" 플래그 + pos_weight 기반의 정확한 비표준성 논거. RESEARCH_SYNTHESIS 표A GRL 행 "focal γ=2" → "focal-style BCE 변형(표기 금지 플래그 포함)" |
| BLK-005 | **FIXED** | label 사용 3지점 재확인: `force_mask_anomaly` `model.py:975–1002`; OD 방향 분기 + `grl_disable_anomaly_loss` `loss.py:244–261`; GRL 타겟 `loss.py:282–350`. train anomaly 실측 0.52–6.20% (EXPERIMENT_PROTOCOL_TRUTH §①) | RESEARCH_SYNTHESIS §② 전면 재구성 (orchestrator 지침의 3단 구조): ②-1 설정(가정, R11 Directive 원문 인용) → ②-2 main 실험 구현 = label 가용성 **상한 케이스** (R13) → ②-3 라벨 희소화 sweep = 일반 케이스 검증 계획 (R32, placeholder 정책). "오염된 unlabeled 다수" 표현 제거로 모순 해소. CODEBASE §4.3도 동일 프레이밍으로 정합화(+"~5%" → 실측 0.52–6.20%) |

## MAJOR (9/9 FIXED)

| ID | 판정 | 재확인 (코드 근거) | 처리 내용 |
|----|------|-------------------|----------|
| MAJ-001 | **FIXED** | `fm_adaptive_lambda=False` 경로: `discrepancy_loss = normal + anomaly + fm_loss_weight×fm` `loss.py:438`; True(271) 경로: FM 제외 `loss.py:436` 후 trainer 추가 `trainer.py:652` | CODEBASE §2.5 Total Loss를 λ_FM_prev / λ_GRL_prev **별도 기호**로 재표기 + 두 경로 모두 FM 포함(위치만 다름) 명시 + 각 λ 공식 병기. §2.3도 양 경로 서술 |
| MAJ-002 | **FIXED** | 패치는 윈도우 내 비중첩 분할(50×10=500) → 윈도우당 t의 패치 1개. coverage = 덮는 윈도우 수 ≈ 500/49 ≈ 10.2. `resolve_test_stride`: 양수 override 우선, 비양수(271 sentinel −1)일 때만 공식 (`utils/experiment.py:35–39`). stride 49 ⊥ patch 10 → 패치 위치 잔차 순환(다양성 논리, docstring `utils/experiment.py:16–34`) | CODEBASE §3.3 coverage 유도 재작성 — "× patch-position coverage" 곱셈 인수 삭제, 윈도우 수 기반 유도 + 서로소 다양성 논리 + sentinel 조건 명시. RESEARCH_SYNTHESIS 표A Patch→Point 행에도 유도 반영 |
| MAJ-003 | **FIXED** | `fpr, tpr, thresholds = roc_curve(...)` `evaluator.py:928`; `find_f1_optimal_idx` 정의 `evaluator.py:215–226` (fpr/tpr + class count로 precision/recall 유도 후 F1 argmax); strict `>` 이진화 `evaluator.py:931` | CODEBASE §5.6: "F1-optimal point on the ROC curve" → "ROC threshold 격자에서 `find_f1_optimal_idx`로 F1 최대화 threshold 선택"으로 정밀화 |
| MAJ-004 | **FIXED** (리뷰 주장 일부 정밀화) | `grl_cls_arch='default'`, `grl_cls_hidden=0` (271 metadata) → default 분기 `model.py:177–186`, `hidden = d_model//2 = 256` `model.py:179`. 구조 = Linear 2개(LayerNorm→Linear(512→256)→GELU→Dropout(0.1)→Linear(256→1)). 단 RESEARCH_SYNTHESIS 표A에는 "1-layer MLP" 문구가 원래 없었음(해당 오기는 271_CONFIG_TRUTH §VIII — 본 fixer 담당 외) — 본 문서에는 예방적으로 "2-layer MLP" 표기 확정 주석 추가 | RESEARCH_SYNTHESIS 표A GRL 행: "2-layer MLP" 명시 + hidden 자동 산출 근거 + 코드 주석 "1-layer"는 hidden-층 수 기준 표현임을 병기(논문 표기 금지) |
| MAJ-005 | **FIXED** | F1-optimal threshold는 test label 필요 (oracle). AR threshold 변형 `_ar` suffix 병산 (`evaluator.py:790, 793–794`, `compute_ar_threshold_metric_set` 호출 `evaluator.py:985`) | CODEBASE §5.6 + RESEARCH_SYNTHESIS §④에 "oracle(best-F1) threshold — 논문 테이블 표기 의무" 경고 추가, leak-free 대안(AR) 병기 |
| MAJ-006 | **FIXED** | 비교군 정책: Q1 라벨 미사용, Q3 라벨은 anomaly 절제(데이터 정제)에만 사용 (RESEARCH_SYNTHESIS §④ 기존 FACT; `comparison/data/unified_loader.py:392–485`). 3지점 모두 라벨이 입력 인자인 함수 (`model.py:975–1002`; `loss.py:244–261, 282–350`) | RESEARCH_SYNTHESIS §②-5 신설: 비지도 방법이 masking 우선순위·손실 방향 분기·GRL 타겟을 구조적으로 정의 불가한 이유(학습 시 라벨 입력 부재)를 3지점별로 코드-직결 서술 — R11 "기존 unsupervised는 labeled 활용 불가" 논리 완성 |
| MAJ-007 | **FIXED** | 실측: `results/experiments/271_20260602_020545_271canon_baseline/` — SMD 22 dirs, SMAP 5, MSL 5, SWaT 2(full/excl22), WaDi 2, PSM 1 = 37 | RESEARCH_SYNTHESIS §① 요약: "총 112 entity" 완료 뉘앙스 제거 → 완료 37 (SMD 22/28·SMAP 5/54·MSL 5/27 진행 중) + "논문은 placeholder 정책(A8/R3)" 명시 |
| MAJ-008 | **FIXED** | Per-K 보고 키 step=5: `PA_K_VALUES = list(range(0,101,5))` `evaluator.py:831`, 사용처 `2111, 2139` (21점). AUC 적분 step=1: `k_values = np.arange(0,101)` `evaluator.py:1034` (101점 trapz) | CODEBASE §6.2: 두 해상도(보고 step5 vs 적분 step1) 명시 구분 + "pa_{K} 키 적분 = pak_auc" 식 서술 금지 경고 |
| MAJ-009 | **FIXED** (stale 플래그 대신 최종값 갱신) | 83.75% 재현 산출: metadata `excl_region22_info.region_length=35,900`, `test_length=224,960`, `metrics.anomaly_ratio=0.190541` → 42,864 anomaly pts → 35,900/42,864 = **0.83753**. 코드 docstring은 근사 "~84%" (`find_swat_region_22` `evaluator.py:2299–2310`; 기존 인용 2302–2306은 docstring 내부 행). 0.944/0.629 → 최종 metadata 실측 `metrics.pak_auc_f1=0.94436` / `metrics_excl_region22.pak_auc_f1=0.62730` (SWaT 완주 — stale 아님) | RESEARCH_SYNTHESIS §④ excl22: 산출 근거(파생식 + EXPERIMENT_PROTOCOL_TRUTH §⑥ 원 CSV 계산 일치) 명시; 수치를 최종 metadata 값(0.9444/0.6273)으로 갱신 — 리뷰 제안(STALE 플래그)보다 강한 해소 |

## MINOR / NOTE (8/8 FIXED)

| ID | 판정 | 재확인 (코드 근거) | 처리 내용 |
|----|------|-------------------|----------|
| MIN-001 | **FIXED** | `LinearLR(start_factor=1e-4)` `trainer.py:169–174` (start_factor 행 171) → 시작 LR = 1e-3×1e-4 = 1e-7 | CODEBASE §5.2: 정확한 시작 LR(1e-7)·스케줄 구성 명시 + 논문 표 기재 지침 |
| MIN-002 | **FIXED** (리뷰 우려 절반 해소-반박) | `self.decoder_pos_encoder` 단일 인스턴스 `model.py:343–346`. 단 `PositionalEncoding`은 **학습 파라미터 없는 고정 sinusoidal buffer** (`model.py:263–279`, `register_buffer`) → 공유/비공유가 수치·학습에 무영향. "설계 결정으로 기술해야 할 수 있다"는 리뷰 우려는 비학습 버퍼이므로 해당 없음 — 오히려 "shared learned pos-enc"로 쓰면 오류 | CODEBASE §1 Encoder/Decoder bullet에 공유 사실 + 비학습 버퍼라 수치 무영향 + 논문 표기 주의 추가 |
| MIN-003 | **FIXED** | classifier는 student decoder 마지막 층 hidden `(num_patches, batch, d_model)`에 패치별 독립 적용(풀링 없음), `squeeze(-1).transpose(0,1)` → `(batch, num_patches)` `model.py:1153–1154`; 손실은 masked 패치만 `valid = patch_has_masked` `loss.py:283–284` | RESEARCH_SYNTHESIS 표A GRL 행 + §②-4 GRL 항목에 패치별 독립 적용·valid mask 세부 추가 |
| MIN-004 | **FIXED** | SMD 완료 22/28 (results dir 실측); 29(machine-3-10)–36(machine-3-3)은 22개 기준 범위 | RESEARCH_SYNTHESIS §⑥ N5: "실측 22/28 기준, 잔여 6 machine 미측정" 한정 추가 |
| NOTE-001 | **FIXED** | warmup ablation 실험 부재 (수치 없음 — 발표 p24 정성 곡선뿐) | RESEARCH_SYNTHESIS 표A warmup 행 RISK → **CRITICAL RISK** 격상 + §⑨ REQUEST-F 신설(Phase 2 필수 ablation: warmup 0/50/250) |
| NOTE-002 | **FIXED** | repo: branch machineA, default main (git status) | RESEARCH_SYNTHESIS §⑦에 공개 전 checklist 4항목(branch 결정/공개 범위/secret 스캔/재현 진입점) 추가 |
| NOTE-003 | **FIXED** | — (구현 사실은 기존 문서 FACT 유지) | RESEARCH_SYNTHESIS §⑥ DAGMM 항목에 "방법 재정의 → reject 리스크 / 'DAGMM-simplified' 표기 후보 / Phase 3 시작 시 최우선 확정" 권고 추가 |
| NOTE-004 | **FIXED** (코드 확인 완료 — 리뷰의 미확인 우려 해소) | `_load_smap_msl_simple_single(spacecraft, channel, safe_cut_margin: int = 10)` `datasets/loaders.py:2527`; 50% 지점이 anomaly region ±10 이내면 `_find_safe_cut_point`(`loaders.py:1050`)가 region 밖으로 cut 이동 (`loaders.py:2591–2596`) | CODEBASE §4.1 SMAP 행: safe-cut의 정확한 코드 근거(margin=10 기본값 + safe-cut 함수)로 교체 — "±10" 표현의 실체 확정 |

## 추가 처리 (리뷰 범위 외, 코드 실측 기반)

- CODEBASE FEEDBACK(`grl_pos_weight`) 부분 RESOLVED: 271 metadata 실측으로 dataset-specific 자동 설정 확인 (SWaT `grl_pos_weight=59.1814` ≠ 고정 19.0).

## 비변경 확인

- 리뷰 "검증 완료 사항 (PASS)" 목록 전 항목은 두 문서에서 무수정 유지.
- `paper_legacy/` 미접근. 코드·실험 환경 무변경 (read-only). 쓰기 파일: 대상 문서 2개 + 본 fix log.
