---
phase: 1
agent: fixer-4
target: paper/01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md
review: paper/99_reviews/p1_protocol_r1.md (adversarial-reviewer-C)
verdict_in: REJECT (BLOCKER 1, MAJOR 4(+M-5), MINOR 7)
result: 12/12 FIXED, 0 REJECTED
last_modified: 2026-06-10
---

# Fix log r2 — EXPERIMENT_PROTOCOL_TRUTH.md (p1_protocol_r1 전수 반영)

> 처리 원칙: 각 발견마다 **리뷰 주장을 코드/데이터로 독립 재검증**한 뒤 수정. 리뷰어 실측치(safe-cut 이동량, 라인 번호, 키 수)도 전부 재계산/재열람으로 재검증했고, 12건 모두 리뷰가 옳았다 (반박 0건). 코드·실험 환경 read-only 준수 — 쓰기는 대상 문서 + 본 fix log 2개 파일뿐.

## 재검증 방법 요약

- `_find_safe_cut_point` 본문 직접 재열람 (`loaders.py:1050-1101`) + **81채널 전수 실측 재현** (dc_vis env, 실제 채널 라벨/`labeled_anomalies.csv`로 함수 직접 호출): SMAP 0/54 이동, MSL 4/27 — D-16 +166 (7.58%), M-1/M-2 −39 (1.71%), S-2 +8 (0.44%) — **리뷰 r1 수치와 bit-exact 일치**.
- `wc -l`: loaders.py=2770줄, noisy.py=85줄. `grep -n`: SMD_simple alias `:2742`, WaDi 레지스트리 `:2698-2699`, SWaT `:2690`(이상 없음), `_compute_threshold_dependent` def `:637`.
- `baseline_common.py:335-354` 재열람: `:343-347`은 NRDetector PU class-prior 주석 — 리뷰 판정대로 threshold 관행 인용으로 부적합.
- `evaluator.py:1363-1373` (Evaluator 생성자 = test_loader/test_dataset만), `:2155-2160` (evaluate docstring), `run_base_experiments.py:2604, 2645-2646, 3215-3240` (best-epoch 선정), `baseline_common.py:1368, 2087-2098` (baseline 동일 기준) 재열람.
- seed: `config.py:322` (`random_seed=42`), `:326-333` (`set_seed`), `run_base_experiments.py:2435` 호출 + 2442/2509/2522/2542 전파; 반복 루프 grep 0건.
- mean 집계: `evaluator.py:272, 278-280, 295-304, 8, 2158`.
- 패리티: MAE `num_epochs=50` (`config.py:264`) = baseline 50 (`baseline_common.py:256, 266, 337`); per-epoch eval `:949+` (`eval_interval=1`); baseline early-stop grep 0건; MAE warmup early-stop 기본 off (`config.py:286`).
- [271c] PSM metadata `metrics` 재집계: **153키, None 0건, `_ar` 10키** — 리뷰 일치.
- `STANDARD_BASELINES` (`experiment_configs.py:24-30`): legacy 6종 (anomaly_transformer, tranad, usad, dagmm, gdn, omnianomaly) 확인 → 5+3+1+6+7=22 정합.
- entity 산식: 1+2+1+28+54+27=**113** (SWaT dual-eval 평가 단위로는 114).
- Notion 덤프 실재 경로 확인: `/home/ykio/.claude/projects/-home-ykio-notebooks-TSMAE/0aa53593-b13e-47f9-bea0-4e3aa040496f/tool-results/mcp-claude_ai_Notion-notion-fetch-{1781093695371,1781093708082}.txt`.
- 부수 확인: `_find_safe_cut_point`의 또 다른 호출처 `:1825`는 `load_smd_block_split`(k6 variant — 논문 미사용 경로)이므로 "논문 데이터셋 중 safe-cut은 SMAP/MSL뿐" 주장 유지에 문제 없음. PSM/SMD train 라벨 파일 부재→zeros 처리: `loaders.py:1672-1675`(PSM), `:1139-1142`(SMD), SMAP/MSL 명시적 zeros `:2602-2604`.

## 발견 ID별 처리표

| ID | 심각도 | 리뷰 주장 재검증 | 판정 | 조치 (문서 내 위치) |
|---|---|---|---|---|
| **B-1** | BLOCKER | `margin=10`=clearance 요건(`:1053, 1071-1073`), 무제한 outward 탐색(`:1080-1083`), 발동 조건="region ±10 이내" — 코드 재열람 일치. 실측 이동량 81채널 전수 재현 — r1 수치와 bit-exact 일치 (D-16 +166=7.58%) | **FIXED** | §② 표 SMAP/MSL 행을 정확한 메커니즘으로 재작성 + **실측 이동량 표 신설** (SMAP 0/54, MSL 4/27, max +166) + Pattern B 함의("D-16 test 166 step 단축") 명기 + `docs/DATASET.md:1151`·[N-COMP] "±10" 표현 **ERRATA 기록** (코드/docs 수정은 범위 외). §① SMAP/MSL 행 표현 교정(§② 참조로 위임). §②-서술재료-2(공정성)를 "분할 규칙(//2)은 전 데이터셋 통일 + SMAP/MSL 경계 조정은 실측 이동량과 함께 정직 공개, '실질 영향 없음' 일반화 금지"로 재구축. FEEDBACK-5 동일 취지로 재작성 |
| **M-1** | MAJOR | 파일 2770줄 (wc -l), `SMD_simple_<machine>` alias 실제 위치 `:2742` (grep) — 리뷰 정확 | **FIXED** | §② 표 SMD 행: `2810-2812` → `2742` |
| **M-2** | MAJOR | `baseline_common.py:343-347` 재열람 — NRDetector PU class-prior 추정 주석 확인, test thresholding 관행과 무관 | **FIXED** | §⑤-4: `:345` 인용 제거 + 부적합 사유 명기 + "문헌 관행" 주장은 **근거 보류·Phase 4 reference 검증 수요로 이관**, 확보 전 논문 사용 금지 명시 |
| **M-3** | MAJOR | Evaluator 생성자=test_loader만(`evaluator.py:1363-1373`), per-epoch callback이 test 지표로 best ckpt 갱신(`run_base:2604,2645-2646`), 최종 best epoch도 test 지표 스캔(`:3215-3240`), baseline 동일(`baseline_common.py:1368,2087-2098`), validation split 부재 — 전부 확인 | **FIXED** | §④ Best-epoch 항목에 "**best epoch = test-split pak_auc_f1 최대 epoch = test-set model selection (전 모델 동일)**" 명시 등재 (file:line 5건) + §⑤-2에 cross-note("이 방어 사용 시 §④ 사실 동반 공개 필수") + **REQUEST-4 신설** (논문 서술 방식 결정: oracle-protocol 명시 공개 등 3안) |
| **M-4** | MAJOR | seed=42 단일 run(반복 루프 0건), mean 집계(`evaluator.py:272/278-280/295-304/2158`), epoch 50 통일·per-epoch eval·early-stop 부재 — 전부 코드 확인; 271_CONFIG_TRUTH.md 실재 확인 | **FIXED** | §④에 "**실행 프로토콜**" 소절 신설: ① 단일 run/seed=42 (분산·CI 보고 불가 명시) ② window→point **mean** 집계 ③ baseline 학습 패리티(epoch 50/eval cadence/best-epoch 기준/early-stop 부재 + 모델별 HP 상이 가능 단서) ④ `271_CONFIG_TRUTH.md` cross-ref. 근거 인덱스에 "실행 프로토콜" 행 추가 |
| **M-5** | MAJOR | WaDi 레지스트리 실제 `:2698-2699` (`:2697`은 주석), SWaT `:2690` 정확 — grep 확인 | **FIXED** | §② 표 WaDi 행: `2697-2698` → `2698-2699` |
| MIN-1 | MINOR | 1+2+1+28+54+27=113 — 산식 재계산 일치 | **FIXED** | §① 표제 112→**113** + 산식 명기 (dual-eval 평가 단위 114 병기) |
| MIN-2 | MINOR | `experiment_configs.py:24-30` legacy 6종 — 5+3+1+6+7=22 정합 확인 | **FIXED** | §③ "SOTA legacy 7"→**6** + 6종 명단 + 총계 산식 명기, 인용 라인 23-31→24-30 정밀화 |
| MIN-3 | MINOR | [271c] PSM metrics 재집계 **153키**·None 0·`_ar` 10키 | **FIXED** | §⑧ REQUEST-1 RESOLVED-2: 149→**153**키 (오기 정정 표기) |
| MIN-4 | MINOR | noisy.py 85줄 (wc -l); class `:7`, `use_noisy_labels` `:52` 정확 | **FIXED** | §⑦ 및 §⑧ 근거 인덱스: `7-87`→**`7-85`** |
| MIN-5 | MINOR | `_compute_threshold_dependent` def `:637` (grep) | **FIXED** | §⑧ RESOLVED-3: `639+`→**`637`** |
| MIN-6 | MINOR | SMAP/MSL train 라벨=명시적 zeros(`:2602-2604`); PSM/SMD는 train 라벨 파일 부재→정상 취급(`:1672-1675`, `:1139-1142`) | **FIXED** | §② 공통 패턴 아래에 "'전부 정상'의 정확한 성격" 1단락 추가 (라벨 부재→정상 취급=분야 표준 가정, 논문 1줄 명시 권장) |
| MIN-7 | MINOR | 세션-상대 `tool-results/` 경로 — 실재 절대경로 확인 | **FIXED** | 헤더의 [N-METH]/[N-COMP] 덤프 경로를 절대경로로 교체 |

## 기타

- frontmatter에 `revision: r2 (fixer-4)` + `review_applied` 추가, 문서 상단에 **정정 이력(r2)** 블록 신설.
- 근거 인덱스의 safe-cut 인용을 `1050-1103`→`1050-1101`로 정밀화 (+clearance/탐색 라인 명기).
- REJECTED 0건 — 리뷰어의 12건 발견·실측치 모두 독립 재검증에서 확인됨.
- 리뷰 §4(검증 통과 항목)는 변경 불요 — 본 수정에서 건드리지 않음 (단, 통과 항목 중 §⑤-2 best-epoch 서술은 M-3와 연동되어 cross-note만 추가).
