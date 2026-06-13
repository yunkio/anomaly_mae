---
phase: 1
agent: protocol-truth-writer
directives: [R12, R13, R24, R28, R29, R30, R31, R32, R33]
revision: r4 (p5 comprehensive-fixer)
review_applied: paper/99_reviews/p1_protocol_r1.md (r2, BLOCKER 1 + MAJOR 4(+M-5) + MINOR 7 — 전수 반영) + paper/99_reviews/p1_rereview_beta_r2.md (r3, RB-1 BLOCKER + RM-1 MINOR) + paper/99_reviews/p5_method_truth_r1.md §5 errata 1·2 (r4)
last_modified: 2026-06-11
---

# EXPERIMENT_PROTOCOL_TRUTH — 실험 프로토콜의 "진실" (코드·Notion·metadata 실측)

> **원칙**: 본 문서의 모든 주장에는 근거(코드 file:line / Notion 덤프 인용 / metadata 필드 / 원데이터 직접 계산)가 붙어 있다.
> 추측이 필요한 항목은 본문에 쓰지 않고 §⑧의 `REQUEST:`/`FEEDBACK:` 블록으로 분리했다.
> 코드 경로는 모두 `/home/ykio/notebooks/TSMAE/` 기준. Notion 덤프 2개 (절대경로 — 세션-독립 접근용):
> - **[N-METH]** = "0 MAE 프로젝트 개요" (snapshot 2026-05-31, `/home/ykio/.claude/projects/-home-ykio-notebooks-TSMAE/0aa53593-b13e-47f9-bea0-4e3aa040496f/tool-results/mcp-claude_ai_Notion-notion-fetch-1781093695371.txt`)
> - **[N-COMP]** = "Baseline Comparison: 22 Active Models + 4 Weakly-Supervised · 9 Datasets · 2 Conditions" (snapshot 2026-06-06, `/home/ykio/.claude/projects/-home-ykio-notebooks-TSMAE/0aa53593-b13e-47f9-bea0-4e3aa040496f/tool-results/mcp-claude_ai_Notion-notion-fetch-1781093708082.txt`)
> - metadata 표본 = `results/experiments/271_20260602_020545_271canon_baseline/` (271canon, 이하 [271c])

> **정정 이력 (r2, 2026-06-10, fixer-4)** — 리뷰 `paper/99_reviews/p1_protocol_r1.md` 전수 반영 (상세 처리표: `paper/99_reviews/p1_protocol_fixlog_r2.md`):
> - **B-1**: SMAP/MSL safe-cut 메커니즘 정정 — margin=10은 이동 한계가 아니라 **clearance 요건**, 탐색은 **무제한 outward** (실측: MSL D-16 +166 steps = 7.58%). "분할 비율에 실질 영향 없음" 무근거 일반화 삭제 → 실측 이동량 표로 교체. 오류 출처 `docs/DATASET.md:1151`은 ERRATA로 기록 (§②).
> - **M-1/M-5**: 깨진 레지스트리 인용 교정 (`loaders.py:2810-2812`→`2742`, `2697-2698`→`2698-2699`).
> - **M-2**: §⑤-4의 `baseline_common.py:345` 인용 제거 (NRDetector PU class-prior 주석 — threshold 관행과 무관) → Phase 4 reference 수요로 이관.
> - **M-3**: **best-epoch = test-split pak_auc_f1 선정 (test-set model selection, 전 모델 동일)** 사실을 §④에 명시 등재 + REQUEST-4 신설.
> - **M-4**: 실행 프로토콜 신설 (§④-실행: 단일 run/seed=42, mean 집계, baseline 학습 패리티, 271_CONFIG_TRUTH cross-ref).
> - **MINOR 1–7**: entity 112→113(산식 명기), SOTA legacy 7→6, PSM metrics 149→153키, noisy.py 87→85, evaluator 639→637, train '라벨 부재→정상 취급' 명시, Notion 덤프 절대경로.

> **정정 이력 (r4, 2026-06-11, p5 comprehensive-fixer)** — Phase 5 method-truth 리뷰 `paper/99_reviews/p5_method_truth_r1.md` §5 errata 반영 (처리표: `paper/99_reviews/p5_fixlog_r2.md`):
> - **E-1 (stale stride)**: §④-실행 2항의 "test stride=1이므로" — 1순위 정본 271_CONFIG_TRUTH r4 §VIII(`resolve_test_stride` = `seq_length // 10 − 1` = **49**, `utils/experiment.py:20–43` 재확인)과 모순되는 stale 서술이었음. "test stride=49 → 한 점을 덮는 window 수 ≈ 500/49 ≈ 10"으로 정정.
> - **E-2 (PA%K 격자 명확화)**: §④ 매핑표 `pa_0_f1` 행의 "K 그리드 `PA_K_VALUES = 0,5,…,100` (evaluator.py:831)"은 **per-K 진단 키(`pa_{k}_f1` 등) 전용 격자**다. 보고 지표 **PA%K-AUC(`pak_auc_f1` 등)의 적분 격자는 별도로 `np.arange(0,101)` = K=0,1,…,100 step 1 (101점)** — `evaluator.py:1034` (`compute_pa_k_auc`), docstring `:998` "sweep K=0,1,...,100 and integrate", 적분 `:1271–1282` `np.trapz(..., k_values)/100.0`. 두 격자를 혼동해 AUC 적분 격자를 {0,5,…,100}으로 오독하지 말 것 (p5 원고 B-1 오류의 근원; 271_CONFIG_TRUTH r4 §VIII "in steps of 1"이 정확).

> **정정 이력 (r3, 2026-06-10, fixer-5)** — 재리뷰 `paper/99_reviews/p1_rereview_beta_r2.md` 반영 (처리표: `paper/99_reviews/p1_fixlog_r3.md`):
> - **RB-1 (BLOCKER)**: §④-실행 3항의 baseline epoch-패리티 주장이 사실과 반대였음 — 실측: MAE(271) **500 epochs·eval 5-epoch 간격**([271c] metadata + 271_CONFIG_TRUTH §II) / unsupervised 22종 **10 epochs**(`baseline_common.py` "2026-06-06: unsupervised unified to 10") / weak 4종 **50 epochs**; baseline eval은 매 epoch. "양쪽 50ep 완주" 서술 삭제, 비대칭 사실 + 실제 공통점(best-epoch 기준 pak_auc_f1·주기평가-후-best 구조·early stopping 부재)으로 재작성. r2가 인용한 `config.py:264`(dataclass default)·`baseline_common.py:256/266`(stale docstring)은 인용 금지로 격하. r2 fixlog의 M-4 "재검증" 행(epoch 50 통일)은 오검증 — 본 r3 fixlog가 대체 (구 fixlog는 쓰기 범위 외라 미수정).
> - **RM-1 (MINOR)**: §④-실행 1항 "모든 실험 단일 run" → MAE/deterministic baseline 한정으로 범위 수정 + baseline `random` 5-run mean±std 예외 명시 (`baseline_common.py:757, 786-796`).

---

## ① 데이터셋 구성 (R33: Simulation·Exathlon 제외)

### 논문 포함 데이터셋 (6 계열, 총 **113** dataset entries)

> 산식 (정정 r2): 1 (SWaT) + 2 (WaDi A1/A2) + 1 (PSM) + 28 (SMD) + 54 (SMAP) + 27 (MSL) = **113** 학습 단위. SWaT dual-eval(full/excl22)을 평가 단위로 따로 세면 **114** 평가 단위 (학습은 1회, §⑥).

| Dataset | Entity 수 | Features | Train 길이 | Test 길이 | Train anomaly | Test anomaly | 근거 |
|---|---|---|---|---|---|---|---|
| **SWaT (A1+A2)** | 1 (학습 1회 + dual eval: full/excl22) | **45** (모델 입력 실측; 원본 51 − combined-constant 6 {P202,P401,P404,P502,P601,P603} — 정정(reconciler 2026-06-10): [271c] `config.num_features=45` + checkpoint `patch_embed=(512,450)`=10×45; §⑧ FEEDBACK-7) | **719,959** (A1 495,000 전체 + A2 앞 224,959) | **224,960** (A2 뒤 50%) | 11,757 pts (**1.63%**) | 42,864 pts (**19.05%**); excl22 평가범위에선 3.68% | 분할식: `loaders.py:2018` (`train_len = n_a1 + mid_a2`); 수치: 원 CSV 직접 계산(본 문서 작성 시 검증, A1=495,000행/A2=449,919행); test ratio ↔ [271c] `SWaT/A1A2_full/experiment_metadata.json: metrics.anomaly_ratio=0.19054` 일치 |
| **WaDi A1** | 1 | 123 | **1,296,001** (14days 1,209,601 + attack 앞 86,400) | **86,401** (attack 뒤 50%) | 6,688 pts (0.52%) | 3,297 pts (**3.82%**) | 분할식: `loaders.py:2201` (`train_len = n_14d + n_atk // 2`); 수치: 원 CSV 직접 계산; ↔ [271c] `WaDi/A1` `anomaly_ratio=0.0382` 일치 |
| **WaDi A2** | 1 | 123 (**확정** — §⑧ FEEDBACK-2 RESOLVED: 원본 127 sensor 중 all-NaN 4개 drop) | **870,972** (14days 784,571 + attack 앞 86,401) | **86,402** | 6,635 pts (0.76%) | 3,342 pts (**3.87%**) | 동일 분할식 `loaders.py:2201`; ↔ [271c] `WaDi/A2` `anomaly_ratio=0.0387` 일치 |
| **PSM** | 1 | 25 | **176,401** (orig train 132,481 전체 + test 앞 43,920) | **43,921** (test 뒤 50%) | 10,929 pts (**6.20%**) | 13,452 pts (**30.63%**) | 분할: `loaders.py:1685-1690` (`test_split = len(test_data)//2`); 수치: `dataset/PSM/test_label.csv` 직접 계산; ↔ [271c] `PSM` `anomaly_ratio=0.30628` 일치; 출처 byte-level 검증: [N-COMP] §7.1 "train 132,481 + test 87,841 × 26 cols / anomaly 27.76% — 로컬과 byte-level 일치" |
| **SMD** | **28 machines** (per-machine 독립 실험 → 평균 보고) | raw 38; constant-col 제거 후 machine별 **29–36** (정정(reconciler 2026-06-10): [271c] 22/28 machine metadata `num_features` 실측 범위 — 최소 29(machine-3-10), 최대 36(machine-3-3); 초판의 "32–38"은 metadata 근거 없음) | machine별: orig train 전체 + test 앞 50% | machine별: test 뒤 50% | machine별 상이 | machine별 상이 (예: [271c] `SMD/machine-1-4` `anomaly_ratio=0.0363`) | machine 목록(28개): `loaders.py:864-875` `SMD_MACHINE_NAMES`; 분할: `loaders.py:1152-1157`; 공식 전체 규모: [N-COMP] §2.1 "28 machines × 38 features / train 708,405 / test 708,420 / anomaly 4.16% — 로컬과 byte-level 일치" (Su et al. KDD 2019, OmniAnomaly repo, MIT) |
| **SMAP** | **54 channels** (per-channel = Pattern B; concat = Pattern A) | 25 (1 telemetry + 24 binary command) | channel별: orig train + test 앞 ~50% (safe-cut) | channel별: test 뒤 ~50% | concat 합계 0.70% | concat 합계 24.54% | channel 목록(54): `loaders.py:2503-2514`; 분할: `loaders.py:2592-2595` (50% 지점 safe-cut — 정확한 메커니즘·실측 이동량은 §② 표 참조; SMAP은 이동 0/54 실측); Pattern A 합계: [N-COMP] §2.1 "total 573,830 / train 355,905 / test 217,925 (train_ratio 0.6202), train anomaly 2,499 (0.70%) / test anomaly 53,473 (24.54%)"; P-2 채널 CSV 중복 → UNION 처리 `loaders.py:2555-2562` |
| **MSL** | **27 channels** | 55 (1 telemetry + 54 binary command) | 〃 | 〃 | concat 합계 1.70% | concat 합계 16.72% | channel 목록(27): `loaders.py:2516-2524`; Pattern A 합계: [N-COMP] §2.1 "total 132,046 / train 95,271 / test 36,775 (train_ratio 0.7215), train anomaly 1.70% / test anomaly 16.72%"; safe-cut moved 4/27 채널 (D-16, M-1, M-2, S-2 — 이동량 실측 §② 표: max +166 steps) |

- **데이터 출처·라이선스** (Phase 4 인용 입력, [N-COMP] §7.1–7.3에서 DOI 검증 완료):
  SWaT = Goh et al. CRITIS 2016 (DOI 10.1007/978-3-319-71368-7_8, iTrust 신청제) / WaDi = Ahmed et al. CySWATER 2017 (DOI 10.1145/3055366.3055375) / SMD = Su et al. KDD 2019 (DOI 10.1145/3292500.3330672, MIT) / PSM = Abdulaal et al. KDD 2021 (DOI 10.1145/3447548.3467174, CC BY 4.0) / SMAP·MSL = Hundman et al. KDD 2018 (DOI 10.1145/3219819.3219845, Telemanom data.zip — Wayback 2022-10-16 snapshot, `loaders.py:2641-2643`).
- **정규화**: 전 데이터셋 per-feature min-max, **train 구간만으로 fit** (leak-free). multi-entity(SMD/SMAP/MSL concat)는 **entity별 독립 fit** (`docs/DATASET.md:1159`, 2026-06-02 수정). 단일 entity(SWaT/WaDi/PSM)는 whole-array fit ≡ per-entity (동치).

### R33: Simulation·Exathlon은 논문 미포함 (명시)
- 두 데이터셋은 **코드/파이프라인에는 존재**한다: Simulation `run_base_experiments.py:256-262`, Exathlon `run_base_experiments.py:324-329, 366-375` (apps {1,2,4,5,6,9}, `loaders.py:1373`).
- 그러나 R33 원문("Simulation 데이터셋 및 exathlon 데이터셋은 포함되지 않을 예정임")에 따라 **논문 실험 표에서 제외**한다.
- 영향: [N-COMP] §3.4의 기존 RankAvg는 "6 dataset (SWaT excl22, WaDi A1, WaDi A2, SMD, PSM, **Exathlon**)" 기준이므로, 논문용 집계는 **Exathlon 열을 빼고 SMAP/MSL을 더해 재계산**해야 한다 (§⑧ FEEDBACK-3).

---

## ② Main split 프로토콜 (R13) — "테스트를 길이 기준 반반, 앞 50%를 train에 편입"

### 코드 구현 (전 데이터셋 동일 규칙 — file:line 전수)

공통 패턴: **Train = [원본 train 파일(전부 정상) | 원본 test 파일의 앞(시간상 과거) 50%]**, **Test = 원본 test 파일의 뒤(시간상 미래) 50%**.

- '전부 정상'의 정확한 성격 (r2 추가): SMAP/MSL은 코드가 train 라벨을 **명시적 zeros**로 부여 (`loaders.py:2602-2604`); PSM/SMD는 **train 라벨 파일 자체가 부재**하여 정상으로 *취급* — 분야 표준 가정 (PSM: `loaders.py:1672-1675` "Load train file (all normal)" + `np.zeros`; SMD: `loaders.py:1139-1142` 동일 패턴). 논문에는 "train 라벨 부재 → 정상 취급(분야 표준 가정)" 1줄 명시 권장.

| Dataset | 분할 구현 | 핵심 라인 |
|---|---|---|
| SWaT | `load_swat_a1a2_raw` (`DATASET_LOADERS['SWaT_A1A2']`, `loaders.py:2690`) | `loaders.py:2005,2018`: `mid_a2 = n_a2 // 2; train_len = n_a1 + mid_a2` |
| WaDi A1/A2 | `load_wadi_14days_raw` (`DATASET_LOADERS['WaDi_14days_A1/_A2']`, `loaders.py:2698-2699`) | `loaders.py:2201`: `train_len = n_14d + n_atk // 2` |
| SMD (28) | `load_smd_simple` (`SMD_simple_<machine>` alias, `loaders.py:2742`) | `loaders.py:1153`: `test_split = len(test_data) // 2`; concat `loaders.py:1159-1161`: `[train_file | test_front | test_back]` |
| PSM | `load_psm` | `loaders.py:1686-1693`: 동일 `// 2` 분할 |
| SMAP/MSL (81) | `_load_smap_msl_simple_single` | `loaders.py:2592-2595`: `target_cut = len(test_arr) // 2` 후 `_find_safe_cut_point(..., margin=10)` (`loaders.py:1050-1101`) — **(정정 r2)** 50% 지점이 **어떤 anomaly region의 ±10 timestep 이내**이면 (`s - margin <= pos <= e + margin`, `:1071-1073` — region 내부뿐 아니라 경계 ±10도 발동) **가장 가까운 안전 지점으로 cut을 이동**. `margin=10`은 **이동 한계가 아니라 clearance 요건** ("A 'safe' position is at least `margin` timestamps away from any anomaly region", docstring `:1053`)이고, 탐색은 target에서 **무제한 outward** (`for offset in range(1, L): for candidate in [target-offset, target+offset]`, `:1080-1083`) — 긴 region 안/근처면 region 전체 + 10 clearance를 벗어날 때까지 얼마든지 멀리 이동한다 |

**SMAP/MSL safe-cut 실측 이동량** (2026-06-10, `_find_safe_cut_point`를 실제 채널 라벨로 직접 호출, 81채널 전수 — r1 리뷰 실측치 bit-exact 재현 확인):

| 채널 | test 길이 | target(50%) | 실제 cut | 이동량 | 해당 채널 test 길이 대비 |
|---|---|---|---|---|---|
| **SMAP (54채널 전부)** | — | — | — | **0** (이동 없음) | 0% |
| MSL D-16 | 2,191 | 1,095 | 1,261 | **+166 steps** | **7.58%** |
| MSL M-1 | 2,277 | 1,138 | 1,099 | −39 | 1.71% |
| MSL M-2 | 2,277 | 1,138 | 1,099 | −39 | 1.71% |
| MSL S-2 | 1,827 | 913 | 921 | +8 | 0.44% |
| MSL 나머지 23채널 | — | — | — | 0 | 0% |

- 함의: **Pattern B(per-channel) 실험에서 MSL D-16은 test가 166 step 짧아진(=train이 그만큼 길어진) split**이다. "±10이라 실질 영향 없음"(초판 서술)은 사실과 다름 — 정확한 서술은 "81채널 중 4채널(전부 MSL)만 이동, 최대 +166 steps(D-16, 해당 채널 test의 7.58%), 나머지 3채널 |Δ|≤39 steps; Pattern A(concat) 규모에서는 MSL 전체 test 36,775 대비 합계 이동 |Δ| 252 steps로 미미".
- "전 81채널 boundary-straddling anomaly 0건"은 여전히 참 — 메커니즘이 보장(안전 지점 = 모든 region에서 ≥10 step 이격)하며, 실측에서도 81채널 모두 안전 지점을 찾았다(`:1085-1101` fallback 미발동).
- ⚠️ **ERRATA (오류 출처 기록)**: `docs/DATASET.md:1151`("pushed outside any anomaly region by ±10 timestamps")과 [N-COMP] §2.1의 "margin=10으로 모두 회피"는 ±10이 **이동 한계**인 것처럼 읽히는 부정확한 표현이며, 본 문서 초판이 이를 그대로 수용했던 것을 r2에서 정정. 코드 자체 주석 `loaders.py:2591`("pushed outside anomaly regions ±margin")도 같은 모호함이 있으나 **함수 본문(`:1050-1101`)이 ground truth**. (docs/DATASET.md 수정은 본 phase 범위 외 — 코드/기존 docs는 read-only.)

- **시간 순서 보존**: 모든 loader가 *앞*(front) 절반을 train에, *뒤*(back) 절반을 test에 둔다 — "시간적으로 더 뒤쪽 데이터를 test로" (R13 원문)와 일치. swap variant(`load_swat_combined_swap` 등)는 존재하나 **비활성** (`run_base_experiments.py:331-333`: "비활성 variant들… 2026-05-17 자로 default DATASETS에서 제외").
- **기록 경계 보호**: orig_train ↔ test_front는 시간적으로 인접하지 않으므로 `run_boundaries`로 표시하여 sliding window가 경계를 넘지 않게 한다 (`loaders.py:1205-1207` (SMD), `1738-1740` (PSM), `2613-2614` (SMAP/MSL), `2026-2032` (SWaT)).

### 논문 서술 재료 (R13 원문 논리 그대로)
1. **동기**: 기존 시계열 이상탐지 벤치마크는 train에 anomaly가 없는 구성이 대부분 → 알려진(labeled) 이상을 학습에 반영하는 본 모델의 설정을 평가할 수 없음. 이를 위해 **원본 test를 길이 기준 반반으로 나눠 앞 50%를 train에 편입** — train에 라벨된 anomaly가 실제로 존재하게 됨 (실측: SWaT train anomaly 1.63%, WaDi A1 0.52%, A2 0.76%, PSM 6.20%, SMAP 0.70%, MSL 1.70% — §① 표).
2. **공정성 (정정 r2)**: 분할 규칙 자체(`// 2`)는 데이터셋별 취사선택 없이 **전 데이터셋 통일**. 단 SMAP/MSL은 50% 지점이 anomaly region을 침범(region 내부 또는 경계 ±10 이내)하는 경우 **가장 가까운 안전 지점으로 cut을 이동하는 경계 조정**(최소 10-step clearance, 탐색 거리 무제한 — §② 표)이 있다. 영향 범위 실측: 81채널 중 **4채널(전부 MSL)만 이동, 최대 +166 steps (D-16, 해당 채널 test의 7.58%)**, 나머지 3채널 |Δ|≤39 steps, SMAP 0건. 논문에는 규칙(//2) + 경계 조정 메커니즘 + **실측 이동량**을 함께 공개할 것 — "실질 영향 없음" 류의 무근거 일반화 금지 (D-16 단일 채널 관점에서는 7.58%로 작지 않음; concat 규모에서는 미미).
3. **시간성**: 미래 데이터(뒤 50%)로만 평가 → 온라인 운영 상황과 부합, look-ahead 없음.
4. **이 설정에서 기존 unsupervised의 최선** = 알려진 이상을 train에서 제거하고 순수 정상으로 학습 (→ §③).

---

## ③ 비교군 label 활용 (R12/R31) — "unsupervised의 최선 = 알려진 이상 제거"

### 구현: `normalonly` variant (실재, 검증됨)
- **단일 구현점**: `comparison/data/unified_loader.py:392-485` `_apply_normalonly()` —
  - train 범위 안의 모든 anomaly region을 절제(excise)하고 (`unified_loader.py:410-415`),
  - 남은 정상 구간들을 `NormalSegment` 리스트로 보존 (`unified_loader.py:434-463`),
  - **절제 지점 + `run_boundaries`를 segment 경계로 등록**하여 sliding window가 비인접 구간을 가로지르지 않게 함 (`unified_loader.py:417-421`, segment-aware windowing: `create_windows_from_segments` `unified_loader.py:603+`).
  - 결과 train 라벨은 전부 0 (`unified_loader.py:470`).
- **variant 선언**: `unified_loader.py:34-36` — "'normalonly': Remove anomaly regions from training data".
- **실험 조건 매핑** ([N-COMP] §2.2 원문):
  - **Q1 (minmax full)**: "train data 안에 anomaly 포함 (실제 운영 환경 가정)" — 라벨 미사용 unsupervised 그대로.
  - **Q3 (minmax normalonly)**: "train data에서 anomaly region 제거 → segment-aware concat" — **라벨을 '제거'라는 형태로 활용한 unsupervised의 최선** (R12).
  - 각 baseline 실험은 두 조건 모두 등록: `comparison/experiment_configs.py:65-66, 85-86, 108-110, 156-157, 243-244, 271-272, 294` 등 (`"variant": "normalonly"`).
- **비교군 22개 모델 목록**: `comparison/experiment_configs.py:24-30` `STANDARD_BASELINES` — simple 5 (random, sensor_range, pca_error, l2_norm, nn_distance) + neural 3 (mlp, mlpmixer, transformer) + GCN-LSTM 1 + **SOTA legacy 6** (anomaly_transformer, tranad, usad, dagmm, gdn, omnianomaly — 정정 r2: 초판 "7"은 오기, 6이어야 5+3+1+6+7=22로 총계 정합) + SOTA new 7 (tfmae, timesnet, dcdetector, memto, moderntcn, catch, npsr).
- **TSMAE(본 모델)의 라벨 활용** (대조축): force-mask-anomaly `config.py:315-319`, GRL anomaly classifier `config.py:126-134` (window-mode), dynamic-margin anomaly loss는 GRL 활성 시 비활성 `config.py:123-125`. [N-METH] C4 원문: "학습 셋에 소량 존재하는 anomaly 라벨을 적극 활용하는 디자인… 본 모델을 semi-supervised setting으로 만드는 결정적 요소."

### R31 방어 논리 재료 (근거 포함)
1. **라벨을 활용하는 기존 시계열 이상탐지 모델 자체가 희소**: 코드베이스에 포팅된 weakly-supervised 비교군은 단 4종 (`deepmil`, `wetas`, `treemil`, `nrdetector` — `comparison/experiment_configs.py:36-44` `WEAK_SUPERVISED_BASELINES`), 이들은 **Q1 전용** (normalonly에선 train 라벨이 전부 0이라 구조적으로 실행 불가; [N-COMP] §3 callout: "Q3 = N/A는 placeholder가 아니라 조건 부적합(구조적)"). 상태: "구현 완료 · CPU dry-test 통과 · GPU 전체 실험 미실행" ([N-COMP] §6.4).
2. **unsupervised 비교군에게 라벨의 '최선의 사용법'을 제공**: unsupervised 방법론에서 train anomaly는 성능을 떨어뜨리는 오염원 → 라벨로 그것을 제거(Q3 normalonly)해 주는 것이 그들에게 가장 유리한 라벨 활용 (R12 원문 논리). 즉 비교는 "라벨 있는 우리 vs 라벨 없는 그들"이 아니라 "**같은 라벨을 각자의 패러다임에서 최선으로 쓴** 비교".
3. **동일 split·동일 평가 코드**: 모든 baseline의 metric은 MAE와 같은 단일 함수로 계산 — `comparison/baseline_common.py:5` "All metric computation uses mae_anomaly/evaluator.py directly — no duplicate code"; `baseline_common.py:553` "`mae_anomaly.evaluator.compute_full_metric_set` — the EXACT function the MAE (pipeline uses)". 데이터 로딩도 MAE raw loader 직접 사용 (`unified_loader.py:4-11`).
4. (보강) 추가로 Q1(라벨 전혀 안 쓴 full train) 조건도 병렬 제시 가능 — [N-COMP] §4 (현재 pending).

---

## ④ 평가지표 (R29/R24) — 내부 변수명 ↔ 정식 학술 명칭 매핑

### 매핑 표 (논문에는 반드시 우측 정식 표기 사용 — R24)

| 내부 키 (코드/metadata) | 정식 학술 명칭 (논문 표기) | 제안 논문 (Phase 4 검증 입력) | 계산 위치 |
|---|---|---|---|
| `vus_roc` | **VUS-ROC** (Volume Under the ROC Surface) | J. Paparrizos, P. Boniol, T. Palpanas, R. S. Tsay, A. Elmore, M. J. Franklin, "Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection," *PVLDB* 15(11), 2022, DOI 10.14778/3551793.3551830 | `evaluator.py:736-743` (공식 `vus` 패키지 `vus.metrics.get_metrics`, slidingWindow=100, min-max 정규화 후 — 공식 예제 그대로) |
| `vus_pr` | **VUS-PR** (Volume Under the Precision-Recall Surface) | 〃 | 〃 |
| `pak_auc_f1` | **PA%K-AUC F1** — PA%K 프로토콜 하 F1을 K=0…100 전 구간 적분(trapezoidal, [0,1] 정규화)한 값; per-K threshold 재최적화(tadpak 방식) | PA%K: S. Kim, K. Choi, H.-S. Choi, B. Lee, S. Yoon, "Towards a Rigorous Evaluation of Time-Series Anomaly Detection," *AAAI* 2022, vol. 36 no. 7, pp. 7194–7201, DOI 10.1609/aaai.v36i7.20680 (구현: github.com/tuslkkk/tadpak) | `evaluator.py:990-1022` (`compute_pa_k_auc`; "Following Kim et al. (AAAI 2022 …), tadpak implementation" `evaluator.py:1003-1005`); PA%K 조정 자체: `evaluator.py:591-634` (paper Eq. 충실: strict `>`, OR-semantics) |
| `pak_auc_prc_auc` (R29의 "pak_auc_pr") | **PA%K-AUC AUC-PR** — PA%K 조정 후 threshold sweep으로 얻은 AUC-PR을 K 전 구간 적분 | 〃 | `evaluator.py:497-588` (per-K), `990+` (적분) |
| `affiliation_f1` (R29의 "affiliated-f1") | **Affiliation F1** (Affiliation precision/recall의 조화평균; Aff-P / Aff-R) | A. Huet, J. M. Navarro, D. Rossi, "Local Evaluation of Time Series Anomaly Detection Algorithms," *KDD* 2022, arXiv:2206.13167 (공식 구현 github.com/ahstat/affiliation-metrics-py) | `evaluator.py:693-695` (출처 주석), `_compute_threshold_dependent` 경유 `evaluator.py:747-748`; AR-threshold 변형 `affiliation_f1_ar`: `evaluator.py:809-813` |
| `pa_0_f1` (R29의 "Pa_f1") | **Point-Adjusted F1 (PA F1)** — 전통적 point adjustment: 한 segment에서 1점이라도 탐지되면 segment 전체 탐지 처리 | PA 프로토콜 원전: H. Xu et al., "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications," *WWW* 2018, DOI 10.1145/3178876.3185996 | `evaluator.py:609-611` — "K=0: ratio > 0 ≡ sum ≥ 1 → conventional PA (Xu et al. WWW 2018)"; **per-K 진단 키(`pa_{k}_*`) 전용** K 그리드 `PA_K_VALUES = 0,5,…,100` `evaluator.py:831` — ⚠️ (r4) 이것은 진단 출력 격자일 뿐, **PA%K-AUC 보고 지표의 적분 격자는 K=0,1,…,100 step 1** (`np.arange(0,101)`, `evaluator.py:1034`; docstring `:998`; 적분 `:1271–1282`) — 두 격자 혼동 금지 |
| (참고) `r_based_f1` | Range-based F1 | N. Tatbul, T. J. Lee, S. Zdonik, M. Alam, J. Gottschlich, "Precision and Recall for Time Series," *NeurIPS* 2018 | `evaluator.py:696-698` |
| (참고) `f1_t` | time-series F1 (TimeSeAD/QuoVadisTAD 계열 보조지표) | — | `evaluator.py:133+, 181+` |

웹 재검증 (2026-06-10): VUS — [ACM DL PVLDB 15(11)](https://dl.acm.org/doi/abs/10.14778/3551793.3551830), [공식 repo](https://github.com/TheDatumOrg/VUS); Affiliation — [arXiv:2206.13167](https://arxiv.org/abs/2206.13167), [공식 구현](https://github.com/ahstat/affiliation-metrics-py); PA%K — [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/20680), [tadpak](https://github.com/tuslkkk/tadpak); PA — [ACM DL WWW'18](https://dl.acm.org/doi/10.1145/3178876.3185996).

### 상호보완성 논리 재료 (각 지표가 보는 관점)
- **VUS-ROC / VUS-PR**: threshold-free + 라벨 경계 buffer를 둔 연속 평가 → **threshold 선택과 라벨 경계 불확실성 양쪽에 강건한 ranking 품질**. VUS-PR은 클래스 불균형(이상 비율 3.7~30.6%, §①)에 강건한 PR 관점.
- **PA%K-AUC (F1 / AUC-PR)**: 이벤트(segment) 수준 탐지 관용도를 K로 매개변수화하고 **K 전 구간 적분으로 특정 K 취사선택(cherry-picking)을 제거** — K=0(관대한 PA)과 K=100(엄격한 point-wise) 사이 전 스펙트럼의 평균 성능. F1 변형은 운영점 품질, AUC-PR 변형은 threshold-robust 변형.
- **Affiliation F1**: 예측-실제 이벤트 간 **시간적 근접도(거리) 기반·이벤트별 지역(local) 평가** — counting 기반 지표가 못 보는 "얼마나 가까이 맞췄나"를 보고, adversarial scoring에 이론적 강건성 보장(Huet et al.의 명시적 설계 목표).
- 즉: VUS(threshold-free 연속 ranking) ⊥ PA%K-AUC(이벤트 관용도 스펙트럼 적분) ⊥ Affiliation(시간 근접도 local 평가) — 세 축이 서로 다른 실패 양상을 잡는다.
- **PA F1 (`pa_0_f1`)의 문제점 (R29: 제시하되 참고하지 않음을 명시)**: Kim et al. AAAI 2022가 입증 — *"PA protocol has a great possibility of overestimating the detection performance; even a random anomaly score can easily turn into a state-of-the-art TAD method"* (웹 재확인, AAAI OJS abstract). 본 코드도 같은 입장으로 PA를 K-스펙트럼의 한 점(K=0)으로만 취급 (`evaluator.py:609-611`). 논문 서술: 선행연구 비교 가능성을 위해 PA F1을 표에 제시하되, 위 근거로 순위 판단에는 사용하지 않음.
- **단일 진실 원천**: 위의 모든 지표는 MAE·baseline 공통의 `compute_full_metric_set` (`evaluator.py:864-987`, "SINGLE SOURCE OF TRUTH … Both pipelines call this" `evaluator.py:874-878`)에서 계산된다.
- **Best-epoch 선정 기준**: `pak_auc_f1` (`config.py:291` `best_epoch_metric: str = 'pak_auc_f1'`, [N-COMP] §3 callout "Best-epoch 기준: pak_auc_f1. 모든 metric 같은 best epoch에서 추출").
- **⚠️ 중요 프로토콜 사실 (r2 추가, M-3): per-epoch 평가는 test split 위에서 수행되며, best epoch도 test 지표로 선정된다 — 즉 test-set model selection이다 (전 모델 동일 조건).**
  - MAE: `Evaluator`는 생성자에서 **test_loader/test_dataset만** 받고 (`evaluator.py:1363-1373`), `evaluate()`는 그 위에서 전 지표를 계산한다 (`evaluator.py:2155-2160`). 학습 중 per-epoch callback이 `config.best_epoch_metric`(=pak_auc_f1)의 test-split 값으로 best checkpoint를 갱신하고 (`run_base_experiments.py:2604, 2645-2646`), 학습 종료 후 최종 best epoch도 동일 per-epoch test 지표 스캔으로 확정한다 (`run_base_experiments.py:3215-3240`).
  - Baseline: 동일 — per-epoch 추론+평가 (`baseline_common.py:949+` "per-epoch inference + synchronous CPU eval", `eval_interval=1`) 후 "best-epoch-by-`pak_auc_f1` selection" (`baseline_common.py:1368` docstring; excl22는 독립 스캔 `baseline_common.py:2087-2098`).
  - 별도 validation split은 존재하지 않는다 (§①·② 분할은 train/test 2-way).
  - **전 모델(MAE+22 baseline)에 동일 적용이므로 비교 공정성은 유지**되지만, 일반화 성능 추정으로는 낙관적 편향 가능성이 있는 설정이다 — 논문 experiments 섹션에 **반드시 공개**해야 할 프로토콜 사실 (숨기면 리뷰어 단골 공격 지점). 서술 방식은 → §⑧ REQUEST-4.

### 실행 프로토콜 (r2 신설, M-4 — 논문 experiments 섹션 필수 기재 사항)

1. **반복 수 / seed 정책 (r3 한정, RM-1 — r2의 "모든 실험"은 과대 일반화)**: **MAE 실험(run_base_experiments) 및 random 제외 baseline**은 dataset entry당 **단일 run** (MAE 파이프라인에 반복·multi-seed 루프 없음 — `run_base_experiments.py` 전수 grep: repeat/n_runs/num_seeds 0건). **예외: baseline `random`은 5회 독립 run → mean±std** — `baseline_common.py:757` `n_runs = 5 if model_name == 'random' else 1`, 집계 `:786-796` (mean이 보고값, std·`per_run_metrics` 병행 저장; [N-COMP] §1.2 random preset seed=None 5-run — NOTION_DIGEST II-2b 기재와 정합). MAE seed는 `config.random_seed = 42` 고정 (`config.py:322`); 학습 시작 시 `set_seed(config.random_seed)` 호출 (`run_base_experiments.py:2435`; python/numpy/torch/cuda 전부 시드 — `config.py:326-333`), 데이터셋·마스킹·DataLoader generator에도 동일 seed 전파 (`run_base_experiments.py:2442, 2509, 2522, 2542`). → **MAE(및 deterministic baseline)는 run 간 분산/신뢰구간 보고 불가** — 논문에 단일-seed(42) 사실을 명시할 것 (수치 발명 금지; random만 mean±std 보고 가능).
2. **Window score → point score 집계 (모든 지표의 입력이 되는 결정적 단계)**: **mean 집계** — 각 timestep의 점수 = 그 timestep을 덮는 모든 (window, patch) 쌍의 patch score **평균**. 구현: `aggregate_patch_scores_to_point_level` (`evaluator.py:295-304`, `method='mean'` 기본 `evaluator.py:272, 302`; mean 산식 `evaluator.py:278-280` bincount-합/coverage). 모듈·evaluate() docstring 모두 명시: "All metrics (including PA%K) use mean-aggregated point-level scores" (`evaluator.py:8, 2158`). test stride=**49**(271_CONFIG_TRUTH r4 §VIII: `resolve_test_stride` = `seq_length // 10 − 1`, `utils/experiment.py:20–43`)이므로 한 점을 다수 window(≈ 500/49 ≈ 10개)가 덮는 mean-ensemble 구조 (window/patch 파라미터 정본: 271_CONFIG_TRUTH.md). *(r4 정정 — 구판의 "test stride=1" 은 stale)*
3. **Baseline 학습 설정 — epoch 수·eval 간격은 패리티가 아니라 비대칭 (r3 정정, RB-1 — r2의 "MAE 50 = baseline 50, 양쪽 모두 50 epoch 완주" 패리티 주장은 사실과 반대였음)**:
   - **① epoch 수 (3단 비대칭)**: MAE(271) = **500 epochs** ([271c] metadata `config.num_epochs=500` — PSM/SWaT full/WaDi A2 직접 조회, 2026-06-10 재확인; 정본 cross-ref `271_CONFIG_TRUTH.md` §II) / **unsupervised baseline 22종 = 10 epochs** (`baseline_common.py:272, 279, 286, 297, 300-302, 308, 314, 323` 등 — 주석 원문 "**2026-06-06: unsupervised unified to 10**") / **weakly-supervised 4종 = 50 epochs** (`baseline_common.py:333, 337, 355, 367, 384` — "2026-06-06: weak unified to 50"). ⚠️ 인용 금지 2건: `config.py:264`의 `num_epochs=50`은 dataclass **default일 뿐 exp271 값이 아니며**, `baseline_common.py:256/:266`의 "epochs=50 user override" docstring/주석은 2026-06-06 통일 **이전의 stale 기록**(바로 아래 실값 10과 모순). `comparison/configs/`의 `sota_epochs=50` override는 weak-SSL 큐 한정 (grep 확인).
   - **② eval 간격 (비대칭)**: MAE = **5 epoch 간격** ([271c] metadata `eval_interval=5`; 실구동 스크립트 상수 `EVAL_INTERVAL=5`, `run_base_experiments.py:94`) vs baseline = **매 epoch** (`eval_interval: int = 1` 기본값, `baseline_common.py:943`; per-epoch 추론+평가 `baseline_common.py:949+`).
   - **③ 실제 공통점 (패리티가 성립하는 부분)**: ⓐ best-epoch 선정 기준 동일 — `pak_auc_f1` (`baseline_common.py:1368`); ⓑ 주기 평가(per-epoch 또는 5-epoch 간격) 후 best-epoch 선택이라는 구조 동일 (위 M-3 항 — test-split selection도 동일); ⓒ early stopping 양쪽 부재 → 각자 설정된 epoch 수 완주 (MAE warmup early-stop은 opt-in·기본 off — `config.py:286`, [271c] `use_teacher_warmup_early_stop=False`; baseline_common에 early stopping 부재, grep 0건).
   - **④** 모델별 하이퍼파라미터는 원 구현 충실 원칙으로 상이할 수 있음 (예: NRDetector `win_size=100` 유지, `baseline_common.py:343` "win_size=100 kept (NOT pipeline 500)") — 개별 값의 정본은 `baseline_common.py` MODEL_CONFIGS.
   - **논문 기재 사항 (사실만)**: epoch 수(500 / 10 / 50)와 eval 간격(5 vs 1)의 비대칭은 experiments 섹션에 **그대로 명시 공개**해야 한다 — 각 모델은 자기 주기 평가에서 선정된 best epoch에서 평가된다(위 ⓐⓑ). 이 차이에 대한 해석·정당화 서술(수렴 특성 등)은 본 문서 범위 밖 — Phase 3/5 결정 사항.
4. **Cross-reference**: MAE 학습 하이퍼파라미터(window=500, patch, stride, lr, batch 등)의 정본은 **`paper/01_research_understanding/271_CONFIG_TRUTH.md`** — §①의 split 표와 윈도잉은 불가분이므로 본 문서와 함께 읽을 것.

---

## ⑤ Threshold 프로토콜 (R30) — "test anomaly 비율 threshold"

### 코드 구현 (실재, file:line)
- `evaluator.py:752-815` `compute_ar_threshold_metric_set`:
  - `ar = float(y.mean())` — **test 평가범위 라벨의 anomaly 비율** (`evaluator.py:785`);
  - `ar_th = np.quantile(s, 1.0 - ar)` — score의 **(1 − anomaly_ratio) 분위수**를 threshold로 (`evaluator.py:790`);
  - `pred = (s > ar_th)` — strict `>` (Kim et al. AAAI 2022 Eq. 1 충실, `evaluator.py:793-794`).
- 산출 키(`_ar` 접미사): `f1_ar, precision_ar, recall_ar, f1_t_ar, precision_t_ar, recall_t_ar, affiliation_{precision,recall,f1}_ar, r_based_f1_ar` + `anomaly_ratio, anomaly_ratio_threshold` (`evaluator.py:764-768, 818-828`).
- **PA%K 계열·prc_auc·vus는 AR threshold로 재계산하지 않음** — "threshold-free or K-integrated → independent of single-threshold choice" (`evaluator.py:769-770`). 즉 R30의 1차 방어("threshold랑 무관한 지표를 같이 제시")가 코드 구조에 이미 내장.
- excl22 평가에서도 동일 함수가 masked span 위에서 호출됨 (`evaluator.py:984-985` "mask-aware: same masked span as core metrics").
- 실측 예 ([271c] `PSM/experiment_metadata.json`): `anomaly_ratio=0.30628`, `anomaly_ratio_threshold=0.001744`, `f1_ar=0.7616`, `affiliation_f1_ar=0.8012`.
- 주의: 같은 metric set에는 **F1-최적 threshold 기반의 기본 키**(`f1_score`, `affiliation_f1` 등 — `evaluator.py:928-931`: ROC curve에서 F1-optimal 지점 선택)도 병존한다. AR 변형 docstring이 명시하듯 optimal-F1 threshold는 "leaks ground truth into threshold choice" (`evaluator.py:761-762`) — **논문 본문 threshold-dependent 수치는 `_ar` 계열을 사용해야 R30과 정합** (§⑧ REQUEST-1).

### 방어 논리 재료 (R30 원문 + 코드 근거)
1. **threshold-무관 지표 병행 제시**: 논문 5지표 중 vus_roc/vus_pr(threshold-free), pak_auc_f1/pak_auc_pr(threshold-sweep + K 적분)은 단일 threshold 선택과 무관 (`evaluator.py:769-770`). threshold가 필요한 것은 affiliation-F1·PA F1 제시값뿐.
2. **평가 protocol일 뿐**: AR threshold는 모델 학습/선택에 개입하지 않음 — best-epoch 선정은 `pak_auc_f1`(threshold 적분형, `config.py:291`)이고, AR threshold는 사후 평가 단계에서만 계산된다 (`compute_full_metric_set` 마지막 단계, `evaluator.py:984-985`). **(r2 주의)** 단, 이 방어를 쓸 때는 best-epoch 선정 자체가 **test-split 지표**로 이뤄진다는 §④ 명기 사실(전 모델 동일)을 함께 공개해야 정직한 서술이 된다 — "AR threshold는 학습에 개입 안 함"과 "model selection이 test 지표 기반"은 별개의 사실이며 후자를 숨기면 안 됨.
3. **전 모델 동일 적용**: baseline도 같은 `compute_full_metric_set` 경유 (§③-3) → 특정 모델에 유리한 threshold 선택 여지 없음.
4. (문헌 관행 — **근거 보류, r2 정정**) "test anomaly-ratio 기반 thresholding은 TSAD 문헌의 표준 관행"이라는 주장 자체는 성립 가능하나, **현재 코드베이스 안에 이를 뒷받침하는 적합한 근거가 없다**. 초판이 인용한 `baseline_common.py:345`는 NRDetector(weak-supervised)의 **PU class-prior 추정**에 관한 주석(`:343-347` — "prior=None → estimated dynamically from train wlabel rate (PU class prior = intrinsic anomaly ratio; dataset-dependent → must be estimated, NOT a fixed constant)")으로, **test score thresholding 관행과 무관**하여 인용을 제거한다. 학술 근거(ratio-based threshold를 사용하는 선행연구 선례 — 예: OmniAnomaly/USAD 계열의 관행 여부)는 **Phase 4 reference 검증 수요로 이관** — 실제 문헌 확보·검증 전까지 논문에서 이 방어 논리를 사용하지 말 것.

---

## ⑥ SWaT '22번 이상 영역' (R28) — excl22의 정의·구현·두 조건의 관계

### 영역의 정의 (실측 검증 완료)
- **SWaT attack ID 22**: test split(A2 뒤 50%) 안의 14개 anomaly region 중 **시간순 첫 번째** region. test-local 좌표 **[2869, 38769), 길이 35,900 pts** = **test anomaly 점들의 83.75%** = **test 전체 길이의 15.96%** (원 CSV 직접 계산 — `evaluator.py:2302-2306` docstring 수치 "~[2869, 38769), length ~35,900 (~84% of all test anomaly points)" 와 정확히 일치).
- **식별 코드**: `evaluator.py:2299-2327` `find_swat_region_22` — "hard-coded identification, no heuristic": 시간순 첫 region + sanity check 길이 ≥ 30,000 (다른 어떤 지원 데이터셋에도 이 크기의 단일 region 없음 → 비-SWaT에선 None 반환, `evaluator.py:2309-2311`).

### 구현: 단일 학습 + dual evaluation
- **학습은 1회**, 평가만 2조건: `run_base_experiments.py:251` "SWaT는 단일 학습 + dual eval (full + excl_region22)"; full worker spawn `run_base_experiments.py:3446-3455`, excl22 worker spawn `run_base_experiments.py:3457-3511` (checkpoints/epoch_scores는 symlink 공유, `3466-3471`).
- **excl22 metric 계산**: `evaluator.py:2334-2366` `compute_metrics_with_exclusion` — `eval_mask[excl_region.start:end] = False` 후 동일 `compute_full_metric_set` 호출; region 22는 region 목록에서도 제거 (`evaluator.py:2361-2363`). VUS/Affiliation도 mask된 span 위에서 계산 ("for excl22 this stops region-22 (84% of SWaT anomalies) leaking back in", `evaluator.py:976-979`).
- **excl22는 자체 best epoch 보유**: excl22 worker가 epoch_scores를 독립 스캔 (`run_base_experiments.py:3501`).
- **학습 중 동시 계산**: `also_excl22 = 'SWaT' in dataset_key` (`run_base_experiments.py:749`), `Evaluator.evaluate(also_excl22=…)` `evaluator.py:2155-2224` (`excl22_*` 접두 키 + `excl22_region_start/end/length` 기록).
- **baseline도 동일 dual 조건**: `comparison/experiment_configs.py`의 `has_excl22`, `unified_loader.py:491-501, 543-568` (excl22 mask), 결과 디렉토리 `SWaT/A1A2_full` · `SWaT/A1A2_excl22` ([N-COMP] §2.1 표).
- **metadata 실측** ([271c]): `SWaT/A1A2_full`: `swat_eval_mode=null`, `metrics.anomaly_ratio=0.19054`, `pak_auc_f1=0.9444`; `SWaT/A1A2_excl22`: `swat_eval_mode="excl22"`, `metrics.anomaly_ratio=0.03683`, `pak_auc_f1=0.6290` — **같은 모델, 평가범위만 다른데 지표가 크게 달라짐** = R28이 말하는 지배 효과의 직접 증거.

### 논문 서술 재료 (R28 "충분한 설명")
1. region 22 하나가 test anomaly 질량의 ~84%를 차지 → full 조건에서는 **이 단일 사건의 탐지 여부가 recall 질량 대부분을 결정**, 사실상 1개 사건 탐지 시험이 되어 모델 간 비교 변별력이 사라짐 (실측: full pak_auc_f1 0.944 vs excl22 0.629).
2. 따라서 **full과 excl22를 모두 보고**하되, 모델 변별은 excl22 기준 — [N-COMP] §3 callout: "SWaT는 excl22 (region 22 제외) 사용" (RankAvg 산정 기준).
3. excl22는 데이터 변경이 아니라 **평가 마스크** (학습·score 산출은 동일; eval_mask로 region 22 구간만 평가에서 제외) — 공정성 훼손 없음, 모든 모델에 동일 적용.

---

## ⑦ 라벨 희소화 sweep (R32) — 현존 자산과 placeholder 설계 입력

### 코드/Notion 실태 (실측)
- **전용 label-ratio sweep 파라미터·스크립트는 현재 코드에 없음**: `label_ratio`/`sparsif*`/`label_sparsity` 등으로 `mae_anomaly/`, `scripts/`, `configs/`, `config.py` 전수 grep — 0건. Notion 덤프 2개에서도 '희소화/sparsity sweep' 계획 서술 0건 (관련 hit은 R32와 무관한 early-stopping hyperparameter sweep뿐). → R32 원문대로 "**진행할 예정**" 상태가 맞다.
- **재사용 가능한 기존 메커니즘 (sweep의 구현 기반)**:
  1. `mae_anomaly/datasets/noisy.py:7-85` (파일 전체 85줄; class def `:7`) `NoisyLabelSlidingWindowDataset` — 학습 시에만 noisy 라벨을 반환하고(`use_noisy_labels = (split=='train')`, `noisy.py:52`), 평가는 원본 라벨 사용 → **라벨 희소화를 '학습 입력'에만 주입하는 정확한 구조**가 이미 있음.
  2. `scripts/run_base_experiments.py:397-416` `apply_normal50_noise` — **train 구간 anomaly region들의 50%를 무작위 선택(region 단위, seed=123)해 라벨 0으로 재라벨**. 호출부 `run_base_experiments.py:2397-2402, 2499-2502`. 현재 전 데이터셋 `normal50: False` (`run_base_experiments.py:260-388`), 2026-05-17부로 기본 비활성 (`run_base_experiments.py:331-333`).
- 라벨이 학습에 들어가는 채널(=희소화의 영향 경로): force_mask_anomaly (`config.py:315`), GRL classifier target (`config.py:126-134`), dynamic margin 분리 — 모두 `point_labels` 경유이므로 NoisyLabel 주입 한 곳으로 일괄 제어됨.

### Placeholder 실험 설계 입력 (위 자산 기준)
- **조작 변수**: 학습 train 구간에서 "라벨된 것으로 취급되는" anomaly region 비율 p ∈ {1.0, 0.75, 0.5, 0.25, 0.1, 0} (p=0.5는 기존 `apply_normal50_noise`와 동일 메커니즘; p=0은 완전 unsupervised 모드 — [N-METH] §5.2.1 "완전 unsupervised setting에서는 GRL이 비활성화되어 C3에 더 의존"과 연결).
- **단위**: region 단위 재라벨(점 단위 아님) — 기존 구현과 동일하며 "알려진 이상 사건" 개념과 일치.
- **고정**: 데이터·split·평가 protocol(§②④⑤) 전부 동일; 변경은 학습 라벨만.
- **가설 재료**: p가 줄어도 성능 열화가 완만하면 "소량 라벨로 충분" 주장 강화 (C4의 실용성).
- ⚠️ 수치는 실험 전 — 논문에는 placeholder로 표기, 수치 발명 금지.

---

## ⑧ 근거 포인터 전수 + REQUEST / FEEDBACK

### 근거 인덱스 (요약)
| 항목 | 근거 |
|---|---|
| 분할 구현 | `mae_anomaly/datasets/loaders.py` — SWaT 2005·2018 / WaDi 2201 / SMD 1153 / PSM 1686 / SMAP·MSL 2592-2595 (+safe-cut `_find_safe_cut_point` 1050-1101: clearance 판정 1071-1073, 무제한 outward 탐색 1080-1083) |
| Loader 레지스트리·활성 데이터셋 | `loaders.py:2688-2724` `DATASET_LOADERS`; `scripts/run_base_experiments.py:254-390` `DATASETS`/`SMD_DATASETS`/`SMAP_MSL_SIMPLE_DATASETS` |
| normalonly | `comparison/data/unified_loader.py:34-36, 392-485, 603+`; `comparison/experiment_configs.py` 전 entry |
| 평가 단일 원천 | `mae_anomaly/evaluator.py:864-987` (+ baseline 위임: `comparison/baseline_common.py:5, 553, 583` ) |
| 지표 출처 주석 | `evaluator.py:689-699` (VUS/Affiliation/R-F1), `1003-1005` (PA%K tadpak), `609-611` (PA=K0, Xu WWW'18) |
| AR threshold | `evaluator.py:752-815` (특히 785, 790, 793-794, 769-770) |
| excl22 | `evaluator.py:2299-2327, 2334-2366, 2155-2224`; `run_base_experiments.py:251, 749, 3457-3511` |
| 라벨 noise 자산 | `mae_anomaly/datasets/noisy.py:7-85`; `run_base_experiments.py:397-416, 2397-2402, 2499-2502` |
| 실행 프로토콜 (r2, 인용 r3 교정) | seed: `config.py:322, 326-333`, `run_base_experiments.py:2435`; random 5-run: `baseline_common.py:757, 786-796`; mean 집계: `evaluator.py:272, 278-280, 295-304, 2158`; best-epoch(test-split): `evaluator.py:1363-1373`, `run_base_experiments.py:2604, 2645-2646, 3215-3240`, `baseline_common.py:1368, 2087-2098`; epoch/eval 설정(비대칭): [271c] metadata `num_epochs=500`·`eval_interval=5` + `run_base_experiments.py:94` vs `baseline_common.py:272-323`(unsup 10)·`:333-384`(weak 50)·`:943`(eval_interval=1) — ⚠️ `config.py:264`·`baseline_common.py:256/266`은 default/stale, 인용 금지 (§④-실행 3항) |
| 실측 수치 | 원 CSV/라벨 직접 계산 (SWaT/WaDi/PSM — 본 문서 §①·⑥, 2026-06-10 수행) + [271c] experiment_metadata.json (PSM, SWaT full/excl22, WaDi A1/A2, SMD m-1-4, SMAP P-1, MSL C-1) |
| Notion | [N-METH] §1.2(데이터셋 표)·§1.3(C1–C4)·§3.6(평가)·§5.4(refs 12건 검증) / [N-COMP] §2.1(출처·byte-level 검증)·§2.2(Q1/Q3)·§3(Q3 결과·excl22 사용)·§6–7(인용·라이선스) |
| 지표 정식명 웹 재검증 | ACM DL 10.14778/3551793.3551830 (VUS) · arXiv:2206.13167 (Affiliation) · AAAI OJS 20680 (PA%K) · ACM DL 10.1145/3178876.3185996 (PA) — 2026-06-10 WebSearch |

### REQUEST (의사결정 필요 — Phase 2+에서 해소)
- **REQUEST-1 (R29×R30 정합)**: R29의 "affiliated-f1"·"Pa_f1"을 **어느 threshold로 보고할지** 확정 필요. 코드에는 (a) F1-최적 threshold 기반 `affiliation_f1`·`pa_0_f1`과 (b) AR threshold 기반 `affiliation_f1_ar`가 병존하나, **PA F1의 AR-threshold 변형(`pa_0_f1_ar`)은 존재하지 않는다** (`evaluator.py:769-770`이 PA%K 계열의 AR 재계산을 명시적으로 제외). R30 취지대로면 affiliation-F1은 `_ar` 키 사용이 정합; PA F1은 (i) 현행 `pa_0_f1`(F1-최적 threshold)을 그대로 쓰고 본문에 threshold를 명기하거나 (ii) AR-threshold PA F1을 추가 구현해야 함. → 사용자 확인 요청.

  **RESOLVED (reconciler, 2026-06-10) — 코드·metadata 사실관계 확정 (보고 threshold의 '선택'은 여전히 사용자 결정 사항이나, 전제 사실은 아래로 확정)**:
  1. **`pa_0_f1`의 threshold**: `compute_full_metric_set`이 ROC curve 위 **F1-최적 threshold** 한 개를 선정하고(`evaluator.py:929-930`: `find_f1_optimal_idx` → `threshold = thresholds[optimal_idx]`), 모든 per-K PA%K(`pa_{k}_*`, K=0 포함)는 **그 동일 threshold로 이진화한 예측**에 PA%K 조정을 적용해 계산된다 (`evaluator.py:951-955`: `compute_pa_k_metrics_from_mean_scores(..., threshold, k, ...)`). 즉 `pa_0_f1` = F1-최적 threshold 기반 PA F1이 맞음.
  2. **`pa_0_f1_ar` 부재 재확인**: `compute_ar_threshold_metric_set`의 출력 키 집합(`evaluator.py:772-781`)에 PA 계열 없음 + docstring 명시(`evaluator.py:769-771`). metadata 실측([271c] PSM `metrics` **153키** 전수 조회 — 정정 r2: 초판 "149키"는 오기, 재실측 153키·None 0건; 2026-06-10 재확인): `_ar` 접미 키는 정확히 10개(`f1_ar, precision_ar, recall_ar, f1_t_ar, precision_t_ar, recall_t_ar, affiliation_{precision,recall,f1}_ar, r_based_f1_ar`)이며 **`pa_*_ar` 계열 0건**.
  3. **affiliation 지표의 threshold 의존성**: affiliation은 **threshold-dependent** (이진화된 pred 필요). `affiliation_f1` = F1-최적 threshold 사용 — `compute_extra_metrics(base_scores, base_labels, threshold, ...)` 호출(`evaluator.py:980-982`) → `pred = (s > float(threshold))` (`evaluator.py:730`) → `_compute_threshold_dependent` (def `evaluator.py:637` — 정정 r2: 초판 "639+"). `affiliation_f1_ar` = AR threshold 사용 — `ar_th = quantile(s, 1-ar)` (`evaluator.py:790`), `pred = (s > ar_th)` (`evaluator.py:794`) → 동일 `_compute_threshold_dependent` 경유 (`evaluator.py:811-813`). 두 변형 모두 metadata에 실재([271c] PSM: `affiliation_f1`·`affiliation_f1_ar` 공존 확인).
  4. 따라서 R30 정합 보고안: affiliation-F1 → `affiliation_f1_ar` 사용 가능(구현 완료·산출 중); PA F1 → `pa_0_f1`(F1-최적 threshold)만 존재하므로 옵션 (i) 채택 시 본문에 "PA F1은 F1-최적 threshold 기준" 명기 필요, 옵션 (ii)는 코드 추가 구현 필요. — 선택은 Phase 2+/사용자.
- **REQUEST-2 (R29 키 확인)**: "pak_auc_pr" = 내부 `pak_auc_prc_auc`로 매핑했음 (PA%K-AUC of AUC-PR). 다른 의도(예: `vus_pr`과 혼동)였는지 확인 요청.

  **RESOLVED (reconciler, 2026-06-10)**: 내부 키는 **`pak_auc_prc_auc`로 확정**. 근거: ① 코드 — `PAK_AUC_KEYS` 튜플 첫 원소가 `'pak_auc_prc_auc'` (`evaluator.py:840-841`); 산출 지점 `evaluator.py:1271` (`'pak_auc_prc_auc': float(np.trapz(prc_aucs, k_values) / 100.0)`). ② metadata 실측 — [271c] PSM `metrics`의 pak 계열 키 전수 = {`pak_auc_f1`, `pak_auc_f1_raw`, `pak_auc_f1_t`, `pak_auc_f1_t_raw`, `pak_auc_prc_auc`, `pak_auc_precision(_raw)`, `pak_auc_recall(_raw)`, `pak_auc_roc_auc`} — **`pak_auc_pr`·`pak_auc_prc`라는 키는 코드·metadata 어디에도 없음**. `vus_pr`은 별도 키로 공존하므로 혼동 여지 없음. §④ 매핑 표의 기존 기재 그대로 유효.
- **REQUEST-3 (비교표 조건 확정)**: 메인 비교표의 baseline 조건이 **Q3(normalonly) 단독**인지 **Q1+Q3 병기**인지 확정 필요. [N-COMP]는 Q3 진행/Q1 pending — R12/R31 논리상 메인은 Q3, Q1은 보조 권장.
- **REQUEST-4 (r2 신설 — best-epoch 공개 서술 방식)**: §④에 명기했듯 best-epoch은 **test-split `pak_auc_f1`로 선정**된다 (전 모델 동일 — test-set model selection; 별도 validation split 없음). 논문 experiments 섹션에서의 공개 서술 방식 결정 필요 — 예: (i) "oracle best-epoch protocol, applied uniformly to all methods" 류의 명시적 한 문장 공개, (ii) limitation 절에 별도 기술, (iii) validation-split 기반 선정 추가 실험 여부. 숨기는 선택지는 없음(진실 문서 원칙 + 리뷰어 단골 공격 지점). → 사용자/Phase 2+ 결정.

### FEEDBACK (사실 확인됨 — 후속 phase 주의사항)
- **FEEDBACK-1 (결과 미완)**: [271c] (`271_20260602_020545_271canon_baseline`) 현재 채워진 entity: SMD 22/28, SMAP 5/54, MSL 5/27 (2026-06-10 ls 실측). 또한 baseline 쪽은 per-entity 정규화(2026-06-02) 이후 **SMD/SMAP/MSL/Exathlon 결과 STALE → 재실행 필요** ([N-COMP] §3 red callout). 논문 표는 큐 완주 후 수치로만 작성할 것.
- **FEEDBACK-2 (WaDi A2 feature 수 불일치)**: [N-METH] §1.2 표는 "WaDi A2 = 127", [N-COMP] §2.1과 loader docstring(`loaders.py:2176` "All features preserved (123)")은 123. 실 실행 시 `data_info['n_features']`로 확정 필요 — 논문 표 작성 시 재확인.

  **RESOLVED (reconciler, 2026-06-10)**: **123으로 확정**. 근거: ① [271c] `WaDi/A2/experiment_metadata.json: config.num_features=123` (학습 모델 입력 차원). ② 현 raw CSV 실측 — `WADI_A2_attack_raw.csv`/`WADI_A2_14days_raw.csv` 모두 124 cols = 123 features + label (2026-06-10 직접 카운트). ③ **127의 출처**: WaDi A2 원본 배포 CSV(`WADI_attackdataLABLE.csv`)의 sensor 컬럼이 127개(= 131 cols − 3 meta − 1 label; `scripts/prepare_raw_datasets.py:411` docstring)이며, 전처리 단계 `handle_nan`이 **all-NaN 컬럼 4개를 drop** — 직접 diff 재계산으로 식별: `2_LS_001_AL`, `2_LS_002_AL`, `2_P_001_STATUS`, `2_P_002_STATUS` (127 − 4 = 123). 즉 [N-METH]의 127은 NaN-drop 전 원본 sensor 수, 논문 표는 **모델 입력 기준 123** 사용 (WaDi A1도 동일하게 123).
- **FEEDBACK-3 (R33 집계 영향)**: 기존 Notion RankAvg는 Exathlon 포함 6-dataset 기준 ([N-COMP] §3.4). 논문용 집계는 Simulation·Exathlon 제외 + SMAP/MSL 포함으로 **재계산 필수**.
- **FEEDBACK-4 (Notion 스냅샷의 stale 정보)**: [N-METH](2026-05-31)는 "SMAP/MSL은 MAE 학습 entry 미통합"이라 기술하나, 현재 코드는 통합 완료 (`run_base_experiments.py:299-312, 379-390`; [271c]에 SMAP/MSL 결과 존재). 코드가 ground truth.
- **FEEDBACK-5 (R13 서술 시 주의 — r2 정정)**: SMAP/MSL만 50% 컷에 safe-cut(**최근접 안전 지점으로 이동 + 최소 margin=10 clearance, 탐색 거리 무제한** — `loaders.py:1050-1101`, 호출 `:2592-2595`)이 있음. "통일된 규칙" 주장 시 이 경계 조정 메커니즘과 **실측 이동량**(MSL 4/27 이동, max +166 steps = D-16 test의 7.58%, 나머지 |Δ|≤39; SMAP 0/54 — §② 실측 표)을 함께 공개해야 정직성·공정성 서술이 성립. "±10 범위 이동"·"실질 영향 없음" 표현 금지 (초판 오류, B-1).
- **FEEDBACK-6 (SWaT train anomaly)**: 이 split에서 SWaT train에도 anomaly 1.63%(11,757 pts, A2 앞 50%의 21개 region)가 존재 — "A1 normal + A2 front" 서술 시 train도 오염(=라벨 활용 대상)임을 명시해야 R13 논리가 완성됨.
- **FEEDBACK-7 (reconciler 2026-06-10 추가 — SWaT feature 수 45 확정 + 재현성 플래그)**: 학습된 271 SWaT 모델의 입력 차원은 **45** (① [271c] full/excl22 `config.num_features=45`, ② `best_config.json: num_features=45`, ③ checkpoint `patch_embed.weight=(512, 450)`=Linear(10×45→512) 실측). 45 = 원본 51 sensor − **combined(A1+A2) 기준 constant 컬럼 6개** {P202, P401, P404, P502, P601, P603} — 2026-06-10 원 CSV에서 동일 산식(`np.std==0` over combined float32) 재계산으로 정확히 재현. ⚠️ 그러나 **현 machineA의 raw CSV(51 features, mtime 2026-05-26) + 현행 `load_swat_a1a2_raw`(constant 제거 코드 없음, 학습 시점 commit e5938eb에서도 동일)** 경로는 51을 반환한다 — 학습 당시 source-machine의 CSV가 constant-제거된 45-feature 버전이었던 것으로 추정되나 machineA에서는 검증 불가. **재실행/재현 시 feature 수 45 일치 여부를 반드시 확인할 것** (미확인 시 checkpoint와 입력 차원 불일치로 로드 실패 또는 결과 비교 불능).
