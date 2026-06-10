---
phase: 1
agent: rereviewer-beta
directives: [R2, R26, R12, R13, R28, R29, R30, R31, R32, R33]
last_modified: 2026-06-10
round: 2 (re-review of r2 fixes)
inputs:
  - paper/01_research_understanding/NOTION_DIGEST.md (r2)
  - paper/01_research_understanding/CONFERENCE_PDF_DIGEST.md (r2)
  - paper/01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md (r2)
  - paper/99_reviews/p1_digests_r1.md + p1_digests_fixlog_r2.md
  - paper/99_reviews/p1_protocol_r1.md + p1_protocol_fixlog_r2.md
sources_reverified:
  - "Notion Page 0 덤프 (75,820 chars decoded) / Page B 덤프 (108,461 chars decoded) — python 디코딩 후 regex 전수 대조"
  - "paper/윤기오_대한산업공학회_2026_춘계.pdf p6, p9–12, p19–22 직접 재독"
  - "mae_anomaly/datasets/loaders.py(:1050-1101 본문 재열람 + safe-cut 4채널 실측 재현), evaluator.py, config.py, comparison/baseline_common.py, experiment_configs.py, scripts/run_base_experiments.py, noisy.py, docs/DATASET.md — 인용 라인 전수 재열람"
  - "[271c] metadata (PSM 153키 재집계, WaDi A2=123, SWaT=45, num_epochs/eval_interval) + comparison/configs sota_epochs 전수 grep"
---

# Phase 1 재리뷰 β (round 2) — digest 2종 + protocol truth

## 판정 요약

| 문서 | 판정 | r1 발견 마감 | 신규 BLOCKER | 신규 MAJOR | 신규 MINOR |
|------|------|------------|--------------|------------|------------|
| NOTION_DIGEST.md | **PASS** | 12/12 해소 확인 | 0 | 0 | 1 |
| CONFERENCE_PDF_DIGEST.md | **PASS** | 7/7 해소 확인 | 0 | 0 | 0 |
| EXPERIMENT_PROTOCOL_TRUTH.md | **FAIL (재수정 필요)** | 12/12 해소 확인 | **1** | 0 | 1 |

총평: r1의 31건(digest 19 + protocol 12) 발견은 **전건 fixlog 기록 + 본문 실반영**을 1:1 대조로 확인했고, 고위험 수정 5건(WaDi A2=123 truth 표 / ×22 산식 / 26종+Ours / R26 라벨 축소 / safe-cut 재서술)은 원천·코드에서 전부 재검증 통과다. safe-cut 실측 표는 **4채널 독립 재현으로 bit-exact 일치**(D-16 +166=7.58%, M-1/M-2 −39, S-2 +8), best-epoch test-set selection 명기는 코드 5개소 재열람으로 정확. 그러나 **M-4로 신설된 "실행 프로토콜"의 baseline epoch-패리티 주장이 코드·Notion·자기 문서(cross-ref한 271_CONFIG_TRUTH)와 모두 모순되는 신규 사실 오류**라 protocol 문서만 FAIL이다. 국소 수정(한 소절 재작성)으로 해소 가능.

---

## A. r1 발견 전수 마감 확인 (1:1 대조)

### NOTION_DIGEST.md — 12/12 FIXED 확인
| ID | 반영 위치 | 재검증 결과 |
|----|----------|------------|
| NB-1 | II-3 표 123 + I-3/I-7/I-9 † 주석 + §IV-11 | Page B 덤프 @18880 `WaDi A2|123` ✓, Page B 전문 "127" **0회** regex 재확인 ✓, PDF p19 123 dim ✓, [271c] `WaDi/A2 config.num_features=123` 실측 ✓ |
| NB-2 | II-1 산식 | 덤프 @23502 Total 문장과 **자구 일치** ("39 base + 2 = 41 × 2 × 22 = 1,804") ✓; Snapshot @3617 "39 dataset runs" vs Total 41의 내부 모순 주석도 원문 정확 ✓ |
| NM-1 | I-10 2-표 분리 | MASTER_ORCHESTRATION_PROMPT **L491 R26 원문 확인** ("비교 대상 모델 reference 및 데이터셋 reference") — [4],[6]–[10],[12]=Page B [B2]/[D1]–[D5]/[D8] 동일 항목 매핑 논리 타당 ✓ |
| NM-2 | II-6 한정 블록 | §3 Status(2026-05-22) @26905 ✓, "영향 모델 15 종(9+6, gcn_lstm 중복)" @101431 ✓, ~70 dirs 백업·삭제 @100619 ✓, 5번 폐기→6번(`6_20260526_085028_*_segaware`) @2972 ✓ |
| NM-3 | II-2b 신설 | §1.2 HP preset 전수 spot 대조(anomaly_transformer 100/512/8/3/bs128, tranad lr=1e-4·LeakyReLU, gdn batch=32 run.sh, omnianomaly 100/500/3, memto stride100·FRESH·Phase1 3ep, catch 192/16/8/0.005, dcdetector 105/[3,5,7], timesnet 128/128/3/3+512/2048 근거, deepmil 128[fixed]·CROSS-PRODUCT·DERIVATIVE_CITED, random seed=None 5-run) — **전부 덤프 원문 일치** ✓; weak label `max(point label over window)` @12008 ✓; provenance 태그 4종 @12542 ✓; fit-on-test vs deepmil leak-free @68453/@69232 ✓; 21개 windowing baseline @102720 ✓ |
| NM-4 | I-1/I-3/I-4/I-5 | round(50×0.15)=8 고정 @19799 ✓, priority 1[anomaly]·1000+η/TopK_8/random subset @20173·@20396 ✓, 두 디코더 self-attn only·cross-attention 없음 @23071 ✓, teacher_only=True 플래그 @33475 ✓, bf16(2026-05-27) @61555 ✓, eval_interval=5 @61255 ✓, random_seed=42 @61850 ✓, validation 5종 @16867 ✓, λ anchor=student decoder 마지막 weight @26224 ✓ |
| Nm-1 | §IV-12 | §2.4 4:1+FM제외 @15402/@15821 vs §3.6 @33883·§4.3.3 @45363·§5.3 @69030 1:1 잔존 — **모순 실재 확인** ✓ |
| Nm-2 | II-1 주석 | Snapshot "weakly-supervised 5종 50 epoch" @3830 ✓, nrdetector_full 실재 @65501/@92660 ✓ |
| Nm-3 | II-4 등급 통일 | Q2/Q4 폐기 — Snapshot @3514 + §2 @26794 양쪽 명시 ✓ |
| Nm-4 | II-4 예외 전체 | @26439 원문과 자구 일치 (simulation/`*_simple`/단일 machine) ✓ |
| Nm-5 | II-3 TEP 행 | §2.1 TEP 행 @22874 (52/—/4 RData 일치) ✓, §6.4 CC0 @76792 ✓, "Harvard Dataverse 공개"·CC0 병기 정확 ✓ |
| Nm-6 | [B4b]/[B5b], IV-6 | manigalati official-affiliated(원저자 contributor) @45720 ✓; dagmm "RELABEL only"·`dagmm_tranad` relabel·직접 비교 금지 — **기결정** @94312 원문 일치, IV-6 재서술 정확 ✓ |

### CONFERENCE_PDF_DIGEST.md — 7/7 FIXED 확인
| ID | 재검증 결과 |
|----|------------|
| PB-1 | p20 직접 재실측: Simple 5 + Neural 3 + SOTA 14(GCN-LSTM…CATCH) + Weak 4 = **26 + Ours** ✓. p22 rank 첨자 (27) 실재(Anomaly Trans. WaDi A1 0.0806₍₂₇₎ 확인) ✓. p34 refs [1]–[25], 모델 refs [8]–[25]=18개 — 혼동원인 주석도 정확 ✓ |
| PM-1 | p12 그림 "Patchify (1D-CNN)" 재확인 ✓ vs Page 0 `patchify_mode='linear'` callout @12967 + 파라미터 표 @38663 ✓ — 모순 실재, ⑧ REQUEST 추가 확인 ✓ |
| Pm-1 | p6 원문 "…해결책**이 될 수 있음**" 재확인 ✓ — 인용 복원 + ⑦-6 조정 적절 ✓ |
| Pm-2 | p9–10 재독: 제목+CVPR 2024만 표기, 저자명 부재 ✓ — 외부 귀속 표기 정확 ✓ |
| Pm-3 | p11 재독: 𝒳ᴺ_lab 실재, ℒ_disc의 ℳᴺ만 𝒳ᴬ_lab 참조 ✓ — "손실 수식에는 𝒳ᴬ_lab만 등장" 재서술 정확 ✓ |
| Pm-4 | p19 표 재독: #Training/#Testing(719,959/224,960·189,060; 1,296,001/86,401; 870,972/86,402; 176,401/43,921), #Regions(14/13/7/7/29), Train AR(1.63/0.52/0.76/6.20) — **전수 일치** ✓ (caption "contiguous … in the test split"도 일치) |
| Pm-5 | p21 표 재독: F1 "no single canonical origin", F1_PA [4] Xu WWW'18, F1_PA%K·PRC_PA%K [5] Kim AAAI'22, VUS [6] Paparrizos PVLDB'22, Aff-F1 [7] Huet KDD'22 — **전수 일치** ✓ |

### EXPERIMENT_PROTOCOL_TRUTH.md — 12/12 반영 확인 (단, M-4 반영분에 신규 오류 — §B)
| ID | 재검증 결과 |
|----|------------|
| B-1 | `_find_safe_cut_point` 본문 재열람: docstring(:1053) clearance 정의, 발동조건 `s-margin<=pos<=e+margin`(:1071-1073), 무제한 outward(:1080-1083), fallback(:1085-1101) — **문서 인용 라인 전부 정확** ✓. margin=10 호출(:2672/:2684, `margin=safe_cut_margin` :2594) ✓. **실측 표 독립 재현: D-16 +166(7.58%)/M-1·M-2 −39(1.71%)/S-2 +8(0.44%) — bit-exact 일치** ✓. 합계 252 steps 산술 ✓. ERRATA의 `docs/DATASET.md:1151`·`loaders.py:2591` 원문 인용 정확 ✓. FEEDBACK-5 재작성 ✓ |
| M-1/M-5 | `grep -n` 재확인: SMD_simple alias **:2742** ✓, WaDi 레지스트리 **:2698-2699** ✓, SWaT :2690 ✓ (loaders.py 2,770줄 ✓) |
| M-2 | §⑤-4 재열람: `:345` 인용 제거 + NRDetector PU-prior 주석(:335-350 재열람으로 부적합 확인) 사유 명기 + Phase 4 이관·사용 금지 명시 ✓ |
| M-3 | `Evaluator.__init__`(:1363-1373, test_loader만) ✓, evaluate docstring(:2155-2160) ✓, best-ckpt 갱신(run_base :2604, :2645-2646) ✓, 최종 스캔(:3215-3240) ✓, baseline 동일(:1368 docstring, :2087-2098 excl22 독립 스캔) ✓, §⑤-2 cross-note + REQUEST-4 신설 ✓ |
| M-4 | 소절 신설 확인 — seed(config.py:322, :326-333, run_base :2435·:2442·:2509·:2522·:2542) ✓, mean 집계(evaluator.py:8, :272, :278-280, :295-304, :2158) ✓, 271_CONFIG_TRUTH cross-ref ✓. **단, ③ 패리티 소항목에 신규 사실 오류 — §B RB-1** |
| MIN-1~7 | 113(산식·114 병기) ✓, legacy 6(experiment_configs.py:24-31 실측, 5+3+1+6+7=22) ✓, **PSM metrics 153키·None 0·`_ar` 정확 10키·`pa_*_ar` 0건 — metadata 재집계 일치** ✓, noisy.py 85줄(class :7, use_noisy_labels :52) ✓, `_compute_threshold_dependent` def :637 ✓, '전부 정상' 성격 단락(SMAP/MSL zeros :2602-2604 / PSM :1672-1675 / SMD :1139-1142) ✓, 덤프 절대경로 실재 ✓ |

RESOLVED 블록 부속 인용 재확인: `find_f1_optimal_idx` :929, `pak_auc_prc_auc` :1271, PAK_AUC_KEYS :840, PA_K_VALUES :831, AR threshold(:785/:790/:793-794/:769-771/:818-828), excl22(:2299/:2334), pak 키 집합 metadata 일치, [271c] SWaT 45/full 0.19054·0.9444/excl22 0.03683·0.6290, PSM 0.30628/0.001744/0.7616/0.8012 — **전부 일치** ✓.

---

## B. 신규 오류 (r2 수정으로 도입)

### RB-1. [BLOCKER — EXPERIMENT_PROTOCOL_TRUTH.md §④ 실행 프로토콜 3항] baseline epoch-패리티 주장이 사실과 반대

문서 주장: "① epoch 수 — MAE `num_epochs=50` (`config.py:264`) = baseline 50 (\"epochs=50 user override\", `baseline_common.py:256, 266`; …) … ④ … → **양쪽 모두 50 epoch 완주**".

**실측 반증 (3중)**:
1. **MAE 쪽**: 논문 정본 실험 [271c]의 metadata는 **`num_epochs=500`, `eval_interval=5`** (PSM/SWaT full/WaDi A2 3건 직접 조회) — 같은 소절 ④항이 정본으로 cross-ref한 `271_CONFIG_TRUTH.md:141`도 `num_epochs=500`. `config.py:264`의 50은 **dataclass default일 뿐 exp271 값이 아니다** (문서 내부 자기모순).
2. **Baseline 쪽**: unsupervised 22종 preset은 전부 **`'epochs': 10`** — `baseline_common.py:272, 279, 286, 297, 300-302, 308, 314, 323` 주석 원문 "**2026-06-06: unsupervised unified to 10**". 50은 **weak 4종만** (:333, :337, :355, :367, :384 "weak unified to 50"). 인용된 `:256/:266`의 "epochs=50 user override"는 **2026-06-06 이전의 stale docstring/주석**으로, 바로 아래 실값(10)과 모순된다. `comparison/configs/`의 `sota_epochs=50` override도 weak-SSL 큐 한정임을 grep으로 확인.
3. **원천 교차**: Page B Snapshot "모든 unsupervised(neural/SOTA) 모델 **10 epoch**, weakly-supervised 5종 50 epoch (2026-06-06 통일)" — **NOTION_DIGEST II-1은 이를 올바르게 기재**하고 있어, r2 시점에 **두 산출물 간 직접 모순**까지 발생했다.

파생 부정확: 같은 항 ②의 "per-epoch eval cadence" 패리티 함의도 틀렸다 — baseline `eval_interval=1` vs MAE `eval_interval=5` ([271c] metadata). 동일한 것은 cadence가 아니라 **best-epoch 선정 기준(pak_auc_f1)과 per-epoch(또는 5-epoch 간격) 평가 후 best 선택이라는 구조**다.

영향: 이 소절은 "논문 experiments 섹션 필수 기재 사항"으로 신설된 것이라, 그대로 논문에 들어가면 **거짓 공정성 주장**(epoch 패리티)이 된다. 실제 사실(MAE 500 / unsup 10 / weak 50, eval 5 vs 1)은 오히려 **공개·방어해야 할 비대칭**이다.

수정 방향: ③항을 "epoch 수·eval 간격은 **패리티가 아니라 비대칭** — MAE 500ep(eval 5ep 간격, [271c]) vs unsup baseline 10ep(eval 매 epoch) vs weak 50ep. 공통점은 ⓐ 동일 best-epoch 기준(pak_auc_f1) ⓑ 양쪽 모두 per-epoch-eval 후 best 선택 ⓒ early stopping 부재(완주). 논문에는 epoch 비대칭을 명시 공개하고 '각 모델은 자기 best epoch에서 평가' 프로토콜로 서술"로 재작성. fixlog의 해당 "재검증" 행도 정정 필요.

### RM-1. [MINOR — EXPERIMENT_PROTOCOL_TRUTH.md §④ 실행 프로토콜 1항] "모든 실험은 dataset entry당 단일 run" 과대 일반화
MAE 파이프라인(run_base_experiments) 한정으로는 참이나, baseline `random`은 2026-06-04부터 **5회 독립 run → mean±std** (Page B §1.2.1; NOTION_DIGEST II-2b에도 기재). "모든 실험" → "MAE 실험(및 deterministic baseline)" 으로 범위 한정 + random 예외 1줄.

### NMr-1. [MINOR — NOTION_DIGEST.md §IV-11] "Page 0은 3개소에서 127" — 실제 4개소
num_features 표(@35093)·d_model 매핑(@12817)·§5.2.1(@66011) 외에 **§1.2 지원 데이터셋 표(@2793)에도 127**이 있다 — digest I-7 표가 바로 그 표를 전사하며 † 주석을 달았으므로 IV-11의 "3개소" 카운트는 자기 본문과도 불일치. (fixlog NB-1 행은 @2793을 "d_model 표"로 오표기.) 실질 무해 — 123 확정 결론·주석 체계 불변. "4개소"로 정정만.

---

## C. 고위험 수정 재추적 결과 (요약)

1. **WaDi A2=123 truth 표**: Page B 원문·PDF p19·[271c] metadata·raw 경로 설명(127=all-NaN 4컬럼 drop 이전) 4중 일치 — **정확** ✓
2. **"× 22 active models" 산식**: 원문 자구 일치 — **정확** ✓
3. **baseline 26종(+Ours=27)**: p20 실측 5+3+14+4, (27) 첨자, Page B 22+4 — 3중 일치 — **정확** ✓
4. **R26 라벨 축소**: R26 원문(L491) 범위와 정합, 7건/5건 분리·귀속 표기·Phase 4 단서 — **정확** ✓
5. **safe-cut 재서술**: 코드 본문·라인 인용·실측 표(독립 재현 bit-exact)·ERRATA — **정확** ✓
6. **best-epoch test-set selection**: 코드 5개소 + validation split 부재 — **정확**, REQUEST-4 적절 ✓
7. **실행 프로토콜 신설**: seed=42·mean 집계·cross-ref는 정확 ✓ / **epoch 패리티는 오류 (RB-1)** ✗

## D. A8 톤 점검 — PASS
세 문서 전수 스캔: "실험 데이터 부족"을 한계로 폄하하는 톤 없음. 한계 서술은 전부 (a) PDF/Notion 원문 귀속([Notion의 주장 — 디자인 한계], p4 라벨링 비용) 또는 (b) 사실 기록 + 정책 동반(§⑦ placeholder 설계+"수치 발명 금지", II-6 "swap-in 값만 사용", §④-실행 "단일-seed 사실 명시"). 참고: NOTION_DIGEST IV-4의 "이 수치 없이는 … 불완전"은 r1부터 있던 사실 기록으로 위반은 아니나, placeholder 정책 문구를 덧붙이면 더 안전 (수정 비강제).

## E. 비고 (산출물 외 메모)
- `p1_digests_fixlog_r2.md` frontmatter "FIXED 18 + PARTIALLY-REVISED 1" vs 본문 표 전건 FIXED — 라벨 불일치 (실해는 없음).
- RB-1 정정 시 `p1_protocol_fixlog_r2.md`의 M-4 "재검증" 서술(epoch 50 통일)도 함께 정정 권고.

## 재심 조건
EXPERIMENT_PROTOCOL_TRUTH.md: **RB-1 재작성(필수) + RM-1 한정(권장)** 후 재심. NOTION_DIGEST.md: NMr-1은 1줄 정정으로 충분 (PASS 유지, 차기 patch에 포함). CONFERENCE_PDF_DIGEST.md: 수정 불요.
