---
phase: 1
agent: adversarial-reviewer-C
directives: [R12, R13, R24, R28, R29, R30, R31, R32, R33]
last_modified: 2026-06-10
---

# Adversarial Review r1 — EXPERIMENT_PROTOCOL_TRUTH.md (reconciler 정정본)

> 검증 방법: 문서가 인용한 **모든 핵심 file:line을 직접 재열람** (loaders.py / evaluator.py / config.py / run_base_experiments.py / unified_loader.py / experiment_configs.py / baseline_common.py / noisy.py / prepare_raw_datasets.py / docs/DATASET.md), **원 CSV·npy 재계산** (SWaT region22, PSM, WaDi 행/열수, SMAP·MSL 81채널 safe-cut 실측), **[271c] metadata 재조회** (PSM 키 전수, SWaT full/excl22, WaDi A1/A2, SMD 22 machines), **Notion 덤프 원문 grep** ([N-COMP] 인용 8건), **웹 재확인** (VUS·PA%K·Affiliation·PA 4편 실재/venue). 코드 read-only 준수.

## 판정 요약

**REJECT — 수정 후 재심사 (BLOCKER 1, MAJOR 4, MINOR 7)**

- **BLOCKER 1건**: SMAP/MSL safe-cut 동작 서술이 코드와 다름 ("±10 timestep 범위 이동" ≠ 실제 무제한 outward 탐색 + margin=10 clearance). 실측으로 반증됨 — **MSL D-16의 cut은 166 timestep 이동** (해당 채널 test 길이의 7.58%). R13 "통일 규칙 + 예외 공개" 서술 재료가 이 오류 위에 서 있다.
- 그 외 핵심 골격은 견고함: split `//2` 전수, normalonly, AR threshold, excl22, R28 수치(83.75%/35,900/[2869,38769) — **원 CSV에서 bit-exact 재현**), RESOLVED 블록 3건(pa_0_f1 / pak_auc_prc_auc / WaDi A2=123), 지표-논문 매핑 4건 — 모두 재검증 통과.
- MAJOR는 (i) EOF 너머를 가리키는 깨진 인용 1건, (ii) R30 방어 §⑤-4의 인용-주장 불일치, (iii) best-epoch이 **test-split 지표로 선정**된다는 사실 미기재, (iv) 실험 섹션 필수 프로토콜 누락(반복 수/seed, score 집계, baseline 패리티).

---

## 1. BLOCKER

### B-1. Safe-cut 동작 서술이 코드 사실과 불일치 + "분할 비율에 실질 영향 없음" 실측 반증 (§① SMAP 행, §② 표 SMAP/MSL 행, §②-서술재료-2, FEEDBACK-5)

**문서 주장**: "50% 지점이 anomaly region 내부면 **±10 timestep 범위에서** region 밖으로 이동", "±10-timestep safe-cut으로 … **분할 비율에 실질 영향 없음**".

**코드 사실** (`mae_anomaly/datasets/loaders.py:1050-1101` `_find_safe_cut_point`):
- `margin=10`은 **이동 한계가 아니라 clearance 요건**이다: "A 'safe' position is at least `margin` timestamps away from any anomaly region" (docstring, :1053).
- 이동 자체는 **무제한 outward 탐색**: `for offset in range(1, L): for candidate in [target-offset, target+offset]` (:1079-1082) — 50% 지점이 긴 anomaly region 안/근처면 region 전체 + 10 clearance를 벗어날 때까지 **얼마든지 멀리** 이동한다.
- 발동 조건도 "region 내부"가 아니라 "**어떤 region의 ±10 이내**" (`s - margin <= pos <= e + margin`, :1071-1073).

**실측 반증** (2026-06-10, 81채널 전수 — 실제 loader 함수 직접 호출):

| 채널 | test_len | target(50%) | 실제 cut | 이동량 | 비율 변화 |
|---|---|---|---|---|---|
| MSL D-16 | 2,191 | 1,095 | 1,261 | **+166 steps** | **7.58%p** |
| MSL M-1 | 2,277 | 1,138 | 1,099 | −39 | 1.71%p |
| MSL M-2 | 2,277 | 1,138 | 1,099 | −39 | 1.71%p |
| MSL S-2 | 1,827 | 913 | 921 | +8 | 0.44%p |

(이동 채널 목록 4/27 + SMAP 0건 자체는 문서 기재와 **정확히 일치** — 목록은 맞고 메커니즘·크기 서술이 틀림.)

**영향**: Pattern B(per-channel) 실험에서 D-16은 test가 166 step 줄어든 split이다 — "±10이라 실질 영향 없음"은 그대로 논문에 쓰면 **사실과 다른 공정성 주장**이 된다. R13의 "전 데이터셋 통일 + 예외 정직 공개" 서술은 (a) 올바른 메커니즘("cut을 가장 가까운 안전 지점으로 이동, 최소 10-step 여유"), (b) **실측 이동량 표** (max 166 steps, 4/81 채널, concat 규모 대비 미미)로 다시 써야 방어 가능하다.

**오류의 뿌리**: `docs/DATASET.md:1151`("pushed outside any anomaly region by ±10 timestamps")과 [N-COMP] §2.1("margin=10으로 모두 회피")의 표현을 그대로 수용 — 그러나 본 문서의 선언된 원칙은 코드 직접 검증이다. 코드 자체 주석(`loaders.py:2591` "pushed outside anomaly regions ±margin")도 같은 모호함을 갖고 있으나 함수 본문이 ground truth.

---

## 2. MAJOR

### M-1. EOF 너머를 가리키는 깨진 인용 — `loaders.py:2810-2812` (§② 표 SMD 행)
파일은 **2,770줄**. `SMD_simple_<machine>` alias 등록의 실제 위치는 **`loaders.py:2742`** (`DATASET_LOADERS[f'SMD_simple_{_mn}'] = …`). 주장 자체는 참이나 인용된 근거가 존재하지 않음 — "모든 주장에 근거" 원칙 위반. 수정: 2742로 교체.

### M-2. R30 방어 §⑤-4 "(문헌 관행)" — 인용-주장 불일치
`baseline_common.py:345`의 "anomaly ratio; dataset-dependent → must be estimated, NOT a fixed constant"는 **NRDetector(weak-supervised)의 PU class-prior 추정**에 관한 주석이다 (전후 문맥 :343-347: "prior=None → estimated dynamically from train wlabel rate (**PU class prior** = intrinsic anomaly ratio…)"). **test score thresholding 관행과 무관**. "test anomaly-ratio thresholding은 TSAD 문헌의 표준 관행" 주장 자체는 (별도 문헌 근거로) 성립 가능하나, 현재 인용으로는 불성립 — 삭제하거나 실제 문헌(예: OmniAnomaly/USAD류의 ratio-based threshold 사용 선례, Phase 4 검증 대상)으로 교체할 것.

### M-3. Best-epoch 선정이 **test-split 지표**로 이뤄진다는 사실 미기재 (§④·⑤)
`Evaluator`는 `test_loader` 위에서 per-epoch 평가한다 (`evaluator.py:1367-1372` — 생성자가 test_loader/test_dataset만 받음; `evaluate()` docstring :2155-2160). 즉 **best epoch = test-set `pak_auc_f1` 최대 epoch** — 일종의 test-set model selection이며, baseline도 동일 ([N-COMP] "모든 metric 같은 best epoch에서 추출"). 문서는 §⑤-2에서 best-epoch 기준을 R30 공정성 방어에 동원하면서 이 사실을 침묵한다. **전 모델 동일 적용이라 비교 공정성은 유지**되지만, 논문 experiments 섹션에 반드시 공개해야 하는 프로토콜 사실 (리뷰어 단골 공격 지점)이며, "진실 문서"라면 누락하면 안 된다. → §④ Best-epoch 항목에 "per-epoch 평가는 test split 위에서 수행되며 best epoch도 test 지표로 선정 (전 모델 동일)" 명기 + REQUEST로 논문 내 서술 방식(예: 'oracle best-epoch protocol, applied uniformly') 결정 요청.

### M-4. 논문 experiments 섹션 필수 프로토콜 누락 (cross-reference도 없음)
1. **반복 수/seed 정책**: 단일 run인지 (random_seed=42, 271_CONFIG_TRUTH.md에만 존재), 분산/CI 보고 불가 사실 — 본 문서 어디에도 없음.
2. **window score → point score 집계**: "mean-aggregated point-level scores" (`evaluator.py:2158`) — 모든 지표의 입력이 되는 결정적 단계인데 §④에 없음.
3. **Baseline 학습 패리티**: baseline의 epoch 수·early stopping·per-epoch eval cadence가 MAE와 동일/상이한지 — §③-3은 metric 계산 동일성만 다룸.
4. window=500/patch=10/stride 등은 271_CONFIG_TRUTH.md가 담당하나 **본 문서에서 단 한 번도 참조하지 않음** — split 표(§①)와 윈도잉은 불가분이므로 명시적 cross-ref 필요.

### M-5. (M-1과 동급 처리) WaDi 레지스트리 인용 off-by-one — `loaders.py:2697-2698` → 실제 **2698-2699**
단독으로는 MINOR이나 M-1과 함께 "레지스트리 인용 블록 전체 재검증 필요" 신호. SWaT `:2690`은 정확.

---

## 3. MINOR

1. **"총 112 dataset entries"** (§① 표제): 1(SWaT)+2(WaDi)+1(PSM)+28(SMD)+54(SMAP)+27(MSL) = **113**. SWaT dual-eval을 2로 세면 114. 112가 나오는 셈법이 없음 — 산식 명기 또는 수정.
2. **비교군 breakdown "SOTA legacy 7"** (§③): 실제 6 (anomaly_transformer, tranad, usad, dagmm, gdn, omnianomaly). 5+3+1+**6**+7=22로 본문 총계(22)와 정합. 현 표기는 23이 되어 자기모순.
3. **RESOLVED-1 "PSM metrics 149키"**: 실측 **153키** (None 0건). 실질 주장(_ar 정확히 10개, `pa_*_ar` 0건, pak 키 집합)은 모두 재확인됨 — 키 총수만 정정.
4. **`noisy.py:7-87`**: 파일은 **85줄**. class 정의 7, `use_noisy_labels` 52는 정확 — 끝 라인만 85로.
5. **`_compute_threshold_dependent` 인용 "evaluator.py:639+"**: 실제 def는 **637**.
6. **"원본 train 파일(전부 정상)"** (§② 공통 패턴): SMAP/MSL은 코드가 train 라벨을 명시적 zeros로 깔지만(`loaders.py:2603-2605`), PSM/SMD는 **train 라벨 파일 자체가 없어 '정상 가정'**임. 논문 정직성 서술상 "라벨 부재 → 정상 취급(분야 표준 가정)" 1줄 권장.
7. **Notion 덤프 경로**: `tool-results/...`는 세션-상대 경로 (실재 확인: `~/.claude/projects/.../0aa53593-*/tool-results/`). 후속 phase agent가 찾을 수 있게 절대경로 또는 `paper/` 하위 영속 사본 권장.

---

## 4. 검증 통과 항목 (적발 시도했으나 깨지지 않은 것)

**코드 근거 재추적 — 전부 일치 (위 적발 건 제외)**:
- Split `//2` 전수: SWaT `loaders.py:2005,2018` / WaDi `:2201` / SMD `:1153` / PSM `:1686` / SMAP·MSL `:2592-2595` — **라인 단위 정확**. run_boundaries `:1206(1205-1207 범위 내), :1740, :2613-2614, :2026-2032` ✓. swap 비활성 `run_base_experiments.py:331-333` ✓.
- normalonly: `unified_loader.py:34-36, 392, 410-415(절제), 417-421(boundary 등록), 470(라벨 0), 603(create_windows_from_segments)` — **전부 정확** (410-415, 417-421, 470은 한 줄도 안 어긋남). 변형 등록 `experiment_configs.py:65-66, 85-86` ✓. WEAK 4종 `:44` + "Q1-ONLY … raise RuntimeError on all-zero train_y" 주석 ✓.
- AR threshold: `evaluator.py:785, 790, 794, 769-770, 818-828` ✓; `PA_K_VALUES :831` ✓; excl22 `:2299, 2305-2306, 2324-2325, 2334, 2352, 2361-2363, 978-979, 984-985` ✓; worker `run_base_experiments.py:251, 749, 3446-3455, 3457, 3466-3471, 3501` — **전부 정확**.
- RESOLVED 블록 3건 재확인: ① `pa_0_f1` = F1-최적 threshold (`:929-930` → `:951-953` per-K에 동일 threshold 전달) ✓ + `pak_auc_f1`은 별도로 per-K 재최적화(tadpak, `:1001-1005` "best" 모드) — 두 서술은 **서로 다른 키에 대한 것으로 모순 아님** ✓; ② `pak_auc_prc_auc` (`:840-841` 첫 원소, `:1271` 산출; metadata에 `pak_auc_pr` 부재) ✓; ③ WaDi A2 = 123 (raw CSV 124 cols = 123+label 실측; metadata num_features=123; `prepare_raw_datasets.py:411` "131 = 3 meta + 127 features + 1 label") ✓.

**R28 수치 — 원 CSV에서 bit-exact 재현**: A1=495,000 / A2=449,919 / test=224,960 / test anomaly=42,864 (19.054%) / **region22=[2869,38769), 35,900 pts, 83.7533%, 15.9584%** / excl22 ratio=0.036835 / train anomaly 11,757 (1.633%), A2-front 21 regions (FEEDBACK-6 ✓). 계산 방법(테스트-로컬 좌표, 시간순 첫 region) 재현 가능하게 기록됨 ✓. metadata 교차치(full pak_auc_f1=0.9444 / excl22=0.6290) ✓.

**기타 수치 재현**: PSM (176,401/43,921/10,929=6.20%/13,452=30.628%/f1_ar=0.7616/affiliation_f1_ar=0.8012/threshold=0.001744) — 전부 일치. WaDi A1/A2 행수 4파일 전수 일치. SMD num_features 실측 29(machine-3-10)–36(machine-3-3), 22/28 — reconciler 정정값 정확. SMAP 54/MSL 27 채널 카운트 ✓. [271c] SMAP 5/54, MSL 5/27 (FEEDBACK-1) ✓. SWaT num_features=45 (full/excl22 metadata) ✓ — 단, FEEDBACK-7의 checkpoint shape (512,450)은 본 리뷰에서 미재현(metadata 2건으로 충분 판단).

**지표 매핑 (R24/R29) — 웹 재확인 전부 실재·정확**: VUS = Paparrizos/Boniol/Palpanas/Tsay/Elmore/Franklin, PVLDB 15(11) 2022 pp.2774-2787, DOI 10.14778/3551793.3551830 ✓; PA%K = Kim/Choi/Choi/Lee/Yoon, AAAI 2022 (OJS 20680, tadpak repo 실재) ✓ — 문서가 인용한 "random anomaly score → SOTA" 문구 abstract 원문 일치 ✓; Affiliation = Huet/Navarro/Rossi, KDD 2022 (arXiv:2206.13167, ACM DOI 10.1145/3534678.3539339) ✓; PA = Xu et al., WWW 2018 DOI 10.1145/3178876.3185996 ✓. 내부키↔코드 산출 위치 매핑 ✓ (vus `:739-744` 공식 패키지 slidingWindow=100, affiliation `:691-695` 출처 주석, PA=K0 `:610`).

**[N-COMP] 인용 충실성**: SMAP 355,905/217,925/0.70%/24.54%, MSL 95,271/36,775, SMD 708,405/708,420/4.16%, PSM 132,481+87,841 byte-level, safe-cut moved 4채널 목록, "Best-epoch 기준 pak_auc_f1", "SWaT는 excl22 사용" — 덤프 원문 grep으로 **전부 원문 일치**.

**R32 (§⑦)**: "미구현 — 진행할 예정" 정리 솔직 ✓ (grep 0건 주장 신뢰 가능 — `apply_normal50_noise :397-416` seed=123 region 단위, `noisy.py:52` split=='train' 분기 — 재확인 ✓). placeholder 설계는 기존 자산 기반으로 구체적이며, **실험 부재를 '한계'로 폄하하는 톤 없음 (A8 위반 없음)** ✓. "수치 발명 금지" 명시 ✓.

**R33 (§①)**: Simulation/Exathlon 코드 실재 (`run_base_experiments.py:256-262, 324-329, 366-375`; `loaders.py:1373` apps [1,2,4,5,6,9]) + 논문 제외 + RankAvg 재계산 필요(FEEDBACK-3) — 정확 ✓.

**R12/R31 (§③)**: Q1/Q3 매핑, "같은 라벨을 각자 패러다임의 최선으로" 논리, 단일 평가 경로 (`baseline_common.py:5, 553` — 라인 단위 정확) ✓.

---

## 5. 수정 요구 (재심 조건)

| # | 심각도 | 조치 |
|---|---|---|
| B-1 | BLOCKER | §①/§②/FEEDBACK-5의 safe-cut 서술을 "최근접 안전 지점으로 이동(무제한 탐색) + 최소 10-step clearance"로 교체하고 실측 이동량(D-16 +166, M-1/M-2 −39, S-2 +8; SMAP 0) 표 추가. "분할 비율에 실질 영향 없음" → 실측 기반 한정 서술로. `docs/DATASET.md:1151` 동반 오류는 ERRATA에 기록 (코드 수정 아님). |
| M-1/M-5 | MAJOR | `loaders.py:2810-2812`→`2742`, `2697-2698`→`2698-2699`. |
| M-2 | MAJOR | §⑤-4 인용 삭제 또는 실제 문헌 근거로 교체. |
| M-3 | MAJOR | best-epoch = test-split 지표 선정 사실 명기 + 논문 서술 방식 REQUEST 추가. |
| M-4 | MAJOR | 반복 수/seed, score 집계(mean), baseline 학습 패리티 추가 (또는 271_CONFIG_TRUTH cross-ref + 부재 항목은 신규 기재). |
| 1–7 | MINOR | 112→정확한 셈법, legacy 7→6, 149→153키, noisy.py 87→85, 639→637, train '정상 가정' 1줄, Notion 덤프 절대경로. |
