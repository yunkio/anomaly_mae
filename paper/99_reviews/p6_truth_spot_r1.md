---
phase: 6
agent: method-truth-spot
directives: [R17, A3]
round: r1
last_modified: 2026-06-11
target: paper/05_manuscript/MANUSCRIPT_v3.md (diff vs MANUSCRIPT_v2.md)
scope: §3 Method · §4 Experiments · Appendix A/C 변경 문장 전수 + 특별 정밀 3건 (§B.1은 Q1/Q3 정밀 검사 범위로 포함)
truth_sources:
  - paper/01_research_understanding/271_CONFIG_TRUTH.md (r4)
  - paper/01_research_understanding/EXPERIMENT_PROTOCOL_TRUTH.md (r4)
  - paper/99_reviews/p6_style_fixlog_r1.md (충돌 조정 §5, 의미 보존 거부 §6 맥락)
  - code spot-checks (read-only): mae_anomaly/loss.py:254-340
verdict: "PASS — BLOCKER 0 · MAJOR 0 · MINOR 3 (전부 기록성; 수정 불요)"
---

# P6 Method-Truth Spot-Check r1 (문체 수정 회귀 검사)

## 0. 판정 요약

**PASS.** v2→v3 diff에서 §3·§4·App A/C의 변경 문장 전수(하단 §1, 47개 변경 지점)를 v2 의미 및 정본
(271_CONFIG_TRUTH r4, EXPERIMENT_PROTOCOL_TRUTH r4)과 대조한 결과 **의미 반전·정본 모순 0건**.
특별 정밀 3건(§2) 모두 정합. Q1/Q3 → 학술 명칭 교체는 본문 11개 출현 전부에서 방향 올바름
(반전 0건; 본문 잔존 Q1/Q3 문자열 0건 — frontmatter 메타데이터만 잔존).
placeholder 무결성 재확인: PH:NUM 고유 ID 31/31, \cite 89, [X.XX] 20 — v2와 동일.
일부 변경은 오히려 **정본 정확도를 개선**했다(§1-E2 "outward" 삭제, §1-C8 "genuinely" 삭제 등).
MINOR 3건은 기록성 메모이며 수정을 요구하지 않는다.

---

## 1. 변경 문장 대조표 (§3 · §4 · App A/C 전수)

표기: ✓ = v2와 기술적 의미 동일 + 정본 일치. ✓+ = 의미 동일 + 정본 정확도 개선. △ = MINOR 기록(§3 참조).

### §3 Methodology

| # | 위치 | v3 변경 내용 | 판정 | 근거 (정본/코드) |
|---|---|---|---|---|
| M1 | §3.1 윈도잉 | "yields" → "is segmented into sliding windows" | ✓ | 표기만 |
| M2 | §3.1 upper bound | dash 절 → 괄호 "(every anomalous timestep … is labeled)" | ✓ | 동일 명제 |
| M3 | §3.1 | "outputs" → "produces a per-timestep anomaly score" | ✓ | 동일 |
| M4 | §3.1 중심 과제 | "the recovery of multi-channel correlation structure" → "Recovering the **normal** multi-channel correlation structure" (분리·"normal" 추가) | △ (MINOR-1) | §3.4 "joint normal correlation structure"·CONFIG_TRUTH §VIII(teacher가 정상 패턴 학습)와 정합하는 **명확화 방향**의 추가 — 모순 없음 |
| M5 | §3.2 5블록 | "training-only label-guided module that couples the Student branch" → "training-only adversarial branch that couples the **Student decoder's hidden states** to a window-level anomaly classifier" | ✓+ | CONFIG_TRUTH §VI: AnomalyClassifierHead "called on student hidden" (`model.py:1150-1154`); fixlog C6(B-031 encoder 오기 거부)의 올바른 합성 |
| M6 | §3.2 stop-gradient | 1문 → 2문 분리; "underpinning the anomaly score" → "on which the anomaly score is based" | ✓ | 동일 명제 (encoder는 Teacher 목적함수로만 최적화) |
| M7 | §3.3 마스킹 산식 | "A fraction — \|M\|=round(N×ρ), with ρ —" → "With masking ratio ρ, a fraction \|M\|=round(N×ρ)" | ✓ | CONFIG_TRUTH §VIII Masking: `round(50×0.15)=8` — 산식 보존 |
| M8 | §3.3 구조 불균형 | "*around* rather than *through*" 은유 → "gains little experience reconstructing anomalous correlation patterns" | ✓ | 의미 동일 (희소 → 무작위 마스킹이 anomaly 패치를 거의 선택 못함) |
| M9 | §3.4 Teacher | 조동사 복원 ("are added", "is passed") | ✓ | 문법만 |
| M10 | §3.4 capacity gap | "faithfully learns" → "accurately captures"; 1문 → 2문 ("Consequently, …") | ✓ | 주장 구조 보존 (matched-capacity 대비 비교, B.5 한정 유지) |
| M11 | §3.4 dual-λ | 1문(89단어) → 3문: λ_GRL "computed adaptively from the clamped ratio of main-loss to GRL-loss gradient norms, evaluated per batch and applied as the previous epoch's average"; λ_rev "sigmoid schedule … ≈0.02 to ≈1" | ✓ | CONFIG_TRUTH §VIII GRL Details: grad-ratio clamp[0,10]·per-batch·prev-epoch average (`trainer.py:751-765, 1317-1319`); λ_rev epoch250≈0.020→499≈0.9999; fixlog C5(B-036 EMA 오기 거부)의 plain-average 서술 보존 |
| M12 | §3.5 Eq.(1) 도입 | "matches the Student's output to" → "**penalizes the discrepancy** between the Student's output and the Teacher's detached output on this subset only" | ✓ | Eq.(1) = P_n 합산 MSE와 정확히 일치; `loss.py:255` normal_patch_mask 한정 실측 |
| M13 | §3.5 Eq.(1) 후속 | "excluded entirely" → "excluded from $L_{\mathrm{OD}}$" | ✓+ | 배제 범위를 수식 항으로 명시 — Eq.(1) 합산 범위와 정합 |
| M14 | §3.5 GRL 손실 | 1문(90단어) → 2문; "negates the gradient …, scaled by λ_rev" → "**multiplies the gradient by −λ_rev**"; "opposes the classifier's search for anomaly-discriminative features" → "**penalizes anomaly-discriminative information in the Student's hidden states**" | ✓ | Eq.(C.2) ∂h̃/∂h = −λ_rev·I 와 정확 일치; 방향 = 억제(suppression) — CONFIG_TRUTH §VI/부록1 #3 (reconciler 방향 정정과 동일 방향); focal "modulating factor derives from the class-prior-weighted cross-entropy" = `loss.py:337-339` `_p_t=exp(−_bce)` 실측 |
| M15 | §3.5 GRL 필요성 | "removes the demand that the Student *follow* the Teacher there" → "removes the requirement that the Student match the Teacher's output at those locations"; "shrinking … most informative" → "thereby reducing the discrepancy where it is most diagnostic" | ✓ | 동일 논증 사슬 (visible-context → 간접 경로 → GRL 차단) |
| M16 | §3.6 LOO | "all patterns forwarded in parallel through the batch dimension" → "all N masked variants processed as one batched forward pass"; 비용 "approximately N single-window forward passes" 유지 | ✓ | 동일 메커니즘·동일 비용 주장 |
| M17 | §3.6 Eq.(4) 주변 | "where r̄ and d̄ are the means …" → "Here, r̄ and d̄ are the means of the patch-level reconstruction errors and discrepancies, **respectively**, …" (분리) | ✓ | respectively 대응 올바름(r̄↔recon, d̄↔disc); CONFIG_TRUTH §VIII Anomaly Score: `scaled_disc = disc·(recon_mean/disc_mean)`, ε=1e-4 양변 가산 — Eq.(4)와 일치 |
| M18 | §3.6 Eq.(6) 주변 | "— indexing the covering windows by u —" → "where u indexes the covering windows" | ✓ | Eq.(6) 기호와 일치; mean 집계 = PROTOCOL_TRUTH §④-2 |
| M19 | §3.6 앙상블 | "single-window reconstruction-context variation" → "context-dependent reconstruction variance" | ✓ | 동일 효과 서술 |

### §4 Experiments

| # | 위치 | v3 변경 내용 | 판정 | 근거 |
|---|---|---|---|---|
| E1 | §4.1.1 Datasets | "SMAP/MSL" → "SMAP, and MSL" (6 family 명시); "113 learning units" → "113 **entities**"; "114 evaluation units" → "114 **evaluation conditions**" | ✓ | PROTOCOL_TRUTH §①: 1+2+1+28+54+27=113 학습 단위, dual-eval 포함 114 평가 단위; fixlog C11 (audit의 WaDi-2-families 오류 정정) — 가족 6 = SWaT·WaDi·PSM·SMD·SMAP·MSL 정합 |
| E2 | §4.1.1 재분할 | "joins" → "is appended to"; "labeled anomalies are **genuinely** present" → "are present" + ratios 분리 "range from 0.52% to 6.20% (SMD per-machine values pending)"; "shift **outward** when it falls within ten timesteps" → "shift when …" + "4 of 81 channels … largest shift being 166 timesteps" | ✓+ | PROTOCOL_TRUTH §①(WaDi A1 0.52% min, PSM 6.20% max, SMD per-machine 상이)·§②(이동 4/81 전부 MSL, +166/−39/−39/+8 — **양방향**이므로 "outward" 삭제는 정본 정확도 개선; trigger = region ±10 이내) |
| E3 | §4.1.1 재정의 | "is a redefinition of the benchmark, not a use of held-out labels" → "redefines the benchmark partition rather than exploiting held-out labels"; NRdetector 7:3 dash→괄호 | ✓ | 동일 명제; R13 방어 보존 |
| E4 | §4.1.1 SWaT dual | "(excl22; …) — same model, same scores, only the evaluation mask differs, identically for all baselines" → "a condition denoted excl22 (…): the model and scores are identical, only the evaluation mask differs, and the same mask is applied to all baselines"; "Table 2 ranks under excl22" → "ranks methods under the excl22 condition" | ✓ | PROTOCOL_TRUTH §⑥: 단일 학습+dual eval, baseline 동일 dual 조건, 83.75%/19.05%→3.68%, 변별은 excl22 — 전부 보존 |
| E5 | §4.1.2 분산 | "We report no cross-seed variance … ; only the random-score baseline is averaged over five runs" → "… **for the main results**, a limitation …, except for the random-score baseline, which is averaged over five runs" | ✓ | PROTOCOL_TRUTH §④-실행 1 (RM-1): 단일 run/seed=42, random만 5-run mean±std — scope 절 추가는 사실 정밀화 |
| E6 | §4.1.2 epoch 비대칭 | 1문 → 3문: 10/50 매 epoch, 500 every-5; "share the same selection criterion, run to their full budget without early stopping, and are reported at their best evaluated epoch"; "budgets reflect convergence — CSMAD needs the 250-epoch warmup" | ✓ | PROTOCOL_TRUTH §④-실행 3 (RB-1 정정 후 사실): 500/10/50·eval 5 vs 1·공통점 ⓐⓑⓒ; CONFIG_TRUTH §VIII Training warmup 250 |
| E7 | §4.1.2 test-set 선택 | "Uniform across methods, this leaves relative rankings unaffected but may bias absolute estimates optimistically" → "Because the criterion is applied uniformly to all methods, relative rankings are unaffected; however, absolute estimates may be optimistically biased" | ✓ | PROTOCOL_TRUTH §④ M-3 (test-set model selection 공개 의무) 보존 |
| E8 | §4.1.2 threshold | 1문(67단어) → 3문; "α is the **measured** anomaly fraction of the evaluation span"; xu2022 대비 절 재배치; "never used in training" 유지 | ✓ | PROTOCOL_TRUTH §⑤: ar=eval-span label mean, (1−ar) quantile, 전 모델 동일, PA%K/VUS 무관 — 대비 구조(xu2022=fixed ratio on validation vs ours=measured) 보존 |
| E9 | §4.1.3 5지표 | 1문(155단어) → 지표당 1문; "removing dependence on any particular K" → "eliminating sensitivity to the choice of K"; "most reliable single TSAD measure" → "most reliable single measure for **time-series anomaly detection**"; Affiliation "measuring the temporal distance" → "quantifying the temporal **proximity**" | ✓ | PROTOCOL_TRUTH §④ 매핑표·상호보완성 재료와 정합; K∈{0,1,…,100} step-1 격자 무변경(E-2 정합); liu2024elephant 주장 scope 보존(fixlog C8); proximity↔distance는 동일 affinity 정의의 양면 — §A.2 정의와 일치 |
| E10 | §4.1.3 직교 관점 | "— with distinct failure modes; reporting all five prevents …" → "Each perspective has distinct failure modes; reporting all five prevents any single failure mode from going undetected." | ✓ | R29 논리 원형 유지 (fixlog C9: E-03의 주장 교체 거부 확인) |
| E11 | §4.1.3 PA F1 | "marked (oracle) for its F1-optimal threshold" → "labelled (oracle) to indicate that its threshold is selected to maximize F1"; "never used for ranking" 유지 | ✓ | PROTOCOL_TRUTH §④ (R29: 제시하되 참고 안 함) |
| E12 | §4.1.4 baseline 구성 | 1문 → 3문 분해; "deep TSAD" → "deep MTSAD"; 9+6+7 구성·간략 DAGMM·§A.1 위임 유지 | ✓ | PROTOCOL_TRUTH §③ STANDARD_BASELINES (5+3+1+6+7=22) + weak 4 |
| E13 | §4.1.4 weak 조건 | "evaluated under the Q1 condition only, since removing all labeled anomalies (Q3) would eliminate the positive windows" → "under the **contaminated-training condition** only (defined below), since the **anomaly-excised condition** removes all labeled anomalies and would eliminate the positive windows" | ✓ | **방향 정확** — §2-① 참조; PROTOCOL_TRUTH §③ R31-1 (weak 4종 Q1 전용, normalonly은 구조적 부적합) |
| E14 | §4.1.4 비교 조건 | "Q3 (normal-only) condition" → "**anomaly-excised condition**" (bold 정의); "Q3 grants" → "the anomaly-excised condition grants"; "Q3 excision" → "Anomaly excision"; "reports Q1 results" → "reports results under the complementary **contaminated-training condition** (training on the full contaminated stream without excision)" | ✓ | §2-① 참조; 정의문이 정본 의미(Q3=절제, Q1=무절제)와 1:1 |
| E15 | §4.2 본문 | NUM-007 문장 분리; "(Q3)" → "(anomaly-excised condition)"; "(Q1)" → "(contaminated-training condition)"; "on SWaT excl22" → "under SWaT's excl22 condition" | ✓ | §2-① 참조; placeholder ID 전부 보존 |
| E16 | §4.2 protocol-effect | "Under (i) the label-dependent pathways self-deactivate with the configuration held fixed (random masking, all-normal OD loss, no GRL loss)" → "Under condition (i), with the configuration held fixed, the label-dependent pathways are **automatically inactive** (random masking, all-normal $L_{\mathrm{OD}}$, no GRL loss)" | ✓ | 동일 메커니즘; 라벨 0 → priority 전부 η(무작위), P_n=M(all-normal), GRL positive 0 → 항 생략(`loss.py:294-302` 실측). **(i)/(ii) 축은 Q1/Q3 명칭과 미혼동** — "standard clean-train split"/"contaminated protocol" 무변경 (fixlog §0.2 별개 축 보존 확인) |
| E17 | §4.3 Row 3/4 | "Without it" → "Without anomaly-priority masking"; "the bifurcated signal" → "the **selective distillation signal** that drives the Student to deviate … while mimicking …" | ✓ | §2-② 참조 |
| E18 | §4.4 Design | dash → 괄호; "region granularity, matching operational records" → "at region granularity, consistent with how operational logs record fault events" | ✓ | 동일 설계 서술 |
| E19 | §4.4 3속성 | "support robustness: (i)…(ii)…(iii)…" → "bound this degradation. First… Second… Third…"; GRL 항 "draws its positive supervision … — batches without a labeled positive **skip** the term —" → "takes its positive supervision …; batches without a labeled positive **omit** the term entirely" | ✓ | `loss.py:294-302` 실측: `if _pos_count == 0: … grl_cls_loss_tensor: None` ("No anomaly in this batch → skip GRL loss") — 코드 사실 그대로; R32 3속성 논리 사슬 보존 |
| E20 | §4.4 결과 | "…, confirming reversion to a pure reconstruction-based detector without falling below the unsupervised floor" → 별도 문장 "This confirms that CSMAD reverts to a purely reconstruction-based detector without falling below the unsupervised floor." | ✓ | 동일 주장; NUM-027 보존 |
| E21 | §4.5 | "four aligned traces —" → ": …"; "fail to track the Teacher" → "fail to **replicate the Teacher's output**" | ✓+ | discrepancy는 출력 수준(o^T vs o^S, Eq.(1)/§3.6) — 더 정밀한 서술 |

### Appendix A

| # | 위치 | v3 변경 내용 | 판정 | 근거 |
|---|---|---|---|---|
| A1 | §A.1 config | "113 learning units" → "113 entities"; "split proportions **implied by**" → "**determined by** the protocol" | ✓ | CONFIG_TRUTH §III: 가변 3키 = num_features(F)·grl_pos_weight(w_+)·sliding_window_train_ratio(분할 비율) — 매핑 1:1 유지 |
| A2 | Table A.1 | `d_\text{model}` → `d_{\mathrm{model}}` | ✓ | 표기만; 값 512 무변경 |
| A3 | §A.1 budgets | "states the per-group budgets disclosed in" → "summarizes the per-group training and evaluation budgets reported in" | ✓ | 동일 |
| A4 | §A.1 baseline 목록 | "nine detectors **adopted from** the protocol study of \cite{sarfraz2024quovadis} — five simple (…) three lightweight (…) GCN-LSTM —" 재구성; "deep TSAD" → "deep MTSAD" | ✓ | PROTOCOL_TRUTH §③: simple5+neural3+GCN-LSTM1 = quovadis 계열 9 — 귀속 범위 동일(v2도 동일 9종에 결부) |
| A5 | §A.1 runs | "all other methods are single runs" → "use a single run"; random "averaged over five independent runs (mean ± std)" 유지 | ✓ | PROTOCOL_TRUTH §④-실행 1 |
| A6 | §A.1 통일 평가 | "consume the identical data partitions" → "receive …"; "precluding implementation-level metric divergence" → "eliminating implementation-dependent metric discrepancies" | ✓ | PROTOCOL_TRUTH §③-3 (단일 compute_full_metric_set) |
| A7 | §A.1 SWaT note | 증거-선행 재배열 (45 = 51 − constant 6 무변경) | ✓ | PROTOCOL_TRUTH §① SWaT 행 |
| A8 | §A.2 PA%K | "K = 100 point-wise scoring" → "K = 100 **recovers** point-wise scoring" | ✓ | strict > 정의상 K=100이면 adjustment 미발동 = point-wise — 정확 |
| A9 | §A.2 VUS | "tolerance window 100 after min–max normalization" → "a tolerance window of 100 **timesteps**, after min–max normalization of scores" | ✓ | PROTOCOL_TRUTH §④ (slidingWindow=100, min-max 정규화 후); 계산 순서 보존 (fixlog SA-005 거부 확인) |
| A10 | §A.2 Affiliation | "convert the temporal distance … into per-event affinity scores" → "measure the temporal proximity …, converted into per-event affinity scores" | ✓ | 동일 정의(거리→affinity); harmonic mean 유지 |
| A11 | §A.2 threshold/집계 | dash → 세미콜론; "consume" → "are computed on" | ✓ | 동일 |
| A12 | §A.3 | "test prefixes" → "test-stream prefixes"; boundary-aware "Q3 condition" → "anomaly-excised condition" | ✓ | §2-① 참조 (절제 경계 보호 = 절제 조건 소속 — 방향 정확, `unified_loader.py:417-421`) |
| A13 | §A.4 | heading 풀어쓰기; "evaluation-local positions" → "positions [2,869, 38,769) (indexed within the evaluation span)" | ✓ | PROTOCOL_TRUTH §⑥: test-local [2869, 38769), 35,900 pts, 83.75%/15.96% — 전부 보존 |

### Appendix C

| # | 위치 | v3 변경 내용 | 판정 | 근거 |
|---|---|---|---|---|
| C1 | Eq.(C.1) 주변 | "$[e_0,e_1]$ the student-training phase … and τ its progress" → "… **is** the student-training phase … and **τ is its normalized progress**" | ✓ | clip(·,0,1)이므로 "normalized" 정확; CONFIG_TRUTH §VIII λ_rev 행 (p=clip((e−250+1)/250,0,1)) |
| C2 | Eq.(C.2) 주변 | "it scales and negates the gradient" → "it **multiplies the gradient by −λ_rev**" | ✓ | Eq.(C.2) = −λ_rev·I 와 문자 그대로 일치 |
| C3 | Eq.(C.3) 주변 | "With ŷ_i …, ℓ_i …, and w_+ …," → "Let ŷ_i … be …, ℓ_i …, and w_+ …:"; focal 대비 "here p_t := e^{−ℓ_i} derives from" → "the present variant **defines** p_t := e^{−ℓ_i} from the pos-weight-adjusted BCE"; "part of the present design rather than an external import" → "introduced as part of the present design rather than adopted from prior work" | ✓ | `loss.py:337-339` (_p_t=exp(−_bce), γ=2) 실측; w_+ 정의 절 무변경 |
| C4 | Eq.(C.4) 후속 | "the reversal coefficient and the loss weight **act multiplicatively** and remain distinct quantities" → "**enter the gradient multiplicatively** and remain distinct quantities" | ✓ | CONFIG_TRUTH §VIII r4 신설 행: 도달 gradient = −λ_rev × λ_GRL_eff × ∂L_cls/∂(GRL 출력) — 곱 구조·별개성 둘 다 보존 (fixlog C10: don't-conflate 경고 유지 확인) |
| C5 | §C.2 | `d_\text{model}` → `d_{\mathrm{model}}` (값 512 무변경) | ✓ | 표기만 |
| C6 | Table C.2 +4행 | r̄/d̄ "per-entity means of r_i/d_i over all (window, patch) pairs (Eq. 4)"; ε "scaling stabilizer in Eq. (4) (=10^{-4})"; c "score combination ratio (Eq. 5; =4)" | ✓ | 본문 Eq.(4)/(5)와 1:1; CONFIG_TRUTH §VIII Anomaly Score: ε=1e-4, `score_recon_disc_ratio=4.0` — **신규 행 수치 정본 일치** |

---

## 2. 특별 정밀 3건

### ① Q3="anomaly-excised" / Q1="contaminated-training" — 전 출현 방향 검사

**정본 기준** (EXPERIMENT_PROTOCOL_TRUTH r4 §③, [N-COMP] §2.2; `unified_loader.py:34-36, 392-485`):
Q3 = normalonly = **train에서 anomaly region 절제**(라벨의 '제거형' 활용; train 라벨 전부 0) /
Q1 = full = **무절제 오염 스트림 그대로 학습**(라벨 미사용). fixlog §0.2의 검증 트레일과 일치.

v3 본문·부록의 전 출현(grep 전수 — 본문 잔존 "Q1"/"Q3" 0건) 방향 판정:

| 행 | 출현 (v3) | 요구 의미 | 판정 |
|---|---|---|---|
| L441 | §4.1.4 weak group "under the contaminated-training condition only … since the anomaly-excised condition removes all labeled anomalies" | weak=Q1 전용 / Q3=라벨 제거 | **정방향** ✓ |
| L445 | §4.1.4 "main comparison uses the **anomaly-excised condition** …: labeled anomaly regions are excised" | Q3=절제 | **정방향** ✓ |
| L446 | §4.1.4 "the anomaly-excised condition grants each unsupervised method this most favorable use of the labels, while CSMAD trains on the full contaminated set without excision" | Q3=unsupervised 최선 / CSMAD=무절제 | **정방향** ✓ |
| L447 | §4.1.4 "Anomaly excision reduces baseline training volume … §B.1 reports results under the complementary **contaminated-training condition** (training on the full contaminated stream without excision)" | Q1=무절제 | **정방향** ✓ |
| L456 | §4.2 "strongest unsupervised competitor (anomaly-excised condition)" | main 비교 unsupervised=Q3 | **정방향** ✓ |
| L458 | §4.2 "NRdetector …, the closest weakly supervised comparison (contaminated-training condition)" | weak=Q1 | **정방향** ✓ |
| L672 | §A.3 "guards the excision boundaries of the anomaly-excised condition" | 절제 경계=Q3 소속 | **정방향** ✓ |
| L699 | §B.1 heading "Contaminated-Training (No-Excision) Condition Results" | Q1 | **정방향** ✓ |
| L701 | §B.1 "The anomaly-excised condition … grants … excision of contaminated training regions" | Q3=절제 | **정방향** ✓ |
| L702 | §B.1 "complementary contaminated-training condition, in which the same 22 … train on the full contaminated stream without excision" | Q1=무절제 | **정방향** ✓ |
| L706 | TAB-B1 caption "Δ … relative to the anomaly-excised condition of Table 2 (positive = contaminated-training better)" | v2 "(positive = Q1 better)"와 동일 부호 규약 | **정방향** ✓ |

**조건 의미 반전 출현 0건 — BLOCKER 해당 없음.** 또한 §4.2 protocol-effect의 별개 축
"(i) standard clean-train split / (ii) the contaminated protocol"은 무변경으로 보존되어
Q1/Q3 축과의 혼동(briefing이 우려했던 conflation)이 발생하지 않았다. → MINOR-3 (명칭 근접성 메모)만 기록.

### ② "loss bifurcation" 대체 표현 ↔ loss.py 정합

코드 사실 (`loss.py:254-261` 실측 + CONFIG_TRUTH §VI): OD 손실은 `normal_patch_mask = patch_is_normal *
patch_has_masked` 한정으로 계산되고, anomaly 측은 271에서 `use_grl ∧ grl_disable_anomaly_loss`로
**하드 제로** — 즉 활성 동작은 "Student 모방 목적함수를 정상 마스킹 패치로 한정".

| 출현 | v3 표현 | 판정 |
|---|---|---|
| Abstract | "a Student imitation loss restricted to normal patches" | ✓ 코드 사실 그대로 (Eq.(1) P_n 한정) |
| §1 기여 2 | "*loss bifurcation*, which restricts the Student decoder's imitation objective to normal-patch outputs" (정의 사용처 — v2 무변경) | ✓ |
| §3.5 heading | "Why gradient reversal is necessary beyond loss bifurcation" (정의된 용어 사용 — 무변경) | ✓ |
| §4.3 Row 4 | "the **selective distillation signal** that drives the Student to deviate from the Teacher on anomalous patches while mimicking it on normal ones" | ✓ — "selective"는 P_n-한정 distillation의 정확한 형용; "deviate/mimic" 구도는 v2 문구 그대로(§3.5 "free to deviate"와 동일 강도 — 능동적 push-up 아님) |
| §5 | "loss bifurcation that restricts Student mimicry to normal patches" | ✓ (B-063의 "bifurcation toward" 비문만 제거) |

부수 확인: §4.4 "batches without a labeled positive omit the term entirely" = `loss.py:294-302`
(`_pos_count == 0 → grl_cls_loss_tensor=None`, 주석 "No anomaly in this batch → skip GRL loss") 실측 일치.
Eq.(3) 후속 "the GRL term contributes only when the batch contains a positive window" (무변경)과도 정합. **모순 없음.**

### ③ 수식 주변 문장 변경 ↔ 수식 정합

| 수식 | 주변 변경 | 정합성 |
|---|---|---|
| Eq.(1) | "penalizes the discrepancy … on this subset only" / "excluded from L_OD" | ✓ 합산 범위 P_n과 일치 |
| Eq.(4) | "Here, r̄ and d̄ … respectively … computed once per entity" + Table C.2 신규 r̄/d̄/ε 행 | ✓ Eq.(4) 기호·ε=10⁻⁴, 코드식 `(mean(recon)+1e-4)/(mean(disc)+1e-4)` 일치 |
| Eq.(5) | "The two components are then combined at a fixed ratio" (분리) + Table C.2 c=4 행 | ✓ c=4 = `score_recon_disc_ratio=4.0` |
| Eq.(6) | "where u indexes the covering windows" | ✓ 첨자 u 정의와 일치 |
| Eq.(C.1) | "τ is its normalized progress" | ✓ clip(·,0,1) |
| Eq.(C.2) | "multiplies the gradient by −λ_rev" | ✓ −λ_rev·I 문자적 일치 (v2 "scales and negates"보다 정밀) |
| Eq.(C.3) | "Let … be …:" + "the present variant defines p_t := e^{−ℓ_i} from the pos-weight-adjusted BCE" | ✓ 수식 (1−e^{−ℓ})^γ·ℓ, γ=2와 일치; `loss.py:337-339` 실측 |
| Eq.(C.4) | "enter the gradient multiplicatively and remain distinct quantities" | ✓ −λ_rev·λ_GRL·∂L_cls/∂(GRL output) 곱 구조와 일치 |

**수식-본문 모순 0건.**

---

## 3. 발견 사항 (rubric 적용)

**BLOCKER: 0건.** **MAJOR: 0건.**

**MINOR (3건 — 전부 기록성, 수정 불요):**

1. **MINOR-1 (§3.1, 표 M4)**: "Recovering the **normal** multi-channel correlation structure" — "normal" 추가는
   순수 문체를 넘는 명확화이나, 방향이 정본(§3.4 Teacher의 정상 상관 구조 학습, CONFIG_TRUTH §VIII)과 정합하고
   후속 문장(anomaly 측 co-occurrence 활용)과도 모순되지 않음. 의미 모호화 아님 — 기록만.
2. **MINOR-2 (§B.1, 표 외)**: "most favorable use of the training labels" → "most favorable **pre-processing
   benefit** from the available labels" — 라벨 활용 일반 → 전처리 이득으로 외연이 약간 좁아졌으나, 절제(excision)는
   실제로 전처리 단계 조작이므로 사실 정확 (§4.1.4 본문에는 "most favorable use of the labels" 원표현 보존).
3. **MINOR-3 (명칭 근접성)**: 신명칭 "contaminated-training condition"(Q1)과 기존 "the contaminated
   protocol"(§4.2 protocol-effect 축 (ii))·"contaminated semi-supervised setting"이 한 절 안에 공존 —
   의미 반전·정본 모순은 없으나(각 사용처에 정의·괄호 부연 존재), Phase 7 결과 채움 시 독자 혼동 방지 재점검 권장.

**정본 정확도 개선으로 판정한 변경**: E2("outward" 삭제 — 실측 이동 부호 양방향, PROTOCOL_TRUTH §② B-1),
E21("track" → "replicate the Teacher's output" — 출력 수준 discrepancy 명시), M5(GRL 결합점 = Student
**decoder** hidden states 명시), M13/C2(수식 문자적 정합 강화), C6(Table C.2 기호 완결성).

## 4. 판정

**PASS** — Phase 6 문체 수정(v2→v3)은 §3 Method·§4 Experiments·Appendix A/C(및 §B.1)의 기술적 의미를
보존했으며, 모든 변경 문장이 271_CONFIG_TRUTH r4·EXPERIMENT_PROTOCOL_TRUTH r4와 일치한다.
Q1/Q3 명칭 교체 11개 출현 전수 정방향, loss-bifurcation 대체 표현 loss.py 정합, 수식-본문 모순 0건.
게이트 차단 사유 없음.
