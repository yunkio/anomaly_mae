---
phase: 5
agent: coverage-auditor
directives: [M10]
last_modified: 2026-06-11
target:
  - paper/05_manuscript/MANUSCRIPT_v2.md (Phase 5 final)
  - paper/05_manuscript/PLACEHOLDER_REGISTRY.md (v2-r3)
  - paper/99_reviews/p5_fixlog_r2.md
inputs_reviews:
  - p5_method_truth_r1.md (23) / p5_adversarial_r1.md (28) / p5_citation_back_r1.md (24)
  - p5_citation_gap_r1.md (15) / p5_plagiarism_r1.md (9)
canon: 271_CONFIG_TRUTH.md r4 / EXPERIMENT_PROTOCOL_TRUTH.md r4 / RESEARCH_SYNTHESIS.md r3 / PAPER_BLUEPRINT.md r3 / refs.bib (49)
verdict: "CONDITIONAL FAIL — Directive 커버리지 전수 충족·고위험 재추적 전건 PASS·A8/인용/R9 스윕 클린, 단 수정 라운드 마감 검증에서 MAJOR 1건 적발 (citation-back PARTIAL-14: fixlog에 FIXED로 기록되었으나 v2 미반영). 해당 1건 적용 + fixlog 정정 시 PASS."
---

# P5 Coverage Gate r1 — 수정 라운드 마감 + Directive 전수 + A8 재스윕

> 임무: 깨뜨리기. 모든 판정은 1차 소스(코드 file:line / canon 문서 / reference card / v2 원문 grep) 직접 재검증 기반. 코드 read-only 준수, `paper_legacy/` 미접촉.

---

## 1. 수정 라운드 마감 검증 (리뷰 5건 ↔ fixlog ↔ v2 — 1:1 대조)

### 1.1 처리 기록 완전성 (리뷰 발견 → fixlog 행 존재)

| 리뷰 | 발견 수 | fixlog 등재 | 누락 |
|---|---|---|---|
| p5_method_truth_r1 (B1–B9, M1–M6, m1–m8) | 23 | 23 (§1) | 0 |
| p5_adversarial_r1 (BP/R8/R1/R9/R10/ARG/PH/ELR/VAG/CRIT/PLAG) | 28 | 28 (§2) | 0 |
| p5_citation_back_r1 (U1–U6 + PARTIAL 1–18) | 24 | 24 (§3; U6=집계 항목, 무조치 명기) | 0 |
| p5_citation_gap_r1 (G-01–G-15) | 15 | 15 (§4) | 0 |
| p5_plagiarism_r1 (F1–F8 + SC-06) | 9 | 9 (§5) | 0 |

**처리 기록 누락 0** — 전 발견이 FIXED / REJECTED+사유 / NO-CHANGE+사유 중 하나로 등재됨.
부기(기록 정밀성, MINOR): fixlog 톱라인 "94 처리 단위(82/5/7)"는 섹션별 헤더 합(98 = FIXED 85 / REJ 5 / NC 8)과 불일치 — 교차 중복(F4=PLAG-01, F5=U5, F7=C-035/G-09, PH-02=BP-05 등) 4건 dedup이 명시되어 있지 않음. 실질 영향 없음.

### 1.2 실반영 검증 (v2 원문 string-level 확인)

FIXED 전 항목을 v2에서 직접 확인했다. 대표 확인 문자열(전수 추적은 §2 재추적표와 본 절 표에 분산):

| 항목 | v2 근거 (line, MANUSCRIPT_v2.md) | 판정 |
|---|---|---|
| B-1 PA%K 격자 | :380 "$K \in \{0, 1, \ldots, 100\}$" / :549 "101-point grid" | ✅ |
| B-2 LayerNorm 제거 | :229 "$\mathbf{z}_i = \mathbf{E}\,\mathrm{vec}(\mathbf{P}_i) + \mathbf{b}$" / :499 "Linear (flatten + projection)" | ✅ |
| B-3 Eq(4) 전역 평균 | :308 "over all (window, patch) pairs of the evaluated series, computed once per entity" | ✅ |
| B-4 Eq(1) 1/(\|Pₙ\|·s·F) | :269 분모 $\|P_n\| \cdot s \cdot F$ | ✅ |
| B-5 w₊ patch ratio | :697 "per-entity normal-to-anomalous patch ratio (… floored at $10^{-3}$)" | ✅ |
| B-6 "512" 제거 | :367 "original implementation preset (Table A.3)" / :524–525 "model-specific" ×2; "512 for baselines" grep 0 | ✅ |
| B-7 test-time LOO | :237 "training-time mechanism; at test time … deterministic leave-one-out" / :505 Table A.1 동일 | ✅ |
| B-8 §4.4 (ii) | :445 "batches without a labeled positive skip the term entirely … treated as negatives" | ✅ |
| B-9 complementary masking 삭제 | :478 "reducing this inference cost is a natural avenue"; 'complementary masking' prose grep 0 | ✅ |
| M-1 AR-threshold 완화+차이 부기 | :376 "(which sets a fixed ratio on a validation split, whereas our $\alpha$ is the measured fraction…)" | ✅ |
| M-2/M-3/M-4/m-1/m-2/m-4/m-7/m-8 | :344 SMD pending / :493 3-key 가변 / :282 masked-region 괄호 / :591 "set to zero by the loading pipeline" / :568 "unadjusted (pre-PA)" / :319 "ensemble effect" / :221 "couples the Student branch" / :258 "per batch … previous epoch's average" | ✅ ×8 |
| M-5/M-6 notation | bare $d$/$w$/$r$/`r_m`/`s_t`/`W_emb` grep 0; ρ/τ/α/a_t/u/E/d_model + upright T/S 적용; Table C.2 (:742–765) 등재 | ✅ |
| BP-01 | :348 "This practice has precedent: NRdetector … at a 7:3 ratio — so that anomalous events fall within the training stream." (독립 문장; 시간순서 단정 없음 — card verbatim "split … by 7:3 ratio" §5.1 + "anomalies are embedded within the training data" §1 지지) | ✅ |
| BP-02/R9-01 3-way 분리 | 각주 :193 구조차("branches off from within the teacher decoder after its first transformer block" — card 발췌 2 정합) / §3.5 :281 작동계층 1문장 / §2.3 본문 :191 중립 적응문만 | ✅ |
| BP-03 3단 구조 | :208–210 "general case … label-availability upper bound … Section 4.4 then validates the general case" (SYNTHESIS §②-1/2/3 그대로) | ✅ |
| BP-04 | :413 "Two findings follow, one per condition." + 조건별 독립 문장 :414–415 | ✅ |
| R8-02 | :150 "(Earlier semi-supervised models … not adversarially through the gradient of the representation itself \cite{xue2022fewpositive, huang2022slavae}; Section 2.2.)" | ✅ |
| R1-01 | :181 "Our use of labels differs in kind: … the label shapes the gradient of a masked-reconstruction pretext" | ✅ |
| R1-02 / R1-03(부분) | :388 TFMAE 브리지 / :157 bullet 3 "a design intended to make … (quantified in Appendix B.5)" | ✅ |
| R10-01 | :212 "labeled anomaly events arise naturally from the operational logs … making the recovery of multi-channel correlation structure the central learning challenge" | ✅ |
| ARG-01 / ARG-02 | :287–289 encoder-문맥 오염 경로 재논증 / :446 "the discrepancy pathway and the adversarial suppression weaken together … bounding the degradation from below" | ✅ |
| PH-01/PH-03/ELR-01/ELR-02 | "Figure 1 contrasts" grep 0 / registry v2-r3 ② GRL 부착 레이블 / :87 "a gradient reversal layer" / Highlights 실측 123·124·120·121·≤125 | ✅ |
| VAG-01/VAG-02/CRIT-01/CRIT-02/PLAG-01/PLAG-02 | :251 "than a matched-capacity decoder would" / §4.2 appositive 부재 / :361 CI 한계 명시 / :380 "(the F1-optimal-threshold variant is excluded from all rankings)" / :189 SDMAE 재구성 / :282 "(exact form: Eq. C.3)" | ✅ |
| U1–U5 (citation-back) | :139 재서술+분리 / :179 two-step→bekker·elkan 재배치 / :181 xue AR 재서술 / :394 bekker 제거+자체 논거 / :189 zhang 수준화 | ✅ |
| G-01..G-14 | :135, :144("closest"), :175, :181, :181(darban), :215([^cs-fn]), :242(prenorm), :281(ristea), :342(원전 5종), :394, :439, :591, :532(TranAD repo) | ✅ |
| F1/F2/F3/SC-06/F7 | :283+:692 ganin ×2 / :183 "identifies this as a novel setting…" / :131 "exhaustive point-level annotation … is infeasible at scale" (6-gram 해체) / :189 abstract 범위 재서술 / :281 inline ristea | ✅ |
| **PARTIAL-14 (TFMAE)** | **:187 "similar patch-and-mask operations in some time-series models \cite{fang2024tfmae}" — v2_draft :178과 동일 (미변경)** | ❌ **미반영** |

### 1.3 ❌ MAJOR 적발 — citation-back PARTIAL-14 미반영 (fixlog 기록 허위)

- fixlog §3 행 14: `"similar masking-based reconstruction objectives in some time-series models" (TFMAE는 patch 단위 아님 — card 정합)` — **FIXED로 기록**.
- 실측: v2 :187 = v2_draft :178 **문자 단위 동일** ("patch-and-mask operations" 잔존).
- 사실 검증: `fang2024tfmae.md` abstract verbatim — TFMAE의 masking은 "**window-based temporal masking** strategy and an **amplitude-based frequency masking** strategy" — patch 단위 아님. 따라서 현 문장은 인용 논문에 patch 단위 masking을 오귀속(주장-강도 불일치, citation-back PARTIAL 원판정 유효).
- 분류: **MAJOR** (의미 손실/오귀속 잔존 + fixlog 무결성 훼손). §5.3 게이트 조건(BLOCKER=0, MAJOR=0)에 따라 **이 1건 해소 전 게이트 통과 불가**.
- 수정안 (fixlog 자신의 문안): "; similar masking-based reconstruction objectives in some time-series models \cite{fang2024tfmae} are independent developments — our design lineage traces to vision MAE." + fixlog §3 행 14를 r3에서 정정 기록.

### 1.4 REJECTED 5건 — Directive 원문 정합 판정

| 기각 | 사유 (fixlog) | Directive 원문 대조 | 판정 |
|---|---|---|---|
| BLOCKER-BP-05 / MAJOR-PH-02 | "수치 없어 논증 작성 불가→실험 선행" 요구는 R3/A8 placeholder 정책과 충돌 | R3 원문: "실험의 경우에는 '해당 실험이 잘 되었다고' 가정하고 글을 작성하고 … 현재 실험데이터가 부족한건 지적하지말고 … placeholder만 만들고". A8: 수치 자리는 inline placeholder. 리뷰 요구는 Phase 5 운영 규칙과 정면 충돌; 실험 의무는 블루프린트 §0.4에 기등재(Phase 6 전제)로 보존 | ✅ 정합 |
| BLOCKER-R8-01 | 성공 가정 하 정성 단정("confirms graceful degradation") 유지 | 동일 R3 근거. 부분 수용(B-8/ARG-02로 논증의 placeholder-독립화) 이행 확인 (:445–446). 리뷰 제안문("once results are confirmed, this claim will be finalized")이야말로 R3 위반 | ✅ 정합 |
| MINOR-R8-03 | R8-01 파생; registry sync 그룹 기연동 | 동일 근거 + MINOR (waive 가능 범주) | ✅ 정합 |
| m-3 (AdamW fused) | Table A.1=canonical 재현 표(R3), fused는 수치 재현성 관련; 본문 prose 미노출(R27 경계) | 리뷰 자체가 "재현성 목적이면 유지 가능" 인정; A1/A2/A3 무관 | ✅ 정합 |
| R1-03 fix(b) ("This ablation is pending" 공개) | R3 "실험데이터 부족 지적 금지" 위반 | R3 원문 그대로 | ✅ 정합 |

비고: BP-05/R8-01은 BLOCKER의 'waive'가 아니라 **Directive 충돌에 따른 판정 기각**(발견 자체가 운영 규칙과 모순)이며, A1/A2/A3 비관련 — §5.3 waive 금지 조항과 충돌하지 않음.

---

## 2. 고위험 수정 재추적 (코드/정본/card — 14건, 전건 직접 재검증)

| # | 항목 | 1차 소스 재검증 | 판정 |
|---|---|---|---|
| 1 | **PA%K step-1 격자** | `evaluator.py:831` `PA_K_VALUES = list(range(0,101,5))` = per-K 진단 키 전용; **보고 지표 적분 격자** `compute_pa_k_auc` 내부 `k_values = np.arange(0, 101)` + docstring "sweep K=0,1,...,100 and integrate" + `np.trapz(..., k_values)/100.0` (:1271–1282). v2 {0,1,…,100}·101-point = 코드 정확; 271truth r4 §VIII·protocol truth r4 E-2와 3자 일치 | ✅ |
| 2 | **Eq (1) 1/(sF)** | `loss.py:240–242` `patch_discrepancy_sum / (mask_inverse.sum(dim=2) × F + 1e-4)` → per-element 평균, 이후 normal 패치 평균 (:254–255). v2 분모 $\|P_n\|\cdot s\cdot F$ 정합; Eq(2) 1/(\|Pₙ\|·d_model)·§3.6 d_i 규약과 일관 | ✅ |
| 3 | **Eq (4) 전역 평균** | `scoring.py:237–241` `fm_active=False` 하드코딩 + `recon_mean = float(recon.mean()) + eps` — 입력은 evaluator의 **전체 (window,patch) patch-score 배열** (`evaluator.py` `_apply_scoring_formula(recon_patches, …)` — cached 전체). "per-entity 전역 평균, once per entity" 서술 정확; ε=10⁻⁴, σ=r+d̃/c, c=4 (`score_recon_disc_ratio=4.0`) 일치 | ✅ |
| 4 | **Eq C.3 w₊** | `run_base_experiments.py:2578–2586` — patch 단위 anomaly 비율 집계 → `max(_patch_ratio, 0.001)` → `(1−r)/r` (SMD m-1-5 999.0 = 하한 유도값, 271truth §III-3b). focal 본체 `loss.py:337–341` `p_t=exp(−BCE)`, γ=2, masked(valid) 패치 평균, pos_weight 내장 — Eq C.3 전 항 일치, "표준 focal 아님" 구분 문장 유지 | ✅ |
| 5 | **LayerNorm 부재 (linear 경로)** | `model.py:628` `patch_embed = nn.Linear(patch_size×F → d_model)` (bare), 적용 :682; LayerNorm은 **비활성 patch_cnn** 경로 전용 `cnn_flatten_proj` (:610–614). v2 수식·Table A.1 모두 제거 확인. (GRL head의 LayerNorm은 별개·정확 — `model.py:181–186` LayerNorm→512→256→GELU→Dropout(0.1)→1) | ✅ |
| 6 | **추론 결정적 LOO** | `evaluator.py:1716` `_use_complementary=False` → :1735+ LOO 기본 분기, 패치별 명시 마스크 + `model(expanded, masking_ratio=0.0, mask=masks)` (:1805–1818); anomaly-priority는 `model.py:975–977` `if (self.training and force_mask_anomaly and point_labels is not None)` — **training 전용 게이트**. v2 :237/:505 정합 | ✅ |
| 7 | **ARG-01 재논증 사실성** | 경로: (a) 가시 패치만 encoder 입력 + 이상 윈도우의 가시 패치가 문맥 보유 (`model.py:970–973` 주석 "excess remain visible as encoder context"; SYNTHESIS 표A masking/encoder 행 "가시 패치들의 전역 맥락 표현"), (b) 공유 latent를 양 decoder가 읽음 — Student는 `latent_visible.detach()` (`model.py:1124`)로 **gradient만 차단, forward 값 전파**, (c) Student의 anomaly 패턴 기억 우회로 = SYNTHESIS 표A GRL 행 원문("GRL이 없으면 student는 학습 중 anomaly 패턴을 기억해 잘 복원할 수 있고"). v2 :287–289는 §②-4(라벨 개입 3지점)·표A 사실 범위 내; "preferentially masked"(전수 단정 아님) 헤지도 코드와 정합 | ✅ |
| 8 | **§4.4 property (ii)** | `loss.py:294–302` `_pos_count == 0 → grl_cls_loss_tensor=None` (배치 단위 skip); window target은 **masked 영역 내** anomaly (`loss.py:166–168` `point_labels*(1−mask)` → :287 broadcast) — unlabeled anomaly 윈도우 = target 0 (negative). v2 "positive supervision exclusively from labeled windows … treated as negatives, never inject an erroneous positive adversarial signal" 정확 (구판의 "gradient 미수령" 허위 제거 확인). M-4 괄호("coincides with y^w under anomaly-priority masking")도 코드 정합 | ✅ |
| 9 | **notation Table C.2 ↔ 본문 전수** | 표 등재 28행 전수 본문 대조 — 충돌 0: bare $d$/$w$/$r$ 0건, $p$=labeled fraction 전용(τ 분리), $T$=길이 vs upright $\mathrm{T}$ 구분 일러두기 존재, $u$ 각주, $\mathbf{E}$/$a_t$/ρ/α 일관. 잔여(비차단): $c$(Eq 5)·$\beta_{FM}/\beta_{GRL}$(Eq C.4)·$\varepsilon$·$e,e_0,e_1$(C.1)은 inline 정의만 있고 표 미등재 — Phase 6 선택 보강 | ✅ (NOTE 1) |
| 10 | **xue 재서술** | card abstract verbatim: "anchor on representative learning of normal operation with **autoregressive (AR) model** along with **loss components to encourage representations that separate normal versus few positive examples**". v2 :181 "an autoregressive normality model with discriminative loss components that separate normal data from the few labeled anomalies" — variational 오귀속 제거·card 범위 내. huang = "semi-supervised variational autoencoder coupled with an active-learning labeling loop" — card 확인 범위 | ✅ |
| 11 | **zhang 서지 수준화** | v2 :189 "A more compact formulation is self-distillation \cite{zhang2022selfdistill}, which performs the distillation within a single architecture." = citation-back §5-U5의 자기 제안 축약판 그대로; "compression 목적"·"teacher+내부 student heads" 내용 귀속 제거 확인. 잔여(비차단): "within a single architecture"는 엄밀히 제목 수준을 약간 초과하나 후보 abstract(§4-1 "knowledge transfer in the same model")와 정합 — 2인 검증 card 등재 시 완전 해소 (fixlog §7 비차단 항목과 일치) | ✅ (NOTE 2) |
| 12 | **D-008 스코핑 (광의 최초성 잔존 0)** | "first" 전수 grep: :149 "the first architecture combining masked-reconstruction self-distillation with gradient reversal … in a contaminated semi-supervised multivariate TSAD setting" (to our knowledge, 스코핑판) / :183 "the first end-to-end multivariate TSAD model that integrates labeled anomalies adversarially — through gradient reversal — into the gradient of a masked-reconstruction self-distillation objective" (동일 골격 정렬). 광의판("self-supervised representation learning objective") 잔존 0 | ✅ |
| 13 | **DeepMIL/WETAS/TreeMIL §2.2 추가분** | card 대조 — DeepMIL: MIL ranking(abstract 4발췌 확인) / WETAS: "instance-level (or weak) anomaly labels … learns discriminative features" / TreeMIL: "only series-level labels are provided … MIL approach". v2 :181 "trains models to classify or rank windows from coarse segment-level annotations … the label is the sole learning signal, with no self-supervised reconstruction pretext" — 3편 공통 사실 범위 내 (재구성 pretext 부재 = card 차별화 노트 정합) | ✅ |
| 14 | **NRdetector·SDMAE·DACAD·bekker 사실 4건** | 7:3 (card §5.1 verbatim) ✓ / WETAS-유래 backbone·multi-stage (card 주의 + dossier D1/D3/D5) ✓ / SDMAE branch-off (card 발췌 "branches out from the teacher after the first transformer block") ✓ / DACAD transfer-setting (card abstract "supervised contrastive loss for the source domain") ✓ / bekker two-step (card §5 verbatim "1) identifying reliable negative examples, 2) learning…") ✓ / elkan SCAR 상수배 보정 ✓ | ✅ |

추가 정합 (보조): λ_rev sigmoid `trainer.py:1201–1211` (p=clip((e−250+1)/250), 2/(1+e^{−10p})−1, 첫 student epoch ≈0.02) = Eq C.1 ✓; λ_GRL `trainer.py:751–765` grad-ratio clamp[0,10] × 0.2, prev-epoch 적용 = Eq C.4 ✓; GRL training-only `model.py:1150–1154` ✓; stop-gradient `model.py:1124` ✓; PA F1 oracle = pre-PA F1-최적 threshold 후 PA 적용 (`evaluator.py:929–931, 951–955`) ✓; SMAP/MSL train 라벨 zeros (`loaders.py:2602–2604`) ✓; baseline batch 모델별 이질 (`baseline_common.py` MODEL_CONFIGS 32–512, weak 4종=32) ✓; test stride 49 = `utils/experiment.py` `W//10−1` ✓ (본문 무주장, Table A.1만).

---

## 3. A8 수치 재스윕 (v2 산문 digit 전수)

방법: frontmatter·HTML 주석 제거 후 정규식 digit-token 전수 추출(약 140 고유 토큰) → 분류.

**① placeholder**: PH:NUM-001..031 (31/31 고유, 중복 0), PH:TXT-001/002 (4 occurrences), FIG-1..4+B1, body TAB-1/2/3 (+appendix TAB 8종, ALG-C1). **[TAB-4] body 마커 0** (frontmatter 노트·SURGEON 주석 내에만 잔존 — 정상). **PH 주석 없는 고아 `[X.XX]`/`[N]`/`[URL]`/`[GPU model]`/`[gradually…]` 토큰 0건** — registry v2-r3 §6 스캔과 정합.

**② 프로토콜 상수 (전수 정본 확인)**: 데이터 규모 12종(719,959 … 36,775), AR 14종(1.63/0.52/0.76/6.20/0.70/1.70/3.82/3.87/19.05/3.68/30.63/24.54/16.72/4.16), 차원(45=51−6 {P202,P401,P404,P502,P601,P603}, 123=127−4, 25, 29–36, 55=1+54, 25=1+24), region 22(2,869/38,769/35,900/83.75%/15.96%/13 잔여), safe-cut 표(2,191/1,095/1,261/+166/7.58%, 2,277/1,138/1,099/−39/1.71%, 1,827/913/921/+8/0.44%, 합계 252, 4/81, ten timesteps), config(500/250/10/0.15/8/42/21/49/4/3/2/8heads/2048/512/0.15/1024/bf16/seed42/lr 10⁻³·10⁻⁴/wd 10⁻³/β(0.9,0.99)/γ=2/β_GRL 0.2/β_FM 1.0/clip[0,10]/10³/ε 10⁻⁴/c=4/≈0.02→≈1/N=50), 평가(101점/K 0–100/VUS window 100/5지표), 구성(6 families/113/114/26/22/9/6/7/4/28/54/27/NRdetector win 100/random 5-run), sweep p∈{1.0,0.75,0.5,0.25,0.1}, §5 "approximately 50×"(registry §5 해소 항목, N=50 유도 + NUM-031 sync 조건), "7:3"(문헌 사실 — card verbatim S), 연도 토큰(서지) — **전건 EXPERIMENT_PROTOCOL_TRUTH r4 / 271_CONFIG_TRUTH r4 / card에서 출처 확인**.

**③ 미등재 결과 수치**: **0건 (BLOCKER 0)**. 실험 결과형 수치(지표 값·개선폭·승패 수·p-value)의 산문 유입 없음 — §4.2/§4.3/§4.4/B.1–B.5 결과 자리는 전부 placeholder.

부기: Highlights 5 bullet 실측 ≤125자 (123/124/120/121/≤125 — bullet 5는 NUM-003 해소 후 Phase 7 재검 플래그 유지, fixlog와 일치).

---

## 4. 인용 key 재검증 + R9 스윕

**인용**: v2 전 `\cite`/`\citet` 추출 — 고유 key **48**, refs.bib(49) 대조 **무효 0**. 미인용 잔여 = `jacob2021exathlon` 1건(의도된 미사용, R33) — fixlog §4 결론과 일치. 격리 3건 준수: zhang(서지+최소구조 — §2 NOTE 2), xu2018kpivae(:545 kim 병기로 내용 지지 분리), ruff(:179 제목 수준 "deep semi-supervised anomaly detection objectives").
MINOR: References 주석 :775 "**44 cited**" — 실측 48 (fix 라운드에서 darban/xiong/xu2023rosas/wang2022hscl 편입 후 stale). Phase 6/7에서 갱신 필요.

**R9 (SDMAE 전수)**: 언급 5곳 전수 — Abstract/Highlights/§1 0건; §2.3 :189 (인용된 선행 서술, 비교 없음) / §2.3 :191 (중립 적응문 1개) / 각주 [^sd-fn] :193 (용어 계보 + 구조차 1 + GRL 부재 + §3.5 포인터 — 블루프린트 결정 ⑤·BP-02 fix가 지정한 각주 전속 배치) / §3.4 :252 (계보 중립) / §3.5 :281 (작동 계층 1문장). **본문 차이-나열 패턴 0건** — 3-way 분리(본문 중립 / 각주 구조 / §3.5 계층) 완성.

---

## 5. Directive 전수 판정 (Phase 5 매핑 — §9.4 정본 기준)

| ID | 판정 | v2 충족 근거 (위치 + 핵심 문자열) |
|---|---|---|
| **T4** (보강 사이클) | ✅ | R36 15건 전부 기존 49 검증 key로 해소 — 신규 reference 수요 0 (gap 리뷰 §0 "(b) 신규 수요 0건"); 후보 발췌 4건 card 미추가·재서술 해소 (fixlog §7, 의무 미니 2인 검증 대상 0); 48 key 전수 유효 (§4) |
| **T5** | ✅ | MANUSCRIPT_v2.md 전체 — 영어 완성 본문 + figure/table 삽입 위치·완성 캡션 (registry §1–2) + 표절 검사 루프 통과 (F1–F3/SC-06 반영 확인) |
| **R1** | ✅ | §2.1 4-family(:173) / §2.2 PU→AD 적용→weak-sup→semi-sup 2편→transfer→NRdetector 단계 구조(:179–183) / contributions 4-bullet 상호배제(:153–159) / §4.1–4.5 실험 구조; R1-01 차별화 문장 반영 |
| **R3** | ✅ | placeholder 49종(NUM 31·TXT 2·FIG 5·TAB 11·ALG 1) + registry 완전 캡션; "실험이 잘 되었다고 가정" 서술 유지(BP-05/R8-01 기각으로 보존 — §1.4); 실험데이터 부족 지적 문장 0 |
| **R4** (예방) | ✅ | 금지 패턴 grep 0건 (delve/showcase/pivotal/realm/landscape/seamlessly/meticulously/holistic/paving/unlock/harness/In conclusion,/important to note/comprehensive — 산문 0; plagiarism 리뷰 AI-phrasing 표 CLEAR + 본 감사 v2 재스윕) |
| **R5** | ✅ | Eq (1)–(6)·C.1–C.5 코드 정합(§2 재추적 1–4·보조) + 기호 충돌 6군 해소(M-5) + 미정의 5건 해소(M-6) + Table C.2(:738–765) |
| **R8** | ✅ | :148 "we propose **CSMAD** …"; :149 D-008 스코핑 최초성; contributions 1–4(:153–159); §2.2 :183 동일 골격 최초성 정렬 |
| **R9** | ✅ | §4 스윕 — 차이 나열 0, 각주 전속, "자연스럽게 언급하고 넘어가는" §2.3 :191 단문 |
| **R10** | ✅ | :212 (운영 로그→다채널 상관→학습 과제) / :230 "encodes cross-channel correlations directly in the token" / :251 (정상 상관 구조 모방 실패) / :213 (필터링은 co-occurrence 구조 폐기) |
| **R11** | ✅ | :208 "**contaminated semi-supervised** setting … large majority of unlabeled windows and a small fraction carrying anomaly labels"; §1 :137–138 (unsupervised의 labeled 미활용 핵심 동기); 3단 구조 :208–210 |
| **R12** | ✅ | :393–394 "labeled anomaly regions are excised … a labeled anomaly can be used only negatively — as a contaminating sample to remove; Q3 grants each unsupervised method this most favorable use of the labels" |
| **R13** | ✅ | :341–348 (원본 train 구조적 라벨 부재 → temporal midpoint re-split → 앞 50% train 편입 → 뒤 50% 미래 평가·lookahead 없음 → "The halving rule is uniform" → NRdetector 선례) |
| **R15** | ✅ | 제목 :73 (D-007 확정) + CSMAD 명명 :148; 불필요 신규 축약어 0 (MTSAD/MAE/PU/GRL 등 표준 용어만) |
| **R16** | ✅ | :348 re-split 선례 / :183 실험·구조 차이 축 / :389 weak-sup Q1-only / :447 label-noise sweep과의 구분 (NRdetector 논리 참조 구조) |
| **R17** | ✅ | 미사용 component 본문 grep 0 (dynamic margin/hinge/softplus/SCAD/discriminator/RevIN/EMA/WDGRL/balanced sampling/annealing/clamp/275K/patch_cnn/complementary masking/memory bank — method-truth §2.2 전수 + 본 감사 재확인); Table A.1 = 271truth r4 §VIII 전사 |
| **R19** | ✅ | :388 baseline 26종 = 실험 섹션 인용 클러스터; related work는 핵심 계승(MAE/SDMAE/zhang)·직접 비교(NRdetector)·보조 차별화(xue/huang/darban)만 |
| **R20** | ✅ | :179 PU 일반 목표(3계열) 서술 + :181 "deep representation learning informed by label signals **remains rare**" + :183 NRdetector 차이점 위주(파이프라인 vs 기울기) |
| **R21** | ✅ | :193 "The self-distillation terminology follows Zhang et al. … and Ristea et al." (선례 방어논리, 각주) |
| **R22** | ✅ | :187 "Our patch-based masking draws directly from this paradigm [vision MAE] … are independent developments — our design lineage traces to vision MAE" (단, TFMAE 묘사 어휘 1건 = §1.3 MAJOR — R22 논지 자체는 충족) |
| **R23** | ✅ | 본문 §4.1.2 핵심 상수만(:360–376); 전체 hyperparameter는 Table A.1·TAB-A3 위임; "Full hyperparameters … are in Appendix" :362 |
| **R24** | ✅(주의 1) | 내부 변수명·지표 키 산문 grep 0 (force_mask_anomaly/pak_auc/normalonly/_ar 등); 정식 명칭 사용(PA%K-AUC F1, VUS-PR/ROC, Affiliation F1). **주의: 내부 조건 코드 "Q1/Q3" 잔존** (:389–395, §B.1) — 매 사용처 정의 동반("Q3 (normal-only)", "Q1 (full contaminated training)") + 블루프린트(게이트 통과본)가 채택한 표기이므로 즉시 위반 아님; Phase 6 terminology-normalizer에 개명/유지 판단 회부 권고 |
| **R25** | ✅ | :92 "Code will be made available at [URL] … upon acceptance" + :480 + :528 (TXT-002 3 occurrences 동일) — "자연스러우면 넣는다" 판단 이행 |
| **R27** | ✅ | 구현 디테일 §A.1/§C.2 위임(D-009); 코드 내부 상태 노출 본문 0 ("implemented but not used" 삭제 — B-9; fused는 Table A.1 한정 — m-3 기각 사유) |
| **R28** | ✅ | :352–355 "a single attack event (region 22) accounts for 83.75% of test anomaly mass … same model, same scores, only the evaluation mask differs, identically for all baselines" + §A.4 유도 전체 |
| **R29** | ✅ | :380–384 5지표 + "three orthogonal perspectives … with distinct failure modes" + PA F1 "(oracle) … never used for ranking: even a random score can reach state-of-the-art levels under it" |
| **R30** | ✅ | :376 AR threshold 정의 + "(1) threshold-free 지표 병행(VUS, PA%K-AUC families are unaffected)" + "α … is never used in training" + 전 모델 동일 적용 + 선례 완화 인용(M-1) |
| **R31** | ✅ | :393–395 (Q3=최선의 라벨 활용 논리 + 학습량 비대칭 정량 인정 "reduces baseline training volume by the train anomaly ratio" + protocol-effect 블록·§B.1 보완) + :389 weak-sup 4종 희소성 |
| **R32** | ✅ | §4.4 전체 — sweep 설계(p 5단계, region 단위) + "Why graceful degradation is expected" 3-property 논리(코드 사실 기반 — §2 재추적 8) + ARG-02 공변 보강 |
| **R33** | ✅ | grep -i "simulation\|exathlon" 산문 0건; 6 families = SWaT/WaDi/PSM/SMD/SMAP/MSL (:334); jacob2021exathlon 미인용 |
| **R34** | ✅ | grep -i "gaussian\|smooth" 산문 0건 ('ensemble smoothing'도 m-4로 제거 확인) |
| **R35** | ✅ | D-009(수식 12→6, 정의·유도 appendix 위임) + D-010(TAB-4 흡수·ablation 3행 강등·enumeration 위임) — 지엽 서술 본문 격리 |
| **R36** | ✅ | G-01..G-15 전수 처리·실반영 확인 (§1.2) — 14 FIXED + 1 NO-CHANGE(리뷰 권고 그대로) |

**미충족 Directive: 0** (단 §1.3 MAJOR는 T4/A1 계열 인용-주장 정합의 마감 결함으로 게이트 차단).

---

## 6. 종합 판정

### CONDITIONAL FAIL → 1건 수정 후 PASS 전환 가능

| 등급 | 건수 | 내용 |
|---|---|---|
| **MAJOR** | **1** | **F-1. citation-back PARTIAL-14 미반영** — v2 :187 "similar patch-and-mask operations … \cite{fang2024tfmae}" 가 draft와 동일 잔존 (fixlog §3 행 14는 FIXED로 허위 기록). TFMAE card abstract = window-temporal + amplitude-frequency masking — patch 단위 오귀속. **조치**: fixlog 자신의 문안("similar masking-based reconstruction objectives in some time-series models")으로 교체 + fixlog r3 정정 기록 + 교체 문장 plagiarism/문체 spot 1회 |
| MINOR | 2 | F-2. References 주석 "44 cited" → 48 (stale, :775). F-3. fixlog 톱라인 94 vs 섹션 합 98 — 교차 중복 dedup 미명시 (기록 정밀성) |
| NOTE | 3 | N-1. Table C.2 미등재 기호 c/β/ε/e₀,e₁ (inline 정의 존재 — Phase 6 선택 보강). N-2. zhang "within a single architecture" — 리뷰 자기 제안 축약판 그대로이나 후보 abstract 2인 검증 card 등재 시 완전 해소 (fixlog §7 비차단 기등재). N-3. 내부 조건 코드 Q1/Q3 — R24-인접, Phase 6 terminology-normalizer 회부 권고 |

- 마감 검증: 처리 기록 99/99 등재(누락 0), 실반영 98/99 (1건 실패 = F-1), REJECTED 5건 사유 전건 Directive 원문 정합.
- 고위험 재추적 14건 + 보조 9건: **전건 PASS** (코드·정본·card 1차 소스 일치).
- A8: 미등재 결과 수치 0, 고아 placeholder 0, registry 정합.
- 인용: 48 key 무효 0, 격리 준수; R9 차이-나열 0.
- Directive 31개(Phase 5 매핑): 충족 근거 전원 확보 — COVERAGE_MATRIX 갱신 가능 (F-1 해소 후 DONE 전이 권고).

### 재감사 조건
F-1 적용(1문장 교체) + p5_fixlog r3 1행 정정 + 교체문 spot 검사 → 본 게이트 PASS 전환 (전면 재감사 불요; 변경 범위가 단일 문장이므로 diff 확인으로 충분).
