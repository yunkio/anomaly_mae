---
phase: 1
agent: adversarial-reviewer-B
directives: [R2, R26]
last_modified: 2026-06-10
targets:
  - paper/01_research_understanding/NOTION_DIGEST.md
  - paper/01_research_understanding/CONFERENCE_PDF_DIGEST.md
sources_verified:
  - "Notion Page 0 MAE (decoded 75,820 chars — digest 표기와 일치, 전문 정독)"
  - "Notion Page B Baseline Comparison (decoded 108,461 chars — digest 표기와 일치, 전문 정독)"
  - "paper/윤기오_대한산업공학회_2026_춘계.pdf (34p 전 페이지 정독)"
---

# Phase 1 Adversarial Review (r1) — NOTION_DIGEST + CONFERENCE_PDF_DIGEST

## 판정 요약

| 문서 | 판정 | BLOCKER | MAJOR | MINOR |
|------|------|---------|-------|-------|
| NOTION_DIGEST.md | **REVISE (BLOCKER 2건 수정 필수)** | 2 | 4 | 6 |
| CONFERENCE_PDF_DIGEST.md | **REVISE (BLOCKER 1건 수정 필수)** | 1 | 1 | 5 |

총평: 두 digest 모두 **수치·인용 전사 정확도는 높은 수준**이다 — NOTION_DIGEST의 Q3 결과표 17행 × 8열 전수 일치, [B1]–[B18]/[D1]–[D8] 서지 전수 일치, PDF_DIGEST의 p22 메인표·p29–33 Avg Rank·p25 백분율·p27 결론 7개 bullet 전부 원문 일치를 확인했다. R2 구분 장치([Notion의 주장]/[검증된 사실 후보], Phase 3 판단 사안, R5 notation 비계승, 수치 동결 금지)도 골격은 일관 적용되어 있다. 그러나 **R26 truth 등급 표에서의 전사 불일치 1건(WaDi A2 features), 산식 전사 누락 1건(1,804 cells), 모델 개수 오기 1건(25종→26종)** 이 발견되어 BLOCKER로 분류한다. 셋 다 국소 수정으로 해결 가능하다.

---

## A. NOTION_DIGEST.md

### BLOCKER

**[NB-1] R26 truth 데이터셋 표의 WaDi A2 Features = 127 — 출처(Page B)는 123**
- digest II-3 "[truth 등급 — R26] Dataset References" 표: `WaDi A2 | 127 | 1 | Ahmed et al. ...`
- **Page B §2.1 표의 WaDi A2 Features는 123** (Page B 전문에 "127"은 단 한 번도 등장하지 않음 — grep 확인). 발표 PDF p19도 WaDi A2 = 123 dim.
- 127은 **Page 0** (MAE 학습 파이프라인: num_features 표·d_model 매핑·§5.2.1)에서 온 값. 즉 두 원천이 **상호 모순**(Page 0=127 vs Page B=123)인데, digest는 Page B 기반 truth 표에 Page 0 값을 무표기로 기입했다. R26 truth 등급 표 내부의 부정확 + 원천 간 모순 은폐 → BLOCKER.
- 수정 방향: II-3 표는 Page B 값(123)으로 환원하고, Page 0(127)과의 모순을 "코드/로더 대조 필요" 항목(digest §IV)에 명시 추가할 것. (두 파이프라인의 전처리 차이일 가능성 — `mae_anomaly/datasets/loaders.py` vs `comparison/unified_loader` 대조로만 해소 가능.)

**[NB-2] II-1 cells 산식 전사 누락 — "41 runs × 2 conditions = 1,804 cells"**
- 원문(Page B Total 문장): "41 runs per condition × 2 conditions **× 22 active models** = 1,804 (model, dataset, condition) cells".
- digest는 "× 22 active models" 인자를 누락하여 산식 자체가 거짓이 됨(41×2=82). 최종 수치 1,804는 맞으나 [검증된 사실 후보] 블록 내 산식 전사 오류 → BLOCKER(수정 1줄).
- 참고: Page B 자체에도 내부 불일치가 있다 — Snapshot callout은 "9 datasets = **39** dataset runs (Pattern A)"라 하고 Total 문장은 "39 base **+ 2 SMAP/MSL = 41**"이라 한다(39에 SMAP/MSL 포함 여부 모순). digest가 41 해석을 택한 것은 합리적이나, 원천 내부 모순임을 표기하면 더 안전.

### MAJOR

**[NM-1] R26 truth 등급의 과적용 — I-10 (Page 0 references 12건 전체)**
- R26의 truth 범위는 "**비교 대상 모델** reference + **데이터셋** reference"다. I-10의 [4](Anomaly Transformer, baseline), [6]–[10], [12](데이터셋 출처)는 범위 내이지만, **[1] He(MAE), [2] Ganin(GRL), [3] Kim(PA%K), [5] Esser(VQGAN), [11] Lin(Focal)은 방법론 인용**으로 R26 적용 대상이 아니다.
- digest 헤더의 R2 경고문은 범위를 올바르게 한정("baseline 모델 reference와 데이터셋 reference는 R26에 따라 truth 등급")했으나, I-10 섹션 라벨이 "[truth 등급 — …]"로 12건 전체를 포괄해 헤더의 한정과 모순. "Notion이 검증 완료라 명시"라는 귀속 표기와 Phase 4 재확인 조항이 완충하지만, 등급 라벨 자체는 분리해야 한다(방법론 인용 5건은 [Notion의 주장 — 검증 완료 주장] 등급으로 강등).

**[NM-2] II-6 Q3 partial 수치의 유효성 한정 누락 — per-entity STALE만 표기, 2026-05-25 paper-faithful 재실행 무효화 미표기**
- digest는 SMD/Exathlon column의 per-entity STALE만 경고한다. 그러나 Page B는 추가로:
  - §3 status: 결과는 **2026-05-22 시점** 채움.
  - 2026-05-25 변경 이력: QuoVadis 9종 line-by-line 수정(pca_error AGGREGATION FIX, mlp/mlpmixer/transformer REIMPL, neural_base predict Pass 2 fix 등) + non-self_norm SOTA clip 제거 → "**영향 모델 15종**, 영향 entries 삭제 후 재실험, 결과 row는 재실험 완료 후 swap-in 예정". 5번 실험 폐기 → 6번 재실행.
  - 2026-06-04 faithfulness pass v2: 12개 모델 추가 수정(tranad/gdn/timesnet/memto/npsr/random 등).
- 즉 **표의 simple/neural/legacy 행 다수가 audit 이전 코드의 수치**로, "swap-in 대기" 상태다. 이 한정 없이 "[검증된 사실 후보 — 수치]"로 제시하면 Phase 3가 이 수치를 과신할 위험. SMD/Exathlon 외 column에도 "2026-05-25/06-04 fidelity 수정 이전 산출, clean re-run 후 swap-in 예정"을 명기할 것.

**[NM-3] 누락 — 비교 실험 조건 정의 (Page B §1.2 + 2026-06-04 faithfulness pass)**
- 과업 기준(비교 실험 페이지의 실험 조건 정의)에서 digest II에 전혀 없음:
  - 모델별 하이퍼파라미터 preset 전체(§1.2.1–1.2.5: win_size/batch/lr 등; 예: anomaly_transformer win=100·d_model=512, omnianomaly seq=100, memto train_stride=100·2-phase, catch win=192, weak 4종 preset + [fixed]/[normalization]/[runtime-estimated]/[impl-invented] provenance 태그 체계).
  - weak label 정의: `max(point label over window)` (train split 한정, leak-free).
  - 2026-06-04 faithfulness pass 핵심 변경(tranad LeakyReLU·lr=1e-4 config-layer, gdn batch=32(run.sh repro), timesnet SMAP-script HP 채택 근거, memto FRESH re-init, random seed=None·5-run mean±std, WETAS-family fit-on-test vs deepmil 유일 leak-free 정규화 구분).
  - 2026-06-02 boundary-safe TEST windowing (21개 windowing baseline, multi-entity 재실행 사유).
- 논문 §Experiments의 fair-comparison 서술과 reproducibility에 직결되는 내용이므로 II에 별도 절(예: II-2b 실험 조건 상세) 추가 권고.

**[NM-4] 누락 — 방법론 페이지 학습 절차 상세 (Page 0 §3)**
- 과업 기준(방법론 페이지의 학습 절차 상세)에서 빠진 것:
  - Teacher-only warmup **메커니즘**: epoch<250에서 student forward는 수행하되 `teacher_only=True` flag로 disc/FM/GRL 손실 비활성(§3.5) — 논문의 학습 절차 서술에 필요.
  - Force-mask-anomaly **priority 공식** (`priority_p = 1[anomaly]·1000 + η_p`, TopK_8, budget 초과 시 random subset) — C1/C4 서술의 구체화 소재.
  - 디코더 구조 결정: `use_transformer_encoder_decoder=True` — **두 디코더 모두 self-attention only(TransformerEncoder), cross-attention 없음** — 아키텍처 그림/서술에 직접 영향.
  - 마스킹 budget이 "약 8개"가 아니라 **round(50×0.15)=8개 고정**(batch 균일, §3.3.2) — digest는 callout의 "약 8개"만 인용.
  - AMP bf16(2026-05-27 fp16→bf16 사유 포함), eval_interval=5, random_seed=42, trainer config validation 5종, GRL adaptive λ의 anchor(w=student decoder 마지막 weight).

### MINOR

**[Nm-1]** Page 0 내부 모순 미표기: §2.4/§4.3.3 스코어링 서술이 한쪽은 "recon:disc=4:1 + FM 제외", 다른 쪽(§3.6 Scoring, §4.3.3 anomaly_score_mode 역할란)은 "disc/FM 정규화 후 **1:1 결합**"으로 잔존(stale text). digest는 4:1만 채택 — 채택은 옳으나 원천 내 모순 자체를 §IV 의심 지점에 추가해야 함.

**[Nm-2]** "weakly-supervised **4종** 50 epoch" — Page B Snapshot 원문은 "weakly-supervised **5종** 50 epoch (2026-06-06 통일)". 페이지 제목/§1.1/§6.4는 4종이고 `nrdetector_full` 변형이 별도로 존재하므로 4종 채택은 합리적이나, 원문이 5종으로 표기된 사실(또는 nrdetector_full의 존재)을 미표기.

**[Nm-3]** R2 라벨 비일관: "Q2/Q4(zscore) 폐기"가 II-1에서는 [검증된 사실 후보], II-4에서는 [Notion의 주장]으로 이중 분류됨. 하나로 통일할 것(사실 후보가 적절).

**[Nm-4]** II-4 per-entity 정규화 예외 목록 축약: 원문 예외 (a)는 "PSM/SWaT/WaDi/**simulation**/**`*_simple`**/**단일 machine**" — digest는 "PSM/SWaT/WaDi"만 기재.

**[Nm-5]** TEP 처리 비대칭: [D6]/[D7](TEP refs)은 수록했으나 II-3 데이터셋 표에 TEP 행(참고용, 비교 미사용)이 없어 D6/D7이 어떤 데이터셋의 reference인지 digest 단독으로는 불명. TEP가 "보유·검증 완료, 비교 실험 미사용(참고)" 상태라는 한 줄 필요. (TEP License CC0도 누락.)

**[Nm-6]** [B4b](usad official-affiliated PyTorch impl 코드 인용) / [B5b](TranAD-저자 DAGMM reimpl 코드 인용) 누락 — R26 전수 대조 관점에서 구현 출처 reference 2건이 빠짐. 특히 [B5b]는 digest §IV-6(dagmm provenance)과 직결. 아울러 Page B는 dagmm에 대해 "scoreboard에서 **dagmm_tranad로 relabel** + energy-DAGMM과 직접 비교 금지"를 이미 **결정**해 두었는데, digest IV-6은 이를 "검토 필요" 수준의 열린 질문으로 약화 서술함.

### 확인 완료 (전수/표본 대조 — 이상 없음)
- Q3 PA%K AUC F1 표 **17행 전수**(MAE 271/B2 + 15 legacy) 8열 모두 원문 일치. RankAvg(1.00/2.00, prc 1.00, f1_t 1.50) 일치. B2 = `gaussian_filter1d(s, sigma=10, mode='reflect')` post-hoc, re-train 불필요 — 원문 일치.
- 모델 목록 22+4 전수 일치(키/논문명/venue/연도/repo/License). [B1]–[B18] 서지 필드 전수 일치. [D1]–[D8] 전수 일치. SMD 32–38/708,405/708,420/4.16%/MIT, PSM 132,481/87,841/27.76%/CC BY 4.0, SMAP/MSL Pattern A 통계·P-2 UNION·safe-cut 4채널, Exathlon 93 traces/19 FScustom/CC BY-NC-SA — 전부 원문 일치.
- Page 0: exp271 고정값, d_model 매핑 표 7행, 학습 단계 표, GRL λ 스케줄 표(0.762/0.965/0.995), 총손실/Recon/OD/FM/GRL 수식, pos_weight≈7.29, 50-forward 추론, 4:1 스코어링, I-8 파라미터 표 36행, I-9 한계 4건, [1]–[12] — 전부 원문 일치.
- REQUEST:/FEEDBACK: 블록 부재 확인(양 페이지 grep 0건) — digest §V 일치.

---

## B. CONFERENCE_PDF_DIGEST.md

### BLOCKER

**[PB-1] "Baseline 25종" — 실제 p20 표는 비교모델 26종(+Ours=27)**
- p20 표 실측: Simple 5(Random/Sensor-Range/PCA/L2-Norm/kNN) + Neural 3(MLP/MLP-Mixer/Transformer) + SOTA 14(GCN-LSTM/Anomaly Trans./TranAD/USAD/DAGMM/GDN/OmniAnomaly/TF-MAE/NPSR/TimesNet/DCdetector/MEMTO/ModernTCN/CATCH) + Weakly-Supervised 4(DeepMIL/WETAS/TreeMIL/NRDetector) = **26개 비교모델** + Ours.
- 교차 증거: (i) p29–33 appendix의 rank 첨자가 (27)까지 존재(27개 모델 순위), (ii) Notion Page B의 22 active + 4 weak = 26과 정합.
- digest는 ①(페이지 맵 p20)과 ⑤(Baseline 절)에서 두 번 "25종/25개 비교모델"로 오기. 개별 모델 나열 자체는 26개를 모두 정확히 적어놓고 카운트만 틀림(p34 references가 [1]–[25]인 것과 혼동 추정). → 전사(계수) 오류, BLOCKER. "26종(+Ours)"으로 수정.

### MAJOR

**[PM-1] 원천 간 모순 미표기 — Patchify "1D-CNN"(PDF p12/15/16) vs Notion exp271 `patchify_mode='linear'`(CNN 미사용)**
- PDF 아키텍처 그림은 일관되게 "Patchify (1D-CNN)"를 명시하고, digest ④도 이를 충실 전사했다. 그러나 Notion Page 0의 본 baseline(exp271)은 "**Linear embedding, CNN 없음**"(patchify_mode='linear', Set C)이다. 논문의 아키텍처 서술·그림에 직결되는 모순(발표 시점 구성 vs 현재 baseline 구성, 또는 발표 그림 오류)인데, digest ⑧ REQUEST는 ℒ_grl/α-scaling/warmup/window 크기만 코드 확정을 요청하고 **patchify mode 모순은 미적시**. REQUEST에 "patchify 1D-CNN(발표) vs linear(Notion exp271) — 발표 당시 config와 현 baseline 중 무엇을 논문 아키텍처로 쓸지 코드·실험 ID 기준 확정"을 추가해야 함.

### MINOR

**[Pm-1]** p6 인용 절단: 원문은 "…semi-supervised learning 접근이 가장 현실적이고 강력한 해결책**이 될 수 있음**"(완화 어미 포함). digest ②는 어미를 절단해 인용했고, ⑦-6에서 이를 "발표체 단정"의 예로 들었다 — 원문은 이미 hedge되어 있으므로 비판 강도가 원문보다 과함. 인용을 원문대로 복원하고 ⑦-6 사례를 조정할 것.

**[Pm-2]** "Ristea et al. 계열" (③ Background 2) — PDF 어디에도 저자명이 없음(제목·CVPR 2024만 표기). 외부 지식 주입이며 사실로는 정확하나, "(저자명은 PDF에 없음 — 외부 확인, Phase 4 재검증 대상)" 표기 필요. 무표기 외부 사실 주입은 창작으로 오인될 수 있는 패턴.

**[Pm-3]** ④ 문제 정식화의 괄호 주석 "(학습에 쓸 수 있는 label은 𝒳ᴬ_lab — known anomaly)" — p11 정식화에는 **𝒳ᴺ_lab(labeled normal)도 존재**하며, 슬라이드는 라벨 사용 범위를 명시하지 않음(손실 수식이 𝒳ᴬ_lab만 참조한다는 점에서의 추론). 추론임을 표기하거나 "손실에는 𝒳ᴬ_lab만 등장"으로 바꿀 것. ② 말미의 "labeled가 anomaly(positive) 쪽에만 소수 존재" 요약도 동일 이슈.

**[Pm-4]** p19 표 미전사 column: #Training/#Testing, #Anomaly Regions, **Train AR(%)** (SWaT 1.63 / WaDi A1 0.52 / A2 0.76 / PSM 6.20). Train AR은 contaminated semi-supervised 프로토콜의 핵심 통계(훈련 오염도)로, "수치 동결 금지" 단서를 달더라도 구조 이해상 기재 가치가 큼.

**[Pm-5]** p21 metric별 reference 미기재: F1_PA←Xu et al. WWW 2018 [4], F1_PA%K/PRC_PA%K←Kim AAAI 2022 [5], VUS-PR/ROC←Paparrizos PVLDB 2022 [6], Aff-F1←Huet KDD 2022 [7]. Phase 4 서지 작업과 평가절 작성에 필요 — 한 줄 표로 추가 권고. (p34 [1]–[25] 전체를 itemize하지 않은 것은 digest 범위상 수용 가능하나, 최소 metric refs는 본문 서술에 직결.)

### 확인 완료 (표본 대조 — 이상 없음)
- 제목/저자/소속/이메일, 목차, p3–7 인용문 전부, p8 "15505회 인용", p10 SD-MAE 요약 인용 2건, p11 수식 3종(𝒳 4분할, ℒ_rec, ℒ_disc — ℳ/ℳᴺ 정의 포함) 및 "(ℒ_grl은 classifier loss)" 주석, p14 s_p 식·auto-scaled α·μ_recon/μ_disc 정의·앙상블 논거, p15–18 정당화 인용 전부, p18 GRL 4 bullet, p19 통계(SWaT 944,919·19.05%·excl22 3.68%, WaDi A1 1,382,402·3.82%, **WaDi A2 123 dim**·3.87%, PSM 220,322·30.63%)·train/test 재구성 프로토콜·excl22 근거 원문, p21 7지표·anomaly-ratio threshold 원문, p22 Ours 수치 10개·Avg Rank 1.25·MLP-Mixer 3.81·Worst 3·Std 0.56·해석 4 bullet, p23 "recon:disc = 4:1" 축 라벨, p24 warm-up 250·student joins, p25 53.9%/48.3%, p27 결론 7 bullet 전문, p29–33 Avg Rank(1.20/1.00/1.00/1.00/1.80, PSM MLP 1.80 동률·Ours 세부 순위) — 전부 원문 일치.
- R5 notation 비계승·수치 동결 금지·Phase 3 판단 사안 표기는 해당 절마다 일관 적용 확인.

---

## C. 종합 권고 (수정 우선순위)

1. **즉시 수정(BLOCKER)**: NB-1(WaDi A2 123/127 모순 명시 + truth 표 환원), NB-2(×22 인자 복원), PB-1(25→26종).
2. **Phase 3 진입 전 수정(MAJOR)**: NM-1(truth 등급 라벨 분리), NM-2(Q3 수치 유효성 한정 강화), NM-3/NM-4(실험 조건·학습 절차 절 보강), PM-1(patchify 모순 REQUEST 추가).
3. MINOR는 일괄 patch 1회로 처리 가능.
4. 후속 검증 의뢰: WaDi A2 feature 수(123 vs 127)와 patchify mode(linear vs 1D-CNN)는 **코드가 유일한 심판**이다 — code-digest/271truth 라인에 두 항목의 코드 기준 확정을 REQUEST로 전달할 것.
