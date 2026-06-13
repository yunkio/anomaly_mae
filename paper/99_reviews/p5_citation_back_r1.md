---
phase: 5
agent: citation-back-auditor
directives: [T4, A1]
last_modified: 2026-06-11
scope: MANUSCRIPT_v2_draft.md 전체 (본문 + Appendix) — 모든 \cite/\citet 인스턴스 전수 역방향 검증
inputs:
  - paper/05_manuscript/MANUSCRIPT_v2_draft.md (v2-draft-r2)
  - paper/04_references/library/ (card 49; 인용 44 key)
  - paper/04_references/CLAIM_CITATION_MAP.md (r3)
quarantine_rule: EXCERPT_UNVERIFIED 3건 (zhang2022selfdistill, xu2018kpivae, ruff2020deepsad) — 서지 수준 인용만 허용; 구체 내용 귀속은 적발
verdict_legend: |
  S  = SUPPORTED      (card 발췌/abstract가 주장을 실제 지지)
  P  = PARTIAL        (부분·약한 지지 — 주장 강도/표현 조정 필요)
  U  = UNSUPPORTED    (card 근거 없음 또는 card 사실과 모순 — 조치안 제시)
  B  = BIBLIO-ONLY    (용어 귀속·존재·출처 수준 인용 — 적합)
totals: {instances: 109, SUPPORTED: 53, PARTIAL: 18, UNSUPPORTED: 6, BIBLIO_ONLY: 32}
---

# P5 Citation Back-Audit r1 — 전 인용 인스턴스 역방향 검증

검증 방향: **본문 → card**. 각 `\cite` 인스턴스에 대해 그 인용이 지지해야 하는 주장 문장을 추출하고, 해당 reference card의 verbatim 발췌/abstract가 그 주장을 실제로 뒷받침하는지 판정했다. 실존 논문에 대한 허위 내용 귀속(할루시네이션) 차단이 목적. 라인 번호는 MANUSCRIPT_v2_draft.md 기준.

---

## §1. 통계 요약

| 판정 | 건수 | 비율 |
|---|---|---|
| SUPPORTED | 53 | 48.6% |
| PARTIAL | 18 | 16.5% |
| UNSUPPORTED | 6 | 5.5% |
| BIBLIO-ONLY (적합) | 32 | 29.4% |
| **합계 (인스턴스)** | **109** | 100% |

- 인용된 고유 key: 44 / 49 (미인용 5: darban2024dacad, jacob2021exathlon, wang2022hscl, xu2023rosas, xiong2020prenorm)
- UNSUPPORTED 6건 중 2건은 본 감사에서 **후보 발췌 확보** (§4 — 2인 검증 경유 후 card 추가 시 해소 가능), 4건은 재서술/인용 교체 필요.
- PARTIAL 18건 중 2건도 후보 발췌 확보로 SUPPORTED 승격 가능 (kim2022rigorous PA%K-AUC).

---

## §2. 인스턴스 전수 표

### §2.1 본문 §1 Introduction

| L | 주장 (요약) | key | 판정 | 처리안 |
|---|---|---|---|---|
| 121 | MTSAD가 산업·안전 응용에서 중요 (CPS/서버/우주 telemetry) | schmidl2022evaluation | S | abstract가 도메인 중요성("production faults, system defects…")을 일반 수준에서 지지. 유지 |
| 121 | 〃 (survey 클러스터) | blazquez2021review | S | abstract "detection of outliers or anomalies that may represent errors or events of interest" — 유지 |
| 122 | 이상이 단일 채널이 아닌 **다채널 상관 편차**로 발현 | xu2022anomalytransformer | **P** | AT 발췌는 **시간적** association(인접 집중) 논리 — 채널 간 상관 주장 직접 지지 약함. **deng2021gdn 추가/교체 권고** (GDN abstract: "detects and explains anomalies which deviate from these [inter-sensor] relationships" — 직접 지지) |
| 122 | 〃 | wu2025catch | **P** | CATCH abstract는 "channel correlations 포착 부족" 문제의식 지지 — 부분 지지. GDN 보강 시 함께 유지 가능 |
| 122 | point 단위 전수 라벨링 비현실적 → 비지도 지배 | wang2025nrdetector | S | card §1 verbatim 지지. **단 A2 경고**: 본문 "labeling every anomalous time point is impractical at scale"이 원문 "labeling every anomalous time point is neither practical nor precise"와 6어절 연속 일치 — 근접 의역. **재표현 권고** (예: "exhaustive point-level annotation is infeasible at scale") |
| 124 | 재구성 계열 클러스터 | zong2018dagmm / su2019omnianomaly / audibert2020usad / song2023memto | S ×4 | 4편 모두 abstract가 재구성(또는 재구성오차+밀도) 기반임을 지지. 유지 |
| 124 | 예측 계열 | deng2021gdn | S | abstract "predict the expected behavior" — 유지 |
| 124 | association-discrepancy·contrastive 계열 ("normal/anomalous attention 패턴의 구조적 격차") | xu2022anomalytransformer | S | association discrepancy + minimax — 지지 |
| 124 | 〃 | yang2023dcdetector | S | "dual attention contrastive representation learning" — 지지 |
| 124 | 〃 | wu2025catch | **P** | CATCH abstract는 자신을 **주파수 patching 기반 재구성** 계열로 위치시킴; masked-attention은 채널 융합용 — "normal/anomalous attention 격차 활용" 귀속은 과함. **재구성 클러스터로 이동 또는 "channel-correlation-aware 재구성" 식 재서술 권고** |
| 124 | "self-supervised 접근(보조 목적함수로 시간 표현 학습)" | tuli2022tranad | **P** | TranAD abstract = adversarial training + self-conditioning + MAML; "self-supervised pre-training" 분류는 약함. 클러스터 명칭 완화 권고 (예: "transformer/일반 백본 기반 보조-목적함수 계열") |
| 124 | 〃 | wu2023timesnet | **P** | TimesNet = task-general backbone (2D-variation). 동일 완화 권고 |
| 128 | 현실 train에 fault/attack 기록 유래 소수 라벨 존재 | wang2025nrdetector | S | card §1 "acquiring weak labels by simply indicating the occurrence of anomalous events…" + abstract "detected abnormal events" — 지지 |
| 130 | 표준 벤치마크 원본 train split에는 **labeled anomaly가 구조적으로 부재** | liu2024elephant | **U** | card 주의사항 명시: "clean-train 가정 명시 발췌 미확보". abstract 페이지 재확인(본 감사 WebFetch)에서도 해당 서술 미발견. **조치 §5-U1** |
| 130 | 〃 | schmidl2022evaluation | **U** | card에 train-split 라벨 구성 관련 발췌 없음. **조치 §5-U1** |
| 135 | NRdetector는 표현학습을 라벨-불가지 사전학습 백본에 위임 | wang2025nrdetector | S | card 주의사항 + NRDETECTOR_DOSSIER D1/D3/D5 (card-fact). 권고: D1/D3/D5 해당 verbatim을 card 본문 발췌로 승격 (Phase 6, 2인 검증) |

### §2.2 본문 §2 Related Work

| L | 주장 (요약) | key | 판정 | 처리안 |
|---|---|---|---|---|
| 164 | 재구성 계열 (§2.1 재서술) | zong / su / audibert / song | S ×4 | 유지 |
| 164 | 예측 계열 | deng2021gdn | S | 유지 |
| 164 | "transformer가 시간 의존성 학습" | xu2022anomalytransformer | S | 유지 |
| 164 | "**inter-channel contrasts**" | yang2023dcdetector | **P** | DCdetector의 contrast는 patch-wise/in-patch **시간 축 dual-view** — 채널 간 대조 아님. "multi-scale contrastive views" 류로 재서술 권고 |
| 164 | 〃 | wu2025catch | **P** | CATCH는 채널 상관 발견 ✓이나 "contrast" 명명·재구성 계열 자기 위치와 어긋남 — 위와 함께 재서술 |
| 164 | "Transformer 기반 self-supervised pre-training을 TSAD에 직접 적용" | tuli2022tranad | **P** | "pre-training" 부적합 (TranAD는 AD 목적함수 직접 학습). 표현 완화 |
| 164 | 〃 | wu2023timesnet | **P** | 동일. TimesNet은 범용 백본 — "범용 시계열 백본의 TSAD 적용" 식 권고 |
| 170 | PU learning 정의 (positive 확보 + unlabeled 혼재) | bekker2020pusurvey | S | abstract verbatim 정확 일치 — 유지 |
| 170 | 〃 | duplessis2014pu | S | abstract 지지 — 유지 |
| 170 | 비용민감 risk 최소화 / non-negative risk estimator | kiryo2017nnpu | S | abstract "non-negative risk estimator for PU learning" + duplessis "cost-sensitive learning" — 유지 |
| 170 | "**two-step 기법: reliable negative 추출 후 분류기 학습**" | elkan2008pu | **U** | **오귀속.** Elkan & Noto는 SCAR 가정 하의 **상수배 확률 보정/가중치** 방법 (card abstract: "predicts probabilities that differ by only a constant factor…") — reliable-negative 2단계 기법이 아님. two-step 정의의 verbatim 출처는 **bekker2020pusurvey §5** (card에 확보됨: "Two-step techniques … 1) identifying reliable negative examples, 2) learning…"). **조치 §5-U2** |
| 170 | "**이미지** AD에 deviation network로 적용 (소수 라벨)" | pang2019devnet | **P** | "few labeled anomalies + deviation network" ✓; **"image" 도메인 한정은 abstract 미지지** (DevNet KDD19 실험은 표 형식 중심 — card 비고 "이미지/표 형식"). "image" 삭제 또는 "image and tabular" 권고 |
| 170 | "deep semi-supervised **one-class** objectives" | ruff2020deepsad | **P** (격리) | 격리 3건 중 1: 서지 수준만 허용. "deep semi-supervised AD" 자체는 제목 수준 지지 ✓이나 "**one-class**"는 내용 귀속 (검증된 abstract에도 'one-class' 없음 — 정보이론/entropy 프레임). **"deep semi-supervised anomaly detection objectives"로 축약 권고** (제목 수준 → 격리 충족). ※ card 주: abstract 자체는 VERIFIED — 격리 목록과 card 상태 간 정합 재확인 필요 |
| 172 | weakly supervised 계열: coarse 라벨로 분류/순위; 라벨이 유일 학습 신호, 재구성 pretext 없음 | sultani2018deepmil / lee2021wetas / liu2024treemil | S ×3 | 3 card 모두 직접 지지 (MIL ranking / classification+alignment / series-level MIL; pretext 부재는 card 차별화 노트와 정합). 유지 |
| 172 | "두 편의 **semi-supervised variational** 모델 … 표현학습은 **largely label-agnostic**: 라벨이 보조 loss로만 들어가고 **latent space의 gradient를 형성하지 않음**" | xue2022fewpositive | **U (모순)** | **최중대 적발 — §3-① 상세.** ① Xue & Yan은 **variational이 아님** (AR/LSTM — abstract verbatim + 본문 확인). ② "label-agnostic/gradient 미형성"은 abstract와 **정면 모순**: "loss components to **encourage representations** that separate normal versus few positive examples" (end-to-end, 본문 확인). **조치 §5-U3 (재서술)** |
| 172 | 〃 | huang2022slavae | **P** | "variational" ✓ (VAE). 단 "라벨이 보조 loss 항으로 들어간다"는 메커니즘 서술은 abstract 미확인 (semi-supervised VAE + active learning만 확인) — 헤지 필요. §5-U3 재서술에 포함 |
| 174 | NRdetector = PU 공식화; **스스로 'PU×TSAD는 novel scenario, 선행 희소' 주장**; 파이프라인(사전학습 백본→고정 표현 위 PU 분류기) | wang2025nrdetector | **P** | 파이프라인·multi-stage = card-fact ✓ (주의사항+dossier). 단 "novel scenario" 자인(自認) verbatim은 **card에 미수록** (MAP C-022에만 인용) — card에 발췌 승격 필요 (2인 검증). WETAS-유래 백본도 dossier-fact. 발췌 승격 전까지 표현 유지 가능하나 근거 포인터 보강 권고 |
| 174 | (인용 없음) "CSMAD는 self-supervised 표현학습 objective의 gradient에 labeled anomaly를 통합하는 **최초의 end-to-end MTSAD 모델**" | — | — | **D-008 위반 — §3-① 상세.** §1 L140의 스코핑판("masked-reconstruction self-distillation + GRL adversarial 결합 최초")과 달리 §2.2의 이 문장은 광의 최초성 주장 — Xue & Yan이 사실상 반례 (AR pretext + 표현형성 loss, end-to-end). §1 스코핑 문구로 통일 필수 |
| 178 | MAE: random patch masking + 재구성 → 강한 전이 표현 | he2022mae | S | abstract verbatim 지지 — 유지 |
| 178 | "유사한 **patch-and-mask** 연산이 일부 시계열 모델에 존재 (독립 발전)" | fang2024tfmae | **P** | TFMAE의 masking은 **window 기반 시간 masking + 진폭 기반 주파수 masking** — patch 단위 아님. "similar masking-based reconstruction objectives" 류로 재서술 권고 |
| 180 | KD의 AD 적용: 사전학습 teacher vs 저용량/랜덤초기화 student 격차 | bergmann2020uninformed | S | abstract 직접 지지 — 유지 |
| 180 | 〃 | deng2022reverse | S | abstract 직접 지지 (reverse distillation, T-S representation discrepancy) — 유지 |
| 180 | "self-distillation은 Zhang et al.이 **efficient network compression용으로 도입; 한 아키텍처에 teacher + 내부 student head들**" | zhang2022selfdistill | **U (격리 위반)** | 격리 3건 중 1 — 서지 수준만 허용인데 **구체 내용 귀속 2건** ("compression 목적", "teacher+내부 student heads" 구조). **단, 본 감사에서 후보 abstract 확보 (§4-1)** — 내용 자체는 후보 발췌와 정합 (deepest classifier→shallower classifiers, same-model transfer, 배포 효율 동기). 2인 검증 경유 card 등재 시 본 문장 유지 가능; 그 전에는 제목 수준으로 축약 |
| 180 | SDMAE: video AD에 self-distillation 적용; MAE 내 deeper teacher + shallower student decoder; T-S 재구성 discrepancy로 스코어링 | ristea2024sdmae | S | card 발췌 1·2·5 + abstract 정확 정합 — 유지 |
| 184 (각주) | "self-distillation 용어는 Zhang et al.을 따름" | zhang2022selfdistill | **B** ✓ | 용어 귀속 — 격리 규칙 충족. 유지 |
| 184 (각주) | 〃 + Ristea et al. | ristea2024sdmae | S | 유지 |
| 184 (각주) | "GRL은 \cite{ristea2024sdmae}의 **unsupervised video setting에는 부재**" | ristea2024sdmae | S | 부재(negative) 주장 — card 전체 발췌(full_html, FULL grade)와 정합. 경미 주의: SDMAE는 합성 이상 supervision 사용 — "unsupervised"는 video AD 관례상 통용되나 "label-free (no real labeled anomalies)" 표현이 더 정밀 (선택) |

### §2.3 본문 §3 Methodology

| L | 주장 (요약) | key | 판정 | 처리안 |
|---|---|---|---|---|
| 217 | linear patchify 원리는 MAE 계보 | he2022mae | S | card A1 발췌 "encoder embeds patches by a linear projection" — 유지 |
| 231 | mask token 삽입 + decoder 재구성 = 표준 MAE 설계 | he2022mae | S | abstract "decoder that reconstructs … from the latent representation and mask tokens" — 유지. ("self-attention-only"는 자기 모델 기술로 읽힘 — MAE 자체 귀속 아님, 문제없음) |
| 238 | "self-distillation 원리의 적응" | zhang2022selfdistill | **B** ✓ | 계보 수준 — 격리 충족 (경계선이나 '원리 계보' 포인터로 적합). 유지 |
| 238 | 〃 | ristea2024sdmae | S | 유지 |
| 244 | λ_rev: \citet{ganin2016dann}의 sigmoid schedule | ganin2016dann | S | card A1 발췌 (λ_p = 2/(1+exp(−γp))−1, γ=10) — 유지 |
| 268 | 표준 focal loss의 modulating factor는 raw prediction 유래 (우리 변형과의 차이) | lin2017focal | S | card A1 발췌 (p_t 정의 Eq.2 + FL Eq.4) — 유지 |
| 267 | (인용 없음) "**SDMAE's anomaly-overlook supervision**" 명명 | (ristea2024sdmae 귀속) | **P** | card 경고: "'anomaly overlook' 용어는 원문에 없음". 원문 메커니즘("reconstruct the original frames **without anomalies**")의 paraphrase로 교체하거나 '우리 명명'임을 표기 — §3-⑤ 상세 |

### §2.4 본문 §4 Experiments + §5

| L | 주장 (요약) | key | 판정 | 처리안 |
|---|---|---|---|---|
| 318 | 데이터셋 출처 5종 (SWaT/WaDi/PSM/SMD/SMAP·MSL) | goh2016swat / ahmed2017wadi / abdulaal2021psm / su2019omnianomaly / hundman2018telemanom | **B** ✓ ×5 | 출처 인용 — card 서지·abstract 정합 (SMD는 OmniAnomaly가 공개 주체임이 abstract로 확인됨). 유지 |
| 331 | 재분할 선례: NRdetector의 7:3 re-split (train에 anomaly 포함) | wang2025nrdetector | S | card verbatim "split the set of all segments by 7:3 ratio…" — 유지 |
| 358 | AR-threshold "(1−r) quantile … **per the convention of** [AT]" | xu2022anomalytransformer | **P** | §3-③ 상세. r-비율 quantile 관행 자체는 A1 verbatim 지지 ✓. 단 AT는 **validation set + 데이터셋별 고정 r(0.1/0.5/1%)**, 본문은 **evaluation set + ground-truth r** — "per the convention of"를 "following the anomaly-ratio thresholding convention introduced by"로 완화하고 차이 1구절 부기 권고. card frontmatter(R30 유지) vs 본문(A1 해제) 상태 모순 정리 필요 |
| 362 | 다지표 철학 클러스터 | kim2022rigorous / paparrizos2022vus / liu2024elephant / wang2025nrdetector | S ×4 | NRdetector abstract "11 different evaluation metrics" 포함 전원 지지 — 유지 |
| 362 | **PA%K-AUC F1** 귀속 (K 적분으로 K 의존 제거) | kim2022rigorous | **P→해소 가능** | card 발췌는 PA%K 프로토콜만 — **AUC-over-K는 card 미수록**. 본 감사에서 후보 verbatim 확보 (§4-2: "it is recommended to measure the area under the curve of F1_PA%K obtained by increasing K from 0 to 100") — 2인 검증 후 card 추가 시 SUPPORTED |
| 362 | VUS-PR/VUS-ROC 제안 논문 | paparrizos2022vus | S | abstract 직접 지지 — 유지 |
| 362 | VUS-PR을 대규모 연구가 최신뢰 지표로 평가 | liu2024elephant | S | abstract verbatim — 유지 |
| 362 | Affiliation F1 (시간 거리 기반 local 평가) | huet2022affiliation | S | abstract 발췌 2·3 — 유지 |
| 366 | PA F1 (K=0)의 원전 | xu2018kpivae | **B** ✓ | 용어 원전 포인터 — 격리 충족. 유지 |
| 366 | "random score도 PA 하에서 SOTA 도달 가능" | kim2022rigorous | S | card 발췌 1 verbatim; 본문은 적절히 paraphrase — 유지 |
| 370 | simple~lightweight 9종 following | sarfraz2024quovadis | S | card: simple 5 + neural 3 + GCN-LSTM(QuoVadis 도입) = 9 — 유지 |
| 370 | 기성 deep TSAD 6종 (baseline 출처) | xu2022anomalytransformer / tuli2022tranad / audibert2020usad / zong2018dagmm / deng2021gdn / su2019omnianomaly | **B** ✓ ×6 | 유지 |
| 370 | 최신 7종 (baseline 출처) | fang2024tfmae / lai2023npsr / wu2023timesnet / yang2023dcdetector / song2023memto / luo2024moderntcn / wu2025catch | **B** ✓ ×7 | 유지 (ModernTCN "Spotlight" 표기는 card상 VERIFY_REQUIRED — 본문 미사용 ✓) |
| 370 | weakly supervised 4종 (라벨 활용 학습) | sultani2018deepmil / lee2021wetas / liu2024treemil / wang2025nrdetector | **B** ✓ ×4 | 유지 |
| 376 | "순수 비지도에서 labeled anomaly의 **최선 활용 = 오염 샘플 제거**" | bekker2020pusurvey | **U** | survey card에 해당 주장 발췌 없음 (PU 정의·계열 분류만). 이 명제는 **설계 논거**이며 MAP C-074의 매핑 근거는 NRdetector §5.1("trained by using only normal segments")이었음. **조치 §5-U4** |
| 387 | NRdetector = 최근접 weak-sup 비교; 구조 차이 = multi-stage 파이프라인 | wang2025nrdetector | S | card-fact (주의사항/dossier) — 유지 |
| 427 | NRdetector sweep은 **incorrect 라벨 비율**(노이즈) 축 — 우리 희소율 축과 상이 | wang2025nrdetector | S | card 발췌 ("robust results under different label noise rates") + card 주의("축 의미 차이 명시 필요")를 본문이 정확히 이행 — 유지 |
| 458 / §5 | (인용 없음) | — | — | — |

### §2.5 Appendix

| L | 주장 (요약) | key | 판정 | 처리안 |
|---|---|---|---|---|
| 511 | baseline 구성 (simple 5 + neural 3 + GCN-LSTM) following 프로토콜 연구 | sarfraz2024quovadis | S | card 활용절과 정확 일치 — 유지 |
| 525 | conventional PA 정의 ("한 점이라도 검출되면 segment 전체 검출 처리") | xu2018kpivae | **P (격리-조건부)** | 격리 3건 중 1 — 내용 귀속에 해당. **단 card 본문에 이미 A1 EXCERPT_RESOLVED verbatim 존재** (§4.2 "if any point in an anomaly segment … all points in this segment are treated as if they can be detected") — 본문 서술과 정확 정합. card frontmatter(abstract_pending)·주의사항(EXCERPT_UNVERIFIED 유지)과 본문 A1 기록의 **상태 모순**이 격리 목록의 원인으로 추정 — 2채널 재검증으로 상태 정리 시 SUPPORTED 승격. 후보 발췌로 §4-3에 재기록 |
| 526 | PA%K 정의 (K% 초과 시에만 adjust; K=0↔PA, K=100↔pointwise) | kim2022rigorous | S | card 발췌 3 (A1, §4 + Fig.6 caption) — "strictly more than"도 "exceeds"와 정합. 유지 |
| 529 | per-K threshold 재최적화 + "**reference implementation** of [kim2022] 따름" | kim2022rigorous | **P** | 구현체(코드) 제공 여부는 card 미확인. "following the protocol of"로 완화하거나 구현 repo 검증 후 각주화 권고 |
| 535 | VUS 정의 (threshold + 시간 허용폭 동시 sweep, 3차원 volume) | paparrizos2022vus | S | abstract (VUS 도입 + threshold-independent + range-aware) — 유지. buffer 세부는 통상 지식 수준 |
| 539 | Affiliation 정의 + "adversarial scoring에 대한 형식적 강건성 보장" | huet2022affiliation | S | abstract "theoretical properties … ensure robustness against adversary strategies" — 유지 |
| 562–568 | Table A.4 Source 열 (7행) | goh / ahmed ×2 / abdulaal / su / hundman ×2 | **B** ✓ ×7 | 출처 인용 — 유지 |
| 665 | λ_rev sigmoid schedule (Eq. C.1) | ganin2016dann | S | card A1 §5.2 verbatim과 수식 일치 — 유지 |
| 681 | 표준 focal loss의 p_t 정의와의 차이 (Eq. C.3 해설) | lin2017focal | S | card A1 §3.1 Eq.2/§3.2 Eq.4 verbatim — 유지 |

---

## §3. 특별 정밀 검증 5건

### ① 최초성·gap 주장 (D-008 스코핑) — **위반 1건 + 오귀속 1건 (중대)**

- **§1 L140 (스코핑판)**: "the first architecture combining masked-reconstruction self-distillation with gradient reversal … in a contaminated semi-supervised multivariate TSAD setting" — **D-008 준수** ✓ (메커니즘 한정 + "to our knowledge").
- **§2.2 L174 (광의판)**: "the first end-to-end model for multivariate TSAD that integrates labeled anomalies into **the gradient of a self-supervised representation learning objective**" — **D-008 위반**. 본 감사에서 Xue & Yan 본문(ar5iv 2207.00705)을 확인한 결과: LSTM-AR pretext(자기지도적 예측) + MSE margin loss + 보조 분류 task가 **end-to-end로 표현을 형성** — 광의판의 사실상 반례. **L140 스코핑 문구로 통일 필수.**
- **xue2022fewpositive 차별화 서술 (L172)**: card 사실과 **2중 불일치** — (i) "variational" 오귀속 (실제 AR/LSTM), (ii) "labels … rather than shaping the gradient of the latent space"는 abstract의 "loss components to **encourage representations** that separate normal versus few positive examples"와 정면 모순. 실존 논문에 대한 허위 내용 귀속에 해당 — **반드시 수정** (조치 §5-U3).
- **huang2022slavae 차별화 서술**: "variational" ✓. 메커니즘("auxiliary loss로만") 단정은 abstract 미확인 — "active-learning 루프 기반 semi-supervised VAE" 수준의 사실 서술 + 차별축(고정 라벨 vs 반복 라벨링 루프; masked-reconstruction/GRL 부재) 중심으로 재서술하면 card 정합.
- 올바른 차별화 축 (card-지지 가능): pretext 종류(AR 예측 vs masked-reconstruction self-distillation), 라벨 개입 방식(직접 판별 loss vs **adversarial gradient-level 억제(GRL)**), SLA-VAE의 active-learning 루프 의존, 도메인(KPI/온라인 시스템 vs 일반 MTSAD 벤치마크).

### ② PA-F1 비판 (kim2022rigorous) — **정합** ✓ (+1건 해소 가능)

- L366 "even a random score can reach state-of-the-art levels": card 발췌 1 verbatim 지지, 본문은 비-verbatim paraphrase ✓ (card의 복사 금지 경고 이행).
- L526 PA%K 정의: card A1 발췌와 정확 정합 ✓ ("strictly more than K%" = "exceeds … K").
- L362 **PA%K-AUC** 귀속: card 발췌만으로는 미지지였으나 **원문 §4.2에서 AUC-over-K 권고 verbatim 확보** (§4-2) — 2인 검증 후 card 추가 시 완전 정합.
- L529 "reference implementation": 미확인 잔여 — 표현 완화 또는 repo 검증.

### ③ AR-threshold 선례 (xu2022anomalytransformer) — **부분 정합 (PARTIAL)**

- 확보 발췌 (card A1, §4 Implementation details): "The threshold δ is determined to make **r proportion data of the validation dataset** labeled as anomalies. … we set r = 0.1% for SWaT, 0.5% for SMD and 1% for other datasets."
- 본문 L358: "(1−r) quantile … **r being the labeled anomaly fraction of the evaluation set** … per the convention of [AT]".
- 정합: r-비율 quantile thresholding **메커니즘**의 선례 귀속 ✓. 불일치: (i) AT는 r을 **데이터셋별 고정 하이퍼파라미터**로 설정, 본문은 **평가셋 ground-truth 비율**; (ii) AT는 **validation set** 분위수, 본문은 evaluation-set score 분포. 본문이 자기 r 정의를 투명히 명시하므로 허위 귀속은 아니나 "per the convention of"는 프로토콜 동일성으로 읽힐 여지 — **"following the anomaly-ratio thresholding convention introduced by [AT]" + (AT는 고정 r/validation 분위수) 1구절 부기 권고**.
- 부수: card frontmatter verified_note("R30 hold maintained")와 card 본문("A1 EXCERPT_RESOLVED, R30 보류 해제")·MAP §6-2(해제)가 모순 — card frontmatter 갱신 필요 (Phase 6).

### ④ R21 계보 (zhang2022selfdistill) — **서지 수준 위반 1건 (해소 가능)**

- L184 각주(용어 귀속)·L238(원리 계보 포인터): **서지 수준 — 적합** ✓.
- L180: "introduced … for efficient network compression, where one architecture contains a teacher and internal student heads" — **내용 귀속 2건으로 격리 위반**. 단 본 감사에서 Semantic Scholar 경유 후보 abstract 확보 (§4-1): "attaches several attention modules and **shallow classifiers at different depths** … distills knowledge from the **deepest classifier to the shallower classifiers**", "knowledge transfer **in the same model**", 동기 = 연산·파라미터 폭증으로 인한 배포 제약 — 본문 서술과 **내용 정합** (할루시네이션 아님). 2인 검증 경유 card 등재 후 유지; 등재 전 출판 시에는 "A more compact formulation is self-distillation \cite{zhang2022selfdistill}" + 구조 상세는 제거하는 축약판 권고.

### ⑤ SDMAE 서술 (ristea2024sdmae) — **구조 서술 정합 ✓, 용어 1건 주의**

- L180 "deeper teacher + shallower student within a masked autoencoder, T-S discrepancy 스코어링" — card abstract·발췌 2·5와 정합 ✓.
- L182 "우리: 공유 encoder에서 **독립 병렬 분기** — SDMAE처럼 **teacher decoder 내부 branch-off가 아님**" — card 발췌 2 verbatim("the student decoder branches out from the teacher after the first transformer block of the main decoder")과 정확 정합 ✓. **branch-off 구조 오서술 없음.**
- L184 "unsupervised video setting" — SDMAE는 합성 이상 supervision 사용; video AD 관례상 "unsupervised" 통용되어 허용 범위이나 "label-free (real labeled anomalies 부재)"가 더 정밀 (선택 수정).
- **L267 "SDMAE's anomaly-overlook supervision"** — card 경고 명시: "'anomaly overlook' 용어는 원문에 없음". 타 논문 메커니즘에 본 논문 조어를 그 논문의 명칭처럼 부여 — **paraphrase로 교체 권고**: 예) "Whereas SDMAE suppresses anomaly reconstruction in the **target/loss space** (training the model to reconstruct anomaly-free targets), our GRL operates in the gradient space …". (target/loss-space 작동 자체는 card 발췌 3이 지지 ✓)

---

## §4. 확보한 후보 발췌 (2인 검증 경유 필요 — card에 직접 추가하지 않음)

> 본 절의 발췌는 모두 본 감사의 WebFetch로 확보한 **후보**다. VERIFICATION_LEDGER 절차(2채널 독립 검증)를 거치기 전에는 card·원고 어디에도 정본으로 사용 금지.

**4-1. zhang2022selfdistill — abstract 전문 후보** (출처: api.semanticscholar.org, DOI 10.1109/TPAMI.2021.3067100 질의, 2026-06-11):
> "Remarkable achievements have been obtained by deep neural networks in the last several years. However, the breakthrough in neural networks accuracy is always accompanied by explosive growth of computation and parameters, which leads to a severe limitation of model deployment. In this paper, we propose a novel knowledge distillation technique named self-distillation to address this problem. Self-distillation attaches several attention modules and shallow classifiers at different depths of neural networks and distills knowledge from the deepest classifier to the shallower classifiers. Different from the conventional knowledge distillation methods where the knowledge of the teacher model is transferred to another student model, self-distillation can be considered as knowledge transfer in the same model - from the deeper layers to the shallow layers. …"
- 지지 대상: L180 (C-028) — "efficient network compression" 동기 + "teacher와 내부 student heads" 구조. S2 미러 기준이므로 IEEE 정본 대조 필요.

**4-2. kim2022rigorous — AUC-over-K 권고 후보** (출처: ar5iv 2109.05257 §4.2, 2026-06-11):
> "If a user wants to remove the dependency on K, it is recommended to measure the area under the curve of F1_PA%K obtained by increasing K from 0 to 100."
- 지지 대상: L362/L528–530 (C-047) — PA%K-AUC F1의 K-적분 귀속. arXiv판 기준이므로 AAAI 정본(ojs PDF) 대조 필요.

**4-3. xu2018kpivae — PA 정의 (card 내 기존 A1 발췌의 재확인 기록)** (카드 본문에 이미 존재; 격리 상태 정리용):
> "if any point in an anomaly segment in the ground truth can be detected by a chosen threshold, we say this segment is detected correctly, and all points in this segment are treated as if they can be detected by this threshold." (§4.2)
- 지지 대상: L525 (C-051). card frontmatter(abstract_pending / EXCERPT_UNVERIFIED 유지)와 본문 A1 기록 모순 — 2채널 재검증으로 상태 확정 필요.

**4-4. xue2022fewpositive — 반증 확정 발췌 후보** (출처: ar5iv 2207.00705, 2026-06-11; D-008 재서술 근거 강화용):
> "We formulated two approaches: 1) MSE margin loss; 2) auxiliary classification task." (방법론: LSTM 기반 AR 모델; end-to-end)
- 용도: §2.2 차별화 재서술 시 "AR 예측 모델 + margin/보조분류 loss" 사실 서술의 근거. (variational 아님 확정.)

**4-5. liu2024elephant — clean-train 서술 확보 실패**: proceedings abstract 페이지 재확인 결과 train-split 라벨 구성 관련 서술 없음. 본문 PDF 정독은 미수행 — L130은 발췌 의존이 아닌 재서술(§5-U1)을 권고.

---

## §5. UNSUPPORTED 6건 — 조치안

**U1. L130 — liu2024elephant + schmidl2022evaluation ("원본 train split에 labeled anomaly 구조적 부재")**
- (b) **재서술 권고 (1순위)**: 사실 명제의 근거를 자체 실측(§A.3 Training-label semantics + EXPERIMENT_PROTOCOL_TRUTH)으로 옮기고, 두 인용은 일반 벤치마크 관행 비판으로 분리. 예: "…whose original training splits contain no labeled anomalies by construction (per-dataset label semantics in Appendix §A.3); benchmark studies have independently criticized dataset and evaluation practices in this field \cite{liu2024elephant, schmidl2022evaluation}."
- (c) 대안: 데이터셋 원논문 5종을 라벨 의미론의 출처로 병기 (§A.3과 동일 구조).
- (a) 원문 fetch: abstract에서 미발견 (§4-5). Elephant 본문 정독은 Phase 6 옵션.

**U2. L170 — elkan2008pu ("two-step reliable-negative 기법")**
- (c) **인용 교체 (1순위)**: two-step 정의의 verbatim 출처는 bekker2020pusurvey §5 (card 확보) — "two-step techniques that extract reliable negatives before training a classifier \cite{bekker2020pusurvey}". elkan2008pu는 "class-prior 기반 확률 보정/가중 (SCAR 가정) \cite{elkan2008pu}"으로 재배치하거나 삭제.
- (b) 대안: 문장을 survey 분류(3계열) 기준으로 재서술하고 개별 원류 인용은 kiryo만 유지.

**U3. L172 — xue2022fewpositive ("semi-supervised variational … label-agnostic … gradient 미형성")**
- (b) **재서술 필수 (유일안)**: "variational" 제거 + "label-agnostic/gradient 미형성" 철회. 권고 문안 방향: "Two earlier semi-supervised models address label scarcity in multivariate time series — an autoregressive normality model with margin and auxiliary-classification losses \cite{xue2022fewpositive}, and a semi-supervised VAE coupled with an active-learning loop \cite{huang2022slavae}. In both, labels act through discriminative loss terms attached to a generative or predictive normality objective; neither employs a masked-reconstruction self-distillation pretext, nor adversarial gradient-level suppression of anomaly information." (차별화 축 = 메커니즘, 라벨의 표현 형성 자체를 부정하지 않음.) §2.2 L174 광의 최초성 문장도 §1 L140 스코핑판으로 동시 교체.

**U4. L376 — bekker2020pusurvey ("비지도의 최선 = 라벨 제거")**
- (b) **재서술 (1순위)**: 설계 논거로 두고 인용 제거: "Under a purely unsupervised objective, a labeled anomaly can only be used negatively — as a contaminating sample to remove; Q3 grants each unsupervised method this most favorable use." 
- (c) 대안: NRdetector §5.1의 normal-only semi-supervised 변형을 선례로 인용 — 단 해당 verbatim("trained by using only normal segments")은 card 미수록 (MAP C-074에만 존재) → 2인 검증 경유 card 발췌 추가 후 \cite{wang2025nrdetector}로 교체.

**U5. L180 — zhang2022selfdistill (격리 위반 — 내용 귀속)**
- (a) **후보 발췌 확보 완료 (§4-1)** — 2인 검증 → card 등재 → 현행 문장 유지 (내용 정합 확인됨).
- 임시안 (검증 전 출판 경로): "A more compact formulation is self-distillation \cite{zhang2022selfdistill}, which performs distillation within a single architecture" 수준으로 축약 (제목+최소 구조).

**U6. L130과 별건 아님 — (집계 명시) U1이 2 인스턴스(liu2024elephant, schmidl2022evaluation)를 포함**: U 합계 6 = U1(2) + U2(1) + U3(1) + U4(1) + U5(1).

---

## §6. CLAIM_CITATION_MAP 갱신 입력 (주장↔근거 발췌 포인터)

Phase 5 fix 후 MAP에 반영할 변경점:

| Claim ID | 현행 MAP 상태 | 본 감사 결과 → 갱신 입력 |
|---|---|---|
| C-008 / C-045 | VERIFIED (clean-train 서술은 verifier 발췌 조건부) | 조건 **불충족 확정** (Elephant/Schmidl 모두 발췌 부재) → 근거를 "자체 실측(§A.3) + 데이터셋 원논문"으로 전환, Elephant/Schmidl은 '벤치마크 관행 비판' 한정 포인터로 강등 |
| C-011 / C-025 | VERIFIED (D-008 재서술 전제) | §1 L140 **준수** / §2.2 L174 **위반** — L174 교체 후 '스코핑판 단일 문구' 기준으로 잠금. xue2022 차별화 문안은 §5-U3 문안으로 교체 (variational 오귀속 제거) |
| C-020 | VERIFIED | two-step 발췌 포인터를 **bekker2020pusurvey §5**로 지정; elkan2008pu 역할 재정의 (SCAR/보정) |
| C-021 | VERIFIED | pang2019devnet "image" 한정 해제 ("image and tabular" 또는 도메인 무표기); ruff2020deepsad "one-class" 표현 제거 (격리 충족형 문구) |
| C-022 / C-024 | VERIFIED | "novel scenario" 자인 verbatim을 wang2025nrdetector card 발췌로 승격 (2인 검증) — L174 근거 포인터 보강 |
| C-028 | VERIFIED (서지 충분 전제) | **후보 abstract 확보 (§4-1)** — 2채널 검증 후 card 등재 시 L180 내용 귀속 허용으로 상향 |
| C-047 | VERIFIED | **AUC-over-K verbatim 후보 확보 (§4-2)** — card 추가 시 'PA%K-AUC' 명명 귀속 완전 지지. L529 "reference implementation"은 별도 확인 항목 신설 |
| C-051 | VERIFIED | card 상태 모순 정리 (frontmatter abstract_pending vs 본문 A1 RESOLVED) — 격리 해제 여부 확정 |
| C-053 | VERIFIED (R30 해제) | 발췌 정합 확인 ✓ + **본문 표현 완화 필요** (validation/고정-r vs evaluation/실측-r 차이 부기); AT card frontmatter의 R30 문구 모순 정리 |
| C-002 / C-014 / C-016 | VERIFIED | CATCH·DCdetector의 클러스터 배치 어휘 조정 (inter-channel contrast → 채널상관/대조 view 구분); C-002에 deng2021gdn 포인터 추가 권고 |
| C-004 / C-015 | VERIFIED | TranAD/TimesNet "self-supervised pre-training" 명명 완화 |
| C-031 | VERIFIED | TFMAE "patch-and-mask" → "masking 기반 재구성"으로 어휘 교정 |
| C-035 | VERIFIED | "anomaly-overlook" 명명을 원문 기반 paraphrase로 교체 (card 경고 이행) |
| C-003 | VERIFIED | L122 근접 의역(A2) 재표현 항목 추가 |

---

## §7. 부수 발견 (card 정합성)

1. **xu2022anomalytransformer card**: frontmatter verified_note "R30 hold maintained" ↔ 본문 "[A1 EXCERPT_RESOLVED … R30 보류 해제]" ↔ MAP §6-2 "보류 해제" — 3소스 중 frontmatter만 구판. Phase 6에서 frontmatter 갱신.
2. **xu2018kpivae card**: frontmatter `excerpt_access: abstract_pending` + 주의사항 "EXCERPT_UNVERIFIED 상태 유지" ↔ 본문 "[A1 RESOLVED] abstract 전사 + §4.2 PA verbatim". 격리 목록(과제 지시)의 근거가 이 구판 상태로 보임 — 2채널 재검증으로 확정 필요.
3. **ruff2020deepsad card**: verified_note "Abstract verbatim confirmed" (미검증은 §3 SAD loss 한정) — 격리 목록과 부분 불일치. abstract 수준 인용까지는 실질 안전하나, 본 감사는 보수적으로 격리 규칙을 적용해 판정함.
4. L122 근접 의역 (NRdetector §1, 6어절 연속 일치) — A2 문체 리스크, Phase 6 문체 패스에서 처리 권고.
