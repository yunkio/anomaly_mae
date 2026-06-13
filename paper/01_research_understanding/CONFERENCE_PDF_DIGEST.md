---
phase: 1
agent: pdf-digest
directives: [T1, R2, R5]
last_modified: 2026-06-10
revision: r2 — p1_digests_r1 리뷰(BLOCKER 1 + MAJOR 1 + MINOR 5) 전수 반영 (fixer-3)
fix_log: paper/99_reviews/p1_digests_fixlog_r2.md
source: paper/윤기오_대한산업공학회_2026_춘계.pdf (34 pages, 한국어 학회 발표자료)
---

# 학회 발표자료 Digest — 「교사-학생 신경망의 비대칭적 학습을 통한 준지도 시계열 이상 탐지」

> **[R2] 참고 자료 지위 명시**: 이 PDF는 '참고 자료'다. 발표의 논리 구성·서술 순서는
> **참고용이며 절대적 기준이 아니다**. 논문 작성 시 충분한 판단 후 선별적으로 활용할 것.
>
> **[R5] Notation 비계승 명시**: 본 문서에 옮겨 적은 모든 수식·기호는 **발표 시점 notation**이며,
> 논문에서는 **그대로 계승하지 않는다**. 논문 notation은 별도 설계한다.

**서지**: 윤기오¹, 백준걸¹* — ¹고려대학교 산업경영공학과, {ykio, jungeol}@korea.ac.kr.
대한산업공학회 2026 춘계 발표자료. 영문 방법명(슬라이드 표기): *Semi-Supervised Time Series Anomaly Detection via Self-Distilled MAEs*.

---

## ① 발표자료 전체 구조 (페이지 맵, 총 34p)

| 페이지 | 내용 |
|---|---|
| p1 | 표지 (제목·저자·소속) |
| p2 | 목차: 1. Problem Definition / 2. Background / 3. Proposed Method / 4. Experiments / 5. Conclusion |
| p3 | 문제 정의: 시계열 이상 탐지 정의 + 문제점(불균형, 판별 변수 부재 → 지도학습 곤란) |
| p4 | 데이터 수집의 한계: labeled vs unlabeled — "데이터의 양 vs 정보의 양" 다이어그램 |
| p5 | Semi-Supervised Learning 개념 (Supervised/Semi/Unsupervised 3분류 그림) |
| p6 | why anomaly detection + semi-supervised? (contaminated 가정) |
| p7 | Semi-supervised Learning vs **Positive-Unlabeled Learning** 결정경계 비교 그림 (텍스트 없이 그림만) |
| p8 | Background 1: MAE — "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022), pretext/downstream 구조도 |
| p9–10 | Background 2: "Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors" (CVPR 2024) — teacher-student framework, discrepancy 개념 |
| p11 | Proposed Method 개요: 데이터셋 분해, 총 손실 3항, L_rec / L_disc 수식, GRL 한 줄 설명 |
| p12 | 프레임워크 — Training 아키텍처 그림 (Patchify 1D-CNN → Masking → Transformer Encoder → Student(+GRL classifier) / Teacher decoder) |
| p13 | 프레임워크 — Inference 개념도: 패치를 하나씩 순서대로 마스킹, patch 개수만큼 추론 반복 |
| p14 | Inference 상세: N번 forward pass, patch-level score 수식, auto-scaled 계수, sliding-window 앙상블 → point-level score |
| p15 | 구성요소 정당화 1 — 재구성 오차 (semi-supervised에서 labeled/unlabeled seamless 활용) |
| p16 | 구성요소 정당화 2 — Masked Autoencoder (전역적 의미 구조, augmentation 효과) |
| p17 | 구성요소 정당화 3 — Self-distillation (abnormal bias 방지, 비대칭 복잡도, regularization) |
| p18 | 구성요소 정당화 4 — Gradient Reversal Layer (negative supervision, DANN 스타일 그림 인용) |
| p19 | Experiments — Dataset: SWaT / SWaT(excl22) / WaDi A1 / WaDi A2 / PSM 통계표 + train/test 재구성 설명 |
| p20 | Experiments — Baseline **26종**(+Ours=27) 표 (Simple / Neural / SOTA / Weakly-Supervised / Ours) |
| p21 | Experiments — Evaluation Metrics 7종 표 + 지표 해설 (anomaly-ratio threshold 사용 명시) |
| p22 | Experiments — 메인 결과표 (5개 벤치마크 × F1_PA%K·Aff-F1) + Mean/Worst/Std Rank 표 + 해석 4줄 |
| p23 | Visualization — SWaT anomaly score timeline (recon / scaled_disc 분해) |
| p24 | Visualization — Learning curves (warm-up 250ep teacher-only → student 합류 후 metric 상승) |
| p25 | Visualization — score breakdown (Normal/Disturbing/Anomaly 카테고리별 recon vs disc 기여) |
| p26 | Visualization — GRL contribution trend (Main vs GRL loss, classifier accuracy) |
| p27 | Conclusion (7개 bullet) |
| p28 | Q&A |
| p29–33 | Appendix: 데이터셋별 full 결과표 (SWaT Full / SWaT excl22 / WaDi A1 / WaDi A2 / PSM — 7개 지표 전부) |
| p34 | Appendix: References [1]–[25] |

---

## ② Problem Definition (p3–7)

발표가 정의한 문제 설정 (원문 한국어 표현 인용):

- **시계열 이상 탐지** (p3): "과거 또는 비슷한 시점의 다른 데이터의 보편적인 패턴에서 벗어나거나
  벗어나려는 징후가 있는 패턴을 찾는 것". 문제점으로 ① "Anomaly를 띄고 있는 데이터가 정상
  데이터에 비해 매우 적어 불균형 문제가 심함", ② "Anomaly 임에도 구분할 수 있는 명확한 변수
  내지는 특성이 없는 경우가 많음" → "**일반적인 지도 학습 방법론을 적용하는데 어려움이 있음**".
- **데이터 수집의 한계** (p4): "라벨이 없는 데이터는 상대적으로 훨씬 적은 비용으로 많은 양을
  수집할 수 있음" / "소수나마 존재하는 라벨이 있는 데이터는 매우 중요한 정보를 담고 있으므로,
  두 종류의 데이터 모두 활용할 수 있는 방법이 필요함". (그림: 데이터의 양은 unlabeled가 압도,
  정보의 양은 labeled가 큼.)
- **Semi-supervised 정의** (p5): "소량의 레이블이 있는 데이터를 활용하여 학습의 방향성을
  설정하면서 대규모의 레이블이 없는 데이터를 최대한 활용하고자 하는 방법론". 핵심 전제:
  "특히 이상 탐지의 경우 레이블이 없는 데이터에는 **정상과 이상이 혼재**되어 있음".
- **why AD + semi-supervised** (p6): "이상 탐지 문제는 레이블이 없는 데이터에 정상과 이상이
  함께 섞여 있는 **contaminated 상황**을 가정" / "소수의 레이블을 보조적으로 사용하여 전체
  데이터를 통제하고 학습을 유도하는 semi-supervised learning 접근이 가장 현실적이고 강력한
  해결책**이 될 수 있음**" *(r2: 절단했던 완화 어미 복원 — 원문은 이미 hedge되어 있음)* /
  "Label이 달린 소수의 anomaly data가 전체 unlabeled 데이터에 대한 방향성과 이상
  구분 기준을 더 뚜렷하게 설정하도록 도움".
- **PU 맥락** (p7): "Semi-supervised Learning"과 "Positive-Unlabeled Learning"을 결정경계
  그림으로 나란히 비교 (텍스트 설명 없음). 발표 설정은 labeled가 anomaly(positive) 쪽에 소수
  존재 + 대량 unlabeled(정상·이상 혼재) 구조로 **추정**되어 PU learning 계열과의 연결을 시각적으로
  암시하는 슬라이드다. *(r2: "anomaly 쪽에만"은 digest의 추론 — p11 정식화에는 𝒳ᴺ_lab(labeled
  normal)도 존재하며 슬라이드는 라벨 사용 범위를 명시하지 않음. 정확히는 "손실 수식에는 𝒳ᴬ_lab만
  등장".)* 단, 발표 본문에서 "PU learning"이라는 용어로 방법을 정식화하지는 않았다
  (학습 정식화는 p11의 semi-supervised 4분할 𝒳 정의로만 제시).

요약: **contaminated(정상+미발견 이상 혼재) unlabeled 다수 + 소수 labeled anomaly**를 함께
쓰는 준지도(semi-supervised) 다변량 시계열 이상 탐지가 발표의 문제 설정이다.

---

## ③ Background — 선행 개념의 배치 순서 (p8–10)

발표는 딱 **2개의 선행 연구**만 배경으로 깔았다 (이 순서):

1. **MAE** (He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022 —
   슬라이드에 "15505회 인용" 표기): masking → encoder → decoder 복원의 pretext task 구조와
   downstream 분류 활용 그림. 시계열 적용 논리의 토대.
2. **SD-MAE for Video AD** ("Self-Distilled Masked Auto-Encoders are
   Efficient Video Anomaly Detectors", CVPR 2024 — *저자명은 PDF에 없음(제목·venue만 표기);
   "Ristea et al."은 외부 지식으로 확인한 귀속이며 Phase 4 재검증 대상* (r2)): shared encoder +
   **teacher decoder / student decoder** 구조. 발표 요약(p10 원문): "Teacher는 정상 데이터를 복원하는 복잡한
   디코더를 가지며, student는 teacher의 출력을 얕은 디코더로 모방하는 데 집중" / "정상
   데이터에서는 teacher와 student의 결과가 일관되지만, 비정상 데이터에서는 둘 사이의
   discrepancy(출력 불일치)가 커지게 됨".

즉 배경 논리는 "MAE의 마스킹-복원 표현학습" → "video 도메인의 self-distilled MAE가 보여준
teacher–student discrepancy = anomaly signal" → (3장에서) "이를 시계열 + semi-supervised로
이식·확장"의 3단 구조다. GRL(DANN)은 Background 장이 아니라 Proposed Method 내부(p18)에서
구성요소 정당화 그림으로만 등장한다.

---

## ④ Proposed Method (p11–18)

> 이하 모든 수식은 **발표 notation이며 논문에 계승하지 않음** [R5].

### 문제 정식화 (p11)
- 데이터셋: 𝒳 = 𝒳ᴺ_lab ∪ 𝒳ᴺ_unl ∪ 𝒳ᴬ_lab ∪ 𝒳ᴬ_unl
  (정상/이상 × labeled/unlabeled의 4분할. *r2 정정: 정식화에는 𝒳ᴺ_lab(labeled normal)도
  존재하며 슬라이드는 라벨 사용 범위를 명시하지 않는다 — 정확한 사실은 "**손실 수식(ℒ_rec/ℒ_disc의
  ℳᴺ 정의)에는 𝒳ᴬ_lab만 등장**"이고, "학습에 쓸 수 있는 label은 𝒳ᴬ_lab뿐"이라는 서술은
  거기서 나온 추론임.*)

### 아키텍처 (p12 그림)
- **Patchify (1D-CNN)** → **Masking** → **Encoder (Transformer Block)** →
  두 갈래 decoder: **Teacher** (깊은 디코더) / **Student** (얕은 디코더),
  Student 쪽에만 **Gradient Reversal Layer + Labeled Anomaly Classifier** 분기 부착.

> ⚠️ **(r2) 원천 간 모순 — Patchify "1D-CNN"(발표) vs `patchify_mode='linear'`(Notion exp271)**:
> PDF p12/15/16 아키텍처 그림은 일관되게 "Patchify (1D-CNN)"를 명시하나, Notion Page 0의
> 본 baseline(exp271)은 **Linear embedding, CNN 미사용**(`patchify_mode='linear'`, Set C;
> `'patch_cnn'`은 코드에 있으나 본 baseline 미사용 옵션)이다. 발표 시점 구성 vs 현 baseline
> 구성의 차이 또는 발표 그림 오류 — 논문 아키텍처 그림/서술에 직결되므로 코드·실험 ID 기준
> 확정 필요 (→ ⑧ REQUEST에 추가).

### 학습 손실 (p11–12, 발표 notation)
- 총 손실: ℒ = ℒ_rec + ℒ_disc + ℒ_grl
- Teacher 재구성 (ℳ = 해당 윈도우의 masked patch index set):
  ℒ_rec = (1/|ℳ|) Σ_{p∈ℳ} ‖x̂ᵀ_p − x_p‖²₂
- Student discrepancy (ℳᴺ = masked patch 중 timestep이 known-anomaly set 𝒳ᴬ_lab에 속하지
  **않는** patch들):
  ℒ_disc = (1/|ℳᴺ|) Σ_{p∈ℳᴺ} ‖x̂ᵀ_p − x̂ˢ_p‖²₂
- ℒ_grl: "Gradient Reversal Layer를 통해 student decoder로부터 알려진 이상 패턴에 대한
  정보를 제거" — 슬라이드 주석 "(ℒ_grl은 classifier loss)". 구체 수식은 발표에 미기재.

### 각 구성요소의 정당화 논리 (p15–18, 발표 원문 표현)
- **재구성 오차** (p15): "재구성 오차 기반 방법은 labeled 데이터와 unlabeled 데이터를 모두
  동일한 네트워크 및 프로세스에서 **seamless**하게 활용할 수 있음" / contaminated 상황에서
  "전체적인 정상 패턴의 구조에 학습하는 데 집중할 수 있음".
- **MAE/masking** (p16): "입력의 전역적인 의미 구조를 깊이 있게 학습하도록 강제하여 단순한
  local fitting 혹은 copy-pasting 등의 문제를 방지" / "시점 간 구조적 흐름을 자연스럽게 학습" /
  매 학습마다 다른 마스킹 → "매번 다른 맥락의 Augmentation 효과" / "일부 특이 패턴(값이 튀는
  패턴 등)에 과적합하는 것을 방지".
- **Self-distillation** (p17): "Abnormal bias를 방지하기 위한 구성요소". 비대칭성:
  "Teacher는 깊고 강력한 디코더 … Student는 얕고 가벼운 디코더를 통해 teacher의 출력을 모방".
  Contaminated setting 논거: "Teacher가 이상 데이터의 복원에서 흔들릴 경우 student는 일관성
  없는 복원을 모방해야 하므로 더 불안정해짐" / "Student는 모델의 복잡도가 teacher보다 작으므로
  소수만 존재하는 이상에 대해서는 더 큰 불일치를 보이게 됨". Regularization 논거:
  "Reconstruction error는 단순 절대 오차 … discrepancy는 inconsistency를 나타냄" / student는
  teacher의 "internal behavior"를 학습하도록 압력 → "일반적이고 robust한 패턴" 학습.
- **GRL** (p18): "Labeled anomaly는 student가 잘 복원해야 할 대상이 아니라, 복원 단서로
  사용하지 못하게 해야 할 **negative supervision**으로 활용" / "GRL 기반의 adversarial
  branch를 **student decoder에만** 연결하여, labeled anomaly와 unlabeled를 구분하는 정보가
  student 표현에 남지 않도록 제어" / "결과적으로 student는 이상 패턴을 안정적으로 복원하지
  못하게 되며, 정상 구간에서는 일관성을 유지하되 anomaly 구간에서는 teacher와의 discrepancy가
  증폭됨". (DANN의 feature extractor/classifier/domain discriminator 그림 인용.)

### Inference & Anomaly Score (p13–14, 발표 notation)
- 훈련-추론 일관성 논거: "훈련 시에는 … 임의의 지점을 무작위로 마스킹 … 추론 시에도 비슷한
  환경을 만들어줘야 일관성 있는 task가 될 수 있음".
- 절차: 윈도우의 모든 패치를 **순서대로 하나씩 마스킹**하여 patch 개수 N번의 forward pass.
- Patch-level score: s_p = α_rec ẽ^rec_p + α_disc ẽ^disc_p,
  여기서 α_(·)는 μ_recon/μ_disc로 **auto-scaled** 되는 계수
  (μ_recon, μ_disc = 훈련 데이터셋 전체의 reconstruction error 및 discrepancy error 평균).
- Point-level score: "각 time stamp는 여러 관점의 patch-level anomaly score를 갖게 되고,
  이 Anomaly score들을 평균 낸 것이 시점 t의 point-level의 Anomaly score" — stride 있는
  여러 window에서 서로 다른 맥락으로 같은 시점이 복원되므로 "**앙상블 효과**를 기대할 수 있음".

---

## ⑤ Experiments (p19–26, p29–33)

> 아래 수치는 모두 **발표 시점 수치 — 논문에 직접 사용 금지** (논문에는 최신 재실험 결과 사용).

### 데이터셋 (p19)
- SWaT / SWaT(excl22) / WaDi A1 / WaDi A2 / PSM, 5개 벤치마크 (다변량 ICS·서버 시계열).
- 표 수치(발표 시점): SWaT 944,919 pts·51 dim·Test AR 19.05% (excl22 시 3.68%), WaDi A1
  1,382,402 pts·123 dim·3.82%, WaDi A2 957,374 pts·123 dim·3.87%, PSM 220,322 pts·25 dim·30.63%.
- *(r2 추가 — p19 표의 나머지 column 전사; 수치 동결 금지 단서 동일 적용)*:
  **#Training / #Testing** — SWaT 719,959/224,960, SWaT(excl22) 719,959/189,060,
  WaDi A1 1,296,001/86,401, WaDi A2 870,972/86,402, PSM 176,401/43,921.
  **#Anomaly Regions** (test split의 contiguous segment 수) — SWaT 14, excl22 13, WaDi A1 7, A2 7, PSM 29.
  **Train AR (%)** — SWaT 1.63 / WaDi A1 0.52 / A2 0.76 / PSM 6.20 — **contaminated semi-supervised
  프로토콜의 핵심 통계(훈련 데이터 오염도)**.
- **핵심 프로토콜 (발표 원문)**: "본 연구는 train과 test 구분을 **재구성**하여, contaminated
  training data에서 semi-supervised 방법론의 강건성을 검증함 (test 데이터의 전반 50%를 훈련
  데이터로 합침)".
- excl22 근거 (원문): "SWaT 데이터셋은 한 이상 영역의 길이가 지나치게 길어 평가를 왜곡하므로,
  해당 이상 영역을 제외한 metric을 따로 제시함 (excl22)".

### Baseline (p20) — **26개 비교모델**(+Ours=27), 4 카테고리
- **Simple (5)**: Random, Sensor-Range, PCA, L2-Norm, kNN (출처: QuoVadisTAD, ICML 2024)
- **Neural (QuoVadisTAD minimal) (3)**: MLP, MLP-Mixer, Transformer
- **SOTA (14)**: GCN-LSTM, Anomaly Transformer, TranAD, USAD, DAGMM, GDN, OmniAnomaly, TF-MAE,
  NPSR, TimesNet, DCdetector, MEMTO, ModernTCN, CATCH
- **Weakly-Supervised (4)**: DeepMIL, WETAS, TreeMIL, NRDetector
- *(r2 정정)*: r1의 "25종"은 계수 오류 — p20 표 실측 5+3+14+4=**26** 비교모델 + Ours.
  교차 증거: p22/p29–33 rank 첨자가 (27)까지 존재(27개 모델 순위), Notion Page B의
  22 active + 4 weak = 26과 정합. (p34 references가 [1]–[25]인 것과 혼동 추정 — 모델 reference는
  [8]–[25]의 18개이며 모델 수와 무관.)

### 평가 지표 (p21) — 7종, 전부 higher-is-better
- Threshold-dependent: F1, F1_PA, Aff-F1 / Threshold-swept: F1_PA%K /
  Threshold-free: PRC_PA%K, VUS-PR, VUS-ROC.
- 단일 threshold가 필요한 경우 "**anomaly-ratio threshold** (flag the top
  test-anomaly-ratio% of points), not an oracle F1-maximizing threshold" 명시.
- 발표의 지표 해설(원문): "F1_PA : 이상 구간 중 한 point만 이상으로 감지해도 해당 구간 전체를
  맞췄다고 평가, 적절한 metric인지에 대한 많은 challenge가 존재함" 등 — 다중 지표 채택의 근거.
- *(r2 추가 — p21 표의 metric별 reference; Phase 4 공식 소스 재확인 대상)*:

  | Metric | p21 기재 reference (p34 번호) |
  |---|---|
  | F1 | "Standard point-wise F1 (no single canonical origin)" — 단일 출처 없음 명시 |
  | F1_PA | Xu et al., "Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications," WWW 2018 [4] |
  | F1_PA%K / PRC_PA%K | Kim et al., "Towards a Rigorous Evaluation of Time-Series Anomaly Detection," AAAI 2022 [5] |
  | VUS-PR / VUS-ROC | Paparrizos et al., "Volume Under the Surface," PVLDB 2022 [6] |
  | Aff-F1 | Huet et al., "Local Evaluation of Time Series Anomaly Detection Algorithms," KDD 2022 [7] |

### 결과의 정성적 주장 (p22, 발표 원문)
- "제안 모델은 SWaT, SWaT excl22, WaDi A1, WaDi A2 전반에서 최상위 성능을 보이며, 특정
  데이터셋에만 특화된 것이 아니라 다양한 benchmark에서 일관적인 강건성을 보임"
- "Aff-F1과, F1_PA%K는 각각 event-level localization과 segment-level coverage를 상호보완적으로 검증"
- "서로 다른 관점의 지표에서 고르게 높은 순위를 기록하여, … event localization, segment
  coverage, score ranking 측면을 모두 안정적으로 만족함"
- PSM 약점 해석(원문): "PSM에서는 F1_PA%K와 Aff-F1 기준으로 일부 모델 대비 근소하게 낮은
  성능을 보이는데, 이는 제안 모델의 window가 비교 모델보다 5배 이상 커서 PSM의 상대적으로
  짧고 국소적인 anomaly segment를 촘촘하게 커버하는 데 불리하게 작용한 것으로 해석할 수 있음"

### 대표 수치 (발표 시점 수치, 논문에 직접 사용 금지)
- 메인 표(p22) Ours (F1_PA%K / Aff-F1): SWaT 0.9072₍₂₎/0.8741₍₁₎, SWaT excl22 0.6297₍₁₎/0.9075₍₁₎,
  WaDi A1 0.8503₍₁₎/0.9165₍₁₎, WaDi A2 0.7947₍₁₎/0.8648₍₁₎, PSM 0.8055₍₂₎/0.8137₍₂₎.
  종합 Avg Rank **1.25** (비교군 최저 3.81=MLP-Mixer), Worst Rank 3, Rank Std 0.56.
- Appendix full 표(p29–33): 7개 지표 전부에서 데이터셋별 Avg Rank — SWaT Full 1.20,
  SWaT excl22 1.00, WaDi A1 1.00, WaDi A2 1.00, PSM 1.80 (PSM에선 MLP 1.80과 동률 수준,
  F1_PA·F1_PA%K·PRC_PA%K·Aff-F1 일부 2~3위).

### Visualization 주장 (p23–26)
- p23: SWaT score timeline — recon 성분과 scaled_disc(표기 "recon:disc = 4:1") 성분 분해;
  disc 성분이 다수 이상 구간에서 또렷한 스파이크.
- p24: 학습 곡선 — **warm-up 250 epochs는 teacher-only**, 이후 student 합류; "student joins"
  시점 이후 pak_auc_f1 − teacher_pak_auc_f1 차이가 +0.1 내외로 상승 (student/discrepancy의
  기여를 보여주는 self-comparison).
- p25: Normal/Disturbing/Anomaly 카테고리별 score 기여 분석 — Anomaly에서 disc 기여 비중이
  더 큼 (53.9% vs Normal 48.3%; 발표 시점 수치).
- p26: GRL loss·기여율·classifier accuracy 추이 — student 합류 직후 GRL 기여가 치솟았다
  감소하며 classifier가 균형점(≈0.5 balanced acc)으로 수렴.

---

## ⑥ Conclusion + 발표가 강조한 contribution/novelty (p27)

> 아래는 발표 원문 표현 그대로. **각 항목이 논문의 공식 contribution이 될지는 Phase 3 판단 사안.**

1. "**Contaminated semi-supervised setting**에서 시계열 이상 탐지를 수행하는 문제를 다룸"
2. "Unlabeled data의 풍부한 구조 정보와 소수 labeled anomaly의 **방향성 있는 감독 신호**를 함께 활용함"
3. "MAE 기반 masking reconstruction을 통해 단순한 값 복원이 아닌 **시계열의 전역적 맥락과 변수 간 구조**를 학습함"
4. "Teacher-student self-distillation을 통해 reconstruction error를 넘어, **모델 간 inconsistency를 anomaly signal**로 활용함"
5. "Teacher는 **안정적인 복원 기준**으로 유지하고, student는 **정상 패턴에 대해서만 teacher의 behavior를 모방**하도록 유도함"
6. "GRL은 student decoder에서만 **known anomaly-specific information을 제거**하여, 이상 패턴이 student의 복원 경로에 흡수되는 것을 억제함"
7. "따라서 정상 구간에서는 낮은 discrepancy를, **anomaly 구간에서는 의도적으로 큰 discrepancy를 형성**하여 anomaly score의 분리도를 향상시킴"

발표 제목이 명시하는 novelty 축: **"비대칭적 학습(asymmetric learning)"** — (a) teacher/student
용량 비대칭, (b) 학습 신호 비대칭(teacher는 전체 masked patch 복원, student는 known-anomaly 제외
patch만 모방), (c) GRL이 student에만 부착되는 구조 비대칭. (이 '비대칭' 프레이밍의 채택 여부도
Phase 3 판단 사안.)

---

## ⑦ 논문 작성 시 참고할 논리 전개 vs 주의할 점

### 참고할 만한 논리 전개 (잘 작동하는 흐름)
1. **문제 동기의 3단 깔때기**: 이상 탐지 일반의 난점(불균형·라벨부재) → 라벨링 비용 비대칭
   (데이터 양 vs 정보 양) → contaminated + few labeled anomaly = semi-supervised가 현실적 —
   이 흐름은 Introduction 동기 부여로 자연스럽다.
2. **Background를 2개 축으로 최소화**: MAE(표현학습 기반) → SD-MAE video(teacher-student
   discrepancy) → "시계열+준지도로 이식" — related work 과잉 없이 기여점이 또렷해지는 구성.
3. **구성요소별 정당화 슬라이드(p15–18)의 논거**: recon의 seamless 활용성, masking의
   augmentation/과적합 방지, student 저용량의 anomaly 민감성, GRL의 negative supervision —
   Method 절의 design rationale 단락 소재로 유용.
4. **평가 방법론의 방어적 설계**: anomaly-ratio threshold(oracle 아님) 명시, 7개 지표 다각화,
   F1_PA 한계 인지, excl22 별도 제시, contaminated train 재구성 프로토콜 — 리뷰어 선제 방어
   논리로 그대로 살릴 가치가 있다.
5. **약점의 정직한 해석**(PSM, window 크기) — limitation 서술 템플릿.

### 주의할 점 (그대로 따르면 안 되는 부분)
1. **[R2] 발표 구성은 청중용 압축본** — Related Work 부재(배경 2편뿐), ablation 표 부재
   (시각화 기반 정성 논증만), ℒ_grl 수식 미기재, 하이퍼파라미터(warm-up 250ep, window 크기,
   patch 크기, recon:disc=4:1 등)가 그림 캡션에만 산재. 논문에서는 전부 정식 정의·표가 필요.
2. **[R5] Notation 비계승**: 𝒳ᴺ_lab류 4분할 표기, x̂ᵀ_p/x̂ˢ_p, ℳ/ℳᴺ, s_p, α_(·), ẽ^rec_p 등은
   발표용 임시 표기다. 논문 notation 체계는 새로 설계한다.
3. **수치 동결 금지**: p19 데이터 통계와 p22/p29–33의 모든 성능 수치는 발표 시점 스냅샷.
   논문에는 최신 재실험(현재 repo 결과)에서 다시 산출한 값만 사용한다.
4. **용어 일관성 미확정**: 발표 내에서 ℒ_disc/L_discrepancy, recon/reconstruction 혼용,
   "Disturbing" 카테고리(p25)는 정의 없이 등장. 논문에서는 용어를 먼저 고정해야 한다.
5. **PU learning 슬라이드(p7)는 그림만 있고 논증이 없음** — 논문에서 PU와의 관계를 언급하려면
   별도의 정확한 자리매김(우리 설정은 PN+U? PU의 변형?)이 필요하며, 발표를 근거로 삼을 수 없다.
6. **주장 강도 점검 필요**: *(r2 조정)* p6 원문은 "가장 현실적이고 강력한 해결책**이 될 수
   있음**"으로 이미 hedge되어 있다(r1의 "발표체 단정" 사례 지목은 인용 절단에 기인한 과함).
   다만 논문에서는 여전히 근거(인용·실험) 보강이 필요하며, self-comparison(p24)을 ablation
   근거로 쓰려면 정식 ablation 실험으로 대체해야 한다.

---

## ⑧ REQUEST / FEEDBACK

REQUEST: (→ method-digest / code-digest agent) 발표에 미기재된 ℒ_grl의 정확한 수식·계수(λ 스케줄 포함), warm-up(250ep) 메커니즘, window/patch/stride 크기, α auto-scaling의 정확한 구현(μ_recon/μ_disc 산출 시점·대상)을 코드(`mae_anomaly/scoring.py`, `loss.py`, `config.py`)에서 확정해 주기 바람. 발표 그림 캡션의 "recon:disc = 4:1"과 p14의 auto-scaled α 서술 간 관계(고정비 vs 자동 스케일)도 코드 기준으로 정리 필요.

REQUEST: *(r2 추가, → code-digest / 271truth 라인)* **Patchify 모순 확정** — 발표 p12/15/16은 "Patchify (1D-CNN)", Notion Page 0 exp271은 `patchify_mode='linear'`(Linear embedding, CNN 미사용). 발표 당시 config와 현 baseline 중 무엇을 논문 아키텍처로 쓸지 **코드·실험 ID 기준으로 확정**해 주기 바람 (`mae_anomaly/config.py`/`model.py` + exp271 metadata). 아키텍처 그림·서술에 직결.

FEEDBACK: (→ blueprint/Phase 3) 발표의 결과 표는 F1_PA%K·Aff-F1 중심의 메인 표 + 7지표 full appendix 구조였고 rank 통계(Mean/Worst/Std)를 종합 지표로 썼다. 이 보고 구조 자체는 논문에서도 효과적일 수 있으나, 채택 여부는 venue 분량 제약과 함께 Phase 3에서 판단할 것. PSM 약점(긴 window)이 발표에서 이미 공개적으로 해석된 만큼, 논문 limitation 절에서 선제적으로 다루는 편이 안전하다.

---

## 정정 이력

**r2 (2026-06-10, fixer-3 — `p1_digests_r1.md` 전수 반영, 상세: `p1_digests_fixlog_r2.md`)**
- **PB-1 (BLOCKER)**: ①(p20 행)·⑤(Baseline 절)의 "25종/25개" → **26종(+Ours=27)** 교정 (p20 표 실측 5+3+14+4; rank 첨자 (27); Page B 22+4=26 정합).
- **PM-1**: ④에 Patchify "1D-CNN"(PDF p12/15/16) vs `patchify_mode='linear'`(Notion exp271) 원천 간 모순 명시 + ⑧에 코드 기준 확정 REQUEST 추가.
- **Pm-1**: ② p6 인용의 완화 어미("이 될 수 있음") 복원 + ⑦-6 "발표체 단정" 사례 조정.
- **Pm-2**: ③ "Ristea et al."이 PDF에 없는 외부 지식 귀속임을 표기 (Phase 4 재검증 대상).
- **Pm-3**: ④ 정식화 괄호 주석과 ② PU 맥락의 "label은 𝒳ᴬ_lab만" 서술을 추론으로 정정 ("손실 수식에는 𝒳ᴬ_lab만 등장"이 정확).
- **Pm-4**: ⑤ p19 표의 #Training/#Testing·#Anomaly Regions·**Train AR(%)** column 전사 추가.
- **Pm-5**: ⑤ p21 metric별 reference 표([4] Xu WWW'18, [5] Kim AAAI'22, [6] Paparrizos PVLDB'22, [7] Huet KDD'22) 추가.
