---
phase: 2
agent: nrdetector-analyst
directives: [R16, R19, R20]
last_modified: 2026-06-11
revision: r2 (fixer — adversarial review paper/99_reviews/p2_dossiers_r1.md 전수 반영: X-M1 MAJOR + N-m1–N-m5 MINOR; 전 수정은 arXiv HTML 2501.11959v1 원문 재확인 후 적용; fixlog: p2_fixlog_r2.md; 정정 이력은 말미 부록)
source_paper: "Noise-Resilient Point-wise Anomaly Detection in Time Series Using Weak Segment Labels (KDD '25, arXiv:2501.11959)"
verification: "전 인용문 arXiv HTML(2501.11959v1) 원문 대조 완료 (2026-06-11)"
---

# NRdetector Dossier — 실험 구성·인용 처리·PU 정당화·차이점 분석

> **경고: verbatim 발췌는 분석 전용 — 논문 본문으로 복사 금지 (A2)**
> 아래 모든 따옴표 인용은 NRdetector 원문(arXiv 2501.11959v1 HTML)에서 발췌한 것으로, 우리 논문에 문장 단위로 재사용해서는 안 된다. 논리 구조와 서술 전략만 참고할 것.

---

## 0. 서지 정보 (Phase 4 검증 입력)

| 항목 | 내용 |
|---|---|
| 제목 | Noise-Resilient Point-wise Anomaly Detection in Time Series Using Weak Segment Labels |
| 저자 | Yaxuan Wang (UCSC), Hao Cheng (HKBU), Jing Xiong (UCSC), Qingsong Wen (Squirrel AI), Han Jia (RIPED/CNPC), Ruixuan Song (UCSC), Liyuan Zhang (DUT & RIPED/CNPC), Zhaowei Zhu (BIAI/ZJUT & D5 Data), Yang Liu (UCSC) — 9인, 원문 헤더에서 직접 검증 |
| Venue | KDD '25 — Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1 (Toronto, 2025-08-03~07) |
| DOI | 10.1145/3690624.3709257 |
| arXiv | 2501.11959 (2025-01) |
| 공식 코드 | https://github.com/UCSC-REAL/NRdetector (MIT 라이선스), 아카이브 https://doi.org/10.5281/zenodo.14676716 |
| 코드 구성 | `main.py`/`solver.py`/`data_loader.py`/`evaluation.py`, `models/`·`modules/`·`metrics/`, `pretrained_model/` 디렉토리 존재(임베딩 사전학습 분리 구조 방증), 동봉 데이터는 EMG만 |

호칭 주의: 원문 표기는 "NRdetector" (소문자 d). 사용자 directive의 "NRDetector"와 다르므로 본문 인용 시 원문 표기를 따를 것.

---

## 1. 논문 핵심 요약

세그먼트 단위 약한 라벨(weak segment label)만으로 point-level 이상탐지를 수행하는 **2-스테이지 PU learning 프레임워크**.

- **설정**: 양성(이상) 세그먼트 라벨의 **일부(40%)만** 주어지고, 나머지(이상+정상)는 전부 unlabeled. point-level 라벨은 전혀 없음. 이를 "라벨 노이즈" 문제로 재정식화 (unlabeled에 섞인 미검출 이상 = 노이즈 라벨, e₁ = P(Ỹ=0|Y=1) = 0.6).
- **Stage-1 (coarse-grained PU learning)**: WETAS 프레임워크 기반 DiCNN(WaveNet) **사전학습 temporal embedding** → confidence 기반 Sample Selector(cosine similarity로 reliable negative 추출 + KNN 그래프 label propagation) → 6-layer MLP 분류기를 PU Criterion으로 학습.
- **PU Criterion** = PU Loss + TC Loss:
  - PU Loss (Eq.5): R_pu = 2π_P·|mean_{X∈X_L} f(X) − 1| + |mean_{X∈X̄_U} f(X) − π_P| — Dist-PU(Zhao et al. 2022)의 distribution alignment 기반 비용민감형 상계(biased upper bound) 최적화. (4.2.3절은 "based on the Non-negative Risk Estimator (Kiryo et al., 2017)"라고 명시하지만 실제 채택 형태는 distribution-alignment 형식.)
  - TC Loss (Eq.6–8): L_smooth(세그먼트 내 인접 점수 차 제곱합) + L_sep(labeled 양성 세그먼트 평균 점수 > unlabeled 평균 점수), λ₁=λ₂=8×10⁻⁵.
- **Stage-2 (fine-grained point detection)**: 양성 예측 세그먼트 내부 점들을 anomaly score로 정렬 → 가설 비율 k로 pseudo-label 생성 → HOC 노이즈 전이 추정기(Zhu et al. 2021c)로 **training-free 임계값 자동화** → point-level 예측.
- 명시적 **multi-stage, not end-to-end**: "The extracted temporal dependencies will be learned through the pre-trained informative representation model." (§4.1)

---

## 2. PU/semi-supervised 설정 정당화 논리 (임무 3) — verbatim 발췌

NRdetector의 동기화 수순은 4단계 구조다: (a) 비지도의 한계 → (b) 완전 지도의 비현실성 → (c) 약한 라벨의 실용성 → (d) 약한 라벨의 노이즈까지 고려해야 현실적.

**(a) 비지도 한계 — 특히 train 데이터 내 오염 논리 (우리 contaminated 프로토콜과 직결):**
> "However, the performance of these unsupervised learning methods is constrained by the lack of prior knowledge concerning true anomalies (Elaziz et al., 2023). They are not good at finding specific anomalous patterns, especially when the anomalies are embedded within the training data for building the normal patterns." (§1 Introduction)

**(b) point-level 완전 지도의 비현실성 (산업 현장 라벨 비용 논리):**
> "However, labeling every anomalous time point is neither practical nor precise due to the significant time and cost required for accurate identification." (§1)

**(c) 약한(이벤트 발생) 라벨의 실용성:**
> "Acquiring weak labels by simply indicating the occurrence of anomalous events is a more practical approach for real-world applications." (§1)

**(d) 양성 라벨은 신뢰 가능, 미라벨은 미검출 가능 — PU 설정의 현장 논리 (핵심):**
> "In real-world TSAD problems, a positive label can be seen as a true annotation because an observed and recorded abnormal event is often verified. However, the other events may be either normal or abnormal events since the abnormal behavior may be missed." (§1)

**기여 주장에서의 설정 신규성 프레이밍:**
> "We focus on a novel and practical scenario in TSAD, where abnormal labels are limited and coarse-grained, indicating a time range rather than an exact time point due to challenges like labeling ambiguity or imprecise event timing." (§1, contribution 1)

**Related work에서 PU를 SSL의 특수형으로 정의:**
> "Positive and Unlabeled (PU) Learning is a special form of semi-supervised learning (SSL). Compared with traditional SSL, this task is much more challenging due to the absence of any known negative labels." (§2 Positive and Unlabeled learning)

**시사점**: (d)의 "verified positive / possibly-missed unlabeled" 논리는 우리 소수-labeled-anomaly 설정의 동기 서술에 그대로 차용 가능한 *논리 구조*다 (문장 재사용 금지). (a)의 "anomalies embedded within the training data" 논리는 우리 contaminated 프로토콜(test 앞 50% train 편입) 정당화의 선례 인용으로 쓸 수 있다.

---

## 3. 실험 구성 상세 분석 (임무 1, R16)

### 3.1 데이터셋과 split (Table 1 원문 수치)

| 데이터셋 | 출처 | #Train | #Test | 차원 | AR(%) | 양성 세그먼트(TPS) |
|---|---|---|---|---|---|---|
| EMG | 근전도 (Lobov et al. 2018) | 304,400 | 130,900 | 8 | 5.8 | 222 |
| SMD | 인터넷 서버 (Su et al. 2019) | 495,870 | 212,550 | 38 | 4.2 | 463 |
| PSM | eBay 서버 (Abdulaal et al. 2021) | 61,488 | 26,353 | 25 | 27.8 | 191 |
| MSL | NASA (Hundman et al. 2018) | 51,610 | 22,119 | 55 | 10.5 | 99 |
| SMAP | NASA (Hundman et al. 2018) | 299,331 | 128,286 | 25 | 12.8 | 506 |

- 전처리: "Following the pre-processing methods in (Xu et al., 2021) [Anomaly Transformer], we split the dataset into consecutive non-overlapping segments by sliding window." — **비중첩** 윈도, L=100 전 데이터셋 고정.
- split: "We split the set of all segments by 7:3 ratio into training and test sets." — 세그먼트 풀을 7:3으로 나눔 (시계열 보존 holdout 여부는 명시하지 않음). 평균 point-level AR 12.22%.
- 윈도 크기 선택의 정당화 서술 방식이 특징적: "running a sliding window in time series data is widely used in TSAD tasks (Shen et al., 2020) and has little influence on the main design (Yang et al., 2023). Thus, we just set the window size L as 100 for all datasets, unlike TreeMIL (Liu et al., 2024) and WETAS (Lee et al., 2021)." (§5.2 — fixer r2, N-m4: TreeMIL/WETAS 뒤 author-year 괄호 2건의 무표기 탈락 복원) — *설계와 무관한 하이퍼파라미터는 선행 인용으로 간단히 처리하고 넘어가는* 경제적 서술.

### 3.2 라벨 설정 — 우리 라벨 희소화 sweep의 직접 참고

**구성 (verbatim, §5.1):**
> "We construct the positive dataset for training by providing only 40% of the anomalous segment-level labels, resulting in a label noise rate of 0.6 (segment-level). The labels for the remaining segments (both anomalous and normal) are not given to construct the unlabeled dataset. We have no access to the point-level labels."

- **단일 주(main) 라벨 비율(40% → e₁=0.6)로 메인 테이블 전체를 구성**하고, 별도 표(Table 4)에서 e₁ ∈ {0.4, 0.2, 0.0}으로 sweep. 즉 "main 1점 + robustness sweep 3점" 구조.
- e₁=0.0이면 모든 양성 세그먼트가 라벨됨 → "turning the problem into a MIL problem"으로 설정 자체가 다른 문제로 환원됨을 명시 — sweep 극단점의 의미를 문제-정의 수준에서 해석해 주는 서술.
- **sweep 결과 서사 (verbatim, §5.3)**:
> "When there is less noise in the label, NRdetector still outperforms WEATS [sic] and TreeMIL, but the gap is not very large. For example, when the label noise rate is 0.2, the F1 scores on the EMG dataset only decrease by 0.051 and 0.077, respectively. However, in the label noise rate of 0.6 cases, the NRdetector completely outperforms the three methods by at least 0.2 on the EMG dataset."

**우리 sweep 설계에의 시사점**: (i) 라벨 희소율 축을 "노이즈율"로도 해석할 수 있게 이중 정의(우리: labeled-anomaly 비율 p ↔ 그들: e₁=1−p)하면 PU 문헌과 접속된다. (ii) main 결과는 단일 희소율로 고정하고 sweep은 별도 표로 — 메인 테이블 비대화를 막는다. (iii) "라벨이 풍부할수록 격차 축소, 희소할수록 격차 확대"라는 robustness 서사를 정량(절대 F1 차)으로 보고하는 패턴, 그리고 극단점(전부 라벨/전무)에서 문제가 어떤 기존 설정으로 환원되는지 명시하는 패턴을 차용할 것.

### 3.3 비교 baseline 구성 — 3계층 + 부록 1계층

메인 테이블(Table 2): "13 competitive baselines"
1. **Unsupervised (6)**: DCdetector, Anomaly Transformer, AutoFormer, FEDformer, TimesNet, One-fits-all — 재구성 기반 point-level score.
2. **Semi-supervised 변형 (4)**: AutoFormer++/FEDformer++/TimesNet++/One-fits-all++ — "These models are trained by using only normal segments... But note that the 'normal segments' are actually the unlabeled segments in our setting." (§5.1 — fixer r2, N-m2: Baselines 단락은 "5.2. Experimental Setting" 헤더 직전에 종료하므로 §5.1 소속; 초판 "§5.2"는 오기) — *기존 비지도 모델을 자기 설정으로 적응시키는 방식을 명시하고, 그 적응이 갖는 정보량의 의미까지 해설* ("know as much label information as our method", §5.3).
3. **Weakly supervised (3)**: DeepMIL, WETAS, TreeMIL. 단, "main baselines" 선언은 3종 전체가 아니라 **WETAS·TreeMIL 2종에만** 부착된다 (fixer r2, N-m2/N-m3 — 초판은 인용을 3종 나열 직후에 붙여 전체 지칭으로 오독 유발) — 원문: "we compare our method with WETAS (Lee et al., 2021) and TreeMIL (Liu et al., 2024), which are the main baselines we need to compare." (§5.1) — 주 경쟁자를 명시적으로 선언하고, 세부 지표 비교(Table 3, 11개 지표)도 이 둘(WETAS/TreeMIL)로만 좁힘.
4. **부록 (Table 7)**: point-level nnPU, Dist-PU, NCAD — 공정성 프레이밍이 정교함:
> "This ensures that the label information known to the point-level PU method is as close as possible to ours, avoiding the situation where knowing too many labels renders the comparison meaningless." (§5.5)
> NCAD에는 point-level 라벨 제공 후: "Note that this kind of labeling actually contains more information than segment-level labeling." (§5.5)

**구성 원리**: 전 계층 비교(Table 2, F1/F1_PA%K만) → 최강 계층과의 정밀 비교(Table 3, 전 지표) → 인접 패러다임과의 정보량-통제 비교(부록). 우리도 "unsupervised 전체 → 주 경쟁(semi/PU 계열) 정밀 비교 → 라벨 정보량 통제 비교"의 깔때기 구조를 쓸 수 있다.

### 3.4 평가지표 (11개) + 채택 정당화 서술

Table 3의 11지표 구성 (캡션·헤더 행 실측 — fixer r2, N-m1 정정): **F1**(point-level, PA 미적용), **P**, **R**, **F1_PA%K**(K 의존 제거 위해 AUC화), **F1_PA**(PA 전략 적용 F1 — 캡션 verbatim: "The F1_PA is the F1 score using the PA strategy"), **Aff-P**, **Aff-R** (Huet et al. 2022), **R_A_R**, **R_A_P**(Range-AUC-ROC/PR), **V_ROC**, **V_PR** (Paparrizos et al. 2022). 초판은 F1_PA를 누락하고 F1-W를 "+보조"로 나열해 11개 구성이 어긋났다 — **F1-W(세그먼트 수준)는 11지표가 아니라 ablation 표(Table 5) 전용 보조 지표**다.

**주의 (Phase 3 차용 시 — N-m1 후속)**: PA는 **main 테이블(Table 2)에서만 배제**되었고, 정밀 비교 표(Table 3)에는 F1_PA가 정식 포함된다. PA 배제 논리를 차용할 때 "전 표에서 PA 미사용"으로 쓰면 원문과 어긋난다 — "main 결과에서 PA 배제 + 멀티지표 표에 참고용 병기" 구조로 정확히 서술할 것.

**PA 배제 정당화 (verbatim, §5.2):**
> "Note that we do not consider the point adjustment (PA) approach ... for the evaluation of all methods in the main table. (Kim et al., 2022) indicates that PA overestimates classifier performance, even though this metric has practical justifications (Xu et al., 2018). Thus, we adopt the optimized PA-based metric, PA%K."
> "Different metrics provide different views for anomaly evaluations."

**시사점**: 우리 평가 셋(VUS/PA%K-AUC/Affiliation)과 거의 동일 — NRdetector를 "동일 평가 철학을 채택한 선행 사례"로 인용하면 지표 선택 정당화가 한 문장으로 끝난다. PA를 main에서 배제하되 "practical justifications" 인정 후 PA%K로 대체하는 양보-반박 구조도 차용 가치 있음.

### 3.5 구현 세부

- 임베딩: WETAS 프레임워크 + DiCNN(WaveNet) 7층, GAP로 세그먼트 임베딩, 사전학습 추출 — **단 "사전학습 후 고정(frozen) 추출"은 INFERENCE (fixer r2, N-m5)**: 원문에 freeze/frozen 명시 없음(전문 grep 0건). "pre-trained" 표현(§4.1) + 공식 코드의 `pretrained_model/` 디렉토리 구조에서의 추론이며, 단정 인용 금지 — 필요 시 공식 코드로 확인 가능 (§6 한계 노트의 7:3 split 처리와 동일 수준) ("the extractor here can be replaced with another temporal feature extractor" — 교체 가능 부품으로 명시, §4.2.1). Transformer로 교체 ablation 존재 (Table 8, DiCNN 소폭 우위).
- 분류기: 6-layer MLP + ReLU + Sigmoid. Adam lr=1e-4, batch 32. baselines는 "suggested hyperparameters reported in the corresponding previous literature"로 구현.
- Ablation: Sample Selector(추출/전파 단독은 해악, 동시 사용만 이득 — Table 5), PU Criterion(BCE 대비 PU Loss 큰 이득 + TC Loss 효과 — Table 6), HOC 임계값 자동화 효과, 구조(DiCNN vs Transformer — Table 8), 하이퍼파라미터(class prior 둔감성, batch size — Fig. 3).

---

## 4. R19 근거 — baseline 인용 처리 방식 (실제 사례 거명)

**핵심 발견: NRdetector의 related work에는 baseline "모델명"이 단 하나도 등장하지 않는다.** (원문 HTML 전수 grep으로 검증)

### 4.1 Related work에서의 처리
§2는 3개 소절(Time Series Anomaly Detection / Learning with Noisy Labels / Positive and Unlabeled learning)이며:

- **비지도 계열은 괄호 인용 클러스터 1개로 일괄 처리**:
> "Extensive research has explored TSAD using deep neural networks in unsupervised settings (Park et al., 2018; Ruff et al., 2018; Shen et al., 2020; Zhang et al., 2022b; Xu et al., 2021, 2018; Wu et al., 2022; Zhou et al., 2023; Sun et al., 2023; Kim et al., 2023; Lai et al., 2024)." (§2)
  - 이 클러스터 안에 Anomaly Transformer(=Xu et al., 2021), TimesNet(=Wu et al., 2022), One-fits-all(=Zhou et al., 2023)이 *익명으로* 포함되지만 개별 논의는 전혀 없음. DCdetector(=Yang et al., 2023)도 일반 명제의 지지 인용으로만 1회 등장.
  - **AutoFormer(Wu et al., 2021)와 FEDformer(Zhou et al., 2022)는 related work에 아예 등장하지 않음** — 실험 섹션이 유일한 인용처.
- **설정을 정의하는 경쟁 계열(weak supervision)만 명제 단위로 논의**: "Weakly supervised approaches (Lee et al., 2021 [WETAS]; Liu et al., 2024 [TreeMIL]; Sultani et al., 2018 [DeepMIL]) optimize models to classify segments accurately by leveraging segment-level labels." — 단, 이들조차 related work에서는 author-year 인용일 뿐 모델명 미표기. 모델명 첫 등장은 §4(WETAS)와 §5 baselines 단락.

### 4.2 실험 섹션에서의 처리 (이름+인용 최초 결합)
> "we compare NRdetector with DCdetector (Yang et al., 2023), Anomaly Transformer (Xu et al., 2021), and some reconstruction-based methods which compute the point-level anomaly scores from the reconstruction of time series, like AutoFormer (Wu et al., 2021), FEDformer (Zhou et al., 2022), TimesNet (Wu et al., 2022), and One-fits-all (Zhou et al., 2023)." (§5.1 Baselines — fixer r2, N-m2: Baselines 단락은 §5.1 소속, 초판 "§5.2"는 오기)

각 baseline에는 카테고리 1문장 + 인용만 부여. 주 경쟁자(WETAS/TreeMIL)만 별도 문장으로 지위를 선언.

### 4.3 우리 논문 적용 규칙 (R19 운영화)
1. **성능 비교 전용 모델**(예: 우리 실험의 비지도 재구성/예측 계열 baseline들)은 related work에 미기재 — 실험 섹션 baselines 단락에서 "이름 (인용)" + 카테고리 1문장이면 충분. NRdetector의 AutoFormer/FEDformer 처리(related work 0회, 실험에서만 인용)가 직접 선례.
2. **설정·문제 정의를 공유하거나 우리 주장에 대립하는 모델**(NRdetector 자신, weak/semi/PU 계열)만 related work에서 명제 단위로 논의.
3. 비지도 일반론이 필요하면 괄호 클러스터 인용 1문장으로 압축 — 개별 모델 소개 문단 불필요.

---

## 5. 차이점 재료 전수 (임무 4, R20)

전제가 되는 본 연구 구조: MAE 기반 단일 모델, **라벨을 표현 학습 자체에 통합**(masking 우선순위, teacher-student discrepancy 방향 학습, GRL suppression), score = 재구성 오차 + discrepancy, contaminated train 프로토콜, VUS/PA%K-AUC/Affiliation 평가.

**라벨 설정 단서 (fixer r2, X-M1 — RESEARCH_SYNTHESIS §② 정본 정합)**: 초판 전제부의 "소수 labeled anomaly + 대량 unlabeled"는 연구의 **설정(가정, R11)**이지 main 실험 구현이 아니다. 정본 3단 구조에 따라: ① **설정(가정)** = 대부분 unlabeled + 소수 labeled anomaly (§②-1); ② **main 271 구현(FACT)** = train 구간의 모든 샘플에 라벨이 존재하는 **label 가용성 상한(upper-bound) 케이스** (§②-2 — "구현상 train 구간의 모든 샘플에 라벨이 존재"); ③ **라벨 희소화 sweep(계획, R32)** = 일부 anomaly가 unlabeled로 잔류하는 일반 케이스의 검증 실험 — 현재 전용 파라미터·스크립트 미구현 (§②-3, EXPERIMENT_PROTOCOL_TRUTH §⑦). "semi-supervised/PU"라는 설정 명명은 **Phase 3 결정 사안**이다 (§②-6 — main 구현은 엄밀한 PU setting이 아니며 "contaminated semi-supervised"에 가까움). 아래 D1–D9의 차이축 자체는 이 명명과 무관하게 성립한다.

| # | 축 | NRdetector | 본 연구 | R20 연결 (차이가 뒷받침하는 주장) |
|---|---|---|---|---|
| D1 | 파이프라인 구조 | **명시적 2-stage 분리형**: 사전학습 임베딩(WETAS/DiCNN, 교체가능 부품; "고정"은 INFERENCE — §3.5, fixer r2 N-m5) → sample selection → MLP PU 분류기 → 사후 point 검출(HOC) | **end-to-end 단일 모델**: 라벨 신호가 표현 학습의 기울기에 직접 개입 | "시계열에서 라벨 정보를 *표현 학습 내부*에 통합한 SSL/PU 연구는 부재 — 기존 유일 계열(NRdetector)조차 표현은 라벨-불가지론적 사전학습에 위임" |
| D2 | 라벨 granularity | 세그먼트 수준 weak label (point 라벨 무접근, "indicating a time range rather than an exact time point") | point/window 수준 anomaly 라벨 소수 | 설정 자체가 다름 → 직접 성능 비교보다 "인접하지만 다른 문제"로 위치 지정 |
| D3 | 라벨 활용 지점 | 라벨은 **분류기 손실**(PU risk)과 **negative 선별**(데이터 큐레이션)에만 사용 | 라벨이 **masking 우선순위·discrepancy 방향·GRL 억제**, 즉 표현이 만들어지는 방식 자체를 결정 | D1과 동일 — 라벨→표현 통합의 신규성 강조 |
| D4 | 노이즈/오염 대응 메커니즘 | **데이터 수준**: confidence 기반 reliable-negative 추출 + label propagation + distribution-alignment PU loss(클래스 prior로 편향 보정) | **기울기 수준**: GRL로 unlabeled 내 의심 구간의 정상-패턴 학습을 억제 | 같은 문제("unlabeled에 이상이 섞임")에 대한 해법 계층이 다름 — sample selection vs representation shaping |
| D5 | 자기지도 신호 | 없음 — 재구성/마스킹 pretext 없이 weak-label 표현(WETAS) 재사용 | MAE 재구성이 backbone 목적함수; 라벨은 그 위에 추가 신호 | "재구성 기반 SSL과 PU 신호의 결합"은 미답 영역이라는 주장 |
| D6 | 이상 점수 | 분류기 출력 f(X)(세그먼트) + 내부 점수 h(X) 랭킹 + HOC **이진화** (임계값 자동 추정이 기여의 일부) | 연속 score = 재구성 오차 + teacher-student discrepancy; 임계값-무관(threshold-free) 지표로 평가 | 점수 구성의 패러다임 차이 (판별 분류 vs 생성-재구성) |
| D7 | 시간 구조 처리 | TC Loss로 점수의 평활성(smoothness)·분리성(separability)을 **손실 제약**으로 부과 | 시간 구조는 패치 단위 MAE 표현이 내재적으로 학습 | 손실-제약 vs 구조-내재 |
| D8 | train 오염 프로토콜 | 7:3 세그먼트 split — train에 이상이 자연 포함되고 그 대부분이 unlabeled (e₁=0.6) | test 앞 50%를 train에 의도적으로 편입하는 contaminated 프로토콜 | 양쪽 모두 "오염된 train에서의 강건성"을 주장 — 동기 인용 가능, 단 구성 방식은 상이 |
| D9 | 라벨 희소화 축 | e₁(노이즈율) sweep {0.6 main, 0.4, 0.2, 0.0}; e₁=0 → MIL로 환원 | labeled-anomaly 희소율 sweep — **R32 계획 단계, 전용 구현·수치 없음** (설계 시 §3.2 패턴 차용; fixer r2, X-M1 단서) | sweep 설계의 선례이자, 축의 의미가 다름(세그먼트 라벨 은닉율 vs 개체 라벨 희소율) |

**R20 서술 전략 정리**:
- Related work의 SSL/PU 소절에서는 PU learning의 일반 목표(negative 부재 하의 분류, 비용민감/샘플선별 양대 계열 — NRdetector §2의 분류 체계 차용 가능)를 충분히 소개하되, **시계열 적용은 극히 드물다**는 점을 강조.
- **주의(정확성)**: NRdetector 자신은 "PU Learning has been employed in the field of anomaly detection ..., including time series (Nguyen et al., 2011; Zhang et al., 2021b)"라고 선행을 인정한다. 따라서 우리의 "거의 없음" 주장은 "**심층 표현 학습과 통합된 PU/SSL 기반 다변량 TSAD는 거의 없으며, 근접 사례인 NRdetector조차 표현 학습과 PU 분류를 분리한다**"로 정밀하게 스코핑해야 검증 공격에 안전하다. NRdetector 스스로 자기 설정을 "a novel and practical scenario in TSAD"로 부른 것도 이 분야 희소성의 방증으로 인용 가능.
- 차이점 우선 서술(공통점 최소화): 공통점은 (i) 소수 양성 라벨 + 대량 unlabeled 동기, (ii) 평가 철학(PA 회피, VUS/affiliation), (iii) train 오염 강건성 주장 — 이 3개만 짧게 인정하고, D1/D3/D5(표현-학습 통합 여부)를 차이의 중심축으로 배치.

---

## 6. 한계·검증 노트

- 모든 verbatim 인용과 수치(Table 1, e₁ sweep, λ 값, baselines 목록, related work 인용 클러스터)는 arXiv HTML v1 원문을 직접 파싱하여 대조 검증함. KDD 최종본(DOI판)과 v1 간 미세 차이는 미확인 — Phase 4에서 ACM판 대조 권장.
- §4.2.3의 "Non-negative Risk Estimator (Kiryo et al., 2017)" 언급과 실제 Eq.5(distribution alignment, biased upper bound) 사이에 표기상 긴장이 있음 — 우리 논문에서 NRdetector의 PU loss를 기술할 때는 "Dist-PU 계열 distribution-alignment 목적함수"로 쓰는 것이 안전.
- 7:3 split이 시간 순서를 보존하는지(앞 70%인지 무작위인지)는 원문에 명시되지 않음 — 인용 시 단정 금지. 필요 시 공식 코드(`data_loader.py`)로 확인 가능.
- DiCNN 임베딩의 "사전학습 후 고정(frozen)" 여부도 동일 수준의 INFERENCE — 원문 freeze/frozen 0건, 코드 구조 방증만 존재 (§3.5, fixer r2 N-m5).
- 원문 오탈자 존재("WEATS", "anomlay") — 인용 시 [sic] 처리.

---

## 부록: 정정 이력

### 2026-06-11 fixer r2 (adversarial review `paper/99_reviews/p2_dossiers_r1.md` 전수 반영; fixlog: `p2_fixlog_r2.md`; 전 항목 arXiv HTML 2501.11959v1 원문 재확인 후 적용)

1. **[X-M1, MAJOR]** §5 전제부 — "소수 labeled anomaly + 대량 unlabeled" 확정 서술을 정본 3단 구조(설정 가정 R11 / main 271 구현 = label 가용성 상한 케이스 / 라벨 희소화 sweep = R32 계획)로 교정 + "semi/PU 명명은 Phase 3 결정 사안" 단서 (RESEARCH_SYNTHESIS §②-1/②-2/②-3/②-6). D9 행에 R32 계획 단계 명시.
2. **[N-m1]** §3.4 — 11지표 구성 정정: **F1_PA 포함**(Table 3 캡션·헤더 실측 "The F1_PA is the F1 score using the PA strategy"), F1-W는 11지표에서 제외(ablation Table 5 전용). "PA는 main 테이블에서만 배제" 과장 방지 주의 추가.
3. **[N-m2]** 인용 3건 귀속 교정 — baseline 목록 인용(§4.2)·semi-supervised ++변형 인용(§3.3)·"main baselines" 인용(§3.3)을 §5.2 → **§5.1** (Baselines 단락은 "5.2. Experimental Setting" 헤더 직전 종료, 원문 실측).
4. **[N-m3]** §3.3 항목 3 — "main baselines" 인용을 DeepMIL 포함 3종 나열 직후가 아니라 **WETAS·TreeMIL 2종**에 정확히 부착 (원문: "we compare our method with WETAS … and TreeMIL …, which are the main baselines").
5. **[N-m4]** §3.1 window-size 인용 — "unlike TreeMIL (Liu et al., 2024) and WETAS (Lee et al., 2021)"의 author-year 괄호 2건 무표기 탈락 복원.
6. **[N-m5]** §3.5/§5 D1/§6 — "사전학습 후 고정 추출" 단정을 INFERENCE로 강등 (원문 freeze/frozen 전문 grep 0건; "pre-trained" + 코드 `pretrained_model/` 구조 방증만 존재).
