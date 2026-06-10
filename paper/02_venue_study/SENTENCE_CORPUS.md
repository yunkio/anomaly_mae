---
phase: 2
agent: corpus-collector
directives: [T2]
related_directives: [T6(준비), R4(준비)]
last_modified: 2026-06-11
verification: "전 인용 WebFetch로 arXiv HTML/ar5iv 원문에서 추출 (2026-06-11). 추출 경로 특성상 §0.3 신뢰도 노트 참조."
---

> **경고: 이 corpus의 verbatim 문장은 문체 기준 비교 전용 — 본문에 복사·근접 의역 절대 금지 (A2).**
> 아래 모든 따옴표 문장은 타 논문 원문 발췌이다. 우리 논문(TSMAE)에 문장 단위로 재사용하거나 단어 몇 개만 바꿔 의역하는 것은 표절이다. 허용 용도는 단 하나 — **Phase 6 문체 검사(T6/R4)에서 "이 분야의 진짜 논문은 이렇게 쓴다"의 기준 표본**으로 우리 원고 문장과 패턴 수준에서 비교하는 것.

# SENTENCE_CORPUS — 탑티어 시계열 이상탐지 논문 문장 표본

Phase 6 문체 검사(ai-phrasing-detector)의 기준 corpus. 섹션 유형 10종 × 각 5–10문장, 출처 표기 + 관용 패턴 1줄 해설. 부록 A(분야 collocation 목록), 부록 B(AI-생성 티 금지 패턴 초안 — corpus 대비 빈도 관찰 기반).

---

## 0. 수집 개요

### 0.1 논문 로스터 (11편)

| 약칭 | 논문 | Venue | 원문 소스 (fetch 2026-06-11) |
|------|------|-------|------|
| AnomTr | Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy | ICLR 2022 (Spotlight) | ar5iv.labs.arxiv.org/html/2110.02642 + arxiv.org/abs/2110.02642 |
| TranAD | TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data | VLDB 2022 | ar5iv.labs.arxiv.org/html/2201.07284 |
| DCdet | DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection | KDD 2023 | ar5iv.labs.arxiv.org/html/2306.10347 + arxiv.org/abs/2306.10347 |
| GDN | Graph Neural Network-Based Anomaly Detection in Multivariate Time Series | AAAI 2021 | ar5iv.labs.arxiv.org/html/2106.06947 |
| MEMTO | MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection | NeurIPS 2023 | ar5iv.labs.arxiv.org/html/2312.02530 |
| TimesNet | TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis | ICLR 2023 | ar5iv.labs.arxiv.org/html/2210.02186 |
| NRdet | Noise-Resilient Point-wise Anomaly Detection in Time Series Using Weak Segment Labels | KDD 2025 | arxiv.org/html/2501.11959v1 |
| RigorEval | Towards a Rigorous Evaluation of Time-series Anomaly Detection | AAAI 2022 (직접 소스: ojs.aaai.org/index.php/AAAI/article/view/20680 — P1 프로토콜 리뷰에서 검증, C-002 주석 2026-06-11; Phase 4 재검증 대상) | ar5iv.labs.arxiv.org/html/2109.05257 |
| TS2Vec | TS2Vec: Towards Universal Representation of Time Series | AAAI 2022 | ar5iv.labs.arxiv.org/html/2106.10466 |
| MAE | Masked Autoencoders Are Scalable Vision Learners | CVPR 2022 | ar5iv.labs.arxiv.org/html/2111.06377 |
| SDMAE | Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors | CVPR 2024 | arxiv.org/html/2306.12041v2 |

구성: 시계열 이상탐지 7편(AnomTr, TranAD, DCdet, GDN, MEMTO, NRdet, RigorEval) + 시계열 일반/self-supervised 2편(TimesNet, TS2Vec) + MAE 계열 인접분야 2편(MAE, SDMAE). NRdet·SDMAE는 Phase 2 anchor dossier 대상 논문과 동일 — 문체 표본은 여기, 구조·논리 분석은 각 dossier 참조.

### 0.2 표기 규약

- 인용문은 `"…"` 안에 fetch된 원문 그대로. 수식 기호는 HTML 추출 한계로 단순화 표기(`X = {x1, …, xN}`, `x_t ∈ R^d`)했고, ar5iv 렌더링 잔재(중복 퍼센트 등)는 제거했다 — 문장 텍스트 자체는 무수정.
- 출처 표기: `(약칭, 섹션)`.
- 각 문장 아래 `→` 줄: 그 문장이 보여주는 관용 패턴 1줄 해설.

### 0.3 신뢰도 노트 (Phase 6 사용자 필독)

추출은 WebFetch(소형 모델 경유 transcription)로 수행 → 미세한 표기 오차 가능성이 0이 아니다.
- **고신뢰(전문 대조 완료)**: AnomTr abstract, DCdet abstract(각 arXiv abs 페이지에서 전문 재현), AnomTr ablation 수치(2차 fetch 교차 확인: "18.34% (76.62→94.96)"), MAE abstract(주지의 원문 표현과 일치).
- **표준 신뢰**: 나머지 인용. 문체 비교 용도로는 충분하나, **수치·고유명이 포함된 인용을 다른 Phase에서 사실 근거로 쓰려면 원문 재확인 필수** (이 문서는 문체 corpus이지 reference card가 아님 — 사실 인용은 Phase 4 library를 거칠 것).
- TranAD §7-4의 "mutlivariate"는 fetch 결과 그대로(원문 오탈자 여부 미확인).

---

## §1. Abstract (10문장)

1. "Unsupervised detection of anomaly points in time series is a challenging problem, which requires the model to derive a distinguishable criterion." (AnomTr, Abstract)
   → 첫 문장에서 task 정의 + 핵심 난점을 한 문장으로 압축; "a challenging problem, which requires …" 구문.
2. "Previous methods tackle the problem mainly through learning pointwise representation or pairwise association, however, neither is sufficient to reason about the intricate dynamics." (AnomTr, Abstract)
   → 선행연구를 2갈래로 분류한 뒤 "neither is sufficient to …"로 공통 한계를 일축.
3. "Our key observation is that due to the rarity of anomalies, it is extremely difficult to build nontrivial associations from abnormal points to the whole series, thereby, the anomalies' associations shall mainly concentrate on their adjacent time points." (AnomTr, Abstract)
   → "Our key observation is that …" — 방법의 출발점인 통찰을 명시적으로 선언하는 관용구.
4. "Technically, we propose the Anomaly Transformer with a new Anomaly-Attention mechanism to compute the association discrepancy." (AnomTr, Abstract)
   → 직관 서술에서 기술 서술로 넘어가는 전환사 "Technically, we propose …".
5. "The Anomaly Transformer achieves state-of-the-art results on six unsupervised time series anomaly detection benchmarks of three applications: service monitoring, space & earth exploration, and water treatment." (AnomTr, Abstract)
   → abstract 마지막 문장 정형: SOTA 주장 + 벤치마크 개수 + 응용 도메인 열거.
6. "Time series anomaly detection is critical for a wide range of applications. It aims to identify deviant samples from the normal sample distribution in time series." (DCdet, Abstract)
   → 도입 2문장 정형: 중요성 한 줄("is critical for a wide range of applications") + task 정의("It aims to identify …").
7. "Reconstruction-based methods still dominate, but the representation learning with anomalies might hurt the performance with its large abnormal loss." (DCdet, Abstract)
   → 지배적 패러다임 인정("still dominate") + hedge 동반 한계 지적("might hurt").
8. "Detecting anomalies in real-world multivariate time series data is challenging due to complex temporal dependencies and inter-variable correlations." (MEMTO, Abstract)
   → 난점 두 가지를 "due to A and B" 병렬로 명시 — TSAD abstract 1문장의 또 다른 정형.
9. "However, these methods still suffer from an over-generalization issue and fail to deliver consistently high performance." (MEMTO, Abstract)
   → 선행 한계 관용 패턴 "suffer from … and fail to …".
10. "This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels." (MAE, Abstract)
    → 단언형 첫 문장 + "Our approach is simple:" 콜론 구문으로 방법을 한 줄 요약 — 자신감 있는 미니멀 스타일.

보조 표본: "Efficient anomaly detection and diagnosis in multivariate time-series data is of great importance for modern industrial applications." (TranAD, Abstract — "is of great importance for" 중요성 관용구) / "In recent years, proposed studies on time-series anomaly detection (TAD) report high F1 scores on benchmark TAD datasets, giving the impression of clear improvements in TAD. However, most studies apply a peculiar evaluation protocol called point adjustment (PA) before scoring." (RigorEval, Abstract — 통념 제시 후 "However"로 뒤집는 비판 논문형 도입).

---

## §2. Intro 도입·문제 제기 (10문장)

1. "Real-world systems always work in a continuous way, which can generate several successive measurements monitored by multi-sensors, such as industrial equipment, space probe, etc." (AnomTr, §1)
   → 현실 시스템 동기 → 데이터 발생 구조로 자연스럽게 유도; "such as" 예시 열거.
2. "But anomalies are usually rare and hidden by vast normal points, making the data labeling hard and expensive. Thus, we focus on time series anomaly detection under the unsupervised setting." (AnomTr, §1)
   → 라벨 부족 난점 → "Thus, we focus on … under the unsupervised setting"으로 문제 세팅 선언. funnel 구조의 종착 문장.
3. "Time series anomaly detection is widely used in real-world applications, including but not limited to industrial equipment status monitoring, financial fraud detection, fault diagnosis, and daily monitoring and maintenance of automobiles." (DCdet, §1)
   → 응용 도메인 4개 내외 열거로 광범위성 입증; "including but not limited to".
4. "Effectively discovering abnormal patterns in systems is crucial to ensure security and avoid economic losses." (DCdet, §1)
   → 중요성 근거 정형: "is crucial to ensure security and avoid economic losses" (안전+경제 손실 페어).
5. "With the rapid growth in interconnected devices and sensors in Cyber-Physical Systems (CPS) such as vehicles, industrial systems and data centres, there is an increasing need to monitor these devices to secure them against attacks." (GDN, §1)
   → "With the rapid growth in …, there is an increasing need to …" — 추세 → 필요 도출 구문.
6. "For instance, in a water treatment plant, there can be numerous sensors measuring water level, flow rates, water quality, valve status, and so on, in each of their many components." (GDN, §1)
   → "For instance"로 구체 시나리오 1개를 깊게 — 추상적 동기를 손에 잡히게 만드는 장치.
7. "Anomalies resulting from malfunctions in critical infrastructures, such as water treatment facilities and space probes, can incur fatal property loss." (MEMTO, §1)
   → 위험 서술 동사 "incur" + 도메인 표준 예시(수처리/우주 탐사선 = SWaT/SMAP 계열 암시).
8. "Time series anomaly detection (TSAD) plays a critical role in many real-world monitoring systems and applications, such as robot-assisted systems, space exploration, and cloud computing. It is the task of discerning unusual or anomalous samples in time series data." (NRdet, §1)
   → "plays a critical role in" + 두 번째 문장에서 task 정의로 줌인; 예시마다 인용 부착이 원문 관행.
9. "Despite the growing interest, video anomaly detection remains a complex task, owing its complexity to the fact that abnormal situations are context-dependent and do not occur very often. This makes it very difficult to collect a representative set of abnormal events for training." (SDMAE, §1)
   → "Despite the growing interest, X remains a complex task" + 난점의 인과 전개("This makes it very difficult to …").
10. "As Industry 4.0 accelerates system automation, consequences of system failures can have a significant social impact." (RigorEval, §1)
    → "As X accelerates …, consequences … can have …" — 시대 배경 1문장 도입(절제된 버전; landscape류 과장 없음).

---

## §3. Intro 기여 서술 (9문장)

1. "Based on the key observation of Association Discrepancy, we propose the Anomaly Transformer with an Anomaly-Attention mechanism, which can model the prior-association and series-associations simultaneously to embody the Association Discrepancy." (AnomTr, §1 contributions)
   → 기여 bullet 정형 "Based on …, we propose X with Y, which can …" — 관찰→설계 인과를 한 문장에.
2. "We propose a minimax strategy to amplify the normal-abnormal distinguishability of the Association Discrepancy and further derive a new association-based detection criterion." (AnomTr, §1 contributions)
   → "We propose … and further derive …" — 기법 + 파생 산출물 연결.
3. "A contrastive learning-based dual-branch attention structure is designed to learn a permutation invariant representation that enlarges the representation differences between normal points and anomalies." (DCdet, §1 contributions)
   → 수동태 기여 bullet("… is designed to learn …") — 능동 "We propose"와 병용되는 변형.
4. "DCdetector achieves performance comparable or superior to state-of-the-art methods on seven multivariate and one univariate time series anomaly detection benchmark datasets." (DCdet, §1 contributions)
   → 실험 기여 bullet 정형: "comparable or superior to" 절제형 주장 + 벤치마크 구성 명시.
5. "We propose GDN, a novel attention-based graph neural network approach which learns a graph of the dependence relationships between sensors, and identifies and explains deviations from these relationships." (GDN, §1 contributions)
   → "We propose X, a novel … approach which …" — 이름+계열+기능 3요소 한 문장.
6. "Our proposed MEMTO is the first multivariate time series anomaly detection method that uses the Gated memory module, which adjusts to diverse normal patterns in a data-driven manner." (MEMTO, §1 contributions)
   → 최초성 주장 정형 "is the first … method that …" (범위를 좁혀 방어 가능하게).
7. "We focus on a novel and practical scenario in TSAD, where abnormal labels are limited and coarse-grained, indicating a time range rather than an exact time point due to challenges like labeling ambiguity or imprecise event timing." (NRdet, §1 contributions)
   → 문제 세팅 자체를 기여로 제시하는 패턴 — "We focus on a novel and practical scenario, where …" (PU/약지도 세팅 정당화에 직접 참고).
8. "To handle the information gap between noisy segment-level labels and missing point-level labels, we propose a new loss function in a contrastive manner, encouraging the smoothness of consecutive points and the separability of points from segments with different labels." (NRdet, §1 contributions)
   → "To handle …, we propose …, encouraging A and B" — 목적 선행 + 분사구 효과 서술.
9. "We propose the TimesNet with TimesBlock to discover multiple periods and capture temporal 2D-variations from transformed 2D tensors by a parameter-efficient inception block." (TimesNet, §1 contributions)
   → 모델+모듈 동시 명명 "the X with Y to [동사1] and [동사2]".

---

## §4. Related Work 포지셔닝 (8문장)

1. "This paper is characterized by a new association-based criterion. Different from the random walk and subsequence-based methods, our criterion is embodied by a co-design of the temporal models for learning more informative time-point associations." (AnomTr, §2)
   → related work 말미의 자기 위치 선언: "Different from …, our …" 차별화 정형.
2. "However, recurrent models like LSTMs are known to be slow and computationally expensive (Audibert et al., 2020)." (TranAD, §2)
   → 선행 계열 한계 지적 시 "are known to be …" + 비판의 출처를 제3자 인용에 귀속(자기 주장화 회피).
3. "Thus, we use transformers to grow the temporal context information sent to an anomaly detector without significantly increasing the computational overheads." (TranAD, §2)
   → 한계 나열 직후 "Thus, we use … without …" — 한계→본 연구 설계 선택으로 즉시 연결.
4. "Reconstruction-based methods learn a model to reconstruct normal samples, and thereby the instances failing be reconstructed by the learned model are anomalies." (DCdet, §2)
   → 패러다임을 1문장으로 정의하는 교과서식 서술(분류 체계 소개의 단위 문장).
5. "Recently, contrastive representative learning has attracted attention due to its diverse design and outstanding performance in downstream tasks in the computer vision field." (DCdet, §2)
   → 인접 분야 동향 수입 정형 "Recently, X has attracted attention due to …".
6. "Classical methods include density-based approaches, linear-model based approaches, distance-based methods, classification models, detector ensembles and many others." (GDN, §2; 원문은 계열마다 인용 부착)
   → 고전 계열 일괄 정리 "Classical methods include …-based approaches, …" — 각 계열에 대표 인용 1개씩 붙이는 관행.
7. "In recent years, graph neural networks (GNNs) have emerged as successful approaches for modelling complex patterns in graph-structured data." (GDN, §2)
   → 부상 기술 도입 정형 "have emerged as successful approaches for …".
8. "One of the primary types of recent deep methods is a reconstruction-based method, which uses an encoder-decoder architecture trained by a self-supervised pretext task of reconstructing input. This approach expects accurate reconstruction for normal samples and high reconstruction errors for anomalies." (MEMTO, §2)
   → 재구성 기반 AD의 표준 가정 서술 — "expects accurate reconstruction for normal samples and high reconstruction errors for anomalies" (TSMAE related work에서 같은 가정을 말할 때의 기준 표현).

---

## §5. Method 도입·notation 도입 (6문장)

1. "Suppose monitoring a successive system of d measurements and recording the equally spaced observations over time. The observed time series X is denoted by a set of time points {x1, x2, …, xN}, where x_t ∈ R^d represents the observation of time t." (AnomTr, §3)
   → "Suppose …" 도입 + "is denoted by …, where … represents …" notation 정형.
2. "Consider a multivariate time-series sequence of length T: X = (x1, x2, …, xT), where each data point x_t ∈ R^d is acquired at a certain timestamp t from industrial sensors or machines." (DCdet, §3)
   → "Consider …" 변형; notation에 물리적 출처("from industrial sensors")를 한 구로 결합.
3. "Given a training input time-series T, for any unseen test time-series T̂ of length T̂, and same modality as the training series, we need to predict Y = {y1, …, yT̂}." (TranAD, §2 problem formulation)
   → "Given …, we need to predict …" — train/test 분리와 출력 정의를 한 문장에.
4. "Our goal is to detect anomalies in testing data, which comes from the same N sensors but over a separate set of T_test time ticks." (GDN, §3)
   → 문제 진술 정형 "Our goal is to detect anomalies in …".
5. "Our GDN method aims to learn relationships between sensors as a graph, and then identifies and explains deviations from the learned patterns." (GDN, §4 overview)
   → method 섹션 첫머리 1문장 로드맵 — 이후 서브섹션들이 이 문장을 분해.
6. "Given a set of time series X = {x1, x2, …, xN} of N instances, the goal is to learn a nonlinear embedding function fθ that maps each xi to its representation ri." (TS2Vec, §Method)
   → 표현학습형 문제 정의 "the goal is to learn a … function fθ that maps …".

---

## §6. Method component 서술 (10문장)

1. "Note that the single-branch self-attention mechanism cannot model the prior-association and series-association simultaneously. We propose the Anomaly-Attention with a two-branch structure." (AnomTr, §3)
   → 컴포넌트 도입 직전 기존 구조의 불충분성 1문장("Note that … cannot …") → 제안.
2. "For the prior-association, we adopt a learnable Gaussian kernel to calculate the prior with respect to the relative temporal distance." (AnomTr, §3)
   → "For [부분 문제], we adopt [도구] to [기능]" — 설계 선택 보고 정형.
3. "In the first phase, the model aims to generate an approximate reconstruction of the input window. The deviation from this inference, referred to as the focus score mentioned previously, facilitates the attention network inside the Transformer Encoder." (TranAD, §4)
   → 다단계 절차 서술 "In the first/second phase, …" + 용어 재참조("referred to as …").
4. "In the second phase, we use the reconstruction loss for the first decoder as a focus score." (TranAD, §4)
   → 산출물의 역할 전환을 "use A as B"로 간결하게.
5. "Each channel in the multivariate time series input is considered as a single time series and divided into patches. Each channel shares the same self-attention network, and the representation results are concatenated as the final output." (DCdet, §3)
   → channel independence + patching 서술의 기준 표현 ("is considered as …", "divided into patches", "shares the same …") — TSMAE patchify 서술과 직결.
6. "Unlike existing graph attention mechanisms, our feature extractor incorporates the sensor embedding vectors vi, which characterize the different behaviors of different types of sensors." (GDN, §4)
   → 컴포넌트 수준 차별화 "Unlike existing …, our … incorporates …, which …".
7. "In order to tackle the problem of over-generalization in reconstruction-based models, we introduce a new memory module mechanism that adjusts to diverse normal patterns in a data-driven manner. In this approach, each item stored in the memory module represents the prototypical features of normal data." (MEMTO, §3)
   → "In order to tackle …, we introduce …" + 구성요소의 의미 부여("each item … represents …").
8. "Following ViT, we divide an image into regular non-overlapping patches. Then we sample a subset of patches and mask (i.e., remove) the remaining ones." (MAE, §Approach)
   → "Following [선행연구], we …" 상속 명시 + "(i.e., …)" 조작적 부연. 짧은 단문 연쇄.
9. "Our encoder is a ViT but applied only on visible, unmasked patches." (MAE, §Approach)
   → 극단적 간결 선언형 컴포넌트 정의 — 군더더기 없는 'X is Y but Z' 구조.
10. "Due to the shared encoder, we are able to leverage the reconstruction discrepancy between the teacher and the student with a minimal computational overhead." (SDMAE, §Method)
    → teacher–student discrepancy 활용 서술의 기준 표현 (TSMAE의 self-distillation 서술과 직결); "leverage"의 도메인 내 정상 용례.

보조 표본: "The student branches out from the original architecture after the first transformer block of the teacher decoder, essentially adding only one transformer block." (SDMAE, §Method — 구조 변경의 비용을 "essentially adding only …"로 정량화).

---

## §7. Experiments setup·protocol (10문장)

1. "We extensively evaluate our model on five real-world datasets with ten competitive baselines." (AnomTr, §4)
   → 실험 섹션 1문장 개관 정형: 데이터셋 수 + baseline 수.
2. "Anomaly Transformer contains 3 layers. We set the channel number of hidden states d_model as 512 and the number of heads h as 8. The training process is early stopped within 10 epochs with the batch size of 32." (AnomTr, §4 implementation)
   → 구현 세부 보고 정형: "We set … as …", "early stopped within N epochs with the batch size of B".
3. "if a time point in a certain successive abnormal segment is detected, all anomalies in this abnormal segment are viewed to be correctly detected." (AnomTr, §4 — point adjustment 정의부)
   → PA 프로토콜의 도메인 표준 기술 — 우리 논문에서 PA를 정의/언급할 때의 내용 기준점 (RigorEval §8-8과 쌍으로 사용).
4. "We compare TranAD with state-of-the-art models for mutlivariate time-series anomaly detection, including MERLIN, LSTM-NDT, DAGMM, OmniAnomaly, MSCRED, MAD-GAN, USAD, MTAD-GAT, CAE-M and GDN." (TranAD, §6)
   → baseline 전수 열거 정형 "We compare X with …, including [목록]". ("mutlivariate"는 fetch 표기 그대로 — §0.3 참조)
5. "To train TranAD, we divide the training time-series into 80% training data and 20% validation data. To avoid model over-fitting, we use early-stopping criteria." (TranAD, §6)
   → split·정칙화 보고: "we divide … into N% … and N% …", "To avoid model over-fitting, we use …".
6. "The datasets contain two weeks of data from normal operations, which are used as training data for the respective models." (GDN, §5)
   → 데이터 구성 설명에 학습 가용 부분(정상 구간)을 명시하는 정형.
7. "We use precision (Prec), recall (Rec) and F1-Score (F1) over the test dataset." (GDN, §5)
   → 평가지표 선언 최소 정형 "We use precision, recall and F1-Score over the test dataset".
8. "We generate sub-series by applying a non-overlapped sliding window with a length of 100 to obtain fixed-length inputs for each dataset." (MEMTO, §4)
   → sliding window 전처리 서술 기준 표현 — "by applying a … sliding window with a length of L to obtain fixed-length inputs".
9. "All experiments are repeated three times, implemented in PyTorch and conducted on a single NVIDIA TITAN RTX 24GB GPU." (TimesNet, §4)
   → 재현성 보고 정형: 반복 횟수 + 프레임워크 + 하드웨어 한 문장.
10. "We adopt various evaluation metrics for comprehensive comparison, including the commonly used F1 score using both segment-level and point-level ground truth, as well as affiliation precision/recall and Volume under the surface metrics." (NRdet, §Experiments)
    → 다중 지표 채택 정당화 "We adopt various evaluation metrics …, including …, as well as …" (R28의 다중 지표 보고와 직결).

보조 표본: "We then follow the same protocol as T-Loss where an SVM classifier with RBF kernel is trained on top of the instance-level representations." (TS2Vec, §Experiments — 선행 프로토콜 상속 명시 "follow the same protocol as X").

---

## §8. 결과 해석·비교 우위 서술 (9문장)

1. "Anomaly Transformer achieves the consistent state-of-the-art on all benchmarks. The results verify the effectiveness of association learning in time series anomaly detection." (AnomTr, §4)
   → 결과 요약 → "The results verify the effectiveness of [설계 원리]" — 수치를 설계 가설 검증으로 되돌리는 정형.
2. "TranAD outperforms the baselines (in terms of F1 score) for all datasets except MSL when we consider the complete dataset for model training." (TranAD, §6)
   → 우위 주장에 예외("except MSL")와 조건("when we consider …")을 명시하는 정직 서술.
3. "Specifically, TranAD achieves improvement of up to 17.06% in F1 score, 14.64% in F1* score, 11.69% in AUC and 11.06% in AUC* scores over the state-of-the-art baseline models." (TranAD, §6)
   → "Specifically, … achieves improvement of up to N% in [지표] … over …" — 지표별 정량 우위 열거.
4. "DCdetector performs better or at least comparable with the Anomaly Transformer in most metrics." (DCdet, §4)
   → 절제된 비교 주장 "performs better or at least comparable with … in most metrics" — 전승 주장 회피.
5. "On WADI, it has 54% higher F-measure than the next best baseline." (GDN, §5)
   → 데이터셋 단위 정량 우위 "N% higher [지표] than the next best baseline".
6. "MEMTO substantially improves the average F1-score on benchmarks compared to the previous state-of-the-art model, Anomaly Transformer, from 93.62% to 95.74%." (MEMTO, §4)
   → "substantially improves … compared to …, from A to B" — 비교 대상 실명 + 절대 수치 페어.
7. "Our proposed NRdetector achieves the best results under the pure F1 score and PA%K F1 score on all benchmark datasets." (NRdet, §Experiments)
   → 지표를 한정한 최고 성능 주장 "achieves the best results under [지표] on all benchmark datasets".
8. "However, when applying the PA protocol, Case 1 appears to yield the state-of-the-art F1PA far beyond the existing methods, except for SMD." (RigorEval, §Experiments; Case 1 = 무작위 anomaly score)
   → 비판적 결과 해석 — "appears to yield"로 표면적 결과와 실제 능력을 분리하는 hedge.
9. "Our method obtains a micro AUC score of 91.3% on Avenue, being only 1.9% below the state-of-the-art object-centric method." (SDMAE, §Experiments)
   → 열세 결과의 공정한 프레이밍 "being only N% below …" — trade-off(속도) 논증의 발판.

---

## §9. Ablation 서술 (8문장)

1. "Each module of our design is effective and necessary." (AnomTr, §4 ablation)
   → ablation 결론 1문장 정형 — "effective and necessary".
2. "our proposed Anomaly Transformer surpasses the pure Transformer by 18.34% (76.62→94.96) absolute improvement." (AnomTr, §4 ablation; 2차 fetch 교차 확인)
   → 기준 모델 대비 "surpasses … by N% (A→B) absolute improvement" — 괄호 안 전후 수치 병기.
3. "Replacing the transformer-based encoder-decoder has the highest performance drop of nearly 11% in terms of the F1 score." (TranAD, §6 ablation)
   → 구성요소 제거 효과 서열화 "has the highest performance drop of nearly N% in terms of [지표]".
4. "Not having the meta-learning in the model has little effect to the F1 scores (≈1%); however, it leads to a nearly 12% drop in F1* scores." (TranAD, §6 ablation)
   → 지표에 따라 갈리는 효과의 양면 보고 "has little effect to A; however, it leads to a … drop in B".
5. "With two-stop gradient modules, we can see that DCdetector gains the best performances. If no stop gradient is contained, DCdetector still works and does not fall into a trivial solution." (DCdet, §4 ablation)
   → 제거 조건에서도 견고함을 별도 문장으로("still works and does not fall into a trivial solution").
6. "Removing the attention mechanism degrades the model's performance most in our experiments. These findings suggest that GDN's use of a learned graph structure, sensor embedding, and attention mechanisms all contribute to its accuracy." (GDN, §5 ablation)
   → "Removing X degrades … most" + 종합 해석 "These findings suggest that … all contribute to …".
7. "The optimal ratios are surprisingly high. The ratio of 75% is good for both linear probing and fine-tuning." (MAE, §Experiments masking-ratio ablation)
   → 발견 보고형 ablation — "surprisingly"는 실측의 의외성에 한정해 사용. 단문 연쇄.
8. "Each and every component contributes towards boosting the performance of the vanilla model. Self-distillation gives the highest boost in terms of the micro AUC." (SDMAE, §Experiments ablation)
   → 전 구성요소 기여 확인 + 최대 기여 컴포넌트 지목 "gives the highest boost in terms of [지표]".

---

## §10. Conclusion (8문장)

1. "This paper studies the unsupervised time series anomaly detection problem. Unlike previous methods, we learn the more informative time-point associations by Transformers." (AnomTr, §Conclusion)
   → "This paper studies …" 회고형 개시 + 차별점 1문장 재진술.
2. "This paper proposes a novel algorithm named DCdetector for time-series anomaly detection. We design a contrastive learning-based dual-branch attention structure in DCdetector to learn a permutation invariant representation." (DCdet, §Conclusion)
   → "This paper proposes … named X for [task]" — 결론 첫 문장에서 명칭·과제 재고정.
3. "We present a transformer based anomaly detection model (TranAD) that can detect and diagnose anomalies for multivariate time-series data. The transformer based encoder-decoder allows quick model training and high detection performance." (TranAD, §Conclusion)
   → "We present … that can …" + 핵심 이점을 "allows A and B"로 압축.
4. "In this work, we proposed our Graph Deviation Network (GDN) approach, which learns a graph of relationships between sensors, and detects deviations from these patterns, while incorporating sensor embeddings. Experiments on two real-world sensor datasets showed that GDN outperformed baselines in accuracy, provides an interpretable model, and helps users to localize and understand anomalies." (GDN, §Conclusion)
   → "In this work, we proposed …" + 실험 요약을 과거형으로("Experiments … showed that …").
5. "This paper presents the TimesNet as a task-general foundation model for time series analysis. Motivated by the multi-periodicity, TimesNet can ravel out intricate temporal variations by a modular architecture." (TimesNet, §Conclusion)
   → 모델의 정체성 재정의 1문장 + 동기 회귀("Motivated by …").
6. "This paper proposes a universal representation learning framework for time series, namely TS2Vec, which applies hierarchical contrasting to learn scale-invariant representations within augmented context views." (TS2Vec, §Conclusion)
   → "…, namely X, which …" — 결론에서 명칭 재도입 변형.
7. "In this paper, we showed for the first time that applying PA can severely overestimate a TAD model's capability, which may not reflect the true modeling performance." (RigorEval, §Conclusion)
   → 발견형(비방법론) 논문의 결론 정형 "we showed for the first time that …" + hedge("may not reflect").
8. "Simple algorithms that scale well are the core of deep learning. Self-supervised learning in vision may now be embarking on a similar trajectory as in NLP." (MAE, §Discussion and Conclusion)
   → 전망형 마무리 — 구체 결과를 넘어 분야 궤적을 1–2문장으로. 과장 없이 "may now be"로 hedge.

---

## 부록 A. 분야 관용 표현·collocation 목록

corpus에서 실제 관찰된 결합만 수록 (괄호: 출처 약칭). Phase 6에서 우리 원고의 해당 개념 서술이 이 결합 범위를 벗어나면 flag.

### A.1 anomaly / anomalies 주변
- 동사(탐지): **detect / identify / discern / pinpoint / localize** anomalies (TranAD, DCdet, NRdet, GDN); "**discovering** abnormal patterns" (DCdet); "detect **and diagnose** anomalies" (TranAD)
- 성질 서술: anomalies "**are usually rare and** hidden by vast normal points" (AnomTr); "do not occur very often" (SDMAE); "can **incur** fatal property loss" (MEMTO); "context-dependent" (SDMAE)
- 명사구: "**abnormal segment**" / "successive abnormal segment" (AnomTr); "deviant samples" (DCdet); "anomalous events" (GDN); "abnormal time points" (AnomTr)

### A.2 anomaly score / criterion
- "**derive** a distinguishable **criterion**" (AnomTr); "association-based **detection criterion**" (AnomTr); "anomaly detection **criteria**" (MEMTO); "anomaly **score**" + "focus score" (TranAD); "randomly generated anomaly score" (RigorEval)
- 주의: score는 compute/derive/use 와 결합; "calculate the anomaly score" 도 통용. score가 "tells/reveals/captures" 류 의인화 동사와 결합하는 용례는 corpus에 없음.

### A.3 reconstruction 주변 (TSMAE 핵심 영역)
- "**reconstruct** the input / normal samples / the missing pixels" (MEMTO, DCdet, MAE); "**reconstruction error/loss**" (MEMTO, TranAD); "**reconstruction discrepancy**" (SDMAE); "an approximate reconstruction of the input window" (TranAD)
- 표준 가정문: "expects **accurate reconstruction for normal samples and high reconstruction errors for anomalies**" (MEMTO); "the instances **failing** [to] **be reconstructed** … are anomalies" (DCdet)
- 한계 명명: "**over-generalization issue**" (MEMTO — 재구성 모델이 이상까지 잘 복원하는 문제의 도메인 표준 명칭)
- masking: "mask **random patches** … and reconstruct the missing pixels" (MAE); "non-overlapping patches" (MAE); "divided into patches" (DCdet); "masking a high proportion of the input" (MAE); "masking **ratio**" (MAE)

### A.4 성능 주장
- "achieves (the consistent) **state-of-the-art** (results) on N benchmarks" (AnomTr, DCdet, TimesNet); "**outperforms the baselines** (in terms of F1 score)" (TranAD, GDN); "**surpasses** X by N% absolute improvement" (AnomTr); "performs **better or at least comparable** with" (DCdet); "improvement of **up to** N% in F1 score" (TranAD); "N% higher F-measure than **the next best baseline**" (GDN); "**substantially improves** … from A to B" (MEMTO); "consistently achieves robust results" (NRdet); "achieves the best results **under** [지표]" (NRdet)
- 열세·조건 명시: "for all datasets **except** MSL" (TranAD); "being **only** 1.9% **below** the state-of-the-art" (SDMAE); "in **most** metrics/benchmarks" (DCdet, NRdet)

### A.5 한계·문제 서술
- "**suffer from** [문제] and **fail to** [기대]" (MEMTO); "limit the detection performance" (TranAD); "are known to be slow and computationally expensive" (TranAD); "has a high possibility of **overestimating** the model performance" (RigorEval); "remains a complex task" (SDMAE); "neither is sufficient to" (AnomTr); "extremely challenging due to" (TimesNet)

### A.6 셋업·프로토콜
- notation 개시: "**Given** … / **Consider** … / **Suppose** …" (TranAD, DCdet, AnomTr); "is denoted by …, **where** x_t ∈ R^d represents …" (AnomTr)
- 전처리: "(non-overlapped/overlapping) **sliding window** with a length of L" (MEMTO); "fixed-length inputs" (MEMTO)
- 학습: "early stopped within N epochs with the batch size of B" (AnomTr); "early-stopping criteria" (TranAD); "divide the training time-series into 80% … and 20% validation" (TranAD)
- 평가: "We use **precision, recall and F1-Score** over the test dataset" (GDN); "**point adjustment (PA)**" + "adjustment strategy" (RigorEval, AnomTr); "F1PA" / "pure F1 score" / "PA%K" (RigorEval, NRdet); "affiliation precision/recall" / "Volume under the surface" (NRdet); "follow the same protocol as X" (TS2Vec); "repeated three times" (TimesNet)
- 데이터: "five real-world datasets" (AnomTr); "benchmark datasets" (전반); "two weeks of data from normal operations … used as training data" (GDN)

### A.7 기여·전개 동사 및 담화 표지
- 기여: "we **propose / present / introduce / design / devise / develop**"; "we focus on"; "is the first … that"; "Based on …, we propose"; "To handle/tackle …, we propose/introduce"
- 담화 표지(corpus 고빈도): "**However**", "**Thus**", "**Recently**", "**Specifically**", "**Technically**", "**For instance**", "**Note that**", "**Following** [선행연구]", "**Unlike / Different from** existing …", "**In this work / In this paper**", "Our key observation is that", "These findings suggest that", "The results verify …", "Motivated by …"
- hedge: "might hurt" (DCdet); "may not reflect" (RigorEval); "appears to yield" (RigorEval); "can incur" (MEMTO)

---

## 부록 B. AI-생성 티 금지 패턴 목록 (초안 — Phase 6 ai-phrasing-detector LEDGER 시드)

판정 기준: 본 corpus(11편, 발췌 약 105문장) 내 등장 빈도 관찰. "0회"는 발췌 표본 기준이며 전수 검사가 아님 — Phase 6에서 LEDGER로 승격 시 사례 축적으로 보강할 것. **금지 = 우리 원고에서 발견 시 무조건 rewrite, 자제 = 1회/논문 수준 초과 시 flag, 허용 = 도메인 실사용 확인.**

### B.1 어휘 단위

| 패턴 | corpus 관찰 | 판정 |
|------|------------|------|
| delve / delves into | 0회 | **금지** |
| showcase(s) | 0회 | **금지** |
| underscore(s) / "highlights the importance of" | 0회 | **금지** |
| "plays a **pivotal** role" | 0회 (실제: "plays a **critical/important** role" — NRdet, TS2Vec) | **금지** — critical/important로 대체 |
| "in the realm of" / "landscape" / "ever-evolving" | 0회 | **금지** |
| seamlessly / meticulously / holistic(ally) | 0회 | **금지** |
| "paving the way (for)" / "unlock" / "harness (the power of)" | 0회 | **금지** |
| "a testament to" / boast(s) | 0회 | **금지** |
| "It is worth noting that" | 0회 | **자제** — 실제 논문에 전무하지 않으나 corpus 미등장; 사용한다면 논문 전체 1회 이하, "Note that"(AnomTr 실사용)으로 대체 권장 |
| remarkable / remarkably | 0회 | **자제** — 의외성 보고는 MAE의 "surprisingly"(실측 한정) 패턴으로 |
| vital / imperative / paramount | 0회 | **자제** — crucial/critical/important(corpus 실사용)로 대체 |
| novel | 4회 내외 — 기여 진술·구성요소 명명에 1회성 사용 ("a novel attention-based … approach", "a novel and practical scenario") | **자제** — 기여 문장 한정, 문단마다 반복 금지 |
| comprehensive | 2회 — 모두 실험 비교 맥락 ("for comprehensive evaluations" DCdet, "for comprehensive comparison" NRdet) | **자제** — 실험 비교 맥락 한정 허용, 일반 형용사 남발 금지 |
| significant(ly) | 수회 — 정량 결과·통계 맥락 | **자제** — 수치 근거 동반 시 허용, 무근거 강조 부사 금지 |
| leverage | 1회 (SDMAE — "leverage the reconstruction discrepancy", 기술적 의미) | **허용** — 기술 맥락 한정; "leverage the power of" 류 buzzword 결합 금지 |
| intricate | 2회 (AnomTr "intricate dynamics", TimesNet "intricate temporal variations") | **허용** — 도메인 실사용 확인; 단 남발 주의 |
| crucial / critical | 4회+ | **허용** |
| robust(ness) | 2회+ | **허용** |
| "aims to" | 3회+ | **허용** |

### B.2 구문·담화 단위

| 패턴 | corpus 관찰 | 판정 |
|------|------------|------|
| 형식적 3연 병렬 ("not only A, but also B, and even C" / "efficiency, scalability, and robustness" 식 추상명사 3종 세트) | 0회 — corpus의 열거는 구체물(데이터셋·지표·응용 도메인) 열거뿐 | **금지** — 추상 덕목 3연 병렬은 rewrite |
| 문단 첫머리 전환부사 연쇄 (Moreover → Furthermore → Additionally 반복) | corpus 전환은 However/Thus/Recently/Specifically/Technically 중심; Moreover/Furthermore 드묾 | **자제** — Furthermore/Moreover 합산 1–2회/섹션 상한 가이드 |
| "In conclusion," 으로 결론 개시 | 0회 — 실제 개시는 "This paper studies/presents/proposes …", "In this work, we proposed …", "We present …" (§10 전체) | **금지** |
| 결론·결과의 만능 과장 ("revolutionize", "groundbreaking", "opens up exciting avenues") | 0회 — 전망도 hedge 동반 ("may now be embarking", MAE) | **금지** |
| 무예외 전승 주장 ("outperforms all baselines on all datasets" 무조건문) | corpus는 예외·조건 명시가 정형 ("except MSL", "in most metrics", "under [지표]") | **자제** — 전승 주장은 표가 실제로 전승일 때만 |
| score/모델 의인화 ("the model strives to", "the score tells us") | 0회 — 의도 표현은 "aims to / is designed to / expects" 한정 | **금지** (aims to/designed to 제외) |
| em-dash 남용 (— 로 절 연결 반복) | corpus 발췌 내 거의 없음 (콤마·세미콜론·콜론 사용) | **자제** |
| "It is important to note/emphasize that" | 0회 | **금지** — "Note that"로 대체 |
| 수사 의문문 남발 | 1회 (GDN abstract 첫 문장 "how can we detect …?" — 의도된 1회성 장치) | **자제** — 쓴다면 논문 전체 1회, abstract/intro 한정 |
| 빈 요약문 ("This demonstrates the effectiveness of our approach" 단독) | corpus는 항상 구체 대상 결합 ("The results verify the effectiveness **of association learning in time series anomaly detection**") | **자제** — effectiveness 주장에는 구체 설계 원리·수치를 결합할 것 |

### B.3 corpus가 보여주는 양성(positive) 신호 — 검사 시 기준점

1. **수치 결합**: 우위 주장 문장의 대부분이 구체 수치·비교 대상 실명을 동반 (§8 전체).
2. **예외 정직성**: "except MSL", "being only 1.9% below" — 약점을 숨기지 않고 프레이밍.
3. **절제된 주장**: "comparable or superior", "better or at least comparable", "in most metrics".
4. **단문 허용**: "Our encoder is a ViT but applied only on visible, unmasked patches." — 짧은 선언문이 자연스러움; 모든 문장을 길게 늘이는 것이 오히려 AI 티.
5. **hedge의 위치**: 한계 지적("might hurt")·전망("may now be")에 hedge, 자기 결과 보고에는 단정("achieves", "outperforms").
6. **관찰→설계 인과 명시**: "Our key observation is that …" → "Based on …, we propose …" 사슬.

---

## 사용 지침 (Phase 6 연계)

1. ai-phrasing-detector는 원고의 각 문장을 (i) 해당 섹션 유형의 §1–§10 표본과 패턴 수준 비교, (ii) 부록 B 금지/자제 목록과 대조, (iii) 부록 A collocation 이탈 여부 검사.
2. 본 문서의 verbatim 문장과 원고 문장의 **표면 유사도가 높게 나오는 경우도 flag** (A2 역방향 검사 — corpus 문장이 원고로 새어 들어간 경우).
3. 부록 B는 초안(시드)이다 — Phase 6에서 검출 사례가 쌓이면 LEDGER로 승격·확장할 것.
