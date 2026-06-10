---
phase: 4
agent: excerpt-curator-1
directives: [T4]
last_modified: 2026-06-11
key: wang2025nrdetector
verification_status: VERIFIED_A
verified_by_A: source-verifier-A2
verified_note: "Authors confirmed from arXiv PDF pdftotext. Abstract verbatim confirmed. DOI confirmed. Pages: VERIFY_REQUIRED (ACM DL 403, DBLP not yet indexed for KDD 2025). Card excerpts from §1 and §3/5 verified from arXiv preprint."
card_grade: FULL
---
# Noise-Resilient Point-wise Anomaly Detection in Time Series Using Weak Segment Labels

**경고: 이 card의 verbatim 발췌·abstract는 검증/문체 대조 전용 — 논문 본문으로 복사·근접 의역 절대 금지 (A2)**

## 서지 (scout 예비 — 검증 전)
저자: Yaxuan Wang, Hao Cheng, Jing Xiong, Qingsong Wen, Han Jia, Ruixuan Song, Liyuan Zhang, Zhaowei Zhu, Yang Liu
Venue: ACM SIGKDD 2025 (KDD'25)
연도: 2025
DOI: 10.1145/3690624.3709257
arXiv: 2501.11959
공식 URL: https://arxiv.org/abs/2501.11959

## Abstract 전문 (verbatim)
"Detecting anomalies in temporal data has gained significant attention across various real-world applications, aiming to identify unusual events and mitigate potential hazards. In practice, situations often involve a mix of segment-level labels (detected abnormal events with segments of time points) and unlabeled data (undetected events), while the ideal algorithmic outcome should be point-level predictions. Therefore, the huge label information gap between training data and targets makes the task challenging. In this study, we formulate the above imperfect information as noisy labels and propose NRdetector, a noise-resilient framework that incorporates confidence-based sample selection, robust segment-level learning, and data-centric point-level detection for multivariate time series anomaly detection. Particularly, to bridge the information gap between noisy segment-level labels and missing point-level labels, we develop a novel loss function that can effectively mitigate the label noise and consider the temporal features. It encourages the smoothness of consecutive points and the separability of points from segments with different labels. Extensive experiments on real-world multivariate time series datasets with 11 different evaluation metrics demonstrate that NRdetector consistently achieves robust results across multiple real-world datasets, outperforming various baselines adapted to operate in our setting."

## 핵심 발췌 (verbatim, 섹션/위치 표기)

> "labeling every anomalous time point is neither practical nor precise due to the significant time and cost required for accurate identification." (§1 Introduction)

커버 claim: C-003, C-007
활용 맥락: §1 도입부에서 완전 라벨링의 비현실성을 논할 때 인용. 우리 논문이 semi-supervised 설정을 선택한 동기로 연결.

---

> "acquiring weak labels by simply indicating the occurrence of anomalous events is a more practical approach for real-world applications." (§1 Introduction)

커버 claim: C-003, C-007
활용 맥락: 세그먼트/이벤트 수준 weak label의 실용성 주장. 우리 논문에서 labeled anomaly의 현실적 존재를 정당화할 때 병행 인용 가능.

---

> "only a small portion of positive data are labeled" (§3.3, PU 학습 공식화)
> "the input of training set 𝒳_PU = 𝒳_L ∪ 𝒳_U where 𝒳_L...represents the labeled positive or the unlabeled subset." (§3.3)

커버 claim: C-005, C-006, C-022
활용 맥락: §2.2 관련연구에서 NRdetector의 PU 공식화를 소개할 때. 우리 설정과의 유사점(labeled anomaly 존재)과 차이점(표현 학습 통합 여부)을 대비.

---

> "we do not consider the point adjustment (PA) approach for the evaluation...PA overestimates classifier performance." (§5.2)

커버 claim: C-052
활용 맥락: §4.1.3 평가지표 선택의 정당화. NRdetector도 PA를 배제했다는 선행 선례로 우리의 동일한 선택을 지지.

---

> "split the set of all segments by 7:3 ratio into training and test sets" (§5.1)

커버 claim: C-046
활용 맥락: §4.1.1 우리의 재분할 프로토콜 설명 시 선례로 인용.

---

> "our method achieves robust results under different label noise rates" (§5.3)
> "when the label noise rate is 0.6 cases, the NRdetector completely outperforms the three methods by at least 0.2 on the EMG dataset." (§5.3, Table 4)

커버 claim: C-078, C-079
활용 맥락: §4.4 label sparsity/noise 분석 섹션에서 우리 실험과의 설계 비교. 축 의미(라벨 희소율 vs 세그먼트 노이즈율) 차이 명시 필요.

---

> "performance … constrained by the lack of prior knowledge concerning true anomalies … especially when the anomalies are embedded within the training data" (§1)

커버 claim: C-005, C-017
활용 맥락: 비지도 방법의 "train = all normal" 가정이 현실에서 깨질 때의 한계를 논할 때.

## 우리 논문에서의 활용

커버 claim: C-003, C-005, C-006, C-007, C-010, C-017, C-022, C-024, C-046, C-052, C-073, C-074, C-078, C-079

- **§1 Introduction**: C-003, C-007 — 라벨링 비용 논리와 semi-supervised 설정 동기
- **§2.1 Related Work**: C-005, C-017 — "train = all normal" 한계 논거
- **§2.2 Related Work**: C-022, C-024 — 가장 근접한 선행 연구로 NRdetector 소개; 사전학습-분류 분리(multi-stage) 구조 vs 우리의 end-to-end 차별화
- **§4.1.1 Experiments**: C-046 — 재분할 프로토콜 선례
- **§4.1.3 Evaluation**: C-052 — PA 배제·VUS/Affiliation 채택 선행 사례
- **§4.1.4 Baselines**: C-073 — baseline 출처
- **§4.3 Ablation**: C-074 — Q3(normalonly) 비교 근거
- **§4.4 Label Sparsity**: C-078, C-079 — 선례 비교 (축 의미 차이 주의)

## 주의사항
- NRdetector는 표현 학습(WETAS/DiCNN 사전학습)과 PU 분류를 분리(multi-stage)한다 — 우리 논문의 end-to-end 차별화 논거에서 이 구조적 차이를 명확히 해야 함. 단순히 "NRdetector도 semi-supervised" 식으로 묶으면 우리 기여가 희석됨.
- §5.3 noise sweep 축(세그먼트 노이즈율)이 우리 §4.4 라벨 희소율 축과 다름 — 직접 수치 비교 금지.
- NRdetector는 시계열이지만 video/sensor 도메인 혼용 — 인용 문맥에서 multivariate TSAD 범위 명시.
