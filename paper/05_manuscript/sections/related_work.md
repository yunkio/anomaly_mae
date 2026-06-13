---
phase: 5
agent: section-drafter-2
directives: [T5, R1, R9, R19, R20, R21, R22]
last_modified: 2026-06-11
section: "§2 Related Work"
page_budget: "1.1p (PAGE_BUDGET §1 정본)"
output_type: "annotated draft (placeholder 포함)"
plagiarism_flags: [RF-006-H1, RF-006-H2, RF-006-H3]
---

# §2. Related Work

<!-- DRAFTER NOTE: 세 소절은 MECE 구조를 유지한다: §2.1은 비지도 TSAD 도메인 공간,
§2.2는 설정(label-informed) 차원의 선행 연구 공간, §2.3은 방법(MAE + distillation) 계보.
세 소절 사이 중복 없음. 분량 합산 목표 1.1p. 괄호 클러스터 인용은 \cite{key} 형태.
수치 없음 — 모든 성능 관련 서술은 §4 Experiments 전속. -->

## 2.1. Multivariate Time Series Anomaly Detection

Deep learning approaches to unsupervised multivariate time series anomaly detection have matured into several well-defined families. Reconstruction-based methods train an encoder–decoder to reproduce normal input; anomalies are flagged where reconstruction error is large \cite{zong2018dagmm,su2019omnianomaly,audibert2020usad,song2023memto}. Prediction-based methods instead model the expected next state from history and score deviations from the forecast \cite{deng2021gdn}. A more recent strand exploits association structure: transformer models that learn temporal dependencies \cite{xu2022anomalytransformer} or inter-channel contrasts \cite{yang2023dcdetector,wu2025catch} produce anomaly scores from the discrepancy between learned and actual patterns. Transformer-based self-supervised pre-training has also been applied directly to the TSAD objective \cite{tuli2022tranad,wu2023timesnet}.

Despite this breadth, every family listed above operates under a shared assumption: the training data are treated as predominantly or entirely normal. When the training stream contains confirmed anomalous events — the contaminated setting that arises naturally in operational logs and failure records — these methods have no mechanism to distinguish known-anomalous from known-normal samples, and the labeled information is either discarded or treated as a source of noise degrading the normal pattern. The present work addresses this structural limitation by integrating labeled anomaly information directly into the representation learning process, rather than relying on post-hoc removal or ignoring it.

<!-- DRAFTER NOTE: TFMAE は §2.3 専属 (ADV MINOR-001); DCdetectorのchannel/patch表現に近い文を避けた (RF-006-H1).
DAGMM は原論文(Zong et al., ICLR 2018)のみ言及 (ADV NOTE-001); §4側で "simplified variant" 注記.
WETAS/DeepMIL/TreeMIL は §2.2 専属 (RT MINOR-03). -->

---

## 2.2. Label-Informed Anomaly Detection: Semi-supervised, PU, and Weakly Supervised

Positive and Unlabeled (PU) learning provides a formal framework for the scenario in which a learner has access to confirmed positive examples and a pool of unlabeled data that may contain additional positives \cite{bekker2020pusurvey,duplessis2014pu}. Two broad solution families have been established: cost-sensitive risk minimization, which corrects for the label bias through a non-negative risk estimator \cite{kiryo2017nnpu}, and two-step techniques, which extract reliable negative examples before training a standard classifier \cite{elkan2008pu}. Outside the time-series domain, these ideas have been adapted to image anomaly detection, including deviation networks that rank anomaly scores with scarce labeled anomalies \cite{pang2019devnet} and deep semi-supervised one-class objectives \cite{ruff2020deepsad}.

In the time-series domain, however, deep representation learning informed by label signals remains rare. A weakly supervised strand that operates on segment-level labels has received recent attention: approaches in this family train models to classify or rank windows using coarse annotations \cite{sultani2018deepmil,lee2021wetas,liu2024treemil}. These methods cast the anomaly problem as a supervised classification task where segment-level labels serve directly as the training objective; they do not employ a self-supervised reconstruction pretext, so the label is the sole learning signal rather than a supplement to representation learning. Two earlier attempts addressed the label-scarcity problem in multivariate time-series through semi-supervised variational models \cite{xue2022fewpositive,huang2022slavae}, but their representation learning remains largely label-agnostic: labels enter through auxiliary loss terms rather than shaping the gradient of the underlying latent space.

The closest precedent to our setting is NRdetector \cite{wang2025nrdetector}, which formulates point-level anomaly detection under noisy segment-level labels as a PU problem. NRdetector explicitly acknowledges that fusing PU learning with time-series anomaly detection is a novel and practical scenario, itself arguing that prior work in this direction is scarce. Its framework is a pipeline: a temporal embedding is first extracted by a pre-trained backbone derived from the WETAS architecture, and a separate PU classifier is then trained on those fixed representations. The label signal therefore guides the classifier's output, not the encoder's gradient — representation learning and label exploitation remain decoupled stages. Our approach differs along this axis: labeled anomaly information enters the gradient of the encoder during training itself, through three orthogonal mechanisms that shape what the model learns to represent rather than what the model predicts at the output. To our knowledge, CSMAD is the first end-to-end model for multivariate TSAD that integrates labeled anomalies into the gradient of a self-supervised representation learning objective.

<!-- DRAFTER NOTE: D-008 スコーピング準拠 — "거의 없음" を "소수 존재하되 본 접근과 상이"에 精緻化.
공통점 3개 (소수 양성 라벨 동기 / PA회피 평가 철학 / train오염 강건성) 만 짧게 인정하고
차이의 중심축 D1/D3/D5 배치 (R20). NRdetector의 "novel and practical scenario" 자기 선언이
이 분야의 희소성 방증으로 기능 (NRDETECTOR_DOSSIER §5 R20 전략). -->

---

## 2.3. Masked Autoencoders and Self-Distillation in Anomaly Detection

The masked autoencoder (MAE) proposed by He et al. \cite{he2022mae} demonstrated that masking random patches of an input image and training a model to reconstruct the missing regions yields strong transferable representations. Our patch-based masking scheme draws directly from this paradigm, adapting it from the spatial domain to windows of multivariate sensor channels; while similar patch-and-mask operations appear in some time-series models \cite{fang2024tfmae}, those are independent developments — our design lineage traces to vision MAE, not to those parallel works.

Knowledge distillation has also been applied to anomaly detection by exploiting the representation gap that naturally emerges between a pre-trained teacher network and a student initialized at random or trained with lower capacity \cite{bergmann2020uninformed,deng2022reverse}. A related and more compact formulation is self-distillation, a concept introduced by Zhang et al. \cite{zhang2022selfdistill} in the context of efficient neural network compression, where a single architecture contains a teacher and one or more student heads that distill knowledge internally rather than from an external model. Ristea et al. \cite{ristea2024sdmae} adapted this architectural paradigm to video anomaly detection, embedding a dual-decoder structure — a deeper teacher decoder and a shallower student decoder — within a masked autoencoder; they term the resulting teacher–student interaction self-distillation following Zhang et al. The anomaly score is derived from the magnitude of the discrepancy between teacher and student reconstructions at test time.

In this work, we adapt a structurally similar self-distillation paradigm to multivariate time series, placing it within a contaminated semi-supervised framework where labeled anomalies actively guide training. Our teacher and student decoders are independent parallel branches off a shared encoder — rather than a branch-off from within the teacher decoder — and the student is additionally trained to suppress anomaly-specific information through a gradient reversal mechanism that operates in representation space rather than in the output or loss space.[^sd-fn]

[^sd-fn]: The self-distillation terminology follows Zhang et al. \cite{zhang2022selfdistill} and Ristea et al. \cite{ristea2024sdmae}, whose student decoder branches off the teacher decoder after its first transformer block. Our student decoder is an independent network receiving encoder representations directly; the gradient reversal layer that adversarially suppresses anomaly information in the student is absent from the video anomaly detection setting of \cite{ristea2024sdmae}. The distinction between operating in the target/loss space versus the gradient space of the representation is elaborated in §3.5.

<!-- DRAFTER NOTE: SDMAE は自然 mention 1–2文 "adapt this paradigm" トーン (オプションC채택, R9遵守).
差異一覧を排除し、脚注に構造差異(branch-off vs 독립 decoder) + 용어계보(R21防御)를 凝縮.
작동 계층 차이(target/loss vs gradient space)는 §3.5 본문 1문장으로 이동 (RT MAJOR-01/MAJOR-08).
RF-006-H2/H3: "leverage the reconstruction discrepancy"·"overlook the anomalies"·"known as self-distillation" 類の表現を使用せず独自の文を生成. -->

<!-- PLACEHOLDER BLOCK (Phase 5 실험 완료 후 교체 필요):
- §2.2 포지셔닝 문장의 "to our knowledge" 주장: 반증 후보 (Xue & Yan IJCNN 2022, SLA-VAE WWW 2022) 최종 분석 후 스코핑 문구 재확인 필요 (D-008 / C-011 / C-025).
- §2.3 마지막 단락의 CSMAD 기능 서술: "contaminated semi-supervised framework" 명칭을 §3.1에서 공식화한 후 여기서 정합성 확인.
- 각주 [^sd-fn] 위치: Phase 7 LaTeX 변환 시 \footnote{} 으로 처리, §3.4 self-distillation 정의 근처 배치로 이동 고려.
-->
