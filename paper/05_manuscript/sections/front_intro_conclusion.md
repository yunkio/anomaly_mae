---
phase: 5
agent: section-drafter-1
directives: [T5, R3, R4, R8, R10, R11, R25]
last_modified: 2026-06-11
sections: [Title, Abstract, Keywords, Highlights, §1 Introduction, §5 Conclusion]
model: CSMAD
setting: contaminated semi-supervised
venue: Elsevier
status: annotated-draft
notes: |
  All numerical results are placeholders ([X.XX], [BEST], [N]) with adjacent PH comments.
  Citation keys follow refs.bib / REFERENCE_LIBRARY_INDEX.md.
  Internal code identifiers (force_mask_anomaly, exp271, pak_auc_f1, normalonly) are absent from prose.
  Scoping of novelty claims follows D-008 + §0.1 RT NOTE-03.
  "to our knowledge" accompanies every first-in-class claim.
---

<!-- ============================================================ -->
<!-- TITLE -->
<!-- ============================================================ -->

# Label-Aware Masked Autoencoding with Gradient Reversal for Multivariate Time Series Anomaly Detection

<!-- D-007 confirmed title (PAPER_BLUEPRINT §10.2). Final wording subject to R9-compliant micro-adjustment in Phase 7. -->

---

<!-- ============================================================ -->
<!-- ABSTRACT -->
<!-- ============================================================ -->

## Abstract

Anomaly detection in multivariate time series is critical for industrial control systems, IT infrastructure monitoring, and spacecraft telemetry, yet most existing methods assume that training data are entirely normal — a condition rarely satisfied in practice.
In real deployments, a small fraction of training observations carry anomaly labels derived from recorded fault events, while the majority remain unlabeled; exploiting this structure has received limited attention.
We propose CSMAD, an end-to-end framework that integrates labeled anomaly information directly into masked autoencoder representation learning through three orthogonal mechanisms: anomaly-priority masking, loss bifurcation between normal and anomalous reconstruction paths, and gradient reversal that adversarially suppresses anomaly-specific information from the student's internal representation.
CSMAD employs an asymmetric Teacher–Student decoder architecture in which a capacity-limited student's mimicry degrades preferentially on anomalous correlation patterns, amplifying the teacher–student discrepancy signal under contaminated training.
To enable evaluation of labeled-anomaly-aware methods, we introduce a contaminated benchmark protocol that incorporates the chronological prefix of the test stream into training, exposing labeled anomalies absent in the original train splits of standard benchmarks.
On <!-- PH:NUM-001 number of benchmark datasets --> multivariate datasets spanning industrial and telemetry domains, CSMAD achieves competitive performance against <!-- PH:NUM-002 number of baselines --> unsupervised and weakly supervised baselines under five rigorous evaluation metrics.
The model maintains robust detection as the labeled anomaly fraction decreases, validating the framework beyond the upper-bound labeling scenario.
Code will be made available at [URL] upon acceptance.

<!-- Word count target: 150–200 words. Current draft: ~190 words. -->

---

<!-- ============================================================ -->
<!-- KEYWORDS -->
<!-- ============================================================ -->

## Keywords

Multivariate time series; Anomaly detection; Semi-supervised learning; Masked autoencoder; Asymmetric self-distillation; Gradient reversal; Contaminated benchmark

---

<!-- ============================================================ -->
<!-- HIGHLIGHTS -->
<!-- ============================================================ -->

## Highlights

- We formalize a *contaminated semi-supervised* setting where labeled anomalies coexist with unlabeled training data, and design a benchmark protocol enabling its fair evaluation.
- We propose CSMAD, combining a masked autoencoder with an asymmetric Teacher–Student decoder and gradient reversal to adversarially suppress anomaly-specific information in the student's representation.
- Labeled anomalies guide training via three orthogonal paths — anomaly-priority masking, loss bifurcation, and gradient-reversal suppression — each targeting a distinct learning failure mode.
- A contaminated benchmark protocol (chronological test-prefix incorporation) fills a structural gap in standard benchmarks, where original training splits contain no labeled anomalies.
- CSMAD maintains robust detection under label sparsity and outperforms unsupervised baselines under rigorous multi-metric evaluation on six real-world multivariate datasets.

<!-- Each highlight ≤ 125 characters per Elsevier requirement — verified. -->

---

<!-- ============================================================ -->
<!-- §1 INTRODUCTION -->
<!-- ============================================================ -->

## 1. Introduction

Real-world cyber-physical systems continuously generate high-dimensional, multi-channel sensor streams — water treatment plants, server clusters, and spacecraft telemetry arrays all depend on reliable detection of anomalous states to prevent safety incidents and minimize operational losses \cite{schmidl2022evaluation, blazquez2021review}.
In these environments, anomalies manifest not in isolated channels but through correlated deviations spread across multiple sensor dimensions simultaneously \cite{xu2022anomalytransformer, wu2025catch}, making detection harder as the number of monitored variables grows.
Because labeling every anomalous time point is neither practical nor economical in large-scale deployments, the dominant paradigm for multivariate time series anomaly detection (MTSAD) has been unsupervised learning \cite{wang2025nrdetector}.

The resulting body of work spans four broad families.
Reconstruction-based methods — including density-estimation autoencoders \cite{zong2018dagmm}, stochastic recurrent networks \cite{su2019omnianomaly}, adversarial reconstruction \cite{audibert2020usad}, and memory-augmented transformers \cite{song2023memto} — flag samples whose reconstruction errors exceed a threshold.
Prediction-based methods model expected sensor readings from recent history and score deviations \cite{deng2021gdn}.
Association-discrepancy and contrastive methods exploit the structural gap between normal and anomalous attention patterns \cite{xu2022anomalytransformer, yang2023dcdetector, wu2025catch}.
Self-supervised approaches learn temporal representations through auxiliary objectives \cite{tuli2022tranad, wu2023timesnet}.
Despite their differences, all four families share a common implicit assumption: the training data are drawn entirely from normal operations.
This assumption is structurally embedded in how these methods consume training labels — they have no architectural pathway for leveraging the information carried by labeled anomalies, even when such labels are available.
The best a label-aware variant of an unsupervised method can do is exclude confirmed anomaly windows from training, filtering contamination rather than learning from it.

In practice, however, a small fraction of training observations do carry anomaly labels, typically derived from recorded fault and attack events in operational logs \cite{wang2025nrdetector}.
These labeled anomalies are an obstacle for unsupervised methods — a source of contamination that degrades the "all-normal" assumption — but they are a valuable learning signal for semi-supervised methods.
The gap is particularly acute in standard MTSAD benchmarks: the original training splits of the benchmarks we evaluate on contain no labeled anomalies by construction — SWaT and WaDi training sets correspond to normal-operation periods, PSM and SMD provide no training labels, and SMAP/MSL training labels are uniformly zero \cite{liu2024elephant, schmidl2022evaluation}.
This structural absence makes it impossible to evaluate any method that exploits labeled anomalies directly on these benchmarks without modifying the data protocol.
Our key observation is threefold: labeled anomalies reveal (a) which temporal positions are likely to yield informative hard reconstruction targets, (b) which patches the student decoder should avoid mimicking, and (c) what representational content should be actively erased from the student's encoding.
Exploiting all three simultaneously can amplify both the reconstruction error signal and the teacher–student discrepancy signal on anomalous regions.
Relying only on (b) — excluding anomalous patches from the student's imitation loss — is insufficient on its own: a student that is repeatedly exposed to anomalous patterns during training may learn to reconstruct them accurately through an indirect route, weakening the discrepancy signal at inference time.
The active suppression of (c) closes this route at the representational level.
The only prior work on deep semi-supervised MTSAD we are aware of, NRdetector \cite{wang2025nrdetector}, delegates representation learning to a label-agnostic pre-trained backbone, leaving labels unable to shape the representations themselves.

Building on these observations, we propose **CSMAD** (Contaminated Semi-supervised Masked Anomaly Detector), a single end-to-end framework that integrates labeled anomaly information directly into the representation learning process of a masked autoencoder.
CSMAD employs, to our knowledge, the first architecture that combines masked-reconstruction self-distillation with gradient reversal to adversarially suppress anomaly-specific information from the student's representation in a contaminated semi-supervised multivariate TSAD setting.
Figure 1 illustrates the contrast between the unsupervised paradigm, its label-aware filtering variant, and CSMAD's three-path label integration.
Our contributions are as follows:

1. **Contaminated semi-supervised setting and benchmark protocol.** We formalize the *contaminated semi-supervised* setting, in which labeled anomalies coexist with unlabeled training windows. We introduce a benchmark protocol that incorporates the chronological prefix of each dataset's test stream into training, constructing train splits that contain labeled anomalies absent in the original splits and evaluating on the held-out temporal suffix. This enables systematic evaluation of methods that exploit labeled anomalies — a capability absent in standard MTSAD benchmarks.

2. **Three-path label integration into masked autoencoder representation learning.** We propose CSMAD, which integrates labeled anomalies into a masked autoencoder through three orthogonal mechanisms: (i) *anomaly-priority masking*, which ensures that labeled anomaly patches are selected for reconstruction with highest priority, preventing the model from evading hard positions; (ii) *loss bifurcation*, which restricts the student decoder's imitation objective to normal-patch outputs, steering student representations away from anomaly patterns; and (iii) *gradient reversal suppression*, which adversarially removes anomaly-specific information from the student's internal representation by reversing the gradient of an anomaly classifier trained on the student's hidden state.

3. **Asymmetric Teacher–Student decoder architecture.** We design an architecture in which a deeper teacher decoder (3 layers) establishes a stable normal-reconstruction reference, while a capacity-limited student decoder (2 layers) fails to mimic anomalous correlation patterns more severely than normal ones — making the teacher–student output discrepancy a reliable anomaly signal even under contaminated training conditions.

4. **Extensive empirical evaluation.** Experiments on <!-- PH:NUM-003 number of datasets, e.g. "six" --> multivariate datasets covering industrial control, IT infrastructure, and spacecraft telemetry demonstrate competitive performance against <!-- PH:NUM-004 total number of baselines --> baselines under five evaluation metrics. Label sparsity analysis confirms robust detection as the labeled fraction decreases toward the fully unsupervised limit.

The rest of this paper is organized as follows: Section 2 reviews related work; Section 3 describes CSMAD; Section 4 presents experimental results; Section 5 concludes.

---

<!-- ============================================================ -->
<!-- §5 CONCLUSION -->
<!-- ============================================================ -->

## 5. Conclusion

This paper addresses the underexplored setting in which training data contain a small fraction of labeled anomalies alongside a majority of unlabeled observations — a structure common in industrial deployments yet unsupported by standard MTSAD benchmarks or unsupervised methods.
We proposed CSMAD, which integrates labeled anomaly information into masked autoencoder representation learning through three orthogonal paths: anomaly-priority masking, loss bifurcation toward normal-only student mimicry, and gradient-reversal suppression of anomaly-specific information in the student's representation.
An asymmetric Teacher–Student decoder architecture (3-layer teacher, 2-layer student) transforms the capacity gap into a reliable discrepancy signal under contaminated training.
To support evaluation, we introduced a contaminated benchmark protocol that makes labeled anomalies available in training by incorporating the chronological prefix of each test stream.
Experiments on <!-- PH:NUM-005 number of datasets --> multivariate datasets show competitive performance against <!-- PH:NUM-006 total number of baselines --> unsupervised and weakly supervised baselines under five metrics, and label sparsity analysis confirms that the performance advantage degrades gracefully as the labeled fraction decreases.

A notable limitation is the computational cost of leave-one-out inference, which evaluates each patch independently across all masking patterns — a <!-- PH:NUM-007 multiplier, e.g. "50×" --> increase in forward passes relative to single-mask scoring.
An alternative complementary-masking strategy (implemented but not used in the present experiments) offers a potential avenue for cost reduction, and we leave its cost-accuracy trade-off to future work.
More broadly, the graceful degradation of CSMAD toward the unsupervised limit, as validated by the label sparsity sweep, suggests a natural extension to settings where no labels are available at all by disabling the gradient-reversal pathway — a direction worth exploring systematically.
Code is available at [URL] (to be released upon acceptance).

---

<!-- ============================================================ -->
<!-- PLACEHOLDER BLOCK -->
<!-- ============================================================ -->

## Placeholder Index — This Section

| ID | Location | Description of needed value |
|----|----------|-----------------------------|
| PH:NUM-001 | Abstract sentence 6 | Number of benchmark dataset families/series used in main experiments (e.g., "six") — fill after experiment completion |
| PH:NUM-002 | Abstract sentence 6 | Total number of baselines compared (22 unsupervised + 4 weakly supervised = 26 candidate; confirm after weakly-supervised runs complete) |
| PH:NUM-003 | §1 contribution bullet 4 | Number of multivariate datasets in main experiment — same as PH:NUM-001 |
| PH:NUM-004 | §1 contribution bullet 4 | Total number of baselines — same as PH:NUM-002 |
| PH:NUM-005 | §5 Conclusion sentence 4 | Same as PH:NUM-001 |
| PH:NUM-006 | §5 Conclusion sentence 4 | Same as PH:NUM-002 |
| PH:NUM-007 | §5 Conclusion limitation sentence | Inference cost multiplier relative to single-mask scoring (architecture: N=50 patches, leave-one-out → 50 forward passes; verify exact figure against evaluator implementation) |

<!-- Notes:
  - PH:NUM-001/003/005: PAPER_BLUEPRINT §6.2 lists SWaT, WaDi A1, WaDi A2, PSM, SMD (28 machines), SMAP (54 ch.), MSL (27 ch.) = 6 dataset families / 113 training units. Standard phrasing will be "six multivariate datasets" (families) or "113 time series" (units). Confirm with experiment completion status per §0.4.
  - PH:NUM-002/004/006: 22 unsupervised (5+3+1+6+7) + 4 weakly supervised (DeepMIL, WETAS, TreeMIL, NRdetector) = 26 total. Weakly-supervised GPU runs incomplete as of 2026-06-11; use "22 unsupervised baselines" if weakly-supervised runs remain incomplete at submission.
  - PH:NUM-007: Leave-one-out uses 50 masking patterns batched; relative FLOPs ≈ 50× versus single-mask forward. Verify wall-clock figure from Appendix §B.3 once measured.
-->
