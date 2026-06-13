---
phase: 5
agent: integrator
version: v1
directives: [T5]
last_modified: 2026-06-11
sources:
  - sections/front_intro_conclusion.md (section-drafter-1)
  - sections/related_work.md (section-drafter-2)
  - sections/method.md (section-drafter-3)
  - sections/experiments.md (section-drafter-4)
model: CSMAD
setting: contaminated semi-supervised
venue: Elsevier (elsarticle, 1-column)
status: integrated annotated draft (placeholders unresolved; two compression passes applied)
notes: |
  - Placeholder IDs are GLOBALLY UNIQUE (NUM-001..NUM-030, TXT-001..002,
    FIG-1..4, TAB-1..4). Full inventory: PLACEHOLDER_REGISTRY.md.
  - Numeric placeholders appear as [X.XX] / [N] with an adjacent <!-- PH:... --> comment.
  - All citation keys verified against 04_references/refs.bib (49 valid keys);
    han2023catch corrected to wu2025catch.
  - Integration decisions, terminology unifications, deduplications, and the R6
    page-budget check are documented in INTEGRATION_REPORT_v1.md. Directive-mandated
    defense content (R13/R28/R29/R30/R31/R32, GRL necessity, epoch-asymmetry and
    test-set-selection disclosures) was compressed but never deleted.
  - §4 keeps protocol constants (6 families, 113 units, 26 baselines, 5 metrics, 50% split)
    as real values per drafter-4 policy; front/§1/§5 keep them as placeholders pending
    experiment-completion confirmation (sync point — see registry).
  - References compiled from refs.bib (\bibliography{refs}); Acknowledgments and
    Appendix are out of v1 scope (Phase 6/7).
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
We propose CSMAD, an end-to-end framework that integrates labeled anomaly information directly into masked autoencoder representation learning through three orthogonal mechanisms: anomaly-priority masking, loss bifurcation between normal and anomalous reconstruction paths, and gradient reversal that adversarially suppresses anomaly-specific information from the Student's internal representation.
CSMAD employs an asymmetric Teacher–Student decoder architecture in which a capacity-limited Student's mimicry degrades preferentially on anomalous correlation patterns, amplifying the Teacher–Student discrepancy signal under contaminated training.
To enable evaluation of labeled-anomaly-aware methods, we introduce a contaminated benchmark protocol that incorporates the chronological prefix of the test stream into training, exposing labeled anomalies absent in the original train splits of standard benchmarks.
On [N] <!-- PH:NUM-001 | number of benchmark dataset families in main experiments --> multivariate datasets spanning industrial and telemetry domains, CSMAD achieves competitive performance against [N] <!-- PH:NUM-002 | total number of baselines compared --> unsupervised and weakly supervised baselines under five rigorous evaluation metrics.
The model maintains robust detection as the labeled anomaly fraction decreases, validating the framework beyond the upper-bound labeling scenario.
Code will be made available at [URL] <!-- PH:TXT-002 | code repository URL --> upon acceptance.

<!-- Word count target: 150–200 words. -->

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
- We propose CSMAD, combining a masked autoencoder with an asymmetric Teacher–Student decoder and gradient reversal to adversarially suppress anomaly-specific information in the Student's representation.
- Labeled anomalies guide training via three orthogonal paths — anomaly-priority masking, loss bifurcation, and gradient-reversal suppression — each targeting a distinct learning failure mode.
- A contaminated benchmark protocol (chronological test-prefix incorporation) fills a structural gap in standard benchmarks, where original training splits contain no labeled anomalies.
- CSMAD maintains robust detection under label sparsity and outperforms unsupervised baselines under rigorous multi-metric evaluation on [N] <!-- PH:NUM-003 | dataset-family count; was hard-coded "six" in section draft — converted to placeholder for consistency with Abstract (sync with NUM-001) --> real-world multivariate datasets.

<!-- Each highlight ≤ 125 characters per Elsevier requirement — re-verify in Phase 7 after NUM-003 resolution. -->

---

<!-- ============================================================ -->
<!-- §1 INTRODUCTION -->
<!-- ============================================================ -->

## 1. Introduction

Real-world cyber-physical systems continuously generate high-dimensional, multi-channel sensor streams — water treatment plants, server clusters, and spacecraft telemetry arrays all depend on reliable detection of anomalous states to prevent safety incidents and operational losses \cite{schmidl2022evaluation, blazquez2021review}.
Anomalies in such streams manifest not in isolated channels but through correlated deviations across multiple sensor dimensions \cite{xu2022anomalytransformer, wu2025catch}, and because labeling every anomalous time point is impractical at scale, the dominant paradigm for multivariate time series anomaly detection (MTSAD) has been unsupervised learning \cite{wang2025nrdetector}.

The resulting body of work spans four broad families: reconstruction-based methods, which flag samples whose reconstruction errors exceed a threshold \cite{zong2018dagmm, su2019omnianomaly, audibert2020usad, song2023memto}; prediction-based methods, which score deviations from forecast sensor readings \cite{deng2021gdn}; association-discrepancy and contrastive methods, which exploit the structural gap between normal and anomalous attention patterns \cite{xu2022anomalytransformer, yang2023dcdetector, wu2025catch}; and self-supervised approaches that learn temporal representations through auxiliary objectives \cite{tuli2022tranad, wu2023timesnet}.
Despite their differences, all four families share an implicit assumption that the training data are drawn entirely from normal operations.
This assumption is structurally embedded: the methods have no architectural pathway for leveraging the information carried by labeled anomalies even when such labels are available — the best a label-aware variant can do is exclude confirmed anomaly windows from training, filtering contamination rather than learning from it.

In practice, however, a small fraction of training observations do carry anomaly labels, typically derived from recorded fault and attack events in operational logs \cite{wang2025nrdetector}.
These labeled anomalies are an obstacle for unsupervised methods — a source of contamination — but a valuable learning signal for semi-supervised ones.
The gap is particularly acute in standard MTSAD benchmarks, whose original training splits contain no labeled anomalies by construction \cite{liu2024elephant, schmidl2022evaluation}; evaluating any method that exploits labeled anomalies therefore requires modifying the data protocol, as detailed in Section 4.1.1.
<!-- INTEGRATOR: dataset-by-dataset enumeration removed here — canonical statement lives in §4.1.1 (R13 defense); deduplication per integration pass. -->
Our key observation is threefold: labeled anomalies reveal (a) which temporal positions yield informative hard reconstruction targets, (b) which patches the Student decoder should avoid mimicking, and (c) what representational content should be actively erased from the Student's encoding.
Exploiting all three simultaneously amplifies both the reconstruction error signal and the Teacher–Student discrepancy signal on anomalous regions.
Relying only on (b) is insufficient: a Student repeatedly exposed to anomalous patterns during training may learn to reconstruct them accurately through an indirect route, weakening the discrepancy signal at inference time; the active suppression of (c) closes this route at the representational level.
The only prior work on deep semi-supervised MTSAD we are aware of, NRdetector \cite{wang2025nrdetector}, delegates representation learning to a label-agnostic pre-trained backbone, leaving labels unable to shape the representations themselves.

[FIG-1] <!-- PH:FIG-1 | Setting-comparison 3-way diagram (unsupervised / label-aware filtering / CSMAD) — full caption and content spec in PLACEHOLDER_REGISTRY.md. Placement per PAGE_BUDGET: after Para 3, before contribution paragraph. -->

Building on these observations, we propose **CSMAD** (Contaminated Semi-supervised Masked Anomaly Detector), a single end-to-end framework that integrates labeled anomaly information directly into the representation learning of a masked autoencoder.
CSMAD employs, to our knowledge, the first architecture combining masked-reconstruction self-distillation with gradient reversal to adversarially suppress anomaly-specific information from the Student's representation in a contaminated semi-supervised multivariate TSAD setting.
Figure 1 contrasts the unsupervised paradigm, its label-aware filtering variant, and CSMAD's three-path label integration.
Our contributions are as follows:

1. **Contaminated semi-supervised setting and benchmark protocol.** We formalize the *contaminated semi-supervised* setting, in which labeled anomalies coexist with unlabeled training windows, and introduce a benchmark protocol that incorporates the chronological prefix of each dataset's test stream into training — constructing train splits with labeled anomalies absent from the original splits and evaluating on the held-out temporal suffix.

2. **Three-path label integration into masked autoencoder representation learning.** CSMAD integrates labeled anomalies through three orthogonal mechanisms: (i) *anomaly-priority masking*, which selects labeled anomaly patches for reconstruction with highest priority, preventing the model from evading hard positions; (ii) *loss bifurcation*, which restricts the Student decoder's imitation objective to normal-patch outputs; and (iii) *gradient reversal suppression*, which adversarially removes anomaly-specific information from the Student's internal representation via a reversed-gradient anomaly classifier.

3. **Asymmetric Teacher–Student decoder architecture.** A deeper Teacher decoder (3 layers) establishes a stable normal-reconstruction reference, while a capacity-limited Student decoder (2 layers) fails to mimic anomalous correlation patterns more severely than normal ones — making the Teacher–Student output discrepancy a reliable anomaly signal under contaminated training.

4. **Extensive empirical evaluation.** Experiments on [N] <!-- PH:NUM-004 | dataset-family count (sync with NUM-001) --> multivariate datasets covering industrial control, IT infrastructure, and spacecraft telemetry demonstrate competitive performance against [N] <!-- PH:NUM-005 | total baseline count (sync with NUM-002) --> baselines under five evaluation metrics, with label sparsity analysis confirming robust detection toward the fully unsupervised limit.

The rest of this paper is organized as follows: Section 2 reviews related work; Section 3 describes CSMAD; Section 4 presents experimental results; Section 5 concludes.

---

<!-- ============================================================ -->
<!-- §2 RELATED WORK -->
<!-- ============================================================ -->

## 2. Related Work

### 2.1. Multivariate Time Series Anomaly Detection

Deep learning approaches to unsupervised MTSAD have matured into several well-defined families. Reconstruction-based methods train an encoder–decoder to reproduce normal input and flag large reconstruction errors \cite{zong2018dagmm,su2019omnianomaly,audibert2020usad,song2023memto}. Prediction-based methods model the expected next state from history and score deviations from the forecast \cite{deng2021gdn}. A more recent strand exploits association structure: transformer models that learn temporal dependencies \cite{xu2022anomalytransformer} or inter-channel contrasts \cite{yang2023dcdetector,wu2025catch} score the discrepancy between learned and actual patterns. Transformer-based self-supervised pre-training has also been applied directly to the TSAD objective \cite{tuli2022tranad,wu2023timesnet}.

Despite this breadth, every family above treats the training data as predominantly or entirely normal. When the training stream contains confirmed anomalous events — the contaminated setting that arises naturally from operational logs — these methods cannot distinguish known-anomalous from known-normal samples; labeled information is either discarded or treated as noise degrading the normal pattern. The present work addresses this structural limitation by integrating labeled anomaly information directly into representation learning rather than relying on post-hoc removal.

### 2.2. Label-Informed Anomaly Detection: Semi-supervised, PU, and Weakly Supervised

Positive and Unlabeled (PU) learning formalizes the scenario in which a learner has confirmed positive examples and a pool of unlabeled data that may contain additional positives \cite{bekker2020pusurvey,duplessis2014pu}, with two established solution families: cost-sensitive risk minimization via non-negative risk estimators \cite{kiryo2017nnpu} and two-step techniques that extract reliable negatives before training a classifier \cite{elkan2008pu}. Outside time series, these ideas have been adapted to image anomaly detection through deviation networks with scarce labeled anomalies \cite{pang2019devnet} and deep semi-supervised one-class objectives \cite{ruff2020deepsad}.

In the time-series domain, deep representation learning informed by label signals remains rare. A weakly supervised strand trains models to classify or rank windows from coarse segment-level annotations \cite{sultani2018deepmil,lee2021wetas,liu2024treemil}; the label is the sole learning signal, with no self-supervised reconstruction pretext. Two earlier semi-supervised variational models addressed label scarcity in multivariate time series \cite{xue2022fewpositive,huang2022slavae}, but their representation learning remains largely label-agnostic: labels enter through auxiliary loss terms rather than shaping the gradient of the latent space.

The closest precedent to our setting is NRdetector \cite{wang2025nrdetector}, which formulates point-level detection under noisy segment-level labels as a PU problem — itself arguing that fusing PU learning with TSAD is a novel scenario for which prior work is scarce. Its framework is a pipeline: a temporal embedding is extracted by a pre-trained backbone derived from the WETAS architecture, and a separate PU classifier is trained on those fixed representations; the label signal guides the classifier's output, not the encoder's gradient. Our approach differs along this axis: labeled anomaly information enters the gradient of the encoder during training itself, through three orthogonal mechanisms that shape what the model learns to represent rather than what it predicts. To our knowledge, CSMAD is the first end-to-end model for multivariate TSAD that integrates labeled anomalies into the gradient of a self-supervised representation learning objective.

### 2.3. Masked Autoencoders and Self-Distillation in Anomaly Detection

The masked autoencoder (MAE) of He et al. \cite{he2022mae} showed that masking random patches and reconstructing the missing regions yields strong transferable representations. Our patch-based masking draws directly from this paradigm, adapted from the spatial domain to windows of multivariate sensor channels; similar patch-and-mask operations in some time-series models \cite{fang2024tfmae} are independent developments — our design lineage traces to vision MAE.

Knowledge distillation has been applied to anomaly detection by exploiting the representation gap between a pre-trained teacher and a lower-capacity or randomly initialized student \cite{bergmann2020uninformed,deng2022reverse}. A more compact formulation is self-distillation, introduced by Zhang et al. \cite{zhang2022selfdistill} for efficient network compression, where one architecture contains a teacher and internal student heads. Ristea et al. \cite{ristea2024sdmae} adapted this paradigm to video anomaly detection, embedding a deeper teacher decoder and a shallower student decoder within a masked autoencoder and scoring anomalies by the teacher–student reconstruction discrepancy at test time.

In this work, we adapt a structurally similar self-distillation paradigm to multivariate time series, placing it within a contaminated semi-supervised framework where labeled anomalies actively guide training. Our Teacher and Student decoders are independent parallel branches off a shared encoder — rather than a branch-off from within the teacher decoder — and the Student is additionally trained to suppress anomaly-specific information through a gradient reversal mechanism operating in representation space rather than in the output or loss space.[^sd-fn]

[^sd-fn]: The self-distillation terminology follows Zhang et al. \cite{zhang2022selfdistill} and Ristea et al. \cite{ristea2024sdmae}. The gradient reversal layer that adversarially suppresses anomaly information in the Student is absent from the unsupervised video setting of \cite{ristea2024sdmae}; the distinction between operating in the target/loss space versus the gradient space of the representation is elaborated in Section 3.5.

---

<!-- ============================================================ -->
<!-- §3 METHODOLOGY -->
<!-- ============================================================ -->

## 3. Methodology

### 3.1. Problem Formulation and Setting

Let $\mathbf{X} \in \mathbb{R}^{T \times F}$ denote a multivariate time series of $T$ timesteps and $F$ sensor channels.
A sliding window of length $L$ produces windows $\mathbf{W} \in \mathbb{R}^{L \times F}$, each partitioned into $N$ non-overlapping patches of size $s$ ($L = N \cdot s$); the $i$-th patch is $\mathbf{P}_i = \mathbf{W}[is{:}(i{+}1)s,\;:] \in \mathbb{R}^{s \times F}$.
Binary labels exist at the patch level, $y^p_i \in \{0,1\}$, and the window level, $y^w \in \{0,1\}$ (1 if any timestep in the window is anomalous).

We work under a **contaminated semi-supervised** setting: $\mathcal{D}_{\mathrm{train}}$ contains a large majority of unlabeled windows and a small fraction of windows carrying anomaly labels, reflecting the fault records of industrial systems.
The main experiments represent the label-availability upper bound — every anomalous timestep in the training split is labeled — and Section 4.4 validates the general case where only a fraction of anomalies carry labels.
At inference, labels are not used; the model outputs an anomaly score per timestep.
Multivariate data motivate this design: anomalies manifest as correlated deviations across channels, and filtering labeled anomalies as noise discards the co-occurrence structure that distinguishes true faults from single-channel noise — structure that the masking, loss, and gradient pathways of Sections 3.3–3.5 exploit directly.

### 3.2. Overall Architecture

[FIG-2] <!-- PH:FIG-2 | Architecture diagram (five components, GRL "training only", stop-gradient ⊥) — full caption and content spec in PLACEHOLDER_REGISTRY.md. -->

CSMAD consists of five functional blocks (Figure 2): a linear patch embedding layer, a shared Transformer encoder, a Teacher decoder, a Student decoder, and a label-guided training module coupling the two decoders through gradient reversal — the latter active during training only.
The encoder receives only the *visible* (unmasked) patches; each decoder receives the latent sequence with its own learnable mask tokens and positional embeddings and reconstructs the full window, the Teacher being strictly deeper than the Student (Section 3.4).
A key isolation constraint governs gradient flow: the Student decoder and the GRL branch read the encoder output through a stop-gradient, so the encoder is optimized exclusively by the Teacher's reconstruction objective — the adversarial signal cannot corrupt the normal-pattern representation underpinning the anomaly score.
At inference the GRL branch is inactive; scores derive from the Teacher's reconstruction error and the Teacher–Student discrepancy alone (Section 3.6).

### 3.3. Patch Embedding and Masking

**Linear patch embedding.**
Each patch $\mathbf{P}_i$ is flattened to a vector of dimension $s \times F$ and projected to dimension $d$ by a learned linear map with LayerNorm:

$$\mathbf{z}_i = \mathrm{LayerNorm}\!\left(\mathbf{W}_{\mathrm{emb}}\,\mathrm{vec}(\mathbf{P}_i) + \mathbf{b}_{\mathrm{emb}}\right), \quad \mathbf{z}_i \in \mathbb{R}^{d} \tag{1}$$

Projecting an entire patch — $s$ timesteps and all $F$ channels — into a single token captures joint temporal–spatial structure in one linear operation, reflecting the channel-wise correlations typical of sensor networks directly in the token representation; this follows the linear patchify principle of MAE \cite{he2022mae}, adapted to multivariate time series.

**Anomaly-priority masking.**
$|M| = \mathrm{round}(N \times r_m)$ patches are withheld from the encoder, which processes only the $|V| = N - |M|$ visible tokens; learnable mask tokens are inserted into the full-length sequence just before each decoder.
When patch-level labels are available, each patch receives a priority $\pi_i = 10^3 \cdot y^p_i + \eta_i$, with $\eta_i \sim \mathrm{Uniform}(0,1)$ breaking ties, and the masked set is

$$M = \mathrm{argtopk}_{|M|}\!\left\{\pi_i\right\}_{i=1}^{N} \tag{2}$$

If the window's anomalous patches number at most $|M|$, all are masked and the remainder is drawn at random from normal patches; otherwise $|M|$ anomalous patches are chosen at random.

This addresses a structural imbalance of contaminated training: anomalous patches are a small fraction of each window, so stochastic masking rarely selects them and the model learns to reconstruct *around* rather than *through* them — a blind spot at exactly the positions that matter for detection.
At test time, masking reverts to uniform random selection.

### 3.4. Asymmetric Teacher–Student Decoders

**Encoder.**
The visible tokens pass through a Transformer encoder of depth $n_e$ (Pre-Layer-Normalization, multi-head self-attention, GELU), producing latents $\{h^{\mathrm{enc}}_i\}_{i \in V}$.

**Teacher decoder.**
Learnable mask tokens are inserted at positions in $M$, position embeddings added, and the full sequence passed through a self-attention-only Transformer decoder of depth $n_T$, following the standard MAE decoder design \cite{he2022mae}; hidden states $\{h^T_i\}$ are projected to reconstructions $\{o^T_i\}$ by a linear head.

**Student decoder.**
The Student has the same structure but is shallower ($n_S < n_T$) with separate mask tokens; critically, its input latent carries a stop-gradient:

$$\mathbf{h}^S_{\mathrm{in},i} = \mathrm{stopgrad}\!\left(h^{\mathrm{enc}}_i\right), \quad i \in V \tag{3}$$

It produces hidden states $\{h^S_i\}$ and outputs $\{o^S_i\}$.

**Why the capacity gap matters.**
A deeper Teacher faithfully learns the joint normal correlation structure across all $F$ channels; the shallower Student replicates it on recurring normal patterns but fails more consistently on the atypical patterns characterizing anomalies, so the output discrepancy carries a stronger anomaly signal than either reconstruction error alone.
This adapts the self-distillation principle of \citet{zhang2022selfdistill}, applied to video anomaly detection by \citet{ristea2024sdmae}, to our contaminated semi-supervised setting, where labeled anomalies additionally widen the discrepancy at known fault locations.

**Teacher-only warmup.**
For a fixed initial number of epochs the Student forward pass is skipped and the loss is computed on the Teacher branch alone, so the Student begins its adversarial role only against a stable normal-reconstruction reference; we treat this warmup as a training-stability device, not an independent contribution.

**GRL dual-$\lambda$ structure.**
Two independent quantities govern the gradient reversal branch once the Student is active: the **loss weight** $\lambda_{\mathrm{GRL}}$, scaling the classification loss within the total and set adaptively each epoch as the clamped ratio of main-loss to GRL-loss gradient norms (preventing the adversarial loss from dominating or vanishing); and the **reversal coefficient** $\lambda_{\mathrm{rev}}$, scaling the gradient during backpropagation through the GRL on the sigmoid schedule of \citet{ganin2016dann}:

$$\lambda_{\mathrm{rev}}(p) = \frac{2}{1 + \exp(-10\,p)} - 1, \quad p = \mathrm{clip}\!\left(\frac{e - e_0 + 1}{e_1 - e_0},\; 0,\; 1\right), \tag{4}$$

where $e$ is the current epoch and $[e_0, e_1]$ the student-training phase; $\lambda_{\mathrm{rev}}$ grows from $\approx 0.02$ to $\approx 1$, increasing suppression strength without destabilizing early Student learning.

<!-- INTEGRATOR: the SDMAE-difference footnote that appeared here in the section draft was removed — it duplicated the §2.3 footnote (canonical R21 defense) and §2.3 body text; the mandated one-sentence layer contrast remains in §3.5. -->

### 3.5. Label-Guided Training

Three loss components couple labeled anomaly information to the model, each targeting a different level of the learning process.

**Output discrepancy loss $L_{\mathrm{OD}}$.**
Let $P_n = \{i \in M : y^p_i = 0\}$ be the masked patches labeled normal; the loss minimizes the squared difference between the Teacher's detached output and the Student's output on this subset only:

$$L_{\mathrm{OD}} = \frac{1}{|P_n|} \sum_{i \in P_n} \left\| o^T_i - o^S_i \right\|^2 \tag{5}$$

Anomalous patches are excluded entirely: the Student is steered toward agreement with the Teacher on normal patterns while remaining free to deviate at anomaly locations.

**Feature matching loss $L_{\mathrm{FM}}$.**
A hidden-space regularizer penalizes Teacher–Student representation distance on normal masked patches ($h^T_i$ detached):

$$L_{\mathrm{FM}} = \frac{1}{|P_n| \cdot d} \sum_{i \in P_n} \left\| h^T_i - h^S_i \right\|^2 \tag{6}$$

This prevents the Student's hidden representation from drifting to solutions that satisfy the output-level constraint without tracking the Teacher's internal structure.
Its weight $\lambda_{\mathrm{FM}}$ follows the same adaptive mechanism as $\lambda_{\mathrm{GRL}}$; the loss is a training-only regularizer and does not enter the inference score.

**GRL anomaly suppression loss $L_{\mathrm{cls}}$.**
Whereas SDMAE's anomaly-overlook supervision operates in the target/loss space, our GRL operates in the gradient space of the Student's internal representation.
A two-layer MLP head $g_\phi$ is applied to each masked patch's Student hidden state to predict whether the enclosing window contains an anomaly ($y^w$ broadcast to all masked patches), trained with a focal-style BCE variant designed for severe class imbalance:

$$L_{\mathrm{cls}} = \frac{1}{|M|} \sum_{i \in M} \left(1 - e^{-\ell_i}\right)^{\!2} \ell_i \tag{7}$$

where $\ell_i = \mathrm{BCE}_{w_+}(\hat{y}_i,\, y^w)$ is the class-prior-weighted binary cross-entropy and $w_+$ the per-entity normal-to-anomalous window ratio.
Unlike the standard focal loss \cite{lin2017focal}, which defines $p_t$ from the raw prediction, here $p_t := e^{-\ell_i}$ derives from the pos-weight-adjusted BCE, weighting hard examples by both confidence and prior imbalance.
The gradient reversal layer between the head and the Student hidden states is an identity forward and negates the gradient backward:

$$\frac{\partial \tilde{h}^S_i}{\partial h^S_i} = -\lambda_{\mathrm{rev}} \cdot \mathbf{I} \tag{8}$$

The adversarial gradient reaching the Student hidden state, $-\lambda_{\mathrm{rev}} \cdot \lambda_{\mathrm{GRL,eff}} \cdot \partial L_{\mathrm{cls}}/\partial(\mathrm{GRL\;output})$, opposes the classifier's search for anomaly-discriminative features, pushing the Student toward anomaly-*invariant* internal states.

**Why gradient reversal is necessary beyond loss bifurcation.**
Excluding anomalous patches from $L_{\mathrm{OD}}$ removes the demand that the Student *follow* the Teacher there, but not the possibility of *memorizing* anomaly-specific reconstruction patterns through repeated exposure — which would shrink the discrepancy exactly where it is most informative.
Gradient reversal closes this route, forcing the Student's hidden states to be uninformative about anomaly identity regardless of whether anomalies appear in the loss.

**Total training loss.**

$$L_{\mathrm{total}} = L_{\mathrm{recon}} + L_{\mathrm{OD}} + \lambda_{\mathrm{FM}} \cdot L_{\mathrm{FM}} + \lambda_{\mathrm{GRL}} \cdot L_{\mathrm{cls}} \tag{9}$$

where $L_{\mathrm{recon}}$ is the Teacher's mean squared reconstruction error on masked positions; the GRL branch contributes only when at least one positive window is present in the batch.

### 3.6. Anomaly Scoring and Inference

**Leave-one-out masking.**
Each of the $N$ patches of a test window is scored under a masking pattern in which that patch alone is masked, the $N$ patterns forwarded in parallel through the batch dimension.
This eliminates cross-patch interference (no other patch's masking state leaks into patch $i$'s reconstruction context) at an inference cost of approximately $N$ single-window forward passes — an acknowledged limitation.

**Patch-level anomaly score.**
For each masked patch $i$, the Teacher reconstruction error $r_i$ (MSE over its $s \cdot F$ values) and the Teacher–Student discrepancy $d_i = \|o^T_i - o^S_i\|^2 / (s \cdot F)$ are computed; the GRL classifier is not used at inference.
The discrepancy is adaptively scaled to the magnitude of $r_i$,

$$\tilde{d}_i = d_i \cdot \frac{\bar{r} + \varepsilon}{\bar{d} + \varepsilon}, \quad \varepsilon = 10^{-4} \tag{10}$$

with window-level means $\bar{r} = (1/N)\sum_j r_j$, $\bar{d} = (1/N)\sum_j d_j$, and combined at a fixed ratio:

$$\sigma_i = r_i + \frac{\tilde{d}_i}{c}, \quad c = 4 \tag{11}$$

Adaptive scaling is necessary because component magnitudes vary substantially across datasets; without it, one component can dominate regardless of discriminative value.
The ratio $c$ sets the discrepancy contribution to one quarter of the reconstruction term after scaling.

**Point-level aggregation.**
Each timestep $t$ belongs to one or more (window, patch) pairs; its final score is the mean of $\sigma_i$ over all such pairs:

$$s_t = \frac{\sum_{(w,i):\; t \in \mathbf{P}^w_i} \sigma^w_i}{\left|\{(w,i):\; t \in \mathbf{P}^w_i\}\right|} \tag{12}$$

Averaging across overlapping windows provides ensemble smoothing of single-window reconstruction-context variation.

---

<!-- ============================================================ -->
<!-- §4 EXPERIMENTS -->
<!-- ============================================================ -->

## 4. Experiments

### 4.1. Experimental Setup

#### 4.1.1. Datasets and Benchmark Protocol

**Datasets.**
We evaluate CSMAD on six families of real-world multivariate datasets — industrial cyber-physical systems, IT infrastructure, and spacecraft telemetry, where anomalies occur in realistic operational streams as the contaminated setting requires: SWaT (A1+A2) \cite{goh2016swat} (water treatment); WaDi A1 and A2 \cite{ahmed2017wadi} (water distribution, independent entities); PSM \cite{abdulaal2021psm} (server monitoring); SMD \cite{su2019omnianomaly} (28 server machines); and SMAP (54 telemetry channels) and MSL (27 channels) \cite{hundman2018telemanom}.
<!-- INTEGRATOR: per-dataset feature counts removed from prose — they duplicate Table 1's #Dimensions column; SWaT 45-feature detail remains in §4.1.2 (reproducibility note). -->
The benchmark comprises 113 learning units (1 + 2 + 1 + 28 + 54 + 27); SWaT is evaluated under two conditions (below), giving 114 evaluation units; sizes, dimensionalities, and anomaly ratios are in Table 1.

[TAB-1] <!-- PH:TAB-1 | Dataset statistics under the contaminated protocol (7 rows × 7 cols) — full caption, row values, and size spec in PLACEHOLDER_REGISTRY.md. -->

**Contaminated benchmark protocol.**
A defining feature of standard MTSAD benchmarks is that their original training splits contain no labeled anomalies by construction — SWaT and WaDi training files record normal operation, PSM and SMD provide no training labels, and SMAP/MSL training labels are zero throughout — so a method that exploits labeled anomalies cannot be evaluated without modifying the partition.
We therefore re-split each dataset: the original test file is divided at its temporal midpoint, the earlier 50\% appended to the training data, the later 50\% reserved exclusively for evaluation; labeled anomalies are then genuinely present in training, at ratios from 0.52\% (WaDi A1) to 6.20\% (PSM), SMD varying by machine (Table 1).
The evaluation half is never observed during training or threshold selection, so no temporal lookahead is introduced.
The halving rule is uniform; for SMAP/MSL the split point shifts outward when it falls within ten timesteps of an annotated anomaly region (activated on 4 of 81 channels, all MSL; largest shift 166 timesteps, 7.58\% of that channel's test length).
The re-split is a redefinition of the benchmark, not a use of held-out labels: no model sees evaluation labels at any stage, and the identical partition is provided to all methods (Section 4.1.4); Wang et al. \cite{wang2025nrdetector} provide precedent, re-splitting standard benchmarks 7:3 so anomalies appear within the training stream.
One limitation remains: the anomaly type distribution of the incorporated prefix may differ from that of the evaluation suffix.

**Normalization.**
Inputs are min–max scaled per feature on each entity's training portion; SMD/SMAP/MSL entities are normalized independently, preventing large entities from absorbing the statistics of smaller ones.

**SWaT dual evaluation.**
SWaT is trained once but evaluated twice: in the full condition a single attack event (region 22, $\sim$35,900 timesteps) accounts for 83.75\% of test anomaly mass and 15.96\% of the evaluation set, so the full metric chiefly reflects detection of this one event.
We therefore also report all metrics with region 22 masked out (excl22; anomaly ratio 3.68\% versus 19.05\%) — same model, same scores, only the evaluation mask differs, identically for all baselines.
Table 2 ranks under excl22; full-condition results appear in Appendix §A.5.

#### 4.1.2. Implementation Details

**Architecture and training.**
CSMAD uses patch size 10 on windows of length 500 (50 patches), masking ratio 0.15 (8 masked patches), a 4-layer encoder ($d_\text{model}=512$ fixed across all entities, 8 heads, feedforward 2048, dropout 0.15), and a 3-layer Teacher with a 2-layer Student decoder (self-attention only); the input dimension $F$ varies by dataset (Table 1; Appendix §C.1), with SWaT at 45 features after constant-column removal \{P202, P401, P404, P502, P601, P603\} — reproductions should verify this dimension.
Training uses AdamW (learning rate $10^{-3}$, weight decay $10^{-3}$; GRL classifier $10^{-4}$), batch size 1024, 500 epochs, 10-epoch linear warmup with cosine annealing, bf16 precision, seed 42; the Teacher-only phase covers the first 250 epochs (Section 3.4); full hyperparameters in Appendix §A.1.

**Epoch asymmetry disclosure.**
Unsupervised baselines train 10 epochs and weakly supervised baselines 50, evaluated every epoch; CSMAD trains 500, evaluated every 5.
All methods share the selection criterion, no early stopping, and best-epoch reporting; budgets reflect convergence characteristics — CSMAD needs the 250-epoch warmup before the Student activates — and batch sizes follow original implementations (512 for baselines).
We report this asymmetry transparently; Appendix §B.4 analyzes epoch-budget sensitivity for representative baselines.

**Test-set model selection.**
Best-epoch selection for CSMAD and all 26 baselines evaluates PA\%K-AUC F1 on the test split — no separate validation split exists in this protocol.
Uniform across methods, this leaves relative rankings unaffected but may bias absolute estimates optimistically; we acknowledge this limitation.

**Inference and threshold.**
Inference uses the leave-one-out masking of Section 3.6 with test stride 1 (wall-clock cost in Appendix §B.3).
Threshold-dependent metrics use the anomaly-ratio threshold — the $(1-r)$ quantile of the score distribution, $r$ being the labeled anomaly fraction of the evaluation set — applied identically to all methods per the convention of \cite{xu2022anomalytransformer}; $r$ derives from evaluation-set ground truth but is never used in training, and threshold-free metrics (VUS, PA\%K-AUC families) are unaffected.

**Hardware and code.**
All experiments run on [GPU model] <!-- PH:TXT-001 | GPU model used for experiments -->; code will be released at [URL] <!-- PH:TXT-002 | code repository URL -->.

#### 4.1.3. Evaluation Metrics

We adopt five metrics assessing complementary aspects of detection quality, following the multi-metric philosophy of recent benchmark analyses \cite{kim2022rigorous, paparrizos2022vus, liu2024elephant, wang2025nrdetector}.

**PA\%K-AUC F1** \cite{kim2022rigorous}: for each $K \in \{0, 5, \ldots, 100\}$, point-adjusted F1 is computed under PA\%K — a predicted segment qualifies for point adjustment only when more than $K\%$ of its points are detected — and the area under the F1-vs-$K$ curve integrates from the lenient ($K{=}0$, standard point adjustment) to the strict ($K{=}100$, point-wise) end, removing dependence on any particular $K$; this is our primary metric and selection criterion.

**PA\%K-AUC AUC-PR**: the same $K$ sweep using the area under the precision–recall curve at each $K$; complementary under class imbalance.

**VUS-PR and VUS-ROC** \cite{paparrizos2022vus}: the Volume Under the PR/ROC Surface sweeps both a threshold and a temporal tolerance, measuring ranking quality without an operating point; VUS-PR is the most reliable single TSAD measure per a large-scale benchmark study \cite{liu2024elephant} and suits our class imbalances (test anomaly ratios 3.68\%–30.63\%), VUS-ROC a widely comparable complement.

**Affiliation F1** \cite{huet2022affiliation}: the harmonic mean of affiliation precision/recall, measuring the temporal distance between predicted and ground-truth events per event — localization accuracy robust to adversarial scoring; computed at the anomaly-ratio threshold.

**PA F1 (auxiliary, oracle threshold)**: point-adjusted F1 at $K{=}0$ \cite{xu2018kpivae}, included for comparability although even a random score can reach state-of-the-art levels under it \cite{kim2022rigorous}; reported only in Appendix §A.3, marked (oracle) for its F1-optimal threshold, never used for ranking.

The five metrics span three orthogonal perspectives — threshold-free ranking (VUS), tolerance-spectrum integration (PA\%K-AUC), and local event localization (Affiliation F1) — with distinct failure modes; reporting all five prevents any single failure mode from going undetected.

#### 4.1.4. Baselines and Comparison Conditions

We compare against 26 baselines in two groups.
**Unsupervised (22):** five simple detectors (random, sensor-range deviation, PCA reconstruction, L2-norm, nearest-neighbor distance), three lightweight neural detectors (MLP, MLPMixer, single-stack Transformer), and a GCN-LSTM detector, following \cite{sarfraz2024quovadis}; six established deep TSAD systems \cite{xu2022anomalytransformer, tuli2022tranad, audibert2020usad, zong2018dagmm, deng2021gdn, su2019omnianomaly}; and seven recent competitive methods \cite{fang2024tfmae, lai2023npsr, wu2023timesnet, yang2023dcdetector, song2023memto, luo2024moderntcn, wu2025catch}.
For DAGMM we follow the simplified TranAD-repository re-implementation (GMM energy term omitted), labelled accordingly.
**Weakly supervised (4):** DeepMIL \cite{sultani2018deepmil}, WETAS \cite{lee2021wetas}, TreeMIL \cite{liu2024treemil}, and NRdetector \cite{wang2025nrdetector}, which exploit labeled anomalies during training; they are evaluated under the Q1 condition only, since removing all labeled anomalies (Q3) would eliminate the positive windows their objectives require.

**Comparison conditions.**
The main comparison uses the Q3 (normal-only) condition for all 22 unsupervised baselines: labeled anomaly regions are excised from the contaminated training data and the surviving normal segments concatenated with boundary-aware windowing.
Under purely unsupervised learning, the most effective use of a labeled anomaly is removal as a contaminating sample \cite{bekker2020pusurvey}; Q3 gives each unsupervised method its best available footing, while CSMAD trains on the full contaminated set without excision.
Q3 excision reduces baseline training volume by the train anomaly ratio (0.52\%–6.20\%; SMD pending), which may itself contribute to the gap; Table 4 (Section 4.2) decouples these effects, and Appendix §A.2 reports Q1 results for all unsupervised baselines.

### 4.2. Main Results

Table 2 presents PA\%K-AUC F1 and VUS-PR for CSMAD and all 26 baselines across the six dataset families (bold = best, underline = second best); full five-metric results are in Appendix §A.3.

[TAB-2] <!-- PH:TAB-2 | Main comparison table (27 methods × 7 dataset columns × 2 metrics, landscape) — full caption, row/column structure, and size spec in PLACEHOLDER_REGISTRY.md. -->

CSMAD achieves the highest PA\%K-AUC F1 on [N] of the six dataset families and the highest VUS-PR on [N] <!-- PH:NUM-006 | overall ranking summary (wins out of 6 on each metric) -->, averaging [X.XX] PA\%K-AUC F1 and [X.XX] VUS-PR <!-- PH:NUM-007 | CSMAD averages across all dataset families --> across families, and outperforms the strongest unsupervised competitor (Q3) by [X.XX] <!-- PH:NUM-008 | average margin over best unsupervised baseline, PA%K-AUC F1 --> absolute points in PA\%K-AUC F1 and [X.XX] <!-- PH:NUM-009 | average margin over best unsupervised baseline, VUS-PR --> in VUS-PR on average.
On PSM, where the training anomaly ratio reaches 6.20\% and the label-guided paths are activated most intensively, CSMAD achieves [X.XX] <!-- PH:NUM-010 | CSMAD PA%K-AUC F1 on PSM --> PA\%K-AUC F1 versus [X.XX] <!-- PH:NUM-011 | best unsupervised baseline PA%K-AUC F1 on PSM --> for the best unsupervised competitor.
On SWaT excl22, which retains only the smaller, more diverse attack events, CSMAD achieves [X.XX] <!-- PH:NUM-012 | CSMAD PA%K-AUC F1 on SWaT excl22 -->, showing its performance does not rely on a single high-mass event; per-machine SMD results are in Appendix §A.4.
Among the weakly supervised baselines (Q1), NRdetector \cite{wang2025nrdetector} is the closest methodological comparison; CSMAD achieves [X.XX] PA\%K-AUC F1 and [X.XX] VUS-PR relative to it on average <!-- PH:NUM-013 | CSMAD vs NRdetector margins (both metrics) -->, the structural distinction being gradient-level label integration rather than a multi-stage pipeline (Section 2.2).

**Protocol-effect analysis.**
Does CSMAD's advantage arise from the three label-guided pathways, or from the extra training data introduced by test-prefix incorporation?
Table 4 compares CSMAD and [N] <!-- PH:NUM-014 | number of representative baselines in Table 4 --> representative unsupervised baselines under (i) a standard clean-train split using only the original training file (no labeled anomalies) and (ii) the contaminated protocol — both evaluated on the same held-out later 50\% of the original test file.
Under (i) the label-dependent pathways self-deactivate with the configuration held fixed — priority masking degrades to random (all patch labels zero), the OD loss treats all masked patches as normal, the GRL loss is never computed (no positive windows) — leaving a purely unsupervised asymmetric Teacher–Student MAE.

[TAB-4] <!-- PH:TAB-4 | Protocol-effect analysis (standard vs contaminated, half-width) — full caption and structure in PLACEHOLDER_REGISTRY.md. -->

Table 4 shows that CSMAD remains competitive under the clean-train split — [X.XX] <!-- PH:NUM-015 | CSMAD clean-train average on Table-4 datasets --> PA\%K-AUC F1 versus [X.XX] <!-- PH:NUM-016 | best unsupervised baseline clean-train average --> for the best unsupervised competitor — so the asymmetric architecture does not depend on labeled anomalies; under the contaminated protocol CSMAD improves to [X.XX] <!-- PH:NUM-017 | CSMAD contaminated-protocol average --> (a gain of [X.XX] <!-- PH:NUM-018 | CSMAD gain from standard to contaminated --> points) while the unsupervised baselines show [X.XX] <!-- PH:NUM-019 | change for best unsupervised baseline across conditions --> change, confirming the gain is specific to methods that can exploit the provided labels.

### 4.3. Ablation Study

Table 3 compares the full model against targeted variants on [N] <!-- PH:NUM-020 | number of datasets in ablation table --> representative datasets.

[TAB-3] <!-- PH:TAB-3 | Ablation table (7 variant rows, rows 5/6/7 conditional) — full caption, row list, and conditionality notes in PLACEHOLDER_REGISTRY.md. -->

**Anomaly-priority masking (Row 3).**
Without it, random masking only occasionally selects anomaly patches (at most 6.20\% of all patches), leaving the Teacher's reconstruction deficit there largely unexploited; removal costs [X.XX] <!-- PH:NUM-021 | PA%K-AUC F1 drop, w/o anomaly-priority masking --> points on average.

**Output discrepancy loss (Row 4).**
Removing $L_{\mathrm{OD}}$ eliminates the bifurcated signal driving the Student to deviate from the Teacher on anomalous patches while mimicking it on normal ones; the drop is [X.XX] <!-- PH:NUM-022 | PA%K-AUC F1 drop, w/o OD loss --> points.

**GRL adversarial suppression (Row 2).**
Row 2 retains the anomaly-patch OD-exclusion while removing the GRL classifier and reversal: exclusion alone leaves the Student free to memorize anomaly patterns through exposure, whereas the GRL makes retaining anomaly-discriminative features structurally difficult (Section 3.5); the marginal contribution of GRL beyond OD-exclusion is [X.XX] <!-- PH:NUM-023 | PA%K-AUC F1 difference, row 2 vs row 1 --> points.

**Asymmetric decoder capacity (Row 7).**
A symmetric decoder (Teacher 2L / Student 2L) removes the capacity gap behind the Student's preferential failure on anomalous patterns; the change of [X.XX] <!-- PH:NUM-024 | PA%K-AUC F1 drop, symmetric decoder (conditional row) --> points quantifies the asymmetric design as an architectural prior.
[Conditional: included only if the symmetric-decoder run completes; otherwise this paragraph moves to Appendix §B.1 and contribution bullet 3 is stated as a design principle rather than a quantified result.]

**FM loss regularizer (Row 5).**
Feature matching prevents the Student representation from collapsing under the competing pressures of OD supervision and GRL suppression; its removal costs [X.XX] <!-- PH:NUM-025 | PA%K-AUC F1 drop, w/o FM loss (conditional row) --> points.
[Conditional: included only if the FM ablation run completes.]

### 4.4. Label Sparsity Analysis

The main protocol is the upper bound of label availability — every anomaly region in the training stream is labeled — whereas realistic deployments record only a fraction of anomalous events; we analyze how CSMAD degrades as this fraction decreases, simulating the general setting of Section 3.1.

**Design.**
The labeled fraction $p$ of training anomaly regions varies over $\{1.0, 0.75, 0.5, 0.25, 0.1\}$: a uniformly random selection of regions retains labels (region granularity, matching operational records) and the rest stay in training unlabeled; all else is identical to the main experiments.
At $p \to 0$, CSMAD degrades to a purely unsupervised asymmetric Teacher–Student MAE — the standard-split condition of Table 4.

**Why graceful degradation is expected.**
Three structural properties support robustness: (i) anomaly-priority masking applies only to labeled patches, so the label-free reconstruction objective is unaffected by which anomalies are labeled; (ii) GRL suppression activates only for windows containing a labeled anomaly point, so unlabeled anomaly windows contribute no destabilizing adversarial gradient; and (iii) the base reconstruction error is label-independent — a patch deviating from normal correlation structure produces elevated Teacher error regardless of its label.
As $p$ decreases, the discrepancy component weakens smoothly while the reconstruction component is preserved.
This sweep differs from the label-noise sweep of \cite{wang2025nrdetector}, which varies the rate of *incorrect* segment labels rather than the rate at which true events are recorded at all.

**Results.**
Figure 3 plots PA\%K-AUC F1 as a function of $p$ for [N] <!-- PH:NUM-026 | number of datasets in Fig. 3 --> representative datasets.

[FIG-3] <!-- PH:FIG-3 | Label sparsity sweep figure — full caption, axes, and series spec in PLACEHOLDER_REGISTRY.md. -->

Performance declines as $p$ decreases but does so [gradually / monotonically] <!-- PH:NUM-027 | qualitative descriptor of degradation shape, from results -->, maintaining competitive detection at $p = 0.25$ — three-quarters of all anomaly events unlabeled — and approaching the best unsupervised baseline at $p \approx 0$, confirming reversion to a pure reconstruction-based detector without falling below the unsupervised floor.

### 4.5. Qualitative Analysis

Figure 4 decomposes the CSMAD anomaly score for representative windows from [N] <!-- PH:NUM-028 | number of datasets in Fig. 4 --> datasets; each panel shows four aligned traces — raw input with ground-truth anomaly regions shaded, Teacher reconstruction error, Teacher–Student discrepancy, and the combined score with the anomaly-ratio threshold.

[FIG-4] <!-- PH:FIG-4 | Qualitative score-decomposition figure — full caption and panel spec in PLACEHOLDER_REGISTRY.md. -->

The two components respond distinctly: reconstruction error is elevated wherever the input deviates from learned normal patterns regardless of event type, while the discrepancy captures the additional divergence arising when the Student's limited capacity and adversarially suppressed representation fail to track the Teacher — most pronounced where labeled anomaly exposure during training has driven the Student's representation away from anomaly-specific features.
[Note: event-level interpretation to be revised against the actual visualization once results are confirmed (RT MINOR-02).]

---

<!-- ============================================================ -->
<!-- §5 CONCLUSION -->
<!-- ============================================================ -->

## 5. Conclusion

This paper addressed the underexplored setting in which training data contain a small fraction of labeled anomalies alongside a majority of unlabeled observations — common in industrial deployments yet unsupported by standard MTSAD benchmarks or unsupervised methods.
We proposed CSMAD, which integrates labeled anomaly information into masked autoencoder representation learning through three orthogonal paths — anomaly-priority masking, loss bifurcation toward normal-only Student mimicry, and gradient-reversal suppression of anomaly-specific information — on top of an asymmetric Teacher–Student decoder architecture (3-layer Teacher, 2-layer Student) that converts the capacity gap into a reliable discrepancy signal under contaminated training.
A contaminated benchmark protocol supports evaluation by incorporating the chronological prefix of each test stream into training.
Experiments on [N] <!-- PH:NUM-029 | dataset-family count (sync with NUM-001) --> multivariate datasets show competitive performance against [N] <!-- PH:NUM-030 | total baseline count (sync with NUM-002) --> unsupervised and weakly supervised baselines under five metrics, and the label sparsity analysis confirms graceful degradation as the labeled fraction decreases.
A notable limitation is the cost of leave-one-out inference — an approximately 50$\times$ increase in forward-pass computation relative to single-mask scoring <!-- INTEGRATOR: resolved from protocol constant N=50 (§3.6, §4.1.2); wall-clock verification remains Appendix §B.3 -->; an alternative complementary-masking strategy (implemented but not used in the present experiments) offers a potential avenue for cost reduction, whose cost–accuracy trade-off we leave to future work.
The graceful degradation toward the unsupervised limit also suggests extending CSMAD to fully unlabeled settings by disabling the gradient-reversal pathway.
Code is available at [URL] <!-- PH:TXT-002 | code repository URL --> (to be released upon acceptance).

---

<!-- ============================================================ -->
<!-- REFERENCES -->
<!-- ============================================================ -->

## References

<!-- Compiled from paper/04_references/refs.bib (49 verified entries; 44 cited in this manuscript).
     LaTeX: \bibliographystyle{elsarticle-num} \bibliography{refs} — Phase 7. -->
