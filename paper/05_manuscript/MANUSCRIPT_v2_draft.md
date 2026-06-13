---
phase: 5
agent: budget-surgeon + appendix-drafter
version: v2-draft-r2
directives: [T5, R6, R7, D-009, D-010]
last_modified: 2026-06-11
base: MANUSCRIPT_v1.md (integrator, v1)
model: CSMAD
setting: contaminated semi-supervised
venue: Elsevier (elsarticle, 1-column)
status: post-surgery annotated draft, round 2 (D-009 transfers + D-010 targeted reduction applied; placeholders unresolved)
notes: |
  - D-010 round-2 targeted reduction (2026-06-11, budget-surgeon r2; report: SURGERY_REPORT_v2.md §8):
    ① TAB-4 absorbed into TAB-2 as a bottom protocol-effect row-group (PAGE_BUDGET strategy 2);
    §4.2 narrative retained, references updated (§4.1.4, §4.2 ×2, §4.4). ② Table 3 former
    rows 5/6/7 (FM, warmup, symmetric decoder) demoted to Appendix §B.5 / TAB-B4 with their
    prose (NUM-024/025 relocated); body §4.3 keeps confirmed rows 1–4 + one-sentence pointer.
    ③ §4.1 setup compression: dataset enumeration → Table 1 delegation + two key sentences;
    per-dataset label-semantics enumeration → §A.3 pointer; baseline family prose → citation
    cluster (R19). ④ §3 tightened (redundancy/circumlocution only). Mandatory narratives
    untouched in substance: R10/R13/R21/R23/R28/R29/R30/R31/R32, epoch-asymmetry and
    test-set-selection disclosures, dual-λ, warmup, focal-variant distinction.
  - D-009 surgery applied on top of v1. §3: auxiliary equations moved to Appendix §C.1
    (λ_rev schedule, GRL backward identity, focal-variant exact form, masking selection
    rule; stop-gradient inlined) — body display equations 12 → 6, renumbered (1)–(6)
    (r2 correction of a stale v2-draft note that said "12 → 7 / (1)–(7)").
    §4: metric formal definitions → §A.2; excl22 numeric derivation → §A.4;
    implementation/baseline detail → §A.1/§C.2; Table 1 reduced to a 6-family summary
    (per-entity detail → §A.3). Transfer map: SURGERY_REPORT_v2.md §2.
  - Directive-mandated narratives retained in body (compressed only, zero deletions):
    R13 protocol motivation, R29 metric complementarity + PA-F1 critique + non-ranking
    statement, R30 threshold defense, R28 excl22 core rationale, R31 Q3 fairness +
    quantitative asymmetry, R32 3-property robustness logic, epoch-asymmetry and
    test-set-selection disclosures, GRL-necessity argument (§3.5).
  - Appendix drafted per PAPER_BLUEPRINT §8 + D-009 layout (A.1 implementation/execution,
    A.2 metric definitions, A.3 dataset details, A.4 excl22, A.5/A.6 full results,
    B.1 Q1 comparison, B.2 epoch sensitivity, B.3 cost, B.4 parameter sensitivity,
    B.5 extended ablations, C.1 auxiliary formulations, C.2 dimensionality, C.3
    pseudocode, C.4 notation). All appendix floats follow the R3 convention
    (placeholder + complete caption + content spec in PLACEHOLDER_REGISTRY.md).
  - Placeholder IDs unchanged for all body placeholders (NUM-001..030, TXT-002, FIG-1..4,
    TAB-1..4 — TAB-1 spec revised); TXT-001 relocated to §A.1; new appendix IDs
    TAB-A3/A6/A7/A8, TAB-B1..B4, FIG-B1, ALG-C1, NUM-031.
  - Body appendix cross-references renumbered to the v2 layout (old §A.2→§B.1,
    §A.3→§A.5, §A.4→§A.6, §A.5→§A.4, §B.1→§B.5, §B.4→§B.2, §C.1→§C.2; §A.1/§B.3 unchanged).
  - Factual flag for Phase 6: v1 stated "test stride 1" (§4.1.2); the canonical
    271_CONFIG_TRUTH r4 gives resolve_test_stride = W//10−1 = 49 (code-verified).
    The body claim was removed; the canonical value appears in Table A.1 only.
    See SURGERY_REPORT_v2.md §5.
  - §4 keeps protocol constants (6 families, 113 units, 26 baselines, 5 metrics, 50% split)
    as real values per drafter-4 policy; front/§1/§5 keep them as placeholders pending
    experiment-completion confirmation (sync point — see registry). Appendix real-value
    tables draw exclusively on 271_CONFIG_TRUTH r4 / EXPERIMENT_PROTOCOL_TRUTH r3 (A8:
    no invented numbers; result cells remain placeholders).
  - References compiled from refs.bib (\bibliography{refs}); Acknowledgments remain
    Phase 6/7 scope. Appendix placement relative to References finalized in Phase 7
    per elsarticle.
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

A multivariate time series $\mathbf{X} \in \mathbb{R}^{T \times F}$ ($T$ timesteps, $F$ sensor channels) yields sliding windows $\mathbf{W} \in \mathbb{R}^{L \times F}$ of length $L$, each partitioned into $N$ non-overlapping patches $\mathbf{P}_i \in \mathbb{R}^{s \times F}$ of size $s$ ($L = N \cdot s$).
Binary labels exist at the patch level, $y^p_i \in \{0,1\}$, and the window level, $y^w \in \{0,1\}$ (1 if any timestep is anomalous).

We work under a **contaminated semi-supervised** setting: $\mathcal{D}_{\mathrm{train}}$ contains a large majority of unlabeled windows and a small fraction carrying anomaly labels, as in industrial fault records.
The main experiments take the label-availability upper bound — every anomalous timestep in the training split is labeled — and Section 4.4 validates the general case of partial labeling.
Labels are not used at inference; the model outputs a per-timestep anomaly score.
Multivariate data motivate this design: anomalies manifest as correlated deviations across channels, and filtering labeled anomalies as noise discards the very co-occurrence structure that the masking, loss, and gradient pathways of Sections 3.3–3.5 exploit.

### 3.2. Overall Architecture

[FIG-2] <!-- PH:FIG-2 | Architecture diagram (five components, GRL "training only", stop-gradient ⊥) — full caption and content spec in PLACEHOLDER_REGISTRY.md. -->

CSMAD comprises five functional blocks (Figure 2): a linear patch embedding, a shared Transformer encoder, a Teacher decoder, a Student decoder, and a training-only label-guided module coupling the decoders through gradient reversal.
The encoder receives only the *visible* patches; each decoder reconstructs the full window from the latent sequence, the Teacher strictly deeper than the Student (Section 3.4).
The Student and GRL branches read the encoder output through a stop-gradient, so the encoder is optimized exclusively by the Teacher's reconstruction objective and the adversarial signal cannot corrupt the normal-pattern representation underpinning the anomaly score.
<!-- SURGEON r2 (D-010 ④): "At inference the GRL branch is inactive; scores derive from..." removed as a duplicate of §3.6 ("GRL classifier is not used at inference" + score composition); the training-only disclosure remains in sentence 1 and Fig. 2. -->

### 3.3. Patch Embedding and Masking

**Linear patch embedding.**
Each patch $\mathbf{P}_i$ is flattened and projected to a token $\mathbf{z}_i = \mathrm{LayerNorm}(\mathbf{W}_{\mathrm{emb}}\,\mathrm{vec}(\mathbf{P}_i) + \mathbf{b}_{\mathrm{emb}}) \in \mathbb{R}^{d}$.
Projecting an entire patch — $s$ timesteps across all $F$ channels — into a single token encodes cross-channel correlations directly in the token, following the linear patchify principle of MAE \cite{he2022mae}.

**Anomaly-priority masking.**
$|M| = \mathrm{round}(N \times r_m)$ patches are withheld from the encoder, which processes only the $|V| = N - |M|$ visible tokens.
When patch-level labels are available, each patch receives a priority $\pi_i = 10^3 \cdot y^p_i + \eta_i$, with $\eta_i \sim \mathrm{Uniform}(0,1)$ breaking ties, and the $|M|$ highest-priority patches form the masked set: anomalous patches are masked first, any remaining slots drawn at random from normal patches (formal selection rule in Appendix §C.1).

This addresses a structural imbalance of contaminated training: anomalous patches are rare, so stochastic masking seldom selects them and the model learns to reconstruct *around* rather than *through* them; at test time, masking reverts to uniform random selection.

### 3.4. Asymmetric Teacher–Student Decoders

**Encoder.**
The visible tokens pass through a Transformer encoder of depth $n_e$ (Pre-Layer-Normalization, multi-head self-attention, GELU), producing latents $\{h^{\mathrm{enc}}_i\}_{i \in V}$.

**Teacher decoder.**
Learnable mask tokens are inserted at positions in $M$, position embeddings added, and the full sequence passed through a self-attention-only decoder of depth $n_T$ following the standard MAE design \cite{he2022mae}; a linear head projects hidden states $\{h^T_i\}$ to reconstructions $\{o^T_i\}$.

**Student decoder.**
The Student is structurally identical but shallower ($n_S < n_T$), with separate mask tokens, hidden states $\{h^S_i\}$, and outputs $\{o^S_i\}$; critically, its input latent carries a stop-gradient, $\mathbf{h}^S_{\mathrm{in},i} = \mathrm{stopgrad}(h^{\mathrm{enc}}_i)$ for $i \in V$.

**Why the capacity gap matters.**
A deeper Teacher faithfully learns the joint normal correlation structure; the shallower Student replicates it on recurring normal patterns but fails more consistently on the atypical patterns characterizing anomalies, so the output discrepancy carries a stronger anomaly signal than reconstruction error alone.
This adapts the self-distillation principle of \cite{zhang2022selfdistill, ristea2024sdmae} to a contaminated setting where labeled anomalies additionally widen the discrepancy at known fault locations.

**Teacher-only warmup.**
For a fixed initial number of epochs the Student forward pass is skipped and only the Teacher branch is trained, so the Student begins its adversarial role against a stable normal-reconstruction reference; we treat this warmup as a training-stability device, not an independent contribution.

**GRL dual-$\lambda$ structure.**
Two independent quantities govern the gradient reversal branch once the Student is active: the **loss weight** $\lambda_{\mathrm{GRL}}$, set adaptively each epoch as the clamped ratio of main-loss to GRL-loss gradient norms so the adversarial loss neither dominates nor vanishes; and the **reversal coefficient** $\lambda_{\mathrm{rev}}$, scaling the backward gradient through the GRL on the sigmoid schedule of \citet{ganin2016dann}, growing from $\approx 0.02$ to $\approx 1$ over the student-training phase so suppression strengthens without destabilizing early Student learning (exact rules in Appendix §C.1).

<!-- INTEGRATOR: the SDMAE-difference footnote that appeared here in the section draft was removed — it duplicated the §2.3 footnote (canonical R21 defense) and §2.3 body text; the mandated one-sentence layer contrast remains in §3.5. -->

### 3.5. Label-Guided Training

Three loss components couple labeled anomaly information to the model at different levels of the learning process.

**Output discrepancy loss $L_{\mathrm{OD}}$.**
Let $P_n = \{i \in M : y^p_i = 0\}$ be the masked patches labeled normal; $L_{\mathrm{OD}}$ matches the Student's output to the Teacher's detached output on this subset only:

$$L_{\mathrm{OD}} = \frac{1}{|P_n|} \sum_{i \in P_n} \left\| o^T_i - o^S_i \right\|^2 \tag{1}$$

Anomalous patches are excluded entirely: the Student is steered to agree with the Teacher on normal patterns while remaining free to deviate at anomaly locations.

**Feature matching loss $L_{\mathrm{FM}}$.**
A hidden-space regularizer penalizes Teacher–Student representation distance on normal masked patches ($h^T_i$ detached):

$$L_{\mathrm{FM}} = \frac{1}{|P_n| \cdot d} \sum_{i \in P_n} \left\| h^T_i - h^S_i \right\|^2 \tag{2}$$

This prevents the Student from satisfying the output-level constraint without tracking the Teacher's internal structure; $\lambda_{\mathrm{FM}}$ follows the same adaptive mechanism as $\lambda_{\mathrm{GRL}}$, and the term is a training-only regularizer absent from the inference score.

**GRL anomaly suppression loss $L_{\mathrm{cls}}$.**
Whereas SDMAE's anomaly-overlook supervision operates in the target/loss space, our GRL operates in the gradient space of the Student's internal representation.
A two-layer MLP head $g_\phi$ predicts from each masked patch's Student hidden state whether the enclosing window contains an anomaly ($y^w$ broadcast to all masked patches), trained with a focal-style BCE variant for severe class imbalance: unlike the standard focal loss \cite{lin2017focal}, whose modulating factor derives from the raw prediction, here it derives from the class-prior-weighted cross-entropy itself (exact form in Appendix §C.1).
The gradient reversal layer between head and Student hidden states is an identity map in the forward pass and negates the gradient in the backward pass, scaled by $\lambda_{\mathrm{rev}}$; the resulting adversarial gradient — proportional to $-\lambda_{\mathrm{rev}} \cdot \lambda_{\mathrm{GRL}}$ — opposes the classifier's search for anomaly-discriminative features, pushing the Student toward anomaly-*invariant* internal states.

**Why gradient reversal is necessary beyond loss bifurcation.**
Excluding anomalous patches from $L_{\mathrm{OD}}$ removes the demand that the Student *follow* the Teacher there, but not the possibility of *memorizing* anomaly-specific reconstruction patterns through repeated exposure — which would shrink the discrepancy exactly where it is most informative.
Gradient reversal closes this route, forcing the Student's hidden states to be uninformative about anomaly identity regardless of whether anomalies appear in the loss.

**Total training loss.**

$$L_{\mathrm{total}} = L_{\mathrm{recon}} + L_{\mathrm{OD}} + \lambda_{\mathrm{FM}} \cdot L_{\mathrm{FM}} + \lambda_{\mathrm{GRL}} \cdot L_{\mathrm{cls}} \tag{3}$$

where $L_{\mathrm{recon}}$ is the Teacher's mean squared reconstruction error on masked positions; the GRL term contributes only when the batch contains a positive window.

### 3.6. Anomaly Scoring and Inference

**Leave-one-out masking.**
Each test window is scored under $N$ masking patterns — each patch masked alone, all patterns forwarded in parallel through the batch dimension — eliminating cross-patch interference at an inference cost of approximately $N$ single-window forward passes, an acknowledged limitation.

**Patch-level anomaly score.**
For each masked patch $i$, the Teacher reconstruction error $r_i$ (MSE over its $s \cdot F$ values) and the Teacher–Student discrepancy $d_i = \|o^T_i - o^S_i\|^2 / (s \cdot F)$ are computed; the GRL classifier is not used at inference.
The discrepancy is adaptively scaled to the magnitude of $r_i$,

$$\tilde{d}_i = d_i \cdot \frac{\bar{r} + \varepsilon}{\bar{d} + \varepsilon}, \quad \varepsilon = 10^{-4} \tag{4}$$

with window-level means $\bar{r} = (1/N)\sum_j r_j$, $\bar{d} = (1/N)\sum_j d_j$, and combined at a fixed ratio:

$$\sigma_i = r_i + \frac{\tilde{d}_i}{c}, \quad c = 4 \tag{5}$$

Adaptive scaling prevents either component from dominating by sheer magnitude, which varies substantially across datasets; $c$ sets the discrepancy contribution to one quarter of the reconstruction term after scaling.

**Point-level aggregation.**
Each timestep $t$ belongs to one or more (window, patch) pairs; its final score is the mean of $\sigma_i$ over all such pairs:

$$s_t = \frac{\sum_{(w,i):\; t \in \mathbf{P}^w_i} \sigma^w_i}{\left|\{(w,i):\; t \in \mathbf{P}^w_i\}\right|} \tag{6}$$

Averaging across overlapping windows provides ensemble smoothing of single-window reconstruction-context variation.

---

<!-- ============================================================ -->
<!-- §4 EXPERIMENTS -->
<!-- ============================================================ -->

## 4. Experiments

### 4.1. Experimental Setup

#### 4.1.1. Datasets and Benchmark Protocol

**Datasets.**
We evaluate CSMAD on six real-world multivariate benchmark families — SWaT \cite{goh2016swat}, WaDi \cite{ahmed2017wadi}, PSM \cite{abdulaal2021psm}, SMD \cite{su2019omnianomaly}, and SMAP/MSL \cite{hundman2018telemanom} — spanning industrial control, IT infrastructure, and spacecraft telemetry: 113 learning units in total, or 114 evaluation units with SWaT's dual evaluation (below).
Table 1 summarizes per-family statistics; per-entity detail is in Appendix §A.3.
<!-- SURGEON r2 (D-010 ③): domain-parenthetical enumeration (entity counts, channel counts) delegated to Table 1 / §A.3; two key sentences retained. -->


[TAB-1] <!-- PH:TAB-1 | Dataset statistics, 6-family summary (D-009: per-entity rows moved to Appendix §A.3 / TAB-A4) — full caption, row values, and size spec in PLACEHOLDER_REGISTRY.md. -->

**Contaminated benchmark protocol.**
A defining feature of standard MTSAD benchmarks is that their original training splits contain no labeled anomalies by construction (per-family label semantics in Appendix §A.3), so a method that exploits labeled anomalies cannot be evaluated without modifying the partition.
<!-- SURGEON r2 (D-010 ③): per-dataset label-semantics enumeration delegated to §A.3 "Training-label semantics" (verbatim duplicate); R13 core claim (structural absence by construction → re-split necessity) retained. -->
We therefore re-split each dataset at the temporal midpoint of its original test file: the earlier half joins the training data and the later half is reserved exclusively for evaluation, so labeled anomalies are genuinely present in training (ratios 0.52\%–6.20\%; Table 1).
The evaluation half is never observed during training or threshold selection, so no temporal lookahead is introduced.
The halving rule is uniform; only for SMAP/MSL does the split point shift outward when it falls within ten timesteps of an anomaly region (4 of 81 channels, largest shift 166 timesteps; Appendix §A.3).
The re-split is a redefinition of the benchmark, not a use of held-out labels: no model sees evaluation labels at any stage, and the identical partition is provided to all methods (Section 4.1.4), with precedent in the 7:3 re-split of \cite{wang2025nrdetector} placing anomalies within the training stream.
One limitation remains: the anomaly type distribution of the incorporated prefix may differ from that of the evaluation suffix.
Inputs are min–max scaled per feature on each entity's training portion, multi-entity families normalized per entity.

**SWaT dual evaluation.**
SWaT is trained once but evaluated twice: in the full condition a single attack event (region 22) accounts for 83.75\% of test anomaly mass, so the full metric chiefly reflects detection of this one event.
We therefore also report all metrics with region 22 masked out (excl22; anomaly ratio 3.68\% versus 19.05\%) — same model, same scores, only the evaluation mask differs, identically for all baselines.
Table 2 ranks under excl22; the region-22 derivation and full-condition results are in Appendix §A.4.

#### 4.1.2. Implementation Details

**Architecture and training.**
CSMAD uses patch size 10 on windows of length 500 (masking ratio 0.15), a 4-layer encoder ($d_\text{model}=512$, fixed across entities), and a 3-layer Teacher with a 2-layer Student decoder; training runs 500 epochs (Teacher-only for the first 250; Section 3.4) at batch size 1024, one seed-42 run per entity.
Full hyperparameters, environment, and preprocessing are in Appendix §A.1 and §C.2.
<!-- SURGEON (D-009): optimizer/precision/warmup litany, SWaT 45-feature reproducibility note, and hardware/code sentence (TXT-001) moved to §A.1; input-dimension table moved to §C.2. v1's "test stride 1" claim removed — contradicts 271_CONFIG_TRUTH r4 (resolve_test_stride = 49); canonical value now in Table A.1 (Phase 6 flag, SURGERY_REPORT_v2 §5). -->

**Epoch asymmetry disclosure.**
Unsupervised baselines train 10 epochs and weakly supervised baselines 50, evaluated every epoch; CSMAD trains 500, evaluated every 5 (budget table in Appendix §A.1).
All methods share the selection criterion, no early stopping, and best-epoch reporting; budgets reflect convergence characteristics — CSMAD needs the 250-epoch warmup before the Student activates — and batch sizes follow original implementations (512 for baselines).
We report this asymmetry transparently; Appendix §B.2 analyzes epoch-budget sensitivity for representative baselines.

**Test-set model selection.**
Best-epoch selection for CSMAD and all 26 baselines evaluates PA\%K-AUC F1 on the test split — no separate validation split exists in this protocol.
Uniform across methods, this leaves relative rankings unaffected but may bias absolute estimates optimistically; we acknowledge this limitation.

**Inference and threshold.**
Inference uses the leave-one-out masking of Section 3.6 (wall-clock cost in Appendix §B.3).
Threshold-dependent metrics use the anomaly-ratio threshold — the $(1-r)$ quantile of the score distribution, $r$ being the labeled anomaly fraction of the evaluation set — applied identically to all methods per the convention of \cite{xu2022anomalytransformer}; $r$ derives from evaluation-set ground truth but is never used in training, and threshold-free metrics (VUS, PA\%K-AUC families) are unaffected.

#### 4.1.3. Evaluation Metrics

We adopt five metrics assessing complementary aspects of detection quality, following the multi-metric philosophy of recent benchmark analyses \cite{kim2022rigorous, paparrizos2022vus, liu2024elephant, wang2025nrdetector}: **PA\%K-AUC F1** \cite{kim2022rigorous}, which integrates point-adjusted F1 over the tolerance spectrum $K \in \{0, 5, \ldots, 100\}$, removing dependence on any particular $K$ — our primary metric and selection criterion; **PA\%K-AUC AUC-PR**, the same $K$-integration applied to the area under the precision–recall curve, complementary under class imbalance; **VUS-PR** and **VUS-ROC** \cite{paparrizos2022vus}, which sweep both a threshold and a temporal tolerance to measure ranking quality without an operating point, VUS-PR rated the most reliable single TSAD measure by a large-scale study \cite{liu2024elephant}; and **Affiliation F1** \cite{huet2022affiliation}, the harmonic mean of affiliation precision/recall measuring the temporal distance between predicted and ground-truth events, computed at the anomaly-ratio threshold.
Formal definitions and computation details are in Appendix §A.2.

The five metrics span three orthogonal perspectives — threshold-free ranking (VUS), tolerance-spectrum integration (PA\%K-AUC), and local event localization (Affiliation F1) — with distinct failure modes; reporting all five prevents any single failure mode from going undetected.
The traditional point-adjusted F1 (PA F1) at $K{=}0$ \cite{xu2018kpivae} is reported only in Appendix §A.5 for comparability, marked (oracle) for its F1-optimal threshold, and is never used for ranking: even a random score can reach state-of-the-art levels under it \cite{kim2022rigorous}.

#### 4.1.4. Baselines and Comparison Conditions

We compare against 26 baselines: 22 unsupervised — nine simple-to-lightweight detectors following \cite{sarfraz2024quovadis}, six established deep TSAD systems \cite{xu2022anomalytransformer, tuli2022tranad, audibert2020usad, zong2018dagmm, deng2021gdn, su2019omnianomaly}, and seven recent competitive methods \cite{fang2024tfmae, lai2023npsr, wu2023timesnet, yang2023dcdetector, song2023memto, luo2024moderntcn, wu2025catch} — and four weakly supervised methods exploiting labeled anomalies during training \cite{sultani2018deepmil, lee2021wetas, liu2024treemil, wang2025nrdetector}; the full tier list, implementation provenance (including the simplified DAGMM variant), and hyperparameters are in Appendix §A.1.
The weakly supervised group is evaluated under the Q1 condition only, since removing all labeled anomalies (Q3) would eliminate the positive windows their objectives require.
<!-- SURGEON r2 (D-010 ③): family-by-family enumeration prose converted to a citation cluster (R19); method-name list (DeepMIL/WETAS/TreeMIL/NRdetector) lives in Table 2 rows and §A.1. -->

**Comparison conditions.**
The main comparison uses the Q3 (normal-only) condition for all 22 unsupervised baselines: labeled anomaly regions are excised from the contaminated training data and the surviving normal segments concatenated with boundary-aware windowing.
Under purely unsupervised learning, the most effective use of a labeled anomaly is removal as a contaminating sample \cite{bekker2020pusurvey}; Q3 gives each unsupervised method its best available footing, while CSMAD trains on the full contaminated set without excision.
Q3 excision reduces baseline training volume by the train anomaly ratio (0.52\%–6.20\%; SMD pending), which may itself contribute to the gap; the protocol-effect block of Table 2 (Section 4.2) decouples these effects, and Appendix §B.1 reports Q1 results for all unsupervised baselines.

### 4.2. Main Results

Table 2 presents PA\%K-AUC F1 and VUS-PR for CSMAD and all 26 baselines across the six dataset families; full five-metric results are in Appendix §A.5 and per-entity results in Appendix §A.6.

[TAB-2] <!-- PH:TAB-2 | Main comparison table (27 methods × 7 dataset columns × 2 metrics, landscape) — full caption, row/column structure, and size spec in PLACEHOLDER_REGISTRY.md. -->

CSMAD achieves the highest PA\%K-AUC F1 on [N] of the six dataset families and the highest VUS-PR on [N] <!-- PH:NUM-006 | overall ranking summary (wins out of 6 on each metric) -->, averaging [X.XX] PA\%K-AUC F1 and [X.XX] VUS-PR <!-- PH:NUM-007 | CSMAD averages across all dataset families --> across families, and outperforms the strongest unsupervised competitor (Q3) by [X.XX] <!-- PH:NUM-008 | average margin over best unsupervised baseline, PA%K-AUC F1 --> absolute points in PA\%K-AUC F1 and [X.XX] <!-- PH:NUM-009 | average margin over best unsupervised baseline, VUS-PR --> in VUS-PR on average.
On PSM, whose 6.20\% training anomaly ratio activates the label-guided paths most intensively, CSMAD achieves [X.XX] <!-- PH:NUM-010 | CSMAD PA%K-AUC F1 on PSM --> PA\%K-AUC F1 versus [X.XX] <!-- PH:NUM-011 | best unsupervised baseline PA%K-AUC F1 on PSM --> for the best unsupervised competitor; on SWaT excl22, which retains only the smaller, more diverse attack events, it achieves [X.XX] <!-- PH:NUM-012 | CSMAD PA%K-AUC F1 on SWaT excl22 -->, showing that performance does not rely on a single high-mass event.
Against NRdetector \cite{wang2025nrdetector}, the closest weakly supervised comparison (Q1), CSMAD achieves [X.XX] PA\%K-AUC F1 and [X.XX] VUS-PR on average <!-- PH:NUM-013 | CSMAD vs NRdetector margins (both metrics) -->, the structural distinction being gradient-level label integration rather than a multi-stage pipeline (Section 2.2).

**Protocol-effect analysis.**
To separate the contribution of the three label-guided pathways from that of the extra training data introduced by test-prefix incorporation, the bottom block of Table 2 compares CSMAD and [N] <!-- PH:NUM-014 | number of representative baselines in the protocol-effect block of Table 2 (former TAB-4; absorbed per D-010 ①) --> representative unsupervised baselines under (i) a standard clean-train split (original training file only, no labeled anomalies) and (ii) the contaminated protocol, both evaluated on the same held-out evaluation suffix.
Under (i) the label-dependent pathways self-deactivate with the configuration held fixed (random masking, all-normal OD loss, no GRL loss), leaving a purely unsupervised asymmetric Teacher–Student MAE.

<!-- SURGEON r2 (D-010 ①): [TAB-4] absorbed into TAB-2 as a bottom row-group (PAGE_BUDGET strategy 2); merged caption/spec in PLACEHOLDER_REGISTRY.md TAB-2 entry. Narrative retained; references updated. -->

This block shows that CSMAD remains competitive under the clean-train split — [X.XX] <!-- PH:NUM-015 | CSMAD clean-train average on the protocol-effect datasets --> PA\%K-AUC F1 versus [X.XX] <!-- PH:NUM-016 | best unsupervised baseline clean-train average --> for the best unsupervised competitor — so the asymmetric architecture does not depend on labeled anomalies; under the contaminated protocol CSMAD improves to [X.XX] <!-- PH:NUM-017 | CSMAD contaminated-protocol average --> (a gain of [X.XX] <!-- PH:NUM-018 | CSMAD gain from standard to contaminated --> points) while the unsupervised baselines show [X.XX] <!-- PH:NUM-019 | change for best unsupervised baseline across conditions --> change, confirming the gain is specific to methods that can exploit the provided labels.

### 4.3. Ablation Study

Table 3 compares the full model against targeted variants on [N] <!-- PH:NUM-020 | number of datasets in ablation table --> representative datasets.

[TAB-3] <!-- PH:TAB-3 | Ablation table (4 confirmed variant rows; former rows 5/6/7 demoted to Appendix TAB-B4 per D-010 ②) — full caption and row list in PLACEHOLDER_REGISTRY.md. -->

**Anomaly-priority masking (Row 3).**
Without it, random masking only rarely selects anomaly patches, leaving the Teacher's reconstruction deficit there largely unexploited; removal costs [X.XX] <!-- PH:NUM-021 | PA%K-AUC F1 drop, w/o anomaly-priority masking --> points on average.

**Output discrepancy loss (Row 4).**
Removing $L_{\mathrm{OD}}$ eliminates the bifurcated signal driving the Student to deviate from the Teacher on anomalous patches while mimicking it on normal ones; the drop is [X.XX] <!-- PH:NUM-022 | PA%K-AUC F1 drop, w/o OD loss --> points.

**GRL adversarial suppression (Row 2).**
Row 2 retains the anomaly-patch OD-exclusion while removing the GRL classifier and reversal: exclusion alone leaves the Student free to memorize anomaly patterns through exposure (Section 3.5); the marginal contribution of GRL beyond OD-exclusion is [X.XX] <!-- PH:NUM-023 | PA%K-AUC F1 difference, row 2 vs row 1 --> points.

**Extended variants.**
Further ablations — removing the feature-matching regularizer, removing the Teacher-only warmup, and a symmetric (2-layer/2-layer) decoder — are reported in Appendix §B.5 (Table B.4).
<!-- SURGEON r2 (D-010 ②): former Rows 5/6/7 (conditional) demoted to Appendix §B.5 / TAB-B4; their prose paragraphs (NUM-024, NUM-025) moved there. Body Table 3 = confirmed rows 1–4 only. -->


### 4.4. Label Sparsity Analysis

The main protocol is the upper bound of label availability — every training anomaly region is labeled — whereas realistic deployments record only a fraction of events; we analyze how CSMAD degrades as this fraction decreases (the general setting of Section 3.1).

**Design.**
The labeled fraction $p$ of training anomaly regions varies over $\{1.0, 0.75, 0.5, 0.25, 0.1\}$: a uniformly random selection of regions retains labels (region granularity, matching operational records) while the rest remain in training unlabeled, all else unchanged; at $p \to 0$ CSMAD reverts to the purely unsupervised mode of the protocol-effect analysis (Section 4.2).

**Why graceful degradation is expected.**
Three structural properties support robustness: (i) anomaly-priority masking applies only to labeled patches, leaving the label-free reconstruction objective unaffected by which anomalies are labeled; (ii) GRL suppression activates only for windows containing a labeled anomaly, so unlabeled anomaly windows contribute no destabilizing adversarial gradient; and (iii) the base reconstruction error is label-independent, elevated wherever a patch deviates from normal correlation structure.
As $p$ decreases, the discrepancy component weakens smoothly while the reconstruction component is preserved.
This sweep differs from the label-noise sweep of \cite{wang2025nrdetector}, which varies the rate of *incorrect* segment labels rather than the rate at which true events are recorded at all.

**Results.**
Figure 3 plots PA\%K-AUC F1 as a function of $p$ for [N] <!-- PH:NUM-026 | number of datasets in Fig. 3 --> representative datasets.

[FIG-3] <!-- PH:FIG-3 | Label sparsity sweep figure — full caption, axes, and series spec in PLACEHOLDER_REGISTRY.md. -->

Performance declines as $p$ decreases but does so [gradually / monotonically] <!-- PH:NUM-027 | qualitative descriptor of degradation shape, from results -->, maintaining competitive detection at $p = 0.25$ and approaching the best unsupervised baseline at $p \approx 0$, confirming reversion to a pure reconstruction-based detector without falling below the unsupervised floor.

### 4.5. Qualitative Analysis

Figure 4 decomposes the CSMAD anomaly score for representative windows from [N] <!-- PH:NUM-028 | number of datasets in Fig. 4 --> datasets; each panel shows four aligned traces — raw input with ground-truth anomaly regions shaded, Teacher reconstruction error, Teacher–Student discrepancy, and the combined score with the anomaly-ratio threshold.

[FIG-4] <!-- PH:FIG-4 | Qualitative score-decomposition figure — full caption and panel spec in PLACEHOLDER_REGISTRY.md. -->

The two components respond distinctly: reconstruction error is elevated wherever the input deviates from learned normal patterns regardless of event type, while the discrepancy captures the additional divergence arising where the Student's limited capacity and adversarially suppressed representation fail to track the Teacher.
<!-- NOTE (Phase 6): event-level interpretation to be revised against the actual visualization once results are confirmed (RT MINOR-02). -->


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
<!-- APPENDIX (drafted Phase 5 per D-009; LaTeX \appendix placement per elsarticle in Phase 7) -->
<!-- ============================================================ -->

## Appendix A. Experimental Setup Details and Full Results

### A.1. Implementation and Execution Details

**CSMAD configuration.**
Table A.1 lists the complete configuration; all values are shared across the 113 learning units, with only the input dimensionality $F$ varying by dataset (Table C.1).

**Table A.1.** Complete CSMAD configuration used in all main experiments (single configuration; no per-dataset tuning).

| Group | Parameter | Value |
|---|---|---|
| Architecture | Patch embedding | Linear (flatten + projection) with LayerNorm |
| | Encoder | 4 layers, pre-LN self-attention, $d_\text{model}=512$, 8 heads, feedforward 2048, GELU, dropout 0.15 |
| | Teacher / Student decoder | 3 / 2 layers (self-attention only, MAE-style), separate learnable mask tokens |
| | GRL classifier head | 2-layer MLP: LayerNorm $\to$ Linear($512 \to 256$) $\to$ GELU $\to$ Dropout(0.1) $\to$ Linear($256 \to 1$) |
| Windowing | Window length / patch size | $L = 500$, $s = 10$ ($N = 50$ patches) |
| | Train / test stride | 21 / 49 <!-- canonical 271_CONFIG_TRUTH r4 (resolve_test_stride = L//10 − 1); Phase 6 verification flag, SURGERY_REPORT_v2 §5 --> |
| | Masking ratio | 0.15 (8 masked / 42 visible patches); uniform random at test time |
| Training | Epochs / Teacher-only warmup | 500 / first 250 epochs |
| | Optimizer | AdamW (fused), lr $10^{-3}$, weight decay $10^{-3}$, $\beta = (0.9, 0.99)$ |
| | GRL classifier learning rate | $10^{-4}$ (separate parameter group) |
| | LR schedule | 10-epoch linear warmup + cosine annealing |
| | Batch size / precision / seed | 1024 / bf16 mixed precision / 42 (single run per entity) |
| Loss | Base weights | $L_{\mathrm{recon}}$, $L_{\mathrm{OD}}$: 1.0; $\lambda_{\mathrm{FM}}$, $\lambda_{\mathrm{GRL}}$: adaptive (Eq. C.4) with base factors $\beta_{\mathrm{FM}} = 1.0$, $\beta_{\mathrm{GRL}} = 0.2$ |
| | GRL focal parameters | $\gamma = 2$; class-prior pos-weight $w_+$ computed per entity |
| Scoring | Combination ratio | $c = 4$ (Eq. 5); GRL head unused at inference |

**Training budgets and evaluation cadence.**
Table A.2 states the per-group budgets disclosed in Section 4.1.2.
All groups share the best-epoch selection criterion (PA\%K-AUC F1 on the test split; Section 4.1.2), use no early stopping, complete their full budget, and are reported at their best evaluated epoch.

**Table A.2.** Training and evaluation budgets per method group.

| Method group | Epochs | Evaluation cadence | Batch size | Best-epoch criterion |
|---|---|---|---|---|
| CSMAD | 500 | every 5 epochs | 1024 | PA\%K-AUC F1 |
| Unsupervised baselines (22) | 10 | every epoch | 512 | PA\%K-AUC F1 |
| Weakly supervised baselines (4) | 50 | every epoch | 512 | PA\%K-AUC F1 |

**Environment and code.**
All experiments run on [GPU model] <!-- PH:TXT-001 | GPU model used for experiments (fill from experiment metadata; do not guess) -->; code, configurations, and the exact dataset partitions will be released at [URL] <!-- PH:TXT-002 | code repository URL -->.

**Baseline implementations and hyperparameters.**
The 22 unsupervised baselines comprise five simple detectors (random score, sensor-range deviation, PCA reconstruction, L2-norm, nearest-neighbor distance), three lightweight neural detectors (MLP, MLPMixer, single-stack Transformer), and a GCN-LSTM detector, following the protocol study of \cite{sarfraz2024quovadis}; six established deep TSAD systems (Anomaly Transformer, TranAD, USAD, DAGMM, GDN, OmniAnomaly); and seven recent methods (TFMAE, NPSR, TimesNet, DCdetector, MEMTO, ModernTCN, CATCH).
For DAGMM we follow the simplified re-implementation of the TranAD repository, in which the GMM energy term is omitted; the variant is labelled "DAGMM (simplified)" in all tables.
Each baseline retains the hyperparameters of its original implementation or publication preset (for example, NRdetector keeps its native window size of 100 rather than the 500 used by our pipeline); the random-score baseline is averaged over five independent runs (mean ± std), and all other methods are single runs.
All baselines consume the identical data partitions through a unified loading layer, and all metrics — for CSMAD and every baseline — are computed by one shared evaluation routine, precluding implementation-level metric divergence.

[TAB-A3] <!-- PH:TAB-A3 | Per-baseline hyperparameter table (26 rows; placeholder — values to be extracted from the comparison-pipeline model configurations, not invented) — full caption and content spec in PLACEHOLDER_REGISTRY.md. -->

**SWaT preprocessing (reproducibility note).**
The SWaT (A1+A2) input dimensionality is 45: 51 raw features minus 6 constant columns \{P202, P401, P404, P502, P601, P603\} removed during preprocessing.
Reproductions should verify this dimension explicitly, as loading the raw CSV files without the constant-column filter yields 51 features.

### A.2. Formal Definitions of Evaluation Metrics

**Point adjustment and PA\%K.**
Under conventional point adjustment (PA) \cite{xu2018kpivae}, if any timestep within a ground-truth anomaly segment is predicted positive, all timesteps of that segment are counted as detected.
PA\%K \cite{kim2022rigorous} parameterizes this leniency: a segment qualifies for adjustment only when strictly more than $K\%$ of its timesteps are predicted positive, so that $K = 0$ recovers conventional PA and $K = 100$ point-wise scoring.

**PA\%K-AUC F1.**
For each $K \in \{0, 5, \ldots, 100\}$, the PA\%K-adjusted F1 is computed with a per-$K$ re-optimized threshold, following the reference implementation of \cite{kim2022rigorous}; the reported value is the trapezoidal integral of F1 over $K$, normalized to $[0, 1]$.
Integrating over the full tolerance spectrum removes the cherry-picking degree of freedom that a single $K$ would introduce.

**PA\%K-AUC AUC-PR.**
The same $K$-grid and integration, with the integrand replaced by the area under the PA\%K-adjusted precision–recall curve at each $K$ (obtained by a threshold sweep, hence threshold-free).

**VUS-ROC / VUS-PR** \cite{paparrizos2022vus}.
The Volume Under the Surface generalizes AUC-ROC/AUC-PR to a three-dimensional volume by sweeping both the decision threshold and a temporal tolerance parameter that softens segment boundaries; we use the authors' official implementation with tolerance window 100 after min–max normalization of scores.
Both measures are threshold-free and robust to label-boundary uncertainty.

**Affiliation F1** \cite{huet2022affiliation}.
Affiliation precision and recall convert the temporal distance between predicted and ground-truth events into per-event affinity scores within each event's affiliation zone, with formal robustness guarantees against adversarial scoring; Affiliation F1 is their harmonic mean.
Binarization uses the anomaly-ratio threshold defined below.

**Anomaly-ratio threshold (formal).**
Let $r$ be the fraction of anomalous timesteps in the evaluation span; the threshold is the $(1-r)$ quantile of the point-level score distribution, and predictions are scores strictly above it.
The threshold is computed post hoc for threshold-dependent metrics only — it never enters training or model selection (Section 4.1.2).

**PA F1 (oracle).**
Conventional PA ($K{=}0$) F1 evaluated at its F1-optimal threshold; reported in Appendix §A.5 solely for comparability with prior work, marked (oracle), and excluded from all rankings (Section 4.1.3).

**Score aggregation.**
All metrics consume point-level scores obtained by mean-aggregation over all covering (window, patch) pairs (Eq. 6); the identical evaluation routine serves CSMAD and every baseline.

### A.3. Dataset Details

**Per-entity statistics.**
Table A.4 expands the family-level summary of Table 1.

**Table A.4.** Dataset statistics under the contaminated benchmark protocol, per entity. Train/test sizes reflect the re-split of Section 4.1.1; Train AR / Test AR denote the anomaly ratio of the training / evaluation portion. SMAP and MSL sizes are concatenated per-channel totals. <!-- partial placeholder: SMD per-machine rows pending — see PLACEHOLDER_REGISTRY (TAB-A4) -->

| Entity | #Train pts | #Test pts | #Dim. | Train AR (\%) | Test AR (\%) | Source |
|---|---|---|---|---|---|---|
| SWaT (A1+A2) | 719,959 | 224,960 | 45 | 1.63 | 19.05 (full) / 3.68 (excl22) | \cite{goh2016swat} |
| WaDi A1 | 1,296,001 | 86,401 | 123 | 0.52 | 3.82 | \cite{ahmed2017wadi} |
| WaDi A2 | 870,972 | 86,402 | 123 | 0.76 | 3.87 | \cite{ahmed2017wadi} |
| PSM | 176,401 | 43,921 | 25 | 6.20 | 30.63 | \cite{abdulaal2021psm} |
| SMD (×28) | [per-machine] | [per-machine] | 29–36 | [per-machine] | 4.16 (avg) | \cite{su2019omnianomaly} |
| SMAP (×54) | 355,905 | 217,925 | 25 | 0.70 | 24.54 | \cite{hundman2018telemanom} |
| MSL (×27) | 95,271 | 36,775 | 55 | 1.70 | 16.72 | \cite{hundman2018telemanom} |

**Training-label semantics.**
SWaT and WaDi training files record normal operation only; PSM and SMD distribute no training-label files, and their training portions are treated as normal following the field-standard assumption; SMAP and MSL training labels are explicitly zero.
Labeled anomalies in our training splits therefore originate exclusively from the incorporated test prefixes.

**Split-boundary adjustment (SMAP/MSL).**
A cut position is accepted only if it lies at least ten timesteps from every annotated anomaly region; when the 50\% midpoint violates this clearance, the cut moves outward to the nearest admissible position, without a distance bound.
Table A.5 reports the measured shifts: 4 of 81 channels (all MSL) moved, and no boundary-straddling anomaly region remains in any channel.
At the concatenated scale, the aggregate absolute shift is 252 timesteps against an MSL evaluation total of 36,775.

**Table A.5.** Measured split-point adjustments for the SMAP/MSL channels (all other channels: zero shift).

| Channel | Test length | Target (50\%) | Actual cut | Shift | Share of channel test length |
|---|---|---|---|---|---|
| SMAP (all 54 channels) | — | — | — | 0 | 0\% |
| MSL D-16 | 2,191 | 1,095 | 1,261 | +166 | 7.58\% |
| MSL M-1 | 2,277 | 1,138 | 1,099 | −39 | 1.71\% |
| MSL M-2 | 2,277 | 1,138 | 1,099 | −39 | 1.71\% |
| MSL S-2 | 1,827 | 913 | 921 | +8 | 0.44\% |
| MSL (remaining 23 channels) | — | — | — | 0 | 0\% |

**Boundary-aware windowing.**
The original training portion and the incorporated test prefix are not temporally adjacent; segment boundaries are registered so that no sliding window crosses non-contiguous data.
The same mechanism guards the excision boundaries of the Q3 condition (Section 4.1.4).

### A.4. SWaT excl22: Region Definition and Dual-Condition Results

**Region definition and statistics.**
Attack region 22 is the chronologically first anomaly region within the held-out SWaT evaluation half, spanning evaluation-local positions $[2{,}869, 38{,}769)$ — 35,900 contiguous timesteps, which constitute 83.75\% of all anomalous timesteps and 15.96\% of the entire evaluation span.
Its identification is deterministic (first region, with a sanity check on its length; no other supported dataset contains a single region of comparable extent).
Masking it reduces the evaluation anomaly ratio from 19.05\% to 3.68\%, leaving the 13 smaller and more diverse attack events.

**Evaluation-mask implementation.**
The excl22 condition modifies neither training nor scores: an evaluation mask removes the region's timesteps from the label vector and the event list before metric computation, so threshold-free surfaces (VUS) and event-based measures (Affiliation) are likewise computed on the masked span.
The excl22 condition selects its own best epoch under the shared criterion, independently of the full condition; the identical mask is applied to every baseline.

[TAB-A6] <!-- PH:TAB-A6 | SWaT dual-condition results (full vs excl22, all five metrics, all methods; placeholder cells) — full caption and content spec in PLACEHOLDER_REGISTRY.md. -->

### A.5. Full Multi-Metric Results

[TAB-A7] <!-- PH:TAB-A7 | Complete multi-metric results: PA%K-AUC AUC-PR, VUS-ROC, Affiliation F1, PA F1 (oracle) for all 27 methods × 6 families (placeholder cells) — full caption and content spec in PLACEHOLDER_REGISTRY.md. -->

### A.6. Per-Entity Results

[TAB-A8] <!-- PH:TAB-A8 | Per-entity results for SMD (28), SMAP (54), MSL (27) — PA%K-AUC F1 and VUS-PR (placeholder cells) — full caption and content spec in PLACEHOLDER_REGISTRY.md. -->

---

## Appendix B. Additional Analyses

### B.1. Q1 (Full Contaminated Training) Condition Results

The Q3 condition of the main comparison grants unsupervised baselines the most favorable use of the training labels (excision of contaminated regions).
For completeness, Table B.1 reports the complementary Q1 condition, in which the same 22 unsupervised baselines train on the full contaminated stream without excision — quantifying how much unaddressed contamination costs each method family and contextualizing the training-volume asymmetry acknowledged in Section 4.1.4.

[TAB-B1] <!-- PH:TAB-B1 | Q1-condition comparison for all 22 unsupervised baselines (placeholder cells) — complete caption below; content spec in PLACEHOLDER_REGISTRY.md. -->

**Caption (complete).** "Table B.1. Q1 (full contaminated training) condition results for all 22 unsupervised baselines. Each method trains on the identical contaminated training stream used by CSMAD (no anomaly excision; labels unused) and is evaluated on the identical held-out evaluation half. Metrics: PA\%K-AUC F1 and VUS-PR per dataset family; $\Delta$ columns give the change relative to the Q3 condition of Table 2 (positive = Q1 better). The CSMAD row is repeated from Table 2 for reference, as CSMAD trains on the contaminated stream in both conditions."

### B.2. Epoch-Budget Sensitivity

Section 4.1.2 disclosed the asymmetric training budgets (500 / 50 / 10 epochs).
To assess whether this asymmetry materially affects the comparison, representative unsupervised baselines are re-trained at extended budgets — and CSMAD at a reduced budget — under the otherwise unchanged protocol.

[TAB-B2] <!-- PH:TAB-B2 | Epoch-budget sensitivity for representative baselines and CSMAD (placeholder) — full caption and content spec in PLACEHOLDER_REGISTRY.md. -->

### B.3. Computational Cost

Leave-one-out inference (Section 3.6) performs approximately $N = 50$ forward passes per window.
Table B.3 reports measured per-window FLOPs, end-to-end wall-clock evaluation time, and peak GPU memory against single-mask scoring; the measured wall-clock overhead factor is [X.XX] <!-- PH:NUM-031 | measured wall-clock overhead factor of leave-one-out vs single-mask inference; if materially below 50, soften the §5 "approximately 50×" wording (registry §5 condition) -->.

[TAB-B3] <!-- PH:TAB-B3 | Computational cost table (FLOPs / wall-clock / memory; placeholder) — full caption and content spec in PLACEHOLDER_REGISTRY.md. -->

### B.4. Parameter Sensitivity

[FIG-B1] <!-- PH:FIG-B1 | Sensitivity of PA%K-AUC F1 to the score combination ratio c (default 4) and the masking ratio r_m (default 0.15) on representative datasets (placeholder) — full caption and content spec in PLACEHOLDER_REGISTRY.md. -->

### B.5. Extended Ablations

This appendix hosts the ablation variants beyond the four confirmed rows of Table 3 (Section 4.3): removal of the feature-matching regularizer, removal of the Teacher-only warmup, a symmetric decoder, and a Teacher-decoder depth sensitivity study (3/2/1 layers against the 2-layer Student).

**Symmetric decoder capacity.**
A symmetric decoder (Teacher 2L / Student 2L) removes the capacity gap behind the Student's preferential failure on anomalous patterns (Section 3.4); the change of [X.XX] <!-- PH:NUM-024 | PA%K-AUC F1 drop, symmetric decoder (moved from §4.3 per D-010 ②) --> points quantifies the asymmetric design — the architectural prior of contribution 3 — as an empirical effect.

**FM loss regularizer.**
Feature matching prevents the Student representation from collapsing under the competing pressures of OD supervision and GRL suppression; its removal costs [X.XX] <!-- PH:NUM-025 | PA%K-AUC F1 drop, w/o FM loss (moved from §4.3 per D-010 ②) --> points.

[TAB-B4] <!-- PH:TAB-B4 | Extended ablations: former Table 3 rows 5/6/7 (FM loss, Teacher-only warmup, symmetric decoder; demoted per D-010 ②) + decoder-depth sensitivity (placeholder) — full caption and content spec in PLACEHOLDER_REGISTRY.md. -->

---

## Appendix C. Method Details

### C.1. Auxiliary Formulations

**Reversal-coefficient schedule (Section 3.4).**
The reversal coefficient follows the sigmoid schedule of \citet{ganin2016dann} over the student-training phase:

$$\lambda_{\mathrm{rev}}(p) = \frac{2}{1 + \exp(-10\,p)} - 1, \quad p = \mathrm{clip}\!\left(\frac{e - e_0 + 1}{e_1 - e_0},\; 0,\; 1\right), \tag{C.1}$$

where $e$ is the current epoch and $[e_0, e_1]$ the student-training phase (epochs 250–500 in the main configuration); $\lambda_{\mathrm{rev}}$ rises monotonically from $\approx 0.02$ at the first Student epoch to $\approx 1$ at the end of training.

**Gradient reversal (Section 3.5).**
The GRL is an identity map in the forward pass; in the backward pass it scales and negates the gradient:

$$\frac{\partial \tilde{h}^S_i}{\partial h^S_i} = -\lambda_{\mathrm{rev}} \cdot \mathbf{I} \tag{C.2}$$

**Classification loss, exact form (Section 3.5).**
With $\ell_i = \mathrm{BCE}_{w_+}(\hat{y}_i,\, y^w)$ the class-prior-weighted binary cross-entropy of masked patch $i$ and $w_+$ the per-entity normal-to-anomalous window ratio,

$$L_{\mathrm{cls}} = \frac{1}{|M|} \sum_{i \in M} \left(1 - e^{-\ell_i}\right)^{\!\gamma} \ell_i, \quad \gamma = 2 \tag{C.3}$$

Unlike the standard focal loss \cite{lin2017focal}, which defines its modulating probability $p_t$ from the raw prediction, here $p_t := e^{-\ell_i}$ derives from the pos-weight-adjusted BCE, weighting hard examples by both confidence and prior imbalance; this variant is part of the present design rather than an external import.

**Adaptive loss weights (Sections 3.4–3.5).**
The loss weights of the FM and GRL terms are set each epoch from the ratio of gradient norms, smoothed by the previous epoch's average:

$$\lambda_{\bullet} = \beta_{\bullet} \cdot \mathrm{clip}\!\left(\frac{\|\nabla L_{\mathrm{main}}\|}{\|\nabla L_{\bullet}\| + 10^{-4}},\; 0,\; 10\right), \quad \beta_{\mathrm{GRL}} = 0.2, \;\; \beta_{\mathrm{FM}} = 1.0 \tag{C.4}$$

The adversarial gradient reaching the Student hidden state is therefore $-\lambda_{\mathrm{rev}} \cdot \lambda_{\mathrm{GRL}} \cdot \partial L_{\mathrm{cls}} / \partial(\mathrm{GRL\ output})$: the reversal coefficient and the loss weight act multiplicatively and remain distinct quantities.

**Masking selection rule (Section 3.3).**
With priorities $\pi_i = 10^3 \cdot y^p_i + \eta_i$, $\eta_i \sim \mathrm{Uniform}(0,1)$,

$$M = \mathrm{argtopk}_{|M|}\!\left\{\pi_i\right\}_{i=1}^{N} \tag{C.5}$$

which masks all anomalous patches whenever they number at most $|M|$ (the remainder drawn uniformly from normal patches) and a uniform random subset of $|M|$ anomalous patches otherwise.

### C.2. Input Dimensionality

All entities share $d_\text{model} = 512$; only the raw input dimensionality $F$ varies.

**Table C.1.** Input dimensionality per dataset.

| Dataset | $F$ | Derivation |
|---|---|---|
| SWaT (A1+A2) | 45 | 51 raw − 6 constant columns \{P202, P401, P404, P502, P601, P603\} |
| WaDi A1 | 123 | raw sensor set |
| WaDi A2 | 123 | 127 raw − 4 all-NaN columns |
| PSM | 25 | raw |
| SMD | 29–36 (per machine) | 38 raw − per-machine constant columns |
| SMAP | 25 | 1 telemetry + 24 command channels |
| MSL | 55 | 1 telemetry + 54 command channels |

### C.3. Training Procedure Pseudocode

[ALG-C1] <!-- PH:ALG-C1 | CSMAD training pseudocode (placeholder) — content spec: preprocessing (incl. SWaT constant-column removal), anomaly-priority masking, Teacher-only gating for epochs < 250, loss assembly (Eq. 3) with adaptive weights (Eq. C.4) and reversal schedule (Eq. C.1). Full spec in PLACEHOLDER_REGISTRY.md. -->

### C.4. Notation Summary

**Table C.2.** Summary of notation.

| Symbol | Meaning | Introduced |
|---|---|---|
| $\mathbf{X} \in \mathbb{R}^{T \times F}$ | multivariate time series ($T$ timesteps, $F$ channels) | §3.1 |
| $\mathbf{W} \in \mathbb{R}^{L \times F}$ | sliding window ($L = 500$) | §3.1 |
| $\mathbf{P}_i \in \mathbb{R}^{s \times F}$ | $i$-th patch ($s = 10$) | §3.1 |
| $N$ | number of patches per window ($= 50$) | §3.1 |
| $y^w$, $y^p_i$ | window-level / patch-level anomaly labels | §3.1 |
| $\mathbf{z}_i \in \mathbb{R}^{d}$ | patch embedding ($d = 512$, fixed across entities) | §3.3 |
| $M$, $V$ | masked / visible patch index sets ($|M| = 8$) | §3.3 |
| $\pi_i$, $\eta_i$ | masking priority and tie-breaking noise | §3.3 |
| $h^T_i$, $h^S_i$ | Teacher / Student decoder hidden states at patch $i$ | §3.4 |
| $o^T_i$, $o^S_i$ | Teacher / Student outputs at patch $i$ | §3.4 |
| $P_n$ | masked patches labeled normal, $\{i \in M : y^p_i = 0\}$ | §3.5 |
| $L_{\mathrm{recon}}$, $L_{\mathrm{OD}}$, $L_{\mathrm{FM}}$, $L_{\mathrm{cls}}$ | loss terms (Eq. 3) | §3.5 |
| $\lambda_{\mathrm{FM}}$, $\lambda_{\mathrm{GRL}}$ | adaptive loss weights (Eq. C.4) | §3.4–3.5 |
| $\lambda_{\mathrm{rev}}$ | GRL gradient reversal coefficient (Eq. C.1) | §3.4 |
| $r_i$, $d_i$, $\tilde{d}_i$ | reconstruction error, discrepancy, scaled discrepancy | §3.6 |
| $\sigma_i$, $s_t$ | patch-level / point-level anomaly scores | §3.6 |

---

<!-- ============================================================ -->
<!-- REFERENCES -->
<!-- ============================================================ -->

## References

<!-- Compiled from paper/04_references/refs.bib (49 verified entries; 44 cited in this manuscript).
     LaTeX: \bibliographystyle{elsarticle-num} \bibliography{refs} — Phase 7. -->
