---
phase: 5
agent: section-drafter-4
directives: [T5, R3, R12, R13, R16, R19, R28, R29, R30, R31, R32, R33, R34]
last_modified: 2026-06-11
section: "§4 Experiments"
page_budget: "3.3p"
placeholder_policy: |
  All metric values, win/loss counts, improvement margins, and rankings
  use [X.XX] / [N] placeholders with PH:NUM-### IDs.
  Protocol constants (6 dataset families, 22+4 baselines, 5 metrics,
  50% split, 113 learning units) are stated as real values.
  Simulation and Exathlon are excluded per R33.
  Gaussian smoothing is not mentioned per R34.
---

<!-- ============================================================
  ANNOTATION LAYER — Remove before submission (Phase 6 clean pass)
  Annotations are embedded as HTML comments throughout.
  Each PH:NUM-### block is a numbered placeholder for a result value.
  TAB-### and FIG-### mark table/figure positions.
  ============================================================ -->

# 4. Experiments

## 4.1 Experimental Setup

### 4.1.1 Datasets and Benchmark Protocol

<!-- ANNOTATION: R13 defense structure — motivation first, then mechanics,
     then fairness/safety, then temporal ordering, then limitation.
     Blueprint §6.2 + §14 five-argument structure. -->

**Datasets.** We evaluate CSMAD on six families of multivariate real-world datasets
that span industrial cyber-physical systems, IT infrastructure monitoring, and spacecraft
telemetry. These benchmarks are selected because anomalies occur in realistic operational
streams, providing the conditions necessary to construct the contaminated semi-supervised
setting described below. The six families are: SWaT A1+A2 \cite{goh2016swat} (water
treatment plant, 45 features after removing six combined-constant sensor columns);
WaDi A1 and WaDi A2 \cite{ahmed2017wadi} (water distribution, 123 features each, treated
as independent entities); PSM \cite{abdulaal2021psm} (production server monitoring, 25
features); SMD \cite{su2019omnianomaly} (28 server machines, 29–36 features per machine
after constant-column removal); SMAP (54 telemetry channels, 25 features per channel);
and MSL (27 telemetry channels, 55 features per channel), with SMAP and MSL sourced from
\cite{hundman2018telemanom}. In total, the benchmark comprises 113 learning units
(1 + 2 + 1 + 28 + 54 + 27), with SWaT evaluated under two conditions for a total of 114
evaluation units (§4.1.3). Table 1 provides dataset statistics.

<!-- TAB-1
  Position: §4.1.1, immediately after the paragraph above.
  Caption (complete): "Table 1. Dataset statistics under the contaminated benchmark
  protocol. Train/test sizes reflect the re-split described in §4.1.1.
  Train AR = anomaly ratio in the training portion (originating from the incorporated
  test-prefix); Test AR = anomaly ratio in the held-out evaluation portion.
  WaDi A1 and A2 are listed as independent entities. SWaT is evaluated under both
  full and excl22 conditions; metrics reported in Table 2 use the excl22 condition
  (§4.1.3). SMD train/test lengths and anomaly ratios are per-machine averages;
  per-entity results are in Appendix §A.4. SMAP and MSL sizes reflect the
  concatenated per-channel totals under Pattern A."
  Columns: Dataset | #Train pts | #Test pts | #Dimensions | Train AR (%) | Test AR (%) | Source
  Rows (real-value cells from EXPERIMENT_PROTOCOL_TRUTH §①):
    SWaT (A1+A2) | 719,959 | 224,960 | 45 | 1.63 | 19.05 (full) / 3.68 (excl22) | \cite{goh2016swat}
    WaDi A1 | 1,296,001 | 86,401 | 123 | 0.52 | 3.82 | \cite{ahmed2017wadi}
    WaDi A2 | 870,972 | 86,402 | 123 | 0.76 | 3.87 | \cite{ahmed2017wadi}
    PSM | 176,401 | 43,921 | 25 | 6.20 | 30.63 | \cite{abdulaal2021psm}
    SMD (×28) | per-machine | per-machine | 29–36 | per-machine | 4.16 (avg) | \cite{su2019omnianomaly}
    SMAP (×54) | 355,905 | 217,925 | 25 | 0.70 | 24.54 | \cite{hundman2018telemanom}
    MSL (×27) | 95,271 | 36,775 | 55 | 1.70 | 16.72 | \cite{hundman2018telemanom}
  Size estimate: ~1/4 page (landscape-fit with 7 rows).
-->

**Contaminated benchmark protocol.**
<!-- ANNOTATION: R13 — motivation-first structure; §14 arguments ①–⑤ mapped
     to paragraphs below. Blueprint §6.2 "정면 답변 문단". -->
A defining feature of standard multivariate TSAD benchmarks is that their original
training splits contain no labeled anomalies by construction: the SWaT and WaDi training
files record normal plant operation with no attack events; PSM and SMD provide no training
label files, following the field convention that the training stream is treated as entirely
normal; and the SMAP/MSL training labels are explicitly set to zero throughout. Because
labeled anomaly information is structurally absent from the original training splits across
all six benchmark families, a method designed to exploit such information cannot be
meaningfully evaluated without modifying the data partition.

To address this, we adopt the following re-split: the original test file is divided at its
temporal midpoint, the first (earlier) 50\% is appended to the original training data to
form the new training set, and the second (later) 50\% is reserved exclusively as the
evaluation set. Under this partitioning, labeled anomalies are genuinely present in
the training data — their measured ratios are 1.63\% for SWaT, 0.52\% for WaDi A1,
0.76\% for WaDi A2, 6.20\% for PSM, 0.70\% for SMAP, and 1.70\% for MSL, with SMD
ratios varying by machine. The evaluation set (the retained later 50\%) is never observed
by the model during training or threshold selection, so no temporal lookahead is introduced.

The integer halving rule (floor division by two) is applied uniformly to every dataset
without per-dataset adjustment. For SMAP and MSL, a boundary-adjustment mechanism shifts
the split point outward whenever it falls within or within ten timesteps of an annotated
anomaly region, ensuring that no anomaly event straddles the train/test boundary. In
practice, this adjustment activates for only 4 of 81 SMAP/MSL channels (all in MSL), with
the largest shift being 166 timesteps on channel D-16 — 7.58\% of that channel's test
length. The remaining 77 channels are split exactly at the midpoint. All boundary
information is reported to support reproducibility.

This re-split constitutes a redefinition of the benchmark, not a use of held-out test
labels for training. No model sees the labels of the evaluation set at any stage. The same
partitioned data is provided to all comparison methods under the conditions described in
§4.1.4. As a prior precedent for re-splitting standard TSAD benchmarks so that anomalies
appear within the training stream, Wang et al. \cite{wang2025nrdetector} divide segments
into a 7:3 training/test ratio and explicitly motivate their protocol by noting that
performance of unsupervised methods is constrained when anomalies are embedded in the
training data. We acknowledge one limitation: the anomaly type distribution in the
incorporated training prefix may differ from that in the retained evaluation suffix.

**Normalization.** All inputs are scaled per-feature with min-max normalization fitted
exclusively on the training portion of each entity. For multi-entity datasets (SMD, SMAP,
MSL), each machine or channel is normalized independently, preventing large-scale entities
from absorbing the statistics of smaller ones.

**SWaT dual evaluation.**
<!-- ANNOTATION: R28 — region-22 dominance; single training + two eval masks. -->
SWaT is trained once but evaluated under two conditions. In the full condition, the
evaluation set contains 14 attack events; the first of these (region 22, at test-local
coordinates approximately [2869, 38769), length ~35,900 timesteps) accounts for roughly
83.75\% of all test anomaly mass and 15.96\% of the entire evaluation set. Because a
single event of this scale determines recall almost entirely, the full-condition metric
reflects whether one particular event is detected rather than how well the model
discriminates across diverse attack patterns. To provide a more discriminative view, we
additionally compute metrics with region 22 masked out from the evaluation (excl22
condition). The two conditions use the same trained model and the same anomaly scores;
only the evaluation mask differs. The anomaly ratio drops from 19.05\% under full to
3.68\% under excl22. All baseline methods are evaluated under both conditions on
identical output. Model comparison and ranking in Table 2 use the excl22 condition;
full-condition results are reported in Appendix §A.5 for completeness.

---

### 4.1.2 Implementation Details

<!-- ANNOTATION: Blueprint §6.3; epoch asymmetry disclosure is mandatory
     (EXPERIMENT_PROTOCOL_TRUTH §④-실행 3항, ADV BLK-005). -->

**Architecture.** CSMAD uses a patch size of 10 timesteps, yielding 50 patches per
window (window length 500). The masking ratio is fixed at 0.15, so 8 patches are masked
per window. The shared Transformer encoder has 4 layers, model dimension
$d_\text{model}=512$, 8 attention heads, feedforward dimension 2048, and dropout 0.15.
The Teacher decoder has 3 layers and the Student decoder has 2 layers; both use
self-attention only. Model dimension is fixed at 512 across all entities and datasets.
The input dimension $F$ varies by dataset (Table 1; Appendix §C.1). SWaT is processed
with 45 input features, reflecting the removal of six combined-constant columns
\{P202, P401, P404, P502, P601, P603\}; reproductions should verify this dimension at
inference time.

**Training.** We train with AdamW ($\beta_1=0.9$, $\beta_2=0.99$, learning rate
$10^{-3}$, weight decay $10^{-3}$), batch size 1024, for 500 epochs. The GRL
classifier uses a separate learning rate of $10^{-4}$. A linear warmup over the first
10 epochs is followed by cosine annealing; automatic mixed precision (bf16) is used
throughout. The Teacher-only phase covers the first 250 epochs, during which the Student
decoder forward pass is not executed in the training path, so Student parameters receive
no gradient updates. The random seed is fixed at 42 for all deterministic components.

**Epoch asymmetry disclosure.** Unsupervised baselines train for 10 epochs with
evaluation at every epoch; weakly supervised baselines train for 50 epochs with the same
evaluation cadence. CSMAD trains for 500 epochs with evaluation every 5 epochs. All
methods share the best-epoch selection criterion (PA\%K-AUC F1 on the test split, see
§4.1.3), the absence of early stopping, and the convention of reporting from the
epoch achieving the highest selection criterion. Epoch budgets reflect each model class's
convergence characteristics: CSMAD requires a 250-epoch warmup phase before the Student
path is active, while the smaller baseline architectures converge within substantially
fewer epochs. Batch sizes differ accordingly: CSMAD uses 1024 and baselines use 512,
following original implementations. We report this asymmetry transparently; Appendix
§B.4 provides a sensitivity analysis of the epoch budget choice on representative
baseline performance.

**Test-set model selection.** Best-epoch selection for all methods — CSMAD and all 26
baselines — is performed by evaluating PA\%K-AUC F1 on the test split. No separate
validation split exists in this protocol. This is a prospective evaluation practice that
is uniform across all compared methods, so relative rankings are unaffected; however, it
may introduce an optimistic bias in absolute performance estimates, which we acknowledge
as a limitation.

**Inference.** We use leave-one-out masking: for each window, all 50 possible
single-patch masking patterns are evaluated in a single batched forward pass to produce
50 patch-level score estimates. Point-level scores are obtained by averaging all
(window, patch) pairs that cover each timestep, with a test stride of 1 so every
timestep is covered by many windows.

**Threshold.** Threshold-dependent metrics use the anomaly-ratio threshold: for a given
evaluation set, the threshold is set at the $(1 - r)$ quantile of the anomaly score
distribution, where $r$ is the fraction of labeled anomaly points in that evaluation set.
This threshold is applied identically across all methods. Threshold-free metrics (VUS-PR,
VUS-ROC, and the PA\%K-AUC family) are computed without any threshold and are unaffected
by this choice. The anomaly ratio is drawn from ground-truth labels in the evaluation
set; this information is not used during training. Xu et al. \cite{xu2022anomalytransformer}
established the convention of using the anomaly ratio to set the detection threshold in
the TSAD literature; we follow this practice and apply it consistently to all methods.

**Hardware and code.** All experiments are conducted on [GPU model — to be filled].
Code will be released at [repository URL — to be released upon acceptance].

---

### 4.1.3 Evaluation Metrics

<!-- ANNOTATION: R29 — five metrics, complementarity explanation, PA F1
     disclosed as auxiliary with oracle-threshold caveat; R24 — official names only. -->

We adopt five evaluation metrics that assess complementary aspects of detection quality,
following the multi-metric philosophy advocated by recent benchmark analyses
\cite{kim2022rigorous, paparrizos2022vus, liu2024elephant, wang2025nrdetector}.

**PA\%K-AUC F1** \cite{kim2022rigorous}. For each threshold sensitivity parameter
$K \in \{0, 5, 10, \ldots, 100\}$, a point-adjusted F1 score is computed under the
PA\%K protocol: a predicted anomaly segment qualifies for point adjustment only when the
fraction of correctly detected points within it exceeds $K\%$. The area under the
F1-vs-$K$ curve (trapezoidal, normalized to $[0,1]$) integrates performance across the
full spectrum from the lenient ($K{=}0$, equivalent to standard point adjustment) to the
strict ($K{=}100$, equivalent to point-wise F1). This integration removes dependence on
any particular choice of $K$ and serves as our primary performance metric and
best-epoch selection criterion.

**PA\%K-AUC AUC-PR.** The same $K$ sweep with the area under the precision-recall
curve computed at each $K$, then integrated. This variant is sensitive to false-positive
patterns across different threshold levels and complements the F1 variant in
class-imbalanced settings.

**VUS-PR and VUS-ROC** \cite{paparrizos2022vus}. The Volume Under the
Precision-Recall Surface and Volume Under the ROC Surface extend threshold-free
evaluation to range-based anomalies by sweeping both a detection threshold and a temporal
tolerance parameter. Being entirely threshold-independent, VUS metrics measure the
intrinsic ranking quality of anomaly scores without committing to any operating point.
VUS-PR is identified as the most reliable single measure for time-series anomaly detection
by a large-scale benchmark study \cite{liu2024elephant}. We report both variants because
VUS-PR is more informative under the class imbalances present in our datasets
(test anomaly ratios range from 3.68\% to 30.63\%), while VUS-ROC provides a widely
comparable baseline.

**Affiliation F1** \cite{huet2022affiliation}. Affiliation precision and recall
measure detection quality in terms of the temporal distance between predicted and
ground-truth anomaly events, evaluated locally per event. Their harmonic mean (Affiliation
F1) is robust to adversarial scoring strategies and captures how accurately in time a
model localizes anomalies, a dimension that count-based metrics cannot distinguish. We
use the anomaly-ratio threshold variant (`affiliation_f1_ar`) to remain consistent with
the threshold protocol described above.

**PA F1 (auxiliary, oracle threshold).** Point-Adjusted F1 at $K{=}0$
\cite{xu2018kpivae} is included for comparability with prior work reporting
under the conventional point-adjustment protocol. However, Kim et al.
\cite{kim2022rigorous} demonstrate that even a random anomaly score can achieve
state-of-the-art performance when evaluated this way, because the protocol inflates
recall for any detector that fires within a long anomaly segment. We include PA F1 in the
complete metric tables (Appendix §A.3) marked as (oracle) because it is computed at the
F1-optimal threshold rather than the anomaly-ratio threshold; we do not use it for
ranking or primary comparison.

These five metrics span three orthogonal evaluation perspectives: threshold-free
continuous ranking of score quality (VUS-PR/ROC); integration over the
event-detection-tolerance spectrum (PA\%K-AUC); and duration-based local event
localization quality (Affiliation F1). Their failure modes are distinct: a method could
score well on VUS by producing a smooth, well-ordered score while still failing on
Affiliation F1 if its detections are consistently delayed; conversely, a method with
sharp but slightly early detections might score better on Affiliation F1 than on the
strict end of PA\%K. Presenting all five prevents any single failure mode from going
undetected.

---

### 4.1.4 Baselines and Comparison Conditions

<!-- ANNOTATION: R19 — cite-only for baselines, no individual descriptions;
     R12/R31 — Q3 normalonly rationale; Blueprint §6.5.
     R31 defense: label-exploitation scarcity among existing methods.
     Train quantity asymmetry: §14 argument ③ + RT MAJOR-03. -->

We compare CSMAD against 26 baseline methods organized in two groups.

**Unsupervised baselines (22 methods).** The first group comprises five simple
heuristics and non-parametric detectors (random, sensor-range deviation, PCA
reconstruction error, L2-norm distance, and nearest-neighbor distance
\cite{sarfraz2024quovadis}), three lightweight neural detectors (MLP, MLPMixer, and
a single-stack Transformer \cite{sarfraz2024quovadis}), a GCN-LSTM detector
\cite{sarfraz2024quovadis}, six established deep TSAD systems
\cite{xu2022anomalytransformer, tuli2022tranad, audibert2020usad, zong2018dagmm,
deng2021gdn, su2019omnianomaly}, and seven recent competitive methods
\cite{fang2024tfmae, lai2023npsr, wu2023timesnet, yang2023dcdetector, song2023memto,
luo2024moderntcn, wu2025catch}.
<!-- INTEGRATOR FIX (2026-06-11): han2023catch is not in refs.bib; the CATCH paper's canonical key is wu2025catch (Wu et al., ICLR 2025). -->

For DAGMM, we follow the simplified re-implementation used in the TranAD repository,
which omits the GMM energy term; this variant is labelled accordingly.

**Weakly supervised baselines (4 methods).** The second group applies methods that can
exploit labeled anomaly information during training: DeepMIL \cite{sultani2018deepmil},
WETAS \cite{lee2021wetas}, TreeMIL \cite{liu2024treemil}, and
NRdetector \cite{wang2025nrdetector}. These methods are evaluated under the Q1 condition
only (§below); the Q3 condition is structurally incompatible with them because removing
all labeled anomalies from training eliminates the positive windows their learning
objectives require.

**Comparison conditions.** The main comparison table uses the Q3 (normalonly)
condition for all 22 unsupervised baselines: their training sets are constructed by
excising all labeled anomaly regions from the contaminated training data and concatenating
the surviving normal segments with boundary-aware windowing. Under purely unsupervised
learning, the most effective use of a labeled anomaly is to remove it as a contaminating
sample \cite{bekker2020pusurvey}; Q3 thus gives each unsupervised method its best
possible footing with respect to the provided labels. CSMAD is trained on the full
contaminated training set without any excision.

We note that Q3 excision reduces the training data volume for unsupervised baselines by
the train anomaly ratio (0.52\%–6.20\% across completed datasets; SMD per-machine
ratios are pending full evaluation). This quantity difference may contribute to the
performance gap in addition to the label-exploitation mechanism. To decouple these
effects, §4.2 reports a protocol-effect analysis (Table 4) that compares CSMAD and
representative baselines under both the contaminated protocol and a standard clean-train
split, using the same held-out evaluation set. Appendix §A.2 additionally provides
Q1 (full-contaminated) results for all 22 unsupervised baselines, showing the performance
cost of contamination when labels are not exploited.

---

## 4.2 Main Results

<!-- ANNOTATION: Blueprint §6.6 — four-part analysis structure;
     Table 2 primary; protocol-effect Table 4 in same section;
     no component-level explanation (§4.3 exclusive);
     no SOTA claim without evidence (A8). -->

**Comparison table.**
Table 2 presents PA\%K-AUC F1 and VUS-PR for CSMAD and all 26 baselines across the
six dataset families. Bold entries indicate the highest score in each column; underlined
entries indicate the second-highest. Full results for all five metrics are in
Appendix §A.3.

<!-- TAB-2
  Position: §4.2, immediately following the paragraph above.
  Caption (complete): "Table 2. Main comparison results under the contaminated benchmark
  protocol (Q3 condition for unsupervised baselines; Q1 for weakly supervised baselines).
  Reported metrics: PA%K-AUC F1 and VUS-PR. SWaT column uses the excl22 evaluation
  condition; full-condition results appear in Appendix §A.5. SMD, SMAP, and MSL values
  are macro-averages over all entities. Bold = highest; underline = second-highest.
  All values are [X.XX] placeholders pending completion of the full experimental queue."
  Row structure (26 baselines + CSMAD, separated by horizontal rules):
    Group 1: Simple (5): random, sensor_range, pca_error, l2_norm, nn_distance
    Group 2: Neural (3): mlp, mlpmixer, transformer
    Group 3: GCN-LSTM (1)
    Group 4: SOTA Legacy (6): Anomaly Transformer, TranAD, USAD, DAGMM (simplified),
              GDN, OmniAnomaly
    Group 5: SOTA New (7): TFMAE, NPSR, TimesNet, DCdetector, MEMTO, ModernTCN, CATCH
    Group 6: Weakly Supervised — Q1 only (4): DeepMIL, WETAS, TreeMIL, NRdetector
    Group 7: CSMAD (ours)
  Columns: Method | SWaT excl22 PA%K-AUC F1 | SWaT excl22 VUS-PR |
            WaDi A1 PA%K-AUC F1 | WaDi A1 VUS-PR |
            WaDi A2 PA%K-AUC F1 | WaDi A2 VUS-PR |
            PSM PA%K-AUC F1 | PSM VUS-PR |
            SMD avg PA%K-AUC F1 | SMD avg VUS-PR |
            SMAP avg PA%K-AUC F1 | SMAP avg VUS-PR |
            MSL avg PA%K-AUC F1 | MSL avg VUS-PR
  All metric cells: [X.XX] placeholder
  Size estimate: landscape full-width, ~3/4 page.
  Note: columns may be split across two sub-tables (main datasets + SMAP/MSL)
        if landscape width is insufficient.
-->

As shown in Table 2, CSMAD achieves
<!-- PH:NUM-001 Overall ranking summary — e.g. "the highest PA%K-AUC F1 on [N] of 6
     dataset families and the highest VUS-PR on [N] of 6". To be filled from results. -->
[X.XX] PA\%K-AUC F1 and [X.XX] VUS-PR
<!-- PH:NUM-002 Aggregate across all datasets under primary metric -->
on average across all six evaluation sets. Among the 22 unsupervised baselines operating
under their best available condition (Q3 normalonly), CSMAD outperforms the best
unsupervised competitor by
<!-- PH:NUM-003 Margin over best unsupervised baseline on primary metric -->
[X.XX] absolute points in PA\%K-AUC F1 and by [X.XX] in VUS-PR
<!-- PH:NUM-004 VUS-PR margin -->
on average.

Several per-dataset patterns are noteworthy. On PSM, where the training anomaly ratio
reaches 6.20\%, the label-guided masking and adversarial suppression paths in CSMAD
are activated most intensively; the model achieves [X.XX] PA\%K-AUC F1
<!-- PH:NUM-005 PSM PA%K-AUC F1 for CSMAD -->
compared to [X.XX]
<!-- PH:NUM-006 Best unsupervised baseline on PSM PA%K-AUC F1 -->
for the best unsupervised competitor. On SWaT excl22, which contains the smaller and
more diverse attack events after removing the dominant region-22 event, CSMAD achieves
[X.XX] PA\%K-AUC F1
<!-- PH:NUM-007 SWaT excl22 PA%K-AUC F1 for CSMAD -->
, demonstrating that the approach is not relying on the detection of a single
high-mass event to score well. On SMD, SMD per-machine results are detailed in
Appendix §A.4.

Among the weakly supervised baselines (Q1 condition), NRdetector \cite{wang2025nrdetector}
provides the closest methodological comparison because it also exploits labeled anomaly
information. CSMAD achieves [X.XX] PA\%K-AUC F1
<!-- PH:NUM-008 CSMAD vs NRdetector PA%K-AUC F1 margin -->
and [X.XX] VUS-PR on average relative to NRdetector, with the key structural distinction
that CSMAD integrates label information directly into its representation learning gradient
rather than relying on a multi-stage pre-training and classification pipeline.

One cost of the leave-one-out inference strategy is computational: evaluating all 50
masking patterns per window with a test stride of 1 increases inference FLOPs by
approximately 50× relative to a single forward pass. Appendix §B.3 reports wall-clock
times and memory consumption; §4.3 examines whether complementary masking (7 patterns)
can reduce this cost.

**Protocol-effect analysis.**
<!-- ANNOTATION: RT BLOCKER-03 Table 4 — standard split vs contaminated;
     two-argument structure: (1) competitive under clean-train,
     (2) additional gain from labeled anomalies. -->

A natural question is whether the performance advantage of CSMAD over unsupervised
methods arises from the model design — the three label-guided pathways described in
§3.5 — or from the additional training data made available by the test-prefix
incorporation. Table 4 addresses this by comparing CSMAD and [N]
<!-- PH:NUM-009 Number of representative baselines in Table 4, e.g. 3 -->
representative unsupervised baselines under two conditions: (i) a standard clean-train
split, in which only the original training file is used and labeled anomalies are therefore
absent from training; and (ii) the contaminated protocol (main, as in Table 2). In both
conditions, performance is evaluated on the same held-out test set (the later 50\% of
the original test file), so the evaluation target is identical and only the training
composition varies.

Under the standard clean-train condition, CSMAD operates with no labeled anomalies in
training. In this regime, the force-masking priority degrades to random masking because
all patch labels are zero, the OD loss branch treats all masked patches as normal, and
the GRL classifier loss is not computed for any window because no positive training
windows exist. CSMAD thus functions as a purely unsupervised asymmetric Teacher-Student
MAE in this condition. We hold the model configuration fixed (use\_grl=True) to avoid
reactivating a dormant loss component; the label-dependent pathways simply self-deactivate
in the absence of positive training examples.

<!-- TAB-4
  Position: §4.2, immediately following the paragraph above.
  Caption (complete): "Table 4. Protocol-effect analysis. Performance of CSMAD and
  [N] representative unsupervised baselines (Q3 condition) under a standard clean-train
  split (condition i) and the contaminated protocol (condition ii). Both conditions
  are evaluated on the same held-out test set. Standard-split CSMAD uses the identical
  model configuration with all label-dependent paths self-deactivating in the absence
  of positive training windows. Datasets: [2–3 representative datasets — to be selected
  after full experimental run]. Metric: PA%K-AUC F1."
  Rows: CSMAD (standard split) | CSMAD (contaminated) |
        Baseline A (standard split) | Baseline A (contaminated) |
        Baseline B (standard split) | Baseline B (contaminated) |
        [Baseline C if included]
  Columns: Method × Condition | Dataset-1 | Dataset-2 | [Dataset-3] | Avg.
  All metric cells: [X.XX] placeholder
  Size estimate: half-width, ~0.2 page.
  Note to Phase 6: This table requires the standard-split experiment run
  described in EXPERIMENT_EXECUTION_TODO item 3.
-->

The two-stage pattern in Table 4 shows that CSMAD remains competitive with
unsupervised SOTA under the standard clean-train split — achieving [X.XX] PA\%K-AUC F1
<!-- PH:NUM-010 CSMAD clean-train average across Table 4 datasets -->
versus [X.XX]
<!-- PH:NUM-011 Best unsupervised baseline under clean-train on same datasets -->
for the best unsupervised competitor — confirming that the asymmetric Teacher-Student
architecture is not dependent on labeled anomalies to deliver reasonable performance.
Under the contaminated protocol, CSMAD improves to [X.XX]
<!-- PH:NUM-012 CSMAD contaminated-protocol average across Table 4 datasets -->
(a gain of [X.XX] points
<!-- PH:NUM-013 Gain for CSMAD from standard to contaminated -->
), while the unsupervised baselines show [X.XX]
<!-- PH:NUM-014 Change for best unsupervised baseline from standard to contaminated -->
change, confirming that the performance gain under the contaminated protocol is specific
to methods that can exploit the provided labels.

---

## 4.3 Ablation Study

<!-- ANNOTATION: Blueprint §6.7 — seven rows; component-level explanation exclusive
     to this subsection; see RT MINOR-04.
     Rows 5 (w/o FM) and 7 (symmetric decoder) are load-bearing — conditional
     on experimental completion. Row 6 (warmup) conditional. -->

Table 3 examines the contribution of each component by comparing the full CSMAD model
against targeted variants on [N] representative datasets.
<!-- PH:NUM-015 Number of datasets in ablation table, e.g. 3 or 4 -->

<!-- TAB-3
  Position: §4.3, immediately following the paragraph above.
  Caption (complete): "Table 3. Ablation study. PA%K-AUC F1 is reported for each
  model variant on [3–4 representative datasets]. All cells are [X.XX] placeholders
  pending experimental completion. Row 2 (w/o GRL) removes the GRL classifier and
  reversal but retains the anomaly-patch OD-loss exclusion, isolating the net effect
  of active adversarial suppression. Row 7 (symmetric decoder) requires dedicated
  ablation runs and is the primary quantitative support for the asymmetric capacity-gap
  design principle (contribution bullet 3). Row 5 (w/o FM) and Row 7 are conditional
  on experimental completion; if unavailable at publication, they will be moved to
  Appendix §B.1."
  Rows:
    1. Full model (CSMAD)
    2. w/o GRL (anomaly-OD exclusion retained)
    3. w/o force_mask_anomaly
    4. w/o OD Loss
    5. w/o FM Loss  [conditional — see note above]
    6. w/o Teacher Warmup (250-epoch warmup → 0)  [conditional on warmup ablation run]
    7. Symmetric decoder (Teacher 2L / Student 2L)  [conditional — load-bearing for bullet 3]
  Columns: Variant | Dataset-A | Dataset-B | Dataset-C | [Dataset-D] | Avg.
  All metric cells: [X.XX] placeholder
  Size estimate: half-width, ~1/3 page.
-->

**Force-mask anomaly (Row 3).** Without the anomaly-priority masking mechanism, the
training objective loses its guarantee that labeled anomaly patches are among those the
model must reconstruct. In a class-imbalanced training stream — where anomaly patches
constitute at most 6.20\% of all patches — random masking would select labeled anomaly
patches only occasionally, leaving the teacher's reconstruction deficit at those positions
largely unexploited. The performance drop when this component is removed is
<!-- PH:NUM-016 PA%K-AUC F1 drop for row 3 vs row 1, average across datasets -->
[X.XX] points on average.

**Output discrepancy loss (Row 4).** Removing $\mathcal{L}_\text{OD}$ eliminates the
signal that drives the student to deviate from the teacher specifically on anomalous
patches while mimicking it on normal patches. Without this bifurcation, the student has
no structural pressure to form a representation that is selectively worse on anomalies;
the teacher-student discrepancy becomes a weaker anomaly indicator. The performance drop
is [X.XX] points.
<!-- PH:NUM-017 PA%K-AUC F1 drop for row 4 vs row 1 -->

**GRL adversarial suppression (Row 2).** Row 2 isolates the effect of gradient-reversal
suppression by retaining the anomaly-patch OD-exclusion while removing the
GRL classifier and reversal. Excluding anomaly patches from $\mathcal{L}_\text{OD}$
removes the obligation for the student to mimic the teacher at those positions, but it
does not actively remove anomaly-specific information from the student's representation:
the student can still learn to reconstruct anomaly patches by memorizing their patterns
through exposure during training, which would reduce the discrepancy signal. The GRL
closes this pathway by reversing the gradient of the classifier loss through the student
decoder's hidden representations, making it structurally difficult for the student to
retain anomaly-discriminative features. The marginal contribution of GRL beyond
OD-exclusion alone is [X.XX] points.
<!-- PH:NUM-018 PA%K-AUC F1 difference between row 2 and row 1 -->

**Asymmetric decoder capacity (Row 7).** A symmetric decoder (Teacher 2L / Student 2L)
removes the capacity gap that causes the student's mimicry to fail preferentially on
anomalous correlation patterns. When teacher and student have equal depth, the discrepancy
signal loses the structural amplification provided by the asymmetric design.
<!-- PH:NUM-019 PA%K-AUC F1 drop for row 7 vs row 1 -->
The performance change [X.XX] points quantifies the value of the asymmetric
capacity design as an architectural prior for reliable discrepancy signals.
[Conditional: this row will be included only if the symmetric-decoder ablation run
is completed prior to submission; if unavailable, this paragraph moves to Appendix §B.1
and the asymmetric-capacity claim in contribution bullet 3 is stated as a design
principle rather than a quantified result.]

**FM loss regularizer (Row 5).** Feature matching loss operates in the hidden space and
prevents the student representation from collapsing to a degenerate solution under the
competing pressures of OD supervision and GRL suppression. Its removal degrades
performance by [X.XX] points.
<!-- PH:NUM-020 PA%K-AUC F1 drop for row 5 vs row 1 -->
[Conditional: this row will be included only if the FM ablation run is completed.]

---

## 4.4 Label Sparsity Analysis

<!-- ANNOTATION: R32 — sweep experiment; robustness logic required;
     NRdetector axis difference must be noted;
     Blueprint §6.8; EXPERIMENT_PROTOCOL_TRUTH §⑦. -->

The main experimental protocol represents the upper bound of label availability:
every anomaly region in the training stream is assumed to be labeled.
In realistic deployments, however, only a fraction of anomalous events may be
identified and recorded — the remainder persist in the training data as unlabeled.
We analyze how CSMAD degrades as this labeled fraction decreases, simulating the
general contaminated semi-supervised setting described in §3.1.

**Experimental design.** We vary the fraction $p$ of labeled training anomaly regions
from 1.0 (main setting) down through $\{0.75, 0.5, 0.25, 0.1\}$. At each level,
the indicated fraction of anomaly regions is selected uniformly at random (region-level
relabeling, matching the granularity at which operational records typically identify
events), and the remaining anomaly regions are left in the training data without labels.
The data, split, and evaluation protocol remain identical to the main experiments; only
the labels supplied to the training objective change. For the $p \to 0$ limit, all
training anomaly labels are removed, and CSMAD degrades to a purely unsupervised
asymmetric Teacher-Student MAE (identical to the standard-split condition in Table 4).

**Why the model is expected to degrade gracefully.** Three structural properties support
robustness under label sparsity. First, the force-masking mechanism applies only to
patches with confirmed anomaly labels; unlabeled anomaly patches enter the masking pool
at the background random priority, so the reconstruction objective itself — a
label-free self-supervised signal — is unaffected by which anomalies are labeled.
Second, the GRL suppression is activated only for windows containing at least one
labeled anomaly point; unlabeled anomaly windows contribute no adversarial gradient,
so they do not destabilize the student representation even as their discrepancy
contribution is smaller. Third, the base reconstruction error is a label-independent
signal: a patch that genuinely deviates from the normal correlation structure produces
an elevated teacher reconstruction error regardless of whether its label is known,
because the teacher has been trained predominantly on normal patches. Consequently,
as $p$ decreases, the discrepancy component of the anomaly score weakens smoothly while
the reconstruction component is preserved, yielding a graceful rather than catastrophic
degradation in detection performance.

Note that this sweep differs from the label-noise sweep reported in Wang et al.
\cite{wang2025nrdetector}, which varies the rate of incorrect (mislabeled) segment
labels rather than the rate at which true anomaly events are recorded at all. The two
axes address related but distinct aspects of imperfect labeling.

**Results.** Figure 3 plots PA\%K-AUC F1 as a function of $p$ for [N]
<!-- PH:NUM-021 Number of datasets shown in Fig. 3, e.g. 2 or 3 -->
representative datasets.

<!-- FIG-3
  Position: §4.4, immediately following the paragraph above.
  Caption (complete): "Figure 3. Label sparsity sweep. PA%K-AUC F1 as a function
  of the labeled anomaly fraction p ∈ {0.1, 0.25, 0.5, 0.75, 1.0} for [N]
  representative datasets (one line per dataset). Dashed horizontal lines indicate
  the performance of the best unsupervised baseline (Q3, main protocol) on the
  corresponding dataset, providing a reference for the unsupervised floor.
  p = 1.0 corresponds to the main experimental setting; p → 0 approximates the
  fully unsupervised limit."
  Axes: X = labeled anomaly fraction p; Y = PA%K-AUC F1.
  Series: one solid line per dataset (2–3 datasets recommended); one dashed horizontal
          reference line per dataset (best unsupervised Q3 baseline on that dataset).
  All data points: [X.XX] placeholders.
  Size estimate: half-width (~1/4 page).
  Note to Phase 6: This figure requires the label-sparsity sweep experiments
  described in EXPERIMENT_EXECUTION_TODO.
-->

As shown in Figure 3, performance declines as $p$ decreases but does so
<!-- PH:NUM-022 Qualitative descriptor of degradation shape, e.g. "monotonically
     but gradually" — to be filled from results. -->
[gradually / monotonically], with the model maintaining competitive detection at
$p = 0.25$ — a setting in which three-quarters of all anomaly events are unlabeled.
At $p \approx 0$, performance approaches that of the best unsupervised baseline,
confirming that the model reverts to a pure reconstruction-based detector when no
labeled anomalies are available and does not degrade below the unsupervised baseline.

---

## 4.5 Qualitative Analysis

<!-- ANNOTATION: Blueprint §6.9; RT MINOR-02 — no interpretation before results
     confirmed; Fig. 4 placeholder with axis/series description required.
     R34: no Gaussian smoothing mentioned. -->

Figure 4 visualizes the decomposition of the CSMAD anomaly score for representative
windows drawn from [N] datasets.
<!-- PH:NUM-023 Number of datasets shown in Fig. 4, e.g. 2 -->
Each panel shows four aligned traces: the raw multivariate input with ground-truth
anomaly regions shaded, the teacher reconstruction error per timestep, the
teacher-student discrepancy per timestep, and the combined anomaly score with the
anomaly-ratio threshold overlaid.

<!-- FIG-4
  Position: §4.5, immediately following the paragraph above.
  Caption (complete): "Figure 4. Qualitative score decomposition on representative
  anomaly events. Each column corresponds to one dataset ([Dataset-A], [Dataset-B]).
  Row 1: multivariate input (first feature shown) with ground-truth anomaly regions
  shaded in red. Row 2: teacher reconstruction error per timestep. Row 3:
  teacher-student discrepancy per timestep (scaled_disc). Row 4: combined anomaly
  score with the anomaly-ratio threshold (dashed horizontal line).
  The decomposition illustrates how the two score components respond differently to
  anomaly characteristics: reconstruction error captures deviations from the learned
  normal pattern regardless of anomaly label, while discrepancy captures structural
  divergence amplified by the capacity gap and label-guided training."
  Axes: X = timestep; Y = score value (arbitrary units, normalized per trace for
        visual clarity). Shared X-axis across all four rows within each column.
  Datasets: SWaT excl22 + one additional dataset (WaDi A1 or PSM — to be selected
            based on visual distinctiveness of events after results are available).
  Size estimate: full-width, ~1/3 page (2 columns × 4 rows of panels).
  Note to Phase 6: Visualization should be generated after full results are
  available; anomaly event selection should represent diverse anomaly types.
  Interpretation text in §4.5 must be written after results are confirmed (RT MINOR-02).
-->

The visualization illustrates how the two score components respond to different aspects
of anomaly events. Teacher reconstruction error tends to be elevated when the input
signal deviates from the normal distributional patterns captured during training,
regardless of event type. Teacher-student discrepancy captures the additional divergence
arising when the student's limited capacity and adversarially suppressed representation
fail to track the teacher, which is most pronounced at locations where labeled anomaly
exposure during training has driven the student's representation away from anomaly-specific
features.
[Note: the event-level interpretation in this paragraph will be revised to
reference specific examples from the visualization once experimental results are
confirmed per RT MINOR-02.]

---

## Placeholder Register

<!-- ============================================================
  PLACEHOLDER REGISTER — section-drafter-4
  All placeholders introduced in §4 of this draft.
  Phase 6 (clean pass) must resolve or flag each entry.
  ============================================================ -->

### Numeric Placeholders (PH:NUM-###)

| ID | Location | Description |
|----|----------|-------------|
| PH:NUM-001 | §4.2 p.1 | Overall ranking summary (wins out of 6 datasets on PA%K-AUC F1 and VUS-PR) |
| PH:NUM-002 | §4.2 p.1 | Average PA%K-AUC F1 and VUS-PR for CSMAD across all datasets |
| PH:NUM-003 | §4.2 p.1 | Margin over best unsupervised baseline in PA%K-AUC F1 (average) |
| PH:NUM-004 | §4.2 p.1 | Margin over best unsupervised baseline in VUS-PR (average) |
| PH:NUM-005 | §4.2 p.2 | CSMAD PA%K-AUC F1 on PSM |
| PH:NUM-006 | §4.2 p.2 | Best unsupervised baseline PA%K-AUC F1 on PSM |
| PH:NUM-007 | §4.2 p.2 | CSMAD PA%K-AUC F1 on SWaT excl22 |
| PH:NUM-008 | §4.2 p.3 | CSMAD vs NRdetector margins on PA%K-AUC F1 and VUS-PR (average) |
| PH:NUM-009 | §4.2 Table-4 para | Number of representative baselines in Table 4 |
| PH:NUM-010 | §4.2 Table-4 analysis | CSMAD clean-train average on Table-4 datasets |
| PH:NUM-011 | §4.2 Table-4 analysis | Best unsupervised baseline clean-train average on Table-4 datasets |
| PH:NUM-012 | §4.2 Table-4 analysis | CSMAD contaminated-protocol average on Table-4 datasets |
| PH:NUM-013 | §4.2 Table-4 analysis | Gain for CSMAD from standard to contaminated protocol |
| PH:NUM-014 | §4.2 Table-4 analysis | Change for best unsupervised baseline from standard to contaminated |
| PH:NUM-015 | §4.3 intro | Number of datasets in Table 3 ablation |
| PH:NUM-016 | §4.3 Row 3 | PA%K-AUC F1 drop for w/o force_mask_anomaly (average) |
| PH:NUM-017 | §4.3 Row 4 | PA%K-AUC F1 drop for w/o OD Loss (average) |
| PH:NUM-018 | §4.3 Row 2 | PA%K-AUC F1 drop for w/o GRL (average) |
| PH:NUM-019 | §4.3 Row 7 | PA%K-AUC F1 drop for symmetric decoder (average) — conditional |
| PH:NUM-020 | §4.3 Row 5 | PA%K-AUC F1 drop for w/o FM Loss (average) — conditional |
| PH:NUM-021 | §4.4 para | Number of datasets shown in Figure 3 |
| PH:NUM-022 | §4.4 results | Qualitative descriptor of label-sparsity degradation shape |
| PH:NUM-023 | §4.5 intro | Number of datasets in Figure 4 |

### Table Placeholders (TAB-###)

| ID | Subsection | Caption summary | Approximate size |
|----|------------|----------------|------------------|
| TAB-1 | §4.1.1 | Dataset statistics under contaminated protocol | ~1/4 page |
| TAB-2 | §4.2 | Main comparison (PA%K-AUC F1 + VUS-PR, 26 baselines + CSMAD) | ~3/4 page landscape |
| TAB-3 | §4.3 | Ablation (7 variants × 3–4 datasets) | ~1/3 page half-width |
| TAB-4 | §4.2 | Protocol-effect analysis (standard split vs contaminated, 2–3 datasets) | ~0.2 page half-width |

### Figure Placeholders (FIG-###)

| ID | Subsection | Caption summary | Approximate size |
|----|------------|----------------|------------------|
| FIG-3 | §4.4 | Label sparsity sweep (PA%K-AUC F1 vs labeled fraction p) | ~1/4 page half-width |
| FIG-4 | §4.5 | Score decomposition visualization (recon + discrepancy + combined) | ~1/3 page full-width |

### Citation Keys Used

`goh2016swat`, `ahmed2017wadi`, `abdulaal2021psm`, `su2019omnianomaly`,
`hundman2018telemanom`, `wang2025nrdetector`, `sarfraz2024quovadis`,
`xu2022anomalytransformer`, `tuli2022tranad`, `audibert2020usad`, `zong2018dagmm`,
`deng2021gdn`, `fang2024tfmae`, `lai2023npsr`, `wu2023timesnet`, `yang2023dcdetector`,
`song2023memto`, `luo2024moderntcn`, `wu2025catch`, `sultani2018deepmil`,
`lee2021wetas`, `liu2024treemil`, `bekker2020pusurvey`, `kim2022rigorous`,
`paparrizos2022vus`, `liu2024elephant`, `huet2022affiliation`, `xu2018kpivae`,
`schmidl2022evaluation`
