---
phase: 5
agent: section-drafter-3
directives: [T5, R5, R9, R10, R17, R21, R22, R23, R24, R27, R35]
last_modified: 2026-06-11
---

# §3 Methodology

## 3.1 Problem Formulation and Setting

Let $\mathbf{X} \in \mathbb{R}^{T \times F}$ denote a multivariate time series of $T$ timesteps and $F$ sensor channels.
During training, a sliding window of length $L$ is applied to produce windows $\mathbf{W} \in \mathbb{R}^{L \times F}$.
Each window is partitioned into $N$ non-overlapping patches of size $s$, so that $L = N \cdot s$.
The $i$-th patch is $\mathbf{P}_i = \mathbf{W}[is{:}(i{+}1)s,\;:] \in \mathbb{R}^{s \times F}$.
Binary labels are available at both the patch level, $y^p_i \in \{0,1\}$, and the window level, $y^w \in \{0,1\}$ (1 if any timestep in the window is anomalous, 0 otherwise).

We work under a **contaminated semi-supervised** setting: the training corpus $\mathcal{D}_{\mathrm{train}}$ contains a large majority of unlabeled windows together with a small fraction of windows that carry anomaly labels.
This reflects the practical reality of cyber-physical and industrial systems, where fault records and maintenance logs provide sparse but reliable anomaly annotations while most of the operational stream remains unannotated.
The main experiments represent the label-availability upper bound of this setting — every anomalous timestep in the training split is labeled — and Section~4.4 validates the model under the general case in which only a fraction of anomalies carry labels.
At inference, labels are not used; the model outputs an anomaly score for each timestep in a label-free manner.

The reason this three-path integration is necessary for multivariate data specifically is that anomalies in high-dimensional sensor streams frequently manifest as correlated deviations across several channels simultaneously.
A model that treats labeled anomalies merely as noise to be filtered loses the spatial and temporal co-occurrence structure that distinguishes true faults from single-channel noise.
Encoding that structure directly into the reconstruction objective, the loss bifurcation, and the gradient-space suppression allows the model to exploit inter-channel correlations as a detection signal rather than an obstacle.


## 3.2 Overall Architecture

[FIG-2]
<!-- INTEGRATOR FIX (2026-06-11): architecture figure is Fig. 2 per blueprint numbering (PAGE_BUDGET §3); Fig. 1 is the §1 setting-comparison diagram. "four functional blocks" → "five" to match the figure's five component regions. -->

CSMAD consists of five functional blocks that operate in sequence during training: a linear patch embedding layer, a shared Transformer encoder, a Teacher decoder, a Student decoder, and a label-guided training module that couples the two decoders through a gradient reversal mechanism.
Figure~2 illustrates the full pipeline; the gradient reversal branch is marked as active during training only.

The shared encoder receives only the *visible* (unmasked) patches and produces a sequence of latent vectors.
Both the Teacher decoder and the Student decoder then receive this latent sequence, but each also receives its own learnable mask token and positional embedding before reconstructing the full window.
The Teacher decoder is strictly deeper than the Student decoder; the architectural gap is deliberate and is discussed in Section~3.4.

A key isolation constraint governs gradient flow: the Student decoder takes the encoder's output through a stop-gradient operation, so losses computed on the Student's predictions cannot propagate back through the encoder.
The same stop-gradient applies to the GRL branch.
Consequently, the shared encoder is optimized exclusively by the Teacher's reconstruction objective, which prevents the adversarial training signal from corrupting the normal-pattern representation that underpins the anomaly score.

At inference, the GRL branch is inactive.
No labels are used; the model produces anomaly scores from the Teacher's reconstruction error and the Teacher–Student output discrepancy alone (Section~3.6).


## 3.3 Patch Embedding and Masking

**Linear patch embedding.**
Each patch $\mathbf{P}_i$ is flattened into a vector of dimension $s \times F$ and projected to an embedding of dimension $d$ by a learned linear map followed by LayerNorm:

$$\mathbf{z}_i = \mathrm{LayerNorm}\!\left(\mathbf{W}_{\mathrm{emb}}\,\mathrm{vec}(\mathbf{P}_i) + \mathbf{b}_{\mathrm{emb}}\right), \quad \mathbf{z}_i \in \mathbb{R}^{d} \tag{1}$$

Projecting an entire patch — comprising $s$ timesteps and all $F$ channels — into a single token means the embedding captures joint temporal–spatial structure within that patch in a single linear operation.
Channel-wise correlations that are typical of sensor networks are therefore reflected directly in the token representation, without the inductive bias that a convolutional kernel would impose on their spatial arrangement.
This design follows the linear patchify principle of masked autoencoders \cite{he2022mae}, adapted here to the multivariate time-series domain.

**Anomaly-priority masking.**
A fixed fraction of patches is masked before the encoder: $|M| = \mathrm{round}(N \times r_m)$ patches are withheld, while the remaining $|V| = N - |M|$ visible patches enter the encoder.
Masking is performed *after* the encoder stage: the encoder processes the $|V|$ visible tokens, and mask tokens are inserted into the full-length sequence just before each decoder.

When patch-level anomaly labels are available, the masking procedure assigns elevated priority to anomalous patches.
Concretely, a priority score $\pi_i = 10^3 \cdot y^p_i + \eta_i$ is computed for each patch, where $\eta_i \sim \mathrm{Uniform}(0,1)$ breaks ties among patches of the same class.
The $|M|$ patches with the highest priority are then selected as the masked set $M$:

$$M = \mathrm{argtopk}_{|M|}\!\left\{\pi_i\right\}_{i=1}^{N} \tag{2}$$

If the number of anomalous patches within the window does not exceed $|M|$, all anomalous patches are masked and the remainder is drawn at random from normal patches.
Otherwise, $|M|$ anomalous patches are chosen at random from those available.

This anomaly-priority mechanism addresses a structural imbalance inherent to contaminated training.
Without it, the stochastic masking process rarely selects anomalous patches — because they constitute only a small fraction of each window — and the model learns to reconstruct around them rather than through them.
By forcing the model to reconstruct from within the anomalous region, the priority mechanism prevents the reconstruction objective from developing a blind spot at exactly the locations that matter most for detection.
At test time, when no labels are available, the masking reverts to uniform random selection; priority information is used only during training.


## 3.4 Asymmetric Teacher–Student Decoders

**Encoder.**
The visible patch tokens pass through a Transformer encoder of depth $n_e$, with Pre-Layer-Normalization, multi-head self-attention, and GELU activations.
The encoder output for the visible set $V$ is the latent sequence $\{h^{\mathrm{enc}}_i\}_{i \in V}$.

**Teacher decoder.**
The full-length sequence is reconstructed by inserting learnable mask tokens at the positions in $M$ and adding position embeddings to all positions before passing through a Transformer decoder of depth $n_T$.
This decoder uses self-attention only, following the standard MAE decoder design \cite{he2022mae}.
The Teacher's hidden states $\{h^T_i\}_{i=1}^{N}$ are projected to patch-reconstruction outputs $\{o^T_i\}_{i=1}^{N}$ by a linear head.

**Student decoder.**
The Student decoder has the same structure but is shallower (depth $n_S < n_T$) and uses a separate set of mask tokens.
Critically, the Student's input latent is the encoder output with a stop-gradient applied:

$$\mathbf{h}^S_{\mathrm{in},i} = \mathrm{stopgrad}\!\left(h^{\mathrm{enc}}_i\right), \quad i \in V \tag{3}$$

The Student decoder produces hidden states $\{h^S_i\}_{i=1}^{N}$ and outputs $\{o^S_i\}_{i=1}^{N}$.

**Why the capacity gap matters for multivariate time series.**
A deeper Teacher, having more modeling capacity, can faithfully learn the joint normal correlation structure across all $F$ sensor channels.
When the shallower Student attempts to replicate this representation, it succeeds on recurring normal patterns but fails more consistently on the atypical correlation patterns that characterize anomalies.
The resulting output discrepancy between Teacher and Student therefore carries a stronger anomaly signal than either reconstruction error alone.
This design adapts the self-distillation principle — introduced by \citet{zhang2022selfdistill} and applied to video anomaly detection by \citet{ristea2024sdmae}$^1$ — to the contaminated semi-supervised setting in multivariate time series, where labeled anomalies provide an additional mechanism to widen the discrepancy precisely at known fault locations.

**Teacher-only warmup.**
During the first phase of training, the Student decoder's forward pass is skipped entirely in the training path: the loss is computed on the Teacher branch alone, and no gradient reaches the Student parameters.
This continues for a fixed number of epochs, after which the Student decoder is activated and trained jointly.
The warmup ensures that the Teacher has established a reliable normal-reconstruction reference before the Student begins its adversarial role; a Student trained against an unstable Teacher would receive a noisy discrepancy signal that could impair convergence.
We treat this warmup as a training stability device rather than an independent contribution.

**GRL dual-$\lambda$ structure.**
Once the Student decoder is active, two distinct scaling quantities govern the gradient reversal branch.
First, the **loss weight** $\lambda_{\mathrm{GRL}}$ scales the classification loss term before it enters the total loss sum; it is set adaptively each epoch as the ratio of the main-loss gradient norm to the GRL-loss gradient norm, clamped to a finite range and multiplied by a fixed coefficient.
This prevents the adversarial loss from dominating or vanishing relative to the reconstruction objective.
Second, the **reversal coefficient** $\lambda_{\mathrm{rev}}$ scales the gradient during backpropagation through the gradient reversal layer; it follows the sigmoid schedule of \citet{ganin2016dann},

$$\lambda_{\mathrm{rev}}(p) = \frac{2}{1 + \exp(-10\,p)} - 1, \quad p = \mathrm{clip}\!\left(\frac{e - e_0 + 1}{e_1 - e_0},\; 0,\; 1\right), \tag{4}$$

where $e$ is the current epoch and $[e_0,\, e_1]$ is the student-training phase.
At the start of the student phase $\lambda_{\mathrm{rev}} \approx 0.02$ and it increases monotonically to $\approx 1$ by the final epoch, so the adversarial suppression strength grows gradually and does not destabilize the Student's early learning.
These two quantities are independent: $\lambda_{\mathrm{GRL}}$ controls how much the classification loss contributes to the total, while $\lambda_{\mathrm{rev}}$ controls the polarity and strength of the gradient that the GRL injects into the Student decoder.

---

$^1$We adopt the self-distillation terminology of \citet{ristea2024sdmae} and adapt its architectural paradigm to multivariate time series.
Unlike SDMAE, where the student decoder branches off from the teacher decoder after its first transformer block \cite{ristea2024sdmae}, our Teacher and Student decoders are independent modules that share only the encoder output.
The Student additionally receives a gradient-reversal suppression signal that operates in the gradient space of the Student's internal representation — a mechanism absent from SDMAE's unsupervised video setting.


## 3.5 Label-Guided Training

Three loss components couple labeled anomaly information to the model, each targeting a different level of the learning process.

**Output discrepancy loss $L_{\mathrm{OD}}$.**
Let $P_n = \{i \in M : y^p_i = 0\}$ denote the subset of masked patches that are labeled normal.
The output discrepancy loss minimizes the mean squared difference between the Teacher's output (detached) and the Student's output, restricted to this normal subset:

$$L_{\mathrm{OD}} = \frac{1}{|P_n|} \sum_{i \in P_n} \left\| o^T_i - o^S_i \right\|^2 \tag{5}$$

Anomalous patches are excluded from this term entirely.
This steers the Student toward close agreement with the Teacher on normal patterns, while making no such demand on anomalous positions — the Student is therefore free to deviate from the Teacher at anomaly locations.

**Feature matching loss $L_{\mathrm{FM}}$.**
Beyond the output level, a hidden-space regularizer penalizes the distance between Teacher and Student internal representations on normal masked patches:

$$L_{\mathrm{FM}} = \frac{1}{|P_n| \cdot d} \sum_{i \in P_n} \left\| h^T_i - h^S_i \right\|^2 \tag{6}$$

where $h^T_i$ is detached.
This loss discourages the Student's hidden representation from drifting far from the Teacher's representation space on normal data, which would otherwise allow the Student to find alternative representations that satisfy the output-level constraint without genuinely tracking the Teacher's internal structure.
Its weight $\lambda_{\mathrm{FM}}$ is set adaptively by the same gradient-norm ratio mechanism as $\lambda_{\mathrm{GRL}}$.
The feature matching loss is a training-only regularizer and does not contribute to the inference anomaly score.

**GRL anomaly suppression loss $L_{\mathrm{cls}}$.**
SDMAE's anomaly-overlook supervision operates in the target/loss space, guiding the model to reconstruct frames without their anomalous content; our GRL operates in the gradient space of the Student's internal representation.
Specifically, a two-layer MLP classifier head $g_\phi$ — consisting of LayerNorm, a hidden linear layer, GELU activation, dropout, and an output linear layer — is applied independently to each masked patch's Student hidden state to predict whether the enclosing window contains an anomaly.
The window-level label $y^w$ is broadcast to all masked patches as the classification target.

The classifier is trained with a focal-style BCE variant that we design to account for the severe class imbalance between normal and anomalous windows:

$$L_{\mathrm{cls}} = \frac{1}{|M|} \sum_{i \in M} \left(1 - e^{-\ell_i}\right)^{\!2} \ell_i \tag{7}$$

where $\ell_i = \mathrm{BCE}_{w_+}(\hat{y}_i,\, y^w)$ is the class-prior-weighted binary cross-entropy for patch $i$, and $w_+$ is the per-entity ratio of normal to anomalous windows derived from the training data.
This differs from the standard focal loss of \citet{lin2017focal}, which defines $p_t$ from the raw model prediction; here $p_t := e^{-\ell_i}$ is derived from the pos-weight-adjusted BCE, making the hard-example weight sensitive to both prediction confidence and the prior imbalance.

The gradient reversal layer sits between the classifier head and the Student hidden states.
During the forward pass it is an identity function; during backpropagation it negates the incoming gradient and multiplies it by $\lambda_{\mathrm{rev}}$:

$$\frac{\partial \tilde{h}^S_i}{\partial h^S_i} = -\lambda_{\mathrm{rev}} \cdot \mathbf{I} \tag{8}$$

The adversarial gradient that reaches the Student hidden state is therefore $-\lambda_{\mathrm{rev}} \cdot \lambda_{\mathrm{GRL,eff}} \cdot \partial L_{\mathrm{cls}}/\partial(\mathrm{GRL\;output})$, where $\lambda_{\mathrm{GRL,eff}}$ is the adaptive loss weight described in Section~3.4.
This gradient opposes the classifier's attempt to find anomaly-discriminative features in the Student's representation, pushing the Student toward anomaly-*invariant* internal states.

**Why gradient reversal is necessary beyond loss bifurcation.**
Excluding anomalous patches from $L_{\mathrm{OD}}$ removes the explicit demand that the Student *follow* the Teacher on anomaly positions, but it does not prevent the Student from *remembering* anomaly-specific reconstruction patterns encountered repeatedly during training.
If the Student develops such a memory, it will reconstruct anomalous patches with low error, reducing the Teacher–Student discrepancy at the very positions where a large discrepancy would be most informative.
The gradient reversal mechanism closes this gap by actively forcing the Student's hidden states to be uninformative about anomaly identity at the representation level, making it structurally harder for the Student to exploit anomaly-specific patterns regardless of whether they appear in the loss.

**Total training loss.**
The four terms combine as follows:

$$L_{\mathrm{total}} = L_{\mathrm{recon}} + L_{\mathrm{OD}} + \lambda_{\mathrm{FM}} \cdot L_{\mathrm{FM}} + \lambda_{\mathrm{GRL}} \cdot L_{\mathrm{cls}} \tag{9}$$

where $L_{\mathrm{recon}}$ is the mean squared reconstruction error of the Teacher on masked positions, $\lambda_{\mathrm{FM}}$ and $\lambda_{\mathrm{GRL}}$ are the adaptive loss weights defined in Section~3.4, and the GRL branch contributes to $L_{\mathrm{cls}}$ only when at least one positive window is present in the batch.


## 3.6 Anomaly Scoring and Inference

**Leave-one-out masking.**
For a test window, each of the $N$ patches is evaluated under a masking pattern in which that patch alone occupies the masked set.
The $N$ patterns are forwarded in parallel by expanding the batch dimension, so the procedure requires a single forward pass per window up to memory batching.
This leave-one-out design ensures that the anomaly score for patch $i$ is computed without any information from the other patches' masking state leaking into patch $i$'s reconstruction context, eliminating cross-patch interference that would arise if multiple patches were masked simultaneously.
The inference cost is approximately $N$ times that of a single forward pass; this is an acknowledged limitation of the approach.

**Patch-level anomaly score.**
For each masked patch $i$, two quantities are computed:
- $r_i$: the Teacher reconstruction error on patch $i$ (mean squared error over its $s \cdot F$ values).
- $d_i$: the Teacher–Student output discrepancy, $d_i = \|o^T_i - o^S_i\|^2 / (s \cdot F)$.

The GRL classifier is not used at inference.
The discrepancy $d_i$ is adaptive-scaled to match the magnitude of $r_i$ across datasets and features:

$$\tilde{d}_i = d_i \cdot \frac{\bar{r} + \varepsilon}{\bar{d} + \varepsilon}, \quad \varepsilon = 10^{-4} \tag{10}$$

where $\bar{r} = (1/N)\sum_j r_j$ and $\bar{d} = (1/N)\sum_j d_j$ are window-level means.
The final per-patch score combines the two components at a fixed ratio:

$$\sigma_i = r_i + \frac{\tilde{d}_i}{c}, \quad c = 4 \tag{11}$$

The adaptive scaling is necessary because the absolute magnitudes of reconstruction error and output discrepancy vary substantially across datasets of different dimensionalities and anomaly types; without it, one component can dominate the score regardless of its discriminative value, reducing the benefit of the joint scoring.
The ratio $c$ sets the relative contribution of the discrepancy term to one quarter of the reconstruction term after scaling.

**Point-level aggregation.**
Each timestep $t$ belongs to one or more (window, patch) pairs.
The final anomaly score $s_t$ is the mean of $\sigma_i$ over all such pairs:

$$s_t = \frac{\sum_{(w,i):\; t \in \mathbf{P}^w_i} \sigma^w_i}{\left|\{(w,i):\; t \in \mathbf{P}^w_i\}\right|} \tag{12}$$

Averaging across multiple windows provides a form of ensemble smoothing: random variation in a single window's reconstruction context is reduced when the same timestep is observed from several overlapping windows at different offsets.


---

## Placeholder Block

| ID | Type | Caption / Specification |
|----|------|------------------------|
| [FIG-2] | Figure | **Caption**: CSMAD architecture overview. *Left panel (training):* Input window is split into $N$ patches; anomaly-priority masking withholds $|M|$ patches (anomalous patches are masked first). Visible patches enter the shared Transformer encoder. Mask tokens are inserted before each decoder. The Teacher decoder (darker, deeper) produces reconstruction outputs $\{o^T_i\}$; the Student decoder (lighter, shallower) produces $\{o^S_i\}$. An AnomalyClassifierHead with gradient reversal (dashed box, labeled **training only**) is applied to the Student's final hidden states. Loss connections: $L_{\mathrm{recon}}$ from Teacher outputs; $L_{\mathrm{OD}}$ and $L_{\mathrm{FM}}$ between Teacher and Student on normal masked patches; $L_{\mathrm{cls}}$ from classifier head to window label. The encoder receives no gradient from Student or GRL (stop-gradient indicated by $\perp$). *Right panel (inference):* GRL branch inactive; leave-one-out masking patterns are batch-parallelized; per-patch scores $\sigma_i$ are averaged to point-level scores $s_t$. **Content specification**: Five color regions — (1) Patch Embedding, (2) Transformer Encoder (shared), (3) Teacher Decoder, (4) Student Decoder, (5) GRL + AnomalyClassifierHead. GRL box requires explicit "training only" annotation in dashed style. Stop-gradient symbol ($\perp$) on Student latent input mandatory. Recommended placement: after §3.2 opening paragraph, full-width column. Approximate size: $1/3$ page. |
| [NUM-r_m] | Numerical | Masking ratio $r_m$. From 271_CONFIG_TRUTH §VIII: `masking_ratio = 0.15`, yielding $|M| = \mathrm{round}(50 \times 0.15) = 8$ masked patches and $|V| = 42$ visible patches per window. |
| [NUM-arch] | Numerical | Architecture dimensions. From 271_CONFIG_TRUTH §VIII: $L = 500$, $s = 10$, $N = 50$, $d = 512$ (fixed across all entities), $n_e = 4$ encoder layers, $n_T = 3$ Teacher decoder layers, $n_S = 2$ Student decoder layers, $\mathrm{nhead} = 8$, $\mathrm{dim\_ff} = 2048$, dropout $= 0.15$. |
| [NUM-c] | Numerical | Score ratio $c = 4$ (`score_recon_disc_ratio = 4.0`, 271_CONFIG_TRUTH §VIII). |
