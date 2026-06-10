---
phase: 1
agent: 271truth-verifier-1
directives: [R17]
last_modified: 2026-06-10
---

# Phase 1 Adversarial Review: 271_CONFIG_TRUTH.md
## Verifier-1 Exhaustive Re-Trace Report

**Verdict: CONDITIONAL FAIL**
**Blockers: 1 | Major: 4 | Minor: 5 | Notes: 3**

All 37 metadata files confirmed present. Config key counts verified:
total keys per entity = 117, varying keys = 3 (`grl_pos_weight`, `num_features`, `sliding_window_train_ratio`),
common keys = 114. This matches the document claim exactly.

---

## Part A — Row-by-Row Re-Trace Table

| Row ID | Document Claim | Primary Source Verified | Verdict | Issue ID |
|--------|---------------|------------------------|---------|----------|
| II-1 | 114 common keys, 37 entities | python cross-check: 114 common, 3 varying, 37 files | PASS | — |
| II-table | All 114 key/value pairs | Spot-checked PSM, SWaT-full, SWaT-excl22, SMD/machine-1-2, WaDi/A1 — all match | PASS | — |
| III-3a | `num_features` table (MSL=55, SMAP=25, PSM=25, SMD varies, SWaT=45, WaDi=123) | All 37 entities verified exact match | PASS | — |
| III-3b | `grl_pos_weight` min=3.14 (SMAP/T-1), max=999.0 (SMD/machine-1-5), SWaT=59.18 | Actual: min=3.1410184667067713 (SMAP/T-1), max=999.0 (SMD/machine-1-5), SWaT=59.1814... | PASS (rounding acceptable) | — |
| III-3c | PSM train_ratio=0.8007 | Actual: 0.8006508655513295 | PASS (4-decimal rounding) | N1 |
| III-3c | SMAP train_ratio ~0.625 | Actual: G-7=0.6167, P-1=0.6262, P-4=0.6255, T-1=0.6251, T-3=0.6255 | MINOR FAIL — G-7 is 0.617, not ~0.625 | M1 |
| III-3c | MSL train_ratio 0.635–0.764 | Actual: C-1=0.744, C-2=0.636, F-7=0.666, P-11=0.764, T-13=0.660 | PASS | — |
| III-3c | SWaT both train_ratio=0.762 | Actual: 0.7619266836628324 | PASS (rounding) | — |
| III-3c | WaDi/A1=0.937 | Actual: 0.9374993670437398 | PASS | — |
| III-3c | WaDi/A2=0.910 | Actual: 0.9097510481797082 | PASS | — |
| III-3c | SMD ~0.75 | Actual: exactly 0.75 (range 0.74998–0.75) | PASS | — |
| IV | SWaT A1A2_full swat_eval_mode=None | metadata: `"swat_eval_mode": null` | PASS | — |
| IV | SWaT A1A2_excl22 swat_eval_mode='excl22' | metadata: `"swat_eval_mode": "excl22"` | PASS | — |
| IV | Both SWaT conditions same trained model | timing.wall_time identical in both files | PASS | — |
| IV | SWaT config best_epoch_metric same (pak_auc_f1) | PARTIAL: config key is same (pak_auc_f1) for both, BUT timing.best_epoch_metric is `excl22_pak_auc_f1` for excl22 entity | MAJOR — document does not acknowledge this distinction | M2 |
| V | 0 blockers (114 keys identical across 37 entities) | Cross-checked, confirmed | PASS | — |
| VI (Encoder row) | `model.py:359-362` — encoder construction | Actual: line 359 `self.encoder = nn.TransformerEncoder(`, line 360 `num_layers=config.num_encoder_layers` | PASS (within range) | — |
| VI (Teacher decoder) | `model.py:407-423` | Actual: `self.teacher_decoder = nn.TransformerEncoder(` at line 419, `num_teacher_decoder_layers` at 421 | MINOR: start-line cited as 407 but actual teacher_decoder block begins at 404-406 (`self.teacher_decoder = None`); construction at 419 | Mi1 |
| VI (Student decoder) | `model.py:445-461` | Actual: `self.student_decoder = nn.TransformerEncoder(` at line 457, `num_student_decoder_layers` at 459 | MINOR: start-line cited as 445; actual block at 443-461 | Mi1 |
| VI (Linear patch) | `model.py:577-580` — "branch `if self.patchify_mode == 'patch_cnn'` is skipped" | Line 580: `if self.patchify_mode == 'patch_cnn':` confirmed; linear path at line 624 | MINOR: cited range 577-580 is the function definition + patch_cnn check; doc says "577-580 branch skipped" but 577-578 are the function def lines, not the branch itself | Mi2 |
| VI (Shared decoder inactive) | `model.py:367-368` | Line 367: `self.shared_decoder = None`, line 368: `if self.num_shared_decoder_layers > 0:` | PASS | — |
| VI (Masking) | `config.py:315-319` for force_mask_anomaly | Line 315: `force_mask_anomaly: bool = True` confirmed | PASS | — |
| VI (mask_after_encoder) | `model.py:1119-1129` | Line 1119: `if self.config.use_student and...`, line 1120: `if self.mask_after_encoder:` — student branch; teacher branch starts at 1028. The doc presents this as the student branch, which is correct | MINOR: cited range is student branch; teacher mask_after_encoder is at lines 1028-1044 — citing only student branch without noting teacher branch has same pattern is incomplete but not wrong | Mi3 |
| VI (Separate mask tokens) | `model.py:499-505` | Line 498: `else:`, lines 499-505 create separate tokens | PASS (off by 1 on start, minor) | — |
| VI (Teacher recon loss) | `loss.py:172-179` | Confirmed lines 172-179 compute teacher_recon_full, reconstruction_loss | PASS | — |
| VI (Output discrepancy) | `loss.py:254-261` | Line 254: `if self.use_output_discrepancy:`, line 255: normal_loss, line 259: `if self.use_grl and self.grl_disable_anomaly_loss:`, line 261: `anomaly_loss = torch.tensor(0.0, ...)` | PASS | — |
| VI (GRL classifier) | `model.py:530-538` instantiation | Lines 530-538 confirmed | PASS | — |
| VI (GRL classifier) | `model.py:1150-1154` called on student hidden | Lines 1150-1154 confirmed | PASS | — |
| VI (GRL classifier) | `trainer.py:746-771` GRL loss added | Lines 746-771 confirmed | PASS | — |
| VI (GRL adaptive lambda) | `trainer.py:751-765` | Lines 751-765 confirmed | PASS | — |
| VI (GRL focal) | `loss.py:337-340` | Lines 337-340 confirmed: `_p_t = torch.exp(-_bce); _focal = ((1-_p_t)**2.0)*_bce` | PASS | — |
| VI (GRL window target) | `loss.py:285-287` | Lines 285-287 confirmed | PASS | — |
| VI (GRL balanced sampling) | `loss.py:313` | Line 313: `if self.grl_balanced_sampling:` | PASS | — |
| VI (WDGRL inactive) | `trainer.py:662` | Line 662 confirmed: checks `_grl_mode == 'wdgrl'` | PASS | — |
| VI (FM active, training only) | `loss.py:414-430` | Lines 414-430 confirmed | PASS | — |
| VI (FM inactive at inference) | `scoring.py:237` | Line 237: `fm_active = False` confirmed | PASS | — |
| VI (Patch-level loss) | `loss.py:225-252` | Line 225: `if self.patch_level_loss:` confirmed at 225 | PASS (minor: cited as 225-252, relevant content ends ~252) | — |
| VI (Teacher-only warmup) | `trainer.py:43-44` | Lines 43-44: warmup epochs assignment | PASS | — |
| VI (Teacher warmup early stop) | `trainer.py:485` — `_es_on = False` | Line 485: `_es_on = getattr(self.config, 'use_teacher_warmup_early_stop', False)` — with config=False, `_es_on = False` | PASS (logic correct; phrasing slightly misleading — it's not hardcoded False, it reads from config) | N2 |
| VI (Teacher output EMA) | `model.py:514` | Line 514: `self._has_teacher_output_ema = bool(getattr(config, 'use_teacher_output_ema', False)) and config.use_teacher` | PASS | — |
| VI (RevIN inactive) | `model.py:312-314` | Lines 312-314 confirmed | PASS | — |
| VI (Discriminator inactive) | `trainer.py:236-237` | Line 237: `self.discriminator = None` | PASS | — |
| VI (SCAD inactive) | `model.py:541` | Line 541: `if getattr(config, 'use_scad', False):` | PASS | — |
| VI (SCAD inactive) | `loss.py:355` | Line 355: `if scad_z is not None and self.use_scad:` | PASS | — |
| VI (Random masking range) | `trainer.py:522-524` | Lines 522-524 confirmed: condition `(_mr_min >= 0 and _mr_max >= 0)` is False | PASS | — |
| VI (Masking ratio annealing) | `config.py:241-243` — "trainer never triggers annealing path" | **WRONG.** Trainer DOES implement annealing at trainer.py:1201 (`if getattr(self.config, 'masking_ratio_anneal', False) and ...`). Path is inactive because `masking_ratio_anneal=False` in config, NOT because the trainer lacks the path. Code evidence (config.py:241-243) only shows the flag definition, not the execution gate. | **BLOCKER** | B1 |
| VI (Complementary masking) | `config.py:226-229` — "evaluator path never enters complementary group logic" | **Misleading/incomplete evidence.** The flag definition is in config.py:226-229, but the active runtime check is in `evaluator.py:1716` (`_use_complementary = getattr(self.config, 'eval_complementary_masking', False)`), and the full complementary-masking path exists at evaluator.py:1737-1745. The cited evidence (config.py:226-229) only proves the flag default; it does NOT show the evaluator never enters the path. The conclusion (INACTIVE) is correct, but the code evidence citation is insufficient. | MAJOR | M3 |
| VI (Shared mask token) | `model.py:492-505` | Confirmed: condition at 495, else branch at 498 | PASS | — |
| VI (freeze_teacher_after_warmup) | `trainer.py:50-55` — "condition not entered" | Lines 50-55 are the config-validation override (forces warmup length), NOT the runtime freeze gate. The actual runtime gate is at trainer.py:1141-1142. The cited lines are a config-init side effect, not the "condition not entered" runtime check. Conclusion (INACTIVE) is correct since flag=False, but cited evidence is the wrong code section. | MAJOR | M4 |
| VI (freeze_encoder_only) | `trainer.py:75-79` — "condition not entered" | Lines 75-79 are a ValueError guard against simultaneous freeze flags, NOT the runtime freeze gate (which is at trainer.py:1169-1179). Conclusion correct, evidence is wrong code section. | MAJOR | M5 |
| VII.1 (Dynamic margin) | "anomaly_loss is zeroed, `_compute_patch_anomaly_loss` is never invoked" | Logic is correct: loss.py:259-261 zeroes anomaly_loss before the else branch (268) that calls `_compute_patch_anomaly_loss`. Confirmed inactive. | PASS | — |
| VII.2–20 | Various INACTIVE items | All verified correct via config values | PASS | — |
| VIII (masking 8 patches) | `round(50 × 0.15) = 8` | `python3: round(50 * 0.15) = 8` — Python3 banker's rounding (7.5 → 8, which is even). Confirmed. | PASS | — |
| VIII (test stride) | `seq_length // 10 - 1 = 49` | `utils/experiment.py:38`: `return max(1, W // 10 - 1)` with W=500 → 499?? Wait: 500//10 - 1 = 50 - 1 = 49. Correct. | PASS | — |
| VIII (Optimizer betas) | `betas=(0.9, 0.99)` | `trainer.py:163`: `betas=(0.9, 0.99)` | PASS | — |
| VIII (Anomaly loss warmup) | `warmup_length = max(250//5, 2) = 50` | `trainer.py:342`: `warmup_length = max(student_start // 5, 2)` with student_start=250 → 50 | PASS | — |
| VIII (GRL arch) | `1-layer MLP: LayerNorm → Linear(d_model, d_model//2=256) → GELU → Dropout(0.1) → Linear(256, 1)` | `model.py:179-185` with grl_cls_hidden=0: `hidden_dim = d_model//2 = 256`, then LayerNorm→Linear(512,256)→GELU→Dropout(0.1)→Linear(256,1) | PASS | — |
| VIII (GRL lambda formula) | `lambda = \|\|grad_main\|\| / (\|\|grad_grl\|\| + 1e-4), clamped [0,10]` | `trainer.py:760`: `(_main_g.norm() / (_grl_g.norm() + 1e-4)).clamp(0.0, 10.0)` | PASS | — |
| VIII (FM formula) | `((teacher_hidden.detach() - student_hidden)**2).mean(dim=-1)` | `loss.py:420` | PASS | — |
| VIII (Anomaly score formula) | `scoring.py:239-253` | `scoring.py:239-256` confirmed | PASS (line range slightly off: ends at ~256, not 253) | N3 |

---

## Part B — Detailed Issue Reports

### B1 — BLOCKER

**Issue ID:** B1
**Severity:** BLOCKER
**Artifact:** `271_CONFIG_TRUTH.md`, Section VI, "Masking ratio annealing" row
**Problematic Claim:**
> "Code Evidence: `config.py:241-243` — flag exists but trainer never triggers annealing path"

**Evidence:**
`trainer.py:1201`:
```python
if getattr(self.config, 'masking_ratio_anneal', False) and epoch >= teacher_warmup:
    _anneal_progress = (epoch - teacher_warmup) / max(self.config.num_epochs - teacher_warmup - 1, 1)
    self._annealed_masking_ratio = (
        self.config.masking_ratio * (1 - _anneal_progress) + _anneal_target * _anneal_progress
    )
```
The trainer absolutely DOES implement and can trigger the annealing path. The path is inactive in experiment 271 only because `masking_ratio_anneal=False` in the config, which causes the `getattr(..., False)` check to evaluate False.

The cited evidence (`config.py:241-243`) shows only the flag definition in the dataclass, not a code path that makes the annealing dead. This is a false claim about the code structure.

**Why this is BLOCKER (R17 relevance):** The "INACTIVE — trainer never triggers annealing path" judgment is substantively wrong as a code claim. A reviewer reading this will form an incorrect mental model of the codebase. The correct statement is: "INACTIVE because `masking_ratio_anneal=False` in config; trainer does implement the path at `trainer.py:1201`."

**Recommended Fix:**
Change the Code Evidence column entry to:
`trainer.py:1201` — `if getattr(self.config, 'masking_ratio_anneal', False) and ...:` — condition evaluates False (flag=False in config); full annealing path at trainer.py:1201-1211 is not entered.
Change the VII exclusion list entry #14 to reflect the same correction.

---

### M2 — MAJOR

**Issue ID:** M2
**Severity:** MAJOR
**Artifact:** `271_CONFIG_TRUTH.md`, Section IV, SWaT Dual-Condition
**Problematic Claim:**
> "Both share **identical** config (same `num_features=45`, same `grl_pos_weight=59.18`, same `sliding_window_train_ratio=0.762`)"

Also in Section II canonical config table: `best_epoch_metric: pak_auc_f1` listed as a common key for ALL 37 entities.

**Evidence:**
`SWaT/A1A2_excl22/experiment_metadata.json`:
- `config.best_epoch_metric = "pak_auc_f1"` (stored config)
- `timing.best_epoch_metric = "excl22_pak_auc_f1"` (actual best-epoch selection metric used)

The runtime best-epoch selection for `A1A2_excl22` used `excl22_pak_auc_f1`, not `pak_auc_f1`. The config key `best_epoch_metric` stores `pak_auc_f1` for both, but the actual evaluation logic uses a derived metric (`excl22_pak_auc_f1`) for the excl22 entity. This means:
1. The "same trained model" claim is correct (wall_time identical).
2. But "identical config" is misleading — the excl22 entity selects its best epoch on a different metric than what the config key `best_epoch_metric` stores.
3. Section II's listing of `best_epoch_metric = pak_auc_f1` as a universal common key is technically correct (it IS stored identically in all 37 configs) but omits the operationally significant distinction that the excl22 evaluation uses a modified metric.

**Recommended Fix:** Add a note in Section IV: "Note: SWaT/A1A2_excl22 timing records `best_epoch_metric = 'excl22_pak_auc_f1'` (best epoch selected on excl22-masked F1), whereas the config key stores `pak_auc_f1`. The config key reflects the template; runtime evaluation overrides the metric name for excl22 evaluation."

---

### M3 — MAJOR

**Issue ID:** M3
**Severity:** MAJOR
**Artifact:** `271_CONFIG_TRUTH.md`, Section VI, "Complementary masking at inference" row
**Problematic Claim:**
> Code Evidence: `config.py:226-229` — flag; evaluator path never enters complementary group logic

**Evidence:**
The evaluator DOES contain the complementary-masking path at `evaluator.py:1716-1745`. The flag check is at evaluator.py:1716:
```python
_use_complementary = getattr(self.config, 'eval_complementary_masking', False)
```
and the path at evaluator.py:1737-1745 runs only if `_use_complementary` is True.

Citing only `config.py:226-229` (the flag definition) as evidence that "the evaluator path never enters complementary group logic" is wrong code evidence. The evidence must be in evaluator.py, not config.py.

**Recommended Fix:** Change Code Evidence to: `evaluator.py:1716` — `_use_complementary = getattr(self.config, 'eval_complementary_masking', False)` evaluates to False; evaluator.py:1737 `if _use_complementary:` branch not entered.

---

### M4 — MAJOR

**Issue ID:** M4
**Severity:** MAJOR
**Artifact:** `271_CONFIG_TRUTH.md`, Section VI, "freeze_teacher_after_warmup" row
**Problematic Claim:**
> Code Evidence: `trainer.py:50-55` — condition not entered

**Evidence:**
`trainer.py:50-55` is the config-validation section that overrides `teacher_only_warmup_epochs` when `freeze_teacher_after_warmup=True`. This runs unconditionally at init and is NOT the runtime gate that determines whether teacher modules are frozen. The actual runtime gate is at `trainer.py:1141-1142`:
```python
if (getattr(self.config, 'freeze_teacher_after_warmup', False) and
        epoch == teacher_warmup and not hasattr(self, '_frozen_eval_modules')):
```

Citing lines 50-55 as the "condition not entered" evidence is citing the wrong code section.

**Recommended Fix:** Change Code Evidence to: `trainer.py:1141-1142` — runtime freeze gate condition evaluates False (`freeze_teacher_after_warmup=False`); no modules are frozen. Note that lines 50-55 are a separate config-validation override (irrelevant to INACTIVE judgment).

---

### M5 — MAJOR

**Issue ID:** M5
**Severity:** MAJOR
**Artifact:** `271_CONFIG_TRUTH.md`, Section VI, "freeze_encoder_only" row
**Problematic Claim:**
> Code Evidence: `trainer.py:75-79` — condition not entered

**Evidence:**
`trainer.py:75-79` is a `ValueError` guard that raises an error if BOTH `freeze_encoder_only=True` AND `freeze_teacher_after_warmup=True` simultaneously. This is not the runtime freeze gate. The actual runtime gate is at `trainer.py:1169-1170`:
```python
if (getattr(self.config, 'freeze_encoder_only', False) and
        epoch == teacher_warmup and not hasattr(self, '_frozen_encoder_modules')):
```

**Recommended Fix:** Change Code Evidence to: `trainer.py:1169-1170` — runtime freeze gate evaluates False (`freeze_encoder_only=False`); no encoder modules are frozen.

---

### M1 — MINOR (note: R17 adjacent)

**Issue ID:** M1
**Severity:** MINOR
**Artifact:** `271_CONFIG_TRUTH.md`, Section III-3c
**Problematic Claim:**
> "SMAP entities: ~0.625"

**Evidence:**
Actual values from metadata:
- SMAP/G-7: 0.6167 (rounds to 0.617, not 0.625)
- SMAP/P-1: 0.6262
- SMAP/P-4: 0.6255
- SMAP/T-1: 0.6251
- SMAP/T-3: 0.6255

SMAP/G-7's train ratio is 0.617, which does not round to 0.625. The "~0.625" approximation is inaccurate for G-7 specifically.

**Recommended Fix:** Change to "SMAP/G-7: 0.617; SMAP/P-1, P-4, T-1, T-3: ~0.625" or use a broader range "0.617–0.626".

---

### Mi1 — MINOR

**Issue ID:** Mi1
**Severity:** MINOR
**Artifact:** Section VI, teacher decoder and student decoder rows
**Problematic Claim:**
- Teacher decoder: `model.py:407-423`
- Student decoder: `model.py:445-461`

**Evidence:**
- Teacher decoder `nn.TransformerEncoder` construction starts at line 419 (not 407); the `self.teacher_decoder = None` initialization and `if config.use_teacher:` check are at 404-406.
- Student decoder `nn.TransformerEncoder` construction is at line 457 (not 445); the `self.student_decoder = None` is at 443.

The cited start lines are off by ~12 lines, pointing to the block preceding the actual TransformerEncoder construction. The conclusion (ACTIVE) is correct.

**Recommended Fix:** Update line ranges: Teacher: `model.py:419-423`; Student: `model.py:457-461`.

---

### Mi2 — MINOR

**Issue ID:** Mi2
**Severity:** MINOR
**Artifact:** Section VI, "Linear patch embedding" row
**Problematic Claim:**
> Code Evidence: `model.py:577-580` — branch `if self.patchify_mode == 'patch_cnn'` is skipped; linear embedding path is used

**Evidence:**
Lines 577-578 are the function definition `def _build_embedding_layers(self, config: Config):` and its docstring open line. The actual `if self.patchify_mode == 'patch_cnn':` branch is at line 580. The linear path is at line 624. The description says "577-580" but 577-578 have no relevance to the skip claim.

**Recommended Fix:** Cite `model.py:580` (patch_cnn branch skipped) and `model.py:624` (linear path entered).

---

### Mi3 — MINOR

**Issue ID:** Mi3
**Severity:** MINOR
**Artifact:** Section VI, "Mask-after-encoder (standard MAE layout)" row
**Problematic Claim:**
> Code Evidence: `model.py:1119-1129` — `if self.mask_after_encoder:` branch; mask tokens inserted before decoders, encoder sees only visible patches

**Evidence:**
Lines 1119-1129 cover only the **student decoder** mask-after-encoder branch. The teacher decoder's mask-after-encoder branch is at `model.py:1028-1044`. Claiming "mask tokens inserted before decoders" based only on the student branch at 1119-1129 is incomplete.

**Recommended Fix:** Add teacher branch reference: `model.py:1028-1044` (teacher) and `model.py:1119-1129` (student).

---

### N1 — NOTE

**Issue ID:** N1
**Severity:** NOTE
**Artifact:** Section III-3c, PSM train ratio
**Claim:** PSM: 0.8007
**Actual:** 0.8006508655513295
This is a rounding (4 significant figures). Not a problem for a config description document.

---

### N2 — NOTE

**Issue ID:** N2
**Severity:** NOTE
**Artifact:** Section VI, "Teacher warmup early stop" row
**Claim:** `trainer.py:485 — _es_on = False`
**Actual:** `_es_on = getattr(self.config, 'use_teacher_warmup_early_stop', False)` — this is not `_es_on = False` hardcoded. It is dynamically read from config. With config=False, it evaluates to False. The description "False" is accurate in context but the code is not a hardcoded False. Low-severity.

---

### N3 — NOTE

**Issue ID:** N3
**Severity:** NOTE
**Artifact:** Section VIII, anomaly score formula
**Claim:** "from `scoring.py:239-253`"
**Actual:** The relevant formula spans scoring.py:239-265 (through the return statement at line 255-265). Line 253 is mid-computation. Not a material error.

---

## Part C — Canonical Config Cross-Check (5 Entity Sample)

Entities sampled: PSM, SWaT/A1A2_full, SWaT/A1A2_excl22, SMD/machine-1-2, WaDi/A1.

All 114 common keys verified identical across these 5 entities via Python extraction. Key highlights:
- `batch_size=1024`, `d_model=512`, `nhead=8`, `seq_length=500`, `patch_size=10`, `num_patches=50` — all identical
- `use_grl=True`, `use_revin=False`, `use_scad=False`, `use_discriminator=False` — all identical
- `margin_type='dynamic'`, `dynamic_margin_k=6` — present but INACTIVE (grl_disable_anomaly_loss=True)
- `sliding_window_test_stride=-1` — sentinel; resolves to 49 via `utils/experiment.py:38`

---

## Part D — Used/Unused Judgment Summary

All ACTIVE/INACTIVE judgments in Section VI are factually correct given the config values. The three MAJOR issues (M3, M4, M5) and one BLOCKER (B1) concern the **code evidence citations**, not the final ACTIVE/INACTIVE verdicts. The B1 item involves a false claim about code structure that could mislead paper reviewers about how the codebase works.

**SWaT dual-condition (M2):** Structural finding. The config-stored `best_epoch_metric` does not reflect the operationally distinct `excl22_pak_auc_f1` used at runtime. This affects reproducibility documentation accuracy.

---

## Part E — Resolution Checklist

- [ ] **B1** Fix masking_ratio_anneal code evidence — cite trainer.py:1201, correct "trainer never triggers" language
- [ ] **M2** Add note in §IV about SWaT excl22 `timing.best_epoch_metric = excl22_pak_auc_f1`
- [ ] **M3** Fix complementary masking evidence — cite evaluator.py:1716, not config.py:226-229
- [ ] **M4** Fix freeze_teacher_after_warmup evidence — cite trainer.py:1141, not trainer.py:50-55
- [ ] **M5** Fix freeze_encoder_only evidence — cite trainer.py:1169, not trainer.py:75-79
- [ ] **M1** Fix SMAP train_ratio approximation for G-7 (0.617, not ~0.625)
- [ ] Mi1 Update teacher/student decoder line ranges
- [ ] Mi2 Update patchify_mode linear embedding line reference
- [ ] Mi3 Add teacher mask_after_encoder branch reference (line 1028)
