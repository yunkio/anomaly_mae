# Changelog

## 2026-02-26 (Update 55): patch_batch_size OOM Fix for Large d_model

### Summary

Reduced `patch_batch_size` from 4 to 2 when `d_model >= 512` to prevent GPU OOM during contribution ratio computation on large test sets (SWaT/WaDi with Set C).

### Changes

**`mae_anomaly/evaluator.py`:**
- `patch_batch_size`: Changed from unconditional `min(num_patches, 4)` to `min(num_patches, 2 if d_model >= 512 else 4)`. Prevents GPU OOM on SWaT (10,689 test windows) and WaDi (4,091 test windows) when d_model=512.

## 2026-02-25 (Update 54): Set C — Dynamic d_model + Linear Embedding + Auto dim_feedforward

### Summary

Added Set C experiment preset with per-dataset dynamic d_model selection and linear patch embedding. `dim_feedforward` is now auto-computed as `4 × d_model` when not explicitly overridden in `make_config()`.

### Changes

**`mae_anomaly/utils/experiment.py`:**
- `resolve_dynamic_d_model(num_features, patch_size)` (NEW): Selects smallest d_model from `[128, 192, 256, 384, 512]` that is ≥ `patch_size × num_features`. Caps at 512.
- `D_MODEL_CANDIDATES` (NEW): Candidate list `[128, 192, 256, 384, 512]`.
- `make_config()`: Auto-computes `dim_feedforward = 4 × d_model` when `'dim_feedforward'` is not in overrides dict. Existing presets (Set A/B) that explicitly pass `dim_feedforward` are unaffected.

**`scripts/run_base_experiments.py`:**
- Set C preset: `patch_size=10, num_patches=50, d_model='dynamic', patchify_mode='linear'`. Other params same as Set B.
- Dynamic resolution: After data loading, if `d_model='dynamic'`, calls `resolve_dynamic_d_model()` with the dataset's actual `num_features` to determine d_model before `make_config()`.
- `--set` argparse: Added `'C'` to choices.

**`set_guideline.md`:**
- Config Presets table: Added Set C column.
- Added "Set C: Dynamic d_model 규칙" subsection.

**`docs/ARCHITECTURE.md`:**
- Default Configuration table: Updated d_model and dim_feedforward descriptions.
- Added "Dynamic d_model (Set C)" subsection.

## 2026-02-25 (Update 53): SMD K=6 Block Split Loader + Epoch Offset Train Augmentation

### Summary

Added SMD per-machine K=6 block split loader for balanced anomaly distribution (~50/50 train/test). Added epoch offset feature that shifts train sliding window start positions each epoch for better generalization with large strides.

### Changes

**`mae_anomaly/datasets/loaders.py`:**
- `load_smd_block_split(machine, k_blocks, parity, margin)` (NEW): Splits a single SMD machine's test file into K blocks with safe boundary snapping (±margin from anomaly regions). Alternates blocks between train/test by parity.
- `_find_safe_cut_point`, `_get_anomaly_regions_local` (NEW): Helpers for boundary placement.
- Registry: `smd_k6_{machine_id}` (parity=0) and `smd_k6_{machine_id}_swap` (parity=1) for all 28 machines.

**`mae_anomaly/dataset_sliding.py`:**
- `SlidingWindowDataset.set_epoch_offset(offset)` (NEW): Shifts window start positions by `offset % stride`, re-extracts window metadata. Train-only; test stays at offset=0.

**`mae_anomaly/config.py`:**
- `epoch_offset: bool = False` (NEW): When True, Trainer applies non-replacement random offsets from `[0, stride)` each epoch. Over `stride` epochs, all positions are covered exactly once.

**`mae_anomaly/trainer.py`:**
- `train()`: When `epoch_offset=True`, pops a random offset from a permutation pool of `[0, stride)` each epoch and calls `train_dataset.set_epoch_offset()`.

**`docs/SMD_BLOCK_SPLIT.md`** (NEW): Experiment guide for K=6 block split methodology.

## 2026-02-24 (Update 52): Point-Level Epoch Eval + Detailed Timing + Layer-Level Batch Profiling

### Summary

Replaced window-level epoch monitoring with point-level metrics. Added comprehensive timing measurement across all pipeline stages. Added first-N-batch per-component + per-layer profiling (replaces PyTorch Profiler) with batch 0 skipped (CUDA warmup distortion).

### Changes

**`mae_anomaly/model.py`:**
- `forward`: Added layer-level profiling support via `_profiling` attribute. When `_profiling=True`, inserts `cuda.synchronize()` between 5 architectural sections (embed_input, masking, encoder, teacher_decoder, student_decoder). Results stored in `_forward_timing` dict.

**`mae_anomaly/trainer.py`:**
- `train_epoch`: Added per-epoch timing (forward_approx, backward_approx, epoch_total) with CUDA sync at epoch boundaries only (~1% overhead)
- `train_epoch`: Added `profile_batches` param — batches 1..N of epoch 1 get per-component `cuda.synchronize()` timing (batch 0 skipped to avoid CUDA warmup distortion)
  - Batch level: data→GPU, model_forward, loss_compute, backward, optimizer_step
  - Layer level (inside model_forward): embed_input, masking, encoder, teacher_decoder, student_decoder
- `train`: Accepts `profile_n_batches`, passes to epoch 0 only. Stores results in `history['batch_profiling']`
- `_print_batch_profiling`: Prints hierarchical profiler-like table (with layer breakdown under Model Forward) immediately after epoch 1, with estimated remaining training time
- `train`: Records per-epoch timing for train_epoch, contrib_ratios, callback → `history['epoch_timings']`

**`scripts/run_base_experiments.py`:**
- `compute_epoch_test_metrics`: Returns inference_time vs eval_time breakdown in metrics dict
- `save_batch_profiling` (NEW): Formats per-batch timing into profiler-like summary table + JSON. Saves `batch_profiling.json` + `batch_profiling.txt`
- Removed `run_profiling` (PyTorch Profiler) — replaced with in-training batch profiling
- Epoch callback: Logs `(infer=Ns eval=Ns)` per eval, accumulates callback_total_time
- Training timing: Separates `pure_train_time` (excludes callback), `contrib_ratios_time`, `epoch_eval_time`
- Final inference: Separates `patch_scores_time` vs `viz_collect_time`
- timing dict: Expanded with all phase timings (wall_time, pure_train_time, epoch_eval_time, etc.)
- Removed dead code: `_cpu_epoch_pointlevel_worker`, `_merge_epoch_pointlevel`, `_epoch_pl_processes`
- `plot_epoch_metrics`: Rewritten for point-level (4 PNGs: prc_auc, f1_t, pa_k_f1, dashboard)

**`mae_anomaly/dataset_sliding.py`:**
- Fixed `train_end` alignment bug (removed stride-dependent boundary shift)

**`set_guideline.md`:**
- Updated epoch callback, epoch_metrics.json format, pipeline, visualization descriptions

## 2026-02-23 (Update 51): Fix force_mask_anomaly Non-Uniform Masking Bug (A-1)

### Summary

Fixed critical bug where `force_mask_anomaly` broke the uniform masking assumption required by `_encode_visible_only` (standard MAE encoder). The old implementation forced ALL anomaly patches to be masked regardless of masking budget, causing variable `num_keep` across the batch. This led to masked patches (including anomaly data) leaking into the encoder for some samples.

### Problem

When `force_mask_anomaly=True` and a sample had anomaly patches that didn't overlap with the random mask:
1. All anomaly patches were force-masked, increasing total masked count beyond `target_num_masked`
2. Different samples in a batch had different numbers of visible patches
3. `_encode_visible_only` used `num_keep` from sample 0, causing:
   - Samples with fewer visible patches: masked patches leaked into encoder (anomaly information leakage)
   - Samples with more visible patches: visible patches incorrectly excluded from encoder

### Fix

Replaced the old force-then-patch approach with **fixed-budget priority-based masking**:
- Masking budget is always exactly `round(num_patches * masking_ratio)` per sample
- Anomaly patches are prioritized for masking within this budget
- If anomaly patches exceed the budget, excess remain visible as encoder context
- Fully vectorized implementation (no per-sample loop) using priority sorting + scatter

### Changes

**`mae_anomaly/model.py`:**
- Rewrote `force_mask_anomaly` section in `forward()` with vectorized priority-based masking
- Added assertion in `_encode_visible_only` to catch non-uniform masking (safety check)

**`mae_anomaly/config.py`:**
- Updated `force_mask_anomaly` description to reflect new priority-based behavior

**`docs/ARCHITECTURE.md`:**
- Updated Force Mask Anomaly section with detailed behavior description

**`docs/TEP_EXPERIMENT_GUIDE.md`:**
- Updated force_mask_anomaly description

**`docs/ABLATION_STUDIES.md`:**
- Updated Force Mask Anomaly experiment description

## 2026-02-17 (Update 50): TEP Experiment Guide + save_dataset_info Fix

### Summary

Added comprehensive TEP experiment guidelines and config files. Fixed `save_dataset_info` bug where fault types >= 10 caused IndexError. Three config files cover quick test, single fault, and all-faults scenarios.

### Changes

**Fixed `scripts/ablation/run_ablation.py`:**
- `save_dataset_info`: Added `_atype_to_name()` helper to safely convert anomaly_type
  - Fault types 1-9: maps to simulation type names (backward compatible)
  - Fault types 10+: maps to `fault_N` (supports TEP fault types 10-20)
- Replaced hardcoded `SLIDING_ANOMALY_TYPE_NAMES` indexing with dynamic type set from `anomaly_regions`

**New `docs/TEP_EXPERIMENT_GUIDE.md`:**
- Part 1: Current model/experiment framework understanding
- Part 2: TEP dataset structure (960 samples/run, fault onset at sample 160, 20 faults)
- Part 3: Dataset comparison (SWaT vs WaDi vs TEP)
- Part 4: Recommended hyperparameters (seq_length=160, stride=5)
- Part 5: Three experiment scenarios (Quick/Single/All)
- Part 6: Execution instructions and result structure

**New config files (`scripts/ablation/configs/`):**
- `tep_quick_test.py`: 1 epoch, fault1 only, stride=11 test (pipeline verification)
- `tep_single_fault.py`: 50 epochs, configurable fault type, full PA%K evaluation
- `tep_all_faults.py`: 50 epochs, all 20 faults, full evaluation

### Key Design Decisions

- `seq_length=160`: aligns with fault onset period (samples 0-159 normal, 160-959 anomalous)
- `patch_size=8, num_patches=20`: efficient patch granularity for 160-sample windows
- `sliding_window_stride=5` (train): ~161 windows per 960-sample run
- `run_boundaries` handled automatically by loader (data_info['run_boundaries'])

---

## 2026-02-17 (Update 49): SMD (Server Machine Dataset) Loader

### Summary

Added SMD dataset loader for 28 server machines with full pipeline compatibility. Uses the same `run_boundaries` mechanism as TEP for handling independent machine boundaries.

### Changes

**New in `mae_anomaly/datasets/loaders.py`:**
- `load_smd()` function: loads all 28 machines or specific subset
- `SMD_MACHINE_NAMES`: list of all 28 machine IDs
- Registry entries: `smd` (all machines) + `smd_machine-X-Y` (28 individual loaders)

**Modified `mae_anomaly/datasets/__init__.py`:**
- Added `load_smd` and `SMD_MACHINE_NAMES` to exports

### Dataset Stats
- 28 machines, 38 features each (37 after constant removal)
- Total: 1,416,825 samples (708,405 train + 708,420 test)
- Test anomaly ratio: 4.16% (29,444 anomaly points)
- 327 anomaly regions across all machines
- train_ratio: 0.5 (train = all normal, test = with anomalies)

---

## 2026-02-15 (Update 48): TEP Dataset Support + Run Boundary Handling

### Summary

Added TEP (Tennessee Eastman Process) dataset loader with support for 20 fault types and independent simulation run handling. Introduced `run_boundaries` mechanism to prevent sliding windows from crossing independent run boundaries.

### Changes

**New in `mae_anomaly/datasets/loaders.py`:**
- `load_tep()` function: loads TEP RData files, supports per-fault-type selection
- `TEP_FAULT_NAMES`: descriptive names for 20 TEP fault types
- Registry entries: `tep` (all faults) and `tep_fault1` through `tep_fault20`

**Modified `mae_anomaly/dataset_sliding.py`:**
- `SlidingWindowDataset.__init__`: new optional `run_boundaries` parameter
- `_extract_windows()`: skips windows that cross run boundaries

**Modified `mae_anomaly/evaluator.py`:**
- Per-type evaluation now discovers anomaly types dynamically from data (supports >9 types)

**Modified `mae_anomaly/trainer.py`:**
- Per-type score computation now handles arbitrary anomaly type indices

**Modified `scripts/ablation/run_ablation.py`:**
- Extracts `run_boundaries` from `data_info` and passes through entire pipeline
- All `SlidingWindowDataset` and `NoisyLabelSlidingWindowDataset` calls updated

**Modified `mae_anomaly/datasets/noisy.py`:**
- `NoisyLabelSlidingWindowDataset`: passes `run_boundaries` to parent class

**Dependencies:**
- `pyreadr` required for loading TEP RData files

---

## 2026-02-15 (Update 47): Script Consolidation — Unified Config System

### Summary

Consolidated 10 separate run_*.py scripts (4,742 lines) into a unified config-based system with single entry point. This eliminates code duplication, reduces maintenance burden, and enables easy experiment reproduction via config files.

### Changes

**New Modules:**
- `mae_anomaly/datasets/loaders.py` - Centralized dataset loaders with registry pattern
- `mae_anomaly/datasets/noisy.py` - NoisyLabelSlidingWindowDataset
- `mae_anomaly/utils/system.py` - GPU memory utilities (free_gpu, mem_status)
- `mae_anomaly/utils/experiment.py` - Config creation helper (make_config)

**Updated Scripts:**
- `scripts/ablation/run_ablation.py` - Extended to support multiple dataset types via DATASET_TYPE config parameter
- `scripts/run_base_experiments.py` - Updated imports to use new modules

**Config System:**
```bash
# Single entry point for all datasets
python scripts/ablation/run_ablation.py --config scripts/ablation/configs/<config>.py
```

**Dataset Types:**
- `simulation` - Generated time series (default)
- `swat_A1A2` - SWaT A1+A2 combined
- `swat_A1A2_swap` - SWaT with swapped halves
- `wadi_14days_A1` - WaDi 14 days + A1
- `wadi_14days_A2` - WaDi 14 days + A2
- `wadi_A2` - WaDi A2 only

**Template Configs:**
- `scripts/ablation/configs/simulation_test.py`
- `scripts/ablation/configs/swat_A1A2_test.py`
- `scripts/ablation/configs/wadi_14days_A1_test.py`
- `scripts/ablation/configs/README.md` - Migration guide

### Files Modified

- `mae_anomaly/datasets/` - New module (loaders.py, noisy.py, __init__.py)
- `mae_anomaly/utils/` - New module (system.py, experiment.py, __init__.py)
- `scripts/ablation/run_ablation.py` - Dataset type support (lines 1337-1354, 1386-1425, 1106-1125)
- `scripts/run_base_experiments.py` - Import updates (lines 42-52, 635)
- `scripts/ablation/configs/` - New config templates and README
- `ablation_guideline.md` - Section 7 added (Unified Config System)

### Archived Scripts

Moved to `.trash/20260215_run_scripts/` (10 files, 4,742 lines):
- run_mae_baseline.py, run_mae_normal50.py
- run_swat_ablation.py, run_swat_A1A2_swap.py, run_swat_A1A2_normal50.py
- run_wadi_ablation.py, run_wadi_14days_ablation.py, run_wadi_14days_normal50.py, run_wadi.py
- run_base_experiments.py (old version)

### Benefits

- ✅ Single codebase eliminates ~4,700 duplicate lines
- ✅ Easy experiment reproduction (share config file)
- ✅ Centralized bug fixes and improvements
- ✅ Type-safe config validation
- ✅ Backward compatible (DATASET_TYPE defaults to 'simulation')

---

## 2026-02-09 (Update 46): Default Parameter Update — enc2, td4, sd1

### Summary

Updated default model parameters based on WaDi 14days ablation study results. The new defaults (enc=2, td=4, sd=1, p=5, d=128) showed +30.7% PRC-AUC improvement on A1_14days and +13.5% on A2_14days compared to original training data.

### New Default Parameters

| Parameter | New Default | Previous | Reason |
|-----------|-------------|----------|--------|
| num_encoder_layers | **2** | 1 | enc=2 +21.7% PRC on A1_14days |
| num_teacher_decoder_layers | **4** | 2 | td=4 optimal for reconstruction |
| num_student_decoder_layers | **1** | 2 | sd=1 creates better discrepancy signal |
| patch_size | 5 | 5 | Maintained (fine-grained best) |
| d_model | 128 | 128 | Maintained |

### Files Modified

- `mae_anomaly/config.py` - Core default values
- `scripts/run_mae_baseline.py`, `run_mae_normal50.py` - Training scripts
- `scripts/run_wadi_ablation.py`, `run_wadi_14days_ablation.py` - WaDi scripts
- `docs/ARCHITECTURE.md` - Architecture documentation
- `docs/PROJECT_SUMMARY.md` - Project summary

### Key Insight

With 14days normal training data, enc=2 gains +21.7% PRC-AUC for A1, making deeper encoders beneficial when more normal patterns are available. The shallow student (sd=1) with deep teacher (td=4) creates optimal capacity gap for discrepancy-based anomaly detection.

---

## 2026-02-05 (Update 45): WaDi A2 Ablation Study Complete

### Summary

Completed 40 ablation experiments on WaDi A2 dataset (172,803 timesteps, 96 features, 7 attack segments). Best configuration: w100_p5_td3_sd1 (F1=0.5728, ROC-AUC=0.9396). Key finding: td3_sd1 outperforms td4_sd1 on A2 (unlike A1), suggesting moderate teacher depth generalizes better for heterogeneous attack types.

### Key Results

| Metric | Best Value | Configuration |
|--------|------------|---------------|
| F1 Score | 0.5728 | w100_p5_td3_sd1 |
| ROC-AUC | 0.9396 | w100_p5_td3_sd1 |
| PRC-AUC | 0.6146 | w500_p5_td3_sd1 |

### A1 vs A2 Comparison

| Parameter | A1 Optimal | A2 Optimal |
|-----------|------------|------------|
| Teacher Decoder | 4 layers | 3 layers |
| Best F1 | 0.6065 | 0.5728 |

### New Files

| File | Description |
|------|-------------|
| `docs/ablation/WaDi/A2_ANALYSIS.md` | Comprehensive analysis document |
| `results/WaDi/A2/ablation_results.csv` | All experiment metrics in CSV format |

---

## 2026-01-31 (Update 44): Fix Phase 2 Defaults — enc1, lr=2e-3

### Summary

Diagnostic testing revealed Phase 2 defaults (enc2, lr=5e-3) cause catastrophic performance degradation at w500. `num_encoder_layers=2` collapses discrepancy signal (disc_d: 2.44→0.21), making teacher/student outputs nearly identical. Combined enc2+td4 drops roc from 0.9855 to 0.7592. `lr=5e-3` also degrades at w500 (-0.040 roc). Corrected defaults: `num_encoder_layers=1`, `learning_rate=2e-3`. Phase 2 config file updated.

### Corrected Parameters

| Parameter | Corrected | Previous | Reason |
|-----------|-----------|----------|--------|
| num_encoder_layers | 1 | 2 | enc2 collapses disc_d at w500 (0.21 vs 2.44) |
| learning_rate | 2e-3 | 5e-3 | lr=5e-3 too aggressive for w500+d128 |

## 2026-01-31 (Update 43): Phase 2 Experiment Plan & Default Parameter Update

### Summary

Updated model default parameters based on Phase 1 ablation analysis (1,014 evaluations). Created Phase 2 experiment plan with 150 configs (600 total evaluations). New defaults reflect Phase 1 optimal findings: larger window (500), larger model (d128/nh8), deeper decoder (td4), higher learning rate (0.005), lower masking ratio (0.15), and stronger discrepancy training (λ=2.0, k=2.0, alw=2).

### Default Parameter Changes

| Parameter | New | Old | Rationale |
|-----------|-----|-----|-----------|
| seq_length | 500 | 100 | Best disturbing-normal separation (H3) |
| d_model | 128 | 64 | Critical for w500 performance (H7) |
| nhead | 8 | 2 | Best mean roc_auc (0.9694) |
| dim_feedforward | 512 | 256 | d_model × 4 |
| num_encoder_layers | 2 | 1 | el=2-3 improves over el=1 |
| num_teacher_decoder_layers | 4 | 2 | Best overall by mean and max |
| patch_size | 20 | 10 | Optimal for w500 (25 patches) |
| masking_ratio | 0.15 | 0.2 | SNR sweet spot 0.08-0.15 |
| lambda_disc | 2.0 | 0.5 | Eliminates scoring mode gap for mask_after |
| dynamic_margin_k | 2.0 | 1.5 | Higher k helps mask_after disc_d |
| anomaly_loss_weight | 2.0 | 1.0 | Boosts mask_after disc_d +22% |
| dropout | 0.15 | 0.1 | Between 0.1 and 0.2 (phase1 best) |
| shared_mask_token | False | True | Separate mask tokens preferred |

### Code Changes

| Component | Changes |
|-----------|---------|
| `config.py` | Updated all default parameter values |

### Documentation Changes

| File | Changes |
|------|---------|
| `docs/ARCHITECTURE.md` | Updated Default Configuration table |
| `docs/ablation/phase2/PHASE2_PLAN.md` | New file: 150-config Phase 2 experiment plan |
| `docs/CHANGELOG.md` | This entry |

---

## 2026-01-30 (Update 42): Point-Level Evaluation Refactor

### Summary

Refactored all evaluation metrics from patch-level to point-level. Primary metrics (roc_auc, f1, precision, recall) now use point-level scores computed by mean-aggregating patch scores to physical timestamps. PA%K metrics use majority voting with the point-level threshold instead of independent threshold optimization per K.

### Code Changes

| Component | Changes |
|-----------|---------|
| `evaluator.py` | `evaluate()`, `evaluate_by_score_type()`, `get_performance_by_anomaly_type()` refactored to point-level; added `_compute_voted_point_predictions()` and `_compute_pa_k_f1_at_threshold()` helpers |
| `visualization/base.py` | `collect_predictions()` and `collect_all_visualization_data()` now return point-level scores/labels as primary, with patch-level data retained for loss stats and voting |
| `visualization/best_model_visualizer.py` | Updated ROC, threshold, detection examples, comparison plots to use point-level data; removed patch-level masked region highlighting from detection plots |
| `run_ablation.py` | No changes needed — CSV column mapping auto-adapts via `**metrics` unpacking |

### Documentation Changes

| File | Changes |
|------|---------|
| ARCHITECTURE.md | Added "Point-Level Aggregation" section, clarified inference metrics |
| VISUALIZATIONS.md | Updated inference mode description for point-level aggregation |
| ABLATION_STUDIES.md | Clarified point-level labeling in evaluation strategy |

## 2026-01-29 (Update 41): Remove last_patch inference mode

### Summary

Removed `last_patch` inference mode entirely. The system now exclusively uses `all_patches` (iterative per-patch masking with N forward passes). Removed `inference_mode` and `mask_last_n` from Config. Deleted 1020 last_patch result directories. Updated all code, configs, and documentation.

### Code Changes

| File | Changes |
|------|---------|
| `config.py` | Removed `inference_mode` and `mask_last_n` fields |
| `evaluator.py` | Deleted `aggregate_scores_to_point_level()`, `compute_point_level_pa_k()`, `_compute_raw_scores_last_patch()`, `_compute_raw_scores_all_patches()`; simplified all methods to remove branching |
| `trainer.py` | Renamed `last_patch_labels` → `window_labels`; `mask_last_n` → `patch_size` |
| `visualization/base.py` | Removed inference_mode branching in all collect functions |
| `visualization/best_model_visualizer.py` | Removed `self.inference_mode` and all conditional branches |
| `visualization/training_visualizer.py` | `last_patch_labels` → `window_labels`; `mask_last_n` → `patch_size` |
| `visualization/data_visualizer.py` | `config.mask_last_n` → `config.patch_size` |
| `visualization/stage2_visualizer.py` | Removed `mask_last_n` from hyperparameter display |
| `run_ablation.py` | Removed inference_mode loops, suffixes, and INFERENCE_MODES handling |
| `configs/phase1.py` | Removed `INFERENCE_MODES` list and `mask_last_n` from experiments |

### Documentation Changes

| File | Changes |
|------|---------|
| `INFERENCE_MODES.md` | Rewritten: single inference process (removed last_patch section and comparison) |
| `ABLATION_EXPERIMENTS.md` | Variants: 12 → 6 per experiment; Total: 2040 → 1020 |
| `ABLATION_STUDIES.md` | Removed inference modes section; updated variant counts |
| `ARCHITECTURE.md` | Simplified inference time description |
| `VISUALIZATIONS.md` | Removed inference mode handling table |
| `DATASET.md` | `config.mask_last_n` → `config.patch_size` |

### Results Cleanup

Deleted 1020 `*_last` directories from `results/experiments/20260128_012500_phase1/`.

---

## 2026-01-29 (Update 40): evaluate_by_score_type, Documentation Sync & Cleanup

### Summary

Implemented `evaluate_by_score_type()` in evaluator to populate 24 CSV columns (disc_only_*, teacher_recon_*, student_recon_*) that were previously always 0. Added student reconstruction error to evaluator cache. Comprehensive documentation sync across all docs. Removed obsolete scripts and analysis files.

### Evaluator Changes

| Change | Description |
|--------|-------------|
| `evaluate_by_score_type(score_type)` | NEW: Evaluate using individual score components ('disc', 'teacher_recon', 'student_recon') |
| Student recon in cache | `_compute_raw_scores_last_patch()` and `_compute_patch_scores_all_patches()` now return 6-tuple including student_recon |
| 24 CSV columns | disc_only_*, teacher_recon_*, student_recon_* now have real values |

### Documentation Sync

Fixed inconsistencies across all documentation files:

| Parameter | Old Doc Value | Corrected Value |
|-----------|--------------|-----------------|
| Patchify modes | linear/cnn_first/patch_cnn | linear/patch_cnn (cnn_first removed) |
| Default patchify_mode | linear | patch_cnn |
| sliding_window_total_length | 440K / 2.2M | 275K |
| anomaly_interval_scale | 1.5 | 0.75 |
| Scoring modes (ablation) | default/adaptive/disc_only | default/adaptive/normalized |
| Anomaly types | 6 / 11 names | 9 types (10 names including normal) |
| Train/test split | 50/50 | 80/20 |
| Default margin_type | hinge | dynamic |
| teacher_only_warmup_epochs | 1 | 3 |
| Feature count (design doc) | 5 | 8 |

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/evaluator.py` | Added `evaluate_by_score_type()`, student_recon in cache |
| `mae_anomaly/model.py` | Minor updates |
| `mae_anomaly/visualization/base.py` | Optimization updates |
| `mae_anomaly/visualization/best_model_visualizer.py` | ROC comparison methods |
| `CLAUDE.md` | Fixed patchify modes, dataset stats, added evaluator mapping |
| `docs/ARCHITECTURE.md` | Removed cnn_first, fixed defaults, added per-component scoring |
| `docs/DATASET.md` | Fixed total_length, interval_scale, anomaly type counts |
| `docs/ABLATION_STUDIES.md` | Fixed masking ratio, scoring modes, dataset size |
| `docs/INFERENCE_MODES.md` | Fixed scoring mode reference |
| `docs/VISUALIZATIONS.md` | Updated date, dataset sizes, removed CNN-First reference |
| `docs/CHANGELOG.md` | This entry |

### Files Removed

| File | Reason |
|------|--------|
| `scripts/ablation/configs/phase2.py` | Obsolete (merged into phase1) |
| `scripts/analyze_phase1_results.py` | One-off analysis script |
| `scripts/deep_analysis_phase1.py` | One-off analysis script |
| `scripts/generate_phase1_report.py` | One-off report generator |
| `docs/ablation_result/phase1/*` | Stale analysis results |
| `scripts/profile_*.py`, `scripts/benchmark_*.py`, `scripts/verify_*.py` | One-off profiling/verification scripts (moved to .trash/) |

---

## 2026-01-28 (Update 39): Segment-Based PA%K Fix & Documentation Update

### Summary

Fixed critical PA%K (Point-Adjust with K%) metric calculation to use proper segment-based detection rates instead of sample-level approximation. Updated all documentation to match current codebase.

### PA%K Metric Fix

**Problem Identified**:
- `plot_performance_by_anomaly_type_comparison()` was using sample-level `compute_pa_k_adjusted_predictions()` which breaks segment structure when filtering by anomaly_type
- This caused incorrect PA%K calculations in visualization

**Solution**:
- Added segment-based PA%K calculation using `compute_segment_pa_k_detection_rate()` in `best_model_visualizer.py`
- Pre-computes point-level scores for each scoring method
- Uses `test_dataset.anomaly_regions` for proper segment-based detection rate

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/visualization/best_model_visualizer.py` | Added segment-based PA%K support in `plot_performance_by_anomaly_type_comparison()` |
| `docs/ARCHITECTURE.md` | Fixed encoder/decoder layer counts, attention heads, masking_ratio (0.2), added missing parameters |
| `docs/DATASET.md` | Updated sliding_window_total_length (440K), stride (11), window counts |
| `docs/ABLATION_STUDIES.md` | Updated to reflect new ablation framework (run_ablation.py) |
| `docs/CHANGELOG.md` | Added this entry |

### Documentation Sync

Updated documentation to match current config.py defaults:

| Parameter | Old Doc Value | New Value |
|-----------|--------------|-----------|
| Encoder layers | 3 | 1 |
| Teacher decoder layers | 4 | 2 |
| Attention heads | 4 | 2 |
| masking_ratio | 0.4 | 0.2 |
| sliding_window_total_length | 2.2M | 440K |
| sliding_window_stride | 10 | 11 |

### Usage

PA%K metrics are now calculated correctly in all visualizations:
- `plot_performance_by_anomaly_type()` - reads from JSON (auto-benefits)
- `plot_performance_by_anomaly_type_comparison()` - now uses segment-based calculation

---

## 2026-01-27 (Update 38): Comprehensive Phase 1 Deep Analysis & Strategic Phase 2 Planning

### Summary

Ultra-deep analysis of 1,398 Phase 1 ablation experiments across 10 strategic focus areas. Generated comprehensive insights document and created 150 Phase 2 experiments organized into 8 targeted groups based on critical findings about balancing discrepancy and reconstruction objectives.

### Key Discoveries

1. **Balance Over Extremes**: High disc_ratio (>4.0) models achieve poor ROC-AUC (0.74-0.88) due to sacrificing reconstruction quality. Best models balance moderate disc_cohens_d (0.9-1.2) with high recon_cohens_d (2.5-3.8).

2. **Reconstruction Quality Dominates**: recon_cohens_d correlates more strongly with ROC-AUC (r=+0.518) than disc_cohens_d (r=+0.445), revealing reconstruction as the foundation for anomaly detection.

3. **Configuration Winners**:
   - Inference: all_patches (+5.9% over last_patch)
   - Scoring: default (best for tuned models)
   - Baseline: w500_p20, d_model=128
   - Teacher-Student: t4s1, t4s2 optimal

4. **Disturbing Normal Challenge**: Best disc_cohens_d_disturbing_vs_anomaly only 0.803 (vs 1.926 for pure normal), identifying this as the key frontier for improvement.

5. **Scarcity of Excellence**: Only 3 models achieved both high disc_d (>1.33) AND high recon_d (>1.73), averaging ROC-AUC 0.942 and PA%80 0.951.

### Analysis Framework (10 Focus Areas)

| Focus Area | Key Finding | Phase 2 Impact |
|------------|-------------|----------------|
| 1. High Disc Ratio | Top 50 models (disc_d 1.88-1.93) average only 0.860 ROC-AUC | GROUP 1: Optimize balance |
| 2. Disc+Recon Balance | Only 3 models meet criteria → GOLDEN ZONE | GROUP 1: Replicate success |
| 3. Modes & Windows | all_patches +0.047 ROC-AUC; w500_p20 strong baseline | GROUP 2: Scale windows |
| 4. Disturbing Separation | 009_w500_p20 achieves 0.803 (best) | GROUP 3: Push beyond 0.85 |
| 5. PA%80 + Disc Ratio | Rare combination, critical for deployment | GROUP 4: Systematic optimization |
| 6. Window-Depth-Masking | Relationships need systematic exploration | GROUP 2, 6, 7 |
| 7. Mask After Optimization | Most top models use mask_after=False | GROUP 8: Lambda tuning |
| 8. Mode Sensitivity | Same model: 0.956 (default) vs 0.928 (adaptive) | GROUP 4: Test systematically |
| 9. High Perf + Disturbing | Achieving both is rare but valuable | GROUP 3: Targeted approach |
| 10. Additional Insights | disc_ratio negatively correlated with ROC-AUC (r=-0.124) | All groups: Avoid extremes |

### Phase 2 Strategic Plan (150 Experiments)

| Group | Experiments | Goal | Strategy |
|-------|-------------|------|----------|
| **1: Balanced Disc+Recon** | 30 | disc_d > 1.2, recon_d > 2.8 | Build on 028_d128_nhead_16, vary masking/lambda |
| **2: Window & Capacity** | 25 | Scaling laws | Test w100/500/1000 with matched capacity |
| **3: Disturbing Separation** | 20 | disc_d_disturbing > 0.85 | Build on 009_w500_p20, vary k/lambda/weight |
| **4: PA%80 Optimization** | 20 | PA%80 > 0.970 | Large windows, high capacity, mode testing |
| **5: Teacher-Student Ratios** | 15 | Optimal T:S balance | Systematic t1s1 through t6s1, balanced ratios |
| **6: Masking Strategy** | 15 | Optimal ratios per d_model | d128: [0.05-0.35], d256: [0.60-0.90] |
| **7: Architecture Depth** | 15 | Optimal encoder-decoder | Systematic depth combinations |
| **8: Lambda Discrepancy** | 10 | Optimal loss weighting | Fine-grained [0.5-3.0] |

### Files Created

| File | Purpose |
|------|---------|
| `docs/ablation_result/PHASE1_COMPREHENSIVE_ANALYSIS.md` | 📄 Complete analysis report (13KB) |
| `docs/ablation_result/phase1_analysis_report.md` | 📊 Executive summary with tables |
| `docs/ablation_result/table1_top10_roc_auc.csv` | 🏆 Top 10 models by ROC-AUC |
| `docs/ablation_result/table2_top10_disc_ratio.csv` | 📈 Top 10 by discrepancy ratio |
| `docs/ablation_result/table3_top10_t_ratio.csv` | 🎯 Top 10 by teacher reconstruction ratio |
| `docs/ablation_result/all_experiments.csv` | 💾 All 1,398 results (1.2MB) |
| `scripts/ablation/configs/phase2.py` | ⚙️ 150 Phase 2 experiment configs |
| `scripts/analyze_phase1_results.py` | 🔧 Analysis script |
| `scripts/generate_phase1_report.py` | 📝 Report generator |
| `docs/CHANGELOG.md` | 📋 UPDATED (this entry) |

### Usage

```bash
# Review comprehensive analysis
cat docs/ablation_result/PHASE1_COMPREHENSIVE_ANALYSIS.md

# Review executive summary
cat docs/ablation_result/phase1_analysis_report.md

# Verify Phase 2 config
python scripts/ablation/configs/phase2.py

# Run Phase 2 experiments
python scripts/ablation/run_ablation.py --config configs/phase2.py
```

### Expected Phase 2 Outcomes

1. **10+ models with ROC-AUC > 0.960** (vs Phase 1 best: 0.9624)
2. **5+ models with disc_d > 1.2 AND recon_d > 2.8** (vs Phase 1: only 3)
3. **disc_cohens_d_disturbing_vs_anomaly > 0.85** (vs Phase 1 best: 0.803)
4. **PA%80 ROC-AUC > 0.970** (vs Phase 1 best: 0.965)
5. **Establish scaling laws** for window size vs model capacity
6. **Identify 2-3 production-ready configurations**

### Documentation Philosophy

- **Insight-Driven**: Each experiment group based on specific Phase 1 insight
- **Hypothesis-Testing**: Clear hypotheses with verification criteria
- **Balanced Approach**: Optimize for balance, not single metric extremes
- **Deployment-Ready**: Focus on PA%80 and disturbing normal separation

---

## 2026-01-27 (Update 37): Phase 1 Analysis and Phase 2 Experiment Planning

### Summary

Comprehensive analysis of 1,392 Phase 1 ablation experiments with deep-dive insights across 10 analysis points. Generated 150 Phase 2 experiment configurations organized into 7 thematic tracks based on Phase 1 findings.

### Key Findings

1. **Best Performance:** ROC-AUC=0.9624 with `mask_after=False`, `d_model=128`, `nhead=16`
2. **Highest Disc Ratio:** 4.26 with `mask_after=True`, `dynamic_margin_k=4.0` (but lower ROC-AUC)
3. **Trade-off Identified:** High disc_ratio negatively correlates with performance (-0.45 with recon_ratio)
4. **Window Size:** w500_p20 achieved ROC-AUC=0.9586 (2nd best), warrants further exploration
5. **Inference Mode:** `all_patches` outperforms `last_patch` by +0.046 ROC-AUC

### Phase 2 Experiment Tracks (150 total)

1. **Track 1 (30):** Balanced Performance Optimization - optimize mask_before configs
2. **Track 2 (25):** Window Size Exploration - systematically test w500/1000/1500
3. **Track 3 (25):** High Disc Ratio Optimization - improve ROC while maintaining high disc
4. **Track 4 (20):** Disturbing Normal Discrimination - optimize disturbing vs anomaly separation
5. **Track 5 (20):** Architectural Depth - systematic encoder-decoder depth exploration
6. **Track 6 (15):** Masking Ratio Fine-tuning - fine-grained search in 0.08-0.3 range
7. **Track 7 (15):** Lambda_disc Exploration - systematic lambda values

### Files

| File | Status |
|------|--------|
| `docs/ablation_result/phase1_top_models_tables.md` | NEW (top 10 models by 3 metrics) |
| `docs/ablation_result/phase1_deep_analysis.md` | NEW (10-point analysis, 22 tables) |
| `docs/ablation_result/PHASE1_SUMMARY_AND_PHASE2_PLAN.md` | NEW (executive summary) |
| `scripts/ablation/configs/phase2/20260127_141642_phase2.py` | NEW (150 phase2 configs) |
| `docs/CHANGELOG.md` | UPDATED |

### Usage

```bash
# View analysis results
cat docs/ablation_result/PHASE1_SUMMARY_AND_PHASE2_PLAN.md

# Run Phase 2 experiments
python scripts/ablation/run_ablation.py --config configs/phase2/20260127_141642_phase2.py
```

---

## 2026-01-27 (Update 36): Unified Ablation Study and Visualization Optimization

### Summary

Unified Phase 1 and Phase 2 ablation configs into single Phase 1 (170 experiments). Added parallel visualization support and optimized data collection with `collect_all_visualization_data()` function for ~2x speedup.

### Changes

1. **Unified Ablation Config** (`scripts/ablation/configs/20260127_052220_phase1.py`):
   - Combined 70 (Phase 1) + 100 (Phase 2) = **170 experiments**
   - Unified base config defaults: d_model=64, nhead=2, masking_ratio=0.2
   - Total expected results: 170 × 2 (mask) × 2 (inference) × 3 (scoring) = **2040**

2. **Visualization Optimization** (`mae_anomaly/visualization/base.py`):
   - Added `collect_all_visualization_data()` - merged function for ~2x speedup
   - Combines `collect_predictions()` and `collect_detailed_data()` into single pass
   - Reduces redundant forward passes

3. **Parallel Visualization** (`mae_anomaly/visualization/parallel.py`):
   - New `ParallelVisualizer` class for multiprocessing-based plot generation
   - New `generate_plots_parallel()` helper function
   - Uses file-based data passing to avoid IPC overhead

4. **Module Exports** (`mae_anomaly/visualization/__init__.py`):
   - Added `collect_all_visualization_data` export
   - Added `ParallelVisualizer`, `generate_plots_parallel` exports

### Usage

```bash
# Run unified Phase 1 (170 experiments × 12 variants = 2040 results)
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py
```

### Files

| File | Status |
|------|--------|
| `scripts/ablation/configs/20260127_052220_phase1.py` | NEW (unified) |
| `mae_anomaly/visualization/base.py` | MODIFIED (collect_all_visualization_data) |
| `mae_anomaly/visualization/parallel.py` | NEW |
| `mae_anomaly/visualization/__init__.py` | MODIFIED |
| `docs/ABLATION_EXPERIMENTS.md` | UPDATED |
| `docs/VISUALIZATIONS.md` | UPDATED |

---

## 2026-01-27: Ablation Study Framework Refactoring

### Summary

Refactored ablation study scripts into a unified, modular framework with separate config files.

### Changes

1. **Unified Runner** (`scripts/ablation/run_ablation.py`):
   - Single entry point for all ablation studies
   - Dynamic config loading from Python files
   - Background visualization with concurrency control
   - Skip-existing and experiment filtering support

2. **Config Files** (`scripts/ablation/configs/`):
   - Modular format for easy extension

3. **Visualization Fix** (`mae_anomaly/visualization/best_model_visualizer.py`):
   - Fixed `best_model_score_contribution_trends.png` for adaptive/normalized modes
   - Now correctly recalculates disc score weights from raw history values

### Usage

```bash
# Run unified Phase 1
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py

# Run specific experiments
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py \
    --experiments 001_default 002_window_200
```

### Files

| File | Status |
|------|--------|
| `scripts/ablation/run_ablation.py` | NEW |
| `scripts/ablation/configs/__init__.py` | NEW |
| `scripts/ablation/configs/20260127_052220_phase1.py` | NEW |
| `scripts/ablation/run_ablation_experiments_*.py` | DEPRECATED |
| `docs/ABLATION_EXPERIMENTS.md` | UPDATED |
| `mae_anomaly/visualization/best_model_visualizer.py` | FIXED |

---

## 2026-01-25 (Update 34): Mixed Precision Training (AMP) Support

### Summary

Added Automatic Mixed Precision (AMP) training support for faster training and reduced memory usage.

### Performance Impact (RTX 3080 Ti)

| Metric | No AMP | AMP | Improvement |
|--------|--------|-----|-------------|
| Training time | 2.89s | 2.40s | **1.20x** |
| Inference time | 4.32s | 3.52s | **1.23x** |
| Training memory | 449 MB | 272 MB | **40% ↓** |
| Inference memory | 1062 MB | 437 MB | **59% ↓** |

### Changes

1. **Config**: Added `use_amp: bool = True` option
2. **Epsilon values**: Changed all `1e-8` → `1e-4` for float16 numerical stability
3. **Trainer**: Added `autocast` and `GradScaler` for mixed precision training
4. **Evaluator**: Added `autocast` for mixed precision inference

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/config.py` | Added `use_amp` option |
| `mae_anomaly/loss.py` | 14x epsilon update |
| `mae_anomaly/trainer.py` | AMP support + 8x epsilon update |
| `mae_anomaly/evaluator.py` | AMP support + 6x epsilon update |

### Notes

- AMP is enabled by default (`use_amp=True`)
- Requires GPU with Tensor Cores (Volta+) for best speedup
- Accuracy is preserved (ROC-AUC difference < 0.01)

---

## 2026-01-25 (Update 33): Performance Optimization - Batched all_patches and Training Params

### Summary

Major performance improvements: batched `all_patches` inference (7x speedup), batch_size=1024, learning_rate=5e-3.

### Changes

1. **Batched all_patches Inference** (~7x speedup):
   - `evaluator.py`: `_compute_patch_scores_all_patches()` now processes all patches in single forward pass
   - `visualization/base.py`: `collect_predictions()` and `collect_detailed_data()` also optimized
   - Before: 10 forward passes per batch (one per patch)
   - After: 1 forward pass per batch (all patches expanded in batch dimension)

2. **Updated Training Parameters**:
   - `batch_size`: 32 → 1024 (better GPU utilization, ~0.6GB VRAM)
   - `learning_rate`: 2e-3 → 5e-3 (faster convergence with larger batch)

3. **Enabled cuDNN Benchmark**:
   - `cudnn.benchmark = True` for auto-tuned convolution algorithms
   - Additional ~20% training speedup

### Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| all_patches per batch | 23.87 ms | 3.43 ms | **7.0x** |
| Training throughput | - | ~3x faster | GPU utilization |

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/config.py` | batch_size=1024, learning_rate=5e-3, cudnn.benchmark=True |
| `mae_anomaly/evaluator.py` | Batched all_patches in `_compute_patch_scores_all_patches()` |
| `mae_anomaly/visualization/base.py` | Batched all_patches in `collect_predictions()`, `collect_detailed_data()` |
| `docs/DATASET.md` | Updated DataLoader example |
| `docs/ABLATION_EXPERIMENTS.md` | Updated learning_rate default |
| `docs/ARCHITECTURE.md` | Updated learning_rate default |

---

## 2026-01-25 (Update 32): Visualization Cleanup and all_patches Mode Fixes

### Summary

Removed redundant visualization functions, fixed `all_patches` mode visualizations, and added new score contribution epoch trends plot.

### Changes

1. **Removed Visualization Functions** (simplification):
   - `plot_score_distribution()` - redundant with score_contribution_analysis
   - `plot_score_components()` - redundant with score_contribution_analysis
   - `plot_teacher_student_comparison()` - not essential for analysis
   - `plot_hypothesis_verification()` - not essential for analysis
   - `plot_feature_contribution_analysis()` - not essential for analysis

2. **Fixed all_patches Mode Visualizations**:
   - Added `_patch_idx_to_window_idx()` helper for index conversion
   - Fixed `plot_detection_examples()`, `plot_case_study_gallery()`, `_plot_sample_detail()` to use window index
   - Fixed `plot_reconstruction_examples()` to skip masked region shading in all_patches mode

3. **Added New Visualization**:
   - `plot_score_contribution_epoch_trends()`: Stacked area plots showing recon/disc score contributions over epochs for each anomaly type (similar to J-L plots), with unified y-axis and starting from epoch 5

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/visualization/best_model_visualizer.py` | Removed 5 functions, added helper and new plot, fixed all_patches mode |
| `docs/VISUALIZATIONS.md` | Updated visualization list and API examples |

---

## 2026-01-25 (Update 31): Fix Dimension Mismatch in collect_detailed_data for all_patches Mode

### Summary

Bug fix: `collect_detailed_data()` now returns consistent window-level shapes for both inference modes.

### Problem

In `all_patches` mode, `collect_detailed_data()` returned mismatched shapes:
- `teacher_errors`, `student_errors`: (n_windows, seq_length) = (2625, 100)
- `labels`, `sample_types`: flattened to (n_windows × num_patches,) = (26250,)

This caused `IndexError` in visualization functions like `plot_teacher_student_comparison()` when using labels as boolean masks on errors.

### Solution

Changed `collect_detailed_data()` to keep window-level labels for `all_patches` mode:
- Use `last_patch_labels` instead of flattened `patch_labels`
- Keep `sample_types` at window level instead of expanding

Note: `collect_predictions()` correctly uses patch-level labels since it also returns patch-level scores for metrics.

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/visualization/base.py` | `collect_detailed_data()` uses window-level labels for all_patches mode |

### Impact

- All 18 visualization files now generate correctly for `all_patches` mode
- Previously only 5 files were generated before the error occurred

---

## 2026-01-25 (Update 30): Fix Visualization Functions to Respect inference_mode

### Summary

Bug fix: `collect_predictions()` and `collect_detailed_data()` in visualization/base.py now respect `config.inference_mode` setting instead of always using last_patch mode.

### Problem

Confusion matrices and other visualizations were identical for both `last_patch` and `all_patches` inference modes because visualization functions ignored the `inference_mode` config:
- Always masked only the last patch
- Always used `last_patch_labels`

### Solution

Updated both functions to handle `inference_mode`:

**For `all_patches` mode:**
- Mask each patch one at a time (N forward passes)
- Compute patch-level labels from `point_labels`
- Flatten scores/labels to (n_windows × num_patches,)

**For `last_patch` mode:**
- Original behavior preserved

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/visualization/base.py` | `collect_predictions()`, `collect_detailed_data()` now check `inference_mode` |
| `docs/ARCHITECTURE.md` | Added inference_mode documentation |
| `docs/VISUALIZATIONS.md` | Added inference_mode handling section |

### Impact

- Confusion matrices now correctly differ between inference modes
- ROC curves, score distributions match evaluation methodology
- Ablation experiments restarted with fix applied

---

## 2026-01-25 (Update 29): Point-Level PA%K with Stride=1 Sliding Window

### Summary

Major update to evaluation methodology: Test set now uses stride=1 sliding windows without downsampling, enabling proper point-level PA%K evaluation with window score aggregation.

### Key Changes

1. **Model Parameters Updated**
   - `patch_size`: 4 → **10** (larger patches for better context)
   - `num_patches`: 25 → **10** (seq_length / patch_size = 100/10)
   - `mask_last_n`: 4 → **10** (matches patch_size)

2. **Test Set Evaluation**
   - Stride forced to 1 for test split (each timestep covered by multiple windows)
   - Downsampling disabled by default (full sliding window coverage)
   - Window scores aggregated to point-level for PA%K

3. **Point-Level Aggregation Methods**
   - **Voting** (default): Majority vote of binary predictions
   - **Mean**: Average of window scores per timestep
   - **Median**: Median of window scores per timestep

### Window Coverage with Stride=1

```
Window w's last patch: [w+90, w+99] (10 timesteps)
Each timestep covered by up to 10 windows
```

### Metrics Separation

| Metric Type | Level | Notes |
|-------------|-------|-------|
| ROC-AUC, F1, Precision, Recall | Sample (window) | Unchanged |
| PA%K F1, PA%K ROC-AUC | **Point (timestep)** | Aggregated via voting |

### Files Modified

| File | Changes |
|------|---------|
| `mae_anomaly/config.py` | `patch_size=10`, `num_patches=10`, `mask_last_n=10`, `point_aggregation_method` |
| `mae_anomaly/dataset_sliding.py` | Stride=1 forced for test, `window_start_indices` added |
| `mae_anomaly/evaluator.py` | `aggregate_scores_to_point_level()`, `compute_point_level_pa_k()` |
| `scripts/run_experiments.py` | Pass `test_dataset` to Evaluator |
| `scripts/run_temp_experiments.py` | Pass `test_dataset` to Evaluator |
| `docs/ARCHITECTURE.md` | Updated patch dimensions |
| `docs/DATASET.md` | Added Point-Level PA%K section |

### Configuration

```python
# New config parameter
config.point_aggregation_method = 'voting'  # 'mean', 'median', 'voting'
```

### Backwards Compatibility

- Evaluator accepts optional `test_dataset` parameter
- Without `test_dataset`, falls back to sample-level PA%K
- Train set behavior unchanged (configurable stride)

---

## 2026-01-25 (Update 28): Comprehensive PA%K Metrics (K=10,20,50,80)

### Summary

Extended PA%K evaluation to compute both F1-score and ROC-AUC for K=10%, 20%, 50%, 80% (total 8 metrics). Added PA%K ROC-AUC computation that applies segment adjustment at each threshold level.

### Key Features

1. **PA%K ROC-AUC Algorithm**: For each threshold, binarize → apply PA%K adjustment → compute TPR/FPR → build ROC curve
2. **8 PA%K Metrics**: F1 + ROC-AUC for each K value (10, 20, 50, 80)
3. **9-Subplot Visualization**: Compare Point-wise, PA%10 (lenient), PA%80 (strict)

### New Metrics

| Metric | Description |
|--------|-------------|
| `pa_10_f1`, `pa_10_roc_auc` | PA%10 (very lenient, 10% segment detection) |
| `pa_20_f1`, `pa_20_roc_auc` | PA%20 (lenient) |
| `pa_50_f1`, `pa_50_roc_auc` | PA%50 (moderate) |
| `pa_80_f1`, `pa_80_roc_auc` | PA%80 (strict, 80% segment detection) |

### Changes

#### 1. Core Implementation (evaluator.py)
- Added `compute_pa_k_roc_auc()` function for threshold-aware PA%K ROC-AUC
- `evaluate()` returns 8 PA%K metrics (F1 + ROC-AUC × 4 K values)
- `get_performance_by_anomaly_type()` includes all 4 K values for detection rates

#### 2. Experiment Scripts
- `run_experiments.py` - Saves all 8 PA%K metrics to results
- `run_temp_experiments.py` - Displays PA%K table in console output

#### 3. Visualization (best_model_visualizer.py)
- New 3×3 grid (9 subplots) showing:
  - Row 1: Point-wise, PA%10, PA%80 detection rates
  - Row 2: All PA%K comparison, PA%10 vs PA%80, Mean scores
  - Row 3: Consistency gap, Sample distribution, Summary statistics

---

## 2026-01-25 (Update 27): PA%K (Point-Adjust with K%) Evaluation Metric

### Summary

Added PA%K evaluation metric (default K=20%) for more realistic time series anomaly detection evaluation. PA%K is a segment-level adjustment that considers an anomaly segment as "detected" if at least K% of its points are flagged.

### Motivation

Point-wise F1 score can be overly harsh for time series anomaly detection because:
- If a model detects 9 out of 10 anomaly points but misses 1, point-wise F1 penalizes heavily
- In practice, detecting ANY point within an anomaly segment is often sufficient for alerting
- PA%K provides a more realistic evaluation by giving credit for partial segment detection

### PA%K Algorithm

```
For each contiguous anomaly segment:
    if (detected_points / total_points) >= K%:
        All points in segment count as DETECTED (TP)
    else:
        All points in segment count as NOT DETECTED (FN)
```

With K=20% (PA%20):
- A segment of 100 points needs only 20 detected points to count as fully detected
- Balanced between leniency and rigor for real-world alerting scenarios

### Changes

#### 1. Core Implementation (evaluator.py)

- Added `compute_pa_k_adjusted_predictions()` function
- Added `compute_pa_k_metrics()` function returning precision, recall, F1
- Updated `evaluate()` to include `pa_k_f1`, `pa_k_precision`, `pa_k_recall`
- Updated `get_performance_by_anomaly_type()` to include `pa_k_detection_rate` per type

#### 2. Experiment Scripts

- `run_experiments.py` - Added PA%K columns to Stage 2 results
- `run_temp_experiments.py` - Added PA%K to summary display and console output

#### 3. Visualization (best_model_visualizer.py)

- Updated `plot_performance_by_anomaly_type()` to show side-by-side comparison:
  - Point-wise detection rate (lighter bars)
  - PA%20 detection rate (darker bars)

### New Metrics in Experiment Results

| Metric | Description |
|--------|-------------|
| `pa_k_f1` | PA%K F1 score (K=20%) |
| `pa_k_precision` | PA%K precision |
| `pa_k_recall` | PA%K recall |
| `pa_k_detection_rate` | Per-anomaly-type PA%K detection rate |

### Files Modified

- `mae_anomaly/evaluator.py` - Core PA%K implementation
- `mae_anomaly/visualization/best_model_visualizer.py` - Visualization update
- `scripts/run_experiments.py` - Result column additions
- `scripts/run_temp_experiments.py` - Display and column updates

---

## 2026-01-25 (Update 26): Remove point_spike Anomaly Type

### Summary

Removed `point_spike` (formerly type 7) from anomaly types. Pattern-based anomalies are renumbered from 8-10 to 7-9.

### Rationale

Point spike anomalies were:
1. Too similar to the existing `spike` anomaly type
2. Very short duration (3-5 timesteps) made them unrealistic for most real-world monitoring scenarios
3. Random feature selection made them inconsistent for systematic evaluation

### Changes

#### Before → After

| Category | Before | After |
|----------|--------|-------|
| Value-based | Types 1-7 | Types 1-6 |
| Pattern-based | Types 8-10 | Types 7-9 |
| Total | 10 types | 9 types |

#### Anomaly Type Renumbering

| Old ID | New ID | Name |
|--------|--------|------|
| 7 | (removed) | point_spike |
| 8 | 7 | correlation_inversion |
| 9 | 8 | temporal_flatline |
| 10 | 9 | frequency_shift |

### Files Modified

- `mae_anomaly/dataset_sliding.py` - Removed point_spike, renumbered types
- `mae_anomaly/visualization/base.py` - Updated anomaly info
- `mae_anomaly/visualization/data_visualizer.py` - Removed point_spike comment
- `docs/DATASET.md` - Updated all references

---

## 2026-01-25 (Update 25): Pattern-Only Anomalies for Meaningful Detection Validation

### Summary

Added 3 new pattern-based anomaly types that maintain normal value ranges but break temporal/correlation patterns. This allows distinguishing between trivial value-based detection (detecting unusual VALUES) and meaningful pattern-based detection (detecting unusual PATTERNS).

### Problem Statement

Previously, ALL anomaly types were ADDITIVE (values increase beyond normal range). This made it impossible to determine if the model was:
- Detecting anomalies because of unusual **VALUES** (trivial statistical detection)
- Detecting anomalies because of unusual **PATTERNS** (meaningful anomaly detection)

### Changes

#### 1. Added 3 Pattern-Only Anomaly Types (dataset_sliding.py)

| Type ID | Name | Description | Pattern Break |
|---------|------|-------------|---------------|
| 7 | correlation_inversion | CPU-Memory correlation breaks | Cross-feature correlation |
| 8 | temporal_flatline | Values freeze (stuck sensor) | Temporal continuity |
| 9 | frequency_shift | Unusual oscillation frequency | Normal periodicity |

All pattern-based anomalies use `np.clip(value, 0.15, 0.85)` to ensure values stay within normal range.

#### 2. Added ANOMALY_CATEGORY Metadata

```python
ANOMALY_CATEGORY = {
    1: 'value', 2: 'value', 3: 'value', 4: 'value',
    5: 'value', 6: 'value',
    7: 'pattern', 8: 'pattern', 9: 'pattern'
}
```

#### 3. Fixed Y-axis Unification in loss_by_anomaly_type (best_model_visualizer.py)

Applied unified y-axis limits across all subplots for fair visual comparison.

#### 4. Added Value vs Pattern Comparison Visualization

New `plot_value_vs_pattern_comparison()` method showing:
- Score distribution comparison (Normal vs Value-based vs Pattern-based)
- Box plot comparison
- Detection rate comparison
- Loss components comparison

#### 5. Distinct Colors for Pattern-Based Anomalies (base.py)

Pattern-based anomalies use cool colors (blue/purple) to visually distinguish from warm-colored value-based anomalies.

### Files Modified

- `mae_anomaly/dataset_sliding.py` - Added anomaly types, category, injection methods
- `mae_anomaly/__init__.py` - Exported ANOMALY_CATEGORY
- `mae_anomaly/visualization/base.py` - Updated get_anomaly_colors()
- `mae_anomaly/visualization/best_model_visualizer.py` - Added visualization, fixed y-axis

### Documentation Updates

- CHANGELOG.md - This entry

---

## 2026-01-24 (Update 24): Per-Feature Min-Max Normalization

### Summary

Replaced data clipping (`np.clip(signals, 0, 1)`) with per-feature min-max normalization. This preserves relative anomaly magnitudes and eliminates boundary artifacts.

### Changes

#### 1. Added Normalization Function

**Modified Files**:
- `mae_anomaly/dataset_sliding.py`

**New Function**:
```python
def _normalize_per_feature(signals: np.ndarray) -> np.ndarray:
    """Per-feature min-max normalization to [0, 1] range.

    This is preferred over clipping because:
    1. Preserves relative magnitude of anomalies (spikes won't be capped)
    2. No artificial saturation at boundaries
    3. More realistic simulation of real-world data preprocessing
    """
    signals = signals.copy()
    for f in range(signals.shape[1]):
        min_val = signals[:, f].min()
        max_val = signals[:, f].max()
        if max_val - min_val > 1e-8:
            signals[:, f] = (signals[:, f] - min_val) / (max_val - min_val)
        else:
            signals[:, f] = 0.5
    return signals.astype(np.float32)
```

---

#### 2. Replaced Clipping with Normalization

**Locations Changed**:

| Method | Before | After |
|--------|--------|-------|
| `_generate_simple_normal_series()` | `np.clip(signals, 0, 1)` | `_normalize_per_feature(signals)` |
| `generate()` | `np.clip(signals, 0, 1)` | `_normalize_per_feature(signals)` |

---

#### 3. Why This Change?

| Aspect | Clipping | Min-Max Normalization |
|--------|----------|----------------------|
| Spike anomalies | Capped at 1.0 (info loss) | Full magnitude preserved |
| Boundary behavior | Flat saturation | Natural distribution |
| Relative magnitudes | Distorted | Preserved exactly |
| Real-world similarity | Artificial | Matches preprocessing |

---

### Documentation Updates

- **DATASET.md**: Added "Data Normalization" section, updated Safety Constraints table
- **CHANGELOG.md**: This entry

---

## 2026-01-24 (Update 23): Dataset Visualization Improvements

### Summary

Improved dataset visualization quality by using dedicated datasets for plotting (without anomaly contamination), added before/after comparisons at same window positions, and cleaned up redundant/misleading visualizations.

### Changes

#### 1. Added `inject_anomalies` Parameter to Generator

**Modified Files**:
- `mae_anomaly/dataset_sliding.py`

**New Parameter**:
```python
def generate(self, inject_anomalies: bool = True) -> Tuple[...]:
    """
    Args:
        inject_anomalies: If True (default), inject anomalies.
                          If False, return pure normal data.
    """
```

This allows visualization code to generate clean normal data for complexity feature demonstrations.

---

#### 2. Improved Dataset Visualizations

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:

| Function | Change |
|----------|--------|
| `plot_anomaly_generation_rules()` | Show only 1 example per anomaly type (was 2) |
| `plot_normal_complexity_features()` | Uses `inject_anomalies=False` for clean comparison |
| `plot_complexity_comparison()` | Uses `inject_anomalies=False` for clean comparison |
| `plot_complexity_vs_anomaly()` | **Completely redesigned**: Before/after comparison at same window position |
| `plot_dataset_statistics()` | **Removed** (hardcoded values were misleading) |

**New `plot_complexity_vs_anomaly()` Design**:
- Row 1: Complexity features (gray=before, blue=after) at same window position
- Row 2: Anomaly injection (gray=before, red=after) at same window position
- Allows clear visualization of what each feature/anomaly actually changes

---

#### 3. Stage 1 Visualization Cleanup

**Modified Files**:
- `mae_anomaly/visualization/experiment_visualizer.py`

**Changes**:

| Function | Change |
|----------|--------|
| `plot_metric_correlations()` | **Removed** (not useful for hyperparameter analysis) |
| `plot_parallel_coordinates()` | **Added interpretation guide** panel explaining how to read the plot |

---

### Documentation Updates

- **VISUALIZATIONS.md**: Updated tables and usage examples
- **CHANGELOG.md**: This entry

---

## 2026-01-24 (Update 22): Comprehensive Visualization Style Consistency

### Summary

Extended VIS_COLORS with additional semantic color keys and applied consistent styling across ALL visualization files, eliminating hardcoded color values.

### Changes

#### 1. Extended VIS_COLORS Constants

**Modified Files**:
- `mae_anomaly/visualization/base.py`

**New Color Keys Added**:
```python
VIS_COLORS = {
    # Primary data types (existing)
    'normal': '#3498DB',
    'anomaly': '#E74C3C',
    'disturbing': '#F39C12',
    'teacher': '#27AE60',
    'student': '#9B59B6',
    'total': '#2ECC71',
    # Region highlighting (NEW)
    'anomaly_region': '#E74C3C',
    'masked_region': '#F1C40F',
    'normal_region': '#27AE60',
    # Darker variants (NEW)
    'normal_dark': '#2980B9',
    'anomaly_dark': '#C0392B',
    'student_dark': '#8E44AD',
    # Detection outcomes (NEW)
    'true_positive': '#27AE60',
    'true_negative': '#3498DB',
    'false_positive': '#F39C12',
    'false_negative': '#E74C3C',
    # General purpose (NEW)
    'baseline': 'black',
    'reference': 'gray',
    'threshold': '#27AE60',
}
```

---

#### 2. Applied VIS_COLORS Across All Visualizers

**Modified Files**:
- `mae_anomaly/visualization/best_model_visualizer.py`
- `mae_anomaly/visualization/experiment_visualizer.py`
- `mae_anomaly/visualization/stage2_visualizer.py`
- `mae_anomaly/visualization/training_visualizer.py`
- `mae_anomaly/visualization/data_visualizer.py`
- `mae_anomaly/visualization/architecture_visualizer.py`

**Changes**:
- Replaced ALL hardcoded hex color values (e.g., `'#3498DB'`) with `VIS_COLORS['normal']`
- Replaced ALL hardcoded color names (e.g., `'red'`, `'yellow'`) with `VIS_COLORS` keys
- Added VIS_COLORS import to files that were missing it
- Used semantic color keys (e.g., `'anomaly_region'` for highlighting anomalies)

---

### Documentation Updates

- **VISUALIZATIONS.md**: Updated VIS_COLORS table with all new keys
- **CHANGELOG.md**: This entry

---

## 2026-01-24 (Update 21): Self-Distillation Training Improvements

### Summary

Added encoder gradient detachment for student decoder, configurable warm-up epochs, detailed learning curve visualization, and consistent color/marker scheme across all visualizations.

### Changes

#### 1. Encoder Gradient Detachment for Student Decoder

**Modified Files**:
- `mae_anomaly/model.py`

**Changes**:
- Student decoder now receives `.detach()`ed encoder output
- Encoder is only updated by teacher reconstruction loss
- Prevents student's conflicting objectives from corrupting encoder representations

**Implementation**:
```python
# In forward():
if self.config.use_student:
    student_latent = latent.detach()  # Detach encoder output
    student_output = self.student_decoder(student_latent)
```

---

#### 2. Configurable Teacher-Only Warm-up Epochs

**Modified Files**:
- `mae_anomaly/config.py`
- `mae_anomaly/trainer.py`
- `mae_anomaly/loss.py`

**New Parameter**:
- `teacher_only_warmup_epochs: int = 1` (default)

**Changes**:
- First N epochs train only teacher model (no discrepancy/student loss)
- Added `teacher_only` parameter to loss function
- Allows teacher to learn basic reconstruction before introducing discrepancy

---

#### 3. Detailed Learning Curve Visualization

**Modified Files**:
- `mae_anomaly/loss.py`
- `mae_anomaly/trainer.py`
- `mae_anomaly/visualization/best_model_visualizer.py`
- `scripts/visualize_all.py`

**New Metrics Tracked**:
- `train_teacher_recon_normal`: Teacher recon loss on normal samples
- `train_teacher_recon_anomaly`: Teacher recon loss on anomaly samples
- `train_student_recon_normal`: Student recon loss on normal samples
- `train_student_recon_anomaly`: Student recon loss on anomaly samples

**New Visualization**: `learning_curve.png`
- 2x3 grid showing detailed loss breakdown:
  - Teacher Reconstruction (Normal vs Anomaly)
  - Student Reconstruction (Normal vs Anomaly)
  - Discrepancy Loss (Normal vs Anomaly)
  - Normal Data: Teacher vs Student
  - Anomaly Data: Teacher vs Student
  - All Losses Combined

---

#### 4. Consistent Visualization Color/Marker Scheme

**Modified Files**:
- `mae_anomaly/visualization/base.py`
- `mae_anomaly/visualization/__init__.py`
- `mae_anomaly/visualization/best_model_visualizer.py`

**New Style Constants** (in `base.py`):
```python
VIS_COLORS = {
    'normal': '#3498DB',      # Blue for normal data
    'anomaly': '#E74C3C',     # Red for anomaly data
    'disturbing': '#F39C12',  # Orange for disturbing normal
    'teacher': '#27AE60',     # Green for teacher model
    'student': '#9B59B6',     # Purple for student model
    'total': '#2ECC71',       # Green for totals
}

VIS_MARKERS = {
    'discrepancy': 's',       # Square for discrepancy loss
    'teacher_recon': 'o',     # Circle for teacher reconstruction
    'student_recon': '^',     # Triangle for student reconstruction
    'total': 'D',             # Diamond for total/combined
}
```

**Applied to**:
- `plot_learning_curve()`: Full color/marker scheme
- `plot_discrepancy_trend()`: Consistent colors
- `plot_pure_vs_disturbing_normal()`: Consistent colors for bar charts

---

### Documentation Updates

- **ARCHITECTURE.md**: Added encoder gradient detachment and warm-up epochs documentation
- **VISUALIZATIONS.md**: Added VIS_COLORS/VIS_MARKERS documentation and learning_curve.png
- **CHANGELOG.md**: This entry

---

## 2026-01-23 (Update 20): Quick Search Dataset Configuration

### Changes
- `quick_length`: 100,000 → 200,000 timesteps
- `quick_train_ratio`: 0.3 → 0.2 (20% train, 80% test)
- "Anomaly Types" → "Anomaly Types (samples)" for clarity
- Removed sample count warning messages

### Files Modified
- `scripts/run_experiments.py`
- `mae_anomaly/dataset_sliding.py`

---

## 2026-01-23 (Update 19): Enhanced Dataset Statistics Display

### Changes
- Now displays **3 dataset views**: Train Set (Raw), Test Set (Raw), Test Set (Downsampled)
- Each view shows **Anomaly Types** distribution (per sample, not per region)
- Clearer output format for experiment monitoring

### Output Format
```
[Quick Dataset - Train Set (Raw)]
  - Pure Normal: X,XXX (XX.X%)
  - Anomaly: XXX (X.X%)
  Anomaly Types:
    - spike: XX
    - memory_leak: XX
    ...

[Quick Dataset - Test Set (Raw)]
  ...

[Quick Dataset - Test Set (Downsampled to 65%:15%:25%)]
  ...
```

### Files Modified
- `scripts/run_experiments.py`

---

## 2026-01-23 (Update 18): Train/Test Set Composition Fix

### Problem
- Only test set statistics were displayed, train set was missing
- Test set ratios were hardcoded as absolute counts (1200:300:500)

### Changes

#### 1. Train/Test Statistics Display
- Now shows both **Train Set (Raw)** and **Test Set (Raw)** statistics
- Train set: no downsampling, natural distribution (~5% anomaly from interval_scale)
- Test set: shows raw distribution + target ratio info

#### 2. Test Set Ratio-Based Downsampling
- **Before**: Hardcoded counts (1200:300:500 = 60:15:25)
- **After**: Ratio-based (65:15:25) scaled to `num_test_samples`
- Config now uses `test_ratio_*` instead of `test_target_*`

#### 3. Dataset Composition
| Split | Pure Normal | Disturbing | Anomaly | Downsampling |
|-------|-------------|------------|---------|--------------|
| Train | Natural | Natural | ~5% | None |
| Test | 65% | 15% | 25% | Yes |

### Files Modified
- `mae_anomaly/config.py`
- `scripts/run_experiments.py`

---

## 2026-01-23 (Update 17): Fix Anomaly Ratio in Quick Search

### Problem
- Previous fix scaled interval proportionally: `quick_interval_scale = base * (quick/full)`
- This reduced interval → more frequent anomalies → 19% anomaly ratio (too high)

### Solution
- Use same `interval_scale` for both quick and full search
- Anomaly ratio determined by interval_scale, not data length
- Consistent ~5% anomaly ratio regardless of dataset size

### Files Modified
- `scripts/run_experiments.py`

---

## 2026-01-23 (Update 16): Quick Search Dataset Size Increase

### Changes
- `quick_length`: 66000 → 100000 (more data for quick search)
- Warning threshold: 200 → 300 (suppress warnings when samples >= 300)

### Files Modified
- `scripts/run_experiments.py`
- `mae_anomaly/dataset_sliding.py`

---

## 2026-01-23 (Update 15): Reduce Periodicity in Complex Normal Data

### Summary

Improved normal data generation to be less strictly periodic, making anomaly detection more challenging and realistic.

### Changes

#### 1. Remove Hard Clipping
- **Before**: Normal data was clipped to `[0.05, 0.70]` range
- **After**: No clipping - natural value distribution
- Reason: Hard clipping made normal data unrealistically bounded and easy to classify

#### 2. Irrational Frequency Ratios
- **Before**: `freq2 ≈ freq1/10`, `freq3 ≈ freq1/50` (integer-like ratios)
- **After**: `freq2 = freq1/(π×[2.8-3.5])`, `freq3 = freq1/(π²×[1.5-2.5])`
- Reason: Integer ratios cause beat patterns to repeat; irrational ratios (π-based) prevent exact repetition

#### 3. Phase Jitter
- **New feature**: Slowly-varying phase offset added to sinusoidal components
- Parameters: `enable_phase_jitter=True`, `phase_jitter_sigma=0.002`, `phase_jitter_smoothing=500`
- Applied with decreasing weight per frequency: fast (1.0), medium (0.7), slow (0.4)
- Result: Even with same frequencies, patterns drift over time

### New NormalDataComplexity Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_phase_jitter` | True | Enable phase jitter |
| `phase_jitter_sigma` | 0.002 | Random walk step size |
| `phase_jitter_smoothing` | 500 | Smoothing window |

### Files Modified
- `mae_anomaly/dataset_sliding.py`
- `docs/DATASET.md`
- `docs/CHANGELOG.md`

---

## 2026-01-23 (Update 14): Experiment Configuration Updates

### Summary

Simplified num_patches options, doubled full search dataset, and improved warning thresholds.

### Changes

#### 1. `num_patches` Grid Reduction
- **Before**: `[10, 25, 50]` (3 values)
- **After**: `[10, 25]` (2 values)
- Reason: 50 patches = 2 timesteps per patch, too granular for effective pattern learning

#### 2. Full Search Dataset Size Doubled
- **Before**: `full_length = 220000`
- **After**: `full_length = 440000`
- Provides more training data for Stage 2 full search

#### 3. Warning Threshold for Sample Count
- Warnings now only appear when sample count < 200 (previously: any shortage)
- Reduces noise during quick searches with limited data

#### 4. Grid Combinations
- **Before**: 2×2×3×3×2×2×2×2×2 = 1152 combinations
- **After**: 2×2×2×3×2×2×2×2×2 = 768 combinations

### Files Modified
- `scripts/run_experiments.py`
- `mae_anomaly/dataset_sliding.py`
- `docs/ABLATION_STUDIES.md`
- `docs/VISUALIZATIONS.md`

---

## 2026-01-23 (Update 13): MAE Architecture Enhancements

### Summary

Added two new architecture parameters for standard MAE masking and separate mask tokens, along with experiment infrastructure improvements.

### New Parameters

#### 1. `mask_after_encoder` (config.py)
- **False (default)**: Mask tokens go through encoder (current behavior)
- **True**: Standard MAE - encode visible patches only, insert mask tokens before decoder

**Implementation**:
- Added `_encode_visible_only()` method: Encodes only visible patches
- Added `_insert_mask_tokens_and_unshuffle()` method: Inserts mask tokens at correct positions
- Modified `forward()` to support both modes

#### 2. `shared_mask_token` (config.py)
- **True (default)**: Single mask token shared between teacher/student
- **False**: Separate learnable mask tokens for teacher and student decoders

**Implementation**:
- Added `_get_mask_token(for_decoder)` method to retrieve appropriate token
- Separate `teacher_mask_token` and `student_mask_token` when not shared

### Experiment Changes

**Modified Files**:
- `scripts/run_experiments.py`
- `scripts/visualize_all.py`

**Parameter Grid Updates**:
```python
DEFAULT_PARAM_GRID = {
    # ... existing parameters ...
    'mask_after_encoder': [False, True],
    'shared_mask_token': [True, False],
}
# Total combinations: 2*2*3*3*2*2*2*2*2 = 1152
```

**Dataset Size Changes**:
- `quick_length`: 200000 → 66000 (1/3 reduction)
- `full_length`: 440000 → 220000 (1/2 reduction)
- `full_epochs`: fixed at 2

**Stage 2 Selection Updates**:
- Added `mask_after_encoder` (top 5 per value)
- Added `shared_mask_token` (top 5 per value)

**Output Cleanup**:
- Removed "Train: X, Test: Y" from Stage 1/2 headers (values were outdated)

### Documentation Updates

**Modified Files**:
- `docs/ARCHITECTURE.md`: Added MAE Masking Architecture and Mask Token Configuration sections
- `docs/ABLATION_STUDIES.md`: Added sections 8 (Mask After Encoder) and 9 (Shared Mask Token)

---

## 2026-01-23 (Update 12.2): Complexity Visualization

### Summary

Added 3 new visualization functions to explain NormalDataComplexity features.

### Changes

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**New Visualizations**:
1. `plot_normal_complexity_features()` - Shows each of 6 complexity features individually
2. `plot_complexity_comparison()` - Simple vs Complex normal data side-by-side
3. `plot_complexity_vs_anomaly()` - Why complexity features don't resemble anomalies

**Output Files**:
- `normal_complexity_features.png` - 6-panel feature explanation
- `complexity_comparison.png` - Simple vs Complex comparison
- `complexity_vs_anomaly.png` - Complexity vs Anomaly discrimination

---

## 2026-01-23 (Update 12.1): Experiment Integration

### Summary

- Exported `NormalDataComplexity` from `mae_anomaly` package
- Updated `run_experiments.py` to use complexity features by default
- Added `--no-complexity` CLI flag to disable complexity features

### Changes

**Modified Files**:
- `mae_anomaly/__init__.py`: Export `NormalDataComplexity`
- `scripts/run_experiments.py`: Use complexity by default, add CLI flag

**Usage**:
```bash
# Default: with complexity (recommended)
python scripts/run_experiments.py

# Without complexity (simple patterns)
python scripts/run_experiments.py --no-complexity
```

---

## 2026-01-23 (Update 12): Normal Data Complexity Features

### Summary

Added 6 configurable complexity features to make normal data more realistic and challenging for anomaly detection models. All features are designed to NOT be confused with anomaly patterns.

### Changes

#### 1. NormalDataComplexity Configuration

**Modified Files**:
- `mae_anomaly/dataset_sliding.py`

**Added**:
- `NormalDataComplexity` dataclass with on/off switches for each feature
- All features enabled by default, individually toggleable

```python
@dataclass
class NormalDataComplexity:
    enable_complexity: bool = True
    enable_regime_switching: bool = True
    enable_multi_scale_periodicity: bool = True
    enable_heteroscedastic_noise: bool = True
    enable_varying_correlations: bool = True
    enable_drift: bool = True
    enable_normal_bumps: bool = True
    # ... detailed parameters for each
```

---

#### 2. Six Complexity Features Implemented

| Feature | Description | Transition Time |
|---------|-------------|-----------------|
| **Regime Switching** | Different operational states | 1500 timesteps |
| **Multi-Scale Periodicity** | 3 overlapping frequencies | Continuous |
| **Heteroscedastic Noise** | Load-dependent variance | Continuous |
| **Time-Varying Correlations** | Slowly changing correlations | Period 15000 ts |
| **Bounded Drift (O-U)** | Mean-reverting random walk | Continuous |
| **Normal Bumps** | Small, gradual load increases | Gaussian envelope |

---

#### 3. Safety Constraints

All complexity features enforce strict constraints to distinguish from anomalies:

| Constraint | Value | Reason |
|------------|-------|--------|
| Transition time | >= 1000 ts | Anomalies are 3-150 ts |
| Value range | [0.05, 0.70] | Anomalies push to 0.7-1.0 |
| Bump magnitude | max 0.10 | Spike adds 0.3-0.6 |
| Bump duration | 100-300 ts | Spike is 10-25 ts |

---

#### 4. Documentation Updated

**Modified Files**:
- `docs/DATASET.md`

**Added**:
- New section "Normal Data Complexity Features"
- Detailed documentation for each feature
- Configuration examples
- Safety constraints explanation

---

### Usage

```python
from mae_anomaly.dataset_sliding import NormalDataComplexity, SlidingWindowTimeSeriesGenerator

# Full complexity (default)
complexity = NormalDataComplexity()

# Simple mode
complexity = NormalDataComplexity(enable_complexity=False)

# Custom
complexity = NormalDataComplexity(
    enable_regime_switching=True,
    enable_normal_bumps=False,
)

generator = SlidingWindowTimeSeriesGenerator(
    total_length=440000,
    complexity=complexity,
    seed=42
)
```

---

## 2026-01-23 (Update 11): Visualization Quality Improvements

### Changes

#### 1. Removed Redundant anomaly_types Visualization

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- Removed `plot_anomaly_types()` from `generate_all()` - redundant with `plot_anomaly_generation_rules()`
- The `anomaly_generation_rules.png` provides more informative visualization using actual dataset samples

---

#### 2. Improved feature_examples Visualization

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- Now displays ALL 8 features (was hardcoded to 5)
- Uses actual `FEATURE_NAMES` for labels (CPU_Usage, Memory_Usage, etc.)
- Dynamic subplot layout based on feature count

---

#### 3. Improved sample_types Visualization with Diverse Sampling

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- Added `select_diverse()` function to randomly sample from shuffled data
- Prevents showing overlapping/similar samples due to stride=10
- Ensures visual diversity in sample type comparison

---

#### 4. Improved patchify_modes as Conceptual Flow Diagrams

**Modified Files**:
- `mae_anomaly/visualization/architecture_visualizer.py`

**Changes**:
- Complete rewrite of `plot_patchify_modes()`
- Now shows conceptual processing pipeline with boxes and arrows
- Three modes clearly differentiated:
  - **CNN-First**: Input → CNN → Patchify → Embed
  - **Patch-CNN**: Input → Patchify → CNN (per patch) → Embed
  - **Linear (MAE)**: Input → Patchify → Linear Projection
- Removed meaningless bar chart comparison

---

#### 5. Improved discrepancy_trend Visualization

**Modified Files**:
- `mae_anomaly/visualization/best_model_visualizer.py`

**Changes**:
- Added standard deviation bands (mean ± std shading)
- Added zoomed view of last patch region (masked region)
- Added box plots showing discrepancy distribution by sample type
- Added statistics text box with mean ± std values
- More informative for analyzing masked region behavior

---

#### 6. Fixed METRIC_COLUMNS in Stage2Visualizer

**Modified Files**:
- `mae_anomaly/visualization/stage2_visualizer.py`

**Changes**:
- Added missing metrics to `METRIC_COLUMNS`:
  - `disturbing_roc_auc`, `disturbing_f1`, `disturbing_precision`, `disturbing_recall`
  - `quick_roc_auc`, `quick_f1`, `quick_disturbing_roc_auc`
  - `roc_auc_improvement`, `selection_criterion`, `stage2_rank`
- Prevents metrics from being incorrectly treated as hyperparameters

---

### Benefits

1. **Cleaner visualizations**: Removed redundant plots, improved clarity
2. **More informative**: All features shown with proper names
3. **Better diversity**: Sample type visualization shows varied data
4. **Conceptual clarity**: Patchify modes now explain the processing pipeline
5. **Statistical rigor**: Discrepancy trend includes uncertainty bands
6. **Correct hyperparameter analysis**: Metrics no longer appear as hyperparameters in Stage 2 plots

---

## 2026-01-23 (Update 10): Dynamic Hyperparameter and Configuration Management

### Changes

#### 1. Dynamic param_keys in visualize_all.py

**Modified Files**:
- `scripts/visualize_all.py`

**Before**: Hardcoded list of hyperparameter keys
```python
param_keys = ['masking_ratio', 'masking_strategy', 'num_patches', ...]
```

**After**: Dynamically extracted from experiment metadata or results
```python
if exp_data['metadata'] and 'param_grid' in exp_data['metadata']:
    param_keys = list(exp_data['metadata']['param_grid'].keys())
else:
    # Fallback: extract from results DataFrame
    param_keys = [c for c in columns if c not in metric_cols]
```

---

#### 2. Dynamic Hyperparameter Lists in stage2_visualizer.py

**Modified Files**:
- `mae_anomaly/visualization/stage2_visualizer.py`

**Changes**:
- Added `METRIC_COLUMNS` class constant for known metric columns
- Added `_get_hyperparam_columns()` helper method
- `plot_all_hyperparameters()`: Now uses dynamic hyperparameter detection
- `plot_hyperparameter_interactions()`: Dynamically generates interaction pairs
- `plot_best_config_summary()`: Uses dynamic hyperparams with fallback descriptions

---

#### 3. Dynamic Categorical Parameters in experiment_visualizer.py

**Modified Files**:
- `mae_anomaly/visualization/experiment_visualizer.py`

**Changes**:
- Added `_get_categorical_params()` helper method
- `plot_summary_dashboard()`: Uses dynamically detected categorical params
- `generate_all()`: Uses dynamic categorical params for comparisons

---

#### 4. Robust get_anomaly_type_info in base.py

**Modified Files**:
- `mae_anomaly/visualization/base.py`

**Changes**:
- `get_anomaly_type_info()` now handles unknown anomaly types gracefully
- Auto-generates descriptions for new anomaly types not in known_info dict
- Always includes all types from `ANOMALY_TYPE_NAMES`

---

### Benefits

1. **No manual updates needed**: Adding new hyperparameters to `DEFAULT_PARAM_GRID` automatically includes them in visualizations
2. **No sync issues**: New anomaly types are automatically handled with auto-generated descriptions
3. **Reduced maintenance**: Less hardcoded values = fewer places to update when configuration changes
4. **Better error handling**: Fallback mechanisms prevent crashes from missing data

---

## 2026-01-23 (Update 9): Visualization Module Modularization

### Changes

#### 1. Modular Visualization Package

**New Directory Structure**:
```
mae_anomaly/
└── visualization/
    ├── __init__.py              # Module exports
    ├── base.py                  # Common utilities, colors, data loading
    ├── data_visualizer.py       # DataVisualizer class
    ├── architecture_visualizer.py  # ArchitectureVisualizer class
    ├── experiment_visualizer.py # ExperimentVisualizer (Stage 1)
    ├── stage2_visualizer.py     # Stage2Visualizer class
    ├── best_model_visualizer.py # BestModelVisualizer class
    └── training_visualizer.py   # TrainingProgressVisualizer class
```

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py): Reduced from ~4900 lines to ~166 lines
- [mae_anomaly/visualization/](../mae_anomaly/visualization/): New modular package

**Benefits**:
- Cleaner, more maintainable code structure
- Each visualizer class in its own file
- Common utilities centralized in `base.py`
- Easy to extend with new visualizers

---

#### 2. Dynamic Color Management

**Modified Files**:
- `mae_anomaly/visualization/base.py`
- `mae_anomaly/visualization/best_model_visualizer.py`
- `mae_anomaly/visualization/training_visualizer.py`

**Changes**:
- Created `get_anomaly_colors()` function that dynamically generates colors for all anomaly types
- Created `SAMPLE_TYPE_COLORS` and `SAMPLE_TYPE_NAMES` constants
- Replaced all hardcoded color dictionaries with dynamic functions
- Colors now automatically adapt when anomaly types are added/removed

**Before** (hardcoded):
```python
colors = {
    'normal': '#3498DB',
    'spike': '#E74C3C',
    # ... manually maintained
}
```

**After** (dynamic):
```python
from mae_anomaly.visualization import get_anomaly_colors
colors = get_anomaly_colors()  # Automatically includes all anomaly types
```

---

#### 3. Dynamic plot_anomaly_generation_rules

**Modified Files**:
- `mae_anomaly/visualization/data_visualizer.py`

**Changes**:
- `plot_anomaly_generation_rules()` now dynamically generates visualizations based on `ANOMALY_TYPE_NAMES`
- Uses actual dataset examples instead of synthetic simulation
- Automatically adapts grid size based on number of anomaly types
- Gets anomaly info (length_range, characteristics) from `ANOMALY_TYPE_CONFIGS`

---

#### 4. Usage Update

**New Import Pattern**:
```python
# Old (from script)
from scripts.visualize_all import DataVisualizer, load_best_model

# New (from module)
from mae_anomaly.visualization import (
    DataVisualizer,
    ArchitectureVisualizer,
    ExperimentVisualizer,
    Stage2Visualizer,
    BestModelVisualizer,
    TrainingProgressVisualizer,
    setup_style,
    load_best_model,
    get_anomaly_colors,
)
```

**Running visualizations** (unchanged):
```bash
python scripts/visualize_all.py  # Still works the same way
```

---

## 2026-01-23 (Update 8): Point Spike Duration Change and Visualization Fixes

### Changes

#### 1. Point Spike Duration Change

**Modified Files**:
- [mae_anomaly/dataset_sliding.py](../mae_anomaly/dataset_sliding.py)
- [docs/DATASET.md](DATASET.md)

**Changes**:
- Point spike duration: (1, 3) → **(3, 5)** timesteps
- Still the shortest anomaly type, but more detectable

```python
# Before
7: {'length_range': (1, 3), 'interval_mean': 4000}

# After
7: {'length_range': (3, 5), 'interval_mean': 4000}
```

---

#### 2. Visualization Color Map Update

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- Updated `plot_loss_by_anomaly_type()` colors: Added `point_spike` color
- Updated `plot_loss_scatter_by_anomaly_type()` colors: Fixed outdated anomaly type names (`noise`, `drift` → actual types)

**Before** (incorrect):
```python
colors = {
    'normal': '#3498DB',
    'spike': '#E74C3C',
    'memory_leak': '#F39C12',
    'noise': '#9B59B6',        # ← Wrong
    'drift': '#1ABC9C',         # ← Wrong
    'network_congestion': '#E67E22'
}
```

**After** (correct):
```python
colors = {
    'normal': '#3498DB',
    'spike': '#E74C3C',
    'memory_leak': '#F39C12',
    'cpu_saturation': '#9B59B6',
    'network_congestion': '#E67E22',
    'cascading_failure': '#1ABC9C',
    'resource_contention': '#16A085',
    'point_spike': '#E91E63',
}
```

---

#### 3. Anomaly-Type Performance Comparison Verification

**Existing Functions (Best Model)**:
- `plot_loss_by_anomaly_type()`: Loss distribution per anomaly type ✓
- `plot_performance_by_anomaly_type()`: Detection rate & mean score per type ✓
- `plot_loss_scatter_by_anomaly_type()`: Loss scatter per type ✓
- `plot_anomaly_type_case_studies()`: TP/FN examples per type ✓

**Existing Functions (Training Progress)**:
- `plot_anomaly_type_learning()`: Detection rate over epochs per type ✓

**Stage 1/2**: Designed for hyperparameter comparison, not anomaly-type analysis (by design)

---

## 2026-01-23 (Update 7): Point Spike Anomaly and Dataset Statistics

### Changes

#### 1. New Anomaly Type: Point Spike

**Modified Files**:
- [mae_anomaly/dataset_sliding.py](../mae_anomaly/dataset_sliding.py)
- [docs/DATASET.md](DATASET.md)

**New Anomaly Type**:
- **point_spike** (type 7): True point anomaly lasting only 3-5 timesteps
- **Unique characteristic**: 2+ random features spike simultaneously
- Makes threshold-based detection on individual features less effective

```python
# Point spike configuration
7: {'length_range': (3, 5), 'interval_mean': 4000}

# Injection logic
def _inject_point_spike(self, signals, start, end):
    # Select 2+ random features
    num_features_to_spike = np.random.randint(2, self.num_features + 1)
    features_to_spike = np.random.choice(self.num_features, num_features_to_spike, replace=False)
    # Apply spike magnitude +0.3 to +0.6 to each selected feature
```

---

#### 2. Dataset Statistics Output

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**New Feature**: When running experiments, dataset statistics are now printed:

```
[Quick Dataset Statistics - Test Set (Raw)]
Sample Types:
  - Pure Normal:       XXXX (XX.X%)
  - Disturbing Normal: XXX (XX.X%)
  - Anomaly:           XXX (XX.X%)
  - Total:             XXXX

Anomaly Types (region count):
  - spike: XX
  - memory_leak: XX
  - cpu_saturation: XX
  - network_congestion: XX
  - cascading_failure: XX
  - resource_contention: XX
  - point_spike: XX
```

---

#### 3. Visualization Code Update

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- `plot_anomaly_type_case_studies()`: Now dynamically uses `ANOMALY_TYPE_NAMES` instead of hardcoded list
- `plot_anomaly_type_learning()`: Now dynamically uses `ANOMALY_TYPE_NAMES` instead of hardcoded list
- Handles any number of anomaly types automatically

---

## 2026-01-23 (Update 6): Reduce Full Search Epochs

### Changes

- Changed `full_epochs` default from **3 to 2** for faster experimentation
- Updated files:
  - [scripts/run_experiments.py](../scripts/run_experiments.py): Function parameter and argparse default
  - [README.md](../README.md): Experiment settings table
  - [docs/ABLATION_STUDIES.md](ABLATION_STUDIES.md): Stage 2 description
  - [docs/VISUALIZATIONS.md](VISUALIZATIONS.md): Settings table

---

## 2026-01-23 (Update 5): Threshold Fix and Hypothesis Verification

### Changes

#### 1. Disturbing Normal Evaluation Fix

**Modified Files**:
- [mae_anomaly/evaluator.py](../mae_anomaly/evaluator.py)

**Problem**:
- Disturbing normal evaluation was using a **separate threshold** calculated only from pure_normal and disturbing_normal samples
- This was incorrect - should use the **global threshold** from the entire dataset

**Fix**:
- Now uses the global optimal threshold (calculated from all samples) for disturbing normal evaluation
- ROC-AUC is threshold-free, so no change needed there
- Precision/Recall/F1 now use the same threshold as overall evaluation

**Before** (incorrect):
```python
d_fpr, d_tpr, d_thresholds = roc_curve(disturbing_labels, disturbing_scores)
d_optimal_idx = np.argmax(d_tpr - d_fpr)
d_threshold = d_thresholds[d_optimal_idx]  # Separate threshold!
d_predictions = (disturbing_scores > d_threshold).astype(int)
```

**After** (correct):
```python
# Use GLOBAL threshold (from entire dataset)
d_predictions = (disturbing_scores > threshold).astype(int)
```

---

#### 2. Hypothesis Verification Visualization

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)
- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md)

**New Visualization**: `hypothesis_verification.png`

Verifies 4 hypotheses about why disturbing normal might outperform pure normal:

1. **H1: Anomaly Hint** - Does anomaly in window increase score?
   - Scatter plot of anomaly ratio vs total score

2. **H2: Transition Effect** - Does recent anomaly affect last patch?
   - Scatter plot of distance from anomaly to last patch vs score

3. **H3: Variance Analysis** - Does pure normal have higher variance?
   - Violin plot comparing score distributions

4. **H4: Classification Rates** - How do FP/TP rates compare with global threshold?
   - Bar chart of classification rates

---

#### 3. Quick Search Epoch Reduction

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)
- [README.md](../README.md)
- [docs/ABLATION_STUDIES.md](../docs/ABLATION_STUDIES.md)
- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md)

**Changes**:
- Stage 1 (Quick Search) epochs: 2 → **1**

**Rationale**:
- Single epoch sufficient for quick screening of 432 combinations
- Significantly reduces experiment time while maintaining ranking quality

**Updated Settings**:
| Stage | Epochs |
|-------|--------|
| Stage 1 (Quick) | 1 |
| Stage 2 (Full) | 3 |

---

## 2026-01-23 (Update 4): Estimated Time Display

### Changes

#### Time Estimation Feature

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Added time estimation based on first model training time
- Displays estimated time for Quick Search, Full Search, and Total
- Considers dataset size, epochs, and model count differences

**Output Format**:
```
>>> Estimated Time (based on 1st model: X.Xs) <<<
  Quick Search: XX분 (432 models × X.Xs)
  Full Search:  XX분 (~60 models × X.Xs)
  Total:        XX분
  (Quick remaining: XX분)
```

**Calculation**:
- Quick Search: `first_model_time × n_models`
- Full Search: `first_model_time × (full_train/quick_train) × (full_epochs/quick_epochs) × n_stage2_models`
  - `full_train/quick_train = 22,000/6,000 ≈ 3.67`
  - `full_epochs/quick_epochs = 3/2 = 1.5`

---

## 2026-01-23 (Update 3): Stage 2 Selection Reduction and Epoch Fine-tuning

### Changes

#### 1. Quick Search Epoch Reduction

**Changes**:
- Stage 1 epochs: 5 → **3**

**Rationale**:
- Further speed up quick search screening
- 3 epochs sufficient to identify promising configurations

---

#### 2. Stage 2 Selection Criteria Reduction

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Per-parameter top models: 10 → **5**
- Overall ROC-AUC top models: 30 → **10**
- Disturbing ROC-AUC top models: 20 → **5**
- Expected Stage 2 models: ~150 → **~50-70** (after deduplication)

**Rationale**:
- Faster full training while maintaining diverse coverage
- Still covers all parameter values with representative models

---

#### 3. Stage 2 Model Count Display

**Changes**:
- Added print statement showing Stage 2 model count during experiment execution
- Format: `>>> Stage 2 will train {N} models (from {M} Stage 1 combinations) <<<`

---

## 2026-01-23 (Update 2): Two-Stage Dataset and Epoch Configuration

### Changes

#### 1. Separate Datasets for Quick/Full Search

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Stage 1 (Quick Search): 200,000 timesteps, train_ratio=0.3 → ~6,000 train, ~14,000 test
- Stage 2 (Full Search): 2,200,000 timesteps, train_ratio=0.5 → ~110,000 train, ~110,000 test
- Test set always uses target_counts 1200:300:500 (total 2,000)

**Rationale**:
- Quick search needs fast iteration (small train set)
- Test set composition should be consistent across stages for fair comparison

---

#### 2. Epoch Count Reduction

**Changes**:
- Stage 1 epochs: 15 → **5**
- Stage 2 epochs: 100 → **30**

**Rationale**:
- Faster experimentation while maintaining reasonable training quality
- Quick search only needs to identify promising configurations

---

## 2026-01-23: Dataset Migration and Hyperparameter Grid Cleanup

### Major Changes

#### 1. Dataset Migration to SlidingWindowDataset

**Modified Files**:
- [mae_anomaly/dataset.py](../mae_anomaly/dataset.py) → Deprecated
- [mae_anomaly/dataset_sliding.py](../mae_anomaly/dataset_sliding.py) → Primary dataset
- [scripts/run_experiments.py](../scripts/run_experiments.py)
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- Replaced `MultivariateTimeSeriesDataset` with `SlidingWindowTimeSeriesGenerator` and `SlidingWindowDataset`
- New dataset features:
  - Continuous sliding window extraction from long time series
  - 8 correlated server metrics (CPU, Memory, DiskIO, Network, ResponseTime, ThreadCount, ErrorRate, QueueLength)
  - 6 realistic anomaly types: spike, memory_leak, cpu_saturation, network_congestion, cascading_failure, resource_contention
  - Three sample types: pure_normal, disturbing_normal, anomaly
  - Train/test split by time (no data leakage)

---

#### 2. Fixed Hyperparameters (margin, lambda_disc)

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)
- [scripts/visualize_all.py](../scripts/visualize_all.py)
- [mae_anomaly/config.py](../mae_anomaly/config.py)

**Changes**:
- `margin` and `lambda_disc` are now fixed at 0.5 (not in hyperparameter grid)
- Reduced hyperparameter search space from 2592 to 288 combinations
- Grid now includes: `masking_ratio`, `masking_strategy`, `num_patches`, `margin_type`, `force_mask_anomaly`, `patch_level_loss`, `patchify_mode`

**Rationale**:
- Preliminary experiments showed margin=0.5 and lambda_disc=0.5 perform well across configurations
- Reducing search space allows more thorough exploration of other hyperparameters

---

#### 3. Stage 2 Selection Criteria Update

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- New selection criteria for Stage 2 (150 diverse candidates):
  - Per-parameter top 10 (e.g., best for each masking_ratio value)
  - Overall top 30 by ROC-AUC
  - Top 20 by disturbing normal ROC-AUC
- Added `masking_strategy` to selection criteria

---

#### 4. num_features Updated (5 → 8)

**Modified Files**:
- [mae_anomaly/config.py](../mae_anomaly/config.py)
- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
- [docs/DATASET.md](../docs/DATASET.md)

**Changes**:
- Default `num_features` changed from 5 to 8
- All documentation diagrams updated to reflect (batch, 100, 8) dimensions

---

#### 5. Visualization Bug Fixes

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Fixes**:
- Updated `param_keys` to remove `margin` and `lambda_disc`
- Added `ANOMALY_TYPE_NAMES` import for visualization
- Fixed `plot_loss_by_anomaly_type` subplot grid (2x3 → dynamic for 7 anomaly types)
- Updated multiple places where margin/lambda_disc were referenced

---

### Documentation Updates

- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md): Updated num_features (5→8), all dimension examples
- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md): Updated param_keys, CSV columns, removed margin/lambda_disc
- [docs/DATASET.md](../docs/DATASET.md): Complete documentation for SlidingWindowDataset
- [docs/CHANGELOG.md](../docs/CHANGELOG.md): This entry

---

## 2026-01-22 (Update 3): Qualitative Case Studies and Late Bloomer Fix

### Major Changes

#### 1. Late Bloomer Algorithm Fix

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Issue Found**:
- Late bloomer analysis used final epoch's threshold for all epochs
- At epoch 0, the model hasn't learned, so all scores are similar
- Using the final threshold at epoch 0 produces incorrect predictions

**Fixes**:
- Implemented per-epoch optimal threshold calculation
- Late bloomers now correctly identified as samples that changed from incorrect to correct classification
- Added two categories:
  - **Late Bloomer Anomalies (FN→TP)**: Missed at start, detected at end
  - **Late Bloomer Normals (FP→TN)**: False alarm at start, correct at end

---

#### 2. Reconstruction Evolution Enhancement

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Changes**:
- Added Student reconstruction alongside Teacher (was Teacher-only)
- Added discrepancy visualization (|Teacher - Student|)
- Shows both reconstruction and discrepancy evolution over epochs
- Key insight: Discrepancy should increase in masked anomaly regions as training progresses

---

#### 3. Qualitative Case Study Visualizations

**New Files** (in `best_model/`):
- `case_study_gallery.png`: Representative TP/TN/FP/FN examples with detailed analysis
- `anomaly_type_case_studies.png`: Per-anomaly-type TP vs FN comparison
- `feature_contribution_analysis.png`: Which features drive anomaly detection
- `hardest_samples.png`: Analysis of hardest-to-detect samples (lowest margin FN/FP)

**New Files** (in `training_progress/`):
- `late_bloomer_case_studies.png`: Detailed time series evolution for late bloomers

**New Methods**:
- `BestModelVisualizer.plot_case_study_gallery()`: Median examples for each outcome
- `BestModelVisualizer.plot_anomaly_type_case_studies()`: Per-type TP/FN comparison
- `BestModelVisualizer.plot_feature_contribution_analysis()`: Feature importance ranking
- `BestModelVisualizer.plot_hardest_samples()`: Hardest FN and FP analysis
- `TrainingProgressVisualizer.plot_late_bloomer_case_studies()`: Detailed late bloomer evolution

---

### Documentation Updates

- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md): Added new visualizations and updated descriptions
- [docs/CHANGELOG.md](../docs/CHANGELOG.md): This changelog entry

---

## 2026-01-22 (Update 2): Visualization Enhancements and Consistency Fixes

### Major Changes

#### 1. Visualization Data Consistency Fix

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py)

**Issue Found**:
- `visualize_all.py` used different evaluation settings than `run_experiments.py`:
  - `anomaly_ratio=0.3` instead of `config.test_anomaly_ratio=0.25`
  - Random masking instead of fixed last-patch masking
  - MAE (absolute error) instead of MSE (squared error)

**Fixes**:
- Changed `anomaly_ratio` to use `config.test_anomaly_ratio` (0.25)
- Changed `collect_predictions()` and `collect_detailed_data()` to use same evaluation as `evaluator.py`:
  - Fixed mask: `mask[:, -config.mask_last_n:] = 0`
  - Forward with `masking_ratio=0.0` and explicit mask
  - MSE computation: `((output - input) ** 2).mean(dim=2)`

---

#### 2. New Data Visualizations

**New Files**:
- `data/anomaly_generation_rules.png`: Detailed rules for each anomaly type
- `data/feature_correlations.png`: Feature correlation matrix and generation rules
- `data/experiment_settings.png`: Experiment settings summary (Stage 1/2)

**Changes**:
- Added `plot_anomaly_generation_rules()`: Shows how each anomaly type is generated
- Added `plot_feature_correlations()`: Shows inter-feature correlations
- Added `plot_experiment_settings()`: Summarizes experiment settings for reproducibility

---

#### 3. Stage 2 Per-Hyperparameter Visualizations

**New Files** (in `stage2/`):
- `hyperparam_masking_ratio.png`
- `hyperparam_num_patches.png`
- `hyperparam_margin.png`
- `hyperparam_lambda_disc.png`
- `hyperparam_margin_type.png`
- `hyperparam_force_mask_anomaly.png`
- `hyperparam_patch_level_loss.png`
- `hyperparam_patchify_mode.png`
- `hyperparameter_interactions.png`
- `best_config_summary.png`

**Changes**:
- Added `plot_hyperparameter_impact()`: Per-hyperparameter detailed analysis
- Added `plot_all_hyperparameters()`: Generate all per-hyperparameter plots
- Added `plot_hyperparameter_interactions()`: Interaction heatmaps
- Added `plot_best_config_summary()`: Best config with Korean descriptions

---

#### 4. Best Model Analysis Improvements

**New Files**:
- `best_model/pure_vs_disturbing_normal.png`: Pure Normal vs Disturbing Normal comparison
- `best_model/discrepancy_trend.png`: Discrepancy trend analysis across time steps

**Changes**:
- Added `plot_pure_vs_disturbing_normal()`: Detailed comparison of sample types
- Added `plot_discrepancy_trend()`: Time-step level discrepancy analysis

---

### Documentation Updates

- [docs/VISUALIZATIONS.md](../docs/VISUALIZATIONS.md): Complete rewrite with all new visualizations

---

## 2026-01-22: Project Cleanup and Patchify Mode

### Major Changes

#### 1. Patchify Mode Feature

**Modified Files**:
- [mae_anomaly/model.py](../mae_anomaly/model.py)
- [mae_anomaly/config.py](../mae_anomaly/config.py)

**Changes**:
- Added `patchify_mode` configuration option with 2 modes:
  - `linear`: Direct patchify + linear projection (MAE original style)
  - `patch_cnn`: Patchify first, then CNN per patch (no cross-patch leakage)
- Updated model to support both patchify modes

**Benefits**:
- Flexibility to test different patchification strategies
- `patch_cnn` mode prevents information leakage across patches
- Better control over local feature extraction

---

#### 2. Visualization Refactoring

**Modified Files**:
- [scripts/visualize_all.py](../scripts/visualize_all.py) (NEW)
- [scripts/run_experiments.py](../scripts/run_experiments.py) (refactored)

**Changes**:
- Moved all visualization code from `run_experiments.py` to dedicated `visualize_all.py`
- Created 5 visualization classes:
  - `DataVisualizer`: Data distribution and sample visualization
  - `ArchitectureVisualizer`: Model architecture diagrams
  - `ExperimentVisualizer`: Stage 1 (Quick Search) results
  - `Stage2Visualizer`: Stage 2 (Full Training) results
  - `BestModelVisualizer`: Best model analysis
- `run_experiments.py` now only handles training and saves results to CSV
- Fixed shape mismatch bugs when data has fewer than 10 rows

---

#### 3. Project Cleanup

**Deleted Files**:
- `tests/integration/` (obsolete test files using old module)
- `REFACTORING_COMPLETE.md`, `REFACTORING_PLAN.md`
- `docs/bugfixes/`, `docs/implementation/`, `docs/analysis/`

**Updated Files**:
- [README.md](../README.md) - Complete rewrite for current structure
- [examples/basic_usage.py](../examples/basic_usage.py) - Updated imports and examples
- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - Added patchify_mode documentation
- [docs/ABLATION_STUDIES.md](../docs/ABLATION_STUDIES.md) - Added patchify_mode experiments

---

### Documentation Updates

- README.md now reflects current project structure
- Added patchify mode examples in basic_usage.py
- Architecture documentation includes all 3 patchify modes
- Ablation studies documentation includes patchify_mode experiments

---

## 2026-01-14: Architecture and Training Updates

### Major Changes

#### 1. Architecture: Transformer → 1D-CNN + Transformer Hybrid

**Modified Files**:
- [mae_anomaly/model.py](../mae_anomaly/model.py)

**Changes**:
- Added 2-layer 1D-CNN before Transformer:
  - Conv1: num_features (5) → d_model//2 (32), kernel=3
  - Conv2: d_model//2 (32) → d_model (64), kernel=3
  - BatchNorm + ReLU after each layer
- Updated patch embedding to work with CNN features:
  - New method: `patchify_cnn()` for CNN output
  - Processes (batch, d_model, seq_length) → (batch, num_patches, d_model*patch_size)
- Updated forward pass:
  - Input → CNN → Patchify → Transformer
  - CNN adds ~6,912 parameters
  - Total parameters: ~513K (was ~505K)

**Benefits**:
- Better local feature extraction
- Combines CNN (local) + Transformer (global) strengths
- Improved representation learning

---

#### 2. Best Model Selection: Match Evaluation Criterion

**Modified Files**:
- [scripts/run_experiments.py](../scripts/run_experiments.py)

**Changes**:
- Best model selection now matches evaluation metric:
  - **Baseline**: Uses total loss (reconstruction + discrepancy)
  - **TeacherOnly**: Uses teacher reconstruction loss
  - **StudentOnly**: Uses student reconstruction loss
- Model selection based on ROC-AUC during grid search

**Rationale**:
- Previous: All experiments used reconstruction loss for model selection
- Issue: Baseline evaluation uses discrepancy, but model selected on reconstruction
- Fix: Model selection criterion now matches what we optimize for in each ablation

---

### Documentation Updates

#### Created Files:
1. [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
   - Complete architecture documentation
   - Component-by-component breakdown
   - Parameter counts and pipeline diagram
   - Design rationale and comparisons

#### Updated Files:
1. [docs/ABLATION_STUDIES.md](../docs/ABLATION_STUDIES.md)
   - Added architecture overview section
   - Updated best model selection notes
   - Clarified evaluation criteria for each ablation

---

---

## Previous Updates (2026-01-14)

### Data and Ablation Updates

1. **Data Size Increase** (5x):
   - Train: 1,000 → 5,000 samples
   - Test: 300 → 1,500 samples

2. **Best Model Checkpointing**:
   - Track training loss during epochs
   - Save model at lowest loss epoch
   - Restore best model after training

3. **Masking Strategy Ablation**:
   - Added patch masking (same-time across features)
   - Added feature-wise masking (independent per feature)
   - Tests importance of cross-feature temporal coherence

4. **Removed Redundant Ablations**:
   - Removed NoDiscrepancy (redundant with TeacherOnly)
   - Removed NoMasking (replaced with more informative experiments)

5. **Cleanup**:
   - Deleted old experiment results
   - Removed unused folders
   - Regenerated visualizations

---

## File Structure

```
mae_anomaly/
├── model.py            [MODIFIED] - Added 1D-CNN layers, patchify modes
├── config.py           [MODIFIED] - Added patchify_mode, margin_type options
├── loss.py             - Self-distillation loss
├── trainer.py          - Training loop
├── evaluator.py        - Evaluation metrics
└── dataset.py          - Synthetic dataset generation

scripts/
├── run_experiments.py  - Two-stage grid search experiment runner
└── visualize_all.py    - Comprehensive visualization generator

docs/
├── ARCHITECTURE.md     - Architecture documentation
├── ABLATION_STUDIES.md - Ablation study documentation
├── VISUALIZATIONS.md   - Visualization guide
└── CHANGELOG.md        - This file
```

---

## Summary

**Total Changes**:
1. ✅ Transformer → 1D-CNN + Transformer hybrid architecture
2. ✅ Best model selection matches evaluation criterion
3. ✅ Comprehensive architecture documentation
4. ✅ Testing and verification scripts

**Status**: All changes implemented, tested, and documented.

**Next Steps**: Run full experiments with updated architecture and training logic.
