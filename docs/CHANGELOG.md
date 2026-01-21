# Changelog

## 2026-01-22: Project Cleanup and Patchify Mode

### Major Changes

#### 1. Patchify Mode Feature

**Modified Files**:
- [mae_anomaly/model.py](../mae_anomaly/model.py)
- [mae_anomaly/config.py](../mae_anomaly/config.py)

**Changes**:
- Added `patchify_mode` configuration option with 3 modes:
  - `linear`: Direct patchify + linear projection (MAE original style)
  - `cnn_first`: CNN on full sequence, then patchify
  - `patch_cnn`: Patchify first, then CNN per patch (no cross-patch leakage)
- Updated model to support all three patchify modes

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
