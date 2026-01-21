# Claude Code Instructions

## Environment (CRITICAL)

**Always use the `dc_vis` conda environment for ALL Python operations.**

```bash
conda activate dc_vis
```

Before running ANY Python script, ensure the environment is activated:
- `python scripts/run_experiments.py`
- `python scripts/visualize_all.py`
- `python examples/basic_usage.py`

---

## Documentation Maintenance Rules

**When modifying code, ALWAYS update the corresponding documentation.**

### Documentation Files and Their Roles

| File | Purpose | Update When... |
|------|---------|----------------|
| `README.md` | Project overview, quick start, installation | Changing project structure, dependencies, or main usage |
| `docs/ARCHITECTURE.md` | Model architecture details, pipeline, components | Modifying model.py, adding/changing model components |
| `docs/ABLATION_STUDIES.md` | Experiment configurations, parameter grid, ablation setup | Changing Config parameters, experiment grid, or evaluation logic |
| `docs/VISUALIZATIONS.md` | Visualization outputs, Stage 2 criteria, CSV formats | Modifying visualize_all.py or run_experiments.py output |
| `docs/CHANGELOG.md` | Version history, major changes | Any significant code changes |

### Specific Update Guidelines

#### When modifying `mae_anomaly/config.py`:
- Update `docs/ARCHITECTURE.md` (Default Configuration table)
- Update `docs/ABLATION_STUDIES.md` (Parameter Grid section)
- Update `docs/VISUALIZATIONS.md` (CSV Output Format)

#### When modifying `mae_anomaly/model.py`:
- Update `docs/ARCHITECTURE.md` (Architecture Components, Pipeline)

#### When modifying `mae_anomaly/loss.py`:
- Update `docs/ARCHITECTURE.md` (Self-Distillation Mechanism, Loss Configuration)

#### When modifying `scripts/run_experiments.py`:
- Update `docs/ABLATION_STUDIES.md` (Code Implementation, Grid Search Parameters)
- Update `docs/VISUALIZATIONS.md` (Stage 2 Selection Criteria)

#### When modifying `scripts/visualize_all.py`:
- Update `docs/VISUALIZATIONS.md` (Visualization Categories, Classes)

---

## Project Structure

```
mae_anomaly/           # Core package
├── config.py          # Configuration dataclass
├── model.py           # MAE model architecture
├── dataset.py         # Synthetic dataset generation
├── loss.py            # Self-distillation loss
├── trainer.py         # Training loop
└── evaluator.py       # Evaluation metrics

scripts/               # Execution scripts
├── run_experiments.py # Two-stage grid search
└── visualize_all.py   # Result visualization

docs/                  # Documentation
├── ARCHITECTURE.md    # Model architecture
├── ABLATION_STUDIES.md # Experiment setup
├── VISUALIZATIONS.md  # Visualization guide
└── CHANGELOG.md       # Change history

examples/              # Usage examples
└── basic_usage.py
```

---

## Key Commands

```bash
# Run experiments (two-stage grid search)
conda activate dc_vis && python scripts/run_experiments.py

# Generate visualizations from results
conda activate dc_vis && python scripts/visualize_all.py --experiment-dir results/experiments/YYYYMMDD_HHMMSS

# Run examples
conda activate dc_vis && python examples/basic_usage.py
```

---

## Architecture Quick Reference

- **Patchify Modes**: `linear` (default), `cnn_first`, `patch_cnn`
- **Masking Strategies**: `patch` (default), `feature_wise`
- **Self-Distillation**: Teacher (4 layers) vs Student (1 layer)
- **Margin Types**: `hinge`, `softplus`, `dynamic`
- **Data**: 100 timesteps, 5 features, 25 patches

---

## Code Style

- Config changes: Add to `Config` dataclass with type hints
- New parameters: Document in config.py comments
- Ablation flags: `use_teacher`, `use_student`, `use_discrepancy_loss`, etc.
