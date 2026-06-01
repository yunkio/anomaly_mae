# Claude Code Instructions

## Environment
**Always use `dc_vis` conda environment**: `conda activate dc_vis`

## Documentation Rules (CRITICAL)

**After ANY code change**: Update docs → Add CHANGELOG entry → `git commit && git push`

| Modified File | Update These Docs |
|--------------|-------------------|
| `config.py` | ARCHITECTURE.md, DATASET.md, ABLATION_STUDIES.md |
| `model.py`, `loss.py` | ARCHITECTURE.md |
| `dataset_sliding.py` | DATASET.md |
| `evaluator.py` | ARCHITECTURE.md, ABLATION_STUDIES.md |
| `run_base_experiments.py`, `run_ablation.py` | ABLATION_STUDIES.md, ABLATION_EXPERIMENTS.md, VISUALIZATIONS.md |
| `visualize_all.py`, `visualization/*.py` | VISUALIZATIONS.md, CHANGELOG.md |

**Commit format**: `Fix:`, `Feat:`, `Refactor:`, `Docs:` + description

## Key Commands
```bash
python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py  # Ablation study
python scripts/visualize_all.py --experiment-dir ...  # Generate visualizations
python scripts/run_base_experiments.py                 # Run base experiments (5 base + 28 SMD + 6 Exathlon = 39 datasets: simulation, SWaT, WaDi A1/A2, PSM, SMD ×28, Exathlon ×6)
```

## Quick Reference

**Architecture**: Patchify (`linear`/`patch_cnn`), Masking (`patch`), Margin (`hinge`/`softplus`/`dynamic`)

**Dataset**: 275K timesteps, 8 features, Window 500, Test stride=1, 9 anomaly types (6 value + 3 pattern)

**Code Style**: Config → dataclass with type hints, Ablation flags → `use_teacher`, `use_student`, etc.
