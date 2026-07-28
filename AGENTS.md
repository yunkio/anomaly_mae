# Codex Project Instructions

## Python Environment

- Always use the `dc_vis` conda environment for Python work in this repository.
- Do not run Python scripts, tests, linters, formatters, package commands, or import-based inspection with the system Python.
- Prefer `conda run -n dc_vis ...` for one-off commands, for example:
  - `conda run -n dc_vis python scripts/example.py`
  - `conda run -n dc_vis pytest`
- For an interactive shell session, activate the environment first:
  - `conda activate dc_vis`
- When inspecting or modifying Python scripts, assume the runtime, dependencies, and import behavior are those of `dc_vis`; any validation command that executes Python code must use that environment.

## Documentation

- After a code change, update the documentation mapped below and add an entry to `docs/CHANGELOG.md` when relevant.
- Do not commit or push unless the user's request explicitly includes publishing the change.
- When a commit is requested, use one of these prefixes: `Fix:`, `Feat:`, `Refactor:`, or `Docs:`.

| Modified file | Update these docs |
|---|---|
| `config.py` | `docs/ARCHITECTURE.md`, `docs/DATASET.md`, `docs/ABLATION_STUDIES.md` |
| `model.py`, `loss.py` | `docs/ARCHITECTURE.md` |
| `dataset_sliding.py` | `docs/DATASET.md` |
| `evaluator.py` | `docs/ARCHITECTURE.md`, `docs/ABLATION_STUDIES.md` |
| `run_base_experiments.py`, `run_ablation.py` | `docs/ABLATION_STUDIES.md`, `docs/VISUALIZATIONS.md` |
| `visualize_all.py`, `visualization/*.py` | `docs/VISUALIZATIONS.md`, `docs/CHANGELOG.md` |

## Key Commands

Run these through the required environment:

```bash
conda run -n dc_vis python scripts/ablation/run_ablation.py --config configs/20260127_052220_phase1.py
conda run -n dc_vis python scripts/visualize_all.py --experiment-dir ...
conda run -n dc_vis python scripts/run_base_experiments.py
```

The base experiment command covers 39 datasets: 5 base datasets, 28 SMD datasets, and 6 Exathlon datasets.

## Project Reference

- Architecture: Patchify (`linear` or `patch_cnn`), masking (`patch`), margin (`hinge`, `softplus`, or `dynamic`).
- Dataset: 275K timesteps, 8 features, window 500, test stride 1, and 9 anomaly types (6 value and 3 pattern).
- Code style: represent configuration as typed dataclasses; use explicit ablation flags such as `use_teacher` and `use_student`.
