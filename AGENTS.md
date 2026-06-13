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
