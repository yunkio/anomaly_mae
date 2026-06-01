#!/usr/bin/env python
"""Delete baseline model results (keep MAE) for re-running experiments."""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATASETS = ["simulation", "simulation_normalonly", "simulation_normal_50"]

# Models to keep (MAE variants)
KEEP_MODELS = ["mae_adaptive", "mae_teacheronly"]

def clean_dataset(dataset_name: str):
    """Remove baseline models and update results.json."""
    results_dir = PROJECT_ROOT / "comparison/results" / dataset_name
    results_json_path = results_dir / "results.json"

    if not results_dir.exists():
        print(f"[SKIP] {dataset_name}: Directory does not exist")
        return

    print(f"\n{'='*70}")
    print(f"Cleaning: {dataset_name}")
    print(f"{'='*70}")

    # Load results.json
    if results_json_path.exists():
        with open(results_json_path, 'r') as f:
            results_data = json.load(f)
    else:
        print(f"  Warning: results.json not found")
        results_data = {"models": {}}

    # List all model directories
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    deleted_dirs = []
    deleted_models = []

    # Delete baseline model directories
    for model_dir in model_dirs:
        model_name = model_dir.name

        if model_name in KEEP_MODELS:
            print(f"  [KEEP] {model_name}")
            continue

        # Delete directory
        try:
            shutil.rmtree(model_dir)
            deleted_dirs.append(model_name)
            print(f"  [DELETE DIR] {model_name}")
        except Exception as e:
            print(f"  [ERROR] Failed to delete {model_name}: {e}")

    # Remove from results.json
    if "models" in results_data:
        original_count = len(results_data["models"])

        # Keep only MAE models
        results_data["models"] = {
            name: metrics
            for name, metrics in results_data["models"].items()
            if name in KEEP_MODELS
        }

        deleted_count = original_count - len(results_data["models"])

        if deleted_count > 0:
            # Save updated results.json
            with open(results_json_path, 'w') as f:
                json.dump(results_data, f, indent=2)

            print(f"\n  Updated results.json:")
            print(f"    Before: {original_count} models")
            print(f"    After:  {len(results_data['models'])} models (MAE only)")
            print(f"    Deleted: {deleted_count} models")

    print(f"\n  Summary:")
    print(f"    Deleted directories: {len(deleted_dirs)}")
    print(f"    Kept: {', '.join(KEEP_MODELS)}")


def main():
    print("="*70)
    print("Delete Baseline Model Results (Keep MAE)")
    print("="*70)

    for dataset in DATASETS:
        clean_dataset(dataset)

    print("\n" + "="*70)
    print("Done! Ready to re-run baseline experiments")
    print("="*70)


if __name__ == "__main__":
    main()
