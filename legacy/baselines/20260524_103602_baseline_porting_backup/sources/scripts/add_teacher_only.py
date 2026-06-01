"""
Add mae_teacheronly results to both simulation and simulation_normalonly results.json files.
"""

import json
from pathlib import Path

mae_teacheronly = {
    "point_level": {
        "prc_auc": 0.3212374058854709,
        "roc_auc": 0.6758204540821525,
        "f1_score": 0.3516219385727716,
        "recall": 0.324020184278293,
        "precision": 0.384364107653027
    },
    "f1_t": {
        "f1_t": 0.38491488184065903,
        "precision_t": 0.38369216724787847,
        "recall_t": 0.38614541421321275
    },
    "pa_k": {
        "pa_10": {
            "prc_auc": 0.7493077598790302,
            "roc_auc": 0.9560197893034905,
            "f1_score": 0.7003487071853038,
            "recall": 0.0,
            "precision": 0.0
        },
        "pa_20": {
            "prc_auc": 0.5709543035124504,
            "roc_auc": 0.9016209946632731,
            "f1_score": 0.5500205319552531,
            "recall": 0.0,
            "precision": 0.0
        },
        "pa_50": {
            "prc_auc": 0.25203867978793215,
            "roc_auc": 0.7103414921005109,
            "f1_score": 0.25532544201660273,
            "recall": 0.0,
            "precision": 0.0
        },
        "pa_80": {
            "prc_auc": 0.1165768598712191,
            "roc_auc": 0.45435523217172763,
            "f1_score": 0.1378796892050203,
            "recall": 0.0,
            "precision": 0.0
        },
        "pa_100": {
            "prc_auc": 0.08294420488514986,
            "roc_auc": 0.27462110321097377,
            "f1_score": 0.06454623044096729,
            "recall": 0.0,
            "precision": 0.0
        }
    },
    "timing": {
        "train_time": 0,
        "inference_time": 527.2
    },
    "source": "results/experiments/20260131_225102_phase2/000_best_model",
    "note": "Teacher reconstruction error only (no discrepancy)"
}

def update_results_file(file_path):
    """Add mae_teacheronly to results.json file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Add mae_teacheronly after mae_adaptive
    data['models']['mae_teacheronly'] = mae_teacheronly

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Updated: {file_path}")


if __name__ == "__main__":
    # Update both files
    simulation_path = Path("/home/ykio/notebooks/claude/comparison/results/simulation/results.json")
    normalonly_path = Path("/home/ykio/notebooks/claude/comparison/results/simulation_normalonly/results.json")

    update_results_file(simulation_path)
    update_results_file(normalonly_path)

    print("\nDone!")
