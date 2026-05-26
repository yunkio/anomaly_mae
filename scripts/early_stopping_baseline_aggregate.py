"""Aggregate baseline best-epoch pak_auc_f1 for 15 active legacy + Neural models
across 6 datasets (SMD = 15 TimeSeAD avg, Exathlon = 6 apps avg),
to compare ranks against MAE 271 + Early-Stopping variant.

Output:
  /home/ykio/notebooks/claude/temp/early_stopping/baseline_aggregated.json
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

BASE_DIR = Path("/home/ykio/notebooks/claude/comparison/results/experiments/"
                "3_20260312_203923_baseline_minmax_normalonly")

# 15 active legacy + Neural models (note: 7 new SOTA are not yet completed)
BASELINE_MODELS = [
    # Simple (5)
    "random", "sensor_range", "pca_error", "l2_norm", "nn_distance",
    # Neural (3)
    "mlp", "mlpmixer", "transformer",
    # SOTA legacy (7)
    "gcn_lstm", "anomaly_transformer", "tranad", "usad", "dagmm", "gdn", "omnianomaly",
]

SMD_15_MACHINES = [
    "machine-1-2", "machine-1-7",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4",
    "machine-2-6", "machine-2-7", "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3",
    "machine-3-6", "machine-3-8", "machine-3-9",
]
EXATHLON_APPS = ["app1", "app2", "app4", "app5", "app6", "app9"]

# Dataset directory mapping
DS_DIRS = {
    "SWaT_excl22": ["SWaT/A1A2_excl22"],
    "WaDi_A1": ["WaDi/A1"],
    "WaDi_A2": ["WaDi/A2"],
    "PSM": ["PSM"],
    "SMD_avg": [f"SMD/{m}" for m in SMD_15_MACHINES],
    "Exathlon_avg": [f"Exathlon/{a}" for a in EXATHLON_APPS],
}


def best_pak_auc_f1(em_path: Path) -> float | None:
    """Return best (max) pak_auc_f1 over all epochs in epoch_metrics.json."""
    if not em_path.exists():
        return None
    try:
        em = json.load(open(em_path))
    except (json.JSONDecodeError, OSError):
        return None
    epochs = em.get("epochs", [])
    if not epochs:
        return None
    vals = [e.get("pak_auc_f1") for e in epochs if e.get("pak_auc_f1") is not None]
    if not vals:
        return None
    return max(vals)


def main():
    results = {}  # {model: {dataset_group: value}}

    for model in BASELINE_MODELS:
        results[model] = {}
        for group, subdirs in DS_DIRS.items():
            vals = []
            for sub in subdirs:
                em = BASE_DIR / sub / model / "epoch_metrics.json"
                v = best_pak_auc_f1(em)
                if v is not None:
                    vals.append(v)
            if vals:
                results[model][group] = mean(vals)
            else:
                results[model][group] = None

    out = {
        "datasets": list(DS_DIRS.keys()),
        "baseline_models": BASELINE_MODELS,
        "values": results,  # {model: {dataset: pak_auc_f1}}
    }

    out_path = Path("/home/ykio/notebooks/claude/temp/early_stopping/baseline_aggregated.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Pretty print
    print("Baseline pak_auc_f1 (best epoch) — SMD avg = 15 TimeSeAD, Exathlon avg = 6 apps")
    print("=" * 110)
    hdr = ["Model"] + list(DS_DIRS.keys())
    print(f"{'Model':22s} " + " ".join(f"{g:14s}" for g in DS_DIRS.keys()))
    for m, vals in results.items():
        line = f"{m:22s} "
        for g in DS_DIRS.keys():
            v = vals.get(g)
            line += f"{v:14.4f} " if v is not None else f"{'NaN':>14s} "
        print(line)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
