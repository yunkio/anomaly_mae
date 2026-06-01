"""
Aggregate Exathlon per-app baseline results into a single per-model summary.

Given a queue output directory containing `Exathlon/app{N}/{model}/epoch_metrics.json`
for N ∈ {1, 2, 4, 5, 6, 9} and 16 baseline models, compute the **mean across the 6
apps** for each (model, metric) pair, using each model's `best_epoch` (selected by
`pak_auc_f1`).

Saves output to `{output_base}/Exathlon/aggregated.csv` and a JSON summary.

Usage:
    python comparison/scripts/aggregate_exathlon.py \\
        --output-base /home/ykio/notebooks/claude/comparison/results/experiments/1_20260312_041500_baseline_minmax
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mae_anomaly.datasets.loaders import EXATHLON_APP_IDS

METRICS_OF_INTEREST = [
    "prc_auc", "roc_auc", "f1_score", "f1_t",
    "pak_auc_f1", "pak_auc_prc_auc", "pak_auc_f1_t",
    "pa_0_f1", "pa_20_f1", "pa_80_f1", "pa_100_f1",
]

BASELINE_MODELS = [
    "random", "sensor_range", "pca_error", "l2_norm", "nn_distance",
    "mlp", "mlpmixer", "transformer", "gcn_lstm",
    "anomaly_transformer", "tranad", "usad", "dagmm", "gdn", "omnianomaly",
]


def best_epoch_metrics(em_path: Path) -> dict | None:
    if not em_path.exists():
        return None
    try:
        em = json.load(open(em_path))
        epochs = em.get("epochs") or em
        if isinstance(epochs, list) and epochs:
            # Pick by pak_auc_f1
            best = max(epochs, key=lambda e: e.get("pak_auc_f1") or -1)
            return best
        # Some simple baselines store flat dict instead of per-epoch list
        if isinstance(em, dict) and "pak_auc_f1" in em:
            return em
        return None
    except Exception as e:
        print(f"  ! error reading {em_path}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-base", required=True)
    args = parser.parse_args()

    base = Path(args.output_base)
    exathlon_dir = base / "Exathlon"
    if not exathlon_dir.exists():
        sys.exit(f"Error: {exathlon_dir} does not exist.")

    rows = []
    for model in BASELINE_MODELS:
        per_app = {}
        for app in EXATHLON_APP_IDS:
            em_path = exathlon_dir / f"app{app}" / model / "epoch_metrics.json"
            best = best_epoch_metrics(em_path)
            if best is None:
                per_app[app] = None
                continue
            per_app[app] = {m: best.get(m) for m in METRICS_OF_INTEREST}

        # Aggregate mean across apps (skip missing)
        row = {"model": model}
        for m in METRICS_OF_INTEREST:
            vals = [per_app[a][m] for a in EXATHLON_APP_IDS
                    if per_app.get(a) is not None and per_app[a].get(m) is not None]
            row[f"{m}_mean"] = float(sum(vals) / len(vals)) if vals else None
            row[f"{m}_n_apps"] = len(vals)
        rows.append(row)
        # Also save per-app row for reference
        for app in EXATHLON_APP_IDS:
            if per_app.get(app) is None:
                continue
            row_p = {"model": model, "app": app}
            row_p.update(per_app[app])
            rows.append(row_p)

    df = pd.DataFrame(rows)
    out_csv = exathlon_dir / "aggregated.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved aggregated results → {out_csv}")

    # Also print summary table
    summary = df[df["app"].isna()].copy() if "app" in df.columns else df.copy()
    summary_cols = ["model"] + [f"{m}_mean" for m in METRICS_OF_INTEREST if f"{m}_mean" in summary.columns]
    print("\n=== Per-model means across 6 Exathlon apps ===\n")
    print(summary[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if x is not None else "—"))


if __name__ == "__main__":
    main()
