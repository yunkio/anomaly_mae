"""Analyze raw early-stopping sweep results.

Inputs:
  /home/ykio/notebooks/claude/temp/early_stopping/sweep_raw.json (from early_stopping_analysis.py)

Outputs:
  - best_per_dataset.json:   best (metric,P,T) per dataset (min performance loss vs oracle)
  - cross_dataset_best.json: best single (metric,P,T) across the 6 dataset groups
                              (SWaT, WaDi_A1, WaDi_A2, PSM, SMD_avg, Exathlon_avg)
  - baseline_compare.json:   plug ES-271 vs 22 active baselines into rank-avg framework
  - summary.md:              human-readable summary
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

RAW_PATH = Path("/home/ykio/notebooks/claude/temp/early_stopping/sweep_raw.json")
OUT_DIR = Path("/home/ykio/notebooks/claude/temp/early_stopping")

SMD_15_MACHINES = [
    "machine-1-2", "machine-1-7",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4",
    "machine-2-6", "machine-2-7", "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3",
    "machine-3-6", "machine-3-8", "machine-3-9",
]
EXATHLON_APPS = ["app1", "app2", "app4", "app5", "app6", "app9"]

DATASET_GROUPS = ["SWaT_excl22", "WaDi_A1", "WaDi_A2", "PSM", "SMD_avg", "Exathlon_avg"]


def config_key(row: dict) -> tuple:
    return (row["metric"], row["patience"], row["thresh_type"], row["thresh_value"])


def aggregate_group(rows_by_ds: dict[str, list[dict]], ds_list: list[str]) -> list[dict]:
    """For a group of datasets (e.g., SMD 15 machines), average pak_auc_f1_at_stop
    over the same (metric,P,T) config. Returns list of {config, mean_pak_auc_f1, ...}."""
    config_to_vals = defaultdict(list)
    config_to_stop_eps = defaultdict(list)
    for ds in ds_list:
        for r in rows_by_ds[ds]:
            k = config_key(r)
            config_to_vals[k].append(r["pak_auc_f1_at_stop"])
            config_to_stop_eps[k].append(r["stop_epoch"])
    out = []
    for k, vals in config_to_vals.items():
        out.append({
            "metric": k[0],
            "patience": k[1],
            "thresh_type": k[2],
            "thresh_value": k[3],
            "mean_pak_auc_f1": mean(vals),
            "mean_stop_epoch": mean(config_to_stop_eps[k]),
            "n": len(vals),
        })
    return out


def main():
    raw = json.load(open(RAW_PATH))

    # ---- Reorganize: dataset → rows ----
    rows_by_ds = {name: r["rows"] for name, r in raw.items()}
    oracle_by_ds = {name: r["oracle_pak_auc_f1"] for name, r in raw.items()}
    oracle_ep_by_ds = {name: r["oracle_epoch"] for name, r in raw.items()}

    # ---- Aggregate group means ----
    smd_keys = [f"SMD_{m}" for m in SMD_15_MACHINES]
    exa_keys = [f"Exathlon_{a}" for a in EXATHLON_APPS]
    smd_rows = aggregate_group(rows_by_ds, smd_keys)
    exa_rows = aggregate_group(rows_by_ds, exa_keys)

    # Convert to dict for lookup
    def by_cfg(rows): return {(r["metric"], r["patience"], r["thresh_type"], r["thresh_value"]): r for r in rows}
    smd_by_cfg = by_cfg(smd_rows)
    exa_by_cfg = by_cfg(exa_rows)

    # Single-dataset rows by config
    single_by_cfg = {ds: by_cfg(rows_by_ds[ds]) for ds in ["SWaT_excl22", "WaDi_A1", "WaDi_A2", "PSM"]}

    # ---- For each dataset (group), find best (metric, P, T) ----
    best_per_group = {}
    # SWaT, WaDi_A1, WaDi_A2, PSM are single-dataset groups
    for ds in ["SWaT_excl22", "WaDi_A1", "WaDi_A2", "PSM"]:
        # Best config = max pak_auc_f1_at_stop, then min stop_epoch as tiebreaker
        best = max(rows_by_ds[ds], key=lambda r: (r["pak_auc_f1_at_stop"], -r["stop_epoch"]))
        oracle = oracle_by_ds[ds]
        loss = oracle - best["pak_auc_f1_at_stop"]
        best_per_group[ds] = {
            "metric": best["metric"],
            "patience": best["patience"],
            "thresh_type": best["thresh_type"],
            "thresh_value": best["thresh_value"],
            "stop_epoch": best["stop_epoch"],
            "pak_auc_f1_at_stop": best["pak_auc_f1_at_stop"],
            "oracle_pak_auc_f1": oracle,
            "oracle_epoch": oracle_ep_by_ds[ds],
            "performance_loss": loss,
            "performance_loss_pct": (loss / oracle * 100) if oracle else None,
        }

    # SMD_avg and Exathlon_avg
    for group_name, group_rows in [("SMD_avg", smd_rows), ("Exathlon_avg", exa_rows)]:
        best = max(group_rows, key=lambda r: (r["mean_pak_auc_f1"], -r["mean_stop_epoch"]))
        # Compute mean oracle for this group
        if group_name == "SMD_avg":
            keys = smd_keys
        else:
            keys = exa_keys
        oracle_mean = mean(oracle_by_ds[k] for k in keys)
        best_per_group[group_name] = {
            "metric": best["metric"],
            "patience": best["patience"],
            "thresh_type": best["thresh_type"],
            "thresh_value": best["thresh_value"],
            "mean_stop_epoch": best["mean_stop_epoch"],
            "mean_pak_auc_f1": best["mean_pak_auc_f1"],
            "mean_oracle_pak_auc_f1": oracle_mean,
            "performance_loss": oracle_mean - best["mean_pak_auc_f1"],
            "performance_loss_pct": (oracle_mean - best["mean_pak_auc_f1"]) / oracle_mean * 100,
        }

    # ---- Cross-dataset best: single (metric,P,T) that performs best across 6 groups ----
    # Compute per-config: for each of 6 groups, what's the pak_auc_f1?
    # Score = mean pak_auc_f1 across 6 groups (each group equally weighted)
    cross = []
    all_configs = set()
    for ds in ["SWaT_excl22", "WaDi_A1", "WaDi_A2", "PSM"]:
        for r in rows_by_ds[ds]:
            all_configs.add(config_key(r))

    for cfg in all_configs:
        try:
            vals = [
                single_by_cfg["SWaT_excl22"][cfg]["pak_auc_f1_at_stop"],
                single_by_cfg["WaDi_A1"][cfg]["pak_auc_f1_at_stop"],
                single_by_cfg["WaDi_A2"][cfg]["pak_auc_f1_at_stop"],
                single_by_cfg["PSM"][cfg]["pak_auc_f1_at_stop"],
                smd_by_cfg[cfg]["mean_pak_auc_f1"],
                exa_by_cfg[cfg]["mean_pak_auc_f1"],
            ]
        except KeyError:
            continue
        # mean stop epoch
        stop_eps = [
            single_by_cfg["SWaT_excl22"][cfg]["stop_epoch"],
            single_by_cfg["WaDi_A1"][cfg]["stop_epoch"],
            single_by_cfg["WaDi_A2"][cfg]["stop_epoch"],
            single_by_cfg["PSM"][cfg]["stop_epoch"],
            smd_by_cfg[cfg]["mean_stop_epoch"],
            exa_by_cfg[cfg]["mean_stop_epoch"],
        ]
        cross.append({
            "metric": cfg[0],
            "patience": cfg[1],
            "thresh_type": cfg[2],
            "thresh_value": cfg[3],
            "mean_pak_auc_f1_across_6_groups": mean(vals),
            "per_group": {
                "SWaT_excl22": vals[0],
                "WaDi_A1": vals[1],
                "WaDi_A2": vals[2],
                "PSM": vals[3],
                "SMD_avg": vals[4],
                "Exathlon_avg": vals[5],
            },
            "per_group_stop_epoch": {
                "SWaT_excl22": stop_eps[0],
                "WaDi_A1": stop_eps[1],
                "WaDi_A2": stop_eps[2],
                "PSM": stop_eps[3],
                "SMD_avg": stop_eps[4],
                "Exathlon_avg": stop_eps[5],
            },
        })
    cross.sort(key=lambda r: r["mean_pak_auc_f1_across_6_groups"], reverse=True)
    top_cross = cross[:50]

    # Oracle scores across 6 groups
    oracle_6 = {
        "SWaT_excl22": oracle_by_ds["SWaT_excl22"],
        "WaDi_A1": oracle_by_ds["WaDi_A1"],
        "WaDi_A2": oracle_by_ds["WaDi_A2"],
        "PSM": oracle_by_ds["PSM"],
        "SMD_avg": mean(oracle_by_ds[k] for k in smd_keys),
        "Exathlon_avg": mean(oracle_by_ds[k] for k in exa_keys),
    }
    oracle_mean_6 = mean(oracle_6.values())

    # ---- Top metric families (group by metric only, averaging over P/T) ----
    metric_summary = defaultdict(list)
    for r in cross:
        metric_summary[r["metric"]].append(r["mean_pak_auc_f1_across_6_groups"])
    metric_best = sorted(
        [(m, max(vs), mean(vs)) for m, vs in metric_summary.items()],
        key=lambda x: x[1], reverse=True,
    )

    # ---- Save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "best_per_dataset.json", "w") as f:
        json.dump({"oracle_6_groups": oracle_6, "oracle_mean_6_groups": oracle_mean_6,
                   "best": best_per_group}, f, indent=2)

    with open(OUT_DIR / "cross_dataset_top50.json", "w") as f:
        json.dump({"oracle_6_groups": oracle_6, "oracle_mean": oracle_mean_6,
                   "top_50_configs": top_cross}, f, indent=2)

    with open(OUT_DIR / "metric_family_ranking.json", "w") as f:
        json.dump({"metrics": [{"metric": m, "best_mean": b, "avg_mean": a} for m, b, a in metric_best]},
                  f, indent=2)

    # ---- Print summary ----
    print("=" * 80)
    print("ORACLE (best pak_auc_f1 over all eval checkpoints, per group)")
    print("=" * 80)
    for g, v in oracle_6.items():
        print(f"  {g:18s}: {v:.4f}  (oracle epoch info in raw)")
    print(f"  {'GROUP_MEAN':18s}: {oracle_mean_6:.4f}")

    print()
    print("=" * 80)
    print("BEST (metric, P, T) PER DATASET/GROUP")
    print("=" * 80)
    for g, b in best_per_group.items():
        if "mean_pak_auc_f1" in b:
            v = b["mean_pak_auc_f1"]
            ep = b.get("mean_stop_epoch", "?")
        else:
            v = b["pak_auc_f1_at_stop"]
            ep = b.get("stop_epoch", "?")
        print(f"  {g:18s}: metric={b['metric']:50s}  P={b['patience']:3d}  T=({b['thresh_type']},{b['thresh_value']})  stop={ep}  val={v:.4f}  loss={b['performance_loss']:.4f} ({b['performance_loss_pct']:.2f}%)")

    print()
    print("=" * 80)
    print("CROSS-DATASET BEST CONFIGS (top 10 by mean over 6 groups)")
    print("=" * 80)
    print(f"  Oracle mean over 6 groups: {oracle_mean_6:.4f}")
    print()
    for i, r in enumerate(top_cross[:10]):
        loss = oracle_mean_6 - r["mean_pak_auc_f1_across_6_groups"]
        loss_pct = loss / oracle_mean_6 * 100
        print(f"  #{i+1:2d}  metric={r['metric']:50s}  P={r['patience']:3d}  "
              f"T=({r['thresh_type']},{r['thresh_value']})  "
              f"mean={r['mean_pak_auc_f1_across_6_groups']:.4f}  loss={loss:.4f} ({loss_pct:.2f}%)")

    print()
    print("=" * 80)
    print("TOP METRIC FAMILIES (best mean across P/T)")
    print("=" * 80)
    for i, (m, b, a) in enumerate(metric_best[:15]):
        print(f"  #{i+1:2d}  {m:55s}  best={b:.4f}  avg={a:.4f}")


if __name__ == "__main__":
    main()
