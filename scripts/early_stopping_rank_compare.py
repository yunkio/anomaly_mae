"""Compare MAE 271 (oracle) + MAE 271 Early-Stopped variants vs 15 active baselines
on 6-dataset rank average (PA%K-AUC F1, Q3 minmax normalonly).

Inputs:
  /home/ykio/notebooks/claude/temp/early_stopping/sweep_raw.json
  /home/ykio/notebooks/claude/temp/early_stopping/baseline_aggregated.json

Output:
  /home/ykio/notebooks/claude/temp/early_stopping/rank_comparison.json
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

OUT_DIR = Path("/home/ykio/notebooks/claude/temp/early_stopping")
SWEEP_PATH = OUT_DIR / "sweep_raw.json"
BASELINE_PATH = OUT_DIR / "baseline_aggregated.json"

SMD_15_MACHINES = [
    "machine-1-2", "machine-1-7",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4",
    "machine-2-6", "machine-2-7", "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3",
    "machine-3-6", "machine-3-8", "machine-3-9",
]
EXATHLON_APPS = ["app1", "app2", "app4", "app5", "app6", "app9"]
DATASET_GROUPS = ["SWaT_excl22", "WaDi_A1", "WaDi_A2", "PSM", "SMD_avg", "Exathlon_avg"]


def main():
    sweep = json.load(open(SWEEP_PATH))
    baseline = json.load(open(BASELINE_PATH))

    # Build per-config × per-group dict from sweep
    # Single-dataset groups: SWaT_excl22, WaDi_A1, WaDi_A2, PSM
    # Multi-dataset groups: SMD_avg (15 machines avg), Exathlon_avg (6 apps avg)

    def config_key(r):
        return (r["metric"], r["patience"], r["thresh_type"], r["thresh_value"])

    # Single-dataset lookup
    single = {}  # {(group, config): val}
    for g in ["SWaT_excl22", "WaDi_A1", "WaDi_A2", "PSM"]:
        for r in sweep[g]["rows"]:
            single[(g, config_key(r))] = r["pak_auc_f1_at_stop"]

    # Multi-dataset aggregation
    def aggregate(group_name, ds_list):
        accum = defaultdict(list)
        for ds in ds_list:
            for r in sweep[ds]["rows"]:
                accum[config_key(r)].append(r["pak_auc_f1_at_stop"])
        return {cfg: mean(vs) for cfg, vs in accum.items()}

    smd_avg = aggregate("SMD_avg", [f"SMD_{m}" for m in SMD_15_MACHINES])
    exa_avg = aggregate("Exathlon_avg", [f"Exathlon_{a}" for a in EXATHLON_APPS])

    # ---- Configs that exist across all 6 groups ----
    all_configs = set(smd_avg.keys()) & set(exa_avg.keys())
    for g in ["SWaT_excl22", "WaDi_A1", "WaDi_A2", "PSM"]:
        cfgs_g = {cfg for (gx, cfg) in single.keys() if gx == g}
        all_configs &= cfgs_g

    # ---- MAE 271 Oracle ----
    oracle_per_group = {
        "SWaT_excl22": sweep["SWaT_excl22"]["oracle_pak_auc_f1"],
        "WaDi_A1": sweep["WaDi_A1"]["oracle_pak_auc_f1"],
        "WaDi_A2": sweep["WaDi_A2"]["oracle_pak_auc_f1"],
        "PSM": sweep["PSM"]["oracle_pak_auc_f1"],
        "SMD_avg": mean(sweep[f"SMD_{m}"]["oracle_pak_auc_f1"] for m in SMD_15_MACHINES),
        "Exathlon_avg": mean(sweep[f"Exathlon_{a}"]["oracle_pak_auc_f1"] for a in EXATHLON_APPS),
    }

    # ---- Per-dataset best ES (per Q3 spec) ----
    # For each dataset, find the (metric, P, T) maximizing pak_auc_f1 (i.e., min loss vs oracle)
    per_dataset_best = {}
    for g in ["SWaT_excl22", "WaDi_A1", "WaDi_A2", "PSM"]:
        rows = sweep[g]["rows"]
        best = max(rows, key=lambda r: r["pak_auc_f1_at_stop"])
        per_dataset_best[g] = {
            "metric": best["metric"], "patience": best["patience"],
            "thresh_type": best["thresh_type"], "thresh_value": best["thresh_value"],
            "value": best["pak_auc_f1_at_stop"], "stop_epoch": best["stop_epoch"],
        }
    # SMD_avg: best config maximizing avg
    smd_best_cfg = max(smd_avg.items(), key=lambda kv: kv[1])
    per_dataset_best["SMD_avg"] = {
        "metric": smd_best_cfg[0][0], "patience": smd_best_cfg[0][1],
        "thresh_type": smd_best_cfg[0][2], "thresh_value": smd_best_cfg[0][3],
        "value": smd_best_cfg[1],
    }
    exa_best_cfg = max(exa_avg.items(), key=lambda kv: kv[1])
    per_dataset_best["Exathlon_avg"] = {
        "metric": exa_best_cfg[0][0], "patience": exa_best_cfg[0][1],
        "thresh_type": exa_best_cfg[0][2], "thresh_value": exa_best_cfg[0][3],
        "value": exa_best_cfg[1],
    }

    # ---- Cross-dataset best ES: single (metric,P,T) maximizing mean over 6 groups ----
    cross_scores = []
    for cfg in all_configs:
        per_g = {
            "SWaT_excl22": single[("SWaT_excl22", cfg)],
            "WaDi_A1": single[("WaDi_A1", cfg)],
            "WaDi_A2": single[("WaDi_A2", cfg)],
            "PSM": single[("PSM", cfg)],
            "SMD_avg": smd_avg[cfg],
            "Exathlon_avg": exa_avg[cfg],
        }
        cross_scores.append({
            "cfg": cfg,
            "mean": mean(per_g.values()),
            "per_group": per_g,
        })
    cross_scores.sort(key=lambda r: r["mean"], reverse=True)

    # ---- Build leaderboard ----
    # Models = 15 baselines + MAE 271 (oracle) + MAE 271-ES (cross-best top 1) + per-dataset best variants
    leaderboard = []

    # Baseline rows
    for m in baseline["baseline_models"]:
        leaderboard.append({
            "model": m,
            "kind": "baseline",
            "per_group": baseline["values"][m],
        })

    # MAE 271 (Oracle)
    leaderboard.append({
        "model": "MAE 271 (Oracle)",
        "kind": "mae_oracle",
        "per_group": oracle_per_group,
    })

    # MAE 271 ES — top 5 cross-dataset configs
    for i, cs in enumerate(cross_scores[:5]):
        cfg = cs["cfg"]
        label = f"MAE 271 ES #{i+1} ({cfg[0]}, P={cfg[1]}, T={cfg[2]}={cfg[3]})"
        leaderboard.append({
            "model": label,
            "kind": "mae_es",
            "cfg": {"metric": cfg[0], "patience": cfg[1], "thresh_type": cfg[2], "thresh_value": cfg[3]},
            "per_group": cs["per_group"],
        })

    # MAE 271 ES — per-dataset oracle (uses different config per dataset; cheating but is bound)
    upper_bound_per_group = {g: per_dataset_best[g]["value"] for g in DATASET_GROUPS}
    leaderboard.append({
        "model": "MAE 271 ES (per-dataset oracle, upper bound)",
        "kind": "mae_es_oracle_per_ds",
        "per_group": upper_bound_per_group,
    })

    # ---- Compute ranks per dataset ----
    # rank: 1 = best (largest pak_auc_f1)
    ranks = []
    for g in DATASET_GROUPS:
        vals = [(i, row["per_group"].get(g)) for i, row in enumerate(leaderboard)]
        # sort descending by value
        vals_sorted = sorted(vals, key=lambda x: (x[1] is None, -(x[1] or 0)))
        rank_map = {idx: i + 1 for i, (idx, _) in enumerate(vals_sorted)}
        ranks.append(rank_map)

    # rank_avg
    for i, row in enumerate(leaderboard):
        per_ds_ranks = []
        for g_idx, g in enumerate(DATASET_GROUPS):
            per_ds_ranks.append(ranks[g_idx][i])
        row["per_ds_ranks"] = {g: r for g, r in zip(DATASET_GROUPS, per_ds_ranks)}
        row["rank_avg"] = mean(per_ds_ranks)
        row["mean_pak_auc_f1"] = mean(
            v for v in row["per_group"].values() if v is not None
        )

    # Sort by rank_avg
    leaderboard.sort(key=lambda r: r["rank_avg"])

    # ---- Save ----
    out = {
        "datasets": DATASET_GROUPS,
        "oracle_per_group": oracle_per_group,
        "per_dataset_best_es": per_dataset_best,
        "top_5_cross_dataset_es": [
            {"metric": cs["cfg"][0], "patience": cs["cfg"][1],
             "thresh_type": cs["cfg"][2], "thresh_value": cs["cfg"][3],
             "mean_pak_auc_f1": cs["mean"], "per_group": cs["per_group"]}
            for cs in cross_scores[:5]
        ],
        "leaderboard": leaderboard,
    }
    with open(OUT_DIR / "rank_comparison.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    # ---- Print ----
    print("=" * 130)
    print("LEADERBOARD (sorted by Rank Avg over 6 dataset groups)")
    print("Datasets: SWaT_excl22, WaDi_A1, WaDi_A2, PSM, SMD_avg (15), Exathlon_avg (6)")
    print("=" * 130)
    hdr = f"{'Rank':4s} {'Model':70s} " + " ".join(f"{g[:8]:>9s}" for g in DATASET_GROUPS) + f" {'RankAvg':>9s} {'MeanPAK':>9s}"
    print(hdr)
    print("-" * 130)
    for i, row in enumerate(leaderboard):
        vals = " ".join(f"{row['per_group'].get(g, math.nan):>9.4f}" for g in DATASET_GROUPS)
        print(f"{i+1:4d} {row['model'][:70]:70s} {vals} {row['rank_avg']:>9.2f} {row['mean_pak_auc_f1']:>9.4f}")

    print()
    print("Per-dataset best ES configs (each dataset chosen INDEPENDENTLY — upper bound):")
    print("=" * 100)
    for g, b in per_dataset_best.items():
        print(f"  {g:15s}  metric={b['metric']:50s}  P={b['patience']:3d}  T=({b['thresh_type']},{b['thresh_value']})  val={b['value']:.4f}")


if __name__ == "__main__":
    main()
