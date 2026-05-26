"""Rank comparison v2: MAE 271 Oracle + ES variants vs 15 baselines."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean as stat_mean

import psutil

OUT_DIR = Path("/home/ykio/notebooks/claude/temp/early_stopping")
RAW = OUT_DIR / "sweep_raw_v2.json"
BASELINE = OUT_DIR / "baseline_aggregated.json"

SMD_15 = ["machine-1-2","machine-1-7","machine-2-1","machine-2-2","machine-2-3",
          "machine-2-4","machine-2-6","machine-2-7","machine-2-9","machine-3-1",
          "machine-3-2","machine-3-3","machine-3-6","machine-3-8","machine-3-9"]
EXATHLON_APPS = ["app1","app2","app4","app5","app6","app9"]
DATASET_GROUPS = ["SWaT_excl22","WaDi_A1","WaDi_A2","PSM","SMD_avg","Exathlon_avg"]


def cfg_key(r):
    return (r["metric"], r["op"], r["patience"], r["thresh_type"], r["thresh_value"])


def main():
    print(f"Loading {RAW.stat().st_size/1e6:.0f} MB sweep...")
    t0 = time.time()
    sweep = json.load(open(RAW))
    print(f"  loaded in {time.time()-t0:.1f}s, RSS={psutil.Process(os.getpid()).memory_info().rss/1e6:.0f}MB")

    baseline = json.load(open(BASELINE))

    # Aggregate per-dataset cfg → (val, stop_ep)
    print("Building cfg maps...")
    cfg_maps = {}
    for name, rd in sweep.items():
        cfg_maps[name] = {cfg_key(r): (r["pak_auc_f1_at_stop"], r["stop_epoch"])
                          for r in rd["rows"]}
    oracle_by_ds = {n: r["oracle_pak_auc_f1"] for n, r in sweep.items()}
    del sweep
    import gc
    gc.collect()
    print(f"  RSS after cleanup: {psutil.Process(os.getpid()).memory_info().rss/1e6:.0f}MB")

    smd_keys = [f"SMD_{m}" for m in SMD_15]
    exa_keys = [f"Exathlon_{a}" for a in EXATHLON_APPS]

    # Group means
    print("Aggregating group means...")
    def group_mean(ds_list):
        accum_val = defaultdict(list)
        accum_ep = defaultdict(list)
        for ds in ds_list:
            for cfg, (v, se) in cfg_maps[ds].items():
                accum_val[cfg].append(v)
                accum_ep[cfg].append(se)
        return {cfg: (sum(vs)/len(vs), sum(accum_ep[cfg])/len(accum_ep[cfg]))
                for cfg, vs in accum_val.items()}

    smd_mean = group_mean(smd_keys)
    exa_mean = group_mean(exa_keys)
    group_tables = {
        "SWaT_excl22": cfg_maps["SWaT_excl22"],
        "WaDi_A1": cfg_maps["WaDi_A1"],
        "WaDi_A2": cfg_maps["WaDi_A2"],
        "PSM": cfg_maps["PSM"],
        "SMD_avg": smd_mean,
        "Exathlon_avg": exa_mean,
    }

    oracle_6 = {
        "SWaT_excl22": oracle_by_ds["SWaT_excl22"],
        "WaDi_A1": oracle_by_ds["WaDi_A1"],
        "WaDi_A2": oracle_by_ds["WaDi_A2"],
        "PSM": oracle_by_ds["PSM"],
        "SMD_avg": stat_mean(oracle_by_ds[k] for k in smd_keys),
        "Exathlon_avg": stat_mean(oracle_by_ds[k] for k in exa_keys),
    }
    oracle_mean_6 = stat_mean(oracle_6.values())

    # Find common cfgs across 6 groups
    common = set(group_tables["SWaT_excl22"].keys())
    for g in ["WaDi_A1","WaDi_A2","PSM","SMD_avg","Exathlon_avg"]:
        common &= set(group_tables[g].keys())
    print(f"  common cfgs: {len(common)}")

    # Top cross-dataset cfgs
    cross_scored = []
    for cfg in common:
        per_group = {g: group_tables[g][cfg][0] for g in DATASET_GROUPS}
        per_group_ep = {g: group_tables[g][cfg][1] for g in DATASET_GROUPS}
        cross_scored.append({
            "cfg": cfg, "mean_6": stat_mean(per_group.values()),
            "per_group": per_group, "per_group_ep": per_group_ep,
        })
    cross_scored.sort(key=lambda r: r["mean_6"], reverse=True)
    top5_cross = cross_scored[:5]

    # Per-dataset best (upper bound)
    upper_bound_per_group = {}
    upper_bound_per_group_ep = {}
    for g in DATASET_GROUPS:
        best = max(group_tables[g].items(), key=lambda kv: kv[1][0])
        upper_bound_per_group[g] = best[1][0]
        upper_bound_per_group_ep[g] = best[1][1]

    # Build leaderboard
    leaderboard = []
    for m in baseline["baseline_models"]:
        leaderboard.append({
            "model": m, "kind": "baseline",
            "per_group": baseline["values"][m],
        })

    leaderboard.append({
        "model": "MAE 271 (Oracle)", "kind": "mae_oracle",
        "per_group": oracle_6,
    })

    leaderboard.append({
        "model": "MAE 271 ES (per-dataset oracle, upper bound)",
        "kind": "mae_es_oracle_per_ds",
        "per_group": upper_bound_per_group,
        "per_group_stop_epoch": upper_bound_per_group_ep,
    })

    for i, cs in enumerate(top5_cross):
        cfg = cs["cfg"]
        m, op, P, tt, tv = cfg
        label = f"MAE 271 ES #{i+1} ({m}, op={op}, P={P}, T={tt}={tv})"
        leaderboard.append({
            "model": label, "kind": "mae_es",
            "cfg": {"metric": m, "op": op, "patience": P,
                    "thresh_type": tt, "thresh_value": tv},
            "per_group": cs["per_group"],
            "per_group_stop_epoch": cs["per_group_ep"],
        })

    # Compute ranks per group, then rank_avg
    for g_idx, g in enumerate(DATASET_GROUPS):
        vals = [(i, row["per_group"].get(g)) for i, row in enumerate(leaderboard)]
        vals_sorted = sorted(vals, key=lambda x: (x[1] is None, -(x[1] or 0)))
        rank_map = {idx: i+1 for i, (idx, _) in enumerate(vals_sorted)}
        for i, row in enumerate(leaderboard):
            row.setdefault("per_ds_rank", {})[g] = rank_map[i]

    for row in leaderboard:
        row["rank_avg"] = stat_mean(row["per_ds_rank"].values())
        row["mean_pak_auc_f1"] = stat_mean(
            v for v in row["per_group"].values() if v is not None
        )
    leaderboard.sort(key=lambda r: r["rank_avg"])

    out = {
        "datasets": DATASET_GROUPS,
        "oracle_per_group": oracle_6,
        "oracle_mean_6": oracle_mean_6,
        "leaderboard": leaderboard,
        "top5_cross_dataset_es": [
            {"metric": cs["cfg"][0], "op": cs["cfg"][1],
             "patience": cs["cfg"][2], "thresh_type": cs["cfg"][3],
             "thresh_value": cs["cfg"][4],
             "mean_6": cs["mean_6"], "per_group": cs["per_group"],
             "per_group_stop_epoch": cs["per_group_ep"]}
            for cs in top5_cross
        ],
        "per_dataset_upper_bound": {
            g: {"value": upper_bound_per_group[g], "stop_epoch": upper_bound_per_group_ep[g]}
            for g in DATASET_GROUPS
        },
    }
    with open(OUT_DIR / "rank_comparison_v2.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n" + "="*130)
    print("LEADERBOARD (sorted by Rank Avg over 6 dataset groups)")
    print("="*130)
    hdr = f"{'Rank':4s} {'Model':72s} " + " ".join(f"{g[:8]:>9s}" for g in DATASET_GROUPS) + f" {'RankAvg':>9s} {'MeanPAK':>9s}"
    print(hdr)
    print("-"*130)
    for i, row in enumerate(leaderboard):
        vals = " ".join(f"{row['per_group'].get(g, float('nan')):>9.4f}" for g in DATASET_GROUPS)
        print(f"{i+1:4d} {row['model'][:72]:72s} {vals} {row['rank_avg']:>9.2f} {row['mean_pak_auc_f1']:>9.4f}")

    print("\nPer-dataset oracle ES (upper bound, stop_epoch shown):")
    print("="*100)
    for g in DATASET_GROUPS:
        print(f"  {g:15s}  upper_bound={upper_bound_per_group[g]:.4f}  stop_epoch={upper_bound_per_group_ep[g]:.1f}  oracle={oracle_6[g]:.4f}")


if __name__ == "__main__":
    main()
