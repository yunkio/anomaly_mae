"""Analyze v2 sweep with multiprocess.

Input: temp/early_stopping/sweep_raw_v2.json
Output:
  - top configs per dataset
  - cross-dataset top configs (mean over 6 groups)
  - metric family ranking
  - stop_epoch distribution per top config (validation of user concern)
"""
from __future__ import annotations

import json
import os
import time
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from statistics import mean as stat_mean

import psutil

RAW_PATH = Path("/home/ykio/notebooks/claude/temp/early_stopping/sweep_raw_v2.json")
OUT_DIR = Path("/home/ykio/notebooks/claude/temp/early_stopping")

SMD_15 = [
    "machine-1-2", "machine-1-7",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4",
    "machine-2-6", "machine-2-7", "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3",
    "machine-3-6", "machine-3-8", "machine-3-9",
]
EXATHLON_APPS = ["app1", "app2", "app4", "app5", "app6", "app9"]
DATASET_GROUPS = ["SWaT_excl22", "WaDi_A1", "WaDi_A2", "PSM", "SMD_avg", "Exathlon_avg"]


def cfg_key(r):
    return (r["metric"], r["op"], r["patience"], r["thresh_type"], r["thresh_value"])


def _aggregate_for_dataset(args):
    """Per-dataset worker: returns {cfg_key: (pa_at_stop, stop_epoch)}."""
    name, rows = args
    out = {}
    for r in rows:
        out[cfg_key(r)] = (r["pak_auc_f1_at_stop"], r["stop_epoch"])
    return name, out


def main():
    print(f"Loading {RAW_PATH} ({RAW_PATH.stat().st_size/1e6:.0f} MB)...")
    t0 = time.time()
    sweep = json.load(open(RAW_PATH))
    print(f"  Loaded {len(sweep)} datasets in {time.time()-t0:.1f}s")

    oracle_by_ds = {n: r["oracle_pak_auc_f1"] for n, r in sweep.items()}
    oracle_ep_by_ds = {n: r["oracle_epoch"] for n, r in sweep.items()}

    # ── Build per-dataset cfg → (value, stop_ep) maps in parallel ──
    print("Aggregating per-dataset cfg maps with multiprocess...")
    items = [(name, sweep[name]["rows"]) for name in sweep]
    t1 = time.time()
    main_rss_b = psutil.Process(os.getpid()).memory_info().rss / 1e6
    with mp.Pool(processes=6) as pool:
        results = pool.map(_aggregate_for_dataset, items, chunksize=1)
    main_rss_a = psutil.Process(os.getpid()).memory_info().rss / 1e6
    print(f"  done in {time.time()-t1:.1f}s. main_RSS {main_rss_b:.0f}→{main_rss_a:.0f}MB")

    cfg_maps = dict(results)
    # Release sweep raw to free memory
    del sweep
    import gc
    gc.collect()
    print(f"  freed sweep raw; main_RSS={psutil.Process(os.getpid()).memory_info().rss/1e6:.0f}MB")

    # ── Aggregate 6-group means ──
    print("Computing group means...")
    smd_keys = [f"SMD_{m}" for m in SMD_15]
    exa_keys = [f"Exathlon_{a}" for a in EXATHLON_APPS]

    def group_mean(ds_list):
        """Compute per-cfg mean over ds_list."""
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
    print(f"  SMD configs: {len(smd_mean)}, Exathlon configs: {len(exa_mean)}")

    # ── Build per-group lookup table ──
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

    # ── Per-dataset best (each group independently) ──
    per_ds_best = {}
    for g, tbl in group_tables.items():
        best_cfg = max(tbl.items(), key=lambda kv: kv[1][0])
        per_ds_best[g] = {
            "metric": best_cfg[0][0],
            "op": best_cfg[0][1],
            "patience": best_cfg[0][2],
            "thresh_type": best_cfg[0][3],
            "thresh_value": best_cfg[0][4],
            "value": best_cfg[1][0],
            "stop_epoch": best_cfg[1][1],
            "oracle": oracle_6[g],
            "loss": oracle_6[g] - best_cfg[1][0],
            "loss_pct": (oracle_6[g] - best_cfg[1][0]) / oracle_6[g] * 100,
        }

    # ── Cross-dataset best: find configs present in ALL 6 groups ──
    print("Finding cross-dataset configs...")
    common = set(group_tables["SWaT_excl22"].keys())
    for g in ["WaDi_A1", "WaDi_A2", "PSM", "SMD_avg", "Exathlon_avg"]:
        common &= set(group_tables[g].keys())
    print(f"  common configs across 6 groups: {len(common)}")

    cross_scores = []
    for cfg in common:
        per_group = {g: group_tables[g][cfg][0] for g in DATASET_GROUPS}
        per_group_ep = {g: group_tables[g][cfg][1] for g in DATASET_GROUPS}
        cross_scores.append({
            "metric": cfg[0], "op": cfg[1],
            "patience": cfg[2], "thresh_type": cfg[3], "thresh_value": cfg[4],
            "mean_6": stat_mean(per_group.values()),
            "per_group": per_group,
            "per_group_stop_epoch": per_group_ep,
        })
    cross_scores.sort(key=lambda r: r["mean_6"], reverse=True)
    top_cross = cross_scores[:100]

    # ── Metric family ranking (best mean over op/P/T per metric) ──
    by_metric_best = defaultdict(lambda: {"mean": -1e18, "cfg": None})
    for r in cross_scores:
        m = r["metric"]
        if r["mean_6"] > by_metric_best[m]["mean"]:
            by_metric_best[m] = {"mean": r["mean_6"], "cfg": r}
    metric_ranking = sorted(by_metric_best.items(), key=lambda kv: -kv[1]["mean"])

    # ── Save ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "best_per_dataset_v2.json", "w") as f:
        json.dump({
            "oracle_6_groups": oracle_6,
            "oracle_mean_6_groups": oracle_mean_6,
            "best": per_ds_best,
        }, f, indent=2)

    with open(OUT_DIR / "cross_dataset_top100_v2.json", "w") as f:
        json.dump({"oracle_6_groups": oracle_6, "oracle_mean": oracle_mean_6,
                   "top_100_configs": top_cross}, f, indent=2)

    with open(OUT_DIR / "metric_family_ranking_v2.json", "w") as f:
        json.dump([{"metric": m, "best_mean_6": v["mean"], "best_cfg": v["cfg"]}
                   for m, v in metric_ranking[:50]], f, indent=2)

    # ── Print summary ──
    print("\n" + "=" * 120)
    print("ORACLE (best pak_auc_f1 over all eval checkpoints, per group)")
    print("=" * 120)
    for g, v in oracle_6.items():
        print(f"  {g:18s}: {v:.4f}")
    print(f"  {'GROUP_MEAN':18s}: {oracle_mean_6:.4f}")

    print("\n" + "=" * 120)
    print("BEST (metric, op, P, T) PER DATASET/GROUP")
    print("=" * 120)
    for g, b in per_ds_best.items():
        print(f"  {g:15s}  m={b['metric']:55s} op={b['op']:14s} P={b['patience']:3d} "
              f"T=({b['thresh_type']},{b['thresh_value']})  stop≈{b['stop_epoch']:.0f}  "
              f"val={b['value']:.4f} loss={b['loss']:.4f} ({b['loss_pct']:.2f}%)")

    print("\n" + "=" * 120)
    print(f"TOP 20 CROSS-DATASET (mean over 6 groups; oracle_mean={oracle_mean_6:.4f})")
    print("=" * 120)
    for i, r in enumerate(top_cross[:20]):
        loss = oracle_mean_6 - r["mean_6"]
        loss_pct = loss / oracle_mean_6 * 100
        eps_str = ",".join(f"{int(r['per_group_stop_epoch'][g]):>3d}" for g in DATASET_GROUPS)
        print(f"  #{i+1:2d} m={r['metric']:55s} op={r['op']:14s} "
              f"P={r['patience']:3d} T=({r['thresh_type']},{r['thresh_value']})  "
              f"mean={r['mean_6']:.4f} loss={loss_pct:5.2f}%  stop_eps=[{eps_str}]")

    print("\n" + "=" * 120)
    print("TOP 20 METRIC FAMILIES (best mean across op/P/T)")
    print("=" * 120)
    for i, (m, v) in enumerate(metric_ranking[:20]):
        c = v["cfg"]
        print(f"  #{i+1:2d}  {m:60s}  best_mean={v['mean']:.4f}  via op={c['op']:14s} P={c['patience']:3d} T=({c['thresh_type']},{c['thresh_value']})")


if __name__ == "__main__":
    main()
