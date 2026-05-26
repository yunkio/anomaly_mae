"""Analyze v3 sweep."""
from __future__ import annotations

import json
import os
import time
import gc
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from statistics import mean as stat_mean

import psutil

RAW = Path("/home/ykio/notebooks/claude/temp/early_stopping/sweep_raw_v3.json")
OUT = Path("/home/ykio/notebooks/claude/temp/early_stopping")
SMD_15 = ["machine-1-2","machine-1-7","machine-2-1","machine-2-2","machine-2-3","machine-2-4",
          "machine-2-6","machine-2-7","machine-2-9","machine-3-1","machine-3-2","machine-3-3",
          "machine-3-6","machine-3-8","machine-3-9"]
EXATHLON_APPS = ["app1","app2","app4","app5","app6","app9"]
DATASET_GROUPS = ["SWaT_excl22","WaDi_A1","WaDi_A2","PSM","SMD_avg","Exathlon_avg"]


def cfg(r):
    return (r["metric"], r["op"], r["dir"], r["rollback"], r["rule"],
            r["P"], r["tt"], r["tv"])


def _agg(args):
    name, rows = args
    out = {}
    for r in rows:
        out[cfg(r)] = (r["v"], r["se"])
    return name, out


def main():
    print(f"Loading {RAW.stat().st_size/1e6:.0f} MB...")
    t0 = time.time()
    sweep = json.load(open(RAW))
    print(f"  loaded in {time.time()-t0:.1f}s, RSS={psutil.Process(os.getpid()).memory_info().rss/1e6:.0f}MB")

    print("Building cfg maps via multiprocess...")
    items = [(n, sweep[n]["rows"]) for n in sweep]
    oracle_by_ds = {n: r["oracle_pak_auc_f1"] for n, r in sweep.items()}
    t1 = time.time()
    with mp.Pool(processes=6) as pool:
        results = pool.map(_agg, items, chunksize=1)
    print(f"  done in {time.time()-t1:.1f}s")

    cfg_maps = dict(results)
    del sweep, results
    gc.collect()

    smd_keys = [f"SMD_{m}" for m in SMD_15]
    exa_keys = [f"Exathlon_{a}" for a in EXATHLON_APPS]

    def group_mean(ds_list):
        accum_v, accum_e = defaultdict(list), defaultdict(list)
        for ds in ds_list:
            for c, (v, se) in cfg_maps[ds].items():
                accum_v[c].append(v); accum_e[c].append(se)
        return {c: (sum(vs)/len(vs), sum(accum_e[c])/len(accum_e[c]))
                for c, vs in accum_v.items()}

    smd_mean = group_mean(smd_keys)
    exa_mean = group_mean(exa_keys)
    group_tables = {
        "SWaT_excl22": cfg_maps["SWaT_excl22"],
        "WaDi_A1": cfg_maps["WaDi_A1"], "WaDi_A2": cfg_maps["WaDi_A2"],
        "PSM": cfg_maps["PSM"], "SMD_avg": smd_mean, "Exathlon_avg": exa_mean,
    }
    oracle_6 = {
        "SWaT_excl22": oracle_by_ds["SWaT_excl22"],
        "WaDi_A1": oracle_by_ds["WaDi_A1"], "WaDi_A2": oracle_by_ds["WaDi_A2"],
        "PSM": oracle_by_ds["PSM"],
        "SMD_avg": stat_mean(oracle_by_ds[k] for k in smd_keys),
        "Exathlon_avg": stat_mean(oracle_by_ds[k] for k in exa_keys),
    }
    oracle_mean_6 = stat_mean(oracle_6.values())

    # Per-dataset best
    per_ds_best = {}
    for g, tbl in group_tables.items():
        b = max(tbl.items(), key=lambda kv: kv[1][0])
        per_ds_best[g] = {
            "metric": b[0][0], "op": b[0][1], "dir": b[0][2], "rollback": b[0][3],
            "rule": b[0][4], "P": b[0][5], "tt": b[0][6], "tv": b[0][7],
            "value": b[1][0], "stop_epoch": b[1][1],
            "oracle": oracle_6[g], "loss": oracle_6[g] - b[1][0],
            "loss_pct": (oracle_6[g] - b[1][0]) / oracle_6[g] * 100,
        }

    # Cross-dataset common
    common = set(group_tables["SWaT_excl22"].keys())
    for g in ["WaDi_A1","WaDi_A2","PSM","SMD_avg","Exathlon_avg"]:
        common &= set(group_tables[g].keys())
    print(f"  common configs: {len(common)}")

    cross = []
    for c in common:
        per_g = {g: group_tables[g][c][0] for g in DATASET_GROUPS}
        per_e = {g: group_tables[g][c][1] for g in DATASET_GROUPS}
        cross.append({"cfg": c, "mean_6": stat_mean(per_g.values()),
                      "per_group": per_g, "per_group_ep": per_e})
    cross.sort(key=lambda r: r["mean_6"], reverse=True)
    top_cross = cross[:50]

    # Metric family best
    by_metric = defaultdict(lambda: {"mean": -1e18, "cfg": None})
    for r in cross:
        m = r["cfg"][0]
        if r["mean_6"] > by_metric[m]["mean"]:
            by_metric[m] = {"mean": r["mean_6"], "cfg": r}
    metric_rank = sorted(by_metric.items(), key=lambda kv: -kv[1]["mean"])

    # ES rule comparison: separate standard / peak_reversal
    standard_top = sorted([r for r in cross if r["cfg"][4] == "standard"],
                          key=lambda x: -x["mean_6"])[:20]
    peak_top = sorted([r for r in cross if r["cfg"][4] == "peak_reversal"],
                      key=lambda x: -x["mean_6"])[:20]

    # Save
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "best_per_dataset_v3.json", "w") as f:
        json.dump({"oracle_6": oracle_6, "oracle_mean": oracle_mean_6,
                   "best": per_ds_best}, f, indent=2)
    with open(OUT / "cross_dataset_top50_v3.json", "w") as f:
        json.dump({"oracle_6": oracle_6, "top_50": top_cross}, f, indent=2, default=str)
    with open(OUT / "metric_family_ranking_v3.json", "w") as f:
        json.dump([{"metric": m, "best_mean": v["mean"], "cfg": v["cfg"]}
                   for m, v in metric_rank[:50]], f, indent=2, default=str)

    # Print
    print("\n" + "="*130)
    print(f"ORACLE 6-group mean = {oracle_mean_6:.4f}")
    print("="*130)
    for g, v in oracle_6.items():
        print(f"  {g:18s}: {v:.4f}")

    print("\n" + "="*130)
    print("PER-DATASET BEST")
    print("="*130)
    for g, b in per_ds_best.items():
        print(f"  {g:15s} m={b['metric']:55s} op={b['op']:14s} dir={b['dir']:10s} "
              f"rule={b['rule']:14s} rb={b['rollback']:24s} P={b['P']} T=({b['tt']},{b['tv']}) "
              f"stop≈{b['stop_epoch']:.0f} val={b['value']:.4f} loss={b['loss_pct']:.2f}%")

    print("\n" + "="*130)
    print(f"TOP 20 CROSS-DATASET (mean 6 groups; oracle_mean={oracle_mean_6:.4f})")
    print("="*130)
    for i, r in enumerate(cross[:20]):
        c = r["cfg"]
        loss_pct = (oracle_mean_6 - r["mean_6"]) / oracle_mean_6 * 100
        eps = ",".join(f"{int(r['per_group_ep'][g]):>3d}" for g in DATASET_GROUPS)
        print(f"  #{i+1:2d} m={c[0]:55s} op={c[1]:14s} dir={c[2]:10s} "
              f"rule={c[4]:14s} rb={c[3]:24s} P={c[5]} T=({c[6]},{c[7]}) "
              f"mean={r['mean_6']:.4f} loss={loss_pct:5.2f}% stops=[{eps}]")

    print("\n" + "="*130)
    print("TOP 20 METRIC FAMILIES (best 6-mean across all op/dir/rb/rule/P/T)")
    print("="*130)
    for i, (m, v) in enumerate(metric_rank[:20]):
        c = v["cfg"]["cfg"]
        print(f"  #{i+1:2d} {m:55s} best={v['mean']:.4f} via op={c[1]:14s} dir={c[2]:10s} "
              f"rule={c[4]:14s} rb={c[3]:24s} P={c[5]} T=({c[6]},{c[7]})")

    print("\n" + "="*130)
    print("STANDARD ES rule — Top 10")
    print("="*130)
    for i, r in enumerate(standard_top[:10]):
        c = r["cfg"]
        print(f"  #{i+1:2d} m={c[0]:50s} op={c[1]:14s} dir={c[2]:10s} "
              f"rb={c[3]:24s} P={c[5]} T=({c[6]},{c[7]}) mean={r['mean_6']:.4f}")

    print("\n" + "="*130)
    print("PEAK_REVERSAL ES rule — Top 10 (★ Type B signal detector)")
    print("="*130)
    for i, r in enumerate(peak_top[:10]):
        c = r["cfg"]
        print(f"  #{i+1:2d} m={c[0]:50s} op={c[1]:14s} dir={c[2]:10s} "
              f"rb={c[3]:24s} P={c[5]} T=({c[6]},{c[7]}) mean={r['mean_6']:.4f}")


if __name__ == "__main__":
    main()
