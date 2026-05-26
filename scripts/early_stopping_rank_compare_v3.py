"""Rank comparison v3."""
from __future__ import annotations
import json, os, gc, time
from collections import defaultdict
from pathlib import Path
from statistics import mean as stat_mean
import psutil

OUT = Path("/home/ykio/notebooks/claude/temp/early_stopping")
RAW = OUT / "sweep_raw_v3.json"
BASELINE = OUT / "baseline_aggregated.json"

SMD_15 = ["machine-1-2","machine-1-7","machine-2-1","machine-2-2","machine-2-3","machine-2-4",
          "machine-2-6","machine-2-7","machine-2-9","machine-3-1","machine-3-2","machine-3-3",
          "machine-3-6","machine-3-8","machine-3-9"]
EXATHLON_APPS = ["app1","app2","app4","app5","app6","app9"]
DATASET_GROUPS = ["SWaT_excl22","WaDi_A1","WaDi_A2","PSM","SMD_avg","Exathlon_avg"]

def cfg(r):
    return (r["metric"], r["op"], r["dir"], r["rollback"], r["rule"],
            r["P"], r["tt"], r["tv"])

def main():
    print(f"Loading {RAW.stat().st_size/1e6:.0f} MB...")
    t0 = time.time()
    sweep = json.load(open(RAW))
    print(f"  loaded in {time.time()-t0:.1f}s, RSS={psutil.Process(os.getpid()).memory_info().rss/1e6:.0f}MB")
    baseline = json.load(open(BASELINE))

    cfg_maps = {}
    for name, rd in sweep.items():
        cfg_maps[name] = {cfg(r): (r["v"], r["se"]) for r in rd["rows"]}
    oracle_by_ds = {n: r["oracle_pak_auc_f1"] for n, r in sweep.items()}
    del sweep; gc.collect()

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

    common = set(group_tables["SWaT_excl22"].keys())
    for g in ["WaDi_A1","WaDi_A2","PSM","SMD_avg","Exathlon_avg"]:
        common &= set(group_tables[g].keys())

    cross = []
    for c in common:
        per_g = {g: group_tables[g][c][0] for g in DATASET_GROUPS}
        per_e = {g: group_tables[g][c][1] for g in DATASET_GROUPS}
        cross.append({"cfg": c, "mean_6": stat_mean(per_g.values()),
                      "per_group": per_g, "per_group_ep": per_e})
    cross.sort(key=lambda r: r["mean_6"], reverse=True)
    top5 = cross[:5]

    upper_bound = {g: max(group_tables[g].items(), key=lambda kv: kv[1][0])
                   for g in DATASET_GROUPS}

    lb = []
    for m in baseline["baseline_models"]:
        lb.append({"model": m, "kind": "baseline", "per_group": baseline["values"][m]})
    lb.append({"model": "MAE 271 (Oracle)", "kind": "mae_oracle", "per_group": oracle_6})
    lb.append({
        "model": "MAE 271 ES (per-ds oracle, upper bound)",
        "kind": "mae_es_upper",
        "per_group": {g: upper_bound[g][1][0] for g in DATASET_GROUPS},
    })
    for i, cs in enumerate(top5):
        c = cs["cfg"]
        label = (f"MAE 271 ES #{i+1} ({c[0]}, op={c[1]}, rule={c[4]}, dir={c[2]}, rb={c[3]}, P={c[5]}, T={c[6]}={c[7]})")
        lb.append({"model": label, "kind": "mae_es",
                   "cfg": dict(zip(["metric","op","dir","rollback","rule","P","tt","tv"], c)),
                   "per_group": cs["per_group"],
                   "per_group_stop_epoch": cs["per_group_ep"]})

    # Compute ranks
    for g_idx, g in enumerate(DATASET_GROUPS):
        vals = [(i, row["per_group"].get(g)) for i, row in enumerate(lb)]
        vs = sorted(vals, key=lambda x: (x[1] is None, -(x[1] or 0)))
        rm = {idx: i+1 for i, (idx, _) in enumerate(vs)}
        for i, row in enumerate(lb):
            row.setdefault("per_ds_rank", {})[g] = rm[i]

    for row in lb:
        row["rank_avg"] = stat_mean(row["per_ds_rank"].values())
        row["mean"] = stat_mean(v for v in row["per_group"].values() if v is not None)
    lb.sort(key=lambda r: r["rank_avg"])

    with open(OUT / "rank_comparison_v3.json", "w") as f:
        json.dump({"oracle_6": oracle_6, "oracle_mean_6": oracle_mean_6,
                   "leaderboard": lb, "top5_cross": [{
                       "metric": cs["cfg"][0], "op": cs["cfg"][1],
                       "dir": cs["cfg"][2], "rollback": cs["cfg"][3],
                       "rule": cs["cfg"][4], "P": cs["cfg"][5],
                       "tt": cs["cfg"][6], "tv": cs["cfg"][7],
                       "mean_6": cs["mean_6"], "per_group": cs["per_group"],
                       "per_group_stop_epoch": cs["per_group_ep"],
                   } for cs in top5]}, f, indent=2, default=str)

    print("\n" + "="*150)
    print("LEADERBOARD (v3 — label-free, P=1-3, peak_reversal + standard, direction/rollback modes)")
    print("="*150)
    hdr = f"{'Rk':3s} {'Model':80s} " + " ".join(f"{g[:8]:>9s}" for g in DATASET_GROUPS) + f" {'RankAvg':>8s} {'Mean':>8s}"
    print(hdr); print("-"*150)
    for i, row in enumerate(lb):
        vals = " ".join(f"{row['per_group'].get(g, float('nan')):>9.4f}" for g in DATASET_GROUPS)
        print(f"{i+1:3d} {row['model'][:80]:80s} {vals} {row['rank_avg']:>8.2f} {row['mean']:>8.4f}")

if __name__ == "__main__":
    main()
