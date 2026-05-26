"""v6a 분석 — per-dataset JSON 파일 streaming + numpy compact storage.

메모리 효율:
- Phase 1: dataset 1개 load → cfg → int_id mapping 만들기 (작은 dataset부터)
- Phase 2: 각 dataset load → numpy array에 직접 저장 → 즉시 free
- Phase 3: 6-group mean (numpy)
- Phase 4: top configs, per warmup/rule/dataset best 추출
"""
from __future__ import annotations
import json, gc, time, os
from pathlib import Path
from statistics import mean
import psutil
import numpy as np

PER_DS_DIR = Path("/home/ykio/notebooks/claude/temp/early_stopping/v6a_per_ds")
SMD_15 = ["machine-1-2","machine-1-7","machine-2-1","machine-2-2","machine-2-3","machine-2-4",
          "machine-2-6","machine-2-7","machine-2-9","machine-3-1","machine-3-2","machine-3-3",
          "machine-3-6","machine-3-8","machine-3-9"]
EXATHLON_APPS = ["app1","app2","app4","app5","app6","app9"]
DG = ["SWaT_excl22","WaDi_A1","WaDi_A2","PSM","SMD_avg","Exathlon_avg"]
SMD_KEYS = [f"SMD_{m}" for m in SMD_15]
EXA_KEYS = [f"Exathlon_{a}" for a in EXATHLON_APPS]
ALL_DS = ["SWaT_excl22","WaDi_A1","WaDi_A2","PSM"] + SMD_KEYS + EXA_KEYS

def rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

def cfg_tuple(r):
    return (r["m"], r["op"], r["d"], r["rb"], r["rule"], r["P"], r["tt"], r["tv"], r["w"], r.get("hp", 0.0))

t0 = time.time()
print(f"Phase 1: building cfg → int_id from first dataset, RSS={rss_gb():.2f}GB")
with open(PER_DS_DIR / f"{ALL_DS[0]}.json") as f:
    first_ds = json.load(f)
cfg_to_id = {}
id_to_cfg = []
for r in first_ds["rows"]:
    c = cfg_tuple(r)
    if c not in cfg_to_id:
        cfg_to_id[c] = len(cfg_to_id)
        id_to_cfg.append(c)
N_CFG = len(cfg_to_id)
print(f"  unique cfgs: {N_CFG}, RSS={rss_gb():.2f}GB")

# Pre-allocate value arrays
ds_vals = np.full((len(ALL_DS), N_CFG), np.nan, dtype=np.float32)
ds_eps = np.full((len(ALL_DS), N_CFG), -1, dtype=np.int32)

# First dataset values
print(f"\nPhase 2: load per-dataset and fill numpy arrays...")
for r in first_ds["rows"]:
    cid = cfg_to_id[cfg_tuple(r)]
    ds_vals[0, cid] = r["v"]
    ds_eps[0, cid] = r["se"]
oracle = {ALL_DS[0]: first_ds["oracle_pak_auc_f1"]}
del first_ds; gc.collect()
print(f"  [1/{len(ALL_DS)}] {ALL_DS[0]:25s} done, RSS={rss_gb():.2f}GB")

# Remaining datasets
for i, ds_name in enumerate(ALL_DS[1:], start=1):
    with open(PER_DS_DIR / f"{ds_name}.json") as f:
        ds = json.load(f)
    for r in ds["rows"]:
        c = cfg_tuple(r)
        cid = cfg_to_id.get(c)
        if cid is None:
            # New cfg — extend arrays
            cid = N_CFG
            cfg_to_id[c] = cid
            id_to_cfg.append(c)
            new_col_v = np.full((len(ALL_DS), 1), np.nan, dtype=np.float32)
            new_col_e = np.full((len(ALL_DS), 1), -1, dtype=np.int32)
            ds_vals = np.hstack([ds_vals, new_col_v])
            ds_eps = np.hstack([ds_eps, new_col_e])
            N_CFG += 1
        ds_vals[i, cid] = r["v"]
        ds_eps[i, cid] = r["se"]
    oracle[ds_name] = ds["oracle_pak_auc_f1"]
    del ds; gc.collect()
    if i % 5 == 0 or i == len(ALL_DS) - 1:
        print(f"  [{i+1}/{len(ALL_DS)}] {ds_name:25s} N_CFG={N_CFG} RSS={rss_gb():.2f}GB")

# Oracle 6-group mean
o6 = {"SWaT_excl22": oracle["SWaT_excl22"], "WaDi_A1": oracle["WaDi_A1"],
      "WaDi_A2": oracle["WaDi_A2"], "PSM": oracle["PSM"],
      "SMD_avg": mean(oracle[k] for k in SMD_KEYS),
      "Exathlon_avg": mean(oracle[k] for k in EXA_KEYS)}
oracle_mean = mean(o6.values())
print(f"\nOracle 6-mean: {oracle_mean:.4f}")

# 6-group means per cfg
print(f"\nPhase 3: 6-group means, RSS={rss_gb():.2f}GB")
ds_idx = {n: i for i, n in enumerate(ALL_DS)}
single_groups = ["SWaT_excl22","WaDi_A1","WaDi_A2","PSM"]
per_g_vals = {g: ds_vals[ds_idx[g]] for g in single_groups}
per_g_vals["SMD_avg"] = np.nanmean(ds_vals[[ds_idx[k] for k in SMD_KEYS]], axis=0)
per_g_vals["Exathlon_avg"] = np.nanmean(ds_vals[[ds_idx[k] for k in EXA_KEYS]], axis=0)
per_g_eps = {g: ds_eps[ds_idx[g]].astype(np.float32) for g in single_groups}
per_g_eps["SMD_avg"] = ds_eps[[ds_idx[k] for k in SMD_KEYS]].astype(np.float32).mean(axis=0)
per_g_eps["Exathlon_avg"] = ds_eps[[ds_idx[k] for k in EXA_KEYS]].astype(np.float32).mean(axis=0)

mean_6 = np.nanmean(np.stack([per_g_vals[g] for g in DG]), axis=0)
mask = np.all(np.stack([~np.isnan(per_g_vals[g]) for g in DG]), axis=0)
mean_6[~mask] = np.nan
print(f"  valid configs: {mask.sum()}/{N_CFG}, RSS={rss_gb():.2f}GB")

# Free per-dataset arrays
del ds_vals; gc.collect()

# Top 30
print("\n" + "="*180)
print(f"TOP 30 CROSS-DATASET (v6a, oracle={oracle_mean:.4f})")
print("="*180)
sorted_ids = np.argsort(-mean_6)
for rank, cid in enumerate(sorted_ids[:30]):
    if np.isnan(mean_6[cid]): break
    c = id_to_cfg[cid]
    mv = float(mean_6[cid])
    loss = (oracle_mean - mv) / oracle_mean * 100
    eps_str = ",".join(f"{int(per_g_eps[g][cid]):>3d}" for g in DG)
    new_marker = "★ " if c[0].startswith("v6_composite_") else "  "
    print(f"  {new_marker}#{rank+1:2d} m={c[0]:48s} op={c[1]:8s} d={c[2]:10s} rule={c[4]:22s} "
          f"rb={c[3]:22s} P={c[5]} T=({c[6]},{c[7]}) w={c[8]:3d} hp={c[9]} "
          f"mean={mv:.4f} loss={loss:5.2f}% stops=[{eps_str}]")

# v6 신규 composite per-metric best
print("\n" + "="*180)
print("★ v6 신규 COMPOSITE METRIC 5개의 성능 (best across all dims)")
print("="*180)
V6_COMPOSITES = ["v6_composite_anomaly_stuckness_index",
                 "v6_composite_variance_weighted_separation",
                 "v6_composite_dynamics_cosine_similarity",
                 "v6_composite_log_anomaly_normal_ratio_student",
                 "v6_composite_acceleration_peak"]
metric_arr = np.array([id_to_cfg[i][0] for i in range(N_CFG)])
for m_name in V6_COMPOSITES:
    cm = (metric_arr == m_name) & mask
    if not cm.any():
        print(f"  {m_name:55s}  NOT FOUND")
        continue
    cids = np.where(cm)[0]
    best_idx = cids[np.nanargmax(mean_6[cids])]
    c = id_to_cfg[best_idx]
    mv = float(mean_6[best_idx])
    loss = (oracle_mean - mv) / oracle_mean * 100
    print(f"  {m_name:55s}  mean={mv:.4f} loss={loss:5.2f}%  "
          f"op={c[1]:8s} d={c[2]:10s} rule={c[4]:22s} P={c[5]} T=({c[6]},{c[7]}) w={c[8]:3d} hp={c[9]}")

# Per warmup best
print("\n" + "="*180)
print("PER WARMUP — Best")
print("="*180)
warmup_arr = np.array([id_to_cfg[i][8] for i in range(N_CFG)])
for w in [0, 50, 100, 150, 200, 240, 250, 260, 270, 280]:
    cm = (warmup_arr == w) & mask
    if not cm.any(): continue
    cids = np.where(cm)[0]
    best_idx = cids[np.nanargmax(mean_6[cids])]
    c = id_to_cfg[best_idx]
    mv = float(mean_6[best_idx])
    loss = (oracle_mean - mv) / oracle_mean * 100
    print(f"  warmup={w:3d}: mean={mv:.4f} loss={loss:5.2f}%  "
          f"m={c[0]:48s} op={c[1]:8s} rule={c[4]:22s} P={c[5]} T=({c[6]},{c[7]}) hp={c[9]}")

# Per rule best
print("\n" + "="*180)
print("PER RULE — Best (포함 hp grid)")
print("="*180)
rule_arr = np.array([id_to_cfg[i][4] for i in range(N_CFG)])
for rule in ["standard","peak_reversal","peak_reversal_reset","baseline_spike",
             "first_local_max","post_drop_peak","kth_peak_2"]:
    cm = (rule_arr == rule) & mask
    if not cm.any(): continue
    cids = np.where(cm)[0]
    best_idx = cids[np.nanargmax(mean_6[cids])]
    c = id_to_cfg[best_idx]
    mv = float(mean_6[best_idx])
    loss = (oracle_mean - mv) / oracle_mean * 100
    print(f"  {rule:22s}  mean={mv:.4f} loss={loss:5.2f}%  "
          f"m={c[0]:45s} op={c[1]:8s} d={c[2]:10s} P={c[5]} T=({c[6]},{c[7]}) w={c[8]:3d} hp={c[9]}")

# Per-dataset best
print("\n" + "="*180)
print("PER-DATASET BEST")
print("="*180)
for g in DG:
    vals_g = per_g_vals[g]
    valid_g = ~np.isnan(vals_g)
    if not valid_g.any(): continue
    cids = np.where(valid_g)[0]
    best_idx = cids[np.nanargmax(vals_g[cids])]
    c = id_to_cfg[best_idx]
    bv = float(vals_g[best_idx])
    be = float(per_g_eps[g][best_idx])
    print(f"  {g:15s} m={c[0]:45s} op={c[1]:8s} d={c[2]:10s} rule={c[4]:22s} "
          f"P={c[5]} T=({c[6]},{c[7]}) w={c[8]:3d} hp={c[9]} val={bv:.4f} stop={be:.0f}")

print(f"\n총 시간: {time.time()-t0:.1f}s, peak RSS={rss_gb():.2f}GB")
