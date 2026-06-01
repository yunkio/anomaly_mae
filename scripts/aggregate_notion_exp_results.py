"""
Aggregate per-experiment metrics for Notion update.

Outputs: /home/ykio/notebooks/claude/temp/notion_exp_results.json

For each experiment in `results/experiments/<exp>_<timestamp>_*/`:
- simulation: pak_auc_f1, pak_auc_prc_auc, best_ep
- swat_full: from SWaT/A1A2_full/
- swat_excl22: from SWaT/A1A2_excl22/
- wadi_A1, wadi_A2
- psm
- smd_full (avg over 28 machines)
- smd_15 (avg over TimeSeAD 15 machines)

Then compute per-DS rank across all experiments, and RankAvg over 4 DS
(swat_excl22, wadi_A1, wadi_A2, smd_15).
"""
import json
from pathlib import Path
from collections import defaultdict

EXP_ROOT = Path("/home/ykio/notebooks/claude/results/experiments")
OUTPUT = Path("/home/ykio/notebooks/claude/temp/notion_exp_results.json")

SMD_15_MACHINES = [
    "machine-1-2", "machine-1-7",
    "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4",
    "machine-2-6", "machine-2-7", "machine-2-9",
    "machine-3-1", "machine-3-2", "machine-3-3",
    "machine-3-6", "machine-3-8", "machine-3-9",
]
assert len(SMD_15_MACHINES) == 15


def safe_load(p):
    if not p.exists():
        return None
    text = p.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: file is truncated mid-write. Salvage top-level metrics
        # via regex (timing.best_epoch + metrics.pak_auc_f1 / pak_auc_prc_auc).
        import re
        out = {"timing": {}, "metrics": {}}
        m = re.search(r'"best_epoch":\s*(\d+)', text)
        if m:
            out["timing"]["best_epoch"] = int(m.group(1))
        for key in ['pak_auc_f1', 'pak_auc_prc_auc', 'pak_auc_f1_raw',
                    'pak_auc_precision', 'pak_auc_recall']:
            m = re.search(rf'"{key}":\s*([-\d.eE]+)', text)
            if m:
                try:
                    out["metrics"][key] = float(m.group(1))
                except ValueError:
                    pass
        if out["metrics"]:
            return out
        return None


def extract_pak(meta_path, swat_mode=None):
    """Return {pak_f1, pak_prc, best_ep} or None.

    swat_mode: 'full' or 'excl22' — pulls from metrics_full or metrics_excl_region22
               if available, falls back to metrics.
    """
    d = safe_load(meta_path)
    if d is None:
        return None
    if swat_mode == "full":
        m = d.get("metrics_full") or d.get("metrics", {})
    elif swat_mode == "excl22":
        m = d.get("metrics_excl_region22") or d.get("metrics", {})
    else:
        m = d.get("metrics", {})
    pak_f1 = m.get("pak_auc_f1")
    pak_prc = m.get("pak_auc_prc_auc")
    best_ep = d.get("timing", {}).get("best_epoch")
    if pak_f1 is None:
        return None
    return {"pak_f1": float(pak_f1),
            "pak_prc": float(pak_prc) if pak_prc is not None else None,
            "best_ep": best_ep}


def aggregate_one_exp(exp_dir):
    """For one experiment directory, extract all per-dataset metrics."""
    out = {}
    # simulation
    sim_meta = exp_dir / "simulation" / "simulation" / "experiment_metadata.json"
    out["simulation"] = extract_pak(sim_meta)

    # SwAT — A1A2_excl22 directory contains BOTH metrics_excl_region22 and metrics_full
    swat_excl22_meta = exp_dir / "SWaT" / "A1A2_excl22" / "experiment_metadata.json"
    out["swat_excl22"] = extract_pak(swat_excl22_meta, swat_mode="excl22")
    # try A1A2_full first, fallback to A1A2_excl22's metrics_full
    swat_full_meta = exp_dir / "SWaT" / "A1A2_full" / "experiment_metadata.json"
    if swat_full_meta.exists():
        out["swat_full"] = extract_pak(swat_full_meta)
    else:
        out["swat_full"] = extract_pak(swat_excl22_meta, swat_mode="full")

    # WaDi
    out["wadi_A1"] = extract_pak(exp_dir / "WaDi" / "A1" / "experiment_metadata.json")
    out["wadi_A2"] = extract_pak(exp_dir / "WaDi" / "A2" / "experiment_metadata.json")

    # PSM (may not exist for older exps)
    psm_meta = exp_dir / "PSM" / "experiment_metadata.json"
    if not psm_meta.exists():
        psm_meta = exp_dir / "PSM" / "PSM" / "experiment_metadata.json"
    out["psm"] = extract_pak(psm_meta) if psm_meta.exists() else None

    # SMD per-machine
    smd_dir = exp_dir / "SMD"
    machines = {}
    if smd_dir.exists():
        for mdir in sorted(smd_dir.iterdir()):
            if not mdir.is_dir(): continue
            mname = mdir.name
            if not mname.startswith("machine-"): continue
            r = extract_pak(mdir / "experiment_metadata.json")
            if r is not None:
                machines[mname] = r

    # Exathlon per-app (6 apps: app1/2/4/5/6/9)
    exa_dir = exp_dir / "Exathlon"
    exa_apps = {}
    if exa_dir.exists():
        for adir in sorted(exa_dir.iterdir()):
            if not adir.is_dir(): continue
            aname = adir.name
            if not aname.startswith("app"): continue
            r = extract_pak(adir / "experiment_metadata.json")
            if r is not None:
                exa_apps[aname] = r
    if exa_apps:
        exa_f1 = [a["pak_f1"] for a in exa_apps.values()]
        exa_prc = [a["pak_prc"] for a in exa_apps.values() if a["pak_prc"] is not None]
        exa_ep = [a["best_ep"] for a in exa_apps.values() if a.get("best_ep") is not None]
        out["exathlon"] = {
            "pak_f1": sum(exa_f1) / len(exa_f1) if exa_f1 else None,
            "pak_prc": sum(exa_prc) / len(exa_prc) if exa_prc else None,
            "best_ep": sum(exa_ep) / len(exa_ep) if exa_ep else None,
            "n_apps": len(exa_apps),
        }
        out["exathlon_apps"] = exa_apps
    else:
        out["exathlon"] = None
        out["exathlon_apps"] = {}

    # SMD aggregates
    if machines:
        full_f1 = [m["pak_f1"] for m in machines.values()]
        full_prc = [m["pak_prc"] for m in machines.values() if m["pak_prc"] is not None]
        full_ep = [m["best_ep"] for m in machines.values() if m.get("best_ep") is not None]
        m15_data = [machines[m] for m in SMD_15_MACHINES if m in machines]
        m15_f1 = [m["pak_f1"] for m in m15_data]
        m15_prc = [m["pak_prc"] for m in m15_data if m["pak_prc"] is not None]
        m15_ep = [m["best_ep"] for m in m15_data if m.get("best_ep") is not None]
        out["smd_full"] = {
            "pak_f1": sum(full_f1) / len(full_f1) if full_f1 else None,
            "pak_prc": sum(full_prc) / len(full_prc) if full_prc else None,
            "best_ep": sum(full_ep) / len(full_ep) if full_ep else None,
            "n_machines": len(machines),
        }
        out["smd_15"] = {
            "pak_f1": sum(m15_f1) / len(m15_f1) if m15_f1 else None,
            "pak_prc": sum(m15_prc) / len(m15_prc) if m15_prc else None,
            "best_ep": sum(m15_ep) / len(m15_ep) if m15_ep else None,
            "n_machines": len(m15_data),
        }
        out["smd_machines"] = machines
    else:
        out["smd_full"] = None
        out["smd_15"] = None
        out["smd_machines"] = {}

    return out


def main():
    # Map exp number → most recent dir
    exp_dirs = {}
    for d in EXP_ROOT.iterdir():
        if not d.is_dir():
            continue
        parts = d.name.split("_", 1)
        if not parts[0].isdigit():
            continue
        n = int(parts[0])
        if n < 119 or n > 304:
            continue
        # Keep latest (alphabetical = chronological by timestamp prefix in name)
        if n not in exp_dirs or d.name > exp_dirs[n].name:
            exp_dirs[n] = d

    print(f"Found {len(exp_dirs)} experiments in 119-285 range")
    print(f"Exp range: {min(exp_dirs.keys())}-{max(exp_dirs.keys())}")
    missing = set(range(119, 285)) - set(exp_dirs.keys())
    print(f"Missing in 119-284 range: {sorted(missing)}")

    # Aggregate each
    results = {}
    for n in sorted(exp_dirs.keys()):
        d = exp_dirs[n]
        try:
            res = aggregate_one_exp(d)
            res["exp_dir"] = d.name
            results[str(n)] = res
            sim_f1 = res["simulation"]["pak_f1"] if res["simulation"] else None
            swat_excl22_f1 = res["swat_excl22"]["pak_f1"] if res["swat_excl22"] else None
            smd15_f1 = res["smd_15"]["pak_f1"] if res["smd_15"] else None
            print(f"  exp {n:3d}: sim={sim_f1}  excl22={swat_excl22_f1}  smd15={smd15_f1}")
        except Exception as e:
            print(f"  exp {n}: ERROR {e}")
            results[str(n)] = {"error": str(e), "exp_dir": d.name}

    # Compute ranks per DS (4 DS for original RankAvg: excl22, A1, A2, smd_15)
    rank_datasets = ["swat_excl22", "wadi_A1", "wadi_A2", "smd_15"]

    # For each rank DS, gather (exp, pak_f1, pak_prc) and rank (universe = all 245 exps)
    for metric_name in ["pak_f1", "pak_prc"]:
        for ds in rank_datasets + ["simulation", "swat_full", "smd_full", "psm", "exathlon"]:
            scores = []
            for exp, r in results.items():
                if "error" in r: continue
                node = r.get(ds)
                if node is None or node.get(metric_name) is None:
                    continue
                scores.append((exp, node[metric_name]))
            # Rank descending (higher score = better = rank 1)
            scores.sort(key=lambda x: -x[1])
            for rank, (exp, _) in enumerate(scores, start=1):
                if "ranks" not in results[exp]:
                    results[exp]["ranks"] = {}
                if metric_name not in results[exp]["ranks"]:
                    results[exp]["ranks"][metric_name] = {}
                results[exp]["ranks"][metric_name][ds] = rank

    # Per-exp: compute Avg and RankAvg over 4 rank datasets
    for exp, r in results.items():
        if "error" in r: continue
        for metric_name in ["pak_f1", "pak_prc"]:
            vals, ranks = [], []
            for ds in rank_datasets:
                node = r.get(ds)
                if node is None or node.get(metric_name) is None:
                    continue
                vals.append(node[metric_name])
                rk = r.get("ranks", {}).get(metric_name, {}).get(ds)
                if rk: ranks.append(rk)
            if vals:
                r.setdefault("avg", {})[metric_name] = sum(vals) / len(vals)
                r.setdefault("n_ds_avg", {})[metric_name] = len(vals)
            if ranks and len(ranks) == 4:
                r.setdefault("rank_avg", {})[metric_name] = sum(ranks) / len(ranks)

    # Save
    OUTPUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved aggregated results: {OUTPUT}")
    print(f"Total exps: {len(results)}")

    # Summary
    n_valid = sum(1 for r in results.values() if "error" not in r)
    print(f"Valid (no error): {n_valid}")
    n_with_sim = sum(1 for r in results.values() if r.get("simulation"))
    n_with_swat_full = sum(1 for r in results.values() if r.get("swat_full"))
    n_with_swat_excl22 = sum(1 for r in results.values() if r.get("swat_excl22"))
    n_with_smd15 = sum(1 for r in results.values() if r.get("smd_15"))
    print(f"  with simulation: {n_with_sim}")
    print(f"  with swat_full: {n_with_swat_full}")
    print(f"  with swat_excl22: {n_with_swat_excl22}")
    print(f"  with smd_15: {n_with_smd15}")


if __name__ == "__main__":
    main()
