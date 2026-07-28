"""Build cached TEP streams for experiment #12 (type-generalization, simple baselines).

Reads the Rieth et al. RData files ONCE and materializes NPZ caches:
  data/test_stream.npz   — fixed shared test set (all 20 faults x 20 runs + 40 FF runs)
  data/train_{fold}.npz  — per-fold contaminated train (FF 240 + seen-family faulty, 60 runs)
  data/train_ffonly.npz  — clean-normal reference train (FF 240 only)
  data/manifest.json     — run allocations, stream stats, assertion results

Stream layout (deterministic):
  train: FF runs ascending, then seen faulty runs (fault asc, run asc)
  test : faulty runs (fault 1..20 asc, run asc), then FF runs asc
Run boundaries are recorded so that NO window/smoothing may cross a run seam.

Data-seed axis (2026-07-24): --data-seed N re-draws the RUN ALLOCATION itself
(tep_common.allocate_runs — sampling without replacement, rng=default_rng(N))
while keeping every set size / fold composition / onset / ordering rule
canonical. Without --data-seed the frozen canonical allocation is used and the
outputs are value-identical to the original build.

Usage:
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/build_tep_data.py                 # canonical -> data/
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/build_tep_data.py --data-seed 40  # -> data_dataseed40/
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/build_tep_data.py --out-dir DIR   # explicit target dir
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tep_common import (
    ALL_FAULTS, FAULT_ONSET_IDX, FOLDS, RUN_LEN, TEP_RAW_DIR,
    allocate_runs, data_dir_for, seen_faults,
)


def _extract_runs(df, run_ids, feature_cols):
    """Return dict {run_id: (RUN_LEN, F) float32}, with hard sanity asserts."""
    out = {}
    sub = df[df['simulationRun'].isin(run_ids)]
    for run_id, g in sub.groupby('simulationRun'):
        g = g.sort_values('sample')
        samples = g['sample'].to_numpy()
        assert len(samples) == RUN_LEN, f"run {run_id}: len {len(samples)} != {RUN_LEN}"
        assert samples[0] == 1 and samples[-1] == RUN_LEN, f"run {run_id}: sample range"
        assert np.all(np.diff(samples) == 1), f"run {run_id}: samples not contiguous"
        out[int(run_id)] = g[feature_cols].to_numpy(dtype=np.float32)
    missing = set(run_ids) - set(out)
    assert not missing, f"missing runs: {sorted(missing)}"
    return out


def _concat_runs(run_arrays, labels_per_run, fault_ids):
    """Concatenate runs -> (X, y, fault_id_per_ts, boundaries, run_table)."""
    X_parts, y_parts, fid_parts, run_table = [], [], [], []
    boundaries, cum = [], 0
    for arr, lab, fid in zip(run_arrays, labels_per_run, fault_ids):
        X_parts.append(arr)
        y_parts.append(lab)
        fid_parts.append(np.full(len(arr), fid, dtype=np.int16))
        run_table.append({'fault': int(fid), 'start': int(cum), 'end': int(cum + len(arr))})
        cum += len(arr)
        boundaries.append(cum)
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    fid = np.concatenate(fid_parts, axis=0)
    return X, y, fid, boundaries[:-1], run_table  # internal boundaries only


def main(data_seed=None, out_dir=None):
    alloc = allocate_runs(data_seed)
    out_dir = out_dir or data_dir_for(data_seed)
    os.makedirs(out_dir, exist_ok=True)
    import pyreadr

    t0 = time.time()
    tag = 'canonical' if data_seed is None else f'data_seed={data_seed}'
    print(f"Building TEP typegen streams [{tag}] -> {out_dir}")
    manifest = {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'data_seed': alloc['data_seed'],
                'fault_onset_idx_0based': FAULT_ONSET_IDX,
                'run_len': RUN_LEN,
                'run_allocation': {
                    'ff_train': alloc['ff_train'],
                    'ff_test': alloc['ff_test'],
                    'faulty_train': {str(f): alloc['faulty_train'][f] for f in ALL_FAULTS},
                    'faulty_test': {str(f): alloc['faulty_test'][f] for f in ALL_FAULTS},
                }}

    # ---- FaultFree_Testing ----
    print("[1/3] Loading TEP_FaultFree_Testing.RData ...")
    ff = pyreadr.read_r(os.path.join(TEP_RAW_DIR, 'TEP_FaultFree_Testing.RData'))
    df_ff = list(ff.values())[0]
    meta_cols = {'faultNumber', 'simulationRun', 'sample'}
    feature_cols = [c for c in df_ff.columns if c not in meta_cols]
    assert len(feature_cols) == 52, f"expected 52 features, got {len(feature_cols)}"
    assert set(df_ff['faultNumber'].unique()) == {0}

    ff_train = _extract_runs(df_ff, alloc['ff_train'], feature_cols)
    ff_test = _extract_runs(df_ff, alloc['ff_test'], feature_cols)
    del df_ff, ff
    print(f"    FF train runs: {len(ff_train)}, FF test runs: {len(ff_test)}"
          f"  ({time.time()-t0:.0f}s)")

    # ---- Faulty_Testing (load once, slice all needed runs, free) ----
    print("[2/3] Loading TEP_Faulty_Testing.RData (836MB, takes a while) ...")
    fl = pyreadr.read_r(os.path.join(TEP_RAW_DIR, 'TEP_Faulty_Testing.RData'))
    df_fl = list(fl.values())[0]
    assert set(df_fl['faultNumber'].unique()) == set(ALL_FAULTS)
    assert [c for c in df_fl.columns if c not in meta_cols] == feature_cols

    faulty_train, faulty_test = {}, {}
    for f in ALL_FAULTS:
        df_f = df_fl[df_fl['faultNumber'] == f]
        train_ids = alloc['faulty_train'][f]
        if train_ids:
            faulty_train[f] = _extract_runs(df_f, train_ids, feature_cols)
        faulty_test[f] = _extract_runs(df_f, alloc['faulty_test'][f], feature_cols)
        print(f"    fault {f:>2}: train runs {len(faulty_train.get(f, {}))}, "
              f"test runs {len(faulty_test[f])}")
    del df_fl, fl
    print(f"    Faulty extraction done ({time.time()-t0:.0f}s)")

    faulty_label = np.zeros(RUN_LEN, dtype=np.int64)
    faulty_label[FAULT_ONSET_IDX:] = 1
    ff_label = np.zeros(RUN_LEN, dtype=np.int64)

    # ---- test stream (fixed, shared across folds) ----
    print("[3/3] Building streams ...")
    arrs, labs, fids = [], [], []
    n_faulty_test_runs = 0
    for f in ALL_FAULTS:
        for r in alloc['faulty_test'][f]:
            arrs.append(faulty_test[f][r]); labs.append(faulty_label); fids.append(f)
            n_faulty_test_runs += 1
    for r in alloc['ff_test']:
        arrs.append(ff_test[r]); labs.append(ff_label); fids.append(0)
    X, y, fid, bounds, run_table = _concat_runs(arrs, labs, fids)
    assert len(X) == (n_faulty_test_runs + len(alloc['ff_test'])) * RUN_LEN
    assert int(y.sum()) == n_faulty_test_runs * (RUN_LEN - FAULT_ONSET_IDX)
    assert not np.isnan(X).any(), "NaN in test stream"
    np.savez_compressed(os.path.join(out_dir, 'test_stream.npz'),
                        X=X, y=y, fault_id=fid,
                        run_boundaries=np.array(bounds, dtype=np.int64))
    with open(os.path.join(out_dir, 'test_run_table.json'), 'w') as fp:
        json.dump(run_table, fp)
    manifest['test'] = {
        'n_samples': int(len(X)), 'n_runs': len(run_table),
        'n_anomaly_pts': int(y.sum()), 'anomaly_ratio': float(y.mean()),
        'faulty_runs_per_fault': len(alloc['faulty_test'][ALL_FAULTS[0]]),
        'ff_runs': len(alloc['ff_test']),
        'order': ('faults 1..20 asc x runs 441..460 asc, then FF 461..500'
                  if data_seed is None else
                  f'faults 1..20 asc x seeded test runs asc, then FF seeded test '
                  f'runs asc (data_seed={data_seed})'),
    }
    print(f"    test_stream: {X.shape}, anomaly {y.mean():.2%}, runs {len(run_table)}")

    # ---- per-fold contaminated train streams ----
    for fold, cfg in FOLDS.items():
        arrs, labs, fids = [], [], []
        for r in alloc['ff_train']:
            arrs.append(ff_train[r]); labs.append(ff_label); fids.append(0)
        n_faulty = 0
        for f in seen_faults(fold):
            fold_train_ids = alloc['faulty_train'][f][:cfg['train_runs_per_fault']]
            assert len(fold_train_ids) == cfg['train_runs_per_fault'], \
                f"{fold}/fault{f}: {len(fold_train_ids)} != {cfg['train_runs_per_fault']}"
            for r in fold_train_ids:
                arrs.append(faulty_train[f][r]); labs.append(faulty_label); fids.append(f)
                n_faulty += 1
        assert n_faulty == 60, f"{fold}: faulty train runs {n_faulty} != 60"
        X, y, fid, bounds, run_table = _concat_runs(arrs, labs, fids)
        assert not np.isnan(X).any()
        np.savez_compressed(os.path.join(out_dir, f'train_{fold}.npz'),
                            X=X, y=y, fault_id=fid,
                            run_boundaries=np.array(bounds, dtype=np.int64))
        manifest[f'train_{fold}'] = {
            'n_samples': int(len(X)), 'n_runs': len(run_table),
            'seen_faults': seen_faults(fold),
            'runs_per_fault': cfg['train_runs_per_fault'],
            'anomaly_ratio': float(y.mean()),
        }
        print(f"    train_{fold}: {X.shape}, anomaly {y.mean():.2%} "
              f"(seen={seen_faults(fold)} x {cfg['train_runs_per_fault']})")

    # ---- clean-normal reference train (B0 analogue) ----
    arrs = [ff_train[r] for r in alloc['ff_train']]
    labs = [ff_label] * len(alloc['ff_train'])
    fids = [0] * len(alloc['ff_train'])
    X, y, fid, bounds, _ = _concat_runs(arrs, labs, fids)
    np.savez_compressed(os.path.join(out_dir, 'train_ffonly.npz'),
                        X=X, y=y, fault_id=fid,
                        run_boundaries=np.array(bounds, dtype=np.int64))
    manifest['train_ffonly'] = {'n_samples': int(len(X)),
                                'n_runs': len(alloc['ff_train']),
                                'anomaly_ratio': 0.0}
    print(f"    train_ffonly: {X.shape}")

    manifest['feature_cols'] = feature_cols
    constant_in_ff = [feature_cols[i] for i in
                      np.where(np.std(X, axis=0) == 0)[0]]
    manifest['constant_features_in_ff_train'] = constant_in_ff  # kept (denom guard)
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s. Manifest: {os.path.join(out_dir, 'manifest.json')}")
    if constant_in_ff:
        print(f"  NOTE: constant-in-FF-train features (kept, denom-guarded): {constant_in_ff}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data-seed', type=int, default=None,
                    help='seed the RUN ALLOCATION itself (default: frozen canonical '
                         'allocation, value-identical to the original build)')
    ap.add_argument('--out-dir', default=None,
                    help='output dir (default: data/ for canonical, '
                         'data_dataseed{N}/ for --data-seed N)')
    args = ap.parse_args()
    main(data_seed=args.data_seed, out_dir=args.out_dir)
