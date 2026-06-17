"""Build cached TEP streams for experiment #12 (type-generalization, simple baselines).

Reads the Rieth et al. RData files ONCE and materializes NPZ caches:
  data/test_stream.npz   — fixed shared test set (all 20 faults x runs 441-460 + FF 461-500)
  data/train_{fold}.npz  — per-fold contaminated train (FF 1-240 + seen-family faulty, 60 runs)
  data/train_ffonly.npz  — clean-normal reference train (FF 1-240 only)
  data/manifest.json     — run allocations, stream stats, assertion results

Stream layout (deterministic):
  train: FF runs ascending, then seen faulty runs (fault asc, run asc)
  test : faulty runs (fault 1..20 asc, run 441..460 asc), then FF runs 461..500
Run boundaries are recorded so that NO window/smoothing may cross a run seam.

Usage:
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/build_tep_data.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tep_common import (
    ALL_FAULTS, DATA_DIR, FAULT_ONSET_IDX, FAULTY_TEST_RUNS, FF_TEST_RUNS,
    FF_TRAIN_RUNS, FOLDS, RUN_LEN, TEP_RAW_DIR, seen_faults,
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


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    import pyreadr

    t0 = time.time()
    manifest = {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'fault_onset_idx_0based': FAULT_ONSET_IDX,
                'run_len': RUN_LEN}

    # ---- FaultFree_Testing ----
    print("[1/3] Loading TEP_FaultFree_Testing.RData ...")
    ff = pyreadr.read_r(os.path.join(TEP_RAW_DIR, 'TEP_FaultFree_Testing.RData'))
    df_ff = list(ff.values())[0]
    meta_cols = {'faultNumber', 'simulationRun', 'sample'}
    feature_cols = [c for c in df_ff.columns if c not in meta_cols]
    assert len(feature_cols) == 52, f"expected 52 features, got {len(feature_cols)}"
    assert set(df_ff['faultNumber'].unique()) == {0}

    ff_train = _extract_runs(df_ff, FF_TRAIN_RUNS, feature_cols)
    ff_test = _extract_runs(df_ff, FF_TEST_RUNS, feature_cols)
    del df_ff, ff
    print(f"    FF train runs: {len(ff_train)}, FF test runs: {len(ff_test)}"
          f"  ({time.time()-t0:.0f}s)")

    # ---- Faulty_Testing (load once, slice all needed runs, free) ----
    print("[2/3] Loading TEP_Faulty_Testing.RData (836MB, takes a while) ...")
    needed_train_runs = {}  # fault -> set of run ids (union over folds)
    for fold, cfg in FOLDS.items():
        for f in seen_faults(fold):
            needed_train_runs.setdefault(f, set()).update(
                range(1, cfg['train_runs_per_fault'] + 1))

    fl = pyreadr.read_r(os.path.join(TEP_RAW_DIR, 'TEP_Faulty_Testing.RData'))
    df_fl = list(fl.values())[0]
    assert set(df_fl['faultNumber'].unique()) == set(ALL_FAULTS)
    assert [c for c in df_fl.columns if c not in meta_cols] == feature_cols

    faulty_train, faulty_test = {}, {}
    for f in ALL_FAULTS:
        df_f = df_fl[df_fl['faultNumber'] == f]
        if f in needed_train_runs:
            faulty_train[f] = _extract_runs(df_f, sorted(needed_train_runs[f]), feature_cols)
        faulty_test[f] = _extract_runs(df_f, FAULTY_TEST_RUNS, feature_cols)
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
    for f in ALL_FAULTS:
        for r in FAULTY_TEST_RUNS:
            arrs.append(faulty_test[f][r]); labs.append(faulty_label); fids.append(f)
    for r in FF_TEST_RUNS:
        arrs.append(ff_test[r]); labs.append(ff_label); fids.append(0)
    X, y, fid, bounds, run_table = _concat_runs(arrs, labs, fids)
    assert len(X) == (len(ALL_FAULTS) * len(FAULTY_TEST_RUNS) + len(FF_TEST_RUNS)) * RUN_LEN
    assert int(y.sum()) == len(ALL_FAULTS) * len(FAULTY_TEST_RUNS) * (RUN_LEN - FAULT_ONSET_IDX)
    assert not np.isnan(X).any(), "NaN in test stream"
    np.savez_compressed(os.path.join(DATA_DIR, 'test_stream.npz'),
                        X=X, y=y, fault_id=fid,
                        run_boundaries=np.array(bounds, dtype=np.int64))
    with open(os.path.join(DATA_DIR, 'test_run_table.json'), 'w') as fp:
        json.dump(run_table, fp)
    manifest['test'] = {
        'n_samples': int(len(X)), 'n_runs': len(run_table),
        'n_anomaly_pts': int(y.sum()), 'anomaly_ratio': float(y.mean()),
        'faulty_runs_per_fault': len(FAULTY_TEST_RUNS), 'ff_runs': len(FF_TEST_RUNS),
        'order': 'faults 1..20 asc x runs 441..460 asc, then FF 461..500',
    }
    print(f"    test_stream: {X.shape}, anomaly {y.mean():.2%}, runs {len(run_table)}")

    # ---- per-fold contaminated train streams ----
    for fold, cfg in FOLDS.items():
        arrs, labs, fids = [], [], []
        for r in FF_TRAIN_RUNS:
            arrs.append(ff_train[r]); labs.append(ff_label); fids.append(0)
        n_faulty = 0
        for f in seen_faults(fold):
            for r in range(1, cfg['train_runs_per_fault'] + 1):
                arrs.append(faulty_train[f][r]); labs.append(faulty_label); fids.append(f)
                n_faulty += 1
        assert n_faulty == 60, f"{fold}: faulty train runs {n_faulty} != 60"
        X, y, fid, bounds, run_table = _concat_runs(arrs, labs, fids)
        assert not np.isnan(X).any()
        np.savez_compressed(os.path.join(DATA_DIR, f'train_{fold}.npz'),
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
    arrs = [ff_train[r] for r in FF_TRAIN_RUNS]
    labs = [ff_label] * len(FF_TRAIN_RUNS)
    fids = [0] * len(FF_TRAIN_RUNS)
    X, y, fid, bounds, _ = _concat_runs(arrs, labs, fids)
    np.savez_compressed(os.path.join(DATA_DIR, 'train_ffonly.npz'),
                        X=X, y=y, fault_id=fid,
                        run_boundaries=np.array(bounds, dtype=np.int64))
    manifest['train_ffonly'] = {'n_samples': int(len(X)), 'n_runs': len(FF_TRAIN_RUNS),
                                'anomaly_ratio': 0.0}
    print(f"    train_ffonly: {X.shape}")

    manifest['feature_cols'] = feature_cols
    constant_in_ff = [feature_cols[i] for i in
                      np.where(np.std(X, axis=0) == 0)[0]]
    manifest['constant_features_in_ff_train'] = constant_in_ff  # kept (denom guard)
    with open(os.path.join(DATA_DIR, 'manifest.json'), 'w') as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s. Manifest: {os.path.join(DATA_DIR, 'manifest.json')}")
    if constant_in_ff:
        print(f"  NOTE: constant-in-FF-train features (kept, denom-guarded): {constant_in_ff}")


if __name__ == '__main__':
    main()
