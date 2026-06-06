#!/usr/bin/env python
"""Backfill missing VUS-PR / VUS-ROC / Affiliation-F1 (+ r_based_f1) into the
*old* experiment results under temp/old_experiments/.

WHY: the old_271 / old_274 runs (May 2026) were evaluated before VUS / affiliation
were added to the per-cell headline metric block. The per-timestep best-epoch
anomaly score is preserved in epoch_scores/epoch_<best>_scores.npz['adaptive_score'];
we recompute the missing threshold-free / range metrics on THAT saved score plus
the ground-truth point labels, and add ONLY the missing keys into
experiment_metadata.json['metrics'] (and ['metrics_excl_region22'] for SWaT).
Existing keys (pak_auc_f1, prc_auc, ...) are NEVER overwritten.

LABEL PROVENANCE (correctness gate — we never write metrics on unverified labels):
  1. OWN  — best-epoch npz already carries point_labels (co-saved, perfectly
            aligned with the score). Trust directly.
  2. LOADER+XVAL — reconstruct test point_labels via the production path
            (DATASET_LOADERS -> SlidingWindowDataset(split='test').point_labels)
            AND require they are element-wise identical to some OTHER dir's OWN
            npz labels for the same cell (proves deterministic reproduction).
  3. LOADER+FAITH — no cross-val source, but loader labels reproduce the stored
            pak_auc_f1 within FAITH_TOL (clean deterministic real datasets:
            SWaT/WaDi).
  Anything else (e.g. simulation with a non-reproducible RNG realization, or a
  corrupt experiment_metadata.json) -> SKIP + report. Never write guessed labels.

Usage:
  python scripts/backfill_old_exp_vus_aff.py            # DRY-RUN (plan only)
  python scripts/backfill_old_exp_vus_aff.py --apply [--workers 6]
"""
import os, sys, json, glob, argparse, shutil, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

ROOT = '/home/ykio/notebooks/TSMAE/temp/old_experiments'
BACKUP = '/home/ykio/notebooks/TSMAE/.trash/0602/old_exp_vus_backfill'
TARGET_DIRS = [
    'old_271_20260508_094241_w500p10e4t3d2_dynamic_linear_minmax_k6',
    'old_274_20260513_055753_w500p10e4t3d2_dynamic_linear_minmax_k6',
]
# keys we ADD when missing (computed only at lite=False)
ADD_KEYS = ['vus_pr', 'vus_roc', 'affiliation_f1', 'affiliation_precision',
            'affiliation_recall', 'r_based_f1']
FAITH_TOL = 2e-3      # clean pak reproduction -> labels accepted outright
XVAL_TOL = 2e-2       # looser band: accept only if labels also cross-validate as
                      # canonical (element-wise == another dir's co-saved labels);
                      # rejects non-reproducible data (e.g. simulation, faith ~6e-2)
N_THRESHOLDS = 200
SLIDING_WINDOW = 100


class _R:
    __slots__ = ('start', 'end')
    def __init__(s, a, b): s.start = a; s.end = b


def regions_from_labels(lbl):
    lbl = np.asarray(lbl).astype(int); out = []; i = 0; n = len(lbl)
    while i < n:
        if lbl[i] == 1:
            j = i
            while j < n and lbl[j] == 1: j += 1
            out.append(_R(i, j)); i = j
        else:
            i += 1
    return out


def cell_to_loader(cname):
    if cname == 'PSM': return 'PSM'
    if cname == 'WaDi/A1': return 'WaDi_14days_A1'
    if cname == 'WaDi/A2': return 'WaDi_14days_A2'
    if cname.startswith('SWaT/'): return 'swat_A1A2'
    if cname.startswith('SMD/'): return 'SMD_simple_' + cname.split('/')[1]
    if cname.startswith('Exathlon/'): return 'Exathlon_simple_' + cname.split('/')[1]
    if cname.startswith('simulation'): return 'simulation'
    return None


def safe_json(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return None


def cellname(base, cell_dir):
    return '/'.join(os.path.relpath(cell_dir, base).split(os.sep))


def best_npz(cell_dir, best_ep):
    p = os.path.join(cell_dir, 'epoch_scores', f'epoch_{best_ep:03d}_scores.npz')
    return p if os.path.exists(p) else None


# ---------- label library (OWN labels from every dir, for cross-val) ----------
def build_label_lib():
    lib = {}  # cellname -> list of (length, labels)
    for top in os.listdir(ROOT):
        base = os.path.join(ROOT, top)
        if not os.path.isdir(base): continue
        for sd in glob.glob(os.path.join(base, '**', 'epoch_scores'), recursive=True):
            cell = os.path.dirname(sd)
            cn = cellname(base, cell)
            fs = glob.glob(os.path.join(sd, 'epoch_*_scores.npz'))
            if not fs: continue
            z = np.load(fs[-1])
            if 'point_labels' in z.files:
                lib.setdefault(cn, []).append(z['point_labels'].astype(int))
    return lib


def loader_test_labels(cname, cfg):
    from mae_anomaly import Config, set_seed
    from mae_anomaly.dataset_sliding import SlidingWindowDataset
    from mae_anomaly.datasets.loaders import get_dataset_loader
    from mae_anomaly.utils.experiment import resolve_test_stride
    config = Config()
    for k, v in (cfg or {}).items():
        if hasattr(config, k): setattr(config, k, v)
    key = cell_to_loader(cname)
    if key is None: return None
    data = get_dataset_loader(key)()
    di = {}
    if len(data) == 6: signals, pl, regs, fn, tr, di = data
    elif len(data) == 5: signals, pl, regs, fn, tr = data
    else: signals, pl, regs, fn = data; tr = 0.5
    config.num_features = signals.shape[1]
    set_seed(getattr(config, 'random_seed', 42))
    td = SlidingWindowDataset(
        signals=signals, point_labels=pl, anomaly_regions=regs,
        window_size=config.seq_length, stride=resolve_test_stride(config),
        mask_last_n=config.patch_size, split='test', train_ratio=tr,
        seed=getattr(config, 'random_seed', 42),
        run_boundaries=(di or {}).get('run_boundaries'),
        normalize_mode=config.normalize_mode,
        minmax_range=getattr(config, 'minmax_range', '0_1'),
        minmax_clamp_min=getattr(config, 'minmax_clamp_min', None),
        minmax_clamp_max=getattr(config, 'minmax_clamp_max', None),
        entity_segments=(di or {}).get('entity_norm_segments'))
    return np.array(td.point_labels).astype(int)


# ---------- worker: resolve labels + compute metrics for one cell ----------
def _process(task):
    from mae_anomaly.evaluator import compute_full_metric_set, compute_metrics_with_exclusion
    base, cell_dir, apply = task
    cn = cellname(base, cell_dir)
    rep = {'cell': os.path.relpath(cell_dir, ROOT), 'decision': None, 'src': None,
           'added': [], 'faith': None, 'err': None}
    meta = safe_json(os.path.join(cell_dir, 'experiment_metadata.json'))
    if meta is None:
        rep['decision'] = 'SKIP'; rep['err'] = 'corrupt/missing experiment_metadata.json'
        return rep
    best_ep = meta.get('timing', {}).get('best_epoch')
    if best_ep is None:
        rep['decision'] = 'SKIP'; rep['err'] = 'no best_epoch'; return rep
    mblock = meta.get('metrics') or {}
    is_swat = cn.startswith('SWaT/')
    is_excl = cn.endswith('A1A2_excl22')

    # locate score npz (excl22 shares the full dir's epoch_scores via symlink)
    np_path = best_npz(cell_dir, best_ep)
    if np_path is None:
        # excl22 may have no own epoch_scores; fall back to sibling full dir
        if is_excl:
            full_dir = cell_dir[:-len('A1A2_excl22')] + 'A1A2_full'
            fmeta = safe_json(os.path.join(full_dir, 'experiment_metadata.json')) or {}
            be2 = best_ep
            np_path = best_npz(full_dir, be2)
        if np_path is None:
            rep['decision'] = 'SKIP'; rep['err'] = f'no epoch_{best_ep:03d}_scores.npz'
            return rep

    z = np.load(np_path)
    score = z['adaptive_score'].astype(np.float64)

    # ---- resolve labels with provenance ----
    src = None; labels = None
    if 'point_labels' in z.files:
        labels = z['point_labels'].astype(int); src = 'own'
    else:
        cfg = meta.get('config', {})
        try:
            lo = loader_test_labels(cn, cfg)
        except Exception as e:
            lo = None; rep['err'] = f'loader: {type(e).__name__}: {str(e)[:60]}'
        if lo is not None and len(lo) == len(score):
            # cross-val: are loader labels element-wise identical to some OTHER
            # dir's OWN (co-saved) labels for this cell? -> proves they are the
            # canonical deterministic dataset labels (not an RNG realization).
            xok = any(len(c) == len(lo) and np.array_equal(c, lo)
                      for c in _LIB.get(cn, []))
            # faithfulness: do loader labels reproduce THIS cell's stored pak?
            # (contaminated by pak-algorithm drift, so used only with thresholds).
            # excl22 cells must apply the region-22 exclusion to match the stored
            # (masked) pak; otherwise the gate sees a huge spurious gap.
            _regs = regions_from_labels(lo)
            if is_excl:
                _info = meta.get('excl_region22_info') or {}
                _r22 = _R(int(_info['region_start']), int(_info['region_end'])) if _info else None
                m0 = (compute_metrics_with_exclusion(score, lo, _regs, _r22, lite=True)
                      if _r22 is not None
                      else compute_full_metric_set(score, lo, _regs, n_thresholds=N_THRESHOLDS,
                                                   sliding_window=SLIDING_WINDOW, lite=True))
            else:
                m0 = compute_full_metric_set(score, lo, _regs, n_thresholds=N_THRESHOLDS,
                                             sliding_window=SLIDING_WINDOW, lite=True)
            stored = mblock.get('pak_auc_f1')
            d = abs(float(m0['pak_auc_f1']) - float(stored)) if stored is not None else None
            rep['faith'] = d
            rep['xval'] = bool(xok)
            # 3-tier acceptance:
            #   (a) faith < FAITH_TOL              -> clean reproduction (SWaT/WaDi)
            #   (b) FAITH_TOL <= faith < XVAL_TOL AND xval-identical -> canonical
            #       labels, gap is pak-algorithm drift (SMD)
            #   else (e.g. simulation, faith ~6e-2 = different RNG data) -> SKIP
            if d is not None and d < FAITH_TOL:
                labels = lo; src = 'loader+faith'
            elif d is not None and d < XVAL_TOL and xok:
                labels = lo; src = 'loader+xval'
        elif lo is not None:
            rep['err'] = f'len mismatch score={len(score)} lbl={len(lo)}'

    if labels is None:
        rep['decision'] = 'SKIP'
        rep['err'] = rep['err'] or 'no reproducible labels (provenance gate failed)'
        return rep
    rep['src'] = src

    ml = min(len(score), len(labels))
    score, labels = score[:ml], labels[:ml]

    # ---- DRY-RUN: decide provenance + which keys are missing, skip heavy VUS ----
    if not apply:
        miss = [k for k in ADD_KEYS if (k not in mblock or mblock.get(k) is None)]
        excl_miss = []
        if is_swat and not is_excl and isinstance(meta.get('metrics_excl_region22'), dict):
            meb = meta['metrics_excl_region22']
            excl_miss = [k for k in ADD_KEYS if (k not in meb or meb.get(k) is None)]
        rep['added'] = miss
        rep['excl_added'] = excl_miss
        rep['decision'] = 'BACKFILL' if (miss or excl_miss) else 'NOOP'
        return rep

    regs = regions_from_labels(labels)

    # ---- compute full (lite=False) metrics ----
    if is_excl:
        info = meta.get('excl_region22_info') or {}
        if not info:
            rep['decision'] = 'SKIP'; rep['err'] = 'excl22 missing region info'; return rep
        r22 = _R(int(info['region_start']), int(info['region_end']))
        full = compute_metrics_with_exclusion(score, labels, regs, r22,
                                              lite=False)
    else:
        full = compute_full_metric_set(score, labels, regs,
                                       n_thresholds=N_THRESHOLDS,
                                       sliding_window=SLIDING_WINDOW, lite=False)

    add = {}
    for k in ADD_KEYS:
        if k in full and (k not in mblock or mblock.get(k) is None):
            add[k] = float(full[k])
    # SWaT full: also fill excl22 block
    excl_add = {}
    if is_swat and not is_excl:
        info = meta.get('excl_region22_info') or {}
        meblock = meta.get('metrics_excl_region22')
        if info and isinstance(meblock, dict):
            r22 = _R(int(info['region_start']), int(info['region_end']))
            fe = compute_metrics_with_exclusion(score, labels, regs, r22, lite=False)
            for k in ADD_KEYS:
                if k in fe and (k not in meblock or meblock.get(k) is None):
                    excl_add[k] = float(fe[k])

    rep['added'] = list(add.keys())
    rep['excl_added'] = list(excl_add.keys())
    rep['decision'] = 'BACKFILL' if (add or excl_add) else 'NOOP'

    if apply and (add or excl_add):
        # backup then write
        rel = os.path.relpath(cell_dir, ROOT)
        bdir = os.path.join(BACKUP, rel)
        os.makedirs(bdir, exist_ok=True)
        src_meta = os.path.join(cell_dir, 'experiment_metadata.json')
        if not os.path.exists(os.path.join(bdir, 'experiment_metadata.json')):
            shutil.copy2(src_meta, os.path.join(bdir, 'experiment_metadata.json'))
        meta.setdefault('metrics', {})
        meta['metrics'].update(add)
        if excl_add and isinstance(meta.get('metrics_excl_region22'), dict):
            meta['metrics_excl_region22'].update(excl_add)
        meta['_vus_aff_backfill'] = {
            'date': '2026-06-02', 'label_src': src, 'best_epoch': best_ep,
            'added': rep['added'], 'excl_added': rep['excl_added'],
            'note': 'VUS/affiliation/r_based recomputed on saved best-epoch adaptive_score.'}
        tmp = src_meta + '.tmp'
        with open(tmp, 'w') as f: json.dump(meta, f, indent=2)
        os.replace(tmp, src_meta)
    return rep


# globals for workers
_LIB = {}


def _init_lib(lib):
    global _LIB
    _LIB = lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    lib = build_label_lib()
    cells = []
    for top in TARGET_DIRS:
        base = os.path.join(ROOT, top)
        for mp in glob.glob(os.path.join(base, '**', 'experiment_metadata.json'), recursive=True):
            cells.append((base, os.path.dirname(mp), args.apply))
    cells.sort(key=lambda t: t[1])
    print(f"{'APPLY' if args.apply else 'DRY-RUN'} — {len(cells)} candidate cells "
          f"(label lib: {len(lib)} cellnames)\n")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_lib,
                             initargs=(lib,)) as ex:
        for rep in ex.map(_process, cells):
            results.append(rep)
            tag = rep['decision']
            extra = ''
            if rep['added'] or rep.get('excl_added'):
                extra = f" +{rep['added']}" + (f" excl+{rep['excl_added']}" if rep.get('excl_added') else '')
            if rep['faith'] is not None: extra += f" faith={rep['faith']:.1e}"
            if rep['err']: extra += f" ERR:{rep['err']}"
            print(f"  [{tag:8s}] src={str(rep['src']):12s} {rep['cell']:60s}{extra}")

    # summary
    from collections import Counter
    dec = Counter(r['decision'] for r in results)
    bsrc = Counter(r['src'] for r in results if r['decision'] == 'BACKFILL')
    print(f"\n=== SUMMARY ({time.time()-t0:.0f}s) ===")
    print("  decisions:", dict(dec))
    print("  backfill label sources:", dict(bsrc))
    skips = [r for r in results if r['decision'] == 'SKIP']
    if skips:
        print(f"  SKIPPED ({len(skips)}):")
        for r in skips:
            print(f"    {r['cell']:60s} {r['err']}")
    json.dump(results, open('/tmp/old_exp_vus_backfill_report.json', 'w'), indent=2)
    print("  report: /tmp/old_exp_vus_backfill_report.json")
    if args.apply:
        print(f"  backups: {BACKUP}")


if __name__ == '__main__':
    main()
