#!/usr/bin/env python
"""Survey: enumerate completed TSMAE experiment cells and classify in-scope for
the 2026-06-01 pre-warmup recon-only backfill.

A cell is IN-SCOPE iff:
  - anomaly_score_mode == 'adaptive'      (disc/FM enter the score only here)
  - teacher_only_warmup_epochs > 0        (there is a warmup window to gate)
  - has epoch_scores/*.npz with >=1 pre-warmup epoch (ep <= warmup)

Produces a JSON manifest with, per cell: config knobs, npz coverage (which
epochs, how many pre-warmup), eval epochs in epoch_metrics, stored best_epoch +
metric, checkpoint epoch, completion flag, and whether the saved adaptive_score
for a sampled pre-warmup epoch already equals teacher_recon_error (i.e. already
recon-only = nothing to do) or differs (needs R1 overwrite).

READ-ONLY. No file is modified.
"""
import os, json, glob, sys
import numpy as np

ROOT = '/home/ykio/notebooks/TSMAE/results/experiments'
OUT = '/tmp/prewarmup_backfill/manifest.json'
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def cell_config(cell_dir):
    """Return (score_mode, warmup, use_fm, source) from the best available config."""
    for name in ('best_config.json', 'config.json'):
        cfg = load_json(os.path.join(cell_dir, name))
        if cfg:
            return (cfg.get('anomaly_score_mode'),
                    cfg.get('teacher_only_warmup_epochs'),
                    cfg.get('use_feature_matching'), name)
    # fallback: experiment_metadata.json may carry a config block
    meta = load_json(os.path.join(cell_dir, 'experiment_metadata.json'))
    if meta and isinstance(meta.get('config'), dict):
        c = meta['config']
        return (c.get('anomaly_score_mode'),
                c.get('teacher_only_warmup_epochs'),
                c.get('use_feature_matching'), 'experiment_metadata.json')
    return (None, None, None, None)


def epoch_metrics_info(cell_dir):
    em = load_json(os.path.join(cell_dir, 'epoch_metrics.json'))
    if not em:
        return None
    rows = em.get('epochs', em if isinstance(em, list) else [])
    eval_epochs = sorted(int(r['epoch']) for r in rows if 'epoch' in r)
    return {'eval_interval': em.get('eval_interval'),
            'n_rows': len(rows), 'eval_epochs': eval_epochs,
            'has_vus': any('vus_pr' in r for r in rows),
            'metric_keys_sample': sorted(rows[0].keys()) if rows else []}


def survey_cell(group, dataset, cell_dir):
    score_mode, warmup, use_fm, cfg_src = cell_config(cell_dir)
    npz_dir = os.path.join(cell_dir, 'epoch_scores')
    npzs = sorted(glob.glob(os.path.join(npz_dir, 'epoch_*_scores.npz')))
    npz_epochs = []
    for p in npzs:
        try:
            npz_epochs.append(int(os.path.basename(p).split('_')[1]))
        except Exception:
            pass
    npz_epochs.sort()
    w = warmup if isinstance(warmup, int) else -1
    pre_warmup_epochs = [e for e in npz_epochs if w > 0 and e <= w]

    em_info = epoch_metrics_info(cell_dir)
    meta = load_json(os.path.join(cell_dir, 'experiment_metadata.json'))
    stored_best_epoch = None
    best_metric_key = None
    if meta:
        stored_best_epoch = (meta.get('best_epoch')
                             or (meta.get('timing') or {}).get('best_epoch'))
        best_metric_key = (meta.get('best_epoch_metric')
                           or (meta.get('timing') or {}).get('best_epoch_metric'))

    # checkpoint epoch
    ckpt_epoch = None
    ckpt_path = os.path.join(cell_dir, 'checkpoints', 'best_checkpoint.pt')
    has_ckpt = os.path.exists(ckpt_path)

    # Sample one pre-warmup npz: does adaptive_score already == teacher_recon_error?
    sample_state = None
    if pre_warmup_epochs:
        sp = os.path.join(npz_dir, f'epoch_{pre_warmup_epochs[len(pre_warmup_epochs)//2]:03d}_scores.npz')
        try:
            d = np.load(sp)
            a = d['adaptive_score']; t = d['teacher_recon_error']
            if np.array_equal(a, t):
                sample_state = 'already_recon_only'
            else:
                md = float(np.max(np.abs(a.astype(np.float64) - t.astype(np.float64))))
                sample_state = f'differs(maxabs={md:.3e})'
        except Exception as e:
            sample_state = f'err:{e}'

    in_scope = (score_mode == 'adaptive' and w > 0 and len(pre_warmup_epochs) > 0)

    return {
        'group': group, 'dataset': dataset, 'cell_dir': cell_dir,
        'config_source': cfg_src,
        'anomaly_score_mode': score_mode,
        'teacher_only_warmup_epochs': warmup,
        'use_feature_matching': use_fm,
        'n_npz': len(npz_epochs),
        'npz_epoch_min': npz_epochs[0] if npz_epochs else None,
        'npz_epoch_max': npz_epochs[-1] if npz_epochs else None,
        'n_pre_warmup_npz': len(pre_warmup_epochs),
        'pre_warmup_epochs': pre_warmup_epochs,
        'epoch_metrics': em_info,
        'stored_best_epoch': stored_best_epoch,
        'best_epoch_metric': best_metric_key,
        'best_epoch_is_prewarmup': (isinstance(stored_best_epoch, int)
                                    and w > 0 and stored_best_epoch <= w),
        'has_best_checkpoint': has_ckpt,
        'epoch_scores_is_symlink': os.path.islink(npz_dir),
        'sample_prewarmup_state': sample_state,
        'in_scope': in_scope,
    }


def main():
    cells = []
    for group in sorted(os.listdir(ROOT)):
        gdir = os.path.join(ROOT, group)
        if not os.path.isdir(gdir):
            continue
        for npz_dir in glob.glob(os.path.join(gdir, '**', 'epoch_scores'), recursive=True):
            cell_dir = os.path.dirname(npz_dir)
            dataset = os.path.relpath(cell_dir, gdir)
            cells.append(survey_cell(group, dataset, cell_dir))

    in_scope = [c for c in cells if c['in_scope']]
    out = {
        'root': ROOT,
        'n_cells_total': len(cells),
        'n_in_scope': len(in_scope),
        'cells': cells,
    }
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)

    # Console summary
    print(f"Total cells with epoch_scores: {len(cells)}")
    print(f"IN-SCOPE (adaptive + warmup>0 + pre-warmup npz): {len(in_scope)}\n")
    print(f"{'GROUP':<42} {'DATASET':<14} {'mode':<9} {'wu':>4} {'fm':>5} "
          f"{'npz':>4} {'pre':>4} {'evals':>5} {'best_ep':>7} {'bp<=wu':>6} {'sample_prewarmup'}")
    for c in cells:
        em = c['epoch_metrics'] or {}
        mark = '' if c['in_scope'] else '  (out)'
        print(f"{c['group']:<42} {c['dataset']:<14} "
              f"{str(c['anomaly_score_mode']):<9} {str(c['teacher_only_warmup_epochs']):>4} "
              f"{str(c['use_feature_matching']):>5} {c['n_npz']:>4} {c['n_pre_warmup_npz']:>4} "
              f"{em.get('n_rows','?'):>5} {str(c['stored_best_epoch']):>7} "
              f"{str(c['best_epoch_is_prewarmup']):>6} {str(c['sample_prewarmup_state'])}{mark}")
    print(f"\nManifest written: {OUT}")


if __name__ == '__main__':
    main()
