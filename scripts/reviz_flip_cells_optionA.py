#!/usr/bin/env python
"""Option A re-viz for the 5 flipped cells (2026-06-01 pre-warmup backfill).

For each cell whose corrected best_epoch differs from the saved best_checkpoint.pt
epoch, the best-model PNGs reflect the OLD epoch. We CANNOT re-run the model at
the new epoch (no per-epoch weights), but the new epoch's npz holds the corrected
point-level score (recon-only if the new best is pre-warmup, else the original
adaptive score) + labels. So we regenerate the SCORE-BASED best-model plots
directly from the npz:
    best_model_prc_curve.png, best_model_roc_curve.png,
    best_model_confusion_matrix.png, score_distribution_by_label.png,
    anomaly_threshold.png
and drop a STALE_VIZ.txt explaining that the model-forward signal plots
(reconstruction, detection examples, case study, hardest samples, feature
profiles) still reflect the OLD epoch (the user deprioritised these).

Old PNGs are backed up under .trash/0531/backfill_viz_backups/<cell>/best_model/.
READ-ONLY w.r.t. npz / metrics; only writes PNGs + marker into the cell viz dir.
"""
import os, sys, json, glob, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix)

sys.path.insert(0, '/home/ykio/notebooks/TSMAE')
from mae_anomaly.evaluator import find_f1_optimal_idx

APPLY_REPORT = '/tmp/prewarmup_backfill/backfill_report.json'
VIZ_BACKUP = '/home/ykio/notebooks/TSMAE/.trash/0531/backfill_viz_backups'
WARMUP = 250
STALE_SIGNAL_PLOTS = ['best_model_reconstruction.png', 'best_model_detection_examples.png',
                      'case_study_gallery.png', 'hardest_samples.png', 'feature_profile.png',
                      'feature_extremes.png', 'feature_dominance.png']


def load_json(p):
    with open(p) as f:
        return json.load(f)


def best_npz_score(cell_dir, ep):
    """Return (point_scores, point_labels) at the new best epoch. recon-only if
    pre-warmup else original adaptive_score (matches the corrected metric)."""
    p = os.path.join(cell_dir, 'epoch_scores', f'epoch_{ep:03d}_scores.npz')
    d = np.load(p)
    score = d['teacher_recon_error'] if ep <= WARMUP else d['adaptive_score']
    return np.asarray(score, float), np.asarray(d['point_labels']).astype(int)


def backup_png(png, cell_rel):
    dst = os.path.join(VIZ_BACKUP, cell_rel, 'best_model', os.path.basename(png))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(png, dst)


def regen(cell_rel, cell_dir, new_ep, old_ep, flipped=True):
    s, y = best_npz_score(cell_dir, new_ep)
    ml = min(len(s), len(y)); s = s[:ml]; y = y[:ml]
    bm = os.path.join(cell_dir, 'visualization', 'best_model')
    os.makedirs(bm, exist_ok=True)
    title_suffix = f"(corrected best ep{new_ep}{' pre-warmup recon-only' if new_ep <= WARMUP else ''})"

    # threshold (F1-optimal, same rule as metric pipeline)
    fpr, tpr, thr = roc_curve(y, s)
    oi = find_f1_optimal_idx(fpr, tpr, y)
    threshold = thr[oi]
    pred = (s > threshold).astype(int)

    def save(name):
        png = os.path.join(bm, name)
        if os.path.exists(png):
            backup_png(png, cell_rel)
        plt.tight_layout(); plt.savefig(png, dpi=150); plt.close()

    # ROC
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#e74c3c', label=f'ROC (AUC={auc(fpr,tpr):.4f})')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC {title_suffix}'); plt.legend(); plt.grid(alpha=.3)
    save('best_model_roc_curve.png')

    # PRC
    prec, rec, _ = precision_recall_curve(y, s)
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, color='#3498db', label=f'PRC (AP={average_precision_score(y,s):.4f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PRC {title_suffix}'); plt.legend(); plt.grid(alpha=.3)
    save('best_model_prc_curve.png')

    # Confusion matrix
    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap='Blues')
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f'{v:,}', ha='center', va='center',
                 color='white' if v > cm.max()/2 else 'black')
    plt.xticks([0, 1], ['Normal', 'Anomaly']); plt.yticks([0, 1], ['Normal', 'Anomaly'])
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion @F1-opt thr {title_suffix}')
    save('best_model_confusion_matrix.png')

    # Score distribution by label
    plt.figure(figsize=(8, 5))
    plt.hist(s[y == 0], bins=80, alpha=.6, label='Normal', color='#2ecc71', density=True)
    plt.hist(s[y == 1], bins=80, alpha=.6, label='Anomaly', color='#e74c3c', density=True)
    plt.axvline(threshold, color='k', ls='--', label=f'F1-opt thr={threshold:.4g}')
    plt.xlabel('Anomaly score'); plt.ylabel('Density'); plt.yscale('log')
    plt.title(f'Score distribution by label {title_suffix}'); plt.legend(); plt.grid(alpha=.3)
    save('score_distribution_by_label.png')

    # Threshold/score timeline (subsampled)
    n = len(s); step = max(1, n // 20000); xs = np.arange(0, n, step)
    plt.figure(figsize=(12, 4))
    plt.plot(xs, s[::step], lw=.5, color='#34495e', label='score')
    anom = np.where(y == 1)[0]
    if len(anom):
        plt.scatter(anom[::max(1, len(anom)//3000)], s[anom[::max(1, len(anom)//3000)]],
                    s=2, color='#e74c3c', label='anomaly', zorder=3)
    plt.axhline(threshold, color='k', ls='--', label=f'thr={threshold:.4g}')
    plt.xlabel('timestep'); plt.ylabel('score'); plt.title(f'Score vs threshold {title_suffix}')
    plt.legend(); plt.grid(alpha=.3)
    save('anomaly_threshold.png')

    # STALE markers on model-forward plots
    stale = []
    for name in STALE_SIGNAL_PLOTS:
        png = os.path.join(bm, name)
        if os.path.exists(png):
            stale.append(name)
    if flipped:
        signal_note = (f"STALE — still reflect OLD ep{old_ep} (model-forward); ep{new_ep} weights "
                       f"do NOT exist on disk → NOT regenerable without retrain. Deprioritised by user.")
    else:
        signal_note = (f"STALE — rendered with the buggy FULL score at ep{new_ep}; best_checkpoint.pt "
                       f"weights for ep{new_ep} DO exist → regenerable via full GPU viz pipeline with "
                       f"force_recon_only=True if needed (deferred).")
    with open(os.path.join(bm, 'STALE_VIZ.txt'), 'w') as f:
        f.write(f"[2026-06-01 pre-warmup recon-only backfill — Option A]\n"
                f"{'best_epoch FLIPPED' if flipped else 'best_epoch unchanged (pre-warmup)'} "
                f"{old_ep} -> {new_ep}.\n"
                f"REGENERATED from ep{new_ep} npz (score-based, correct recon-only):\n"
                f"  best_model_roc_curve / best_model_prc_curve / best_model_confusion_matrix\n"
                f"  / score_distribution_by_label / anomaly_threshold\n"
                f"Signal-reconstruction plots: {signal_note}\n  " + ', '.join(stale) + "\n")
    return {'cell': cell_rel, 'new_ep': new_ep, 'old_ep': old_ep,
            'regenerated': ['roc', 'prc', 'confusion', 'score_dist', 'threshold'], 'stale': stale}


def main():
    rep = load_json(APPLY_REPORT)
    # Option A applies to ALL stale best-model cells: flipped (new best != old)
    # OR pre-warmup best (score now recon-only). Both have a corrected best-epoch
    # npz with the recon-only point score; regenerate the score-based plots from it.
    stale = [r for r in rep['reports']
             if r.get('best_flipped') or r.get('new_best_is_prewarmup')]
    n_flip = sum(1 for r in stale if r.get('best_flipped'))
    print(f"Option-A re-viz for {len(stale)} stale cells "
          f"({n_flip} flipped, {len(stale)-n_flip} pre-warmup-best)\n")
    out = []
    for r in stale:
        cell_rel = r['cell']; cell_dir = r['cell_dir']
        res = regen(cell_rel, cell_dir, r['new_best_epoch'], r['old_best_epoch'],
                    flipped=r.get('best_flipped', False))
        out.append(res)
        tag = 'FLIP' if r.get('best_flipped') else 'pre-best'
        print(f"  [{tag:8s}] {cell_rel:<50} ep{res['old_ep']}->{res['new_ep']}  "
              f"regen={len(res['regenerated'])} plots, stale_signal={len(res['stale'])}")
    with open('/tmp/prewarmup_backfill/flip_reviz_report.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\ndone. backups: {VIZ_BACKUP}")


if __name__ == '__main__':
    main()
