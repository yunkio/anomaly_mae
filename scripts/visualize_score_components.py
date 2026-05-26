"""
Visualize anomaly score components on the actual scale used by the adaptive scoring formula.

Adaptive scoring formula (evaluator.py:1053-1107):
    scaled_disc = disc * (recon.mean() / disc.mean())
    scaled_fm   = fm   * (recon.mean() / fm.mean())     # only when use_feature_matching
    student_error = (w_disc * scaled_disc + w_fm * scaled_fm) / (w_disc + w_fm)
    anomaly_score = recon + student_error

The npz at {exp}/{dataset}/epoch_scores/epoch_XXX_scores.npz stores raw values:
    adaptive_score, teacher_recon_error, discrepancy_error, (fm_error if saved)

Visualization plots SCALED values (the actual contribution to anomaly_score):
    Subplot 1: anomaly_score + threshold
    Subplot 2: teacher_recon_error (raw == scaled, weight=1)
    Subplot 3: scaled_disc = disc * (recon.mean()/disc.mean()) * w_disc/(w_disc+w_fm)
    Subplot 4: scaled_fm  = fm  * (recon.mean()/fm.mean())  * w_fm/(w_disc+w_fm)   [only if fm available]
"""

import json
import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '/home/ykio/notebooks/claude')

from mae_anomaly.datasets.loaders import (
    load_swat_combined,
    load_simulation,
    _load_raw_csv,
)


def get_best_epoch_and_threshold(metrics_path):
    data = json.load(open(metrics_path))
    best = max(data['epochs'], key=lambda e: e.get('pak_auc_f1', 0))
    return (best['epoch'], best.get('optimal_threshold', None),
            best.get('pak_auc_f1', None), best.get('pak_auc_prc_auc', None))


def load_test_point_labels(name):
    """Return point_labels for test portion of dataset (matches 274/271 etc. evaluation slice)."""
    if name == 'SWaT_full':
        out = load_swat_combined()
        signals, point_labels, anomaly_regions, feature_names, train_ratio = out[:5]
        train_end = int(len(point_labels) * train_ratio)
        return point_labels[train_end:]
    elif name == 'WaDi_A1':
        path = '/home/ykio/notebooks/claude/dataset/WaDi/WADI.A1_9 Oct 2017/WADI_A1_attack_raw.csv'
        _, attack_labels, _ = _load_raw_csv(path)
        mid = len(attack_labels) // 2
        return attack_labels[mid:]
    elif name == 'WaDi_A2':
        path = '/home/ykio/notebooks/claude/dataset/WaDi/WADI.A2_19 Nov 2019/WADI_A2_attack_raw.csv'
        _, attack_labels, _ = _load_raw_csv(path)
        mid = len(attack_labels) // 2
        return attack_labels[mid:]
    elif name == 'simulation':
        out = load_simulation()
        signals, point_labels, anomaly_regions, feature_names, train_ratio = out[:5]
        train_end = int(len(point_labels) * train_ratio)
        return point_labels[train_end:]
    else:
        raise ValueError(name)


def extract_anomaly_regions(point_labels):
    regions = []
    in_anom = False
    start = 0
    for i, lbl in enumerate(point_labels):
        if lbl == 1 and not in_anom:
            start = i
            in_anom = True
        elif lbl == 0 and in_anom:
            regions.append((start, i))
            in_anom = False
    if in_anom:
        regions.append((start, len(point_labels)))
    return regions


def _compute_detection_ratios(anomaly_score, threshold, test_regions):
    """For each anomaly region, return ratio = (#points with score >= threshold) / region_len."""
    ratios = []
    for s, e in test_regions:
        seg = anomaly_score[s:e]
        if len(seg) == 0:
            ratios.append(0.0)
        else:
            ratios.append(float((seg >= threshold).sum()) / float(len(seg)))
    return ratios


def _layout_label_positions(centers, label_width, x_min, x_max):
    """Greedy 1-D non-overlap layout. Returns x position for each label so that
    boxes of width=label_width never overlap. Tries to stay near the desired
    center; pushes right if collision; final reverse pass to respect x_max.
    """
    n = len(centers)
    if n == 0:
        return []
    # Sort by desired center to process left → right
    order = sorted(range(n), key=lambda i: centers[i])
    positions = [None] * n
    cur = x_min - label_width  # so first can be at x_min
    for idx in order:
        desired = centers[idx]
        new_pos = max(desired, cur + label_width)
        positions[idx] = new_pos
        cur = new_pos
    # Reverse pass: if last exceeds x_max, push left
    if positions[order[-1]] > x_max:
        cur = x_max + label_width
        for idx in reversed(order):
            positions[idx] = min(positions[idx], cur - label_width)
            cur = positions[idx]
    return positions


def plot_anomaly_threshold(out_path, name, exp_id, best_epoch, pak_auc_f1, pak_auc_prc, threshold,
                           anomaly_score, recon, disc, scaled_disc, scaled_fm,
                           test_regions, x_max, w_disc, w_fm, fm_in_score):
    """Generate the standard anomaly_threshold visualization with detection-ratio
    annotations on each anomaly region (anomaly_score subplot only).

    Design choices for readability:
      - Region opacity proportional to detection ratio (faint → low, vivid → high)
      - Numeric label "0.80" near the top of each region (inside the subplot)
        rotated 90° if region is narrow
      - recon/scaled_disc/scaled_fm subplots keep simple uniform shading to avoid clutter
    """
    # Detection ratios per region (only meaningful on anomaly_score)
    if threshold is not None:
        det_ratios = _compute_detection_ratios(anomaly_score, threshold, test_regions)
    else:
        det_ratios = [0.0] * len(test_regions)

    n_subplots = 4 if fm_in_score else 3
    fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 3 * n_subplots), sharex=True)
    if n_subplots == 1:
        axes = [axes]
    x = np.arange(x_max)

    panels = [
        ('anomaly_score (= recon + scaled_disc' + (' + scaled_fm' if fm_in_score else '') + ')',
         anomaly_score, 'black', threshold),
        ('recon (weight=1)', recon, 'tab:blue', None),
        (f'scaled_disc = disc × (recon.mean/disc.mean) × w_disc/W   [w_disc={w_disc:.2f}]',
         scaled_disc, 'tab:green', None),
    ]
    if fm_in_score:
        panels.append(
            (f'scaled_fm = fm × (recon.mean/fm.mean) × w_fm/W   [w_fm={w_fm:.2f}]',
             scaled_fm, 'tab:orange', None)
        )

    for idx_ax, (ax, (label, series, color, thr)) in enumerate(zip(axes, panels)):
        is_score_panel = (idx_ax == 0)
        # Shade anomaly regions
        for ri, (s, e) in enumerate(test_regions):
            if is_score_panel and threshold is not None:
                alpha = 0.15 + 0.50 * det_ratios[ri]
            else:
                alpha = 0.25
            ax.axvspan(s, e, alpha=alpha, color='red', zorder=1)
        # Plot series
        ax.plot(x, series, color=color, linewidth=0.7, alpha=0.9, zorder=3)
        if thr is not None:
            ax.axhline(thr, color='black', linestyle='--', linewidth=1.0, alpha=0.7,
                       label=f'threshold={thr:.4f}')

        # Detection-ratio annotations — anomaly_score subplot only
        if is_score_panel and threshold is not None and test_regions:
            n = len(test_regions)
            # Auto-shrink font + label-width when many regions
            if n <= 30:
                fontsize = 8; label_w_frac = 0.025
            elif n <= 60:
                fontsize = 7; label_w_frac = 0.018
            else:
                fontsize = 6; label_w_frac = 0.014
            label_width = x_max * label_w_frac
            # Required total width if single row → number of rows needed
            rows_needed = max(2, int(np.ceil(n * label_width / (x_max * 0.9))))
            rows_needed = min(rows_needed, 6)  # cap rows

            # Reserve top margin proportional to row count
            y_min, y_max = ax.get_ylim()
            margin = (y_max - y_min) * (0.18 + 0.06 * rows_needed)
            new_top = y_max + margin
            ax.set_ylim(top=new_top)

            # Distribute rows evenly inside top margin
            row_ys = [new_top - margin * (0.12 + 0.78 * (r + 0.5) / rows_needed)
                      for r in range(rows_needed)]

            # Assign each region to a row (round-robin), then layout within row
            centers = [(s + e) / 2.0 for s, e in test_regions]
            row_indices = {r: [] for r in range(rows_needed)}
            for i in range(n):
                row_indices[i % rows_needed].append(i)
            label_x_by_idx = {}
            for r, idxs in row_indices.items():
                pos = _layout_label_positions(
                    [centers[i] for i in idxs], label_width, 0, x_max)
                for k, i in enumerate(idxs):
                    label_x_by_idx[i] = pos[k]

            for ri, ((s, e), ratio) in enumerate(zip(test_regions, det_ratios)):
                cx = centers[ri]
                lx = label_x_by_idx[ri]
                ly = row_ys[ri % rows_needed]
                # Leader line: from top of data area to label
                ax.plot([cx, lx], [y_max, ly - margin * 0.04],
                        color='gray', linewidth=0.35, alpha=0.5, zorder=8)
                ax.text(lx, ly, f'{ratio:.2f}',
                        ha='center', va='center',
                        fontsize=fontsize, color='black', fontweight='normal',
                        zorder=10,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='lightgray', alpha=0.95, linewidth=0.4))
            # Legend
            anom_patch = mpatches.Patch(color='red', alpha=0.4,
                                        label='anomaly (opacity ∝ detection ratio)')
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles=[anom_patch] + handles, loc='upper left',
                      fontsize=9, framealpha=0.9)

        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Test point index')
    axes[0].set_xlim(0, x_max)
    prc_str = f', pak_auc_prc={pak_auc_prc:.4f}' if pak_auc_prc is not None else ''
    fig.suptitle(
        f'Anomaly Threshold — {name} ({exp_id}, best_epoch={best_epoch}, '
        f'pak_auc_f1={pak_auc_f1:.4f}{prc_str})', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()


def compute_scaled_components_ema(scores_npz, w_disc, w_fm, use_fm, alpha=0.01):
    """OPTION 3: Exponential moving average — recent values weighted more.

    Updates means via:
        ema_recon_mean(t) = alpha * recon(t) + (1-alpha) * ema_recon_mean(t-1)
    Smaller alpha → smoother / slower adaptation.
    Initialization: ema = first value (warm start).
    """
    recon = scores_npz['teacher_recon_error'].astype(np.float64)
    disc = scores_npz['discrepancy_error'].astype(np.float64)
    fm = scores_npz['fm_error'].astype(np.float64) if (
        use_fm and 'fm_error' in scores_npz.files) else None

    def _ema(arr, alpha):
        out = np.zeros_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out + 1e-4

    ema_recon = _ema(recon, alpha)
    ema_disc = _ema(disc, alpha)
    if fm is not None:
        ema_fm = _ema(fm, alpha)
        denom = max(w_disc + w_fm, 1e-6)
        scaled_disc = disc * (ema_recon / ema_disc) * (w_disc / denom)
        scaled_fm = fm * (ema_recon / ema_fm) * (w_fm / denom)
        anomaly_score = recon + scaled_disc + scaled_fm
    else:
        scaled_disc = disc * (ema_recon / ema_disc)
        scaled_fm = None
        anomaly_score = recon + scaled_disc

    return (recon.astype(np.float32), disc.astype(np.float32),
            fm.astype(np.float32) if fm is not None else None,
            scaled_disc.astype(np.float32),
            (scaled_fm.astype(np.float32) if scaled_fm is not None else None),
            anomaly_score.astype(np.float32))


def compute_scaled_components_streaming(scores_npz, w_disc, w_fm, use_fm):
    """OPTION 2: Streaming (cumulative running mean) — scale factor updated per timestep.

    For each timestep t:
        running_recon_mean(t) = mean(recon[0..t])
        running_disc_mean(t)  = mean(disc[0..t])
        scaled_disc(t) = disc(t) * (running_recon_mean(t) / running_disc_mean(t)) * w_disc/(w_disc+w_fm)
        anomaly_score(t) = recon(t) + scaled_disc(t) + scaled_fm(t)

    Note: 'cold start' on early timesteps — running mean unreliable until n is large.
    """
    recon = scores_npz['teacher_recon_error']
    disc = scores_npz['discrepancy_error']
    fm = scores_npz['fm_error'] if (use_fm and 'fm_error' in scores_npz.files) else None

    # Cumulative running means
    n = np.arange(1, len(recon) + 1, dtype=np.float64)
    running_recon_mean = np.cumsum(recon, dtype=np.float64) / n + 1e-4
    running_disc_mean = np.cumsum(disc, dtype=np.float64) / n + 1e-4

    if fm is not None:
        running_fm_mean = np.cumsum(fm, dtype=np.float64) / n + 1e-4
        denom = max(w_disc + w_fm, 1e-6)
        scaled_disc = disc * (running_recon_mean / running_disc_mean) * (w_disc / denom)
        scaled_fm = fm * (running_recon_mean / running_fm_mean) * (w_fm / denom)
        anomaly_score = recon + scaled_disc + scaled_fm
    else:
        scaled_disc = disc * (running_recon_mean / running_disc_mean)
        scaled_fm = None
        anomaly_score = recon + scaled_disc

    return (recon, disc, fm, scaled_disc.astype(np.float32),
            (scaled_fm.astype(np.float32) if scaled_fm is not None else None),
            anomaly_score.astype(np.float32))


def compute_scaled_components(scores_npz, w_disc, w_fm, use_fm):
    """Compute the scaled components going into anomaly_score from raw point-level arrays.

    Uses point-level means for scaling (boundary effect is < 0.5% vs patch-level mean used
    by evaluator — negligible in practice). Matches the structure of evaluator's adaptive
    scoring formula.
    """
    recon = scores_npz['teacher_recon_error']
    disc = scores_npz['discrepancy_error']
    fm = scores_npz['fm_error'] if (use_fm and 'fm_error' in scores_npz.files) else None

    recon_mean = float(recon.mean()) + 1e-4
    disc_mean = float(disc.mean()) + 1e-4

    if fm is not None:
        fm_mean = float(fm.mean()) + 1e-4
        denom = max(w_disc + w_fm, 1e-6)
        scaled_disc = disc * (recon_mean / disc_mean) * (w_disc / denom)
        scaled_fm = fm * (recon_mean / fm_mean) * (w_fm / denom)
    else:
        scaled_disc = disc * (recon_mean / disc_mean)
        scaled_fm = None

    return recon, disc, fm, scaled_disc, scaled_fm


def _compute_anomaly_regions(labels):
    """Return list of (start, end) tuples (end exclusive)."""
    regions = []
    in_anom = False
    start = 0
    for i, lbl in enumerate(labels):
        if lbl == 1 and not in_anom:
            start = i; in_anom = True
        elif lbl == 0 and in_anom:
            regions.append((start, i)); in_anom = False
    if in_anom:
        regions.append((start, len(labels)))
    return regions


def _pa_k_adjust(predictions, regions, k_percent):
    """Apply PA%K segment adjustment. If at least k% of a region is detected,
    set all points in that region to 1. Returns adjusted predictions."""
    adj = predictions.copy()
    if not regions:
        return adj
    for s, e in regions:
        seg = adj[s:e]
        if len(seg) == 0:
            continue
        detected_ratio = float(seg.sum()) / float(len(seg))
        if detected_ratio * 100.0 >= k_percent:
            adj[s:e] = 1
    return adj


def _eval_metrics(score, labels):
    """Compute ROC/PR AUC + best F1 + threshold + PA%K AUC.
    PA%K AUC = mean of F1/PR/ROC over K in [0,100] step 5 with re-optimized threshold."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 precision_recall_curve, roc_curve)
    if len(np.unique(labels)) < 2:
        return {'roc': float('nan'), 'pr': float('nan'),
                'best_f1': float('nan'), 'threshold': float('nan'),
                'pak_auc_f1': float('nan'), 'pak_auc_prc': float('nan')}
    roc = roc_auc_score(labels, score)
    pr = average_precision_score(labels, score)
    # Best F1 + threshold (no PA)
    p, r, thresh = precision_recall_curve(labels, score)
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1), 0)
    best_idx = f1.argmax()
    best_f1 = float(f1[best_idx])
    # precision_recall_curve returns thresholds shape = (n-1,)
    best_thr = float(thresh[min(best_idx, len(thresh) - 1)])

    # PA%K AUC — sweep K=0..100 step 5; per K: best F1 over thresholds + step-AP PRC.
    # NOTE: this used to use (prec * rec) as a "rough proxy" for PRC-AUC which is
    # mathematically wrong. Now uses proper step-AP (sklearn average_precision_score
    # equivalent) for PRC-AUC and best-F1 over threshold sweep.
    regions = _compute_anomaly_regions(labels)
    k_grid = np.arange(0, 101, 5)
    n_thr = 100  # threshold sweep size (matches eval-time density)
    thr_grid = np.linspace(score.min() - 1e-6, score.max() + 1e-6, n_thr)
    # Sort thresholds descending so recall ascends in the resulting sweep
    thr_grid_desc = np.sort(thr_grid)[::-1]
    f1_by_k = []
    prc_by_k = []
    for k in k_grid:
        # Per-threshold (TP, FP, FN) after PA%K adjustment
        precs = np.zeros(n_thr); recs = np.zeros(n_thr); f1s = np.zeros(n_thr)
        for ti, t in enumerate(thr_grid_desc):
            preds = (score >= t).astype(np.int32)
            adj = _pa_k_adjust(preds, regions, k)
            tp = float(((adj == 1) & (labels == 1)).sum())
            fp = float(((adj == 1) & (labels == 0)).sum())
            fn = float(((adj == 0) & (labels == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precs[ti] = p; recs[ti] = r
            f1s[ti] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        # Step-AP: prepend (rec=0, prec=1); AP = Σ (R_n − R_{n−1}) * P_n
        rec_step = np.concatenate([[0.0], recs])
        prec_step = np.concatenate([[1.0], precs])
        ap = float(np.sum(np.diff(rec_step) * prec_step[1:]))
        ap = max(0.0, min(1.0, ap))
        f1_by_k.append(float(f1s.max()))
        prc_by_k.append(ap)
    # Uniform K-grid mean (== trapz / 100 over K∈{0,5,...,100} with normalization)
    pak_auc_f1 = float(np.mean(f1_by_k))
    pak_auc_prc = float(np.mean(prc_by_k))

    return {'roc': float(roc), 'pr': float(pr),
            'best_f1': best_f1, 'threshold': best_thr,
            'pak_auc_f1': pak_auc_f1, 'pak_auc_prc': pak_auc_prc}


def visualize_one(exp_dir, dataset_sub, ds_name, output_path, mode='default',
                   compare_metrics=True, ema_alpha=0.01):
    """Generate anomaly_threshold.png for one (experiment, dataset).

    Args:
        mode: 'default' / 'streaming' / 'ema'
        compare_metrics: If True, print metrics for all 3 modes
        ema_alpha: EMA decay (smaller=smoother)
    """
    metrics_path = os.path.join(exp_dir, dataset_sub, 'epoch_metrics.json')
    best_epoch, threshold, pak_auc_f1, pak_auc_prc = get_best_epoch_and_threshold(metrics_path)

    # Load config to know weights + fm usage
    cfg_path = os.path.join(exp_dir, dataset_sub, 'best_config.json')
    cfg = json.load(open(cfg_path))
    use_fm = bool(cfg.get('use_feature_matching', False))
    w_disc = float(cfg.get('eval_disc_weight', -1))
    w_fm = float(cfg.get('eval_fm_weight', -1))
    if w_disc < 0:
        w_disc = 1.0
    if w_fm < 0:
        w_fm = float(cfg.get('fm_loss_weight', 1.0))

    # Load scores npz at best epoch
    npz_path = os.path.join(exp_dir, dataset_sub,
                             f'epoch_scores/epoch_{best_epoch:03d}_scores.npz')
    if not os.path.exists(npz_path):
        candidates = sorted(glob.glob(os.path.join(exp_dir, dataset_sub,
                                                     'epoch_scores/epoch_*_scores.npz')))
        if not candidates:
            print(f"  [{ds_name}] no epoch_scores npz — skip")
            return False
        npz_path = candidates[-1]
    scores = np.load(npz_path)

    if mode == 'streaming':
        recon, disc, fm, scaled_disc, scaled_fm, anomaly_score = \
            compute_scaled_components_streaming(scores, w_disc, w_fm, use_fm)
    elif mode == 'ema':
        recon, disc, fm, scaled_disc, scaled_fm, anomaly_score = \
            compute_scaled_components_ema(scores, w_disc, w_fm, use_fm, alpha=ema_alpha)
    elif mode == 'option4':
        # OPTION 4: EMA applied directly to the final adaptive_score (post-hoc smoothing).
        # Matches the C10 transform used in temp/eval_C10*.py.
        anomaly_score_raw = scores['adaptive_score'].astype(np.float64)
        out = np.empty_like(anomaly_score_raw)
        out[0] = anomaly_score_raw[0]
        a = float(ema_alpha)
        for i in range(1, len(anomaly_score_raw)):
            out[i] = a * anomaly_score_raw[i] + (1.0 - a) * out[i - 1]
        anomaly_score = out.astype(np.float32)
        recon, disc, fm, scaled_disc, scaled_fm = compute_scaled_components(
            scores, w_disc, w_fm, use_fm)
    elif mode == 'option5':
        # OPTION 5: two-sided Gaussian smoothing on the adaptive_score (== B2 from
        # the post-hoc transform comparison; best mean pak_auc_f1 in that sweep).
        # sigma is passed via ema_alpha argument (re-used to avoid extra plumbing).
        from scipy.ndimage import gaussian_filter1d
        sigma = float(ema_alpha)
        adaptive_raw = scores['adaptive_score'].astype(np.float64)
        anomaly_score = gaussian_filter1d(adaptive_raw, sigma=sigma, mode='reflect').astype(np.float32)
        recon, disc, fm, scaled_disc, scaled_fm = compute_scaled_components(
            scores, w_disc, w_fm, use_fm)
    elif mode == 'option6':
        # OPTION 6: image-based pipeline with user-modified parameters.
        #   alpha_recon = 0.5, alpha_disc = 0.5  (balanced)
        #   c = 50  (permissive clip)
        #   K = 19  (matches EMA alpha=0.1 effective span)
        #   stats = full test (not normal-only)
        a_r, a_d, c_clip, K_ma, eps_floor = 0.5, 0.5, 50.0, 19, 1e-4
        recon_arr = scores['teacher_recon_error'].astype(np.float64)
        disc_arr = scores['discrepancy_error'].astype(np.float64)

        def _robust_stats_all(s):
            mu = float(np.median(s))
            mad = float(np.median(np.abs(s - mu)))
            q25, q75 = np.quantile(s, [0.25, 0.75])
            iqr_sc = float((q75 - q25) / 1.349)
            std = float(np.std(s))
            sigma = max(1.4826 * mad, iqr_sc, std, eps_floor, 1e-12)
            return mu, sigma

        mu_r, sigma_r = _robust_stats_all(recon_arr)
        mu_d, sigma_d = _robust_stats_all(disc_arr)
        z_r = np.clip((recon_arr - mu_r) / sigma_r, -c_clip, c_clip)
        z_d = np.clip((disc_arr - mu_d) / sigma_d, -c_clip, c_clip)
        s_eval = a_r * z_r + a_d * z_d
        import pandas as _pd
        anomaly_score = _pd.Series(s_eval).rolling(K_ma, min_periods=1).mean().to_numpy().astype(np.float32)
        recon, disc, fm, scaled_disc, scaled_fm = compute_scaled_components(
            scores, w_disc, w_fm, use_fm)
    else:  # default
        anomaly_score = scores['adaptive_score']
        recon, disc, fm, scaled_disc, scaled_fm = compute_scaled_components(
            scores, w_disc, w_fm, use_fm)
    fm_in_score = use_fm and fm is not None

    # Test point labels
    point_labels = load_test_point_labels(ds_name)
    m = min(len(anomaly_score), len(point_labels))
    anomaly_score = anomaly_score[:m]
    recon = recon[:m]
    scaled_disc = scaled_disc[:m]
    if scaled_fm is not None:
        scaled_fm = scaled_fm[:m]
    point_labels = point_labels[:m]
    test_regions = extract_anomaly_regions(point_labels)

    exp_id = os.path.basename(exp_dir.rstrip('/')).split('_')[0]
    title_suffix_map = {
        'default': '',
        'streaming': ' [OPTION 2: streaming mean]',
        'ema': f' [OPTION 3: EMA alpha={ema_alpha}]',
        'option4': f' [OPTION 4: EMA on adaptive_score alpha={ema_alpha}]',
        'option5': f' [OPTION 5: two-sided Gaussian smoothing sigma={ema_alpha}]',
        'option6': ' [OPTION 6: robust-z(all-data) + clip(c=50) + 0.5*z_r + 0.5*z_d + SMA(K=19)]',
    }
    title_suffix = title_suffix_map.get(mode, '')

    # Use the per-mode anomaly_score and re-derived threshold
    m_self = _eval_metrics(anomaly_score, point_labels)
    plot_threshold = m_self['threshold']  # recomputed threshold for this mode

    print(f"  [{ds_name} / {mode}] best_ep={best_epoch} "
          f"ROC={m_self['roc']:.4f} PR={m_self['pr']:.4f} "
          f"best_F1={m_self['best_f1']:.4f} thr={plot_threshold:.5f} "
          f"PAK_F1={m_self['pak_auc_f1']:.4f} PAK_PRC={m_self['pak_auc_prc']:.4f}")

    plot_anomaly_threshold(
        out_path=output_path,
        name=ds_name + title_suffix, exp_id=f'exp{exp_id}',
        best_epoch=best_epoch,
        pak_auc_f1=m_self['pak_auc_f1'], pak_auc_prc=m_self['pak_auc_prc'],
        threshold=plot_threshold,
        anomaly_score=anomaly_score, recon=recon, disc=disc,
        scaled_disc=scaled_disc, scaled_fm=scaled_fm,
        test_regions=test_regions, x_max=m,
        w_disc=w_disc, w_fm=w_fm, fm_in_score=fm_in_score,
    )
    print(f"  [{ds_name} / {mode}] saved {output_path}")
    return m_self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', type=int, default=None,
                    help='Experiment number (default: best RankAvg available)')
    ap.add_argument('--out-dir', default='/home/ykio/notebooks/claude/temp')
    ap.add_argument('--option4-alpha', type=float, default=None,
                    help='If set, additionally run OPTION 4 (EMA on adaptive_score) with this alpha.')
    ap.add_argument('--option5-sigma', type=float, default=None,
                    help='If set, additionally run OPTION 5 (two-sided Gaussian smoothing of adaptive_score) with this sigma.')
    ap.add_argument('--option5-suffix', type=str, default='_option5',
                    help='Filename suffix for OPTION 5 outputs (default _option5; use _option5_best for sweep-best plot).')
    ap.add_argument('--option6', action='store_true',
                    help='If set, additionally run OPTION 6 (image pipeline: 0.5*z_r+0.5*z_d, c=50, K=19, all-data stats).')
    args = ap.parse_args()

    if args.exp is None:
        # Pick the highest-RankAvg available — fallback chain: 274 → 271 → 273 → 269
        for candidate in [274, 271, 273, 269, 265]:
            dirs = sorted(glob.glob(
                f'/home/ykio/notebooks/claude/results/experiments/{candidate}_*/'))
            if dirs:
                args.exp = candidate
                exp_dir = dirs[0]
                break
        else:
            raise RuntimeError("No suitable experiment found")
    else:
        dirs = sorted(glob.glob(
            f'/home/ykio/notebooks/claude/results/experiments/{args.exp}_*/'))
        if not dirs:
            raise RuntimeError(f"exp{args.exp} not found")
        exp_dir = dirs[0]

    print(f"Using experiment: exp{args.exp}  ({exp_dir})")
    os.makedirs(args.out_dir, exist_ok=True)

    targets = [
        ('SWaT_full', 'SWaT/A1A2_full', 'score_components_swat_full.png'),
        ('WaDi_A1', 'WaDi/A1', 'score_components_wadi_a1.png'),
        ('WaDi_A2', 'WaDi/A2', 'score_components_wadi_a2.png'),
        ('simulation', 'simulation/simulation', 'score_components_simulation.png'),
    ]
    all_metrics = {}  # {(mode, ds_name): metrics}
    modes_runs = [
        ('default', '', 'DEFAULT (test-set mean) scoring', 0.01),
        ('streaming', '_option2', 'OPTION 2 (streaming/cumulative mean) scoring', 0.01),
        ('ema', '_option3', 'OPTION 3 (EMA on scaling factors, alpha=0.01) scoring', 0.01),
    ]
    if args.option4_alpha is not None:
        modes_runs.append(
            ('option4', '_option4',
             f'OPTION 4 (EMA on adaptive_score, alpha={args.option4_alpha}) scoring',
             float(args.option4_alpha))
        )
    if args.option5_sigma is not None:
        modes_runs.append(
            ('option5', args.option5_suffix,
             f'OPTION 5 (two-sided Gaussian smoothing, sigma={args.option5_sigma}) scoring',
             float(args.option5_sigma))
        )
    if args.option6:
        modes_runs.append(
            ('option6', '_option6',
             'OPTION 6 (image pipeline: 0.5*z_r+0.5*z_d, c=50, K=19, all-data stats) scoring',
             0.0)
        )
    for mode, suffix, banner, mode_alpha in modes_runs:
        print(f"\n{'='*70}\n{banner}\n{'='*70}")
        for name, sub, fname in targets:
            try:
                fname2 = fname if suffix == '' else fname.replace('.png', f'{suffix}.png')
                metrics = visualize_one(
                    exp_dir, sub, name,
                    os.path.join(args.out_dir, fname2),
                    mode=mode, ema_alpha=mode_alpha)
                if metrics is not None and not isinstance(metrics, bool):
                    all_metrics[(mode, name)] = metrics
            except Exception as e:
                print(f"  [{name}] ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary comparison table
    print(f"\n{'='*100}\nSUMMARY: 3-mode metric comparison\n{'='*100}")
    print(f"{'Dataset':<14} {'Mode':<11} {'ROC':>7} {'PR':>7} {'best_F1':>9} "
          f"{'thr':>10} {'PAK_F1':>8} {'PAK_PRC':>8}")
    print('-' * 100)
    for name, _, _ in targets:
        for mode, _, _, _ in modes_runs:
            m = all_metrics.get((mode, name))
            if m is None:
                continue
            print(f"{name:<14} {mode:<11} {m['roc']:>7.4f} {m['pr']:>7.4f} "
                  f"{m['best_f1']:>9.4f} {m['threshold']:>10.5f} "
                  f"{m['pak_auc_f1']:>8.4f} {m['pak_auc_prc']:>8.4f}")
        print()


if __name__ == '__main__':
    main()
