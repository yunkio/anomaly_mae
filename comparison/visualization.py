"""
Visualization utilities for baseline comparison experiments.

Generates:
- epoch_metrics/: Training dynamics plots (epoch_prc_auc, epoch_f1_t, epoch_pa_k_f1, epoch_pak_auc, epoch_dashboard)
- best_model_prc_curve.png: Precision-Recall curve for best epoch
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_baseline_style():
    """Set consistent plot style for baseline visualizations."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
    })


def plot_baseline_epoch_metrics(epoch_metrics_list, output_dir):
    """Generate epoch-wise metric trend plots for baseline models.

    Similar to MAE's plot_epoch_metrics but without teacher/discriminator metrics.

    Args:
        epoch_metrics_list: List of epoch metric dicts from epoch_metrics.json
        output_dir: Directory to save plots (visualization/epoch_metrics/)
    """
    setup_baseline_style()
    os.makedirs(output_dir, exist_ok=True)

    epochs = [m['epoch'] for m in epoch_metrics_list]
    if len(epochs) < 2:
        return

    # 1. PRC-AUC Evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [m.get('prc_auc', 0) for m in epoch_metrics_list],
            color='#e74c3c', label='PRC-AUC', marker='o', markersize=4, linewidth=2)
    best_idx = np.argmax([m.get('pak_auc_f1', 0) for m in epoch_metrics_list])
    ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5, label=f'Best epoch ({epochs[best_idx]})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PRC-AUC')
    ax.set_title('Point-Level PRC-AUC Over Training')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_prc_auc.png'), dpi=150)
    plt.close(fig)

    # 2. F1_T Evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [m.get('f1_t', 0) for m in epoch_metrics_list],
            color='#e74c3c', label='F1_T', marker='o', markersize=4, linewidth=2)
    ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5, label=f'Best epoch ({epochs[best_idx]})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1_T')
    ax.set_title('Point-Level F1_T Over Training')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_f1_t.png'), dpi=150)
    plt.close(fig)

    # 3. PA%K F1 Evolution (PA0, PA20, PA50, PA100)
    pa_ks = [0, 20, 50, 100]
    pa_colors = ['#95a5a6', '#e74c3c', '#f39c12', '#2ecc71']
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, c in zip(pa_ks, pa_colors):
        ax.plot(epochs, [m.get(f'pa_{k}_f1', 0) for m in epoch_metrics_list],
                color=c, label=f'PA{k}% F1', marker='o', markersize=4)
    ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5, label=f'Best epoch ({epochs[best_idx]})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PA%K F1')
    ax.set_title('Point-Level PA%K F1 Over Training')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_pa_k_f1.png'), dpi=150)
    plt.close(fig)

    # 4. PA%K AUC Evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, [m.get('pak_auc_f1', 0) for m in epoch_metrics_list],
            color='#8e44ad', label='PAK AUC F1', marker='D', markersize=5, linewidth=2)
    ax.plot(epochs, [m.get('pak_auc_prc_auc', 0) for m in epoch_metrics_list],
            color='#e67e22', label='PAK AUC PRC', marker='^', markersize=5, linewidth=2)
    ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5, label=f'Best epoch ({epochs[best_idx]})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PAK AUC')
    ax.set_title('Point-Level PA%K AUC Over Training')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_pak_auc.png'), dpi=150)
    plt.close(fig)

    # 5. Combined dashboard (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0][0]
    ax.plot(epochs, [m.get('prc_auc', 0) for m in epoch_metrics_list],
            color='#e74c3c', linewidth=2)
    ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5)
    ax.set_title('PRC-AUC')

    ax = axes[0][1]
    ax.plot(epochs, [m.get('f1_t', 0) for m in epoch_metrics_list],
            color='#e74c3c', linewidth=2)
    ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5)
    ax.set_title('F1_T')

    ax = axes[1][0]
    for k, c in zip(pa_ks, pa_colors):
        ax.plot(epochs, [m.get(f'pa_{k}_f1', 0) for m in epoch_metrics_list],
                color=c, label=f'PA{k}%', linewidth=1.5)
    ax.set_title('PA%K F1')
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1][1]
    ax.plot(epochs, [m.get('pak_auc_f1', 0) for m in epoch_metrics_list],
            color='#8e44ad', label='PAK F1', linewidth=2)
    ax.plot(epochs, [m.get('pak_auc_prc_auc', 0) for m in epoch_metrics_list],
            color='#e67e22', label='PAK PRC', linewidth=2)
    ax.axvline(x=epochs[best_idx], color='green', linestyle='--', alpha=0.5)
    ax.set_title('PAK AUC')
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel('Epoch', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Baseline Training Dynamics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'epoch_dashboard.png'), dpi=150)
    plt.close(fig)


def plot_baseline_prc_curve(scores, labels, output_dir, model_name=''):
    """Plot Precision-Recall curve from anomaly scores.

    Args:
        scores: 1D anomaly scores array
        labels: 1D binary labels (0=normal, 1=anomaly)
        output_dir: Directory to save best_model_prc_curve.png
        model_name: Model name for title
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    setup_baseline_style()
    os.makedirs(output_dir, exist_ok=True)

    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='#e74c3c', linewidth=2,
            label=f'PRC-AUC = {ap:.4f}')
    ax.fill_between(recall, precision, alpha=0.1, color='#e74c3c')

    # Random baseline
    anomaly_ratio = labels.mean()
    ax.axhline(y=anomaly_ratio, color='gray', linestyle='--', alpha=0.5,
               label=f'Random ({anomaly_ratio:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {model_name}' if model_name else 'Precision-Recall Curve')
    ax.legend(loc='best')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'best_model_prc_curve.png'), dpi=150)
    plt.close(fig)


def _layout_label_positions(centers, label_width, x_min, x_max):
    """Greedy 1-D non-overlap label layout (within a row).

    Mirrors MAE best_model_visualizer._layout_label_positions to keep
    visual layout consistent across MAE + baseline anomaly_threshold plots.
    """
    n = len(centers)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: centers[i])
    positions = [None] * n
    cur = x_min - label_width
    for idx in order:
        new_pos = max(centers[idx], cur + label_width)
        positions[idx] = new_pos
        cur = new_pos
    if positions[order[-1]] > x_max:
        cur = x_max + label_width
        for idx in reversed(order):
            positions[idx] = min(positions[idx], cur - label_width)
            cur = positions[idx]
    return positions


def plot_baseline_anomaly_threshold(scores, labels, regions, threshold,
                                     output_path, model_name='', dataset_name='',
                                     extra=None):
    """Plot per-timestep anomaly score timeline with threshold + GT regions
    + per-region detection ratio annotations + normal-region FPR in title.

    Faithfully mirrors MAE's `plot_anomaly_threshold` (single-panel version
    for baselines which only have one composite anomaly score):
    - Black line: scores
    - Red shading per GT anomaly region; opacity ∝ detection ratio
      `alpha = 0.15 + 0.50 * detection_ratio`
    - Black horizontal dashed line: threshold
    - Detection ratio text `{ratio:.2f}` above each region with greedy
      non-overlap row layout (lines connecting region center to text)
    - Title: `Anomaly Threshold Timeline — best_epoch=..., pak_auc_f1=...,
              FPR(non-anomaly)=...`
      where FPR(non-anomaly) is the fraction of normal-region timesteps
      whose score >= threshold (false positive rate on normal points).

    Detection ratio per region (s, e):
        detection_ratio = (anomaly_score[s:e] >= threshold).sum() / (e - s)

    Normal-region FPR (added per user directive 2026-05-25):
        normal_mask = (labels == 0)
        fpr = (anomaly_score[normal_mask] >= threshold).sum() / normal_mask.sum()

    Region derivation: GT regions are derived from `labels` (binary 0/1)
    if `regions` is None or empty; otherwise honored directly.

    Args:
        scores: 1D anomaly scores array (length T_test, float32)
        labels: 1D binary labels (0=normal, 1=anomaly), length T_test
        regions: optional list of AnomalyRegion(start, end, ...) or (start, end);
                 if None or empty, regions are derived from `labels`
        threshold: optimal threshold from best epoch (None → no threshold ops)
        output_path: full file path to save
        model_name: model identifier (e.g., 'anomaly_transformer')
        dataset_name: dataset/experiment identifier (e.g., 'swat_a1a2_normalonly')
        extra: optional dict with keys 'best_epoch', 'pak_auc_f1', 'prc_auc'
    """
    setup_baseline_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    scores = np.asarray(scores)
    labels = np.asarray(labels) if labels is not None else None
    T = len(scores)
    if labels is not None:
        m = min(T, len(labels))
        scores = scores[:m]
        labels = labels[:m]
    else:
        m = T
    x = np.arange(m)

    # 1. Derive anomaly regions from labels if not supplied (MAE-style).
    test_regions = []
    if regions is not None and len(regions) > 0:
        for r in regions:
            r_start = getattr(r, 'start', r[0] if hasattr(r, '__getitem__') else None)
            r_end = getattr(r, 'end', r[1] if hasattr(r, '__getitem__') else None)
            if r_start is None or r_end is None:
                continue
            r_start = max(0, int(r_start))
            r_end = min(m, int(r_end))
            if r_end > r_start:
                test_regions.append((r_start, r_end))
    elif labels is not None:
        in_anom = False
        start = 0
        for i, lbl in enumerate(labels):
            if lbl == 1 and not in_anom:
                start = i
                in_anom = True
            elif lbl == 0 and in_anom:
                test_regions.append((start, i))
                in_anom = False
        if in_anom:
            test_regions.append((start, m))

    # 2. Detection ratio per region (anomaly recall within each GT region)
    if threshold is not None and np.isfinite(threshold) and test_regions:
        det_ratios = []
        for s, e in test_regions:
            seg = scores[s:e]
            if len(seg) == 0:
                det_ratios.append(0.0)
            else:
                det_ratios.append(float((seg >= threshold).sum()) / float(len(seg)))
    else:
        det_ratios = [0.0] * len(test_regions)

    # 3. Normal-region FPR (false positive rate on non-anomaly timesteps).
    fpr_normal = None
    if threshold is not None and np.isfinite(threshold) and labels is not None:
        normal_mask = (labels == 0)
        n_normal = int(normal_mask.sum())
        if n_normal > 0:
            n_fp = int(((scores >= threshold) & normal_mask).sum())
            fpr_normal = n_fp / n_normal

    # 4. Plot
    fig, ax = plt.subplots(figsize=(16, 5))

    # 4a. Shade regions with opacity ∝ detection ratio
    for ri, (s, e) in enumerate(test_regions):
        if threshold is not None:
            alpha = 0.15 + 0.50 * det_ratios[ri]
        else:
            alpha = 0.25
        ax.axvspan(s, e, alpha=alpha, color='red', zorder=1)

    # 4b. Score line
    ax.plot(x, scores, color='black', linewidth=0.7, alpha=0.9, zorder=3,
            label='_nolegend_')

    # 4c. Threshold line
    if threshold is not None and np.isfinite(threshold):
        ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.0,
                   alpha=0.7, label=f'threshold={threshold:.4f}', zorder=4)

    # 4d. Detection ratio annotations above regions (MAE layout)
    if threshold is not None and test_regions:
        n_reg = len(test_regions)
        if n_reg <= 30:
            fontsize = 8; lw_frac = 0.025
        elif n_reg <= 60:
            fontsize = 7; lw_frac = 0.018
        else:
            fontsize = 6; lw_frac = 0.014
        label_width = m * lw_frac
        rows_needed = max(2, int(np.ceil(n_reg * label_width / (m * 0.9))))
        rows_needed = min(rows_needed, 6)

        y_min, y_max = ax.get_ylim()
        margin = (y_max - y_min) * (0.18 + 0.06 * rows_needed)
        new_top = y_max + margin
        ax.set_ylim(top=new_top)
        row_ys = [new_top - margin * (0.12 + 0.78 * (r + 0.5) / rows_needed)
                  for r in range(rows_needed)]

        centers = [(s + e) / 2.0 for s, e in test_regions]
        row_indices = {r: [] for r in range(rows_needed)}
        for i in range(n_reg):
            row_indices[i % rows_needed].append(i)
        label_x_by_idx = {}
        for r, idxs in row_indices.items():
            pos = _layout_label_positions(
                [centers[i] for i in idxs], label_width, 0, m)
            for k, i in enumerate(idxs):
                label_x_by_idx[i] = pos[k]

        for ri, ((s, e), ratio) in enumerate(zip(test_regions, det_ratios)):
            cx = centers[ri]
            lx = label_x_by_idx[ri]
            ly = row_ys[ri % rows_needed]
            ax.plot([cx, lx], [y_max, ly - margin * 0.04],
                    color='gray', linewidth=0.35, alpha=0.5, zorder=8)
            ax.text(lx, ly, f'{ratio:.2f}',
                    ha='center', va='center',
                    fontsize=fontsize, color='black', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='lightgray', alpha=0.95, linewidth=0.4))
        import matplotlib.patches as mpatches
        anom_patch = mpatches.Patch(color='red', alpha=0.4,
                                    label='anomaly (opacity ∝ detection ratio)')
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=[anom_patch] + handles, loc='upper left',
                  fontsize=9, framealpha=0.9)
    else:
        ax.legend(loc='upper left', fontsize=9)

    ax.set_ylabel('Anomaly Score', fontsize=9)
    ax.set_xlabel('Test point index')
    ax.set_xlim(0, m)
    ax.grid(alpha=0.3)

    # 5. Title with best_epoch / pak_auc_f1 / pak_auc_prc + FPR(non-anomaly)
    title_parts = ['Anomaly Threshold Timeline']
    if model_name:
        title_parts.append(f'{model_name}')
    if dataset_name:
        title_parts.append(f'{dataset_name}')
    title_line1 = ' — '.join(title_parts)

    subtitle_parts = []
    if extra is not None:
        if 'best_epoch' in extra and extra['best_epoch'] is not None:
            subtitle_parts.append(f"best_epoch={extra['best_epoch']}")
        if 'pak_auc_f1' in extra:
            subtitle_parts.append(f"pak_auc_f1={extra['pak_auc_f1']:.4f}")
        if 'prc_auc' in extra:
            subtitle_parts.append(f"pak_auc_prc={extra['prc_auc']:.4f}")
    if fpr_normal is not None:
        subtitle_parts.append(f"FPR(non-anomaly)={fpr_normal:.4f}")
    subtitle = ', '.join(subtitle_parts)

    fig.suptitle(f'{title_line1}\n{subtitle}' if subtitle else title_line1,
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
