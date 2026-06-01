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
