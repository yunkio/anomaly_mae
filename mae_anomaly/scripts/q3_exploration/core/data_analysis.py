"""
Dataset-level deep analysis utilities (Q3 v6).

본 module은 method-level이 아닌 **data-level / score-level** 분석:

- visualize_dataset_score: raw signal + score + labels overlay
- per_dataset_oracle_channel_mixing: 4-channel optimal weight search
- per_dataset_oracle_pak: theoretical maximum (가능한 모든 transformation)
- per_channel_anomaly_alignment: 각 channel의 label alignment 측정
- anomaly_isolation_profile: 각 anomaly region의 contextual isolation
- score_to_label_alignment: score sequence의 label alignment quality 정량화
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr


def per_channel_anomaly_separation(channel_score, point_labels, eval_mask=None):
    """각 channel의 anomaly vs normal mean separation 측정.

    Returns:
        separation: (anomaly_mean - normal_mean) / normal_std
        snr_anomaly: 평균 in-anomaly score vs out-anomaly std
    """
    if eval_mask is not None:
        mask = eval_mask
    else:
        mask = np.ones_like(point_labels, dtype=bool)

    labels_m = point_labels[mask]
    scores_m = channel_score[mask]

    in_anom = scores_m[labels_m == 1]
    out_anom = scores_m[labels_m == 0]

    if len(in_anom) < 5 or len(out_anom) < 5:
        return 0.0, 0.0

    sep = (in_anom.mean() - out_anom.mean()) / (out_anom.std() + 1e-9)
    return float(sep), float((in_anom.mean() - out_anom.mean()) / (out_anom.std() + in_anom.std() + 1e-9))


def per_dataset_oracle_channel_mixing(pt_r, pt_d, pt_s, pt_f, point_labels, regions,
                                        eval_mask=None, n_grid=11):
    """Oracle channel mixing: 모든 (w_r, w_d, w_s, w_f) 조합에서 best AUC.

    Args:
        pt_r, pt_d, pt_s, pt_f: per-point channel scores (post-aggregate)
        point_labels: binary labels
        regions: anomaly regions
        eval_mask: optional mask
        n_grid: weight grid resolution

    Returns:
        best_weights: (w_r, w_d, w_s, w_f)
        best_pak: best pak_auc_f1
        all_results: list of (weights, pak)
    """
    from mae_anomaly.scripts.q3_exploration.core.evaluation import pak_auc_f1

    # Normalize channels
    channels = [pt_r, pt_d, pt_s, pt_f]
    channels_norm = []
    for ch in channels:
        if ch.std() > 1e-9:
            channels_norm.append((ch - ch.min()) / (ch.max() - ch.min() + 1e-9))
        else:
            channels_norm.append(np.zeros_like(ch))

    # Grid search: simplex of weights
    grid = np.linspace(0, 1, n_grid)
    best_pak = -np.inf
    best_weights = (0.25, 0.25, 0.25, 0.25)
    all_results = []

    # Constrained sum=1: iterate w_r, w_d, w_s; w_f = 1 - sum
    for w_r in grid:
        for w_d in grid:
            if w_r + w_d > 1.01: continue
            for w_s in grid:
                if w_r + w_d + w_s > 1.01: continue
                w_f = 1.0 - w_r - w_d - w_s
                if w_f < -0.001: continue
                w_f = max(0, w_f)

                mixed = (w_r * channels_norm[0] + w_d * channels_norm[1] +
                         w_s * channels_norm[2] + w_f * channels_norm[3])
                # Apply standard gauss10 smooth (level the playing field)
                mixed_smoothed = gaussian_filter1d(mixed.astype(np.float64),
                                                     sigma=10, mode='reflect')
                pak = pak_auc_f1(mixed_smoothed, point_labels, regions, eval_mask)
                all_results.append(((w_r, w_d, w_s, w_f), float(pak)))
                if pak > best_pak:
                    best_pak = pak
                    best_weights = (w_r, w_d, w_s, w_f)

    return best_weights, float(best_pak), all_results


def anomaly_isolation_profile(score_sequence, regions, eval_mask=None,
                                context_size=200):
    """각 anomaly region의 contextual isolation 측정.

    Isolation = 본 region의 max score vs 같은 context의 second-highest peak.
    높을수록 detection 쉬움.

    Args:
        score_sequence: per-point anomaly score
        regions: list of AnomalyRegion
        context_size: timesteps before/after region for context

    Returns:
        per_region_isolation: list of dicts with isolation metrics
    """
    profiles = []
    for r in regions:
        if eval_mask is not None and not eval_mask[r.start:r.end].any():
            continue
        ctx_start = max(0, r.start - context_size)
        ctx_end = min(len(score_sequence), r.end + context_size)

        in_region = score_sequence[r.start:r.end]
        in_max = float(in_region.max()) if len(in_region) > 0 else 0.0
        in_mean = float(in_region.mean()) if len(in_region) > 0 else 0.0

        # Context (excluding region)
        ctx = np.concatenate([score_sequence[ctx_start:r.start],
                               score_sequence[r.end:ctx_end]])
        if len(ctx) < 5:
            ctx_max = ctx_mean = ctx_std = 0
        else:
            ctx_max = float(ctx.max())
            ctx_mean = float(ctx.mean())
            ctx_std = float(ctx.std())

        isolation = (in_max - ctx_mean) / (ctx_std + 1e-9)
        contrast = (in_max - ctx_max) / (ctx_std + 1e-9)
        segment_internal_var = float(np.std(in_region)) if len(in_region) > 1 else 0.0

        profiles.append({
            'start': int(r.start), 'end': int(r.end),
            'length': int(r.end - r.start),
            'in_max': in_max, 'in_mean': in_mean,
            'ctx_max': ctx_max, 'ctx_mean': ctx_mean, 'ctx_std': ctx_std,
            'isolation': float(isolation),  # how far above context mean
            'contrast': float(contrast),    # how far above context max
            'internal_variability': segment_internal_var,
        })
    return profiles


def score_label_alignment_metrics(score_sequence, point_labels, regions, eval_mask=None):
    """Score sequence의 label alignment quality (전체).

    Returns:
        - mean_score_in_anom: avg score inside anomaly
        - mean_score_out_anom: avg score outside
        - separation: standardized difference
        - point_pearson: per-point pearson correlation (score, labels)
        - point_spearman: per-point spearman
        - region_recall_top_k: top-k% scores가 anomaly region에 hit하는 ratio
    """
    if eval_mask is None:
        eval_mask = np.ones_like(point_labels, dtype=bool)

    scores_m = score_sequence[eval_mask]
    labels_m = point_labels[eval_mask]

    in_anom = scores_m[labels_m == 1]
    out_anom = scores_m[labels_m == 0]

    metrics = {}
    if len(in_anom) >= 5 and len(out_anom) >= 5:
        metrics['mean_score_in_anom'] = float(in_anom.mean())
        metrics['mean_score_out_anom'] = float(out_anom.mean())
        metrics['separation'] = float((in_anom.mean() - out_anom.mean()) /
                                       (out_anom.std() + 1e-9))
    else:
        metrics['mean_score_in_anom'] = 0
        metrics['mean_score_out_anom'] = 0
        metrics['separation'] = 0

    # Point-level correlation
    if scores_m.std() > 1e-9 and labels_m.std() > 1e-9:
        rp, _ = pearsonr(scores_m, labels_m)
        rs, _ = spearmanr(scores_m, labels_m)
        metrics['point_pearson'] = float(rp)
        metrics['point_spearman'] = float(rs)
    else:
        metrics['point_pearson'] = 0
        metrics['point_spearman'] = 0

    # Region recall at top-k%: 각 top-k% high-score positions가 anomaly region 안에 있는 비율
    n_anom = labels_m.sum()
    if n_anom > 0:
        sorted_indices = np.argsort(-scores_m)
        for top_k in [0.5, 1, 2, 5, 10]:
            n_top = int(len(scores_m) * top_k / 100)
            if n_top == 0:
                metrics[f'precision_at_top_{top_k}pct'] = 0
                continue
            top_indices = sorted_indices[:n_top]
            hit_rate = labels_m[top_indices].mean()
            metrics[f'precision_at_top_{top_k}pct'] = float(hit_rate)

    return metrics


def per_channel_oracle_with_smoothing(pt_r, pt_d, pt_s, pt_f, point_labels, regions,
                                        eval_mask=None,
                                        sigma_candidates=None,
                                        n_grid_mix=6):
    """Joint oracle: channel mixing × σ smoothing.

    더 expensive but theoretical ceiling 더 잘 measure.
    """
    from mae_anomaly.scripts.q3_exploration.core.evaluation import pak_auc_f1
    if sigma_candidates is None:
        sigma_candidates = [1, 2, 5, 10, 20, 50, 100, 200]

    # Normalize
    channels = [pt_r, pt_d, pt_s, pt_f]
    channels_norm = []
    for ch in channels:
        if ch.std() > 1e-9:
            channels_norm.append((ch - ch.min()) / (ch.max() - ch.min() + 1e-9))
        else:
            channels_norm.append(np.zeros_like(ch))

    grid = np.linspace(0, 1, n_grid_mix)
    best_pak = -np.inf
    best_params = None

    for w_r in grid:
        for w_d in grid:
            if w_r + w_d > 1.01: continue
            for w_s in grid:
                if w_r + w_d + w_s > 1.01: continue
                w_f = max(0, 1.0 - w_r - w_d - w_s)
                mixed = (w_r * channels_norm[0] + w_d * channels_norm[1] +
                         w_s * channels_norm[2] + w_f * channels_norm[3])
                for sigma in sigma_candidates:
                    smoothed = gaussian_filter1d(mixed.astype(np.float64),
                                                  sigma=sigma, mode='reflect')
                    pak = pak_auc_f1(smoothed, point_labels, regions, eval_mask)
                    if pak > best_pak:
                        best_pak = pak
                        best_params = ((w_r, w_d, w_s, w_f), sigma)

    return best_params, float(best_pak)


def visualize_dataset(alias, scoring_module, output_path):
    """Generate visualization PNG for a dataset.

    Plots:
    - 4 channel scores (recon, disc, student, fm)
    - Baseline smoothed (gauss10)
    - Best method (P12) smoothed
    - Anomaly regions overlay
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mae_anomaly.scripts.q3_exploration.core.data import DatasetScores
    from mae_anomaly.scripts.q3_exploration.core.scoring import (
        per_channel_points, adaptive_combine, gauss
    )

    is_swat_excl = (alias == 'swat_excl22')
    ds = DatasetScores.load(alias, is_swat_excl)
    if ds is None:
        return None

    pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
    base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
    base_smoothed = gauss(base, 10)

    # Plot
    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)

    # Each channel
    for ax, (label, ch) in zip(axes[:4],
                                 [('recon', pt_r), ('disc', pt_d),
                                  ('student', pt_s), ('fm', pt_f)]):
        ax.plot(ch, lw=0.5, alpha=0.7, color='steelblue')
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=0.3)
        # Anomaly regions
        for r in ds.regions:
            ax.axvspan(r.start, r.end, alpha=0.2, color='red')

    # Adaptive baseline
    axes[4].plot(base_smoothed, lw=0.8, color='black')
    axes[4].set_ylabel('gauss10', fontsize=10)
    axes[4].grid(alpha=0.3)
    for r in ds.regions:
        axes[4].axvspan(r.start, r.end, alpha=0.2, color='red')

    # Labels
    axes[5].fill_between(range(len(ds.point_labels)), 0, ds.point_labels,
                          color='red', alpha=0.7)
    axes[5].set_ylabel('label', fontsize=10)
    axes[5].set_ylim(0, 1.2)
    axes[5].grid(alpha=0.3)
    axes[5].set_xlabel('Timestep')

    plt.suptitle(f'Dataset: {alias}  (n_regions={len(ds.regions)})', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return output_path
