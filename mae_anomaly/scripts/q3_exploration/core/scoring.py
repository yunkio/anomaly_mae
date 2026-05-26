"""
Scoring utilities — aggregation, smoothing, channel combination.

Self-contained: 외부 dependency 없음 (scipy.ndimage만 사용).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def aggregate_K50(patch_scores, ws, num_patches, patch_size, total_length):
    """Per-point average of ~24 patch contributions (stride=21 subsample).

    Args:
        patch_scores: (n_windows, num_patches) per-window per-patch scores
        ws: (n_windows,) window start indices
        num_patches, patch_size, total_length: scalars

    Returns:
        (total_length,) per-point aggregated scores
    """
    target = np.arange(num_patches, dtype=np.int64)
    n_w = len(patch_scores)
    patch_starts = ws[:, None] + target[None, :] * patch_size
    scores = patch_scores[:, target]
    offsets = np.arange(patch_size)
    t_all = (patch_starts[:, :, None] + offsets[None, None, :]).ravel()
    s_all = np.broadcast_to(scores[:, :, None], (n_w, len(target), patch_size)).ravel()
    valid = (t_all >= 0) & (t_all < total_length)
    point_sum = np.bincount(t_all[valid], weights=s_all[valid], minlength=total_length)
    point_cnt = np.bincount(t_all[valid], minlength=total_length)
    return np.where(point_cnt > 0, point_sum / np.maximum(point_cnt, 1), 0.0)


def adaptive_combine(recon, disc, fm, use_fm=True):
    """Adaptive score combination (이전 274 protocol).
    score = recon + (scaled_disc + scaled_fm) / 2
    """
    rm = recon.mean() + 1e-4
    sd = disc * (rm / (disc.mean() + 1e-4))
    if use_fm and np.abs(fm).sum() > 0:
        sfm = fm * (rm / (fm.mean() + 1e-4))
        st = (sd + sfm) / 2.0
    else:
        st = sd
    return recon + st


def gauss(score, sigma):
    """Gaussian smoothing with safe sigma."""
    return gaussian_filter1d(score.astype(np.float64),
                              sigma=max(sigma, 0.5), mode='reflect')


def zscore(x):
    m, s = x.mean(), x.std() + 1e-10
    return (x - m) / s


def stride_subsample(scores, ws, stride=21):
    """Subsample (n_w, ...) scores at stride."""
    mask = np.arange(len(ws)) % stride == 0
    return scores[mask], ws[mask]


def point_score_from_loo(ds, sigma=10, stride=21, use_fm=True):
    """Standard point-level score: aggregate_K50 + adaptive_combine + gauss.

    Args:
        ds: DatasetScores
        sigma: gaussian smoothing sigma
        stride: subsampling stride (default 21 matches Tier-0 baseline)
        use_fm: include feature matching channel
    Returns:
        (total_length,) smoothed adaptive score
    """
    r_s, ws_s = stride_subsample(ds.recon, ds.window_start_indices, stride)
    d_s, _ = stride_subsample(ds.disc, ds.window_start_indices, stride)
    f_s, _ = stride_subsample(ds.fm, ds.window_start_indices, stride)
    pt_r = aggregate_K50(r_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    pt_d = aggregate_K50(d_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    pt_f = aggregate_K50(f_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=use_fm)
    return gauss(base, sigma)


def per_channel_points(ds, stride=21):
    """Returns 4 point-level channels (no smoothing): pt_r, pt_d, pt_s, pt_f."""
    r_s, ws_s = stride_subsample(ds.recon, ds.window_start_indices, stride)
    d_s, _ = stride_subsample(ds.disc, ds.window_start_indices, stride)
    s_s, _ = stride_subsample(ds.student, ds.window_start_indices, stride)
    f_s, _ = stride_subsample(ds.fm, ds.window_start_indices, stride)
    pt_r = aggregate_K50(r_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    pt_d = aggregate_K50(d_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    pt_s = aggregate_K50(s_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    pt_f = aggregate_K50(f_s, ws_s, ds.num_patches, ds.patch_size, ds.total_length)
    return pt_r, pt_d, pt_s, pt_f
