"""
Phase A — Unsupervised E9 Sigma Estimation

E9 adapt_single은 median_seg(test labels)로부터 σ를 계산 → semi-supervised.
본 실험은 labels 없이 score sequence 자체에서 σ를 추정하는 3가지 방법 검증.

Methods:
- A1: Peak width estimation (95th percentile threshold + run lengths)
- A2: Multi-σ ensemble agreement (서로 다른 σ로 smoothed scores 일치도)
- A3: KDE-based FWHM (score density mode width)

Comparison baselines:
- baseline_gauss10 (σ=10 fixed)
- E9 adapt (σ = median_seg / 3, semi-supervised; 우리의 ceiling)
- E9 oracle (per-dataset best σ ∈ sweep grid)
"""
import sys
from pathlib import Path
import numpy as np
import json
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, median_anomaly_segment_length, get_per_group
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    point_score_from_loo, gauss, zscore
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import (
    pak_auc_f1, wilcoxon_test, per_group_summary
)


def estimate_sigma_v1_peak_width(scores, percentile=95):
    """A1: Peak width via 95th percentile threshold + run lengths."""
    threshold = np.percentile(scores, percentile)
    binary = scores > threshold
    if binary.sum() == 0:
        return 10.0  # fallback

    # Find connected runs
    diffs = np.diff(binary.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if binary[0]:
        starts = np.concatenate([[0], starts])
    if binary[-1]:
        ends = np.concatenate([ends, [len(binary)]])

    if len(starts) == 0:
        return 10.0
    run_lengths = ends - starts
    # Robust: use median of runs longer than 2 timesteps (filter brief spikes)
    valid_runs = run_lengths[run_lengths > 2]
    if len(valid_runs) == 0:
        return 10.0
    estimated_seg = np.median(valid_runs)
    return max(estimated_seg / 3.0, 0.5)


def estimate_sigma_v2_multi_agreement(base_score, sigmas=[3, 10, 30, 100, 300]):
    """A2: Multi-σ ensemble agreement.

    각 σ로 smoothed score 간 correlation을 측정.
    Anomaly가 specific scale에 잘 detect되면, 그 σ에서 score variance가 anomaly 위치에 집중되어
    다른 σ들과 더 strong correlation.

    Strategy: 각 σ smoothed score의 z-norm 후 pairwise correlation matrix.
    가장 mean correlation 높은 σ를 선택 (consensus winner).
    """
    smoothed = [gaussian_filter1d(base_score, s, mode='reflect') for s in sigmas]
    z_normed = [zscore(s) for s in smoothed]

    n = len(sigmas)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Pearson correlation on full sequence
                corr_matrix[i, j] = np.corrcoef(z_normed[i], z_normed[j])[0, 1]

    # 각 σ의 mean correlation (자기 자신 제외)
    mean_corr = (corr_matrix.sum(axis=1) - 1) / (n - 1)
    best_idx = np.argmax(mean_corr)
    return sigmas[best_idx]


def estimate_sigma_v3_kde(base_score, n_samples=5000):
    """A3: KDE-based FWHM estimation.

    Score distribution의 primary mode width를 통해 σ 추정.
    Anomaly가 distinct mode를 형성하면 그 mode width가 σ proxy.
    """
    # Subsample if too long
    if len(base_score) > n_samples:
        idx = np.random.RandomState(0).choice(len(base_score), n_samples, replace=False)
        sample = base_score[idx]
    else:
        sample = base_score

    try:
        kde = gaussian_kde(sample, bw_method='silverman')
    except Exception:
        return 10.0

    xs = np.linspace(sample.min(), sample.max(), 500)
    density = kde(xs)

    # Find primary mode + FWHM
    peak_idx = np.argmax(density)
    peak_val = density[peak_idx]
    half_max = peak_val / 2.0

    # Left FWHM bound
    left_mask = density[:peak_idx] < half_max
    if left_mask.sum() > 0:
        left_idx = np.where(left_mask)[0][-1]
    else:
        left_idx = 0

    # Right FWHM bound
    right_mask = density[peak_idx:] < half_max
    if right_mask.sum() > 0:
        right_idx = peak_idx + np.where(right_mask)[0][0]
    else:
        right_idx = len(density) - 1

    fwhm_in_xs = xs[right_idx] - xs[left_idx]
    # FWHM이 score range 단위이므로 timestep 단위로 직접 변환 불가
    # 대신 anomaly score sequence의 본질적 시간 scale로 mapping
    # 휴리스틱: FWHM이 score range의 상대 비율로 시간 scale 결정
    score_range = base_score.max() - base_score.min() + 1e-9
    fwhm_ratio = fwhm_in_xs / score_range

    # σ estimate: fwhm_ratio가 크면 (mode가 넓으면) anomaly가 spread → 큰 σ
    # 작으면 → 짧은 anomaly → 작은 σ
    # 본 heuristic은 정확하지 않을 수 있음 — empirically calibrate 필요
    estimated_sigma = 10.0 * (1 + fwhm_ratio * 5)  # heuristic mapping
    return max(min(estimated_sigma, 200.0), 0.5)


def estimate_sigma_v4_score_peak_width(base_score, percentile=90, n_bins=50):
    """A4 (보완): Peak width estimation on smoothed score (gauss10 사용).
    A1보다 robust한 variant.
    """
    smoothed = gaussian_filter1d(base_score, sigma=10, mode='reflect')
    threshold = np.percentile(smoothed, percentile)
    binary = smoothed > threshold

    if binary.sum() < 5:
        return 10.0

    diffs = np.diff(binary.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if binary[0]:
        starts = np.concatenate([[0], starts])
    if binary[-1]:
        ends = np.concatenate([ends, [len(binary)]])

    if len(starts) == 0:
        return 10.0
    run_lengths = ends - starts
    valid_runs = run_lengths[run_lengths > 5]
    if len(valid_runs) == 0:
        return 10.0
    return max(np.median(valid_runs) / 3.0, 0.5)


def process_one(alias, swat_excl22):
    """Single dataset evaluation."""
    ds = DatasetScores.load(alias, swat_excl22)
    if ds is None:
        return None

    # Baseline: gauss10
    baseline = point_score_from_loo(ds, sigma=10, stride=21, use_fm=True)
    baseline_pak = pak_auc_f1(baseline, ds.point_labels, ds.regions, ds.eval_mask)

    # E9 adapt (semi-supervised, our ceiling for unsup)
    median_seg = median_anomaly_segment_length(ds.regions)
    sigma_adapt = median_seg / 3.0
    e9_score = point_score_from_loo(ds, sigma=sigma_adapt, stride=21, use_fm=True)
    e9_pak = pak_auc_f1(e9_score, ds.point_labels, ds.regions, ds.eval_mask)

    # Base score (pre-smoothing) for unsupervised estimation
    from mae_anomaly.scripts.q3_exploration.core.scoring import per_channel_points, adaptive_combine
    pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
    base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)

    # A1: peak width on UNsmoothed score
    sigma_a1 = estimate_sigma_v1_peak_width(base_unsmoothed)
    a1_score = gauss(base_unsmoothed, sigma_a1)
    a1_pak = pak_auc_f1(a1_score, ds.point_labels, ds.regions, ds.eval_mask)

    # A2: multi-σ agreement
    sigma_a2 = estimate_sigma_v2_multi_agreement(base_unsmoothed)
    a2_score = gauss(base_unsmoothed, sigma_a2)
    a2_pak = pak_auc_f1(a2_score, ds.point_labels, ds.regions, ds.eval_mask)

    # A3: KDE-based
    sigma_a3 = estimate_sigma_v3_kde(base_unsmoothed)
    a3_score = gauss(base_unsmoothed, sigma_a3)
    a3_pak = pak_auc_f1(a3_score, ds.point_labels, ds.regions, ds.eval_mask)

    # A4: peak width on PRE-smoothed
    sigma_a4 = estimate_sigma_v4_score_peak_width(base_unsmoothed)
    a4_score = gauss(base_unsmoothed, sigma_a4)
    a4_pak = pak_auc_f1(a4_score, ds.point_labels, ds.regions, ds.eval_mask)

    return {
        'alias': alias,
        'baseline_pak': baseline_pak,
        'e9_pak': e9_pak,
        'median_seg': median_seg,
        'sigma_adapt': sigma_adapt,
        'a1_sigma': float(sigma_a1), 'a1_pak': a1_pak,
        'a2_sigma': float(sigma_a2), 'a2_pak': a2_pak,
        'a3_sigma': float(sigma_a3), 'a3_pak': a3_pak,
        'a4_sigma': float(sigma_a4), 'a4_pak': a4_pak,
    }


def main():
    targets = iter_dataset_aliases()
    print(f"Phase A — Unsupervised σ estimation, {len(targets)} datasets")

    results = {}
    for i, (alias, swat) in enumerate(targets, 1):
        try:
            r = process_one(alias, swat)
            if r is None:
                continue
            results[alias] = r
            print(f"[{i:2d}/{len(targets)}] {alias:<25s} base={r['baseline_pak']:.4f} "
                  f"e9={r['e9_pak']:.4f} A1={r['a1_pak']:.4f}(σ={r['a1_sigma']:.1f}) "
                  f"A2={r['a2_pak']:.4f}(σ={r['a2_sigma']:.0f}) "
                  f"A3={r['a3_pak']:.4f}(σ={r['a3_sigma']:.1f}) "
                  f"A4={r['a4_pak']:.4f}(σ={r['a4_sigma']:.1f})",
                  flush=True)
        except Exception as e:
            print(f"FAILED {alias}: {e}")

    # Save
    out = Path(__file__).parent.parent / 'results' / 'phaseA_unsupervised_sigma.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    # Aggregate
    print("\n=== Aggregate Δ vs baseline gauss10 ===")
    for method in ['e9', 'a1', 'a2', 'a3', 'a4']:
        deltas = {a: r[f'{method}_pak'] - r['baseline_pak'] for a, r in results.items()}
        baseline_scores = [r['baseline_pak'] for r in results.values()]
        method_scores = [r[f'{method}_pak'] for r in results.values()]
        mean_d = np.mean(list(deltas.values()))
        wins = sum(1 for d in deltas.values() if d > 0)
        losses = sum(1 for d in deltas.values() if d < 0)
        cata = sum(1 for d in deltas.values() if d < -0.05)
        p = wilcoxon_test(method_scores, baseline_scores, alternative='greater')
        print(f"  {method:<5s}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p(>)={p:.3f}")

    # Per-group
    print("\n=== Per-group breakdown for best unsupervised (A1) ===")
    deltas_a1 = {a: r['a1_pak'] - r['baseline_pak'] for a, r in results.items()}
    summary = per_group_summary(deltas_a1, get_per_group)
    for g, s in summary.items():
        print(f"  {g:<12s} n={s['n']:2d}  meanΔ={s['mean_delta']:+.4f}  "
              f"W/L={s['wins']}/{s['losses']}  cata={s['catastrophic']}")

    # Adapt-vs-unsupervised correlation
    print("\n=== Adapt σ vs unsupervised σ correlation ===")
    for method in ['a1', 'a2', 'a3', 'a4']:
        adapt_sigmas = [r['sigma_adapt'] for r in results.values()]
        method_sigmas = [r[f'{method}_sigma'] for r in results.values()]
        rho = np.corrcoef(adapt_sigmas, method_sigmas)[0, 1]
        print(f"  {method} σ vs adapt σ: Pearson ρ = {rho:.3f}")


if __name__ == "__main__":
    main()
