"""
Dataset Difficulty Analysis — what makes a dataset hard?

본 script:
1. Compute hardness metric (mean Δ across top-K methods)
2. Compute anomaly characteristic features per dataset:
   - median segment length, segment variance, anomaly ratio
   - score distribution: SNR, skewness, kurtosis
   - n_regions, segment length spread
3. Correlation between hardness and these features
4. Regression: hardness ~ characteristics
5. Identify which characteristics predict hardness most

Output:
- dataset_difficulty_correlations.json
- dataset_difficulty_regression.json
- difficulty_vs_features.png (scatter plots)
"""
import sys
from pathlib import Path
import numpy as np
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, median_anomaly_segment_length,
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    per_channel_points, adaptive_combine, gauss,
)
from mae_anomaly.scripts.q3_exploration.core.meta_aggregation import load_all_results


def extract_dataset_characteristics(alias, swat_excl22):
    """Extract per-dataset characteristic features."""
    ds = DatasetScores.load(alias, swat_excl22)
    if ds is None:
        return None

    pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
    base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
    base_smoothed = gauss(base, 10)

    seg_lens = [r.end - r.start for r in ds.regions]
    if not seg_lens:
        return None

    # SNR: difference between in-anomaly mean and outside mean / outside std
    in_anom_mask = np.zeros(ds.total_length, dtype=bool)
    for r in ds.regions:
        in_anom_mask[r.start:r.end] = True
    out_anom_mask = ~in_anom_mask
    if ds.eval_mask is not None:
        in_anom_mask = in_anom_mask & ds.eval_mask
        out_anom_mask = out_anom_mask & ds.eval_mask

    if out_anom_mask.sum() < 10 or in_anom_mask.sum() < 1:
        snr = 0.0
    else:
        in_mean = base_smoothed[in_anom_mask].mean()
        out_mean = base_smoothed[out_anom_mask].mean()
        out_std = base_smoothed[out_anom_mask].std() + 1e-9
        snr = (in_mean - out_mean) / out_std

    from scipy.stats import skew, kurtosis
    return {
        'alias': alias,
        'median_seg_length': float(np.median(seg_lens)),
        'mean_seg_length': float(np.mean(seg_lens)),
        'max_seg_length': float(np.max(seg_lens)),
        'std_seg_length': float(np.std(seg_lens) if len(seg_lens) > 1 else 0.0),
        'iqr_seg_length': float(np.percentile(seg_lens, 75) - np.percentile(seg_lens, 25)
                                 if len(seg_lens) > 4 else 0.0),
        'log_median_seg': float(np.log10(np.median(seg_lens) + 1)),
        'log_mean_seg': float(np.log10(np.mean(seg_lens) + 1)),
        'n_regions': len(seg_lens),
        'log_n_regions': float(np.log10(len(seg_lens) + 1)),
        'anomaly_ratio': float(ds.point_labels.mean()),
        'snr': float(snr),
        'log_snr': float(np.log10(max(snr, 0.1) + 1)),
        'score_skewness': float(skew(base_smoothed)),
        'score_kurtosis': float(kurtosis(base_smoothed)),
        'score_iqr': float(np.percentile(base_smoothed, 75)
                            - np.percentile(base_smoothed, 25)),
        'total_length': len(base_smoothed),
        'log_total_length': float(np.log10(len(base_smoothed))),
        'recon_disc_ratio': float(pt_r.mean() / (pt_d.mean() + 1e-9)),
        'log_recon_disc_ratio': float(np.log(pt_r.mean() / (pt_d.mean() + 1e-9) + 1e-9)),
    }


def main():
    # Stage 1: Get all meta results
    matrix = load_all_results()
    aliases, method_names, delta_matrix = matrix.to_matrix()

    print(f"Matrix: {delta_matrix.shape}")

    # Stage 2: Compute hardness (mean Δ across top-20 methods)
    mean_deltas = np.nanmean(delta_matrix, axis=0)
    top20_method_idx = np.argsort(-mean_deltas)[:20]
    hardness = np.nanmean(delta_matrix[:, top20_method_idx], axis=1)

    # Stage 3: Per-dataset characteristics
    print("\nExtracting dataset characteristics...")
    swat_excl22_aliases = {'swat_excl22': True}
    char_records = []
    valid_indices = []
    for i, alias in enumerate(aliases):
        is_swat_excl = (alias == 'swat_excl22')
        record = extract_dataset_characteristics(alias, is_swat_excl)
        if record is not None:
            record['hardness'] = float(hardness[i])
            char_records.append(record)
            valid_indices.append(i)

    print(f"Extracted characteristics for {len(char_records)}/{len(aliases)} datasets")

    # Stage 4: Correlation analysis
    feature_keys = ['log_median_seg', 'log_mean_seg', 'max_seg_length',
                     'std_seg_length', 'iqr_seg_length', 'log_n_regions',
                     'anomaly_ratio', 'log_snr', 'score_skewness',
                     'score_kurtosis', 'score_iqr', 'log_total_length',
                     'log_recon_disc_ratio']

    X = np.array([[r[k] for k in feature_keys] for r in char_records])
    y = np.array([r['hardness'] for r in char_records])

    print("\n=== Feature vs Hardness Correlations ===")
    print(f"{'Feature':<25s} {'Pearson':>10s} {'Spearman':>10s}")
    correlations = {}
    for j, k in enumerate(feature_keys):
        r_p, p_p = pearsonr(X[:, j], y)
        r_s, p_s = spearmanr(X[:, j], y)
        correlations[k] = {
            'pearson': float(r_p), 'pearson_p': float(p_p),
            'spearman': float(r_s), 'spearman_p': float(p_s),
        }
        marker = ' *' if abs(r_p) > 0.4 else ''
        print(f"{k:<25s} {r_p:>+10.3f} {r_s:>+10.3f}{marker}")

    # Stage 5: Regression model — which features predict hardness?
    print("\n=== Linear Regression: hardness ~ features ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = Ridge(alpha=0.5)
    lr.fit(X_scaled, y)
    r2 = lr.score(X_scaled, y)
    print(f"R² = {r2:.3f}")
    print(f"\nFeature importance (standardized coefficients):")
    for k, c in sorted(zip(feature_keys, lr.coef_), key=lambda x: -abs(x[1])):
        print(f"  {k:<25s}: {c:+.4f}")

    # LOO cross-validation R²
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X_scaled):
        m = Ridge(alpha=0.5)
        m.fit(X_scaled[train_idx], y[train_idx])
        preds[test_idx[0]] = m.predict(X_scaled[test_idx])[0]
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_loo = 1 - ss_res / ss_tot
    print(f"LOO R² = {r2_loo:.3f}")

    # Stage 6: Visualizations
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()
    top_correlations = sorted(correlations.items(),
                               key=lambda x: -abs(x[1]['pearson']))[:12]

    for ax_idx, (feat, corr_data) in enumerate(top_correlations):
        ax = axes[ax_idx]
        x_vals = [r[feat] for r in char_records]
        ax.scatter(x_vals, y, c=['green' if h > 0 else 'red' for h in y],
                    alpha=0.6, s=30)
        ax.set_xlabel(feat)
        ax.set_ylabel('Hardness (mean Δ top-20 methods)')
        ax.set_title(f'r_p={corr_data["pearson"]:+.3f} r_s={corr_data["spearman"]:+.3f}')
        ax.axhline(0, color='black', linestyle=':', alpha=0.3)
        # Trend line
        z = np.polyfit(x_vals, y, 1)
        x_range = np.linspace(min(x_vals), max(x_vals), 50)
        ax.plot(x_range, np.polyval(z, x_range), 'b--', alpha=0.5)

    plt.suptitle(f'Dataset Hardness vs Anomaly Characteristics\n'
                  f'(Ridge R²={r2:.3f}, LOO R²={r2_loo:.3f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'difficulty_vs_features.png', dpi=120, bbox_inches='tight')
    plt.close()

    # Stage 7: Predicting per-dataset best method using characteristics
    print("\n=== Per-Dataset Best Method Analysis ===")
    # Method clustering (top 30 methods)
    top30_idx = np.argsort(-mean_deltas)[:30]
    method_best_counts = {}
    valid_aliases = [r['alias'] for r in char_records]
    for r_idx, alias in enumerate(valid_aliases):
        original_alias_idx = aliases.index(alias)
        deltas_for_dataset = delta_matrix[original_alias_idx, top30_idx]
        best_local_idx = np.argmax(deltas_for_dataset)
        best_method = method_names[top30_idx[best_local_idx]]
        method_best_counts.setdefault(best_method, []).append(alias)

    # Save analysis
    analysis_result = {
        'feature_correlations': correlations,
        'regression_r2_train': float(r2),
        'regression_r2_loo': float(r2_loo),
        'feature_importance': dict(zip(feature_keys, lr.coef_.tolist())),
        'dataset_characteristics': char_records,
        'top30_best_method_per_dataset': {
            alias: {
                'best_method': method_names[top30_idx[np.argmax(delta_matrix[aliases.index(alias), top30_idx])]],
                'best_delta': float(np.max(delta_matrix[aliases.index(alias), top30_idx])),
                'hardness': float(hardness[aliases.index(alias)]),
            }
            for alias in valid_aliases
        }
    }
    with open(output_dir / 'dataset_difficulty_analysis.json', 'w') as f:
        json.dump(analysis_result, f, indent=2)

    print(f"\nSaved analysis: {output_dir / 'dataset_difficulty_analysis.json'}")
    print(f"Plot: {output_dir / 'difficulty_vs_features.png'}")


if __name__ == "__main__":
    main()
