"""
P26 — Training Distribution Audit (Q3 v7, PRIORITY 4)

본 실험은 train data quality + train-test distribution shift를 정량.

Hypothesis:
- H1: Train data에 anomaly-like (unstable) regions가 contaminated
  → Model이 본 patterns를 normal로 학습 → 본 type의 test anomaly에 약함
- H2: Train ↔ Test distribution shift가 큰 dataset일수록 274 model 약함
- H3: Train data의 unique pattern coverage가 낮은 dataset에서 OOD anomaly detect 가능

Goal:
- Per-dataset train quality 정량
- Train quality와 PAK-AUC F1 (baseline 274 model)의 correlation 측정
- Train quality 낮은 dataset 식별 → training-time intervention 우선순위 결정
"""
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, get_per_group
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    per_channel_points, adaptive_combine, gauss
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import pak_auc_f1
from mae_anomaly.scripts.q3_exploration.core.training_audit import (
    train_signal_stats, train_unstable_periods,
    train_test_distribution_distance, train_anomaly_density_estimate,
)

sys.path.insert(0, '/home/ykio/notebooks/claude')
from mae_anomaly.datasets.loaders import DATASET_LOADERS


def get_train_test_signals(alias, ds):
    """Load full signal and split into train/test according to ds.point_labels length."""
    if alias.startswith('smd_machine-'):
        machine = alias.replace('smd_machine-', '')
        loader_name = f'smd_simple_machine-{machine}'
    elif alias.startswith('exathlon_app'):
        app_num = alias.replace('exathlon_app', '')
        loader_name = f'exathlon_app{app_num}'
    elif alias in ('swat_full', 'swat_excl22'):
        loader_name = 'swat_A1A2'
    elif alias == 'wadi_A1':
        loader_name = 'WaDi_14days_A1'
    elif alias == 'wadi_A2':
        loader_name = 'WaDi_14days_A2'
    elif alias == 'psm':
        loader_name = 'PSM'
    elif alias == 'simulation':
        loader_name = 'simulation'
    else:
        return None, None
    if loader_name not in DATASET_LOADERS:
        return None, None
    try:
        data = DATASET_LOADERS[loader_name]()
        signals = data[0]
        test_len = len(ds.point_labels)
        full_len = len(signals)
        split = full_len - test_len
        return signals[:split], signals[split:]
    except Exception as e:
        print(f"  Error loading {alias}: {e}")
        return None, None


def main():
    print("=" * 80)
    print("P26 — Training Distribution Audit (Q3 v7)")
    print("=" * 80)

    output_dir = Path(__file__).parent.parent / 'results' / 'P26_training_audit'
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = iter_dataset_aliases()

    # ============= Stage 1: Train data statistics per dataset =============
    print("\n--- Stage 1: Train data statistics ---")

    per_dataset_train_quality = {}
    per_dataset_baseline_pak = {}
    per_dataset_n_unstable = {}
    per_dataset_distribution_shift = {}

    for alias, swat_excl22 in targets:
        ds = DatasetScores.load(alias, swat_excl22)
        if ds is None: continue
        train_sigs, test_sigs = get_train_test_signals(alias, ds)
        if train_sigs is None or len(train_sigs) < 200:
            continue

        # Baseline 274 PAK
        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        base_smoothed = gauss(base, 10)
        base_pak = pak_auc_f1(base_smoothed, ds.point_labels, ds.regions, ds.eval_mask)
        per_dataset_baseline_pak[alias] = float(base_pak)

        # Train stats
        stats = train_signal_stats(train_sigs)
        if stats is None: continue

        mean_skew = float(np.mean([abs(s['skew']) for s in stats]))
        mean_kurt = float(np.mean([abs(s['kurt']) for s in stats]))
        mean_unique = float(np.mean([s['unique_ratio'] for s in stats]))
        per_dataset_train_quality[alias] = {
            'n_train': len(train_sigs),
            'n_features': train_sigs.shape[1],
            'mean_abs_skew': mean_skew,
            'mean_abs_kurt': mean_kurt,
            'mean_unique_ratio': mean_unique,
        }

        # Unstable periods
        unstable, _ = train_unstable_periods(train_sigs, window_size=200, std_threshold=3.0)
        per_dataset_n_unstable[alias] = {
            'n_unstable_periods': len(unstable),
            'total_unstable_length': int(sum(u[1] - u[0] for u in unstable)),
            'pct_unstable': float(sum(u[1] - u[0] for u in unstable) / len(train_sigs) * 100),
            'max_instability_score': float(max([u[2] for u in unstable]) if unstable else 0),
        }

        # Distribution shift
        dist_feat = train_test_distribution_distance(train_sigs, test_sigs,
                                                       n_features_subset=20)
        if dist_feat:
            per_dataset_distribution_shift[alias] = {
                'mean_shift': float(np.mean([f['mean_shift'] for f in dist_feat])),
                'max_shift': float(np.max([f['mean_shift'] for f in dist_feat])),
                'mean_var_ratio': float(np.mean([f['var_ratio'] for f in dist_feat])),
                'mean_wasserstein': float(np.mean([f['wasserstein'] for f in dist_feat])),
                'mean_range_iou': float(np.mean([f['range_iou'] for f in dist_feat])),
            }

    print(f"  Datasets analyzed: {len(per_dataset_train_quality)}")

    # ============= Stage 2: Per-dataset summary =============
    print("\n--- Stage 2: Per-dataset train quality summary ---")
    print(f"  {'Dataset':<25s} {'Base':>6s} {'#unstable':>10s} {'%unstable':>10s} {'meanShift':>10s} {'IOU':>6s}")
    for alias in sorted(per_dataset_train_quality.keys(),
                         key=lambda x: per_dataset_baseline_pak.get(x, 0)):
        base = per_dataset_baseline_pak.get(alias, 0)
        n_unst = per_dataset_n_unstable.get(alias, {})
        shift = per_dataset_distribution_shift.get(alias, {})
        print(f"  {alias:<25s} {base:>6.3f} "
              f"{n_unst.get('n_unstable_periods', 0):>10d} "
              f"{n_unst.get('pct_unstable', 0):>10.2f}% "
              f"{shift.get('mean_shift', 0):>10.3f} "
              f"{shift.get('mean_range_iou', 0):>6.3f}")

    # ============= Stage 3: Correlation between train quality and baseline pak =============
    print("\n--- Stage 3: Correlation analysis ---")

    aliases_with_all = [a for a in per_dataset_train_quality.keys()
                         if a in per_dataset_baseline_pak
                         and a in per_dataset_n_unstable
                         and a in per_dataset_distribution_shift]

    if len(aliases_with_all) >= 5:
        baseline_arr = np.array([per_dataset_baseline_pak[a] for a in aliases_with_all])
        pct_unstable_arr = np.array([per_dataset_n_unstable[a]['pct_unstable'] for a in aliases_with_all])
        mean_shift_arr = np.array([per_dataset_distribution_shift[a]['mean_shift'] for a in aliases_with_all])
        range_iou_arr = np.array([per_dataset_distribution_shift[a]['mean_range_iou'] for a in aliases_with_all])
        mean_kurt_arr = np.array([per_dataset_train_quality[a]['mean_abs_kurt'] for a in aliases_with_all])

        corrs = {
            'pak_vs_pct_unstable': float(np.corrcoef(baseline_arr, pct_unstable_arr)[0, 1]),
            'pak_vs_mean_shift': float(np.corrcoef(baseline_arr, mean_shift_arr)[0, 1]),
            'pak_vs_range_iou': float(np.corrcoef(baseline_arr, range_iou_arr)[0, 1]),
            'pak_vs_mean_kurt': float(np.corrcoef(baseline_arr, mean_kurt_arr)[0, 1]),
        }
        print(f"  Correlations with baseline PAK-AUC F1:")
        for name, c in corrs.items():
            interp = 'strong' if abs(c) > 0.5 else 'moderate' if abs(c) > 0.3 else 'weak'
            sign = '+' if c > 0 else ''
            print(f"    {name:<25s}: r = {sign}{c:.3f}  ({interp})")

    # ============= Stage 4: Anomaly contamination estimate =============
    print("\n--- Stage 4: Train anomaly contamination estimate (top 10 datasets) ---")
    contamination_per_ds = {}
    for alias, swat_excl22 in targets[:30]:  # Limit to first 30 for time
        ds = DatasetScores.load(alias, swat_excl22)
        if ds is None: continue
        train_sigs, test_sigs = get_train_test_signals(alias, ds)
        if train_sigs is None or len(train_sigs) < 200: continue

        contam = train_anomaly_density_estimate(train_sigs, ds.regions, test_sigs,
                                                   context_size=200, similarity_threshold=0.5)
        if contam:
            contamination_per_ds[alias] = {
                'n_regions': len(contam),
                'mean_contam_ratio': float(np.mean([r['contam_ratio'] for r in contam])),
                'max_contam_ratio': float(np.max([r['contam_ratio'] for r in contam])),
                'mean_train_min': float(np.mean([r['train_min_distance'] for r in contam])),
            }

    print(f"  {'Dataset':<25s} {'n_reg':>6s} {'meanContam':>11s} {'maxContam':>11s} {'minDist':>10s}")
    for alias in sorted(contamination_per_ds.keys(),
                         key=lambda x: -contamination_per_ds[x]['mean_contam_ratio'])[:15]:
        c = contamination_per_ds[alias]
        print(f"  {alias:<25s} {c['n_regions']:>6d} "
              f"{c['mean_contam_ratio']:>11.3f} "
              f"{c['max_contam_ratio']:>11.3f} "
              f"{c['mean_train_min']:>10.3f}")

    # ============= Stage 5: Identify worst-quality train datasets =============
    print("\n--- Stage 5: Worst-quality training datasets (priority for intervention) ---")

    # Composite score: high % unstable + high mean shift + low IOU
    quality_score = {}
    for a in aliases_with_all:
        unst = per_dataset_n_unstable[a]['pct_unstable']
        shift = per_dataset_distribution_shift[a]['mean_shift']
        iou = per_dataset_distribution_shift[a]['mean_range_iou']
        # Higher = worse
        score = unst / 100.0 + shift + (1 - iou)
        quality_score[a] = score

    print("  Top 10 worst-quality train (potential candidate for retraining priority):")
    print(f"  {'Dataset':<25s} {'Quality':>9s} {'Base PAK':>9s}")
    for a in sorted(quality_score.keys(), key=lambda x: -quality_score[x])[:10]:
        print(f"  {a:<25s} {quality_score[a]:>9.3f} {per_dataset_baseline_pak[a]:>9.3f}")

    # ============= Stage 6: Save =============
    save_data = {
        'n_datasets': len(per_dataset_train_quality),
        'train_quality': per_dataset_train_quality,
        'baseline_pak': per_dataset_baseline_pak,
        'n_unstable': per_dataset_n_unstable,
        'distribution_shift': per_dataset_distribution_shift,
        'contamination_estimates': contamination_per_ds,
        'quality_score': quality_score if 'quality_score' in dir() else {},
        'correlations': corrs if 'corrs' in dir() else {},
    }
    with open(output_dir / 'P26_training_audit.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved: {output_dir / 'P26_training_audit.json'}")


if __name__ == "__main__":
    main()
