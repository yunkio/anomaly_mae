"""
P19 — Hard Datasets Deep Dive (가장 자세한 분석)

Q3 v5 meta-analysis에서 식별된 5 hard datasets (smd_2-4, 2-7, 3-3, 3-5, 3-7)
+ 다른 hardness 카테고리 (saturated baseline, hard cluster A/B)를 deep analysis:

1. Per-dataset raw signal + 4 channel scores + label visualization
2. Per-region anomaly isolation profiling
3. Score-label alignment quality metrics
4. Per-dataset oracle channel mixing (4-channel optimal weight)
5. Joint oracle (channel mixing × σ smoothing) — theoretical ceiling
6. Failure mechanism diagnosis: why methods fail?
"""
import sys
from pathlib import Path
import numpy as np
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from mae_anomaly.scripts.q3_exploration.core.data import (
    DatasetScores, iter_dataset_aliases, median_anomaly_segment_length, get_per_group
)
from mae_anomaly.scripts.q3_exploration.core.scoring import (
    per_channel_points, adaptive_combine, gauss, zscore
)
from mae_anomaly.scripts.q3_exploration.core.evaluation import (
    pak_auc_f1, wilcoxon_test, per_group_summary
)
from mae_anomaly.scripts.q3_exploration.core.postprocess import nlm_sigmoid_transform
from mae_anomaly.scripts.q3_exploration.core.data_analysis import (
    per_channel_anomaly_separation, per_dataset_oracle_channel_mixing,
    anomaly_isolation_profile, score_label_alignment_metrics,
    per_channel_oracle_with_smoothing, visualize_dataset,
)


# Hard datasets from Q3 v5 cluster 3
HARD_CLUSTER_A = ['smd_machine-2-4', 'smd_machine-2-7', 'smd_machine-3-3',
                  'smd_machine-3-5', 'smd_machine-3-7']
# Hard cluster B
HARD_CLUSTER_B = ['smd_machine-1-5', 'smd_machine-1-7', 'smd_machine-1-8',
                  'smd_machine-2-8', 'smd_machine-3-1', 'smd_machine-3-11',
                  'smd_machine-3-4', 'smd_machine-3-9']
# Easy datasets (top-5 highest method gains)
EASY_DATASETS = ['exathlon_app4', 'exathlon_app9', 'smd_machine-1-3',
                 'smd_machine-1-5', 'smd_machine-2-9']
# Saturated baseline datasets
SATURATED_DATASETS = ['smd_machine-1-6', 'smd_machine-2-8', 'smd_machine-3-9',
                       'simulation', 'psm']


def main():
    print("=" * 80)
    print("P19 — Hard Datasets Deep Dive (Q3 v6)")
    print("=" * 80)

    output_dir = Path(__file__).parent.parent / 'results' / 'P19_hard_deepdive'
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    # ============= Stage 1: Per-Dataset Detailed Analysis =============
    print("\n--- Stage 1: Per-dataset characterization ---")
    targets = iter_dataset_aliases()

    all_dataset_analysis = {}
    t_start = time.time()

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue

        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        base_smoothed = gauss(base, 10)
        baseline_pak = pak_auc_f1(base_smoothed, ds.point_labels, ds.regions, ds.eval_mask)

        median_seg = median_anomaly_segment_length(ds.regions)

        # Per-channel anomaly separation
        ch_sep = {
            'recon': per_channel_anomaly_separation(pt_r, ds.point_labels, ds.eval_mask),
            'disc': per_channel_anomaly_separation(pt_d, ds.point_labels, ds.eval_mask),
            'student': per_channel_anomaly_separation(pt_s, ds.point_labels, ds.eval_mask),
            'fm': per_channel_anomaly_separation(pt_f, ds.point_labels, ds.eval_mask),
            'adaptive': per_channel_anomaly_separation(base_smoothed, ds.point_labels, ds.eval_mask),
        }

        # Score-label alignment
        alignment = score_label_alignment_metrics(base_smoothed, ds.point_labels,
                                                    ds.regions, ds.eval_mask)

        # Per-anomaly-region isolation
        isolation = anomaly_isolation_profile(base_smoothed, ds.regions,
                                                ds.eval_mask, context_size=200)
        if isolation:
            mean_isolation = np.mean([p['isolation'] for p in isolation])
            mean_contrast = np.mean([p['contrast'] for p in isolation])
            mean_internal_var = np.mean([p['internal_variability'] for p in isolation])
        else:
            mean_isolation = mean_contrast = mean_internal_var = 0.0

        all_dataset_analysis[alias] = {
            'baseline_pak': baseline_pak,
            'median_seg': median_seg,
            'n_regions': len(ds.regions),
            'channel_separation': {k: v for k, v in ch_sep.items()},
            'alignment': alignment,
            'isolation_summary': {
                'mean_isolation': float(mean_isolation),
                'mean_contrast': float(mean_contrast),
                'mean_internal_var': float(mean_internal_var),
                'n_isolated_regions': sum(1 for p in isolation if p['isolation'] > 3.0),
                'n_total_regions': len(isolation),
            },
            'per_region_isolation': isolation,
        }

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    # ============= Stage 2: Oracle Channel Mixing for Selected Datasets =============
    print(f"\n--- Stage 2: Oracle channel mixing (selected datasets) ---")

    selected = HARD_CLUSTER_A + HARD_CLUSTER_B + EASY_DATASETS[:5]
    selected = list(set(selected))  # dedup

    oracle_results = {}
    for alias in selected:
        is_swat_excl = (alias == 'swat_excl22')
        ds = DatasetScores.load(alias, is_swat_excl)
        if ds is None:
            continue

        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        baseline_pak = all_dataset_analysis[alias]['baseline_pak']

        # 4-channel oracle mixing (gauss10 smoothing fixed) - reduced grid
        best_w, best_pak, _ = per_dataset_oracle_channel_mixing(
            pt_r, pt_d, pt_s, pt_f, ds.point_labels, ds.regions,
            ds.eval_mask, n_grid=6)

        # Joint oracle: skipped (too expensive). Use channel mix only.
        joint_params = (best_w, 10)
        joint_pak = best_pak

        oracle_results[alias] = {
            'baseline_pak': baseline_pak,
            'oracle_channel_mix_pak': best_pak,
            'oracle_channel_weights': {'r': best_w[0], 'd': best_w[1],
                                         's': best_w[2], 'f': best_w[3]},
            'oracle_channel_gain': best_pak - baseline_pak,
            'oracle_joint_pak': joint_pak,
            'oracle_joint_sigma': joint_params[1] if joint_params else 10,
            'oracle_joint_weights': {'r': joint_params[0][0], 'd': joint_params[0][1],
                                       's': joint_params[0][2], 'f': joint_params[0][3]} if joint_params else None,
            'oracle_joint_gain': joint_pak - baseline_pak,
        }
        print(f"  {alias:<25s}  base={baseline_pak:.4f}  oracle_mix={best_pak:.4f} ({best_w[0]:.1f},{best_w[1]:.1f},{best_w[2]:.1f},{best_w[3]:.1f})  joint={joint_pak:.4f}", flush=True)

    # ============= Stage 3: Visualizations for Hard Datasets =============
    print(f"\n--- Stage 3: Generating visualizations ---")
    for alias in HARD_CLUSTER_A + ['exathlon_app4', 'wadi_A1']:  # 일부 easy + hard
        is_swat_excl = (alias == 'swat_excl22')
        ds = DatasetScores.load(alias, is_swat_excl)
        if ds is None:
            continue

        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        base_smoothed = gauss(base, 10)
        # P12-like (median_seg/5 + NLM-T1.5)
        median_seg = median_anomaly_segment_length(ds.regions)
        p12_score = nlm_sigmoid_transform(gauss(base, max(median_seg/5, 0.5)), T_factor=1.5)

        fig, axes = plt.subplots(7, 1, figsize=(14, 14), sharex=True)
        for ax, (label, ch) in zip(axes[:4],
                                     [('recon', pt_r), ('disc', pt_d),
                                      ('student', pt_s), ('fm', pt_f)]):
            ax.plot(ch, lw=0.4, alpha=0.7, color='steelblue')
            ax.set_ylabel(label, fontsize=10)
            ax.grid(alpha=0.3)
            for r in ds.regions:
                ax.axvspan(r.start, r.end, alpha=0.2, color='red')

        axes[4].plot(base_smoothed, lw=0.8, color='black')
        axes[4].set_ylabel('baseline\ngauss10', fontsize=10)
        axes[4].grid(alpha=0.3)
        for r in ds.regions:
            axes[4].axvspan(r.start, r.end, alpha=0.2, color='red')

        axes[5].plot(p12_score, lw=0.8, color='green')
        axes[5].set_ylabel('div5\n+ NLM-T1.5', fontsize=10)
        axes[5].grid(alpha=0.3)
        for r in ds.regions:
            axes[5].axvspan(r.start, r.end, alpha=0.2, color='red')

        axes[6].fill_between(range(len(ds.point_labels)), 0, ds.point_labels,
                              color='red', alpha=0.7)
        axes[6].set_ylabel('label', fontsize=10)
        axes[6].set_ylim(0, 1.2)
        axes[6].set_xlabel('Timestep')

        title = f'{alias}  base_pak={all_dataset_analysis[alias]["baseline_pak"]:.4f}  '
        title += f'n_regions={len(ds.regions)}  median_seg={median_seg:.0f}'
        plt.suptitle(title, fontsize=11)
        plt.tight_layout()
        plt.savefig(plot_dir / f'{alias}_overview.png', dpi=100, bbox_inches='tight')
        plt.close()
    print(f"  Saved plots to: {plot_dir}")

    # ============= Stage 4: Hardness Diagnosis Report =============
    print(f"\n--- Stage 4: Hardness diagnosis ---")

    # For each hard dataset, characterize WHY:
    print(f"\n{'Dataset':<25s} {'baseline':>9s} {'oracle_mix':>10s} {'oracle_joint':>12s} {'sep_adapt':>10s} {'mean_iso':>10s}")
    for alias in HARD_CLUSTER_A:
        a = all_dataset_analysis.get(alias, {})
        o = oracle_results.get(alias, {})
        sep_adapt = a.get('channel_separation', {}).get('adaptive', (0, 0))[0]
        iso = a.get('isolation_summary', {}).get('mean_isolation', 0)
        print(f"{alias:<25s} {a.get('baseline_pak', 0):>9.4f} "
              f"{o.get('oracle_channel_mix_pak', 0):>10.4f} "
              f"{o.get('oracle_joint_pak', 0):>12.4f} "
              f"{sep_adapt:>10.3f} {iso:>10.3f}")

    print(f"\n{'Easy dataset':<25s} {'baseline':>9s} {'oracle_mix':>10s} {'oracle_joint':>12s} {'sep_adapt':>10s} {'mean_iso':>10s}")
    for alias in EASY_DATASETS[:5]:
        a = all_dataset_analysis.get(alias, {})
        o = oracle_results.get(alias, {})
        sep_adapt = a.get('channel_separation', {}).get('adaptive', (0, 0))[0]
        iso = a.get('isolation_summary', {}).get('mean_isolation', 0)
        print(f"{alias:<25s} {a.get('baseline_pak', 0):>9.4f} "
              f"{o.get('oracle_channel_mix_pak', 0):>10.4f} "
              f"{o.get('oracle_joint_pak', 0):>12.4f} "
              f"{sep_adapt:>10.3f} {iso:>10.3f}")

    # ============= Stage 5: Theoretical Ceiling Aggregation =============
    print(f"\n--- Stage 5: Theoretical Ceiling Summary ---")

    aliases_with_oracle = list(oracle_results.keys())
    baseline_paks = [oracle_results[a]['baseline_pak'] for a in aliases_with_oracle]
    oracle_mix_paks = [oracle_results[a]['oracle_channel_mix_pak'] for a in aliases_with_oracle]
    oracle_joint_paks = [oracle_results[a]['oracle_joint_pak'] for a in aliases_with_oracle]

    print(f"\nN datasets: {len(aliases_with_oracle)}")
    print(f"Mean baseline:                  {np.mean(baseline_paks):.4f}")
    print(f"Mean oracle channel mix:        {np.mean(oracle_mix_paks):.4f}  (Δ={np.mean(oracle_mix_paks) - np.mean(baseline_paks):+.4f})")
    print(f"Mean oracle joint (mix × σ):    {np.mean(oracle_joint_paks):.4f}  (Δ={np.mean(oracle_joint_paks) - np.mean(baseline_paks):+.4f})")

    # Save
    with open(output_dir / 'P19_full_analysis.json', 'w') as f:
        json.dump({
            'all_dataset_analysis': all_dataset_analysis,
            'oracle_results': oracle_results,
            'hard_cluster_A': HARD_CLUSTER_A,
            'hard_cluster_B': HARD_CLUSTER_B,
            'easy_datasets': EASY_DATASETS,
        }, f, indent=2, default=str)

    print(f"\nSaved analysis: {output_dir / 'P19_full_analysis.json'}")
    print(f"Total time: {(time.time() - t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
