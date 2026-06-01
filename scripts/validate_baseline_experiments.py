#!/usr/bin/env python3
"""Comprehensive baseline experiment verification.

Checks all 280 experiment directories for:
1. File/directory completeness
2. epoch_metrics.json content validity (NaN, Inf, all-zero, frozen values)
3. scores.npz content validity
4. Cross-queue sanity checks
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

BASE = Path('/home/ykio/notebooks/claude/comparison/results/experiments')
QUEUES = {
    'Q1': '1_20260312_041500_baseline_minmax',
    'Q2': '2_20260312_144923_baseline_zscore',
    'Q3': '3_20260312_203923_baseline_minmax_normalonly',
    'Q4': '4_20260313_020446_baseline_zscore_normalonly',
}
DATASETS = ['simulation/simulation', 'SWaT/A1A2_full', 'SWaT/A1A2_excl22', 'WaDi/A1', 'WaDi/A2']
ALL_MODELS = ['random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance',
              'mlp', 'mlpmixer', 'transformer', 'gcn_lstm',
              'anomaly_transformer', 'tranad', 'usad', 'dagmm', 'gdn', 'omnianomaly']
SIM_MODELS = ['random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance',
              'mlp', 'mlpmixer', 'transformer', 'gcn_lstm', 'anomaly_transformer']
STAT_MODELS = {'random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance'}

# Expected test lengths for scores.npz (verified from actual data)
EXPECTED_TEST_LENGTHS = {
    'simulation/simulation': 55000,
    'SWaT/A1A2_full': 224960,
    'WaDi/A1': 86401,
    'WaDi/A2': 86402,
}

# DL visualization expected files
VIZ_EPOCH_FILES = {'epoch_dashboard.png', 'epoch_f1_t.png', 'epoch_pa_k_f1.png', 'epoch_pak_auc.png', 'epoch_prc_auc.png'}
VIZ_BEST_FILES = {'best_model_prc_curve.png'}

# Key metrics to check
KEY_METRICS = ['roc_auc', 'prc_auc', 'f1_score', 'pak_auc_f1', 'precision', 'recall']


def check_epoch_metrics(filepath, expected_epochs, qname, ds, model_name):
    """Check epoch_metrics.json for validity."""
    issues = []
    try:
        data = json.load(open(filepath))
    except Exception as e:
        return [f"PARSE_ERROR: {e}"]

    epochs = data.get('epochs', [])
    eval_interval = data.get('eval_interval', 1)

    if len(epochs) != expected_epochs:
        issues.append(f"EPOCH_COUNT: {len(epochs)} (expected {expected_epochs})")

    for i, ep in enumerate(epochs):
        ep_num = i + 1

        # Check for NaN/Inf in all numeric values
        for key, val in ep.items():
            if key.startswith('_'):
                continue
            if isinstance(val, float):
                if np.isnan(val):
                    issues.append(f"NaN: epoch {ep_num}, {key}")
                elif np.isinf(val):
                    issues.append(f"Inf: epoch {ep_num}, {key}")

    # Check for frozen metrics across epochs (only meaningful for multi-epoch)
    if len(epochs) > 1:
        for metric in KEY_METRICS:
            values = [ep.get(metric) for ep in epochs if ep.get(metric) is not None]
            if len(values) > 1:
                unique = set(f"{v:.10f}" for v in values)
                if len(unique) == 1:
                    issues.append(f"FROZEN: {metric} = {values[0]:.6f} across all {len(values)} epochs")

    # Check for all-zero key metrics in best epoch
    if epochs:
        best = max(epochs, key=lambda e: e.get('pak_auc_f1', 0) or 0)
        zero_metrics = [m for m in KEY_METRICS if best.get(m) == 0.0]
        if zero_metrics and model_name != 'random':
            issues.append(f"ZERO_METRICS (best epoch): {', '.join(zero_metrics)}")

    return issues


def check_scores_npz(filepath, ds):
    """Check scores.npz for validity."""
    issues = []
    try:
        data = np.load(str(filepath))
    except Exception as e:
        return [f"LOAD_ERROR: {e}"]

    if 'anomaly_score' not in data:
        return ["MISSING_KEY: anomaly_score"]

    scores = data['anomaly_score']
    expected_len = EXPECTED_TEST_LENGTHS.get(ds)

    if expected_len and len(scores) != expected_len:
        issues.append(f"SCORE_LENGTH: {len(scores)} (expected {expected_len})")

    if np.all(scores == 0):
        issues.append(f"ALL_ZERO_SCORES: {len(scores)} values all 0")

    if np.any(np.isnan(scores)):
        n_nan = np.sum(np.isnan(scores))
        issues.append(f"NaN_SCORES: {n_nan}/{len(scores)} values are NaN")

    if np.any(np.isinf(scores)):
        n_inf = np.sum(np.isinf(scores))
        issues.append(f"Inf_SCORES: {n_inf}/{len(scores)} values are Inf")

    return issues


def check_directory(d, qname, ds, model_name):
    """Full check of a single model directory."""
    issues = []
    is_stat = model_name in STAT_MODELS
    is_excl22 = ds == 'SWaT/A1A2_excl22'
    is_dl = not is_stat
    expected_epochs = 1 if is_stat else 10

    # --- File existence ---
    em = d / 'epoch_metrics.json'
    md = d / 'metadata.json'

    if not d.exists():
        return [("CRITICAL", "DIR_MISSING")]

    if not em.exists():
        issues.append(("CRITICAL", "epoch_metrics.json MISSING"))
    else:
        metric_issues = check_epoch_metrics(em, expected_epochs, qname, ds, model_name)
        for mi in metric_issues:
            severity = "CRITICAL" if mi.startswith("PARSE_ERROR") or mi.startswith("EPOCH_COUNT") else "WARNING"
            issues.append((severity, mi))

    if not md.exists():
        issues.append(("CRITICAL", "metadata.json MISSING"))

    if is_excl22:
        # excl22: only epoch_metrics.json and metadata.json expected
        return issues

    # Non-excl22 checks
    snpz = d / 'scores.npz'
    if not snpz.exists():
        issues.append(("CRITICAL", "scores.npz MISSING"))
    else:
        score_issues = check_scores_npz(snpz, ds)
        for si in score_issues:
            severity = "WARNING" if si.startswith("ALL_ZERO") else "CRITICAL"
            issues.append((severity, si))

    if is_dl:
        # model/
        model_dir = d / 'model'
        if not model_dir.exists():
            issues.append(("CRITICAL", "model/ MISSING"))
        else:
            if not (model_dir / 'model.pt').exists():
                # Some SOTA models save differently
                has_any = any(model_dir.iterdir())
                if not has_any:
                    issues.append(("CRITICAL", "model/ EMPTY"))

        # epoch_scores/
        es_dir = d / 'epoch_scores'
        if not es_dir.exists():
            issues.append(("CRITICAL", "epoch_scores/ MISSING"))
        else:
            expected_files = {f'epoch_{i:03d}_scores.npz' for i in range(1, 11)}
            actual_files = {f.name for f in es_dir.iterdir() if f.is_file()}
            missing = expected_files - actual_files
            if missing:
                issues.append(("WARNING", f"epoch_scores/ missing {len(missing)} files: {sorted(missing)[:3]}..."))

        # visualization/
        viz_dir = d / 'visualization'
        if not viz_dir.exists():
            issues.append(("WARNING", "visualization/ MISSING"))
        else:
            # epoch_metrics/ (5 pngs)
            epoch_viz = viz_dir / 'epoch_metrics'
            if not epoch_viz.exists():
                issues.append(("WARNING", "visualization/epoch_metrics/ MISSING"))
            else:
                actual = {f.name for f in epoch_viz.iterdir()}
                missing_viz = VIZ_EPOCH_FILES - actual
                if missing_viz:
                    issues.append(("WARNING", f"visualization/epoch_metrics/ missing: {missing_viz}"))

            # best_model/
            best_viz = viz_dir / 'best_model'
            if not best_viz.exists():
                issues.append(("WARNING", "visualization/best_model/ MISSING"))
    else:
        # Statistical model - only best_model viz
        viz_dir = d / 'visualization'
        if not viz_dir.exists():
            issues.append(("WARNING", "visualization/ MISSING"))
        else:
            best_viz = viz_dir / 'best_model'
            if not best_viz.exists():
                issues.append(("WARNING", "visualization/best_model/ MISSING"))

    return issues


def main():
    all_issues = []
    total = 0
    clean = 0
    critical_count = 0
    warning_count = 0

    # Performance summary for cross-check
    perf_data = {}  # (qname, ds, model) -> best pak_auc_f1

    for qname, qdir in QUEUES.items():
        for ds in DATASETS:
            models = SIM_MODELS if ds == 'simulation/simulation' else ALL_MODELS
            for model_name in models:
                total += 1
                d = BASE / qdir / ds / model_name
                issues = check_directory(d, qname, ds, model_name)

                if issues:
                    for severity, msg in issues:
                        all_issues.append((qname, ds, model_name, severity, msg))
                        if severity == "CRITICAL":
                            critical_count += 1
                        else:
                            warning_count += 1
                else:
                    clean += 1

                # Collect performance for cross-check
                em = d / 'epoch_metrics.json'
                if em.exists():
                    try:
                        data = json.load(open(em))
                        epochs = data.get('epochs', [])
                        if epochs:
                            best_pak = max(ep.get('pak_auc_f1', 0) or 0 for ep in epochs)
                            perf_data[(qname, ds, model_name)] = best_pak
                    except:
                        pass

    # === Print Report ===
    print("=" * 80)
    print(f"  BASELINE EXPERIMENT VERIFICATION REPORT")
    print(f"  Checked: {total} directories")
    print(f"  Clean: {clean}, With issues: {total - clean}")
    print(f"  Critical: {critical_count}, Warnings: {warning_count}")
    print("=" * 80)

    if all_issues:
        print("\n--- ISSUES ---\n")
        # Group by severity
        for severity in ["CRITICAL", "WARNING"]:
            items = [(q, ds, m, s, msg) for q, ds, m, s, msg in all_issues if s == severity]
            if items:
                print(f"\n[{severity}] ({len(items)} issues)")
                print("-" * 60)
                for q, ds, m, s, msg in items:
                    print(f"  {q}/{ds}/{m}: {msg}")

    # Cross-queue sanity check
    print("\n\n--- CROSS-QUEUE PERFORMANCE SUMMARY (best pak_auc_f1) ---\n")
    print(f"{'Model':<22} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8}   Flags")
    print("-" * 80)

    for ds in ['simulation/simulation', 'SWaT/A1A2_full', 'WaDi/A1', 'WaDi/A2']:
        models = SIM_MODELS if ds == 'simulation/simulation' else ALL_MODELS
        ds_short = ds.split('/')[-1]
        print(f"\n  [{ds_short}]")
        for model_name in models:
            vals = []
            flags = []
            for qname in QUEUES:
                v = perf_data.get((qname, ds, model_name))
                vals.append(f"{v:.4f}" if v is not None else "  N/A ")

            # Flag if random outperforms a DL model
            for qname in QUEUES:
                rand_v = perf_data.get((qname, ds, 'random'), 0)
                v = perf_data.get((qname, ds, model_name), 0)
                if model_name not in STAT_MODELS and v > 0 and v < rand_v * 0.9:
                    flags.append(f"{qname}<random")

            flag_str = ", ".join(flags) if flags else ""
            print(f"  {model_name:<20} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8}   {flag_str}")

    # excl22 summary
    print(f"\n\n--- SWaT/A1A2_excl22 PERFORMANCE ---\n")
    print(f"{'Model':<22} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8}")
    print("-" * 60)
    for model_name in ALL_MODELS:
        vals = []
        for qname in QUEUES:
            v = perf_data.get((qname, 'SWaT/A1A2_excl22', model_name))
            vals.append(f"{v:.4f}" if v is not None else "  N/A ")
        print(f"  {model_name:<20} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8}")

    print("\n" + "=" * 80)
    print(f"VERDICT: {critical_count} CRITICAL, {warning_count} WARNING")
    if critical_count == 0:
        print("All experiments structurally complete.")
    print("=" * 80)

    return critical_count


if __name__ == '__main__':
    sys.exit(main())
