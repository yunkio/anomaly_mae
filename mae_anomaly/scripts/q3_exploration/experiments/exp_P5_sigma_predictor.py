"""
P5 — σ Predictor (Meta-Learning)

P1-P4의 발견: F5 clustering의 cluster-conditional σ가 가장 큰 leverage.
본 실험은 **continuous σ predictor**를 supervised regression으로 학습:

Inputs (per-dataset features):
- Unsupervised signature features (10 dim)
- (optional) Supervised features (median_seg, n_regions, anomaly_ratio)

Target:
- Per-dataset oracle σ multiplier (best k in {1.5, 2, 2.5, ..., 8}) — leave-one-out
- 또는 oracle pak_auc_f1을 최대화하는 σ 자체

Models:
- M1: Random Forest Regressor on unsup features
- M2: Random Forest Regressor on sup+unsup features
- M3: kNN regression (k=5)
- M4: Linear regression

Validation:
- Leave-One-Out (LOO) cross-validation: 각 dataset의 σ를 다른 38개 dataset으로 학습 후 예측
- Predicted σ로 evaluation → pak_auc_f1 delta 측정
"""
import sys
from pathlib import Path
import numpy as np
import json
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

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
from mae_anomaly.scripts.q3_exploration.core.clustering import (
    extract_signature_unsupervised, extract_signature_supervised,
)
from mae_anomaly.scripts.q3_exploration.core.postprocess import nlm_sigmoid_transform


def find_oracle_sigma(ds, base_unsmoothed, sigma_candidates):
    """Per-dataset best σ via grid search."""
    best_sigma, best_pak = None, -np.inf
    for sigma in sigma_candidates:
        smoothed = gauss(base_unsmoothed, sigma)
        pak = pak_auc_f1(smoothed, ds.point_labels, ds.regions, ds.eval_mask)
        if pak > best_pak:
            best_pak = pak
            best_sigma = sigma
    return best_sigma, best_pak


def find_oracle_sigma_nlm(ds, base_unsmoothed, sigma_candidates, T=1.5):
    """Per-dataset best σ with NLM-T fixed."""
    best_sigma, best_pak = None, -np.inf
    for sigma in sigma_candidates:
        smoothed = gauss(base_unsmoothed, sigma)
        if T is not None:
            final = nlm_sigmoid_transform(smoothed, T_factor=T)
        else:
            final = smoothed
        pak = pak_auc_f1(final, ds.point_labels, ds.regions, ds.eval_mask)
        if pak > best_pak:
            best_pak = pak
            best_sigma = sigma
    return best_sigma, best_pak


def main():
    targets = iter_dataset_aliases()
    print(f"P5 — σ Predictor Meta-Learning, {len(targets)} datasets")

    # Sigma candidates (continuous-ish)
    sigma_candidates = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300]

    # Step 1: Build dataset and extract features + oracle σ
    print("\n--- Stage 1: Feature extraction + oracle σ ---")
    aliases = []
    X_unsup = []
    X_sup_full = []
    y_oracle_no_nlm = []  # oracle σ without NLM
    y_oracle_with_nlm = []  # oracle σ with NLM T=1.5
    median_segs = []
    baseline_paks = []
    oracle_no_nlm_paks = []
    oracle_with_nlm_paks = []
    base_unsmoothed_dict = {}

    for i, (alias, swat) in enumerate(targets, 1):
        ds = DatasetScores.load(alias, swat)
        if ds is None:
            continue

        pt_r, pt_d, pt_s, pt_f = per_channel_points(ds, stride=21)
        base_unsmoothed = adaptive_combine(pt_r, pt_d, pt_f, use_fm=True)
        median_seg = median_anomaly_segment_length(ds.regions)
        base_smoothed_for_sig = gauss(base_unsmoothed, 10)

        baseline_pak = pak_auc_f1(base_smoothed_for_sig,
                                   ds.point_labels, ds.regions, ds.eval_mask)

        # Oracle σ search
        oracle_no_nlm_sigma, oracle_no_nlm_pak = find_oracle_sigma(
            ds, base_unsmoothed, sigma_candidates)
        oracle_with_nlm_sigma, oracle_with_nlm_pak = find_oracle_sigma_nlm(
            ds, base_unsmoothed, sigma_candidates, T=1.5)

        # Features
        unsup = extract_signature_unsupervised(base_smoothed_for_sig, pt_r, pt_d, pt_f)
        sup = extract_signature_supervised(ds.regions, ds.point_labels, baseline_pak)

        aliases.append(alias)
        X_unsup.append([unsup[k] for k in sorted(unsup.keys())])
        X_sup_full.append([sup[k] for k in sorted(sup.keys())] + list([unsup[k] for k in sorted(unsup.keys())]))
        y_oracle_no_nlm.append(oracle_no_nlm_sigma)
        y_oracle_with_nlm.append(oracle_with_nlm_sigma)
        median_segs.append(median_seg)
        baseline_paks.append(baseline_pak)
        oracle_no_nlm_paks.append(oracle_no_nlm_pak)
        oracle_with_nlm_paks.append(oracle_with_nlm_pak)
        base_unsmoothed_dict[alias] = (base_unsmoothed, median_seg, ds)

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    X_unsup = np.array(X_unsup)
    X_sup_full = np.array(X_sup_full)
    y_no_nlm = np.log(np.array(y_oracle_no_nlm) + 1e-9)  # log target
    y_with_nlm = np.log(np.array(y_oracle_with_nlm) + 1e-9)
    median_segs = np.array(median_segs)
    baseline_paks = np.array(baseline_paks)
    oracle_no_nlm_paks = np.array(oracle_no_nlm_paks)
    oracle_with_nlm_paks = np.array(oracle_with_nlm_paks)

    print(f"\nFeatures: X_unsup={X_unsup.shape}, X_sup_full={X_sup_full.shape}")
    print(f"Oracle σ range (no NLM): {np.exp(y_no_nlm).min():.1f} - {np.exp(y_no_nlm).max():.1f}")
    print(f"Oracle σ range (with NLM T=1.5): {np.exp(y_with_nlm).min():.1f} - {np.exp(y_with_nlm).max():.1f}")

    # Step 2: Fit models with LOO cross-validation
    print("\n--- Stage 2: LOO Cross-Validation ---")

    models_to_test = {
        'M1_rf_unsup': lambda: ('rf', X_unsup, RandomForestRegressor(n_estimators=100, random_state=42)),
        'M2_rf_sup_full': lambda: ('rf', X_sup_full, RandomForestRegressor(n_estimators=100, random_state=42)),
        'M3_knn_unsup': lambda: ('knn', X_unsup, KNeighborsRegressor(n_neighbors=5)),
        'M4_ridge_unsup': lambda: ('ridge', X_unsup, Ridge(alpha=1.0)),
        'M5_log_med_seg': lambda: ('linear', np.log(median_segs.reshape(-1, 1) + 1), Ridge(alpha=0.1)),
    }

    # Heuristic baseline: median_seg / 3 and median_seg / 5
    print(f"\nHeuristic σ baselines:")
    for div in [3, 5]:
        predicted_sigmas = median_segs / div
        # Eval each dataset
        eval_paks_no_nlm = []
        eval_paks_with_nlm = []
        for alias, sigma_pred in zip(aliases, predicted_sigmas):
            base_unsmoothed, _, ds = base_unsmoothed_dict[alias]
            sm = gauss(base_unsmoothed, max(sigma_pred, 0.5))
            eval_paks_no_nlm.append(pak_auc_f1(sm, ds.point_labels, ds.regions, ds.eval_mask))
            nlm_sm = nlm_sigmoid_transform(sm, T_factor=1.5)
            eval_paks_with_nlm.append(pak_auc_f1(nlm_sm, ds.point_labels, ds.regions, ds.eval_mask))

        d_no = np.array(eval_paks_no_nlm) - baseline_paks
        d_with = np.array(eval_paks_with_nlm) - baseline_paks
        print(f"  div={div}: no_nlm meanΔ={d_no.mean():+.4f} (vs oracle {(oracle_no_nlm_paks - baseline_paks).mean():+.4f})  "
              f"with_nlm meanΔ={d_with.mean():+.4f} (vs oracle {(oracle_with_nlm_paks - baseline_paks).mean():+.4f})")

    # Each model: LOO predict σ, evaluate
    all_model_results = {}
    target_options = [
        ('no_nlm', y_no_nlm, oracle_no_nlm_paks, False),
        ('with_nlm', y_with_nlm, oracle_with_nlm_paks, True),
    ]

    for target_name, y, oracle_paks, use_nlm in target_options:
        print(f"\n--- Target: oracle σ ({target_name}) ---")
        for model_name, model_factory in models_to_test.items():
            model_type, X, model_template = model_factory()
            # Scale X
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X) if model_type != 'linear' else X

            # LOO
            loo = LeaveOneOut()
            predicted_sigmas = np.zeros(len(aliases))
            for train_idx, test_idx in loo.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train = y[train_idx]
                # Refit fresh model
                if model_type == 'rf':
                    m = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_type == 'knn':
                    m = KNeighborsRegressor(n_neighbors=5)
                elif model_type == 'ridge':
                    m = Ridge(alpha=1.0)
                elif model_type == 'linear':
                    m = Ridge(alpha=0.1)
                m.fit(X_train, y_train)
                pred_log = m.predict(X_test)
                predicted_sigmas[test_idx[0]] = np.exp(pred_log[0]) - 1e-9
                if predicted_sigmas[test_idx[0]] < 0.5:
                    predicted_sigmas[test_idx[0]] = 0.5

            # Evaluate predicted σ
            eval_paks = []
            for alias, sigma_pred in zip(aliases, predicted_sigmas):
                base_unsmoothed, _, ds = base_unsmoothed_dict[alias]
                sm = gauss(base_unsmoothed, sigma_pred)
                if use_nlm:
                    sm = nlm_sigmoid_transform(sm, T_factor=1.5)
                eval_paks.append(pak_auc_f1(sm, ds.point_labels, ds.regions, ds.eval_mask))

            deltas = np.array(eval_paks) - baseline_paks
            mean_d = float(deltas.mean())
            wins = int((deltas > 0).sum())
            losses = int((deltas < 0).sum())
            cata = int((deltas < -0.05).sum())
            p = wilcoxon_test(eval_paks, baseline_paks, alternative='greater')

            oracle_gap = (oracle_paks - baseline_paks).mean() - mean_d
            oracle_capture = mean_d / (oracle_paks - baseline_paks).mean() * 100 if (oracle_paks - baseline_paks).mean() > 0 else 0

            print(f"  {model_name:<20s}: meanΔ={mean_d:+.4f}  W/L={wins:2d}/{losses:2d}  cata={cata}  p={p:.4f}  "
                  f"oracle_gap={oracle_gap:+.4f} ({oracle_capture:.1f}% of oracle)")

            all_model_results[f'{target_name}_{model_name}'] = {
                'mean_delta': mean_d, 'wins': wins, 'losses': losses, 'cata': cata,
                'p_value': p, 'oracle_capture_pct': oracle_capture,
                'predicted_sigmas': predicted_sigmas.tolist(),
            }

    # Step 3: Hybrid — use heuristic + small ML correction
    print("\n--- Stage 3: Heuristic + Residual Learning ---")

    for div in [3, 5]:
        for use_nlm in [False, True]:
            heuristic_sigmas = median_segs / div
            heuristic_log_sigmas = np.log(heuristic_sigmas + 1e-9)
            y = y_with_nlm if use_nlm else y_no_nlm
            residuals = y - heuristic_log_sigmas

            # Train RF on residual
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_unsup)
            loo = LeaveOneOut()
            corrected_sigmas = np.zeros(len(aliases))
            for train_idx, test_idx in loo.split(X_scaled):
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X_scaled[train_idx], residuals[train_idx])
                pred_residual = rf.predict(X_scaled[test_idx])[0]
                corrected_log = heuristic_log_sigmas[test_idx[0]] + pred_residual
                corrected_sigmas[test_idx[0]] = max(np.exp(corrected_log) - 1e-9, 0.5)

            # Eval
            eval_paks = []
            for alias, sigma_pred in zip(aliases, corrected_sigmas):
                base_unsmoothed, _, ds = base_unsmoothed_dict[alias]
                sm = gauss(base_unsmoothed, sigma_pred)
                if use_nlm:
                    sm = nlm_sigmoid_transform(sm, T_factor=1.5)
                eval_paks.append(pak_auc_f1(sm, ds.point_labels, ds.regions, ds.eval_mask))

            deltas = np.array(eval_paks) - baseline_paks
            mean_d = float(deltas.mean())
            wins = int((deltas > 0).sum())
            p = wilcoxon_test(eval_paks, baseline_paks, alternative='greater')
            print(f"  Heuristic(div={div}, NLM={use_nlm}) + ML residual: meanΔ={mean_d:+.4f}  W={wins}/{len(aliases)-wins}  p={p:.4f}")

    # Save
    out = Path(__file__).parent.parent / 'results' / 'P5_sigma_predictor.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'aliases': aliases,
            'oracle_no_nlm_sigmas': [float(s) for s in np.exp(y_no_nlm)],
            'oracle_with_nlm_sigmas': [float(s) for s in np.exp(y_with_nlm)],
            'baseline_paks': baseline_paks.tolist(),
            'oracle_no_nlm_paks': oracle_no_nlm_paks.tolist(),
            'oracle_with_nlm_paks': oracle_with_nlm_paks.tolist(),
            'model_results': all_model_results,
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
