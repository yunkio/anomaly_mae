"""
P10 — Stacking Meta-Learner

P5에서 simple log linear regression (M5)이 best (+0.0197).
본 P10은 stacked ensemble로 추가 leverage:

1. 8개 base learners (heuristic, linear, RF, KNN, multiple kernel)
2. Out-of-fold predictions으로 meta-features 생성 (LOO + nested CV)
3. Meta-learner (linear, ridge, RF small)가 base predictions를 조합
4. Final ensemble pak_auc_f1 evaluation

Improvement over P5:
- P5: single model selection (M5 best)
- P10: weighted combination of multiple models
- Meta-learner가 어떤 model을 trust할지 learn
"""
import sys
from pathlib import Path
import numpy as np
import json

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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.svm import SVR


def build_base_learners():
    """다양한 base learner 정의."""
    return {
        'ridge_a1': lambda: Ridge(alpha=1.0),
        'ridge_a10': lambda: Ridge(alpha=10.0),
        'lasso': lambda: Lasso(alpha=0.05),
        'elastic': lambda: ElasticNet(alpha=0.05, l1_ratio=0.5),
        'rf_50': lambda: RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42),
        'rf_100': lambda: RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        'gbm': lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
        'knn_3': lambda: KNeighborsRegressor(n_neighbors=3),
        'knn_5': lambda: KNeighborsRegressor(n_neighbors=5),
        'svr_rbf': lambda: SVR(kernel='rbf', C=1.0, gamma='auto'),
    }


def get_meta_learners():
    return {
        'meta_ridge': lambda: Ridge(alpha=1.0),
        'meta_lasso': lambda: Lasso(alpha=0.01),
        'meta_lr_constrained': 'constrained_linear',  # weights >= 0, sum=1
    }


def constrained_blender(base_preds_train, y_train, base_preds_test):
    """Non-negative constrained linear blend (weights >= 0, sum = 1)."""
    from scipy.optimize import minimize
    n_base = base_preds_train.shape[1]

    def loss(w):
        return ((base_preds_train @ w - y_train) ** 2).mean()

    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1.0},
    ]
    bounds = [(0, 1)] * n_base
    x0 = np.ones(n_base) / n_base

    try:
        res = minimize(loss, x0, bounds=bounds, constraints=constraints,
                        method='SLSQP', options={'maxiter': 200})
        weights = res.x
    except Exception:
        weights = x0
    return base_preds_test @ weights, weights


def find_oracle_sigma_with_nlm(ds, base_unsmoothed, sigma_candidates, T=1.5):
    best_sigma, best_pak = None, -np.inf
    for sigma in sigma_candidates:
        smoothed = gauss(base_unsmoothed, sigma)
        if T is not None:
            final = nlm_sigmoid_transform(smoothed, T_factor=T)
        else:
            final = smoothed
        pak = pak_auc_f1(final, ds.point_labels, ds.regions, ds.eval_mask)
        if pak > best_pak:
            best_pak, best_sigma = pak, sigma
    return best_sigma, best_pak


def main():
    targets = iter_dataset_aliases()
    print(f"P10 — Stacking Meta-Learner, {len(targets)} datasets")

    sigma_candidates = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300]

    # Stage 1: Build dataset
    print("\n--- Stage 1: Feature extraction + oracle σ ---")
    aliases = []
    X_sup_full = []  # supervised + unsupervised features
    X_unsup = []
    y_oracle = []  # log(oracle σ with NLM T=1.5)
    base_unsmoothed_dict = {}
    baseline_paks = []
    oracle_paks = []
    median_segs = []

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
        oracle_sigma, oracle_pak = find_oracle_sigma_with_nlm(
            ds, base_unsmoothed, sigma_candidates, T=1.5)

        unsup = extract_signature_unsupervised(base_smoothed_for_sig, pt_r, pt_d, pt_f)
        sup = extract_signature_supervised(ds.regions, ds.point_labels, baseline_pak)

        aliases.append(alias)
        feat_unsup = sorted(unsup.items())
        feat_sup = sorted(sup.items())

        X_unsup.append([v for _, v in feat_unsup])
        X_sup_full.append([v for _, v in feat_unsup] + [v for _, v in feat_sup])
        y_oracle.append(np.log(oracle_sigma + 1e-9))
        baseline_paks.append(baseline_pak)
        oracle_paks.append(oracle_pak)
        median_segs.append(median_seg)
        base_unsmoothed_dict[alias] = (base_unsmoothed, median_seg, ds)

        if i % 10 == 0 or i == len(targets):
            print(f"  [{i:2d}/{len(targets)}] processed", flush=True)

    X_unsup = np.array(X_unsup)
    X_sup_full = np.array(X_sup_full)
    y = np.array(y_oracle)
    baseline_paks = np.array(baseline_paks)
    oracle_paks = np.array(oracle_paks)
    median_segs = np.array(median_segs)

    print(f"\nFeatures: X_unsup={X_unsup.shape}, X_sup_full={X_sup_full.shape}")

    # Stage 2: Generate out-of-fold predictions for base learners
    print("\n--- Stage 2: Base learner OOF predictions (LOO) ---")
    n = len(aliases)
    base_learners = build_base_learners()
    base_preds_all = {name: np.zeros(n) for name in base_learners}

    # Use X_sup_full as input for all learners
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sup_full)

    loo = LeaveOneOut()
    for fold_i, (train_idx, test_idx) in enumerate(loo.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y[train_idx]
        for name, learner_factory in base_learners.items():
            try:
                model = learner_factory()
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                base_preds_all[name][test_idx[0]] = pred[0]
            except Exception as e:
                base_preds_all[name][test_idx[0]] = np.log(median_segs[test_idx[0]] / 5 + 1e-9)
        if (fold_i + 1) % 10 == 0:
            print(f"  LOO fold {fold_i+1}/{n}", flush=True)

    # Also add simple heuristics as base learners
    base_preds_all['heuristic_log_med_seg_div5'] = np.log(np.maximum(median_segs / 5, 0.5))
    base_preds_all['heuristic_log_med_seg_div3'] = np.log(np.maximum(median_segs / 3, 0.5))
    base_preds_all['heuristic_log_med_seg_div7'] = np.log(np.maximum(median_segs / 7, 0.5))

    print(f"\nN base predictors: {len(base_preds_all)}")

    # Evaluate each base learner
    print("\n--- Stage 3: Individual base learner evaluation ---")
    base_summary = {}
    for name, preds in base_preds_all.items():
        predicted_sigmas = np.exp(preds) - 1e-9
        predicted_sigmas = np.maximum(predicted_sigmas, 0.5)

        eval_paks = []
        for j, alias in enumerate(aliases):
            base_unsmoothed, _, ds = base_unsmoothed_dict[alias]
            smoothed = gauss(base_unsmoothed, predicted_sigmas[j])
            final = nlm_sigmoid_transform(smoothed, T_factor=1.5)
            eval_paks.append(pak_auc_f1(final, ds.point_labels, ds.regions, ds.eval_mask))
        eval_paks = np.array(eval_paks)
        deltas = eval_paks - baseline_paks
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(eval_paks.tolist(), baseline_paks.tolist(), alternative='greater')
        oracle_capture = mean_d / (oracle_paks - baseline_paks).mean() * 100

        base_summary[name] = {
            'mean_delta': mean_d, 'wins': wins, 'losses': losses,
            'cata': cata, 'p_value': p, 'oracle_capture_pct': float(oracle_capture),
        }

    sorted_base = sorted(base_summary.items(), key=lambda x: -x[1]['mean_delta'])
    print(f"\n{'Base learner':<35s} {'meanΔ':>10s} {'W/L':>9s} {'cata':>5s} {'p':>8s} {'capture':>9s}")
    for name, s in sorted_base:
        print(f"{name:<35s} {s['mean_delta']:>+10.4f} {s['wins']:>2d}/{s['losses']:<2d}      {s['cata']:>5d} {s['p_value']:>8.4f} {s['oracle_capture_pct']:>8.1f}%")

    # Stage 4: Stacking meta-learners
    print("\n--- Stage 4: Meta-learner stacking ---")
    base_preds_matrix = np.column_stack([base_preds_all[name]
                                          for name in base_preds_all.keys()])
    base_names = list(base_preds_all.keys())
    print(f"Base prediction matrix: {base_preds_matrix.shape}")

    # LOO for meta-learner
    meta_results = {}
    for meta_name in ['ridge_a1', 'ridge_a10', 'lasso_001', 'constrained_blend', 'simple_mean']:
        final_preds = np.zeros(n)
        weights_history = []

        for train_idx, test_idx in loo.split(base_preds_matrix):
            X_meta_train = base_preds_matrix[train_idx]
            X_meta_test = base_preds_matrix[test_idx]
            y_train_meta = y[train_idx]

            if meta_name == 'ridge_a1':
                m = Ridge(alpha=1.0); m.fit(X_meta_train, y_train_meta)
                final_preds[test_idx[0]] = m.predict(X_meta_test)[0]
                weights_history.append(m.coef_)
            elif meta_name == 'ridge_a10':
                m = Ridge(alpha=10.0); m.fit(X_meta_train, y_train_meta)
                final_preds[test_idx[0]] = m.predict(X_meta_test)[0]
                weights_history.append(m.coef_)
            elif meta_name == 'lasso_001':
                m = Lasso(alpha=0.01); m.fit(X_meta_train, y_train_meta)
                final_preds[test_idx[0]] = m.predict(X_meta_test)[0]
                weights_history.append(m.coef_)
            elif meta_name == 'constrained_blend':
                pred, w = constrained_blender(X_meta_train, y_train_meta, X_meta_test)
                final_preds[test_idx[0]] = pred[0]
                weights_history.append(w)
            elif meta_name == 'simple_mean':
                final_preds[test_idx[0]] = X_meta_test.mean(axis=1)[0]
                weights_history.append(np.ones(len(base_names)) / len(base_names))

        # Eval
        predicted_sigmas = np.maximum(np.exp(final_preds) - 1e-9, 0.5)
        eval_paks = []
        for j, alias in enumerate(aliases):
            base_unsmoothed, _, ds = base_unsmoothed_dict[alias]
            smoothed = gauss(base_unsmoothed, predicted_sigmas[j])
            final = nlm_sigmoid_transform(smoothed, T_factor=1.5)
            eval_paks.append(pak_auc_f1(final, ds.point_labels, ds.regions, ds.eval_mask))
        eval_paks = np.array(eval_paks)
        deltas = eval_paks - baseline_paks
        mean_d = float(deltas.mean())
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        cata = int((deltas < -0.05).sum())
        p = wilcoxon_test(eval_paks.tolist(), baseline_paks.tolist(), alternative='greater')
        capture = mean_d / (oracle_paks - baseline_paks).mean() * 100

        # Average weights across folds (for interpretation)
        avg_weights = np.array(weights_history).mean(axis=0)

        meta_results[meta_name] = {
            'mean_delta': mean_d, 'wins': wins, 'losses': losses,
            'cata': cata, 'p_value': p, 'oracle_capture_pct': float(capture),
            'avg_weights': avg_weights.tolist(),
        }

    print(f"\n{'Meta-learner':<22s} {'meanΔ':>10s} {'W/L':>9s} {'cata':>5s} {'p':>8s} {'capture':>9s}")
    for name, s in sorted(meta_results.items(), key=lambda x: -x[1]['mean_delta']):
        print(f"{name:<22s} {s['mean_delta']:>+10.4f} {s['wins']:>2d}/{s['losses']:<2d}      {s['cata']:>5d} {s['p_value']:>8.4f} {s['oracle_capture_pct']:>8.1f}%")

    # Best meta-learner — print weights
    best_meta = max(meta_results, key=lambda x: meta_results[x]['mean_delta'])
    print(f"\n=== Best meta-learner: {best_meta} ===")
    print(f"  mean Δ: {meta_results[best_meta]['mean_delta']:+.4f}")
    print(f"  W/L: {meta_results[best_meta]['wins']}/{meta_results[best_meta]['losses']}")
    print(f"  p: {meta_results[best_meta]['p_value']:.4f}")

    print(f"\n  Average weights:")
    weights = meta_results[best_meta]['avg_weights']
    for name, w in sorted(zip(base_names, weights), key=lambda x: -abs(x[1]))[:10]:
        print(f"    {name:<35s}: {w:+.4f}")

    # Save
    out = Path(__file__).parent.parent / 'results' / 'P10_stacking.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'base_summary': base_summary,
            'meta_results': meta_results,
            'best_meta': best_meta,
            'aliases': aliases,
            'oracle_paks': oracle_paks.tolist(),
            'baseline_paks': baseline_paks.tolist(),
            'base_names': base_names,
        }, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
