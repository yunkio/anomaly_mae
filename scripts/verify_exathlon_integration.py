"""Exathlon dataset integration consistency verification — MAE + Comparison.

Run after any Exathlon-related change to ensure all touchpoints are consistent.

Usage:
    python scripts/verify_exathlon_integration.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def check(label, cond, details=""):
    status = "✅" if cond else "❌"
    print(f"  {status} {label}" + (f" — {details}" if details else ""))
    return cond


def main():
    print("=" * 70)
    print("Exathlon integration verification")
    print("=" * 70)

    all_ok = True

    # 1. MAE loader
    print("\n[1/8] MAE loader (mae_anomaly/datasets/loaders.py)")
    from mae_anomaly.datasets.loaders import (
        load_exathlon, EXATHLON_APP_IDS, DATASET_LOADERS,
    )
    all_ok &= check("EXATHLON_APP_IDS == [1,2,4,5,6,9]",
                    EXATHLON_APP_IDS == [1, 2, 4, 5, 6, 9])
    for app in EXATHLON_APP_IDS:
        all_ok &= check(f"DATASET_LOADERS['exathlon_app{app}'] registered",
                        f'exathlon_app{app}' in DATASET_LOADERS)

    # 2. Data files
    print("\n[2/8] Data files (dataset/Exathlon/)")
    exa_root = PROJECT_ROOT / 'dataset' / 'Exathlon'
    all_ok &= check("ground_truth.csv exists",
                    (exa_root / 'ground_truth.csv').exists())
    for app in EXATHLON_APP_IDS:
        n_csv = len(list((exa_root / f'app{app}').glob('*.csv')))
        all_ok &= check(f"app{app} has CSV files (n={n_csv})", n_csv > 0)

    # 3. load_exathlon smoke test
    print("\n[3/8] load_exathlon smoke test (app1)")
    try:
        sigs, lbl, regs, fn, tr, info = load_exathlon(app=1)
        all_ok &= check(f"shape={sigs.shape}, anomaly_regions={len(regs)}, train_ratio={tr:.3f}",
                        sigs.shape == (90897, 19) and len(regs) >= 1)
    except Exception as e:
        all_ok &= check(f"load_exathlon error: {e}", False)

    # 4. UnifiedLoader (Comparison)
    print("\n[4/8] UnifiedLoader (comparison/data/unified_loader.py)")
    from comparison.data.unified_loader import UnifiedLoader
    try:
        loader = UnifiedLoader(dataset='exathlon', app=1, normalize_mode='minmax').load()
        all_ok &= check(f"UnifiedLoader features shape: {loader.features.shape}",
                        loader.features.shape[1] == 19)
    except Exception as e:
        all_ok &= check(f"UnifiedLoader error: {e}", False)

    # 5. comparison/experiment_configs.py
    print("\n[5/8] comparison/experiment_configs.py")
    from comparison.experiment_configs import EXPERIMENT_CONFIGS
    for app in EXATHLON_APP_IDS:
        all_ok &= check(f"exathlon_app{app} config",
                        f'exathlon_app{app}' in EXPERIMENT_CONFIGS)
        all_ok &= check(f"exathlon_app{app}_normalonly config",
                        f'exathlon_app{app}_normalonly' in EXPERIMENT_CONFIGS)

    # 6. MAE base_experiments registry
    print("\n[6/8] scripts/run_base_experiments.py")
    from scripts.run_base_experiments import (
        DATASETS, SMD_DATASETS, EXATHLON_DATASETS, aggregate_exathlon_results,
    )
    all_ok &= check(f"len(EXATHLON_DATASETS) == 6 (got {len(EXATHLON_DATASETS)})",
                    len(EXATHLON_DATASETS) == 6)
    all_ok &= check(f"Total datasets = 5+28+6=39 (got {len(DATASETS)+len(SMD_DATASETS)+len(EXATHLON_DATASETS)})",
                    len(DATASETS) + len(SMD_DATASETS) + len(EXATHLON_DATASETS) == 39)
    all_ok &= check("aggregate_exathlon_results callable",
                    callable(aggregate_exathlon_results))

    # 7. Queue files
    print("\n[7/8] configs/baseline_exathlon_*.json")
    for q in ['minmax', 'zscore', 'minmax_normalonly', 'zscore_normalonly']:
        p = PROJECT_ROOT / 'configs' / f'baseline_exathlon_{q}.json'
        all_ok &= check(f"baseline_exathlon_{q}.json exists", p.exists())

    # 8. Docs
    print("\n[8/8] Documentation")
    for doc, term in [
        ('CLAUDE.md', '6 Exathlon'),
        ('set_guideline.md', 'Exathlon'),
        ('ablation_guideline.md', 'exathlon_app'),
        ('docs/DATASET.md', 'Exathlon'),
        ('docs/CHANGELOG.md', 'Exathlon dataset MAE-side integration'),
        ('docs/ABLATION_STUDIES.md', 'exathlon_app1'),
    ]:
        p = PROJECT_ROOT / doc
        if p.exists():
            content = p.read_text()
            all_ok &= check(f"{doc} mentions '{term}'", term in content)
        else:
            all_ok &= check(f"{doc} exists", False)

    print()
    print("=" * 70)
    if all_ok:
        print("✅ ALL CHECKS PASSED — Exathlon integration is consistent")
    else:
        print("❌ SOME CHECKS FAILED — see above")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
