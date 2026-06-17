"""TEP #12 — noisy-label(부분 라벨) sweep, simple-baseline floor curve.

질문: train 오염(seen-family faulty 60 runs) 중 n%만 라벨이 알려져 있고
(100-n)%는 unlabeled로 남는 환경에서, 검출기가 잔류 오염에 얼마나 버티는가?

Label-blind simple 모델의 등가 실험 (oracle-cleaning equivalence):
  라벨이 달린 run에 대해 label-consuming 방법이 할 수 있는 이상적 행동 = 학습에서
  제거(purge). 따라서 simple 모델의 "n% labeled" 조건 = train에서 labeled run을
  제거하고 unlabeled (100-n)% run만 오염으로 남긴 것.
    n=0%   → 60 runs 전부 잔류 (= 본 실험 contaminated 조건)
    n=100% → 오염 0 (= ffonly clean 조건)
  이 곡선이 MAE 비교의 floor: MAE-A(GRL purging)가 n% 라벨로 이 oracle-cleaning
  곡선에 근접하면 "라벨 소비 = 오염 정화" 해석(H1)이 지지되고, 못 미치는 폭이
  unlabeled 오염에 대한 취약성, 넘어서는 폭이 cleaning 이상의 가치다.

사전 등록 규칙:
  - labeled run 선택: 각 seen fault의 run 1..N 중 앞쪽 k = (n/100)*N개
    (deterministic; n ∈ {0,20,50,80,100} → f_step N=10: k=0,2,5,8,10 /
     f_ds N=30: k=0,6,15,24,30 — 모두 정수, 반올림 없음)
  - folds: f_step·f_ds (설계 §6.2의 중립 구조 규칙 쌍 — seen 다양성 최대/최소)
  - models: pca_error, sensor_range / 평가: 기존 스택 그대로 (full+partition)

Usage:
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/run_label_sweep.py \
      --results-dir scripts/TEP/results/12_..._tep_typegen_simple
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tep_common import DATA_DIR, FF_TRAIN_RUNS, FOLDS, RUN_LEN, seen_faults
from run_tep_simple import load_test, run_condition  # noqa: E402  (reuses eval stack)

SWEEP_PCTS = [0, 20, 50, 80, 100]
SWEEP_FOLDS = ['f_step', 'f_ds']
SWEEP_MODELS = ['pca_error', 'sensor_range']


def build_sweep_train(fold: str, labeled_pct: int) -> np.ndarray:
    """train = FF 240 runs + 각 seen fault의 run (k+1..N)  (앞쪽 k개 = labeled → 제거)."""
    d = np.load(os.path.join(DATA_DIR, f'train_{fold}.npz'))
    X, fid = d['X'], d['fault_id']
    n_ff = len(FF_TRAIN_RUNS)
    n_per_fault = FOLDS[fold]['train_runs_per_fault']
    k = labeled_pct * n_per_fault // 100
    assert labeled_pct * n_per_fault % 100 == 0, "non-integer k — pct/N mismatch"

    blocks = [X[:n_ff * RUN_LEN]]                     # FF runs (always kept)
    n_blocks = len(X) // RUN_LEN
    within = {}
    kept_faulty = 0
    for b in range(n_ff, n_blocks):
        s = b * RUN_LEN
        f = int(fid[s])
        within[f] = within.get(f, 0) + 1              # 1-based within-fault index
        if within[f] > k:                             # first k runs = labeled → removed
            blocks.append(X[s:s + RUN_LEN])
            kept_faulty += 1
    expected = len(seen_faults(fold)) * (n_per_fault - k)
    assert kept_faulty == expected, (kept_faulty, expected)
    return np.concatenate(blocks, axis=0), kept_faulty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--folds', default=','.join(SWEEP_FOLDS),
                    help='comma-separated fold subset (parallel execution)')
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    folds = args.folds.split(',')

    test_X, test_y, fault_id, run_bounds, run_table = load_test()
    summary = []
    for fold in folds:
        from tep_common import unseen_faults
        parts = {'seen_': set(seen_faults(fold)),
                 'unseen_': set(unseen_faults(fold))}
        for pct in SWEEP_PCTS:
            train_X, kept = build_sweep_train(fold, pct)
            for model in SWEEP_MODELS:
                out = results_dir / 'sweep' / fold / model / f'labeled_{pct:03d}'
                if (out / 'epoch_metrics.json').exists():
                    print(f"=== sweep {fold} / {model} / labeled {pct}% — SKIP (완료) ===")
                    continue
                print(f"\n=== sweep {fold} / {model} / labeled {pct}% "
                      f"(잔류 오염 {kept} runs, train {len(train_X):,}) ===")
                m = run_condition(train_X, model, out,
                                  test_X, test_y, fault_id, run_bounds, run_table,
                                  parts, f'tep_sweep_{fold}_l{pct}',
                                  per_fault=False, smoke=False)
                summary.append({'fold': fold, 'model': model, 'labeled_pct': pct,
                                'kept_faulty_runs': kept,
                                'seen_pak_auc_f1': m.get('seen_pak_auc_f1'),
                                'unseen_pak_auc_f1': m.get('unseen_pak_auc_f1'),
                                'full_pak_auc_f1': m.get('pak_auc_f1')})
    with open(results_dir / 'sweep' / f"sweep_summary_{'_'.join(folds)}.json", 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSweep done: {results_dir / 'sweep'}")


if __name__ == '__main__':
    main()
