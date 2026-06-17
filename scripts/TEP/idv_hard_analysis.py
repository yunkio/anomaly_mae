"""IDV 3/9/15 (excluded-hard) deep-dive — are they REALLY near-indistinguishable?

Design §2.2 claims closed-loop control fully compensates IDV 3/9/15, making them
near-undetectable for ANY method, and excludes them from headline aggregation by
the quantitative rule (post-onset mean max|z| < 2x fault-free variation).
This script stress-tests that claim at three levels:

  L1. Point-wise separability ceiling: per-feature ROC-AUC of post-onset faulty
      points (test runs 441-460) vs FF-test points (runs 461-500). If even the
      BEST single feature is ~0.5, no point-wise detector can work.
  L2. Run-level aggregation ceiling: aggregate each run's post-onset segment
      (mean / std per feature over 800 samples), then run-level AUC
      (20 faulty runs vs 40 FF runs). Tests whether temporal context
      (the regime a window-500 model operates in) reveals the fault even though
      points are indistinguishable.
  L3. Model-score level: per-fault roc/prc of the 5 simple models on 3/9/15
      (from per_fault_metrics.json) vs the random baseline and vs positive rate.

References IDV1 (easy step) and IDV16/19 (subtle-usable) are included as anchors.

Usage:
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/idv_hard_analysis.py \
      --results-dir scripts/TEP/results/12_..._tep_typegen_simple
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tep_common import (
    DATA_DIR, EXCLUDED_HARD, FAMILY, FAULT_ONSET_IDX, RUN_LEN,
)

FAMILY_OF = {f: fam for fam, fs in FAMILY.items() for f in fs}
ANCHORS = [1, 16, 19]            # easy step + two subtle-usable references
TARGETS = EXCLUDED_HARD + ANCHORS


def load_all():
    te = np.load(os.path.join(DATA_DIR, 'test_stream.npz'))
    with open(os.path.join(DATA_DIR, 'test_run_table.json')) as fp:
        rt = json.load(fp)
    with open(os.path.join(DATA_DIR, 'manifest.json')) as fp:
        man = json.load(fp)
    return te['X'], te['y'], rt, man['feature_cols']


def post_onset_segments(X, rt, fault):
    """list of (800, F) post-onset segments for the fault's 20 test runs."""
    return [X[r['start'] + FAULT_ONSET_IDX:r['end']] for r in rt
            if r['fault'] == fault]


def ff_segments(X, rt):
    """FF test runs: use the SAME tail length (800) for symmetry."""
    return [X[r['end'] - (RUN_LEN - FAULT_ONSET_IDX):r['end']] for r in rt
            if r['fault'] == 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--models',
                    default='random,sensor_range,pca_error,l2_norm,nn_distance')
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    models = args.models.split(',')

    X, y, rt, feat_names = load_all()
    ff_segs = ff_segments(X, rt)
    ff_pts = np.concatenate(ff_segs, axis=0)          # (40*800, F)

    lines = ['# IDV 3/9/15 (excluded-hard) 심층 검증\n',
             '질문: "폐루프 제어가 완전 보상해 사실상 구분 불가"라는 설계 §2.2의 주장이 '
             '(L1) point 수준, (L2) run-집계 수준, (L3) 모델 score 수준에서 모두 성립하는가?  \n'
             f'대상: IDV {EXCLUDED_HARD} + 참조 anchor IDV {ANCHORS} '
             '(1=쉬운 step, 16/19=usable 중 가장 subtle).\n']

    # ---------- L1: point-wise per-feature AUC ceiling ----------
    lines.append('## L1. Point 수준 분리 한계 (per-feature ROC-AUC, post-onset faulty pts vs FF pts)\n')
    lines.append('| fault | family | **best-feature AUC** | top-3 features (AUC) | mean-shift 최대 효과크기 d | std-ratio 최대 |')
    lines.append('|---|---|---|---|---|---|')
    ff_mu, ff_sd = ff_pts.mean(axis=0), ff_pts.std(axis=0)
    ff_sd_safe = np.where(ff_sd == 0, 1.0, ff_sd)
    l1_best = {}
    for f in TARGETS:
        segs = post_onset_segments(X, rt, f)
        pts = np.concatenate(segs, axis=0)            # (20*800, F)
        lab = np.concatenate([np.ones(len(pts)), np.zeros(len(ff_pts))])
        data = np.concatenate([pts, ff_pts], axis=0)
        aucs = np.array([
            max(a := roc_auc_score(lab, data[:, j]), 1 - a)   # direction-free
            for j in range(data.shape[1])
        ])
        d_eff = np.abs(pts.mean(axis=0) - ff_mu) / ff_sd_safe
        sratio = np.maximum(pts.std(axis=0) / ff_sd_safe,
                            ff_sd_safe / np.where(pts.std(axis=0) == 0, 1, pts.std(axis=0)))
        top3 = np.argsort(aucs)[::-1][:3]
        l1_best[f] = float(aucs.max())
        lines.append(
            f"| IDV{f} | {FAMILY_OF.get(f, 'EXCL-HARD')} | **{aucs.max():.3f}** | "
            + ', '.join(f'{feat_names[j]}({aucs[j]:.3f})' for j in top3)
            + f" | {d_eff.max():.2f} ({feat_names[int(d_eff.argmax())]}) "
            + f"| {sratio.max():.2f} ({feat_names[int(sratio.argmax())]}) |")
    lines.append('')

    # ---------- L2: run-level aggregation ceiling ----------
    lines.append('## L2. Run-집계 수준 분리 한계 (run당 800-sample 집계 후 20 vs 40 runs AUC)\n')
    lines.append('window 모델(W=500)이 쓸 수 있는 "시간 맥락 집계"의 상한 근사. '
                 'mean-집계와 std-집계 각각의 best-feature run-level AUC.\n')
    lines.append('| fault | best AUC (run-mean) | feature | best AUC (run-std) | feature | 판정 |')
    lines.append('|---|---|---|---|---|---|')
    ff_run_mean = np.stack([s.mean(axis=0) for s in ff_segs])   # (40, F)
    ff_run_std = np.stack([s.std(axis=0) for s in ff_segs])
    l2_best = {}
    for f in TARGETS:
        segs = post_onset_segments(X, rt, f)
        rm = np.stack([s.mean(axis=0) for s in segs])           # (20, F)
        rs = np.stack([s.std(axis=0) for s in segs])
        lab = np.concatenate([np.ones(len(rm)), np.zeros(len(ff_run_mean))])
        auc_m = np.array([max(a := roc_auc_score(
            lab, np.concatenate([rm[:, j], ff_run_mean[:, j]])), 1 - a)
            for j in range(rm.shape[1])])
        auc_s = np.array([max(a := roc_auc_score(
            lab, np.concatenate([rs[:, j], ff_run_std[:, j]])), 1 - a)
            for j in range(rs.shape[1])])
        best = max(auc_m.max(), auc_s.max())
        l2_best[f] = float(best)
        verdict = ('분리 가능' if best >= 0.9 else
                   '부분 분리' if best >= 0.75 else '비식별')
        lines.append(
            f"| IDV{f} | {auc_m.max():.3f} | {feat_names[int(auc_m.argmax())]} "
            f"| {auc_s.max():.3f} | {feat_names[int(auc_s.argmax())]} | {verdict} |")
    lines.append('')

    # ---------- L3: model-score level ----------
    lines.append('## L3. 모델 score 수준 (per-fault, 각 fault 20 runs + FF 40 runs; '
                 'positive rate = 800x20/(800x20+38400) ≈ 29.4% point 기준)\n')
    lines.append('| fault | model | roc_auc | prc_auc | pak_auc_f1 | 해석 |')
    lines.append('|---|---|---|---|---|---|')
    for f in TARGETS:
        for m in models:
            # prefer a contaminated fold where f is NOT in train; ffonly is clean ref
            pf_path = results_dir / 'ffonly' / m / 'per_fault_metrics.json'
            if not pf_path.exists():
                continue
            with open(pf_path) as fp:
                pf = json.load(fp).get(str(f), {})
            roc, prc = pf.get('roc_auc'), pf.get('prc_auc')
            interp = ''
            if roc is not None:
                interp = ('random 동등' if abs(roc - 0.5) < 0.05 else
                          '약한 신호' if roc < 0.75 else '식별')
            lines.append(f"| IDV{f} | {m}@ffonly | "
                         f"{roc:.3f} | {prc:.3f} | {pf.get('pak_auc_f1', 0):.3f} | {interp} |"
                         if roc is not None else
                         f"| IDV{f} | {m}@ffonly | — | — | — | 결과 없음 |")
    lines.append('')

    # ---------- verdict ----------
    lines.append('## 종합 판정\n')
    for f in EXCLUDED_HARD:
        l1, l2 = l1_best.get(f), l2_best.get(f)
        if l1 is not None and l2 is not None:
            if l2 < 0.75:
                v = ('**완전 비식별** — point(best-feature AUC '
                     f'{l1:.3f})과 run-집계({l2:.3f}) 모두 분리 불가. '
                     '"어떤 방법으로도(window 집계 포함) 비식별" 주장 지지.')
            elif l2 >= 0.9 and l1 < 0.7:
                v = (f'**point 비식별 / run-집계 분리 가능** (L1 {l1:.3f} vs L2 {l2:.3f}) — '
                     'point-wise 방법은 못 잡지만 시간-맥락 모델(W=500)은 잡을 수 있는 후보. '
                     '설계의 "어떤 방법으로도 비식별" 문구를 "point-wise 비식별"로 한정 필요.')
            else:
                v = f'**경계 사례** (L1 {l1:.3f}, L2 {l2:.3f}) — appendix에 양 수치 병기 권장.'
            lines.append(f'- IDV{f}: {v}')
    lines.append('')
    lines.append('Anchor 대조: ' + ', '.join(
        f'IDV{f} L1={l1_best.get(f):.3f}/L2={l2_best.get(f):.3f}'
        for f in ANCHORS) + ' (subtle-usable도 run-집계에서는 분리됨이 정상).')

    out = results_dir / 'idv_hard_report.md'
    with open(out, 'w') as fp:
        fp.write('\n'.join(lines))
    print(f'Report written: {out}')


if __name__ == '__main__':
    main()
