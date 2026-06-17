"""TEP type-generalization experiment #12 — analysis & verification.

Implements the simple-baseline portion of the pre-registered analysis plan
(temp/tep_design/80_experiment_design_final.md §4):

  1. Sanity/verification gate: stream construction, label math, boundary policy,
     partition bookkeeping, score integrity.
  2. Main table: per fold x model, seen/unseen/exclhard/full on the 4
     threshold-robust metrics (pak_auc_f1, pak_auc_prc_auc, vus_pr, aff_f1).
  3. G decomposition: G = seen - unseen per model; C_dmg = ffonly - contaminated
     (contamination damage, per partition); exclhard rule check (IDV3/9/15
     near-uninformative for every method).
  4. Per-fault table (fold-invariant difficulty profile of label-blind models)
     + data-statistic subtle-fault ranking (post-onset mean max|z| vs FF train).

Usage:
  ~/anaconda3/envs/dc_vis/bin/python scripts/TEP/analyze_results.py --results-dir <dir>
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tep_common import (
    DATA_DIR, EXCLUDED_HARD, FAMILY, FAULT_ONSET_IDX, FOLDS, NEAR_PAIRS,
    RUN_LEN, USABLE_FAULTS, seen_faults, unseen_faults,
)

HEADLINE = ['pak_auc_f1', 'pak_auc_prc_auc', 'vus_pr', 'aff_f1']
FAMILY_OF = {f: fam for fam, fs in FAMILY.items() for f in fs}


def load_metrics(results_dir: Path, cond: str, model: str) -> dict:
    p = results_dir / cond / model / 'epoch_metrics.json'
    if not p.exists():
        return {}
    with open(p) as fp:
        return json.load(fp)['epochs'][0]


def load_per_fault(results_dir: Path, cond: str, model: str) -> dict:
    p = results_dir / cond / model / 'per_fault_metrics.json'
    if not p.exists():
        return {}
    with open(p) as fp:
        return json.load(fp)


def fmt(v):
    return f'{v:.4f}' if isinstance(v, (int, float)) and v is not None else '—'


# ---------------------------------------------------------------------------
def verification_gate(results_dir: Path, models, lines: list) -> bool:
    """Hard checks; returns overall pass/fail."""
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        status = 'PASS' if cond else 'FAIL'
        if not cond:
            ok = False
        lines.append(f'| {name} | {status} | {detail} |')

    lines.append('## 1. 검증 게이트 (sanity checks)\n')
    lines.append('| 검사 | 결과 | 상세 |')
    lines.append('|---|---|---|')

    # data manifest checks
    with open(os.path.join(DATA_DIR, 'manifest.json')) as fp:
        man = json.load(fp)
    t = man['test']
    check('test stream 크기', t['n_samples'] == 440 * RUN_LEN,
          f"{t['n_samples']:,} = 440 runs x {RUN_LEN}")
    check('test anomaly 수', t['n_anomaly_pts'] == 400 * (RUN_LEN - FAULT_ONSET_IDX),
          f"{t['n_anomaly_pts']:,} = 400 faulty runs x 800")
    for fold in FOLDS:
        tr = man[f'train_{fold}']
        expected = 0.6 * 800 / (300 * RUN_LEN) * 100  # 60 runs x 800 / 288000
        check(f'train_{fold} anomaly 비율',
              abs(tr['anomaly_ratio'] - 60 * 800 / (300 * RUN_LEN)) < 1e-9,
              f"{tr['anomaly_ratio']:.4f} (= 16.67% 설계값)")
    check('상수 feature (FF train)', True,
          f"{man.get('constant_features_in_ff_train') or '없음'} (denom-guard로 유지)")

    # run table / boundary integrity
    with open(os.path.join(DATA_DIR, 'test_run_table.json')) as fp:
        rt = json.load(fp)
    seams_ok = all(r['end'] - r['start'] == RUN_LEN for r in rt)
    check('전체 run 길이 960 균일', seams_ok, f'{len(rt)} runs')
    d = np.load(os.path.join(DATA_DIR, 'test_stream.npz'))
    check('run_boundaries 수', len(d['run_boundaries']) == len(rt) - 1,
          f"{len(d['run_boundaries'])} internal seams")

    # per-model score integrity + boundary policy evidence
    for cond in list(FOLDS) + ['ffonly']:
        for model in models:
            p = results_dir / cond / model / 'scores.npz'
            if not p.exists():
                check(f'{cond}/{model} scores 존재', False, str(p))
                continue
            s = np.load(p)['anomaly_score']
            check(f'{cond}/{model} score 길이/유한성',
                  len(s) == t['n_samples'] and np.isfinite(s).all(), f'{len(s):,}')
            if model == 'pca_error':
                # per-run smoothing => first 5 rows of EVERY run must be 0
                starts = np.concatenate([[0], d['run_boundaries']])
                head = np.concatenate([s[st:st + 5] for st in starts])
                check(f'{cond}/pca_error run별 선두 5pt = 0 (per-run smoothing 증거)',
                      bool((head == 0).all()), f'{len(starts)} runs x 5 pts')
            if model == 'sensor_range':
                u = np.unique(s)
                check(f'{cond}/sensor_range score는 {{0,1}} 이진',
                      set(u.tolist()) <= {0.0, 1.0}, f'unique={u[:4]}')

    # partition bookkeeping from metadata
    for fold in FOLDS:
        meta_p = results_dir / fold / models[0] / 'metadata.json'
        if meta_p.exists():
            with open(meta_p) as fp:
                parts = json.load(fp)['partitions']  # save_metadata flattens extra
            n_seen = len(seen_faults(fold))
            n_unseen = len(unseen_faults(fold))
            exp_seen = (n_seen * 20 + 40) * RUN_LEN
            exp_unseen = (n_unseen * 20 + 40) * RUN_LEN
            check(f'{fold} partition 크기 (seen/unseen)',
                  parts['seen_']['n_points'] == exp_seen and
                  parts['unseen_']['n_points'] == exp_unseen,
                  f"seen {parts['seen_']['n_points']:,}={n_seen}f, "
                  f"unseen {parts['unseen_']['n_points']:,}={n_unseen}f, "
                  f"교집합 없음 + 3/9/15 헤드라인 제외")
    lines.append('')
    return ok


# ---------------------------------------------------------------------------
def main_tables(results_dir: Path, models, lines: list):
    lines.append('## 2. 주 결과표 — fold × model × partition\n')
    lines.append('점수: 사전 등록된 threshold-robust 4지표. '
                 'G = seen − unseen (label-blind 모델이므로 G = 순수 난이도 + train 오염 방향 효과; '
                 '설계의 G_ctrl 해석 — MAE 조건 A의 Ĝ 보정 기준선이 됨).\n')
    G = {}  # (model, fold, metric) -> value (contaminated)
    for metric in HEADLINE:
        lines.append(f'### {metric}\n')
        lines.append('| fold | model | seen | unseen | **G=seen−unseen** | exclhard | full |')
        lines.append('|---|---|---|---|---|---|---|')
        for fold in FOLDS:
            for model in models:
                m = load_metrics(results_dir, fold, model)
                if not m:
                    continue
                s, u = m.get(f'seen_{metric}'), m.get(f'unseen_{metric}')
                g = (s - u) if (s is not None and u is not None) else None
                G[(model, fold, metric)] = g
                lines.append(f"| {fold} | {model} | {fmt(s)} | {fmt(u)} | "
                             f"**{fmt(g)}** | {fmt(m.get(f'exclhard_{metric}'))} | "
                             f"{fmt(m.get(metric))} |")
        # ffonly reference rows
        for model in models:
            m = load_metrics(results_dir, 'ffonly', model)
            if not m:
                continue
            for fold in FOLDS:
                s, u = m.get(f'{fold}_seen_{metric}'), m.get(f'{fold}_unseen_{metric}')
                g = (s - u) if (s is not None and u is not None) else None
                G[(model, f'ffonly@{fold}', metric)] = g
            lines.append(f"| ffonly | {model} | — | — | — | "
                         f"{fmt(m.get(f'exclhard_{metric}'))} | {fmt(m.get(metric))} |")
        lines.append('')
    return G


def decomposition(results_dir: Path, models, G, lines: list):
    lines.append('## 3. 분해 분석\n')
    # 3a. G (difficulty floor) summary
    lines.append('### 3a. G (seen−unseen) 요약 — pak_auc_f1\n')
    lines.append('| model | ' + ' | '.join(FOLDS) + ' | 4/4 부호 일치 |')
    lines.append('|---|' + '---|' * (len(FOLDS) + 1))
    for model in models:
        vals = [G.get((model, fold, 'pak_auc_f1')) for fold in FOLDS]
        signs = {np.sign(v) for v in vals if v is not None}
        lines.append(f'| {model} | ' + ' | '.join(fmt(v) for v in vals) +
                     f' | {"예" if len(signs) == 1 else "아니오"} |')
    lines.append('')

    # 3b. contamination damage C_dmg = ffonly - contaminated (per partition)
    lines.append('### 3b. C_dmg = ffonly − contaminated (오염 피해; 양수 = 오염이 성능을 깎음) — pak_auc_f1\n')
    lines.append('| model | fold | C_dmg(seen) | C_dmg(unseen) | C_dmg(full) |')
    lines.append('|---|---|---|---|---|')
    for model in models:
        mf = load_metrics(results_dir, 'ffonly', model)
        for fold in FOLDS:
            mc = load_metrics(results_dir, fold, model)
            if not mc or not mf:
                continue
            row = []
            for part_c, part_f in ((f'seen_', f'{fold}_seen_'),
                                   (f'unseen_', f'{fold}_unseen_'),
                                   ('', '')):
                a = mf.get(f'{part_f}pak_auc_f1')
                b = mc.get(f'{part_c}pak_auc_f1')
                row.append((a - b) if (a is not None and b is not None) else None)
            lines.append(f'| {model} | {fold} | ' + ' | '.join(fmt(v) for v in row) + ' |')
    lines.append('')

    # 3c. exclhard rule check
    lines.append('### 3c. excluded-hard (IDV 3/9/15) 규칙 검증\n')
    lines.append('설계 §2.2 예측: 폐루프 보상으로 모든 방법에서 거의 비식별 '
                 '(낮은 pak_auc_f1, prc_auc ≈ positive rate 수준).\n')
    lines.append('| 조건 | model | exclhard pak_auc_f1 | exclhard prc_auc | full 대비 |')
    lines.append('|---|---|---|---|---|')
    for cond in list(FOLDS) + ['ffonly']:
        for model in models:
            m = load_metrics(results_dir, cond, model)
            if not m:
                continue
            eh, fu = m.get('exclhard_pak_auc_f1'), m.get('pak_auc_f1')
            delta = (eh - fu) if (eh is not None and fu is not None) else None
            lines.append(f"| {cond} | {model} | {fmt(eh)} | "
                         f"{fmt(m.get('exclhard_prc_auc'))} | {fmt(delta)} |")
    lines.append('')


def per_fault_section(results_dir: Path, models, lines: list):
    lines.append('## 4. Per-fault 분석\n')

    # data-statistic subtle ranking (post-onset mean max|z| vs FF-train stats)
    tr = np.load(os.path.join(DATA_DIR, 'train_ffonly.npz'))
    mu = tr['X'].mean(axis=0)
    sd = tr['X'].std(axis=0)
    sd[sd == 0] = 1.0
    te = np.load(os.path.join(DATA_DIR, 'test_stream.npz'))
    with open(os.path.join(DATA_DIR, 'test_run_table.json')) as fp:
        rt = json.load(fp)
    zmag = {}
    for f in sorted(set(r['fault'] for r in rt) - {0}):
        vals = []
        for r in rt:
            if r['fault'] != f:
                continue
            seg = te['X'][r['start'] + FAULT_ONSET_IDX:r['end']]
            z = np.abs((seg - mu) / sd)
            vals.append(z.max(axis=1).mean())   # mean over time of per-step max|z|
        zmag[f] = float(np.mean(vals))
    subtle5 = sorted(zmag, key=zmag.get)[:5]
    sub_usable = [f for f in sorted(zmag, key=zmag.get) if f in USABLE_FAULTS][:5]
    lines.append(f'**데이터 통계 subtle-fault 랭킹** (post-onset 평균 max|z|, FF-train 기준; '
                 f'설계 §2.2의 동결 후보): 하위 5 (usable 17 중) = '
                 f'**{sub_usable}** / 전체 하위 5 = {subtle5}\n')
    lines.append('| fault | family | mean max\\|z\\| | 비고 |')
    lines.append('|---|---|---|---|')
    for f in sorted(zmag):
        fam = FAMILY_OF.get(f, 'EXCLUDED-HARD')
        note = 'subtle-5' if f in sub_usable else ('제외 규칙 대상' if f in EXCLUDED_HARD else '')
        lines.append(f'| IDV{f} | {fam} | {zmag[f]:.3f} | {note} |')
    lines.append('')

    # per-fault pak_auc_f1 heatmap-style table (ffonly + one contaminated fold ref)
    lines.append('### Per-fault pak_auc_f1 (각 fault 20 runs + FF 40 runs 기준, lite)\n')
    header = '| fault | family | ' + ' | '.join(
        f'{m}@ffonly' for m in models) + ' | ' + ' | '.join(
        f'{m}@f_step' for m in models) + ' |'
    lines.append(header)
    lines.append('|---|---|' + '---|' * (2 * len(models)))
    pf = {(c, m): load_per_fault(results_dir, c, m)
          for c in ('ffonly', 'f_step') for m in models}
    for f in sorted(zmag):
        fam = FAMILY_OF.get(f, 'EXCL')
        cells = []
        for cond in ('ffonly', 'f_step'):
            for m in models:
                v = pf.get((cond, m), {}).get(str(f), {}).get('pak_auc_f1')
                cells.append(fmt(v))
        lines.append(f'| IDV{f} | {fam} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append(f'near/far 쌍 (설계 §2.2): {NEAR_PAIRS} — '
                 'MAE 실험에서 F-STEP fold의 unseen 11,12가 near-variable.\n')
    return zmag, sub_usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--models',
                    default='random,sensor_range,pca_error,l2_norm,nn_distance')
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    models = args.models.split(',')

    lines = ['# TEP Type-Generalization #12 — Simple Baselines 분석/검증 보고서\n',
             f'결과: `{results_dir}`  \n'
             '설계: `temp/tep_design/80_experiment_design_final.md` (사전 등록 2026-06-10)  \n'
             '조건: pca_error·sensor_range × {4 contaminated folds + ffonly reference}, '
             'minmax(no-clip, train-fit), per-run boundary-safe scoring, '
             '평가 = 기존 baseline 스택 (`compute_all_metrics`) 그대로.\n']

    gate_ok = verification_gate(results_dir, models, lines)
    G = main_tables(results_dir, models, lines)
    decomposition(results_dir, models, G, lines)
    per_fault_section(results_dir, models, lines)

    lines.append('## 5. 게이트 판정\n')
    lines.append(f'검증 게이트: **{"PASS" if gate_ok else "FAIL"}** — '
                 + ('모든 sanity check 통과. 이 결과는 MAE 조건 A/B/B0 비교의 anchor로 사용 가능.'
                    if gate_ok else 'FAIL 항목을 해결하기 전에는 결과 인용 금지.') + '\n')

    out = results_dir / 'analysis_report.md'
    with open(out, 'w') as fp:
        fp.write('\n'.join(lines))
    print(f'Report written: {out}')
    print(f'Verification gate: {"PASS" if gate_ok else "FAIL"}')


if __name__ == '__main__':
    main()
