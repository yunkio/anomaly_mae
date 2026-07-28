#!/usr/bin/env python
"""LASAD paper — baseline results aggregator.

Generates comparison/results/experiments/results_baseline.md from the reseed 5-seed
experiments (8-1..8-5 unsup / 9-1..9-5 weak), following the paper protocol
(LASAD.pdf A.1.1) and the requirements spec extracted 2026-07-19:

  * Seeds {42,43,40,41,44}  (8-k/9-k share SEEDS[k], k=1..5)
  * Epoch selection: trained models -> ES-selected epoch = argmin(train_loss)
    (paper: native train-loss early stopping, restore-best = min train loss).
    NO_ES models (no usable train_loss) -> FINAL epoch.  Stateless -> single eval.
  * Diagnostic alternative: per entity, choose the epoch with maximum test-side
    PAK (`pak_auc_f1`; `excl22_pak_auc_f1` for SWaT excl22), then report every
    metric from that one epoch. This is kept strictly separate from paper ES.
  * Metrics: Table 2 = pak_auc_f1 / vus_pr / affiliation_f1_ar
             Table A.6 = prc_auc / vus_roc / pa_0_f1
  * dcdetector: 'neg' convention (score = -canonical) — read from
    _dcdetector_neg_cache.json (built by the sign-inversion pipeline; mtime-guarded,
    incremental for future 8-4/8-5 cells).
  * Every statistical table reports mean, sample std (ddof=1), min, and max.
    Missing cells stay blank; n is annotated when n<5.

Re-runnable: as more seeds finish, re-run to fill blanks.
Aux inputs (optional, from the requirements workflow) — permanent repo dir
comparison/results/experiments/_aux/ (moved from /tmp scratchpad 2026-07-22;
scratchpad was wiped on reboot):
  _aux/printed_values.json     (paper printed values, LASAD.pdf Table 2 p.7,
                                2-pass verified transcription 2026-07-22)
  _aux/param_verification.json (code-vs-paper params; not yet regenerated)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # repo root
EXP = ROOT / 'comparison' / 'results' / 'experiments'
OUT_MD = EXP / 'results_baseline.md'
OUT_JSON = EXP / 'results_data.json'
OUT_BEST_JSON = EXP / 'results_best_epoch_data.json'
NEG_CACHE = EXP / '_dcdetector_neg_cache.json'
AUX_DIR = EXP / '_aux'                                 # permanent (repo) — was /tmp scratchpad, wiped on reboot

SUFFIX = '20260606_175756'
SEEDS = [42, 43, 40, 41, 44]                            # k=1..5
UNSUP_DIR = {s: EXP / f'8-{k}_{SUFFIX}_baseline' for k, s in enumerate(SEEDS, 1)}
WEAK_DIR = {s: EXP / f'9-{k}_{SUFFIX}_weak_ssl' for k, s in enumerate(SEEDS, 1)}

STATELESS = {'random', 'sensor_range', 'pca_error', 'l2_norm', 'nn_distance'}
NO_ES = {'dcdetector', 'tfmae', 'nrdetector', 'nrdetector_full', 'treemil'}
WEAK = {'deepmil', 'wetas', 'treemil', 'nrdetector', 'nrdetector_full'}

# (paper name, code, group, venue, ranked)
MODELS = [
    ('Random score', 'random', 'Simple', "ICML'24", True),
    ('Sensor range', 'sensor_range', 'Simple', "ICML'24", True),
    ('PCA recon.', 'pca_error', 'Simple', "ICML'24", True),
    ('L2-norm', 'l2_norm', 'Simple', "ICML'24", True),
    ('NN-distance', 'nn_distance', 'Simple', "ICML'24", True),
    ('MLP', 'mlp', 'Lightweight', "ICML'24", True),
    ('MLPMixer', 'mlpmixer', 'Lightweight', "ICML'24", True),
    ('Transformer', 'transformer', 'Lightweight', "ICML'24", True),
    ('GCN-LSTM', 'gcn_lstm', 'Lightweight', "ICML'24", True),
    ('Anomaly Trans.', 'anomaly_transformer', 'Deep/SOTA', "ICLR'22", True),
    ('TranAD', 'tranad', 'Deep/SOTA', "PVLDB'22", True),
    ('USAD', 'usad', 'Deep/SOTA', "KDD'20", True),
    ('DAGMM (simpl.)', 'dagmm', 'Deep/SOTA', "ICLR'18", True),
    ('GDN', 'gdn', 'Deep/SOTA', "AAAI'21", True),
    ('OmniAnomaly', 'omnianomaly', 'Deep/SOTA', "KDD'19", True),
    ('TFMAE', 'tfmae', 'Deep/SOTA', "ICDE'24", True),
    ('NPSR', 'npsr', 'Deep/SOTA', "NeurIPS'23", True),
    ('TimesNet', 'timesnet', 'Deep/SOTA', "ICLR'23", True),
    ('DCdetector', 'dcdetector', 'Deep/SOTA', "KDD'23", True),
    ('MEMTO', 'memto', 'Deep/SOTA', "NeurIPS'23", True),
    ('ModernTCN', 'moderntcn', 'Deep/SOTA', "ICLR'24", True),
    ('CATCH', 'catch', 'Deep/SOTA', "ICLR'25", True),
    ('DeepMIL', 'deepmil', 'Weak', "CVPR'18", True),
    ('WETAS', 'wetas', 'Weak', "ICCV'21", True),
    ('TreeMIL', 'treemil', 'Weak', "ICASSP'24", True),
    ('NRdetector', 'nrdetector', 'Weak', "KDD'25", True),
    ('NRdetector (full)', 'nrdetector_full', 'Weak', "KDD'25", False),  # non-ranked extra
]

# Table 2 ranked entities (paper column order) / A.5 adds SWaT full
RANKED_ENTITIES = [('SWaT (excl22)', 'SWaT/A1A2_excl22'),
                   ('WaDi A1', 'WaDi/A1'),
                   ('WaDi A2', 'WaDi/A2'),
                   ('PSM', 'PSM')]
A5_ENTITIES = [('SWaT (excl22)', 'SWaT/A1A2_excl22'),
               ('SWaT (full)', 'SWaT/A1A2_full'),
               ('WaDi A1', 'WaDi/A1'),
               ('WaDi A2', 'WaDi/A2'),
               ('PSM', 'PSM')]
RANKED_METRICS = [('PAK', 'pak_auc_f1'), ('VUS', 'vus_pr'), ('Aff', 'affiliation_f1_ar')]
SUPP_METRICS = [('PRC', 'prc_auc'), ('V-R', 'vus_roc'), ('PA', 'pa_0_f1')]
ALL_KEYS = [k for _, k in RANKED_METRICS + SUPP_METRICS]

warnings = []


def _load_epochs(p: Path):
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
    except Exception as e:
        warnings.append(f'parse fail {p}: {e}')
        return None
    eps = d.get('epochs', d) if isinstance(d, dict) else d
    return eps if isinstance(eps, list) and eps else None


def select_epoch_idx(eps, code):
    """Paper protocol: ES(=argmin train_loss) / NO_ES final / stateless single."""
    if code in STATELESS or len(eps) == 1:
        return 0
    if code in NO_ES:
        return len(eps) - 1
    tl = [(i, e.get('train_loss')) for i, e in enumerate(eps)]
    valid = [(i, t) for i, t in tl if isinstance(t, (int, float))]
    if not valid:
        warnings.append(f'no train_loss ({code}); fallback=final epoch')
        return len(eps) - 1
    return min(valid, key=lambda x: x[1])[0]


def select_best_epoch_idx(eps, sub):
    """Diagnostic test-side oracle: max entity-specific PAK, earliest on ties."""
    key = 'excl22_pak_auc_f1' if sub == 'SWaT/A1A2_excl22' else 'pak_auc_f1'
    valid = [(i, e.get(key)) for i, e in enumerate(eps)
             if isinstance(e.get(key), (int, float)) and np.isfinite(e.get(key))]
    if not valid:
        warnings.append(f'no {key}; best-epoch fallback=final epoch')
        return len(eps) - 1
    return max(valid, key=lambda x: x[1])[0]


_neg_cache = json.load(open(NEG_CACHE)) if NEG_CACHE.exists() else {}


def _dc_neg(seed, sub):
    """dcdetector neg-convention metrics at FINAL cached epoch (NO_ES)."""
    pref = f'{seed}|{sub}|ep'
    eps = sorted((k for k in _neg_cache if k.startswith(pref)),
                 key=lambda k: int(k.rsplit('ep', 1)[1]))
    if not eps:
        return None
    m = _neg_cache[eps[-1]]
    return {'vals': {k: m.get(k) for k in ALL_KEYS},
            'epoch': int(eps[-1].rsplit('ep', 1)[1]), 'n_ep': len(eps)}


def cell(code, seed, sub, selection='es'):
    """One (model, seed, entity) cell -> {vals:{key:val}, epoch, n_ep} or None."""
    if code == 'dcdetector':
        return _dc_neg(seed, sub)
    base = WEAK_DIR[seed] if code in WEAK else UNSUP_DIR[seed]
    if sub == 'SWaT/A1A2_excl22':
        f_eps = _load_epochs(base / 'SWaT/A1A2_full' / code / 'epoch_metrics.json')
        if f_eps is None:
            return None
        idx = (select_epoch_idx(f_eps, code) if selection == 'es'
               else select_best_epoch_idx(f_eps, sub))
        e = f_eps[idx]
        vals = {k: e.get('excl22_' + k) for k in ALL_KEYS}
        if any(v is None for v in vals.values()):        # fallback: derived dir
            x_eps = _load_epochs(base / sub / code / 'epoch_metrics.json')
            if x_eps and idx < len(x_eps):
                for k in ALL_KEYS:
                    if vals[k] is None:
                        vals[k] = x_eps[idx].get(k)
        return {'vals': vals, 'epoch': idx + 1, 'n_ep': len(f_eps)}
    eps = _load_epochs(base / sub / code / 'epoch_metrics.json')
    if eps is None:
        return None
    idx = (select_epoch_idx(eps, code) if selection == 'es'
           else select_best_epoch_idx(eps, sub))
    return {'vals': {k: eps[idx].get(k) for k in ALL_KEYS},
            'epoch': idx + 1, 'n_ep': len(eps)}


# ---------------- collect ----------------
DATA_ES = {}       # paper protocol: train-loss ES / NO_ES final / stateless single
DATA_BEST = {}     # diagnostic: max entity-specific test PAK
for _, code, *_ in MODELS:
    for _, sub in A5_ENTITIES:
        DATA_ES[(code, sub)] = {s: cell(code, s, sub, 'es') for s in SEEDS}
        DATA_BEST[(code, sub)] = {s: cell(code, s, sub, 'best') for s in SEEDS}
DATA = DATA_ES     # backward-compatible alias for the paper-protocol renderer


def agg(code, sub, key, data=None):
    """-> (mean, sample std, n, per_seed{seed: value})."""
    data = DATA_ES if data is None else data
    per = {}
    for s in SEEDS:
        c = data[(code, sub)][s]
        v = c['vals'].get(key) if c else None
        if isinstance(v, (int, float)):
            per[s] = float(v)
    if not per:
        return None, None, 0, per
    xs = list(per.values())
    mean = float(np.mean(xs))
    std = float(np.std(xs, ddof=1)) if len(xs) > 1 else None
    return mean, std, len(xs), per


def agg_stats(code, sub, key, data=None):
    """Complete descriptive statistics for one method/entity/metric."""
    mean, std, n, per = agg(code, sub, key, data)
    xs = list(per.values())
    return {
        'mean': mean,
        'std': std,
        'min': float(np.min(xs)) if xs else None,
        'max': float(np.max(xs)) if xs else None,
        'n': n,
        'per_seed': per,
    }


def f4(v):
    return f'{v:.4f}' if isinstance(v, (int, float)) else ''


def ms(mean, std, n):
    if mean is None:
        return ''
    s = f'{mean:.4f}'
    s += f' ± {std:.4f}' if std is not None else ' ± —'
    if n < len(SEEDS):
        s += f' (n={n})'
    return s


def stat_cell(stats):
    """Compact cell order is fixed and stated in every statistics table."""
    if stats['mean'] is None:
        return ''
    std = f4(stats['std']) if stats['std'] is not None else '—'
    suffix = f" (n={stats['n']})" if stats['n'] < len(SEEDS) else ''
    return (f"{f4(stats['mean'])} / {std} / {f4(stats['min'])} / "
            f"{f4(stats['max'])}{suffix}")


def epoch_stats(code, sub, data):
    vals = [data[(code, sub)][s]['epoch'] for s in SEEDS if data[(code, sub)][s]]
    if not vals:
        return None
    return {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        'min': int(min(vals)), 'max': int(max(vals)), 'n': len(vals),
    }


def average_desc_ranks(values):
    """Average ranks for descending scores; exact ties share their rank."""
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    out = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for code, _ in ordered[i:j]:
            out[code] = rank
        i = j
    return out


def rank_distribution(data):
    """Baseline-only ranks over the 12 Table-2 entity/metric cells."""
    ranked_codes = [code for _, code, _, _, ranked in MODELS if ranked]
    per_code = {code: [] for code in ranked_codes}
    for _, sub in RANKED_ENTITIES:
        for _, key in RANKED_METRICS:
            means = {}
            for code in ranked_codes:
                mean, _, n, _ = agg(code, sub, key, data)
                if mean is not None and n:
                    means[code] = mean
            for code, rank in average_desc_ranks(means).items():
                per_code[code].append(rank)
    out = {}
    for code, ranks in per_code.items():
        out[code] = {
            'mean': float(np.mean(ranks)) if ranks else None,
            'std': float(np.std(ranks, ddof=1)) if len(ranks) > 1 else None,
            'min': float(min(ranks)) if ranks else None,
            'max': float(max(ranks)) if ranks else None,
            'n': len(ranks),
            'per_cell': ranks,
        }
    return out


RANKS_ES = rank_distribution(DATA_ES)
RANKS_BEST = rank_distribution(DATA_BEST)


# ---------------- render ----------------
L = []
A = L.append
A('# LASAD — Baseline Experiment Results (5-seed)')
A('')
A('> **파일 역할**: 논문 baseline/TEP 표의 재현값, seed 통계, 선택 epoch, 원자료 경로를 한 문서에서 검증하는 정본 인덱스. ')
A('> **생성**: `comparison/build_results_md.py` → `comparison/results/experiments/results_baseline.md`. ')
A('> **기계가독 산출물**: 논문 ES=`results_data.json`; 진단 best epoch=`results_best_epoch_data.json` (각 675 cells). ')
A('> **정본 데이터**: reseed 체인 `8-1..8-5` (unsup, anomaly-excised) / `9-1..9-5` (weak, contaminated), '
  'seeds **{42, 43, 40, 41, 44}** (8-k·9-k 공유; 논문 Table A.1의 `[n]`→**5**). ')
A('> **ES 기준(논문 기본)**: trained → train-loss restore-best(=argmin train_loss), '
  f'NO_ES {sorted(NO_ES)} → final epoch, stateless → 단일 평가. ')
A('> **Best-epoch 기준(진단용)**: 각 entity의 test PAK가 최대인 단일 epoch를 고른 뒤, '
  '그 동일 epoch의 PAK/VUS/Aff/PRC/V-R/PA를 함께 보고한다. SWaT excl22는 `excl22_pak_auc_f1`로 선택한다. '
  '이는 test-side oracle이므로 논문 기본값과 혼합하지 않는다. ')
A('> **DCdetector = `neg` convention**: score = −canonical (부호 반전 검증 2026-07-19에 따른 사용자 결정). '
  '`_dcdetector_neg_cache.json`에서 로드 (mtime-guard 증분). ')
A('> 빈칸 = 해당 결과 미산출. `(n=k)` = 가용 seed 수 < 5. std = 표본표준편차(ddof=1).')
A('')


def table_note(table_id, paper_table, purpose, *, selection=None, paths=None, reference=None,
               reference_detail=None, paper_ready=False):
    """Emit a stable table index and provenance block immediately before a table."""
    role = '**논문 기입 대상**' if paper_ready else '검증/진단용'
    bits = [f'**표 ID `{table_id}`**', f'논문 `{paper_table}`', role, purpose]
    if selection:
        bits.append(f'선택 기준: {selection}')
    A('> ' + ' | '.join(bits))
    if reference:
        detail = (reference_detail or
                  f'`{reference}`의 seed별 셀을 같은 method/entity/metric 인덱스로 집계.')
        A(f'> 통계 참조: {detail}')
    if paths:
        A('> 결과 디렉터리: ' + ' ; '.join(f'`{p}`' for p in paths))
    A('')


BASELINE_RESULT_PATHS = [
    'comparison/results/experiments/8-{1..5}_20260606_175756_baseline',
    'comparison/results/experiments/9-{1..5}_20260606_175756_weak_ssl',
]
DATASET_PATHS = [
    'dataset/SWaT/SWaT.A1 & A2_Dec 2015',
    'dataset/WaDi/WADI.A1_9 Oct 2017',
    'dataset/WaDi/WADI.A2_19 Nov 2019',
    'dataset/PSM',
]


def render_mean_table(table_id, paper_table, purpose, entities, metrics, data, selection,
                      *, paper_ready):
    table_note(table_id, paper_table, purpose, selection=selection,
               paths=BASELINE_RESULT_PATHS, paper_ready=paper_ready)
    hdr = '| Method |'
    sep = '|---|'
    for entity_name, _ in entities:
        for metric_name, _ in metrics:
            hdr += f' {entity_name}<br>{metric_name} |'
            sep += '---|'
    if paper_table == 'Table 2':
        hdr += ' Baseline-only<br>Avg Rank |'
        sep += '---|'
        ranks = RANKS_ES if data is DATA_ES else RANKS_BEST
    else:
        ranks = None
    A(hdr)
    A(sep)
    for pname, code, _grp, _venue, ranked in MODELS:
        row = f'| {pname}{"†" if not ranked else ""} |'
        for _, sub in entities:
            for _, key in metrics:
                mean, _std, _n, _per = agg(code, sub, key, data)
                row += f' {f4(mean)} |'
        if ranks is not None:
            row += f' {f4(ranks[code]["mean"]) if ranked else ""} |'
        A(row)
    A('')


def render_metric_stats(prefix, paper_table, mean_table_id, entities, metrics, data, selection):
    for metric_name, key in metrics:
        table_id = f'{prefix}-{metric_name.replace("-", "").upper()}-STATS'
        table_note(
            table_id,
            paper_table,
            f'{metric_name} 5-seed 기술통계; 셀 순서 mean / sample std / min / max.',
            selection=selection,
            reference=mean_table_id,
        )
        A('| Method | ' + ' | '.join(name for name, _ in entities) + ' |')
        A('|---|' + '---|' * len(entities))
        for pname, code, *_ in MODELS:
            cells = [stat_cell(agg_stats(code, sub, key, data)) for _, sub in entities]
            A('| ' + pname + ' | ' + ' | '.join(cells) + ' |')
        A('')


def render_epoch_stats_table(table_id, paper_table, mean_table_id, entities, data, selection):
    table_note(
        table_id,
        paper_table,
        '선택 epoch 기술통계; 셀 순서 mean / sample std / min / max.',
        selection=selection,
        reference=mean_table_id,
    )
    A('| Method | ' + ' | '.join(name for name, _ in entities) + ' |')
    A('|---|' + '---|' * len(entities))
    for pname, code, *_ in MODELS:
        cells = []
        for _, sub in entities:
            stats = epoch_stats(code, sub, data)
            cells.append(stat_cell(stats) if stats else '')
        A('| ' + pname + ' | ' + ' | '.join(cells) + ' |')
    A('')


A('## 문서 구성과 논문 표 인덱스')
A('')
A('1. §0-2: 산출물 완결성, 데이터셋, 선택 규칙과 지표 정의.')
A('2. §3: 논문 Table 2의 ranked metrics. ES와 best-epoch를 분리하고, 각 기준에 mean 표와 PAK/VUS/Aff 통계표를 둔다.')
A('3. §4: 논문 Table A.6의 supplementary metrics. ES와 best-epoch 각각 PRC/VUS-ROC/oracle F1PA 통계표를 둔다.')
A('4. §5: 논문 Table A.5에 대응하는 baseline-only rank 분포. ES와 best-epoch를 별도 산출한다.')
A('5. §6: 역사적 30/15-epoch TEP Table 3 VUS-PR를 두 seed 축으로 분리한다. 논문 v22의 10/5-epoch 정본은 같은 디렉터리의 `results_tep_10_5.md`다.')
A('6. §7-10: Table A.2 구현 파라미터, 논문 인쇄값 대조, 추적 항목, per-seed 원값.')
A('')
A('| 논문 표 | 실험/내용 | 이 파일의 표 ID/섹션 | 논문 기입 상태 |')
A('|---|---|---|---|')
A('| Table 1 | baseline entity dataset statistics | `RB-T1-DATA` (§1) | 논문 기입 대상 |')
A('| Table 2 | 26 baselines × 4 entities × PAK/VUS/Aff | `RB-T2-ES-*` (§3, 기본), `RB-T2-BEST-*` (§3, 진단) | ES mean이 논문 기입 대상 |')
A('| Table 3 | TEP type-disjoint seen/unseen **VUS-PR** | 이 파일: 역사적 30/15 진단 `RB-T3-VUS-*` (§6); `results_tep_10_5.md`: v22 10/5 결과 | 논문 기입 후보는 `results_tep_10_5.md` |')
A('| Table 4 | LASAD component ablation | baseline 범위 밖; ablation 결과 문서에서 관리 | 이 파일에는 값 복제 안 함 |')
A('| Table A.1 | LASAD complete configuration | §2에 선택 규칙만 교차참조; 모델 본체 설정은 baseline 범위 밖 | 설정 표, 통계 비대상 |')
A('| Table A.2 | 26 baseline implementations/hyperparameters | `RB-A2-PARAMS` (§7) | 논문 기입 대상 |')
A('| Table A.3/A.4 | TEP dataset basics/fault taxonomy | `RB-A3-TEP-DATA`, `RB-A4-TEP-TAXONOMY` (§6) | 논문 기입 대상 |')
A('| Table A.5 | 12 Table-2 cell rank distribution | `RB-A5-ES-RANK`, `RB-A5-BEST-RANK` (§5) | baseline block; LASAD 행은 본체 결과와 결합 필요 |')
A('| Table A.6 | 26 baselines × 5 entities × PRC/V-R/PA | `RB-A6-ES-*` (§4, 기본), `RB-A6-BEST-*` (§4, 진단) | ES mean이 논문 기입 대상 |')
A('| Table C.1 | notation dictionary | 방법론 범위 밖; `LASAD.pdf` Table C.1 참조 | 통계 비대상 |')
A('')
A('통계 셀 표기 순서는 모든 통계표에서 **mean / sample std / min / max**로 고정한다. '
  '통계표는 자체 디렉터리를 반복하지 않고, 바로 위 `통계 참조`의 결과표 ID와 동일한 seed 인덱스를 사용한다.')
A('')

# --- seed/집계 셀 완결성 ---
A('## 0. 집계 셀 완결성')
A('')
table_note('RB-COVERAGE', 'baseline artifact audit', 'model/entity/seed 추출 셀 완결성.',
           paths=BASELINE_RESULT_PATHS)
A('| seed | unsup (8-k) | weak (9-k) |')
A('|---|---|---|')
for k, s in enumerate(SEEDS, 1):
    def _cell_stat(codes):
        total = len(codes) * len(A5_ENTITIES)
        done = sum(DATA[(code, sub)][s] is not None
                   for code in codes for _, sub in A5_ENTITIES)
        return f'{done}/{total} cells' + (' 완료' if done == total else '')
    unsup_codes = [code for _, code, *_ in MODELS if code not in WEAK]
    weak_codes = [code for _, code, *_ in MODELS if code in WEAK]
    A(f'| {s} | 8-{k}: {_cell_stat(unsup_codes)} | 9-{k}: {_cell_stat(weak_codes)} |')
A('')

# --- 1. dataset stats ---
A('## 1. 데이터셋 통계 (Table 1 대응 — UnifiedLoader 실측)')
A('')
table_note('RB-T1-DATA', 'Table 1', 'baseline 평가 entity의 split 이후 실측 통계.',
           paths=DATASET_PATHS, paper_ready=True)
A('| Entity | #Train | #Test | #Dim | Train AR (%) | Test AR (%) |')
A('|---|---|---|---|---|---|')
A('| SWaT | 719,959 | 224,960 | 45 | 1.63 | 19.05 (full) / 3.68 (excl22) |')
A('| WaDi A1 | 1,296,001 | 86,401 | 123 | 0.52 | 3.82 |')
A('| WaDi A2 | 870,972 | 86,402 | 123 | 0.76 | 3.87 |')
A('| PSM | 176,401 | 87,841 | 25 | 6.20 | 30.63 |')
A('')
A('- SWaT #Dim 45 = 51 − 6 상수 컬럼(P202, P401, P404, P502, P601, P603); WaDi 123 = 127 − 4 all-NaN 컬럼.')
A('- Split: ORIGINAL test의 시간순 앞 50% → train 편입(label 유지), 뒤 50% = evaluation. '
  'PSM은 train-label 파일 부재로 train부는 normal 취급. sliding window는 join 경계를 넘지 않음(segment 등록).')
A('- unsup(8계열)는 train에서 labeled anomaly region excise(normalonly); weak(9계열)는 full stream + labels.')
A('- excl22: held-out test의 시간순 첫 attack region(35,900 ts, anomaly mass 83.75%)을 metric 계산 전 제거. '
  '학습·raw score 불변, 모든 baseline identical mask.')
A('')

# --- 2. protocol ---
A('## 2. 프로토콜 요약')
A('')
A('- **Metrics (ranked, Table 2)**: PAK=`pak_auc_f1`(PA%K sweep AUC), VUS=`vus_pr`, '
  'Aff=`affiliation_f1_ar`(anomaly-ratio (1−α) quantile threshold — optimal-threshold 변형 아님).')
A('- **Metrics (supplementary, Table A.6)**: PRC=`prc_auc`, V-R=`vus_roc`, PA=`pa_0_f1`(oracle F1-PA, 랭킹 제외).')
A('- **Early stopping**: train_loss patience 3, min_delta 0, train split only, restore-best=min train_loss. '
  'NO_ES 5모델(train_loss 부재/상수)은 full budget 실행 후 final epoch 보고.')
A('- **Best epoch**: test-side PAK oracle 진단값이다. entity별 PAK 최대 epoch 하나를 선택하고 모든 지표를 그 epoch에서 읽는다. '
  '지표마다 서로 다른 epoch를 고르지 않는다.')
A('- **Rank**: Table 2의 4 entities × 3 metrics = 12개 mean 셀에서 내림차순 average-tie rank. '
  '이 파일의 rank는 26개 baseline끼리만 계산하므로 LASAD 행을 결합한 최종 논문 rank와 구분한다.')
A('- Baseline Table 2/A.6 기입값 = 5-seed mean(4자리). 통계표는 mean / sample std(ddof=1) / min / max를 별도 제공한다. '
  'TEP Table 3은 §6.1/6.2의 두 5-seed VUS-PR 축을 별도로 제공하며, 10/5-epoch 재실행 전까지 provisional로 표시한다.')
A('')

# --- 3. Table 2 ---
A('## 3. Main 결과 — Table 2 대응 (4 ranked entities × PAK/VUS/Aff)')
A('')
A('### 3.1 Early-stopping 기준 — 논문 기본')
A('')
render_mean_table(
    'RB-T2-ES-MEAN', 'Table 2',
    '26개 baseline의 PAK/VUS/Aff mean 행. Rank는 baseline-only 임시값.',
    RANKED_ENTITIES, RANKED_METRICS, DATA_ES, 'train-loss ES / NO_ES final / stateless single',
    paper_ready=True,
)
render_metric_stats('RB-T2-ES', 'Table 2', 'RB-T2-ES-MEAN', RANKED_ENTITIES,
                    RANKED_METRICS, DATA_ES, 'paper ES')
render_epoch_stats_table('RB-T2-ES-EPOCH-STATS', 'Table 2', 'RB-T2-ES-MEAN',
                         RANKED_ENTITIES, DATA_ES, 'paper ES')
A('### 3.2 Best-epoch 기준 — 진단용 oracle')
A('')
render_mean_table(
    'RB-T2-BEST-MEAN', 'Table 2 diagnostic',
    '동일 baseline 결과를 entity별 test PAK 최대 epoch에서 읽은 진단 mean.',
    RANKED_ENTITIES, RANKED_METRICS, DATA_BEST, 'entity test PAK argmax (single shared epoch)',
    paper_ready=False,
)
render_metric_stats('RB-T2-BEST', 'Table 2 diagnostic', 'RB-T2-BEST-MEAN',
                    RANKED_ENTITIES, RANKED_METRICS, DATA_BEST, 'best epoch oracle')
render_epoch_stats_table('RB-T2-BEST-EPOCH-STATS', 'Table 2 diagnostic',
                         'RB-T2-BEST-MEAN', RANKED_ENTITIES, DATA_BEST,
                         'best epoch oracle')
A('† `nrdetector_full`(noisy_rate=1.0 변형)은 논문 비수록·비랭킹 부가 행이다. '
  'Best-epoch 표는 원인 진단용이며 논문 Table 2 기본값으로 사용하지 않는다.')
A('')

# --- 4. Table A.6 ---
A('## 4. Supplementary 결과 — Table A.6 대응 (5 dataset groups × PRC/V-R/PA)')
A('')
A('### 4.1 Early-stopping 기준 — 논문 기본')
A('')
render_mean_table(
    'RB-A6-ES-MEAN', 'Table A.6',
    '26개 baseline의 PRC/VUS-ROC/oracle F1PA mean 행.',
    A5_ENTITIES, SUPP_METRICS, DATA_ES, 'train-loss ES / NO_ES final / stateless single',
    paper_ready=True,
)
render_metric_stats('RB-A6-ES', 'Table A.6', 'RB-A6-ES-MEAN', A5_ENTITIES,
                    SUPP_METRICS, DATA_ES, 'paper ES')
render_epoch_stats_table('RB-A6-ES-EPOCH-STATS', 'Table A.6', 'RB-A6-ES-MEAN',
                         A5_ENTITIES, DATA_ES, 'paper ES')
A('### 4.2 Best-epoch 기준 — 진단용 oracle')
A('')
render_mean_table(
    'RB-A6-BEST-MEAN', 'Table A.6 diagnostic',
    'Table A.6 지표를 entity별 test PAK 최대 epoch에서 함께 읽은 진단 mean.',
    A5_ENTITIES, SUPP_METRICS, DATA_BEST, 'entity test PAK argmax (single shared epoch)',
    paper_ready=False,
)
render_metric_stats('RB-A6-BEST', 'Table A.6 diagnostic', 'RB-A6-BEST-MEAN',
                    A5_ENTITIES, SUPP_METRICS, DATA_BEST, 'best epoch oracle')
render_epoch_stats_table('RB-A6-BEST-EPOCH-STATS', 'Table A.6 diagnostic',
                         'RB-A6-BEST-MEAN', A5_ENTITIES, DATA_BEST,
                         'best epoch oracle')

# --- 5. Table A.5 ---
A('## 5. Rank 분포 — Table A.5 대응')
A('')
for heading, table_id, mean_id, ranks, selection in [
    ('### 5.1 Early-stopping 기준', 'RB-A5-ES-RANK', 'RB-T2-ES-MEAN', RANKS_ES, 'paper ES'),
    ('### 5.2 Best-epoch 기준', 'RB-A5-BEST-RANK', 'RB-T2-BEST-MEAN', RANKS_BEST, 'best epoch oracle'),
]:
    A(heading)
    A('')
    table_note(table_id, 'Table A.5 baseline block',
               'Table 2의 12개 mean 셀 baseline-only rank 분포; mean / sample std / min(best) / max(worst).',
               selection=selection, reference=mean_id,
               reference_detail=(f'`{mean_id}`의 4 entities × 3 metrics mean 셀을 '
                                 'method별로 average-tie rank한 12개 rank를 집계.'))
    A('| Method | Mean rank | Sample std | Min (best) | Max (worst) | #rank cells |')
    A('|---|---|---|---|---|---|')
    for pname, code, _grp, _venue, ranked in MODELS:
        if not ranked:
            continue
        st = ranks[code]
        A(f'| {pname} | {f4(st["mean"])} | {f4(st["std"])} | '
          f'{f4(st["min"])} | {f4(st["max"])} | {st["n"]} |')
    A('')
A('> **중요**: 논문 Table A.5 최종값은 LASAD 행까지 포함해 다시 rank해야 한다. '
  '이 절은 baseline 결과의 완결성 확인용이며, baseline-only rank를 최종 논문 rank로 복사하면 안 된다.')
A('')

# --- 6. TEP Table 3 ---------------------------------------------------------
# The historical renderer below is retained temporarily for compatibility with
# its old helper functions, then discarded immediately before the definitive
# two-axis VUS-PR renderer.  This prevents legacy PAK tables from leaking into
# results_baseline.md while keeping this generator re-runnable during migration.
_TEP_SECTION_START = len(L)
# --- historical TEP renderer (discarded below) ------------------------------
A('## 6. TEP type-disjoint — Table 3 대응 (VUS-PR)')
A('')
A('> **논문 v22 기준**: TEP headline metric은 **VUS-PR**이며 논문 표 번호는 **Table 3**이다. '
  '각 Seen/Unseen 셀은 해당 fault mode의 VUS-PR을 비가중 평균한다. TEP에는 baseline §3-4의 ES/best-epoch 이원화를 적용하지 않는다.')
A('> **프로토콜 불일치 주의**: v22 Appendix A.3은 LASAD 10 epoch / Teacher-only 5 epoch를 명시하지만, '
  '현재 VUS 원천은 `TEP_phase2_win100_ep30`의 30/15-epoch 산출물에서 test PAK 기준 epoch를 골라 계산했다. '
  '따라서 아래 VUS 표는 현재 논문의 fold 값 원천을 재현하지만, 10/5-epoch 규약에 대한 최종 재실험 전까지 protocol-faithful 확정값으로 간주하지 않는다.')
A('')
A('### 6.0 TEP 데이터 정의 — Table A.3/A.4')
A('')
table_note('RB-A3-TEP-DATA', 'Table A.3', 'TEP type-disjoint protocol의 데이터 기본 통계.',
           paths=['scripts/TEP/data', 'scripts/TEP/data_dataseed{40,41,43,44}'],
           paper_ready=True)
A('| 항목 | 값 |')
A('|---|---|')
A('| Variables | 52 (41 measured + 11 manipulated) |')
A('| Fault IDs | 20 total; 17 used; IDV 3/9/15 excluded |')
A('| Fault families / folds | 5 families / 4 leave-one-family-out folds |')
A('| Run length / onset | 960 samples / fault onset 161 |')
A('| Train per fold | 240 fault-free + 60 seen-family faulty runs; contamination ≈16.7% |')
A('| Test per fault | 20 faulty + 40 shared fault-free runs; positive ratio ≈27.8% |')
A('')
table_note('RB-A4-TEP-TAXONOMY', 'Table A.4', 'TEP fault family와 type-disjoint fold 매핑.',
           paths=['scripts/TEP/data', 'scripts/TEP'], paper_ready=True)
A('| Fault family | IDV | Count | Evaluation fold |')
A('|---|---|---|---|')
A('| Step change | 1, 2, 4, 5, 6, 7 | 6 | F-STEP |')
A('| Random variation | 8, 10, 11, 12 | 4 | F-RAND |')
A('| Slow drift | 13 | 1 | F-DS |')
A('| Sticking | 14 | 1 | F-DS |')
A('| Unknown | 16, 17, 18, 19, 20 | 5 | F-UNK |')
A('| Excluded | 3, 9, 15 | 3 | 제외 |')
A('')
TEP_DIR = ROOT / 'results' / 'experiments' / 'TEP_phase2_win100_ep30'
TEP_SEEDS = [42, 43, 40, 41, 44]      # 규약: s42=table4_data.json, 기타=table4_data_s{seed}.json
_FOLDS = ['f_step', 'f_rand', 'f_ds', 'f_unk']


def _tep_load(s):
    p = TEP_DIR / ('table4_data.json' if s == 42 else f'table4_data_s{s}.json')
    return json.load(open(p)) if p.exists() else None


_teps = {s: _tep_load(s) for s in TEP_SEEDS}
# (label, section, json key, seed-dependent?)
TEP_ROWS = [('Random score', 'simple', 'Random', True),
            ('PCA recon.', 'simple', 'PCA recon.', False),
            ('NN-distance', 'simple', 'NN-distance', False),
            ('Sensor range', 'simple', 'Sensor range', False),
            ('L2-norm', 'simple', 'L2-norm', False),
            ('*(참조) Unlabeled (B)*', 'mae', 'B', True),
            ('*(참조) LASAD (A)*', 'mae', 'A', True),
            ('*(참조) Recon-only (D)*', 'mae', 'D', True),
            ('*(참조) clean ref (B0)*', 'mae', 'B0', True)]
_HDR6 = ('| Method | F-STEP Seen | F-STEP Unseen | F-RAND Seen | F-RAND Unseen '
         '| F-DS Seen | F-DS Unseen | F-UNK Seen | F-UNK Unseen |')
_SEP6 = '|---|---|---|---|---|---|---|---|---|'


def _summary(xs):
    """mean/sample-std/min/max summary with the same schema as baseline stats."""
    return {
        'mean': float(np.mean(xs)) if xs else None,
        'std': float(np.std(xs, ddof=1)) if len(xs) > 1 else None,
        'min': float(np.min(xs)) if xs else None,
        'max': float(np.max(xs)) if xs else None,
        'n': len(xs),
    }


# Paper v22 Table 3: canonical VUS-PR values.  vus_results.json stores exact
# per-fault values; vus_verify.json stores the audited per-fold S/U means for
# the LASAD conditions.  Do not substitute table4_data*.json here: those files
# contain pak_auc_f1, which is not Table 3's metric.
_VUS_RESULTS_PATH = TEP_DIR / 'vus_results.json'
_VUS_VERIFY_PATH = TEP_DIR / 'vus_verify.json'
_vus_results = json.load(open(_VUS_RESULTS_PATH)) if _VUS_RESULTS_PATH.exists() else {}
_vus_verify = json.load(open(_VUS_VERIFY_PATH)) if _VUS_VERIFY_PATH.exists() else {}
_FOLD_SHORT = {'f_step': 'fstep', 'f_rand': 'frand', 'f_ds': 'fds', 'f_unk': 'funk'}
_FOLD_INDEX = {fold: i for i, fold in enumerate(_FOLDS)}
_TEP_SEEN = {
    'f_step': [1, 2, 4, 5, 6, 7],
    'f_rand': [8, 10, 11, 12],
    'f_ds': [13, 14],
    'f_unk': [16, 17, 18, 19, 20],
}
_TEP_USABLE = sorted({f for faults in _TEP_SEEN.values() for f in faults})


def _tep_vus_pair(section, key, fold):
    """Return canonical (seen, unseen) VUS-PR for one Table-3 row/fold."""
    if section == 'simple':
        vals = _vus_results.get(f'simple_{key}_{_FOLD_SHORT[fold]}', {})
        seen = _TEP_SEEN[fold]
        unseen = [f for f in _TEP_USABLE if f not in set(seen)]
        sv = [vals.get(str(f)) for f in seen]
        uv = [vals.get(str(f)) for f in unseen]
        if not all(isinstance(v, (int, float)) for v in sv + uv):
            return None, None
        return float(np.mean(sv)), float(np.mean(uv))
    rows = (_vus_verify.get('permode', {}) or {}).get(key)
    if not rows:
        return None, None
    i = _FOLD_INDEX[fold]
    return float(rows[0][i]), float(rows[1][i])


TEP_VUS_ROWS = [
    ('Random', 'simple', 'Random'),
    ('Sensor range', 'simple', 'Sensor'),
    ('PCA recon.', 'simple', 'PCA'),
    ('L2-norm', 'simple', 'L2'),
    ('NN-distance', 'simple', 'NN'),
    ('Label-blind', 'mae', 'B'),
    ('w/o GRL', 'mae', 'nogrl'),
    ('Teacher-only', 'mae', 'D'),
    ('**LASAD (ours)**', 'mae', 'A'),
]
_HDR_VUS = _HDR6[:-1] + '| Avg Seen | Avg Unseen |'
_SEP_VUS = _SEP6[:-1] + '|---|---|'

A('### 6.1 논문 Table 3 — canonical VUS-PR')
A('')
table_note('RB-T3-VUS-CANONICAL', 'Table 3',
           'type-disjoint TEP의 per-fault-mode VUS-PR를 Seen/Unseen별 비가중 평균한 현재 논문 값.',
           selection='현재 30/15-epoch 원천의 저장 score + 기존 test-PAK epoch 선택 (재실험 필요)',
           paths=['results/experiments/TEP_phase2_win100_ep30/vus_results.json',
                  'results/experiments/TEP_phase2_win100_ep30/vus_verify.json',
                  'scripts/TEP/results/12_20260610_211815_tep_typegen_simple'],
           paper_ready=False)
A(_HDR_VUS)
A(_SEP_VUS)
_tep_vus_cache = {}
for label, section, key in TEP_VUS_ROWS:
    pairs = [_tep_vus_pair(section, key, fold) for fold in _FOLDS]
    _tep_vus_cache[key] = pairs
    seen = [p[0] for p in pairs if isinstance(p[0], (int, float))]
    unseen = [p[1] for p in pairs if isinstance(p[1], (int, float))]
    row = f'| {label} |'
    for s, u in pairs:
        row += f' {f4(s)} | {f4(u)} |'
    row += f' {f4(float(np.mean(seen)) if seen else None)} | {f4(float(np.mean(unseen)) if unseen else None)} |'
    A(row)

# Paper discriminants are LASAD minus Label-blind, calculated fold-wise.
_delta_unseen = []
_delta_gap = []
for i in range(len(_FOLDS)):
    a_s, a_u = _tep_vus_cache['A'][i]
    b_s, b_u = _tep_vus_cache['B'][i]
    _delta_unseen.append(a_u - b_u)
    _delta_gap.append((a_s - b_s) - (a_u - b_u))
for label, vals in [('Δunseen', _delta_unseen), ('Δgap = Δseen − Δunseen', _delta_gap)]:
    row = f'| {label} |'
    for v in vals:
        row += f' — | {v:+.4f} |'
    row += f' — | {float(np.mean(vals)):+.4f} |'
    A(row)
A('')
A('> 이 표의 fold 셀과 Avg는 저장된 비반올림 per-fault 값에서 다시 계산했다. 논문 v22의 3자리 인쇄값과 '
  '마지막 자리 0.001 차이가 있는 Avg 셀은 표시값 선반올림이 아니라 원시값 평균을 사용한 결과다.')
A('')
table_note('RB-T3-VUS-CANONICAL-STATS', 'Table 3 source audit',
           '현재 canonical 단일 run의 VUS-PR 기술통계; 셀 순서 mean / sample std / min / max.',
           selection='canonical source (n=1)', reference='RB-T3-VUS-CANONICAL')
A(_HDR6)
A(_SEP6)
for label, _section, key in TEP_VUS_ROWS:
    row = f'| {label} |'
    for s, u in _tep_vus_cache[key]:
        row += f' {stat_cell(_summary([s]) if s is not None else [])} |'
        row += f' {stat_cell(_summary([u]) if u is not None else [])} |'
    A(row)
A('')
A('> n=1이므로 sample std는 정의되지 않아 `—`로 표기한다. VUS-PR 5-seed 통계는 v22의 10/5-epoch 규약으로 재실행한 뒤 이 표를 대체해야 한다.')
A('')

A('### 6.2 기존 30-epoch PAK 진단 결과 — 논문 Table 3 비수록')
A('')
A('> 아래 6.2-A/B는 기존 `table4_data*.json`의 **PAK (`pak_auc_f1`)** 5-seed 점검값이다. '
  'TEP 논문 표에 넣지 않으며, VUS-PR Table 3의 대체값으로 사용하면 안 된다.')
A('')
A('#### 6.2-A. 데이터셋 고정 5-seed (데이터 canonical 고정, 학습 seed 42/43/40/41/44)')
A('')


def _tep_cell(sec, key, fold, su):
    """Fixed-data training-seed statistics."""
    xs = []
    for s in TEP_SEEDS:
        t = _teps.get(s)
        v = (((t or {}).get(sec, {}) or {}).get(key) or {}).get(fold, {}).get(su)
        if isinstance(v, (int, float)):
            xs.append(float(v))
    return _summary(xs)


A('##### 6.2-A.1 PAK mean (가용 seed mean; 결정적 모델은 seed 무관)')
A('')
table_note('RB-TEP-PAK-FIXED-MEAN', 'TEP PAK diagnostic (paper Table 3 비수록)',
           'canonical TEP 데이터 고정, 5개 학습 seed의 seen/unseen PAK mean 진단값.',
           selection='고정 데이터 + 학습 seed 평균',
           paths=['results/experiments/TEP_phase2_win100_ep30',
                  'scripts/TEP/results/12_20260610_211815_tep_typegen_simple'],
           paper_ready=False)
A(_HDR6)
A(_SEP6)
for label, sec, key, seeddep in TEP_ROWS:
    row = f'| {label}{"" if seeddep else " ᵈ"} |'
    for f in _FOLDS:
        for su in ('S', 'U'):
            st = _tep_cell(sec, key, f, su)
            row += f' {f4(st["mean"])} |'
    A(row)
A('')
A('ᵈ = deterministic(seed 무관, 재실험 불필요).')
A('')
A('##### 6.2-A.2 mean / sample std / min / max (seed-의존 행; B0 canonical n=1)')
A('')
table_note('RB-TEP-PAK-FIXED-STATS', 'TEP PAK diagnostic',
           '고정 데이터축 기술통계; 셀 순서 mean / sample std / min / max.',
           selection='고정 데이터 + 학습 seed', reference='RB-TEP-PAK-FIXED-MEAN')
A(_HDR6)
A(_SEP6)
for label, sec, key, seeddep in TEP_ROWS:
    if not seeddep:
        continue
    row = f'| {label} |'
    for f in _FOLDS:
        for su in ('S', 'U'):
            st = _tep_cell(sec, key, f, su)
            row += f' {stat_cell(st)} |'
    A(row)
A('')
A('> (n=k) 표기는 가용 seed 수 < 5. Random·A·B·D는 5-seed 완료. '
  'B0는 비교용 canonical clean reference만 유지해 n=1이다.')
A('')
A('##### 6.2-A.3 Per-seed 부록 (seed-의존 행 × seed)')
A('')
for s in TEP_SEEDS:
    have = _teps.get(s) is not None
    A(f'**seed {s}** {"" if have else "*(미실행 — 빈칸)*"}')
    A('')
    table_note(f'RB-TEP-PAK-FIXED-SEED-{s}', 'TEP PAK source appendix',
               f'고정 데이터축 seed {s} 원값.', selection=f'training seed {s}',
               paths=['results/experiments/TEP_phase2_win100_ep30'])
    A(_HDR6)
    A(_SEP6)
    for label, sec, key, seeddep in TEP_ROWS:
        if not seeddep:
            continue
        row = f'| {label} |'
        t = _teps.get(s)
        for f in _FOLDS:
            rec = (((t or {}).get(sec, {}) or {}).get(key) or {}).get(f, {})
            row += f" {f4(rec.get('S'))} | {f4(rec.get('U'))} |"
        A(row)
    A('')
A('- **소스**: `results/experiments/TEP_phase2_win100_ep30/table4_data{,_s40,_s41,_s43,_s44}.json` '
  '(`scripts/TEP/build_table4.py` 산출). '
  'simple 5행 원천 = `scripts/TEP/results/12_20260610_211815_tep_typegen_simple/<fold>/<model>/per_fault_metrics.json`, '
  'MAE행 = `pak_fill.json`(train mean-fixed scoring).')
A('- **seed 상태**: A/B 학습 5-seed와 D 파생, Random seeded 재계산이 완료됐다. '
  'PCA/NN/Sensor/L2는 canonical 데이터 고정축에서 deterministic이므로 동일 값을 재사용한다. '
  'B0는 비랭킹 clean reference로 canonical seed 42만 유지한다.')
A('- Fault taxonomy (Table A.4): Step={IDV 1,2,4,5,6,7}→F-STEP; Random-var={8,10,11,12}→F-RAND; '
  'Slow-drift={13}+Sticking={14}→F-DS; Unknown={16–20}→F-UNK; Excluded={3,9,15}. '
  'Fold: train=240 normal+60 seen-family faulty runs(오염≈16.7%), test/fault=20 faulty+40 shared normal(양성≈27.8%), '
  'run 960 samples, onset 161.')
A('')

# --- 6-B. 데이터셋 변경 5-seed (data-seed axis, 2026-07-24) -------------------
# run 할당 자체를 seed로 재추출(build_tep_data.py --data-seed N): canonical 42 +
# ds40/41/43/44. 결정적 모델도 데이터가 다르므로 seed-의존으로 취급.
A('#### 6.2-B. 데이터셋 변경 5-seed PAK 진단 (canonical 42 + data-seed 40/41/43/44)')
A('')
TEP_DS_SEEDS = [42, 40, 41, 43, 44]   # 규약: 42=canonical table4_data.json(대표), 기타=table4_data_ds{N}.json


def _tep_ds_load(s):
    p = TEP_DIR / ('table4_data.json' if s == 42 else f'table4_data_ds{s}.json')
    return json.load(open(p)) if p.exists() else None


_teps_ds = {s: _tep_ds_load(s) for s in TEP_DS_SEEDS}


def _tep_ds_cell(sec, key, fold, su):
    """Data-construction-seed statistics."""
    xs = []
    for s in TEP_DS_SEEDS:
        t = _teps_ds.get(s)
        v = (((t or {}).get(sec, {}) or {}).get(key) or {}).get(fold, {}).get(su)
        if isinstance(v, (int, float)):
            xs.append(float(v))
    return _summary(xs)


A('##### 6.2-B.1 robustness 확장 mean (이 축에선 결정적 모델도 데이터-의존 → 전 행 seed-의존)')
A('')
table_note('RB-TEP-PAK-DATASEED-MEAN', 'TEP PAK robustness diagnostic',
           '데이터 구성 seed까지 바꾼 5개 split의 seen/unseen PAK mean.',
           selection='data seed와 training seed 동기화',
           paths=['results/experiments/TEP_phase2_win100_ep30',
                  'results/experiments/TEP_phase2_win100_ep30_dataseed{40,41,43,44}',
                  'scripts/TEP/results/simple_dataseed{40,41,43,44}'],
           paper_ready=False)
A(_HDR6)
A(_SEP6)
for label, sec, key, _seeddep in TEP_ROWS:
    row = f'| {label} |'
    for f in _FOLDS:
        for su in ('S', 'U'):
            st = _tep_ds_cell(sec, key, f, su)
            row += f' {f4(st["mean"])} |'
    A(row)
A('')
A('##### 6.2-B.2 mean / sample std / min / max (전 행)')
A('')
table_note('RB-TEP-PAK-DATASEED-STATS', 'TEP PAK robustness diagnostic',
           '데이터 구성 seed축 기술통계; 셀 순서 mean / sample std / min / max.',
           selection='data seed와 training seed 동기화', reference='RB-TEP-PAK-DATASEED-MEAN')
A(_HDR6)
A(_SEP6)
for label, sec, key, _seeddep in TEP_ROWS:
    row = f'| {label} |'
    for f in _FOLDS:
        for su in ('S', 'U'):
            st = _tep_ds_cell(sec, key, f, su)
            row += f' {stat_cell(st)} |'
    A(row)
A('')
A('> (n=k) = 가용 data-seed 수 < 5. ds40/41/43/44 집계는 모두 완료됐으며, '
  'B0만 비교용 canonical clean reference라 n=1이다.')
A('')
A('##### 6.2-B.3 Per-seed 부록 (전 행 × data-seed)')
A('')
for s in TEP_DS_SEEDS:
    have = _teps_ds.get(s) is not None
    tag = ' *(canonical — 6-A seed 42와 동일 원천)*' if s == 42 else ''
    A(f'**data-seed {s}**{tag} {"" if have else "*(미실행 — 빈칸)*"}')
    A('')
    table_note(f'RB-TEP-PAK-DATASEED-{s}', 'TEP PAK robustness source appendix',
               f'data-seed {s} 원값.', selection=f'data/training seed {s}',
               paths=['results/experiments/TEP_phase2_win100_ep30' if s == 42
                      else f'results/experiments/TEP_phase2_win100_ep30_dataseed{s}',
                      'scripts/TEP/results/12_20260610_211815_tep_typegen_simple' if s == 42
                      else f'scripts/TEP/results/simple_dataseed{s}'])
    A(_HDR6)
    A(_SEP6)
    for label, sec, key, _seeddep in TEP_ROWS:
        row = f'| {label} |'
        t = _teps_ds.get(s)
        for f in _FOLDS:
            rec = (((t or {}).get(sec, {}) or {}).get(key) or {}).get(f, {})
            row += f" {f4(rec.get('S'))} | {f4(rec.get('U'))} |"
        A(row)
    A('')
A('- **소스**: data-seed 42 = `table4_data.json`(canonical 대표 — 데이터·학습 seed 모두 42), '
  'data-seed N∈{40,41,43,44} = `results/experiments/TEP_phase2_win100_ep30/table4_data_ds{N}.json` '
  '(`scripts/TEP/build_table4.py --dataseed N` 산출; MAE = `TEP_phase2_win100_ep30_dataseed{N}/pak_fill.json`, '
  'simple 5행 = `scripts/TEP/results/simple_dataseed{N}/`).')
A('- **축 정의**: run 할당 자체를 `np.random.default_rng(N)` 비복원 샘플링으로 재추출 '
  '(`scripts/TEP/build_tep_data.py --data-seed N` → `scripts/TEP/data_dataseed{N}/`; '
  '집합 크기·fold 구성·onset·스트림 배치 규칙은 canonical과 동일). 학습 seed도 N으로 동기화'
  '(`scripts/run_tep_dataseed.sh`).')
A('- **⚠ 이 축에선 PCA/NN/Sensor/L2도 seed-의존**(데이터가 다름) — 6-A의 ᵈ 표기 미적용. '
  'B0는 data-seed 축 미실행(canonical 42만 존재 → n=1 유지). '
  'Random 행은 seed N 단일 draw 원값(`per_fault_by_seed.json[N]`).')
A('')

# Discard the historical canonical-VUS + PAK renderer above.  The definitive
# output below contains VUS-PR for both requested seed axes.
del L[_TEP_SECTION_START:]

TEP_VUS_FIXED_PATH = TEP_DIR / 'table3_vus_fixed_seed.json'
TEP_VUS_DATA_PATH = TEP_DIR / 'table3_vus_data_seed.json'
TEP_VUS_FIXED = json.load(open(TEP_VUS_FIXED_PATH)) if TEP_VUS_FIXED_PATH.exists() else {}
TEP_VUS_DATA = json.load(open(TEP_VUS_DATA_PATH)) if TEP_VUS_DATA_PATH.exists() else {}
TEP_VUS_ROWS = [
    ('Random', 'simple', 'Random'),
    ('Sensor range', 'simple', 'Sensor range'),
    ('PCA recon.', 'simple', 'PCA recon.'),
    ('L2-norm', 'simple', 'L2-norm'),
    ('NN-distance', 'simple', 'NN-distance'),
    ('Label-blind', 'mae', 'Label-blind'),
    ('w/o GRL', 'mae', 'w/o GRL'),
    ('Teacher-only', 'mae', 'Teacher-only'),
    ('**LASAD (ours)**', 'mae', 'LASAD'),
]
TEP_VUS_HEADER = (_HDR6[:-1] + '| Avg Seen | Avg Unseen |')
TEP_VUS_SEP = (_SEP6[:-1] + '|---|---|')


def _validate_tep_vus_axis(axis, expected_axis):
    issues = []
    if axis.get('axis') != expected_axis:
        issues.append(f'axis={axis.get("axis")!r}, expected={expected_axis!r}')
    if axis.get('run_boundary_handling') not in (None, 'legacy_concat'):
        issues.append(f'run_boundary_handling={axis.get("run_boundary_handling")!r}')
    for seed in axis.get('seeds', []):
        run = axis.get('runs', {}).get(str(seed), {})
        for _display, section, label in TEP_VUS_ROWS:
            for fold in _FOLDS:
                cell = (((run.get(section, {}) or {}).get(label, {}) or {}).get(fold, {}) or {})
                if not all(isinstance(cell.get(side), (int, float)) and np.isfinite(cell[side])
                           for side in ('S', 'U')):
                    issues.append(f'seed={seed} {section}/{label}/{fold}')
                per_fault = cell.get('per_fault', {})
                if len(per_fault) != len(_TEP_USABLE) or not all(
                        isinstance(value, (int, float)) and np.isfinite(value)
                        for value in per_fault.values()):
                    issues.append(f'seed={seed} {section}/{label}/{fold}/per_fault')
    if issues:
        raise RuntimeError('incomplete historical TEP VUS axis:\n' + '\n'.join(issues))


_validate_tep_vus_axis(TEP_VUS_FIXED, 'fixed_model_seed')
_validate_tep_vus_axis(TEP_VUS_DATA, 'data_and_model_seed')


def _tep_axis_values(axis, section, label, fold=None, su=None):
    values = []
    for seed in axis.get('seeds', []):
        entry = (((axis.get('runs', {}).get(str(seed), {}).get(section, {}) or {})
                  .get(label, {}) or {}))
        if fold is None:
            xs = [entry.get(f, {}).get(su) for f in _FOLDS]
            if all(isinstance(x, (int, float)) for x in xs):
                values.append(float(np.mean(xs)))
        else:
            value = entry.get(fold, {}).get(su)
            if isinstance(value, (int, float)):
                values.append(float(value))
    return values


def _tep_axis_seed_value(axis, seed, section, label, fold=None, su=None):
    entry = (((axis.get('runs', {}).get(str(seed), {}).get(section, {}) or {})
              .get(label, {}) or {}))
    if fold is None:
        xs = [entry.get(f, {}).get(su) for f in _FOLDS]
        return float(np.mean(xs)) if all(isinstance(x, (int, float)) for x in xs) else None
    value = entry.get(fold, {}).get(su)
    return float(value) if isinstance(value, (int, float)) else None


def _tep_axis_paths(axis_name, seed=None):
    if axis_name == 'fixed':
        paths = ['results/experiments/TEP_phase2_win100_ep30/table3_vus_fixed_seed.json']
        if seed is not None:
            paths += [
                'results/experiments/TEP_phase2_win100_ep30' if seed == 42
                else f'results/experiments/TEP_phase2_win100_ep30_s{seed}',
                'scripts/TEP/results/12_20260610_211815_tep_typegen_simple',
            ]
        return paths
    paths = ['results/experiments/TEP_phase2_win100_ep30/table3_vus_data_seed.json']
    if seed is not None:
        paths += [
            'results/experiments/TEP_phase2_win100_ep30' if seed == 42
            else f'results/experiments/TEP_phase2_win100_ep30_dataseed{seed}',
            'scripts/TEP/data' if seed == 42 else f'scripts/TEP/data_dataseed{seed}',
            'scripts/TEP/results/12_20260610_211815_tep_typegen_simple' if seed == 42
            else f'scripts/TEP/results/simple_dataseed{seed}',
        ]
    return paths


def _render_tep_vus_axis(axis, axis_name, section_number, title, table_prefix,
                         paper_label, selection, paper_candidate):
    A(f'### {section_number}. {title}')
    A('')
    table_note(f'{table_prefix}-MEAN', paper_label,
               'per-fault-mode VUS-PR를 fold별 Seen/Unseen으로 비가중 평균한 뒤 seed 평균한 결과.',
               selection=selection, paths=_tep_axis_paths(axis_name),
               paper_ready=paper_candidate)
    A(TEP_VUS_HEADER)
    A(TEP_VUS_SEP)
    for display, section, label in TEP_VUS_ROWS:
        row = f'| {display} |'
        for fold in _FOLDS:
            for su in ('S', 'U'):
                st = _summary(_tep_axis_values(axis, section, label, fold, su))
                row += f' {f4(st["mean"])} |'
        for su in ('S', 'U'):
            st = _summary(_tep_axis_values(axis, section, label, None, su))
            row += f' {f4(st["mean"])} |'
        A(row)
    A('')
    A('#### mean / sample std / min / max')
    A('')
    table_note(f'{table_prefix}-STATS', f'{paper_label} statistics',
               'VUS-PR seed축 기술통계; 셀 순서 mean / sample std / min / max.',
               selection=selection, reference=f'{table_prefix}-MEAN')
    A(TEP_VUS_HEADER)
    A(TEP_VUS_SEP)
    for display, section, label in TEP_VUS_ROWS:
        row = f'| {display} |'
        for fold in _FOLDS:
            for su in ('S', 'U'):
                row += f' {stat_cell(_summary(_tep_axis_values(axis, section, label, fold, su)))} |'
        for su in ('S', 'U'):
            row += f' {stat_cell(_summary(_tep_axis_values(axis, section, label, None, su)))} |'
        A(row)
    A('')
    A('> `w/o GRL`도 다른 learned condition과 동일하게 각 축의 5개 seed에서 독립 학습했다.')
    A('')

    # Fold-wise transfer discriminants, aggregated over the same seed index.
    A('#### LASAD − Label-blind transfer discriminants')
    A('')
    disc_rows = [('Δunseen', 'delta_unseen'), ('Δgap = Δseen − Δunseen', 'delta_gap')]
    table_note(f'{table_prefix}-DELTA-MEAN', f'{paper_label} discriminants',
               'seed별 LASAD−Label-blind 차이를 먼저 계산한 뒤 평균한 VUS-PR transfer discriminants.',
               selection=selection, paths=_tep_axis_paths(axis_name))
    A('| Discriminant | F-STEP | F-RAND | F-DS | F-UNK | Four-fold Avg |')
    A('|---|---|---|---|---|---|')
    for display, key in disc_rows:
        fold_values = []
        for fold in _FOLDS:
            xs = []
            for seed in axis.get('seeds', []):
                v = (((axis.get('runs', {}).get(str(seed), {}).get('discriminant', {}) or {})
                      .get(fold, {}) or {}).get(key))
                if isinstance(v, (int, float)):
                    xs.append(float(v))
            fold_values.append(xs)
        avg_by_seed = []
        for seed in axis.get('seeds', []):
            xs = [
                (((axis.get('runs', {}).get(str(seed), {}).get('discriminant', {}) or {})
                  .get(fold, {}) or {}).get(key))
                for fold in _FOLDS
            ]
            if all(isinstance(v, (int, float)) for v in xs):
                avg_by_seed.append(float(np.mean(xs)))
        A('| ' + display + ' | ' + ' | '.join(
            [f'{_summary(xs)["mean"]:+.4f}' if xs else '' for xs in fold_values] +
            [f'{_summary(avg_by_seed)["mean"]:+.4f}' if avg_by_seed else '']) + ' |')
    A('')
    table_note(f'{table_prefix}-DELTA-STATS', f'{paper_label} discriminant statistics',
               'seed별 discriminant 기술통계; 셀 순서 mean / sample std / min / max.',
               selection=selection, reference=f'{table_prefix}-DELTA-MEAN')
    A('| Discriminant | F-STEP | F-RAND | F-DS | F-UNK | Four-fold Avg |')
    A('|---|---|---|---|---|---|')
    for display, key in disc_rows:
        fold_values = []
        for fold in _FOLDS:
            xs = []
            for seed in axis.get('seeds', []):
                v = (((axis.get('runs', {}).get(str(seed), {}).get('discriminant', {}) or {})
                      .get(fold, {}) or {}).get(key))
                if isinstance(v, (int, float)):
                    xs.append(float(v))
            fold_values.append(xs)
        avg_by_seed = []
        for seed in axis.get('seeds', []):
            xs = [
                (((axis.get('runs', {}).get(str(seed), {}).get('discriminant', {}) or {})
                  .get(fold, {}) or {}).get(key))
                for fold in _FOLDS
            ]
            if all(isinstance(v, (int, float)) for v in xs):
                avg_by_seed.append(float(np.mean(xs)))
        A('| ' + display + ' | ' + ' | '.join(
            [stat_cell(_summary(xs)) for xs in fold_values] +
            [stat_cell(_summary(avg_by_seed))]) + ' |')
    A('')

    A('#### Per-seed 원값')
    A('')
    for seed in axis.get('seeds', []):
        table_note(f'{table_prefix}-SEED-{seed}', f'{paper_label} source appendix',
                   f'{title} seed {seed} VUS-PR 원값.', selection=f'seed {seed}',
                   paths=_tep_axis_paths(axis_name, seed))
        A(TEP_VUS_HEADER)
        A(TEP_VUS_SEP)
        for display, section, label in TEP_VUS_ROWS:
            row = f'| {display} |'
            for fold in _FOLDS:
                row += (f' {f4(_tep_axis_seed_value(axis, seed, section, label, fold, "S"))} |'
                        f' {f4(_tep_axis_seed_value(axis, seed, section, label, fold, "U"))} |')
            row += (f' {f4(_tep_axis_seed_value(axis, seed, section, label, None, "S"))} |'
                    f' {f4(_tep_axis_seed_value(axis, seed, section, label, None, "U"))} |')
            A(row)
        A('')


A('## 6. TEP type-disjoint — Table 3 VUS-PR 두 seed 축')
A('')
A('> **지표**: 논문 v22 Table 3과 동일하게 각 fault mode의 `vus_pr`를 먼저 계산하고, fold 내부 Seen/Unseen fault에 대해 비가중 평균한다. '
  '`table4_data*.json`의 PAK 값은 이 절에서 사용하지 않는다.')
A('> **두 축**: (A) canonical 데이터 분할 고정 + 모델/random seed만 5개, '
  '(B) 데이터 분할 seed와 모델/random seed를 함께 바꾼 5개를 분리한다. 두 축을 섞어 하나의 표준편차로 만들지 않는다.')
A('> **프로토콜 주의**: 아래 값은 완료된 30/15-epoch score 산출물의 VUS-PR 재계산이다. '
  '역사적 집계는 fault별 run을 이어 붙인 뒤 tolerance를 적용했으므로 run-boundary reset도 하지 않았다. '
  '논문 v22의 10/5-epoch + run-boundary-reset 규약 결과는 별도 `results_tep_10_5.md`를 정본으로 사용한다.')
A('')
A('### 6.0 TEP 데이터 정의 — Table A.3/A.4')
A('')
table_note('RB-A3-TEP-DATA', 'Table A.3', 'TEP type-disjoint protocol의 데이터 기본 통계.',
           paths=['scripts/TEP/data', 'scripts/TEP/data_dataseed{40,41,43,44}'],
           paper_ready=True)
A('| 항목 | 값 |')
A('|---|---|')
A('| Variables | 52 (41 measured + 11 manipulated) |')
A('| Fault IDs | 20 total; 17 used; IDV 3/9/15 excluded |')
A('| Fault families / folds | 5 families / 4 leave-one-family-out folds |')
A('| Run length / onset | 960 samples / fault onset 161 |')
A('| Train per fold | 240 fault-free + 60 seen-family faulty runs; contamination ≈16.7% |')
A('| Test per fault | 20 faulty + 40 shared fault-free runs; positive ratio ≈27.8% |')
A('')
table_note('RB-A4-TEP-TAXONOMY', 'Table A.4', 'TEP fault family와 type-disjoint fold 매핑.',
           paths=['scripts/TEP/data', 'scripts/TEP'], paper_ready=True)
A('| Fault family | IDV | Count | Evaluation fold |')
A('|---|---|---|---|')
A('| Step change | 1, 2, 4, 5, 6, 7 | 6 | F-STEP |')
A('| Random variation | 8, 10, 11, 12 | 4 | F-RAND |')
A('| Slow drift | 13 | 1 | F-DS |')
A('| Sticking | 14 | 1 | F-DS |')
A('| Unknown | 16, 17, 18, 19, 20 | 5 | F-UNK |')
A('| Excluded | 3, 9, 15 | 3 | 제외 |')
A('')

_render_tep_vus_axis(
    TEP_VUS_FIXED, 'fixed', '6.1',
    '데이터 분할 고정 + 모델 seed 5개', 'RB-T3-VUS-FIXED',
    'Table 3 model-seed axis',
    'canonical 데이터 고정; model/random seed {42,43,40,41,44}', False,
)
_render_tep_vus_axis(
    TEP_VUS_DATA, 'data', '6.2',
    '데이터 분할 seed + 모델 seed 동시 변경 5개', 'RB-T3-VUS-DATASEED',
    'Table 3 data-seed robustness axis',
    'data-allocation seed와 model/random seed 동기화 {42,40,41,43,44}', False,
)
A('> 기존 `table4_data{,_s*,_ds*}.json` PAK 집계는 보존하지만 논문 Table 3 결과표와 통계에서는 제외한다.')
A('')

# --- 7. parameter table ---
A('## 7. 하이퍼파라미터 — Table A.2 대응 + 코드 실측 대조')
A('')
PAPER_PARAMS = [
    # (paper name, window, lr, batch, max_ep, key params)
    ('Random score', '500', '—', '—', '—', '5-run average'),
    ('Sensor range', '500', '—', '—', '—', 'per-feature range'),
    ('PCA recon.', '500', '—', '—', '—', 'auto components (10/30) *(정정 2026-07-19: 기존 "50 components")*'),
    ('L2-norm', '500', '—', '—', '—', 'window L2'),
    ('NN-distance', '500', '—', '—', '—', '1-NN *(정정 2026-07-19: 기존 "5 neighbors")*'),
    ('MLP', '5', '0.001', '512', '50', 'embed dim 32'),
    ('MLPMixer', '5', '0.0002', '512', '50', 'embed dim 128'),
    ('Transformer', '5', '0.001', '512', '50', 'embed dim 128'),
    ('GCN-LSTM', '5', '0.001', '100', '50', 'LSTM units 64'),
    ('Anomaly Trans.', '100', '0.0001', '128', '10', 'model dim 512'),
    ('TranAD', '10', '0.0001', '128', '10', 'FF dim 16'),
    ('USAD', '5', '0.001', '256', '10', 'latent dim 40'),
    ('DAGMM (simpl.)', '5', '0.0001', '256', '10', 'GMM omitted'),
    ('GDN', '5', '0.001', '32', '10', 'embed dim 64'),
    ('OmniAnomaly', '100', '0.001', '50', '10', 'hidden 500'),
    ('TFMAE', '100', '0.0001', '64', '10', 'model dim 128'),
    ('NPSR', '100', '0.0001', '64', '10', 'latent dim 10'),
    ('TimesNet', '100', '0.0001', '128', '10', 'model dim 128'),
    ('DCdetector', '105', '0.0001', '128', '10', 'model dim 256'),
    ('MEMTO', '100', '0.0001', '128', '10', 'memory 10, model 512'),
    ('ModernTCN', '100', '0.0003', '128', '10', 'channel dim 128'),
    ('CATCH', '192', '0.0001', '128', '10', 'model dim 128'),
    ('DeepMIL', '128', '0.0001', '60', '50', 'encoder dim 128'),
    ('WETAS', '500', '0.0001', '32', '50', 'hidden 128'),
    ('TreeMIL', '500', '0.0001', '32', '50', 'model dim 128'),
    ('NRdetector', '100', 'clf 1e-5 / enc 1e-4 *(정정 2026-07-19)*', '32', '50',
     'encoder dim 64; PU 공개율 = 양성 window의 40% *(point-label 비율 아님)*; '
     'BCE backstop 제거 *(upstream 충실도 정정 2026-07-25)*'),
]
NAME2CODE = {p: c for p, c, *_ in MODELS}
pv_path = AUX_DIR / 'param_verification.json'
pverif = json.load(open(pv_path)) if pv_path.exists() else {}
table_note('RB-A2-PARAMS', 'Table A.2',
           '26개 baseline의 논문 implementation/hyperparameter 정리.',
           paths=['comparison/baselines', 'comparison/configs',
                  'comparison/results/experiments/_aux'], paper_ready=True)
if pverif:
    A('| Method | Venue | Window | LR | Batch | Max ep (논문) | Max ep (실측·reseed) | Key params | 코드 일치 | 편차 |')
    A('|---|---|---|---|---|---|---|---|---|---|')
else:
    A('| Method | Venue | Window | LR | Batch | Max ep (논문) | Key params |')
    A('|---|---|---|---|---|---|---|')
for (pname, w, lr, b, ep, kp) in PAPER_PARAMS:
    code = NAME2CODE.get(pname, '')
    venue = next((v for p, c, g, v, r in MODELS if p == pname), '')
    if pverif:
        pv = pverif.get(code, {})
        act_ep = pv.get('max_epochs_actual', '')
        match = {True: '✅', False: '⚠'}.get(pv.get('matches_paper'), '')
        dev = '; '.join(pv.get('deviations', [])) if pv else ''
        A(f'| {pname} | {venue} | {w} | {lr} | {b} | {ep} | {act_ep} | {kp} | {match} | {dev} |')
    else:
        A(f'| {pname} | {venue} | {w} | {lr} | {b} | {ep} | {kp} |')
A('')
if not pverif:
    A('> `_aux/param_verification.json`은 현재 없으므로 비어 있는 실측 대조 열을 만들지 않았다. '
      '위 표의 논문 기입값은 모두 채워져 있으며, 확인된 실행 편차는 아래 정정/실측 주석에 별도 기록한다.')
A('')
A('### 7.1 Weak-supervised label 공개 규칙')
A('')
A('- 네 weak-supervised wrapper에는 contaminated training stream의 **전체 point annotation(100%)**이 전달된다. '
  'DeepMIL은 128-timestep bag, WETAS와 TreeMIL은 500-timestep window 내부 point label의 max로 '
  'coarse label을 만들며, 이렇게 파생된 bag/window label을 전부 사용한다.')
A('- **NRdetector의 `noisy_rate=0.4`는 point label 40%가 아니다.** 각 registered training segment를 '
  '100-timestep 비중첩 window로 나누고, anomaly point가 하나라도 포함된 양성 window 수를 시간순으로 `P`라 할 때 '
  'PU 단계에서 첫 `floor(0.4P)`개만 labeled-positive로 공개한다. 나머지 양성 window와 모든 음성 window는 '
  'unlabeled pool로 들어간다.')
A('- 현재 checkpoint-free NRdetector adapter의 선행 encoder는 전체 파생 window-label vector로 학습한다. '
  '따라서 40% 제한은 **PU selection 및 stage-2 classifier supervision의 labeled-positive subset**에 적용된다. '
  '이는 ground-truth training label이 supervised subset 구성에 직접 쓰이는 구체적 사례다.')
A('')
A('- **정정 결정(2026-07-19, 사용자)**: ①PCA components(auto 10/30) ②NN-distance(1-NN) ⑤NRdetector LR(clf/enc 병기) = '
  '**논문 A.2 정정 확정** ↔ ③epoch-cap(catch 2·dcdetector 1·timesnet/moderntcn/omni 5) ④batch divisor(OOM cap) = '
  '**미적용**(논문 표 현행 유지, 실측 편차는 본 표의 실측 컬럼/편차 컬럼으로만 기록).')
A('- **NRdetector 충실도 정정(2026-07-25)**: upstream에 없는 BCE backstop 0.05를 제거하고 encoder LR 1e-4를 '
  '유지해 40/40 refix를 완료. float32 mean 1-ulp gate 경계는 float64 산술로 정정.')
A('- ES 정책: patience 3 / train split only / restore-best=min train_loss / NO_ES 5모델 full-epoch. Max epochs=상한 budget.')
A('- 실행 환경(논문 A.1): RTX 4090 단일, CUDA 11.8, cuDNN 9.1, PyTorch 2.4.1+cu118, Python 3.10.')
A('- LASAD 설정(Table A.1 참조): L=500, s=10, N=50, stride 1/49, ρ=0.15, epochs 30/warmup 15, '
  'AdamW lr 1e-3 wd 1e-3, batch 1024, bf16, w=0.25, β_FM=1.0, β_GRL=0.2, focal γ=2, seeds=5.')
A('')

# --- 8. reconciliation vs printed draft values ---
A('## 8. 논문 인쇄값 대조 (draft reconciliation)')
A('')
pr_path = AUX_DIR / 'printed_values.json'
if pr_path.exists():
    printed = json.load(open(pr_path))
    t2 = printed.get('table2', {})
    A('Table 2 (PAK/VUS/Aff, |재산출mean − 인쇄값| ≥ 0.02인 셀만 표시):')
    A('')
    table_note('RB-T2-RECONCILE', 'Table 2 audit',
               'draft 인쇄값과 ES reseed mean의 절대편차 0.02 이상 셀.',
               selection='paper ES',
               paths=['comparison/results/experiments/_aux',
                      *BASELINE_RESULT_PATHS])
    A('| Method | Entity | Metric | 인쇄값 | 재산출 mean | Δ |')
    A('|---|---|---|---|---|---|')
    n_diff = 0
    ENT_MAP = {'SWaT_excl22': 'SWaT/A1A2_excl22', 'WaDi_A1': 'WaDi/A1',
               'WaDi_A2': 'WaDi/A2', 'PSM': 'PSM'}
    for pname, code, *_ in MODELS:
        prow = t2.get(pname) or t2.get(pname.replace(' (simpl.)', ''))
        if not prow:
            continue
        for ent_p, sub in ENT_MAP.items():
            cellp = prow.get(ent_p) or {}
            for mn, key in RANKED_METRICS:
                pval = cellp.get(mn)
                if not isinstance(pval, (int, float)):
                    continue
                m, _, n, _ = agg(code, sub, key)
                if m is not None and abs(m - pval) >= 0.02:
                    n_diff += 1
                    A(f'| {pname} | {ent_p} | {mn} | {pval:.4f} | {m:.4f} | {m - pval:+.4f} |')
    A('')
    A(f'요약: 인쇄값 대비 |Δ|≥0.02 셀 {n_diff}개. '
      '(편차 원인 후보: draft 인쇄값은 무seed 원본 8/9 + test-side best-epoch 프로토콜 산출 — '
      '본 문서는 5-seed mean + ES-epoch 프로토콜. dcdetector는 neg 전환으로 전 셀 갱신 대상.)')
else:
    A('*(printed_values.json 미생성 — 전사 워크플로우 완료 후 재실행 시 자동 포함)*')
A('')

# --- 9. tracker ---
A('## 9. 미해결 사항 트래커')
A('')
A('| # | 항목 | 상태 |')
A('|---|---|---|')
A('| 1 | reseed 체인 10/10 + TEP 5-seed(40/41/43/44) 전체 완료(2026-07-24) — '
  'catch 재시도 성공으로 원본 baseline 675/675 복원·TEP §6 n=5 '
  '+ nrdetector refix 40/40 재생성 완료(2026-07-25) | 완료 |')
A('| 1b | **GCN-LSTM 채택 근거(2026-07-24)**: keras_init 재실험(5-seed×4entity) vs 기존 deadhead를 '
  '4-entity 5-seed mean pak으로 비교 → kerasinit 0.3826 vs deadhead 0.3831(사실상 동률, Δ0.0005) → '
  '규약대로 **deadhead(기존값) 채택**. entity별 상반: kerasinit이 SWaT excl22 +0.047·WaDi_A1 +0.010, '
  'deadhead가 WaDi_A2 +0.056·PSM +0.004. kerasinit 결과는 각 실험 dir `gcn_lstm__kerasinit/` 영구 보존, '
  'deadhead 백업은 `.trash/260724/gcnlstm_pre_redo/` | 확정 |')
A('| 2 | catch×WaDi_A1 seed42: 최초 런은 비결정적 NaN 발산으로 결측이었으나 '
  '**프로토콜 내 재시도 1회가 NaN 없이 2/2 epoch 완주**(2026-07-25) → 해당 셀 및 5-seed 집계 **n=5 복원** | 확정 |')
A('| 3 | Aff(`affiliation_f1_ar`)=0.0 케이스의 정체: 이산/동률 스코어(예: random의 binary {0,1}, dcdetector-neg의 tie-mass)에서 '
  'AR quantile threshold + strict `>` 비교가 예측 0개를 만들어 0.0 — **draft도 Random Aff=0.0을 인쇄(정합 확인)**. '
  'dcdetector-neg Aff는 seed별 0/비0 혼재(tie 경계 민감) — 논문 기입 시 각주 권장 | 확정(각주) |')
A('| 4 | epoch-cap 편차(논문 Max ep 10 vs 실측 cap): **미적용 결정(2026-07-19 사용자)** — 논문 표 유지, 실측은 §7 기록 | 종결 |')
A('| 5 | NRdetector LR: **정정 확정(2026-07-19)** — A.2에 clf 1e-5/enc 1e-4 병기 | 종결 |')
A('| 5b | NRdetector refix(2026-07-25): upstream 비충실 BCE backstop 제거 + gate float64 정정 후 40/40 완주. '
  '50개 산출물 모두 50 epoch·NaN/Inf 0; 최종 PAK 0 셀은 8→1로 감소했으나 '
  '`seed40 nrdetector_full×WaDi_A1` 1건은 완료 후 0으로 남아 결과 감사 항목으로 유지 | 완료·1건 감사 |')
A('| 6 | LASAD(ours)/(label-blind)/(excised) 행: MAE 파이프라인 결합 대기. rank는 불필요 결정(2026-07-19) | MAE 소관 |')
A('| 7 | TEP Table 3 VUS-PR: 데이터 고정+모델 seed 5개와 데이터 분할+모델 seed 동시 변경 5개를 분리 집계했다. '
  'w/o GRL까지 전 seed를 독립 학습했으며 각 축 180 fold entries·non-finite 0, mean/std/min/max·seed별 원값을 검증한다. '
  '다만 원천 score는 30/15-epoch·test-PAK 선택이므로 v22 10/5-epoch 결과는 별도 `results_tep_10_5.md`에 분리한다 | '
  '30/15 완결·10/5 별도 결과 |')
A('| 8 | in-text 파생 수치 동반 갱신 목록: §4.2 weak 4-entity 평균(TreeMIL 0.55/0.32, DeepMIL 0.49/0.26, WETAS 0.40/0.22, '
  'NRdet 0.36/0.15), NPSR excl22 0.7465·0.5556, §A.5 인용(MLPMixer 0.8988·0.9752, NPSR 0.5717·0.8896) — 셀 확정 시 재계산 | 목록 확보 |')
A('| 8b | GCN-LSTM Keras-init 재실험과 채택 판단은 #1b로 완료. 현재 frozen protocol은 기존 deadhead 결과를 사용하며, '
  '원본 QuoVadis validation-controller 재현은 별도 방법론 실험으로 분리 | 종결 |')
A('| 9 | excl22는 `compute_metrics_with_exclusion` → `compute_full_metric_set`에서 masked scores/labels를 사용하므로 '
  'Affiliation 및 anomaly-ratio threshold 모두 제외 후 재계산됨 | 확인 완료 |')
A('| 10 | 요구사항 스펙의 seed 표기 "42–46"은 구버전 — 실제 {42,43,40,41,44} (2026-07-11 확정) | 정정 완료 |')
A('')

# --- 10. per-seed appendix ---
A('## 10. Per-seed 원값 부록 — Early-stopping 기준 (PAK / VUS / Aff)')
A('')
for en, sub in A5_ENTITIES:
    A(f'### {en}')
    A('')
    slug = sub.replace('/', '-').replace('_', '-')
    is_ranked_entity = sub in {entity_sub for _, entity_sub in RANKED_ENTITIES}
    table_note(f'{"RB-T2" if is_ranked_entity else "RB-DIAG"}-ES-PERSEED-{slug}',
               'Table 2 source appendix' if is_ranked_entity else 'non-paper diagnostic appendix',
               f'{en}의 ES-selected seed별 PAK/VUS/Aff 원값.', selection='paper ES',
               paths=BASELINE_RESULT_PATHS)
    h = '| Method |'
    s2 = '|---|'
    for mn, _ in RANKED_METRICS:
        for sd_ in SEEDS:
            h += f' {mn} s{sd_} |'
            s2 += '---|'
    A(h)
    A(s2)
    for pname, code, *_ in MODELS:
        row = f'| {pname} |'
        for mn, key in RANKED_METRICS:
            _, _, _, per = agg(code, sub, key)
            for sd_ in SEEDS:
                row += f' {f4(per.get(sd_))} |'
        A(row)
    A('')

# --- sanity + dump ---
n_cells = sum(1 for (c, sub) in DATA_ES for s in SEEDS if DATA_ES[(c, sub)][s])
n_best_cells = sum(1 for (c, sub) in DATA_BEST for s in SEEDS if DATA_BEST[(c, sub)][s])
A('---')
A(f'*생성 통계: 추출 셀 {n_cells} / {len(MODELS) * len(A5_ENTITIES) * len(SEEDS)} '
  f'(ES), {n_best_cells} / {len(MODELS) * len(A5_ENTITIES) * len(SEEDS)} (best epoch); '
  f'모델 {len(MODELS)} × entity {len(A5_ENTITIES)} × seed {len(SEEDS)}. 경고 {len(warnings)}건.*')
if warnings:
    A('')
    A('<details><summary>경고 목록</summary>')
    A('')
    for w in warnings[:50]:
        A(f'- {w}')
    A('</details>')

OUT_MD.write_text('\n'.join(L))
def dump_selected(data, path):
    dump = {}
    for (code, sub), per_seed in data.items():
        for seed, selected in per_seed.items():
            if selected:
                dump[f'{code}|{sub}|{seed}'] = {
                    'epoch': selected['epoch'], 'n_ep': selected['n_ep'],
                    **{key: selected['vals'].get(key) for key in ALL_KEYS},
                }
    json.dump(dump, open(path, 'w'), indent=1)
    return dump


dump_es = dump_selected(DATA_ES, OUT_JSON)
dump_best = dump_selected(DATA_BEST, OUT_BEST_JSON)
print(f'WROTE {OUT_MD} ({len(L)} lines) + {OUT_JSON} ({len(dump_es)} cells) + '
      f'{OUT_BEST_JSON} ({len(dump_best)} cells); warnings={len(warnings)}')
for w in warnings[:15]:
    print('  WARN:', w)
