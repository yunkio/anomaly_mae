#!/usr/bin/env python
"""LASAD paper — baseline results aggregator.

Generates comparison/results/experiments/results.md from the reseed 5-seed
experiments (8-1..8-5 unsup / 9-1..9-5 weak), following the paper protocol
(LASAD.pdf A.1.1) and the requirements spec extracted 2026-07-19:

  * Seeds {42,43,40,41,44}  (8-k/9-k share SEEDS[k], k=1..5)
  * Epoch selection: trained models -> ES-selected epoch = argmin(train_loss)
    (paper: native train-loss early stopping, restore-best = min train loss).
    NO_ES models (no usable train_loss) -> FINAL epoch.  Stateless -> single eval.
    NOTE: max-pak_auc_f1 best-epoch (test-side) is deliberately NOT used.
  * Metrics: Table 2 = pak_auc_f1 / vus_pr / affiliation_f1_ar
             Table A.5 = prc_auc / vus_roc / pa_0_f1
  * dcdetector: 'neg' convention (score = -canonical) — read from
    _dcdetector_neg_cache.json (built by the sign-inversion pipeline; mtime-guarded,
    incremental for future 8-4/8-5 cells).
  * Missing cells (experiments not yet finished) stay blank; mean±std over
    available seeds with n annotated when n<5.

Re-runnable: as more seeds finish, re-run to fill blanks.
Aux inputs (optional, from the requirements workflow):
  scratchpad .../lasad_requirements/printed_values.json   (paper printed values)
  scratchpad .../lasad_requirements/param_verification.json (code-vs-paper params)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # repo root
EXP = ROOT / 'comparison' / 'results' / 'experiments'
OUT_MD = EXP / 'results.md'
OUT_JSON = EXP / 'results_data.json'
NEG_CACHE = EXP / '_dcdetector_neg_cache.json'
AUX_DIR = Path('/tmp/claude-1000/-home-ykio-notebooks-claude/'
               'e7052938-c36a-454b-99c0-00488e1852d8/scratchpad/lasad_requirements')

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


def cell(code, seed, sub):
    """One (model, seed, entity) cell -> {vals:{key:val}, epoch, n_ep} or None."""
    if code == 'dcdetector':
        return _dc_neg(seed, sub)
    base = WEAK_DIR[seed] if code in WEAK else UNSUP_DIR[seed]
    if sub == 'SWaT/A1A2_excl22':
        f_eps = _load_epochs(base / 'SWaT/A1A2_full' / code / 'epoch_metrics.json')
        if f_eps is None:
            return None
        idx = select_epoch_idx(f_eps, code)     # epoch chosen on the NATIVE run
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
    idx = select_epoch_idx(eps, code)
    return {'vals': {k: eps[idx].get(k) for k in ALL_KEYS},
            'epoch': idx + 1, 'n_ep': len(eps)}


# ---------------- collect ----------------
DATA = {}          # (code, sub) -> {seed: cell}
for _, code, *_ in MODELS:
    for _, sub in A5_ENTITIES:
        DATA[(code, sub)] = {s: cell(code, s, sub) for s in SEEDS}


def agg(code, sub, key):
    """-> (mean, std, n, per_seed{seed:val})"""
    per = {}
    for s in SEEDS:
        c = DATA[(code, sub)][s]
        v = c['vals'].get(key) if c else None
        if isinstance(v, (int, float)):
            per[s] = float(v)
    if not per:
        return None, None, 0, per
    xs = list(per.values())
    mean = float(np.mean(xs))
    std = float(np.std(xs, ddof=1)) if len(xs) > 1 else None
    return mean, std, len(xs), per


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


# ---------------- ranks: 미산출 (2026-07-19 사용자 결정 — rank 행 불필요) ----------------


# ---------------- render ----------------
L = []
A = L.append
A('# LASAD — Baseline Experiment Results (5-seed)')
A('')
A('> **생성**: `comparison/build_results_md.py` (재실행 시 완료된 seed 자동 반영). ')
A('> **정본 데이터**: reseed 체인 `8-1..8-5` (unsup, anomaly-excised) / `9-1..9-5` (weak, contaminated), '
  'seeds **{42, 43, 40, 41, 44}** (8-k·9-k 공유; 논문 Table A.1의 `[n]`→**5**). ')
A('> **Epoch 선택**(논문 A.1.1): trained → train-loss ES restore-best(=argmin train_loss), '
  f'NO_ES {sorted(NO_ES)} → final epoch, stateless → 단일 평가. test-side(best-pak) 선택 미사용. ')
A('> **DCdetector = `neg` convention**: score = −canonical (부호 반전 검증 2026-07-19에 따른 사용자 결정). '
  '`_dcdetector_neg_cache.json`에서 로드 (mtime-guard 증분). ')
A('> 빈칸 = 실험 미완료 (8-4 진행 중 / 9-4·8-5·9-5 대기). `(n=k)` = 가용 seed 수 < 5. std = 표본표준편차(ddof=1).')
A('')

# --- seed/실험 완료 현황 ---
A('## 0. 실험 진행 현황')
A('')
A('| seed | unsup (8-k) | weak (9-k) |')
A('|---|---|---|')
for k, s in enumerate(SEEDS, 1):
    def _stat(base, total):
        qs = base / 'queue_status.json'
        if not qs.exists():
            return '미시작'
        q = json.load(open(qs))
        return f"{q['queue_done']}/{q['queue_total']} done, err {q['queue_error']}"
    A(f'| {s} | 8-{k}: {_stat(UNSUP_DIR[s], 88)} | 9-{k}: {_stat(WEAK_DIR[s], 20)} |')
A('')

# --- 1. dataset stats ---
A('## 1. 데이터셋 통계 (Table 1 대응 — UnifiedLoader 실측)')
A('')
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
A('- **Metrics (supplementary, Table A.5)**: PRC=`prc_auc`, V-R=`vus_roc`, PA=`pa_0_f1`(oracle F1-PA, 랭킹 제외).')
A('- **Early stopping**: train_loss patience 3, min_delta 0, train split only, restore-best=min train_loss. '
  'NO_ES 5모델(train_loss 부재/상수)은 full budget 실행 후 final epoch 보고.')
A('- **Rank**: 미산출 — 2026-07-19 사용자 결정(rank 행 불필요).')
A('- 논문 기입값 = mean(4자리). 본 문서는 mean±std 병기.')
A('')

# --- 3. Table 2 ---
A('## 3. Main 결과 — Table 2 대응 (4 ranked entities × PAK/VUS/Aff)')
A('')
hdr = '| Method |'
sep = '|---|'
for en, _ in RANKED_ENTITIES:
    for mn, _ in RANKED_METRICS:
        hdr += f' {en}<br>{mn} |'
        sep += '---|'
A('### 3.1 논문 기입용 (mean)')
A('')
A(hdr)
A(sep)
for pname, code, grp, venue, ranked in MODELS:
    row = f'| {pname}{"†" if not ranked else ""} |'
    for _, sub in RANKED_ENTITIES:
        for _, key in RANKED_METRICS:
            m, sd, n, _ = agg(code, sub, key)
            row += f' {f4(m)} |'
    A(row)
A('')
A('† `nrdetector_full`(noisy_rate=1.0 변형)은 논문 비수록·비랭킹 부가 행 (paired 정책 유지).')
A('')
A('### 3.2 mean ± std 상세')
A('')
A(hdr)
A(sep)
for pname, code, grp, venue, ranked in MODELS:
    row = f'| {pname} |'
    for _, sub in RANKED_ENTITIES:
        for _, key in RANKED_METRICS:
            m, sd, n, _ = agg(code, sub, key)
            row += f' {ms(m, sd, n)} |'
    A(row)
A('')

# --- 4. Table A.5 ---
A('## 4. Supplementary 결과 — Table A.5 대응 (5 dataset groups × PRC/V-R/PA)')
A('')
hdr5 = '| Method |'
sep5 = '|---|'
for en, _ in A5_ENTITIES:
    for mn, _ in SUPP_METRICS:
        hdr5 += f' {en}<br>{mn} |'
        sep5 += '---|'
A('### 4.1 논문 기입용 (mean)')
A('')
A(hdr5)
A(sep5)
for pname, code, grp, venue, ranked in MODELS:
    row = f'| {pname}{"†" if not ranked else ""} |'
    for _, sub in A5_ENTITIES:
        for _, key in SUPP_METRICS:
            m, sd, n, _ = agg(code, sub, key)
            row += f' {f4(m)} |'
    A(row)
A('')
A('### 4.2 mean ± std 상세')
A('')
A(hdr5)
A(sep5)
for pname, code, grp, venue, ranked in MODELS:
    row = f'| {pname} |'
    for _, sub in A5_ENTITIES:
        for _, key in SUPP_METRICS:
            m, sd, n, _ = agg(code, sub, key)
            row += f' {ms(m, sd, n)} |'
    A(row)
A('')

# --- 5. Table A.6: 미산출 ---
A('## 5. Rank 분포 — 미산출')
A('')
A('> 2026-07-19 사용자 결정: rank 행(Table 2 rank·mean Rank·Table A.6) 불필요 — 산출하지 않음.')
A('')

# --- 6. TEP Table 4 (source: results/experiments/TEP_phase2_win100_ep30/table4_data.json) ---
A('## 6. TEP type-disjoint — Table 4 대응 (simple 5 baselines 40셀 + MAE 참조행)')
A('')
TEP_T4 = ROOT / 'results' / 'experiments' / 'TEP_phase2_win100_ep30' / 'table4_data.json'
_tep = json.load(open(TEP_T4)) if TEP_T4.exists() else None
_FOLDS = ['f_step', 'f_rand', 'f_ds', 'f_unk']
A('| Method | F-STEP Seen | F-STEP Unseen | F-RAND Seen | F-RAND Unseen | F-DS Seen | F-DS Unseen | F-UNK Seen | F-UNK Unseen |')
A('|---|---|---|---|---|---|---|---|---|')


def _tep_row(label, rec):
    row = f'| {label} |'
    for f in _FOLDS:
        c = (rec or {}).get(f) or {}
        row += f" {f4(c.get('S'))} | {f4(c.get('U'))} |"
    A(row)


for pname, jkey in [('Random score', 'Random'), ('PCA recon.', 'PCA recon.'),
                    ('NN-distance', 'NN-distance'), ('Sensor range', 'Sensor range'),
                    ('L2-norm', 'L2-norm')]:
    _tep_row(pname, (_tep or {}).get('simple', {}).get(jkey))
for label, jkey in [('*(참조) Unlabeled (B)*', 'B'), ('*(참조) LASAD (A)*', 'A'),
                    ('*(참조) Recon-only (D)*', 'D'), ('*(참조) clean ref (B0)*', 'B0')]:
    _tep_row(label, (_tep or {}).get('mae', {}).get(jkey))
A('')
A('- **소스**: `results/experiments/TEP_phase2_win100_ep30/table4_data.json` (`scripts/TEP/build_table4.py` 산출). '
  'simple 5행 원천 = `scripts/TEP/results/12_20260610_211815_tep_typegen_simple/<fold>/<model>/per_fault_metrics.json`, '
  'MAE행 = `pak_fill.json`(train mean-fixed scoring).')
A('- **⚠ seed 상태**: 전 셀 **single-seed(42)**. simple 5 중 PCA/NN/Sensor/L2는 deterministic(seed 무관)이나 '
  '**Random 행은 unseeded 단일 draw**(run_tep_simple.py:197-232, 5-run 평균 아님 — 결함, 재실행 시 seeded 5-draw 필요). '
  '5-seed 확장 필요분: MAE 36 GPU runs(~16 GPU-h: A×16, B×16, B0×4; D는 파생) + Random seeded 재실행(CPU) + '
  'pak_fill/build_table4 다중-seed 집계 확장.')
A('- Fault taxonomy (Table A.4): Step={IDV 1,2,4,5,6,7}→F-STEP; Random-var={8,10,11,12}→F-RAND; '
  'Slow-drift={13}+Sticking={14}→F-DS; Unknown={16–20}→F-UNK; Excluded={3,9,15}. '
  'Fold: train=240 normal+60 seen-family faulty runs(오염≈16.7%), test/fault=20 faulty+40 shared normal(양성≈27.8%), '
  'run 960 samples, onset 161.')
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
    ('NRdetector', '100', 'clf 1e-5 / enc 1e-4 *(정정 2026-07-19)*', '32', '50', 'encoder dim 64, BCE backstop 0.05'),
]
NAME2CODE = {p: c for p, c, *_ in MODELS}
pv_path = AUX_DIR / 'param_verification.json'
pverif = json.load(open(pv_path)) if pv_path.exists() else {}
A('| Method | Venue | Window | LR | Batch | Max ep (논문) | Max ep (실측·reseed) | Key params | 코드 일치 | 편차 |')
A('|---|---|---|---|---|---|---|---|---|---|')
for (pname, w, lr, b, ep, kp) in PAPER_PARAMS:
    code = NAME2CODE.get(pname, '')
    venue = next((v for p, c, g, v, r in MODELS if p == pname), '')
    pv = pverif.get(code, {})
    act_ep = pv.get('max_epochs_actual', '')
    match = {True: '✅', False: '⚠'}.get(pv.get('matches_paper'), '')
    dev = '; '.join(pv.get('deviations', [])) if pv else ''
    A(f'| {pname} | {venue} | {w} | {lr} | {b} | {ep} | {act_ep} | {kp} | {match} | {dev} |')
A('')
A('- **정정 결정(2026-07-19, 사용자)**: ①PCA components(auto 10/30) ②NN-distance(1-NN) ⑤NRdetector LR(clf/enc 병기) = '
  '**논문 A.2 정정 확정** ↔ ③epoch-cap(catch 2·dcdetector 1·timesnet/moderntcn/omni 5) ④batch divisor(OOM cap) = '
  '**미적용**(논문 표 현행 유지, 실측 편차는 본 표의 실측 컬럼/편차 컬럼으로만 기록).')
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
A('| 1 | 8-4(seed41) 진행 중, 9-4·8-5(seed44)·9-5 대기 — 완료 시 본 스크립트 재실행 | 진행 |')
A('| 2 | catch×WaDi_A1 seed42: 학습 NaN 발산으로 **결측**(재현 불가 fail) → 해당 셀 n=4 | 확정(각주) |')
A('| 3 | Aff(`affiliation_f1_ar`)=0.0 케이스의 정체: 이산/동률 스코어(예: random의 binary {0,1}, dcdetector-neg의 tie-mass)에서 '
  'AR quantile threshold + strict `>` 비교가 예측 0개를 만들어 0.0 — **draft도 Random Aff=0.0을 인쇄(정합 확인)**. '
  'dcdetector-neg Aff는 seed별 0/비0 혼재(tie 경계 민감) — 논문 기입 시 각주 권장 | 확정(각주) |')
A('| 4 | epoch-cap 편차(논문 Max ep 10 vs 실측 cap): **미적용 결정(2026-07-19 사용자)** — 논문 표 유지, 실측은 §7 기록 | 종결 |')
A('| 5 | NRdetector LR: **정정 확정(2026-07-19)** — A.2에 clf 1e-5/enc 1e-4 병기 | 종결 |')
A('| 6 | LASAD(ours)/(label-blind)/(excised) 행: MAE 파이프라인 결합 대기. rank는 불필요 결정(2026-07-19) | MAE 소관 |')
A('| 7 | TEP Table 4: 소스 확정·값 채움 완료(§6, 2026-07-19 조사). 잔여 = single-seed(42) 상태 — 5-seed 확장 여부 사용자 결정 대기(필요분 §6 명기) | 조사 완료 |')
A('| 8 | in-text 파생 수치 동반 갱신 목록: §4.2 weak 4-entity 평균(TreeMIL 0.55/0.32, DeepMIL 0.49/0.26, WETAS 0.40/0.22, '
  'NRdet 0.36/0.15), NPSR excl22 0.7465·0.5556, §A.5 인용(MLPMixer 0.8988·0.9752, NPSR 0.5717·0.8896) — 셀 확정 시 재계산 | 목록 확보 |')
A('| 8b | GCN-LSTM 인쇄값(WaDi PAK ~0.64)은 dead-head 붕괴 前 era(exp6, wd=1e-4) 산출 — 현 5-seed 재현치(~0.23-0.26)와 구조적 격차. '
  '충실도 감사(2026-07-19)의 dead ReLU head 발견과 일치. **결정(2026-07-19): 옵션 A — Keras-init 이식+dead-head guard 후 재실험**(체인 완주 후 실행, 실패 시 wd=1e-4 fallback) | **A 승인·실행 대기** |')
A('| 9 | Affiliation α의 excl22 재계산 여부 (`compute_metrics_with_exclusion`) 코드 확인 | 확인 필요 |')
A('| 10 | 요구사항 스펙의 seed 표기 "42–46"은 구버전 — 실제 {42,43,40,41,44} (2026-07-11 확정) | 정정 완료 |')
A('')

# --- 10. per-seed appendix ---
A('## 10. Per-seed 원값 부록 (PAK / VUS / Aff)')
A('')
for en, sub in A5_ENTITIES:
    A(f'### {en}')
    A('')
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
n_cells = sum(1 for (c, sub) in DATA for s in SEEDS if DATA[(c, sub)][s])
A('---')
A(f'*생성 통계: 추출 셀 {n_cells} / {len(MODELS) * len(A5_ENTITIES) * len(SEEDS)} '
  f'(모델 {len(MODELS)} × entity {len(A5_ENTITIES)} × seed {len(SEEDS)}). 경고 {len(warnings)}건.*')
if warnings:
    A('')
    A('<details><summary>경고 목록</summary>')
    A('')
    for w in warnings[:50]:
        A(f'- {w}')
    A('</details>')

OUT_MD.write_text('\n'.join(L))
# machine-readable dump
dump = {}
for (code, sub), per_seed in DATA.items():
    for s, c in per_seed.items():
        if c:
            dump[f'{code}|{sub}|{s}'] = {'epoch': c['epoch'], 'n_ep': c['n_ep'], **{k: c['vals'].get(k) for k in ALL_KEYS}}
json.dump(dump, open(OUT_JSON, 'w'), indent=1)
print(f'WROTE {OUT_MD} ({len(L)} lines) + {OUT_JSON} ({len(dump)} cells); warnings={len(warnings)}')
for w in warnings[:15]:
    print('  WARN:', w)
