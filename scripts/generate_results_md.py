#!/usr/bin/env python
"""Generate results/experiments/official/results.md — every number needed for the LASAD paper (v22) tables.

Deterministic + idempotent: re-run any time (`conda activate dc_vis && python scripts/generate_results_md.py`);
it re-scans results/experiments/official/, picks the newest COMPLETE run dir per (condition, seed), extracts
metrics at the per-cell recon_snr early-stop epoch (the paper's convention — NOT the oracle pak-best), and
rewrites results.md. Pending (condition, seed) runs render as "—" and fill in automatically as the campaign
(sens3seed -> paper5seed -> p5sens) completes them.

READ-ONLY on everything except results.md. Hard verification anchors are asserted before writing; on any
anchor failure the script exits non-zero WITHOUT touching results.md.

Epoch-selection recipe (recon_snr ES, per cell):
  - series = training_histories.json["0"]["train_recon_snr"] (1-indexed epochs)
  - warmup = best_config.json["teacher_only_warmup_epochs"] (verified per cell)
  - stream e = warmup+1 .. last: ema = snr_e (first) else 0.2*snr_e + 0.8*ema;
    best-so-far ema with patience 2; ES epoch = best epoch (bep), not the stop epoch.
  - PROXY: runs with no labeled train anomalies (blind, exclanom, unlab100 stand-in) have
    train_recon_snr = None -> use the SAME SEED's baseline ES epoch for that cell.
  - SWaT/A1A2_excl22 shares training with SWaT/A1A2_full -> use the full cell's ES epoch,
    read excl22's own epoch_metrics.json at that epoch.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

# ----------------------------------------------------------------------------- constants
OFFICIAL = "/home/ykio/notebooks/TSMAE/results/experiments/official"
OUT_PATH = os.path.join(OFFICIAL, "results.md")

# entity name -> cell subdir
CELLS = [
    ("SWaT_full", "SWaT/A1A2_full"),
    ("SWaT_ranked", "SWaT/A1A2_excl22"),
    ("WaDi_A1", "WaDi/A1"),
    ("WaDi_A2", "WaDi/A2"),
    ("PSM", "PSM"),
]
CELL_DIR = dict(CELLS)
ALL_ENTITIES = [e for e, _ in CELLS]
TRAIN_ENTITIES = ["SWaT_full", "WaDi_A1", "WaDi_A2", "PSM"]  # own training histories
PAPER_ENTITIES = ["SWaT_ranked", "WaDi_A1", "WaDi_A2", "PSM"]  # Table 2 / 4 / B.1 entities
B2_ENTITIES = ["SWaT_full", "SWaT_ranked", "WaDi_A1", "WaDi_A2", "PSM"]  # Table B.2 incl SWaT_full

SEEDS = [40, 41, 42, 43, 44]
EMA_ALPHA = 0.2
ES_PATIENCE = 2

METRICS_MAIN = [("PAK", "pak_auc_f1"), ("VUS-PR", "vus_pr"), ("Aff-F1", "affiliation_f1_ar")]
METRICS_SUPP = [("PRC-AUC", "prc_auc"), ("VUS-ROC", "vus_roc"), ("PA%K=0 F1", "pa_0_f1")]
METRIC_KEYS = [k for _, k in METRICS_MAIN + METRICS_SUPP]

# condition key -> (dir tag or None for baseline, paper name, exact config override vs baseline)
CONDITIONS = {
    "baseline":  (None,        "LASAD (ours)",             "— (paper defaults, full LASAD)"),
    "exclanom":  ("exclanom",  "Positive-excised",         "`train_exclude_anomaly_segments=True`"),
    "blind":     ("blind",     "Label-blind",              "`blind_train_labels=True`"),
    "nogrl":     ("nogrl",     "w/o GRL",                  "`use_grl=False anomaly_loss_weight=0`"),
    "nofm":      ("nofm",      "w/o FM",                   "`use_feature_matching=False`"),
    "noforce":   ("noforce",   "w/o anomaly-priority masking", "`force_mask_anomaly=False`"),
    "nostudent": ("nostudent", "w/o Student (Teacher-only)", "`use_student=False`"),
    "td3sd3":    ("td3sd3",    "Symmetric decoders (3/3)", "`num_teacher_decoder_layers=3 num_student_decoder_layers=3`"),
    "unlab10r":  ("unlab10r",  "10% unlabeled",            "`train_label_mask_frac=0.10 train_label_mask_random=True`"),
    "unlab25r":  ("unlab25r",  "25% unlabeled",            "`train_label_mask_frac=0.25 train_label_mask_random=True`"),
    "unlab50r":  ("unlab50r",  "50% unlabeled",            "`train_label_mask_frac=0.50 train_label_mask_random=True`"),
    "unlab75r":  ("unlab75r",  "75% unlabeled",            "`train_label_mask_frac=0.75 train_label_mask_random=True`"),
    "maskr005":  ("maskr005",  "rho=0.05",                 "`masking_ratio=0.05`"),
    "maskr010":  ("maskr010",  "rho=0.10",                 "`masking_ratio=0.10`"),
    "maskr030":  ("maskr030",  "rho=0.30",                 "`masking_ratio=0.30`"),
    "maskr050":  ("maskr050",  "rho=0.50",                 "`masking_ratio=0.50`"),
    "maskr060":  ("maskr060",  "rho=0.60",                 "`masking_ratio=0.60`"),
    "maskr075":  ("maskr075",  "rho=0.75",                 "`masking_ratio=0.75`"),
    "maskr090":  ("maskr090",  "rho=0.90",                 "`masking_ratio=0.90`"),
}
# blind @ seed42 stand-in (user-approved functional equivalence): unlab100 tag dir
BLIND_STANDIN_SEED = 42
BLIND_STANDIN_TAG = "unlab100"

ABLATION_CONDS = ["baseline", "nogrl", "nofm", "noforce", "nostudent", "td3sd3"]
SPARSITY_LEVELS = [  # (label, condition key)
    ("0% (= baseline)", "baseline"),
    ("10%", "unlab10r"),
    ("25%", "unlab25r"),
    ("50%", "unlab50r"),
    ("75%", "unlab75r"),
    ("100% (= blind)", "blind"),
]
RHO_LEVELS = [
    ("0.05", "maskr005"),
    ("0.10", "maskr010"),
    ("0.15 (= baseline)", "baseline"),
    ("0.30", "maskr030"),
    ("0.50", "maskr050"),
    ("0.60", "maskr060"),
    ("0.75", "maskr075"),
    ("0.90", "maskr090"),
]
W_VALUES = ["0.1", "0.25 (= baseline)", "0.5", "0.75", "1", "1.5", "2"]

# pending-queue order (mirrors scripts/run_official_{sens3seed,paper5seed,paper5seed_sens}_after.py)
LS_TAGS = ["unlab10r", "unlab25r", "unlab50r", "unlab75r"]
RHO_TAGS = ["maskr005", "maskr010", "maskr030", "maskr050", "maskr060", "maskr075", "maskr090"]
P5_TAGS = ["exclanom", "blind", "nogrl", "nofm", "noforce", "nostudent", "td3sd3"]


def build_queue():
    q = []
    for block in (LS_TAGS, RHO_TAGS):          # sens3seed: sparsity block then rho block, seed-major
        for seed in (40, 41):
            for t in block:
                q.append((t, seed, "sens3seed"))
    for seed in (40, 41, 43, 44):              # paper5seed: seed-major over excised->blind->ablations
        for t in P5_TAGS:
            q.append((t, seed, "paper5seed"))
    for block in (LS_TAGS, RHO_TAGS):          # p5sens: sparsity block then rho block, seeds 43,44
        for seed in (43, 44):
            for t in block:
                q.append((t, seed, "p5sens"))
    return q


# ----------------------------------------------------------------------------- data access
def _dir_pattern(seed, tag):
    if tag is None:
        return re.compile(r"271_\d{8}_\d{6}_30ep_%d" % seed + r"$")
    return re.compile(r"271_\d{8}_\d{6}_30ep_%d_%s" % (seed, re.escape(tag)) + r"$")


def _is_complete(run_dir):
    return all(os.path.exists(os.path.join(run_dir, sub, "epoch_metrics.json")) for _, sub in CELLS)


_ALL_DIRS = None


def all_dirs():
    global _ALL_DIRS
    if _ALL_DIRS is None:
        _ALL_DIRS = sorted(
            d for d in os.listdir(OFFICIAL) if os.path.isdir(os.path.join(OFFICIAL, d)) and d.startswith("271_")
        )
    return _ALL_DIRS


def find_run(seed, tag):
    """Newest COMPLETE dir matching (seed, tag); also report whether a partial (in-flight) dir exists."""
    pat = _dir_pattern(seed, tag)
    matches = [d for d in all_dirs() if pat.match(d)]
    complete = [d for d in matches if _is_complete(os.path.join(OFFICIAL, d))]
    partial = [d for d in matches if d not in complete]
    return (complete[-1] if complete else None), partial  # name sort == timestamp sort


def load_json(*parts):
    with open(os.path.join(*parts)) as fh:
        return json.load(fh)


def recon_snr_series(cell_dir):
    th = load_json(cell_dir, "training_histories.json")
    inner = th.get("0", th[next(iter(th))])
    return inner.get("train_recon_snr")


def compute_es_epoch(snr, warmup):
    """recon_snr ES recipe. Returns best epoch (1-indexed) or None if series missing/degenerate."""
    if snr is None or len(snr) <= warmup:
        return None
    post = snr[warmup:]
    if any(v is None or not isinstance(v, (int, float)) or not math.isfinite(v) for v in post):
        return None
    if len(set(post)) == 1 and len(post) > 1:  # constant series = degenerate
        return None
    ema = None
    best = None
    bep = None
    wait = 0
    for e in range(warmup + 1, len(snr) + 1):
        v = snr[e - 1]
        ema = v if ema is None else EMA_ALPHA * v + (1.0 - EMA_ALPHA) * ema
        if best is None or ema > best:
            best, bep, wait = ema, e, 0
        else:
            wait += 1
            if wait >= ES_PATIENCE:
                break
    return bep


def metrics_by_epoch(cell_dir):
    recs = load_json(cell_dir, "epoch_metrics.json")["epochs"]
    return {r["epoch"]: r for r in recs}


def load_run(dir_name, baseline_es=None):
    """Extract per-entity ES epochs + metric values for one run dir.

    baseline_es: same-seed baseline {entity: es_epoch} used as PROXY when train_recon_snr is
    unavailable (label-free runs). Returns dict(dir, es, proxy_entities, metrics, warmup).
    """
    run_dir = os.path.join(OFFICIAL, dir_name)
    es, proxy = {}, []
    warmups = {}
    for ent in TRAIN_ENTITIES:
        cell = os.path.join(run_dir, CELL_DIR[ent])
        warmup = load_json(cell, "best_config.json")["teacher_only_warmup_epochs"]
        warmups[ent] = warmup
        bep = compute_es_epoch(recon_snr_series(cell), warmup)
        if bep is None:
            if baseline_es is None:
                raise RuntimeError(f"{dir_name}/{ent}: recon_snr unusable and no baseline proxy available")
            bep = baseline_es[ent]
            proxy.append(ent)
        es[ent] = bep
    es["SWaT_ranked"] = es["SWaT_full"]  # shared training
    metrics = {}
    for ent in ALL_ENTITIES:
        cell = os.path.join(run_dir, CELL_DIR[ent])
        rec = metrics_by_epoch(cell).get(es[ent])
        if rec is None:
            raise RuntimeError(f"{dir_name}/{ent}: epoch {es[ent]} not present in epoch_metrics.json")
        metrics[ent] = {k: rec[k] for k in METRIC_KEYS}
    return {"dir": dir_name, "es": es, "proxy": proxy, "metrics": metrics, "warmups": warmups}


# ----------------------------------------------------------------------------- collection
def collect():
    """Return runs[(cond_key, seed)] -> run record (with 'label'), plus partial-dir map."""
    runs = {}
    partials = {}
    # baselines first (proxy source)
    for seed in SEEDS:
        name, part = find_run(seed, None)
        if part:
            partials[("baseline", seed)] = part
        if name:
            rec = load_run(name)
            rec["label"] = str(seed)
            runs[("baseline", seed)] = rec
    for cond, (tag, _, _) in CONDITIONS.items():
        if tag is None:
            continue
        for seed in SEEDS:
            use_tag, label = tag, str(seed)
            if cond == "blind" and seed == BLIND_STANDIN_SEED:
                use_tag, label = BLIND_STANDIN_TAG, "42*"
            name, part = find_run(seed, use_tag)
            if part:
                partials[(cond, seed)] = part
            if name is None:
                continue
            base = runs.get(("baseline", seed))
            rec = load_run(name, baseline_es=base["es"] if base else None)
            rec["label"] = label
            runs[(cond, seed)] = rec
    return runs, partials


# ----------------------------------------------------------------------------- anchors
def check_anchors(runs):
    """Hard verification anchors — raise AssertionError on any failure (before writing output)."""
    tol = 1e-6
    b40 = runs[("baseline", 40)]
    assert b40["es"]["PSM"] == 16, f"anchor: seed40 baseline PSM ES epoch {b40['es']['PSM']} != 16"
    expected = {
        "pak_auc_f1": 0.83219054,
        "vus_pr": 0.80112969,
        "vus_roc": 0.88039909,
        "affiliation_f1_ar": 0.80827500,
        "prc_auc": 0.78601621,
    }
    for k, v in expected.items():
        got = b40["metrics"]["PSM"][k]
        assert abs(got - v) <= tol, f"anchor: seed40 baseline PSM {k} {got:.8f} != {v:.8f}"
    b42 = runs[("baseline", 42)]
    for ent, ep in [("SWaT_full", 16), ("SWaT_ranked", 16), ("PSM", 16), ("WaDi_A2", 16), ("WaDi_A1", 30)]:
        assert b42["es"][ent] == ep, f"anchor: seed42 baseline {ent} ES epoch {b42['es'][ent]} != {ep}"
    pak = b42["metrics"]["SWaT_ranked"]["pak_auc_f1"]
    assert 0.55 <= pak <= 0.65, f"anchor: seed42 baseline SWaT excl22 pak_auc_f1 {pak:.6f} outside [0.55, 0.65]"


# ----------------------------------------------------------------------------- formatting
def f4(v):
    return "—" if v is None else f"{v:.4f}"


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = sum(vals) / n
    if n == 1:
        return m, None, 1
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))  # sample std (ddof=1)
    return m, s, n


def ms_cell(vals):
    m, s, n = mean_std(vals)
    if n == 0:
        return "—"
    if s is None:
        return f"{m:.4f} ± —"
    return f"{m:.4f} ± {s:.4f}"


def seed_label(cond, seed):
    return "42*" if (cond == "blind" and seed == BLIND_STANDIN_SEED) else str(seed)


def per_seed_table(runs, cond, entity, metric_pairs):
    """Rows = seeds + Mean±Std; cols = metrics, for one (condition, entity)."""
    lines = ["| Seed | " + " | ".join(n for n, _ in metric_pairs) + " |",
             "|---|" + "---|" * len(metric_pairs)]
    cols = {k: [] for _, k in metric_pairs}
    n_avail = 0
    for seed in SEEDS:
        rec = runs.get((cond, seed))
        vals = []
        for _, k in metric_pairs:
            v = rec["metrics"][entity][k] if rec else None
            vals.append(v)
            cols[k].append(v)
        if rec:
            n_avail += 1
        lines.append(f"| {seed_label(cond, seed)} | " + " | ".join(f4(v) for v in vals) + " |")
    mark = " ✓" if n_avail == 5 else ""
    lines.append(f"| **Mean ± Std** (n={n_avail}){mark} | "
                 + " | ".join(f"**{ms_cell(cols[k])}**" if n_avail else "—" for _, k in metric_pairs) + " |")
    return "\n".join(lines)


def entity_mean_table(runs, cond, entities, metric_pairs):
    """Rows = seeds + Mean±Std; cols = metrics; each cell = mean over `entities` (Table 4 style)."""
    lines = ["| Seed | " + " | ".join(f"{n} (4-entity mean)" for n, _ in metric_pairs) + " |",
             "|---|" + "---|" * len(metric_pairs)]
    cols = {k: [] for _, k in metric_pairs}
    n_avail = 0
    for seed in SEEDS:
        rec = runs.get((cond, seed))
        vals = []
        for _, k in metric_pairs:
            v = (sum(rec["metrics"][e][k] for e in entities) / len(entities)) if rec else None
            vals.append(v)
            cols[k].append(v)
        if rec:
            n_avail += 1
        lines.append(f"| {seed_label(cond, seed)} | " + " | ".join(f4(v) for v in vals) + " |")
    mark = " ✓" if n_avail == 5 else ""
    lines.append(f"| **Mean ± Std** (n={n_avail}){mark} | "
                 + " | ".join(f"**{ms_cell(cols[k])}**" if n_avail else "—" for _, k in metric_pairs) + " |")
    return "\n".join(lines)


def condition_block(runs, cond, entities, metric_pairs, heading_level="###"):
    _, paper_name, _ = CONDITIONS[cond]
    out = []
    for ent in entities:
        out.append(f"{heading_level} {paper_name} — {ent}\n")
        out.append(per_seed_table(runs, cond, ent, metric_pairs))
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------- md sections
def sec_header(runs, partials, pending):
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# LASAD Official Results (paper v22 tables) — per-seed + mean±std",
        "",
        f"*Generated: {now} — regenerate with:* `conda activate dc_vis && python scripts/generate_results_md.py`",
        "",
        "**Extraction convention** — all numbers are taken at the **recon_snr early-stop (ES) epoch**, "
        "*not* the oracle pak-best epoch:",
        "",
        "- Per cell: from `training_histories.json` `train_recon_snr` (1-indexed epochs), post-warmup "
        "(`teacher_only_warmup_epochs`, =15 for these runs, verified per cell) EMA stream "
        "`ema = 0.2*snr_e + 0.8*ema`; best-so-far EMA with patience 2; **ES epoch = best epoch** "
        "(not the stop epoch). Identical for full 30-epoch and ES-halted (shorter) runs.",
        "- **Proxy rule**: label-free runs (blind, positive-excised, and the blind@42 stand-in) have no "
        "labeled train anomalies, so `train_recon_snr` is undefined (all-None) — they use the **same "
        "seed's baseline ES epoch** for the same cell.",
        "- `SWaT_ranked` (= SWaT/A1A2_excl22, region-22 excluded) shares training with `SWaT_full` "
        "(same run): it uses the full cell's ES epoch and reads its own `epoch_metrics.json` at that epoch.",
        "- **blind @ seed42 stand-in (`42*`)**: the label-blind condition at seed 42 is represented by the "
        f"`{BLIND_STANDIN_TAG}` run (`train_label_mask_frac=1.0` — 100% of labeled anomalies treated as "
        "unlabeled), user-approved as functionally equivalent to `blind_train_labels=True`.",
        "- Mean ± Std = sample std (ddof=1) over **available seeds only**, annotated (n=X); ✓ marks n=5. "
        "Empty cells (pending runs) render as —. Tables show 4 decimal places.",
        "",
        "## Campaign status",
        "",
        "| Condition | Tag | Seeds complete | n/5 |",
        "|---|---|---|---|",
    ]
    for cond, (tag, paper_name, _) in CONDITIONS.items():
        done = [seed_label(cond, s) for s in SEEDS if (cond, s) in runs]
        tag_s = "(none)" if tag is None else f"`{tag}`"
        if cond == "blind":
            tag_s += f" (+`{BLIND_STANDIN_TAG}`@42)"
        lines.append(f"| {paper_name} | {tag_s} | {', '.join(done) if done else '—'} | {len(done)}/5 |")
    lines += ["", "**Pending runs, in queue order** (`sens3seed` → `paper5seed` → `p5sens`):", ""]
    if not pending:
        lines.append("- (none — campaign complete)")
    else:
        in_flight = [p for p in pending if p[3]]
        rest = [p for p in pending if not p[3]]
        i = 1
        for tag, seed, q, _ in in_flight:
            lines.append(f"{i}. `{tag}` @ seed {seed} ({q}) — **in-flight**")
            i += 1
        for tag, seed, q, _ in rest:
            lines.append(f"{i}. `{tag}` @ seed {seed} ({q})")
            i += 1
    lines.append("")
    return "\n".join(lines)


def sec_config(runs):
    b42 = runs[("baseline", 42)]
    cfg = load_json(OFFICIAL, b42["dir"], "PSM", "best_config.json")
    summary = [
        ("Window / patch", f"seq_length={cfg['seq_length']}, patch_size={cfg['patch_size']}, "
                           f"num_patches={cfg['num_patches']}, patchify={cfg['patchify_mode']}"),
        ("Encoder", f"d_model={cfg['d_model']}, nhead={cfg['nhead']}, layers={cfg['num_encoder_layers']}, "
                    f"dim_ff={cfg['dim_feedforward']}, dropout={cfg['dropout']}"),
        ("Decoders", f"teacher={cfg['num_teacher_decoder_layers']} layers, "
                     f"student={cfg['num_student_decoder_layers']} layers, "
                     f"shared={cfg['num_shared_decoder_layers']}"),
        ("Masking", f"masking_ratio (rho)={cfg['masking_ratio']}, strategy=patch, "
                    f"mask_after_encoder={cfg['mask_after_encoder']}, "
                    f"force_mask_anomaly={cfg['force_mask_anomaly']}"),
        ("Training", f"num_epochs={cfg['num_epochs']}, batch_size={cfg['batch_size']}, "
                     f"lr={cfg['learning_rate']}, weight_decay={cfg['weight_decay']}, "
                     f"lr warmup_epochs={cfg['warmup_epochs']}, "
                     f"teacher_only_warmup_epochs={cfg['teacher_only_warmup_epochs']}, "
                     f"AMP={cfg['use_amp']} ({cfg['amp_dtype']})"),
        ("Loss", f"margin_type={cfg['margin_type']} (k={cfg['dynamic_margin_k']}, margin={cfg['margin']}), "
                 f"lambda_disc={cfg['lambda_disc']}, anomaly_loss_weight={cfg['anomaly_loss_weight']} "
                 f"({cfg['anomaly_loss_direction']}), normal_loss_weight={cfg['normal_loss_weight']}, "
                 f"patch_level_loss={cfg['patch_level_loss']}, "
                 f"balance={cfg['loss_balance_mode']}"),
        ("GRL", f"use_grl={cfg['use_grl']}, mode={cfg['grl_mode']}, weight={cfg['grl_loss_weight']}, "
                f"focal={cfg['grl_use_focal']}, adaptive_lambda={cfg['grl_adaptive_lambda']}, "
                f"target={cfg['grl_target_mode']}, pos_weight=auto (labeled-anomaly ratio; "
                f"{cfg['grl_pos_weight']:.4f} on PSM@42)"),
        ("Feature matching", f"use_feature_matching={cfg['use_feature_matching']}, "
                             f"weight={cfg['fm_loss_weight']}, metric={cfg['fm_distance_metric']}, "
                             f"adaptive_lambda={cfg['fm_adaptive_lambda']}, "
                             f"output_discrepancy={cfg['use_output_discrepancy']}"),
        ("Scoring", f"anomaly_score_mode={cfg['anomaly_score_mode']}, "
                    f"score_recon_disc_ratio={cfg['score_recon_disc_ratio']} "
                    f"(= disc weight w={1.0/cfg['score_recon_disc_ratio']:.2f}), "
                    f"paper metrics computed with sliding window 100"),
        ("Normalization", f"{cfg['normalize_mode']} {cfg['minmax_range']} "
                          f"(clamp [{cfg['minmax_clamp_min']}, {cfg['minmax_clamp_max']}])"),
        ("Data", f"sliding-window dataset, train stride={cfg['sliding_window_stride']}, "
                 f"test stride=1, train_ratio={cfg['sliding_window_train_ratio']:.4f}"),
        ("Seed", "random_seed = 40/41/42/43/44 (+ PYTHONHASHSEED=seed)"),
    ]
    lines = [
        "## 1. Configuration",
        "",
        f"Baseline reference: `{b42['dir']}` (seed 42, PSM cell `best_config.json`; per-cell configs are "
        "identical up to dataset-derived fields such as `num_features` and auto `grl_pos_weight`).",
        "",
        "| Group | Values |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in summary]
    lines += [
        "",
        "<details><summary>Full best_config.json (baseline seed 42, PSM)</summary>",
        "",
        "```json",
        json.dumps(cfg, indent=1),
        "```",
        "",
        "</details>",
        "",
        "### Per-condition overrides (vs baseline)",
        "",
        "Launcher base: `official=True num_epochs=30 official_keep_checkpoints=False` "
        "(+ `use_reconsnr_es_halt=True` for the newer labeled-condition runs — the ES halt only shortens "
        "training past the ES epoch and does not change the reported ES-epoch numbers; it is OFF for "
        "label-free conditions, which use the proxy rule instead).",
        "",
        "| Condition | Dir tag | Override |",
        "|---|---|---|",
    ]
    for cond, (tag, paper_name, override) in CONDITIONS.items():
        lines.append(f"| {paper_name} | `{tag if tag else '(none)'}` | {override} |")
    lines += [
        f"| Label-blind @ seed42 stand-in (`42*`) | `{BLIND_STANDIN_TAG}` | "
        "`train_label_mask_frac=1.0` (≡ label-blind; user-approved equivalence) |",
        "",
        "`grl_pos_weight` is auto-derived from the labeled-anomaly ratio, so it shifts with the "
        "label-sparsity conditions and saturates (999.0) for label-free runs.",
        "",
    ]
    return "\n".join(lines)


def sec_table2(runs):
    out = [
        "## 2. Table 2 — main comparison (LASAD / positive-excised / label-blind)",
        "",
        "Entities: SWaT_ranked (= A1A2_excl22), WaDi_A1, WaDi_A2, PSM. "
        "Metrics: PAK (`pak_auc_f1`), VUS-PR (`vus_pr`), Aff-F1 (`affiliation_f1_ar`).",
        "",
    ]
    for cond in ["baseline", "exclanom", "blind"]:
        out.append(condition_block(runs, cond, PAPER_ENTITIES, METRICS_MAIN))
    return "\n".join(out)


def sec_ablation(runs):
    out = [
        "## 3. Table 4 + B.1 — ablation",
        "",
        "Per-entity tables (Table B.1) + 4-entity mean over {SWaT_ranked, WaDi_A1, WaDi_A2, PSM} (Table 4).",
        "",
    ]
    for cond in ABLATION_CONDS:
        _, paper_name, _ = CONDITIONS[cond]
        out.append(condition_block(runs, cond, PAPER_ENTITIES, METRICS_MAIN))
        out.append(f"### {paper_name} — 4-entity mean (Table 4 row)\n")
        out.append(entity_mean_table(runs, cond, PAPER_ENTITIES, METRICS_MAIN))
        out.append("")
    return "\n".join(out)


def sec_a6(runs):
    out = [
        "## 4. Table A.6 — supplementary metrics (LASAD ours only)",
        "",
        "Metrics: PRC-AUC (`prc_auc`), VUS-ROC (`vus_roc`), PA%K=0 F1 (`pa_0_f1`).",
        "",
    ]
    out.append(condition_block(runs, "baseline", B2_ENTITIES, METRICS_SUPP))
    return "\n".join(out)


def sec_sparsity(runs):
    out = [
        "## 5. Table B.2 + Fig 6 — label sparsity",
        "",
        "Unlabeled-anomaly fraction 0% (= baseline) → 100% (= label-blind); GROUP-random (100-ts bins) "
        "label masking (`unlab*r` runs). Entities include SWaT_full as in Table B.2.",
        "",
    ]
    for label, cond in SPARSITY_LEVELS:
        out.append(f"### Sparsity {label}\n")
        for ent in B2_ENTITIES:
            out.append(f"#### Sparsity {label} — {ent}\n")
            out.append(per_seed_table(runs, cond, ent, METRICS_MAIN))
            out.append("")
    return "\n".join(out)


def sec_rho(runs):
    out = [
        "## 6. Fig B.1(b) — masking-ratio (rho) sensitivity",
        "",
        "rho ∈ {0.05, 0.10, 0.15 (= baseline), 0.30, 0.50, 0.60, 0.75, 0.90}. The paper plots VUS-PR; "
        "all three paper metrics are tabulated.",
        "",
    ]
    for label, cond in RHO_LEVELS:
        out.append(f"### rho = {label}\n")
        for ent in PAPER_ENTITIES:
            out.append(f"#### rho = {label} — {ent}\n")
            out.append(per_seed_table(runs, cond, ent, METRICS_MAIN))
            out.append("")
    return "\n".join(out)


def sec_w(runs):
    out = [
        "## 7. Fig B.1(a) — score-weight (w) sensitivity [skeleton]",
        "",
        "**Not filled by this generator.** w does not require retraining: each baseline seed's "
        "`epoch_scores/epoch_NNN_scores.npz` stores per-point recon/disc scores, so the w-sweep is "
        "post-hoc recomputable at the ES epoch (recon + w·disc, sliding window 100) — to be filled by a "
        "separate pass. Baseline w = 1/score_recon_disc_ratio = 0.25.",
        "",
        "| w | " + " | ".join(PAPER_ENTITIES) + " |",
    ]
    out.append("|---|" + "---|" * len(PAPER_ENTITIES))
    for w in W_VALUES:
        out.append(f"| {w} | " + " | ".join("—" for _ in PAPER_ENTITIES) + " |")
    out.append("")
    return "\n".join(out)


def sec_provenance(runs):
    out = [
        "## 8. Provenance — run dirs and selected epochs",
        "",
        "ES epochs per training cell (SWaT_ranked shares SWaT_full's epoch). "
        "`proxy` = epoch taken from the same seed's baseline (label-free run).",
        "",
        "| Condition | Seed | Run dir | ES epochs (SWaT_full=ranked / WaDi_A1 / WaDi_A2 / PSM) | Proxy |",
        "|---|---|---|---|---|",
    ]
    for cond, (tag, paper_name, _) in CONDITIONS.items():
        for seed in SEEDS:
            rec = runs.get((cond, seed))
            if rec is None:
                continue
            es = rec["es"]
            es_s = f"{es['SWaT_full']} / {es['WaDi_A1']} / {es['WaDi_A2']} / {es['PSM']}"
            proxy = ", ".join(rec["proxy"]) if rec["proxy"] else "—"
            if rec["proxy"]:
                proxy += f" (from baseline@{seed})"
            out.append(f"| {paper_name} | {rec['label']} | `{rec['dir']}` | {es_s} | {proxy} |")
    out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------------------- main
def main():
    runs, partials = collect()
    check_anchors(runs)

    complete_pairs = set(runs.keys())
    pending = []
    seen = set()
    for tag, seed, q in build_queue():
        cond = tag  # queue tags == condition keys
        if (cond, seed) in complete_pairs or (cond, seed) in seen:
            continue
        seen.add((cond, seed))
        in_flight = (cond, seed) in partials
        pending.append((tag, seed, q, in_flight))

    sections = [
        sec_header(runs, partials, pending),
        sec_config(runs),
        sec_table2(runs),
        sec_ablation(runs),
        sec_a6(runs),
        sec_sparsity(runs),
        sec_rho(runs),
        sec_w(runs),
        sec_provenance(runs),
    ]
    md = "\n".join(sections).rstrip() + "\n"
    with open(OUT_PATH, "w") as fh:
        fh.write(md)

    print(f"anchors OK; {len(runs)} (condition, seed) runs complete; {len(pending)} pending; wrote {OUT_PATH}")
    for tag, seed, q, fl in pending:
        print(f"  pending: {q}: {tag}@{seed}" + (" [in-flight]" if fl else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
