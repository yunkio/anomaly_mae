#!/usr/bin/env python3
"""Backfill per-epoch ``train_loss`` into historical baseline ``epoch_metrics.json``.

The training code (``baseline_common._attach_train_loss``, added 2026-06-22) only
records ``train_loss`` for runs executed AFTER that change. Older experiments
(8/10/...) computed the same per-epoch mean loss but only emitted it to *stdout*.
This script recovers that value from the captured queue logs and injects it into
each epoch entry of the existing ``epoch_metrics.json`` — in the SAME form the
live code now produces (``train_loss`` key per epoch; ``null`` when unavailable).

Attribution is exact, not heuristic:
  * Only logs that actually wrote to a given experiment's output dir are parsed.
  * Each job's ``[tag] Experiment: <DATASET>/<VARIANT>`` line maps tag -> result
    path; the tag prefix maps -> model. So (model, experiment_path) is the key,
    matched 1:1 against the on-disk result dir. No token guessing.
  * Logs are merged oldest->newest (file mtime); the newest log that ran a given
    (model, experiment_path) wins — matching "the result dir = last run".
  * Loss value == what ``_attach_train_loss`` would have stored: dagmm = L1+L2,
    npsr = M_pt+M_seq, others = printed epoch-mean (6-decimal precision from log).

Safety: writes ``epoch_metrics.json.bak`` (once) before the first edit of a file,
atomic replace, idempotent (re-runs only refresh the train_loss key). Non-DL
baselines get ``train_loss: null`` so the schema is uniform across every file.
"""
from __future__ import annotations
import argparse, json, os, re, sys, glob, math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESBASE = os.path.join(HERE, "results", "experiments")

DL_MODELS = ['anomaly_transformer', 'catch', 'dagmm', 'dcdetector', 'gcn_lstm',
             'gdn', 'memto', 'mlp', 'mlpmixer', 'moderntcn', 'npsr', 'omnianomaly',
             'tfmae', 'timesnet', 'tranad', 'transformer', 'usad']
NONDL = {'random', 'l2_norm', 'nn_distance', 'pca_error', 'sensor_range'}
_MODELS_BY_LEN = sorted(DL_MODELS, key=len, reverse=True)   # mlpmixer before mlp

# experiment output dir -> queue logs that wrote into it (any order; merged by mtime)
LOG_DIR = os.path.join(HERE, "..", "temp", "baseline_experiment_run")
LOG_MAP = {
    "8_20260606_175756_baseline": [
        "chain_8_9_modelmajor.log",
        "chain_8_9_resumed.log",
        "watcher_10_11.log",
        "chain_10_11_12_resumed_0613.log",
    ],
    "10_20260615_230000_baseline_contaminated": [
        "chain_contaminated_20260615.log",
    ],
}

_F = r"(-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf|-inf))"
TAGLINE = re.compile(rb"^\s*\[([A-Za-z][A-Za-z0-9_.\-]*)\]\s*(.*)$")
EXPLINE = re.compile(r"Experiment:\s*(\S+)")

# (epoch, loss) extractors — specific formats first, generic last.
PAT_DAGMM = re.compile(rf"Epoch\s+(\d+)/\d+:\s*L1={_F},\s*L2={_F}")            # dagmm: L1+L2
PAT_NPSR  = re.compile(rf"Epoch\s+(\d+):\s*M_pt={_F}\s+M_seq={_F}")            # npsr: pt+seq
PAT_RECL  = re.compile(rf"Epoch\s+(\d+):\s*rec_loss\s*=\s*{_F}")              # anomaly_transformer
PAT_EPLOSS= re.compile(rf"Epoch\s+(\d+):\s*loss\s*=\s*{_F}")                  # catch/dcdetector/tfmae/timesnet/moderntcn/memto
PAT_WETAS = re.compile(rf"Epoch\s*\[(\d+)/\d+\]\s*loss=\s*{_F}")              # wetas
PAT_TRAIN = re.compile(rf"Training:\s*\[(\d+)/\d+\].*?Loss:\s*{_F}")          # gcn_lstm/gdn/omnianomaly/tranad/usad
PAT_GEN   = re.compile(rf"(?<!model=)\[(\d+)/\d+\]\s*Loss:\s*{_F}")           # generic loop: mlp/mlpmixer/transformer


def _flt(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def match_loss(rest):
    """Return (epoch:int, loss:float|None) for a per-epoch summary line, else (None,None)."""
    m = PAT_DAGMM.search(rest)
    if m:
        a, b = _flt(m.group(2)), _flt(m.group(3))
        return int(m.group(1)), (None if a is None or b is None else a + b)
    m = PAT_NPSR.search(rest)
    if m:
        a, b = _flt(m.group(2)), _flt(m.group(3))
        return int(m.group(1)), (None if a is None or b is None else a + b)
    for p in (PAT_RECL, PAT_EPLOSS, PAT_WETAS, PAT_TRAIN, PAT_GEN):
        m = p.search(rest)
        if m:
            return int(m.group(1)), _flt(m.group(2))
    return None, None


def model_of(tag):
    for mdl in _MODELS_BY_LEN:
        if tag == mdl or tag.startswith(mdl + "_"):
            return mdl
    return None


def parse_log(path):
    """One log -> {(model, experiment_path): {epoch: loss}}."""
    tag_exp = {}
    tag_loss = defaultdict(dict)
    with open(path, "rb") as fh:
        data = fh.read()
    for seg in re.split(rb"[\r\n]", data):
        m = TAGLINE.match(seg)
        if not m:
            continue
        tag = m.group(1).decode("utf8", "ignore")
        rest = m.group(2).decode("utf8", "ignore")
        em = EXPLINE.search(rest)
        if em:
            tag_exp[tag] = em.group(1)
            continue
        if "[BATCH]" in rest:
            continue
        ep, loss = match_loss(rest)
        if ep is not None:
            tag_loss[tag][ep] = loss
    out = {}
    for tag, eps in tag_loss.items():
        mdl = model_of(tag)
        exp = tag_exp.get(tag)
        if mdl and exp:
            out[(mdl, exp)] = eps
    return out


def build_index(expdir):
    """Merge that experiment's logs oldest->newest (latest run wins per key)."""
    logs = []
    for name in LOG_MAP[expdir]:
        p = os.path.join(LOG_DIR, name)
        if os.path.exists(p):
            logs.append(p)
        else:
            print(f"  [WARN] missing log: {name}")
    logs.sort(key=os.path.getmtime)          # chronological
    index = {}
    for p in logs:
        for k, v in parse_log(p).items():
            index[k] = v                     # wholesale last-wins per (model, exp_path)
    return index


def result_files(expdir):
    """Yield (path, model, exp_path) for every epoch_metrics.json in the exp dir."""
    base = os.path.join(RESBASE, expdir)
    for path in glob.glob(f"{base}/**/epoch_metrics.json", recursive=True):
        rel = os.path.relpath(os.path.dirname(path), base)   # DATASET[/VARIANT]/model
        parts = rel.split(os.sep)
        model = parts[-1]
        exp_path = "/".join(parts[:-1])
        yield path, model, exp_path


def lookup(index, model, exp_path):
    d = index.get((model, exp_path))
    if d is None and exp_path.endswith("_excl22"):
        # excl22 is a derived eval of the *_full training run — same train_loss.
        d = index.get((model, exp_path[:-len("_excl22")] + "_full"))
    return d


def inject(path, epoch_loss):
    """Set train_loss on every epoch entry (after 'epoch'); return (n_set, n_epochs)."""
    with open(path, "rb") as f:
        original = f.read()
    doc = json.loads(original)
    eps = doc.get("epochs", [])
    n_set = 0
    new_eps = []
    for e in eps:
        val = epoch_loss.get(e.get("epoch")) if epoch_loss else None
        if val is not None:
            n_set += 1
        # rebuild preserving key order, train_loss right after 'epoch'
        ne = {}
        for k, v in e.items():
            if k == "train_loss":
                continue
            ne[k] = v
            if k == "epoch":
                ne["train_loss"] = val
        if "epoch" not in e:
            ne["train_loss"] = val
        new_eps.append(ne)
    doc["epochs"] = new_eps
    # No per-file .bak: a full pre-edit snapshot of every epoch_metrics.json is
    # taken to comparison/.trash/<date>/ before --apply (keeps result dirs clean,
    # avoids colliding with add_teacher_only.py's .bak semantics). Atomic replace.
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)
    return n_set, len(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run report)")
    ap.add_argument("--experiments", nargs="*", default=list(LOG_MAP.keys()))
    args = ap.parse_args()

    for expdir in args.experiments:
        if expdir not in LOG_MAP:
            print(f"[SKIP] no log mapping for {expdir}")
            continue
        print(f"\n{'='*70}\nEXP {expdir}\n{'='*70}")
        index = build_index(expdir)
        print(f"  log index: {len(index)} (model, experiment_path) runs with loss")

        stat = defaultdict(lambda: {"files": 0, "full": 0, "partial": 0, "none": 0,
                                     "nondl": 0, "corrupt": 0})
        gaps = []
        for path, model, exp_path in result_files(expdir):
            fam = exp_path.split("/")[0]
            s = stat[model]
            s["files"] += 1
            try:
                with open(path) as f:
                    doc = json.load(f)
                n_ep = len(doc.get("epochs", []))
            except Exception:
                s["corrupt"] += 1
                continue
            if model in NONDL:
                s["nondl"] += 1
                if args.apply:
                    inject(path, None)
                continue
            d = lookup(index, model, exp_path)
            n_have = sum(1 for e in doc.get("epochs", []) if d and d.get(e.get("epoch")) is not None)
            if n_ep and n_have == n_ep:
                s["full"] += 1
            elif n_have > 0:
                s["partial"] += 1
                gaps.append(f"PARTIAL {exp_path}/{model}: {n_have}/{n_ep}")
            else:
                s["none"] += 1
                gaps.append(f"NONE    {exp_path}/{model}: 0/{n_ep}")
            if args.apply:
                inject(path, d)

        # summary
        tot = {k: sum(s[k] for s in stat.values()) for k in
               ("files", "full", "partial", "none", "nondl", "corrupt")}
        print(f"  result files: {tot['files']}  (DL full={tot['full']} "
              f"partial={tot['partial']} none={tot['none']} | nonDL={tot['nondl']} "
              f"corrupt={tot['corrupt']})")
        print(f"  {'model':<20}{'files':>6}{'full':>6}{'partial':>8}{'none':>6}")
        for mdl in sorted(stat):
            s = stat[mdl]
            tag = "" if mdl in NONDL else (" <<" if (s["partial"] or s["none"]) else "")
            print(f"  {mdl:<20}{s['files']:>6}{s['full']:>6}{s['partial']:>8}{s['none']:>6}{tag}")
        if gaps:
            print(f"  --- {len(gaps)} DL runs with missing/partial loss ---")
            for g in gaps[:60]:
                print("   ", g)
            if len(gaps) > 60:
                print(f"    ... +{len(gaps)-60} more")
        print(f"  [{'APPLIED' if args.apply else 'DRY-RUN'}]")


if __name__ == "__main__":
    main()
