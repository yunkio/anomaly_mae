#!/usr/bin/env python3
"""Comprehensive Phase 1 status report — emitted by Monitor every 15 min."""
import os, re, subprocess, time, json, sys
from datetime import datetime, timedelta
from pathlib import Path

LOG  = Path(open('/tmp/exp271_train_log.txt').read().strip())
PID_FILE = Path('/tmp/exp271_train_pid.txt')
WRAPPER_PID = int(open(PID_FILE).read().strip()) if PID_FILE.exists() else 0
EXPDIR_ROOT = Path('/home/ykio/notebooks/TSMAE/results/experiments')

# Find the actual training worker (run_queue.py wrapper spawns run_base_experiments.py as child)
def find_training_pid(wrapper_pid):
    """Return (wrapper_pid, worker_pid).

    Two cases:
    1. run_queue.py launcher → bash wrapper → python child (worker). Returns (wrapper, child).
    2. setsid nohup bash -c "exec python ..." → bash exec'd to python, single PID.
       Returns (wrapper, wrapper) so heartbeat doesn't read worker as 0.
    """
    if wrapper_pid <= 0: return (0, 0)
    try:
        os.kill(wrapper_pid, 0)
    except: return (wrapper_pid, 0)

    # Check the wrapper's own cmdline — if it's the training script itself, no child needed.
    try:
        wrapper_cmd = subprocess.check_output(['ps', '-p', str(wrapper_pid), '-o', 'cmd='], text=True).strip()
    except Exception:
        wrapper_cmd = ''
    if 'run_base_experiments.py' in wrapper_cmd or 'run_ablation.py' in wrapper_cmd:
        return (wrapper_pid, wrapper_pid)

    # Otherwise look for a child running the training script.
    try:
        children = subprocess.check_output(['pgrep', '-P', str(wrapper_pid)], text=True).strip().split()
        for cpid in children:
            cmd = subprocess.check_output(['ps', '-p', cpid, '-o', 'cmd=']).decode()
            if 'run_base_experiments.py' in cmd or 'run_ablation.py' in cmd:
                return (wrapper_pid, int(cpid))
        if children:
            return (wrapper_pid, int(children[0]))
    except Exception:
        pass
    return (wrapper_pid, 0)

PID, WORKER_PID = find_training_pid(WRAPPER_PID)

def alive(pid):
    if pid <= 0: return False
    try: os.kill(pid, 0); return True
    except: return False

def sh(cmd, timeout=8):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return ""

def fmt_dur(sec):
    if sec is None or sec < 0: return "?"
    sec = int(sec)
    h, m, s = sec//3600, (sec%3600)//60, sec%60
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"

def fmt_eta(sec):
    if sec is None or sec <= 0: return "?"
    eta_dt = datetime.now() + timedelta(seconds=sec)
    return f"{fmt_dur(sec)} (finish ~{eta_dt.strftime('%H:%M')})"

# ---------- ALWAYS-FULL-LOG markers (grep cheap, few hits) ----------
# These are emitted once per dataset / once per init — use grep for full log.
exp_dirs = sh(f"grep -aoE 'Experiment dir:[[:space:]]+\\S+' '{LOG}'").splitlines()
exp_dirs = [l.split(':',1)[1].strip() for l in exp_dirs if ':' in l]
amp_line = sh(f"grep -aE 'AMP: use_amp=' '{LOG}' | head -1")

# Parse teacher_only_warmup_epochs from log (config_override line at queue start).
# Falls back to 250 (exp271 canonical) if not found. Used to mask pre-warmup
# disc_snr / dis / recon_s display to "-" since they are not meaningful before
# student joining. 2026-05-29 update — replaces hardcoded WARMUP = 250.
_warmup_match = sh(f"grep -aoE 'teacher_only_warmup_epochs[= ]+[0-9]+' '{LOG}' | head -1")
try:
    WARMUP = int(re.search(r'(\d+)', _warmup_match).group(1)) if _warmup_match else 250
except (AttributeError, ValueError):
    WARMUP = 250

# unique dataset dirs in order; current = last
unique_ds = []
for d in exp_dirs:
    if d not in unique_ds: unique_ds.append(d)
ds_idx = max(1, len(unique_ds))  # 1-based; minimum 1 once training started
ds_total = 4  # queue says 4 datasets

current_path = unique_ds[-1] if unique_ds else None
completed_paths = unique_ds[:-1] if len(unique_ds) > 1 else []

# Parse ds name (last 2 parts: e.g. SWaT/A1A2_full)
def ds_short(p):
    parts = Path(p).parts
    return '/'.join(parts[-2:])

# ---------- TAIL-ONLY parse (~last 500 epoch lines, cheap) ----------
RE_EPOCH_DONE = re.compile(r'Epoch (\d+)/(\d+):\s*100%\|.*?\|\s*\d+/\d+\s*\[(\d{2}:\d{2})<00:00,\s*([\d.]+)it/s\]')
RE_EPOCH_INPROG = re.compile(r'Epoch (\d+)/(\d+):\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[(\d{2}:\d{2})<(\d{2}:\d{2}),\s*([\d.]+)it/s\]')
# Number atom: matches positive/negative decimals, scientific notation, AND nan/inf.
# Critical: without nan|inf, a single nan in any field drops the entire eval line
# from parsing → blanks in the monitoring table. 2026-05-29 fix.
NUM = r'(?:[-+]?(?:nan|inf|\d+\.?\d*(?:[eE][+-]?\d+)?))'

# New log format (family-ordered: point-strict | PA%K | event/range | VUS | diag).
# 2026-05-27 onward: prc f1 f1_t pak_f1 pak_prc aff_f1 rf1 vus_pr vus_roc d_snr.
# 2026-05-29 onward: also optional recon_t, recon_s, dis fields after s_loss
# (added for explicit student-vs-teacher recon distinction in monitoring; the
# legacy t_loss=train_rec_loss, s_loss=train_disc_loss kept for backward compat).
# All number fields use the NUM atom to handle nan/inf without dropping the line.
RE_EVAL_NEW = re.compile(
    r'\[Epoch\s*(\d+)\]\s+'
    rf'PRC=({NUM})\s+F1=({NUM})\s+F1_T=({NUM})\s+'
    rf'PAK_F1=({NUM})\s+PAK_PRC=({NUM})\s+'
    rf'AFF_F1=({NUM})\s+RF1=({NUM})\s+'
    rf'VUS_PR=({NUM})\s+VUS_ROC=({NUM})\s+'
    rf'd_SNR=({NUM})(?:\s+recon_SNR=({NUM}))?\s+\|\s+'
    rf't_loss=({NUM})\s+s_loss=({NUM})'
    rf'(?:\s+recon_t=({NUM}))?(?:\s+recon_s=({NUM}))?(?:\s+dis=({NUM}))?'
    r'(?:\s+d_loss=' + NUM + r')?'
    r'\s+\(infer=(\d+)s\s+eval=(\d+)s\)(?:\s+\[async\])?(\s+★)?'
)
# Old format (exp271 SWaT and earlier) — fallback for backward compat
RE_EVAL_OLD = re.compile(
    rf'\[Epoch\s*(\d+)\]\s+PRC=({NUM})\s+PAK_F1=({NUM})\s+PAK_PRC=({NUM})\s+'
    rf'F1_T=({NUM})\s+d_SNR=({NUM})\s+\|\s+t_loss=({NUM})\s+s_loss=({NUM})'
    r'\s+\(infer=(\d+)s\s+eval=(\d+)s\)(?:\s+\[async\])?(\s+★)?'
)
# Catch-all for diagnostic: any line that looks like an eval line but didn't
# match either NEW or OLD. Used to emit a stderr warning so we know WHY
# a metric became blank (regex drift vs genuine missing data).
RE_EVAL_LOOSE = re.compile(r'\[Epoch\s*\d+\]\s+PRC=')
RE_CRIT = re.compile(r'\[CRITICAL\]\s+epoch\s+(\d+):\s+(\d+)\s+batch')
# Dataset transition marker (e.g., "# [2/4] WaDi_A1") — resets eval history
# so best-eval picker stays within the current dataset.
RE_DATASET_MARK = re.compile(r'^#\s*\[(\d+)/(\d+)\]\s+(\S+)')
# SWaT excl22 dual-eval line (2026-05-27): printed right after each [Epoch N] line.
# Format: "              [excl22] PRC=... F1=... F1_T=... PAK_F1=... PAK_PRC=... AFF_F1=... RF1=..."
RE_EVAL_EXCL22 = re.compile(
    r'\[excl22\]\s+'
    r'PRC=([\d.]+)\s+F1=([\d.]+)\s+F1_T=([\d.]+)\s+'
    r'PAK_F1=([\d.]+)\s+PAK_PRC=([\d.]+)\s+'
    r'AFF_F1=([\d.]+)\s+RF1=([\d.]+)'
)

# Read whole log if <=50MB (typical 4-dataset run ~5-10MB),
# otherwise last 20MB. Tail-only window caused best-eval mis-identification
# when sparse eval lines (~1 per 5 epochs) got pushed beyond window by
# dense tqdm progress bars (2026-05-27 fix).
with open(LOG, 'rb') as f:
    f.seek(0, 2); end = f.tell()
    if end <= 50 * 1024 * 1024:
        f.seek(0)
    else:
        f.seek(end - 20 * 1024 * 1024)
    tail_lines = f.read().decode('utf-8', errors='replace').splitlines()

def _f(s):
    """Float-coerce a string captured by NUM, including nan/inf. None-safe."""
    if s is None:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return float('nan')

done_epochs, inprog, evals, crit = [], None, [], []
parse_warnings = []  # eval-line strings we saw but couldn't parse → stderr later
for line in tail_lines:
    m = RE_DATASET_MARK.search(line)
    if m:
        # New dataset begins — clear per-dataset state so best-eval, done_epochs,
        # and in-progress reflect only the CURRENT dataset.
        done_epochs = []
        evals = []
        inprog = None
        continue
    m = RE_EPOCH_DONE.search(line)
    if m:
        ep, tot = int(m.group(1)), int(m.group(2))
        mm, ss = m.group(3).split(':')
        done_epochs.append((ep, tot, int(mm)*60+int(ss)))
        continue
    m = RE_EPOCH_INPROG.search(line)
    if m:
        ep, tot = int(m.group(1)), int(m.group(2))
        # only treat as in-progress if pct < 100 (the 100% case is caught by EPOCH_DONE)
        if int(m.group(3)) < 100:
            mr, sr = m.group(7).split(':')
            inprog = dict(ep=ep, tot=tot, pct=int(m.group(3)), k=int(m.group(4)), kt=int(m.group(5)),
                          remain=int(mr)*60+int(sr), its=float(m.group(8)))
        continue
    m = RE_EVAL_NEW.search(line)
    if m:
        evals.append(dict(
            epoch=int(m.group(1)),
            # point-strict
            prc=_f(m.group(2)), f1=_f(m.group(3)), f1_t=_f(m.group(4)),
            # PA%K
            pak_f1=_f(m.group(5)), pak_prc=_f(m.group(6)),
            # event/range
            aff_f1=_f(m.group(7)), r_f1=_f(m.group(8)),
            # VUS
            vus_pr=_f(m.group(9)), vus_roc=_f(m.group(10)),
            # diagnostic (d_snr always present; recon_snr 2026-05-29 onward optional)
            d_snr=_f(m.group(11)),
            recon_snr=_f(m.group(12)),
            t_loss=_f(m.group(13)), s_loss=_f(m.group(14)),
            # NEW (2026-05-29): optional explicit recon/discrepancy split
            recon_t=_f(m.group(15)), recon_s=_f(m.group(16)), dis=_f(m.group(17)),
            infer=int(m.group(18)), eval=int(m.group(19)), best=bool(m.group(20))
        ))
        continue
    m = RE_EVAL_OLD.search(line)
    if m:
        evals.append(dict(
            epoch=int(m.group(1)), prc=_f(m.group(2)), pak_f1=_f(m.group(3)),
            pak_prc=_f(m.group(4)), f1_t=_f(m.group(5)),
            f1=None, vus_pr=None, vus_roc=None, aff_f1=None, r_f1=None,  # old-format
            d_snr=_f(m.group(6)), recon_snr=None,
            t_loss=_f(m.group(7)), s_loss=_f(m.group(8)),
            recon_t=None, recon_s=None, dis=None,  # not in old format
            infer=int(m.group(9)), eval=int(m.group(10)), best=bool(m.group(11))
        ))
        continue
    # Diagnostic: line looks like an eval line but matched neither NEW nor OLD.
    # Likely cause: log format change (new field added without parser update) or
    # malformed value (e.g., 1e-308 truncated). Record for stderr report so the
    # blanks-cause is visible instead of silently dropped.
    if RE_EVAL_LOOSE.search(line):
        parse_warnings.append(line.strip()[:200])
        continue
    # SWaT excl22 dual-eval line — attach to the most recent eval entry
    m = RE_EVAL_EXCL22.search(line)
    if m and evals:
        evals[-1]['excl22'] = {
            'prc': float(m.group(1)), 'f1': float(m.group(2)), 'f1_t': float(m.group(3)),
            'pak_f1': float(m.group(4)), 'pak_prc': float(m.group(5)),
            'aff_f1': float(m.group(6)), 'r_f1': float(m.group(7)),
        }
        continue
    m = RE_CRIT.search(line)
    if m: crit.append((int(m.group(1)), int(m.group(2))))

last_ep = done_epochs[-1] if done_epochs else None
last_eval = evals[-1] if evals else None
best_eval = max(evals, key=lambda e: e['pak_f1']) if evals else None

# Speed analysis
recent = done_epochs[-20:] if done_epochs else []
mean_train_sec = sum(s for _,_,s in recent)/len(recent) if recent else 0.0
recent_5 = done_epochs[-5:] if done_epochs else []
mean_train_recent = sum(s for _,_,s in recent_5)/len(recent_5) if recent_5 else 0.0
mean_eval_sec = sum(e['infer'] + e['eval'] for e in evals[-5:])/min(5, len(evals)) if evals else 0
per_epoch_avg = mean_train_sec + mean_eval_sec/5

if last_ep:
    remain_ep = last_ep[1] - last_ep[0]
    ds_eta = remain_ep * per_epoch_avg
else:
    remain_ep, ds_eta = None, None

# ---------- HARDWARE ----------
gpu_q = sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,fan.speed,pstate,clocks.gr,clocks.mem --format=csv,noheader,nounits")
gpu = {}
if gpu_q:
    parts = [x.strip() for x in gpu_q.split(',')]
    keys = ['mem_u','mem_t','util','mem_util','pw','pl','temp','fan','ps','cgr','cmem']
    gpu = dict(zip(keys, parts))

# CPU load
loadavg = open('/proc/loadavg').read().split()[:3]
ncpu = os.cpu_count()

# process-level for ACTUAL training worker (not the run_queue wrapper which is idle)
# top -bn2 -d 0.2 → second sample has valid %CPU diff. ps -o rss for clean KB RSS.
proc_cpu = proc_mem = proc_rss_kb = "?"
measure_pid = WORKER_PID if WORKER_PID > 0 else PID
proc_stats = sh(f"top -bn2 -d 0.2 -p {measure_pid} 2>/dev/null | tail -1")
if proc_stats:
    cols = proc_stats.split()
    if len(cols) >= 10 and cols[0].isdigit():
        proc_cpu, proc_mem = cols[8], cols[9]
proc_rss_kb = sh(f"ps -p {measure_pid} -o rss= 2>/dev/null").strip()

# Memory
free_out = sh("free -m")
mem_used = mem_total = mem_avail = swap_used = "?"
for ln in free_out.splitlines():
    if ln.startswith('Mem:'):
        p = ln.split()
        mem_total, mem_used, mem_avail = int(p[1]), int(p[2]), int(p[6])
    if ln.startswith('Swap:'):
        p = ln.split()
        swap_used = int(p[2])

# Disk
df_parts = sh("df -BM /home/ykio/notebooks/TSMAE/results | tail -1").split()
res_size = sh(f"du -sh {EXPDIR_ROOT} 2>/dev/null").split('\t')[0] if EXPDIR_ROOT.exists() else "?"
exp_root = Path(current_path).parents[1] if current_path else None  # results/experiments/271_.../  (drop dataset/subdir)
exp_size = sh(f"du -sh {exp_root} 2>/dev/null").split('\t')[0] if exp_root else "?"

# ---------- COMPLETED DATASET METRICS ----------
def scan_completed(p):
    """Return small summary of completed dataset dir."""
    p = Path(p)
    out = {'path': p, 'best': None, 'final': None, 'epochs': None, 'files': []}
    # Look for metrics.json / best_metrics / training_log etc.
    for name in ['metrics.json', 'best_metrics.json', 'final_metrics.json', 'summary.json', 'training_metrics.json']:
        f = p / name
        if f.exists():
            try:
                d = json.load(open(f))
                out['files'].append(name)
                if isinstance(d, dict):
                    out[name] = d
            except: pass
    # Heuristic: look for a best-epoch row in any json
    return out

# ---------- FORMAT ----------
now = datetime.now()
runtime = sh(f"ps -p {PID} -o etime=").strip() if alive(PID) else "(process dead)"
out = []
out.append(f"\n========== STATUS [{now:%H:%M:%S}] ==========")
out.append(f"PIDs: wrapper={PID} worker={WORKER_PID} ({'alive' if alive(PID) and alive(WORKER_PID) else 'DEAD'}) | runtime: {runtime}")
if amp_line:
    m = re.search(r'AMP: use_amp=(\S+), dtype=(\S+), scaler=(\S+)', amp_line)
    if m: out.append(f"AMP: use_amp={m.group(1)} dtype={m.group(2)} scaler={m.group(3)}")
out.append(f"Log: {LOG}")
out.append(f"Exp dir: {Path(current_path).parts[-3] if current_path else '?'}")

# Progress
out.append(f"\n[PROGRESS]")
out.append(f"  Dataset    : {ds_idx}/{ds_total}  |  current: {ds_short(current_path) if current_path else '?'}")
if last_ep:
    out.append(f"  Epoch      : {last_ep[0]}/{last_ep[1]} done")
if inprog and (not last_ep or inprog['ep'] > last_ep[0]):
    out.append(f"  In-progress: ep {inprog['ep']} at {inprog['pct']}% ({inprog['k']}/{inprog['kt']}), {inprog['its']:.2f}it/s, eta {inprog['remain']}s")
if mean_train_sec > 0:
    out.append(f"  Speed      : train ~{mean_train_sec:.2f}s/ep (last20) / ~{mean_train_recent:.2f}s/ep (last5)  |  eval ~{mean_eval_sec:.0f}s/eval  |  overall ~{per_epoch_avg:.2f}s/ep amortized")
if ds_eta and remain_ep is not None:
    out.append(f"  Dataset ETA: {fmt_eta(ds_eta)}  (epochs remaining: {remain_ep})")
# Rough overall ETA — assume remaining datasets average same wall-clock as current.
# Current dataset wall-clock so far ≈ (current_epoch * per_epoch_avg).
ds_remaining_count = max(0, ds_total - ds_idx)
if ds_eta is not None and ds_remaining_count > 0 and last_ep:
    avg_ds_wall = last_ep[1] * per_epoch_avg  # full-dataset wall-clock estimate
    overall_remain = ds_eta + ds_remaining_count * avg_ds_wall
    out.append(f"  Overall ETA: {fmt_eta(overall_remain)}  (datasets remaining after current: {ds_remaining_count}; assumes ~{fmt_dur(avg_ds_wall)}/dataset — may be high for smaller datasets)")
elif ds_remaining_count == 0 and ds_eta is not None:
    out.append(f"  Overall ETA: {fmt_eta(ds_eta)}  (last dataset)")

# Metrics
out.append(f"\n[METRICS — current dataset]")
def _fmt(v, fmt='.4f'):
    if v is None: return '   —  '
    try: return format(float(v), fmt)
    except: return '   —  '

def _emit_eval(label, ev):
    """Emit one eval row in the standard family-ordered format.
    Pre-warmup masks re_s/dis/dis_snr to None → '-' display."""
    _is_pre = ev['epoch'] <= WARMUP
    _d_snr  = None if _is_pre else ev.get('d_snr')
    _dis    = None if _is_pre else ev.get('dis')
    _re_s   = None if _is_pre else ev.get('recon_s')
    out.append(
        f"  {label} (ep {ev['epoch']:>3}):"
        f"  [point] f1={_fmt(ev.get('f1'))} prc={_fmt(ev['prc'])} f1_t={_fmt(ev['f1_t'])}"
        f"  [PA%K] pak_f1={_fmt(ev['pak_f1'])} pak_prc={_fmt(ev['pak_prc'])}"
        f"  [range] aff_f1={_fmt(ev.get('aff_f1'))} r_f1={_fmt(ev.get('r_f1'))}"
        f"  [diag] t_re={_fmt(ev.get('recon_t'))} re_snr={_fmt(ev.get('recon_snr'))}"
        f" s_re={_fmt(_re_s)} dis={_fmt(_dis)} dis_snr={_fmt(_d_snr)}"
    )
    if ev.get('excl22'):
        e = ev['excl22']
        out.append(
            f"               (excl22):"
            f"  [point] f1={_fmt(e['f1'])} prc={_fmt(e['prc'])} f1_t={_fmt(e['f1_t'])}"
            f"  [PA%K] pak_f1={_fmt(e['pak_f1'])} pak_prc={_fmt(e['pak_prc'])}"
            f"  [range] aff_f1={_fmt(e['aff_f1'])} r_f1={_fmt(e['r_f1'])}"
        )

# --- MILESTONE rows (2026-05-29) — emit all key milestone evals so the
# monitoring table accumulates instead of losing prior bests when a new
# best appears. Rows: first, best@pre-warmup, last-pre-warmup, best@post-
# warmup, best-overall, latest. Same-epoch labels are merged via "≡".
if evals:
    first_eval = evals[0]
    pre_evals = [e for e in evals if e['epoch'] <= WARMUP]
    post_evals = [e for e in evals if e['epoch'] > WARMUP]
    best_pre  = max(pre_evals, key=lambda e: e['pak_f1']) if pre_evals else None
    last_pre  = pre_evals[-1] if pre_evals else None
    best_post = max(post_evals, key=lambda e: e['pak_f1']) if post_evals else None
    best_all  = max(evals, key=lambda e: e['pak_f1'])
    latest_ev = evals[-1]

    # Build epoch → list-of-labels map (so same-epoch rows merge)
    label_map = {}  # epoch → list of (priority, label)
    def _add(ev, label, priority):
        if ev is None:
            return
        label_map.setdefault(ev['epoch'], []).append((priority, label))
    _add(first_eval, "first eval",                 0)
    _add(best_pre,   "best @ pre-warmup ☆",        1)
    _add(last_pre,   "last pre-warmup",            2)
    _add(best_post,  "best @ post-warmup ♦",       3)
    _add(best_all,   "best overall ★",             4)
    _add(latest_ev,  "latest",                     5)
    # 50-epoch interval markers (50, 100, 150, ..., 450, 500). 2026-05-29 update.
    # Forces visibility of every-50-ep checkpoint. Skipped when the epoch
    # already has a milestone label (first/best/latest/etc.) — user does not
    # want "≡ 50ep mark" merged onto existing milestone rows. The evals list
    # only contains logged evals (≤ current training epoch), so future
    # 50ep marks never appear.
    _milestone_epochs = set(label_map.keys())  # snapshot before 50ep adds
    for ev in evals:
        if ev['epoch'] % 50 == 0 and ev['epoch'] not in _milestone_epochs:
            _add(ev, "50ep mark",                  6)

    # Build set of emitted milestone epochs (for per-column best computation
    # downstream — we want the bold to be applied only across milestone rows,
    # not across all evals).
    emitted_epochs = sorted(label_map.keys())
    emitted_evs = [ev for ev in evals if ev['epoch'] in label_map]

    # Per-column best-of-milestones — emitted as a separate [BEST PER COLUMN]
    # block so the consumer (AI status report) bolds the correct cell even
    # when the displayed text values are tied (display .4f hides precision —
    # e.g. 0.000218 vs 0.000201 both print as 0.0002). 2026-05-29 update.
    # Direction rule (per user spec):
    #   t_re, s_re (loss-like) → min is best
    #   all other 11 columns → max is best
    # Output for each column: "<col>: ep N = <val_full_precision>"
    # AI uses the epoch reference to bold ONE specific row (with tiebreak: if
    # multiple epochs tie at exact float value, all of them bold).
    MAX_COLS = [
        ('f1',         lambda e: e.get('f1')),
        ('prc',        lambda e: e.get('prc')),
        ('f1_t',       lambda e: e.get('f1_t')),
        ('pak_f1',     lambda e: e.get('pak_f1')),
        ('pak_prc',    lambda e: e.get('pak_prc')),
        ('aff_f1',     lambda e: e.get('aff_f1')),
        ('r_f1',       lambda e: e.get('r_f1')),
        ('re_snr',     lambda e: e.get('recon_snr')),
        ('dis',        lambda e: e.get('dis')),
        ('dis_snr',    lambda e: e.get('d_snr')),
    ]
    MIN_COLS = [
        ('t_re',       lambda e: e.get('recon_t')),
        ('s_re',       lambda e: e.get('recon_s')),
    ]
    # For pre-warmup rows, dis / dis_snr / s_re are masked — exclude those rows
    # for those columns when computing best.
    def _val_for_col(ev, col, getter):
        v = getter(ev)
        if v is None:
            return None
        if col in ('s_re', 'dis', 'dis_snr') and ev['epoch'] <= WARMUP:
            return None  # masked at display
        return v

    def _best_per_column(evs, columns, mode):
        results = {}
        for col, getter in columns:
            cands = []
            for ev in evs:
                v = _val_for_col(ev, col, getter)
                if v is not None:
                    try:
                        cands.append((float(v), ev['epoch']))
                    except (TypeError, ValueError):
                        continue
            if not cands:
                results[col] = (None, [])
                continue
            best_v = min(c[0] for c in cands) if mode == 'min' else max(c[0] for c in cands)
            # Tie-break: when multiple epochs match best_v (full-precision tie),
            # return ONLY the latest one (max epoch). Per user 2026-05-29 spec
            # — bold last tied row, not all of them.
            tied_eps = [ep for v, ep in cands if v == best_v]
            results[col] = (best_v, [max(tied_eps)])
        return results

    best_max = _best_per_column(emitted_evs, MAX_COLS, 'max')
    best_min = _best_per_column(emitted_evs, MIN_COLS, 'min')
    out.append("\n[BEST PER COLUMN — milestone rows only]")
    for col, _getter in MAX_COLS + MIN_COLS:
        mode = 'min' if col in ('t_re', 's_re') else 'max'
        info = best_min.get(col) if mode == 'min' else best_max.get(col)
        if info is None or info[0] is None:
            out.append(f"  {col:<10}: (no data)")
        else:
            v, eps = info
            eps_str = ",".join(f"ep{e}" for e in eps)
            out.append(f"  {col:<10}: {mode} = {v:.6f} @ {eps_str}")

    # Emit milestones in chronological order, merging same-epoch labels with ≡
    ev_by_epoch = {e['epoch']: e for e in evals}
    for ep in sorted(label_map.keys()):
        labels_sorted = [lbl for _, lbl in sorted(label_map[ep])]
        merged_label = " ≡ ".join(labels_sorted)
        _emit_eval(merged_label, ev_by_epoch[ep])

# Legacy t_loss/s_loss summary (only — milestone block above already emitted
# all rows incl. latest, best, first, warmup-boundary).
if last_eval:
    out.append(f"  Latest losses (legacy):         "
               f"t_loss={last_eval['t_loss']:.4f}  s_loss={last_eval['s_loss']:.4f}  "
               f"(infer={last_eval['infer']}s eval={last_eval['eval']}s)")
if best_eval:
    _btl = best_eval.get('t_loss')
    _bsl = best_eval.get('s_loss')
    if _btl is not None or _bsl is not None:
        out.append(
            f"  Best losses    (legacy):         "
            f"t_loss={_fmt(_btl)}  s_loss={_fmt(_bsl)}"
        )
if not evals:
    out.append(f"  (no eval logged yet — eval interval = 5)")

# ---------- SPEED HEALTH JUDGMENT (2026-05-27) ----------
# Per-dataset baseline (s/ep) from known-good runs with VUS skipped + canonical config.
# Format: (lo, hi). Lo = ideal (cold cache, no contention); hi = realistic full-load steady-state.
# Keys checked via substring against ds_short output (e.g. "SWaT/A1A2_full", "WaDi/A1").
# More specific keys first (e.g. WaDi/A1 before WaDi) so they match first.
BASELINES = [
    ('SWaT/A1A2',  (8, 10)),
    ('WaDi/A1',    (15, 25)),
    ('WaDi/A2',    (15, 25)),
    ('PSM',        (10, 15)),
    ('simulation', (3, 5)),
]

def judge_speed(ds_name, speed_last5, eval_cost_s, epoch):
    """Return (verdict, reason) for current speed."""
    if speed_last5 is None or speed_last5 <= 0:
        return ('unknown', 'no speed data')
    base = None
    for k, v in BASELINES:
        if k in (ds_name or ''):
            base = (k, v)
            break
    if base is None:
        return ('unknown', f"no baseline for ds={ds_name}")
    base_key, (lo, hi) = base
    ratio = speed_last5 / hi
    # Grace period: first 10 epochs of a dataset
    if epoch is not None and epoch < 10:
        return ('🟢 초기 transition', f"ep{epoch}<10 grace period (last5={speed_last5:.1f}s)")
    if ratio <= 1.5:
        return ('정상', f"last5={speed_last5:.1f}s/ep, baseline {base_key} {lo}-{hi}, ratio={ratio:.2f}x")
    elif ratio <= 2.5:
        return ('주의 ⚠️',
                f"last5={speed_last5:.1f}s/ep, baseline {lo}-{hi}, ratio={ratio:.2f}x — "
                f"CPU contention 의심 (check: ps aux --sort=-%cpu | head -5)")
    elif ratio <= 5.0:
        return ('심각 (저속) 🟠',
                f"last5={speed_last5:.1f}s/ep, baseline {lo}-{hi}, ratio={ratio:.2f}x — "
                f"bg-worker stuck 의심 (check: pgrep -af spawn_main)")
    else:
        return ('🚨 stuck 의심',
                f"last5={speed_last5:.1f}s/ep, baseline {lo}-{hi}, ratio={ratio:.2f}x — "
                f"메인 학습 정지 가능. 자동 정지 X. 사용자 확인 필요.")

_cur_ds_short = ds_short(current_path) if current_path else ''
_cur_epoch = last_ep[0] if last_ep else None
verdict, reason = judge_speed(_cur_ds_short, mean_train_recent, mean_eval_sec, _cur_epoch)

# ---------- AUTO-INTERPRETATION (rule-based, 2026-05-27) ----------
# Read prior monitoring_log.jsonl entries to derive trend-aware interpretation.
# This runs BEFORE writing the current entry, so we get history from prior ticks only.
def _interpret(current_path, last_eval, best_eval, last_ep, mean_train_recent,
               verdict_now, current_run_root):
    """Rule-based interpretation with explicit evidence per field.

    Each verdict carries its supporting data (dataset, epoch, metric values) so
    that reading monitoring_log.jsonl alone is enough to reconstruct WHY each
    judgment was made — no cross-referencing other files needed.
    """
    # Evidence dict: collected from current monitoring snapshot
    evidence = {
        'dataset': ds_short(current_path) if current_path else None,
        'current_train_epoch': last_ep[0] if last_ep else None,
        'total_epochs': last_ep[1] if last_ep else None,
        'latest_eval_epoch': last_eval.get('epoch') if last_eval else None,
        'latest_pak_f1': last_eval.get('pak_f1') if last_eval else None,
        'latest_prc': last_eval.get('prc') if last_eval else None,
        'latest_f1_t': last_eval.get('f1_t') if last_eval else None,
        'latest_disc_snr': last_eval.get('d_snr') if last_eval else None,
        'best_epoch': best_eval.get('epoch') if best_eval else None,
        'best_pak_f1': best_eval.get('pak_f1') if best_eval else None,
        'warmup_boundary': 250,  # teacher_only_warmup_epochs from exp271 canonical
        'eval_interval': 5,
        'speed_last5': mean_train_recent,
    }
    interp = {
        'phase': None,
        'phase_reason': None,
        'best_held_ticks': 0,
        'best_held_evals': None,
        'latest_vs_best': None,
        'speed_trend_3': [],
        'speed_trend_label': None,
        'eval_lag_epochs': None,
        'notable': [],
        'evidence': evidence,
    }
    lines = []

    # Phase: pre-warmup vs post-warmup
    # teacher_only_warmup_epochs hard-coded from exp271 canonical (250). May read from config later.
    WARMUP = 250
    cur_ep = (last_eval['epoch'] if last_eval else (_cur_epoch or 0))
    ds_name = evidence['dataset'] or '?'
    if cur_ep <= WARMUP:
        interp['phase'] = 'pre-warmup'
        interp['phase_reason'] = (f'dataset={ds_name}, latest_eval_epoch={cur_ep} ≤ '
                                  f'teacher_only_warmup_epochs={WARMUP} → teacher-only training (student frozen)')
    else:
        interp['phase'] = 'post-warmup'
        interp['phase_reason'] = (f'dataset={ds_name}, latest_eval_epoch={cur_ep} > '
                                  f'teacher_only_warmup_epochs={WARMUP} → student joined (teacher+student co-training)')

    # Eval lag
    if last_ep and last_eval:
        interp['eval_lag_epochs'] = last_ep[0] - last_eval['epoch']

    # Latest vs best
    if last_eval and best_eval and last_eval.get('pak_f1') is not None and best_eval.get('pak_f1') is not None:
        interp['latest_vs_best'] = round(last_eval['pak_f1'] - best_eval['pak_f1'], 4)

    # Read prior jsonl entries to compute trends
    prior_entries = []
    log_path = None
    if current_run_root and current_run_root.exists():
        log_path = current_run_root / 'monitoring_log.jsonl'
        if log_path.exists():
            try:
                with open(log_path) as f:
                    for line in f:
                        try:
                            prior_entries.append(json.loads(line))
                        except Exception:
                            pass
            except Exception:
                pass

    # Speed trend (last 3 ticks including current)
    prior_speeds = [e.get('speed_last5') for e in prior_entries[-2:]
                    if e.get('speed_last5') is not None]
    if mean_train_recent is not None:
        speeds_seq = prior_speeds + [round(mean_train_recent, 2)]
        interp['speed_trend_3'] = speeds_seq
        if len(speeds_seq) >= 2:
            delta = speeds_seq[-1] - speeds_seq[0]
            rel = abs(delta) / max(speeds_seq[0], 0.1)
            if rel < 0.10:
                interp['speed_trend_label'] = 'stable'
            elif delta > 0:
                interp['speed_trend_label'] = 'degrading' if rel > 0.20 else 'slow-degrading'
            else:
                interp['speed_trend_label'] = 'improving'

    # Best-held tick count (consecutive prior ticks where best_pak_f1 == current best)
    if best_eval and best_eval.get('pak_f1') is not None:
        held = 0
        for e in reversed(prior_entries):
            if e.get('best_pak_f1') == best_eval['pak_f1'] and e.get('best_epoch') == best_eval.get('epoch'):
                held += 1
            else:
                break
        interp['best_held_ticks'] = held

    # Best-held eval count (how many evals since best)
    if last_eval and best_eval and last_eval.get('epoch') and best_eval.get('epoch'):
        # eval interval = 5 (assumed)
        gap = last_eval['epoch'] - best_eval['epoch']
        interp['best_held_evals'] = gap // 5 if gap >= 0 else 0

    # ---- Notable observations (rule-fired alerts) ----
    # Every notable string carries inline evidence so log-only reading is sufficient.
    notable = []

    # Best update detection (compare to previous tick)
    if prior_entries:
        prev_best = prior_entries[-1].get('best_pak_f1')
        prev_best_ep = prior_entries[-1].get('best_epoch')
        cur_best = best_eval.get('pak_f1') if best_eval else None
        cur_best_ep = best_eval.get('epoch') if best_eval else None
        if prev_best is not None and cur_best is not None and cur_best > prev_best + 1e-6:
            notable.append(
                f"★ best PAK_F1 updated [{ds_name}]: "
                f"{prev_best:.4f}@ep{prev_best_ep} → {cur_best:.4f}@ep{cur_best_ep} "
                f"(+{cur_best-prev_best:.4f}) — metric: pak_auc_f1"
            )

    # Plateau detection (best held > 5 evals)
    if interp['best_held_evals'] is not None and interp['best_held_evals'] >= 5:
        notable.append(
            f"plateau [{ds_name}]: best {interp['best_held_evals']} evals 미경신 "
            f"(best={best_eval['pak_f1']:.4f}@ep{best_eval['epoch']}, "
            f"latest={last_eval['pak_f1']:.4f}@ep{last_eval['epoch']}, "
            f"gap={last_eval['epoch']-best_eval['epoch']}ep — metric: pak_auc_f1)"
        )

    # Divergence detection (latest << best)
    if interp['latest_vs_best'] is not None and interp['latest_vs_best'] < -0.05:
        notable.append(
            f"divergence 의심 [{ds_name}]: latest vs best = {interp['latest_vs_best']:+.4f} "
            f"(best={best_eval['pak_f1']:.4f}@ep{best_eval['epoch']}, "
            f"latest={last_eval['pak_f1']:.4f}@ep{last_eval['epoch']} — metric: pak_auc_f1)"
        )

    # Phase transition (just crossed warmup)
    if prior_entries:
        prev_eval_ep = prior_entries[-1].get('latest_eval_epoch')
        if prev_eval_ep is not None and last_eval and last_eval.get('epoch'):
            if prev_eval_ep <= WARMUP < last_eval['epoch']:
                notable.append(
                    f"🎉 student-joining boundary 통과 [{ds_name}]: "
                    f"prev_eval_ep={prev_eval_ep} ≤ {WARMUP} < latest_eval_ep={last_eval['epoch']} "
                    f"→ post-warmup 진입 (student now training, expect PAK_F1 boost)"
                )

    # Speed degradation alert
    if interp['speed_trend_label'] == 'degrading':
        seq = interp['speed_trend_3']
        pct_str = f"{(seq[-1]-seq[0])/seq[0]*100:+.1f}%" if seq[0] > 0 else "N/A (prior=0, dataset transition)"
        notable.append(
            f"⚠️ speed degrading [{ds_name}]: "
            f"speed_last5 sequence={seq} s/ep (3-tick window), "
            f"Δ={seq[-1]-seq[0]:+.2f}s ({pct_str})"
        )

    # Eval lag growing (bg-worker contention indicator)
    if interp['eval_lag_epochs'] is not None and interp['eval_lag_epochs'] > 30:
        notable.append(
            f"⚠️ eval lag = {interp['eval_lag_epochs']} epochs [{ds_name}] "
            f"(current_epoch={evidence['current_train_epoch']}, "
            f"latest_eval_epoch={evidence['latest_eval_epoch']}, eval_interval=5) "
            f"→ async queue 적체; bg-worker 의심"
        )

    # Bg-worker bottleneck (verdict-driven)
    if verdict_now and ('심각' in verdict_now or '🚨' in verdict_now):
        notable.append(
            f"🚨 speed 진단 = '{verdict_now}' [{ds_name}]: "
            f"speed_last5={mean_train_recent:.2f}s/ep → 사용자 확인 필요 (자동 정지 X)"
        )

    interp['notable'] = notable

    # ---- Human-readable lines for status report ----
    lines.append(f"  Phase             : **{interp['phase']}**  ({interp['phase_reason']})")
    if interp['best_held_ticks'] > 0:
        lines.append(f"  Best held         : {interp['best_held_ticks']} tick(s) "
                     f"= {interp['best_held_evals']} eval(s) "
                     f"({best_eval.get('pak_f1'):.4f} @ ep {best_eval.get('epoch')})")
    if interp['latest_vs_best'] is not None:
        sign = '+' if interp['latest_vs_best'] >= 0 else ''
        lines.append(f"  Latest vs best    : {sign}{interp['latest_vs_best']:+.4f}")
    if interp['speed_trend_label']:
        seq_str = ' → '.join(f'{s:.1f}' for s in interp['speed_trend_3'])
        lines.append(f"  Speed trend (3)   : [{seq_str}] s/ep — {interp['speed_trend_label']}")
    if interp['eval_lag_epochs'] is not None:
        lines.append(f"  Eval lag          : {interp['eval_lag_epochs']} epochs")
    if notable:
        lines.append(f"  Notable           :")
        for n in notable:
            lines.append(f"    • {n}")
    return interp, lines

# Determine run root for interp (same logic as later jsonl write block)
_run_root_for_interp = None
if current_path:
    p_ = Path(current_path)
    for ancestor in [p_, p_.parent, p_.parent.parent, p_.parent.parent.parent]:
        if ancestor.name and re.match(r'^\d+_', ancestor.name):
            _run_root_for_interp = ancestor
            break

interpretation, interp_lines = _interpret(
    current_path, last_eval, best_eval, last_ep, mean_train_recent,
    verdict, _run_root_for_interp,
)

# ---------- SPEED HEALTH ----------
out.append(f"\n[SPEED HEALTH]")
out.append(f"  🩺 학습 속도 진단: {verdict}")
out.append(f"     {reason}")

# ---------- INTERPRETATION (rule-based auto-analysis) ----------
out.append(f"\n[INTERPRETATION — rule-based]")
for line in interp_lines:
    out.append(line)

# Health
out.append(f"\n[HEALTH]")
if crit:
    out.append(f"  !! NaN/Inf events: {len(crit)} total — latest epoch {crit[-1][0]} ({crit[-1][1]} batches skipped)")
else:
    out.append(f"  NaN/Inf: none  |  grad_norm logger active")

# Hardware
out.append(f"\n[HARDWARE]")
if gpu:
    out.append(f"  GPU  | mem {gpu.get('mem_u','?')}/{gpu.get('mem_t','?')}MiB ({100*int(gpu.get('mem_u',0))//max(1,int(gpu.get('mem_t',1)))}%)  util {gpu.get('util','?')}%  mem-bw {gpu.get('mem_util','?')}%  power {gpu.get('pw','?')}/{gpu.get('pl','?')}W  temp {gpu.get('temp','?')}°C  fan {gpu.get('fan','?')}%  pstate {gpu.get('ps','?')}  clk gr/mem {gpu.get('cgr','?')}/{gpu.get('cmem','?')}MHz")
proc_rss_str = f"{int(proc_rss_kb)/1024:.0f}MiB" if proc_rss_kb.isdigit() else "?"
out.append(f"  CPU  | load1/5/15: {'/'.join(loadavg)} of {ncpu}c  |  worker(pid={WORKER_PID}): %CPU={proc_cpu} %MEM={proc_mem} RSS={proc_rss_str}")
out.append(f"  RAM  | {mem_used}/{mem_total}MiB used ({100*int(mem_used)//max(1,int(mem_total)) if mem_used != '?' else '?'}%)  available {mem_avail}MiB  |  swap {swap_used}MiB")
if len(df_parts) >= 5:
    out.append(f"  Disk | results FS: {df_parts[2]}/{df_parts[1]} used ({df_parts[4]} full)  |  results/ {res_size}  this exp dir: {exp_size}")

# Completed datasets in this run (with metrics if any)
if completed_paths:
    out.append(f"\n[COMPLETED DATASETS — this exp271 run]")
    for p in completed_paths:
        info = scan_completed(p)
        files_str = ", ".join(info['files']) if info['files'] else "(no summary files written yet)"
        out.append(f"  {ds_short(p)}: {files_str}")
        # try to extract a best metric
        for fn in info['files']:
            d = info.get(fn, {})
            if isinstance(d, dict):
                # find PAK_F1 / best_pak_f1 / similar keys
                for k in ['best_pak_f1','pak_f1','best_PAK_F1','PAK_F1','pak_auc_f1']:
                    if k in d:
                        out.append(f"      {k}: {d[k]}")
                        break

out.append("=" * 50)
print('\n'.join(out), flush=True)

# ---------- MONITORING LOG (append to exp_dir/monitoring_log.jsonl) ----------
# Per user request 2026-05-27: store monitoring records inside the run's exp dir
# (NOT /tmp), so they survive across sessions and travel with the experiment results.
try:
    if current_path:
        # current_path = .../271_<ts>_.../<dataset>/<sub>  →  run_root = .../271_<ts>_...
        p = Path(current_path)
        run_root = None
        for ancestor in [p, p.parent, p.parent.parent, p.parent.parent.parent]:
            if ancestor.name and re.match(r'^\d+_', ancestor.name):
                run_root = ancestor
                break
        if run_root and run_root.exists():
            log_path = run_root / 'monitoring_log.jsonl'
            def _to_int(x):
                try: return int(x)
                except: return None
            def _to_float(x):
                try: return float(x)
                except: return None
            record = {
                'ts': now.isoformat(),
                'runtime': runtime,
                'dataset_idx': ds_idx,
                'dataset': _cur_ds_short,
                'epoch': _cur_epoch,
                'total_epochs': last_ep[1] if last_ep else None,
                'speed_last5': mean_train_recent,
                'speed_last20': mean_train_sec,
                'eval_cost': mean_eval_sec,
                'amortized': per_epoch_avg,
                'gpu_util': _to_int(gpu.get('util')) if gpu else None,
                'gpu_mem_mb': _to_int(gpu.get('mem_u')) if gpu else None,
                'gpu_temp': _to_int(gpu.get('temp')) if gpu else None,
                'gpu_power': _to_float(gpu.get('pw')) if gpu else None,
                'cpu_load1': _to_float(loadavg[0]) if loadavg else None,
                'worker_cpu_pct': _to_float(proc_cpu),
                'ram_used_mb': _to_int(mem_used),
                'latest_eval_epoch': last_eval['epoch'] if last_eval else None,
                'latest_pak_f1': last_eval.get('pak_f1') if last_eval else None,
                'best_pak_f1': best_eval.get('pak_f1') if best_eval else None,
                'best_epoch': best_eval.get('epoch') if best_eval else None,
                'nan_inf_count': len(crit) if crit else 0,
                'judgment_verdict': verdict,
                'judgment_reason': reason,
                'interpretation': interpretation,  # rule-based auto-analysis dict
            }
            with open(log_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
except Exception as e:
    # Logging is best-effort; never block monitoring output
    print(f"  [monitoring_log warn] {type(e).__name__}: {e}", file=sys.stderr)
