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
# New log format (family-ordered: point-strict | PA%K | event/range | VUS | diag)
# 2026-05-27 onward: prc f1 f1_t pak_f1 pak_prc aff_f1 rf1 vus_pr vus_roc d_snr
RE_EVAL_NEW = re.compile(
    r'\[Epoch\s*(\d+)\]\s+'
    r'PRC=([\d.]+)\s+F1=([\d.]+)\s+F1_T=([\d.]+)\s+'
    r'PAK_F1=([\d.]+)\s+PAK_PRC=([\d.]+)\s+'
    r'AFF_F1=([\d.]+)\s+RF1=([\d.]+)\s+'
    r'VUS_PR=([\d.]+)\s+VUS_ROC=([\d.]+)\s+'
    r'd_SNR=([\d.]+)\s+\|\s+'
    r't_loss=([\d.eE+-]+)\s+s_loss=([\d.eE+-]+)\s+\(infer=(\d+)s\s+eval=(\d+)s\)(?:\s+\[async\])?(\s+★)?'
)
# Old format (exp271 SWaT and earlier) — fallback for backward compat
RE_EVAL_OLD = re.compile(
    r'\[Epoch\s*(\d+)\]\s+PRC=([\d.]+)\s+PAK_F1=([\d.]+)\s+PAK_PRC=([\d.]+)\s+F1_T=([\d.]+)\s+d_SNR=([\d.]+)\s+\|\s+t_loss=([\d.eE+-]+)\s+s_loss=([\d.eE+-]+)\s+\(infer=(\d+)s\s+eval=(\d+)s\)(?:\s+\[async\])?(\s+★)?'
)
RE_CRIT = re.compile(r'\[CRITICAL\]\s+epoch\s+(\d+):\s+(\d+)\s+batch')

with open(LOG, 'rb') as f:
    f.seek(0, 2); end = f.tell()
    f.seek(max(0, end - 400_000))
    tail_lines = f.read().decode('utf-8', errors='replace').splitlines()

done_epochs, inprog, evals, crit = [], None, [], []
for line in tail_lines:
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
            prc=float(m.group(2)), f1=float(m.group(3)), f1_t=float(m.group(4)),
            # PA%K
            pak_f1=float(m.group(5)), pak_prc=float(m.group(6)),
            # event/range
            aff_f1=float(m.group(7)), r_f1=float(m.group(8)),
            # VUS
            vus_pr=float(m.group(9)), vus_roc=float(m.group(10)),
            # diagnostic
            d_snr=float(m.group(11)),
            t_loss=float(m.group(12)), s_loss=float(m.group(13)),
            infer=int(m.group(14)), eval=int(m.group(15)), best=bool(m.group(16))
        ))
        continue
    m = RE_EVAL_OLD.search(line)
    if m:
        evals.append(dict(
            epoch=int(m.group(1)), prc=float(m.group(2)), pak_f1=float(m.group(3)),
            pak_prc=float(m.group(4)), f1_t=float(m.group(5)),
            f1=None, vus_pr=None, vus_roc=None, aff_f1=None, r_f1=None,  # old-format
            d_snr=float(m.group(6)),
            t_loss=float(m.group(7)), s_loss=float(m.group(8)),
            infer=int(m.group(9)), eval=int(m.group(10)), best=bool(m.group(11))
        ))
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

if last_eval:
    # Family-ordered: point-strict | PA%K | event/range | VUS | diag
    out.append(
        f"  Latest eval (ep {last_eval['epoch']:>3}):"
        f"  [point] prc={_fmt(last_eval['prc'])} f1={_fmt(last_eval.get('f1'))} f1_t={_fmt(last_eval['f1_t'])}"
        f"  [PA%K] pak_auc_f1={_fmt(last_eval['pak_f1'])} pak_auc_prc={_fmt(last_eval['pak_prc'])}"
        f"  [range] aff_f1={_fmt(last_eval.get('aff_f1'))} r_f1={_fmt(last_eval.get('r_f1'))}"
        f"  [VUS] vus_pr={_fmt(last_eval.get('vus_pr'))} vus_roc={_fmt(last_eval.get('vus_roc'))}"
        f"  [diag] disc_snr={_fmt(last_eval['d_snr'])}"
    )
    out.append(f"  Latest losses:                  t_loss={last_eval['t_loss']:.4f}  s_loss={last_eval['s_loss']:.4f}  (infer={last_eval['infer']}s eval={last_eval['eval']}s)")
if best_eval:
    flag = " ★ (current best)" if best_eval is evals[-1] and best_eval['best'] else ""
    out.append(
        f"  Best pak_auc_f1 (ep {best_eval['epoch']:>3}):"
        f"  [point] prc={_fmt(best_eval['prc'])} f1={_fmt(best_eval.get('f1'))} f1_t={_fmt(best_eval['f1_t'])}"
        f"  [PA%K] pak_auc_f1={_fmt(best_eval['pak_f1'])} pak_auc_prc={_fmt(best_eval['pak_prc'])}"
        f"  [range] aff_f1={_fmt(best_eval.get('aff_f1'))} r_f1={_fmt(best_eval.get('r_f1'))}"
        f"  [VUS] vus_pr={_fmt(best_eval.get('vus_pr'))} vus_roc={_fmt(best_eval.get('vus_roc'))}"
        f"  [diag] disc_snr={_fmt(best_eval['d_snr'])}{flag}"
    )
if not evals:
    out.append(f"  (no eval logged yet — eval interval = 5)")

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
