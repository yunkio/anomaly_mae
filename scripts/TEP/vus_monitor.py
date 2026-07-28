"""vus_monitor.py — reliable progress monitor for the VUS-PR computation + memory/CPU.
Categorizes vus_results.json keys (MAE / breadth / simple), shows %, rate, ETA, memory."""
import json, os, time, subprocess, sys
R = 'results/experiments/TEP_phase2_win100_ep30'
P = f'{R}/vus_results.json'

# expected key sets
MAE = [f'{c}_{fk}' for fk in ('fstep', 'frand', 'fds', 'funk') for c in ('A', 'B', 'D', 'nogrl')] + ['B0']  # 17
BR = []
for ho in ('step', 'rand', 'ds', 'unk'):
    tf = [f for f in ('step', 'rand', 'ds', 'unk') if f != ho]
    import itertools
    for k in range(4):
        for sub in itertools.combinations(tf, k):
            tag = 'lofo' if k == 3 else ('k0' if k == 0 else '-'.join(f for f in tf if f in sub))
            BR.append(f'br_{ho}_{tag}')
SIMPLE = [f'simple_{b}_{fk}' for b in ('Random', 'PCA', 'NN', 'Sensor', 'L2') for fk in ('fstep', 'frand', 'fds', 'funk')]  # 20
TOTAL = len(MAE) + len(BR) + len(SIMPLE)

d = json.load(open(P)) if os.path.exists(P) else {}
have = set(d.keys())
def pct(keys): n = sum(1 for k in keys if k in have); return n, len(keys), 100 * n / len(keys)

print(f'=== VUS-PR 진행 모니터 ({time.strftime("%H:%M:%S")}) ===')
for name, keys in [('MAE conditions', MAE), ('Breadth (Shapley)', BR), ('Simple baselines', SIMPLE)]:
    n, t, p = pct(keys)
    bar = '█' * int(p / 5) + '░' * (20 - int(p / 5))
    print(f'  {name:<18} [{bar}] {n:>2}/{t} ({p:.0f}%)')
n_all = len(have & set(MAE + BR + SIMPLE))
print(f'  {"전체":<18} {n_all}/{TOTAL} ({100*n_all/TOTAL:.0f}%)')

# running?
try:
    pids = subprocess.check_output(['pgrep', '-f', 'build_vus.py']).decode().split()
except Exception:
    pids = []
print(f'  build_vus: {"실행중 PID="+",".join(pids) if pids else "종료"}')

# current key (last done in log)
try:
    log = open('/tmp/an_vus.txt').read().splitlines()
    last = [l for l in log if 'done ' in l][-1] if any('done ' in l for l in log) else '?'
    print(f'  마지막 완료: {last.strip()}')
except Exception:
    pass

# memory
mem = subprocess.check_output(['free', '-g']).decode().splitlines()[1].split()
print(f'  메모리: {mem[2]}G 사용 / {mem[1]}G 총 / {mem[6]}G 여유  | python proc: {subprocess.check_output(["pgrep","-c","python"]).decode().strip()}')

# ETA from mtime rate (rough)
if pids and n_all < TOTAL:
    age = time.time() - os.path.getmtime(P)
    print(f'  (마지막 저장 {age:.0f}s 전; 잔여 {TOTAL-n_all} keys)')
