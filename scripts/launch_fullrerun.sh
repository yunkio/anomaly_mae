#!/usr/bin/env bash
# FRESH full re-run from exp271 (271 -> 274 -> 285..313, 7 datasets, bf16/lr=0.001).
# Prepared 2026-06-02. PRECONDITION: results/experiments/ is EMPTY (old partial moved to
# .trash/0602_pre_full_rerun2/). Each exp dir = {num}_{TS}_{suffix} with a single fresh TS;
# run_base --output-base bypasses auto-numbering → no directory tangling.
set -euo pipefail
cd /home/ykio/notebooks/TSMAE
source /home/ykio/anaconda3/etc/profile.d/conda.sh && conda activate dc_vis

QUEUE=configs/queue_fullrerun_20260601_190603.json

# Safety: refuse to start if any experiment dir already exists (prevents tangling).
if [ -n "$(ls -A results/experiments 2>/dev/null)" ]; then
  echo "ABORT: results/experiments/ is not empty — move leftovers to .trash first:"
  ls -1 results/experiments/
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG=temp/phase1_logs/fullrerun_${TS}.log
mkdir -p temp/phase1_logs

setsid nohup python -u scripts/run_fullrerun.py "$QUEUE" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/exp271_train_pid.txt
echo "$LOG" > /tmp/exp271_train_log.txt
echo "LAUNCHED fresh fullrerun  pid=$PID  log=$LOG  TS=$TS"
echo "  queue=$QUEUE (31 exp: 271,274,285..313 x 7 datasets, bf16/lr=0.001)"
