#!/usr/bin/env bash
# Wait for the already-queued paper NRDetector encoder-LR experiment, then run
# the two nrdetector_full LR arms. The observed process is never signaled.
set -euo pipefail

cd /home/ykio/notebooks/claude
NRD_QUEUE_PID=${1:?usage: queue_nrdetector_full_lr_grid_after_nrd.sh NRD_QUEUE_PID}
UPSTREAM_LOG=temp/baseline_experiment_run/nrdetector_encoder_lr_1e-5_queue.log
LOG=temp/baseline_experiment_run/nrdetector_full_lr_grid_queue.log

exec > >(tee -a "$LOG") 2>&1
echo "===== NRDETECTOR_FULL LR GRID QUEUE START $(date '+%F %T') ====="
UPSTREAM_LINES=$(wc -l < "$UPSTREAM_LOG" 2>/dev/null || printf '0')

if [ -r "/proc/$NRD_QUEUE_PID/stat" ]; then
  NRD_START=$(awk '{print $22}' "/proc/$NRD_QUEUE_PID/stat")
  echo "[$(date '+%F %T')] waiting for upstream NRDetector queue pid=$NRD_QUEUE_PID"
  while [ -r "/proc/$NRD_QUEUE_PID/stat" ]; do
    CURRENT_START=$(awk '{print $22}' "/proc/$NRD_QUEUE_PID/stat" 2>/dev/null || true)
    [ "$CURRENT_START" = "$NRD_START" ] || break
    sleep 60
  done
fi

if ! tail -n "+$((UPSTREAM_LINES + 1))" "$UPSTREAM_LOG" \
    | grep -q 'NRDETECTOR encoder-lr QUEUE COMPLETE'; then
  echo "ERROR: upstream NRDetector queue ended without its completion marker; nrdetector_full grid was not started"
  exit 1
fi

echo "[$(date '+%F %T')] upstream complete; starting nrdetector_full LR grid"
bash scripts/run_nrdetector_full_lr_grid_5seed.sh
echo "===== NRDETECTOR_FULL LR GRID QUEUE COMPLETE $(date '+%F %T') ====="
