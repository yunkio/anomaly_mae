#!/usr/bin/env bash
# Wait for the already-running TEP completion service, then start the queued
# NRDetector encoder-lr sensitivity run. The observed process is never signaled.
set -euo pipefail

cd /home/ykio/notebooks/claude
TEP_PID=${1:?usage: queue_nrdetector_encoder_lr_after_tep.sh TEP_PIPELINE_PID}
TEP_LOG=temp/baseline_experiment_run/tep_completion_pipeline.log
LOG=temp/baseline_experiment_run/nrdetector_encoder_lr_1e-5_queue.log

exec > >(tee -a "$LOG") 2>&1
echo "===== NRDETECTOR encoder-lr QUEUE START $(date '+%F %T') ====="

if [ -r "/proc/$TEP_PID/stat" ]; then
  TEP_START=$(awk '{print $22}' "/proc/$TEP_PID/stat")
  echo "[$(date '+%F %T')] waiting for TEP completion pid=$TEP_PID"
  while [ -r "/proc/$TEP_PID/stat" ]; do
    CURRENT_START=$(awk '{print $22}' "/proc/$TEP_PID/stat" 2>/dev/null || true)
    [ "$CURRENT_START" = "$TEP_START" ] || break
    sleep 60
  done
fi

if ! grep -q 'TEP COMPLETION PIPELINE COMPLETE' "$TEP_LOG"; then
  echo "ERROR: TEP pipeline ended without its completion marker; NRDetector was not started"
  exit 1
fi

echo "[$(date '+%F %T')] TEP complete; starting NRDetector encoder_lr=1e-5 queue"
bash scripts/run_nrdetector_encoder_lr_1e5_5seed.sh
echo "===== NRDETECTOR encoder-lr QUEUE COMPLETE $(date '+%F %T') ====="
