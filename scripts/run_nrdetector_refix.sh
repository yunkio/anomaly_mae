#!/usr/bin/env bash
# =============================================================================
# NRdetector refix re-run (2026-07-24)
#   Fixes applied to comparison/baselines/nrdetector/wrapper.py:
#     (i)  gate threshold float64 (constant-seg knife-edge -> deterministic all-pass)
#     (ii) BCE backstop removed (misfired at ep1-2 on WaDi/SWaT natural BCE floor)
#   Scope: nrdetector + nrdetector_full (paired) x 4 experiments x 5 seeds = 40 runs
#   Output: SAME standard dirs (9-k weak_ssl). Old results already MOVED to
#           temp/0724/nrdetector_pre_fix/ (structure preserved) by the orchestrator.
#   Waits for any running run_baseline.py (catch retry) to finish before starting.
#   Resumable: auto-skips cells whose epoch_metrics.json already exists.
# =============================================================================
set -u
cd /home/ykio/notebooks/claude || exit 1
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
EXPROOT=comparison/results/experiments
RUNDIR=temp/baseline_experiment_run
TS=$(date +%Y%m%d_%H%M%S)
MASTER="$RUNDIR/nrdetector_refix_${TS}.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

log "=== NRDETECTOR REFIX START (pid $$) — waiting for GPU (catch retry) ==="
while pgrep -f "run_baseline.py" > /dev/null; do sleep 300; done
log "GPU free — starting 40 runs"

SEEDS=(42 43 40 41 44)
FAIL=0
for k in 1 2 3 4 5; do
  seed=${SEEDS[$((k-1))]}
  out=$(ls -d $EXPROOT/9-${k}_*weak_ssl)
  for model in nrdetector nrdetector_full; do
    for exp in psm swat_a1a2 wadi_14days_A1 wadi_14days_A2; do
      # skip if already done (resume safety)
      case $exp in
        psm) sub=PSM;; swat_a1a2) sub=SWaT/A1A2_full;;
        wadi_14days_A1) sub=WaDi/A1;; wadi_14days_A2) sub=WaDi/A2;;
      esac
      if [ -f "$out/$sub/$model/epoch_metrics.json" ]; then
        log "SKIP 9-$k $model $exp (exists)"; continue
      fi
      rlog="$RUNDIR/nrdetector_refix_9-${k}_${model}_${exp}_${TS}.log"
      log ">>> RUN 9-$k seed=$seed $model x $exp"
      "$PY" comparison/run_baseline.py \
          --experiment "$exp" --model "$model" \
          --output-base "$out" \
          --sota-epochs 50 --eval-interval 1 --normalize-mode minmax \
          --seed "$seed" --early-stop \
          >> "$rlog" 2>&1
      rc=$?
      log "<<< DONE 9-$k $model $exp (exit $rc)"
      [ $rc -ne 0 ] && FAIL=$((FAIL+1))
    done
  done
done
log "=== NRDETECTOR REFIX COMPLETE (failures: $FAIL) ==="
