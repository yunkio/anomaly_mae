#!/usr/bin/env bash
# =============================================================================
# GCN-LSTM redo — R4 option A (Keras init semantics + dead-head guard)
#
#   Re-runs the 8번 reseed gcn_lstm cells with --gcnlstm-keras-init:
#     5 seeds (8-k <- 42,43,40,41,44) x 4 experiments
#     (psm / swat_a1a2 / wadi_14days_A1 / wadi_14days_A2, normalonly variant)
#   Per run: run_baseline.py --seed <S> --early-stop --neural-epochs 50
#            --normalize-mode minmax --eval-interval 1 --gcnlstm-keras-init
#   Output OVERWRITES the existing 8-k dirs' gcn_lstm cells, so Step 1 first
#   MOVES every pre-redo gcn_lstm result dir to .trash/<YYMMDD>/gcnlstm_pre_redo/
#   (one-time: if a backup destination already exists, the move is skipped so a
#   RE-LAUNCH can never trash freshly redone results).
#   Resumable: run_baseline.py auto-skips cells that already have
#   epoch_metrics.json, so re-launching continues where it stopped.
#
#   DO NOT start while the reseed chain (or any other baseline run) occupies
#   the GPU — the launcher refuses to start if another run_baseline process is
#   alive (override: ALLOW_CONCURRENT=1).
# =============================================================================
set -u
cd /home/ykio/notebooks/claude || exit 1
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python

SUFFIX=20260606_175756          # original 8번 dir timestamp suffix
EXPROOT=comparison/results/experiments
RUNDIR=temp/baseline_experiment_run
mkdir -p "$RUNDIR"
TS=$(date +%Y%m%d_%H%M%S)
MASTER="$RUNDIR/gcnlstm_redo_${TS}.log"
STATEF="$RUNDIR/gcnlstm_redo_current.txt"
TRASH="/home/ykio/notebooks/claude/.trash/$(date +%y%m%d)/gcnlstm_pre_redo"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

# --- Safety gate: never overlap another baseline run (GPU chain) ---
if pgrep -f "comparison/run_baseline" >/dev/null 2>&1; then
  if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    echo "FATAL: another comparison/run_baseline* process is running (GPU chain busy)."
    echo "       Wait for the chain to finish, or override with ALLOW_CONCURRENT=1."
    exit 1
  fi
  echo "WARNING: ALLOW_CONCURRENT=1 — starting despite a live run_baseline process."
fi

SEEDS=(42 43 40 41 44)          # 8-k <- SEEDS[k-1] (reseed-chain parity)
EXPERIMENTS=(psm_normalonly swat_a1a2_normalonly wadi_14days_A1_normalonly wadi_14days_A2_normalonly)
# results_dir_name per experiment (comparison/experiment_configs.py)
declare -A EXP_DIRS=(
  [psm_normalonly]="PSM"
  [swat_a1a2_normalonly]="SWaT/A1A2_full"
  [wadi_14days_A1_normalonly]="WaDi/A1"
  [wadi_14days_A2_normalonly]="WaDi/A2"
)

log "=== GCN-LSTM REDO START (pid $$) ==="
log "Seeds 42,43,40,41,44 -> 8-1..8-5 | 4 normalonly experiments | keras_init=ON"
log "Backup dir: $TRASH"

# --- Step 1: one-time backup (MOVE) of pre-redo gcn_lstm result dirs ---
n_moved=0; n_kept=0; n_absent=0
for k in 1 2 3 4 5; do
  base="$EXPROOT/8-${k}_${SUFFIX}_baseline"
  for exp in "${EXPERIMENTS[@]}"; do
    src="$base/${EXP_DIRS[$exp]}/gcn_lstm"
    dst="$TRASH/8-${k}_${SUFFIX}_baseline/${EXP_DIRS[$exp]}/gcn_lstm"
    if [ -e "$dst" ]; then
      # Backup already taken on a previous launch -> whatever sits in src now is
      # NEW redo output (or nothing). Never move it; resume handles the rest.
      log "  [KEEP] backup exists, leaving current dir for resume: $src"
      n_kept=$((n_kept+1))
    elif [ -d "$src" ]; then
      mkdir -p "$(dirname "$dst")"
      mv "$src" "$dst"
      log "  [BACKUP] $src -> $dst"
      n_moved=$((n_moved+1))
    else
      log "  [NONE] nothing to back up: $src"
      n_absent=$((n_absent+1))
    fi
  done
done
log "Backup summary: moved=$n_moved kept(resume)=$n_kept absent=$n_absent (expected 20 cells total)"

# --- Step 2: 5 seeds x 4 experiments (seed-major; resumable via auto-skip) ---
fail=0
for k in 1 2 3 4 5; do
  seed=${SEEDS[$((k-1))]}
  base="$EXPROOT/8-${k}_${SUFFIX}_baseline"
  for exp in "${EXPERIMENTS[@]}"; do
    tag="8-${k}_${exp}"
    rlog="$RUNDIR/gcnlstm_redo_${tag}_${TS}.log"
    echo "$tag seed=$seed (started $(date +%H:%M:%S))" > "$STATEF"
    log ">>> RUN $tag  seed=$seed  out=$base"
    log "    log: $rlog"
    "$PY" comparison/run_baseline.py \
        --experiment "$exp" \
        --model gcn_lstm \
        --output-base "$base" \
        --normalize-mode minmax \
        --neural-epochs 50 \
        --eval-interval 1 \
        --seed "$seed" --early-stop \
        --gcnlstm-keras-init \
        >> "$rlog" 2>&1
    rc=$?
    log "<<< DONE $tag (exit $rc)"
    [ "$rc" -ne 0 ] && fail=$((fail+1))
  done
done

log "=== GCN-LSTM REDO COMPLETE (failures: $fail) ==="
echo "ALL DONE $(date +%H:%M:%S) failures=$fail" > "$STATEF"
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
