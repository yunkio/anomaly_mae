#!/bin/bash
# Complete the missing 30/15-epoch w/o-GRL runs for TEP Table 3.
#
# Axes filled (canonical seed 42 already exists and is only validated):
#   1. fixed canonical data, model seeds 40/41/43/44
#   2. data-allocation + model seeds 40/41/43/44
#
# Each run is a full 30-epoch training with a 15-epoch Teacher-only phase,
# matching the existing historical TEP roots.  Completed cells are skipped only
# after their epoch/score artifacts are validated.  Incomplete directories are
# preserved under temp/tep_table3_partial_quarantine before a clean cell rerun.
set -euo pipefail

cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
RB=scripts/run_base_experiments.py
BASE=results/experiments/TEP_phase2_win100_ep30
LOG=temp/baseline_experiment_run/tep_table3_nogrl_multiseed.log
STAMP=$(date '+%Y%m%d_%H%M%S')
QUARANTINE="temp/tep_table3_partial_quarantine/$STAMP"
FOLDS="fstep frand fds funk"
SEEDS="40 41 43 44"

mkdir -p "$(dirname "$LOG")" "$QUARANTINE"

is_complete() {
  local dir=$1
  "$PY" - "$dir" <<'PY'
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
try:
    rows = json.loads((d / "epoch_metrics.json").read_text())["epochs"]
    epochs = [int(row["epoch"]) for row in rows]
    scores = sorted((d / "epoch_scores").glob("epoch_*_scores.npz"))
    ok = (
        (d / "experiment_metadata.json").is_file()
        and (d / "best_epoch_train_scores.npz").is_file()
        and epochs == list(range(3, 31, 3))
        and len(scores) == 10
        and scores[-1].name == "epoch_030_scores.npz"
    )
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

quarantine_partial() {
  local dir=$1
  [ -e "$dir" ] || return 0
  local resolved
  resolved=$(realpath -m "$dir")
  case "$resolved" in
    "$PWD"/results/experiments/*) ;;
    *) echo "REFUSE unsafe quarantine target: $resolved" | tee -a "$LOG"; return 1 ;;
  esac
  local rel=${resolved#"$PWD"/results/experiments/}
  local dst="$QUARANTINE/$rel"
  mkdir -p "$(dirname "$dst")"
  mv -- "$resolved" "$dst"
  echo "[$(date '+%F %T')] quarantined partial: $rel" | tee -a "$LOG"
}

run_cell() {
  local axis=$1 seed=$2 root=$3 datadir=$4 fold=$5
  local ds="TEP_typegen_${fold}"
  local out="$root/phase2_nogrl/TEP/typegen_${fold}"
  if is_complete "$out"; then
    echo "[$(date '+%F %T')] SKIP complete: $axis seed=$seed $fold" | tee -a "$LOG"
    return 0
  fi
  quarantine_partial "$out"
  echo "[$(date '+%F %T')] START: $axis seed=$seed $fold" | tee -a "$LOG"
  local ov="official=True batch_size=768 official_keep_checkpoints=False random_seed=${seed} num_epochs=30 teacher_only_warmup_epochs=15 seq_length=100 patch_size=5 num_patches=20"
  if [ -n "$datadir" ]; then
    TEP_TYPEGEN_DATA_DIR="$datadir" "$PY" "$RB" --set A --dataset "$ds" \
      --output-base "$root/phase2_nogrl" --config-override $ov \
      use_grl=False anomaly_loss_weight=0.0 >> "$LOG" 2>&1
  else
    "$PY" "$RB" --set A --dataset "$ds" --output-base "$root/phase2_nogrl" \
      --config-override $ov use_grl=False anomaly_loss_weight=0.0 >> "$LOG" 2>&1
  fi
  if ! is_complete "$out"; then
    echo "[$(date '+%F %T')] ERROR incomplete after run: $out" | tee -a "$LOG"
    return 1
  fi
  echo "[$(date '+%F %T')] DONE: $axis seed=$seed $fold" | tee -a "$LOG"
}

echo "===== TEP TABLE3 w/o-GRL 30/15 START $(date '+%F %T') =====" | tee -a "$LOG"

# The canonical seed-42 w/o-GRL run is the shared fifth member of both axes.
for fold in $FOLDS; do
  canonical="$BASE/phase2_nogrl/TEP/typegen_${fold}"
  if ! is_complete "$canonical"; then
    echo "ERROR: canonical seed42 w/o-GRL is incomplete: $canonical" | tee -a "$LOG"
    exit 1
  fi
done
echo "[$(date '+%F %T')] canonical seed42 validated (4/4)" | tee -a "$LOG"

for seed in $SEEDS; do
  for fold in $FOLDS; do
    run_cell fixed "$seed" "${BASE}_s${seed}" "" "$fold"
  done
done

for seed in $SEEDS; do
  datadir="$PWD/scripts/TEP/data_dataseed${seed}"
  [ -f "$datadir/test_stream.npz" ] || {
    echo "ERROR: missing data seed directory: $datadir" | tee -a "$LOG"
    exit 1
  }
  for fold in $FOLDS; do
    run_cell data_and_model "$seed" "${BASE}_dataseed${seed}" "$datadir" "$fold"
  done
done

echo "===== TEP TABLE3 w/o-GRL 30/15 COMPLETE $(date '+%F %T') =====" | tee -a "$LOG"
