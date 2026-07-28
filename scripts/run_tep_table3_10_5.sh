#!/bin/bash
# Paper-horizon TEP Table 3 training: 10 total epochs / 5 Teacher-only epochs.
#
# Runs every learned Table-3 condition over both requested five-seed axes:
#   - LASAD (phase2_A)
#   - Label-blind (phase2_B)
#   - w/o GRL (phase2_nogrl)
# Teacher-only is evaluated from LASAD's final-epoch teacher reconstruction,
# so it requires no separate training.  The five simple baselines reuse their
# already complete score artifacts because they have no epoch budget.
#
# Unique learned cells: 9 seed/data allocations x 3 conditions x 4 folds = 108.
# Canonical seed 42 is shared by the two axes.
set -euo pipefail

cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
RB=scripts/run_base_experiments.py
BASE=results/experiments/TEP_table3_win100_ep10_warm5
LOG=temp/baseline_experiment_run/tep_table3_10_5.log
STAMP=$(date '+%Y%m%d_%H%M%S')
QUARANTINE="temp/tep_table3_10_5_partial_quarantine/$STAMP"
FOLDS="fstep frand fds funk"
SEEDS="40 41 43 44"
CONDITIONS="A B nogrl"

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
    meta = json.loads((d / "experiment_metadata.json").read_text())
    cfg = meta.get("config", meta)
    scores = sorted((d / "epoch_scores").glob("epoch_*_scores.npz"))
    ok = (
        int(cfg.get("num_epochs", -1)) == 10
        and int(cfg.get("teacher_only_warmup_epochs", -1)) == 5
        and epochs == [3, 6, 9, 10]
        and len(scores) == 4
        and scores[-1].name == "epoch_010_scores.npz"
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
    "$PWD"/results/experiments/TEP_table3_win100_ep10_warm5*) ;;
    *) echo "REFUSE unsafe quarantine target: $resolved" | tee -a "$LOG"; return 1 ;;
  esac
  local rel=${resolved#"$PWD"/results/experiments/}
  local dst="$QUARANTINE/$rel"
  mkdir -p "$(dirname "$dst")"
  mv -- "$resolved" "$dst"
  echo "[$(date '+%F %T')] quarantined partial: $rel" | tee -a "$LOG"
}

condition_args() {
  case "$1" in
    A) echo "phase2_A|" ;;
    B) echo "phase2_B|blind_train_labels=True" ;;
    nogrl) echo "phase2_nogrl|use_grl=False anomaly_loss_weight=0.0" ;;
    *) return 1 ;;
  esac
}

run_condition_group() {
  local axis=$1 seed=$2 root=$3 datadir=$4 condition=$5
  local spec phase extra
  spec=$(condition_args "$condition")
  phase=${spec%%|*}
  extra=${spec#*|}
  local fold out
  local datasets=()
  for fold in $FOLDS; do
    out="$root/$phase/TEP/typegen_${fold}"
    if is_complete "$out"; then
      echo "[$(date '+%F %T')] SKIP complete: $axis seed=$seed $condition $fold" | tee -a "$LOG"
      continue
    fi
    quarantine_partial "$out"
    datasets+=("TEP_typegen_${fold}")
    echo "[$(date '+%F %T')] QUEUE: $axis seed=$seed $condition $fold" | tee -a "$LOG"
  done
  [ ${#datasets[@]} -gt 0 ] || return 0

  echo "[$(date '+%F %T')] START GROUP: $axis seed=$seed $condition folds=${datasets[*]}" | tee -a "$LOG"
  local ov="official=True batch_size=768 official_keep_checkpoints=False random_seed=${seed} num_epochs=10 teacher_only_warmup_epochs=5 seq_length=100 patch_size=5 num_patches=20"
  if [ -n "$datadir" ]; then
    TEP_TYPEGEN_DATA_DIR="$datadir" "$PY" "$RB" --set A --dataset "${datasets[@]}" \
      --output-base "$root/$phase" --config-override $ov $extra >> "$LOG" 2>&1
  else
    "$PY" "$RB" --set A --dataset "${datasets[@]}" --output-base "$root/$phase" \
      --config-override $ov $extra >> "$LOG" 2>&1
  fi
  for fold in $FOLDS; do
    out="$root/$phase/TEP/typegen_${fold}"
    if ! is_complete "$out"; then
      echo "[$(date '+%F %T')] ERROR incomplete after group: $out" | tee -a "$LOG"
      return 1
    fi
  done
  echo "[$(date '+%F %T')] DONE GROUP: $axis seed=$seed $condition (4/4 folds complete)" | tee -a "$LOG"
}

run_all_conditions() {
  local axis=$1 seed=$2 root=$3 datadir=$4
  for condition in $CONDITIONS; do
    run_condition_group "$axis" "$seed" "$root" "$datadir" "$condition"
  done
}

echo "===== TEP TABLE3 10/5 START $(date '+%F %T') =====" | tee -a "$LOG"

# Canonical seed/data allocation 42 is shared by both axes.
run_all_conditions canonical 42 "$BASE" ""

for seed in $SEEDS; do
  run_all_conditions fixed "$seed" "${BASE}_s${seed}" ""
done

for seed in $SEEDS; do
  datadir="$PWD/scripts/TEP/data_dataseed${seed}"
  [ -f "$datadir/test_stream.npz" ] || {
    echo "ERROR: missing data seed directory: $datadir" | tee -a "$LOG"
    exit 1
  }
  run_all_conditions data_and_model "$seed" "${BASE}_dataseed${seed}" "$datadir"
done

echo "===== TEP TABLE3 10/5 TRAINING COMPLETE $(date '+%F %T') =====" | tee -a "$LOG"
