#!/bin/bash
# TEP simple-baseline runner — DATA-SEED axis (2026-07-24). CPU-ONLY.
# Per data-seed N in {40,41,43,44}: 5 simple models (random/sensor_range/pca_error/
# l2_norm/nn_distance) x 4 contaminated folds on the seeded streams
# (scripts/TEP/data_dataseed{N}) -> scripts/TEP/results/simple_dataseed{N}/.
# Per-fault metrics are computed (per_fault=True default in run_tep_simple) —
# required by build_table4 --dataseed (fold-matched macro S/U + Random per-seed
# raw values from per_fault_by_seed.json[N]).
#
# CPU-only, so it CAN run alongside GPU jobs, but scheduling is the
# orchestrator's call (baseline queue load) — do not auto-start.
# Resumable: a fold x model whose per_fault_metrics.json already exists is skipped.
#
# Usage:
#   bash scripts/run_tep_simple_dataseed.sh 40        # one data-seed
#   bash scripts/run_tep_simple_dataseed.sh           # all: 40 41 43 44 (sequential)
# nohup:
#   nohup bash scripts/run_tep_simple_dataseed.sh > /tmp/tep_simple_dataseed.log 2>&1 &
# After finishing (and after the MAE data-seed runs), refresh the tables:
#   for s in 40 41 43 44; do
#     /home/ykio/anaconda3/envs/dc_vis/bin/python scripts/TEP/build_table4.py --dataseed $s
#   done
set -u
cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
MODELS="random sensor_range pca_error l2_norm nn_distance"
FOLDS="f_step f_rand f_ds f_unk"

run_dataseed() {  # $1=data-seed
  local seed=$1
  local DATADIR="scripts/TEP/data_dataseed${seed}"
  local OUTDIR="scripts/TEP/results/simple_dataseed${seed}"
  local LOG=/tmp/tep_simple_dataseed${seed}.log
  if [ "$seed" = "42" ]; then
    echo "REFUSE: data-seed 42 = canonical dataset — canonical simple results already exist" | tee -a "$LOG"
    return 1
  fi
  if [ ! -f "$DATADIR/test_stream.npz" ]; then
    echo "MISSING: $DATADIR — build first: $PY scripts/TEP/build_tep_data.py --data-seed $seed" | tee -a "$LOG"
    return 1
  fi
  echo "===== SIMPLE DATA-SEED $seed START $(date '+%F %T') (data: $DATADIR -> $OUTDIR) =====" | tee -a "$LOG"
  for fold in $FOLDS; do
    for model in $MODELS; do
      if [ -f "$OUTDIR/$fold/$model/per_fault_metrics.json" ]; then
        echo "  [skip] $fold/$model (per_fault_metrics.json exists)" | tee -a "$LOG"
        continue
      fi
      echo "----- [$(date '+%F %T')] ds${seed} $fold / $model -----" | tee -a "$LOG"
      nice -n 10 $PY scripts/TEP/run_tep_simple.py --fold "$fold" --models "$model" \
          --data-dir "$DATADIR" --results-dir "$OUTDIR" >> "$LOG" 2>&1
      echo "  exit $? ($fold/$model)" | tee -a "$LOG"
    done
  done
  echo "===== SIMPLE DATA-SEED $seed DONE $(date '+%F %T') =====" | tee -a "$LOG"
}

if [ $# -ge 1 ]; then
  SEEDS="$@"
else
  SEEDS="40 41 43 44"
fi
for s in $SEEDS; do
  run_dataseed "$s"
done
echo "ALL SIMPLE DATA-SEEDS DONE ($SEEDS)"
