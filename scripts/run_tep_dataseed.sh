#!/bin/bash
# TEP Type-Gen Phase 2 — DATA-SEED axis runner (2026-07-24).
# New experiment axis: the DATASET ALLOCATION itself is re-drawn per seed
# (scripts/TEP/build_tep_data.py --data-seed N -> scripts/TEP/data_dataseed{N}/,
# run IDs sampled without replacement; sizes/folds/onset/ordering canonical).
# Canonical seed 42 = the existing TEP_phase2_win100_ep30 run (data + train seed 42).
#
# Per data-seed N in {40,41,43,44}: Phase2-A + Phase2-B (B0 excluded; D derives from
# A's teacher_recon via pak_fill) x 4 folds, with BOTH the dataset (TEP_TYPEGEN_DATA_DIR)
# and the training seed (random_seed=N) set to N, into
# results/experiments/TEP_phase2_win100_ep30_dataseed{N}. After both groups:
# pak_fill.py --root <root> + build_table4.py --dataseed N (->
# results/experiments/TEP_phase2_win100_ep30/table4_data_ds{N}.json).
# NOTE pak_fill/build_table4 need no data override: every dataseed test stream is
# layout-identical to canonical (same y/run_table — verified at build time).
#
# WAIT GATE: blocks until the running baseline queue (catch retries via
# run_baseline.py + nrdetector refix) fully exits — do NOT remove; this script
# runs GPU jobs and must not overlap the queue.
# Resumable: run_base_experiments.py skips datasets whose results dir already has
# experiment_metadata.json — re-invoking resumes where it stopped.
#
# Usage:
#   bash scripts/run_tep_dataseed.sh 40          # one data-seed
#   bash scripts/run_tep_dataseed.sh             # all: 40 41 43 44 (sequential)
# nohup:
#   nohup bash scripts/run_tep_dataseed.sh    > /tmp/tep_dataseed_all.log 2>&1 &
#   nohup bash scripts/run_tep_dataseed.sh 40 > /tmp/tep_dataseed_s40.log 2>&1 &
set -u
cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
RB="scripts/run_base_experiments.py"
BASE_ROOT=results/experiments/TEP_phase2_win100_ep30
DSA="TEP_typegen_fstep TEP_typegen_frand TEP_typegen_fds TEP_typegen_funk"

# ---- wait gate: baseline queue (catch retry) + nrdetector refix must finish ----
echo "[$(date '+%F %T')] wait gate: checking run_baseline.py / run_nrdetector_refix ..."
while pgrep -f "run_baseline.py|run_nrdetector_refix" >/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] wait gate clear — starting data-seed axis"

run_group() {  # $1=label $2=root $3=outsub $4=datasets $5=extra_override $6=log $7=ov $8=datadir
  echo "########## [$(date '+%F %T')] START $1 ##########" | tee -a "$6"
  TEP_TYPEGEN_DATA_DIR="$8" $PY "$RB" --set A --dataset $4 --output-base "$2/$3" \
      --config-override $7 $5 >> "$6" 2>&1
  local rc=$?
  echo "########## [$(date '+%F %T')] END $1 (exit $rc) ##########" | tee -a "$6"
  return $rc
}

run_dataseed() {  # $1=data-seed
  local seed=$1
  local ROOT="${BASE_ROOT}_dataseed${seed}"
  local DATADIR="$PWD/scripts/TEP/data_dataseed${seed}"
  local LOG=/tmp/tep_phase2_dataseed${seed}.log
  local OV="official=True batch_size=768 official_keep_checkpoints=False random_seed=${seed} num_epochs=30 teacher_only_warmup_epochs=15 seq_length=100 patch_size=5 num_patches=20"
  if [ "$seed" = "42" ]; then
    echo "REFUSE: data-seed 42 = canonical dataset/run ($BASE_ROOT) — not re-running." | tee -a "$LOG"
    return 1
  fi
  if [ ! -f "$DATADIR/test_stream.npz" ]; then
    echo "MISSING: $DATADIR — build first: $PY scripts/TEP/build_tep_data.py --data-seed $seed" | tee -a "$LOG"
    return 1
  fi
  echo "===== DATA-SEED $seed START $(date '+%F %T') (root: $ROOT, data: $DATADIR) =====" | tee -a "$LOG"
  run_group "ds${seed}-Phase2-A" "$ROOT" phase2_A "$DSA" ""                        "$LOG" "$OV" "$DATADIR"
  run_group "ds${seed}-Phase2-B" "$ROOT" phase2_B "$DSA" "blind_train_labels=True" "$LOG" "$OV" "$DATADIR"
  echo "----- [$(date '+%F %T')] aggregate: pak_fill --root $ROOT -----" | tee -a "$LOG"
  $PY scripts/TEP/pak_fill.py --root "$ROOT" >> "$LOG" 2>&1
  echo "----- [$(date '+%F %T')] aggregate: build_table4 --dataseed $seed -----" | tee -a "$LOG"
  $PY scripts/TEP/build_table4.py --dataseed "$seed" >> "$LOG" 2>&1
  echo "===== DATA-SEED $seed DONE $(date '+%F %T') =====" | tee -a "$LOG"
}

if [ $# -ge 1 ]; then
  SEEDS="$@"
else
  SEEDS="40 41 43 44"
fi
for s in $SEEDS; do
  run_dataseed "$s"
done
echo "ALL DATA-SEEDS DONE ($SEEDS)"
