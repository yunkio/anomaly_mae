#!/usr/bin/env bash
# =============================================================================
# Reseed chain — 8/9 baseline experiments repeated 5× with fixed seeds + real ES
#   Order (interleaved 8 9 8 9): 8-1,9-1,8-2,9-2,8-3,9-3,8-4,9-4,8-5,9-5
#   Seeds:   8-k & 9-k share SEEDS[k] -> 42,43,40,41,44
#            (2026-07-11 조정: 최종 seed 세트 40~44. 42(k1)/43(k2) 유지, 이후 40(k3)/41(k4)/44(k5))
#   Each run: run_baseline_queue.py --seed <S> --early-stop  (train_loss patience-3,
#             restore-best=min-train_loss; 5 no-train_loss models run full, no ES).
#   Resumable: the queue runner auto-skips entries that already have epoch_metrics.json,
#             so re-launching this script continues where it stopped.
# =============================================================================
set -u
cd /home/ykio/notebooks/claude || exit 1
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python

Q8=comparison/configs/baseline_queue_reseed8_normalonly.json
Q9=comparison/configs/baseline_queue_reseed9_weak.json
SUFFIX=20260606_175756          # keep original timestamp suffix per user's dir scheme
EXPROOT=comparison/results/experiments
RUNDIR=temp/baseline_experiment_run
mkdir -p "$RUNDIR"
TS=$(date +%Y%m%d_%H%M%S)
MASTER="$RUNDIR/reseed_chain_${TS}.log"
STATEF="$RUNDIR/reseed_chain_current.txt"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

for f in "$Q8" "$Q9"; do
  [ -f "$f" ] || { log "FATAL: queue file missing: $f"; exit 1; }
done

log "=== RESEED CHAIN START (pid $$) ==="
log "Q8=$Q8 (792 unsup entries)  Q9=$Q9 (180 weak entries)"
log "Order: 8-1,9-1,8-2,9-2,8-3,9-3,8-4,9-4,8-5,9-5 | seeds 42,43,40,41,44 | --early-stop"

run_queue() {           # $1=queue  $2=output-base  $3=seed  $4=tag
  local queue="$1" out="$2" seed="$3" tag="$4"
  local rlog="$RUNDIR/reseed_${tag}_${TS}.log"
  echo "$tag seed=$seed -> $out (started $(date +%H:%M:%S))" > "$STATEF"
  log ">>> RUN $tag  seed=$seed  out=$out"
  log "    log: $rlog"
  "$PY" comparison/run_baseline_queue.py \
      --queue "$queue" \
      --output-base "$out" \
      --seed "$seed" --early-stop \
      >> "$rlog" 2>&1
  local rc=$?
  log "<<< DONE $tag (exit $rc)"
  return $rc
}

SEEDS=(42 43 40 41 44)   # 2026-07-11: 40~44 세트. 42(8-1/9-1)·43(8-2/9-2) 유지, 이후 40,41,44
# 2026-07-14: 8-1/9-1(seed42)·8-2/9-2(seed43) 완료 → k=3부터 재개(seed40).
#   plain resume이 8-1의 실패 entry(catch×WaDi_A1, epoch_metrics 없음)를 seed42로 ~8h 재실행하는
#   낭비를 피하고 "8-3부터 seed40" 의도를 정확히 구현. 8-3 dir은 seed44로 잘못 시작된 것을 .trash 백업 후 제거함.
for k in 3 4 5; do
  seed=${SEEDS[$((k-1))]}
  run_queue "$Q8" "$EXPROOT/8-${k}_${SUFFIX}_baseline"   "$seed" "8-${k}"
  run_queue "$Q9" "$EXPROOT/9-${k}_${SUFFIX}_weak_ssl"   "$seed" "9-${k}"
done

log "=== RESEED CHAIN COMPLETE ==="
echo "ALL DONE $(date +%H:%M:%S)" > "$STATEF"
