#!/usr/bin/env bash
# Autonomous master (2026-06-11, user granted full autonomy):
#   Phase 1: finish exp297 MSL recovery (5 entities) at d_model=768 — ONE MSL per
#            run_base process (fresh host memory each) + the new per-10-epoch gc
#            fix in trainer.py bounds the d_model=768 RAM bloat. Retry on OOM
#            (resume-from-checkpoint). Skips already-done MSL, resumes partials.
#   Phase 2: launch the rest of the queue (exp298 resume + 299-314) via v2c,
#            detached, and register its log so monitoring picks it up.
set -u
cd /home/ykio/notebooks/TSMAE
source /home/ykio/anaconda3/etc/profile.d/conda.sh && conda activate dc_vis
D=$(ls -d results/experiments/297_*dyn_dmodel)
OV=$(python3 -c "import json;o=[e['config_override'] for e in json.load(open('configs/queue_dedup_renumbered_v5.json'))['experiments'] if e['exp_num']==297][0];print(o.replace('batch_size=1024','batch_size=512'))")
echo "=== MASTER START $(date '+%F %T') | dir=$(basename $D) ==="

# ---- Phase 1: MSL recovery, per-MSL, retry-on-OOM ----
for m in C-1 C-2 F-7 P-11 T-13; do
  meta="$D/MSL/$m/experiment_metadata.json"
  for attempt in 1 2 3 4; do
    if [ -f "$meta" ]; then echo ">>> MSL/$m DONE"; break; fi
    echo ">>> MSL/$m attempt $attempt $(date '+%T') (single proc, gc-fix, resume if ckpt)"
    python -u scripts/run_base_experiments.py --set C --no-wait --output-base "$D" \
      --dataset "MSL_simple_$m" --config-override "$OV" >> "$D/MSL/_master_$m.log" 2>&1
    echo ">>> MSL/$m attempt $attempt rc=$? $(date '+%T')"
    sleep 3
  done
done
NMSL=$(find "$D/MSL" -name experiment_metadata.json 2>/dev/null | wc -l)
echo "=== Phase 1 done: MSL $NMSL/5 | exp297 total $(find "$D" -name experiment_metadata.json|wc -l)/37 $(date '+%T') ==="
if [ "$NMSL" -lt 5 ]; then echo "!!! MSL incomplete ($NMSL/5) — NOT launching queue, manual check needed"; exit 1; fi

# ---- Phase 2: resume exp298 + run 299-314 (detached) ----
TS=$(date +%Y%m%d_%H%M%S); QLOG=temp/phase1_logs/resume_dedup_v2c_$TS.log
echo "=== Phase 2: launch v2c (exp298 resume + 299-314) -> $QLOG $(date '+%T') ==="
setsid nohup python -u scripts/resume_dedup_v2c.py configs/queue_dedup_renumbered_v5.json > "$QLOG" 2>&1 &
sleep 8
echo "$QLOG" > /tmp/exp271_train_log.txt
QPID=$(ps -eo pid,cmd -ww | awk '/resume_dedup_v2c\.py/ && !/awk/ {print $1; exit}')
echo "$QPID" > /tmp/exp271_train_pid.txt
echo "=== MASTER DONE: queue launcher=$QPID log=$QLOG $(date '+%T') ==="
