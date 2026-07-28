#!/usr/bin/env bash
# NRDetector encoder-LR sensitivity: encoder_lr=1e-5, official noisy_rate=0.4.
# Scope: 5 seeds x 4 paper entities = 20 independent 50-epoch runs.
# nrdetector_full is intentionally excluded because it is a non-paper ablation.
set -euo pipefail

cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py
OUTROOT=comparison/results/experiments/nrdetector_encoder_lr_1e-5_5seed
LOG=temp/baseline_experiment_run/nrdetector_encoder_lr_1e-5_5seed.log
STAMP=$(date '+%Y%m%d_%H%M%S')
QUARANTINE="temp/nrdetector_encoder_lr_1e-5_partial_quarantine/$STAMP"
SEEDS=(42 43 40 41 44)
EXPERIMENTS=(psm swat_a1a2 wadi_14days_A1 wadi_14days_A2)

mkdir -p "$OUTROOT" "$(dirname "$LOG")" "$QUARANTINE"

result_dir() {
  local seed=$1 exp=$2
  local sub
  case "$exp" in
    psm) sub=PSM ;;
    swat_a1a2) sub=SWaT/A1A2_full ;;
    wadi_14days_A1) sub=WaDi/A1 ;;
    wadi_14days_A2) sub=WaDi/A2 ;;
    *) echo "unknown experiment: $exp" >&2; return 1 ;;
  esac
  printf '%s/seed%s/%s/nrdetector\n' "$OUTROOT" "$seed" "$sub"
}

is_complete() {
  local dir=$1 seed=$2
  "$PY" - "$dir" "$seed" <<'PY'
import json
import math
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
try:
    payload = json.loads((d / "epoch_metrics.json").read_text())
    rows = payload.get("epochs", payload) if isinstance(payload, dict) else payload
    epochs = [int(row["epoch"]) for row in rows]
    scores = sorted((d / "epoch_scores").glob("epoch_*_scores.npz"))
    config = json.loads((d / "model" / "config.json").read_text())
    meta = json.loads((d / "metadata.json").read_text())
    attrs = meta["parameters"]["all_model_attributes"]
    ok = (
        epochs == list(range(1, 51))
        and len(scores) == 50
        and scores[-1].name == "epoch_050_scores.npz"
        and (d / "scores.npz").is_file()
        and math.isclose(float(config["encoder_lr"]), 1e-5, rel_tol=0, abs_tol=1e-12)
        and math.isclose(float(attrs["encoder_lr"]), 1e-5, rel_tol=0, abs_tol=1e-12)
        and int(meta["parameters"]["seed"]) == seed
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
    "$PWD"/comparison/results/experiments/nrdetector_encoder_lr_1e-5_5seed/*) ;;
    *) echo "REFUSE unsafe quarantine target: $resolved" | tee -a "$LOG"; return 1 ;;
  esac
  local rel=${resolved#"$PWD"/}
  local dst="$QUARANTINE/$rel"
  mkdir -p "$(dirname "$dst")"
  mv -- "$resolved" "$dst"
  echo "[$(date '+%F %T')] quarantined partial: $rel" | tee -a "$LOG"
}

echo "===== NRDETECTOR encoder_lr=1e-5 5-SEED START $(date '+%F %T') =====" | tee -a "$LOG"
for seed in "${SEEDS[@]}"; do
  for exp in "${EXPERIMENTS[@]}"; do
    outbase="$OUTROOT/seed${seed}"
    dir=$(result_dir "$seed" "$exp")
    if is_complete "$dir" "$seed"; then
      echo "[$(date '+%F %T')] SKIP complete: seed=$seed exp=$exp" | tee -a "$LOG"
      continue
    fi
    quarantine_partial "$dir"
    echo "[$(date '+%F %T')] START: seed=$seed exp=$exp" | tee -a "$LOG"
    "$PY" "$RUNNER" \
      --experiment "$exp" --model nrdetector \
      --output-base "$outbase" \
      --sota-epochs 50 --eval-interval 1 --normalize-mode minmax \
      --seed "$seed" --early-stop \
      --nrdetector-encoder-lr 1e-5 \
      >> "$LOG" 2>&1
    if ! is_complete "$dir" "$seed"; then
      echo "[$(date '+%F %T')] ERROR incomplete after run: $dir" | tee -a "$LOG"
      exit 1
    fi
    echo "[$(date '+%F %T')] DONE: seed=$seed exp=$exp" | tee -a "$LOG"
  done
done

"$PY" comparison/build_nrdetector_encoder_lr_summary.py >> "$LOG" 2>&1
echo "===== NRDETECTOR encoder_lr=1e-5 5-SEED COMPLETE $(date '+%F %T') =====" | tee -a "$LOG"
