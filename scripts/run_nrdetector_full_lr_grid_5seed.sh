#!/usr/bin/env bash
# NRDetector-full LR grid. The two arms share classifier_lr=1e-5 and differ
# only in encoder_lr. Each arm has its own explicit result directory.
set -euo pipefail

cd /home/ykio/notebooks/claude
PY=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py
OUTROOT=comparison/results/experiments/nrdetector_full_lr_grid_5seed
MASTER_LOG=temp/baseline_experiment_run/nrdetector_full_lr_grid_5seed.log
CLASSIFIER_LR=1e-5
SEEDS=(42 43 40 41 44)
EXPERIMENTS=(psm swat_a1a2 wadi_14days_A1 wadi_14days_A2)
ENCODER_LRS=(1e-4 1e-5)
STAMP=$(date '+%Y%m%d_%H%M%S')
QUARANTINE="temp/nrdetector_full_lr_grid_partial_quarantine/$STAMP"

mkdir -p "$OUTROOT" "$(dirname "$MASTER_LOG")" "$QUARANTINE"

arm_name() {
  case "$1" in
    1e-4) printf 'encoder_lr_1e-4__classifier_lr_1e-5\n' ;;
    1e-5) printf 'encoder_lr_1e-5__classifier_lr_1e-5\n' ;;
    *) echo "unsupported encoder LR: $1" >&2; return 1 ;;
  esac
}

result_dir() {
  local encoder_lr=$1 seed=$2 exp=$3 sub arm
  arm=$(arm_name "$encoder_lr")
  case "$exp" in
    psm) sub=PSM ;;
    swat_a1a2) sub=SWaT/A1A2_full ;;
    wadi_14days_A1) sub=WaDi/A1 ;;
    wadi_14days_A2) sub=WaDi/A2 ;;
    *) echo "unknown experiment: $exp" >&2; return 1 ;;
  esac
  printf '%s/%s/seed%s/%s/nrdetector_full\n' "$OUTROOT" "$arm" "$seed" "$sub"
}

is_complete() {
  local dir=$1 seed=$2 encoder_lr=$3
  "$PY" - "$dir" "$seed" "$encoder_lr" "$CLASSIFIER_LR" <<'PY'
import json
import math
import pathlib
import re
import sys

d = pathlib.Path(sys.argv[1])
seed = int(sys.argv[2])
encoder_lr = float(sys.argv[3])
classifier_lr = float(sys.argv[4])
try:
    payload = json.loads((d / "epoch_metrics.json").read_text(encoding="utf-8"))
    rows = payload.get("epochs", payload) if isinstance(payload, dict) else payload
    epochs = [int(row["epoch"]) for row in rows]
    score_paths = sorted((d / "epoch_scores").glob("epoch_*_scores.npz"))
    score_epochs = sorted(
        int(re.fullmatch(r"epoch_(\d+)_scores\.npz", p.name).group(1))
        for p in score_paths
    )
    config = json.loads((d / "model" / "config.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    attrs = meta["parameters"]["all_model_attributes"]
    overrides = meta["parameters"]["epoch_overrides"]
    close = lambda a, b: math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)
    ok = (
        epochs == list(range(1, 51))
        and score_epochs == list(range(1, 51))
        and (d / "scores.npz").is_file()
        and meta["model_name"] == "nrdetector_full"
        and int(meta["parameters"]["seed"]) == seed
        and close(config["encoder_lr"], encoder_lr)
        and close(config["noisy_rate"], 1.0)
        and close(attrs["encoder_lr"], encoder_lr)
        and close(attrs["lr"], classifier_lr)
        and close(attrs["noisy_rate"], 1.0)
        and close(overrides["nrdetector_encoder_lr"], encoder_lr)
        and close(overrides["nrdetector_classifier_lr"], classifier_lr)
    )
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

quarantine_partial() {
  local dir=$1
  [ -e "$dir" ] || return 0
  local resolved rel dst
  resolved=$(realpath -m "$dir")
  case "$resolved" in
    "$PWD"/comparison/results/experiments/nrdetector_full_lr_grid_5seed/*) ;;
    *) echo "REFUSE unsafe quarantine target: $resolved" | tee -a "$MASTER_LOG"; return 1 ;;
  esac
  rel=${resolved#"$PWD"/}
  dst="$QUARANTINE/$rel"
  mkdir -p "$(dirname "$dst")"
  mv -- "$resolved" "$dst"
  echo "[$(date '+%F %T')] quarantined partial: $rel" | tee -a "$MASTER_LOG"
}

"$PY" - "$OUTROOT" <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
payload = {
    "experiment": "nrdetector_full encoder/classifier LR grid",
    "model": "nrdetector_full",
    "label_semantics": "noisy_rate=1.0; all positive windows revealed, negatives remain unlabeled (PU)",
    "encoder_lr_arms": [1e-4, 1e-5],
    "classifier_lr": 1e-5,
    "epochs": 50,
    "selection": "NO_ES fixed final epoch 50; best-PAK is diagnostic only",
    "seeds": [42, 43, 40, 41, 44],
    "experiments": ["psm", "swat_a1a2", "wadi_14days_A1", "wadi_14days_A2"],
    "result_arms": {
        "encoder_1e-4_classifier_1e-5": "encoder_lr_1e-4__classifier_lr_1e-5",
        "encoder_1e-5_classifier_1e-5": "encoder_lr_1e-5__classifier_lr_1e-5",
    },
}
path = root / "experiment_manifest.json"
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, path)
PY

echo "===== NRDETECTOR_FULL LR GRID START $(date '+%F %T') =====" | tee -a "$MASTER_LOG"
for encoder_lr in "${ENCODER_LRS[@]}"; do
  arm=$(arm_name "$encoder_lr")
  arm_log="temp/baseline_experiment_run/nrdetector_full_${arm}_5seed.log"
  echo "[$(date '+%F %T')] ARM START: model=nrdetector_full encoder_lr=$encoder_lr classifier_lr=$CLASSIFIER_LR root=$OUTROOT/$arm" | tee -a "$MASTER_LOG" "$arm_log"
  for seed in "${SEEDS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
      outbase="$OUTROOT/$arm/seed${seed}"
      dir=$(result_dir "$encoder_lr" "$seed" "$exp")
      if is_complete "$dir" "$seed" "$encoder_lr"; then
        echo "[$(date '+%F %T')] SKIP complete: encoder_lr=$encoder_lr seed=$seed exp=$exp" | tee -a "$MASTER_LOG" "$arm_log"
        continue
      fi
      quarantine_partial "$dir"
      echo "[$(date '+%F %T')] START: encoder_lr=$encoder_lr classifier_lr=$CLASSIFIER_LR seed=$seed exp=$exp" | tee -a "$MASTER_LOG" "$arm_log"
      "$PY" "$RUNNER" \
        --experiment "$exp" --model nrdetector_full \
        --output-base "$outbase" \
        --sota-epochs 50 --eval-interval 1 --normalize-mode minmax \
        --seed "$seed" --early-stop \
        --nrdetector-encoder-lr "$encoder_lr" \
        --nrdetector-classifier-lr "$CLASSIFIER_LR" \
        >> "$arm_log" 2>&1
      if ! is_complete "$dir" "$seed" "$encoder_lr"; then
        echo "[$(date '+%F %T')] ERROR incomplete after run: $dir" | tee -a "$MASTER_LOG" "$arm_log"
        exit 1
      fi
      echo "[$(date '+%F %T')] DONE: encoder_lr=$encoder_lr seed=$seed exp=$exp" | tee -a "$MASTER_LOG" "$arm_log"
    done
  done
  echo "[$(date '+%F %T')] ARM COMPLETE: encoder_lr=$encoder_lr classifier_lr=$CLASSIFIER_LR" | tee -a "$MASTER_LOG" "$arm_log"
done

"$PY" comparison/build_nrdetector_full_lr_grid_summary.py >> "$MASTER_LOG" 2>&1
echo "===== NRDETECTOR_FULL LR GRID COMPLETE $(date '+%F %T') =====" | tee -a "$MASTER_LOG"
