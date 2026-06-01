#!/bin/bash
# Rerun AT and OmniAnomaly on WaDi A1/A2 for all 4 queues
# Fix: train_stride=5 in wadi_14days preset (reduces numpy array from 56 GiB to 11 GiB)

PYTHON=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py
MODELS="anomaly_transformer omnianomaly"

# Queue configs: output_base, normalize_mode, experiment_A1, experiment_A2
declare -A QUEUES
QUEUES[Q1_base]="/home/ykio/notebooks/claude/comparison/results/experiments/1_20260312_041500_baseline_minmax"
QUEUES[Q1_norm]="minmax"
QUEUES[Q1_expA1]="wadi_14days_A1"
QUEUES[Q1_expA2]="wadi_14days_A2"

QUEUES[Q2_base]="/home/ykio/notebooks/claude/comparison/results/experiments/2_20260312_144923_baseline_zscore"
QUEUES[Q2_norm]="zscore"
QUEUES[Q2_expA1]="wadi_14days_A1"
QUEUES[Q2_expA2]="wadi_14days_A2"

QUEUES[Q3_base]="/home/ykio/notebooks/claude/comparison/results/experiments/3_20260312_203923_baseline_minmax_normalonly"
QUEUES[Q3_norm]="minmax"
QUEUES[Q3_expA1]="wadi_14days_A1_normalonly"
QUEUES[Q3_expA2]="wadi_14days_A2_normalonly"

QUEUES[Q4_base]="/home/ykio/notebooks/claude/comparison/results/experiments/4_20260313_020446_baseline_zscore_normalonly"
QUEUES[Q4_norm]="zscore"
QUEUES[Q4_expA1]="wadi_14days_A1_normalonly"
QUEUES[Q4_expA2]="wadi_14days_A2_normalonly"

TOTAL=0
SUCCESS=0
FAILED=0

echo "============================================"
echo "  WaDi AT/OmniAnomaly Re-experiment"
echo "  Start: $(date)"
echo "============================================"

for Q in Q1 Q2 Q3 Q4; do
    BASE="${QUEUES[${Q}_base]}"
    NORM="${QUEUES[${Q}_norm]}"
    EXPA1="${QUEUES[${Q}_expA1]}"
    EXPA2="${QUEUES[${Q}_expA2]}"

    echo ""
    echo ">>> $Q (normalize=$NORM)"

    for EXP_KEY in A1 A2; do
        if [ "$EXP_KEY" = "A1" ]; then
            EXP="$EXPA1"
        else
            EXP="$EXPA2"
        fi

        for MODEL in $MODELS; do
            TOTAL=$((TOTAL + 1))
            echo ""
            echo "--- [$TOTAL/16] $Q / WaDi $EXP_KEY / $MODEL ---"
            echo "  Experiment: $EXP"
            echo "  Output: $BASE"
            echo "  Normalize: $NORM"
            echo "  Start: $(date)"

            $PYTHON $RUNNER \
                --experiment "$EXP" \
                --model "$MODEL" \
                --output-base "$BASE" \
                --eval-interval 1 \
                --normalize-mode "$NORM" \
                --force

            if [ $? -eq 0 ]; then
                echo "  Result: SUCCESS"
                SUCCESS=$((SUCCESS + 1))
            else
                echo "  Result: FAILED (exit code $?)"
                FAILED=$((FAILED + 1))
            fi
            echo "  End: $(date)"
        done
    done
done

echo ""
echo "============================================"
echo "  Re-experiment Complete"
echo "  End: $(date)"
echo "  Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED"
echo "============================================"
