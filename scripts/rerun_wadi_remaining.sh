#!/bin/bash
# Run remaining AT/OmniAnomaly WaDi experiments (10 of 16)
# Q1 complete, Q2 A1 complete, remaining: Q2/A2 + Q3 + Q4

PYTHON=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py
MODELS="anomaly_transformer omnianomaly"

TOTAL=0
SUCCESS=0
FAILED=0

echo "============================================"
echo "  WaDi AT/OmniAnomaly Remaining Experiments"
echo "  Start: $(date)"
echo "============================================"

run_one() {
    local EXP="$1" MODEL="$2" BASE="$3" NORM="$4" LABEL="$5"
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "--- [$TOTAL/10] $LABEL ---"
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
}

# Q2: zscore — only A2 remaining
Q2="/home/ykio/notebooks/claude/comparison/results/experiments/2_20260312_144923_baseline_zscore"
for M in $MODELS; do
    run_one "wadi_14days_A2" "$M" "$Q2" "zscore" "Q2/A2/$M"
done

# Q3: minmax_normalonly — all 4
Q3="/home/ykio/notebooks/claude/comparison/results/experiments/3_20260312_203923_baseline_minmax_normalonly"
for DS in "wadi_14days_A1_normalonly" "wadi_14days_A2_normalonly"; do
    for M in $MODELS; do
        run_one "$DS" "$M" "$Q3" "minmax" "Q3/${DS}/$M"
    done
done

# Q4: zscore_normalonly — all 4
Q4="/home/ykio/notebooks/claude/comparison/results/experiments/4_20260313_020446_baseline_zscore_normalonly"
for DS in "wadi_14days_A1_normalonly" "wadi_14days_A2_normalonly"; do
    for M in $MODELS; do
        run_one "$DS" "$M" "$Q4" "zscore" "Q4/${DS}/$M"
    done
done

echo ""
echo "============================================"
echo "  Complete: $(date)"
echo "  Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED"
echo "============================================"
