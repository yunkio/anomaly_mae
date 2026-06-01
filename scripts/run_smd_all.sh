#!/bin/bash
# Run SMD baseline experiments for all 4 queues (Q1-Q4)
# 28 machines × 15 models × 4 queues = 1,680 experiments
# Uses load_smd_simple() (train + front 50% test / back 50% test)

set -e

PYTHON=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py

Q1="/home/ykio/notebooks/claude/comparison/results/experiments/1_20260312_041500_baseline_minmax"
Q2="/home/ykio/notebooks/claude/comparison/results/experiments/2_20260312_144923_baseline_zscore"
Q3="/home/ykio/notebooks/claude/comparison/results/experiments/3_20260312_203923_baseline_minmax_normalonly"
Q4="/home/ykio/notebooks/claude/comparison/results/experiments/4_20260313_020446_baseline_zscore_normalonly"

MACHINES=(
    machine-1-1 machine-1-2 machine-1-3 machine-1-4
    machine-1-5 machine-1-6 machine-1-7 machine-1-8
    machine-2-1 machine-2-2 machine-2-3 machine-2-4
    machine-2-5 machine-2-6 machine-2-7 machine-2-8
    machine-2-9
    machine-3-1 machine-3-2 machine-3-3 machine-3-4
    machine-3-5 machine-3-6 machine-3-7 machine-3-8
    machine-3-9 machine-3-10 machine-3-11
)

run_queue() {
    local QDIR="$1"
    local QNAME="$2"
    local NORM="$3"
    local SUFFIX="$4"  # "" or "_normalonly"

    echo ""
    echo "============================================"
    echo "  $QNAME: SMD (${#MACHINES[@]} machines × 15 models)"
    echo "  Output: $QDIR/SMD/"
    echo "  Start: $(date)"
    echo "============================================"

    local TOTAL=$((${#MACHINES[@]} * 15))
    local CURRENT=0
    local SUCCESS=0
    local FAILED=0

    for machine in "${MACHINES[@]}"; do
        local EXP_KEY="smd_${machine}${SUFFIX}"

        echo ""
        echo "--- [$QNAME $((CURRENT/15 + 1))/${#MACHINES[@]}] $machine ---"
        echo "Experiment: $EXP_KEY"

        $PYTHON $RUNNER \
            --experiment "$EXP_KEY" \
            --model all \
            --output-base "$QDIR" \
            --eval-interval 1 \
            --normalize-mode "$NORM"

        local STATUS=$?
        CURRENT=$((CURRENT + 15))

        if [ $STATUS -eq 0 ]; then
            SUCCESS=$((SUCCESS + 15))
        else
            FAILED=$((FAILED + 15))
            echo "WARNING: $machine failed (exit=$STATUS)"
        fi

        echo "STATUS — $QNAME $machine done ($CURRENT/$TOTAL, success=$SUCCESS, failed=$FAILED)"
    done

    echo ""
    echo "============================================"
    echo "  $QNAME COMPLETE: $SUCCESS/$TOTAL succeeded, $FAILED failed"
    echo "  End: $(date)"
    echo "============================================"
}

echo "============================================"
echo "  SMD Baseline: All 4 Queues"
echo "  Total: ${#MACHINES[@]} machines × 15 models × 4 queues"
echo "  Start: $(date)"
echo "============================================"

# Q3 first (minmax_normalonly) — historically most important
run_queue "$Q3" "Q3" "minmax" "_normalonly"

# Then Q1, Q2, Q4
run_queue "$Q1" "Q1" "minmax" ""
run_queue "$Q2" "Q2" "zscore" ""
run_queue "$Q4" "Q4" "zscore" "_normalonly"

echo ""
echo "============================================"
echo "  ALL QUEUES COMPLETE"
echo "  End: $(date)"
echo "============================================"
