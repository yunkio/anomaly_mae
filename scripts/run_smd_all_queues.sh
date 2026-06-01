#!/bin/bash
# Run SMD baseline experiments for all queues (Q1, Q2, Q4)
# Q3 is already running separately

PYTHON=/home/ykio/anaconda3/envs/dc_vis/bin/python
RUNNER=comparison/run_baseline.py

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

# Queue definitions: name, dir, norm_mode, experiment_suffix
declare -A QUEUE_DIRS
QUEUE_DIRS[Q1]="/home/ykio/notebooks/claude/comparison/results/experiments/1_20260312_041500_baseline_minmax"
QUEUE_DIRS[Q2]="/home/ykio/notebooks/claude/comparison/results/experiments/2_20260312_144923_baseline_zscore"
QUEUE_DIRS[Q4]="/home/ykio/notebooks/claude/comparison/results/experiments/4_20260313_020446_baseline_zscore_normalonly"

declare -A QUEUE_NORM
QUEUE_NORM[Q1]="minmax"
QUEUE_NORM[Q2]="zscore"
QUEUE_NORM[Q4]="zscore"

# Q1/Q2: non-normalonly, Q4: normalonly
declare -A QUEUE_SUFFIX
QUEUE_SUFFIX[Q1]=""
QUEUE_SUFFIX[Q2]=""
QUEUE_SUFFIX[Q4]="_normalonly"

TOTAL=0
SUCCESS=0
FAILED=0

for QUEUE in Q1 Q2 Q4; do
    QDIR="${QUEUE_DIRS[$QUEUE]}"
    NORM="${QUEUE_NORM[$QUEUE]}"
    SUFFIX="${QUEUE_SUFFIX[$QUEUE]}"
    
    QUEUE_TOTAL=$((${#MACHINES[@]} * 2))
    QUEUE_CURRENT=0
    
    echo ""
    echo "============================================"
    echo "  $QUEUE: SMD Baseline (${#MACHINES[@]} machines × 2 parities)"
    echo "  Norm: $NORM, Suffix: ${SUFFIX:-none}"
    echo "  Output: $QDIR/SMD/"
    echo "  Start: $(date)"
    echo "============================================"
    
    for MACHINE in "${MACHINES[@]}"; do
        for PARITY in 0 1; do
            TOTAL=$((TOTAL + 1))
            QUEUE_CURRENT=$((QUEUE_CURRENT + 1))
            EXP_KEY="smd_${MACHINE}_p${PARITY}${SUFFIX}"
            
            echo ""
            echo "--- [$QUEUE $QUEUE_CURRENT/$QUEUE_TOTAL] $MACHINE/parity_$PARITY ---"
            echo "  Start: $(date)"
            
            $PYTHON $RUNNER \
                --experiment "$EXP_KEY" \
                --model all \
                --output-base "$QDIR" \
                --eval-interval 1 \
                --normalize-mode "$NORM" \
                --neural-epochs 10 \
                --sota-epochs 10 \
                --at-epochs 10 \
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
    
    echo ""
    echo "  $QUEUE complete: $QUEUE_CURRENT runs"
done

echo ""
echo "============================================"
echo "  All Queues Complete (Q1+Q2+Q4)"
echo "  End: $(date)"
echo "  Total: $TOTAL, Success: $SUCCESS, Failed: $FAILED"
echo "============================================"
