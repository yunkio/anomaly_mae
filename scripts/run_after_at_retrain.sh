#!/bin/bash
# Wait for AT retrain to finish, then run Q3/Q4 neural re-experiments

echo "Waiting for AT retrain (PID checking retrain_at_all.sh)..."

# Wait for retrain_at_all.sh to finish
while pgrep -f "retrain_at_all.sh" > /dev/null 2>&1; do
    sleep 60
done

echo "AT retrain finished at $(date). Starting Q3/Q4 neural re-run..."
bash scripts/rerun_q3q4_neural.sh
