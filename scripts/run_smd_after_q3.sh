#!/bin/bash
# Wait for Q3 SMD to finish, then run Q1/Q2/Q4

echo "Waiting for Q3 SMD (PID checking run_smd_q3.sh)..."
while pgrep -f "run_smd_q3.sh" > /dev/null 2>&1; do
    sleep 60
done

echo "Q3 SMD finished at $(date). Starting Q1/Q2/Q4..."
bash scripts/run_smd_all_queues.sh
