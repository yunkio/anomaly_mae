#!/bin/bash
# Telemetry collector — 10s polling, GPU + system metrics
# Usage: collect_telemetry.sh OUTPUT_CSV
# Writes header on first run; appends if file exists.

CSV="${1:?Usage: $0 OUTPUT_CSV}"

if [ ! -s "$CSV" ]; then
  echo "iso,unix,gpu_mem_used,gpu_mem_free,gpu_mem_total,gpu_util,gpu_util_mem,gpu_power,gpu_power_limit,gpu_temp,gpu_clock_sm,gpu_clock_mem,gpu_pstate,ram_used,ram_free,ram_buffcache,ram_available,ram_total,swap_used,swap_total,load_1m,load_5m,load_15m,procs_r,procs_b,vm_si,vm_so,io_bi,io_bo,sys_in,sys_cs,cpu_us,cpu_sy,cpu_id,cpu_wa,cpu_st" > "$CSV"
fi

while true; do
  ISO=$(date '+%Y-%m-%dT%H:%M:%S')
  UNIX=$(date +%s)

  # GPU (11 fields)
  GPU=$(nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem,pstate --format=csv,noheader,nounits | tr -d ' \n')

  # free (5 mem fields + 2 swap fields)
  MEM=$(free -m | awk '/^Mem:/ {printf "%d,%d,%d,%d,%d", $3, $4, $6, $7, $2}')
  SWAP=$(free -m | awk '/^Swap:/ {printf "%d,%d", $3, $2}')

  # load avg (3 fields)
  LOAD=$(cat /proc/loadavg | awk '{printf "%s,%s,%s", $1, $2, $3}')

  # vmstat (13 fields: r,b,si,so,bi,bo,in,cs,us,sy,id,wa,st) — blocks ~1s
  VM=$(vmstat 1 2 | tail -1 | awk '{printf "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", $1,$2,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17}')

  echo "$ISO,$UNIX,$GPU,$MEM,$SWAP,$LOAD,$VM" >> "$CSV"

  sleep 9  # vmstat blocks ~1s; total cycle ≈ 10s
done
