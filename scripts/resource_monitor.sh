#!/bin/bash
# Persistent resource timeline — writes to PROJECT dir (survives reboot). Catches what actually
# grows before a slowdown: Linux RAM/swap/cache, python/bg-worker/orphan counts, loky sems, GPU.
LOG=/home/ykio/notebooks/TSMAE/tmp/resource_monitor.log
while true; do
  ts=$(date '+%F %T')
  read used avail cached < <(free -m | awk '/Mem/{print $3, $7, $6}')
  swap=$(free -m | awk '/Swap/{print $3}')
  load=$(cut -d' ' -f1 /proc/loadavg)
  py=$(ps -eo cmd | grep -c '[p]ython')
  bgw=$(ps -eo cmd | grep -cE '[m]ultiprocessing.spawn|_cpu_eval_viz')
  orphan=$(ps -eo ppid,cmd | awk '$1==1 && /[m]ultiprocessing.spawn/' | wc -l)
  loky=$(ls /dev/shm/sem.loky-* 2>/dev/null | wc -l)
  semmp=$(ls /dev/shm/sem.mp-* 2>/dev/null | wc -l)
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' MiB')
  rss=$(ps -eo rss,cmd | awk '/[p]ython/{s+=$1} END{print int(s/1024)}')
  vmmem=$(cat /proc/meminfo | awk '/MemTotal/{t=$2}/MemAvailable/{a=$2} END{print int((t-a)/1024)}')
  echo "$ts load=$load mem_used=${used}M avail=${avail}M cached=${cached}M swap=${swap}M linux_committed=${vmmem}M py=$py bgw=$bgw orphan=$orphan loky=$loky semmp=$semmp gpu=${gpu}MiB py_rss=${rss}M" >> "$LOG"
  sleep 120
done
