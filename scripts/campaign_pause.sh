#!/bin/bash
# 캠페인 일시중단 (복사-실행 가능). launcher를 PGID 번호로만 graceful 정지 → run_base·bg-worker 동반 정리.
# self-heal(graceful backstop + startup reaper, commit 7177186) 덕에 정상 정지 시 누수 0.
# 워처(자원모니터·results 자동재생성)는 유지한다. 중단시점은 완료 매트릭스가 ground truth.
set -u
cd /home/ykio/notebooks/TSMAE || exit 1
MYPGID=$(ps -o pgid= -p $$ | tr -d ' ')

echo "=== 중단 전 진행 ==="
ps -eo pid,etimes,args | awk -v m="$MYPGID" '/run_base_experiments\.py --set/ && /envs\/dc_vis/{n=split($0,a,"/271_");print "  실행중: 271_"substr(a[2],1,34)" (et="int($2/60)"분)"}'

# launcher PGID 확보 (내 PGID 제외 — self-match 방지). ⚠️ pkill -f / pgrep -f "문자열" 절대 사용 금지.
PGIDS=$(ps -eo pgid,args | awk -v m="$MYPGID" '$1!=m && /run_official_.*_after\.py/ && /envs\/dc_vis/ {print $1}' | sort -un)
echo "  정지 대상 launcher PGID: $(echo $PGIDS | tr '\n' ' ')"
for g in $PGIDS; do [ "$g" = "$MYPGID" ] && continue; kill -TERM -"$g" 2>/dev/null; done
sleep 6
for g in $PGIDS; do [ "$g" = "$MYPGID" ] && continue; if ps -eo pgid | grep -qx " *$g"; then kill -KILL -"$g" 2>/dev/null; fi; done
sleep 4

echo "=== 정지 + 누수 검증 (전부 PGID 제외) ==="
echo "  launcher: $(ps -eo pgid,args|awk -v m="$MYPGID" '$1!=m && /run_official_.*_after\.py/ && /envs\/dc_vis/'|wc -l)"
echo "  run_base: $(ps -eo pgid,args|awk -v m="$MYPGID" '$1!=m && /run_base_experiments\.py --set/ && /envs\/dc_vis/'|wc -l)"
echo "  고아: $(ps -eo ppid,cmd|awk '$1==1 && /[m]ultiprocessing.spawn/'|wc -l)  좀비: $(ps -eo stat|grep -c '^Z')  loky: $(ls /dev/shm/sem.loky-* 2>/dev/null|wc -l)"
echo "  GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null|head -1)  Mem avail: $(free -g|awk '/Mem/{print $7}')G"
echo "정지 완료. 재개는 campaign_resume.sh (사용자 '재개' 지시 후)."
