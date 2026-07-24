#!/bin/bash
# 캠페인 재개 (복사-실행 가능). 사용자가 "재개" 지시한 뒤에만 실행.
# 하는 일: (1) 이미 실행중이면 ABORT, (2) dc_vis torch 확인, (3) 미완(중단/실패) run dir 삭제,
#          (4) 워처(자원모니터·results 자동재생성) 죽었으면 재기동, (5) 3 launcher를 dc_vis로 기동,
#          (6) 40초 후 검증. Monitor 재장전은 호출자(assistant)가 별도로 한다.
set -u
cd /home/ykio/notebooks/TSMAE || exit 1
DCPY=/home/ykio/anaconda3/envs/dc_vis/bin/python
MYPGID=$(ps -o pgid= -p $$ | tr -d ' ')
active() { ps -eo pgid,args | awk -v m="$MYPGID" -v re="$1" '$1!=m && /envs\/dc_vis/ && $0 ~ re'; }

# (1) 이미 실행중이면 중단
if [ -n "$(active 'run_base_experiments\.py --set')" ] || [ -n "$(active 'run_official_.*_after\.py')" ]; then
  echo "ABORT: 캠페인이 이미 실행 중입니다. 먼저 campaign_pause.sh 로 정지하세요."; exit 1
fi
# (2) dc_vis torch/cuda
$DCPY -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null \
  || { echo "ABORT: dc_vis torch/cuda 사용불가 — 환경 확인"; exit 1; }

# (3) 미완 campaign run dir 삭제 (중단/실패분 → 재실행되도록). 완료(생성기 기준 3파일)만 남긴다.
$DCPY - <<'PY'
import glob, os, shutil
OFF="results/experiments/official"
CELLS=["PSM","SWaT/A1A2_full","WaDi/A1","WaDi/A2"]
FILES=("epoch_metrics.json","best_config.json","training_histories.json")
TAGS=[""]+["exclanom","blind","nogrl","nofm","noforce","nostudent","td3sd3",
           "unlab10r","unlab25r","unlab50r","unlab75r",
           "maskr005","maskr010","maskr030","maskr050","maskr060","maskr075","maskr090"]
seeds=[40,41,42,43,44]
keep=set()
for s in seeds:
    for t in TAGS:
        pat=f"{OFF}/271_*_30ep_{s}" + (f"_{t}" if t else "")
        for d in glob.glob(pat):
            b=os.path.basename(d)
            # exact seed_tag match (no-tag: name ends with _<seed>)
            if t=="" and not b.endswith(f"_30ep_{s}"): continue
            comp=all(os.path.exists(os.path.join(d,c,f)) for c in CELLS for f in FILES)
            if comp: keep.add(d)
# incomplete campaign dirs 삭제 (완료 dir·비-campaign dir은 건드리지 않음)
removed=0
for s in seeds:
    for t in TAGS:
        pat=f"{OFF}/271_*_30ep_{s}" + (f"_{t}" if t else "")
        for d in glob.glob(pat):
            b=os.path.basename(d)
            if t=="" and not b.endswith(f"_30ep_{s}"): continue
            if d not in keep:
                print("  삭제(incomplete):", b); shutil.rmtree(d, ignore_errors=True); removed+=1
print(f"  미완 dir {removed}개 삭제")
PY

# (4) 워처 재기동 (죽었을 때만; PGID 제외로 판정)
mon=$(ps -eo pid,pgid,args | awk -v m="$MYPGID" '$2!=m && /resource_monitor\.sh/ && $0 !~ /awk/' | wc -l)
wat=$(ps -eo pid,pgid,args | awk -v m="$MYPGID" '$2!=m && /results_autoregen\.sh/ && $0 !~ /awk/' | wc -l)
[ "$mon" -eq 0 ] && { setsid nohup bash scripts/resource_monitor.sh   </dev/null >/dev/null 2>&1 & echo "  자원모니터 재기동"; } || echo "  자원모니터 유지"
[ "$wat" -eq 0 ] && { setsid nohup bash scripts/results_autoregen.sh  </dev/null >/dev/null 2>&1 & echo "  results워처 재기동"; } || echo "  results워처 유지"

# (5) 3 launcher 기동 (반드시 dc_vis 명시경로)
setsid nohup $DCPY scripts/run_official_sens3seed_after.py       >> /tmp/official_sens3seed.log      2>&1 </dev/null &
setsid nohup $DCPY scripts/run_official_paper5seed_after.py      >> /tmp/official_paper5seed.log     2>&1 </dev/null &
setsid nohup env PYTHONHASHSEED=42 $DCPY scripts/run_official_paper5seed_sens_after.py >> /tmp/official_paper5seed_sens.log 2>&1 </dev/null &
echo "  3 launcher 기동 — 40초 후 검증..."
sleep 40

# (6) 검증 (전부 PGID 제외)
echo "=== 검증 ==="
echo "  launcher: $(ps -eo pgid,args|awk -v m="$MYPGID" '$1!=m && /run_official_.*_after\.py/ && /envs\/dc_vis/'|wc -l)/3"
echo "  run_base: $(ps -eo pgid,args|awk -v m="$MYPGID" '$1!=m && /run_base_experiments\.py --set/ && /envs\/dc_vis/'|wc -l)/1"
echo "  torch crash(재기동후): $(grep -ac 'No module named' /tmp/official_sens3seed.log 2>/dev/null)"
echo "  고아: $(ps -eo ppid,cmd|awk '$1==1 && /[m]ultiprocessing.spawn/'|wc -l)"
ps -eo pid,etimes,args|awk -v m="$MYPGID" '/run_base_experiments\.py --set/ && /envs\/dc_vis/{n=split($0,a,"/271_");print "  실행중: 271_"substr(a[2],1,34)}'
echo "재개 완료. Monitor 재장전은 assistant가 별도로 수행."
