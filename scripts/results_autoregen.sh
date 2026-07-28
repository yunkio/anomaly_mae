#!/bin/bash
# results_lasad.md 자동 재생성. 신호 = 생성기와 동일한 "완료 run dir 개수"(5셀 각 epoch_metrics+best_config+
# training_histories 존재). 로그 rotate/launcher 재기동 무관. 마지막 파일 기록 레이스에도 카운트가
# 늦게 오르므로 최종 상태가 반드시 반영된다.
source /home/ykio/anaconda3/etc/profile.d/conda.sh; conda activate dc_vis
cd /home/ykio/notebooks/TSMAE
count_complete() {
  local n=0 d ok c f
  for d in results/experiments/official/271_*_30ep_*; do
    [ -d "$d" ] || continue
    ok=1
    for c in PSM SWaT/A1A2_full SWaT/A1A2_excl22 WaDi/A1 WaDi/A2; do
      for f in epoch_metrics.json best_config.json training_histories.json; do
        [ -f "$d/$c/$f" ] || { ok=0; break 2; }
      done
    done
    [ "$ok" -eq 1 ] && n=$((n+1))
  done
  echo "$n"
}
last=-1
while true; do
  cur=$(count_complete)
  if [ "$cur" != "$last" ]; then
    python scripts/generate_results_md.py >> /tmp/results_autoregen.log 2>&1 \
      && echo "$(date '+%F %T') regen (complete_dirs $last->$cur)" >> /tmp/results_autoregen.log
    last=$cur
  fi
  sleep 120
done
