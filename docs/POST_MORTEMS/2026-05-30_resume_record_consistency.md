# Post-mortem — 2026-05-30 — Resume record-consistency (score-contribution off-by-one + lost eval records)

## Summary

정지 후 resume 된 모든 실험에서 두 종류의 **기록 손상**이 발생했다. 둘 다 원인은
같다 — checkpoint 가 "그 epoch 의 모든 기록이 완성되기 *전*" 시점의 상태를 떠서
저장했기 때문이다.

1. **score-contribution off-by-one.** resume 후 완주한 run 의
   `training_histories.json` 에서 per-epoch contribution-ratio 배열
   (`epoch_recon_ratio_*`, `epoch_disc_ratio_*`, `epoch_anomaly_type_scores`, …)
   이 `epoch` 보다 정확히 1 짧았다 (예: `epoch=500` 인데 `epoch_recon_ratio_normal`
   길이 499). `plot_score_contribution_analysis` 의 `stackplot` 은 x/y 길이가
   같아야 하므로 `ValueError` → `_safe_plot` 가 swallow → `best_model_score_contribution.png`
   가 **생성 안 됨**.

2. **lost eval records.** resume 시 복원되는 `epoch_metrics_list` 가 pause 직전
   1–2 개의 eval 을 누락한 채로 저장돼 있었다. 최종 `epoch_metrics.json` 에 eval
   epoch 이 비는 구멍이 생겼다 (예: 271_lr SWaT-full `[285,290]` 누락, 271_lr
   WaDi/A1 `[275,280,350,355]` 누락).

## Impact

| 산출물 | 증상 |
| ---- | ---- |
| `training_histories.json` 의 contrib/timing 키 | resume run 에서 len == epoch-1 (off-by-one) |
| `visualization/best_model/best_model_score_contribution.png` | stackplot crash → 파일 누락 |
| `epoch_metrics.json` | pause 경계의 eval epoch 1–2 개 누락 (영구 손실) |
| `best_checkpoint.pt` 선정 | 누락된 eval 이 best 였을 경우 best_epoch 놓칠 수 있음 |

영향 받은 실험 (eval 누락 확인): `271_lr_20260529_225351_baseline/SWaT/A1A2_full`
(`[285,290]`), `271_lr_20260529_225351_baseline/WaDi/A1` (`[275,280,350,355]`),
진행 중이던 `274_lr_20260529_225351_balsamp/WaDi/A2` (`[40,45]` + off-by-one).
나머지 lr dataset 은 pause 를 한 번도 안 겪었거나 경계 운이 좋아 clean.

resume 을 한 번도 안 한 run 은 영향 없음.

## Root cause

두 버그 모두 **checkpoint 스냅샷 타이밍**이 원인이다.

### (1) off-by-one — 스냅샷이 contrib append 보다 앞섬

`Trainer.train()` 의 epoch loop 끝에서 기록 순서는:

```
L1155  history['epoch'].append(epoch+1)        # epoch 번호
L1156  history['train_loss'].append(...)        # 손실들
 ...
L1225  epoch_callback(epoch, model, history)    # ← (구) checkpoint 가 여기서 history 스냅샷
L1245  history['epoch_recon_ratio_*'].append(...) # contribution ratio (epoch 번호보다 늦게 append)
```

구 코드의 checkpoint 저장은 `epoch_callback`(eval 콜백) 안에서 일어났다. 이
시점은 `epoch` 은 이미 append 됐지만 contribution-ratio 들은 **아직** append
되기 전이다. 그래서 스냅샷된 history 는 `len(epoch)=N`, `len(contrib)=N-1`.
이 off-by-one 이 checkpoint 에 박제되고, resume → 완주 시 최종 history 까지
전파됐다.

### (2) lost eval records — eval 이 저장 *뒤* 비동기로 기록됨

per-epoch eval 은 GPU 를 막지 않으려고 background thread 에서 돈다. 구
`_run_bg_all` 순서는:

```
A. join prev_thread
B. drain queue → 이전 epoch 의 eval 결과를 epoch_metrics_list 에 기록
C. torch.save(checkpoint)            # ← epoch_metrics_list 는 여기서 "이전까지"만 반영
D. 이번 epoch 의 eval 실행 → 결과를 queue 에 put (다음 thread 가 drain)
```

즉 epoch N 의 eval 은 step D 에서 queue 에 들어가고, 그것을 epoch_metrics_list 에
반영하는 건 그 *다음* eval-epoch thread (N+EVAL_INTERVAL) 의 step B 다. checkpoint_N
은 step C 에서 이미 저장됐으므로 epoch N (그리고 직전 몇 개) 의 eval 을 담지
못한다. pause(kill) 가 그 사이에 떨어지면 queue 안의 결과는 영구 손실.

추가로, checkpoint build 시점에 `epoch_metrics_list` 를 동기적으로
`list(...)` 스냅샷했기 때문에, background drain 이 채우기도 전의 (더 짧은)
리스트가 박제됐다.

## Fix

핵심 원칙: **checkpoint 는 그 epoch 의 모든 기록이 완성된 *후*에만 저장한다.**

1. **trainer 에 `post_epoch_callback` 추가** (`mae_anomaly/trainer.py`, tracked).
   epoch loop 의 *가장 끝* (contrib·timing 포함 모든 append 완료 후) 에 호출된다.
   checkpoint 저장을 이 콜백으로 옮겨 history 가 항상 len == epoch 으로 일관.

2. **eval-before-checkpoint 불변식** (`scripts/run_base_experiments.py`, gitignored
   runner — 디스크에서 활성). `_run_bg_all` 순서를 재배치:
   ```
   A. join prev_thread
   B. 이번 epoch 의 eval 실행 (_compute_cpu_eval — 큐 제거, 결과 반환)
   C. _process_eval_result(eval) → epoch_metrics_list 에 기록 (best 복사는 보류)
   D. epoch_metrics_list 를 checkpoint 에 fold-in
   E. torch.save(checkpoint)         # 이제 "checkpoint_N 존재 ⟺ ep N 까지 eval 기록 완료"
   F. is_best 면 latest(=ckpt_N) → best 복사 (latest 가 이미 ckpt_N 이라 race-free)
   ```
   thread 가 자기 eval 을 직접 기록하므로 result queue 는 폐기.

3. **CPU-clone 스냅샷.** `model.state_dict()` 는 live param 과 메모리를 공유하므로,
   async save 가 다음 epoch 의 mutate 된 weight 를 잡을 수 있다. `_clone_state_to_cpu`
   / `_clone_optim_state` 로 그 epoch 의 weight 를 CPU 복제 후 저장.

4. **strict resume normalization** (로드측 방어). 구 buggy checkpoint 를 resume 할
   때 `epoch` 을 `1..N` 으로 재구성하고 per-epoch 키를 전부 len==ckpt_epoch 으로
   강제 (짧으면 마지막 값 back-fill, 길면 truncate). `batch_profiling` 같은
   per-BATCH 키는 명시적으로 제외 (`_NON_PER_EPOCH`).

5. **viz 방어.** `plot_score_contribution_analysis` 가 x(epoch) 와 6 개 ratio 배열을
   min length 로 정렬한 후 stackplot (구 checkpoint 산출 PNG 도 깨지지 않게).

6. **완료 dataset gap backfill.** 유실된 eval epoch 의 point-score npz
   (`epoch_scores/epoch_NNN_scores.npz`) 가 디스크에 남아 있으면, 그 점수로
   메트릭을 재계산해 `epoch_metrics.json` 에 채운다. npz 가 없는
   (274_lr WaDi/A2 `[40,45]`) 경우는 복구 불가 → 해당 dataset 은 깨끗한 재학습.

## Verification

`simulation` dataset 로 full pipeline kill→resume→완주 검증 (kill@ep13, resume,
finish ep14):

- `epoch` = `[1..14]` 연속 — skip/dup 0
- per-epoch 43 키 전부 len 14 (contrib 포함)
- `epoch_metrics.json` evals = `[5,10,14]` — 누락 0 (구 코드면 `[14]` 만 남음)
- `batch_profiling` len 9 보존 (정규화서 제외)
- kill 직전 checkpoint: `epoch=10`, `epoch_metrics_list=[5,10]`, `best_checkpoint.pt`
  도 `[5,10]` (eval-before-checkpoint 불변식 성립)

## Prevention

- checkpoint 저장은 **반드시** `post_epoch_callback` (epoch loop 의 끝) 에서만.
  `epoch_callback`(mid-epoch) 에서 history 를 스냅샷하지 말 것.
- 비동기로 채워지는 리스트 (`epoch_metrics_list`) 를 checkpoint 에 넣을 때는,
  그 epoch 의 비동기 작업이 **끝난 뒤** thread 안에서 fold-in.
- resume 후 `epoch` 과 모든 per-epoch 키의 길이가 같은지 strict-normalize 로 보장.
