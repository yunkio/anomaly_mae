---
name: training-status
description: Generate a comprehensive monitoring status report for active training runs (exp271, ablation queue 285-305, etc.). Use during long-running experiments — produces the exact structured markdown report the user requires (프로세스/진행/Metrics/Health/Hardware/Δ). Invoke this skill on every Monitor tick AND whenever the user asks for status. Never abbreviate or skip sections.
---

# Training Status Report

This skill produces a status report for an active TSMAE training run. It is the **single source of truth** for monitoring report format — the user requires every report to look identical (same sections, same fields, same emoji headers) so they can scan deltas at a glance.

---

## When to use

- Active TSMAE training run lasting >30 min (single experiment or queue).
- User says "모니터링", "status", "현재 상태", or similar.
- Periodic Monitor tick fires (default cadence: 15 min; debugging: 30 s).
- Dataset transition / queue completion / crash event.

## When NOT to use

- One-off "is training alive?" check during a short discussion → a plain one-liner is fine.
- The training process has clearly exited and the user is asking about analysis or next steps.

---

## Procedure

1. **Collect raw data** via the helper script:
   ```bash
   cd /home/ykio/notebooks/TSMAE
   source /home/ykio/anaconda3/etc/profile.d/conda.sh && conda activate dc_vis
   python scripts/monitor_status.py
   ```
   The script reads `/tmp/exp271_train_log.txt` (or whichever PID/log files the
   current run has registered) and emits a plaintext block with all needed
   fields parsed from training log, nvidia-smi, top, free, df, du.

2. **Identify prior tick** — look back in the conversation for the most recent
   status report (timestamp + epoch + best metric). If none exists this session,
   note "(첫 번째 갱신 — delta 없음)" in the Δ section.

3. **Reformat into the markdown block below** (DO NOT emit the raw output —
   wrap it). Bold the dynamic numbers that matter most: current epoch, best
   metric, GPU util / temp / fan, ETA. Highlight metric improvements with ★.

4. **Speed-health judgment (REQUIRED — see "Speed health rubric" below)**:
   Every tick MUST include a "🩺 학습 속도 진단" line with verdict (정상/주의/심각)
   based on baseline-vs-observed ratio. DO NOT stop training based on judgment —
   only report. Append the judgment record to the monitoring log file (see step 5).

4b. **Auto-interpretation block (REQUIRED — see "Interpretation rules" below)**:
    `monitor_status.py` emits a `[INTERPRETATION — rule-based]` block with:
    - Phase (pre-warmup / post-warmup based on `last_eval.epoch <= 250`)
    - Best held: number of ticks/evals current best PAK_F1 has stayed unchanged
    - Latest vs best: delta of `latest_pak_f1 − best_pak_f1`
    - Speed trend (last 3 ticks): sequence + label (stable / improving / degrading)
    - Eval lag: `current_epoch − latest_eval_epoch` (large lag → bg-worker hint)
    - Notable: rule-fired alerts (best update / plateau / divergence / phase transition / etc.)

    The same `interpretation` dict is stored in monitoring_log.jsonl per tick.
    Status report MUST surface this block verbatim. DO NOT auto-stop based on interpretation.

5. **Append monitoring record to log file** at:
   `<exp_dir>/monitoring_log.jsonl` (under the run's exp dir, NOT /tmp).
   The exp_dir path is read from `/tmp/exp271_train_log.txt` parent (or from
   monitor_status.py's `[INIT] exp_dir` line). Each tick appends one JSON line
   with: `{ts, epoch, dataset, speed_last5, speed_last20, eval_cost, gpu_util,
   cpu_load1, worker_cpu_pct, judgment: <verdict>, judgment_reason: <text>,
   best_pak_f1, latest_pak_f1}`. The file is owned by the run; survives across
   ticks; gets archived with the exp dir.

6. **Special cases** (see "Special cases" section below):
   - Dataset just transitioned → prepend ✅ 완료된 dataset summary.
   - Crash / NaN / OOM → 🚨 alert section at top + traceback excerpt.
   - Queue / experiment completed → 📦 final summary + suggested next step.

7. **Monitor watchdog (자동 알림 끊김 방지) — REQUIRED EVERY TICK**:
   In-harness Monitor task (timeout 1h, undocumented ~2h cap) may die silently.
   **On every tick** of this skill, after producing the status report, check whether
   any active Monitor task exists for this run; if **NO active Monitor task is found**,
   **AUTOMATICALLY re-arm** a new one via the Monitor tool with this exact spec:

   ```
   Monitor(
     description: "exp271 retake: 15-min heartbeat + crash watch + speed alerts (auto-rearmed <ts>)",
     timeout_ms: 3600000,
     persistent: true,
     command: """
       LOG_PTR=/tmp/exp271_train_log.txt
       HEARTBEAT_LOG=/tmp/exp271_heartbeat.log
       ( tail -F "$HEARTBEAT_LOG" 2>/dev/null & \\
         TRAIN_LOG="$(cat $LOG_PTR 2>/dev/null)"
         [ -n "$TRAIN_LOG" ] && tail -F "$TRAIN_LOG" 2>/dev/null | \\
           grep -E --line-buffered "Traceback|CRITICAL|OOM|CUDA out of memory|Killed|RuntimeError|Completed:|##### \\[[0-9]+/[0-9]+\\]|ERROR:" &
         wait
       ) 2>&1
       """
   )
   ```

   Also check the **out-of-harness heartbeat daemon**:
   - `ps -p $(cat /tmp/exp271_heartbeat_pid.txt) -o stat` — if dead, **re-spawn** with self-heal loop:
     ```
     setsid nohup bash -c 'LOG=/tmp/exp271_heartbeat.log; while true; do fd1=$(readlink /proc/$$/fd/1 2>/dev/null); if [ ! -e "$LOG" ] || [[ "$fd1" == *"(deleted)"* ]]; then exec >> "$LOG" 2>&1; fi; echo "=== HEARTBEAT $(date \"+%Y-%m-%d %H:%M:%S\") ==="; python /home/ykio/notebooks/TSMAE/scripts/monitor_status.py 2>&1; echo ""; sleep 900; done' </dev/null >/tmp/exp271_heartbeat.log 2>&1 &
     ```

   **Self-heal rationale (added 2026-05-28)**: bash redirect `>file` opens the
   log file at daemon launch time. If the log file is later unlinked (e.g., by
   a `rm -f /tmp/exp271_*.log` cleanup), the daemon's fd 1 still points to the
   now-deleted inode → all subsequent heartbeat writes go to a "ghost file"
   invisible to the filesystem. The self-heal loop checks every iteration
   whether `/proc/$$/fd/1` is still bound to the real log path; if not (either
   missing or `(deleted)` suffix), `exec >> "$LOG" 2>&1` re-opens stdout/stderr
   to a fresh inode at the same path. Side benefit: cleanup commands can
   safely `rm` the log mid-run — daemon recovers within one cycle (≤15 min).

   Both layers must remain alive across the run. If both die, jsonl logging stops.
   Report watchdog status in the report's 🛡️ Health section:
   `Monitor task: alive/re-armed/dead` and `Heartbeat daemon: alive/re-spawned/dead`.

---

## Speed health rubric (REQUIRED on every tick)

Compute the verdict from observed `speed_last5` (s/ep) and the per-dataset
baseline. Baselines reflect known-good runs with VUS skipped + canonical config:

| Dataset       | Baseline s/ep | Note |
|---------------|---------------|------|
| SWaT_A1A2     | 8–10 s/ep     | 33 batches/ep, full GPU util |
| WaDi_A1       | 15–25 s/ep    | 60 batches/ep, larger dataset |
| WaDi_A2       | 15–25 s/ep    | similar to A1 |
| PSM           | 10–15 s/ep    | medium size |
| simulation    | 3–5 s/ep      | small dataset |

Verdict bands (ratio = `speed_last5 / baseline_high`):

| Ratio        | Verdict       | Action |
|--------------|---------------|--------|
| ≤ 1.5×       | **정상**      | Continue, no concern |
| 1.5× – 2.5×  | **주의**      | Report ⚠️; check CPU contention / dataloader workers |
| 2.5× – 5×    | **심각 (저속)** | Report 🟠; bg-worker bottleneck likely; check `ps aux \| grep python` |
| > 5×         | **🚨 stuck 의심** | Report 🚨; suspect bg-worker stuck, dataloader hang, or OOM near-miss. **DO NOT auto-kill** — show ps tree + py-spy hint, let user decide |

Special cases:
- **last5 spike on dataset start** (first 10 epochs): grace period; mark as
  "🟢 초기 transition 정상" if epoch < 10
- **eval phase**: speed_last5 includes eval cost; if `(speed_last5 - eval_cost/5) ≤ 1.5× baseline`, mark as 정상

Rule: **The judgment is informational only. Never auto-kill or auto-stop based
on speed verdict.** The user makes the call. Skill must always report verdict
+ rationale + recommended check commands.

Example judgment outputs:

```
🩺 학습 속도 진단: **정상** (last5=9.2s/ep, baseline SWaT 8-10, ratio=0.92×)
🩺 학습 속도 진단: **주의** ⚠️ (last5=42s/ep, baseline WaDi 15-25, ratio=1.68× — CPU contention 의심,
   확인: `ps aux --sort=-%cpu | head -5`)
🩺 학습 속도 진단: **심각 (저속)** 🟠 (last5=85s/ep, baseline WaDi 15-25, ratio=3.4× —
   bg-worker stuck 의심. fd 확인: `ls /proc/<bg-pid>/fd/`)
🩺 학습 속도 진단: **🚨 stuck 의심** (last5=180s/ep, baseline 15-25, ratio=7.2× — 메인 학습 정지 가능.
   사용자 확인 필요. 자동 정지 X)
```

---

## Interpretation rules (rule-based auto-analysis)

`monitor_status.py` emits an `[INTERPRETATION — rule-based]` block on every tick.
This is **rule-based** (deterministic) — no LLM inference. The same dict is
stored in `monitoring_log.jsonl` so trend analysis post-run is possible.

Fields:

| Field | Source | Use |
|---|---|---|
| `phase` | `pre-warmup` if `last_eval.epoch ≤ 250` else `post-warmup` | dataset phase context |
| `best_held_ticks` | # consecutive prior monitoring ticks where best_pak_f1 == current | plateau detection |
| `best_held_evals` | `(last_eval.epoch − best.epoch) / eval_interval` | plateau in eval units |
| `latest_vs_best` | `latest_pak_f1 − best_pak_f1` | divergence indicator |
| `speed_trend_3` | last 3 ticks' `speed_last5` sequence | trend context |
| `speed_trend_label` | `stable` (Δ<10%) / `improving` / `slow-degrading` (10-20% ↑) / `degrading` (>20% ↑) | trend assessment |
| `eval_lag_epochs` | `current_epoch − latest_eval_epoch` | async queue backlog |
| `notable` | list of triggered rule strings | event log |

Notable rules (auto-fired alerts):
- `★ best PAK_F1 updated: X → Y (+Δ)` — when best PAK_F1 improves vs prior tick
- `plateau: best N evals 미경신` — when `best_held_evals ≥ 5`
- `divergence 의심: latest vs best = ...` — when `latest_vs_best < -0.05`
- `🎉 student-joining boundary 통과` — when crossing ep 250
- `⚠️ speed degrading` — when 3-tick speed trend label is `degrading`
- `⚠️ eval lag = N epochs (bg-worker 의심)` — when `eval_lag_epochs > 30`
- `🚨 speed 진단 = '심각/stuck' → 사용자 확인 필요` — verdict-driven escalation

These are **informational**. Skill never auto-stops based on interpretation.

---

## Required structure (exact — do not simplify)

```
## 📊 <exp_name> Phase <N> Status — `HH:MM:SS` (runtime XhYYm)

### 🔵 프로세스
| 항목 | 값 |
|---|---|
| Wrapper PID | <pid> (alive/DEAD) |
| Worker PID  | <pid> (alive/DEAD) — script name, thread name |
| AMP         | bf16/fp16, scaler=none/enabled ✓ |
| Exp dir     | <directory name only> |

### 🟢 진행 상황
| 항목 | 값 |
|---|---|
| Dataset      | **N/M** — <dataset name> |
| Epoch        | **N/total done** (PCT%) |
| Speed        | train **A.BBs/ep** (last 20) / **C.DDs/ep** (last 5) |
| Eval cost    | ~Es per eval (every K epochs) |
| Amortized    | **F.FFs/ep** |
| Dataset ETA  | **HhMMm** (finish ~HH:MM) — N epoch 남음 |
| Overall ETA  | **HhMMm** (finish ~HH:MM) — N dataset 남음 |

### 🎯 Metrics (<current dataset name>)

Metrics are grouped by **family** (left → right: strict → tolerant, then training losses):

| Family | 메트릭 | 특징 |
|---|---|---|
| **Point/window** | `f1` `prc` `f1_t` | `f1` is **sklearn point-level F1 at F1-optimal threshold** (NO point-adjustment), `prc` honest PR-AUC |
| **PA%K** | `pak_f1` `pak_prc` | point-adjust K=0..100 integrated (TSAD-tolerant) — short names (was `pak_f1` / `pak_prc`) |
| **Event/Range** | `aff_f1` `r_f1` | interval-level evaluation (distance / overlap) |
| **Diagnostic / Training loss (5종, 2026-05-29 짧은 이름)** | `t_re` `re_snr` `s_re` `dis` `dis_snr` | teacher recon + teacher recon SNR + student recon + discrepancy + discrepancy SNR. **Pre-warmup 에서는 `s_re / dis / dis_snr` 가 - 로 표시** (의미 misleading 또는 forward-skip sentinel). `t_re / re_snr` 은 teacher-only 이므로 pre-warmup 에서도 유효 |
| **Legacy aliases** (table 에는 비표시) | `t_loss` `s_loss` | `train_rec_loss` (joint recon) / `train_disc_loss` (==dis). monitor 의 `Latest losses (legacy):` 줄에서 확인. 새 표에서는 `t_re / dis` 사용 |

**VUS는 모니터에서 제외** (2026-05-27 fix): per-epoch eval에서 VUS 계산이 bg-worker bottleneck의 주원인이라 lite mode에서 스킵됨. 학습 종료 후 best epoch에서만 offline 계산되어 `epoch_metrics.json`에 저장. 모니터링 리포트에서 vus_pr/vus_roc 컬럼은 **표시하지 않음**.

**Diagnostic / Training loss 5종 (2026-05-29 짧은 이름)**:

| 컬럼 | History key / Source | 의미 | Pre-warmup 표시 | 비고 |
|---|---|---|---|---|
| `t_re` | `train_teacher_recon_normal` | **teacher 단독** recon loss (normal 샘플 평균) | real value | teacher 는 warmup 중 학습 진행 — 의미 그대로 |
| `re_snr` | `loss_stats['recon_SNR']` | teacher recon SNR — `(recon_a − recon_n) / (σ_a + σ_n + ε)`. Cohen's-d 형 분리도 | real value | teacher-only 분리도 → warmup 중에도 유효 |
| `s_re` | `train_student_recon_normal` | **student 단독** recon loss | - | 2026-05-29 forward-skip optimization — `model.forward(teacher_only=True)` 가 student decoder/projection/GRL/SCAD 통째 skip → loss.py 가 0.0 sentinel 로 설정 → 표 에서는 - 표기. ~22% transformer forward compute 절감. Post-warmup 부터 real 값 |
| `dis` | `train_disc_loss` | output discrepancy (teacher↔student 출력 차이) | - | loss.py:196 `if not teacher_only` block 안에서만 계산 → warmup 중 0 → 표 에서는 - |
| `dis_snr` | `loss_stats['disc_SNR']` | discrepancy SNR — `(disc_a − disc_n) / (σ_a + σ_n + ε)`. 학생-교사 분리도 | - | (Option C 2026-05-29) Evaluator 는 학습 안된 student (random init) 와 teacher 의 출력 차이를 계산하므로 mathematical value 는 존재하지만 anomaly detection signal 로 해석 misleading. 표시상 - 로 마스킹. Post-warmup 0 → 양수 전환되어 의미 회복 (전환점 dynamics 는 monitor 진행 history 에서 확인 가능) |

→ pre-2026-05-29 로그 (exp271 SWaT 등 현재 진행 중) 는 `t_re / s_re / dis / re_snr` 컬럼이 모두 - (legacy log 라서 신규 필드 미존재). entry 2 (274) 부터 채워짐.
→ 그 외 모든 row 의 `s_re / dis / dis_snr` 가 - 면 그 row 가 **pre-warmup epoch** 임을 시사 (legacy log 인데 post-warmup 인 경우와 구별 필요 — `t_re / re_snr` 으로 판단).

Comparison table (across key epochs). **Required rows (do not omit)**, sorted
chronologically by epoch:

| 시점 | f1 | prc | f1_t | pak_f1 | pak_prc | aff_f1 | r_f1 | t_re | re_snr | s_re | dis | dis_snr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ep N (초기, first eval) | … | … | … | … | … | … | … | … | … | … | … | … |
| **ep N (best @ pre-warmup, ep ≤ `teacher_only_warmup_epochs`)** ☆ | **…** | **…** | … | … | … | … | … | … | … | - | - | - |
| ep N (last pre-warmup, closest eval ≤ warmup_end) — _only if already reached_ | … | … | … | … | … | … | … | … | … | - | - | - |
| **ep N (best @ post-warmup, ep > `teacher_only_warmup_epochs`)** ♦ — _only if any post-warmup eval exists_ | **…** | **…** | … | … | … | … | … | … | … | … | … | … |
| ep N (peak/milestone, optional) | … | … | … | … | … | … | … | … | … | … | … | … |
| **ep N (best overall)** ★ | **…** | **…** | … | … | … | … | … | … | … | … | … | … |
| ep N (latest) | … | … | … | … | … | … | … | … | … | … | … | … |

→ 13 columns total: 8 detection + 5 diagnostic/loss (`t_re, re_snr, s_re, dis, dis_snr` — 사용자 지정 순서, 2026-05-29).
→ Pre-warmup row (☆ + last pre-warmup) 에서는 `s_re`, `dis`, `dis_snr` 3 셀이 - (의미 misleading 또는 forward-skip sentinel). `t_re`, `re_snr` 은 teacher-only 라 pre-warmup 에서도 real value.
→ Legacy `t_loss / s_loss` 는 monitor 출력 `Latest losses (legacy):` 줄에서 확인 가능하지만 **표에는 추가하지 않음** (`t_re / dis` 와 의미적으로 중복).
→ Warmup boundary 는 hardcoded 250 아니라 **`teacher_only_warmup_epochs` config 값` (helper 가 매번 로그에서 파싱). 다른 entry 가 다른 warmup 을 가질 수 있음.

**Per-column best-bold rule (REQUIRED, 2026-05-29 v3 — helper-driven)**:

Helper output 의 **`[BEST PER COLUMN — milestone rows only]`** block 만 참조해서 bold 결정.
**시각 비교 / 표시된 4자리 값 으로 추측 금지** (precision 손실로 잘못된 tie 판정 가능).

Block 형식:
```
[BEST PER COLUMN — milestone rows only]
  f1        : max = 0.889500 @ ep205
  prc       : max = 0.914800 @ ep150
  ...
  t_re      : min = 0.000200 @ ep150,ep190,ep200,ep205  ← multi-tie
```

규칙:
1. 각 컬럼당 helper 가 명시한 **정확히 하나의 epoch 행만** bold. 다른 행 절대 bold 금지.
2. **Tie 시 마지막 (최신) epoch 만 bold** — helper 가 이미 tie-break 적용해서 단일 epoch 반환. AI 는 그것만 bold.
3. `(no data)` 인 컬럼 (예: pre-warmup 만 emitted 된 상황의 `s_re/dis/dis_snr`) → 그 컬럼 전체 bold 없음.
4. Helper 출력에 없는 컬럼은 bold 처리하지 말 것.

방향성 (helper 가 이미 적용):
- `t_re, s_re` → min (낮을수록 좋음)
- 그 외 → max (높을수록 좋음)
- Pre-warmup 의 `s_re/dis/dis_snr` 는 helper 가 자동으로 제외 (mask 처리됨)

목적: full-precision 비교 보장 + 단일 source-of-truth 화.

**Chronological placement** of the new `best @ post-warmup` row: insert
between `last pre-warmup` and `best overall`, in the position determined by
its epoch number. If `best @ post-warmup` epoch ≡ `best overall` epoch
(common when post-warmup performs better than pre-warmup), merge into one
row labeled `ep N (best @ post-warmup ≡ best overall) ♦★` (see Deduplication
rule below).

**Pending row placement**: If `last pre-warmup` or `best @ post-warmup` has
NOT yet been reached (current epoch < `teacher_only_warmup_epochs` or no
post-warmup eval yet), DO NOT place it in the middle of the table. Instead,
drop the row from the table body entirely and append a single pending-marker
line BELOW the table:

> _Pending: `last pre-warmup` row will appear once ep ≥ warmup_end (N) — ≈Mm 후_
> _Pending: `best @ post-warmup` row will appear once any eval > warmup_end exists_

Rationale: rows in the comparison table represent actual datapoints. A row
filled with `—` cells interrupts chronological reading and looks like real
missing data. Pending markers go below as a separate note.

**Warmup boundary rows (CRITICAL — must always be included)**:

1. **`best @ pre-warmup`** (☆) = best `pak_f1` among epochs where `epoch ≤ config.teacher_only_warmup_epochs`.
   For exp271 (`teacher_only_warmup_epochs=250`): best within ep 5..250.
   This is the **teacher-only performance ceiling** — important because the model
   dynamics fundamentally change at warmup boundary (student joins training).
2. **`last pre-warmup`** = the eval closest to but not exceeding `teacher_only_warmup_epochs`.
   For exp271 with eval_interval=5: this is ep 250 if reached, else the last ep ≤ 250.
   Shows the model state at the moment student starts training.
3. **`best @ post-warmup`** (♦) = best `pak_f1` among epochs where `epoch > config.teacher_only_warmup_epochs`.
   For exp271: best within ep 251..500. **Symmetric to `best @ pre-warmup`** —
   shows the student-joined performance peak. Comparing ☆ vs ♦ reveals whether
   student joining helps (♦ > ☆, e.g., SWaT/WaDi_A1) or hurts (♦ < ☆, e.g., WaDi_A2).
   Pending until any post-warmup eval exists.
4. **`best overall`** (★) = best `pak_f1` across ALL evals so far.
   Either ☆ ≡ ★ (pre-warmup wins, e.g., WaDi_A2) or ♦ ≡ ★ (post-warmup wins, e.g., SWaT/WaDi_A1).

If current training is still in pre-warmup phase (current_epoch ≤ warmup), the
"last pre-warmup" row can be marked `(not yet — current ep < warmup)`, the
"best @ post-warmup" row is omitted (pending marker only), and
"best overall" ≡ "best @ pre-warmup".

If the config has no warmup (`teacher_only_warmup_epochs = 0` or absent), drop
all three warmup-specific rows (☆, last pre-warmup, ♦).

**Deduplication rule (avoid duplicate epoch rows)**:

When two semantically distinct labels refer to the SAME epoch, MERGE into ONE
row by concatenating the labels with `≡`. Examples:

- During pre-warmup, `best @ pre-warmup ≡ best overall` (same epoch): show as
  ONE row labeled `ep N (best @ pre-warmup ≡ best overall) ☆★`.
- Post-warmup-wins case: `best @ post-warmup ≡ best overall` → `ep N (best @ post-warmup ≡ best overall) ♦★`.
- Pre-warmup-wins case (WaDi_A2 pattern): `best @ pre-warmup ≡ best overall` even after post-warmup exists → `ep N (best @ pre-warmup ≡ best overall) ☆★`, and `best @ post-warmup ♦` appears as a separate row (different epoch, lower PAK_F1).
- After warmup ends, if `last pre-warmup` happens to coincide with the previous best epoch: merge labels.
- If `latest` epoch equals `best overall`: merge into `ep N (latest ≡ best overall) ★`.

NEVER emit two rows with the same epoch number — they confuse the reader and
suggest there are more distinct datapoints than actually exist.

**SWaT dual-eval — 2-table format (REQUIRED, 2026-05-29)**:

For datasets where the helper output contains a `(excl22):` line (SWaT_A1A2),
emit **two separate Comparison tables**, both per the column spec above:

- **Table 1 — `<ds>` Full**: rows use the primary (`Latest eval (ep N):` /
  `Best pak_f1 (ep N):`) detection metrics, **plus** the loss columns
  (`t_re`, `s_re`, `dis`) and `dis_snr` from the helper output.
- **Table 2 — `<ds>` excl22**: rows use the `(excl22):` line metrics for the
  SAME epoch as Table 1. **detection-only columns** (prc, f1, f1_t,
  pak_f1, pak_prc, aff_f1, r_f1) — no disc_snr, no loss columns.

Add an explanatory line below Table 2:

> _Training losses (`t_re`, `s_re`, `dis`) and `dis_snr` are
> dataset-wide and identical to Table 1 — they do not depend on which test
> region is masked, so they are not duplicated here._

For non-SWaT datasets (WaDi, PSM, simulation), emit ONE Comparison table.

**Missing-cell rule (CRITICAL — 빈칸 절대 금지, 2026-05-29 강화)**:

The skill MUST produce a fully-filled table. The following cases are the ONLY
legitimate reasons a cell may be missing — every case requires an explicit
**inline footnote** explaining WHY, NOT a silent `—`:

1. **Legacy log format (pre-2026-05-29)** — `recon_t / recon_s / dis` columns:
   exp271 SWaT and earlier runs do NOT have these fields in log line.
   - Action: show literal "**(legacy log: pre-2026-05-29 → recon split not emitted)**"
     in a single footnote BELOW the table, applied to the columns once.
   - Do NOT use `—` per-cell — too noisy.

2. **Pre-2026-05-27 detection columns** — `f1 / aff_f1 / r_f1`:
   - Action: similar footnote "**(legacy log: pre-2026-05-27 → f1/aff_f1/r_f1 not emitted)**".

3. **Helper output missing a metric for ONE specific epoch** (not all):
   - This usually indicates a **parser miss** (regex didn't match that line).
   - Action:
     (a) Re-run `python scripts/monitor_status.py 2>&1` and check stderr for
         `parse_warnings` block — it lists which `[Epoch N]` lines failed.
     (b) Either fix the regex or skip that specific row entirely.
   - Do NOT fill with `—`.

4. **Pre-warmup `s_re` = 0.0000**: this is a **real value**, not missing.
   student is frozen during warmup so its loss is genuinely zero.
   - Action: show as `0.0000`, do NOT mark as `—`.

5. **`Pending` rows** (warmup boundary not reached): handled separately — keep
   below the table as `> _Pending: ..._` markers, not as table rows.

**Procedural enforcement on every tick**:
- Step 1: Read full helper output INCLUDING stderr.
- Step 2: For each milestone row to emit, verify the helper provides ALL required cells.
- Step 3: If a cell is missing, apply rule 1-5 above. **Never silently emit `—`**.
- Step 4: If the helper's `parse_warnings` block is non-empty, prepend a
  🚨 alert above the table listing the failed epoch numbers.

**Dataset-mixing diagnostic recovery (CRITICAL — 2026-05-31, post-incident)**:

monitor_status.py parses the shared queue log, which accumulates eval lines from
ALL entries. When the current dataset matches a dataset from a PRIOR queue entry
(e.g. entry N's SWaT after entry N-1's SWaT, or WaDi/A1 after a prior WaDi/A1),
the helper's `[BEST PER COLUMN]` and `≥50ep` milestone rows MIX the prior entry's
stale values into the current dataset — `pak_f1`, `re_snr`, the `(excl22)` lines,
etc. all become unreliable. (User accepts this mixing — do NOT fix the helper.)

**The required-structure table MUST still be produced in full. Mixing is NOT an
excuse to drop columns or collapse to a `pak_f1`-only table.** Two hard rules,
both born from a real failure (post-warmup `s_re/dis/dis_snr` were omitted while
student was active):

1. **NEVER drop the 5 diagnostic columns** (`t_re re_snr s_re dis dis_snr`).
   Post-warmup, `s_re/dis/dis_snr` are REAL student-teacher signals and are the
   single most important thing to show once student has joined — omitting them is
   the worst possible cut. A `pak_f1`-only table is a skill violation.

2. **Recover the real current-entry values from the raw log** when the helper is
   mixed. The full eval line is always present; filter by entry prefix AND eval
   cost (which differs per dataset, so it separates same-named datasets across
   entries):
   ```bash
   QLOG=$(cat /tmp/exp271_train_log.txt)
   # eval cost signature: SWaT ~8-11s (dual full+excl22) | WaDi ~2s | PSM ~4s | simulation ~2s
   grep -E "exp<ENTRY>.*\[Epoch [0-9]+\] PRC=" "$QLOG" | grep "eval=<Ns>s" | tail -N
   ```
   Raw-line → table-column map (run_base_experiments.py eval line):
   `recon_t→t_re`, `recon_SNR→re_snr`, `recon_s→s_re`, `dis→dis`, `d_SNR→dis_snr`,
   `PAK_F1→pak_f1`, `PAK_PRC→pak_prc`, `PRC→prc`, `F1→f1`, `F1_T→f1_t`,
   `AFF_F1→aff_f1`, `RF1→r_f1`.
   Pre-warmup rows: `s_re=0.0000`/`dis=0.0000` are REAL (student frozen) → show as
   `-` per Option-C masking. Post-warmup rows: show the real numbers.

3. **SWaT: BOTH required tables, always.** Table 2 (excl22) is REQUIRED and is NOT
   exempt from mixing recovery. The raw `[excl22] PRC=...` line immediately follows
   each `[Epoch N]` Full line — recover it with `grep -A1`:
   ```bash
   grep -A1 -E "exp<ENTRY>.*\[Epoch <N>\] PRC=" "$QLOG" | grep "excl22"
   ```
   Emit Table 1 (Full, all 12 cols incl. diag) AND Table 2 (excl22, 7 detection
   cols) for the SAME rows. Dropping Table 2 is a skill violation, same class as
   dropping the diag columns.

4. **Emit EXACTLY the required structure — never improvise.** Do NOT add ad-hoc
   tables (e.g. a standalone "diag trend" table) and do NOT drop required pieces
   (Table 2 excl22, the 5 diag columns, the warmup-boundary rows). When mixing
   forces recovery, recover into the SAME required tables — the output shape must
   be identical to a non-mixed tick. Improvising a different layout is the recurring
   "이상하게 출력" failure the user has flagged repeatedly.

5. On every post-warmup tick, before emitting, ASK: "are `s_re/dis/dis_snr` real for
   post-warmup rows, AND is the SWaT excl22 Table 2 present?" If either is missing,
   STOP and recover via rules 2-3.

**Metric provenance** (입증된 출처만 사용 — 추정 금지):
- `prc` = sklearn PR-AUC (point-level, honest)
- `f1` = **`f1_score`** (sklearn point-level F1 at F1-optimal threshold). NOT `pa_0_f1` (that's lenient PA). For exp272 onward, log line emits sklearn f1_score for the `F1=` field.
- `f1_t` = F1 at adaptive threshold (in-house `compute_f1_t_at_threshold`, TS-window-aware)
- `pak_f1`, `pak_prc` = PA%K AUC over F1 / PR-AUC, integrated K=0..100 (in-house `compute_pa_k_auc`)
- `aff_f1` = Huet et al. KDD 2022 Affiliation-F1 ([ahstat/affiliation-metrics-py](https://github.com/ahstat/affiliation-metrics-py), `pip install git+...`)
- `r_f1` = Tatbul et al. NeurIPS 2018 Range-based F1 ([TheDatumOrg/TSB-AD](https://github.com/TheDatumOrg/TSB-AD), `pip install TSB-AD`), fixed-threshold mode (`preds=` at PA%K optimal threshold) for per-epoch cost
- `dis_snr` (구 `dis_snr`) = discrepancy SNR — `(disc_a − disc_n) / (σ_a + σ_n + ε)`. Cohen's-d-style effect size of student↔teacher discrepancy. Positive = anomaly higher (정상). |값|>0.8 강한 / 0.3-0.8 중간 / <0.1 분리 없음. **Pre-warmup 에서는 - 표시** (Option C, 2026-05-29 — random-init student 측정값은 anomaly detection signal 아님).
- `re_snr` (구 `recon_snr`, 2026-05-29 신규) = teacher recon SNR — `(recon_a − recon_n) / (σ_a + σ_n + ε)`. Teacher 단독 분리도. **Pre-warmup 에서도 real value** (teacher 학습 진행). `dis_snr` 과 짝.
- _(vus_pr/vus_roc는 final best epoch에서만 계산되며 모니터에 표시하지 않음 — `epoch_metrics.json`에서 확인 가능)_

**Training loss provenance** (2026-05-29 신규 짧은 이름):
- `t_re` (구 `t_re`) = `trainer.history['train_teacher_recon_normal'][epoch-1]` — teacher 단독 recon loss [trainer.py:264, 1141](mae_anomaly/trainer.py#L264).
- `s_re` (구 `s_re`) = `trainer.history['train_student_recon_normal'][epoch-1]` — student 단독 recon loss [trainer.py:265, 1142](mae_anomaly/trainer.py#L265). **Pre-warmup: -** (forward-skip 0 sentinel).
- `dis` = `trainer.history['train_disc_loss'][epoch-1]` — output discrepancy. **Pre-warmup: -** (loss.py:196 gate).
- `t_loss` (legacy, table 비표시) = `train_rec_loss` — 합쳐진 reconstruction loss. "teacher loss" 아님.
- `s_loss` (legacy, table 비표시) = `train_disc_loss` — **`dis` 와 동일 값**. "student loss" 아님.
- Log line emit site: [run_base_experiments.py:2333-2335](scripts/run_base_experiments.py#L2333). Parser regex: `monitor_status.py:RE_EVAL_NEW`.
- Pre-warmup masking: helper 가 `eval.epoch <= teacher_only_warmup_epochs` (로그 파싱) 시 `s_re / dis / dis_snr` 을 None 으로 마스크 → 표 -. Warmup 값은 hardcoded 아니라 helper 가 매번 `grep -aoE 'teacher_only_warmup_epochs[= ]+[0-9]+'` 로 추출.

- (one-line observations: metric breakthrough, plateau, divergence, etc.)

**Backward compat note**: experiments before 2026-05-27 22:30 (exp271 SWaT included) do NOT have f1/aff_f1/r_f1 in log — show `—` for those columns. exp272 onward has the full set (still no VUS in per-epoch, only at final best epoch).

**Anomaly-ratio threshold variants (epoch_metrics.json only, NOT displayed in monitor)**:
exp271 resume (2026-05-27 ~04:50 onward) saves additional `_ar`-suffixed metrics where threshold = `(1 - anomaly_ratio)`-th quantile of scores instead of optimal-F1 threshold. This decouples metric evaluation from F1-greedy threshold selection (which leaks ground truth).

Keys saved per epoch (auto-saved via `epoch_metrics.json`):
- `anomaly_ratio`, `anomaly_ratio_threshold`
- `f1_ar`, `precision_ar`, `recall_ar` (point-strict)
- `f1_t_ar`, `precision_t_ar`, `recall_t_ar` (time-series F1)
- `affiliation_f1_ar`, `affiliation_precision_ar`, `affiliation_recall_ar`
- `r_based_f1_ar`

Use for post-hoc analysis. PA%K family (`pak_auc_*`) is K-integrated → no `_ar` variant needed.

### 🛡️ Health
| 항목 | 상태 |
|---|---|
| NaN/Inf 누적 | **N** (안정/!!!) |
| grad_norm logger | 활성 |
| (others if relevant) | |

### 🔥 Hardware
| 컴포넌트 | 값 | 비고 |
|---|---|---|
| GPU mem      | A / B MiB (X%) | <bf16 note> |
| GPU util     | **X%** | full load / mixed |
| GPU mem-bw   | X% | bandwidth utilization |
| GPU power    | A / B W | X% TDP |
| GPU temp     | **X°C** | <↑↓ if delta meaningful> |
| GPU fan      | **X%** | <↑↓ if delta meaningful> |
| Pstate / clk | P0/P2 / gr·memMHz | boost / idle |
| CPU load     | A / B / C of Nc | 여유 / 과부하 |
| Worker proc  | X% CPU, Y% MEM, RSS Z MiB | core saturated / idle |
| RAM          | A / B GiB (X%), swap N | 여유 / 부족 |
| Disk         | A / B GiB (X%) — exp dir M MiB | 여유 / 위험 |

### 📈 직전 갱신 (HH:MM) 대비 변화
| 항목 | 이전 | 현재 | Δ |
|---|---|---|---|
| Epoch | … | … | +X |
| Best PAK_F1 | … | … | ±X ★ (if updated) |
| GPU temp | … | … | ±X°C |
| GPU fan | … | … | ±X%p |
| GPU power | … | … | ±X W |
| Dataset ETA | … | … | ±X min |

### ⏰ 다음 갱신: ~HH:MM
```

---

## Hard rules

1. **Never simplify across ticks.** Same sections, same field set, every time.
2. **Korean labels** in section headers and tables (user is Korean-speaking).
3. **Emoji on section headers IS approved here** — overrides default no-emoji
   rule (user explicitly requested this style 2026-05-27).
4. **Tables, not bullet lists**, for structured data.
5. **Bold the numbers that matter** so user can scan in <5 seconds.
6. **Always include the Δ table** — even if all zeros, show "no change".

---

## Special cases

### Dataset transition (e.g., SWaT done → WaDi starts)

Insert at the top, BEFORE 진행 상황:

```
### ✅ 완료된 dataset: <name>
| 항목 | 값 |
|---|---|
| 총 epoch | N |
| Wall-clock | XhYYm |
| Best PAK_F1 | **X.XXXX** @ ep N |
| Final eval (ep last) | PRC=… PAK_F1=… F1_T=… |
| Saved | <path to best ckpt or metrics> |
```

The 🎯 Metrics section then reflects the NEW current dataset (fresh start).
Note in a footnote: "Speed history reset for new dataset."

### Queue / experiment complete

Replace ⏰ "다음 갱신" with:

```
### 🎉 학습 종료
| Dataset | Best PAK_F1 | Best epoch | Wall-clock |
|---|---|---|---|
| SWaT/A1A2_full | … | … | … |
| WaDi/A1 | … | … | … |
| WaDi/A2 | … | … | … |
| PSM | … | … | … |

다음 단계 제안:
- Plot generation (scripts/visualize_all.py)
- Queue 285-305 시작 (사용자 확인 후)
- Notion 결과 업로드
```

### Crash / NaN / OOM

🛡️ Health section moves to top. Add 🚨 section:

```
### 🚨 ALERT — <crash type>
- Worker PID: <pid> (DEAD / OOM-killed / NaN trap)
- Time of incident: HH:MM:SS
- Last successful epoch: N
- Traceback (last 50 lines of log):
  ```
  …
  ```

다음 단계 제안:
- (recovery options specific to error type)
```

Replace ⏰ "다음 갱신" with proposed recovery steps.

---

## Helper script details

`scripts/monitor_status.py` (in the project) parses:
- `/tmp/exp271_train_log.txt` for current log path
- `/tmp/exp271_train_pid.txt` for wrapper PID (worker PID resolved via pgrep -P)
- Training log for `Epoch N/total: 100%|...| K/K [MM:SS<00:00, X.XXit/s]` lines
- Training log for `[Epoch N] PRC=... PAK_F1=...` eval lines
- Training log for `[CRITICAL] epoch N: M batch` NaN/Inf events
- `nvidia-smi --query-gpu=...` for GPU state
- `top -bn2 -d 0.2 -p PID` for worker %CPU/%MEM
- `free -m`, `df -BM`, `du -sh` for memory/disk

If you change the training log format (`trainer.py` print statements), update
the regex patterns in `scripts/monitor_status.py` accordingly.

---

## Why this skill exists

User feedback (2026-05-27):

> "아까 모니터링 구조화가 매우 마음에 들었어. 근데 항상 모니터링 냅두면
> 점점 더 간략화 하거나 생략되거나 포맷이 깨지더라고."

Without a pinned spec invoked **fresh** every tick, LLM context drift + "don't
repeat myself" pressure compresses the format. Skill invocation reloads this
SKILL.md from scratch each time, eliminating drift.

To change the format permanently, edit this file. All future status reports
(this session + future sessions + manual `/training-status` invocations) will
pick up the change immediately.
