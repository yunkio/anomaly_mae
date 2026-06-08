# 2026-06-08 — Finalize metadata computed at the WRONG epoch (best_checkpoint async misalignment)

## Symptom
For many cells (esp. small/simple datasets), `experiment_metadata.json["metrics"]`
did **not** reflect the selected best epoch. `timing.best_epoch` and
`timing.best_epoch_score` were correct, and `epoch_metrics.json` / the saved
`epoch_scores/*.npz` were correct, but the final metrics block diverged — by up
to **~0.05 pak_auc_f1** in the worst simple cells.

Example — `271/SMAP/P-4`: `timing.best_epoch=255`, `best_epoch_score=0.4858`
(== `epoch_metrics@255` == npz@255 recompute), but
`metrics.pak_auc_f1 = 0.4337` (== `epoch_metrics@470`).

## Detection
The re-experiment Phase 4 strict-consistency check compared
`metadata.vus_pr/roc` against `_compute_vus_for_npz_file(npz@best_epoch)` for all
cells. **176 / 200** non-excl22 FLIP cells mismatched (non-flip cells were
already recomputed from npz@best, so they were clean). The mismatches were NOT a
float32/float64 precision issue (diffs up to 1e-2, not 1e-5) — they pointed to a
different *epoch*.

## Root cause
`_bg_worker_body` finalize computes the metadata by **re-forwarding
`best_checkpoint.pt`**. That checkpoint is written by the *online* best-metric
tracker: when the best per-epoch eval result is processed, the current `latest`
weights are copied to `best_checkpoint.pt`. But the per-epoch eval pipeline is
**async and far slower than training** on small datasets (log shows
`train ~0.1 s/ep` vs `eval ~3 s`), so the eval result is processed long after the
model has advanced. The checkpoint therefore stores **later-epoch weights** while
being **labeled** with the best epoch. The finalize re-forward of those weights
yields metrics at the wrong epoch.

Smoking gun (log): `Best model: epoch 255 (pak_auc_f1=0.4858), loaded from
best_checkpoint.pt` — yet the resulting `[...] Eval done: PAK_AUC_F1=0.4337`.
The loaded weights do not match the labeled epoch.

Base cells (SWaT/WaDi/PSM) were mostly unaffected (slower eval-to-train ratio →
smaller backlog; `best_model.pt.best_epoch == timing.best_epoch`). The bug was
masked because the offline `recompute_evalrevert` (used for non-flip cells) and
the **excl22 finalize block already read `npz@best_epoch`** instead of
re-forwarding — so excl22 was always correct.

## Fix
- **Code** (`run_base_experiments.py` `_bg_worker_body`): after the re-forward,
  recompute the primary / disc / teacher metadata from the **saved
  `npz@best_epoch`** (the authoritative best-epoch snapshot that drove selection),
  mirroring the excl22 block. `compute_full_metric_set(..., eval_mask=None,
  n_thresholds=200, sliding_window=100, lite=False)`. Wrapped in try/except (no-op
  fallback to the re-forward if the npz is missing).
- **Data** (current results): recomputed all 210 flip-cell metadata from
  `npz@best_epoch` (`scripts/reexp_phase4_fix_flip_metadata.py`). Phase 4 re-verify
  → ALL CONSISTENT (360/360). Verified `metadata == epoch_metrics@best` across all
  7 dataset/eval types.

## Lesson / prevention
The metadata for the best epoch must come from the **persisted best-epoch artifact**
(npz / epoch_metrics), never from re-forwarding a checkpoint whose weight↔epoch
alignment depends on async timing. Treat `best_checkpoint.pt` as suitable for
ad-hoc inference only, not as the source of record for best-epoch metrics.
Related single-source-of-truth post-mortems: `2026-05-29_fm_score_omission`,
`2026-05-30_resume_record_consistency`.
