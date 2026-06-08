# 2026-06-08 — Finalize metadata diverged from the best-epoch score (finalize evaluate-path scoring ≠ npz/selection/viz)

> **2026-06-09 correction.** An earlier version of this post-mortem blamed an
> "async best_checkpoint misalignment (later-epoch weights)". A weight-level
> forensic re-run **disproved that**: `best_checkpoint.pt` IS `model@best_epoch`
> (bit-identical to a per-epoch clone at the selected epoch, verified on 4 cells:
> SMAP/P-4@255, MSL/C-2@295, SMD/machine-1-4@470, SMAP/T-3@470). The real cause is
> a **score-path divergence in the finalize's `evaluate(lite=False)`**, NOT the
> checkpoint or the epoch. The data/code fix (use `npz@best`) was correct either
> way; only the explanation was wrong.

## Symptom
For many cells (esp. small/simple datasets), `experiment_metadata.json["metrics"]`
did not match the saved best-epoch score. `timing.best_epoch`,
`epoch_metrics.json`, the per-epoch `epoch_scores/*.npz`, AND the best_model
**visualizations** were all correct, but the final metrics block diverged — by up
to **~0.05 pak_auc_f1** on the worst simple cells.

Example — `271/SMAP/P-4`: `timing.best_epoch=255`; `epoch_metrics@255` = npz@255
recompute = `0.4858`; but `metrics.pak_auc_f1 = 0.4337`.

## Root cause (corrected)
The finalize computes the metadata via `evaluator.set_eval_context(epoch=best);
evaluator.evaluate(lite=False)` on `best_checkpoint.pt`. That checkpoint is the
**correct** `model@best_epoch` (forensically verified at the weight level). The
divergence is in the **anomaly-score computation of the `evaluate(lite=False)`
path**, which does not match the score produced by the per-epoch eval
(`_evaluate_all_parallel`), the best-epoch *selection*, and the viz
(`derive_pred_data`) — all of which share the `mae_anomaly/scoring.py`
single-source path and agree with the saved npz. i.e. on the SAME `model@255`,
`evaluate(lite=False)` yields `0.4337` while the npz / selection / viz score
yields `0.4858`. This is the FM-omission / score-path-duplication class of bug
(cf. `2026-05-29_fm_score_omission`): a second scoring path drifted from the
single source.

Why it looked like an epoch bug: `0.4337` happened to be numerically close to
`epoch_metrics@470` (`0.4338`), so a value-only match wrongly suggested "loaded
the epoch-470 model". The weight-level forensic re-run dispelled this.

## Why the viz were always fine
The best_model figures are generated from `model@best_epoch` (correct checkpoint)
via `derive_pred_data`, whose score is bit-equal (Δ≈1.8e-5, float32 noise) to the
npz/selection score. So the viz were always at the correct epoch with the correct
score — only the metadata block (a separate `evaluate` call) was wrong.

## Fix
- **Code** (`run_base_experiments.py` `_bg_worker_body`): after the re-forward,
  recompute the primary / disc / teacher metadata from the **saved
  `npz@best_epoch`** (the authoritative, selection- and viz-consistent score),
  mirroring the excl22 block. `compute_full_metric_set(eval_mask=None,
  n_thresholds=200, sliding_window=100, lite=False)`, try/except fallback.
- **Data**: recomputed all 210 flip-cell metadata from `npz@best` (non-flip 160
  were already npz-based). 

## Verification (2026-06-09)
- **Audit A** (`scripts/reexp_comprehensive_audit.py`, all 370 cells, no re-run):
  `metadata.metrics == compute_full_metric_set(npz@best)` for every metric
  (pak_auc_f1, pak_auc_prc_auc, prc_auc, f1, vus_pr/roc, affiliation_f1,
  r_based_f1) AND `best_epoch == argmax(epoch_metrics)` → **370/370 OK, 0 issue**.
- **Audit B** (`scripts/reexp_auditB_forensic.py`, 4 simple flip cells, exact-config
  re-run): `best_checkpoint == model@best_epoch` (weight bit-identical) AND npz
  bit-identical to the original → **4/4 OK**. So viz are at the correct epoch.
- **Determinism**: re-running with the exact saved `best_config.json` reproduces
  the original bit-for-bit. The earlier "non-reproducible" scare was a Set-C
  preset drift (`d_model 512→256`, `batch_size 1024→512`, `dynamic_margin_k 6→2`),
  not GPU/code non-determinism. Always reconstruct configs from `best_config.json`,
  never from the live Set preset.

## Lesson
Keep ALL anomaly-score computation on the single `scoring.py` path. The finalize
metadata must come from the persisted best-epoch artifact (npz), never from a
second `evaluate` call whose scoring can drift. Value-only epoch attribution is
unsafe — confirm checkpoint↔epoch identity at the weight level before concluding
an "epoch" bug. Related: `2026-05-29_fm_score_omission`,
`2026-05-30_resume_record_consistency`.
