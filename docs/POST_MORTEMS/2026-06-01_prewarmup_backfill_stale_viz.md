# Pre-warmup backfill — visualization status (2026-06-01)

Epoch-metric curves regenerated: **45/45** cells (from corrected `epoch_metrics.json`; old PNGs backed up under `.trash/0531/backfill_viz_backups/`).

## best_model/*.png status (model-forward viz)

| status | count | meaning |
|---|---|---|
| CORRECT | 29 | post-warmup best; full score unaffected → kept as-is |
| STALE_REGENERABLE | 11 | pre-warmup best, not flipped; weights exist → regenerable with `force_recon_only=True` (deferred) |
| STALE_NOT_REGENERABLE | 5 | best_epoch flipped; no per-epoch weights → metrics scalars recomputed, per-sample PNGs reflect OLD epoch |

## Per-cell detail

| cell | best (old→new) | best_model viz |
|---|---|---|
| 271_20260529_100418_271canon_baseline/WaDi/A1 | 340→340 | CORRECT |
| 271_20260529_100418_271canon_baseline/WaDi/A2 | 350→350 | CORRECT |
| 271_20260529_100418_271canon_baseline/SWaT/A1A2_full | 370→370 | CORRECT |
| 271_20260529_100418_271canon_baseline/SWaT/A1A2_excl22 | 270→270 | CORRECT |
| 271_20260529_100418_271canon_baseline/PSM | 105→100 (FLIP) | STALE_NOT_REGENERABLE |
| 271_lr2_20260529_225351_baseline/WaDi/A1 | 400→400 | CORRECT |
| 271_lr2_20260529_225351_baseline/WaDi/A2 | 240→240 | STALE_REGENERABLE |
| 271_lr2_20260529_225351_baseline/SWaT/A1A2_full | 320→320 | CORRECT |
| 271_lr2_20260529_225351_baseline/SWaT/A1A2_excl22 | 260→260 | CORRECT |
| 271_lr2_20260529_225351_baseline/PSM | 205→205 | STALE_REGENERABLE |
| 271_lr_20260529_225351_baseline/WaDi/A1 | 255→255 | CORRECT |
| 271_lr_20260529_225351_baseline/WaDi/A2 | 245→240 (FLIP) | STALE_NOT_REGENERABLE |
| 271_lr_20260529_225351_baseline/SWaT/A1A2_full | 370→180 (FLIP) | STALE_NOT_REGENERABLE |
| 271_lr_20260529_225351_baseline/SWaT/A1A2_excl22 | 260→260 | CORRECT |
| 271_lr_20260529_225351_baseline/PSM | 60→60 | STALE_REGENERABLE |
| 274_20260529_100418_274canon_balsamp/WaDi/A1 | 355→355 | CORRECT |
| 274_20260529_100418_274canon_balsamp/WaDi/A2 | 320→320 | CORRECT |
| 274_20260529_100418_274canon_balsamp/SWaT/A1A2_full | 415→415 | CORRECT |
| 274_20260529_100418_274canon_balsamp/SWaT/A1A2_excl22 | 290→290 | CORRECT |
| 274_20260529_100418_274canon_balsamp/PSM | 110→110 | STALE_REGENERABLE |
| 274_lr2_20260529_225351_balsamp/WaDi/A1 | 400→400 | CORRECT |
| 274_lr2_20260529_225351_balsamp/WaDi/A2 | 240→240 | STALE_REGENERABLE |
| 274_lr2_20260529_225351_balsamp/SWaT/A1A2_full | 380→380 | CORRECT |
| 274_lr2_20260529_225351_balsamp/SWaT/A1A2_excl22 | 445→445 | CORRECT |
| 274_lr2_20260529_225351_balsamp/PSM | 205→205 | STALE_REGENERABLE |
| 274_lr_20260529_225351_balsamp/WaDi/A1 | 205→205 | STALE_REGENERABLE |
| 274_lr_20260529_225351_balsamp/WaDi/A2 | 415→415 | CORRECT |
| 274_lr_20260529_225351_balsamp/SWaT/A1A2_full | 415→415 | CORRECT |
| 274_lr_20260529_225351_balsamp/SWaT/A1A2_excl22 | 465→465 | CORRECT |
| 274_lr_20260529_225351_balsamp/PSM | 60→60 | STALE_REGENERABLE |
| 285_20260529_225351_no_fm/WaDi/A1 | 280→280 | CORRECT |
| 285_20260529_225351_no_fm/WaDi/A2 | 215→215 | STALE_REGENERABLE |
| 285_20260529_225351_no_fm/SWaT/A1A2_full | 280→280 | CORRECT |
| 285_20260529_225351_no_fm/SWaT/A1A2_excl22 | 280→280 | CORRECT |
| 285_20260529_225351_no_fm/PSM | 95→490 (FLIP) | STALE_NOT_REGENERABLE |
| 286_20260529_225351_clamp_pm4/WaDi/A1 | 225→165 (FLIP) | STALE_NOT_REGENERABLE |
| 286_20260529_225351_clamp_pm4/WaDi/A2 | 180→180 | STALE_REGENERABLE |
| 286_20260529_225351_clamp_pm4/SWaT/A1A2_full | 270→270 | CORRECT |
| 286_20260529_225351_clamp_pm4/SWaT/A1A2_excl22 | 270→270 | CORRECT |
| 286_20260529_225351_clamp_pm4/PSM | 300→300 | CORRECT |
| 287_20260529_225351_unmask/WaDi/A1 | 325→325 | CORRECT |
| 287_20260529_225351_unmask/WaDi/A2 | 260→260 | CORRECT |
| 287_20260529_225351_unmask/SWaT/A1A2_full | 460→460 | CORRECT |
| 287_20260529_225351_unmask/SWaT/A1A2_excl22 | 270→270 | CORRECT |
| 287_20260529_225351_unmask/PSM | 95→95 | STALE_REGENERABLE |

### STALE_NOT_REGENERABLE cells (flipped — require retrain to regenerate best-model PNGs)
- **271_20260529_100418_271canon_baseline/PSM**: best_epoch flipped 105->100; no per-epoch weights for the new best epoch (only best_checkpoint.pt=105 on disk). metrics SCALARS recomputed; best-model per-sample PNGs reflect the OLD epoch.
- **271_lr_20260529_225351_baseline/WaDi/A2**: best_epoch flipped 245->240; no per-epoch weights for the new best epoch (only best_checkpoint.pt=245 on disk). metrics SCALARS recomputed; best-model per-sample PNGs reflect the OLD epoch.
- **271_lr_20260529_225351_baseline/SWaT/A1A2_full**: best_epoch flipped 370->180; no per-epoch weights for the new best epoch (only best_checkpoint.pt=370 on disk). metrics SCALARS recomputed; best-model per-sample PNGs reflect the OLD epoch.
- **285_20260529_225351_no_fm/PSM**: best_epoch flipped 95->490; no per-epoch weights for the new best epoch (only best_checkpoint.pt=95 on disk). metrics SCALARS recomputed; best-model per-sample PNGs reflect the OLD epoch.
- **286_20260529_225351_clamp_pm4/WaDi/A1**: best_epoch flipped 225->165; no per-epoch weights for the new best epoch (only best_checkpoint.pt=225 on disk). metrics SCALARS recomputed; best-model per-sample PNGs reflect the OLD epoch.

### STALE_REGENERABLE cells (pre-warmup best; regenerable offline w/ weights)
- 271_lr2_20260529_225351_baseline/WaDi/A2: best_epoch=240 (pre-warmup)
- 271_lr2_20260529_225351_baseline/PSM: best_epoch=205 (pre-warmup)
- 271_lr_20260529_225351_baseline/PSM: best_epoch=60 (pre-warmup)
- 274_20260529_100418_274canon_balsamp/PSM: best_epoch=110 (pre-warmup)
- 274_lr2_20260529_225351_balsamp/WaDi/A2: best_epoch=240 (pre-warmup)
- 274_lr2_20260529_225351_balsamp/PSM: best_epoch=205 (pre-warmup)
- 274_lr_20260529_225351_balsamp/WaDi/A1: best_epoch=205 (pre-warmup)
- 274_lr_20260529_225351_balsamp/PSM: best_epoch=60 (pre-warmup)
- 285_20260529_225351_no_fm/WaDi/A2: best_epoch=215 (pre-warmup)
- 286_20260529_225351_clamp_pm4/WaDi/A2: best_epoch=180 (pre-warmup)
- 287_20260529_225351_unmask/PSM: best_epoch=95 (pre-warmup)
