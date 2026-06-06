"""The single STORY_REGISTRY — backend-design.md §6.2 (P2-04).

ONE source of truth, consumed by BOTH the renderers and the ``GET /gif/stories``
manifest. The story→metric membership is EXPLICIT here (never re-derived from prose).
``story_id`` may later be lifted onto registry ``viz_hint`` (P-FB1); until then this
table is authoritative and the manifest reflects it.

BINDING P3 constraints baked in:
  * P3-01 — story ``what_carries`` (GIF-6) declares ``per_epoch_lines`` = adaptive
    (``metrics``) + ``teacher_*`` ONLY, plus 4 ``best_epoch_markers`` for the 4 score
    variants. The renderer NEVER fabricates per-epoch student/disc trajectories.
  * P3-02 — story ``signal_lives`` (GIF-5) ``anom_type_over_training`` sub-mode binds
    the 2-level pivot ``pivot[type][sub_metric]`` (default ``recon_ratio``).

FB-R3-10 / FB-R3-11 — DISPLAY metadata (additive; ``story_id`` stays STABLE):
  * ``story_id`` is the INTERNAL identity (cache sidecars, ``/gif/list``, viz_manifest
    cross-link, render dispatch) — it is NEVER renumbered.
  * ``display_order`` (int | None) gives the VISIBLE 1..N ordering of the primary story
    list (line_race is no longer "GIF-8" between 5 and 6). ``compare_grid`` is hidden
    from the primary list (``display_order=None``) — it is reached via the side-by-side
    compare toggle, not a primary card.
  * ``display_label`` is the human label shown for the position (e.g. "GIF-1").
  * ``description`` is a short, accurate sentence per story, grounded in what each story
    ACTUALLY plots (verified against the renderers in render.py + the model/metric
    semantics doc) — shown in the Story picker / selected-story header.
"""
from __future__ import annotations

from typing import Any

# story name -> story_id (GIF-1..6, + the synchronized compare grid GIF-7, + the
# FB-3 line-race export GIF-8)
STORY_IDS: dict[str, int] = {
    "climb_plateau": 1,
    "warmup_join": 2,
    "loss_drift": 3,
    "bar_race": 4,
    "signal_lives": 5,
    "what_carries": 6,
    "compare_grid": 7,
    "line_race": 8,
    "metric_diff": 9,
}

STORY_REGISTRY: list[dict[str, Any]] = [
    {
        "story_id": 1, "story": "climb_plateau",
        "display_order": 1, "display_label": "GIF-1",
        "label": "Detection climb & plateau",
        "description": ("The detection metric climbing then plateauing over eval epochs "
                        "(the pre-warmup span is greyed; the best epoch is starred)."),
        "allowed_metric_families": ["detection.pak", "detection.point",
                                    "detection.threshold_free", "detection"],
        "default_metric_keys": ["pak_auc_f1"],
        "pair_with": ["f1_score"],
        "overlay_with": ["teacher_pak_auc_f1"],
        "chart_kinds": ["animated_line"],
        "data_axes": ["eval-epoch"],
        "priority": "must", "sub_modes": [],
    },
    {
        "story_id": 2, "story": "warmup_join",
        "display_order": 2, "display_label": "GIF-2",
        "label": "Warmup → student-join (signature shot)",
        "description": ("The warmup → student-join inflection — the separation / SNR "
                        "signature as the student joins after the greyed warmup span."),
        "allowed_metric_families": ["detection", "diagnostic.separation",
                                    "loss.training"],
        "default_metric_keys": ["disc_snr"],
        "pair_with": ["teacher_pak_auc_f1"],
        "overlay_with": [],
        "chart_kinds": ["animated_line"],
        "data_axes": ["eval-epoch"],
        "priority": "must", "sub_modes": [],
    },
    {
        "story_id": 3, "story": "loss_drift",
        "display_order": 3, "display_label": "GIF-3",
        "label": "Loss descent & separation drift",
        "description": ("Reconstruction / discrepancy separation drift over training "
                        "epochs (NPZ-free default; an optional NPZ sub-mode animates the "
                        "per-timestep score-histogram pulling normal vs anomaly apart)."),
        # FB-14: widen the selectable surface to EVERY collected per-training-epoch
        # family so ALL training metrics are reachable — teacher/student reconstruction
        # & discrepancy losses (loss.training), the 18 recon/disc ratio & raw/score
        # separation+context series (diagnostic.separation / diagnostic.context), the
        # GRL adversarial schedules (diagnostic.grl), per-feature arrays
        # (diagnostic.array), timings (meta.timing). The picker lists every POPULATED
        # ``history``-namespace family; the renderer still accepts any history.series
        # key the caller selects.
        "allowed_metric_families": ["loss.training", "diagnostic.separation",
                                    "diagnostic.context", "diagnostic.grl",
                                    "diagnostic.array", "meta.timing",
                                    "meta.count"],
        # RF-1 (D-1): the DEFAULT sub-mode is ``npz_free`` (separation drift over the
        # training-epoch axis), so the declared default must be a real
        # training-history series key. ``reconstruction_loss`` is NOT a
        # training-history series key on any leaf → the first-click default used to
        # render empty. ``epoch_recon_ratio_anomaly`` + ``epoch_disc_ratio_anomaly``
        # are real separation series (the verifier confirmed both render valid
        # GIF89a today). ``_frames_loss_drift`` also has a key-availability fallback.
        "default_metric_keys": ["epoch_recon_ratio_anomaly",
                                "epoch_disc_ratio_anomaly"],
        "chart_kinds": ["animated_line", "animated_hist_drift"],
        "data_axes": ["eval-epoch", "npz", "training-epoch"],
        "priority": "should",
        "sub_modes": [
            {"id": "npz_free", "label": "NPZ-free separation drift",
             "source": "training_history.epoch_{recon,disc}_ratio_*",
             "data_axis": "training-epoch"},
            {"id": "npz_hist", "label": "Per-timestep score histogram drift",
             "source": "epoch_scores/*.npz", "data_axis": "npz"},
        ],
    },
    {
        "story_id": 4, "story": "bar_race",
        "display_order": 4, "display_label": "GIF-4",
        "label": "Cross-experiment bar-chart race",
        "description": ("A best-so-far ranking race across the selected runs — bars "
                        "reorder by the registry sort direction, aligned by epoch number "
                        "(a run missing an epoch stays a gap, never 0)."),
        "allowed_metric_families": ["detection.pak", "detection", "loss.training"],
        "default_metric_keys": ["pak_auc_f1"],
        "chart_kinds": ["bar_chart_race"],
        "data_axes": ["eval-epoch"],
        "priority": "must", "sub_modes": [],
    },
    {
        "story_id": 5, "story": "signal_lives",
        "display_order": 6, "display_label": "GIF-6",
        "label": "Where the signal lives",
        "description": ("Where the signal lives — a per-feature reconstruction-error "
                        "heatmap revealed over epochs, or per-anomaly-type evolution from "
                        "the 2-level pivot (P3-02)."),
        "allowed_metric_families": ["diagnostic.array", "detection.point",
                                    "diagnostic.separation"],
        "default_metric_keys": ["_train_feature_recon_mean"],
        "chart_kinds": ["feature_heatmap", "grouped_bar", "animated_curve"],
        "data_axes": ["eval-epoch", "anomaly-type", "training-epoch"],
        "priority": "could",
        "sub_modes": [
            {"id": "feature_heatmap", "label": "Per-feature error heatmap over epochs",
             "source": "epoch_metrics._*_feature_*", "data_axis": "eval-epoch"},
            {"id": "anom_type_over_training",
             "label": "Per-anomaly-type evolution (over training)",
             "source": "training_history.nested.epoch_anomaly_type_scores",
             "data_axis": "training-epoch", "default_sub_metric": "recon_ratio"},
        ],
    },
    {
        "story_id": 7, "story": "compare_grid",
        # FB-R3-10: HIDDEN from the primary story list (display_order=None) — compare_grid
        # is reached via the side-by-side compare TOGGLE, not a primary card. The display
        # label is a name (not a "GIF-N" position) so it never claims a primary slot.
        "display_order": None, "display_label": "Compare grid",
        "label": "Synchronized side-by-side compare",
        "description": ("Synchronized side-by-side — one combined GIF, one panel per "
                        "pinned run, revealed in lockstep over the union epoch axis "
                        "(reached via the compare toggle, not a primary card)."),
        # one combined GIF: one sub-panel per pinned leaf, revealed in lockstep over the
        # union epoch axis — the synchronized-playhead surface (frontend-design §6.3).
        "allowed_metric_families": ["detection.pak", "detection",
                                    "detection.threshold_free", "loss.training"],
        "default_metric_keys": ["pak_auc_f1"],
        "chart_kinds": ["compare_grid"],
        "data_axes": ["eval-epoch"],
        "priority": "should", "sub_modes": [],
        "synchronized": True,
    },
    {
        "story_id": 8, "story": "line_race",
        # FB-3 export: one LINE per pinned leaf over the union eval-epoch axis; each
        # frame draws every leaf's line UP TO the animated epoch with the value AT that
        # epoch (real per-eval-epoch ``epoch_metrics`` series — NOT best-epoch-fixed),
        # distinct per-leaf colors, 5-epoch native grid (eval_interval=5), non-looping
        # (M-3/FB-9: play once, freeze last frame). The in-app interactive race with
        # seek/stop-at-epoch is the client <LineRace> (RV3); this GIF is the
        # shareable/export artifact. Aligns by epoch NUMBER; gaps stay gaps (never 0).
        "label": "Line race (per-epoch overtaking)",
        "display_order": 5, "display_label": "GIF-5",
        "description": ("A per-epoch line race — each run's REAL per-eval-epoch value; "
                        "lines overtake as they climb, aligned by epoch number and never "
                        "best-epoch-fixed (gaps stay gaps)."),
        "allowed_metric_families": ["detection.pak", "detection",
                                    "detection.threshold_free", "loss.training"],
        "default_metric_keys": ["pak_auc_f1"],
        "chart_kinds": ["animated_line"],
        "data_axes": ["eval-epoch"],
        "priority": "should", "sub_modes": [],
    },
    {
        "story_id": 6, "story": "what_carries",
        "display_order": 7, "display_label": "GIF-7",
        "label": "What carries the detector (score variants)",
        "description": ("What carries the detector — adaptive + teacher as per-epoch "
                        "lines, plus the 4 score variants as best-epoch markers "
                        "(student / disc are best-epoch scalars only, P3-01)."),
        "allowed_metric_families": ["detection.pak", "detection"],
        "default_metric_keys": ["pak_auc_f1"],
        # P3-01: only adaptive(metrics) + teacher_* are per-eval-epoch SERIES.
        "per_epoch_lines": ["metrics", "teacher"],
        # the 4 best-epoch SCALAR markers (never per-epoch student/disc trajectories).
        "best_epoch_markers": ["metrics", "teacher_recon_metrics",
                               "student_recon_metrics", "disc_metrics"],
        "chart_kinds": ["animated_line"],
        "data_axes": ["eval-epoch"],
        "priority": "should", "sub_modes": [],
        "p3_note": "student_*/disc are BEST-EPOCH SCALARS only — 4 markers, never lines",
    },
    {
        "story_id": 9, "story": "metric_diff",
        # FB-R4-02: animate the per-eval-epoch DIFFERENCE of two metrics (A − B), e.g.
        # (pak_auc_f1 − teacher_pak_auc_f1) = the discrepancy's marginal detection
        # contribution per epoch. BOTH series are pulled from the leaf's per-eval-epoch
        # epoch_metrics (F5: any discovered metric pair via params.metric_a/metric_b);
        # A−B is computed per aligned epoch; a GAP (None) is left at any epoch where
        # EITHER series is missing — so a best-epoch-only-scalar metric (student/disc,
        # which has no per-epoch series) yields a fully-gapped diff line, NEVER a
        # fabricated per-epoch value (P3-01). A zero reference line marks A==B.
        "display_order": 8, "display_label": "GIF-8",
        "label": "Metric difference (A − B) over epochs",
        "description": ("The per-eval-epoch DIFFERENCE of two metrics (A − B) — e.g. "
                        "(best − teacher) PA%K-AUC F1 = the discrepancy's marginal "
                        "detection contribution per epoch; a zero line marks A=B and "
                        "epochs where either metric has no per-epoch value stay gaps "
                        "(never fabricated)."),
        "allowed_metric_families": ["detection.pak", "detection",
                                    "detection.threshold_free", "diagnostic.separation",
                                    "loss.training"],
        "default_metric_keys": ["pak_auc_f1", "teacher_pak_auc_f1"],
        # FB-R4-02: the diff is parameterized by an A/B metric pair. The picker reads
        # these param NAMES + the default pair; the renderer accepts params.metric_a /
        # params.metric_b (registry-driven, F5) and falls back to default_metric_keys
        # [0]/[1] (then to the single metric_keys[0] as A and its teacher_* as B).
        "diff_params": {"metric_a": "pak_auc_f1", "metric_b": "teacher_pak_auc_f1"},
        "chart_kinds": ["animated_line"],
        "data_axes": ["eval-epoch"],
        "priority": "should", "sub_modes": [],
    },
]

_BY_NAME = {s["story"]: s for s in STORY_REGISTRY}
_BY_ID = {s["story_id"]: s for s in STORY_REGISTRY}


def get_story(story: str) -> dict[str, Any]:
    if story in _BY_NAME:
        return _BY_NAME[story]
    raise KeyError(f"unknown story {story!r}")
