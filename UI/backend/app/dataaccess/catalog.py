"""Dynamic F5 metric catalog — backend-design.md §3.1.

Build the metric key set by UNION over:
  1. every ``epoch_metrics.epochs[].<key>`` (UNION over ALL entries, not the first)
  2. the 4 (+conditional) ``experiment_metadata`` score-variant blocks
  3. ``experiment_metadata.loss_stats.<key>``
  4. ALL ``training_histories['0']`` flat lists + the nested anomaly-type sub-metrics
     (P2-01) — kept on the separate ``history.`` namespace
  5. ``anomaly_type_metrics.json`` per-type metric keys (dynamic vocabulary)

Each key is then resolved registry-entry-first, else via the ordered fallback rules.
NO hard-coded metric list anywhere — the catalog is whatever the data contains.

P3-03 NOTE (key_namespaces.history doc extension, applied here in code): the
non-``train_*`` per-training-epoch families carry their resolution namespace below so
the catalog DOCUMENTS where each key resolves, even while the registry JSON's
``key_namespaces.history`` text still only names ``train_*`` (the registry doc-touch
is RT-1, non-blocking). The functional resolution already works via fallback.rules.
"""
from __future__ import annotations

from typing import Any, Optional

from .models import (
    BestSnapshot,
    EpochSeries,
    MetricMeta,
    TrainingHistory,
)
from .registry import MetricRegistry

# Documented namespace -> human note for the non-train_* history families (P3-03).
# This is the in-code realization of the RT-1 registry doc extension; it does NOT
# change resolution (which is registry-first then fallback), only documents it.
HISTORY_NAMESPACE_DOCS: dict[str, str] = {
    "train_*": "per-training-epoch training curves (lower-better losses, neutral schedules)",
    "epoch_recon_ratio_*": "diagnostic.context — recon component contribution share (recon/(recon+disc)) per sample type; context (FB-R4-03: NOT an a/n ratio)",
    "epoch_disc_ratio_*": "diagnostic.context — disc component contribution share (disc/(recon+disc)) per sample type; context (FB-R4-03: NOT an a/n ratio)",
    "epoch_recon_score_*": "neutral context (per-training-epoch recon magnitude)",
    "epoch_disc_score_*": "neutral context (per-training-epoch disc magnitude)",
    "epoch_raw_recon_*": "neutral context (raw recon)",
    "epoch_raw_disc_*": "neutral context (raw disc)",
    "epoch_anomaly_type_scores.<type>.recon_ratio|disc_ratio": "diagnostic.separation — up",
    "epoch_anomaly_type_scores.<type>.recon_score|disc_score|count": "neutral context",
    "epoch_timings": "meta.timing",
}


class CatalogEntry(MetricMeta):
    """A MetricMeta plus the source namespace tag it was discovered under."""


def _union_epoch_keys(epoch: Optional[EpochSeries]) -> set[str]:
    if epoch is None:
        return set()
    return set(epoch.series.keys()) | set(epoch.array_keys)


def _union_metadata_keys(meta: Optional[BestSnapshot]) -> dict[str, set[str]]:
    """Return {namespace: keyset} for the 4 score blocks + loss_stats."""
    out: dict[str, set[str]] = {}
    if meta is None:
        return out
    for block, kv in meta.score_variants.items():
        out[f"score_variant:{block}"] = set(kv.keys())
    if meta.loss_stats:
        out["loss_stats"] = set(meta.loss_stats.keys())
    return out


def _union_history_keys(hist: Optional[TrainingHistory]) -> set[str]:
    if hist is None:
        return set()
    keys: set[str] = set(hist.series.keys()) | set(hist.empty_keys)
    # nested anomaly-type sub-metrics: namespaced history.<key>.<sub_metric>
    for nk, piv in hist.nested.items():
        for sm in piv.sub_metrics:
            keys.add(f"{nk}.{sm}")
    return keys


def _union_anomaly_type_keys(atm: Optional[dict[str, dict]]) -> set[str]:
    if not atm:
        return set()
    keys: set[str] = set()
    for _type, metrics in atm.items():
        if isinstance(metrics, dict):
            keys |= set(metrics.keys())
    return keys


# M-1 (R2-01/R2-09/FB-6): the excl22 PREFIX phenomenon lives ONLY in the
# epoch_metrics namespace of the SWaT-*full* leaf (verified 145 keys); the prefix
# encodes the SWaT eval VARIANT, not a different metric. We strip + re-resolve there
# so the entry inherits the proper display/direction/family AND tag it
# variant_scope="excl22". The raw ``.key`` is kept (F5). variant_scope tagging needs
# LEAF PROVENANCE — which only the catalog build (where the leaf is known) has — so
# this is BACKEND-ONLY (R2-01: the frontend must NOT re-derive provenance).
_EXCL22_PREFIX = "excl22_"


def build_catalog(
    registry: MetricRegistry,
    *,
    epoch: Optional[EpochSeries] = None,
    metadata: Optional[BestSnapshot] = None,
    history: Optional[TrainingHistory] = None,
    anomaly_types: Optional[dict[str, dict]] = None,
    leaf_variant_scope: str = "default",
) -> list[CatalogEntry]:
    """UNION the metric set over all sources and resolve each key (F5).

    Returns one CatalogEntry per (key, namespace) pairing discovered. The same bare
    key under two namespaces (e.g. ``pak_auc_f1`` in epoch_metrics and in a score
    variant) is listed once per namespace so the picker can scope by source.

    ``leaf_variant_scope`` (M-1, ADDITIVE, default "default") is the SWaT eval variant
    of THE LEAF this catalog is built from:
      * "default"  — non-SWaT leaf (or a SWaT leaf with no variant suffix);
      * "full"     — the SWaT-full leaf (its plain keys ARE the full eval);
      * "excl22"   — the SWaT-excl22 leaf (its plain keys ARE the excl22 eval).
    On a "full" leaf, an ``epoch_metrics`` key starting ``excl22_`` is the SWaT excl22
    eval embedded in the full leaf: we STRIP the prefix, RE-RESOLVE the stripped bare
    key, and tag the resulting entry variant_scope="excl22" — while keeping the entry's
    raw ``.key`` literal (so ``/api/metrics-catalog`` still serves ``excl22_*``; F5).
    A non-prefixed key on a "full" leaf is variant_scope="full". All other entries
    inherit ``leaf_variant_scope`` ("default" preserves the legacy behaviour exactly).
    """
    entries: list[CatalogEntry] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(key: str, namespace: str) -> None:
        # Decide the per-entry variant_scope + the key actually resolved.
        scope = leaf_variant_scope
        resolve_key = key
        if (namespace == "epoch_metrics" and leaf_variant_scope == "full"
                and key.startswith(_EXCL22_PREFIX)):
            # excl22 embedded in the SWaT-full leaf: strip + re-resolve, tag excl22.
            resolve_key = key[len(_EXCL22_PREFIX):]
            scope = "excl22"
        sig = (key, namespace, scope)
        if sig in seen:
            return
        seen.add(sig)
        # Resolve under the STRIPPED key so display/direction/family/viz_hint are
        # proper (not the inferred ``excl22_*`` fallback); then restore the RAW key
        # on the emitted entry so F5 (literal-key serve) holds.
        meta = registry.resolve(resolve_key, namespace=namespace)
        payload = meta.model_dump()
        payload["key"] = key            # keep the raw catalog key (F5)
        payload["variant_scope"] = scope
        entries.append(CatalogEntry(**payload))

    for k in sorted(_union_epoch_keys(epoch)):
        _add(k, "epoch_metrics")
    for namespace, keyset in _union_metadata_keys(metadata).items():
        for k in sorted(keyset):
            _add(k, namespace)
    for k in sorted(_union_history_keys(history)):
        _add(k, "history")
    for k in sorted(_union_anomaly_type_keys(anomaly_types)):
        _add(k, "anomaly_type")

    return entries


def global_catalog(per_leaf_catalogs: list[list[CatalogEntry]]) -> list[CatalogEntry]:
    """UNION across leaves for the Overview metric picker (a key dropped in a newer
    run simply drops from that run's leaf catalog; the global set is the union).

    The dedup signature includes ``variant_scope`` (M-1) so the SWaT-full plain
    ``affiliation_f1`` (scope "full") and the excl22-stripped ``excl22_affiliation_f1``
    (scope "excl22") both survive into the union as DISTINCT entries — they carry
    DIFFERENT numbers and must never silently merge (the R2-01 data-loss guard). For a
    "default"-scope key the signature collapses to the legacy ``(key, namespace)`` pair
    (default-scope keys from different leaves still dedup to one entry, unchanged)."""
    by_sig: dict[tuple[str, str, str], CatalogEntry] = {}
    for cat in per_leaf_catalogs:
        for e in cat:
            by_sig.setdefault((e.key, e.namespace, e.variant_scope), e)
    return [by_sig[k] for k in sorted(by_sig)]
