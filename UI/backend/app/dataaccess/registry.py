"""Metric-semantics registry loader + resolver + self-test gate.

backend-design.md §3.2/§3.3. This is the F5 heart: NO hard-coded metric list — every
key is resolved registry-entry-first, then via the registry's ordered fallback rules.

The registry's own ``resolver_self_test`` suite is IMPLEMENTED and RUN here as a
startup/CI gate (``run_self_test``); ``/health`` goes red on any FAIL.

Resolution precedence (per registry ``resolver_self_test.note``):
  1. explicit ``entries[full_key]``         (some explicit entries are namespaced,
     e.g. ``loss_stats.recon_SNR``)
  2. explicit ``entries[strip_namespace(key)]``
  3. ordered ``fallback.rules`` (first firing rule wins) -> inferred=True
  4. ``fallback.default_entry``              -> inferred=True
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .models import MetricMeta

# Namespace prefixes the resolver strips before an entries[] lookup. These are
# CONTAINER names (the stable part of the contract), not metric keys.
_NS_PREFIXES = ("loss_stats.", "history.", "anomaly_type.", "score_variant:")


def strip_namespace(key: str) -> str:
    """Strip a known container-namespace prefix to get the bare metric key."""
    for p in _NS_PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


class _CompiledRule:
    __slots__ = ("id", "substrings", "regex", "direction", "family", "unit")

    def __init__(self, raw: dict[str, Any]):
        self.id: str = raw["id"]
        self.substrings: list[str] = raw.get("match_substring", []) or []
        rx = raw.get("match_regex")
        self.regex: Optional[re.Pattern] = re.compile(rx) if rx else None
        self.direction: str = raw.get("direction", "neutral")
        self.family: str = raw.get("family", "uncategorized")
        self.unit: str = raw.get("unit", "mixed")

    def fires(self, key: str) -> bool:
        if any(sub in key for sub in self.substrings):
            return True
        if self.regex is not None and self.regex.search(key):
            return True
        return False


class MetricRegistry:
    """Loaded registry + the single ``resolve()`` chokepoint."""

    def __init__(self, doc: dict[str, Any]):
        self._doc = doc
        self.version: str = doc.get("_meta", {}).get("version", "?")
        self.entries: dict[str, dict] = doc.get("entries", {})
        self.primary: str = doc.get("_meta", {}).get("primary_selection_metric", "pak_auc_f1")

        fb = doc.get("fallback", {})
        self._default_entry: dict = fb.get("default_entry", {})
        rules_by_id = {r["id"]: _CompiledRule(r) for r in fb.get("rules", [])}
        # rule_order may name 'default_entry' as a terminal sentinel — keep only real rules.
        self._ordered_rules: list[_CompiledRule] = [
            rules_by_id[rid] for rid in fb.get("rule_order", []) if rid in rules_by_id
        ]
        self._self_test = doc.get("resolver_self_test", {})
        self._anom_type_sem = doc.get("anomaly_type_metric_semantics", {})
        # tiny memoization keyed on the resolution inputs.
        self._cache: dict[tuple[str, str], MetricMeta] = {}

    # ── classmethods ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: Path) -> "MetricRegistry":
        with open(path, "rb") as f:
            doc = json.loads(f.read().decode("utf-8"))
        return cls(doc)

    # ── the resolver ─────────────────────────────────────────────────────────
    def _fallback(self, key: str) -> tuple[str, dict]:
        """Return (rule_id, synth_entry) from the ordered fallback rules."""
        for rule in self._ordered_rules:
            if rule.fires(key):
                return rule.id, {
                    "family": rule.family,
                    "direction": rule.direction,
                    "unit": rule.unit,
                    "phase_validity": "all",
                    "viz_hint": {"chart": "animated_line", "direction": rule.direction,
                                 "group": rule.family, "inferred": True},
                }
        de = self._default_entry
        return "default_entry", {
            "family": de.get("family", "uncategorized"),
            "direction": de.get("direction", "neutral"),
            "unit": de.get("unit", "mixed"),
            "phase_validity": de.get("phase_validity", "all"),
            "viz_hint": de.get("viz_hint", {"chart": "animated_line",
                                            "direction": "neutral",
                                            "group": "uncategorized", "inferred": True}),
        }

    def fallback_rule_id(self, key: str) -> str:
        """Public helper for the self-test: which rule fires for ``key`` (fallback-only)."""
        return self._fallback(key)[0]

    def resolve(self, key: str, *, namespace: str = "epoch_metrics") -> MetricMeta:
        """Resolve a single metric key to its MetricMeta (registry-first, else fallback).

        ``namespace`` is the container the key came from; it is recorded on the meta
        and used for anomaly-type sub-metric semantics, but the lookup tries the
        explicit entry under BOTH the full key and the namespace-stripped key first.
        """
        ck = (key, namespace)
        cached = self._cache.get(ck)
        if cached is not None:
            return cached

        bare = strip_namespace(key)
        entry: Optional[dict] = None
        inferred = False
        rule_id: Optional[str] = None

        # 1/2: explicit entry under the full key, then the stripped key.
        for lookup in (key, bare):
            if lookup in self.entries:
                entry = self.entries[lookup]
                break

        # anomaly-type sub-metrics resolve via the dedicated table when no entry.
        if entry is None and namespace == "anomaly_type" and bare in self._anom_type_sem:
            sem = self._anom_type_sem[bare]
            entry = {
                "display_name": bare,
                "family": sem.get("family", "uncategorized"),
                "direction": sem.get("direction", "neutral"),
                "phase_validity": "all",
                "unit": "mixed",
                "viz_hint": {"chart": "animated_line", "direction": sem.get("direction", "neutral"),
                             "group": sem.get("family", "uncategorized")},
            }

        # 3/4: fallback rules / default.
        if entry is None:
            rule_id, entry = self._fallback(bare)
            inferred = True

        meta = MetricMeta(
            key=key,
            display_name=entry.get("display_name", bare),
            family=entry.get("family", "uncategorized"),
            direction=entry.get("direction", "neutral"),
            phase_validity=entry.get("phase_validity", "all"),
            unit=entry.get("unit", "mixed"),
            description=entry.get("description", ""),
            viz_hint=dict(entry.get("viz_hint", {})),
            namespace=namespace,
            inferred=inferred,
            inferred_direction=inferred,
            rule_id=rule_id,
        )
        self._cache[ck] = meta
        return meta

    # ── the self-test gate (§3.3) ─────────────────────────────────────────────
    def run_self_test(self) -> dict[str, Any]:
        """Execute the registry's ``resolver_self_test`` suite against THIS resolver.

        Returns ``{status: PASS|FAIL, groups: {...}, failures: [...]}``. Mirrors the
        precedence rule in the registry note: a key with an EXPLICIT entry is exempt
        from the fallback assertions (its explicit direction governs); the
        assertions reason about the FALLBACK path only, so we use ``fallback_rule_id``.
        """
        failures: list[str] = []
        group_results: dict[str, dict] = {}
        st = self._self_test

        def _check(group: str, expect_rule: Optional[str], expect_not: Optional[str],
                   keys: list[str], expect_direction: Optional[str] = None):
            bad: list[str] = []
            for k in keys:
                bare = strip_namespace(k)
                rid = self.fallback_rule_id(bare)
                if expect_rule is not None and rid != expect_rule:
                    bad.append(f"{k}->{rid} (want {expect_rule})")
                if expect_not is not None and rid == expect_not:
                    bad.append(f"{k}->{rid} (must NOT be {expect_not})")
                if expect_direction is not None:
                    _, synth = self._fallback(bare)
                    if synth["direction"] != expect_direction:
                        bad.append(f"{k} dir={synth['direction']} (want {expect_direction})")
            group_results[group] = {"n": len(keys), "fail": bad}
            if bad:
                failures.extend(f"{group}: {b}" for b in bad)

        g = st.get("assert_no_loss_magnitude_in_explicit_neutral", {})
        _check("no_loss_magnitude_in_explicit_neutral", None, "explicit_neutral", g.get("keys", []))

        g = st.get("assert_count_keys_still_neutral", {})
        _check("count_keys_still_neutral", "explicit_neutral", None, g.get("keys", []),
               expect_direction="neutral")

        g = st.get("assert_bare_recon_not_forced_loss", {})
        _check("bare_recon_not_forced_loss", "default_entry", None, g.get("keys", []))

        g = st.get("assert_separation_still_up", {})
        _check("separation_still_up", "snr_separation", None, g.get("keys", []),
               expect_direction="up")

        g = st.get("assert_detection_still_up", {})
        _check("detection_still_up", "higher_better_detection", None, g.get("keys", []),
               expect_direction="up")

        status = "PASS" if not failures else "FAIL"
        return {"status": status, "groups": group_results, "failures": failures,
                "registry_version": self.version}
