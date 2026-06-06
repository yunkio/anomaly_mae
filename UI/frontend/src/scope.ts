/* Shared "comparison scope" (2026-06-04 pin refactor).
 *
 * Pins are MODELS. Any comparison view = (selected models) × (ONE dataset). This
 * module owns the two pieces of that idea:
 *   • useScopeDataset(models)  — the datasets the selected models have IN COMMON,
 *     plus the currently-chosen dataset/variant (persisted in the store, validated
 *     against the live intersection so a stale choice never yields a dead request).
 *   • useCompareSelection()    — which models are being compared (defaults to the
 *     pinned set, with a lightweight include/exclude + add), and the ready-to-send
 *     `leaves` built from (models × scope dataset). Every comparison view consumes
 *     this so the SAME models/dataset stay selected across Compare / GIF / Config.
 */
import { useMemo, useState } from "react";
import { useExperiments } from "./api/queries";
import { useStore } from "./store";
import type { ExperimentRef, LeafSel } from "./api/types";

export interface ScopeOption {
  dataset_key: string;
  dataset_key_url: string;
  variants: string[];
}

// Datasets a viewer most likely wants to compare on, surfaced first in the dropdown.
const DATASET_PRIORITY = ["SWaT_A1A2", "PSM", "WaDi_A1", "WaDi_A2"];

/** The canonical default variant for a dataset: SWaT compares on excl22 (region-22 is
 * an inflated outlier per DECISION_LOG); everything else has the single "_" variant. */
export function preferredVariant(variants: string[]): string {
  if (variants.includes("excl22")) return "excl22";
  return variants[0] ?? "_";
}

/** A variant is "real" (worth a chip) only when a dataset has more than one. */
export function hasRealVariants(variants: string[]): boolean {
  return variants.filter((v) => v && v !== "_").length > 1;
}

/** A compact, human label for a model id:
 *  "[prefix_]288_20260603_094631_no_focal" -> "288·no_focal" (keeps new_/old_/legacy_
 *  prefix). The suffix is optional → "legacy_271_20260508_094241" -> "legacy·271". */
export function shortModel(expId: string): string {
  const m = expId.match(/^(?:([A-Za-z]+)_)?(\d+)_\d{8}_\d{6}(?:_(.+))?$/);
  if (!m) return expId;
  const [, prefix, num, suffix] = m;
  return `${prefix ? prefix + "·" : ""}${num}${suffix ? "·" + suffix : ""}`;
}

export interface ScopeResult {
  options: ScopeOption[];
  datasetKey: string | null;
  variant: string | null;
  current: ScopeOption | null;
  setDatasetKey: (k: string) => void;
  setVariant: (v: string) => void;
  loading: boolean;
  /** models≥1 but they share no dataset → callers show an explicit message, not 0. */
  noCommon: boolean;
}

/** The datasets `models` have in common + the chosen dataset/variant. */
export function useScopeDataset(models: string[]): ScopeResult {
  const expsQ = useExperiments();
  const scopeDatasetKey = useStore((s) => s.scopeDatasetKey);
  const scopeVariant = useStore((s) => s.scopeVariant);
  const setScope = useStore((s) => s.setScope);
  const modelKey = models.join("|");

  const options = useMemo<ScopeOption[]>(() => {
    const exps = expsQ.data ?? [];
    if (models.length === 0) return [];
    const byId = new Map<string, ExperimentRef>(exps.map((e) => [e.exp_id, e]));
    const selected = models.map((m) => byId.get(m)).filter(Boolean) as ExperimentRef[];
    if (selected.length === 0) return [];
    let common: string[] | null = null; // intersection of dataset_keys across models
    const meta = new Map<string, ScopeOption>();
    for (const e of selected) {
      const here = new Set<string>();
      for (const ds of e.datasets ?? []) {
        here.add(ds.dataset_key);
        if (!meta.has(ds.dataset_key))
          meta.set(ds.dataset_key, {
            dataset_key: ds.dataset_key,
            dataset_key_url: ds.dataset_key_url ?? ds.dataset_key.replace(/\//g, "~"),
            variants: ds.variants ?? ["_"],
          });
      }
      common = common === null ? [...here] : common.filter((k) => here.has(k));
    }
    const ordered = (common ?? []).slice();
    ordered.sort((a, b) => {
      const ra = DATASET_PRIORITY.indexOf(a),
        rb = DATASET_PRIORITY.indexOf(b);
      return (ra < 0 ? 999 : ra) - (rb < 0 ? 999 : rb) || a.localeCompare(b);
    });
    return ordered.map((k) => meta.get(k)!);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expsQ.data, modelKey]);

  const datasetKey = useMemo(() => {
    if (options.length === 0) return null;
    if (scopeDatasetKey && options.some((o) => o.dataset_key === scopeDatasetKey)) return scopeDatasetKey;
    return options[0].dataset_key;
  }, [options, scopeDatasetKey]);

  const current = options.find((o) => o.dataset_key === datasetKey) || null;

  const variant = useMemo(() => {
    if (!current) return null;
    if (scopeVariant && current.variants.includes(scopeVariant)) return scopeVariant;
    return preferredVariant(current.variants);
  }, [current, scopeVariant]);

  return {
    options,
    datasetKey,
    variant,
    current,
    setDatasetKey: (k) => {
      const opt = options.find((o) => o.dataset_key === k);
      setScope(k, opt ? preferredVariant(opt.variants) : null);
    },
    setVariant: (v) => setScope(datasetKey, v),
    loading: expsQ.isLoading,
    noCommon: models.length > 0 && options.length === 0 && !expsQ.isLoading,
  };
}

export interface CompareSelection {
  /** models actually included in the comparison (pins minus excluded). */
  models: string[];
  /** all pinned models (the candidate pool shown with checkboxes). */
  candidates: string[];
  isIncluded: (expId: string) => boolean;
  toggleInclude: (expId: string) => void;
  scope: ScopeResult;
  /** ready-to-POST leaves = models × the scope dataset (empty until a dataset exists). */
  leaves: LeafSel[];
}

/** The shared model+dataset selection backing every comparison view. Models default
 * to the pinned set; an include/exclude toggle lets you narrow to "just these 2"
 * without unpinning. `leaves` is what the compare/GIF/config endpoints consume. */
export function useCompareSelection(): CompareSelection {
  const pins = useStore((s) => s.pins);
  // excluded = pinned models the user unchecked for THIS comparison (new pins are
  // included by default; nothing is excluded until the user opts a model out).
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const models = useMemo(() => pins.filter((p) => !excluded.has(p)), [pins, excluded]);
  const scope = useScopeDataset(models);

  const leaves = useMemo<LeafSel[]>(() => {
    if (!scope.datasetKey) return [];
    return models.map((m) => ({ exp_id: m, dataset_key: scope.datasetKey as string, variant: scope.variant }));
  }, [models, scope.datasetKey, scope.variant]);

  return {
    models,
    candidates: pins,
    isIncluded: (id) => !excluded.has(id),
    toggleInclude: (id) =>
      setExcluded((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      }),
    scope,
    leaves,
  };
}
