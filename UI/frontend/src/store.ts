/* Global cross-view state.
 *
 * Pins are MODELS (experiment ids), not model-dataset pairs (2026-06-04 refactor):
 * you pin the runs you care about, then pick ONE dataset to compare them on. The
 * dataset lives in a shared "scope" so the same models stay compared as you move
 * Compare -> GIF -> Config. Pins/theme/scope persist to IndexedDB (UTIL-2, never to
 * results/, G1). Categorical colors are assigned by pin order so a model is the same
 * color in every view (§5; viz Story 4). */
import { create } from "zustand";
import { get as idbGet, set as idbSet } from "idb-keyval";
import type { LeafSel } from "./api/types";

export const CAT_VARS = [
  "--cat-1",
  "--cat-2",
  "--cat-3",
  "--cat-4",
  "--cat-5",
  "--cat-6",
  "--cat-7",
  "--cat-8",
  "--cat-9",
  "--cat-10",
  "--cat-11",
  "--cat-12",
] as const;

/** stable leaf id used to zip backend compare rows (exp|dataset|variant). */
export function leafId(l: LeafSel): string {
  return `${l.exp_id}|${l.dataset_key}|${l.variant ?? "_"}`;
}

export function catColor(index: number): string {
  const v = CAT_VARS[index % CAT_VARS.length];
  return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888";
}

type Theme = "light" | "dark";

/** Performance basis (PERF_BASIS_SPEC) — which epoch snapshot + which threshold the
 * perf TABLES (rankings / leaderboard / compare matrix) resolve their values under.
 * Default (best, optimal) = current behaviour, byte-identical. */
export type EpochBasis = "best" | "best_post" | "es" | "last" | "300" | "350" | "400" | "450";
export type ThresholdBasis = "optimal" | "anomaly_ratio";

interface AppState {
  theme: Theme;
  liveRefresh: boolean;
  /** ordered list of pinned MODELS (exp_id). order = color order. */
  pins: string[];
  /** shared comparison scope: which dataset the pinned models are compared on. */
  scopeDatasetKey: string | null;
  scopeVariant: string | null;
  metricPick: string[];
  /** perf basis (default best/optimal = current). persisted to tsmae.perfBasis. */
  epochBasis: EpochBasis;
  thresholdBasis: ThresholdBasis;
  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
  setLiveRefresh: (v: boolean) => void;
  pinModel: (expId: string) => void;
  unpinModel: (expId: string) => void;
  togglePinModel: (expId: string) => void;
  clearPins: () => void;
  isModelPinned: (expId: string) => boolean;
  modelIndex: (expId: string) => number;
  setScope: (datasetKey: string | null, variant: string | null) => void;
  setMetricPick: (keys: string[]) => void;
  setEpochBasis: (b: EpochBasis) => void;
  setThresholdBasis: (b: ThresholdBasis) => void;
  hydrate: () => Promise<void>;
}

/** Migrate a persisted pin payload to the model-only shape. Old builds stored
 * `LeafSel[]` (model-dataset-variant); collapse to unique exp_id, first-seen order.
 * New builds store `string[]`; pass through. Never throws on junk. */
function migratePins(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  const out: string[] = [];
  const seen = new Set<string>();
  for (const item of raw) {
    let id: string | null = null;
    if (typeof item === "string") id = item;
    else if (item && typeof item === "object" && typeof (item as any).exp_id === "string")
      id = (item as any).exp_id;
    if (id && !seen.has(id)) {
      seen.add(id);
      out.push(id);
    }
  }
  return out;
}

export const useStore = create<AppState>((set, getState) => ({
  theme: "light",
  liveRefresh: false,
  pins: [],
  scopeDatasetKey: null,
  scopeVariant: null,
  metricPick: [],
  epochBasis: "best",
  thresholdBasis: "optimal",

  setTheme: (theme) => {
    document.documentElement.setAttribute("data-theme", theme);
    void idbSet("tsmae.theme", theme);
    set({ theme });
  },
  toggleTheme: () => {
    const t = getState().theme === "light" ? "dark" : "light";
    getState().setTheme(t);
  },
  setLiveRefresh: (liveRefresh) => set({ liveRefresh }),

  pinModel: (expId) => {
    if (!expId || getState().pins.includes(expId)) return;
    const pins = [...getState().pins, expId];
    void idbSet("tsmae.pins", pins);
    set({ pins });
  },
  unpinModel: (expId) => {
    const pins = getState().pins.filter((p) => p !== expId);
    void idbSet("tsmae.pins", pins);
    set({ pins });
  },
  togglePinModel: (expId) => {
    if (getState().pins.includes(expId)) getState().unpinModel(expId);
    else getState().pinModel(expId);
  },
  clearPins: () => {
    void idbSet("tsmae.pins", []);
    set({ pins: [] });
  },
  isModelPinned: (expId) => getState().pins.includes(expId),
  modelIndex: (expId) => getState().pins.indexOf(expId),

  setScope: (scopeDatasetKey, scopeVariant) => {
    void idbSet("tsmae.scope", { datasetKey: scopeDatasetKey, variant: scopeVariant });
    set({ scopeDatasetKey, scopeVariant });
  },
  setMetricPick: (metricPick) => set({ metricPick }),

  setEpochBasis: (epochBasis) => {
    void idbSet("tsmae.perfBasis", { epochBasis, thresholdBasis: getState().thresholdBasis });
    set({ epochBasis });
  },
  setThresholdBasis: (thresholdBasis) => {
    void idbSet("tsmae.perfBasis", { epochBasis: getState().epochBasis, thresholdBasis });
    set({ thresholdBasis });
  },

  hydrate: async () => {
    const theme = ((await idbGet("tsmae.theme")) as Theme) || "light";
    const pins = migratePins(await idbGet("tsmae.pins"));
    // persist the migrated shape so a refresh never re-migrates the same junk.
    void idbSet("tsmae.pins", pins);
    const scope =
      ((await idbGet("tsmae.scope")) as { datasetKey: string | null; variant: string | null } | undefined) || {
        datasetKey: null,
        variant: null,
      };
    const perfBasis =
      ((await idbGet("tsmae.perfBasis")) as
        | { epochBasis: EpochBasis; thresholdBasis: ThresholdBasis }
        | undefined) || { epochBasis: "best", thresholdBasis: "optimal" };
    document.documentElement.setAttribute("data-theme", theme);
    set({
      theme,
      pins,
      scopeDatasetKey: scope.datasetKey ?? null,
      scopeVariant: scope.variant ?? null,
      epochBasis: perfBasis.epochBasis ?? "best",
      thresholdBasis: perfBasis.thresholdBasis ?? "optimal",
    });
  },
}));
