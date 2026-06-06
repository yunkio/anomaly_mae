/* Dataset Detail (IA-3) — the analytic core. A tabbed workbench (Radix tabs):
 * Trends · Score variants (P3-01) · Anomaly types · Separation · Training-epoch
 * (P2-01) · NPZ scrubber · Animations (6 GIFs scoped here) · Gallery.
 * SWaT full/excl22 toggle binds ?variant=.
 *
 * F-3 (P0-3): on SWaT, the variant DEFAULTS to excl22 (the canonical eval — region-22 is
 * a labeling artifact that inflates the full score). `full` stays selectable (badged
 * "inflated incl. region-22"); the redundant default≡full button is collapsed into a
 * clean 2-way full / excl22 toggle. */
import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import * as Tabs from "@radix-ui/react-tabs";
import { useDataset, useScoreEpochs, useEpochSeries } from "../api/queries";
import { dsUrl } from "../api/client";
import AsyncPanel from "../components/AsyncPanel";
import { Card, StateChip, fmt } from "../components/common";
import ScoreFormulaCard from "../components/ScoreFormulaCard";
import EpochTrendPanel from "../components/panels/EpochTrendPanel";
import ScoreVariantPanel from "../components/panels/ScoreVariantPanel";
import AnomalyTypePanel from "../components/panels/AnomalyTypePanel";
import SeparationPanel from "../components/panels/SeparationPanel";
import TrainingHistoryPanel from "../components/panels/TrainingHistoryPanel";
import InsightsPanel from "../components/panels/InsightsPanel";
import ComponentTrajectoriesPanel from "../components/panels/ComponentTrajectoriesPanel";
import ComponentOverlayPanel from "../components/panels/ComponentOverlayPanel";
import HardestWindowsPanel from "../components/panels/HardestWindowsPanel";
import NpzScrubber from "../components/panels/NpzScrubber";
import PngGallery from "../components/panels/PngGallery";
import GifViewer from "../components/gif/GifViewer";
import { useGifStories } from "../api/queries";
import { useStore } from "../store";
import type { GifSpec, LeafSel } from "../api/types";

export default function DatasetDetail() {
  const { expId, dsKey } = useParams();
  const exp = expId!;
  const dataset = decodeURIComponent(dsKey!).replace(/~/g, "/");
  // F-3 (P0-3): SWaT leaves DEFAULT to excl22 (canonical); other leaves keep null. The
  // SWaT family is recognizable from the dataset key itself (no data round-trip needed).
  const isSwatKey = dataset.toLowerCase().includes("swat");
  const [variant, setVariant] = useState<string | null>(isSwatKey ? "excl22" : null);
  // re-default if the user navigates between datasets without unmounting the route.
  useEffect(() => {
    setVariant(dataset.toLowerCase().includes("swat") ? "excl22" : null);
  }, [exp, dataset]);
  const ds = useDataset(exp, dataset, variant);
  const togglePin = useStore((s) => s.togglePinModel);
  const isModelPinned = useStore((s) => s.isModelPinned);
  useStore((s) => s.pins); // re-render on pin changes

  // F-10 (A-7): an IN-PROGRESS leaf whose eval-epoch Trends are still empty (no detection
  // metric series yet) but which already has per-timestep NPZ epochs should land on the
  // NPZ-scrubber tab — an empty Trends tab is a dead first impression. We probe the eval
  // series (pak_auc_f1) and the score-epoch inventory; once Trends has points the default
  // returns to "trends". (Hooks live in the component body — never in a render-prop.)
  const scoreEpochsQ = useScoreEpochs(exp, dataset, variant);
  const trendProbeQ = useEpochSeries(exp, dataset, ["pak_auc_f1"], variant);
  const trendsEmpty = (trendProbeQ.data?.epochs?.length ?? 0) === 0;
  const hasNpzEpochs = (scoreEpochsQ.data?.epochs?.length ?? 0) > 0;
  const inProgress = ds.data?.state === "in_progress";
  // only flip once BOTH probes have resolved, so the default doesn't flicker mid-load.
  const probesReady = !trendProbeQ.isLoading && !scoreEpochsQ.isLoading;
  const defaultTab = probesReady && inProgress && trendsEmpty && hasNpzEpochs ? "npz" : "trends";
  // a stable key forces <Tabs.Root> to re-mount onto the resolved default exactly once
  // (Radix `defaultValue` is read only on mount; without this it would stay on "trends").
  const tabsKey = `${exp}|${dataset}|${variant ?? "_"}|${defaultTab}`;

  return (
    <div className="col" style={{ gap: 20 }}>
      <AsyncPanel query={ds} emptyLabel="Dataset not found.">
        {(d) => {
          const warmup = d.warmup_epochs ?? null;
          const leaf: LeafSel = { exp_id: exp, dataset_key: dataset, variant };
          const lid = `${exp}|${dataset}|${variant ?? "_"}`;
          const isSwat = dataset.toLowerCase().includes("swat") || (d.present && "metrics_excl_region22" in (d as any));
          return (
            <div className="col" style={{ gap: 20 }}>
              <div>
                <h1 className="view-title">
                  {dataset} <span className="subtle" style={{ fontSize: "1rem" }}>· {exp}</span>
                </h1>
                <div className="row wrap" style={{ gap: 10 }}>
                  <StateChip state={d.state} />
                  {warmup != null && <span className="chip">warmup = {warmup}</span>}
                  {d.num_features != null && <span className="chip">{d.num_features} features</span>}
                  {(d.caveats ?? []).map((c) => (
                    <span className="chip warn" key={c}>
                      {c}
                    </span>
                  ))}
                  <button
                    className={`btn sm ${isModelPinned(exp) ? "primary" : ""}`}
                    onClick={() => togglePin(exp)}
                    title={isModelPinned(exp) ? "unpin this model" : "pin this model (compare it on any dataset in Compare)"}
                  >
                    {isModelPinned(exp) ? "model pinned ✓" : "+ pin model"}
                  </button>
                </div>
                <div style={{ marginTop: 10 }}>
                  <ScoreFormulaCard formula={d.score_formula} />
                </div>
                {/* F-3 (P0-3): SWaT full/excl22 variant toggle (DISC-5) — only on SWaT (or
                    any leaf that actually has sibling variants). The default IS excl22 (the
                    canonical eval), so the redundant "default" button is gone — a clean
                    2-way full / excl22 toggle. `full` is badged "inflated incl. region-22"
                    so the score difference is never read as a real improvement. */}
                <div className="toolbar" style={{ marginTop: 10 }}>
                  {isSwat && (
                    <>
                      <div className="pill-toggle" title="evaluation variant (excl22 is canonical; full includes the region-22 labeling artifact)">
                        <button className={variant === "excl22" ? "on" : ""} onClick={() => setVariant("excl22")} title="canonical eval — region-22 labeling artifact excluded">
                          excl22 · canonical
                        </button>
                        <button className={variant === "full" ? "on" : ""} onClick={() => setVariant("full")} title="includes the region-22 labeling artifact (inflated)">
                          full
                        </button>
                      </div>
                      {variant === "full" && (
                        <span className="chip warn" title="the full-eval score is inflated by the region-22 labeling artifact; excl22 is the canonical comparison">
                          inflated incl. region-22
                        </span>
                      )}
                    </>
                  )}
                  {d.state === "in_progress" && <span className="ribbon">● still running — trends to last epoch</span>}
                </div>
              </div>

              <Tabs.Root key={tabsKey} defaultValue={defaultTab}>
                <Tabs.List className="tabs-list">
                  {[
                    ["trends", "Trends"],
                    ["variants", "Score variants"],
                    ["insights", "Insights"],
                    ["anom", "Anomaly types"],
                    ["sep", "Separation / SNR"],
                    ["history", "Training-epoch"],
                    ["npz", "NPZ scrubber"],
                    ["gifs", "Animations"],
                    ["gallery", "Gallery"],
                  ].map(([v, label]) => (
                    <Tabs.Trigger className="tab-trigger" value={v} key={v}>
                      {label}
                    </Tabs.Trigger>
                  ))}
                </Tabs.List>

                <Tabs.Content value="trends">
                  <Card>
                    <EpochTrendPanel exp={exp} dsKey={dataset} variant={variant} warmup={warmup} />
                  </Card>
                </Tabs.Content>
                <Tabs.Content value="variants">
                  <Card title="What carries the detector (P3-01)">
                    <ScoreVariantPanel exp={exp} dsKey={dataset} variant={variant} warmup={warmup} />
                  </Card>
                </Tabs.Content>
                <Tabs.Content value="insights">
                  {/* EVO-2b FE-B: the flagship insight panels lead the Insights tab —
                      F-5 component detection trajectories (+ I2 efficacy verdict +
                      self-validation), F-6 score-vs-time component overlay, F-8 hardest
                      windows — above the B1/B2/B3/B5 views. All additive; each panel is
                      self-contained (own hooks, AsyncPanel states, honest gaps). */}
                  <div className="col" style={{ gap: 20 }}>
                    <ComponentTrajectoriesPanel exp={exp} dsKey={dataset} variant={variant} warmup={warmup} />
                    <ComponentOverlayPanel exp={exp} dsKey={dataset} variant={variant} />
                    <HardestWindowsPanel exp={exp} dsKey={dataset} variant={variant} />
                    <InsightsPanel exp={exp} dsKey={dataset} variant={variant} warmup={warmup} />
                  </div>
                </Tabs.Content>
                <Tabs.Content value="anom">
                  <Card title="Per-anomaly-type breakdown">
                    <AnomalyTypePanel exp={exp} dsKey={dataset} variant={variant} />
                  </Card>
                </Tabs.Content>
                <Tabs.Content value="sep">
                  <Card title="Separation / SNR diagnostics">
                    <SeparationPanel exp={exp} dsKey={dataset} variant={variant} />
                  </Card>
                </Tabs.Content>
                <Tabs.Content value="history">
                  <Card title="Training-epoch history (P2-01)">
                    <TrainingHistoryPanel exp={exp} dsKey={dataset} variant={variant} warmup={warmup} />
                  </Card>
                </Tabs.Content>
                <Tabs.Content value="npz">
                  <Card title="Per-timestep NPZ scrubber">
                    <NpzScrubber exp={exp} dsKey={dataset} variant={variant} warmup={warmup} />
                  </Card>
                </Tabs.Content>
                <Tabs.Content value="gifs">
                  <DatasetGifs leaf={leaf} />
                </Tabs.Content>
                <Tabs.Content value="gallery">
                  <Card title="Pre-rendered visualizations">
                    <PngGallery exp={exp} dsKey={dataset} variant={variant} />
                  </Card>
                </Tabs.Content>
              </Tabs.Root>
            </div>
          );
        }}
      </AsyncPanel>
    </div>
  );
}

/* The flagship GIFs scoped to this leaf (frontend-design §6.4 embedded). */
function DatasetGifs({ leaf }: { leaf: LeafSel }) {
  const stories = useGifStories();
  // single-leaf stories only: bar_race (cross-run) + compare_grid (multi-panel) excluded.
  // R3-SR-01 / FB-R3-10: this is the SECOND render site of the story label — sort by
  // display_order and render display_label (the same fix as GifStudio) so line_race is no
  // longer "GIF-8" between 5 and 6 here either; surface the description too (FB-R3-11).
  const single = (stories.data?.stories ?? [])
    .filter((s) => s.story !== "bar_race" && s.story !== "compare_grid" && s.display_order != null)
    .slice()
    .sort((a, b) => (a.display_order ?? 999) - (b.display_order ?? 999));
  return (
    <div className="card-grid">
      {single.map((s) => {
        const params: Record<string, any> = {};
        if (s.sub_modes?.[0]) {
          params.sub_mode = s.sub_modes[0].id;
          if (s.sub_modes[0].default_sub_metric) params.sub_metric = s.sub_modes[0].default_sub_metric;
        }
        const spec: GifSpec = {
          story: s.story,
          metric_keys: s.default_metric_keys.length ? s.default_metric_keys : ["pak_auc_f1"],
          experiment_set: [leaf],
          variant: leaf.variant,
          max_epoch: null,
          params,
        };
        return (
          <Card title={`${s.display_label ?? `GIF-${s.display_order ?? s.story_id}`} · ${s.label}`} key={s.story}>
            {s.description && (
              <div className="subtle" style={{ fontSize: "var(--fs-small)", marginBottom: 6, whiteSpace: "normal" }}>
                {s.description}
              </div>
            )}
            <GifViewer spec={spec} height={280} autoRender={false} />
            <div className="subtle" style={{ fontSize: "var(--fs-small)", marginTop: 6 }}>
              {s.allowed_metric_families.join(", ")}
            </div>
          </Card>
        );
      })}
    </div>
  );
}
