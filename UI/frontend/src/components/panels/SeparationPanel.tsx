/* <SeparationPanel> (DS-2) — the recon / disc magnitude pairs as a NEUTRAL pair
 * (NOT lower-better losses, P1-02) + the ratio / SNR / Cohen's-d as the ranked-up
 * signal; negative SNR warn-hued (AC-F4.6).
 *
 * F-7 (V-2): the magnitudes are now FACETED BY FAMILY — recon and disc get SEPARATE
 * sub-charts, EACH WITH ITS OWN Y-SCALE (recon and disc magnitudes live on very
 * different scales; one shared axis flattens whichever is smaller). Within each facet
 * the normal-vs-anomaly magnitudes are grouped into EXPLICIT pairs (one grouped bar per
 * "subject" — normal / pure-normal / disturbing / anomaly — with normal and anomaly
 * sitting side by side) so "anomaly > normal" reads at a glance. P1-02 neutral labeling
 * is kept: the magnitude is CONTEXT (anomaly > normal is GOOD, not a lower-better loss);
 * the ranking signal is the ratio / SNR / Cohen's-d below. */
import { useMemo } from "react";
import ChartFrame from "../charts/ChartFrame";
import { useSeparation } from "../../api/queries";
import AsyncPanel from "../AsyncPanel";
import { baseLayout, PLOTLY_CONFIG } from "../../viz/plotlyTemplate";
import type { Separation } from "../../api/types";

const tok = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

type Pair = Separation["pairs"][number];

/* derive the family facet for a magnitude key. Prefer the registry-supplied `family`
 * (e.g. "reconstruction"/"discrepancy"); fall back to the key's leading token. The
 * facet bucket is a coarse recon | disc | other so the two engines split cleanly. */
function facetOf(p: Pair): string {
  const fam = (p.family || "").toLowerCase();
  const key = p.key.toLowerCase();
  if (fam.includes("recon") || key.startsWith("recon") || key.startsWith("reconstruction")) return "recon";
  if (fam.includes("disc") || key.startsWith("disc") || key.startsWith("discrepancy")) return "disc";
  return "other";
}

/* the "subject" a magnitude belongs to (the x-group within a facet), and whether it is a
 * std band (kept separate so a mean and its std don't share a bar group). Strips the
 * family prefix and any trailing _std. */
function subjectOf(key: string): { subject: string; isStd: boolean } {
  let s = key.replace(/^recon_|^reconstruction_|^disc_|^discrepancy_/i, "");
  const isStd = /_std$/i.test(s);
  s = s.replace(/_std$/i, "");
  return { subject: s || key, isStd };
}

/* canonical ordering so normal and anomaly always pair up left→right. */
const SUBJECT_ORDER = ["normal", "pure_normal", "disturbing", "anomaly", "loss"];
function subjectRank(s: string): number {
  const i = SUBJECT_ORDER.indexOf(s);
  return i < 0 ? SUBJECT_ORDER.length : i;
}
/* the bar color emphasizes the normal↔anomaly contrast (the explicit pair). */
function subjectColor(s: string): string {
  if (s === "anomaly") return tok("--cat-2");
  if (s.includes("normal")) return tok("--cat-11");
  return tok("--cat-6");
}

const FACET_LABEL: Record<string, string> = {
  recon: "Reconstruction (teacher) magnitudes",
  disc: "Discrepancy (student) magnitudes",
  other: "Other magnitudes",
};

export default function SeparationPanel({ exp, dsKey, variant }: { exp: string; dsKey: string; variant: string | null }) {
  const q = useSeparation(exp, dsKey, variant);

  // F-7: bucket the neutral magnitude pairs by family (recon vs disc), keeping the
  // mean-magnitudes (excluding _std bands) so each facet shows the normal-vs-anomaly
  // pairing cleanly. Computed in the body (never in the render-prop) per the hooks guard.
  const facets = useMemo(() => {
    const pairs = q.data?.pairs ?? [];
    const byFacet = new Map<string, Pair[]>();
    for (const p of pairs) {
      const { isStd } = subjectOf(p.key);
      if (isStd) continue; // std bands clutter the pairing; the mean magnitudes carry it
      const f = facetOf(p);
      if (!byFacet.has(f)) byFacet.set(f, []);
      byFacet.get(f)!.push(p);
    }
    // stable facet order: recon, disc, other
    return ["recon", "disc", "other"]
      .filter((f) => byFacet.get(f)?.length)
      .map((f) => {
        const rows = byFacet
          .get(f)!
          .slice()
          .sort((a, b) => subjectRank(subjectOf(a.key).subject) - subjectRank(subjectOf(b.key).subject));
        return { facet: f, rows };
      });
  }, [q.data]);

  return (
    <AsyncPanel
      query={q}
      isEmpty={(d) => (!d.pairs || d.pairs.length === 0) && (!d.ratios_snr || d.ratios_snr.length === 0)}
      emptyLabel="No separation diagnostics for this leaf."
      height={300}
    >
      {(d) => {
        const ratioColors = d.ratios_snr.map((r) => ((r.value ?? 0) < 0 ? tok("--sem-warn") : tok("--sem-good")));
        const ratioTrace: any = {
          x: d.ratios_snr.map((r) => r.value),
          y: d.ratios_snr.map((r) => r.key),
          type: "bar",
          orientation: "h",
          marker: { color: ratioColors },
          hovertemplate: "%{y}: %{x:.4f}<extra></extra>",
        };
        return (
          <div className="col" style={{ gap: 16 }}>
            <div>
              <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
                Magnitude pairs by family (NEUTRAL context — anomaly &gt; normal is GOOD; not a lower-better loss). Each
                family has its OWN y-scale; normal &amp; anomaly are grouped as an explicit pair.
              </div>
              {/* F-7: one sub-chart per family, EACH with its own y-axis. */}
              <div className="card-grid">
                {facets.map(({ facet, rows }) => {
                  // group the bars by SUBJECT so normal and anomaly sit side by side; one
                  // bar per subject, colored to emphasize the normal↔anomaly contrast.
                  const subjects = rows.map((r) => subjectOf(r.key).subject);
                  const trace: any = {
                    x: subjects,
                    y: rows.map((r) => r.value),
                    type: "bar",
                    marker: { color: subjects.map(subjectColor) },
                    hovertemplate: "%{x}: %{y:.5f}<extra></extra>",
                  };
                  return (
                    <div key={facet}>
                      <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
                        {FACET_LABEL[facet] ?? facet}
                      </div>
                      <ChartFrame
                        name={`separation_${facet}_magnitudes`}
                        data={[trace]}
                        layout={baseLayout({
                          height: 240,
                          showlegend: false,
                          // F-7: an INDEPENDENT y-axis per facet (autorange over THIS
                          // family only) — recon and disc never share a flattening scale.
                          yaxis: { ...baseLayout().yaxis, title: { text: "magnitude" }, autorange: true },
                          xaxis: { ...baseLayout().xaxis, automargin: true },
                        })}
                        config={PLOTLY_CONFIG}
                        style={{ width: "100%", height: 240 }}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
            <div>
              <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600 }}>
                Ranked separation signal (↑) — ratios / SNR / Cohen's d · negative = anti-correlated (warn). This is what
                you RANK BY; the magnitudes above are context.
              </div>
              <ChartFrame
                name="separation_ranked_signal"
                data={[ratioTrace]}
                layout={baseLayout({
                  height: Math.max(220, d.ratios_snr.length * 26),
                  showlegend: false,
                  margin: { l: 220, r: 16, t: 8, b: 32 },
                  yaxis: { ...baseLayout().yaxis, automargin: true },
                })}
                config={PLOTLY_CONFIG}
                style={{ width: "100%", height: Math.max(220, d.ratios_snr.length * 26) }}
              />
              {d.ratios_snr.some((r) => (r.value ?? 0) < 0) && (
                <div className="chip warn">negative separation present — real anti-correlation, shown not hidden</div>
              )}
            </div>
          </div>
        );
      }}
    </AsyncPanel>
  );
}
