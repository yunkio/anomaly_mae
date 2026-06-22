/* MetricQuickPicks (2026-06-16) — a small row of one-click ranking metrics pulled to
 * the FRONT for convenience. The full runtime catalog stays available via the adjacent
 * <MetricPicker>; this just shortcuts the metrics asked for most often. Affiliation F1
 * is offered as TWO distinct keys — optimal-threshold (`affiliation_f1`) and anomaly-
 * ratio-threshold (`affiliation_f1_ar`) — so each is a one-click rank, independent of
 * the global Threshold toggle. The active metric is highlighted. */

interface Props {
  value: string;
  onChange: (key: string) => void;
}

/* key = the actual ranking metric key (resolved by the registry); label = the chip text. */
const QUICK: { key: string; label: string; title: string }[] = [
  { key: "pak_auc_f1", label: "pak_auc_f1", title: "PA%K-AUC F1 (primary; threshold-free)" },
  { key: "prc_auc", label: "prc", title: "PR-AUC (prc_auc; threshold-free)" },
  { key: "vus_pr", label: "vus_pr", title: "VUS-PR (threshold-free)" },
  { key: "affiliation_f1", label: "Affiliation F1", title: "Affiliation F1 @ optimal threshold" },
  { key: "affiliation_f1_ar", label: "Affiliation F1 (AR)", title: "Affiliation F1 @ anomaly-ratio threshold" },
];

export default function MetricQuickPicks({ value, onChange }: Props) {
  return (
    <span className="row wrap" style={{ gap: 6, alignItems: "center" }}>
      <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>quick</span>
      {QUICK.map((q) => (
        <button
          key={q.key}
          className={`btn sm ${value === q.key ? "primary" : ""}`}
          onClick={() => onChange(q.key)}
          title={q.title}
        >
          {q.label}
        </button>
      ))}
    </span>
  );
}
