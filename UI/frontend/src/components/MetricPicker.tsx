/* <MetricPicker> — a compact single-select metric chooser for cross-cutting
 * toolbars (Rankings IA-8, Compare IA-7, ScoreVariantPanel). It is a thin Radix
 * dropdown wrapper around <MetricFamilySelector multi={false}> bound to the RUNTIME
 * catalog — so there is ZERO hard-coded metric list and a renamed/new/inferred
 * metric appears automatically (review-P4-frontend F-01; F5/AC-F5.1).
 *
 * The current value shows on the trigger; the dropdown body is the full grouped,
 * direction-badged, namespace-faceted, searchable selector. Picking one closes it. */
import { useMemo, useState } from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import MetricFamilySelector from "./MetricFamilySelector";
import type { CatalogEntry } from "../api/types";

interface Props {
  catalog: CatalogEntry[];
  value: string;
  onChange: (key: string) => void;
  /** restrict to these namespaces (e.g. ScoreVariantPanel → epoch_metrics) */
  namespaceFilter?: string[];
  /** restrict to families (e.g. a GIF story's allowed families) */
  familyFilter?: string[];
  label?: string;
  /** loading text while the catalog query is still in flight */
  loading?: boolean;
  disabled?: boolean;
  /** single-leaf SWaT variant ("full"/"excl22") → hide the other variant's metrics. */
  leafVariant?: string | null;
}

export default function MetricPicker({
  catalog,
  value,
  onChange,
  namespaceFilter,
  familyFilter,
  label = "metric",
  loading = false,
  disabled = false,
  leafVariant,
}: Props) {
  const [open, setOpen] = useState(false);
  // direction arrow for the current value, read from the catalog (registry-driven).
  const cur = useMemo(() => catalog.find((c) => c.key === value), [catalog, value]);
  const arrow = cur?.direction === "up" ? " ↑" : cur?.direction === "down" ? " ↓" : cur ? " ◦" : "";

  return (
    <label className="row" style={{ gap: 6 }}>
      {label}
      <DropdownMenu.Root open={open} onOpenChange={setOpen}>
        <DropdownMenu.Trigger asChild>
          <button className="select" style={{ minWidth: 170, textAlign: "left", cursor: "pointer" }} disabled={disabled}>
            <span className="mono" style={{ fontSize: "0.82rem" }}>
              {value}
              {arrow}
            </span>
            <span style={{ float: "right", opacity: 0.6 }}>▾</span>
          </button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content
            className="metric-picker-pop"
            sideOffset={4}
            align="start"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              boxShadow: "var(--shadow-2)",
              padding: "var(--space-2)",
              width: 360,
              maxWidth: "90vw",
              zIndex: 60,
            }}
          >
            {loading ? (
              <div className="subtle" style={{ padding: 8 }}>
                loading runtime catalog…
              </div>
            ) : catalog.length === 0 ? (
              <div className="subtle" style={{ padding: 8 }}>
                No metrics discovered yet — pin/scan a run.
              </div>
            ) : (
              <MetricFamilySelector
                catalog={catalog}
                selected={value ? [value] : []}
                multi={false}
                namespaceFilter={namespaceFilter}
                familyFilter={familyFilter}
                leafVariant={leafVariant}
                onChange={(keys) => {
                  if (keys.length) {
                    onChange(keys[0]);
                    setOpen(false);
                  }
                }}
              />
            )}
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
    </label>
  );
}
