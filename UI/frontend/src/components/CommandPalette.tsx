/* <CommandPalette> — the global Cmd/Ctrl-K palette (frontend-design §2.6 / §3.0,
 * AC-F3.5 navigability). A Radix dialog that searches:
 *   • experiments  (GET /experiments)            → navigate to /exp/:id
 *   • datasets (leaves)                           → navigate + pin
 *   • the metric catalog (GET /metrics-catalog)   → jump to Rankings ranked by it
 *   • the 8 views                                 → navigate
 * Keyboard: Cmd/Ctrl-K opens, ↑/↓ move, Enter runs, Esc closes. Everything is fed by
 * the RUNTIME catalog / discovery — no hard-coded metric or dataset list (F5). */
import { useEffect, useMemo, useRef, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { useNavigate } from "react-router-dom";
import { useExperiments, useGlobalMetricsCatalog } from "../api/queries";
import { dsUrl } from "../api/client";
import { metricCatalogView } from "../api/metricView";
import { useStore } from "../store";

type Action = "navigate" | "pin";
interface Item {
  id: string;
  kind: "view" | "experiment" | "dataset" | "metric";
  label: string;
  hint?: string;
  run: () => void;
  action: Action;
}

const VIEWS: { label: string; to: string }[] = [
  { label: "Overview", to: "/" },
  { label: "Rankings", to: "/rankings" },
  { label: "Compare Workbench", to: "/compare" },
  { label: "Explorer", to: "/explorer" },
  { label: "Config + Diff", to: "/config" },
  { label: "Gallery", to: "/gallery" },
  { label: "GIF Studio", to: "/gif" },
];

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const nav = useNavigate();
  const pinModel = useStore((s) => s.pinModel);

  // data sources (cached; only fetched while the app is mounted)
  const exps = useExperiments();
  const catalog = useGlobalMetricsCatalog();

  // global Cmd/Ctrl-K toggle
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) {
      setQ("");
      setActive(0);
      // focus after the dialog mounts
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  const items = useMemo<Item[]>(() => {
    const out: Item[] = [];
    const close = () => setOpen(false);

    for (const v of VIEWS)
      out.push({
        id: `view:${v.to}`,
        kind: "view",
        label: v.label,
        hint: "view",
        action: "navigate",
        run: () => {
          nav(v.to);
          close();
        },
      });

    for (const e of exps.data ?? []) {
      out.push({
        id: `exp:${e.exp_id}`,
        kind: "experiment",
        label: e.exp_id,
        hint: `experiment · ${e.state}`,
        action: "navigate",
        run: () => {
          nav(`/exp/${encodeURIComponent(e.exp_id)}`);
          close();
        },
      });
      // pin the MODEL (you pick a dataset to compare on later, in Compare).
      out.push({
        id: `pin:${e.exp_id}`,
        kind: "experiment",
        label: `Pin model ${e.exp_id}`,
        hint: "pin model",
        action: "pin",
        run: () => {
          pinModel(e.exp_id);
          close();
        },
      });
      for (const d of e.datasets ?? []) {
        out.push({
          id: `ds:${e.exp_id}|${d.dataset_key}`,
          kind: "dataset",
          label: `${e.exp_id.split("_")[0]} · ${d.dataset_key}`,
          hint: "dataset · ↵ open",
          action: "navigate",
          run: () => {
            nav(`/exp/${encodeURIComponent(e.exp_id)}/ds/${dsUrl(d.dataset_key)}`);
            close();
          },
        });
      }
    }

    // De-dup the "Rank by …" commands through the shared metricCatalogView() chokepoint
    // so the palette does not list one metric N times (per namespace/scope) — same
    // FB-1 policy as the pickers; the canonical (non-prefixed) key is emitted (RV2-05).
    for (const c of metricCatalogView(catalog.data ?? [])) {
      const arrow = c.direction === "up" ? "↑" : c.direction === "down" ? "↓" : "◦";
      out.push({
        id: `metric:${c.variant_scope}:${c.namespace}:${c.key}`,
        kind: "metric",
        label: `Rank by ${c.display_label} ${arrow}`,
        hint: `metric · ${c.family}${c.inferred ? " · inferred" : ""}`,
        action: "navigate",
        run: () => {
          nav(`/rankings?metric=${encodeURIComponent(c.key)}`);
          close();
        },
      });
    }
    return out;
  }, [exps.data, catalog.data, nav, pinModel]);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return items.slice(0, 40);
    const scored = items
      .filter((it) => it.label.toLowerCase().includes(s) || (it.hint || "").toLowerCase().includes(s))
      .slice(0, 60);
    return scored;
  }, [items, q]);

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      filtered[active]?.run();
    }
  }

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Portal>
        <Dialog.Overlay
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.32)", zIndex: 80 }}
        />
        <Dialog.Content
          aria-label="Command palette"
          onOpenAutoFocus={(e) => e.preventDefault()}
          style={{
            position: "fixed",
            top: "14vh",
            left: "50%",
            transform: "translateX(-50%)",
            width: "min(640px, 92vw)",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-lg)",
            boxShadow: "var(--shadow-3)",
            zIndex: 81,
            overflow: "hidden",
          }}
        >
          <Dialog.Title style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)" }}>
            Command palette
          </Dialog.Title>
          <input
            ref={inputRef}
            className="input"
            placeholder="Search experiments, datasets, metrics, views…  (Esc to close)"
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setActive(0);
            }}
            onKeyDown={onKeyDown}
            style={{ width: "100%", border: 0, borderBottom: "1px solid var(--border)", borderRadius: 0, padding: "14px 16px", fontSize: "1rem" }}
          />
          <div style={{ maxHeight: "52vh", overflowY: "auto" }}>
            {filtered.length === 0 && (
              <div className="subtle" style={{ padding: 16 }}>
                No matches. {exps.isLoading || catalog.isLoading ? "Loading discovery…" : ""}
              </div>
            )}
            {filtered.map((it, i) => (
              <button
                key={it.id}
                className="cmdk-row"
                onMouseEnter={() => setActive(i)}
                onClick={() => it.run()}
                style={{
                  display: "flex",
                  width: "100%",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 12,
                  padding: "9px 16px",
                  border: 0,
                  background: i === active ? "var(--surface-2)" : "transparent",
                  cursor: "pointer",
                  textAlign: "left",
                }}
              >
                <span className="row" style={{ gap: 8 }}>
                  <span className={`chip ${it.kind}`} style={{ fontSize: "0.66rem" }}>
                    {it.kind}
                  </span>
                  <span className="mono" style={{ fontSize: "0.85rem" }}>
                    {it.label}
                  </span>
                </span>
                <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
                  {it.hint}
                </span>
              </button>
            ))}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
