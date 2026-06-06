/* App shell: persistent left rail (8 views + GIF Studio) + top context bar
 * (pin tray, health dot, rescan, live-refresh, theme). frontend-design §3.0. */
import { NavLink, useLocation } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useHealth } from "../api/queries";
import { api } from "../api/client";
import { useStore, catColor } from "../store";
import { shortModel } from "../scope";
import CommandPalette from "./CommandPalette";

const NAV = [
  { to: "/", label: "Overview", ico: "◎", end: true },
  { to: "/rankings", label: "Rankings", ico: "▦" },
  { to: "/compare", label: "Compare", ico: "⇄" },
  { to: "/explorer", label: "Explorer", ico: "∿" },
  { to: "/config", label: "Config + Diff", ico: "⚙" },
  { to: "/gallery", label: "Gallery", ico: "▤" },
  { to: "/gif", label: "GIF Studio", ico: "►" },
];

function HealthDot() {
  const { data } = useHealth();
  const st =
    typeof data?.resolver_selftest === "string"
      ? data?.resolver_selftest
      : (data?.resolver_selftest as any)?.status;
  const ok = (data?.status === "ok" || data?.status === "PASS") && st !== "FAIL";
  return (
    <span
      className="row"
      style={{ gap: 6 }}
      title={`backend ${data?.status ?? "?"} · resolver self-test ${st ?? "?"} · registry ${data?.registry ?? "?"}`}
    >
      <span
        style={{
          width: 9,
          height: 9,
          borderRadius: 99,
          background: ok ? "var(--sem-good)" : "var(--sem-warn)",
          display: "inline-block",
        }}
      />
      <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
        {ok ? "healthy" : "degraded"}
      </span>
    </span>
  );
}

function PinTray() {
  const pins = useStore((s) => s.pins);
  const unpin = useStore((s) => s.unpinModel);
  if (pins.length === 0)
    return (
      <span className="subtle" style={{ fontSize: "var(--fs-small)" }}>
        No pinned models — pin from Overview / Rankings, then Compare.
      </span>
    );
  return (
    <div className="pin-tray" title="Pinned models — pick a dataset in Compare / GIF / Config to compare them">
      {pins.map((expId, i) => (
        <span className="pin-chip" key={expId} title={expId}>
          <span className="swatch" style={{ background: catColor(i) }} />
          <span className="mono" style={{ fontSize: "0.72rem" }}>
            {shortModel(expId)}
          </span>
          <button onClick={() => unpin(expId)} aria-label="unpin model">
            ×
          </button>
        </span>
      ))}
    </div>
  );
}

function Breadcrumb() {
  const loc = useLocation();
  const parts = loc.pathname.split("/").filter(Boolean);
  return (
    <div className="breadcrumb">
      <NavLink to="/">Overview</NavLink>
      {parts.map((p, i) => (
        <span key={i}>
          {" › "}
          <span className="mono">{decodeURIComponent(p)}</span>
        </span>
      ))}
    </div>
  );
}

export default function AppShell({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const theme = useStore((s) => s.theme);
  const toggleTheme = useStore((s) => s.toggleTheme);
  const live = useStore((s) => s.liveRefresh);
  const setLive = useStore((s) => s.setLiveRefresh);

  async function rescan() {
    await api.rescan();
    await qc.invalidateQueries();
  }

  const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

  return (
    <div className="app-shell">
      <CommandPalette />
      <nav className="rail">
        <div className="rail-brand">
          TSMAE
          <small>experiment dashboard</small>
        </div>
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            end={n.end}
            className={({ isActive }) => `rail-link ${isActive ? "active" : ""}`}
          >
            <span className="ico">{n.ico}</span>
            {n.label}
          </NavLink>
        ))}
        <div className="rail-spacer" />
        <div className="rail-foot">
          <button className="btn sm" onClick={toggleTheme}>
            {theme === "light" ? "◐ Dark" : "◑ Light"}
          </button>
          <label className="row" style={{ gap: 6, fontSize: "var(--fs-small)" }}>
            <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
            live refresh
          </label>
          <button className="btn sm" onClick={rescan}>
            ⟳ Rescan
          </button>
          <HealthDot />
        </div>
      </nav>
      <header className="topbar">
        <Breadcrumb />
        <span className="grow" style={{ flex: 1 }} />
        <button
          className="btn sm"
          title="Command palette — search experiments, datasets, metrics"
          onClick={() =>
            window.dispatchEvent(
              new KeyboardEvent("keydown", { key: "k", ctrlKey: !isMac, metaKey: isMac })
            )
          }
          style={{ marginRight: 12 }}
        >
          ⌘K search
        </button>
        <PinTray />
      </header>
      <main className="content">{children}</main>
    </div>
  );
}
