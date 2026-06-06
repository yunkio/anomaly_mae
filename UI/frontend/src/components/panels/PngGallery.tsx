/* <PngGallery> (VIZ-1/2/3) — enumerates the present PNGs from /viz-manifest,
 * serves them via /files/png, badges stale viz, lightbox (Radix dialog). Shows an
 * "open animated version" link when a cached GIF exists for this leaf
 * (gif_available, FE-FB-5).
 *
 * F-9 (A-5): the "open animated version" cross-link DEEP-LINKS this exact leaf + a
 * story into GIF Studio — it PINS the leaf (GifStudio renders from the pin set) and
 * navigates to /gif?leaf=<leafId>&story=<story> so the Studio opens already focused on
 * THIS leaf, not a generic nav to whatever was last selected. */
import { useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { useNavigate } from "react-router-dom";
import { useVizManifest } from "../../api/queries";
import { api } from "../../api/client";
import AsyncPanel from "../AsyncPanel";
import { useStore } from "../../store";

export default function PngGallery({ exp, dsKey, variant }: { exp: string; dsKey: string; variant: string | null }) {
  const q = useVizManifest(exp, dsKey, variant);
  const nav = useNavigate();
  const pinModel = useStore((s) => s.pinModel);

  const [open, setOpen] = useState<{ url: string; name: string } | null>(null);

  // F-9: deep-link THIS leaf + a story into GIF Studio. Pin the MODEL first (idempotent),
  // then carry the full leaf (exp|dataset|variant) + story in the URL so GifStudio can
  // select the model AND set the shared dataset scope to this leaf on mount.
  function openAnimated(story = "climb_plateau") {
    pinModel(exp);
    const lid = `${exp}|${dsKey}|${variant ?? "_"}`;
    nav(`/gif?leaf=${encodeURIComponent(lid)}&story=${encodeURIComponent(story)}`);
  }

  return (
    <AsyncPanel
      query={q}
      isEmpty={(d) => Object.values(d.categories || {}).every((v) => !v || v.length === 0)}
      emptyLabel="No pre-rendered PNGs for this leaf."
      height={200}
    >
      {(d) => (
        <div className="col" style={{ gap: 12 }}>
          {d.stale && d.stale.length > 0 && (
            <div className="chip warn" title={d.stale_reason ?? ""}>
              stale viz — reflects an old best epoch ({d.stale.length})
            </div>
          )}
          {d.gif_available && (
            <button
              className="btn sm primary"
              style={{ alignSelf: "flex-start" }}
              title="open GIF Studio focused on THIS leaf (pins it + deep-links the story)"
              onClick={() => openAnimated()}
            >
              ► open animated versions ({d.gifs.length})
            </button>
          )}
          {Object.entries(d.categories).map(([cat, paths]) =>
            paths && paths.length ? (
              <div key={cat}>
                <div className="subtle" style={{ fontSize: "var(--fs-small)", fontWeight: 600, marginBottom: 6 }}>
                  {cat} ({paths.length})
                </div>
                <div
                  style={{
                    display: "grid",
                    gap: 10,
                    gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                  }}
                >
                  {paths.map((p) => {
                    const abs = d.abs_paths?.[p] ?? p;
                    const url = api.pngUrl(abs);
                    const fname = p.split("/").pop() || "image.png";
                    // R3-SR-02: the download <a> is a SIBLING OVERLAY, never a child of
                    // the lightbox <button> (nesting an <a> in a <button> is invalid HTML
                    // and the button's onClick swallows the anchor click).
                    return (
                      <div key={p} style={{ position: "relative" }}>
                        <button
                          onClick={() => setOpen({ url, name: fname })}
                          style={{
                            border: "1px solid var(--border)",
                            borderRadius: "var(--radius-md)",
                            overflow: "hidden",
                            background: "var(--surface-2)",
                            cursor: "zoom-in",
                            padding: 0,
                            width: "100%",
                            display: "block",
                          }}
                          title={p}
                        >
                          <img src={url} alt={p} loading="lazy" style={{ width: "100%", display: "block" }} />
                          <div className="subtle" style={{ fontSize: "0.7rem", padding: "2px 6px", textAlign: "left" }}>
                            {fname}
                          </div>
                        </button>
                        <a
                          className="btn sm"
                          href={url}
                          download={fname}
                          title="download this image (original PNG)"
                          onClick={(e) => e.stopPropagation()}
                          style={{ position: "absolute", top: 6, right: 6, padding: "1px 6px", lineHeight: 1.4 }}
                        >
                          ↓
                        </a>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null
          )}

          <Dialog.Root open={!!open} onOpenChange={(o) => !o && setOpen(null)}>
            <Dialog.Portal>
              <Dialog.Overlay className="dialog-overlay" />
              <Dialog.Content className="dialog-content">
                <Dialog.Title className="card-title">Visualization</Dialog.Title>
                {open && <img src={open.url} alt="full" style={{ maxWidth: "86vw", maxHeight: "80vh" }} />}
                <div className="row" style={{ marginTop: 8, gap: 8 }}>
                  {/* FB-R3-03: lightbox image download (the Dialog anchor is not inside a
                      button — valid HTML). */}
                  {open && (
                    <a className="btn sm" href={open.url} download={open.name}>
                      ↓ download PNG
                    </a>
                  )}
                  <Dialog.Close asChild>
                    <button className="btn sm">close</button>
                  </Dialog.Close>
                </div>
              </Dialog.Content>
            </Dialog.Portal>
          </Dialog.Root>
        </div>
      )}
    </AsyncPanel>
  );
}
