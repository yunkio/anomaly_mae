/* Dependency-free guard for the F-02 recurrence class (Rules-of-Hooks).
 *
 * eslint-plugin-react-hooks is the ideal tool, but the build box is OFFLINE and the
 * build is `tsc -b && vite build` (no lint step). This static check scans every src
 * file for a React hook call (`useMemo`/`useState`/`useEffect`/…) that appears INSIDE
 * an <AsyncPanel> children render-prop — i.e. after a `{(...) => {` arrow that follows
 * an <AsyncPanel ...> open tag. AsyncPanel runs `children(data)` only on the success
 * render, so a hook there changes the hook count across query-state transitions and
 * crashes (review-P4-frontend F-02). Hooks must live in the component body.
 *
 * Run: node scripts/check-render-prop-hooks.mjs   (exit 1 on any violation)
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, extname } from "node:path";

const SRC = new URL("../src", import.meta.url).pathname;
const HOOK = /\buse(Memo|State|Effect|Ref|Callback|Reducer|Context|Query|LayoutEffect)\s*\(/;

function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const s = statSync(p);
    if (s.isDirectory()) out.push(...walk(p));
    else if ([".ts", ".tsx"].includes(extname(p))) out.push(p);
  }
  return out;
}

const violations = [];
for (const file of walk(SRC)) {
  const text = readFileSync(file, "utf8");
  const lines = text.split("\n");
  // track whether we are inside an <AsyncPanel ...> ... children render-prop body
  let inAsyncOpen = false; // saw `<AsyncPanel` but not yet the `>` + render-prop
  let renderPropDepth = 0; // brace depth of the render-prop body; 0 = not inside
  let braceTrack = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/<AsyncPanel\b/.test(line)) inAsyncOpen = true;
    // the render-prop opens with `{(...) => {` (possibly across lines after the tag)
    if (inAsyncOpen && /\{\s*\([^)]*\)\s*=>\s*\{/.test(line)) {
      renderPropDepth = 1;
      braceTrack = 0;
      inAsyncOpen = false;
      continue;
    }
    if (renderPropDepth > 0) {
      // crude brace accounting to know when the render-prop body closes
      for (const ch of line) {
        if (ch === "{") braceTrack++;
        else if (ch === "}") braceTrack--;
      }
      if (HOOK.test(line)) {
        violations.push(`${file}:${i + 1}: ${line.trim()}`);
      }
      if (braceTrack < 0) renderPropDepth = 0; // body closed
    }
  }
}

if (violations.length) {
  console.error("Rules-of-Hooks violation — hook(s) inside an <AsyncPanel> render-prop:");
  for (const v of violations) console.error("  " + v);
  console.error("\nHoist these hooks into the component body (compute from query.data with a null-guard).");
  process.exit(1);
}
console.log("OK: no hooks inside any <AsyncPanel> render-prop.");
