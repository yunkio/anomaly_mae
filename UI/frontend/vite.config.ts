import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build emits to ./dist, which the backend serves via its StaticFiles mount
// (UI/backend/app/main.py: `/assets` mount + SPA history-fallback to index.html).
// During `npm run dev`, /api and /gif are proxied to the running uvicorn backend
// so the SPA hits the real, read-only data API on the same origin contract.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1600,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
