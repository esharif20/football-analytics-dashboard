import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { defineConfig } from "vite";

// Core plugins (always available)
const plugins: any[] = [react(), tailwindcss()];

// Manus-specific plugins - optional, only loaded when available
try {
  const { jsxLocPlugin } = await import("@builder.io/vite-plugin-jsx-loc");
  plugins.push(jsxLocPlugin());
} catch { /* Not available outside Manus */ }

try {
  const { vitePluginManusRuntime } = await import("vite-plugin-manus-runtime");
  plugins.push(vitePluginManusRuntime());
} catch { /* Not available outside Manus */ }

export default defineConfig({
  plugins,
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "src"),
    },
  },
  root: path.resolve(import.meta.dirname),
  publicDir: path.resolve(import.meta.dirname, "public"),
  build: {
    outDir: path.resolve(import.meta.dirname, "..", "dist", "public"),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/uploads": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:8000",
        ws: true,
        // Suppress noisy "socket ended" errors when backend restarts
        // or connection drops during long pipeline runs
        configure: (proxy) => {
          proxy.on("error", () => {});
        },
      },
    },
  },
});
