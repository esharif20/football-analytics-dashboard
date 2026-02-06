import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import express, { type Express } from "express";
import fs from "fs";
import { type Server } from "http";
import { nanoid } from "nanoid";
import path from "path";
import { createServer as createViteServer } from "vite";

const PROJECT_ROOT = path.resolve(import.meta.dirname, "../../..");
const FRONTEND_ROOT = path.resolve(PROJECT_ROOT, "frontend");

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

const viteConfig = {
  plugins,
  resolve: {
    alias: {
      "@": path.resolve(FRONTEND_ROOT, "src"),
    },
  },
  root: FRONTEND_ROOT,
  publicDir: path.resolve(FRONTEND_ROOT, "public"),
  build: {
    outDir: path.resolve(PROJECT_ROOT, "dist/public"),
    emptyOutDir: true,
  },
  server: {
    host: true,
    allowedHosts: [
      ".manuspre.computer",
      ".manus.computer",
      ".manus-asia.computer",
      ".manuscomputer.ai",
      ".manusvm.computer",
      "localhost",
      "127.0.0.1",
    ],
    fs: {
      strict: true,
      deny: ["**/.*"],
    },
  },
};

export async function setupVite(app: Express, server: Server) {
  const serverOptions = {
    middlewareMode: true,
    hmr: { server },
    allowedHosts: true as const,
  };

  const vite = await createViteServer({
    ...viteConfig,
    configFile: false,
    server: serverOptions,
    appType: "custom",
  });

  app.use(vite.middlewares);
  app.use("*", async (req, res, next) => {
    const url = req.originalUrl;

    try {
      const clientTemplate = path.resolve(FRONTEND_ROOT, "index.html");

      // always reload the index.html file from disk incase it changes
      let template = await fs.promises.readFile(clientTemplate, "utf-8");
      template = template.replace(
        `src="/src/main.tsx"`,
        `src="/src/main.tsx?v=${nanoid()}"`
      );
      const page = await vite.transformIndexHtml(url, template);
      res.status(200).set({ "Content-Type": "text/html" }).end(page);
    } catch (e) {
      vite.ssrFixStacktrace(e as Error);
      next(e);
    }
  });
}

export function serveStatic(app: Express) {
  const distPath =
    process.env.NODE_ENV === "development"
      ? path.resolve(PROJECT_ROOT, "dist", "public")
      : path.resolve(import.meta.dirname, "public");
  if (!fs.existsSync(distPath)) {
    console.error(
      `Could not find the build directory: ${distPath}, make sure to build the client first`
    );
  }

  app.use(express.static(distPath));

  // fall through to index.html if the file doesn't exist
  app.use("*", (_req, res) => {
    res.sendFile(path.resolve(distPath, "index.html"));
  });
}
