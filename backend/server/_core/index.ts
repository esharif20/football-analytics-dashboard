import "dotenv/config";
import express from "express";
import { createServer } from "http";
import net from "net";
import path from "path";
import { createExpressMiddleware } from "@trpc/server/adapters/express";
import { registerOAuthRoutes } from "./oauth";
import { appRouter } from "../routers";
import { createContext } from "./context";
import { serveStatic, setupVite } from "./vite";
import { isLocalDevMode } from "./localMode";
import { getLocalStorageDir } from "../storage";

function isPortAvailable(port: number): Promise<boolean> {
  return new Promise(resolve => {
    const server = net.createServer();
    server.listen(port, () => {
      server.close(() => resolve(true));
    });
    server.on("error", () => resolve(false));
  });
}

async function findAvailablePort(startPort: number = 3000): Promise<number> {
  for (let port = startPort; port < startPort + 20; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }
  throw new Error(`No available port found starting from ${startPort}`);
}

async function startServer() {
  const app = express();
  const server = createServer(app);
  
  // Log mode on startup
  if (isLocalDevMode()) {
    console.log("\n🏠 Running in LOCAL DEV MODE");
    console.log("   - Auth: Bypassed (auto-logged in as Local Developer)");
    console.log("   - Storage: Local filesystem (./uploads/)\n");
  }
  
  // Configure body parser with larger size limit for file uploads
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ limit: "50mb", extended: true }));
  
  // Serve local uploads in dev mode
  if (isLocalDevMode()) {
    const uploadsDir = path.resolve(getLocalStorageDir());
    app.use("/uploads", express.static(uploadsDir));
  }
  
  // OAuth callback under /api/oauth/callback
  registerOAuthRoutes(app);
  
  // Worker API endpoints (for Python worker to call)
  app.get("/api/worker/pending", async (req, res) => {
    try {
      const { getPendingAnalyses } = await import("../db");
      const analyses = await getPendingAnalyses();
      res.json({ analyses });
    } catch (error) {
      console.error("Worker API error:", error);
      res.status(500).json({ error: "Failed to get pending analyses" });
    }
  });
  
  app.post("/api/worker/analysis/:id/status", async (req, res) => {
    try {
      const { updateAnalysisStatus } = await import("../db");
      const id = parseInt(req.params.id);
      const { status, currentStage, progress, error: errorMessage } = req.body;
      await updateAnalysisStatus(id, status, progress || 0, currentStage, errorMessage);
      res.json({ success: true });
    } catch (error) {
      console.error("Worker API error:", error);
      res.status(500).json({ error: "Failed to update status" });
    }
  });
  
  app.post("/api/worker/analysis/:id/complete", async (req, res) => {
    try {
      const { updateAnalysisResults, updateAnalysisStatus } = await import("../db");
      const id = parseInt(req.params.id);
      const { annotatedVideo, radarVideo, analytics, tracks } = req.body;
      
      await updateAnalysisResults(id, {
        annotatedVideoUrl: annotatedVideo,
        radarVideoUrl: radarVideo,
        analyticsDataUrl: analytics ? JSON.stringify(analytics) : undefined,
        trackingDataUrl: tracks ? JSON.stringify(tracks) : undefined,
      });
      await updateAnalysisStatus(id, "completed", 100, "done");
      
      res.json({ success: true });
    } catch (error) {
      console.error("Worker API error:", error);
      res.status(500).json({ error: "Failed to complete analysis" });
    }
  });
  
  // tRPC API
  app.use(
    "/api/trpc",
    createExpressMiddleware({
      router: appRouter,
      createContext,
    })
  );
  // development mode uses Vite, production mode uses static files
  if (process.env.NODE_ENV === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  const preferredPort = parseInt(process.env.PORT || "3000");
  const port = await findAvailablePort(preferredPort);

  if (port !== preferredPort) {
    console.log(`Port ${preferredPort} is busy, using port ${port} instead`);
  }

  server.listen(port, () => {
    console.log(`Server running on http://localhost:${port}/`);
  });
}

startServer().catch(console.error);
