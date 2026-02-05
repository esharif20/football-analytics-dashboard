import "dotenv/config";
import express from "express";
import { createServer } from "http";
import fs from "fs";
import path from "path";
import { createExpressMiddleware } from "@trpc/server/adapters/express";
import { registerOAuthRoutes } from "./oauth";
import { appRouter } from "../routers";
import { createContext } from "./context";

function serveStatic(app: express.Express) {
  // In production, static files are in the same directory as index.js
  const distPath = path.resolve(import.meta.dirname, "public");
  
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

async function startServer() {
  const app = express();
  const server = createServer(app);
  
  // Configure body parser with larger size limit for file uploads
  app.use(express.json({ limit: "50mb" }));
  app.use(express.urlencoded({ limit: "50mb", extended: true }));
  
  // OAuth callback under /api/oauth/callback
  registerOAuthRoutes(app);
  
  // Worker API endpoints (for Python worker to call)
  // These are public endpoints that the external GPU worker uses
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
  
  // Worker video upload endpoint - accepts base64 encoded video
  app.post("/api/worker/upload-video", async (req, res) => {
    try {
      const { storagePut } = await import("../storage");
      const { videoData, fileName, contentType } = req.body;
      
      if (!videoData || !fileName) {
        return res.status(400).json({ error: "Missing videoData or fileName" });
      }
      
      // Decode base64 video data
      const buffer = Buffer.from(videoData, "base64");
      
      // Upload to storage
      const key = `processed-videos/${Date.now()}-${fileName}`;
      const { url } = await storagePut(key, buffer, contentType || "video/mp4");
      
      console.log(`[Worker] Uploaded video: ${fileName} -> ${url}`);
      res.json({ success: true, url });
    } catch (error) {
      console.error("Worker upload error:", error);
      res.status(500).json({ error: "Failed to upload video" });
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
  
  // Serve static files in production
  serveStatic(app);

  const port = parseInt(process.env.PORT || "3000");

  server.listen(port, () => {
    console.log(`Server running on http://localhost:${port}/`);
  });
}

startServer().catch(console.error);
