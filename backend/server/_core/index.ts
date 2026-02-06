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
import { isLocalDevMode, getLocalDevUser } from "./localMode";
import { getLocalStorageDir, storagePut } from "../storage";
import { createVideo } from "../db";
import multer from "multer";
import { nanoid } from "nanoid";

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
  
  // Multipart video upload endpoint (avoids base64 encoding in browser)
  const upload = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 500 * 1024 * 1024 } // 500MB limit
  });
  
  app.post("/api/upload/video", upload.single("video"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "No video file provided" });
      }
      
      // Get user - in local dev mode use local user, otherwise would need auth
      const user = isLocalDevMode() ? getLocalDevUser() : null;
      if (!user) {
        return res.status(401).json({ error: "Authentication required" });
      }
      
      const title = req.body.title || req.file.originalname;
      const description = req.body.description || null;
      const fileName = req.file.originalname;
      const mimeType = req.file.mimetype;
      const fileSize = req.file.size;
      
      const fileKey = `videos/${user.id}/${nanoid()}-${fileName}`;
      const { url } = await storagePut(fileKey, req.file.buffer, mimeType);
      
      const videoId = await createVideo({
        userId: user.id,
        title,
        description,
        originalUrl: url,
        fileKey,
        fileSize,
        mimeType,
      });
      
      console.log(`[Upload] Video saved: ${fileName} (${(fileSize / (1024 * 1024)).toFixed(1)}MB) -> id=${videoId}`);
      res.json({ id: videoId, url });
    } catch (error) {
      console.error("Video upload error:", error);
      res.status(500).json({ error: "Failed to upload video" });
    }
  });
  
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
