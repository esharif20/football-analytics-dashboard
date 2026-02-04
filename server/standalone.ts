/**
 * Standalone server entry point
 * Runs the dashboard without any Manus dependencies (OAuth, Forge, etc.)
 * Uses SQLite for database and local filesystem for storage
 */
import express from "express";
import * as path from "path";
import * as fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Import standalone database
import * as db from "./db.standalone";

const app = express();
app.use(express.json({ limit: "100mb" }));

// Serve static files from uploads directory
const uploadsDir = path.join(process.cwd(), "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}
app.use("/uploads", express.static(uploadsDir));

// Create default local user on startup
const LOCAL_USER = {
  id: 1,
  openId: "local-dev-user",
  name: "Local Developer",
  email: "dev@localhost",
  loginMethod: "local",
  role: "admin" as const,
  createdAt: new Date(),
  updatedAt: new Date(),
  lastSignedIn: new Date(),
};

// Initialize local user
db.upsertUser({
  openId: LOCAL_USER.openId,
  name: LOCAL_USER.name,
  email: LOCAL_USER.email,
  loginMethod: LOCAL_USER.loginMethod,
  role: LOCAL_USER.role,
});

// Auth endpoints - always return local user
app.get("/api/trpc/auth.me", (req, res) => {
  res.json({
    result: {
      data: LOCAL_USER,
    },
  });
});

app.post("/api/trpc/auth.logout", (req, res) => {
  res.json({
    result: {
      data: { success: true },
    },
  });
});

// Video endpoints
app.get("/api/trpc/videos.list", async (req, res) => {
  try {
    const videos = await db.getVideosByUserId(LOCAL_USER.id);
    res.json({ result: { data: videos } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

app.get("/api/trpc/videos.get", async (req, res) => {
  try {
    const input = JSON.parse(req.query.input as string || "{}");
    const video = await db.getVideoById(input.id);
    res.json({ result: { data: video } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

app.post("/api/trpc/videos.upload", async (req, res) => {
  try {
    const { title, description, fileData, fileName, mimeType, fileSize } = req.body;
    
    // Save file locally
    const fileKey = `${Date.now()}-${fileName}`;
    const filePath = path.join(uploadsDir, fileKey);
    
    // Decode base64 file data
    const buffer = Buffer.from(fileData, "base64");
    fs.writeFileSync(filePath, buffer);
    
    const videoId = await db.createVideo({
      userId: LOCAL_USER.id,
      title,
      description,
      originalUrl: `/uploads/${fileKey}`,
      fileKey,
      fileSize,
      mimeType,
      duration: null,
      fps: null,
      width: null,
      height: null,
      frameCount: null,
    });
    
    const video = await db.getVideoById(videoId);
    res.json({ result: { data: video } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

// Analysis endpoints
app.get("/api/trpc/analysis.list", async (req, res) => {
  try {
    const analyses = await db.getAnalysesByUserId(LOCAL_USER.id);
    res.json({ result: { data: analyses } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

app.get("/api/trpc/analysis.get", async (req, res) => {
  try {
    const input = JSON.parse(req.query.input as string || "{}");
    const analysis = await db.getAnalysisById(input.id);
    res.json({ result: { data: analysis } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

app.get("/api/trpc/analysis.getByVideo", async (req, res) => {
  try {
    const input = JSON.parse(req.query.input as string || "{}");
    const analyses = await db.getAnalysesByVideoId(input.videoId);
    res.json({ result: { data: analyses } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

app.post("/api/trpc/analysis.create", async (req, res) => {
  try {
    const { videoId, mode } = req.body;
    
    const analysisId = await db.createAnalysis({
      videoId,
      userId: LOCAL_USER.id,
      mode,
      status: "pending",
      progress: 0,
      currentStage: "Initializing",
      errorMessage: null,
      annotatedVideoUrl: null,
      radarVideoUrl: null,
      trackingDataUrl: null,
      analyticsDataUrl: null,
      startedAt: null,
      completedAt: null,
      processingTimeMs: null,
    });
    
    const analysis = await db.getAnalysisById(analysisId);
    res.json({ result: { data: analysis } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

app.post("/api/trpc/analysis.updateStatus", async (req, res) => {
  try {
    const { id, status, progress, currentStage, errorMessage } = req.body;
    await db.updateAnalysisStatus(id, status, progress, currentStage, errorMessage);
    const analysis = await db.getAnalysisById(id);
    res.json({ result: { data: analysis } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

// Statistics endpoints
app.get("/api/trpc/statistics.get", async (req, res) => {
  try {
    const input = JSON.parse(req.query.input as string || "{}");
    const stats = await db.getStatisticsByAnalysisId(input.analysisId);
    res.json({ result: { data: stats } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

// Events endpoints
app.get("/api/trpc/events.list", async (req, res) => {
  try {
    const input = JSON.parse(req.query.input as string || "{}");
    const events = await db.getEventsByAnalysisId(input.analysisId);
    res.json({ result: { data: events } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

// Tracks endpoints
app.get("/api/trpc/tracks.get", async (req, res) => {
  try {
    const input = JSON.parse(req.query.input as string || "{}");
    const tracks = await db.getTracksByAnalysisId(input.analysisId);
    res.json({ result: { data: tracks } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

// Commentary endpoints
app.get("/api/trpc/commentary.get", async (req, res) => {
  try {
    const input = JSON.parse(req.query.input as string || "{}");
    const commentary = await db.getCommentaryByAnalysisId(input.analysisId);
    res.json({ result: { data: commentary } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

app.post("/api/trpc/commentary.generate", async (req, res) => {
  try {
    const { analysisId, type } = req.body;
    
    // Generate simple commentary without LLM
    const commentaryId = await db.createCommentary({
      analysisId,
      eventId: null,
      frameStart: 0,
      frameEnd: 100,
      type: type || "tactical",
      content: "Analysis commentary will be generated when the pipeline completes processing.",
      confidence: 1.0,
      groundingData: null,
    });
    
    const commentary = await db.getCommentaryByAnalysisId(analysisId);
    res.json({ result: { data: commentary } });
  } catch (error) {
    res.status(500).json({ error: { message: String(error) } });
  }
});

// Serve Vite dev server in development or static files in production
const PORT = process.env.PORT || 3000;

async function startServer() {
  if (process.env.NODE_ENV === "production") {
    // Serve built static files
    const clientDist = path.join(__dirname, "../client/dist");
    app.use(express.static(clientDist));
    app.get("*", (req, res) => {
      res.sendFile(path.join(clientDist, "index.html"));
    });
  } else {
    // In development, proxy to Vite
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
      root: path.join(__dirname, "../client"),
    });
    app.use(vite.middlewares);
  }

  app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   🏈 Football Analysis Dashboard - STANDALONE MODE         ║
║                                                            ║
║   Server running at: http://localhost:${PORT}                ║
║                                                            ║
║   ✓ No external dependencies required                      ║
║   ✓ SQLite database (data/football.db)                     ║
║   ✓ Local file storage (uploads/)                          ║
║   ✓ Auto-logged in as Local Developer                      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
  `);
  });
}

startServer().catch(console.error);
