/**
 * Standalone database module using SQLite
 * This allows running the dashboard without MySQL/external database
 */
import Database from "better-sqlite3";
import * as path from "path";
import * as fs from "fs";

// Database file path - stored in project root
const DB_PATH = path.join(process.cwd(), "data", "football.db");

// Ensure data directory exists
const dataDir = path.dirname(DB_PATH);
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

// Create SQLite database connection
const sqlite = new Database(DB_PATH);

// Initialize tables
const initSQL = `
  CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    openId TEXT NOT NULL UNIQUE,
    name TEXT,
    email TEXT,
    loginMethod TEXT,
    role TEXT DEFAULT 'user' NOT NULL,
    createdAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updatedAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    lastSignedIn INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );

  CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    userId INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    originalUrl TEXT NOT NULL,
    fileKey TEXT NOT NULL,
    duration REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    frameCount INTEGER,
    fileSize INTEGER,
    mimeType TEXT,
    createdAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updatedAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );

  CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    videoId INTEGER NOT NULL,
    userId INTEGER NOT NULL,
    mode TEXT NOT NULL,
    status TEXT DEFAULT 'pending' NOT NULL,
    progress INTEGER DEFAULT 0 NOT NULL,
    currentStage TEXT,
    errorMessage TEXT,
    annotatedVideoUrl TEXT,
    radarVideoUrl TEXT,
    trackingDataUrl TEXT,
    analyticsDataUrl TEXT,
    startedAt INTEGER,
    completedAt INTEGER,
    processingTimeMs INTEGER,
    createdAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updatedAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );

  CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysisId INTEGER NOT NULL,
    type TEXT NOT NULL,
    frameNumber INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    playerId INTEGER,
    teamId INTEGER,
    targetPlayerId INTEGER,
    startX REAL,
    startY REAL,
    endX REAL,
    endY REAL,
    success INTEGER,
    confidence REAL,
    metadata TEXT,
    createdAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );

  CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysisId INTEGER NOT NULL,
    frameNumber INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    playerPositions TEXT,
    ballPosition TEXT,
    teamFormations TEXT,
    voronoiData TEXT,
    createdAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );

  CREATE TABLE IF NOT EXISTS statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysisId INTEGER NOT NULL,
    possessionTeam1 REAL,
    possessionTeam2 REAL,
    passesTeam1 INTEGER,
    passesTeam2 INTEGER,
    passAccuracyTeam1 REAL,
    passAccuracyTeam2 REAL,
    shotsTeam1 INTEGER,
    shotsTeam2 INTEGER,
    distanceCoveredTeam1 REAL,
    distanceCoveredTeam2 REAL,
    avgSpeedTeam1 REAL,
    avgSpeedTeam2 REAL,
    heatmapDataTeam1 TEXT,
    heatmapDataTeam2 TEXT,
    passNetworkTeam1 TEXT,
    passNetworkTeam2 TEXT,
    createdAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updatedAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );

  CREATE TABLE IF NOT EXISTS commentary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysisId INTEGER NOT NULL,
    eventId INTEGER,
    frameStart INTEGER,
    frameEnd INTEGER,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL,
    groundingData TEXT,
    createdAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );
`;

sqlite.exec(initSQL);
console.log("[Database] SQLite initialized at:", DB_PATH);

// Types
export interface User {
  id: number;
  openId: string;
  name: string | null;
  email: string | null;
  loginMethod: string | null;
  role: "user" | "admin";
  createdAt: Date;
  updatedAt: Date;
  lastSignedIn: Date;
}

export interface InsertUser {
  openId: string;
  name?: string | null;
  email?: string | null;
  loginMethod?: string | null;
  role?: "user" | "admin";
  lastSignedIn?: Date;
}

export interface Video {
  id: number;
  userId: number;
  title: string;
  description: string | null;
  originalUrl: string;
  fileKey: string;
  duration: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  frameCount: number | null;
  fileSize: number | null;
  mimeType: string | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface Analysis {
  id: number;
  videoId: number;
  userId: number;
  mode: string;
  status: string;
  progress: number;
  currentStage: string | null;
  errorMessage: string | null;
  annotatedVideoUrl: string | null;
  radarVideoUrl: string | null;
  trackingDataUrl: string | null;
  analyticsDataUrl: string | null;
  startedAt: Date | null;
  completedAt: Date | null;
  processingTimeMs: number | null;
  createdAt: Date;
  updatedAt: Date;
}

export interface Statistics {
  id: number;
  analysisId: number;
  possessionTeam1: number | null;
  possessionTeam2: number | null;
  passesTeam1: number | null;
  passesTeam2: number | null;
  passAccuracyTeam1: number | null;
  passAccuracyTeam2: number | null;
  shotsTeam1: number | null;
  shotsTeam2: number | null;
  distanceCoveredTeam1: number | null;
  distanceCoveredTeam2: number | null;
  avgSpeedTeam1: number | null;
  avgSpeedTeam2: number | null;
  heatmapDataTeam1: any;
  heatmapDataTeam2: any;
  passNetworkTeam1: any;
  passNetworkTeam2: any;
  createdAt: Date;
  updatedAt: Date;
}

export interface Event {
  id: number;
  analysisId: number;
  type: string;
  frameNumber: number;
  timestamp: number;
  playerId: number | null;
  teamId: number | null;
  targetPlayerId: number | null;
  startX: number | null;
  startY: number | null;
  endX: number | null;
  endY: number | null;
  success: boolean | null;
  confidence: number | null;
  metadata: any;
  createdAt: Date;
}

export interface Track {
  id: number;
  analysisId: number;
  frameNumber: number;
  timestamp: number;
  playerPositions: any;
  ballPosition: any;
  teamFormations: any;
  voronoiData: any;
  createdAt: Date;
}

export interface Commentary {
  id: number;
  analysisId: number;
  eventId: number | null;
  frameStart: number | null;
  frameEnd: number | null;
  type: string;
  content: string;
  confidence: number | null;
  groundingData: any;
  createdAt: Date;
}

// User operations
export async function upsertUser(user: InsertUser): Promise<void> {
  const existing = sqlite.prepare("SELECT * FROM users WHERE openId = ?").get(user.openId);
  
  if (existing) {
    sqlite.prepare(`
      UPDATE users SET 
        name = COALESCE(?, name),
        email = COALESCE(?, email),
        loginMethod = COALESCE(?, loginMethod),
        role = COALESCE(?, role),
        updatedAt = strftime('%s', 'now'),
        lastSignedIn = strftime('%s', 'now')
      WHERE openId = ?
    `).run(user.name, user.email, user.loginMethod, user.role, user.openId);
  } else {
    sqlite.prepare(`
      INSERT INTO users (openId, name, email, loginMethod, role)
      VALUES (?, ?, ?, ?, ?)
    `).run(user.openId, user.name || null, user.email || null, user.loginMethod || null, user.role || 'user');
  }
}

export async function getUserByOpenId(openId: string): Promise<User | undefined> {
  const row = sqlite.prepare("SELECT * FROM users WHERE openId = ?").get(openId) as any;
  if (!row) return undefined;
  
  return {
    ...row,
    createdAt: new Date(row.createdAt * 1000),
    updatedAt: new Date(row.updatedAt * 1000),
    lastSignedIn: new Date(row.lastSignedIn * 1000),
  };
}

// Video operations
export async function createVideo(video: Omit<Video, 'id' | 'createdAt' | 'updatedAt'>): Promise<number> {
  const result = sqlite.prepare(`
    INSERT INTO videos (userId, title, description, originalUrl, fileKey, duration, fps, width, height, frameCount, fileSize, mimeType)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    video.userId, video.title, video.description || null, video.originalUrl, video.fileKey,
    video.duration || null, video.fps || null, video.width || null, video.height || null,
    video.frameCount || null, video.fileSize || null, video.mimeType || null
  );
  return result.lastInsertRowid as number;
}

export async function getVideoById(id: number): Promise<Video | undefined> {
  const row = sqlite.prepare("SELECT * FROM videos WHERE id = ?").get(id) as any;
  if (!row) return undefined;
  return {
    ...row,
    createdAt: new Date(row.createdAt * 1000),
    updatedAt: new Date(row.updatedAt * 1000),
  };
}

export async function getVideosByUserId(userId: number): Promise<Video[]> {
  const rows = sqlite.prepare("SELECT * FROM videos WHERE userId = ? ORDER BY createdAt DESC").all(userId) as any[];
  return rows.map(row => ({
    ...row,
    createdAt: new Date(row.createdAt * 1000),
    updatedAt: new Date(row.updatedAt * 1000),
  }));
}

// Analysis operations
export async function createAnalysis(analysis: Omit<Analysis, 'id' | 'createdAt' | 'updatedAt'>): Promise<number> {
  const result = sqlite.prepare(`
    INSERT INTO analyses (videoId, userId, mode, status, progress, currentStage)
    VALUES (?, ?, ?, ?, ?, ?)
  `).run(analysis.videoId, analysis.userId, analysis.mode, analysis.status || 'pending', analysis.progress || 0, analysis.currentStage || null);
  return result.lastInsertRowid as number;
}

export async function getAnalysisById(id: number): Promise<Analysis | undefined> {
  const row = sqlite.prepare("SELECT * FROM analyses WHERE id = ?").get(id) as any;
  if (!row) return undefined;
  return {
    ...row,
    startedAt: row.startedAt ? new Date(row.startedAt * 1000) : null,
    completedAt: row.completedAt ? new Date(row.completedAt * 1000) : null,
    createdAt: new Date(row.createdAt * 1000),
    updatedAt: new Date(row.updatedAt * 1000),
  };
}

export async function getAnalysesByVideoId(videoId: number): Promise<Analysis[]> {
  const rows = sqlite.prepare("SELECT * FROM analyses WHERE videoId = ? ORDER BY createdAt DESC").all(videoId) as any[];
  return rows.map(row => ({
    ...row,
    startedAt: row.startedAt ? new Date(row.startedAt * 1000) : null,
    completedAt: row.completedAt ? new Date(row.completedAt * 1000) : null,
    createdAt: new Date(row.createdAt * 1000),
    updatedAt: new Date(row.updatedAt * 1000),
  }));
}

export async function getAnalysesByUserId(userId: number): Promise<Analysis[]> {
  const rows = sqlite.prepare("SELECT * FROM analyses WHERE userId = ? ORDER BY createdAt DESC").all(userId) as any[];
  return rows.map(row => ({
    ...row,
    startedAt: row.startedAt ? new Date(row.startedAt * 1000) : null,
    completedAt: row.completedAt ? new Date(row.completedAt * 1000) : null,
    createdAt: new Date(row.createdAt * 1000),
    updatedAt: new Date(row.updatedAt * 1000),
  }));
}

export async function updateAnalysisStatus(
  id: number,
  status: string,
  progress?: number,
  currentStage?: string,
  errorMessage?: string
): Promise<void> {
  sqlite.prepare(`
    UPDATE analyses SET 
      status = ?,
      progress = COALESCE(?, progress),
      currentStage = COALESCE(?, currentStage),
      errorMessage = ?,
      updatedAt = strftime('%s', 'now')
    WHERE id = ?
  `).run(status, progress, currentStage, errorMessage || null, id);
}

export async function updateAnalysisResults(
  id: number,
  results: {
    annotatedVideoUrl?: string;
    radarVideoUrl?: string;
    trackingDataUrl?: string;
    analyticsDataUrl?: string;
    processingTimeMs?: number;
  }
): Promise<void> {
  sqlite.prepare(`
    UPDATE analyses SET 
      annotatedVideoUrl = COALESCE(?, annotatedVideoUrl),
      radarVideoUrl = COALESCE(?, radarVideoUrl),
      trackingDataUrl = COALESCE(?, trackingDataUrl),
      analyticsDataUrl = COALESCE(?, analyticsDataUrl),
      processingTimeMs = COALESCE(?, processingTimeMs),
      completedAt = strftime('%s', 'now'),
      updatedAt = strftime('%s', 'now')
    WHERE id = ?
  `).run(
    results.annotatedVideoUrl || null,
    results.radarVideoUrl || null,
    results.trackingDataUrl || null,
    results.analyticsDataUrl || null,
    results.processingTimeMs || null,
    id
  );
}

// Statistics operations
export async function getStatisticsByAnalysisId(analysisId: number): Promise<Statistics | undefined> {
  const row = sqlite.prepare("SELECT * FROM statistics WHERE analysisId = ?").get(analysisId) as any;
  if (!row) return undefined;
  return {
    ...row,
    heatmapDataTeam1: row.heatmapDataTeam1 ? JSON.parse(row.heatmapDataTeam1) : null,
    heatmapDataTeam2: row.heatmapDataTeam2 ? JSON.parse(row.heatmapDataTeam2) : null,
    passNetworkTeam1: row.passNetworkTeam1 ? JSON.parse(row.passNetworkTeam1) : null,
    passNetworkTeam2: row.passNetworkTeam2 ? JSON.parse(row.passNetworkTeam2) : null,
    createdAt: new Date(row.createdAt * 1000),
    updatedAt: new Date(row.updatedAt * 1000),
  };
}

// Events operations
export async function getEventsByAnalysisId(analysisId: number): Promise<Event[]> {
  const rows = sqlite.prepare("SELECT * FROM events WHERE analysisId = ? ORDER BY frameNumber").all(analysisId) as any[];
  return rows.map(row => ({
    ...row,
    success: row.success === 1,
    metadata: row.metadata ? JSON.parse(row.metadata) : null,
    createdAt: new Date(row.createdAt * 1000),
  }));
}

// Tracks operations
export async function getTracksByAnalysisId(analysisId: number): Promise<Track[]> {
  const rows = sqlite.prepare("SELECT * FROM tracks WHERE analysisId = ? ORDER BY frameNumber").all(analysisId) as any[];
  return rows.map(row => ({
    ...row,
    playerPositions: row.playerPositions ? JSON.parse(row.playerPositions) : null,
    ballPosition: row.ballPosition ? JSON.parse(row.ballPosition) : null,
    teamFormations: row.teamFormations ? JSON.parse(row.teamFormations) : null,
    voronoiData: row.voronoiData ? JSON.parse(row.voronoiData) : null,
    createdAt: new Date(row.createdAt * 1000),
  }));
}

// Commentary operations
export async function getCommentaryByAnalysisId(analysisId: number): Promise<Commentary[]> {
  const rows = sqlite.prepare("SELECT * FROM commentary WHERE analysisId = ? ORDER BY frameStart").all(analysisId) as any[];
  return rows.map(row => ({
    ...row,
    groundingData: row.groundingData ? JSON.parse(row.groundingData) : null,
    createdAt: new Date(row.createdAt * 1000),
  }));
}

export async function createCommentary(commentary: Omit<Commentary, 'id' | 'createdAt'>): Promise<number> {
  const result = sqlite.prepare(`
    INSERT INTO commentary (analysisId, eventId, frameStart, frameEnd, type, content, confidence, groundingData)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    commentary.analysisId,
    commentary.eventId || null,
    commentary.frameStart || null,
    commentary.frameEnd || null,
    commentary.type,
    commentary.content,
    commentary.confidence || null,
    commentary.groundingData ? JSON.stringify(commentary.groundingData) : null
  );
  return result.lastInsertRowid as number;
}

// Export getDb for compatibility
export async function getDb() {
  return sqlite;
}


// Worker API functions

// Get pending analyses for worker
export async function getPendingAnalyses(): Promise<Array<Analysis & { videoUrl: string }>> {
  const rows = sqlite.prepare(`
    SELECT a.*, v.originalUrl as videoUrl 
    FROM analyses a 
    JOIN videos v ON a.videoId = v.id 
    WHERE a.status = 'pending' 
    ORDER BY a.createdAt ASC
  `).all() as any[];
  
  return rows.map(row => ({
    ...row,
    videoUrl: row.videoUrl,
    startedAt: row.startedAt ? new Date(row.startedAt * 1000) : null,
    completedAt: row.completedAt ? new Date(row.completedAt * 1000) : null,
    createdAt: new Date(row.createdAt * 1000),
    updatedAt: new Date(row.updatedAt * 1000),
  }));
}

// Statistics creation
export async function createStatistics(stats: Omit<Statistics, 'id' | 'createdAt' | 'updatedAt'>): Promise<number> {
  const result = sqlite.prepare(`
    INSERT INTO statistics (
      analysisId, possessionTeam1, possessionTeam2, passesTeam1, passesTeam2,
      passAccuracyTeam1, passAccuracyTeam2, shotsTeam1, shotsTeam2,
      distanceCoveredTeam1, distanceCoveredTeam2, avgSpeedTeam1, avgSpeedTeam2,
      heatmapDataTeam1, heatmapDataTeam2, passNetworkTeam1, passNetworkTeam2
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(
    stats.analysisId,
    stats.possessionTeam1 ?? null,
    stats.possessionTeam2 ?? null,
    stats.passesTeam1 ?? null,
    stats.passesTeam2 ?? null,
    stats.passAccuracyTeam1 ?? null,
    stats.passAccuracyTeam2 ?? null,
    stats.shotsTeam1 ?? null,
    stats.shotsTeam2 ?? null,
    stats.distanceCoveredTeam1 ?? null,
    stats.distanceCoveredTeam2 ?? null,
    stats.avgSpeedTeam1 ?? null,
    stats.avgSpeedTeam2 ?? null,
    stats.heatmapDataTeam1 ? JSON.stringify(stats.heatmapDataTeam1) : null,
    stats.heatmapDataTeam2 ? JSON.stringify(stats.heatmapDataTeam2) : null,
    stats.passNetworkTeam1 ? JSON.stringify(stats.passNetworkTeam1) : null,
    stats.passNetworkTeam2 ? JSON.stringify(stats.passNetworkTeam2) : null
  );
  return result.lastInsertRowid as number;
}

// Video hash cache table
sqlite.exec(`
  CREATE TABLE IF NOT EXISTS video_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    videoHash TEXT NOT NULL,
    mode TEXT NOT NULL,
    modelConfig TEXT,
    results TEXT NOT NULL,
    createdAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    UNIQUE(videoHash, mode, modelConfig)
  );
`);

export interface CachedResult {
  id: number;
  videoHash: string;
  mode: string;
  modelConfig: any;
  results: any;
  createdAt: Date;
}

// Get cached result by video hash
export async function getCachedResult(videoHash: string): Promise<CachedResult | null> {
  const row = sqlite.prepare(`
    SELECT * FROM video_cache WHERE videoHash = ? ORDER BY createdAt DESC LIMIT 1
  `).get(videoHash) as any;
  
  if (!row) return null;
  
  return {
    ...row,
    modelConfig: row.modelConfig ? JSON.parse(row.modelConfig) : null,
    results: JSON.parse(row.results),
    createdAt: new Date(row.createdAt * 1000),
  };
}

// Save result to cache
export async function saveCachedResult(
  videoHash: string,
  mode: string,
  modelConfig: any,
  results: any
): Promise<void> {
  sqlite.prepare(`
    INSERT OR REPLACE INTO video_cache (videoHash, mode, modelConfig, results)
    VALUES (?, ?, ?, ?)
  `).run(
    videoHash,
    mode,
    modelConfig ? JSON.stringify(modelConfig) : null,
    JSON.stringify(results)
  );
}
