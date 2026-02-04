import { int, mysqlEnum, mysqlTable, text, timestamp, varchar, json, float, boolean } from "drizzle-orm/mysql-core";

/**
 * Core user table backing auth flow.
 */
export const users = mysqlTable("users", {
  id: int("id").autoincrement().primaryKey(),
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Pipeline modes available for video analysis
 */
export const pipelineModes = ["all", "radar", "team", "track", "players", "ball", "pitch"] as const;
export type PipelineMode = typeof pipelineModes[number];

/**
 * Processing status for video analysis
 */
export const processingStatuses = ["pending", "uploading", "processing", "completed", "failed"] as const;
export type ProcessingStatus = typeof processingStatuses[number];

/**
 * Videos table - stores uploaded match footage
 */
export const videos = mysqlTable("videos", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  title: varchar("title", { length: 255 }).notNull(),
  description: text("description"),
  originalUrl: text("originalUrl").notNull(),
  fileKey: varchar("fileKey", { length: 512 }).notNull(),
  duration: float("duration"),
  fps: float("fps"),
  width: int("width"),
  height: int("height"),
  frameCount: int("frameCount"),
  fileSize: int("fileSize"),
  mimeType: varchar("mimeType", { length: 64 }),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Video = typeof videos.$inferSelect;
export type InsertVideo = typeof videos.$inferInsert;

/**
 * Analyses table - stores pipeline processing jobs
 */
export const analyses = mysqlTable("analyses", {
  id: int("id").autoincrement().primaryKey(),
  videoId: int("videoId").notNull(),
  userId: int("userId").notNull(),
  mode: mysqlEnum("mode", pipelineModes).notNull(),
  status: mysqlEnum("status", processingStatuses).default("pending").notNull(),
  progress: int("progress").default(0).notNull(),
  currentStage: varchar("currentStage", { length: 128 }),
  errorMessage: text("errorMessage"),
  // Output files
  annotatedVideoUrl: text("annotatedVideoUrl"),
  radarVideoUrl: text("radarVideoUrl"),
  trackingDataUrl: text("trackingDataUrl"),
  analyticsDataUrl: text("analyticsDataUrl"),
  // Processing metadata
  startedAt: timestamp("startedAt"),
  completedAt: timestamp("completedAt"),
  processingTimeMs: int("processingTimeMs"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Analysis = typeof analyses.$inferSelect;
export type InsertAnalysis = typeof analyses.$inferInsert;

/**
 * Events table - detected match events (passes, shots, challenges, etc.)
 */
export const events = mysqlTable("events", {
  id: int("id").autoincrement().primaryKey(),
  analysisId: int("analysisId").notNull(),
  type: varchar("type", { length: 64 }).notNull(),
  frameNumber: int("frameNumber").notNull(),
  timestamp: float("timestamp").notNull(),
  playerId: int("playerId"),
  teamId: int("teamId"),
  targetPlayerId: int("targetPlayerId"),
  startX: float("startX"),
  startY: float("startY"),
  endX: float("endX"),
  endY: float("endY"),
  success: boolean("success"),
  confidence: float("confidence"),
  metadata: json("metadata"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type Event = typeof events.$inferSelect;
export type InsertEvent = typeof events.$inferInsert;

/**
 * Tracks table - stores tracking data snapshots for quick access
 */
export const tracks = mysqlTable("tracks", {
  id: int("id").autoincrement().primaryKey(),
  analysisId: int("analysisId").notNull(),
  frameNumber: int("frameNumber").notNull(),
  timestamp: float("timestamp").notNull(),
  playerPositions: json("playerPositions"),
  ballPosition: json("ballPosition"),
  teamFormations: json("teamFormations"),
  voronoiData: json("voronoiData"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type Track = typeof tracks.$inferSelect;
export type InsertTrack = typeof tracks.$inferInsert;

/**
 * Statistics table - aggregated match statistics
 */
export const statistics = mysqlTable("statistics", {
  id: int("id").autoincrement().primaryKey(),
  analysisId: int("analysisId").notNull(),
  possessionTeam1: float("possessionTeam1"),
  possessionTeam2: float("possessionTeam2"),
  passesTeam1: int("passesTeam1"),
  passesTeam2: int("passesTeam2"),
  passAccuracyTeam1: float("passAccuracyTeam1"),
  passAccuracyTeam2: float("passAccuracyTeam2"),
  shotsTeam1: int("shotsTeam1"),
  shotsTeam2: int("shotsTeam2"),
  distanceCoveredTeam1: float("distanceCoveredTeam1"),
  distanceCoveredTeam2: float("distanceCoveredTeam2"),
  avgSpeedTeam1: float("avgSpeedTeam1"),
  avgSpeedTeam2: float("avgSpeedTeam2"),
  heatmapDataTeam1: json("heatmapDataTeam1"),
  heatmapDataTeam2: json("heatmapDataTeam2"),
  passNetworkTeam1: json("passNetworkTeam1"),
  passNetworkTeam2: json("passNetworkTeam2"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Statistics = typeof statistics.$inferSelect;
export type InsertStatistics = typeof statistics.$inferInsert;

/**
 * Commentary table - AI-generated tactical analysis
 */
export const commentary = mysqlTable("commentary", {
  id: int("id").autoincrement().primaryKey(),
  analysisId: int("analysisId").notNull(),
  eventId: int("eventId"),
  frameStart: int("frameStart"),
  frameEnd: int("frameEnd"),
  type: varchar("type", { length: 64 }).notNull(),
  content: text("content").notNull(),
  confidence: float("confidence"),
  groundingData: json("groundingData"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type Commentary = typeof commentary.$inferSelect;
export type InsertCommentary = typeof commentary.$inferInsert;
