import { integer, sqliteTable, text, real } from "drizzle-orm/sqlite-core";

/**
 * SQLite schema for standalone local development
 * This allows running the dashboard without MySQL/external database
 */

export const users = sqliteTable("users", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  openId: text("openId").notNull().unique(),
  name: text("name"),
  email: text("email"),
  loginMethod: text("loginMethod"),
  role: text("role", { enum: ["user", "admin"] }).default("user").notNull(),
  createdAt: integer("createdAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
  updatedAt: integer("updatedAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
  lastSignedIn: integer("lastSignedIn", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

export const pipelineModes = ["all", "radar", "team", "track", "players", "ball", "pitch"] as const;
export type PipelineMode = typeof pipelineModes[number];

export const processingStatuses = ["pending", "uploading", "processing", "completed", "failed"] as const;
export type ProcessingStatus = typeof processingStatuses[number];

export const videos = sqliteTable("videos", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  userId: integer("userId").notNull(),
  title: text("title").notNull(),
  description: text("description"),
  originalUrl: text("originalUrl").notNull(),
  fileKey: text("fileKey").notNull(),
  duration: real("duration"),
  fps: real("fps"),
  width: integer("width"),
  height: integer("height"),
  frameCount: integer("frameCount"),
  fileSize: integer("fileSize"),
  mimeType: text("mimeType"),
  createdAt: integer("createdAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
  updatedAt: integer("updatedAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
});

export type Video = typeof videos.$inferSelect;
export type InsertVideo = typeof videos.$inferInsert;

export const analyses = sqliteTable("analyses", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  videoId: integer("videoId").notNull(),
  userId: integer("userId").notNull(),
  mode: text("mode", { enum: pipelineModes }).notNull(),
  status: text("status", { enum: processingStatuses }).default("pending").notNull(),
  progress: integer("progress").default(0).notNull(),
  currentStage: text("currentStage"),
  errorMessage: text("errorMessage"),
  annotatedVideoUrl: text("annotatedVideoUrl"),
  radarVideoUrl: text("radarVideoUrl"),
  trackingDataUrl: text("trackingDataUrl"),
  analyticsDataUrl: text("analyticsDataUrl"),
  startedAt: integer("startedAt", { mode: "timestamp" }),
  completedAt: integer("completedAt", { mode: "timestamp" }),
  processingTimeMs: integer("processingTimeMs"),
  createdAt: integer("createdAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
  updatedAt: integer("updatedAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
});

export type Analysis = typeof analyses.$inferSelect;
export type InsertAnalysis = typeof analyses.$inferInsert;

export const events = sqliteTable("events", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  analysisId: integer("analysisId").notNull(),
  type: text("type").notNull(),
  frameNumber: integer("frameNumber").notNull(),
  timestamp: real("timestamp").notNull(),
  playerId: integer("playerId"),
  teamId: integer("teamId"),
  targetPlayerId: integer("targetPlayerId"),
  startX: real("startX"),
  startY: real("startY"),
  endX: real("endX"),
  endY: real("endY"),
  success: integer("success", { mode: "boolean" }),
  confidence: real("confidence"),
  metadata: text("metadata", { mode: "json" }),
  createdAt: integer("createdAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
});

export type Event = typeof events.$inferSelect;
export type InsertEvent = typeof events.$inferInsert;

export const tracks = sqliteTable("tracks", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  analysisId: integer("analysisId").notNull(),
  frameNumber: integer("frameNumber").notNull(),
  timestamp: real("timestamp").notNull(),
  playerPositions: text("playerPositions", { mode: "json" }),
  ballPosition: text("ballPosition", { mode: "json" }),
  teamFormations: text("teamFormations", { mode: "json" }),
  voronoiData: text("voronoiData", { mode: "json" }),
  createdAt: integer("createdAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
});

export type Track = typeof tracks.$inferSelect;
export type InsertTrack = typeof tracks.$inferInsert;

export const statistics = sqliteTable("statistics", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  analysisId: integer("analysisId").notNull(),
  possessionTeam1: real("possessionTeam1"),
  possessionTeam2: real("possessionTeam2"),
  passesTeam1: integer("passesTeam1"),
  passesTeam2: integer("passesTeam2"),
  passAccuracyTeam1: real("passAccuracyTeam1"),
  passAccuracyTeam2: real("passAccuracyTeam2"),
  shotsTeam1: integer("shotsTeam1"),
  shotsTeam2: integer("shotsTeam2"),
  distanceCoveredTeam1: real("distanceCoveredTeam1"),
  distanceCoveredTeam2: real("distanceCoveredTeam2"),
  avgSpeedTeam1: real("avgSpeedTeam1"),
  avgSpeedTeam2: real("avgSpeedTeam2"),
  heatmapDataTeam1: text("heatmapDataTeam1", { mode: "json" }),
  heatmapDataTeam2: text("heatmapDataTeam2", { mode: "json" }),
  passNetworkTeam1: text("passNetworkTeam1", { mode: "json" }),
  passNetworkTeam2: text("passNetworkTeam2", { mode: "json" }),
  createdAt: integer("createdAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
  updatedAt: integer("updatedAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
});

export type Statistics = typeof statistics.$inferSelect;
export type InsertStatistics = typeof statistics.$inferInsert;

export const commentary = sqliteTable("commentary", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  analysisId: integer("analysisId").notNull(),
  eventId: integer("eventId"),
  frameStart: integer("frameStart"),
  frameEnd: integer("frameEnd"),
  type: text("type").notNull(),
  content: text("content").notNull(),
  confidence: real("confidence"),
  groundingData: text("groundingData", { mode: "json" }),
  createdAt: integer("createdAt", { mode: "timestamp" }).notNull().$defaultFn(() => new Date()),
});

export type Commentary = typeof commentary.$inferSelect;
export type InsertCommentary = typeof commentary.$inferInsert;
