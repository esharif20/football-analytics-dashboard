import { eq, desc, and } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { 
  InsertUser, users, 
  videos, InsertVideo, Video,
  analyses, InsertAnalysis, Analysis,
  events, InsertEvent, Event,
  tracks, InsertTrack, Track,
  statistics, InsertStatistics, Statistics,
  commentary, InsertCommentary, Commentary,
  PipelineMode, ProcessingStatus
} from "../drizzle/schema";
import { ENV } from './_core/env';

let _db: ReturnType<typeof drizzle> | null = null;

export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

// ==================== User Queries ====================

export async function upsertUser(user: InsertUser): Promise<void> {
  if (!user.openId) {
    throw new Error("User openId is required for upsert");
  }

  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot upsert user: database not available");
    return;
  }

  try {
    const values: InsertUser = {
      openId: user.openId,
    };
    const updateSet: Record<string, unknown> = {};

    const textFields = ["name", "email", "loginMethod"] as const;
    type TextField = (typeof textFields)[number];

    const assignNullable = (field: TextField) => {
      const value = user[field];
      if (value === undefined) return;
      const normalized = value ?? null;
      values[field] = normalized;
      updateSet[field] = normalized;
    };

    textFields.forEach(assignNullable);

    if (user.lastSignedIn !== undefined) {
      values.lastSignedIn = user.lastSignedIn;
      updateSet.lastSignedIn = user.lastSignedIn;
    }
    if (user.role !== undefined) {
      values.role = user.role;
      updateSet.role = user.role;
    } else if (user.openId === ENV.ownerOpenId) {
      values.role = 'admin';
      updateSet.role = 'admin';
    }

    if (!values.lastSignedIn) {
      values.lastSignedIn = new Date();
    }

    if (Object.keys(updateSet).length === 0) {
      updateSet.lastSignedIn = new Date();
    }

    await db.insert(users).values(values).onDuplicateKeyUpdate({
      set: updateSet,
    });
  } catch (error) {
    console.error("[Database] Failed to upsert user:", error);
    throw error;
  }
}

export async function getUserByOpenId(openId: string) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot get user: database not available");
    return undefined;
  }

  const result = await db.select().from(users).where(eq(users.openId, openId)).limit(1);
  return result.length > 0 ? result[0] : undefined;
}

// ==================== Video Queries ====================

export async function createVideo(video: InsertVideo): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const result = await db.insert(videos).values(video);
  return result[0].insertId;
}

export async function getVideoById(id: number): Promise<Video | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(videos).where(eq(videos.id, id)).limit(1);
  return result[0];
}

export async function getVideosByUserId(userId: number): Promise<Video[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(videos).where(eq(videos.userId, userId)).orderBy(desc(videos.createdAt));
}

export async function deleteVideo(id: number): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  await db.delete(videos).where(eq(videos.id, id));
}

// ==================== Analysis Queries ====================

export async function createAnalysis(analysis: InsertAnalysis): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const result = await db.insert(analyses).values(analysis);
  return result[0].insertId;
}

export async function getAnalysisById(id: number): Promise<Analysis | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(analyses).where(eq(analyses.id, id)).limit(1);
  return result[0];
}

export async function getAnalysesByVideoId(videoId: number): Promise<Analysis[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(analyses).where(eq(analyses.videoId, videoId)).orderBy(desc(analyses.createdAt));
}

export async function getAnalysesByUserId(userId: number): Promise<Analysis[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(analyses).where(eq(analyses.userId, userId)).orderBy(desc(analyses.createdAt));
}

export async function updateAnalysisStatus(
  id: number, 
  status: ProcessingStatus, 
  progress: number, 
  currentStage?: string,
  errorMessage?: string
): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  const updateData: Partial<Analysis> = { status, progress };
  if (currentStage) updateData.currentStage = currentStage;
  if (errorMessage) updateData.errorMessage = errorMessage;
  if (status === "processing" && progress === 0) {
    updateData.startedAt = new Date();
  }
  if (status === "completed" || status === "failed") {
    updateData.completedAt = new Date();
  }
  
  await db.update(analyses).set(updateData).where(eq(analyses.id, id));
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
  const db = await getDb();
  if (!db) return;
  
  await db.update(analyses).set(results).where(eq(analyses.id, id));
}

// ==================== Worker Queries ====================

export async function getPendingAnalyses(): Promise<Array<{
  id: number;
  videoId: number;
  videoUrl: string;
  mode: string;
  modelConfig: Record<string, string>;
}>> {
  const db = await getDb();
  if (!db) return [];
  
  // Get pending analyses with their video URLs
  const result = await db
    .select({
      id: analyses.id,
      videoId: analyses.videoId,
      mode: analyses.mode,
      videoUrl: videos.originalUrl,
    })
    .from(analyses)
    .innerJoin(videos, eq(analyses.videoId, videos.id))
    .where(eq(analyses.status, "pending"))
    .orderBy(analyses.createdAt)
    .limit(10);
  
  return result.map(r => ({
    id: r.id,
    videoId: r.videoId,
    videoUrl: r.videoUrl,
    mode: r.mode,
    modelConfig: {},
  }));
}

// ==================== Event Queries ====================

export async function createEvents(eventList: InsertEvent[]): Promise<void> {
  const db = await getDb();
  if (!db || eventList.length === 0) return;
  
  await db.insert(events).values(eventList);
}

export async function getEventsByAnalysisId(analysisId: number): Promise<Event[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(events).where(eq(events.analysisId, analysisId)).orderBy(events.frameNumber);
}

export async function getEventsByType(analysisId: number, type: string): Promise<Event[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(events)
    .where(and(eq(events.analysisId, analysisId), eq(events.type, type)))
    .orderBy(events.frameNumber);
}

// ==================== Track Queries ====================

export async function createTracks(trackList: InsertTrack[]): Promise<void> {
  const db = await getDb();
  if (!db || trackList.length === 0) return;
  
  // Insert in batches to avoid query size limits
  const batchSize = 100;
  for (let i = 0; i < trackList.length; i += batchSize) {
    const batch = trackList.slice(i, i + batchSize);
    await db.insert(tracks).values(batch);
  }
}

export async function getTracksByAnalysisId(analysisId: number): Promise<Track[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(tracks).where(eq(tracks.analysisId, analysisId)).orderBy(tracks.frameNumber);
}

export async function getTrackAtFrame(analysisId: number, frameNumber: number): Promise<Track | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(tracks)
    .where(and(eq(tracks.analysisId, analysisId), eq(tracks.frameNumber, frameNumber)))
    .limit(1);
  return result[0];
}

// ==================== Statistics Queries ====================

export async function createStatistics(stats: InsertStatistics): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const result = await db.insert(statistics).values(stats);
  return result[0].insertId;
}

export async function getStatisticsByAnalysisId(analysisId: number): Promise<Statistics | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(statistics).where(eq(statistics.analysisId, analysisId)).limit(1);
  return result[0];
}

export async function updateStatistics(analysisId: number, stats: Partial<InsertStatistics>): Promise<void> {
  const db = await getDb();
  if (!db) return;
  
  await db.update(statistics).set(stats).where(eq(statistics.analysisId, analysisId));
}

// ==================== Commentary Queries ====================

export async function createCommentary(commentaryItem: InsertCommentary): Promise<number> {
  const db = await getDb();
  if (!db) throw new Error("Database not available");
  
  const result = await db.insert(commentary).values(commentaryItem);
  return result[0].insertId;
}

export async function getCommentaryByAnalysisId(analysisId: number): Promise<Commentary[]> {
  const db = await getDb();
  if (!db) return [];
  
  return db.select().from(commentary).where(eq(commentary.analysisId, analysisId)).orderBy(commentary.frameStart);
}

export async function getCommentaryByEventId(eventId: number): Promise<Commentary | undefined> {
  const db = await getDb();
  if (!db) return undefined;
  
  const result = await db.select().from(commentary).where(eq(commentary.eventId, eventId)).limit(1);
  return result[0];
}
