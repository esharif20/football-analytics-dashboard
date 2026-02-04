import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { z } from "zod";
import { TRPCError } from "@trpc/server";
import { 
  createVideo, getVideoById, getVideosByUserId, deleteVideo,
  createAnalysis, getAnalysisById, getAnalysesByVideoId, getAnalysesByUserId,
  updateAnalysisStatus, updateAnalysisResults,
  createEvents, getEventsByAnalysisId, getEventsByType,
  createTracks, getTracksByAnalysisId, getTrackAtFrame,
  createStatistics, getStatisticsByAnalysisId, updateStatistics,
  createCommentary, getCommentaryByAnalysisId
} from "./db";
import { storagePut } from "./storage";
import { invokeLLM } from "./_core/llm";
import { pipelineModes, processingStatuses } from "../drizzle/schema";
import { PIPELINE_MODES, PROCESSING_STAGES } from "../shared/types";
import { nanoid } from "nanoid";

export const appRouter = router({
  system: systemRouter,
  
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return { success: true } as const;
    }),
  }),

  // Video management
  video: router({
    list: protectedProcedure.query(async ({ ctx }) => {
      return getVideosByUserId(ctx.user.id);
    }),

    get: protectedProcedure
      .input(z.object({ id: z.number() }))
      .query(async ({ ctx, input }) => {
        const video = await getVideoById(input.id);
        if (!video || video.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Video not found" });
        }
        return video;
      }),

    upload: protectedProcedure
      .input(z.object({
        title: z.string().min(1).max(255),
        description: z.string().optional(),
        fileName: z.string(),
        fileSize: z.number(),
        mimeType: z.string(),
        fileBase64: z.string(),
      }))
      .mutation(async ({ ctx, input }) => {
        const fileBuffer = Buffer.from(input.fileBase64, "base64");
        const fileKey = `videos/${ctx.user.id}/${nanoid()}-${input.fileName}`;
        
        const { url } = await storagePut(fileKey, fileBuffer, input.mimeType);
        
        const videoId = await createVideo({
          userId: ctx.user.id,
          title: input.title,
          description: input.description || null,
          originalUrl: url,
          fileKey,
          fileSize: input.fileSize,
          mimeType: input.mimeType,
        });
        
        return { id: videoId, url };
      }),

    delete: protectedProcedure
      .input(z.object({ id: z.number() }))
      .mutation(async ({ ctx, input }) => {
        const video = await getVideoById(input.id);
        if (!video || video.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Video not found" });
        }
        await deleteVideo(input.id);
        return { success: true };
      }),
  }),

  // Analysis management
  analysis: router({
    list: protectedProcedure.query(async ({ ctx }) => {
      return getAnalysesByUserId(ctx.user.id);
    }),

    listByVideo: protectedProcedure
      .input(z.object({ videoId: z.number() }))
      .query(async ({ ctx, input }) => {
        const video = await getVideoById(input.videoId);
        if (!video || video.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Video not found" });
        }
        return getAnalysesByVideoId(input.videoId);
      }),

    get: protectedProcedure
      .input(z.object({ id: z.number() }))
      .query(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.id);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        return analysis;
      }),

    create: protectedProcedure
      .input(z.object({
        videoId: z.number(),
        mode: z.enum(pipelineModes),
      }))
      .mutation(async ({ ctx, input }) => {
        const video = await getVideoById(input.videoId);
        if (!video || video.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Video not found" });
        }
        
        const analysisId = await createAnalysis({
          videoId: input.videoId,
          userId: ctx.user.id,
          mode: input.mode,
          status: "pending",
          progress: 0,
        });
        
        return { id: analysisId };
      }),

    updateStatus: protectedProcedure
      .input(z.object({
        id: z.number(),
        status: z.enum(processingStatuses),
        progress: z.number().min(0).max(100),
        currentStage: z.string().optional(),
        errorMessage: z.string().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.id);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        
        await updateAnalysisStatus(
          input.id,
          input.status,
          input.progress,
          input.currentStage,
          input.errorMessage
        );
        
        return { success: true };
      }),

    updateResults: protectedProcedure
      .input(z.object({
        id: z.number(),
        annotatedVideoUrl: z.string().optional(),
        radarVideoUrl: z.string().optional(),
        trackingDataUrl: z.string().optional(),
        analyticsDataUrl: z.string().optional(),
        processingTimeMs: z.number().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.id);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        
        const { id, ...results } = input;
        await updateAnalysisResults(id, results);
        
        return { success: true };
      }),

    getModes: publicProcedure.query(() => {
      return PIPELINE_MODES;
    }),

    getStages: publicProcedure.query(() => {
      return PROCESSING_STAGES;
    }),

    // Terminate a running analysis
    terminate: protectedProcedure
      .input(z.object({ id: z.number() }))
      .mutation(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.id);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        
        if (analysis.status !== "processing" && analysis.status !== "pending") {
          throw new TRPCError({ code: "BAD_REQUEST", message: "Analysis is not running" });
        }
        
        await updateAnalysisStatus(
          input.id,
          "failed",
          analysis.progress,
          analysis.currentStage || undefined,
          "Terminated by user"
        );
        
        return { success: true };
      }),

    // Get ETA for processing
    getEta: protectedProcedure
      .input(z.object({ id: z.number() }))
      .query(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.id);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        
        // Calculate ETA based on progress and elapsed time
        const startTime = analysis.createdAt?.getTime() || Date.now();
        const elapsed = Date.now() - startTime;
        const progress = analysis.progress || 1;
        
        // Estimate total time based on current progress
        const estimatedTotal = (elapsed / progress) * 100;
        const remaining = Math.max(0, estimatedTotal - elapsed);
        
        // Stage-based estimates (in seconds)
        const stageEstimates: Record<string, number> = {
          uploading: 5,
          loading: 3,
          detecting: 60,
          tracking: 10,
          classifying: 30,
          mapping: 20,
          computing: 15,
          rendering: 45,
        };
        
        const currentStage = analysis.currentStage || "uploading";
        const stageIndex = PROCESSING_STAGES.findIndex(s => s.id === currentStage);
        
        // Calculate remaining time based on stages
        let stageBasedRemaining = 0;
        for (let i = stageIndex; i < PROCESSING_STAGES.length; i++) {
          const stageId = PROCESSING_STAGES[i].id;
          const estimate = stageEstimates[stageId] || 10;
          if (i === stageIndex) {
            // Partial time for current stage
            const stageProgress = (progress % (100 / PROCESSING_STAGES.length)) / (100 / PROCESSING_STAGES.length);
            stageBasedRemaining += estimate * (1 - stageProgress);
          } else {
            stageBasedRemaining += estimate;
          }
        }
        
        // Use weighted average of both estimates
        const finalEstimate = (remaining / 1000 * 0.4) + (stageBasedRemaining * 0.6);
        
        return {
          elapsedMs: elapsed,
          remainingMs: Math.round(finalEstimate * 1000),
          estimatedTotalMs: Math.round((elapsed + finalEstimate * 1000)),
          currentStage,
          stageIndex,
          totalStages: PROCESSING_STAGES.length,
        };
      }),
  }),

  // Events
  events: router({
    list: protectedProcedure
      .input(z.object({ analysisId: z.number() }))
      .query(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.analysisId);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        return getEventsByAnalysisId(input.analysisId);
      }),

    listByType: protectedProcedure
      .input(z.object({ analysisId: z.number(), type: z.string() }))
      .query(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.analysisId);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        return getEventsByType(input.analysisId, input.type);
      }),

    create: protectedProcedure
      .input(z.object({
        analysisId: z.number(),
        events: z.array(z.object({
          type: z.string(),
          frameNumber: z.number(),
          timestamp: z.number(),
          playerId: z.number().optional(),
          teamId: z.number().optional(),
          targetPlayerId: z.number().optional(),
          startX: z.number().optional(),
          startY: z.number().optional(),
          endX: z.number().optional(),
          endY: z.number().optional(),
          success: z.boolean().optional(),
          confidence: z.number().optional(),
          metadata: z.any().optional(),
        })),
      }))
      .mutation(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.analysisId);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        
        const eventsWithAnalysisId = input.events.map(e => ({
          ...e,
          analysisId: input.analysisId,
        }));
        
        await createEvents(eventsWithAnalysisId);
        return { success: true, count: input.events.length };
      }),
  }),

  // Tracks
  tracks: router({
    list: protectedProcedure
      .input(z.object({ analysisId: z.number() }))
      .query(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.analysisId);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        return getTracksByAnalysisId(input.analysisId);
      }),

    getAtFrame: protectedProcedure
      .input(z.object({ analysisId: z.number(), frameNumber: z.number() }))
      .query(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.analysisId);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        return getTrackAtFrame(input.analysisId, input.frameNumber);
      }),

    create: protectedProcedure
      .input(z.object({
        analysisId: z.number(),
        tracks: z.array(z.object({
          frameNumber: z.number(),
          timestamp: z.number(),
          playerPositions: z.any(),
          ballPosition: z.any().optional(),
          teamFormations: z.any().optional(),
          voronoiData: z.any().optional(),
        })),
      }))
      .mutation(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.analysisId);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        
        const tracksWithAnalysisId = input.tracks.map(t => ({
          ...t,
          analysisId: input.analysisId,
        }));
        
        await createTracks(tracksWithAnalysisId);
        return { success: true, count: input.tracks.length };
      }),
  }),

  // Statistics
  statistics: router({
    get: protectedProcedure
      .input(z.object({ analysisId: z.number() }))
      .query(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.analysisId);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        return getStatisticsByAnalysisId(input.analysisId);
      }),

    create: protectedProcedure
      .input(z.object({
        analysisId: z.number(),
        possessionTeam1: z.number().optional(),
        possessionTeam2: z.number().optional(),
        passesTeam1: z.number().optional(),
        passesTeam2: z.number().optional(),
        passAccuracyTeam1: z.number().optional(),
        passAccuracyTeam2: z.number().optional(),
        shotsTeam1: z.number().optional(),
        shotsTeam2: z.number().optional(),
        distanceCoveredTeam1: z.number().optional(),
        distanceCoveredTeam2: z.number().optional(),
        avgSpeedTeam1: z.number().optional(),
        avgSpeedTeam2: z.number().optional(),
        heatmapDataTeam1: z.any().optional(),
        heatmapDataTeam2: z.any().optional(),
        passNetworkTeam1: z.any().optional(),
        passNetworkTeam2: z.any().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.analysisId);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        
        const statsId = await createStatistics(input);
        return { id: statsId };
      }),
  }),

  // AI Commentary
  commentary: router({
    list: protectedProcedure
      .input(z.object({ analysisId: z.number() }))
      .query(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.analysisId);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }
        return getCommentaryByAnalysisId(input.analysisId);
      }),

    generate: protectedProcedure
      .input(z.object({
        analysisId: z.number(),
        type: z.enum(["match_summary", "tactical_analysis", "event_commentary", "player_analysis"]),
        context: z.object({
          events: z.array(z.any()).optional(),
          statistics: z.any().optional(),
          trackingData: z.any().optional(),
          focusPlayerId: z.number().optional(),
          timeRange: z.object({ start: z.number(), end: z.number() }).optional(),
        }),
      }))
      .mutation(async ({ ctx, input }) => {
        const analysis = await getAnalysisById(input.analysisId);
        if (!analysis || analysis.userId !== ctx.user.id) {
          throw new TRPCError({ code: "NOT_FOUND", message: "Analysis not found" });
        }

        const systemPrompt = `You are an expert football tactical analyst. Generate insightful, grounded analysis based ONLY on the provided tracking data and statistics. Do not make up information not present in the data. Be specific about player movements, positions, and tactical patterns you observe.`;

        const userPrompt = buildCommentaryPrompt(input.type, input.context);

        const response = await invokeLLM({
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
        });

        const rawContent = response.choices[0]?.message?.content;
        const content = typeof rawContent === "string" ? rawContent : "Unable to generate commentary.";

        const commentaryId = await createCommentary({
          analysisId: input.analysisId,
          type: input.type,
          content,
          groundingData: input.context,
          frameStart: input.context.timeRange?.start,
          frameEnd: input.context.timeRange?.end,
        });

        return { id: commentaryId, content };
      }),
  }),
});

function buildCommentaryPrompt(type: string, context: any): string {
  const { events, statistics, trackingData, focusPlayerId, timeRange } = context;

  switch (type) {
    case "match_summary":
      return `Provide a tactical summary of this match segment based on the following data:
      
Statistics: ${JSON.stringify(statistics, null, 2)}

Key Events: ${JSON.stringify(events?.slice(0, 20), null, 2)}

Focus on possession patterns, team shape, and key moments. Be specific about what the data shows.`;

    case "tactical_analysis":
      return `Analyze the tactical patterns in this match segment:

Statistics: ${JSON.stringify(statistics, null, 2)}

Events: ${JSON.stringify(events?.slice(0, 30), null, 2)}

Discuss formation, pressing patterns, build-up play, and defensive organization based on the data.`;

    case "event_commentary":
      return `Provide detailed commentary on these specific events:

Events: ${JSON.stringify(events, null, 2)}

Describe each event in context, explaining the tactical significance and player involvement.`;

    case "player_analysis":
      return `Analyze the performance of player ${focusPlayerId} based on this tracking data:

Player Tracking: ${JSON.stringify(trackingData, null, 2)}

Related Events: ${JSON.stringify(events?.filter((e: any) => e.playerId === focusPlayerId), null, 2)}

Discuss positioning, movement patterns, involvement in play, and effectiveness.`;

    default:
      return `Analyze the following football match data and provide insights:

${JSON.stringify(context, null, 2)}`;
  }
}

export type AppRouter = typeof appRouter;
