/**
 * Shared types for the football analysis platform
 */

// Pipeline modes
export const PIPELINE_MODES = {
  all: {
    id: "all",
    name: "Full Analysis",
    description: "Complete pipeline: detection, tracking, team classification, pitch mapping, and analytics",
    outputs: ["annotated_video", "radar_video", "tracking_data", "analytics"],
    icon: "Layers",
  },
  radar: {
    id: "radar",
    name: "Radar View",
    description: "2D pitch visualization with player positions and ball trajectory",
    outputs: ["radar_video", "tracking_data"],
    icon: "Radar",
  },
  team: {
    id: "team",
    name: "Team Analysis",
    description: "Team classification and formation detection using SigLIP embeddings",
    outputs: ["annotated_video", "team_data"],
    icon: "Users",
  },
  track: {
    id: "track",
    name: "Object Tracking",
    description: "Player and ball tracking with ByteTrack persistence",
    outputs: ["annotated_video", "tracking_data"],
    icon: "Target",
  },
  players: {
    id: "players",
    name: "Player Detection",
    description: "YOLOv8 player and referee detection with bounding boxes",
    outputs: ["annotated_video"],
    icon: "User",
  },
  ball: {
    id: "ball",
    name: "Ball Tracking",
    description: "Ball detection with SAHI slicer and trajectory interpolation",
    outputs: ["annotated_video", "ball_data"],
    icon: "Circle",
  },
  pitch: {
    id: "pitch",
    name: "Pitch Mapping",
    description: "Keypoint detection and homography transformation to pitch coordinates",
    outputs: ["pitch_overlay", "homography_data"],
    icon: "Map",
  },
} as const;

export type PipelineMode = keyof typeof PIPELINE_MODES;

// Processing stages
export const PROCESSING_STAGES = [
  { id: "upload", name: "Uploading Video", weight: 5 },
  { id: "load", name: "Loading Frames", weight: 10 },
  { id: "detect", name: "Detecting Objects", weight: 25 },
  { id: "track", name: "Tracking Players", weight: 20 },
  { id: "team", name: "Classifying Teams", weight: 15 },
  { id: "pitch", name: "Mapping Pitch", weight: 10 },
  { id: "analytics", name: "Computing Analytics", weight: 10 },
  { id: "render", name: "Rendering Output", weight: 5 },
] as const;

export type ProcessingStage = typeof PROCESSING_STAGES[number]["id"];

// Event types
export const EVENT_TYPES = {
  pass: { name: "Pass", color: "#22c55e", icon: "ArrowRight" },
  shot: { name: "Shot", color: "#ef4444", icon: "Target" },
  challenge: { name: "Challenge", color: "#f59e0b", icon: "Swords" },
  interception: { name: "Interception", color: "#3b82f6", icon: "Shield" },
  dribble: { name: "Dribble", color: "#8b5cf6", icon: "Footprints" },
  cross: { name: "Cross", color: "#06b6d4", icon: "ArrowUpRight" },
  clearance: { name: "Clearance", color: "#64748b", icon: "ArrowUp" },
  foul: { name: "Foul", color: "#dc2626", icon: "AlertTriangle" },
} as const;

export type EventType = keyof typeof EVENT_TYPES;

// Player position type
export interface PlayerPosition {
  id: number;
  trackId: number;
  teamId: number;
  x: number;
  y: number;
  pixelX: number;
  pixelY: number;
  speed?: number;
  direction?: number;
  jerseyNumber?: number;
}

// Ball position type
export interface BallPosition {
  x: number;
  y: number;
  pixelX: number;
  pixelY: number;
  confidence: number;
  interpolated: boolean;
}

// Frame data type
export interface FrameData {
  frameNumber: number;
  timestamp: number;
  players: PlayerPosition[];
  ball: BallPosition | null;
  possession?: number;
}

// Heatmap data type
export interface HeatmapData {
  grid: number[][];
  maxValue: number;
  minValue: number;
  resolution: { x: number; y: number };
}

// Pass network node
export interface PassNetworkNode {
  playerId: number;
  avgX: number;
  avgY: number;
  passesReceived: number;
  passesMade: number;
}

// Pass network edge
export interface PassNetworkEdge {
  from: number;
  to: number;
  count: number;
  successRate: number;
}

// Pass network data
export interface PassNetworkData {
  nodes: PassNetworkNode[];
  edges: PassNetworkEdge[];
}

// Match statistics
export interface MatchStatistics {
  possession: { team1: number; team2: number };
  passes: { team1: number; team2: number };
  passAccuracy: { team1: number; team2: number };
  shots: { team1: number; team2: number };
  distanceCovered: { team1: number; team2: number };
  avgSpeed: { team1: number; team2: number };
  maxSpeed: { team1: number; team2: number };
}

// Voronoi cell
export interface VoronoiCell {
  playerId: number;
  teamId: number;
  vertices: [number, number][];
  area: number;
}

// API response types
export interface AnalysisResponse {
  id: number;
  videoId: number;
  mode: PipelineMode;
  status: "pending" | "uploading" | "processing" | "completed" | "failed";
  progress: number;
  currentStage: string | null;
  errorMessage: string | null;
  annotatedVideoUrl: string | null;
  radarVideoUrl: string | null;
  trackingDataUrl: string | null;
  analyticsDataUrl: string | null;
  createdAt: Date;
  completedAt: Date | null;
}

export interface VideoResponse {
  id: number;
  title: string;
  description: string | null;
  originalUrl: string;
  duration: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  frameCount: number | null;
  createdAt: Date;
}
