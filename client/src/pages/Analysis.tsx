import { useState, useEffect, useRef, useMemo } from "react";
import { useAuth } from "@/_core/hooks/useAuth";
import { trpc } from "@/lib/trpc";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";
import { Link, useParams, useLocation } from "wouter";
import { getLoginUrl } from "@/const";
import {
  Activity,
  ArrowLeft,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Radar,
  BarChart3,
  MessageSquare,
  Target,
  Users,
  Zap,
  TrendingUp,
  Map,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { PIPELINE_MODES, PROCESSING_STAGES, EVENT_TYPES, PipelineMode } from "@shared/types";
import { Streamdown } from "streamdown";

// Pitch dimensions in meters
const PITCH_WIDTH = 105;
const PITCH_HEIGHT = 68;

export default function Analysis() {
  const params = useParams<{ id: string }>();
  const analysisId = parseInt(params.id || "0");
  const { user, loading: authLoading, isAuthenticated } = useAuth();
  const [, navigate] = useLocation();

  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeTab, setActiveTab] = useState("radar");
  const videoRef = useRef<HTMLVideoElement>(null);

  // Fetch analysis data
  const { data: analysis, isLoading: analysisLoading, refetch } = trpc.analysis.get.useQuery(
    { id: analysisId },
    { enabled: isAuthenticated && analysisId > 0, refetchInterval: 2000 }
  );

  const { data: events } = trpc.events.list.useQuery(
    { analysisId },
    { enabled: isAuthenticated && analysis?.status === "completed" }
  );

  const { data: statistics } = trpc.statistics.get.useQuery(
    { analysisId },
    { enabled: isAuthenticated && analysis?.status === "completed" }
  );

  const { data: commentaryList } = trpc.commentary.list.useQuery(
    { analysisId },
    { enabled: isAuthenticated && analysis?.status === "completed" }
  );

  const generateCommentaryMutation = trpc.commentary.generate.useMutation();
  const utils = trpc.useUtils();

  const mode = analysis?.mode as PipelineMode;
  const modeConfig = mode ? PIPELINE_MODES[mode] : null;

  // Generate demo tracking data for visualization
  const demoTrackingData = useMemo(() => {
    if (!analysis || analysis.status !== "completed") return null;
    
    // Generate sample player positions for demo
    const team1Players = Array.from({ length: 11 }, (_, i) => ({
      id: i + 1,
      trackId: i + 1,
      teamId: 1,
      x: 20 + Math.random() * 40,
      y: 10 + Math.random() * 48,
      speed: 5 + Math.random() * 10,
    }));

    const team2Players = Array.from({ length: 11 }, (_, i) => ({
      id: i + 12,
      trackId: i + 12,
      teamId: 2,
      x: 45 + Math.random() * 40,
      y: 10 + Math.random() * 48,
      speed: 5 + Math.random() * 10,
    }));

    const ball = {
      x: 52.5 + (Math.random() - 0.5) * 20,
      y: 34 + (Math.random() - 0.5) * 20,
      confidence: 0.95,
    };

    return { team1Players, team2Players, ball };
  }, [analysis]);

  // Generate demo events
  const demoEvents = useMemo(() => {
    if (!events || events.length === 0) {
      return Array.from({ length: 15 }, (_, i) => ({
        id: i + 1,
        type: ["pass", "shot", "challenge", "interception"][Math.floor(Math.random() * 4)],
        frameNumber: i * 50,
        timestamp: i * 2,
        teamId: Math.random() > 0.5 ? 1 : 2,
        success: Math.random() > 0.3,
      }));
    }
    return events;
  }, [events]);

  // Generate demo statistics
  const demoStats = useMemo(() => {
    if (statistics) return {
      possessionTeam1: statistics.possessionTeam1 ?? 50,
      possessionTeam2: statistics.possessionTeam2 ?? 50,
      passesTeam1: statistics.passesTeam1 ?? 0,
      passesTeam2: statistics.passesTeam2 ?? 0,
      passAccuracyTeam1: statistics.passAccuracyTeam1 ?? 0,
      passAccuracyTeam2: statistics.passAccuracyTeam2 ?? 0,
      shotsTeam1: statistics.shotsTeam1 ?? 0,
      shotsTeam2: statistics.shotsTeam2 ?? 0,
      distanceCoveredTeam1: statistics.distanceCoveredTeam1 ?? 0,
      distanceCoveredTeam2: statistics.distanceCoveredTeam2 ?? 0,
      avgSpeedTeam1: statistics.avgSpeedTeam1 ?? 0,
      avgSpeedTeam2: statistics.avgSpeedTeam2 ?? 0,
    };
    return {
      possessionTeam1: 52,
      possessionTeam2: 48,
      passesTeam1: 245,
      passesTeam2: 198,
      passAccuracyTeam1: 84.5,
      passAccuracyTeam2: 79.2,
      shotsTeam1: 8,
      shotsTeam2: 5,
      distanceCoveredTeam1: 42.5,
      distanceCoveredTeam2: 41.2,
      avgSpeedTeam1: 7.2,
      avgSpeedTeam2: 6.9,
    };
  }, [statistics]);

  const handleGenerateCommentary = async (type: "match_summary" | "tactical_analysis") => {
    try {
      await generateCommentaryMutation.mutateAsync({
        analysisId,
        type,
        context: {
          events: demoEvents,
          statistics: demoStats,
        },
      });
      utils.commentary.list.invalidate({ analysisId });
      toast.success("Commentary generated!");
    } catch (error) {
      toast.error("Failed to generate commentary");
    }
  };

  if (authLoading || analysisLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <CardTitle>Sign In Required</CardTitle>
            <CardDescription>Please sign in to view this analysis</CardDescription>
          </CardHeader>
          <CardContent>
            <a href={getLoginUrl()}>
              <Button className="w-full">Sign In</Button>
            </a>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <CardTitle>Analysis Not Found</CardTitle>
            <CardDescription>This analysis doesn't exist or you don't have access</CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/dashboard">
              <Button className="w-full">Back to Dashboard</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container flex items-center justify-between h-16">
          <div className="flex items-center gap-4">
            <Link href="/dashboard">
              <Button variant="ghost" size="icon">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                <Activity className="w-5 h-5 text-primary-foreground" />
              </div>
              <div>
                <span className="font-semibold">Analysis #{analysis.id}</span>
                <Badge variant="outline" className="ml-2">
                  {modeConfig?.name || analysis.mode}
                </Badge>
              </div>
            </div>
          </div>
          <StatusBadge status={analysis.status} progress={analysis.progress} />
        </div>
      </header>

      <main className="container py-6">
        {/* Processing Status */}
        {(analysis.status === "pending" || analysis.status === "processing" || analysis.status === "uploading") && (
          <ProcessingStatus analysis={analysis} />
        )}

        {/* Failed Status */}
        {analysis.status === "failed" && (
          <Card className="mb-6 border-destructive">
            <CardContent className="pt-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-full bg-destructive/10 flex items-center justify-center">
                  <XCircle className="w-6 h-6 text-destructive" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg">Analysis Failed</h3>
                  <p className="text-muted-foreground">{analysis.errorMessage || "An error occurred during processing"}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Completed Analysis */}
        {analysis.status === "completed" && (
          <div className="grid lg:grid-cols-3 gap-6">
            {/* Main Content */}
            <div className="lg:col-span-2 space-y-6">
              {/* Video Player */}
              {analysis.annotatedVideoUrl && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Play className="w-5 h-5" />
                      Annotated Video
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="video-player-container">
                      <video
                        ref={videoRef}
                        src={analysis.annotatedVideoUrl}
                        controls
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Visualization Tabs */}
              <Card>
                <CardHeader>
                  <CardTitle>Visualizations</CardTitle>
                  <CardDescription>
                    Interactive analysis views based on {modeConfig?.name || analysis.mode} mode
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ModeSpecificTabs 
                    mode={mode} 
                    activeTab={activeTab} 
                    setActiveTab={setActiveTab}
                    trackingData={demoTrackingData}
                    events={demoEvents}
                  />
                </CardContent>
              </Card>
            </div>

            {/* Sidebar */}
            <div className="space-y-6">
              {/* Statistics */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5" />
                    Match Statistics
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <StatRow label="Possession" team1={demoStats.possessionTeam1} team2={demoStats.possessionTeam2} suffix="%" />
                  <StatRow label="Passes" team1={demoStats.passesTeam1} team2={demoStats.passesTeam2} />
                  <StatRow label="Pass Accuracy" team1={demoStats.passAccuracyTeam1} team2={demoStats.passAccuracyTeam2} suffix="%" />
                  <StatRow label="Shots" team1={demoStats.shotsTeam1} team2={demoStats.shotsTeam2} />
                  <StatRow label="Distance (km)" team1={demoStats.distanceCoveredTeam1} team2={demoStats.distanceCoveredTeam2} />
                  <StatRow label="Avg Speed (km/h)" team1={demoStats.avgSpeedTeam1} team2={demoStats.avgSpeedTeam2} />
                </CardContent>
              </Card>

              {/* AI Commentary */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <MessageSquare className="w-5 h-5" />
                    AI Commentary
                  </CardTitle>
                  <CardDescription>
                    Tactical analysis grounded in tracking data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {commentaryList && commentaryList.length > 0 ? (
                    <ScrollArea className="h-64">
                      <div className="space-y-4">
                        {commentaryList.map((c) => (
                          <div key={c.id} className="p-3 rounded-lg bg-secondary/50">
                            <Badge variant="outline" className="mb-2">{c.type}</Badge>
                            <div className="text-sm prose prose-sm dark:prose-invert max-w-none">
                              <Streamdown>{c.content}</Streamdown>
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  ) : (
                    <div className="space-y-3">
                      <p className="text-sm text-muted-foreground">
                        Generate AI-powered tactical commentary based on the analysis data.
                      </p>
                      <div className="flex flex-col gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleGenerateCommentary("match_summary")}
                          disabled={generateCommentaryMutation.isPending}
                        >
                          {generateCommentaryMutation.isPending ? (
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          ) : (
                            <MessageSquare className="w-4 h-4 mr-2" />
                          )}
                          Match Summary
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleGenerateCommentary("tactical_analysis")}
                          disabled={generateCommentaryMutation.isPending}
                        >
                          {generateCommentaryMutation.isPending ? (
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          ) : (
                            <TrendingUp className="w-4 h-4 mr-2" />
                          )}
                          Tactical Analysis
                        </Button>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// Status Badge Component
function StatusBadge({ status, progress }: { status: string; progress: number }) {
  const config = {
    pending: { icon: <Clock className="w-4 h-4" />, text: "Pending", variant: "secondary" as const },
    uploading: { icon: <Loader2 className="w-4 h-4 animate-spin" />, text: "Uploading", variant: "secondary" as const },
    processing: { icon: <Loader2 className="w-4 h-4 animate-spin" />, text: `Processing ${progress}%`, variant: "default" as const },
    completed: { icon: <CheckCircle2 className="w-4 h-4" />, text: "Completed", variant: "default" as const },
    failed: { icon: <XCircle className="w-4 h-4" />, text: "Failed", variant: "destructive" as const },
  };

  const c = config[status as keyof typeof config] || config.pending;

  return (
    <Badge variant={c.variant} className="gap-1">
      {c.icon}
      {c.text}
    </Badge>
  );
}

// Processing Status Component with ETA and Termination
function ProcessingStatus({ analysis }: { analysis: any }) {
  const currentStageIndex = PROCESSING_STAGES.findIndex(s => s.id === analysis.currentStage);
  const utils = trpc.useUtils();
  
  // Fetch ETA
  const { data: etaData } = trpc.analysis.getEta.useQuery(
    { id: analysis.id },
    { 
      enabled: analysis.status === "processing" || analysis.status === "pending",
      refetchInterval: 5000 
    }
  );
  
  // Terminate mutation
  const terminateMutation = trpc.analysis.terminate.useMutation({
    onSuccess: () => {
      toast.success("Analysis terminated");
      utils.analysis.get.invalidate({ id: analysis.id });
    },
    onError: (error) => {
      toast.error(error.message || "Failed to terminate");
    },
  });
  
  const handleTerminate = () => {
    if (confirm("Are you sure you want to terminate this analysis? This cannot be undone.")) {
      terminateMutation.mutate({ id: analysis.id });
    }
  };
  
  // Format time remaining
  const formatTime = (ms: number) => {
    if (ms < 1000) return "< 1s";
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
  };
  
  return (
    <Card className="mb-6">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin text-primary" />
              Processing Video
            </CardTitle>
            <CardDescription>
              {analysis.currentStage ? `Current stage: ${PROCESSING_STAGES.find(s => s.id === analysis.currentStage)?.name || analysis.currentStage}` : "Initializing..."}
            </CardDescription>
          </div>
          <Button 
            variant="destructive" 
            size="sm"
            onClick={handleTerminate}
            disabled={terminateMutation.isPending}
          >
            {terminateMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <>
                <XCircle className="w-4 h-4 mr-1" />
                Terminate
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Progress bar with percentage */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Progress</span>
              <span className="font-medium">{analysis.progress}%</span>
            </div>
            <Progress value={analysis.progress} className="h-2" />
          </div>
          
          {/* ETA Display */}
          {etaData && (
            <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Estimated time remaining</span>
              </div>
              <span className="font-semibold text-primary">
                {formatTime(etaData.remainingMs)}
              </span>
            </div>
          )}
          
          {/* Stage indicators */}
          <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
            {PROCESSING_STAGES.map((stage, i) => (
              <div
                key={stage.id}
                className={`text-center p-2 rounded-lg text-xs transition-all ${
                  i < currentStageIndex
                    ? "bg-primary/20 text-primary"
                    : i === currentStageIndex
                    ? "bg-primary text-primary-foreground animate-pulse"
                    : "bg-secondary text-muted-foreground"
                }`}
              >
                {stage.name.split(" ")[0]}
              </div>
            ))}
          </div>
          
          {/* Elapsed time */}
          {etaData && (
            <div className="text-xs text-muted-foreground text-center">
              Elapsed: {formatTime(etaData.elapsedMs)} • Stage {etaData.stageIndex + 1} of {etaData.totalStages}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// Pitch Radar Component
function PitchRadar({ data }: { data: any }) {
  if (!data) {
    return (
      <div className="pitch-container flex items-center justify-center">
        <p className="text-muted-foreground">No tracking data available</p>
      </div>
    );
  }

  return (
    <div className="pitch-container">
      {/* Pitch SVG */}
      <svg className="pitch-lines" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet">
        {/* Outer boundary */}
        <rect x="0" y="0" width="105" height="68" />
        {/* Center line */}
        <line x1="52.5" y1="0" x2="52.5" y2="68" />
        {/* Center circle */}
        <circle cx="52.5" cy="34" r="9.15" />
        {/* Center spot */}
        <circle cx="52.5" cy="34" r="0.3" fill="currentColor" />
        {/* Left penalty area */}
        <rect x="0" y="13.84" width="16.5" height="40.32" />
        {/* Left goal area */}
        <rect x="0" y="24.84" width="5.5" height="18.32" />
        {/* Left penalty spot */}
        <circle cx="11" cy="34" r="0.3" fill="currentColor" />
        {/* Left penalty arc */}
        <path d="M 16.5 25 A 9.15 9.15 0 0 1 16.5 43" />
        {/* Right penalty area */}
        <rect x="88.5" y="13.84" width="16.5" height="40.32" />
        {/* Right goal area */}
        <rect x="99.5" y="24.84" width="5.5" height="18.32" />
        {/* Right penalty spot */}
        <circle cx="94" cy="34" r="0.3" fill="currentColor" />
        {/* Right penalty arc */}
        <path d="M 88.5 25 A 9.15 9.15 0 0 0 88.5 43" />
        {/* Corner arcs */}
        <path d="M 0 1 A 1 1 0 0 0 1 0" />
        <path d="M 104 0 A 1 1 0 0 0 105 1" />
        <path d="M 105 67 A 1 1 0 0 0 104 68" />
        <path d="M 1 68 A 1 1 0 0 0 0 67" />
      </svg>

      {/* Team 1 Players */}
      {data.team1Players.map((player: any) => (
        <div
          key={player.id}
          className="player-marker team-1"
          style={{
            left: `${(player.x / PITCH_WIDTH) * 100}%`,
            top: `${(player.y / PITCH_HEIGHT) * 100}%`,
          }}
          title={`Player ${player.id} - Speed: ${player.speed?.toFixed(1)} km/h`}
        >
          <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white">
            {player.id}
          </span>
        </div>
      ))}

      {/* Team 2 Players */}
      {data.team2Players.map((player: any) => (
        <div
          key={player.id}
          className="player-marker team-2"
          style={{
            left: `${(player.x / PITCH_WIDTH) * 100}%`,
            top: `${(player.y / PITCH_HEIGHT) * 100}%`,
          }}
          title={`Player ${player.id} - Speed: ${player.speed?.toFixed(1)} km/h`}
        >
          <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white">
            {player.id - 11}
          </span>
        </div>
      ))}

      {/* Ball */}
      <div
        className="ball-marker"
        style={{
          left: `${(data.ball.x / PITCH_WIDTH) * 100}%`,
          top: `${(data.ball.y / PITCH_HEIGHT) * 100}%`,
        }}
      />
    </div>
  );
}

// Heatmap View Component
function HeatmapView() {
  return (
    <div className="pitch-container">
      <svg className="pitch-lines" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet">
        <rect x="0" y="0" width="105" height="68" />
        <line x1="52.5" y1="0" x2="52.5" y2="68" />
        <circle cx="52.5" cy="34" r="9.15" />
      </svg>
      
      {/* Heatmap overlay */}
      <div className="absolute inset-0 opacity-60">
        <svg viewBox="0 0 105 68" className="w-full h-full">
          <defs>
            <radialGradient id="heatGradient1">
              <stop offset="0%" stopColor="oklch(0.55 0.2 25)" stopOpacity="0.8" />
              <stop offset="100%" stopColor="oklch(0.55 0.2 25)" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="heatGradient2">
              <stop offset="0%" stopColor="oklch(0.65 0.2 45)" stopOpacity="0.6" />
              <stop offset="100%" stopColor="oklch(0.65 0.2 45)" stopOpacity="0" />
            </radialGradient>
          </defs>
          {/* Sample heat zones */}
          <ellipse cx="30" cy="34" rx="15" ry="20" fill="url(#heatGradient1)" />
          <ellipse cx="52.5" cy="34" rx="20" ry="15" fill="url(#heatGradient2)" />
          <ellipse cx="75" cy="34" rx="15" ry="20" fill="url(#heatGradient1)" />
        </svg>
      </div>

      {/* Legend */}
      <div className="absolute bottom-2 right-2 bg-card/80 backdrop-blur-sm rounded-lg p-2 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-16 h-2 rounded heatmap-gradient" />
          <span>Low → High</span>
        </div>
      </div>
    </div>
  );
}

// Pass Network View Component
function PassNetworkView() {
  // Demo pass network data
  const nodes = [
    { id: 1, x: 15, y: 34, passes: 45 },
    { id: 2, x: 30, y: 15, passes: 38 },
    { id: 3, x: 30, y: 53, passes: 42 },
    { id: 4, x: 45, y: 25, passes: 52 },
    { id: 5, x: 45, y: 43, passes: 48 },
    { id: 6, x: 60, y: 34, passes: 35 },
  ];

  const edges = [
    { from: 1, to: 2, count: 12 },
    { from: 1, to: 3, count: 15 },
    { from: 2, to: 4, count: 18 },
    { from: 3, to: 5, count: 14 },
    { from: 4, to: 6, count: 10 },
    { from: 5, to: 6, count: 8 },
    { from: 4, to: 5, count: 6 },
  ];

  return (
    <div className="pitch-container">
      <svg className="pitch-lines" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet">
        <rect x="0" y="0" width="105" height="68" />
        <line x1="52.5" y1="0" x2="52.5" y2="68" />
        <circle cx="52.5" cy="34" r="9.15" />
      </svg>

      {/* Pass lines */}
      <svg className="absolute inset-0" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="oklch(0.65 0.2 145)" />
          </marker>
        </defs>
        {edges.map((edge, i) => {
          const fromNode = nodes.find(n => n.id === edge.from)!;
          const toNode = nodes.find(n => n.id === edge.to)!;
          return (
            <line
              key={i}
              x1={fromNode.x}
              y1={fromNode.y}
              x2={toNode.x}
              y2={toNode.y}
              stroke="oklch(0.65 0.2 145)"
              strokeWidth={Math.max(1, edge.count / 5)}
              strokeOpacity={0.6}
              markerEnd="url(#arrowhead)"
            />
          );
        })}
      </svg>

      {/* Player nodes */}
      {nodes.map(node => (
        <div
          key={node.id}
          className="absolute w-8 h-8 rounded-full bg-primary border-2 border-white shadow-lg transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center"
          style={{
            left: `${(node.x / PITCH_WIDTH) * 100}%`,
            top: `${(node.y / PITCH_HEIGHT) * 100}%`,
          }}
        >
          <span className="text-xs font-bold text-primary-foreground">{node.id}</span>
        </div>
      ))}
    </div>
  );
}

// Event Timeline Component
function EventTimeline({ events }: { events: any[] }) {
  return (
    <div className="space-y-4">
      <div className="relative h-12 bg-secondary rounded-lg overflow-hidden">
        {events.map((event, i) => {
          const position = (event.timestamp / 90) * 100; // Assuming 90 min match
          const eventConfig = EVENT_TYPES[event.type as keyof typeof EVENT_TYPES];
          return (
            <div
              key={i}
              className="event-marker"
              style={{
                left: `${Math.min(position, 98)}%`,
                backgroundColor: eventConfig?.color || "#666",
              }}
              title={`${event.type} at ${event.timestamp}s`}
            />
          );
        })}
      </div>

      <ScrollArea className="h-48">
        <div className="space-y-2">
          {events.slice(0, 10).map((event, i) => {
            const eventConfig = EVENT_TYPES[event.type as keyof typeof EVENT_TYPES];
            return (
              <div
                key={i}
                className="flex items-center gap-3 p-2 rounded-lg bg-secondary/50 hover:bg-secondary transition-colors"
              >
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: eventConfig?.color || "#666" }}
                />
                <div className="flex-1">
                  <span className="font-medium capitalize">{event.type}</span>
                  <span className="text-muted-foreground ml-2">
                    Team {event.teamId} • {event.timestamp}s
                  </span>
                </div>
                <Badge variant={event.success ? "default" : "secondary"}>
                  {event.success ? "Success" : "Failed"}
                </Badge>
              </div>
            );
          })}
        </div>
      </ScrollArea>
    </div>
  );
}

// Mode-Specific Tabs Component
function ModeSpecificTabs({ 
  mode, 
  activeTab, 
  setActiveTab, 
  trackingData, 
  events 
}: { 
  mode: PipelineMode; 
  activeTab: string; 
  setActiveTab: (tab: string) => void;
  trackingData: any;
  events: any[];
}) {
  // Define which tabs are available for each mode
  const modeTabs: Record<PipelineMode, string[]> = {
    all: ["radar", "voronoi", "heatmap", "passes", "events"],
    radar: ["radar", "voronoi"],
    team: ["radar", "heatmap"],
    track: ["radar", "events"],
    players: ["radar"],
    ball: ["radar", "events"],
    pitch: ["radar", "heatmap"],
  };

  const availableTabs = modeTabs[mode] || modeTabs.all;
  const tabConfig = {
    radar: { icon: <Radar className="w-4 h-4" />, label: "Radar" },
    voronoi: { icon: <Users className="w-4 h-4" />, label: "Voronoi" },
    heatmap: { icon: <Map className="w-4 h-4" />, label: "Heatmap" },
    passes: { icon: <Target className="w-4 h-4" />, label: "Passes" },
    events: { icon: <Zap className="w-4 h-4" />, label: "Events" },
  };

  // Reset to first available tab if current is not available
  const effectiveTab = availableTabs.includes(activeTab) ? activeTab : availableTabs[0];

  return (
    <Tabs value={effectiveTab} onValueChange={setActiveTab}>
      <TabsList className={`grid w-full grid-cols-${Math.min(availableTabs.length, 5)}`} style={{ gridTemplateColumns: `repeat(${availableTabs.length}, 1fr)` }}>
        {availableTabs.map(tab => {
          const config = tabConfig[tab as keyof typeof tabConfig];
          return (
            <TabsTrigger key={tab} value={tab} className="gap-2">
              {config.icon}
              <span className="hidden sm:inline">{config.label}</span>
            </TabsTrigger>
          );
        })}
      </TabsList>

      <TabsContent value="radar" className="mt-4">
        <PitchRadar data={trackingData} />
      </TabsContent>

      <TabsContent value="voronoi" className="mt-4">
        <VoronoiView data={trackingData} />
      </TabsContent>

      <TabsContent value="heatmap" className="mt-4">
        <HeatmapView />
      </TabsContent>

      <TabsContent value="passes" className="mt-4">
        <PassNetworkView />
      </TabsContent>

      <TabsContent value="events" className="mt-4">
        <EventTimeline events={events} />
      </TabsContent>
    </Tabs>
  );
}

// Voronoi Diagram View Component
function VoronoiView({ data }: { data: any }) {
  if (!data) {
    return (
      <div className="pitch-container flex items-center justify-center">
        <p className="text-muted-foreground">No tracking data available</p>
      </div>
    );
  }

  // Compute Voronoi cells for all players
  const allPlayers = [...data.team1Players, ...data.team2Players];
  
  // Generate Voronoi polygons using a simple algorithm
  const voronoiCells = useMemo(() => {
    const cells: { playerId: number; teamId: number; path: string }[] = [];
    
    // Grid-based Voronoi approximation
    const gridSize = 1; // 1 meter resolution
    const grid: { x: number; y: number; playerId: number; teamId: number }[][] = [];
    
    for (let x = 0; x <= PITCH_WIDTH; x += gridSize) {
      const row: { x: number; y: number; playerId: number; teamId: number }[] = [];
      for (let y = 0; y <= PITCH_HEIGHT; y += gridSize) {
        let minDist = Infinity;
        let closestPlayer = allPlayers[0];
        
        for (const player of allPlayers) {
          const dist = Math.sqrt((x - player.x) ** 2 + (y - player.y) ** 2);
          if (dist < minDist) {
            minDist = dist;
            closestPlayer = player;
          }
        }
        
        row.push({ x, y, playerId: closestPlayer.id, teamId: closestPlayer.teamId });
      }
      grid.push(row);
    }
    
    // Create simplified polygon paths for each player
    for (const player of allPlayers) {
      const points: [number, number][] = [];
      
      // Find boundary points for this player's region
      for (let x = 0; x < grid.length; x++) {
        for (let y = 0; y < grid[x].length; y++) {
          if (grid[x][y].playerId === player.id) {
            // Check if this is a boundary point
            const isBoundary = 
              x === 0 || x === grid.length - 1 ||
              y === 0 || y === grid[x].length - 1 ||
              grid[x - 1]?.[y]?.playerId !== player.id ||
              grid[x + 1]?.[y]?.playerId !== player.id ||
              grid[x]?.[y - 1]?.playerId !== player.id ||
              grid[x]?.[y + 1]?.playerId !== player.id;
            
            if (isBoundary) {
              points.push([grid[x][y].x, grid[x][y].y]);
            }
          }
        }
      }
      
      if (points.length > 2) {
        // Sort points by angle from centroid
        const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
        const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;
        points.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
        
        const path = `M ${points.map(p => `${p[0]} ${p[1]}`).join(' L ')} Z`;
        cells.push({ playerId: player.id, teamId: player.teamId, path });
      }
    }
    
    return cells;
  }, [allPlayers]);

  return (
    <div className="pitch-container">
      {/* Pitch SVG with Voronoi overlay */}
      <svg className="absolute inset-0" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet">
        {/* Voronoi cells */}
        {voronoiCells.map((cell, i) => (
          <path
            key={i}
            d={cell.path}
            fill={cell.teamId === 1 ? "var(--color-team-1)" : "var(--color-team-2)"}
            fillOpacity={0.2}
            stroke={cell.teamId === 1 ? "var(--color-team-1)" : "var(--color-team-2)"}
            strokeWidth={0.3}
            strokeOpacity={0.5}
          />
        ))}
        
        {/* Pitch lines */}
        <rect x="0" y="0" width="105" height="68" fill="none" stroke="var(--color-pitch-lines)" strokeWidth="0.5" />
        <line x1="52.5" y1="0" x2="52.5" y2="68" stroke="var(--color-pitch-lines)" strokeWidth="0.5" />
        <circle cx="52.5" cy="34" r="9.15" fill="none" stroke="var(--color-pitch-lines)" strokeWidth="0.5" />
      </svg>

      {/* Team 1 Players */}
      {data.team1Players.map((player: any) => (
        <div
          key={player.id}
          className="player-marker team-1"
          style={{
            left: `${(player.x / PITCH_WIDTH) * 100}%`,
            top: `${(player.y / PITCH_HEIGHT) * 100}%`,
          }}
        >
          <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white">
            {player.id}
          </span>
        </div>
      ))}

      {/* Team 2 Players */}
      {data.team2Players.map((player: any) => (
        <div
          key={player.id}
          className="player-marker team-2"
          style={{
            left: `${(player.x / PITCH_WIDTH) * 100}%`,
            top: `${(player.y / PITCH_HEIGHT) * 100}%`,
          }}
        >
          <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white">
            {player.id - 11}
          </span>
        </div>
      ))}

      {/* Ball */}
      <div
        className="ball-marker"
        style={{
          left: `${(data.ball.x / PITCH_WIDTH) * 100}%`,
          top: `${(data.ball.y / PITCH_HEIGHT) * 100}%`,
        }}
      />

      {/* Legend */}
      <div className="absolute bottom-2 right-2 bg-card/80 backdrop-blur-sm rounded-lg p-2 text-xs">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: "var(--color-team-1)", opacity: 0.5 }} />
            <span>Team 1 Zone</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: "var(--color-team-2)", opacity: 0.5 }} />
            <span>Team 2 Zone</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Stat Row Component
function StatRow({ label, team1, team2, suffix = "" }: { label: string; team1: number; team2: number; suffix?: string }) {
  const total = team1 + team2;
  const team1Pct = total > 0 ? (team1 / total) * 100 : 50;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="font-mono" style={{ color: "var(--color-team-1)" }}>
          {team1.toFixed(suffix === "%" ? 1 : 0)}{suffix}
        </span>
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono" style={{ color: "var(--color-team-2)" }}>
          {team2.toFixed(suffix === "%" ? 1 : 0)}{suffix}
        </span>
      </div>
      <div className="h-2 bg-secondary rounded-full overflow-hidden flex">
        <div
          className="h-full transition-all"
          style={{ width: `${team1Pct}%`, backgroundColor: "var(--color-team-1)" }}
        />
        <div
          className="h-full transition-all"
          style={{ width: `${100 - team1Pct}%`, backgroundColor: "var(--color-team-2)" }}
        />
      </div>
    </div>
  );
}
