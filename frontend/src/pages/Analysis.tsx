import { useState, useEffect, useRef, useMemo, useCallback } from "react";
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
  Video,
  Eye,
  Sparkles,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { PIPELINE_MODES, PROCESSING_STAGES, EVENT_TYPES, PipelineMode } from "@/shared/types";
import { Streamdown } from "streamdown";
import { useWebSocket, WSMessage } from "@/hooks/useWebSocket";

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

  // Real-time progress state from WebSocket
  const [realtimeProgress, setRealtimeProgress] = useState<{
    progress?: number;
    currentStage?: string;
    eta?: number;
  } | null>(null);

  // WebSocket callbacks for real-time updates
  const handleWSProgress = useCallback((data: WSMessage["data"]) => {
    if (data) {
      setRealtimeProgress({
        progress: data.progress,
        currentStage: data.currentStage,
        eta: data.eta,
      });
    }
  }, []);

  const handleWSComplete = useCallback(() => {
    setRealtimeProgress(null);
    utils.analysis.get.invalidate({ id: analysisId });
  }, [analysisId]);

  const handleWSError = useCallback((error: string) => {
    console.error("WebSocket error:", error);
    setRealtimeProgress(null);
    utils.analysis.get.invalidate({ id: analysisId });
  }, [analysisId]);

  // Connect to WebSocket for real-time updates
  const { isConnected: wsConnected } = useWebSocket({
    analysisId,
    onProgress: handleWSProgress,
    onComplete: handleWSComplete,
    onError: handleWSError,
    enabled: isAuthenticated && analysisId > 0,
  });

  const utils = trpc.useUtils();

  // Fetch analysis data
  const { data: analysis, isLoading: analysisLoading, refetch } = trpc.analysis.get.useQuery(
    { id: analysisId },
    { 
      enabled: isAuthenticated && analysisId > 0, 
      refetchInterval: wsConnected ? 10000 : 2000
    }
  );

  // Merge real-time progress with analysis data
  const analysisWithRealtime = useMemo(() => {
    if (!analysis) return null;
    if (!realtimeProgress) return analysis;
    return {
      ...analysis,
      progress: realtimeProgress.progress ?? analysis.progress,
      currentStage: realtimeProgress.currentStage ?? analysis.currentStage,
      eta: realtimeProgress.eta,
    };
  }, [analysis, realtimeProgress]);

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

  const mode = analysis?.mode as PipelineMode;
  const modeConfig = mode ? PIPELINE_MODES[mode] : null;

  // Generate demo tracking data for visualization
  const demoTrackingData = useMemo(() => {
    if (!analysis || analysis.status !== "completed") return null;
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

  // --- Loading / Auth / Not Found States ---
  if (authLoading || analysisLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full border-2 border-primary/20 border-t-primary animate-spin" />
            <Activity className="w-6 h-6 text-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
          </div>
          <p className="text-muted-foreground text-sm animate-pulse">Loading analysis...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="glass-card p-8 max-w-md w-full text-center space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto">
            <Activity className="w-8 h-8 text-primary" />
          </div>
          <h2 className="text-xl font-bold">Sign In Required</h2>
          <p className="text-muted-foreground text-sm">Please sign in to view this analysis</p>
          <a href={getLoginUrl()}>
            <Button className="w-full mt-2">Sign In</Button>
          </a>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="glass-card p-8 max-w-md w-full text-center space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-destructive/10 border border-destructive/20 flex items-center justify-center mx-auto">
            <XCircle className="w-8 h-8 text-destructive" />
          </div>
          <h2 className="text-xl font-bold">Analysis Not Found</h2>
          <p className="text-muted-foreground text-sm">This analysis doesn't exist or you don't have access</p>
          <Link href="/dashboard">
            <Button className="w-full mt-2">Back to Dashboard</Button>
          </Link>
        </div>
      </div>
    );
  }

  // --- Main Analysis Page ---
  return (
    <div className="min-h-screen bg-background">
      {/* Subtle gradient background overlay */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-[128px]" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent/5 rounded-full blur-[128px]" />
      </div>

      {/* Header */}
      <header className="relative border-b border-border/50 bg-card/30 backdrop-blur-xl sticky top-0 z-50">
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
        <div className="container flex items-center justify-between h-16">
          <div className="flex items-center gap-4">
            <Link href="/dashboard">
              <Button variant="ghost" size="icon" className="rounded-xl hover:bg-primary/10 transition-colors">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/20 flex items-center justify-center">
                <Activity className="w-5 h-5 text-primary" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="font-bold text-lg">Analysis #{analysis.id}</h1>
                  <Badge variant="outline" className="border-primary/30 text-primary bg-primary/5 text-xs">
                    {modeConfig?.name || analysis.mode}
                  </Badge>
                </div>
                {analysis.createdAt && (
                  <p className="text-xs text-muted-foreground">
                    {formatDistanceToNow(new Date(analysis.createdAt), { addSuffix: true })}
                  </p>
                )}
              </div>
            </div>
          </div>
          <StatusBadge status={analysis.status} progress={analysis.progress} />
        </div>
      </header>

      <main className="container py-8 relative z-10">
        {/* Processing Status */}
        {(analysis.status === "pending" || analysis.status === "processing" || analysis.status === "uploading") && (
          <ProcessingStatus analysis={analysisWithRealtime || analysis} wsConnected={wsConnected} />
        )}

        {/* Failed Status */}
        {analysis.status === "failed" && (
          <div className="glass-card border-destructive/30 mb-8 p-6">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-destructive/10 border border-destructive/20 flex items-center justify-center shrink-0">
                <XCircle className="w-7 h-7 text-destructive" />
              </div>
              <div>
                <h3 className="font-bold text-lg">Analysis Failed</h3>
                <p className="text-muted-foreground text-sm mt-1">{analysis.errorMessage || "An error occurred during processing"}</p>
              </div>
            </div>
          </div>
        )}

        {/* Completed Analysis - Bento Grid Layout */}
        {analysis.status === "completed" && (
          <div className="space-y-6">
            {/* Top Row: Video + Stats side by side */}
            <div className="grid lg:grid-cols-5 gap-6">
              {/* Video Player - takes 3 cols */}
              <div className="lg:col-span-3">
                {analysis.annotatedVideoUrl ? (
                  <div className="glass-card overflow-hidden group">
                    <div className="p-4 pb-3 flex items-center gap-2 border-b border-border/30">
                      <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                        <Video className="w-4 h-4 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-sm">Annotated Video</h3>
                        <p className="text-xs text-muted-foreground">AI-processed output</p>
                      </div>
                    </div>
                    <div className="p-4 pt-3">
                      <div className="video-player-container rounded-xl overflow-hidden ring-1 ring-white/5">
                        <video
                          ref={videoRef}
                          src={analysis.annotatedVideoUrl}
                          controls
                          className="w-full h-full object-contain"
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="glass-card p-8 flex flex-col items-center justify-center min-h-[300px]">
                    <div className="w-16 h-16 rounded-2xl bg-muted/50 flex items-center justify-center mb-4">
                      <Video className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <p className="text-muted-foreground text-sm">No annotated video available for this mode</p>
                  </div>
                )}
              </div>

              {/* Stats Panel - takes 2 cols */}
              <div className="lg:col-span-2 space-y-6">
                {/* Match Statistics */}
                <div className="glass-card overflow-hidden">
                  <div className="p-4 pb-3 flex items-center justify-between border-b border-border/30">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                        <BarChart3 className="w-4 h-4 text-primary" />
                      </div>
                      <h3 className="font-semibold text-sm">Match Statistics</h3>
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <div className="flex items-center gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: "var(--color-team-1)" }} />
                        <span className="text-muted-foreground">Team 1</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: "var(--color-team-2)" }} />
                        <span className="text-muted-foreground">Team 2</span>
                      </div>
                    </div>
                  </div>
                  <div className="p-4 space-y-4">
                    <StatRow label="Possession" team1={demoStats.possessionTeam1} team2={demoStats.possessionTeam2} suffix="%" />
                    <StatRow label="Passes" team1={demoStats.passesTeam1} team2={demoStats.passesTeam2} />
                    <StatRow label="Pass Accuracy" team1={demoStats.passAccuracyTeam1} team2={demoStats.passAccuracyTeam2} suffix="%" />
                    <StatRow label="Shots" team1={demoStats.shotsTeam1} team2={demoStats.shotsTeam2} />
                    <StatRow label="Distance (km)" team1={demoStats.distanceCoveredTeam1} team2={demoStats.distanceCoveredTeam2} />
                    <StatRow label="Avg Speed (km/h)" team1={demoStats.avgSpeedTeam1} team2={demoStats.avgSpeedTeam2} />
                  </div>
                </div>

                {/* Quick Stats Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <QuickStat 
                    label="Total Events" 
                    value={demoEvents.length.toString()} 
                    icon={<Zap className="w-4 h-4" />}
                    color="primary"
                  />
                  <QuickStat 
                    label="Pass Rate" 
                    value={`${((demoStats.passAccuracyTeam1 + demoStats.passAccuracyTeam2) / 2).toFixed(0)}%`}
                    icon={<Target className="w-4 h-4" />}
                    color="accent"
                  />
                  <QuickStat 
                    label="Total Shots" 
                    value={(demoStats.shotsTeam1 + demoStats.shotsTeam2).toString()}
                    icon={<Activity className="w-4 h-4" />}
                    color="team1"
                  />
                  <QuickStat 
                    label="Avg Speed" 
                    value={`${((demoStats.avgSpeedTeam1 + demoStats.avgSpeedTeam2) / 2).toFixed(1)}`}
                    icon={<TrendingUp className="w-4 h-4" />}
                    color="team2"
                  />
                </div>
              </div>
            </div>

            {/* Bottom Row: Visualizations + Commentary */}
            <div className="grid lg:grid-cols-5 gap-6">
              {/* Visualizations - takes 3 cols */}
              <div className="lg:col-span-3">
                <div className="glass-card overflow-hidden">
                  <div className="p-4 pb-3 flex items-center gap-2 border-b border-border/30">
                    <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
                      <Eye className="w-4 h-4 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-sm">Visualizations</h3>
                      <p className="text-xs text-muted-foreground">Interactive analysis views &middot; {modeConfig?.name || analysis.mode} mode</p>
                    </div>
                  </div>
                  <div className="p-4">
                    <ModeSpecificTabs 
                      mode={mode} 
                      activeTab={activeTab} 
                      setActiveTab={setActiveTab}
                      trackingData={demoTrackingData}
                      events={demoEvents}
                    />
                  </div>
                </div>
              </div>

              {/* AI Commentary - takes 2 cols */}
              <div className="lg:col-span-2">
                <div className="glass-card overflow-hidden h-full">
                  <div className="p-4 pb-3 flex items-center gap-2 border-b border-border/30">
                    <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center">
                      <Sparkles className="w-4 h-4 text-accent" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-sm">AI Commentary</h3>
                      <p className="text-xs text-muted-foreground">Tactical analysis grounded in tracking data</p>
                    </div>
                  </div>
                  <div className="p-4">
                    {commentaryList && commentaryList.length > 0 ? (
                      <ScrollArea className="h-[400px]">
                        <div className="space-y-3">
                          {commentaryList.map((c) => (
                            <div key={c.id} className="p-4 rounded-xl bg-secondary/30 border border-border/20 hover:border-border/40 transition-colors">
                              <Badge variant="outline" className="mb-2 text-xs border-accent/30 text-accent">{c.type}</Badge>
                              <div className="text-sm prose prose-sm dark:prose-invert max-w-none leading-relaxed">
                                <Streamdown>{c.content}</Streamdown>
                              </div>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    ) : (
                      <div className="space-y-4">
                        <div className="text-center py-6">
                          <div className="w-12 h-12 rounded-2xl bg-accent/5 border border-accent/10 flex items-center justify-center mx-auto mb-3">
                            <MessageSquare className="w-6 h-6 text-accent/60" />
                          </div>
                          <p className="text-sm text-muted-foreground">
                            Generate AI-powered tactical commentary based on the analysis data.
                          </p>
                        </div>
                        <div className="grid gap-2">
                          <Button
                            variant="outline"
                            className="w-full justify-start gap-3 h-12 rounded-xl border-border/30 hover:border-primary/30 hover:bg-primary/5 transition-all"
                            onClick={() => handleGenerateCommentary("match_summary")}
                            disabled={generateCommentaryMutation.isPending}
                          >
                            {generateCommentaryMutation.isPending ? (
                              <Loader2 className="w-4 h-4 animate-spin text-primary" />
                            ) : (
                              <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                                <MessageSquare className="w-4 h-4 text-primary" />
                              </div>
                            )}
                            <div className="text-left">
                              <div className="text-sm font-medium">Match Summary</div>
                              <div className="text-xs text-muted-foreground">Overview of key moments</div>
                            </div>
                          </Button>
                          <Button
                            variant="outline"
                            className="w-full justify-start gap-3 h-12 rounded-xl border-border/30 hover:border-accent/30 hover:bg-accent/5 transition-all"
                            onClick={() => handleGenerateCommentary("tactical_analysis")}
                            disabled={generateCommentaryMutation.isPending}
                          >
                            {generateCommentaryMutation.isPending ? (
                              <Loader2 className="w-4 h-4 animate-spin text-accent" />
                            ) : (
                              <div className="w-8 h-8 rounded-lg bg-accent/10 flex items-center justify-center shrink-0">
                                <TrendingUp className="w-4 h-4 text-accent" />
                              </div>
                            )}
                            <div className="text-left">
                              <div className="text-sm font-medium">Tactical Analysis</div>
                              <div className="text-xs text-muted-foreground">In-depth strategic breakdown</div>
                            </div>
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// ==================== Sub-Components ====================

// Quick Stat Card
function QuickStat({ label, value, icon, color }: { label: string; value: string; icon: React.ReactNode; color: string }) {
  const colorMap: Record<string, string> = {
    primary: "bg-primary/10 text-primary border-primary/20",
    accent: "bg-accent/10 text-accent border-accent/20",
    team1: "bg-[var(--color-team-1)]/10 text-[var(--color-team-1)] border-[var(--color-team-1)]/20",
    team2: "bg-[var(--color-team-2)]/10 text-[var(--color-team-2)] border-[var(--color-team-2)]/20",
  };
  const iconColorMap: Record<string, string> = {
    primary: "bg-primary/10 text-primary",
    accent: "bg-accent/10 text-accent",
    team1: "text-[var(--color-team-1)]",
    team2: "text-[var(--color-team-2)]",
  };

  return (
    <div className="glass-card p-4 group hover:border-border/60 transition-all">
      <div className="flex items-center gap-2 mb-2">
        <span className={iconColorMap[color] || "text-primary"}>{icon}</span>
        <span className="text-xs text-muted-foreground uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-2xl font-bold font-mono">{value}</div>
    </div>
  );
}

// Status Badge Component
function StatusBadge({ status, progress }: { status: string; progress: number }) {
  const config: Record<string, { icon: React.ReactNode; text: string; className: string }> = {
    pending: { 
      icon: <Clock className="w-3.5 h-3.5" />, 
      text: "Pending", 
      className: "bg-muted/50 text-muted-foreground border-border/30" 
    },
    uploading: { 
      icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />, 
      text: "Uploading", 
      className: "bg-primary/10 text-primary border-primary/20" 
    },
    processing: { 
      icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />, 
      text: `Processing ${progress}%`, 
      className: "bg-primary/10 text-primary border-primary/20" 
    },
    completed: { 
      icon: <CheckCircle2 className="w-3.5 h-3.5" />, 
      text: "Completed", 
      className: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" 
    },
    failed: { 
      icon: <XCircle className="w-3.5 h-3.5" />, 
      text: "Failed", 
      className: "bg-destructive/10 text-destructive border-destructive/20" 
    },
  };

  const c = config[status] || config.pending;

  return (
    <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border ${c.className}`}>
      {c.icon}
      {c.text}
    </div>
  );
}

// Processing Status Component
function ProcessingStatus({ analysis, wsConnected = false }: { analysis: any; wsConnected?: boolean }) {
  const currentStageIndex = PROCESSING_STAGES.findIndex(s => s.id === analysis.currentStage);
  const utils = trpc.useUtils();
  
  const { data: etaData } = trpc.analysis.getEta.useQuery(
    { id: analysis.id },
    { 
      enabled: (analysis.status === "processing" || analysis.status === "pending") && !wsConnected,
      refetchInterval: wsConnected ? 30000 : 5000 
    }
  );
  
  const displayEta = analysis.eta !== undefined ? analysis.eta * 1000 : etaData?.remainingMs;
  
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
    <div className="glass-card mb-8 overflow-hidden">
      {/* Animated gradient top bar */}
      <div className="h-1 bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%] animate-[shimmer_2s_linear_infinite]" />
      
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
                <Loader2 className="w-7 h-7 animate-spin text-primary" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center">
                <span className="text-[10px] font-bold text-primary">{analysis.progress}%</span>
              </div>
            </div>
            <div>
              <h3 className="font-bold text-lg">Processing Video</h3>
              <p className="text-sm text-muted-foreground">
                {analysis.currentStage 
                  ? PROCESSING_STAGES.find(s => s.id === analysis.currentStage)?.name || analysis.currentStage
                  : "Initializing..."}
              </p>
            </div>
          </div>
          <Button 
            variant="outline" 
            size="sm"
            className="rounded-xl border-destructive/30 text-destructive hover:bg-destructive/10 hover:border-destructive/50"
            onClick={handleTerminate}
            disabled={terminateMutation.isPending}
          >
            {terminateMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <>
                <XCircle className="w-4 h-4 mr-1.5" />
                Terminate
              </>
            )}
          </Button>
        </div>

        {/* Progress bar */}
        <div className="space-y-3">
          <div className="h-2 bg-secondary/50 rounded-full overflow-hidden">
            <div 
              className="h-full rounded-full bg-gradient-to-r from-primary to-accent transition-all duration-500 ease-out relative"
              style={{ width: `${analysis.progress}%` }}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent bg-[length:200%_100%] animate-[shimmer_1.5s_linear_infinite]" />
            </div>
          </div>
          
          {/* ETA */}
          {(displayEta !== undefined || etaData) && (
            <div className="flex items-center justify-between px-1">
              <div className="flex items-center gap-2">
                <Clock className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">Estimated time remaining</span>
                {wsConnected && (
                  <span className="text-[10px] text-emerald-400 font-medium px-1.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">LIVE</span>
                )}
              </div>
              <span className="text-sm font-semibold font-mono text-primary">
                {formatTime(displayEta ?? etaData?.remainingMs ?? 0)}
              </span>
            </div>
          )}
        </div>

        {/* Stage indicators */}
        <div className="mt-6 grid grid-cols-4 sm:grid-cols-8 gap-2">
          {PROCESSING_STAGES.map((stage, i) => {
            const isComplete = i < currentStageIndex;
            const isCurrent = i === currentStageIndex;
            return (
              <div
                key={stage.id}
                className={`relative text-center p-2.5 rounded-xl text-xs font-medium transition-all duration-300 ${
                  isComplete
                    ? "bg-primary/15 text-primary border border-primary/20"
                    : isCurrent
                    ? "bg-primary/20 text-primary border border-primary/40 shadow-[0_0_12px_rgba(var(--color-primary),0.15)]"
                    : "bg-secondary/30 text-muted-foreground border border-transparent"
                }`}
              >
                {isComplete && (
                  <CheckCircle2 className="w-3 h-3 absolute top-1 right-1 text-primary" />
                )}
                {isCurrent && (
                  <div className="absolute inset-0 rounded-xl bg-primary/5 animate-pulse" />
                )}
                <span className="relative">{stage.name.split(" ")[0]}</span>
              </div>
            );
          })}
        </div>
        
        {/* Elapsed time */}
        {etaData && (
          <div className="text-xs text-muted-foreground text-center mt-4">
            Elapsed: {formatTime(etaData.elapsedMs)} &middot; Stage {Math.max(1, etaData.stageIndex + 1)} of {etaData.totalStages}
          </div>
        )}
      </div>
    </div>
  );
}

// Pitch Radar Component - Premium
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
      {/* Premium pitch SVG with gradient and glow */}
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet" style={{ zIndex: 2 }}>
        <defs>
          {/* Pitch grass gradient */}
          <linearGradient id="pitchGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(255,255,255,0.03)" />
            <stop offset="50%" stopColor="rgba(255,255,255,0)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.05)" />
          </linearGradient>
          {/* Line glow filter */}
          <filter id="lineGlow">
            <feGaussianBlur stdDeviation="0.3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Vignette */}
          <radialGradient id="vignette" cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor="transparent" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.35)" />
          </radialGradient>
          {/* Center spot glow */}
          <radialGradient id="spotGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.15)" />
            <stop offset="100%" stopColor="transparent" />
          </radialGradient>
        </defs>

        {/* Grass overlay */}
        <rect x="0" y="0" width="105" height="68" fill="url(#pitchGrad)" />

        {/* Pitch lines with glow */}
        <g filter="url(#lineGlow)" stroke="rgba(255,255,255,0.5)" strokeWidth="0.4" fill="none">
          {/* Outer boundary */}
          <rect x="0.5" y="0.5" width="104" height="67" rx="0.5" />
          {/* Center line */}
          <line x1="52.5" y1="0.5" x2="52.5" y2="67.5" />
          {/* Center circle */}
          <circle cx="52.5" cy="34" r="9.15" />
          {/* Center spot */}
          <circle cx="52.5" cy="34" r="0.6" fill="rgba(255,255,255,0.6)" stroke="none" />
          {/* Left penalty area */}
          <rect x="0.5" y="13.84" width="16.5" height="40.32" />
          {/* Left goal area */}
          <rect x="0.5" y="24.84" width="5.5" height="18.32" />
          {/* Left penalty spot */}
          <circle cx="11" cy="34" r="0.4" fill="rgba(255,255,255,0.5)" stroke="none" />
          {/* Left penalty arc */}
          <path d="M 16.5 25 A 9.15 9.15 0 0 1 16.5 43" />
          {/* Right penalty area */}
          <rect x="88" y="13.84" width="16.5" height="40.32" />
          {/* Right goal area */}
          <rect x="99" y="24.84" width="5.5" height="18.32" />
          {/* Right penalty spot */}
          <circle cx="94" cy="34" r="0.4" fill="rgba(255,255,255,0.5)" stroke="none" />
          {/* Right penalty arc */}
          <path d="M 88.5 25 A 9.15 9.15 0 0 0 88.5 43" />
          {/* Corner arcs */}
          <path d="M 0.5 1.5 A 1 1 0 0 0 1.5 0.5" />
          <path d="M 103.5 0.5 A 1 1 0 0 0 104.5 1.5" />
          <path d="M 104.5 66.5 A 1 1 0 0 0 103.5 67.5" />
          <path d="M 1.5 67.5 A 1 1 0 0 0 0.5 66.5" />
          {/* Goal lines (subtle) */}
          <rect x="-1.5" y="29" width="2" height="10" rx="0.3" strokeWidth="0.3" stroke="rgba(255,255,255,0.3)" />
          <rect x="104.5" y="29" width="2" height="10" rx="0.3" strokeWidth="0.3" stroke="rgba(255,255,255,0.3)" />
        </g>

        {/* Center circle glow */}
        <circle cx="52.5" cy="34" r="12" fill="url(#spotGlow)" />

        {/* Vignette overlay */}
        <rect x="0" y="0" width="105" height="68" fill="url(#vignette)" />
      </svg>

      {/* Team legend */}
      <div className="absolute top-3 right-3 flex items-center gap-3 bg-black/40 backdrop-blur-sm rounded-lg px-3 py-1.5 text-[11px] font-medium" style={{ zIndex: 20 }}>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full" style={{ background: 'linear-gradient(135deg, oklch(0.65 0.22 25), oklch(0.5 0.2 25))', boxShadow: '0 0 6px rgba(220,50,50,0.4)' }} />
          <span className="text-white/80">Team 1</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full" style={{ background: 'linear-gradient(135deg, oklch(0.6 0.2 200), oklch(0.45 0.18 200))', boxShadow: '0 0 6px rgba(50,130,200,0.4)' }} />
          <span className="text-white/80">Team 2</span>
        </div>
      </div>

      {/* Player markers */}
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
          <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white drop-shadow-sm" style={{ zIndex: 1 }}>
            {player.id}
          </span>
        </div>
      ))}

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
          <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white drop-shadow-sm" style={{ zIndex: 1 }}>
            {player.id - 11}
          </span>
        </div>
      ))}

      {/* Ball marker */}
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

// Heatmap View Component - Premium
function HeatmapView() {
  return (
    <div className="pitch-container">
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet" style={{ zIndex: 2 }}>
        <defs>
          <linearGradient id="heatPitchGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(255,255,255,0.03)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.05)" />
          </linearGradient>
          <filter id="heatLineGlow">
            <feGaussianBlur stdDeviation="0.3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <radialGradient id="heatVignette" cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor="transparent" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.3)" />
          </radialGradient>
          <radialGradient id="heatZone1"><stop offset="0%" stopColor="oklch(0.55 0.22 25)" stopOpacity="0.85" /><stop offset="100%" stopColor="oklch(0.55 0.2 25)" stopOpacity="0" /></radialGradient>
          <radialGradient id="heatZone2"><stop offset="0%" stopColor="oklch(0.65 0.22 45)" stopOpacity="0.7" /><stop offset="100%" stopColor="oklch(0.65 0.2 45)" stopOpacity="0" /></radialGradient>
          <radialGradient id="heatZone3"><stop offset="0%" stopColor="oklch(0.6 0.2 145)" stopOpacity="0.5" /><stop offset="100%" stopColor="oklch(0.6 0.2 145)" stopOpacity="0" /></radialGradient>
          <filter id="heatBlur"><feGaussianBlur stdDeviation="3" /></filter>
        </defs>

        <rect x="0" y="0" width="105" height="68" fill="url(#heatPitchGrad)" />

        {/* Pitch lines */}
        <g filter="url(#heatLineGlow)" stroke="rgba(255,255,255,0.35)" strokeWidth="0.35" fill="none">
          <rect x="0.5" y="0.5" width="104" height="67" rx="0.5" />
          <line x1="52.5" y1="0.5" x2="52.5" y2="67.5" />
          <circle cx="52.5" cy="34" r="9.15" />
          <circle cx="52.5" cy="34" r="0.5" fill="rgba(255,255,255,0.4)" stroke="none" />
          <rect x="0.5" y="13.84" width="16.5" height="40.32" />
          <rect x="0.5" y="24.84" width="5.5" height="18.32" />
          <rect x="88" y="13.84" width="16.5" height="40.32" />
          <rect x="99" y="24.84" width="5.5" height="18.32" />
        </g>

        {/* Heat zones */}
        <g filter="url(#heatBlur)" style={{ mixBlendMode: 'screen' }}>
          <ellipse cx="28" cy="30" rx="14" ry="18" fill="url(#heatZone1)" />
          <ellipse cx="52.5" cy="34" rx="18" ry="14" fill="url(#heatZone2)" />
          <ellipse cx="77" cy="38" rx="14" ry="18" fill="url(#heatZone1)" />
          <ellipse cx="40" cy="20" rx="10" ry="12" fill="url(#heatZone3)" />
          <ellipse cx="65" cy="48" rx="10" ry="12" fill="url(#heatZone3)" />
        </g>

        <rect x="0" y="0" width="105" height="68" fill="url(#heatVignette)" />
      </svg>

      <div className="absolute bottom-3 right-3 bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2 text-[11px] font-medium" style={{ zIndex: 20 }}>
        <div className="flex items-center gap-2">
          <div className="w-20 h-2 rounded-full heatmap-gradient" />
          <span className="text-white/70">Low &rarr; High</span>
        </div>
      </div>
    </div>
  );
}

// Pass Network View Component
function PassNetworkView() {
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
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet" style={{ zIndex: 2 }}>
        <defs>
          <linearGradient id="passNetGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(255,255,255,0.03)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.05)" />
          </linearGradient>
          <filter id="passLineGlow">
            <feGaussianBlur stdDeviation="0.3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <radialGradient id="passVignette" cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor="transparent" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.3)" />
          </radialGradient>
          <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="oklch(0.7 0.18 145)" fillOpacity="0.7" />
          </marker>
          <filter id="edgeGlow">
            <feGaussianBlur stdDeviation="0.5" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        <rect x="0" y="0" width="105" height="68" fill="url(#passNetGrad)" />

        {/* Pitch lines */}
        <g filter="url(#passLineGlow)" stroke="rgba(255,255,255,0.35)" strokeWidth="0.35" fill="none">
          <rect x="0.5" y="0.5" width="104" height="67" rx="0.5" />
          <line x1="52.5" y1="0.5" x2="52.5" y2="67.5" />
          <circle cx="52.5" cy="34" r="9.15" />
          <circle cx="52.5" cy="34" r="0.5" fill="rgba(255,255,255,0.4)" stroke="none" />
          <rect x="0.5" y="13.84" width="16.5" height="40.32" />
          <rect x="0.5" y="24.84" width="5.5" height="18.32" />
          <rect x="88" y="13.84" width="16.5" height="40.32" />
          <rect x="99" y="24.84" width="5.5" height="18.32" />
        </g>

        {/* Pass edges */}
        <g filter="url(#edgeGlow)">
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
                stroke="oklch(0.7 0.18 145)"
                strokeWidth={Math.max(0.5, edge.count / 6)}
                strokeOpacity={0.5}
                markerEnd="url(#arrowhead)"
              />
            );
          })}
        </g>

        <rect x="0" y="0" width="105" height="68" fill="url(#passVignette)" />
      </svg>

      {nodes.map(node => (
        <div
          key={node.id}
          className="absolute w-7 h-7 rounded-full transform -translate-x-1/2 -translate-y-1/2 flex items-center justify-center"
          style={{
            left: `${(node.x / PITCH_WIDTH) * 100}%`,
            top: `${(node.y / PITCH_HEIGHT) * 100}%`,
            background: 'linear-gradient(135deg, oklch(0.6 0.2 200), oklch(0.45 0.18 200))',
            boxShadow: '0 0 10px rgba(50,130,200,0.5), 0 0 20px rgba(50,130,200,0.2), inset 0 1px 1px rgba(255,255,255,0.2)',
            border: '1.5px solid rgba(255,255,255,0.25)',
            zIndex: 15,
          }}
        >
          <span className="text-[10px] font-bold text-white drop-shadow-sm">{node.id}</span>
        </div>
      ))}
    </div>
  );
}

// Event Timeline Component
function EventTimeline({ events }: { events: any[] }) {
  return (
    <div className="space-y-4">
      <div className="relative h-12 bg-secondary/30 rounded-xl overflow-hidden border border-border/20">
        {events.map((event, i) => {
          const position = (event.timestamp / 90) * 100;
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
                className="flex items-center gap-3 p-3 rounded-xl bg-secondary/20 border border-border/10 hover:border-border/30 hover:bg-secondary/30 transition-all"
              >
                <div
                  className="w-3 h-3 rounded-full shrink-0"
                  style={{ backgroundColor: eventConfig?.color || "#666" }}
                />
                <div className="flex-1 min-w-0">
                  <span className="font-medium capitalize text-sm">{event.type}</span>
                  <span className="text-muted-foreground text-xs ml-2">
                    Team {event.teamId} &middot; {event.timestamp}s
                  </span>
                </div>
                <Badge 
                  variant="outline" 
                  className={`text-xs shrink-0 ${event.success ? 'border-emerald-500/30 text-emerald-400 bg-emerald-500/5' : 'border-border/30 text-muted-foreground'}`}
                >
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

  const effectiveTab = availableTabs.includes(activeTab) ? activeTab : availableTabs[0];

  return (
    <Tabs value={effectiveTab} onValueChange={setActiveTab}>
      <TabsList 
        className="w-full bg-secondary/30 border border-border/20 rounded-xl p-1 h-auto"
        style={{ gridTemplateColumns: `repeat(${availableTabs.length}, 1fr)`, display: 'grid' }}
      >
        {availableTabs.map(tab => {
          const config = tabConfig[tab as keyof typeof tabConfig];
          return (
            <TabsTrigger 
              key={tab} 
              value={tab} 
              className="gap-2 rounded-lg data-[state=active]:bg-primary/15 data-[state=active]:text-primary data-[state=active]:shadow-none py-2.5"
            >
              {config.icon}
              <span className="hidden sm:inline text-xs font-medium">{config.label}</span>
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

  const allPlayers = [...data.team1Players, ...data.team2Players];
  
  const voronoiCells = useMemo(() => {
    const cells: { playerId: number; teamId: number; path: string }[] = [];
    const gridSize = 1;
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
    
    for (const player of allPlayers) {
      const points: [number, number][] = [];
      for (let x = 0; x < grid.length; x++) {
        for (let y = 0; y < grid[x].length; y++) {
          if (grid[x][y].playerId === player.id) {
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
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet" style={{ zIndex: 2 }}>
        <defs>
          <linearGradient id="voronoiPitchGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(255,255,255,0.03)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.05)" />
          </linearGradient>
          <filter id="voronoiLineGlow">
            <feGaussianBlur stdDeviation="0.3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <radialGradient id="voronoiVignette" cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor="transparent" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.3)" />
          </radialGradient>
        </defs>

        <rect x="0" y="0" width="105" height="68" fill="url(#voronoiPitchGrad)" />

        {/* Voronoi cells */}
        {voronoiCells.map((cell, i) => (
          <path
            key={i}
            d={cell.path}
            fill={cell.teamId === 1 ? "var(--color-team-1)" : "var(--color-team-2)"}
            fillOpacity={0.15}
            stroke={cell.teamId === 1 ? "var(--color-team-1)" : "var(--color-team-2)"}
            strokeWidth={0.25}
            strokeOpacity={0.4}
          />
        ))}

        {/* Pitch lines */}
        <g filter="url(#voronoiLineGlow)" stroke="rgba(255,255,255,0.45)" strokeWidth="0.4" fill="none">
          <rect x="0.5" y="0.5" width="104" height="67" rx="0.5" />
          <line x1="52.5" y1="0.5" x2="52.5" y2="67.5" />
          <circle cx="52.5" cy="34" r="9.15" />
          <circle cx="52.5" cy="34" r="0.5" fill="rgba(255,255,255,0.5)" stroke="none" />
          <rect x="0.5" y="13.84" width="16.5" height="40.32" />
          <rect x="0.5" y="24.84" width="5.5" height="18.32" />
          <rect x="88" y="13.84" width="16.5" height="40.32" />
          <rect x="99" y="24.84" width="5.5" height="18.32" />
        </g>

        <rect x="0" y="0" width="105" height="68" fill="url(#voronoiVignette)" />
      </svg>

      {data.team1Players.map((player: any) => (
        <div
          key={player.id}
          className="player-marker team-1"
          style={{
            left: `${(player.x / PITCH_WIDTH) * 100}%`,
            top: `${(player.y / PITCH_HEIGHT) * 100}%`,
          }}
        >
          <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white drop-shadow-sm" style={{ zIndex: 1 }}>
            {player.id}
          </span>
        </div>
      ))}

      {data.team2Players.map((player: any) => (
        <div
          key={player.id}
          className="player-marker team-2"
          style={{
            left: `${(player.x / PITCH_WIDTH) * 100}%`,
            top: `${(player.y / PITCH_HEIGHT) * 100}%`,
          }}
        >
          <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white drop-shadow-sm" style={{ zIndex: 1 }}>
            {player.id - 11}
          </span>
        </div>
      ))}

      <div
        className="ball-marker"
        style={{
          left: `${(data.ball.x / PITCH_WIDTH) * 100}%`,
          top: `${(data.ball.y / PITCH_HEIGHT) * 100}%`,
        }}
      />

      <div className="absolute bottom-3 right-3 bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2 text-[11px] font-medium" style={{ zIndex: 20 }}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: "var(--color-team-1)", opacity: 0.5 }} />
            <span className="text-white/70">Team 1 Zone</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: "var(--color-team-2)", opacity: 0.5 }} />
            <span className="text-white/70">Team 2 Zone</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Stat Row Component - Premium Design
function StatRow({ label, team1, team2, suffix = "" }: { label: string; team1: number; team2: number; suffix?: string }) {
  const total = team1 + team2;
  const team1Pct = total > 0 ? (team1 / total) * 100 : 50;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-mono text-sm font-semibold tabular-nums" style={{ color: "var(--color-team-1)" }}>
          {team1.toFixed(suffix === "%" ? 1 : 0)}{suffix}
        </span>
        <span className="text-xs text-muted-foreground font-medium uppercase tracking-wider">{label}</span>
        <span className="font-mono text-sm font-semibold tabular-nums" style={{ color: "var(--color-team-2)" }}>
          {team2.toFixed(suffix === "%" ? 1 : 0)}{suffix}
        </span>
      </div>
      <div className="h-1.5 bg-secondary/30 rounded-full overflow-hidden flex gap-px">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ 
            width: `${team1Pct}%`, 
            backgroundColor: "var(--color-team-1)",
            boxShadow: "0 0 8px var(--color-team-1)"
          }}
        />
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ 
            width: `${100 - team1Pct}%`, 
            backgroundColor: "var(--color-team-2)",
            boxShadow: "0 0 8px var(--color-team-2)"
          }}
        />
      </div>
    </div>
  );
}
