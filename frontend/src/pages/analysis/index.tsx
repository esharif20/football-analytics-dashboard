import { useState, useMemo, useCallback, useEffect } from "react";
import { useAuth } from "@/_core/hooks/useAuth";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { analysisApi, eventsApi, statisticsApi, commentaryApi } from "@/lib/api-local";
import { Button } from "@/components/ui/button";
import { CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { toast } from "sonner";
import { Link, useParams, useLocation } from "wouter";
import { getLoginUrl } from "@/const";
import {
  Activity, ArrowLeft, XCircle, BarChart3, Eye, Zap,
  Filter, ChevronDown, RotateCcw, Flame, Move, Gauge, ArrowUpDown,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { PIPELINE_MODES, type PipelineMode } from "@/shared/types";
import { useWebSocket, WSMessage } from "@/hooks/useWebSocket";

import { TeamColorsCtx, TEAM1_DEFAULT, TEAM2_DEFAULT, AnimatedSection } from "./context";
import { useDemoStats, useDemoPlayerStats } from "./hooks";
import { VideoPlayer } from "./components/VideoPlayer";
import { QuickStat, StatusBadge, StatRow, ProcessingStatus } from "./components/StatsPanel";
import { PossessionDonut, TeamPerformanceRadar, StatsComparisonBar, TeamShapeChart, DefensiveLineChart, PressingIntensityChart } from "./components/ChartsGrid";
import { ModeSpecificTabs } from "./components/PitchVisualizations";
import { AICommentarySection } from "./components/AICommentary";
import { PlayerStatsTable } from "./components/PlayerStats";
import { EventTimeline } from "./components/EventTimeline";
import { PipelinePerformanceCard, ComingSoonCard, BallTrajectoryDiagram, PlayerInteractionGraph } from "./components/PipelineInfo";

export default function Analysis() {
  const params = useParams<{ id: string }>();
  const analysisId = parseInt(params.id || "0");
  const { loading: authLoading, isAuthenticated } = useAuth();
  const [, navigate] = useLocation();
  const [activeTab, setActiveTab] = useState("radar");
  const [aiTab, setAiTab] = useState("tactical");
  const [filterTeam, setFilterTeam] = useState<"all" | "team1" | "team2">("all");
  const [filterOpen, setFilterOpen] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState<number | null>(null);
  const [realtimeProgress, setRealtimeProgress] = useState<{
    progress?: number; currentStage?: string; eta?: number;
  } | null>(null);

  const handleWSProgress = useCallback((data: WSMessage["data"]) => {
    if (data) {
      setRealtimeProgress({ progress: data.progress, currentStage: data.currentStage, eta: data.eta });
    }
  }, []);

  const queryClient = useQueryClient();

  const handleWSComplete = useCallback(() => {
    setRealtimeProgress(null);
    queryClient.invalidateQueries({ queryKey: ["analysis", analysisId] });
  }, [analysisId, queryClient]);

  const handleWSError = useCallback((error: string) => {
    console.error("WebSocket error:", error);
    setRealtimeProgress(null);
    queryClient.invalidateQueries({ queryKey: ["analysis", analysisId] });
  }, [analysisId, queryClient]);

  const { data: analysis, isLoading: analysisLoading } = useQuery({
    queryKey: ["analysis", analysisId],
    queryFn: () => analysisApi.get(analysisId),
    enabled: isAuthenticated && analysisId > 0,
    refetchInterval: undefined,
  });

  const wsToken = analysis?.wsToken as string | undefined;

  const { isConnected: wsConnected } = useWebSocket({
    analysisId, wsToken,
    onProgress: handleWSProgress, onComplete: handleWSComplete, onError: handleWSError,
    enabled: isAuthenticated && analysisId > 0,
  });

  useEffect(() => {
    if (!analysisId || !isAuthenticated) return;
    if (wsConnected) { queryClient.invalidateQueries({ queryKey: ["analysis", analysisId] }); }
  }, [wsConnected, analysisId, isAuthenticated, queryClient]);

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

  const { data: events } = useQuery({
    queryKey: ["events", analysisId], queryFn: () => eventsApi.list(analysisId),
    enabled: isAuthenticated && analysis?.status === "completed",
  });
  const { data: statistics } = useQuery({
    queryKey: ["statistics", analysisId], queryFn: () => statisticsApi.get(analysisId),
    enabled: isAuthenticated && analysis?.status === "completed",
  });
  const { data: commentaryList } = useQuery({
    queryKey: ["commentary", analysisId], queryFn: () => commentaryApi.list(analysisId),
    enabled: isAuthenticated && analysis?.status === "completed",
  });

  // Dynamic team colors from pipeline (falls back to defaults)
  const teamColors = useMemo(() => ({
    TEAM1_HEX: statistics?.teamColorTeam1 || TEAM1_DEFAULT,
    TEAM2_HEX: statistics?.teamColorTeam2 || TEAM2_DEFAULT,
  }), [statistics?.teamColorTeam1, statistics?.teamColorTeam2]);
  const { TEAM1_HEX, TEAM2_HEX } = teamColors;

  const generateCommentaryMutation = useMutation({
    mutationFn: (data: { type: string; context?: any }) => commentaryApi.generate(analysisId, data),
  });
  const rerunMutation = useMutation({
    mutationFn: (data: { videoId: number; mode: string; fresh?: boolean }) => analysisApi.create(data),
  });

  const handleRerun = useCallback(async (fresh: boolean) => {
    if (!analysis) return;
    try {
      const { id } = await rerunMutation.mutateAsync({
        videoId: analysis.videoId, mode: analysis.mode as PipelineMode, fresh,
      });
      toast.success(fresh ? "Fresh re-run started" : "Re-run started (using cache)");
      navigate(`/analysis/${id}`);
    } catch (e: any) { toast.error(e.message || "Failed to start re-run"); }
  }, [analysis, rerunMutation, navigate]);

  const mode = analysis?.mode as PipelineMode;
  const modeConfig = mode ? PIPELINE_MODES[mode] : null;
  // ==================== Analysis Data ====================
  const analyticsData = useMemo(() => {
    if (!analysis?.analyticsDataUrl) return null;
    try {
      return typeof analysis.analyticsDataUrl === "string"
        ? JSON.parse(analysis.analyticsDataUrl) : analysis.analyticsDataUrl;
    } catch { return null; }
  }, [analysis?.analyticsDataUrl]);

  const demoTrackingData = useMemo(() => null, []);

  const demoEvents = useMemo(() => {
    const source = (events && events.length > 0) ? events
      : Array.isArray(analyticsData?.events) ? analyticsData.events : [];
    return source
      .map((e: any, idx: number) => ({
        id: e.id ?? idx + 1, type: e.type ?? e.event_type ?? "unknown",
        frameNumber: e.frameNumber ?? e.frame_idx ?? 0, timestamp: e.timestamp ?? e.timestamp_sec ?? 0,
        teamId: e.teamId ?? e.team_id ?? null, playerId: e.playerId ?? e.player_track_id ?? null,
        targetPlayerId: e.targetPlayerId ?? e.target_player_track_id ?? null, success: e.success ?? null,
      }))
      .filter((e: any) => typeof e.type === "string");
  }, [events, analyticsData?.events]);

  const ballTrajectoryPoints = useMemo(() => {
    const pitchPositions = analyticsData?.ball_path?.pitch_positions;
    if (!Array.isArray(pitchPositions) || pitchPositions.length < 2) return undefined;
    return pitchPositions.map((p: [number, number] | number[]) => ({ x: p[0] / 100, y: p[1] / 100 }));
  }, [analyticsData?.ball_path?.pitch_positions]);

  const demoStats = useDemoStats(statistics, analyticsData, demoEvents);
  const demoPlayerStats = useDemoPlayerStats(analyticsData, demoEvents);

  const handleGenerateCommentary = async (type: "match_summary" | "tactical_analysis") => {
    try {
      await generateCommentaryMutation.mutateAsync({ type, context: { events: demoEvents, statistics: demoStats } });
      queryClient.invalidateQueries({ queryKey: ["commentary", analysisId] });
      toast.success("Commentary generated!");
    } catch { toast.error("Failed to generate commentary"); }
  };

  // --- Loading / Auth / Not Found States ---
  const fullPageCenter = "min-h-screen bg-background flex items-center justify-center";
  if (authLoading || analysisLoading) return (
    <div className={fullPageCenter}><div className="flex flex-col items-center gap-4">
      <div className="relative"><div className="w-16 h-16 rounded-full border-2 border-primary/20 border-t-primary animate-spin" /><Activity className="w-6 h-6 text-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" /></div>
      <p className="text-muted-foreground text-sm animate-pulse">Loading analysis...</p>
    </div></div>
  );
  if (!isAuthenticated) return (
    <div className={fullPageCenter}><div className="glass-card p-8 max-w-md w-full text-center space-y-4">
      <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto"><Activity className="w-8 h-8 text-primary" /></div>
      <h2 className="text-xl font-bold">Sign In Required</h2>
      <p className="text-muted-foreground text-sm">Please sign in to view this analysis</p>
      <a href={getLoginUrl()}><Button className="w-full mt-2">Sign In</Button></a>
    </div></div>
  );
  if (!analysis) return (
    <div className={fullPageCenter}><div className="glass-card p-8 max-w-md w-full text-center space-y-4">
      <div className="w-16 h-16 rounded-2xl bg-destructive/10 border border-destructive/20 flex items-center justify-center mx-auto"><XCircle className="w-8 h-8 text-destructive" /></div>
      <h2 className="text-xl font-bold">Analysis Not Found</h2>
      <p className="text-muted-foreground text-sm">This analysis doesn&apos;t exist or you don&apos;t have access</p>
      <Link href="/dashboard"><Button className="w-full mt-2">Back to Dashboard</Button></Link>
    </div></div>
  );

  // ==================== Main Render ====================
  return (
    <TeamColorsCtx.Provider value={teamColors}>
    <div className="min-h-screen bg-background">
      <div className="analysis-bg"><div className="floating-orb" /><div className="floating-orb" /><div className="floating-orb" /></div>

      {/* Header */}
      <header className="relative border-b border-white/[0.06] bg-black/20 backdrop-blur-xl sticky top-0 z-50">
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent" />
        <div className="container flex items-center justify-between h-16">
          <div className="flex items-center gap-4">
            <Link href="/dashboard">
              <Button variant="ghost" size="icon" className="rounded-xl hover:bg-primary/10 hover:scale-105 active:scale-95 transition-all duration-200">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/20 to-emerald-500/5 border border-emerald-500/20 flex items-center justify-center shadow-[0_0_15px_rgba(52,211,153,0.1)]">
                <Activity className="w-5 h-5 text-emerald-400" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="font-bold text-lg">Analysis #{analysis.id}</h1>
                  <Badge variant="outline" className="border-primary/30 text-primary bg-primary/5 text-xs">{modeConfig?.name || analysis.mode}</Badge>
                </div>
                {analysis.createdAt && (
                  <p className="text-xs text-muted-foreground">{formatDistanceToNow(new Date(analysis.createdAt), { addSuffix: true })}</p>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {(analysis.status === "completed" || analysis.status === "failed") && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" className="rounded-xl gap-2 text-xs border-border/30 hover:border-orange-500/40 hover:bg-orange-500/10 hover:text-orange-400 transition-all duration-300" disabled={rerunMutation.isPending}>
                    <RotateCcw className={`w-3.5 h-3.5 ${rerunMutation.isPending ? "animate-spin" : ""}`} />
                    {rerunMutation.isPending ? "Starting..." : "Re-run"}
                    <ChevronDown className="w-3 h-3" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuItem onClick={() => handleRerun(false)} className="gap-2 cursor-pointer">
                    <RotateCcw className="w-4 h-4" />
                    <div><div className="font-medium text-xs">Re-run (Use Cache)</div><div className="text-[10px] text-muted-foreground">Faster — reuse cached tracking data</div></div>
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => handleRerun(true)} className="gap-2 cursor-pointer">
                    <Flame className="w-4 h-4 text-orange-400" />
                    <div><div className="font-medium text-xs">Re-run (Fresh)</div><div className="text-[10px] text-muted-foreground">Full pipeline from scratch, no cache</div></div>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            {analysis.status === "completed" && (
              <Button variant="outline" size="sm"
                className={`rounded-xl gap-2 text-xs transition-all duration-300 ${filterOpen ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400" : "border-border/30 hover:border-emerald-500/20"}`}
                onClick={() => setFilterOpen(!filterOpen)}>
                <Filter className="w-3.5 h-3.5" />Filters
                <ChevronDown className={`w-3 h-3 transition-transform duration-300 ${filterOpen ? "rotate-180" : ""}`} />
              </Button>
            )}
            <StatusBadge status={analysis.status} progress={analysis.progress} />
          </div>
        </div>
        {/* Filter Bar */}
        <div className={`overflow-hidden transition-all duration-500 ease-out ${filterOpen ? "max-h-20 opacity-100" : "max-h-0 opacity-0"}`}>
          <div className="container py-3 flex items-center gap-4 border-t border-white/[0.04]">
            <span className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Team</span>
            {(["all", "team1", "team2"] as const).map((t) => (
              <button key={t} onClick={() => setFilterTeam(t)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-300 ${filterTeam === t ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 shadow-[0_0_10px_rgba(52,211,153,0.1)]" : "bg-white/[0.03] text-muted-foreground border border-transparent hover:bg-white/[0.06] hover:text-foreground"}`}>
                {t === "all" ? "All" : t === "team1" ? (
                  <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM1_HEX }} />Team 1</span>
                ) : (
                  <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM2_HEX }} />Team 2</span>
                )}
              </button>
            ))}
            <div className="ml-auto flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Time Range</span>
              <Badge variant="outline" className="text-[10px] border-amber-400/20 text-amber-400/70 bg-amber-500/5">Full Match</Badge>
            </div>
          </div>
        </div>
      </header>

      <main className="container py-8 relative z-10">
        {(analysis.status === "pending" || analysis.status === "processing" || analysis.status === "uploading") && (
          <ProcessingStatus analysis={analysisWithRealtime || analysis} wsConnected={wsConnected} />
        )}
        {analysis.status === "failed" && (
          <div className="glass-card border-destructive/30 mb-8 p-6 hover-lift">
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
        {analysis.status === "completed" && (
          <div className="space-y-6">
            {/* Row 1: Video + Match Stats */}
            <div className="grid lg:grid-cols-5 gap-6">
              <VideoPlayer analysis={analysis} videoUrl={analysis.annotatedVideoUrl || null} />
              <AnimatedSection className="lg:col-span-2 space-y-6" delay={100}>
                <div className="glass-card overflow-hidden hover-lift group">
                  <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full bg-emerald-500/5 blur-3xl group-hover:bg-emerald-500/10 transition-colors duration-700" />
                  <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="section-icon icon-primary"><BarChart3 className="w-4 h-4 text-emerald-400" /></div>
                        <CardTitle className="text-sm font-semibold">Match Statistics</CardTitle>
                      </div>
                      <div className="flex items-center gap-3 text-xs">
                        <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full animate-pulse-slow" style={{ backgroundColor: TEAM1_HEX }} /><span className="text-muted-foreground">Team 1</span></div>
                        <div className="flex items-center gap-1.5"><div className="w-2.5 h-2.5 rounded-full animate-pulse-slow" style={{ backgroundColor: TEAM2_HEX }} /><span className="text-muted-foreground">Team 2</span></div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-4 space-y-4">
                    <StatRow label="POSSESSION (%)" team1={demoStats.possessionTeam1} team2={demoStats.possessionTeam2} suffix="%" />
                    <StatRow label="PASSES" team1={demoStats.passesTeam1} team2={demoStats.passesTeam2} />
                    <StatRow label="SHOTS" team1={demoStats.shotsTeam1} team2={demoStats.shotsTeam2} />
                    <StatRow label="DISTANCE (KM)" team1={demoStats.distanceCoveredTeam1} team2={demoStats.distanceCoveredTeam2} />
                    <StatRow label="AVG SPEED (KM/H)" team1={demoStats.avgSpeedTeam1} team2={demoStats.avgSpeedTeam2} />
                    <StatRow label="MAX SPEED (KM/H)" team1={demoStats.maxSpeedTeam1} team2={demoStats.maxSpeedTeam2} />
                  </CardContent>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <QuickStat label="Poss. Changes" value={demoStats.possessionChanges.toString()} icon={<ArrowUpDown className="w-4 h-4" />} color="primary" />
                  <QuickStat label="Ball Dist (km)" value={demoStats.ballDistance.toFixed(1)} icon={<Move className="w-4 h-4" />} color="accent" />
                  <QuickStat label="Avg Speed" value={`${((demoStats.avgSpeedTeam1 + demoStats.avgSpeedTeam2) / 2).toFixed(1)}`} icon={<Gauge className="w-4 h-4" />} color="team1" />
                  <QuickStat label="Ball Top Speed" value={demoStats.ballMaxSpeed.toFixed(1)} icon={<Zap className="w-4 h-4" />} color="team2" />
                </div>
              </AnimatedSection>
            </div>
            {/* Row 2: Charts */}
            <div className="grid lg:grid-cols-3 gap-6">
              <AnimatedSection delay={50}><PossessionDonut team1={demoStats.possessionTeam1} team2={demoStats.possessionTeam2} /></AnimatedSection>
              <AnimatedSection delay={100}><TeamPerformanceRadar stats={demoStats} /></AnimatedSection>
              <AnimatedSection delay={150}><StatsComparisonBar stats={demoStats} /></AnimatedSection>
            </div>
            {/* Row 3: Timeline Charts */}
            <div className="grid lg:grid-cols-3 gap-6">
              <AnimatedSection delay={50}><TeamShapeChart /></AnimatedSection>
              <AnimatedSection delay={100}><DefensiveLineChart /></AnimatedSection>
              <AnimatedSection delay={150}><PressingIntensityChart /></AnimatedSection>
            </div>
            {/* Row 4: Pitch Visualizations + AI Commentary */}
            <div className="grid lg:grid-cols-5 gap-6">
              <AnimatedSection className="lg:col-span-3" delay={50}>
                <div className="glass-card overflow-hidden hover-lift group">
                  <div className="absolute -top-10 -left-10 w-32 h-32 rounded-full bg-emerald-500/5 blur-3xl group-hover:bg-emerald-500/10 transition-colors duration-700" />
                  <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
                    <div className="flex items-center gap-3">
                      <div className="section-icon icon-primary"><Eye className="w-4 h-4 text-emerald-400" /></div>
                      <div>
                        <CardTitle className="text-sm font-semibold">Pitch Visualizations</CardTitle>
                        <CardDescription className="text-xs mt-0.5">Interactive analysis views &middot; {modeConfig?.name || analysis.mode} mode</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-4">
                    <ModeSpecificTabs mode={mode} activeTab={activeTab} setActiveTab={setActiveTab} trackingData={demoTrackingData}
                      events={demoEvents} filterTeam={filterTeam} selectedPlayer={selectedPlayer} setSelectedPlayer={setSelectedPlayer} />
                  </CardContent>
                </div>
              </AnimatedSection>
              <AnimatedSection className="lg:col-span-2" delay={100}>
                <AICommentarySection aiTab={aiTab} setAiTab={setAiTab} commentaryList={commentaryList}
                  generateCommentaryMutation={generateCommentaryMutation} handleGenerateCommentary={handleGenerateCommentary} />
              </AnimatedSection>
            </div>
            {/* Row 5: Ball Trajectory + Player Interaction */}
            <div className="grid lg:grid-cols-2 gap-6">
              <AnimatedSection delay={50}><BallTrajectoryDiagram ballTrajectoryPoints={ballTrajectoryPoints} /></AnimatedSection>
              <AnimatedSection delay={100}>
                <PlayerInteractionGraph filterTeam={filterTeam} interactionTeam1={statistics?.passNetworkTeam1} interactionTeam2={statistics?.passNetworkTeam2} />
              </AnimatedSection>
            </div>
            {/* Row 6: Per-Player Stats Table */}
            <AnimatedSection delay={50}>
              <PlayerStatsTable players={demoPlayerStats} filterTeam={filterTeam} selectedPlayer={selectedPlayer} setSelectedPlayer={setSelectedPlayer} />
            </AnimatedSection>
            {/* Row 7: Pipeline Performance + Coming Soon */}
            <div className="grid lg:grid-cols-2 gap-6">
              <AnimatedSection delay={50}><PipelinePerformanceCard mode={mode} /></AnimatedSection>
              <AnimatedSection delay={100}><ComingSoonCard /></AnimatedSection>
            </div>
          </div>
        )}
      </main>
    </div>
    </TeamColorsCtx.Provider>
  );
}
