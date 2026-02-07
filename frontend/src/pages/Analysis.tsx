import { useState, useRef, useMemo, useCallback, useEffect } from "react";
import { useAuth } from "@/_core/hooks/useAuth";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { analysisApi, eventsApi, statisticsApi, commentaryApi } from "@/lib/api-local";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { toast } from "sonner";
import { Link, useParams, useLocation } from "wouter";
import { getLoginUrl } from "@/const";
import {
  Activity,
  ArrowLeft,
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
  Crosshair,
  Gauge,
  Timer,
  ArrowUpDown,
  Filter,
  ChevronDown,
  ChevronRight,
  Bot,
  Lock,
  Footprints,
  GitBranch,
  Circle,
  Hash,
  Shield,
  Flame,
  Move,
  RotateCcw,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { PIPELINE_MODES, PROCESSING_STAGES, EVENT_TYPES, PipelineMode } from "@/shared/types";
import { Streamdown } from "streamdown";
import { useWebSocket, WSMessage } from "@/hooks/useWebSocket";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar as RechartsRadar,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  AreaChart,
  Area,
  LineChart,
  Line,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";

// ==================== Constants ====================
const PITCH_WIDTH = 105;
const PITCH_HEIGHT = 68;
const TEAM1_HEX = "#e05252";
const TEAM2_HEX = "#4a9ede";

// ==================== Custom Tooltip ====================
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="text-[10px] text-muted-foreground mb-1 uppercase tracking-wider">{label}</p>
      {payload.map((p: any, i: number) => (
        <p key={i} className="text-xs font-semibold font-mono" style={{ color: p.color }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(1) : p.value}
        </p>
      ))}
    </div>
  );
}

// ==================== Section Wrapper (animation helper) ====================
function AnimatedSection({ children, className = "", delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setVisible(true); obs.disconnect(); } }, { threshold: 0.1 });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

// ==================== Main Component ====================
export default function Analysis() {
  const params = useParams<{ id: string }>();
  const analysisId = parseInt(params.id || "0");
  const { user, loading: authLoading, isAuthenticated } = useAuth();
  const [, navigate] = useLocation();

  const [activeTab, setActiveTab] = useState("radar");
  const [aiTab, setAiTab] = useState("tactical");
  const [filterTeam, setFilterTeam] = useState<"all" | "team1" | "team2">("all");
  const [filterOpen, setFilterOpen] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Real-time progress state from WebSocket
  const [realtimeProgress, setRealtimeProgress] = useState<{
    progress?: number;
    currentStage?: string;
    eta?: number;
  } | null>(null);

  const handleWSProgress = useCallback((data: WSMessage["data"]) => {
    if (data) {
      setRealtimeProgress({
        progress: data.progress,
        currentStage: data.currentStage,
        eta: data.eta,
      });
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

  const { isConnected: wsConnected } = useWebSocket({
    analysisId,
    onProgress: handleWSProgress,
    onComplete: handleWSComplete,
    onError: handleWSError,
    enabled: isAuthenticated && analysisId > 0,
  });

  const { data: analysis, isLoading: analysisLoading } = useQuery({
    queryKey: ["analysis", analysisId],
    queryFn: () => analysisApi.get(analysisId),
    enabled: isAuthenticated && analysisId > 0,
    refetchInterval: wsConnected ? 10000 : 2000,
  });

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
    queryKey: ["events", analysisId],
    queryFn: () => eventsApi.list(analysisId),
    enabled: isAuthenticated && analysis?.status === "completed",
  });

  const { data: statistics } = useQuery({
    queryKey: ["statistics", analysisId],
    queryFn: () => statisticsApi.get(analysisId),
    enabled: isAuthenticated && analysis?.status === "completed",
  });

  const { data: commentaryList } = useQuery({
    queryKey: ["commentary", analysisId],
    queryFn: () => commentaryApi.list(analysisId),
    enabled: isAuthenticated && analysis?.status === "completed",
  });

  const generateCommentaryMutation = useMutation({
    mutationFn: (data: { type: string; context?: any }) => commentaryApi.generate(analysisId, data),
  });
  const rerunMutation = useMutation({
    mutationFn: (data: { videoId: number; mode: string }) => analysisApi.create(data),
  });

  const handleRerun = useCallback(async (fresh: boolean) => {
    if (!analysis) return;
    try {
      const { id } = await rerunMutation.mutateAsync({
        videoId: analysis.videoId,
        mode: analysis.mode as PipelineMode,
      });
      toast.success(fresh ? "Fresh re-run started" : "Re-run started (using cache)");
      navigate(`/analysis/${id}`);
    } catch (e: any) {
      toast.error(e.message || "Failed to start re-run");
    }
  }, [analysis, rerunMutation, navigate]);

  const mode = analysis?.mode as PipelineMode;
  const modeConfig = mode ? PIPELINE_MODES[mode] : null;

  // ==================== Demo Data ====================
  const demoTrackingData = useMemo(() => {
    if (!analysis || analysis.status !== "completed") return null;
    const team1Players = Array.from({ length: 11 }, (_, i) => ({
      id: i + 1, trackId: i + 1, teamId: 1,
      x: 15 + Math.sin(i * 0.8) * 12 + i * 3,
      y: 8 + (i % 4) * 15 + Math.cos(i) * 4,
      speed: 5 + Math.random() * 10,
    }));
    const team2Players = Array.from({ length: 11 }, (_, i) => ({
      id: i + 12, trackId: i + 12, teamId: 2,
      x: 55 + Math.sin(i * 0.8) * 12 + i * 3,
      y: 8 + (i % 4) * 15 + Math.cos(i) * 4,
      speed: 5 + Math.random() * 10,
    }));
    const ball = { x: 52.5 + (Math.random() - 0.5) * 20, y: 34 + (Math.random() - 0.5) * 20, confidence: 0.95 };
    return { team1Players, team2Players, ball };
  }, [analysis]);

  const demoEvents = useMemo(() => {
    if (events && events.length > 0) return events;
    return Array.from({ length: 15 }, (_, i) => ({
      id: i + 1,
      type: ["pass", "shot", "challenge", "interception"][Math.floor(Math.random() * 4)],
      frameNumber: i * 50,
      timestamp: i * 6,
      teamId: Math.random() > 0.5 ? 1 : 2,
      success: Math.random() > 0.3,
    }));
  }, [events]);

  const demoStats = useMemo(() => {
    if (statistics)
      return {
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
      possessionTeam1: 52, possessionTeam2: 48,
      passesTeam1: 245, passesTeam2: 198,
      passAccuracyTeam1: 84.5, passAccuracyTeam2: 79.2,
      shotsTeam1: 8, shotsTeam2: 5,
      distanceCoveredTeam1: 42.5, distanceCoveredTeam2: 41.2,
      avgSpeedTeam1: 7.2, avgSpeedTeam2: 6.9,
    };
  }, [statistics]);

  // Per-player demo stats (keyed by track ID)
  const demoPlayerStats = useMemo(() => {
    if (!demoTrackingData) return [];
    const all = [...demoTrackingData.team1Players, ...demoTrackingData.team2Players];
    return all.map((p) => ({
      trackId: p.trackId,
      teamId: p.teamId,
      distance: +(2 + Math.random() * 6).toFixed(1),
      avgSpeed: +(4 + Math.random() * 6).toFixed(1),
      maxSpeed: +(18 + Math.random() * 14).toFixed(1),
      passes: Math.floor(10 + Math.random() * 40),
      passAcc: +(60 + Math.random() * 35).toFixed(0),
      sprints: Math.floor(2 + Math.random() * 15),
    }));
  }, [demoTrackingData]);

  const handleGenerateCommentary = async (type: "match_summary" | "tactical_analysis") => {
    try {
      await generateCommentaryMutation.mutateAsync({
        type,
        context: { events: demoEvents, statistics: demoStats },
      });
      queryClient.invalidateQueries({ queryKey: ["commentary", analysisId] });
      toast.success("Commentary generated!");
    } catch {
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
          <p className="text-muted-foreground text-sm">This analysis doesn&apos;t exist or you don&apos;t have access</p>
          <Link href="/dashboard">
            <Button className="w-full mt-2">Back to Dashboard</Button>
          </Link>
        </div>
      </div>
    );
  }

  // ==================== Main Render ====================
  return (
    <div className="min-h-screen bg-background">
      {/* Animated gradient background */}
      <div className="analysis-bg">
        <div className="floating-orb" />
        <div className="floating-orb" />
        <div className="floating-orb" />
      </div>

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
          <div className="flex items-center gap-3">
            {/* Re-run */}
            {(analysis.status === "completed" || analysis.status === "failed") && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="rounded-xl gap-2 text-xs border-border/30 hover:border-orange-500/40 hover:bg-orange-500/10 hover:text-orange-400 transition-all duration-300"
                    disabled={rerunMutation.isPending}
                  >
                    <RotateCcw className={`w-3.5 h-3.5 ${rerunMutation.isPending ? "animate-spin" : ""}`} />
                    {rerunMutation.isPending ? "Starting..." : "Re-run"}
                    <ChevronDown className="w-3 h-3" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuItem onClick={() => handleRerun(false)} className="gap-2 cursor-pointer">
                    <RotateCcw className="w-4 h-4" />
                    <div>
                      <div className="font-medium text-xs">Re-run (Use Cache)</div>
                      <div className="text-[10px] text-muted-foreground">Faster — reuse cached tracking data</div>
                    </div>
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => handleRerun(true)} className="gap-2 cursor-pointer">
                    <Flame className="w-4 h-4 text-orange-400" />
                    <div>
                      <div className="font-medium text-xs">Re-run (Fresh)</div>
                      <div className="text-[10px] text-muted-foreground">Full pipeline from scratch, no cache</div>
                    </div>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            {/* Filter Toggle */}
            {analysis.status === "completed" && (
              <Button
                variant="outline"
                size="sm"
                className={`rounded-xl gap-2 text-xs transition-all duration-300 ${filterOpen ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400" : "border-border/30 hover:border-emerald-500/20"}`}
                onClick={() => setFilterOpen(!filterOpen)}
              >
                <Filter className="w-3.5 h-3.5" />
                Filters
                <ChevronDown className={`w-3 h-3 transition-transform duration-300 ${filterOpen ? "rotate-180" : ""}`} />
              </Button>
            )}
            <StatusBadge status={analysis.status} progress={analysis.progress} />
          </div>
        </div>

        {/* Filter Bar (slide down) */}
        <div className={`overflow-hidden transition-all duration-500 ease-out ${filterOpen ? "max-h-20 opacity-100" : "max-h-0 opacity-0"}`}>
          <div className="container py-3 flex items-center gap-4 border-t border-white/[0.04]">
            <span className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Team</span>
            {(["all", "team1", "team2"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setFilterTeam(t)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-300 ${
                  filterTeam === t
                    ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 shadow-[0_0_10px_rgba(52,211,153,0.1)]"
                    : "bg-white/[0.03] text-muted-foreground border border-transparent hover:bg-white/[0.06] hover:text-foreground"
                }`}
              >
                {t === "all" ? "All" : t === "team1" ? (
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM1_HEX }} />
                    Team 1
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: TEAM2_HEX }} />
                    Team 2
                  </span>
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
        {/* Processing Status */}
        {(analysis.status === "pending" || analysis.status === "processing" || analysis.status === "uploading") && (
          <ProcessingStatus analysis={analysisWithRealtime || analysis} wsConnected={wsConnected} />
        )}

        {/* Failed Status */}
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

        {/* ==================== Completed Analysis ==================== */}
        {analysis.status === "completed" && (
          <div className="space-y-6">

            {/* Row 1: Video + Match Stats */}
            <div className="grid lg:grid-cols-5 gap-6">
              {/* Video Player */}
              <AnimatedSection className="lg:col-span-3" delay={0}>
                {analysis.annotatedVideoUrl ? (
                  <div className="glass-card overflow-hidden hover-lift group">
                    <div className="absolute -top-10 -left-10 w-32 h-32 rounded-full bg-sky-500/5 blur-3xl group-hover:bg-sky-500/10 transition-colors duration-700" />
                    <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
                      <div className="flex items-center gap-3">
                        <div className="section-icon icon-accent">
                          <Video className="w-4 h-4 text-sky-400" />
                        </div>
                        <div>
                          <CardTitle className="text-sm font-semibold">Annotated Video</CardTitle>
                          <CardDescription className="text-xs mt-0.5">AI-processed output with bounding boxes &amp; track IDs</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-4">
                      <div className="video-player-container rounded-xl overflow-hidden ring-1 ring-white/5">
                        <video ref={videoRef} src={analysis.annotatedVideoUrl} controls className="w-full h-full object-contain" />
                      </div>
                    </CardContent>
                  </div>
                ) : (
                  <div className="glass-card p-8 flex flex-col items-center justify-center min-h-[300px] hover-lift">
                    <div className="w-16 h-16 rounded-2xl bg-muted/50 flex items-center justify-center mb-4">
                      <Video className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <p className="text-muted-foreground text-sm">No annotated video available for this mode</p>
                  </div>
                )}
              </AnimatedSection>

              {/* Match Statistics + Quick Stats */}
              <AnimatedSection className="lg:col-span-2 space-y-6" delay={100}>
                <div className="glass-card overflow-hidden hover-lift group">
                  <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full bg-emerald-500/5 blur-3xl group-hover:bg-emerald-500/10 transition-colors duration-700" />
                  <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="section-icon icon-primary">
                          <BarChart3 className="w-4 h-4 text-emerald-400" />
                        </div>
                        <CardTitle className="text-sm font-semibold">Match Statistics</CardTitle>
                      </div>
                      <div className="flex items-center gap-3 text-xs">
                        <div className="flex items-center gap-1.5">
                          <div className="w-2.5 h-2.5 rounded-full animate-pulse-slow" style={{ backgroundColor: TEAM1_HEX }} />
                          <span className="text-muted-foreground">Team 1</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-2.5 h-2.5 rounded-full animate-pulse-slow" style={{ backgroundColor: TEAM2_HEX }} />
                          <span className="text-muted-foreground">Team 2</span>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-4 space-y-4">
                    <StatRow label="POSSESSION" team1={demoStats.possessionTeam1} team2={demoStats.possessionTeam2} suffix="%" />
                    <StatRow label="PASSES" team1={demoStats.passesTeam1} team2={demoStats.passesTeam2} />
                    <StatRow label="PASS ACCURACY" team1={demoStats.passAccuracyTeam1} team2={demoStats.passAccuracyTeam2} suffix="%" />
                    <StatRow label="SHOTS" team1={demoStats.shotsTeam1} team2={demoStats.shotsTeam2} />
                    <StatRow label="DISTANCE (KM)" team1={demoStats.distanceCoveredTeam1} team2={demoStats.distanceCoveredTeam2} />
                    <StatRow label="AVG SPEED (KM/H)" team1={demoStats.avgSpeedTeam1} team2={demoStats.avgSpeedTeam2} />
                  </CardContent>
                </div>

                {/* Quick Stats Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <QuickStat label="Total Events" value={demoEvents.length.toString()} icon={<Zap className="w-4 h-4" />} color="primary" />
                  <QuickStat label="Pass Rate" value={`${((demoStats.passAccuracyTeam1 + demoStats.passAccuracyTeam2) / 2).toFixed(0)}%`} icon={<Target className="w-4 h-4" />} color="accent" />
                  <QuickStat label="Total Shots" value={(demoStats.shotsTeam1 + demoStats.shotsTeam2).toString()} icon={<Crosshair className="w-4 h-4" />} color="team1" />
                  <QuickStat label="Avg Speed" value={`${((demoStats.avgSpeedTeam1 + demoStats.avgSpeedTeam2) / 2).toFixed(1)}`} icon={<Gauge className="w-4 h-4" />} color="team2" />
                </div>
              </AnimatedSection>
            </div>

            {/* Row 2: Charts — Possession + Radar + Bar */}
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
                      <div className="section-icon icon-primary">
                        <Eye className="w-4 h-4 text-emerald-400" />
                      </div>
                      <div>
                        <CardTitle className="text-sm font-semibold">Pitch Visualizations</CardTitle>
                        <CardDescription className="text-xs mt-0.5">Interactive analysis views &middot; {modeConfig?.name || analysis.mode} mode</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="p-4">
                    <ModeSpecificTabs
                      mode={mode}
                      activeTab={activeTab}
                      setActiveTab={setActiveTab}
                      trackingData={demoTrackingData}
                      events={demoEvents}
                      filterTeam={filterTeam}
                      selectedPlayer={selectedPlayer}
                      setSelectedPlayer={setSelectedPlayer}
                    />
                  </CardContent>
                </div>
              </AnimatedSection>

              {/* AI Commentary with Tabs */}
              <AnimatedSection className="lg:col-span-2" delay={100}>
                <AICommentarySection
                  aiTab={aiTab}
                  setAiTab={setAiTab}
                  commentaryList={commentaryList}
                  generateCommentaryMutation={generateCommentaryMutation}
                  handleGenerateCommentary={handleGenerateCommentary}
                />
              </AnimatedSection>
            </div>

            {/* Row 5: Ball Trajectory + Player Interaction Network */}
            <div className="grid lg:grid-cols-2 gap-6">
              <AnimatedSection delay={50}>
                <BallTrajectoryDiagram />
              </AnimatedSection>
              <AnimatedSection delay={100}>
                <PlayerInteractionGraph filterTeam={filterTeam} />
              </AnimatedSection>
            </div>

            {/* Row 6: Per-Player Stats Table */}
            <AnimatedSection delay={50}>
              <PlayerStatsTable
                players={demoPlayerStats}
                filterTeam={filterTeam}
                selectedPlayer={selectedPlayer}
                setSelectedPlayer={setSelectedPlayer}
              />
            </AnimatedSection>

            {/* Row 7: Pipeline Performance + Coming Soon */}
            <div className="grid lg:grid-cols-2 gap-6">
              <AnimatedSection delay={50}>
                <PipelinePerformanceCard mode={mode} />
              </AnimatedSection>
              <AnimatedSection delay={100}>
                <ComingSoonCard />
              </AnimatedSection>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// ==================== AI Commentary Section ====================
function AICommentarySection({
  aiTab, setAiTab, commentaryList, generateCommentaryMutation, handleGenerateCommentary,
}: {
  aiTab: string;
  setAiTab: (t: string) => void;
  commentaryList: any;
  generateCommentaryMutation: any;
  handleGenerateCommentary: (type: "match_summary" | "tactical_analysis") => void;
}) {
  return (
    <div className="glass-card overflow-hidden h-full hover-lift group">
      <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full bg-violet-500/5 blur-3xl group-hover:bg-violet-500/10 transition-colors duration-700" />
      <CardHeader className="pb-0 pt-5 px-5 relative">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-9 h-9 rounded-xl bg-violet-500/10 flex items-center justify-center border border-violet-500/20">
            <Sparkles className="w-4 h-4 text-violet-400" />
          </div>
          <div>
            <CardTitle className="text-sm font-semibold">AI Commentary</CardTitle>
            <CardDescription className="text-xs mt-0.5">Tactical analysis grounded in tracking data</CardDescription>
          </div>
        </div>
        {/* AI Tabs */}
        <div className="flex gap-1 bg-black/20 rounded-xl p-1 border border-white/[0.04]">
          {[
            { id: "tactical", label: "Tactical Analysis", icon: <TrendingUp className="w-3.5 h-3.5" /> },
            { id: "commentary", label: "Commentary", icon: <MessageSquare className="w-3.5 h-3.5" /> },
            { id: "chat", label: "Chat Agent", icon: <Bot className="w-3.5 h-3.5" />, comingSoon: true },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => !tab.comingSoon && setAiTab(tab.id)}
              className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-2 rounded-lg text-[11px] font-medium transition-all duration-300 relative ${
                aiTab === tab.id && !tab.comingSoon
                  ? "bg-violet-500/15 text-violet-400 shadow-[0_0_10px_rgba(139,92,246,0.1)] border border-violet-500/20"
                  : tab.comingSoon
                  ? "text-muted-foreground/50 cursor-not-allowed"
                  : "text-muted-foreground hover:text-foreground hover:bg-white/[0.04]"
              }`}
            >
              {tab.icon}
              <span className="hidden sm:inline">{tab.label}</span>
              {tab.comingSoon && (
                <span className="absolute -top-1.5 -right-1 px-1 py-0.5 rounded text-[7px] font-bold bg-amber-500/20 text-amber-400 border border-amber-500/20 leading-none">
                  SOON
                </span>
              )}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="p-4">
        {aiTab === "chat" ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="w-16 h-16 rounded-2xl bg-amber-500/5 border border-amber-500/10 flex items-center justify-center mb-4 relative">
              <Bot className="w-8 h-8 text-amber-400/60" />
              <Lock className="w-4 h-4 text-amber-400/80 absolute -bottom-1 -right-1" />
            </div>
            <h4 className="font-semibold text-sm mb-1">Chat Agent</h4>
            <p className="text-xs text-muted-foreground max-w-[200px]">
              Interactive AI chat for real-time tactical Q&amp;A is coming soon.
            </p>
            <Badge variant="outline" className="mt-3 text-[10px] border-amber-400/20 text-amber-400/70 bg-amber-500/5">
              Coming Soon
            </Badge>
          </div>
        ) : (
          <>
            {commentaryList && commentaryList.length > 0 ? (
              <ScrollArea className="h-[350px]">
                <div className="space-y-3">
                  {commentaryList
                    .filter((c: any) =>
                      aiTab === "tactical"
                        ? c.type === "tactical_analysis"
                        : c.type === "match_summary"
                    )
                    .map((c: any) => (
                      <div key={c.id} className="p-4 rounded-xl bg-secondary/30 border border-border/20 hover:border-violet-500/20 transition-all duration-300">
                        <Badge variant="outline" className="mb-2 text-xs border-violet-500/30 text-violet-400">{c.type}</Badge>
                        <div className="text-sm prose prose-sm dark:prose-invert max-w-none leading-relaxed">
                          <Streamdown>{c.content}</Streamdown>
                        </div>
                      </div>
                    ))}
                  {commentaryList.filter((c: any) =>
                    aiTab === "tactical" ? c.type === "tactical_analysis" : c.type === "match_summary"
                  ).length === 0 && (
                    <EmptyCommentaryState
                      type={aiTab === "tactical" ? "tactical_analysis" : "match_summary"}
                      handleGenerate={handleGenerateCommentary}
                      isPending={generateCommentaryMutation.isPending}
                    />
                  )}
                </div>
              </ScrollArea>
            ) : (
              <EmptyCommentaryState
                type={aiTab === "tactical" ? "tactical_analysis" : "match_summary"}
                handleGenerate={handleGenerateCommentary}
                isPending={generateCommentaryMutation.isPending}
              />
            )}
          </>
        )}
      </CardContent>
    </div>
  );
}

function EmptyCommentaryState({
  type, handleGenerate, isPending,
}: {
  type: "match_summary" | "tactical_analysis";
  handleGenerate: (t: "match_summary" | "tactical_analysis") => void;
  isPending: boolean;
}) {
  return (
    <div className="space-y-4">
      <div className="text-center py-6">
        <div className="w-12 h-12 rounded-2xl bg-violet-500/5 border border-violet-500/10 flex items-center justify-center mx-auto mb-3">
          {type === "tactical_analysis" ? (
            <TrendingUp className="w-6 h-6 text-violet-400/60" />
          ) : (
            <MessageSquare className="w-6 h-6 text-violet-400/60" />
          )}
        </div>
        <p className="text-sm text-muted-foreground">
          {type === "tactical_analysis"
            ? "Generate AI-powered tactical breakdown based on tracking data."
            : "Generate a match summary with key moments and insights."}
        </p>
      </div>
      <Button
        variant="outline"
        className="w-full justify-center gap-2 h-11 rounded-xl border-violet-500/20 hover:border-violet-500/40 hover:bg-violet-500/5 transition-all duration-300 group"
        onClick={() => handleGenerate(type)}
        disabled={isPending}
      >
        {isPending ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <>
            <Sparkles className="w-4 h-4 text-violet-400 group-hover:scale-110 transition-transform duration-300" />
            <span className="text-sm font-medium">Generate {type === "tactical_analysis" ? "Tactical Analysis" : "Match Summary"}</span>
          </>
        )}
      </Button>
    </div>
  );
}

// ==================== Ball Trajectory Diagram ====================
function BallTrajectoryDiagram() {
  const trajectoryPoints = useMemo(() => {
    const pts: { x: number; y: number }[] = [];
    let cx = 52.5, cy = 34;
    for (let i = 0; i < 60; i++) {
      cx += (Math.random() - 0.48) * 6;
      cy += (Math.random() - 0.5) * 4;
      cx = Math.max(2, Math.min(103, cx));
      cy = Math.max(2, Math.min(66, cy));
      pts.push({ x: cx, y: cy });
    }
    return pts;
  }, []);

  const pathD = useMemo(() => {
    if (trajectoryPoints.length < 2) return "";
    let d = `M ${trajectoryPoints[0].x} ${trajectoryPoints[0].y}`;
    for (let i = 1; i < trajectoryPoints.length; i++) {
      const prev = trajectoryPoints[i - 1];
      const curr = trajectoryPoints[i];
      const cpx = (prev.x + curr.x) / 2;
      const cpy = (prev.y + curr.y) / 2;
      d += ` Q ${prev.x + (curr.x - prev.x) * 0.3} ${prev.y + (curr.y - prev.y) * 0.1}, ${cpx} ${cpy}`;
    }
    return d;
  }, [trajectoryPoints]);

  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute -top-10 -left-10 w-32 h-32 rounded-full bg-amber-500/5 blur-3xl group-hover:bg-amber-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
              <Move className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <CardTitle className="text-sm font-semibold">Ball Trajectory</CardTitle>
              <CardDescription className="text-xs mt-0.5">Movement path across the pitch</CardDescription>
            </div>
          </div>
          <Badge variant="outline" className="text-[10px] border-amber-400/20 text-amber-400/70 bg-amber-500/5">Placeholder</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-4">
        <div className="pitch-container">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet" style={{ zIndex: 2 }}>
            <defs>
              <linearGradient id="trajGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="rgba(251,191,36,0.1)" />
                <stop offset="50%" stopColor="rgba(251,191,36,0.7)" />
                <stop offset="100%" stopColor="rgba(251,191,36,1)" />
              </linearGradient>
              <filter id="trajGlow">
                <feGaussianBlur stdDeviation="0.8" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
              <filter id="trajLineGlow"><feGaussianBlur stdDeviation="0.4" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
            </defs>
            {/* Pitch lines */}
            <g filter="url(#trajLineGlow)" stroke="rgba(52,211,153,0.35)" strokeWidth="0.3" fill="none">
              <rect x="0.5" y="0.5" width="104" height="67" rx="0.5" />
              <line x1="52.5" y1="0.5" x2="52.5" y2="67.5" />
              <circle cx="52.5" cy="34" r="9.15" />
              <circle cx="52.5" cy="34" r="0.5" fill="rgba(52,211,153,0.3)" stroke="none" />
              <rect x="0.5" y="13.84" width="16.5" height="40.32" />
              <rect x="0.5" y="24.84" width="5.5" height="18.32" />
              <rect x="88" y="13.84" width="16.5" height="40.32" />
              <rect x="99" y="24.84" width="5.5" height="18.32" />
            </g>
            {/* Trajectory path */}
            <path d={pathD} fill="none" stroke="url(#trajGrad)" strokeWidth="0.8" strokeLinecap="round" filter="url(#trajGlow)" className="ball-trajectory-path" />
            {/* Trail dots */}
            {trajectoryPoints.filter((_, i) => i % 4 === 0).map((pt, i) => (
              <circle key={i} cx={pt.x} cy={pt.y} r={0.6 + (i / trajectoryPoints.length) * 0.8} fill="rgba(251,191,36,0.5)" opacity={0.3 + (i / trajectoryPoints.length) * 0.7} />
            ))}
            {/* End marker */}
            <circle cx={trajectoryPoints[trajectoryPoints.length - 1].x} cy={trajectoryPoints[trajectoryPoints.length - 1].y} r="1.8" fill="rgba(251,191,36,0.9)" stroke="white" strokeWidth="0.4">
              <animate attributeName="r" values="1.5;2.2;1.5" dur="2s" repeatCount="indefinite" />
            </circle>
          </svg>
        </div>
      </CardContent>
    </div>
  );
}

// ==================== Player Interaction Graph ====================
function PlayerInteractionGraph({ filterTeam }: { filterTeam: "all" | "team1" | "team2" }) {
  const nodes = useMemo(() => [
    { id: 1, teamId: 1, x: 15, y: 34, passes: 45 },
    { id: 2, teamId: 1, x: 28, y: 14, passes: 38 },
    { id: 3, teamId: 1, x: 28, y: 54, passes: 42 },
    { id: 4, teamId: 1, x: 42, y: 24, passes: 52 },
    { id: 5, teamId: 1, x: 42, y: 44, passes: 48 },
    { id: 6, teamId: 1, x: 50, y: 34, passes: 35 },
    { id: 12, teamId: 2, x: 58, y: 34, passes: 40 },
    { id: 13, teamId: 2, x: 70, y: 14, passes: 36 },
    { id: 14, teamId: 2, x: 70, y: 54, passes: 44 },
    { id: 15, teamId: 2, x: 82, y: 24, passes: 50 },
    { id: 16, teamId: 2, x: 82, y: 44, passes: 46 },
    { id: 17, teamId: 2, x: 92, y: 34, passes: 32 },
  ], []);

  const edges = useMemo(() => [
    { from: 1, to: 2, count: 14 }, { from: 1, to: 3, count: 16 },
    { from: 2, to: 4, count: 20 }, { from: 3, to: 5, count: 15 },
    { from: 4, to: 6, count: 12 }, { from: 5, to: 6, count: 10 },
    { from: 4, to: 5, count: 8 }, { from: 2, to: 3, count: 6 },
    { from: 12, to: 13, count: 13 }, { from: 12, to: 14, count: 17 },
    { from: 13, to: 15, count: 18 }, { from: 14, to: 16, count: 14 },
    { from: 15, to: 17, count: 11 }, { from: 16, to: 17, count: 9 },
    { from: 15, to: 16, count: 7 },
  ], []);

  const filteredNodes = filterTeam === "all" ? nodes : nodes.filter((n) => (filterTeam === "team1" ? n.teamId === 1 : n.teamId === 2));
  const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));
  const filteredEdges = edges.filter((e) => filteredNodeIds.has(e.from) && filteredNodeIds.has(e.to));

  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full bg-cyan-500/5 blur-3xl group-hover:bg-cyan-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-500/10 flex items-center justify-center border border-cyan-500/20">
              <GitBranch className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <CardTitle className="text-sm font-semibold">Player Interaction Network</CardTitle>
              <CardDescription className="text-xs mt-0.5">Pass connections between players</CardDescription>
            </div>
          </div>
          <Badge variant="outline" className="text-[10px] border-cyan-400/20 text-cyan-400/70 bg-cyan-500/5">Placeholder</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-4">
        <div className="pitch-container">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet" style={{ zIndex: 2 }}>
            <defs>
              <filter id="netLineGlow"><feGaussianBlur stdDeviation="0.3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
              <filter id="netEdgeGlow"><feGaussianBlur stdDeviation="0.5" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
              <marker id="netArrow" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
                <polygon points="0 0, 6 2, 0 4" fill="rgba(255,255,255,0.4)" />
              </marker>
            </defs>
            {/* Pitch lines */}
            <g filter="url(#netLineGlow)" stroke="rgba(52,211,153,0.25)" strokeWidth="0.25" fill="none">
              <rect x="0.5" y="0.5" width="104" height="67" rx="0.5" />
              <line x1="52.5" y1="0.5" x2="52.5" y2="67.5" />
              <circle cx="52.5" cy="34" r="9.15" />
              <rect x="0.5" y="13.84" width="16.5" height="40.32" />
              <rect x="88" y="13.84" width="16.5" height="40.32" />
            </g>
            {/* Edges */}
            <g filter="url(#netEdgeGlow)">
              {filteredEdges.map((edge, i) => {
                const from = filteredNodes.find((n) => n.id === edge.from)!;
                const to = filteredNodes.find((n) => n.id === edge.to)!;
                if (!from || !to) return null;
                const color = from.teamId === 1 ? TEAM1_HEX : TEAM2_HEX;
                return (
                  <line key={i} x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                    stroke={color} strokeWidth={Math.max(0.3, edge.count / 8)}
                    strokeOpacity={0.4 + (edge.count / 40)} markerEnd="url(#netArrow)"
                  />
                );
              })}
            </g>
          </svg>
          {/* Player nodes */}
          {filteredNodes.map((node) => (
            <div
              key={node.id}
              className="player-node-interaction"
              style={{
                left: `${(node.x / PITCH_WIDTH) * 100}%`,
                top: `${(node.y / PITCH_HEIGHT) * 100}%`,
                "--node-color": node.teamId === 1 ? TEAM1_HEX : TEAM2_HEX,
              } as React.CSSProperties}
            >
              <span className="text-[9px] font-bold text-white drop-shadow-md">{node.id}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </div>
  );
}

// ==================== Per-Player Stats Table ====================
function PlayerStatsTable({
  players, filterTeam, selectedPlayer, setSelectedPlayer,
}: {
  players: any[];
  filterTeam: "all" | "team1" | "team2";
  selectedPlayer: number | null;
  setSelectedPlayer: (id: number | null) => void;
}) {
  const filtered = filterTeam === "all" ? players : players.filter((p) => (filterTeam === "team1" ? p.teamId === 1 : p.teamId === 2));

  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute -top-10 -left-10 w-32 h-32 rounded-full bg-violet-500/5 blur-3xl group-hover:bg-violet-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="section-icon icon-accent">
              <Hash className="w-4 h-4 text-sky-400" />
            </div>
            <div>
              <CardTitle className="text-sm font-semibold">Per-Player Statistics</CardTitle>
              <CardDescription className="text-xs mt-0.5">Individual metrics by track ID &middot; Click to highlight on pitch</CardDescription>
            </div>
          </div>
          <Badge variant="outline" className="text-[10px] border-violet-400/20 text-violet-400/70 bg-violet-500/5">Placeholder</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border/20">
                <th className="text-left py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Track ID</th>
                <th className="text-left py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Team</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Distance (km)</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Avg Speed</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Max Speed</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Passes</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Pass Acc%</th>
                <th className="text-right py-3 px-4 text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Sprints</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p) => {
                const isSelected = selectedPlayer === p.trackId;
                const teamColor = p.teamId === 1 ? TEAM1_HEX : TEAM2_HEX;
                return (
                  <tr
                    key={p.trackId}
                    onClick={() => setSelectedPlayer(isSelected ? null : p.trackId)}
                    className={`border-b border-border/10 cursor-pointer transition-all duration-300 ${
                      isSelected
                        ? "bg-white/[0.06] shadow-[inset_3px_0_0_0_var(--row-color)]"
                        : "hover:bg-white/[0.03]"
                    }`}
                    style={{ "--row-color": teamColor } as React.CSSProperties}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <div className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold text-white border border-white/20"
                          style={{ backgroundColor: teamColor }}>
                          {p.trackId}
                        </div>
                        <span className="font-mono font-semibold">#{p.trackId}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-medium" style={{ backgroundColor: `${teamColor}20`, color: teamColor }}>
                        Team {p.teamId}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-mono">{p.distance}</td>
                    <td className="py-3 px-4 text-right font-mono">{p.avgSpeed} km/h</td>
                    <td className="py-3 px-4 text-right font-mono font-semibold" style={{ color: teamColor }}>{p.maxSpeed} km/h</td>
                    <td className="py-3 px-4 text-right font-mono">{p.passes}</td>
                    <td className="py-3 px-4 text-right font-mono">{p.passAcc}%</td>
                    <td className="py-3 px-4 text-right">
                      <span className="inline-flex items-center gap-1">
                        <Flame className="w-3 h-3 text-orange-400" />
                        <span className="font-mono">{p.sprints}</span>
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </div>
  );
}

// ==================== Coming Soon Card ====================
function ComingSoonCard() {
  return (
    <div className="glass-card overflow-hidden hover-lift group h-full">
      <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full bg-rose-500/5 blur-3xl group-hover:bg-rose-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-rose-500/10 flex items-center justify-center border border-rose-500/20">
            <Sparkles className="w-4 h-4 text-rose-400" />
          </div>
          <div>
            <CardTitle className="text-sm font-semibold">Coming Soon</CardTitle>
            <CardDescription className="text-xs mt-0.5">Planned features for future releases</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-5 space-y-3">
        {[
          { icon: <Users className="w-4 h-4" />, title: "Player Detail Dashboard", desc: "Individual player deep-dive with per-track-ID heatmaps, speed profiles, and event timelines", color: "violet" },
          { icon: <Bot className="w-4 h-4" />, title: "Interactive Chat Agent", desc: "Ask tactical questions in natural language and get AI-powered answers grounded in match data", color: "amber" },
          { icon: <Radar className="w-4 h-4" />, title: "Formation Detection", desc: "Automatic formation classification (4-3-3, 4-4-2, etc.) from player positions over time", color: "cyan" },
          { icon: <Shield className="w-4 h-4" />, title: "Defensive Metrics", desc: "PPDA, high press success rate, recoveries, and defensive action zones", color: "emerald" },
        ].map((item, i) => {
          const colorMap: Record<string, string> = {
            violet: "bg-violet-500/10 text-violet-400 border-violet-500/20",
            amber: "bg-amber-500/10 text-amber-400 border-amber-500/20",
            cyan: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
            emerald: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
          };
          return (
            <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/[0.04] hover:border-white/[0.08] hover:bg-white/[0.04] transition-all duration-300 group/item">
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center border shrink-0 ${colorMap[item.color]} group-hover/item:scale-110 transition-transform duration-300`}>
                {item.icon}
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <h4 className="text-xs font-semibold">{item.title}</h4>
                  <Lock className="w-3 h-3 text-muted-foreground/50" />
                </div>
                <p className="text-[11px] text-muted-foreground mt-0.5 leading-relaxed">{item.desc}</p>
              </div>
            </div>
          );
        })}
      </CardContent>
    </div>
  );
}

// ==================== CHART COMPONENTS ====================

function PossessionDonut({ team1, team2 }: { team1: number; team2: number }) {
  const data = [
    { name: "Team 1", value: team1, color: TEAM1_HEX },
    { name: "Team 2", value: team2, color: TEAM2_HEX },
  ];
  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full bg-rose-500/5 blur-3xl group-hover:bg-rose-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30">
        <div className="flex items-center gap-3">
          <div className="section-icon icon-danger"><Target className="w-4 h-4 text-rose-400" /></div>
          <CardTitle className="text-sm">Possession</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-4">
        <div className="relative">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={data} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={4} dataKey="value" strokeWidth={0} animationBegin={200} animationDuration={1200}>
                {data.map((entry, index) => (
                  <Cell key={index} fill={entry.color} style={{ filter: `drop-shadow(0 0 8px ${entry.color}60)` }} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <span className="text-3xl font-bold font-mono text-rose-400" style={{ textShadow: "0 0 20px rgba(220,50,50,0.3)" }}>{team1}%</span>
            <span className="text-[10px] text-muted-foreground uppercase tracking-widest mt-1">Team 1</span>
          </div>
        </div>
        <div className="flex justify-center gap-6 mt-3">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full ring-2 ring-rose-400/20" style={{ backgroundColor: TEAM1_HEX }} />
            <span className="text-xs text-muted-foreground">Team 1 &mdash; <span className="text-rose-400 font-semibold">{team1}%</span></span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full ring-2 ring-sky-400/20" style={{ backgroundColor: TEAM2_HEX }} />
            <span className="text-xs text-muted-foreground">Team 2 &mdash; <span className="text-sky-400 font-semibold">{team2}%</span></span>
          </div>
        </div>
      </CardContent>
    </div>
  );
}

function TeamPerformanceRadar({ stats }: { stats: any }) {
  const data = [
    { subject: "Passing", team1: stats.passAccuracyTeam1, team2: stats.passAccuracyTeam2, fullMark: 100 },
    { subject: "Speed", team1: (stats.avgSpeedTeam1 / 12) * 100, team2: (stats.avgSpeedTeam2 / 12) * 100, fullMark: 100 },
    { subject: "Shots", team1: (stats.shotsTeam1 / 15) * 100, team2: (stats.shotsTeam2 / 15) * 100, fullMark: 100 },
    { subject: "Distance", team1: (stats.distanceCoveredTeam1 / 55) * 100, team2: (stats.distanceCoveredTeam2 / 55) * 100, fullMark: 100 },
    { subject: "Possession", team1: stats.possessionTeam1, team2: stats.possessionTeam2, fullMark: 100 },
    { subject: "Accuracy", team1: stats.passAccuracyTeam1, team2: stats.passAccuracyTeam2, fullMark: 100 },
  ];
  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute -top-10 -left-10 w-32 h-32 rounded-full bg-sky-500/5 blur-3xl group-hover:bg-sky-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30">
        <div className="flex items-center gap-3">
          <div className="section-icon icon-accent"><Radar className="w-4 h-4 text-sky-400" /></div>
          <CardTitle className="text-sm">Team Performance</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-2">
        <ResponsiveContainer width="100%" height={240}>
          <RadarChart cx="50%" cy="50%" outerRadius="68%" data={data}>
            <PolarGrid stroke="rgba(255,255,255,0.06)" gridType="polygon" />
            <PolarAngleAxis dataKey="subject" tick={{ fill: "rgba(255,255,255,0.55)", fontSize: 10, fontWeight: 500 }} />
            <PolarRadiusAxis tick={false} axisLine={false} domain={[0, 100]} />
            <RechartsRadar name="Team 1" dataKey="team1" stroke={TEAM1_HEX} fill={TEAM1_HEX} fillOpacity={0.15} strokeWidth={2} animationDuration={1500} animationBegin={300} />
            <RechartsRadar name="Team 2" dataKey="team2" stroke={TEAM2_HEX} fill={TEAM2_HEX} fillOpacity={0.15} strokeWidth={2} animationDuration={1500} animationBegin={500} />
            <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "12px" }} iconType="circle" iconSize={8} formatter={(value: string) => <span style={{ color: "rgba(255,255,255,0.6)" }}>{value}</span>} />
          </RadarChart>
        </ResponsiveContainer>
      </CardContent>
    </div>
  );
}

function StatsComparisonBar({ stats }: { stats: any }) {
  const data = [
    { name: "Passes", team1: stats.passesTeam1, team2: stats.passesTeam2 },
    { name: "Shots", team1: stats.shotsTeam1 * 20, team2: stats.shotsTeam2 * 20 },
    { name: "Dist", team1: stats.distanceCoveredTeam1 * 5, team2: stats.distanceCoveredTeam2 * 5 },
    { name: "Acc%", team1: stats.passAccuracyTeam1 * 2.5, team2: stats.passAccuracyTeam2 * 2.5 },
  ];
  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute -bottom-10 -right-10 w-32 h-32 rounded-full bg-emerald-500/5 blur-3xl group-hover:bg-emerald-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30">
        <div className="flex items-center gap-3">
          <div className="section-icon icon-primary"><BarChart3 className="w-4 h-4 text-emerald-400" /></div>
          <CardTitle className="text-sm">Stats Comparison</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-2">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data} barGap={3} barCategoryGap="25%">
            <defs>
              <linearGradient id="barT1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={TEAM1_HEX} stopOpacity={0.9} />
                <stop offset="100%" stopColor={TEAM1_HEX} stopOpacity={0.5} />
              </linearGradient>
              <linearGradient id="barT2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={TEAM2_HEX} stopOpacity={0.9} />
                <stop offset="100%" stopColor={TEAM2_HEX} stopOpacity={0.5} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="name" tick={{ fill: "rgba(255,255,255,0.55)", fontSize: 10, fontWeight: 500 }} axisLine={false} tickLine={false} />
            <YAxis hide />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="team1" name="Team 1" fill="url(#barT1)" radius={[6, 6, 0, 0]} animationDuration={1200} animationBegin={200} />
            <Bar dataKey="team2" name="Team 2" fill="url(#barT2)" radius={[6, 6, 0, 0]} animationDuration={1200} animationBegin={400} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </div>
  );
}

// Timeline Charts
function TeamShapeChart() {
  const data = useMemo(() => Array.from({ length: 20 }, (_, i) => ({
    minute: i * 4.5 + 1,
    team1: 25 + Math.sin(i * 0.5) * 8 + Math.random() * 4,
    team2: 28 + Math.cos(i * 0.4) * 6 + Math.random() * 4,
  })), []);
  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute -top-8 -left-8 w-24 h-24 rounded-full bg-violet-500/5 blur-3xl group-hover:bg-violet-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="section-icon icon-primary"><BarChart3 className="w-4 h-4 text-emerald-400" /></div>
            <div>
              <CardTitle className="text-sm">Team Compactness</CardTitle>
              <CardDescription className="text-[10px]">Formation spread over time (m)</CardDescription>
            </div>
          </div>
          <Badge variant="outline" className="text-[10px] border-violet-400/20 text-violet-400/70 bg-violet-500/5">Planned</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-2">
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="compactT1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={TEAM1_HEX} stopOpacity={0.35} />
                <stop offset="95%" stopColor={TEAM1_HEX} stopOpacity={0} />
              </linearGradient>
              <linearGradient id="compactT2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={TEAM2_HEX} stopOpacity={0.35} />
                <stop offset="95%" stopColor={TEAM2_HEX} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="minute" tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${Math.round(v)}'`} />
            <YAxis tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 9 }} axisLine={false} tickLine={false} domain={[10, 45]} />
            <Tooltip content={<ChartTooltip />} />
            <Area type="monotone" dataKey="team1" name="Team 1" stroke={TEAM1_HEX} fill="url(#compactT1)" strokeWidth={2} animationDuration={1500} />
            <Area type="monotone" dataKey="team2" name="Team 2" stroke={TEAM2_HEX} fill="url(#compactT2)" strokeWidth={2} animationDuration={1500} />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </div>
  );
}

function DefensiveLineChart() {
  const data = useMemo(() => Array.from({ length: 20 }, (_, i) => ({
    minute: i * 4.5 + 1,
    team1: 35 + Math.sin(i * 0.3) * 12 + Math.random() * 3,
    team2: 70 - Math.sin(i * 0.3) * 12 + Math.random() * 3,
  })), []);
  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute -top-8 -right-8 w-24 h-24 rounded-full bg-amber-500/5 blur-3xl group-hover:bg-amber-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
              <ArrowUpDown className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <CardTitle className="text-sm">Defensive Line</CardTitle>
              <CardDescription className="text-[10px]">Average defensive height (m from goal)</CardDescription>
            </div>
          </div>
          <Badge variant="outline" className="text-[10px] border-amber-400/20 text-amber-400/70 bg-amber-500/5">Planned</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-2">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="minute" tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${Math.round(v)}'`} />
            <YAxis tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 9 }} axisLine={false} tickLine={false} domain={[20, 85]} />
            <Tooltip content={<ChartTooltip />} />
            <Line type="monotone" dataKey="team1" name="Team 1" stroke={TEAM1_HEX} strokeWidth={2.5} dot={false} animationDuration={1500} strokeLinecap="round" />
            <Line type="monotone" dataKey="team2" name="Team 2" stroke={TEAM2_HEX} strokeWidth={2.5} dot={false} animationDuration={1500} strokeLinecap="round" />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </div>
  );
}

function PressingIntensityChart() {
  const data = useMemo(() => Array.from({ length: 20 }, (_, i) => ({
    minute: i * 4.5 + 1,
    team1: 40 + Math.sin(i * 0.6) * 25 + Math.random() * 10,
    team2: 45 + Math.cos(i * 0.5) * 20 + Math.random() * 10,
  })), []);
  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute -bottom-8 -left-8 w-24 h-24 rounded-full bg-orange-500/5 blur-3xl group-hover:bg-orange-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-orange-500/10 flex items-center justify-center border border-orange-500/20">
              <Zap className="w-4 h-4 text-orange-400" />
            </div>
            <div>
              <CardTitle className="text-sm">Pressing Intensity</CardTitle>
              <CardDescription className="text-[10px]">High-press actions per 5-min window</CardDescription>
            </div>
          </div>
          <Badge variant="outline" className="text-[10px] border-orange-400/20 text-orange-400/70 bg-orange-500/5">Planned</Badge>
        </div>
      </CardHeader>
      <CardContent className="p-2">
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="pressT1" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={TEAM1_HEX} stopOpacity={0.4} />
                <stop offset="95%" stopColor={TEAM1_HEX} stopOpacity={0} />
              </linearGradient>
              <linearGradient id="pressT2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={TEAM2_HEX} stopOpacity={0.4} />
                <stop offset="95%" stopColor={TEAM2_HEX} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="minute" tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${Math.round(v)}'`} />
            <YAxis tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 9 }} axisLine={false} tickLine={false} />
            <Tooltip content={<ChartTooltip />} />
            <Area type="monotone" dataKey="team1" name="Team 1" stroke={TEAM1_HEX} fill="url(#pressT1)" strokeWidth={2} animationDuration={1500} />
            <Area type="monotone" dataKey="team2" name="Team 2" stroke={TEAM2_HEX} fill="url(#pressT2)" strokeWidth={2} animationDuration={1500} />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </div>
  );
}

// Pipeline Performance Card
function PipelinePerformanceCard({ mode }: { mode: PipelineMode }) {
  const stages = [
    { name: "Detection", model: "YOLOv8x", metric: "99.4% mAP@50", time: "~12s/frame", icon: <Crosshair className="w-4 h-4" />, color: "emerald" },
    { name: "Ball Detection", model: "YOLOv8x + SAHI", metric: "92.5% mAP@50", time: "~8s/frame", icon: <Target className="w-4 h-4" />, color: "amber" },
    { name: "Tracking", model: "ByteTrack", metric: "25 fps", time: "~2s/frame", icon: <Activity className="w-4 h-4" />, color: "sky" },
    { name: "Team Class.", model: "SigLIP + KMeans", metric: "k=2 clusters", time: "~5s total", icon: <Users className="w-4 h-4" />, color: "violet" },
    { name: "Pitch Map", model: "YOLOv8x-pose", metric: "0.97 mAP@50", time: "~3s/frame", icon: <Map className="w-4 h-4" />, color: "rose" },
    { name: "Analytics", model: "Custom", metric: "8 metrics", time: "~1s total", icon: <BarChart3 className="w-4 h-4" />, color: "orange" },
  ];
  const colorClasses: Record<string, { bg: string; text: string; border: string }> = {
    emerald: { bg: "bg-emerald-500/10", text: "text-emerald-400", border: "hover:border-emerald-500/20" },
    amber: { bg: "bg-amber-500/10", text: "text-amber-400", border: "hover:border-amber-500/20" },
    sky: { bg: "bg-sky-500/10", text: "text-sky-400", border: "hover:border-sky-500/20" },
    violet: { bg: "bg-violet-500/10", text: "text-violet-400", border: "hover:border-violet-500/20" },
    rose: { bg: "bg-rose-500/10", text: "text-rose-400", border: "hover:border-rose-500/20" },
    orange: { bg: "bg-orange-500/10", text: "text-orange-400", border: "hover:border-orange-500/20" },
  };
  return (
    <div className="glass-card overflow-hidden hover-lift group">
      <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/3 via-transparent to-violet-500/3" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
        <div className="flex items-center gap-3">
          <div className="section-icon icon-primary"><Timer className="w-4 h-4 text-emerald-400" /></div>
          <div>
            <CardTitle className="text-sm font-semibold">Pipeline Architecture</CardTitle>
            <CardDescription className="text-xs mt-0.5">Model performance from CS350 evaluation</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 relative">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {stages.map((stage, i) => {
            const cc = colorClasses[stage.color] || colorClasses.emerald;
            return (
              <div key={i} className={`p-3 rounded-xl bg-white/[0.02] border border-white/[0.04] ${cc.border} transition-all duration-300 group/stage relative overflow-hidden hover:scale-[1.02] hover:bg-white/[0.04]`}>
                <div className={`absolute -top-4 -right-4 w-12 h-12 rounded-full ${cc.bg} blur-xl opacity-0 group-hover/stage:opacity-100 transition-opacity duration-500`} />
                <div className="relative">
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`${cc.text} opacity-70 group-hover/stage:opacity-100 transition-opacity duration-300`}>{stage.icon}</span>
                    <span className="text-xs font-semibold truncate">{stage.name}</span>
                  </div>
                  <div className="space-y-1">
                    <p className="text-[10px] text-muted-foreground">{stage.model}</p>
                    <p className={`text-xs font-mono font-bold ${cc.text}`}>{stage.metric}</p>
                    <p className="text-[10px] text-muted-foreground">{stage.time}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </div>
  );
}

// ==================== Sub-Components ====================

function QuickStat({ label, value, icon, color }: { label: string; value: string; icon: React.ReactNode; color: string }) {
  const colorMap: Record<string, { text: string; bg: string }> = {
    primary: { text: "text-emerald-400", bg: "bg-emerald-500/10" },
    accent: { text: "text-sky-400", bg: "bg-sky-500/10" },
    team1: { text: "text-rose-400", bg: "bg-rose-500/10" },
    team2: { text: "text-blue-400", bg: "bg-blue-500/10" },
  };
  const c = colorMap[color] || colorMap.primary;
  return (
    <div className="glass-card p-4 group hover:border-white/10 hover-lift relative overflow-hidden">
      <div className={`absolute -top-4 -right-4 w-20 h-20 rounded-full ${c.bg} blur-2xl opacity-40 group-hover:opacity-60 transition-opacity duration-500`} />
      <div className="relative">
        <div className="flex items-center gap-2 mb-3">
          <div className={`w-8 h-8 rounded-lg ${c.bg} flex items-center justify-center group-hover:scale-110 transition-transform duration-300`}>
            <span className={c.text}>{icon}</span>
          </div>
          <span className="text-[11px] text-muted-foreground uppercase tracking-wider font-medium">{label}</span>
        </div>
        <div className={`text-2xl font-bold font-mono ${c.text}`}>{value}</div>
      </div>
    </div>
  );
}

function StatusBadge({ status, progress }: { status: string; progress: number }) {
  const config: Record<string, { icon: React.ReactNode; text: string; className: string }> = {
    pending: { icon: <Clock className="w-3.5 h-3.5" />, text: "Pending", className: "bg-muted/50 text-muted-foreground border-border/30" },
    uploading: { icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />, text: "Uploading", className: "bg-primary/10 text-primary border-primary/20" },
    processing: { icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />, text: `Processing ${progress}%`, className: "bg-primary/10 text-primary border-primary/20" },
    completed: { icon: <CheckCircle2 className="w-3.5 h-3.5" />, text: "Completed", className: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" },
    failed: { icon: <XCircle className="w-3.5 h-3.5" />, text: "Failed", className: "bg-destructive/10 text-destructive border-destructive/20" },
  };
  const c = config[status] || config.pending;
  return (
    <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border ${c.className}`}>
      {c.icon}
      {c.text}
    </div>
  );
}

function ProcessingStatus({ analysis, wsConnected = false }: { analysis: any; wsConnected?: boolean }) {
  const currentStageIndex = PROCESSING_STAGES.findIndex((s) => s.id === analysis.currentStage);
  const queryClient = useQueryClient();

  const { data: etaData } = useQuery({
    queryKey: ["analysis-eta", analysis.id],
    queryFn: () => analysisApi.eta(analysis.id),
    enabled: (analysis.status === "processing" || analysis.status === "pending") && !wsConnected,
    refetchInterval: wsConnected ? 30000 : 5000,
  });

  const displayEta = analysis.eta !== undefined ? analysis.eta * 1000 : etaData?.remainingMs;

  const terminateMutation = useMutation({
    mutationFn: () => analysisApi.terminate(analysis.id),
    onSuccess: () => { toast.success("Analysis terminated"); queryClient.invalidateQueries({ queryKey: ["analysis", analysis.id] }); },
    onError: (error: any) => { toast.error(error.message || "Failed to terminate"); },
  });

  const handleTerminate = () => {
    if (confirm("Are you sure you want to terminate this analysis? This cannot be undone.")) {
      terminateMutation.mutate();
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
                  ? PROCESSING_STAGES.find((s) => s.id === analysis.currentStage)?.name || analysis.currentStage
                  : "Initializing..."}
              </p>
            </div>
          </div>
          <Button
            variant="outline" size="sm"
            className="rounded-xl border-destructive/30 text-destructive hover:bg-destructive/10 hover:border-destructive/50 transition-all duration-300"
            onClick={handleTerminate} disabled={terminateMutation.isPending}
          >
            {terminateMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : (<><XCircle className="w-4 h-4 mr-1.5" />Terminate</>)}
          </Button>
        </div>
        <div className="space-y-3">
          <div className="h-2 bg-secondary/50 rounded-full overflow-hidden">
            <div className="h-full rounded-full bg-gradient-to-r from-primary to-accent transition-all duration-500 ease-out relative" style={{ width: `${analysis.progress}%` }}>
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent bg-[length:200%_100%] animate-[shimmer_1.5s_linear_infinite]" />
            </div>
          </div>
          {(displayEta !== undefined || etaData) && (
            <div className="flex items-center justify-between px-1">
              <div className="flex items-center gap-2">
                <Clock className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">Estimated time remaining</span>
                {wsConnected && (
                  <span className="text-[10px] text-emerald-400 font-medium px-1.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">LIVE</span>
                )}
              </div>
              <span className="text-sm font-semibold font-mono text-primary">{formatTime(displayEta ?? etaData?.remainingMs ?? 0)}</span>
            </div>
          )}
        </div>
        <div className="mt-6 grid grid-cols-4 sm:grid-cols-8 gap-2">
          {PROCESSING_STAGES.map((stage, i) => {
            const isComplete = i < currentStageIndex;
            const isCurrent = i === currentStageIndex;
            return (
              <div key={stage.id} className={`relative text-center p-2.5 rounded-xl text-xs font-medium transition-all duration-500 ${
                isComplete ? "bg-primary/15 text-primary border border-primary/20"
                : isCurrent ? "bg-primary/20 text-primary border border-primary/40 shadow-[0_0_12px_rgba(var(--color-primary),0.15)]"
                : "bg-secondary/30 text-muted-foreground border border-transparent"
              }`}>
                {isComplete && <CheckCircle2 className="w-3 h-3 absolute top-1 right-1 text-primary" />}
                {isCurrent && <div className="absolute inset-0 rounded-xl bg-primary/5 animate-pulse" />}
                <span className="relative">{stage.name.split(" ")[0]}</span>
              </div>
            );
          })}
        </div>
        {etaData && (
          <div className="text-xs text-muted-foreground text-center mt-4">
            Elapsed: {formatTime(etaData.elapsedMs)} &middot; Stage {Math.max(1, etaData.stageIndex + 1)} of {etaData.totalStages}
          </div>
        )}
      </div>
    </div>
  );
}

// ==================== Pitch Visualization Components ====================

function PitchRadar({ data, selectedPlayer }: { data: any; selectedPlayer?: number | null }) {
  if (!data) return <div className="pitch-container flex items-center justify-center"><p className="text-muted-foreground">No tracking data available</p></div>;
  return (
    <div className="pitch-container">
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet" style={{ zIndex: 2 }}>
        <defs>
          <linearGradient id="pitchGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(255,255,255,0.03)" />
            <stop offset="50%" stopColor="rgba(255,255,255,0)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.05)" />
          </linearGradient>
          <filter id="lineGlow"><feGaussianBlur stdDeviation="0.4" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="centerGlow"><feGaussianBlur stdDeviation="1.5" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <radialGradient id="vignette" cx="50%" cy="50%" r="60%"><stop offset="0%" stopColor="transparent" /><stop offset="100%" stopColor="rgba(0,0,0,0.35)" /></radialGradient>
        </defs>
        <rect x="0" y="0" width="105" height="68" fill="url(#pitchGrad)" />
        <circle cx="52.5" cy="34" r="16" fill="rgba(52,211,153,0.04)" filter="url(#centerGlow)" />
        <g filter="url(#lineGlow)" stroke="rgba(52,211,153,0.45)" strokeWidth="0.35" fill="none">
          <rect x="0.5" y="0.5" width="104" height="67" rx="0.5" />
          <line x1="52.5" y1="0.5" x2="52.5" y2="67.5" />
          <circle cx="52.5" cy="34" r="9.15" />
          <circle cx="52.5" cy="34" r="0.7" fill="rgba(52,211,153,0.5)" stroke="none" />
          <rect x="0.5" y="13.84" width="16.5" height="40.32" />
          <rect x="0.5" y="24.84" width="5.5" height="18.32" />
          <rect x="88" y="13.84" width="16.5" height="40.32" />
          <rect x="99" y="24.84" width="5.5" height="18.32" />
          <path d="M 16.5 24.84 A 9.15 9.15 0 0 1 16.5 43.16" />
          <path d="M 88.5 24.84 A 9.15 9.15 0 0 0 88.5 43.16" />
          <circle cx="11" cy="34" r="0.5" fill="rgba(52,211,153,0.3)" stroke="none" />
          <circle cx="94" cy="34" r="0.5" fill="rgba(52,211,153,0.3)" stroke="none" />
          <path d="M 0.5 1.5 A 1 1 0 0 0 1.5 0.5" />
          <path d="M 103.5 0.5 A 1 1 0 0 0 104.5 1.5" />
          <path d="M 0.5 66.5 A 1 1 0 0 1 1.5 67.5" />
          <path d="M 103.5 67.5 A 1 1 0 0 1 104.5 66.5" />
          <line x1="0" y1="30" x2="0" y2="38" strokeWidth="0.8" strokeOpacity="0.2" />
          <line x1="105" y1="30" x2="105" y2="38" strokeWidth="0.8" strokeOpacity="0.2" />
        </g>
        <rect x="0" y="0" width="105" height="68" fill="url(#vignette)" />
      </svg>
      {/* Team legend */}
      <div className="absolute top-3 right-3 bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2 text-[11px] font-medium" style={{ zIndex: 20 }}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full" style={{ backgroundColor: TEAM1_HEX }} /><span className="text-white/70">Team 1</span></div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full" style={{ backgroundColor: TEAM2_HEX }} /><span className="text-white/70">Team 2</span></div>
        </div>
      </div>
      {/* Players */}
      {data.team1Players.map((player: any) => (
        <PlayerNode key={player.id} player={player} teamId={1} isSelected={selectedPlayer === player.trackId} />
      ))}
      {data.team2Players.map((player: any) => (
        <PlayerNode key={player.id} player={player} teamId={2} isSelected={selectedPlayer === player.trackId} />
      ))}
      <div className="ball-marker" style={{ left: `${(data.ball.x / PITCH_WIDTH) * 100}%`, top: `${(data.ball.y / PITCH_HEIGHT) * 100}%` }} />
    </div>
  );
}

// Premium Player Node
function PlayerNode({ player, teamId, isSelected }: { player: any; teamId: number; isSelected?: boolean }) {
  const color = teamId === 1 ? TEAM1_HEX : TEAM2_HEX;
  return (
    <div
      className={`player-node ${isSelected ? "player-node-selected" : ""}`}
      style={{
        left: `${(player.x / PITCH_WIDTH) * 100}%`,
        top: `${(player.y / PITCH_HEIGHT) * 100}%`,
        "--player-color": color,
      } as React.CSSProperties}
    >
      <span className="player-node-label">{player.trackId}</span>
    </div>
  );
}

function HeatmapView() {
  return (
    <div className="pitch-container">
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet" style={{ zIndex: 2 }}>
        <defs>
          <filter id="heatLineGlow"><feGaussianBlur stdDeviation="0.3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <radialGradient id="hz1"><stop offset="0%" stopColor="oklch(0.55 0.22 25)" stopOpacity="0.85" /><stop offset="100%" stopColor="oklch(0.55 0.2 25)" stopOpacity="0" /></radialGradient>
          <radialGradient id="hz2"><stop offset="0%" stopColor="oklch(0.65 0.22 45)" stopOpacity="0.7" /><stop offset="100%" stopColor="oklch(0.65 0.2 45)" stopOpacity="0" /></radialGradient>
          <radialGradient id="hz3"><stop offset="0%" stopColor="oklch(0.6 0.2 145)" stopOpacity="0.5" /><stop offset="100%" stopColor="oklch(0.6 0.2 145)" stopOpacity="0" /></radialGradient>
          <filter id="heatBlur"><feGaussianBlur stdDeviation="3" /></filter>
        </defs>
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
        <g filter="url(#heatBlur)" style={{ mixBlendMode: "screen" }}>
          <ellipse cx="28" cy="30" rx="14" ry="18" fill="url(#hz1)" />
          <ellipse cx="52.5" cy="34" rx="18" ry="14" fill="url(#hz2)" />
          <ellipse cx="77" cy="38" rx="14" ry="18" fill="url(#hz1)" />
          <ellipse cx="40" cy="20" rx="10" ry="12" fill="url(#hz3)" />
          <ellipse cx="65" cy="48" rx="10" ry="12" fill="url(#hz3)" />
        </g>
      </svg>
      <div className="absolute bottom-3 right-3 bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2 text-[11px] font-medium" style={{ zIndex: 20 }}>
        <div className="flex items-center gap-2">
          <div className="w-20 h-2 rounded-full heatmap-gradient" />
          <span className="text-white/70">Low → High</span>
        </div>
      </div>
    </div>
  );
}

function PassNetworkView() {
  const nodes = [
    { id: 1, x: 15, y: 34, passes: 45 }, { id: 2, x: 30, y: 15, passes: 38 },
    { id: 3, x: 30, y: 53, passes: 42 }, { id: 4, x: 45, y: 25, passes: 52 },
    { id: 5, x: 45, y: 43, passes: 48 }, { id: 6, x: 60, y: 34, passes: 35 },
  ];
  const edges = [
    { from: 1, to: 2, count: 12 }, { from: 1, to: 3, count: 15 },
    { from: 2, to: 4, count: 18 }, { from: 3, to: 5, count: 14 },
    { from: 4, to: 6, count: 10 }, { from: 5, to: 6, count: 8 }, { from: 4, to: 5, count: 6 },
  ];
  return (
    <div className="pitch-container">
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet" style={{ zIndex: 2 }}>
        <defs>
          <filter id="passLineGlow"><feGaussianBlur stdDeviation="0.3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="oklch(0.7 0.18 145)" fillOpacity="0.7" />
          </marker>
          <filter id="edgeGlow"><feGaussianBlur stdDeviation="0.5" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
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
        <g filter="url(#edgeGlow)">
          {edges.map((edge, i) => {
            const fromNode = nodes.find((n) => n.id === edge.from)!;
            const toNode = nodes.find((n) => n.id === edge.to)!;
            return <line key={i} x1={fromNode.x} y1={fromNode.y} x2={toNode.x} y2={toNode.y} stroke="oklch(0.7 0.18 145)" strokeWidth={Math.max(0.5, edge.count / 6)} strokeOpacity={0.5} markerEnd="url(#arrowhead)" />;
          })}
        </g>
      </svg>
      {nodes.map((node) => (
        <div key={node.id} className="player-node" style={{
          left: `${(node.x / PITCH_WIDTH) * 100}%`, top: `${(node.y / PITCH_HEIGHT) * 100}%`,
          "--player-color": TEAM2_HEX,
        } as React.CSSProperties}>
          <span className="player-node-label">{node.id}</span>
        </div>
      ))}
    </div>
  );
}

function VoronoiView({ data }: { data: any }) {
  if (!data) return <div className="pitch-container flex items-center justify-center"><p className="text-muted-foreground">No tracking data available</p></div>;
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
          if (dist < minDist) { minDist = dist; closestPlayer = player; }
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
            const isBoundary = x === 0 || x === grid.length - 1 || y === 0 || y === grid[x].length - 1 ||
              grid[x - 1]?.[y]?.playerId !== player.id || grid[x + 1]?.[y]?.playerId !== player.id ||
              grid[x]?.[y - 1]?.playerId !== player.id || grid[x]?.[y + 1]?.playerId !== player.id;
            if (isBoundary) points.push([grid[x][y].x, grid[x][y].y]);
          }
        }
      }
      if (points.length > 2) {
        const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
        const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;
        points.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
        cells.push({ playerId: player.id, teamId: player.teamId, path: `M ${points.map((p) => `${p[0]} ${p[1]}`).join(" L ")} Z` });
      }
    }
    return cells;
  }, [allPlayers]);
  return (
    <div className="pitch-container">
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 105 68" preserveAspectRatio="xMidYMid meet" style={{ zIndex: 2 }}>
        <defs>
          <filter id="voronoiLineGlow"><feGaussianBlur stdDeviation="0.3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        {voronoiCells.map((cell, i) => (
          <path key={i} d={cell.path} fill={cell.teamId === 1 ? TEAM1_HEX : TEAM2_HEX} fillOpacity={0.15} stroke={cell.teamId === 1 ? TEAM1_HEX : TEAM2_HEX} strokeWidth={0.25} strokeOpacity={0.4} />
        ))}
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
      </svg>
      {data.team1Players.map((player: any) => (
        <PlayerNode key={player.id} player={player} teamId={1} />
      ))}
      {data.team2Players.map((player: any) => (
        <PlayerNode key={player.id} player={player} teamId={2} />
      ))}
      <div className="ball-marker" style={{ left: `${(data.ball.x / PITCH_WIDTH) * 100}%`, top: `${(data.ball.y / PITCH_HEIGHT) * 100}%` }} />
      <div className="absolute bottom-3 right-3 bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2 text-[11px] font-medium" style={{ zIndex: 20 }}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded" style={{ backgroundColor: TEAM1_HEX, opacity: 0.5 }} /><span className="text-white/70">Team 1</span></div>
          <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded" style={{ backgroundColor: TEAM2_HEX, opacity: 0.5 }} /><span className="text-white/70">Team 2</span></div>
        </div>
      </div>
    </div>
  );
}

function EventTimeline({ events }: { events: any[] }) {
  return (
    <div className="space-y-4">
      <div className="relative h-12 bg-secondary/30 rounded-xl overflow-hidden border border-border/20">
        {events.map((event, i) => {
          const position = (event.timestamp / 90) * 100;
          const eventConfig = EVENT_TYPES[event.type as keyof typeof EVENT_TYPES];
          return <div key={i} className="event-marker" style={{ left: `${Math.min(position, 98)}%`, backgroundColor: eventConfig?.color || "#666" }} title={`${event.type} at ${event.timestamp}s`} />;
        })}
      </div>
      <ScrollArea className="h-48">
        <div className="space-y-2">
          {events.slice(0, 10).map((event, i) => {
            const eventConfig = EVENT_TYPES[event.type as keyof typeof EVENT_TYPES];
            return (
              <div key={i} className="flex items-center gap-3 p-3 rounded-xl bg-secondary/20 border border-border/10 hover:border-border/30 hover:bg-secondary/30 transition-all duration-300">
                <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: eventConfig?.color || "#666" }} />
                <div className="flex-1 min-w-0">
                  <span className="font-medium capitalize text-sm">{event.type}</span>
                  <span className="text-muted-foreground text-xs ml-2">Team {event.teamId} &middot; {event.timestamp}s</span>
                </div>
                <Badge variant="outline" className={`text-xs shrink-0 ${event.success ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/5" : "border-border/30 text-muted-foreground"}`}>
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

// ==================== Mode-Specific Tabs ====================
function ModeSpecificTabs({
  mode, activeTab, setActiveTab, trackingData, events, filterTeam, selectedPlayer, setSelectedPlayer,
}: {
  mode: PipelineMode; activeTab: string; setActiveTab: (t: string) => void;
  trackingData: any; events: any[]; filterTeam: string;
  selectedPlayer: number | null; setSelectedPlayer: (id: number | null) => void;
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
  const tabConfig: Record<string, { icon: React.ReactNode; label: string }> = {
    radar: { icon: <Radar className="w-4 h-4" />, label: "Radar" },
    voronoi: { icon: <Users className="w-4 h-4" />, label: "Voronoi" },
    heatmap: { icon: <Map className="w-4 h-4" />, label: "Heatmap" },
    passes: { icon: <Target className="w-4 h-4" />, label: "Passes" },
    events: { icon: <Zap className="w-4 h-4" />, label: "Events" },
  };
  const effectiveTab = availableTabs.includes(activeTab) ? activeTab : availableTabs[0];
  return (
    <Tabs value={effectiveTab} onValueChange={setActiveTab}>
      <TabsList className="w-full bg-black/20 border border-white/[0.06] rounded-xl p-1 h-auto backdrop-blur-sm" style={{ gridTemplateColumns: `repeat(${availableTabs.length}, 1fr)`, display: "grid" }}>
        {availableTabs.map((tab) => {
          const config = tabConfig[tab];
          return (
            <TabsTrigger key={tab} value={tab} className="gap-2 rounded-lg data-[state=active]:bg-emerald-500/15 data-[state=active]:text-emerald-400 data-[state=active]:shadow-[0_0_12px_rgba(52,211,153,0.1)] data-[state=active]:border data-[state=active]:border-emerald-500/20 data-[state=inactive]:text-muted-foreground py-2.5 transition-all duration-300">
              {config.icon}
              <span className="hidden sm:inline text-xs font-medium">{config.label}</span>
            </TabsTrigger>
          );
        })}
      </TabsList>
      <TabsContent value="radar" className="mt-4"><PitchRadar data={trackingData} selectedPlayer={selectedPlayer} /></TabsContent>
      <TabsContent value="voronoi" className="mt-4"><VoronoiView data={trackingData} /></TabsContent>
      <TabsContent value="heatmap" className="mt-4"><HeatmapView /></TabsContent>
      <TabsContent value="passes" className="mt-4"><PassNetworkView /></TabsContent>
      <TabsContent value="events" className="mt-4"><EventTimeline events={events} /></TabsContent>
    </Tabs>
  );
}

// ==================== Stat Row ====================
function StatRow({ label, team1, team2, suffix = "" }: { label: string; team1: number; team2: number; suffix?: string }) {
  const total = team1 + team2;
  const team1Pct = total > 0 ? (team1 / total) * 100 : 50;
  return (
    <div className="space-y-2 group">
      <div className="flex items-center justify-between">
        <span className="font-mono text-sm font-bold tabular-nums" style={{ color: TEAM1_HEX, textShadow: `0 0 10px ${TEAM1_HEX}40` }}>
          {team1.toFixed(suffix === "%" ? 1 : 0)}{suffix}
        </span>
        <span className="text-[10px] text-muted-foreground font-semibold tracking-[0.15em] uppercase">{label}</span>
        <span className="font-mono text-sm font-bold tabular-nums" style={{ color: TEAM2_HEX, textShadow: `0 0 10px ${TEAM2_HEX}40` }}>
          {team2.toFixed(suffix === "%" ? 1 : 0)}{suffix}
        </span>
      </div>
      <div className="h-2 bg-white/[0.03] rounded-full overflow-hidden flex gap-[2px] group-hover:h-2.5 transition-all duration-300">
        <div className="h-full rounded-full transition-all duration-1000 ease-out relative overflow-hidden" style={{ width: `${team1Pct}%`, backgroundColor: TEAM1_HEX, boxShadow: `0 0 12px ${TEAM1_HEX}80, 0 0 4px ${TEAM1_HEX}` }}>
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent" style={{ backgroundSize: "200% 100%" }} />
        </div>
        <div className="h-full rounded-full transition-all duration-1000 ease-out relative overflow-hidden" style={{ width: `${100 - team1Pct}%`, backgroundColor: TEAM2_HEX, boxShadow: `0 0 12px ${TEAM2_HEX}80, 0 0 4px ${TEAM2_HEX}` }}>
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent" style={{ backgroundSize: "200% 100%" }} />
        </div>
      </div>
    </div>
  );
}
