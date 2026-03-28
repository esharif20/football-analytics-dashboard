import React from "react";
import { useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { analysisApi } from "@/lib/api-local";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import {
  Clock,
  Loader2,
  CheckCircle2,
  XCircle,
  Flame,
} from "lucide-react";
import { PROCESSING_STAGES } from "@/shared/types";
import { useTeamColors } from "../context";

export function QuickStat({ label, value, icon, color }: { label: string; value: string; icon: React.ReactNode; color: string }) {
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

export function StatusBadge({ status, progress }: { status: string; progress: number }) {
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

export function StatRow({ label, team1, team2, suffix = "" }: { label: string; team1: number; team2: number; suffix?: string }) {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors();
  const total = team1 + team2;
  const team1Pct = total > 0 ? (team1 / total) * 100 : 50;
  return (
    <div className="space-y-2 group">
      <div className="flex items-center justify-between">
        <span className="font-mono text-sm font-bold tabular-nums" style={{ color: TEAM1_HEX, textShadow: `0 0 10px ${TEAM1_HEX}40` }}>
          {team1.toFixed(1)}{suffix}
        </span>
        <span className="text-[10px] text-muted-foreground font-semibold tracking-[0.15em] uppercase">{label}</span>
        <span className="font-mono text-sm font-bold tabular-nums" style={{ color: TEAM2_HEX, textShadow: `0 0 10px ${TEAM2_HEX}40` }}>
          {team2.toFixed(1)}{suffix}
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

export function ProcessingStatus({ analysis, wsConnected = false }: { analysis: any; wsConnected?: boolean }) {
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
