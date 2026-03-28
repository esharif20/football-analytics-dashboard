import { useMemo } from "react";
import { CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
} from "recharts";
import {
  Target,
  Radar,
  BarChart3,
  ArrowUpDown,
  Zap,
} from "lucide-react";
import { useTeamColors, ChartTooltip } from "../context";

export function PossessionDonut({ team1, team2 }: { team1: number; team2: number }) {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors();
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
            <span className="text-3xl font-bold font-mono" style={{ color: TEAM1_HEX, textShadow: `0 0 20px ${TEAM1_HEX}4D` }}>{team1.toFixed(1)}%</span>
            <span className="text-[10px] text-muted-foreground uppercase tracking-widest mt-1">Team 1</span>
          </div>
        </div>
        <div className="flex justify-center gap-6 mt-3">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: TEAM1_HEX, boxShadow: `0 0 0 2px ${TEAM1_HEX}33` }} />
            <span className="text-xs text-muted-foreground">Team 1 &mdash; <span className="font-semibold" style={{ color: TEAM1_HEX }}>{team1.toFixed(1)}%</span></span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: TEAM2_HEX, boxShadow: `0 0 0 2px ${TEAM2_HEX}33` }} />
            <span className="text-xs text-muted-foreground">Team 2 &mdash; <span className="font-semibold" style={{ color: TEAM2_HEX }}>{team2.toFixed(1)}%</span></span>
          </div>
        </div>
      </CardContent>
    </div>
  );
}

export function TeamPerformanceRadar({ stats }: { stats: any }) {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors();
  const data = [
    { subject: "Possession", team1: stats.possessionTeam1, team2: stats.possessionTeam2, fullMark: 100 },
    { subject: "Distance", team1: (stats.distanceCoveredTeam1 / 55) * 100, team2: (stats.distanceCoveredTeam2 / 55) * 100, fullMark: 100 },
    { subject: "Avg Speed", team1: (stats.avgSpeedTeam1 / 12) * 100, team2: (stats.avgSpeedTeam2 / 12) * 100, fullMark: 100 },
    { subject: "Max Speed", team1: (stats.maxSpeedTeam1 / 40) * 100, team2: (stats.maxSpeedTeam2 / 40) * 100, fullMark: 100 },
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

export function StatsComparisonBar({ stats }: { stats: any }) {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors();
  const data = [
    { name: "Dist (km)", team1: stats.distanceCoveredTeam1 * 5, team2: stats.distanceCoveredTeam2 * 5 },
    { name: "Avg Spd", team1: stats.avgSpeedTeam1 * 20, team2: stats.avgSpeedTeam2 * 20 },
    { name: "Max Spd", team1: stats.maxSpeedTeam1 * 5, team2: stats.maxSpeedTeam2 * 5 },
    { name: "Poss %", team1: stats.possessionTeam1 * 2.5, team2: stats.possessionTeam2 * 2.5 },
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
export function TeamShapeChart() {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors();
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

export function DefensiveLineChart() {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors();
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

export function PressingIntensityChart() {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors();
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
