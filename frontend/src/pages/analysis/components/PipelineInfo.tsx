import React from 'react'
import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import { dijkstraPassRoute } from '../utils/pathSimplify'
import { CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tooltip as RadixTooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip'
import {
  Activity,
  Target,
  Users,
  BarChart3,
  Map,
  Timer,
  Crosshair,
  Sparkles,
  Bot,
  Lock,
  Shield,
  Radar,
  Zap,
  Move,
  GitBranch,
} from 'lucide-react'
import { useTeamColors, PITCH_WIDTH, PITCH_HEIGHT } from '../context'
import { rdpSimplify, catmullRomToSvgPath } from '../utils/pathSimplify'
import type { PipelineMode } from '../context'

// ==================== Interfaces ====================
export interface InteractionNode {
  playerId: number
  teamId: number
  avgX: number
  avgY: number
  avgSpeed: number
  passCount?: number
  isBall?: boolean
  betweenness?: number
  clustering?: number
  degreeCentrality?: number
}
export interface InteractionEdge {
  from: number
  to: number
  weight: number
  passCount?: number
  isBallEdge?: boolean
}
export interface TimelineSegment {
  label: string
  startFrame: number
  endFrame: number
  edges: InteractionEdge[]
}
export interface InteractionGraphData {
  nodes: InteractionNode[]
  edges: InteractionEdge[]
  timeline?: TimelineSegment[]
}

// ==================== Pipeline Performance Card ====================
export function PipelinePerformanceCard({ mode }: { mode: PipelineMode }) {
  const stages = [
    {
      name: 'Detection',
      model: 'YOLOv8x',
      metric: '99.4% mAP@50',
      time: '~12s/frame',
      icon: <Crosshair className="w-4 h-4" />,
      color: 'emerald',
    },
    {
      name: 'Ball Detection',
      model: 'YOLOv8x + SAHI',
      metric: '92.5% mAP@50',
      time: '~8s/frame',
      icon: <Target className="w-4 h-4" />,
      color: 'amber',
    },
    {
      name: 'Tracking',
      model: 'ByteTrack',
      metric: '25 fps',
      time: '~2s/frame',
      icon: <Activity className="w-4 h-4" />,
      color: 'sky',
    },
    {
      name: 'Team Class.',
      model: 'SigLIP + KMeans',
      metric: 'k=2 clusters',
      time: '~5s total',
      icon: <Users className="w-4 h-4" />,
      color: 'violet',
    },
    {
      name: 'Pitch Map',
      model: 'YOLOv8x-pose',
      metric: '0.97 mAP@50',
      time: '~3s/frame',
      icon: <Map className="w-4 h-4" />,
      color: 'rose',
    },
    {
      name: 'Analytics',
      model: 'Custom',
      metric: '8 metrics',
      time: '~1s total',
      icon: <BarChart3 className="w-4 h-4" />,
      color: 'orange',
    },
  ]
  const colorClasses: Record<string, { bg: string; text: string; border: string }> = {
    emerald: {
      bg: 'bg-emerald-500/10',
      text: 'text-emerald-400',
      border: 'hover:border-emerald-500/20',
    },
    amber: { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'hover:border-amber-500/20' },
    sky: { bg: 'bg-sky-500/10', text: 'text-sky-400', border: 'hover:border-sky-500/20' },
    violet: {
      bg: 'bg-violet-500/10',
      text: 'text-violet-400',
      border: 'hover:border-violet-500/20',
    },
    rose: { bg: 'bg-rose-500/10', text: 'text-rose-400', border: 'hover:border-rose-500/20' },
    orange: {
      bg: 'bg-orange-500/10',
      text: 'text-orange-400',
      border: 'hover:border-orange-500/20',
    },
  }
  return (
    <motion.div
      whileHover={{ y: -6, transition: { duration: 0.3, ease: [0.22, 1, 0.36, 1] } }}
      className="glass-card overflow-hidden group relative"
    >
      <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/3 via-transparent to-violet-500/3" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
        <div className="flex items-center gap-3">
          <div className="section-icon icon-primary">
            <Timer className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <CardTitle className="text-base font-black uppercase tracking-tight" style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}>Pipeline Architecture</CardTitle>
            <CardDescription className="text-xs mt-0.5">
              Model performance from CS350 evaluation
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-4 relative">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {stages.map((stage, i) => {
            const cc = colorClasses[stage.color] || colorClasses.emerald
            return (
              <div
                key={i}
                className={`p-3 rounded-xl bg-white/[0.02] border border-white/[0.04] ${cc.border} transition-all duration-300 group/stage relative overflow-hidden hover:scale-[1.02] hover:bg-white/[0.04]`}
              >
                <div
                  className={`absolute -top-4 -right-4 w-12 h-12 rounded-full ${cc.bg} blur-xl opacity-0 group-hover/stage:opacity-100 transition-opacity duration-500`}
                />
                <div className="relative">
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className={`${cc.text} opacity-70 group-hover/stage:opacity-100 transition-opacity duration-300`}
                    >
                      {stage.icon}
                    </span>
                    <span className="text-xs font-semibold truncate">{stage.name}</span>
                  </div>
                  <div className="space-y-1">
                    <p className="text-[10px] text-muted-foreground">{stage.model}</p>
                    <p className={`text-xs font-mono font-bold ${cc.text}`}>{stage.metric}</p>
                    <p className="text-[10px] text-muted-foreground">{stage.time}</p>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </motion.div>
  )
}

// ==================== Coming Soon Card ====================
export function ComingSoonCard() {
  return (
    <motion.div
      whileHover={{ y: -6, transition: { duration: 0.3, ease: [0.22, 1, 0.36, 1] } }}
      className="glass-card overflow-hidden group relative h-full"
    >
      <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full bg-rose-500/5 blur-3xl group-hover:bg-rose-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-rose-500/10 flex items-center justify-center border border-rose-500/20">
            <Sparkles className="w-4 h-4 text-rose-400" />
          </div>
          <div>
            <CardTitle className="text-sm font-semibold">Coming Soon</CardTitle>
            <CardDescription className="text-xs mt-0.5">
              Planned features for future releases
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-5 space-y-3">
        {[
          {
            icon: <Users className="w-4 h-4" />,
            title: 'Player Detail Dashboard',
            desc: 'Individual player deep-dive with per-track-ID heatmaps, speed profiles, and event timelines',
            color: 'violet',
          },
          {
            icon: <Bot className="w-4 h-4" />,
            title: 'Interactive Chat Agent',
            desc: 'Ask tactical questions in natural language and get AI-powered answers grounded in match data',
            color: 'amber',
          },
          {
            icon: <Radar className="w-4 h-4" />,
            title: 'Formation Detection',
            desc: 'Automatic formation classification (4-3-3, 4-4-2, etc.) from player positions over time',
            color: 'cyan',
          },
          {
            icon: <Shield className="w-4 h-4" />,
            title: 'Defensive Metrics',
            desc: 'PPDA, high press success rate, recoveries, and defensive action zones',
            color: 'emerald',
          },
        ].map((item, i) => {
          const colorMap: Record<string, string> = {
            violet: 'bg-violet-500/10 text-violet-400 border-violet-500/20',
            amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
            cyan: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
            emerald: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
          }
          return (
            <div
              key={i}
              className="flex items-start gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/[0.04] hover:border-white/[0.08] hover:bg-white/[0.04] transition-all duration-300 group/item"
            >
              <div
                className={`w-8 h-8 rounded-lg flex items-center justify-center border shrink-0 ${colorMap[item.color]} group-hover/item:scale-110 transition-transform duration-300`}
              >
                {item.icon}
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <h4 className="text-xs font-semibold">{item.title}</h4>
                  <Lock className="w-3 h-3 text-muted-foreground/50" />
                </div>
                <p className="text-[11px] text-muted-foreground mt-0.5 leading-relaxed">
                  {item.desc}
                </p>
              </div>
            </div>
          )
        })}
      </CardContent>
    </motion.div>
  )
}

// ==================== Ball Trajectory Diagram ====================
const TRAIL_LENGTH = 150
const TRAIL_SEGMENTS = 5

// Opacity and stroke per fade segment (0=oldest, 4=newest)
const SEGMENT_STYLE = [
  { opacity: 0.08, strokeWidth: 0.3, color: 'rgba(255,255,255,0.15)' },
  { opacity: 0.18, strokeWidth: 0.4, color: 'rgba(255,255,255,0.40)' },
  { opacity: 0.35, strokeWidth: 0.5, color: 'rgba(255,200,80,0.65)'  },
  { opacity: 0.60, strokeWidth: 0.6, color: 'rgba(251,191,36,0.85)'  },
  { opacity: 0.95, strokeWidth: 0.9, color: 'rgba(251,191,36,1)'     },
]

export function BallTrajectoryDiagram({
  ballTrajectoryPoints,
  currentFrame,
  minFrame,
  maxFrame,
}: {
  ballTrajectoryPoints?: { x: number; y: number }[]
  currentFrame?: number
  minFrame?: number
  maxFrame?: number
}) {
  const hasRealData = !!ballTrajectoryPoints && ballTrajectoryPoints.length >= 2

  const { ghostPath, segments, ballHead } = useMemo(() => {
    if (!hasRealData || !ballTrajectoryPoints) {
      return { ghostPath: '', segments: [], ballHead: null }
    }

    const pts = ballTrajectoryPoints
    const total = pts.length

    // Map current frame → array index
    let endIdx = total - 1
    if (
      currentFrame !== undefined &&
      minFrame !== undefined &&
      maxFrame !== undefined &&
      maxFrame > minFrame
    ) {
      const progress = (currentFrame - minFrame) / (maxFrame - minFrame)
      endIdx = Math.min(total - 1, Math.floor(progress * total))
    }

    const startIdx = Math.max(0, endIdx - TRAIL_LENGTH)

    // Ghost: all points before the window (very faint historical context)
    const ghostPoints = pts.slice(0, startIdx + 1)
    const ghostPath = ghostPoints.length >= 2
      ? catmullRomToSvgPath(rdpSimplify(ghostPoints, 1.5), 0.4)
      : ''

    // Active trail: split into TRAIL_SEGMENTS fade segments
    const windowPts = pts.slice(startIdx, endIdx + 1)
    const segSize = Math.ceil(windowPts.length / TRAIL_SEGMENTS)
    const segments = SEGMENT_STYLE.map((style, i) => {
      const segStart = i * segSize
      const segEnd = Math.min(segStart + segSize + 1, windowPts.length) // +1 overlap for continuity
      const segPts = windowPts.slice(segStart, segEnd)
      if (segPts.length < 2) return null
      const simplified = rdpSimplify(segPts, 0.3)
      return { ...style, pathD: catmullRomToSvgPath(simplified, 0.4) }
    }).filter(Boolean)

    // Ball head: current position
    const ballHead = endIdx >= 0 ? pts[endIdx] : null

    return { ghostPath, segments, ballHead }
  }, [hasRealData, ballTrajectoryPoints, currentFrame, minFrame, maxFrame])

  return (
    <motion.div
      whileHover={{ y: -6, transition: { duration: 0.3, ease: [0.22, 1, 0.36, 1] } }}
      className="glass-card overflow-hidden group relative"
    >
      <div className="absolute -top-10 -left-10 w-32 h-32 rounded-full bg-amber-500/5 blur-3xl group-hover:bg-amber-500/10 transition-colors duration-700" />
      <CardHeader className="pb-4 pt-5 px-5 border-b border-border/30 relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
              <Move className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <CardTitle className="text-base font-black uppercase tracking-tight" style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}>Ball Trajectory</CardTitle>
              <CardDescription className="text-xs mt-0.5">
                Movement path across the pitch
              </CardDescription>
            </div>
          </div>
          {!hasRealData && (
            <Badge
              variant="outline"
              className="text-[10px] border-white/10 text-white/30 bg-white/5"
            >
              No Data
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-4">
        <div className="pitch-container">
          {!hasRealData ? (
            <div className="absolute inset-0 flex items-center justify-center z-10">
              <div className="text-center">
                <div className="text-white/20 text-xs">No trajectory data available</div>
                <div className="text-white/10 text-[10px] mt-1">Complete an analysis to see ball movement</div>
              </div>
            </div>
          ) : null}
          <svg
            className="absolute inset-0 w-full h-full"
            viewBox="0 0 105 68"
            preserveAspectRatio="xMidYMid meet"
            style={{ zIndex: 2 }}
          >
            <defs>
              <filter id="trajGlow">
                <feGaussianBlur stdDeviation="0.8" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="trajLineGlow">
                <feGaussianBlur stdDeviation="0.4" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="ballHeadGlow">
                <feGaussianBlur stdDeviation="1.2" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            {/* Pitch lines */}
            <g
              filter="url(#trajLineGlow)"
              stroke="rgba(52,211,153,0.2)"
              strokeWidth="0.3"
              fill="none"
            >
              <rect x="0.5" y="0.5" width="104" height="67" rx="0.5" />
              <line x1="52.5" y1="0.5" x2="52.5" y2="67.5" />
              <circle cx="52.5" cy="34" r="9.15" />
              <circle cx="52.5" cy="34" r="0.5" fill="rgba(52,211,153,0.3)" stroke="none" />
              <rect x="0.5" y="13.84" width="16.5" height="40.32" />
              <rect x="0.5" y="24.84" width="5.5" height="18.32" />
              <rect x="88" y="13.84" width="16.5" height="40.32" />
              <rect x="99" y="24.84" width="5.5" height="18.32" />
            </g>
            {hasRealData && (
              <>
                {/* Ghost trail — all historical movement before the window */}
                {ghostPath && (
                  <path
                    d={ghostPath}
                    fill="none"
                    stroke="rgba(251,191,36,0.03)"
                    strokeWidth="0.2"
                    strokeLinecap="round"
                  />
                )}
                {/* Active trailing window — 5 fade segments */}
                {segments.map((seg, i) => seg && (
                  <path
                    key={i}
                    d={seg.pathD}
                    fill="none"
                    stroke={seg.color}
                    strokeWidth={seg.strokeWidth}
                    strokeOpacity={seg.opacity}
                    strokeLinecap="round"
                    filter={i === segments.length - 1 ? 'url(#trajGlow)' : undefined}
                  />
                ))}
                {/* Ball head — current position with pulse */}
                {ballHead && (
                  <>
                    <circle
                      cx={ballHead.x}
                      cy={ballHead.y}
                      r="2.5"
                      fill="rgba(251,191,36,0.15)"
                      filter="url(#ballHeadGlow)"
                    />
                    <circle
                      cx={ballHead.x}
                      cy={ballHead.y}
                      r="1.4"
                      fill="rgba(251,191,36,0.9)"
                      stroke="white"
                      strokeWidth="0.4"
                    >
                      <animate attributeName="r" values="1.5;2.8;1.5" dur="1.5s" repeatCount="indefinite" />
                      <animate attributeName="stroke-opacity" values="0.9;0.3;0.9" dur="1.5s" repeatCount="indefinite" />
                    </circle>
                  </>
                )}
              </>
            )}
          </svg>
        </div>
      </CardContent>
    </motion.div>
  )
}

// ==================== Player Interaction Graph ====================
export function PlayerInteractionGraph({
  filterTeam,
  interactionTeam1,
  interactionTeam2,
}: {
  filterTeam: 'all' | 'team1' | 'team2'
  interactionTeam1?: InteractionGraphData | null
  interactionTeam2?: InteractionGraphData | null
}) {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors()
  const hasRealData = !!(interactionTeam1?.nodes?.length || interactionTeam2?.nodes?.length)
  const [activeSegment, setActiveSegment] = useState<number>(-1) // -1 = Full Match
  const [graphView, setGraphView] = useState<'all' | 'team1' | 'team2'>(filterTeam)

  // Merge timeline segments from both team graphs
  const timelineSegments = useMemo(() => {
    const t1 = interactionTeam1?.timeline
    const t2 = interactionTeam2?.timeline
    if (!t1?.length && !t2?.length) return null
    return t1?.length ? t1 : (t2 ?? null)
  }, [interactionTeam1, interactionTeam2])

  const hasTimeline = !!timelineSegments?.length

  // Merge both teams' data — pass-based network (no ball node)
  const { allNodes, allEdges } = useMemo(() => {
    type NodeT = {
      id: number
      teamId: number
      x: number
      y: number
      speed: number
      passCount: number
      betweenness?: number
      clustering?: number
      degreeCentrality?: number
    }
    type EdgeT = { from: number; to: number; weight: number; teamId: number; passCount: number }

    if (hasRealData) {
      const nodes: NodeT[] = []
      const edges: EdgeT[] = []
      const seenIds = new Set<number>()

      for (const graph of [interactionTeam1, interactionTeam2]) {
        if (!graph) continue
        for (const n of graph.nodes) {
          if (n.playerId === -1 || n.isBall) continue
          if (seenIds.has(n.playerId)) continue
          seenIds.add(n.playerId)
          nodes.push({
            id: n.playerId,
            teamId: n.teamId,
            x: n.avgX / 100,
            y: n.avgY / 100,
            speed: n.avgSpeed,
            passCount: n.passCount ?? 0,
            betweenness: n.betweenness,
            clustering: n.clustering,
            degreeCentrality: n.degreeCentrality,
          })
        }
        for (const e of graph.edges) {
          if (e.isBallEdge) continue
          if (e.weight < 0.2) continue
          const fromNode = graph.nodes.find((n) => n.playerId === e.from)
          edges.push({
            from: e.from,
            to: e.to,
            weight: e.weight,
            teamId: fromNode?.teamId ?? 1,
            passCount: e.passCount ?? 0,
          })
        }
      }

      return { allNodes: nodes, allEdges: edges }
    }

    // Demo data fallback
    const demoNodes: NodeT[] = [
      { id: 1, teamId: 1, x: 15, y: 34, speed: 7.2, passCount: 8 },
      { id: 2, teamId: 1, x: 28, y: 14, speed: 8.1, passCount: 12 },
      { id: 3, teamId: 1, x: 28, y: 54, speed: 7.8, passCount: 10 },
      { id: 4, teamId: 1, x: 42, y: 24, speed: 9.3, passCount: 18 },
      { id: 5, teamId: 1, x: 42, y: 44, speed: 8.5, passCount: 14 },
      { id: 6, teamId: 1, x: 50, y: 34, speed: 6.9, passCount: 6 },
      { id: 12, teamId: 2, x: 58, y: 34, speed: 7.5, passCount: 9 },
      { id: 13, teamId: 2, x: 70, y: 14, speed: 8.0, passCount: 15 },
      { id: 14, teamId: 2, x: 70, y: 54, speed: 7.4, passCount: 11 },
      { id: 15, teamId: 2, x: 82, y: 24, speed: 9.1, passCount: 16 },
      { id: 16, teamId: 2, x: 82, y: 44, speed: 8.8, passCount: 13 },
      { id: 17, teamId: 2, x: 92, y: 34, speed: 6.5, passCount: 5 },
    ]
    const demoEdges: EdgeT[] = [
      { from: 1, to: 2, weight: 0.7, teamId: 1, passCount: 5 },
      { from: 1, to: 3, weight: 0.8, teamId: 1, passCount: 6 },
      { from: 2, to: 4, weight: 1.0, teamId: 1, passCount: 8 },
      { from: 3, to: 5, weight: 0.75, teamId: 1, passCount: 5 },
      { from: 4, to: 6, weight: 0.6, teamId: 1, passCount: 4 },
      { from: 5, to: 6, weight: 0.5, teamId: 1, passCount: 3 },
      { from: 12, to: 13, weight: 0.65, teamId: 2, passCount: 4 },
      { from: 12, to: 14, weight: 0.85, teamId: 2, passCount: 6 },
      { from: 13, to: 15, weight: 0.9, teamId: 2, passCount: 7 },
      { from: 14, to: 16, weight: 0.7, teamId: 2, passCount: 5 },
      { from: 15, to: 17, weight: 0.55, teamId: 2, passCount: 3 },
      { from: 16, to: 17, weight: 0.45, teamId: 2, passCount: 2 },
    ]
    return { allNodes: demoNodes, allEdges: demoEdges }
  }, [hasRealData, interactionTeam1, interactionTeam2])

  // Build a lookup of segment edges for the active segment
  const segmentEdgeLookup = useMemo(() => {
    if (activeSegment < 0 || !hasTimeline) return null
    const lookup: Record<string, number> = {}
    for (const graph of [interactionTeam1, interactionTeam2]) {
      if (!graph?.timeline?.[activeSegment]) continue
      for (const e of graph.timeline[activeSegment].edges) {
        lookup[`${e.from}-${e.to}`] = e.weight
      }
    }
    return lookup
  }, [activeSegment, hasTimeline, interactionTeam1, interactionTeam2])

  const isNodeVisible = (node: { teamId: number }) =>
    graphView === 'all' || (graphView === 'team1' ? node.teamId === 1 : node.teamId === 2)

  const isEdgeVisible = (edge: { from: number; to: number }) => {
    const fromNode = allNodes.find((n) => n.id === edge.from)
    const toNode = allNodes.find((n) => n.id === edge.to)
    if (!fromNode || !toNode) return false
    return isNodeVisible(fromNode) && isNodeVisible(toNode)
  }

  // Single-pass node repulsion to reduce overlap
  const nudgedNodes = useMemo(() => {
    const nodes = allNodes.map(n => ({ ...n }))
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x
        const dy = nodes[j].y - nodes[i].y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 6 && dist > 0) {
          const push = (6 - dist) / 2
          const nx = dx / dist
          const ny = dy / dist
          nodes[i].x = Math.max(2, Math.min(103, nodes[i].x - nx * push))
          nodes[i].y = Math.max(2, Math.min(66, nodes[i].y - ny * push))
          nodes[j].x = Math.max(2, Math.min(103, nodes[j].x + nx * push))
          nodes[j].y = Math.max(2, Math.min(66, nodes[j].y + ny * push))
        }
      }
    }
    return nodes
  }, [allNodes])

  // Normalize edge weights to prevent absurdly thick lines
  const maxWeight = useMemo(() => Math.max(...allEdges.map(e => e.weight), 1), [allEdges])

  // Segment-aware node pass counts (for node resizing when period is selected)
  const segmentNodePassCounts = useMemo(() => {
    if (!segmentEdgeLookup) return null
    const counts: Record<number, number> = {}
    for (const [key, w] of Object.entries(segmentEdgeLookup)) {
      const [fromStr, toStr] = key.split('-')
      const from = parseInt(fromStr)
      const to = parseInt(toStr)
      counts[from] = (counts[from] ?? 0) + w
      counts[to] = (counts[to] ?? 0) + w
    }
    return counts
  }, [segmentEdgeLookup])

  // Node sizing: scale by pass involvement (segment-aware when period selected)
  const maxPassCount = useMemo(() => {
    if (segmentNodePassCounts) {
      return Math.max(...Object.values(segmentNodePassCounts), 1)
    }
    return Math.max(...allNodes.map((n) => n.passCount), 1)
  }, [allNodes, segmentNodePassCounts])

  const getNodeSize = (nodeId: number, passCount: number) => {
    const effectiveCount = segmentNodePassCounts ? (segmentNodePassCounts[nodeId] ?? 0) : passCount
    return 20 + (effectiveCount / maxPassCount) * 16
  }

  // Dijkstra's key passing route: from leftmost to rightmost player (dominant build-up path)
  const keyRoute = useMemo(() => {
    const visibleNodes = nudgedNodes.filter(n =>
      graphView === 'all' || (graphView === 'team1' ? n.teamId === 1 : n.teamId === 2)
    )
    if (visibleNodes.length < 2) return new Set<string>()
    const sorted = [...visibleNodes].sort((a, b) => a.x - b.x)
    const startId = sorted[0].id
    const endId = sorted[sorted.length - 1].id
    const visibleEdges = allEdges.filter(e => {
      const fn = nudgedNodes.find(n => n.id === e.from)
      const tn = nudgedNodes.find(n => n.id === e.to)
      return fn && tn && isEdgeVisible(e)
    })
    const path = dijkstraPassRoute(visibleNodes, visibleEdges, startId, endId)
    const routeSet = new Set<string>()
    for (let i = 0; i < path.length - 1; i++) {
      routeSet.add(`${path[i]}-${path[i + 1]}`)
      routeSet.add(`${path[i + 1]}-${path[i]}`) // undirected
    }
    return routeSet
  }, [nudgedNodes, allEdges, graphView])

  return (
    <motion.div
      whileHover={{ y: -6, transition: { duration: 0.3, ease: [0.22, 1, 0.36, 1] } }}
      className="glass-card overflow-hidden group relative"
    >
      <div className="absolute -top-10 -right-10 w-32 h-32 rounded-full bg-cyan-500/5 blur-3xl group-hover:bg-cyan-500/10 transition-colors duration-700" />
      <CardHeader className="pb-3 pt-5 px-5 border-b border-border/30 relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-500/10 flex items-center justify-center border border-cyan-500/20">
              <GitBranch className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <CardTitle className="text-base font-black uppercase tracking-tight" style={{ fontFamily: 'Barlow Condensed, Inter, sans-serif' }}>Passing Network</CardTitle>
              <CardDescription className="text-xs mt-0.5">
                Pass connections between players
              </CardDescription>
            </div>
          </div>
          {!hasRealData && (
            <Badge
              variant="outline"
              className="text-[10px] border-cyan-400/20 text-cyan-400/70 bg-cyan-500/5"
            >
              Placeholder
            </Badge>
          )}
        </div>
        {/* Team filter buttons */}
        <div className="flex items-center gap-1.5 mt-3">
          {(['all', 'team1', 'team2'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setGraphView(t)}
              className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-all ${
                graphView === t
                  ? t === 'team1'
                    ? 'bg-red-500/15 text-red-400 border border-red-500/25'
                    : t === 'team2'
                      ? 'bg-blue-500/15 text-blue-400 border border-blue-500/25'
                      : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/25'
                  : 'text-muted-foreground hover:text-foreground hover:bg-white/5 border border-transparent'
              }`}
            >
              <span className="flex items-center gap-1.5">
                {t !== 'all' && (
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ backgroundColor: t === 'team1' ? TEAM1_HEX : TEAM2_HEX }}
                  />
                )}
                {t === 'all' ? 'Combined' : t === 'team1' ? 'Team 1' : 'Team 2'}
              </span>
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="p-4">
        <div className="pitch-container">
          <svg
            className="absolute inset-0 w-full h-full"
            viewBox="0 0 105 68"
            preserveAspectRatio="xMidYMid meet"
            style={{ zIndex: 2 }}
          >
            <defs>
              {/* Small subtle arrowheads — scaled proportionally to thin edges */}
              <marker id="arr-t1" markerWidth="3" markerHeight="2.5" refX="2.8" refY="1.25" orient="auto" markerUnits="strokeWidth">
                <polygon points="0 0, 3 1.25, 0 2.5" fill={TEAM1_HEX} fillOpacity="0.7" />
              </marker>
              <marker id="arr-t2" markerWidth="3" markerHeight="2.5" refX="2.8" refY="1.25" orient="auto" markerUnits="strokeWidth">
                <polygon points="0 0, 3 1.25, 0 2.5" fill={TEAM2_HEX} fillOpacity="0.7" />
              </marker>
              {/* Key route markers — brighter */}
              <marker id="arr-route" markerWidth="3" markerHeight="2.5" refX="2.8" refY="1.25" orient="auto" markerUnits="strokeWidth">
                <polygon points="0 0, 3 1.25, 0 2.5" fill="white" fillOpacity="0.9" />
              </marker>
              <filter id="routeGlow">
                <feGaussianBlur stdDeviation="0.5" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>
            {/* Pitch lines — subtle, recede behind graph */}
            <g stroke="rgba(52,211,153,0.15)" strokeWidth="0.2" fill="none">
              <rect x="0.5" y="0.5" width="104" height="67" rx="0.5" />
              <line x1="52.5" y1="0.5" x2="52.5" y2="67.5" />
              <circle cx="52.5" cy="34" r="9.15" />
              <rect x="0.5" y="13.84" width="16.5" height="40.32" />
              <rect x="88" y="13.84" width="16.5" height="40.32" />
            </g>
            {/* Edges — curved quadratic Bezier, clamped thickness, animated transitions */}
            <g>
              {allEdges.map((edge, i) => {
                const from = nudgedNodes.find((n) => n.id === edge.from)
                const to = nudgedNodes.find((n) => n.id === edge.to)
                if (!from || !to) return null
                const visible = isEdgeVisible(edge)

                let active = true
                if (segmentEdgeLookup) {
                  active = (segmentEdgeLookup[`${edge.from}-${edge.to}`] ?? 0) > 0
                }

                // Curved quadratic Bezier with perpendicular offset
                const mx = (from.x + to.x) / 2
                const my = (from.y + to.y) / 2
                const dx = to.x - from.x
                const dy = to.y - from.y
                const len = Math.sqrt(dx * dx + dy * dy) || 1
                const offset = Math.min(3, len * 0.15)
                const cpx = mx - (dy / len) * offset
                const cpy = my + (dx / len) * offset
                const d = `M ${from.x} ${from.y} Q ${cpx} ${cpy} ${to.x} ${to.y}`

                // Normalized + clamped thickness
                const normalizedW = edge.weight / maxWeight
                const baseWidth = Math.max(0.15, Math.min(1.2, normalizedW * 1.2))
                const isKeyRoute = keyRoute.has(`${edge.from}-${edge.to}`)
                const strokeWidth = isKeyRoute ? Math.max(0.6, baseWidth * 1.5) : baseWidth
                const opacity = visible ? (active ? (isKeyRoute ? 0.85 : 0.35) : 0.04) : 0
                const delay = Math.min(i * 15, 300)

                const color = isKeyRoute ? 'white' : (edge.teamId === 1 ? TEAM1_HEX : TEAM2_HEX)
                const arrowId = isKeyRoute ? 'url(#arr-route)' : (edge.teamId === 1 ? 'url(#arr-t1)' : 'url(#arr-t2)')

                return (
                  <path
                    key={`${edge.from}-${edge.to}`}
                    d={d}
                    fill="none"
                    className="interaction-edge"
                    stroke={color}
                    strokeWidth={strokeWidth}
                    strokeOpacity={opacity}
                    strokeLinecap="round"
                    markerEnd={visible && active ? arrowId : undefined}
                    filter={isKeyRoute && visible && active ? 'url(#routeGlow)' : undefined}
                    style={{ transitionDelay: `${delay}ms` }}
                  />
                )
              })}
            </g>
          </svg>
          {/* Player nodes — dynamic size by pass involvement */}
          {nudgedNodes.map((node) => {
            const visible = isNodeVisible(node)
            const teamLabel = node.teamId === 1 ? 'Team 1' : 'Team 2'
            const nodeSize = getNodeSize(node.id, node.passCount)
            return (
              <RadixTooltip key={node.id}>
                <TooltipTrigger asChild>
                  <div
                    className="player-node-interaction"
                    style={
                      {
                        left: `${(node.x / PITCH_WIDTH) * 100}%`,
                        top: `${(node.y / PITCH_HEIGHT) * 100}%`,
                        '--node-color': node.teamId === 1 ? TEAM1_HEX : TEAM2_HEX,
                        '--node-size': `${nodeSize}px`,
                        opacity: visible ? 1 : 0,
                        transform: `translate(-50%, -50%) scale(${visible ? 1 : 0.5})`,
                        pointerEvents: visible ? 'auto' : 'none',
                      } as React.CSSProperties
                    }
                  >
                    <span className="text-[9px] font-bold text-white drop-shadow-md leading-none">
                      {node.id}
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="top" className="text-xs">
                  <div>
                    <span className="font-semibold">#{node.id}</span> {teamLabel}
                    {hasRealData
                      ? ` · ${node.speed.toFixed(1)} km/h · ${node.passCount} passes`
                      : ''}
                  </div>
                  {node.betweenness !== undefined && (
                    <div className="text-muted-foreground mt-0.5">
                      Betw: {Math.round(node.betweenness * 100)}% · Clust:{' '}
                      {Math.round((node.clustering ?? 0) * 100)}% · Deg:{' '}
                      {Math.round((node.degreeCentrality ?? 0) * 100)}%
                    </div>
                  )}
                </TooltipContent>
              </RadixTooltip>
            )
          })}
        </div>
        {/* Legend */}
        <div className="flex items-center gap-4 mt-2 px-1">
          <div className="flex items-center gap-1.5 text-[9px] text-muted-foreground">
            <svg width="24" height="12" viewBox="0 0 24 12">
              <circle cx="4" cy="6" r="3" fill="currentColor" opacity="0.4" />
              <circle cx="16" cy="6" r="5" fill="currentColor" opacity="0.4" />
            </svg>
            <span>Node size = pass involvement</span>
          </div>
          <div className="flex items-center gap-1.5 text-[9px] text-muted-foreground">
            <svg width="24" height="12" viewBox="0 0 24 12">
              <line
                x1="0"
                y1="9"
                x2="10"
                y2="9"
                stroke="currentColor"
                strokeWidth="1"
                opacity="0.4"
              />
              <line
                x1="14"
                y1="5"
                x2="24"
                y2="5"
                stroke="currentColor"
                strokeWidth="3"
                opacity="0.4"
              />
            </svg>
            <span>Line thickness = passes between pair</span>
          </div>
          {keyRoute.size > 0 && (
            <div className="flex items-center gap-1.5 text-[9px] text-white/50">
              <svg width="18" height="12" viewBox="0 0 18 12">
                <line x1="0" y1="6" x2="18" y2="6" stroke="white" strokeWidth="1.5" opacity="0.7" />
              </svg>
              <span>Key route</span>
            </div>
          )}
        </div>
        {/* Timeline segment scrubber */}
        {hasTimeline && (
          <div className="flex items-center gap-1.5 mt-3 px-1 flex-wrap">
            <span className="text-[9px] uppercase tracking-widest text-muted-foreground font-semibold mr-1">
              Period
            </span>
            <button
              onClick={() => setActiveSegment(-1)}
              className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-all ${
                activeSegment === -1
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'text-muted-foreground hover:text-foreground hover:bg-white/5 border border-transparent'
              }`}
            >
              Full
            </button>
            {timelineSegments!.map((seg, i) => (
              <button
                key={i}
                onClick={() => setActiveSegment(i)}
                className={`px-2.5 py-1 rounded-md text-[10px] font-mono font-medium transition-all ${
                  activeSegment === i
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                    : 'text-muted-foreground hover:text-foreground hover:bg-white/5 border border-transparent'
                }`}
              >
                {seg.label}
              </button>
            ))}
          </div>
        )}
      </CardContent>
    </motion.div>
  )
}
