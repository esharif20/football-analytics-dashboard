import { useMemo } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Radar, Target, Users, Zap, Map } from 'lucide-react'
import { useTeamColors, PITCH_WIDTH, PITCH_HEIGHT, EVENT_TYPES } from '../context'
import type { PipelineMode } from '../context'

export function PitchRadar({
  data,
  selectedPlayer,
}: {
  data: any
  selectedPlayer?: number | null
}) {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors()
  if (!data)
    return (
      <div className="pitch-container flex items-center justify-center">
        <p className="text-muted-foreground">No tracking data available</p>
      </div>
    )
  return (
    <div className="pitch-container">
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 105 68"
        preserveAspectRatio="xMidYMid meet"
        style={{ zIndex: 2 }}
      >
        <defs>
          <linearGradient id="pitchGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(255,255,255,0.03)" />
            <stop offset="50%" stopColor="rgba(255,255,255,0)" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.05)" />
          </linearGradient>
          <filter id="lineGlow">
            <feGaussianBlur stdDeviation="0.4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="centerGlow">
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <radialGradient id="vignette" cx="50%" cy="50%" r="60%">
            <stop offset="0%" stopColor="transparent" />
            <stop offset="100%" stopColor="rgba(0,0,0,0.35)" />
          </radialGradient>
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
      <div
        className="absolute top-3 right-3 bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2 text-[11px] font-medium"
        style={{ zIndex: 20 }}
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: TEAM1_HEX }} />
            <span className="text-white/70">Team 1</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: TEAM2_HEX }} />
            <span className="text-white/70">Team 2</span>
          </div>
        </div>
      </div>
      {/* Players */}
      {data.team1Players.map((player: any) => (
        <PlayerNode
          key={player.id}
          player={player}
          teamId={1}
          isSelected={selectedPlayer === player.trackId}
        />
      ))}
      {data.team2Players.map((player: any) => (
        <PlayerNode
          key={player.id}
          player={player}
          teamId={2}
          isSelected={selectedPlayer === player.trackId}
        />
      ))}
      <div
        className="ball-marker"
        style={{
          left: `${(data.ball.x / PITCH_WIDTH) * 100}%`,
          top: `${(data.ball.y / PITCH_HEIGHT) * 100}%`,
        }}
      />
    </div>
  )
}

// Premium Player Node
export function PlayerNode({
  player,
  teamId,
  isSelected,
}: {
  player: any
  teamId: number
  isSelected?: boolean
}) {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors()
  const color = teamId === 1 ? TEAM1_HEX : TEAM2_HEX
  return (
    <div
      className={`player-node ${isSelected ? 'player-node-selected' : ''}`}
      style={
        {
          left: `${(player.x / PITCH_WIDTH) * 100}%`,
          top: `${(player.y / PITCH_HEIGHT) * 100}%`,
          '--player-color': color,
        } as React.CSSProperties
      }
    >
      <span className="player-node-label">{player.trackId}</span>
    </div>
  )
}

export function HeatmapView() {
  return (
    <div className="pitch-container">
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 105 68"
        preserveAspectRatio="xMidYMid meet"
        style={{ zIndex: 2 }}
      >
        <defs>
          <filter id="heatLineGlow">
            <feGaussianBlur stdDeviation="0.3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <radialGradient id="hz1">
            <stop offset="0%" stopColor="oklch(0.55 0.22 25)" stopOpacity="0.85" />
            <stop offset="100%" stopColor="oklch(0.55 0.2 25)" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="hz2">
            <stop offset="0%" stopColor="oklch(0.65 0.22 45)" stopOpacity="0.7" />
            <stop offset="100%" stopColor="oklch(0.65 0.2 45)" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="hz3">
            <stop offset="0%" stopColor="oklch(0.6 0.2 145)" stopOpacity="0.5" />
            <stop offset="100%" stopColor="oklch(0.6 0.2 145)" stopOpacity="0" />
          </radialGradient>
          <filter id="heatBlur">
            <feGaussianBlur stdDeviation="3" />
          </filter>
        </defs>
        <g
          filter="url(#heatLineGlow)"
          stroke="rgba(255,255,255,0.35)"
          strokeWidth="0.35"
          fill="none"
        >
          <rect x="0.5" y="0.5" width="104" height="67" rx="0.5" />
          <line x1="52.5" y1="0.5" x2="52.5" y2="67.5" />
          <circle cx="52.5" cy="34" r="9.15" />
          <circle cx="52.5" cy="34" r="0.5" fill="rgba(255,255,255,0.4)" stroke="none" />
          <rect x="0.5" y="13.84" width="16.5" height="40.32" />
          <rect x="0.5" y="24.84" width="5.5" height="18.32" />
          <rect x="88" y="13.84" width="16.5" height="40.32" />
          <rect x="99" y="24.84" width="5.5" height="18.32" />
        </g>
        <g filter="url(#heatBlur)" style={{ mixBlendMode: 'screen' }}>
          <ellipse cx="28" cy="30" rx="14" ry="18" fill="url(#hz1)" />
          <ellipse cx="52.5" cy="34" rx="18" ry="14" fill="url(#hz2)" />
          <ellipse cx="77" cy="38" rx="14" ry="18" fill="url(#hz1)" />
          <ellipse cx="40" cy="20" rx="10" ry="12" fill="url(#hz3)" />
          <ellipse cx="65" cy="48" rx="10" ry="12" fill="url(#hz3)" />
        </g>
      </svg>
      <div
        className="absolute bottom-3 right-3 bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2 text-[11px] font-medium"
        style={{ zIndex: 20 }}
      >
        <div className="flex items-center gap-2">
          <div className="w-20 h-2 rounded-full heatmap-gradient" />
          <span className="text-white/70">Low → High</span>
        </div>
      </div>
    </div>
  )
}

export function PassNetworkView() {
  const { TEAM2_HEX } = useTeamColors()
  const nodes = [
    { id: 1, x: 15, y: 34, passes: 45 },
    { id: 2, x: 30, y: 15, passes: 38 },
    { id: 3, x: 30, y: 53, passes: 42 },
    { id: 4, x: 45, y: 25, passes: 52 },
    { id: 5, x: 45, y: 43, passes: 48 },
    { id: 6, x: 60, y: 34, passes: 35 },
  ]
  const edges = [
    { from: 1, to: 2, count: 12 },
    { from: 1, to: 3, count: 15 },
    { from: 2, to: 4, count: 18 },
    { from: 3, to: 5, count: 14 },
    { from: 4, to: 6, count: 10 },
    { from: 5, to: 6, count: 8 },
    { from: 4, to: 5, count: 6 },
  ]
  return (
    <div className="pitch-container">
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 105 68"
        preserveAspectRatio="xMidYMid meet"
        style={{ zIndex: 2 }}
      >
        <defs>
          <filter id="passLineGlow">
            <feGaussianBlur stdDeviation="0.3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="oklch(0.7 0.18 145)" fillOpacity="0.7" />
          </marker>
          <filter id="edgeGlow">
            <feGaussianBlur stdDeviation="0.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <g
          filter="url(#passLineGlow)"
          stroke="rgba(255,255,255,0.35)"
          strokeWidth="0.35"
          fill="none"
        >
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
            const fromNode = nodes.find((n) => n.id === edge.from)!
            const toNode = nodes.find((n) => n.id === edge.to)!
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
            )
          })}
        </g>
      </svg>
      {nodes.map((node) => (
        <div
          key={node.id}
          className="player-node"
          style={
            {
              left: `${(node.x / PITCH_WIDTH) * 100}%`,
              top: `${(node.y / PITCH_HEIGHT) * 100}%`,
              '--player-color': TEAM2_HEX,
            } as React.CSSProperties
          }
        >
          <span className="player-node-label">{node.id}</span>
        </div>
      ))}
    </div>
  )
}

export function VoronoiView({ data }: { data: any }) {
  const { TEAM1_HEX, TEAM2_HEX } = useTeamColors()
  const voronoiCells = useMemo(() => {
    if (!data) return []
    const allPlayers = [...data.team1Players, ...data.team2Players]
    const cells: { playerId: number; teamId: number; path: string }[] = []
    const gridSize = 1
    const grid: { x: number; y: number; playerId: number; teamId: number }[][] = []
    for (let x = 0; x <= PITCH_WIDTH; x += gridSize) {
      const row: { x: number; y: number; playerId: number; teamId: number }[] = []
      for (let y = 0; y <= PITCH_HEIGHT; y += gridSize) {
        let minDist = Infinity
        let closestPlayer = allPlayers[0]
        for (const player of allPlayers) {
          const dist = Math.sqrt((x - player.x) ** 2 + (y - player.y) ** 2)
          if (dist < minDist) {
            minDist = dist
            closestPlayer = player
          }
        }
        row.push({ x, y, playerId: closestPlayer.id, teamId: closestPlayer.teamId })
      }
      grid.push(row)
    }
    for (const player of allPlayers) {
      const points: [number, number][] = []
      for (let x = 0; x < grid.length; x++) {
        for (let y = 0; y < grid[x].length; y++) {
          if (grid[x][y].playerId === player.id) {
            const isBoundary =
              x === 0 ||
              x === grid.length - 1 ||
              y === 0 ||
              y === grid[x].length - 1 ||
              grid[x - 1]?.[y]?.playerId !== player.id ||
              grid[x + 1]?.[y]?.playerId !== player.id ||
              grid[x]?.[y - 1]?.playerId !== player.id ||
              grid[x]?.[y + 1]?.playerId !== player.id
            if (isBoundary) points.push([grid[x][y].x, grid[x][y].y])
          }
        }
      }
      if (points.length > 2) {
        const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length
        const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length
        points.sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx))
        cells.push({
          playerId: player.id,
          teamId: player.teamId,
          path: `M ${points.map((p) => `${p[0]} ${p[1]}`).join(' L ')} Z`,
        })
      }
    }
    return cells
  }, [data])
  if (!data)
    return (
      <div className="pitch-container flex items-center justify-center">
        <p className="text-muted-foreground">No tracking data available</p>
      </div>
    )
  return (
    <div className="pitch-container">
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 105 68"
        preserveAspectRatio="xMidYMid meet"
        style={{ zIndex: 2 }}
      >
        <defs>
          <filter id="voronoiLineGlow">
            <feGaussianBlur stdDeviation="0.3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {voronoiCells.map((cell, i) => (
          <path
            key={i}
            d={cell.path}
            fill={cell.teamId === 1 ? TEAM1_HEX : TEAM2_HEX}
            fillOpacity={0.15}
            stroke={cell.teamId === 1 ? TEAM1_HEX : TEAM2_HEX}
            strokeWidth={0.25}
            strokeOpacity={0.4}
          />
        ))}
        <g
          filter="url(#voronoiLineGlow)"
          stroke="rgba(255,255,255,0.45)"
          strokeWidth="0.4"
          fill="none"
        >
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
      <div
        className="ball-marker"
        style={{
          left: `${(data.ball.x / PITCH_WIDTH) * 100}%`,
          top: `${(data.ball.y / PITCH_HEIGHT) * 100}%`,
        }}
      />
      <div
        className="absolute bottom-3 right-3 bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2 text-[11px] font-medium"
        style={{ zIndex: 20 }}
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: TEAM1_HEX, opacity: 0.5 }} />
            <span className="text-white/70">Team 1</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: TEAM2_HEX, opacity: 0.5 }} />
            <span className="text-white/70">Team 2</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// EventTimeline is imported from its own file in the parent — used inline here for ModeSpecificTabs
import { EventTimeline } from './EventTimeline'

// ==================== Mode-Specific Tabs ====================
export function ModeSpecificTabs({
  mode,
  activeTab,
  setActiveTab,
  trackingData,
  events,
  filterTeam,
  selectedPlayer,
  setSelectedPlayer,
}: {
  mode: PipelineMode
  activeTab: string
  setActiveTab: (t: string) => void
  trackingData: any
  events: any[]
  filterTeam: string
  selectedPlayer: number | null
  setSelectedPlayer: (id: number | null) => void
}) {
  const modeTabs: Record<PipelineMode, string[]> = {
    all: ['radar', 'voronoi', 'heatmap', 'passes', 'events'],
    radar: ['radar', 'voronoi'],
    team: ['radar', 'heatmap'],
    track: ['radar', 'events'],
    players: ['radar'],
    ball: ['radar', 'events'],
    pitch: ['radar', 'heatmap'],
  }
  const availableTabs = modeTabs[mode] || modeTabs.all
  const tabConfig: Record<string, { icon: React.ReactNode; label: string }> = {
    radar: { icon: <Radar className="w-4 h-4" />, label: 'Radar' },
    voronoi: { icon: <Users className="w-4 h-4" />, label: 'Voronoi' },
    heatmap: { icon: <Map className="w-4 h-4" />, label: 'Heatmap' },
    passes: { icon: <Target className="w-4 h-4" />, label: 'Passes' },
    events: { icon: <Zap className="w-4 h-4" />, label: 'Events' },
  }
  const effectiveTab = availableTabs.includes(activeTab) ? activeTab : availableTabs[0]
  return (
    <Tabs value={effectiveTab} onValueChange={setActiveTab}>
      <TabsList
        className="w-full bg-black/20 border border-white/[0.06] rounded-xl p-1 h-auto backdrop-blur-sm"
        style={{ gridTemplateColumns: `repeat(${availableTabs.length}, 1fr)`, display: 'grid' }}
      >
        {availableTabs.map((tab) => {
          const config = tabConfig[tab]
          return (
            <TabsTrigger
              key={tab}
              value={tab}
              className="gap-2 rounded-lg data-[state=active]:bg-emerald-500/15 data-[state=active]:text-emerald-400 data-[state=active]:shadow-[0_0_12px_rgba(52,211,153,0.1)] data-[state=active]:border data-[state=active]:border-emerald-500/20 data-[state=inactive]:text-muted-foreground py-2.5 transition-all duration-300"
            >
              {config.icon}
              <span className="hidden sm:inline text-xs font-medium">{config.label}</span>
            </TabsTrigger>
          )
        })}
      </TabsList>
      <TabsContent value="radar" className="mt-4">
        <PitchRadar data={trackingData} selectedPlayer={selectedPlayer} />
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
  )
}
