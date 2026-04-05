import { useMemo } from 'react'
import type { TrackRow } from './useAllTracks'

export interface MetricDataPoint {
  minute: number
  team1: number
  team2: number
}

export interface TrackMetrics {
  compactness: MetricDataPoint[]
  defensiveLine: MetricDataPoint[]
  pressing: MetricDataPoint[]
}

const WINDOW_SIZE = 30 // frames per window
const FPS = 25

function stdDev(values: number[]): number {
  if (values.length === 0) return 0
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

function computeCompactness(rows: TrackRow[]): { team1: number; team2: number } {
  const t1x: number[] = [],
    t1y: number[] = []
  const t2x: number[] = [],
    t2y: number[] = []

  for (const row of rows) {
    if (!row.playerPositions) continue
    for (const pos of Object.values(row.playerPositions)) {
      if (pos.teamId === 0) {
        t1x.push(pos.x)
        t1y.push(pos.y)
      } else {
        t2x.push(pos.x)
        t2y.push(pos.y)
      }
    }
  }

  // Spread = average of x-stddev and y-stddev, normalized to pitch scale
  const team1 = t1x.length > 1 ? (stdDev(t1x) / 105 + stdDev(t1y) / 68) * 50 : 0
  const team2 = t2x.length > 1 ? (stdDev(t2x) / 105 + stdDev(t2y) / 68) * 50 : 0

  return { team1, team2 }
}

function computeDefensiveLine(rows: TrackRow[]): { team1: number; team2: number } {
  // Defensive line = average x of the 4 deepest (lowest x for T1, highest x for T2) outfield players
  const t1xs: number[] = []
  const t2xs: number[] = []

  for (const row of rows) {
    if (!row.playerPositions) continue
    const t1 = Object.values(row.playerPositions)
      .filter((p) => p.teamId === 0 && !p.isGoalkeeper)
      .map((p) => p.x)
      .sort((a, b) => a - b) // ascending: lowest x = deepest for team 1
    const t2 = Object.values(row.playerPositions)
      .filter((p) => p.teamId === 1 && !p.isGoalkeeper)
      .map((p) => p.x)
      .sort((a, b) => b - a) // descending: highest x = deepest for team 2
    if (t1.length >= 2)
      t1xs.push(
        t1.slice(0, Math.min(4, t1.length)).reduce((a, b) => a + b, 0) / Math.min(4, t1.length)
      )
    if (t2.length >= 2)
      t2xs.push(
        t2.slice(0, Math.min(4, t2.length)).reduce((a, b) => a + b, 0) / Math.min(4, t2.length)
      )
  }

  const team1 = t1xs.length > 0 ? t1xs.reduce((a, b) => a + b, 0) / t1xs.length : 35
  // Team 2's defensive line in terms of distance from their own goal (x=105):
  // lower x = further from goal = defensive line pushed higher
  const team2Raw = t2xs.length > 0 ? t2xs.reduce((a, b) => a + b, 0) / t2xs.length : 70
  // Convert to consistent scale: distance from defending goal (metres)
  return { team1, team2: 105 - team2Raw }
}

const PRESSING_RADIUS_M = 15 // players within 15m of ball are considered pressing

function computePressing(rows: TrackRow[]): { team1: number; team2: number } {
  // Pressing intensity = weighted score from:
  // 1. Proximity: count of outfield players within PRESSING_RADIUS_M of ball
  // 2. Velocity toward ball: closing speed of those players (frame-to-frame displacement toward ball)
  // Scores are normalized 0-100 and averaged across the window.

  const t1Scores: number[] = []
  const t2Scores: number[] = []

  // Build per-player previous positions for velocity calculation
  const prevPositions = new Map<string, { x: number; y: number }>()

  for (const row of rows) {
    if (!row.playerPositions) continue
    const ball = row.ballPosition?.pitchPos
    if (!ball) continue
    const [bx, by] = ball

    let t1Proximity = 0,
      t1Count = 0
    let t2Proximity = 0,
      t2Count = 0
    let t1Closing = 0,
      t2Closing = 0

    for (const [playerId, pos] of Object.entries(row.playerPositions)) {
      if (pos.isGoalkeeper) continue

      const dx = pos.x - bx
      const dy = pos.y - by
      const distToBall = Math.sqrt(dx * dx + dy * dy)

      // Velocity toward ball (closing speed): positive = moving toward ball
      let closingSpeed = 0
      const prev = prevPositions.get(playerId)
      if (prev) {
        const prevDist = Math.sqrt((prev.x - bx) ** 2 + (prev.y - by) ** 2)
        closingSpeed = Math.max(0, prevDist - distToBall) // only count players closing in
      }

      if (pos.teamId === 0) {
        t1Count++
        if (distToBall <= PRESSING_RADIUS_M) {
          t1Proximity++
          t1Closing += closingSpeed
        }
      } else {
        t2Count++
        if (distToBall <= PRESSING_RADIUS_M) {
          t2Proximity++
          t2Closing += closingSpeed
        }
      }

      prevPositions.set(playerId, { x: pos.x, y: pos.y })
    }

    // Combine proximity ratio (0-1) and normalized closing speed into 0-100 score
    const proximityWeight = 0.6
    const velocityWeight = 0.4
    const maxClosingPerPlayer = 2 // ~2m/frame closing speed cap for normalization

    if (t1Count > 0) {
      const proxScore = t1Proximity / t1Count
      const velScore =
        t1Proximity > 0 ? Math.min(1, t1Closing / t1Proximity / maxClosingPerPlayer) : 0
      t1Scores.push((proximityWeight * proxScore + velocityWeight * velScore) * 100)
    }
    if (t2Count > 0) {
      const proxScore = t2Proximity / t2Count
      const velScore =
        t2Proximity > 0 ? Math.min(1, t2Closing / t2Proximity / maxClosingPerPlayer) : 0
      t2Scores.push((proximityWeight * proxScore + velocityWeight * velScore) * 100)
    }
  }

  const avg = (arr: number[]) => (arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0)
  return { team1: avg(t1Scores), team2: avg(t2Scores) }
}

export function useTrackMetrics(tracks: TrackRow[]): TrackMetrics {
  return useMemo(() => {
    if (tracks.length === 0) {
      return { compactness: [], defensiveLine: [], pressing: [] }
    }

    const compactness: MetricDataPoint[] = []
    const defensiveLine: MetricDataPoint[] = []
    const pressing: MetricDataPoint[] = []

    const minFrame = tracks[0].frameNumber
    const maxFrame = tracks[tracks.length - 1].frameNumber
    const totalFrames = maxFrame - minFrame

    // Create evenly spaced windows
    const numWindows = Math.max(8, Math.min(20, Math.floor(tracks.length / WINDOW_SIZE)))
    const framesPerWindow = Math.floor(totalFrames / numWindows)

    for (let w = 0; w < numWindows; w++) {
      const windowStart = minFrame + w * framesPerWindow
      const windowEnd = windowStart + framesPerWindow
      const windowRows = tracks.filter(
        (t) => t.frameNumber >= windowStart && t.frameNumber < windowEnd
      )
      if (windowRows.length === 0) continue

      const minute = parseFloat((windowStart / FPS / 60).toFixed(1))
      const midFrame = windowStart + framesPerWindow / 2
      const minuteLabel = parseFloat((midFrame / FPS / 60).toFixed(1))

      const c = computeCompactness(windowRows)
      const d = computeDefensiveLine(windowRows)
      const p = computePressing(windowRows)

      compactness.push({
        minute: minuteLabel,
        team1: parseFloat(c.team1.toFixed(1)),
        team2: parseFloat(c.team2.toFixed(1)),
      })
      defensiveLine.push({
        minute: minuteLabel,
        team1: parseFloat(d.team1.toFixed(1)),
        team2: parseFloat(d.team2.toFixed(1)),
      })
      pressing.push({
        minute: minuteLabel,
        team1: parseFloat(p.team1.toFixed(1)),
        team2: parseFloat(p.team2.toFixed(1)),
      })
    }

    return { compactness, defensiveLine, pressing }
  }, [tracks])
}
