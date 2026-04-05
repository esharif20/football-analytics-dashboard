import { useMemo } from 'react'
import { pixelToPitch, ballCmToPitch } from '../utils/coordinateTransform'
import type { TrackRow } from './useAllTracks'

const DEFAULT_VIDEO_W = 1920
const DEFAULT_VIDEO_H = 1080

export interface PitchPlayer {
  id: number
  trackId: number
  x: number
  y: number
  teamId: number
}

export interface PitchFrameData {
  team1Players: PitchPlayer[]
  team2Players: PitchPlayer[]
  ball: { x: number; y: number }
}

/** Find the track row closest to `currentFrame`, favouring the floor. */
function findNearestRow(
  tracks: TrackRow[],
  tracksByFrame: Map<number, TrackRow>,
  currentFrame: number
): TrackRow | null {
  const exact = tracksByFrame.get(currentFrame)
  if (exact) return exact
  if (tracks.length === 0) return null
  let best = tracks[0]
  let bestDist = Math.abs(tracks[0].frameNumber - currentFrame)
  for (const t of tracks) {
    const d = Math.abs(t.frameNumber - currentFrame)
    if (d < bestDist) {
      bestDist = d
      best = t
    }
  }
  return best
}

export function useFrameData(
  tracks: TrackRow[],
  tracksByFrame: Map<number, TrackRow>,
  currentFrame: number,
  videoWidth = DEFAULT_VIDEO_W,
  videoHeight = DEFAULT_VIDEO_H
): PitchFrameData | null {
  return useMemo(() => {
    if (tracksByFrame.size === 0) return null

    const row = findNearestRow(tracks, tracksByFrame, currentFrame)
    if (!row) return null

    const team1Players: PitchPlayer[] = []
    const team2Players: PitchPlayer[] = []

    if (row.playerPositions) {
      let i = 0
      for (const [trackIdStr, pos] of Object.entries(row.playerPositions)) {
        const trackId = parseInt(trackIdStr)
        const { x, y } = pixelToPitch(pos.x, pos.y, videoWidth, videoHeight)
        // Backend teamId is 0-indexed; PitchRadar expects 1-indexed
        const pitchPlayer: PitchPlayer = { id: i++, trackId, x, y, teamId: pos.teamId + 1 }
        if (pos.teamId === 0) {
          team1Players.push(pitchPlayer)
        } else {
          team2Players.push(pitchPlayer)
        }
      }
    }

    // Ball position: prefer pitchPos (cm → units), fallback to pixelPos
    let ball = { x: 52.5, y: 34 }
    if (row.ballPosition?.pitchPos) {
      ball = ballCmToPitch(row.ballPosition.pitchPos as [number, number])
    } else if (row.ballPosition?.pixelPos) {
      ball = pixelToPitch(
        row.ballPosition.pixelPos[0],
        row.ballPosition.pixelPos[1],
        videoWidth,
        videoHeight
      )
    }

    return { team1Players, team2Players, ball }
  }, [tracks, tracksByFrame, currentFrame, videoWidth, videoHeight])
}
