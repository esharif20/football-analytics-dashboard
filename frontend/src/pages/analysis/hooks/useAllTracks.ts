import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { tracksApi } from '@/lib/api-local'

export interface TrackRow {
  id: number
  analysisId: number
  frameNumber: number
  timestamp: number
  playerPositions: Record<
    string,
    { x: number; y: number; teamId: number; isGoalkeeper?: boolean }
  > | null
  ballPosition: {
    pixelPos: [number, number] | null
    pitchPos: [number, number] | null
  } | null
  teamFormations: { possessionTeamId: number | null } | null
  voronoiData: null
}

const PAGE_SIZE = 500

async function fetchAllTracks(analysisId: number): Promise<TrackRow[]> {
  const page1 = await tracksApi.list(analysisId, { offset: 0, limit: PAGE_SIZE })
  if (page1.length < PAGE_SIZE) return page1
  const page2 = await tracksApi.list(analysisId, { offset: PAGE_SIZE, limit: PAGE_SIZE })
  return [...page1, ...page2]
}

export function useAllTracks(analysisId: number, enabled: boolean) {
  const { data: tracks = [], isLoading } = useQuery<TrackRow[]>({
    queryKey: ['tracks', analysisId],
    queryFn: () => fetchAllTracks(analysisId),
    enabled,
    staleTime: Infinity,
  })

  const { tracksByFrame, minFrame, maxFrame } = useMemo(() => {
    const map = new Map<number, TrackRow>()
    let min = Infinity
    let max = -Infinity
    for (const t of tracks) {
      map.set(t.frameNumber, t)
      if (t.frameNumber < min) min = t.frameNumber
      if (t.frameNumber > max) max = t.frameNumber
    }
    return {
      tracksByFrame: map,
      minFrame: min === Infinity ? 0 : min,
      maxFrame: max === -Infinity ? 0 : max,
    }
  }, [tracks])

  return { tracks, tracksByFrame, isLoading, frameCount: tracks.length, minFrame, maxFrame }
}
