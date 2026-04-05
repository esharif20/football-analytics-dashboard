interface Point {
  x: number
  y: number
}

/**
 * Ramer-Douglas-Peucker polyline simplification.
 * Reduces a dense point array to key waypoints.
 * epsilon: max allowed deviation in pitch units (~0.8 works well for football pitch 105x68)
 */
export function rdpSimplify(points: Point[], epsilon = 0.8): Point[] {
  if (points.length <= 2) return points

  // Find the point with maximum distance from the line between first and last
  const first = points[0]
  const last = points[points.length - 1]

  let maxDist = 0
  let maxIdx = 0

  const dx = last.x - first.x
  const dy = last.y - first.y
  const lineLenSq = dx * dx + dy * dy

  for (let i = 1; i < points.length - 1; i++) {
    let dist: number
    if (lineLenSq === 0) {
      // first === last
      const ddx = points[i].x - first.x
      const ddy = points[i].y - first.y
      dist = Math.sqrt(ddx * ddx + ddy * ddy)
    } else {
      // Perpendicular distance from point to line
      const t = ((points[i].x - first.x) * dx + (points[i].y - first.y) * dy) / lineLenSq
      const projX = first.x + t * dx
      const projY = first.y + t * dy
      const ddx = points[i].x - projX
      const ddy = points[i].y - projY
      dist = Math.sqrt(ddx * ddx + ddy * ddy)
    }
    if (dist > maxDist) {
      maxDist = dist
      maxIdx = i
    }
  }

  if (maxDist > epsilon) {
    const left = rdpSimplify(points.slice(0, maxIdx + 1), epsilon)
    const right = rdpSimplify(points.slice(maxIdx), epsilon)
    return [...left.slice(0, -1), ...right]
  }

  return [first, last]
}

/**
 * Convert an array of points to a smooth cubic Bezier SVG path string
 * using Catmull-Rom spline conversion.
 * tension: 0.5 = default smoothing
 */
export function catmullRomToSvgPath(points: Point[], tension = 0.4): string {
  if (points.length < 2) return ''
  if (points.length === 2) {
    return `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y}`
  }

  let d = `M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`

  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)]
    const p1 = points[i]
    const p2 = points[i + 1]
    const p3 = points[Math.min(points.length - 1, i + 2)]

    // Catmull-Rom → cubic Bezier control points
    const cp1x = p1.x + (p2.x - p0.x) * tension
    const cp1y = p1.y + (p2.y - p0.y) * tension
    const cp2x = p2.x - (p3.x - p1.x) * tension
    const cp2y = p2.y - (p3.y - p1.y) * tension

    d += ` C ${cp1x.toFixed(2)} ${cp1y.toFixed(2)}, ${cp2x.toFixed(2)} ${cp2y.toFixed(2)}, ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`
  }

  return d
}

/**
 * Remove outlier ball detections caused by detector noise.
 * A point is an outlier if it jumps more than maxDelta pitch units from BOTH its neighbors.
 * Multiple passes remove cascading artifacts.
 */
export function rejectOutliers(points: Point[], maxDelta = 8): Point[] {
  if (points.length < 3) return points

  let result = [...points]
  // Two passes to remove cascading false detections
  for (let pass = 0; pass < 2; pass++) {
    const filtered: Point[] = [result[0]]
    for (let i = 1; i < result.length - 1; i++) {
      const prev = filtered[filtered.length - 1]
      const curr = result[i]
      const next = result[i + 1]
      const dPrev = Math.hypot(curr.x - prev.x, curr.y - prev.y)
      const dNext = Math.hypot(curr.x - next.x, curr.y - next.y)
      // Keep point only if it's within maxDelta of at least one neighbor
      if (dPrev <= maxDelta || dNext <= maxDelta) {
        filtered.push(curr)
      }
    }
    filtered.push(result[result.length - 1])
    result = filtered
  }
  return result
}

/**
 * Moving average smoothing to reduce jitter from imperfect ball detection.
 * windowSize: number of frames to average (odd numbers work best)
 */
export function movingAverage(points: Point[], windowSize = 5): Point[] {
  if (points.length < windowSize) return points
  const half = Math.floor(windowSize / 2)
  return points.map((_, i) => {
    const start = Math.max(0, i - half)
    const end = Math.min(points.length - 1, i + half)
    let sx = 0,
      sy = 0,
      count = 0
    for (let j = start; j <= end; j++) {
      sx += points[j].x
      sy += points[j].y
      count++
    }
    return { x: sx / count, y: sy / count }
  })
}

/**
 * Dijkstra's shortest path through a pass network graph.
 * Returns the sequence of node IDs forming the minimum-cost route
 * from startId to endId, where cost = 1 / edge.weight (heavier passes = shorter path).
 * Returns [] if no path exists.
 */
export function dijkstraPassRoute(
  nodes: { id: number; x: number; y: number }[],
  edges: { from: number; to: number; weight: number }[],
  startId: number,
  endId: number
): number[] {
  const dist: Record<number, number> = {}
  const prev: Record<number, number | null> = {}
  const visited = new Set<number>()

  for (const n of nodes) {
    dist[n.id] = Infinity
    prev[n.id] = null
  }
  dist[startId] = 0

  // Build adjacency list (undirected — a pass can flow either way)
  const adj: Record<number, { to: number; cost: number }[]> = {}
  for (const n of nodes) adj[n.id] = []
  for (const e of edges) {
    if (e.weight <= 0) continue
    const cost = 1 / e.weight // higher weight = lower cost = preferred route
    adj[e.from]?.push({ to: e.to, cost })
    adj[e.to]?.push({ to: e.from, cost })
  }

  // Simple O(n²) Dijkstra — graph is tiny (≤30 nodes)
  const unvisited = new Set(nodes.map((n) => n.id))
  while (unvisited.size > 0) {
    // Pick unvisited node with minimum distance
    let u = -1
    let minDist = Infinity
    for (const id of unvisited) {
      if (dist[id] < minDist) {
        minDist = dist[id]
        u = id
      }
    }
    if (u === -1 || u === endId) break
    unvisited.delete(u)
    visited.add(u)

    for (const { to, cost } of adj[u] ?? []) {
      if (visited.has(to)) continue
      const alt = dist[u] + cost
      if (alt < dist[to]) {
        dist[to] = alt
        prev[to] = u
      }
    }
  }

  // Reconstruct path
  if (dist[endId] === Infinity) return []
  const path: number[] = []
  let curr: number | null = endId
  while (curr !== null) {
    path.unshift(curr)
    curr = prev[curr] ?? null
  }
  return path
}
