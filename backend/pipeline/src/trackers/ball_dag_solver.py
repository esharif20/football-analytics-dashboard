"""DAG-based globally optimal ball trajectory solver.

Builds a temporal DAG of all candidate ball detections and finds the
minimum-cost path via forward dynamic programming.  For single-object
tracking on a DAG this reduces to O(V+E) — typically <10ms.

Inspired by the muSSP paper (min-cost flow for MOT), simplified for
the single-ball case.
"""

from typing import Dict, List, Optional

import numpy as np


def optimize_ball_trajectory(
    ball_tracks: List[Dict],
    raw_candidates: List[List[Dict]],
    max_gap: int = 5,
    alpha: float = 1.0,
    gamma: float = 50.0,
    delta: float = 100.0,
) -> List[Dict]:
    """Find the globally optimal ball trajectory via DAG shortest path.

    Args:
        ball_tracks: Per-frame ball track dicts (same format as tracker output).
        raw_candidates: Per-frame list of candidate dicts with ``bbox`` and
            ``confidence`` keys, produced by the tracker's filtering stage.
        max_gap: Maximum frame gap for DAG edges (default 5).
        alpha: Spatial distance cost weight.
        gamma: Confidence penalty weight (penalises low-confidence detections).
        delta: Gap penalty per skipped frame.

    Returns:
        Updated ``ball_tracks`` with optimal detections applied.  Non-optimal
        frames are cleared so that ``interpolate_ball_tracks()`` can fill gaps.
    """
    n_frames = len(ball_tracks)
    if n_frames == 0 or len(raw_candidates) == 0:
        return ball_tracks

    # Pad raw_candidates if shorter than ball_tracks
    while len(raw_candidates) < n_frames:
        raw_candidates.append([])

    # ── Phase A: Build node list ─────────────────────────────────────
    # Each node = (frame_idx, candidate_idx) or sentinel SOURCE/SINK
    # Node data: center_xy, confidence, frame_idx, candidate_idx
    SOURCE = -1
    SINK = -2

    # nodes[i] = (frame, cand_idx, cx, cy, conf)
    nodes: List[tuple] = []
    node_id_map: Dict[int, List[int]] = {}  # frame -> list of node indices

    for frame_idx in range(n_frames):
        cands = raw_candidates[frame_idx]
        if not cands:
            continue
        frame_nodes = []
        for cand_idx, cand in enumerate(cands):
            bbox = cand.get("bbox")
            conf = cand.get("confidence", 0.5)
            if bbox is None:
                continue
            cx = (bbox[0] + bbox[2]) * 0.5
            cy = (bbox[1] + bbox[3]) * 0.5
            nid = len(nodes)
            nodes.append((frame_idx, cand_idx, cx, cy, conf))
            frame_nodes.append(nid)
        if frame_nodes:
            node_id_map[frame_idx] = frame_nodes

    if not nodes:
        return ball_tracks

    n_nodes = len(nodes)

    # ── Phase B: Forward DP ──────────────────────────────────────────
    INF = float("inf")
    # dist[i] = min cost to reach node i from SOURCE
    dist = np.full(n_nodes, INF, dtype=np.float64)
    pred = np.full(n_nodes, -1, dtype=np.int64)

    # SOURCE -> first max_gap frames (zero cost)
    sorted_frames = sorted(node_id_map.keys())
    for fi in sorted_frames:
        if fi > max_gap:
            break
        for nid in node_id_map[fi]:
            _, _, _, _, conf = nodes[nid]
            cost = gamma * (1.0 - conf)
            if cost < dist[nid]:
                dist[nid] = cost
                pred[nid] = SOURCE

    # Iterate frames in order, relax outgoing edges
    for fi in sorted_frames:
        for nid_from in node_id_map[fi]:
            if dist[nid_from] == INF:
                continue
            f_from, _, cx_from, cy_from, _ = nodes[nid_from]

            # Connect to candidates in frames f_from+1 .. f_from+max_gap
            for gap in range(1, max_gap + 1):
                target_frame = f_from + gap
                if target_frame >= n_frames:
                    break
                if target_frame not in node_id_map:
                    continue

                for nid_to in node_id_map[target_frame]:
                    _, _, cx_to, cy_to, conf_to = nodes[nid_to]
                    spatial = ((cx_from - cx_to) ** 2 + (cy_from - cy_to) ** 2) ** 0.5
                    cost = (
                        alpha * spatial / gap
                        + gamma * (1.0 - conf_to)
                        + delta * (gap - 1)
                    )
                    new_dist = dist[nid_from] + cost
                    if new_dist < dist[nid_to]:
                        dist[nid_to] = new_dist
                        pred[nid_to] = nid_from

    # ── Phase C: Backtrack from best terminal node ───────────────────
    # Find the node closest to SINK (last max_gap frames with min cost)
    best_terminal = -1
    best_cost = INF
    for fi in reversed(sorted_frames):
        if fi < n_frames - max_gap - 1:
            break
        for nid in node_id_map[fi]:
            if dist[nid] < best_cost:
                best_cost = dist[nid]
                best_terminal = nid

    if best_terminal < 0:
        return ball_tracks

    # Trace path
    path_nodes: List[int] = []
    cur = best_terminal
    while cur >= 0:  # SOURCE is -1, stops the loop
        path_nodes.append(cur)
        cur = int(pred[cur])
    path_nodes.reverse()

    # Build set of (frame, cand_idx) on the optimal path
    optimal_frames: Dict[int, int] = {}
    for nid in path_nodes:
        frame_idx, cand_idx, _, _, _ = nodes[nid]
        optimal_frames[frame_idx] = cand_idx

    # Apply optimal detections: keep only frames on the path
    for frame_idx in range(n_frames):
        if frame_idx in optimal_frames:
            cand_idx = optimal_frames[frame_idx]
            cand = raw_candidates[frame_idx][cand_idx]
            ball_tracks[frame_idx][1] = {
                "bbox": list(cand["bbox"]),
                "confidence": cand.get("confidence", 0.5),
                "dag_optimized": True,
            }
        else:
            # Clear non-optimal frames for interpolation to fill
            if 1 in ball_tracks[frame_idx]:
                existing = ball_tracks[frame_idx][1]
                if not existing.get("dag_optimized"):
                    del ball_tracks[frame_idx][1]

    return ball_tracks
