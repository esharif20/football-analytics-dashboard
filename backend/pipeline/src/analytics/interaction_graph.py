"""Compute passing-network graphs for each team.

Nodes are players at their average pitch positions with speed labels
and pass involvement counts.  Edges represent pass connections between
player pairs, with weight normalized by pass count.  Proximity data
is only used to connect orphan players (zero passes) to their nearest
teammate.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
    _HAS_NETWORKX = True
except ImportError:
    _HAS_NETWORKX = False

try:
    from utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance
except ImportError:
    from src.utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance

from .types import FootballEvent, KinematicStats


# Proximity threshold in cm (15 m) — used for orphan fallback only
_PROXIMITY_THRESHOLD_CM = 1500.0

# Ball-player proximity threshold in cm (10 m)
_BALL_PROXIMITY_THRESHOLD_CM = 1000.0

# Sample every Nth frame for proximity computation
_FRAME_SAMPLE_STEP = 5


# ── Helper: format time label ────────────────────────────────────────────
def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


# ── Helper 1: gather per-frame pitch positions ──────────────────────────
def _gather_pitch_data(
    tracks: Dict[str, List[Dict]],
    per_frame_transformers: Dict,
    frame_range: Optional[Tuple[int, int]] = None,
) -> Tuple[
    Dict[int, List[Tuple[float, float]]],   # player_positions
    Dict[int, int],                          # player_team_ids
    List[Tuple[float, float]],               # ball_positions
    Dict[int, Dict[int, Tuple[float, float]]],  # per_frame_player_pitch
    Dict[int, Tuple[float, float]],          # per_frame_ball_pitch
]:
    """Iterate sampled frames and project player/ball positions to pitch coords.

    Args:
        tracks: Full track dict with players, goalkeepers, ball.
        per_frame_transformers: Per-frame ViewTransformers.
        frame_range: Optional (start, end) to restrict frames.

    Returns:
        (player_positions, player_team_ids, ball_positions,
         per_frame_player_pitch, per_frame_ball_pitch)
    """
    player_frames = tracks.get("players", [])
    gk_frames = tracks.get("goalkeepers", [])
    ball_frames = tracks.get("ball", [])
    n_frames = max(len(player_frames), len(gk_frames), len(ball_frames))

    start = frame_range[0] if frame_range else 0
    end = frame_range[1] if frame_range else n_frames

    player_positions: Dict[int, List[Tuple[float, float]]] = {}
    player_team_ids: Dict[int, int] = {}
    ball_positions: List[Tuple[float, float]] = []
    per_frame_player_pitch: Dict[int, Dict[int, Tuple[float, float]]] = {}
    per_frame_ball_pitch: Dict[int, Tuple[float, float]] = {}

    for frame_idx in range(start, min(end, n_frames), _FRAME_SAMPLE_STEP):
        transformer = per_frame_transformers.get(frame_idx)
        if transformer is None:
            continue

        # -- Players + goalkeepers --
        frame_entities: Dict[int, Dict] = {}
        if frame_idx < len(player_frames):
            frame_entities.update(player_frames[frame_idx])
        if frame_idx < len(gk_frames):
            frame_entities.update(gk_frames[frame_idx])

        frame_pitch: Dict[int, Tuple[float, float]] = {}
        for track_id, data in frame_entities.items():
            bbox = data.get("bbox")
            if bbox is None:
                continue

            foot = get_foot_position(bbox)
            try:
                arr = np.array([foot], dtype=np.float32)
                pitch = transformer.transform_points(arr)
                px, py = float(pitch[0][0]), float(pitch[0][1])
            except Exception:
                continue

            if not (0 <= px <= 10500 and 0 <= py <= 6800):
                continue

            frame_pitch[track_id] = (px, py)

            if track_id not in player_positions:
                player_positions[track_id] = []
            player_positions[track_id].append((px, py))

            if track_id not in player_team_ids:
                tid = data.get("team_id")
                if tid is not None:
                    player_team_ids[track_id] = tid

        per_frame_player_pitch[frame_idx] = frame_pitch

        # -- Ball --
        if frame_idx < len(ball_frames):
            ball_frame = ball_frames[frame_idx]
            ball_data = ball_frame.get(1)
            if ball_data and ball_data.get("bbox") is not None:
                center = get_center_of_bbox(ball_data["bbox"])
                try:
                    arr = np.array([center], dtype=np.float32)
                    pitch = transformer.transform_points(arr)
                    bx, by = float(pitch[0][0]), float(pitch[0][1])
                    if 0 <= bx <= 10500 and 0 <= by <= 6800:
                        ball_positions.append((bx, by))
                        per_frame_ball_pitch[frame_idx] = (bx, by)
                except Exception:
                    pass

    return player_positions, player_team_ids, ball_positions, per_frame_player_pitch, per_frame_ball_pitch


# ── Helper 2: compute proximity counts ──────────────────────────────────
def _compute_proximity(
    per_frame_player_pitch: Dict[int, Dict[int, Tuple[float, float]]],
    per_frame_ball_pitch: Dict[int, Tuple[float, float]],
    player_team_ids: Dict[int, int],
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int]]:
    """Count same-team player-player and ball-player proximity frames.

    Returns:
        (proximity_counts, ball_proximity_counts)
    """
    proximity_counts: Dict[Tuple[int, int], int] = {}
    ball_proximity_counts: Dict[int, int] = {}

    for frame_idx, frame_pitch in per_frame_player_pitch.items():
        # Player-player proximity
        track_ids = list(frame_pitch.keys())
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                tid_a, tid_b = track_ids[i], track_ids[j]
                team_a = player_team_ids.get(tid_a)
                team_b = player_team_ids.get(tid_b)
                if team_a is None or team_b is None or team_a != team_b:
                    continue

                dist = measure_distance(frame_pitch[tid_a], frame_pitch[tid_b])
                if dist < _PROXIMITY_THRESHOLD_CM:
                    pair = (min(tid_a, tid_b), max(tid_a, tid_b))
                    proximity_counts[pair] = proximity_counts.get(pair, 0) + 1

        # Ball-player proximity
        ball_pos = per_frame_ball_pitch.get(frame_idx)
        if ball_pos is not None:
            for track_id, player_pos in frame_pitch.items():
                dist = measure_distance(ball_pos, player_pos)
                if dist < _BALL_PROXIMITY_THRESHOLD_CM:
                    ball_proximity_counts[track_id] = ball_proximity_counts.get(track_id, 0) + 1

    return proximity_counts, ball_proximity_counts


# ── Helper 3: count pass connections ────────────────────────────────────
def _compute_pass_counts(
    events: List[FootballEvent],
    frame_range: Optional[Tuple[int, int]] = None,
) -> Dict[Tuple[int, int], int]:
    """Count pass pairs, optionally filtered to a frame range."""
    pass_counts: Dict[Tuple[int, int], int] = {}
    for ev in events:
        if ev.event_type != "pass":
            continue
        if ev.player_track_id is None or ev.target_player_track_id is None:
            continue
        if frame_range is not None and not (frame_range[0] <= ev.frame_idx < frame_range[1]):
            continue
        pair = (min(ev.player_track_id, ev.target_player_track_id),
                max(ev.player_track_id, ev.target_player_track_id))
        pass_counts[pair] = pass_counts.get(pair, 0) + 1
    return pass_counts


# ── Helper 4: pass-primary edge weights ─────────────────────────────────
def _pass_primary_weights(
    pass_counts: Dict[Tuple[int, int], int],
) -> Dict[Tuple[int, int], float]:
    """Edge weight = raw pass count. Proximity used only for orphan handling."""
    return {pair: float(count) for pair, count in pass_counts.items()}


# ── Helper 5a: compute graph-theoretic metrics via NetworkX ───────────
def _compute_graph_metrics(nodes: List[Dict], edges: List[Dict]) -> None:
    """Attach betweenness, clustering, and degree centrality to each node.

    Operates in-place on the node dicts.  Requires NetworkX and at least
    3 nodes to produce meaningful metrics.
    """
    if not _HAS_NETWORKX or len(nodes) < 3:
        return

    G = nx.Graph()
    node_ids = {n["playerId"] for n in nodes}
    G.add_nodes_from(node_ids)
    for e in edges:
        if e["from"] in node_ids and e["to"] in node_ids:
            G.add_edge(e["from"], e["to"], weight=e["weight"])

    betweenness = nx.betweenness_centrality(G, weight=None)
    clustering = nx.clustering(G, weight="weight")
    degree = nx.degree_centrality(G)

    for n in nodes:
        pid = n["playerId"]
        n["betweenness"] = round(betweenness.get(pid, 0.0), 3)
        n["clustering"] = round(clustering.get(pid, 0.0), 3)
        n["degreeCentrality"] = round(degree.get(pid, 0.0), 3)


# ── Helper 5b: normalize and build per-team edges ────────────────────
def _build_team_graph(
    team_id: int,
    team_nodes: Dict[int, List[Dict]],
    raw_weights: Dict[Tuple[int, int], float],
    pass_counts: Dict[Tuple[int, int], int],
    proximity_counts: Optional[Dict[Tuple[int, int], int]] = None,
    min_weight: float = 0.05,
) -> Optional[Dict]:
    """Build a single team's graph dict with nodes and normalized edges.

    Edges are pass-primary. Orphan players (zero passes) get a thin
    synthetic edge to their nearest teammate.
    """
    nodes = list(team_nodes.get(team_id, []))
    if len(nodes) < 2:
        return None

    node_ids = {n["playerId"] for n in nodes}
    node_pos = {n["playerId"]: (n["avgX"], n["avgY"]) for n in nodes}

    # Player-player edges (normalized within team)
    team_edges_raw = []
    for (a, b), w in raw_weights.items():
        if a in node_ids and b in node_ids:
            team_edges_raw.append((a, b, w))

    max_pp_w = max((w for _, _, w in team_edges_raw), default=0)

    # Find each player's strongest edge (guaranteed inclusion)
    best_edge_for: Dict[int, Tuple[int, int, float]] = {}
    for a, b, w in team_edges_raw:
        for pid in (a, b):
            if pid not in best_edge_for or w > best_edge_for[pid][2]:
                best_edge_for[pid] = (a, b, w)

    # Players with zero pass edges: connect to nearest teammate
    orphan_ids = node_ids - set(best_edge_for.keys())
    for orphan in orphan_ids:
        if orphan not in node_pos:
            continue
        ox, oy = node_pos[orphan]
        best_dist = float("inf")
        best_neighbor = None
        for pid in node_ids:
            if pid == orphan or pid not in node_pos:
                continue
            px, py = node_pos[pid]
            d = ((ox - px) ** 2 + (oy - py) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_neighbor = pid
        if best_neighbor is not None:
            pair_key = (min(orphan, best_neighbor), max(orphan, best_neighbor))
            best_edge_for[orphan] = (pair_key[0], pair_key[1], 0.0)

    guaranteed_pairs: set = set()
    for _, (a, b, _) in best_edge_for.items():
        guaranteed_pairs.add((min(a, b), max(a, b)))

    edges = []
    seen_pairs: set = set()
    for a, b, w in team_edges_raw:
        normalized = round(w / max_pp_w, 3) if max_pp_w > 0 else 0
        pair = (min(a, b), max(a, b))
        raw_pass = pass_counts.get(pair, 0)
        is_guaranteed = pair in guaranteed_pairs
        if (normalized > min_weight or is_guaranteed) and pair not in seen_pairs:
            seen_pairs.add(pair)
            edges.append({
                "from": a, "to": b,
                "weight": max(normalized, 0.08 if is_guaranteed else 0),
                "passCount": raw_pass,
            })

    # Add synthetic edges for orphans (not in raw edges at all)
    for pair in guaranteed_pairs:
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            edges.append({"from": pair[0], "to": pair[1], "weight": 0.08, "passCount": 0})

    _compute_graph_metrics(nodes, edges)

    return {"nodes": nodes, "edges": edges}


# ── Helper: build team_nodes dict from positions + kinematics ───────────
def _build_team_nodes(
    avg_positions: Dict[int, Tuple[float, float]],
    player_team_ids: Dict[int, int],
    player_kinematics: Optional[Dict[int, KinematicStats]] = None,
    pass_counts: Optional[Dict[Tuple[int, int], int]] = None,
) -> Dict[int, List[Dict]]:
    """Build per-team node lists (remapping team 0/1 → 1/2)."""
    # Compute per-player total passes (sent + received)
    player_pass_totals: Dict[int, int] = {}
    if pass_counts:
        for (a, b), count in pass_counts.items():
            player_pass_totals[a] = player_pass_totals.get(a, 0) + count
            player_pass_totals[b] = player_pass_totals.get(b, 0) + count

    team_nodes: Dict[int, List[Dict]] = {1: [], 2: []}
    for track_id, (avg_x, avg_y) in avg_positions.items():
        raw_tid = player_team_ids.get(track_id)
        if raw_tid not in (0, 1):
            continue
        team_id = raw_tid + 1  # 0 → 1, 1 → 2

        avg_speed = 0.0
        if player_kinematics and track_id in player_kinematics:
            ks = player_kinematics[track_id]
            if ks.avg_speed_m_per_sec is not None:
                avg_speed = round(ks.avg_speed_m_per_sec * 3.6, 1)  # m/s → km/h

        team_nodes[team_id].append({
            "playerId": track_id,
            "teamId": team_id,
            "avgX": round(avg_x, 1),
            "avgY": round(avg_y, 1),
            "avgSpeed": avg_speed,
            "passCount": player_pass_totals.get(track_id, 0),
        })
    return team_nodes


# ── Helper: compute average positions ───────────────────────────────────
def _compute_averages(
    player_positions: Dict[int, List[Tuple[float, float]]],
    ball_positions: List[Tuple[float, float]],
) -> Tuple[Dict[int, Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Compute average positions for players and ball."""
    avg_positions: Dict[int, Tuple[float, float]] = {}
    for track_id, positions in player_positions.items():
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        avg_positions[track_id] = (sum(xs) / len(xs), sum(ys) / len(ys))

    ball_avg: Optional[Tuple[float, float]] = None
    if ball_positions:
        bxs = [p[0] for p in ball_positions]
        bys = [p[1] for p in ball_positions]
        ball_avg = (sum(bxs) / len(bxs), sum(bys) / len(bys))

    return avg_positions, ball_avg


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def compute_interaction_graphs(
    tracks: Dict[str, List[Dict]],
    events: List[FootballEvent],
    per_frame_transformers: Optional[Dict] = None,
    player_kinematics: Optional[Dict[int, KinematicStats]] = None,
    fps: float = 25.0,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Compute passing-network graphs for both teams.

    Args:
        tracks: Full track dictionary with players, goalkeepers, ball.
        events: Detected football events (passes used for edge weighting).
        per_frame_transformers: Per-frame ViewTransformers for pitch coords.
        player_kinematics: Pre-computed kinematics for speed labels.
        fps: Video frame rate.

    Returns:
        (team1_graph, team2_graph) — each is a dict with "nodes" and "edges",
        or (None, None) if no homography data is available.
    """
    if not per_frame_transformers:
        return None, None

    player_positions, player_team_ids, ball_positions, per_frame_player_pitch, per_frame_ball_pitch = \
        _gather_pitch_data(tracks, per_frame_transformers)

    if not player_positions:
        return None, None

    avg_positions, _ball_avg = _compute_averages(player_positions, ball_positions)
    proximity_counts, _ball_proximity_counts = _compute_proximity(
        per_frame_player_pitch, per_frame_ball_pitch, player_team_ids)
    pass_counts = _compute_pass_counts(events)
    raw_weights = _pass_primary_weights(pass_counts)
    team_nodes = _build_team_nodes(avg_positions, player_team_ids, player_kinematics, pass_counts)

    return (
        _build_team_graph(1, team_nodes, raw_weights, pass_counts, proximity_counts),
        _build_team_graph(2, team_nodes, raw_weights, pass_counts, proximity_counts),
    )


def compute_interaction_graph_timeline(
    tracks: Dict[str, List[Dict]],
    events: List[FootballEvent],
    per_frame_transformers: Optional[Dict] = None,
    player_kinematics: Optional[Dict[int, KinematicStats]] = None,
    fps: float = 25.0,
    n_segments: int = 5,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Compute interaction graphs with timeline segments for both teams.

    Same as compute_interaction_graphs but also produces per-segment edge
    weights so the frontend can show how the graph evolves over time.

    Args:
        tracks: Full track dictionary with players, goalkeepers, ball.
        events: Detected football events.
        per_frame_transformers: Per-frame ViewTransformers.
        player_kinematics: Pre-computed kinematics for speed labels.
        fps: Video frame rate.
        n_segments: Number of equal-duration timeline segments.

    Returns:
        (team1_graph, team2_graph) — each dict has "nodes", "edges", and
        "timeline" (list of segments with per-segment edges).
        Returns (None, None) if no homography data is available.
    """
    if not per_frame_transformers:
        return None, None

    # Full-video pass: gather all pitch data
    player_positions, player_team_ids, ball_positions, per_frame_player_pitch, per_frame_ball_pitch = \
        _gather_pitch_data(tracks, per_frame_transformers)

    if not player_positions:
        return None, None

    avg_positions, _ball_avg = _compute_averages(player_positions, ball_positions)

    # Full-video edges
    proximity_counts, _ball_proximity_counts = _compute_proximity(
        per_frame_player_pitch, per_frame_ball_pitch, player_team_ids)
    pass_counts = _compute_pass_counts(events)
    raw_weights = _pass_primary_weights(pass_counts)
    team_nodes = _build_team_nodes(avg_positions, player_team_ids, player_kinematics, pass_counts)

    graph1 = _build_team_graph(1, team_nodes, raw_weights, pass_counts, proximity_counts)
    graph2 = _build_team_graph(2, team_nodes, raw_weights, pass_counts, proximity_counts)

    if graph1 is None and graph2 is None:
        return None, None

    # ── Compute per-segment edges ────────────────────────────────────
    # Determine frame range from collected data
    all_frames = sorted(per_frame_player_pitch.keys())
    if not all_frames:
        return graph1, graph2

    min_frame = all_frames[0]
    max_frame = all_frames[-1] + 1  # exclusive end
    total_frames = max_frame - min_frame
    segment_size = max(1, total_frames // n_segments)

    for team_id, graph in [(1, graph1), (2, graph2)]:
        if graph is None:
            continue

        node_ids = {n["playerId"] for n in graph["nodes"]}
        timeline = []

        for seg_idx in range(n_segments):
            seg_start = min_frame + seg_idx * segment_size
            seg_end = min_frame + (seg_idx + 1) * segment_size if seg_idx < n_segments - 1 else max_frame

            start_sec = seg_start / fps
            end_sec = seg_end / fps
            label = f"{_format_time(start_sec)}-{_format_time(end_sec)}"

            seg_passes = _compute_pass_counts(events, frame_range=(seg_start, seg_end))
            seg_raw = _pass_primary_weights(seg_passes)

            # Normalize per-segment (not global) — relative relationships per period
            seg_team_raw = [(a, b, w) for (a, b), w in seg_raw.items() if a in node_ids and b in node_ids]
            max_pp = max((w for _, _, w in seg_team_raw), default=0)

            seg_edges = []
            for a, b, w in seg_team_raw:
                norm = round(w / max_pp, 3) if max_pp > 0 else 0
                pair = (min(a, b), max(a, b))
                raw_pass = seg_passes.get(pair, 0)
                if norm > 0.05:
                    seg_edges.append({"from": a, "to": b, "weight": norm, "passCount": raw_pass})

            timeline.append({
                "label": label,
                "startFrame": seg_start,
                "endFrame": seg_end,
                "edges": seg_edges,
            })

        graph["timeline"] = timeline

    return graph1, graph2
