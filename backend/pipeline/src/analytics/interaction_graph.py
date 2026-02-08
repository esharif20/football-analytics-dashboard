"""Compute spatial interaction graphs for each team.

Nodes are players at their average pitch positions with speed labels.
The ball is included as a special node (playerId=-1, teamId=0).
Edges represent interaction strength: a weighted combination of spatial
proximity frequency and pass connections.  Ball-player edges are weighted
by how often a player is near the ball.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance
except ImportError:
    from src.utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance

from .types import FootballEvent, KinematicStats


# Proximity threshold in cm (15 m) for player-player
_PROXIMITY_THRESHOLD_CM = 1500.0

# Ball-player proximity threshold in cm (10 m — tighter, more meaningful)
_BALL_PROXIMITY_THRESHOLD_CM = 1000.0

# Pass bonus multiplier (passes are rarer but more meaningful)
_PASS_WEIGHT = 5

# Sample every Nth frame for proximity computation
_FRAME_SAMPLE_STEP = 5

# Special ID for the ball node
BALL_NODE_ID = -1


def compute_interaction_graphs(
    tracks: Dict[str, List[Dict]],
    events: List[FootballEvent],
    per_frame_transformers: Optional[Dict] = None,
    player_kinematics: Optional[Dict[int, KinematicStats]] = None,
    fps: float = 25.0,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Compute interaction graphs for both teams.

    Args:
        tracks: Full track dictionary with players, goalkeepers, ball.
        events: Detected football events (passes used for edge weighting).
        per_frame_transformers: Per-frame ViewTransformers for pitch coords.
        player_kinematics: Pre-computed kinematics for speed labels.
        fps: Video frame rate.

    Returns:
        (team1_graph, team2_graph) — each is a dict with "nodes" and "edges"
        (including the ball node and ball-player edges),
        or (None, None) if no homography data is available.
    """
    if not per_frame_transformers:
        return None, None

    player_frames = tracks.get("players", [])
    gk_frames = tracks.get("goalkeepers", [])
    ball_frames = tracks.get("ball", [])
    n_frames = max(len(player_frames), len(gk_frames), len(ball_frames))
    if n_frames == 0:
        return None, None

    # ── 1. Build per-player and ball average pitch positions ─────────
    player_positions: Dict[int, List[Tuple[float, float]]] = {}
    player_team_ids: Dict[int, int] = {}
    ball_positions: List[Tuple[float, float]] = []

    # Also collect per-frame positions for proximity in step 2
    per_frame_player_pitch: Dict[int, Dict[int, Tuple[float, float]]] = {}
    per_frame_ball_pitch: Dict[int, Tuple[float, float]] = {}

    for frame_idx in range(0, n_frames, _FRAME_SAMPLE_STEP):
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
            # Ball tracks use key 1 typically
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

    if not player_positions:
        return None, None

    # Compute average positions
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

    # ── 2. Compute proximity matrices ────────────────────────────────
    # Player-player (same-team only)
    proximity_counts: Dict[Tuple[int, int], int] = {}
    # Ball-player (all players)
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

    # ── 3. Add pass connections ──────────────────────────────────────
    pass_counts: Dict[Tuple[int, int], int] = {}
    for ev in events:
        if ev.event_type == "pass" and ev.player_track_id is not None and ev.target_player_track_id is not None:
            pair = (min(ev.player_track_id, ev.target_player_track_id),
                    max(ev.player_track_id, ev.target_player_track_id))
            pass_counts[pair] = pass_counts.get(pair, 0) + 1

    # ── 4. Build combined edge weights ───────────────────────────────
    all_pairs = set(proximity_counts.keys()) | set(pass_counts.keys())
    raw_weights: Dict[Tuple[int, int], float] = {}
    for pair in all_pairs:
        prox = proximity_counts.get(pair, 0)
        passes = pass_counts.get(pair, 0)
        raw_weights[pair] = prox + passes * _PASS_WEIGHT

    # ── 5. Build nodes and split by team ─────────────────────────────
    # Pipeline uses team_ids 0/1 → remap to 1/2 for frontend display
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
        })

    # ── 6. Build per-team graphs with ball node ──────────────────────
    def _build_team_graph(team_id: int) -> Optional[Dict]:
        nodes = list(team_nodes.get(team_id, []))
        if len(nodes) < 2:
            return None

        # Add ball node
        if ball_avg is not None:
            nodes.append({
                "playerId": BALL_NODE_ID,
                "teamId": 0,  # neutral — belongs to both teams
                "avgX": round(ball_avg[0], 1),
                "avgY": round(ball_avg[1], 1),
                "avgSpeed": 0.0,
                "isBall": True,
            })

        node_ids = {n["playerId"] for n in nodes if n["playerId"] != BALL_NODE_ID}

        # Player-player edges (normalized within team)
        team_edges_raw = []
        for (a, b), w in raw_weights.items():
            if a in node_ids and b in node_ids:
                team_edges_raw.append((a, b, w))

        max_pp_w = max((w for _, _, w in team_edges_raw), default=0)

        edges = []
        for a, b, w in team_edges_raw:
            normalized = round(w / max_pp_w, 3) if max_pp_w > 0 else 0
            if normalized > 0.05:
                edges.append({"from": a, "to": b, "weight": normalized})

        # Ball-player edges (normalized within team)
        if ball_avg is not None:
            team_ball_raw = []
            for pid in node_ids:
                count = ball_proximity_counts.get(pid, 0)
                if count > 0:
                    team_ball_raw.append((pid, count))

            max_bp_w = max((c for _, c in team_ball_raw), default=0)
            for pid, count in team_ball_raw:
                normalized = round(count / max_bp_w, 3) if max_bp_w > 0 else 0
                if normalized > 0.05:
                    edges.append({
                        "from": BALL_NODE_ID,
                        "to": pid,
                        "weight": normalized,
                        "isBallEdge": True,
                    })

        return {"nodes": nodes, "edges": edges}

    return _build_team_graph(1), _build_team_graph(2)
