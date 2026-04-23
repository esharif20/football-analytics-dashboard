"""Extract per-frame tracking data from Supabase PostgreSQL for evaluation.

Reads tracks, events, analyses, and statistics tables, computes frame-level
derived metrics (centroids, compactness, zone occupancy, ball trajectory),
and estimates team formations from player positions.

Usage:
    python -m backend.evaluation.db_extractor \
        --analysis-id 18 \
        --output eval_output/dissertation/db_grounded/
"""

import argparse
import asyncio
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import asyncpg

# Optional scipy for convex hull — fall back to bounding-box area if missing.
try:
    from scipy.spatial import ConvexHull
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    print("WARNING: scipy not available — using bounding-box area for compactness")

# ── Constants ─────────────────────────────────────────────────────────────────

_PITCH_LENGTH_M = 105.0   # X direction (attacking direction), metres
_PITCH_WIDTH_M = 68.0     # Y direction, metres

# Pitch thirds (metres along X; playerPositions.pitchX is in metres)
_THIRD_DEF_MAX = 35.0
_THIRD_MID_MAX = 70.0

# Formation clustering thresholds (metres along X)
_LINE_DEFENDERS_MAX = 35.0
_LINE_DEF_MIDS_MAX = 60.0
_LINE_ATK_MIDS_MAX = 75.0

# Minimum players with pitch coords to compute centroid
_MIN_CENTROID_PLAYERS = 3

# Formation temporal window size (frames)
_FORMATION_WINDOW = 150


# ── Database queries ──────────────────────────────────────────────────────────


async def _fetch_tracks(conn: asyncpg.Connection, analysis_id: int) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT
            "frameNumber",
            "timestamp",
            "playerPositions",
            "ballPosition",
            "teamFormations"
        FROM tracks
        WHERE "analysisId" = $1
        ORDER BY "frameNumber" ASC
        """,
        analysis_id,
    )
    result = []
    for row in rows:
        def _parse(val):
            if val is None:
                return None
            if isinstance(val, str):
                return json.loads(val)
            return val

        result.append(
            {
                "frameNumber": row["frameNumber"],
                "timestamp": float(row["timestamp"]) if row["timestamp"] is not None else None,
                "playerPositions": _parse(row["playerPositions"]) or {},
                "ballPosition": _parse(row["ballPosition"]),
                "teamFormations": _parse(row["teamFormations"]),
            }
        )
    return result


async def _fetch_events(conn: asyncpg.Connection, analysis_id: int) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT
            type,
            "frameNumber",
            "timestamp",
            "playerId",
            "teamId",
            "targetPlayerId",
            "startX",
            "startY",
            "endX",
            "endY",
            success,
            confidence
        FROM events
        WHERE "analysisId" = $1
        ORDER BY "frameNumber" ASC
        """,
        analysis_id,
    )
    result = []
    for row in rows:
        result.append(
            {
                "type": row["type"],
                "frameNumber": row["frameNumber"],
                "timestamp": float(row["timestamp"]) if row["timestamp"] is not None else None,
                "playerId": row["playerId"],
                "teamId": row["teamId"],
                "targetPlayerId": row["targetPlayerId"],
                "startX": float(row["startX"]) if row["startX"] is not None else None,
                "startY": float(row["startY"]) if row["startY"] is not None else None,
                "endX": float(row["endX"]) if row["endX"] is not None else None,
                "endY": float(row["endY"]) if row["endY"] is not None else None,
                "success": row["success"],
                "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
            }
        )
    return result


async def _fetch_analytics_json(conn: asyncpg.Connection, analysis_id: int) -> dict:
    row = await conn.fetchrow(
        'SELECT "analyticsDataUrl" FROM analyses WHERE id = $1',
        analysis_id,
    )
    if row is None:
        print(f"  WARNING: No analyses row found for id={analysis_id}")
        return {}
    raw = row["analyticsDataUrl"]
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"  WARNING: Could not parse analyticsDataUrl: {exc}")
        return {}


async def _fetch_statistics(conn: asyncpg.Connection, analysis_id: int) -> dict:
    row = await conn.fetchrow(
        """
        SELECT
            "possessionTeam1", "possessionTeam2",
            "passesTeam1", "passesTeam2",
            "distanceCoveredTeam1", "distanceCoveredTeam2",
            "avgSpeedTeam1", "avgSpeedTeam2",
            "maxSpeedTeam1", "maxSpeedTeam2",
            "possessionChanges",
            "teamColorTeam1", "teamColorTeam2",
            "passNetworkTeam1", "passNetworkTeam2"
        FROM statistics
        WHERE "analysisId" = $1
        """,
        analysis_id,
    )
    if row is None:
        print(f"  WARNING: No statistics row found for analysisId={analysis_id}")
        return {}

    def _parse(val):
        if val is None:
            return None
        if isinstance(val, str):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return val
        return val

    return {
        "possessionTeam1": float(row["possessionTeam1"]) if row["possessionTeam1"] is not None else None,
        "possessionTeam2": float(row["possessionTeam2"]) if row["possessionTeam2"] is not None else None,
        "passesTeam1": row["passesTeam1"],
        "passesTeam2": row["passesTeam2"],
        "distanceCoveredTeam1": float(row["distanceCoveredTeam1"]) if row["distanceCoveredTeam1"] is not None else None,
        "distanceCoveredTeam2": float(row["distanceCoveredTeam2"]) if row["distanceCoveredTeam2"] is not None else None,
        "avgSpeedTeam1": float(row["avgSpeedTeam1"]) if row["avgSpeedTeam1"] is not None else None,
        "avgSpeedTeam2": float(row["avgSpeedTeam2"]) if row["avgSpeedTeam2"] is not None else None,
        "maxSpeedTeam1": float(row["maxSpeedTeam1"]) if row["maxSpeedTeam1"] is not None else None,
        "maxSpeedTeam2": float(row["maxSpeedTeam2"]) if row["maxSpeedTeam2"] is not None else None,
        "possessionChanges": row["possessionChanges"],
        "teamColorTeam1": row["teamColorTeam1"],
        "teamColorTeam2": row["teamColorTeam2"],
        "passNetworkTeam1": _parse(row["passNetworkTeam1"]),
        "passNetworkTeam2": _parse(row["passNetworkTeam2"]),
    }


# ── Pitch-coord helpers ───────────────────────────────────────────────────────


def _safe_float(val) -> float | None:
    """Return float or None, discarding NaN/Inf."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _player_pitch_coords(player_data: dict) -> tuple[float | None, float | None]:
    """Extract (pitchX, pitchY) from a player entry, returning (None, None) if missing."""
    px = _safe_float(player_data.get("pitchX"))
    py = _safe_float(player_data.get("pitchY"))
    return px, py


def _team_pitch_points(player_positions: dict, team_id: int) -> list[tuple[float, float]]:
    """Collect all (pitchX, pitchY) pairs for players of a given team in one frame."""
    points = []
    for pid, pdata in player_positions.items():
        if not isinstance(pdata, dict):
            continue
        if pdata.get("teamId") != team_id:
            continue
        px, py = _player_pitch_coords(pdata)
        if px is not None and py is not None:
            points.append((px, py))
    return points


# ── Frame-level metric computation ────────────────────────────────────────────


def _compute_centroid(points: list[tuple[float, float]]) -> list[float] | None:
    if len(points) < _MIN_CENTROID_PLAYERS:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def _compute_convex_hull_area_m2(points: list[tuple[float, float]]) -> float | None:
    """Convex hull area in m² (input already in metres)."""
    if len(points) < 3:
        return None
    if _SCIPY_AVAILABLE:
        try:
            import numpy as np
            pts = np.array(points)
            hull = ConvexHull(pts)
            return float(hull.volume)  # in 2D, ConvexHull.volume = area (m²)
        except Exception:
            pass
    # Fallback: bounding-box area
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _pitch_third(pitch_x: float) -> str:
    """Return 'defensive', 'middle', or 'attacking' based on pitchX (cm)."""
    if pitch_x < _THIRD_DEF_MAX:
        return "defensive"
    if pitch_x < _THIRD_MID_MAX:
        return "middle"
    return "attacking"


def _compute_frame_metrics(frames: list[dict], fps: float) -> dict:
    """Compute all frame-level derived metrics from the per-frame track data."""
    team_centroids = []
    inter_team_distance = []
    compactness = []
    ball_trajectory = []

    # For zone occupancy: accumulate player frame counts per (team, zone)
    zone_counts: dict[str, dict[str, int]] = {
        "team_1": {"defensive": 0, "middle": 0, "attacking": 0},
        "team_2": {"defensive": 0, "middle": 0, "attacking": 0},
    }
    zone_total: dict[str, int] = {"team_1": 0, "team_2": 0}

    # For possession sequence
    possession_sequence = []
    prev_possession = None

    prev_ball_x: float | None = None
    prev_ball_y: float | None = None
    dt = 1.0 / fps if fps > 0 else 1.0 / 25.0

    for frame_data in frames:
        frame_num = frame_data["frameNumber"]
        player_positions = frame_data["playerPositions"]

        # Team IDs present in this frame
        team_ids_in_frame = set()
        for pdata in player_positions.values():
            if isinstance(pdata, dict) and "teamId" in pdata:
                tid = pdata["teamId"]
                if tid is not None:
                    team_ids_in_frame.add(tid)

        # Player positions use teamId 1 (team 1) and 0 (team 2)
        t1_points = _team_pitch_points(player_positions, 1)
        t2_points = _team_pitch_points(player_positions, 0)

        c1 = _compute_centroid(t1_points)
        c2 = _compute_centroid(t2_points)

        team_centroids.append(
            {"frame": frame_num, "team_1": c1, "team_2": c2}
        )

        # Inter-team distance in metres (pitchX/pitchY already in metres)
        if c1 is not None and c2 is not None:
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            dist_m = math.sqrt(dx * dx + dy * dy)
            inter_team_distance.append({"frame": frame_num, "distance_m": dist_m})
        else:
            inter_team_distance.append({"frame": frame_num, "distance_m": None})

        # Compactness (convex hull area) per team
        area1 = _compute_convex_hull_area_m2(t1_points) if len(t1_points) >= 3 else None
        area2 = _compute_convex_hull_area_m2(t2_points) if len(t2_points) >= 3 else None
        compactness.append({"frame": frame_num, "team_1_m2": area1, "team_2_m2": area2})

        # Zone occupancy — accumulate per player per frame (teamId 1 = T1, 0 = T2)
        for team_key, team_id in [("team_1", 1), ("team_2", 0)]:
            pts = _team_pitch_points(player_positions, team_id)
            for (px, _py) in pts:
                zone = _pitch_third(px)
                zone_counts[team_key][zone] += 1
                zone_total[team_key] += 1

        # Possession sequence
        tf = frame_data.get("teamFormations") or {}
        poss_team = tf.get("possessionTeamId") if isinstance(tf, dict) else None
        if poss_team is not None and poss_team != prev_possession:
            possession_sequence.append({"frame": frame_num, "team": poss_team})
            prev_possession = poss_team

        # Ball trajectory
        ball = frame_data.get("ballPosition")
        ball_x: float | None = None
        ball_y: float | None = None
        speed: float | None = None

        if isinstance(ball, dict):
            pitch_pos = ball.get("pitchPos")
            if isinstance(pitch_pos, (list, tuple)) and len(pitch_pos) >= 2:
                # pitchPos is in cm — convert to metres for consistency
                raw_x = _safe_float(pitch_pos[0])
                raw_y = _safe_float(pitch_pos[1])
                ball_x = raw_x / 100.0 if raw_x is not None else None
                ball_y = raw_y / 100.0 if raw_y is not None else None

        if ball_x is not None and ball_y is not None:
            if prev_ball_x is not None and prev_ball_y is not None:
                ddx = ball_x - prev_ball_x
                ddy = ball_y - prev_ball_y
                raw_speed = math.sqrt(ddx * ddx + ddy * ddy) / dt
                # Discard teleportation artefacts — real ball max ~50 m/s
                speed = raw_speed if raw_speed <= 50.0 else None
                # Also discard prev position if it looks like a jump (reset chain)
                if speed is None:
                    prev_ball_x = None
                    prev_ball_y = None
                else:
                    prev_ball_x = ball_x
                    prev_ball_y = ball_y
            else:
                prev_ball_x = ball_x
                prev_ball_y = ball_y
        else:
            prev_ball_x = None
            prev_ball_y = None

        ball_trajectory.append(
            {"frame": frame_num, "pitchX": ball_x, "pitchY": ball_y, "speed_m_per_s": speed}
        )

    # Normalise zone occupancy to fractions
    pitch_zone_occupancy: dict = {}
    for team_key in ("team_1", "team_2"):
        total = zone_total[team_key]
        if total > 0:
            pitch_zone_occupancy[team_key] = {
                z: zone_counts[team_key][z] / total
                for z in ("defensive", "middle", "attacking")
            }
        else:
            pitch_zone_occupancy[team_key] = {
                "defensive": 0.0, "middle": 0.0, "attacking": 0.0
            }

    return {
        "team_centroids": team_centroids,
        "inter_team_distance_m": inter_team_distance,
        "compactness_m2": compactness,
        "pitch_zone_occupancy": pitch_zone_occupancy,
        "possession_sequence": possession_sequence,
        "ball_trajectory": ball_trajectory,
    }


# ── Formation estimation ──────────────────────────────────────────────────────


def _aggregate_player_positions(
    frames: list[dict], team_id: int, min_appearance_frac: float = 0.10
) -> dict[str, list[float]]:
    """Compute mean pitchX/pitchY per player; filter spurious tracks.

    Only keeps players who appear in at least `min_appearance_frac` of frames
    (default 10%), to discard re-id noise. Returns at most 11 players (10
    outfield + GK).
    """
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    counts: dict[str, int] = defaultdict(int)
    total_frames = len(frames)

    for frame_data in frames:
        player_positions = frame_data["playerPositions"]
        for pid, pdata in player_positions.items():
            if not isinstance(pdata, dict):
                continue
            if pdata.get("teamId") != team_id:
                continue
            px, py = _player_pitch_coords(pdata)
            if px is not None and py is not None:
                sums[pid][0] += px
                sums[pid][1] += py
                counts[pid] += 1

    # Filter: require min_appearance_frac of total frames
    min_count = max(1, int(total_frames * min_appearance_frac))
    qualified = {pid: cnt for pid, cnt in counts.items() if cnt >= min_count}

    # Keep only the top 11 most-appearing players (realistic squad size)
    top_pids = sorted(qualified, key=lambda p: qualified[p], reverse=True)[:11]

    avg_positions: dict[str, list[float]] = {}
    for pid in top_pids:
        total = counts[pid]
        avg_positions[pid] = [sums[pid][0] / total, sums[pid][1] / total]
    return avg_positions


def _assign_to_formation_lines(avg_positions: dict[str, list[float]]) -> list[int]:
    """
    Cluster outfield players into formation lines using 1-D k-means on pitchX.

    Uses k=3 lines (defenders / midfielders / forwards) which is robust to
    the noisy track IDs produced by CV re-identification. Returns player counts
    per line from defensive (low pitchX) to attacking (high pitchX).
    """
    if not avg_positions:
        return []

    xs = [pos[0] for pos in avg_positions.values()]
    n = len(xs)
    if n <= 3:
        return [n]

    k = 3
    xs_sorted = sorted(xs)
    # Initialise centroids at evenly spaced percentiles
    centroids = [xs_sorted[int(i * (n - 1) / (k - 1))] for i in range(k)]

    for _ in range(30):  # iterate to convergence
        clusters: list[list[float]] = [[] for _ in range(k)]
        for x in xs:
            nearest = min(range(k), key=lambda i: abs(x - centroids[i]))
            clusters[nearest].append(x)
        new_centroids = []
        for i, cluster in enumerate(clusters):
            new_centroids.append(sum(cluster) / len(cluster) if cluster else centroids[i])
        if new_centroids == centroids:
            break
        centroids = new_centroids

    # Sort clusters from defensive to attacking end by centroid
    sorted_clusters = sorted(zip(centroids, clusters), key=lambda x: x[0])
    counts = [len(c) for _, c in sorted_clusters if c]
    return counts


def _counts_to_formation_str(counts: list[int]) -> str:
    if not counts:
        return "unknown"
    return "-".join(str(c) for c in counts)


def _formation_confidence(frames: list[dict], team_id: int, avg_positions: dict) -> float:
    """
    Estimate confidence as 1 - normalised_mean_pitchX_variance for outfield players.
    Higher variance → lower confidence.
    """
    if not avg_positions:
        return 0.0

    variances = []
    player_x_frames: dict[str, list[float]] = defaultdict(list)

    for frame_data in frames:
        for pid, pdata in frame_data["playerPositions"].items():
            if not isinstance(pdata, dict):
                continue
            if pdata.get("teamId") != team_id:
                continue
            if pid not in avg_positions:
                continue
            px, _ = _player_pitch_coords(pdata)
            if px is not None:
                player_x_frames[pid].append(px)

    for pid, xs in player_x_frames.items():
        if len(xs) < 2:
            continue
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        variances.append(var)

    if not variances:
        return 0.5  # insufficient data

    mean_var = sum(variances) / len(variances)
    # Normalise by a reference variance (~15m = half a third of pitch)
    norm_var = mean_var / (15.0 ** 2)
    confidence = max(0.0, min(1.0, 1.0 - norm_var))
    return round(confidence, 3)


def _estimate_formation_for_frames(
    frames: list[dict], team_id: int
) -> dict:
    """Estimate formation for a given team over the provided frames."""
    avg_positions = _aggregate_player_positions(frames, team_id)

    if not avg_positions:
        return {
            "formation": "unknown",
            "confidence": 0.0,
            "avg_positions": {},
            "line_counts": [],
        }

    # Sort players by pitchX to find goalkeeper (deepest = min pitchX)
    sorted_by_x = sorted(avg_positions.items(), key=lambda kv: kv[1][0])

    # Exclude goalkeeper (player closest to own goal = lowest pitchX)
    outfield = dict(sorted_by_x[1:]) if len(sorted_by_x) > 1 else dict(sorted_by_x)

    line_counts = _assign_to_formation_lines(outfield)
    formation_str = _counts_to_formation_str(line_counts)
    confidence = _formation_confidence(frames, team_id, outfield)

    return {
        "formation": formation_str,
        "confidence": confidence,
        "avg_positions": {pid: pos for pid, pos in avg_positions.items()},
        "line_counts": line_counts,
    }


def _estimate_formations(frames: list[dict]) -> dict:
    """Estimate formations for both teams, globally and per temporal window."""
    team_1 = _estimate_formation_for_frames(frames, 1)
    team_2 = _estimate_formation_for_frames(frames, 0)  # teamId=0 in player positions

    # Temporal formations: sliding non-overlapping windows of FORMATION_WINDOW frames
    temporal = []
    for start_idx in range(0, len(frames), _FORMATION_WINDOW):
        window = frames[start_idx: start_idx + _FORMATION_WINDOW]
        if not window:
            continue
        window_start_frame = window[0]["frameNumber"]
        t1_win = _estimate_formation_for_frames(window, 1)
        t2_win = _estimate_formation_for_frames(window, 0)
        temporal.append(
            {
                "window_start": window_start_frame,
                "team_1": t1_win["formation"],
                "team_2": t2_win["formation"],
            }
        )

    return {
        "team_1": team_1,
        "team_2": team_2,
        "temporal": temporal,
    }


# ── FPS estimation ────────────────────────────────────────────────────────────


def _estimate_fps(frames: list[dict]) -> float:
    """Estimate FPS from frame timestamps. Falls back to 25.0."""
    timestamps = [
        f["timestamp"]
        for f in frames
        if f.get("timestamp") is not None
    ]
    if len(timestamps) < 2:
        return 25.0
    total_time = timestamps[-1] - timestamps[0]
    total_frames = len(timestamps) - 1
    if total_time <= 0:
        return 25.0
    return total_frames / total_time


# ── Main extraction function ──────────────────────────────────────────────────


async def extract_analysis_data(analysis_id: int, db_url: str) -> dict:
    """Extract per-frame tracking data from Supabase for a given analysis.

    Args:
        analysis_id: The analysis ID to extract data for.
        db_url: asyncpg-compatible DSN (postgresql://user:pass@host:port/db).

    Returns:
        Nested dict with keys: analysis_id, analytics, per_frame, events_db,
        statistics, frame_metrics, formations.
    """
    print(f"Connecting to database for analysis_id={analysis_id}...")
    conn = await asyncpg.connect(db_url)
    try:
        print("Fetching analytics JSON from analyses table...")
        analytics = await _fetch_analytics_json(conn, analysis_id)
        print(f"  Analytics keys: {list(analytics.keys())[:8]}")

        print("Fetching tracks...")
        raw_frames = await _fetch_tracks(conn, analysis_id)
        print(f"  Loaded {len(raw_frames)} track frames")

        print("Fetching events...")
        events_db = await _fetch_events(conn, analysis_id)
        print(f"  Loaded {len(events_db)} events")

        print("Fetching statistics...")
        statistics = await _fetch_statistics(conn, analysis_id)
        print(f"  Statistics keys: {list(statistics.keys())}")

    finally:
        await conn.close()

    # Build per_frame structure
    fps = _estimate_fps(raw_frames)
    print(f"Estimated FPS: {fps:.2f}")

    per_frame_frames = []
    for f in raw_frames:
        tf = f.get("teamFormations") or {}
        poss_team_id = tf.get("possessionTeamId") if isinstance(tf, dict) else None
        per_frame_frames.append(
            {
                "frameNumber": f["frameNumber"],
                "timestamp": f["timestamp"],
                "playerPositions": f["playerPositions"],
                "ballPosition": f["ballPosition"],
                "possessionTeamId": poss_team_id,
            }
        )

    per_frame = {
        "frames": per_frame_frames,
        "total_frames": len(per_frame_frames),
        "fps": fps,
    }

    # Compute frame metrics (uses the raw_frames which include teamFormations)
    print("Computing frame-level metrics...")
    frame_metrics = _compute_frame_metrics(raw_frames, fps)
    valid_centroids = sum(
        1 for c in frame_metrics["team_centroids"] if c["team_1"] is not None
    )
    valid_ball = sum(
        1 for b in frame_metrics["ball_trajectory"] if b["pitchX"] is not None
    )
    print(f"  Centroids computed for {valid_centroids}/{len(raw_frames)} frames")
    print(f"  Ball position available for {valid_ball}/{len(raw_frames)} frames")

    # Estimate formations
    print("Estimating formations...")
    formations = _estimate_formations(raw_frames)
    print(f"  Team 1 formation: {formations['team_1']['formation']} "
          f"(confidence={formations['team_1']['confidence']:.2f})")
    print(f"  Team 2 formation: {formations['team_2']['formation']} "
          f"(confidence={formations['team_2']['confidence']:.2f})")

    return {
        "analysis_id": analysis_id,
        "analytics": analytics,
        "per_frame": per_frame,
        "events_db": events_db,
        "statistics": statistics,
        "frame_metrics": frame_metrics,
        "formations": formations,
    }


# ── JSON serialisation helper ─────────────────────────────────────────────────


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that replaces non-finite floats with null."""

    def default(self, obj):  # type: ignore[override]
        return super().default(obj)

    def iterencode(self, o, _one_shot=False):
        # Replace nan/inf before serialisation
        return super().iterencode(self._clean(o), _one_shot)

    def _clean(self, obj):
        if isinstance(obj, float):
            if not math.isfinite(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: self._clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean(v) for v in obj]
        return obj


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-frame data from Supabase PostgreSQL"
    )
    parser.add_argument("--analysis-id", type=int, required=True)
    parser.add_argument(
        "--db-url",
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:54322/postgres",
        ),
    )
    parser.add_argument(
        "--output",
        default="eval_output/dissertation/db_grounded/",
    )
    args = parser.parse_args()

    data = asyncio.run(extract_analysis_data(args.analysis_id, args.db_url))

    out_path = Path(args.output) / f"{args.analysis_id}_db_ground_truth.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, cls=_SafeEncoder, indent=2)
    print(f"Saved: {out_path}")
    print(f"Frames: {data['per_frame']['total_frames']}")
    print(f"Events: {len(data['events_db'])}")
    print(f"Formation T1: {data['formations']['team_1']['formation']}")
    print(f"Formation T2: {data['formations']['team_2']['formation']}")
