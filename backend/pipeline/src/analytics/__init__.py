"""Football analytics module for possession, kinematics, and ball tracking."""

import json
from typing import Dict, List, Optional

from utils.logging_config import get_logger

_logger = get_logger("analytics")

try:
    from pitch import ViewTransformer, SoccerPitchConfiguration
except ImportError:
    from src.pitch import ViewTransformer, SoccerPitchConfiguration

from .types import (
    FramePosition,
    PossessionEvent,
    PossessionStats,
    KinematicStats,
    BallPath,
    FootballEvent,
    AnalyticsResult,
)
from .possession import PossessionCalculator, compute_possession_stats
from .kinematics import KinematicsCalculator, compute_kinematics
from .ball_path import BallPathTracker, draw_ball_path_on_pitch
from .events import EventDetector
from .interaction_graph import compute_interaction_graphs, compute_interaction_graph_timeline


class AnalyticsEngine:
    """Main interface for computing all analytics."""

    def __init__(
        self,
        fps: float = 25.0,
        pitch_config: Optional[SoccerPitchConfiguration] = None,
        control_threshold_px: float = 100.0,
        control_threshold_cm: float = 300.0,
    ):
        """Initialize analytics engine.

        Args:
            fps: Video frame rate.
            pitch_config: Pitch configuration.
            control_threshold_px: Possession distance threshold (pixels).
            control_threshold_cm: Possession distance threshold (cm).
        """
        self.fps = fps
        self.pitch_config = pitch_config or SoccerPitchConfiguration()

        self.possession_calc = PossessionCalculator(
            control_threshold_px=control_threshold_px,
            control_threshold_cm=control_threshold_cm,
        )
        self.kinematics_calc = KinematicsCalculator(
            fps=fps,
            pitch_config=self.pitch_config,
        )
        self.ball_path_tracker = BallPathTracker(fps=fps)
        self.event_detector = EventDetector(fps=fps)

    def compute(
        self,
        tracks: Dict[str, List[Dict]],
        transformer: Optional[ViewTransformer] = None,
        per_frame_transformers: Optional[Dict[int, ViewTransformer]] = None,
        ball_metrics: Optional[Dict] = None,
    ) -> AnalyticsResult:
        """Compute all analytics from tracks.

        Args:
            tracks: Full track dictionary with ball, players, goalkeepers, referees.
            transformer: ViewTransformer for real-world coordinates.
            per_frame_transformers: Per-frame ViewTransformers (preferred over single transformer).
            ball_metrics: Optional ball tracking quality metrics from compute_ball_metrics().

        Returns:
            AnalyticsResult with all computed metrics.
        """
        # Possession
        possession_events = self.possession_calc.calculate_all_frames(tracks)
        possession_stats = self.possession_calc.aggregate_stats(possession_events)

        # Kinematics — prefer per-frame transformers
        player_kinematics = self.kinematics_calc.compute_all_player_stats(
            tracks, transformer, per_frame_transformers=per_frame_transformers
        )
        ball_kinematics = self.kinematics_calc.compute_ball_stats(
            tracks, transformer, per_frame_transformers=per_frame_transformers
        )

        # Ball path — use per-frame transformers
        self.ball_path_tracker.accumulate_from_tracks(
            tracks, transformer, per_frame_transformers=per_frame_transformers
        )
        ball_path = self.ball_path_tracker.get_ball_path()

        # Event detection (passes, shots, tackles)
        events = self.event_detector.detect(
            possession_events, tracks,
            per_frame_transformers=per_frame_transformers,
        )
        _logger.info("Detected %d events (%s)",
                      len(events),
                      ", ".join(f"{t}:{c}" for t, c in
                                sorted(EventDetector.count_by_team_and_type(events).items())))

        # Interaction graphs (proximity + pass weighted)
        ig_team1, ig_team2 = compute_interaction_graph_timeline(
            tracks, events,
            per_frame_transformers=per_frame_transformers,
            player_kinematics=player_kinematics,
            fps=self.fps, n_segments=5,
        )
        if ig_team1 or ig_team2:
            _logger.info("Interaction graphs: team1=%d nodes/%d edges, team2=%d nodes/%d edges",
                          len(ig_team1["nodes"]) if ig_team1 else 0,
                          len(ig_team1["edges"]) if ig_team1 else 0,
                          len(ig_team2["nodes"]) if ig_team2 else 0,
                          len(ig_team2["edges"]) if ig_team2 else 0)

        return AnalyticsResult(
            possession=possession_stats,
            player_kinematics=player_kinematics,
            ball_kinematics=ball_kinematics,
            ball_path=ball_path,
            fps=self.fps,
            homography_available=(transformer is not None or bool(per_frame_transformers)),
            ball_metrics=ball_metrics,
            events=events,
            interaction_graph_team1=ig_team1,
            interaction_graph_team2=ig_team2,
        )


def print_analytics_summary(result: AnalyticsResult) -> None:
    """Print formatted analytics summary to console.

    Args:
        result: AnalyticsResult to print.
    """
    from utils.pipeline_logger import banner, config_table, metric, divider

    banner("Analytics Summary")

    # Possession
    p = result.possession
    config_table("Possession", {
        "Team 1": f"{p.team_1_percentage:.1f}% ({p.team_1_frames} frames)",
        "Team 2": f"{p.team_2_percentage:.1f}% ({p.team_2_frames} frames)",
        "Contested": f"{p.contested_frames} frames",
        "Changes": str(p.possession_changes),
    })

    # Ball stats
    b = result.ball_kinematics
    if b.total_distance_m is not None:
        metric("Ball distance", b.total_distance_m, "m")
        if b.avg_speed_m_per_sec is not None:
            metric("Ball avg speed", b.avg_speed_m_per_sec * 3.6, "km/h")
        if b.max_speed_m_per_sec is not None:
            metric("Ball max speed", b.max_speed_m_per_sec * 3.6, "km/h")
    else:
        metric("Ball distance (px)", b.total_distance_px)

    bp = result.ball_path
    metric("Direction changes", bp.direction_changes)

    # Events summary
    if result.events:
        counts = EventDetector.count_by_team_and_type(result.events)
        divider()
        _logger.info("  Events detected: %d total", len(result.events))
        for etype, team_counts in sorted(counts.items()):
            parts = [f"Team {t}: {c}" for t, c in sorted(team_counts.items())]
            metric(f"  {etype.capitalize()}", ", ".join(parts))

    # Top 5 players by distance
    if result.player_kinematics:
        divider()
        _logger.info("  Top 5 players by distance:")
        sorted_players = sorted(
            result.player_kinematics.values(),
            key=lambda x: x.total_distance_m if x.total_distance_m else x.total_distance_px,
            reverse=True,
        )[:5]

        for i, player in enumerate(sorted_players, 1):
            team_str = f"Team {player.team_id}" if player.team_id is not None else "Unknown"
            if player.total_distance_m is not None:
                speed_str = ""
                if player.avg_speed_m_per_sec is not None:
                    speed_str = f", avg {player.avg_speed_m_per_sec:.2f} m/s"
                if player.max_speed_m_per_sec is not None:
                    speed_str += f", max {player.max_speed_m_per_sec:.2f} m/s"
                metric(f"  {i}. Track {player.track_id} ({team_str})", f"{player.total_distance_m:.0f}m{speed_str}")
            else:
                metric(f"  {i}. Track {player.track_id} ({team_str})", f"{player.total_distance_px:.0f}px")


def export_analytics_json(result: AnalyticsResult, filepath: str) -> None:
    """Export analytics to JSON file.

    Args:
        result: AnalyticsResult to export.
        filepath: Output file path.
    """
    def serialize(obj):
        """Recursively serialize objects to JSON-compatible format."""
        import math
        if hasattr(obj, '__dict__'):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [serialize(v) for v in obj]
        elif isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return [serialize(v) for v in obj]
        elif isinstance(obj, float):
            # inf/nan are not JSON-compliant — replace with None
            return obj if math.isfinite(obj) else None
        elif isinstance(obj, (int, str, bool, type(None))):
            return obj
        else:
            # Catch numpy scalar types (np.float64, np.int64, etc.)
            try:
                f = float(obj)
                return f if math.isfinite(f) else None
            except (TypeError, ValueError):
                return str(obj)

    data = serialize(result)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    _logger.info(f"Analytics exported to: {filepath}")


def export_tracks_json(
    tracks: Dict,
    result: "AnalyticsResult",
    filepath: str,
    max_frames: int = 750,
) -> None:
    """Export per-frame tracking data to JSON for database ingestion.

    Produces a list of frame objects suitable for POST /api/worker/tracks/{id}.
    Ball positions come from result.ball_path.positions (already computed).
    Player positions are derived from the raw tracks dict (bbox center-bottom point).
    Frame count is capped at max_frames via uniform downsampling.

    Args:
        tracks: Raw tracker output dict with keys: players, ball, goalkeepers, referees.
        result: AnalyticsResult (provides ball_path, possession, fps).
        filepath: Output file path.
        max_frames: Maximum rows to output (default 750 — 30s clip at 25fps).
    """
    import math

    fps = result.fps if result.fps else 25.0

    # Build ball position lookup: frame_idx -> {pixelPos, pitchPos}
    ball_lookup: Dict[int, Dict] = {}
    for fp in result.ball_path.positions:
        ball_lookup[fp.frame_idx] = {
            "pixelPos": [fp.pixel_pos[0], fp.pixel_pos[1]] if fp.pixel_pos else None,
            "pitchPos": [fp.pitch_pos[0], fp.pitch_pos[1]] if fp.pitch_pos else None,
        }

    # Build possession lookup: frame_idx -> team_id
    poss_lookup: Dict[int, int] = {}
    for pe in result.possession.events:
        poss_lookup[pe.frame_idx] = pe.team_id

    # Determine total frame count from players track list
    players_list = tracks.get("players", [])
    total_frames = len(players_list)
    if total_frames == 0:
        _logger.warning("export_tracks_json: no player frames found in tracks dict")
        return

    # Compute downsample stride (keep every stride-th frame; always include frame 0)
    stride = max(1, math.ceil(total_frames / max_frames))
    selected_frames = list(range(0, total_frames, stride))
    # Ensure we do not exceed max_frames
    selected_frames = selected_frames[:max_frames]

    def _safe(v):
        """Replace inf/nan with None for JSON safety."""
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        return v

    frame_rows = []
    for frame_idx in selected_frames:
        timestamp = _safe(frame_idx / fps)

        # Ball position for this frame
        ball_pos = ball_lookup.get(frame_idx)

        # Player positions: dict of track_id -> {x, y, teamId}
        # x, y = center-bottom of bbox (foot contact point, more stable than center)
        player_positions: Dict[str, Dict] = {}
        frame_detections = players_list[frame_idx] if frame_idx < len(players_list) else {}
        for track_id, det in frame_detections.items():
            bbox = det.get("bbox") or det.get("bbox_tlwh")
            if not bbox or len(bbox) < 4:
                continue
            # bbox format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            cx = _safe((x1 + x2) / 2)
            cy = _safe(float(y2))  # bottom-center (foot contact)
            player_positions[str(track_id)] = {
                "x": cx,
                "y": cy,
                "teamId": det.get("team_id"),
            }

        # Also include goalkeepers in player_positions
        gk_list = tracks.get("goalkeepers", [])
        if frame_idx < len(gk_list):
            for track_id, det in gk_list[frame_idx].items():
                bbox = det.get("bbox") or det.get("bbox_tlwh")
                if not bbox or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                player_positions[str(track_id)] = {
                    "x": _safe((x1 + x2) / 2),
                    "y": _safe(float(y2)),
                    "teamId": det.get("team_id"),
                    "isGoalkeeper": True,
                }

        frame_rows.append({
            "frameNumber": frame_idx,
            "timestamp": timestamp,
            "ballPosition": ball_pos,
            "playerPositions": player_positions if player_positions else None,
            "possessionTeamId": poss_lookup.get(frame_idx),
        })

    with open(filepath, "w") as f:
        json.dump(frame_rows, f)

    _logger.info(
        "Tracks exported: %d frames (total=%d, stride=%d, max=%d) → %s",
        len(frame_rows), total_frames, stride, max_frames, filepath,
    )


__all__ = [
    # Main engine
    "AnalyticsEngine",

    # Convenience functions
    "compute_possession_stats",
    "compute_kinematics",
    "print_analytics_summary",
    "export_analytics_json",
    "export_tracks_json",
    "draw_ball_path_on_pitch",

    # Types
    "FramePosition",
    "PossessionEvent",
    "PossessionStats",
    "KinematicStats",
    "BallPath",
    "FootballEvent",
    "AnalyticsResult",

    # Sub-modules
    "PossessionCalculator",
    "KinematicsCalculator",
    "BallPathTracker",
    "EventDetector",
    "compute_interaction_graphs",
    "compute_interaction_graph_timeline",
]
