"""Speed and distance metrics calculation."""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pitch import ViewTransformer, SoccerPitchConfiguration
    from utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance
except ImportError:
    from src.pitch import ViewTransformer, SoccerPitchConfiguration
    from src.utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance
from .types import KinematicStats, FramePosition


# Conversion constants
CM_TO_M = 0.01

# Speed clamp — world-class sprint is ~36 km/h
MAX_PLAYER_SPEED_KMH = 40.0


class KinematicsCalculator:
    """Calculate speed and distance metrics for players and ball."""

    def __init__(
        self,
        fps: float = 25.0,
        pitch_config: Optional[SoccerPitchConfiguration] = None,
    ):
        """Initialize kinematics calculator.

        Args:
            fps: Video frame rate for time calculations.
            pitch_config: Pitch configuration for dimensions.
        """
        self.fps = fps
        self.pitch_config = pitch_config or SoccerPitchConfiguration()

    def extract_positions(
        self,
        tracks: Dict[str, List[Dict]],
        entity_type: str,  # "players", "goalkeepers", "ball"
        track_id: Optional[int] = None,
    ) -> List[FramePosition]:
        """Extract position history for an entity.

        Args:
            tracks: Full track dictionary.
            entity_type: Type of entity ("players", "goalkeepers", "ball").
            track_id: Track ID for players/goalkeepers (ignored for ball).

        Returns:
            List of FramePosition for frames where entity is present.
        """
        positions = []
        entity_tracks = tracks.get(entity_type, [])

        for frame_idx, frame_data in enumerate(entity_tracks):
            # Ball uses track_id=1, players use their track_id
            key = 1 if entity_type == "ball" else track_id

            if entity_type == "ball":
                if key not in frame_data:
                    continue
                info = frame_data[key]
            else:
                if track_id is None or track_id not in frame_data:
                    continue
                info = frame_data[track_id]

            bbox = info.get("bbox")
            if bbox is None:
                continue

            # Ball uses center, players use foot position
            if entity_type == "ball":
                pos = get_center_of_bbox(bbox)
            else:
                pos = get_foot_position(bbox)

            positions.append(FramePosition(
                frame_idx=frame_idx,
                pixel_pos=pos,
                pitch_pos=None,  # Set later if homography available
                timestamp_sec=frame_idx / self.fps,
            ))

        return positions

    def transform_to_pitch_coords(
        self,
        positions: List[FramePosition],
        transformer: Optional[ViewTransformer],
    ) -> List[FramePosition]:
        """Transform pixel positions to pitch coordinates.

        Args:
            positions: List of FramePosition with pixel_pos.
            transformer: ViewTransformer for homography. None skips transformation.

        Returns:
            Same list with pitch_pos populated where possible.
        """
        if transformer is None:
            return positions

        for pos in positions:
            try:
                pixel_array = np.array([pos.pixel_pos], dtype=np.float32)
                pitch_array = transformer.transform_points(pixel_array)
                pos.pitch_pos = tuple(pitch_array[0])
            except Exception:
                pos.pitch_pos = None

        return positions

    def transform_to_pitch_coords_per_frame(
        self,
        positions: List[FramePosition],
        per_frame_transformers: Dict[int, ViewTransformer],
    ) -> List[FramePosition]:
        """Transform pixel positions using per-frame homographies.

        Each position uses the transformer for its own frame index,
        allowing accuracy even when the camera is panning.

        Args:
            positions: List of FramePosition with pixel_pos.
            per_frame_transformers: Mapping frame_idx → ViewTransformer.

        Returns:
            Same list with pitch_pos populated where a transformer exists.
        """
        for pos in positions:
            transformer = per_frame_transformers.get(pos.frame_idx)
            if transformer is None:
                continue
            try:
                pixel_array = np.array([pos.pixel_pos], dtype=np.float32)
                pitch_array = transformer.transform_points(pixel_array)
                pos.pitch_pos = tuple(pitch_array[0])
            except Exception:
                pos.pitch_pos = None
        return positions

    def compute_distances_and_speeds_adaptive(
        self,
        positions: List[FramePosition],
    ) -> Tuple[List[float], List[float], List[Optional[float]], List[Optional[float]]]:
        """Compute distances/speeds using pitch coords where available.

        Unlike ``compute_distances_and_speeds`` which requires *all* positions
        to have pitch coords (all-or-nothing), this method produces real-world
        values for each segment that has both endpoints in pitch space, and
        ``None`` for segments missing one or both.

        Returns:
            (distances_px, speeds_px, distances_m_or_none, speeds_m_per_sec_or_none)
            The last two lists contain float or None per segment.
        """
        if len(positions) < 2:
            return [], [], [], []

        distances_px: List[float] = []
        speeds_px: List[float] = []
        distances_m: List[Optional[float]] = []
        speeds_m: List[Optional[float]] = []

        for i in range(1, len(positions)):
            prev = positions[i - 1]
            curr = positions[i]

            d_px = measure_distance(prev.pixel_pos, curr.pixel_pos)
            distances_px.append(d_px)

            frame_gap = max(1, curr.frame_idx - prev.frame_idx)
            speeds_px.append(d_px / frame_gap)

            if prev.pitch_pos is not None and curr.pitch_pos is not None:
                d_cm = measure_distance(prev.pitch_pos, curr.pitch_pos)
                d_m = d_cm * CM_TO_M
                time_sec = frame_gap / self.fps
                distances_m.append(d_m)
                speeds_m.append(d_m / time_sec)
            else:
                distances_m.append(None)
                speeds_m.append(None)

        return distances_px, speeds_px, distances_m, speeds_m

    @staticmethod
    def _interpolate_gaps(values: List[Optional[float]]) -> List[float]:
        """Fill None gaps via linear interpolation; edge Nones use nearest value."""
        n = len(values)
        if n == 0:
            return []

        result = list(values)

        # Find indices of valid (non-None) values
        valid_indices = [i for i, v in enumerate(result) if v is not None]
        if not valid_indices:
            return [0.0] * n

        # Fill leading Nones
        first_valid = valid_indices[0]
        for i in range(first_valid):
            result[i] = result[first_valid]

        # Fill trailing Nones
        last_valid = valid_indices[-1]
        for i in range(last_valid + 1, n):
            result[i] = result[last_valid]

        # Interpolate interior gaps
        for k in range(len(valid_indices) - 1):
            left = valid_indices[k]
            right = valid_indices[k + 1]
            if right - left > 1:
                v_left = result[left]
                v_right = result[right]
                for j in range(left + 1, right):
                    t = (j - left) / (right - left)
                    result[j] = v_left + t * (v_right - v_left)

        return result  # type: ignore[return-value]

    def compute_distances_and_speeds(
        self,
        positions: List[FramePosition],
    ) -> Tuple[List[float], List[float], Optional[List[float]], Optional[List[float]]]:
        """Compute frame-to-frame distances and speeds.

        Args:
            positions: List of FramePosition.

        Returns:
            Tuple of (distances_px, speeds_px_per_frame, distances_m, speeds_m_per_sec).
            Real-world values are None if no pitch coordinates available.
        """
        if len(positions) < 2:
            return [], [], None, None

        distances_px = []
        speeds_px = []
        distances_m = []
        speeds_m_per_sec = []

        has_pitch = all(p.pitch_pos is not None for p in positions)

        for i in range(1, len(positions)):
            prev = positions[i - 1]
            curr = positions[i]

            # Pixel distance
            d_px = measure_distance(prev.pixel_pos, curr.pixel_pos)
            distances_px.append(d_px)

            # Frame gap (handles missing frames due to filtering)
            frame_gap = max(1, curr.frame_idx - prev.frame_idx)
            speeds_px.append(d_px / frame_gap)

            # Real-world distance (if available)
            if has_pitch and prev.pitch_pos and curr.pitch_pos:
                d_cm = measure_distance(prev.pitch_pos, curr.pitch_pos)
                d_m = d_cm * CM_TO_M
                distances_m.append(d_m)

                # Speed: distance / time
                time_sec = frame_gap / self.fps
                speeds_m_per_sec.append(d_m / time_sec)

        return (
            distances_px,
            speeds_px,
            distances_m if has_pitch else None,
            speeds_m_per_sec if has_pitch else None,
        )

    def compute_stats(
        self,
        positions: List[FramePosition],
        track_id: int,
        entity_type: str,
        team_id: Optional[int] = None,
    ) -> KinematicStats:
        """Compute full kinematic statistics for an entity.

        Args:
            positions: List of FramePosition.
            track_id: Entity track ID.
            entity_type: "player", "goalkeeper", or "ball".
            team_id: Team ID for players (optional).

        Returns:
            KinematicStats with all metrics.
        """
        if len(positions) < 2:
            return KinematicStats(
                track_id=track_id,
                entity_type=entity_type,
                team_id=team_id,
                total_distance_px=0.0,
                total_distance_m=None,
                speeds_px_per_frame=[],
                speeds_m_per_sec=None,
                avg_speed_px=0.0,
                avg_speed_m_per_sec=None,
                max_speed_px=0.0,
                max_speed_m_per_sec=None,
            )

        dist_px, speeds_px, dist_m, speeds_m = self.compute_distances_and_speeds(positions)

        return KinematicStats(
            track_id=track_id,
            entity_type=entity_type,
            team_id=team_id,
            total_distance_px=sum(dist_px) if dist_px else 0.0,
            total_distance_m=sum(dist_m) if dist_m else None,
            speeds_px_per_frame=speeds_px,
            speeds_m_per_sec=speeds_m,
            avg_speed_px=float(np.mean(speeds_px)) if speeds_px else 0.0,
            avg_speed_m_per_sec=float(np.mean(speeds_m)) if speeds_m else None,
            max_speed_px=float(max(speeds_px)) if speeds_px else 0.0,
            max_speed_m_per_sec=float(max(speeds_m)) if speeds_m else None,
        )

    def compute_all_player_stats(
        self,
        tracks: Dict[str, List[Dict]],
        transformer: Optional[ViewTransformer] = None,
    ) -> Dict[int, KinematicStats]:
        """Compute kinematics for all players and goalkeepers.

        Args:
            tracks: Full track dictionary.
            transformer: ViewTransformer for real-world coordinates.

        Returns:
            Dictionary mapping track_id to KinematicStats.
        """
        all_stats = {}

        # Get all unique track IDs from players and goalkeepers
        track_ids = set()
        for entity_type in ["players", "goalkeepers"]:
            for frame_data in tracks.get(entity_type, []):
                track_ids.update(frame_data.keys())

        for track_id in track_ids:
            # Try players first, then goalkeepers
            for entity_type in ["players", "goalkeepers"]:
                positions = self.extract_positions(tracks, entity_type, track_id)
                if positions:
                    positions = self.transform_to_pitch_coords(positions, transformer)

                    # Get team_id from first available frame
                    team_id = None
                    for frame_data in tracks.get(entity_type, []):
                        if track_id in frame_data:
                            team_id = frame_data[track_id].get("team_id")
                            if team_id is not None:
                                break

                    stats = self.compute_stats(
                        positions, track_id, entity_type.rstrip("s"), team_id
                    )
                    all_stats[track_id] = stats
                    break

        return all_stats

    def compute_ball_stats(
        self,
        tracks: Dict[str, List[Dict]],
        transformer: Optional[ViewTransformer] = None,
    ) -> KinematicStats:
        """Compute kinematics for the ball.

        Args:
            tracks: Full track dictionary.
            transformer: ViewTransformer for real-world coordinates.

        Returns:
            KinematicStats for the ball.
        """
        positions = self.extract_positions(tracks, "ball")
        positions = self.transform_to_pitch_coords(positions, transformer)
        return self.compute_stats(positions, track_id=1, entity_type="ball")


    def build_per_frame_lookup(
        self,
        tracks: Dict[str, List[Dict]],
        transformer: Optional[ViewTransformer] = None,
        per_frame_transformers: Optional[Dict[int, ViewTransformer]] = None,
        smooth_window: int = 5,
    ) -> Dict[int, Dict[int, Tuple[float, float]]]:
        """Build per-frame speed/distance lookup for video annotation.

        Args:
            tracks: Full track dictionary.
            transformer: Single ViewTransformer applied to all frames (legacy).
            per_frame_transformers: Per-frame ViewTransformers (preferred).
                When provided, uses adaptive pipeline with gap interpolation.
            smooth_window: Number of frames for speed smoothing window.

        Returns:
            {track_id: {frame_idx: (speed_kmh, cumulative_distance_m)}}
            Speed is smoothed over ``smooth_window`` frames.
            When no homography, uses pixel-based estimates (px/frame → rough km/h).
        """
        lookup: Dict[int, Dict[int, Tuple[float, float]]] = {}

        for entity_type in ["players", "goalkeepers"]:
            track_ids: set[int] = set()
            for frame_data in tracks.get(entity_type, []):
                track_ids.update(frame_data.keys())

            for track_id in track_ids:
                positions = self.extract_positions(tracks, entity_type, track_id)
                if len(positions) < 2:
                    continue

                # --- Adaptive per-frame pipeline ---
                if per_frame_transformers:
                    positions = self.transform_to_pitch_coords_per_frame(
                        positions, per_frame_transformers,
                    )
                    dist_px, speeds_px, dist_m_opt, speeds_m_opt = (
                        self.compute_distances_and_speeds_adaptive(positions)
                    )

                    # Count how many segments have real-world data
                    real_count = sum(1 for s in speeds_m_opt if s is not None)
                    use_real = real_count >= len(speeds_m_opt) * 0.5

                    if use_real:
                        speeds_m_filled = self._interpolate_gaps(speeds_m_opt)
                        dist_m_filled = self._interpolate_gaps(dist_m_opt)
                        raw_speeds = speeds_m_filled
                        raw_dists = dist_m_filled
                    else:
                        raw_speeds = speeds_px
                        raw_dists = dist_px
                        use_real = False
                else:
                    # Legacy single-transformer path
                    positions = self.transform_to_pitch_coords(positions, transformer)
                    dist_px, speeds_px, dist_m, speeds_m = (
                        self.compute_distances_and_speeds(positions)
                    )
                    use_real = speeds_m is not None and len(speeds_m) > 0
                    raw_speeds = speeds_m if use_real else speeds_px
                    raw_dists = dist_m if use_real else dist_px

                # Smooth speeds over a window
                smoothed = []
                for i in range(len(raw_speeds)):
                    start = max(0, i - smooth_window // 2)
                    end = min(len(raw_speeds), i + smooth_window // 2 + 1)
                    smoothed.append(float(np.mean(raw_speeds[start:end])))

                # Build frame-indexed lookup
                per_frame: Dict[int, Tuple[float, float]] = {}
                cumulative = 0.0
                for i in range(len(smoothed)):
                    frame_idx = positions[i + 1].frame_idx
                    cumulative += raw_dists[i]

                    if use_real:
                        speed_kmh = min(smoothed[i] * 3.6, MAX_PLAYER_SPEED_KMH)
                        dist_total = cumulative
                    else:
                        speed_kmh = smoothed[i] * self.fps * 0.05
                        dist_total = cumulative * 0.05

                    per_frame[frame_idx] = (speed_kmh, dist_total)

                lookup[track_id] = per_frame

        return lookup


def compute_kinematics(
    tracks: Dict[str, List[Dict]],
    fps: float = 25.0,
    transformer: Optional[ViewTransformer] = None,
) -> Tuple[Dict[int, KinematicStats], KinematicStats]:
    """Convenience function to compute all kinematics.

    Args:
        tracks: Full track dictionary.
        fps: Video frame rate.
        transformer: ViewTransformer for real-world coordinates.

    Returns:
        Tuple of (player_stats_dict, ball_stats).
    """
    calculator = KinematicsCalculator(fps=fps)
    player_stats = calculator.compute_all_player_stats(tracks, transformer)
    ball_stats = calculator.compute_ball_stats(tracks, transformer)
    return player_stats, ball_stats
