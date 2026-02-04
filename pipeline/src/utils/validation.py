"""Runtime validation and sanity checks for the football analysis pipeline.

This module provides fail-fast validation for:
- Input video readability and properties
- Track structure invariants
- Homography validity
- Team clustering validity
- Coordinate frame consistency

Usage:
    from utils.validation import (
        validate_video_input,
        validate_tracks,
        validate_homography,
        validate_team_clustering,
    )

    # At pipeline start
    validate_video_input(video_path)

    # After tracking
    validate_tracks(tracks)

    # After homography computation
    validate_homography(transformer, frame_keypoints, pitch_keypoints)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class ValidationError(Exception):
    """Base exception for validation failures."""
    pass


class VideoValidationError(ValidationError):
    """Video input validation failed."""
    pass


class TrackValidationError(ValidationError):
    """Track structure validation failed."""
    pass


class HomographyValidationError(ValidationError):
    """Homography validation failed."""
    pass


class TeamClusteringError(ValidationError):
    """Team clustering validation failed."""
    pass


# =============================================================================
# Video Validation
# =============================================================================

def validate_video_input(
    video_path: str | Path,
    min_fps: float = 1.0,
    min_frames: int = 1,
    min_resolution: Tuple[int, int] = (64, 64),
) -> Dict[str, Any]:
    """Validate that a video file is readable and has valid properties.

    Args:
        video_path: Path to video file
        min_fps: Minimum acceptable FPS
        min_frames: Minimum number of frames
        min_resolution: Minimum (width, height)

    Returns:
        Dict with video properties (fps, frame_count, width, height, duration)

    Raises:
        VideoValidationError: If validation fails
    """
    video_path = Path(video_path)

    # Check file exists
    if not video_path.exists():
        raise VideoValidationError(f"Video file not found: {video_path}")

    # Check file is not empty
    if video_path.stat().st_size == 0:
        raise VideoValidationError(f"Video file is empty: {video_path}")

    # Try to open with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise VideoValidationError(f"Failed to open video: {video_path}")

    try:
        # Get properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Validate FPS
        if fps <= 0 or fps < min_fps:
            raise VideoValidationError(
                f"Invalid FPS: {fps} (minimum: {min_fps})"
            )

        # Validate frame count
        if frame_count <= 0 or frame_count < min_frames:
            raise VideoValidationError(
                f"Invalid frame count: {frame_count} (minimum: {min_frames})"
            )

        # Validate resolution
        if width < min_resolution[0] or height < min_resolution[1]:
            raise VideoValidationError(
                f"Resolution too small: {width}x{height} (minimum: {min_resolution[0]}x{min_resolution[1]})"
            )

        # Try to read first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            raise VideoValidationError("Failed to read first frame from video")

        # Verify frame shape matches reported dimensions
        if frame.shape[1] != width or frame.shape[0] != height:
            logger.warning(
                f"Frame shape mismatch: reported {width}x{height}, actual {frame.shape[1]}x{frame.shape[0]}"
            )

        duration = frame_count / fps

        logger.info(
            f"Video validated: {frame_count} frames, {fps:.2f} fps, "
            f"{width}x{height}, {duration:.2f}s"
        )

        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "path": str(video_path),
        }

    finally:
        cap.release()


# =============================================================================
# Track Validation
# =============================================================================

def validate_tracks(
    tracks: Dict[str, List[Dict]],
    num_frames: Optional[int] = None,
    require_ball: bool = False,
) -> Dict[str, Any]:
    """Validate track structure invariants.

    Expected structure:
        tracks["players"][frame_idx][track_id] = {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "team_id": int (optional),
        }
        tracks["ball"][frame_idx][track_id] = {
            "bbox": [x1, y1, x2, y2],
            "confidence": float (optional),
        }

    Args:
        tracks: Track dictionary from tracker
        num_frames: Expected number of frames (if known)
        require_ball: Whether ball must be detected in at least some frames

    Returns:
        Dict with track statistics

    Raises:
        TrackValidationError: If validation fails
    """
    required_keys = ["players", "ball"]
    optional_keys = ["goalkeepers", "referees"]

    # Check required keys
    for key in required_keys:
        if key not in tracks:
            raise TrackValidationError(f"Missing required track key: {key}")

    # Check list lengths match
    players_len = len(tracks["players"])
    ball_len = len(tracks["ball"])

    if players_len != ball_len:
        raise TrackValidationError(
            f"Track length mismatch: players={players_len}, ball={ball_len}"
        )

    if num_frames is not None and players_len != num_frames:
        raise TrackValidationError(
            f"Track length {players_len} doesn't match expected frames {num_frames}"
        )

    # Validate structure of each frame
    stats = {
        "num_frames": players_len,
        "total_player_detections": 0,
        "total_ball_detections": 0,
        "frames_with_ball": 0,
        "unique_player_ids": set(),
        "frames_with_team_assignment": 0,
    }

    for frame_idx in range(players_len):
        # Validate players
        players_frame = tracks["players"][frame_idx]
        if not isinstance(players_frame, dict):
            raise TrackValidationError(
                f"Frame {frame_idx}: players must be dict, got {type(players_frame)}"
            )

        for track_id, track_data in players_frame.items():
            _validate_track_entry(track_data, frame_idx, track_id, "player")
            stats["total_player_detections"] += 1
            stats["unique_player_ids"].add(track_id)

            if "team_id" in track_data:
                stats["frames_with_team_assignment"] += 1

        # Validate ball
        ball_frame = tracks["ball"][frame_idx]
        if not isinstance(ball_frame, dict):
            raise TrackValidationError(
                f"Frame {frame_idx}: ball must be dict, got {type(ball_frame)}"
            )

        if ball_frame:
            stats["frames_with_ball"] += 1
            for track_id, track_data in ball_frame.items():
                _validate_track_entry(track_data, frame_idx, track_id, "ball")
                stats["total_ball_detections"] += 1

    # Check ball detection rate
    if require_ball and stats["frames_with_ball"] == 0:
        raise TrackValidationError("No ball detected in any frame")

    stats["unique_player_ids"] = len(stats["unique_player_ids"])
    ball_rate = stats["frames_with_ball"] / stats["num_frames"] * 100

    logger.info(
        f"Tracks validated: {stats['num_frames']} frames, "
        f"{stats['unique_player_ids']} unique players, "
        f"ball in {stats['frames_with_ball']} frames ({ball_rate:.1f}%)"
    )

    return stats


def _validate_track_entry(
    track_data: Dict,
    frame_idx: int,
    track_id: int,
    entity_type: str,
) -> None:
    """Validate a single track entry."""
    if not isinstance(track_data, dict):
        raise TrackValidationError(
            f"Frame {frame_idx}, {entity_type} {track_id}: "
            f"track_data must be dict, got {type(track_data)}"
        )

    # Check bbox exists and is valid
    if "bbox" not in track_data:
        raise TrackValidationError(
            f"Frame {frame_idx}, {entity_type} {track_id}: missing bbox"
        )

    bbox = track_data["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise TrackValidationError(
            f"Frame {frame_idx}, {entity_type} {track_id}: "
            f"bbox must be [x1, y1, x2, y2], got {bbox}"
        )

    x1, y1, x2, y2 = bbox
    if x2 < x1 or y2 < y1:
        raise TrackValidationError(
            f"Frame {frame_idx}, {entity_type} {track_id}: "
            f"invalid bbox (x2 < x1 or y2 < y1): {bbox}"
        )


# =============================================================================
# Homography Validation
# =============================================================================

def validate_homography(
    matrix: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    min_keypoints: int = 4,
    max_reprojection_error: float = 50.0,
) -> Dict[str, Any]:
    """Validate homography matrix quality.

    Args:
        matrix: 3x3 homography matrix
        source_points: Source keypoints (N, 2)
        target_points: Target keypoints (N, 2)
        min_keypoints: Minimum keypoints required
        max_reprojection_error: Maximum average reprojection error (pixels)

    Returns:
        Dict with validation metrics

    Raises:
        HomographyValidationError: If validation fails
    """
    # Check matrix shape
    if matrix is None:
        raise HomographyValidationError("Homography matrix is None")

    if matrix.shape != (3, 3):
        raise HomographyValidationError(
            f"Invalid matrix shape: {matrix.shape}, expected (3, 3)"
        )

    # Check for NaN/Inf
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        raise HomographyValidationError("Homography matrix contains NaN or Inf")

    # Check keypoint count
    num_keypoints = len(source_points)
    if num_keypoints < min_keypoints:
        raise HomographyValidationError(
            f"Insufficient keypoints: {num_keypoints} (minimum: {min_keypoints})"
        )

    # Compute reprojection error
    source_homogeneous = np.hstack([
        source_points,
        np.ones((num_keypoints, 1))
    ])
    projected = (matrix @ source_homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    errors = np.linalg.norm(projected - target_points, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    if mean_error > max_reprojection_error:
        raise HomographyValidationError(
            f"High reprojection error: {mean_error:.2f}px (max allowed: {max_reprojection_error}px)"
        )

    # Check determinant (should be positive for valid homography)
    det = np.linalg.det(matrix)
    if det <= 0:
        logger.warning(f"Homography determinant is non-positive: {det:.4f}")

    stats = {
        "num_keypoints": num_keypoints,
        "mean_reprojection_error": mean_error,
        "max_reprojection_error": max_error,
        "determinant": det,
    }

    logger.debug(
        f"Homography validated: {num_keypoints} keypoints, "
        f"mean error {mean_error:.2f}px, det={det:.4f}"
    )

    return stats


# =============================================================================
# Team Clustering Validation
# =============================================================================

def validate_team_clustering(
    labels: np.ndarray,
    min_cluster_size: int = 3,
    expected_clusters: int = 2,
) -> Dict[str, Any]:
    """Validate team clustering results.

    Args:
        labels: Cluster labels for each sample
        min_cluster_size: Minimum samples per cluster
        expected_clusters: Expected number of clusters (typically 2)

    Returns:
        Dict with clustering statistics

    Raises:
        TeamClusteringError: If validation fails
    """
    if labels is None or len(labels) == 0:
        raise TeamClusteringError("Empty cluster labels")

    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    if num_clusters != expected_clusters:
        raise TeamClusteringError(
            f"Expected {expected_clusters} clusters, got {num_clusters}"
        )

    cluster_sizes = {}
    for label in unique_labels:
        size = np.sum(labels == label)
        cluster_sizes[int(label)] = size

        if size < min_cluster_size:
            raise TeamClusteringError(
                f"Cluster {label} too small: {size} samples (minimum: {min_cluster_size})"
            )

    # Check for degenerate clustering (one cluster much larger than other)
    sizes = list(cluster_sizes.values())
    ratio = max(sizes) / min(sizes) if min(sizes) > 0 else float('inf')

    if ratio > 10:
        logger.warning(
            f"Imbalanced clusters: sizes {cluster_sizes}, ratio {ratio:.1f}"
        )

    stats = {
        "num_clusters": num_clusters,
        "cluster_sizes": cluster_sizes,
        "size_ratio": ratio,
        "total_samples": len(labels),
    }

    logger.info(
        f"Team clustering validated: {num_clusters} clusters, "
        f"sizes {cluster_sizes}, ratio {ratio:.2f}"
    )

    return stats


# =============================================================================
# Coordinate Frame Validation
# =============================================================================

def validate_coordinate_frame(
    positions: np.ndarray,
    frame_type: str,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """Validate that positions are in the expected coordinate frame.

    Args:
        positions: Array of (x, y) positions, shape (N, 2)
        frame_type: Either "pixel" or "pitch_meters"
        bounds: Optional (min_x, min_y, max_x, max_y) bounds

    Raises:
        ValidationError: If positions appear to be in wrong coordinate frame
    """
    if positions is None or len(positions) == 0:
        return

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValidationError(
            f"Invalid positions shape: {positions.shape}, expected (N, 2)"
        )

    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)

    if frame_type == "pixel":
        # Pixel coordinates should be positive and reasonably bounded
        if min_x < -100 or min_y < -100:
            logger.warning(
                f"Pixel coordinates have large negative values: min=({min_x:.1f}, {min_y:.1f})"
            )

        # Check if values look like pitch meters (0-105, 0-68 range)
        if 0 <= min_x <= 105 and 0 <= max_x <= 105 and 0 <= min_y <= 68 and 0 <= max_y <= 68:
            logger.warning(
                "Pixel coordinates look like pitch meters - possible coordinate frame mismatch"
            )

    elif frame_type == "pitch_meters":
        # Pitch coordinates should be within field bounds (with some margin)
        if min_x < -10 or max_x > 115 or min_y < -10 or max_y > 78:
            logger.warning(
                f"Pitch coordinates out of expected range: "
                f"x=[{min_x:.1f}, {max_x:.1f}], y=[{min_y:.1f}, {max_y:.1f}]"
            )

    if bounds:
        bound_min_x, bound_min_y, bound_max_x, bound_max_y = bounds
        if min_x < bound_min_x or max_x > bound_max_x or min_y < bound_min_y or max_y > bound_max_y:
            raise ValidationError(
                f"Positions out of bounds: [{min_x:.1f}, {min_y:.1f}] to [{max_x:.1f}, {max_y:.1f}], "
                f"expected within [{bound_min_x}, {bound_min_y}] to [{bound_max_x}, {bound_max_y}]"
            )
