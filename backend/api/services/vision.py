"""Vision utilities for extracting video keyframes.

Ported from backend/evaluation/vlm_comparison.py — proven in evaluation
to boost grounding quality from 61.5% (text-only) to 90.9% (annotated frames).
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def extract_keyframes(video_path: str, n_frames: int = 5) -> list:
    """Extract uniformly-spaced keyframes from video.

    Args:
        video_path: Absolute path to the video file.
        n_frames: Number of keyframes to extract (default 5).

    Returns:
        List of numpy arrays (BGR frames). Empty list on failure.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Could not open video for frame extraction: %s", video_path)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        logger.warning("Video has no frames: %s", video_path)
        return []

    # Avoid very first and very last frames (often black/transition)
    start = max(0, int(total * 0.05))
    end = min(total - 1, int(total * 0.95))
    indices = np.linspace(start, end, min(n_frames, total), dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)

    cap.release()
    logger.info("Extracted %d keyframes from %s", len(frames), video_path)
    return frames


def frames_to_jpeg_bytes(frames: list) -> list[bytes]:
    """Convert numpy BGR frames to JPEG bytes for LLM vision APIs.

    Args:
        frames: List of numpy arrays from extract_keyframes().

    Returns:
        List of JPEG-encoded byte strings.
    """
    import cv2

    result = []
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            result.append(bytes(buf))
    return result


def resolve_local_path(url: str, storage_dir: str) -> str | None:
    """Convert a /uploads/ URL or raw path to an absolute local filesystem path.

    Handles:
    - Full HTTP/HTTPS URLs (e.g. from ngrok/RunPod) — path component is extracted
    - /uploads/-relative paths
    - Absolute filesystem paths

    Returns None if the file does not exist locally.
    """
    if not url:
        return None

    path_str = url.strip()

    # Strip HTTP/HTTPS prefix — extract just the path component so that
    # URLs like https://host.ngrok-free.dev/uploads/videos/foo.mp4 resolve correctly.
    if path_str.startswith(("http://", "https://")):
        from urllib.parse import urlparse

        path_str = urlparse(path_str).path

    if path_str.startswith("/uploads/"):
        relative = path_str.replace("/uploads/", "", 1)
        resolved = Path(storage_dir).resolve() / relative
        if resolved.exists():
            return str(resolved)
        logger.debug("Vision file not found at resolved path: %s", resolved)
        return None

    file_path = Path(path_str)
    if file_path.exists():
        return str(file_path)

    logger.debug("Vision file not found: %s", path_str)
    return None
