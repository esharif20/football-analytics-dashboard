"""Shared pipeline utilities."""

from pathlib import Path
from typing import List, TYPE_CHECKING

import numpy as np

from utils.logging_config import get_logger

logger = get_logger("base")

from config import (
    PLAYER_DETECTION_MODEL_PATH,
    BALL_DETECTION_MODEL_PATH,
    IMG_SIZE,
    CONF_THRESHOLD,
    NMS_IOU,
    MAX_DET,
    BALL_CLASS_ID,
    PAD_BALL,
    PLAYER_MODEL_SOURCE,
    BALL_MODEL_SOURCE,
    YOLOV8_PLAYER_MODEL,
    YOLOV8_BALL_MODEL,
)
from utils.video_utils import read_video
from utils.cache import stub_path
from trackers.tracker import Tracker, TrackerConfig
from trackers.ball_config import BallConfig

# Re-export for backward compatibility
get_stub_path = stub_path


def load_frames(source_video_path: str) -> List[np.ndarray]:
    """Load video frames from file.

    Args:
        source_video_path: Path to video file

    Returns:
        List of video frames
    """
    logger.info(f"Loading video: {source_video_path}")
    frames = read_video(source_video_path)
    logger.info(f"Loaded {len(frames)} frames")
    return frames


def build_tracker(
    device: str | None,
    det_batch_size: int,
    use_ball_model: bool,
    fast_ball: bool,
    ball_config: BallConfig,
    use_ball_model_weights: bool,
    player_model_source: str | None = None,
    ball_model_source: str | None = None,
) -> Tracker:
    """Build tracker with configuration.

    Args:
        device: Device for inference (cpu, cuda, mps)
        det_batch_size: Detection batch size (0=auto)
        use_ball_model: Whether to use dedicated ball model
        fast_ball: Disable slicer for speed
        ball_config: Ball tracking configuration
        use_ball_model_weights: Whether to use dedicated ball model weights

    Returns:
        Configured Tracker instance
    """
    # Determine player model path based on source (prefer TensorRT .engine)
    p_source = player_model_source or PLAYER_MODEL_SOURCE
    if p_source == "custom" and PLAYER_DETECTION_MODEL_PATH.exists():
        engine_path = PLAYER_DETECTION_MODEL_PATH.with_suffix('.engine')
        player_model_path = str(engine_path) if engine_path.exists() else str(PLAYER_DETECTION_MODEL_PATH)
    else:
        player_model_path = YOLOV8_PLAYER_MODEL

    # Determine ball model path based on source (prefer TensorRT .engine)
    b_source = ball_model_source or BALL_MODEL_SOURCE
    ball_model_path = None
    if use_ball_model and use_ball_model_weights:
        if b_source == "custom" and BALL_DETECTION_MODEL_PATH.exists():
            engine_path = BALL_DETECTION_MODEL_PATH.with_suffix('.engine')
            ball_model_path = str(engine_path) if engine_path.exists() else str(BALL_DETECTION_MODEL_PATH)
        elif b_source == "yolov8":
            ball_model_path = None  # Falls back to multi-class detection

    config = TrackerConfig(
        det_batch_size=det_batch_size,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        nms=NMS_IOU,
        max_det=MAX_DET,
        ball_id=BALL_CLASS_ID,
        pad_ball=PAD_BALL,
        ball_model_path=ball_model_path,
        ball_config=ball_config,  # Pass BallConfig directly
        ball_use_slicer=not fast_ball,  # Slicer enable/disable still external
    )
    tracker = Tracker(model_path=player_model_path, config=config, device=device)

    if use_ball_model:
        if ball_model_path is None:
            logger.info("Ball model: fallback to multi-class model")
        else:
            logger.info(f"Ball model: {ball_model_path}")
            logger.info(f"Ball conf: {ball_config.conf}")

    return tracker
