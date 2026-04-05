"""Configuration for the ML event detection module."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch


@dataclass
class SlidingWindowConfig:
    """All tunable parameters for sliding-window event inference.

    Thresholds and best_w were tuned on the DFL validation set with
    checkpoint effnet_bce-epoch=19-val_loss=0.261.ckpt.
    """

    # ----- Model -----
    # Path to the .ckpt file. Defaults to models/event_detection.ckpt
    # relative to the pipeline root (backend/pipeline/).
    checkpoint_path: str = str(
        Path(__file__).resolve().parents[1] / "pipeline" / "models" / "event_detection.ckpt"
    )

    # ----- Sliding window -----
    window_seconds: float = 2.0       # Matches training clip length
    stride_seconds: float = 0.5       # Hop between windows
    n_frames_per_window: int = 15     # 5 triplets × 3 consecutive frames
    n_triplets: int = 5

    # ----- Model architecture -----
    img_size: int = 1024
    num_classes: int = 4
    class_names: Tuple[str, ...] = ("background", "challenge", "play", "throwin")

    # Class-balanced weights from validation tuning (background, challenge, play, throwin).
    # These are multiplied against sigmoid probabilities before argmax, not thresholds.
    best_w: Tuple[float, ...] = (1.5, 5.0, 4.0, 4.0)

    # ----- Per-class detection thresholds (tuned, final pass) -----
    # background, challenge, play, throwin
    detection_thresholds: Tuple[float, ...] = (0.50, 0.40, 0.55, 0.35)

    # ----- Temporal NMS -----
    nms_window_seconds: float = 2.0

    # ----- Batching -----
    # Number of windows to stack into a single GPU forward pass.
    batch_size: int = 8

    # ----- Hardware -----
    device: str = field(
        default_factory=lambda: os.environ.get(
            "EVENT_DETECTOR_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    fp16: bool = True


def config_from_env(base: SlidingWindowConfig | None = None) -> SlidingWindowConfig:
    """Return a config with env-var overrides applied to *base* (or defaults)."""
    cfg = base or SlidingWindowConfig()

    ckpt_env = os.environ.get("EVENT_MODEL_PATH")
    if ckpt_env:
        cfg.checkpoint_path = ckpt_env

    enabled_env = os.environ.get("EVENT_DETECTION_ENABLED", "1")
    # Caller checks this; stored on config for convenience
    cfg._enabled = enabled_env.lower() not in ("0", "false", "no")  # type: ignore[attr-defined]

    batch_env = os.environ.get("EVENT_BATCH_SIZE")
    if batch_env and batch_env.isdigit():
        cfg.batch_size = int(batch_env)

    return cfg
