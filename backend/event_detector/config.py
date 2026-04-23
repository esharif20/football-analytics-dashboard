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
    window_seconds: float = 1.5       # §4.8: 1.5 s deployment window (38 frames @ 25 fps)
    stride_seconds: float = 0.5       # Hop between windows
    n_frames_per_window: int = 38     # §4.8: 1.5 s × 25 fps = 37.5 → 38 frames
    n_triplets: int = 5

    # ----- Model architecture -----
    img_size: int = 1024
    num_classes: int = 4
    class_names: Tuple[str, ...] = ("background", "challenge", "play", "throwin")

    # Class-balanced weights from validation tuning (background, challenge, play, throwin).
    # These are multiplied against sigmoid probabilities before argmax, not thresholds.
    best_w: Tuple[float, ...] = (1.5, 5.0, 4.0, 4.0)

    # ----- Per-class detection thresholds (tuned on all-7 ground truth, ±2s) -----
    # background, challenge, play, throwin
    # Test7+Test1 grid: challenge 0.45→0.60, throwin 0.50→0.80 → F1=0.688 (P=R=0.688)
    # All-7 re-tune:  throwin 0.80→0.95 → micro F1 0.741→0.782 (+4.1%), macro 0.728→0.766
    # challenge R=0.286 is a model limitation (5/7 GT challenges classified as play/bg)
    detection_thresholds: Tuple[float, ...] = (0.50, 0.60, 0.55, 0.95)

    # ----- Per-class temporal NMS windows (§4.8 deployment values) -----
    # background (unused), challenge, play, throwin
    nms_window_seconds_per_class: Tuple[float, ...] = (2.0, 2.0, 2.5, 3.0)

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
    # Caller checks this; stored on config for convenience.
    cfg._enabled = enabled_env.lower() not in ("0", "false", "no")  # type: ignore[attr-defined]

    batch_env = os.environ.get("EVENT_BATCH_SIZE")
    if batch_env and batch_env.isdigit():
        cfg.batch_size = int(batch_env)

    return cfg
