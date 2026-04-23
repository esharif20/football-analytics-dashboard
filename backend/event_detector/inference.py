"""Sliding-window inference for the ML event detection model."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from .config import SlidingWindowConfig
from .model import EfficientNetBCE_v3
from .postprocess import (
    compute_frame_numbers,
    merge_nearby_events,
    temporal_nms,
    to_football_events,
)

logger = logging.getLogger(__name__)

# ImageNet statistics used during training — must not change.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_window(
    frames: List[np.ndarray],
    img_size: int,
    n_triplets: int,
) -> np.ndarray:
    """Convert *n_triplets × 3* BGR frames to a (T, 3, H, W) float32 array.

    Preprocessing is identical to training:
    - Convert BGR → grayscale
    - Resize to img_size × img_size
    - Normalise with ImageNet mean/std (applied per-channel to grayscale)

    Returns shape: (n_triplets, 3, img_size, img_size)
    """
    processed: List[np.ndarray] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        gray = gray.astype(np.float32) / 255.0
        processed.append(gray)

    triplets: List[np.ndarray] = []
    for t in range(n_triplets):
        base = t * 3
        stack = np.stack(processed[base : base + 3], axis=0)  # (3, H, W)
        for c in range(3):
            stack[c] = (stack[c] - _MEAN[c]) / _STD[c]
        triplets.append(stack)

    return np.stack(triplets, axis=0)  # (T, 3, H, W)


class EventModelInference:
    """High-level interface for sliding-window football event detection.

    Usage::

        inf = EventModelInference.from_config(cfg)
        if inf is None:
            # checkpoint missing — skip gracefully
            return []
        events = inf.detect_events(frames, fps)
    """

    def __init__(self, model: EfficientNetBCE_v3, cfg: SlidingWindowConfig) -> None:
        self._model = model
        self._cfg = cfg
        self._device = cfg.device
        self._fp16 = cfg.fp16 and cfg.device == "cuda"

        self._best_w = torch.tensor(
            cfg.best_w,
            device=self._device,
            dtype=torch.float16 if self._fp16 else torch.float32,
        )

    @classmethod
    def from_config(cls, cfg: SlidingWindowConfig) -> Optional["EventModelInference"]:
        """Load model from checkpoint. Returns None if unavailable."""
        model = EfficientNetBCE_v3.load_checkpoint(cfg.checkpoint_path, device="cpu")
        if model is None:
            return None
        model = model.to(cfg.device).eval()
        if cfg.fp16 and cfg.device == "cuda":
            model = model.half()
        return cls(model, cfg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_window_tensor(
        self, all_frames: List[np.ndarray], start_frame: int, end_frame: int
    ) -> np.ndarray:
        """Sample n_frames_per_window frames from [start, end) and preprocess."""
        indices = np.linspace(
            start_frame,
            end_frame - 1,
            self._cfg.n_frames_per_window,
            dtype=int,
        )
        frames = [all_frames[i] for i in indices]
        return _preprocess_window(frames, self._cfg.img_size, self._cfg.n_triplets)

    @torch.no_grad()
    def _run_batch(self, windows: List[np.ndarray]) -> torch.Tensor:
        """Forward pass for a batch of pre-processed windows.

        Args:
            windows: List of (T, 3, H, W) float32 arrays.

        Returns:
            Sigmoid probabilities, shape (N, num_classes), on CPU float32.
        """
        arr = np.stack(windows, axis=0)  # (N, T, 3, H, W)
        x = torch.from_numpy(arr).to(self._device)
        if self._fp16:
            x = x.half()
        logits = self._model(x)           # (N, 4)
        probs = torch.sigmoid(logits)
        return probs.float().cpu()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_events(
        self,
        frames: List[np.ndarray],
        fps: float,
    ) -> List[Dict]:
        """Run sliding-window inference and return formatted event dicts.

        Args:
            frames: Pre-loaded BGR frames (the same list already in pipeline memory).
            fps: Video frame rate.

        Returns:
            List of event dicts compatible with FootballEvent serialisation format.
            Events with class "background" are filtered out.
            Empty list if no events detected.
        """
        cfg = self._cfg
        total_frames = len(frames)
        duration = total_frames / fps

        starts = np.arange(0, duration - cfg.window_seconds, cfg.stride_seconds)
        if len(starts) == 0:
            logger.debug("Video too short for sliding-window inference (%.1fs)", duration)
            return []

        logger.info(
            "Event detection: %.1fs clip, %d windows (stride=%.1fs, window=%.1fs) | "
            "thresholds=%s | nms_per_class=%s",
            duration,
            len(starts),
            cfg.stride_seconds,
            cfg.window_seconds,
            dict(zip(cfg.class_names, cfg.detection_thresholds)),
            dict(zip(cfg.class_names, cfg.nms_window_seconds_per_class)),
        )

        raw_detections: List[Dict] = []
        pending_windows: List[np.ndarray] = []
        pending_meta: List[float] = []  # start_sec for each pending window

        def _flush(windows: List[np.ndarray], metas: List[float]) -> None:
            probs_batch = self._run_batch(windows)           # (N, 4)
            weighted_batch = probs_batch * self._best_w.cpu()
            for i, start_sec in enumerate(metas):
                probs = probs_batch[i]
                weighted = weighted_batch[i]
                pred_idx = int(weighted.argmax())
                ts = float(start_sec + cfg.window_seconds / 2)
                for cls_idx in range(cfg.num_classes):
                    conf = float(probs[cls_idx])
                    if conf >= cfg.detection_thresholds[cls_idx]:
                        raw_detections.append(
                            {
                                "timestamp": ts,
                                "class_idx": cls_idx,
                                "class_name": cfg.class_names[cls_idx],
                                "confidence": conf,
                                "pred_class": cfg.class_names[pred_idx],
                                "weighted_score": float(weighted[cls_idx]),
                            }
                        )

        n_total = len(starts)
        logged_pct = -1
        for wi, start_sec in enumerate(starts):
            end_sec = start_sec + cfg.window_seconds
            if end_sec > duration:
                break

            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            window_arr = self._build_window_tensor(frames, start_frame, end_frame)
            pending_windows.append(window_arr)
            pending_meta.append(float(start_sec))

            if len(pending_windows) >= cfg.batch_size:
                _flush(pending_windows, pending_meta)
                pending_windows, pending_meta = [], []

            pct = int(wi / n_total * 100)
            if pct // 10 > logged_pct // 10:
                logger.info("Event detection: %d%%", pct)
                logged_pct = pct

        if pending_windows:
            _flush(pending_windows, pending_meta)

        nms_results = temporal_nms(raw_detections, cfg.nms_window_seconds_per_class)
        events = [d for d in nms_results if d["class_name"] != "background"]
        events = merge_nearby_events(events, merge_window_seconds=1.0)
        events = compute_frame_numbers(events, fps)

        logger.info(
            "Event detection complete: %d events detected (%s)",
            len(events),
            ", ".join(
                f"{cls}:{sum(1 for e in events if e['class_name']==cls)}"
                for cls in cfg.class_names[1:]  # skip background
            ),
        )

        return to_football_events(events, fps)
