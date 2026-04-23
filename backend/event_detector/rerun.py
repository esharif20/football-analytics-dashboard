"""Standalone CLI to re-run §4.8 ML event detection on existing analyses.

Two modes:

1. **API mode** (default) — fetches video URL from the dashboard API, runs
   inference, and POSTs events back.  Requires API access and WORKER_API_KEY.

       cd backend
       python -m event_detector.rerun --analysis-id 13 --analysis-id 17

2. **Offline mode** (for GPU pods without API access) — takes video path
   directly and writes events to a JSON file.  POST the JSON manually later.

       EVENT_MODEL_PATH=/workspace/pipeline/models/event_detection.ckpt \\
       PYTHONPATH=/workspace/pipeline \\
       python3 -m event_detector.rerun \\
           --analysis-id 13 \\
           --video-path /workspace/rerun_inputs/clip13.mp4 \\
           --output-json /workspace/rerun_inputs/events_13.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("event_detector.rerun")


# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULT_UPLOADS_ROOT = str(
    Path(__file__).resolve().parents[1] / "uploads"
)


def _resolve_video_path(original_url: str, uploads_root: str) -> Path:
    """Convert an API-relative URL like /uploads/videos/1/foo.mp4 to a local path."""
    rel = original_url.lstrip("/")
    return Path(uploads_root) / Path(rel).relative_to("uploads")


# ── Frame loading ─────────────────────────────────────────────────────────────

def load_frames(video_path: Path) -> tuple[List, float]:
    """Return (frames_bgr, fps).  Raises RuntimeError on failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    logger.info("Loaded %d frames @ %.2f fps from %s", len(frames), fps, video_path.name)
    return frames, fps


# ── API helpers ───────────────────────────────────────────────────────────────

def _headers(worker_key: str) -> dict:
    return {"X-Worker-Key": worker_key, "Content-Type": "application/json"}


def fetch_analysis(dashboard_url: str, analysis_id: int, worker_key: str) -> dict:
    import requests
    url = f"{dashboard_url}/api/analyses/{analysis_id}"
    r = requests.get(url, headers=_headers(worker_key), timeout=30)
    r.raise_for_status()
    return r.json()


def post_ml_events(
    dashboard_url: str,
    analysis_id: int,
    events: List[dict],
    worker_key: str,
) -> dict:
    import requests
    url = f"{dashboard_url}/api/worker/analysis/{analysis_id}/ml-events"
    payload = {"events": events}
    r = requests.post(url, headers=_headers(worker_key), data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json()


# ── Inference core ────────────────────────────────────────────────────────────

def _run_inference(video_path: Path) -> tuple[List[dict], float]:
    """Load model + video, run §4.8 sliding-window inference.

    Returns (events, fps) where events is the list of ML event dicts.
    """
    from event_detector.config import SlidingWindowConfig, config_from_env
    from event_detector.inference import EventModelInference

    cfg = config_from_env(SlidingWindowConfig())

    if not getattr(cfg, "_enabled", True):
        raise RuntimeError("EVENT_DETECTION_ENABLED=0 — aborting")

    if not Path(cfg.checkpoint_path).exists():
        raise RuntimeError(
            f"Checkpoint missing: {cfg.checkpoint_path}\n"
            "Set EVENT_MODEL_PATH env var or place the .ckpt at that path."
        )

    logger.info(
        "Config — window=%.1fs, thresholds=%s, nms_per_class=%s",
        cfg.window_seconds,
        dict(zip(cfg.class_names, cfg.detection_thresholds)),
        dict(zip(cfg.class_names, cfg.nms_window_seconds_per_class)),
    )

    frames, fps = load_frames(video_path)

    inf = EventModelInference.from_config(cfg)
    if inf is None:
        raise RuntimeError(f"Model failed to load from {cfg.checkpoint_path}")

    events = inf.detect_events(frames, fps)
    return events, fps


# ── Per-analysis runners ──────────────────────────────────────────────────────

def run_offline(
    analysis_id: int,
    video_path: Path,
    output_json: Path,
) -> None:
    """Infer-only mode — write events JSON without any API calls."""
    logger.info("=== Analysis %d (offline) ===", analysis_id)
    try:
        events, _ = _run_inference(video_path)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return

    logger.info("Detected %d ML events for analysis %d", len(events), analysis_id)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"analysis_id": analysis_id, "events": events}
    output_json.write_text(json.dumps(payload, indent=2))
    logger.info("Events written to %s", output_json)


def run_api(
    analysis_id: int,
    dashboard_url: str,
    worker_key: str,
    uploads_root: str,
) -> None:
    """End-to-end mode — fetch video via API, infer, POST events back."""
    logger.info("=== Analysis %d (API mode) ===", analysis_id)

    try:
        meta = fetch_analysis(dashboard_url, analysis_id, worker_key)
    except Exception as exc:
        logger.error("Failed to fetch analysis %d: %s", analysis_id, exc)
        return

    video_meta = meta.get("video") or {}
    original_url = video_meta.get("originalUrl") or meta.get("originalUrl")
    if not original_url:
        logger.error(
            "No originalUrl in analysis %d response: %s",
            analysis_id,
            list(meta.keys()),
        )
        return

    video_path = _resolve_video_path(original_url, uploads_root)
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        return

    try:
        events, _ = _run_inference(video_path)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return

    logger.info("Detected %d ML events for analysis %d", len(events), analysis_id)
    if not events:
        logger.warning("No events detected — nothing to store")
        return

    try:
        result = post_ml_events(dashboard_url, analysis_id, events, worker_key)
        logger.info("Stored: %s", result)
    except Exception as exc:
        logger.error("Failed to post ML events for analysis %d: %s", analysis_id, exc)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run §4.8 ML event detection on existing analyses.\n\n"
            "Offline mode (--video-path + --output-json): runs on pod, no API needed.\n"
            "API mode (default): fetches video URL from dashboard, POSTs results back."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analysis-id",
        dest="analysis_ids",
        type=int,
        action="append",
        required=True,
        metavar="ID",
        help="Analysis ID (repeatable)",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="(Offline mode) Path to the source video file. Skips API fetch.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="(Offline mode) Write events JSON here instead of POSTing to API.",
    )
    parser.add_argument(
        "--dashboard-url",
        default=os.environ.get("DASHBOARD_URL", "http://localhost:8000"),
        help="(API mode) Base URL of the FastAPI backend",
    )
    parser.add_argument(
        "--uploads-root",
        default=os.environ.get("PIPELINE_BASE", _DEFAULT_UPLOADS_ROOT),
        help="(API mode) Root directory where video uploads are stored",
    )
    args = parser.parse_args()

    offline_mode = args.output_json is not None or args.video_path is not None

    if offline_mode:
        if len(args.analysis_ids) > 1 and args.video_path:
            parser.error(
                "--video-path can only be used with a single --analysis-id in offline mode."
            )
        if args.video_path and not args.video_path.exists():
            parser.error(f"--video-path does not exist: {args.video_path}")

        for aid in args.analysis_ids:
            video_path = args.video_path
            output_json = args.output_json or Path(f"events_{aid}.json")
            # If multiple IDs but no video-path, we can't auto-resolve without API
            if video_path is None:
                parser.error(
                    "Offline mode requires --video-path when --output-json is set."
                )
            run_offline(aid, video_path, output_json)

    else:
        worker_key = os.environ.get("WORKER_API_KEY", "")
        if not worker_key:
            logger.warning("WORKER_API_KEY not set — relying on dev-mode bypass")
        for aid in args.analysis_ids:
            run_api(
                analysis_id=aid,
                dashboard_url=args.dashboard_url.rstrip("/"),
                worker_key=worker_key,
                uploads_root=args.uploads_root,
            )


if __name__ == "__main__":
    main()
