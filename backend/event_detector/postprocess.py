"""Post-processing utilities for raw ML event detections."""

from __future__ import annotations

from typing import Dict, List, Sequence, Union


def temporal_nms(
    detections: List[Dict],
    nms_window_seconds: Union[float, Sequence[float]],
) -> List[Dict]:
    """Per-class temporal NMS — keeps highest-confidence detection per window.

    Args:
        detections: Raw detections list (dicts with timestamp, class_idx, confidence …).
        nms_window_seconds: Suppression window in seconds. Can be:
            - A single float applied uniformly to all classes (legacy behaviour).
            - A sequence of floats indexed by class_idx for per-class windows.
              §4.8 deployment values: (2.0, 2.0, 2.5, 3.0) for
              (background, challenge, play, throwin).

    Returns:
        Sorted (by timestamp) list of surviving detections.
    """
    if not detections:
        return []

    by_class: Dict[int, List[Dict]] = {}
    for d in detections:
        by_class.setdefault(d["class_idx"], []).append(d)

    kept: List[Dict] = []
    for cls_idx, dets in by_class.items():
        # Resolve the NMS window for this class.
        if isinstance(nms_window_seconds, (int, float)):
            window = float(nms_window_seconds)
        else:
            window = float(nms_window_seconds[cls_idx]) if cls_idx < len(nms_window_seconds) else float(nms_window_seconds[-1])

        dets = sorted(dets, key=lambda x: x["confidence"], reverse=True)
        suppressed: set[int] = set()
        for i, d in enumerate(dets):
            if i in suppressed:
                continue
            kept.append(d)
            for j, other in enumerate(dets):
                if j != i and j not in suppressed:
                    if abs(d["timestamp"] - other["timestamp"]) < window:
                        suppressed.add(j)

    return sorted(kept, key=lambda x: x["timestamp"])


def merge_nearby_events(
    events: List[Dict],
    merge_window_seconds: float = 1.0,
) -> List[Dict]:
    """Merge same-class events within *merge_window_seconds*, keeping the peak."""
    if not events:
        return events
    merged: List[Dict] = []
    used: set[int] = set()
    for i, ev in enumerate(events):
        if i in used:
            continue
        group = [ev]
        for j, other in enumerate(events):
            if j != i and j not in used and ev["class_name"] == other["class_name"]:
                if abs(ev["timestamp"] - other["timestamp"]) <= merge_window_seconds:
                    group.append(other)
                    used.add(j)
        best = max(group, key=lambda x: x["confidence"])
        merged.append(best)
        used.add(i)
    return sorted(merged, key=lambda x: x["timestamp"])


def compute_frame_numbers(events: List[Dict], fps: float) -> List[Dict]:
    """Add integer frame_number derived from event timestamp and fps."""
    for ev in events:
        ev["frame_number"] = int(round(ev["timestamp"] * fps))
    return events


def to_football_events(
    events: List[Dict],
    fps: float,
    model_version: str = "effnet_bce_v3",
) -> List[Dict]:
    """Convert raw ML detection dicts to the FootballEvent-compatible dict format.

    The output matches what the worker's /complete endpoint expects when it iterates
    analytics["events"]:
        event_type, frame_idx, timestamp_sec, confidence, (optional extras)

    team_id and player_track_id are left None — the model does not track identities.
    """
    result: List[Dict] = []
    for ev in events:
        result.append(
            {
                "event_type": ev["class_name"],
                "frame_idx": int(round(ev["timestamp"] * fps)),
                "timestamp_sec": ev["timestamp"],
                "team_id": None,
                "player_track_id": None,
                "target_player_track_id": None,
                "confidence": ev["confidence"],
                "success": None,
                "pitch_start": None,
                "pitch_end": None,
                # Extra metadata stored in the metadata JSON column via the Event model.
                # The worker router currently does not map this field, but it's preserved
                # for future querying / display.
                "_ml_source": model_version,
                "_weighted_score": ev.get("weighted_score"),
            }
        )
    return result
