"""Proxy tracking quality metrics — no ground truth required.

Computes fragmentation rate, track lifetime distribution, detection density,
ball tracking quality, and team assignment stability from pipeline JSON outputs.

Usage:
    python -m backend.evaluation.tracking_quality \\
        --analytics path/to/analytics.json \\
        --tracks path/to/tracks.json \\
        --output eval_output/tracking/

Note on methodology:
    MOTA/IDF1 require MOTChallenge ground truth (frame-level bounding box
    annotations for every player). Generating this manually for even a 10s
    clip at 25fps requires annotating ~5,000 bounding boxes. We instead
    report proxy metrics that measure tracker operational quality from the
    existing pipeline outputs. This is acknowledged as a limitation.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    EvalConfig,
    ensure_output_dir,
    latex_table,
    load_analytics,
    load_tracks,
    save_figure,
    save_latex_table,
)


# ── Core computation ─────────────────────────────────────────────────────────


def compute_tracking_proxy_metrics(tracks: list[dict]) -> dict[str, Any]:
    """Compute proxy quality metrics from per-frame tracks JSON.

    Args:
        tracks: List of per-frame dicts from export_tracks_json().
                Each frame has: frameNumber, timestamp, playerPositions, ballPosition.

    Returns:
        Dict with computed metrics.
    """
    total_frames = len(tracks)

    # --- Track appearance analysis ---
    # Collect which frames each track_id appears in
    track_frames: dict[int, list[int]] = defaultdict(list)
    frame_player_counts: list[int] = []
    team_flip_counts: dict[int, int] = defaultdict(int)
    team_per_frame: dict[int, list[int]] = defaultdict(list)

    for frame in tracks:
        players = frame.get("playerPositions", {})
        # playerPositions is a dict {trackId: {x, y, teamId, ...}} from the pipeline
        if isinstance(players, dict):
            players_iter = players.items()
        else:
            players_iter = ((p.get("trackId", p.get("id", -1)), p) for p in players)
        frame_player_counts.append(len(players))
        frame_idx = frame.get("frameNumber", 0)
        for tid_raw, p in players_iter:
            tid = int(tid_raw) if isinstance(tid_raw, str) else tid_raw
            track_frames[tid].append(frame_idx)
            team_id = p.get("teamId", -1) if isinstance(p, dict) else -1
            team_per_frame[tid].append(team_id)

    # --- Track lifetime distribution ---
    lifetimes = {tid: len(frames) for tid, frames in track_frames.items()}
    lifetimes_arr = np.array(list(lifetimes.values()), dtype=float)

    # --- Fragmentation rate ---
    # Count contiguous segments per track (gaps = frames where track disappears)
    total_segments = 0
    for tid, frame_list in track_frames.items():
        sorted_frames = sorted(frame_list)
        segments = 1
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i] - sorted_frames[i - 1] > 1:
                segments += 1
        total_segments += segments

    n_tracks = len(track_frames)
    fragmentation_rate = (total_segments - n_tracks) / n_tracks if n_tracks > 0 else 0.0

    # --- Team assignment stability ---
    # Per track: proportion of frames where team_id matches the majority assignment
    unstable_tracks = 0
    for tid, team_sequence in team_per_frame.items():
        if len(team_sequence) < 3:
            continue
        majority = max(set(team_sequence), key=team_sequence.count)
        instability = sum(1 for t in team_sequence if t != majority) / len(team_sequence)
        if instability > 0.05:  # >5% frames with wrong team = unstable
            unstable_tracks += 1
    team_stability_rate = 1.0 - (unstable_tracks / n_tracks) if n_tracks > 0 else 1.0

    # --- Detection density per frame ---
    density_arr = np.array(frame_player_counts, dtype=float)

    # --- Ball presence ---
    ball_present = sum(
        1 for f in tracks if f.get("ballPosition") is not None
    )
    ball_present_pct = ball_present / total_frames * 100 if total_frames > 0 else 0.0

    # --- Short-lived track rate (proxy for ID switches / false positives) ---
    short_lived = sum(1 for lt in lifetimes.values() if lt < 10)
    short_lived_rate = short_lived / n_tracks if n_tracks > 0 else 0.0

    return {
        "total_frames": total_frames,
        "n_unique_tracks": n_tracks,
        "total_segments": total_segments,
        "fragmentation_rate": fragmentation_rate,
        "mean_lifetime_frames": float(lifetimes_arr.mean()) if len(lifetimes_arr) else 0.0,
        "median_lifetime_frames": float(np.median(lifetimes_arr)) if len(lifetimes_arr) else 0.0,
        "min_lifetime_frames": int(lifetimes_arr.min()) if len(lifetimes_arr) else 0,
        "max_lifetime_frames": int(lifetimes_arr.max()) if len(lifetimes_arr) else 0,
        "short_lived_rate": short_lived_rate,
        "mean_detections_per_frame": float(density_arr.mean()) if len(density_arr) else 0.0,
        "std_detections_per_frame": float(density_arr.std()) if len(density_arr) else 0.0,
        "ball_present_pct": ball_present_pct,
        "team_stability_rate": team_stability_rate,
        "lifetimes": lifetimes,
        "frame_player_counts": frame_player_counts,
    }


def _extract_ball_metrics(analytics: dict) -> dict:
    """Pull ball tracking quality metrics from analytics JSON."""
    return analytics.get("ball_metrics") or {}


# ── Output generation ────────────────────────────────────────────────────────


def _plot_lifetime_histogram(lifetimes: dict[int, int], output_dir: str) -> None:
    vals = list(lifetimes.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=20, color="#4f86c6", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Track lifetime (frames)")
    ax.set_ylabel("Number of tracks")
    ax.set_title("ByteTrack — Track Lifetime Distribution")
    ax.axvline(
        x=10, color="tomato", linestyle="--", linewidth=1.2,
        label="Short-lived threshold (10 frames)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "track_lifetime_histogram", output_dir)


def _plot_detection_density(counts: list[int], output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(counts, linewidth=0.8, color="#4f86c6")
    ax.fill_between(range(len(counts)), counts, alpha=0.2, color="#4f86c6")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Player detections")
    ax.set_title("Detections per Frame")
    fig.tight_layout()
    save_figure(fig, "detection_density", output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────


def run(config: EvalConfig) -> dict:
    out = str(config.output_dir)
    ensure_output_dir(out)

    analytics = load_analytics(config.analytics_path)
    tracks = load_tracks(config.tracks_path)

    metrics = compute_tracking_proxy_metrics(tracks)
    ball_m = _extract_ball_metrics(analytics)

    # --- Summary table ---
    rows = [
        ["Total frames", metrics["total_frames"]],
        ["Unique track IDs", metrics["n_unique_tracks"]],
        ["Total track segments", metrics["total_segments"]],
        ["Fragmentation rate", f"{metrics['fragmentation_rate']:.3f}"],
        ["Mean track lifetime (frames)", f"{metrics['mean_lifetime_frames']:.1f}"],
        ["Median track lifetime (frames)", f"{metrics['median_lifetime_frames']:.1f}"],
        ["Short-lived track rate (<10 frames)", f"{metrics['short_lived_rate']:.1%}"],
        ["Mean detections per frame", f"{metrics['mean_detections_per_frame']:.1f}"],
        ["Std detections per frame", f"{metrics['std_detections_per_frame']:.1f}"],
        ["Team assignment stability", f"{metrics['team_stability_rate']:.1%}"],
        ["Ball present", f"{metrics['ball_present_pct']:.1f}%"],
    ]
    save_latex_table(
        headers=["Metric", "Value"],
        rows=rows,
        caption="ByteTrack proxy quality metrics",
        name="tracking_summary",
        output_dir=out,
        label="tab:tracking_quality",
    )
    print("\n=== Tracking Quality Metrics ===")
    for metric, value in rows:
        print(f"  {metric}: {value}")

    # --- Ball metrics table (if available) ---
    if ball_m:
        ball_rows = [
            [k.replace("_", " ").title(), f"{v:.3f}" if isinstance(v, float) else v]
            for k, v in ball_m.items()
            if isinstance(v, (int, float))
        ]
        save_latex_table(
            headers=["Metric", "Value"],
            rows=ball_rows,
            caption="Ball tracking quality metrics",
            name="ball_quality",
            output_dir=out,
            label="tab:ball_quality",
        )
        print("\n=== Ball Tracking Quality ===")
        for metric, value in ball_rows:
            print(f"  {metric}: {value}")

    # --- Figures ---
    _plot_lifetime_histogram(metrics["lifetimes"], out)
    _plot_detection_density(metrics["frame_player_counts"], out)

    # Save raw metrics JSON for reference
    result = {**metrics, "ball_metrics": ball_m}
    result.pop("lifetimes", None)  # Redundant in JSON
    result.pop("frame_player_counts", None)
    (Path(out) / "tracking_metrics.json").write_text(
        json.dumps(result, indent=2)
    )

    print(f"\nOutputs saved to: {out}/")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute proxy tracking quality metrics"
    )
    parser.add_argument("--analytics", required=True, help="Path to *_analytics.json")
    parser.add_argument("--tracks", required=True, help="Path to *_tracks.json")
    parser.add_argument("--output", default="eval_output/tracking", help="Output directory")
    args = parser.parse_args()

    config = EvalConfig(
        analytics_path=args.analytics,
        tracks_path=args.tracks,
        output_dir=args.output,
    )
    run(config)


if __name__ == "__main__":
    main()
