"""Physical plausibility checks for tracking pipeline output.

Tests whether tracked player/ball positions violate physical constraints
(speed limits, player counts, pitch bounds). Zero ground truth required.

Usage:
    python3 -m evaluation.tracking_plausibility \
        --tracks ../eval_output/10_tracks.json \
        --output ../eval_output/phase16/plausibility/
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Add backend/ to sys.path for relative imports when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation._common import (
    ensure_output_dir,
    load_tracks,
    save_figure,
    save_latex_table,
)


# ── Physical constants ────────────────────────────────────────────────────────

# Maximum plausible sprint speed for a professional player
_MAX_PLAYER_SPEED_KMH = 36.0

# Maximum plausible ball speed (a hard shot can reach ~120 km/h)
_MAX_BALL_SPEED_KMH = 150.0

# Teleportation threshold: if a player moves > 5m in one 1/fps-second step
_MAX_JUMP_M = 5.0

# Expected player count range on the pitch (including subs warming up,
# referees NOT included — this covers outfield + GKs from both teams)
_PLAYER_COUNT_RANGE = (18, 26)

# Standard FIFA pitch dimensions in metres
_PITCH_LENGTH_M = 105.0
_PITCH_WIDTH_M = 68.0


# ── Helpers ───────────────────────────────────────────────────────────────────


def _has_pitch_coords(frame: dict) -> bool:
    """Return True if at least one player in the frame has pitchX/pitchY."""
    for pos in (frame.get("playerPositions") or {}).values():
        if "pitchX" in pos and "pitchY" in pos:
            return True
    return False


def _player_positions(frame: dict) -> dict[str, dict]:
    """Return playerPositions dict (empty if None)."""
    return frame.get("playerPositions") or {}


def _ball_position(frame: dict) -> dict | None:
    return frame.get("ballPosition")


# ── Check functions ───────────────────────────────────────────────────────────


def check_speed_outliers(
    frames: list[dict],
    fps: float = 25.0,
    max_kmh: float = _MAX_PLAYER_SPEED_KMH,
) -> dict[str, Any]:
    """Compute per-track speeds between consecutive frames; flag outliers.

    A violation is counted when any track exceeds `max_kmh` for 2+ consecutive
    frames (single-frame spikes are noise; sustained high speed is implausible).

    Returns:
        n_violations: total sustained-speed violation events across all tracks.
        violation_rate: fraction of (track, frame) pairs that are violations.
        worst_speed_kmh: highest speed observed (km/h).
        units: "m/s converted from metres" or "pixels/frame (no pitch coords)".
    """
    # Track previous positions: track_id -> (x, y) — prefer pitch, else pixel
    prev: dict[str, tuple[float, float]] = {}
    # Count consecutive high-speed frames per track
    consecutive: dict[str, int] = {}

    n_violations = 0
    total_pairs = 0
    worst_speed_kmh = 0.0
    use_pitch = any(_has_pitch_coords(f) for f in frames)
    units = "metres (pitch coords)" if use_pitch else "pixels (no pitch coords available)"

    # m per pixel estimate used only when no pitch coords available
    # (rough: typical broadcast frame is ~1920px wide for 105m pitch)
    PX_TO_M = 105.0 / 1920.0

    for frame in frames:
        players = _player_positions(frame)
        for tid, pos in players.items():
            if use_pitch:
                px = pos.get("pitchX")
                py = pos.get("pitchY")
                if px is None or py is None:
                    continue
                x, y = float(px), float(py)
            else:
                x = float(pos.get("x", 0.0))
                y = float(pos.get("y", 0.0))

            if tid in prev:
                prev_x, prev_y = prev[tid]
                dx = x - prev_x
                dy = y - prev_y
                dist = math.sqrt(dx * dx + dy * dy)

                if use_pitch:
                    dist_m = dist  # already in metres
                else:
                    dist_m = dist * PX_TO_M

                speed_mps = dist_m * fps  # metres per second
                speed_kmh = speed_mps * 3.6

                if speed_kmh > worst_speed_kmh:
                    worst_speed_kmh = speed_kmh

                if speed_kmh > max_kmh:
                    consecutive[tid] = consecutive.get(tid, 0) + 1
                    if consecutive[tid] >= 2:
                        n_violations += 1
                else:
                    consecutive[tid] = 0

                total_pairs += 1

            prev[tid] = (x, y)

    violation_rate = n_violations / total_pairs if total_pairs > 0 else 0.0
    return {
        "check": "speed_outliers",
        "n_violations": n_violations,
        "violation_rate": round(violation_rate, 4),
        "worst_speed_kmh": round(worst_speed_kmh, 2),
        "threshold_kmh": max_kmh,
        "units": units,
    }


def check_position_jumps(
    frames: list[dict],
    fps: float = 25.0,
    max_jump_m: float = _MAX_JUMP_M,
) -> dict[str, Any]:
    """Detect teleportation: track jumping > max_jump_m in a single frame step.

    At 25fps, 5m/frame = 450 km/h — physically impossible for a human.
    These are very likely ID switches in the tracker.

    Returns:
        n_jumps: total teleportation events.
        jump_rate: fraction of (track, frame) pairs with jump.
        likely_id_switches: same count (each jump is likely an ID switch).
    """
    prev: dict[str, tuple[float, float]] = {}
    n_jumps = 0
    total_pairs = 0
    use_pitch = any(_has_pitch_coords(f) for f in frames)
    PX_TO_M = 105.0 / 1920.0

    for frame in frames:
        players = _player_positions(frame)
        for tid, pos in players.items():
            if use_pitch:
                px = pos.get("pitchX")
                py = pos.get("pitchY")
                if px is None or py is None:
                    continue
                x, y = float(px), float(py)
            else:
                x, y = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))

            if tid in prev:
                prev_x, prev_y = prev[tid]
                dx = x - prev_x
                dy = y - prev_y
                dist = math.sqrt(dx * dx + dy * dy)
                dist_m = dist if use_pitch else dist * PX_TO_M
                if dist_m > max_jump_m:
                    n_jumps += 1
                total_pairs += 1

            prev[tid] = (x, y)

    jump_rate = n_jumps / total_pairs if total_pairs > 0 else 0.0
    return {
        "check": "position_jumps",
        "n_jumps": n_jumps,
        "jump_rate": round(jump_rate, 4),
        "likely_id_switches": n_jumps,
        "threshold_m": max_jump_m,
    }


def check_player_count(
    frames: list[dict],
    expected_range: tuple[int, int] = _PLAYER_COUNT_RANGE,
) -> dict[str, Any]:
    """Count players per frame; flag frames outside the expected range.

    Returns:
        mean_count: mean player count across all frames.
        std_count: standard deviation.
        violation_rate: fraction of frames outside expected_range.
        min_count, max_count: observed extremes.
    """
    counts = []
    for frame in frames:
        players = _player_positions(frame)
        counts.append(len(players))

    if not counts:
        return {
            "check": "player_count",
            "mean_count": 0.0,
            "std_count": 0.0,
            "violation_rate": 1.0,
            "min_count": 0,
            "max_count": 0,
            "expected_range": list(expected_range),
        }

    arr = np.array(counts, dtype=float)
    low, high = expected_range
    violations = int(np.sum((arr < low) | (arr > high)))
    return {
        "check": "player_count",
        "mean_count": round(float(arr.mean()), 2),
        "std_count": round(float(arr.std()), 2),
        "violation_rate": round(violations / len(counts), 4),
        "min_count": int(arr.min()),
        "max_count": int(arr.max()),
        "expected_range": list(expected_range),
    }


def check_pitch_bounds(
    frames: list[dict],
    pitch_length: float = _PITCH_LENGTH_M,
    pitch_width: float = _PITCH_WIDTH_M,
) -> dict[str, Any]:
    """If pitchX/pitchY are available, check all players are within pitch bounds.

    Returns:
        skipped: True if no pitch coordinates found.
        n_out_of_bounds: player-frame pairs outside pitch rectangle.
        out_of_bounds_rate: fraction of player-frame pairs.
    """
    has_any_pitch = any(_has_pitch_coords(f) for f in frames)
    if not has_any_pitch:
        return {
            "check": "pitch_bounds",
            "skipped": True,
            "reason": "No pitchX/pitchY coordinates in tracks data",
        }

    n_out = 0
    total = 0
    for frame in frames:
        for pos in _player_positions(frame).values():
            px = pos.get("pitchX")
            py = pos.get("pitchY")
            if px is None or py is None:
                continue
            total += 1
            if not (0.0 <= float(px) <= pitch_length and 0.0 <= float(py) <= pitch_width):
                n_out += 1

    return {
        "check": "pitch_bounds",
        "skipped": False,
        "n_out_of_bounds": n_out,
        "out_of_bounds_rate": round(n_out / total, 4) if total > 0 else 0.0,
        "pitch_length_m": pitch_length,
        "pitch_width_m": pitch_width,
    }


def check_ball_speed(
    frames: list[dict],
    fps: float = 25.0,
    max_kmh: float = _MAX_BALL_SPEED_KMH,
) -> dict[str, Any]:
    """Same speed check on ball position.

    Returns:
        n_violations: frames where ball exceeds max_kmh.
        violation_rate: fraction of consecutive ball-frame pairs.
        worst_speed_kmh: highest ball speed observed.
    """
    prev_ball: tuple[float, float] | None = None
    n_violations = 0
    total_pairs = 0
    worst_speed_kmh = 0.0
    use_pitch = any(
        (_ball_position(f) or {}).get("pitchPos") is not None for f in frames
    )
    PX_TO_M = 105.0 / 1920.0

    for frame in frames:
        ball = _ball_position(frame)
        if ball is None:
            prev_ball = None
            continue

        if use_pitch:
            pp = ball.get("pitchPos")
            if pp is None or len(pp) < 2:
                prev_ball = None
                continue
            # pitchPos in cm (from pipeline) → convert to metres
            x, y = float(pp[0]) / 100.0, float(pp[1]) / 100.0
        else:
            px = ball.get("pixelPos")
            if px is None or len(px) < 2:
                prev_ball = None
                continue
            x, y = float(px[0]) * PX_TO_M, float(px[1]) * PX_TO_M

        if prev_ball is not None:
            dx = x - prev_ball[0]
            dy = y - prev_ball[1]
            dist_m = math.sqrt(dx * dx + dy * dy)
            speed_kmh = dist_m * fps * 3.6
            worst_speed_kmh = max(worst_speed_kmh, speed_kmh)
            if speed_kmh > max_kmh:
                n_violations += 1
            total_pairs += 1

        prev_ball = (x, y)

    return {
        "check": "ball_speed",
        "n_violations": n_violations,
        "violation_rate": round(n_violations / total_pairs, 4) if total_pairs > 0 else 0.0,
        "worst_speed_kmh": round(worst_speed_kmh, 2),
        "threshold_kmh": max_kmh,
    }


def compute_plausibility_summary(frames: list[dict], fps: float = 25.0) -> dict[str, Any]:
    """Run all plausibility checks and compute an overall plausibility rate.

    The plausibility_rate is the fraction of frames passing ALL applicable
    checks (speed within limits, player count in range, positions in bounds).

    Returns:
        Combined report dict with individual check results plus overall
        plausibility_rate.
    """
    speed_result = check_speed_outliers(frames, fps=fps)
    jump_result = check_position_jumps(frames, fps=fps)
    count_result = check_player_count(frames)
    bounds_result = check_pitch_bounds(frames)
    ball_result = check_ball_speed(frames, fps=fps)

    # Compute per-frame pass/fail for each applicable check
    # A frame passes if: no speed violation in that frame + player count in range
    # Pitch bounds checked if available
    has_pitch = not bounds_result.get("skipped", True)
    use_pitch = any(_has_pitch_coords(f) for f in frames)
    PX_TO_M = 105.0 / 1920.0

    prev: dict[str, tuple[float, float]] = {}
    consecutive: dict[str, int] = {}

    n_frames = len(frames)
    passing_frames = 0

    low_count, high_count = _PLAYER_COUNT_RANGE

    for frame in frames:
        frame_pass = True

        # Player count check
        player_count = len(_player_positions(frame))
        if not (low_count <= player_count <= high_count):
            frame_pass = False

        # Speed check per player in this frame
        players = _player_positions(frame)
        for tid, pos in players.items():
            if use_pitch:
                px = pos.get("pitchX")
                py = pos.get("pitchY")
                if px is None or py is None:
                    prev[tid] = (0.0, 0.0)
                    continue
                x, y = float(px), float(py)
            else:
                x, y = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))

            if tid in prev:
                dx = x - prev[tid][0]
                dy = y - prev[tid][1]
                dist = math.sqrt(dx * dx + dy * dy)
                dist_m = dist if use_pitch else dist * PX_TO_M
                speed_kmh = dist_m * fps * 3.6
                if speed_kmh > _MAX_PLAYER_SPEED_KMH:
                    consecutive[tid] = consecutive.get(tid, 0) + 1
                    if consecutive[tid] >= 2:
                        frame_pass = False
                else:
                    consecutive[tid] = 0
            prev[tid] = (x, y)

        # Pitch bounds check (if pitch coords available)
        if has_pitch:
            for pos in players.values():
                px = pos.get("pitchX")
                py = pos.get("pitchY")
                if px is None or py is None:
                    continue
                if not (0.0 <= float(px) <= _PITCH_LENGTH_M and 0.0 <= float(py) <= _PITCH_WIDTH_M):
                    frame_pass = False
                    break

        if frame_pass:
            passing_frames += 1

    plausibility_rate = passing_frames / n_frames if n_frames > 0 else 0.0

    return {
        "n_frames": n_frames,
        "plausibility_rate": round(plausibility_rate, 4),
        "passing_frames": passing_frames,
        "checks": {
            "speed_outliers": speed_result,
            "position_jumps": jump_result,
            "player_count": count_result,
            "pitch_bounds": bounds_result,
            "ball_speed": ball_result,
        },
    }


# ── Output generation ─────────────────────────────────────────────────────────


def _build_per_frame_speeds(frames: list[dict], fps: float = 25.0) -> list[float]:
    """Return a flat list of player speed values (km/h) across all frame transitions."""
    prev: dict[str, tuple[float, float]] = {}
    speeds: list[float] = []
    use_pitch = any(_has_pitch_coords(f) for f in frames)
    PX_TO_M = 105.0 / 1920.0

    for frame in frames:
        for tid, pos in _player_positions(frame).items():
            if use_pitch:
                px = pos.get("pitchX")
                py = pos.get("pitchY")
                if px is None or py is None:
                    continue
                x, y = float(px), float(py)
            else:
                x, y = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))

            if tid in prev:
                dx = x - prev[tid][0]
                dy = y - prev[tid][1]
                dist = math.sqrt(dx * dx + dy * dy)
                dist_m = dist if use_pitch else dist * PX_TO_M
                speeds.append(dist_m * fps * 3.6)

            prev[tid] = (x, y)

    return speeds


def _build_per_frame_counts(frames: list[dict]) -> list[int]:
    return [len(_player_positions(f)) for f in frames]


def plot_speed_distribution(frames: list[dict], output_dir: str, fps: float = 25.0) -> None:
    """Histogram of player speeds with vertical line at max threshold."""
    speeds = _build_per_frame_speeds(frames, fps=fps)
    if not speeds:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(speeds, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(_MAX_PLAYER_SPEED_KMH, color="red", linestyle="--", linewidth=1.5,
               label=f"Max plausible ({_MAX_PLAYER_SPEED_KMH} km/h)")
    ax.set_xlabel("Player speed (km/h)")
    ax.set_ylabel("Count (track-frame pairs)")
    ax.set_title("Distribution of Inter-Frame Player Speeds")
    ax.legend()
    save_figure(fig, "speed_distribution", output_dir)


def plot_player_count_timeseries(frames: list[dict], output_dir: str) -> None:
    """Line plot of player count per frame."""
    counts = _build_per_frame_counts(frames)
    if not counts:
        return

    low, high = _PLAYER_COUNT_RANGE
    frame_nums = [f.get("frameNumber", i) for i, f in enumerate(frames)]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frame_nums, counts, linewidth=0.8, color="steelblue")
    ax.axhline(low, color="orange", linestyle="--", linewidth=1.2, label=f"Min expected ({low})")
    ax.axhline(high, color="red", linestyle="--", linewidth=1.2, label=f"Max expected ({high})")
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Player count")
    ax.set_title("Player Count per Frame")
    ax.legend(loc="lower right")
    save_figure(fig, "player_count_timeseries", output_dir)


def build_plausibility_latex_table(summary: dict, output_dir: str) -> None:
    """LaTeX table: check name, pass rate, worst-case value."""
    checks = summary.get("checks", {})
    rows = []

    def _pass_rate(r: dict) -> str:
        vr = r.get("violation_rate")
        if vr is None:
            return "N/A"
        return f"{(1 - vr) * 100:.1f}\\%"

    # Speed outliers
    sr = checks.get("speed_outliers", {})
    rows.append([
        "Player speed outliers",
        _pass_rate(sr),
        f"{sr.get('worst_speed_kmh', 0.0):.1f} km/h",
        f"Sustained > {sr.get('threshold_kmh', _MAX_PLAYER_SPEED_KMH)} km/h for 2+ frames",
    ])

    # Position jumps
    jr = checks.get("position_jumps", {})
    rows.append([
        "Position jumps (ID switches)",
        _pass_rate(jr),
        f"{jr.get('n_jumps', 0)} jumps",
        f"> {jr.get('threshold_m', _MAX_JUMP_M)} m/frame",
    ])

    # Player count
    cr = checks.get("player_count", {})
    rows.append([
        "Player count in range",
        _pass_rate(cr),
        f"min={cr.get('min_count', '?')}, max={cr.get('max_count', '?')}",
        f"Expected {cr.get('expected_range', _PLAYER_COUNT_RANGE)}",
    ])

    # Pitch bounds
    br = checks.get("pitch_bounds", {})
    if br.get("skipped"):
        rows.append([
            "Pitch bounds",
            "N/A",
            "N/A",
            "Skipped (no pitch coords)",
        ])
    else:
        rows.append([
            "Pitch bounds",
            _pass_rate(br),
            f"{br.get('n_out_of_bounds', 0)} out-of-bounds",
            f"Pitch {br.get('pitch_length_m', 105)}m x {br.get('pitch_width_m', 68)}m",
        ])

    # Ball speed
    bl = checks.get("ball_speed", {})
    rows.append([
        "Ball speed",
        _pass_rate(bl),
        f"{bl.get('worst_speed_kmh', 0.0):.1f} km/h",
        f"> {bl.get('threshold_kmh', _MAX_BALL_SPEED_KMH)} km/h",
    ])

    # Overall
    rows.append([
        "\\textbf{Overall plausibility}",
        f"\\textbf{{{summary.get('plausibility_rate', 0.0) * 100:.1f}\\%}}",
        f"{summary.get('passing_frames', 0)}/{summary.get('n_frames', 0)} frames",
        "All applicable checks passing",
    ])

    save_latex_table(
        headers=["Check", "Pass rate", "Worst case", "Threshold"],
        rows=rows,
        caption="Physical plausibility checks on tracking pipeline output",
        name="plausibility_summary",
        output_dir=output_dir,
        label="tab:plausibility",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Physical plausibility checks for tracking pipeline output"
    )
    parser.add_argument(
        "--tracks",
        required=True,
        help="Path to tracks JSON (from export_tracks_json)",
    )
    parser.add_argument(
        "--output",
        default="../eval_output/phase16/plausibility/",
        help="Output directory for reports and figures",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Video frame rate (default: 25.0)",
    )
    args = parser.parse_args()

    out = ensure_output_dir(args.output)
    print(f"Loading tracks from: {args.tracks}")
    frames = load_tracks(args.tracks)
    print(f"Loaded {len(frames)} frames.")

    print("Running plausibility checks...")
    summary = compute_plausibility_summary(frames, fps=args.fps)

    # Write JSON report
    report_path = out / "plausibility_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Report written: {report_path}")

    # LaTeX table
    build_plausibility_latex_table(summary, args.output)
    print(f"LaTeX table written: {out / 'plausibility_summary.tex'}")

    # Figures
    plot_speed_distribution(frames, args.output, fps=args.fps)
    print(f"Speed distribution figure written: {out / 'speed_distribution.pdf'}")

    plot_player_count_timeseries(frames, args.output)
    print(f"Player count timeseries written: {out / 'player_count_timeseries.pdf'}")

    # Summary to stdout
    pr = summary["plausibility_rate"]
    print(f"\nOverall plausibility rate: {pr * 100:.1f}% ({summary['passing_frames']}/{summary['n_frames']} frames)")
    for name, check in summary["checks"].items():
        vr = check.get("violation_rate", "N/A")
        if isinstance(vr, float):
            print(f"  {name}: violation_rate={vr:.4f}")
        else:
            print(f"  {name}: {check.get('reason', 'skipped')}")


if __name__ == "__main__":
    main()
