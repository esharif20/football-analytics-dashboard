"""Median Absolute Deviation (MAD) trajectory cleanup.

Computes per-track adaptive speed thresholds so that stationary goalkeepers
and sprinting wingers each get appropriate outlier limits instead of a single
hard-coded constant.
"""

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt


def mad_threshold(
    values: npt.NDArray[np.float64],
    k: float = 3.0,
) -> float:
    """Compute adaptive outlier threshold: median + k * MAD.

    Args:
        values: 1-D array of observations (e.g. per-frame speeds in cm).
        k: Multiplier on the MAD.  3.0 ≈ 99.7% for Gaussian-like data.

    Returns:
        Threshold above which an observation is considered an outlier.
        If the array is empty, returns ``inf`` (never reject).
    """
    if values.size == 0:
        return float("inf")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    # Avoid zero-MAD (constant track) from rejecting everything
    if mad == 0.0:
        return max(median * 2.0, 1.0)

    return median + k * mad


def cleanup_positions(
    positions: List[Tuple[float, float]],
    frame_indices: List[int],
    k: float = 3.0,
    hard_cap: Optional[float] = None,
) -> Tuple[List[Optional[Tuple[float, float]]], List[float]]:
    """Remove outlier positions using MAD on per-frame displacement.

    Outlier frames are replaced with ``None``; callers can interpolate
    over the gaps.

    Args:
        positions: List of (x, y) coordinates in consistent units (e.g. cm).
        frame_indices: Corresponding frame numbers (used for frame-gap scaling).
        k: MAD multiplier.
        hard_cap: Optional absolute maximum displacement per frame (safety net).
            For players pass ``MAX_PLAYER_SPEED`` converted to cm/frame.

    Returns:
        (cleaned_positions, per_segment_speeds) where cleaned_positions has
        ``None`` at outlier indices.
    """
    n = len(positions)
    if n < 3:
        return list(positions), []  # type: ignore[arg-type]

    pts = np.asarray(positions, dtype=np.float64)
    fi = np.asarray(frame_indices, dtype=np.float64)

    # Per-segment displacement normalised by frame gap
    diffs = np.diff(pts, axis=0)
    gaps = np.maximum(np.diff(fi), 1.0)
    speeds = np.linalg.norm(diffs, axis=1) / gaps  # cm per frame

    threshold = mad_threshold(speeds, k)
    if hard_cap is not None:
        threshold = min(threshold, hard_cap)

    cleaned: List[Optional[Tuple[float, float]]] = [positions[0]]
    for i in range(len(speeds)):
        if speeds[i] > threshold:
            cleaned.append(None)
        else:
            cleaned.append(positions[i + 1])

    return cleaned, speeds.tolist()


def interpolate_gaps(
    positions: List[Optional[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """Linear-interpolate over ``None`` gaps, edge-fill if needed.

    Args:
        positions: List with ``None`` at outlier indices.

    Returns:
        Fully filled list (no ``None``).
    """
    n = len(positions)
    if n == 0:
        return []

    result: List[Optional[Tuple[float, float]]] = list(positions)

    valid = [(i, p) for i, p in enumerate(result) if p is not None]
    if not valid:
        return [(0.0, 0.0)] * n

    # Edge-fill leading/trailing Nones
    first_i, first_p = valid[0]
    last_i, last_p = valid[-1]
    for i in range(first_i):
        result[i] = first_p
    for i in range(last_i + 1, n):
        result[i] = last_p

    # Interpolate interior gaps
    for vi in range(len(valid) - 1):
        li, lp = valid[vi]
        ri, rp = valid[vi + 1]
        if ri - li > 1:
            for j in range(li + 1, ri):
                t = (j - li) / (ri - li)
                result[j] = (
                    lp[0] + t * (rp[0] - lp[0]),
                    lp[1] + t * (rp[1] - lp[1]),
                )

    return result  # type: ignore[return-value]
