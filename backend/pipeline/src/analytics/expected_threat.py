"""Expected Threat (xT) grid for ball progression value.

12×8 lookup matrix from Karun Singh's open-source xT model
(https://karun.in/blog/expected-threat.html), trained on StatsBomb open data.
Each cell represents the probability of scoring from that pitch zone.

References:
    - Shaw & Sudarshan (2020): "Expected Threat" framework
    - Singh (2019): Open-source 12×8 xT matrix
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .types import FootballEvent
except ImportError:
    from types import FootballEvent  # type: ignore


# ── xT lookup matrix (12 columns × 8 rows) ────────────────────────────────
# Column 0 = own goal-line, column 11 = opponent goal-line.
# Row 0 = bottom touchline, row 7 = top touchline.
# Values are expected goal probability (0.0–1.0).
# Source: Karun Singh's 12×8 model (public domain).

_XT_MATRIX = np.array([
    [0.008, 0.008, 0.009, 0.010, 0.010, 0.011, 0.011, 0.011, 0.012, 0.013, 0.016, 0.026],
    [0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.020, 0.026, 0.041],
    [0.009, 0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.022, 0.028, 0.039, 0.072],
    [0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.021, 0.026, 0.036, 0.057, 0.112],
    [0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.021, 0.026, 0.036, 0.057, 0.112],
    [0.009, 0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.022, 0.028, 0.039, 0.072],
    [0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.020, 0.026, 0.041],
    [0.008, 0.008, 0.009, 0.010, 0.010, 0.011, 0.011, 0.011, 0.012, 0.013, 0.016, 0.026],
], dtype=np.float32)  # shape: (8, 12)

_N_COLS = 12
_N_ROWS = 8

# Pitch dimensions (cm) — standard FIFA pitch
_PITCH_LENGTH_CM = 10500.0
_PITCH_WIDTH_CM = 6800.0


# ── Core lookup ───────────────────────────────────────────────────────────────


def _pitch_to_grid(x_cm: float, y_cm: float) -> Tuple[int, int]:
    """Map pitch coordinates (cm) to (col, row) in the xT grid."""
    col = int(np.clip(x_cm / _PITCH_LENGTH_CM * _N_COLS, 0, _N_COLS - 1))
    row = int(np.clip(y_cm / _PITCH_WIDTH_CM * _N_ROWS, 0, _N_ROWS - 1))
    return col, row


def lookup_xt(x_cm: float, y_cm: float) -> float:
    """Return xT value for a pitch position in cm."""
    col, row = _pitch_to_grid(x_cm, y_cm)
    return float(_XT_MATRIX[row, col])


def compute_xt_gained(
    start_cm: Tuple[float, float],
    end_cm: Tuple[float, float],
    attacking_dir: int = 1,
) -> float:
    """Compute xT gained from moving ball from start to end.

    Args:
        start_cm: Ball position before action (x, y) in cm.
        end_cm: Ball position after action (x, y) in cm.
        attacking_dir: +1 if team attacks toward higher x, -1 otherwise.

    Returns:
        xT gained (positive = threat increased, negative = decreased).
    """
    if attacking_dir < 0:
        # Flip x coordinate so matrix is always oriented attack-right
        sx = _PITCH_LENGTH_CM - start_cm[0]
        ex = _PITCH_LENGTH_CM - end_cm[0]
        start_flipped = (sx, start_cm[1])
        end_flipped = (ex, end_cm[1])
    else:
        start_flipped = start_cm
        end_flipped = end_cm

    xt_start = lookup_xt(*start_flipped)
    xt_end = lookup_xt(*end_flipped)
    return round(xt_end - xt_start, 4)


# ── Team-level aggregation ────────────────────────────────────────────────────


def compute_team_xt(
    events: List,
    team_id: int,
    team_dir: int = 1,
) -> Dict[str, Optional[float]]:
    """Aggregate xT added by a team across all pass/carry events.

    Args:
        events: List of FootballEvent (dataclass or dict).
        team_id: Team to aggregate for (1 or 2).
        team_dir: Attacking direction (+1 or -1) for team.

    Returns:
        {
            "total_xt": float,
            "xt_per_pass": float | None,
            "progressive_xt": float,  # xT from progressive passes only
        }
    """
    total_xt = 0.0
    progressive_xt = 0.0
    pass_count = 0

    for ev in events:
        if isinstance(ev, dict):
            etype = ev.get("event_type", "")
            etid = ev.get("team_id")
            p_start = ev.get("pitch_start")
            p_end = ev.get("pitch_end")
            is_prog = ev.get("is_progressive")
        else:
            etype = getattr(ev, "event_type", "")
            etid = getattr(ev, "team_id", None)
            p_start = getattr(ev, "pitch_start", None)
            p_end = getattr(ev, "pitch_end", None)
            is_prog = getattr(ev, "is_progressive", None)

        if etid != team_id or etype not in ("pass", "carry"):
            continue
        if p_start is None or p_end is None:
            continue

        xt = compute_xt_gained(p_start, p_end, team_dir)
        total_xt += xt
        pass_count += 1

        if is_prog:
            progressive_xt += xt

    return {
        "total_xt": round(total_xt, 3),
        "xt_per_pass": round(total_xt / pass_count, 4) if pass_count > 0 else None,
        "progressive_xt": round(progressive_xt, 3),
        "pass_count_with_coords": pass_count,
    }
