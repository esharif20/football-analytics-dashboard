"""Tactical metrics computation from pitch-space player positions.

Computes per-window team shape, pressing, and space control metrics from
per-frame (x, y, teamId) tracking data projected to pitch coordinates.

References:
    - Zhang et al. 2025: Team tactical analysis pipeline
    - arXiv:2501.04712: Pressing intensity from tracking data
    - FC Python: Convex hulls for football
    - floodlight library: Stretch index
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from utils.bbox_utils import get_foot_position, get_center_of_bbox
except ImportError:
    from src.utils.bbox_utils import get_foot_position, get_center_of_bbox

from .types import FootballEvent, PossessionEvent


# ── Constants ────────────────────────────────────────────────────────────────

# Window size in frames (30 frames ≈ 1.2s at 25fps)
_WINDOW_SIZE = 30

# Sample every Nth frame within a window for efficiency
_FRAME_SAMPLE_STEP = 3

# Pressing: defenders within this distance (cm) of ball when out of possession
_PRESSING_RADIUS_CM = 1000.0  # 10 metres

# Pitch bounds (cm) — standard FIFA pitch
_PITCH_LENGTH_CM = 10500.0
_PITCH_WIDTH_CM = 6800.0

# Minimum players needed for convex hull (scipy requires >= 3)
_MIN_HULL_PLAYERS = 3

# Defensive line: use deepest N outfield players
_DEFENSIVE_LINE_N = 4


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class TacticalWindow:
    """Tactical metrics for a single time window."""
    start_frame: int
    end_frame: int
    minute: float

    team_1_compactness: Optional[float] = None   # convex hull area in m²
    team_2_compactness: Optional[float] = None
    team_1_stretch_index: Optional[float] = None  # mean dist to centroid in m
    team_2_stretch_index: Optional[float] = None
    team_1_length: Optional[float] = None         # max x-spread in m
    team_2_length: Optional[float] = None
    team_1_width: Optional[float] = None          # max y-spread in m
    team_2_width: Optional[float] = None
    team_1_defensive_line: Optional[float] = None  # metres from goal
    team_2_defensive_line: Optional[float] = None
    team_1_pressing_intensity: Optional[float] = None  # 0-1 ratio
    team_2_pressing_intensity: Optional[float] = None
    inter_team_distance: Optional[float] = None   # metres between centroids
    # Phase-of-play (A1): "ip", "oop", "dat", "adt", "contested"
    phase_team_1: Optional[str] = None
    phase_team_2: Optional[str] = None
    # Voronoi territory control (A2)
    team_1_territory_pct: Optional[float] = None
    team_2_territory_pct: Optional[float] = None
    team_1_opp_half_territory_pct: Optional[float] = None
    team_2_opp_half_territory_pct: Optional[float] = None
    # Press classification (A5)
    team_1_press_type: Optional[str] = None   # "high", "mid", "low"
    team_2_press_type: Optional[str] = None
    team_1_counter_press: Optional[bool] = None
    team_2_counter_press: Optional[bool] = None


@dataclass
class TacticalMetrics:
    """Complete tactical metrics output."""
    windows: List[TacticalWindow] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    ppda_team_1: Optional[float] = None
    ppda_team_2: Optional[float] = None


# ── Core computation ─────────────────────────────────────────────────────────


def _convex_hull_area(points: np.ndarray) -> Optional[float]:
    """Compute convex hull area in m² from pitch coords (cm)."""
    if len(points) < _MIN_HULL_PLAYERS:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return hull.volume / 1e4  # cm² → m²
    except Exception:
        return None


def _stretch_index(points: np.ndarray) -> Optional[float]:
    """Mean distance of players to their centroid, in metres."""
    if len(points) < 2:
        return None
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    return float(dists.mean()) / 100.0  # cm → m


def _team_spread(points: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Return (length_m, width_m) — max x-spread and y-spread."""
    if len(points) < 2:
        return None, None
    length = float(points[:, 0].max() - points[:, 0].min()) / 100.0
    width = float(points[:, 1].max() - points[:, 1].min()) / 100.0
    return length, width


def _defensive_line_height(
    points: np.ndarray,
    attacking_direction: int,
) -> Optional[float]:
    """Mean x-position of deepest N outfield players, in metres from own goal.

    Args:
        points: (N, 2) pitch positions in cm.
        attacking_direction: +1 if attacking toward x=10500, -1 if toward x=0.
    """
    if len(points) < _DEFENSIVE_LINE_N:
        return None

    x_vals = points[:, 0]
    if attacking_direction > 0:
        # Attacking right → defensive line is lowest x values
        sorted_x = np.sort(x_vals)[:_DEFENSIVE_LINE_N]
    else:
        # Attacking left → defensive line is highest x values
        sorted_x = np.sort(x_vals)[-_DEFENSIVE_LINE_N:]

    return float(sorted_x.mean()) / 100.0  # cm → m


def _pressing_intensity(
    defending_positions: np.ndarray,
    ball_pos: Optional[Tuple[float, float]],
) -> Optional[float]:
    """Fraction of defending players within pressing radius of ball.

    Returns 0-1 ratio (e.g. 0.6 = 60% of defenders pressing).
    """
    if ball_pos is None or len(defending_positions) == 0:
        return None
    ball = np.array(ball_pos)
    dists = np.linalg.norm(defending_positions - ball, axis=1)
    n_pressing = int((dists < _PRESSING_RADIUS_CM).sum())
    return n_pressing / len(defending_positions)


def _voronoi_territory(
    team_1_pts: np.ndarray,
    team_2_pts: np.ndarray,
    pitch_length: float = _PITCH_LENGTH_CM,
    pitch_width: float = _PITCH_WIDTH_CM,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Grid-based Voronoi territory approximation (Fernandez & Bornn 2018).

    Divides pitch into a 60×40 grid, assigns each cell to nearest player.
    Returns (team_1_pct, team_2_pct, team_1_opp_half_pct, team_2_opp_half_pct).
    All values are fractions 0-1.
    """
    if len(team_1_pts) == 0 or len(team_2_pts) == 0:
        return None, None, None, None

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None, None, None, None

    all_pts = np.vstack([team_1_pts, team_2_pts])
    labels = np.array([1] * len(team_1_pts) + [2] * len(team_2_pts))

    # 60×40 grid over the pitch
    xs = np.linspace(0, pitch_length, 60)
    ys = np.linspace(0, pitch_width, 40)
    gx, gy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    tree = cKDTree(all_pts)
    _, idx = tree.query(grid_pts)
    grid_labels = labels[idx]

    total = len(grid_pts)
    t1_pct = float((grid_labels == 1).sum()) / total
    t2_pct = float((grid_labels == 2).sum()) / total

    # Opponent-half territory: team attacking higher x occupies right half
    mid_x = pitch_length / 2.0
    t1_mean_x = team_1_pts[:, 0].mean()
    t2_mean_x = team_2_pts[:, 0].mean()

    if t1_mean_x < t2_mean_x:
        # Team 1 attacks right (positive x)
        t1_opp_mask = grid_pts[:, 0] > mid_x
        t2_opp_mask = grid_pts[:, 0] <= mid_x
    else:
        t1_opp_mask = grid_pts[:, 0] <= mid_x
        t2_opp_mask = grid_pts[:, 0] > mid_x

    t1_opp_total = t1_opp_mask.sum()
    t2_opp_total = t2_opp_mask.sum()
    t1_opp_pct = float((grid_labels[t1_opp_mask] == 1).sum()) / t1_opp_total if t1_opp_total > 0 else None
    t2_opp_pct = float((grid_labels[t2_opp_mask] == 2).sum()) / t2_opp_total if t2_opp_total > 0 else None

    return t1_pct, t2_pct, t1_opp_pct, t2_opp_pct


def _classify_phase(
    win_start: int,
    win_end: int,
    poss_lookup: Dict[int, int],
    fps: float,
) -> Tuple[str, str]:
    """Classify phase-of-play for each team (Rein & Memmert 2016).

    Returns (phase_team_1, phase_team_2) where each is one of:
    "ip" (in possession), "oop" (out of possession),
    "dat" (defensive→attack transition), "adt" (attack→defensive transition),
    "contested".
    """
    transition_frames = int(5.0 * fps)  # 5-second lookback for transition detection

    # Count possession in window
    t1_count = sum(1 for fi in range(win_start, win_end) if poss_lookup.get(fi) == 1)
    t2_count = sum(1 for fi in range(win_start, win_end) if poss_lookup.get(fi) == 2)
    total = win_end - win_start

    if total == 0:
        return "contested", "contested"

    # Determine dominant possession
    if t1_count / total > 0.60:
        dominant = 1
    elif t2_count / total > 0.60:
        dominant = 2
    else:
        return "contested", "contested"

    # Check if possession was recently won (transition in last 5 seconds)
    lookback = max(0, win_start - transition_frames)
    prev_dominant = 0
    prev_count: Dict[int, int] = {1: 0, 2: 0}
    for fi in range(lookback, win_start):
        t = poss_lookup.get(fi, 0)
        if t in (1, 2):
            prev_count[t] += 1
    lookback_total = win_start - lookback
    if lookback_total > 0:
        if prev_count[1] / lookback_total > 0.55:
            prev_dominant = 1
        elif prev_count[2] / lookback_total > 0.55:
            prev_dominant = 2

    if dominant == 1:
        if prev_dominant == 2:
            return "dat", "adt"  # Team 1 just won ball
        return "ip", "oop"
    else:
        if prev_dominant == 1:
            return "adt", "dat"  # Team 2 just won ball
        return "oop", "ip"


def _classify_press_type(
    defensive_line_m: Optional[float],
    pressing_intensity: Optional[float],
) -> Optional[str]:
    """Classify pressing block type (FIFA TSG 2022).

    High press: defensive line > 60m from own goal AND pressing > 0.3
    Mid-block:  defensive line 35-60m
    Low block:  defensive line < 35m
    """
    if defensive_line_m is None:
        return None
    if defensive_line_m > 60 and (pressing_intensity or 0.0) > 0.3:
        return "high"
    if defensive_line_m >= 35:
        return "mid"
    return "low"


def _detect_counter_press(
    phase: str,
    prev_phase: Optional[str],
    pressing_intensity: Optional[float],
) -> Optional[bool]:
    """Detect counter-press: high pressing intensity immediately after losing ball."""
    if prev_phase in ("ip", "dat") and phase == "oop":
        return pressing_intensity is not None and pressing_intensity > 0.30
    return False


def _compute_ppda(
    events: List[FootballEvent],
    possession_events: List[PossessionEvent],
) -> Tuple[Optional[float], Optional[float]]:
    """Approximate PPDA (Passes Per Defensive Action) per team.

    PPDA = opponent passes allowed / own defensive actions (challenges).
    Lower PPDA = more aggressive pressing.

    Team 1 PPDA = (passes by team 2) / (challenges by team 1)
    Team 2 PPDA = (passes by team 1) / (challenges by team 2)
    """
    passes_by_team: Dict[int, int] = {1: 0, 2: 0}
    challenges_by_team: Dict[int, int] = {1: 0, 2: 0}

    for ev in events:
        etype = ev.event_type if hasattr(ev, "event_type") else ev.get("event_type", "")
        tid = ev.team_id if hasattr(ev, "team_id") else ev.get("team_id")
        if tid not in (1, 2):
            continue
        if etype == "pass":
            passes_by_team[tid] += 1
        elif etype == "challenge":
            challenges_by_team[tid] += 1

    ppda_1 = (passes_by_team[2] / challenges_by_team[1]) if challenges_by_team[1] > 0 else None
    ppda_2 = (passes_by_team[1] / challenges_by_team[2]) if challenges_by_team[2] > 0 else None
    return ppda_1, ppda_2


# ── TacticalCalculator ───────────────────────────────────────────────────────


class TacticalCalculator:
    """Compute tactical metrics from tracking data using pitch-space positions.

    Follows the `_gather_pitch_data()` pattern from interaction_graph.py
    for converting pixel positions to pitch coordinates.
    """

    def __init__(self, fps: float = 25.0, window_size: int = _WINDOW_SIZE):
        self.fps = fps
        self.window_size = window_size

    def compute(
        self,
        tracks: Dict[str, list],
        per_frame_transformers: Optional[Dict] = None,
        events: Optional[List[FootballEvent]] = None,
        possession_events: Optional[List[PossessionEvent]] = None,
    ) -> TacticalMetrics:
        """Compute all tactical metrics.

        Args:
            tracks: Full track dict with players, goalkeepers, ball.
            per_frame_transformers: Per-frame ViewTransformers for pitch projection.
            events: Detected events (for PPDA calculation).
            possession_events: Per-frame possession (for pressing context).

        Returns:
            TacticalMetrics with per-window data and aggregated summary.
        """
        if not per_frame_transformers:
            return TacticalMetrics(summary={"error": "no homography available"})

        # Gather pitch-space data for all frames
        pitch_data = self._gather_all_pitch_data(tracks, per_frame_transformers)
        if not pitch_data:
            return TacticalMetrics(summary={"error": "no pitch data gathered"})

        # Build possession lookup for pressing context
        poss_lookup: Dict[int, int] = {}
        if possession_events:
            for pe in possession_events:
                fi = pe.frame_idx if hasattr(pe, "frame_idx") else pe.get("frame_idx", 0)
                tid = pe.team_id if hasattr(pe, "team_id") else pe.get("team_id", 0)
                poss_lookup[fi] = tid

        # Compute per-window metrics
        frame_indices = sorted(pitch_data.keys())
        if not frame_indices:
            return TacticalMetrics()

        total_frames = frame_indices[-1] + 1
        windows: List[TacticalWindow] = []
        prev_phases: Dict[int, Optional[str]] = {1: None, 2: None}

        for win_start in range(0, total_frames, self.window_size):
            win_end = min(win_start + self.window_size, total_frames)
            minute = (win_start / self.fps) / 60.0

            # Collect positions within this window
            team_1_pts: List[Tuple[float, float]] = []
            team_2_pts: List[Tuple[float, float]] = []
            ball_pts: List[Tuple[float, float]] = []
            pressing_team_1: List[Tuple[float, float]] = []
            pressing_team_2: List[Tuple[float, float]] = []
            pressing_ball: List[Optional[Tuple[float, float]]] = []

            for fi in range(win_start, win_end, _FRAME_SAMPLE_STEP):
                fd = pitch_data.get(fi)
                if fd is None:
                    continue

                for tid, pos in fd["players"]:
                    if tid == 1:
                        team_1_pts.append(pos)
                    elif tid == 2:
                        team_2_pts.append(pos)

                if fd["ball"] is not None:
                    ball_pts.append(fd["ball"])

                # For pressing: collect defending team positions when opponent has ball
                poss_team = poss_lookup.get(fi, 0)
                if poss_team == 2:
                    # Team 2 has ball → team 1 is defending/pressing
                    for t, pos in fd["players"]:
                        if t == 1:
                            pressing_team_1.append(pos)
                    pressing_ball.append(fd["ball"])
                elif poss_team == 1:
                    # Team 1 has ball → team 2 is defending/pressing
                    for t, pos in fd["players"]:
                        if t == 2:
                            pressing_team_2.append(pos)
                    pressing_ball.append(fd["ball"])

            # Convert to numpy for computation
            t1 = np.array(team_1_pts) if team_1_pts else np.empty((0, 2))
            t2 = np.array(team_2_pts) if team_2_pts else np.empty((0, 2))

            # Determine attacking direction from centroids
            t1_centroid = t1.mean(axis=0) if len(t1) > 0 else None
            t2_centroid = t2.mean(axis=0) if len(t2) > 0 else None

            # Convention: team with lower mean x attacks right (+1)
            if t1_centroid is not None and t2_centroid is not None:
                t1_dir = 1 if t1_centroid[0] < t2_centroid[0] else -1
                t2_dir = -t1_dir
                inter_dist = float(np.linalg.norm(t1_centroid - t2_centroid)) / 100.0
            else:
                t1_dir, t2_dir = 1, -1
                inter_dist = None

            # Compute per-team metrics
            t1_len, t1_wid = _team_spread(t1)
            t2_len, t2_wid = _team_spread(t2)

            # Pressing intensity
            p1_arr = np.array(pressing_team_1) if pressing_team_1 else np.empty((0, 2))
            p2_arr = np.array(pressing_team_2) if pressing_team_2 else np.empty((0, 2))
            # Use mean ball position for pressing calculation
            mean_ball = None
            valid_balls = [b for b in ball_pts if b is not None]
            if valid_balls:
                mean_ball = (
                    np.mean([b[0] for b in valid_balls]),
                    np.mean([b[1] for b in valid_balls]),
                )

            # Phase-of-play (A1)
            phase_t1, phase_t2 = _classify_phase(win_start, win_end, poss_lookup, self.fps)

            # Voronoi territory (A2) — use mean positions per window
            t1_territory, t2_territory, t1_opp_territory, t2_opp_territory = (
                _voronoi_territory(t1, t2) if len(t1) >= 3 and len(t2) >= 3 else (None, None, None, None)
            )

            # Press classification (A5)
            t1_def_line = _defensive_line_height(t1, t1_dir)
            t2_def_line = _defensive_line_height(t2, t2_dir)
            t1_press_int = _pressing_intensity(p1_arr, mean_ball)
            t2_press_int = _pressing_intensity(p2_arr, mean_ball)

            t1_press_type = _classify_press_type(t1_def_line, t1_press_int)
            t2_press_type = _classify_press_type(t2_def_line, t2_press_int)
            t1_counter = _detect_counter_press(phase_t1, prev_phases[1], t1_press_int)
            t2_counter = _detect_counter_press(phase_t2, prev_phases[2], t2_press_int)

            prev_phases[1] = phase_t1
            prev_phases[2] = phase_t2

            window = TacticalWindow(
                start_frame=win_start,
                end_frame=win_end,
                minute=round(minute, 2),
                team_1_compactness=_convex_hull_area(t1),
                team_2_compactness=_convex_hull_area(t2),
                team_1_stretch_index=_stretch_index(t1),
                team_2_stretch_index=_stretch_index(t2),
                team_1_length=t1_len,
                team_2_length=t2_len,
                team_1_width=t1_wid,
                team_2_width=t2_wid,
                team_1_defensive_line=t1_def_line,
                team_2_defensive_line=t2_def_line,
                team_1_pressing_intensity=t1_press_int,
                team_2_pressing_intensity=t2_press_int,
                inter_team_distance=inter_dist,
                phase_team_1=phase_t1,
                phase_team_2=phase_t2,
                team_1_territory_pct=t1_territory,
                team_2_territory_pct=t2_territory,
                team_1_opp_half_territory_pct=t1_opp_territory,
                team_2_opp_half_territory_pct=t2_opp_territory,
                team_1_press_type=t1_press_type,
                team_2_press_type=t2_press_type,
                team_1_counter_press=t1_counter,
                team_2_counter_press=t2_counter,
            )
            windows.append(window)

        # PPDA from events
        ppda_1, ppda_2 = _compute_ppda(events or [], possession_events or [])

        # Aggregate summary
        summary = self._aggregate_summary(windows, ppda_1, ppda_2)

        return TacticalMetrics(
            windows=windows,
            summary=summary,
            ppda_team_1=ppda_1,
            ppda_team_2=ppda_2,
        )

    def _gather_all_pitch_data(
        self,
        tracks: Dict[str, list],
        per_frame_transformers: Dict,
    ) -> Dict[int, Dict]:
        """Gather pitch-space positions for all sampled frames.

        Returns dict: frame_idx -> {"players": [(team_id, (x,y))], "ball": (x,y)|None}
        """
        player_frames = tracks.get("players", [])
        gk_frames = tracks.get("goalkeepers", [])
        ball_frames = tracks.get("ball", [])
        n_frames = max(len(player_frames), len(gk_frames), len(ball_frames), 1)

        result: Dict[int, Dict] = {}

        for frame_idx in range(0, n_frames, _FRAME_SAMPLE_STEP):
            transformer = per_frame_transformers.get(frame_idx)
            if transformer is None:
                continue

            players: List[Tuple[int, Tuple[float, float]]] = []

            # Players + goalkeepers
            frame_entities: Dict = {}
            if frame_idx < len(player_frames):
                frame_entities.update(player_frames[frame_idx])
            if frame_idx < len(gk_frames):
                frame_entities.update(gk_frames[frame_idx])

            for track_id, data in frame_entities.items():
                bbox = data.get("bbox")
                if bbox is None:
                    continue
                raw_tid = data.get("team_id")
                if raw_tid not in (0, 1):
                    continue
                # Remap 0→1, 1→2 to match possession convention
                team_id = raw_tid + 1

                foot = get_foot_position(bbox)
                try:
                    arr = np.array([foot], dtype=np.float32)
                    pitch = transformer.transform_points(arr)
                    px, py = float(pitch[0][0]), float(pitch[0][1])
                except Exception:
                    continue

                if not (0 <= px <= _PITCH_LENGTH_CM and 0 <= py <= _PITCH_WIDTH_CM):
                    continue

                players.append((team_id, (px, py)))

            # Ball
            ball_pos = None
            if frame_idx < len(ball_frames):
                ball_data = ball_frames[frame_idx].get(1)
                if ball_data and ball_data.get("bbox") is not None:
                    center = get_center_of_bbox(ball_data["bbox"])
                    try:
                        arr = np.array([center], dtype=np.float32)
                        pitch = transformer.transform_points(arr)
                        bx, by = float(pitch[0][0]), float(pitch[0][1])
                        if 0 <= bx <= _PITCH_LENGTH_CM and 0 <= by <= _PITCH_WIDTH_CM:
                            ball_pos = (bx, by)
                    except Exception:
                        pass

            if players:
                result[frame_idx] = {"players": players, "ball": ball_pos}

        return result

    @staticmethod
    def _aggregate_summary(
        windows: List[TacticalWindow],
        ppda_1: Optional[float],
        ppda_2: Optional[float],
    ) -> Dict:
        """Aggregate window metrics into a summary dict for LLM grounding."""
        if not windows:
            return {}

        def _mean(vals: list) -> Optional[float]:
            filtered = [v for v in vals if v is not None]
            return round(float(np.mean(filtered)), 2) if filtered else None

        def _windows_by_phase(team: int, phase: str) -> List[TacticalWindow]:
            attr = f"phase_team_{team}"
            return [w for w in windows if getattr(w, attr, None) == phase]

        # Overall averages
        summary: Dict = {
            "team_1_avg_compactness_m2": _mean([w.team_1_compactness for w in windows]),
            "team_2_avg_compactness_m2": _mean([w.team_2_compactness for w in windows]),
            "team_1_avg_stretch_index_m": _mean([w.team_1_stretch_index for w in windows]),
            "team_2_avg_stretch_index_m": _mean([w.team_2_stretch_index for w in windows]),
            "team_1_avg_length_m": _mean([w.team_1_length for w in windows]),
            "team_2_avg_length_m": _mean([w.team_2_length for w in windows]),
            "team_1_avg_width_m": _mean([w.team_1_width for w in windows]),
            "team_2_avg_width_m": _mean([w.team_2_width for w in windows]),
            "team_1_avg_defensive_line_m": _mean([w.team_1_defensive_line for w in windows]),
            "team_2_avg_defensive_line_m": _mean([w.team_2_defensive_line for w in windows]),
            "team_1_avg_pressing_intensity": _mean([w.team_1_pressing_intensity for w in windows]),
            "team_2_avg_pressing_intensity": _mean([w.team_2_pressing_intensity for w in windows]),
            "avg_inter_team_distance_m": _mean([w.inter_team_distance for w in windows]),
            "ppda_team_1": round(ppda_1, 2) if ppda_1 is not None else None,
            "ppda_team_2": round(ppda_2, 2) if ppda_2 is not None else None,
            # Territory control (A2)
            "team_1_avg_territory_pct": _mean([w.team_1_territory_pct for w in windows]),
            "team_2_avg_territory_pct": _mean([w.team_2_territory_pct for w in windows]),
            "team_1_avg_opp_half_territory_pct": _mean([w.team_1_opp_half_territory_pct for w in windows]),
            "team_2_avg_opp_half_territory_pct": _mean([w.team_2_opp_half_territory_pct for w in windows]),
            # Press classification distribution (A5)
            "team_1_press_type_distribution": {
                pt: sum(1 for w in windows if w.team_1_press_type == pt)
                for pt in ("high", "mid", "low")
            },
            "team_2_press_type_distribution": {
                pt: sum(1 for w in windows if w.team_2_press_type == pt)
                for pt in ("high", "mid", "low")
            },
            "team_1_counter_press_windows": sum(1 for w in windows if w.team_1_counter_press),
            "team_2_counter_press_windows": sum(1 for w in windows if w.team_2_counter_press),
        }

        # Phase-of-play breakdown (A1): per-phase averages for key metrics
        for team in (1, 2):
            for phase in ("ip", "oop", "dat", "adt"):
                phase_wins = _windows_by_phase(team, phase)
                if not phase_wins:
                    continue
                prefix = f"team_{team}_{phase}"
                summary[f"{prefix}_compactness_m2"] = _mean(
                    [getattr(w, f"team_{team}_compactness") for w in phase_wins]
                )
                summary[f"{prefix}_stretch_index_m"] = _mean(
                    [getattr(w, f"team_{team}_stretch_index") for w in phase_wins]
                )
                summary[f"{prefix}_defensive_line_m"] = _mean(
                    [getattr(w, f"team_{team}_defensive_line") for w in phase_wins]
                )
                summary[f"{prefix}_pressing_intensity"] = _mean(
                    [getattr(w, f"team_{team}_pressing_intensity") for w in phase_wins]
                )
                summary[f"{prefix}_territory_pct"] = _mean(
                    [getattr(w, f"team_{team}_territory_pct") for w in phase_wins]
                )
                summary[f"{prefix}_window_count"] = len(phase_wins)

        return summary
