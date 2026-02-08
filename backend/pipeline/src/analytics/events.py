"""Football event detection from tracking data.

Detects passes, shots, and tackles by analysing ball-possession transitions,
ball velocity spikes, and team-change patterns.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance
except ImportError:
    from src.utils.bbox_utils import get_center_of_bbox, get_foot_position, measure_distance

from .types import FootballEvent, PossessionEvent


# ── Tuning constants ────────────────────────────────────────────────

# Minimum frames between two events of the same type to avoid duplicates
_MIN_EVENT_GAP_FRAMES = 5

# Pass detection
_PASS_MAX_FRAMES = 30  # Max frames for ball to travel between teammates
_PASS_MIN_FRAMES = 2   # Ignore instant same-frame flickers

# Shot detection — ball speed threshold in cm/frame (≈ 60 km/h at 25 fps)
_SHOT_SPEED_THRESHOLD_CM = 200.0  # Generous: a hard shot is 100-130 km/h

# Goal regions on a standard pitch (10500 × 6800 cm)
# Behind either goal-line, within the goal-width band (≈ 732 cm = 7.32 m)
_GOAL_Y_MIN = (6800 - 732) / 2   # ≈ 3034 cm
_GOAL_Y_MAX = (6800 + 732) / 2   # ≈ 3766 cm
_GOAL_X_MARGIN = 800  # Within 8 m of goal-line

# Tackle: possession changes between teams within this many frames
_TACKLE_WINDOW = 8


class EventDetector:
    """Detect football events from per-frame possession and ball tracking."""

    def __init__(self, fps: float = 25.0, pitch_length: float = 10500.0,
                 pitch_width: float = 6800.0):
        self.fps = fps
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    # ── Public API ──────────────────────────────────────────────────

    def detect(
        self,
        possession_events: List[PossessionEvent],
        tracks: Dict[str, List[Dict]],
        per_frame_transformers: Optional[Dict] = None,
    ) -> List[FootballEvent]:
        """Run all detectors and return merged, sorted events.

        Args:
            possession_events: Per-frame possession from PossessionCalculator.
            tracks: Full tracks dict with "ball", "players", "goalkeepers".
            per_frame_transformers: Optional per-frame ViewTransformers for
                ball pitch-coordinate lookups.

        Returns:
            Deduplicated list of FootballEvent sorted by frame index.
        """
        events: List[FootballEvent] = []
        events.extend(self._detect_passes(possession_events))
        events.extend(self._detect_shots(possession_events, tracks,
                                          per_frame_transformers))
        events.extend(self._detect_tackles(possession_events))
        events.sort(key=lambda e: e.frame_idx)
        return self._deduplicate(events)

    # ── Pass detection ──────────────────────────────────────────────

    def _detect_passes(
        self,
        possession_events: List[PossessionEvent],
    ) -> List[FootballEvent]:
        """Detect passes: possession changes from player A → player B on
        the same team within ``_PASS_MAX_FRAMES``.
        """
        passes: List[FootballEvent] = []

        # Build runs of controlled-possession by the same player
        i = 0
        n = len(possession_events)

        while i < n:
            ev = possession_events[i]

            # Skip uncontrolled / contested frames
            if not ev.is_controlled or ev.team_id == 0 or ev.player_track_id is None:
                i += 1
                continue

            # Start of a possession run for (player, team)
            run_player = ev.player_track_id
            run_team = ev.team_id
            run_start = ev.frame_idx
            j = i + 1

            # Advance through consecutive frames for this player
            while j < n:
                ej = possession_events[j]
                if ej.is_controlled and ej.player_track_id == run_player and ej.team_id == run_team:
                    j += 1
                else:
                    break

            run_end_frame = possession_events[j - 1].frame_idx

            # Look ahead for the next controlled possession by same team
            k = j
            while k < n and (k - j) < _PASS_MAX_FRAMES:
                ek = possession_events[k]
                if ek.is_controlled and ek.team_id == run_team and ek.player_track_id is not None:
                    if ek.player_track_id != run_player:
                        gap = ek.frame_idx - run_end_frame
                        if _PASS_MIN_FRAMES <= gap <= _PASS_MAX_FRAMES:
                            passes.append(FootballEvent(
                                event_type="pass",
                                frame_idx=run_end_frame,
                                timestamp_sec=run_end_frame / self.fps,
                                team_id=run_team,
                                player_track_id=run_player,
                                target_player_track_id=ek.player_track_id,
                                confidence=max(0.5, 1.0 - gap / _PASS_MAX_FRAMES),
                                success=True,
                            ))
                    break
                # If possession goes to the other team, not a pass
                if ek.is_controlled and ek.team_id != 0 and ek.team_id != run_team:
                    break
                k += 1

            i = j

        return passes

    # ── Shot detection ──────────────────────────────────────────────

    def _detect_shots(
        self,
        possession_events: List[PossessionEvent],
        tracks: Dict[str, List[Dict]],
        per_frame_transformers: Optional[Dict] = None,
    ) -> List[FootballEvent]:
        """Detect shots: ball velocity spike + trajectory toward goal region."""
        shots: List[FootballEvent] = []
        ball_frames = tracks.get("ball", [])
        n = len(ball_frames)
        if n < 3:
            return shots

        # Get ball pitch positions
        ball_positions: List[Optional[Tuple[float, float]]] = []
        for fi, bf in enumerate(ball_frames):
            pos = self._ball_pitch_pos(fi, bf, per_frame_transformers)
            ball_positions.append(pos)

        # Compute per-frame ball speed (cm/frame)
        speeds: List[float] = [0.0]
        for i in range(1, n):
            prev = ball_positions[i - 1]
            curr = ball_positions[i]
            if prev is not None and curr is not None:
                d = measure_distance(prev, curr)
                speeds.append(d)
            else:
                speeds.append(0.0)

        # Scan for velocity spikes heading toward a goal
        for i in range(2, n):
            if speeds[i] < _SHOT_SPEED_THRESHOLD_CM:
                continue

            curr_pos = ball_positions[i]
            prev_pos = ball_positions[i - 1]
            if curr_pos is None or prev_pos is None:
                continue

            # Check direction: heading toward either goal-line?
            dx = curr_pos[0] - prev_pos[0]
            toward_goal_left = dx < -50 and curr_pos[0] < _GOAL_X_MARGIN
            toward_goal_right = dx > 50 and curr_pos[0] > (self.pitch_length - _GOAL_X_MARGIN)

            if not (toward_goal_left or toward_goal_right):
                continue

            # Check Y is within goal-width band
            if not (_GOAL_Y_MIN <= curr_pos[1] <= _GOAL_Y_MAX):
                continue

            # Attribute to last possessing player
            team_id = None
            player_id = None
            # Look back a few frames for last controlled possession
            poss_idx = min(i, len(possession_events) - 1)
            for look_back in range(min(10, poss_idx + 1)):
                pe = possession_events[poss_idx - look_back]
                if pe.is_controlled and pe.team_id != 0:
                    team_id = pe.team_id
                    player_id = pe.player_track_id
                    break

            shots.append(FootballEvent(
                event_type="shot",
                frame_idx=i,
                timestamp_sec=i / self.fps,
                team_id=team_id,
                player_track_id=player_id,
                confidence=min(1.0, speeds[i] / (_SHOT_SPEED_THRESHOLD_CM * 2)),
                pitch_start=prev_pos,
                pitch_end=curr_pos,
            ))

        return shots

    # ── Tackle / duel detection ─────────────────────────────────────

    def _detect_tackles(
        self,
        possession_events: List[PossessionEvent],
    ) -> List[FootballEvent]:
        """Detect tackles: rapid team-to-team possession changes."""
        tackles: List[FootballEvent] = []
        n = len(possession_events)

        i = 0
        while i < n:
            ev = possession_events[i]
            if not ev.is_controlled or ev.team_id == 0:
                i += 1
                continue

            team_a = ev.team_id
            player_a = ev.player_track_id

            # Look ahead for a quick team change
            for j in range(i + 1, min(i + _TACKLE_WINDOW + 1, n)):
                ej = possession_events[j]
                if ej.is_controlled and ej.team_id != 0 and ej.team_id != team_a:
                    tackles.append(FootballEvent(
                        event_type="challenge",
                        frame_idx=ej.frame_idx,
                        timestamp_sec=ej.frame_idx / self.fps,
                        team_id=ej.team_id,  # Team that won the ball
                        player_track_id=ej.player_track_id,
                        target_player_track_id=player_a,  # Player who lost ball
                        confidence=max(0.4, 1.0 - (j - i) / _TACKLE_WINDOW),
                    ))
                    i = j  # Skip past this tackle
                    break
            else:
                i += 1
                continue
            i += 1

        return tackles

    # ── Helpers ──────────────────────────────────────────────────────

    def _ball_pitch_pos(
        self,
        frame_idx: int,
        ball_frame: Dict,
        per_frame_transformers: Optional[Dict],
    ) -> Optional[Tuple[float, float]]:
        """Get ball pitch position for a frame, if possible."""
        if 1 not in ball_frame:
            return None
        bbox = ball_frame[1].get("bbox")
        if bbox is None:
            return None

        pixel_pos = get_center_of_bbox(bbox)

        if per_frame_transformers and frame_idx in per_frame_transformers:
            try:
                arr = np.array([pixel_pos], dtype=np.float32)
                pitch = per_frame_transformers[frame_idx].transform_points(arr)
                return (float(pitch[0][0]), float(pitch[0][1]))
            except Exception:
                pass
        return None

    @staticmethod
    def _deduplicate(events: List[FootballEvent]) -> List[FootballEvent]:
        """Remove events of the same type that are too close together."""
        if not events:
            return events

        result: List[FootballEvent] = [events[0]]
        for ev in events[1:]:
            last = result[-1]
            if ev.event_type == last.event_type and (ev.frame_idx - last.frame_idx) < _MIN_EVENT_GAP_FRAMES:
                # Keep the one with higher confidence
                if ev.confidence is not None and (last.confidence is None or ev.confidence > last.confidence):
                    result[-1] = ev
            else:
                result.append(ev)
        return result

    # ── Aggregation helpers for Statistics ───────────────────────────

    @staticmethod
    def count_by_team_and_type(
        events: List[FootballEvent],
    ) -> Dict[str, Dict[int, int]]:
        """Count events by type and team.

        Returns:
            {"pass": {1: 5, 2: 3}, "shot": {1: 2, 2: 1}, ...}
        """
        counts: Dict[str, Dict[int, int]] = {}
        for ev in events:
            if ev.event_type not in counts:
                counts[ev.event_type] = {}
            tid = ev.team_id or 0
            counts[ev.event_type][tid] = counts[ev.event_type].get(tid, 0) + 1
        return counts
