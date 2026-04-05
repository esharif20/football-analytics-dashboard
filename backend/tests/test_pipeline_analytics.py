"""Tests for the pipeline analytics modules.

Tests cover:
- bbox_utils: float precision, measure_distance
- kinematics: speed calculations, smoothing, dead zone, ball speed cap
- possession: pixel vs real-world thresholds, aggregation
- events: pitch dimensions, shot detection threshold, goal region bounds
"""

import sys
import os
import math
from unittest.mock import MagicMock
from collections import deque

# ── Mock heavy CV dependencies before any pipeline imports ──────────────
# The pipeline utils/__init__.py imports video_utils which requires cv2,
# and pitch modules require cv2 for homography.  Mock them so analytics
# tests can run without GPU / OpenCV installed.

_cv2_mock = MagicMock()
sys.modules.setdefault("cv2", _cv2_mock)
sys.modules.setdefault("supervision", MagicMock())
sys.modules.setdefault("ultralytics", MagicMock())
sys.modules.setdefault("torch", MagicMock())

# Add src to path so pipeline modules can be imported
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline", "src")
sys.path.insert(0, _src_dir)

import pytest
import numpy as np


# ─── bbox_utils tests ──────────────────────────────────────────────────────


class TestBboxUtils:
    """Test bbox utility functions return float and correct values."""

    def test_get_center_of_bbox_returns_float(self):
        from utils.bbox_utils import get_center_of_bbox
        cx, cy = get_center_of_bbox((10, 20, 30, 40))
        assert isinstance(cx, float), f"Expected float, got {type(cx)}"
        assert isinstance(cy, float), f"Expected float, got {type(cy)}"
        assert cx == 20.0
        assert cy == 30.0

    def test_get_center_of_bbox_preserves_subpixel(self):
        from utils.bbox_utils import get_center_of_bbox
        cx, cy = get_center_of_bbox((10, 20, 31, 41))
        assert cx == 20.5, f"Expected 20.5, got {cx}"
        assert cy == 30.5, f"Expected 30.5, got {cy}"

    def test_get_foot_position_returns_float(self):
        from utils.bbox_utils import get_foot_position
        fx, fy = get_foot_position((10, 20, 30, 40))
        assert isinstance(fx, float), f"Expected float, got {type(fx)}"
        assert isinstance(fy, float), f"Expected float, got {type(fy)}"
        assert fx == 20.0
        assert fy == 40.0

    def test_get_foot_position_preserves_subpixel(self):
        from utils.bbox_utils import get_foot_position
        fx, fy = get_foot_position((10, 20, 31, 41))
        assert fx == 20.5, f"Expected 20.5, got {fx}"
        assert fy == 41.0

    def test_measure_distance(self):
        from utils.bbox_utils import measure_distance
        d = measure_distance((0, 0), (3, 4))
        assert d == pytest.approx(5.0)

    def test_measure_distance_zero(self):
        from utils.bbox_utils import measure_distance
        d = measure_distance((5, 5), (5, 5))
        assert d == pytest.approx(0.0)


# ─── kinematics tests ──────────────────────────────────────────────────────


class TestKinematics:
    """Test kinematics calculator: speed, distance, smoothing, caps."""

    def _make_positions(self, coords, fps=25.0):
        """Helper to create FramePositions with pitch coords."""
        from analytics.types import FramePosition
        positions = []
        for i, (px, py) in enumerate(coords):
            positions.append(FramePosition(
                frame_idx=i,
                pixel_pos=(0.0, 0.0),
                pitch_pos=(px, py),
                timestamp_sec=i / fps,
            ))
        return positions

    def test_speed_cap_applied_to_players(self):
        """Player speeds should be capped at MAX_PLAYER_SPEED_KMH."""
        from analytics.kinematics import KinematicsCalculator, MAX_PLAYER_SPEED_KMH

        calc = KinematicsCalculator(fps=25.0)
        # 10000 cm in 1 frame at 25fps = 10000*0.01 m / 0.04s = 2500 m/s → capped
        positions = self._make_positions([(0, 0), (10000, 0)])
        stats = calc.compute_stats(positions, track_id=1, entity_type="player", use_adaptive=True)
        max_speed_kmh = stats.max_speed_m_per_sec * 3.6 if stats.max_speed_m_per_sec else 0
        assert max_speed_kmh <= MAX_PLAYER_SPEED_KMH + 0.1

    def test_speed_cap_applied_to_ball(self):
        """Ball speeds should be capped at MAX_BALL_SPEED_KMH (200)."""
        from analytics.kinematics import KinematicsCalculator, MAX_BALL_SPEED_KMH

        calc = KinematicsCalculator(fps=25.0)
        # Extreme distance to trigger cap
        positions = self._make_positions([(0, 0), (100000, 0)])
        stats = calc.compute_stats(positions, track_id=1, entity_type="ball", use_adaptive=True)
        max_speed_kmh = stats.max_speed_m_per_sec * 3.6 if stats.max_speed_m_per_sec else 0
        assert max_speed_kmh <= MAX_BALL_SPEED_KMH + 0.1, (
            f"Ball speed {max_speed_kmh:.1f} km/h exceeds cap {MAX_BALL_SPEED_KMH}"
        )

    def test_ball_speed_cap_constant_exists(self):
        """Verify that ball speed cap constant exists and is reasonable."""
        from analytics.kinematics import MAX_BALL_SPEED_KMH
        assert MAX_BALL_SPEED_KMH == 200.0

    def test_smooth_positions_alpha_040(self):
        """With alpha=0.40, positions should respond within a few frames."""
        from analytics.kinematics import KinematicsCalculator
        from analytics.types import FramePosition

        # Sudden jump from (0,0) to (1000,0)
        positions = [
            FramePosition(frame_idx=0, pixel_pos=(0, 0), pitch_pos=(0.0, 0.0)),
            FramePosition(frame_idx=1, pixel_pos=(0, 0), pitch_pos=(1000.0, 0.0)),
            FramePosition(frame_idx=2, pixel_pos=(0, 0), pitch_pos=(1000.0, 0.0)),
            FramePosition(frame_idx=3, pixel_pos=(0, 0), pitch_pos=(1000.0, 0.0)),
            FramePosition(frame_idx=4, pixel_pos=(0, 0), pitch_pos=(1000.0, 0.0)),
        ]
        smoothed = KinematicsCalculator._smooth_pitch_positions(positions, alpha=0.40)
        # After 4 frames at α=0.40, should be at least 80% of the way to 1000
        assert smoothed[4].pitch_pos[0] > 800, (
            f"After 4 frames, pos should be >800 but is {smoothed[4].pitch_pos[0]:.1f}"
        )

    def test_smooth_positions_new_faster_than_old(self):
        """New alpha=0.40 should converge dramatically faster than old alpha=0.10."""
        from analytics.kinematics import KinematicsCalculator
        from analytics.types import FramePosition

        def _make():
            return [
                FramePosition(frame_idx=0, pixel_pos=(0, 0), pitch_pos=(0.0, 0.0)),
                FramePosition(frame_idx=1, pixel_pos=(0, 0), pitch_pos=(1000.0, 0.0)),
                FramePosition(frame_idx=2, pixel_pos=(0, 0), pitch_pos=(1000.0, 0.0)),
                FramePosition(frame_idx=3, pixel_pos=(0, 0), pitch_pos=(1000.0, 0.0)),
                FramePosition(frame_idx=4, pixel_pos=(0, 0), pitch_pos=(1000.0, 0.0)),
            ]

        smoothed_old = KinematicsCalculator._smooth_pitch_positions(_make(), alpha=0.10)
        smoothed_new = KinematicsCalculator._smooth_pitch_positions(_make(), alpha=0.40)
        assert smoothed_new[4].pitch_pos[0] > smoothed_old[4].pitch_pos[0] * 1.5, (
            "New smoothing should reach target significantly faster than old"
        )

    def test_dead_zone_preserves_real_movement(self):
        """Speeds above 0.4 m/s should NOT be zeroed."""
        from analytics.kinematics import KinematicsCalculator

        calc = KinematicsCalculator(fps=25.0)
        # 100 cm per frame at 25fps = 1 m / 0.04 sec = 25 m/s
        positions = self._make_positions([(0, 0), (100, 0)])
        _, _, dist_m, speeds_m = calc.compute_distances_and_speeds_adaptive(
            positions, max_speed_m_per_sec=40.0 / 3.6
        )
        assert speeds_m[0] is not None
        assert speeds_m[0] > 0.4, "Real movement should not be zeroed by dead zone"

    def test_dead_zone_value_is_0_4(self):
        """Dead zone should be 0.4 m/s (not the old 1.0 m/s)."""
        import inspect
        from analytics.kinematics import KinematicsCalculator
        source = inspect.getsource(KinematicsCalculator.build_per_frame_lookup)
        assert "dead_zone = 0.4" in source, "Dead zone should be 0.4, not 1.0"

    def test_ema_alpha_is_0_20(self):
        """Display EMA alpha should be 0.20 (not old 0.06)."""
        import inspect
        from analytics.kinematics import KinematicsCalculator
        source = inspect.getsource(KinematicsCalculator.build_per_frame_lookup)
        assert "ema_alpha = 0.20" in source, "EMA alpha should be 0.20, not 0.06"

    def test_compute_distances_basic(self):
        """Basic distance computation with known pitch positions."""
        from analytics.kinematics import KinematicsCalculator

        calc = KinematicsCalculator(fps=25.0)
        # 300 cm apart → 3 metres
        positions = self._make_positions([(0, 0), (300, 0)])
        _, _, dist_m, speeds_m = calc.compute_distances_and_speeds(positions)
        assert dist_m is not None
        assert len(dist_m) == 1
        assert dist_m[0] == pytest.approx(3.0)  # 300 cm * 0.01 = 3 m

    def test_compute_speed_basic(self):
        """Speed = distance / time, verify the calculation."""
        from analytics.kinematics import KinematicsCalculator

        calc = KinematicsCalculator(fps=25.0)
        # 500 cm in 1 frame at 25fps = 5m / 0.04s = 125 m/s → but capped at 40/3.6
        positions = self._make_positions([(0, 0), (500, 0)])
        _, _, dist_m, speeds_m = calc.compute_distances_and_speeds(
            positions, max_speed_m_per_sec=200.0
        )
        assert speeds_m is not None
        # 500 cm = 5m, time = 1/25 = 0.04s, speed = 5/0.04 = 125 m/s
        assert speeds_m[0] == pytest.approx(125.0)


# ─── possession tests ──────────────────────────────────────────────────────


class TestPossession:
    """Test possession calculator: thresholds, aggregation."""

    def test_pixel_threshold_close_player(self):
        """Player within 100px should have possession."""
        from analytics.possession import PossessionCalculator

        calc = PossessionCalculator()
        event = calc.calculate_frame_possession(
            ball_frame={1: {"bbox": (100, 100, 110, 110)}},
            players_frame={
                1: {"bbox": (80, 90, 120, 130), "team_id": 0},
            },
            goalkeepers_frame={},
            frame_idx=0,
            use_real_units=False,
        )
        assert event.is_controlled is True
        assert event.team_id == 1  # team_id 0 → exposed as 1

    def test_pixel_threshold_far_player(self):
        """Player beyond 100px should NOT have possession."""
        from analytics.possession import PossessionCalculator

        calc = PossessionCalculator()
        event = calc.calculate_frame_possession(
            ball_frame={1: {"bbox": (100, 100, 110, 110)}},
            players_frame={
                1: {"bbox": (300, 300, 340, 360), "team_id": 0},
            },
            goalkeepers_frame={},
            frame_idx=0,
            use_real_units=False,
        )
        assert event.is_controlled is False
        assert event.team_id == 0  # contested

    def test_real_units_threshold(self):
        """With use_real_units=True, threshold should be 300 cm not 100 px."""
        from analytics.possession import PossessionCalculator

        calc = PossessionCalculator()
        # Ball at (500,500), player foot at (700,520) — distance ≈ 200.1 cm
        # With real units threshold of 300 cm, this should be controlled
        event = calc.calculate_frame_possession(
            ball_frame={1: {"bbox": (495, 495, 505, 505)}},
            players_frame={
                1: {"bbox": (690, 500, 710, 520), "team_id": 0},
            },
            goalkeepers_frame={},
            frame_idx=0,
            use_real_units=True,
        )
        assert event.is_controlled is True

    def test_calculate_all_frames_accepts_per_frame_transformers(self):
        """calculate_all_frames should accept per_frame_transformers param."""
        from analytics.possession import PossessionCalculator

        calc = PossessionCalculator()
        tracks = {
            "ball": [{1: {"bbox": (50, 50, 60, 60)}}, {1: {"bbox": (50, 50, 60, 60)}}],
            "players": [
                {1: {"bbox": (40, 40, 60, 70), "team_id": 0}},
                {1: {"bbox": (40, 40, 60, 70), "team_id": 0}},
            ],
            "goalkeepers": [{}, {}],
        }
        # With per_frame_transformers, should use real units
        events = calc.calculate_all_frames(tracks, per_frame_transformers={0: "dummy"})
        assert len(events) == 2

    def test_aggregation(self):
        """Verify possession stats aggregation correctness."""
        from analytics.possession import PossessionCalculator
        from analytics.types import PossessionEvent

        events = [
            PossessionEvent(frame_idx=0, team_id=1, player_track_id=1, distance_to_ball=50, is_controlled=True),
            PossessionEvent(frame_idx=1, team_id=1, player_track_id=1, distance_to_ball=50, is_controlled=True),
            PossessionEvent(frame_idx=2, team_id=2, player_track_id=2, distance_to_ball=50, is_controlled=True),
            PossessionEvent(frame_idx=3, team_id=0, player_track_id=None, distance_to_ball=999, is_controlled=False),
        ]
        calc = PossessionCalculator()
        stats = calc.aggregate_stats(events)
        assert stats.team_1_frames == 2
        assert stats.team_2_frames == 1
        assert stats.contested_frames == 1
        assert stats.team_1_percentage == pytest.approx(66.67, abs=0.1)
        assert stats.team_2_percentage == pytest.approx(33.33, abs=0.1)
        assert stats.possession_changes == 1


# ─── events tests ──────────────────────────────────────────────────────────


class TestEventDetector:
    """Test event detector: pitch dimensions, shot threshold, goal regions."""

    def test_default_pitch_dimensions(self):
        """EventDetector should use 12000×7000 by default."""
        from analytics.events import EventDetector
        det = EventDetector()
        assert det.pitch_length == 12000.0
        assert det.pitch_width == 7000.0

    def test_goal_region_bounds_match_pitch(self):
        """Goal Y region should be centered on the pitch width (7000)."""
        from analytics.events import EventDetector, _GOAL_WIDTH_CM
        det = EventDetector()  # 12000 × 7000
        expected_y_min = (7000 - _GOAL_WIDTH_CM) / 2
        expected_y_max = (7000 + _GOAL_WIDTH_CM) / 2
        assert det._goal_y_min == pytest.approx(expected_y_min)
        assert det._goal_y_max == pytest.approx(expected_y_max)

    def test_custom_pitch_dimensions(self):
        """Goal regions should adapt to custom pitch size."""
        from analytics.events import EventDetector, _GOAL_WIDTH_CM
        det = EventDetector(pitch_length=10500, pitch_width=6800)
        expected_y_min = (6800 - _GOAL_WIDTH_CM) / 2
        expected_y_max = (6800 + _GOAL_WIDTH_CM) / 2
        assert det._goal_y_min == pytest.approx(expected_y_min)
        assert det._goal_y_max == pytest.approx(expected_y_max)

    def test_shot_threshold_is_reasonable(self):
        """Shot threshold should catch shots >= ~80 km/h, not 180 km/h."""
        from analytics.events import _SHOT_SPEED_THRESHOLD_CM
        # At 25 fps: threshold_cm/frame * fps * 0.01 * 3.6 = km/h
        speed_kmh = _SHOT_SPEED_THRESHOLD_CM * 25 * 0.01 * 3.6
        assert speed_kmh < 120, f"Shot threshold = {speed_kmh:.0f} km/h — too high"
        assert speed_kmh > 50, f"Shot threshold = {speed_kmh:.0f} km/h — too low"

    def test_pass_detection(self):
        """Passes: same-team player A → player B should be detected."""
        from analytics.events import EventDetector
        from analytics.types import PossessionEvent

        det = EventDetector(fps=25.0)
        events = []
        # Player 1 (team 1) has possession for frames 0-9
        for i in range(10):
            events.append(PossessionEvent(
                frame_idx=i, team_id=1, player_track_id=1,
                distance_to_ball=50, is_controlled=True,
            ))
        # Gap frames 10-14 (contested)
        for i in range(10, 15):
            events.append(PossessionEvent(
                frame_idx=i, team_id=0, player_track_id=None,
                distance_to_ball=500, is_controlled=False,
            ))
        # Player 2 (team 1) gets possession at frame 15
        for i in range(15, 25):
            events.append(PossessionEvent(
                frame_idx=i, team_id=1, player_track_id=2,
                distance_to_ball=50, is_controlled=True,
            ))

        tracks = {"ball": [{1: {"bbox": (50, 50, 60, 60)}} for _ in range(25)]}
        detected = det.detect(events, tracks)
        passes = [e for e in detected if e.event_type == "pass"]
        assert len(passes) >= 1, "Should detect at least one pass"
        assert passes[0].team_id == 1
        assert passes[0].player_track_id == 1
        assert passes[0].target_player_track_id == 2

    def test_tackle_detection(self):
        """Tackles: rapid team-to-team possession changes should be detected."""
        from analytics.events import EventDetector
        from analytics.types import PossessionEvent

        det = EventDetector(fps=25.0)
        events = []
        # Team 1 has possession
        for i in range(5):
            events.append(PossessionEvent(
                frame_idx=i, team_id=1, player_track_id=1,
                distance_to_ball=50, is_controlled=True,
            ))
        # Quick switch to team 2
        for i in range(5, 10):
            events.append(PossessionEvent(
                frame_idx=i, team_id=2, player_track_id=2,
                distance_to_ball=50, is_controlled=True,
            ))

        tracks = {"ball": [{1: {"bbox": (50, 50, 60, 60)}} for _ in range(10)]}
        detected = det.detect(events, tracks)
        tackles = [e for e in detected if e.event_type == "challenge"]
        assert len(tackles) >= 1, "Should detect at least one tackle"

    def test_deduplication(self):
        """Events of same type within 5 frames should be deduplicated."""
        from analytics.events import EventDetector
        from analytics.types import FootballEvent

        det = EventDetector()
        events = [
            FootballEvent(event_type="pass", frame_idx=10, timestamp_sec=0.4,
                          team_id=1, confidence=0.8),
            FootballEvent(event_type="pass", frame_idx=12, timestamp_sec=0.48,
                          team_id=1, confidence=0.9),
        ]
        deduped = det._deduplicate(events)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.9  # Higher confidence kept


# ─── rolling window possession overlay tests ───────────────────────────────


class TestRollingWindowPossession:
    """Test that the rolling-window approach is responsive."""

    def test_rolling_window_responds_to_changes(self):
        """After switching teams, possession % should converge correctly."""
        window = 250
        recent: deque = deque(maxlen=window)
        results = []

        # 100 frames team 1, then 100 frames team 2
        for _ in range(100):
            recent.append(1)
            t1 = sum(1 for t in recent if t == 1)
            t2 = sum(1 for t in recent if t == 2)
            total = t1 + t2
            results.append((t1 / total * 100) if total > 0 else 50)

        for _ in range(100):
            recent.append(2)
            t1 = sum(1 for t in recent if t == 1)
            t2 = sum(1 for t in recent if t == 2)
            total = t1 + t2
            results.append((t1 / total * 100) if total > 0 else 50)

        # At frame 199 (100 team1 + 100 team2), should be 50%
        assert results[199] == pytest.approx(50.0)

        # Push team 1 out of window completely (150 more team2 frames)
        for _ in range(150):
            recent.append(2)
            t1 = sum(1 for t in recent if t == 1)
            t2 = sum(1 for t in recent if t == 2)
            total = t1 + t2
            results.append((t1 / total * 100) if total > 0 else 50)

        # After 350 total (100 team1, 250 team2), team 1 fully out of window
        assert results[-1] == pytest.approx(0.0)

    def test_rolling_window_not_rigid(self):
        """Unlike cumulative, rolling window should reflect recent changes."""
        window = 250
        recent: deque = deque(maxlen=window)

        # 500 frames of team 1
        for _ in range(500):
            recent.append(1)
        # Then 250 frames of team 2 (fills entire window)
        for _ in range(250):
            recent.append(2)

        t1 = sum(1 for t in recent if t == 1)
        t2 = sum(1 for t in recent if t == 2)
        total = t1 + t2
        pct = (t1 / total * 100) if total > 0 else 50

        # Should be 0% team 1 — rolling window only sees last 250 frames
        assert pct == pytest.approx(0.0), (
            f"Expected 0% team1, got {pct:.1f}% — window should forget old data"
        )


# ─── pass detection boundary / edge-case tests ────────────────────────────


class TestPassEdgeCases:
    """Test pass detection boundary conditions and cross-team transfers."""

    def _make_poss(self, frame_idx, team_id, player_id, controlled=True):
        from analytics.types import PossessionEvent
        return PossessionEvent(
            frame_idx=frame_idx,
            team_id=team_id,
            player_track_id=player_id,
            distance_to_ball=50,
            is_controlled=controlled,
        )

    def test_pass_max_frames_boundary_detected(self):
        """A pass whose ball-travel gap equals _PASS_MAX_FRAMES should be detected."""
        from analytics.events import EventDetector, _PASS_MAX_FRAMES

        det = EventDetector(fps=25.0)
        events = []
        # Player 1 holds the ball frames 0-4; run_end_frame = 4
        for i in range(5):
            events.append(self._make_poss(i, team_id=1, player_id=1))
        # Contested gap so that receiver lands at exactly run_end_frame + _PASS_MAX_FRAMES
        # run_end_frame = 4, receiver at 4 + _PASS_MAX_FRAMES → gap = _PASS_MAX_FRAMES exactly
        recv_frame = 4 + _PASS_MAX_FRAMES
        for i in range(5, recv_frame):
            events.append(self._make_poss(i, team_id=0, player_id=None, controlled=False))
        events.append(self._make_poss(recv_frame, team_id=1, player_id=2))

        tracks = {"ball": [{1: {"bbox": (50, 50, 60, 60)}} for _ in range(recv_frame + 1)]}
        detected = det.detect(events, tracks)
        passes = [e for e in detected if e.event_type == "pass"]
        assert len(passes) >= 1, (
            f"Pass with gap == _PASS_MAX_FRAMES ({_PASS_MAX_FRAMES}) should be detected"
        )

    def test_pass_beyond_max_frames_not_detected(self):
        """A pass whose ball-travel gap exceeds _PASS_MAX_FRAMES should NOT be detected."""
        from analytics.events import EventDetector, _PASS_MAX_FRAMES

        det = EventDetector(fps=25.0)
        events = []
        for i in range(5):
            events.append(self._make_poss(i, team_id=1, player_id=1))
        # Gap of _PASS_MAX_FRAMES + 5 contested frames
        gap = _PASS_MAX_FRAMES + 5
        for i in range(5, 5 + gap):
            events.append(self._make_poss(i, team_id=0, player_id=None, controlled=False))
        recv_frame = 5 + gap
        events.append(self._make_poss(recv_frame, team_id=1, player_id=2))

        tracks = {"ball": [{1: {"bbox": (50, 50, 60, 60)}} for _ in range(recv_frame + 1)]}
        detected = det.detect(events, tracks)
        passes = [e for e in detected if e.event_type == "pass"]
        assert len(passes) == 0, (
            f"Pass with gap > _PASS_MAX_FRAMES ({_PASS_MAX_FRAMES}) should NOT be detected"
        )

    def test_cross_team_transfer_not_counted_as_pass(self):
        """Possession switching between teams should NOT generate a pass event."""
        from analytics.events import EventDetector

        det = EventDetector(fps=25.0)
        events = []
        # Team 1 player 1 holds the ball
        for i in range(10):
            events.append(self._make_poss(i, team_id=1, player_id=1))
        # Team 2 player 5 immediately gets the ball (tackle, not pass)
        for i in range(10, 20):
            events.append(self._make_poss(i, team_id=2, player_id=5))

        tracks = {"ball": [{1: {"bbox": (50, 50, 60, 60)}} for _ in range(20)]}
        detected = det.detect(events, tracks)
        passes = [e for e in detected if e.event_type == "pass"]
        assert len(passes) == 0, (
            "Cross-team possession transfer must NOT produce a pass event"
        )


# ─── shot detector without homography ─────────────────────────────────────


class TestShotDetectorEdgeCases:
    """Test shot detection edge cases."""

    def test_shot_no_transformer_no_shots(self):
        """Without per_frame_transformers, shot detector cannot compute cm speeds → 0 shots."""
        from analytics.events import EventDetector
        from analytics.types import PossessionEvent

        det = EventDetector(fps=25.0)
        # Build a scenario where ball moves a lot but we have no homography
        ball_frames = [{1: {"bbox": (i * 10, 100, i * 10 + 10, 110)}} for i in range(50)]
        tracks = {"ball": ball_frames}
        poss_events = [
            PossessionEvent(
                frame_idx=i, team_id=1, player_track_id=1,
                distance_to_ball=50, is_controlled=True,
            )
            for i in range(50)
        ]
        detected = det.detect(poss_events, tracks, per_frame_transformers=None)
        shots = [e for e in detected if e.event_type == "shot"]
        assert len(shots) == 0, (
            "Without homography, ball positions cannot be converted to cm → no shots"
        )


# ─── kinematics edge-case tests ────────────────────────────────────────────


class TestKinematicsEdgeCases:
    """Additional kinematics accuracy and graceful-degradation tests."""

    def _make_positions(self, coords, fps=25.0):
        from analytics.types import FramePosition
        return [
            FramePosition(
                frame_idx=i, pixel_pos=(0.0, 0.0),
                pitch_pos=(px, py), timestamp_sec=i / fps,
            )
            for i, (px, py) in enumerate(coords)
        ]

    def test_speed_known_value(self):
        """100 cm apart in 1 frame at 25 fps → exactly 25.0 m/s (no capping)."""
        from analytics.kinematics import KinematicsCalculator

        calc = KinematicsCalculator(fps=25.0)
        positions = self._make_positions([(0, 0), (100, 0)])
        _, _, dist_m, speeds_m = calc.compute_distances_and_speeds_adaptive(
            positions, max_speed_m_per_sec=1000.0  # high cap — don't clamp
        )
        assert speeds_m[0] is not None
        assert speeds_m[0] == pytest.approx(25.0, rel=1e-6), (
            f"Expected 25.0 m/s, got {speeds_m[0]}"
        )
        assert dist_m[0] == pytest.approx(1.0, rel=1e-6), (
            f"Expected 1.0 m, got {dist_m[0]}"
        )

    def test_compute_stats_no_pitch_coords_graceful(self):
        """compute_stats without pitch coords should return None distance/speed, not crash."""
        from analytics.kinematics import KinematicsCalculator
        from analytics.types import FramePosition

        calc = KinematicsCalculator(fps=25.0)
        # Positions with NO pitch_pos (pixel-only)
        positions = [
            FramePosition(frame_idx=0, pixel_pos=(0.0, 0.0), pitch_pos=None, timestamp_sec=0.0),
            FramePosition(frame_idx=1, pixel_pos=(50.0, 0.0), pitch_pos=None, timestamp_sec=0.04),
            FramePosition(frame_idx=2, pixel_pos=(100.0, 0.0), pitch_pos=None, timestamp_sec=0.08),
        ]
        stats = calc.compute_stats(positions, track_id=1, entity_type="player", use_adaptive=False)
        # Should not crash; real-world values should be None
        assert stats.total_distance_m is None
        assert stats.avg_speed_m_per_sec is None
        assert stats.max_speed_m_per_sec is None
        # Pixel values should still be computed
        assert stats.total_distance_px > 0
        assert stats.avg_speed_px > 0


# ─── possession no-contested edge case ─────────────────────────────────────


class TestPossessionEdgeCases:
    """Additional possession stats edge cases."""

    def test_aggregate_all_contested(self):
        """When all frames are contested, percentages should be 0 and not divide by zero."""
        from analytics.possession import PossessionCalculator
        from analytics.types import PossessionEvent

        events = [
            PossessionEvent(
                frame_idx=i, team_id=0, player_track_id=None,
                distance_to_ball=999, is_controlled=False,
            )
            for i in range(10)
        ]
        calc = PossessionCalculator()
        stats = calc.aggregate_stats(events)
        assert stats.team_1_percentage == 0.0
        assert stats.team_2_percentage == 0.0
        assert stats.contested_frames == 10
        assert stats.possession_changes == 0

    def test_aggregate_no_contested(self):
        """When there are zero contested frames, percentages sum to 100."""
        from analytics.possession import PossessionCalculator
        from analytics.types import PossessionEvent

        events = [
            PossessionEvent(frame_idx=i, team_id=1 + (i % 2), player_track_id=i % 2 + 1,
                            distance_to_ball=50, is_controlled=True)
            for i in range(10)
        ]
        calc = PossessionCalculator()
        stats = calc.aggregate_stats(events)
        assert stats.contested_frames == 0
        assert stats.team_1_percentage + stats.team_2_percentage == pytest.approx(100.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

