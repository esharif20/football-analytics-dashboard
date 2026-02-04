"""
Analytics computation module.
Calculates possession, kinematics, and detects events.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PossessionStats:
    """Ball possession statistics."""
    team_1_frames: int = 0
    team_2_frames: int = 0
    contested_frames: int = 0
    no_ball_frames: int = 0
    
    @property
    def team_1_percentage(self) -> float:
        total = self.team_1_frames + self.team_2_frames
        return (self.team_1_frames / total * 100) if total > 0 else 0
    
    @property
    def team_2_percentage(self) -> float:
        total = self.team_1_frames + self.team_2_frames
        return (self.team_2_frames / total * 100) if total > 0 else 0


@dataclass
class PlayerKinematics:
    """Player movement statistics."""
    track_id: int
    team_id: Optional[int]
    positions: List[Tuple[float, float]] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    
    @property
    def total_distance(self) -> float:
        if len(self.positions) < 2:
            return 0
        total = 0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i-1][0]
            dy = self.positions[i][1] - self.positions[i-1][1]
            total += np.sqrt(dx*dx + dy*dy)
        return total
    
    @property
    def avg_speed(self) -> float:
        return np.mean(self.speeds) if self.speeds else 0
    
    @property
    def max_speed(self) -> float:
        return max(self.speeds) if self.speeds else 0


@dataclass
class BallStats:
    """Ball statistics."""
    positions: List[Tuple[float, float]] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    detected_frames: int = 0
    total_frames: int = 0
    
    @property
    def detection_rate(self) -> float:
        return self.detected_frames / self.total_frames if self.total_frames > 0 else 0
    
    @property
    def total_distance(self) -> float:
        if len(self.positions) < 2:
            return 0
        total = 0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i-1][0]
            dy = self.positions[i][1] - self.positions[i-1][1]
            total += np.sqrt(dx*dx + dy*dy)
        return total
    
    @property
    def avg_speed(self) -> float:
        return np.mean(self.speeds) if self.speeds else 0


@dataclass
class Event:
    """Detected event."""
    type: str  # possession_change, pass, shot, high_speed_ball, etc.
    frame_idx: int
    time: float  # seconds
    team_id: Optional[int] = None
    player_id: Optional[int] = None
    position: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnalyticsEngine:
    """
    Computes analytics from tracking data.
    """
    
    def __init__(
        self,
        fps: int = 25,
        possession_distance_threshold: float = 50.0,
        speed_smoothing_window: int = 5,
        high_speed_threshold: float = 25.0  # km/h
    ):
        """
        Initialize the analytics engine.
        
        Args:
            fps: Video frame rate
            possession_distance_threshold: Max distance to ball for possession (pixels)
            speed_smoothing_window: Window for speed smoothing
            high_speed_threshold: Threshold for high-speed ball events (km/h)
        """
        self.fps = fps
        self.possession_distance_threshold = possession_distance_threshold
        self.speed_smoothing_window = speed_smoothing_window
        self.high_speed_threshold = high_speed_threshold
    
    def compute_possession(
        self,
        tracks: List[Dict],
        team_assignments: Dict[int, int]
    ) -> PossessionStats:
        """
        Compute ball possession statistics.
        
        Args:
            tracks: List of frame tracks
            team_assignments: Mapping of player track_id to team_id
            
        Returns:
            PossessionStats
        """
        stats = PossessionStats()
        
        for frame_tracks in tracks:
            # Get ball position
            ball_tracks = frame_tracks.get("ball", {})
            if not ball_tracks:
                stats.no_ball_frames += 1
                continue
            
            ball = list(ball_tracks.values())[0]
            ball_cx = (ball.bbox[0] + ball.bbox[2]) / 2
            ball_cy = (ball.bbox[1] + ball.bbox[3]) / 2
            
            # Find closest player
            min_dist = float('inf')
            closest_team = None
            
            for player_id, player in frame_tracks.get("players", {}).items():
                player_cx = (player.bbox[0] + player.bbox[2]) / 2
                player_cy = (player.bbox[1] + player.bbox[3]) / 2
                
                dist = np.sqrt((ball_cx - player_cx)**2 + (ball_cy - player_cy)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_team = team_assignments.get(player_id)
            
            # Assign possession
            if min_dist <= self.possession_distance_threshold and closest_team is not None:
                if closest_team == 0:
                    stats.team_1_frames += 1
                else:
                    stats.team_2_frames += 1
            else:
                stats.contested_frames += 1
        
        return stats
    
    def compute_player_kinematics(
        self,
        tracks: List[Dict],
        team_assignments: Dict[int, int],
        transformers: Optional[List] = None
    ) -> Dict[int, PlayerKinematics]:
        """
        Compute player movement statistics.
        
        Args:
            tracks: List of frame tracks
            team_assignments: Mapping of player track_id to team_id
            transformers: Optional list of ViewTransformers for pitch coordinates
            
        Returns:
            Dictionary mapping track_id to PlayerKinematics
        """
        kinematics = {}
        
        for frame_idx, frame_tracks in enumerate(tracks):
            transformer = transformers[frame_idx] if transformers else None
            
            for player_id, player in frame_tracks.get("players", {}).items():
                if player_id not in kinematics:
                    kinematics[player_id] = PlayerKinematics(
                        track_id=player_id,
                        team_id=team_assignments.get(player_id)
                    )
                
                # Get position
                cx = (player.bbox[0] + player.bbox[2]) / 2
                cy = (player.bbox[1] + player.bbox[3]) / 2
                
                # Transform to pitch coordinates if available
                if transformer and transformer.is_valid:
                    pitch_pos = transformer.transform_point(cx, cy)
                    if pitch_pos:
                        cx, cy = pitch_pos
                
                kinematics[player_id].positions.append((cx, cy))
                
                # Calculate speed
                if len(kinematics[player_id].positions) >= 2:
                    prev = kinematics[player_id].positions[-2]
                    dx = cx - prev[0]
                    dy = cy - prev[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    speed = dist * self.fps  # units per second
                    kinematics[player_id].speeds.append(speed)
        
        return kinematics
    
    def compute_ball_stats(
        self,
        tracks: List[Dict],
        transformers: Optional[List] = None
    ) -> BallStats:
        """
        Compute ball statistics.
        
        Args:
            tracks: List of frame tracks
            transformers: Optional list of ViewTransformers
            
        Returns:
            BallStats
        """
        stats = BallStats(total_frames=len(tracks))
        
        for frame_idx, frame_tracks in enumerate(tracks):
            ball_tracks = frame_tracks.get("ball", {})
            
            if not ball_tracks:
                continue
            
            stats.detected_frames += 1
            ball = list(ball_tracks.values())[0]
            
            cx = (ball.bbox[0] + ball.bbox[2]) / 2
            cy = (ball.bbox[1] + ball.bbox[3]) / 2
            
            # Transform if available
            transformer = transformers[frame_idx] if transformers else None
            if transformer and transformer.is_valid:
                pitch_pos = transformer.transform_point(cx, cy)
                if pitch_pos:
                    cx, cy = pitch_pos
            
            stats.positions.append((cx, cy))
            
            # Calculate speed
            if len(stats.positions) >= 2:
                prev = stats.positions[-2]
                dx = cx - prev[0]
                dy = cy - prev[1]
                dist = np.sqrt(dx*dx + dy*dy)
                speed = dist * self.fps
                stats.speeds.append(speed)
        
        return stats
    
    def detect_events(
        self,
        tracks: List[Dict],
        team_assignments: Dict[int, int],
        possession_stats: PossessionStats,
        ball_stats: BallStats
    ) -> List[Event]:
        """
        Detect events from tracking data.
        
        Args:
            tracks: List of frame tracks
            team_assignments: Player team assignments
            possession_stats: Computed possession stats
            ball_stats: Computed ball stats
            
        Returns:
            List of detected events
        """
        events = []
        
        # Track possession changes
        last_possession_team = None
        
        for frame_idx, frame_tracks in enumerate(tracks):
            time = frame_idx / self.fps
            
            # Check ball possession
            ball_tracks = frame_tracks.get("ball", {})
            if ball_tracks:
                ball = list(ball_tracks.values())[0]
                ball_cx = (ball.bbox[0] + ball.bbox[2]) / 2
                ball_cy = (ball.bbox[1] + ball.bbox[3]) / 2
                
                # Find possessing player
                min_dist = float('inf')
                possessing_team = None
                possessing_player = None
                
                for player_id, player in frame_tracks.get("players", {}).items():
                    player_cx = (player.bbox[0] + player.bbox[2]) / 2
                    player_cy = (player.bbox[1] + player.bbox[3]) / 2
                    dist = np.sqrt((ball_cx - player_cx)**2 + (ball_cy - player_cy)**2)
                    
                    if dist < min_dist and dist < self.possession_distance_threshold:
                        min_dist = dist
                        possessing_team = team_assignments.get(player_id)
                        possessing_player = player_id
                
                # Detect possession change
                if possessing_team is not None and possessing_team != last_possession_team:
                    if last_possession_team is not None:
                        events.append(Event(
                            type="possession_change",
                            frame_idx=frame_idx,
                            time=time,
                            team_id=possessing_team,
                            player_id=possessing_player,
                            position=(ball_cx, ball_cy),
                            metadata={"from_team": last_possession_team}
                        ))
                    last_possession_team = possessing_team
        
        # Detect high-speed ball movements
        if ball_stats.speeds:
            for i, speed in enumerate(ball_stats.speeds):
                # Convert to km/h (assuming meters)
                speed_kmh = speed * 3.6
                if speed_kmh > self.high_speed_threshold:
                    events.append(Event(
                        type="high_speed_ball",
                        frame_idx=i + 1,
                        time=(i + 1) / self.fps,
                        position=ball_stats.positions[i + 1] if i + 1 < len(ball_stats.positions) else None,
                        metadata={"speed_kmh": speed_kmh}
                    ))
        
        # Sort events by time
        events.sort(key=lambda e: e.time)
        
        return events
    
    def to_dict(
        self,
        possession: PossessionStats,
        player_kinematics: Dict[int, PlayerKinematics],
        ball_stats: BallStats,
        events: List[Event]
    ) -> Dict[str, Any]:
        """
        Convert analytics to dictionary format.
        """
        return {
            "possession": {
                "team_1_percentage": possession.team_1_percentage,
                "team_2_percentage": possession.team_2_percentage,
                "team_1_frames": possession.team_1_frames,
                "team_2_frames": possession.team_2_frames,
                "contested_frames": possession.contested_frames,
                "no_ball_frames": possession.no_ball_frames
            },
            "player_stats": {
                str(k): {
                    "track_id": v.track_id,
                    "team_id": v.team_id,
                    "total_distance": v.total_distance,
                    "avg_speed": v.avg_speed,
                    "max_speed": v.max_speed
                }
                for k, v in player_kinematics.items()
            },
            "ball_stats": {
                "total_distance": ball_stats.total_distance,
                "avg_speed": ball_stats.avg_speed,
                "detection_rate": ball_stats.detection_rate,
                "detected_frames": ball_stats.detected_frames,
                "total_frames": ball_stats.total_frames
            },
            "events": [
                {
                    "type": e.type,
                    "frame_idx": e.frame_idx,
                    "time": e.time,
                    "team_id": e.team_id,
                    "player_id": e.player_id,
                    "position": e.position,
                    "metadata": e.metadata
                }
                for e in events
            ]
        }
