"""
Object detection and tracking module.
Uses YOLOv8 for detection and ByteTrack for tracking.
Matches the original repo's tracker.py structure.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from ultralytics import YOLO
    import supervision as sv
except ImportError:
    raise ImportError("Please install ultralytics and supervision: pip install ultralytics supervision")


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Track:
    """Single track with ID."""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    team_id: Optional[int] = None


class FootballTracker:
    """
    Football object tracker using YOLOv8 + ByteTrack.
    
    Detects and tracks:
    - Players (class 0)
    - Goalkeepers (class 1)
    - Referees (class 2)
    - Ball (class 3)
    """
    
    # Class mapping for football detection
    CLASS_NAMES = {
        0: "player",
        1: "goalkeeper", 
        2: "referee",
        3: "ball"
    }
    
    def __init__(
        self,
        player_model_path: str,
        ball_model_path: Optional[str] = None,
        device: str = "auto",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        """
        Initialize the tracker.
        
        Args:
            player_model_path: Path to YOLOv8 model for player detection
            ball_model_path: Optional path to separate ball detection model
            device: Device to run on (auto, cuda, mps, cpu)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            track_thresh: ByteTrack tracking threshold
            track_buffer: ByteTrack buffer size
            match_thresh: ByteTrack matching threshold
        """
        self.device = self._resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Load player detection model
        print(f"Loading player detection model from {player_model_path}...")
        self.player_model = YOLO(player_model_path)
        
        # Load ball detection model (optional - uses player model if not provided)
        self.ball_model = None
        if ball_model_path and Path(ball_model_path).exists():
            print(f"Loading ball detection model from {ball_model_path}...")
            self.ball_model = YOLO(ball_model_path)
        
        # Initialize ByteTrack trackers for each object type
        self.player_tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=25
        )
        
        self.ball_tracker = sv.ByteTrack(
            track_activation_threshold=0.1,  # Lower threshold for ball
            lost_track_buffer=60,  # Longer buffer for ball occlusions
            minimum_matching_threshold=0.5,
            frame_rate=25
        )
    
    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device != "auto":
            return device
        
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def detect_frame(self, frame: np.ndarray) -> Dict[str, List[Detection]]:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Dictionary with detections by class name
        """
        detections = {
            "players": [],
            "goalkeepers": [],
            "referees": [],
            "ball": []
        }
        
        # Run player model
        results = self.player_model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Parse detections
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = tuple(box.xyxy[0].cpu().numpy())
            
            class_name = self.CLASS_NAMES.get(class_id, "unknown")
            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name
            )
            
            if class_name == "player":
                detections["players"].append(detection)
            elif class_name == "goalkeeper":
                detections["goalkeepers"].append(detection)
            elif class_name == "referee":
                detections["referees"].append(detection)
            elif class_name == "ball":
                detections["ball"].append(detection)
        
        # Run separate ball model if available
        if self.ball_model is not None:
            ball_results = self.ball_model.predict(
                frame,
                conf=0.1,  # Lower threshold for ball
                device=self.device,
                verbose=False
            )[0]
            
            for box in ball_results.boxes:
                confidence = float(box.conf[0])
                bbox = tuple(box.xyxy[0].cpu().numpy())
                
                detection = Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=3,
                    class_name="ball"
                )
                detections["ball"].append(detection)
        
        return detections
    
    def track_frame(
        self,
        frame: np.ndarray,
        detections: Dict[str, List[Detection]]
    ) -> Dict[str, Dict[int, Track]]:
        """
        Run tracking on detections for a single frame.
        
        Args:
            frame: BGR image as numpy array
            detections: Detections from detect_frame()
            
        Returns:
            Dictionary with tracks by class name, keyed by track ID
        """
        tracks = {
            "players": {},
            "goalkeepers": {},
            "referees": {},
            "ball": {}
        }
        
        # Convert player detections to supervision format
        player_dets = detections["players"] + detections["goalkeepers"] + detections["referees"]
        if player_dets:
            xyxy = np.array([d.bbox for d in player_dets])
            confidence = np.array([d.confidence for d in player_dets])
            class_id = np.array([d.class_id for d in player_dets])
            
            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )
            
            # Run ByteTrack
            tracked = self.player_tracker.update_with_detections(sv_detections)
            
            # Parse tracked objects
            for i in range(len(tracked)):
                track_id = int(tracked.tracker_id[i])
                bbox = tuple(tracked.xyxy[i])
                conf = float(tracked.confidence[i])
                cls_id = int(tracked.class_id[i])
                cls_name = self.CLASS_NAMES.get(cls_id, "player")
                
                track = Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                )
                
                if cls_name == "goalkeeper":
                    tracks["goalkeepers"][track_id] = track
                elif cls_name == "referee":
                    tracks["referees"][track_id] = track
                else:
                    tracks["players"][track_id] = track
        
        # Track ball separately
        if detections["ball"]:
            xyxy = np.array([d.bbox for d in detections["ball"]])
            confidence = np.array([d.confidence for d in detections["ball"]])
            class_id = np.array([3] * len(detections["ball"]))
            
            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )
            
            tracked = self.ball_tracker.update_with_detections(sv_detections)
            
            for i in range(len(tracked)):
                track_id = int(tracked.tracker_id[i])
                bbox = tuple(tracked.xyxy[i])
                conf = float(tracked.confidence[i])
                
                track = Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_id=3,
                    class_name="ball"
                )
                tracks["ball"][track_id] = track
        
        return tracks
    
    def process_video(
        self,
        frames: List[np.ndarray],
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Process all frames in a video.
        
        Args:
            frames: List of BGR frames
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (all_detections, all_tracks)
        """
        all_detections = []
        all_tracks = []
        
        for i, frame in enumerate(frames):
            # Detect
            detections = self.detect_frame(frame)
            all_detections.append(detections)
            
            # Track
            tracks = self.track_frame(frame, detections)
            all_tracks.append(tracks)
            
            # Progress callback
            if progress_callback and i % 10 == 0:
                progress_callback(i / len(frames), f"Processing frame {i}/{len(frames)}")
        
        return all_detections, all_tracks
    
    def interpolate_ball(
        self,
        tracks: List[Dict[str, Dict[int, Track]]],
        max_gap: int = 30
    ) -> List[Dict[str, Dict[int, Track]]]:
        """
        Interpolate missing ball positions.
        
        Args:
            tracks: List of frame tracks
            max_gap: Maximum gap to interpolate
            
        Returns:
            Tracks with interpolated ball positions
        """
        # Find ball track ID (should be consistent)
        ball_positions = []
        for frame_tracks in tracks:
            if frame_tracks["ball"]:
                ball_track = list(frame_tracks["ball"].values())[0]
                cx = (ball_track.bbox[0] + ball_track.bbox[2]) / 2
                cy = (ball_track.bbox[1] + ball_track.bbox[3]) / 2
                ball_positions.append((cx, cy, ball_track))
            else:
                ball_positions.append(None)
        
        # Find gaps and interpolate
        interpolated_tracks = [dict(t) for t in tracks]  # Copy
        
        i = 0
        while i < len(ball_positions):
            if ball_positions[i] is None:
                # Find start and end of gap
                gap_start = i
                while i < len(ball_positions) and ball_positions[i] is None:
                    i += 1
                gap_end = i
                gap_length = gap_end - gap_start
                
                # Interpolate if gap is small enough and we have endpoints
                if gap_length <= max_gap and gap_start > 0 and gap_end < len(ball_positions):
                    start_pos = ball_positions[gap_start - 1]
                    end_pos = ball_positions[gap_end]
                    
                    if start_pos and end_pos:
                        for j in range(gap_start, gap_end):
                            t = (j - gap_start + 1) / (gap_length + 1)
                            cx = start_pos[0] + t * (end_pos[0] - start_pos[0])
                            cy = start_pos[1] + t * (end_pos[1] - start_pos[1])
                            
                            # Create interpolated track
                            ref_track = start_pos[2]
                            w = ref_track.bbox[2] - ref_track.bbox[0]
                            h = ref_track.bbox[3] - ref_track.bbox[1]
                            
                            interp_track = Track(
                                track_id=ref_track.track_id,
                                bbox=(cx - w/2, cy - h/2, cx + w/2, cy + h/2),
                                confidence=0.5,  # Lower confidence for interpolated
                                class_id=3,
                                class_name="ball"
                            )
                            interpolated_tracks[j]["ball"] = {interp_track.track_id: interp_track}
            else:
                i += 1
        
        return interpolated_tracks
