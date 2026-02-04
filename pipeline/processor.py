"""
Football Processor - Core CV Pipeline

Wraps YOLOv8, ByteTrack, SigLIP, and homography into a unified processor.
Based on the Spatio-Temporal-GNN-Football-Analysis pipeline.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict

import supervision as sv
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for the football processor."""
    mode: str = "all"
    device: str = "cpu"
    use_custom_models: bool = True
    
    # Detection
    det_model_path: str = "yolov8x.pt"
    ball_model_path: Optional[str] = None
    pitch_model_path: Optional[str] = None
    det_conf: float = 0.25
    det_imgsz: int = 1280
    det_batch_size: int = 16
    
    # Ball tracking
    ball_conf: float = 0.15
    ball_imgsz: int = 640
    ball_use_slicer: bool = True
    ball_slice_wh: Tuple[int, int] = (640, 640)
    ball_overlap_wh: Tuple[int, int] = (128, 128)
    
    # Team classification
    team_stride: int = 30
    team_batch_size: int = 32
    team_max_crops: int = 2000
    
    # Pitch detection
    pitch_conf: float = 0.3
    pitch_imgsz: int = 1280
    roboflow_api_key: Optional[str] = None
    
    # Output
    output_dir: str = "/tmp/football_analysis"
    video_fps: float = 25.0


# ============================================================================
# Pitch Configuration
# ============================================================================

@dataclass
class SoccerPitchConfiguration:
    """Standard soccer pitch dimensions in centimeters."""
    width: int = 10500  # 105m
    length: int = 6800  # 68m
    
    penalty_box_width: int = 4032
    penalty_box_length: int = 1650
    goal_box_width: int = 1832
    goal_box_length: int = 550
    
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100
    
    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """Return 32 keypoint vertices for the pitch."""
        w, l = self.width, self.length
        pbw, pbl = self.penalty_box_width, self.penalty_box_length
        gbw, gbl = self.goal_box_width, self.goal_box_length
        
        # Standard 32 keypoints matching common pitch detection models
        return [
            # Corners
            (0, 0), (w/2, 0), (w, 0),
            (0, l/2), (w/2, l/2), (w, l/2),
            (0, l), (w/2, l), (w, l),
            # Left penalty box
            (0, (l-pbw)/2), (pbl, (l-pbw)/2),
            (0, (l+pbw)/2), (pbl, (l+pbw)/2),
            # Right penalty box
            (w, (l-pbw)/2), (w-pbl, (l-pbw)/2),
            (w, (l+pbw)/2), (w-pbl, (l+pbw)/2),
            # Left goal box
            (0, (l-gbw)/2), (gbl, (l-gbw)/2),
            (0, (l+gbw)/2), (gbl, (l+gbw)/2),
            # Right goal box
            (w, (l-gbw)/2), (w-gbl, (l-gbw)/2),
            (w, (l+gbw)/2), (w-gbl, (l+gbw)/2),
            # Centre circle (approximated with 8 points)
            (w/2, l/2 - self.centre_circle_radius),
            (w/2 + self.centre_circle_radius, l/2),
            (w/2, l/2 + self.centre_circle_radius),
            (w/2 - self.centre_circle_radius, l/2),
            # Penalty spots
            (self.penalty_spot_distance, l/2),
            (w - self.penalty_spot_distance, l/2),
        ]


# ============================================================================
# Football Processor
# ============================================================================

class FootballProcessor:
    """Main processor for football video analysis."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._init_models()
        self.pitch_config = SoccerPitchConfiguration()
        
    def _init_models(self):
        """Initialize detection models."""
        from ultralytics import YOLO
        
        # Main detection model
        self.det_model = YOLO(self.config.det_model_path)
        self._select_device(self.det_model)
        
        # Ball detection model (optional)
        self.ball_model = None
        if self.config.ball_model_path and Path(self.config.ball_model_path).exists():
            self.ball_model = YOLO(self.config.ball_model_path)
            # Keep ball model on CPU for slicer efficiency
            
        # Pitch detection model (optional)
        self.pitch_model = None
        if self.config.pitch_model_path and Path(self.config.pitch_model_path).exists():
            self.pitch_model = YOLO(self.config.pitch_model_path)
            self._select_device(self.pitch_model)
            
        # ByteTrack tracker
        self.tracker = sv.ByteTrack()
        
        # Team classifier (lazy loaded)
        self._team_classifier = None
        
        print(f"Models initialized on device: {self.config.device}")
        
    def _select_device(self, model):
        """Select compute device for model."""
        try:
            import torch
            if self.config.device == "cuda" and torch.cuda.is_available():
                model.to("cuda")
            elif self.config.device == "mps" and torch.backends.mps.is_available():
                model.to("mps")
        except Exception:
            pass
    
    # ========================================================================
    # Video Loading
    # ========================================================================
    
    def load_video(self, video_path: str) -> List[np.ndarray]:
        """Load video frames into memory."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.config.video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        print(f"Loaded {len(frames)} frames at {self.config.video_fps} fps")
        return frames
    
    # ========================================================================
    # Detection
    # ========================================================================
    
    def detect(
        self, 
        frames: List[np.ndarray],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[sv.Detections]:
        """Run object detection on all frames."""
        detections = []
        batch_size = self.config.det_batch_size
        total = len(frames)
        
        for i in tqdm(range(0, total, batch_size), desc="Detecting"):
            batch = frames[i:i + batch_size]
            results = self.det_model.predict(
                batch,
                conf=self.config.det_conf,
                imgsz=self.config.det_imgsz,
                verbose=False,
            )
            
            for result in results:
                det = sv.Detections.from_ultralytics(result)
                detections.append(det)
            
            if progress_callback:
                progress_callback((i + len(batch)) / total)
        
        return detections
    
    def detect_ball(self, frames: List[np.ndarray]) -> List[sv.Detections]:
        """Run ball-specific detection using SAHI slicer."""
        if self.ball_model is None:
            return [sv.Detections.empty() for _ in frames]
        
        detections = []
        
        # Create slicer callback
        def slicer_callback(image_slice: np.ndarray) -> sv.Detections:
            result = self.ball_model.predict(
                image_slice,
                conf=self.config.ball_conf,
                imgsz=self.config.ball_imgsz,
                verbose=False,
            )[0]
            return sv.Detections.from_ultralytics(result)
        
        slicer = sv.InferenceSlicer(
            callback=slicer_callback,
            slice_wh=self.config.ball_slice_wh,
            overlap_wh=self.config.ball_overlap_wh,
            overlap_filter=sv.OverlapFilter.NONE,
        )
        
        for frame in tqdm(frames, desc="Ball detection"):
            det = slicer(frame)
            detections.append(det)
        
        return detections
    
    # ========================================================================
    # Tracking
    # ========================================================================
    
    def track(
        self, 
        frames: List[np.ndarray], 
        detections: List[sv.Detections]
    ) -> Dict[str, List[Dict]]:
        """Track objects across frames using ByteTrack."""
        tracks = {
            "players": [],
            "goalkeepers": [],
            "referees": [],
            "ball": [],
        }
        
        # Class ID mapping (adjust based on your model)
        class_names = self.det_model.names
        player_id = self._get_class_id(class_names, ["player", "person"])
        goalkeeper_id = self._get_class_id(class_names, ["goalkeeper"])
        referee_id = self._get_class_id(class_names, ["referee"])
        ball_id = self._get_class_id(class_names, ["ball", "football"])
        
        self.tracker.reset()
        
        for frame_idx, det in enumerate(tqdm(detections, desc="Tracking")):
            frame_tracks = {
                "players": {},
                "goalkeepers": {},
                "referees": {},
                "ball": {},
            }
            
            if len(det) == 0:
                for key in tracks:
                    tracks[key].append(frame_tracks[key])
                continue
            
            # Filter and track by class
            # Players
            if player_id is not None:
                player_mask = det.class_id == player_id
                player_det = det[player_mask]
                if len(player_det) > 0:
                    tracked = self.tracker.update_with_detections(player_det)
                    for i, track_id in enumerate(tracked.tracker_id):
                        if track_id is not None:
                            frame_tracks["players"][int(track_id)] = {
                                "bbox": tracked.xyxy[i].tolist(),
                                "confidence": float(tracked.confidence[i]) if tracked.confidence is not None else 1.0,
                            }
            
            # Goalkeepers
            if goalkeeper_id is not None:
                gk_mask = det.class_id == goalkeeper_id
                gk_det = det[gk_mask]
                for i in range(len(gk_det)):
                    frame_tracks["goalkeepers"][i] = {
                        "bbox": gk_det.xyxy[i].tolist(),
                        "confidence": float(gk_det.confidence[i]) if gk_det.confidence is not None else 1.0,
                    }
            
            # Referees
            if referee_id is not None:
                ref_mask = det.class_id == referee_id
                ref_det = det[ref_mask]
                for i in range(len(ref_det)):
                    frame_tracks["referees"][i] = {
                        "bbox": ref_det.xyxy[i].tolist(),
                        "confidence": float(ref_det.confidence[i]) if ref_det.confidence is not None else 1.0,
                    }
            
            # Ball
            if ball_id is not None:
                ball_mask = det.class_id == ball_id
                ball_det = det[ball_mask]
                if len(ball_det) > 0:
                    # Take highest confidence ball
                    best_idx = np.argmax(ball_det.confidence) if ball_det.confidence is not None else 0
                    frame_tracks["ball"][1] = {
                        "bbox": ball_det.xyxy[best_idx].tolist(),
                        "confidence": float(ball_det.confidence[best_idx]) if ball_det.confidence is not None else 1.0,
                    }
            
            for key in tracks:
                tracks[key].append(frame_tracks[key])
        
        # Interpolate ball tracks
        tracks["ball"] = self._interpolate_ball(tracks["ball"])
        
        return tracks
    
    def _get_class_id(self, names: Dict, candidates: List[str]) -> Optional[int]:
        """Get class ID from model names."""
        candidates_lower = {c.lower() for c in candidates}
        for idx, name in names.items():
            if str(name).lower() in candidates_lower:
                return int(idx)
        return None
    
    def _interpolate_ball(self, ball_tracks: List[Dict]) -> List[Dict]:
        """Interpolate missing ball positions."""
        # Find frames with ball detections
        positions = []
        for i, frame in enumerate(ball_tracks):
            if 1 in frame:
                bbox = frame[1]["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                positions.append((i, cx, cy, bbox))
        
        if len(positions) < 2:
            return ball_tracks
        
        # Interpolate between detections
        result = [{} for _ in ball_tracks]
        
        for i in range(len(positions) - 1):
            start_idx, sx, sy, s_bbox = positions[i]
            end_idx, ex, ey, e_bbox = positions[i + 1]
            
            # Copy original detection
            result[start_idx][1] = ball_tracks[start_idx][1].copy()
            
            # Interpolate if gap is reasonable (< 30 frames)
            gap = end_idx - start_idx
            if gap > 1 and gap < 30:
                for j in range(1, gap):
                    t = j / gap
                    cx = sx + t * (ex - sx)
                    cy = sy + t * (ey - sy)
                    
                    # Interpolate bbox size
                    w = (s_bbox[2] - s_bbox[0]) + t * ((e_bbox[2] - e_bbox[0]) - (s_bbox[2] - s_bbox[0]))
                    h = (s_bbox[3] - s_bbox[1]) + t * ((e_bbox[3] - e_bbox[1]) - (s_bbox[3] - s_bbox[1]))
                    
                    result[start_idx + j][1] = {
                        "bbox": [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                        "confidence": 0.5,  # Interpolated
                        "interpolated": True,
                    }
        
        # Copy last detection
        if positions:
            last_idx = positions[-1][0]
            result[last_idx][1] = ball_tracks[last_idx][1].copy()
        
        return result
    
    # ========================================================================
    # Team Classification
    # ========================================================================
    
    def classify_teams(
        self, 
        frames: List[np.ndarray], 
        tracks: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """Classify players into teams using SigLIP + UMAP + KMeans."""
        try:
            import torch
            import umap
            from sklearn.cluster import KMeans
            from transformers import AutoProcessor, SiglipVisionModel
        except ImportError:
            print("Team classification dependencies not available")
            return tracks
        
        print("Extracting player crops...")
        crops_by_id = self._collect_crops(frames, tracks)
        
        if not crops_by_id:
            print("No crops collected for team classification")
            return tracks
        
        # Flatten crops
        all_crops = []
        owners = []
        for pid, crops in crops_by_id.items():
            for crop in crops:
                all_crops.append(crop)
                owners.append(pid)
        
        print(f"Processing {len(all_crops)} crops for team classification...")
        
        # Extract features using SigLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(device)
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        
        # Convert crops to PIL
        from PIL import Image
        pil_crops = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in all_crops]
        
        # Batch process
        embeddings = []
        batch_size = self.config.team_batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(pil_crops), batch_size), desc="Embedding"):
                batch = pil_crops[i:i + batch_size]
                inputs = processor(images=batch, return_tensors="pt").to(device)
                outputs = model(**inputs)
                emb = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                embeddings.append(emb)
        
        embeddings = np.concatenate(embeddings)
        
        # UMAP + KMeans
        print("Running UMAP dimensionality reduction...")
        reducer = umap.UMAP(n_components=3)
        projections = reducer.fit_transform(embeddings)
        
        print("Clustering teams...")
        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(projections)
        
        # Assign teams by majority vote
        votes = defaultdict(lambda: np.zeros(2, dtype=int))
        for pid, label in zip(owners, labels):
            votes[pid][label] += 1
        
        track_team = {pid: int(np.argmax(counts)) for pid, counts in votes.items()}
        
        # Compute team colors
        team_colors = self._compute_team_colors(crops_by_id, track_team)
        
        # Apply to tracks
        for frame_idx in range(len(frames)):
            for pid, info in tracks["players"][frame_idx].items():
                team_id = track_team.get(pid)
                if team_id is not None:
                    info["team_id"] = team_id
                    if team_id in team_colors:
                        info["team_color"] = team_colors[team_id]
        
        # Assign goalkeepers by proximity
        self._assign_goalkeeper_teams(frames, tracks, team_colors)
        
        self.team_colors = team_colors
        return tracks
    
    def _collect_crops(
        self, 
        frames: List[np.ndarray], 
        tracks: Dict[str, List[Dict]]
    ) -> Dict[int, List[np.ndarray]]:
        """Collect player crops for team classification."""
        crops_by_id = {}
        stride = self.config.team_stride
        total_crops = 0
        
        for frame_idx in range(0, len(frames), stride):
            frame = frames[frame_idx]
            for pid, info in tracks["players"][frame_idx].items():
                bbox = info.get("bbox")
                if bbox is None:
                    continue
                
                x1, y1, x2, y2 = [int(v) for v in bbox]
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] < 10 or crop.shape[1] < 6:
                    continue
                
                crops_by_id.setdefault(pid, []).append(crop)
                total_crops += 1
                
                if self.config.team_max_crops > 0 and total_crops >= self.config.team_max_crops:
                    return crops_by_id
        
        return crops_by_id
    
    def _compute_team_colors(
        self, 
        crops_by_id: Dict[int, List[np.ndarray]], 
        track_team: Dict[int, int]
    ) -> Dict[int, Tuple[int, int, int]]:
        """Compute representative colors for each team."""
        colors_by_team = {0: [], 1: []}
        
        for pid, crops in crops_by_id.items():
            team_id = track_team.get(pid)
            if team_id is None:
                continue
            
            for crop in crops:
                # Sample from torso region
                h, w = crop.shape[:2]
                y1, y2 = int(h * 0.2), int(h * 0.6)
                x1, x2 = int(w * 0.2), int(w * 0.8)
                region = crop[y1:y2, x1:x2]
                if region.size == 0:
                    region = crop
                mean_color = region.reshape(-1, 3).mean(axis=0)
                colors_by_team[team_id].append(mean_color)
        
        team_colors = {}
        for team_id, samples in colors_by_team.items():
            if samples:
                median = np.median(np.vstack(samples), axis=0)
                team_colors[team_id] = tuple(int(c) for c in median)
        
        return team_colors
    
    def _assign_goalkeeper_teams(
        self, 
        frames: List[np.ndarray], 
        tracks: Dict[str, List[Dict]],
        team_colors: Dict[int, Tuple[int, int, int]]
    ):
        """Assign goalkeepers to teams by proximity to team centroids."""
        for frame_idx in range(len(frames)):
            players = tracks["players"][frame_idx]
            goalkeepers = tracks["goalkeepers"][frame_idx]
            
            if not goalkeepers:
                continue
            
            # Compute team centroids
            team_positions = {0: [], 1: []}
            for info in players.values():
                team_id = info.get("team_id")
                bbox = info.get("bbox")
                if team_id is not None and bbox is not None:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    team_positions[team_id].append((cx, cy))
            
            centroids = {}
            for team_id, positions in team_positions.items():
                if positions:
                    centroids[team_id] = np.mean(positions, axis=0)
            
            if len(centroids) < 2:
                continue
            
            # Assign each goalkeeper
            for gk_id, gk_info in goalkeepers.items():
                bbox = gk_info.get("bbox")
                if bbox is None:
                    continue
                
                gk_pos = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                
                dist_0 = np.linalg.norm(gk_pos - centroids.get(0, gk_pos))
                dist_1 = np.linalg.norm(gk_pos - centroids.get(1, gk_pos))
                
                team_id = 0 if dist_0 < dist_1 else 1
                gk_info["team_id"] = team_id
                if team_id in team_colors:
                    gk_info["team_color"] = team_colors[team_id]
    
    # ========================================================================
    # Pitch Detection
    # ========================================================================
    
    def detect_pitch(self, frames: List[np.ndarray]) -> List[Dict]:
        """Detect pitch keypoints in all frames."""
        pitch_data = []
        pitch_vertices = np.array(self.pitch_config.vertices, dtype=np.float32)
        
        for frame in tqdm(frames, desc="Pitch detection"):
            if self.pitch_model is not None:
                # Use local model
                result = self.pitch_model.predict(
                    frame,
                    conf=self.config.pitch_conf,
                    imgsz=self.config.pitch_imgsz,
                    verbose=False,
                )[0]
                
                keypoints = result.keypoints
                frame_kp = np.empty((0, 2), dtype=np.float32)
                pitch_kp = np.empty((0, 2), dtype=np.float32)
                conf_mask = np.zeros(len(pitch_vertices), dtype=bool)
                
                if keypoints is not None and keypoints.xy is not None:
                    xy = keypoints.xy[0].cpu().numpy() if hasattr(keypoints.xy[0], 'cpu') else keypoints.xy[0]
                    conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(len(xy))
                    
                    if len(xy) == len(pitch_vertices):
                        conf_mask = conf > 0.5
                        if conf_mask.any():
                            frame_kp = xy[conf_mask].astype(np.float32)
                            pitch_kp = pitch_vertices[conf_mask]
                
                pitch_data.append({
                    "frame_keypoints": frame_kp,
                    "pitch_keypoints": pitch_kp,
                    "conf_mask": conf_mask,
                })
            else:
                # Use Roboflow API as fallback
                pitch_data.append(self._detect_pitch_roboflow(frame, pitch_vertices))
        
        return pitch_data
    
    def _detect_pitch_roboflow(self, frame: np.ndarray, pitch_vertices: np.ndarray) -> Dict:
        """Detect pitch using Roboflow API."""
        api_key = self.config.roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")
        
        if not api_key:
            return {
                "frame_keypoints": np.empty((0, 2), dtype=np.float32),
                "pitch_keypoints": np.empty((0, 2), dtype=np.float32),
                "conf_mask": np.zeros(len(pitch_vertices), dtype=bool),
            }
        
        # TODO: Implement Roboflow API call
        # For now, return empty
        return {
            "frame_keypoints": np.empty((0, 2), dtype=np.float32),
            "pitch_keypoints": np.empty((0, 2), dtype=np.float32),
            "conf_mask": np.zeros(len(pitch_vertices), dtype=bool),
        }
    
    # ========================================================================
    # Homography
    # ========================================================================
    
    def compute_homography(self, pitch_data: List[Dict]) -> List[Optional[np.ndarray]]:
        """Compute homography matrices for each frame."""
        transformers = []
        
        for data in pitch_data:
            frame_kp = data["frame_keypoints"]
            pitch_kp = data["pitch_keypoints"]
            
            if len(frame_kp) < 4:
                transformers.append(None)
                continue
            
            try:
                H, _ = cv2.findHomography(frame_kp, pitch_kp)
                transformers.append(H)
            except Exception:
                transformers.append(None)
        
        return transformers
    
    # ========================================================================
    # Analytics
    # ========================================================================
    
    def compute_analytics(
        self, 
        tracks: Dict[str, List[Dict]], 
        transformers: Optional[List[Optional[np.ndarray]]] = None
    ) -> Dict[str, Any]:
        """Compute match analytics from tracks."""
        analytics = {
            "possession": self._compute_possession(tracks),
            "player_stats": self._compute_player_stats(tracks, transformers),
            "ball_stats": self._compute_ball_stats(tracks, transformers),
        }
        return analytics
    
    def _compute_possession(self, tracks: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Compute possession statistics."""
        team_frames = {0: 0, 1: 0}
        contested = 0
        
        for frame_idx in range(len(tracks["ball"])):
            ball = tracks["ball"][frame_idx].get(1)
            if ball is None:
                contested += 1
                continue
            
            ball_bbox = ball["bbox"]
            ball_center = np.array([(ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2])
            
            # Find closest player
            min_dist = float("inf")
            closest_team = None
            
            for pid, info in tracks["players"][frame_idx].items():
                team_id = info.get("team_id")
                bbox = info.get("bbox")
                if team_id is None or bbox is None:
                    continue
                
                player_center = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])  # Foot position
                dist = np.linalg.norm(ball_center - player_center)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_team = team_id
            
            # Check goalkeepers too
            for gk_id, info in tracks["goalkeepers"][frame_idx].items():
                team_id = info.get("team_id")
                bbox = info.get("bbox")
                if team_id is None or bbox is None:
                    continue
                
                gk_center = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                dist = np.linalg.norm(ball_center - gk_center)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_team = team_id
            
            if min_dist < 100 and closest_team is not None:  # Within 100px
                team_frames[closest_team] += 1
            else:
                contested += 1
        
        total = sum(team_frames.values()) + contested
        if total == 0:
            total = 1
        
        return {
            "team_1_percentage": team_frames[0] / total * 100,
            "team_2_percentage": team_frames[1] / total * 100,
            "team_1_frames": team_frames[0],
            "team_2_frames": team_frames[1],
            "contested_frames": contested,
        }
    
    def _compute_player_stats(
        self, 
        tracks: Dict[str, List[Dict]], 
        transformers: Optional[List[Optional[np.ndarray]]]
    ) -> Dict[int, Dict[str, Any]]:
        """Compute per-player statistics."""
        player_stats = {}
        
        # Track positions over time
        positions = defaultdict(list)
        
        for frame_idx in range(len(tracks["players"])):
            for pid, info in tracks["players"][frame_idx].items():
                bbox = info.get("bbox")
                if bbox is None:
                    continue
                
                # Foot position
                pos = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                
                # Transform to pitch coordinates if available
                if transformers and transformers[frame_idx] is not None:
                    H = transformers[frame_idx]
                    pos_h = np.array([pos[0], pos[1], 1])
                    pitch_pos = H @ pos_h
                    if pitch_pos[2] != 0:
                        pos = pitch_pos[:2] / pitch_pos[2]
                
                positions[pid].append((frame_idx, pos))
        
        # Compute stats for each player
        for pid, pos_list in positions.items():
            if len(pos_list) < 2:
                continue
            
            # Total distance
            total_dist = 0
            speeds = []
            
            for i in range(1, len(pos_list)):
                prev_frame, prev_pos = pos_list[i - 1]
                curr_frame, curr_pos = pos_list[i]
                
                dist = np.linalg.norm(curr_pos - prev_pos)
                frame_diff = curr_frame - prev_frame
                
                if frame_diff > 0:
                    total_dist += dist
                    speeds.append(dist / frame_diff * self.config.video_fps)
            
            # Get team info from last frame
            team_id = None
            for frame_idx in range(len(tracks["players"]) - 1, -1, -1):
                if pid in tracks["players"][frame_idx]:
                    team_id = tracks["players"][frame_idx][pid].get("team_id")
                    break
            
            player_stats[pid] = {
                "track_id": pid,
                "team_id": team_id,
                "total_distance_px": total_dist,
                "total_distance_m": total_dist / 100 if transformers else None,  # Rough conversion
                "avg_speed_px": np.mean(speeds) if speeds else 0,
                "max_speed_px": max(speeds) if speeds else 0,
                "frames_tracked": len(pos_list),
            }
        
        return player_stats
    
    def _compute_ball_stats(
        self, 
        tracks: Dict[str, List[Dict]], 
        transformers: Optional[List[Optional[np.ndarray]]]
    ) -> Dict[str, Any]:
        """Compute ball statistics."""
        positions = []
        
        for frame_idx, frame_data in enumerate(tracks["ball"]):
            if 1 in frame_data:
                bbox = frame_data[1]["bbox"]
                pos = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                positions.append((frame_idx, pos))
        
        if len(positions) < 2:
            return {
                "total_distance_px": 0,
                "avg_speed_px": 0,
                "max_speed_px": 0,
                "detection_rate": len(positions) / len(tracks["ball"]) if tracks["ball"] else 0,
            }
        
        total_dist = 0
        speeds = []
        
        for i in range(1, len(positions)):
            prev_frame, prev_pos = positions[i - 1]
            curr_frame, curr_pos = positions[i]
            
            dist = np.linalg.norm(curr_pos - prev_pos)
            frame_diff = curr_frame - prev_frame
            
            if frame_diff > 0 and frame_diff < 30:  # Ignore large gaps
                total_dist += dist
                speeds.append(dist / frame_diff * self.config.video_fps)
        
        return {
            "total_distance_px": total_dist,
            "avg_speed_px": np.mean(speeds) if speeds else 0,
            "max_speed_px": max(speeds) if speeds else 0,
            "detection_rate": len(positions) / len(tracks["ball"]) if tracks["ball"] else 0,
        }
    
    # ========================================================================
    # Event Detection
    # ========================================================================
    
    def detect_events(
        self, 
        tracks: Dict[str, List[Dict]], 
        analytics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect events from tracking data."""
        events = []
        
        # Detect possession changes
        prev_team = None
        possession_start = 0
        
        for frame_idx in range(len(tracks["ball"])):
            ball = tracks["ball"][frame_idx].get(1)
            if ball is None:
                continue
            
            ball_bbox = ball["bbox"]
            ball_center = np.array([(ball_bbox[0] + ball_bbox[2]) / 2, ball_bbox[3]])
            
            # Find closest player
            min_dist = float("inf")
            closest_team = None
            closest_player = None
            
            for pid, info in tracks["players"][frame_idx].items():
                team_id = info.get("team_id")
                bbox = info.get("bbox")
                if team_id is None or bbox is None:
                    continue
                
                player_foot = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                dist = np.linalg.norm(ball_center - player_foot)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_team = team_id
                    closest_player = pid
            
            if min_dist < 80 and closest_team is not None:
                if prev_team is not None and closest_team != prev_team:
                    # Possession change
                    events.append({
                        "type": "possession_change",
                        "frame": frame_idx,
                        "time": frame_idx / self.config.video_fps,
                        "from_team": prev_team,
                        "to_team": closest_team,
                        "player_id": closest_player,
                    })
                
                prev_team = closest_team
        
        # Detect potential passes (ball movement between players of same team)
        # This is a simplified heuristic
        ball_positions = []
        for frame_idx, frame_data in enumerate(tracks["ball"]):
            if 1 in frame_data:
                bbox = frame_data[1]["bbox"]
                ball_positions.append({
                    "frame": frame_idx,
                    "x": (bbox[0] + bbox[2]) / 2,
                    "y": (bbox[1] + bbox[3]) / 2,
                })
        
        # Detect high-speed ball movements (potential shots/passes)
        for i in range(1, len(ball_positions)):
            prev = ball_positions[i - 1]
            curr = ball_positions[i]
            
            frame_diff = curr["frame"] - prev["frame"]
            if frame_diff == 0:
                continue
            
            dist = np.sqrt((curr["x"] - prev["x"])**2 + (curr["y"] - prev["y"])**2)
            speed = dist / frame_diff * self.config.video_fps
            
            if speed > 500:  # High speed threshold (pixels/second)
                events.append({
                    "type": "high_speed_ball",
                    "frame": curr["frame"],
                    "time": curr["frame"] / self.config.video_fps,
                    "speed": speed,
                    "description": "Potential pass or shot",
                })
        
        return events
    
    # ========================================================================
    # Rendering
    # ========================================================================
    
    def render_outputs(
        self,
        frames: List[np.ndarray],
        tracks: Dict[str, List[Dict]],
        pitch_data: Optional[List[Dict]],
        transformers: Optional[List[Optional[np.ndarray]]],
        analysis_id: str
    ) -> Dict[str, str]:
        """Render output videos."""
        output_dir = Path(self.config.output_dir) / analysis_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get team colors
        team_colors = getattr(self, "team_colors", {0: (255, 191, 0), 1: (147, 20, 255)})
        
        # Annotated video
        annotated_path = output_dir / "annotated.mp4"
        self._render_annotated(frames, tracks, team_colors, str(annotated_path))
        
        # Radar video (if pitch data available)
        radar_path = None
        if pitch_data and transformers:
            radar_path = output_dir / "radar.mp4"
            self._render_radar(frames, tracks, pitch_data, transformers, team_colors, str(radar_path))
        
        return {
            "annotated": str(annotated_path),
            "radar": str(radar_path) if radar_path else None,
        }
    
    def _render_annotated(
        self,
        frames: List[np.ndarray],
        tracks: Dict[str, List[Dict]],
        team_colors: Dict[int, Tuple[int, int, int]],
        output_path: str
    ):
        """Render annotated video with bounding boxes."""
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.config.video_fps, (w, h))
        
        for frame_idx, frame in enumerate(tqdm(frames, desc="Rendering annotated")):
            annotated = frame.copy()
            
            # Draw players
            for pid, info in tracks["players"][frame_idx].items():
                bbox = info.get("bbox")
                team_id = info.get("team_id", 0)
                if bbox is None:
                    continue
                
                color = team_colors.get(team_id, (255, 255, 255))
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Draw ellipse at feet
                center = (int((x1 + x2) / 2), int(y2))
                axes = (int((x2 - x1) / 2), int((x2 - x1) / 4))
                cv2.ellipse(annotated, center, axes, 0, 0, 360, color, 2)
                
                # Draw label
                label = f"#{pid}"
                cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw goalkeepers
            for gk_id, info in tracks["goalkeepers"][frame_idx].items():
                bbox = info.get("bbox")
                team_id = info.get("team_id", 0)
                if bbox is None:
                    continue
                
                color = team_colors.get(team_id, (255, 255, 255))
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, "GK", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw ball
            ball = tracks["ball"][frame_idx].get(1)
            if ball:
                bbox = ball["bbox"]
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                r = int((bbox[2] - bbox[0]) / 2)
                cv2.circle(annotated, (cx, cy), max(r, 5), (255, 255, 255), -1)
                cv2.circle(annotated, (cx, cy), max(r, 5), (0, 0, 0), 2)
            
            writer.write(annotated)
        
        writer.release()
    
    def _render_radar(
        self,
        frames: List[np.ndarray],
        tracks: Dict[str, List[Dict]],
        pitch_data: List[Dict],
        transformers: List[Optional[np.ndarray]],
        team_colors: Dict[int, Tuple[int, int, int]],
        output_path: str
    ):
        """Render radar view video."""
        # Create pitch background
        pitch_w, pitch_h = 1050, 680  # Scaled pitch dimensions
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.config.video_fps, (w, h))
        
        for frame_idx, frame in enumerate(tqdm(frames, desc="Rendering radar")):
            output = frame.copy()
            
            H = transformers[frame_idx] if frame_idx < len(transformers) else None
            if H is None:
                writer.write(output)
                continue
            
            # Create radar overlay
            radar = np.zeros((pitch_h, pitch_w, 3), dtype=np.uint8)
            radar[:] = (20, 30, 20)  # Dark green background
            
            # Draw pitch lines
            cv2.rectangle(radar, (0, 0), (pitch_w - 1, pitch_h - 1), (50, 100, 50), 2)
            cv2.line(radar, (pitch_w // 2, 0), (pitch_w // 2, pitch_h), (50, 100, 50), 2)
            cv2.circle(radar, (pitch_w // 2, pitch_h // 2), 91, (50, 100, 50), 2)
            
            # Transform and draw players
            for pid, info in tracks["players"][frame_idx].items():
                bbox = info.get("bbox")
                team_id = info.get("team_id", 0)
                if bbox is None:
                    continue
                
                foot_pos = np.array([[(bbox[0] + bbox[2]) / 2, bbox[3], 1]])
                pitch_pos = (H @ foot_pos.T).T
                if pitch_pos[0, 2] != 0:
                    px = int(pitch_pos[0, 0] / pitch_pos[0, 2] / 100)
                    py = int(pitch_pos[0, 1] / pitch_pos[0, 2] / 100)
                    
                    if 0 <= px < pitch_w and 0 <= py < pitch_h:
                        color = team_colors.get(team_id, (255, 255, 255))
                        cv2.circle(radar, (px, py), 8, color, -1)
                        cv2.circle(radar, (px, py), 8, (0, 0, 0), 1)
            
            # Draw ball
            ball = tracks["ball"][frame_idx].get(1)
            if ball:
                bbox = ball["bbox"]
                ball_pos = np.array([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, 1]])
                pitch_pos = (H @ ball_pos.T).T
                if pitch_pos[0, 2] != 0:
                    px = int(pitch_pos[0, 0] / pitch_pos[0, 2] / 100)
                    py = int(pitch_pos[0, 1] / pitch_pos[0, 2] / 100)
                    
                    if 0 <= px < pitch_w and 0 <= py < pitch_h:
                        cv2.circle(radar, (px, py), 5, (255, 255, 255), -1)
            
            # Overlay radar on frame
            radar_h = int(h * 0.3)
            radar_w = int(radar_h * pitch_w / pitch_h)
            radar_resized = cv2.resize(radar, (radar_w, radar_h))
            
            # Position at bottom center
            x_offset = (w - radar_w) // 2
            y_offset = h - radar_h - 20
            
            # Blend with transparency
            alpha = 0.7
            roi = output[y_offset:y_offset + radar_h, x_offset:x_offset + radar_w]
            output[y_offset:y_offset + radar_h, x_offset:x_offset + radar_w] = \
                cv2.addWeighted(roi, 1 - alpha, radar_resized, alpha, 0)
            
            writer.write(output)
        
        writer.release()
