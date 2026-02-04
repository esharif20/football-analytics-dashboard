"""
Pitch detection and homography transformation module.
Matches the original repo's view_transformer.py structure.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import cv2


@dataclass
class PitchKeypoints:
    """Detected pitch keypoints."""
    keypoints: Dict[str, Tuple[float, float]]  # name -> (x, y) in image coords
    confidence: float
    frame_idx: int


@dataclass  
class PitchDimensions:
    """Standard pitch dimensions in meters."""
    length: float = 105.0
    width: float = 68.0
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.3
    goal_area_length: float = 5.5
    goal_area_width: float = 18.3
    center_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0


# Standard pitch keypoint positions (normalized 0-1)
PITCH_KEYPOINTS_NORMALIZED = {
    "top_left": (0, 0),
    "top_right": (1, 0),
    "bottom_left": (0, 1),
    "bottom_right": (1, 1),
    "center_top": (0.5, 0),
    "center_bottom": (0.5, 1),
    "center": (0.5, 0.5),
    "left_penalty_top": (0, 0.5 - 40.3/68/2),
    "left_penalty_bottom": (0, 0.5 + 40.3/68/2),
    "left_penalty_arc": (16.5/105, 0.5),
    "right_penalty_top": (1, 0.5 - 40.3/68/2),
    "right_penalty_bottom": (1, 0.5 + 40.3/68/2),
    "right_penalty_arc": (1 - 16.5/105, 0.5),
}


class PitchDetector:
    """
    Detects pitch keypoints using a local model or Roboflow API.
    """
    
    def __init__(
        self,
        method: str = "local",
        local_model_path: Optional[str] = None,
        roboflow_api_key: Optional[str] = None,
        roboflow_model_id: str = "football-field-detection-f07vi/15",
        device: str = "auto"
    ):
        """
        Initialize the pitch detector.
        
        Args:
            method: Detection method ("local" or "roboflow")
            local_model_path: Path to local keypoint detection model
            roboflow_api_key: Roboflow API key (for roboflow method)
            roboflow_model_id: Roboflow model ID
            device: Device for inference
        """
        self.method = method
        self.local_model_path = local_model_path
        self.roboflow_api_key = roboflow_api_key
        self.roboflow_model_id = roboflow_model_id
        self.device = device
        
        self._model = None
        self._rf_model = None
    
    def _load_local_model(self):
        """Load local keypoint detection model."""
        if self._model is None and self.local_model_path:
            if Path(self.local_model_path).exists():
                print(f"Loading pitch detection model from {self.local_model_path}...")
                try:
                    from ultralytics import YOLO
                    self._model = YOLO(self.local_model_path)
                except Exception as e:
                    print(f"Warning: Could not load pitch model: {e}")
                    self._model = "failed"
            else:
                print(f"Warning: Pitch model not found at {self.local_model_path}")
                self._model = "failed"
    
    def _load_roboflow_model(self):
        """Load Roboflow model."""
        if self._rf_model is None and self.roboflow_api_key:
            try:
                from roboflow import Roboflow
                rf = Roboflow(api_key=self.roboflow_api_key)
                project_id, version = self.roboflow_model_id.rsplit("/", 1)
                project = rf.workspace().project(project_id)
                self._rf_model = project.version(int(version)).model
                print("Loaded Roboflow pitch detection model")
            except Exception as e:
                print(f"Warning: Could not load Roboflow model: {e}")
                self._rf_model = "failed"
    
    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> Optional[PitchKeypoints]:
        """
        Detect pitch keypoints in a frame.
        
        Args:
            frame: BGR image
            frame_idx: Frame index
            
        Returns:
            PitchKeypoints or None if detection failed
        """
        if self.method == "local":
            return self._detect_local(frame, frame_idx)
        else:
            return self._detect_roboflow(frame, frame_idx)
    
    def _detect_local(self, frame: np.ndarray, frame_idx: int) -> Optional[PitchKeypoints]:
        """Detect using local model."""
        self._load_local_model()
        
        if self._model is None or self._model == "failed":
            # Fallback to Roboflow if local fails
            if self.roboflow_api_key:
                print("Falling back to Roboflow API...")
                return self._detect_roboflow(frame, frame_idx)
            return None
        
        try:
            results = self._model.predict(frame, verbose=False)[0]
            
            # Parse keypoints from model output
            keypoints = {}
            if hasattr(results, 'keypoints') and results.keypoints is not None:
                kpts = results.keypoints.xy[0].cpu().numpy()
                conf = results.keypoints.conf[0].cpu().numpy() if results.keypoints.conf is not None else np.ones(len(kpts))
                
                # Map keypoints to names (depends on model training)
                keypoint_names = list(PITCH_KEYPOINTS_NORMALIZED.keys())
                for i, (kpt, c) in enumerate(zip(kpts, conf)):
                    if i < len(keypoint_names) and c > 0.3:
                        keypoints[keypoint_names[i]] = (float(kpt[0]), float(kpt[1]))
            
            if len(keypoints) >= 4:
                return PitchKeypoints(
                    keypoints=keypoints,
                    confidence=float(np.mean(list(conf))),
                    frame_idx=frame_idx
                )
        except Exception as e:
            print(f"Warning: Local pitch detection failed: {e}")
        
        return None
    
    def _detect_roboflow(self, frame: np.ndarray, frame_idx: int) -> Optional[PitchKeypoints]:
        """Detect using Roboflow API."""
        self._load_roboflow_model()
        
        if self._rf_model is None or self._rf_model == "failed":
            return None
        
        try:
            # Save frame temporarily for Roboflow
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, frame)
                prediction = self._rf_model.predict(f.name, confidence=30).json()
                Path(f.name).unlink()
            
            # Parse predictions
            keypoints = {}
            for pred in prediction.get("predictions", []):
                name = pred.get("class", "")
                x = pred.get("x", 0)
                y = pred.get("y", 0)
                
                # Map Roboflow class names to our keypoint names
                mapped_name = self._map_roboflow_class(name)
                if mapped_name:
                    keypoints[mapped_name] = (x, y)
            
            if len(keypoints) >= 4:
                return PitchKeypoints(
                    keypoints=keypoints,
                    confidence=0.8,
                    frame_idx=frame_idx
                )
        except Exception as e:
            print(f"Warning: Roboflow pitch detection failed: {e}")
        
        return None
    
    def _map_roboflow_class(self, class_name: str) -> Optional[str]:
        """Map Roboflow class names to standard keypoint names."""
        # This mapping depends on the specific Roboflow model
        mapping = {
            "corner_top_left": "top_left",
            "corner_top_right": "top_right",
            "corner_bottom_left": "bottom_left",
            "corner_bottom_right": "bottom_right",
            "center_circle_top": "center_top",
            "center_circle_bottom": "center_bottom",
            "center_spot": "center",
            # Add more mappings as needed
        }
        return mapping.get(class_name.lower(), class_name if class_name in PITCH_KEYPOINTS_NORMALIZED else None)


class ViewTransformer:
    """
    Transforms pixel coordinates to pitch coordinates using homography.
    """
    
    def __init__(
        self,
        pitch_dimensions: PitchDimensions = None,
        min_keypoints: int = 4,
        reprojection_error_threshold: float = 10.0
    ):
        """
        Initialize the view transformer.
        
        Args:
            pitch_dimensions: Pitch dimensions in meters
            min_keypoints: Minimum keypoints required for homography
            reprojection_error_threshold: Maximum allowed reprojection error
        """
        self.pitch_dimensions = pitch_dimensions or PitchDimensions()
        self.min_keypoints = min_keypoints
        self.reprojection_error_threshold = reprojection_error_threshold
        
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.is_valid: bool = False
    
    def compute_homography(self, keypoints: PitchKeypoints) -> bool:
        """
        Compute homography matrix from detected keypoints.
        
        Args:
            keypoints: Detected pitch keypoints
            
        Returns:
            True if homography was computed successfully
        """
        if len(keypoints.keypoints) < self.min_keypoints:
            self.is_valid = False
            return False
        
        # Get source (image) and destination (pitch) points
        src_points = []
        dst_points = []
        
        for name, (img_x, img_y) in keypoints.keypoints.items():
            if name in PITCH_KEYPOINTS_NORMALIZED:
                norm_x, norm_y = PITCH_KEYPOINTS_NORMALIZED[name]
                pitch_x = norm_x * self.pitch_dimensions.length
                pitch_y = norm_y * self.pitch_dimensions.width
                
                src_points.append([img_x, img_y])
                dst_points.append([pitch_x, pitch_y])
        
        if len(src_points) < self.min_keypoints:
            self.is_valid = False
            return False
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Compute homography
        try:
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            
            if H is None:
                self.is_valid = False
                return False
            
            # Check reprojection error
            transformed = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H)
            errors = np.linalg.norm(transformed.reshape(-1, 2) - dst_points, axis=1)
            mean_error = np.mean(errors)
            
            if mean_error > self.reprojection_error_threshold:
                self.is_valid = False
                return False
            
            self.homography_matrix = H
            self.inverse_homography = np.linalg.inv(H)
            self.is_valid = True
            return True
            
        except Exception as e:
            print(f"Warning: Homography computation failed: {e}")
            self.is_valid = False
            return False
    
    def transform_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Transform a point from image to pitch coordinates.
        
        Args:
            x: Image x coordinate
            y: Image y coordinate
            
        Returns:
            (pitch_x, pitch_y) in meters, or None if transform invalid
        """
        if not self.is_valid or self.homography_matrix is None:
            return None
        
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        
        return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))
    
    def transform_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Transform multiple points from image to pitch coordinates.
        
        Args:
            points: Array of shape (N, 2) with image coordinates
            
        Returns:
            Array of shape (N, 2) with pitch coordinates, or None
        """
        if not self.is_valid or self.homography_matrix is None:
            return None
        
        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(points, self.homography_matrix)
        
        return transformed.reshape(-1, 2)
    
    def inverse_transform_point(self, pitch_x: float, pitch_y: float) -> Optional[Tuple[float, float]]:
        """
        Transform a point from pitch to image coordinates.
        
        Args:
            pitch_x: Pitch x coordinate in meters
            pitch_y: Pitch y coordinate in meters
            
        Returns:
            (image_x, image_y) in pixels, or None if transform invalid
        """
        if not self.is_valid or self.inverse_homography is None:
            return None
        
        point = np.array([[[pitch_x, pitch_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.inverse_homography)
        
        return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))


class HomographySmoother:
    """
    Smooths homography matrices across frames to reduce jitter.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the smoother.
        
        Args:
            window_size: Number of frames to average
        """
        self.window_size = window_size
        self.buffer: List[np.ndarray] = []
    
    def smooth(self, homography: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Add a homography to the buffer and return smoothed result.
        
        Args:
            homography: Homography matrix (3x3) or None
            
        Returns:
            Smoothed homography matrix or None
        """
        if homography is None:
            # Use last valid if available
            if self.buffer:
                return self.buffer[-1]
            return None
        
        self.buffer.append(homography.copy())
        
        # Keep buffer at window size
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        # Average matrices
        if len(self.buffer) == 0:
            return None
        
        smoothed = np.mean(self.buffer, axis=0)
        return smoothed
    
    def reset(self):
        """Clear the buffer."""
        self.buffer = []
