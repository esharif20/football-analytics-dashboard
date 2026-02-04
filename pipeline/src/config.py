"""
Configuration module for the football analysis pipeline.
Matches the structure of the original Spatio-Temporal-GNN-Football-Analysis repo.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
STUBS_DIR = BASE_DIR / "stubs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
STUBS_DIR.mkdir(exist_ok=True)


@dataclass
class DetectionConfig:
    """Detection model configuration."""
    player_model_path: str = str(MODELS_DIR / "yolov8x.pt")
    ball_model_path: Optional[str] = str(MODELS_DIR / "ball_detection.pt")
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5
    device: str = "auto"  # auto, cuda, mps, cpu


@dataclass
class TrackingConfig:
    """ByteTrack tracking configuration."""
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 25


@dataclass
class TeamAssignerConfig:
    """Team classification configuration."""
    embedding_model: str = "google/siglip-base-patch16-224"
    n_clusters: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 3
    min_samples_per_cluster: int = 5


@dataclass
class PitchConfig:
    """Pitch detection and homography configuration."""
    # Detection method: "local" uses custom model, "roboflow" uses API
    detection_method: Literal["local", "roboflow"] = "local"
    local_model_path: Optional[str] = str(MODELS_DIR / "pitch_detection.pt")
    
    # Roboflow API settings (fallback)
    roboflow_api_key: Optional[str] = os.getenv("ROBOFLOW_API_KEY")
    roboflow_model_id: str = "football-field-detection-f07vi/15"
    
    # Homography settings
    min_keypoints: int = 4
    reprojection_error_threshold: float = 10.0
    smoothing_window: int = 5
    
    # Pitch dimensions (FIFA standard in meters)
    pitch_length: float = 105.0
    pitch_width: float = 68.0


@dataclass
class AnalyticsConfig:
    """Analytics computation configuration."""
    possession_ball_distance_threshold: float = 50.0  # pixels
    speed_smoothing_window: int = 5
    min_pass_distance: float = 10.0  # meters
    high_speed_threshold: float = 25.0  # km/h for ball


@dataclass
class OutputConfig:
    """Output video and file configuration."""
    output_dir: str = str(OUTPUT_DIR)
    video_codec: str = "mp4v"
    video_fps: int = 25
    annotated_video: bool = True
    radar_video: bool = True
    tracking_json: bool = True
    analytics_json: bool = True


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all sub-configs."""
    # Pipeline mode
    mode: Literal["all", "radar", "team", "track", "players", "ball", "pitch"] = "all"
    
    # Camera type (for future broadcast angle support)
    camera_type: Literal["tactical", "broadcast"] = "tactical"
    
    # Sub-configurations
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    team_assigner: TeamAssignerConfig = field(default_factory=TeamAssignerConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Caching
    use_cache: bool = True
    cache_dir: str = str(STUBS_DIR)
    
    # Logging
    verbose: bool = True
    log_level: str = "INFO"


# Pipeline mode features (what each mode includes)
PIPELINE_MODE_FEATURES = {
    "all": {
        "detection": True,
        "tracking": True,
        "team_classification": True,
        "pitch_detection": True,
        "homography": True,
        "analytics": True,
        "annotated_video": True,
        "radar_video": True,
    },
    "radar": {
        "detection": True,
        "tracking": True,
        "team_classification": True,
        "pitch_detection": True,
        "homography": True,
        "analytics": False,
        "annotated_video": False,
        "radar_video": True,
    },
    "team": {
        "detection": True,
        "tracking": True,
        "team_classification": True,
        "pitch_detection": False,
        "homography": False,
        "analytics": False,
        "annotated_video": True,
        "radar_video": False,
    },
    "track": {
        "detection": True,
        "tracking": True,
        "team_classification": False,
        "pitch_detection": False,
        "homography": False,
        "analytics": False,
        "annotated_video": True,
        "radar_video": False,
    },
    "players": {
        "detection": True,
        "tracking": False,
        "team_classification": False,
        "pitch_detection": False,
        "homography": False,
        "analytics": False,
        "annotated_video": True,
        "radar_video": False,
    },
    "ball": {
        "detection": True,  # Ball only
        "tracking": True,
        "team_classification": False,
        "pitch_detection": False,
        "homography": False,
        "analytics": False,
        "annotated_video": True,
        "radar_video": False,
    },
    "pitch": {
        "detection": False,
        "tracking": False,
        "team_classification": False,
        "pitch_detection": True,
        "homography": True,
        "analytics": False,
        "annotated_video": True,
        "radar_video": False,
    },
}


def get_device() -> str:
    """Auto-detect the best available device."""
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_config_from_env() -> PipelineConfig:
    """Load configuration from environment variables."""
    config = PipelineConfig()
    
    # Override from environment
    if os.getenv("PIPELINE_MODE"):
        config.mode = os.getenv("PIPELINE_MODE")
    
    if os.getenv("DEVICE"):
        config.detection.device = os.getenv("DEVICE")
    elif config.detection.device == "auto":
        config.detection.device = get_device()
    
    if os.getenv("DET_MODEL_PATH"):
        config.detection.player_model_path = os.getenv("DET_MODEL_PATH")
    
    if os.getenv("BALL_MODEL_PATH"):
        config.detection.ball_model_path = os.getenv("BALL_MODEL_PATH")
    
    if os.getenv("PITCH_MODEL_PATH"):
        config.pitch.local_model_path = os.getenv("PITCH_MODEL_PATH")
    
    if os.getenv("ROBOFLOW_API_KEY"):
        config.pitch.roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if os.getenv("OUTPUT_DIR"):
        config.output.output_dir = os.getenv("OUTPUT_DIR")
    
    return config
