"""
CLI argument parsing for the football analysis pipeline.
Matches the interface of the original repo.
"""

import argparse
from typing import Optional
from ..config import PipelineConfig, PIPELINE_MODE_FEATURES


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the pipeline CLI."""
    parser = argparse.ArgumentParser(
        description="Football Match Analysis Pipeline - Spatio-Temporal Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Modes:
  all      Full pipeline with all features (detection, tracking, teams, pitch, analytics)
  radar    Detection + tracking + teams + pitch radar overlay
  team     Detection + tracking + team classification
  track    Detection + tracking only
  players  Player detection only (no tracking)
  ball     Ball detection and tracking only
  pitch    Pitch keypoint detection and homography only

Camera Types:
  tactical   Wide-angle tactical/DFL Bundesliga style footage (default)
  broadcast  Standard broadcast camera angle (Coming Soon)

Examples:
  python main.py --video input.mp4 --mode all
  python main.py --video input.mp4 --mode radar --output-dir ./results
  python main.py --video input.mp4 --mode all --use-roboflow
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    # Pipeline mode
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="all",
        choices=list(PIPELINE_MODE_FEATURES.keys()),
        help="Pipeline mode (default: all)"
    )
    
    # Camera type
    parser.add_argument(
        "--camera-type", "-c",
        type=str,
        default="tactical",
        choices=["tactical", "broadcast"],
        help="Camera type - broadcast is Coming Soon (default: tactical)"
    )
    
    # Model options
    parser.add_argument(
        "--det-model",
        type=str,
        default=None,
        help="Path to player detection model (default: models/yolov8x.pt)"
    )
    
    parser.add_argument(
        "--ball-model",
        type=str,
        default=None,
        help="Path to ball detection model (default: models/ball_detection.pt)"
    )
    
    parser.add_argument(
        "--pitch-model",
        type=str,
        default=None,
        help="Path to pitch detection model (default: models/pitch_detection.pt)"
    )
    
    # Pitch detection method
    parser.add_argument(
        "--use-roboflow",
        action="store_true",
        help="Use Roboflow API for pitch detection instead of local model"
    )
    
    parser.add_argument(
        "--roboflow-key",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    
    # Device
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run inference on (default: auto)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: output/<video_name>/)"
    )
    
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Skip generating annotated video"
    )
    
    parser.add_argument(
        "--no-radar",
        action="store_true",
        help="Skip generating radar video"
    )
    
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving tracking/analytics JSON files"
    )
    
    # Caching
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of intermediate results"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing cache before running"
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    return parser


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_parser()
    return parser.parse_args(args)


def args_to_config(args: argparse.Namespace) -> PipelineConfig:
    """Convert parsed arguments to PipelineConfig."""
    config = PipelineConfig()
    
    # Pipeline mode
    config.mode = args.mode
    
    # Camera type
    config.camera_type = args.camera_type
    if args.camera_type == "broadcast":
        print("\n⚠️  Broadcast camera angle support is Coming Soon!")
        print("    Currently only tactical/wide-angle footage is supported.")
        print("    Proceeding with tactical mode...\n")
        config.camera_type = "tactical"
    
    # Device
    if args.device != "auto":
        config.detection.device = args.device
    
    # Model paths
    if args.det_model:
        config.detection.player_model_path = args.det_model
    
    if args.ball_model:
        config.detection.ball_model_path = args.ball_model
    
    if args.pitch_model:
        config.pitch.local_model_path = args.pitch_model
    
    # Pitch detection method
    if args.use_roboflow:
        config.pitch.detection_method = "roboflow"
        if args.roboflow_key:
            config.pitch.roboflow_api_key = args.roboflow_key
    
    # Output options
    if args.output_dir:
        config.output.output_dir = args.output_dir
    
    config.output.annotated_video = not args.no_annotated
    config.output.radar_video = not args.no_radar
    config.output.tracking_json = not args.no_json
    config.output.analytics_json = not args.no_json
    
    # Caching
    config.use_cache = not args.no_cache
    
    # Logging
    if args.quiet:
        config.verbose = False
        config.log_level = "ERROR"
    elif args.verbose:
        config.verbose = True
        config.log_level = "DEBUG"
    
    return config
