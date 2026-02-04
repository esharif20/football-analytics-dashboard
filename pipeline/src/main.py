#!/usr/bin/env python3
"""
Football Analysis Pipeline - Main Entry Point

Usage:
    python main.py --video input.mp4 --mode all
    python main.py --video input.mp4 --mode radar --use-roboflow
    python main.py --video input.mp4 --mode team --output-dir ./results

For help:
    python main.py --help
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cli import parse_args, args_to_config
from config import PipelineConfig, PIPELINE_MODE_FEATURES, load_config_from_env
from trackers import FootballTracker
from team_assigner import TeamAssigner
from pitch import PitchDetector, ViewTransformer, HomographySmoother
from analytics import AnalyticsEngine


def load_video(video_path: str) -> tuple:
    """Load video and return frames and metadata."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Loading video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    return frames, {"fps": fps, "width": width, "height": height, "total_frames": len(frames)}


def save_video(frames: list, output_path: str, fps: int = 25):
    """Save frames to video file."""
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Saved video: {output_path}")


def run_pipeline(video_path: str, config: PipelineConfig) -> dict:
    """
    Run the football analysis pipeline.
    
    Args:
        video_path: Path to input video
        config: Pipeline configuration
        
    Returns:
        Dictionary with all results
    """
    start_time = time.time()
    features = PIPELINE_MODE_FEATURES[config.mode]
    
    # Create output directory
    video_name = Path(video_path).stem
    output_dir = Path(config.output.output_dir) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Football Analysis Pipeline - Mode: {config.mode.upper()}")
    print(f"{'='*60}\n")
    
    # Load video
    print("[1/7] Loading video...")
    frames, video_meta = load_video(video_path)
    
    results = {
        "video_path": video_path,
        "mode": config.mode,
        "video_meta": video_meta,
        "tracks": None,
        "team_assignments": None,
        "pitch_data": None,
        "analytics": None,
        "events": None,
        "outputs": {}
    }
    
    # Detection and Tracking
    if features["detection"]:
        print("\n[2/7] Running detection and tracking...")
        tracker = FootballTracker(
            player_model_path=config.detection.player_model_path,
            ball_model_path=config.detection.ball_model_path,
            device=config.detection.device,
            confidence_threshold=config.detection.confidence_threshold
        )
        
        detections, tracks = tracker.process_video(
            frames,
            progress_callback=lambda p, m: print(f"  {m}") if config.verbose else None
        )
        
        # Interpolate ball
        if features["tracking"]:
            print("  Interpolating ball positions...")
            tracks = tracker.interpolate_ball(tracks)
        
        results["tracks"] = tracks
        print(f"  Detected {len(set(pid for f in tracks for pid in f.get('players', {})))} unique players")
    
    # Team Classification
    if features["team_classification"] and results["tracks"]:
        print("\n[3/7] Classifying teams...")
        team_assigner = TeamAssigner(
            embedding_model=config.team_assigner.embedding_model,
            device=config.detection.device
        )
        
        team_assignments = team_assigner.classify_teams(frames, results["tracks"])
        gk_teams = team_assigner.assign_goalkeepers(frames, results["tracks"])
        
        # Apply teams to tracks
        results["tracks"] = team_assigner.apply_teams_to_tracks(results["tracks"], gk_teams)
        results["team_assignments"] = {**team_assignments, **gk_teams}
        results["team_colors"] = team_assigner.team_colors
        
        print(f"  Team 1: {sum(1 for t in team_assignments.values() if t == 0)} players")
        print(f"  Team 2: {sum(1 for t in team_assignments.values() if t == 1)} players")
    
    # Pitch Detection
    transformers = [None] * len(frames)
    if features["pitch_detection"]:
        print("\n[4/7] Detecting pitch keypoints...")
        pitch_detector = PitchDetector(
            method=config.pitch.detection_method,
            local_model_path=config.pitch.local_model_path,
            roboflow_api_key=config.pitch.roboflow_api_key,
            device=config.detection.device
        )
        
        smoother = HomographySmoother(window_size=config.pitch.smoothing_window)
        pitch_data = []
        valid_count = 0
        
        for i, frame in enumerate(frames):
            if i % 10 == 0 and config.verbose:
                print(f"  Processing frame {i}/{len(frames)}")
            
            keypoints = pitch_detector.detect(frame, i)
            
            if keypoints and features["homography"]:
                transformer = ViewTransformer(min_keypoints=config.pitch.min_keypoints)
                if transformer.compute_homography(keypoints):
                    # Smooth homography
                    smoothed_H = smoother.smooth(transformer.homography_matrix)
                    transformer.homography_matrix = smoothed_H
                    transformer.is_valid = True
                    transformers[i] = transformer
                    valid_count += 1
            
            pitch_data.append(keypoints)
        
        results["pitch_data"] = pitch_data
        print(f"  Valid homographies: {valid_count}/{len(frames)} frames")
    
    # Analytics
    if features["analytics"] and results["tracks"] and results.get("team_assignments"):
        print("\n[5/7] Computing analytics...")
        analytics_engine = AnalyticsEngine(
            fps=video_meta["fps"],
            possession_distance_threshold=config.analytics.possession_ball_distance_threshold
        )
        
        possession = analytics_engine.compute_possession(
            results["tracks"],
            results["team_assignments"]
        )
        
        player_kinematics = analytics_engine.compute_player_kinematics(
            results["tracks"],
            results["team_assignments"],
            transformers
        )
        
        ball_stats = analytics_engine.compute_ball_stats(results["tracks"], transformers)
        
        events = analytics_engine.detect_events(
            results["tracks"],
            results["team_assignments"],
            possession,
            ball_stats
        )
        
        results["analytics"] = analytics_engine.to_dict(
            possession, player_kinematics, ball_stats, events
        )
        results["events"] = events
        
        print(f"  Possession: Team 1 {possession.team_1_percentage:.1f}% | Team 2 {possession.team_2_percentage:.1f}%")
        print(f"  Events detected: {len(events)}")
    
    # Render outputs
    print("\n[6/7] Rendering outputs...")
    
    if config.output.annotated_video and features.get("annotated_video", True):
        print("  Rendering annotated video...")
        annotated_frames = render_annotated_video(
            frames, results["tracks"], results.get("team_colors")
        )
        annotated_path = str(output_dir / f"{video_name}_annotated.mp4")
        save_video(annotated_frames, annotated_path, video_meta["fps"])
        results["outputs"]["annotated"] = annotated_path
    
    if config.output.radar_video and features.get("radar_video", False):
        print("  Rendering radar video...")
        radar_frames = render_radar_video(
            frames, results["tracks"], transformers, results.get("team_colors")
        )
        radar_path = str(output_dir / f"{video_name}_radar.mp4")
        save_video(radar_frames, radar_path, video_meta["fps"])
        results["outputs"]["radar"] = radar_path
    
    # Save JSON outputs
    print("\n[7/7] Saving data files...")
    
    if config.output.tracking_json and results["tracks"]:
        tracks_path = str(output_dir / f"{video_name}_tracks.json")
        save_tracks_json(results["tracks"], tracks_path)
        results["outputs"]["tracks_json"] = tracks_path
    
    if config.output.analytics_json and results.get("analytics"):
        analytics_path = str(output_dir / f"{video_name}_analytics.json")
        with open(analytics_path, 'w') as f:
            json.dump(results["analytics"], f, indent=2, default=str)
        print(f"  Saved: {analytics_path}")
        results["outputs"]["analytics_json"] = analytics_path
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline Complete!")
    print(f"{'='*60}")
    print(f"  Mode: {config.mode}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(frames)*1000:.1f}ms/frame)")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")
    
    return results


def render_annotated_video(frames, tracks, team_colors):
    """Render annotated video with bounding boxes and labels."""
    annotated = []
    
    team_1_color = team_colors.team_1_color if team_colors else (0, 255, 0)
    team_2_color = team_colors.team_2_color if team_colors else (0, 0, 255)
    
    for i, frame in enumerate(frames):
        annotated_frame = frame.copy()
        
        if tracks and i < len(tracks):
            frame_tracks = tracks[i]
            
            # Draw players
            for player_id, player in frame_tracks.get("players", {}).items():
                x1, y1, x2, y2 = map(int, player.bbox)
                color = team_1_color if player.team_id == 0 else team_2_color
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, str(player_id), (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw goalkeepers
            for gk_id, gk in frame_tracks.get("goalkeepers", {}).items():
                x1, y1, x2, y2 = map(int, gk.bbox)
                color = (255, 255, 0)  # Yellow for GK
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"GK{gk_id}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw ball
            for ball_id, ball in frame_tracks.get("ball", {}).items():
                x1, y1, x2, y2 = map(int, ball.bbox)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(annotated_frame, (cx, cy), 10, (0, 255, 255), -1)
        
        annotated.append(annotated_frame)
    
    return annotated


def render_radar_video(frames, tracks, transformers, team_colors, pitch_size=(700, 450)):
    """Render 2D radar view of the pitch."""
    radar_frames = []
    
    team_1_color = team_colors.team_1_color if team_colors else (0, 255, 0)
    team_2_color = team_colors.team_2_color if team_colors else (0, 0, 255)
    
    # Pitch dimensions in meters
    pitch_length = 105.0
    pitch_width = 68.0
    
    for i, frame in enumerate(frames):
        # Create pitch background
        radar = np.zeros((pitch_size[1], pitch_size[0], 3), dtype=np.uint8)
        radar[:] = (34, 139, 34)  # Green field
        
        # Draw pitch lines
        cv2.rectangle(radar, (10, 10), (pitch_size[0]-10, pitch_size[1]-10), (255, 255, 255), 2)
        cv2.line(radar, (pitch_size[0]//2, 10), (pitch_size[0]//2, pitch_size[1]-10), (255, 255, 255), 2)
        cv2.circle(radar, (pitch_size[0]//2, pitch_size[1]//2), 50, (255, 255, 255), 2)
        
        if tracks and i < len(tracks):
            frame_tracks = tracks[i]
            transformer = transformers[i] if transformers and i < len(transformers) else None
            
            # Draw players
            for player_id, player in frame_tracks.get("players", {}).items():
                cx = (player.bbox[0] + player.bbox[2]) / 2
                cy = (player.bbox[1] + player.bbox[3]) / 2
                
                # Transform to pitch coordinates
                if transformer and transformer.is_valid:
                    pitch_pos = transformer.transform_point(cx, cy)
                    if pitch_pos:
                        # Scale to radar size
                        rx = int(pitch_pos[0] / pitch_length * (pitch_size[0] - 20) + 10)
                        ry = int(pitch_pos[1] / pitch_width * (pitch_size[1] - 20) + 10)
                        
                        color = team_1_color if player.team_id == 0 else team_2_color
                        cv2.circle(radar, (rx, ry), 8, color, -1)
                        cv2.circle(radar, (rx, ry), 8, (255, 255, 255), 1)
            
            # Draw ball
            for ball_id, ball in frame_tracks.get("ball", {}).items():
                cx = (ball.bbox[0] + ball.bbox[2]) / 2
                cy = (ball.bbox[1] + ball.bbox[3]) / 2
                
                if transformer and transformer.is_valid:
                    pitch_pos = transformer.transform_point(cx, cy)
                    if pitch_pos:
                        rx = int(pitch_pos[0] / pitch_length * (pitch_size[0] - 20) + 10)
                        ry = int(pitch_pos[1] / pitch_width * (pitch_size[1] - 20) + 10)
                        cv2.circle(radar, (rx, ry), 6, (0, 255, 255), -1)
        
        radar_frames.append(radar)
    
    return radar_frames


def save_tracks_json(tracks, output_path):
    """Save tracks to JSON file."""
    # Convert Track objects to dictionaries
    serializable_tracks = []
    
    for frame_tracks in tracks:
        frame_dict = {}
        for category in ["players", "goalkeepers", "referees", "ball"]:
            frame_dict[category] = {}
            for track_id, track in frame_tracks.get(category, {}).items():
                frame_dict[category][str(track_id)] = {
                    "bbox": list(track.bbox),
                    "confidence": track.confidence,
                    "team_id": track.team_id
                }
        serializable_tracks.append(frame_dict)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_tracks, f)
    
    print(f"  Saved: {output_path}")


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Check for broadcast camera type
    if args.camera_type == "broadcast":
        print("\n" + "="*60)
        print("⚠️  BROADCAST CAMERA ANGLE - COMING SOON")
        print("="*60)
        print("\nThe broadcast/normal camera angle pipeline is under development.")
        print("This requires different models trained on broadcast footage.")
        print("\nCurrently supported: Tactical/wide-angle footage (DFL Bundesliga style)")
        print("\nProceeding with tactical mode for now...")
        print("="*60 + "\n")
    
    # Convert to config
    config = args_to_config(args)
    
    # Run pipeline
    try:
        results = run_pipeline(args.video, config)
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
