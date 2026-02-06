#!/usr/bin/env python3
"""
Football Analysis Worker Service

A background process that:
1. Polls the dashboard API for pending analyses
2. Runs the CV pipeline on new videos
3. Posts results back to the dashboard

Dependencies: requests, torch, ultralytics, opencv-python, supervision
"""

import os
import sys
import json
import time
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Use requests library for HTTP (urllib has issues with HTTP/2)
try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Run: pip install requests")
    sys.exit(1)

# Configuration
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://localhost:3000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))  # seconds
MODELS_DIR = Path(__file__).parent / "models"
INPUT_DIR = Path(__file__).parent / "input_videos"
OUTPUT_DIR = Path(__file__).parent / "output_videos"
STUBS_DIR = Path(__file__).parent / "stubs"
CACHE_DIR = Path(__file__).parent / "cache"

# Model URLs (hosted on CDN)
MODEL_URLS = {
    "player_detection.pt": "https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/XAzhckYwibJeQRhg.pt",
    "ball_detection.pt": "https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/NiUwnYcULyjvIBhr.pt",
    "pitch_detection.pt": "https://files.manuscdn.com/user_upload_by_module/session_file/310519663334363677/pSlXgeDoBtmXQHTJ.pt",
}

# Ensure directories exist
for d in [INPUT_DIR, OUTPUT_DIR, STUBS_DIR, CACHE_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str, level: str = "INFO"):
    """Simple logging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def download_models():
    """Download models from CDN if not already present."""
    for model_name, url in MODEL_URLS.items():
        model_path = MODELS_DIR / model_name
        if model_path.exists():
            log(f"Model already exists: {model_name}")
            continue
        
        log(f"Downloading model: {model_name}...")
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            log(f"Downloaded: {model_name} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            log(f"Failed to download {model_name}: {e}", "ERROR")


def api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Optional[Dict]:
    """Make HTTP request to dashboard API using requests library."""
    url = f"{DASHBOARD_URL}/api{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.request(method, url, json=data, timeout=30)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        log(f"HTTP Error {e.response.status_code}: {e}", "ERROR")
        return None
    except requests.exceptions.ConnectionError as e:
        log(f"Connection Error: {e}", "ERROR")
        return None
    except requests.exceptions.Timeout as e:
        log(f"Timeout Error: {e}", "ERROR")
        return None
    except Exception as e:
        log(f"Request error: {e}", "ERROR")
        return None


def compute_video_hash(video_path: Path) -> str:
    """Compute SHA256 hash of video file for caching."""
    sha256 = hashlib.sha256()
    with open(video_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_cache_key(video_hash: str, mode: str, model_config: Dict[str, str]) -> str:
    """Generate cache key from video hash + mode + model config."""
    config_str = json.dumps(model_config, sort_keys=True)
    combined = f"{video_hash}:{mode}:{config_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def check_cache(cache_key: str) -> Optional[Dict]:
    """Check if results exist in cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except:
            pass
    return None


def save_to_cache(cache_key: str, results: Dict):
    """Save results to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, "w") as f:
        json.dump(results, f)
    log(f"Saved results to cache: {cache_key}")


def download_video(video_url: str, video_id: str) -> Optional[Path]:
    """Download video from URL to local input directory."""
    output_path = INPUT_DIR / f"{video_id}.mp4"
    
    if output_path.exists():
        log(f"Video already exists: {output_path}")
        return output_path
    
    # Handle relative URLs from local storage mode
    # Local storage returns paths like /uploads/videos/... without a scheme
    if video_url.startswith('/') and not video_url.startswith('//'):
        video_url = f"{DASHBOARD_URL}{video_url}"
        log(f"Resolved relative URL to: {video_url}")
    
    try:
        log(f"Downloading video: {video_url}")
        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        log(f"Downloaded to: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return output_path
    except Exception as e:
        log(f"Failed to download video: {e}", "ERROR")
        return None


# Track processing start time for ETA calculation
_processing_start_time: Optional[float] = None
_processing_total_frames: int = 0


def update_analysis_status(analysis_id: str, status: str, stage: str = "", progress: int = 0, error: str = "", eta: Optional[int] = None):
    """Update analysis status in dashboard with optional ETA."""
    data = {
        "status": status,
        "currentStage": stage,
        "progress": progress,
    }
    if error:
        data["error"] = error
    if eta is not None:
        data["eta"] = eta
    
    api_request(f"/worker/analysis/{analysis_id}/status", "POST", data)


# Map dashboard mode names to pipeline Mode enum values
MODE_MAPPING = {
    "all": "ALL",
    "radar": "RADAR",
    "team": "TEAM_CLASSIFICATION",
    "track": "PLAYER_TRACKING",
    "players": "PLAYER_DETECTION",
    "ball": "BALL_DETECTION",
    "pitch": "PITCH_DETECTION",
}


def run_pipeline(video_path: Path, analysis_id: str, mode: str, model_config: Dict[str, str]) -> Dict:
    """Run the CV pipeline on a video."""
    global _processing_start_time
    
    output_video = OUTPUT_DIR / f"{analysis_id}_annotated.mp4"
    radar_video = OUTPUT_DIR / f"{analysis_id}_radar.mp4"
    
    # Record start time for ETA calculation
    _processing_start_time = time.time()
    
    # Map dashboard mode to pipeline mode
    pipeline_mode = MODE_MAPPING.get(mode, mode.upper())
    log(f"Mode mapping: {mode} -> {pipeline_mode}")
    
    # Build command - use run_pipeline.py wrapper for correct imports
    pipeline_dir = Path(__file__).parent
    run_script = pipeline_dir / "run_pipeline.py"
    
    # Detect GPU
    device = "cuda" if os.path.exists("/dev/nvidia0") or os.path.exists("/dev/nvidia-uvm") else "cpu"
    
    cmd = [
        sys.executable, str(run_script),
        "--source-video-path", str(video_path),
        "--target-video-path", str(output_video),
        "--mode", pipeline_mode,
        "--device", device,
    ]
    
    # Always use custom models when they exist (fine-tuned models are better)
    # Note: --player-model and --pitch-model expect 'custom' or 'yolov8'/'roboflow', not file paths
    # The pipeline uses config.py to find the actual model files
    player_model = MODELS_DIR / "player_detection.pt"
    if player_model.exists():
        cmd.extend(["--player-model", "custom"])
        log(f"Using custom player model: {player_model}")
    
    ball_model = MODELS_DIR / "ball_detection.pt"
    if ball_model.exists():
        cmd.extend(["--ball-model-source", "custom"])
        log(f"Using custom ball model: {ball_model}")
    
    pitch_model = MODELS_DIR / "pitch_detection.pt"
    if pitch_model.exists():
        cmd.extend(["--pitch-model", "custom"])
        log(f"Using custom pitch model: {pitch_model}")
    
    log(f"Running pipeline: {' '.join(cmd)}")
    
    # Set up environment - ensure src/ is on PYTHONPATH for subprocess
    env = os.environ.copy()
    src_dir = str(pipeline_dir / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{existing_pythonpath}" if existing_pythonpath else src_dir
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(pipeline_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output and update progress
        stages = ["detecting", "tracking", "classifying", "mapping", "computing", "rendering"]
        stage_weights = [30, 20, 15, 10, 10, 15]  # Approximate time weights per stage
        current_stage_idx = 0
        
        for line in process.stdout:
            line = line.strip()
            if line:
                log(f"Pipeline: {line}")
                
                # Detect stage changes from output
                for i, stage in enumerate(stages):
                    if stage.lower() in line.lower():
                        current_stage_idx = i
                        # Calculate weighted progress
                        progress = sum(stage_weights[:i]) + stage_weights[i] // 2
                        progress = min(progress, 95)  # Cap at 95% until complete
                        
                        # Calculate ETA based on elapsed time and progress
                        eta = None
                        if _processing_start_time and progress > 0:
                            elapsed = time.time() - _processing_start_time
                            estimated_total = elapsed / (progress / 100)
                            eta = int(estimated_total - elapsed)
                        
                        update_analysis_status(analysis_id, "processing", stage, progress, eta=eta)
                        break
        
        process.wait()
        
        if process.returncode != 0:
            raise Exception(f"Pipeline exited with code {process.returncode}")
        
    except Exception as e:
        log(f"Pipeline error: {e}", "ERROR")
        return {"success": False, "error": str(e)}
    
    # Collect results
    results = {
        "success": True,
        "annotatedVideo": str(output_video) if output_video.exists() else None,
        "radarVideo": str(radar_video) if radar_video.exists() else None,
        "tracks": None,
        "analytics": None,
    }
    
    # Load tracking data if available
    video_name = video_path.stem
    tracks_file = STUBS_DIR / video_name / "tracks.pkl"
    analytics_file = OUTPUT_DIR / f"{video_name}_analytics.json"
    
    if analytics_file.exists():
        with open(analytics_file, "r") as f:
            results["analytics"] = json.load(f)
    
    return results


def upload_video_to_s3(video_path: Path, analysis_id: str) -> Optional[str]:
    """Upload video to S3 via dashboard API and return the URL."""
    if not video_path.exists():
        log(f"Video file not found: {video_path}", "ERROR")
        return None
    
    try:
        import base64
        log(f"Uploading video to S3: {video_path.name}")
        
        # Read and encode video
        with open(video_path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Upload via API
        response = api_request("/worker/upload-video", "POST", {
            "videoData": video_data,
            "fileName": video_path.name,
            "contentType": "video/mp4"
        })
        
        if response and response.get("success"):
            url = response.get("url")
            log(f"Video uploaded successfully: {url}")
            return url
        else:
            log(f"Failed to upload video: {response}", "ERROR")
            return None
    except Exception as e:
        log(f"Error uploading video: {e}", "ERROR")
        return None


def process_pending_analysis(analysis: Dict) -> bool:
    """Process a single pending analysis."""
    analysis_id = analysis.get("id")
    video_url = analysis.get("videoUrl")
    video_id = analysis.get("videoId")
    mode = analysis.get("mode", "all")
    model_config = analysis.get("modelConfig", {})
    
    log(f"Processing analysis {analysis_id} (mode: {mode})")
    
    # Update status to processing
    update_analysis_status(analysis_id, "processing", "downloading", 5)
    
    # Download video
    video_path = download_video(video_url, video_id)
    if not video_path:
        update_analysis_status(analysis_id, "failed", error="Failed to download video")
        return False
    
    update_analysis_status(analysis_id, "processing", "loading", 10)
    
    # Check cache
    video_hash = compute_video_hash(video_path)
    cache_key = get_cache_key(video_hash, mode, model_config)
    
    cached_results = check_cache(cache_key)
    if cached_results:
        log(f"Cache hit! Using cached results for {cache_key}")
        # Post cached results
        api_request(f"/worker/analysis/{analysis_id}/complete", "POST", cached_results)
        return True
    
    # Run pipeline
    results = run_pipeline(video_path, analysis_id, mode, model_config)
    
    if results.get("success"):
        # Upload annotated video to S3
        update_analysis_status(analysis_id, "processing", "uploading", 95)
        
        annotated_video_path = results.get("annotatedVideo")
        if annotated_video_path:
            annotated_url = upload_video_to_s3(Path(annotated_video_path), analysis_id)
            if annotated_url:
                results["annotatedVideo"] = annotated_url
            else:
                log("Failed to upload annotated video", "WARN")
        
        radar_video_path = results.get("radarVideo")
        if radar_video_path:
            radar_url = upload_video_to_s3(Path(radar_video_path), f"{analysis_id}_radar")
            if radar_url:
                results["radarVideo"] = radar_url
        
        # Save to cache
        save_to_cache(cache_key, results)
        
        # Post results to dashboard
        update_analysis_status(analysis_id, "completed", "done", 100)
        api_request(f"/worker/analysis/{analysis_id}/complete", "POST", results)
        log(f"Analysis {analysis_id} completed successfully")
        return True
    else:
        update_analysis_status(analysis_id, "failed", error=results.get("error", "Unknown error"))
        log(f"Analysis {analysis_id} failed: {results.get('error')}", "ERROR")
        return False


def poll_for_work():
    """Poll dashboard for pending analyses."""
    response = api_request("/worker/pending")
    if response and response.get("analyses"):
        return response["analyses"]
    return []


def main():
    """Main worker loop."""
    log("=" * 60)
    log("Football Analysis Worker Service")
    log("=" * 60)
    log(f"Dashboard URL: {DASHBOARD_URL}")
    log(f"Poll interval: {POLL_INTERVAL}s")
    log(f"Models dir: {MODELS_DIR}")
    log(f"Cache dir: {CACHE_DIR}")
    log("")
    
    # Download models if not present
    download_models()
    
    # Check for models
    models = list(MODELS_DIR.glob("*.pt"))
    if models:
        log(f"Found models: {[m.name for m in models]}")
    else:
        log("No custom models found - will use pretrained models", "WARN")
    
    log("")
    log("Starting worker loop...")
    log("")
    
    while True:
        try:
            # Poll for pending work
            pending = poll_for_work()
            
            if pending:
                log(f"Found {len(pending)} pending analyses")
                for analysis in pending:
                    process_pending_analysis(analysis)
            
            # Wait before next poll
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            log("Shutting down worker...")
            break
        except Exception as e:
            log(f"Worker error: {e}", "ERROR")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
