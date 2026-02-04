#!/usr/bin/env python3
"""
Football Analysis Worker Service

A background process that:
1. Polls the dashboard API for pending analyses
2. Runs the CV pipeline on new videos
3. Posts results back to the dashboard

No external dependencies - uses only stdlib + pipeline requirements.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

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


def download_models():
    """Download models from CDN if not already present."""
    for model_name, url in MODEL_URLS.items():
        model_path = MODELS_DIR / model_name
        if model_path.exists():
            log(f"Model already exists: {model_name}")
            continue
        
        log(f"Downloading model: {model_name}...")
        try:
            urllib.request.urlretrieve(url, model_path)
            log(f"Downloaded: {model_name} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            log(f"Failed to download {model_name}: {e}", "ERROR")


def log(msg: str, level: str = "INFO"):
    """Simple logging without external dependencies."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Optional[Dict]:
    """Make HTTP request to dashboard API."""
    url = f"{DASHBOARD_URL}/api{endpoint}"
    
    try:
        if data:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method=method
            )
        else:
            req = urllib.request.Request(url, method=method)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        log(f"HTTP Error {e.code}: {e.reason}", "ERROR")
        return None
    except urllib.error.URLError as e:
        log(f"URL Error: {e.reason}", "ERROR")
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
    
    try:
        log(f"Downloading video: {video_url}")
        urllib.request.urlretrieve(video_url, output_path)
        log(f"Downloaded to: {output_path}")
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


def run_pipeline(video_path: Path, analysis_id: str, mode: str, model_config: Dict[str, str]) -> Dict:
    """Run the CV pipeline on a video."""
    global _processing_start_time
    
    output_video = OUTPUT_DIR / f"{analysis_id}_annotated.mp4"
    radar_video = OUTPUT_DIR / f"{analysis_id}_radar.mp4"
    
    # Record start time for ETA calculation
    _processing_start_time = time.time()
    
    # Build command
    cmd = [
        sys.executable, "main.py",
        "--source-video-path", str(video_path),
        "--target-video-path", str(output_video),
        "--mode", mode,
    ]
    
    # Add model paths if custom models selected
    if model_config.get("player") == "custom" and (MODELS_DIR / "player_detection.pt").exists():
        cmd.extend(["--player-model", str(MODELS_DIR / "player_detection.pt")])
    
    if model_config.get("ball") == "custom" and (MODELS_DIR / "ball_detection.pt").exists():
        cmd.extend(["--ball-model-source", "custom"])
    
    if model_config.get("pitch") == "custom" and (MODELS_DIR / "pitch_detection.pt").exists():
        cmd.extend(["--pitch-model", str(MODELS_DIR / "pitch_detection.pt")])
    
    log(f"Running pipeline: {' '.join(cmd)}")
    
    # Run pipeline
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent),
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


def process_pending_analysis(analysis: Dict) -> bool:
    """Process a single pending analysis."""
    analysis_id = analysis.get("id")
    video_url = analysis.get("videoUrl")
    video_id = analysis.get("videoId")
    mode = analysis.get("mode", "all")
    model_config = analysis.get("modelConfig", {})
    
    log(f"Processing analysis {analysis_id} (mode: {mode})")
    
    # Update status to processing
    update_analysis_status(analysis_id, "processing", "uploading", 5)
    
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
