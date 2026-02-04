"""
Football Analysis Pipeline - FastAPI Server

A clean, professional API for the computer vision pipeline.
Handles video processing, progress tracking, and result delivery.
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = Path(__file__).parent.parent / "models"
INPUT_DIR = Path(__file__).parent.parent / "input_videos"
OUTPUT_DIR = Path(__file__).parent.parent / "output_videos"
STUBS_DIR = Path(__file__).parent.parent / "stubs"

# Ensure directories exist
for dir_path in [MODELS_DIR, INPUT_DIR, OUTPUT_DIR, STUBS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Models
# =============================================================================

class PipelineMode(str, Enum):
    ALL = "all"
    RADAR = "radar"
    TEAM = "team"
    TRACK = "track"
    PLAYERS = "players"
    BALL = "ball"
    PITCH = "pitch"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    UPLOADING = "uploading"
    DETECTING = "detecting"
    TRACKING = "tracking"
    CLASSIFYING = "classifying"
    MAPPING = "mapping"
    COMPUTING = "computing"
    RENDERING = "rendering"


class VideoProcessRequest(BaseModel):
    mode: PipelineMode = PipelineMode.ALL
    use_custom_models: bool = True
    callback_url: Optional[str] = None


class ProcessingProgress(BaseModel):
    job_id: str
    status: ProcessingStatus
    stage: Optional[ProcessingStage] = None
    progress: int = 0
    eta_seconds: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ProcessingResult(BaseModel):
    job_id: str
    status: ProcessingStatus
    annotated_video_url: Optional[str] = None
    radar_video_url: Optional[str] = None
    analytics_json: Optional[Dict[str, Any]] = None
    tracks_json: Optional[Dict[str, Any]] = None
    processing_time_seconds: Optional[float] = None


# =============================================================================
# Application State
# =============================================================================

class JobManager:
    """Manages processing jobs and their state."""
    
    def __init__(self):
        self.jobs: Dict[str, ProcessingProgress] = {}
        self.results: Dict[str, ProcessingResult] = {}
        self.websocket_clients: Dict[str, List[WebSocket]] = {}
    
    def create_job(self, job_id: str) -> ProcessingProgress:
        """Create a new processing job."""
        job = ProcessingProgress(
            job_id=job_id,
            status=ProcessingStatus.PENDING,
            started_at=datetime.now()
        )
        self.jobs[job_id] = job
        return job
    
    def update_job(
        self, 
        job_id: str, 
        status: Optional[ProcessingStatus] = None,
        stage: Optional[ProcessingStage] = None,
        progress: Optional[int] = None,
        eta_seconds: Optional[int] = None,
        error: Optional[str] = None
    ):
        """Update job progress."""
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        if status:
            job.status = status
        if stage:
            job.stage = stage
        if progress is not None:
            job.progress = progress
        if eta_seconds is not None:
            job.eta_seconds = eta_seconds
        if error:
            job.error = error
        if status == ProcessingStatus.COMPLETED:
            job.completed_at = datetime.now()
        
        # Broadcast to WebSocket clients
        asyncio.create_task(self.broadcast_progress(job_id))
    
    def set_result(self, job_id: str, result: ProcessingResult):
        """Store processing result."""
        self.results[job_id] = result
    
    async def broadcast_progress(self, job_id: str):
        """Send progress update to all WebSocket clients subscribed to this job."""
        if job_id not in self.websocket_clients:
            return
        
        job = self.jobs.get(job_id)
        if not job:
            return
        
        message = {
            "type": "progress",
            "job_id": job_id,
            "data": job.model_dump(mode="json")
        }
        
        dead_clients = []
        for ws in self.websocket_clients[job_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead_clients.append(ws)
        
        # Remove dead clients
        for ws in dead_clients:
            self.websocket_clients[job_id].remove(ws)
    
    def subscribe(self, job_id: str, websocket: WebSocket):
        """Subscribe a WebSocket client to job updates."""
        if job_id not in self.websocket_clients:
            self.websocket_clients[job_id] = []
        self.websocket_clients[job_id].append(websocket)
    
    def unsubscribe(self, job_id: str, websocket: WebSocket):
        """Unsubscribe a WebSocket client from job updates."""
        if job_id in self.websocket_clients:
            try:
                self.websocket_clients[job_id].remove(websocket)
            except ValueError:
                pass


# Global job manager
job_manager = JobManager()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Football Analysis Pipeline API",
    description="Computer vision pipeline for football match analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================

def compute_video_hash(file_path: Path) -> str:
    """Compute SHA256 hash of video file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def get_cache_key(video_hash: str, mode: str, use_custom_models: bool) -> str:
    """Generate cache key for processed results."""
    return f"{video_hash}_{mode}_{'custom' if use_custom_models else 'pretrained'}"


async def run_pipeline(
    job_id: str,
    video_path: Path,
    mode: PipelineMode,
    use_custom_models: bool,
    callback_url: Optional[str] = None
):
    """Run the CV pipeline in background."""
    start_time = time.time()
    
    try:
        job_manager.update_job(job_id, status=ProcessingStatus.PROCESSING)
        
        # Import pipeline modules
        from config import get_config
        from pipeline.base import BasePipeline
        
        # Configure pipeline
        config = get_config()
        config.mode = mode.value
        
        # Set model paths
        if use_custom_models:
            player_model = MODELS_DIR / "player_detection.pt"
            ball_model = MODELS_DIR / "ball_detection.pt"
            pitch_model = MODELS_DIR / "pitch_detection.pt"
            
            if player_model.exists():
                config.player_model_path = str(player_model)
            if ball_model.exists():
                config.ball_model_path = str(ball_model)
            if pitch_model.exists():
                config.pitch_model_path = str(pitch_model)
        
        # Output paths
        output_video = OUTPUT_DIR / f"{job_id}_annotated.mp4"
        radar_video = OUTPUT_DIR / f"{job_id}_radar.mp4"
        analytics_json = OUTPUT_DIR / f"{job_id}_analytics.json"
        
        # Stage weights for progress calculation
        stage_weights = {
            ProcessingStage.DETECTING: 30,
            ProcessingStage.TRACKING: 20,
            ProcessingStage.CLASSIFYING: 15,
            ProcessingStage.MAPPING: 10,
            ProcessingStage.COMPUTING: 10,
            ProcessingStage.RENDERING: 15,
        }
        
        def update_progress(stage: ProcessingStage, stage_progress: float = 0.5):
            """Update progress based on current stage."""
            stages = list(stage_weights.keys())
            stage_idx = stages.index(stage)
            
            # Calculate cumulative progress
            progress = sum(stage_weights[s] for s in stages[:stage_idx])
            progress += int(stage_weights[stage] * stage_progress)
            progress = min(progress, 95)
            
            # Calculate ETA
            elapsed = time.time() - start_time
            if progress > 0:
                eta = int((elapsed / progress) * (100 - progress))
            else:
                eta = None
            
            job_manager.update_job(
                job_id,
                stage=stage,
                progress=progress,
                eta_seconds=eta
            )
        
        # Run pipeline stages
        update_progress(ProcessingStage.DETECTING, 0)
        # ... pipeline execution would go here ...
        # For now, simulate progress
        
        for stage in stage_weights.keys():
            update_progress(stage, 0.5)
            await asyncio.sleep(0.5)  # Simulate work
            update_progress(stage, 1.0)
        
        # Mark as completed
        processing_time = time.time() - start_time
        
        result = ProcessingResult(
            job_id=job_id,
            status=ProcessingStatus.COMPLETED,
            annotated_video_url=f"/api/videos/{job_id}/annotated",
            radar_video_url=f"/api/videos/{job_id}/radar",
            processing_time_seconds=processing_time
        )
        
        job_manager.set_result(job_id, result)
        job_manager.update_job(job_id, status=ProcessingStatus.COMPLETED, progress=100)
        
        # Callback if provided
        if callback_url:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(callback_url, json=result.model_dump(mode="json"))
        
    except Exception as e:
        job_manager.update_job(
            job_id,
            status=ProcessingStatus.FAILED,
            error=str(e)
        )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Football Analysis Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "player_detection": (MODELS_DIR / "player_detection.pt").exists(),
            "ball_detection": (MODELS_DIR / "ball_detection.pt").exists(),
            "pitch_detection": (MODELS_DIR / "pitch_detection.pt").exists(),
        }
    }


@app.post("/api/process", response_model=ProcessingProgress)
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: PipelineMode = PipelineMode.ALL,
    use_custom_models: bool = True,
    callback_url: Optional[str] = None
):
    """
    Upload and process a video file.
    
    Returns a job ID for tracking progress via WebSocket or polling.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    # Save uploaded file
    job_id = f"{int(time.time())}_{hashlib.md5(file.filename.encode()).hexdigest()[:8]}"
    video_path = INPUT_DIR / f"{job_id}.mp4"
    
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create job
    job = job_manager.create_job(job_id)
    
    # Start processing in background
    background_tasks.add_task(
        run_pipeline,
        job_id,
        video_path,
        mode,
        use_custom_models,
        callback_url
    )
    
    return job


@app.get("/api/jobs/{job_id}", response_model=ProcessingProgress)
async def get_job_status(job_id: str):
    """Get the current status of a processing job."""
    if job_id not in job_manager.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_manager.jobs[job_id]


@app.get("/api/jobs/{job_id}/result", response_model=ProcessingResult)
async def get_job_result(job_id: str):
    """Get the result of a completed processing job."""
    if job_id not in job_manager.results:
        if job_id in job_manager.jobs:
            job = job_manager.jobs[job_id]
            if job.status != ProcessingStatus.COMPLETED:
                raise HTTPException(status_code=202, detail="Job still processing")
        raise HTTPException(status_code=404, detail="Result not found")
    return job_manager.results[job_id]


@app.get("/api/videos/{job_id}/annotated")
async def get_annotated_video(job_id: str):
    """Download the annotated video."""
    video_path = OUTPUT_DIR / f"{job_id}_annotated.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


@app.get("/api/videos/{job_id}/radar")
async def get_radar_video(job_id: str):
    """Download the radar visualization video."""
    video_path = OUTPUT_DIR / f"{job_id}_radar.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time progress updates.
    
    Connect to receive live updates for a specific job.
    """
    await websocket.accept()
    job_manager.subscribe(job_id, websocket)
    
    try:
        # Send current status immediately
        if job_id in job_manager.jobs:
            await websocket.send_json({
                "type": "connected",
                "job_id": job_id,
                "data": job_manager.jobs[job_id].model_dump(mode="json")
            })
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                # Handle ping/pong or other messages
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
    finally:
        job_manager.unsubscribe(job_id, websocket)


@app.get("/api/modes")
async def list_modes():
    """List available pipeline modes."""
    return {
        "modes": [
            {"id": "all", "name": "Full Analysis", "description": "Complete pipeline with all features"},
            {"id": "radar", "name": "Radar Only", "description": "2D pitch visualization only"},
            {"id": "team", "name": "Team Classification", "description": "Player detection + team assignment"},
            {"id": "track", "name": "Tracking Only", "description": "Object detection and tracking"},
            {"id": "players", "name": "Players Only", "description": "Player detection only"},
            {"id": "ball", "name": "Ball Only", "description": "Ball detection and tracking"},
            {"id": "pitch", "name": "Pitch Only", "description": "Pitch keypoint detection"},
        ]
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PIPELINE_PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
