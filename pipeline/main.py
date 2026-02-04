"""
Football Analysis Pipeline Service

FastAPI service that processes uploaded videos through the CV pipeline:
- YOLOv8 player/ball/goalkeeper detection
- ByteTrack object tracking with ID persistence
- SigLIP + UMAP + KMeans team classification
- Pitch keypoint detection and homography
- Analytics computation and event detection
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Pipeline imports (will be available when running with proper env)
try:
    from processor import FootballProcessor, ProcessingConfig
except ImportError:
    FootballProcessor = None
    ProcessingConfig = None

app = FastAPI(
    title="Football Analysis Pipeline",
    description="CV pipeline for football match analysis",
    version="1.0.0"
)

# CORS for dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Models
# ============================================================================

class PipelineMode(str, Enum):
    ALL = "all"
    RADAR = "radar"
    TEAM = "team"
    TRACK = "track"
    PLAYERS = "players"
    BALL = "ball"
    PITCH = "pitch"


class ProcessingStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    LOADING = "loading"
    DETECTION = "detection"
    TRACKING = "tracking"
    TEAM_CLASSIFICATION = "team_classification"
    PITCH_DETECTION = "pitch_detection"
    HOMOGRAPHY = "homography"
    ANALYTICS = "analytics"
    RENDERING = "rendering"
    COMPLETE = "complete"


class ProcessingRequest(BaseModel):
    video_url: str
    analysis_id: str
    mode: PipelineMode = PipelineMode.ALL
    use_custom_models: bool = True
    callback_url: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class ProcessingProgress(BaseModel):
    analysis_id: str
    status: ProcessingStatus
    stage: ProcessingStage
    progress: float  # 0-100
    message: str
    current_frame: Optional[int] = None
    total_frames: Optional[int] = None


class ProcessingResult(BaseModel):
    analysis_id: str
    status: ProcessingStatus
    tracks: Optional[Dict[str, Any]] = None
    analytics: Optional[Dict[str, Any]] = None
    events: Optional[List[Dict[str, Any]]] = None
    output_video_url: Optional[str] = None
    radar_video_url: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# In-memory job tracking
# ============================================================================

jobs: Dict[str, ProcessingProgress] = {}
results: Dict[str, ProcessingResult] = {}


# ============================================================================
# Pipeline Processing
# ============================================================================

async def update_progress(
    analysis_id: str,
    status: ProcessingStatus,
    stage: ProcessingStage,
    progress: float,
    message: str,
    current_frame: Optional[int] = None,
    total_frames: Optional[int] = None,
    callback_url: Optional[str] = None
):
    """Update job progress and optionally notify callback."""
    jobs[analysis_id] = ProcessingProgress(
        analysis_id=analysis_id,
        status=status,
        stage=stage,
        progress=progress,
        message=message,
        current_frame=current_frame,
        total_frames=total_frames,
    )
    
    # Send callback if URL provided
    if callback_url:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    callback_url,
                    json=jobs[analysis_id].model_dump(),
                    timeout=10.0
                )
        except Exception as e:
            print(f"Callback failed: {e}")


async def process_video(request: ProcessingRequest):
    """Main video processing pipeline."""
    analysis_id = request.analysis_id
    
    try:
        # Stage 1: Loading video
        await update_progress(
            analysis_id, ProcessingStatus.PROCESSING, ProcessingStage.LOADING,
            5, "Loading video...", callback_url=request.callback_url
        )
        
        if FootballProcessor is None:
            raise ImportError("Pipeline processor not available. Install dependencies.")
        
        # Initialize processor
        config = ProcessingConfig(
            mode=request.mode.value,
            use_custom_models=request.use_custom_models,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        )
        processor = FootballProcessor(config)
        
        # Download video
        video_path = await download_video(request.video_url, analysis_id)
        frames = processor.load_video(video_path)
        total_frames = len(frames)
        
        await update_progress(
            analysis_id, ProcessingStatus.PROCESSING, ProcessingStage.LOADING,
            10, f"Loaded {total_frames} frames", total_frames=total_frames,
            callback_url=request.callback_url
        )
        
        # Stage 2: Detection
        await update_progress(
            analysis_id, ProcessingStatus.PROCESSING, ProcessingStage.DETECTION,
            15, "Running object detection...", callback_url=request.callback_url
        )
        
        detections = processor.detect(frames, progress_callback=lambda p: asyncio.create_task(
            update_progress(
                analysis_id, ProcessingStatus.PROCESSING, ProcessingStage.DETECTION,
                15 + p * 0.2, f"Detection: {int(p*100)}%", callback_url=request.callback_url
            )
        ))
        
        # Stage 3: Tracking
        await update_progress(
            analysis_id, ProcessingStatus.PROCESSING, ProcessingStage.TRACKING,
            35, "Tracking objects...", callback_url=request.callback_url
        )
        
        tracks = processor.track(frames, detections)
        
        # Stage 4: Team Classification (if applicable)
        if request.mode in [PipelineMode.ALL, PipelineMode.TEAM, PipelineMode.RADAR]:
            await update_progress(
                analysis_id, ProcessingStatus.PROCESSING, ProcessingStage.TEAM_CLASSIFICATION,
                50, "Classifying teams...", callback_url=request.callback_url
            )
            tracks = processor.classify_teams(frames, tracks)
        
        # Stage 5: Pitch Detection
        if request.mode in [PipelineMode.ALL, PipelineMode.PITCH, PipelineMode.RADAR]:
            await update_progress(
                analysis_id, ProcessingStatus.PROCESSING, ProcessingStage.PITCH_DETECTION,
                60, "Detecting pitch keypoints...", callback_url=request.callback_url
            )
            pitch_data = processor.detect_pitch(frames)
        else:
            pitch_data = None
        
        # Stage 6: Homography
        if pitch_data and request.mode in [PipelineMode.ALL, PipelineMode.RADAR]:
            await update_progress(
                analysis_id, ProcessingStatus.PROCESSING, ProcessingStage.HOMOGRAPHY,
                70, "Computing homography...", callback_url=request.callback_url
            )
            transformers = processor.compute_homography(pitch_data)
        else:
            transformers = None
        
        # Stage 7: Analytics
        await update_progress(
            analysis_id, ProcessingStatus.PROCESSING, ProcessingStage.ANALYTICS,
            80, "Computing analytics...", callback_url=request.callback_url
        )
        
        analytics = processor.compute_analytics(tracks, transformers)
        events = processor.detect_events(tracks, analytics)
        
        # Stage 8: Rendering
        await update_progress(
            analysis_id, ProcessingStatus.PROCESSING, ProcessingStage.RENDERING,
            90, "Rendering output videos...", callback_url=request.callback_url
        )
        
        output_urls = processor.render_outputs(
            frames, tracks, pitch_data, transformers, analysis_id
        )
        
        # Complete
        await update_progress(
            analysis_id, ProcessingStatus.COMPLETED, ProcessingStage.COMPLETE,
            100, "Processing complete!", callback_url=request.callback_url
        )
        
        results[analysis_id] = ProcessingResult(
            analysis_id=analysis_id,
            status=ProcessingStatus.COMPLETED,
            tracks=tracks,
            analytics=analytics,
            events=events,
            output_video_url=output_urls.get("annotated"),
            radar_video_url=output_urls.get("radar"),
        )
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Processing failed: {error_msg}")
        
        await update_progress(
            analysis_id, ProcessingStatus.FAILED, ProcessingStage.COMPLETE,
            0, f"Processing failed: {str(e)}", callback_url=request.callback_url
        )
        
        results[analysis_id] = ProcessingResult(
            analysis_id=analysis_id,
            status=ProcessingStatus.FAILED,
            error=error_msg,
        )


async def download_video(url: str, analysis_id: str) -> str:
    """Download video from URL to local path."""
    output_dir = Path(f"/tmp/football_analysis/{analysis_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "input.mp4"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)
    
    return str(output_path)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_available": FootballProcessor is not None,
    }


@app.post("/process", response_model=ProcessingProgress)
async def start_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Start video processing job."""
    analysis_id = request.analysis_id
    
    # Initialize job
    jobs[analysis_id] = ProcessingProgress(
        analysis_id=analysis_id,
        status=ProcessingStatus.QUEUED,
        stage=ProcessingStage.LOADING,
        progress=0,
        message="Job queued",
    )
    
    # Start background processing
    background_tasks.add_task(process_video, request)
    
    return jobs[analysis_id]


@app.get("/status/{analysis_id}", response_model=ProcessingProgress)
async def get_status(analysis_id: str):
    """Get processing status for a job."""
    if analysis_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[analysis_id]


@app.get("/result/{analysis_id}", response_model=ProcessingResult)
async def get_result(analysis_id: str):
    """Get processing result for a completed job."""
    if analysis_id not in results:
        if analysis_id in jobs:
            return ProcessingResult(
                analysis_id=analysis_id,
                status=jobs[analysis_id].status,
            )
        raise HTTPException(status_code=404, detail="Job not found")
    return results[analysis_id]


@app.delete("/job/{analysis_id}")
async def cancel_job(analysis_id: str):
    """Cancel a processing job."""
    if analysis_id in jobs:
        jobs[analysis_id].status = ProcessingStatus.FAILED
        jobs[analysis_id].message = "Job cancelled"
    return {"status": "cancelled"}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PIPELINE_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
