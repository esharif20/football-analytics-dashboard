"""
Analysis Router - Video analysis management and processing
Simplified - no authentication required, with stub caching
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from enum import Enum

from api.services.database import (
    get_video_by_id, create_analysis, get_analysis_by_id,
    get_all_analyses, get_analyses_by_video, get_analysis_by_video_and_mode,
    update_analysis_status, update_analysis_results,
    get_all_stubs_for_video, save_stub
)
from api.services.websocket_manager import manager

router = APIRouter()

class PipelineMode(str, Enum):
    all = "all"
    radar = "radar"
    track = "track"
    team = "team"
    players = "players"
    ball = "ball"
    pitch = "pitch"

class ProcessingStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

# Pipeline modes matching original repo
PIPELINE_MODES = [
    {"id": "all", "name": "Full Analysis", "description": "Complete pipeline: detection, tracking, team assignment, analytics"},
    {"id": "radar", "name": "Radar View", "description": "2D pitch visualization with player positions"},
    {"id": "track", "name": "Tracking Only", "description": "Player and ball tracking without team assignment"},
    {"id": "team", "name": "Team Assignment", "description": "Tracking with team color classification"},
    {"id": "players", "name": "Players Only", "description": "Player detection and tracking only"},
    {"id": "ball", "name": "Ball Only", "description": "Ball detection and interpolation"},
    {"id": "pitch", "name": "Pitch Keypoints", "description": "Pitch keypoint detection for homography"},
]

PROCESSING_STAGES = [
    {"id": "loading", "name": "Loading Video", "weight": 5},
    {"id": "detecting", "name": "Detecting Objects", "weight": 30},
    {"id": "tracking", "name": "Tracking", "weight": 20},
    {"id": "classifying", "name": "Team Classification", "weight": 15},
    {"id": "mapping", "name": "Pitch Mapping", "weight": 10},
    {"id": "rendering", "name": "Rendering Output", "weight": 20},
]

class AnalysisCreate(BaseModel):
    videoId: int
    mode: PipelineMode

class AnalysisStatusUpdate(BaseModel):
    status: ProcessingStatus
    progress: int
    currentStage: Optional[str] = None
    errorMessage: Optional[str] = None

@router.get("")
async def list_analyses():
    """List all analyses"""
    return get_all_analyses()

@router.get("/modes")
async def get_modes():
    """Get available pipeline modes"""
    return PIPELINE_MODES

@router.get("/stages")
async def get_stages():
    """Get processing stages"""
    return PROCESSING_STAGES

@router.get("/by-video/{video_id}")
async def list_by_video(video_id: int):
    """List analyses for a specific video"""
    video = get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return get_analyses_by_video(video_id)

@router.get("/{analysis_id}")
async def get_analysis(analysis_id: int):
    """Get analysis by ID"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@router.post("")
async def create_new_analysis(data: AnalysisCreate, background_tasks: BackgroundTasks):
    """Create a new analysis - checks for cached results first"""
    video = get_video_by_id(data.videoId)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check if we already have a completed analysis for this video+mode
    existing = get_analysis_by_video_and_mode(data.videoId, data.mode.value)
    if existing:
        return {
            "id": existing["id"],
            "cached": True,
            "message": "Using cached analysis results"
        }
    
    # Create new analysis
    analysis_id = create_analysis(
        video_id=data.videoId,
        mode=data.mode.value
    )
    
    # Start processing in background
    background_tasks.add_task(process_video, analysis_id, video, data.mode.value)
    
    return {"id": analysis_id, "cached": False}

@router.put("/{analysis_id}/status")
async def update_status(analysis_id: int, data: AnalysisStatusUpdate):
    """Update analysis status"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    update_analysis_status(
        analysis_id,
        data.status.value,
        data.progress,
        data.currentStage,
        data.errorMessage
    )
    
    # Broadcast progress via WebSocket
    await manager.broadcast_analysis_progress(
        analysis_id,
        data.status.value,
        data.progress,
        data.currentStage
    )
    
    return {"success": True}

@router.post("/{analysis_id}/terminate")
async def terminate_analysis(analysis_id: int):
    """Terminate a running analysis"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if analysis["status"] not in ["processing", "pending"]:
        raise HTTPException(status_code=400, detail="Analysis is not running")
    
    update_analysis_status(
        analysis_id,
        "failed",
        analysis["progress"],
        analysis.get("current_stage"),
        "Terminated by user"
    )
    
    await manager.broadcast_analysis_error(analysis_id, "Terminated by user")
    
    return {"success": True}

@router.get("/{analysis_id}/eta")
async def get_eta(analysis_id: int):
    """Get ETA for processing"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    from datetime import datetime
    
    created_at = datetime.fromisoformat(analysis["created_at"]) if analysis["created_at"] else datetime.now()
    elapsed = (datetime.now() - created_at).total_seconds() * 1000
    progress = analysis["progress"] or 1
    
    estimated_total = (elapsed / progress) * 100
    remaining = max(0, estimated_total - elapsed)
    
    current_stage = analysis.get("current_stage") or "loading"
    stage_index = next((i for i, s in enumerate(PROCESSING_STAGES) if s["id"] == current_stage), 0)
    
    return {
        "elapsedMs": int(elapsed),
        "remainingMs": int(remaining),
        "estimatedTotalMs": int(elapsed + remaining),
        "currentStage": current_stage,
        "stageIndex": stage_index,
        "totalStages": len(PROCESSING_STAGES)
    }

async def process_video(analysis_id: int, video: dict, mode: str):
    """Background task to process video through CV pipeline"""
    import asyncio
    import os
    import sys
    import time
    
    start_time = time.time()
    
    try:
        # Update status to processing
        update_analysis_status(analysis_id, "processing", 0, "loading")
        await manager.broadcast_analysis_progress(analysis_id, "processing", 0, "loading")
        
        # Get video file path
        video_path = video.get("file_path", "")
        if not video_path or not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")
        
        # Check for cached stubs
        video_hash = video.get("file_hash", "")
        cached_stubs = get_all_stubs_for_video(video_hash, mode) if video_hash else {}
        
        if cached_stubs:
            # Use cached results
            update_analysis_status(analysis_id, "processing", 50, "loading_cache")
            await manager.broadcast_analysis_progress(analysis_id, "processing", 50, "loading_cache")
            await asyncio.sleep(0.5)
        
        # Import and run pipeline
        pipeline_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pipeline")
        sys.path.insert(0, pipeline_path)
        
        try:
            from src.main import run_pipeline
            
            # Create output directory
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "outputs", str(analysis_id))
            os.makedirs(output_dir, exist_ok=True)
            
            # Progress callback
            async def progress_callback(stage: str, progress: int):
                update_analysis_status(analysis_id, "processing", progress, stage)
                await manager.broadcast_analysis_progress(analysis_id, "processing", progress, stage)
            
            # Run pipeline
            result = run_pipeline(
                video_path=video_path,
                mode=mode,
                output_dir=output_dir,
                use_stubs=bool(cached_stubs),
                progress_callback=progress_callback
            )
            
            # Save results
            processing_time = int((time.time() - start_time) * 1000)
            update_analysis_results(
                analysis_id,
                annotated_video_path=result.get("annotated_video"),
                radar_video_path=result.get("radar_video"),
                tracking_data_path=result.get("tracking_data"),
                analytics_data_path=result.get("analytics_data"),
                processing_time_ms=processing_time
            )
            
            # Save stubs for future caching
            if video_hash:
                for stub_type, stub_path in result.get("stubs", {}).items():
                    save_stub(video_hash, mode, stub_type, stub_path)
            
            update_analysis_status(analysis_id, "completed", 100, "completed")
            await manager.broadcast_analysis_complete(analysis_id, {
                "message": "Analysis completed successfully",
                "processingTimeMs": processing_time
            })
            
        except ImportError:
            # Pipeline not available - simulate processing for demo
            stages = [
                ("detecting", 25),
                ("tracking", 50),
                ("classifying", 70),
                ("mapping", 85),
                ("rendering", 100),
            ]
            
            for stage, progress in stages:
                update_analysis_status(analysis_id, "processing", progress, stage)
                await manager.broadcast_analysis_progress(analysis_id, "processing", progress, stage)
                await asyncio.sleep(1)  # Simulate processing time
            
            processing_time = int((time.time() - start_time) * 1000)
            update_analysis_status(analysis_id, "completed", 100, "completed")
            await manager.broadcast_analysis_complete(analysis_id, {
                "message": "Analysis completed (demo mode - pipeline not installed)",
                "processingTimeMs": processing_time
            })
        
    except Exception as e:
        update_analysis_status(analysis_id, "failed", 0, None, str(e))
        await manager.broadcast_analysis_error(analysis_id, str(e))
