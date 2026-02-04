"""
Analysis Router - Video analysis management and processing
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

from api.routers.auth import require_user
from api.services.database import (
    get_video_by_id, create_analysis, get_analysis_by_id,
    get_analyses_by_user, get_analyses_by_video,
    update_analysis_status, update_analysis_results
)
from api.services.websocket_manager import manager

router = APIRouter()

class PipelineMode(str, Enum):
    full = "full"
    radar = "radar"
    tracking = "tracking"
    broadcast = "broadcast"

class ProcessingStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

PIPELINE_MODES = [
    {"id": "full", "name": "Full Analysis", "description": "Complete pipeline with all features"},
    {"id": "radar", "name": "Radar View", "description": "2D pitch visualization only"},
    {"id": "tracking", "name": "Tracking Only", "description": "Player and ball tracking"},
    {"id": "broadcast", "name": "Broadcast Camera", "description": "For standard TV footage (Coming Soon)"},
]

PROCESSING_STAGES = [
    {"id": "uploading", "name": "Uploading", "weight": 5},
    {"id": "loading", "name": "Loading Video", "weight": 5},
    {"id": "detecting", "name": "Detecting Players", "weight": 25},
    {"id": "tracking", "name": "Tracking Objects", "weight": 15},
    {"id": "classifying", "name": "Classifying Teams", "weight": 15},
    {"id": "mapping", "name": "Mapping to Pitch", "weight": 10},
    {"id": "computing", "name": "Computing Analytics", "weight": 10},
    {"id": "rendering", "name": "Rendering Output", "weight": 15},
]

class AnalysisCreate(BaseModel):
    videoId: int
    mode: PipelineMode

class AnalysisStatusUpdate(BaseModel):
    status: ProcessingStatus
    progress: int
    currentStage: Optional[str] = None
    errorMessage: Optional[str] = None

class AnalysisResultsUpdate(BaseModel):
    annotatedVideoUrl: Optional[str] = None
    radarVideoUrl: Optional[str] = None
    trackingDataUrl: Optional[str] = None
    analyticsDataUrl: Optional[str] = None
    processingTimeMs: Optional[int] = None

@router.get("")
async def list_analyses(request: Request, user: dict = Depends(require_user)):
    """List all analyses for current user"""
    analyses = get_analyses_by_user(user["id"])
    return analyses

@router.get("/modes")
async def get_modes():
    """Get available pipeline modes"""
    return PIPELINE_MODES

@router.get("/stages")
async def get_stages():
    """Get processing stages"""
    return PROCESSING_STAGES

@router.get("/by-video/{video_id}")
async def list_by_video(video_id: int, request: Request, user: dict = Depends(require_user)):
    """List analyses for a specific video"""
    video = get_video_by_id(video_id)
    if not video or video["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Video not found")
    return get_analyses_by_video(video_id)

@router.get("/{analysis_id}")
async def get_analysis(analysis_id: int, request: Request, user: dict = Depends(require_user)):
    """Get analysis by ID"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@router.post("")
async def create_new_analysis(
    data: AnalysisCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    user: dict = Depends(require_user)
):
    """Create a new analysis"""
    video = get_video_by_id(data.videoId)
    if not video or video["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Video not found")
    
    analysis_id = create_analysis(
        video_id=data.videoId,
        user_id=user["id"],
        mode=data.mode.value
    )
    
    # Start processing in background
    background_tasks.add_task(process_video, analysis_id, video, data.mode.value)
    
    return {"id": analysis_id}

@router.put("/{analysis_id}/status")
async def update_status(
    analysis_id: int,
    data: AnalysisStatusUpdate,
    request: Request,
    user: dict = Depends(require_user)
):
    """Update analysis status"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis or analysis["user_id"] != user["id"]:
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

@router.put("/{analysis_id}/results")
async def update_results(
    analysis_id: int,
    data: AnalysisResultsUpdate,
    request: Request,
    user: dict = Depends(require_user)
):
    """Update analysis results"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    update_analysis_results(
        analysis_id,
        data.annotatedVideoUrl,
        data.radarVideoUrl,
        data.trackingDataUrl,
        data.analyticsDataUrl,
        data.processingTimeMs
    )
    
    return {"success": True}

@router.post("/{analysis_id}/terminate")
async def terminate_analysis(
    analysis_id: int,
    request: Request,
    user: dict = Depends(require_user)
):
    """Terminate a running analysis"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis or analysis["user_id"] != user["id"]:
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
async def get_eta(analysis_id: int, request: Request, user: dict = Depends(require_user)):
    """Get ETA for processing"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    from datetime import datetime
    
    # Calculate ETA based on progress and elapsed time
    created_at = datetime.fromisoformat(analysis["created_at"]) if analysis["created_at"] else datetime.now()
    elapsed = (datetime.now() - created_at).total_seconds() * 1000
    progress = analysis["progress"] or 1
    
    estimated_total = (elapsed / progress) * 100
    remaining = max(0, estimated_total - elapsed)
    
    current_stage = analysis.get("current_stage") or "uploading"
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
    
    try:
        # Update status to processing
        update_analysis_status(analysis_id, "processing", 0, "loading")
        await manager.broadcast_analysis_progress(analysis_id, "processing", 0, "loading")
        
        # Get video file path
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "uploads")
        video_path = os.path.join(upload_dir, video.get("file_key", ""))
        
        if not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")
        
        # Import pipeline modules
        pipeline_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pipeline")
        sys.path.insert(0, pipeline_path)
        
        stages = [
            ("detecting", 25),
            ("tracking", 40),
            ("classifying", 55),
            ("mapping", 70),
            ("computing", 85),
            ("rendering", 100),
        ]
        
        for stage, progress in stages:
            update_analysis_status(analysis_id, "processing", progress, stage)
            await manager.broadcast_analysis_progress(analysis_id, "processing", progress, stage)
            await asyncio.sleep(0.5)  # Simulate processing time
        
        # For now, mark as completed (actual pipeline integration would go here)
        update_analysis_status(analysis_id, "completed", 100, "completed")
        await manager.broadcast_analysis_complete(analysis_id, {
            "message": "Analysis completed successfully"
        })
        
    except Exception as e:
        update_analysis_status(analysis_id, "failed", 0, None, str(e))
        await manager.broadcast_analysis_error(analysis_id, str(e))
