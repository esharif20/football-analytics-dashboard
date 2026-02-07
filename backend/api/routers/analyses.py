from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db, get_current_user
from ..models import User, Video, Analysis
from ..schemas import AnalysisCreate, AnalysisStatusUpdate, AnalysisResultsUpdate, _row_to_dict

router = APIRouter(tags=["analyses"])

# Pipeline modes constant (mirrors backend/shared/types.ts)
PIPELINE_MODES = {
    "all": {"id": "all", "name": "Full Analysis", "description": "Complete pipeline: detection, tracking, team classification, pitch mapping, and analytics", "outputs": ["annotated_video", "radar_video", "tracking_data", "analytics"], "icon": "Layers"},
    "radar": {"id": "radar", "name": "Radar View", "description": "2D pitch visualization with player positions and ball trajectory", "outputs": ["radar_video", "tracking_data"], "icon": "Radar"},
    "team": {"id": "team", "name": "Team Analysis", "description": "Team classification and formation detection using SigLIP embeddings", "outputs": ["annotated_video", "team_data"], "icon": "Users"},
    "track": {"id": "track", "name": "Object Tracking", "description": "Player and ball tracking with ByteTrack persistence", "outputs": ["annotated_video", "tracking_data"], "icon": "Target"},
    "players": {"id": "players", "name": "Player Detection", "description": "YOLOv8 player and referee detection with bounding boxes", "outputs": ["annotated_video"], "icon": "User"},
    "ball": {"id": "ball", "name": "Ball Tracking", "description": "Ball detection with SAHI slicer and trajectory interpolation", "outputs": ["annotated_video", "ball_data"], "icon": "Circle"},
    "pitch": {"id": "pitch", "name": "Pitch Mapping", "description": "Keypoint detection and homography transformation to pitch coordinates", "outputs": ["pitch_overlay", "homography_data"], "icon": "Map"},
}

PROCESSING_STAGES = [
    {"id": "upload", "name": "Uploading Video", "weight": 5},
    {"id": "load", "name": "Loading Frames", "weight": 10},
    {"id": "detect", "name": "Detecting Objects", "weight": 25},
    {"id": "track", "name": "Tracking Players", "weight": 20},
    {"id": "team", "name": "Classifying Teams", "weight": 15},
    {"id": "pitch", "name": "Mapping Pitch", "weight": 10},
    {"id": "analytics", "name": "Computing Analytics", "weight": 10},
    {"id": "render", "name": "Rendering Output", "weight": 5},
]


@router.get("/analysis")
async def list_analyses(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Analysis).where(Analysis.userId == user.id).order_by(desc(Analysis.createdAt))
    )
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.get("/analysis/modes")
async def get_modes():
    return PIPELINE_MODES


@router.get("/analysis/stages")
async def get_stages():
    return PROCESSING_STAGES


@router.get("/analysis/by-video/{video_id}")
async def list_by_video(video_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # Verify video belongs to user
    vr = await db.execute(select(Video).where(Video.id == video_id).limit(1))
    video = vr.scalar_one_or_none()
    if not video or video.userId != user.id:
        raise HTTPException(status_code=404, detail="Video not found")
    result = await db.execute(
        select(Analysis).where(Analysis.videoId == video_id).order_by(desc(Analysis.createdAt))
    )
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis or analysis.userId != user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _row_to_dict(analysis)


@router.post("/analysis")
async def create_analysis(body: AnalysisCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # Verify video belongs to user
    vr = await db.execute(select(Video).where(Video.id == body.videoId).limit(1))
    video = vr.scalar_one_or_none()
    if not video or video.userId != user.id:
        raise HTTPException(status_code=404, detail="Video not found")

    analysis = Analysis(
        videoId=body.videoId,
        userId=user.id,
        mode=body.mode,
        status="pending",
        progress=0,
        skipCache=body.fresh,
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)
    return {"id": analysis.id}


@router.put("/analysis/{analysis_id}/status")
async def update_status(analysis_id: int, body: AnalysisStatusUpdate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis or analysis.userId != user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")

    analysis.status = body.status
    analysis.progress = body.progress
    if body.currentStage is not None:
        analysis.currentStage = body.currentStage
    if body.errorMessage is not None:
        analysis.errorMessage = body.errorMessage
    if body.status == "processing" and body.progress == 0:
        analysis.startedAt = datetime.utcnow()
    if body.status in ("completed", "failed"):
        analysis.completedAt = datetime.utcnow()

    await db.commit()
    return {"success": True}


@router.put("/analysis/{analysis_id}/results")
async def update_results(analysis_id: int, body: AnalysisResultsUpdate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis or analysis.userId != user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if body.annotatedVideoUrl is not None:
        analysis.annotatedVideoUrl = body.annotatedVideoUrl
    if body.radarVideoUrl is not None:
        analysis.radarVideoUrl = body.radarVideoUrl
    if body.trackingDataUrl is not None:
        analysis.trackingDataUrl = body.trackingDataUrl
    if body.analyticsDataUrl is not None:
        analysis.analyticsDataUrl = body.analyticsDataUrl
    if body.processingTimeMs is not None:
        analysis.processingTimeMs = body.processingTimeMs

    await db.commit()
    return {"success": True}


@router.post("/analysis/{analysis_id}/terminate")
async def terminate_analysis(analysis_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis or analysis.userId != user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if analysis.status not in ("processing", "pending"):
        raise HTTPException(status_code=400, detail="Analysis is not running")

    analysis.status = "failed"
    analysis.errorMessage = "Terminated by user"
    analysis.completedAt = datetime.utcnow()
    await db.commit()
    return {"success": True}


@router.get("/analysis/{analysis_id}/eta")
async def get_eta(analysis_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis or analysis.userId != user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")

    import time
    now_ms = int(time.time() * 1000)

    start_time = now_ms
    if analysis.createdAt:
        ts = int(analysis.createdAt.timestamp() * 1000)
        start_time = ts

    elapsed = now_ms - start_time
    progress = max(analysis.progress or 1, 1)

    estimated_total = (elapsed / progress) * 100
    remaining = max(0, estimated_total - elapsed)

    stage_estimates = {
        "upload": 5, "uploading": 5,
        "load": 3, "loading": 3, "downloading": 5,
        "detect": 60, "detecting": 60,
        "track": 10, "tracking": 10,
        "team": 30, "classifying": 30,
        "pitch": 20, "mapping": 20,
        "analytics": 15, "computing": 15,
        "render": 45, "rendering": 45,
        "done": 0,
    }

    current_stage = analysis.currentStage or "uploading"
    stage_index = 0
    for i, s in enumerate(PROCESSING_STAGES):
        if s["id"] == current_stage:
            stage_index = i
            break

    stage_based_remaining = 0.0
    for i in range(stage_index, len(PROCESSING_STAGES)):
        stage_id = PROCESSING_STAGES[i]["id"]
        estimate = stage_estimates.get(stage_id, 10)
        if i == stage_index:
            stage_progress = (progress % (100 / len(PROCESSING_STAGES))) / (100 / len(PROCESSING_STAGES))
            stage_based_remaining += estimate * (1 - min(stage_progress, 1))
        else:
            stage_based_remaining += estimate

    final_estimate = (remaining / 1000 * 0.4) + (stage_based_remaining * 0.6)

    return {
        "elapsedMs": elapsed,
        "remainingMs": round(max(0, final_estimate * 1000)),
        "estimatedTotalMs": round(max(0, elapsed + final_estimate * 1000)),
        "currentStage": current_stage,
        "stageIndex": stage_index,
        "totalStages": len(PROCESSING_STAGES),
    }
