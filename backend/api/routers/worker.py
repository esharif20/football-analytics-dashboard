import base64
import json
import time
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db
from ..models import Analysis, Video, Statistic
from ..schemas import WorkerStatusUpdate, WorkerCompleteRequest, WorkerUploadVideo, _row_to_dict
from ..storage import storage_put, reencode_to_h264
from ..ws import broadcast_progress, broadcast_complete, broadcast_error

router = APIRouter(prefix="/worker", tags=["worker"])


@router.get("/pending")
async def get_pending(db: AsyncSession = Depends(get_db)):
    """Return pending analyses for the worker to pick up."""
    result = await db.execute(
        select(Analysis.id, Analysis.videoId, Analysis.mode, Analysis.skipCache, Video.originalUrl)
        .join(Video, Analysis.videoId == Video.id)
        .where(Analysis.status == "pending")
        .order_by(Analysis.createdAt)
        .limit(10)
    )
    rows = result.all()
    analyses = []
    for row in rows:
        analyses.append({
            "id": row[0],
            "videoId": row[1],
            "videoUrl": row[4],
            "mode": row[2],
            "skipCache": row[3],
            "modelConfig": {},
        })
    return {"analyses": analyses}


@router.post("/analysis/{analysis_id}/status")
async def update_worker_status(analysis_id: int, body: WorkerStatusUpdate, db: AsyncSession = Depends(get_db)):
    """Worker sends progress updates during processing."""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis:
        return {"success": False, "error": "Analysis not found"}

    analysis.status = body.status
    analysis.progress = body.progress
    if body.currentStage:
        analysis.currentStage = body.currentStage
    if body.error:
        analysis.errorMessage = body.error
    if body.status == "processing" and body.progress == 0:
        analysis.startedAt = datetime.utcnow()
    if body.status in ("completed", "failed"):
        analysis.completedAt = datetime.utcnow()

    await db.commit()

    # Broadcast via WebSocket
    if body.status == "failed" and body.error:
        await broadcast_error(analysis_id, body.error)
    else:
        await broadcast_progress(analysis_id, body.status, body.progress, body.currentStage, body.eta)

    return {"success": True}


@router.post("/analysis/{analysis_id}/complete")
async def complete_analysis(analysis_id: int, body: WorkerCompleteRequest, db: AsyncSession = Depends(get_db)):
    """Worker submits final results."""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis:
        return {"success": False, "error": "Analysis not found"}

    # Update analysis with result URLs
    if body.annotatedVideo:
        analysis.annotatedVideoUrl = body.annotatedVideo
    if body.radarVideo:
        analysis.radarVideoUrl = body.radarVideo
    if body.analytics:
        analysis.analyticsDataUrl = json.dumps(body.analytics) if isinstance(body.analytics, dict) else str(body.analytics)
    if body.tracks:
        analysis.trackingDataUrl = json.dumps(body.tracks) if isinstance(body.tracks, dict) else str(body.tracks)

    # Mark completed
    analysis.status = "completed"
    analysis.progress = 100
    analysis.currentStage = "done"
    analysis.completedAt = datetime.utcnow()

    # Create statistics record from analytics data
    if body.analytics and isinstance(body.analytics, dict):
        a = body.analytics
        stat = Statistic(
            analysisId=analysis_id,
            possessionTeam1=a.get("possession", {}).get("team1", 50),
            possessionTeam2=a.get("possession", {}).get("team2", 50),
            passesTeam1=a.get("passes", {}).get("team1", 0),
            passesTeam2=a.get("passes", {}).get("team2", 0),
            passAccuracyTeam1=a.get("passAccuracy", {}).get("team1", 0),
            passAccuracyTeam2=a.get("passAccuracy", {}).get("team2", 0),
            shotsTeam1=a.get("shots", {}).get("team1", 0),
            shotsTeam2=a.get("shots", {}).get("team2", 0),
            distanceCoveredTeam1=a.get("distance", {}).get("team1", 0),
            distanceCoveredTeam2=a.get("distance", {}).get("team2", 0),
            avgSpeedTeam1=a.get("avgSpeed", {}).get("team1", 0),
            avgSpeedTeam2=a.get("avgSpeed", {}).get("team2", 0),
        )
        db.add(stat)

    await db.commit()

    # Broadcast completion via WebSocket
    await broadcast_complete(analysis_id, {"analytics": body.analytics, "tracks": body.tracks})

    return {"success": True}


@router.post("/upload-video")
async def upload_video(body: WorkerUploadVideo, db: AsyncSession = Depends(get_db)):
    """Worker uploads processed video files (base64 encoded)."""
    video_bytes = base64.b64decode(body.videoData)
    key = f"processed-videos/{int(time.time() * 1000)}-{body.fileName}"
    result = await storage_put(key, video_bytes, body.contentType)

    # Re-encode to H.264 for browser compatibility (pipeline outputs mp4v)
    if body.contentType.startswith("video/") and result.get("path"):
        reencode_to_h264(result["path"])

    return {"success": True, "url": result["url"]}
