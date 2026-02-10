import base64
import json
import logging
import time
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db
from ..models import Analysis, Video, Statistic, Event
from ..schemas import WorkerStatusUpdate, WorkerCompleteRequest, WorkerUploadVideo, _row_to_dict
from ..storage import storage_put, reencode_to_h264
from ..ws import broadcast_progress, broadcast_complete, broadcast_error

logger = logging.getLogger(__name__)
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


def _strip_analytics_for_storage(analytics: dict) -> dict:
    """Strip large per-frame arrays from analytics to fit in DB TEXT column.

    Keeps summary stats, removes per-frame speed arrays, event lists, and
    position lists that can be hundreds of KB.
    """
    out = {}
    for key, val in analytics.items():
        if key == "possession" and isinstance(val, dict):
            # Keep summary stats, drop the per-frame events list
            out[key] = {k: v for k, v in val.items() if k != "events"}
        elif key == "player_kinematics" and isinstance(val, dict):
            # Keep summary stats per player, drop per-frame speed arrays
            stripped = {}
            for pid, pdata in val.items():
                if isinstance(pdata, dict):
                    stripped[pid] = {
                        k: v for k, v in pdata.items()
                        if k not in ("speeds_px_per_frame", "speeds_m_per_sec")
                    }
                else:
                    stripped[pid] = pdata
            out[key] = stripped
        elif key == "ball_kinematics" and isinstance(val, dict):
            out[key] = {
                k: v for k, v in val.items()
                if k not in ("speeds_px_per_frame", "speeds_m_per_sec")
            }
        elif key == "ball_path" and isinstance(val, dict):
            # Keep summary + downsampled pitch_positions, drop pixel positions
            stripped = {k: v for k, v in val.items() if k != "positions"}
            pitch_pos = val.get("pitch_positions", [])
            if len(pitch_pos) > 200:
                step = len(pitch_pos) / 200
                pitch_pos = [pitch_pos[int(i * step)] for i in range(200)]
            stripped["pitch_positions"] = pitch_pos
            out[key] = stripped
        elif key in ("interaction_graph_team1", "interaction_graph_team2"):
            # Stored in Statistic.passNetworkTeam1/Team2 instead
            continue
        elif key == "events" and isinstance(val, list):
            # Keep event summary only — full events are stored in the events table
            out[key] = [
                {k: v for k, v in e.items() if k not in ("pitch_start", "pitch_end")}
                for e in val if isinstance(e, dict)
            ]
        else:
            out[key] = val
    return out


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
    if body.analytics and isinstance(body.analytics, dict):
        # Strip large per-frame arrays to fit in TEXT column (~64KB limit)
        stripped = _strip_analytics_for_storage(body.analytics)
        analysis.analyticsDataUrl = json.dumps(stripped)
    elif body.analytics:
        analysis.analyticsDataUrl = str(body.analytics)
    if body.tracks:
        analysis.trackingDataUrl = json.dumps(body.tracks) if isinstance(body.tracks, dict) else str(body.tracks)

    # Mark completed
    analysis.status = "completed"
    analysis.progress = 100
    analysis.currentStage = "done"
    analysis.completedAt = datetime.utcnow()

    # Delete any existing statistics for this analysis (re-run case)
    existing_stats = await db.execute(select(Statistic).where(Statistic.analysisId == analysis_id))
    for old_stat in existing_stats.scalars().all():
        await db.delete(old_stat)

    # Create statistics record from analytics data
    # Analytics structure from pipeline: {possession: {team_1_percentage, ...}, player_kinematics: {...}, ...}
    if body.analytics and isinstance(body.analytics, dict):
        a = body.analytics
        poss = a.get("possession", {})

        # Aggregate player kinematics by team
        pk = a.get("player_kinematics", {})
        team_distances: dict[int, list[float]] = {0: [], 1: []}
        team_avg_speeds: dict[int, list[float]] = {0: [], 1: []}
        team_max_speeds: dict[int, list[float]] = {0: [], 1: []}

        for _pid, pdata in pk.items():
            if not isinstance(pdata, dict):
                continue
            tid = pdata.get("team_id")
            if tid not in (0, 1):
                continue
            d = pdata.get("total_distance_m")
            if d is not None:
                team_distances[tid].append(d)
            avg_s = pdata.get("avg_speed_m_per_sec")
            if avg_s is not None:
                team_avg_speeds[tid].append(avg_s)
            max_s = pdata.get("max_speed_m_per_sec")
            if max_s is not None:
                team_max_speeds[tid].append(max_s)

        def _sum_to_km(vals: list[float]) -> float | None:
            return round(sum(vals) / 1000, 2) if vals else None

        def _avg_to_kmh(vals: list[float]) -> float | None:
            return round(sum(vals) / len(vals) * 3.6, 1) if vals else None

        def _max_to_kmh(vals: list[float]) -> float | None:
            return round(max(vals) * 3.6, 1) if vals else None

        # Ball kinematics
        bk = a.get("ball_kinematics", {})
        bp = a.get("ball_path", {})

        # Count passes and shots per team from detected events
        events_list = a.get("events", [])
        passes_t1 = sum(1 for e in events_list if isinstance(e, dict) and e.get("event_type") == "pass" and e.get("team_id") == 1)
        passes_t2 = sum(1 for e in events_list if isinstance(e, dict) and e.get("event_type") == "pass" and e.get("team_id") == 2)
        shots_t1 = sum(1 for e in events_list if isinstance(e, dict) and e.get("event_type") == "shot" and e.get("team_id") == 1)
        shots_t2 = sum(1 for e in events_list if isinstance(e, dict) and e.get("event_type") == "shot" and e.get("team_id") == 2)

        stat = Statistic(
            analysisId=analysis_id,
            possessionTeam1=round(poss.get("team_1_percentage", 50), 1),
            possessionTeam2=round(poss.get("team_2_percentage", 50), 1),
            passesTeam1=passes_t1 or None,
            passesTeam2=passes_t2 or None,
            shotsTeam1=shots_t1 or None,
            shotsTeam2=shots_t2 or None,
            distanceCoveredTeam1=_sum_to_km(team_distances[0]),
            distanceCoveredTeam2=_sum_to_km(team_distances[1]),
            avgSpeedTeam1=_avg_to_kmh(team_avg_speeds[0]),
            avgSpeedTeam2=_avg_to_kmh(team_avg_speeds[1]),
            maxSpeedTeam1=_max_to_kmh(team_max_speeds[0]),
            maxSpeedTeam2=_max_to_kmh(team_max_speeds[1]),
            possessionChanges=poss.get("possession_changes"),
            ballDistance=round(bp.get("total_distance_m", 0) / 1000, 2) if bp.get("total_distance_m") else None,
            ballAvgSpeed=round(bk.get("avg_speed_m_per_sec", 0) * 3.6, 1) if bk.get("avg_speed_m_per_sec") else None,
            ballMaxSpeed=round(bk.get("max_speed_m_per_sec", 0) * 3.6, 1) if bk.get("max_speed_m_per_sec") else None,
            directionChanges=bp.get("direction_changes"),
            passNetworkTeam1=a.get("interaction_graph_team1"),
            passNetworkTeam2=a.get("interaction_graph_team2"),
        )

        # Extract team jersey colors (BGR→hex)
        tc = a.get("team_colors", {})
        def _bgr_to_hex(bgr):
            if not bgr or len(bgr) != 3:
                return None
            b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
            return f"#{r:02X}{g:02X}{b:02X}"
        stat.teamColorTeam1 = _bgr_to_hex(tc.get("0") or tc.get(0))
        stat.teamColorTeam2 = _bgr_to_hex(tc.get("1") or tc.get(1))
        db.add(stat)

        # Store detected events in the events table
        if events_list:
            # Delete existing events for re-run case
            existing_events = await db.execute(select(Event).where(Event.analysisId == analysis_id))
            for old_ev in existing_events.scalars().all():
                await db.delete(old_ev)

            for ev in events_list:
                if not isinstance(ev, dict):
                    continue
                db.add(Event(
                    analysisId=analysis_id,
                    type=ev.get("event_type", "unknown"),
                    frameNumber=ev.get("frame_idx", 0),
                    timestamp=ev.get("timestamp_sec", 0.0),
                    playerId=ev.get("player_track_id"),
                    teamId=ev.get("team_id"),
                    targetPlayerId=ev.get("target_player_track_id"),
                    startX=ev.get("pitch_start", [None, None])[0] if ev.get("pitch_start") else None,
                    startY=ev.get("pitch_start", [None, None])[1] if ev.get("pitch_start") else None,
                    endX=ev.get("pitch_end", [None, None])[0] if ev.get("pitch_end") else None,
                    endY=ev.get("pitch_end", [None, None])[1] if ev.get("pitch_end") else None,
                    success=ev.get("success"),
                    confidence=ev.get("confidence"),
                ))

    try:
        await db.commit()
    except Exception as e:
        logger.error("Failed to commit analysis %s completion: %s", analysis_id, e)
        await db.rollback()
        return {"success": False, "error": f"Database error: {str(e)[:200]}"}

    # Broadcast completion via WebSocket (send only lightweight signal;
    # full analytics can be huge and may contain unserializable types)
    await broadcast_complete(analysis_id, None)

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
