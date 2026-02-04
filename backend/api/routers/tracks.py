"""
Tracks Router - Player and ball tracking data
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional, List, Any

from api.routers.auth import require_user
from api.services.database import (
    get_analysis_by_id, create_tracks, get_tracks_by_analysis, get_track_at_frame
)

router = APIRouter()

class TrackCreate(BaseModel):
    frameNumber: int
    timestamp: float
    playerPositions: Any
    ballPosition: Optional[Any] = None
    teamFormations: Optional[Any] = None
    voronoiData: Optional[Any] = None

class TracksCreateRequest(BaseModel):
    analysisId: int
    tracks: List[TrackCreate]

@router.get("/{analysis_id}")
async def list_tracks(analysis_id: int, request: Request, user: dict = Depends(require_user)):
    """List all tracks for an analysis"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return get_tracks_by_analysis(analysis_id)

@router.get("/{analysis_id}/frame/{frame_number}")
async def get_track_frame(
    analysis_id: int,
    frame_number: int,
    request: Request,
    user: dict = Depends(require_user)
):
    """Get track data at specific frame"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    track = get_track_at_frame(analysis_id, frame_number)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found for this frame")
    return track

@router.post("")
async def create_new_tracks(
    data: TracksCreateRequest,
    request: Request,
    user: dict = Depends(require_user)
):
    """Create multiple track records"""
    analysis = get_analysis_by_id(data.analysisId)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    tracks_data = [
        {
            "analysis_id": data.analysisId,
            "frame_number": t.frameNumber,
            "timestamp": t.timestamp,
            "player_positions": t.playerPositions,
            "ball_position": t.ballPosition,
            "team_formations": t.teamFormations,
            "voronoi_data": t.voronoiData,
        }
        for t in data.tracks
    ]
    
    create_tracks(tracks_data)
    return {"success": True, "count": len(data.tracks)}
