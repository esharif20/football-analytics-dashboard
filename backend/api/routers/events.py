"""
Events Router - Match events management
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional, List, Any

from api.routers.auth import require_user
from api.services.database import (
    get_analysis_by_id, create_events, get_events_by_analysis, get_events_by_type
)

router = APIRouter()

class EventCreate(BaseModel):
    type: str
    frameNumber: int
    timestamp: float
    playerId: Optional[int] = None
    teamId: Optional[int] = None
    targetPlayerId: Optional[int] = None
    startX: Optional[float] = None
    startY: Optional[float] = None
    endX: Optional[float] = None
    endY: Optional[float] = None
    success: Optional[bool] = None
    confidence: Optional[float] = None
    metadata: Optional[Any] = None

class EventsCreateRequest(BaseModel):
    analysisId: int
    events: List[EventCreate]

@router.get("/{analysis_id}")
async def list_events(analysis_id: int, request: Request, user: dict = Depends(require_user)):
    """List all events for an analysis"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return get_events_by_analysis(analysis_id)

@router.get("/{analysis_id}/by-type/{event_type}")
async def list_events_by_type(
    analysis_id: int,
    event_type: str,
    request: Request,
    user: dict = Depends(require_user)
):
    """List events of a specific type"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return get_events_by_type(analysis_id, event_type)

@router.post("")
async def create_new_events(
    data: EventsCreateRequest,
    request: Request,
    user: dict = Depends(require_user)
):
    """Create multiple events"""
    analysis = get_analysis_by_id(data.analysisId)
    if not analysis or analysis["user_id"] != user["id"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    events_data = [
        {
            "analysis_id": data.analysisId,
            "type": e.type,
            "frame_number": e.frameNumber,
            "timestamp": e.timestamp,
            "player_id": e.playerId,
            "team_id": e.teamId,
            "target_player_id": e.targetPlayerId,
            "start_x": e.startX,
            "start_y": e.startY,
            "end_x": e.endX,
            "end_y": e.endY,
            "success": e.success,
            "confidence": e.confidence,
            "metadata": e.metadata,
        }
        for e in data.events
    ]
    
    create_events(events_data)
    return {"success": True, "count": len(data.events)}
