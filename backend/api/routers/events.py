"""
Events Router - Match events management
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any

from api.services.database import get_analysis_by_id

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
async def list_events(analysis_id: int):
    """List all events for an analysis"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    # Events would be loaded from analytics JSON
    return []

@router.get("/{analysis_id}/by-type/{event_type}")
async def list_events_by_type(analysis_id: int, event_type: str):
    """List events of a specific type"""
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return []

@router.post("")
async def create_new_events(data: EventsCreateRequest):
    """Create multiple events"""
    analysis = get_analysis_by_id(data.analysisId)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    # Events are stored in analytics JSON, not separate table
    return {"success": True, "count": len(data.events)}
