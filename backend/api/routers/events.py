from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db, get_current_user
from ..models import User, Analysis, Event
from ..schemas import EventsCreate, _row_to_dict

router = APIRouter(prefix="/events", tags=["events"])


async def _verify_analysis_owner(analysis_id: int, user: User, db: AsyncSession) -> Analysis:
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis or analysis.userId != user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


@router.get("/{analysis_id}")
async def list_events(analysis_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _verify_analysis_owner(analysis_id, user, db)
    result = await db.execute(
        select(Event).where(Event.analysisId == analysis_id).order_by(Event.frameNumber)
    )
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.get("/{analysis_id}/by-type/{event_type}")
async def list_events_by_type(analysis_id: int, event_type: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _verify_analysis_owner(analysis_id, user, db)
    result = await db.execute(
        select(Event)
        .where(and_(Event.analysisId == analysis_id, Event.type == event_type))
        .order_by(Event.frameNumber)
    )
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.post("")
async def create_events(body: EventsCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _verify_analysis_owner(body.analysisId, user, db)

    for e in body.events:
        event = Event(
            analysisId=body.analysisId,
            type=e.type,
            frameNumber=e.frameNumber,
            timestamp=e.timestamp,
            playerId=e.playerId,
            teamId=e.teamId,
            targetPlayerId=e.targetPlayerId,
            startX=e.startX,
            startY=e.startY,
            endX=e.endX,
            endY=e.endY,
            success=e.success,
            confidence=e.confidence,
            event_metadata=e.metadata,
        )
        db.add(event)

    await db.commit()
    return {"success": True, "count": len(body.events)}
