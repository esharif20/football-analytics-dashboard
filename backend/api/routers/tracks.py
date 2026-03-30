from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_current_user, get_db
from ..models import Analysis, Track, User
from ..schemas import TracksCreate, _row_to_dict

router = APIRouter(prefix="/tracks", tags=["tracks"])


async def _verify_analysis_owner(analysis_id: int, user: User, db: AsyncSession) -> Analysis:
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis or analysis.userId != user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


@router.get("/{analysis_id}")
async def list_tracks(
    analysis_id: int,
    offset: int = 0,
    limit: int = 200,
    frame_start: int | None = None,
    frame_end: int | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _verify_analysis_owner(analysis_id, user, db)
    query = select(Track).where(Track.analysisId == analysis_id)
    if frame_start is not None:
        query = query.where(Track.frameNumber >= frame_start)
    if frame_end is not None:
        query = query.where(Track.frameNumber <= frame_end)
    query = query.order_by(Track.frameNumber).offset(offset).limit(min(limit, 500))
    result = await db.execute(query)
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.get("/{analysis_id}/frame/{frame_number}")
async def get_at_frame(
    analysis_id: int,
    frame_number: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _verify_analysis_owner(analysis_id, user, db)
    result = await db.execute(
        select(Track)
        .where(and_(Track.analysisId == analysis_id, Track.frameNumber == frame_number))
        .limit(1)
    )
    track = result.scalar_one_or_none()
    if not track:
        return None
    return _row_to_dict(track)


@router.post("")
async def create_tracks(
    body: TracksCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    await _verify_analysis_owner(body.analysisId, user, db)

    batch_size = 100
    items = body.tracks
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        for t in batch:
            track = Track(
                analysisId=body.analysisId,
                frameNumber=t.frameNumber,
                timestamp=t.timestamp,
                playerPositions=t.playerPositions,
                ballPosition=t.ballPosition,
                teamFormations=t.teamFormations,
                voronoiData=t.voronoiData,
            )
            db.add(track)
        await db.flush()

    await db.commit()
    return {"success": True, "count": len(items)}
