from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db, get_current_user
from ..models import User, Analysis, Statistic
from ..schemas import StatisticsCreate, _row_to_dict

router = APIRouter(prefix="/statistics", tags=["statistics"])


async def _verify_analysis_owner(analysis_id: int, user: User, db: AsyncSession) -> Analysis:
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis or analysis.userId != user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


@router.get("/{analysis_id}")
async def get_statistics(analysis_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _verify_analysis_owner(analysis_id, user, db)
    result = await db.execute(
        select(Statistic).where(Statistic.analysisId == analysis_id).limit(1)
    )
    stat = result.scalar_one_or_none()
    if not stat:
        return None
    return _row_to_dict(stat)


@router.post("")
async def create_statistics(body: StatisticsCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _verify_analysis_owner(body.analysisId, user, db)

    stat = Statistic(
        analysisId=body.analysisId,
        possessionTeam1=body.possessionTeam1,
        possessionTeam2=body.possessionTeam2,
        passesTeam1=body.passesTeam1,
        passesTeam2=body.passesTeam2,
        passAccuracyTeam1=body.passAccuracyTeam1,
        passAccuracyTeam2=body.passAccuracyTeam2,
        shotsTeam1=body.shotsTeam1,
        shotsTeam2=body.shotsTeam2,
        distanceCoveredTeam1=body.distanceCoveredTeam1,
        distanceCoveredTeam2=body.distanceCoveredTeam2,
        avgSpeedTeam1=body.avgSpeedTeam1,
        avgSpeedTeam2=body.avgSpeedTeam2,
        heatmapDataTeam1=body.heatmapDataTeam1,
        heatmapDataTeam2=body.heatmapDataTeam2,
        passNetworkTeam1=body.passNetworkTeam1,
        passNetworkTeam2=body.passNetworkTeam2,
    )
    db.add(stat)
    await db.commit()
    await db.refresh(stat)
    return {"id": stat.id}
