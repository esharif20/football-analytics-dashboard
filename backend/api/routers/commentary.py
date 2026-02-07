from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db, get_current_user
from ..models import User, Analysis, Commentary
from ..schemas import CommentaryGenerate, _row_to_dict

router = APIRouter(prefix="/commentary", tags=["commentary"])


async def _verify_analysis_owner(analysis_id: int, user: User, db: AsyncSession) -> Analysis:
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis or analysis.userId != user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


@router.get("/{analysis_id}")
async def list_commentary(analysis_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _verify_analysis_owner(analysis_id, user, db)
    result = await db.execute(
        select(Commentary).where(Commentary.analysisId == analysis_id).order_by(Commentary.frameStart)
    )
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.post("/{analysis_id}")
async def generate_commentary(analysis_id: int, body: CommentaryGenerate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _verify_analysis_owner(analysis_id, user, db)

    # Generate placeholder commentary (LLM integration preserved as stub)
    content = "Analysis commentary will be generated when the pipeline completes processing."

    item = Commentary(
        analysisId=analysis_id,
        type=body.type,
        content=content,
        confidence=1.0,
        groundingData=body.context if body.context else None,
        frameStart=body.context.get("timeRange", {}).get("start") if body.context else None,
        frameEnd=body.context.get("timeRange", {}).get("end") if body.context else None,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return {"id": item.id, "content": content}
