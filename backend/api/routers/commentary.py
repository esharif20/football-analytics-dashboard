import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_db, get_current_user
from ..models import User, Analysis, Commentary
from ..schemas import CommentaryGenerate, _row_to_dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/commentary", tags=["commentary"])


async def _verify_analysis_owner(analysis_id: int, user: User, db: AsyncSession) -> Analysis:
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id).limit(1))
    analysis = result.scalar_one_or_none()
    if not analysis or analysis.userId != user.id:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


async def _load_analytics_data(analysis: Analysis) -> dict:
    """Load analytics JSON data for an analysis.

    Supports three storage shapes:
    - raw JSON string stored in the database
    - a local filesystem path
    - a /uploads/ URL pointing at local static storage
    """
    url = analysis.analyticsDataUrl
    if not url:
        raise HTTPException(
            status_code=400,
            detail="No analytics data available. Run the pipeline with analytics enabled first.",
        )

    try:
        if isinstance(url, str):
            stripped = url.strip()

            if stripped.startswith("{") or stripped.startswith("["):
                return json.loads(stripped)

            path_str = stripped
            if path_str.startswith("/uploads/"):
                from ..config import settings

                storage_dir = Path(settings.LOCAL_STORAGE_DIR).resolve()
                relative = path_str.replace("/uploads/", "", 1)
                path_str = str(storage_dir / relative)

            file_path = Path(path_str)
            if file_path.exists():
                with open(file_path, "r") as f:
                    return json.load(f)

        raise HTTPException(
            status_code=400,
            detail="Analytics data file not found. The pipeline may not have completed successfully.",
        )
    except json.JSONDecodeError as e:
        logger.error("Failed to parse analytics data from %s: %s", url, e)
        raise HTTPException(status_code=500, detail="Failed to load analytics data.")
    except OSError as e:
        logger.error("Failed to read analytics data from %s: %s", url, e)
        raise HTTPException(status_code=500, detail="Failed to load analytics data.")


@router.get("/types")
async def list_analysis_types():
    """Return available tactical analysis types."""
    from ..services.tactical import TacticalAnalyzer
    return TacticalAnalyzer.available_types()


@router.get("/{analysis_id}")
async def list_commentary(analysis_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _verify_analysis_owner(analysis_id, user, db)
    result = await db.execute(
        select(Commentary).where(Commentary.analysisId == analysis_id).order_by(Commentary.frameStart)
    )
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.post("/{analysis_id}")
async def generate_commentary(analysis_id: int, body: CommentaryGenerate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    analysis = await _verify_analysis_owner(analysis_id, user, db)

    # Determine analysis type — use body.type or default to match_overview
    analysis_type = body.type or "match_overview"

    try:
        # Load the analytics data from the pipeline output
        analytics_data = await _load_analytics_data(analysis)

        # Generate tactical analysis via LLM
        from ..services.tactical import TacticalAnalyzer
        analyzer = TacticalAnalyzer()
        result = await analyzer.analyze(analytics_data, analysis_type)

        content = result["content"]
        grounding = result["grounding_data"]

        # Merge any user-provided context into grounding data
        if body.context:
            grounding["user_context"] = body.context

    except RuntimeError as e:
        # No LLM provider available — return a helpful error
        logger.warning("LLM generation failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store in database
    item = Commentary(
        analysisId=analysis_id,
        type=analysis_type,
        content=content,
        confidence=1.0,
        groundingData=grounding,
        frameStart=body.context.get("timeRange", {}).get("start") if body.context else None,
        frameEnd=body.context.get("timeRange", {}).get("end") if body.context else None,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return {"id": item.id, "content": content, "type": analysis_type}
