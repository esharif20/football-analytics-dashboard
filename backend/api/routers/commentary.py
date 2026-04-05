import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_current_user, get_db
from ..models import Analysis, Commentary, User
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
                with open(file_path) as f:
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


def _resolve_video_path(analysis: Analysis) -> str | None:
    """Resolve the annotated video URL to a local file path for frame extraction."""
    url = analysis.annotatedVideoUrl
    if not url:
        return None
    from ..config import settings
    from ..services.vision import resolve_local_path

    return resolve_local_path(url, settings.LOCAL_STORAGE_DIR)


@router.get("/types")
async def list_analysis_types():
    """Return available tactical analysis types."""
    from ..services.tactical import TacticalAnalyzer

    return TacticalAnalyzer.available_types()


@router.get("/{analysis_id}")
async def list_commentary(
    analysis_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    await _verify_analysis_owner(analysis_id, user, db)
    result = await db.execute(
        select(Commentary)
        .where(Commentary.analysisId == analysis_id)
        .order_by(Commentary.frameStart)
    )
    return [_row_to_dict(r) for r in result.scalars().all()]


@router.post("/{analysis_id}")
async def generate_commentary(
    analysis_id: int,
    body: CommentaryGenerate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    analysis = await _verify_analysis_owner(analysis_id, user, db)

    # Determine analysis type — use body.type or default to match_overview
    analysis_type = body.type or "match_overview"

    try:
        analytics_data = await _load_analytics_data(analysis)
        video_path = _resolve_video_path(analysis)

        from ..services.tactical import TacticalAnalyzer

        analyzer = TacticalAnalyzer()
        result = await analyzer.analyze(analytics_data, analysis_type, video_path=video_path)

        content = result["content"]
        grounding = result["grounding_data"]

        # Merge any user-provided context into grounding data
        if body.context:
            grounding["user_context"] = body.context

    except RuntimeError as e:
        logger.warning("LLM generation failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
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


@router.post("/{analysis_id}/stream")
async def stream_commentary(
    analysis_id: int,
    body: CommentaryGenerate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stream tactical analysis as SSE events.

    Yields: data: {"text": "<chunk>"}\n\n  for each text chunk
    Final:  data: {"done": true, "id": <id>, "type": "<type>"}\n\n
    Error:  data: {"error": "<message>"}\n\n
    """
    analysis = await _verify_analysis_owner(analysis_id, user, db)
    analysis_type = body.type or "match_overview"

    try:
        analytics_data = await _load_analytics_data(analysis)
    except HTTPException as exc:
        error_detail = exc.detail

        async def error_stream():
            yield f"data: {json.dumps({'error': error_detail})}\n\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    video_path = _resolve_video_path(analysis)

    async def event_stream():
        from ..database import async_session
        from ..models import Commentary as CommentaryModel
        from ..services.tactical import TacticalAnalyzer

        analyzer = TacticalAnalyzer()
        full_content = ""

        try:
            async for chunk in analyzer.stream_analyze(
                analytics_data, analysis_type, video_path=video_path
            ):
                full_content += chunk
                yield f"data: {json.dumps({'text': chunk})}\n\n"

        except RuntimeError as e:
            logger.warning("LLM streaming failed: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return
        except Exception as e:
            logger.error("Unexpected streaming error: %s", e)
            yield f"data: {json.dumps({'error': 'Generation failed'})}\n\n"
            return

        if not full_content:
            yield f"data: {json.dumps({'error': 'Empty response from LLM'})}\n\n"
            return

        # Save to DB using a fresh session (avoids dependency lifecycle issues)
        try:
            if async_session is not None:
                grounding = {
                    "analysis_type": analysis_type,
                    "vision_augmented": video_path is not None,
                    "streamed": True,
                }
                if body.context:
                    grounding["user_context"] = body.context

                async with async_session() as save_db:
                    item = CommentaryModel(
                        analysisId=analysis_id,
                        type=analysis_type,
                        content=full_content,
                        confidence=1.0,
                        groundingData=grounding,
                        frameStart=body.context.get("timeRange", {}).get("start")
                        if body.context
                        else None,
                        frameEnd=body.context.get("timeRange", {}).get("end")
                        if body.context
                        else None,
                    )
                    save_db.add(item)
                    await save_db.commit()
                    await save_db.refresh(item)
                    yield f"data: {json.dumps({'done': True, 'id': item.id, 'type': analysis_type})}\n\n"
            else:
                yield f"data: {json.dumps({'done': True, 'id': None, 'type': analysis_type})}\n\n"
        except Exception as e:
            logger.error("Failed to save streamed commentary: %s", e)
            yield f"data: {json.dumps({'done': True, 'id': None, 'type': analysis_type})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
