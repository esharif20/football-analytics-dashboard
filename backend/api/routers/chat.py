"""Chat router — interactive tactical Q&A grounded in match analytics."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_current_user, get_db
from ..models import User
from ..routers.commentary import _load_analytics_data, _verify_analysis_owner
from ..services.tactical import GroundingFormatter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

CHAT_SYSTEM_PROMPT = """\
You are an expert football tactical analyst with deep knowledge of the match data provided.
You have been given structured match analytics extracted from computer vision analysis.

Your role: Answer questions about this specific match based on the data provided.

Rules:
- Ground every answer in the data — never invent statistics or player names
- Use track IDs (e.g. "Player #4 (Team 1)") since real player names are not available
- Cite specific numbers when asked about performance metrics
- If a question cannot be answered from the available data, say so clearly
- Be concise and direct — coaching-staff briefing style

Match data context:
{grounded_markdown}"""


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@router.post("/{analysis_id}")
async def chat(
    analysis_id: int,
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Interactive tactical Q&A grounded in match analytics.

    Request body:
        messages: List of {role: "user"|"assistant", content: str}

    Returns:
        {role: "assistant", content: str}
    """
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    analysis = await _verify_analysis_owner(analysis_id, user, db)

    try:
        analytics_data = await _load_analytics_data(analysis)
    except HTTPException:
        raise

    try:
        grounded_markdown = GroundingFormatter.format(analytics_data)
        system_prompt = CHAT_SYSTEM_PROMPT.format(grounded_markdown=grounded_markdown)

        from ..services.llm_providers import get_provider

        provider = get_provider()
        messages = [{"role": m.role, "content": m.content} for m in body.messages]
        content = await provider.chat(system_prompt, messages)

    except RuntimeError as e:
        logger.warning("Chat LLM failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

    return {"role": "assistant", "content": content}
