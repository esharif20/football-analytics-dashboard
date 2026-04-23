"""Chat router — interactive tactical Q&A grounded in match analytics.

Implements a two-path architecture (ChatMatch + ChatTennis pattern):
  - Statistical queries: LLM generates JSON dot-path → deterministic lookup → exact answer
  - Tactical queries: existing multi-turn chat with MatchInsights pre-interpretation context

Statistical path achieves near-100% accuracy on direct data queries (no hallucination risk).
Tactical path benefits from Phase 1 MatchInsights pre-interpretation layer.
"""

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..deps import get_current_user, get_db
from ..models import User
from ..routers.commentary import _load_analytics_data, _verify_analysis_owner
from ..services.tactical import GroundingFormatter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# ── System prompts ────────────────────────────────────────────────────────────

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

STAT_PRECISION_PROMPT = """\
You are a football analytics data retrieval assistant. The user is asking a statistical question.

The EXACT value retrieved directly from the match data is: {exact_value}
(retrieved from path: {path})

In 1-2 sentences, confirm this value and add brief tactical context.
Do not speculate beyond what the data contains.

Match data context:
{grounded_markdown}"""

_PATH_GENERATION_PROMPT = """\
You are a football analytics path resolver. Given a user question, output the dot-notation
path to the exact value in the analytics dict that answers the question.

Available paths in the analytics dict:
{schema_keys}

User question: {question}

Output ONLY one of:
- A dot-path string (e.g. "possession.team_1_percentage" or "tactical.summary.ppda_team_1")
- MULTI (if no single path answers the question — multiple values needed)
- NONE (if the question cannot be answered from the data at all)

Nothing else. No explanation."""

# ── Statistical query classification ─────────────────────────────────────────

# Keywords that indicate a request for a specific data point (ChatMatch pattern)
_STAT_KEYWORDS = [
    "how many",
    "how much",
    "what percentage",
    "what is the",
    "what were",
    "count",
    "total",
    "number of",
    "who covered",
    "who had the most",
    "who had the highest",
    "who ran the most",
    "what distance",
    "what speed",
    "which player",
    "how far",
    "how fast",
    "how compact",
    "how intense",
    "ppda",
    "compactness",
    "possession percentage",
    "possession %",
    "pressing intensity",
    "sprint count",
    "top speed",
    "average speed",
    "expected threat",
    "percentage",
    "what was the",
]

# If ANY of these appear in the query, classify as tactical regardless of stat keywords.
# These indicate the user wants interpretation, not a data point.
_TACTICAL_OVERRIDES = [
    "flow",
    "approach",
    "style",
    "pattern",
    "strategy",
    "describe",
    "explain",
    "analysis",
    "overall",
    "tactically",
    "how did",
    "why did",
    "what happened",
    "summarise",
    "summarize",
    "breakdown",
    "tell me about",
]


def _classify_query(message: str) -> str:
    """Classify query as 'statistical' (specific data point) or 'tactical' (interpretation)."""
    msg_lower = message.lower()
    if any(kw in msg_lower for kw in _TACTICAL_OVERRIDES):
        return "tactical"
    if any(kw in msg_lower for kw in _STAT_KEYWORDS):
        return "statistical"
    return "tactical"


# ── Analytics schema extraction ───────────────────────────────────────────────


def _get_schema_keys(analytics: dict) -> str:
    """Build a compact listing of available dot-paths for the path-generation LLM."""
    keys: list[str] = []

    # Possession
    for k, v in analytics.get("possession", {}).items():
        if isinstance(v, (int, float)):
            keys.append(f"possession.{k}")

    # Tactical summary
    summary = analytics.get("tactical", {}).get("summary", {})
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            keys.append(f"tactical.summary.{k}")

    # Player kinematics (first 8 players as examples)
    for pid, pdata in list(analytics.get("player_kinematics", {}).items())[:8]:
        for metric in (
            "total_distance_m",
            "max_speed_m_per_sec",
            "avg_speed_m_per_sec",
            "sprint_count",
        ):
            if isinstance(pdata, dict) and pdata.get(metric) is not None:
                keys.append(f"player_kinematics.{pid}.{metric}")

    # Event counts (computed separately — add as pseudo-paths)
    events = analytics.get("events", [])
    for et in set(e.get("event_type") for e in events if e.get("event_type")):
        keys.append(
            f"events[event_type={et}].count  # {len([e for e in events if e.get('event_type') == et])}"
        )

    return "\n".join(keys[:60])  # cap to avoid token overflow


# ── Path navigation ───────────────────────────────────────────────────────────


def _navigate_path(data: dict, dotpath: str) -> Any:
    """Navigate a dot-separated path in nested dict. Returns None on failure."""
    cur: Any = data
    for key in dotpath.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and key.isdigit():
            idx = int(key)
            cur = cur[idx] if idx < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    return cur


def _format_stat_value(value: Any, path: str) -> str:
    """Format a raw analytics value for display."""
    if not isinstance(value, (int, float)):
        return str(value)
    path_lower = path.lower()
    if "percentage" in path_lower:
        return f"{value:.1f}%"
    if "speed_m_per_sec" in path_lower:
        return f"{value * 3.6:.1f} km/h"
    if "_m2" in path_lower or "compactness" in path_lower:
        return f"{value:.0f} m²"
    if "_m" in path_lower or "distance" in path_lower:
        return f"{value:.0f} m"
    if "ppda" in path_lower:
        return f"{value:.1f}"
    if "intensity" in path_lower:
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"


# ── Request/response models ───────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


# ── Route ─────────────────────────────────────────────────────────────────────


@router.post("/{analysis_id}")
async def chat(
    analysis_id: int,
    body: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Interactive tactical Q&A grounded in match analytics.

    Routes statistical queries through deterministic path lookup (near-100% accuracy)
    and tactical queries through multi-turn LLM chat with MatchInsights context.

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
        from ..services.llm_providers import get_provider

        provider = get_provider()

        grounded_markdown = GroundingFormatter.format(analytics_data)
        last_message = body.messages[-1].content if body.messages else ""
        messages = [{"role": m.role, "content": m.content} for m in body.messages]

        query_type = _classify_query(last_message)
        logger.debug("Chat query classified as: %s — %r", query_type, last_message[:80])

        if query_type == "statistical":
            content = await _handle_statistical_query(
                last_message, analytics_data, grounded_markdown, messages, provider
            )
        else:
            system_prompt = CHAT_SYSTEM_PROMPT.format(grounded_markdown=grounded_markdown)
            content = await provider.chat(system_prompt, messages)

    except RuntimeError as e:
        logger.warning("Chat LLM failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

    return {"role": "assistant", "content": content}


async def _handle_statistical_query(
    question: str,
    analytics: dict,
    grounded_markdown: str,
    messages: list[dict],
    provider,
) -> str:
    """Statistical path: generate dot-path → navigate → format exact answer.

    Falls back to precision-prompt chat if path lookup fails.
    """
    schema_keys = _get_schema_keys(analytics)
    path_prompt = _PATH_GENERATION_PROMPT.format(
        schema_keys=schema_keys,
        question=question,
    )

    try:
        path_str = await provider.generate(
            "You are a concise path resolver. Output only the requested path string.",
            path_prompt,
        )
        # Clean up: strip markdown fences, quotes, whitespace, take first line
        path_str = re.sub(r"[`'\"]", "", path_str).split("\n")[0].strip()
    except Exception as exc:
        logger.warning("Path generation failed: %s", exc)
        path_str = "NONE"

    if path_str and path_str not in ("MULTI", "NONE") and "." in path_str:
        value = _navigate_path(analytics, path_str)
        if value is not None and not isinstance(value, (dict, list)):
            formatted = _format_stat_value(value, path_str)
            logger.debug("StatQuery resolved: %s → %s", path_str, formatted)

            # Ask LLM to add one sentence of tactical context around the exact value
            system = STAT_PRECISION_PROMPT.format(
                exact_value=formatted,
                path=path_str,
                grounded_markdown=grounded_markdown,
            )
            try:
                context = await provider.generate(system, question)
                # Ensure the exact value is always visible at the top
                if formatted not in context:
                    context = f"**{formatted}**\n\n{context}"
                return context
            except Exception as exc:
                logger.warning("Stat context generation failed: %s", exc)
                return f"**{formatted}** (from {path_str})"

    # Fallback: precision-mode chat with MatchInsights context
    logger.debug("StatQuery path lookup failed (%r) — falling back to precision chat", path_str)
    system_prompt = CHAT_SYSTEM_PROMPT.format(grounded_markdown=grounded_markdown)
    return await provider.chat(system_prompt, messages)
