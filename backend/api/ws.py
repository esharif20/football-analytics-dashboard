import json
import hmac
import hashlib
from urllib.parse import parse_qs
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy import select

from .config import settings
from .database import async_session
from .models import Analysis

# In-memory registry: analysis_id -> set of connected WebSocket clients
_subscribers: dict[int, set[WebSocket]] = {}


async def websocket_endpoint(ws: WebSocket, analysis_id: int):
    """
    WebSocket endpoint: /ws/{analysis_id}
    The frontend connects here to receive real-time progress updates for a specific analysis.
    """
    # Basic auth: require a signed token unless in permissive local dev mode.
    if not await _authorize_ws(ws, analysis_id):
        return

    await ws.accept()

    if analysis_id not in _subscribers:
        _subscribers[analysis_id] = set()
    _subscribers[analysis_id].add(ws)

    # Send connected confirmation
    await ws.send_json({"type": "connected", "analysisId": analysis_id})

    try:
        while True:
            # Keep connection alive; client doesn't send meaningful data
            data = await ws.receive_text()
            # Optionally handle subscribe/unsubscribe messages from legacy clients
            try:
                msg = json.loads(data)
                if msg.get("type") == "subscribe" and msg.get("analysisId"):
                    new_id = int(msg["analysisId"])
                    if new_id not in _subscribers:
                        _subscribers[new_id] = set()
                    _subscribers[new_id].add(ws)
            except (json.JSONDecodeError, ValueError):
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _subscribers.get(analysis_id, set()).discard(ws)
        # Also remove from any other subscriptions
        for s in _subscribers.values():
            s.discard(ws)


async def broadcast_progress(analysis_id: int, status: str, progress: int, current_stage: str | None, eta: int | None = None):
    msg = {
        "type": "progress",
        "analysisId": analysis_id,
        "data": {
            "status": status,
            "progress": progress,
            "currentStage": current_stage,
            "eta": eta,
        },
    }
    await _broadcast(analysis_id, msg)


async def broadcast_complete(analysis_id: int, result: dict | None = None):
    msg = {
        "type": "complete",
        "analysisId": analysis_id,
        "data": {
            "status": "completed",
            "progress": 100,
            "currentStage": "done",
            "result": result,
        },
    }
    await _broadcast(analysis_id, msg)
    # Clean up subscriptions
    _subscribers.pop(analysis_id, None)


async def broadcast_error(analysis_id: int, error: str):
    msg = {
        "type": "error",
        "analysisId": analysis_id,
        "data": {
            "status": "failed",
            "error": error,
        },
    }
    await _broadcast(analysis_id, msg)
    _subscribers.pop(analysis_id, None)


async def _broadcast(analysis_id: int, msg: dict):
    subs = _subscribers.get(analysis_id, set()).copy()
    dead: list[WebSocket] = []
    text = json.dumps(msg)
    for ws in subs:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _subscribers.get(analysis_id, set()).discard(ws)


async def _authorize_ws(ws: WebSocket, analysis_id: int) -> bool:
    # Allow permissive access only when explicitly in local dev and no worker key configured
    qs = parse_qs(ws.scope.get("query_string", b"").decode()) if ws.scope else {}
    token = (qs.get("token") or [None])[0]

    if settings.LOCAL_DEV_MODE and not settings.WORKER_API_KEY and not token:
        return True

    if token is None:
        await ws.accept()
        await ws.close(code=4401, reason="Missing token")
        return False

    if async_session is None:
        await ws.accept()
        await ws.close(code=1011, reason="Database unavailable")
        return False

    async with async_session() as db:
        res = await db.execute(select(Analysis.userId).where(Analysis.id == analysis_id).limit(1))
        user_id = res.scalar_one_or_none()

    if user_id is None:
        await ws.accept()
        await ws.close(code=4404, reason="Analysis not found")
        return False

    secret = (settings.JWT_SECRET or "dev-secret").encode()
    expected = hmac.new(secret, f"{analysis_id}:{user_id}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(token, expected):
        await ws.accept()
        await ws.close(code=4401, reason="Invalid token")
        return False

    return True
