import json
from fastapi import WebSocket, WebSocketDisconnect

# In-memory registry: analysis_id -> set of connected WebSocket clients
_subscribers: dict[int, set[WebSocket]] = {}


async def websocket_endpoint(ws: WebSocket, analysis_id: int):
    """
    WebSocket endpoint: /ws/{analysis_id}
    The frontend connects here to receive real-time progress updates for a specific analysis.
    """
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
