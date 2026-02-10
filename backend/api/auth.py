from datetime import datetime
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send
from sqlalchemy import select

from .config import settings
from .database import async_session
from .models import User


class AutoLoginMiddleware:
    """
    Pure ASGI middleware (not BaseHTTPMiddleware) to avoid breaking streaming
    responses like video file serving.

    In LOCAL_DEV_MODE, automatically authenticate every request as the dev user.
    Skips non-HTTP scopes (websocket) and static file paths (/uploads).
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http" and settings.LOCAL_DEV_MODE and async_session is not None:
            path = scope.get("path", "")
            # Skip auth for static files — no user needed
            if not path.startswith("/uploads"):
                request = Request(scope)
                async with async_session() as db:
                    result = await db.execute(
                        select(User).where(User.openId == settings.OWNER_OPEN_ID).limit(1)
                    )
                    user = result.scalar_one_or_none()

                    if user is None:
                        user = User(
                            openId=settings.OWNER_OPEN_ID,
                            name="Local Developer",
                            email="dev@localhost",
                            loginMethod="local",
                            role="admin",
                            lastSignedIn=datetime.utcnow(),
                        )
                        db.add(user)
                        await db.commit()
                        await db.refresh(user)
                    else:
                        user.lastSignedIn = datetime.utcnow()
                        await db.commit()
                        await db.refresh(user)

                    request.state.user = user

        await self.app(scope, receive, send)
