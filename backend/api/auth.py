import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import settings
from .database import async_session
from .models import User

logger = logging.getLogger("api.auth")


@dataclass
class FallbackUser:
    """In-memory user when the DB is unavailable in local dev mode."""

    id: int = 1
    openId: str = "local-dev-user"
    name: str = "Local Developer"
    email: str = "dev@localhost"
    loginMethod: str = "local"
    role: str = "admin"
    createdAt: datetime = field(default_factory=lambda: datetime.now(UTC))
    updatedAt: datetime = field(default_factory=lambda: datetime.now(UTC))
    lastSignedIn: datetime = field(default_factory=lambda: datetime.now(UTC))


class AutoLoginMiddleware:
    """
    Pure ASGI middleware (not BaseHTTPMiddleware) to avoid breaking streaming
    responses like video file serving.

    In LOCAL_DEV_MODE, automatically authenticate every request as the dev user.
    Falls back to an in-memory user if the database is unavailable.
    Skips non-HTTP scopes (websocket) and static file paths (/uploads).
    """

    def __init__(self, app: ASGIApp):
        self.app = app
        self._dev_mode = settings.LOCAL_DEV_MODE is True
        if self._dev_mode:
            logger.warning(
                "AutoLoginMiddleware is ACTIVE — all requests auto-authenticated as dev user. "
                "Set LOCAL_DEV_MODE=false to disable."
            )

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "http" and self._dev_mode:
            path = scope.get("path", "")
            # Skip auth for static files — no user needed
            if not path.startswith("/uploads"):
                request = Request(scope)
                user = None

                # Try to get/create user in DB; fall back gracefully if DB is down
                if async_session is not None:
                    try:
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
                                    lastSignedIn=datetime.now(UTC),
                                )
                                db.add(user)
                                await db.commit()
                                await db.refresh(user)
                            else:
                                user.lastSignedIn = datetime.now(UTC)
                                await db.commit()
                                await db.refresh(user)
                    except Exception:
                        # DB is unavailable — use in-memory fallback
                        user = None

                # If no DB user (DB down or not configured), use in-memory fallback
                if user is None:
                    user = FallbackUser(openId=settings.OWNER_OPEN_ID)

                request.state.user = user

        await self.app(scope, receive, send)
