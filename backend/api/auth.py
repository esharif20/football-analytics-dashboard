from datetime import datetime
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from sqlalchemy import select, insert, update

from .config import settings
from .database import async_session
from .models import User


class AutoLoginMiddleware(BaseHTTPMiddleware):
    """
    In LOCAL_DEV_MODE, automatically authenticate every request as the dev user.
    Finds or creates the user with OWNER_OPEN_ID on first request.
    """

    async def dispatch(self, request: Request, call_next):
        if settings.LOCAL_DEV_MODE and async_session is not None:
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

        response = await call_next(request)
        return response
