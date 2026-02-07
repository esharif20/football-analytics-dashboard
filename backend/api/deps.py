from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from .database import async_session
from .models import User


async def get_db() -> AsyncSession:
    if async_session is None:
        raise HTTPException(status_code=503, detail="Database not available")
    async with async_session() as session:
        yield session


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """
    In local dev mode, auto-login returns a fixed dev user.
    The user is set on the request state by the auto-login middleware in auth.py.
    """
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
