"""pytest configuration — adds api and backend directories to sys.path.

- API_ROOT (backend/api/) → enables `from services.tactical import ...`
- BACKEND_ROOT (backend/) → enables `from api.routers.commentary import ...`
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

API_ROOT = Path(__file__).resolve().parent.parent  # backend/api/
BACKEND_ROOT = API_ROOT.parent  # backend/

for p in (str(API_ROOT), str(BACKEND_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Set LOCAL_DEV_MODE before importing app (so AutoLoginMiddleware activates)
os.environ.setdefault("LOCAL_DEV_MODE", "true")
os.environ.setdefault("JWT_SECRET", "ci-test-secret-not-for-prod")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://skip:skip@skip/skip")

from api.deps import get_db  # noqa: E402
from api.main import app  # noqa: E402


@pytest_asyncio.fixture
async def client():
    """AsyncClient with DB dependency overridden — no live DB needed."""

    async def override_get_db():
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
                all=MagicMock(return_value=[]),
            )
        )
        mock_db.commit = AsyncMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
