from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import settings

# Accept DATABASE_URL as-is — must be a valid SQLAlchemy async URL
# Local dev: postgresql+asyncpg://postgres:postgres@localhost:54322/postgres
_url = settings.DATABASE_URL

engine = create_async_engine(_url, echo=False, pool_pre_ping=True) if _url else None

async_session = (
    async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False) if engine else None
)


class Base(DeclarativeBase):
    pass
