from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from .config import settings

# Convert mysql:// to mysql+aiomysql:// for async support
_url = settings.DATABASE_URL
if _url.startswith("mysql://"):
    _url = _url.replace("mysql://", "mysql+aiomysql://", 1)
elif _url.startswith("mysql2://"):
    _url = _url.replace("mysql2://", "mysql+aiomysql://", 1)

engine = create_async_engine(_url, echo=False, pool_pre_ping=True) if _url else None

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False) if engine else None


class Base(DeclarativeBase):
    pass
