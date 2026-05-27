"""Async SQLAlchemy session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    # SQLite requires `check_same_thread=False` for async usage.
    connect_args={"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {},
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:  # type: ignore[misc]
    """FastAPI dependency that yields an async DB session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
