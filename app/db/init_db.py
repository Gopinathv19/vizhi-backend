"""Database initialization – creates all tables on first run."""

from __future__ import annotations

from app.db.session import engine
from app.models.db_models import Base


async def init_db() -> None:
    """Create tables if they don't exist yet."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
