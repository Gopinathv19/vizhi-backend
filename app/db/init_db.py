"""Database initialization – creates all tables on first run."""

from __future__ import annotations

from sqlalchemy import text

from app.config.settings import settings
from app.db.session import engine
from app.models.db_models import Base


async def _sync_sqlite_agents_schema(conn) -> None:
    """Remove stale dev columns left behind by create_all-only schema changes."""
    result = await conn.execute(text("PRAGMA table_info(agents)"))
    columns = {row[1] for row in result.fetchall()}
    stale_columns = {"owner", "preferred_model"} & columns
    if not stale_columns:
        return

    await conn.exec_driver_sql("PRAGMA foreign_keys=OFF")
    await conn.exec_driver_sql(
        """
        CREATE TABLE agents_new (
            id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            api_key_hash TEXT NOT NULL,
            masked_key TEXT NOT NULL,
            tags TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
            PRIMARY KEY (id),
            UNIQUE (agent_id)
        )
        """
    )
    await conn.exec_driver_sql(
        """
        INSERT INTO agents_new (
            id,
            agent_id,
            name,
            description,
            api_key_hash,
            masked_key,
            tags,
            status,
            created_at,
            updated_at
        )
        SELECT
            id,
            agent_id,
            name,
            description,
            api_key_hash,
            masked_key,
            tags,
            status,
            created_at,
            updated_at
        FROM agents
        """
    )
    await conn.exec_driver_sql("DROP TABLE agents")
    await conn.exec_driver_sql("ALTER TABLE agents_new RENAME TO agents")
    await conn.exec_driver_sql("PRAGMA foreign_keys=ON")


async def init_db() -> None:
    """Create tables if they don't exist yet."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if settings.database_url.startswith("sqlite"):
            await _sync_sqlite_agents_schema(conn)
