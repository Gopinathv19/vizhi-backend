from __future__ import annotations

"""
Lazy TTL Cache Service — Read-through cache for low-frequency config tables.

Strategy:
  • agents          →  Cached locally for 300s (5 min) TTL
  • model_connections →  Cached locally for 300s (5 min) TTL

On first access: fetch from cloud (remote PostgreSQL) and store locally.
After TTL expires: next request refreshes from cloud automatically.
On write (create/update/delete): write directly to cloud, then invalidate cache.

This pattern ensures:
  - Fast reads (local SQLite) on the hot path
  - Fresh data after changes propagate within 5 minutes
  - No constant sync overhead for rarely-changing config tables
"""

import asyncio
import logging
import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import local_session_factory, remote_session_factory
from app.models.db_models import AgentRow, ModelConnectionRow

logger = logging.getLogger("vizhi.cache")

# Default TTL in seconds (5 minutes)
DEFAULT_TTL = 300


class TableCache:
    """
    Simple in-memory TTL tracker per table.
    Tracks the last time a table was hydrated from remote.
    """

    def __init__(self, ttl: int = DEFAULT_TTL) -> None:
        self._ttl = ttl
        self._last_hydrated: float = 0.0
        self._lock = asyncio.Lock()

    def is_stale(self) -> bool:
        """Return True if cache TTL has expired and data needs refresh."""
        return (time.monotonic() - self._last_hydrated) >= self._ttl

    def mark_fresh(self) -> None:
        """Reset the TTL timer after a successful hydration."""
        self._last_hydrated = time.monotonic()

    def invalidate(self) -> None:
        """
        Mark cache as freshly populated from local write.

        After a local write (create/update/delete/rotate), we do NOT want the
        next read to pull from remote — because remote hasn't received the sync
        yet (sync runs every 30s). Instead we reset the TTL timer to "just now"
        so the next read trusts the local SQLite data we just wrote.

        Remote will receive the update on the next background sync cycle.
        """
        self._last_hydrated = time.monotonic()

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock


class CacheService:
    """
    Manages read-through TTL caching for agents and model_connections.

    Usage in API routes:
        await cache_service.ensure_agents_fresh()      # before reading agents locally
        await cache_service.ensure_models_fresh()      # before reading models locally
        cache_service.invalidate_agents()              # after writing an agent
        cache_service.invalidate_models()              # after writing a model
    """

    def __init__(self) -> None:
        self._agents_cache = TableCache(ttl=DEFAULT_TTL)
        self._models_cache = TableCache(ttl=DEFAULT_TTL)

    # ── Public: Ensure fresh data ────────────────────────────────────

    async def ensure_agents_fresh(self) -> None:
        """
        If the local agents cache is stale, refresh it from remote.
        No-op if remote is not configured or cache is still fresh.
        """
        if not self._is_remote_available():
            return

        if not self._agents_cache.is_stale():
            return

        async with self._agents_cache.lock:
            # Double-check after acquiring lock (another task may have refreshed)
            if not self._agents_cache.is_stale():
                return
            await self._refresh_table(AgentRow)
            self._agents_cache.mark_fresh()

    async def ensure_models_fresh(self) -> None:
        """
        If the local model_connections cache is stale, refresh it from remote.
        No-op if remote is not configured or cache is still fresh.
        """
        if not self._is_remote_available():
            return

        if not self._models_cache.is_stale():
            return

        async with self._models_cache.lock:
            if not self._models_cache.is_stale():
                return
            await self._refresh_table(ModelConnectionRow)
            self._models_cache.mark_fresh()

    # ── Public: Cache invalidation ──────────────────────────────────

    def invalidate_agents(self) -> None:
        """Call this after any write to agents (create/update/delete/rotate)."""
        self._agents_cache.invalidate()
        logger.debug("Agent cache invalidated")

    def invalidate_models(self) -> None:
        """Call this after any write to model_connections."""
        self._models_cache.invalidate()
        logger.debug("Model connections cache invalidated")

    # ── Internal ─────────────────────────────────────────────────────

    def _is_remote_available(self) -> bool:
        return remote_session_factory is not None

    async def _refresh_table(self, model_class: type) -> None:
        """
        Pull all rows from remote and upsert into local SQLite.
        Uses INSERT OR REPLACE semantics (SQLite) to handle conflicts.
        """
        table_name = model_class.__tablename__
        try:
            async with remote_session_factory() as remote_db:
                result = await remote_db.execute(select(model_class))
                remote_rows = result.scalars().all()

            if not remote_rows:
                logger.debug(f"Cache refresh: no rows in remote {table_name}")
                return

            # Build column → attribute key mapper
            mapper = model_class.__mapper__
            col_to_attr: dict[str, str] = {}
            for prop in mapper.column_attrs:
                for col in prop.columns:
                    col_to_attr[col.name] = prop.key

            async with local_session_factory() as local_db:
                for row in remote_rows:
                    row_dict = {
                        col_name: getattr(row, attr_key)
                        for col_name, attr_key in col_to_attr.items()
                    }
                    # Check if row already exists locally
                    pk_col_name = list(model_class.__table__.primary_key.columns)[0].name
                    pk_value = row_dict[pk_col_name]
                    existing = await local_db.get(model_class, pk_value)

                    if existing:
                        # Update all columns
                        for col_name, attr_key in col_to_attr.items():
                            if col_name != pk_col_name:
                                setattr(existing, attr_key, getattr(row, attr_key))
                    else:
                        local_db.add(model_class(**row_dict))

                await local_db.commit()

            logger.info(f"📦 Cache refreshed: {table_name} ({len(remote_rows)} rows from remote)")

        except Exception as e:
            logger.error(f"Cache refresh failed for {table_name}: {e}", exc_info=True)
            # Don't raise — fall back to stale local data


# ── Module-level singleton ──────────────────────────────────────────
cache_service = CacheService()
