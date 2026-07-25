from __future__ import annotations

"""
Dual-DB Sync Service — Selective sync between local SQLite and remote PostgreSQL.

Strategy: Write-Behind Cache (Selective)
─────────────────────────────────────────
Sync Strategy:
  • queries, responses, agent_jobs, agent_runtime  →  Synced every 30s (high-frequency)
  • agents, model_connections                       →  Lazy-loaded with 300s TTL cache
  • users, auth_accounts                            →  Cloud-only (no local sync)

This avoids unnecessary bandwidth, reduces sync conflicts, and keeps sensitive
auth data secure in the cloud only.
"""

import asyncio
import logging
import time
from typing import Type

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.session import local_session_factory, remote_session_factory
from app.models.db_models import (
    Base,
    AgentRow,
    AgentRuntimeRow,
    ModelConnectionRow,
    QueryRow,
    ResponseRow,
    AgentJobRow,
)

logger = logging.getLogger("vizhi.sync")

# ── Tables to sync locally every sync_interval seconds ──────────────────────
# These are high-frequency write tables — they must be fast (local SQLite).
# Ordered by FK dependency: parent tables first.
SYNC_TABLES: list[Type[Base]] = [
    AgentRow,           # parent of queries/responses/agent_jobs
    AgentRuntimeRow,    # agent status/heartbeat — frequently updated
    ModelConnectionRow, # model config — frequently read on every inference
    QueryRow,           # high-frequency writes per API call
    ResponseRow,        # paired with every query
    AgentJobRow,        # agent job tracking
]

# ── Tables NOT synced locally (cloud-only) ───────────────────────────────────
# UserRow and AuthAccountRow are kept in the remote DB only.
# Reason: security (password hashes), real-time auth consistency, no sync conflicts.

# Per-table: additional unique constraint names for conflict resolution
EXTRA_UNIQUE_CONSTRAINTS: dict[str, list[str]] = {
    "agents": ["agents_agent_id_key"],
    "agent_jobs": ["agent_jobs_query_id_key"],
}


class SyncService:
    """
    Manages selective one-way sync from local SQLite → remote PostgreSQL (Supabase).

    Lifecycle:
      1. hydrate_local_from_remote()  — on container startup (empty local)
      2. start()                      — begins background sync loop
      3. stop()                       — final flush + cancel loop
    """

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._sync_count: int = 0
        self._last_error: str | None = None

    # ── Properties ──────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        """Check if remote database is configured and sync is enabled."""
        return (
            settings.sync_enabled
            and bool(settings.remote_database_url)
            and remote_session_factory is not None
        )

    @property
    def status(self) -> dict:
        """Return current sync status (useful for health endpoints)."""
        return {
            "configured": self.is_configured,
            "running": self._running,
            "sync_count": self._sync_count,
            "last_error": self._last_error,
            "synced_tables": [t.__tablename__ for t in SYNC_TABLES],
            "cloud_only_tables": ["users", "auth_accounts"],
        }

    # ── Startup: Hydrate local from remote ──────────────────────────

    async def hydrate_local_from_remote(self) -> None:
        """
        On container startup, if the local SQLite is empty,
        pull only the SYNC_TABLES data from remote PostgreSQL.

        users and auth_accounts are NOT hydrated — they stay cloud-only.
        """
        if not self.is_configured:
            logger.info("Remote DB not configured — skipping hydration")
            return

        # Use AgentRow count as proxy for "is local DB populated?"
        async with local_session_factory() as local_db:
            result = await local_db.execute(select(func.count(AgentRow.id)))
            local_count = result.scalar() or 0

        if local_count > 0:
            logger.info(
                f"Local DB already has {local_count} agents — skipping hydration"
            )
            return

        logger.info("Local DB is empty — hydrating selected tables from remote...")

        try:
            async with remote_session_factory() as remote_db:
                async with local_session_factory() as local_db:
                    total_rows = 0
                    for model_class in SYNC_TABLES:
                        count = await self._copy_table(
                            source_db=remote_db,
                            target_db=local_db,
                            model_class=model_class,
                            direction="remote → local",
                        )
                        total_rows += count
                    await local_db.commit()

            logger.info(f"✅ Hydration complete — {total_rows} total rows pulled")
            logger.info("ℹ️  users & auth_accounts are cloud-only — not hydrated locally")
        except Exception as e:
            logger.error(f"❌ Hydration failed: {e}", exc_info=True)
            self._last_error = f"Hydration failed: {e}"

    # ── Background Sync Loop ────────────────────────────────────────

    async def start(self) -> None:
        """Start the background sync loop."""
        if not self.is_configured:
            logger.info("Remote sync disabled — running local-only mode")
            return

        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        logger.info(
            f"🔄 Selective sync started (interval: {settings.sync_interval}s, "
            f"tables: {[t.__tablename__ for t in SYNC_TABLES]})"
        )

    async def stop(self) -> None:
        """Stop the sync loop and perform a final flush."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Final flush — push any remaining local data to remote
        if self.is_configured:
            logger.info("Performing final sync before shutdown...")
            try:
                await self._sync_all_tables()
                logger.info("✅ Final sync complete")
            except Exception as e:
                logger.error(f"❌ Final sync failed: {e}", exc_info=True)

    async def _sync_loop(self) -> None:
        """Periodically sync local changes to remote."""
        while self._running:
            try:
                await asyncio.sleep(settings.sync_interval)
                await self._sync_all_tables()
                self._sync_count += 1
                self._last_error = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync error: {e}", exc_info=True)
                self._last_error = str(e)
                # Don't crash the loop — wait and retry
                await asyncio.sleep(5)

    # ── Core Sync Logic ─────────────────────────────────────────────

    async def _sync_all_tables(self) -> None:
        """
        Push all SYNC_TABLES local data to remote (local → remote).

        users and auth_accounts are intentionally excluded — they are cloud-only.
        Each table gets its OWN remote session so that a failure in one table
        does NOT poison the session and block the remaining tables from syncing.
        """
        async with local_session_factory() as local_db:
            for model_class in SYNC_TABLES:
                try:
                    async with remote_session_factory() as remote_db:
                        await self._push_table(
                            local_db=local_db,
                            remote_db=remote_db,
                            model_class=model_class,
                        )
                        await remote_db.commit()
                except Exception as e:
                    logger.error(
                        f"Failed to sync {model_class.__tablename__}: {e}",
                        exc_info=True,
                    )
                    self._last_error = f"{model_class.__tablename__}: {e}"

    async def _push_table(
        self,
        local_db: AsyncSession,
        remote_db: AsyncSession,
        model_class: Type[Base],
    ) -> None:
        """
        Sync a single table from local → remote using PostgreSQL UPSERT.

        Executes one row at a time to avoid SQLAlchemy compilation issues.
        """
        local_result = await local_db.execute(select(model_class))
        local_rows = local_result.scalars().all()

        if not local_rows:
            return

        table = model_class.__table__
        pk_col_name = list(table.primary_key.columns)[0].name

        update_col_names = [
            col.name for col in table.columns if col.name != pk_col_name
        ]

        mapper = model_class.__mapper__
        col_to_attr: dict[str, str] = {}
        for prop in mapper.column_attrs:
            for col in prop.columns:
                col_to_attr[col.name] = prop.key

        upserted = 0
        skipped = 0
        for row in local_rows:
            row_dict = {
                col_name: getattr(row, attr_key)
                for col_name, attr_key in col_to_attr.items()
            }

            stmt = pg_insert(table).values(**row_dict)

            if update_col_names:
                stmt = stmt.on_conflict_do_update(
                    index_elements=[pk_col_name],
                    set_={col_name: stmt.excluded[col_name] for col_name in update_col_names},
                )
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=[pk_col_name])

            try:
                await remote_db.execute(stmt)
                upserted += 1
            except Exception as row_err:
                err_str = str(row_err)
                await remote_db.rollback()
                if "ForeignKeyViolationError" in err_str or "UniqueViolationError" in err_str:
                    skipped += 1
                elif "UndefinedColumnError" in err_str:
                    col_hint = err_str.split("column")[-1].split("of")[0].strip().strip('"')
                    logger.warning(
                        f"  ⚠️  Remote {model_class.__tablename__} is missing column "
                        f'"{col_hint}". Run migrations in Supabase SQL Editor to fix this.'
                    )
                    break
                else:
                    raise

        if upserted or skipped:
            logger.info(
                f"  ✅ {model_class.__tablename__}: "
                f"{upserted} upserted"
                + (f", {skipped} skipped" if skipped else "")
            )

    # ── Helpers ──────────────────────────────────────────────────────

    async def _copy_table(
        self,
        source_db: AsyncSession,
        target_db: AsyncSession,
        model_class: Type[Base],
        direction: str = "",
    ) -> int:
        """
        Copy all rows from source to target for a given model.
        Used only during hydration (startup).
        """
        result = await source_db.execute(select(model_class))
        rows = result.scalars().all()

        mapper = model_class.__mapper__
        col_to_attr: dict[str, str] = {}
        for prop in mapper.column_attrs:
            for col in prop.columns:
                col_to_attr[col.name] = prop.key

        count = 0
        for row in rows:
            row_dict = {
                col_name: getattr(row, attr_key)
                for col_name, attr_key in col_to_attr.items()
            }
            target_db.add(model_class(**row_dict))
            count += 1

        if count > 0:
            logger.info(f"  {direction} {model_class.__tablename__}: {count} rows")

        return count

    # ── Manual Sync (for API/admin endpoints) ────────────────────────

    async def force_sync(self) -> dict:
        """Trigger an immediate sync (useful for admin endpoints)."""
        if not self.is_configured:
            return {"status": "error", "message": "Remote DB not configured"}

        try:
            await self._sync_all_tables()
            self._sync_count += 1
            return {"status": "ok", "sync_count": self._sync_count}
        except Exception as e:
            self._last_error = str(e)
            return {"status": "error", "message": str(e)}


# ── Module-level singleton ──────────────────────────────────────────
sync_service = SyncService()
