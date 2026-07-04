"""
Dual-DB Sync Service — Background sync between local SQLite and remote PostgreSQL.

Strategy: Write-Behind Cache
─────────────────────────────
- All API reads/writes go to local SQLite first (fast, zero-latency)
- Background task periodically pushes changes to remote PostgreSQL (Supabase)
- On startup, if local is empty (fresh Docker container), hydrate from remote
- On shutdown, final flush ensures no data loss

This pattern is commonly known as:
  • Write-Behind Cache / Write-Through Cache
  • Edge-to-Cloud Sync (PouchDB↔CouchDB, Firebase offline, Realm Sync)
  • CQRS with local materialised view
"""

from __future__ import annotations

import asyncio
import logging
import datetime as _dt
from typing import Type

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.db.session import local_session_factory, remote_session_factory
from app.models.db_models import (
    Base,
    UserRow,
    AuthAccountRow,
    AgentRow,
    AgentRuntimeRow,
    ModelConnectionRow,
    QueryRow,
    ResponseRow,
    AgentJobRow,
)

logger = logging.getLogger("vizhi.sync")

# Tables to sync, ordered by dependency (parents first so FK constraints pass)
SYNC_TABLES: list[Type[Base]] = [
    UserRow,
    AuthAccountRow,
    AgentRow,
    AgentRuntimeRow,
    ModelConnectionRow,
    QueryRow,
    ResponseRow,
    AgentJobRow,
]


class SyncService:
    """
    Manages bidirectional sync between local SQLite and remote PostgreSQL.

    Lifecycle:
      1. hydrate_local_from_remote()  — on container startup
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
        }

    # ── Startup: Hydrate local from remote ──────────────────────────

    async def hydrate_local_from_remote(self) -> None:
        """
        On container startup, merge data from remote PostgreSQL (Supabase)
        into the local SQLite cache.
        """
        if not self.is_configured:
            logger.info("Remote DB not configured — skipping hydration")
            return

        async with local_session_factory() as local_db:
            result = await local_db.execute(select(func.count(UserRow.id)))
            local_count = result.scalar() or 0

        if local_count > 0:
            logger.info(
                f"Local DB already has {local_count} users — merging remote updates..."
            )
        else:
            logger.info("Local DB is empty — hydrating from remote...")

        try:
            async with remote_session_factory() as remote_db:
                async with local_session_factory() as local_db:
                    total_rows = 0
                    id_maps: dict[type[Base], dict[str, str]] = {}
                    for model_class in SYNC_TABLES:
                        count = await self._sync_table(
                            source_db=remote_db,
                            target_db=local_db,
                            model_class=model_class,
                            id_maps=id_maps,
                            direction="remote → local",
                        )
                        total_rows += count
                    await local_db.commit()

            logger.info(f"✅ Hydration complete — {total_rows} total rows merged")
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
            f"🔄 Sync started (interval: {settings.sync_interval}s, "
            f"batch_size: {settings.sync_batch_size})"
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
        """Merge remote changes into local, then push local changes to remote."""
        async with local_session_factory() as local_db:
            async with remote_session_factory() as remote_db:
                try:
                    pull_id_maps: dict[type[Base], dict[str, str]] = {}
                    for model_class in SYNC_TABLES:
                        await self._sync_table(
                            source_db=remote_db,
                            target_db=local_db,
                            model_class=model_class,
                            id_maps=pull_id_maps,
                            direction="remote → local",
                        )
                    await local_db.commit()

                    push_id_maps: dict[type[Base], dict[str, str]] = {}
                    for model_class in SYNC_TABLES:
                        await self._sync_table(
                            source_db=local_db,
                            target_db=remote_db,
                            model_class=model_class,
                            id_maps=push_id_maps,
                            direction="local → remote",
                        )
                    await remote_db.commit()
                except Exception:
                    await local_db.rollback()
                    await remote_db.rollback()
                    raise

    async def _sync_table(
        self,
        source_db: AsyncSession,
        target_db: AsyncSession,
        model_class: Type[Base],
        id_maps: dict[type[Base], dict[str, str]],
        direction: str = "",
    ) -> int:
        """
        Sync a single table using application-level upsert logic.

        Rows are first matched by primary key. For tables with unique business
        keys, rows are also matched by that identity so local/cloud databases
        with different generated primary keys can still converge.
        """
        table_name = model_class.__tablename__
        pk_col_name = list(model_class.__table__.primary_key.columns)[0].name

        source_result = await source_db.execute(select(model_class))
        source_rows = source_result.scalars().all()

        if not source_rows:
            return 0

        inserted = updated = 0

        for row in source_rows:
            row_dict = {
                col.name: getattr(row, col.name)
                for col in model_class.__table__.columns
            }
            self._apply_id_maps(row_dict, id_maps)
            row_pk = row_dict[pk_col_name]

            target_row = await self._find_target_row(
                target_db=target_db,
                model_class=model_class,
                row_dict=row_dict,
                pk_col_name=pk_col_name,
            )
            if target_row is None:
                target_db.add(model_class(**row_dict))
                inserted += 1
            else:
                target_pk = getattr(target_row, pk_col_name)
                if target_pk != row_pk:
                    id_maps.setdefault(model_class, {})[row_pk] = target_pk
                self._update_target_row(
                    target_row=target_row,
                    row_dict=row_dict,
                    pk_col_name=pk_col_name,
                )
                updated += 1

        if inserted or updated:
            logger.debug(
                f"  {direction} {table_name}: +{inserted} inserted, ~{updated} updated"
            )

        return inserted + updated

    # ── Helpers ──────────────────────────────────────────────────────

    async def _find_target_row(
        self,
        *,
        target_db: AsyncSession,
        model_class: Type[Base],
        row_dict: dict,
        pk_col_name: str,
    ) -> Base | None:
        pk_attr = getattr(model_class, pk_col_name)
        result = await target_db.execute(
            select(model_class).where(pk_attr == row_dict[pk_col_name])
        )
        target_row = result.scalars().first()
        if target_row is not None:
            return target_row

        identity = self._natural_identity(model_class, row_dict)
        if not identity:
            return None

        stmt = select(model_class)
        for col_name, value in identity.items():
            stmt = stmt.where(getattr(model_class, col_name) == value)
        result = await target_db.execute(stmt)
        return result.scalars().first()

    def _natural_identity(
        self,
        model_class: Type[Base],
        row_dict: dict,
    ) -> dict[str, object] | None:
        if model_class is UserRow:
            return {"email": row_dict["email"]}
        if model_class is AuthAccountRow:
            if row_dict.get("provider_user_id") is not None:
                return {
                    "provider": row_dict["provider"],
                    "provider_user_id": row_dict["provider_user_id"],
                }
            return {
                "user_id": row_dict["user_id"],
                "provider": row_dict["provider"],
            }
        if model_class is AgentRow:
            return {"agent_id": row_dict["agent_id"]}
        if model_class is AgentRuntimeRow:
            return {"agent_id": row_dict["agent_id"]}
        if model_class is AgentJobRow:
            return {"query_id": row_dict["query_id"]}
        return None

    def _apply_id_maps(
        self,
        row_dict: dict,
        id_maps: dict[type[Base], dict[str, str]],
    ) -> None:
        user_id = row_dict.get("user_id")
        if user_id in id_maps.get(UserRow, {}):
            row_dict["user_id"] = id_maps[UserRow][user_id]

        agent_id = row_dict.get("agent_id")
        if agent_id in id_maps.get(AgentRow, {}):
            row_dict["agent_id"] = id_maps[AgentRow][agent_id]

        query_id = row_dict.get("query_id")
        if query_id in id_maps.get(QueryRow, {}):
            row_dict["query_id"] = id_maps[QueryRow][query_id]

    def _update_target_row(
        self,
        *,
        target_row: Base,
        row_dict: dict,
        pk_col_name: str,
    ) -> None:
        for column in target_row.__table__.columns:
            if column.name == pk_col_name:
                continue
            setattr(target_row, column.name, row_dict[column.name])

    # ── Manual Sync (for API/admin endpoints) ───────────────────────

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
