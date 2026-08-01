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
from typing import Type

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
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

# Per-table: additional unique constraint names that must be handled with
# ON CONFLICT DO NOTHING (beyond the primary key which is always upserted).
# If the remote has a UNIQUE constraint on a column OTHER than the PK, and a
# local row conflicts on that column with a DIFFERENT PK value (e.g. same email
# registered twice via different auth flows), we skip that row rather than fail.
EXTRA_UNIQUE_CONSTRAINTS: dict[str, list[str]] = {
    "users": ["users_email_key"],
    "agents": ["agents_agent_id_key"],
    "agent_jobs": ["agent_jobs_query_id_key"],
    "auth_accounts": ["uq_auth_provider_user", "uq_auth_user_provider"],
}


class SyncService:
    """
    Manages one-way sync from local SQLite → remote PostgreSQL (Supabase).

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
        }

    # ── Startup: Hydrate local from remote ──────────────────────────

    async def hydrate_local_from_remote(self) -> None:
        """
        On container startup, if the local SQLite is empty,
        pull all data from remote PostgreSQL (Supabase).

        Skipped if local already has data — avoids overwriting live data
        on a running instance that merely restarted.
        """
        if not self.is_configured:
            logger.info("Remote DB not configured — skipping hydration")
            return

        # Use user count as a proxy for "is local DB populated?"
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
        """
        Push all local data to remote (local → remote).

        Each table gets its OWN remote session so that a failure in one table
        (e.g. a unique constraint violation) does NOT poison the session and
        block the remaining tables from syncing.
        """
        async with local_session_factory() as local_db:
            for model_class in SYNC_TABLES:
                # Fresh session per table — errors in one table cannot
                # contaminate subsequent tables via a shared rolled-back session.
                try:
                    async with remote_session_factory() as remote_db:
                        await self._push_table(
                            local_db=local_db,
                            remote_db=remote_db,
                            model_class=model_class,
                            id_maps=pull_id_maps,
                            direction="remote → local",
                        )
                        await remote_db.commit()
                except Exception as e:
                    # Log and continue — don't let one bad table stop the rest
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
        Sync a single table from local → remote using a true PostgreSQL UPSERT.

        Uses INSERT ... ON CONFLICT (pk) DO UPDATE SET ... executed one row at a
        time to avoid SQLAlchemy compilation issues with list-valued .values().
        Each row is safely upserted without unique constraint errors.
        """
        # Fetch all local rows
        local_result = await local_db.execute(select(model_class))
        local_rows = local_result.scalars().all()

        if not local_rows:
            return

        table = model_class.__table__
        pk_col_name = list(table.primary_key.columns)[0].name

        # Column names to update on conflict (everything except the PK)
        update_col_names = [
            col.name for col in table.columns if col.name != pk_col_name
        ]

        # Pre-build the mapper: db column name → ORM attribute key
        # This correctly resolves aliases like metadata_ → "metadata" column.
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

            # Build a fresh statement per row — avoids the MetaData/_is_bind_parameter
            # bug that occurs when pg_insert().values(list_of_dicts) is used with
            # .on_conflict_do_update() referencing stmt.excluded on a pre-built stmt.
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
                # FK violation: parent row not yet in remote — skip silently,
                # it will be retried on the next sync cycle once the parent syncs.
                # UniqueViolation on a NON-PK constraint (e.g. users_email_key):
                # The same email exists in remote under a different id.
                # Skip the local row — the remote version takes precedence.
                # Schema error (UndefinedColumn): remote table is missing a column —
                # user must run the migration SQL in Supabase SQL Editor.
                await remote_db.rollback()
                if "ForeignKeyViolationError" in err_str or "UniqueViolationError" in err_str:
                    skipped += 1
                elif "UndefinedColumnError" in err_str:
                    col_hint = err_str.split("column")[-1].split("of")[0].strip().strip('"')
                    logger.warning(
                        f"  ⚠️  Remote {model_class.__tablename__} is missing column "
                        f'"{col_hint}". Run migrations/remote_supabase_schema_sync.sql '
                        f"in Supabase SQL Editor to fix this."
                    )
                    # Skip remaining rows for this table — all will fail the same way
                    break
                else:
                    raise

        if upserted or skipped:
            logger.info(
                f"  ✅ {model_class.__tablename__}: "
                f"{upserted} upserted"
                + (f", {skipped} skipped (FK not yet synced)" if skipped else "")
            )

    # ── Helpers ──────────────────────────────────────────────────────

    async def _copy_table(
        self,
        source_db: AsyncSession,
        target_db: AsyncSession,
        model_class: Type[Base],
        id_maps: dict[type[Base], dict[str, str]],
        direction: str = "",
    ) -> int:
        """
        Copy all rows from source to target for a given model.
        Used only during hydration (startup). Does not check for conflicts —
        assumes target is empty.
        """
        result = await source_db.execute(select(model_class))
        rows = result.scalars().all()

        # Build mapper: db column name → ORM attribute key (handles aliases like metadata_)
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
