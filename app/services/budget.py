"""Token Budget Enforcement Service.

Provides pre-flight budget checks for both agent tokens and model tokens.

Budget limits are stored on AgentRow and ModelConnectionRow:
  - budget_usd     : maximum cumulative USD spend (None = unlimited)
  - budget_tokens  : maximum cumulative total tokens (None = unlimited)
  - budget_reset_at: ISO-8601 datetime after which spending counters reset
                     (None = lifetime / never resets)

Actual "spent so far" figures are derived from the live responses/queries
tables so they are always accurate even across restarts.

Usage:
    from app.services.budget import check_agent_budget, check_model_budget

    # raises HTTP 429 if the budget is exceeded
    await check_agent_budget(db, agent_id="ag_xxx")
    await check_model_budget(db, model_connection_id="mc_xxx")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import AgentRow, ModelConnectionRow, QueryRow, ResponseRow

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_reset_at(reset_at_str: str | None) -> datetime | None:
    """Parse an ISO-8601 budget_reset_at string into a timezone-aware datetime."""
    if not reset_at_str:
        return None
    try:
        dt = datetime.fromisoformat(reset_at_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


async def _spent_for_agent(
    db: AsyncSession,
    agent_id: str,
    since: datetime | None,
) -> tuple[float, int]:
    """Return (total_cost_usd, total_tokens) spent by agent_id since `since`.
    If `since` is None, return all-time totals.
    """
    # Join queries → responses on query_id, filter by agent_id
    base_filter = QueryRow.agent_id == agent_id
    if since:
        base_filter = base_filter & (QueryRow.timestamp >= since)

    result = await db.execute(
        select(
            func.coalesce(func.sum(ResponseRow.estimated_cost), 0.0),
            func.coalesce(
                func.sum(ResponseRow.input_tokens + ResponseRow.output_tokens), 0
            ),
        )
        .join(QueryRow, ResponseRow.query_id == QueryRow.id)
        .where(base_filter)
        .where(ResponseRow.status_code == 200)
    )
    row = result.one()
    return float(row[0]), int(row[1])


async def _spent_for_model(
    db: AsyncSession,
    model_connection_id: str,
    since: datetime | None,
) -> tuple[float, int]:
    """Return (total_cost_usd, total_tokens) logged against a model connection
    principal_id (which equals the model connection's ID as stored in queries.agent_id
    when a model-token is used directly).
    """
    base_filter = QueryRow.agent_id == model_connection_id
    if since:
        base_filter = base_filter & (QueryRow.timestamp >= since)

    result = await db.execute(
        select(
            func.coalesce(func.sum(ResponseRow.estimated_cost), 0.0),
            func.coalesce(
                func.sum(ResponseRow.input_tokens + ResponseRow.output_tokens), 0
            ),
        )
        .join(QueryRow, ResponseRow.query_id == QueryRow.id)
        .where(base_filter)
        .where(ResponseRow.status_code == 200)
    )
    row = result.one()
    return float(row[0]), int(row[1])


# ── Public API ───────────────────────────────────────────────────────────

async def check_agent_budget(db: AsyncSession, agent_id: str) -> None:
    """Raise HTTP 429 if the agent has exceeded its configured budget.

    This is a pure pre-flight check — no writes are performed.
    """
    result = await db.execute(
        select(AgentRow).where(AgentRow.agent_id == agent_id)
    )
    agent = result.scalars().first()
    if agent is None:
        return  # agent not found — let auth layer handle it

    budget_usd = agent.budget_usd
    budget_tokens = agent.budget_tokens
    if budget_usd is None and budget_tokens is None:
        return  # no budget configured — unlimited

    reset_at = _parse_reset_at(agent.budget_reset_at)
    # If reset window has passed, the budget effectively resets to 0 spent
    since = reset_at if (reset_at and reset_at <= _now_utc()) else None

    spent_usd, spent_tokens = await _spent_for_agent(db, agent_id, since)

    if budget_usd is not None and spent_usd >= budget_usd:
        logger.warning(
            "Agent %s exceeded USD budget: spent=%.4f limit=%.4f",
            agent_id, spent_usd, budget_usd,
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "budget_exceeded",
                "scope": "agent",
                "type": "cost",
                "limit_usd": budget_usd,
                "spent_usd": round(spent_usd, 6),
                "message": (
                    f"Agent '{agent.name}' has reached its USD spending limit "
                    f"(${budget_usd:.4f}). Spent: ${spent_usd:.4f}."
                ),
            },
        )

    if budget_tokens is not None and spent_tokens >= budget_tokens:
        logger.warning(
            "Agent %s exceeded token budget: spent=%d limit=%d",
            agent_id, spent_tokens, budget_tokens,
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "budget_exceeded",
                "scope": "agent",
                "type": "tokens",
                "limit_tokens": budget_tokens,
                "spent_tokens": spent_tokens,
                "message": (
                    f"Agent '{agent.name}' has reached its token limit "
                    f"({budget_tokens:,} tokens). Spent: {spent_tokens:,}."
                ),
            },
        )


async def check_model_budget(
    db: AsyncSession, model_connection_id: str, model_connection_row: ModelConnectionRow | None = None
) -> None:
    """Raise HTTP 429 if the model-token has exceeded its configured budget.

    Pass `model_connection_row` directly if it's already been fetched to avoid
    an extra DB round-trip.
    """
    mc = model_connection_row
    if mc is None:
        result = await db.execute(
            select(ModelConnectionRow).where(ModelConnectionRow.id == model_connection_id)
        )
        mc = result.scalars().first()
    if mc is None:
        return  # not found — let auth layer handle it

    budget_usd = mc.budget_usd
    budget_tokens = mc.budget_tokens
    if budget_usd is None and budget_tokens is None:
        return  # unlimited

    reset_at = _parse_reset_at(mc.budget_reset_at)
    since = reset_at if (reset_at and reset_at <= _now_utc()) else None

    spent_usd, spent_tokens = await _spent_for_model(db, model_connection_id, since)

    label = mc.token_name or mc.model_name

    if budget_usd is not None and spent_usd >= budget_usd:
        logger.warning(
            "ModelConnection %s exceeded USD budget: spent=%.4f limit=%.4f",
            model_connection_id, spent_usd, budget_usd,
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "budget_exceeded",
                "scope": "model_token",
                "type": "cost",
                "limit_usd": budget_usd,
                "spent_usd": round(spent_usd, 6),
                "message": (
                    f"Model token '{label}' has reached its USD spending limit "
                    f"(${budget_usd:.4f}). Spent: ${spent_usd:.4f}."
                ),
            },
        )

    if budget_tokens is not None and spent_tokens >= budget_tokens:
        logger.warning(
            "ModelConnection %s exceeded token budget: spent=%d limit=%d",
            model_connection_id, spent_tokens, budget_tokens,
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "budget_exceeded",
                "scope": "model_token",
                "type": "tokens",
                "limit_tokens": budget_tokens,
                "spent_tokens": spent_tokens,
                "message": (
                    f"Model token '{label}' has reached its token limit "
                    f"({budget_tokens:,} tokens). Spent: {spent_tokens:,}."
                ),
            },
        )


async def get_agent_budget_status(
    db: AsyncSession, agent_id: str
) -> dict:
    """Return current budget usage for an agent (used by GET /v1/agents/:id/budget)."""
    result = await db.execute(select(AgentRow).where(AgentRow.agent_id == agent_id))
    agent = result.scalars().first()
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    reset_at = _parse_reset_at(agent.budget_reset_at)
    since = reset_at if (reset_at and reset_at <= _now_utc()) else None
    spent_usd, spent_tokens = await _spent_for_agent(db, agent_id, since)

    return {
        "agent_id": agent_id,
        "budget_usd": agent.budget_usd,
        "budget_tokens": agent.budget_tokens,
        "budget_reset_at": agent.budget_reset_at,
        "spent_usd": round(spent_usd, 6),
        "spent_tokens": spent_tokens,
        "remaining_usd": round(agent.budget_usd - spent_usd, 6) if agent.budget_usd is not None else None,
        "remaining_tokens": (agent.budget_tokens - spent_tokens) if agent.budget_tokens is not None else None,
        "exceeded": (
            (agent.budget_usd is not None and spent_usd >= agent.budget_usd)
            or (agent.budget_tokens is not None and spent_tokens >= agent.budget_tokens)
        ),
    }


async def get_model_budget_status(
    db: AsyncSession, model_connection_id: str
) -> dict:
    """Return current budget usage for a model token."""
    result = await db.execute(
        select(ModelConnectionRow).where(ModelConnectionRow.id == model_connection_id)
    )
    mc = result.scalars().first()
    if mc is None:
        raise HTTPException(status_code=404, detail="Model connection not found")

    reset_at = _parse_reset_at(mc.budget_reset_at)
    since = reset_at if (reset_at and reset_at <= _now_utc()) else None
    spent_usd, spent_tokens = await _spent_for_model(db, model_connection_id, since)

    return {
        "model_connection_id": model_connection_id,
        "budget_usd": mc.budget_usd,
        "budget_tokens": mc.budget_tokens,
        "budget_reset_at": mc.budget_reset_at,
        "spent_usd": round(spent_usd, 6),
        "spent_tokens": spent_tokens,
        "remaining_usd": round(mc.budget_usd - spent_usd, 6) if mc.budget_usd is not None else None,
        "remaining_tokens": (mc.budget_tokens - spent_tokens) if mc.budget_tokens is not None else None,
        "exceeded": (
            (mc.budget_usd is not None and spent_usd >= mc.budget_usd)
            or (mc.budget_tokens is not None and spent_tokens >= mc.budget_tokens)
        ),
    }
