"""Persistence service — saves queries and responses to the database."""

from __future__ import annotations

import json
import uuid
import datetime as _dt

from sqlalchemy import case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import (
    AgentJobRow,
    AgentRuntimeRow,
    AgentRow,
    ModelConnectionRow,
    QueryRow,
    ResponseRow,
)
from app.providers.base import ProviderResponse
from app.schemas.responses import (
    DashboardResponse,
    DashboardTotals,
    MetricPoint,
    MetricsResponse,
    RequestEventResponse,
)


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _id() -> str:
    return uuid.uuid4().hex[:12]


# ── Write operations ────────────────────────────────────────────────────


async def persist_query(
    db: AsyncSession,
    *,
    agent_id: str,
    user_id: str | None,
    provider: str,
    model: str,
    sdk_type: str | None,
    messages: list[dict],
    endpoint: str = "/v1/chat/completions",
) -> QueryRow:
    """Save an incoming query before calling the provider."""
    row = QueryRow(
        id=f"q_{_id()}",
        user_id=user_id,
        agent_id=agent_id,
        provider=provider,
        model=model,
        sdk_type=sdk_type,
        input_messages=json.dumps(messages),
        endpoint=endpoint,
        timestamp=_utcnow(),
    )
    db.add(row)
    await db.flush()
    return row


async def persist_response(
    db: AsyncSession,
    *,
    query_id: str,
    provider_response: ProviderResponse | None = None,
    status_code: int = 200,
    error_message: str | None = None,
    latency_ms: int = 0,
) -> ResponseRow:
    """Save the provider response (or error) after the call."""
    row = ResponseRow(
        id=f"r_{_id()}",
        query_id=query_id,
        response=json.dumps(provider_response.raw_response) if provider_response else None,
        latency_ms=provider_response.latency_ms if provider_response else latency_ms,
        input_tokens=provider_response.input_tokens if provider_response else 0,
        output_tokens=provider_response.output_tokens if provider_response else 0,
        status_code=status_code,
        error_message=error_message,
        estimated_cost=_estimate_cost(provider_response) if provider_response else 0.0,
        timestamp=_utcnow(),
    )
    db.add(row)
    await db.flush()

    # Increment model connection usage count and update stats.
    if provider_response and status_code < 400:
        result = await db.execute(
            select(ModelConnectionRow).where(
                ModelConnectionRow.model_name == provider_response.model,
                ModelConnectionRow.status == "active",
            )
        )
        mc = result.scalars().first()
        if mc:
            mc.usage_count += 1
            mc.last_used_at = _utcnow()
            mc.total_tokens_consumed += (provider_response.input_tokens + provider_response.output_tokens)
            mc.total_cost += row.estimated_cost

    return row


async def upsert_agent_runtime(
    db: AsyncSession,
    *,
    agent_id: str,
    device_name: str = "",
    os_name: str = "",
    agent_version: str = "",
    status: str = "online",
    available_engines: list[str] | None = None,
) -> AgentRuntimeRow:
    result = await db.execute(
        select(AgentRuntimeRow).where(AgentRuntimeRow.agent_id == agent_id)
    )
    row = result.scalars().first()
    if row is None:
        row = AgentRuntimeRow(
            agent_id=agent_id,
            device_name=device_name,
            os_name=os_name,
            agent_version=agent_version,
            status=status,
            last_heartbeat=_utcnow(),
            available_engines=json.dumps(available_engines or []),
        )
        db.add(row)
    else:
        row.device_name = device_name or row.device_name
        row.os_name = os_name or row.os_name
        row.agent_version = agent_version or row.agent_version
        row.status = status
        row.last_heartbeat = _utcnow()
        if available_engines is not None:
            row.available_engines = json.dumps(available_engines)
    await db.flush()
    return row


async def persist_agent_job(
    db: AsyncSession,
    *,
    query_id: str,
    user_id: str | None,
    agent_id: str,
    provider: str,
    model: str,
    sdk_type: str | None,
    messages: list[dict],
    endpoint: str = "/v1/chat/completions",
    kind: str = "chat",
    stream: bool = False,
    metadata: dict | None = None,
) -> AgentJobRow:
    row = AgentJobRow(
        id=f"j_{_id()}",
        query_id=query_id,
        user_id=user_id,
        agent_id=agent_id,
        provider=provider,
        model=model,
        sdk_type=sdk_type,
        endpoint=endpoint,
        kind=kind,
        input_payload=json.dumps({"messages": messages}),
        stream=1 if stream else 0,
        metadata_=json.dumps(metadata or {}),
        status="queued",
        attempt_count=0,
    )
    db.add(row)
    await db.flush()
    return row


async def claim_next_agent_job(
    db: AsyncSession,
    *,
    agent_id: str,
) -> AgentJobRow | None:
    stmt = (
        select(AgentJobRow)
        .where(AgentJobRow.status == "queued")
        .where(AgentJobRow.agent_id == agent_id)
        .order_by(AgentJobRow.created_at.asc())
        .with_for_update(skip_locked=True)
    )
    result = await db.execute(stmt)
    current = result.scalars().first()
    if not current:
        return None
    current.status = "running"
    current.agent_id = agent_id
    current.attempt_count += 1
    current.claimed_at = _utcnow()
    current.updated_at = _utcnow()
    await db.flush()
    return current


async def complete_agent_job(
    db: AsyncSession,
    *,
    agent_id: str,
    job_id: str,
    status: str,
    output: dict,
    error: str = "",
    usage: dict | None = None,
    completed_at: str = "",
) -> tuple[AgentJobRow, ResponseRow]:
    result = await db.execute(select(AgentJobRow).where(AgentJobRow.id == job_id))
    job = result.scalars().first()
    if not job:
        raise ValueError("Job not found")
    if job.agent_id and job.agent_id != agent_id:
        raise ValueError("Job is assigned to a different agent")

    job.status = "completed"
    if error:
        job.error_message = error
    job.completed_at = _utcnow()
    job.updated_at = _utcnow()

    response_row = ResponseRow(
        id=f"r_{_id()}",
        query_id=job.query_id,
        response=json.dumps(output),
        latency_ms=int((usage or {}).get("latency_ms", 0)),
        input_tokens=int((usage or {}).get("prompt_tokens", 0)),
        output_tokens=int((usage or {}).get("completion_tokens", 0)),
        status_code=200 if status == "completed" else 500,
        error_message=error or None,
        estimated_cost=0.0,
        timestamp=_utcnow(),
    )
    db.add(response_row)
    await db.flush()
    return job, response_row


# ── Read operations ─────────────────────────────────────────────────────


async def get_recent_requests(
    db: AsyncSession,
    limit: int = 50,
    user_id: str | None = None,
    since: _dt.datetime | None = None,
    agent_id: str | None = None,
    model_id: str | None = None,
) -> list[RequestEventResponse]:
    """Fetch recent queries + responses joined, for dashboard display."""
    stmt = select(QueryRow).order_by(desc(QueryRow.timestamp)).limit(limit)
    if user_id is not None:
        stmt = stmt.where(QueryRow.user_id == user_id)
    if since is not None:
        stmt = stmt.where(QueryRow.timestamp >= since)
    if agent_id is not None:
        stmt = stmt.where(QueryRow.agent_id == agent_id)
    if model_id is not None:
        stmt = stmt.where(QueryRow.model == model_id)
    q_result = await db.execute(stmt)
    queries = q_result.scalars().all()

    items: list[RequestEventResponse] = []
    for q in queries:
        r_result = await db.execute(
            select(ResponseRow).where(ResponseRow.query_id == q.id)
        )
        r = r_result.scalars().first()

        # Parse prompt messages from stored JSON
        try:
            prompt: list[dict] = json.loads(q.input_messages) if q.input_messages else []
        except (json.JSONDecodeError, TypeError):
            prompt = []

        # Extract readable response text from the stored raw response JSON
        response_text = ""
        if r and r.response:
            try:
                resp_data = json.loads(r.response)
                if isinstance(resp_data, dict):
                    if "choices" in resp_data and resp_data["choices"]:
                        choice = resp_data["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            response_text = choice["message"]["content"] or ""
                        elif "text" in choice:
                            response_text = choice["text"] or ""
                    elif "content" in resp_data:
                        response_text = str(resp_data["content"])
            except (json.JSONDecodeError, TypeError, KeyError):
                response_text = (r.response or "")[:500]

        items.append(
            RequestEventResponse(
                id=q.id,
                timestamp=q.timestamp.isoformat() if q.timestamp else "",
                agent_id=q.agent_id,
                model=q.model,
                provider=q.provider,
                endpoint=q.endpoint,
                status=r.status_code if r else 0,
                latency_ms=r.latency_ms if r else 0,
                input_tokens=r.input_tokens if r else 0,
                output_tokens=r.output_tokens if r else 0,
                estimated_cost=r.estimated_cost if r else 0.0,
                error_message=r.error_message if r else None,
                prompt=prompt,
                response_text=response_text,
            )
        )
    return items


async def build_dashboard(db: AsyncSession, user_id: str) -> DashboardResponse:
    """Build the full dashboard payload consumed by the frontend."""
    # Totals
    agent_count = (
        await db.execute(
            select(func.count(AgentRow.id)).where(AgentRow.user_id == user_id)
        )
    ).scalar() or 0
    model_count = (
        await db.execute(
            select(func.count(ModelConnectionRow.id)).where(
                ModelConnectionRow.user_id == user_id
            )
        )
    ).scalar() or 0
    active_models = (
        await db.execute(
            select(func.count(ModelConnectionRow.id)).where(
                ModelConnectionRow.status == "active",
                ModelConnectionRow.user_id == user_id,
            )
        )
    ).scalar() or 0

    # Today's request stats
    today_start = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_queries = (
        await db.execute(
            select(func.count(QueryRow.id)).where(QueryRow.timestamp >= today_start)
            .where(QueryRow.user_id == user_id)
        )
    ).scalar() or 0

    owned_query_ids = select(QueryRow.id).where(QueryRow.user_id == user_id)
    token_result = await db.execute(
        select(
            func.coalesce(func.sum(ResponseRow.input_tokens), 0),
            func.coalesce(func.sum(ResponseRow.output_tokens), 0),
            func.count(case((ResponseRow.status_code >= 400, ResponseRow.id))),
        ).where(
            ResponseRow.timestamp >= today_start,
            ResponseRow.query_id.in_(owned_query_ids),
        )
    )
    token_row = token_result.one()
    total_input = token_row[0]
    total_output = token_row[1]
    errors = token_row[2]

    totals = DashboardTotals(
        agents=agent_count,
        model_tokens=model_count,
        requests_today=today_queries,
        tokens_consumed=total_input + total_output,
        errors=errors,
        active_models=active_models,
    )

    # Metric time series — bucket by 3-hour intervals for last 24h
    metric_series = await _build_timeseries(db, today_start, user_id=user_id)

    # Recent requests
    recent = await get_recent_requests(db, limit=20, user_id=user_id)

    return DashboardResponse(
        totals=totals,
        metric_series=metric_series,
        recent_requests=recent,
    )


async def build_metrics(
    db: AsyncSession,
    user_id: str,
    time_range: str = "24h",
    agent_id: str | None = None,
    model_id: str | None = None,
) -> MetricsResponse:
    """Build metrics response for /v1/metrics with optional filters."""
    now = _utcnow()
    _range_map = {
        "1h": _dt.timedelta(hours=1),
        "24h": _dt.timedelta(hours=24),
        "7d": _dt.timedelta(days=7),
        "30d": _dt.timedelta(days=30),
    }
    delta = _range_map.get(time_range, _dt.timedelta(hours=24))
    since = now - delta

    series = await _build_timeseries(
        db, since, user_id=user_id, time_range=time_range,
        agent_id=agent_id, model_id=model_id,
    )
    recent = await get_recent_requests(
        db, limit=50, user_id=user_id,
        since=since, agent_id=agent_id, model_id=model_id,
    )
    return MetricsResponse(metric_series=series, requests=recent)


async def _build_timeseries(
    db: AsyncSession,
    since: _dt.datetime,
    user_id: str | None = None,
    time_range: str = "24h",
    agent_id: str | None = None,
    model_id: str | None = None,
) -> list[MetricPoint]:
    """Build bucketed metric points scaled to the requested time range."""
    points: list[MetricPoint] = []

    # Determine bucket count and interval based on time_range
    _bucket_config: dict[str, tuple[int, _dt.timedelta]] = {
        "1h":  (12, _dt.timedelta(minutes=5)),
        "24h": (8,  _dt.timedelta(hours=3)),
        "7d":  (7,  _dt.timedelta(days=1)),
        "30d": (30, _dt.timedelta(days=1)),
    }
    bucket_count, bucket_size = _bucket_config.get(time_range, (8, _dt.timedelta(hours=3)))

    for i in range(bucket_count):
        bucket_start = since + i * bucket_size
        bucket_end = bucket_start + bucket_size

        # Build sub-query for owned query IDs with optional agent/model filters
        owned_query_stmt = select(QueryRow.id)
        if user_id is not None:
            owned_query_stmt = owned_query_stmt.where(QueryRow.user_id == user_id)
        if agent_id is not None:
            owned_query_stmt = owned_query_stmt.where(QueryRow.agent_id == agent_id)
        if model_id is not None:
            owned_query_stmt = owned_query_stmt.where(QueryRow.model == model_id)

        stmt = select(
            func.count(ResponseRow.id),
            func.coalesce(func.sum(ResponseRow.input_tokens), 0),
            func.coalesce(func.sum(ResponseRow.output_tokens), 0),
            func.coalesce(func.avg(ResponseRow.latency_ms), 0),
            func.count(case((ResponseRow.status_code >= 400, ResponseRow.id))),
        ).where(
            ResponseRow.timestamp >= bucket_start,
            ResponseRow.timestamp < bucket_end,
            ResponseRow.query_id.in_(owned_query_stmt),
        )

        result = await db.execute(stmt)
        row = result.one()

        # Format label based on range
        if time_range == "1h":
            label = bucket_start.strftime("%H:%M")
        elif time_range in ("7d", "30d"):
            label = bucket_start.strftime("%b %d")
        else:
            label = bucket_start.strftime("%H:%M")

        points.append(
            MetricPoint(
                time=label,
                requests=row[0],
                input_tokens=row[1],
                output_tokens=row[2],
                latency=int(row[3]),
                errors=row[4],
            )
        )

    return points


# ── Helpers ─────────────────────────────────────────────────────────────


def _estimate_cost(resp: ProviderResponse) -> float:
    """Rough cost estimation per 1K tokens.  Very approximate for P0."""
    # Simplified pricing (USD per 1K tokens)
    _pricing: dict[str, tuple[float, float]] = {
        "openai": (0.005, 0.015),
        "anthropic": (0.003, 0.015),
        "gemini": (0.001, 0.002),
        "qwen": (0.002, 0.006),
        "local": (0.0, 0.0),
    }
    input_rate, output_rate = _pricing.get(resp.provider, (0.005, 0.015))
    return round(
        (resp.input_tokens / 1000) * input_rate
        + (resp.output_tokens / 1000) * output_rate,
        4,
    )
