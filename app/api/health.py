"""Provider Health Status API.

GET  /v1/health/providers          – list all providers with their current health
POST /v1/health/providers/refresh  – trigger an immediate probe (admin/internal)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.health_checker import (
    get_all_provider_health,
    run_health_checks,
    ProviderHealth,
)

router = APIRouter(prefix="/v1/health", tags=["health"])


# ── Response Schemas ───────────────────────────────────────────────────


class ProviderHealthResponse(BaseModel):
    provider: str
    label: str
    status: str          # "operational" | "degraded" | "down" | "unknown" | "unconfigured"
    latency_ms: int
    last_checked: str
    message: str
    incident_count: int


class HealthSummaryResponse(BaseModel):
    overall_status: str  # "operational" | "degraded" | "partial_outage" | "major_outage"
    providers: list[ProviderHealthResponse]
    checked_at: str


# ── Helpers ────────────────────────────────────────────────────────────


def _compute_overall_status(providers: list[ProviderHealth]) -> str:
    """Derive an aggregate status from individual provider statuses."""
    # Ignore unconfigured / unknown when counting
    known = [p for p in providers if p.status not in ("unconfigured", "unknown")]
    if not known:
        return "unknown"
    statuses = {p.status for p in known}
    if statuses == {"operational"}:
        return "operational"
    if "operational" not in statuses:
        return "major_outage"
    if "down" in statuses:
        return "partial_outage"
    return "degraded"


def _to_response(p: ProviderHealth) -> ProviderHealthResponse:
    return ProviderHealthResponse(
        provider=p.provider,
        label=p.label,
        status=p.status,
        latency_ms=p.latency_ms,
        last_checked=p.last_checked,
        message=p.message,
        incident_count=p.incident_count,
    )


# ── Routes ─────────────────────────────────────────────────────────────


@router.get("/providers", response_model=HealthSummaryResponse)
async def get_provider_health() -> HealthSummaryResponse:
    """
    Return the current health status of all configured AI providers.

    Results are served from the in-memory cache updated every 60 seconds
    by the background health-check loop.
    """
    from datetime import datetime, timezone

    providers = get_all_provider_health()
    return HealthSummaryResponse(
        overall_status=_compute_overall_status(providers),
        providers=[_to_response(p) for p in providers],
        checked_at=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/providers/refresh", response_model=HealthSummaryResponse)
async def refresh_provider_health() -> HealthSummaryResponse:
    """
    Force an immediate health probe of all providers and return fresh results.
    Useful for the frontend 'Refresh' button.
    """
    from datetime import datetime, timezone

    try:
        await run_health_checks()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    providers = get_all_provider_health()
    return HealthSummaryResponse(
        overall_status=_compute_overall_status(providers),
        providers=[_to_response(p) for p in providers],
        checked_at=datetime.now(timezone.utc).isoformat(),
    )
