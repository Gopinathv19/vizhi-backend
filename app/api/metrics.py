"""Metrics retrieval endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.responses import MetricsResponse
from app.services.persistence import build_metrics

router = APIRouter(prefix="/v1/metrics", tags=["metrics"])


@router.get("")
async def get_metrics(
    db: AsyncSession = Depends(get_db),
) -> MetricsResponse:
    """Return aggregated metrics: timeseries + recent requests."""
    return await build_metrics(db)
