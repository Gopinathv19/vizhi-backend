"""Dashboard summary endpoint consumed by the frontend."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.responses import DashboardResponse
from app.services.persistence import build_dashboard

router = APIRouter(prefix="/v1/dashboard", tags=["dashboard"])


@router.get("")
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
) -> DashboardResponse:
    """Combined dashboard payload: totals + metric series + recent requests.

    This endpoint returns the exact shape the frontend ``useDashboard()`` expects.
    """
    return await build_dashboard(db)
