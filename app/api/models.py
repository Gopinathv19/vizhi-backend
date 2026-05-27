"""Model registry / connection API endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.db_models import ModelConnectionRow
from app.schemas.requests import CreateModelConnectionRequest
from app.schemas.responses import ModelConnectionResponse

router = APIRouter(prefix="/v1/models", tags=["models"])

# ── Default provider endpoints ──────────────────────────────────────────

_DEFAULT_ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "ollama": "http://localhost:11434",
}


def _row_to_response(row: ModelConnectionRow) -> ModelConnectionResponse:
    return ModelConnectionResponse(
        id=row.id,
        provider=row.provider,
        model_name=row.model_name,
        endpoint_url=row.endpoint_url or "",
        status=row.status,
        sdk_type=row.sdk_type,
        metadata=row.metadata_,
        usage_count=row.usage_count,
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_model_connection(
    body: CreateModelConnectionRequest,
    db: AsyncSession = Depends(get_db),
) -> ModelConnectionResponse:
    """Register a new model connection."""
    endpoint = body.endpoint_url or _DEFAULT_ENDPOINTS.get(body.provider.lower(), "")

    row = ModelConnectionRow(
        id=f"mt_{uuid.uuid4().hex[:8]}",
        provider=body.provider.lower(),
        model_name=body.model_name,
        endpoint_url=endpoint,
        status="active",
        sdk_type=body.sdk_type,
        metadata_=body.metadata,
    )
    db.add(row)
    await db.flush()
    return _row_to_response(row)


@router.get("")
async def list_models(
    db: AsyncSession = Depends(get_db),
) -> list[ModelConnectionResponse]:
    """List all registered model connections."""
    result = await db.execute(
        select(ModelConnectionRow).order_by(ModelConnectionRow.created_at.desc())
    )
    return [_row_to_response(row) for row in result.scalars().all()]


@router.get("/registry")
async def model_registry() -> list[dict]:
    """Return the static registry of all known models across providers."""
    return [
        {"model": "gpt-4o", "provider": "openai"},
        {"model": "gpt-4o-mini", "provider": "openai"},
        {"model": "gpt-4.1", "provider": "openai"},
        {"model": "gpt-4.1-mini", "provider": "openai"},
        {"model": "gpt-4.1-nano", "provider": "openai"},
        {"model": "o3", "provider": "openai"},
        {"model": "o3-mini", "provider": "openai"},
        {"model": "o4-mini", "provider": "openai"},
        {"model": "claude-sonnet-4-20250514", "provider": "anthropic"},
        {"model": "claude-3-5-sonnet-20241022", "provider": "anthropic"},
        {"model": "claude-3-5-haiku-20241022", "provider": "anthropic"},
        {"model": "claude-3-haiku-20240307", "provider": "anthropic"},
        {"model": "gemini-2.5-flash", "provider": "gemini"},
        {"model": "gemini-2.5-pro", "provider": "gemini"},
        {"model": "gemini-2.0-flash", "provider": "gemini"},
        {"model": "qwen-max", "provider": "qwen"},
        {"model": "qwen-plus", "provider": "qwen"},
        {"model": "qwen-turbo", "provider": "qwen"},
    ]
