
"""Core chat completion gateway endpoint."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.api_key import ChatCredential, resolve_chat_credential
from app.db.session import get_db
from app.schemas.requests import ChatCompletionRequest
from app.schemas.responses import (
    ChatChoice,
    ChatChoiceMessage,
    ChatCompletionResponse,
    ChatUsage,
    VizhiMetadata,
)
from app.services.persistence import persist_query, persist_response
from app.services.router import provider_router

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    credential: ChatCredential = Depends(resolve_chat_credential),
    db: AsyncSession = Depends(get_db),
) -> ChatCompletionResponse:
    """OpenRouter-style chat completion gateway.

    Flow:
    1. Validate API key → resolve agent  (done by dependency)
    2. Resolve provider + model
    3. Persist query
    4. Call upstream provider
    5. Persist response
    6. Return OpenAI-compatible response
    """
    requested_model = body.model
    token_model = credential.model_name
    if token_model and requested_model and requested_model != token_model:
        raise HTTPException(
            status_code=400,
            detail="Requested model does not match the model bound to this token",
        )

    model_name = token_model or requested_model
    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="Model is required when using an agent token",
        )

    # 2. Resolve provider
    try:
        provider, resolved_model = provider_router.resolve(model_name, body.call_sdk)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 3. Persist query
    messages_raw = [m.model_dump() for m in body.messages]
    query_row = await persist_query(
        db,
        agent_id=credential.principal_id,
        provider=provider.provider_name,
        model=resolved_model,
        sdk_type=body.call_sdk,
        messages=messages_raw,
    )

    # 4. Call provider
    start = time.perf_counter_ns()
    try:
        result = await provider.chat_completion(
            model=resolved_model,
            messages=messages_raw,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    except Exception as exc:
        latency = (time.perf_counter_ns() - start) // 1_000_000
        # 5a. Persist error
        await persist_response(
            db,
            query_id=query_row.id,
            status_code=502,
            error_message=str(exc),
            latency_ms=latency,
        )
        await db.commit()
        raise HTTPException(
            status_code=502,
            detail=f"Provider error: {exc}",
        )

    # 5b. Persist success
    await persist_response(
        db,
        query_id=query_row.id,
        provider_response=result,
        status_code=200,
    )

    # 6. Build OpenAI-compatible response
    return ChatCompletionResponse(
        id=f"vzr_{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=result.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatChoiceMessage(content=result.content),
                finish_reason=result.finish_reason,
            )
        ],
        usage=ChatUsage(
            prompt_tokens=result.input_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
        ),
        vizhi_metadata=VizhiMetadata(
            agent_id=credential.principal_id,
            provider=result.provider,
            latency_ms=result.latency_ms,
            query_id=query_row.id,
        ),
    )
