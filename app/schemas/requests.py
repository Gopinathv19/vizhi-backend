"""Pydantic request schemas for API validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Chat Completion ─────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user", "assistant", "system"])
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., examples=["openai/gpt-4o-mini", "claude-sonnet-4"])
    messages: list[ChatMessage]
    call_sdk: str | None = Field(
        None,
        description="SDK adapter hint: openai-sdk, claude-sdk, raw-http",
        examples=["openai-sdk"],
    )
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=128_000)


# ── Agent CRUD ──────────────────────────────────────────────────────────


class CreateAgentRequest(BaseModel):
    name: str = Field(..., min_length=2)
    description: str = Field("", min_length=0)
    cid: str = Field(
        ...,
        min_length=3,
        description="Stable agent identifier (like Falcon CID)",
        examples=["agent_prod_001"],
    )
    owner: str = Field("", min_length=0)
    tags: str = Field("", description="Comma-separated tags")
    preferred_model: str | None = None


class UpdateAgentRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    owner: str | None = None
    tags: str | None = None
    preferred_model: str | None = None
    status: str | None = None


# ── Model Connection ───────────────────────────────────────────────────


class CreateModelConnectionRequest(BaseModel):
    provider: str = Field(..., examples=["openai", "anthropic", "gemini", "qwen", "ollama"])
    model_name: str = Field(..., examples=["gpt-4o-mini"])
    endpoint_url: str = Field("", description="Custom endpoint URL (required for self-hosted)")
    sdk_type: str | None = Field(None, examples=["openai-sdk", "claude-sdk"])
    metadata: str | None = None
