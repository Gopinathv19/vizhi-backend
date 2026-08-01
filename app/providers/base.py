"""Abstract base class for all LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator


@dataclass
class ProviderResponse:
    """Standardised result returned by every provider adapter."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""
    raw_response: dict = field(default_factory=dict)
    latency_ms: int = 0
    finish_reason: str = "stop"


class BaseProvider(ABC):
    """Interface that every provider adapter must implement."""

    provider_name: str = "base"

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ProviderResponse:
        """Execute a non-streaming chat completion request against the upstream provider."""

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion as OpenAI-compatible SSE lines.

        Yields raw ``data: {...}`` strings.  The default implementation falls
        back to the non-streaming path and wraps the single response as a pair
        of SSE chunks so callers always get a valid stream regardless of
        whether the upstream supports streaming.

        Provider subclasses may override this method to enable true token-by-
        token streaming when the upstream supports it.
        """
        # Default: buffer the full response and emit as a synthetic stream
        result = await self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        import json as _json, uuid as _uuid, time as _time

        chunk_id = f"vzr_{_uuid.uuid4().hex[:12]}"
        created = int(_time.time())

        # Single content chunk
        yield "data: " + _json.dumps({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": result.model or model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": result.content},
                "finish_reason": None,
            }],
        })

        # Final chunk with finish_reason
        yield "data: " + _json.dumps({
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": result.model or model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": result.finish_reason or "stop",
            }],
            "usage": {
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
        })

        yield "data: [DONE]"
