"""Anthropic (Claude) provider adapter using raw httpx."""

from __future__ import annotations

import time

import httpx

from app.config.settings import settings
from app.providers.base import BaseProvider, ProviderResponse

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ProviderResponse:
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")

        # Anthropic separates system messages from the messages list.
        system_parts: list[str] = []
        api_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        payload: dict = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)

        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

        start = time.perf_counter_ns()
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                _ANTHROPIC_MESSAGES_URL, json=payload, headers=headers
            )
        latency = (time.perf_counter_ns() - start) // 1_000_000

        resp.raise_for_status()
        data = resp.json()

        # Extract text content from the response content blocks.
        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )

        usage = data.get("usage", {})

        return ProviderResponse(
            content=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=data.get("model", model),
            provider=self.provider_name,
            raw_response=data,
            latency_ms=latency,
            finish_reason=data.get("stop_reason", "end_turn"),
        )
