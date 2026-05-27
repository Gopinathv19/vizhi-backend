"""Qwen (DashScope) provider adapter using raw httpx.

Uses the DashScope-compatible OpenAI-style endpoint.
"""

from __future__ import annotations

import time

import httpx

from app.config.settings import settings
from app.providers.base import BaseProvider, ProviderResponse

_DASHSCOPE_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class QwenProvider(BaseProvider):
    provider_name = "qwen"

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ProviderResponse:
        if not settings.qwen_api_key:
            raise RuntimeError("QWEN_API_KEY is not configured")

        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {settings.qwen_api_key}",
            "Content-Type": "application/json",
        }

        start = time.perf_counter_ns()
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(_DASHSCOPE_CHAT_URL, json=payload, headers=headers)
        latency = (time.perf_counter_ns() - start) // 1_000_000

        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return ProviderResponse(
            content=choice["message"]["content"],
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=data.get("model", model),
            provider=self.provider_name,
            raw_response=data,
            latency_ms=latency,
            finish_reason=choice.get("finish_reason", "stop"),
        )
