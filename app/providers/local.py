"""Self-hosted model provider adapter (Ollama / vLLM / TGI).

All three expose OpenAI-compatible /v1/chat/completions endpoints.
"""

from __future__ import annotations

import time

import httpx

from app.config.settings import settings
from app.providers.base import BaseProvider, ProviderResponse


class LocalProvider(BaseProvider):
    provider_name = "local"

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ProviderResponse:
        # Ollama native endpoint
        url = f"{self._base_url}/api/chat"
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        start = time.perf_counter_ns()
        async with httpx.AsyncClient(timeout=300) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                # Fall back to OpenAI-compatible endpoint (vLLM/TGI)
                return await self._openai_compat(
                    model, messages, temperature, max_tokens
                )
            except httpx.ConnectError:
                raise RuntimeError(
                    f"Cannot connect to local model server at {self._base_url}"
                )

        latency = (time.perf_counter_ns() - start) // 1_000_000
        data = resp.json()

        return ProviderResponse(
            content=data.get("message", {}).get("content", ""),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            model=model,
            provider=self.provider_name,
            raw_response=data,
            latency_ms=latency,
            finish_reason="stop",
        )

    async def _openai_compat(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
    ) -> ProviderResponse:
        """Fallback to OpenAI-compatible /v1/chat/completions (vLLM/TGI)."""
        url = f"{self._base_url}/v1/chat/completions"
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        start = time.perf_counter_ns()
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(url, json=payload)
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
