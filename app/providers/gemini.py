"""Google Gemini provider adapter using raw httpx."""

from __future__ import annotations

import time

import httpx

from app.config.settings import settings
from app.providers.base import BaseProvider, ProviderResponse

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider(BaseProvider):
    provider_name = "gemini"

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ProviderResponse:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        # Convert OpenAI-style messages → Gemini contents format.
        system_instruction: str | None = None
        contents: list[dict] = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}],
                })

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        url = f"{_GEMINI_BASE}/{model}:generateContent?key={settings.gemini_api_key}"

        start = time.perf_counter_ns()
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
        latency = (time.perf_counter_ns() - start) // 1_000_000

        resp.raise_for_status()
        data = resp.json()

        # Extract content from Gemini response.
        candidates = data.get("candidates", [{}])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(p.get("text", "") for p in parts)

        usage_meta = data.get("usageMetadata", {})

        return ProviderResponse(
            content=text,
            input_tokens=usage_meta.get("promptTokenCount", 0),
            output_tokens=usage_meta.get("candidatesTokenCount", 0),
            model=model,
            provider=self.provider_name,
            raw_response=data,
            latency_ms=latency,
            finish_reason=candidates[0].get("finishReason", "STOP") if candidates else "STOP",
        )
