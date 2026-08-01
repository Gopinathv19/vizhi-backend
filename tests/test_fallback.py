"""
tests/test_fallback.py
======================
Unit + integration tests for the Automatic Provider Fallback feature.

Run from the vizhi-backend directory:

    pytest tests/test_fallback.py -v

No live API keys needed — every provider call is patched with a mock.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest

from app.providers.base import ProviderResponse
from app.services.router import (
    FallbackResult,
    ProviderRouter,
    _FALLBACK_CHAIN,
    _is_retryable,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _ok_response(provider: str = "openai", model: str = "gpt-4o-mini") -> ProviderResponse:
    return ProviderResponse(
        content="Hello from the test!",
        input_tokens=10,
        output_tokens=20,
        model=model,
        provider=provider,
        latency_ms=42,
        finish_reason="stop",
    )


def _make_router() -> ProviderRouter:
    """Fresh router with cleared singleton cache."""
    from app.services import router as router_module
    router_module._instances.clear()
    return ProviderRouter()


MESSAGES = [{"role": "user", "content": "Say hello"}]


# ── _is_retryable ──────────────────────────────────────────────────────

class TestIsRetryable:
    def test_rate_limit_is_retryable(self):
        assert _is_retryable(RuntimeError("429 rate limit exceeded"))

    def test_timeout_is_retryable(self):
        assert _is_retryable(RuntimeError("connection timeout"))

    def test_503_is_retryable(self):
        assert _is_retryable(RuntimeError("503 Service Unavailable"))

    def test_overloaded_is_retryable(self):
        assert _is_retryable(RuntimeError("Model is overloaded"))

    def test_auth_error_is_not_retryable(self):
        assert not _is_retryable(RuntimeError("401 Unauthorized — invalid API key"))

    def test_not_found_is_not_retryable(self):
        assert not _is_retryable(RuntimeError("404 model not found"))


# ── resolve_with_fallbacks ──────────────────────────────────────────────

class TestResolveWithFallbacks:
    def test_primary_first(self):
        router = _make_router()
        candidates = router.resolve_with_fallbacks("openai/gpt-4o-mini")
        assert candidates[0][2] == "openai"

    def test_fallback_providers_included(self):
        router = _make_router()
        candidates = router.resolve_with_fallbacks("openai/gpt-4o-mini")
        provider_names = [c[2] for c in candidates]
        # Must include at least one fallback provider
        assert len(provider_names) > 1

    def test_fallback_chain_matches_config(self):
        router = _make_router()
        candidates = router.resolve_with_fallbacks("openai/gpt-4o-mini")
        expected_fallbacks = [p for p, _ in _FALLBACK_CHAIN["openai"]]
        returned_fallbacks = [c[2] for c in candidates[1:]]
        # Returned fallbacks must be a subset of the configured chain
        for prov in returned_fallbacks:
            assert prov in expected_fallbacks

    def test_model_prefix_resolution(self):
        router = _make_router()
        candidates = router.resolve_with_fallbacks("claude-3-opus-20240229")
        assert candidates[0][2] == "anthropic"


# ── chat_with_fallback — happy path ────────────────────────────────────

class TestChatWithFallbackHappyPath:
    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback(self):
        router = _make_router()
        ok = _ok_response("openai")

        with patch.object(
            router.resolve_with_fallbacks("openai/gpt-4o-mini")[0][0],
            "chat_completion",
            new=AsyncMock(return_value=ok),
        ):
            result = await router.chat_with_fallback(
                model="openai/gpt-4o-mini",
                messages=MESSAGES,
                max_retries_per_provider=1,
                base_backoff_seconds=0,
            )

        assert result.used_fallback is False
        assert result.final_provider == "openai"
        assert result.response.content == "Hello from the test!"
        assert result.fallback_attempts == []


# ── chat_with_fallback — fallback triggered ─────────────────────────────

class TestChatWithFallbackTriggered:
    @pytest.mark.asyncio
    async def test_falls_back_on_503(self):
        """Primary provider raises 503 → fallback provider succeeds."""
        router = _make_router()
        candidates = router.resolve_with_fallbacks("openai/gpt-4o-mini")
        primary = candidates[0][0]
        fallback = candidates[1][0]
        fallback_name = candidates[1][2]

        ok = _ok_response(provider=fallback_name)

        primary.chat_completion = AsyncMock(
            side_effect=RuntimeError("503 Service Unavailable from openai")
        )
        fallback.chat_completion = AsyncMock(return_value=ok)

        result = await router.chat_with_fallback(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            max_retries_per_provider=1,
            base_backoff_seconds=0,
        )

        assert result.used_fallback is True
        assert result.final_provider == fallback_name
        assert len(result.fallback_attempts) >= 1

    @pytest.mark.asyncio
    async def test_non_retryable_skips_retry_immediately(self):
        """A non-retryable error (e.g., 401) skips to next provider without retry."""
        router = _make_router()
        candidates = router.resolve_with_fallbacks("openai/gpt-4o-mini")
        primary = candidates[0][0]
        fallback = candidates[1][0]

        ok = _ok_response(provider=candidates[1][2])

        call_count = {"n": 0}

        async def primary_side_effect(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("401 Unauthorized — check your API key")

        primary.chat_completion = AsyncMock(side_effect=primary_side_effect)
        fallback.chat_completion = AsyncMock(return_value=ok)

        result = await router.chat_with_fallback(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            max_retries_per_provider=3,   # would retry 3 times if retryable
            base_backoff_seconds=0,
        )

        # Non-retryable: primary called exactly once, not 3 times
        assert call_count["n"] == 1
        assert result.used_fallback is True

    @pytest.mark.asyncio
    async def test_retryable_retries_before_fallback(self):
        """A retryable error (rate limit) retries the same provider first."""
        router = _make_router()
        candidates = router.resolve_with_fallbacks("openai/gpt-4o-mini")
        primary = candidates[0][0]
        fallback = candidates[1][0]

        ok = _ok_response(provider=candidates[1][2])
        call_count = {"n": 0}

        async def primary_side_effect(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("429 rate limit exceeded")

        primary.chat_completion = AsyncMock(side_effect=primary_side_effect)
        fallback.chat_completion = AsyncMock(return_value=ok)

        result = await router.chat_with_fallback(
            model="openai/gpt-4o-mini",
            messages=MESSAGES,
            max_retries_per_provider=2,
            base_backoff_seconds=0,   # zero sleep so test is fast
        )

        # Should have retried 2 times on primary before falling back
        assert call_count["n"] == 2
        assert result.used_fallback is True


# ── chat_with_fallback — all providers fail ─────────────────────────────

class TestChatWithFallbackAllFail:
    @pytest.mark.asyncio
    async def test_raises_when_all_providers_fail(self):
        router = _make_router()
        candidates = router.resolve_with_fallbacks("openai/gpt-4o-mini")

        for provider, _, _ in candidates:
            provider.chat_completion = AsyncMock(
                side_effect=RuntimeError("503 provider down")
            )

        with pytest.raises(RuntimeError, match="All providers exhausted"):
            await router.chat_with_fallback(
                model="openai/gpt-4o-mini",
                messages=MESSAGES,
                max_retries_per_provider=1,
                base_backoff_seconds=0,
            )


# ── FallbackResult dataclass ────────────────────────────────────────────

class TestFallbackResult:
    def test_defaults(self):
        r = FallbackResult(
            response=_ok_response(),
            final_provider="openai",
            final_model="gpt-4o-mini",
        )
        assert r.used_fallback is False
        assert r.fallback_attempts == []

    def test_used_fallback_true(self):
        r = FallbackResult(
            response=_ok_response("anthropic"),
            final_provider="anthropic",
            final_model="claude",
            used_fallback=True,
            fallback_attempts=["provider=openai model=gpt retry=1: 503"],
        )
        assert r.used_fallback is True
        assert "openai" in r.fallback_attempts[0]
