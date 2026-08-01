"""Provider routing — resolves model + call_sdk to the correct adapter.

Includes automatic provider fallback:
  - resolve()              → single provider (original behaviour)
  - resolve_with_fallbacks() → ordered list of (provider, model) candidates
  - chat_with_fallback()  → executes the request, auto-falls-back on failure
  - chat_with_fallback_stream() → same but yields SSE chunks (streaming)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

from app.providers.anthropic import AnthropicProvider
from app.providers.base import BaseProvider, ProviderResponse
from app.providers.gemini import GeminiProvider
from app.providers.local import LocalProvider
from app.providers.openai import OpenAIProvider
from app.providers.qwen import QwenProvider
from app.providers.huggingface import HuggingFaceProvider

logger = logging.getLogger(__name__)

# ── Provider name → class mapping ──────────────────────────────────────

_PROVIDERS: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "qwen": QwenProvider,
    "dashscope": QwenProvider,
    "local": LocalProvider,
    "ollama": LocalProvider,
    "vllm": LocalProvider,
    "tgi": LocalProvider,
    "huggingface": HuggingFaceProvider,
    "hf": HuggingFaceProvider,
    # Legacy named open-model providers — all route to HuggingFace
    "llama": HuggingFaceProvider,
    "mistral": HuggingFaceProvider,
    "deepseek": HuggingFaceProvider,
}

# ── SDK → provider heuristic ───────────────────────────────────────────

_SDK_TO_PROVIDER: dict[str, str] = {
    "openai-sdk": "openai",
    "claude-sdk": "anthropic",
    "anthropic-sdk": "anthropic",
    "gemini-sdk": "gemini",
    "qwen-sdk": "qwen",
    "raw-http": "openai",       # default to OpenAI-compatible format
    "vizhi-sdk": "openai",      # Vizhi SDK uses OpenAI-compatible wire format
    "hf-sdk": "huggingface",
    "huggingface-sdk": "huggingface",
}

# ── Model prefix → provider heuristic ──────────────────────────────────

_MODEL_PREFIX_HINTS: dict[str, str] = {
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude": "anthropic",
    "gemini": "gemini",
    "qwen": "qwen",
    # Open models — default to HuggingFace (free tier)
    "llama": "huggingface",
    "mistral": "huggingface",
    "mixtral": "huggingface",
    "phi": "huggingface",
    "deepseek": "huggingface",
    "meta-llama": "huggingface",
}

# ── Fallback chain: if primary provider fails, try these in order ───────
# Maps primary provider → ordered list of fallback provider names.
# The fallback model is the "best available" generic model for each provider.

_FALLBACK_CHAIN: dict[str, list[tuple[str, str]]] = {
    "openai": [
        ("anthropic", "claude-3-5-haiku-20241022"),
        ("gemini", "gemini-2.0-flash"),
        ("qwen", "qwen-plus"),
        ("huggingface", "meta-llama/Llama-3.1-8B-Instruct"),
        ("local", "llama"),
    ],
    "anthropic": [
        ("openai", "gpt-4o-mini"),
        ("gemini", "gemini-2.0-flash"),
        ("qwen", "qwen-plus"),
        ("huggingface", "meta-llama/Llama-3.1-8B-Instruct"),
        ("local", "llama"),
    ],
    "gemini": [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-20241022"),
        ("qwen", "qwen-plus"),
        ("huggingface", "meta-llama/Llama-3.1-8B-Instruct"),
        ("local", "llama"),
    ],
    "qwen": [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-20241022"),
        ("gemini", "gemini-2.0-flash"),
        ("huggingface", "Qwen/Qwen2.5-7B-Instruct"),
        ("local", "llama"),
    ],
    "huggingface": [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-20241022"),
        ("gemini", "gemini-2.0-flash"),
        ("qwen", "qwen-plus"),
        ("local", "llama"),
    ],
    "local": [
        ("huggingface", "meta-llama/Llama-3.1-8B-Instruct"),
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-3-5-haiku-20241022"),
        ("gemini", "gemini-2.0-flash"),
        ("qwen", "qwen-plus"),
    ],
}

# ── Errors that should trigger a fallback (transient / provider-side) ──

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RETRYABLE_KEYWORDS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "service unavailable",
    "overloaded",
    "timeout",
    "connection",
    "502",
    "503",
    "504",
    "500",
)

# ── Errors that are NEVER retryable (config / auth issues) ─────────────
# If an error message contains any of these, the fallback chain is
# stopped immediately and the error is surfaced to the caller.

_NON_RETRYABLE_KEYWORDS = (
    "not configured",
    "api key is not configured",
    "api_key is not configured",
    "base_url is not configured",
    "not set",
    "401",
    "403",
    "invalid_api_key",
    "invalid api key",
    "authentication",
    "permission denied",
    "insufficient_quota",
    "billing",
)


def _is_non_retryable(exc: Exception) -> bool:
    """Return True if this error should stop the fallback chain immediately.

    Config errors (missing keys, auth failures) must never be silently
    swallowed by routing to a different provider — that leads to confusing
    behaviour where an OpenAI token silently serves HuggingFace responses.
    """
    msg = str(exc).lower()
    return any(kw in msg for kw in _NON_RETRYABLE_KEYWORDS)


def _is_retryable(exc: Exception) -> bool:
    """Heuristically decide whether an exception warrants a fallback attempt.

    An error is retryable only if it is transient AND not a config/auth error.
    """
    if _is_non_retryable(exc):
        return False
    msg = str(exc).lower()
    return any(kw in msg for kw in _RETRYABLE_KEYWORDS)


# ── Singleton adapter instances ─────────────────────────────────────────

_instances: dict[str, BaseProvider] = {}


def _get_provider(name: str) -> BaseProvider:
    """Return a cached provider instance."""
    if name not in _instances:
        cls = _PROVIDERS.get(name)
        if cls is None:
            raise ValueError(f"Unknown provider: {name}")
        _instances[name] = cls()
    return _instances[name]


# ── Result dataclass returned by chat_with_fallback() ──────────────────

@dataclass
class FallbackResult:
    """Wraps a ProviderResponse with metadata about the fallback journey."""

    response: ProviderResponse
    # The provider that ultimately succeeded (may differ from the original).
    final_provider: str
    # The model that was ultimately used.
    final_model: str
    # Was a fallback required?
    used_fallback: bool = False
    # Human-readable list of "provider: reason" strings for each failure.
    fallback_attempts: list[str] = field(default_factory=list)


# ── ProviderRouter ──────────────────────────────────────────────────────

class ProviderRouter:
    """Resolves ``(model, call_sdk)`` to ``(provider_instance, model_name)``
    and supports automatic multi-provider fallback."""

    # ── Single-provider resolution (original behaviour) ─────────────────

    def resolve(
        self,
        model: str,
        call_sdk: str | None = None,
    ) -> tuple[BaseProvider, str]:
        """Return ``(provider, resolved_model_name)``.

        Resolution order:
        1. Explicit prefix ``provider/model`` (e.g. ``openai/gpt-4o-mini``)
        2. ``call_sdk`` parameter hint
        3. Model name prefix heuristic
        4. Raise ``ValueError``
        """
        provider_name, model_name = self._resolve_name(model, call_sdk)
        return _get_provider(provider_name), model_name

    def _resolve_name(
        self,
        model: str,
        call_sdk: str | None = None,
    ) -> tuple[str, str]:
        """Return ``(provider_name, model_name)`` without instantiating."""
        provider_name: str | None = None
        model_name = model

        # 1. Explicit prefix: "openai/gpt-4o-mini"
        if "/" in model:
            parts = model.split("/", 1)
            candidate = parts[0].lower()
            if candidate in _PROVIDERS:
                provider_name = candidate
                model_name = parts[1]

        # 2. SDK hint
        if provider_name is None and call_sdk:
            provider_name = _SDK_TO_PROVIDER.get(call_sdk.lower())

        # 3. Model name prefix
        if provider_name is None:
            lower = model_name.lower()
            for prefix, prov in _MODEL_PREFIX_HINTS.items():
                if lower.startswith(prefix):
                    provider_name = prov
                    break

        if provider_name is None:
            raise ValueError(
                f"Cannot resolve provider for model '{model}'. "
                "Use 'provider/model' format or specify 'call_sdk'."
            )

        return provider_name, model_name

    # ── Fallback-aware candidate list ───────────────────────────────────

    def resolve_with_fallbacks(
        self,
        model: str,
        call_sdk: str | None = None,
    ) -> list[tuple[BaseProvider, str, str]]:
        """Return an ordered list of ``(provider, model_name, provider_name)`` candidates.

        The first entry is the primary resolution.
        Subsequent entries are fallbacks from ``_FALLBACK_CHAIN``.
        """
        primary_provider_name, primary_model = self._resolve_name(model, call_sdk)
        candidates: list[tuple[BaseProvider, str, str]] = [
            (_get_provider(primary_provider_name), primary_model, primary_provider_name)
        ]

        for fallback_provider_name, fallback_model_hint in _FALLBACK_CHAIN.get(
            primary_provider_name, []
        ):
            try:
                fb_provider = _get_provider(fallback_provider_name)
                candidates.append(
                    (fb_provider, fallback_model_hint, fallback_provider_name)
                )
            except ValueError:
                # Provider not registered — skip silently
                pass

        return candidates

    # ── Main entry-point: execute with automatic fallback ───────────────

    async def chat_with_fallback(
        self,
        model: str,
        messages: list[dict],
        call_sdk: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        max_retries_per_provider: int = 2,
        base_backoff_seconds: float = 1.0,
    ) -> FallbackResult:
        """Execute a chat completion with automatic provider fallback.

        Args:
            model: Model name (optionally ``provider/model``).
            messages: Chat messages.
            call_sdk: SDK hint for provider resolution.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            max_retries_per_provider: How many times to retry the *same* provider
                on a transient error before moving to the next one.
            base_backoff_seconds: Base sleep for exponential backoff between retries.

        Returns:
            ``FallbackResult`` with the successful response and fallback metadata.

        Raises:
            RuntimeError: If all providers (and their retries) are exhausted.
        """
        candidates = self.resolve_with_fallbacks(model, call_sdk)
        all_errors: list[str] = []

        for attempt_idx, (provider, resolved_model, provider_name) in enumerate(candidates):
            is_primary = attempt_idx == 0

            for retry in range(max_retries_per_provider):
                try:
                    logger.info(
                        "Attempting provider=%s model=%s (attempt %d/%d)",
                        provider_name,
                        resolved_model,
                        retry + 1,
                        max_retries_per_provider,
                    )
                    t0 = time.perf_counter()
                    response = await provider.chat_completion(
                        model=resolved_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    logger.info(
                        "Success: provider=%s model=%s latency=%dms",
                        provider_name,
                        resolved_model,
                        elapsed_ms,
                    )
                    return FallbackResult(
                        response=response,
                        final_provider=provider_name,
                        final_model=resolved_model,
                        used_fallback=not is_primary,
                        fallback_attempts=all_errors,
                    )

                except Exception as exc:
                    error_msg = (
                        f"provider={provider_name} model={resolved_model} "
                        f"retry={retry + 1}: {exc}"
                    )
                    logger.warning("Provider attempt failed: %s", error_msg)
                    all_errors.append(error_msg)

                    # Config / auth errors: raise immediately — never fall back.
                    if _is_non_retryable(exc):
                        logger.error(
                            "Non-retryable error on %s (config/auth) — stopping fallback chain: %s",
                            provider_name, exc,
                        )
                        raise

                    retryable = _is_retryable(exc)

                    if retryable and retry < max_retries_per_provider - 1:
                        # Exponential backoff before retrying the same provider
                        backoff = base_backoff_seconds * (2 ** retry)
                        logger.info(
                            "Retryable error on %s — backing off %.1fs before retry",
                            provider_name,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue

                    break

        raise RuntimeError(
            f"All providers exhausted after {len(candidates)} provider(s). "
            f"Errors: [{'; '.join(all_errors)}]"
        )


    # ── Streaming: execute with automatic fallback ──────────────────────

    async def chat_with_fallback_stream(
        self,
        model: str,
        messages: list[dict],
        call_sdk: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        max_retries_per_provider: int = 2,
        base_backoff_seconds: float = 1.0,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion with automatic provider fallback.

        Fallback only applies *before* the first chunk is yielded.  Once
        a provider starts streaming successfully the connection is committed
        to that provider.

        Yields OpenAI-compatible ``data: {...}`` SSE lines.
        Terminates with ``data: [DONE]``.

        Raises:
            RuntimeError: If all providers (and their retries) are exhausted
                          before any chunk is produced.
        """
        candidates = self.resolve_with_fallbacks(model, call_sdk)
        all_errors: list[str] = []

        for attempt_idx, (provider, resolved_model, provider_name) in enumerate(candidates):
            is_primary = attempt_idx == 0

            for retry in range(max_retries_per_provider):
                try:
                    logger.info(
                        "Stream attempt: provider=%s model=%s (attempt %d/%d)",
                        provider_name, resolved_model, retry + 1, max_retries_per_provider,
                    )
                    # Pull the first chunk — this is where connection errors surface.
                    # If it succeeds we commit to this provider for the rest of the stream.
                    stream_gen = provider.chat_completion_stream(
                        model=resolved_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    # Eagerly fetch the first item to catch immediate errors
                    first_chunk: str | None = None
                    async for chunk in stream_gen:
                        first_chunk = chunk
                        break

                    if first_chunk is None:
                        # Empty stream — treat as an error and try next provider
                        raise RuntimeError("Provider returned an empty stream")

                    logger.info(
                        "Stream committed: provider=%s model=%s fallback=%s",
                        provider_name, resolved_model, not is_primary,
                    )

                    # Yield first chunk then exhaust the rest
                    yield first_chunk
                    async for chunk in stream_gen:
                        yield chunk
                    return

                except Exception as exc:
                    error_msg = (
                        f"provider={provider_name} model={resolved_model} "
                        f"retry={retry + 1}: {exc}"
                    )
                    logger.warning("Stream attempt failed: %s", error_msg)
                    all_errors.append(error_msg)

                    # Config / auth errors: raise immediately — never fall back.
                    if _is_non_retryable(exc):
                        logger.error(
                            "Non-retryable error on %s (config/auth) — stopping fallback chain: %s",
                            provider_name, exc,
                        )
                        raise

                    retryable = _is_retryable(exc)
                    if retryable and retry < max_retries_per_provider - 1:
                        backoff = base_backoff_seconds * (2 ** retry)
                        logger.info(
                            "Retryable stream error on %s — backing off %.1fs",
                            provider_name, backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue

                    break

        raise RuntimeError(
            f"All providers exhausted (streaming) after {len(candidates)} provider(s). "
            f"Errors: [{'; '.join(all_errors)}]"
        )


# ── Module-level singleton ──────────────────────────────────────────────

provider_router = ProviderRouter()
