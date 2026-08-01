"""Central inference switchboard for all provider adapters.

Each provider has its own backend branch.  The backend is resolved from
``provider_name`` — the logical name set by the router (openai, anthropic,
gemini, qwen, local, huggingface).

Key resolution priority (per provider):
  1. The provider-specific env key  (e.g. OPENAI_API_KEY)
  2. HF_TOKEN / HUGGINGFACE_API_KEY for the huggingface backend
  3. CUSTOM_INFERENCE_API_KEY for the legacy custom backend

Set INFERENCE_BACKEND to a comma-separated fallback order:
  INFERENCE_BACKEND=openai,anthropic,gemini,qwen,huggingface
  or leave it empty to let the router pick based on ``provider_name``.

Provider → endpoint map:
  openai      → https://api.openai.com/v1  (OpenAI-compatible)
  anthropic   → https://api.anthropic.com/v1/messages  (native Anthropic)
  gemini      → https://generativelanguage.googleapis.com/v1beta/openai  (OpenAI-compat shim)
  qwen        → https://dashscope.aliyuncs.com/compatible-mode/v1  (OpenAI-compat)
  huggingface → https://router.huggingface.co/v1  (OpenAI-compat)
  local       → http://localhost:11434/v1  (Ollama / vLLM / TGI)
  custom      → CUSTOM_INFERENCE_BASE_URL  (any OpenAI-compat endpoint)
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

import httpx

from app.config.settings import settings
from app.providers.base import ProviderResponse


_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# ── Provider endpoint table ────────────────────────────────────────────────────

_PROVIDER_ENDPOINTS: dict[str, str] = {
    "openai":      "https://api.openai.com/v1",
    "anthropic":   "https://api.anthropic.com",          # special handling below
    "gemini":      "https://generativelanguage.googleapis.com/v1beta/openai",
    "google":      "https://generativelanguage.googleapis.com/v1beta/openai",
    "qwen":        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope":   "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "local":       "",    # resolved from settings.ollama_base_url at runtime
    "ollama":      "",
    "huggingface": "",    # resolved from settings.huggingface_base_url at runtime
    "custom":      "",    # resolved from settings.custom_inference_base_url at runtime
}

# ── HuggingFace open-model pool (used when routing to HF) ─────────────────────

_HF_MODEL_POOL: list[str] = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]


# ── Public API ────────────────────────────────────────────────────────────────

async def chat_completion(
    *,
    provider_name: str,
    model: str,
    messages: list[dict],
    temperature: float = 1.0,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> ProviderResponse:
    """Route a chat completion to the correct provider backend.

    Resolution order:
    1. If ``INFERENCE_BACKEND`` is set, follow that order (legacy / override mode).
    2. Otherwise resolve by ``provider_name`` directly.
    """
    backend_override = settings.inference_backend.strip().lower()

    if backend_override and backend_override != "auto":
        # Legacy / override mode — try each listed backend in order
        errors: list[str] = []
        for backend in [b.strip() for b in backend_override.split(",") if b.strip()]:
            try:
                return await _dispatch_call(
                    backend=backend,
                    provider_name=provider_name,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
        raise RuntimeError("; ".join(errors) or "All configured backends failed")

    # Auto mode — route by provider_name
    return await _dispatch_call(
        backend=_provider_to_backend(provider_name),
        provider_name=provider_name,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


async def chat_completion_stream(
    *,
    provider_name: str,
    model: str,
    messages: list[dict],
    temperature: float = 1.0,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """Stream a chat completion as OpenAI-compatible SSE lines.

    Yields raw ``data: {...}`` strings (without trailing newlines).
    Terminates with ``data: [DONE]``.
    """
    backend_override = settings.inference_backend.strip().lower()

    if backend_override and backend_override != "auto":
        errors: list[str] = []
        for backend in [b.strip() for b in backend_override.split(",") if b.strip()]:
            try:
                async for chunk in _dispatch_stream(
                    backend=backend,
                    provider_name=provider_name,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    yield chunk
                return
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
        raise RuntimeError("; ".join(errors) or "All configured backends failed (stream)")
        return

    async for chunk in _dispatch_stream(
        backend=_provider_to_backend(provider_name),
        provider_name=provider_name,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    ):
        yield chunk


# ── Internal dispatchers ──────────────────────────────────────────────────────
# Two separate functions to keep the type signatures clean:
#   _dispatch_call()   → awaitable, returns ProviderResponse
#   _dispatch_stream() → async generator, yields SSE strings

async def _dispatch_call(
    *,
    backend: str,
    provider_name: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    **kwargs: Any,
) -> ProviderResponse:
    """Dispatch a non-streaming chat completion."""
    if backend == "openai":
        _check_key("OPENAI", settings.openai_api_key)
        return await _openai_compat_call(
            url=f"{_PROVIDER_ENDPOINTS['openai'].rstrip('/').removesuffix('/v1')}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(settings.openai_api_key),
            payload=_build_payload(model, messages, temperature, max_tokens, stream=False),
            backend_name="openai",
            model=model,
            provider_name="openai",
        )

    if backend in ("anthropic", "claude"):
        # _anthropic_headers() already raises if key is missing
        return await _anthropic_call(
            url="https://api.anthropic.com/v1/messages",
            headers=_anthropic_headers(),
            payload=_build_anthropic_payload(model, messages, temperature, max_tokens, stream=False),
            model=model,
        )

    if backend in ("gemini", "google"):
        _check_key("GEMINI", settings.gemini_api_key)
        return await _openai_compat_call(
            url=f"{_PROVIDER_ENDPOINTS['gemini'].rstrip('/')}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(settings.gemini_api_key),
            payload=_build_payload(model, messages, temperature, max_tokens, stream=False),
            backend_name="gemini",
            model=model,
            provider_name="gemini",
        )

    if backend in ("qwen", "dashscope"):
        _check_key("QWEN", settings.qwen_api_key)
        return await _openai_compat_call(
            url=f"{_PROVIDER_ENDPOINTS['qwen'].rstrip('/').removesuffix('/v1')}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(settings.qwen_api_key),
            payload=_build_payload(model, messages, temperature, max_tokens, stream=False),
            backend_name="qwen",
            model=model,
            provider_name="qwen",
        )

    if backend == "huggingface":
        hf_key = settings.huggingface_api_key or settings.hf_token
        _check_key("HUGGINGFACE", hf_key)
        hf_model = _pick_hf_model(model)
        base = (settings.huggingface_base_url or "https://router.huggingface.co/v1").rstrip("/").removesuffix("/v1")
        return await _openai_compat_call(
            url=f"{base}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(hf_key),
            payload=_build_payload(hf_model, messages, temperature, max_tokens, stream=False),
            backend_name="huggingface",
            model=hf_model,
            provider_name="huggingface",
        )

    if backend in ("local", "ollama", "vllm", "tgi"):
        local_base = (kwargs.get("base_url") or settings.ollama_base_url or "http://localhost:11434").rstrip("/")
        return await _openai_compat_call(
            url=f"{local_base}{_CHAT_COMPLETIONS_PATH}",
            headers={"Content-Type": "application/json"},
            payload=_build_payload(model, messages, temperature, max_tokens, stream=False),
            backend_name="local",
            model=model,
            provider_name="local",
        )

    if backend == "custom":
        custom_base = (kwargs.get("base_url") or settings.custom_inference_base_url or "").rstrip("/")
        _check_url("CUSTOM", custom_base)
        return await _openai_compat_call(
            url=f"{custom_base}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(settings.custom_inference_api_key),
            payload=_build_payload(model, messages, temperature, max_tokens, stream=False),
            backend_name="custom",
            model=model,
            provider_name=provider_name,
        )

    raise RuntimeError(f"Unknown inference backend: '{backend}'")


async def _dispatch_stream(
    *,
    backend: str,
    provider_name: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """Dispatch a streaming chat completion — yields SSE lines."""
    if backend == "openai":
        _check_key("OPENAI", settings.openai_api_key)
        async for chunk in _openai_compat_stream_gen(
            url=f"{_PROVIDER_ENDPOINTS['openai'].rstrip('/').removesuffix('/v1')}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(settings.openai_api_key),
            payload=_build_payload(model, messages, temperature, max_tokens, stream=True),
            backend_name="openai",
            model=model,
        ):
            yield chunk
        return

    if backend in ("anthropic", "claude"):
        async for chunk in _anthropic_stream_gen(
            url="https://api.anthropic.com/v1/messages",
            headers=_anthropic_headers(),
            payload=_build_anthropic_payload(model, messages, temperature, max_tokens, stream=True),
            model=model,
        ):
            yield chunk
        return

    if backend in ("gemini", "google"):
        _check_key("GEMINI", settings.gemini_api_key)
        async for chunk in _openai_compat_stream_gen(
            url=f"{_PROVIDER_ENDPOINTS['gemini'].rstrip('/')}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(settings.gemini_api_key),
            payload=_build_payload(model, messages, temperature, max_tokens, stream=True),
            backend_name="gemini",
            model=model,
        ):
            yield chunk
        return

    if backend in ("qwen", "dashscope"):
        _check_key("QWEN", settings.qwen_api_key)
        async for chunk in _openai_compat_stream_gen(
            url=f"{_PROVIDER_ENDPOINTS['qwen'].rstrip('/').removesuffix('/v1')}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(settings.qwen_api_key),
            payload=_build_payload(model, messages, temperature, max_tokens, stream=True),
            backend_name="qwen",
            model=model,
        ):
            yield chunk
        return

    if backend == "huggingface":
        hf_key = settings.huggingface_api_key or settings.hf_token
        _check_key("HUGGINGFACE", hf_key)
        hf_model = _pick_hf_model(model)
        base = (settings.huggingface_base_url or "https://router.huggingface.co/v1").rstrip("/").removesuffix("/v1")
        async for chunk in _openai_compat_stream_gen(
            url=f"{base}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(hf_key),
            payload=_build_payload(hf_model, messages, temperature, max_tokens, stream=True),
            backend_name="huggingface",
            model=hf_model,
        ):
            yield chunk
        return

    if backend in ("local", "ollama", "vllm", "tgi"):
        local_base = (kwargs.get("base_url") or settings.ollama_base_url or "http://localhost:11434").rstrip("/")
        async for chunk in _openai_compat_stream_gen(
            url=f"{local_base}{_CHAT_COMPLETIONS_PATH}",
            headers={"Content-Type": "application/json"},
            payload=_build_payload(model, messages, temperature, max_tokens, stream=True),
            backend_name="local",
            model=model,
        ):
            yield chunk
        return

    if backend == "custom":
        custom_base = (kwargs.get("base_url") or settings.custom_inference_base_url or "").rstrip("/")
        _check_url("CUSTOM", custom_base)
        async for chunk in _openai_compat_stream_gen(
            url=f"{custom_base}{_CHAT_COMPLETIONS_PATH}",
            headers=_bearer_headers(settings.custom_inference_api_key),
            payload=_build_payload(model, messages, temperature, max_tokens, stream=True),
            backend_name="custom",
            model=model,
        ):
            yield chunk
        return

    raise RuntimeError(f"Unknown inference backend: '{backend}'")


# ── Small shared helpers ──────────────────────────────────────────────────────

def _bearer_headers(api_key: str) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _build_payload(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        p["max_tokens"] = max_tokens
    return p


def _check_key(name: str, key: str | None) -> None:
    if not key:
        raise RuntimeError(
            f"{name}_API_KEY is not configured. Add it to your .env file."
        )


def _check_url(name: str, url: str | None) -> None:
    if not url:
        raise RuntimeError(
            f"{name}_BASE_URL is not configured. Add it to your .env file."
        )


def _anthropic_headers() -> dict[str, str]:
    """Return Anthropic-native auth headers."""
    api_key = settings.anthropic_api_key
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured. Add it to your .env file.")
    return {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }


def _build_anthropic_payload(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    *,
    stream: bool,
) -> dict[str, Any]:
    """Convert OpenAI-style messages to Anthropic Messages API payload."""
    system_parts: list[str] = []
    anthropic_messages: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            anthropic_messages.append({"role": role, "content": content})

    payload: dict[str, Any] = {
        "model": model or "claude-3-5-haiku-20241022",
        "messages": anthropic_messages,
        "max_tokens": max_tokens or 1024,
        "temperature": temperature,
    }
    if system_parts:
        payload["system"] = "\n".join(system_parts)
    if stream:
        payload["stream"] = True
    return payload



# ── OpenAI-compatible backend (OpenAI, Gemini, Qwen, HuggingFace, Local) ──────

async def _openai_compat(
    *,
    backend_name: str,
    base_url: str,
    api_key: str,
    require_api_key: bool,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    stream: bool,
    provider_name: str,
):
    if not base_url:
        raise RuntimeError(f"{backend_name.upper()} base URL is not configured")
    if require_api_key and not api_key:
        raise RuntimeError(
            f"{backend_name.upper()} API key is not configured. "
            f"Set the appropriate key in your .env file."
        )

    url = f"{base_url.rstrip('/').removesuffix('/v1')}{_CHAT_COMPLETIONS_PATH}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # NOTE: stream path returns the async generator directly (NOT awaited)
    # so callers can do `async for chunk in _openai_compat(...)`.
    if stream:
        return _openai_compat_stream_gen(url=url, headers=headers, payload=payload, backend_name=backend_name, model=model)
    else:
        return await _openai_compat_call(url=url, headers=headers, payload=payload, backend_name=backend_name, model=model, provider_name=provider_name)


async def _openai_compat_call(
    *,
    url: str,
    headers: dict,
    payload: dict,
    backend_name: str,
    model: str,
    provider_name: str,
) -> ProviderResponse:
    start = time.perf_counter_ns()
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload, headers=headers)
    latency = (time.perf_counter_ns() - start) // 1_000_000

    if resp.is_error:
        raise RuntimeError(
            f"{resp.status_code} {resp.reason_phrase} from {backend_name} "
            f"for model '{model}': {resp.text}"
        )

    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})

    return ProviderResponse(
        content=choice.get("message", {}).get("content", ""),
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        model=data.get("model", model),
        provider=provider_name,
        raw_response=data,
        latency_ms=latency,
        finish_reason=choice.get("finish_reason", "stop"),
    )


async def _openai_compat_stream_gen(
    *,
    url: str,
    headers: dict,
    payload: dict,
    backend_name: str,
    model: str,
) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.is_error:
                body = await resp.aread()
                raise RuntimeError(
                    f"{resp.status_code} {resp.reason_phrase} from {backend_name} "
                    f"for model '{model}': {body.decode(errors='replace')}"
                )
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if line:
                    yield line


# ── Anthropic native backend ──────────────────────────────────────────────────

async def _anthropic_backend(
    *,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    stream: bool,
):
    """Anthropic uses a different wire format from OpenAI.

    Converts OpenAI-style messages to Anthropic format transparently.
    Uses the Anthropic Messages API (not OpenAI-compatible).
    """
    api_key = settings.anthropic_api_key
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. Add it to your .env file."
        )

    # Separate system prompt from conversation messages
    system_parts: list[str] = []
    anthropic_messages: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            anthropic_messages.append({"role": role, "content": content})

    payload: dict[str, Any] = {
        "model": model or "claude-3-5-haiku-20241022",
        "messages": anthropic_messages,
        "max_tokens": max_tokens or 1024,
    }
    if system_parts:
        payload["system"] = "\n".join(system_parts)
    if temperature is not None:
        payload["temperature"] = temperature
    if stream:
        payload["stream"] = True

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    url = "https://api.anthropic.com/v1/messages"

    if stream:
        return _anthropic_stream_gen(url=url, headers=headers, payload=payload, model=model)
    else:
        return await _anthropic_call(url=url, headers=headers, payload=payload, model=model)


async def _anthropic_call(
    *, url: str, headers: dict, payload: dict, model: str
) -> ProviderResponse:
    start = time.perf_counter_ns()
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload, headers=headers)
    latency = (time.perf_counter_ns() - start) // 1_000_000

    if resp.is_error:
        raise RuntimeError(
            f"{resp.status_code} {resp.reason_phrase} from anthropic "
            f"for model '{model}': {resp.text}"
        )

    data = resp.json()
    content_blocks = data.get("content", [])
    content_text = "".join(
        block.get("text", "") for block in content_blocks if block.get("type") == "text"
    )
    usage = data.get("usage", {})

    return ProviderResponse(
        content=content_text,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        model=data.get("model", model),
        provider="anthropic",
        raw_response=data,
        latency_ms=latency,
        finish_reason=data.get("stop_reason", "stop"),
    )


async def _anthropic_stream_gen(
    *, url: str, headers: dict, payload: dict, model: str
) -> AsyncGenerator[str, None]:
    """Convert Anthropic's SSE format to OpenAI-compatible SSE format."""
    import json as _json

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.is_error:
                body = await resp.aread()
                raise RuntimeError(
                    f"{resp.status_code} from anthropic for model '{model}': "
                    f"{body.decode(errors='replace')}"
                )

            response_model = model
            chunk_index = 0

            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    yield "data: [DONE]"
                    break

                try:
                    event = _json.loads(data_str)
                except _json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                # message_start carries model info
                if event_type == "message_start":
                    msg = event.get("message", {})
                    response_model = msg.get("model", model)

                # content_block_delta carries the actual token
                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    token = delta.get("text", "")
                    if token:
                        openai_chunk = {
                            "id": f"anth_{chunk_index}",
                            "object": "chat.completion.chunk",
                            "model": response_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": token},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {_json.dumps(openai_chunk)}"
                        chunk_index += 1

                # message_delta carries finish_reason
                elif event_type == "message_delta":
                    delta = event.get("delta", {})
                    stop_reason = delta.get("stop_reason")
                    if stop_reason:
                        finish_chunk = {
                            "id": f"anth_{chunk_index}",
                            "object": "chat.completion.chunk",
                            "model": response_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": stop_reason,
                                }
                            ],
                        }
                        yield f"data: {_json.dumps(finish_chunk)}"
                        chunk_index += 1

                elif event_type == "message_stop":
                    yield "data: [DONE]"
                    break


# ── Helpers ───────────────────────────────────────────────────────────────────

def _provider_to_backend(provider_name: str) -> str:
    """Map a logical provider name to its backend key."""
    mapping = {
        "openai":      "openai",
        "anthropic":   "anthropic",
        "claude":      "anthropic",
        "gemini":      "gemini",
        "google":      "gemini",
        "qwen":        "qwen",
        "dashscope":   "qwen",
        "local":       "local",
        "ollama":      "local",
        "vllm":        "local",
        "tgi":         "local",
        "huggingface": "huggingface",
        "hf":          "huggingface",
    }
    backend = mapping.get(provider_name.lower())
    if backend:
        return backend
    # Unknown provider — fall back to whichever key is configured
    return _auto_detect_backend()


def _auto_detect_backend() -> str:
    """Pick the first configured backend based on available API keys."""
    if settings.openai_api_key:
        return "openai"
    if settings.anthropic_api_key:
        return "anthropic"
    if settings.gemini_api_key:
        return "gemini"
    if settings.qwen_api_key:
        return "qwen"
    if settings.huggingface_api_key or settings.hf_token:
        return "huggingface"
    if settings.custom_inference_base_url:
        return "custom"
    return "huggingface"  # last resort


def _pick_hf_model(requested_model: str) -> str:
    """For HuggingFace routing, use requested model if it looks like an HF path,
    otherwise pick the first model from the pool."""
    if "/" in requested_model and not requested_model.startswith("gpt") and not requested_model.startswith("claude"):
        return requested_model
    # Pick first from pool (caller may shuffle for load balancing)
    return _HF_MODEL_POOL[0]


def _model_map() -> dict[str, str]:
    """Legacy model map from INFERENCE_MODEL_MAP env var."""
    if not settings.inference_model_map:
        return {}
    try:
        parsed = json.loads(settings.inference_model_map)
    except json.JSONDecodeError as exc:
        raise RuntimeError("INFERENCE_MODEL_MAP must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("INFERENCE_MODEL_MAP must be a JSON object")
    return {str(key): str(value) for key, value in parsed.items()}
