"""Provider Health Checker Service.

Probes each known provider's inference endpoint with a minimal
lightweight request to determine whether it is reachable and
returning valid responses.

Results are cached in-memory and refreshed every HEALTH_CHECK_INTERVAL_SECONDS
(default: 60 s) via a background asyncio loop that starts with the app.

Each ProviderHealth record contains:
  - provider     : canonical name ("openai", "anthropic", …)
  - status       : "operational" | "degraded" | "down" | "unknown"
  - latency_ms   : round-trip time of the last probe (0 if not checked)
  - last_checked : ISO-8601 timestamp of the last probe
  - message      : human-readable detail (error text on failure)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)

# ── Configurable thresholds ─────────────────────────────────────────────
HEALTH_CHECK_INTERVAL_SECONDS: int = 60   # how often to re-probe
LATENCY_DEGRADED_MS: int = 3_000         # above this → "degraded"
PROBE_TIMEOUT_SECONDS: float = 10.0       # per-request timeout

# ── Provider probe definitions ──────────────────────────────────────────
# Each entry: (canonical_name, probe_url, requires_auth_header)
# We probe the cheapest / smallest endpoint possible (models list, health,
# or a minimal chat completion with max_tokens=1).

_PROVIDERS_TO_CHECK: list[dict] = [
    {
        "name": "openai",
        "label": "OpenAI",
        "probe_url": "https://api.openai.com/v1/models",
        "method": "GET",
        "headers_fn": lambda: {"Authorization": f"Bearer {settings.openai_api_key}"},
        "requires_key": "openai_api_key",
    },
    {
        "name": "anthropic",
        "label": "Anthropic",
        "probe_url": "https://api.anthropic.com/v1/models",
        "method": "GET",
        "headers_fn": lambda: {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
        },
        "requires_key": "anthropic_api_key",
    },
    {
        "name": "gemini",
        "label": "Google Gemini",
        "probe_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "method": "GET",
        "headers_fn": lambda: {},
        "params_fn": lambda: {"key": settings.gemini_api_key},
        "requires_key": "gemini_api_key",
    },
    {
        "name": "huggingface",
        "label": "HuggingFace Inference",
        "probe_url": f"{settings.huggingface_base_url.rstrip('/').removesuffix('/v1')}/v1/models",
        "method": "GET",
        "headers_fn": lambda: {
            "Authorization": f"Bearer {settings.huggingface_api_key or settings.hf_token}"
        },
        "requires_key": "huggingface_api_key",
    },
    {
        "name": "qwen",
        "label": "Qwen / DashScope",
        "probe_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
        "method": "GET",
        "headers_fn": lambda: {"Authorization": f"Bearer {settings.qwen_api_key}"},
        "requires_key": "qwen_api_key",
    },
    {
        "name": "local",
        "label": "Local / Ollama",
        "probe_url": f"{settings.ollama_base_url.rstrip('/')}/api/tags",
        "method": "GET",
        "headers_fn": lambda: {},
        "requires_key": None,     # no key needed for local
        "is_local": True,         # connection-refused → "not running", not "down"
    },
]


# ── Data model ─────────────────────────────────────────────────────────

@dataclass
class ProviderHealth:
    provider: str
    label: str
    status: str           # "operational" | "degraded" | "down" | "unknown" | "unconfigured"
    latency_ms: int = 0
    last_checked: str = ""
    message: str = ""
    incident_count: int = 0   # consecutive failures since last success


# ── In-memory cache (refreshed by the background loop) ─────────────────

_health_cache: dict[str, ProviderHealth] = {}
_check_lock = asyncio.Lock()
_bg_task: asyncio.Task | None = None


def get_all_provider_health() -> list[ProviderHealth]:
    """Return a snapshot of the current health cache (used by the API)."""
    if not _health_cache:
        # Not yet probed — return "unknown" placeholders
        return [
            ProviderHealth(
                provider=p["name"],
                label=p["label"],
                status="unknown",
                message="Health check not yet run",
            )
            for p in _PROVIDERS_TO_CHECK
        ]
    return list(_health_cache.values())


# ── Individual provider probe ───────────────────────────────────────────

async def _probe_provider(cfg: dict) -> ProviderHealth:
    """Probe a single provider and return a ProviderHealth result."""
    name = cfg["name"]
    label = cfg["label"]
    probe_url = cfg["probe_url"]
    method = cfg.get("method", "GET")
    requires_key = cfg.get("requires_key")

    # Skip if the required API key is not configured
    if requires_key:
        key_value = getattr(settings, requires_key, "")
        if not key_value:
            return ProviderHealth(
                provider=name,
                label=label,
                status="unconfigured",
                message=f"No API key configured ({requires_key}). Set it in .env to enable health checks.",
                last_checked=_now(),
            )

    headers = cfg.get("headers_fn", lambda: {})()
    params = cfg.get("params_fn", lambda: {})()

    try:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS) as client:
            resp = await client.request(
                method,
                probe_url,
                headers=headers,
                params=params,
            )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        is_local = cfg.get("is_local", False)

        # For the local Ollama provider, a 503 means the model server is
        # running but has no models loaded / is starting up — treat as
        # "not running" (unconfigured) rather than "down".
        if is_local and resp.status_code >= 500:
            return ProviderHealth(
                provider=name,
                label=label,
                status="unconfigured",
                latency_ms=latency_ms,
                last_checked=_now(),
                message=(
                    f"Ollama returned HTTP {resp.status_code}. "
                    "It may still be starting up or have no models loaded. "
                    "Run `ollama pull <model>` to load a model."
                ),
                incident_count=0,
            )

        # Treat 2xx and 4xx (auth errors etc.) as "endpoint is reachable"
        # Only 5xx / connection errors → "down"
        if resp.status_code < 500:
            if latency_ms > LATENCY_DEGRADED_MS:
                status = "degraded"
                message = f"Reachable but slow ({latency_ms}ms > {LATENCY_DEGRADED_MS}ms threshold)"
            else:
                status = "operational"
                message = f"Responding normally ({latency_ms}ms)"
        else:
            status = "down"
            message = f"HTTP {resp.status_code}: {resp.text[:200]}"

        prev = _health_cache.get(name)
        incident_count = 0 if status == "operational" else ((prev.incident_count + 1) if prev else 1)

        return ProviderHealth(
            provider=name,
            label=label,
            status=status,
            latency_ms=latency_ms,
            last_checked=_now(),
            message=message,
            incident_count=incident_count,
        )

    except httpx.TimeoutException:
        prev = _health_cache.get(name)
        return ProviderHealth(
            provider=name,
            label=label,
            status="down",
            latency_ms=int(PROBE_TIMEOUT_SECONDS * 1000),
            last_checked=_now(),
            message=f"Request timed out after {PROBE_TIMEOUT_SECONDS}s",
            incident_count=(prev.incident_count + 1) if prev else 1,
        )
    except (httpx.ConnectError, ConnectionRefusedError, OSError) as exc:
        # For local providers (Ollama), a connection-refused means the server
        # simply isn't running — report as "not_running" rather than "down".
        is_local = cfg.get("is_local", False)
        if is_local:
            return ProviderHealth(
                provider=name,
                label=label,
                status="unconfigured",
                latency_ms=0,
                last_checked=_now(),
                message=(
                    f"Ollama is not running at {probe_url}. "
                    "Start it with `ollama serve` to enable local model inference."
                ),
                incident_count=0,
            )
        prev = _health_cache.get(name)
        return ProviderHealth(
            provider=name,
            label=label,
            status="down",
            latency_ms=0,
            last_checked=_now(),
            message=str(exc)[:300],
            incident_count=(prev.incident_count + 1) if prev else 1,
        )
    except Exception as exc:
        prev = _health_cache.get(name)
        return ProviderHealth(
            provider=name,
            label=label,
            status="down",
            latency_ms=0,
            last_checked=_now(),
            message=str(exc)[:300],
            incident_count=(prev.incident_count + 1) if prev else 1,
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Full health check sweep ─────────────────────────────────────────────

async def run_health_checks() -> None:
    """Probe all providers concurrently and update the in-memory cache."""
    async with _check_lock:
        results = await asyncio.gather(
            *[_probe_provider(cfg) for cfg in _PROVIDERS_TO_CHECK],
            return_exceptions=True,
        )
        for item in results:
            if isinstance(item, ProviderHealth):
                _health_cache[item.provider] = item
                logger.debug(
                    "Health[%s] = %s (%dms)", item.provider, item.status, item.latency_ms
                )
            else:
                logger.warning("Health check probe raised an exception: %s", item)


# ── Background loop ─────────────────────────────────────────────────────

async def _background_loop() -> None:
    """Runs forever: probe all providers, sleep, repeat."""
    while True:
        try:
            await run_health_checks()
        except Exception:
            logger.exception("Unexpected error in health check loop")
        await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)


def start_background_health_checks() -> None:
    """Start the background health-check loop (called from app lifespan)."""
    global _bg_task
    if _bg_task is None or _bg_task.done():
        _bg_task = asyncio.create_task(_background_loop())
        logger.info(
            "Provider health check loop started (interval=%ds)", HEALTH_CHECK_INTERVAL_SECONDS
        )


def stop_background_health_checks() -> None:
    """Cancel the background health-check loop (called on app shutdown)."""
    global _bg_task
    if _bg_task and not _bg_task.done():
        _bg_task.cancel()
        logger.info("Provider health check loop stopped")
