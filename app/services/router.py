"""Provider routing — resolves model + call_sdk to the correct adapter."""

from __future__ import annotations

from app.providers.anthropic import AnthropicProvider
from app.providers.base import BaseProvider
from app.providers.gemini import GeminiProvider
from app.providers.local import LocalProvider
from app.providers.openai import OpenAIProvider
from app.providers.qwen import QwenProvider

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
}

# ── SDK → provider heuristic ───────────────────────────────────────────

_SDK_TO_PROVIDER: dict[str, str] = {
    "openai-sdk": "openai",
    "claude-sdk": "anthropic",
    "anthropic-sdk": "anthropic",
    "gemini-sdk": "gemini",
    "qwen-sdk": "qwen",
    "raw-http": "openai",       # default to OpenAI-compatible format
    "vizhi-sdk": "openai",      # future SDK uses OpenAI-compatible wire format
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
    "llama": "local",
    "mistral": "local",
    "phi": "local",
    "deepseek": "local",
}

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


class ProviderRouter:
    """Resolves ``(model, call_sdk)`` to ``(provider_instance, model_name)``."""

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
        provider_name: str | None = None
        model_name = model

        # 1. Explicit prefix: "openai/gpt-4o-mini" → provider=openai, model=gpt-4o-mini
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

        return _get_provider(provider_name), model_name


# Module-level singleton.
provider_router = ProviderRouter()
