"""Abstract base class for all LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
        """Execute a chat completion request against the upstream provider."""
