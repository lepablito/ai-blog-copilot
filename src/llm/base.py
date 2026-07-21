"""Shared vocabulary for every LLM provider.

The error hierarchy is the important part: `RetryableError` means "same
provider, try again in a moment", `FatalError` means "this provider will never
answer this request — move on". Everything the client does about backoff and
failover keys off that distinction.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, str]


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    raw: dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """Base for every provider failure."""


class RetryableError(LLMError):
    """Transient: timeouts, connection resets, 429, 5xx. Worth another attempt."""


class FatalError(LLMError):
    """Permanent for this request: bad credentials, malformed request, 4xx."""


class Provider(Protocol):
    name: str
    model: str

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = ...,
        max_tokens: int = ...,
        force_json: bool = ...,
    ) -> LLMResponse: ...
