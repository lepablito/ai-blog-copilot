"""The single entry point every other module uses to talk to a model.

Two failure axes, deliberately kept separate:

* *Within* a provider — retry with exponential backoff, but only for errors
  that a retry could plausibly fix (`RetryableError`).
* *Across* providers — walk down the chain. A `FatalError` (bad key, malformed
  request) skips the remaining attempts entirely: waiting will not mint a valid
  API key.

Only when every tier is exhausted does `AllProvidersFailed` surface, carrying
the last error from each provider so the cause is never guesswork.
"""

import json
import random
import re
import time
from collections.abc import Callable
from typing import Any

from .base import FatalError, LLMError, LLMResponse, Message, Provider, RetryableError

FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

REPAIR_INSTRUCTION = (
    "Your previous reply could not be parsed as JSON.\n"
    "Parser error: {error}\n\n"
    "Reply again with the same content as a single valid JSON object or array. "
    "Output raw JSON only — no prose, no markdown fences, no trailing commas."
)


class AllProvidersFailed(LLMError):
    """Every tier in the chain failed. Carries the last error from each."""


class JSONRepairFailed(LLMError):
    """The model returned unparseable JSON twice — original attempt and repair."""


class LLMClient:
    def __init__(
        self,
        providers: list[Provider],
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        jitter: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        recorder: Callable[..., None] | None = None,
    ):
        self._providers = providers
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._jitter = jitter
        self._sleep = sleep
        self._clock = clock
        self._recorder = recorder
        self._last_call: dict[str, float] = {}

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self._providers]

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        force_json: bool = False,
        purpose: str = "",
    ) -> LLMResponse:
        failures: dict[str, str] = {}

        for provider in self._providers:
            for attempt in range(1, self._max_attempts + 1):
                self._pace(provider)
                try:
                    response = provider.generate(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        force_json=force_json,
                    )
                except FatalError as exc:
                    failures[provider.name] = str(exc)
                    self._record(provider, purpose, ok=False, error=exc)
                    break  # a retry cannot fix this — next provider
                except RetryableError as exc:
                    failures[provider.name] = str(exc)
                    self._record(provider, purpose, ok=False, error=exc)
                    if attempt < self._max_attempts:
                        self._sleep(self._backoff(attempt, exc))
                else:
                    self._record(provider, purpose, ok=True, response=response)
                    return response

        raise AllProvidersFailed(_describe(failures))

    def generate_json(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        purpose: str = "",
    ) -> Any:
        """Like `generate`, but guarantees a parsed object or list.

        One repair round only. If the model cannot produce valid JSON twice in
        a row, the caller needs to know rather than sit in a loop burning
        tokens.
        """
        response = self.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            force_json=True,
            purpose=purpose,
        )
        try:
            return parse_json(response.text)
        except ValueError as first_error:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": REPAIR_INSTRUCTION.format(error=first_error)},
            ]
            repaired = self.generate(
                repair_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                force_json=True,
                purpose=f"{purpose}:repair" if purpose else "repair",
            )
            try:
                return parse_json(repaired.text)
            except ValueError as second_error:
                raise JSONRepairFailed(
                    f"invalid JSON after one repair attempt: {second_error}"
                ) from second_error

    def _pace(self, provider: Provider) -> None:
        """Wait until this provider's own rate limit allows another call.

        Enforced here rather than inside each provider so that every wait in
        the system goes through one injected sleep, and so that a chain that
        falls through to a second tier does not inherit the first one's clock.

        Time already spent counts: the model takes seconds to answer and the
        agent spends more running tools, so sleeping a full interval on top of
        that would pace the run at half the rate the quota allows.
        """
        interval = getattr(provider, "min_interval", 0.0)
        if not interval:
            return

        last = self._last_call.get(provider.name)
        if last is not None:
            remaining = interval - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)
        self._last_call[provider.name] = self._clock()

    def _backoff(self, attempt: int, error: RetryableError | None = None) -> float:
        """How long to wait before trying the same provider again.

        A rate limit knows its own window, so when the error carries one that
        number wins outright. Exponential backoff is for the case nobody knows:
        a struggling server, a dropped connection.
        """
        if error is not None and error.retry_after is not None:
            return error.retry_after
        return self._base_delay * (2 ** (attempt - 1)) + random.uniform(0, self._jitter)

    def _record(
        self,
        provider: Provider,
        purpose: str,
        *,
        ok: bool,
        response: LLMResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        if self._recorder is None:
            return
        self._recorder(
            provider=provider.name,
            model=provider.model,
            purpose=purpose,
            ok=ok,
            error_type=type(error).__name__ if error else None,
            latency_ms=response.latency_ms if response else 0,
            prompt_tokens=response.prompt_tokens if response else 0,
            completion_tokens=response.completion_tokens if response else 0,
        )


def parse_json(text: str) -> Any:
    """Parse a model reply as JSON, tolerating markdown fences.

    Bare scalars are rejected: `42` parses fine but is never a valid tool call
    or topic list, and letting it through only defers the failure.
    """
    stripped = text.strip()
    fenced = FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1)

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{exc.msg} at line {exc.lineno} column {exc.colno}") from exc

    if not isinstance(value, dict | list):
        raise ValueError(f"expected a JSON object or array, got {type(value).__name__}")
    return value


def _describe(failures: dict[str, str]) -> str:
    if not failures:
        return "no providers configured"
    detail = "; ".join(f"{name}: {error}" for name, error in failures.items())
    return f"all {len(failures)} provider(s) failed — {detail}"
