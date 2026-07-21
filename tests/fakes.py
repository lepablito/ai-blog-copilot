"""Scripted stand-in for a Provider.

Lets the client tests drive failure sequences without touching httpx at all —
the wire format is already covered in test_providers.py.
"""

from llm.base import LLMResponse, Message


class FakeProvider:
    """Replays `script` in order; the last entry repeats once exhausted.

    An entry is either a string (returned as response text) or an exception
    instance (raised). `script=[RetryableError("x")]` therefore means "always
    fails with a retryable error".
    """

    def __init__(self, name: str, script: list, *, model: str = "fake-1"):
        if not script:
            raise ValueError("script must not be empty")
        self.name = name
        self.model = model
        self._script = list(script)
        self._index = 0
        self.calls = 0
        self.last_messages: list[Message] | None = None
        self.last_force_json: bool | None = None

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        force_json: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        self.last_messages = messages
        self.last_force_json = force_json

        entry = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1

        if isinstance(entry, Exception):
            raise entry
        return LLMResponse(
            text=entry,
            provider=self.name,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1,
        )
