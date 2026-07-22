"""Staying inside a quota instead of discovering it by being refused.

Gemini's free tier allows 5 requests per minute and 20 per day. A radar pass
makes 9 to 11 calls and, left alone, fires them as fast as the model answers —
the successful CI run on 2026-07-22 made 9 calls in about 40 seconds and 2 came
back refused.

Two separate mechanisms, because they solve different halves of the problem:
pacing stops us provoking a 429, and the backoff decides what to do when one
arrives anyway.
"""

import httpx
import pytest
import respx

from llm.base import RetryableError
from llm.client import LLMClient
from llm.providers import RATE_LIMIT_DELAY, GeminiProvider, NimProvider
from tests.fakes import FakeProvider

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)
PROMPT = [{"role": "user", "content": "hola"}]


class FakeClock:
    """A clock that only moves when something sleeps."""

    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


# --- pacing: do not provoke the limit in the first place ---


def test_a_paced_provider_waits_out_its_interval_between_calls():
    clock = FakeClock()
    provider = FakeProvider("gemini", ["{}"])
    provider.min_interval = 12.0
    client = LLMClient([provider], sleep=clock.sleep, clock=clock)

    client.generate(PROMPT)
    client.generate(PROMPT)

    assert clock.slept == [12.0]


def test_the_first_call_is_never_delayed():
    """Nothing has been spent yet. Making the very first call wait would add
    twelve seconds to every run for no reason at all."""
    clock = FakeClock()
    provider = FakeProvider("gemini", ["{}"])
    provider.min_interval = 12.0

    LLMClient([provider], sleep=clock.sleep, clock=clock).generate(PROMPT)

    assert clock.slept == []


def test_time_already_spent_elsewhere_counts_towards_the_interval():
    """The model itself takes seconds to answer, and the agent spends more
    running tools. Sleeping a full interval on top of that would pace the run
    at half the rate the quota actually allows."""
    clock = FakeClock()
    provider = FakeProvider("gemini", ["{}"])
    provider.min_interval = 12.0
    client = LLMClient([provider], sleep=clock.sleep, clock=clock)

    client.generate(PROMPT)
    clock.advance(9.0)
    client.generate(PROMPT)

    assert clock.slept == [3.0]


def test_a_provider_with_no_declared_limit_is_not_paced():
    """Local Ollama has no quota. It should not inherit Gemini's."""
    clock = FakeClock()
    client = LLMClient([FakeProvider("ollama", ["{}"])], sleep=clock.sleep, clock=clock)

    client.generate(PROMPT)
    client.generate(PROMPT)

    assert clock.slept == []


def test_each_provider_keeps_its_own_schedule():
    """Falling through to NIM must not start Gemini's clock, and vice versa."""
    clock = FakeClock()
    gemini = FakeProvider("gemini", ["{}"])
    gemini.min_interval = 12.0
    nim = FakeProvider("nim", ["{}"])
    client = LLMClient([gemini, nim], sleep=clock.sleep, clock=clock)

    client.generate(PROMPT)
    client._pace(nim)
    client._pace(nim)

    assert clock.slept == []


def test_gemini_is_paced_for_five_requests_per_minute_and_the_rest_are_not():
    assert GeminiProvider(api_key="k").min_interval == pytest.approx(12.0)
    assert NimProvider(api_key="k").min_interval == 0.0


# --- backoff: what to do when a 429 arrives anyway ---


@respx.mock
def test_a_429_carries_the_servers_own_retry_after():
    """Guessing is worse than reading. If the server says how long to wait,
    that is the number."""
    respx.post(GEMINI_URL).mock(return_value=httpx.Response(429, headers={"Retry-After": "37"}))

    with pytest.raises(RetryableError) as caught:
        GeminiProvider(api_key="k").generate(PROMPT)

    assert caught.value.retry_after == 37.0


@respx.mock
def test_a_429_with_no_hint_waits_long_enough_to_clear_a_per_minute_window():
    """The default backoff waits about a second. A per-minute quota does not
    care about a second, so all three attempts would burn inside the same
    window and the tier would be abandoned for no reason."""
    respx.post(GEMINI_URL).mock(return_value=httpx.Response(429))

    with pytest.raises(RetryableError) as caught:
        GeminiProvider(api_key="k").generate(PROMPT)

    assert caught.value.retry_after == RATE_LIMIT_DELAY
    assert RATE_LIMIT_DELAY >= 12


@respx.mock
def test_an_unparseable_retry_after_falls_back_rather_than_crashing():
    """Retry-After may legally be an HTTP date. Reading one is not worth the
    code; failing the whole call over it certainly is not."""
    respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    )

    with pytest.raises(RetryableError) as caught:
        GeminiProvider(api_key="k").generate(PROMPT)

    assert caught.value.retry_after == RATE_LIMIT_DELAY


def test_a_5xx_has_no_retry_after_and_keeps_exponential_backoff():
    """A struggling server is not a quota. Waiting twenty seconds for it would
    just make a bad run slower."""
    error = RetryableError("HTTP 503")

    assert error.retry_after is None


def test_the_client_honours_retry_after_instead_of_its_own_backoff():
    clock = FakeClock()
    provider = FakeProvider("gemini", [RetryableError("HTTP 429", retry_after=37.0), "{}"])
    client = LLMClient([provider], sleep=clock.sleep, clock=clock, base_delay=1.0, jitter=0.0)

    client.generate(PROMPT)

    assert clock.slept == [37.0]
