import pytest

from llm.base import FatalError, RetryableError
from llm.client import AllProvidersFailed, JSONRepairFailed, LLMClient
from tests.fakes import FakeProvider

PROMPT = [{"role": "user", "content": "hola"}]


def client(*providers, **kwargs):
    """A client whose backoff never actually sleeps."""
    kwargs.setdefault("sleep", lambda _seconds: None)
    return LLMClient(list(providers), **kwargs)


# --- the fallback chain ---


def test_uses_first_provider_when_it_works():
    primary = FakeProvider("gemini", ["ok"])
    secondary = FakeProvider("nim", ["should not be reached"])

    response = client(primary, secondary).generate(PROMPT)

    assert response.text == "ok"
    assert response.provider == "gemini"
    assert secondary.calls == 0


def test_falls_through_to_next_provider_when_first_is_exhausted():
    primary = FakeProvider("gemini", [RetryableError("503")])
    secondary = FakeProvider("nim", ["rescued"])

    response = client(primary, secondary, max_attempts=3).generate(PROMPT)

    assert response.text == "rescued"
    assert response.provider == "nim"
    assert primary.calls == 3, "primary should have used all its attempts first"


def test_retryable_error_is_retried_up_to_max_attempts():
    provider = FakeProvider("gemini", [RetryableError("429"), RetryableError("429"), "third time"])

    response = client(provider, max_attempts=3).generate(PROMPT)

    assert response.text == "third time"
    assert provider.calls == 3


def test_fatal_error_is_not_retried_and_moves_on_immediately():
    primary = FakeProvider("gemini", [FatalError("401 bad key")])
    secondary = FakeProvider("nim", ["rescued"])

    response = client(primary, secondary, max_attempts=3).generate(PROMPT)

    assert response.provider == "nim"
    assert primary.calls == 1, "a bad key will not fix itself — do not retry it"


def test_all_providers_failing_raises_with_per_provider_detail():
    primary = FakeProvider("gemini", [FatalError("401 bad key")])
    secondary = FakeProvider("nim", [RetryableError("503 upstream")])

    with pytest.raises(AllProvidersFailed) as excinfo:
        client(primary, secondary, max_attempts=2).generate(PROMPT)

    message = str(excinfo.value)
    assert "gemini" in message and "401 bad key" in message
    assert "nim" in message and "503 upstream" in message


def test_backoff_grows_exponentially_between_attempts():
    slept: list[float] = []
    provider = FakeProvider("gemini", [RetryableError("429"), RetryableError("429"), "ok"])

    LLMClient([provider], max_attempts=3, sleep=slept.append, base_delay=1.0, jitter=0.0).generate(
        PROMPT
    )

    assert slept == [1.0, 2.0]


def test_empty_chain_raises_immediately():
    with pytest.raises(AllProvidersFailed):
        client().generate(PROMPT)


# --- JSON handling ---


def test_generate_json_parses_a_plain_object():
    provider = FakeProvider("gemini", ['{"angle": "practical"}'])

    assert client(provider).generate_json(PROMPT) == {"angle": "practical"}


def test_generate_json_strips_markdown_fences():
    provider = FakeProvider("gemini", ['```json\n{"angle": "theoretical"}\n```'])

    assert client(provider).generate_json(PROMPT) == {"angle": "theoretical"}


def test_generate_json_asks_the_provider_for_json():
    provider = FakeProvider("gemini", ["{}"])

    client(provider).generate_json(PROMPT)

    assert provider.last_force_json is True


def test_generate_json_repairs_invalid_json_once():
    provider = FakeProvider("gemini", ["not json at all", '{"fixed": true}'])

    assert client(provider).generate_json(PROMPT) == {"fixed": True}
    assert provider.calls == 2


def test_repair_prompt_feeds_the_parse_error_back():
    provider = FakeProvider("gemini", ["not json at all", "{}"])

    client(provider).generate_json(PROMPT)

    repair_prompt = " ".join(m["content"] for m in provider.last_messages)
    assert "not json at all" in repair_prompt, "the model must see what it actually sent"
    assert "JSON" in repair_prompt


def test_generate_json_gives_up_after_one_repair():
    provider = FakeProvider("gemini", ["still not json"])

    with pytest.raises(JSONRepairFailed):
        client(provider).generate_json(PROMPT)

    assert provider.calls == 2, "one original attempt plus exactly one repair"


def test_generate_json_rejects_a_bare_scalar():
    """`42` is valid JSON but never a valid tool call or topic list."""
    provider = FakeProvider("gemini", ["42", "42"])

    with pytest.raises(JSONRepairFailed):
        client(provider).generate_json(PROMPT)
