import httpx
import pytest
import respx

from llm.base import FatalError, RetryableError
from llm.providers import GeminiProvider, NimProvider, OllamaProvider

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)
NIM_BASE = "https://integrate.api.nvidia.com/v1"
NIM_URL = f"{NIM_BASE}/chat/completions"
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE}/api/chat"

PROMPT = [{"role": "user", "content": "di hola"}]


def make_gemini():
    return GeminiProvider(api_key="k", model="gemini-2.5-flash"), GEMINI_URL


def make_nim():
    return NimProvider(api_key="k", base_url=NIM_BASE, model="meta/llama-3.3-70b-instruct"), NIM_URL


def make_ollama():
    return OllamaProvider(base_url=OLLAMA_BASE, model="qwen3:30b-a3b"), OLLAMA_URL


ALL_PROVIDERS = [make_gemini, make_nim, make_ollama]


# --- happy paths: same LLMResponse shape from three different wire formats ---


@respx.mock
def test_gemini_returns_text_and_token_counts():
    respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "hola"}]}}],
                "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3},
            },
        )
    )

    provider, _ = make_gemini()
    response = provider.generate(PROMPT)

    assert response.text == "hola"
    assert response.provider == "gemini"
    assert response.model == "gemini-2.5-flash"
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 3
    assert response.latency_ms >= 0


@respx.mock
def test_nim_returns_text_and_token_counts():
    respx.post(NIM_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "hola"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )
    )

    provider, _ = make_nim()
    response = provider.generate(PROMPT)

    assert response.text == "hola"
    assert response.provider == "nim"
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 3


@respx.mock
def test_ollama_returns_text_and_token_counts():
    respx.post(OLLAMA_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hola"},
                "prompt_eval_count": 11,
                "eval_count": 3,
            },
        )
    )

    provider, _ = make_ollama()
    response = provider.generate(PROMPT)

    assert response.text == "hola"
    assert response.provider == "ollama"
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 3


# --- error mapping: identical semantics across all three ---


@pytest.mark.parametrize("factory", ALL_PROVIDERS)
@pytest.mark.parametrize("status", [429, 500, 502, 503])
@respx.mock
def test_transient_http_status_raises_retryable(factory, status):
    provider, url = factory()
    respx.post(url).mock(return_value=httpx.Response(status, json={"error": "nope"}))

    with pytest.raises(RetryableError):
        provider.generate(PROMPT)


@pytest.mark.parametrize("factory", ALL_PROVIDERS)
@pytest.mark.parametrize("status", [400, 401, 403, 404])
@respx.mock
def test_client_http_status_raises_fatal(factory, status):
    provider, url = factory()
    respx.post(url).mock(return_value=httpx.Response(status, json={"error": "nope"}))

    with pytest.raises(FatalError):
        provider.generate(PROMPT)


@pytest.mark.parametrize("factory", ALL_PROVIDERS)
@respx.mock
def test_network_failure_raises_retryable(factory):
    provider, url = factory()
    respx.post(url).mock(side_effect=httpx.ConnectError("connection refused"))

    with pytest.raises(RetryableError):
        provider.generate(PROMPT)


@pytest.mark.parametrize("factory", ALL_PROVIDERS)
@respx.mock
def test_timeout_raises_retryable(factory):
    provider, url = factory()
    respx.post(url).mock(side_effect=httpx.ReadTimeout("too slow"))

    with pytest.raises(RetryableError):
        provider.generate(PROMPT)


@pytest.mark.parametrize("factory", ALL_PROVIDERS)
@respx.mock
def test_unparseable_body_raises_retryable(factory):
    """A 200 that isn't JSON is a broken response, not a broken request."""
    provider, url = factory()
    respx.post(url).mock(return_value=httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(RetryableError):
        provider.generate(PROMPT)


# --- request shaping ---


@respx.mock
def test_gemini_sends_system_message_as_system_instruction():
    route = respx.post(GEMINI_URL).mock(
        return_value=httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        )
    )

    provider, _ = make_gemini()
    provider.generate([{"role": "system", "content": "eres conciso"}, *PROMPT])

    body = route.calls.last.request.read().decode()
    assert "systemInstruction" in body
    assert "eres conciso" in body


@respx.mock
def test_force_json_asks_each_provider_for_json():
    for factory, marker in (
        (make_gemini, "application/json"),
        (make_nim, "json_object"),
        (make_ollama, '"format":"json"'),
    ):
        provider, url = factory()
        route = respx.post(url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                    "choices": [{"message": {"content": "{}"}}],
                    "message": {"content": "{}"},
                },
            )
        )
        provider.generate(PROMPT, force_json=True)
        body = route.calls.last.request.read().decode().replace(" ", "")
        assert marker.replace(" ", "") in body


@respx.mock
def test_a_reply_that_is_all_reasoning_and_no_answer_is_an_error():
    """Ollama''s num_predict budget covers the thinking tokens too. A reasoning
    model on a long prompt can spend the whole budget deliberating and return
    an empty `content` alongside a full `thinking` -- a 200 OK carrying nothing.
    Passing that on as "" made the Studio draft a section, log 2048 completion
    tokens, and display an empty box."""
    provider, url = make_ollama()
    respx.post(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"content": "", "thinking": "Let me consider the structure..."},
                "eval_count": 2048,
            },
        )
    )

    with pytest.raises(FatalError, match="reasoning"):
        provider.generate(PROMPT, max_tokens=2048)


@respx.mock
def test_an_empty_reply_with_no_reasoning_behind_it_is_left_alone():
    """A model that simply had nothing to say is not the same failure, and the
    callers that ask for JSON already handle an unparseable empty string."""
    provider, url = make_ollama()
    respx.post(url).mock(return_value=httpx.Response(200, json={"message": {"content": ""}}))

    assert provider.generate(PROMPT).text == ""
