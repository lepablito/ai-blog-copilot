"""One class per provider, all speaking plain REST over httpx.

No vendor SDKs: three providers with three different wire formats is little
enough code to own outright, and owning it keeps the whole chain mockable with
a single tool (respx) and free of dependency drift.

Every provider funnels through `_post`, so the retryable/fatal decision is made
in exactly one place and all three behave identically under failure.
"""

import time

import httpx

from .base import FatalError, LLMResponse, Message, RetryableError

# 408 and 429 are the 4xx codes worth waiting out; every 5xx is the server's
# problem, not the request's.
RETRYABLE_STATUS = {408, 429}


def _post(url: str, *, payload: dict, headers: dict[str, str], timeout: float) -> tuple[dict, int]:
    """POST JSON and return (decoded body, latency in ms).

    Raises RetryableError or FatalError — never a raw httpx exception, so
    callers only ever handle the two cases they can act on.
    """
    started = time.perf_counter()
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.TransportError as exc:  # timeouts, DNS, refused connections
        raise RetryableError(f"transport failure: {type(exc).__name__}: {exc}") from exc
    latency_ms = int((time.perf_counter() - started) * 1000)

    status = response.status_code
    if status in RETRYABLE_STATUS or status >= 500:
        raise RetryableError(f"HTTP {status}: {response.text[:200]}")
    if status >= 400:
        raise FatalError(f"HTTP {status}: {response.text[:200]}")

    try:
        data = response.json()
    except ValueError as exc:
        # A 200 carrying HTML is a proxy or gateway hiccup — worth retrying.
        raise RetryableError(f"response body is not JSON: {response.text[:200]}") from exc

    return data, latency_ms


class GeminiProvider:
    """Tier 1. Native Google REST API — its own message format."""

    name = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", *, timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self._timeout = timeout

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        force_json: bool = False,
    ) -> LLMResponse:
        generation_config: dict[str, object] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if force_json:
            generation_config["responseMimeType"] = "application/json"

        # Gemini has no "system" role — system text rides in its own field, and
        # "assistant" is spelled "model".
        payload: dict[str, object] = {
            "contents": [
                {
                    "role": "model" if m["role"] == "assistant" else "user",
                    "parts": [{"text": m["content"]}],
                }
                for m in messages
                if m["role"] != "system"
            ],
            "generationConfig": generation_config,
        }
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        data, latency_ms = _post(
            f"{self.BASE_URL}/models/{self.model}:generateContent",
            payload=payload,
            headers={"x-goog-api-key": self.api_key},
            timeout=self._timeout,
        )

        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text=_gemini_text(data),
            provider=self.name,
            model=self.model,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0),
            latency_ms=latency_ms,
            raw=data,
        )


class NimProvider:
    """Tier 2. NVIDIA NIM speaks the OpenAI chat-completions dialect."""

    name = "nim"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "meta/llama-3.3-70b-instruct",
        *,
        timeout: float = 60.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        force_json: bool = False,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if force_json:
            payload["response_format"] = {"type": "json_object"}

        data, latency_ms = _post(
            f"{self.base_url}/chat/completions",
            payload=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self._timeout,
        )

        usage = data.get("usage") or {}
        return LLMResponse(
            text=_openai_text(data),
            provider=self.name,
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            raw=data,
        )


class OllamaProvider:
    """Tier 3. Local models — no key, and the only tier that survives no network."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:30b-a3b",
        *,
        timeout: float = 300.0,  # local generation on CPU is slow, not broken
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = timeout

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        force_json: bool = False,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if force_json:
            payload["format"] = "json"

        data, latency_ms = _post(
            f"{self.base_url}/api/chat",
            payload=payload,
            headers={},
            timeout=self._timeout,
        )

        return LLMResponse(
            text=(data.get("message") or {}).get("content", ""),
            provider=self.name,
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            raw=data,
        )


def _gemini_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts)


def _openai_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content", "") or ""
