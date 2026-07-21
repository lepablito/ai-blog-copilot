"""Builds the provider chain from the environment.

A tier without credentials is dropped here, at construction time, and never
enters the chain. Discovering a missing key three retries into a live run
wastes time and muddies the logs.
"""

from collections.abc import Mapping

from .base import LLMError, Provider
from .providers import GeminiProvider, NimProvider, OllamaProvider


class NoProvidersConfigured(LLMError):
    """Nothing in the chain — every tier lacked credentials."""


def build_chain(env: Mapping[str, str]) -> list[Provider]:
    """Return the fallback chain in priority order: Gemini → NIM → Ollama.

    `LLM_ONLY` pins the chain to a single tier, which is how the test suite and
    the CI workflow avoid spending API credits when they only need one.
    """
    only = (env.get("LLM_ONLY") or "").strip().lower()
    chain: list[Provider] = []

    gemini_key = (env.get("GEMINI_API_KEY") or "").strip()
    if gemini_key:
        chain.append(
            GeminiProvider(
                api_key=gemini_key,
                model=env.get("GEMINI_MODEL") or "gemini-2.5-flash",
            )
        )

    nim_key = (env.get("NVIDIA_NIM_API_KEY") or "").strip()
    if nim_key:
        chain.append(
            NimProvider(
                api_key=nim_key,
                base_url=env.get("NVIDIA_NIM_BASE_URL") or "https://integrate.api.nvidia.com/v1",
                model=env.get("NVIDIA_NIM_MODEL") or "meta/llama-3.3-70b-instruct",
            )
        )

    # Ollama is the floor of the chain: local, unauthenticated, and the only
    # tier that still answers with no network at all.
    chain.append(
        OllamaProvider(
            base_url=env.get("OLLAMA_BASE_URL") or "http://localhost:11434",
            model=env.get("OLLAMA_MODEL") or "qwen3:30b-a3b",
        )
    )

    if only:
        chain = [p for p in chain if p.name == only]
        if not chain:
            raise NoProvidersConfigured(
                f"LLM_ONLY={only!r} but that provider has no credentials configured"
            )

    if not chain:
        raise NoProvidersConfigured("no provider could be configured from the environment")
    return chain
