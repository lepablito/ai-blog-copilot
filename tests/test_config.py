import pytest

from llm.config import NoProvidersConfigured, build_chain

FULL_ENV = {
    "GEMINI_API_KEY": "g",
    "NVIDIA_NIM_API_KEY": "n",
    "OLLAMA_BASE_URL": "http://localhost:11434",
}


def test_chain_order_is_nim_then_gemini_then_ollama():
    """NIM leads: Gemini's free tier is 20 requests a day, which one radar pass
    nearly exhausts, so it is held in reserve as the fallback rather than spent
    as the default. Ollama stays the floor — the only offline tier."""
    assert [p.name for p in build_chain(FULL_ENV)] == ["nim", "gemini", "ollama"]


def test_provider_without_a_key_is_left_out_of_the_chain():
    """Better to know the tier is missing at startup than mid-run."""
    env = {**FULL_ENV, "NVIDIA_NIM_API_KEY": ""}

    assert [p.name for p in build_chain(env)] == ["gemini", "ollama"]


def test_ollama_needs_no_key():
    assert [p.name for p in build_chain({})] == ["ollama"]


def test_llm_only_restricts_the_chain_to_one_tier():
    chain = build_chain({**FULL_ENV, "LLM_ONLY": "ollama"})

    assert [p.name for p in chain] == ["ollama"]


def test_llm_only_naming_a_tier_without_credentials_fails_loudly():
    with pytest.raises(NoProvidersConfigured):
        build_chain({"LLM_ONLY": "gemini"})


def test_hosted_providers_get_a_timeout_that_fits_a_full_agent_step():
    """NIM timed out on all three attempts in CI, returning zero tokens, while
    working locally on smaller prompts. By late in a run the prompt carries
    every observation so far — 20k tokens — and a 70B model does not answer
    that inside a chat-sized 60 seconds."""
    nim, gemini, _ollama = build_chain(FULL_ENV)

    assert nim._timeout >= 120
    assert gemini._timeout >= 120


def test_models_come_from_the_environment():
    chain = build_chain({**FULL_ENV, "GEMINI_MODEL": "gemini-2.5-pro"})
    gemini = next(p for p in chain if p.name == "gemini")

    assert gemini.model == "gemini-2.5-pro"
