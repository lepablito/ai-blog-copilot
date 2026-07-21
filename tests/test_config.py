import pytest

from llm.config import NoProvidersConfigured, build_chain

FULL_ENV = {
    "GEMINI_API_KEY": "g",
    "NVIDIA_NIM_API_KEY": "n",
    "OLLAMA_BASE_URL": "http://localhost:11434",
}


def test_chain_order_is_gemini_then_nim_then_ollama():
    assert [p.name for p in build_chain(FULL_ENV)] == ["gemini", "nim", "ollama"]


def test_provider_without_a_key_is_left_out_of_the_chain():
    """Better to know the tier is missing at startup than mid-run."""
    env = {**FULL_ENV, "GEMINI_API_KEY": ""}

    assert [p.name for p in build_chain(env)] == ["nim", "ollama"]


def test_ollama_needs_no_key():
    assert [p.name for p in build_chain({})] == ["ollama"]


def test_llm_only_restricts_the_chain_to_one_tier():
    chain = build_chain({**FULL_ENV, "LLM_ONLY": "ollama"})

    assert [p.name for p in chain] == ["ollama"]


def test_llm_only_naming_a_tier_without_credentials_fails_loudly():
    with pytest.raises(NoProvidersConfigured):
        build_chain({"LLM_ONLY": "gemini"})


def test_models_come_from_the_environment():
    chain = build_chain({**FULL_ENV, "GEMINI_MODEL": "gemini-2.5-pro"})

    assert chain[0].model == "gemini-2.5-pro"
