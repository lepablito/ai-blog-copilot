import pytest

from llm.calls_log import CallLog, format_summary


@pytest.fixture
def log(tmp_path):
    return CallLog(tmp_path / "radar.db")


def record(log, provider="gemini", ok=True, error_type=None, model="gemini-2.5-flash", **kwargs):
    log.record(
        provider=provider,
        model=model,
        purpose=kwargs.get("purpose", "radar"),
        ok=ok,
        error_type=error_type,
        latency_ms=kwargs.get("latency_ms", 100),
        prompt_tokens=kwargs.get("prompt_tokens", 1000),
        completion_tokens=kwargs.get("completion_tokens", 100),
    )


def test_an_empty_log_summarises_to_nothing(log):
    assert log.summary_by_provider() == []


def test_calls_are_grouped_by_provider(log):
    record(log, provider="gemini")
    record(log, provider="gemini")
    record(log, provider="ollama", model="qwen3:4b")

    summary = {row["provider"]: row for row in log.summary_by_provider()}

    assert summary["gemini"]["calls"] == 2
    assert summary["ollama"]["calls"] == 1


def test_failures_are_counted_separately(log):
    record(log, ok=True)
    record(log, ok=False, error_type="RetryableError")
    record(log, ok=False, error_type="FatalError")

    (row,) = log.summary_by_provider()

    assert row["calls"] == 3
    assert row["failures"] == 2


def test_tokens_and_cost_are_summed(log):
    record(log, prompt_tokens=1_000_000, completion_tokens=1_000_000)
    record(log, prompt_tokens=1_000_000, completion_tokens=1_000_000)

    (row,) = log.summary_by_provider()

    assert row["prompt_tokens"] == 2_000_000
    assert row["completion_tokens"] == 2_000_000
    assert row["est_cost_usd"] > 0


def test_only_calls_since_a_cutoff_are_counted(log):
    """The workflow summarises one pass, not the whole cached history."""
    record(log, provider="gemini")
    cutoff = "2999-01-01T00:00:00+00:00"

    assert log.summary_by_provider(since=cutoff) == []


# --- the line that lands in the Actions log ---


def test_the_summary_line_names_the_provider_that_answered(log):
    record(log, provider="gemini")

    rendered = format_summary(log.summary_by_provider())

    assert "gemini" in rendered
    assert "1 call" in rendered


def test_the_summary_flags_a_provider_that_only_failed(log):
    """Gemini failing while NIM quietly covers is exactly the silent
    degradation this line exists to surface."""
    record(log, provider="gemini", ok=False, error_type="FatalError")
    record(log, provider="nim", ok=True, model="meta/llama-3.3-70b-instruct")

    rendered = format_summary(log.summary_by_provider())

    assert "gemini" in rendered and "nim" in rendered
    assert "2 failed" not in rendered
    assert "1 failed" in rendered


def test_an_empty_summary_says_so_rather_than_printing_a_blank(log):
    assert "no calls" in format_summary([]).lower()
