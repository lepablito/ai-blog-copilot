import sqlite3

import pytest

from llm.base import RetryableError
from llm.calls_log import CallLog
from llm.client import AllProvidersFailed, LLMClient
from tests.fakes import FakeProvider

PROMPT = [{"role": "user", "content": "hola"}]


@pytest.fixture
def log(tmp_path):
    return CallLog(tmp_path / "radar.db")


def rows(log):
    with sqlite3.connect(log.db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM llm_calls ORDER BY id")]


def test_records_a_successful_call(log):
    log.record(
        provider="gemini",
        model="gemini-2.5-flash",
        purpose="radar",
        ok=True,
        error_type=None,
        latency_ms=420,
        prompt_tokens=1000,
        completion_tokens=200,
    )

    (row,) = rows(log)
    assert row["provider"] == "gemini"
    assert row["model"] == "gemini-2.5-flash"
    assert row["purpose"] == "radar"
    assert row["ok"] == 1
    assert row["error_type"] is None
    assert row["latency_ms"] == 420
    assert row["prompt_tokens"] == 1000
    assert row["completion_tokens"] == 200
    assert row["created_at"]


def test_records_a_failed_call_with_its_error_type(log):
    log.record(
        provider="gemini",
        model="gemini-2.5-flash",
        purpose="radar",
        ok=False,
        error_type="RetryableError",
        latency_ms=0,
        prompt_tokens=0,
        completion_tokens=0,
    )

    (row,) = rows(log)
    assert row["ok"] == 0
    assert row["error_type"] == "RetryableError"


def test_priced_model_gets_a_non_zero_cost_estimate(log):
    log.record(
        provider="gemini",
        model="gemini-2.5-flash",
        purpose="radar",
        ok=True,
        error_type=None,
        latency_ms=1,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )

    (row,) = rows(log)
    assert row["est_cost_usd"] > 0


def test_local_model_is_free(log):
    log.record(
        provider="ollama",
        model="qwen3:30b-a3b",
        purpose="radar",
        ok=True,
        error_type=None,
        latency_ms=1,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )

    (row,) = rows(log)
    assert row["est_cost_usd"] == 0


def test_opening_the_same_database_twice_is_safe(tmp_path):
    first = CallLog(tmp_path / "radar.db")
    first.record(
        provider="ollama",
        model="m",
        purpose="",
        ok=True,
        error_type=None,
        latency_ms=1,
        prompt_tokens=1,
        completion_tokens=1,
    )
    second = CallLog(tmp_path / "radar.db")

    assert len(rows(second)) == 1, "re-opening must not wipe or duplicate the schema"


# --- wired into the client: every attempt leaves a trace ---


def test_client_logs_one_row_per_attempt_including_failures(log):
    provider = FakeProvider("gemini", [RetryableError("429"), RetryableError("429"), "ok"])
    client = LLMClient([provider], max_attempts=3, sleep=lambda _s: None, recorder=log.record)

    client.generate(PROMPT, purpose="radar")

    recorded = rows(log)
    assert [r["ok"] for r in recorded] == [0, 0, 1]
    assert all(r["purpose"] == "radar" for r in recorded)


def test_client_logs_the_fallback_hop(log):
    primary = FakeProvider("gemini", [RetryableError("503")])
    secondary = FakeProvider("nim", ["rescued"])
    client = LLMClient(
        [primary, secondary], max_attempts=2, sleep=lambda _s: None, recorder=log.record
    )

    client.generate(PROMPT, purpose="radar")

    recorded = rows(log)
    assert [(r["provider"], r["ok"]) for r in recorded] == [
        ("gemini", 0),
        ("gemini", 0),
        ("nim", 1),
    ]


def test_total_failure_is_still_fully_logged(log):
    provider = FakeProvider("gemini", [RetryableError("503")])
    client = LLMClient([provider], max_attempts=2, sleep=lambda _s: None, recorder=log.record)

    with pytest.raises(AllProvidersFailed):
        client.generate(PROMPT, purpose="radar")

    assert len(rows(log)) == 2
