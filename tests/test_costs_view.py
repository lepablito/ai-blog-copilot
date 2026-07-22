"""Aggregations behind the Costs tab.

The point of this tab is not the money — three tiers of which two are free
means the bill is rounding error. It is that a fallback chain can silently
degrade: the primary fails for a week, the second tier quietly answers
everything, and every run stays green. These numbers are how that shows up.
"""

import sqlite3

import pytest

from llm.calls_log import CallLog
from studio.costs_data import by_purpose, daily_costs, percentile, provider_stats, totals


@pytest.fixture
def db(tmp_path):
    return tmp_path / "radar.db"


def log_call(
    log: CallLog,
    *,
    provider: str = "gemini",
    model: str = "gemini-2.5-flash",
    purpose: str = "radar:step",
    ok: bool = True,
    latency_ms: int = 100,
    prompt_tokens: int = 1000,
    completion_tokens: int = 100,
) -> None:
    log.record(
        provider=provider,
        model=model,
        purpose=purpose,
        ok=ok,
        error_type=None if ok else "RetryableError",
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def test_percentile_picks_the_nearest_rank():
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40


def test_percentile_of_nothing_is_zero_rather_than_a_crash():
    """A provider configured but never reached has no latencies at all, and
    that is a normal state, not an error."""
    assert percentile([], 50) == 0


def test_stats_report_the_success_rate_per_provider(db):
    log = CallLog(db)
    log_call(log, provider="gemini", ok=True)
    log_call(log, provider="gemini", ok=False)
    log_call(log, provider="gemini", ok=True)
    log_call(log, provider="nim", ok=True)

    stats = {row["provider"]: row for row in provider_stats(db)}

    assert stats["gemini"]["calls"] == 3
    assert stats["gemini"]["failures"] == 1
    assert stats["gemini"]["success_rate"] == pytest.approx(2 / 3)
    assert stats["nim"]["success_rate"] == 1.0


def test_latency_percentiles_ignore_failed_calls(db):
    """A failure records latency 0 — the exception fires before the timer is
    read. Averaging those in would make a provider look faster the more it
    breaks, which is exactly backwards."""
    log = CallLog(db)
    log_call(log, latency_ms=1000, ok=True)
    log_call(log, latency_ms=2000, ok=True)
    log_call(log, latency_ms=0, ok=False)
    log_call(log, latency_ms=0, ok=False)

    [row] = provider_stats(db)

    assert row["p50_ms"] == 1000
    assert row["p95_ms"] == 2000


def test_a_provider_that_only_ever_failed_reports_no_latency(db):
    log = CallLog(db)
    log_call(log, ok=False, latency_ms=0)

    [row] = provider_stats(db)

    assert row["p50_ms"] == 0
    assert row["success_rate"] == 0.0


def test_cost_uses_the_price_table_and_free_tiers_stay_free(db):
    log = CallLog(db)
    log_call(
        log,
        provider="gemini",
        model="gemini-2.5-flash",
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )
    log_call(
        log, provider="ollama", model="qwen3:30b-a3b", prompt_tokens=1_000_000, completion_tokens=0
    )

    stats = {row["provider"]: row for row in provider_stats(db)}

    assert stats["gemini"]["est_cost_usd"] == pytest.approx(0.30)
    assert stats["ollama"]["est_cost_usd"] == 0.0


def test_totals_add_up_across_providers(db):
    log = CallLog(db)
    log_call(log, provider="gemini", ok=True)
    log_call(log, provider="nim", ok=False)

    summary = totals(provider_stats(db))

    assert summary["calls"] == 2
    assert summary["failures"] == 1
    assert summary["success_rate"] == 0.5


def test_totals_of_an_empty_history_do_not_divide_by_zero(db):
    summary = totals([])

    assert summary["calls"] == 0
    assert summary["success_rate"] == 0.0


def test_costs_are_grouped_by_day_for_a_trend(db):
    log = CallLog(db)
    log_call(log)
    log_call(log)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE llm_calls SET created_at = '2026-07-20T10:00:00+00:00' WHERE id = 1")

    days = {row["date"]: row for row in daily_costs(db)}

    assert days["2026-07-20"]["calls"] == 1
    assert len(days) == 2


def test_purposes_separate_the_agent_from_the_studio(db):
    """The radar runs unattended and the Studio runs while someone waits. One
    number covering both hides which of them is spending."""
    log = CallLog(db)
    log_call(log, purpose="radar:step")
    log_call(log, purpose="radar:step")
    log_call(log, purpose="studio:section")

    purposes = {row["purpose"]: row["calls"] for row in by_purpose(db)}

    assert purposes == {"radar:step": 2, "studio:section": 1}


def test_a_missing_database_reads_as_empty_and_is_not_created(db):
    assert provider_stats(db) == []
    assert daily_costs(db) == []
    assert by_purpose(db) == []
    assert not db.exists()
