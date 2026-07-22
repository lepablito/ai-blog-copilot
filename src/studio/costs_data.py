"""Aggregations over `llm_calls`. No Streamlit imports here on purpose.

The bill is not the interesting part — two of the three tiers are free, so the
money is rounding error. What these numbers are for is the failure mode a
fallback chain makes possible: the primary provider breaks, the second tier
quietly absorbs everything, and every run stays green for a week. A success
rate per tier is how that becomes visible.
"""

import math
import sqlite3
from pathlib import Path


def percentile(values: list[int], p: float) -> int:
    """Nearest-rank percentile. 0 for an empty series.

    Nearest-rank rather than interpolated: with the handful of calls a run
    makes, an interpolated p95 would invent a latency nothing ever took.
    """
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def provider_stats(db_path: Path | str, *, since: str | None = None) -> list[dict]:
    """Per provider: volume, reliability, tokens, cost and latency spread."""
    rows = _query(
        db_path,
        """
        SELECT provider, ok, latency_ms, prompt_tokens, completion_tokens, est_cost_usd
        FROM llm_calls
        """,
        since=since,
    )

    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["provider"], []).append(row)

    stats = []
    for provider, calls in sorted(grouped.items()):
        failures = sum(1 for c in calls if not c["ok"])
        # Only successful calls have a meaningful latency: a transport failure
        # raises before the timer is read and lands here as 0. Including those
        # would make a provider look faster the more it broke.
        latencies = [c["latency_ms"] for c in calls if c["ok"]]

        stats.append(
            {
                "provider": provider,
                "calls": len(calls),
                "failures": failures,
                "success_rate": (len(calls) - failures) / len(calls),
                "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                "completion_tokens": sum(c["completion_tokens"] for c in calls),
                "est_cost_usd": sum(c["est_cost_usd"] for c in calls),
                "p50_ms": percentile(latencies, 50),
                "p95_ms": percentile(latencies, 95),
            }
        )
    return stats


def daily_costs(db_path: Path | str, *, since: str | None = None) -> list[dict]:
    """Calls, tokens and cost per day, oldest first."""
    rows = _query(
        db_path,
        """
        SELECT substr(created_at, 1, 10)  AS date,
               COUNT(*)                   AS calls,
               SUM(prompt_tokens + completion_tokens) AS tokens,
               SUM(est_cost_usd)          AS est_cost_usd
        FROM llm_calls
        """,
        since=since,
        suffix=" GROUP BY date ORDER BY date",
    )
    return [dict(row) for row in rows]


def by_purpose(db_path: Path | str, *, since: str | None = None) -> list[dict]:
    """Split by what the call was for.

    The radar runs unattended and the Studio runs while someone waits. One
    number covering both hides which of them is doing the spending.
    """
    rows = _query(
        db_path,
        """
        SELECT purpose,
               COUNT(*)              AS calls,
               SUM(prompt_tokens + completion_tokens) AS tokens,
               SUM(est_cost_usd)     AS est_cost_usd
        FROM llm_calls
        """,
        since=since,
        suffix=" GROUP BY purpose ORDER BY calls DESC, purpose",
    )
    return [dict(row) for row in rows]


def totals(stats: list[dict]) -> dict:
    """Roll the per-provider rows into one line."""
    calls = sum(row["calls"] for row in stats)
    failures = sum(row["failures"] for row in stats)
    return {
        "calls": calls,
        "failures": failures,
        "success_rate": (calls - failures) / calls if calls else 0.0,
        "prompt_tokens": sum(row["prompt_tokens"] for row in stats),
        "completion_tokens": sum(row["completion_tokens"] for row in stats),
        "est_cost_usd": sum(row["est_cost_usd"] for row in stats),
    }


def _query(
    db_path: Path | str, query: str, *, since: str | None, suffix: str = ""
) -> list[sqlite3.Row]:
    """Run a read against the call log, or return nothing if there is none yet.

    A missing database reads as empty and stays missing: `CallLog()` would
    create the table, and an empty one sitting next to a tool that has never
    run says nothing about why the page is blank.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    parameters: list[object] = []
    if since:
        query += " WHERE created_at >= ?"
        parameters.append(since)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(query + suffix, parameters).fetchall()
