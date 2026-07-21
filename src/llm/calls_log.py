"""Every model call — successes, retries and dead ends — lands in SQLite.

This is the raw material for the cost dashboard, but its real value shows up
earlier than that: with a three-tier fallback chain it is entirely possible for
the system to "work" while the primary provider has been failing silently for
days. A row per attempt makes that visible.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,
    provider          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    purpose           TEXT    NOT NULL DEFAULT '',
    ok                INTEGER NOT NULL,
    error_type        TEXT,
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    est_cost_usd      REAL    NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created_at ON llm_calls(created_at);
"""

# USD per million tokens, (input, output). Hand-maintained and therefore an
# ESTIMATE — treat the cost column as an order-of-magnitude signal, not a bill.
# Anything absent from this table (NIM free tier, local Ollama) costs nothing.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = PRICING_PER_MTOK.get(model)
    if price is None:
        return 0.0
    input_price, output_price = price
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


class CallLog:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def record(
        self,
        *,
        provider: str,
        model: str,
        purpose: str,
        ok: bool,
        error_type: str | None,
        latency_ms: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO llm_calls (
                    created_at, provider, model, purpose, ok, error_type,
                    latency_ms, prompt_tokens, completion_tokens, est_cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    provider,
                    model,
                    purpose,
                    int(ok),
                    error_type,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    estimate_cost_usd(model, prompt_tokens, completion_tokens),
                ),
            )

    def summary_by_provider(self, *, since: str | None = None) -> list[dict]:
        """Aggregate calls per provider, newest tier first.

        `since` scopes it to a single pass. Without it the whole history is
        summarised, which is what the cost dashboard wants and the workflow
        does not.
        """
        query = """
            SELECT provider,
                   COUNT(*)                        AS calls,
                   SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failures,
                   SUM(prompt_tokens)              AS prompt_tokens,
                   SUM(completion_tokens)          AS completion_tokens,
                   SUM(est_cost_usd)               AS est_cost_usd,
                   MAX(latency_ms)                 AS max_latency_ms
            FROM llm_calls
        """
        parameters: list[object] = []
        if since:
            query += " WHERE created_at >= ?"
            parameters.append(since)
        query += " GROUP BY provider ORDER BY provider"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(query, parameters)]


def format_summary(rows: list[dict]) -> str:
    """One line per provider, for the CI log.

    In a workflow run the database lives in an Actions cache and nobody ever
    looks at it, so a chain that silently degraded to its second tier would go
    unnoticed for weeks. This puts it in the log where a green run still shows
    what actually answered.
    """
    if not rows:
        return "LLM calls: no calls recorded."

    lines = ["LLM calls this run:"]
    for row in rows:
        call_word = "call" if row["calls"] == 1 else "calls"
        detail = (
            f"  {row['provider']:<8} {row['calls']} {call_word}"
            f", {row['failures']} failed"
            f", {row['prompt_tokens']:,} in / {row['completion_tokens']:,} out"
        )
        if row["est_cost_usd"]:
            detail += f", ~${row['est_cost_usd']:.4f}"
        lines.append(detail)
    return "\n".join(lines)
