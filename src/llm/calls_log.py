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
