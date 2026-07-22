"""The Costs tab: what the fallback chain has been doing, and what it cost.

Read-only over the same `radar.db`. The headline number is cheap to produce and
not very interesting — two of the three tiers are free. The reason this tab
exists is the column next to it: a chain that has quietly fallen through to its
second tier keeps every run green, and only a per-tier success rate shows it.
"""

from pathlib import Path

import streamlit as st

from studio.costs_data import by_purpose, daily_costs, provider_stats, totals
from studio.radar_data import PRESETS, since_for


def render(db_path: Path | str) -> None:
    st.subheader("What the chain has spent")

    preset = st.selectbox("Window", list(PRESETS), index=len(PRESETS) - 1, key="costs-window")
    since = since_for(preset)

    stats = provider_stats(db_path, since=since)
    if not stats:
        st.info("No calls recorded in this window.")
        return

    _headline(totals(stats))
    _per_provider(stats)
    _breakdowns(db_path, since)

    st.caption(
        "Cost is an estimate from a hand-maintained price table, and only "
        "Gemini has prices in it — NIM's free tier and local Ollama count as "
        "zero. Treat it as an order of magnitude, not a bill."
    )


def _headline(summary: dict) -> None:
    calls, spend, tokens = st.columns(3)
    calls.metric("Calls", f"{summary['calls']:,}")
    spend.metric("Estimated cost", f"${summary['est_cost_usd']:.4f}")
    tokens.metric(
        "Tokens",
        f"{summary['prompt_tokens'] + summary['completion_tokens']:,}",
        help="Prompt plus completion, across every attempt including failures.",
    )

    if summary["failures"]:
        st.warning(
            f"{summary['failures']} of {summary['calls']} calls failed "
            f"({summary['success_rate']:.0%} succeeded). Check which tier below: "
            "a chain that keeps working while its first tier is broken looks "
            "healthy from the outside."
        )


def _per_provider(stats: list[dict]) -> None:
    st.markdown("### By provider")
    st.dataframe(
        [
            {
                "Provider": row["provider"],
                "Calls": row["calls"],
                "Failed": row["failures"],
                "Success": f"{row['success_rate']:.0%}",
                "Tokens in": row["prompt_tokens"],
                "Tokens out": row["completion_tokens"],
                "p50 latency": _ms(row["p50_ms"]),
                "p95 latency": _ms(row["p95_ms"]),
                "Est. cost": f"${row['est_cost_usd']:.4f}",
            }
            for row in stats
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption("Latency percentiles cover successful calls only — a failed call records no time.")


def _breakdowns(db_path: Path | str, since: str | None) -> None:
    daily, purposes = st.columns(2)

    with daily:
        st.markdown("### Per day")
        rows = daily_costs(db_path, since=since)
        st.dataframe(
            [
                {
                    "Date": row["date"],
                    "Calls": row["calls"],
                    "Tokens": row["tokens"],
                    "Est. cost": f"${row['est_cost_usd']:.4f}",
                }
                for row in rows
            ],
            hide_index=True,
            width="stretch",
        )

    with purposes:
        st.markdown("### Per purpose")
        st.dataframe(
            [
                {
                    "Purpose": row["purpose"] or "(unlabelled)",
                    "Calls": row["calls"],
                    "Tokens": row["tokens"],
                    "Est. cost": f"${row['est_cost_usd']:.4f}",
                }
                for row in by_purpose(db_path, since=since)
            ],
            hide_index=True,
            width="stretch",
        )


def _ms(value: int) -> str:
    return f"{value / 1000:.1f}s" if value >= 1000 else f"{value} ms"
