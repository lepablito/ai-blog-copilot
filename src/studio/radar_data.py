"""Reading side of the Radar tab. No Streamlit imports live here on purpose.

The tab is read-only over `radar.db`: the agent writes it, the UI never does.
"""

from datetime import date, timedelta
from pathlib import Path

from radar.store import Store

# Ordered: this is also the order of the selectbox. `None` means no lower bound.
PRESETS: dict[str, int | None] = {
    "Today": 0,
    "Last 7 days": 6,
    "Last 30 days": 29,
    "All time": None,
}

ANGLES = ("theoretical", "practical")


def since_for(preset: str, *, today: date | None = None) -> str | None:
    """Turn a preset label into the earliest date to show, or None for all."""
    days = PRESETS.get(preset)
    if days is None:
        return None
    return ((today or date.today()) - timedelta(days=days)).isoformat()


def load_topics(
    db_path: Path | str,
    *,
    angle: str | None = None,
    since: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """Topic records from the database, newest first.

    A database that does not exist yet reads as empty and stays that way.
    `Store()` would create the tables, and an empty radar.db sitting next to a
    radar that has never run is a confusing thing to leave behind.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    return Store(db_path).recent_records(angle=angle, since=since, limit=limit)


def all_links(topic: dict) -> list[str]:
    """Sources then citations, first occurrence wins, no repeats."""
    seen: dict[str, None] = {}
    for url in [*topic.get("sources", []), *topic.get("citations", [])]:
        seen.setdefault(url, None)
    return list(seen)


def group_by_date(records: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group records under their date heading, newest day first.

    Order within a day is left exactly as it came out of the query.
    """
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record.get("date", ""), []).append(record)
    return sorted(grouped.items(), key=lambda pair: pair[0], reverse=True)
