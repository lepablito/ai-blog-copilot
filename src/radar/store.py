"""History for the radar: runs and the topics they produced.

Shares `radar.db` with `llm.calls_log` — both create their own tables with
`IF NOT EXISTS` and neither owns the file, so opening either one first is safe.

Failed runs are recorded too. A row saying "eight steps, no valid answer" is
the most useful thing to have when the daily job starts misbehaving; deleting
it would leave a gap exactly where the evidence should be.
"""

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .schema import Topic

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    goal            TEXT    NOT NULL,
    hours           INTEGER NOT NULL,
    status          TEXT    NOT NULL,
    steps_used      INTEGER,
    stopped_because TEXT
);

CREATE TABLE IF NOT EXISTS topics (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            INTEGER NOT NULL REFERENCES runs(id),
    date              TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    title             TEXT    NOT NULL,
    summary           TEXT    NOT NULL,
    sources           TEXT    NOT NULL,
    why_now           TEXT    NOT NULL,
    angle             TEXT    NOT NULL,
    estimated_effort  TEXT    NOT NULL,
    suggested_outline TEXT    NOT NULL,
    citations         TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_topics_date ON topics(date);
CREATE INDEX IF NOT EXISTS idx_topics_angle ON topics(angle);
"""


class Store:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def start_run(self, *, goal: str, hours: int) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO runs (started_at, goal, hours, status) VALUES (?, ?, ?, 'running')",
                (_now(), goal, hours),
            )
            return cursor.lastrowid

    def finish_run(
        self, run_id: int, *, status: str, steps_used: int, stopped_because: str
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE runs SET finished_at = ?, status = ?, steps_used = ?, "
                "stopped_because = ? WHERE id = ?",
                (_now(), status, steps_used, stopped_because, run_id),
            )

    def save_topics(self, run_id: int, topics: Iterable[Topic]) -> None:
        now = _now()
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO topics (
                    run_id, date, created_at, title, summary, sources, why_now,
                    angle, estimated_effort, suggested_outline, citations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        now[:10],
                        now,
                        topic.title,
                        topic.summary,
                        json.dumps(topic.sources, ensure_ascii=False),
                        topic.why_now,
                        topic.angle,
                        topic.estimated_effort,
                        json.dumps(topic.suggested_outline, ensure_ascii=False),
                        json.dumps(topic.citations, ensure_ascii=False),
                    )
                    for topic in topics
                ],
            )

    def recent_topics(
        self, *, angle: str | None = None, since: str | None = None, limit: int = 200
    ) -> list[Topic]:
        query = "SELECT * FROM topics"
        conditions, parameters = [], []
        if angle:
            conditions.append("angle = ?")
            parameters.append(angle)
        if since:
            conditions.append("date >= ?")
            parameters.append(since)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT ?"
        parameters.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [_to_topic(row) for row in conn.execute(query, parameters)]

    def export_records(self, *, limit: int = 1000) -> list[dict]:
        """Topics as plain dicts, carrying the date the run happened."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM topics ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{"date": row["date"], **_to_topic(row).as_dict()} for row in rows]

    def export_json(self, path: Path | str, *, generated_at: str | None = None) -> Path:
        """Write the history as JSON for the daily workflow to commit.

        Merges with whatever is already in the file rather than replacing it.
        The workflow keeps radar.db in a GitHub Actions cache, and caches are
        evicted after a week of disuse — a plain overwrite would then quietly
        commit the deletion of every earlier topic. The committed file is the
        durable copy; the database is just the working one.

        Deterministic on purpose: sorted keys and a stable order, so a day with
        nothing new produces no diff and therefore no commit.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = _existing_topics(path)
        merged: dict[tuple[str, str], dict] = {}
        for record in [*existing, *self.export_records()]:
            merged[(record.get("date", ""), record.get("title", ""))] = record

        topics = sorted(
            merged.values(),
            key=lambda r: (r.get("date", ""), r.get("title", "")),
            reverse=True,
        )

        # Nothing new: leave the file exactly as it is. Rewriting it just to
        # move `generated_at` would make every quiet day look like a change,
        # and the workflow would commit one. Creating the file for the first
        # time is of course a change, however empty it is.
        if path.exists() and topics == existing:
            return path

        payload = {"generated_at": generated_at or _now(), "topics": topics}
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def _existing_topics(path: Path) -> list[dict]:
    """Read the topics already committed to `path`.

    A corrupt file raises rather than being ignored. Treating it as empty would
    turn one bad byte into a commit that deletes the archive — exactly the
    failure this merge exists to prevent.
    """
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"existing export {str(path)!r} is not valid JSON: {exc}") from exc

    topics = payload.get("topics") if isinstance(payload, dict) else None
    if not isinstance(topics, list):
        raise ValueError(f"existing export {str(path)!r} has no 'topics' array")

    return [t for t in topics if isinstance(t, dict)]


def _to_topic(row: sqlite3.Row) -> Topic:
    return Topic(
        title=row["title"],
        summary=row["summary"],
        sources=json.loads(row["sources"]),
        why_now=row["why_now"],
        angle=row["angle"],
        suggested_outline=json.loads(row["suggested_outline"]),
        estimated_effort=row["estimated_effort"],
        citations=json.loads(row["citations"]),
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
