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

from .schema import Topic, parse_topics

# Written into the goal of the synthetic run an import opens, so a row that came
# from a file stays tellable from one the agent actually produced.
IMPORT_GOAL = "imported from an export"

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
    citations         TEXT    NOT NULL,
    -- NULL means live. A dismissed topic keeps its row: deleting it would let
    -- `import_json` insert it again on the next fetch, since it recognises a
    -- topic it already has by (date, title) being present at all.
    dismissed_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_topics_date ON topics(date);
CREATE INDEX IF NOT EXISTS idx_topics_angle ON topics(angle);
"""

# Columns added to `topics` after the first radar.db was written. The statements
# above leave an existing table exactly as they found it, so without this a
# database from before one of these columns existed keeps its old shape and
# every read fails on the column it lacks. Names are ours, not anyone's input.
LATER_COLUMNS = {"dismissed_at": "TEXT"}


class Store:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)
            _add_later_columns(conn)

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

    def dismiss(self, *, date: str, title: str) -> int:
        """Hide a topic from every read. Returns how many rows were marked.

        Identified by `(date, title)` — the same identity `export_json` merges
        on and `import_json` deduplicates on. A third way of naming a topic
        would be a fourth thing to keep in agreement.

        The row survives, which is what makes the dismissal stick: `import_json`
        takes a `(date, title)` already present as a topic it need not insert,
        and looks at every row rather than only the live ones.

        Dismissing something that is not there returns 0 rather than raising.
        Two tabs open on one database is enough to reach that.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE topics SET dismissed_at = ? "
                "WHERE date = ? AND title = ? AND dismissed_at IS NULL",
                (_now(), date, title),
            )
            return cursor.rowcount

    def _select(self, *, angle: str | None, since: str | None, limit: int) -> list[sqlite3.Row]:
        query = "SELECT * FROM topics"
        # Every read in the application comes through here, so one condition
        # hides dismissed topics from the tab, the drafting picker and the
        # export alike.
        conditions, parameters = ["dismissed_at IS NULL"], []
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
            return conn.execute(query, parameters).fetchall()

    def recent_topics(
        self, *, angle: str | None = None, since: str | None = None, limit: int = 200
    ) -> list[Topic]:
        return [_to_topic(row) for row in self._select(angle=angle, since=since, limit=limit)]

    def recent_records(
        self, *, angle: str | None = None, since: str | None = None, limit: int = 200
    ) -> list[dict]:
        """Topics as plain dicts, carrying the date of the run that found them.

        `Topic` deliberately has no date field — it is the agent's output
        contract, and the agent does not decide when it ran. The date belongs
        to the row, so anything that displays history reads it from here.
        """
        rows = self._select(angle=angle, since=since, limit=limit)
        return [{"date": row["date"], **_to_topic(row).as_dict()} for row in rows]

    def export_records(self, *, limit: int = 1000) -> list[dict]:
        return self.recent_records(limit=limit)

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

    def import_json(self, path: Path | str) -> int:
        """Read an export back in, inserting only what this database lacks.

        The inverse of `export_json` and keyed the same way, on `(date, title)`.
        The workflow commits the JSON and leaves radar.db in an Actions cache
        nobody downloads, so this is the only route by which a run on a runner
        reaches a Studio on a laptop.

        Returns the number of topics inserted.
        """
        records = _existing_topics(Path(path))
        if not records:
            # Also the empty-file case, and the reason for checking before
            # parsing: `parse_topics` treats an empty list as a failed answer,
            # which is right for the agent and wrong for an import.
            return 0

        # The same gate the agent's own output goes through. An export is a file
        # on disk that anyone can edit, and a hand-written `javascript:` source
        # should not reach the Studio just because it arrived by another door.
        topics = parse_topics(records)
        dates = [str(record.get("date", "")) for record in records]

        with sqlite3.connect(self.db_path) as conn:
            known = {(row[0], row[1]) for row in conn.execute("SELECT date, title FROM topics")}

        # Every row, not `recent_records` — its limit would let anything older
        # than the last 200 topics back in as a duplicate.
        fresh = [
            (date, topic)
            for date, topic in zip(dates, topics, strict=True)
            if (date, topic.title) not in known
        ]
        if not fresh:
            return 0

        # Oldest first, so that a rising id keeps meaning "newer" — the
        # assumption `_select` encodes by ordering on id alone. An export is
        # sorted newest-first, so inserting it as it comes would give the oldest
        # day the highest id, and the LIMIT on that query would then cut away
        # the newest topics instead of the oldest. Sorted here rather than
        # trusted from the file: a hand-edited export need not be in any order.
        fresh.sort(key=lambda pair: (pair[0], pair[1].title))

        # Opened only once there is something to record. A run row per click
        # would fill the table with rows describing no work at all.
        run_id = self.start_run(goal=IMPORT_GOAL, hours=0)
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
                        # The record's own date, not today's: `save_topics`
                        # stamps the current day because the agent does not
                        # choose when it ran, but an imported topic already
                        # knows. `created_at` stays now — that is when the row
                        # entered *this* database.
                        date,
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
                    for date, topic in fresh
                ],
            )

        self.finish_run(run_id, status="imported", steps_used=0, stopped_because="import")
        return len(fresh)


def _add_later_columns(conn: sqlite3.Connection) -> None:
    """Bring an older `topics` table up to the current shape.

    Runs on every open and does nothing once there is nothing left to add,
    which is what makes it safe to call unconditionally. SQLite has no
    `ADD COLUMN IF NOT EXISTS`, so the check is a read of the table's shape.
    """
    present = {row[1] for row in conn.execute("PRAGMA table_info(topics)")}
    for name, kind in LATER_COLUMNS.items():
        if name not in present:
            conn.execute(f"ALTER TABLE topics ADD COLUMN {name} {kind}")


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
