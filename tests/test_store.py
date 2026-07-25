import json
import sqlite3

import pytest

from radar.schema import InvalidTopics, Topic
from radar.store import Store


def topic(title="A topic", angle="practical"):
    return Topic(
        title=title,
        summary="A summary.",
        sources=["https://example.com/a", "https://example.com/b"],
        why_now="It is new.",
        angle=angle,
        suggested_outline=["One", "Two"],
        estimated_effort="medium",
        citations=["https://example.com/a"],
    )


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "radar.db")


def rows(store, table):
    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY id")]


# --- runs ---


def test_a_run_is_recorded_when_it_starts(store):
    run_id = store.start_run(goal="find things", hours=48)

    (row,) = rows(store, "runs")
    assert row["id"] == run_id
    assert row["goal"] == "find things"
    assert row["hours"] == 48
    assert row["status"] == "running"
    assert row["started_at"]
    assert row["finished_at"] is None


def test_finishing_a_run_closes_it_out(store):
    run_id = store.start_run(goal="g", hours=48)

    store.finish_run(run_id, status="ok", steps_used=3, stopped_because="final_answer")

    (row,) = rows(store, "runs")
    assert row["status"] == "ok"
    assert row["steps_used"] == 3
    assert row["stopped_because"] == "final_answer"
    assert row["finished_at"]


def test_a_failed_run_is_still_recorded(store):
    """A run that produced nothing is the most interesting one to look back at."""
    run_id = store.start_run(goal="g", hours=48)

    store.finish_run(run_id, status="failed", steps_used=8, stopped_because="step_limit")

    assert rows(store, "runs")[0]["status"] == "failed"


# --- topics ---


def test_topics_round_trip_with_their_lists_intact(store):
    run_id = store.start_run(goal="g", hours=48)

    store.save_topics(run_id, [topic()])

    (row,) = rows(store, "topics")
    assert row["title"] == "A topic"
    assert json.loads(row["sources"]) == ["https://example.com/a", "https://example.com/b"]
    assert json.loads(row["suggested_outline"]) == ["One", "Two"]
    assert row["run_id"] == run_id
    assert row["date"], "the UI groups by day"


def test_reading_topics_back_reconstructs_them(store):
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic()])

    (read,) = store.recent_topics()

    assert read.title == "A topic"
    assert read.sources == ["https://example.com/a", "https://example.com/b"]


def test_topics_can_be_filtered_by_angle(store):
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic("practical one"), topic("theory one", angle="theoretical")])

    found = store.recent_topics(angle="theoretical")

    assert [t.title for t in found] == ["theory one"]


def test_topics_come_back_newest_first(store):
    first = store.start_run(goal="g", hours=48)
    store.save_topics(first, [topic("older")])
    second = store.start_run(goal="g", hours=48)
    store.save_topics(second, [topic("newer")])

    assert [t.title for t in store.recent_topics()] == ["newer", "older"]


def test_history_survives_reopening_the_database(tmp_path):
    first = Store(tmp_path / "radar.db")
    run_id = first.start_run(goal="g", hours=48)
    first.save_topics(run_id, [topic()])

    second = Store(tmp_path / "radar.db")

    assert len(second.recent_topics()) == 1


def test_the_store_coexists_with_the_call_log(tmp_path):
    """Both write to radar.db — neither may clobber the other's schema."""
    from llm.calls_log import CallLog

    store = Store(tmp_path / "radar.db")
    log = CallLog(tmp_path / "radar.db")
    log.record(
        provider="ollama",
        model="m",
        purpose="radar",
        ok=True,
        error_type=None,
        latency_ms=1,
        prompt_tokens=1,
        completion_tokens=1,
    )
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic()])

    assert len(rows(store, "llm_calls")) == 1
    assert len(rows(store, "topics")) == 1


# --- the JSON export the workflow commits ---


def test_export_writes_readable_json(store, tmp_path):
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic()])
    target = tmp_path / "data" / "topics.json"

    store.export_json(target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["topics"][0]["title"] == "A topic"
    assert payload["generated_at"]


def test_export_of_unchanged_data_is_byte_identical(store, tmp_path):
    """The daily workflow commits only when the file changes. Non-deterministic
    ordering or key shuffling would produce a commit every single day."""
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic("a"), topic("b"), topic("c")])
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"

    store.export_json(first, generated_at="2026-07-22T06:00:00+00:00")
    store.export_json(second, generated_at="2026-07-22T06:00:00+00:00")

    assert first.read_bytes() == second.read_bytes()


def test_export_keeps_history_the_database_no_longer_has(store, tmp_path):
    """The daily workflow keeps radar.db in a GitHub Actions cache, and caches
    are evicted after a week. Exporting only what the database currently holds
    would commit the loss of every earlier topic."""
    target = tmp_path / "topics.json"
    target.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "topics": [
                    {
                        "date": "2026-01-01",
                        "title": "From a run whose cache expired",
                        "summary": "s",
                        "sources": ["https://example.com/old"],
                        "why_now": "w",
                        "angle": "practical",
                        "estimated_effort": "medium",
                        "suggested_outline": ["One"],
                        "citations": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic("From today")])

    store.export_json(target)

    titles = [t["title"] for t in json.loads(target.read_text(encoding="utf-8"))["topics"]]
    assert "From a run whose cache expired" in titles
    assert "From today" in titles


def test_a_topic_present_in_both_is_not_duplicated(store, tmp_path):
    target = tmp_path / "topics.json"
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic("Only once")])
    store.export_json(target)

    store.export_json(target)

    titles = [t["title"] for t in json.loads(target.read_text(encoding="utf-8"))["topics"]]
    assert titles.count("Only once") == 1


def test_exported_topics_carry_their_date(store, tmp_path):
    target = tmp_path / "topics.json"
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic()])

    store.export_json(target)

    assert json.loads(target.read_text(encoding="utf-8"))["topics"][0]["date"]


def test_an_unreadable_existing_export_stops_the_write(store, tmp_path):
    """Better a failed workflow step than a commit that silently drops history."""
    target = tmp_path / "topics.json"
    target.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(ValueError):
        store.export_json(target)

    assert target.read_text(encoding="utf-8") == "{ this is not json"


def test_re_exporting_unchanged_topics_leaves_the_file_untouched(store, tmp_path):
    """`generated_at` moves on every run. If it were written unconditionally the
    file would differ every day, the workflow's "commit only if changed" check
    would never filter anything, and the repo would collect an empty commit a
    day forever."""
    target = tmp_path / "topics.json"
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic()])
    store.export_json(target, generated_at="2026-07-22T06:00:00+00:00")
    before = target.read_bytes()

    store.export_json(target, generated_at="2026-07-23T06:00:00+00:00")

    assert target.read_bytes() == before


def test_a_new_topic_does_update_the_timestamp(store, tmp_path):
    target = tmp_path / "topics.json"
    first_run = store.start_run(goal="g", hours=48)
    store.save_topics(first_run, [topic("day one")])
    store.export_json(target, generated_at="2026-07-22T06:00:00+00:00")

    second_run = store.start_run(goal="g", hours=48)
    store.save_topics(second_run, [topic("day two")])
    store.export_json(target, generated_at="2026-07-23T06:00:00+00:00")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["generated_at"] == "2026-07-23T06:00:00+00:00"
    assert len(payload["topics"]) == 2


def test_export_creates_missing_directories(store, tmp_path):
    store.export_json(tmp_path / "deep" / "nested" / "topics.json")

    assert (tmp_path / "deep" / "nested" / "topics.json").exists()


def test_export_ends_with_a_newline(store, tmp_path):
    """Otherwise every diff shows a 'No newline at end of file' marker."""
    target = tmp_path / "topics.json"

    store.export_json(target)

    assert target.read_text(encoding="utf-8").endswith("\n")


# --- reading that export back in ---


def written_export(path, *topics):
    """A topics.json shaped exactly like the one the workflow commits."""
    path.write_text(
        json.dumps({"generated_at": "2026-07-24T23:24:49+00:00", "topics": list(topics)}),
        encoding="utf-8",
    )
    return path


def record(title="A topic", *, date="2026-07-24", angle="practical"):
    return {"date": date, **topic(title, angle).as_dict()}


def test_import_inserts_the_topics_it_finds(store, tmp_path):
    target = written_export(tmp_path / "topics.json", record("Speculative decoding"))

    assert store.import_json(target) == 1
    assert [r["title"] for r in store.recent_records()] == ["Speculative decoding"]


def test_imported_topics_keep_their_own_date(store, tmp_path):
    """`save_topics` stamps the current day, which is right for a live pass and
    wrong here: the date belongs to the run that found the topic. Restamping it
    would file yesterday's topics under today and, worse, change the (date,
    title) key so the next import would insert them all over again."""
    target = written_export(tmp_path / "topics.json", record(date="2026-07-24"))

    store.import_json(target)

    (row,) = store.recent_records()
    assert row["date"] == "2026-07-24"


def test_importing_the_same_file_twice_inserts_nothing_the_second_time(store, tmp_path):
    target = written_export(tmp_path / "topics.json", record("Only once"))
    store.import_json(target)

    assert store.import_json(target) == 0
    assert len(store.recent_records()) == 1


def test_an_import_with_nothing_new_opens_no_run(store, tmp_path):
    """A run row per click would fill the runs table with rows that recorded
    no work at all."""
    target = written_export(tmp_path / "topics.json", record())
    store.import_json(target)
    before = len(rows(store, "runs"))

    store.import_json(target)

    assert len(rows(store, "runs")) == before


def test_a_missing_file_imports_nothing_and_opens_no_run(store, tmp_path):
    assert store.import_json(tmp_path / "absent.json") == 0
    assert rows(store, "runs") == []


def test_an_unreadable_file_stops_the_import(store, tmp_path):
    target = tmp_path / "topics.json"
    target.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(ValueError):
        store.import_json(target)


def test_an_import_is_distinguishable_from_a_real_pass(store, tmp_path):
    target = written_export(tmp_path / "topics.json", record())

    store.import_json(target)

    (run,) = rows(store, "runs")
    assert run["status"] == "imported"


def test_a_topic_with_a_non_http_source_never_reaches_the_database(store, tmp_path):
    """topics.json is a file on disk that anyone can edit. The schema is the
    same gate the agent's output goes through, and it applies here too."""
    poisoned = record()
    poisoned["sources"] = ["javascript:alert(1)"]
    target = written_export(tmp_path / "topics.json", poisoned)

    with pytest.raises(InvalidTopics):
        store.import_json(target)

    assert store.recent_records() == []


def test_an_import_of_several_days_still_reads_back_newest_first(store, tmp_path):
    """`recent_records` orders by id, taking a higher id to mean newer. An
    export is sorted newest-first, so inserting it in file order would give the
    oldest day the highest id — and because the query carries a LIMIT, a large
    import would then truncate away the newest topics rather than the oldest."""
    target = written_export(
        tmp_path / "topics.json",
        record("Newest", date="2026-07-24"),
        record("Middle", date="2026-07-23"),
        record("Oldest", date="2026-07-22"),
    )

    store.import_json(target)

    assert [r["date"] for r in store.recent_records()] == [
        "2026-07-24",
        "2026-07-23",
        "2026-07-22",
    ]


def test_export_then_import_into_a_fresh_database_round_trips(store, tmp_path):
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic("a"), topic("b", angle="theoretical")])
    target = tmp_path / "topics.json"
    store.export_json(target)

    fresh = Store(tmp_path / "fresh.db")
    assert fresh.import_json(target) == 2

    assert fresh.recent_records() == store.recent_records()


# --- dismissing a topic ---


def dismissable(store, title="A topic", *, date="2026-07-24", angle="practical"):
    """A topic in the database under a date we control, so it can be named."""
    run_id = store.start_run(goal="g", hours=48)
    store.save_topics(run_id, [topic(title, angle)])
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE topics SET date = ? WHERE title = ?", (date, title))
    return date, title


def test_a_dismissed_topic_is_no_longer_listed(store):
    date, title = dismissable(store, "Not worth writing")

    assert store.dismiss(date=date, title=title) == 1

    assert store.recent_records() == []


def test_dismissing_one_topic_leaves_its_neighbours_alone(store):
    date, title = dismissable(store, "Goes")
    dismissable(store, "Stays", date=date)

    store.dismiss(date=date, title=title)

    assert [r["title"] for r in store.recent_records()] == ["Stays"]


def test_a_dismissed_topic_stays_gone_under_an_angle_filter(store):
    """The angle filter builds a different WHERE clause, so it is a different
    query and needs its own proof."""
    date, title = dismissable(store, "Attention maths", angle="theoretical")

    store.dismiss(date=date, title=title)

    assert store.recent_records(angle="theoretical") == []


def test_dismissing_a_topic_that_is_not_there_changes_nothing(store):
    """Two tabs open on the same database is enough to reach this."""
    assert store.dismiss(date="2026-07-24", title="Never existed") == 0


def test_a_dismissed_topic_does_not_come_back_when_the_export_is_imported(store, tmp_path):
    """The whole point of a tombstone over a DELETE. `import_json` skips it
    because its duplicate check reads every row rather than going through
    `_select` — if that query is ever "tidied up" to use `_select`, every
    dismissed topic silently returns on the next fetch. This test is what
    turns that into a red build."""
    date, title = dismissable(store, "Dismissed but archived")
    target = written_export(tmp_path / "topics.json", record(title, date=date))
    store.dismiss(date=date, title=title)

    assert store.import_json(target) == 0

    assert store.recent_records() == []


def test_exporting_does_not_drop_a_dismissed_topic_from_the_archive(store, tmp_path):
    """A dismissal is local to this Studio. The committed file is the durable
    copy and must not lose a topic because someone hid it from their own view."""
    date, title = dismissable(store, "Archived anyway")
    target = tmp_path / "topics.json"
    store.export_json(target)

    store.dismiss(date=date, title=title)
    store.export_json(target)

    titles = [t["title"] for t in json.loads(target.read_text(encoding="utf-8"))["topics"]]
    assert title in titles


# --- opening a database written before dismissals existed ---


OLD_TOPICS_TABLE = """
CREATE TABLE topics (
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
"""


def old_database(path):
    """A radar.db from before the dismissed_at column existed.

    Built by hand rather than by checking out an old revision, so it keeps
    describing the shape it is meant to describe when SCHEMA moves on.
    """
    with sqlite3.connect(path) as conn:
        conn.executescript(OLD_TOPICS_TABLE)
        conn.execute(
            """
            INSERT INTO topics (
                run_id, date, created_at, title, summary, sources, why_now,
                angle, estimated_effort, suggested_outline, citations
            ) VALUES (1, '2026-07-24', '2026-07-24T00:00:00+00:00', 'Older topic',
                      's', '[]', 'w', 'practical', 'medium', '[]', '[]')
            """
        )
    return path


def test_an_existing_database_gains_the_column_when_opened(tmp_path):
    """radar.db already exists on the machines that matter, and
    CREATE TABLE IF NOT EXISTS will not add a column to a table already there."""
    path = old_database(tmp_path / "radar.db")

    store = Store(path)

    assert [r["title"] for r in store.recent_records()] == ["Older topic"]


def test_a_topic_from_an_older_database_can_be_dismissed(tmp_path):
    store = Store(old_database(tmp_path / "radar.db"))

    assert store.dismiss(date="2026-07-24", title="Older topic") == 1

    assert store.recent_records() == []


def test_opening_a_migrated_database_twice_is_harmless(tmp_path):
    """The migration runs on every open. A second ALTER TABLE would raise."""
    path = old_database(tmp_path / "radar.db")

    Store(path)
    store = Store(path)

    assert len(store.recent_records()) == 1
