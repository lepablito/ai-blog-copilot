import json
import sqlite3

import pytest

from radar.schema import Topic
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
