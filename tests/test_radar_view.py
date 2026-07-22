"""The Radar tab's logic, tested without Streamlit.

Everything the tab decides — which window, which angle, how rows group under a
heading — lives in `studio.radar_data` as plain functions. Streamlit only draws
the result. A test that needed a running server to check a date filter would
not be worth writing.
"""

from datetime import date

import pytest

from radar.schema import Topic
from radar.store import Store
from studio.radar_data import PRESETS, all_links, group_by_date, load_topics, since_for


def _topic(title: str, *, angle: str = "practical") -> Topic:
    return Topic(
        title=title,
        summary="summary",
        sources=["https://example.com/a"],
        why_now="published today",
        angle=angle,
        suggested_outline=["intro"],
    )


@pytest.fixture
def db(tmp_path):
    return tmp_path / "radar.db"


def test_presets_map_to_a_since_date():
    today = date(2026, 7, 22)

    assert since_for("Today", today=today) == "2026-07-22"
    assert since_for("Last 7 days", today=today) == "2026-07-16"


def test_all_time_has_no_lower_bound():
    assert since_for("All time", today=date(2026, 7, 22)) is None


def test_every_preset_offered_to_the_user_resolves():
    """The selectbox is built from PRESETS, so an entry with no rule behind it
    would only fail once someone clicked it."""
    for preset in PRESETS:
        since_for(preset, today=date(2026, 7, 22))


def test_topics_carry_the_date_of_the_run_that_found_them(db):
    store = Store(db)
    run = store.start_run(goal="g", hours=24)
    store.save_topics(run, [_topic("Speculative decoding")])

    [record] = load_topics(db)

    assert record["title"] == "Speculative decoding"
    assert record["date"] == date.today().isoformat()


def test_angle_filter_narrows_the_list(db):
    store = Store(db)
    run = store.start_run(goal="g", hours=24)
    store.save_topics(
        run,
        [_topic("Attention maths", angle="theoretical"), _topic("Shipping a RAG eval")],
    )

    titles = [r["title"] for r in load_topics(db, angle="theoretical")]

    assert titles == ["Attention maths"]


def test_a_window_that_predates_every_topic_returns_nothing(db):
    store = Store(db)
    run = store.start_run(goal="g", hours=24)
    store.save_topics(run, [_topic("Speculative decoding")])

    assert load_topics(db, since="2000-01-01") != []
    assert load_topics(db, since="2099-01-01") == []


def test_a_missing_database_reads_as_empty_and_is_not_created(db):
    """Opening the app before the first radar run must not leave an empty
    radar.db behind — the next real run would find tables but no history and
    nothing would say why."""
    assert load_topics(db) == []
    assert not db.exists()


def test_links_merge_sources_and_citations_without_repeating_one():
    """The agent routinely cites a page it also listed as a source. Printing it
    twice under one heading just looks like a bug."""
    topic = {
        "sources": ["https://a.example/x", "https://b.example/y"],
        "citations": ["https://b.example/y", "https://c.example/z"],
    }

    assert all_links(topic) == [
        "https://a.example/x",
        "https://b.example/y",
        "https://c.example/z",
    ]


def test_links_survive_a_topic_with_no_citations():
    assert all_links({"sources": ["https://a.example/x"]}) == ["https://a.example/x"]


def test_grouping_keeps_dates_newest_first_and_rows_in_order():
    records = [
        {"date": "2026-07-20", "title": "older"},
        {"date": "2026-07-22", "title": "first of today"},
        {"date": "2026-07-22", "title": "second of today"},
    ]

    assert group_by_date(records) == [
        (
            "2026-07-22",
            [
                {"date": "2026-07-22", "title": "first of today"},
                {"date": "2026-07-22", "title": "second of today"},
            ],
        ),
        ("2026-07-20", [{"date": "2026-07-20", "title": "older"}]),
    ]
