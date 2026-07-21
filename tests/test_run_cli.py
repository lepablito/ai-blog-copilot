import json

from radar.run import build_registry, main

GOAL_FRAGMENT = "worth a technical post"


def test_the_registry_exposes_exactly_the_three_sources():
    registry = build_registry()

    assert set(registry.names) == {"fetch_hackernews", "fetch_rss", "fetch_article_text"}


def test_every_registered_tool_is_described_for_the_prompt():
    described = build_registry().describe()

    for name in ("fetch_hackernews", "fetch_rss", "fetch_article_text"):
        assert name in described
    assert "hours" in described, "the model needs to know it can set the window"


def test_main_prints_topics_as_json(monkeypatch, capsys):
    from radar.agent import RunResult
    from radar.schema import Topic

    topic = Topic(
        title="A topic",
        summary="A summary.",
        sources=["https://example.com/a"],
        why_now="It is new.",
        angle="practical",
        suggested_outline=["One"],
    )

    monkeypatch.setattr(
        "radar.run.run_agent",
        lambda **_kwargs: RunResult([topic], 2, "final_answer", []),
    )

    exit_code = main(["--hours", "24"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["topics"][0]["title"] == "A topic"
    assert payload["steps_used"] == 2
    assert payload["stopped_because"] == "final_answer"


def test_a_successful_run_is_persisted(monkeypatch, tmp_path, capsys):
    from radar.agent import RunResult
    from radar.schema import Topic
    from radar.store import Store

    topic = Topic(
        title="Persisted",
        summary="s",
        sources=["https://example.com/a"],
        why_now="w",
        angle="theoretical",
        suggested_outline=["One"],
    )
    monkeypatch.setattr(
        "radar.run.run_agent", lambda **_kw: RunResult([topic], 2, "final_answer", [])
    )
    db = tmp_path / "radar.db"

    main(["--db", str(db)])

    stored = Store(db).recent_topics()
    assert [t.title for t in stored] == ["Persisted"]


def test_a_failed_run_is_persisted_too(monkeypatch, tmp_path, capsys):
    """The runs worth inspecting later are exactly the ones that produced
    nothing — losing them would leave a gap where the evidence should be."""
    import sqlite3

    from radar.agent import AgentFailed

    def boom(**_kwargs):
        raise AgentFailed("no valid answer")

    monkeypatch.setattr("radar.run.run_agent", boom)
    db = tmp_path / "radar.db"

    assert main(["--db", str(db)]) == 1

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        (row,) = [dict(r) for r in conn.execute("SELECT * FROM runs")]
    assert row["status"] == "failed"
    assert row["finished_at"]


def test_export_is_written_when_asked(monkeypatch, tmp_path):
    from radar.agent import RunResult
    from radar.schema import Topic

    topic = Topic(
        title="Exported",
        summary="s",
        sources=["https://example.com/a"],
        why_now="w",
        angle="practical",
        suggested_outline=["One"],
    )
    monkeypatch.setattr(
        "radar.run.run_agent", lambda **_kw: RunResult([topic], 1, "final_answer", [])
    )
    export = tmp_path / "data" / "topics.json"

    main(["--db", str(tmp_path / "radar.db"), "--export", str(export)])

    assert json.loads(export.read_text(encoding="utf-8"))["topics"][0]["title"] == "Exported"


def test_main_reports_a_failed_run_without_a_traceback(monkeypatch, capsys):
    from radar.agent import AgentFailed

    def boom(**_kwargs):
        raise AgentFailed("the model never produced a valid answer")

    monkeypatch.setattr("radar.run.run_agent", boom)

    exit_code = main([])

    assert exit_code == 1
    assert "never produced a valid answer" in capsys.readouterr().err


def test_main_reports_a_dead_provider_chain_without_a_traceback(monkeypatch, capsys):
    from llm.client import AllProvidersFailed

    def boom(**_kwargs):
        raise AllProvidersFailed("gemini: 401; ollama: connection refused")

    monkeypatch.setattr("radar.run.run_agent", boom)

    exit_code = main([])

    assert exit_code == 1
    assert "connection refused" in capsys.readouterr().err
