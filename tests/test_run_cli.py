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


def test_the_run_reports_which_provider_answered(monkeypatch, tmp_path, capsys):
    """In CI the database goes to a cache nobody reads. A chain that quietly
    degraded to its second tier has to be visible in the log itself."""
    from llm.calls_log import CallLog
    from radar.agent import RunResult
    from radar.schema import Topic

    db = tmp_path / "radar.db"
    topic = Topic(
        title="T",
        summary="s",
        sources=["https://example.com/a"],
        why_now="w",
        angle="practical",
        suggested_outline=["One"],
    )

    def fake_run(**_kwargs):
        CallLog(db).record(
            provider="nim",
            model="meta/llama-3.3-70b-instruct",
            purpose="radar:step",
            ok=True,
            error_type=None,
            latency_ms=900,
            prompt_tokens=1200,
            completion_tokens=300,
        )
        return RunResult([topic], 1, "final_answer", [])

    monkeypatch.setattr("radar.run.run_agent", fake_run)

    main(["--db", str(db)])

    reported = capsys.readouterr().err
    assert "nim" in reported
    assert "1,200 in" in reported


def test_the_provider_summary_is_printed_even_when_the_run_fails(monkeypatch, tmp_path, capsys):
    """A failed run is precisely when you want to know who was asked."""
    from llm.calls_log import CallLog
    from radar.agent import AgentFailed

    db = tmp_path / "radar.db"

    def boom(**_kwargs):
        CallLog(db).record(
            provider="gemini",
            model="gemini-2.5-flash",
            purpose="radar:step",
            ok=False,
            error_type="FatalError",
            latency_ms=10,
            prompt_tokens=100,
            completion_tokens=0,
        )
        raise AgentFailed("no valid answer")

    monkeypatch.setattr("radar.run.run_agent", boom)

    main(["--db", str(db)])

    reported = capsys.readouterr().err
    assert "gemini" in reported
    assert "1 failed" in reported


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
