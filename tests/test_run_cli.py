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
        "radar.run.execute",
        lambda **_kwargs: RunResult([topic], 2, "final_answer", []),
    )

    exit_code = main(["--hours", "24"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["topics"][0]["title"] == "A topic"
    assert payload["steps_used"] == 2
    assert payload["stopped_because"] == "final_answer"


def test_main_reports_a_failed_run_without_a_traceback(monkeypatch, capsys):
    from radar.agent import AgentFailed

    def boom(**_kwargs):
        raise AgentFailed("the model never produced a valid answer")

    monkeypatch.setattr("radar.run.execute", boom)

    exit_code = main([])

    assert exit_code == 1
    assert "never produced a valid answer" in capsys.readouterr().err


def test_main_reports_a_dead_provider_chain_without_a_traceback(monkeypatch, capsys):
    from llm.client import AllProvidersFailed

    def boom(**_kwargs):
        raise AllProvidersFailed("gemini: 401; ollama: connection refused")

    monkeypatch.setattr("radar.run.execute", boom)

    exit_code = main([])

    assert exit_code == 1
    assert "connection refused" in capsys.readouterr().err
