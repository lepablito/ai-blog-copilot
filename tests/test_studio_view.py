"""The Studio tab driven through Streamlit's own test harness.

These are here for one reason: the drafting code was correct and the tab was
still broken. A section was generated, the tokens were logged, and the text
area came back empty, because a keyed widget takes its content from session
state and ignores `value=` once that key exists. No test of `draft_section`
could have caught that — only something that runs Streamlit's widget lifecycle.
"""

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

PROBE = "tests/apps/studio_probe.py"

TOPIC = {
    "title": "Speculative decoding in production",
    "summary": "Draft models cut latency without changing outputs.",
    "why_now": "A paper landed today.",
    "angle": "practical",
    "sources": ["https://arxiv.org/abs/2607.18476"],
    "suggested_outline": ["What it is", "What it costs"],
}


def app_with_outline(**state) -> AppTest:
    app = AppTest.from_file(PROBE, default_timeout=30)
    app.session_state["selected_topic"] = TOPIC
    app.session_state["outline"] = ["What it is", "What it costs"]
    for key, value in state.items():
        app.session_state[key] = value
    return app.run()


def test_without_a_topic_the_tab_explains_where_to_get_one():
    app = AppTest.from_file(PROBE, default_timeout=30).run()

    assert "Radar tab" in app.info[0].value


def test_a_drafted_section_actually_lands_in_its_text_area():
    """The regression. It generated, it logged, it displayed nothing."""
    app = app_with_outline(fake_reply="Speculative decoding runs a small model first.")

    app.button(key="draft-0").click().run()

    assert app.text_area(key="text-0-What it is").value == (
        "Speculative decoding runs a small model first."
    )


def test_drafting_one_section_leaves_the_others_alone():
    app = app_with_outline()

    app.button(key="draft-0").click().run()

    assert app.text_area(key="text-1-What it costs").value == ""


def test_editing_a_section_by_hand_survives_a_rerun():
    """Everything the model writes is meant to be editable. An edit that a
    button press elsewhere silently reverted would make the tab useless."""
    app = app_with_outline()

    app.text_area(key="text-0-What it is").set_value("My own words.").run()
    app.button(key="draft-1").click().run()

    assert app.text_area(key="text-0-What it is").value == "My own words."


def test_the_export_stage_appears_once_there_is_a_draft():
    app = app_with_outline()

    app.button(key="draft-0").click().run()

    assert any("3. Export" in md.value for md in app.markdown)


def test_nothing_to_export_before_a_section_is_written():
    app = app_with_outline()

    assert any("Draft a section before exporting" in c.value for c in app.caption)
