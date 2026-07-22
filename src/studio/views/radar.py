"""The Radar tab: what the agent found, newest first.

Read-only over radar.db. The agent writes that file; nothing here does.

Everything on screen came from a model that read untrusted pages, so it is
rendered as text: `st.markdown` escapes HTML unless asked not to, and it is
never asked. Source URLs are safe to link because the topic schema rejects
anything that is not http(s) before it is ever stored.
"""

from pathlib import Path

import streamlit as st

from studio.radar_data import PRESETS, all_links, group_by_date, load_topics, since_for

ANGLE_LABELS = {"All angles": None, "Theoretical": "theoretical", "Practical": "practical"}


def render(db_path: Path | str) -> None:
    st.subheader("Topics the radar found")

    window, angle = _filters()
    records = load_topics(db_path, angle=ANGLE_LABELS[angle], since=window)

    if not records:
        st.info(
            "Nothing here yet. Run `uv run python -m radar.run --hours 24` "
            "or dispatch the Daily radar workflow, then reload."
        )
        return

    st.caption(f"{len(records)} topic(s)")
    for day, topics in group_by_date(records):
        st.markdown(f"### {day}")
        for index, topic in enumerate(topics):
            _topic_card(topic, key=f"{day}-{index}")


def _filters() -> tuple[str | None, str]:
    left, right = st.columns(2)
    preset = left.selectbox("Window", list(PRESETS), index=len(PRESETS) - 1)
    angle = right.selectbox("Angle", list(ANGLE_LABELS))
    return since_for(preset), angle


def _topic_card(topic: dict, *, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{topic['title']}**")
        st.caption(f"{topic['angle']} · effort: {topic['estimated_effort']}")
        st.markdown(topic["summary"])

        st.markdown(f"**Why now:** {topic['why_now']}")

        with st.expander("Outline and sources"):
            for bullet in topic["suggested_outline"]:
                st.markdown(f"- {bullet}")
            st.markdown("**Sources**")
            for url in all_links(topic):
                st.markdown(f"- {url}")

        if st.button("Write this one", key=f"write-{key}"):
            st.session_state["selected_topic"] = topic
            st.success("Loaded into the Studio tab.")
