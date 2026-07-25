"""The Radar tab: what the agent found, newest first.

Read-only over radar.db. The agent writes that file; nothing here does.

Everything on screen came from a model that read untrusted pages, so it is
rendered as text: `st.markdown` escapes HTML unless asked not to, and it is
never asked. Source URLs are safe to link because the topic schema rejects
anything that is not http(s) before it is ever stored.
"""

from pathlib import Path

import streamlit as st

from radar.schema import InvalidTopics
from radar.store import Store
from studio import repo_ops
from studio.radar_data import PRESETS, all_links, group_by_date, load_topics, since_for

ANGLE_LABELS = {"All angles": None, "Theoretical": "theoretical", "Practical": "practical"}

# What the workflow commits. radar.db is gitignored and on a runner lives only
# as an Actions cache, so this file is the whole route from a nightly run to
# the database behind this tab.
TOPICS_JSON = Path("data") / "topics.json"

# Holds the (date, title) of the one topic awaiting a dismissal confirmation.
PENDING_DISMISSAL = "pending_dismissal"


def render(db_path: Path | str) -> None:
    st.subheader("Topics the radar found")

    # Drawn before the query below, which is what lets an import show up in the
    # same pass: Streamlit reruns the script on a click, the rows land here,
    # and `load_topics` a few lines down already sees them. Calling `st.rerun`
    # instead would repaint correctly but wipe the message saying what happened.
    _controls(db_path)

    window, angle = _filters()
    records = load_topics(db_path, angle=ANGLE_LABELS[angle], since=window)

    if not records:
        st.info(
            "Nothing here yet. Run `uv run python -m radar.run --hours 24`, "
            "or launch a run above and fetch it when it finishes."
        )
        return

    st.caption(f"{len(records)} topic(s)")
    for day, topics in group_by_date(records):
        st.markdown(f"### {day}")
        for index, topic in enumerate(topics):
            _topic_card(topic, db_path=db_path, key=f"{day}-{index}")


def _controls(db_path: Path | str) -> None:
    launch, fetch = st.columns(2)

    if launch.button("Launch a radar run", use_container_width=True):
        outcome = repo_ops.dispatch_radar()
        _report(outcome)
        if outcome.ok:
            st.markdown(f"[Follow it on GitHub]({repo_ops.ACTIONS_URL})")

    if fetch.button("Fetch the latest topics", use_container_width=True):
        _fetch(db_path)


def _fetch(db_path: Path | str) -> None:
    outcome = repo_ops.pull()
    if not outcome.ok:
        _report(outcome)
        return

    try:
        added = Store(db_path).import_json(TOPICS_JSON)
    except (ValueError, InvalidTopics) as exc:
        # A corrupt or hand-mangled export. Saying which is more use than a
        # traceback, and nothing has been written at this point.
        st.error(f"{TOPICS_JSON} could not be read: {exc}")
        return

    if added:
        st.success(f"{added} new topic(s).")
    else:
        st.info("Nothing new — the radar has not committed anything since the last fetch.")


def _report(outcome: repo_ops.Outcome) -> None:
    if outcome.ok:
        st.success(outcome.message)
        return

    st.error(outcome.message)
    if outcome.detail:
        with st.expander("What it said"):
            st.code(outcome.detail)


def _filters() -> tuple[str | None, str]:
    left, right = st.columns(2)
    preset = left.selectbox("Window", list(PRESETS), index=len(PRESETS) - 1)
    angle = right.selectbox("Angle", list(ANGLE_LABELS))
    return since_for(preset), angle


def _topic_card(topic: dict, *, db_path: Path | str, key: str) -> None:
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

        write, dismiss = st.columns(2)

        if write.button("Write this one", key=f"write-{key}"):
            st.session_state["selected_topic"] = topic
            st.success("Loaded into the Studio tab.")

        _dismiss_control(dismiss, topic, db_path=db_path, key=key)


def _dismiss_control(column, topic: dict, *, db_path: Path | str, key: str) -> None:
    """Dismiss, behind a confirmation. A dismissal cannot be undone from here.

    Which card is awaiting confirmation is held in one slot naming the topic,
    rather than a flag per card. Card keys carry the row's position in the day,
    and positions shift as soon as something is dismissed — a per-card flag
    would be left pointing at whichever topic moved into that slot. It also
    means opening a second confirmation quietly closes the first, which is what
    someone changing their mind about which topic to drop would expect.
    """
    identity = (topic["date"], topic["title"])

    if st.session_state.get(PENDING_DISMISSAL) != identity:
        if column.button("Dismiss", key=f"dismiss-{key}"):
            st.session_state[PENDING_DISMISSAL] = identity
            # The Dismiss button has already been drawn this pass, so without a
            # rerun the question would not appear until the next click.
            st.rerun()
        return

    column.caption("Dismiss this topic?")
    yes, no = column.columns(2)

    if yes.button("Yes", key=f"yes-{key}", type="primary"):
        Store(db_path).dismiss(date=topic["date"], title=topic["title"])
        st.session_state.pop(PENDING_DISMISSAL, None)
        # Needed here, unlike the fetch button above: the cards are drawn after
        # `load_topics` has already read the database, so this pass still holds
        # the dismissed topic and would leave the card on screen.
        st.rerun()

    if no.button("No", key=f"no-{key}"):
        st.session_state.pop(PENDING_DISMISSAL, None)
        st.rerun()
