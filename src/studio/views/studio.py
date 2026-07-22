"""The Studio tab: a radar topic becomes a `.md` file, with Pablo driving.

Three stages down the page — outline, sections, export — because that is the
order the work happens in. Nothing runs on its own: every model call is behind
a button, and everything it returns lands in a text area that can be edited or
thrown away.

The draft lives in `st.session_state`. Streamlit re-runs this whole script on
every interaction, so anything not kept there is lost the moment a button is
pressed.
"""

import os
from functools import partial
from pathlib import Path

import streamlit as st

from llm.client import AllProvidersFailed
from studio.drafting import draft_section, generate_outline, revise
from studio.export import (
    assemble,
    known_projects,
    list_posts,
    render_post,
    split_front_matter,
    write_post,
)

OUTPUT_DIR = Path("output/posts")


def render(client_factory) -> None:
    st.subheader("Turn a topic into a draft")

    topic = st.session_state.get("selected_topic")
    if topic:
        st.markdown(f"**{topic['title']}**")
        st.caption(topic["why_now"])

        _outline_stage(client_factory, topic)
        if st.session_state.get("outline"):
            _drafting_stage(client_factory, topic)
            _export_stage(topic)
    else:
        st.info("Pick a topic in the Radar tab first — its 'Write this one' button lands here.")

    # Independent of any draft in progress: what has already been written out.
    _exported_posts()


def _outline_stage(client_factory, topic: dict) -> None:
    st.markdown("### 1. Outline")

    if st.button("Propose an outline"):
        with st.spinner("Thinking about structure…"):
            headings = _guarded(partial(generate_outline, client_factory(), topic))
        if headings:
            st.session_state["outline"] = headings

    # The radar already suggested one. Offering it saves a call when it is good
    # enough, which it often is.
    has_suggestion = not st.session_state.get("outline") and topic.get("suggested_outline")
    if has_suggestion and st.button("Use the radar's outline"):
        st.session_state["outline"] = list(topic["suggested_outline"])

    if st.session_state.get("outline"):
        edited = st.text_area(
            "One heading per line",
            value="\n".join(st.session_state["outline"]),
            height=160,
        )
        st.session_state["outline"] = [line.strip() for line in edited.splitlines() if line.strip()]


def _drafting_stage(client_factory, topic: dict) -> None:
    st.markdown("### 2. Sections")
    sections: dict[str, str] = st.session_state.setdefault("sections", {})

    outline = st.session_state["outline"]

    for index, heading in enumerate(outline):
        # Keyed on the heading, not just the position: after a revision the
        # outline collapses to a single "Draft", and a key of "text-0" would
        # hand the new box the old section's leftover state.
        box = f"text-{index}-{heading}"
        st.session_state.setdefault(box, sections.get(heading, ""))

        with st.expander(heading, expanded=not sections.get(heading)):
            if st.button("Draft this section", key=f"draft-{index}"):
                # Earlier sections only: a section should not be written around
                # text that is going to follow it.
                so_far = assemble([(h, sections.get(h, "")) for h in outline[:index]])
                with st.spinner(f"Writing “{heading}”…"):
                    # partial, not a lambda: a closure defined in a loop binds
                    # the loop variable by reference, and the linter is right
                    # to object even though this one is called immediately.
                    text = _guarded(
                        partial(
                            draft_section,
                            client_factory(),
                            topic=topic,
                            heading=heading,
                            outline=outline,
                            so_far=so_far,
                        )
                    )
                if text:
                    # Written into the widget's own state and rerun, not passed
                    # as `value=`. A keyed widget takes its content from session
                    # state and ignores `value` once that key exists, so the
                    # obvious version drafted the section, logged the tokens,
                    # and displayed an empty box.
                    st.session_state[box] = text
                    sections[heading] = text
                    st.rerun()

            sections[heading] = st.text_area("Text", height=220, key=box)

    _revision_box(client_factory)


def _revision_box(client_factory) -> None:
    instruction = st.text_input(
        "Revise the whole draft",
        placeholder="make it shorter · add a code example · more concrete",
    )
    if st.button("Apply", disabled=not instruction.strip()):
        sections = st.session_state["sections"]
        headings = st.session_state["outline"]
        draft = assemble([(h, sections.get(h, "")) for h in headings])

        with st.spinner("Revising…"):
            revised = _guarded(
                lambda: revise(client_factory(), draft=draft, instruction=instruction)
            )

        if revised:
            # A revision rewrites the whole document, so section boundaries no
            # longer line up. Collapsing to one editable block is honest about
            # that; pretending to re-split it would silently drop text.
            st.session_state["outline"] = ["Draft"]
            st.session_state["sections"] = {"Draft": revised}
            st.rerun()


def _export_stage(topic: dict) -> None:
    st.markdown("### 3. Export")
    sections = st.session_state.get("sections", {})
    body = assemble([(h, sections.get(h, "")) for h in st.session_state["outline"]])

    if not body.strip():
        st.caption("Draft a section before exporting.")
        return

    title = st.text_input("Title", value=topic["title"])
    description = st.text_area(
        "Description (shown in listings and link previews)", value=topic["summary"], height=80
    )
    tags = st.text_input("Tags, comma separated", value=topic["angle"])

    portfolio = os.getenv("PORTFOLIO_PATH")
    projects = known_projects(portfolio)
    related = st.selectbox("Related project", ["(none)", *projects]) if projects else "(none)"
    if not projects:
        st.caption(
            "Set PORTFOLIO_PATH to link a case study. Without it the field is "
            "left out, since an unverifiable slug breaks the blog's build."
        )

    post = render_post(
        title=title,
        description=description,
        tags=[t for t in tags.split(",")],
        body=body,
        related_project=None if related == "(none)" else related,
        portfolio_path=portfolio,
    )
    with st.expander("Preview the file"):
        st.code(post, language="markdown")

    overwrite = st.checkbox("Replace the file if it already exists")
    if st.button("Write to output/posts/", type="primary"):
        try:
            path = write_post(
                OUTPUT_DIR,
                overwrite=overwrite,
                title=title,
                description=description,
                tags=[t for t in tags.split(",")],
                body=body,
                related_project=None if related == "(none)" else related,
                portfolio_path=portfolio,
            )
        except FileExistsError as exc:
            st.error(f"{exc}")
            return

        st.success(f"Wrote {path}")
        st.caption(
            "Exported as a draft. Copying it into the portfolio and committing "
            "it is a step you take yourself."
        )


def _exported_posts() -> None:
    st.divider()
    st.markdown("### Exported posts")

    posts = list_posts(OUTPUT_DIR)
    if not posts:
        st.caption(f"Nothing written yet. Exported posts land in `{OUTPUT_DIR}/`.")
        return

    labels = {f"{p['title']}  ·  {p['date'] or 'no date'}": p for p in posts}
    chosen = labels[st.selectbox("Pick a post to read", list(labels))]

    raw = Path(chosen["path"]).read_text(encoding="utf-8")
    fields, body = split_front_matter(raw)

    if fields.get("description"):
        st.caption(fields["description"])

    # The body came from a model that read untrusted pages, so it is rendered
    # as Markdown with HTML left inert — `st.markdown` escapes it unless asked
    # not to, and it is not asked. Same trust boundary as the Radar tab.
    st.markdown(body)

    with st.expander("Raw file"):
        st.code(raw, language="markdown")
    st.download_button(
        "Download .md",
        data=raw,
        file_name=Path(chosen["path"]).name,
        mime="text/markdown",
    )
    st.caption(f"On disk at `{chosen['path']}`. Copying it into the blog is a step you take.")


def _guarded(action):
    """Run an LLM-backed action, turning a dead chain into a message.

    A traceback in the middle of the page tells the user nothing they can act
    on, and Streamlit keeps it there until the next rerun.
    """
    try:
        return action()
    except AllProvidersFailed as exc:
        st.error(f"No provider answered: {exc}")
    except ValueError as exc:
        st.error(str(exc))
    return None
