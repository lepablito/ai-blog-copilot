"""Entry point: `uv run streamlit run src/studio/app.py`.

Three tabs, one job each. This module wires them together and owns nothing
else: every decision worth testing lives in a plain module beside it.
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file as a script, not as a package module, so `src` is
# not on the path the way it is under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from studio.views import radar as radar_view  # noqa: E402

DB_PATH = Path("radar.db")


def main() -> None:
    st.set_page_config(page_title="AI Blog Copilot", page_icon="📡", layout="wide")
    st.title("AI Blog Copilot")

    radar_tab, studio_tab, costs_tab = st.tabs(["📡 Radar", "✍️ Studio", "💸 Costs"])

    with radar_tab:
        radar_view.render(DB_PATH)
    with studio_tab:
        st.info("Coming next: turn a radar topic into a draft.")
    with costs_tab:
        st.info("Coming next: what the fallback chain has cost.")


main()
