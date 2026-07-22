"""Entry point: `uv run streamlit run src/studio/app.py`.

Three tabs, one job each. This module wires them together and owns nothing
else: every decision worth testing lives in a plain module beside it.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Streamlit runs this file as a script, not as a package module, so `src` is
# not on the path the way it is under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.calls_log import CallLog  # noqa: E402
from llm.client import LLMClient  # noqa: E402
from llm.config import build_chain  # noqa: E402
from studio.views import radar as radar_view  # noqa: E402
from studio.views import studio as studio_view  # noqa: E402

DB_PATH = Path("radar.db")


def client_factory() -> LLMClient:
    """A client per call, sharing the radar's call log.

    Built fresh each time rather than cached in session state: the chain is
    read from the environment, and a key added to `.env` mid-session should
    take effect on the next click rather than on the next restart. Costs from
    the Studio land in the same table as the agent's, which is the point of
    having a costs tab at all.
    """
    return LLMClient(build_chain(os.environ), recorder=CallLog(DB_PATH).record)


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="AI Blog Copilot", page_icon="📡", layout="wide")
    st.title("AI Blog Copilot")

    radar_tab, studio_tab, costs_tab = st.tabs(["📡 Radar", "✍️ Studio", "💸 Costs"])

    with radar_tab:
        radar_view.render(DB_PATH)
    with studio_tab:
        studio_view.render(client_factory)
    with costs_tab:
        st.info("Coming next: what the fallback chain has cost.")


main()
