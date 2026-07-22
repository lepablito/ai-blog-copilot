"""A one-tab Streamlit script for AppTest to drive.

Not the real app: it renders only the Studio view and hands it a client backed
by a scripted fake, so the test exercises Streamlit's actual widget and
session-state behaviour without touching a model. That behaviour is the whole
point — the bug this guards against was Streamlit's, not the drafting code's.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.fakes import FakeProvider  # noqa: E402

from llm.client import LLMClient  # noqa: E402
from studio.views import studio as studio_view  # noqa: E402

REPLY = st.session_state.get("fake_reply", "Drafted prose for the section.")

studio_view.render(lambda: LLMClient([FakeProvider("fake", [REPLY])], sleep=lambda _: None))
