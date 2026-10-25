"""Smoke tests: every dashboard page loads, and insight cards render as HTML.  Run:  pytest"""
import os

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "dashboard.py")
PAGES = ["summary", "daily", "weekly", "monthly", "insights", "prediction"]


def run_page(page):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["page"] = page
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_loads_without_errors(page):
    at = run_page(page)
    assert not at.exception
    assert not at.error


def test_insight_cards_render_as_html_not_code():
    # A blank line or indentation inside the HTML makes Markdown show it as a code block.
    at = run_page("insights")
    cards = [m.value for m in at.markdown if "border-radius:10px" in m.value]
    assert cards, "no insight cards found"
    for card in cards:
        assert "\n" not in card
