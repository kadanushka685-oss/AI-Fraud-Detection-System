"""Smoke tests for the Streamlit app, run headlessly with Streamlit's AppTest helper.

These tests are skipped automatically if streamlit or plotly is not installed.
"""
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
PAGES = [
    "Dashboard",
    "Fraud Prediction",
    "Batch CSV Analysis",
    "Analytics",
    "Transaction History",
    "About Project",
]


def _new_app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP_PATH), default_timeout=120)


def test_app_starts_without_exceptions():
    at = _new_app().run()
    assert not at.exception


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions(page):
    at = _new_app().run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
