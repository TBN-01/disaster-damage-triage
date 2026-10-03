from pathlib import Path

import pytest


def test_portable_demo_opens_without_xbd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    streamlit = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("TRIAGE_RESULTS_DIR", str(tmp_path / "no-xbd-results"))
    monkeypatch.setenv("TRIAGE_REVIEW_DB", str(tmp_path / "reviews.sqlite3"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "demo.py")).run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "The result", "Across disasters", "Review desk", "Explore the queue", "Error gallery", "How it works",
    ]
