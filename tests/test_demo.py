from pathlib import Path

import pytest


def test_portable_demo_opens_without_xbd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("TRIAGE_RESULTS_DIR", str(tmp_path / "no-xbd-results"))
    monkeypatch.setenv("TRIAGE_REVIEW_DB", str(tmp_path / "reviews.sqlite3"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "demo.py")).run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "The story", "Try a review", "Explore the results",
    ]
    assert any(button.label == "Save my answer and reveal the label" for button in app.button)

    answer = next(radio for radio in app.radio if radio.label == "From these images, what do you think?")
    answer.set_value("Looks badly damaged")
    next(button for button in app.button if button.label == "Save my answer and reveal the label").click()
    app.run(timeout=30)
    assert not app.exception
    assert any(button.label == "Review the next building" for button in app.button)

    next(button for button in app.button if button.label == "Review the next building").click().run(timeout=30)
    assert not app.exception
    assert any(button.label == "Save my answer and reveal the label" for button in app.button)
