from pathlib import Path
import json

import pytest

from disaster_triage.sample_demo import make_sample_demo
from disaster_triage.train import evaluate


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

    event = next(item for item in app.selectbox if item.label == "Pick a disaster")
    event.set_value("mexico-earthquake").run(timeout=30)
    assert not app.exception


def test_demo_with_local_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    predictions = make_sample_demo(tmp_path / "crops")
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    y = predictions["severe"].to_numpy()
    metrics = {
        "test_event": "local-test",
        "baseline": evaluate(y, predictions["baseline_score"].to_numpy()),
        "model": evaluate(y, predictions["model_score"].to_numpy()),
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setenv("TRIAGE_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("TRIAGE_REVIEW_DB", str(tmp_path / "reviews.sqlite3"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "demo.py")).run(timeout=30)
    assert not app.exception
    assert any(item.label == "Human review budget" for item in app.slider)
