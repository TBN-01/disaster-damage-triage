import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from disaster_triage.inspect_data import inspect
from disaster_triage.prepare import polygon_box, prepare
from disaster_triage.train import evaluate, train
from disaster_triage.ui_data import case_filter, ranked_queue
from disaster_triage.cross_validate import budget_curves, cross_validate, tile_bootstrap_interval, top_review
from disaster_triage.review_store import load_reviews, save_review
from disaster_triage.sample_demo import make_sample_demo


def make_tile(root: Path, event: str, number: int) -> None:
    images = root / "train" / "images"
    labels = root / "train" / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    stem = f"{event}_{number:08d}"
    pre = Image.new("RGB", (128, 128), "gray")
    post = pre.copy()
    draw = ImageDraw.Draw(post)
    # Right building changes on every tile; left building is unchanged.
    draw.rectangle((70, 25, 100, 55), fill=(20 + number * 10, 15, 15))
    pre.save(images / f"{stem}_pre_disaster.png")
    post.save(images / f"{stem}_post_disaster.png")
    features = [
        {"wkt": "POLYGON ((20 25, 50 25, 50 55, 20 55, 20 25))", "properties": {"subtype": "no-damage"}},
        {"wkt": "POLYGON ((70 25, 100 25, 100 55, 70 55, 70 25))", "properties": {"subtype": "major-damage"}},
    ]
    (labels / f"{stem}_post_disaster.json").write_text(json.dumps({"features": {"xy": features}}), encoding="utf-8")


def test_polygon_box() -> None:
    assert polygon_box("POLYGON ((1 2, 3 2, 3 4, 1 4, 1 2))") == (1.0, 2.0, 3.0, 4.0)


def test_prepare_and_event_holdout(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for event in ("event-a", "event-b", "event-c"):
        for tile in range(4):
            make_tile(raw, event, tile)
    summary = inspect(raw)
    assert summary["event-c"]["severe"] == 4
    frame = prepare(raw, tmp_path / "processed")
    assert len(frame) == 24
    assert set(frame["severe"]) == {0, 1}
    result = train(tmp_path / "processed" / "manifest.csv", "event-c", tmp_path / "results")
    assert "event-c" not in result["train_events"]
    assert result["model"]["samples"] == 8
    assert result["model"]["severe_buildings"] == 4
    assert (tmp_path / "results" / "predictions.csv").exists()


def test_review_metric() -> None:
    y = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    result = evaluate(y, scores)
    # Top 20% means reviewing one of five buildings, capturing one of two damaged ones.
    assert result["recall_at_20_percent_reviewed"] == 0.5


def test_queue_budget_and_error_filters() -> None:
    frame = pd.DataFrame({
        "sample_id": ["a", "b", "c", "d", "e"],
        "baseline_score": [0.1, 0.9, 0.8, 0.2, 0.3],
        "severe": [1, 1, 0, 1, 0],
    })
    ranked, summary = ranked_queue(frame, "baseline_score", 0.2)
    assert summary == {
        "review_count": 1, "total_count": 5, "severe_found": 1,
        "severe_total": 3, "recall": 1 / 3, "precision": 1.0,
    }
    assert ranked.iloc[0]["sample_id"] == "b"
    assert case_filter(ranked, "Severe cases missed")["sample_id"].tolist() == ["d", "a"]
    assert case_filter(ranked, "False alarms").empty


def test_queue_rejects_invalid_budget() -> None:
    frame = pd.DataFrame({"severe": [1], "score": [0.5]})
    for budget in (0, -0.1, 1.1):
        try:
            ranked_queue(frame, "score", budget)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid budget {budget} to fail")


def test_cross_event_evaluation_uses_whole_events(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for event in ("event-a", "event-b", "event-c"):
        for tile in range(2):
            make_tile(raw, event, tile)
    frame = prepare(raw, tmp_path / "processed")
    result = cross_validate(tmp_path / "processed" / "manifest.csv", tmp_path / "cross", bootstrap_repetitions=10)
    assert len(result) == 6
    assert set(result["event"]) == set(frame["event"])
    assert all(result["buildings"] == 4)
    assert all(result["tiles"] == 2)
    assert (tmp_path / "cross" / "fold_predictions.csv").exists()
    assert (tmp_path / "cross" / "review_budget_curves.csv").exists()
    assert (tmp_path / "cross" / "summary.json").exists()


def test_budget_curves_match_top_review() -> None:
    predictions = pd.DataFrame({
        "event": ["one"] * 5, "method": ["Simple image change"] * 5,
        "severe": [1, 0, 1, 0, 0], "score": [0.9, 0.8, 0.1, 0.2, 0.3],
    })
    curves = budget_curves(predictions)
    at_twenty = curves[curves["budget_percent"] == 20].iloc[0]
    assert at_twenty["reviewed"] == 1
    assert at_twenty["severe_found"] == 1
    assert at_twenty["recall"] == 0.5
    assert at_twenty["precision"] == 1.0


def test_top_review_and_tile_interval() -> None:
    y = np.array([1, 0, 1, 0, 0])
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.3])
    assert top_review(y, score) == (1, 2, 1)
    low, high = tile_bootstrap_interval(y, score, np.array(["a", "a", "b", "b", "b"]), repetitions=20)
    assert 0 <= low <= high <= 1


def test_synthetic_demo_and_review_persistence(tmp_path: Path) -> None:
    frame = make_sample_demo(tmp_path / "sample")
    assert len(frame) == 24
    assert all(Path(path).exists() for path in frame["post_crop"])
    assert len(make_sample_demo(tmp_path / "sample")) == 24
    database = tmp_path / "reviews.sqlite3"
    save_review(database, "synthetic-demo-v1", "DEMO-001", "Unsure")
    save_review(database, "synthetic-demo-v1", "DEMO-001", "Severe", "  I saw a broken roof. ")
    assert load_reviews(database, "synthetic-demo-v1")["decision"].tolist() == ["Severe"]
    assert load_reviews(database, "synthetic-demo-v1")["note"].tolist() == ["I saw a broken roof."]
    assert load_reviews(database, "other").empty


def test_existing_review_database_gets_note_column(tmp_path: Path) -> None:
    database = tmp_path / "older_reviews.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE reviews (dataset_key TEXT NOT NULL, sample_id TEXT NOT NULL, "
            "decision TEXT NOT NULL, reviewed_at TEXT NOT NULL, PRIMARY KEY (dataset_key, sample_id))"
        )
        connection.execute("INSERT INTO reviews VALUES ('demo', 'one', 'Unsure', 'old-date')")
    assert load_reviews(database, "demo")["note"].tolist() == [""]
    save_review(database, "demo", "one", "Severe", "New note")
    assert load_reviews(database, "demo")["note"].tolist() == ["New note"]

