import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from disaster_triage.inspect_data import inspect
from disaster_triage.prepare import polygon_box, prepare
from disaster_triage.train import evaluate, train


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

