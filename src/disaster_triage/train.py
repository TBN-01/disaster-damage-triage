"""Train and evaluate building-damage triage on a held-out disaster."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

from .features import image_features


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    prediction = (score >= threshold).astype(int)
    n_review = max(1, math.ceil(len(y) * 0.2))
    top = np.argsort(-score)[:n_review]
    positives = int(y.sum())
    return {
        "samples": int(len(y)),
        "severe_buildings": positives,
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(y, score)) if positives else None,
        "recall_at_20_percent_reviewed": float(y[top].sum() / positives) if positives else None,
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def train(manifest_path: Path, test_event: str, out_dir: Path, event_prefix: str | None = None) -> dict:
    frame = pd.read_csv(manifest_path)
    required = {"sample_id", "event", "severe", "pre_crop", "post_crop"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if event_prefix:
        frame = frame[frame["event"].str.startswith(event_prefix)].copy()
    if test_event not in set(frame["event"]):
        raise ValueError(f"Unknown test event {test_event!r}; options: {sorted(frame['event'].unique())}")

    vectors = [image_features(row.pre_crop, row.post_crop) for row in frame.itertuples(index=False)]
    x = np.stack([v[0] for v in vectors])
    difference = np.array([v[1] for v in vectors])
    train_mask = (frame["event"] != test_event).to_numpy()
    test_mask = ~train_mask
    y = frame["severe"].to_numpy(dtype=int)
    if len(set(y[train_mask])) < 2 or len(set(y[test_mask])) < 2:
        raise ValueError("Both train and held-out event need severe and non-severe examples")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    model.fit(x[train_mask], y[train_mask])
    model_score = model.predict_proba(x[test_mask])[:, 1]
    baseline_score = difference[test_mask]
    y_test = y[test_mask]

    results = {
        "test_event": test_event,
        "train_events": sorted(frame.loc[train_mask, "event"].unique().tolist()),
        "train_samples": int(train_mask.sum()),
        "train_severe": int(y[train_mask].sum()),
        "data_source": sorted(frame["source"].dropna().unique().tolist()) if "source" in frame else ["official-xbd-polygons"],
        "baseline": evaluate(y_test, baseline_score, threshold=float(np.median(difference[train_mask]))),
        "model": evaluate(y_test, model_score),
        "baseline_definition": "Mean absolute pixel difference; threshold is training-set median.",
        "model_definition": "Random forest on summary and coarse spatial image-change features.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = frame.loc[test_mask].copy()
    predictions["model_score"] = model_score
    predictions["baseline_score"] = baseline_score
    predictions.sort_values("model_score", ascending=False).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=Path("data/processed/manifest.csv"), type=Path)
    parser.add_argument("--test-event", required=True)
    parser.add_argument("--out", default=Path("results"), type=Path)
    parser.add_argument("--event-prefix", help="Restrict training and test data to events with this name prefix")
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.test_event, args.out, args.event_prefix), indent=2))


if __name__ == "__main__":
    main()

