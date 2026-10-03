"""Leave one whole disaster out at a time; never train on test-event buildings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .features import image_features
from .train import make_model


def top_review(y: np.ndarray, scores: np.ndarray, fraction: float = 0.2) -> tuple[int, int, int]:
    """Return severe found, severe total, and review count; stable ties by row order."""
    count = max(1, math.ceil(len(y) * fraction))
    top = np.argsort(-scores, kind="stable")[:count]
    return int(y[top].sum()), int(y.sum()), count


def tile_bootstrap_interval(
    y: np.ndarray, scores: np.ndarray, tiles: np.ndarray,
    repetitions: int = 500, seed: int = 42,
) -> tuple[float | None, float | None]:
    """Resample tiles, not neighboring buildings, for a descriptive 95% interval."""
    unique, tile_index = np.unique(tiles, return_inverse=True)
    groups = [np.flatnonzero(tile_index == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed)
    recalls = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[i] for i in chosen])
        found, positives, _ = top_review(y[rows], scores[rows])
        if positives:
            recalls.append(found / positives)
    if not recalls:
        return None, None
    low, high = np.quantile(recalls, [0.025, 0.975])
    return float(low), float(high)


def cross_validate(
    manifest_path: Path, out_dir: Path, bootstrap_repetitions: int = 500,
    feature_cache: Path | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    required = {"sample_id", "event", "tile", "severe", "pre_crop", "post_crop"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame["event"].nunique() < 2:
        raise ValueError("At least two events are required")

    ids = frame["sample_id"].astype(str).to_numpy()
    cached = None
    if feature_cache and feature_cache.exists():
        with np.load(feature_cache) as archive:
            if np.array_equal(archive["sample_id"], ids):
                cached = archive["features"], archive["difference"]
    if cached is None:
        vectors = [image_features(row.pre_crop, row.post_crop) for row in frame.itertuples(index=False)]
        x = np.stack([item[0] for item in vectors])
        difference = np.array([item[1] for item in vectors])
        if feature_cache:
            feature_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(feature_cache, sample_id=ids, features=x, difference=difference)
    else:
        x, difference = cached

    y = frame["severe"].to_numpy(dtype=int)
    events = frame["event"].astype(str).to_numpy()
    tiles = frame["tile"].astype(str).to_numpy()
    rows = []
    predictions = []
    for fold, event in enumerate(sorted(np.unique(events))):
        test = events == event
        train = ~test
        if len(np.unique(y[train])) < 2:
            raise ValueError(f"Training events for {event} have only one class")
        model = make_model()
        model.fit(x[train], y[train])
        scores = {
            "Simple image change": difference[test],
            "Random forest": model.predict_proba(x[test])[:, 1],
        }
        for method, score in scores.items():
            found, positives, count = top_review(y[test], score)
            low, high = tile_bootstrap_interval(
                y[test], score, tiles[test], repetitions=bootstrap_repetitions, seed=42 + fold,
            )
            rows.append({
                "event": event, "method": method, "buildings": int(test.sum()),
                "tiles": int(len(np.unique(tiles[test]))), "severe": positives,
                "severe_prevalence": positives / int(test.sum()),
                "reviewed": count, "severe_found": found,
                "recall_at_20_percent_reviewed": found / positives if positives else np.nan,
                "recall_ci_low": low, "recall_ci_high": high,
                "average_precision": float(average_precision_score(y[test], score)) if positives else np.nan,
            })
            predictions.append(pd.DataFrame({
                "sample_id": ids[test], "event": event, "tile": tiles[test],
                "severe": y[test], "method": method, "score": score,
            }))
        print(f"Finished held-out event: {event}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "event_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out_dir / "fold_predictions.csv", index=False)
    summary = {
        "design": "Leave one disaster event out; 20% review budget; 200-tree random forest fitted on other events only.",
        "sample_buildings": int(len(frame)),
        "events": int(len(np.unique(events))),
        "bootstrap": "95% descriptive percentile interval from resampling image tiles within each held-out event; not field-performance uncertainty.",
        "bootstrap_repetitions": bootstrap_repetitions,
        "methods": {},
    }
    for method, group in result.groupby("method"):
        valid = group["recall_at_20_percent_reviewed"].dropna()
        summary["methods"][method] = {
            "mean_event_recall": float(valid.mean()),
            "median_event_recall": float(valid.median()),
            "min_event_recall": float(valid.min()),
            "max_event_recall": float(valid.max()),
            "events_with_severe_cases": int(len(valid)),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", default=Path("results/cross-event"), type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--feature-cache", type=Path, help="Optional local cache; IDs must match the manifest")
    args = parser.parse_args()
    result = cross_validate(args.manifest, args.out, args.bootstrap_repetitions, args.feature_cache)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
