"""Small, testable calculations used by the research demo."""

from __future__ import annotations

import math

import pandas as pd


def ranked_queue(frame: pd.DataFrame, score_column: str, review_fraction: float) -> tuple[pd.DataFrame, dict]:
    """Rank buildings and summarize a fixed human-review budget.

    Scores are relative rankings, not calibrated damage probabilities. Ties
    retain manifest order so results are deterministic across app reruns.
    """
    if frame.empty:
        raise ValueError("The prediction table is empty")
    if score_column not in frame or "severe" not in frame:
        raise ValueError("Predictions need a score column and severe labels")
    if not 0 < review_fraction <= 1:
        raise ValueError("Review fraction must be between 0 and 1")

    ranked = frame.sort_values(score_column, ascending=False, kind="stable").reset_index(drop=True).copy()
    count = max(1, math.ceil(len(ranked) * review_fraction))
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["review_priority"] = ranked["rank"] <= count
    severe = ranked["severe"].astype(bool)
    caught = int((severe & ranked["review_priority"]).sum())
    total_severe = int(severe.sum())
    return ranked, {
        "review_count": count,
        "total_count": len(ranked),
        "severe_found": caught,
        "severe_total": total_severe,
        "recall": caught / total_severe if total_severe else None,
        "precision": caught / count,
    }


def case_filter(frame: pd.DataFrame, choice: str) -> pd.DataFrame:
    """Apply an explainable review outcome filter to an already ranked queue."""
    severe = frame["severe"].astype(bool)
    priority = frame["review_priority"].astype(bool)
    if choice == "All buildings":
        return frame
    if choice == "Priority queue":
        return frame[priority]
    if choice == "Severe cases found":
        return frame[priority & severe]
    if choice == "Severe cases missed":
        return frame[~priority & severe]
    if choice == "False alarms":
        return frame[priority & ~severe]
    raise ValueError(f"Unknown filter: {choice}")
