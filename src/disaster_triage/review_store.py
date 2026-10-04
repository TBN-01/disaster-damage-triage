"""Local-only persistence for blind human review decisions."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CHOICES = {"Severe", "Not severe", "Unsure"}


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            dataset_key TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (dataset_key, sample_id)
        )
    """)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(reviews)")}
    if "note" not in columns:
        connection.execute("ALTER TABLE reviews ADD COLUMN note TEXT NOT NULL DEFAULT ''")
    return connection


def save_review(path: Path, dataset_key: str, sample_id: str, decision: str, note: str = "") -> None:
    if decision not in CHOICES:
        raise ValueError(f"Invalid review decision: {decision}")
    with connect(path) as connection:
        connection.execute(
            "INSERT INTO reviews (dataset_key, sample_id, decision, reviewed_at, note) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(dataset_key, sample_id) DO UPDATE SET decision=excluded.decision, "
            "reviewed_at=excluded.reviewed_at, note=excluded.note",
            (dataset_key, sample_id, decision, datetime.now(timezone.utc).isoformat(), note.strip()[:500]),
        )


def load_reviews(path: Path, dataset_key: str) -> pd.DataFrame:
    with connect(path) as connection:
        return pd.read_sql_query(
            "SELECT sample_id, decision, note, reviewed_at FROM reviews WHERE dataset_key = ? ORDER BY reviewed_at",
            connection, params=(dataset_key,),
        )


def clear_reviews(path: Path, dataset_key: str) -> int:
    """Remove answers for one image set without touching other saved reviews."""
    with connect(path) as connection:
        result = connection.execute("DELETE FROM reviews WHERE dataset_key = ?", (dataset_key,))
    return result.rowcount
