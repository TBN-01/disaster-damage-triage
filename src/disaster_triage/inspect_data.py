"""Summarize usable xBD labels before extracting image crops."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .prepare import NOT_SEVERE, SEVERE, event_from_tile, image_paths


def inspect(data_root: Path) -> dict:
    summary = defaultdict(lambda: Counter())
    files = sorted(data_root.rglob("*_post_disaster.json"))
    for label_file in files:
        pre, post, tile = image_paths(label_file)
        event = event_from_tile(tile)
        if not pre.exists() or not post.exists():
            summary[event]["missing_image_pairs"] += 1
            continue
        summary[event]["image_pairs"] += 1
        doc = json.loads(label_file.read_text(encoding="utf-8"))
        for feature in doc.get("features", {}).get("xy", []):
            subtype = feature.get("properties", {}).get("subtype")
            if subtype in SEVERE:
                summary[event]["severe"] += 1
            elif subtype in NOT_SEVERE:
                summary[event]["not_severe"] += 1
            else:
                summary[event]["excluded_unclassified"] += 1
    return {event: dict(counts) for event, counts in sorted(summary.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.data_root), indent=2))


if __name__ == "__main__":
    main()

